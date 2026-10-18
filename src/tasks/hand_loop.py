"""一局(场)的对局主循环, 按阶段展开.

**为什么单独一个模块**: `_play_one_game` 原来是 786 行单函数 + 40 个局部状态
变量, 协议读数(网络读牌)与视觉读数(截屏识别)在同一段代码里逐句交替, 想理解
"这一轮到底按什么顺序判断"只能从头读到尾找 `continue`. 这里把它拆成**显式
阶段序列**: 每个阶段一个方法, 返回 `Tick` 说明"本轮接下来怎么走", 顺序写在
`_PHASES` 里 —— 顺序变成数据而不是控制流, 单测可以直接调单个阶段.

**没有重写策略**: 代码是逐段搬过来的, 判据、阈值、注释里的实跑依据都保留.
这是一次结构重构, 不是重新设计 —— 那 786 行里的守卫(立直状态机、手间隙、
张数硬校验、摸切守卫、赤 5 保留、杠牌剥离)全部是实跑撞出来的, 换一套写法
等于把踩过的坑重踩一遍.

阶段顺序(改顺序 = 改行为, 动手前先想清楚依赖什么):
    0  收工检查        用户点了停止
    1  协议空闲跳帧    协议说"什么都没给我" -> 不抓帧不跑视觉
    2  抓帧
    3  页面出口        准备页 / 结算页 / 奖励页 / 协议报收局
    4  抓包存活        链路是我们起的, 挂了立刻切视觉
    5  牌桌闸门        牌桌/发牌特征没出来 -> 整轮不感知不决策
    6  一手结束过场    和了/流局顺位画面 -> 点确认, 本轮不读手牌
    7  摸牌稳定窗      协议刚发牌的那零点几秒不读手牌
    8  感知            读手牌 + 按钮, 再走协议对账
    9  手间隙          视觉报清空而协议未收局的那一两秒
    10 分析            低置信/空档/副露记账/时钟/出牌 EV
    11 决策            decide_turn
    12 闸门            发牌闸门(按手重臂) + 张数硬校验
    13 卡死            感知完全不变且没动作
    14 等待            在等别人
    15 执行            双帧确认 -> 点 -> 收尾(立直状态机/和了/宣言牌验证)
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.detection import low_confidence
from src.instrumentation import bench
from src.mahjong.table import (
    ACTION_DISCARD,
    ACTION_RIICHI,
    ACTION_WAIT,
    ACTION_WIN,
    TABLE_GAVE_UP,
    TABLE_IGNORING,
    TABLE_WAITING,
    TurnDecision,
    decide_turn,
    has_yaku_pair,
    is_my_discard_turn,
    meld_taken_kinds,
)
from src.tasks import tuning
from src.tasks.glue import (
    _draw_settle_pending,
    _ev_record_of,
    _hand_conflict_veto,
    _proto_clock_turn_of,
    _proto_meld_window_of,
    _proto_round_open_of,
    _proto_round_seq_of,
    _proto_turn_of,
    _trust_skips_confirm,
)
from src.vision import StuckDetector


class Tick(enum.Enum):
    """一个阶段跑完后, 本轮该怎么继续."""

    PROCEED = "proceed"        # 继续下一个阶段
    SKIP = "skip"              # 本轮到此为止, 睡一个轮询间隔再来
    ROUND_OVER = "over"        # 本局正常结束(成功)
    ABORT = "abort"            # 收工/超时(失败)


#: 闸门未开时不允许执行的动作. 发牌动画与摊牌动画期间, 张数可能恰好"合法",
#: 决策层会算出动作, 但游戏根本不收 —— 实跑里第一手和摊牌后都打出去过.
DEAL_BLOCKED_ACTIONS = frozenset({ACTION_DISCARD, ACTION_RIICHI})


def _conflict_why(note: str) -> str:
    """从一条对账 note 里取出"为什么对不上"(末尾括号里那段).

    note 形如 `牌=14/11 协议按钮=无 协议:轮到我打牌 (张数对不上: 协议14/视觉11)`,
    告警里只要最后那段 —— 前面的倒计时/按钮明细在这里是噪音. 拿不到就原样返回.
    """
    if "(" in note and note.rstrip().endswith(")"):
        return note.rsplit("(", 1)[-1].rstrip().rstrip(")")
    return note


@dataclass
class RoundState:
    """一轮迭代里的数据, 阶段之间靠它传递(不靠 self 上的隐式字段)."""

    cfg: dict
    game: int
    spec: Any
    vision: Any
    deadline: float
    stuck: StuckDetector
    turns: int = 0
    unknown_streak: int = 0
    # 立直状态机: 点下立直 -> 必须打宣言牌 -> 真的少了一张才清位.
    pending_riichi: bool = False
    riichi_active: bool = False
    # 本轮出牌是不是"立直选牌态里的宣言牌".
    declare_turn: bool = False
    # 本轮感知结果
    frame: Any = None
    hand: Any = None
    buttons: dict = field(default_factory=dict)
    button_conf: dict = field(default_factory=dict)
    net_exempt: Sequence[int] = ()
    net_note: dict | None = None
    net_trust: str | None = None
    # 本轮分析结果
    allowed: list | None = None
    blurry: list | None = None
    ev_rank: Any = None
    ev_shadow: dict | None = None
    # 本轮决策
    decision: Any = None


class HandLoop:
    """一次 `_play_one_game` 的主循环. 通过 `self.t` 调用任务上的能力."""

    #: 阶段序列. 顺序即行为, 见模块 docstring 的顺序表.
    _PHASES = (
        "_phase_should_stop",
        "_phase_idle_skip",
        "_phase_grab",
        "_phase_pages",
        "_phase_capture_watch",
        "_phase_table_gate",
        "_phase_hand_result",
        "_phase_draw_settle",
        "_phase_perceive",
        "_phase_hand_gap",
        "_phase_analyze",
        "_phase_deal_gate",
        "_phase_stuck",
        "_phase_wait",
        "_phase_execute",
    )

    def __init__(self, task, st: RoundState) -> None:
        self.t = task
        self.st = st

    # ------------------------------------------------------------------
    # 驱动
    # ------------------------------------------------------------------
    def run(self) -> bool:
        st = self.st
        while time.time() < st.deadline:
            # bench 挂钩(默认关闭): 记录主循环周期.
            bench.tick_loop()
            tick = self._run_phases()
            if tick is Tick.SKIP:
                self.t.sleep(tuning.TURN_POLL_INTERVAL)
                continue
            if tick is Tick.ROUND_OVER:
                return True
            if tick is Tick.ABORT:
                return False
            # PROCEED 不会出现在这里: 最后一个阶段必然给出一个出口.
        frame = self.t._grab()
        self.t._save_unknown_page(frame, f"第{st.game}局超时")
        self.t.log_error(f"第 {st.game} 局超时（{tuning.GAME_TIMEOUT}s），已截图留证")
        return False

    def _run_phases(self) -> Tick:
        for name in self._PHASES:
            tick = getattr(self, name)()
            if tick is not Tick.PROCEED:
                return tick
        return Tick.SKIP

    # ------------------------------------------------------------------
    # 0-2: 收工 / 跳帧 / 抓帧
    # ------------------------------------------------------------------
    def _phase_should_stop(self) -> Tick:
        # 框架的 paused 标志不会自动打断任务代码, 必须显式问.
        if self.t._should_yield():
            self.t.log_info("收到停止指令，中止本局")
            return Tick.ABORT
        return Tick.PROCEED

    def _phase_idle_skip(self) -> Tick:
        # 等别人出牌的十几秒里协议说"什么都没给我", 画面也不会变: 不抓帧不跑
        # 视觉, 睡一个轮询间隔等协议翻页. 收局后 available() 翻 False, 自然放行.
        if self.t._proto_idle_skip():
            return Tick.SKIP
        return Tick.PROCEED

    def _phase_grab(self) -> Tick:
        self.st.frame = self.t._grab()
        if self.st.frame is None:
            return Tick.SKIP
        return Tick.PROCEED

    # ------------------------------------------------------------------
    # 3: 页面出口
    # ------------------------------------------------------------------
    def _phase_pages(self) -> Tick:
        st, t = self.st, self.t
        frame = st.frame

        # 回到准备页 = 这局结束了. 判据用静态特征而不是开始按钮.
        if t._at_ready_page(st.spec, st.cfg, frame):
            t.log_info(f"第 {st.game} 局结束（回到准备页），共 {st.turns} 轮")
            return Tick.ROUND_OVER

        # 结算页: 必须排在准备页之后、读手牌之前 —— 结算页没有手牌, 让牌面模型
        # 去读只会读到牌河(实测 10 张), 决策全是"在等别人".
        if t._handle_result_page(frame, st.cfg, st.spec, st.game, st.turns):
            return Tick.ROUND_OVER

        # 终局奖励页: 盖住整块画面且没有可匹配按钮, 不点掉会一直卡着.
        if t._handle_reward_page(frame, st.cfg, st.game):
            return Tick.SKIP

        # 协议说这一场打完了(NotifyGameEndResult / NotifyGameTerminate):
        # 视觉的结算页/准备页模板都可能认不出来, 协议不会. 给一小段宽限让画面
        # 切到结算页; 宽限到了也收局, 交给上层的 _back_to_lobby 继续点.
        # 只认"整场"结束, 单手结束(_last_round_end)不算.
        over, reason = t._protocol_says_game_over()
        if over:
            if t._game_over_since == 0.0:
                t._game_over_since = time.time()
                why = f"（{reason}）" if reason else ""
                t.log_info(f"  协议报告本局结束{why}，等结算页出现")
            if t._handle_result_page(frame, st.cfg, st.spec, st.game, st.turns):
                t.log_info(f"第 {st.game} 局结束（协议 + 结算页），共 {st.turns} 轮")
                return Tick.ROUND_OVER
            if time.time() - t._game_over_since >= tuning.ROUND_SETTLE_GRACE:
                t.log_info(f"第 {st.game} 局结束（协议），共 {st.turns} 轮")
                return Tick.ROUND_OVER
        elif t._game_over_since:
            # 协议把"打完"收回去了(服务端 authGame 一到就清 game_over): 计时
            # 起点必须跟着清, 否则宽限被上次的误报吃掉.
            t._game_over_since = 0.0
        return Tick.PROCEED

    # ------------------------------------------------------------------
    # 4-5: 抓包存活 / 牌桌闸门
    # ------------------------------------------------------------------
    def _phase_capture_watch(self) -> Tick:
        # 链路是本任务 spawn 的, 它挂了必须立刻知道 —— 等桥的 stale_after 超时
        # 反推的那 10 秒里, 协议读数冻结在最后一帧会被当真值用.
        if self.t._capture_watch() is False and not self.t._capture_broken:
            self.t._capture_down()
        return Tick.PROCEED

    def _phase_table_gate(self) -> Tick:
        st, t = self.st, self.t
        # 牌桌还没出来(加载页 / 对局开始页 / 和了过场): 手牌区根本没有手牌,
        # 硬读只会得到"认出 0~3 张" + 一堆噪音. 整轮不感知也不决策.
        gate = t._table_gate(st.cfg, st.frame)
        if gate == TABLE_WAITING:
            return Tick.SKIP
        # 牌桌特征失灵超过 TABLE_WAIT_LIMIT 之后不再读手牌: 那时读到的只会是
        # 牌河/过场的幻觉牌. 靠外层 deadline 兜底不锁死.
        if gate in (TABLE_GAVE_UP, TABLE_IGNORING):
            bench.count_state("table_gave_up_skip")
            return Tick.SKIP
        # bench 挂钩: 给接下来的手牌读数打阶段标签.
        bench.set_phase(
            "active" if (t._deal.opened and not t._deal.settling)
            else "settling" if t._deal.settling else "deal_pending")
        return Tick.PROCEED

    # ------------------------------------------------------------------
    # 6-7: 一手结束过场 / 摸牌稳定窗
    # ------------------------------------------------------------------
    def _phase_hand_result(self) -> Tick:
        # 和了/流局的顺位结算画面: 手牌行被面板遮着, 读数全是"9~11 张 + 空档"
        # 的假警报, 这一轮不读手牌.
        if self.t._handle_hand_result(self.st.frame, self.st.cfg, self.st.game):
            return Tick.SKIP
        return Tick.PROCEED

    def _phase_draw_settle(self) -> Tick:
        # 协议刚给自己发牌的这零点几秒里不读手牌不做决策, 等摸到那张落定.
        # 等待轮也必须喂桥: 桥是 pull 式的, 不 poll 状态冻在上一帧.
        if _draw_settle_pending(self.t._net):
            try:
                self.t._net.poll()
            except Exception:  # noqa: BLE001 - 桥坏了不影响这一轮等待
                pass
            return Tick.SKIP
        return Tick.PROCEED

    # ------------------------------------------------------------------
    # 8: 感知(视觉 + 协议对账)
    # ------------------------------------------------------------------
    def _phase_perceive(self) -> Tick:
        st, t = self.st, self.t
        # 协议已经报"整场打完", 现在等的是结算页(`ROUND_SETTLE_GRACE`) ——
        # 这十几秒画面上是摊牌残影, 每轮硬读只会得到 14/13/7/9 张来回跳的
        # 垃圾读数, 还要往 data/debug 灌 lowconf / 待标注素材(永不自动删).
        # 收局判定与结算页识别都在 `_phase_pages`(抓帧后、这一步之前)完成,
        # 所以这里直接停手不会漏掉收局.
        if t._game_over_since:
            bench.count_state("game_over_settle_skip")
            return Tick.SKIP
        # 协议明确说这一手已经结束(Hule/NoTile/LiuJu 已到, 下一手 NewRound
        # 未到): 接下来是摊牌/收牌/结算/发牌动画 —— 该走的是**页面与按钮**
        # 识别(阶段 3/5/6 每轮都在跑), 不是手牌识别(2026-10-18 用户拍板).
        # 残影读数只会喂出"稳健出牌"噪音和假记账: 19:51 局实锤, 门清立直
        # 荣和后收牌动画被读成"8 张活手牌", 连续 3 帧把正确的"副露 0 组"
        # 重同步成 2 组. 闸门还开着就先按协议权威宣告收手(force_rearm),
        # 手尾记账走阶段 12 的 rearmed 分支同一条路; 协议没说话(None, 纯
        # 视觉模式)时保持老行为.
        if _proto_round_open_of(t) is False:
            if t._deal.opened:
                t._deal.force_rearm(time.time())
            bench.count_state("round_over_ui_only")
            return Tick.SKIP
        st.hand = t._read_hand(st.vision, st.frame, st.cfg)
        st.buttons = st.vision.read_buttons(st.frame)
        # 协议对账: 影子档只记日志; 接管档用协议多重集纠正手牌标签(槽位/坐标
        # 不动)并按协议裁按钮. 任何异常/不可用/桥停滞都静默回退纯视觉.
        st.hand, st.buttons, st.net_exempt, st.net_note, st.net_trust = t._net_step(
            st.hand, st.buttons)
        return Tick.PROCEED

    def _phase_hand_gap(self) -> Tick:
        st, t = self.st, self.t
        # 视觉刚报"上一手结束"而协议还没收局的那一两秒: 摊牌/结算画面会被读成
        # 一手 14 张散牌, 张数判据照旧说"轮到我". 必须放在协议对账**之后**:
        # 间隙期一样要 poll 桥, 不然永远收不到收局信号, 守卫变永久闸门.
        if t._hand_gap_poll():
            return Tick.SKIP
        if st.net_trust in ("mismatch", "ambiguous"):
            # 协议有手牌但和视觉对不上(或有歧义): 信任档的失败方向必须是
            # "看得见" —— 这是"再发生就该去修感知"的证据, 存帧.
            t._save_proto_mismatch(st.frame, st.net_note)
        # 张数矛盾从哪一刻开始: 只有它需要"等多久"的计时(见 _hand_conflict_veto).
        if st.net_trust == "conflict":
            if not t._net_conflict_since:
                t._net_conflict_since = time.time()
            # 刚矛盾的那零点几秒通常是**摸牌动画/相邻框被并成一张**: 协议在
            # ActionDealTile 那一刻手里就有 14 张, 画面还没画完. 那不是感知
            # 缺陷, 等下一帧就恢复 —— 不给它存 `待标注_` 素材、也不告警,
            # 否则每摸一张牌就灌一张(2026-10-18 13:48 局: 一条手 2 张全是这个).
            waited_conf = time.time() - t._net_conflict_since
            if waited_conf >= tuning.NET_CONFLICT_SETTLE:
                t._save_proto_mismatch(st.frame, st.net_note)
                t._log_repeat_once(
                    "net_conflict", "conflict",
                    f"  协议手牌与视觉张数对不上已达 {waited_conf:.1f}s"
                    f"（{_conflict_why(st.net_note)}）—— 以协议为准：这手牌"
                    f"不作为决策输入，等下一帧", level="warning")
        else:
            t._net_conflict_since = 0.0
        st.button_conf = {name: boxes[0].conf for name, boxes in st.buttons.items()}
        return Tick.PROCEED

    # ------------------------------------------------------------------
    # 10: 分析
    # ------------------------------------------------------------------
    def _phase_analyze(self) -> Tick:
        st, t = self.st, self.t
        hand = st.hand

        # 协议时钟临近 -> 压缩本轮节拍(跳过拟人延时与二次确认), 拿它换"别被托管".
        clock = t._net_clock()
        t._urgent = t._is_urgent(clock)

        # 出牌 EV: 一次计算供两处用(影子列 + 接管判据), 不算两遍.
        ev_rows, ev_wall = (t._ev_plan()
                            if (tuning.EV_EVAL_SHADOW or tuning.EV_EVAL_TAKEOVER)
                            else (None, None))
        st.ev_shadow = (_ev_record_of(ev_rows, ev_wall)
                        if (tuning.EV_EVAL_SHADOW and ev_rows) else None)
        if st.ev_shadow:
            t._log_repeat_once("ev_shadow", str(st.ev_shadow.get("summary") or ""),
                               f"  [出牌EV/影子] {st.ev_shadow['summary']}")
        st.ev_rank = ev_rows if tuning.EV_EVAL_TAKEOVER else None

        # 立直意图的兜底寿命: 点下立直后本该下一轮就打宣言牌, 但"选牌态认不出
        # 高亮 / 被托管 / 手牌读不出 14 张"任何一个都会让它一直挂着, 挂着时
        # decide_turn 恒返回 WAIT = 整局不出牌. 必须显式到期作废.
        if (st.pending_riichi and t._riichi_since
                and time.time() - t._riichi_since > tuning.RIICHI_DECLARE_TIMEOUT):
            t.log_warning(
                f"  立直点下去 {time.time() - t._riichi_since:.1f}s 还没"
                f"打出宣言牌（超 {tuning.RIICHI_DECLARE_TIMEOUT:.0f}s），"
                f"当这次立直没成，清掉意图回普通打牌")
            st.pending_riichi = False
            t._riichi_since = 0.0
            t._riichi_misses = 0
        # 立直**生效**后的反向核对: 只认协议明确说没有(带宽限期), 见该方法.
        # 放在这里而不是验证那一刻: 回执要等下一次摸牌才下发.
        self._riichi_recheck()

        # 立直选牌态: 点下立直后雀魂让**能打的牌高亮, 其余变暗**, 点一张高亮
        # 牌即完成宣言. 所以只在高亮牌里挑最优(点变暗的牌游戏不响应).
        st.allowed = t._bright_indices(st.frame, hand) if st.pending_riichi else None

        self._analyze_confidence()
        self._analyze_melds()

        st.decision = decide_turn(
            hand.tiles, st.button_conf,
            player_count=st.spec.player_count,
            meld_policy=tuning.MELD_POLICY,
            allow_riichi=tuning.ALLOW_RIICHI,
            meld_min_conf=tuning.MELD_BUTTON_MIN_CONF,
            riichi_min_conf=tuning.RIICHI_BUTTON_MIN_CONF,
            babei_policy=tuning.BABEI_POLICY,
            allowed_indices=st.allowed,
            blurry_indices=st.blurry,
            missing_slots=hand.missing_slots,
            button_centers={name: t._box_center(boxes[0])
                            for name, boxes in st.buttons.items()},
            blurry_draw=t._drawn_blurry,
            melded=t._melded_tiles,
            # 协议权威闸: 只有接管档且桥新鲜时才是 True/False, 其余 None(不参战).
            my_turn=_proto_turn_of(t),
            proto_turn=_proto_clock_turn_of(t),
            # 鸣牌窗口的正面说法: 给 my_turn 那道整轮否决开一个口子(缺了它
            # 每一个碰/吃机会都会被"协议没有出牌操作"拦成跳过).
            proto_meld=_proto_meld_window_of(t),
            round_open=_proto_round_open_of(t),
            ev_rank=st.ev_rank,
            rng=t._rng,
            # 刚摸到的那张(协议值, 带赤记号): 保留赤 5 全靠它.
            drawn_tile=getattr(t, "_drawn_tile", None),
        )
        # 张数就和协议矛盾: 这一手牌连张数都是错的, 打哪张都是瞎猜(见 veto 注释).
        # **没有超时放行**(2026-10-18 20:21 局用户拍板): 视觉坏了就等下一帧,
        # 绝不"超 2s 退视觉继续打" —— 退回去那一刻视觉可能还是假牌/动画框.
        conflict = st.net_trust == "conflict"
        waited = (time.time() - t._net_conflict_since
                  if conflict and t._net_conflict_since else 0.0)
        if _hand_conflict_veto(st.net_trust, st.decision):
            st.decision = TurnDecision(
                ACTION_WAIT, suspect=True,
                reason=f"协议与视觉张数对不上（{_conflict_why(st.net_note)}），"
                       f"以协议为准：这手牌不作为决策输入，"
                       f"本轮不动等下一帧（已等 {waited:.1f}s）")
        st.turns += 1
        # 把**这一帧**一起传进去存证: 让 _record_turn 自己再抓会拿到下一轮的
        # 画面, 留证帧和日志里的 hand 就对不上了.
        t._record_turn(st.game, st.turns, hand, st.buttons, st.decision, st.frame,
                       net=st.net_note, ev=st.ev_shadow,
                       trust=st.net_trust)
        return Tick.PROCEED

    def _analyze_confidence(self) -> None:
        """低置信槽位 / 空档 / 摸牌模糊标志."""
        st, t = self.st, self.t
        hand = st.hand

        # 低置信槽位喂给 decide_turn 走**稳健出牌**: 枚举它们可能的真实取值,
        # 挑一张怎么认都不踩坑的打. 曾经试过"有低置信就整轮不出牌", 数据上
        # 已证伪: 位置 4 在 turn 176~185 连续 10 轮同一个错类别, conf 稳定在
        # 0.64~0.67, 等多久都不恢复 —— 那样会有 73% 的时间在空等.
        st.blurry = (low_confidence(hand.boxes, float(blur))
                     if (blur := tuning.MIN_TILE_CONF) else None)
        if st.blurry and st.net_exempt:
            # 这几槽的真值协议已给定, 不再是"认不准": 再按低置信枚举反而变差.
            st.blurry = [i for i in st.blurry if i not in st.net_exempt]
        if st.blurry and tuning.NET_TRUST_PROTO and st.net_trust == "trusted":
            # 信任档: 全部槽位真值都由协议给定.
            st.blurry = []
        if st.blurry:
            confs = ", ".join(f"{hand.boxes[i].label} {hand.boxes[i].conf:.2f}"
                              for i in st.blurry)
            t._log_repeat_once(
                "blurry",
                "|".join(f"{i}:{hand.boxes[i].label}" for i in st.blurry),
                f"  槽位 {st.blurry} 置信度偏低（{confs}），"
                f"这一轮按稳健出牌算")
        # 刚摸的牌(最右槽)低置信 -> 置粘性标志(打出前不清): 听牌摸切守卫的输入.
        # count % 3 == 2 = "摸牌后"的张数(14/11/8/5/2), 最右槽就是刚摸的那张.
        if (hand.count % 3 == 2 and st.blurry
                and len(hand.tiles) - 1 in st.blurry):
            t._drawn_blurry = True
        # 按间距看出来的漏认: 和置信度无关, 张数可能整个是错的, 必须一路传到
        # 决策层, 否则它会拿一个巧合合法的张数(10)去判"是不是轮到我".
        if hand.missing_slots:
            t._log_repeat_once(
                "missing",
                f"{hand.missing_slots}:{hand.count}",
                f"  手牌行里有约 {hand.missing_slots} 张牌的空档（只认出 "
                f"{hand.count} 张），张数不可信",
                level="warning")

    def _analyze_melds(self) -> None:
        """副露记账: 从"点碰/杠前手牌"与之后稳定读数的差集反推副露牌种."""
        st, t = self.st, self.t
        hand = st.hand

        if (t._pre_meld is not None
                and time.time() - t._pre_meld_at > tuning.PRE_MELD_TIMEOUT):
            # 对账超时: 点了之后从没数出"少 2 张以上" —— 加杠(只挪 1 张, 不增加
            # 副露组数)或按钮没点上. 放弃对账, 不然 _pre_meld 会挂到这一手结束,
            # 把张数硬校验和后续对账一起堵死.
            t._log_repeat_once(
                "pre_meld_expire", f"{len(t._pre_meld)}",
                "  副露对账超时（可能是加杠或按钮没点上），放弃本次对账")
            t._pre_meld = None
        if (t._pre_meld is not None and not hand.missing_slots
                and len(hand.tiles) <= len(t._pre_meld) - 2):
            for tile in meld_taken_kinds(t._pre_meld, hand.tiles):
                if t._melded_tiles.count(tile) < 3:
                    t._melded_tiles += [tile] * 3
            t._pre_meld = None
            # 数出了"少 2 张以上" = 真的鸣进一组, 组数 +1(纯按手数, 加杠/没点上
            # 到不了这里). max 防任何一边多记.
            t._meld_count = max(t._meld_count + 1, len(t._melded_tiles) // 3)

    # ------------------------------------------------------------------
    # 12: 发牌闸门 + 张数硬校验
    # ------------------------------------------------------------------
    def _phase_deal_gate(self) -> Tick:
        st, t = self.st, self.t
        hand = st.hand

        # 两种"这一手不在打"都要拦住: a) 发牌还没结束(张数逐张出现, 中间会撞上
        # "合法张数"); b) 上一手已经结束(摊牌动画里和了牌正面朝上, 读出来是一手
        # 干净的 13~14 张). 根因是同一个: 把"一手牌的状态"当成了"整场任务的状态".
        state = t._deal.see(hand.count, hand.missing_slots, time.time(),
                            melded=t._kan_pending,
                            # 清空边沿的协议否决: 鸣牌动画帧手牌可能只读出 1 张,
                            # 协议说这一手还在打时绝不重臂.
                            round_open=_proto_round_open_of(t))
        if state.just_opened:
            t._hand_seq += 1
            # 记住"这一手是协议的第几手": 手间隙守卫靠它区分"协议还停在
            # 这一手"和"协议已经翻到下一手"(见 _hand_gap_arm).
            t._round_seq_at_open = _proto_round_seq_of(t)
            t.log_info(f"  手牌发齐了（{hand.count} 张），开始按正常逻辑打"
                       f"—— 第 {t._hand_seq} 手开局")
            t._record({"event": "hand_start", "game": st.game,
                       "seq": t._hand_seq, "count": hand.count,
                       "melds": t._meld_count})
            # 双保险归零: 重臂已经清过一次, 这里再清 —— 发齐 13 张的一手副露
            # 必然为 0, 万一重臂边沿被加速过场跳掉, 这道闸保证不跨手污染.
            t._meld_count = 0
            t._count_mismatch_streak = 0
            bench.set_phase("active")
            # 「鸣」默认不点: 点着它雀魂就不画吃碰杠按钮, 决策层连"要不要鸣"
            # 都没机会回答(真机两局协议给了 25 次鸣牌机会, 视觉一次没见到面板).
            # 只有 AUTO_DECLINE_MELD(老的省动作模式)才需要按役对决定点不点,
            # 那时役对判定**不能只看一帧** —— just_opened 那帧把 7z 误读成 6z,
            # 幻影"白板对"让整手没点「鸣」. 默认模式下这段整体跳过.
            yaku = False
            if tuning.AUTO_DECLINE_MELD:
                _, hand2 = t._read_hand_frame()
                yaku1 = has_yaku_pair(hand.tiles, hand.missing_slots, tuning.ROUND_WIND)
                yaku2 = (has_yaku_pair(hand2.tiles, hand2.missing_slots, tuning.ROUND_WIND)
                         if hand2 is not None else yaku1)
                yaku = yaku1 and yaku2
            t._ensure_auto_toggles(st.cfg, st.frame, yaku)

        if state.rearmed:
            t.log_info("  上一手已结束（手牌区清空），发牌闸门重新上膛"
                       f"—— 第 {t._hand_seq} 手结束，本手副露 {t._meld_count} 组")
            t._record({"event": "hand_end", "game": st.game,
                       "seq": t._hand_seq, "melds": t._meld_count})
            self._rearm_hand()
        if state.gave_up:
            t.log_warning(
                f"和了之后 {tuning.GAME_GRACE_PERIOD:.0f}s 没等到结算页，"
                f"退回旧行为继续打。若反复出现，请把 logs 和留证帧发出来")

        if not state.done and st.decision.action in DEAL_BLOCKED_ACTIONS:
            t.log_info(
                f"  这一手没在进行中（手牌 {hand.count} 张，闸门未开），"
                f"这一轮不{st.decision.action}")
            return Tick.SKIP

        # 张数硬校验: 记住本手副露组数后, 暗牌数只有 13-3n / 14-3n 两个合法值.
        # 读数落在集合外 = 读数被污染(副露混进手牌 / 漏认 / 幻觉), 这一轮什么都
        # 不做 —— 三碰局里 8 张恰好 %3==2, 决策层会反复想打已碰掉的牌.
        if state.done and not t._hand_count_ok(hand, st.frame):
            return Tick.SKIP
        return Tick.PROCEED

    def _rearm_hand(self) -> None:
        """换到新的一手: 所有"属于一手"的状态归零."""
        t = self.t
        # 上一手的"这条已经报过"不该继续生效.
        t._repeat_log.clear()
        # 雀魂的自动开关**每一手都重置回关**(实跑: 一局 6 手只在第 1 手点了
        # 「和/鸣」), 所以换手必须复位, 让 just_opened 时真的去点.
        t._toggles_ok = False
        t._kan_pending = False
        t._melded_tiles = []
        t._pre_meld = None
        t._meld_count = 0
        t._count_mismatch_streak = 0
        # 张数矛盾的计时器也属于一手: 新一手重新起算(旧计时会让上一手的
        # 矛盾秒数直接算到这一手头上).
        t._net_conflict_since = 0.0
        # 立直只属于一手: 流局/别人和了后游戏自动开下一手.
        self.st.riichi_active = False
        self.st.pending_riichi = False
        t._riichi_since = 0.0
        t._riichi_active_at = 0.0
        t._my_riichi_confirmed = False
        t._drawn_blurry = False
        # 手间隙入闩: 下面几秒里结算/摊牌画面会被读成一手合法的 14 张散牌.
        t._hand_gap_arm()

    # ------------------------------------------------------------------
    # 13-14: 卡死 / 等待
    # ------------------------------------------------------------------
    def _phase_stuck(self) -> Tick:
        st, t = self.st, self.t
        # 放在决策之后: 它需要知道"这一轮有没有动手". 开局宽限期内不算 ——
        # 发牌动画期间手牌逐张出现, 张数本来就在变.
        if time.time() >= t._deal.grace_until and st.stuck.feed(
            st.hand.count, set(st.buttons), acted=st.decision.action != ACTION_WAIT
        ):
            t._save_unknown_page(
                st.frame, f"连续{st.stuck.same_count}轮无变化_疑似对局已结束")
            t.log_warning(
                f"连续 {st.stuck.same_count} 轮感知结果完全一样（手牌 "
                f"{st.hand.count} 张、按钮 {sorted(st.buttons) or '无'}）且没有任何动作，"
                f"判定这局已经结束但页面没认出来（多半是结算页），已截图留证。"
                f"请用「截图标注」框出结算页的特征区域。")
            return Tick.ROUND_OVER
        return Tick.PROCEED

    def _phase_wait(self) -> Tick:
        st, t = self.st, self.t
        if st.decision.action != ACTION_WAIT:
            st.unknown_streak = 0
            return Tick.PROCEED
        # "在等别人"是每局几十轮的正常状态, 不算"认不出来"; 只有 suspect
        # (张数不对 / 认出怪牌 / 牌效算不动)才计数.
        if st.decision.suspect:
            if time.time() < t._deal.grace_until:
                pass                                    # 开局动画期, 正常
            elif not st.hand.reliable:
                # 读数本身不可信(有空档)时不计数也不留证: 一手结束的顺位画面
                # 读数就是 9~11 张 + 空档, 拿它当"无法识别"只会存无用截图.
                pass
            else:
                st.unknown_streak += 1
                if st.unknown_streak == tuning.UNKNOWN_PAGE_AFTER:
                    t._save_unknown_page(st.frame, f"对局内无法识别_{st.decision.reason}")
        return Tick.SKIP

    # ------------------------------------------------------------------
    # 15: 执行
    # ------------------------------------------------------------------
    def _phase_execute(self) -> Tick:
        st, t = self.st, self.t
        hand = st.hand

        # 连点都没反应时**这一轮不再点**. StuckDetector 看不到这种情况(它要求
        # "这 N 轮没有任何动作", 而我们每轮都在点), 画面冻住时会盲点同一张牌,
        # 甚至把「确定要离开房间吗？」模态框点出来.
        if (st.decision.action == ACTION_DISCARD
                and t._discard_misses >= tuning.DISCARD_MISS_LIMIT):
            t._save_unknown_page(st.frame, f"连续{t._discard_misses}轮出牌没打出去")
            t.log_warning(
                f"连续 {t._discard_misses} 轮点了出牌但手牌一张没少，"
                f"已截图留证。多半是画面上压着模态框（如「确定要离开房间吗？」）"
                f"或画面冻住了，雀魂根本没收到点击。这一轮先不动，下一轮重新认。")
            t._discard_misses = 0
            return Tick.SKIP

        # 这一轮的出牌是不是"立直选牌态里的宣言牌": 只有选牌态真的被认出来
        # (allowed 非 None)且**不是**耐心耗尽后的降级, 才算数. 张数也要是"摸牌
        # 后"那一档, 否则误判的选牌态会把普通出牌当成宣言牌.
        st.declare_turn = (st.pending_riichi and st.allowed is not None
                           and t._riichi_misses == 0
                           and is_my_discard_turn(hand.count))

        # 立直状态机: 点下立直 -> 必须先打宣言牌 -> 真的少了一张才清位.
        # 清位判据是**手牌数**, 不是"这一轮决定了 DISCARD": 雀魂在立直后只认
        # 宣言牌, 点错的地方它直接忽略, 手牌数不会变.
        # **落锁时机**: 立直意图只能在动作**真的执行之后**才置位 —— 早先在这里
        # 先置位再去做双帧确认, 确认否决掉那一轮立直(动作根本没点出去)时意图
        # 却留下来了: 下一轮起状态机以为"我在选宣言牌", 拿 13 张手牌走选牌态
        # 分支, 牌效引擎抛 needs a post-draw hand, 每 0.5s 重试一次白等 26 秒;
        # 拿到第 14 张时又把一张普通牌当宣言牌打掉, 之后整局拒绝出牌被托管.
        if (st.pending_riichi and st.decision.action == ACTION_DISCARD
                and not st.declare_turn):
            # 点了立直但游戏没进选牌态(立直没点成): 按普通出牌打掉, 意图作废.
            st.pending_riichi = False
            t._riichi_since = 0.0
        if st.riichi_active and st.decision.action == ACTION_DISCARD:
            # 立直生效后雀魂自动摸切: 我们的任何"选牌出牌"都是无效点击.
            t._log_repeat_once("riichi_auto", "active",
                               "  立直生效中：雀魂自动摸切，不主动出牌")
            return Tick.SKIP

        # 动作前双帧一致性确认: 错动作(过场乱出牌 / 滑入动画期漏碰)全发生在
        # "牌桌还没稳定下来"的时候, 而幻觉读数和动画帧撑不过第二帧. 这道闸同时
        # 是**单帧误检按钮的过滤器**(假按钮通常只在一帧里出现) —— 这也是为什么
        # 立直意图必须等这里放行之后才置位.
        if _trust_skips_confirm(st.net_trust, st.decision.action,
                                st.declare_turn, st.pending_riichi,
                                _proto_meld_window_of(t)):
            t._log_repeat_once(
                "trust_skip", st.decision.action,
                "  [信任档] 协议=真值、视觉只给坐标，跳过双帧确认直接执行")
        else:
            ok, why = t._confirm_stable(st.decision, hand, st.buttons, st.cfg,
                                        st.spec, st.vision,
                                        st.pending_riichi, st.ev_rank)
            if not ok:
                t._log_repeat_once(
                    "unstable", st.decision.action,
                    f"  画面还没稳定（{why}），本轮不动，下一轮重新决策")
                return Tick.SKIP

        t._execute(st.decision, hand, st.buttons, st.cfg, st.game, st.turns)
        if st.decision.action == ACTION_RIICHI:
            # 到这里立直按钮**真的点下去了**. 记出生时刻做兜底: 之后迟迟打不出
            # 宣言牌时, 上面那道寿命闸会清掉它.
            st.pending_riichi = True
            t._riichi_since = time.time()
            t.log_info("  立直已宣布, 接下来必须打宣言牌")
        # 光标趴在手牌上会把牌顶起来, 实跑里正好挡住立直按钮的模板.
        t._park_mouse(st.frame)

        # 和了之后这一手就结束了, 但雀魂还会放一段摊牌动画(和了牌正面朝上摆回
        # 手牌位置, 读出来是干净的 13~14 张), 然后才是结算页 —— 必须显式封锁,
        # 否则那段动画里会一直点手牌区.
        if st.decision.action == ACTION_WIN:
            t._deal.agari(time.time())
            t.log_info("  和了，等摊牌 / 结算过完，期间不再读手牌和出牌")

        if st.declare_turn and st.decision.action == ACTION_DISCARD:
            self._verify_declaration()
        return Tick.PROCEED

    def _verify_declaration(self) -> None:
        """出牌后重新看一眼手牌: 真的少了一张就算把宣言牌打出去.

        **置位判据只有视觉**(手牌真的少了一张), 协议在这里只做**正向加强**:
        拿到了回执就再也不撤销, 拿不到**什么也不做** —— 回执
        (`ActionDealTile.liqi`)是随下一次摸牌下发的, 实测比宣言牌晚
        0.4~0.6s, 而这里只隔 `DISCARD_VERIFY_DELAY` + 一帧识别, 那一刻问
        必然问不到. 2026-10-18 真机就是这么把一次**成立了的**立直撤销掉的
        (同一手 `ActionHule` 明写 `liqi: true`).
        反向核对带宽限期, 见 `_riichi_recheck`.
        """
        st, t = self.st, self.t
        t.sleep(tuning.DISCARD_VERIFY_DELAY)
        # 残帧会让读数比稳定态少一张, 而这里的判据正是"少没少", 复读一次再判.
        _frame, after = t._read_hand_retry(st.hand.count - 1)
        if after is None or after.count >= st.hand.count:
            return
        t.log_info("  宣言牌已打出，立直生效：之后每回合雀魂自动摸切，"
                   "不再主动出牌")
        st.pending_riichi = False
        t._riichi_since = 0.0
        st.riichi_active = True
        t._riichi_active_at = time.time()
        t._riichi_misses = 0
        if t._proto_my_riichi() is True:
            t._my_riichi_confirmed = True

    def _riichi_recheck(self) -> None:
        """立直生效后的**反向核对**: 协议明确说没有 -> 撤销(带宽限期).

        与 `_verify_declaration` 的分工: 那个只负责置位(视觉事实), 这个负责
        "置错了怎么办". 唯一允许撤销的判据是协议**明确**回答 False(桥可用 +
        知道我是哪个座位 + 座位表里没有我), 且已经过了
        `tuning.RIICHI_VERIFY_GRACE` —— 沉默不是反对.

        为什么必须保留这条撤销: 立直按钮没点上、但选牌态被 `_bright_indices`
        的降级分支当成"全亮"时, 我们会把一张普通出牌当宣言牌, 置上
        `riichi_active` 之后**整手牌不再主动出牌**(只等雀魂托管). 那比误判
        一张牌严重得多, 所以核对要留, 只是不能急.
        """
        st, t = self.st, self.t
        if not st.riichi_active or t._my_riichi_confirmed:
            return
        if not t._riichi_active_at:
            return
        elapsed = time.time() - t._riichi_active_at
        if elapsed <= tuning.RIICHI_VERIFY_GRACE:
            return
        if t._proto_my_riichi() is not False:
            return
        t.log_warning(
            f"  立直置位 {elapsed:.1f}s 后协议里我这一座位仍没有立直回执"
            f"—— 当这次立直没成，撤销状态回普通打牌")
        st.riichi_active = False
        t._riichi_active_at = 0.0
