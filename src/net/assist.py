"""网络读牌接入主循环的对账层(纯逻辑, 不依赖 ok-script/不碰屏幕).

为什么需要"对账"而不是"直接替代":

- 协议给的是**值**的真源: 暗手精确到张、可用操作列表(吃碰杠/立直/和/拔北
  的 type 是确定枚举)、dora/分数/牌山;
- 视觉给的是**位置**的真源: 点击任何按钮/牌都要屏幕坐标, 而协议里没有坐标.

两者不能互相替代, 只能对账:

- 手牌: 用协议多重集去纠正视觉读出的标签, **槽位与坐标一律不动** ——
  这样 `decide_turn` 返回的 index 仍然指向屏幕上的那一槽, 而标签已经是对的.
  这是修 1s/8p 这类"位置对、名字错"的关键.
- 按钮: 协议说了才算 —— 视觉多认的按钮一律砍掉(防误鸣). 协议有而视觉没
  看到的按钮**保留不了**(没有检测框就没有坐标可点), 只如实上报.

对账用不了(桥不可用 / 张数不等 / 池子对不上) -> 一律返回 None, 调用方继续
走纯视觉. 与 §7.4 同构: 宁可漏判不可锁死.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from src.mahjong.ev_eval import draws_from_wall, visible_from
from src.mahjong.tiles import screen_order

from .bridge import LiqiBridge
from .state import CALL_BUTTONS
from .tiles import is_tile, normalize

#: spool 命名约定: tools/mitm/liqi_addon.py 落盘用 data.debug 的同一约定.
SPOOL_GLOB = "liqi_events_*.jsonl"
#: 只认这么久以内更新过的 spool. 旧文件一律当"上一场留下的", 不接.
#: 口径见 tuning.NET_MAX_SPOOL_AGE 的注释: 必须短于"两局之间的常见间隔",
#: 否则上一局打完的 spool 会被误当本局, 桥重放出一份已结束的牌局
#: (round_open=False) -> 整局 no-round、网络读数静默失效.
DEFAULT_MAX_SPOOL_AGE = 90.0


def discover_spool(debug_dir: Path | str, *, now: float | None = None,
                   max_age: float = DEFAULT_MAX_SPOOL_AGE) -> Path | None:
    """在 data/debug 里找"当前正在写"的 spool: 最新、非空、且在 max_age 内.

    找不到返回 None(整个网络读牌静默不参战, 主循环照旧走视觉).
    """
    root = Path(debug_dir)
    if not root.is_dir():
        return None
    now = time.time() if now is None else now
    best: tuple[Path, float] | None = None
    try:
        candidates = list(root.glob(SPOOL_GLOB))
    except OSError:
        return None
    for path in candidates:
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size <= 0 or now - stat.st_mtime > max_age:
            continue
        if best is None or stat.st_mtime > best[1]:
            best = (path, stat.st_mtime)
    return best[0] if best else None


@dataclass
class HandReconcile:
    """协议暗手铺到视觉槽位顺序上的结果."""

    labels: list[str] | None = None          # 与视觉等长同序; None = 用不了
    changed: int = 0                         # 被纠正的槽位数
    unknown_slots: list[int] = field(default_factory=list)   # 纠正过的槽位下标
    ambiguous: bool = False                  # 两槽以上待补: 槽位归属有歧义
    #: 协议和张数就对不上(两边都有值), 或者**布局动画中**(协议与视觉对
    #: "独立槽在不在"的说法相反, 见 reconcile_hand 的守卫): 这一帧的几何
    #: 坐标都不可信 —— 一张都不能拿去算牌效、更不能照着它点牌. 与"协议
    #: 压根没说话"要分开.
    count_conflict: bool = False
    reason: str = ""                         # 用不了/有歧义的原因(日志用)

    @property
    def usable(self) -> bool:
        return self.labels is not None and not self.ambiguous


def reconcile_hand(proto_tiles, vision_tiles, *,
                   drawn: str | None = None,
                   separated: bool | None = None,
                   vision_reliable: bool = True) -> HandReconcile:
    """协议暗手 -> 视觉槽位顺序上的标签. **标签只有一个来源: 协议.**

    为什么不再"保留视觉读对的、只补读错的"(2026-10-18 第三次返工):
    那样做只保证**多重集**正确, 管不了"哪张牌在哪个槽"; 而点击是按槽位下标
    点的. 实跑教训两次都出在同一个缝里:
      - 14:32 局: 视觉把末槽读成 2z、0 槽读成池子里还剩的一张, 补完多重集
        全对、位置全错, 决策按"13 槽是 2z"点下去, 打掉的是刚摸的 8p;
      - 15:21 局: 视觉整条标签被多帧投票推了一格(多重集仍对), 日志写
        "打 3p"、实际打出 4p.
    两次都是"多重集对、位置错", 靠对账本身永远查不出来.

    现在的做法是**把协议算成屏幕顺序, 直接铺到槽位上**(`tiles.screen_order`):
    雀魂把暗牌按 万->筒->索->字 理好. 视觉在这里只贡献"屏幕上从左到右一共
    几个槽、每个槽的坐标"(点击要用的东西).

    两条前提(不满足就不铺, 交给调用方等下一帧):
    - 张数必须相等 —— 不等说明视觉这手牌不是"某张读错"而是整手假牌,
      标 `count_conflict`, 消费方必须整手不信、本轮不动(见 glue.
      _hand_conflict_veto, 没有超时放行);
    - 视觉这帧的张数必须可信(`vision_reliable`, 即 `missing_slots == 0`).
      有漏认空档时槽位序列与屏幕不再一一对应, 铺上去等于猜.

    `drawn` / `separated` 一起决定布局(2026-10-18 16:21 局三处实锤后定稿):
    - `drawn` 是协议里**刚摸到那张的牌名**(`GameState.drawn`, 来自
      `ActionDealTile.tile`; 庄家开局 14 张客户端整手排序、摸牌插进排序位,
      **不记** drawn, 见 state._new_round), 不是"列表末位": 鸣牌之后
      手里 11 张(%3==2)但没有摸牌, 客户端把 11 张全部重排, 按"末位=刚摸"
      铺会把从摸牌排序位置起的所有标签推后一格 —— 点"9p"打出 4p 就是它;
    - `separated` 是**视觉看到的"末位独立槽在不在"**(末两框中心距明显大于
      中位间距): 摸牌分离动画有先后, 分离没做完时按独立槽铺照样串位.
      所以:
        drawn 给了 + separated=True  -> `sort(其余)` + `[drawn]`(常规摸牌帧);
        drawn 给了 + separated=False -> **布局动画中, 不铺**(见下);
        drawn 给了 + separated=None  -> 按独立槽铺(视觉框拿不到, 信协议);
        drawn=None                   -> 整手排序(等待帧 / 鸣牌后 / 庄家开局).

    **布局一致性守卫(2026-10-18 18:56 局实锤)**: 协议与视觉对"独立槽在不在"
    的说法**相反**时, 这帧的几何坐标不可信 —— 摸牌插入/分离动画正在进行:
      - drawn 给了但 separated=False: 摸牌还没分出去, 分出去后中间的牌
        会整体挪一格, 现在按整手排序铺、点下去时槽位已经移位;
      - drawn=None 但 separated=True: 协议说这手没有摸牌(庄家开局/等待帧/
        鸣牌后, 整手排序不该有缝), 视觉末尾却有缝 —— 缝正在合拢
        (18:56 局: "发齐了"闸门放行时 6m 还在往第 3 位飞, 3s 后才落定,
        按动画中的框点"1z"打出 9s).
    两种都标 `count_conflict`(同一后果: 本轮不动等下一帧, 见 glue.
    _hand_conflict_veto —— 没有超时退视觉), 等下一帧动画做完再铺.

    返回的 `labels` 与 `vision_tiles` **等长同序**: 第 i 个标签就是第 i 个
    检测框(坐标)上该点的牌. `changed` 是"视觉读错了几张"(纯统计, 不影响
    结果); `unknown_slots` 是这些槽位的下标.
    """
    if not proto_tiles or not vision_tiles:
        return HandReconcile(reason="empty")
    if len(proto_tiles) != len(vision_tiles):
        return HandReconcile(
            count_conflict=True,
            reason=f"张数对不上: 协议{len(proto_tiles)}/视觉{len(vision_tiles)}")
    if not all(is_tile(t) for t in proto_tiles):
        return HandReconcile(reason="proto tile invalid")
    if not vision_reliable:
        # 张数都不可信: 槽位序列与屏幕不对应, 这一帧不铺(等下一帧).
        return HandReconcile(reason="视觉张数不可信(有漏认空档), 不铺标签")
    if separated is not None and ((drawn is None) == separated):
        # 布局一致性守卫: 协议与视觉对"独立槽在不在"说法相反 = 动画进行中,
        # 这帧的框坐标马上就要变, 铺上去点下去也是串位(见上面 docstring).
        # 真值表: drawn=None+有缝 -> 缝在合拢; drawn=牌名+无缝 -> 缝还没分开.
        why = ("协议有摸牌但视觉还没分出独立槽" if drawn is not None
               else "协议无摸牌(整手排序)但视觉末尾有缝")
        return HandReconcile(
            count_conflict=True,
            reason=f"布局动画中({why}), 等下一帧")

    try:
        if drawn is not None and separated is False:
            # 客户端还没把摸牌分出去(整手连续排序): 摸牌按排序位置落在中间.
            labels = screen_order(proto_tiles, drawn=None)
        else:
            labels = screen_order(proto_tiles, drawn=drawn)
    except ValueError as exc:            # 协议里出现认不出的牌名
        return HandReconcile(reason=f"协议牌名无法解析: {exc}")

    wrong = [i for i, (a, b) in enumerate(zip(labels, vision_tiles))
             if normalize(a) != (normalize(b) if is_tile(b) else None)]
    return HandReconcile(
        labels=labels,
        changed=len(wrong),
        unknown_slots=wrong,
        ambiguous=False,
        reason="",
    )


@dataclass
class ButtonReconcile:
    """协议为准的按钮集合(只保留视觉也看见的, 因为只有它有坐标)."""

    keys: list[str] = field(default_factory=list)
    vision_only: list[str] = field(default_factory=list)   # 视觉多认, 已砍
    proto_only: list[str] = field(default_factory=list)    # 协议有但无坐标
    cut_skipped: bool = False    # 协议无操作/视觉有鸣牌按钮 -> 为防滞后不裁


def reconcile_buttons(vision_keys, proto_keys) -> ButtonReconcile:
    """按协议裁按钮: 交集才保留; 视觉多认的砍掉(防误鸣).

    例外(防协议滞后): 协议一条操作都没给、视觉却看见吃碰杠/和 时, **不裁** ——
    更可能是我方状态机落后于画面(比如某个新动作没建模), 此时砍按钮会让机器人
    面对一个真实的鸣牌机会却按兵不动. 只记录, 交由视觉与既有守卫决定.
    """
    vision = list(dict.fromkeys(vision_keys))
    proto = list(dict.fromkeys(proto_keys))
    proto_set, vision_set = set(proto), set(vision)
    says_nothing = not proto and bool(vision_set & CALL_BUTTONS)
    return ButtonReconcile(
        keys=list(vision) if says_nothing
        else [k for k in vision if k in proto_set],
        vision_only=[] if says_nothing
        else [k for k in vision if k not in proto_set],
        proto_only=[k for k in proto if k not in vision_set],
        cut_skipped=says_nothing,
    )


@dataclass
class NetStep:
    """一轮对账的产出. 影子模式只用 status/note 打日志."""

    available: bool = False
    status: str = ""
    hand: object | None = None                 # assist: 纠正后的 HandReading
    buttons: dict | None = None                # assist: 裁剪后的检测框表
    blurry_exempt: frozenset = frozenset()
    note: str = ""
    #: 协议手牌与视觉对上(计数一致且无歧义) = 本轮手牌标签就是协议真值,
    #: 消费方可据此跳过低置信/双帧重算(见 tuning.NET_TRUST_PROTO).
    trusted: bool = False
    #: 协议**看得出手牌多重集不对, 但定不到位**(2 槽以上待补): `hand` 里的
    #: 多重集已是协议真值, 可"哪张牌在哪个槽"还是猜的 —— 消费方可以拿它算
    #: 牌效, 但**不能**照着它的下标点牌(点下去是别的牌).
    ambiguous: bool = False
    #: 协议与视觉**张数就对不上**(两边都有值). 与 ambiguous 的区别: 那里多重集
    #: 是协议真值, 这里视觉那一手牌连张数都是错的 —— 决策层一张都不能用.
    conflict: bool = False


class NetAssist:
    """主循环每轮 poll 一次的门面. 桥不可用时一切返回"用不了"."""

    def __init__(self, path: Path | str | None = None, *,
                 debug_dir: Path | str | None = None,
                 max_spool_age: float = DEFAULT_MAX_SPOOL_AGE,
                 clock=time.time):
        self._explicit = Path(path) if path else None
        self._debug_dir = debug_dir
        self._max_spool_age = max_spool_age
        self._clock = clock
        self._bridge: LiqiBridge | None = None
        self._path: Path | None = None

    # ------------------------------------------------------------- lifecycle

    @property
    def path(self) -> Path | None:
        return self._path

    def _discover(self) -> Path | None:
        if self._explicit is not None:
            return self._explicit
        if self._debug_dir is None:
            return None
        return discover_spool(self._debug_dir, now=self._clock(),
                              max_age=self._max_spool_age)

    def _ensure_bridge(self) -> LiqiBridge | None:
        if self._bridge is not None:
            return self._bridge
        path = self._discover()
        if path is None:
            return None
        self._path = path
        self._bridge = LiqiBridge(path)
        return self._bridge

    def poll(self) -> int:
        """读新事件. 返回应用数(没桥时 0). 绝不抛."""
        bridge = self._ensure_bridge()
        if bridge is None:
            return 0
        try:
            return bridge.poll()
        except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
            return 0

    def status(self) -> str:
        bridge = self._ensure_bridge()
        if bridge is None:
            return "no-spool"
        try:
            return bridge.status()
        except Exception:  # noqa: BLE001
            return "bridge-error"

    def available(self) -> bool:
        bridge = self._ensure_bridge()
        if bridge is None:
            return False
        try:
            return bridge.available()
        except Exception:  # noqa: BLE001
            return False

    def status_detail(self) -> str:
        """status() 说"不可用"时, 补一句**到底是哪条报文**把桥判死的.

        **为什么必须有**(2026-10-17 真机): 日志只有一句"协议出现未建模动作,
        已回退视觉", 而一次未建模动作会把 protocol_suspect 一路挂到下一局开局
        —— 于是表现成"某一整手牌 net 全空", 现场完全看不出是哪条报文害的, 只能
        事后把整个 spool 重放一遍才查出来(那次是 `ActionChiPengGang type=-1`,
        根因是 `get("type", -1) or 0` 把 proto3 省略的 0=吃 判成了漂移).

        优先报 `unhandled` 计数最高的几条; 没有就退回 `last_event`. 绝不抛.
        """
        st = self.state
        if st is None:
            return ""
        try:
            items = getattr(st, "unhandled", None) or {}
            if items:
                top = sorted(items.items(), key=lambda kv: (-kv[1], str(kv[0])))
                return "; ".join(f"{k} ×{v}" for k, v in top[:3])
            return str(getattr(st, "last_event", "") or "")
        except Exception:  # noqa: BLE001
            return ""

    @property
    def state(self):
        bridge = self._bridge
        return bridge.state if bridge is not None else None

    def seconds_since_self_draw(self) -> float | None:
        """距离协议给自己发上一张牌过了几秒; 拿不到就是 None(不参战).

        消费方是摸牌渐入动画的稳定窗(见 tuning.NET_DRAW_SETTLE): 摸到那张
        落定前读手牌, 低置信/对不上/存帧全是动画噪声. 返回 None = 协议不可用,
        稳定窗不生效(多看一眼永远安全).
        """
        st = self.state
        if st is None:
            return None
        try:
            ts = float(getattr(st, "last_self_draw_at", 0.0) or 0.0)
        except Exception:  # noqa: BLE001 - 老状态对象没这个字段时不参战
            return None
        if ts <= 0.0:
            return None
        try:
            return max(0.0, time.time() - ts)
        except Exception:  # noqa: BLE001
            return None

    # ---------------------------------------------------- 牌局起止(协议判定)

    def session(self) -> dict:
        """协议对"这场牌走到哪了"的判断(不依赖 available(): 桥断了也能看).

        与 available() 的分工: available() 回答"**现在能不能拿协议来对账读数**"
        (要求这一手在打 + 有座位 + 喂料新鲜 + 协议没可疑); session() 回答
        "**这场牌整个的进度**" —— 主循环等匹配成功看 match_started、判收局看
        game_over, 这两个时刻根本还没有"进行中的一手", available() 必然是 False.

        没有 state(桥没建起来/抓包没开)时全部返回保守值, 调用方据此回退视觉.
        """
        st = self.state
        if st is None:
            return {"match_started": False, "round_open": False,
                    "game_over": False, "reason": "", "last_round_end": ""}
        return {
            "match_started": bool(getattr(st, "match_started", False)),
            "round_open": bool(getattr(st, "round_open", False)),
            "game_over": bool(getattr(st, "game_over", False)),
            "reason": str(getattr(st, "game_over_reason", "") or ""),
            "last_round_end": str(getattr(st, "last_round_end", "") or ""),
        }

    def match_started(self) -> bool:
        """协议说这一场开始了没有(等匹配成功时用). 先 poll 一次再问.

        绝不抛: 抓包没开/桥异常时返回 False, 调用方回退视觉判定.
        """
        try:
            self.poll()
        except Exception:  # noqa: BLE001
            return False
        return self.session()["match_started"]

    def game_over(self) -> bool:
        """协议说这一场打完了没有(收局判定用). 先 poll 一次再问."""
        try:
            self.poll()
        except Exception:  # noqa: BLE001
            return False
        return self.session()["game_over"]

    # ------------------------------------------- 时机与桌况(协议侧真源)

    def turn_clock(self) -> dict:
        """协议说的"轮到我了吗 / 还剩几秒".

        与 available() 分开的理由和 session() 一样: "还剩几秒"恰恰要在**决策那
        一刻**取, 而那一刻画面刚稳定、桥未必喂够料. 拿不到就给保守值
        (my_turn=False, left=None) —— 调用方据此走老行为, 见 tuning.NET_TURN_CLOCK.

        `offered` 单独报出来, 因为 **"协议没说话" 和 "协议说不是你" 是两件事**:
        offered 为空 = 服务端还没有给我操作列表(刚摸牌/刚过场/上一个窗口已作废),
        这时 my_turn 的 False 只是"没有信息"的保守填充, 拿它当否决就是拿沉默当
        反对 —— 会把一次真出牌否掉, 整手牌拖进托管(见 PlayOneGameTask._proto_clock_read).
        """
        st = self.state
        if st is None:
            return {"my_turn": False, "offered": False, "left": None,
                    "budget": 0.0}
        try:
            return {"my_turn": bool(st.protocol_says_my_turn()),
                    "offered": bool(st.offered),
                    "left": st.turn_time_left(),
                    "budget": float(st.turn_budget())}
        except Exception:  # noqa: BLE001 - 老状态对象没这些方法时不参战
            return {"my_turn": False, "offered": False, "left": None,
                    "budget": 0.0}

    def round_seq(self) -> int | None:
        """协议开过几手了(单调递增). None = 没有状态对象(不参战).

        与 `round_open` 的分工: 那个回答"这一手还在打吗", 这个回答"协议翻到
        第几手了". `ActionNewRound` 一到手就 +1, 与 (chang,ju,ben) 的取值无关
        —— 友人场连打两场, 两场的东一局都是 (0,0,0), 只有这个计数分得开.
        """
        st = self.state
        if st is None:
            return None
        try:
            return int(getattr(st, "round_seq", 0) or 0)
        except Exception:  # noqa: BLE001 - 老状态对象没这个字段时不参战
            return None

    def round_open(self) -> bool | None:
        """协议说"这一手还在打吗". None = 协议不在这场对局里(不参战).

        收局守卫的依据(见 table.decide_turn 的 round_open): 别家胡牌/流局后
        `round_open` 翻 False, 直到下一手 ActionNewRound 才回 True —— 那段
        画面上的手牌是结算残留, 任何"稳定出牌"读数都是幻觉. 注意它**独立于
        available()**: 收局后 available() 已经 False, 其余协议闸门全部哑火,
        这一个标志反而是当时唯一还活着的协议信号.

        `match_started` 为假(没开抓包/还没进对局)时**必须返回 None**: 消费者
        拿 False 当"这一手确定结束"去否决出牌, 协议压根不存在时否决 = 把
        所有出牌永久拦死.
        """
        st = self.state
        if st is None:
            return None
        try:
            if not st.match_started:
                return None
            return bool(st.round_open)
        except Exception:  # noqa: BLE001 - 老状态对象没这个字段时不参战
            return None

    def meld_options(self) -> list[dict]:
        """当前可鸣的牌(吃/碰/大明杠)与协议给的具体组合."""
        st = self.state
        if st is None:
            return []
        try:
            return list(st.meld_options())
        except Exception:  # noqa: BLE001
            return []

    def table_view(self) -> dict:
        """桌况摘要(手牌/副露/宝牌/场风/牌山) —— `ev_inputs()` 拿它拼估值入参.

        **只取协议能确定给出的东西**, 不掺视觉读数: 影子档的目的是把"策略好
        不好"和"识别准不准"分开量.
        """
        st = self.state
        if st is None:
            return {}
        try:
            return {
                "hand": list(st.hand),
                "melds": st.meld_count(),
                "melded": st.melded_tiles(),
                # 协议侧副露**对象**(出牌 EV 重建"鸣牌后的手"要用它们;
                # tiles/melds 只有扁平牌与个数, 造不出 jansou 的 Meld).
                "meld_objs": list(st.melds),
                "doras": list(st.doras),
                "player_count": st.player_count,
                "field_wind": st.field_wind(),
                "seat_wind": (st.seat_wind(st.my_seat)
                              if st.my_seat is not None else None),
                "waits": list(st.my_waits),
                "wall": st.wall,
            }
        except Exception:  # noqa: BLE001
            return {}

    def ev_inputs(self) -> dict:
        """EV 出牌估值的入参(牌河/副露/宝牌指示牌/座位/牌山), 全部协议事实.

        这里做的是**翻译**: 把 GameState 里各家的牌河与副露摊平成 jansou 要的
        `visible` 数组(见 src/mahjong/ev_eval.visible_from), 再把牌山换算成
        "我还能摸几张". 策略本身不在这一层.

        `visible` 的口径是"**手牌之外**看得见的牌", 所以手牌**不进去** ——
        `st.players[my_seat].melds` 天然是空的(自己的副露记在 `st.melds`),
        不会重复计. 任何异常都返回空 dict, 调用方走老路径.
        """
        view = self.table_view()
        if not view:
            return {}
        st = self.state
        try:
            my_melds = list(st.melds)
            all_melds = list(my_melds)
            discards: list[str] = []
            for seat in sorted(st.players):
                player = st.players[seat]
                discards.extend(tile for tile, _ in player.discards)
                all_melds.extend(player.melds)
            indicators = list(view["doras"])
            my_seat = st.my_seat if st.my_seat is not None else 0
            # 人数交叉校验: `player_count` 在见过 AuthGame 之前是默认值 4, 而
            # **传错人数是静默灾难** —— 三麻按 4 算只是概率略偏, 反过来(四人
            # 场按 3 算)会让手里 2m-8m 变成"永远组不成面子", 整手牌和不了,
            # `win`/`points` 恒 0, 排序退化成按牌名升序(见 ev_eval.outside_play)。
            # 座位表是更硬的事实, 两者矛盾时以座位表为准。
            player_count = view["player_count"]
            seats = len(st.players)
            if seats in (3, 4) and seats != player_count:
                player_count = seats
            return {
                "hand": list(view["hand"]),
                "my_melds": my_melds,
                "melds": view["melds"],
                "visible": visible_from(discards=discards, melds=all_melds,
                                        indicators=indicators),
                "draws": draws_from_wall(view["wall"], player_count),
                "discards": discards,
                "doras": indicators,
                "player_count": player_count,
                "round_wind": view["field_wind"],
                "seat_wind": st.seat_wind(my_seat),
                "my_seat": my_seat,
                "wall": view["wall"],
            }
        except Exception:  # noqa: BLE001
            return {}

    def reset_session(self) -> None:
        """把"这一场"的起止标志清掉(连打时等下一场匹配之前调用).

        为什么必须有: match_started 一旦为真就永远为真 —— 上一场打完、下一场
        还在匹配排队时它还是 True, 于是"等匹配成功"的协议优先判定会立刻
        假阳返回. 绝不抛.

        `last_step` 一起清: 水位是**按局**去的, 而"新一场的第一手"可能和
        上一场最后一手同号(友人场 一局: 两场都是东一局 (0,0,0)) —— 不清就
        会把新一场的所有事件当 syncGame 重放吞掉, 整场不出牌(2026-10-18).
        """
        st = self.state
        if st is None:
            return
        for attr, val in (("match_started", False), ("game_over", False),
                          ("game_over_reason", ""), ("round_open", False),
                          ("round_id", None), ("last_step", -1)):
            try:
                setattr(st, attr, val)
            except Exception:  # noqa: BLE001 - 老状态对象没这字段时不参战
                pass

    # ------------------------------------------------------------------ step

    def step(self, hand, buttons, *, assist: bool) -> NetStep:
        """一轮对账: poll 一次, 拿协议状态与视觉读数比.

        assist=False(影子): 只产出 status/note 供日志, hand/buttons 恒 None.
        assist=True (接管): 手牌标签纠正 + 按钮裁剪同时生效.
        """
        self.poll()
        if not self.available():
            return NetStep(available=False, status=self.status())
        state = self.state
        if state is None:
            return NetStep(available=False, status="no-state")

        proto_hand = list(getattr(state, "hand", None) or [])
        vision_tiles = list(getattr(hand, "tiles", None) or [])
        # "协议末位 = 刚摸的那张(屏幕最右独立槽)"的判据是**张数奇偶**, 不是
        # 有没有副露: 三麻暗手 张数%3==2 才是"刚摸完"(14/11/8/5/2), %3==1
        # 是"等别人打/鸣牌窗口"(13/10/7/4/1) —— 那时屏幕上 13 张全部理好、
        # 没有独立槽, 协议 hand 还是**发牌顺序**(state._new_round 原样存),
        # 必须整手排序. 开局按"有没有副露"判会把 13 张等待帧错切成
        # "排序前 12 张+末位追加", 铺出来整手串位.
        # "刚摸到的那张"以协议为准(ActionDealTile.tile; 庄家开局 14 张客户端
        # 整手排序、不记 drawn, 见 state._new_round), 绝不按张数奇偶猜 ——
        # 鸣牌后手里 3n+2 张但没有摸牌, 按奇偶会把协议列表末位钉到最右独立
        # 槽, 客户端其实是整手重排(16:21 局点"9p"打出 4p 就是这么来的).
        drawn = getattr(state, "drawn", None) or None
        if drawn is not None and not is_tile(drawn):
            drawn = None
        # 视觉看"末位独立槽"分离了没有: 摸牌分离动画有先后 —— 新手第一巡
        # 客户端先摆 14 张连续排序, 过一会儿才分出独立槽; 分离前按独立槽铺
        # 标签照样串位(16:21 局点"3z"打出 2s). 判据: 末两框中心距明显大于
        # 中位框距(实测分离帧 ~1.3 倍, 连续帧 1.0 倍). 框拿不到就信协议.
        separated = _last_slot_separated(getattr(hand, "boxes", None))
        # 视觉自己报了漏认空档时, 槽位序列与屏幕不再一一对应, 不铺(等下一帧).
        vision_reliable = not (getattr(hand, "missing_slots", None) or [])
        rec = reconcile_hand(proto_hand, vision_tiles,
                             drawn=drawn, separated=separated,
                             vision_reliable=vision_reliable)
        proto_keys = state.offered_button_keys()
        vision_keys = list(buttons.keys()) if buttons else []
        btn = reconcile_buttons(vision_keys, proto_keys)

        notes = [f"牌={len(proto_hand)}/{len(vision_tiles)}",
                 f"协议按钮={proto_keys or '无'}"]
        # 时机: 协议说"轮到我"与视觉张数推出的"轮到我"是不是一回事.
        # 这是"视觉判出牌时机能不能被接管"的直接证据(见 tuning.NET_TURN_CLOCK),
        # 所以两个方向分开记 —— "协议说该打牌而视觉没看出来"会漏一手,
        # "视觉看着该出牌而协议说不是"则可能是一张幻觉牌点了出去, 代价完全不同.
        clock = self.turn_clock()
        vis_turn = len(vision_tiles) % 3 == 2
        if clock["my_turn"]:
            notes.append("协议:轮到我打牌")
        if clock["my_turn"] and not vis_turn:
            notes.append("协议该打牌但视觉没看出来")
        elif vis_turn and not clock["my_turn"]:
            notes.append("视觉看着像该出牌但协议说不是")
        if clock["left"] is not None:
            notes.append(f"协议倒计时剩{clock['left']:.0f}s")
        # 对手立直/立直棒/听牌这些协议侧事实一律进 note: 决策层不吃它们,
        # 但复盘时要靠它判断"那巡为什么这么打"
        try:
            opp_riichi = state.opponent_riichi()
        except Exception:  # noqa: BLE001 - 老状态对象没这方法时不参战
            opp_riichi = []
        if opp_riichi:
            notes.append(f"对手立直={opp_riichi}")
        if getattr(state, "liqibang", 0):
            notes.append(f"立直棒={state.liqibang}")
        if getattr(state, "my_waits", None):
            notes.append(f"听={state.my_waits}")
        if getattr(state, "my_zhenting", False):
            notes.append("我振听")
        if rec.changed:
            notes.append(f"纠正{rec.changed}槽{rec.unknown_slots}")
        if btn.vision_only:
            notes.append(f"砍掉视觉多认{btn.vision_only}")
        if btn.proto_only:
            notes.append(f"协议有但视觉没看到{btn.proto_only}")
        if btn.cut_skipped:
            notes.append("协议无操作而视觉有鸣牌按钮, 不裁")
        if rec.reason:
            notes.append(f"({rec.reason})")
        step = NetStep(available=True, status="ok", note=" ".join(notes),
                       trusted=rec.usable, ambiguous=rec.ambiguous,
                       conflict=rec.count_conflict)

        if not assist:
            return step
        if rec.changed and rec.labels is not None and (rec.usable or rec.ambiguous):
            # 有歧义也照铺: 铺出来的**多重集**就是协议真值, 牌效据此算才对;
            # 槽位归属的不可靠性由 ambiguous 标志交给消费方处理(不许按这个
            # 下标点牌). 不铺的话决策只能拿视觉那手错牌去算 —— 2026-10-18
            # 13:19 局: 视觉把摸到的 7m 读成 9p、还漏了一张 2z, 牌效照错的
            # 手牌算了一整手.
            step.hand = _relabel(hand, rec.labels)
            step.blurry_exempt = frozenset(rec.unknown_slots)
        if btn.vision_only:
            cut = set(btn.vision_only)
            step.buttons = {k: v for k, v in buttons.items() if k not in cut}
        return step


def _relabel(hand, labels):
    """套协议标签回视觉手牌(坐标/置信度全保留). 延迟 import 免拖慢非对局路径."""
    from src.detector import with_labels
    return with_labels(hand, labels)


def _last_slot_separated(boxes) -> bool | None:
    """视觉判断"最右那格是不是独立槽"(摸牌分离动画做完没有).

    雀魂把刚摸的牌和前面整手之间留一道缝 —— 末两框的中心距比中位框距大一截
    (实帧实测: 分离后 ~1.3 倍, 没分离/没有独立槽时 1.0 倍). 框数不足 3 个
    或坐标缺失时返回 None(判不了, 调用方信协议).

    为什么必须看视觉而不是只信"协议说有摸牌": 新手第一巡客户端先把 14 张
    **连续**排序摆出来, 摸牌分离动画过一会儿才做完 —— 2026-10-18 16:21 局:
    同一手前一帧末槽框距均匀(没分离), 0.3s 后的另一帧出现缺口. 分离前按
    独立槽铺标签, 点"3z"打出去的是连续排序那一槽的 2s.
    """
    try:
        pts = sorted((float(b.x1) + float(b.x2)) / 2.0 for b in boxes)
    except (TypeError, AttributeError, ValueError):
        return None
    if len(pts) < 3:
        return None
    gaps = [b - a for a, b in zip(pts, pts[1:])]
    if any(g <= 0 for g in gaps):
        return None
    median = sorted(gaps)[len(gaps) // 2]
    return gaps[-1] > median * 1.2
