"""liqi 事件 -> 完整牌局模型 GameState(纯逻辑, 不依赖 ok-script).

与服务端真实下发的信息对齐(比视觉读数多、且是确定性的):

- 自己的暗手: 精确到张, 含赤 5("0p" 这种原始串也留着)
- 三家每张出牌 + 是否摸切(moqie), 副露明细, 立直标记 + 立直棒数
- dora(随杠增长), 分数, 场风/座位风, 本场/立直棒, 牌山余数

**立直有四条独立下发线索(全部建模, 少一条就会漏判对手立直)**:

1. `ActionDiscardTile.is_liqi` / `is_wliqi` —— 立直宣言牌(两立直走 `is_wliqi`);
2. `ActionDealTile.liqi`(`LiQiSuccess`) —— 立直**成立**回执: 摸下一张牌时下发
   `{seat, score(-1000), liqibang, failed}`, **这是唯一带"扣 1000 点 + 立直棒 +1"
   的权威来源**, 也是最容易被漏掉的一条(雀魂用它宣布立直成立);
3. `ActionRevealTile.seat + is_liqi/is_wliqi` —— 立直宣言牌的公开(横放宣言牌);
4. `HuleInfo.liqi` —— 和牌者的立直标记(交叉校验; 有立直才可能有 `li_doras` 裏ドラ).

四条线索都不带"只看得到自己"的限制, 所以**任一路径成立即认定立直**. 只吃第 1
条是有风险的: 已有实证是"立直宣言牌"进 `is_liqi`, 但"立直成立"的权威
(扣 1000 点 + 立直棒 +1)在 `LiQiSuccess`, 少了它连自己的立直棒都对不上账.
- **此刻我能做什么**: 每次轮到我, 协议都会带 OptionalOperationList,
  即吃/碰/杠/立直/自摸/和 的具体选项(combination 是具体牌组合)
- **"轮到我"有权威判据**: `operation_list` 里的 `type=1` 就是「打牌」——
  它没有按钮框, 所以过去只被当成"有个未知操作", 白白浪费了服务端直接给出的
  时机. 另外同一份 operation 还带 `time_fixed` / `time_add`(思考时间, ms),
  配合本地收到时刻就能算出"还剩几秒"(见 turn_time_left).

拿不到的(服务端本来就不发): 其他三家的暗手; 他家摸到但没打的牌
(ActionDealTile 只把 tile 发给摸牌者本人).

与项目铁律对齐:
- 任何事件解析失败都不抛: 记 protocol_suspect + unhandled 计数,
  让上层决定回退视觉(宁可漏判不可锁死);
- 事件按 step 去重(syncGame 重放会重复喂);
- 牌张合法性用 net.tiles 校验, 不造第二套记号.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

from . import liqi
from .liqi import LiqiMessage, MsgType, decode_nested_action
from .tiles import is_tile, normalize

# OptionalOperation.type -> 决策层按钮 key(与 src/mahjong/table.py 的常量对齐).
# 数值口径(与 Akagi/MahjongCopilot 及牌谱内 operation_list.type 一致):
#   他家打牌后给我的: 2=吃 3=碰 5=大明杠 9=荣和
#   我摸牌后给自己的: 1=打牌 4=暗杠 6=加杠 7=立直 8=自摸 11=拔北(三麻)
#   (10=九种九牌不映射按钮, 12/13/14/16/17 是换牌/定缺等特殊规则, 不进本表)
OP_TYPE_TO_BUTTON = {
    2: "chi",       # 吃
    3: "peng",      # 碰
    4: "gang",      # 暗杠(不破门清, 决策层单独判, 见 table.decide_turn 第 5 步)
    5: "gang",      # 大明杠
    6: "gang",      # 加杠
    7: "lizhi",     # 立直
    8: "zimo",      # 自摸
    9: "he",        # 荣和
    11: "babei",    # 拔北(三人麻将; 北在本项目不是手牌槽位, 见 table.py ACTION_BABEI)
}
# 有吃碰杠/荣和机会时, "跳过"始终是可选项(雀魂一定给这个按钮)
CALL_BUTTONS = frozenset({"chi", "peng", "gang", "he"})

# OptionalOperation.type 里唯一的"不是按钮"的那一项: 1 = 打牌.
# 它出现在 operation_list 里就表示**服务端在说"轮到你打牌了"** —— 这是"轮到我"
# 的权威判据, 比视觉的"张数 % 3 == 2"硬(那个是从读出来的张数反推的, 漏认一张
# 就推错; 见 table.is_my_discard_turn). 之所以过去没建模, 是因为它没有按钮框,
# 只被当成"有个未知操作"跳过了.
OP_TYPE_DISCARD = 1

#: 只有这三种是"鸣"(别人打牌后给我的): 2=吃 3=碰 5=大明杠.
#: 暗杠(4)/加杠(6)是自己回合的操作, 不属于"鸣牌机会", 不在这里.
OP_TYPE_TO_MELD = {2: "chi", 3: "peng", 5: "minggang"}

# ActionChiPengGang.type
CPG_CHI, CPG_PENG, CPG_MINGGANG = 0, 1, 2
# ActionAnGangAddGang.type(majsoul_wrapper sdk 的映射; 验证夜需实测确认)
AAG_JIAGANG, AAG_ANGANG = 2, 3

# 与日麻无关的动作(川麻换牌/定缺/血战、四川杠结果等)。它们必须被吃掉而不是
# 当成"协议漂移": 一旦走 _unhandled 就把 protocol_suspect 置真 -> 上层判定
# 桥不可用 -> 整局静默回退视觉, 而其实我们什么都没漏.
BENIGN_ACTIONS = frozenset({
    "ActionUnveilTile",        # 揭示(三麻/川麻的公开)
    "ActionLockTile",          # 川麻定缺
    "ActionChangeTile",        # 川麻换三张
    "ActionSelectGap",         # 川麻定缺选项
    "ActionFillAwaitingTiles", # 补齐待牌(血战)
    "ActionGangResult",        # 川麻杠结果
    "ActionGangResultEnd",     # 川麻杠结果结束
    "ActionHuleXueZhanMid",    # 血战中途和
    "ActionHuleXueZhanEnd",    # 血战结束和
    "ActionNewCard",           # 补牌标记
})

MAX_CONCEALED = 14
MIN_CONCEALED = 2


@dataclass
class Meld:
    """一副副露."""

    kind: str                 # chi/peng/minggang/angang/jiagang
    tiles: list[str]          # 整副(含被鸣的那张)
    called: str               # 被鸣/被杠的那张
    from_seat: int            # 来源座位(暗杠/加杠记自己)


@dataclass
class PlayerPublic:
    """一家的公开信息(自己也有一份, 手牌单独放在 GameState.hand)."""

    seat: int
    score: int = 0
    discards: list[tuple[str, bool | None]] = field(default_factory=list)
    melds: list[Meld] = field(default_factory=list)
    riichi: bool = False
    wliichi: bool = False     # 两立直(ActionDiscardTile.is_wliqi)
    zhenting: bool = False    # 这张牌是否让人振听(ActionDiscardTile.zhenting)
    drew_unknown: int = 0     # 摸过几张看不到的牌(只计数)
    babei: int = 0            # 拔北张数(三麻; 拔北不算副露, 不破门清)


@dataclass
class GameState:
    """一手牌(或一场牌)的完整可见状态. apply() 喂 LiqiMessage."""

    my_seat: int | None = None
    account_id: int | None = None
    player_count: int = 4

    in_game: bool = False
    round_open: bool = False

    # --- 协议判"整场"起止(与 round_open 的"这一手"分开, 见 tuning.ROUND_BY_PROTOCOL)
    # 为什么单独记: 主循环等匹配成功看 match_started(ResAuthGame.is_game_start),
    # 判收局看 game_over(NotifyGameEndResult / NotifyGameTerminate). 只用
    # round_open 区分不了"这一手结束、下一手马上开"和"整场打完回大厅".
    match_started: bool = False   # 这一场真的开始了(匹配成功进对局)
    game_over: bool = False       # 这一场打完了(整场结束/被中断)
    game_over_reason: str = ""    # 结束原因: "" = 正常打完; 否则 = 中断原因
    last_round_end: str = ""      # 最近一手怎么结束的: hule/notile/liuju

    chang: int = 0            # 场风 0..3 = 东南西北
    ju: int = 0
    ben: int = 0              # 连庄棒
    liqibang: int = 0         # 立直棒
    wall: int | None = None   # 牌山余量(不含 dead wall 的口径随版本, 只做参考)
    scores: list[int] = field(default_factory=list)
    doras: list[str] = field(default_factory=list)

    hand: list[str] = field(default_factory=list)   # 暗手, 摸到的牌在末尾
    drawn: str | None = None                        # 刚摸的那张(未打时)
    melds: list[Meld] = field(default_factory=list)  # 我的副露
    players: dict[int, PlayerPublic] = field(default_factory=dict)

    my_waits: list[str] = field(default_factory=list)   # 协议给我的听牌(有才给)
    last_liqi: dict | None = None   # 最近一次"立直成立"回执(LiQiSuccess 原样)
    # 最近一次**给自己发牌**(ActionDealTile 且是我的座位)的本地时刻.
    # 摸到那张有渐入动画, 落定前读数低置信、reconcile 对不上、存帧全是
    # 动画噪声(2026-10-17 4 张 lowconf 帧全是它). 只存原始事实, 等多久
    # 由消费方算, 与 offered_at 同一个口径.
    last_self_draw_at: float = 0.0

    offered: list[dict] = field(default_factory=list)   # 当前可执行操作(原始)
    offered_step: int = -1
    # 收到这份操作列表的本地时刻 + 服务端给的思考时间(ms).
    # 时钟不在这里判: 状态机是纯逻辑(单测直接构造), 剩多少秒由 turn_time_left
    # 现算 —— 只存"原始事实", 不存推导结果, 免得时钟一冻就再也不准.
    offered_at: float = 0.0
    time_fixed: int = 0     # 固定思考时间(三麻实测 5000ms)
    time_add: int = 0       # 补时池(三麻实测 20000ms, 随回合消耗递减)
    my_zhenting: bool = False    # 我是否振听(同巡/舍牌/立直后振听)

    last_step: int = -1
    # 当前局的唯一身份 (chang, ju, ben). 真实牌谱实测: 每局 ActionNewRound
    # 会把 ActionPrototype.step 归零重开, 所以去重水位必须按"局"计,
    # 否则第二局及以后的 step 全都 <= 上一局的水位, 整局被当成重放丢掉.
    round_id: tuple[int, int, int] | None = None
    #: 已经开过几手(单调递增, 跨场不清零).
    #:
    #: `round_id` 是"这一手是哪一手"的**内容**, 而友人场每场都从东一局 (0,0,0)
    #: 开, 连打两场时内容会重复 —— 于是"协议有没有翻到下一手"没法只看它.
    #: 这个计数器只增不减, 是那种判断(见 tasks.PlayOneGameTask._hand_gap_arm)
    #: 唯一可靠的依据.
    round_seq: int = 0
    protocol_suspect: bool = False
    unhandled: dict[str, int] = field(default_factory=dict)
    last_event: str = ""

    # ------------------------------------------------------------------ API

    def apply(self, msg: LiqiMessage) -> bool:
        """喂一条报文. 返回是否被消费(False = 忽略/重复)."""
        try:
            return self._apply(msg)
        except Exception:  # noqa: BLE001 - 状态机永不把异常抛给主循环
            self.protocol_suspect = True
            self.last_event = f"apply-failed: {msg.method} {msg.action or ''}"
            return False

    def concealed(self) -> int:
        return len(self.hand)

    def meld_count(self) -> int:
        return len(self.melds)

    def melded_tiles(self) -> list[str]:
        """我的副露里的牌(扁平, 含被鸣的那张)."""
        out: list[str] = []
        for m in self.melds:
            out.extend(m.tiles)
        return out

    def concealed_legal(self) -> bool:
        """暗牌张数合法(与 table.expected_concealed_counts 等价, 见单测)."""
        n = self.concealed()
        return MIN_CONCEALED <= n <= MAX_CONCEALED and n % 3 in (1, 2)

    def is_my_turn(self) -> bool:
        """轮到我出牌: 暗牌 %3==2(摸完牌该我打). 副露后同样成立."""
        return self.round_open and self.concealed() % 3 == 2

    def seat_wind(self, seat: int) -> str:
        """座位风. seat_list 从东家起排, 所以 seat 0 = 东."""
        return f"{seat + 1}z"

    def field_wind(self) -> str:
        """场风(chang 0..3 = 东南西北)."""
        return f"{self.chang + 1}z"

    def offered_button_keys(self) -> list[str]:
        """当前可执行操作 -> 决策层按钮 key 列表."""
        keys = []
        for op in self.offered:
            key = OP_TYPE_TO_BUTTON.get(int(op.get("type", -1)))
            if key and key not in keys:
                keys.append(key)
        if CALL_BUTTONS & set(keys) and "tiaoguo" not in keys:
            keys.append("tiaoguo")   # 有鸣牌/荣和机会就一定有"跳过"
        return keys

    def protocol_says_my_turn(self) -> bool:
        """服务端明确说"轮到你打牌"没有(当前操作列表里有「打牌」).

        与 is_my_turn()(张数 % 3 == 2)是**两条独立判据**: 那个是从读出来的张数
        反推的, 漏认一张就推错; 这个直接来自服务端. 影子档拿它俩对比, 量的就是
        "视觉判的时机准不准"(见 tuning.NET_TURN_CLOCK).
        """
        return any(int(op.get("type", -1)) == OP_TYPE_DISCARD
                   for op in self.offered if isinstance(op, dict))

    def turn_budget(self) -> float:
        """服务端给这一动多少思考时间(秒). 0 = 协议没给/这份操作不是给我的."""
        total = max(0, int(self.time_fixed or 0)) + max(0, int(self.time_add or 0))
        return total / 1000.0 if total else 0.0

    def turn_time_left(self, now: float | None = None) -> float | None:
        """这一动还剩多少秒(负 = 已经超了). None = 协议没给这个信息.

        计时从"收到这份操作列表"起算: 服务端的 `time_add` 是**剩余补时池**,
        已经包含此前各巡的消耗, 所以本地只需要算这一段.
        """
        budget = self.turn_budget()
        if budget <= 0 or self.offered_at <= 0:
            return None
        return budget - ((time.time() if now is None else float(now))
                         - self.offered_at)

    def meld_options(self) -> list[dict]:
        """当前可鸣的牌(吃/碰/大明杠), 带协议给的具体组合.

        `combination` 的真实形状(2026-10-16 spool 实测):
            吃   type=2  ["1s|3s"]      —— 每一项是**一种**吃法, "|" 分隔要用的牌
            碰   type=3  ["3z|3z"]
            立直 type=7  ["9s", "6p"]   —— 这个 type 的每一项是"能打的牌", 不是组合
        所以只对 2/3/5 解析组合; 一个候选 = 一组要用的牌(**不含**被鸣的那张,
        吃的两项就是那两张搭子). 没给组合就整条丢掉: 估值没有输入不该出声.
        """
        out: list[dict] = []
        for op in self.offered:
            if not isinstance(op, dict):
                continue
            kind = OP_TYPE_TO_MELD.get(int(op.get("type", -1)))
            if kind is None:
                continue
            combos: list[list[str]] = []
            for item in (op.get("combination") or []):
                tiles = [t for t in str(item).split("|") if is_tile(t)]
                if tiles:
                    combos.append(tiles)
            if combos:
                out.append({"kind": kind, "options": combos})
        return out

    def riichi_seats(self) -> list[int]:
        """已立直的座位(升序). 对手立直是防守决策的第一信息, 必须能看到."""
        return sorted(s for s, p in self.players.items() if p.riichi)

    def opponent_riichi(self) -> list[int]:
        """对手里已立直的座位(不含我)."""
        return [s for s in self.riichi_seats()
                if self.my_seat is None or s != self.my_seat]

    def decide_inputs(self, *, meld_policy: str = "peng",
                      allow_riichi: bool = True) -> dict:
        """组装 src.mahjong.table.decide_turn 的入参.

        网络读数没有"置信度"概念: 按钮一律 1.0, blurry/missing_slots 恒空.
        注意手牌顺序是协议顺序, 不等于屏幕顺序 -- 点击坐标映射是接线期
        的事(见 docs/ARCHITECTURE.md 第七节待办).
        """
        buttons = {key: 1.0 for key in self.offered_button_keys()}
        return {
            "tiles": list(self.hand),
            "buttons": buttons,
            "player_count": self.player_count,
            "meld_policy": meld_policy,
            "allow_riichi": allow_riichi,
            "melded": self.melded_tiles(),
            "blurry_indices": [],
            "missing_slots": 0,
            "blurry_draw": False,
            "allowed_indices": None,
            "button_centers": None,
        }

    def snapshot(self) -> dict:
        """紧凑摘要(日志/演示用).

        立直信息必须在这里露出来: 影子档只打 status/note, 决策层也只吃
        decide_turn 的既有入参 -- 不露出来就等于"解析了但没人看得见".
        """
        return {
            "my_seat": self.my_seat,
            "player_count": self.player_count,
            # 牌局起止(协议优先判定用, 见 tuning.ROUND_BY_PROTOCOL):
            # match_started = 这一场开了; round_open = 这一手在打;
            # game_over = 这一场完了(原因在 game_over_reason).
            "match_started": self.match_started,
            "round_open": self.round_open,
            "game_over": self.game_over,
            "game_over_reason": self.game_over_reason,
            "last_round_end": self.last_round_end,
            "concealed": self.concealed(),
            "hand": list(self.hand),
            "drawn": self.drawn,
            "melds": [f"{m.kind}:{'+'.join(m.tiles)}" for m in self.melds],
            "doras": list(self.doras),
            "wall": self.wall,
            "scores": list(self.scores),
            "field_wind": self.field_wind(),
            "offered": self.offered_button_keys(),
            # "轮到我"的两条判据一起露出来, 日志里一眼能看出哪里对不上:
            # proto_my_turn = 服务端的权威说法(见 protocol_says_my_turn);
            # can_discard   = 视觉口径的推导(张数 % 3 == 2, 与 is_my_turn 同源).
            "proto_my_turn": self.protocol_says_my_turn(),
            "can_discard": self.is_my_turn(),
            # 思考时间的**原始值**(ms), 剩多少秒不在这里算 —— snapshot 要可复现.
            "time_fixed": self.time_fixed,
            "time_add": self.time_add,
            "meld_options": [o["kind"] for o in self.meld_options()],
            "liqibang": self.liqibang,
            "riichi": self.riichi_seats(),
            "opponent_riichi": self.opponent_riichi(),
            "waits": list(self.my_waits),
            "zhenting": self.my_zhenting,
            "discards": {
                s: [(t, mq) for t, mq in p.discards]
                for s, p in sorted(self.players.items())
            },
            "players": {
                s: {
                    "riichi": p.riichi,
                    "wliichi": p.wliichi,
                    "zhenting": p.zhenting,
                    "babei": p.babei,
                    "melds": [f"{m.kind}:{'+'.join(m.tiles)}" for m in p.melds],
                    "discards": len(p.discards),
                    "drew_unknown": p.drew_unknown,
                }
                for s, p in sorted(self.players.items())
            },
        }

    # -------------------------------------------------------------- dispatch

    def _apply(self, msg: LiqiMessage) -> bool:
        if msg.method == liqi.ACTION_METHOD:
            # data 为 None = 解码失败(丢); data 为 {} 是合法报文 --
            # 例如自己拔北的 ActionBaBei, proto3 把 seat=0 省略后就是空表.
            if not msg.action or msg.data is None:
                return False
            if not self._accept_step(msg.action, msg.data, msg.step):
                return False    # syncGame 重放/重复投递
            self._apply_action(msg.action, msg.data, msg.step)
            if msg.step > self.last_step:
                self.last_step = msg.step    # 水位: 本局见过的最新高步
            return True
        if msg.method == ".lq.FastTest.authGame":
            return self._auth_game(msg)
        if msg.method == ".lq.NotifyGameEndResult":
            self.in_game = False
            self.round_open = False
            self.game_over = True
            self.game_over_reason = ""
            self.last_event = "NotifyGameEndResult"
            return True
        if msg.method == ".lq.NotifyGameTerminate":
            # 被踢 / 断线超时 / 服务端异常终止: 这一场不会再有后续事件了.
            # 和不处理它的区别: 主循环会一直等本局的后续信号直到 GAME_TIMEOUT.
            data = msg.data if isinstance(msg.data, dict) else {}
            self.in_game = False
            self.round_open = False
            self.game_over = True
            self.game_over_reason = str(data.get("reason") or "terminated")
            self.last_event = f"NotifyGameTerminate reason={self.game_over_reason}"
            return True
        if msg.method in (".lq.FastTest.enterGame", ".lq.FastTest.syncGame"):
            return self._game_restore(msg)
        return False

    def _apply_action(self, name: str, data: dict, step: int) -> None:
        handler = _ACTION_HANDLERS.get(name)
        if handler is None:
            self._unhandled(name)
            return
        self.last_event = name
        handler(self, data, step)

    def _accept_step(self, action: str, data: dict, step: int) -> bool:
        """按"局"去重: 返回 False 表示这条是已处理过的旧事件(重放/重复).

        真实牌谱实测的关键事实: ActionPrototype.step 每局从 0 重开(开局的
        ActionNewRound 把计数器归零). 若整场只用一个水位, 第二局的 step 会
        全部 <= 上一局的水位而被当成 syncGame 重放丢掉 -- 整局状态全空.
        局身份 =(chang, ju, ben), 换局即重置水位; 同局重放(rid 相同)仍被
        水位挡住, 不会双重应用.
        """
        if action == "ActionNewRound":
            rid = (int(data.get("chang", 0) or 0),
                   int(data.get("ju", 0) or 0),
                   int(data.get("ben", 0) or 0))
            if rid != self.round_id:
                self.round_id = rid
                self.last_step = -1      # 新一局: 水位重置
                self.round_seq += 1      # "协议翻到下一手了"的唯一可靠依据
        return not (step > 0 and step <= self.last_step)

    def _unhandled(self, name: str) -> None:
        self.unhandled[name] = self.unhandled.get(name, 0) + 1
        if name not in ("ActionMJStart",):
            # 新版本协议出现没建模的动作: 不抛, 标记可疑让上层知道
            self.protocol_suspect = True

    def _suspect(self, why: str) -> None:
        self.protocol_suspect = True
        self.last_event = f"suspect: {why}"

    # ------------------------------------------------------------- handlers

    def _auth_game(self, msg: LiqiMessage) -> bool:
        data = msg.data or {}
        if msg.msg_type == MsgType.REQ:
            aid = data.get("account_id")
            if aid is not None:
                self.account_id = int(aid)
            return True
        if msg.msg_type != MsgType.RES:
            return False
        seat_list = data.get("seat_list") or []
        if not seat_list:
            return False
        self.player_count = len(seat_list)
        if self.account_id is not None and self.account_id in seat_list:
            self.my_seat = seat_list.index(self.account_id)
        self.in_game = True
        # is_game_start = 这一场真的开始了(区别于观战/回放接入). 主循环拿它
        # 代替"等对局开始页横幅"这个视觉判定(见 tuning.ROUND_BY_PROTOCOL).
        if bool(data.get("is_game_start", False)):
            self.match_started = True
        # 新的一场: 清掉上一场的结束标志(连打时同一个 GameState 会跨场复用).
        self.game_over = False
        self.game_over_reason = ""
        # **新的一场必须把 step 水位也归零**(2026-10-18 13:06 实跑, 整场不出牌):
        # `_accept_step` 只在 `ActionNewRound` 的 (chang, ju, ben) 变化时才重置
        # 水位, 而"友人场 一局"每场都从**东一局 (0,0,0)** 开: 上一场打到 step 90
        # 收局, 这一场的 ActionNewRound(step=1) 被当成 syncGame 重放丢弃 -> 之后
        # 整场每条事件都 <= 90 全被吞 -> round_open 永远停在上一场的 False ->
        # `decide_turn` 的收局守卫整场拒绝出牌(视觉手牌照常更新, 所以画面上
        # 像"程序在发呆", 日志里每轮都是"协议说这一手已经结束").
        # authGame 是"新一场"的权威信号; 断线重连也会重发它, 重置后紧跟的
        # syncGame 会从 step 0 重放整手事件, 状态重建结果一致(幂等).
        self.round_id = None
        self.last_step = -1
        self.last_event = f"authGame seat_list={seat_list}"
        return True

    def _action_mj_start(self, data: dict, step: int) -> None:
        self.in_game = True

    def _new_round(self, data: dict, step: int) -> None:
        tiles = data.get("tiles") or []
        if not all(is_tile(t) for t in tiles):
            self._suspect(f"new_round tiles invalid: {tiles!r}")
            return      # 不拿垃圾污染手牌: 上层见 suspect 回退视觉
        self.hand = list(tiles)
        # 庄家第一巡开局就是 14 张(配牌 13 + 第一摸, 一次给全), 但客户端**整手
        # 排序**, 摸的那张插进它的排序位, 不留独立槽(2026-10-18 18:56 局实锤:
        # 摸 6m 排在第 3 位, 出牌校验复读到的 13 张含 6m; 点"1z"(按摸牌钉最右
        # 铺的下标)实际打出 9s —— 整体错一格). 旧证据"列表末位 2s 在最右"与
        # 此不矛盾: 字牌排序本来就在最末, 两种摆法看起来一样. 所以这里**不记**
        # drawn, 对账层按整手排序铺(tiles.screen_order drawn=None).
        self.drawn = None
        self.melds = []
        self.offered = []
        self.my_waits = []
        self.my_zhenting = False
        self.last_liqi = None        # 新一局: 立直相关一起清零
        self.last_self_draw_at = 0.0
        # 注意不在这里设 last_step: 水位由 _apply 统一推进(见过即水位),
        # syncGame 重放本手旧事件(step 更小)自然被挡, 不搞第二套基线.
        scores = data.get("scores") or []
        if scores:
            self.scores = [int(s) for s in scores]
            self.player_count = len(scores)
        self.players = {i: PlayerPublic(seat=i, score=self.scores[i])
                        for i in range(self.player_count)}
        self.doras = list(data.get("doras") or [])
        self.chang = int(data.get("chang", 0) or 0)
        self.ju = int(data.get("ju", 0) or 0)
        self.ben = int(data.get("ben", 0) or 0)
        self.liqibang = int(data.get("liqibang", 0) or 0)
        wall = data.get("left_tile_count")
        self.wall = int(wall) if wall is not None else None
        self.round_open = True
        self.in_game = True
        # 新的一手开了 -> 这一场显然还在打. 防的是"上一场没收到
        # NotifyGameEndResult(spool 断了)然后又续上同一份文件"的残留标志.
        self.game_over = False
        self.game_over_reason = ""
        self.match_started = True
        # 新手牌解析成功 = 协议这一手是通的: 清掉上一手的 suspect.
        # 拿不到合法手牌时上面已提前 return, suspect 不会被误清.
        self.protocol_suspect = False
        # 开局配牌就带可执行操作(打牌 / 三麻的拔北): 不能丢, 否则手里有北
        # 也不会去拔(决策层看不到按钮).
        self._take_offered(data)

    def _deal_tile(self, data: dict, step: int) -> None:
        seat = int(data.get("seat", 0) or 0)
        tile = data.get("tile") or ""
        wall = data.get("left_tile_count")
        if wall is not None:
            self.wall = int(wall)
        self._sync_doras(data)
        self._sync_scores(data)
        self._sync_liqibang(data)
        mine = (seat == self.my_seat) if self.my_seat is not None else bool(tile)
        if mine and is_tile(tile):
            # 自己的摸牌: 协议把 tile 只发给本人, 这是"是不是我"的判据
            self.hand.append(tile)
            self.drawn = tile
            # 渐入动画计时起点(见 last_self_draw_at 注释)
            self.last_self_draw_at = time.time()
        else:
            player = self._player(seat)
            if player is not None:
                player.drew_unknown += 1
        # LiQiSuccess: 立直"成立"回执(扣 1000 点 + 立直棒 +1). 摸牌时随牌下发,
        # 是立直最权威的一条线索 —— 漏了它对手立直就完全看不见.
        # 注意这里**不能**拿外层摸牌的 seat 兜底: 回执是在下一家摸牌时下发的,
        # 外层 seat 是摸牌的人, 不是立直的人.
        self._apply_liqi_success(data.get("liqi"))
        self._sync_waits(data, mine)
        self._take_offered(data)

    def _discard_tile(self, data: dict, step: int) -> None:
        seat = int(data.get("seat", 0) or 0)
        tile = data.get("tile") or ""
        moqie = bool(data.get("moqie", False))
        is_liqi = bool(data.get("is_liqi", False))
        is_wliqi = bool(data.get("is_wliqi", False))
        if not is_tile(tile):
            self._suspect(f"discard tile invalid: {tile!r}")
            return
        self._sync_doras(data)
        self._sync_scores(data)
        self._sync_liqibang(data)
        player = self._player(seat)
        if self.my_seat is not None and seat == self.my_seat:
            if not self._remove_from_hand([tile]):
                self._suspect(f"discard {tile} not in hand {self.hand}")
            self.drawn = None
            self.my_waits = []       # 打完牌, 上一巡的听牌信息作废
        if player is not None:
            # 自己的出牌也记河: 完整牌局模型不分你我
            player.discards.append((tile, moqie))
            if is_liqi or is_wliqi:
                player.riichi = True
                player.wliichi = player.wliichi or is_wliqi
            if "zhenting" in data:
                player.zhenting = bool(data.get("zhenting"))
        self._sync_waits(data, self._is_mine(seat))
        self._take_offered(data)

    def _reveal_tile(self, data: dict, step: int) -> None:
        """立直宣言牌的公开(ActionRevealTile).

        字段与 ActionDiscardTile 同构(is_liqi/is_wliqi/moqie/tile/tingpais/
        zhenting/liqibang), 区别是它**只讲立直这件事**, 不带 doras.
        处理原则: 以"立直标记"为主; 宣言牌只有在还没进过河时才补记一张,
        免得与正常的 ActionDiscardTile 重复记河.
        """
        seat = int(data.get("seat", 0) or 0)
        tile = data.get("tile") or ""
        moqie = bool(data.get("moqie", False))
        is_liqi = bool(data.get("is_liqi", False))
        is_wliqi = bool(data.get("is_wliqi", False))
        self._sync_scores(data)
        self._sync_liqibang(data)
        player = self._player(seat)
        if player is not None:
            if is_liqi or is_wliqi or "liqibang" in data:
                player.riichi = True
            if is_wliqi:
                player.wliichi = True
            if "zhenting" in data:
                player.zhenting = bool(data.get("zhenting"))
            if is_tile(tile) and not (player.discards
                                      and player.discards[-1][0] == tile):
                player.discards.append((tile, moqie))
        if self._is_mine(seat) and is_tile(tile):
            self._remove_from_hand([tile])
            self.drawn = None
        self._sync_waits(data, self._is_mine(seat))
        self._take_offered(data)

    def _chi_peng_gang(self, data: dict, step: int) -> None:
        seat = int(data.get("seat", 0) or 0)
        # 这里**缺键就是 0**, 而 0 是 CPG_CHI(吃) —— 这是合法报文, 不是协议漂移.
        # 绝不能写成 `data.get("type", -1) or 0`: 缺键时先拿到 -1, 而 -1 是真值,
        # 后面那个 `or 0` 根本拦不住, 于是"吃"被当成未建模动作 -> `_unhandled`
        # -> protocol_suspect 置真 -> **整手牌**的协议读数全废(复位要等下一局开局).
        # 2026-10-17 真机实测代价: 第二场東1局第 9 手别人一个吃, 那一手 111 轮
        # 全程 net 空、纯视觉在打.
        type_ = int(data.get("type") or 0)
        tiles = [t for t in (data.get("tiles") or []) if is_tile(t)]
        froms = [int(f or 0) for f in (data.get("froms") or [])]
        if not tiles or len(tiles) != len(froms):
            self._suspect(f"chi_peng_gang malformed: {data!r}")
            return
        kind = {CPG_CHI: "chi", CPG_PENG: "peng", CPG_MINGGANG: "minggang"}.get(type_)
        if kind is None:
            self._unhandled(f"ActionChiPengGang type={type_}")
            return
        mine = self.my_seat is not None and seat == self.my_seat
        called_idx = None
        if mine and froms:
            # 被鸣的那张: 来源不是我的那张
            called_idx = next((i for i, f in enumerate(froms) if f != self.my_seat), None)
        elif froms:
            # 他人的副露: 调用牌 = 来源不是鸣牌者本人的那张
            called_idx = next((i for i, f in enumerate(froms) if f != seat), None)
        called = tiles[called_idx] if called_idx is not None else tiles[0]
        from_seat = froms[called_idx] if called_idx is not None else -1
        my_tiles = [t for i, t in enumerate(tiles) if i != called_idx]
        meld = Meld(kind=kind, tiles=list(tiles), called=called, from_seat=from_seat)
        if mine:
            if not self._remove_from_hand(my_tiles):
                self._suspect(f"{kind} tiles {my_tiles} not in hand {self.hand}")
            self.melds.append(meld)
            self.drawn = None
        else:
            player = self._player(seat)
            if player is not None:
                player.melds.append(meld)
        self._take_offered(data)

    def _an_gang_add_gang(self, data: dict, step: int) -> None:
        seat = int(data.get("seat", 0) or 0)
        # 同 `_chi_peng_gang`: proto3 省略 int 字段 == 0. 这里的 0 不是合法的
        # 加杠/暗杠(AAG_JIAGANG/AAG_ANGANG = 2/3), 所以缺键仍会走 `_unhandled`
        # —— 那是"真读不懂", 不是被 -1 骗出来的假警报.
        type_ = int(data.get("type") or 0)
        tile = data.get("tiles") or ""
        if not is_tile(tile):
            self._suspect(f"an_gang_add_gang tile invalid: {tile!r}")
            return
        self._sync_doras(data)
        kind = {AAG_JIAGANG: "jiagang", AAG_ANGANG: "angang"}.get(type_)
        if kind is None:
            self._unhandled(f"ActionAnGangAddGang type={type_}")
            return
        mine = self.my_seat is not None and seat == self.my_seat
        if mine:
            if kind == "jiagang":
                for m in self.melds:
                    if m.kind == "peng" and normalize(m.tiles[0]) == normalize(tile):
                        m.kind = "jiagang"
                        m.tiles.append(tile)
                        break
                else:
                    self._suspect(f"jiagang {tile} without peng meld")
                self._remove_from_hand([tile])
            else:
                if not self._remove_from_hand([tile] * 4):
                    self._suspect(f"angang {tile}x4 not in hand {self.hand}")
                self.melds.append(Meld(kind="angang", tiles=[tile] * 4,
                                       called=tile, from_seat=self.my_seat or 0))
            self.drawn = None
        else:
            player = self._player(seat)
            if player is not None:
                player.melds.append(Meld(kind=kind, tiles=[tile] * 4, called=tile,
                                         from_seat=seat))
        self._take_offered(data)

    def _ba_bei(self, data: dict, step: int) -> None:
        """拔北(三人麻将): 从暗手抽掉一张北, 随后 ActionDealTile 补一张.

        拔北不算副露、不破门清, 所以只从暗手移除 + 计数, 不进 self.melds
        (与 table.py 的 babei 语义一致: 北留在手里也能当役牌, 拔不拔由策略定).
        报文里没有 tile 字段 -- 三麻只有"北"(4z)可以拔.
        """
        seat = int(data.get("seat", 0) or 0)
        self._sync_doras(data)
        mine = self.my_seat is None or seat == self.my_seat
        if mine:
            if not self._remove_from_hand(["4z"]):
                self._suspect(f"babei but no north held: hand={self.hand}")
            self.drawn = None            # 拔北后进入"等补牌"状态
        player = self._player(seat)
        if player is not None:
            player.babei += 1
        self._take_offered(data)

    def _hule(self, data: dict, step: int) -> None:
        self.round_open = False
        self.last_round_end = "hule"
        self._sync_scores(data)
        self._sync_liqibang(data)
        hules = data.get("hules") or []
        seats = []
        for info in hules:
            if not isinstance(info, dict):
                continue
            # seat 是 proto3 字段: 和牌者是 seat 0 时会被省略, 不能因此跳过
            seat = int(info.get("seat", 0) or 0)
            seats.append(seat)
            player = self._player(seat)
            if player is not None:
                # HuleInfo.liqi = 和牌者的立直标记(交叉校验第四线索);
                # 有立直才可能有裏ドラ, 这里一并留着.
                if bool(info.get("liqi", False)):
                    player.riichi = True
            self.my_waits = [] if self._is_mine(seat) else self.my_waits
        if seats:
            mine = [s for s in seats if self._is_mine(s)]
            self.last_event = (
                f"hule seats={seats} mine={mine}" if mine
                else f"hule seats={seats}")

    def _no_tile(self, data: dict, step: int) -> None:
        self.round_open = False
        self.last_round_end = "notile"
        scores = data.get("scores") or []
        if scores and isinstance(scores[0], dict):
            self._sync_scores(scores[0])

    def _liu_ju(self, data: dict, step: int) -> None:
        self.round_open = False
        self.last_round_end = "liuju"

    def _new_dora(self, data: dict, step: int) -> None:
        self._sync_doras(data)

    def _benign(self, data: dict, step: int) -> None:
        """与本项目规则无关的动作(川麻换牌/定缺/血战等): 只吃掉, 不动状态.

        走这里而不是 `_unhandled`, 是因为 `_unhandled` 会把 protocol_suspect
        置真 -> 上层判定桥不可用 -> 整局静默回退视觉. 这些动作在日麻里不该
        出现, 但万一出现也绝不能让它把整条链路判死.
        """
        if data:
            self._sync_scores(data)
            self._sync_liqibang(data)
            self._sync_doras(data)

    # -------------------------------------------------------------- helpers

    def _player(self, seat: int) -> PlayerPublic | None:
        if seat not in self.players:
            if not 0 <= seat < max(self.player_count, 4):
                return None
            self.players[seat] = PlayerPublic(seat=seat)
        return self.players[seat]

    def _is_mine(self, seat: int) -> bool:
        """seat 是不是我. my_seat 未知时不敢认(宁可漏判)."""
        return self.my_seat is not None and int(seat) == self.my_seat

    def _apply_liqi_success(self, liqi) -> None:
        """LiQiSuccess -> 立直成立: 记玩家立直 + 立直棒 + 分数.

        这是"立直宣言牌"之外的独立线索: 摸牌/杠后补牌时随牌下发,
        带 seat(谁立的)、score(通常 -1000)、liqibang(桌上棒数)、failed。
        `failed=True` 表示立直被取消(宣言牌被荣和), 这时**不**标记立直.

        `seat` 缺键时按 **0** 处理, 这是 proto3 语义而不是"随便兜底":
        报文里 `liqi` 是**下一家摸牌**时下发的, 外层 `ActionDealTile.seat`
        是摸牌的人、不是立直的人, 拿它兜底会把立直记错座位.
        真 spool 两个样本(2026-10-16 idx421 / 2026-10-18)都是
        `liqi={'score': 24000, 'liqibang': 1}` + 外层 seat=1, 而立直的是
        seat 0 —— 缺键恰好等于 0.
        """
        if not isinstance(liqi, dict) or not liqi:
            return
        self.last_liqi = dict(liqi)
        if bool(liqi.get("failed", False)):
            return
        seat = int(liqi.get("seat", 0) or 0)
        player = self._player(seat)
        if player is not None:
            player.riichi = True
        if liqi.get("liqibang") is not None:
            try:
                self.liqibang = max(self.liqibang, int(liqi["liqibang"]))
            except (TypeError, ValueError):
                pass
        self.last_event = f"liqi established seat={seat}"

    def _sync_liqibang(self, data: dict) -> None:
        """立直棒只增不减(和牌时由分数一起结算, 不必在这里减)."""
        if "liqibang" not in data:
            return
        try:
            self.liqibang = max(self.liqibang, int(data.get("liqibang") or 0))
        except (TypeError, ValueError):
            pass

    def _sync_waits(self, data: dict, mine: bool) -> None:
        """协议给的听牌列表(只在自己的报文里出现): 存下来供日志/守卫参考.

        `tingpais` 有两种形状, **tile 的含义正好相反**, 不能一视同仁
        (2026-10-16 拿真 spool 复算时才发现):

        - `ActionDealTile.tingpais` = `repeated TingPaiDiscardInfo`
          `{tile: 要打哪张牌, zhenting, infos: [打了它之后听什么]}` ——
          这里的 `tile` 是**打牌候选**, 真正的听牌在 `infos[].tile`.
          实测(2026-10-16 spool idx61):
          `[{'tile':'6p','infos':[{'tile':'9s'}]}, {'tile':'9s','infos':[{'tile':'6p'}]}]`
          = 打 6p 听 9s / 打 9s 听 6p.
        - `ActionDiscardTile.tingpais` = `repeated TingPaiInfo`(扁平), 没有
          `infos`, 这时 `tile` 本身就是听牌.

        两种都以"所有 `infos[].tile` 的并集"为准; 没有 `infos` 才退化成
        取 `item.tile`.
        """
        if not mine:
            return
        tingpais = data.get("tingpais")
        if not isinstance(tingpais, list) or not tingpais:
            return
        tiles: list[str] = []
        for item in tingpais:
            if not isinstance(item, dict):
                continue
            infos = item.get("infos")
            if isinstance(infos, list) and infos:
                for info in infos:
                    if isinstance(info, dict) and is_tile(info.get("tile")):
                        tiles.append(info["tile"])
            elif is_tile(item.get("tile")):
                tiles.append(item["tile"])
        if tiles:
            self.my_waits = list(dict.fromkeys(tiles))   # 去重保序

    def _addressed_to_me(self, operation: dict) -> bool:
        seat = operation.get("seat")
        if seat is None:
            seat = 0     # proto3 省略 0 值: 缺席就是 seat 0(东家)
        return self.my_seat is None or int(seat) == self.my_seat

    def _take_offered(self, data: dict) -> None:
        """推进局面的事件要么带来"给我的"操作列表, 要么让旧选项作废.

        选项窗口是瞬时的: 我回应了(或没回应)之后, 下一个事件到来时旧
        选项一定已经过期 -- 不清掉就会在错误的时机(比如我已经摸牌了)
        还拿着一个碰按钮, 让决策层以为能鸣牌.
        """
        operation = data.get("operation")
        if isinstance(operation, dict) and self._addressed_to_me(operation):
            self.offered = list(operation.get("operation_list") or [])
            self.offered_step = self.last_step
            self.offered_at = time.time()
            # time_fixed + time_add = 这一动的思考时间预算(ms). 两个字段都可能
            # 缺席(不同版本/不同房型), 缺席就当 0 —— turn_time_left 会返回 None,
            # 调用方据此走老行为(不压缩节拍).
            self.time_fixed = int(operation.get("time_fixed") or 0)
            self.time_add = int(operation.get("time_add") or 0)
        else:
            self.offered = []
            # 选项过期 -> 时钟一起作废: 留着它会让"还剩几秒"在下一动之前一直是
            # 个陈旧的真值, 拿它压缩节拍就是在跟一个不存在的倒计时赛跑.
            self.offered_at = 0.0
            self.time_fixed = 0
            self.time_add = 0

    def _remove_from_hand(self, tiles: list[str]) -> bool:
        """按牌种从暗手移除. 网络读数精确到张, 不需要槽位概念."""
        for tile in tiles:
            kind = normalize(tile)
            for i, held in enumerate(self.hand):
                if normalize(held) == kind:
                    self.hand.pop(i)
                    break
            else:
                return False
        return True

    def _sync_doras(self, data: dict) -> None:
        doras = data.get("doras")
        if isinstance(doras, list) and len(doras) > len(self.doras):
            if all(is_tile(d) for d in doras):
                self.doras = list(doras)

    def _sync_scores(self, data: dict) -> None:
        scores = data.get("scores")
        if isinstance(scores, list) and scores and all(
                isinstance(s, int) for s in scores):
            self.scores = list(scores)

    # ------------------------------------------------------- game restore

    def _game_restore(self, msg: LiqiMessage) -> bool:
        """enterGame/syncGame 的响应: 断线重连/中途接入的整手恢复."""
        data = msg.data or {}
        restore = data.get("game_restore") or {}
        if not restore:
            return False
        snapshot = restore.get("snapshot") or {}
        actions = restore.get("actions") or []
        if snapshot:
            self._seed_snapshot(snapshot)
        replayed = 0
        for act in actions:
            name = act.get("name")
            raw = act.get("data")
            if not name or not raw:
                continue
            step = int(act.get("step", 0) or 0)
            try:
                inner = decode_nested_action(name, base64.b64decode(raw))
            except Exception:  # noqa: BLE001
                inner = None
            if inner is None:
                self._unhandled(f"restore:{name}")
                continue
            if not self._accept_step(name, inner, step):
                continue        # 同局重放: 按局水位挡住, 不双重应用
            self._apply_action(name, inner, step)
            if step > self.last_step:
                self.last_step = step
            replayed += 1
        self.last_event = f"game_restore actions={replayed}"
        self.in_game = True
        return True

    def _seed_snapshot(self, snap: dict) -> None:
        """GameSnapshot -> 当前状态(仅在没有可重放 actions 时用它兜底)."""
        if self.my_seat is None and snap.get("index_player") is not None:
            self.my_seat = int(snap["index_player"])
        self.chang = int(snap.get("chang", self.chang) or 0)
        self.ju = int(snap.get("ju", self.ju) or 0)
        self.ben = int(snap.get("ben", self.ben) or 0)
        self.liqibang = int(snap.get("liqibang", self.liqibang) or 0)
        # GameSnapshot.zhenting = 我这一巡是否振听(振听时不能荣和)
        self.my_zhenting = bool(snap.get("zhenting", self.my_zhenting))
        wall = snap.get("left_tile_count")
        if wall is not None:
            self.wall = int(wall)
        hands = snap.get("hands") or []
        if hands and all(is_tile(t) for t in hands):
            self.hand = list(hands)
            self.drawn = None
        doras = snap.get("doras") or []
        if doras and all(is_tile(d) for d in doras):
            self.doras = list(doras)
        players = snap.get("players") or []
        if players:
            self.player_count = len(players)
            self.scores = [int(p.get("score", 0) or 0) for p in players]
            self.players = {}
            for i, p in enumerate(players):
                pub = PlayerPublic(seat=i, score=int(p.get("score", 0) or 0))
                pub.discards = [(t, None) for t in (p.get("qipais") or [])
                                if is_tile(t)]
                # liqiposition: 立直宣言牌在弃牌里的下标, 未立直为 -1.
                # 旧代码写 bool(... or 0) —— 对未立直者的 -1 会判成 True
                # (整桌都"立直"了), 必须按 >=0 判. 字段缺席一律当未立直.
                pos = p.get("liqiposition")
                pub.riichi = pos is not None and int(pos) >= 0
                for ming in p.get("mings") or []:
                    tiles = [t for t in (ming.get("tile") or []) if is_tile(t)]
                    if not tiles:
                        continue
                    type_ = int(ming.get("type", 1) or 0)
                    kind = {0: "chi", 1: "peng", 2: "minggang",
                            3: "angang"}.get(type_, "peng")
                    if type_ == 1 and len(tiles) == 4:
                        kind = "jiagang"
                    froms = [int(f or 0) for f in (ming.get("from") or [])]
                    pub.melds.append(Meld(kind=kind, tiles=tiles, called=tiles[0],
                                          from_seat=froms[0] if froms else -1))
                self.players[i] = pub
            if self.my_seat is not None and self.my_seat in self.players:
                # 快照里"我"的副露就是 self.melds(暗手已由 hands 单独给)
                self.melds = list(self.players[self.my_seat].melds)
        self.round_open = True
        self.in_game = True
        # 快照/重连恢复: 有进行中的一手就说明这一场还没结束, 别残留上一场的
        # game_over(否则主循环会立刻判"收局").
        self.game_over = False
        self.game_over_reason = ""
        self.match_started = True


_ACTION_HANDLERS = {
    "ActionMJStart": GameState._action_mj_start,
    "ActionNewRound": GameState._new_round,
    "ActionDealTile": GameState._deal_tile,
    "ActionDiscardTile": GameState._discard_tile,
    "ActionRevealTile": GameState._reveal_tile,
    "ActionChiPengGang": GameState._chi_peng_gang,
    "ActionAnGangAddGang": GameState._an_gang_add_gang,
    "ActionBaBei": GameState._ba_bei,
    "ActionHule": GameState._hule,
    "ActionNoTile": GameState._no_tile,
    "ActionLiuJu": GameState._liu_ju,
    "ActionNewDora": GameState._new_dora,
}
# 与日麻无关但协议里存在的动作: 注册成 benign no-op, 免得被当成协议漂移
for _name in BENIGN_ACTIONS:
    _ACTION_HANDLERS[_name] = GameState._benign
