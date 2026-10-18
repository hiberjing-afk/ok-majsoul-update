"""用 jansou 自带的 EV 引擎做"进攻向"决策估价(纯逻辑, 可单测).

为什么要有这一层
----------------
我们之前只用了 jansou 的 `discard_evaluation()`, 而且**一个可选参数都没传**:

    brain.choose_discard() -> discard_evaluation(hand, player_count=3)

`visible` 默认 None = "牌河里一张牌都没看见", 于是进张数永远按满 4 张算
(虚高); 宝牌指示牌更是完全没参与. jansou 自己其实两个都支持:

    discard_evaluation(hand, visible=[34]int)      进张按可见牌扣减
    ev.discard_outlooks(counts, num_melds, draws,
                        visible=..., valuation=Valuation(...))
        每张候选牌: 到 N 张内听牌的概率 / 和牌的概率 / 和了的平均打点

`Valuation` 吃的是**宝牌指示牌**(不是真宝牌, jansou 自己换下一张), 加上
自风/场风/副露/赤 5, 于是"打点"这一半也由引擎算, 不需要我们再手写一套役种.

这一层只做**翻译与调用**: 把协议看到的牌河/副露/指示牌翻成 jansou 的两个
入参形状, 再把结果翻回我们的记号. 策略本身仍然全在 jansou 里.

几个必须记住的坑(都踩过, 都有测试)
--------------------------------
1. `parse_mpsz()` 返回的是 **list**. `Valuation.dora_indicators` 收 list 时
   `Valuation` 就变成**不可哈希**, 而 `ev._value()` 用它做缓存 key, 于是
   `discard_outlooks(valuation=...)` 直接抛 `TypeError: unhashable type`.
   所以进 jansou 之前一律 `tuple()`. (`Hand` 没事, 它自己会归一化.)
2. `visible` 的口径是"**手牌之外**看得见的牌": 牌河 + 全部副露(含我自己的) +
   已翻开的指示牌. 手牌本身**不要**放进去 —— `_remaining()` 是
   `4 - 手牌 - visible`, 放重了就扣两次.
3. `Valuation.melds` 必须给出**真副露**且个数与 `num_melds` 严格相等, 否则
   `_check_valuation` 直接抛. 而且副露是不是役牌对"打点"影响很大, 不能像
   牌效那条路那样拿 `_fake_melds` 的占位暗杠糊弄.

另外两条不是"参数写错", 是"接口选错", 一样耗时间:

4. jansou 有两个展望入口, **张数档要求相反**: `discard_outlooks` 要**摸牌后**
   的手牌(14/11/8/5/2 张), `ev.hand_outlook` 要**静止**手牌(13/10/7/4/1 张).
   传错档的报错是 `a hand with N melds cannot have M concealed tiles` ——
   看起来像"引擎不工作", 其实只是张数档不对. 本项目统一只用前者: 鸣牌前后
   都是"必须打一张"的时刻, 口径才一致(要整手牌的展望就取 `evaluate_discards`
   的第一名, 见 `best_line`).
5. 副露的**来源**在三人场与四人场映射不同: 出牌顺序是 s -> s+1, 所以
   offset 1 恒为下家, offset n-1 恒为上家, **只有四人场才有对家**(offset 2).
   按四人场去套三麻, 会把上家算成对家, 而 jansou 校验"吃只能来自上家" ——
   套错时每一副吃都造不出来, 整次估价静默变空.
6. **`player_count` 传错是"静默灾难"级**(2026-10-17 实测踩到): 三麻的牌种里
   没有 2m-8m(`kinds_in_play(3)` 剔掉), 所以把一局**四人场**按 3 传进去时,
   手里若有一张 4m, 那张牌在引擎眼里"永远不可能组成面子" → 整手牌**和不了**,
   `win` 与 `points` 一起恒为 0 —— **不抛任何异常**。而排序键是 `win * points`,
   全零时排序退化成按牌名升序, 于是"最优出牌"变成 `1p`(字典序最小). 影子里
   照常出数, 只是每个数字都无意义.
   守卫见 `outside_play()`: 手牌里出现该人数下不存在的牌 → 直接返回空列表 ——
   宁可不给结论, 也不给一个"按牌名排序"的假结论.


开销(2026-10-17 实测, 40 副随机三人手牌, 每手换牌不复用缓存):
    discard_evaluation 带 visible   0.10 ms/手
    discard_outlooks beam=1 draws=8  均值 2.9 ms, 最差 39.9 ms
    discard_outlooks beam=4 draws=12 均值 0.5 ms, 最差 5.3 ms
对局里一轮的预算是秒级, 所以每轮真跑得起.
"""

from __future__ import annotations

from dataclasses import dataclass

from jansou.analysis import ev
from jansou.analysis.efficiency import discard_evaluation
from jansou.core.hand import CallSource, Hand, Meld, MeldType
from jansou.core.notation import parse_mpsz
from jansou.core.rules import preset
from jansou.core.tiles import NUM_KINDS, TileKind, counts_by_kind, kinds_in_play

from src.mahjong.tiles import kind_name_to_tile, parse_tile, to_mpsz

#: 我们记号 -> jansou 的 0..33 下标. 直接从 TileKind 建, 不手抄表(手抄必漂).
_INDEX: dict[str, int] = {
    kind_name_to_tile(kind.name): int(kind) for kind in TileKind
}

#: 三人麻将的规则预设. jansou 里没有 "sanma" 这个名字, 雀魂三麻是
#: "mahjong-soul-3p"(有食断/无八连庄那一套), 别按四人预设算打点.
RULES_3P = preset("mahjong-soul-3p")
RULES_4P = preset("mahjong-soul")

#: 我们的副露 kind -> jansou 的 MeldType.
_MELD_TYPE = {
    "chi": MeldType.CHII,
    "peng": MeldType.PON,
    "minggang": MeldType.DAIMINKAN,
    "angang": MeldType.ANKAN,
    "jiagang": MeldType.SHOUMINKAN,
}

#: 相对座位 -> 鸣牌来源. **人数不同映射不同**: 出牌顺序是 s -> s+1,
#: 所以 offset 1 恒为下家, offset n-1 恒为上家, 四人场才有"对家"(offset 2).
#: 三麻按四人场去套会把上家算成对家 —— 而 jansou 校验"吃只能来自上家",
#: 套错的话每一副吃都造不出来, 整次估价静默变空(看起来像"引擎不工作").
def _source_for(offset: int, player_count: int):
    if player_count <= 0:
        return None
    offset %= player_count
    if offset == 1 % player_count:
        return CallSource.SHIMOCHA
    if offset == (player_count - 1) % player_count:
        return CallSource.KAMICHA
    if player_count == 4 and offset == 2:
        return CallSource.TOIMEN
    return None

#: 一次估价最多往前看几张摸牌. 三人场一手大约 18-20 巡, 16 已经够宽.
MAX_DRAWS = 16


@dataclass(frozen=True)
class EvDiscard:
    """打某一张牌的预期结果."""

    tile: str                 # 打这张
    shanten: int              # 打出后的向听数
    acceptance: int           # 打出后的进张数(已按 visible 扣减)
    ready: float              # draws 张内听牌的概率
    win: float                # draws 张内和牌的概率
    points: float             # 和了时的平均打点(0 = 没有 valuation / 无役)
    draws: int = 0            # 上面三条曲线对应多少张摸牌

    @property
    def ev(self) -> float:
        """期望打点 = 和牌率 x 和了时打点."""
        return self.win * self.points

    def summary(self) -> str:
        return (f"打{self.tile}: 向听{self.shanten} 进张{self.acceptance} "
                f"听牌率{self.ready:.2f} 和牌率{self.win:.2f} "
                f"打点{self.points:.0f} 期望{self.ev:.0f}")

    def as_record(self) -> dict:
        return {"tile": self.tile, "shanten": self.shanten,
                "acceptance": self.acceptance, "ready": round(self.ready, 4),
                "win": round(self.win, 4), "points": round(self.points, 1),
                "ev": round(self.ev, 1), "draws": self.draws}


@dataclass(frozen=True)
class HandValue:
    """一手和牌值多少(完整役种/翻/符/点数, 由 jansou.scoring 算)."""

    han: int
    fu: int
    points: int
    limit: str
    is_yakuman: bool
    yaku: tuple[tuple[str, int], ...] = ()

    def summary(self) -> str:
        names = "/".join(f"{n}{v}" for n, v in self.yaku) or "-"
        return (f"{self.han}翻{self.fu}符 {self.points}点 {self.limit} "
                f"[{names}]")

    def as_record(self) -> dict:
        return {"han": self.han, "fu": self.fu, "points": self.points,
                "limit": self.limit, "yakuman": self.is_yakuman,
                "yaku": [[n, v] for n, v in self.yaku]}


# ------------------------------------------------------------------ 翻译

def index_of(tile: str) -> int:
    """我们的记号 -> 0..33. 认不出来抛 ValueError(调用方负责兜底).

    **先过 parse_tile**: 协议里赤 5 写作 "0m"/"0p"/"0s", 而 `_INDEX` 的键是
    "5m"/"5p"/"5s".     不折叠的话赤 5 会被当成"不认识的牌"静默丢掉 —— 表现是
    "牌河里明明有三张赤 5, 进张还是按满 4 张算", 数字看着正常, 只是偏大.

    非字符串一律抛 ValueError(而不是让 parse_tile 抛 AttributeError):
    调用方的兜底统一接 `(KeyError, ValueError, TypeError)`.
    """
    if not isinstance(tile, str):
        raise ValueError(f"不是一张牌: {tile!r}")
    return _INDEX[parse_tile(tile)]


def counts_array(tiles) -> list[int]:
    """牌列表 -> 长度 34 的计数数组(手牌用)."""
    if not tiles:
        return [0] * NUM_KINDS
    return list(counts_by_kind(parse_mpsz(to_mpsz(list(tiles)))))


#: 各人数下"不存在"的牌种下标(三麻剔掉 2m-8m). 从 jansou 现算, 不手抄.
_OUT_OF_PLAY: dict[int, frozenset[int]] = {
    n: frozenset(int(k) for k in TileKind if k not in kinds_in_play(n))
    for n in (3, 4)
}


def outside_play(tiles, player_count: int = 3) -> list[str]:
    """手牌里"这个人数下根本不存在"的牌.

    非空 = 调用方给的两个事实自相矛盾(`player_count` 或手牌有一个是错的).
    **必须当成硬错误拦下来**: 三麻没有 2m-8m, 而引擎遇到这些牌不会报错 ——
    它只是把整手牌算成"永远和不了", `win`/`points` 一起恒为 0, 排序随即
    退化成按牌名升序。那种"照常出数但每个数字都无意义"的结果比报错危险得多。
    """
    bad = _OUT_OF_PLAY.get(player_count)
    if bad is None:
        return []
    out = []
    for tile in tiles or ():
        try:
            if index_of(tile) in bad:
                out.append(tile)
        except (KeyError, ValueError, TypeError):
            out.append(tile)        # 认不出来的牌也算"不该在这手牌里"
    return out


def has_win_signal(rows) -> bool:
    """这批候选里有没有"能赢"的信号.

    全是 0 说明引擎这手牌给不出可用结论(`win`/`points` 都不动) —— 影子日志
    应当如实标注, 否则会拿一个纯粹由牌名决定的"最优出牌"当真。
    """
    return any(row.win > 0.0 for row in rows or ())


def _meld_tiles(meld) -> list[str]:
    """副露的牌(扁平). 接受协议侧的 Meld 对象或裸字符串序列.

    拿不到可迭代的东西时返回空而不是抛: 这一层是给"网络层随便塞点什么都
    不能拖垮主循环"用的.
    """
    tiles = getattr(meld, "tiles", None)
    if tiles is None:
        tiles = meld
    if isinstance(tiles, str) or not hasattr(tiles, "__iter__"):
        return []
    return [t for t in tiles if isinstance(t, str)]


def visible_from(discards=(), melds=(), indicators=()) -> list[int]:
    """牌河 + 副露 + 已翻开的指示牌 -> 长度 34 的"手牌之外看得见"计数.

    **不要把手牌放进来**: jansou 的 `_remaining()` 是 `4 - 手牌 - visible`,
    手牌放重了会被扣两次, 进张数偏低(表现为"算出来的进张总是比实际少")。
    """
    out = [0] * NUM_KINDS
    for tile in discards or ():
        try:
            out[index_of(tile)] += 1
        except (KeyError, ValueError, TypeError):
            continue
    for meld in melds or ():
        for tile in _meld_tiles(meld):
            try:
                out[index_of(tile)] += 1
            except (KeyError, ValueError, TypeError):
                continue
    for tile in indicators or ():
        try:
            out[index_of(tile)] += 1
        except (KeyError, ValueError, TypeError):
            continue
    return out


def make_melds(melds, my_seat: int = 0, player_count: int = 3) -> tuple:
    """协议侧的副露 -> jansou 的 Meld 元组.

    jansou 对副露的校验很严(`_check_claim`): 吃/碰/大明杠/加杠都必须给出
    **被鸣的那张**与**来源**(吃还只能来自上家), 暗杠则三者全空. 造不出来的
    那一副直接跳过 —— 但调用方要清楚: 跳一副会让个数与 `num_melds` 对不上,
    jansou 随即抛 `_check_valuation`, 整次估价返回空. 宁可算不出来, 也不能
    按错的副露算打点(碰没碰役牌差好几翻).
    """
    out = []
    for meld in melds or ():
        try:
            mtype = _MELD_TYPE.get(getattr(meld, "kind", None))
            if mtype is None:
                continue
            tile_objs = tuple(parse_mpsz(to_mpsz(_meld_tiles(meld))))
            want = 4 if mtype in (MeldType.DAIMINKAN, MeldType.ANKAN,
                                  MeldType.SHOUMINKAN) else 3
            if len(tile_objs) != want:
                continue
            if mtype is MeldType.ANKAN:
                out.append(Meld(type=mtype, tiles=tile_objs))
                continue
            called_text = getattr(meld, "called", None)
            if not called_text:
                continue
            called = parse_mpsz(called_text)[0]
            if called not in tile_objs:
                continue
            from_seat = getattr(meld, "from_seat", None)
            # 负数 = 协议侧"来源不明"(见 state.py 的 -1 约定). 别让它参与取模:
            # -1 % 3 == 2 会被算成"上家", 于是一副来源不明的吃静悄悄变成合法吃.
            if not isinstance(from_seat, int) or from_seat < 0:
                continue
            source = _source_for(from_seat - my_seat, player_count)
            if source is None:
                continue
            if mtype is MeldType.SHOUMINKAN:
                # 加杠: tiles 前 3 张是原来的碰, 第 4 张是加进去的那张.
                out.append(Meld(type=mtype, tiles=tile_objs, called=called,
                                source=source, added=tile_objs[3]))
            else:
                out.append(Meld(type=mtype, tiles=tile_objs, called=called,
                                source=source))
        except Exception:  # noqa: BLE001 -- 一副造不出来不该拖垮整次估价
            continue
    return tuple(out)


def _wind(tile: str) -> ev.Wind:
    """我们记号的 1z-4z -> jansou 的风. 认不出来当东风(宁可少算符, 不算错役)."""
    try:
        num = int(str(tile)[0])
    except (TypeError, ValueError, IndexError):
        return ev.Wind.EAST
    return ev.Wind(max(0, min(3, num - 1)))


def valuation_of(doras=(), *, round_wind: str = "1z",
                 seat_wind: str | None = None, melds=(), player_count: int = 3):
    """建 jansou 的 Valuation.

    `doras` 是**宝牌指示牌**(协议下发的就是它), jansou 自己换下一张 —— 这里
    千万别先换成真宝牌, 否则宝牌会翻一位(看起来在工作, 结果是错的)。
    """
    rules = RULES_3P if player_count == 3 else RULES_4P
    return ev.Valuation(
        rules=rules,
        melds=tuple(melds or ()),
        dora_indicators=tuple(parse_mpsz(to_mpsz(list(doras or ())))),
        round_wind=_wind(round_wind),
        seat_wind=_wind(seat_wind or round_wind),
    )


def draws_from_wall(wall, player_count: int = 3) -> int:
    """牌山剩余 -> 我还能摸几张. 三家轮着摸, 所以是 wall / 人数."""
    try:
        left = int(wall)
    except (TypeError, ValueError):
        return 0
    if left <= 0:
        return 0
    return max(0, min(MAX_DRAWS, left // max(1, player_count)))


# ------------------------------------------------------------------ 估价

def evaluate_discards(hand, *, melds=(), visible=None, draws: int = 8,
                      player_count: int = 3, doras=(), round_wind: str = "1z",
                      seat_wind: str | None = None, beam: int = 1,
                      my_seat: int = 0, with_points: bool = True,
                      ) -> list[EvDiscard]:
    """每张可打的牌: 向听 / 进张 / 听牌率 / 和牌率 / 打点.

    hand 必须是**摸牌后**的手牌(暗牌 14/11/8/5/2 张). 返回按期望打点降序;
    没有打点时(见 `with_points`)退化成按和牌率降序.
    任何一步出错都返回空列表 —— 网络层的估价永远不能拖垮主循环.

    `hand` 里出现 `player_count` 下不存在的牌(三麻的 2m-8m)时**也返回空列表**:
    引擎不会为这个报错, 它只会把整手牌算成永远和不了, 然后把排序退化成
    "按牌名升序" —— 见 `outside_play()`。调用方拿到空列表 = 这一轮没有结论,
    好过拿到一条看起来正常、实则由字典序决定的"最优出牌"。

    调用方想知道"这份结论值不值得信", 用 `has_win_signal(rows)`: 全零的
    `win` 说明引擎这手牌给不出可用信号(结果已按牌效兜底, 但不要当真)。

    `with_points=False` 会**跳过打点估值**(不建 `Valuation`), 只算进张/听牌率/
    和牌率. 这不是省一点点: 2026-10-17 实测(已装 jansou-native, 每手换牌不复用
    缓存)同一批 60 副三人手牌, draws=8:

        不带打点: p50  1.8 ms   p90  20.8 ms   max 135 ms
        带打点  : p50 16.2 ms   p90 163.7 ms   max 727 ms

    差的那一个数量级全在"给每一种可能和牌的组合算役种/符/点数"上 —— 走的是
    jansou.scoring 的纯 Python 路径, 原生加速模块管不到. 主循环里同步跑
    700ms 是有风险的(会吃掉思考倒计时), 所以影子的默认档不带打点.

    打点在排序里只做**平局裁决**(见函数尾部的长注释): 胡牌率/进张是主键,
    打点量级不参与方向判断. 生产档(`EV_EVAL_POINTS=False`)根本不算它 ——
    走 `evaluate_efficiency`, 只看向听与真实进张.
    """
    try:
        # 先查"这手牌在这个人数下成不成立". 不查的话, 一张不存在的牌(比如三麻
        # 里的 4m)会让整手牌永远和不了 -> win/points 恒 0 -> 排序退化成按牌名
        # 升序, 而且**不抛异常**. 见 `outside_play()` 与 `has_win_signal()`.
        if outside_play(hand, player_count):
            return []
        meld_objs = make_melds(melds, my_seat=my_seat,
                               player_count=player_count)
        counts = counts_array(hand)
        if sum(counts) not in (2, 5, 8, 11, 14):
            return []
        seen = (list(visible) if visible is not None
                else [0] * NUM_KINDS)
        if len(seen) != NUM_KINDS:
            seen = [0] * NUM_KINDS
        # 牌山/可见牌不够时 jansou 会抛, 这里先自己夹一下, 别让它抛.
        unseen = sum(4 - counts[i] - min(seen[i], 4) for i in range(NUM_KINDS))
        horizon = max(0, min(int(draws), unseen, MAX_DRAWS))
        if horizon <= 0:
            return []
        val = (valuation_of(doras, round_wind=round_wind,
                            seat_wind=seat_wind, melds=meld_objs,
                            player_count=player_count)
               if with_points else None)
        # `max_distance` **必须显式给 None**: 它的默认值是 2, 而它比的是
        # `shanten_counts(这手牌)`. 起手 3 向听(三人场最常见的那种)会直接命中
        # `distance > max_distance` 的保护分支, 返回**全零**的 ready/win/points
        # 曲线 —— 排序随即退化成"按牌种名升序", 看起来照样有输出(每张都
        # 0.00/0.00/0), 只是完全没有意义. 这个坑不显式关掉就永远在.
        outlooks = ev.discard_outlooks(
            counts, len(meld_objs), horizon, visible=tuple(seen),
            player_count=player_count, beam=beam, max_distance=None,
            valuation=val)
        # 顺带把"扣了可见牌之后的进张"也取出来(生产路径要用的就是它).
        eff = {kind_name_to_tile(o.discard.name): o for o in
               discard_evaluation(_hand_of(hand, meld_objs),
                                  visible=list(seen),
                                  player_count=player_count)}
    except Exception:  # noqa: BLE001
        return []

    out: list[EvDiscard] = []
    for kind, outlook in outlooks.items():
        tile = kind_name_to_tile(kind.name)
        eff_row = eff.get(tile)
        pts_curve = outlook.points or ()
        points = (pts_curve[horizon] if len(pts_curve) > horizon else 0.0)
        out.append(EvDiscard(
            tile=tile,
            shanten=int(eff_row.shanten) if eff_row else -1,
            acceptance=int(eff_row.total_acceptance) if eff_row else 0,
            ready=float(outlook.ready[horizon]),
            win=float(outlook.win[horizon]),
            points=float(points),
            draws=horizon,
        ))
    # 有打点时按期望打点排; 没有打点时 ev 恒为 0, 自然落到和牌率那一档.
    #
    # **再往后两级是"防退化"的**, 不是锦上添花: 引擎在某些局面会给出一整列
    # 全零的 win/points(最典型的是"这手牌这个人数下和不了" —— 见 `outside_play`),
    # 那时只按 (-ev, -win, -points) 排就等于**按牌名升序**, 而牌名顺序跟牌效毫无
    # 关系(实测会把"最优出牌"算成字典序最小的 1p)。所以补上 ready 与进张:
    # 引擎没有信号时, 至少退回"牌效"这条久经验证的尺子, 而不是随机。
    #
    # **打点为什么在平局裁决位**: win/accept 全平时若落到牌名字典序, 排序
    # 就与牌效无关了(实测 6m 顶掉 9m). 打点排在这两者之后, 只在它们精确
    # 打平时说话 —— 这时它的小边际噪声伤不到方向, 却能分出发得出高下的
    # 那一手. 排序键:
    #
    #   胡牌率 -> 进张 -> 打点 -> 听牌率 -> 牌名
    #
    # 打点只做**平局裁决**(win 和 acceptance 都精确打平才说话) —— 原先
    # "打点当主键"的噪声(换指示牌方向反转 2~3 倍)全发生在主键位置的小边际
    # 上; 挪到裁决位, 只在树估给不出区分的低信号局面上出场, 噪声伤不到方向.
    # 注: 这一档只在 `EV_EVAL_POINTS=True` 时生效(默认档走纯牌效).
    out.sort(key=lambda d: (-d.win, -d.acceptance, -d.points,
                            -d.ready, d.tile))
    return out


def hand_value(hand, win_tile: str, *, melds=(), doras=(),
               round_wind: str = "1z", seat_wind: str | None = None,
               is_tsumo: bool = True, player_count: int = 3,
               my_seat: int = 0) -> HandValue | None:
    """这手牌和了值多少. 和不了(或无役)返回 None.

    hand 可以是 13 张档(和牌张还在别人手里/牌河里)也可以是 14 张档
    (自摸, 和牌张已在自己手上) —— 差几张由 (暗牌 + 3*副露) % 3 判.
    """
    try:
        meld_objs = make_melds(melds, my_seat=my_seat,
                               player_count=player_count)
        tiles = [t for t in hand if isinstance(t, str)]
        if (len(tiles) + 3 * len(meld_objs)) % 3 == 1:
            tiles.append(win_tile)          # 13 张档: 补上和牌张
        from jansou.scoring.score import score

        rules = RULES_3P if player_count == 3 else RULES_4P
        ctx = ev.WinContext(
            rules=rules,
            round_wind=_wind(round_wind),
            seat_wind=_wind(seat_wind or round_wind),
            is_tsumo=is_tsumo,
            dora_indicators=tuple(parse_mpsz(to_mpsz(list(doras or ())))),
        )
        res = score(Hand(tuple(parse_mpsz(to_mpsz(tiles))), meld_objs),
                    parse_mpsz(win_tile)[0], ctx)
    except Exception:  # noqa: BLE001
        return None
    return HandValue(
        han=int(res.han),
        fu=int(res.fu.total) if hasattr(res.fu, "total") else 0,
        points=int(res.payment.total),
        limit=res.limit.name,
        is_yakuman=bool(res.is_yakuman),
        yaku=tuple((y.yaku.name, int(y.value)) for y in res.yaku),
    )


def evaluate_efficiency(hand, *, melds=(), visible=None, draws: int = 0,
                        player_count: int = 3, my_seat: int = 0,
                        ) -> list[EvDiscard]:
    """纯牌效排序: 只认向听与进张, 不碰 EV 展望、不算打点.

    产出与 `evaluate_discards` **同一个类型**(EvDiscard), 只是 `ready`/`win`/
    `points` 恒为 0 —— 下游的排序/记录/取牌一行都不用改, 换一个函数就把
    "期望打点最大"换成"最快听牌", 两个档可以随时对调.

    为什么值得单独开一个: 三麻 14 张带 visible 实测, `discard_evaluation`
    **0.09 ms/次**, 而 `evaluate_discards` 带打点是 34 ms/次(项目自己那批
    60 副手牌的 p90 是 164 ms, max 727 ms). 只看进张时那几百倍是白付的,
    而**出牌决策是整个主循环里唯一必须每轮都跑的估价** —— 慢就意味着
    "倒计时危急就不算了"这类妥协, 以及跳帧.

    排序键 = (向听升序, 进张降序, 字牌优先, 牌名升序). 后两条只在**前两条
    完全打平**时才起作用, 牌效结论不受影响 —— 与 `brain._honors_first` 是
    同一条规矩(孤张先打字牌).

    入参口径与 `evaluate_discards` 完全一致(手牌/副露/visible/人数), 出错
    同样返回空列表.
    """
    try:
        if outside_play(hand, player_count):
            return []
        meld_objs = make_melds(melds, my_seat=my_seat,
                               player_count=player_count)
        counts = counts_array(hand)
        if sum(counts) not in (2, 5, 8, 11, 14):
            return []
        seen = (list(visible) if visible is not None else [0] * NUM_KINDS)
        if len(seen) != NUM_KINDS:
            seen = [0] * NUM_KINDS
        options = discard_evaluation(_hand_of(hand, meld_objs),
                                     visible=list(seen),
                                     player_count=player_count)
    except Exception:  # noqa: BLE001 - 估价永不拖垮主循环
        return []
    rows = [EvDiscard(tile=kind_name_to_tile(o.discard.name),
                      shanten=int(o.shanten),
                      acceptance=int(o.total_acceptance),
                      ready=0.0, win=0.0, points=0.0, draws=draws)
            for o in options]
    rows.sort(key=lambda r: (r.shanten, -r.acceptance,
                             0 if r.tile.endswith("z") else 1, r.tile))
    return rows


def best_line(hand, **kwargs) -> EvDiscard | None:
    """整手牌最好的一手(= evaluate_discards(...)[0]).

    鸣牌前后要对比的"这手能到什么程度", 指的就是它: 两边都是**必须打一张**
    的时刻(摸牌后 / 鸣牌后), 比的是同一件事, 不像 `ev.hand_outlook` 那样
    要求"静止手牌"(13/10/7/4/1 张) —— 两种张数档混着传必然报张数错, 而且
    报出来的错长得很像"引擎不工作".
    """
    rows = evaluate_discards(hand, **kwargs)
    return rows[0] if rows else None


# ---------------------------------------------------------------- 内部工具

def _hand_of(tiles, meld_objs=()) -> Hand:
    """构造 jansou 的 Hand(**用真副露**, 不再用占位暗杠).

    brain._hand_of 用的是 `_fake_melds` 占位暗杠 —— 那条路只算牌效, 副露内容
    无所谓; 但估价要看役种, 碰的是不是役牌直接决定打点, 占位就骗了自己.
    """
    return Hand(tuple(parse_mpsz(to_mpsz(list(tiles)))), tuple(meld_objs))
