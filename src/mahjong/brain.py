"""出牌决策: jansou 算牌效 + 随机性注入(纯逻辑, 可单测).

策略定位是**纯牌效进攻流**: 只看自己的手牌, 不看牌河, 不读对手, 不防守
(见 docs/STRATEGY.md). 这样做有两个理由:

- 像"认真但不精通的玩家", 行为上不引人注目;
- 实现简单, 而且**不需要识别牌河和对手手牌**, 感知层的工作量小一大截.

出牌由 jansou 的 `discard_evaluation()` 给出, 它返回所有可打的牌, 按
(打出后的向听数升序, 进张数降序, 牌种升序) 排好 -- 也就是 options[0] 最优.

随机性不是可选的装饰, 而是这个策略的核心伪装: 每次都打"计算机算出来的最优解"
本身就是最强的脚本特征. 所以:

    - 5% 打次优(进张第二多的那张) -- 2026-10-07 从 20% 下调: 用户指出
      乱打太频繁容易胡不了, "偶尔打错一两次"就够了
    - 5% 从所有可打的牌里随机挑一张
    - 剩下的打最优

注意 `player_count`: 我们的房间是三人麻将, 传 3.jansou 会据此把 2m-8m
从牌种里剔除, 进张数也会按三人麻将的剩余张数算.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from jansou.analysis.efficiency import discard_evaluation
from jansou.analysis.shanten import is_tenpai, shanten
from jansou.analysis.waits import waits
from jansou.core.hand import Hand
from jansou.core.notation import parse_mpsz

from src.mahjong.tiles import kind_name_to_tile, to_mpsz

# 打出的牌是怎么选出来的. 记进日志, 出问题时能回溯.
PICK_BEST = "best"
PICK_SECOND = "second"
PICK_RANDOM = "random"

# 次优率 / 随机率: **2026-10-08 用户拍板归零** —— "随机出牌可以删掉,
# 就按最优解出牌就行了". 历史轨迹: 2026-10-07 曾从 20% 收到 5%(20% 时
# 一手里常有三四张"看不懂的乱打", 直接胡不了); 实跑一晚后用户确认伪装
# 收益抵不过胡不了牌的损失, 彻底删掉. 机制保留(rate 参数还在), 便于
# 测试注入, 但生产路径恒为 best.
DEFAULT_SUBOPTIMAL_RATE = 0.0
DEFAULT_RANDOM_RATE = 0.0


@dataclass(frozen=True)
class DiscardPlan:
    """一次出牌的决策结果.

    tile:        要打的牌("3m")
    picked:      best / second / random -- 为什么选它
    shanten:     打出后的向听数
    acceptance:  打出后的进张数
    """

    tile: str
    picked: str
    shanten: int
    acceptance: int


# 占位暗杠用的记号. 内容对牌效没有任何影响(见 _fake_melds 的说明),
# 只需要是张"合法暗杠" -- jansou 构造 Meld 时会强校验张数与同种性.
_PLACEHOLDER_KAN = "1m"


def _fake_melds(meld_count: int) -> tuple:
    """造 meld_count 个占位副露.

    我们的策略是"纯牌效进攻流"(docs/STRATEGY.md): 只算自己的暗牌怎么凑快,
    不读牌河也不读对手. 副露具体是碰的还是吃的, 是哪几张牌, 对"还差几张
    听牌"没有任何影响 -- jansou 也只用到**个数**, 不看内容.

    所以这里用同一个占位暗杠重复 meld_count 次即可. 不能塞个空对象进去:
    Meld 是 frozen dataclass, 构造时就会校验"暗杠必须 4 张同种牌".
    """
    from jansou.core.hand import Meld, MeldType

    if meld_count <= 0:
        return ()
    kan = Meld(type=MeldType.ANKAN,
               tiles=tuple(parse_mpsz(_PLACEHOLDER_KAN * 4)))
    return tuple(kan for _ in range(meld_count))


def _hand_of(tiles: list[str], meld_count: int) -> Hand:
    """构造 jansou 的 Hand.

    **副露必须真的传进去**(2026-10-06 修): 早先这里把 meld_count 收下就丢,
    只按暗牌构造 Hand. 于是 jansou 以为"0 副露", 而我们塞给它 8 张暗牌,
    它当场抛 "a hand with 0 melds cannot have 7 concealed tiles".

    根因值得记一笔: 那条报错**不是**牌效算不动, 而是"牌河被误识成手牌"
    -- 和牌画面上雀魂把和了的手牌摊在画面下方, 牌面模型把那排牌读成了我们的
    手牌(实测 7/10 张). 修好这个之后非法张数会在构造 Hand 时直接抛
    ValueError, 决策层靠它把这类帧挡在外面(见 table.py 的 sane 判据).
    """
    spec = to_mpsz(tiles)
    if not spec:
        raise ValueError("手牌是空的")
    return Hand(concealed=parse_mpsz(spec), melds=_fake_melds(meld_count))
def shanten_of(tiles: list[str], meld_count: int = 0) -> int:
    """当前手牌的向听数(0 = 听牌)."""
    return shanten(_hand_of(tiles, meld_count))


def is_tenpai_hand(tiles: list[str], meld_count: int = 0) -> bool:
    """是否听牌."""
    return is_tenpai(_hand_of(tiles, meld_count))


def waits_of(tiles: list[str], player_count: int = 3) -> list[str]:
    """听哪些牌, 返回升序的记号列表."""
    return sorted(kind_name_to_tile(t.name) for t in waits(
        _hand_of(tiles, 0), player_count=player_count))


def _honors_first(options):
    """平分时字牌优先打: 对 jansou 的排序做一次稳定重排.

    jansou 按 (向听升序, 进张降序, 牌种升序) 排, 牌种升序里数牌(p/s)排在
    字牌(z)前面. 于是"孤张字牌 vs 孤张数牌"这种**完全平分**的局面(两张
    都不参与任何搭子, 向听和进张一字不差)永远优先打数牌、留白板 -- 2026-
    10-07 实跑(turn 47/57)用户看着别扭: 人类的直觉是孤张先打字牌.

    稳定排序保证同组内部的库排序原样保留, 只把"字牌"提到同分组的最前.
    只有向听和进张**都相同**才影响顺序, 牌效结论不受影响.
    """
    return sorted(
        options,
        key=lambda o: (o.shanten, -o.total_acceptance,
                       0 if kind_name_to_tile(o.discard.name).endswith("z") else 1),
    )


def _all_kinds(player_count: int) -> tuple[str, ...]:
    """该人数规则下的全部牌种(3P 万子只有 1m/9m)."""
    if player_count == 3:
        return ALL_TILES
    return (tuple(f"{n}m" for n in range(1, 10))
            + tuple(f"{n}{s}" for s in "ps" for n in range(1, 10))
            + tuple(f"{n}z" for n in range(1, 8)))


def _extension_ukeire(tiles13: list[str], meld_count: int,
                      player_count: int) -> int:
    """二阶进张: 13 张的手再摸一张(每种牌各试一次)后, 最好出牌的进张数之和.

    衡量的是"未来的发展潜力"而不只是当前一步. 一阶进张(打完后的直接进张)
    完全相同的两个候选, 二阶可以分开 -- 典型是 6-8-9 打 6 还是打 9:
    打 6 留 8-9 边张(只等 7), 打 9 留 6-8 嵌张(摸 5 -> 打 8 组成 5-6 两面,
    摸 7 直接完成) -- 一阶进张都是"7 共 4 张", 但二阶合计 846 vs 818
    (2026-10-09 实测), 嵌张的发展潜力更高, 人类的打法也是打 9.
    """
    total = 0
    for t in _all_kinds(player_count):
        try:
            opts = discard_evaluation(_hand_of(tiles13 + [t], meld_count),
                                      player_count=player_count)
        except ValueError:
            continue  # 第 5 张同种牌等凑不出的假设, 跳过
        if opts:
            total += opts[0].total_acceptance
    return total


# 平分裁决最多展开几张候选: 二阶进张每次约 3ms(27 种牌 x 0.11ms), 候选一多
# 就拖决策节奏. 平分组通常只有 2~3 张, 上限 4 张足够; 更长的平分组(一堆孤张)
# 本来就没有"发展潜力"可言, 保持原序即可.
MAX_EXTENSION_CANDIDATES = 4


def _extension_tiebreak(options, tiles: list[str], meld_count: int,
                        player_count: int, keep_kinds: frozenset[str] = frozenset()):
    """一阶(向听, 进张)完全平分的候选里, 按二阶进张(发展潜力)重排.

    只动"与第一名向听数和进张数都相同"的前缀组(它们在排序里必然连续地
    排在最前), 其余候选原样保留. 平分组按二阶进张降序稳定重排 -- 二阶也
    相同时保持 _honors_first 给出的顺序(字牌优先/牌种升序).

    keep_kinds 非空时(2026-10-09 東2局实锤后新增): 打掉这些种类会**拆散
    役牌对子**(meld_policy=peng 下役对是碰的资本, 打掉等于自废武功),
    平分组里把它们排到最后. 只作平分裁决 -- 若打役对严格更优(进张更多),
    仍按牌效打, 不为保对牺牲向听.
    options 是 jansou 的候选对象(牌种走 o.discard.name).
    """
    if len(options) < 2:
        return options
    top = options[0]
    tied = [o for o in options
            if o.shanten == top.shanten
            and o.total_acceptance == top.total_acceptance]
    kind_of = {id(o): kind_name_to_tile(o.discard.name) for o in tied}
    if len(set(kind_of.values())) < 2:
        return options                       # 平分组里只有一种牌, 无从裁决
    tied = tied[:MAX_EXTENSION_CANDIDATES]
    future: dict[str, int] = {}
    for o in tied:
        tile = kind_name_to_tile(o.discard.name)
        if tile in future:
            continue
        rest = list(tiles)
        rest.remove(tile)                    # 同种牌多张时移哪张都一样
        future[tile] = _extension_ukeire(rest, meld_count, player_count)
    tied_ids = {id(o) for o in tied}
    untied = [o for o in options if id(o) not in tied_ids]

    def breaks_yaku(o) -> int:
        k = kind_name_to_tile(o.discard.name)
        return int(k in keep_kinds and tiles.count(k) >= 2)

    return sorted(tied, key=lambda o: (breaks_yaku(o),
                                       -future[kind_name_to_tile(o.discard.name)])) \
        + untied


def choose_discard(
    tiles: list[str],
    meld_count: int = 0,
    *,
    rng: random.Random | None = None,
    player_count: int = 3,
    suboptimal_rate: float = DEFAULT_SUBOPTIMAL_RATE,
    random_rate: float = DEFAULT_RANDOM_RATE,
    allowed_tiles: frozenset[str] | set[str] | None = None,
    keep_kinds: frozenset[str] = frozenset(),
) -> DiscardPlan:
    """决定打哪张牌.

    tiles 必须是"摸牌后"的手牌: 副露数 0 时 14 张, 每多一个副露少 3 张
    (jansou 的约束是 concealed in (14, 11, 8, 5, 2)).

    allowed_tiles 非None 时**只在这几种牌里选**(立直选牌态用): 画面上
    其余牌是变暗的, 点上去雀魂不响应, 等于白等. 白名单按**牌种**给而不是
    下标, 因为同一张牌在手牌里可能有多张, 而它们的高亮状态是一致的.
    随机性在过滤之后仍然生效 -- 次优/随机也只在合法候选里挑, 不会挑到
    点不动的牌.
    """
    rng = rng if rng is not None else random
    options = _honors_first(
        discard_evaluation(_hand_of(tiles, meld_count), player_count=player_count
    ))
    if allowed_tiles is not None:
        options = [o for o in options
                   if kind_name_to_tile(o.discard.name) in allowed_tiles]
    if not options:
        raise ValueError("没有可打的牌")
    # 一阶(向听, 进张)完全平分时, 用二阶进张裁决发展方向(6-8-9 打 9 之类);
    # keep_kinds 非空时平分组里不优先打役牌对子(见 _extension_tiebreak).
    options = _extension_tiebreak(options, tiles, meld_count, player_count,
                                  keep_kinds=keep_kinds)

    roll = rng.random()
    if roll < random_rate and len(options) > 1:
        index, picked = rng.randrange(len(options)), PICK_RANDOM
    elif roll < random_rate + suboptimal_rate and len(options) > 1:
        index, picked = 1, PICK_SECOND
    else:
        index, picked = 0, PICK_BEST

    option = options[index]
    return DiscardPlan(
        tile=kind_name_to_tile(option.discard.name),
        picked=picked,
        shanten=int(option.shanten),
        acceptance=int(option.total_acceptance),
    )


def rank_discards(
    tiles: list[str], meld_count: int = 0, *, player_count: int = 3,
    keep_kinds: frozenset[str] = frozenset(),
) -> list[DiscardPlan]:
    """按牌效从好到坏列出所有可打的牌(不带随机性). 用于调试与单测."""
    options = _honors_first(
        discard_evaluation(_hand_of(tiles, meld_count),
                           player_count=player_count))
    options = _extension_tiebreak(options, tiles, meld_count, player_count,
                                  keep_kinds=keep_kinds)
    return [
        DiscardPlan(
            tile=kind_name_to_tile(o.discard.name),
            picked=PICK_BEST,
            shanten=int(o.shanten),
            acceptance=int(o.total_acceptance),
        )
        for o in options
    ]


# ------------------------------------------------------------- 模糊手牌下的稳健出牌

# 三人麻将的全部牌种(万子去掉 2m~8m). 枚举假设时要的就是这个全集.
ALL_TILES: tuple[str, ...] = (
    tuple(f"{n}m" for n in (1, 9))
    + tuple(f"{n}{s}" for s in "ps" for n in range(1, 10))
    + tuple(f"{n}z" for n in range(1, 8))
)

# 每多一张认不准的牌, 假设组合数就乘以 27. 实测单次 discard_evaluation
# 约 0.11ms(200 副随机手牌), 于是:
#   1 张 -> 27 组约 4ms, 2 张 -> 729 组约 95ms, 3 张 -> 19683 组约 2.9s.
# 2.9s 在对局里绝对不能同步等, 所以超过这个数就改走"逐张展开"近似(见 robust_discard).
# 注意: 上限若 < 729, 2 张认不准的情形会静默退化成近似路径, 必须真的检查.
MAX_ROBUST_COMBOS = 729

# 假设的权重按"视觉混淆距离"衰减: 距离 d 的假设权重 = CONFUSION_DECAY ** d.
# 0.55 的含义是"隔一个点数的可能性只有紧邻的一半, 隔两个只有四分之一".
# 实测(turn 185)这个参数在 0.25~1.0 之间结论完全一致(都选 3z), 只有退到
# 0.0(等于只看最可能的那个假设)才会翻回 9p -- 说明结论不依赖这个值, 是稳健的.
CONFUSION_DECAY = 0.55

# 超过这个张数就放弃"所有组合一起枚举", 改用逐张展开.
FULL_ENUM_MAX_SLOTS = 2


def confusion_distance(seen: str, candidate: str) -> int:
    """"认成seen 的那张牌"与"候选牌"有多容易被认混(越小越像).

    这个距离是视觉层面的直觉, 不是从数据里学出来的:

    - 同点数不同花色(比如 5m/5p): 牌面图案完全不同, 距离 4;
    - 跨数牌与字牌: 一眼就能看出类别, 距离 5;
    - 同花色相邻点数(7p/8p, 3z/4z): 最常见的误识, 距离 1;
    - 同花色隔两个: 少见但确实存在(实测位置 4 认成 7p, 真值可能是 5p),
      距离 2, 正好落在 ±1 候选集之外 -- 这就是"覆盖不全"的来源.

    字牌的点数距离封顶到 3: 1z/4z/7z 都是常见误识对象, 再远的字牌之间
    几乎不会混, 没必要区分.
    """
    if seen == candidate:
        return 0
    if seen[1] != candidate[1]:
        return 4 if candidate[1] != "z" and seen[1] != "z" else 5
    lo, hi = int(seen[0]), int(candidate[0])
    step = abs(lo - hi)
    return min(step, 3) if seen[1] == "z" else step


def hypothesis_weights(seen: str, *,
                       decay: float = CONFUSION_DECAY) -> dict[str, float]:
    """低置信牌"认成seen"时, 每一种真实取值的可能性权重."""
    return {c: decay ** confusion_distance(seen, c) for c in ALL_TILES}


def candidates_for(tile: str) -> list[str]:
    """这张低置信牌可能是什么.

    **返回全部 27 种牌种**, 按可能性从高到低排.

    对每张认不准的牌枚举**全部 34 种**候选, 不只限同花色相邻点数: 位置 4
    认成 7p 时, 真值可能是 5p/9p, 只给 ±1 会把这些全漏掉. 实测单次
    discard_evaluation 约 0.095ms, 2 张低置信全展开(27^2 组)只要 84ms.

    排序按可能性权重从高到低, 只是为了让枚举时的 early-abort 更容易命中,
    不影响结果(所有假设都会算到).
    """
    weights = hypothesis_weights(tile)
    return sorted(ALL_TILES, key=lambda c: (-weights[c], c))


@dataclass(frozen=True)
class RobustPlan:
    """模糊手牌下的出牌决策.

    tile:        要打的牌
    stable:      True = 无论那些低置信牌实际是什么, 这张都是最优的
    shanten:     按"认到的牌"算的向听数(仅供参考, 认错时没有意义)
    cases:       枚举了多少个假设组合
    alternatives: 各假设下的最优牌有几种不同取值(1 = 结论一致)
    """

    tile: str
    stable: bool
    shanten: int
    cases: int
    alternatives: int


def robust_discard(
    tiles: list[str],
    blurry: list[int] | frozenset[int] | set[int] | None = None,
    meld_count: int = 0,
    *,
    player_count: int = 3,
    max_combos: int = MAX_ROBUST_COMBOS,
    full_enum_max_slots: int = FULL_ENUM_MAX_SLOTS,
) -> RobustPlan:
    """手牌里有认不准的牌时, 挑一个"怎么认都不会错"的牌打.

    **要解决的问题**(实跑 turn 185, 用户报的"已经有 789p 还把 9p 打了"):

        真手牌是 789p, 牌面模型把第二张读成了 7p(conf 0.66), 于是
        "789p"变成"779p". 牌效**正确地**算出了"打 9p 最优"(向听1 进张7),
        打出去直接退向 -- 而真实手牌下最优是打 3z(向听1 进张21).

    **为什么不能靠"别点那张低置信的牌"**: 实测 6 个已证实误识的轮次里,
    只有 1 个真的点了低置信那张. 其余 5 个点的是**置信度正常**的牌, 只是
    手牌里混进了一张错牌, 把"哪张该打"的全局判断带偏了. 所以防护必须做在
    **决策层**, 不是点击层.

    **做法**: 把每张低置信牌可能的真实取值列出来(全部 27 种, 按视觉混淆
    距离加权), 枚举组合, 分别算最优出牌, 再挑一张综合最好的.

    **评分是"加权期望名次", 不是"最坏名次"**. 这个改动很关键, 记一笔:

        最早用的是最坏名次(每张牌在所有假设下的**最大**名次, 取最小者).
        它在 ±1 候选集上看起来没问题, 但一旦把候选放宽到全 27 就崩了 --
        turn 185 反而选出退向的 9p. 查下来是: 3z 在真值下排第 1(向听1 进张21),
        只在"idx4 是 2p"这个假设下跌到第 9, 而 2p 与认出来的 7p 毫无视觉相似性,
        是典型的离群假设. 最大值被这种几乎不可能的假设污染, 好牌被拖累.

        改成按视觉混淆距离加权的期望名次之后, 离群假设权重极低, 结论回到3z.
        而且实测在 decay 0.25~1.0 之间结论完全一致, 只有退到 0.0 才会翻回9p
        -- 说明这不是调参调出来的, 是本来就该这样.

    **候选牌只能来自"认准了"的槽位**(即不在 blurry 里的位置). 这条很关键:
        枚举低置信槽位的时候会让它变成任意牌种, 于是某个假设下最优解可能是
        "9s" -- 可是屏幕上那只手牌里根本没有 9s(认出来的是 8s), 点下去打
        不出去. 实测正是这么翻车的: 槽位 11 假设成 9s 后, 稳健解选了 9s,
        然后 pick_tile_index 返回 None. 低置信槽位自己**可以**被打(点那个
        位置就是打那张牌), 但它"可能变成"的那个牌种不能作为候选.

    **超过两张低置信就改走"逐张展开"**: 3 张全展开是 19683 组约 2.9s, 对局里
        不能等. 改成每次只展开一张(其余保持认出的值)后只要 6~12ms, 而在实跑
        日志的全部 20 个触发轮次上与全展开**结论完全一致**. 理由是低置信牌
        之间的误识近似独立, 一次只动一张已经足够. 这条一致性有测试守着.

    blurry 传 None 或空集合 = 全部牌都认准了, 直接返回普通最优解
    (不做任何枚举, 零额外开销).
    """

    def ranked(hand: list[str]) -> list[DiscardPlan]:
        return [
            DiscardPlan(
                tile=kind_name_to_tile(o.discard.name), picked=PICK_BEST,
                shanten=int(o.shanten), acceptance=int(o.total_acceptance))
            for o in _honors_first(
                discard_evaluation(_hand_of(hand, meld_count),
                                   player_count=player_count))
        ]

    base = ranked(tiles)
    if not base:
        raise ValueError("没有可打的牌")
    if not blurry:
        return RobustPlan(base[0].tile, True, base[0].shanten, 1, 1)

    slots = sorted(i for i in blurry if 0 <= i < len(tiles))
    if not slots:
        return RobustPlan(base[0].tile, True, base[0].shanten, 1, 1)

    # 候选牌只能来自"认准了"的槽位 -- 见docstring 里那条约束.
    certain = {t for i, t in enumerate(tiles) if i not in set(slots)}

    # 每张低置信牌的候选与权重(权重来自视觉混淆距离, 见 hypothesis_weights).
    weights = [hypothesis_weights(tiles[i]) for i in slots]

    # cases 累加: 统计一共算了几种假设组合(逐张展开时是可数的加权样本数).
    # 加权名次之和, 越小越好.
    score: dict[str, float] = {}
    best_shanten = base[0].shanten
    evaluated = 0

    # 顺便把"各假设下最优牌的种类"记下来 -- stable 就靠它. 只在组合枚举
    # 模式下有完整意义; 逐张展开时用权重最高的那个假设代表(见_iter_hypotheses).
    tops: set[str] = set()

    def accumulate(hand: list[str], weight: float,
                   record_top: bool = False) -> list[DiscardPlan]:
        """算一个假设, 把它的名次按 weight 累加进 score."""
        try:
            table = ranked(hand)
        except ValueError:
            return []                        # 这组假设凑不出合法手牌
        if record_top and table:
            tops.add(table[0].tile)
        seen: set[str] = set()
        for position, option in enumerate(table):
            if option.tile in seen:
                continue                      # 同一种牌只算它在该假设下的首次名次
            seen.add(option.tile)
            score[option.tile] = score.get(option.tile, 0.0) + weight * position
        return table

    if len(slots) > full_enum_max_slots or len(ALL_TILES) ** len(slots) > max_combos:
        # 太多张认不准 -> 逐张展开近似, 见 docstring 里的说明.
        for slot, weights_of_slot in zip(slots, weights):
            for value in ALL_TILES:
                hand = list(tiles)
                hand[slot] = value
                table = accumulate(hand, weights_of_slot[value])
                evaluated += 1
                if table:
                    best_shanten = min(best_shanten, table[0].shanten)
        # 逐张展开下"最可能的那组假设"= 每一张都取权重最高的取值.
        best_combo = tuple(max(w, key=lambda c: (w[c], c)) for w in weights)
        hand = list(tiles)
        for slot, value in zip(slots, best_combo):
            hand[slot] = value
        try:
            table = ranked(hand)
        except ValueError:
            table = []
        if table:
            tops.add(table[0].tile)
    else:
        # 组合一起枚举: [(假设取值元组, 权重), ...]
        combos: list[tuple[tuple[str, ...], float]] = [((), 1.0)]
        for weights_of_slot in weights:
            combos = [(c + (v,), p * weights_of_slot[v])
                      for c, p in combos for v in ALL_TILES]
        for combo, weight in combos:
            hand = list(tiles)
            for slot, value in zip(slots, combo):
                hand[slot] = value
            table = accumulate(hand, weight, record_top=True)
            evaluated += 1
            if table:
                best_shanten = min(best_shanten, table[0].shanten)

    if not score:
        return RobustPlan(base[0].tile, False, base[0].shanten, evaluated, 0)

    # 只在认准的牌里挑 -- 见 docstring 里那条"不能凭空造牌"的约束.
    pool = {t: s for t, s in score.items() if t in certain}
    pick = min(pool, key=lambda t: (pool[t], t)) if pool else None
    if pick is None:
        # 所有假设的最优都指向低置信槽位自己的牌种, 或者都不在 certain 里.
        # 退而求其次: 只看认到的牌, 打普通最优.
        return RobustPlan(base[0].tile, False, best_shanten, evaluated, 0)
    return RobustPlan(pick, len(tops) == 1, best_shanten, evaluated, len(tops))
