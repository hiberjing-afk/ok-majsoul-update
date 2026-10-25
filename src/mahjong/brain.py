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

from src.mahjong.tiles import kind_name_to_tile, parse_tile, to_mpsz

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
    ValueError, 调用方据此放弃这一帧.
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


#: 三人麻将的全部牌种(万子只有 1m/9m). 进张枚举按它遍历.
ALL_TILES: tuple[str, ...] = (
    tuple(f"{n}m" for n in (1, 9))
    + tuple(f"{n}{s}" for s in "ps" for n in range(1, 10))
    + tuple(f"{n}z" for n in range(1, 8))
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


def _kind_index(tiles: list[str], kind: str) -> int | None:
    """手牌里"牌种为 kind"的那一张的下标; 容忍赤 5 的两种写法.

    `kind` 是 jansou 的 34 类记号("5m"), 而协议手牌里赤 5 写作 "0m".
    找不到就返回 None —— 调用方据此跳过这条平局裁决. **这里不能抛**:
    牌效是"EV 算不出来"时的兜底路径, 抛一次就把整场对局打成"决策异常停手".
    """
    for index, held in enumerate(tiles):
        try:
            if parse_tile(held) == kind:
                return index
        except ValueError:
            continue
    return None


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
        index = _kind_index(rest, tile)
        if index is None:
            continue          # 手牌里没这张(异常数据): 不做二阶裁决, 排序时垫底
        rest.pop(index)       # 同种牌多张时移哪张都一样
        future[tile] = _extension_ukeire(rest, meld_count, player_count)
    tied_ids = {id(o) for o in tied}
    untied = [o for o in options if id(o) not in tied_ids]

    def breaks_yaku(o) -> int:
        k = kind_name_to_tile(o.discard.name)
        return int(k in keep_kinds and tiles.count(k) >= 2)

    # 拿不到二阶进张的(见上面的 continue)按 -1 计入, 排序时排最后.
    return sorted(tied, key=lambda o: (breaks_yaku(o),
                                       -future.get(kind_name_to_tile(o.discard.name), -1))) \
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
    visible: list[int] | None = None,
) -> list[DiscardPlan]:
    """按牌效从好到坏列出所有可打的牌(不带随机性). 用于调试与单测.

    `visible` 是"手牌之外看得见的牌"的 34 类计数(牌河 + 全部副露 + 已翻
    指示牌, 见 ev_eval.visible_from). 给了它, 进张就按**真实剩张**算: 牌河里
    已经躺了三张 5p, 再指望摸 5p 就是自欺. 不给(None) 就是老口径 —— 每种牌
    恒按 4 张算, 进张系统性虚高.

    **不要把 14 张以外的张数档传进来**: `discard_evaluation` 只认 post-draw
    手(14/11/8/5/2 张), 13 张档它会直接抛 ValueError.
    """
    options = _honors_first(
        discard_evaluation(_hand_of(tiles, meld_count),
                           player_count=player_count, visible=visible))
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
