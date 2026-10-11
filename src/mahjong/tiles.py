"""牌的记号与 jansou 之间的转换(纯逻辑, 不依赖 ok-script / 截图).

统一用天凤 / jansou 的 mpsz 写法, 识别层和决策层之间只传这一种字符串:

    1m .. 9m   万子
    1p .. 9p   筒子
    1s .. 9s   索子
    1z .. 7z   字牌(1z 东, 2z 南, 3z 西, 4z 北, 5z 白, 6z 发, 7z 中)

赤宝牌在这里**不单独表示**: 赤 5 的牌效和普通 5 完全一样(只影响番数),
记作 "5p" / "5s" 即可. 识别层如果写成 "0p"(天凤的赤 5 写法), normalize()
会先折叠成 "5p".

三人麻将(player_count=3)去掉 2m-8m, 所以 "2m" 这种牌根本不该出现在
识别结果里 -- `tiles_out_of_play()` 就是用来抓这种识别错误的.
"""

from __future__ import annotations

SUITS = "mpsz"
HONOR_NAMES = ["东", "南", "西", "北", "白", "发", "中"]

# jansou 的 TileKind 名字 -> 我们的记号
_KIND_TO_TILE = {"EAST": "1z", "SOUTH": "2z", "WEST": "3z", "NORTH": "4z",
                 "HAKU": "5z", "HATSU": "6z", "CHUN": "7z"}


def parse_tile(text: str) -> str:
    """把 "3m" / "0p" / "东" 这类写法归一到标准记号. 认不出来就抛 ValueError."""
    text = text.strip()
    if not text:
        raise ValueError("空字符串不是一张牌")

    if text in HONOR_NAMES:
        return f"{HONOR_NAMES.index(text) + 1}z"

    if len(text) != 2:
        raise ValueError(f"无法解析的牌: {text!r}")

    num, suit = text[0], text[1].lower()
    if suit not in SUITS:
        raise ValueError(f"未知花色: {text!r}")
    if not num.isdigit():
        raise ValueError(f"未知的牌: {text!r}")

    # 天凤用 0 表示赤 5, 牌效上等同普通 5. 字牌没有赤宝牌, "0z" 是非法输入.
    if num == "0":
        if suit == "z":
            raise ValueError(f"字牌没有赤宝牌: {text!r}")
        num = "5"
    if suit == "z":
        if not 1 <= int(num) <= 7:
            raise ValueError(f"字牌只有 1z-7z: {text!r}")
    elif not 1 <= int(num) <= 9:
        raise ValueError(f"数牌只有 1-9: {text!r}")
    return f"{num}{suit}"


def to_mpsz(tiles: list[str]) -> str:
    """牌列表 -> mpsz 字符串, 交给 jansou 的 parse_mpsz.

    按花色分组并升序排列, 例如 ["3m", "1m", "东"] -> "13m 1z".
    空手牌返回空字符串.
    """
    buckets: dict[str, list[int]] = {s: [] for s in SUITS}
    for tile in tiles:
        tile = parse_tile(tile)
        buckets[tile[1]].append(int(tile[0]))
    parts = []
    for suit in SUITS:
        if buckets[suit]:
            parts.append("".join(str(n) for n in sorted(buckets[suit])) + suit)
    return " ".join(parts)


def from_mpsz(spec: str) -> list[str]:
    """mpsz 字符串 -> 牌列表(升序). to_mpsz 的逆运算."""
    out: list[str] = []
    for group in spec.split():
        suit = group[-1]
        if suit not in SUITS:
            raise ValueError(f"未知花色: {group!r}")
        for ch in group[:-1]:
            if not ch.isdigit():
                raise ValueError(f"无法解析: {group!r}")
            out.append(f"{ch}{suit}")
    return sorted(out)


def kind_name_to_tile(name: str) -> str:
    """jansou 的 TileKind 名字(如 'M3' / 'EAST') -> 我们的记号('3m' / '1z')."""
    if name in _KIND_TO_TILE:
        return _KIND_TO_TILE[name]
    if len(name) == 2 and name[0] in "MPS":
        return f"{name[1]}{name[0].lower()}"
    raise ValueError(f"未知的 TileKind: {name!r}")


def is_out_of_play(tile: str, player_count: int = 3) -> bool:
    """这局里根本不存在的牌?(单人判断).

    三人麻将去掉 2m-8m, 所以认出 "4m" 一定是识别错了(或用了四人的权重).
    四人麻将永远 False.
    """
    if player_count == 4:
        return False
    kind = parse_tile(tile)
    return kind[1] == "m" and 2 <= int(kind[0]) <= 8


def out_of_play_indices(tiles: list[str], player_count: int = 3) -> list[int]:
    """返回"这局不可能的牌"在 `tiles` 里的**下标**(去重, 保持原顺序).

    为什么要下标而不是牌名(2026-10-06 实跑踩出来的):
    `tiles_out_of_play()` 只返回牌名, 上层拿它只能"整轮停手". 但**一张牌被
    认错并不等于整个局面没救** -- 真实场景是手牌 13 张里错 1 张(实测 8s 被认成
    4m, conf=0.871 还不低), 而 robust_discard 恰恰就是为"某几张牌认不准"设计的:
    枚举这些槽位的真实取值, 挑一张怎么认都不会踩坑的打.
    之前的问题 robust_discard 只在 blurry_indices(低置信槽位) 非空时触发, 而
    **"出现不可能的牌"是另一条独立的信号, 没接进去** -- 于是一张错牌把整局
    卡成永久 wait(23:21 实跑 70 轮一张牌没打).
    所以这里必须返回下标, 让调用方把它并进待纠正集合.
    """
    if player_count == 4:
        return []
    seen: set[int] = set()
    for i, tile in enumerate(tiles):
        if is_out_of_play(tile, player_count):
            seen.add(i)
    return sorted(seen)


def tiles_out_of_play(tiles: list[str], player_count: int = 3) -> list[str]:
    """返回这局里根本不存在的牌(用来抓识别错误, 报错文案用).

    三人麻将去掉 2m-8m, 所以认出 "2m" 一定是识别错了(或用了四人的模板).
    四人麻将返回空列表.

    要拿下标去纠正误识请用 `out_of_play_indices()`.
    """
    return [parse_tile(t) for t in tiles if is_out_of_play(t, player_count)]


def counts(tiles: list[str]) -> dict[str, int]:
    """统计每种牌有几张."""
    result: dict[str, int] = {}
    for tile in tiles:
        tile = parse_tile(tile)
        result[tile] = result.get(tile, 0) + 1
    return result
