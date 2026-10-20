"""对局规则的纯函数: 役牌对子 / 赤 5 选槽 / EV 落地 / 杠不拆搭.

事件驱动内核(`src.engine`)只依赖这些稳定规则, 不依赖任何视觉/页面状态.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

#: 三元牌(白/发/中): 任何场合都是役. 场风也是役, 由调用方按房型传入
#: (见 tuning.ROUND_WIND: 友人场只有东场, 场风恒为东); 自风要读座位,
#: 没读, 不碰, 宁可少碰也不碰死.
YAKU_HONORS = ("5z", "6z", "7z")


def yaku_pair_kinds(tiles: list[str], missing_slots: int = 0,
                    round_wind: str = "1z") -> frozenset[str]:
    """手里握着的役牌对子(及以上)的种类集合.

    `has_yaku_pair` 的"是/否"版, 出牌裁决(keep_kinds: 平分时不优先打役对)
    与决策日志都要用种类而非布尔. missing_slots > 0 = 读数不可信, 返回空.
    """
    if missing_slots:
        return frozenset()
    return frozenset(t for t in (*YAKU_HONORS, round_wind)
                     if tiles.count(t) >= 2)


def has_yaku_pair(tiles: list[str], missing_slots: int = 0,
                  round_wind: str = "1z") -> bool:
    """手里是否握着役牌对子 -- "有役才碰"的判据."""
    if missing_slots:
        return False
    return bool(yaku_pair_kinds(tiles, missing_slots, round_wind))


def pick_tile_index(tiles: list[str], tile: str,
                    rng: random.Random | None = None,
                    allowed_indices: list[int] | None = None,
                    drawn_tile: str | None = None) -> int | None:
    """在手牌里找这张牌的位置; **刚摸的那张**(末位)匹配时优先打它.

    `drawn_tile` 是协议给的"刚摸到的那一张"(带赤记号: "0p"/"0s"); 同种牌
    有多个槽位时用它保留赤 5(只做平局裁决). `allowed_indices` 非 None 时
    只在这些下标里找(立直选牌态).
    """
    positions = [i for i, t in enumerate(tiles) if t == tile]
    if allowed_indices is not None:
        allow = set(allowed_indices)
        positions = [i for i in positions if i in allow]
    if not positions:
        return None
    if len(positions) > 1 and drawn_tile is not None:
        if drawn_tile == "0" + tile[1]:
            return positions[0]       # 最右是刚摸到的赤 -> 打左边那张
        return positions[-1]          # 最左是赤 -> 打右边那张
    if len(tiles) % 3 == 2 and positions[-1] == len(tiles) - 1:
        return positions[-1]      # 刚摸的那张就是它 -> 直接打摸到的牌
    return positions[0]


def ev_tile_of(row) -> str | None:
    """EV 候选里那一张牌的名字. 接受 `EvDiscard` 对象或裸字符串."""
    if isinstance(row, str):
        return row
    tile = getattr(row, "tile", None)
    return tile if isinstance(tile, str) else None


def pick_by_ev_rank(tiles: list[str], ev_rank: Sequence | None,
                    allowed_tiles: set[str] | None = None):
    """按 EV 排序(最优在前)挑**第一张落地得了的牌**, 返回那个候选(或 None).

    只做落地过滤, 不重新排序(排序是引擎的事):
    - 候选不在这一帧认出的手牌里 -> 跳过;
    - 给了 allowed_tiles(立直选牌态的高亮集合)就只在这个集合里挑.
    全都落不了地时返回 None, 调用方回退到牌效 —— **宁可回退, 不硬打**.
    """
    have = set(tiles)
    for row in ev_rank or ():
        tile = ev_tile_of(row)
        if not tile or tile not in have:
            continue
        if allowed_tiles is not None and tile not in allowed_tiles:
            continue
        return row
    return None


#: 暗杠/加杠的"形中性"半径. 与要杠的牌相距这么多以内的同花色牌, 都可能
#: 和它组成顺子: 7 的 ±2 正好覆盖 567 / 678 / 789 三种含 7 的顺子.
_KAN_NEIGHBOR_REACH = 2


def kan_keeps_shape(tile: str, tiles: Sequence[str]) -> bool:
    """杠掉这张牌会不会拆散手里的搭子.

    字牌恒为 True; `tiles` 里那几张要杠的牌自己会被跳过.
    """
    if not tile or len(tile) != 2 or tile[1] == "z":
        return True
    suit = tile[1]
    rank = 5 if tile[0] == "0" else int(tile[0])   # 赤 5 折叠
    for other in tiles:
        if len(other) != 2 or other[1] != suit:
            continue
        other_rank = 5 if other[0] == "0" else int(other[0])
        if other_rank == rank:
            continue
        if abs(other_rank - rank) <= _KAN_NEIGHBOR_REACH:
            return False
    return True
