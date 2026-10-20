"""liqi 牌字符串的校验与归一(纯逻辑).

雀魂协议里的牌直接就是 mpsz 字符串("3s"/"0p"/"1z"), 与
src/mahjong/tiles.py 的记号同源 -- 这里只做校验和拆分, **不造第二套
牌记号**. 赤 5 在协议里写作 "0m"/"0p"/"0s", parse_tile 会折叠成 "5p"
(赤宝牌只影响番数, 牌效与普通 5 相同, 这是项目刻意的决定).
"""

from __future__ import annotations

import re

from src.mahjong.tiles import parse_tile

# 数牌 0-9m/p/s(0 = 赤 5) + 字牌 1-7z(东南西北白发中)
_TILE_RE = re.compile(r"^(?:[0-9][mps]|[1-7]z)$")


def is_tile(text: object) -> bool:
    """协议里这个字符串是不是一张合法的牌?"""
    return isinstance(text, str) and bool(_TILE_RE.match(text))


def normalize(text: str) -> str:
    """协议牌字符串 -> 项目标准记号(赤 5 折叠成 5p 等). 非法输入抛 ValueError."""
    return parse_tile(text)


def all_tiles_valid(tiles: object) -> bool:
    """list[str] 里每一张都是合法牌(空列表算 False -- 空手牌是可疑信号)."""
    return bool(tiles) and all(is_tile(t) for t in tiles)
