"""决策 -> 屏幕坐标: 视觉只回答"第几个槽/哪个按钮".

这里刻意**不读牌名**: 手牌标签由协议按雀魂理牌规则排好, 视觉只需要给出
从左到右的 n 个框. 牌面模型读出的 `label/conf` 一律不用, 所以低置信/漏认/
布局动画的对账在这条路上都不存在.

按钮也一样: 协议 `operation_list` 已经说了此刻该有哪些按钮, 视觉只负责
在画面里找到对应种类的那一个框.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.mahjong.rules import pick_tile_index
from src.mahjong.tiles import screen_order
from src.net.state import GameState
from src.net.tiles import normalize


def screen_tiles(state: GameState) -> list[str]:
    """协议暗手 -> 屏幕从左到右的牌名(与出牌下标同一套 `screen_order`)."""
    return screen_order(list(state.hand), drawn=state.drawn or None)


def slot_index(state: GameState, tile: str) -> int | None:
    """要打的牌 -> 屏幕槽位下标.

    先走 `table.pick_tile_index`(它带"优先打刚摸的那张 / 保留赤 5"的规则),
    再按归一化牌种兜底: jansou 返回 "5p", 而协议手牌里可能是赤 5 "0p".
    """
    order = screen_tiles(state)
    index = pick_tile_index(order, tile, drawn_tile=state.drawn or None)
    if index is not None:
        return index
    target = normalize(tile)
    for index, held in enumerate(order):
        if normalize(held) == target:
            return index
    return None


def pick_button(groups: dict, kind: str):
    """从 `read_buttons` 的分组里挑协议要的那个按钮框.

    `win` 可能是 `he` 或 `zimo`; 叫牌类(chi/peng/gang)如果模型只认出一个,
    就把它当成协议要的那个(协议已经确定种类, 视觉只缺标签).
    """
    aliases = {
        "win": ("he", "zimo"),
        "skip": ("tiaoguo",),
    }
    for name in (kind, *aliases.get(kind, ())):
        boxes = groups.get(name) or []
        if boxes:
            return boxes[0]
    if kind in ("chi", "peng", "gang"):
        call_boxes = [b for name in ("chi", "peng", "gang")
                      for b in (groups.get(name) or [])]
        if len(call_boxes) == 1:
            return call_boxes[0]
    return None


@dataclass
class VisionLocator:
    """用现有 `GameVision` 找坐标; 找不到就返回 None(调用方重试, 不猜)."""

    vision: object

    def hand_boxes(self, frame, expected: int) -> list | None:
        """读手牌行, 只取几何框. 框数对不上 = 这一帧不能用."""
        if frame is None:
            return None
        reading = self.vision.read_hand(frame)
        boxes = list(getattr(reading, "boxes", None) or [])
        if len(boxes) < expected:
            return None
        # 副露/暗杠偶尔会漏进手牌行右端(模型把竖着的杠牌也当手牌); 协议暗手
        # 只有 expected 张, 手牌行从左往右, 多出来的必在右端, 取前 expected 个.
        return boxes[:expected]

    def button_box(self, frame, kind: str):
        """按协议给的动作种类找按钮框."""
        if frame is None:
            return None
        return pick_button(self.vision.read_buttons(frame), kind)
