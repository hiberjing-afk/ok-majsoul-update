"""窗口位置计算的纯逻辑(不依赖 ok-script, 不碰 win32), 便于单元测试.

真实鼠标点击(Pynput)有两个前提:

1. 窗口必须完整落在虚拟屏幕内 -- 真实光标到不了屏幕外, 窗口哪怕只有
   目标按钮那一半在屏幕外, 按钮就永远点不到;
2. 窗口必须在前台 -- PynputInteraction.clickable() 不满足时会静默跳过点击.

这里只放"算位置"的部分, 真正调 win32 移动窗口的部分留在任务里,
这样算错了可以在没有游戏窗口的情况下被测出来.
"""

from __future__ import annotations

Rect = tuple[int, int, int, int]  # (left, top, right, bottom)
Screen = tuple[int, int, int, int]  # (x, y, width, height) 虚拟屏幕


def is_inside(rect: Rect, screen: Screen) -> bool:
    """窗口外框是否完整落在虚拟屏幕内."""
    left, top, right, bottom = rect
    sx, sy, sw, sh = screen
    return left >= sx and top >= sy and right <= sx + sw and bottom <= sy + sh


def clamp_into_screen(rect: Rect, screen: Screen) -> tuple[int, int] | None:
    """算出把窗口完整移进屏幕所需的左上角坐标.

    只做平移, 不改变窗口尺寸. 已经完整在屏幕内时返回 None(不用动).
    窗口比屏幕还大时, 优先对齐左上角(至少保证左上区域可点).
    """
    if is_inside(rect, screen):
        return None

    left, top, right, bottom = rect
    sx, sy, sw, sh = screen
    width = right - left
    height = bottom - top

    new_x = left
    if right > sx + sw:
        new_x = sx + sw - width
    if new_x < sx:
        new_x = sx

    new_y = top
    if bottom > sy + sh:
        new_y = sy + sh - height
    if new_y < sy:
        new_y = sy

    return int(new_x), int(new_y)
