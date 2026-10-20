"""一次决策的输出(纯数据).

刻意做得很窄: 决策只描述"做什么 + 用哪张牌/哪组牌", 不描述"点哪个坐标".
坐标由 `locator` 在后面按协议屏序/按钮种类查, 这样决策可以离线回放.
"""

from __future__ import annotations

from dataclasses import dataclass

# 动作种类. 和旧 `table.ACTION_*` 同名, 但这里是协议事件驱动的窄集合.
DISCARD = "discard"
RIICHI = "riichi"
CHI = "chi"
PENG = "peng"
GANG = "gang"
WIN = "win"
BABEI = "babei"
SKIP = "skip"
WAIT = "wait"
HAND_OVER = "hand_over"


@dataclass(frozen=True)
class Decision:
    """一个可执行/可记录的动作.

    - discard: `tile` 是要打的牌(协议牌名, 赤 5 保留 "0p")
    - riichi: `tile` 是宣言牌, `tiles` 是协议允许的宣言牌集合
    - peng/gang/chi: `tiles` 是协议给的组合(不含被鸣的那张)
    - win/babei/skip/wait/hand_over: 无牌参数
    """

    action: str
    tile: str | None = None
    tiles: tuple[str, ...] = ()
    reason: str = ""
