"""协议事件名与谓词.

`GameState.on_event` 给的是 liqi 的 Action 名 + data; 这里只集中处理
"这条事件对我意味着什么", 避免 `hand.py` 里散落字符串和 seat 判断.
"""

from __future__ import annotations

from src.net.state import GameState

NEW_ROUND = "ActionNewRound"
DEAL_TILE = "ActionDealTile"
DISCARD_TILE = "ActionDiscardTile"
CHI_PENG_GANG = "ActionChiPengGang"
AN_GANG_ADD_GANG = "ActionAnGangAddGang"
BA_BEI = "ActionBaBei"
HAND_OVER_ACTIONS = frozenset({"ActionHule", "ActionNoTile", "ActionLiuJu"})


def is_mine(data: dict, state: GameState) -> bool:
    """这条事件的 seat 是不是我(proto3 缺省 seat=0)."""
    if state.my_seat is None:
        return False
    try:
        return int(data.get("seat", 0) or 0) == state.my_seat
    except (TypeError, ValueError):
        return False


def is_post_meld_discard(data: dict) -> bool:
    """这次 ActionChiPengGang 是我方吃/碰(之后要自己打一张).

    大明杠(type=2)之后先补牌, 不立即打; 0/1 = 吃/碰.
    """
    try:
        return int(data.get("type") or 0) in (0, 1)
    except (TypeError, ValueError):
        return False
