"""出哪张牌: EV 候选优先, 没有就 jansou 纯牌效.

只做"选一张", 不判断该不该出牌/该不该立直 —— 那些在 rules.py.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from src.mahjong.brain import choose_discard
from src.mahjong.rules import ev_tile_of, pick_by_ev_rank, yaku_pair_kinds
from src.net.state import GameState


def choose(state: GameState, allowed: Iterable[str] | None = None,
           rng: random.Random | None = None,
           ev_rows: Sequence | None = None) -> str:
    """返回一个牌名; `allowed` 非空时只在这个集合里选(立直宣言牌)."""
    ev = _ev_tile(state, ev_rows, allowed)
    if ev is not None:
        return ev
    plan = choose_discard(
        list(state.hand),
        len(state.melds),
        rng=rng or random.Random(),
        player_count=state.player_count,
        allowed_tiles=list(allowed) if allowed else None,
        keep_kinds=yaku_pair_kinds(list(state.hand), 0, state.field_wind()),
    )
    return plan.tile


def _ev_tile(state: GameState, ev_rows: Sequence | None,
             allowed: Iterable[str] | None) -> str | None:
    if not ev_rows:
        return None
    row = pick_by_ev_rank(list(state.hand), ev_rows,
                          set(allowed) if allowed else None)
    return ev_tile_of(row) if row is not None else None
