"""出哪张牌: EV 候选优先, 没有就 jansou 纯牌效.

只做"选一张", 不判断该不该出牌/该不该立直 —— 那些在 rules.py.

**牌名口径在这个入口收口**: 协议手牌里赤 5 写作 "0m"/"0p"/"0s", 而
jansou / EV 一律是 "5m"/"5p"/"5s". 进来先把两边都归一成 34 类记号, 下游
(`pick_by_ev_rank` / `choose_discard` / `brain`) 只见一种口径 —— 以前三处
各自比较赤 5, 结果是: 带赤 5 的手牌在 EV 里永远选不中、立直宣言牌是赤 5
时白名单过滤成空(`ValueError: 没有可打的牌`, 2026-10-21 00:53 因此停了
一整场)、牌效的平局裁决 `list.remove` 还会直接抛.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from src.mahjong.brain import choose_discard
from src.mahjong.rules import ev_tile_of, pick_by_ev_rank, yaku_pair_kinds
from src.net.state import GameState
from src.net.tiles import normalize


def choose(state: GameState, allowed: Iterable[str] | None = None,
           rng: random.Random | None = None,
           ev_rows: Sequence | None = None) -> str:
    """返回一个牌名(34 类口径); `allowed` 非空时只在这个集合里选(立直宣言牌)."""
    hand = [normalize(tile) for tile in state.hand]
    kinds = {normalize(tile) for tile in allowed} if allowed else None
    ev = _ev_tile(hand, ev_rows, kinds)
    if ev is not None:
        return ev
    plan = choose_discard(
        hand,
        len(state.melds),
        rng=rng or random.Random(),
        player_count=state.player_count,
        allowed_tiles=kinds,
        keep_kinds=yaku_pair_kinds(hand, 0, state.field_wind()),
    )
    return plan.tile


def _ev_tile(hand: list[str], ev_rows: Sequence | None,
             kinds: set[str] | None) -> str | None:
    if not ev_rows:
        return None
    row = pick_by_ev_rank(hand, ev_rows, kinds)
    return ev_tile_of(row) if row is not None else None
