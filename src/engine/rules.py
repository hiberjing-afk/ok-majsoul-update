"""协议状态 -> 动作规则(纯函数, 可离线回放).

这里刻意不碰视觉置信度: 牌名/张数/可选操作都来自协议, 视觉只负责把
`Decision.tile` 映射到屏幕槽位.

规则只保留用户拍板的三条鸣牌硬规则:
- 碰只碰役牌对子(三元牌 + 场风 + 自风);
- 吃永不做; 大明杠不做;
- 暗杠/加杠只在 `kan_keeps_shape`(不拆搭子)时做.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from src.mahjong.rules import kan_keeps_shape, yaku_pair_kinds
from src.net.state import GameState
from src.net.tiles import is_tile, normalize

from .decisions import (
    BABEI,
    DISCARD,
    GANG,
    PENG,
    RIICHI,
    SKIP,
    WAIT,
    WIN,
    Decision,
)
from .discard import choose

# OptionalOperation.type(与 src.net.state.OP_TYPE_TO_BUTTON 同源, 这里只取
# 决策需要的几个; 不 import 那张表是为了让规则本身能单独读).
OP_PENG = 3
OP_ANKAN = 4
OP_MINGGANG = 5
OP_JIAGANG = 6
OP_LIZHI = 7


def plan_discard(state: GameState, *, rng: random.Random | None = None,
                 allow_riichi: bool = True, require_turn: bool = True,
                 ev_rows: Sequence | None = None) -> Decision:
    """轮到我摸完牌/庄家开局: 出哪张(或立直宣言牌).

    require_turn=False 用于**鸣牌后的打牌**: 碰/吃事件本身不带 operation_list,
    服务端也不再说 type=1, 但手牌变成 3n+2 就是该我们打一张.
    `ev_rows` 是 `ev_eval` 的候选(最优在前); 有它时按它选, 没有再回牌效.
    """
    if not state.round_open or state.my_seat is None:
        return Decision(WAIT, reason="no_round")
    if require_turn and not state.protocol_says_my_turn():
        return Decision(WAIT, reason="not_my_turn")
    if len(state.hand) % 3 != 2:
        return Decision(WAIT, reason=f"bad_count:{len(state.hand)}")
    if _is_riichi(state):
        # 立直后雀魂自动摸切; 我们任何点击都不该发生.
        return Decision(WAIT, reason="riichi_auto")

    allowed = _lizhi_tiles(state) if allow_riichi else []
    if allowed:
        tile = choose(state, allowed, rng, ev_rows)
        return Decision(RIICHI, tile=tile, tiles=tuple(allowed),
                        reason="lizhi")
    tile = choose(state, None, rng, ev_rows)
    return Decision(DISCARD, tile=tile,
                    reason="ev" if ev_rows else "efficiency")


def plan_response(state: GameState, *, rng: random.Random | None = None,
                  allow_riichi: bool = True,
                  ev_rows: Sequence | None = None) -> Decision:
    """当前 operation_list 是给我的: 和/立直/拔北/杠/碰/跳过/出牌."""
    keys = set(state.offered_button_keys())
    if not keys:
        return Decision(WAIT, reason="no_options")

    if keys & {"he", "zimo"}:
        return Decision(WIN, reason="win")

    if "lizhi" in keys and allow_riichi:
        allowed = _lizhi_tiles(state)
        if allowed:
            tile = choose(state, allowed, rng, ev_rows)
            return Decision(RIICHI, tile=tile, tiles=tuple(allowed),
                            reason="lizhi")
        return Decision(WAIT, reason="lizhi_without_allowed_tiles")

    if "babei" in keys:
        return Decision(BABEI, reason="babei")

    if "gang" in keys:
        gang = _pick_gang(state)
        if gang is not None:
            return gang

    if "peng" in keys:
        peng = _pick_peng(state)
        if peng is not None:
            return peng

    if state.protocol_says_my_turn():
        return plan_discard(state, rng=rng, allow_riichi=allow_riichi,
                            ev_rows=ev_rows)

    if keys & {"chi", "peng", "gang", "he"}:
        return Decision(SKIP, reason="decline_call")
    return Decision(WAIT, reason="nothing")


def _is_riichi(state: GameState) -> bool:
    if state.my_seat is None:
        return False
    player = state.players.get(state.my_seat)
    return bool(player and player.riichi)


def _ops(state: GameState, op_type: int) -> list[dict]:
    out = []
    for op in state.offered:
        if not isinstance(op, dict):
            continue
        try:
            if int(op.get("type", -1)) != op_type:
                continue
        except (TypeError, ValueError):
            continue
        out.append(op)
    return out


def _combos(op: dict) -> list[tuple[str, ...]]:
    """OptionalOperation.combination -> 去重后的牌组合.

    吃是 "1s|3s"(一种吃法), 碰/杠/立直是单张; 两种都按 "|" 拆.
    """
    out: list[tuple[str, ...]] = []
    for item in op.get("combination") or []:
        tiles = tuple(t for t in str(item).split("|") if is_tile(t))
        if tiles and tiles not in out:
            out.append(tiles)
    return out


def _lizhi_tiles(state: GameState) -> list[str]:
    """立直时可以打的牌(type=7 的 combination). 没给就返回空.

    返回**34 类口径**(赤 5 "0m" 折成 "5m"): 下游拿它跟 jansou 的牌种比较
    (见 `discard.choose`), 两边写法不统一会把白名单过滤成空.
    """
    out: list[str] = []
    for op in _ops(state, OP_LIZHI):
        for combo in _combos(op):
            kind = normalize(combo[0])
            if kind not in out:
                out.append(kind)
    return out


def _pick_peng(state: GameState) -> Decision | None:
    yaku = yaku_pair_kinds(list(state.hand), 0, state.field_wind())
    if not yaku:
        return None
    for op in _ops(state, OP_PENG):
        for combo in _combos(op):
            if normalize(combo[0]) in yaku:
                return Decision(PENG, tiles=combo,
                                reason=f"yaku_pair:{normalize(combo[0])}")
    return None


def _pick_gang(state: GameState) -> Decision | None:
    for op in state.offered:
        if not isinstance(op, dict):
            continue
        try:
            kind = int(op.get("type", -1))
        except (TypeError, ValueError):
            continue
        # 大明杠破门清, 规则上不做(用户拍板的三条硬规则之一).
        if kind == OP_MINGGANG:
            continue
        if kind not in (OP_ANKAN, OP_JIAGANG):
            continue
        combos = _combos(op)
        if not combos:
            combos = _fallback_kan_combos(state, kind)
        for combo in combos:
            tile = normalize(combo[0])
            if kan_keeps_shape(tile, list(state.hand)):
                return Decision(GANG, tiles=combo,
                                reason=f"kan_type:{kind}")
    return None


def _fallback_kan_combos(state: GameState, kind: int) -> list[tuple[str, ...]]:
    """协议没给 combination 时的兜底: 从手牌/副露里找出能杠的那张.

    正常报文都带 combination; 这里只防老版本/异常报文, 不猜具体组合.
    """
    if kind == OP_ANKAN:
        return [(t,) for t in dict.fromkeys(state.hand)
                if state.hand.count(t) >= 4]
    if kind == OP_JIAGANG:
        out = []
        for meld in state.melds:
            if not meld.tiles:
                continue
            tile = normalize(meld.tiles[0])
            if state.hand.count(tile) > 0 and (tile,) not in out:
                out.append((tile,))
        return out
    return []
