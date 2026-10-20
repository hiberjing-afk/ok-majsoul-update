"""一手牌的事件驱动控制器.

它只做一件事: 协议事件进来, 该我动的时候产出一个 `Decision`.

一手从 `ActionNewRound` 开始, 到 `ActionHule/NoTile/LiuJu` 结束; 下一手自然
由下一个 `ActionNewRound` 重置, 所以不需要从画面猜"发牌完了没有/上一手结束
没有". 庄家开局 14 张的 `ActionNewRound` 自己就带 type=1(打牌)操作; 立直后
`state.players[my_seat].riichi` 为真, 后续摸牌一律不点(雀魂自动摸切), 新一手
的 `ActionNewRound` 会把它清掉.

视觉完全不在这里出现: 决策只吃协议, 坐标由 `locator` 在 runner 里补.
"""

from __future__ import annotations

import random
from typing import Callable

from src.net.state import GameState

from . import events
from .decisions import HAND_OVER, WAIT, Decision
from .rules import plan_discard, plan_response

CALL_KEYS = frozenset({"chi", "peng", "gang", "he"})
SPECIAL_KEYS = CALL_KEYS | frozenset({"lizhi", "babei", "zimo"})


class HandController:
    """一条协议事件 -> 一个决策. 状态只属于"这一手"."""

    def __init__(self, emit: Callable[[Decision], None] | None = None, *,
                 rng: random.Random | None = None,
                 allow_riichi: bool = True,
                 ev_provider: Callable[[], object] | None = None) -> None:
        self._emit_fn = emit
        self._rng = rng or random.Random()
        self._allow_riichi = allow_riichi
        self._ev_provider = ev_provider
        self.phase = "idle"
        self.decisions: list[Decision] = []
        self.pending: list[Decision] = []
        self._last_key: tuple | None = None
        self._new_round = False
        self.last_step = -1

    # ------------------------------------------------------------------ API

    def on_event(self, action: str, data: dict, step: int,
                 state: GameState) -> None:
        """`GameState.on_event` 的签名. 每个被消费的 Action 调一次."""
        self.last_step = step
        if action == events.NEW_ROUND:
            self.phase = "waiting"
            self._last_key = None
            self._new_round = True
            # 庄家开局就是 14 张(配牌 13 + 第一摸); 不依赖 type=1 是否下发.
            self._maybe_act(state, step, action,
                            force_turn=len(state.hand) % 3 == 2)
            return
        if action == events.DEAL_TILE:
            if events.is_mine(data, state):
                self.phase = "my_turn"
                # 摸牌事件本身就是"轮到我了"; operation_list 里的 type=1 只是
                # 第二份证据, 有的版本/局面(立直后)会省略它.
                self._maybe_act(state, step, action, force_turn=True)
            return
        if action == events.DISCARD_TILE:
            self.phase = "waiting"
            self._maybe_act(state, step, action)
            return
        if action in (events.CHI_PENG_GANG, events.AN_GANG_ADD_GANG,
                      events.BA_BEI):
            if events.is_mine(data, state):
                self.phase = "my_turn"
                # 碰/吃事件本身不带 operation_list; 手牌 3n+2 就是该我们打一张.
                # 杠/拔北之后会补一张, 等 ActionDealTile 再说.
                post_meld = (action == events.CHI_PENG_GANG
                             and events.is_post_meld_discard(data))
                self._maybe_act(state, step, action, post_meld=post_meld)
            return
        if action in events.HAND_OVER_ACTIONS:
            self.phase = "over"
            self._emit(Decision(HAND_OVER, reason=action), step, action)

    # ------------------------------------------------------------- internal

    def _maybe_act(self, state: GameState, step: int, trigger: str,
                   post_meld: bool = False, force_turn: bool = False) -> None:
        if self.phase == "over" or not state.round_open or state.my_seat is None:
            return

        keys = set(state.offered_button_keys())
        ev_rows = None
        if self._ev_provider is not None:
            try:
                ev_rows = self._ev_provider()
            except Exception:  # noqa: BLE001 - EV 算不出来就回牌效
                ev_rows = None
        # 先把"和/立直/拔北/杠/碰/跳过"这些特殊窗口处理掉. 正常摸牌只有
        # type=1, keys 为空, 才落到出牌. 优先级:
        # 和 > 立直 > 拔北 > 杠/碰 > 跳过 > 出牌.
        if (state.meld_options() or keys & SPECIAL_KEYS):
            decision = plan_response(state, rng=self._rng,
                                     allow_riichi=self._allow_riichi,
                                     ev_rows=ev_rows)
            if decision.action != WAIT:
                self._emit(decision, step, trigger)
                return
        if (state.protocol_says_my_turn() or post_meld or force_turn) \
                and len(state.hand) % 3 == 2:
            decision = plan_discard(
                state, rng=self._rng, allow_riichi=self._allow_riichi,
                require_turn=not (post_meld or force_turn),
                ev_rows=ev_rows)
            self._emit(decision, step, trigger)

    def _emit(self, decision: Decision, step: int, trigger: str) -> None:
        if decision.action == WAIT:
            return      # 等待不落日志/不重复执行
        key = (step, decision.action, decision.tile, decision.tiles)
        if key == self._last_key:
            return      # 同一条事件只产一次(防重复点击)
        self._last_key = key
        self.decisions.append(decision)
        self.pending.append(decision)
        if self._emit_fn is not None:
            self._emit_fn(decision)

    def pop_pending(self) -> list[Decision]:
        """取走尚未执行的决策(runner 每轮调用一次)."""
        out, self.pending = self.pending, []
        return out

    def pop_new_round(self) -> bool:
        """这一手是不是刚开始(新手第一次动作前点亮自动开关)."""
        if not self._new_round:
            return False
        self._new_round = False
        return True
