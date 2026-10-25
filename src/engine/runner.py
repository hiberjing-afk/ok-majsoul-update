"""把事件驱动内核接到一局真实对局上: 轮询协议 -> 执行决策.

它不负责导航(大厅/段位场/准备页仍由任务层做), 只负责**一局内**:

    grab frame -> 页面出口 -> net.poll()(事件进控制器) -> 执行决策

执行层只做三件事: 手牌槽位、按钮框、点击. 找不到框就下一轮重试, 超时
放弃并记录; 不再用视觉去"猜一个动作该不该做".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .decisions import (
    BABEI,
    CHI,
    DISCARD,
    GANG,
    HAND_OVER,
    PENG,
    RIICHI,
    SKIP,
    WAIT,
    WIN,
    Decision,
)
from .hand import HandController
from .locator import VisionLocator, slot_index

#: 一条决策执行失败(手牌槽位还在动/按钮没出现)最多重试多久. 开局发牌动画
#: 实测可到 4s 以上; 3s 会让第一条出牌直接放弃(2026-10-19 23:00 实跑).
EXECUTE_RETRY_SECONDS = 12.0
#: 主循环轮询间隔. 协议事件是 pull 的, 这个间隔决定响应延迟.
POLL_INTERVAL = 0.2
#: 坐标稳定窗: 框连续这么久没动才点. 只防"发牌/按钮滑入动画中点下去",
#: 不参与决策.
STABLE_SECONDS = 0.3
STABLE_TOL = 3.0


@dataclass
class RunnerHooks:
    """任务层需要提供的少数能力(全部是动作, 不含决策)."""

    grab: Callable[[], object]
    at_ready: Callable[[object], bool]
    result_page: Callable[[object], bool]
    hand_result: Callable[[object], bool]
    click_box: Callable[[object, str, str], None]
    should_stop: Callable[[], bool]
    sleep: Callable[[float], None]
    log: Callable[[str], None]
    new_round: Callable[[object], None] | None = None
    tick: Callable[[object], None] | None = None
    capture_alive: Callable[[], bool] | None = None
    discard_delay: Callable[[], float] | None = None
    clock: Callable[[], float] = time.time
    timeout: float = 1800.0


class EngineRunner:
    def __init__(self, net, controller: HandController, locator: VisionLocator,
                 hooks: RunnerHooks) -> None:
        self.net = net
        self.controller = controller
        self.locator = locator
        self.hooks = hooks
        self._slot_centers: tuple[float, ...] | None = None
        self._slot_since = 0.0
        self._button_center: float | None = None
        self._button_since = 0.0
        self._lizhi_done: tuple | None = None
        self._delayed_key: tuple | None = None
        self._poll_error: str | None = None

    def run(self) -> bool:
        """跑完一局. True = 正常回到准备页/结算页/整场结束."""
        deadline = time.time() + self.hooks.timeout
        retry: tuple[Decision, float] | None = None
        while time.time() < deadline:
            if self.hooks.should_stop():
                return False
            if (self.hooks.capture_alive is not None
                    and not self.hooks.capture_alive()):
                self.hooks.log("  [engine] 抓包链路已断，停止本局（不做视觉盲打）")
                return False
            frame = self.hooks.grab()
            if frame is None:
                self.hooks.sleep(POLL_INTERVAL)
                continue

            if self.hooks.at_ready(frame):
                return True
            if self.hooks.result_page(frame):
                return True
            if self.hooks.tick is not None:
                self.hooks.tick(frame)
            if self.controller.phase == "over":
                # 一手结束的顺位结算画面: 有「局确认」就点, 没有就等下一手
                # ActionNewRound 把 phase 重置.
                if self.hooks.hand_result(frame):
                    self.hooks.sleep(POLL_INTERVAL)
                    continue

            try:
                self.net.poll()
                self._poll_error = None
            except Exception as exc:  # noqa: BLE001 - 一条坏事件不能打掉整场
                # 决策层抛异常 = 这一轮的牌没算出来, 丢掉这一条决策就行: 雀魂
                # 倒计时到点会自动摸切, 下一个事件到来时状态已经翻页, 很可能
                # 就正常了(2026-10-21 00:53 一条 ValueError 停掉了整场).
                # 同一条错误只报一次, 免得每 0.2s 刷屏; poll 成功一次后复位.
                message = str(exc)
                if message != self._poll_error:
                    self._poll_error = message
                    self.hooks.log(
                        f"  [engine] 协议事件处理异常（丢掉这一轮，继续打）: {message}")
                self.controller.pop_pending()
                retry = None
                if getattr(getattr(self.net, "state", None), "game_over", False):
                    return True
                self.hooks.sleep(POLL_INTERVAL)
                continue
            state = getattr(self.net, "state", None)
            if getattr(state, "protocol_suspect", False):
                self.hooks.log("  [engine] 协议出现未建模动作，停手（不盲打）")
                return False

            if self.hooks.new_round is not None and self.controller.pop_new_round():
                # 新一手: 不管这一手有没有轮到我, 都要复位自动开关(游戏每手
                # 会把「和/鸣/拔」重置回关). 不能等到"第一个决策"才复位 ——
                # 非庄家开局可能整手都没有决策, 开关就一直是上一手的状态.
                self.hooks.new_round(frame)
                # 上一手没执行完的决策不能带过来: 牌已经换了一副, 照着旧决策
                # 点下去就是乱点(实测会把上一手的立直重放进新一手).
                retry = None
                self._clear_exec_state()
            pending = self.controller.pop_pending()
            if pending:
                retry = (pending[-1], time.time())   # 新决策覆盖旧的重试
            if retry is not None:
                ok, why = self._execute(retry[0], frame)
                if ok:
                    retry = None
                    self._clear_exec_state()
                elif time.time() - retry[1] >= EXECUTE_RETRY_SECONDS:
                    self.hooks.log(
                        f"  [engine] 执行 {retry[0].action} 失败（{why}），"
                        f"放弃这一条，等下一个事件")
                    retry = None

            if getattr(getattr(self.net, "state", None), "game_over", False):
                return True
            self.hooks.sleep(POLL_INTERVAL)
        self.hooks.log("  [engine] 本局超时")
        return False

    # ------------------------------------------------------------- execution

    def _clear_exec_state(self) -> None:
        """一条决策执行完之后, 清掉"这一步已经做过"的标记.

        这些标记唯一的作用是**重试时别重复点击**(立直按钮只点一次), 所以
        作用域是一次决策, 不是整场。以前从不清: 同一场里两次立直的
        (action, tile, tiles) 恰好一样时, 第二次会以为按钮点过了, 只点手牌
        ——立直就没立成(2026-10-21 01:33 实跑, 两次都是 `riichi 9s ['9s']`)。
        新一手和"执行成功"两个点都要清: 前者防跨手串味, 后者防这一手内出现
        第二次同 key 的立直。
        """
        self._lizhi_done = None
        self._delayed_key = None

    # -------------------------------------------------------------- stability

    def _stable_slots(self, boxes) -> bool:
        try:
            centers = tuple((float(b.x1) + float(b.x2)) / 2.0 for b in boxes)
        except (TypeError, AttributeError, ValueError):
            return False
        now = self.hooks.clock()
        prev = self._slot_centers
        if (prev is None or len(prev) != len(centers)
                or any(abs(a - b) > STABLE_TOL for a, b in zip(prev, centers))):
            self._slot_centers = centers
            self._slot_since = now
            return False
        return now - self._slot_since >= STABLE_SECONDS

    def _stable_button(self, box) -> bool:
        try:
            center = (float(box.x1) + float(box.x2)) / 2.0
        except (TypeError, AttributeError, ValueError):
            return False
        now = self.hooks.clock()
        if (self._button_center is None
                or abs(self._button_center - center) > STABLE_TOL):
            self._button_center = center
            self._button_since = now
            return False
        return now - self._button_since >= STABLE_SECONDS

    def _execute(self, decision: Decision, frame) -> tuple[bool, str]:
        if decision.action in (WAIT, HAND_OVER):
            return True, ""
        state = getattr(self.net, "state", None)
        if state is None:
            return False, "no_state"

        if decision.action in (DISCARD, RIICHI):
            key = (decision.action, decision.tile, decision.tiles)
            if self.hooks.discard_delay is not None and self._delayed_key != key:
                self._delayed_key = key
                delay = max(0.0, float(self.hooks.discard_delay()))
                if delay > 0:
                    self.hooks.log(
                        f"  [engine] 想 {delay:.1f}s 后 "
                        f"{'立直' if decision.action == RIICHI else '打'} "
                        f"{decision.tile or ''}")
                    self.hooks.sleep(delay)
            if decision.action == RIICHI:
                key = (decision.action, decision.tile, decision.tiles)
                if self._lizhi_done != key:
                    box = self.locator.button_box(frame, "lizhi")
                    if box is None:
                        return False, "没有立直按钮框"
                    if not self._stable_button(box):
                        return False, "立直按钮还在动"
                    self.hooks.click_box(box, "按钮", "lizhi")
                    self._lizhi_done = key
                    self.hooks.sleep(POLL_INTERVAL)
                frame = self.hooks.grab()
                if frame is None:
                    return False, "立直后抓帧失败"
            boxes = self.locator.hand_boxes(frame, len(state.hand))
            if boxes is None:
                return False, f"手牌槽位不是 {len(state.hand)} 个"
            if not self._stable_slots(boxes):
                return False, "手牌槽位还在动"
            index = slot_index(state, decision.tile or "")
            if index is None or index >= len(boxes):
                return False, f"要打的 {decision.tile} 不在手牌槽位里"
            self.hooks.click_box(boxes[index], "手牌", decision.tile or "")
            return True, ""

        kind = {
            PENG: "peng", GANG: "gang", CHI: "chi",
            WIN: "win", BABEI: "babei", SKIP: "skip",
        }.get(decision.action)
        if kind is None:
            return True, ""
        box = self.locator.button_box(frame, kind)
        if box is None:
            return False, f"没有 {kind} 按钮框"
        if not self._stable_button(box):
            return False, f"{kind} 按钮还在动"
        self.hooks.click_box(box, "按钮", kind)
        return True, ""
