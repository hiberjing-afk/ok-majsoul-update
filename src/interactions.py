"""拟人化交互层: 在 PynputInteraction 之上只改"鼠标怎么过去".

为什么单独一层:

    ok-script 自带的 PynputInteraction 已经能点中雀魂(Unity uGUI 只认真实系统
    光标位置, 实测 PostMessage 系全部无效), 所以不需要重写输入通道. 但它点得
    "不像人": controller.position = (x, y) 一步瞬移, 然后立刻按下 -- 光标这一帧
    还在 A, 下一帧已经在 B 并且已经按下去了.

    这里只把"移动"这一段换成贝塞尔轨迹(src/mouse_path.py), 其余(前台判定,
    帧坐标 -> 屏幕坐标换算, 鼠标键映射)全部复用父类, 避免重复实现父类里那些
    踩过坑的细节.

为什么做成 ok-script 的 interaction 而不是写进任务里:

    后续的对局内打牌任务也要点(出牌 / 碰 / 杠 / 立直), 放在交互层能让所有任务
    自动获得同样的轨迹, 不用每个任务各写一遍. config.py 里把本类放进
    windows.interaction 列表即可生效(DeviceManager 支持列表里放类对象).

注意: 用了真实鼠标之后脚本会占用光标, 运行期间不要手动挪鼠标 / 切窗口.

前台这件事有一个必须知道的坑(踩过): ok-script 的 TaskExecutor 在
`execute()` 里先 `while self.paused: sleep(1)`, 而 `sleep()` 内部只有在
`interaction.should_capture()` 为真时才真正计时. Pynput 的 should_capture
就是 `is_foreground()`. 于是"游戏不在前台"会让整个 executor 卡在取任务之前,
任务体一次都跑不到 -- 任务里写多少 `bring_to_front()` 都没用, 因为代码根本没
机会执行. 所以拉前台必须做在**交互层**, 见下面的 `should_capture`.
"""

from __future__ import annotations

import math
import random
import time

from ok.device.interaction_methods.pynput import PynputInteraction
from ok.util.logger import Logger

from src.mouse_path import plan_path

logger = Logger.get_logger(__name__)

# 默认值. 想调参不用改代码: 在 configs/ 里覆盖, 或者构造时传 move_config.
DEFAULT_MOVE_CONFIG = {
    # ---- 轨迹形状 ----
    "curvature": 0.18,           # 控制点偏移量 / 距离, 越大弧度越明显
    "overshoot_prob": 0.25,      # 冲过目标再修正回来的概率
    "overshoot_px": (2.0, 9.0),  # 冲过头的距离范围(像素)
    "jitter_sigma": 0.6,         # 每个采样点的位置抖动(像素)
    "px_per_step": 28.0,         # 每个采样点大约走多少像素
    "min_steps": 8,
    "max_steps": 36,
    # ---- 时序 ----
    "pre_press_pause": (0.03, 0.10),  # 到位后, 按下之前停一下(人在瞄准)
    "down_time": (0.04, 0.11),        # 按住多久(父类写死 0.01s, 太短)
    # ---- 点击之后的小动作 ----
    "idle_drift_prob": 0.5,      # 点完之后随手挪开一点的概率(不点击)
    "idle_drift_px": (3.0, 8.0),  # 挪开的距离范围(像素)
    # ---- 前台 ----
    # 目标窗口不在前台时由交互层主动拉回来(见 should_capture 的说明).
    # 关掉它就退回 ok-script 原生行为: 不在前台时整个 executor 挂起,
    # 必须用户手动点一下游戏窗口才能继续.
    "auto_foreground": True,
    # 两次拉前台之间的最小间隔(秒). executor 每 0.1s 问一次 should_capture,
    # 不节流会疯狂调 SetForegroundWindow.
    "auto_foreground_interval": 2.0,
    # 单次拉前台的重试次数与间隔: 刚启动时窗口可能还没稳定, 一次不一定成.
    "auto_foreground_retries": 3,
    "auto_foreground_retry_delay": 0.4,
}


class HumanLikePynputInteraction(PynputInteraction):
    """Pynput + 拟人轨迹的点击.

    和父类的差别只有两处:
      1. 点击前用贝塞尔轨迹滑过去, 不是瞬移;
      2. 按下 / 按住 / 点后停顿都加了随机量.
    前台判定, 坐标换算这些一律沿用父类.
    """

    def __init__(self, capture, hwnd_window, move_config: dict | None = None):
        super().__init__(capture, hwnd_window)
        cfg = dict(DEFAULT_MOVE_CONFIG)
        if move_config:
            cfg.update(move_config)
        self.move_config = cfg
        self._rng = random.Random()
        self._last_foreground_pull: float = 0.0
        self._foreground_pull_failed: bool = False

    # ------------------------------------------------------------------
    # 前台
    # ------------------------------------------------------------------

    def should_capture(self) -> bool:
        """能不能继续抓帧 / 计时 -- 不在前台时**自己把窗口拉回来**.

        父类的实现是 `return self.clickable()`, 也就是"只看游戏在不在前台".
        问题在于 ok-script 的 executor 主循环:

            while not exit:
                if self.paused:
                    sleep(1)          # <- 卡在这里
                task = self.next_task()
                task.run()            # <- 永远到不了

        `sleep()` 内部只在 `should_capture()` 为真时才真正走时间, 所以游戏不在
        前台时 executor 连"取任务"这一步都做不完, 任务体一次都跑不到. 任务里
        写 `bring_to_front()` 没用 -- 那段代码还没开始执行. (实测: 16:57:04 点
        开始, 卡到 16:57:46 手动点一下游戏窗口才继续.)

        父类的 `on_run()` 里其实调了 `bring_to_front()`, 但它在 `next_task()`
        之后, 同样够不到. 唯一能拦住的位置就是这里, 所以在这一层拉前台.

        拉成功就回答 True, executor 立刻恢复计时; 用户中途切走也会被拉回来.
        """
        if self.clickable():
            self._foreground_pull_failed = False
            return True
        if not self.move_config.get("auto_foreground", True):
            return False
        if self._pull_foreground():
            self._foreground_pull_failed = False
            return True
        # 拉不过来说明窗口可能被最小化/锁屏/UAC 挡住了, 这时不能一直狂刷,
        # 但也必须明确报出来 -- 之前"任务不动"没有任何提示, 很难查.
        if not self._foreground_pull_failed:
            self._foreground_pull_failed = True
            logger.warning(
                "游戏窗口不在前台且拉不过来，脚本已暂停等待；"
                "请确认雀魂窗口没有被最小化，也没有被其他程序挡住")
        return False

    def _pull_foreground(self, force: bool = False) -> bool:
        """把游戏窗口拉到前台(带节流与重试). 成功返回 True.

        force=True 时跳过节流 -- 点击前用. 点击是低频动作, 不需要为它省,
        而省下来的一次机会恰好是"两轮之间用户切走窗口"这种最需要补救的情况.
        """
        now = time.time()
        interval = self.move_config["auto_foreground_interval"]
        if not force and now - self._last_foreground_pull < interval:
            return False
        self._last_foreground_pull = now

        hwnd_window = getattr(self, "hwnd_window", None)
        if hwnd_window is None:
            return False

        retries = int(self.move_config["auto_foreground_retries"])
        for attempt in range(retries):
            try:
                if hwnd_window.is_foreground():
                    return True
                hwnd_window.bring_to_front()
                # 焦点切换有动画, 立刻查可能还是旧窗口, 稍等再确认.
                time.sleep(self.move_config["auto_foreground_retry_delay"])
            except Exception as e:  # noqa: BLE001 - 拉前台失败不能拖垮点击
                logger.warning(f"拉游戏窗口到前台失败({e})")
                return False
            if hwnd_window.is_foreground():
                if attempt:
                    logger.info(f"游戏窗口已拉回前台(第 {attempt + 1} 次尝试)")
                return True
        return False

    # ------------------------------------------------------------------
    # 点击
    # ------------------------------------------------------------------

    def click(self, x=-1, y=-1, move_back=False, name=None, down_time=0.01,
              move=False, key="left"):
        """先把光标沿轨迹移到目标, 停一下, 再按下抬起.

        down_time 是调用方给的按住时长(BaseTask 默认 0.02s, 太短), 这里取
        max(调用方的值, 随机值): 调用方明确要长按不会被我们缩短.
        """
        from pynput import mouse

        if not self.clickable():
            # 父类在这里直接放弃点击, 但我们能自己把窗口拉回前台, 所以先试一次
            # (should_capture 已经拉过一次, 这一层是防止两轮之间用户切走了).
            if not self._pull_foreground(force=True) and not self.clickable():
                logger.info("窗口不在前台，拉不过来，跳过点击")
                return

        controller = mouse.Controller()
        current_pos = controller.position if move_back else None
        target = None

        if x != -1 and y != -1:
            target = self.capture.get_abs_cords(x, y)
            self._move_to(controller, target)
            time.sleep(self._uniform("pre_press_pause"))

        button = self.get_mouse_button(key)
        controller.press(button)
        time.sleep(max(down_time, self._uniform("down_time")))
        controller.release(button)

        if target is not None and self._rng.random() < self.move_config["idle_drift_prob"]:
            self._idle_drift(controller, target)

        if current_pos is not None:
            controller.position = current_pos

    # ------------------------------------------------------------------
    # 移动
    # ------------------------------------------------------------------

    def move(self, x, y):
        """任务里的 self.move(): 覆写父类的瞬移, 同样走贝塞尔轨迹.

        用途是"只挪光标不点击"(任务用它把光标停到安全区). 父类
        PynputInteraction.move 是 controller.position 一步瞬移, 不像人.
        """
        from pynput import mouse

        if not self.clickable():
            return
        controller = mouse.Controller()
        self._move_to(controller, self.capture.get_abs_cords(x, y))

    def _move_to(self, controller, target) -> None:
        """沿规划好的轨迹把光标移到 target(屏幕绝对坐标)."""
        try:
            start = controller.position
        except Exception:  # 读不到当前位置就退化成"从终点出发", 轨迹为空
            start = target

        stops = plan_path(
            start,
            target,
            rng=self._rng,
            curvature=self.move_config["curvature"],
            overshoot_prob=self.move_config["overshoot_prob"],
            overshoot_px=self.move_config["overshoot_px"],
            jitter_sigma=self.move_config["jitter_sigma"],
            px_per_step=self.move_config["px_per_step"],
            min_steps=self.move_config["min_steps"],
            max_steps=self.move_config["max_steps"],
        )
        for px, py, delay in stops:
            controller.position = (px, py)
            if delay > 0:
                time.sleep(delay)

    def _idle_drift(self, controller, target) -> None:
        """点完之后把光标随手挪开几像素(不再点击).

        人点完不会把光标钉死在原处. 这里只移动不点击, 所以落在哪儿都无害.
        """
        r = self._rng.uniform(*self.move_config["idle_drift_px"])
        angle = self._rng.uniform(0.0, math.tau)
        px = int(round(target[0] + r * math.cos(angle)))
        py = int(round(target[1] + r * math.sin(angle)))
        controller.position = (px, py)

    def _uniform(self, key: str) -> float:
        lo, hi = self.move_config[key]
        return self._rng.uniform(lo, hi)
