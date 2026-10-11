"""拟人鼠标轨迹规划(纯逻辑, 不依赖 pynput / ok-script, 方便单测).

为什么需要:
    ok-script 自带的 PynputInteraction.click() 是:

        controller.position = (abs_x, abs_y)   # 一步瞬移
        controller.press(button)               # 立刻按下

    也就是"闪烁点击": 光标这一帧还在 A, 下一帧已经在 B 并且已经按下去了.
    真人的鼠标不会瞬移. 这里只负责把"从 A 到 B 中间该经过哪些点, 每点停多久"
    算出来, 真正的输入动作在 src/interactions.py 里.

三条真人鼠标的可观测特征, 对应这里的三个设计点:

    1. 不是直线 -- 用三次贝塞尔曲线, 两个控制点沿"垂直于起终点连线"的方向
       随机推开, 于是轨迹带一点弧度, 且每次弧度不一样.
    2. 不是匀速 -- 按**时间**均匀采样 smoothstep(t) 之后的曲线. smoothstep
       首尾导数为 0, 所以同样的时间间隔内, 首尾走的距离短, 中间长,
       "慢-快-慢"的加减速自动成立, 不需要再单独算速度曲线.
    3. 偶尔过点 -- 小概率让曲线终点略微冲过目标, 再拉回来修正一下.

另外还有一条"能不能手动复现"的硬约束: 移动耗时必须随距离增长,
否则跨屏 1000px 和挪 30px 用一样的时间, 那是机器不是人.
travel_duration() 里 100px -> 约 0.15s, 1000px -> 约 0.64s(再被上限截断).
"""

from __future__ import annotations

import math
import random

# 距离 -> 耗时: base + 距离 * per_px, 再夹到 [min_duration, max_duration].
# 真人鼠标: 短距离挪一下 ~0.15s, 横跨整个屏幕 ~0.5-0.7s.
TRAVEL_BASE = 0.09
TRAVEL_PER_PX = 0.00055
TRAVEL_MIN = 0.12
TRAVEL_MAX = 0.55

# 每步大约走多少像素. 决定轨迹的"采样密度": 太稀会有明显的折线感,
# 太密则每一步都是一次 SendInput, 白白占用时间.
PX_PER_STEP = 28.0
MIN_STEPS = 8
MAX_STEPS = 36


def smoothstep(t: float) -> float:
    """3t^2 - 2t^3: 首尾导数为 0 的缓动函数, 用来把"时间"映射成"曲线参数"."""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def travel_duration(
    distance: float,
    *,
    base: float = TRAVEL_BASE,
    per_px: float = TRAVEL_PER_PX,
    min_duration: float = TRAVEL_MIN,
    max_duration: float = TRAVEL_MAX,
) -> float:
    """距离(像素) -> 移动耗时(秒). 单调递增, 两端有夹逼."""
    return min(max_duration, max(min_duration, base + distance * per_px))


def step_count(
    distance: float,
    *,
    px_per_step: float = PX_PER_STEP,
    min_steps: int = MIN_STEPS,
    max_steps: int = MAX_STEPS,
) -> int:
    """距离(像素) -> 轨迹采样点数. 至少 2 个点(起点 + 终点)."""
    n = int(round(distance / px_per_step)) + 1
    return max(2, min(max_steps, max(min_steps, n)))


def _cubic_bezier(p0, p1, p2, p3, t: float) -> tuple[float, float]:
    """三次贝塞尔在参数 t 处的坐标."""
    mt = 1.0 - t
    a = mt * mt * mt
    b = 3.0 * mt * mt * t
    c = 3.0 * mt * t * t
    d = t * t * t
    return (
        a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
        a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
    )


def plan_path(
    start: tuple[float, float],
    end: tuple[float, float],
    rng: random.Random | None = None,
    *,
    curvature: float = 0.18,
    overshoot_prob: float = 0.25,
    overshoot_px: tuple[float, float] = (2.0, 9.0),
    jitter_sigma: float = 0.6,
    px_per_step: float = PX_PER_STEP,
    min_steps: int = MIN_STEPS,
    max_steps: int = MAX_STEPS,
    overshoot_min_distance: float = 40.0,
    correction_delay: tuple[float, float] = (0.02, 0.05),
) -> list[tuple[int, int, float]]:
    """规划从 start 到 end 的鼠标轨迹.

    返回 [(x, y, sleep_after), ...]: 依次把光标移到 (x, y), 然后睡 sleep_after 秒.
    约定:

    - 最后一个点一定是 end(四舍五入后的整数坐标), 保证点击落点精确;
    - 最后一个点的 sleep_after 是 0.0, 点击前的停顿由调用方决定;
    - 总耗时约等于 travel_duration(距离), 只多不少(过点修正会额外加几十毫秒).

    超出屏幕的点不在这里夹逼: Windows 的 SetCursorPos 自己会把光标夹回屏幕内,
    硬夹反而会把轨迹压出一段贴边的直线, 看起来更假.
    """
    rng = rng if rng is not None else random
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx, dy = ex - sx, ey - sy
    distance = math.hypot(dx, dy)

    if distance < 1.0:
        # 已经在目标上了, 没有轨迹可言. 返回一个点, 调用方照常点击即可.
        return [(int(round(ex)), int(round(ey)), 0.0)]

    n = step_count(distance, px_per_step=px_per_step,
                   min_steps=min_steps, max_steps=max_steps)
    duration = travel_duration(distance)

    # 单位方向向量 + 它的法向量. 控制点沿法向量推开 -> 轨迹不是直线.
    ux, uy = dx / distance, dy / distance
    nx, ny = -uy, ux
    amp = curvature * distance * rng.uniform(-1.0, 1.0)
    c1 = (sx + dx * 0.30 + nx * amp * 0.70, sy + dy * 0.30 + ny * amp * 0.70)
    c2 = (sx + dx * 0.70 + nx * amp * 0.30, sy + dy * 0.70 + ny * amp * 0.30)

    # 过点: 让曲线的终点略微冲过目标, 最后再修正回来.
    overshoot = distance >= overshoot_min_distance and rng.random() < overshoot_prob
    if overshoot:
        r = rng.uniform(*overshoot_px)
        curve_end = (ex + ux * r, ey + uy * r)
    else:
        curve_end = (ex, ey)

    stops: list[tuple[int, int, float]] = []
    per_step = duration / (n - 1)
    for i in range(n):
        t = smoothstep(i / (n - 1))
        x, y = _cubic_bezier((sx, sy), c1, c2, curve_end, t)
        x += rng.gauss(0.0, jitter_sigma)
        y += rng.gauss(0.0, jitter_sigma)
        stops.append((int(round(x)), int(round(y)), per_step))

    if overshoot:
        # 曲线末端停在"冲过头"的位置, 这里补一步修正回真正的目标点.
        stops[-1] = (stops[-1][0], stops[-1][1], rng.uniform(*correction_delay))
        stops.append((int(round(ex)), int(round(ey)), 0.0))
    else:
        stops[-1] = (int(round(ex)), int(round(ey)), 0.0)

    return stops
