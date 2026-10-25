"""颜色判据的纯逻辑(不依赖 ok-script), 便于单元测试.

雀魂的 UI 用颜色区分"开关型按钮"的选中状态:

- 选中  -> 橙色 / 黄橙色底
- 未选中 -> 深蓝色底

实测(1080p 截图, ROI 取按钮矩形):

    =================== ============
    按钮                 橙色占比
    =================== ============
    四人麻将(选中)        0.891
    半庄(选中)            0.874
    创建按钮(橙色动作键)   0.799
    三人麻将(未选中)       0.002
    一局(未选中)           0.000
    东风(未选中)           0.000
    人机(未选中)           0.005
    =================== ============

选中态与未选中态差了两个数量级, 所以用一个很宽松的阈值就能可靠区分,
不会因为主题 / 亮度差异误判.

同一套思路也用于对局内左上操作条的自动开关: 开 = 亮绿字, 关 = 灰蓝字
(`is_toggle_green`).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 橙色在 OpenCV 的 HSV 里: H 为 0-179, 橙色约 8-35 度
ORANGE_HUE_RANGE = (8, 35)
ORANGE_SAT_MIN = 90
ORANGE_VAL_MIN = 120

Box = tuple[int, int, int, int]  # (x, y, w, h)


def _clip_box(frame: np.ndarray, box: Box) -> Box | None:
    """把 box 收窄到图像范围内. 完全越界时返回 None."""
    h, w = frame.shape[:2]
    x, y, bw, bh = (int(v) for v in box)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + bw), min(h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def orange_ratio(frame: np.ndarray, box: Box) -> float:
    """统计区域里"选中态橙色"的像素占比, 返回 0.0-1.0.

    frame 为空 / box 越界时返回 0.0(按"未选中"处理, 更保守: 宁可多点一次,
    也不要把已选中的按钮点掉).
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return 0.0
    clipped = _clip_box(frame, box)
    if clipped is None:
        return 0.0
    x, y, w, h = clipped
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lo, hi = ORANGE_HUE_RANGE
    mask = (hue >= lo) & (hue <= hi) & (sat > ORANGE_SAT_MIN) & (val > ORANGE_VAL_MIN)
    return float(mask.mean())


def is_selected_orange(frame: np.ndarray, box: Box, threshold: float = 0.5) -> bool:
    """判断开关型按钮是否已经是"选中(橙色)"状态.

    默认阈值 0.5, 落在实测的 0.005 与 0.874 之间, 有足够安全裕度.
    """
    return orange_ratio(frame, box) >= threshold


# 对局内左上操作条的开关(理/和/鸣/切/拔): 开=亮绿字, 关=灰蓝字.
# 绿字的 HSV 色相约 60~90, 灰蓝字饱和度低且偏蓝; 背景是暗藏蓝.
GREEN_HUE_RANGE = (35, 95)
GREEN_SAT_MIN = 80
GREEN_VAL_MIN = 110


def green_ratio(frame: np.ndarray, box: Box) -> float:
    """统计区域里"亮绿字"的像素占比, 返回 0.0-1.0.

    用于对局内自动开关(自动和了/不鸣牌/自动拔北)的状态判断:
    开=亮绿(实测算术: 理字绿字占比 ~0.05-0.13), 关=灰蓝(实测 ~0.000).
    越界/空图返回 0.0(按"关"处理: 开关点一次无害, 幂等).
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return 0.0
    clipped = _clip_box(frame, box)
    if clipped is None:
        return 0.0
    x, y, w, h = clipped
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lo, hi = GREEN_HUE_RANGE
    mask = (hue >= lo) & (hue <= hi) & (sat > GREEN_SAT_MIN) & (val > GREEN_VAL_MIN)
    return float(mask.mean())


def is_toggle_green(frame: np.ndarray, box: Box, threshold: float = 0.02) -> bool:
    """对局内自动开关是否已经打开(亮绿). 阈值 0.02 落在实测的 0.000 与 0.05+ 之间."""
    return green_ratio(frame, box) >= threshold


def page_changed(before: np.ndarray | None, after: np.ndarray | None,
                 threshold: float = 1.0) -> bool:
    """两张整帧画面是不是**翻了页**(缩到 64x36 灰度后比平均绝对差).

    只用于"点了一下之后页面动没动"这种**整页级**变化(翻页 = 大片像素同时
    换掉, 差值是几十这个量级), 阈值 1.0 只挡掉编码噪声. 拿不到可比的两帧
    (None / 尺寸不同)一律算"变了" —— 判不出来时不许声称"画面没动".

    **不能拿它判"对局里静止"**: 一张手牌只占整帧千分之几, 差 0.0004,
    和什么都没有完全区分不开(见下面卡死检测那段说明).
    """
    if before is None or after is None:
        return True
    before = np.asarray(before)
    after = np.asarray(after)
    if before.ndim != 3 or before.shape != after.shape:
        return True
    size = (64, 36)
    a = cv2.resize(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), size,
                   interpolation=cv2.INTER_AREA)
    b = cv2.resize(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), size,
                   interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(a, b))) > threshold


# ---- 卡死检测(纯逻辑, 不碰像素) ----
#
# 用途: 对局结束后会进结算页, 画面一动不动, 而牌河被牌面模型误识成"手牌"
# (实测 10 张, 刚好是 1 副露的合法张数), 于是每一轮都判成"在等别人", 一直
# 空转到 game_timeout(实跑踩过, 白等 15 分钟).
#
# **为什么不用"像素差"判静止**(试过, 不行): 缩略图平均差对"手牌多了一张"
# 这种局部变化几乎没反应(实测 0.0004, 因为一张牌只占整帧千分之几); 分块取
# 最大值也只在背景与牌面反差够大时才测得到 -- 结算页底部本来就是浅色台面,
# 在那里画一张白牌和什么都不画完全一样. 也就是说像素判据的漏检和误检都不可控.
#
# 改用**感知层面的"什么都没变"**: 手牌张数没变 + 认出的按钮集合没变, 而且
# 期间没有任何动作(没出牌也没点按钮). 这三个信号都是我们本来就在算的东西,
# 不引入新的敏感参数, 而且能直接写单测.
#
# 阈值取 40 轮: 对局里"在等别人"最长是等 3 家的回合计时(雀魂单回合约 6~10 秒,
# 托管上限 30 秒), 40 轮 × 0.3 秒 = 12 秒已经远超正常等待.
