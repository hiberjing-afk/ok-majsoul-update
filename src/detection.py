"""YOLO 检测结果的后处理(纯逻辑, 不依赖 torch / ultralytics / ok-script).

参考实现 liqwang/majsoul-bot 用 supervision 做这一步; 我们没装那个包,
而且它的过滤规则写死在 detector 里没法单测, 所以自己实现一遍.

后处理解决三件事:

1. 剔除异常框. 同一个模型偶尔会在一张牌上叠两个框, 或者在牌桌背景上冒出
   一个又大又扁的框. 用宽 / 高 / 纵坐标的中位数做容差过滤.
2. 分槽. 手牌横着排一列, 同一张牌常被给出多个类别(比如 5p 和 6p 各一个框).
   按横坐标聚成一列一列的槽, 每个槽只留置信度最高的那个.
3. 排序. 从左到右输出, 和玩家看到的顺序一致.

输入统一是 `Detection`(左上角 / 右下角像素坐标 + 类别名 + 置信度),
模型相关的转换在 src/detector.py 里做, 这里只吃 / 只吐这个结构.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

# 异常框过滤的默认容差.
# 同一排手牌的牌面大小几乎一致, 偏离中位数 35% 以上基本可以断定是误检.
SIZE_TOL = 0.35
# 中心纵坐标偏离中位数超过 0.6 倍中位高就丢: 摸到的那张牌会稍微偏上一点,
# 但不会偏出一整个牌高.
Y_TOL = 0.60
# 框太少时中位数就是自己, 过滤会把所有框都留下来(等于没过滤),
# 但也可能因为容差计算退化而误杀, 所以干脆跳过.
MIN_BOXES_TO_FILTER = 4

# 分槽: 相邻两张牌的中心横坐标间距若小于 中位宽 * 这个比例, 就归到同一槽.
SLOT_GAP_RATIO = 0.70

#: 副露牌与手牌的框高比门槛: 框高不足最大框高这个比例的框按副露剔掉.
#:
#: 实测(2026-10-11 三碰局, frame_lowconf_1011_010343): 雀魂把自己的副露
#: **竖着摆在手牌行右端**(碰 1 组在 x≈1630-1870, 碰 3 组扩到 x≈1102-1870),
#: 牌面比手牌小一圈 —— 手牌框高 ≈131px, 副露框高 ≈76px ≈ 0.58 倍。
#: 手牌识别区(x≤1824)必然把副露区包进去, 牌面模型分不清"这是副露",
#: 会把副露牌一起读进来(横放的刚碰动画牌模型不认, 竖着的成排副露全认)。
#:
#: 为什么 filter_outliers 兜不住: 它按**中位数**剔离群值 —— 副露 ≤2 组时
#: 副露框占少数, 中位数偏向手牌, 副露框被当离群值剔掉(碰 1/碰 2 都正常);
#: **碰 3 组后副露框(8)多于手牌框(5), 中位数倒向副露, 真手牌 5 张全被
#: 当离群值剔光**, 读数变成纯副露内容 [6z,6z,3s,3s,3s,5s,5s,5s] ——
#: 决策层于是反复想打已经碰掉的牌, 点副露雀魂不响应, 死循环到本局结束。
#:
#: 判据因此必须与**数量无关**: 只留框高 ≥ 最大框高这个比例的框。
#: 副露(0.58)永远被剔, 手牌(1.0)永远保留, 副露多少组都一样。
#: 暗杠的行尾竖牌是全尺寸的(高度同手牌), 不受影响; 它由任务层在
#: "自己刚杠过"的状态下按大空档单独剥离.
MELD_HEIGHT_RATIO = 0.75


def drop_meld_tiles(boxes: Iterable[Detection],
                    max_ratio: float = MELD_HEIGHT_RATIO) -> list[Detection]:
    """剔掉混进手牌读数里的**副露牌**(框高明显小于手牌的框).

    输入是 `self.tiles.detect()` 的原始输出(未分槽); 输出喂给
    `read_row`。必须在 filter_outliers **之前**做 —— 那一步基于中位数,
    副露框数量一多中位数就倒向副露, 会把真手牌剔光(见 MELD_HEIGHT_RATIO
    的说明)。

    手牌一张都没检出、只有副露框时原样返回(剔无可剔, 张数守卫会兜住)。
    """
    items = list(boxes)
    if len(items) < 2:
        return items
    tallest = max(b.height for b in items)
    kept = [b for b in items if b.height >= max_ratio * tallest]
    return kept or items


@dataclass(frozen=True)
class Detection:
    """一个检测框. 坐标是像素, 相对于**传入模型的那张图**的左上角."""

    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    # ---- ok-script Box 的兼容别名 ----
    # 任务基类的 _random_point_in_box 读的是 Box 的 x / y / width / height,
    # YOLO 的框也直接丢给它用(对局内点手牌 / 点按钮), 所以补上整数化的 x / y.
    @property
    def x(self) -> int:
        return int(round(self.x1))

    @property
    def y(self) -> int:
        return int(round(self.y1))

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    def shifted(self, dx: float = 0.0, dy: float = 0.0) -> "Detection":
        """把坐标平移到另一套坐标系(比如裁出来的手牌区 -> 整帧)."""
        if dx == 0.0 and dy == 0.0:
            return self
        return replace(self, x1=self.x1 + dx, y1=self.y1 + dy,
                       x2=self.x2 + dx, y2=self.y2 + dy)

    def scaled(self, sx: float, sy: float) -> "Detection":
        """把坐标按比例缩放(模型输入被 resize 过时用)."""
        if sx == 1.0 and sy == 1.0:
            return self
        return replace(self, x1=self.x1 * sx, y1=self.y1 * sy,
                       x2=self.x2 * sx, y2=self.y2 * sy)


def median(values: Sequence[float]) -> float:
    """中位数. 空序列返回 0(调用方自己保证不会拿 0 当除数)."""
    data = sorted(float(v) for v in values)
    if not data:
        return 0.0
    mid = len(data) // 2
    if len(data) % 2:
        return data[mid]
    return (data[mid - 1] + data[mid]) / 2.0


def filter_outliers(boxes: Iterable[Detection],
                    size_tol: float = SIZE_TOL,
                    y_tol: float = Y_TOL,
                    min_boxes: int = MIN_BOXES_TO_FILTER) -> list[Detection]:
    """丢掉尺寸 / 位置明显不合群的框.

    判据全部基于中位数, 这样即使来了几个离谱的框也不会把阈值带跑偏.
    """
    items = list(boxes)
    if len(items) < min_boxes:
        return items

    mw = median([b.width for b in items])
    mh = median([b.height for b in items])
    my = median([b.cy for b in items])
    if mw <= 0 or mh <= 0:
        return items

    kept = []
    for box in items:
        if abs(box.width - mw) > size_tol * mw:
            continue
        if abs(box.height - mh) > size_tol * mh:
            continue
        if abs(box.cy - my) > y_tol * mh:
            continue
        kept.append(box)
    # 极端情况下全被过滤掉(说明这堆框本来就不成形状), 宁可原样返回.
    return kept or items


def group_slots(boxes: Iterable[Detection],
                gap_ratio: float = SLOT_GAP_RATIO) -> list[list[Detection]]:
    """按中心横坐标把框聚成一列一列的槽(每张牌一个槽).

    先按 cx 升序, 再线性扫描: 和槽内已有框的最右中心距离超过阈值就开新槽.
    阈值取 中位宽 * gap_ratio -- 手牌是等距排的, 牌间距略小于牌宽,
    用 0.7 倍牌宽能稳稳地把相邻两张分开又不会把同一张的多个框拆开.
    """
    items = sorted(boxes, key=lambda b: b.cx)
    if not items:
        return []

    gap = median([b.width for b in items]) * gap_ratio
    slots: list[list[Detection]] = [[items[0]]]
    rightmost = items[0].cx
    for box in items[1:]:
        if box.cx - rightmost > gap:
            slots.append([box])
        else:
            slots[-1].append(box)
        rightmost = max(rightmost, box.cx)
    return slots


def best_per_slot(slots: Iterable[Sequence[Detection]]) -> list[Detection]:
    """每个槽留置信度最高的那个框, 按槽的顺序(即从左到右)返回."""
    out = []
    for slot in slots:
        if not slot:
            continue
        out.append(max(slot, key=lambda b: b.conf))
    return out


def read_row(boxes: Iterable[Detection], **kwargs) -> list[Detection]:
    """一整条后处理: 剔除异常框 -> 分槽 -> 每槽取最优. 从左到右."""
    return best_per_slot(group_slots(filter_outliers(boxes, **kwargs)))


#: 手牌行里"漏认"的判据: 相邻两张间距达到牌宽的这么多倍, 那里就少了牌.
#:
#: 实测(2026-10-07, 真实对局帧逐帧量): 正常相邻间距在 **1.00~1.05** 倍之间
#: 非常稳定(牌是等间距排的), 所以"少了一张"会留下一道 ≥2 倍的空档 --
#: 1.5 这个门槛离两端都很远。
MISSING_GAP_RATIO = 1.5

#: 摸到的那张牌与手牌之间的固定间隔, 同样是牌宽的倍数.
#:
#: 实测 **2.37~2.39**(极稳), 也就是说它**本身就 ≥ MISSING_GAP_RATIO**, 必须
#: 单独减掉, 否则每一轮都会被误判成"少了一张"。这是这套判据唯一的坑。
DRAWN_TILE_OFFSET = 1.4


def missing_slots(boxes: Sequence[Detection]) -> int:
    """手牌行里"少了几张牌" —— 由**间距**推断, 和置信度无关.

    为什么需要它(2026-10-07 实跑, 用户报的"打着打着就不打了"):

        牌面模型漏认 3~4 张时, 认出来的张数(10 张)恰好落进
        `valid_concealed_counts`(10 = 1 副露 13 张), 于是 `is_plausible_
        concealed` 认为"认得挺全", `is_my_discard_turn(10)` 为假, 决策层
        输出一句 **"在等别人"** 就完事了 —— 而 "在等别人" 的 suspect 是
        False, 上层既不计入 unknown_streak 也不存帧, 于是一路静默空转到
        40 轮的卡死兜底才报错。

    牌是**等间距**排的, 所以"漏认"在几何上是可见的: 只要有两张相邻的牌
    之间出现 ≥1.5 倍牌宽的空档, 那里就一定少了牌。这个信号**不依赖置信度**,
    因此置信度门槛怎么调都不会把它弄丢。

    末尾那段间隔要减掉摸牌偏移(DRAWN_TILE_OFFSET), 那一段是正常的(见常量
    说明)。

    返回"少了几张"的估计值; 0 表示这条手牌行的间距是连续的。
    """
    if len(boxes) < 2:
        return 0
    ordered = sorted(boxes, key=lambda b: b.x1)
    pitch = median([b.x2 - b.x1 for b in ordered])
    if pitch <= 0:
        return 0

    missing = 0
    last = len(ordered) - 2
    for index, (left, right) in enumerate(zip(ordered, ordered[1:])):
        ratio = (right.x1 - left.x1) / pitch
        if index == last:
            ratio -= DRAWN_TILE_OFFSET        # 末尾是刚摸的牌, 间隔本来就大
        if ratio < MISSING_GAP_RATIO:
            continue
        missing += max(1, int(round(ratio)) - 1)
    return missing


#: 暗杠牌与暗牌之间的间隔, 是牌宽的多少倍.
#:
#: 实测(2026-10-08 20:47 局 t340 帧): 暗杠的 4 张牌**竖着**摆在手牌行
#: 最右端, 与暗牌之间的空档 ≈ 4.8 倍牌宽. 对照组: 相邻牌 1.00~1.05 倍,
#: 摸切间隔 2.37~2.39 倍 —— 3.0 取在中间, 与两端都隔着一个量级.


def group_by_label(boxes: Iterable[Detection]) -> dict[str, list[Detection]]:
    """按类别名分组, 组内按置信度降序. 认对局按钮时用."""
    groups: dict[str, list[Detection]] = {}
    for box in boxes:
        groups.setdefault(box.label, []).append(box)
    for items in groups.values():
        items.sort(key=lambda b: b.conf, reverse=True)
    return groups


def region_pixels(width: int, height: int,
                  region: Sequence[float]) -> tuple[int, int, int, int]:
    """比例区域 (top, bottom, left, right) -> 像素边界 (y0, y1, x0, x1).

    边界会夹到画面内, 免得比例写错时裁出空图.
    """
    top, bottom, left, right = (float(v) for v in region)
    y0 = max(0, min(height, int(round(height * top))))
    y1 = max(0, min(height, int(round(height * bottom))))
    x0 = max(0, min(width, int(round(width * left))))
    x1 = max(0, min(width, int(round(width * right))))
    if y1 <= y0 or x1 <= x0:
        return (0, height, 0, width)
    return (y0, y1, x0, x1)
