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
#: 暗杠的行尾竖牌是全尺寸的(高度同手牌), 不受影响, 仍由
#: split_setaside(仅 _kan_pending 时)按大空档剥离。
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


def row_labels(boxes: Iterable[Detection], **kwargs) -> list[str]:
    """read_row 之后只要类别名, 用来直接拼手牌."""
    return [b.label for b in read_row(boxes, **kwargs)]


#: 手牌行里, 相邻两张牌的正常间距是牌宽的多少倍.
#:
#: 实测(2026-10-07, 真实对局帧逐帧量): 正常相邻间距在 **1.00~1.05** 倍之间
#: 非常稳定(牌是等间距排的), 所以"少了一张"会留下一道 ≥2 倍的空档 --
#: 1.5 这个门槛离两端都很远。
SLOT_GAP_RATIO = 1.5

#: 摸到的那张牌与手牌之间的固定间隔, 同样是牌宽的倍数.
#:
#: 实测 **2.37~2.39**(极稳), 也就是说它**本身就 ≥ SLOT_GAP_RATIO**, 必须
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
        if ratio < SLOT_GAP_RATIO:
            continue
        missing += max(1, int(round(ratio)) - 1)
    return missing


#: 暗杠牌与暗牌之间的间隔, 是牌宽的多少倍.
#:
#: 实测(2026-10-08 20:47 局 t340 帧): 暗杠的 4 张牌**竖着**摆在手牌行
#: 最右端, 与暗牌之间的空档 ≈ 4.8 倍牌宽. 对照组: 相邻牌 1.00~1.05 倍,
#: 摸切间隔 2.37~2.39 倍 —— 3.0 取在中间, 与两端都隔着一个量级.
SETASIDE_GAP_RATIO = 3.0


def split_setaside(boxes: Sequence[Detection],
                   min_ratio: float = SETASIDE_GAP_RATIO,
                   ) -> tuple[list[Detection], list[Detection]]:
    """把手牌行按**最后一道大空档**切成 (主体, 行尾副牌区).

    为什么需要(2026-10-08 20:47 局实锤): 自己暗杠后, 雀魂把 4 张杠牌竖着
    摆在手牌行最右端, 牌面模型把它们当成手牌一起读进来 —— 11 张 + 空档 6,
    张数、间距、出牌定位全被带歪(死锁一整手的根因). 吃/碰的副露是**横牌**,
    模型天然不认, 唯独暗杠竖牌混得进来.

    用法: 只在**我们自己刚点过杠**的状态(kan_pending)下调用, 丢弃第二段.
    大空档只认 ≥ min_ratio 倍牌宽的 —— 摸切间隔(2.4)挤不进来, 所以刚摸
    的牌留在主体里; 主体内部的漏认空档(1.5~2.5)也不会误切.

    返回 (主体框列表, 行尾副牌框列表); 没有大空档时副牌区为空.
    """
    if len(boxes) < 2:
        return list(boxes), []
    ordered = sorted(boxes, key=lambda b: b.x1)
    pitch = median([b.x2 - b.x1 for b in ordered])
    if pitch <= 0:
        return list(boxes), []
    cut = None
    for index, (left, right) in enumerate(zip(ordered, ordered[1:])):
        if (right.x1 - left.x1) / pitch >= min_ratio:
            cut = index + 1
    if cut is None:
        return list(boxes), []
    return ordered[:cut], ordered[cut:]


def group_by_label(boxes: Iterable[Detection]) -> dict[str, list[Detection]]:
    """按类别名分组, 组内按置信度降序. 认对局按钮时用."""
    groups: dict[str, list[Detection]] = {}
    for box in boxes:
        groups.setdefault(box.label, []).append(box)
    for items in groups.values():
        items.sort(key=lambda b: b.conf, reverse=True)
    return groups


def pick(boxes: Iterable[Detection], label: str,
         min_conf: float = 0.0) -> Detection | None:
    """取某个类别里置信度最高的框; 没有或置信度不够返回 None."""
    best: Detection | None = None
    for box in boxes:
        if box.label != label:
            continue
        if box.conf < min_conf:
            continue
        if best is None or box.conf > best.conf:
            best = box
    return best


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


# ------------------------------------------------- 立直宣言牌: 按高亮/变暗认
#
# **2026-10-06 重要更正(用户纠正 + 实跑日志复核)**:
# 之前这里写的是"立直后雀魂把宣言牌单独渲染得更矮(118px vs 132px)并放到
# 分离位", 据此写了 is_declaration(). 那个结论是**错的**, 两处都错:
#
# 1) "更矮" 测到的其实是**刚摸到的那张牌**. 复核 t76~t83 的真实框坐标:
#    t77(点立直那轮) 末位 3p 的 h=118, t78 起变成 h=128~129, 与其余牌同高.
#    也就是说 118px 是"刚摸的牌"的渲染样式, 立直与否都会出现, 且**只出现
#    那一轮** -- 拿它当宣言牌判据, 恰好在真正需要判宣言牌的时候不成立.
#
# 2) "分离位左边距" 同理. 之前记的"13 张时末位左边距 188~190px"是错的,
#    真实值是 96px(比值 1.01~1.03, 即**没有**分离). 那个 188~190 来自
#    统计脚本取了 max(gaps) 而不是 gaps[-1], 把中间某个异常值当成了末位.
#
# **真正的机理(用户指出, 与日志一致)**: 点下立直后雀魂进入"选牌"状态,
# **能打的牌高亮, 其余变暗**; 此时点任意一张高亮的牌即完成立直宣言.
# 日志佐证: t78~t82 每轮 buttons=[] -- 如果真锁定了必须打宣言牌, 一定会
# 出现确认/跳过按钮, 但没有. 而 t83 打出 2p 后手牌 14->13 成功, 且手牌
# 变成**已排序**(2p3p4p5p6p7p...), 那才是宣言成功的一刻.
# 换句话说 t78~t82 点不进去, 是因为点在**变暗**的牌上, 雀魂不响应.
#
# 所以判据换成: 找出**高亮**的那些牌, 而不是找"哪张长得特殊".
# 高亮牌通常不止一张(任何能组成立直的牌都可以选), 选哪张由牌效决定 --
# 立直宣言牌是"我们打出去的那张", 不是"游戏指定的那张".


def tile_brightness(frame, box, inset: float = 0.0) -> float | None:
    """取一个框内部的平均亮度(0~255); 裁不出来返回 None.

    只看牌面中央那块: 牌的边缘有描边和阴影, 立直栏(拿出来的牌)会盖住
    一部分, 边缘像素会把"变暗"稀释掉. inset 是四边各缩进的比例.
    """
    h = frame.shape[0]
    w = frame.shape[1]
    bh, bw = box.height, box.width
    if bh <= 0 or bw <= 0:
        return None
    dx, dy = bw * inset, bh * inset
    x1, y1 = int(box.x1 + dx), int(box.y1 + dy)
    x2, y2 = int(box.x2 - dx), int(box.y2 - dy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = frame[y1:y2, x1:x2]
    if patch is None or patch.size == 0:
        return None
    # 转灰度: 高亮/变暗是**亮度**变化, 与色相无关. 用 numpy 避开
    # 逐像素循环(14 张牌每帧都调, 不能慢).
    gray = patch[..., 0] * 0.299 + patch[..., 1] * 0.587 + patch[..., 2] * 0.114
    return float(gray.mean())


# 高亮组与变暗组之间的最小亮度差. 小于这个差就认为"没有明显分组"(不在
# 选牌态). 实测标定见 tests/TestDetection.py::TestBrightSplit, 改这个数
# 之前必须带负样本验: 正常对局(所有牌一样亮)绝不能被分成两组.
BRIGHT_GAP = 12.0


def split_bright_dim(boxes: Sequence[Detection], brightness: Sequence[float],
                     gap: float = BRIGHT_GAP) -> tuple[list[int], list[int]]:
    """把手牌分成"高亮"和"变暗"两组, 返回两组的下标.

    **做法: 在"最大亮度间隙"处切开, 而不是拿中位数加固定偏移.**
    后者有个隐蔽的失效场景: 高亮牌占多数时中位数落在亮组, 于是
    "中位数 + gap" 把整组都排除, 分组直接崩掉(实测 11 亮 3 暗 ->
    两组全空). 而"最大间隙"对哪一组是多数都成立: 它只看相邻亮度值之间
    落差最大的那个位置, 正好就是亮暗之间的那道坎.

    排序后找到相邻亮度落差最大的下标 k. order 是**升序**, 所以亮牌在
    order[k+1:] 那一侧, 暗牌在 order[:k+1]. 落差不够 gap 就返回两个空组
    -- 调用方据此判定"不在选牌态".

    返回的下标按**原始位置**升序(不是亮度顺序), 因为调用方拿它当点击
    候选集用, 顺序不该携带"谁更亮"的额外含义.
    """
    n = len(brightness)
    if n < 3:
        return [], list(range(n))
    order = sorted(range(n), key=lambda i: brightness[i])
    values = [brightness[i] for i in order]
    best_k, best_drop = -1, 0.0
    for k in range(n - 1):
        drop = values[k + 1] - values[k]
        if drop > best_drop:
            best_drop, best_k = drop, k
    if best_k < 0 or best_drop < gap:
        return [], []
    return sorted(order[best_k + 1:]), sorted(order[:best_k + 1])


# ------------------------------------------------------- 低置信度闸门


# 低于这个置信度的牌算"没认准".
#
# **这个阈值仍按"盖住全部已证实误识"来定, 2026-10-07 用新配置重测过, 结论是
# 保持 0.78 不动。**
#
# 原来 0.78 的来历(2026-10-06 那一局, 456 帧有手牌的帧): 把"已证实误识"
# (出牌后手牌张数掉超过 1 张, 见 PlayOneGameTask._discard)的 6 轮挑出来,
# 它们每轮**最低**一张牌的置信度是 0.42/0.42/0.43/0.46/0.66/0.74 -- 全部
# ≤0.74. 而最低置信度 ≥0.80 的 4 轮(turn 83/106/111/121)**没有一个是误识**.
#
# 那批数据是在牌面被压到 **49px** 的 TILE_IMGSZ=(224,1024) 下采的, 整手牌
# conf 中位数只有 0.482 -- 也就是"0.78"曾是在一个整体偏低的分布上划的线.
# 怀疑过它在新配置下会偏严, 所以 2026-10-07 拿 192 张对局帧(2601 张牌)重测,
# 当前 TILE_IMGSZ=(256,1024):
#
#   conf 中位 0.926      最低 0.152
#   <0.30  3.5%   [0.3,0.5) 5.3%   [0.5,0.6) 3.5%   [0.6,0.7) 2.3%
#   [0.7,0.8) 4.2%   [0.8,0.9) 17.6%   >=0.90 63.6%
#
# 分布是**两头重中间轻**: 63.6% 落在 ≥0.90(认得很准), 另有 8.8% 掉在 <0.50
# (基本可以断定认错). 0.78 正好切在中间那段"说不清"的少数派上, 平均每帧
# 只有 2.18/13.5 张落在它下面 —— 也就是"偶尔要谨慎一下", 不是"每轮都在猜".
# 拿它当闸门是合适的, 所以**不改**.
#
# 为什么不放宽: 已证实误识里最高的那张是 0.74, 阈值一旦降到 0.70 就盖不住
# 它了 -- 而"盖住全部已证实误识"是目前唯一有证据支撑的判据. 拿"少触发几次
# 稳健出牌"(只是多花 20~170ms)去换"漏掉一个已证实的误识"(结果是打错牌),
# 这笔账始终不划算.
#
# 注意这不是"低于它的牌一定认错", 而是"低于它的槽位不能全信". 这些槽位
# 会喂给稳健出牌(见 src/mahjong/brain.py::robust_discard): 枚举它们可能的真实
# 取值, 挑一张怎么认都不踩坑的牌打. 另外三麻里"这局不存在的牌"(2m-8m)是
# **独立于置信度**的第二类误识信号, 见 src/mahjong/tiles.py::out_of_play_indices.
#
# 踩过的坑: 早先想的是"低置信就整轮不出牌, 等下一帧认准", 数据上已证伪并
# 删掉 -- 位置 4 在 turn 176~185 连续 10 轮输出同一个错类别, conf 稳定在
# 0.64~0.67, 等多久都不恢复; 那样会有 73% 的时间在空等, 节奏全拖垮. 更关键
# 的是误识牌从不被直接点(6 个已证实误识里只有 1 个点了低置信那张), 真正的
# 危害是它污染了"哪张该打"的全局判断 -- 防护必须做在决策层, 不是点击层.
MIN_TILE_CONF = 0.78


def low_confidence(boxes: Sequence[Detection],
                   threshold: float = MIN_TILE_CONF) -> list[int]:
    """返回置信度低于阈值的槽位下标(升序); 全部达标返回空表.

    **这个闸门是为了防"牌效算错"这件事, 不是为了防"认不出画面".**

    误识本身是静默的: 牌面模型认错了牌, 画面上什么异常都看不出来, 系统
    会**认真**拿这手错牌去算牌效, 然后精确地打出一张"最优但根本不存在
    在屏幕上"的牌. 实跑的现场(turn 185): 真手牌是 789p, 模型把第二张 7p
    读成了 7p(conf 0.66), 于是"789p"变成"779p" -- 9p 成了孤张, 牌效
    算出"打 9p 最优"(向听1 进张7), 打出去直接退向. 而真实手牌下最优是
    打 3z(向听1 进张21). 日志里 25 轮出牌回放, 22 轮与复算完全一致,
    3 轮差异都能被"次优/随机"档位解释 -- **决策层可复现, 错的是输入**.

    与多帧投票互补: 投票只在"多帧不一致"时改变结果, 而这里认错的那张
    牌可能**连续十几帧都稳定地认成同一个错类别**(实测位置 4 有 18/19/17
    帧的连续低置信段), 所以投票救不了, 只能把低置信槽位交给决策层的
    稳健出牌(枚举可能取值, 挑一张怎么认都不踩坑的).

    早先的方案是"低置信就整轮不出牌, 等下一帧认准", 已经被数据证伪并删掉,
    原因见上面 MIN_TILE_CONF 那段注释(等不回来 + 73% 时间空等 + 误识牌从不被
    直接点). 这里的定位是**筛出不可信的槽位**喂给稳健出牌, 不是拦住这一轮.
    """
    return [i for i, b in enumerate(boxes) if b.conf < threshold]


# ---------------------------------------------------------------- 多帧投票


def vote_labels(frames: Sequence[Sequence[Detection]],
                min_conf: float = 0.0) -> list[str] | None:
    """对同一个槽位做多帧多数表决, 压掉单帧的低置信度误识别.

    **为什么需要(2026-10-06 实跑发现)**: 牌面模型对某些牌型(比如被选中
    抬起, 或者被立直栏的牌挤住的那张)置信度会掉到 0.6~0.7, 同一张牌在
    连续帧里会在两个类别之间来回跳. 实测 index=4 的位置跳变率 2.2%,
    是全手牌里最高的(其余位置多在 0.3%)-- 于是系统会认真算出一张
    "牌效最优但根本不是那张牌"的牌去打, 实跑里连续 5 轮点同一张牌
    都打不出去(手牌数一直是 14).

    为什么不靠调高 TILE_CONF 解决: 实测 1244 个框里真牌最低 0.433,
    而0.40~0.75 这一档里**既有真牌也有误识**, 单纯抬高阈值会把真牌丢掉,
    丢掉一张牌的危害比认错更严重(张数直接不对, 整轮判成"认漏了").
    投票是只在"低置信度"时才出手的补丁, 不动主阈值.

    frames 是同一帧位置的多张截图各自跑完 read_row 的结果(从左到右).
    返回值每槽**一个类别名**, 取该槽票数最多的类别; 票数平手时取整体置信度
    最高的那个框的类别, 也就是"没有任何异议就照用".

    **只返回标签, 不返回框**(2026-10-07 改): 原来返回"该类别里置信度最高的
    那个框", 而那个框**可能来自好几帧之前**. 于是 jsonl 里记的 `hand_conf` /
    `hand_boxes` 根本不是当前帧的测量值 —— 拿留证帧重跑模型对不上(实测 t164
    记录写 8s 置信 0.43, 重跑却是 5m 0.53), 日志还会先报 `8s->5m` 再报
    `5m->8s` 自己打自己, 事后复盘直接被带偏. 几何和置信度是**当前帧**的
    测量事实, 投票只该动"这张牌是什么"这一件事.
    """
    if not frames:
        return None
    usable = [f for f in frames if f]
    if not usable:
        return None

    width = min(len(f) for f in usable)
    if width == 0:
        return None

    out: list[str] = []
    for i in range(width):
        column = [f[i] for f in usable if len(f) > i
                  and f[i].conf >= min_conf] or [f[i] for f in usable
                                                if len(f) > i]
        if not column:
            return None
        tally: dict[str, list[Detection]] = {}
        for box in column:
            tally.setdefault(box.label, []).append(box)
        out.append(max(tally, key=lambda lab: (len(tally[lab]),
                                               max(b.conf for b in tally[lab]))))
    return out
