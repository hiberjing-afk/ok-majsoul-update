"""吃法选择条的划分: 从一堆牌框里切出"N 个选项", 再和协议组合对上号.

## 真实布局(2026-10-17 真帧实证, chiopt_1017_141351/142045/142115)

一次吃有**多种吃法**时(手里 3s4s5s(赤)6s, 上家打 4s), 雀魂点「吃」之后弹
"请选择要吃的牌"面板. 它**不是**"每条选项画 3 张"—— 实拍是:

    [3s] [5s(赤)]   |   [5s(赤)] [6s]
     └─ 选项1        ─┴ 空隙 ─┴   选项2 ─┘

**每条选项只画协议组合里给的那两张手牌; 被吃的那张(在牌河里)不显示**;
选项之间用一段明显的空隙分隔. 所以上面那行 4 个框就是 2 个选项:
`3s+0s`(吃 4s) 和 `0s+6s`(吃 4s). 这直接推翻了本模块最初"一条 = 3 张"的
假设 —— 没有真帧之前按游戏常识猜的, 猜错了.

## 这里做的事情

1. `split_options()`: 只按**几何形态**把牌框聚成组 —— "一组 = 2 张挨在一起的
   牌", 不假设面板在屏幕的哪个位置. 空隙按**牌宽的比例**判, 不绑分辨率.
2. `claimed_from_combos()`: 从协议给的"每种吃法用哪两张手牌"反推被吃的那张.
   仅用于记录/核对 —— **匹配不需要它**(选项只画手牌, 见上).
3. `match_options()`: 把每组选项的**牌种多重集**和协议组合对齐. 认牌用已有的
   34 类牌面模型, 不需要新模型、不需要人工标注 —— **协议就是标签**.

## 赤 5 的歧义只能靠红色像素解开

组合 `["4s","0s"]` 与 `["4s","5s"]` 折叠后是**同一种** `{4s,5s}` —— 两条
选项在牌种层面无法区分, 唯一区别是面板上那条画的是**红色 5s**. 实测
(`chiopt_1017_141351`): 赤 5s 的红色像素占比 ~0.25, 普通绿 5s 是 0.0 ——
中间隔着一个量级, 一个固定阈值就能分开. `aka_ratio()` 就是这个检测.

## 刻意的边界

- 不 import cv2: 红色检测只碰 numpy 数组, 框由调用方传进来, 单测可直接构造.
- 划分/匹配错了不会点错: 唯一命中才给下标; 认不出、两组撞在一起、张数不对、
  赤歧义解不开 —— 全部返回 None, 调用方据此不点(退回跳过). 点错一条只是吃成
  别的搭子(合法动作), 但能不错就不错.
- 面板零真帧的历史已经结束: 真机帧在 `data/debug/chiopt_*.png`(2026-10-18 采的),
  直接 `read_panel()` 跑即可. **采样机会已经没有了**: 主任务不再点「吃」——
  策略里没有吃, 点开面板只会让那一嘴停摆 20~25 秒(见 `tuning` 里"吃"那一段).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.detection import Detection, median, region_pixels
from src.mahjong.tiles import parse_tile

#: 一条选项 = 协议组合里给的**两张手牌**(被吃的那张不显示, 见模块说明).
OPTION_SIZE = 2

#: 组内相邻两张的间隙超过**牌宽的这么多倍**就断组. 同一选项里的牌是紧挨着画的
#: (间隙 ~1px), 选项之间有 ~0.5 牌宽的空隙. 用"牌宽的比例"而不是固定像素,
#: 是为了不绑死分辨率(1080p 与 720p 都能用).
SPLIT_GAP_RATIO = 0.25

#: 一个框的红色像素占比超过这个值, 就认为画的是**赤牌**(红 5).
#: 实测(真帧 chiopt_1017_141351/142115): 赤 5s ≈ 0.24~0.25, 普通数牌 0.0,
#: 中间隔着一个量级; 取 0.12 两边都留足余量.
AKA_RED_RATIO = 0.12


@dataclass(frozen=True)
class OptionRow:
    """一条候选吃法(2 张手牌)的检测结果."""

    boxes: tuple[Detection, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        """从左到右(或从上到下)读到的牌种, 已折叠赤 5."""
        return tuple(parse_tile(b.label) for b in self.boxes)

    @property
    def center(self) -> tuple[float, float]:
        """这一条的几何中心 —— 要真的点它的时候, 点这里."""
        return (sum(b.cx for b in self.boxes) / len(self.boxes),
                sum(b.cy for b in self.boxes) / len(self.boxes))

    @property
    def min_conf(self) -> float:
        return min(b.conf for b in self.boxes)

    def text(self) -> str:
        return "".join(self.labels)


def _cluster(boxes: Sequence[Detection], *, along: str) -> list[list[Detection]]:
    """沿 `along` 方向把框聚成"行/列".

    判据是**重叠**: 两个框在 `along` 轴上的区间重叠超过较窄那个的一半, 就认为
    画在同一条线上. 用重叠而不是"中心差小于某值", 是因为面板里的牌可能大小
    不一(被选中的那条有时会放大), 中心差会被尺寸带偏.
    """
    lo = (lambda b: b.y1) if along == "y" else (lambda b: b.x1)
    hi = (lambda b: b.y2) if along == "y" else (lambda b: b.x2)
    ordered = sorted(boxes, key=lambda b: (lo(b) + hi(b)) / 2.0)
    lines: list[list[Detection]] = []
    cur: list[Detection] = []
    band_lo = band_hi = 0.0
    for b in ordered:
        if not cur:
            cur, band_lo, band_hi = [b], lo(b), hi(b)
            continue
        overlap = min(band_hi, hi(b)) - max(band_lo, lo(b))
        shortest = min(band_hi - band_lo, hi(b) - lo(b))
        if shortest > 0 and overlap >= 0.5 * shortest:
            cur.append(b)
            band_lo = min(band_lo, lo(b))
            band_hi = max(band_hi, hi(b))
        else:
            lines.append(cur)
            cur, band_lo, band_hi = [b], lo(b), hi(b)
    if cur:
        lines.append(cur)
    return lines


def _split_line(line: Sequence[Detection], *, along: str) -> list[list[Detection]]:
    """把一条线按间隙切成若干段(每段应当正好 2 张)."""
    lo = (lambda b: b.x1) if along == "x" else (lambda b: b.y1)
    hi = (lambda b: b.x2) if along == "x" else (lambda b: b.y2)
    ordered = sorted(line, key=lambda b: (lo(b) + hi(b)) / 2.0)
    if not ordered:
        return []
    extent = median([(hi(b) - lo(b)) for b in ordered])
    gap_min = SPLIT_GAP_RATIO * extent
    groups: list[list[Detection]] = []
    cur = [ordered[0]]
    cur_hi = hi(ordered[0])
    for b in ordered[1:]:
        if lo(b) - cur_hi > gap_min:
            groups.append(cur)
            cur = [b]
        else:
            cur.append(b)
        cur_hi = max(cur_hi, hi(b))
    groups.append(cur)
    return groups


def _runs(groups: Iterable[Sequence[Detection]]) -> list[list[Detection]]:
    return [list(g) for g in groups if len(g) == OPTION_SIZE]


def split_options(boxes: Iterable[Detection], *,
                  expect: int | None = None) -> list[OptionRow]:
    """把面板上的牌框切成"若干条候选吃法"(每条 2 张手牌).

    `expect` 非空时(= 协议说这一嘴有几种吃法)用来在"横排/竖排"两种读法之间
    选一个: 先按横排切(雀魂的选择条是横着排的, 真帧实证), 只能切出 2 张一组、
    但组数对不上时再试竖排. 两种都切不出的, 返回空列表 —— 也就是"这次划分
    失败, 调用方别点".
    """
    boxes = [b for b in boxes if b is not None]
    if len(boxes) < OPTION_SIZE:
        return []

    def _read(lines: list[list[Detection]], along: str) -> list[list[Detection]]:
        return _runs(g for line in lines for g in _split_line(line, along=along))

    horiz = _read(_cluster(boxes, along="y"), "x")
    vert = _read(_cluster(boxes, along="x"), "y")
    picked = horiz
    if not picked:
        picked = vert
    elif vert and expect is not None and len(horiz) != expect \
            and len(vert) == expect:
        picked = vert
    # 组内排序: 横向组靠 cx, 竖向组靠 cy; 两者用 (cy, cx) 一次搞定.
    return [OptionRow(tuple(sorted(g, key=lambda b: (round(b.cy), b.cx))))
            for g in picked]


def _clean(boxes: list[Detection]) -> list[Detection]:
    """丢掉面板读数里的两类杂质(真帧 142115 实锤):

    ① 字牌框 —— 吃必是顺子, 选择条里**只会有数牌**; 实测一个 5z(conf 0.30)
       的误检恰好落在选项空隙里, 把两组桥成一整条 6 张, 划分全灭.
    ② 碎框 —— 宽度不足中位 6 成的框是面板装饰/分割线的误检(实测 41px vs
       中位 78px), 它同样会填进空隙.
    """
    boxes = [b for b in boxes if not b.label.endswith("z")]
    if not boxes:
        return []
    med = median([b.x2 - b.x1 for b in boxes])
    return [b for b in boxes if (b.x2 - b.x1) >= 0.6 * med]


def read_panel(detector, frame, region, *, expect: int | None = None
               ) -> list[OptionRow]:
    """在一个搜索区域里跑牌面模型, 再切成候选组.

    `detector` 是 `src.detector.YoloDetector`(34 类牌面), `region` 是
    `(上, 下, 左, 右)` 的相对比例, 与 `detector.HAND_REGION` 同格式.
    真帧(2026-10-17)实证面板牌行在 y≈70~81%、x≈34~72%, 取一片略大的范围.
    """
    height, width = frame.shape[:2]
    y0, y1, x0, x1 = region_pixels(width, height, region)
    boxes = detector.detect(frame[y0:y1, x0:x1], offset=(float(x0), float(y0)))
    return split_options(_clean(list(boxes)), expect=expect)


def aka_ratio(frame, box) -> float:
    """一个框里**红色像素**的占比(0~1). 赤牌(红 5)远高于普通牌.

    只用 numpy 的通道比较, 不引入 cv2 依赖. 阈值语义见 `AKA_RED_RATIO`.
    """
    x1, y1 = max(int(box.x1), 0), max(int(box.y1), 0)
    x2, y2 = int(box.x2), int(box.y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    b = crop[:, :, 0].astype(int)
    g = crop[:, :, 1].astype(int)
    r = crop[:, :, 2].astype(int)
    red = (r > 110) & (r - g > 50) & (r - b > 50)
    return float(red.mean())


def row_has_aka(frame, row: OptionRow) -> bool:
    """这条选项里有没有画着赤牌(红 5)?"""
    return any(aka_ratio(frame, b) >= AKA_RED_RATIO for b in row.boxes)


def _completions(pair: Sequence[str]) -> set[str]:
    """两张手牌 + 哪一张能凑成顺子?(返回所有可能的那一张)

    字牌直接返回空集 —— 字牌不能吃, 也不存在"连续三张".
    """
    a, b = parse_tile(pair[0]), parse_tile(pair[1])
    if a[-1] != b[-1] or a[-1] == "z":
        return set()
    suit = a[-1]
    ranks = [int(a[0]), int(b[0])]
    out: set[str] = set()
    for rank in range(1, 10):
        triple = sorted(ranks + [rank])
        if len(set(triple)) == 3 and triple[1] == triple[0] + 1 \
                and triple[2] == triple[1] + 1:
            out.add(f"{rank}{suit}")
    return out


def claimed_from_combos(combos: Sequence[Sequence[str]]) -> str | None:
    """从"每种吃法用哪两张手牌"反推**被吃的那张**.

    协议里的 `combination` 只给搭子(两张), 不给被吃的那张(它在更早的
    `ActionDiscardTile` 里, 追时序容易错位). 但一吃必定是三张连续数牌, 所以
    每项的"缺的那张"就是被吃的牌; 所有项必须指向**唯一**一张才返回 ——
    像 `["4s","0s"]` 与 `["4s","5s"]` 这种"赤/普通 5 二选一"的组合, 交集中
    有两张候选(3s 和 6s 都说得通), 这里如实返回 None: 反正匹配用不到它
    (选项只画手牌), 它只进记录供人核对.
    """
    cands: set[str] | None = None
    for combo in combos:
        tiles = [parse_tile(t) for t in combo if t]
        if len(tiles) != 2:
            return None
        here = _completions(tiles)
        cands = here if cands is None else (cands & here)
        if not cands:
            return None
    if not cands or len(cands) != 1:
        return None
    return next(iter(cands))


def _folded(combo: Sequence[str]) -> tuple[str, ...] | None:
    tiles = [parse_tile(t) for t in combo if t]
    if len(tiles) != 2:
        return None
    return tuple(sorted(tiles))


def match_options(rows: Sequence[OptionRow],
                  combos: Sequence[Sequence[str]],
                  claimed: str | None = None,
                  frame=None) -> list[int | None]:
    """协议里**每种**吃法对应面板上第几组?(逐项返回 `rows` 的下标)

    匹配是牌种多重集相等的**唯一命中**; 出现赤/普通 5 二选一(两组折叠后
    同牌种, 真帧 141351 实证)时, 用红色像素配对: "组合里带 0x 的"对到
    "画着红牌的"那组, 普通组合对到剩下那组 —— 这一步需要 `frame`; 没给或
    红色检测解不开, 整组返回 None(不点).

    `claimed` 只进记录/核对(可选), 不参与匹配 —— 见 `claimed_from_combos`.
    """
    folded = [_folded(c) for c in combos]
    if any(f is None for f in folded):
        return [None] * len(combos)

    out: list[int | None] = [None] * len(combos)
    groups: dict[tuple[str, ...], list[int]] = {}
    for idx, f in enumerate(folded):
        groups.setdefault(f, []).append(idx)
    for f, idxs in groups.items():
        hits = [i for i, row in enumerate(rows)
                if tuple(sorted(row.labels)) == f]
        if len(idxs) == 1 and len(hits) == 1:
            out[idxs[0]] = hits[0]
            continue
        # 撞车组: 只解"赤/普通二选一"(恰好两组同牌种、恰好一个组合带 0x).
        if len(idxs) != 2 or len(hits) != 2 or frame is None:
            continue
        has_aka = [row_has_aka(frame, rows[i]) for i in hits]
        aka_combo = [j for j in idxs
                     if any(str(t).startswith("0") for t in combos[j])]
        if sum(has_aka) == 1 and len(aka_combo) == 1:
            red_row = hits[0] if has_aka[0] else hits[1]
            plain_combo = idxs[0] if idxs[1] == aka_combo[0] else idxs[1]
            out[aka_combo[0]] = red_row
            out[plain_combo] = hits[1] if red_row == hits[0] else hits[0]
    return out
