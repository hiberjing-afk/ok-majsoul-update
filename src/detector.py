"""对局内感知: 用现成的 YOLO 权重认牌面和对局按钮.

两个权重都不是我们自己训练的, 直接复用上游 liqwang/majsoul-bot(MIT)
detector/ 下提交的成品, 所以不要改 models/ 里的文件名, 也不要重新训练:

    models/mahjong.pt     34 类: 1m-9m / 1p-9p / 1s-9s / 1z-7z.
                          没有赤 5 类 -- 赤 5 的牌效和普通 5 完全一样,
                          识别成 "5p" 正好和 src/mahjong/tiles.py 的
                          parse_tile("0p") -> "5p" 折叠对上.
    models/majsoul_UI.pt  13 类: 对局内的动作按钮.
                          chi / peng / gang / lizhi / zimo / he / tiaoguo /
                          babei / liuju / confirm / match / silver / 3p-east

本模块只负责"跑模型 + 把结果转成 Detection", 排序 / 去噪 / 分槽这些纯逻辑
在 src/detection.py 里(那边没有 torch 依赖, 可以单测).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.detection import (
    Detection,
    drop_meld_tiles,
    group_by_label,
    missing_slots,
    read_row,
    region_pixels,
)
from src.instrumentation import bench
from src.paths import TILE_MODEL, UI_MODEL

logger = logging.getLogger(__name__)

# 手牌区在整帧里的位置(相对比例: 上, 下, 左, 右).
# 沿用上游参考实现的取值, 雀魂的手牌就是贴着底边横排一条.
#
# **右边从 0.90 放宽到 0.95(2026-10-06 23:42 实跑)**:
# 右侧留0.90 = 1920 帧里只到 x=1728, 而实测手牌一直排到 **x~1720** --
# **最后那张牌被切成一半**, 连带后面几张整片漏检. 同一帧实测:
#
#   right=0.90 (x~1728)  认出  8 张(真实13 张)
#   right=0.95 (x~1824)  认出 10 张  ← 当前取值
#   right=0.98 (x~1881)  认出 10 张, 但左边多裁进来一些桌面装饰
#   right=1.00 (x=1920)   认出  9 张, 反而变少(裁切比例变了模型内部resize 也变)
#
# 左边 0.10 保留: 手牌最左也就在 x~220, 再往左是"魂牌"(自己的头像),
# 放开会把头像框当牌认进来.
HAND_REGION = (0.75, 1.0, 0.10, 0.95)

# 牌面模型的输入尺寸.
#
# **这个值是 2026-10-07 用 322 张实跑留证帧量出来的, 别再照抄上游的 (224,1024)
# 也不要再改回 (640,1920)。** 定这个值只看一个指标: **低置信张数**
# (conf < MIN_TILE_CONF 的槽位有多少个), 因为它直接决定这一轮走不走稳健出牌、
# 会不会把好搭子拆了打 -- 也就是实跑里"没有看懂逻辑"的那些盘。
#
# 只算认满 >=13 张的 193 张对局帧 (发牌/结算那种残帧会污染统计, 单列):
#
#   imgsz          平均中位 conf     平均低置信张数    西风(3z) 判定
#   (224, 896)        0.915            3.53
#   (256,1024)        0.925            2.18         3z  ✓   ← 当前取值
#   (288,1152)        0.924            2.63
#   (320,1280)        0.920            2.86         3z  ✓
#   (640,1920)        0.765            7.35         4m  ✗   ← 旧值
#
# 旧值 (640,1920) 的来历见下一段, 那份实测**问题出在只看单帧的最小值**: 它挑的
# `16-46-59.913_打一局友人场_original.png` 今天重测, (256,1024) 中位 0.96 低 1 个、
# (640,1920) 中位 0.90 低 5 个 —— 连那张帧都推翻了旧结论。
#
# 为什么会这样(2026-10-06 的推断是错的): 当时的理由写的是"牌面保持原始像素,
# 一张不缩". 但手牌区裁出来只有 1632 宽, 喂进 (640,1920) 时 ultralytics 会按
# 宽度把图**放大** 1.18 倍, 牌面反而从 92px 变成 108px。这个权重是在牌面
# **50~60px** 那一档上训的(上游是 1280x720 窗口 + (224,1024): 裁切约 1088 宽,
# 缩放 0.94, 一张牌 ~57px)。我们 1080p 下喂 108px 等于把牌放大一倍, 分布完全
# 对不上, 于是模型开始"认得出框、认不准字" —— 低置信全落在 0.6~0.77 这一档。
# (256,1024) 缩放 0.627, 一张牌回到 58px, 正好落在训练分布里。
#
# 三个直接后果, 都是实跑踩到过的:
#   1. **西风被认成 4m** (conf 0.58~0.80)。三麻里 4m 不存在, 所以还能被
#      out_of_play 兜住当误识枚举; **四人麻将里 4m 合法**, 就会当成真牌用。
#   2. t63 一手 3p4p5p 被读成 3p4p6p6p8p (9/14 张低置信), 于是"正确地"
#      算出去打 3p —— 拆了一个现成的顺子。
#   3. 平均每轮 7.5/13 张低置信, 几乎每一轮都在走稳健出牌, 决策全是"在
#      几种可能取值下都不踩坑"的妥协, 看起来就是没逻辑。
# 换成 (256,1024) 后同样这些帧: 低置信 2.18/13, 西风 = 3z, 3p4p5p 读得出来。
#
# 上限 1920 原先是担心"窗口再大就没信息", 这个担心仍然成立, 但它不是这个
# 参数要解决的问题: 这里调的是**模型输入**, 不是裁切范围。裁切范围归 HAND_REGION。
TILE_IMGSZ = (256, 1024)
# 按钮模型跑整帧. 按钮比手牌大得多, 640 已经够.
UI_IMGSZ = 640

# 置信度下限. 牌面宁可漏认也不要认错(认错会直接打错牌),
# 按钮漏认最多是这轮不动, 下轮再试.
#
# **这里有两个门槛, 职责完全不同, 别再混起来看**:
#
#   TILE_CONF      "这个槽位上有没有牌"  -> 决定**张数**(occupancy).
#   MIN_TILE_CONF  "认出来的这张牌是几"  -> 识别质量的标尺(见 tuning).
#
# TILE_CONF 从 0.40 降到 0.15 的依据(2026-10-07 实跑帧逐张核对):
#
#   卡死那一帧上, 有 **4 个槽位是真的有牌** 但最高置信度只有
#   0.325 / 0.169 / 0.362 / 0.271 -- 全被 0.40 滤掉, 于是 13 张读成 10 张.
#   而 10 恰好是"1 副露时的合法暗牌数", 所以既没报"张数不对"也没报 suspect,
#   决策层一路输出"在等别人"空转到卡死兜底(详见 detection.missing_slots).
#   阈值降到 0.15 后同一帧读出 **13 张**, 与画面逐张点数一致.
#
# 原来的 0.40 是照着"0.30~0.40 这一档只见过误识"定的, 但那份统计是在
# TILE_IMGSZ 还是 (224,1024) 的时候做的; 改成 (256,1024) 之后(牌面回到模型
# 训练时那一档大小), 真牌落进这一档已经实测到了. 也就是说**旧结论的前提
# 已经不存在**.
#
# (256,1024) 下同一张卡死帧的槽位置信度整体抬到了 0.87 中位, 但 0.15 这个
# 门槛**故意保留不动**: 它管的是"槽位上有没有牌"这个几何事实, 调低只会多留
# 几个低置信框, 而那几个框正好是"这一轮该不该稳健出牌"的判据; 调高则会把
# 认对了框但认不准字的真牌整张丢掉(就是卡死那一帧 13 张读成 10 张的成因).
#
# 代价与兜底: 门槛降低会多留一些低置信框, 但
#   - 每槽只留最优(倍率 1.00~1.05 的等间距, 分槽很干净);
#   - 张数仍然要过 missing_slots 这道关;
#   - 对局内核只拿视觉的**框坐标**, 牌名/标签一律来自协议.
# 也就是说: 低置信框只影响"有几张牌"这个几何事实, 不会直接被当成确定标签用.
TILE_CONF = 0.15
# UI 模型跑整帧. 按钮比手牌大得多, 640 已经够.
#
# **imgsz=640 是实测的最优值, 别调大(2026-10-06 23:42)**:
# 同一帧(卡在"可以碰"那一张)扫不同输入尺寸:
#   320   只认出 tiaoguo=0.064(碰按钮整个没了)
#   640   tiaoguo=0.768 + peng=0.227      ← 当前取值, 两个按钮都在
#   1280  tiaoguo=0.426 + 一堆 0.02x 的噪声, peng 反而没了
#   1920  全是0.01x 的噪声(confirm/3p-east/zimo/match), 真实按钮全丢
# 也就是说这个权重**就是在 640 上训的**, 放大只会让它输出垃圾.
UI_IMGSZ = 640

# UI 的置信度下限.
#
# **从 0.25 降到 0.15(2026-10-06 23:42 实跑踩坑)**:
# 卡在"别人出的牌我可以碰"那一次, 日志里 `buttons` 全程是 `{}`,
# 但同一张帧上实测 peng 的 conf = **0.227** -- 认出来了, 却被0.25 门槛滤掉,
# 于是既没点碰也没点跳过, 干等到超时.
#
# 为什么之前 0.25 没出事: 之前只遇到过 tiaoguo(0.768)和 confirm/liuju 这类
# 高置信按钮, 全都远高于 0.25, 所以门槛取值一直没被验证过.
#
# 为什么不用更低的 0.10: conf=0.05~0.10 这一档在同一帧上开始冒出噪声框
# (silver=0.022 / confirm=0.012 之类), 而 0.227 与噪声之间还隔着
# 一个数量级, 0.15 裕度充足.
UI_CONF = 0.15


@dataclass
class HandReading:
    """一次手牌识别的结果."""

    tiles: list[str]                 # 从左到右的牌, 如 ["1m", "5p", ...]
    boxes: list[Detection]           # 和 tiles 一一对应, 坐标是整帧坐标
    region: tuple[int, int, int, int]  # 用到的手牌区 (y0, y1, x0, x1)
    #: 按**间距**推断出的"漏认了几张"(见 src/detection.missing_slots).
    #: 和置信度无关: 即使所有认出来的牌都很确定, 只要行里有一段空档,
    #: 这里就 >0, 说明 tiles 的张数**不可信**.
    missing_slots: int = 0

    @property
    def count(self) -> int:
        return len(self.tiles)

    def box_of(self, index: int) -> Detection | None:
        if 0 <= index < len(self.boxes):
            return self.boxes[index]
        return None


class YoloDetector:
    """ultralytics 权重的一层薄封装.

    延迟加载: 构造时不碰 torch, 第一次 detect() 才 import ultralytics +
    读权重. 这样只跑模板匹配的任务(比如建房)不会被拖慢 1~2 秒.
    """

    def __init__(self, weight: str | Path, imgsz: int | Sequence[int] = 640,
                 conf: float = 0.25, iou: float = 0.45, device: str | None = None,
                 augment: bool = False):
        self.weight = Path(weight)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.augment = augment
        self._device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self.weight.exists():
                raise FileNotFoundError(
                    f"找不到权重 {self.weight}; 先跑 tools/fetch_models.py 下载")
            from ultralytics import YOLO  # 延迟 import, 避免启动时拉起 torch
            logger.info("加载权重 %s", self.weight)
            self._model = YOLO(str(self.weight))
        return self._model

    @property
    def device(self) -> str:
        """没显式指定就用 GPU; 没有 CUDA 环境时退回 CPU(会慢 20~50 倍)."""
        if self._device:
            return self._device
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - 没装 torch 时不会走到这里
            self._device = "cpu"
        return self._device

    @property
    def names(self) -> dict[int, str]:
        return dict(self.model.names)

    def detect(self, image: np.ndarray,
               offset: tuple[float, float] = (0.0, 0.0)) -> list[Detection]:
        """跑一次推理, 返回原始检测框(未过滤).

        offset 是把坐标搬回整帧用的 (dx, dy); 传入的是裁剪图时给裁掉的左上角.
        """
        # bench 挂钩(默认关闭): 按权重名计一次推理(mahjong / majsoul_UI),
        # 记录耗时, 异常计数后原样抛出. 注意 TTA 的一次 detect() 内部是
        # 多次前向, 这里按"一次 detect 调用"计, 与 GameVision 的调用频率对应.
        with bench.inference(self.weight.stem):
            results = self.model.predict(
                source=image, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                device=self.device, augment=self.augment, verbose=False)
            return self._to_detections(results[0], offset)

    @staticmethod
    def _to_detections(result, offset: tuple[float, float]) -> list[Detection]:
        names = result.names
        boxes = getattr(result, "boxes", None)
        out: list[Detection] = []
        if boxes is None or len(boxes) == 0:
            return out
        dx, dy = offset
        for box in boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            out.append(Detection(
                label=str(names[int(box.cls[0])]), conf=float(box.conf[0]),
                x1=x1 + dx, y1=y1 + dy, x2=x2 + dx, y2=y2 + dy))
        return out

    def warmup(self, shape: tuple[int, int, int] = (720, 1280, 3)) -> float:
        """用一张空图跑一遍, 把 CUDA / cuDNN 初始化做完. 返回耗时(秒).

        第一次真推理会额外花 1~2 秒做初始化, 进对局前先 warmup 掉,
        免得轮到我们出牌时卡在初始化上被计时器判超时.
        """
        import time

        blank = np.zeros(shape, dtype=np.uint8)
        t0 = time.perf_counter()
        self.detect(blank)
        return time.perf_counter() - t0


def _make_detector(weight: str | Path, *, imgsz, conf: float,
                   device: str | None, augment: bool,
                   backend: str = "auto"):
    """按 backend 选推理后端.

    - "onnx": 强制 ONNX(缺 .onnx 文件会抛错);
    - "torch": 强制 ultralytics/torch;
    - "auto"(默认): onnxruntime 可用且同名 .onnx 存在 -> ONNX, 否则 torch.
      ONNX/DirectML 与 torch(CUDA) 验收时逐框一致, 且运行时不再依赖 torch ——
      依赖从 ~3GB 降到 ~300MB.
    """
    import os
    choice = backend if backend != "auto" else os.environ.get(
        "OK_BACKEND", "auto")
    if choice == "auto":
        from importlib.util import find_spec
        has_ort = find_spec("onnxruntime") is not None
        has_onnx = Path(weight).with_suffix(".onnx").exists()
        choice = "onnx" if (has_ort and has_onnx) else "torch"
    if choice == "onnx":
        from src.onnx_backend import OnnxDetector
        return OnnxDetector(weight, imgsz=imgsz, conf=conf, device=device,
                            augment=augment)
    return YoloDetector(weight, imgsz=imgsz, conf=conf, device=device,
                        augment=augment)


class GameVision:
    """对局内的眼睛: 手牌 + 动作按钮.

    两个权重各管一块:
    - 牌面只在手牌区跑(整帧跑既慢又容易被牌河 / 副露干扰);
    - 按钮跑整帧, 因为 chi / peng / lizhi 这些按钮浮在牌桌中间.
    """

    def __init__(self, tile_weight: str | Path = TILE_MODEL,
                 ui_weight: str | Path = UI_MODEL,
                 hand_region: Sequence[float] = HAND_REGION,
                 device: str | None = None,
                 backend: str = "auto"):
        # 牌面模型开 TTA(augment=True) —— 抄上游 liqwang/majsoul-bot 的做法
        # (`detector.py`: `self.mahjong_model.predict(source=image, imgsz=(224,1024),
        # augment=True)`).
        #
        # 196 张干净对局帧实测: 开 TTA 低置信/帧 3.68 -> 3.21, 槽位中位conf
        #
        #                    低置信/帧   槽位中位conf   张数不对   跨帧标签翻转
        #   不开 TTA            3.68       0.918       34 帧      0.10 次/对
        #   开 TTA              3.21       0.936       26 帧      0.04 次/对
        #
        # 换掉的标签里 `6p -> 7p` 67 处、`9p -> 8p` 若干处, 逐张裁图人工核对确认
        # **那些牌本来就是 7 筒 / 8 筒, 是原标签错了** —— 也就是 TTA 把真错改对了,
        # 顺带把这些牌的 conf 从 0.2~0.6 抬到 0.8+。代价: 推理 49 -> 86 ms/帧。
        #
        # **UI 模型故意不开**: 结算页 `confirm` 的实测 conf 只有 0.151~0.156,
        # 就压在 `UI_CONF = 0.15` 上(高 0.003); TTA 挪动这几个千分位就可能把
        # 翻页按钮整张弄丢, 得不偿失。
        self.tiles = _make_detector(tile_weight, imgsz=TILE_IMGSZ, conf=TILE_CONF,
                                    device=device, augment=True, backend=backend)
        self.ui = _make_detector(ui_weight, imgsz=UI_IMGSZ, conf=UI_CONF,
                                 device=device, augment=False, backend=backend)
        self.hand_region = tuple(float(v) for v in hand_region)

    def read_hand(self, frame: np.ndarray) -> HandReading:
        """认手牌. 裁出手牌区 -> 跑牌面模型 -> 剔副露 -> 分槽去噪 -> 从左到右.

        同时算 `missing_slots`: 牌是等间距排的, 所以"漏认"在几何上留得下痕迹
        (见 src/detection.missing_slots). 张数不可信时调用方必须知道。

        剔副露(drop_meld_tiles)必须在分槽前做: 雀魂把副露竖着摆在手牌行
        右端且比手牌小一圈, 模型会把副露牌当手牌读进来; 副露组数一多,
        filter_outliers 的中位数会倒向副露, 把真手牌当离群值剔光
        (2026-10-11 三碰局实锤, 见 detection.MELD_HEIGHT_RATIO)。
        """
        # bench 挂钩(默认关闭): 阶段标签由任务主循环在读之前设置
        # (见 src/instrumentation.py 模块 docstring 的 phase 语义).
        with bench.hand_read():
            height, width = frame.shape[:2]
            y0, y1, x0, x1 = region_pixels(width, height, self.hand_region)
            crop = frame[y0:y1, x0:x1]
            boxes = drop_meld_tiles(
                self.tiles.detect(crop, offset=(float(x0), float(y0))))
            ordered = read_row(boxes)
            return HandReading(tiles=[b.label for b in ordered], boxes=ordered,
                               region=(y0, y1, x0, x1),
                               missing_slots=missing_slots(ordered))

    def read_buttons(self, frame: np.ndarray) -> dict[str, list[Detection]]:
        """认对局按钮. 返回 类别名 -> 按置信度降序的框列表(整帧坐标)."""
        return group_by_label(self.ui.detect(frame))

    def warmup(self, shape: tuple[int, int, int] = (720, 1280, 3)) -> None:
        """两个模型各预热一遍. 进对局前调一次."""
        for tag, detector in (("tiles", self.tiles), ("ui", self.ui)):
            cost = detector.warmup(shape)
            logger.info("%s 预热 %.0f ms (device=%s)", tag, cost * 1000,
                        detector.device)
