"""ONNX Runtime 推理后端: 与 ultralytics/torch 完全对齐的检测器.

为什么要有它(2026-10-10, v1.1): torch(CUDA 版)带 2.5GB 的 DLL, 打包分发
几乎不可行; ok-ww 系的成熟做法是把权重导出 ONNX, 用 onnxruntime-directml
推理(DirectML 走 DirectX 12, N/A/I 卡都能加速, 运行库只有 ~25MB). 导出
之后运行时可以**完全不装 torch**.

对齐(验收 = 与 torch 后端逐框一致, 见 tests/TestOnnxParity.py):

- **LetterBox**: 等比缩放放进 imgsz, 上下/左右各补一半 114 灰边(与
  ultralytics LetterBox 同一套 round 规则).
- **TTA(augment=True)**: ultralytics DetectionModel._predict_augment 的
  三次前向 —— 原图(scale=1), 水平翻转再缩 0.83, 缩 0.67 —— 缩放用
  bilinear 插值后**向右/向下**补 0.447(=114/255, 归一化空间)到 32 的倍数;
  每次前向的预测框先 /scale 回原坐标(翻转的那次还要 x = 宽 - x), 再按
  _clip_augmented 裁掉大尺度 pass 的 P3 头 / 小尺度 pass 的 P5 尾,
  最后三份拼在一起做**一次** NMS.
- **NMS**: 先按"全类最大分"过滤 conf, 每个候选只留最优类, 按分数降序,
  同类间贪心抑制(IoU > iou_thres), 最多 max_det=300 个 -- 与
  ultralytics non_max_suppression(agnostic=False) 同语义.

类别名/stride 从 ONNX 的 metadata_props 里读(ultralytics 导出时写入),
不需要 .pt 在场.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.detection import Detection
from src.instrumentation import bench

logger = logging.getLogger(__name__)

#: TTA 的(缩放, 是否水平翻转)组合, 顺序与 ultralytics 一致.
TTA_PASSES: tuple[tuple[float, bool], ...] = (
    (1.0, False), (0.83, True), (0.67, False))
#: TTA 缩放后补边用的灰度值(归一化空间; 114/255, 与 ultralytics 相同).
_TTA_PAD_VALUE = 0.447
#: NMS 最多保留的框数(ultralytics 的 max_det 默认值).
MAX_DET = 300


def letterbox(img: np.ndarray, new_h: int, new_w: int,
              stride: int = 32) -> tuple[np.ndarray, float, float, float]:
    """ultralytics LetterBox(auto=True, rect) 的复刻: 等比缩放 + **最小**补边.

    生产 predict 走的是 rect=True/auto=True 路径(实测 args.rect=True):
    补边只补到 stride 的倍数, 而不是补满整个 imgsz. 例如 270x1632 的手牌
    区配 (256,1024) 的 imgsz, 实际输入是 192x1024 而不是 256x1024 --
    这个差异会改变感受野, 必须逐像素对齐.
    返回 (图, 缩放比 r, 水平补边 dw, 垂直补边 dh), 供检测框逆映射回原图.
    """
    h0, w0 = img.shape[:2]
    r = min(new_h / h0, new_w / w0)
    unpad_w, unpad_h = round(w0 * r), round(h0 * r)
    if (w0, h0) != (unpad_w, unpad_h):
        img = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_w - unpad_w) % stride, (new_h - unpad_h) % stride
    dw, dh = dw / 2.0, dh / 2.0                      # center=True
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, dw, dh


def _to_chw(img: np.ndarray) -> np.ndarray:
    """HWC uint8(BGR) -> 1x3xHxW float32(/255), **转成 RGB** 再换轴.

    ultralytics 前处理的最后一步是 img.transpose(2,0,1)[::-1] -- 通道翻成
    RGB 再喂模型(权重是按 RGB 训的). 漏了这步等于红绿对调, 梅花色系
    (红/绿)的标签会大面积互串 -- 2026-10-10 实测 60 手里 29 手读数漂移.
    """
    x = img.astype(np.float32) / 255.0
    return x[:, :, ::-1].transpose(2, 0, 1)[None]


def _xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    out = xywh.copy()
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return out


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """标准 xyxy IoU 矩阵(len(a) x len(b)), 供贪心 NMS 用."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None] - inter
    return inter / np.clip(union, 1e-9, None)


def nms(pred: np.ndarray, conf_thres: float, iou_thres: float,
        num_classes: int) -> tuple[list[Detection], list[int]]:
    """ultralytics non_max_suppression(agnostic=False) 的 numpy 复刻.

    pred: (1, 4+nc, M) 原始输出, xywh 像素坐标.
    返回 (Detection 列表[label 留空, 由调用方查 names], 对应的类别 id).
    """
    p = pred[0].T                                  # (M, 4+nc)
    scores = p[:, 4:4 + num_classes]
    cls_conf = scores.max(axis=1)
    cls_id = scores.argmax(axis=1)
    keep = cls_conf > conf_thres
    if not keep.any():
        return [], []
    p, cls_conf, cls_id = p[keep], cls_conf[keep], cls_id[keep]
    boxes = _xywh2xyxy(p[:, :4])

    order = np.argsort(-cls_conf)[:30000]          # ultralytics 的 max_nms
    boxes, cls_conf, cls_id = boxes[order], cls_conf[order], cls_id[order]

    picked: list[int] = []
    # 按类分组做贪心抑制(等价于 torchvision batched_nms 的同类隔离).
    for c in np.unique(cls_id):
        idx = np.where(cls_id == c)[0]
        idx = idx[np.argsort(-cls_conf[idx])]
        sup = np.zeros(len(idx), dtype=bool)
        for i, a in enumerate(idx):
            if sup[i]:
                continue
            picked.append(int(a))
            if len(picked) >= MAX_DET:
                break
            rest = idx[i + 1:][~sup[i + 1:]]
            if len(rest):
                ious = _iou_matrix(boxes[a:a + 1], boxes[rest])[0]
                sup[i + 1:][~sup[i + 1:]] |= ious > iou_thres
        if len(picked) >= MAX_DET:
            break
    picked = picked[:MAX_DET]
    dets = [Detection(label="", conf=float(cls_conf[i]),
                      x1=float(boxes[i][0]), y1=float(boxes[i][1]),
                      x2=float(boxes[i][2]), y2=float(boxes[i][3]))
            for i in picked]
    return dets, [int(cls_id[i]) for i in picked]


def _tta_scale(img01: np.ndarray, ratio: float, gs: int) -> np.ndarray:
    """ultralytics scale_img 的 numpy 复刻(输入/输出都是归一化 CHW).

    bilinear 缩放到 int(h*ratio) x int(w*ratio), 再向右/向下补
    0.447 灰边到 32 的倍数(注意**不居中**, 与 torch 实现一致).
    """
    if ratio == 1.0:
        return img01
    _, _, h, w = img01.shape
    sh, sw = int(h * ratio), int(w * ratio)
    small = cv2.resize(img01[0].transpose(1, 2, 0), (sw, sh),
                       interpolation=cv2.INTER_LINEAR)
    out = np.full((3, math.ceil(h * ratio / gs) * gs,
                   math.ceil(w * ratio / gs) * gs), _TTA_PAD_VALUE,
                  dtype=np.float32)
    out[:, :sh, :sw] = small.transpose(2, 0, 1)
    return out[None]


def _clip_augmented(out: np.ndarray, input_hw: tuple[int, int],
                    stride: int, nl: int, position: int,
                    total_passes: int) -> np.ndarray:
    """ultralytics _clip_augmented: 裁掉大图 pass 的 P5 尾 / 小图 pass 的 P3 头.

    v8 输出的 anchor 沿最后一维按 P3->P4->P5(步距 8/16/32)排列, P3 的
    anchor 数是 P5 的 4^(nl-1) 倍. position 是本 pass 在 TTA 序列里的
    下标: 只有第一份裁尾(它的输入最大, P5 特征图最没信息)、最后一份裁头.
    """
    a_last = (input_hw[0] // stride) * (input_hw[1] // stride)
    g = sum(4 ** x for x in range(nl))             # = (4^nl - 1) / 3
    n = out.shape[2]
    if position == 0 and total_passes > 1:
        i = (n // g) * 4 ** 0
        return out[:, :, :-i]
    if position == total_passes - 1 and total_passes > 1:
        i = (n // g) * 4 ** (nl - 1)
        return out[:, :, i:]
    return out


class OnnxDetector:
    """与 YoloDetector 同接口的 ONNX 后端(weight 指向 .pt, 实际读同名 .onnx)."""

    def __init__(self, weight: str | Path, imgsz: int | Sequence[int] = 640,
                 conf: float = 0.25, iou: float = 0.45,
                 device: str | None = None, augment: bool = False):
        self.weight = Path(weight)
        self.onnx_path = self.weight.with_suffix(".onnx")
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.augment = augment
        self._device = device
        self._session = None
        self._names: dict[int, str] = {}
        self._stride = 32
        self._num_classes = 0
        # 检测头层数(供 TTA 的 anchor 头尾裁剪)从首次前向的输出形状反推 --
        # 不能硬编码: 上游权重有 3 层(P3-P5)也有 4 层(P2-P5, 如牌面模型)的.
        self._nl: int | None = None

    # ---- 会话与元数据 -------------------------------------------------
    def _load(self) -> None:
        import onnxruntime as ort
        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"找不到 {self.onnx_path}; 用 ultralytics 从 {self.weight} "
                f"导出(YOLO(pt).export(format='onnx', imgsz=..., dynamic=True))")
        logger.info("加载 ONNX %s", self.onnx_path)
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self.onnx_path),
                                             providers=providers)
        meta = {m.key: m.value for m in
                __import__("onnx").load(str(self.onnx_path)).metadata_props}
        self._names = {int(k): str(v) for k, v in
                       ast.literal_eval(meta["names"]).items()}
        self._num_classes = len(self._names)
        self._stride = int(json.loads(meta["stride"])[0]
                           if meta["stride"].startswith("[") else meta["stride"])

    @property
    def session(self):
        if self._session is None:
            self._load()
        return self._session

    @property
    def device(self) -> str:
        """实际生效的执行后端: dml(GPU) 或 cpu."""
        if self._device:
            return self._device
        prov = self.session.get_providers()[0]
        self._device = "dml" if "Dml" in prov else "cpu"
        return self._device

    @property
    def names(self) -> dict[int, str]:
        if not self._names:
            self._load()
        return self._names

    # ---- 推理 ---------------------------------------------------------
    def _forward(self, chw: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {"images": chw})[0]
        if self._nl is None:
            # 用一次真实输出反推检测层数: N = A_last * (4^nl - 1) / 3.
            a_last = (chw.shape[2] // self._stride) * (chw.shape[3] // self._stride)
            ratio = out.shape[2] / a_last
            self._nl = round(math.log(3 * ratio + 1, 4))
        return out

    def _detect_tta(self, img: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """augment=True 的路径: 三次前向 + 反变换 + 拼接(还没做 NMS).

        返回 (拼接后的原始预测, letterbox 缩放比 r, 补边 dw, dh) --
        预测框在 letterbox 空间, 逆映射由 detect() 统一做.
        """
        nh, nw = self._sizes(img.shape)
        _ = self.session            # 确保 stride/元数据已加载(取舍见 letterbox)
        base, r, dw, dh = letterbox(img, nh, nw, self._stride)
        x1 = _to_chw(base)
        outs = []
        passes = TTA_PASSES if self.augment else TTA_PASSES[:1]
        for pos, (scale, flip) in enumerate(passes):
            if scale == 1.0 and not flip:
                xi, in_hw = x1, (nh, nw)
            else:
                xf = x1[:, :, :, ::-1] if flip else x1   # 翻宽度轴(lv=3, 与 torch 一致)
                xi = _tta_scale(xf, scale, self._stride)
                in_hw = xi.shape[2:]
            out = self._forward(np.ascontiguousarray(xi))
            out = out.astype(np.float32)
            out[:, :4] /= scale
            if flip:
                out[:, 0] = nw - out[:, 0]     # img_size 取自原始 letterbox 尺寸
            out = _clip_augmented(out, in_hw, self._stride,
                                  self._nl or 3, pos, len(passes))
            outs.append(out)
        return np.concatenate(outs, axis=2), r, dw, dh

    def _sizes(self, shape: tuple) -> tuple[int, int]:
        """imgsz 参数(int 或 (h, w)) -> (h, w)."""
        if isinstance(self.imgsz, int):
            return self.imgsz, self.imgsz
        h, w = self.imgsz
        return int(h), int(w)

    def detect(self, image: np.ndarray,
               offset: tuple[float, float] = (0.0, 0.0)) -> list[Detection]:
        """跑一次推理, 返回原始检测框(未过滤). 与 YoloDetector.detect 同语义."""
        # bench 挂钩(默认关闭): 与 YoloDetector.detect 相同的计次/计时口径,
        # 一次 detect 调用 = 一次推理记录(内部 TTA 的多次前向不单计).
        with bench.inference(self.weight.stem):
            pred, r, dw, dh = self._detect_tta(image)
            dets, cls_ids = nms(pred, self.conf, self.iou, self._num_classes)
            dx, dy = offset
            out: list[Detection] = []
            for d, c in zip(dets, cls_ids):
                # NMS 在 letterbox 空间做(与 ultralytics 相同), 之后逆映射回原图.
                x1 = (d.x1 - dw) / r
                y1 = (d.y1 - dh) / r
                x2 = (d.x2 - dw) / r
                y2 = (d.y2 - dh) / r
                out.append(Detection(label=self.names[c], conf=d.conf,
                                     x1=x1 + dx, y1=y1 + dy,
                                     x2=x2 + dx, y2=y2 + dy))
            return out

    def warmup(self, shape: tuple[int, int, int] = (720, 1280, 3)) -> float:
        blank = np.zeros(shape, dtype=np.uint8)
        t0 = time.perf_counter()
        self.detect(blank)
        return time.perf_counter() - t0
