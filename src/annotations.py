"""标注(原生 COCO)的读取与坐标换算(纯逻辑, 不依赖 ok-script, 可单测).

标注只有**一份真源**:

    ok_templates/coco_annotations.json   +  同目录下的截图 PNG

它由 ok-script 自带的「模板」页签产出(`ok/ui/qt/tasks/TemplateTab.py` 截图、
`MarkUpWindow.py` 画框), 我们不再自己实现标注器, 也不再有"自研 JSON 导出成
COCO"这一层 -- 历史教训见 docs/ARCHITECTURE.md 的「标注」一节。

两类约定:

- **类目名 = "页名/区域描述"**. 例: `大厅页/友人场`.
  框架要求类目名全局唯一(`FeatureSet.read_from_json` 遇到同名会抛
  `ValueError: Multiple boxes found for category`), 而"返回大厅"这类描述在
  多个页面上都有, 所以必须带页名前缀。带的另一个好处是流程定义里
  `(页名, 区域描述)` 这对字符串可以原样保留。
- **截图文件名必须是 ASCII**. 框架用 `cv2.imread` 读截图, 在 Windows 上不认
  非 ASCII 路径; 而且框架是拿"COCO 文件所在目录"去 join 这个文件名的, 所以
  只能写文件名(不能写绝对路径)。

本模块只做三件事: 把 COCO 读成 {类目名: Annotation}; 算"标注坐标系 ->
当前帧坐标系"的缩放比; 把标注框换算成"外扩了搜索余量"的帧内矩形。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

#: 标注截图的前缀: 任务遇到"认不出来的画面"会把当前帧丢进标注目录并带上
#: 这个前缀, 方便在「模板」页签里一眼认出"这是要补标注的那张".
#: 它不出现在 COCO 里, 所以不会被当成页面特征参与匹配.
UNANNOTATED_PREFIX = "待标注_"


@dataclass(frozen=True)
class Annotation:
    """一个标注框. 坐标是**标注截图坐标系**下的像素."""

    category: str                    # "大厅页/友人场"
    page: str                        # "大厅页"
    keyword: str                     # "友人场"
    image: str                       # 截图文件名(相对 COCO 所在目录)
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    image_size: tuple[int, int]      # 标注坐标系尺寸 (w, h)


def category_name(page: str, keyword: str) -> str:
    """类目名 = "页名/区域描述"(全局唯一)."""
    return f"{page}/{keyword}"


def split_category(category: str) -> tuple[str, str]:
    """"页名/区域描述" 拆回 (页名, 区域描述).

    只在**第一个**斜杠处切; 没有斜杠时整串当页名、区域描述为空串(标注工具允许
    不按约定命名, 这里不能抛异常, 否则一份坏标注会让整份 COCO 读不出来)。
    """
    page, _, keyword = category.partition("/")
    return page, keyword


def load_annotations(coco_path: str | Path) -> dict[str, Annotation]:
    """读 COCO, 返回 {类目名: Annotation}.

    文件不存在返回空字典(调用方据此报"标注不完整"); 文件在但读不动会抛异常
    -- 那种情况必须让调用方看见, 否则会伪装成"缺少某个页面", 排查方向全错。

    引用了不存在 id 的 annotation 会被跳过(单一坏条目不该让整份标注不可用),
    但同一类目出现两个不同框时抛异常: 框架那边一定会炸, 早点报出来更好。
    """
    coco_path = Path(coco_path)
    if not coco_path.is_file():
        return {}

    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {int(i["id"]): i for i in data.get("images") or []}
    categories = {int(c["id"]): c for c in data.get("categories") or []}

    out: dict[str, Annotation] = {}
    for ann in data.get("annotations") or []:
        image = images.get(int(ann.get("image_id", -1)))
        category = categories.get(int(ann.get("category_id", -1)))
        if image is None or category is None:
            continue
        name = str(category.get("name") or "")
        if not name:
            continue
        box = tuple(int(v) for v in (ann.get("bbox") or (0, 0, 0, 0)))
        if len(box) != 4:
            continue
        page, keyword = split_category(name)
        annotation = Annotation(
            category=name, page=page, keyword=keyword,
            image=str(image.get("file_name") or ""), bbox=box,
            image_size=(int(image.get("width", 0)), int(image.get("height", 0))),
        )
        existing = out.get(name)
        if existing is not None and existing.bbox != annotation.bbox:
            raise ValueError(
                f"类目 '{name}' 在 COCO 里有两个不同的框"
                f"({existing.bbox} 与 {annotation.bbox}); 框架不接受同名类目, "
                f"请把区域描述改成互不相同的名字"
            )
        out[name] = annotation
    return out


def category_names(annotations: Iterable[Annotation]) -> set[str]:
    return {a.category for a in annotations}


def missing_categories(required: Sequence[str],
                       have: set[str]) -> list[str]:
    """required 里缺哪些(保持输入顺序, 去重). 返回空表表示齐全."""
    missing: list[str] = []
    for name in required:
        if name not in have and name not in missing:
            missing.append(name)
    return missing


def scale_factor(ann_size: tuple[int, int],
                 frame_size: tuple[int, int]) -> tuple[float, float]:
    """标注坐标系 -> 当前帧坐标系的缩放比.

    标注截图和运行时帧的分辨率可能不同(窗口 resize / DPI 虚拟化 / 换显示器),
    两边必须换算, 否则**搜索区**会落到错误的位置。

    只用来换算搜索区: 模板本身的缩放由框架自己做(`FeatureSet.read_from_json`
    里的 adjust_coordinates, 按 COCO 声明的 width/height 对帧尺寸调整), 实测
    1080p 标注在 720p 帧上仍然命中 0.993。

    尺寸取不到(0)时按 1.0 处理。
    """
    aw, ah = ann_size
    fw, fh = frame_size
    sx = fw / aw if aw > 0 and fw > 0 else 1.0
    sy = fh / ah if ah > 0 and fh > 0 else 1.0
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        return 1.0, 1.0
    return sx, sy


def frame_box(bbox: Sequence[int], ann_size: tuple[int, int],
              frame_size: tuple[int, int],
              expand: float = 0.0) -> tuple[int, int, int, int]:
    """把标注框换算到帧坐标, 并向外扩 `expand` 倍作为搜索范围.

    外扩是为了容忍按钮在页面上有轻微偏移(窗口缩放 / UI 微调)。返回的矩形会
    夹到帧内, 且保证右下角不小于左上角。
    """
    fw, fh = frame_size
    sx, sy = scale_factor(ann_size, frame_size)
    x = int(round(int(bbox[0]) * sx))
    y = int(round(int(bbox[1]) * sy))
    w = max(1, int(round(int(bbox[2]) * sx)))
    h = max(1, int(round(int(bbox[3]) * sy)))

    ex, ey = int(w * expand), int(h * expand)
    x1 = max(0, x - ex)
    y1 = max(0, y - ey)
    x2 = min(int(fw), x + w + ex)
    y2 = min(int(fh), y + h + ey)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)
