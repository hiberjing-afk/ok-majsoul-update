"""项目内固定路径常量.

集中管理, 避免各任务里散落硬编码的字符串路径.
所有路径都是相对仓库根目录的相对路径(ok-script 启动时 cwd = 仓库根).
"""

from pathlib import Path

# 标注目录 == ok-script 自带「模板」页签的工作目录(见 ok/core/template_store.py)。
#
# 用框架这个目录而不是自建 assets/annotations, 是为了让标注只有一份真源:
# 页签里截图 / 画框直接写 coco_annotations.json, 任务端 find_one 直接读它,
# 中间不需要任何"导出 / 转换"环节(历史教训见 docs/ARCHITECTURE.md)。
#
# 目录内容(截图 + COCO)进版本库(2026-10-25 起): 它是运行期唯一真源, 必须能被
# 下一版自动带上, 否则改一条标注就得手工重传一次资源包。改标注 = 普通提交。
TEMPLATES_DIR = Path("ok_templates")
COCO_ANNOTATIONS = TEMPLATES_DIR / "coco_annotations.json"

# YOLO 权重(对局内感知用). 86MB, 太大不进版本库(见 .gitignore),
# 用 tools/fetch_models.py 从上游仓库下载.
MODELS_DIR = Path("models")
TILE_MODEL = MODELS_DIR / "mahjong.pt"
UI_MODEL = MODELS_DIR / "majsoul_UI.pt"

# 运行时临时产物(不提交, 由 tools/cleanup.py 定期清理)
SCREENSHOTS_DIR = Path("screenshots")
DEBUG_ARTIFACTS_DIR = Path("data") / "debug"
LOGS_DIR = Path("logs")
