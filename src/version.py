"""版本与更新来源的单一出处.

发布流程见 docs/RELEASE.md。发布前必须把 UPDATE_REPO 填成真实的
GitHub 仓库（owner/repo 形式），否则启动时跳过更新检查。
"""

from __future__ import annotations

APP_VERSION = "2.0.0"

# GitHub 仓库（owner/repo）。空字符串 = 未发布状态，跳过更新检查。
UPDATE_REPO = "hiberjing-afk/ok-majsoul"

# Release 资产命名前缀（tools/release.py 与 CI 保持一致）
UPDATE_ASSET_PREFIX = "ok-majsoul"
