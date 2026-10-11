"""启动时检查更新 + 自更新.

设计：
- 失败一律放行（离线 / 限流 / 仓库未配置都不阻塞启动），只在 stdout 提示。
- 更新 = 从 GitHub Releases 下载整包 zip，解压覆盖安装目录（代码与模板），
  **绝不覆盖**用户数据目录（configs/ models/ logs/ screenshots/ data/ ok_tasks/）。
- 纯逻辑（版本比较 / 覆盖白名单）与网络分离，纯逻辑有单元测试。

**私有仓库授权**（2026-10-10）：仓库保持私有，授权用户凭 GitHub
Fine-grained PAT（只读本仓库 Contents）下载 Release。令牌来源（按序）：
环境变量 `OKMAJSOUL_GH_TOKEN` -> `configs/update_token.txt`（安装器会把
安装时输入的令牌写在这里；configs 属用户数据，永不被更新覆盖）。
资产下载统一走 API 端点（Accept: application/octet-stream）——公开仓库
匿名可下，私有仓库带令牌可下，两条路同一个代码路径。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

# 用户数据目录：自更新解压时跳过
USER_DIRS = frozenset({"configs", "models", "logs", "screenshots", "data",
                       "ok_tasks", ".git", "__pycache__", ".venv"})

#: 令牌的环境变量名与默认落盘位置（configs 属用户数据，更新不覆盖）
TOKEN_ENV_VAR = "OKMAJSOUL_GH_TOKEN"
TOKEN_FILE = "configs/update_token.txt"

#: 主程序包资产名（严格形态 ok-majsoul-X.Y.Z.zip）。Release 上还挂着
#: ok-majsoul-setup-X.Y.Z.zip（安装器，只有 install.bat，没有代码/模板），
#: 挑错会毁掉自更新与资源引导——严格匹配优先，宽松兜底。
MAIN_PKG_RE = re.compile(r"^ok-majsoul-\d+\.\d+\.\d+\.zip$")


def load_token(env_var: str = TOKEN_ENV_VAR,
               config_path: Path | None = None) -> str:
    """读授权令牌：环境变量优先，其次 configs/update_token.txt。

    都没有就返回空串（= 匿名模式，只对公开仓库有效）。令牌文件读出来
    去掉首尾空白；文件不存在/读不了都当无令牌（自更新失败一律放行，
    这里也保持同一姿态——读令牌失败不该让程序崩）。
    """
    tok = (os.environ.get(env_var) or "").strip()
    if tok:
        return tok
    if config_path is None:
        config_path = (Path(__file__).resolve().parent.parent
                       / TOKEN_FILE)
    try:
        return config_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def parse_version(text: str) -> tuple[int, ...]:
    """'v1.2.3' -> (1, 2, 3)；解析失败返回空元组（视为不可比）。"""
    t = text.strip().lstrip("vV")
    parts: list[int] = []
    for chunk in t.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            return ()
        parts.append(int(digits))
    return tuple(parts)


def is_newer(remote: str, local: str) -> bool:
    """远端版本是否比本地新（语义化逐段比较，缺段按 0 补）。"""
    a, b = parse_version(remote), parse_version(local)
    if not a or not b:
        return False
    width = max(len(a), len(b))
    a = a + (0,) * (width - len(a))
    b = b + (0,) * (width - len(b))
    return a > b


@dataclass(frozen=True)
class ReleaseInfo:
    tag: str            # 如 "v1.0.1"
    notes: str
    asset_name: str
    asset_url: str      # API 端点(assets/{id}); 配 Accept: octet-stream 即可下载


def fetch_latest_release(repo: str, timeout: float = 8.0,
                         asset_prefix: str = "ok-majsoul",
                         token: str | None = None) -> ReleaseInfo | None:
    """查 GitHub Releases 最新版；任何失败返回 None（放行启动）。

    token=None 时自动 load_token()；公开仓库匿名即可，私有仓库必须带
    Fine-grained PAT（只读 Contents）。
    """
    if not repo or "/" not in repo:
        return None
    if token is None:
        token = load_token()
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    # 收集全部 zip 资产: 优先严格形态的主程序包, 挑不到再退回宽松匹配,
    # 避免抓到 ok-majsoul-setup-X.Y.Z.zip(安装器, 只有 install.bat).
    hits: list[tuple[str, str]] = []
    for asset in data.get("assets", []):
        name = asset.get("name", "")
        if name.startswith(asset_prefix) and name.endswith(".zip"):
            hits.append((name, asset.get("url", "")))
    if not hits:
        return None
    strict = [(n, u) for n, u in hits if MAIN_PKG_RE.match(n)]
    # 存 API 端点而不是 browser_download_url: 私有仓库的
    # browser_download_url 带令牌也 302 到匿名下载, 必走 API。
    asset_name, asset_url = strict[0] if strict else hits[0]
    return ReleaseInfo(tag=data.get("tag_name", ""),
                       notes=data.get("body", "") or "",
                       asset_name=asset_name,
                       asset_url=asset_url)


def check_for_update(repo: str, local_version: str,
                     asset_prefix: str = "ok-majsoul",
                     token: str | None = None) -> ReleaseInfo | None:
    """有可用更新返回 ReleaseInfo，否则 None。"""
    release = fetch_latest_release(repo, asset_prefix=asset_prefix, token=token)
    if release and is_newer(release.tag, local_version):
        return release
    return None


def download(url: str, dest: Path, timeout: float = 300.0,
             token: str | None = None) -> bool:
    """下载 release 资产到 dest；失败返回 False。

    资产 URL 是 API 端点, 必须带 Accept: application/octet-stream 才会
    返回二进制而不是 JSON 元数据; 私有仓库再加 Authorization。
    """
    if token is None:
        token = load_token()
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Accept": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp, \
                open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        return True
    except Exception:
        return False


def extract_update(zip_path: Path, install_dir: Path) -> list[str]:
    """把更新包解压覆盖进安装目录，跳过用户数据目录.

    返回实际覆盖的顶层条目名。zip 内若有完整配置/模型目录会被跳过，
    这样用户配置与权重永不回退。
    """
    overwritten: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename.replace("\\", "/")
            top = name.split("/")[0]
            if top in USER_DIRS:
                continue
            target = install_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if top not in overwritten:
                overwritten.append(top)
    return overwritten


def self_update(release: ReleaseInfo, install_dir: Path | None = None) -> bool:
    """下载最新包并解压覆盖。成功返回 True；全程不留半包（先下临时再替换）。"""
    root = install_dir or Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="ok-majsoul-update-") as td:
        pkg = Path(td) / release.asset_name
        if not download(release.asset_url, pkg):
            return False
        try:
            extract_update(pkg, root)
        except Exception:
            return False
    return True


# ---------------------------------------------------------------------------
# 首次运行资源引导（PyAppify 启动器渠道, 2026-10-11）
#
# 启动器只同步代码（tools/sync_update_repo.py 白名单），装好的 working 目录
# 里没有 ok_templates/ 与 models/。这里在启动时检测缺失并从主仓库 Release
# 的主程序包里**只**解压这两样（约 120MB，含隐私截图，永不进任何仓库）。
# ---------------------------------------------------------------------------

#: 资源在包内的顶层目录
RESOURCE_TOP_DIRS = frozenset({"ok_templates", "models"})

#: 缺一即触发引导的哨兵文件（models 只取推理用的 onnx，跳过 86MB 的 .pt）
RESOURCE_MARKERS = (
    "ok_templates/coco_annotations.json",
    "models/mahjong.onnx",
    "models/majsoul_UI.onnx",
)


def missing_resources(install_dir: Path) -> list[str]:
    """返回缺失的哨兵文件（空列表 = 资源齐了，不需要引导）。"""
    return [m for m in RESOURCE_MARKERS if not (install_dir / m).exists()]


def extract_resources(zip_path: Path, install_dir: Path) -> list[str]:
    """只解压 ok_templates/** 与 models/*.onnx，返回涉及的顶层目录。

    与 extract_update 的方向相反：那个跳过用户数据目录（更新已装好的
    环境），这个**只要**资源目录（全新安装补资源）。主程序包里 models/
    还有 .pt/.bak 等训练用权重，这里一律不解，省 100MB+ 流量。
    """
    touched: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename.replace("\\", "/")
            top = name.split("/")[0]
            if top not in RESOURCE_TOP_DIRS:
                continue
            if top == "models" and not name.endswith(".onnx"):
                continue
            target = install_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if top not in touched:
                touched.append(top)
    return touched


def ensure_resources(install_dir: Path | None = None,
                     repo: str | None = None,
                     *, fetch=None, download_fn=None) -> bool:
    """资源缺失时从最新 Release 拉主程序包补齐；成功补齐返回 True。

    fetch / download_fn 可注入（测试用）；失败只打印警告并返回 False，
    绝不抛异常阻塞启动——下次启动哨兵仍缺，会自动重试。
    """
    root = install_dir or Path(__file__).resolve().parent.parent
    missing = missing_resources(root)
    if not missing:
        return False
    print(f"[BOOTSTRAP] 缺少运行资源: {', '.join(missing)}")
    fetch = fetch or fetch_latest_release
    download_fn = download_fn or download
    if repo is None:
        from src.version import UPDATE_REPO
        repo = UPDATE_REPO
    release = fetch(repo)
    if release is None:
        print("[BOOTSTRAP] 查不到 Release（未发布/离线/无令牌），跳过。"
              "私有仓库需配置令牌：环境变量 OKMAJSOUL_GH_TOKEN 或 "
              "configs/update_token.txt")
        return False
    with tempfile.TemporaryDirectory(prefix="ok-majsoul-res-") as td:
        pkg = Path(td) / release.asset_name
        if not download_fn(release.asset_url, pkg):
            print("[BOOTSTRAP] 资源包下载失败，下次启动重试。")
            return False
        try:
            tops = extract_resources(pkg, root)
        except Exception as exc:
            print(f"[BOOTSTRAP] 资源解压失败: {exc}")
            return False
    still = missing_resources(root)
    if still:
        print(f"[BOOTSTRAP] 资源包里没有: {', '.join(still)}，"
              "请检查 Release 主程序包是否完整。")
        return False
    print(f"[BOOTSTRAP] 资源就绪（来自 {release.tag}: {', '.join(tops)}）。")
    return True


def bootstrap_resources_quietly() -> None:
    """main.py 启动时调用：资源引导永不阻塞启动。"""
    try:
        ensure_resources()
    except Exception:
        pass  # 与 check_update_quietly 同一姿态


if __name__ == "__main__":
    import sys

    from src.version import APP_VERSION, UPDATE_REPO

    release = check_for_update(UPDATE_REPO, APP_VERSION)
    if not release:
        token_from = ("环境变量" if os.environ.get(TOKEN_ENV_VAR)
                      else "配置文件" if load_token() else "无令牌")
        print(f"已是最新版本（{APP_VERSION}）或更新源未配置/不可达"
              f"（令牌: {token_from}）。")
        sys.exit(0)
    print(f"发现新版本 {release.tag}（当前 {APP_VERSION}），开始更新…")
    print(f"更新日志：\n{release.notes}\n")
    if self_update(release):
        print(f"[OK] 已更新到 {release.tag}，重新启动程序生效。")
    else:
        print("[FAIL] 更新失败（网络或磁盘问题），当前版本不受影响。")
        sys.exit(1)
