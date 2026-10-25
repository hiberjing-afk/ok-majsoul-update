"""项目内运行时 `.runtime/` 的唯一定义: 东西装在哪、运行时怎么找.

分工只有一句: **装由 `tools/bootstrap.py` 负责, 找由这里负责**. 所以
`src/` 与 `tools/` 任何地方都不再出现候选路径表 —— 加一个运行时依赖 =
bootstrap 的 ASSETS 加一行, 消费方写 `runtime_paths.xxx` 取到, 到此为止.

> 注意: 本模块只依赖标准库. bootstrap 在依赖还没装的时候就 import 它.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
PYTHON = RUNTIME / "python" / "python.exe"
SITE = RUNTIME / "site-packages"
DEV_SITE = RUNTIME / "dev-site"
BIN = RUNTIME / "bin"
DOWNLOADS = RUNTIME / "downloads"
PIP_CACHE = RUNTIME / "pip-cache"
STAMP = RUNTIME / "runtime.json"
#: mitmproxy 的 CA 与配置. 默认在 `%USERPROFILE%\\.mitmproxy`, 用
#: `--set confdir` 挪进项目: 拷目录换机时证书跟着走, 不必重新签发再信任一次.
MITM_CONF = RUNTIME / "mitmproxy"

MIHOMO = BIN / "mihomo.exe"
CA_CERT_NAME = "mitmproxy-ca-cert.cer"

#: 用项目解释器直接跑 mitmdump 的入口函数(等价于 pip 的 console script).
#: 不给 mitmdump.exe / .cmd 外壳: 前者把安装时的解释器绝对路径写死(不可搬运),
#: 后者是 cmd.exe → python 两层进程, 杀外壳会留下还在监听端口的孤儿.
MITMDUMP_ENTRY = "from mitmproxy.tools.main import mitmdump; mitmdump()"


def find_mihomo(explicit: str = "") -> str | None:
    """mihomo 可执行文件: 显式/环境变量 > `.runtime/bin` > PATH.

    只认"项目内的那份"和"用户自己愿意提供的那份". 不去猜本机装过什么 ——
    猜出来的路径换台机器就没了, 那正是 2026-10-20 报"找不到 mihomo" 的原因.
    """
    for cand in (explicit, os.environ.get("OKMAJSOUL_MIHOMO", "")):
        if cand and Path(cand).is_file():
            return cand
    if MIHOMO.is_file():
        return str(MIHOMO)
    return shutil.which("mihomo") or shutil.which("verge-mihomo")


def mitmdump_cmd() -> list[str] | None:
    """启动 mitmdump 的命令前缀(不含参数); 没装好返回 None."""
    if not PYTHON.is_file() or not (SITE / "mitmproxy").is_dir():
        return None
    return [str(PYTHON), "-c", MITMDUMP_ENTRY]


def ca_cert_path() -> Path:
    return MITM_CONF / CA_CERT_NAME
