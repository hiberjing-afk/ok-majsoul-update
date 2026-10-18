"""调试产物的自动修剪: 只删"每轮诊断帧", 证据类产物一律不碰.

为什么必须有(2026-10-17 实测): `save_frames` 每轮存一张 1080p PNG(约 2.4MB),
一轮 0.3s、一局 200+ 轮、连打多局还能跨天 —— 几天下来 `data/debug` 堆到
**74323 个文件 / 169.5 GB**(D 盘 1.7T 的一成), 而同一个目录里所有对局记录
(`game_*.jsonl`, 一局 0.3~11MB)+ 协议 spool 加起来只有 **50 MB**。
也就是说 99.9% 的磁盘被可再生的诊断帧吃掉了。

手动跑 `tools/cleanup.py` 不现实(没人会记得), 而且它原来的做法是**整个清空**
`data/debug` —— 会把 `game_*.jsonl` 对局记录和 liqi spool 一起删掉, 而那恰恰是
复盘、以及协议层回归夹具(`tests/fixtures/liqi_sanma_2rounds.jsonl`)的血缘。

策略: 只匹配每轮诊断帧 `frame_g<局>_t<轮>_<时刻>.png`, 三条规则取最严:
    1) 早于 `keep_days` 天的一律删;
    2) 文件数超过 `max_files` 时从**最老**的开始补删, 直到降到上限;
    3) 总字节超过 `max_bytes` 时同样从最老的补删。
"最老优先"是有意的: `tools/_frames.py` 的抽样也是取最近那一小时组的帧
(见那个模块的说明), 所以保新删旧正好对得上打标/挖训练集的工作流。

**永不自动删**(即使长得像帧):
    frame_lowconf_*   低置信留证帧(误识事故的唯一现场证据)
    frame_gate_*      牌桌特征失灵现场帧
    待标注_*          认不出的页面留证(等人工框区域)
    *.jsonl           对局记录 + 协议 spool
    *.log, tcp_tap*   mitm/TUN 侧产物

本模块只用标准库(不碰 cv2/torch/numpy), 单测可以直接 import。
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

#: 每轮诊断帧的命名: `frame_g<局号>_t<轮次>_<HHMMSS>.png`
TURN_FRAME_RE = re.compile(r"^frame_g\d+_t\d+_\d{6}\.png$")

#: 证据类前缀: 名字里也带 frame_, 但绝不能当诊断帧删掉.
PROTECTED_PREFIXES = ("frame_lowconf_", "frame_gate_", "待标注_")


def is_turn_frame(name: str) -> bool:
    """这个文件名是不是"可再生的每轮诊断帧"."""
    if name.startswith(PROTECTED_PREFIXES):
        return False
    return bool(TURN_FRAME_RE.match(name))


@dataclass
class PruneResult:
    """修剪结果. `freed` 在 dry-run 下是"将会释放"的字节数."""

    scanned: int = 0      # 扫到的诊断帧总数
    removed: int = 0      # 判为要删的张数(dry-run 下同样是"将要删")
    kept: int = 0
    freed: int = 0
    errors: int = 0

    @property
    def freed_mb(self) -> float:
        return self.freed / (1024 * 1024)

    @property
    def changed(self) -> bool:
        return self.removed > 0


def _collect(root: Path) -> tuple[list[tuple[float, int, Path]], int]:
    """扫出 (mtime, size, path) 并按 mtime 升序(最老的在前)."""
    entries: list[tuple[float, int, Path]] = []
    errors = 0
    with os.scandir(root) as it:
        for e in it:
            if not e.is_file() or not is_turn_frame(e.name):
                continue
            try:
                st = e.stat()
            except OSError:
                errors += 1
                continue
            entries.append((st.st_mtime, st.st_size, Path(e.path)))
    entries.sort(key=lambda t: t[0])
    return entries, errors


def prune_turn_frames(
    debug_dir: Path | str = "data/debug",
    *,
    keep_days: float = 3.0,
    max_files: int = 3000,
    max_bytes: int = 3 * 1024 ** 3,
    apply: bool = True,
) -> PruneResult:
    """按三条上限修剪每轮诊断帧, 保留最新的.

    `apply=False` 只统计(给 `tools\\cleanup.py` 的 dry-run 用)。
    `keep_days <= 0` 表示关闭时间规则(仍受文件数/体积限制)。
    `max_files <= 0` / `max_bytes <= 0` 同理表示关闭对应规则。

    **永不抛异常**: 目录不存在、权限不足、扫描到一半被删都只是当成"没事发生",
    清理永远是"锦上添花", 不该拦住开局(调用方还会再兜一层)。
    """
    res = PruneResult()
    root = Path(debug_dir)
    if not root.is_dir():
        return res

    try:
        entries, errors = _collect(root)
    except OSError:
        return res
    res.errors += errors
    res.scanned = len(entries)
    if not entries:
        return res

    age_cutoff = (time.time() - keep_days * 86400) if keep_days > 0 else None

    doomed: set[Path] = set()
    if age_cutoff is not None:
        doomed.update(p for mt, _, p in entries if mt < age_cutoff)

    keep = [(mt, sz, p) for mt, sz, p in entries if p not in doomed]
    total = sum(sz for _, sz, _ in keep)
    count = len(keep)
    for _, sz, p in keep:
        over_count = max_files > 0 and count > max_files
        over_bytes = max_bytes > 0 and total > max_bytes
        if not (over_count or over_bytes):
            break
        doomed.add(p)
        count -= 1
        total -= sz

    res.kept = len(entries) - len(doomed)
    for _, sz, p in entries:
        if p not in doomed:
            continue
        res.removed += 1
        res.freed += sz
        if not apply:
            continue
        try:
            p.unlink()
        except OSError:
            res.errors += 1
    return res
