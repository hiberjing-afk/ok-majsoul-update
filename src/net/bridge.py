"""spool JSONL -> GameState 的尾读桥(纯逻辑, 不依赖 ok-script).

数据流: mitmproxy addon(tools/mitm/liqi_addon.py) 把解码后的 liqi 事件
按行追加到 spool 文件; 本模块按字节偏移只读新增部分, 喂给 GameState.
主任务以后每轮 poll() 一次拿状态, 不需要知道 mitmproxy 的存在.

可用性语义 —— **只有三条, 没有"多久没事件"这一条**:

- `round_open` 为假 / 不知道自己是哪家 -> 这份状态不是"这一手"的, 上层停手.
  抓包没截到流量时状态自然停在这个分支(永远没有 ActionNewRound);
- `state.protocol_suspect` -> 协议版本可能漂了, 上层告警并停手;
- 链路死活**只问进程**(`capture.CaptureSupervisor.alive()`), 不问时间.

牌局里正常的静默就有十几秒(吃碰杠/立直/和了的应答窗口、立直后自动摸切、和了
演出), 用"多久没新事件"反推会把正常静默误判成断链; 子进程还在不在是更硬的证据.

文件协议: 一行一个 JSON(to_dict 的 LiqiMessage + ts/dir), 坏行跳过
并计数(半行写入是 append 模式的固有风险, 不能让它毒死整个桥).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .liqi import LiqiMessage
from .state import GameState


def spool_path_for(start: float | None = None) -> Path:
    """默认 spool 路径: data/debug/liqi_events_<时间戳>.jsonl."""
    from src.paths import DEBUG_ARTIFACTS_DIR

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(start or time.time()))
    return DEBUG_ARTIFACTS_DIR / f"liqi_events_{stamp}.jsonl"


class SpoolWriter:
    """addon 侧: 追加写 spool."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def write(self, msg: LiqiMessage, ts: float | None = None) -> None:
        record = msg.to_dict()
        record["ts"] = ts if ts is not None else time.time()
        record["dir"] = "c" if msg.from_client else "s"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1


class SpoolReader:
    """按字节偏移尾读 spool. 偏移只在内存里(进程重启从头读是安全的:

    状态机会按 step 去重, 重复事件不会双重应用)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._offset = 0
        self.bad_lines = 0

    def read_new(self) -> list[dict]:
        """返回自上次读取以来的新行(已解析的 dict 列表)."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self._offset:
            self._offset = 0        # 文件被截断/轮换了, 从头读
        if size == self._offset:
            return []
        out: list[dict] = []
        with self.path.open("r", encoding="utf-8") as fh:
            fh.seek(self._offset)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except (ValueError, TypeError):
                    self.bad_lines += 1   # 半行/坏行: 跳过, 不毒死桥
            self._offset = fh.tell()
        return out


@dataclass
class LiqiBridge:
    """任务层唯一的入口: poll() -> 最新 GameState."""

    path: Path | str
    state: GameState = field(default_factory=GameState)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._reader = SpoolReader(self.path)
        self._warned_missing = False

    def poll(self) -> int:
        """读新事件并应用到状态机. 返回应用的事件数."""
        applied = 0
        for record in self._reader.read_new():
            msg = LiqiMessage.from_dict(record)
            if self.state.apply(msg):
                applied += 1
        return applied

    def available(self) -> bool:
        """状态能不能拿去用(有牌局 + 知道自己是哪家 + 协议没可疑)."""
        if not self.state.round_open or self.state.my_seat is None:
            return False
        return not self.state.protocol_suspect

    def status(self) -> str:
        if not self.path.exists():
            return "no-spool"
        if not self.state.round_open:
            return "no-round"
        if self.state.my_seat is None:
            return "no-seat"
        if self.state.protocol_suspect:
            return "protocol-suspect"
        return "ok"
