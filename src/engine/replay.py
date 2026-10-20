"""把一份 liqi spool 回放进新内核, 产出决策序列.

这是离线验收工具: 把一份真实 spool 重放进内核, 逐手检查决策是否合理;
`tests/TestEngineReplay.py` 用固定夹具锁住庄家 14 张、碰、下一手重置.
也可以直接 `python -m src.engine.replay <spool>` 看一遍.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

from src.net.bridge import SpoolReader
from src.net.liqi import LiqiMessage
from src.net.state import GameState

from .decisions import Decision
from .hand import HandController


@dataclass(frozen=True)
class ReplayStep:
    step: int
    action: str
    decision: Decision
    ts: float


@dataclass
class ReplayResult:
    steps: list[ReplayStep] = field(default_factory=list)
    state: GameState = field(default_factory=GameState)


def replay_spool(path: Path | str, *, rng_seed: int = 0,
                 allow_riichi: bool = True) -> ReplayResult:
    """回放一份 spool. 不碰网络/屏幕, 纯状态机 + 规则."""
    result = ReplayResult(state=GameState())
    current = {"step": -1, "action": "", "ts": 0.0}

    def collect(decision: Decision) -> None:
        result.steps.append(ReplayStep(
            step=current["step"], action=current["action"],
            decision=decision, ts=current["ts"]))

    controller = HandController(collect, rng=random.Random(rng_seed),
                                allow_riichi=allow_riichi)

    def on_event(action: str, data: dict, step: int, state: GameState) -> None:
        current.update(step=step, action=action)
        controller.on_event(action, data, step, state)

    result.state.on_event = on_event

    for record in SpoolReader(path).read_new():
        current["ts"] = float(record.get("ts") or 0.0)
        try:
            msg = LiqiMessage.from_dict(record)
        except Exception:  # noqa: BLE001 - 坏行/旧版本字段, 跳过
            continue
        result.state.apply(msg)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description="回放 liqi spool 并打印新内核决策")
    parser.add_argument("spool", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-riichi", action="store_true")
    args = parser.parse_args()
    result = replay_spool(args.spool, rng_seed=args.seed,
                          allow_riichi=not args.no_riichi)
    for item in result.steps:
        d = item.decision
        print(f"step={item.step:>4} {item.action:<24} -> "
              f"{d.action:<9} tile={d.tile or '-':<3} "
              f"tiles={','.join(d.tiles) or '-':<12} {d.reason}")
    print(f"\n{len(result.steps)} decisions, "
          f"rounds={result.state.round_seq}, game_over={result.state.game_over}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
