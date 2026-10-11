"""可选的运行时基准测量(instrumentation), **默认完全关闭**.

用途: 建立可重复的性能基线 —— 截图/手牌识别/模型推理的频率与耗时、
主循环周期、出牌 vs 非出牌阶段的手牌识别次数、CPU/GPU 占用、推理异常数.

开关与时长(环境变量, 不设就全程零开销):
    OKMAJSOUL_BENCH=1              开启采集
    OKMAJSOUL_BENCH_SECONDS=180    固定时长(秒), 到时自动落盘; 不设则跑
                                   到程序退出时经 atexit 落盘

关闭时的开销: 每个挂钩点一次属性读取, 无 I/O、无线程、无行为变化.
开启时的开销: 事件追加写 JSONL(每秒约十几行) + 每秒一次 CPU/GPU 采样线程,
全部为"记录后原样放行", 异常照常抛出(只多计一次 error), 不改变任何行为.

输出(bench_out/benchmark_<启动时间戳>/):
    events.jsonl    结构化事件日志, 每行一个 JSON 事件
                    (grab / hand_read / inference / state / loop / sample / error)
    inferences.csv  每次模型推理一行: name, t_s, ms
    summary.json    汇总: 频率、P50/P95/P99、阶段分布、CPU/GPU 统计

阶段(phase)的语义(由 PlayOneGameTask 主循环在每次读手牌前设置):
    active        出牌阶段: 发牌闸门已开(DealWatch.opened)且不在和了封锁
    deal_pending  非出牌阶段: 发牌中/换手后闸门未开
    settling      非出牌阶段: 和了封锁期(等摊牌/结算)
    gave_up       非出牌阶段: 牌桌特征失灵超过 TABLE_WAIT_LIMIT 后退回旧行为
    unknown       读手牌发生在任务上下文之外(如预热)

CPU 采样用 psutil(系统 + 本进程); GPU 用 pynvml(NVML, 0 号卡整机占用,
不区分进程 —— DirectML 推理不会体现在进程级指标里, 只能看整机).
两者缺失时 summary 里如实标注原因, 不伪造数据.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

#: 出牌阶段的唯一标签; 其余全部算"非出牌阶段".
ACTIVE_PHASE = "active"

# 出牌阶段手牌读数/推理的时延都落在几十~几百毫秒一档, 列表存原始值,
# 落盘时才算分位数 —— 量级(每秒几条)远不需要环形缓冲.


def _pct(values: list[float], p: float) -> float | None:
    """线性插值分位数. 空列表返回 None, 不参与 JSON 时按 null 落盘."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _lat_stats(ms: list[float]) -> dict:
    """一组耗时(毫秒)的统计: count/mean/p50/p95/p99/max."""
    if not ms:
        return {"count": 0}
    return {
        "count": len(ms),
        "mean_ms": round(sum(ms) / len(ms), 2),
        "p50_ms": round(_pct(ms, 0.50), 2),
        "p95_ms": round(_pct(ms, 0.95), 2),
        "p99_ms": round(_pct(ms, 0.99), 2),
        "max_ms": round(max(ms), 2),
    }


class _GrabMark:
    """grab() 上下文里让调用方标记"这次截图没拿到新帧"的小句柄."""

    def __init__(self) -> None:
        self.failed = False

    def fail(self) -> None:
        self.failed = True


class Bench:
    """全局单例(模块尾部的 `bench`). 所有方法在关闭时都是零副作用快速返回."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("OKMAJSOUL_BENCH", "").strip() == "1"
        self.duration = float(os.environ.get("OKMAJSOUL_BENCH_SECONDS", "0") or 0)
        #: 当前阶段标签, 由任务主循环在读手牌前设置(见模块 docstring).
        self.phase: str = "unknown"

        self._t0 = 0.0                     # bench 起算时刻(time.time())
        self._finished = not self.enabled
        self._lock = threading.Lock()
        self._events = None                # events.jsonl 的文件句柄
        self._out_dir: Path | None = None
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None
        self._cpu_note: str | None = None  # 采不了 CPU/GPU 时如实写原因
        self._gpu_note: str | None = None

        # 原始记录(落盘时汇总; events.jsonl 是实时流, 这里是分位数原料)
        self._last_tick: float | None = None
        self._loop_periods: list[float] = []   # 秒
        self._grabs: list[tuple] = []          # (t_s, ok, ms)
        self._hand_reads: list[tuple] = []     # (t_s, phase, ms)
        self._inferences: list[tuple] = []     # (name, t_s, ms)
        self._states: dict[str, int] = {}      # 门控 kind -> 次数
        self._errors: dict[str, int] = {}      # 异常来源 -> 次数
        self._samples: list[tuple] = []        # (t_s, cpu_sys, cpu_proc, gpu, mem)

        if self.enabled:
            self._start()

    # ---- 生命周期 -----------------------------------------------------
    def _start(self) -> None:
        self._t0 = time.time()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._out_dir = Path("bench_out") / f"benchmark_{stamp}"
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._events = open(self._out_dir / "events.jsonl", "a",
                            encoding="utf-8")
        self._sampler = threading.Thread(target=self._sampler_loop,
                                         name="bench-sampler", daemon=True)
        self._sampler.start()
        atexit.register(self.finish)
        print(f"[BENCH] 已开启基准采集, 每 {self.duration:.0f}s 自动落盘"
              if self.duration > 0 else
              f"[BENCH] 已开启基准采集, 程序退出时落盘")
        print(f"[BENCH] 输出目录: {self._out_dir}")

    def finish(self) -> None:
        """停止采样并把汇总写盘. 幂等: 重复调用只落盘一次."""
        if not self.enabled or self._finished:
            return
        self._finished = True
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=2.0)
        elapsed = time.time() - self._t0
        with self._lock:
            inf = list(self._inferences)
            grabs = list(self._grabs)
            hand = list(self._hand_reads)
            loops = list(self._loop_periods)
            states = dict(self._states)
            errors = dict(self._errors)
            samples = list(self._samples)
        try:
            self._write_outputs(elapsed, inf, grabs, hand, loops,
                                states, errors, samples)
        finally:
            if self._events is not None:
                self._events.close()
                self._events = None

    # ---- 挂钩点(任务/检测器调用) ---------------------------------------
    def tick_loop(self) -> None:
        """主循环每轮开头调用一次: 记录相邻两轮的间隔(= 主循环周期)."""
        if not self.enabled or self._finished:
            return
        now = time.monotonic()
        with self._lock:
            if self._last_tick is not None:
                self._loop_periods.append(now - self._last_tick)
            self._last_tick = now
        self._emit({"ev": "loop", "t": self._rel()})
        if self.duration > 0 and time.time() - self._t0 >= self.duration:
            self.finish()

    def set_phase(self, phase: str) -> None:
        """设置当前阶段标签(下一次 hand_read 记录用它). 无条件执行, 开销为零."""
        self.phase = phase

    @contextmanager
    def grab(self):
        """包住一次截图: 计次 + 计时; 调用方用 mark.fail() 标记没拿到新帧."""
        mark = _GrabMark()
        if not self.enabled or self._finished:
            yield mark
            return
        t0 = time.perf_counter()
        yield mark
        ms = (time.perf_counter() - t0) * 1000.0
        t = self._rel()
        ok = not mark.failed
        with self._lock:
            self._grabs.append((t, ok, ms))
        self._emit({"ev": "grab", "ok": ok, "ms": round(ms, 2), "t": round(t, 3)})

    @contextmanager
    def hand_read(self):
        """包住一次手牌识别(GameVision.read_hand): 计次 + 计时 + 当前阶段."""
        if not self.enabled or self._finished:
            yield
            return
        phase = self.phase
        t0 = time.perf_counter()
        yield
        ms = (time.perf_counter() - t0) * 1000.0
        t = self._rel()
        with self._lock:
            self._hand_reads.append((t, phase, ms))
        self._emit({"ev": "hand_read", "phase": phase,
                    "ms": round(ms, 2), "t": round(t, 3)})

    @contextmanager
    def inference(self, name: str):
        """包住一次模型推理(detect): 计次 + 计时; 异常计数后原样抛出."""
        if not self.enabled or self._finished:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        except Exception:
            with self._lock:
                self._errors[name] = self._errors.get(name, 0) + 1
            self._emit({"ev": "error", "where": name, "t": self._rel()})
            raise
        ms = (time.perf_counter() - t0) * 1000.0
        t = self._rel()
        with self._lock:
            self._inferences.append((name, t, ms))
        self._emit({"ev": "inference", "name": name,
                    "ms": round(ms, 2), "t": round(t, 3)})

    def count_state(self, kind: str) -> None:
        """计一次游戏状态识别(页面门控/模板探测). kind 如 ready_page/table_gate."""
        if not self.enabled or self._finished:
            return
        with self._lock:
            self._states[kind] = self._states.get(kind, 0) + 1
        self._emit({"ev": "state", "kind": kind, "t": self._rel()})

    # ---- 内部 ----------------------------------------------------------
    def _rel(self) -> float:
        return round(time.time() - self._t0, 3)

    def _emit(self, event: dict) -> None:
        if self._events is None:
            return
        with self._lock:
            self._events.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._events.flush()

    def _sampler_loop(self) -> None:
        """每秒采一次 CPU(系统+进程) 与 GPU(整机). 依赖缺失时只记原因."""
        psutil = proc = None
        try:
            import psutil
            proc = psutil.Process()
            psutil.cpu_percent(interval=None)     # 首调只做基线, 返回 0.0
            proc.cpu_percent(interval=None)
        except Exception as e:
            self._cpu_note = f"psutil 不可用: {e}"
        nvml = handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            nvml = pynvml
            pynvml.nvmlDeviceGetUtilizationRates(handle)  # 预热
        except Exception as e:
            self._gpu_note = f"pynvml/NVML 不可用(0 号卡): {e}"
            nvml = handle = None
        while not self._stop.wait(1.0):
            try:
                t = self._rel()
                cpu_sys = psutil.cpu_percent(interval=None) if psutil else None
                cpu_proc = proc.cpu_percent(interval=None) if proc else None
                gpu_util = gpu_mem = None
                if nvml is not None:
                    u = nvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = u.gpu
                    gpu_mem = nvml.nvmlDeviceGetMemoryInfo(handle).used / 1048576
                with self._lock:
                    self._samples.append((t, cpu_sys, cpu_proc,
                                          gpu_util, gpu_mem))
                self._emit({"ev": "sample", "t": round(t, 2),
                            "cpu_sys": cpu_sys, "cpu_proc": cpu_proc,
                            "gpu_util": gpu_util,
                            "gpu_mem_mb": round(gpu_mem) if gpu_mem else None})
            except Exception:
                continue  # 单次采样失败不影响后续(不伪造数据, 只是少一个点)

    def _write_outputs(self, elapsed: float, inf, grabs, hand, loops,
                       states, errors, samples) -> None:
        out = self._out_dir
        # 1) 每次推理一行 CSV
        with open(out / "inferences.csv", "w", encoding="utf-8", newline="") as f:
            f.write("name,t_s,ms\n")
            for name, t, ms in inf:
                f.write(f"{name},{t:.3f},{ms:.2f}\n")
        # 2) 汇总
        by_name: dict[str, list[float]] = {}
        for name, _, ms in inf:
            by_name.setdefault(name, []).append(ms)
        active = [t for t, ph, _ in hand if ph == ACTIVE_PHASE]
        non_active = [t for t, ph, _ in hand if ph != ACTIVE_PHASE]
        phase_break: dict[str, int] = {}
        for _, ph, _ in hand:
            phase_break[ph] = phase_break.get(ph, 0) + 1

        def series(idx: int, pred=None) -> list[float]:
            return [s[idx] for s in samples
                    if s[idx] is not None and (pred is None or pred(s))]

        cpu_sys = series(1)
        cpu_proc = series(2)
        gpu_util = series(3)
        gpu_mem = series(4)
        grab_ok = sum(1 for _, ok, _ in grabs if ok)
        summary = {
            "started_at": datetime.fromtimestamp(self._t0).isoformat(),
            "elapsed_s": round(elapsed, 1),
            "duration_limit_s": self.duration or None,
            "phase_labels": {
                "active": ACTIVE_PHASE,
                "non_active": "其余全部(deal_pending/settling/gave_up/unknown)",
            },
            "rates_per_s": {
                "screenshots": round(len(grabs) / elapsed, 3) if elapsed else None,
                "hand_reads": round(len(hand) / elapsed, 3) if elapsed else None,
                "model_inferences": round(len(inf) / elapsed, 3) if elapsed else None,
                "hand_reads_active_phase": round(len(active) / elapsed, 3) if elapsed else None,
                "hand_reads_non_active_phase": round(len(non_active) / elapsed, 3) if elapsed else None,
            },
            "counts": {
                "screenshots": len(grabs),
                "screenshots_failed": len(grabs) - grab_ok,
                "hand_reads": len(hand),
                "hand_reads_active_phase": len(active),
                "hand_reads_non_active_phase": len(non_active),
                "hand_reads_phase_breakdown": phase_break,
                "model_inferences": len(inf),
                "model_inferences_by_name": {k: len(v) for k, v in by_name.items()},
                "game_state_recognitions": states,
                "errors": errors,
            },
            "latency": {
                "model_inference_by_name": {k: _lat_stats(v)
                                            for k, v in by_name.items()},
                "model_inference_all": _lat_stats([ms for _, _, ms in inf]),
                "hand_read": _lat_stats([ms for _, _, ms in hand]),
                "screenshot": _lat_stats([ms for _, _, ms in grabs]),
                "main_loop_period": _lat_stats([p * 1000 for p in loops]),
            },
            "cpu": {
                "note": self._cpu_note,
                "system_percent": _lat_stats(cpu_sys),
                "process_percent": _lat_stats(cpu_proc),
            },
            "gpu": {
                "note": self._gpu_note or
                        "NVML 整机占用(0 号卡), 不区分进程; DirectML 推理无进程级指标",
                "util_percent": _lat_stats(gpu_util),
                "mem_used_mb": _lat_stats(gpu_mem),
            },
            "samples_count": len(samples),
        }
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[BENCH] 采集结束({elapsed:.0f}s), 结果已写入 {out}")


#: 全局单例. import 即读环境变量: 未设 OKMAJSOUL_BENCH 时 enabled=False,
#: 所有挂钩点走快速返回路径, 行为与没有这个模块完全一致.
bench = Bench()
