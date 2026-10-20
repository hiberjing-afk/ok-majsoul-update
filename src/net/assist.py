"""协议门面: 找 spool -> 建桥 -> 接事件 -> 读状态.

对局内核只吃协议事件(见 `src/engine/`), 视觉只回答坐标. 这一层负责把
`LiqiBridge` / `GameState` 包成任务层能用的少数几个问题:

    attach_events()  把内核的回调接到 GameState.on_event
    poll()           读 spool 新事件(桥是 pull 式的)
    available()      这一手能不能拿协议来决策
    ev_inputs()      出牌 EV 估值需要的协议事实

桥不可用(spool 没出现 / 抓包没开 / 文件坏了)时所有方法都安静返回保守值,
绝不向主循环抛异常.
"""

from __future__ import annotations

import time
from pathlib import Path

from src.mahjong.ev_eval import draws_from_wall, visible_from

from .bridge import LiqiBridge

#: spool 命名约定: tools/mitm/liqi_addon.py 落盘用 data.debug 的同一约定.
SPOOL_GLOB = "liqi_events_*.jsonl"
#: 只认这么久以内更新过的 spool. 旧文件一律当"上一场留下的", 不接.
DEFAULT_MAX_SPOOL_AGE = 90.0


def discover_spool(debug_dir: Path | str, *, now: float | None = None,
                   max_age: float = DEFAULT_MAX_SPOOL_AGE) -> Path | None:
    """在 data/debug 里找"当前正在写"的 spool: 最新、非空、且在 max_age 内."""
    root = Path(debug_dir)
    if not root.is_dir():
        return None
    now = time.time() if now is None else now
    best: tuple[Path, float] | None = None
    try:
        candidates = list(root.glob(SPOOL_GLOB))
    except OSError:
        return None
    for path in candidates:
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size <= 0 or now - stat.st_mtime > max_age:
            continue
        if best is None or stat.st_mtime > best[1]:
            best = (path, stat.st_mtime)
    return best[0] if best else None


class NetAssist:
    """任务层每轮 poll 一次的门面. 桥不可用时一切返回"用不了"."""

    def __init__(self, path: Path | str | None = None, *,
                 debug_dir: Path | str | None = None,
                 max_spool_age: float = DEFAULT_MAX_SPOOL_AGE):
        self._explicit = Path(path) if path else None
        self._debug_dir = debug_dir
        self._max_spool_age = max_spool_age
        self._bridge: LiqiBridge | None = None
        self._path: Path | None = None
        self._event_sink = None

    # ------------------------------------------------------------- lifecycle

    @property
    def path(self) -> Path | None:
        # 显式路径即使还没落盘也要报出来(任务层用它写日志/记录); 建桥仍然
        # 等文件出现, 见 _discover 与 attach_events 的 drain 标记.
        return self._path or self._explicit

    def _discover(self) -> Path | None:
        if self._explicit is not None:
            # 显式 spool 还没落盘时不建桥: attach 的 drain 标记要等到文件
            # 出现后由 _ensure_bridge 消费, 否则会把历史事件喂给新内核.
            return self._explicit if self._explicit.is_file() else None
        if self._debug_dir is None:
            return None
        return discover_spool(self._debug_dir, max_age=self._max_spool_age)

    def _ensure_bridge(self) -> LiqiBridge | None:
        if self._bridge is not None:
            return self._bridge
        path = self._discover()
        if path is None:
            return None
        self._path = path
        self._bridge = LiqiBridge(path)
        self._bridge.state.on_event = self._event_sink
        return self._bridge

    def attach_events(self, callback, *, drain: bool = False) -> bool:
        """把事件驱动内核接到桥的 `GameState.on_event` 上.

        `drain=True` 用于任务开局前: 先把旧 spool 里的历史事件读完(不接
        控制器), 再把水位清零, 之后只处理本场新事件.

        桥还没建(spool 未出现)也能先登记, `_ensure_bridge` 建桥时会补上.
        返回"当前是否已经接上"; 没接上不是错误(抓包还没开始).
        """
        self._event_sink = callback
        bridge = self._ensure_bridge()
        if bridge is None:
            return False
        if drain:
            bridge.state.on_event = None
            try:
                while bridge.poll():
                    pass
            except Exception:  # noqa: BLE001 - 历史坏行不该阻止本场
                pass
            self.reset_session()
        bridge.state.on_event = callback
        return True

    def poll(self) -> int:
        """读新事件. 返回应用数(没桥时 0).

        报文解析错误由 `GameState.apply` 吃掉; **引擎回调的异常向上抛**,
        由 `EngineRunner` 停手并记日志 —— 决策异常不能被当成"网络不可用"
        静默吞掉.
        """
        bridge = self._ensure_bridge()
        if bridge is None:
            return 0
        return bridge.poll()

    def status(self) -> str:
        bridge = self._ensure_bridge()
        if bridge is None:
            return "no-spool"
        try:
            return bridge.status()
        except Exception:  # noqa: BLE001
            return "bridge-error"

    def available(self) -> bool:
        bridge = self._ensure_bridge()
        if bridge is None:
            return False
        try:
            return bridge.available()
        except Exception:  # noqa: BLE001
            return False

    @property
    def state(self):
        bridge = self._bridge
        return bridge.state if bridge is not None else None

    def session(self) -> dict:
        """协议对"这场牌走到哪了"的判断(不依赖 available()).

        没有 state(桥没建起来/抓包没开)时返回保守值, 调用方按"协议不可用"处理.
        """
        st = self.state
        if st is None:
            return {"match_started": False, "round_open": False,
                    "game_over": False, "reason": "", "last_round_end": ""}
        return {
            "match_started": bool(getattr(st, "match_started", False)),
            "round_open": bool(getattr(st, "round_open", False)),
            "game_over": bool(getattr(st, "game_over", False)),
            "reason": str(getattr(st, "game_over_reason", "") or ""),
            "last_round_end": str(getattr(st, "last_round_end", "") or ""),
        }

    def match_started(self) -> bool:
        """协议说这一场开始了没有(等匹配成功时用). 先 poll 一次再问."""
        try:
            self.poll()
        except Exception:  # noqa: BLE001
            return False
        return self.session()["match_started"]

    def reset_session(self) -> None:
        """把"这一场"的起止标志清掉(连打时等下一场匹配之前调用).

        `last_step` 一起清: 水位是按局去的, 新一场第一手可能与上一场最后一手
        同号 —— 不清会把新一场的事件当 syncGame 重放吞掉.
        """
        st = self.state
        if st is None:
            return
        for attr, val in (("match_started", False), ("game_over", False),
                          ("game_over_reason", ""), ("round_open", False),
                          ("round_id", None), ("last_step", -1)):
            try:
                setattr(st, attr, val)
            except Exception:  # noqa: BLE001 - 老状态对象没这字段时不参战
                pass

    # ------------------------------------------------------------ EV inputs

    def table_view(self) -> dict:
        """桌况摘要(手牌/副露/宝牌/场风/牌山), 只取协议能确定给出的东西."""
        st = self.state
        if st is None:
            return {}
        try:
            return {
                "hand": list(st.hand),
                "melds": st.meld_count(),
                "melded": st.melded_tiles(),
                "meld_objs": list(st.melds),
                "doras": list(st.doras),
                "player_count": st.player_count,
                "field_wind": st.field_wind(),
                "seat_wind": (st.seat_wind(st.my_seat)
                              if st.my_seat is not None else None),
                "waits": list(st.my_waits),
                "wall": st.wall,
            }
        except Exception:  # noqa: BLE001
            return {}

    def ev_inputs(self) -> dict:
        """EV 出牌估值的入参(牌河/副露/宝牌指示牌/座位/牌山), 全部协议事实.

        把各家的牌河与副露摊平成 jansou 要的 `visible` 数组, 再把牌山换算成
        "我还能摸几张". 策略本身不在这里.
        """
        view = self.table_view()
        if not view:
            return {}
        st = self.state
        try:
            my_melds = list(st.melds)
            all_melds = list(my_melds)
            discards: list[str] = []
            for seat in sorted(st.players):
                player = st.players[seat]
                discards.extend(tile for tile, _ in player.discards)
                all_melds.extend(player.melds)
            indicators = list(view["doras"])
            my_seat = st.my_seat if st.my_seat is not None else 0
            # 人数交叉校验: player_count 在见过 AuthGame 之前是默认值 4, 而
            # 传错人数是静默灾难 —— 四人场按三麻算会让 2m-8m 永远组不成面子.
            player_count = view["player_count"]
            seats = len(st.players)
            if seats in (3, 4) and seats != player_count:
                player_count = seats
            return {
                "hand": list(view["hand"]),
                "my_melds": my_melds,
                "melds": view["melds"],
                "visible": visible_from(discards=discards, melds=all_melds,
                                        indicators=indicators),
                "draws": draws_from_wall(view["wall"], player_count),
                "discards": discards,
                "doras": indicators,
                "player_count": player_count,
                "round_wind": view["field_wind"],
                "seat_wind": st.seat_wind(my_seat),
                "my_seat": my_seat,
                "wall": view["wall"],
            }
        except Exception:  # noqa: BLE001
            return {}
