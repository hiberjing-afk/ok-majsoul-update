"""友人场自动打牌: 建房 -> 加电脑 -> 开始 -> 按牌效打.

**测试用**: 友人场不计段位, 是唯一能随便跑的环境(见 AGENTS 的反检测一节).
房型是两个下拉框组合出来的: 人数(三人/四人) x 时长(一局/东风局), 有效组合
由 rooms.spec_by_combination 决定, 没注册的组合会明确报错.

与段位场的差别全在 rooms.RoomSpec.ranked: 这里有准备页、要加电脑、点"开始";
段位场是点完场次自动匹配.
"""

from __future__ import annotations

from src.rooms import (
    LENGTH_OPTIONS,
    PLAYER_OPTIONS,
    RoomSpec,
    spec_by_combination,
    spec_by_key,
)
from src.tasks.CreateFriendsRoomTask import DEFAULT_LENGTH, DEFAULT_PLAYERS
from src.tasks.PlayOneGameTask import PlayOneGameTask
from src.widget_types import mark_unrenderable_hidden

_DEFAULT_CONFIG = {
    "players": DEFAULT_PLAYERS,   # "三人" / "四人"
    "length": DEFAULT_LENGTH,     # "一局" / "东风局"
    # room = players + length 推导出来的, **不要手填**; 只在命令行批量跑时写.
    "room": "",
    "max_games": 1,
    "shutdown_after": False,
}

_DESCRIPTIONS = {
    "players": "人数：三人 / 四人",
    "length": "时长：一局 / 东风局",
}


class PlayFriendsTask(PlayOneGameTask):
    """友人场: 建房加电脑 + 按牌效打(不计段位, 测试用)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "友人场自动打牌"
        self.description = (
            "友人场（测试用，不计段位）：建房加电脑，按牌效打完指定的局数"
        )
        self._init_default_config()
        self.config_description = {**self.config_description, **_DESCRIPTIONS}
        self.config_type = {
            "players": {"type": "drop_down", "options": list(PLAYER_OPTIONS)},
            "length": {"type": "drop_down", "options": list(LENGTH_OPTIONS)},
            # room 是推导出来的, hidden 让它不渲染但值仍能从老配置读到.
            "room": {"type": "drop_down", "options": [], "hidden": True},
            "max_games": {"min": 0, "max": 99},
        }
        mark_unrenderable_hidden(self.config_type, self.default_config)

    def _default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    def _resolve_spec(self, cfg: dict) -> RoomSpec:
        """人数 + 时长 -> 规格. 非法组合直接抛, 不静默回退."""
        if cfg.get("room"):
            return spec_by_key(cfg["room"])
        return spec_by_combination(
            cfg.get("players") or DEFAULT_PLAYERS,
            cfg.get("length") or DEFAULT_LENGTH,
        )
