"""段位场自动打牌: 选 段位 x 对局, 自动匹配、循环连打.

与友人场的差别不在按钮名而在**流程形状**(见 rooms.RoomSpec.ranked):

    段位场  大厅 -> 段位场 -> 段位(银之间/金之间) -> 对局(四人东/四南/三东/三南)
            -> 自动匹配开局, 没有准备页/没有"开始"/没有加电脑
    友人场  大厅 -> 友人场 -> 创建房间 -> 选人数时长 -> 准备页点开始

界面选项 2026-10-25 用户要求细化成两段: 先选段位, 再选场次. 两张列表页都可能
需要滚动(最下面的卡片在 1080p 上露不全), 点它们走"找 -> 滚 -> 找"
(见 CreateFriendsRoomTask._click_list_item).
"""

from __future__ import annotations

from src.rooms import (
    RANKED_GAME_OPTIONS,
    RANKED_TIER_OPTIONS,
    RoomSpec,
    ranked_spec,
    spec_by_key,
)
from src.tasks.PlayOneGameTask import PlayOneGameTask
from src.widget_types import mark_unrenderable_hidden

# 两个下拉框的默认值: 取选项表第一项(银之间 / 四人东), 保证它们一定在
# options 里(见 CreateFriendsRoomTask.config_type 的说明: 默认值不在选项里
# 下拉框会显示空白).
_DEFAULT_CONFIG = {
    "tier": RANKED_TIER_OPTIONS[0],
    "game": RANKED_GAME_OPTIONS[0],
    "max_games": 1,
    "shutdown_after": False,
}

#: 段位场专属配置项的中文说明.
_DESCRIPTIONS = {
    "tier": "段位：银之间 / 金之间",
    "game": "场次：四人东 / 四人南 / 三人东 / 三人南",
}


class PlayRankedTask(PlayOneGameTask):
    """段位场: 自动匹配 + 循环连打."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "段位场自动打牌"
        self.description = (
            "段位场：选段位和场次，点进去自动匹配，打完接着排下一局"
        )
        self._init_default_config()
        self.config_description = {**self.config_description, **_DESCRIPTIONS}
        # 房型换成"段位 + 场次"两个下拉框. 基类(PlayOneGameTask)只管公共旋钮,
        # 这里补上本任务特有的两个. 默认值必须与 _DEFAULT_CONFIG 逐字相等.
        self.config_type = {
            "tier": {"type": "drop_down", "options": list(RANKED_TIER_OPTIONS)},
            "game": {"type": "drop_down", "options": list(RANKED_GAME_OPTIONS)},
            "max_games": {"min": 0, "max": 99},
        }
        # 换过 config_type 要重新隐藏"框架渲染不了的类型"(见基类说明).
        mark_unrenderable_hidden(self.config_type, self.default_config)

    def _default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    def _resolve_spec(self, cfg: dict) -> RoomSpec:
        """段位 + 场次 -> 规格. 非法组合由 rooms.ranked_spec 直接抛."""
        if cfg.get("room"):
            # 命令行/老配置写了 room 就以它为准(批量跑固定场次用).
            return spec_by_key(cfg["room"])
        return ranked_spec(
            cfg.get("tier") or RANKED_TIER_OPTIONS[0],
            cfg.get("game") or RANKED_GAME_OPTIONS[0],
        )
