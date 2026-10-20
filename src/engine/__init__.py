"""事件驱动的对局内核(纯逻辑, 不碰截图/鼠标).

- **协议是牌局事实的唯一来源**: 一手从 `ActionNewRound` 开始, 到
  `ActionHule/NoTile/LiuJu` 结束; 轮到自己由 operation_list 的 type=1 说,
  鸣牌窗口由 type=2/3/5 说, 立直可打的牌由 type=7 的 combination 说.
- **视觉只回答坐标**: 手牌槽位/按钮框, 不回答"这张牌是什么".
- **一手一个控制器**: `HandController.on_event` 收到协议事件就决策, 不再
  轮询画面猜"发牌完了没有/上一手结束没有".

任务层通过 `GameState.on_event` 接入, 由 `runner.EngineRunner` 驱动一局.
"""

from __future__ import annotations
