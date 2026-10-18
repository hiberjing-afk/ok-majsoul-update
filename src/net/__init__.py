"""liqi 网络读牌: 只读解析雀魂 WebSocket 协议, 重建牌局模型.

模块划分(全部纯逻辑, 不 import ok-script):
    net.liqi     wire 层: ws 帧 -> (方法, 动作, 数据 dict)
    net.tiles    liqi 牌字符串校验(赤 5 / 记号折叠)
    net.state    liqi 事件 -> GameState(手牌/牌河/副露/dora/可执行操作)
    net.bridge   spool JSONL -> GameState 的尾读桥(给任务层用)

安全边界(与 AGENTS.md 反检测一节一致): 只读. 从不构造/修改任何发往
服务端的报文 -- 出牌一律仍走真实鼠标点击. 详见 docs/ARCHITECTURE.md 第七节.
"""
