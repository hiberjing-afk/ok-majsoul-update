from __future__ import annotations

"""打一整局友人场三人麻将(可循环连打).

在「创建友人场房间」的基础上多做两件事: 点"开始", 然后把这一局打完.
开房流程、模板匹配、拟人点击全部复用 CreateFriendsRoomTask, 这里只加对局内的
感知-决策-执行循环.

一局的循环是这样的(每轮一次抓帧):

    1. 先看是不是回到准备页了(准备页的静态特征) -> 是就说明这局结束了
    2. 再看是不是结算页 -> 是就点"确认", 一直点到离开(一局可能连着好几页)
    3. 牌桌出来了没有? 没有就整轮跳过(加载 / 对局开始页 / 和了过场),
       不读手牌也不决策 —— 见 TABLE_SCREEN 的实测数值
    4. YOLO 认手牌(牌面模型) + 认按钮(UI 模型)
    5. src/mahjong/table.py 决定这一轮干什么(和牌 / 立直 / 拔北 / 跳过 /
       确认 / 出牌 / 等)
    6. 这一手还在不在打? 不在(发牌中 / 上一手已结束)就只准等, 不许出牌 /
       立直 / 副露 / 拔北 —— 见 table.DealWatch + DEAL_BLOCKED_ACTIONS
    7. 执行: 点按钮 或 点手牌; 出牌会再抓一帧确认手牌真的少了一张
    8. 每一轮的全部信息写进 data/debug/game_<时间戳>.jsonl

为什么要记这么细: 牌面模型只在真实对局里才见得到牌, 大厅里根本没法验证.
日志里留了每一帧认出的手牌(带置信度和坐标) / 认出的按钮 / 决策理由 /
点击坐标 / 点击后的手牌数, 打完一局回来照着日志就能判断识别准不准.

出牌的点击方式(单击直接打出 还是 选中再确认)是**自适应**的: 点一下之后
手牌没少就再点一下同一位置, 并把这件事记下来, 下次直接按确认模式走.

认不出来的过渡页面(结算页之类)会截图存到标注目录(带 `待标注_` 前缀) --
在 GUI 的「模板」页签里找到它、框出几个区域, 流程就能认了.

**卡死检测**: 对局结束会进结算页, 画面一动不动, 而牌河会被牌面模型误识成
"手牌"(实测 10 张), 于是每一轮都判成"在等别人", 一直空转到 game_timeout
(实跑踩过: 白等 15 分钟). 现在连续 stuck_after_turns 轮画面没变化就判定
"这局已经结束但页面没认出来", 截图留证后返回, 不再空转.

**前台**: ok-script 的 executor 在游戏不在前台时会整个挂起(取任务之前就
卡住), 所以拉前台做在交互层 src/interactions.py 的 should_capture 里 --
不需要手动把游戏点上前台.

配置: configs/PlayOneGameTask.json。界面上只有三个用户旋钮:
    players / length         房型(两个下拉框, 组合出房间规格)
    max_games                连打几局(0 = 只建房不开始)
其余阈值与延迟全部是 `src/tasks/tuning.py` 里的代码常量.
每轮存帧(`save_frames`)与稳健出牌开关(`robust_discard`)已移除: 正常帧
没有诊断价值(协议信任档落地), 稳健出牌恒开(它只在低置信出现时参战).
"""

import json
import random
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from src.annotations import missing_categories
from src.detection import (
    low_confidence,
    missing_slots,
    split_bright_dim,
    split_setaside,
    tile_brightness,
)
from src.detector import GameVision
from src.imageio import imwrite_unicode
from src.instrumentation import bench
from src.mahjong import timing
from src.mahjong.table import (
    ACTION_BABEI,
    ACTION_DISCARD,
    ACTION_MELD,
    ACTION_RIICHI,
    BUTTON_GANG,
    TABLE_GAVE_UP,
    DealWatch,
    TableWatch,
    action_confirmed,
    decide_turn,
    expected_concealed_counts,
    meld_count_of,
    pick_tile_index,
)
from src.net.assist import NetAssist
from src.paths import DEBUG_ARTIFACTS_DIR
from src.retention import prune_turn_frames
from src.rooms import RoomSpec, ready_marker, required_categories, spec_by_key, start_step
from src.tasks import tuning
from src.tasks.CreateFriendsRoomTask import _DEFAULT_CONFIG as _BASE_CONFIG
from src.tasks.CreateFriendsRoomTask import CreateFriendsRoomTask, TaskStopped
from src.tasks.hand_loop import HandLoop, RoundState
from src.vision import StuckDetector, is_toggle_green

# 结算页(对局结束的点数结算). 页面名与区域名对应
# ok_templates/coco_annotations.json 里的 "结算页/终局" 与 "结算页/确认".
# 不用 RoomScreens 字段: 结算页与房间规格无关, 任何规格打完都是它.
RESULT_SCREEN = "结算页"
RESULT_FEATURE = "终局"# 只用来探测"是不是结算页"
RESULT_CONFIRM = "确认"    # 点它回到准备页

# 终局奖励页(2026-10-11 用户报: 段位场打过几局后偶发弹出「获得奖励」页).
# 页面名与区域名对应 ok_templates/coco_annotations.json 里的 "奖励页/获得奖励".
# 这一页**没有可匹配的确认按钮**(现有结算页判据「终局」标题/「确认」按钮都
# 命不中它), 画面一动不动, 不处理就会卡死在页面上. 用户实测: 在原结算页
# 「确认」按钮的位置点一下就能翻页 —— 所以判页靠「获得奖励」标题, 点击照
# 结算页确认按钮的标注位置(见 tuning.RESULT_CONFIRM_CENTER, 1080p 基准).
REWARD_SCREEN = "奖励页"
REWARD_FEATURE = "获得奖励"  # 只用来探测"是不是奖励页"

# 对局中的画面特征: "牌桌已经画出来了". 对应 COCO 里的 "牌桌页/操作条".
#
# 用它当**整轮感知与决策的开关**, 而不是继续靠"手牌认出来几张"判断
# (2026-10-07 用户提的做法): 点完"开始"之后先加载、再过一个"对局开始页"
# (中间写着「胜负」那一页, 会停好几秒), 这段时间手牌区里根本没有手牌,
# 牌面模型只能从立绘花纹里幻觉出零星几张, 于是每一轮都刷"只认出 0~3 张牌"
# 和"这一轮按稳健出牌算" —— 实跑开局刷了 20 轮(约 10 秒)全是这种噪音,
# 而真正该做的是"等牌桌出来"。
#
# 为什么挑这条**左侧「理/和/鸣/切/拔」操作条**当特征(实测 ~300 张帧):
#
#   场景                    最高/最低相似度
#   对局内(含发牌/立直/过场)   最低 0.986
#   加载页 / 对局开始页         最高 0.283
#   大厅 / 各准备页 / 活动 / 段位选择  最高 0.29
#   结算页(result.png)        0.988（也算"在牌桌", 但结算页在前面就处理掉了）
#   每日任务完成页(实拍)        0.330
#   和了摊牌过场(t114/t115)    0.226 —— 特征消失, 于是过场里不会乱动手
#
# 它是**静态** UI: 位置固定、不随牌桌的推拉镜头移动, 这正是"牌桌中央那块"
# 不能用的原因 —— 牌桌在 0.7 倍和 1.0 倍两个镜头下, 同一块模板相似度
# 只有 0.72, 换个回合就失效(实测 t30 是 0.72、t100 是 1.000)。
TABLE_SCREEN = "牌桌页"
TABLE_FEATURE = "操作条"

# 一手结束的顺位结算画面(和了/流局之后弹出): 四个玩家的顺位/点数变动面板 +
# 右下角「(N) 确认」按钮(N = 还没点的人数, **动态**; 「确认」二字是静态 UI,
# 模板只框这两个字, 实测对 (1)/(2)/(3) 三种计数都命中 1.000, 对正常对局帧
# 最高只有 0.37 —— 阈值 0.88 之下裕度巨大).
#
# 为什么要专门认它(2026-10-08 实跑): 这画面上自己的手牌行被面板遮掉一截,
# 牌面模型读出 9~11 张 + 一排空档, 每一手结束都要刷 3~6s 的"张数不可信"
# 假警报, 用户看到的就是"识别不到一把牌结束、总说牌没读到"; 而且画面停留
# 期间 done 还是上一手的旧值, 若碰巧读出 14 张还会去点手牌. 认出它就点
# 「确认」(别人都确认后游戏本来也会自动走, 点了只是更快), 这一轮不再读手牌.
# 流局画面没有这个按钮(约 5s 自动过), 那种就靠 DealWatch 的清空边沿兜底.
HAND_RESULT_SCREEN = "牌局页"
HAND_RESULT_CONFIRM = "局确认"

# 只有"这一手真的在进行中"才允许的动作. 闸门没开时(发牌中 / 上一手已结束)
# 这几类一律不执行, 因为它们的共同点是**必须先有一手真手牌**:
# 出牌要挑一张真牌, 立直要满足听牌形, 吃碰杠和拔北都要有对应的真手牌结构.
# 反过来说, 摊牌 / 结算画面上牌面模型读出来的"手牌"张数可能完全合法
# (实跑 t190 读 14 张、missing_slots=0), 只靠张数是拦不住的.
#
# 故意**不包含** win / confirm / skip:
#   - confirm 就是结算页上那个按钮(实跑 t142 / t155 靠它翻页, 且它的置信度
#     只有 0.151~0.156, 就压在 UI_CONF 上, 拦掉它等于把翻页交给 StuckDetector);
#   - win 只在真轮到自己且手里 14 张时才出现;
#   - skip(跳过吃碰杠)点错了也只是多等一轮.
DEAL_BLOCKED_ACTIONS = (ACTION_DISCARD, ACTION_RIICHI, ACTION_MELD,
                        ACTION_BABEI)

# ---- 场次: 段位场(主任务, 循环连打) / 友人场(测试用) ----
# 段位场流程与友人场形状不同(无准备页/无开始按钮, 点"四人东"自动匹配开局),
# 见 rooms.RoomSpec.ranked 的说明. 2026-10-09 用户拍板: 段位场才是日常任务.
MODE_RANKED = "段位场"
MODE_FRIENDS = "友人场"
MODE_OPTIONS = (MODE_RANKED, MODE_FRIENDS)

# 网络读牌不可用时, 把"到底为什么"说清楚.
# 2026-10-16 实机教训: 只打 `status` 的话, "没开抓包"这局打出来的是
# `no-round`(因为目录里剩着上一局的旧 spool, 桥重放出一份已结束的牌局),
# 读日志的人会以为"还没发牌", 完全查不到真正的原因.
_NET_UNAVAILABLE_HINT = {
    "no-spool": "没找到本局的 spool —— 抓包没开?(TUN + mitmdump, 见 docs/NET_CAPTURE.md)",
    "no-round": "spool 里没有进行中的一手 —— 多半是上一局留下的旧文件",
    "no-seat": "协议里还没有我的座位(未发牌/未进对局)",
    "protocol-suspect": "协议出现未建模动作, 已回退视觉",
    "bridge-error": "桥内部异常, 已回退视觉",
    "no-state": "桥拿不到牌局状态, 已回退视觉",
}

# `no-round` 的第二种含义: 这一场正在打, 只是刚好在两手之间(和了演出 /
# 发牌动画). 2026-10-16 实测: 每手之间有十几秒就是这个状态, 日志一路在说
# "多半是上一局留下的旧文件", 把人往错方向带. 判据见 _net_unavailable_hint.
_NET_ROUND_GAP_HINT = "这一场在打, 只是当前没有进行中的一手（两手之间的过场）"


# 协议胶水层搬到了 src/tasks/glue.py(协议读数 -> 决策输入的翻译, 可独立测).
# 这里显式再导出: 单测与工具一直按 `PlayOneGameTask._proto_turn_of` 这种
# 模块属性访问, 搬走后保留同一组名字, 不必去改所有调用点.
from src.tasks.glue import (  # noqa: F401
    _draw_settle_pending,
    _ev_record_of,
    _plan_ev,
    _prefer_non_aka_box_impl,
    _proto_clock_read,
    _proto_clock_turn_of,
    _proto_meld_window_of,
    _proto_round_open_of,
    _proto_turn_of,
    _trust_skips_confirm,
    _urgent_now,
)

# 在开房任务的默认配置之上, 加对局内需要的参数
_DEFAULT_CONFIG = {
    **_BASE_CONFIG,
    # 场次下拉框: 段位场 = 银之间四人东自动匹配(主任务); 友人场 = 按下面的
    # 人数/时长建房加电脑(测试用). 选了段位场时 players/length 两个下拉框
    # 不参与建房; 牌桌页标注与自动开关行位一律按 RoomSpec.player_count 判
    # (见 _players_key), 与下拉框无关.
    "mode": MODE_RANKED,
    # 房型沿用开房任务的两个下拉框(players + length), 不在这里重复设默认值.
    # 原因: 打牌任务内部会调 CreateFriendsRoomTask 去建房, 两边必须选同一个
    # 房型, 而下拉框的默认值来自 _BASE_CONFIG -- 只改一边的默认值会导致
    # "界面选四人但打牌任务按三人建房"这种极难查的错位.
    #
    # **这里只放用户真正常用的**(2026-10-07 用户决定), 其余二十多项
    # 内部调参全搬去了 src/tasks/tuning.py -- 框架会把 default_config 里
    # 每一项都渲染成控件, 塞满阈值/延迟只会让卡片没法看.
    "max_games": 1,                 # 连打几局
}


class PlayOneGameTask(CreateFriendsRoomTask):
    """建房 -> 点开始 -> 打完一局(可连打)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "自动打牌（段位场/友人场）"
        self.description = ("段位场(默认): 银之间四人东自动匹配、循环连打; "
                            "友人场(测试): 建房加电脑按牌效打")
        # config_type(房型两个下拉框)从 CreateFriendsRoomTask 继承下来,
        # 不重复声明 -- 父子两个任务卡片上的下拉框必须一致, 否则会出现
        # "建房选四人, 打牌按三人" 的错位. 见 _DEFAULT_CONFIG 的说明.
        self._vision: GameVision | None = None
        self._record_path: Path | None = None
        self._net: NetAssist | None = None        # 网络读牌对账门面
        self._net_log_at: float = 0.0             # 对账日志限流时间戳
        self._capture = None              # 抓包链路(mitmdump + TUN)生命周期
        self._capture_spool: Path | None = None   # 本场 spool 路径(显式交给桥)
        self._need_confirm_click = False   # 出牌是否需要"选中再确认"
        self._rng = random.Random()
        self._cfg: dict = {}
        # 本类覆写了 _default_config(), 所以要**再调一次**把结果写进框架读的
        # self.default_config. 不调的话界面拿到的是基类在 __init__ 之前算的
        # 那份(不含本类的对局内参数), 甚至可能是空 dict 导致整个卡片不展开.
        # 见 CreateFriendsRoomTask._init_default_config 的说明.
        self._init_default_config()
        # 场次下拉框: 在继承的 players/length/room 之上加一个 mode.
        # 默认值必须与 _DEFAULT_CONFIG["mode"] 及 MODE_OPTIONS 逐字一致
        # (否则下拉框显示空白, 见 CreateFriendsRoomTask.__init__ 的说明).
        self.config_type = {
            **self.config_type,
            "mode": {"type": "drop_down", "options": list(MODE_OPTIONS)},
            # 连打局数上下限: 不声明时框架默认上限 99999999, 一旦进去一个
            # 千万级误值, 箭头点不到头、框里的数也几乎没法选中改写
            # (2026-10-11 实测踩坑, 配置里出现过 11212218).
            "max_games": {"min": 0, "max": 99},
        }

    def _default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    def _resolve_spec(self, cfg: dict) -> RoomSpec:
        """按配置算出要开什么房间(覆写: 加了段位场 mode).

        显式 room 字段仍然最优先(命令行批量跑用); mode="段位场"(默认)
        固定 RANKED_SILVER_4P, players/length 两个下拉框不参与;
        mode="友人场" 走基类的组合推导.
        """
        if cfg.get("room"):
            return spec_by_key(cfg["room"])
        if cfg.get("mode") == MODE_RANKED:
            return spec_by_key("ranked_silver_4p")
        # 不写 super(): 测试把未绑定方法绑到 SimpleNamespace 上, super()
        # 在那种 self 上走不了 MRO; 显式按类调用对两种 self 都成立.
        return CreateFriendsRoomTask._resolve_spec(self, cfg)

    def _players_key(self, cfg: dict) -> str:
        """牌桌页标注 / 自动开关行位按哪套"人数"取.

        段位场模式下 players 下拉框不参与建房, 人数以 RoomSpec 为准
        (ranked_silver_4p 是四人); 友人场沿用下拉框. _spec 用 getattr
        兜底: 测试注入的 stub 不一定带这个属性.
        """
        spec = getattr(self, "_spec", None)
        if spec is not None and getattr(spec, "player_count", None) == 4:
            return "四人"
        return cfg.get("players") or ""

    # ---------------------------------------------------------------
    # 入口: 房间就绪之后开始打
    # ---------------------------------------------------------------

    def _after_room_ready(self, cfg: dict) -> None:
        self._cfg = cfg
        # 钳制连打局数: 控件范围(见 config_type)挡住了界面乱值, 但命令行
        # --once max_games 仍可写任意 int; range(1, N+1) 遇到千万级 N 就是
        # 事实上的死循环, 这里兜底一道.
        cfg["max_games"] = max(0, min(int(cfg["max_games"]), 99))
        if cfg["max_games"] <= 0:
            self.log_info("max_games=0，停在准备页不开始")
            return

        spec = self._resolve_spec(cfg)
        missing = self._missing_annotations(spec)
        if missing:
            raise RuntimeError(f"房间 '{spec.label}' 缺标注: {missing}")

        # 先修剪上一批留下的每轮诊断帧, 再开始记这一批(见 _prune_debug_frames)。
        self._prune_debug_frames()
        self._open_record()
        self._ensure_vision()          # 副作用: 模型没加载就先加载

        # 抓包链路(mitmdump + mihomo TUN)已在 `_before_room_flow` 起好 ——
        # 段位场点完"四人东"就自动匹配, 分钟级的匹配期也要读协议(2026-10-17
        # 用户拍板"提前到任务开始"). 本方法只管打; 停止由 `_after_room_flow`
        # 的 finally 兜底(停止指令/异常退出都会关掉自己起的进程).
        self._run_games(cfg, spec)

    def _before_room_flow(self, cfg: dict) -> None:
        """任务一开始(点四人东之前)就起抓包. 覆写基类钩子."""
        self._cfg = cfg
        self._start_capture()

    def _after_room_flow(self) -> None:
        """房间流程收尾(覆写基类钩子): 关掉自己起的抓包进程."""
        self._stop_capture()

    def _run_games(self, cfg: dict, spec) -> None:
        """连打循环(段位场 / 友人场两路). 抓包在上层已经起好."""
        # 网络读牌门面: 本场只建一次, 贯穿整场(段位场连打多局共用同一个
        # spool 文件与同一个 GameState). 关闭档不构造(免得白扫目录).
        # 桥不可用时一切对账静默回退视觉, 绝不拖垮主循环.
        self._net = self._build_net(cfg)
        self._net_log_at = 0.0

        if spec.ranked:
            self._ranked_games(cfg, spec)
            return

        for game in range(1, cfg["max_games"] + 1):
            # 连打好几局时这里是长循环, 必须先问停止(见 _should_yield).
            if self._should_yield():
                self.log_info("收到停止指令, 不再开下一局")
                return
            self.log_info(f"===== 第 {game}/{cfg['max_games']} 局 =====")
            # 上一局的结算页可能没点干净(终局页出现有延迟), 先清掉再等
            # "开始" —— 等待期间也会顺手点掉慢弹出来的页面, 见 _press_start.
            self._press_start(spec, cfg, game)
            # 点完「开始」这一刻服务端还没发 authGame —— 上一场的
            # game_over/match_started 必须先清掉, 否则新一局一进去就误判
            # "协议说本局结束"(见 _reset_net_session).
            self._reset_net_session()
            self._wait_leave_ready_page(spec, cfg)

            finished = self._play_one_game(cfg, game)
            self._record({"event": "game_end", "game": game,
                          "finished": finished})
            if not finished:
                # 用户喊停导致的收工不是失败(见 _should_yield 的说明),
                # 别记成 error 让人误以为出了故障.
                if self.stopped:
                    self.log_info("收到停止指令, 结束连打")
                    return
                self.log_error(f"第 {game} 局没能正常结束，停止连打")
                break
            if self._capture_broken:
                self.log_warning(
                    f"[抓包] 链路已废，第 {game} 局打完即收工"
                    f"（不再打剩余 {cfg['max_games'] - game} 局）")
                break

            if game < cfg["max_games"]:
                lo, hi = tuning.BETWEEN_GAMES_DELAY_RANGE
                delay = self._rng.uniform(lo, hi)
                self.log_info(f"局间休息 {delay:.1f}s")
                self.sleep(delay)

        self.log_info(f"对局记录: {self._record_path}")

    # ---------------------------------------------------------------
    # 段位场循环连打(2026-10-09, 用户拍板的主任务)
    # ---------------------------------------------------------------

    def _ranked_games(self, cfg: dict, spec) -> None:
        """段位场循环: 匹配 -> 打完 -> 回大厅 -> 重新排队 -> 下一局.

        与友人场连打的两点不同: ①开下一局不靠准备页的"开始"按钮, 而是
        回大厅把 段位场->银之间->四人东 整条流程重新点一遍(段位场没有
        准备页, 见 rooms.RoomSpec.ranked); ②匹配是分钟级的, 开局前要等
        匹配成功(见 _wait_matchmaking).
        """
        for game in range(1, cfg["max_games"] + 1):
            # 连打好几局时这里是长循环, 必须先问停止(见 _should_yield).
            if self._should_yield():
                self.log_info("收到停止指令, 不再开下一局")
                return
            self.log_info(f"===== 第 {game}/{cfg['max_games']} 局（段位场） =====")
            self._wait_matchmaking(cfg, spec, game)

            finished = self._play_one_game(cfg, game)
            self._record({"event": "game_end", "game": game,
                          "finished": finished})
            if not finished:
                if self.stopped:
                    self.log_info("收到停止指令, 结束连打")
                    return
                self.log_error(f"第 {game} 局没能正常结束，停止连打")
                break
            if self._capture_broken:
                self.log_warning(
                    f"[抓包] 链路已废，第 {game} 局打完即收工"
                    f"（不再打剩余 {cfg['max_games'] - game} 局）")
                break

            if game < cfg["max_games"]:
                lo, hi = tuning.BETWEEN_GAMES_DELAY_RANGE
                delay = self._rng.uniform(lo, hi)
                self.log_info(f"局间休息 {delay:.1f}s，然后回大厅重新排队")
                self.sleep(delay)
                self._back_to_lobby(cfg, spec, game)
                self._run_flow(cfg)

        self.log_info(f"对局记录: {self._record_path}")

    def _wait_matchmaking(self, cfg: dict, spec, game: int) -> None:
        """点完"四人东"后等匹配成功.

        判据二选一: 「对局开始页」的胜负横幅, 或牌桌页操作条(加载快时
        横幅一闪而过直接进对局). 段位场匹配是**分钟级**的(取决于时段),
        所以这里用独立的 MATCHMAKING_TIMEOUT, 不能沿用秒级页面等待.
        """
        candidates = [(spec.screens.ready, spec.screens.ready_marker),
                      (self._table_screen(cfg), TABLE_FEATURE)]
        candidates = [(s, k) for s, k in candidates if self._has_region(s, k)]
        deadline = time.time() + tuning.MATCHMAKING_TIMEOUT
        next_status = time.time() + 30.0
        frame = None
        # 连打时"上一场的 match_started"还留在状态里 —— 不清掉的话下面
        # 协议优先会立刻假阳返回(误以为新一场已经开始). 只清标志, 不动桥.
        self._reset_net_session()
        while time.time() < deadline:
            if self._should_yield():
                raise TaskStopped("等待匹配期间收到停止指令")
            # 协议优先: 抓包在跑时 authGame(is_game_start) 一到就算匹配成功,
            # 不用等视觉横幅 —— 横幅在加载快的机器上一闪而过, 模板也可能
            # 因为版本更新而认不出来(那时视觉这条路等于永远 False).
            if self._protocol_says_matching():
                self.log_info("    匹配成功（协议: authGame is_game_start）")
                return
            frame = self._grab()
            for screen, keyword in candidates:
                if self._locate(screen, keyword, cfg, frame,
                                strict=True, verbose=False) is not None:
                    self.log_info(
                        f"    匹配成功（检测到 [{screen}] '{keyword}'）")
                    return
            now = time.time()
            if now >= next_status:
                waited = now - (deadline - tuning.MATCHMAKING_TIMEOUT)
                self.log_info(f"    还在匹配（已等 {waited:.0f}s，"
                              f"上限 {tuning.MATCHMAKING_TIMEOUT:.0f}s）")
                next_status = now + 30.0
            self.sleep(tuning.PAGE_POLL_INTERVAL)
        self._save_wait_timeout_frame(frame, spec.screens.ready,
                                      spec.screens.ready_marker)
        raise RuntimeError(
            f"匹配超时（{tuning.MATCHMAKING_TIMEOUT:.0f}s 没等到对局开始），已中止任务")

    def _back_to_lobby(self, cfg: dict, spec, game: int) -> None:
        """终局后等游戏回到大厅(下一轮排队的起点).

        期间继续处理慢拍弹出的结算页(见 _drain_result_pages 的说明) --
        段位场终局页可能带段位分变动弹窗, 不点掉会挡住大厅.
        """
        deadline = time.time() + tuning.LOBBY_RETURN_TIMEOUT
        while time.time() < deadline:
            if self._should_yield():
                raise TaskStopped("等待回大厅期间收到停止指令")
            frame = self._grab()
            if self._handle_result_page(frame, cfg, spec, game, 0):
                continue
            # 奖励页也可能慢一拍才弹出来(见 _handle_reward_page 的说明),
            # 挡在大厅前面 —— 不点掉 LOBBY_RETURN_TIMEOUT 就白等满了.
            if self._handle_reward_page(frame, cfg, game):
                continue
            if self._locate("大厅页", "段位场", cfg, frame,
                            strict=True, verbose=False) is not None:
                self.log_info("    已回到大厅")
                return
            self.sleep(tuning.PAGE_POLL_INTERVAL)
        self._save_wait_timeout_frame(self._grab(), "大厅页", "段位场")
        raise RuntimeError(
            f"终局后 {tuning.LOBBY_RETURN_TIMEOUT:.0f}s 没等到大厅页，已中止任务")

    def _missing_annotations(self, spec) -> list[str]:
        return missing_categories(required_categories(spec),
                                  set(self._annotations))

    def _prune_debug_frames(self) -> None:
        """修剪上一批留下的每轮诊断帧(见 src/retention.py).

        为什么放在任务启动时: 早年 `save_frames` 每轮一张 1080p PNG, 连打几天
        就是上百 GB(2026-10-17 实测 74323 张 / 169.5GB), 而"定期手动跑清理脚本"
        这件事在实践里等于没做。跟抓包一样, 该自动的都自动掉。

        只删 `frame_g<局>_t<轮>_*.png`; 低置信/闸门/待标注留证帧与 jsonl
        对局记录一律不碰。**永不抛**: 清理是锦上添花, 失败只记 warning,
        绝不能因为删文件失败就不让开局。
        """
        if not tuning.DEBUG_PRUNE_AUTO:
            return
        try:
            res = prune_turn_frames(
                DEBUG_ARTIFACTS_DIR,
                keep_days=tuning.DEBUG_FRAME_KEEP_DAYS,
                max_files=tuning.DEBUG_FRAME_MAX_FILES,
                max_bytes=tuning.DEBUG_FRAME_MAX_BYTES,
                apply=True,
            )
        except Exception as exc:  # noqa: BLE001 - 清理失败不拖垮开局
            self.log_warning(
                f"调试帧修剪失败（{type(exc).__name__}），跳过本次清理")
            return
        if res.changed:
            self.log_info(
                f"已清理调试帧 {res.removed}/{res.scanned} 张"
                f"（释放 {res.freed_mb:.0f}MB，保留 {res.kept} 张）")
        elif res.errors:
            self.log_warning(f"调试帧修剪有 {res.errors} 项失败，跳过")

    # ---------------------------------------------------------------
    # 一局的循环
    # ---------------------------------------------------------------

    def _at_ready_page(self, spec, cfg: dict, frame) -> bool:
        bench.count_state("ready_page")  # bench 挂钩(默认关闭): 状态识别计数
        """当前画面是不是准备页.

        **判据是静态特征而不是"开始"按钮** -- 理由见 rooms.RoomScreens
        的 ready_marker: 开始按钮跨帧稳定分数只有 0.836, 低于探测阈值 0.88,
        拿它判等于永远返回 False. 这里的上一版就是那么写的, 后果是每局
        结束都认不出来, 一直空转到 game_timeout.
        """
        marker = ready_marker(spec)
        return self._locate(
            marker[0], marker[1], cfg, frame, strict=True, verbose=False
        ) is not None

    # ---------------------------------------------------------------
    # 抓包链路(mitmdump + mihomo TUN)的起停
    # ---------------------------------------------------------------

    def _start_capture(self) -> None:
        """按 tuning.CAPTURE_AUTO 起抓包链路. 已在跑的复用, 失败静默降级.

        绝不抛: 抓包只是"让协议能读到"的前提, 读不到照样能纯视觉打完
        (网络读牌永远是增强不是依赖, 见 AGENTS.md 铁律 3).
        """
        self._capture = None
        self._capture_spool = None
        if not tuning.CAPTURE_AUTO:
            return
        # 关闭档连抓包都不起: 用户明确要脱离网络(一键回退纯视觉).
        if (self._cfg.get("net_read") or tuning.NET_READ) == tuning.NET_READ_OFF:
            return
        try:
            from src.net.capture import CaptureSupervisor

            supervisor = CaptureSupervisor(
                port=tuning.CAPTURE_MITM_PORT,
                mitm_bin=tuning.CAPTURE_MITMDUMP,
                mihomo_bin=tuning.CAPTURE_MIHOMO,
                start_timeout=tuning.CAPTURE_START_TIMEOUT,
                tun_device=tuning.CAPTURE_TUN_DEVICE,
                upstream=tuning.CAPTURE_UPSTREAM,
                install_ca=tuning.CAPTURE_INSTALL_CA,
            )
            status = supervisor.start()
        except Exception as exc:  # noqa: BLE001 - 抓包永不拖垮任务
            self.log_error(f"抓包启动异常（{type(exc).__name__}: {exc}），"
                           f"改用纯视觉（见 docs/NET_CAPTURE.md）")
            return
        self._capture = supervisor
        if status.spool is not None:
            self._capture_spool = status.spool
        self.log_info(f"[抓包] {status.summary()}")
        for note in status.notes:
            self.log_info(f"[抓包]   {note}")
        for err in status.errors:
            self.log_warning(f"[抓包]   {err}")
        if not status.ok:
            self.log_warning(
                "[抓包] 抓包链路没就位 —— 网络读牌会自动回退视觉；"
                "排查见 docs/NET_CAPTURE.md §六")

    def _stop_capture(self) -> None:
        """关掉**我们自己起的**抓包进程. 复用来的一个都不碰. 绝不抛."""
        supervisor = self._capture
        if supervisor is None:
            return
        self._capture = None
        try:
            for line in supervisor.stop():
                self.log_info(f"[抓包] {line}")
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------
    # 协议判牌局起止(协议优先, 视觉兜底)
    # ---------------------------------------------------------------

    def _protocol_says_matching(self) -> bool:
        """协议说"匹配成功、已进对局"(ResAuthGame.is_game_start).

        关掉 ROUND_BY_PROTOCOL / 没建门面 / 桥不可用一律返回 False, 调用方
        接着走视觉判定 —— 这就是"兜底"那两个字的落点.
        """
        if not tuning.ROUND_BY_PROTOCOL or self._net is None:
            return False
        try:
            return self._net.match_started()
        except Exception:  # noqa: BLE001
            return False

    def _protocol_says_game_over(self) -> tuple[bool, str]:
        """协议说"这一场打完了"(NotifyGameEndResult / NotifyGameTerminate).

        返回 (是否结束, 原因). 原因非空 = 被中断(reason 原样带回日志).
        """
        if not tuning.ROUND_BY_PROTOCOL or self._net is None:
            return False, ""
        try:
            session = self._net.session()
        except Exception:  # noqa: BLE001
            return False, ""
        if not session["game_over"]:
            return False, ""
        return True, session["reason"]

    def _reset_net_session(self) -> None:
        """开新一局之前, 把"上一场"的起止标志清掉(友人场路径).

        **为什么必须有**(2026-10-16 实机踩到): 同一个 GameState 贯穿整轮连打,
        `match_started` / `game_over` 是"这一场"的粘性标志。段位场在
        `_wait_matchmaking` 里清过, 友人场走的是 `_press_start` +
        `_wait_leave_ready_page`(根本不经过 `_wait_matchmaking`) —— 于是
        点完「开始」到服务端 authGame 到达之间那一两秒, 上一场的
        `game_over=True` 还挂在状态里, `_play_one_game` 一进去就报
        "协议报告本局结束", 并顺手把 `_game_over_since` 也点着了
        (后果见主循环 1.6 分支)。实测那一次侥幸没出事 —— 上一场的结算页
        刚好已经被点掉, `_handle_result_page` 返回了 False; 但只要结算页还在
        屏幕上, 新一局会在第 0 轮被判"打完"直接跳过, 白烧一局。
        """
        if self._net is None or not tuning.ROUND_BY_PROTOCOL:
            return
        st = getattr(self._net, "state", None)
        # 已经在"这一手进行中"就别清: `round_open` 是事件流推导出来的,
        # 清掉以后要等下一手 ActionNewRound 才会重新点亮 —— 而那一手的事件
        # 早就被消费掉了, 会让整手都读不到协议(比误报严重得多)。
        if st is not None and getattr(st, "round_open", False):
            return
        try:
            self._net.reset_session()
        except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
            pass

    def _build_net(self, cfg: dict) -> NetAssist | None:
        """按档位构造网络读牌门面(关闭档返回 None, 整条路径不参战).

        档位住在 tuning.NET_READ(与 MELD_POLICY / BABEI_POLICY 同类的一次性
        策略开关, 不是每局都要改的用户控件, 所以不进界面); cfg 里显式写了
        "net_read" 时以它为准, 留一个不改源码就能覆盖的口子.

        spool 优先用抓包模块给的那个路径: 自动抓包起的 mitmdump 写进
        `_capture_spool`, 显式指过去比"去目录里猜哪份最新"准得多(见
        tuning.NET_MAX_SPOOL_AGE 里那个"把上一局旧文件当本局"的踩坑).
        抓包没自动起(CAPTURE_AUTO 关掉 / 手动抓包)时退回按 mtime 发现.
        """
        mode = cfg.get("net_read") or tuning.NET_READ
        if mode == tuning.NET_READ_OFF:
            return None
        # getattr 兜底: 单测把未绑定方法绑到 stub self 上, stub 不带这个字段.
        return NetAssist(path=getattr(self, "_capture_spool", None),
                         debug_dir=DEBUG_ARTIFACTS_DIR,
                         max_spool_age=tuning.NET_MAX_SPOOL_AGE)

    def _net_unavailable_hint(self, status: str) -> str:
        """"不可用"状态配一句能照着查的中文提示.

        `no-round` 有歧义, 必须分开说(2026-10-16 实机踩到): 这一场正在打、
        只是刚好在两手之间(和了演出 / 发牌动画)时状态也是 no-round ——
        那十几秒里日志一直在说"多半是上一局留下的旧文件", 把人往错方向带.
        判据用 session(): 这一场已经开始过(match_started)、又还没结束 →
        就是局间空档. 其余情况沿用 _NET_UNAVAILABLE_HINT.
        """
        if status == "no-round" and self._net is not None:
            try:
                session = self._net.session()
            except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
                # 桥都坏了就别猜原因(猜错比不说更坏): 退成"只有状态码".
                return ""
            if (session and session.get("match_started")
                    and not session.get("game_over")):
                return _NET_ROUND_GAP_HINT
        return _NET_UNAVAILABLE_HINT.get(status, "")

    def _net_suspect_detail(self, status: str) -> str:
        """把"哪条报文让桥判了可疑"带进日志(见 NetAssist.status_detail).

        只在 protocol-suspect 时问: 其余状态的"明细"是无关信息, 加进去只会
        让那行日志变长、把真正该看的状态码挤掉. 绝不抛.
        """
        if status != "protocol-suspect" or self._net is None:
            return ""
        try:
            return self._net.status_detail()
        except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
            return ""

    def _net_step(self, hand, buttons):
        """一轮网络读牌对账.

        返回 (hand, buttons, exempt, note, trust):
        - 影子对账档: hand/buttons 原样返回, note 只进日志与出牌记录;
        - 接管读数档: hand 换协议纠错后的标签(坐标不动)、buttons 裁掉协议
          没认的(视觉多认的按钮 = 误鸣风险); exempt 是被协议纠正过的槽位,
          调用方要把它们从 blurry 里剔除(我们已经确知它们的真值了).
        - trust: "trusted"=协议手牌与视觉对上(计数一致无歧义), 本轮标签就是
          协议真值, 消费方可跳过低置信/双帧重算(见 tuning.NET_TRUST_PROTO);
          "mismatch"=协议有手牌但对不上(读不出/读不动), 调用方该存现场帧;
          "ambiguous"=**多重集已由协议修正, 但槽位归属定不下来**(2 槽以上待补).
          消费方可以拿这手牌算牌效, 但**不能照它的下标点牌** —— 标错的槽上
          点下去是另一张牌;
          "conflict"=**张数就和协议对不上**(两边都有值). 这一手牌连张数都是错的,
          一张都不能用来算牌效/点牌 —— 以协议为准, 本轮不动等下一帧
          (见 glue._hand_conflict_veto, 没有超时退视觉);
          "off"=关闭/不可用/异常, 协议没说话.
        - 关闭 / 不可用 / 抛异常: 一律原样返回, 走纯视觉.
        """
        mode = self._cfg.get("net_read") or tuning.NET_READ
        if self._net is None or mode == tuning.NET_READ_OFF:
            return hand, buttons, frozenset(), "", "off"
        assist = (mode == tuning.NET_READ_ASSIST)
        try:
            step = self._net.step(hand, buttons, assist=assist)
        except Exception as exc:  # noqa: BLE001 - 网络层永不拖垮主循环
            self._log_repeat_once(
                "net_err", type(exc).__name__,
                f"  网络读牌对账异常（{type(exc).__name__}），本轮回退视觉",
                level="warning")
            return hand, buttons, frozenset(), "", "off"
        if not step.available:
            hint = self._net_unavailable_hint(step.status)
            detail = self._net_suspect_detail(step.status)
            why = f"（{step.status}）" if not hint else f"（{step.status}: {hint}）"
            if detail:
                why += f" 明细={detail}"
            self._log_repeat_once("net_na", f"{step.status}|{detail}",
                                  f"  网络读牌不可用{why}，走视觉")
            return hand, buttons, frozenset(), "", "off"
        now = time.time()
        if now - self._net_log_at >= tuning.NET_LOG_INTERVAL:
            self._net_log_at = now
            self.log_info(f"  [网络读牌/{mode}] {step.note}")
        if not assist:
            return hand, buttons, frozenset(), step.note, "off"
        if step.hand is not None:
            hand = step.hand
        if step.buttons is not None:
            buttons = step.buttons
        if getattr(step, "conflict", False):
            trust = "conflict"
        elif getattr(step, "ambiguous", False):
            trust = "ambiguous"
        else:
            trust = "trusted" if getattr(step, "trusted", False) else "mismatch"
        return hand, buttons, step.blurry_exempt, step.note, trust

    def _net_clock(self) -> dict:
        """协议给的"时机": 轮到我了吗 / 还剩几秒(拿不到就是保守值). 绝不抛.

        与 _net_step 分开的理由: 那个是**对账读数**(要求 available(): 这一手在
        打 + 喂料新鲜 + 协议不可疑), 这个是"当前这一动还剩多少时间" —— 恰恰要
        在决策那一刻取. 见 tuning.NET_TURN_CLOCK.
        """
        if self._net is None or not tuning.NET_TURN_CLOCK:
            return {"my_turn": False, "left": None}
        mode = self._cfg.get("net_read") or tuning.NET_READ
        if mode == tuning.NET_READ_OFF:
            return {"my_turn": False, "left": None}
        try:
            return self._net.turn_clock()
        except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
            return {"my_turn": False, "left": None}

    @staticmethod
    def _is_urgent(clock: dict) -> bool:
        """协议倒计时快到了没有(见 tuning.NET_URGENT_SECONDS).

        **只认"还没超时的正数"**: left 为负说明桥滞后或计时口径不对, 拿它压缩
        节拍等于跟一个不存在的倒计时赛跑, 而且会一直压(机器人变得机械).
        """
        if not tuning.NET_TURN_CLOCK:
            return False
        left = clock.get("left")
        if left is None:
            return False
        try:
            return 0.0 <= float(left) <= tuning.NET_URGENT_SECONDS
        except (TypeError, ValueError):
            return False

    def _proto_turn(self) -> bool | None:
        """协议对"现在该不该我动"的权威说法. None = 不接管/不可信 -> 不否决.

        三个前提缺一不可, 都是"宁可不用":
        ① 必须已经切到**接管档** —— 这道闸是"整轮否决", 否决掉一个出牌就会
           把整手牌拖到被托管(见下一段), 所以只在接管档参战;
        ② `NET_TURN_VETO` 必须**单独 arm**. 与 NET_READ 拆开见 tuning.py:
           同一档位里"读数接管"不改变动作种类(纠标签 / 裁误检按钮, 两个方向
           都是减害), 而"整轮否决"误一次就报废一手牌 —— 且它在影子档是零样本
           (从没真正执行过). 不让一条零验证的机制搭 NET_READ 的便车;
        ③ 桥必须真的可用(`available()`: 这一手在打 + 喂料新鲜 + 协议不可疑).
           桥滞后时那份操作列表可能早已过期, 拿"过期窗口里没有打牌选项"去否决
           出牌, 会把整手牌拖到被托管 —— 那比误打一张牌严重得多.
        """
        mode = self._cfg.get("net_read") or tuning.NET_READ
        if mode != tuning.NET_READ_ASSIST or not tuning.NET_TURN_VETO:
            return None
        return _proto_clock_read(self._net, mode)

    def _proto_clock_turn(self) -> bool | None:
        """协议对"这一刻有没有我的出牌操作"的说法. None = 不可用/不可信.

        **不看档位** —— 与 `_proto_turn` 的区别就在这里. 它只喂给
        `decide_turn` 的 `proto_turn`(单独立直按钮的过滤器), 否决不了任何别的
        动作, 也就不可能把整手牌拖进托管, 所以影子档也该参战:
        2026-10-17 东三局, 那一刻协议给的可用操作是 **碰**(type 3)、手里 13 张,
        而视觉在同一帧里读出一个 lizhi(conf 0.714, 从"上家立直"的宣告画面里
        误读出来) —— 整轮否决当时是 None(影子档不参战), 于是没人拦它, 一个
        单帧误检报废了一整手牌.
        """
        return _proto_clock_read(
            self._net, self._cfg.get("net_read") or tuning.NET_READ)

    def _proto_meld_window(self) -> bool | None:
        """协议说"此刻开着**给我的**鸣牌窗口"(吃/碰/大明杠).

        它是 `_proto_turn` 那道整轮否决的**唯一缺口**: 鸣牌窗口里协议只给
        鸣牌操作、不给打牌操作, 于是 `protocol_says_my_turn()` 必然为假 ——
        那不是"不轮到你", 那是"这一口轮到你要不要鸣". 夹具
        `liqi_sanma_2rounds` 的两次 `ActionDiscardTile` 给的 operation_list
        都是 `[3]`, 实测坐实.

        与 `_proto_turn` **同一条前提**(同一个开关、同一份新鲜度): 否决没 arm
        时 my_turn 恒 None, 这个缺口也就没有存在的必要, 直接 None 不参战.
        取 `meld_options()` 而不是自己翻 offered: 协议 → 鸣牌语义的翻译点
        只有那一个, 两处各写一份迟早漂移.
        """
        if not tuning.NET_TURN_VETO:
            return None
        if (self._cfg.get("net_read") or tuning.NET_READ) == tuning.NET_READ_OFF:
            return None
        if self._net is None:
            return None
        try:
            if not self._net.available():
                return None
            return bool(self._net.meld_options())
        except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
            return None

    def _proto_round_open(self) -> bool | None:
        """协议说"这一手还在打吗"(收局守卫的输入, 见 decide_turn.round_open).

        **不看档位、不要求 available()** —— 它否决的只是"收局画面上的残留
        出牌", 那种读数在任何档位都是幻觉; False 之外全都不参战(None).
        """
        if self._net is None:
            return None
        try:
            return self._net.round_open()
        except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
            return None

    def _proto_round_seq(self) -> int | None:
        """协议开过几手了(单调递增). None = 协议不参战.

        手间隙守卫用它区分"协议还停在这一手"和"协议已经翻到下一手" ——
        `round_open` 说不了这件事(两种情况它都是 True), `round_id` 在连打
        两场时又会重复(每场都从东一局 (0,0,0) 开), 只有计数器分得开.
        """
        net = getattr(self, "_net", None)
        if net is None:
            return None
        try:
            return net.round_seq()
        except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
            return None

    def _net_fresh(self) -> bool:
        """协议链路还在吗. **只问进程死活, 不问"多久没事件"**.

        抓包是我们自己 spawn 的子进程, 活了就一定有数据可读(有没有事件是牌局
        自己的事: 别家长考、我方吃碰杠/立直的选择窗口, 服务端本来就不发事件);
        真死了由 `CaptureSupervisor.alive()` 报, 主循环会把 `self._net` 置空并
        收工. 所以这里要回答的只是"还有没有协议可等", 而不是"喂料新不新鲜" ——
        用超时反推新鲜度是那条 10s 误判的根源(见 tuning.NET_STALE_AFTER 的注释,
        那不是这个问题该用的判据).

        没起抓包(用户手动抓/USER 关掉 CAPTURE_AUTO)也返回 True: 那是用户的选择,
        不能当成"坏了".
        """
        if getattr(self, "_net", None) is None:
            return False
        return not getattr(self, "_capture_broken", False)

    def _hand_gap_arm(self) -> None:
        """手间隙入闩: 视觉刚报"上一手结束"(DealWatch 的 rearmed 分支).

        视觉的"手牌区空了"比协议的收局信号(Hule/NoTile)早到, 中间那一两秒里
        摊牌/结算画面会被读成一手 14 张的散牌, 张数判据(14 % 3 == 2)照旧说
        "轮到我", 协议那头 `offered` 也还是上一动的残留(同样说"轮到我") ——
        两条判据一起假阳性, 于是对着幻影牌出牌并截一堆低置信存证.

        **只在"协议还停在这一手"时才入闩**(2026-10-18 13:50 局实锤的死锁):
        那一局我们刚点完「确认」进下一手, 视觉在结算→发牌的过场里看到手牌区
        清空, `rearmed` 照样触发; 而入闩时协议**已经**翻到下一手了, 于是
        "先见 round_open=False、再见 True"这个出口永远等不到 —— 整整 30 秒
        不出牌, 雀魂直接把我们的 5z 托管打了出去.
        判据用 `round_seq`: 它比"上一手开牌时是第几手"大, 就说明协议已经翻页,
        这次 `rearmed` 是过场不是收局, 不入闩.
        """
        if not tuning.HAND_GAP_GUARD:
            return
        ro = _proto_round_open_of(self)
        if ro is None:
            return          # 协议不在这场对局里 -> 不参战, 维持纯视觉老行为
        seq = self._proto_round_seq()
        opened_at = getattr(self, "_round_seq_at_open", None)
        if opened_at is None:
            return          # 这一局还没开过手: 没有"收局残留"要防
        if seq is not None and seq > opened_at:
            return          # 协议已经翻到下一手: 过场, 不是收局
        if not self._net_fresh():
            return          # 抓包断了: 等不到出口信号, 入了就是锁死
        self._hand_gap = True
        self._gap_since = time.time()
        # 收局信号已经先到了就别再等它一遍.
        self._gap_saw_closed = (ro is False)

    def _hand_gap_poll(self) -> bool:
        """还在"上一手已结束、下一手还没开"的间隙里吗. True = 这一轮跳过.

        出口**只看协议**: 先见到 round_open=False(收局信号到了), 再见到
        round_open=True(下一手 ActionNewRound 到了). 反过来就是"视觉已经翻篇、
        协议还没翻"的那段窗口, 那期间一律不出牌. 见 2026-10-18 东一局/东三局
        实锤: 收局画面上读到 14 张散牌, 对着它算了 EV 并真点了 2z / 2m / 2s.
        """
        # 用 getattr: 单测把主循环绑到 SimpleNamespace 假对象上跑, 那些对象
        # 没有这个字段(与 _shallow 同一套路).
        if not getattr(self, "_hand_gap", False):
            return False
        ro = _proto_round_open_of(self)
        if ro is False:
            self._gap_saw_closed = True
        elif ro is True and self._gap_saw_closed:
            self._hand_gap_clear()
            self.log_info("  协议报新的一手开了，手间隙结束，恢复决策")
            return False
        if not self._net_fresh():
            # 抓包断了: 出口信号(收局 False -> 开局 True)永远不会来, 硬等
            # HAND_GAP_TIMEOUT 就是把整手牌让给托管. 立刻退回视觉.
            self.log_warning(
                "  手间隙里抓包断了（桥不新鲜），等不到协议的收局/开局信号，"
                "立刻退回纯视觉判断")
            self._hand_gap_clear()
            return False
        if time.time() - self._gap_since >= tuning.HAND_GAP_TIMEOUT:
            # 兜底(铁律: 宁可漏判不可锁死): 协议迟迟不给信号(桥断了/对面卡在
            # 结算页/整场打完)就退回纯视觉老行为, 绝不把出牌永久拦死.
            self.log_warning(
                f"  手间隙等了 {tuning.HAND_GAP_TIMEOUT:.0f}s 没等到协议的收局"
                f"/开局信号，退回纯视觉判断")
            self._hand_gap_clear()
            return False
        return True

    def _hand_gap_clear(self) -> None:
        self._hand_gap = False
        self._gap_since = 0.0
        self._gap_saw_closed = False

    def _capture_watch(self) -> bool:
        """我们起的抓包进程还活着吗. 返回 True = 没死(或本来就没起).

        抓包链路是**本任务 spawn 的**, 进程退了本来就该立刻知道, 不用等桥
        的 stale_after 超时从"没新事件"反推 —— 那 10 秒里协议读数冻结在最后
        一帧, 会被当成真值用. 见 capture.CaptureSupervisor.alive.
        """
        if not tuning.CAPTURE_WATCH:
            return True
        cap = getattr(self, "_capture", None)
        if cap is None:
            return True
        try:
            return cap.alive() is not False
        except Exception:  # noqa: BLE001 - 监控本身永不拖垮主循环
            return True

    def _capture_down(self) -> None:
        """抓包链路废了的处理: 切视觉 + 打完本局收工(见 _run_games).

        `self._net = None` 一行就够 —— 现有 36 处 `_net is None` 守卫全部
        自动生效, 纯视觉老路一字不改地恢复. 链路没起过(复用外部实例 /
        CAPTURE_AUTO 关着)时不进这里: 那是用户的选择, 照常打完所有场次.
        """
        self._capture_broken = True
        self._net = None
        self.log_error(
            "[抓包] mitmdump/mihomo 进程已退出，协议读数不再可信 ——"
            "立刻切纯视觉，打完本局就收工（不再打剩余场次）。"
            "排查见 docs/NET_CAPTURE.md §六")

    def _proto_idle_skip(self) -> bool:
        """协议说"现在什么都没给我、也不轮到我" -> 这一轮不抓帧, 睡过去.

        见 tuning.NET_IDLE_SKIP. 任何拿不到都返回 False(多看一眼永远安全).

        **跳帧前必须 poll 一次桥**(2026-10-17 真机实锤的"轮到我还傻等 9 秒"):
        桥是 pull 式的, 只有 poll() 才读 spool 新事件; 跳帧路径原来只读不喂,
        协议状态冻在上一帧 —— 摸了牌 my_turn 都翻了 True, 这里看的还是几秒前
        的"什么都没给我", 于是一路睡到下一次睁眼(旧版是满 9 秒)才看见.
        poll 一次只是 tail 一下 spool 文件, 微秒级, 每个跳帧轮都喂得起.

        **没有"连跳 N 轮强制看一帧"了**(2026-10-18 删, 换成时间兜底): 那条
        兜底每 9 秒睁一次眼, 正好落在立直自动摸切与局间结算这两种画面上, 读
        出来的全是"13 张稳定读数"和"7 张残影". 见 tuning.NET_IDLE_LOOK_INTERVAL.

        **只管"协议可用、且明确说没我的事"这一种情形**: 局间(available() False)
        不在这里跳 —— 那几秒画面上是顺位面板, 要点头「确认」, 闭眼就点不了.
        """
        if not tuning.NET_IDLE_SKIP or self._net is None:
            return False
        net = self._net
        try:
            net.poll()
            if net.available():
                clock = net.turn_clock()
                skip = not (clock.get("offered") or clock.get("my_turn"))
                left = clock.get("left")
                if (skip and isinstance(left, (int, float))
                        and left <= tuning.NET_URGENT_SECONDS):
                    skip = False
            else:
                # 局间(本手已收、下一手未开) **照常看**: 那几秒里画面上是
                # "和了/流局"的顺位面板, 要点「确认」才会进下一手 —— 把眼睛
                # 闭上就等于不点确认, 只能等下一次强制睁眼(旧版最多 9 秒,
                # 2026-10-18 13:49 局就白等了那一段). 现在改由更下层挡:
                #   - `_phase_hand_result` 认面板、点确认;
                #   - `_phase_hand_gap` 在"协议已收局、这一手还没翻页"时不出牌;
                #   - 结算残影的低置信帧/标注素材改由 `_proto_round_open_of`
                #     说 False 时静音.
                skip = False
        except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
            return False
        if not skip:
            self._idle_skip_since = 0.0
            return False
        self._idle_skips += 1
        since = getattr(self, "_idle_skip_since", 0.0) or 0.0
        if not since:
            self._idle_skip_since = time.time()
            return True
        if time.time() - since >= tuning.NET_IDLE_LOOK_INTERVAL:
            # 病态兜底: 协议可能把"下一手开了"漏掉了, 睁眼看一帧.
            self._idle_skip_since = 0.0
            return False
        return True

    # 这里原来有一个 `_chi_option_probe`: 遇上"一嘴吃有 2~3 种组合"就点一下
    # 「吃」, 把雀魂的选择条拍下来验证划分, 然后**什么都不选**, 让服务端倒计时
    # 把这嘴作废. 2026-10-18 删掉 —— 它只在友人场开, 但代价是实打实的:
    #   - 面板一开, 服务端就不再发事件(它在等我们选), 那一嘴的 20~25 秒整段
    #     停摆, 还正好是旧 stale 兜底要动手的窗口(见 tuning.NET_READ 下面那段);
    #   - 用户当场看到的是"点了吃、弹出候选框、又不选", 一场里撞了两次
    #     (14:30:38 三组合 / 14:32:17 两组合, 各停 24s).
    # 现在策略里**根本没有吃**(MELD_POLICY="peng": 只碰役牌), 采样拿到的标签
    # 换不来任何决策, 所以整条链路(点按钮/拍面板/写 chi_options)全部去掉.
    # 划分算法本身还在 `src.mahjong.chi_panel`(纯函数 + 单测, 离线可用), 需要
    # 采样时用真帧跑它, 不要在真机上点.
    def _ev_plan(self):
        """(候选, 牌山) —— 出牌决策与出牌记录共用的那一次估值.

        只做取参, 逻辑在模块级 `_plan_ev`(单测会把它绑到假对象上跑).
        顺带把"刚摸到的那一张"存到 `self._drawn_tile`: 视觉是 34 类, 认不出
        赤 5, 而保留赤 5 必须知道刚摸的是不是赤(见 table.pick_tile_index).
        """
        rows, wall = _plan_ev(self._net, self._cfg, urgent=_urgent_now(self))
        self._drawn_tile = self._proto_drawn()
        return rows, wall

    def _proto_drawn(self) -> str | None:
        """协议说的"刚摸到的那一张"(带赤记号, 如 "0p"), 拿不到就是 None.

        协议手牌的**末位**就是刚摸到的那张 —— 见 `net.state` 里 `hand` 那行
        注释("暗手, 摸到的牌在末尾"), 项目本来就按这个约定在维护. 所以这里
        不需要协议手牌与视觉手牌同序: 我们只问一张牌, 不问整个排列.
        """
        net = self._net
        if net is None:
            return None
        try:
            hand = (net.ev_inputs() or {}).get("hand") or []
        except Exception:  # noqa: BLE001 - 读不到就不保留赤, 不影响出牌
            return None
        if len(hand) % 3 != 2:      # 14 张档才是"刚摸完牌"
            return None
        return hand[-1] if hand else None

    def _ev_shadow(self) -> dict | None:
        """影子档的出牌 EV 记录(`EV_EVAL_SHADOW`): 只写日志/jsonl, 不动手.

        计算本身在 `_plan_ev` —— 接管档用的是**同一份**结果(见
        tuning.EV_EVAL_TAKEOVER), 所以这里只管取数与打包.
        """
        if not tuning.EV_EVAL_SHADOW:
            return None
        # 用 getattr 而不是 self._net/_cfg: 单测把本方法绑到 SimpleNamespace
        # 假对象上跑(那些对象只有这两个属性), 语义与"网络层永不拖垮主循环"一致.
        rows, wall = _plan_ev(getattr(self, "_net", None),
                              getattr(self, "_cfg", None),
                              urgent=_urgent_now(self))
        return _ev_record_of(rows, wall)

    def _play_one_game(self, cfg: dict, game: int) -> bool:
        """打完一局, 回到准备页返回 True; 超时/异常返回 False."""
        spec = self._resolve_spec(cfg)
        vision = self._ensure_vision()
        # 网络读牌(liqi)对账门面: 整场共用一个(在 _run_games 里建好),
        # 这里只在没建过时补建 —— 直调本方法的测试/脚本仍然可用.
        # 桥不可用时一切对账静默回退视觉, 绝不拖垮主循环.
        if self._net is None:
            self._net = self._build_net(cfg)
        self._net_log_at = 0.0
        # 协议说"这一场完了"之后的计时起点(0 = 还没说过), 见下面的 1.6 分支.
        self._game_over_since = 0.0
        # 协议倒计时是否临近(每轮由 _net_clock 更新): 为真时压缩这一轮的节拍,
        # 免得想太久被雀魂托管. 见 tuning.NET_TURN_CLOCK.
        self._urgent = False
        deadline = time.time() + tuning.GAME_TIMEOUT
        # 开局宽限期: 点开始之后画面还要过加载 / 发牌动画, 这段时间手牌是
        # 逐张出现的(实测 0 张 -> 3 张 -> 12 张 -> 13 张, 前后约 10 秒).
        # 这属于正常, 不能算"认不出来", 也不该存待标注截图.
        # 卡死检测: 连续多少轮"感知结果完全一样且没有任何动作". 对局结束进结算页
        # 之后画面是静止的, 而牌河会被误识成"手牌"(实测 10 张, 刚好是 1 副露的
        # 合法张数), 于是每一轮都判成"在等别人", 一直空转到 game_timeout.
        # 判据见 src/vision.py 的 StuckDetector(为什么不用像素差也在那里).
        stuck = StuckDetector(tuning.STUCK_AFTER_TURNS)
        # 立直相关的两个状态机字段(意图 / 已生效)现在住在 RoundState 里,
        # 与它们的注释一起 —— 见 src/tasks/hand_loop.py。
        self._riichi_misses = 0
        # 立直意图的出生时刻(time.time()). 点下立直后超过
        # tuning.RIICHI_DECLARE_TIMEOUT 还没打出宣言牌就放弃这个意图 ——
        # 它挂了整手牌级别, 卡住不会自愈(见该常量的说明).
        self._riichi_since = 0.0
        # 立直生效的**协议证据**: 我们自己那一座位有没有真的收到
        # LiQiSuccess. False = 还没看到(不等于没有), 只用来配合宽限期做反向
        # 核对(见 tuning.RIICHI_VERIFY_GRACE), **绝不在验证那一刻否决**.
        self._my_riichi_confirmed = False
        # 立直**生效**的出生时刻(time.time()). 反向核对的宽限起点:
        # 回执随下一次摸牌下发, 太早问必然问不到.
        self._riichi_active_at = 0.0
        # 协议空闲跳帧的连续计数(见 _proto_idle_skip 与 NET_IDLE_SKIP).
        self._idle_skips = 0
        self._idle_skip_since = 0.0
        self._cfg_riichi_patience = int(tuning.RIICHI_DECLARE_PATIENCE or 0)
        self._cfg_bright_inset = float(tuning.BRIGHT_INSET or 0.0)
        # 连续几轮"点了出牌但手牌一张没少". 这是唯一能**立刻**发现"点了没反应"
        # 的信号: StuckDetector 的判据要求"这 N 轮没有任何动作", 而我们每轮都在
        # 点, 所以它看不到画面冻住 / 压着模态框的情况(见 tuning.DISCARD_MISS_LIMIT).
        self._discard_misses = 0
        # 低置信存帧护栏: 每局计数(防意外刷盘), 见 _save_low_conf_frame.
        self._low_conf_saves = 0
        # 自动开关(和/鸣/拔)是否已经确认点亮过: 每局复位, 见 _ensure_auto_toggles.
        self._toggles_ok = False
        # 这一手里我们自己点过杠(暗杠/明杠): 杠牌会竖着摆进手牌行右端,
        # 读手牌时要按间距把那段剥掉(见 _strip_kan_setaside / detection.
        # split_setaside). 换手(重臂)复位 —— 新的一手行里没有杠牌.
        self._kan_pending = False
        # 刚摸的牌(最右槽)本轮摸牌期间是否出现过低置信(粘性): 打出这张
        # 牌之前一直为真 —— 2026-10-09 实锤, 同一张牌一帧 0.34 一帧 0.95,
        # 只看当前帧会让高置信误读绕过听牌摸切守卫(见 table.decide_turn
        # 第 8.0.5 步). 打出后/换手后复位.
        self._drawn_blurry = False
        # 手间隙守卫(见 _hand_gap_arm / tuning.HAND_GAP_GUARD).
        self._hand_gap = False
        self._gap_since = 0.0
        self._gap_saw_closed = False
        # "这一手是协议的第几手"(见 _hand_gap_arm): None = 还没开过手.
        self._round_seq_at_open = None
        # 抓包链路已废(见 _capture_down): 切视觉 + 打完本局收工.
        self._capture_broken = False
        # 协议说的"刚摸到的那一张"(带赤记号, 如 "0p"). 保留赤 5 全靠它 ——
        # 视觉是 34 类, 认不出赤, 只有协议说得清刚摸的是不是赤 5(见
        # _proto_drawn 与 table.pick_tile_index). 桥不新鲜时恒为 None.
        self._drawn_tile = None
        # 副露记账(2026-10-09 東2局实锤后新增): 我们读不到副露区, 从
        # "点碰/杠前手牌"与"之后稳定读数"的差集反推副露牌种(见
        # table.meld_taken_kinds). _melded_tiles 喂给决策层的加杠支持
        # 判据(副露 3 张 + 手里 1 张 = 加杠, 只看暗杠判据时加杠会被
        # conf 门槛误杀); _pre_meld 是"点了碰/杠还没算出拿走什么"的
        # 暂存. 换手(重臂)复位 —— 副露只属于一手.
        self._melded_tiles: list[str] = []
        self._pre_meld: list[str] | None = None
        # 副露对账起点(点了碰/杠那一刻), 见 tuning.PRE_MELD_TIMEOUT 的说明.
        self._pre_meld_at: float = 0.0
        # 副露**组数**记账(2026-10-11 用户提议落地): 与 _melded_tiles(牌种,
        # 加杠判据用)并行, 这一份只管"这一手已有几组副露", 是张数硬校验
        # (_hand_count_ok)的输入. **只从数牌来**: 读数稳定且比点按钮前少了
        # 2 张以上, 组数才 +1 —— 加杠只从手里挪 1 张进副露, 数牌数不出来,
        # 正好不该 +1(加杠不增加副露组数); 按钮没点上数不出减少, 也不 +1.
        # **按"手"记, 不按"局"记**: 一场(东风/半庄)有很多手, 副露只属于
        # 一手 —— just_opened(这一手发齐 13 张, 副露必然为 0)和重臂(上一手
        # 结束)两处都归零, 两道闸互相兜底.
        self._meld_count: int = 0
        # 张数硬校验的连续不符计数(见 _hand_count_ok): 攒到
        # tuning.MELD_COUNT_RESYNC_FRAMES 就认定是记账错了, 按读数重同步.
        self._count_mismatch_streak: int = 0
        # 第几手(从 1 起): just_opened 递增, 进对局记录(jsonl)用.
        self._hand_seq: int = 0
        # 发牌闸门 + 开局宽限期 + 和了封锁, 三者是同一个"这一手还在不在打"
        # 的问题, 所以合到一个状态机里(见 table.DealWatch 的论证与实测数值).
        #
        # **为什么不能按"任务"而是必须按"手"**: 一次 _play_game 跑的是**整场**
        # (东风局/半庄会连着开好几手), 中间隔着一个结算页就换手了 —— 而老代码
        # 的 `deal_done` 一旦为真就永远是真、宽限期 25 秒之后恒为过期, 于是
        # **第二手起闸门必然失效**: 实跑 t158 在东 2 局发到 8 张时就出了牌.
        # 重臂的判据是"手牌区从有牌变成空"这个边沿(见 DealWatch.see).
        #
        # 发牌期间手牌是**逐张出现**的(实跑 t9~t19: 3 -> 4 -> 8 -> 13), 中间那些
        # 张数会撞上"合法张数" —— 8 恰好等于 2 副露时的暗牌数, 于是
        # is_my_discard_turn(8) 为真, 决策层就照着一手还没发完的牌算出了出牌.
        # 2026-10-07 实跑 t18 就是这样: 手牌才 8 张, 它决定打 1m, 点了两下都没
        # 反应(那时候雀魂根本不收出牌), 还把"出牌模式"误判成"选中再确认".
        # 判据为什么是">=13": 开局不可能有副露(副露得等别人先打牌), 所以"手牌
        # 发齐"这件事只可能落在 13(或摸牌后的 14)张上. 见过的第一手 >=13 之后
        # 就可以按正常逻辑走了(那时才可能真的副露到 10/7/4 张).
        #
        # 为什么不用"点完开始等 N 秒": 写死秒数要么不够(机器慢/加载久), 要么
        # 太久 —— 而太久会撞上雀魂自己的出牌计时被**托管**, 那是比打错一张牌
        # 严重得多的事故(见 docs/ANTI_DETECTION.md). 这里等的是"发牌结束"这个
        # 事实, 不是时间; 宽限期只当兜底(见 tuning.GAME_GRACE_PERIOD).
        self._deal = DealWatch(
            grace_period=tuning.GAME_GRACE_PERIOD,
            grace_until=time.time() + tuning.GAME_GRACE_PERIOD)
        # 牌桌特征"还在不在"的状态机. 用来在认不出来超过 TABLE_WAIT_LIMIT 之后
        # **放弃等待**, 免得特征万一失灵就把任务锁死成"一张牌不打"
        # (见那个常量的说明, 以及 table.TableWatch).
        self._table_watch = TableWatch()
        # 闸门放弃时存诊断帧的"一次"开关: 同一段放弃只存第一帧, 牌桌重新认出
        # 后复位(下一段放弃再存). 2026-10-11 三人场连续两局在点完自动开关后
        # ~20s 闸门放弃(视觉上无从判断当时游戏显示的是什么), 没有这张帧只能瞎猜.
        self._gate_giveup_saved = False
        # 协议手牌对不上时最近一次留证的时刻(0 = 还没存过), 限频见
        # tuning.NET_TRUST_MISMATCH_SHOT_INTERVAL.
        self._proto_mismatch_at = 0.0
        # "协议张数与视觉张数互相矛盾"从哪一刻开始(0 = 现在没矛盾). 秒数用于
        # 日志显示与留证阈值(NET_CONFLICT_SETTLE); **没有超时退视觉**(2026-10-18
        # 20:21 局用户拍板): 退回去那一刻视觉可能还是假牌/动画框, 等成托管也
        # 比点错牌强.
        self._net_conflict_since = 0.0
        # "同一手牌里重复出现的同一条告警只报第一次"的去重表: 键 -> 上次的签名.
        # 每重臂一手(手牌区清空 -> 重新发牌)清空, 见 _log_repeat_once.
        self._repeat_log: dict[str, str] = {}

        # 主循环拆到 src/tasks/hand_loop.py: 阶段序列 + 每轮状态.
        # 这里只剩"这一局开始时的状态初始化"和"跑起来".
        st = RoundState(cfg=cfg, game=game, spec=spec, vision=vision,
                        deadline=deadline, stuck=stuck)
        return HandLoop(self, st).run()

    def _net_riichi_seats(self) -> list[int] | None:
        """协议里已立直的座位; 协议不可用/拿不到就返回 None(= 不否决).

        **必须先 poll**: 桥是 pull 式的, 只读不喂拿到的永远是上一轮
        `_net_step` 那一帧的冻结状态. 2026-10-18 真机: 立直回执随下一次摸牌
        下发(比宣言牌晚 0.5s), 而本方法在验证那一刻读, 没有 poll 就看不到
        —— 于是把一次**已经成立**的立直判成"没有回执"并撤销.

        只读门面, 永不抛 —— 与网络层其它入口同一条原则.
        """
        if self._net is None:
            return None
        mode = self._cfg.get("net_read") or tuning.NET_READ
        if mode == tuning.NET_READ_OFF:
            return None
        try:
            self._net.poll()
            if not self._net.available():
                return None
            state = getattr(self._net, "state", None)
            seats = getattr(state, "riichi_seats", None)
            return list(seats()) if callable(seats) else None
        except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
            return None

    def _net_my_seat(self) -> int | None:
        """协议里的"我的座位"; 拿不到返回 None."""
        if self._net is None:
            return None
        try:
            state = getattr(self._net, "state", None)
            seat = getattr(state, "my_seat", None)
            return int(seat) if seat is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _proto_my_riichi(self) -> bool | None:
        """协议对我"这一手立直了没"的三态回答.

        True = 拿到了我座位的立直回执 / False = **明确**没有 / None = 不参战.
        **三态区分是这条的关键**: 桥不可用、座位表还没建、不知道我是哪个
        座位, 三种"拿不到"都必须是 None —— 拿沉默当反对会把一次真立直
        撤销掉, 而撤销之后的代价是"整手不再主动出牌"(靠雀魂托管收尾).

        False 这一支才允许做否决, 且调用方还要再叠一道
        `tuning.RIICHI_VERIFY_GRACE` 宽限: 回执是随下一次摸牌下发的, 刚打完
        宣言牌那一刻**本来就不该有**.
        """
        seats = self._net_riichi_seats()
        if seats is None:
            return None
        mine = self._net_my_seat()
        if mine is None:
            return None
        return mine in seats

    def _bright_indices(self, frame, hand) -> list[int] | None:
        """立直选牌态里哪些牌是高亮的(能点的); 不在选牌态返回 None.

        判据是**亮度**: 变暗的牌点上去雀魂不响应(实跑里白等过 5 轮, 手牌数
        一直是 14), 高亮的才能点. 高亮牌通常不止一张 -- 任何能组成立直的
        牌都可以当宣言牌, 选哪张由牌效决定.

        返回 None 表示"这帧看不出选牌态"(没有明显的亮暗分组), 这时该空等
        一轮重认, **不能猜**: 猜错就是点在变暗的牌上, 纯白等.
        连续认不出超过 riichi_declare_patience 轮就降级为"按普通出牌处理"
        (返回全部下标), 免得无限空转到 game_timeout -- 实跑踩过 15 分钟.

        亮暗分组用中位数当分界(见 src/detection.split_bright_dim), 不依赖
        "整副手牌偏亮还是偏暗"这种全局设置.
        """
        boxes = hand.boxes
        if len(boxes) < 3:
            return None
        values = [tile_brightness(frame, b, inset=self._cfg_bright_inset)
                  for b in boxes]
        if any(v is None for v in values):
            self.log_warning("  有牌裁不出亮度区域，本轮不判定高亮")
            return None
        bright, dim = split_bright_dim(boxes, values)
        if not bright:
            self._riichi_misses += 1
            if self._riichi_misses >= self._cfg_riichi_patience:
                self.log_warning(
                    f"  连{self._riichi_misses} 轮认不出高亮牌（亮度"
                    + ",".join(f"{v:.0f}" for v in values)
                    + "），按普通出牌处理")
                return list(range(len(boxes)))
            self.log_warning(
                "  立直选牌态但没分出高亮牌（亮度"
                + ",".join(f"{v:.0f}" for v in values) + "），下轮重试")
            return None
        self.log_info(
            f"  立直选牌态：高亮 {len(bright)}/{len(boxes)} 张"
            f"（下标 {bright}，亮度 "
            + ",".join(f"{values[i]:.0f}" for i in bright)
            + f"，其余 {values[dim[0]]:.0f} 左右）")
        self._riichi_misses = 0
        return bright

    def _read_hand(self, vision: GameVision, frame, cfg: dict):
        """读手牌: 剥离杠牌暂存区 + 低置信留证. 标签一律交给协议对账纠正.

        历史上这里还有一层"多帧投票"(连续几帧同一槽位取众数)压低置信误识,
        2026-10-18 整体删除: 投票改动的是**标签**, 而用户的拍板是"标签只有
        协议一个来源, 视觉不一致时保留截图" —— 投票与协议对账叠着跑, 等于
        在协议真值之上再叠一层视觉猜想. 15:21 局(日志写打 3p 实际 4p)与
        16:21 局("纠正4槽"后点 9p 打出 4p)都是它把标签推了一格造成的.
        低置信槽位的真正出路是决策层的稳健出牌(见 detection.low_confidence).
        """
        hand = self._strip_kan_setaside(vision.read_hand(frame))
        self._save_low_conf_frame(frame, hand)
        return hand

    def _log_repeat_once(self, key: str, signature: str, message: str,
                         level: str = "info") -> None:
        """同一手牌里签名没变的同一条告警只打第一次.

        **为什么需要**(2026-10-07 实跑复盘): 主循环 0.3s 一轮, 而"某槽位置信度
        偏低"会**连续几十轮**都成立 —— 实测 t176~t185 位置 4 连续 10 轮低置信,
        653 条"置信度偏低"里绝大多数是同一处的复读, 真正的新信息被埋在刷屏里,
        还把 logs/ok-script.log 撑到几十万字节, 事后"跑完比对日志大小"的基线
        也跟着失效.

        签名一变(槽位组合变了 / 认出的牌变了 / 漏认张数变了)就立刻重报一次,
        所以"情况真的变坏"不会被吞掉; 只有"完全没变"才静默. 签名按**手**清空
        (发牌重臂处), 因为换了一手牌, 上一次的判断不该继续生效.
        """
        if self._repeat_log.get(key) == signature:
            return
        self._repeat_log[key] = signature
        (self.log_warning if level == "warning" else self.log_info)(message)

    def _handle_result_page(self, frame, cfg: dict, spec, game: int,
                            turns: int) -> bool:
        """认出结算页就点"确认", 一直点到离开结算页为止. 返回 True 表示这局已收尾.

        为什么必须专门处理: 结算页上手牌区是空的, 牌面模型会把牌河误识成
        "手牌"(实测 10 张), 而那张画面一动不动 -- 于是每一轮都判成"在等别人",
        一直空转到 game_timeout(实跑踩过: 白等 15 分钟).
        StuckDetector 是兜底(40 轮 ≈ 12 秒), 这里是正常路径(0.3 秒).

        注意 "确认" 按钮在结算页刚出来时可能还没出现(要先等对手确认 /
        点数结算动画), 所以探测不到不算失败, 下一轮继续轮询.

        **为什么要循环**(2026-10-07 用户报"终局页其实有两个"): 一局打完会连着
        好几页 —— 先是点数结算(可能被 UI 模型的 confirm 顺手点掉), 然后是我
        们标注的「终局页」, 如果当天有每日任务完成, 后面还跟着一页「任务完成页」
        (带「完成」印章 + 铜币奖励进度)。实拍对比那两页: 「结算页/终局」命中
        0.989、「结算页/确认」命中 0.999, **位置一分不差**, 所以不用为新页加
        标注; 但点一次就 return 会把最后一页留在屏幕上 —— 连打时下一局的
        "开始"按钮根本找不到, 要白等满 START_WAIT_TIMEOUT 才报错。
        上限 RESULT_MAX_PAGES 防止页面认错时空转到超时。
        """
        bench.count_state("result_page")  # bench 挂钩(默认关闭)
        pages = 0
        while pages < tuning.RESULT_MAX_PAGES:
            # 判页特征两个, 命中任一就算结算页: 主 = 「终局」标题, 兜底 =
            # 「确认」按钮. 2026-10-07 21:41 实跑: 结算页标题被人物皮肤挡住
            # (实测只剩 0.660, 阈值 0.88), 页面认不出来, 任务在终局页上干等
            # 到用户手动关程序. 而确认按钮是 UI 素材, 皮肤帧上裸匹配 1.000
            # 分、位置就在标注处. 局内流局/和了摊牌画面的确认是另一种样式
            # (实测 0.31~0.37 且在底部中央, 不在搜索区里), 不会误触.
            box = (self._locate(RESULT_SCREEN, RESULT_FEATURE, cfg, frame,
                                strict=True, verbose=False)
                   or self._locate(RESULT_SCREEN, RESULT_CONFIRM, cfg, frame,
                                   strict=True, verbose=False))
            if box is None:
                break
            pages += 1
            self.log_info(f"第 {game} 局已打完（认出结算页第 {pages} 页），"
                          f"点「{RESULT_CONFIRM}」")
            confirm = self._wait_for_button(
                RESULT_SCREEN, RESULT_CONFIRM, cfg,
                timeout=tuning.RESULT_CONFIRM_TIMEOUT)
            if confirm is None:
                # 先试奖励页: 「获得奖励」弹窗也带「终局」标题(实测 NCC
                # 1.000、位置一分不差), 会被上面的判据认成"又一页结算页",
                # 但它没有「确认」按钮, 等满超时也点不掉. 2026-10-12 02:39
                # 实跑: 三个调用点都把 _handle_reward_page 排在本函数后面,
                # 永远轮不到 —— 结果在 _back_to_lobby 里空转 60s 后
                # RuntimeError 把整轮连打掀翻. 在这里转交(它内部重抓帧
                # 复查, 点干净为止), 三个调用点就都覆盖到了.
                if self._handle_reward_page(frame, cfg, game):
                    return True
                # 页面认出来了但按钮还没出现: 可能是点数结算还在算, 或者这一局
                # 是中途认输/逃跑, 结算页长得不一样. 截图留证后按"结束"处理,
                # 交给外层的卡死检测收尾.
                self._save_unknown_page(frame, "结算页但没有确认按钮")
                self.log_warning(
                    f"认出结算页但等不到「{RESULT_CONFIRM}」"
                    f"（{tuning.RESULT_CONFIRM_TIMEOUT}s），已截图留证。"
                    f"若这一局确实打完了，请用「截图标注」核对结算页的"
                    f"「{RESULT_CONFIRM}」区域是否需要重新框。")
                return True

            delay = timing.idle_delay(self._rng)
            self.log_info(f"  想 {delay:.1f}s 后点 {RESULT_CONFIRM}")
            self.sleep(delay)
            self._click_box(confirm, RESULT_SCREEN, RESULT_CONFIRM, cfg)
            # `confirm` 是**框架的 Box**（`_wait_for_button` -> `_locate` -> `find_one`）,
            # 不是我们自己的 Detection。两者的"取中心"接口形态不同（Box 是方法、
            # Detection 是 property）, 所以统一走 `_box_center` —— 别写 `.cx`/`.cy`,
            # 那是 Detection 才有的别名, 实跑在这里抛过 AttributeError 把整局判成
            # 失败（2026-10-07）。
            self._record({"event": "result_confirm", "game": game, "turn": turns,
                          "page": pages, "point": list(self._box_center(confirm))})
            self.sleep(tuning.RESULT_CONFIRM_GAP)
            # 再抓一帧看还有没有下一页(任务完成页). 用新帧而不是这一轮开头那张 --
            # 那一张是"点确认之前"的画面, 拿它判会永远认为还有一页.
            nxt = self._grab()
            if nxt is None:
                break
            frame = nxt

        if pages == 0:
            return False
        if pages >= tuning.RESULT_MAX_PAGES:
            self.log_warning(
                f"连着点了 {pages} 页结算页还是没回到准备页，"
                f"停止点击（上限 {tuning.RESULT_MAX_PAGES}），交给外层处理")
        self.log_info(f"第 {game} 局结束（结算页 {pages} 页 -> 确认），共 {turns} 轮")
        return True

    def _handle_reward_page(self, frame, cfg: dict, game: int) -> bool:
        """认出终局奖励页就照结算页确认位点击, 点到翻页为止.

        返回 True 表示这一帧认出并处理过奖励页(至少点了一次).

        为什么要专门处理(2026-10-11 用户报): 段位场打过几局后可能弹出
        「获得奖励」页, 盖住整块画面 —— 现有结算页判据(「终局」标题 /
        「确认」按钮)在这一页上**全都命不中**(它没有按钮素材可标),
        画面一动不动, 任务就卡死在页面上直到 game_timeout. 用户实测:
        在原结算页「确认」按钮的位置点一下就能翻页 —— 所以判页靠新标注
        的「获得奖励」标题(ok_templates/reward.png, 实测自匹配 1.000、
        全帧第二峰 0.326, 阈值 0.88 下唯一), 点击照搬标注中心坐标
        (tuning.RESULT_CONFIRM_CENTER, 按帧分辨率缩放).

        与 _handle_result_page 一样点完要复查: 弹窗可能不止一层(奖励列表
        后面还有进度条翻页), 抓新帧看标题还在不在, 在就再点; 上限
        tuning.REWARD_MAX_CLICKS 防误判时在一个坐标上无限空点.
        """
        bench.count_state("reward_page")  # bench 挂钩(默认关闭)
        clicks = 0
        while clicks < tuning.REWARD_MAX_CLICKS:
            box = self._locate(REWARD_SCREEN, REWARD_FEATURE, cfg, frame,
                               strict=True, verbose=False)
            if box is None:
                break
            clicks += 1
            h, w = frame.shape[:2]
            px = round(tuning.RESULT_CONFIRM_CENTER[0] * w / 1920)
            py = round(tuning.RESULT_CONFIRM_CENTER[1] * h / 1080)
            delay = timing.idle_delay(self._rng)
            self.log_info(f"第 {game} 局后认出奖励页（第 {clicks} 次），"
                          f"{delay:.1f}s 后点旧确认位 ({px}, {py})")
            self.sleep(delay)
            self._ensure_foreground()
            self.click(x=px, y=py)
            self._record({"event": "reward_confirm", "game": game,
                          "click": clicks, "point": [px, py]})
            self.sleep(tuning.RESULT_CONFIRM_GAP)
            # 抓新帧而不是拿点击前的画面复查 —— 否则永远认为还有一页
            # (见 _handle_result_page 里同样的坑).
            nxt = self._grab()
            if nxt is None:
                break
            frame = nxt
        if clicks == 0:
            return False
        if clicks >= tuning.REWARD_MAX_CLICKS:
            self.log_warning(
                f"点了 {clicks} 次奖励页标题还在，停止点击（上限 "
                f"{tuning.REWARD_MAX_CLICKS}），交给外层处理")
        return True

    def _drain_result_pages(self, cfg: dict, spec, game: int,
                            max_wait: float = 15.0) -> None:
        """把还留在屏幕上的结算页点干净(最多等 max_wait 秒).

        为什么要专门做(2026-10-07 19:17 实跑): _handle_result_page 点完一页
        后抓新帧看还有没有下一页, 但下一页(终局排名 / 任务完成页)可能**晚
        一拍**才画出来 —— 那一帧没认出它就收工了, 页面留在屏幕上, 连打的
        "开始"按钮根本不在这一页, 干等满 START_WAIT_TIMEOUT. 用户看到的
        就是"循环两次卡在终局页". 所以每局之间、等"开始"之前先清一遍;
        点掉一页后继续轮询, 给更晚出现的页面机会, 直到一页都没有为止.
        """
        deadline = time.time() + max_wait
        while time.time() < deadline:
            if self._should_yield():
                raise TaskStopped("清理结算页期间收到停止指令")
            frame = self._grab()
            if self._handle_reward_page(frame, cfg, game):
                self.log_info("  又点掉一页奖励页，看看还有没有下一页")
                continue
            if not self._handle_result_page(frame, cfg, spec, game, 0):
                return
            self.log_info("  又点掉一页结算页，看看还有没有下一页")

    def _handle_hand_result(self, frame, cfg: dict, game: int) -> bool:
        """认出一手结束的顺位结算画面就点「确认」, 返回 True 表示这轮别读手牌.

        背景与实测数值见 HAND_RESULT_SCREEN 常量的说明. 与整局的结算页
        (_handle_result_page)是两回事: 这个画面**每一手**结束都弹一次,
        「确认」按钮带未确认人数(动态), 模板只框静态的「确认」二字;
        流局画面没有这个按钮, 约 5s 自动过 —— 漏判无代价(回落到读手牌的
        旧路径), 所以探测用严格阈值且不做宽松重试(宁可漏判不误判).

        重复点击是安全的: 按钮在别人确认前一直显示, 人也是这么连点的;
        日志按手去重(_repeat_log 在换手时清空), 不会刷屏.
        """
        bench.count_state("hand_result")  # bench 挂钩(默认关闭)
        box = self._locate(HAND_RESULT_SCREEN, HAND_RESULT_CONFIRM, cfg, frame,
                           strict=True, verbose=False)
        if box is None:
            return False
        first = self._repeat_log.get("hand_result") != "1"
        self._log_repeat_once(
            "hand_result", "1",
            "  一手结束（和了/流局顺位结算画面），点「确认」继续")
        delay = timing.idle_delay(self._rng)
        self.sleep(delay)
        self._click_box(box, HAND_RESULT_SCREEN, HAND_RESULT_CONFIRM, cfg)
        self._record({"event": "hand_result_confirm", "game": game,
                      "point": list(self._box_center(box))})
        if first:
            # 点掉「(N)确认」= 这一手正式结束. 必须显式重臂发牌闸门: 点确认
            # **加速了过场**, 画面可能直接切到下一手发牌, "手牌区清空"的边沿
            # 根本不出现 —— 上一手的 riichi_active 等状态全部滞留, 下一手
            # 每轮决策出牌却一张不打(2026-10-08 实跑: 东 2 干等 30 秒, 用户
            # 以为任务死了). 真正的状态复位统一在主循环 rearmed 分支.
            self._deal.force_rearm(time.time())
        self.sleep(tuning.RESULT_CONFIRM_GAP)
        return True

    def _press_start(self, spec, cfg: dict, game: int) -> None:
        """开下一局: 先清残留结算页, 再等"开始"并点击.

        等"开始"期间每 20s 留一张现场帧 —— 这次实跑卡在没标注的页面上,
        光看日志只有一句相似度 0.2, 不知道停在哪儿; 有帧才能补标注.
        """
        step = start_step(spec)
        self._drain_result_pages(cfg, spec, game)
        deadline = time.time() + tuning.START_WAIT_TIMEOUT
        next_score = time.time() + 5.0
        next_evidence = time.time() + 20.0
        frame = None
        while time.time() < deadline:
            if self._should_yield():
                raise TaskStopped(f"等待 '{step.target}' 期间收到停止指令")
            frame = self._grab()
            # 等待期间也可能有结算页慢悠悠地弹出来, 见 _drain_result_pages.
            if self._handle_result_page(frame, cfg, spec, game, 0):
                continue
            box = self._locate(step.screen, step.target, cfg, frame,
                               verbose=False)
            if box is not None:
                self.log_info(
                    f"    匹配成功 '{step.target}' -> ({box.x},{box.y}) "
                    f"{box.width}x{box.height} conf={box.confidence:.3f}")
                self._click_box(box, step.screen, step.target, cfg)
                return
            now = time.time()
            if now >= next_score:
                next_score = now + 5.0
                best = self._best_match(step.screen, step.target, cfg, frame)
                self.log_info(
                    f"    还在等 '{step.target}'"
                    + (f"，当前最高相似度 {best[0]:.3f}" if best else "")
                    + f"（阈值 {tuning.TEMPLATE_THRESHOLD}）")
            if next_evidence and now >= next_evidence:
                next_evidence = None
                self._save_unknown_page(
                    frame, f"等开始_20s_画面未识别_第{game}局前")
            self.sleep(tuning.PAGE_POLL_INTERVAL)
        self._save_wait_timeout_frame(frame, step.screen, step.target)
        self._diagnose_missing(step.screen, step.target, cfg)
        raise RuntimeError(
            f"等待 '{step.screen}/{step.target}' 出现超时"
            f"（{tuning.START_WAIT_TIMEOUT}s），已中止任务")

    def _table_screen(self, cfg: dict) -> str:
        """牌桌门控用哪套标注: 3P / 4P 的操作条内容不同(4P 没有「拔」,
        各行位置也不同), 3P 标的模板在 4P 帧上永远匹配不到 —— 2026-10-07
        四人场实跑开局连等 20s 的根因。"""
        if self._players_key(cfg) == "四人":
            return f"{TABLE_SCREEN}四人"
        return TABLE_SCREEN

    def _at_table(self, cfg: dict, frame) -> bool:
        """这一帧的牌桌出来了没有(= 在对局中/结算画面上). 判据与实测见 TABLE_SCREEN.

        用严格阈值且不做宽松重试: 这是**页面门控**, 判错的代价不对称 ——
        误判成"在牌桌"只是白跑一轮感知(和以前一样), 误判成"不在牌桌"会让任务
        一张牌都不打, 所以宁可保守; 同时外层有 TABLE_WAIT_LIMIT 兜底。
        阈值用专用的放宽值(TABLE_DETECT_THRESHOLD=0.80, 非 0.88): 竖条字符
        就是自动开关, 点亮变绿后模板分会掉到 0.88 附近(见 tuning 注释,
        2026-10-11 三人场"点完开关就报没进牌桌"的根因), 非牌桌页面最高
        只有 0.33, 放宽到 0.80 任何开关状态都稳过、裕度依然巨大。
        """
        return self._locate(self._table_screen(cfg), TABLE_FEATURE, cfg, frame,
                            strict=True, verbose=False,
                            strict_threshold=tuning.TABLE_DETECT_THRESHOLD) is not None

    def _table_gate(self, cfg: dict, frame) -> str:
        """这一帧该不该跳过感知. 返回 table 的四个状态常量之一.

        调用方把 TABLE_WAITING 当成"牌桌还没出来, 这一轮别动"; TABLE_OK 按
        正常逻辑继续; **TABLE_GAVE_UP / TABLE_IGNORING 也整轮跳过**(2026-10-10
        审计修改②: 放弃后不再读手牌, 靠外层 GAME_TIMEOUT 兜底) —— 也就是说
        这个门控**只会漏判不会锁死**: 特征失灵超过 TABLE_WAIT_LIMIT 后放弃
        感知, 但页面级门控(准备页/结算页)仍在上方正常处理, 不会卡死过场。

        为什么要单独抽出来: 它同时管着三件事, 每一件出问题都很难在实跑里看出来
        —— 计时上限、重新认出后复位、以及"放弃"的警告**只能报一次**(循环每
        0.3s 一轮, 不限流就是几百条刷屏). 见 table.TableWatch 与
        tests/TestPlayTask.py::TestTableGate.
        """
        bench.count_state("table_gate")  # bench 挂钩(默认关闭)
        present = self._at_table(cfg, frame)
        if not present and self._table_watch.gone_since == 0.0:
            self.log_info(
                "还没进牌桌（加载 / 对局开始页 / 过场），等牌桌出来再读手牌")
        state = self._table_watch.see(present, time.time(),
                                      limit=tuning.TABLE_WAIT_LIMIT)
        if state == TABLE_GAVE_UP:
            self.log_warning(
                f"牌桌特征连续 {self._table_watch.gone_for(time.time()):.0f}s "
                f"认不出来（阈值 {tuning.DETECT_THRESHOLD}），停止读手牌，"
                f"等特征恢复或超时兜底；若一直这样，请在「模板」页签核对 "
                f"'{self._table_screen(cfg)}/{TABLE_FEATURE}' 的标注区域")
            self._save_gate_giveup_frame(frame)
        elif present:
            self._gate_giveup_saved = False
        return state

    def _save_proto_mismatch(self, frame, note: str) -> None:
        """协议手牌与视觉对不上: 留一张现场帧(限频), 本轮照旧走视觉.

        信任档(NET_TRUST_PROTO)的失败方向必须是"看得见": 悄悄回退 = 协议漂移
        (某条报文漏记/多记一张牌)永远不会被发现, 而"信任协议"恰恰建立在
        "它被持续核对"上. 走 _save_unknown_page(待标注_ 前缀, 不会被修剪),
        与 frame_gate_/frame_lowconf_ 同属保留帧.
        """
        now = time.time()
        if (now - getattr(self, "_proto_mismatch_at", 0.0)
                < tuning.NET_TRUST_MISMATCH_SHOT_INTERVAL):
            return
        self._proto_mismatch_at = now
        self._save_unknown_page(frame, f"协议手牌对不上 {note[:60]}")

    def _save_gate_giveup_frame(self, frame) -> None:
        """闸门放弃时留一张现场帧(一次放弃段只存第一张, 牌桌恢复后重新武装).

        为什么必须有: 闸门放弃意味着"画面和标注对不上", 但放弃后不再读手牌
        也不点任何东西 —— 游戏当时到底显示的是什么(过场? 动画? 界面变了?),
        日志里一个字都没有, 只能靠用户肉眼回忆. 2026-10-11 三人场两局都是
        点完自动开关后 ~20s 放弃, 没有帧, 根因查不动. 存进 data/debug/
        (frame_gate_ 前缀), tools/label_tiles 的扫描队列会自动捡走 frame_*.png.
        """
        if frame is None or self._gate_giveup_saved:
            return
        self._gate_giveup_saved = True
        try:
            DEBUG_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%m%d_%H%M%S")
            path = DEBUG_ARTIFACTS_DIR / f"frame_gate_{ts}.png"
            imwrite_unicode(path, frame)
            self.log_warning(f"  已存牌桌特征失灵现场帧: {path.name}")
        except Exception as e:  # noqa: BLE001
            self.log_warning(f"  闸门现场帧保存失败: {e}")

    def _confirm_stable(self, decision, hand, buttons, cfg: dict, spec,
                        vision: GameVision,
                        pending_riichi: bool, ev_rank=None) -> tuple[bool, str]:
        """动作前抓第二帧重算一遍, 判据在 table.action_confirmed(纯逻辑).

        第二帧的输入完全镜像主循环那一轮: 同样的读手牌(杠牌剥离)、
        读按钮、高亮牌、低置信槽位, 只是 rng 换成固定种子 —— 同一读数下
        决策必须可复现, 否则"要打的牌变了"会被随机性(5% 次优)误触发.
        """
        if _urgent_now(self):
            # 协议倒计时快到了: 二次确认还要再花 ~0.55s + 抓一帧, 这一轮花不起.
            # 有协议时钟兜底的时候, "按老样子直接动手"也比被托管强 —— 托管会
            # 替我们连吃带碰还乱打. 见 tuning.NET_URGENT_SECONDS.
            return True, "协议倒计时快到了，跳过二次确认"
        self.sleep(tuning.STABLE_CONFIRM_DELAY)
        frame2 = self._grab()
        if frame2 is None:
            return False, "二次抓帧失败"
        hand2 = self._read_hand(vision, frame2, cfg)
        buttons2 = vision.read_buttons(frame2)
        conf2 = {name: boxes[0].conf for name, boxes in buttons2.items()}
        centers2 = {name: self._box_center(boxes[0])
                    for name, boxes in buttons2.items()}
        allowed2 = (self._bright_indices(frame2, hand2)
                    if pending_riichi else None)
        blurry2 = (low_confidence(hand2.boxes, float(tuning.MIN_TILE_CONF))
                   if tuning.MIN_TILE_CONF else None)
        # 镜像主循环的粘性摸切标志: 第二帧里刚摸的牌也可能暴露低置信
        # (第一帧高置信误读正是要防的场景, 见 table.decide_turn 8.0.5).
        if (hand2.count % 3 == 2 and blurry2
                and len(hand2.tiles) - 1 in blurry2):
            self._drawn_blurry = True
        second = decide_turn(
            hand2.tiles, conf2,
            player_count=spec.player_count,
            meld_policy=tuning.MELD_POLICY,
            allow_riichi=tuning.ALLOW_RIICHI,
            meld_min_conf=tuning.MELD_BUTTON_MIN_CONF,
            riichi_min_conf=tuning.RIICHI_BUTTON_MIN_CONF,
            babei_policy=tuning.BABEI_POLICY,
            allowed_indices=allowed2,
            blurry_indices=blurry2,
            missing_slots=hand2.missing_slots,
                button_centers=centers2,
                blurry_draw=self._drawn_blurry,
                melded=self._melded_tiles,
                # 镜像第一帧的协议权威闸: 两帧必须同参, 否则"动作变了"
                # 会被闸门差异误触发(见 action_confirmed 的判据).
                my_turn=_proto_turn_of(self),
                proto_turn=_proto_clock_turn_of(self),
                round_open=_proto_round_open_of(self),
                # 第二帧沿用第一帧那份 EV 排序: 同一手里的牌河/宝牌/牌山不会在
                # 0.4s 里变, 而 pick_by_ev_rank 会按第二帧的读数落地 —— 两帧
                # 同参才不会把"目标函数不同"误判成"要打的牌变了".
                ev_rank=ev_rank,
                rng=random.Random(0),
                drawn_tile=getattr(self, "_drawn_tile", None),
            )
        center1 = None
        if decision.button:
            boxes1 = buttons.get(decision.button) or []
            center1 = self._box_center(boxes1[0]) if boxes1 else None
        ok, why = action_confirmed(decision, second, center1,
                                   centers2.get(decision.button),
                                   tol=tuning.BUTTON_STABLE_TOL)
        if not ok:
            # 把两帧的差异摊开报: 只有"动作变了"三个字时, 事后无法区分到底是
            # 按钮漏检(视觉问题)还是手牌读数抖了(感知问题)、还是协议闸翻转.
            # 2026-10-18 14:58 局就是只留了三个字, 复盘只能靠猜.
            why += (f"；第一帧 {decision.action}/{decision.button or '-'}"
                    f" 手牌 {getattr(hand, 'count', '?')} 张"
                    f"(空档 {getattr(hand, 'missing_slots', '?')},"
                    f" 按钮 {sorted(buttons) or '无'}),"
                    f" 第二帧 {second.action}/{second.button or '-'}"
                    f" 手牌 {hand2.count} 张(空档 {hand2.missing_slots},"
                    f" 按钮 {sorted(buttons2) or '无'})")
        return ok, why

    def _execute(self, decision, hand, buttons, cfg: dict,
                 game: int, turns: int) -> None:
        """执行一个动作, 并按人的节奏等一会儿.

        协议倒计时临近时(见 tuning.NET_TURN_CLOCK)把拟人延时压到
        `NET_URGENT_DELAY`: 少一点"像人"总好过被雀魂托管接管 —— 托管会替我们
        连吃带碰还乱打, 代价大得多. 这是**唯一**会压缩节拍的情形, 正常局面照旧.
        """
        urgent = _urgent_now(self)
        if urgent:
            left = getattr(self, "_urgent_left", None)
            how = f"剩 {left:.1f}s" if isinstance(left, (int, float)) else "临期"
            self.log_warning(f"  协议倒计时{how}，压缩节拍立即行动")
        if decision.action == ACTION_DISCARD:
            delay = (tuning.NET_URGENT_DELAY if urgent
                     else timing.discard_delay(self._rng))
            self.log_info(f"  想 {delay:.1f}s 后打 {decision.tile}"
                          f"（{decision.reason}）")
            self.sleep(delay)
            self._discard(decision, hand, cfg, game, turns)
            # 出牌动作已发出: 这一次摸牌结束, 摸切粘性标志复位(下次摸牌
            # 重新观察). 点了没反应的场景由 _discard_misses 机制兜底,
            # 且下一轮同牌再低置信会重新置位.
            self._drawn_blurry = False
            return

        boxes = buttons.get(decision.button) or []
        if not boxes:
            self.log_warning(f"  按钮 {decision.button} 没有框，跳过")
            return

        delay = (tuning.NET_URGENT_DELAY if urgent
                 else self._action_delay(decision))
        self.log_info(f"  想 {delay:.1f}s 后点 {decision.button}"
                      f"（{decision.reason}）")
        self.sleep(delay)
        self._click_box(boxes[0], "按钮", decision.button, cfg)
        self._record({"event": "click", "game": game, "turn": turns,
                      "action": decision.action, "button": decision.button,
                      "point": list(self._box_center(boxes[0]))})
        if decision.action == ACTION_MELD and decision.button == BUTTON_GANG:
            # 杠的 4 张牌马上会竖着摆进手牌行右端(2026-10-08 实锤的死锁
            # 根源), 之后所有读手牌都要把它们剥掉, 直到这一手结束.
            self._kan_pending = True
        if decision.action == ACTION_MELD:
            # 记下点击前手牌与时刻: 之后读数稳定时从差集反推副露牌种与组数
            # (数牌: 少 2 张以上才算一组 —— 加杠只挪 1 张, 数不出来, 也正
            # 好不该 +1; 见 tuning.PRE_MELD_TIMEOUT 与主循环的副露记账).
            # 组数**不在这里 +1**: 按钮有没有点成、是不是加杠, 都得靠之后
            # 的读数说话.
            self._pre_meld = list(hand.tiles)
            self._pre_meld_at = time.time()

    def _hand_count_ok(self, hand, frame) -> bool:
        """张数硬校验: 手牌张数必须落在"记住的副露组数"对应的两个合法值里.

        为什么要这道闸(2026-10-11 三碰局实锤): 三碰后副露被读进手牌,
        张数 8 恰好 %3==2, 决策层判"该我出牌"并算出打 3s —— 而那是已碰掉
        的牌, 点副露雀魂不响应, 死循环到本局结束. 老的 is_my_discard_turn
        是从张数**反推**副露数(count%3==2 ⟺ 存在某个 n 使 count=14-3n),
        污染的读数自己就能把自己圆过去; 现在副露组数是**独立数牌**记的
        (点按钮后读数少了 2 张以上才 +1, 见 _meld_count), 读数圆不了.

        返回 False = 这一轮读数不可信, 主循环跳过本轮不出任何动作.

        四个出口, 按保守程度排:

        - 协议明确说这一手已经结束(`round_open is False`, 过场/和牌演出):
          眼前这"手牌"是残影, 根本没有"活手牌的张数"可比 —— 既不闸也
          绝不重同步(2026-10-18 19:51 局实锤: 门清立直荣和, 收牌动画连续
          3 帧只读出 8 张, 旧逻辑把正确的"副露 0 组"记账重同步成 2 组.
          记账是对的, 读数是错的). 协议没说话(None)或说还在打(True)时
          照常闸 —— 纯视觉模式下这道闸仍然有效.
        - `missing_slots > 0`(漏认空档)或 `_pre_meld` 未结(副露过渡帧):
          不判 —— 张数本来就不可信/正在变, 交给既有守卫(决策层 suspect /
          稳健出牌 / 双帧确认), 不在这里抢戏.
        - 张数合法: 清零连续计数, 放行.
        - 张数不合法: 计数 +1, 存证 + 告警(每次事件只报一次), 返回 False.
          **攒到 MELD_COUNT_RESYNC_FRAMES 还不符, 说明多半是副露记账本身
          错了**(比如对账超时把真碰当成了没点上), 按当前读数反推重同步
          —— 守卫宁可漏判不可锁死, 一个"账错的记忆"不能把这一手锁成
          一张不打. 反推只在张数本身合法(meld_count_of 有定义)且
          >= 4(防空行幻觉)时做, 推不出来的(6/9 张这类夹缝值)继续等.
        """
        if _proto_round_open_of(self) is False:
            self._count_mismatch_streak = 0
            return True
        if hand.missing_slots or self._pre_meld is not None:
            return True
        expected = expected_concealed_counts(self._meld_count)
        if hand.count in expected:
            self._count_mismatch_streak = 0
            return True
        self._count_mismatch_streak += 1
        self._log_repeat_once(
            "count_mismatch", f"{self._meld_count}:{hand.count}",
            f"  手牌 {hand.count} 张与副露 {self._meld_count} 组的理论张数"
            f"（{'/'.join(str(c) for c in sorted(expected))} 张）不符，"
            f"这一轮读数不可信，不出牌", level="warning")
        if self._count_mismatch_streak == 1:
            # 只有事件的第一帧留证: 后续帧画面相同, 存了也是刷盘.
            # 收局后不留证: 那时读数本来就是残影, 不是"感知坏了".
            if _proto_round_open_of(self) is not False:
                self._save_unknown_page(
                    frame, f"张数{hand.count}与副露{self._meld_count}组不符")
        if self._count_mismatch_streak >= tuning.MELD_COUNT_RESYNC_FRAMES:
            derivable = None
            if 4 <= hand.count <= 14:
                if hand.count % 3 == 2:
                    derivable = meld_count_of(hand.count)   # (14-c)//3
                elif hand.count % 3 == 1:
                    derivable = (13 - hand.count) // 3
            if derivable is not None and derivable != self._meld_count:
                self.log_warning(
                    f"  连续 {self._count_mismatch_streak} 轮张数与副露记忆"
                    f"不符，按当前读数重同步：副露 {self._meld_count} 组 -> "
                    f"{derivable} 组（张数 {hand.count}）")
                self._meld_count = derivable
            self._count_mismatch_streak = 0
        return False

    def _ensure_auto_toggles(self, cfg: dict, frame, has_yaku: bool) -> None:
        """把左上操作条的「和/(鸣)/(拔)」自动开关点亮(2026-10-07 用户提议).

        自动和了 + 自动拔北恒开. 「鸣」= **自动拒绝鸣牌**(吃碰杠一整类):
        **点着它雀魂就不画吃碰杠面板**, 决策层连"这一把要不要鸣"都没机会
        回答, 所以默认**不点**(`tuning.AUTO_DECLINE_MELD=False`, 2026-10-17
        真机拍板 —— 那一场协议给了 25 次机会、视觉一次没见到面板). 只有
        `AUTO_DECLINE_MELD=True` 才回到老行为: 有役牌对子(白/发/中/场风东)
        不点鸣让碰的机会弹出来, 没役对点鸣自动拒. 详见 tuning.auto_toggles_on.

        has_yaku 只在 AUTO_DECLINE_MELD=True 时才有意义, 那时由调用方判定并做
        **两帧共识**(just_opened 那帧的 z 牌单帧误读会造出幻影役对, 见主循环).
        开关状态靠颜色(开=亮绿 关=灰蓝, 见 vision.is_toggle_green 与
        tuning.AUTO_TOGGLE_ROWS 的实测数值), 幂等 —— 已开就不碰.

        安全措施: 「切」(自摸切)只读不点; 若发现它被误点亮(点相邻行手滑)
        立即点回去并**复读确认真的关掉了**(2026-10-11 实跑踩坑: 旧代码点
        一下就算数, 那一下落空后「切」开着自动摸切替人打完整局, 而说好的
        "下一手重试"因任务中途被停根本没发生); 每次点击后复读确认, 意图
        开关没点亮的下一手重试.
        """
        if self._toggles_ok:
            return
        rows = tuning.AUTO_TOGGLE_ROWS.get(self._players_key(cfg))
        if rows is None:
            return
        toggles = tuning.auto_toggles_on(has_yaku)
        h, w = frame.shape[:2]

        def rect(name: str) -> tuple[int, int, int, int]:
            x0, x1, y0, y1 = rows[name]
            return (int(x0 * w), int(y0 * h), int((x1 - x0) * w), int((y1 - y0) * h))

        # 「必须关掉」的开关: 目前只有「切」(自摸切) —— 它开着会自动打出我们摸到
        # 的牌. 「鸣」(自动拒绝鸣牌)不在这里: 用户 2026-10-17 明确"不需要主动关,
        # 开局就是不点就行"(雀魂的自动开关每手本来就复位回关), 所以对「鸣」只做
        # "不去点亮", 见 tuning.auto_toggles_on / auto_toggles_off.
        must_off = [(name, rect(name)) for name in tuning.auto_toggles_off()
                    if name in rows]
        for name, px in must_off:
            if is_toggle_green(frame, px):
                self.log_warning(f"  自动开关:「{name}」开着"
                                 f"(会替我们决定), 帮忙关闭")
                self._click_toggle_row(px, name)
                self.sleep(0.4)
                frame = self.next_frame()
        pending = [(name, rect(name)) for name in toggles
                   if name in rows and not is_toggle_green(frame, rect(name))]
        if not pending:
            self._toggles_ok = True
            return
        for name, px in pending:
            self._click_toggle_row(px, name)
            self.sleep(0.35)    # 连点之间留间隔: 点太快游戏偶尔吞点击/错行
        self.sleep(0.5)
        frame = self.next_frame()
        failed = [name for name, px in pending if not is_toggle_green(frame, px)]
        # 误点亮防护(2026-10-11 升级, 2026-10-17 扩到「鸣」): 点掉之后必须复读
        # 确认, 关不掉就再点(最多两轮); 意图开关没点亮的顺手补点. 意图全绿 +
        # 「必须关」全灰才算就绪, 否则不置 _toggles_ok, 下一手从头再来.
        for _ in range(2):
            still_on = [(n, px) for n, px in must_off if is_toggle_green(frame, px)]
            if not failed and not still_on:
                break
            for name, px in still_on:
                self.log_warning(f"  自动开关:「{name}」亮着, 点掉")
                self._click_toggle_row(px, name)
                self.sleep(0.5)
            for name, px in pending:
                if name in failed:
                    self.log_warning(f"  自动开关:「{name}」没点亮, 补点")
                    self._click_toggle_row(px, name)
                    self.sleep(0.5)
            self.sleep(0.3)
            frame = self.next_frame()
            failed = [name for name, px in pending
                      if not is_toggle_green(frame, px)]
        left_on = [n for n, px in must_off if is_toggle_green(frame, px)]
        if failed or left_on:
            msgs = []
            if failed:
                msgs.append(f"没点亮: {failed}")
            if left_on:
                msgs.append(f"关不掉: {left_on}"
                            f"（「切」会替你打牌; 「鸣」会让吃碰杠面板不出现）")
            self.log_warning(f"  自动开关{'; '.join(msgs)}"
                             f"（反复失败 = 点击没落在竖条字符上, "
                             f"请核对 tuning.AUTO_TOGGLE_ROWS 行位标定）")
        else:
            self._toggles_ok = True
            on = [n for n in toggles if n in rows]
            note = "自动和了" + ("/不鸣牌" if "鸣" in on
                                 else "/可鸣牌(面板会弹出来)")
            if "拔" in on:
                note += "/自动拔北"
            self.log_info(f"  自动开关已就绪: {'/'.join(on)} （{note}）")

    def _click_toggle_row(self, px: tuple[int, int, int, int], name: str) -> None:
        """点操作条上的一个自动开关行(行矩形 = (x, y, w, h) 像素)."""
        x, y, w, h = px
        self.log_info(f"    点击自动开关「{name}」")
        self.click(x=x + w // 2, y=y + h // 2)

    def _park_mouse(self, frame) -> None:
        """点完东西把光标沿拟人轨迹停到牌桌安全区(见 tuning.PARK_ZONE).

        光标趴在手牌上, 悬停的牌会被游戏顶起来 —— 实跑里正好挡住立直按钮
        的模板匹配; 光标箭头本身也可能被牌面模型读进下一轮的手牌读数.
        """
        (x0, x1), (y0, y1) = tuning.PARK_ZONE
        h, w = frame.shape[0], frame.shape[1]
        px = int(self._rng.uniform(x0, x1) * w)
        py = int(self._rng.uniform(y0, y1) * h)
        self.move(px, py)

    def _action_delay(self, decision) -> float:
        from src.mahjong.table import (
            ACTION_BABEI,
            ACTION_CONFIRM,
            ACTION_MELD,
            ACTION_RIICHI,
            ACTION_SKIP,
            ACTION_WIN,
        )
        if decision.action == ACTION_WIN:
            return timing.win_delay(self._rng)
        if decision.action == ACTION_RIICHI:
            return timing.riichi_delay(self._rng)
        if decision.action in (ACTION_MELD, ACTION_BABEI):
            return timing.meld_delay(self._rng)
        if decision.action in (ACTION_SKIP, ACTION_CONFIRM):
            return timing.idle_delay(self._rng)
        return timing.idle_delay(self._rng)

    def _prefer_non_aka_box(self, tile: str, hand, box):
        """同牌种里有赤宝牌在场时, 出牌**避开画着红牌的那个槽**.

        逻辑在模块级 `_prefer_non_aka_box_impl`(单测把 `_discard` 绑到假对象
        上跑, 假对象没有这些属性 —— 拿不到就原样返回, 不猜).
        """
        grab = getattr(self, "_grab", None)
        net = getattr(self, "_net", None)
        if not callable(grab) or net is None:
            return box
        return _prefer_non_aka_box_impl(net, grab, tile, hand, box)

    def _discard(self, decision, hand, cfg: dict, game: int, turns: int) -> bool:
        """点手牌出牌, 并确认手牌真的少了一张.

        雀魂可以设成"单击直接打出"或"选中后再确认"; 这里不猜设置, 而是看结果:
        点一下手牌没少就再点一下. 确认过一次之后记下来, 后面每轮直接点两下.

        第二次点**必须重新定位这张牌**: 第一次点完牌会被"举起来"(上移十几像素),
        原坐标已经落空了, 照着旧框再点一次只会点在空隙上.

        返回"这张牌确认打出去了没有". False 表示点了但手牌一张没少 —— 要么我们
        认错了"该我出牌", 要么雀魂压根没收到这次点击. 调用方靠连续 False 的
        次数发现冻屏 / 模态框(见 tuning.DISCARD_MISS_LIMIT); 核实被关掉或
        抓不到帧时一律返回 True, 不冤枉这一轮.
        """
        box = hand.box_of(decision.index)
        if box is None:
            self.log_warning(f"  第 {decision.index} 张牌没有框，跳过")
            return True                     # 没点, 不算"点了没反应"
        prefer = getattr(self, "_prefer_non_aka_box", None)
        if callable(prefer):                # 假对象上没有这个方法(见 impl 说明)
            box = prefer(decision.tile, hand, box)

        before = hand.count
        point = self._random_point_in_box(
            box, tuning.SAFE_MARGIN_RATIO, tuning.CLICK_RANGE
        )
        self._ensure_foreground()
        self.click(x=point[0], y=point[1])
        self.log_info(f"  出牌 {decision.tile} -> ({point[0]}, {point[1]})")
        self._record({"event": "click", "game": game, "turn": turns,
                      "action": ACTION_DISCARD, "tile": decision.tile,
                      "index": decision.index, "point": [point[0], point[1]],
                      "hand_before": before})

        if not tuning.VERIFY_DISCARD:
            return True

        if self._need_confirm_click:      # 上一轮已经确认过是"选中再确认"
            self.sleep(tuning.DISCARD_CONFIRM_GAP)
            return self._confirm_discard(decision, hand, cfg, game, turns, before)

        self.sleep(tuning.DISCARD_VERIFY_DELAY)
        frame_after, after = self._read_hand_retry(before - 1)
        if after is None:
            return True                     # 抓不到帧, 不当成"点了没反应"
        self._record({"event": "discard_verify", "game": game, "turn": turns,
                      "before": before, "after": after.count,
                      "tiles_after": after.tiles})
        if after.count < before:
            # **掉超过一张就是识别错了**(2026-10-06 实跑 13 轮踩到):
            # 一次出牌只可能少一张, 掉 2~3 张说明我们认出的手牌里有多余的牌
            # -- 牌面模型把牌河/副露误认成了手牌. 典型现场(t185): 系统认成
            # 14 张(多认了张 7p), 照这个错手牌算出"牌效最优"的 9p 打出去,
            # 而真手牌里 9p 正是搭 789p 的关键牌, 打完直接退向.
            # 这时候必须留证: 光记日志没用, 得看到画面才能查是哪张认错.
            if before - after.count > 1:
                self.log_error(
                    f"  出牌后手牌从 {before} 掉到 {after.count}"
                    f"（掉了 {before - after.count} 张），说明这轮认错了手牌。"
                    f"认定打出的是 {decision.tile}，实际可能不是这张。"
                    f"已截图留证，下一轮会重新认。")
                # 留证帧必须是**判定当帧**(上面那张复读后的帧), 不能在这里
                # 重新 _grab() —— 抓到的已经是下一轮的画面了, 和日志里的
                # 张数对不上(2026-10-07 排查时被这样带偏过一轮).
                self._save_unknown_page(
                    frame_after,
                    f"出牌后张数异常_{before}掉到{after.count}_认错手牌")
            self._discard_misses = 0        # 牌真的打出去了
            return True

        self.log_info("  手牌没变化，再点一次（可能是'选中再确认'出牌）")
        self._need_confirm_click = True
        return self._confirm_discard(decision, after, cfg, game, turns, before)

    def _confirm_discard(self, decision, hand, cfg: dict,
                         game: int, turns: int, before: int) -> bool:
        """补第二下: 用最新一帧重新定位要打的那张牌再点.

        返回"这次终于打出去了没有". 失败时把 `_discard_misses` 加一 ——
        连续失败就是"点了没反应"的唯一现场证据(见 tuning.DISCARD_MISS_LIMIT).
        """
        index = pick_tile_index(hand.tiles, decision.tile, self._rng,
                                drawn_tile=getattr(self, "_drawn_tile", None))
        box = hand.box_of(index) if index is not None else None
        if box is None:
            box = hand.box_of(decision.index)
        if box is None:
            self.log_warning(f"  '{decision.tile}' 在最新画面里找不到了，不补点")
            return True                     # 没点, 不算"点了没反应"

        point = self._random_point_in_box(
            box, tuning.SAFE_MARGIN_RATIO, tuning.CLICK_RANGE
        )
        self.click(x=point[0], y=point[1])
        self.sleep(tuning.DISCARD_CONFIRM_DELAY)
        # 同样要防重排动画残帧: 这里读成 before 张会被判成"没打出去"(miss),
        # 而残帧的成因就是那张牌还叠在手上 —— 于是连点两下也白点.
        _frame, again = self._read_hand_retry(before - 1)
        self._record({"event": "discard_verify", "game": game, "turn": turns,
                      "before": before,
                      "after": again.count if again else None,
                      "confirm_click": True,
                      "tiles_after": again.tiles if again else []})
        if again is None:
            return True                     # 抓不到帧, 不当成"点了没反应"
        if again.count >= before:
            self._discard_misses += 1
            self.log_warning(
                f"  点了两下还是 {again.count} 张，这张牌可能没打出去"
                f"（连续 {self._discard_misses} 次）")
            return False
        self._discard_misses = 0
        return True

    # ---------------------------------------------------------------
    # 感知
    # ---------------------------------------------------------------

    def _ensure_vision(self) -> GameVision:
        if self._vision is None:
            self._vision = GameVision()
            frame = self._grab()
            self._vision.warmup(frame.shape if frame is not None else (720, 1280, 3))
            self.log_info("YOLO 已预热")
        return self._vision

    def _save_low_conf_frame(self, frame, hand) -> None:
        """护栏(2026-10-10): 读数里有 conf<LOW_CONF_SAVE_CONF 的槽位就留证帧.

        为什么必须有: 误识是静默的(见 detection.low_confidence 的长注释),
        MIN_TILE_CONF 只保决策不保证据 —— 1s(一条)被读成 8p 那次摸牌位
        conf 0.80, 决策照打不误, 事后只能靠用户拿肉眼对牌谱才发现. 当时
        只要有这一张帧, "摸进来的根本不是 8p"一眼可辨.

        存进 data/debug/(frame_lowconf_ 前缀): tools/label_tiles 的扫描队列
        自动捡走 frame_*.png, 误识帧直接变成下一轮微调的标注素材.

        去重: 签名 = 低置信槽位的 (下标, 置信) 序列, 记在 _repeat_log 里 ——
        和告警复读一个机制, 换手(发牌重臂)自动清空, 同一手里同一处低置信
        只存第一帧; 签名一变(新槽位低了/置信变了)立刻再存, 新现场不吞.
        限额: 每局 LOW_CONF_SAVE_MAX_PER_GAME 张, 防止极端局面刷盘.
        """
        if frame is None or hand is None or not hand.boxes:
            return
        # 收局后静音: 这一手已经结束(协议 round_open=False), 画面是摊牌/顺位
        # 面板, 读出来的 7~11 张碎片不是"低置信误识现场" —— 存了只是刷盘
        # (2026-10-18 13:50:03 那张 frame_lowconf 就是结算残影).
        if _proto_round_open_of(self) is False:
            return
        # 发牌/整理动画静音: 牌还没发齐(deal watch 没确认)或刚发齐的这几秒
        # 里牌在滑动, 低置信是动画常态不是误识现场, 存帧纯刷盘(2026-10-17
        # 16:48 局开局连存数张). 静音只影响存帧, 读数/投票照常.
        deal = getattr(self, "_deal", None)
        if deal is not None:
            if not getattr(deal, "opened", False):
                return
            opened_at = float(getattr(deal, "opened_at", 0.0) or 0.0)
            if opened_at and (time.time() - opened_at
                              < tuning.LOWCONF_MUTE_AFTER_DEAL):
                return
        # 摸牌渐入动画静音(与 _draw_settle_pending 同一个窗, 见
        # tuning.NET_DRAW_SETTLE): 万一某轮还是读到了(临期跳过等待等),
        # 动画期的低置信也不留证 —— 它是动画常态不是误识现场(2026-10-17
        # 4 张 lowconf 帧全是它, 用户逐帧判定).
        try:
            drawn_secs = (self._net.seconds_since_self_draw()
                          if self._net is not None else None)
        except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
            drawn_secs = None
        if (drawn_secs is not None
                and drawn_secs < tuning.NET_DRAW_SETTLE):
            return
        low = low_confidence(hand.boxes, tuning.LOW_CONF_SAVE_CONF)
        if not low:
            return
        if self._low_conf_saves >= tuning.LOW_CONF_SAVE_MAX_PER_GAME:
            return
        sig = tuple((i, round(hand.boxes[i].conf, 2)) for i in low)
        if self._repeat_log.get("low_conf_save") == sig:
            return
        self._repeat_log["low_conf_save"] = sig
        self._low_conf_saves += 1
        try:
            DEBUG_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%m%d_%H%M%S")
            path = DEBUG_ARTIFACTS_DIR / f"frame_lowconf_{ts}.png"
            imwrite_unicode(path, frame)
            detail = ", ".join(
                f"#{i} {hand.tiles[i]} conf={hand.boxes[i].conf:.2f}"
                for i in low if i < len(hand.tiles))
            self.log_warning(f"  低置信存帧: {path.name}（{detail}）")
        except Exception as e:  # noqa: BLE001
            self.log_warning(f"  低置信存帧失败: {e}")

    def _read_hand_frame(self):
        """抓一帧并读手牌, 返回 `(frame, HandReading)`. 抓不到帧是 `(None, None)`.

        调用方需要**判定当帧**时用它(留证帧必须是决策/判定用的那一张, 见
        `_record_turn`); 主循环的读手牌走 `_read_hand(vision, frame, cfg)`.
        """
        frame = self._grab()
        if frame is None:
            return None, None
        return frame, self._strip_kan_setaside(
            self._ensure_vision().read_hand(frame))

    def _strip_kan_setaside(self, hand):
        """这一手里我们自己杠过的话, 把挤进手牌行右端的杠牌剥掉.

        暗杠的 4 张牌竖着摆在行最右端, 与暗牌隔 ~4.8 倍牌宽的大空档
        (detection.SETASIDE_GAP_RATIO 的实测依据), 牌面模型分不清杠牌和
        暗牌, 会一起读进来 —— 实测 2026-10-08 20:47 局: 暗杠后读成
        11 张 + 空档 6, 张数/间距/出牌定位全被带歪, 还把 count+missing=17
        灌进摊牌幻觉判据引发闸门误重臂, 死锁了整整一手(修复见
        DealWatch.see 的 melded 参数).

        剥离只发生在 `self._kan_pending`(本手我们点过杠按钮)时: 平时行里
        没有杠牌, 这层逻辑完全不参与. 剥完把剩余段重新数一遍间距空档 ——
        杠牌与暗牌之间的大空档正是 missing_slots 膨胀的来源, 剥掉后剩余
        段通常是连续的.
        """
        if not self._kan_pending or hand is None or len(hand.boxes) < 2:
            return hand
        main, tail = split_setaside(hand.boxes)
        if not tail:
            return hand
        shrunk = replace(hand,
                         boxes=main,
                         tiles=[b.label for b in main],
                         missing_slots=missing_slots(main))
        self.log_info(f"  剥掉行尾杠牌 {len(tail)} 张（本手我们杠过），"
                      f"手牌按 {shrunk.count} 张算")
        return shrunk

    def _read_hand_retry(self, expect: int):
        """读手牌; 第一次读数与 `expect` 不符**或张数不可信**时复读一次.

        **为什么需要**(2026-10-07 实跑复盘): 出牌后 / 补第二下之后立刻读的那一帧,
        29% 的时候不是稳定态 —— 28 次出牌校验里 8 次读数与本局的稳定值不符,
        其中 **5 次整整少认一张牌**(14 掉到 12), 缺的正是**刚摸进来、正在插回
        手牌位的那张**: 0.6s 的 `DISCARD_VERIFY_DELAY` 撞在了手牌重排动画上。

        后果不只是刷 5 条 ERROR + 5 张待标注图 —— `before - after == 1` 正是
        "牌打出去了"的判据, 少认一张会把它算成 2, 于是**明明打出去了却报
        "认错手牌"**; 反方向也能把"没打出去"算成打出去了(残帧里那张牌还叠在
        手上, 而漏认恰好把 14 张削成 13), 那会把出牌模式误判成"选中再确认",
        白点第二下还污染这个标志位。所以触发复读的条件是两条:
        `count != expect`(**张数不等于预期**)或 `not reliable`(**有漏认空档**,
        见 HandReading.missing_slots)。

        两张里选哪张: **可靠的那张优先**(残帧的病是漏认, 稳定态总能读全);
        两张都可靠或都不可靠时, 取离 `expect` 更近的。复读失败(抓不到帧)就
        沿用第一次, **绝不**因为复读失败就改动判定。

        代价: "选中再确认"那条路(只在一局第一次发生时走)多花一个
        `DISCARD_VERIFY_DELAY`; 正常出牌且读数可靠时不会触发复读。
        """
        frame, hand = self._read_hand_frame()
        if hand is None or (hand.count == expect and hand.reliable):
            return frame, hand
        self.sleep(tuning.DISCARD_VERIFY_DELAY)
        frame2, hand2 = self._read_hand_frame()
        if hand2 is None:
            return frame, hand
        if hand2.reliable != hand.reliable:
            if hand2.reliable:
                frame, hand = frame2, hand2
        elif abs(hand2.count - expect) < abs(hand.count - expect):
            frame, hand = frame2, hand2
        self.log_info(
            f"  出牌校验第一帧读数可疑（期望 {expect} 张），复读后按 "
            f"{hand.count} 张算")
        return frame, hand

    def _wait_leave_ready_page(self, spec, cfg: dict) -> None:
        """点完"开始"后等画面离开准备页, 否则循环第一轮就会误判成"这局结束了".

        注意判据是"还在准备页"而不是"开始按钮还在": 点了开始之后按钮会
        消失/变灰, 用它判反而更脆(见 _at_ready_page 的说明).
        """
        deadline = time.time() + tuning.PAGE_WAIT_TIMEOUT
        while time.time() < deadline:
            if self._should_yield():
                self.log_info("收到停止指令, 不再等进对局")
                return
            frame = self._grab()
            if frame is None:
                self.sleep(tuning.PAGE_POLL_INTERVAL)
                continue
            if not self._at_ready_page(spec, cfg, frame):
                self.log_info("已进入对局")
                return
            self.sleep(tuning.PAGE_POLL_INTERVAL)
        self.log_warning("点了开始但画面还停在准备页，继续尝试")

    # ---------------------------------------------------------------
    # 记录
    # ---------------------------------------------------------------

    def _open_record(self) -> None:
        DEBUG_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._record_path = DEBUG_ARTIFACTS_DIR / f"game_{ts}.jsonl"
        self.log_info(f"对局记录: {self._record_path}")

    def _record(self, payload: dict) -> None:
        if self._record_path is None:
            return
        payload["t"] = round(time.time(), 3)
        try:
            with open(self._record_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001 - 记录失败不能影响打牌
            self.log_warning(f"写记录失败: {e}")

    def _record_turn(self, game: int, turns: int, hand, buttons,
                     decision, frame=None, net: str = "",
                     ev: dict | None = None,
                     trust: str = "") -> None:
        """记一轮, 并把**决策当帧**存到 data/debug/.

        `frame` 必须是这一轮真正读过的**那一帧**, 不能省。原来这里不接 frame,
        存帧时自己重新 `_grab()` 一张 —— 抓到的已经是下一次循环的画面了, 于是
        `frame_g1_tN.png` 与第 N 轮的决策**不是同一帧**。它不一定差很远, 但在
        "手牌正在逐张发出来"的那几秒里能差 5 张牌, 2026-10-07 排查时直接被带偏
        (拿留证帧重跑模型, 结果和日志里的 hand 对不上, 白查了一轮)。

        `ev` 是影子档的出牌 EV 估值(见 _ev_shadow): 同一行里既有我们实际打的
        那张, 也有引擎排的前三名, 所以"我们打的那张排第几、少多少期望"事后
        直接从 jsonl 算得出来, 不用再回放协议 spool.
        """
        self._record({
            "event": "turn", "game": game, "turn": turns,
            "hand": hand.tiles,
            "hand_conf": [round(b.conf, 3) for b in hand.boxes],
            "hand_boxes": [[round(b.x1), round(b.y1), round(b.x2), round(b.y2)]
                           for b in hand.boxes],
            "hand_region": list(hand.region),
            # 间距判出来的漏认张数. 记下来才能在事后一眼看出"那次是不是
            # 认漏了" —— 2026-10-07 排查静默空转时, 正因为 jsonl 里没有这个
            # 字段, 只能靠手工量框间距反推.
            "missing_slots": hand.missing_slots,
            "buttons": {name: round(boxes[0].conf, 3)
                        for name, boxes in buttons.items()},
            "button_centers": {name: list(self._box_center(boxes[0]))
                               for name, boxes in buttons.items()},
            "action": decision.action,
            "button": decision.button,
            "tile": decision.tile,
            "index": decision.index,
            "reason": decision.reason,
            # 网络读牌对账摘要(影子/接管档都有; 关闭档为空串). 事后能拿
            # 它逐轮对比"协议看到的 vs 视觉读到的", 判断接管档能不能开.
            "net": net,
            # 信任档三态(trusted/mismatch/off). 没有它, "那轮双帧为什么没跳过"
            # 只能靠猜 —— 2026-10-17 16:21 首个出牌轮多花 0.5s 走了双帧, jsonl
            # 里却查不到当时 trust 是什么.
            "trust": trust or None,
            # 出牌 EV 估值(不在"该我打牌"的时刻为 null).
            "ev": ev,
        })
        # 每轮存帧(save_frames)已删(2026-10-17): 正常帧没有诊断价值, 留证
        # 只剩 lowconf/gate/待标注 三类自动帧. 决策当帧的完整性由上面那行
        # jsonl 里的 hand_boxes 保证.

    def _save_frame(self, tag: str, frame=None) -> None:
        """存一帧留证. `frame` 给了就用它(决策当帧), 没给才去抓新的."""
        if frame is None:
            frame = self._grab()
        if frame is None:
            return
        ts = datetime.now().strftime("%H%M%S")
        out = DEBUG_ARTIFACTS_DIR / f"frame_{tag}_{ts}.png"
        imwrite_unicode(out, frame)
