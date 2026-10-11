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

配置: configs/PlayOneGameTask.json。界面上只有四个用户旋钮:
    players / length         房型(两个下拉框, 组合出房间规格)
    max_games                连打几局(0 = 只建房不开始)
    save_frames              每轮是否存帧到 data/debug/
    robust_discard           手牌有认不准的牌时是否走稳健出牌
其余阈值与延迟全部是 `src/tasks/tuning.py` 里的代码常量.
"""

import json
import random
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from src.annotations import missing_categories
from src.detection import (MIN_TILE_CONF, low_confidence, missing_slots,
                           split_bright_dim, split_setaside, tile_brightness,
                           vote_labels)
from src.detector import GameVision, with_voted_labels
from src.imageio import imwrite_unicode
from src.instrumentation import bench
from src.mahjong import timing
from src.mahjong.table import (ACTION_BABEI, ACTION_DISCARD, ACTION_MELD,
                               ACTION_RIICHI, ACTION_WAIT, ACTION_WIN,
                               BUTTON_GANG, TABLE_GAVE_UP, TABLE_IGNORING,
                               TABLE_WAITING,
                               DealWatch, TableWatch, action_confirmed,
                               decide_turn, expected_concealed_counts,
                               has_yaku_pair, meld_count_of,
                               meld_taken_kinds, pick_tile_index)
from src.paths import DEBUG_ARTIFACTS_DIR
from src.rooms import RoomSpec, ready_marker, required_categories, spec_by_key, start_step
from src.tasks import tuning
from src.tasks.CreateFriendsRoomTask import (CreateFriendsRoomTask,
                                             TaskStopped,
                                             _DEFAULT_CONFIG as _BASE_CONFIG)
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
    # 每轮存帧到 data/debug/. 为什么值得常开见 tuning.SAVE_FRAMES 的注释.
    "save_frames": tuning.SAVE_FRAMES,
    # 手牌里有认不准的牌时别拿"这手牌算出来的最优"直接打. 背景与实跑案例
    # 见 tuning.ROBUST_DISCARD 的注释.
    "robust_discard": tuning.ROBUST_DISCARD,
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

        self._open_record()
        vision = self._ensure_vision()

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
        while time.time() < deadline:
            if self._should_yield():
                raise TaskStopped("等待匹配期间收到停止指令")
            frame = self._grab()
            for screen, keyword in candidates:
                if self._locate(screen, keyword, cfg, frame,
                                strict=True, verbose=False) is not None:
                    self.log_info(
                        f"    匹配成功（检测到 [{screen}] '{keyword}'）")
                    return
            now = time.time()
            if now >= next_status:
                self.log_info(f"    还在匹配（已等 {now - (deadline - tuning.MATCHMAKING_TIMEOUT):.0f}s，"
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

    def _play_one_game(self, cfg: dict, game: int) -> bool:
        """打完一局, 回到准备页返回 True; 超时/异常返回 False."""
        spec = self._resolve_spec(cfg)
        vision = self._ensure_vision()
        deadline = time.time() + tuning.GAME_TIMEOUT
        # 开局宽限期: 点开始之后画面还要过加载 / 发牌动画, 这段时间手牌是
        # 逐张出现的(实测 0 张 -> 3 张 -> 12 张 -> 13 张, 前后约 10 秒).
        # 这属于正常, 不能算"认不出来", 也不该存待标注截图.
        # 卡死检测: 连续多少轮"感知结果完全一样且没有任何动作". 对局结束进结算页
        # 之后画面是静止的, 而牌河会被误识成"手牌"(实测 10 张, 刚好是 1 副露的
        # 合法张数), 于是每一轮都判成"在等别人", 一直空转到 game_timeout.
        # 判据见 src/vision.py 的 StuckDetector(为什么不用像素差也在那里).
        stuck = StuckDetector(tuning.STUCK_AFTER_TURNS)
        unknown_streak = 0
        turns = 0
        # 立直之后必须打宣言牌. 点下立直那一刻置位, 真的少了一张才清掉.
        pending_riichi = False
        # 立直**已生效**(宣言牌打出、游戏确认了立直): 之后雀魂每回合自动
        # 摸切, 我们的任何"选牌出牌"都是无效点击 —— 2026-10-07 19:16 实跑,
        # 宣言后连打 2z/1m/7z/9s, 游戏里手牌全程冻结(turn 230~276 的 13 张
        # 读数一字不差), 用户看到的就是"立直后一直尝试出牌". 上一版没有
        # 这个状态: 宣言牌一打出, 任务就把立直忘了, 回到正常打牌逻辑.
        riichi_active = False
        # 连认几轮认不出宣言牌(立直后牌面模型偶尔认不到那个更矮的渲染).
        # 认不出高亮牌的连续轮数(由 _bright_indices 自己累加), 超上限就
        # 降级为按普通出牌处理, 免得无限空转到 game_timeout.
        self._riichi_misses = 0
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
        # 多帧投票的缓存: 连续几帧各自 read_row 的结果, 用来压低置信度误识别.
        vote_buffer: list[list] = []
        # "同一手牌里重复出现的同一条告警只报第一次"的去重表: 键 -> 上次的签名.
        # 每重臂一手(手牌区清空 -> 重新发牌)清空, 见 _log_repeat_once.
        self._repeat_log: dict[str, str] = {}

        while time.time() < deadline:
            # bench 挂钩(默认关闭): 记录主循环周期 + 固定时长到点落盘.
            bench.tick_loop()
            # 用户点了停止就立刻收工, 别再抓帧/点击(见 _should_yield 的说明:
            # 框架的paused 标志不会自动打断任务代码, 必须显式问).
            if self._should_yield():
                self.log_info("收到停止指令, 中止本局")
                return False
            frame = self._grab()
            if frame is None:
                self.sleep(tuning.TURN_POLL_INTERVAL)
                continue

            # 1) 回到准备页 = 这局结束了
            #    判据用静态特征而不是开始按钮(见 _at_ready_page 的说明)
            if self._at_ready_page(spec, cfg, frame):
                self.log_info(f"第 {game} 局结束（回到准备页），共 {turns} 轮")
                return True

            # 1.5) 结算页 = 对局已经打完, 点"确认"回准备页.
            #放在准备页判定之后, 读手牌之前: 结算页上根本没有手牌, 让牌面
            # 模型去读只会读到牌河(实测 10 张), 决策全是"在等别人".
            if self._handle_result_page(frame, cfg, spec, game, turns):
                return True

            # 1.55) 终局奖励页(偶发弹窗, 2026-10-11 用户报): 盖住整块画面且
            #       没有可匹配的按钮, 不点掉会一直卡着 —— 见
            #       _handle_reward_page 的说明. 多数局不会命中, 一次
            #       strict 定位(静态素材)的代价可以忽略.
            if self._handle_reward_page(frame, cfg, game):
                self.sleep(tuning.TURN_POLL_INTERVAL)
                continue

            # 1.7) 牌桌还没出来: 加载 / 对局开始页(写「胜负」那一页) / 和了过场.
            #      这段时间手牌区里根本没有手牌, 让牌面模型去读只会得到
            #      "只认出 0~3 张" + 一堆"按稳健出牌算"的噪音(实跑开局刷了
            #      20 轮约 10 秒), 而且那些张数还可能撞上"合法张数"把人骗到
            #      发牌动画里去出牌. 所以这里**整轮不感知也不决策** —— 判据
            #      用静态的牌桌特征, 见 TABLE_SCREEN 的实测数值.
            # bench 挂钩(默认关闭): 门控结果要留给下面的分支用, 先存局部.
            gate = self._table_gate(cfg, frame)
            if gate == TABLE_WAITING:
                self.sleep(tuning.TURN_POLL_INTERVAL)
                continue

            # 审计报告 G 节修改②(2026-10-10 实施): 牌桌特征失灵超过
            # TABLE_WAIT_LIMIT 之后, **不再读手牌**。旧行为是"按旧逻辑裸读
            # 继续打", 但连牌桌特征都认不出来时, 手牌区读到的只会是牌河 /
            # 过场的幻觉牌(实测结算页牌河被读成 10 张"合法手牌"空转到超时,
            # 就是同一类问题), 决策没有任何可信输入。这里整轮跳过感知与
            # 决策, 靠外层 deadline(GAME_TIMEOUT)兜底不锁死; 上方的页面级
            # 门控(准备页 / 结算页 / 一手结束)不受影响, 过场自己能走完。
            # 计数进 bench(默认关), 量化这个兜底实际触发频率。
            if gate in (TABLE_GAVE_UP, TABLE_IGNORING):
                bench.count_state("table_gave_up_skip")
                self.sleep(tuning.TURN_POLL_INTERVAL)
                continue

            # bench 挂钩(默认关闭): 给接下来的手牌读数打阶段标签.
            # "出牌阶段" = 发牌闸门已开且不在和了封锁; 除此之外的
            # (发牌中 / 和了封锁)都是非出牌阶段的读手牌。
            # (gave_up 不在此列: 上面的分支已经跳过, 放弃后不再读手牌。)
            bench.set_phase(
                "active" if (self._deal.opened and not self._deal.settling)
                else "settling" if self._deal.settling else "deal_pending")

            # 1.75) 一手结束的顺位结算画面(和了/流局后): 点「确认」过场,
            #       这一轮不读手牌 —— 这画面上手牌行被面板遮着, 读数全是
            #       "9~11 张 + 空档"的假警报(见 HAND_RESULT_SCREEN 的说明).
            if self._handle_hand_result(frame, cfg, game):
                self.sleep(tuning.TURN_POLL_INTERVAL)
                continue

            hand = self._read_hand_voted(vision, frame, cfg, vote_buffer)
            buttons = vision.read_buttons(frame)
            button_conf = {name: boxes[0].conf for name, boxes in buttons.items()}

            # 立直选牌态: 点下立直后雀魂让**能打的牌高亮, 其余变暗**, 此时
            # 点一张高亮牌即完成立直宣言(用户指出, 见 log t76~t83 复核).
            # 我早先以为"雀魂指定了宣言牌并特殊渲染", 那是错的 -- 实跑里
            # t78~t82 点的都是变暗的牌, 雀魂不响应, 手牌数一直是 14.
            # 所以这里算的是"哪些牌高亮", 牌效只在这些牌里挑最优.
            allowed = (self._bright_indices(frame, hand) if pending_riichi
                       else None)

            # 低置信槽位: 牌面模型认不准的牌. 这些槽位会喂给 decide_turn 走
            # **稳健出牌** -- 枚举它们可能的真实取值, 挑一张怎么认都不踩坑的
            # 打(见 src/mahjong/brain.py::robust_discard).
            #
            # 这里曾经试过"有低置信就整轮不出牌, 等下一帧认准", 数据上已证伪
            # 并删掉: 位置 4 在 turn 176~185 连续 10 轮输出同一个错类别,
            # conf 稳定在 0.64~0.67, 等多久都不恢复 -- 实测那样会有 73% 的
            # 时间在空等, 节奏全拖垮了. 而误识牌从不被直接点(6 个已证实误识
            # 里只有 1 个点了低置信那张), 真正的问题是它污染了全局判断.
            blurry = (low_confidence(hand.boxes, float(blur))
                      if (blur := tuning.MIN_TILE_CONF) else None)
            if blurry:
                self._log_repeat_once(
                    "blurry",
                    "|".join(f"{i}:{hand.boxes[i].label}" for i in blurry),
                    f"  槽位 {blurry} 置信度偏低"
                    f"（{', '.join(f'{hand.boxes[i].label} {hand.boxes[i].conf:.2f}' for i in blurry)}），"
                    f"这一轮按稳健出牌算")
            # 刚摸的牌(最右槽)低置信 -> 置粘性标志(打出前不清): 听牌摸切
            # 守卫的输入, 见 table.decide_turn 第 8.0.5 步. count % 3 == 2
            # = 这是"摸牌后"的张数(14/11/8/5/2), 最右槽就是刚摸的那张.
            if (hand.count % 3 == 2 and blurry
                    and len(hand.tiles) - 1 in blurry):
                self._drawn_blurry = True
            # 按间距看出来的漏认: 和置信度无关, 张数可能整个是错的.
            # 必须一路传到决策层, 否则它会拿一个巧合合法的张数(10)去判
            # "是不是轮到我", 得出"在等别人"然后静默空转(见 table.decide_turn).
            if hand.missing_slots:
                self._log_repeat_once(
                    "missing",
                    f"{hand.missing_slots}:{hand.count}",
                    f"  手牌行里有约 {hand.missing_slots} 张牌的空档（只认出 "
                    f"{hand.count} 张），张数不可信",
                    level="warning")

            # 副露记账: 刚点过碰/杠且读数已稳定(张数比点击前至少少 2 --
            # 碰拿走一对, 杠只会更多), 从差集反推副露牌种(见
            # table.meld_taken_kinds). 这是决策层加杠支持判据的输入.
            if (self._pre_meld is not None
                    and time.time() - self._pre_meld_at
                    > tuning.PRE_MELD_TIMEOUT):
                # 对账超时: 点击后从没数出"少 2 张以上" —— 加杠(只挪 1 张,
                # 不增加副露组数)或按钮没点上. 放弃对账, 不然 _pre_meld 会
                # 挂到这一手结束, 把张数硬校验和后续对账一起堵死.
                self._log_repeat_once(
                    "pre_meld_expire", f"{len(self._pre_meld)}",
                    "  副露对账超时（可能是加杠或按钮没点上），放弃本次对账")
                self._pre_meld = None
            if (self._pre_meld is not None and not hand.missing_slots
                    and len(hand.tiles) <= len(self._pre_meld) - 2):
                for t in meld_taken_kinds(self._pre_meld, hand.tiles):
                    if self._melded_tiles.count(t) < 3:
                        self._melded_tiles += [t] * 3
                self._pre_meld = None
                # 数出了"少 2 张以上" = 真的鸣进一组, 组数 +1(纯按手数,
                # 加杠/没点上到不了这里). max 防任何一边多记.
                self._meld_count = max(self._meld_count + 1,
                                       len(self._melded_tiles) // 3)

            decision = decide_turn(
                hand.tiles, button_conf,
                player_count=spec.player_count,
                meld_policy=tuning.MELD_POLICY,
                allow_riichi=tuning.ALLOW_RIICHI,
                meld_min_conf=tuning.MELD_BUTTON_MIN_CONF,
                babei_policy=tuning.BABEI_POLICY,
                allowed_indices=allowed,
                blurry_indices=blurry,
                robust_discard_enabled=cfg["robust_discard"],
                missing_slots=hand.missing_slots,
                button_centers={name: self._box_center(boxes[0])
                                for name, boxes in buttons.items()},
                blurry_draw=self._drawn_blurry,
                melded=self._melded_tiles,
                rng=self._rng,
            )
            turns += 1
            # 把**这一帧**一起传进去存证, 别让 _record_turn 自己再抓一张
            # (抓到的已经是下一轮的画面, 留证帧会和日志里的 hand 对不上).
            self._record_turn(game, turns, hand, buttons, decision, frame)

            # 1.8) 这一手到底还在不在打? 两种"不在打"都要拦住:
            #
            #   a) 发牌还没结束. 发牌动画期间手牌逐张出现, 中间那些张数会撞上
            #      "合法张数"(8 = 2 副露的暗牌数), 决策层以为轮到自己, 照着一手
            #      没发完的牌算出出牌 —— 实跑里几乎每一局的**第一手**都是这么
            #      打出去的(见 table.deal_is_done).
            #   b) 上一手已经结束. 和了之后的摊牌动画里, 我们自己的和了牌是
            #      **正面朝上**摆着的, 读出来是一手干净的 13~14 张, 只看张数
            #      完全像正常打牌 —— 实跑 t190/t191/t195 就在摊牌动画里打了
            #      3 次牌.
            #
            # 两个 bug 的根因是同一个: 把"一手牌的状态"当成了"整场任务的状态".
            # 一次 _play_game 跑的是**整场**(东风局/半庄, 中间会连着开好几手),
            # 所以闸门必须按"手"重臂 —— 详细论证与实测数值见 table.DealWatch.
            state = self._deal.see(hand.count, hand.missing_slots, time.time(),
                                   melded=self._kan_pending)
            if state.just_opened:
                self._hand_seq += 1
                self.log_info(f"  手牌发齐了（{hand.count} 张），开始按正常逻辑打"
                              f"—— 第 {self._hand_seq} 手开局")
                # 开局记录(2026-10-11 用户提议): jsonl 里显式记每一手的起点,
                # 配合 hand_end 就能离线对齐"每一手的张数/副露数/决策"时间线.
                self._record({"event": "hand_start", "game": game,
                              "seq": self._hand_seq, "count": hand.count,
                              "melds": self._meld_count})
                # 双保险归零: 重臂已经清过一次, 这里再清 —— 发齐 13 张的
                # 一手副露必然为 0(副露得等别人先打牌), 万一重臂边沿被
                # 加速过场跳掉, 这道闸保证副露记账不会跨手污染.
                self._meld_count = 0
                self._count_mismatch_streak = 0
                # 新一手的第一件事: 把左上「和/鸣/(拔)」自动开关点亮(幂等,
                # 已确认过就只是三块小区域各读一次颜色). 「鸣」按本手手牌
                # 动态决定: 有役牌对子就不点(让碰机会活着), 没有就点(自动拒).
                #
                # 役对判定**不能只看一帧**(22:57 局实锤): just_opened 那帧
                # 把 7z 误读成 6z, 幻影"白板对"让整手没点「鸣」—— 鸣牌机会
                # 于是全靠我们手动应答, 一旦被双标签守卫卡住就被托管吃牌
                # (用户报的"没点鸣, 还在吃牌"). 这里再读一帧, **两帧都见
                # 役对**才让「鸣」保持关闭: 单帧幻影的代价只是这手少碰一次
                # (无实损), 而错误地不开「鸣」的代价是被托管接管应答.
                bench.set_phase("active")  # bench 挂钩: 发齐了, 属出牌阶段
                _, hand2 = self._read_hand_frame()
                yaku1 = has_yaku_pair(hand.tiles, hand.missing_slots,
                                      tuning.ROUND_WIND)
                yaku2 = (has_yaku_pair(hand2.tiles, hand2.missing_slots,
                                       tuning.ROUND_WIND)
                         if hand2 is not None else yaku1)
                self._ensure_auto_toggles(cfg, frame, yaku1 and yaku2)
            if state.rearmed:
                self.log_info("  上一手已结束（手牌区清空），发牌闸门重新上膛"
                              f"—— 第 {self._hand_seq} 手结束，本手副露 "
                              f"{self._meld_count} 组")
                self._record({"event": "hand_end", "game": game,
                              "seq": self._hand_seq,
                              "melds": self._meld_count})
                # 换了一手牌, 上一手的"这条已经报过"不该继续生效.
                self._repeat_log.clear()
                # 雀魂的自动开关**每一手都重置回关**(实跑 2026-10-08: 一局 6 手
                # 只在第 1 手点了「和/鸣」, 后面 5 手全是关的) —— 所以换手必须
                # 复位, 让 just_opened 时的 _ensure_auto_toggles 真的去点.
                self._toggles_ok = False
                # 新的一手行里没有杠牌(杠牌只属于一手), 剥离逻辑退场.
                self._kan_pending = False
                # 新的一手 = 副露记账归零(副露只属于一手).
                self._melded_tiles = []
                self._pre_meld = None
                # 副露组数与张数校验状态同归零(见 _hand_count_ok).
                self._meld_count = 0
                self._count_mismatch_streak = 0
                # 新的一手 = 立直状态归零(立直只属于一手; 流局/别人和了后
                # 游戏自动开下一手, 我们没立直就得正常打).
                riichi_active = False
                pending_riichi = False
                # 新的一手 = 摸牌模糊标志归零(属于"这一次摸牌").
                self._drawn_blurry = False
            if state.gave_up:
                self.log_warning(
                    f"和了之后 {tuning.GAME_GRACE_PERIOD:.0f}s 没等到结算页，"
                    f"退回旧行为继续打。若反复出现，请把 logs 和留证帧发出来")
            if not state.done and decision.action in DEAL_BLOCKED_ACTIONS:
                self.log_info(
                    f"  这一手没在进行中（手牌 {hand.count} 张，闸门未开），"
                    f"这一轮不{decision.action}")
                self.sleep(tuning.TURN_POLL_INTERVAL)
                continue

            # 1.7.5) 张数硬校验(2026-10-11 用户提议落地, 三碰局实锤的洞):
            # 记住了本手的副露组数, 暗牌数就只有 13-3n / 14-3n 两个合法值.
            # 读数张数落在集合外 = 读数被污染(副露混进手牌 / 漏认 / 幻觉),
            # 这一轮什么都不做 —— 特别是不出牌: 三碰局里 8 张恰好 %3==2,
            # 决策层判"该我出牌"然后反复想打已碰掉的牌, 就是缺这道闸.
            # 只在闸门已开(state.done)后执行: 发牌动画期的过渡张数由闸门管.
            if state.done and not self._hand_count_ok(hand, frame):
                self.sleep(tuning.TURN_POLL_INTERVAL)
                continue

            # 2) 卡死检测. 放在决策之后, 因为它需要知道"这一轮有没有动手".
            #    开局宽限期内不算: 发牌动画期间手牌是逐张出现的, 张数本来就在变.
            if time.time() >= self._deal.grace_until and stuck.feed(
                hand.count, set(buttons), acted=decision.action != ACTION_WAIT
            ):
                self._save_unknown_page(
                    frame, f"连续{stuck.same_count}轮无变化_疑似对局已结束")
                self.log_warning(
                    f"连续 {stuck.same_count} 轮感知结果完全一样（手牌 "
                    f"{hand.count} 张、按钮 {sorted(buttons) or '无'}）且没有任何动作，"
                    f"判定这局已经结束但页面没认出来（多半是结算页），已截图留证。"
                    f"请用「截图标注」框出结算页的特征区域。")
                return True

            if decision.action == ACTION_WAIT:
                # "在等别人"是每局几十轮的正常状态, 不算"认不出来";
                # 只有 suspect(张数不对 / 认出怪牌 / 牌效算不动)才计数.
                if decision.suspect:
                    if time.time() < self._deal.grace_until:
                        pass                    # 开局动画期, 正常
                    elif not hand.reliable:
                        # 读数本身不可信(有空档)时不计数也不留证: 一手结束的
                        # 顺位结算画面(流局那种没有「确认」按钮的)读数就是
                        # 9~11 张 + 空档, 拿它当"无法识别"存了 3 张无用截图
                        # (2026-10-08 实跑). 真正卡住的页面由 StuckDetector
                        # 兜底, 那条路不看手牌读数.
                        pass
                    else:
                        unknown_streak += 1
                        if unknown_streak == tuning.UNKNOWN_PAGE_AFTER:
                            self._save_unknown_page(
                                frame, f"对局内无法识别_{decision.reason}")
                self.sleep(tuning.TURN_POLL_INTERVAL)
                continue

            # 连点都没反应时**这一轮不再点**. StuckDetector 看不到这种情况
            # (它要求"这 N 轮没有任何动作", 而我们每轮都在点), 所以 2026-10-07
            # 那次画面冻住之后照样盲点同一张牌, 一直空转到 40 轮才兜底, 而且
            # 把雀魂的「确定要离开房间吗？」模态框点了出来. 现在 3 轮(≈1 秒)
            # 就留证并停一手 —— 停下来本身还让 StuckDetector 有机会触发。
            if (decision.action == ACTION_DISCARD
                    and self._discard_misses >= tuning.DISCARD_MISS_LIMIT):
                self._save_unknown_page(
                    frame, f"连续{self._discard_misses}轮出牌没打出去")
                self.log_warning(
                    f"连续 {self._discard_misses} 轮点了出牌但手牌一张没少，"
                    f"已截图留证。多半是画面上压着模态框（如「确定要离开房间吗？」）"
                    f"或画面冻住了，雀魂根本没收到点击。这一轮先不动，下一轮重新认。")
                self._discard_misses = 0
                self.sleep(tuning.TURN_POLL_INTERVAL)
                continue

            unknown_streak = 0
            # 这一轮的出牌是不是"立直选牌态里的宣言牌": 只有选牌态真的被
            # 认出来(allowed 非 None)且**不是**耐心耗尽后的降级(那时
            # _riichi_misses 已累计), 才算数. 点了立直但游戏没进选牌态
            # (点立直失败/被取消)时 allowed 是 None, 那种出牌是普通出牌.
            declare_turn = (pending_riichi and allowed is not None
                            and self._riichi_misses == 0)
            # 立直状态机: 点下立直 -> 必须先打宣言牌 -> 真的少了一张才清位.
            # 清位判据是**手牌数**, 不是"这一轮决定了 DISCARD": 雀魂在立直后
            # 只认宣言牌, 点错的地方它直接忽略, 手牌数不会变. 早先只看
            # decision.action 就清位, 结果点失败一次就把立直状态丢了, 下一轮
            # 又按牌效去算 -- 正是实跑里"连点 5 轮同一张牌"的成因.
            if decision.action == ACTION_RIICHI:
                pending_riichi = True
                self.log_info("  立直已宣布, 接下来必须打宣言牌")
            elif (pending_riichi and decision.action == ACTION_DISCARD
                    and not declare_turn):
                # 点了立直但游戏没进选牌态(立直没点成): 这一轮按普通出牌
                # 打掉, 立直意图作废, 回到正常打牌.
                pending_riichi = False
            if riichi_active and decision.action == ACTION_DISCARD:
                # 立直生效后雀魂自动摸切: 忽略出牌决策(点了也是无效点击),
                # 也不必再"选牌". 杠(立直中的暗杠)/和了/确认不受影响.
                self._log_repeat_once(
                    "riichi_auto", "active",
                    "  立直生效中：雀魂自动摸切，不主动出牌")
                self.sleep(tuning.TURN_POLL_INTERVAL)
                continue
            # 1.9) 动作前双帧一致性确认(2026-10-08 用户提议"多截几次取
            #      最稳"落地): 这几次实跑的错动作(过场画面乱出牌 / 滑入
            #      动画期漏碰)全都发生在"牌桌还没稳定下来"的时候, 而幻觉
            #      读数和动画帧撑不过第二帧 —— 所以决策出来后等一小段,
            #      重抓一帧用同参数重算一遍, 两帧一致才动手; 不一致就本轮
            #      不动, 回主循环重新决策(每 0.3s 一轮, 天然重试, 不会锁死;
            #      真正稳定的画面下一轮必然通过). 判据见 table.action_confirmed.
            #      副产品: meld 只会在按钮滑入结束(位置稳定)后才点, 动画期
            #      的低置信帧从此不在决策窗口里; 第一帧漏认 peng 误判"跳过"
            #      时, 第二帧认出 peng 会让两帧不一致, 真机会不会被 skip 掉.
            ok, why = self._confirm_stable(decision, hand, buttons, cfg,
                                           spec, vision, vote_buffer,
                                           pending_riichi)
            if not ok:
                self._log_repeat_once(
                    "unstable", decision.action,
                    f"  画面还没稳定（{why}），本轮不动，下一轮重新决策")
                self.sleep(tuning.TURN_POLL_INTERVAL)
                continue
            self._execute(decision, hand, buttons, cfg, game, turns)
            # 点完把光标停到安全区: 光标趴在手牌上, 悬停的牌会被游戏顶起来,
            # 实跑里正好挡住立直按钮的模板(2026-10-07 21:41); 光标箭头本身
            # 也不该出现在下一轮的手牌读数里.
            self._park_mouse(frame)

            # 和了之后这一手就结束了, 但雀魂还会放一段摊牌动画(我们自己的和了牌
            # 正面朝上摆回手牌位置, 摊牌期间读出来是干净的 13~14 张, 只看张数
            # 跟正常打牌没区别), 然后才是结算页. 实跑 t189 点了自摸之后,
            # t190/t191/t195 又"打"了 3 次牌 —— 所以这里必须显式封锁,
            # 否则那段动画里会一直点手牌区. 解除条件与兜底见 table.DealWatch.
            if decision.action == ACTION_WIN:
                self._deal.agari(time.time())
                self.log_info("  和了，等摊牌 / 结算过完，期间不再读手牌和出牌")

            # 出牌后重新看一眼手牌: 真的少了一张才算把宣言牌打出去.
            # _discard 内部已经会为"选中再确认"再点一次, 所以这里读到的
            # 已经是最终结果.
            if declare_turn and decision.action == ACTION_DISCARD:
                self.sleep(tuning.DISCARD_VERIFY_DELAY)
                # 同 _discard: 残帧会让读数比稳定态少一张, 而这里的判据正是
                # "少没少", 复读一次再判, 免得把打出去的宣言牌判成没打出去.
                _frame, after = self._read_hand_retry(hand.count - 1)
                if after is not None and after.count < hand.count:
                    self.log_info(
                        "  宣言牌已打出，立直生效：之后每回合雀魂自动摸切，"
                        "不再主动出牌")
                    pending_riichi = False
                    riichi_active = True
                    self._riichi_misses = 0

        frame = self._grab()
        self._save_unknown_page(frame, f"第{game}局超时")
        self.log_error(f"第 {game} 局超时（{tuning.GAME_TIMEOUT}s），已截图留证")
        return False

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

    def _read_hand_voted(self, vision: GameVision, frame, cfg: dict,
                         buffer: list) :
        """读手牌, 并按配置做多帧投票压掉低置信度误识别.

        **为什么需要**(2026-10-06 立直那局的实测): 牌面模型对某些位置的牌
        置信度只有 0.60~0.74, 同一张牌在连续帧里会在两个类别之间来回跳.
        实测 585 轮里index=4 的跳变率 2.2%, 是全手牌最高的(其余多在0.3%).
        后果很具体: 系统会认真算出"牌效最优"的牌, 但那张牌根本不是屏幕上
        那张, 于是点下去打不出去 -- 实跑里连点5 轮, 手牌数一直是 14.

        **为什么不靠调高 TILE_CONF**: 实测 1244 个框里真牌最低 0.433,
        而 0.40~0.75这一档**既有真牌也有误识**. 单纯抬阈值会把真牌整个丢掉,
        丢牌的危害比认错更严重(张数立刻不对, 整轮判成"认漏了"→ 假警报).
        投票是只在"多帧不一致"时才改变结果的补丁, 不动主阈值.

        buffer 复用同一个list, 避免每轮新建(它是热路径). 每次调用推进一格,
        只保留最近 hand_vote_frames 帧. 手牌张数变了(摸牌/出牌/副露)就清空
        -- 那种情况下前后两帧的槽位根本不是同一张牌, 投票没有意义.
        """
        hand = self._strip_kan_setaside(vision.read_hand(frame))
        self._save_low_conf_frame(frame, hand)
        frames = int(tuning.HAND_VOTE_FRAMES or 0)
        if frames <= 1:
            buffer.clear()
            return hand

        prev = buffer[-1] if buffer else None
        # 摸牌帧(13->14, 11->8, 8->5): 前 n-1 张是**原来就在屏上的同一批牌**,
        # 雀魂摸牌只把新牌放到右端独立位, 原 13 张位置不动 -> 前缀槽位一一对
        # 应, 可以对齐投票; 新摸的最后一槽没有历史, 用当帧原读.
        # 2026-10-11 東1局9巡实锤(牌谱核对): 摸牌帧槽5 8p 被读成 1s(conf 0.95),
        # 而**张数一变缓冲就清空, 决策帧恰好永远是裸单帧** -> 按误读手牌打了
        # 9p(真实手牌最优是打 5p). 8p<->1s 在 20:16:17-25 之间来回翻转了两次,
        # 等待帧都有投票保护, 唯独决策帧没有 -- 本修复把这个洞补上.
        # 其余张数变化(出牌后重排/副露)槽位内容会换牌, 仍然清空.
        drew = (prev is not None and len(prev) == len(hand.boxes) - 1
                and len(hand.boxes) % 3 == 2)
        if prev is not None and len(prev) != len(hand.boxes) and not drew:
            buffer.clear()
        buffer.append(hand.boxes)
        if len(buffer) > frames:
            del buffer[0]

        if len(buffer) < 2:
            return hand
        labels = vote_labels(list(buffer))
        if (drew and labels is not None
                and len(labels) == len(hand.boxes) - 1):
            labels = list(labels) + [hand.boxes[-1].label]
        return self._voted(hand, labels, cfg)

    def _voted(self, hand, labels, cfg: dict):
        """把投票结果套回 HandReading; 投票无效或无变化时原样返回.

        真正的搬迁在 `src.detector.with_voted_labels`(用 replace 保证除
        标签外的字段一个都不丢, 尤其是 `missing_slots`). 这里只负责日志。

        日志里的置信度是**当前帧**那个框的观测值, 不是投票选中的那个框的
        —— 投票只改"这张牌叫什么", 几何和置信度都是当帧的事实, 见 vote_labels.
        """
        if labels is None or len(labels) != len(hand.boxes):
            return hand
        changed = [(i, hand.boxes[i].label, labels[i],
                    round(hand.boxes[i].conf, 3))
                   for i in range(len(labels)) if labels[i] != hand.boxes[i].label]
        if changed:
            self.log_info(f"  多帧投票纠正了 {len(changed)} 张牌: "
                          + ", ".join(f"#{i} {a}->{b}(当帧置信 {c})"
                                      for i, a, b, c in changed))
        return with_voted_labels(hand, labels)

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
                f"还没进牌桌（加载 / 对局开始页 / 过场），等牌桌出来再读手牌")
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
                        vision: GameVision, vote_buffer: list,
                        pending_riichi: bool) -> tuple[bool, str]:
        """动作前抓第二帧重算一遍, 判据在 table.action_confirmed(纯逻辑).

        第二帧的输入完全镜像主循环那一轮: 同样的读手牌(含投票/杠牌剥离)、
        读按钮、高亮牌、低置信槽位, 只是 rng 换成固定种子 —— 同一读数下
        决策必须可复现, 否则"要打的牌变了"会被随机性(5% 次优)误触发.
        投票缓冲区复用同一个 list: 第二帧也是一帧真实画面, 推进一格是
        它应得的(和下一轮主循环读帧没有区别).
        """
        self.sleep(tuning.STABLE_CONFIRM_DELAY)
        frame2 = self._grab()
        if frame2 is None:
            return False, "二次抓帧失败"
        hand2 = self._read_hand_voted(vision, frame2, cfg, vote_buffer)
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
            babei_policy=tuning.BABEI_POLICY,
            allowed_indices=allowed2,
            blurry_indices=blurry2,
            robust_discard_enabled=cfg["robust_discard"],
            missing_slots=hand2.missing_slots,
                button_centers=centers2,
                blurry_draw=self._drawn_blurry,
                melded=self._melded_tiles,
                rng=random.Random(0),
            )
        center1 = None
        if decision.button:
            boxes1 = buttons.get(decision.button) or []
            center1 = self._box_center(boxes1[0]) if boxes1 else None
        return action_confirmed(decision, second, center1,
                                centers2.get(decision.button),
                                tol=tuning.BUTTON_STABLE_TOL)

    def _execute(self, decision, hand, buttons, cfg: dict,
                 game: int, turns: int) -> None:
        """执行一个动作, 并按人的节奏等一会儿."""
        if decision.action == ACTION_DISCARD:
            delay = timing.discard_delay(self._rng)
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

        delay = self._action_delay(decision)
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

        三个出口, 按保守程度排:

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

        自动和了 + 自动拔北恒开; 「鸣」开关按**本手手牌**动态决定
        (2026-10-08 用户拍板): 有役牌对子(白/发/中/场风东)就不点「鸣」,
        让碰的机会弹出来(配合 MELD_POLICY="peng"); 没有役对就点「鸣」
        自动拒绝鸣牌. 注意「鸣」=**自动拒绝鸣牌**, 有役对时绝不能开,
        否则雀魂会自动拒掉所有碰的机会(见 tuning.auto_toggles_on).
        has_yaku 由调用方判定并做**两帧共识**(just_opened 那帧的 z 牌
        单帧误读会造出幻影役对, 见主循环的说明).
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

        cut = rect("切")
        if is_toggle_green(frame, cut):
            self.log_warning("  自动开关:「切」(自摸切)开着, 会自动打出摸到的牌, 帮忙关闭")
            self._click_toggle_row(cut, "切")
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
        # 「切」误点亮防护(2026-10-11 升级): 点回去之后必须复读确认, 关不掉
        # 就再点(最多两轮); 意图开关没点亮的顺手补点. 全绿 + 「切」全灰才算
        # 就绪, 否则不置 _toggles_ok, 下一手从头再来.
        for _ in range(2):
            if not failed and not is_toggle_green(frame, cut):
                break
            if is_toggle_green(frame, cut):
                self.log_warning("  自动开关:「切」(自摸切)亮着, 点掉")
                self._click_toggle_row(cut, "切")
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
        cut_green = is_toggle_green(frame, cut)
        if failed or cut_green:
            msgs = []
            if failed:
                msgs.append(f"没点亮: {failed}")
            if cut_green:
                msgs.append("「切」关不掉, 自动摸切会替你打牌（下一手再试）")
            self.log_warning(f"  自动开关{'; '.join(msgs)}"
                             f"（反复失败 = 点击没落在竖条字符上, "
                             f"请核对 tuning.AUTO_TOGGLE_ROWS 行位标定）")
        else:
            self._toggles_ok = True
            on = [n for n in toggles if n in rows]
            note = "自动和了" + ("/不鸣牌" if "鸣" in on else "/可鸣牌")
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
        from src.mahjong.table import (ACTION_BABEI, ACTION_CONFIRM,
                                       ACTION_MELD, ACTION_RIICHI,
                                       ACTION_SKIP, ACTION_WIN)
        if decision.action == ACTION_WIN:
            return timing.win_delay(self._rng)
        if decision.action == ACTION_RIICHI:
            return timing.riichi_delay(self._rng)
        if decision.action in (ACTION_MELD, ACTION_BABEI):
            return timing.meld_delay(self._rng)
        if decision.action in (ACTION_SKIP, ACTION_CONFIRM):
            return timing.idle_delay(self._rng)
        return timing.idle_delay(self._rng)

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
        index = pick_tile_index(hand.tiles, decision.tile, self._rng)
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

    def _read_hand(self):
        """抓一帧新手牌. 抓不到帧返回 None(调用方跳过这一轮的判断)."""
        return self._read_hand_frame()[1]

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
        `_record_turn`); 只要读数的话用 `_read_hand` 就够了.
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
                     decision, frame=None) -> None:
        """记一轮, 并把**决策当帧**存到 data/debug/.

        `frame` 必须是这一轮真正读过的**那一帧**, 不能省。原来这里不接 frame,
        存帧时自己重新 `_grab()` 一张 —— 抓到的已经是下一次循环的画面了, 于是
        `frame_g1_tN.png` 与第 N 轮的决策**不是同一帧**。它不一定差很远, 但在
        "手牌正在逐张发出来"的那几秒里能差 5 张牌, 2026-10-07 排查时直接被带偏
        (拿留证帧重跑模型, 结果和日志里的 hand 对不上, 白查了一轮)。
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
        })
        if self._cfg.get("save_frames"):
            self._save_frame(f"g{game}_t{turns}", frame)

    def _save_frame(self, tag: str, frame=None) -> None:
        """存一帧留证. `frame` 给了就用它(决策当帧), 没给才去抓新的."""
        if frame is None:
            frame = self._grab()
        if frame is None:
            return
        ts = datetime.now().strftime("%H%M%S")
        out = DEBUG_ARTIFACTS_DIR / f"frame_{tag}_{ts}.png"
        imwrite_unicode(out, frame)
