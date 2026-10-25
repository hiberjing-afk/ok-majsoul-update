from __future__ import annotations

"""段位场 / 友人场连打: 导航 + 抓包 + 事件驱动对局内核.

导航(大厅 -> 段位场/准备页 -> 开始)复用 CreateFriendsRoomTask; 本类负责
一局内的内核接线, 以及牌桌之外的页面处理:

    1. 回到准备页 / 结算页 -> 交给页面处理, 这一局结束
    2. 抓包链路 -> NetAssist -> GameState.apply() 产出协议事件
    3. HandController 由事件产出决策(和 / 立直 / 拔北 / 鸣牌 / 出牌 / 等)
    4. VisionLocator 只把决策翻译成手牌槽位或按钮坐标, 再点击
    5. 每一轮的决策与坐标写进 data/debug/game_<时间戳>.jsonl

对局内不再按视觉轮询猜"发牌完了没有 / 上一手结束没有": 这些事实由协议事件
(ActionNewRound / ActionHule / ...)给出, 视觉只回答坐标.

**前台**: ok-script 的 executor 在游戏不在前台时会整个挂起, 所以拉前台做在
交互层 src/interactions.py 的 should_capture 里, 不需要手动点回游戏.

配置: 见两个子类(段位场 / 友人场)各自的 configs/*.json。界面只保留用户旋钮:
    max_games                连打几局(0 = 只建房不开始)
    shutdown_after           打满指定局数后关机
  段位场另有 tier / game, 友人场另有 players / length(见子类的 _default_config).
其余阈值与延迟全部在 `src/tasks/tuning.py`.
"""

import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

from src.annotations import missing_categories
from src.detector import GameVision
from src.instrumentation import bench
from src.mahjong import timing
from src.net.assist import NetAssist
from src.paths import DEBUG_ARTIFACTS_DIR
from src.retention import prune_turn_frames
from src.rooms import (
    RANKED_LOCKED,
    RoomSpec,
    ready_marker,
    required_categories,
    spec_by_key,
    start_step,
)
from src.tasks import tuning
from src.tasks.CreateFriendsRoomTask import CreateFriendsRoomTask, TaskStopped
from src.vision import is_toggle_green, page_changed

# 结算页(对局结束的点数结算). 页面名与区域名对应
# ok_templates/coco_annotations.json 里的 "结算页/终局" 与 "结算页/确认".
# 不用 RoomScreens 字段: 结算页与房间规格无关, 任何规格打完都是它.
RESULT_SCREEN = "结算页"
RESULT_FEATURE = "终局"# 只用来探测"是不是结算页"
RESULT_CONFIRM = "确认"    # 点它回到准备页

# 终局那一串页面里的「获得奖励」(段位场打满几局后偶发, 不是必现).
#
# 为什么必须给它自己的标注(2026-10-21 实跑卡死 60s 的成因): 这一页也画着
# 「终局」标题, 但那两个字压在立绘上, 而 「结算页/终局」模板是从 result.png
# 的深色背景上裁的 —— 实测同一帧上 结算页/终局 只有 **0.654**(奖励页自己的
# 截图 reward.png 上更低, 0.652), 阈值 0.88 根本进不来; 「确认」按钮这一页
# 没有(0.199). 而 「奖励页/获得奖励」自匹配 1.000、全帧次峰 0.26。
# 所以判页只能靠它自己的标题, 别再用别的页的标题去猜。
REWARD_SCREEN = "奖励页"
REWARD_FEATURE = "获得奖励"

# 终局之后可能出现的所有页面特征(每页用**自己**的特征, 见上).
# 一个来源: 检测(判页)与启动期"缺标注"告警都读这一张表.
RESULT_PAGE_FEATURES: tuple[tuple[str, str], ...] = (
    (RESULT_SCREEN, RESULT_FEATURE),
    (RESULT_SCREEN, RESULT_CONFIRM),
    (REWARD_SCREEN, REWARD_FEATURE),
)

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
# 流局画面没有这个按钮(约 5s 自动过), 下一手 ActionNewRound 会重置。
HAND_RESULT_SCREEN = "牌局页"
HAND_RESULT_CONFIRM = "局确认"

# ---- 场次: 段位场(主任务) / 友人场(测试用) ----
# 2026-10-25 用户要求把两者拆成两张任务卡(段位场要选 段位 x 对局, 友人场要选
# 人数 x 时长, 两套选项毫不相干, 挤在一个下拉框里只会互相干扰). 本类退成
# **公共基类**, 不再注册成任务卡片:
#
#   PlayRankedTask  段位场: tier(银之间/金之间) x game(四人东/四人南/三人东/三人南)
#   PlayFriendsTask 友人场: players(三人/四人) x length(一局/东风局)
#
# 两条流程形状不同的地方在 rooms.RoomSpec.ranked, 与这里无关.


# 基类的默认配置: 只放"两种场次都要"的旋钮. 子类在上面叠加自己的房型选项.
_DEFAULT_CONFIG = {
    # 连打几局. **这里只放用户真正常用的**(2026-10-07 用户决定), 其余二十多项
    # 内部调参全搬去了 src/tasks/tuning.py -- 框架会把 default_config 里
    # 每一项都渲染成控件, 塞满阈值/延迟只会让卡片没法看.
    "max_games": 1,
    # 打满指定局数后关机(2026-10-25 用户要求). 只在**正常打满**时触发,
    # 中途报错/用户喊停都不会关 —— 否则调试时天天被关机.
    "shutdown_after": False,
}

#: 关机配置项的中文说明(框架会把 config_description 显示在配置项旁边).
_CONFIG_DESCRIPTIONS = {
    "max_games": "连打几局（0 = 只建房不开始）",
    "shutdown_after": "打满指定局数后关机（60 秒缓冲，可 shutdown /a 取消）",
}


class PlayOneGameTask(CreateFriendsRoomTask):
    """建房 -> 点开始 -> 打完一局(可连打). **基类, 不直接注册成任务.**

    两个子类只差"开什么房"(见模块说明), 流程/内核/收尾全在这里.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "自动打牌"
        self.description = "打牌流程基类（请用段位场 / 友人场任务）"
        self._vision: GameVision | None = None
        self._record_path: Path | None = None
        self._net: NetAssist | None = None        # 网络读牌对账门面
        self._engine = None                       # 事件驱动对局内核
        self._capture = None              # 抓包链路(mitmdump + TUN)生命周期
        self._capture_spool: Path | None = None   # 本场 spool 路径(显式交给桥)
        self._rng = random.Random()
        self._cfg: dict = {}
        # 本类覆写了 _default_config(), 所以要**再调一次**把结果写进框架读的
        # self.default_config. 不调的话界面拿到的是基类在 __init__ 之前算的
        # 那份(不含本类的对局内参数), 甚至可能是空 dict 导致整个卡片不展开.
        # 见 CreateFriendsRoomTask._init_default_config 的说明.
        self._init_default_config()
        self.config_description = dict(_CONFIG_DESCRIPTIONS)
        # 子类在 super().__init__() 之后覆写 config_type 添加自己的房型下拉框.
        self.config_type = {
            # 连打局数上下限: 不声明时框架默认上限 99999999, 一旦进去一个
            # 千万级误值, 箭头点不到头、框里的数也几乎没法选中改写
            # (2026-10-11 实测踩坑, 配置里出现过 11212218).
            "max_games": {"min": 0, "max": 99},
        }

    def _default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    def _resolve_spec(self, cfg: dict) -> RoomSpec:
        """按配置算出要开什么房间. **子类必须实现**(各自有一套选项).

        显式的 room 字段仍然最优先(命令行批量跑用).
        """
        room = cfg.get("room")
        if room:
            return spec_by_key(room)
        raise NotImplementedError(
            f"{type(self).__name__} 没有指定房间规格（子类必须实现 _resolve_spec）")

    def _players_key(self, cfg: dict) -> str:
        """牌桌页标注 / 自动开关行位按哪套"人数"取.

        人数以 RoomSpec 为准(段位场由对局名推出来, 与任何下拉框无关;
        友人场由 players 下拉框组合成规格). _spec 用 getattr 兜底:
        测试注入的 stub 不一定带这个属性.
        """
        spec = getattr(self, "_spec", None)
        if spec is not None:
            return "四人" if spec.player_count == 4 else "三人"
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
            # 指路那句是给**老用户**的: 启动器渠道的素材只在首次安装时下载
            # (update 仓库是公开的, 截图不能带, 见 docs/RELEASE.md), 升级代码
            # 不会自动换模板 -> 选了新加的场次就在这里撞墙.
            raise RuntimeError(
                f"房间 '{spec.label}' 缺标注: {missing}\n"
                f"如果这一档是最近才加的, 那是本地素材还是旧的: "
                f"运行 python -m src.update --apply 更新后重试。")
        missing_result = self._missing_result_features()
        if missing_result:
            self.log_warning(
                f"终局页面特征缺标注: {missing_result} —— 打牌不受影响, "
                f"但弹到这一页会点不掉（见 src/tasks/PlayOneGameTask.py 里 "
                f"RESULT_PAGE_FEATURES 的说明）")
        self._warn_optional_annotations(spec)

        # 先修剪上一批留下的每轮诊断帧, 再开始记这一批(见 _prune_debug_frames)。
        self._prune_debug_frames()
        self._open_record()
        self._ensure_vision()          # 副作用: 模型没加载就先加载

        # 抓包链路(mitmdump + mihomo TUN)已在 `_before_room_flow` 起好 ——
        # 段位场点完"四人东"就自动匹配, 分钟级的匹配期也要读协议(2026-10-17
        # 用户拍板"提前到任务开始"). 本方法只管打; 停止由 `_after_room_flow`
        # 的 finally 兜底(停止指令/异常退出都会关掉自己起的进程).
        completed = self._run_games(cfg, spec)
        self._shutdown_after_task(cfg, completed)

    def _shutdown_after_task(self, cfg: dict, completed: bool) -> None:
        """打满指定局数后关机(用户 2026-10-25 要的选项).

        **只有正常打满才关**: 中途出错、匹配超时、用户点停止都不关 —— 否则
        白天调试时每挂一次就被关一次机.
        """
        if not cfg.get("shutdown_after"):
            return
        if not completed:
            self.log_info("任务没有正常打满局数，按约定不关机")
            return
        # 留一分钟缓冲: 这是整台机器, 误配一次代价太大.
        self.log_warning(
            f"{tuning.SHUTDOWN_DELAY}s 后关机（想取消: shutdown /a）")
        try:
            subprocess.run(
                ["shutdown", "/s", "/t", str(tuning.SHUTDOWN_DELAY),
                 "/c", "ok-majsoul 任务已完成，即将关机"],
                check=False, timeout=10,
            )
        except Exception as e:  # noqa: BLE001 - 关不掉只影响收尾, 别打断任务日志
            self.log_error(f"执行关机命令失败: {e}")

    def _before_room_flow(self, cfg: dict) -> None:
        """任务一开始(点四人东之前)就起抓包. 覆写基类钩子."""
        self._cfg = cfg
        self._start_capture()

    def _after_room_flow(self) -> None:
        """房间流程收尾(覆写基类钩子): 关掉自己起的抓包进程."""
        self._stop_capture()

    def _run_games(self, cfg: dict, spec) -> bool:
        """连打循环(段位场 / 友人场两路). 抓包在上层已经起好.

        返回**是否正常打满了 cfg["max_games"] 局**(中途出错/用户喊停 -> False),
        只有 True 才会触发"完成后关机"(见 _shutdown_after_task).
        """
        # 网络读牌门面: 本场只建一次, 贯穿整场(段位场连打多局共用同一个
        # spool 文件与同一个 GameState). 关闭档不构造(免得白扫目录).
        # 桥不可用时内核停手, 不盲打.
        self._net = self._build_net(cfg)
        self._attach_engine(cfg, drain=True)

        if spec.ranked:
            completed = self._ranked_games(cfg, spec)
            self._hold_final_result(cfg)
            return completed

        completed = False
        for game in range(1, cfg["max_games"] + 1):
            # 连打好几局时这里是长循环, 必须先问停止(见 _should_yield).
            if self._should_yield():
                self.log_info("收到停止指令, 不再开下一局")
                return False
            self._report_progress(game, cfg["max_games"])
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
                    return False
                self.log_error(f"第 {game} 局没能正常结束，停止连打")
                break

            if game < cfg["max_games"]:
                lo, hi = tuning.BETWEEN_GAMES_DELAY_RANGE
                delay = self._rng.uniform(lo, hi)
                self.log_info(f"局间休息 {delay:.1f}s")
                self.sleep(delay)
        else:
            # 循环一次都没 break/return = 每一局都正常打完.
            completed = True

        self._hold_final_result(cfg)
        self.log_info(f"对局记录: {self._record_path}")
        return completed

    def _report_progress(self, game: int, total: int) -> None:
        """把"第几局 / 共几局"写进任务卡上的常驻信息区.

        框架的 TaskCard 会把 `task.info` 渲染成一张表(见 ok 的
        TaskTab.update_task_info), 运行期间每 0.5s 刷新一次 —— 这就是用户
        要的"常驻展示"(2026-10-25), 比让他去翻日志稳.
        """
        self.info_set("局数", f"第 {game} 局 / 共 {total} 局")

    # ---------------------------------------------------------------
    # 段位场循环连打
    # ---------------------------------------------------------------

    def _ranked_games(self, cfg: dict, spec) -> bool:
        """段位场循环: 匹配 -> 打完 -> 回大厅 -> 重新排队 -> 下一局.

        与友人场连打的两点不同: ①开下一局不靠准备页的"开始"按钮, 而是
        回大厅把 段位场->段位->对局 整条流程重新点一遍(段位场没有
        准备页, 见 rooms.RoomSpec.ranked); ②匹配是分钟级的, 开局前要等
        匹配成功(见 _wait_matchmaking).

        返回是否正常打满局数, 见 _run_games.
        """
        completed = False
        for game in range(1, cfg["max_games"] + 1):
            # 连打好几局时这里是长循环, 必须先问停止(见 _should_yield).
            if self._should_yield():
                self.log_info("收到停止指令, 不再开下一局")
                return False
            self._report_progress(game, cfg["max_games"])
            self.log_info(f"===== 第 {game}/{cfg['max_games']} 局（段位场） =====")
            self._wait_matchmaking(cfg, spec, game)

            finished = self._play_one_game(cfg, game)
            self._record({"event": "game_end", "game": game,
                          "finished": finished})
            if not finished:
                if self.stopped:
                    self.log_info("收到停止指令, 结束连打")
                    return False
                self.log_error(f"第 {game} 局没能正常结束，停止连打")
                break

            if game < cfg["max_games"]:
                lo, hi = tuning.BETWEEN_GAMES_DELAY_RANGE
                delay = self._rng.uniform(lo, hi)
                self.log_info(f"局间休息 {delay:.1f}s，然后回大厅重新排队")
                self.sleep(delay)
                self._back_to_lobby(cfg, spec, game)
                self._run_flow(cfg)
        else:
            completed = True

        self.log_info(f"对局记录: {self._record_path}")
        return completed

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

    def _hold_final_result(self, cfg: dict) -> None:
        """最后一局打完先等终局页出现, 让用户看清再关抓包/退程序.

        只探测/等待, **不点确认**; 12s 内没等到就照常收工, 不把任务判失败.
        2026-10-19 用户反馈: 最后一手和了后程序立刻退出, 终局页一闪而过.
        """
        if self.stopped:
            return
        deadline = time.time() + tuning.FINAL_RESULT_TIMEOUT
        while time.time() < deadline:
            if self._should_yield():
                return
            try:
                frame = self._grab()
            except Exception:  # noqa: BLE001 - 收尾等待失败不该改任务结果
                return
            if self._locate(RESULT_SCREEN, RESULT_FEATURE, cfg, frame,
                            strict=True, verbose=False) is not None:
                self.log_info("  终局页已出，保留抓包 3s 再收工")
                self.sleep(tuning.FINAL_RESULT_HOLD_SECONDS)
                return
            self.sleep(tuning.PAGE_POLL_INTERVAL)
        self.log_info("  没等到终局页（可能已自动翻页），直接收工")

    def _back_to_lobby(self, cfg: dict, spec, game: int) -> None:
        """终局后一路点到回大厅(下一轮排队的起点).

        段位场终局不是一页, 而是一串: 本局排名 -> 分数变化 -> 获得奖励 ->
        完成任务(2026-10-22 用户确认的顺序; 奖励页偶发, 不是每局都有).
        逐个标页面特征太脆 —— 「获得奖励」页上连「终局」标题都只有 0.65,
        见 REWARD_SCREEN 的说明 —— 而这一串页面的**共同点**是: 点「确认」
        那个位置就能翻页, 没有按钮的奖励页点同一点也能过(用户实测).

        所以这里不做页面识别: 每轮只问"回到大厅了吗", 没回就在确认位点一下,
        直到回大厅或等满 LOBBY_RETURN_TIMEOUT 判失败. 页数/顺序都不参与判断,
        多一页少一页都能过.

        **不设点击次数上限**(2026-10-22 实跑教训): 上一版点到 12 次就只轮询
        不点了, 那天终局恰好赶上客户端断线重连 —— 12 次点击全被重连遮罩吞掉,
        第 12 次落在 21s, 客户端 31s 才重连回来, 之后 40s 一次都没再点,
        于是一路等到超时. 点击被吞是外因, 我们自己的错是"点了 12 次就放弃"。
        上限只有 LOBBY_RETURN_TIMEOUT 一个.

        代价控制靠`先问大厅再点`: 大厅一旦认出来立刻停手; 认不出来时最多
        点到超时(那个坐标在大厅上是「寻觅/商家」那排入口), 这是用户拍板的边界.
        """
        deadline = time.time() + tuning.LOBBY_RETURN_TIMEOUT
        clicks = 0
        stalled = 0     # 连续几次点击画面一个像素都没动
        self.log_info(f"第 {game} 局已打完，连点确认位回大厅")
        frame = self._grab()
        while time.time() < deadline:
            if self._should_yield():
                raise TaskStopped("等待回大厅期间收到停止指令")
            if frame is None:
                # 抓帧失败时 _grab 会退回缓存帧; 连缓存都没有就别算坐标.
                self.sleep(tuning.PAGE_POLL_INTERVAL)
                frame = self._grab()
                continue
            if self._locate(spec.screens.lobby, spec.screens.entry, cfg, frame,
                            strict=True, verbose=False) is not None:
                self.log_info(f"    已回到大厅（终局后点了 {clicks} 次）")
                return
            nxt = self._click_result_confirm_spot(frame, game, clicks + 1)
            if nxt is None:
                break
            clicks += 1
            if page_changed(frame, nxt):
                stalled = 0
            else:
                stalled += 1
                if stalled == tuning.LOBBY_STALL_CLICKS:
                    # 只告警不改变行为: 重连回来以后那一下点击仍然要发出去.
                    self.log_warning(
                        f"  连点 {clicks} 次画面没有任何变化，点击可能被"
                        f"断线重连/失焦吞掉了，继续点到 "
                        f"{tuning.LOBBY_RETURN_TIMEOUT:.0f}s 上限")
            frame = nxt
        self._save_wait_timeout_frame(frame, spec.screens.lobby,
                                      spec.screens.entry)
        raise RuntimeError(
            f"终局后 {tuning.LOBBY_RETURN_TIMEOUT:.0f}s 没等到大厅页"
            f"（点确认位 {clicks} 次），已中止任务")

    def _missing_annotations(self, spec) -> list[str]:
        return missing_categories(required_categories(spec),
                                  set(self._annotations))

    def _warn_optional_annotations(self, spec) -> None:
        """段位场有几个"缺了不拦、但会让人看不懂报错"的标注.

        「段位不符」只有一处用途: 列表里找不到目标场次时, 告诉我们那是
        **段位不允许**而不是"模板没认出来". 缺了照样能打, 但报错会变成
        "滚遍也没找到四人南", 让人以为是标注坏了.
        """
        if not spec.ranked:
            return
        category = f"{spec.screens.settings}/{RANKED_LOCKED}"
        if category not in self._annotations:
            self.log_warning(
                f"缺标注 '{category}'：段位打不了某一档时会报"
                f"「找不到场次」而不是「段位不符」（不影响正常打牌）")

    def _missing_result_features(self) -> list[str]:
        """终局那串页面的特征缺了哪些.

        只告警不中止: 缺一个特征照样能打牌, 只是弹到那一页会点不掉
        (2026-10-21 就是这么卡死 60s 的). 这类"少一个就不能识别"的东西
        必须在开局前说出来, 否则表现成打了很多局之后偶发卡住, 极难查.
        """
        return missing_categories(
            [f"{screen}/{feature}" for screen, feature in RESULT_PAGE_FEATURES],
            set(self._annotations))

    def _prune_debug_frames(self) -> None:
        """修剪上一批留下的每轮诊断帧(见 src/retention.py).

        只删 `frame_g<局>_t<轮>_*.png`; 低置信/闸门/待标注留证帧与 jsonl
        对局记录一律不碰. **永不抛**: 清理失败只记 warning, 不影响开局.
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
    # 页面判定
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
        """按 tuning.CAPTURE_AUTO 起抓包链路. 已在跑的复用, 失败只记日志.

        对局内核依赖协议事件, 抓包起不来时本局无法打牌; 这里不抛异常,
        由 `_run_engine_game` 的统一入口报错并停手, 把排查信息留在日志里.
        """
        self._capture = None
        self._capture_spool = None
        if not tuning.CAPTURE_AUTO:
            return
        # 关闭档连抓包都不起(仅排查用, 此时内核无法工作).
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
                           f"本局无法打牌（见 docs/NET_CAPTURE.md）")
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
                "[抓包] 抓包链路没就位 —— 对局内核读不到协议，本局会停手；"
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
    # 协议门面与事件驱动内核
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
        """按开关构造协议门面. 关闭档返回 None(此时内核无法工作, 仅排查用).

        开关住在 tuning.NET_READ; cfg 里显式写了 "net_read" 时以它为准,
        留一个不改源码就能覆盖的口子.

        spool 优先用抓包模块给的那个路径: 自动抓包起的 mitmdump 写进
        `_capture_spool`, 显式指过去比"去目录里猜哪份最新"准得多.
        抓包没自动起(CAPTURE_AUTO 关掉 / 手动抓包)时退回按 mtime 发现.
        """
        mode = cfg.get("net_read") or tuning.NET_READ
        if mode == tuning.NET_READ_OFF:
            return None
        # getattr 兜底: 单测把未绑定方法绑到 stub self 上, stub 不带这个字段.
        return NetAssist(path=getattr(self, "_capture_spool", None),
                         debug_dir=DEBUG_ARTIFACTS_DIR,
                         max_spool_age=tuning.NET_MAX_SPOOL_AGE)

    def _attach_engine(self, cfg: dict, *, drain: bool = False) -> bool:
        """把事件驱动内核接到桥的协议事件上. 返回当前是否已接上.

        `drain=True` 只给 `_run_games` 在**开局前**用: 先把旧 spool 读完并清
        水位, 避免对历史牌谱一口气产出几十条决策. `_play_one_game` 里重复
        调用时不再 drain, 否则会把本场已经发生的事件吞掉.
        """
        if self._net is None:
            return False
        if getattr(self, "_engine", None) is None:
            from src.engine.hand import HandController
            self._engine = HandController(emit=self._engine_log,
                                          rng=random.Random(self._rng.random()),
                                          ev_provider=self._engine_ev_rows)
            self._engine_net = None
        if getattr(self, "_engine_net", None) is self._net:
            return True
        attached = self._net.attach_events(self._engine.on_event, drain=drain)
        self._engine_net = self._net
        self.log_info("  事件驱动内核已接入"
                      + ("" if attached else "（等 spool 出现后自动接）"))
        return attached

    def _engine_ev_rows(self):
        """出牌 EV 候选(协议事实 -> 引擎排序)."""
        if not tuning.EV_EVAL_TAKEOVER or self._net is None:
            return None
        try:
            from src.tasks.glue import _plan_ev
            rows, _wall = _plan_ev(self._net, getattr(self, "_cfg", None))
            return rows
        except Exception:  # noqa: BLE001 - EV 算不出来就回牌效
            return None

    def _engine_log(self, decision) -> None:
        bits = [decision.action]
        if decision.tile:
            bits.append(decision.tile)
        if decision.tiles:
            bits.append(",".join(decision.tiles))
        if decision.reason:
            bits.append(f"({decision.reason})")
        self.log_info("  [engine] " + " ".join(bits))
        try:
            self._record({"event": "engine",
                          "action": decision.action, "tile": decision.tile,
                          "tiles": list(decision.tiles),
                          "reason": decision.reason})
        except Exception:  # noqa: BLE001 - 记录失败不影响对局
            pass

    def _run_engine_game(self, cfg: dict, game: int, spec) -> bool:
        """用事件驱动内核跑完一局(执行档)."""
        from src.engine.locator import VisionLocator
        from src.engine.runner import EngineRunner, RunnerHooks

        if self._engine is None:
            self.log_error("事件驱动内核没接上（抓包不可用），本局不打")
            return False
        self.log_info("  [engine] 本局走事件驱动内核")
        self._engine.pop_pending()
        vision = self._ensure_vision()
        hooks = RunnerHooks(
            grab=lambda: self._grab(),
            at_ready=lambda frame: self._at_ready_page(spec, cfg, frame),
            result_page=lambda frame: self._handle_result_page(
                frame, cfg, spec, game, 0),
            hand_result=lambda frame: self._handle_hand_result(
                frame, cfg, game),
            click_box=lambda box, screen, target: self._engine_click(
                box, screen, target, cfg),
            should_stop=lambda: self._should_yield(),
            sleep=lambda seconds: self.sleep(seconds),
            log=lambda text: self.log_info(text),
            new_round=lambda frame: self._engine_new_round(frame, cfg),
            tick=lambda frame: self._engine_tick(frame, cfg),
            discard_delay=lambda: self._rng.uniform(
                *tuning.ENGINE_DISCARD_DELAY_RANGE),
            capture_alive=lambda: self._capture_watch(),
            timeout=tuning.GAME_TIMEOUT,
        )
        return EngineRunner(self._net, self._engine, VisionLocator(vision),
                            hooks).run()

    def _engine_click(self, box, screen: str, target: str, cfg: dict) -> None:
        """点完立刻把鼠标移回安全区.

        光标停在手牌上会触发雀魂的悬停/听牌展示, 挡按钮模板(2026-10-19
        用户实跑: 听牌框挡住杠按钮), 牌被顶起来后下一次点击也可能落空.
        `_park_mouse` 走的是贝塞尔轨迹, 不是瞬移.
        """
        self._click_box(box, screen, target, cfg)
        try:
            self._park_mouse(self._grab())
        except Exception as exc:  # noqa: BLE001 - 停鼠标失败不该停任务
            self.log_warning(f"  鼠标移回安全区失败（{type(exc).__name__}: {exc}）")

    def _engine_new_round(self, frame, cfg: dict) -> None:
        """新一手: 自动开关会被游戏复位, 下一 tick 重新点亮."""
        self._toggles_ok = False
        self._engine_toggle_at = 0.0

    def _engine_tick(self, frame, cfg: dict) -> None:
        if self._toggles_ok or self._engine is None:
            return
        if self._engine.phase not in ("waiting", "my_turn"):
            return
        now = time.time()
        if now - getattr(self, "_engine_toggle_at", 0.0) < 2.0:
            return
        self._engine_toggle_at = now
        # 刚进新一局时还有加载动画/等别人慢网络: 牌桌特征没出来之前点
        # 「和」只会点空. 先确认在牌桌, 再点自动开关.
        if not self._at_table(cfg, frame):
            self._log_repeat_once(
                "engine_not_table", "1",
                "  [engine] 还没进牌桌（加载/等别人），先不点自动开关")
            return
        try:
            self._ensure_auto_toggles(cfg, frame, False)
        except Exception as exc:  # noqa: BLE001 - 开关点不上不该停掉整局
            self.log_warning(f"  自动开关点亮失败（{type(exc).__name__}: {exc}），"
                             f"稍后重试")


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

    def _play_one_game(self, cfg: dict, game: int) -> bool:
        """打完一局, 回到准备页返回 True; 超时/异常返回 False."""
        spec = self._resolve_spec(cfg)
        self._ensure_vision()
        if self._net is None:
            self._net = self._build_net(cfg)
        attached = self._attach_engine(cfg)
        if self._engine is None or self._net is None or not attached:
            self.log_error("事件驱动内核没接上（抓包/spool 不可用），本局不打")
            return False
        self._cfg = cfg
        self._toggles_ok = False
        self._engine_toggle_at = 0.0
        self._repeat_log = {}
        return self._run_engine_game(cfg, game, spec)

    def _log_repeat_once(self, key: str, signature: str, message: str,
                         level: str = "info") -> None:
        """同一手牌里签名没变的同一条告警只打第一次, 避免轮询刷屏.

        签名一变(槽位/牌/张数变了)就立刻重报; 签名按手清空.
        """
        if self._repeat_log.get(key) == signature:
            return
        self._repeat_log[key] = signature
        (self.log_warning if level == "warning" else self.log_info)(message)

    def _handle_result_page(self, frame, cfg: dict, spec, game: int,
                            turns: int) -> bool:
        """处理终局结算页, 一直点到画面不再是结算页为止.

        一局结束可能连着好几页: 点数结算、终局排名、任务完成页, 以及没有
        确认按钮的「获得奖励」页. **每页用自己那一页的特征判**(见
        RESULT_PAGE_FEATURES): 「终局」标题 / 「确认」按钮 / 「获得奖励」.
        认出奖励页就直接在确认位点一下(它没有按钮, 点任意位置都能翻页);
        结算页能匹配到「确认」就点按钮.

        返回 True 表示这一局已经收尾, 调用方不必再读手牌.
        上限 RESULT_MAX_PAGES 防止页面认错时无限空点.

        终局后回大厅那一路**不用这里**(它是一串页面的连点, 见 _back_to_lobby);
        这里服务的是"开局前把残留的结算页点干净"(_drain_result_pages)与
        内核在局内的 result_page 钩子.
        """
        bench.count_state("result_page")  # bench 挂钩(默认关闭)
        pages = 0
        while pages < tuning.RESULT_MAX_PAGES:
            if self._locate(REWARD_SCREEN, REWARD_FEATURE, cfg, frame,
                            strict=True, verbose=False) is not None:
                # 奖励页没有「确认」, 等它只会白等满 RESULT_CONFIRM_TIMEOUT.
                pages += 1
                self.log_info(f"第 {game} 局已打完（认出奖励页第 {pages} 页）")
                nxt = self._click_result_confirm_spot(frame, game, pages)
                if nxt is None:
                    break
                frame = nxt
                continue
            box = (self._locate(RESULT_SCREEN, RESULT_FEATURE, cfg, frame,
                                strict=True, verbose=False)
                   or self._locate(RESULT_SCREEN, RESULT_CONFIRM, cfg, frame,
                                   strict=True, verbose=False))
            if box is None:
                break
            pages += 1
            self.log_info(f"第 {game} 局已打完（认出结算页第 {pages} 页）")
            confirm = self._wait_for_button(
                RESULT_SCREEN, RESULT_CONFIRM, cfg,
                timeout=tuning.RESULT_CONFIRM_TIMEOUT)
            if confirm is None:
                self.log_info(
                    f"  这一页没有「{RESULT_CONFIRM}」按钮，"
                    f"按点击翻页处理")
                nxt = self._click_result_confirm_spot(frame, game, pages)
                if nxt is None:
                    break
                frame = nxt
                continue

            delay = timing.idle_delay(self._rng)
            self.log_info(f"  想 {delay:.1f}s 后点 {RESULT_CONFIRM}")
            self.sleep(delay)
            self._click_box(confirm, RESULT_SCREEN, RESULT_CONFIRM, cfg)
            # confirm 是框架的 Box(center 是方法), 取中心统一走 _box_center.
            self._record({"event": "result_confirm", "game": game, "turn": turns,
                          "page": pages, "point": list(self._box_center(confirm))})
            self.sleep(tuning.RESULT_CONFIRM_GAP)
            # 用点击后的新帧判下一页: 拿点击前的画面会永远认为还有一页.
            nxt = self._grab()
            if nxt is None:
                break
            frame = nxt

        if pages == 0:
            return False
        if pages >= tuning.RESULT_MAX_PAGES:
            self.log_warning(
                f"连着处理了 {pages} 页(结算/奖励)还是没离开，"
                f"停止点击（上限 {tuning.RESULT_MAX_PAGES}），交给外层处理")
        self.log_info(f"第 {game} 局结束（结算/奖励页 {pages} 页），共 {turns} 轮")
        return True

    def _click_result_confirm_spot(self, frame, game: int, page: int):
        """在结算页「确认」的标注中心点一下, 返回点击后的新帧.

        用在没有「确认」按钮的页(「获得奖励」等)与终局后那串页面的连点
        (见 _back_to_lobby): 那一块有按钮时点中的就是按钮, 没有按钮时
        点页面任意位置也能翻页(用户实测), 所以同一个坐标两种页面都成立.
        """
        h, w = frame.shape[:2]
        px = round(tuning.RESULT_CONFIRM_CENTER[0] * w / 1920)
        py = round(tuning.RESULT_CONFIRM_CENTER[1] * h / 1080)
        delay = timing.idle_delay(self._rng)
        self.log_info(f"  {delay:.1f}s 后在确认位点一下翻页 ({px}, {py})")
        self.sleep(delay)
        self._ensure_foreground()
        self.click(x=px, y=py)
        self._record({"event": "result_click_through", "game": game,
                      "page": page, "point": [px, py]})
        self.sleep(tuning.RESULT_CONFIRM_GAP)
        return self._grab()

    def _drain_result_pages(self, cfg: dict, spec, game: int,
                            max_wait: float = 15.0) -> None:
        """把还留在屏幕上的结算页点干净(最多等 max_wait 秒).

        点掉一页后继续轮询: 下一页(终局排名 / 任务完成页)可能晚一拍才画
        出来, 那一帧没认出就会把页面留在屏幕上, 连打的"开始"按钮根本不在
        这一页. 每局之间、等"开始"之前先清一遍, 直到一页都没有为止.
        """
        deadline = time.time() + max_wait
        while time.time() < deadline:
            if self._should_yield():
                raise TaskStopped("清理结算页期间收到停止指令")
            frame = self._grab()
            if not self._handle_result_page(frame, cfg, spec, game, 0):
                return
            self.log_info("  又点掉一页结算页，看看还有没有下一页")

    def _handle_hand_result(self, frame, cfg: dict, game: int) -> bool:
        """认出一手结束的顺位结算画面就点「确认」, 返回 True 表示这轮别读手牌.

        背景与实测数值见 HAND_RESULT_SCREEN 常量的说明. 与整局的结算页
        (_handle_result_page)是两回事: 这个画面**每一手**结束都弹一次,
        「确认」按钮带未确认人数(动态), 模板只框静态的「确认」二字;
        流局画面没有这个按钮, 约 5s 自动过 —— 漏判无代价(下一轮照常等
        ActionNewRound), 所以探测用严格阈值且不做宽松重试(宁可漏判不误判).

        重复点击是安全的: 按钮在别人确认前一直显示, 人也是这么连点的;
        日志按手去重(_repeat_log 在换手时清空), 不会刷屏.
        """
        bench.count_state("hand_result")  # bench 挂钩(默认关闭)
        box = self._locate(HAND_RESULT_SCREEN, HAND_RESULT_CONFIRM, cfg, frame,
                           strict=True, verbose=False)
        if box is None:
            return False
        self._log_repeat_once(
            "hand_result", "1",
            "  一手结束（和了/流局顺位结算画面），点「确认」继续")
        delay = timing.idle_delay(self._rng)
        self.sleep(delay)
        self._click_box(box, HAND_RESULT_SCREEN, HAND_RESULT_CONFIRM, cfg)
        self._record({"event": "hand_result_confirm", "game": game,
                      "point": list(self._box_center(box))})
        self.sleep(tuning.RESULT_CONFIRM_GAP)
        return True

    def _press_start(self, spec, cfg: dict, game: int) -> None:
        """开下一局: 先清残留结算页, 再等"开始"并点击.

        等待期间每 20s 留一张现场帧, 卡在未标注页面时能直接看到停在哪.
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

    def _ensure_vision(self) -> GameVision:
        if self._vision is None:
            self._vision = GameVision()
            frame = self._grab()
            self._vision.warmup(frame.shape if frame is not None else (720, 1280, 3))
            self.log_info("YOLO 已预热")
        return self._vision

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
