from __future__ import annotations

"""创建友人场房间任务。

自动化流程（由房间规格生成，见 `src/rooms.py`）:

    0) 登录入口判定  -- 每天第一次登录雀魂会先停在"活动页", 平时直接进大厅。
                        先等两页之一出现, 再由流程第一步的 detect 特征决定
                        要不要点"返回大厅"。
    1) 活动页              -> 点 "返回大厅"（仅当探测到"活动标题"时执行）
    2) 大厅页              -> 点 "友人场"
    3) 友人场选择房间页     -> 点 "创建房间"
    4) 友人场创建房间页     -> 点人数按钮、时长按钮（都是开关型）、"创建房间"
    5) 准备页              -> 逐个点 "添加电脑N"

不点"开始"按钮，留给用户自己继续。

**房间类型是配置项 `room`，不是代码常量**（2026-10-06 用户新增四人东风标注后）:

| room            | 内容                       | 电脑位 |
|-----------------|----------------------------|--------|
| `friends_3p`    | 友人场 三人麻将 一局(默认) | 2      |
| `friends_4p`    | 友人场 四人麻将 东风局      | 3      |
| `ranked_silver_4p` | 段位场 银之间四人场半庄  | 0（真人对局）|

流程由 `build_room_flow(spec)` 按规格生成，"三人麻将" vs "四人麻将"、"一局" vs
"东风局"、准备页页名、加几个电脑全部随之改变 —— 换房间不用改这个文件。
写错 key 会立刻抛异常，不静默退回默认。

两类步骤, 处理方式不同:

    - 跳转型(默认): 按钮外观稳定, 用"模板匹配确认位置 -> 区域随机点"。
      匹配不到就中止, 绝不盲点。
    - 开关型(toggle=True): 选中/未选中外观完全不同(橙 vs 蓝), 模板只能覆盖
      未选中态, 所以改用"固定区域 + 颜色判断选中态": 已选中就跳过,
      没选中才点。这样任务可重复执行, 不会把已选中的按钮点掉。
      **四人东风的 "四人麻将" 和 "东风局" 都是开关型**, 漏点任何一个
      都开不出想要的局。

反检测设计（防止脚本点击被识别）:

- 鼠标移动: 点击前沿贝塞尔轨迹滑过去(带弧度 / 慢-快-慢的加减速 / 偶尔过点
  再修正), 不是瞬移。耗时随距离增长: 100px 约 0.15s, 跨屏约 0.55s。
  实现见 src/mouse_path.py(纯逻辑) 与 src/interactions.py(输入层)。
- 点击位置：区域内随机位置（不是中心点），带轻微高斯微抖动，模拟真人手抖
- 每次点击的延迟：0.1-0.4 秒随机；按下时长 0.04-0.11s 随机（不是写死的 10ms）
- 每步之间的等待：0.6-1.5 秒随机（模拟人读一下界面再动手）
- 点完之后小概率把光标随手挪开几像素（只移动不点击，零风险）
- 交互方式：真实鼠标事件（config 里 interaction=HumanLikePynput），需要游戏在前台；
  实测 PostMessage 对雀魂无效（uGUI 用真实光标位置做点击命中，不读消息坐标）
- 任务开始前会自动把窗口拉回屏幕内并置前台；运行期间请不要动鼠标 / 切窗口

关于"要不要加随机无意义操作"：这一段**不加**。建房间阶段服务器只看得到
"开了个房间"，没有可分析的行为；而页面上返回 / 关闭 / 设置就在附近, 乱点的
风险远大于收益。真正需要随机行为的是对局内（摸牌后鼠标在手牌上滑一滑、
偶尔多停一两秒），那个放到打牌任务里做。

安全策略（防止误点击导致损失）:

- 跳转型步骤必须用模板匹配确认按钮位置。匹配不到 -> 直接中止任务
- 每一步点击前都会先轮询等待该步的目标按钮出现（页面转场动画不会误判）

配置文件：configs/CreateFriendsRoomTask.json。界面上只有"人数/时长"两个下拉框,
其余阈值与延迟是代码常量, 全部在 `src/tasks/tuning.py`(理由见那个模块的说明)。

识别只有**一条路径**: ok-script 自带的 `self.find_one(category, box=...)`, 读
`ok_templates/coco_annotations.json`。标注由框架自带「模板」页签产出, 我们不再
自己实现标注器、不再有"导出成 COCO"的中间层、也没了第二套自研 matchTemplate
路径 —— 双路径并存正是 2026-10-07 那个"卡在步骤 0 两天"的 bug 的温床(默认路径
恰恰是唯一没被测到的路径)。详见 docs/ARCHITECTURE.md。
"""

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import win32api
import win32con
import win32gui
from ok import BaseTask
from ok.feature.Box import Box

from src.annotations import (
    UNANNOTATED_PREFIX,
    Annotation,
    category_name,
    frame_box,
    load_annotations,
    missing_categories,
    split_category,
)
from src.imageio import imread_unicode, imwrite_unicode
from src.instrumentation import bench
from src.paths import COCO_ANNOTATIONS, DEBUG_ARTIFACTS_DIR, TEMPLATES_DIR
from src.rooms import (LENGTH_OPTIONS, PLAYER_OPTIONS, LOGIN_ENTRY, RoomSpec,
                       Step, build_room_flow, ready_marker,
                       required_categories, spec_by_combination, spec_by_key)
from src.tasks import tuning
from src.vision import is_selected_orange
from src.widget_types import mark_unrenderable_hidden
from src.window import clamp_into_screen


# 待标注截图的 hint 会拼进文件名, 而 hint 里可能带决策 reason(含空格 / 括号),
# Windows 文件名不允许的字符在这里清掉.
_FILENAME_BAD = '\\/:*?"<>|'


def _safe_filename(hint: str) -> str:
    cleaned = hint
    for ch in _FILENAME_BAD:
        cleaned = cleaned.replace(ch, "_")
    return cleaned.strip()[:40] or "未命名"


class TaskStopped(Exception):
    """用户点了停止, 任务主动收工.

    为什么要单独一类: 等待类的函数超时也返回 None/False, 调用方只有靠异常
    才能区分"按钮真的没出现(要报错+留证)"和"用户喊停了(不该报错, 更不该
    再存一张没用的现场截图)".

    继承 Exception 而不是 BaseException: 框架要能正常捕获它并优雅收尾.
    """


# 登录后的入口判定: 命中哪个就在哪个页面, 都命中不了说明还没进入游戏界面.
# 这里只用来"等画面就绪", 真正的动作在流程的第一步(带 detect).
LOGIN_ENTRY_CHECK: list[tuple[str, str]] = LOGIN_ENTRY

# 界面两个下拉框的默认值: 三人 + 一局(与之前硬编码的房型一致, 别无理由改动).
#
# 注意这两个值**不是**房间 key, 而是"人数"和"时长"两个人话选项, 组合起来才是
# 房型. 换成别的是在界面上选, 或改下面两个常量, 不是改代码 --
# FLOW 在任务启动时按 self._spec 现场生成(见 run).
DEFAULT_PLAYERS = PLAYER_OPTIONS[0]
DEFAULT_LENGTH = LENGTH_OPTIONS[0]
DEFAULT_ROOM = ""          # 留空 = 由 players + length 推导

# ================================================================
# 默认配置(可在 configs/CreateFriendsRoomTask.json 覆盖)
# ================================================================
#
# **这里只放界面上要出现的项**(2026-10-07 用户决定).
# 框架会把 default_config 里的每一项都渲染成控件, 而 ok-script 没有
# "隐藏但保留配置"的机制 -- 所以内部调参是从配置里**拿出去**, 而不是
# 想办法隐藏, 全部搬在 src/tasks/tuning.py(照 ok-nte 的做法).
# 详见那个模块的说明.
_DEFAULT_CONFIG = {
    # ---- 房型: 两个下拉框, 组合出要开的房间 ----
    # 界面上是两个下拉框(见 __init__ 的 config_type 说明), 这里给默认值.
    # **值必须与 rooms.PLAYER_OPTIONS / LENGTH_OPTIONS 里的字符串逐字相等**,
    # 否则下拉框会显示空白(LabelAndDropDown 用 find_index_in_list 定位,
    # 找不到就是 -1). 改这里之前先看那两个常量.
    "players": DEFAULT_PLAYERS,   # "三人" / "四人"
    "length": DEFAULT_LENGTH,     # "一局" / "东风局"
    # room = players + length 推导出来的, **不要手填**.
    # 只在需要固定房型(比如命令行批量跑)时才写它, 写了就以它为准.
    # 它在界面上是 hidden 的 -- 见 __init__ 里的 config_type.
    "room": "",
}


class CreateFriendsRoomTask(BaseTask):
    """一键创建友人场房间(不点开始)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "创建友人场房间"
        self.description = (
            "从大厅出发：点友人场→创建房间→选人数与时长→加电脑（不开始）"
        )
        # **顺序要求**: _init_default_config() 必须在下面那段"自动标hidden"
        # 的循环**之前**调用 -- 循环要遍历 self.default_config 才能知道每个
        # 配置项的类型. BaseTask.__init__ 里它是空 dict, 不先填充的话循环
        # 什么都遍历不到(这正是第一次修没修干净的原因).
        self._init_default_config()
        # ---- 界面上的下拉框(ok-script 的 drop_down 控件)----
        #
        # 框架机制(2026-10-07 查证 ok-script 2.0.8 源码确认):
        #   任务实例的 `config_type` 字典里写 {"type": "drop_down", "options": [...]},
        #   界面就渲染成下拉框(ConfigCard -> ConfigItemFactory.config_widget ->
        #   LabelAndDropDown, 见 ok/ui/qt/tasks/).
        #
        # **选项值必须和 _DEFAULT_CONFIG 里的默认值逐字相等**, 否则下拉框显示空白:
        # LabelAndDropDown 用 find_index_in_list(options, 当前值, -1) 定下标,
        # 找不到就是 -1, 框里什么都不显示(实测确认). 所以:
        #   - 一律用**字符串**, 不用 int(用 int 3 配 options ["3","4"] 会 -1);
        #   - 默认值必须是 options 里的一项.
        #
        # 为什么拆成"人数"和"时长"两个而不是一个"房型"下拉: 用户要的就是
        # 「四人 or 三人」「一局 or 东风」这两个维度分开选. 组合的有效性由
        # rooms.py 的 ROOM_SPECS 决定 -- 没注册的组合会明确报错, 而不是
        # 悄悄开出一个错的局(比如三人麻将本来就没有东风局).
        self.config_type = {
            "players": {"type": "drop_down", "options": list(PLAYER_OPTIONS)},
            "length": {"type": "drop_down", "options": list(LENGTH_OPTIONS)},
            # 房型是 players+length 推导出来的, 不是用户直接选的东西.
            # hidden 让它不渲染成控件, 但值仍会从配置文件读到(兼容旧的
            # configs/*.json 里已经写好的 room 字段).
            "room": {"type": "drop_down", "options": [], "hidden": True},
        }
        # **所有框架渲染不了的配置项必须标hidden**(2026-10-07 实测踩坑,
        # 直接让整个程序开不了):
        #   框架的 `config_widget`(ok/ui/qt/tasks/ConfigItemFactory.py)在
        #   类型分支里只认 bool / int / float / str / list(-> ModifyListItem)
        #   以及 config_type 里显式声明的类型; 遇到 **tuple 直接抛**
        #       ValueError: invalid type <class 'tuple'>, value (0.25, 0.7)
        #   而这个异常发生在 TaskCard 构造里 -> 主窗口 `show_main_window()`
        #   直接炸掉, **整个程序打不开**(日志里的完整栈: MainWindow ->
        #   OneTimeTaskTab.refresh_ui -> TaskCard -> ConfigCard.__addConfig
        #   -> config_widget).
        #
        # 为什么以前一直没炸: 这些 tuple 是"任务内部用的参数", 在
        # `self.default_config` 一直是 {} 的那段时间里, 界面**根本没渲染
        # 任何控件**(见 _init_default_config 的说明), 于是这段代码路径
        # 从没被执行过. 一旦 default_config 非空, 全部配置项都会被渲染,
        # 雷就炸了. **教训: 填 default_config 之前必须先确认每个值都是
        # 框架认识的类型.**
        #
        # 这里是**按数据自动生成**而不是手写清单: 新增一个 tuple 型配置项时
        # 会被自动隐藏, 不用记得回来补一行(手写清单漏一条就又开不了机).
        # 判定规则本身在 src/widget_types.py, 有单测;这里只负责调用.
        mark_unrenderable_hidden(self.config_type, self.default_config)
        # 界面上就只剩房型那两个下拉框. 注意它们**不是被隐藏的** --
        # _DEFAULT_CONFIG 里只有players / length / room 三个键, room 已
        # 显式声明为 hidden, 界面自然只有两个. 内部调参压根不在配置里,
        # 见 _DEFAULT_CONFIG 上方的说明与 src/tasks/tuning.py.
        # 标注: 类目名("页名/区域描述") -> Annotation. 只有一份真源, 见
        # src/annotations.py 的模块说明.
        self._annotations: dict[str, Annotation] = {}
        self._jitter_sigma: float = 2.0
        self._stopped: bool = False
        # 房间规格与由它生成的流程. run() 里按配置选规格后现场生成,
        # 所以换房间类型不用改代码, 也不存在"上一局的 FLOW 残留".
        self._spec: RoomSpec = spec_by_combination(DEFAULT_PLAYERS, DEFAULT_LENGTH)
        self._flow: list[Step] = []

    # ---------------------------------------------------------------
    # 入口
    # ---------------------------------------------------------------

    def run(self):
        super().run()

        cfg = self._load_task_config()
        self.log_info(f"配置: {cfg}")
        self._log_interaction()

        # 先定规格再校验: 校验和流程都依赖 self._spec / self._flow.
        # 房型由两个下拉框(players + length)组合而来; 老的 room 字段如果还在
        # 配置文件里(兼容旧 configs/*.json), 以它为准 -- 否则用户之前手改过
        # room 的配置会在升级后被悄悄改回默认房型.
        self._spec = self._resolve_spec(cfg)
        self._flow = build_room_flow(self._spec)
        self.log_info(f"房间规格: {self._spec.key} = {self._spec.label}"
                      f"（{len(self._flow)} 步到准备页）")

        self._load_annotations(COCO_ANNOTATIONS)
        if not self._validate_flow():
            return

        t0 = time.time()
        try:
            self._ensure_window_ready()
            self._enter_lobby(cfg)
            self._run_flow(cfg)
            # 段位场没有准备页: 点完"四人东"就自动匹配, ready_marker
            # (对局开始页/胜负)由打牌任务的匹配等待去认, 这里跳过自检
            # -- 否则它会拿 PAGE_WAIT_TIMEOUT(秒级)去等分钟级的匹配.
            if not self._spec.ranked:
                self._verify_ready_page(cfg)
            # 钩子: 打牌任务在这里接着点"开始"并打完一局. 建房任务什么都不做.
            self._after_room_ready(cfg)
            self.log_info(f"创建房间任务完成，用时 {time.time() - t0:.1f}s")
        except TaskStopped as e:
            # 用户主动喊停, 不是失败. 不记 error 也不往上抛, 否则日志里会留一条
            # 红色的"任务失败", 让人误以为出了故障.
            self.log_info(f"任务已按用户要求停止（{e}）")
        except Exception as e:
            self.log_error(f"任务失败: {e}")
            raise

    # ---------------------------------------------------------------
    # 房型解析
    # ---------------------------------------------------------------

    def _resolve_spec(self, cfg: dict) -> RoomSpec:
        """按配置算出要开什么房间.

        优先级: 显式的 room > players + length 组合. room 默认是空串(界面上
        hidden), 只有需要固定房型时才手填. **组合非法直接抛**, 不静默回退 --
        开出一个错的局比崩掉更糟.
        """
        room = cfg.get("room")
        if room:
            return spec_by_key(room)
        return spec_by_combination(
            cfg.get("players") or DEFAULT_PLAYERS,
            cfg.get("length") or DEFAULT_LENGTH,
        )

    # ---------------------------------------------------------------
    # 窗口就位(真实鼠标点击的前提)
    # ---------------------------------------------------------------

    def _hwnd_window(self):
        return self.executor.device_manager.hwnd_window

    def _ensure_window_ready(self) -> None:
        """真实鼠标点击的两个前提: 窗口完整在屏幕内 + 窗口在前台.

        - 雀魂的 uGUI 用真实系统光标位置做点击命中, PostMessage 无效,
          所以必须用真实鼠标(Pynput); 真实光标到不了屏幕外, 窗口一旦
          被拖出屏幕(哪怕只有目标按钮那一半), 点击就会落在屏幕外.
        - PynputInteraction 的 clickable() 要求窗口前台, 否则点击被静默跳过.
        """
        hwnd_window = self._hwnd_window()
        if hwnd_window is None:
            raise RuntimeError("找不到游戏窗口，请确认雀魂已启动")

        if self._pull_window_into_screen(hwnd_window):
            self.log_info("游戏窗口原本超出屏幕范围，已自动拉回屏幕内")
        self._ensure_foreground()
        self.log_info("窗口就位: 在屏幕内且已置前台")

    def _pull_window_into_screen(self, hwnd_window) -> bool:
        """窗口外框若超出虚拟屏幕, 平移回屏幕内. 返回是否移动过."""
        hwnd = hwnd_window.hwnd
        screen = (
            win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN),
            win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN),
            win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN),
            win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN),
        )
        target = clamp_into_screen(win32gui.GetWindowRect(hwnd), screen)
        if target is None:
            return False
        win32gui.SetWindowPos(
            hwnd, 0, target[0], target[1], 0, 0,
            win32con.SWP_NOSIZE | win32con.SWP_NOZORDER,
        )
        self.sleep(0.3)
        return True

    def _ensure_foreground(self) -> None:
        """确保游戏窗口在前台; 不在前台就带上(真实点击需要)."""
        hwnd_window = self._hwnd_window()
        if hwnd_window is None:
            return
        for _ in range(3):
            # 虽然只有 3 次(最长 1.5 秒), 但规则统一成"所有带 sleep 的循环
            # 都先问一次停止", 比在这里判"这次够短所以不用问"更好维护.
            if self._should_yield():
                return
            if hwnd_window.is_foreground():
                return
            hwnd_window.bring_to_front()
            self.sleep(0.5)
        if not hwnd_window.is_foreground():
            raise RuntimeError(
                "无法把雀魂带到前台；真实鼠标点击需要游戏窗口在前台，"
                "任务运行期间请不要切换窗口"
            )

    def _log_interaction(self) -> None:
        try:
            interaction_cls = self.executor.device_manager.win_interaction_class
            if interaction_cls:
                self.log_info(f"Interaction: {interaction_cls.__name__}")
            else:
                self.log_info("Interaction: 未设置")
        except Exception as e:
            self.log_info(f"Interaction: 未知 ({e})")

    def _validate_flow(self) -> bool:
        """启动前校验标注齐全性, 缺东西就一次性报清楚, 不要跑到一半才炸.

        条件步(带 detect 的步)缺标注只警告不阻塞: 最坏情况是每天首次登录时
        任务停在活动页, 由后续步骤的等待超时报错, 不会误点.
        """
        required: list[str] = []
        for step in self._flow:
            if step.detect and not self._has_region(step.screen, step.detect):
                continue
            required.append(category_name(step.screen, step.target))
        required.extend(required_categories(self._spec))

        missing = missing_categories(required, set(self._annotations))

        if not self._has_region("活动页", "活动标题"):
            self.log_warning(
                "缺少 '活动页/活动标题' 标注，将跳过"
                "'每日首次登录停在活动页'的处理"
            )

        if missing:
            self.log_error(
                f"标注不完整，任务未启动。缺少类目: {missing}。"
                f"请在游戏界面里点「模板」页签截图并框出这些区域"
                f"（类目名必须是 '页名/区域描述'，与流程里的字符串完全一致），"
                f"保存到 {COCO_ANNOTATIONS}。"
            )
            return False
        return True

    def _has_region(self, screen_path: str, keyword: str) -> bool:
        return category_name(screen_path, keyword) in self._annotations

    # ---------------------------------------------------------------
    # 流程
    # ---------------------------------------------------------------

    def _enter_lobby(self, cfg: dict) -> None:
        """等画面进入"活动页"或"大厅页", 确认游戏已登录.

        这里不做任何点击 -- 是否需要从活动页返回大厅, 由 FLOW 第一步的
        detect 特征决定.
        """
        candidates = [(s, kw) for s, kw in LOGIN_ENTRY
                      if self._has_region(s, kw)]
        self.log_info(f"步骤 0: 等待游戏进入可用页面，候选 {candidates}")
        if self._wait_for_any(candidates, cfg) is None:
            raise RuntimeError(
                "等待页面超时：既不在大厅页也不在活动页，请确认游戏已登录并进入界面"
            )

    def _run_flow(self, cfg: dict) -> None:
        """按 self._flow 依次操作, 最后停在准备页(不点开始)."""
        total = len(self._flow)
        ready_checked = False
        for i, step in enumerate(self._flow, start=1):
            # 点了停止就别再往下点按钮了 -- 流程每一步都是真实鼠标点击,
            # 停下来之后还继续点会误操作游戏(见 _should_yield 的说明).
            if self._should_yield():
                self.log_info("收到停止指令, 中止建房流程")
                return
            label = f"步骤 {i}/{total}: [{step.screen}] 点击 '{step.target}'"

            if step.detect and not self._detect_present(step.screen, step.detect, cfg):
                self.log_info(f"{label} -- 跳过（未检测到 '{step.detect}'）")
                continue

            # 第一次碰到准备页时, 先确认画面上的准备页确实是规格指定的那一页.
            # 为什么必须做(2026-10-06 新增四人房后发现的真风险):
            # 三人/四人准备页的"添加电脑1/2"和"开始"按钮**在两个页面上
            # 长得一模一样且坐标几乎重合**, 实测模板互相匹配 **1.000**. 所以
            # 一旦页数选错(比如配置 friends_4p 但游戏里其实是三人房),
            # 模板匹配会**愉快地匹配到自己那个框**然后照点不误 -- 点下去
            # 才发现加错了电脑数, 而这时已经点过了"创建房间".
            # 用准备页标识(静态规则说明块)前置校验, 是唯一能在点下去之前
            # 发现选错房型的时机.
            if not ready_checked and step.screen == self._spec.screens.ready:
                self._confirm_ready_page_before_click(cfg)
                ready_checked = True

            self.log_info(label)
            if step.toggle:
                self._click_toggle(step.screen, step.target, cfg)
            else:
                self._click_button(step.screen, step.target, cfg)

            if i < total:
                self._wait_human_delay(cfg)

        if self._spec.ranked:
            self.log_info("段位场排队流程执行完毕，等待自动匹配开局")
        else:
            self.log_info("已停在准备页（未点开始），请手动确认")

    def _confirm_ready_page_before_click(self, cfg: dict) -> None:
        """点准备页按钮之前, 先确认当前画面的准备页是规格指定的那一个.

        拿规格里所有已知的准备页互相比一遍: 必须命中自己那个, 且不能是
        别人那个. 认不出就报错退出 -- 宁可不开房, 也不能开错人数的房.
        """
        want = self._spec.screens.ready
        if not self._has_region(want, self._spec.screens.ready_marker):
            self.log_warning(
                f"准备页 '{want}' 缺少页面特征"
                f" '{self._spec.screens.ready_marker}'，跳过房型前置校验"
                f"（仍会按配置点击，但若页数选错可能点错按钮）")
            return

        candidates = []
        marker = self._spec.screens.ready_marker
        for annotation in self._annotations.values():
            if annotation.keyword == marker and annotation.page:
                candidates.append((annotation.page, marker))
        candidates = sorted(set(candidates))
        if len(candidates) < 2:
            self.log_info("只有一种准备页标注，房型前置校验跳过")
            return

        hit = self._wait_for_any(candidates, cfg)
        if hit is None:
            raise RuntimeError(
                f"没认出任何准备页（试过 {[c[0] for c in candidates]}），"
                f"房型无法确认，任务中止")
        if hit[0] != want:
            raise RuntimeError(
                f"房间配置是 '{self._spec.key}'（{self._spec.label}），"
                f"应该停在 '{want}'，但画面上是 '{hit[0]}'。"
                f"两人准备页的『添加电脑』『开始』按钮长得一样且互相匹配 1.000，"
                f"继续点下去会加错电脑数。请检查 room 配置，"
                f"或先在游戏里退回到正确的房间再运行。")

        self.log_info(f"房型校验通过: 画面确为 '{want}'")

    def _after_room_ready(self, cfg: dict) -> None:
        """房间就绪后的钩子. 基类什么都不做, 打牌任务(子类)在这里开局."""

    def _default_config(self) -> dict:
        """默认配置. 子类(打牌任务)覆写这个来加自己的参数."""
        return dict(_DEFAULT_CONFIG)

    def _init_default_config(self) -> None:
        """把 _default_config() 的结果写进**框架认的那个**属性上.

        **这个赋值不能省**(2026-10-07 实测踩坑): ok-script 的界面读的是
        `task.default_config`(基类在 `BaseTask.__init__` 里初始化为 {}),
        传给 `TaskCard` -> `ConfigCard._init_config_content` ->
        `__initWidget`. 那里第 56 行是

            if not self.config or not (self.config.has_user_config()
                                       or self.default_config
                                       or self.config_type):
                self._on_empty_config_content()      # 整块内容不渲染

        `self.config` 是**用户配置文件**的内容, 我们的是空 `{}`; 于是
        `has_user_config()` 走 all() 空迭代返回 True 救不回来,
        只能靠 `self.default_config` 非空. 而它若一直是 {}, 三个条件全不成立
        -> 走"空配置"分支 -> **任务卡片根本不展开, 一个控件都没有,
        只剩标题和"开始"按钮**. 这正是用户看到的现象.

        也就是说: `config_type` 里的 `drop_down` 声明再对, 只要
        `default_config` 没填, 界面就一个下拉框都不会画.
        """
        self.default_config = self._default_config()

    def _verify_ready_page(self, cfg: dict) -> None:
        """收尾自检: 确认停在准备页, 且所有电脑位都加上了."""
        spec = self._spec
        # 用静态特征判定"在准备页", 不用开始按钮 -- 见 rooms.RoomScreens.
        # ready_marker 的说明: 开始按钮跨帧稳定分数 0.836 < 探测阈值 0.88,
        # 拿它判页面等于永远判不出来, 实跑会白等满 page_wait_timeout.
        marker = ready_marker(spec)
        if not self._has_region(*marker):
            self.log_warning(
                f"准备页缺少页面特征 '{marker[1]}'，"
                f"无法自动判定是否停在准备页（不影响建房，只影响开局）")
        elif self._wait_for_any([marker], cfg) is None:
            self.log_warning("未检测到准备页特征，请手动确认房间状态")
            return

        ready_page = spec.screens.ready
        remaining = [
            kw
            for kw in spec.screens.bots
            if self._has_region(ready_page, kw)
            and self._detect_present(ready_page, kw, cfg)
        ]
        if remaining:
            self.log_warning(f"仍有未添加的电脑位 {remaining}，请手动确认")
        else:
            self.log_info(f"{len(spec.screens.bots)} 个电脑位均已添加，"
                          f"停在准备页（未点开始）")

    # ---------------------------------------------------------------
    # 等待页面 / 按钮出现
    # ---------------------------------------------------------------

    def _wait_for_any(
        self, candidates: list[tuple[str, str]], cfg: dict
    ) -> Optional[tuple[str, str]]:
        """轮询若干(页, 区域)特征, 返回第一个命中的 (页, 区域); 超时返回 None.

        用户喊停时抛 TaskStopped 而不是返回 None -- 返回 None 会被调用方当成
        "页面没出现", 于是报错甚至再存一张没用的现场截图.
        """
        deadline = time.time() + tuning.PAGE_WAIT_TIMEOUT
        while True:
            if self._should_yield():
                raise TaskStopped("等待页面期间收到停止指令")
            frame = self._grab()
            for screen, keyword in candidates:
                if not self._has_region(screen, keyword):
                    continue
                if self._locate(screen, keyword, cfg, frame, strict=True, verbose=False):
                    self.log_info(f"    检测到页面 [{screen}]（特征 '{keyword}'）")
                    return screen, keyword
            if time.time() >= deadline:
                return None
            self.sleep(tuning.PAGE_POLL_INTERVAL)

    def _wait_for_button(self, screen: str, keyword: str, cfg: dict,
                         timeout: Optional[float] = None) -> Optional[Box]:
        """轮询等待目标按钮出现, 返回匹配到的 Box; 超时返回 None.

        ``timeout`` 覆盖默认的 `tuning.PAGE_WAIT_TIMEOUT`(秒). 少数步骤需要
        更长的等待 -- 比如点完"创建房间"后的"开始"按钮, 建房+进场的转场
        比普通页面跳转久(实测有一次30s 都匹配不上). **传参**而不是改配置项,
        因为这类"某一步特别慢"是流程事实, 不是用户要调的参数.

        等待期间每隔几秒把当前最高相似度打进日志: 按钮"样子变了但还在"和
        "按钮根本不在这一页"这两种情况最后都会超时, 只有分数能提前区分.

        超时时留一张现场截图(data/debug/, 不是标注目录) -- 等按钮这件事
        实跑踩过好几次白等(开始按钮等满 45 秒), 光看日志只有一句"超时",
        没有画面根本没法判断是"按钮样子变了"还是"压根不在这一页".

        用户喊停时抛 TaskStopped(见 _wait_for_any 的说明).
        """
        timeout = tuning.PAGE_WAIT_TIMEOUT if timeout is None else timeout
        deadline = time.time() + timeout
        next_score_log = time.time() + 5.0
        while True:
            if self._should_yield():
                raise TaskStopped(f"等待 '{screen}/{keyword}' 期间收到停止指令")
            frame = self._grab()
            box = self._locate(screen, keyword, cfg, frame, verbose=False)
            if box is not None:
                self.log_info(
                    f"    匹配成功 '{keyword}' -> ({box.x},{box.y}) "
                    f"{box.width}x{box.height} conf={box.confidence:.3f}"
                )
                return box
            now = time.time()
            if now >= next_score_log:
                next_score_log = now + 5.0
                best = self._best_match(screen, keyword, cfg, frame)
                self.log_info(
                    f"    还在等 '{keyword}'"
                    + (f"，当前最高相似度 {best[0]:.3f}" if best else "")
                    + f"（阈值 {tuning.TEMPLATE_THRESHOLD}）"
                )
            if now >= deadline:
                self._save_wait_timeout_frame(frame, screen, keyword)
                return None
            self.sleep(tuning.PAGE_POLL_INTERVAL)

    def _save_wait_timeout_frame(self, frame, screen: str, keyword: str) -> None:
        """等按钮超时时把现场帧存到 data/debug/, 供事后核对.

        存**调试目录**而不是标注目录: 这是"等超时"的证据, 不该混进标注网格里
        (要补标注的截图走 _save_unknown_page, 它带 `待标注_` 前缀)。
        """
        if frame is None:
            return
        try:
            DEBUG_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%m%d_%H%M%S")
            path = DEBUG_ARTIFACTS_DIR / f"等待超时_{_safe_filename(keyword)}_{ts}.png"
            imwrite_unicode(path, frame)
            self.log_info(f"已存等待超时的现场画面: {path}")
        except Exception as e:  # noqa: BLE001
            self.log_warning(f"存现场画面失败: {e}")

    def _detect_present(self, screen: str, keyword: str, cfg: dict) -> bool:
        """判断某个页面特征是否出现在当前画面(严格阈值, 不做宽松重试)."""
        if not self._has_region(screen, keyword):
            return False
        return self._locate(
            screen, keyword, cfg, self._grab(), strict=True, verbose=False
        ) is not None

    # ---------------------------------------------------------------
    # 点击
    # ---------------------------------------------------------------

    def _click_button(self, screen: str, keyword: str, cfg: dict,
                       timeout: Optional[float] = None) -> None:
        """跳转型按钮: 等它出现 -> 模板匹配确认位置 -> 区域内随机点.

        模板匹配失败 = 不确定按钮在哪 = 直接中止, 绝不盲点.

        ``timeout`` 透传给 `_wait_for_button`, 见那里的说明.
        """
        box = self._wait_for_button(screen, keyword, cfg, timeout=timeout)
        if box is None:
            self._diagnose_missing(screen, keyword, cfg)
            raise RuntimeError(
                f"等待 '{screen}/{keyword}' 出现超时（{timeout}s），"
                f"已中止任务"
            )
        self._click_box(box, screen, keyword, cfg)

    def _diagnose_missing(self, screen: str, keyword: str, cfg: dict) -> None:
        """等不到按钮时, 把"到底差多少"说清楚, 并留一张待标注截图.

        匹配失败只有两种可能, 得分能直接区分:

        - 标注区域内得分也不高 -> 按钮样子变了(选中态 / 禁用态 / 换了文案),
                                需要在「模板」页签里按当前状态重标这个区域;
        - 区域内低但全图高     -> 按钮挪位置了, 标注框要挪.

        两种情况都把当前帧存成待标注截图(带 `待标注_` 前缀, 直接落在标注目录
        里), 打开「模板」页签就能看到它并补框.
        """
        frame = self._grab()
        if frame is None:
            return
        near = self._best_match(screen, keyword, cfg, frame)
        far = self._best_match(screen, keyword, cfg, frame, whole_frame=True)
        where = f" @ {far[1]}" if far else ""
        self.log_warning(
            f"    诊断 '{screen}/{keyword}': 标注区域内最高相似度 "
            f"{'%.3f' % near[0] if near else 'N/A'}"
            f"（阈值 {tuning.TEMPLATE_THRESHOLD}）, "
            f"全图最高 {round(far[0], 3) if far else 'N/A'}{where}"
        )
        self._save_unknown_page(frame, f"{screen}_{keyword}_未匹配")

    def _best_match(self, screen: str, keyword: str, cfg: dict, frame,
                    whole_frame: bool = False
                    ) -> Optional[tuple[float, tuple[int, int]]]:
        """该区域(或整帧)内的最高相似度与位置, **不看阈值**. 诊断用.

        框架在阈值不达标时什么都不返回, 所以这里把阈值压到 0.01 再问一次 --
        只为拿到"差多少"这个数字, 不参与任何点击决策.
        """
        category = category_name(screen, keyword)
        if category not in self._annotations:
            return None
        if whole_frame:
            box = Box(0, 0, frame.shape[1], frame.shape[0])
        else:
            box = self._search_box_for(screen, keyword, cfg, frame)
            if box is None:
                return None
        try:
            found = self.find_one(category, box=box, threshold=0.01, frame=frame)
        except Exception:  # noqa: BLE001 - 纯诊断, 失败不影响主流程
            return None
        if not found:
            return None
        return float(found.confidence), (found.x, found.y)

    def _click_toggle(self, screen: str, keyword: str, cfg: dict) -> None:
        """开关型按钮: 位置固定, 靠颜色判断选中态, 已选中则跳过.

        这类按钮(三人麻将 / 四人麻将 / 一局 / 东风局)选中后底色由蓝变橙,
        从"未选中"截图裁出的模板匹配不上选中的按钮, 所以不能沿用
        "匹配不到就中止"的策略. 流程:

        1. 按标注区域 + 分辨率换算得到当前帧里的按钮矩形;
        2. 颜色判断: 已经是橙色(选中) -> 跳过, 保证任务可重复执行;
        3. 否则用模板匹配确认按钮确实在预期位置(防止页面不对时误点);
        4. 匹配到才点, 匹配不到直接中止.

        **模板只能覆盖"标注时的那个状态"**(2026-10-06 新增四人房后确认):
        用户标注"四人麻将"时游戏里已经是四人, 所以那份模板是**橙色选中态**;
        而"三人麻将"那份是蓝色未选中态. 两份模板各自只在自己那个状态下
        能匹配上. 所以步骤 3 匹配不到时, 错误信息要说清是"标注状态与当前
        需要的相反", 而不是笼统报"模板匹配不到" -- 后者会让人以为是分辨率
        或搜索区的问题, 方向完全错了.
        """
        annotation = self._annotations.get(category_name(screen, keyword))
        if annotation is None:
            raise RuntimeError(f"缺少标注 '{screen}/{keyword}'，无法判断按钮状态")
        frame = self._grab()
        if frame is None:
            raise RuntimeError("截图失败，无法判断按钮状态")

        box = frame_box(annotation.bbox, annotation.image_size,
                        (frame.shape[1], frame.shape[0]))
        if is_selected_orange(frame, box, tuning.SELECTED_ORANGE_RATIO):
            self.log_info(f"    '{keyword}' 已是选中态，跳过")
            return

        matched = self._locate(screen, keyword, cfg, frame, verbose=False)
        if matched is None:
            raise RuntimeError(
                f"'{screen}/{keyword}' 既不是选中态、模板也匹配不到。"
                f"最可能的原因是**标注时的选中态与现在需要的相反**: "
                f"开关型按钮选中后是橙色, 而模板只能覆盖标注那一刻的状态。"
                f"请用「模板」页签在按钮处于"
                f"{'未选中' if self._template_is_selected_orange(screen, keyword) else '选中'}"
                f"态时重标这个区域。"
                f"已中止任务，避免点错。"
            )
        self.log_info(
            f"    确认按钮位置 ({matched.x},{matched.y}) {matched.width}x{matched.height} "
            f"conf={matched.confidence:.3f}"
        )
        self._click_box(matched, screen, keyword, cfg)

    def _template_is_selected_orange(self, screen: str, keyword: str) -> bool:
        """标注截图上这个开关按钮是不是橙色的(选中态).

        只读标注目录里的截图, 不碰游戏窗口. 用来在报错时告诉用户
        "你的模板是选中态还是未选中态", 省得他自己去猜.
        """
        try:
            annotation = self._annotations.get(category_name(screen, keyword))
            if annotation is None or not annotation.image:
                return False
            ann_img = imread_unicode(TEMPLATES_DIR / annotation.image)
            if ann_img is None:
                return False
            return is_selected_orange(ann_img, annotation.bbox, 0.5)
        except Exception:  # noqa: BLE001 - 这只是个诊断信息, 失败不影响主流程
            return False

    def _click_box(self, box: Box, screen: str, keyword: str, cfg: dict) -> None:
        """在 Box 内随机取点并点击, 带手抖与随机延时."""
        self._ensure_foreground()
        point = self._random_point_in_box(
            box, tuning.SAFE_MARGIN_RATIO, tuning.CLICK_RANGE
        )
        if tuning.SAVE_DEBUG_ARTIFACTS:
            self._save_click_artifact(point, cfg)
        self.click(x=point[0], y=point[1])
        self.log_info(f"    点击 [{screen}] '{keyword}' -> ({point[0]}, {point[1]})")
        self.sleep(random.uniform(0.1, 0.4))

    def _box_center(self, box: Box) -> Tuple[int, int]:
        """框中心的整数像素坐标, 对框架 Box 和自研 Detection 都成立.

        **为什么不用现成的取中心接口**: 两种框各有一个, 但形态不一样 ——
        框架 `ok.Box.center` 是**方法**(得写 `box.center()`), 而
        `src.detection.Detection.center` 是 **property**(写 `box.center`)。
        写错任一个都是运行时才炸: 实跑在结算页 `confirm.cx` 上抛
        `AttributeError: 'Box' object has no attribute 'cx'`, 整局被判成失败
        (2026-10-07)。两者唯一共有的东西是 `x / y / width / height`, 所以
        这里只依赖那四个, 谁传进来都对。
        """
        return (round(box.x + box.width / 2.0),
                round(box.y + box.height / 2.0))

    def _random_point_in_box(
        self, box: Box, safe_margin_ratio: float, click_range: Tuple[float, float]
    ) -> Tuple[int, int]:
        """在 Box 区域内生成一个随机点击位置(带手抖抖动).

        - 用 [lo, hi] 区间做均匀随机(模拟人点击时的分布, 不总是中心)
        - 叠加轻微高斯抖动(模拟手抖)
        - 强制在 [x+safe, x+w-safe] 范围内(防止点到边缘外)
        """
        x, y, w, h = box.x, box.y, box.width, box.height
        safe = int(min(w, h) * safe_margin_ratio)
        safe_x = max(0, min(safe, w // 2))
        safe_y = max(0, min(safe, h // 2))

        lo, hi = click_range
        ox = int(random.uniform(lo, hi) * w + random.gauss(0, self._jitter_sigma))
        oy = int(random.uniform(lo, hi) * h + random.gauss(0, self._jitter_sigma))
        ox = max(safe_x, min(w - 1 - safe_x, ox))
        oy = max(safe_y, min(h - 1 - safe_y, oy))
        return x + ox, y + oy

    def _wait_human_delay(self, cfg: dict) -> None:
        """步骤之间的随机等待."""
        lo, hi = tuning.STEP_DELAY_RANGE
        d = random.uniform(lo, hi)
        self.sleep(d)
        self.log_info(f"    等待 {d:.2f}s")

    # ---------------------------------------------------------------
    # 模板匹配
    # ---------------------------------------------------------------

    def _locate(
        self,
        screen_path: str,
        keyword: str,
        cfg: dict,
        frame,
        strict: bool = False,
        verbose: bool = True,
        strict_threshold: float | None = None,
    ) -> Optional[Box]:
        """在当前帧里定位标注区域描述对应的按钮.

        只有一条路径: ok-script 的 FeatureSet(即 `self.find_one`), 读
        `ok_templates/coco_annotations.json`。

        - **模板的缩放由框架自己做**(read_from_json 里按 COCO 声明的
          width/height 对帧尺寸做 adjust_coordinates); 实测 1080p 标注在
          720p 帧上仍命中 0.993, 所以我们不再自己缩放模板。
        - **搜索区必须自己算**: 框架在 `box=None` 时只按 variance 外扩几个
          像素(见 FeatureSet.find_one_feature), 对我们的场景太窄。

        strict=True 用更严的探测阈值且不做宽松重试(页面探测用: 宁可漏判也不
        误判, 否则会在错误的页面上点按钮)。
        """
        if frame is None:
            return None
        category = category_name(screen_path, keyword)
        if category not in self._annotations:
            self.log_debug(f"    没有 '{category}' 这个标注")
            return None
        box = self._search_box_for(screen_path, keyword, cfg, frame)
        if box is None:
            return None

        thresholds = ([(strict_threshold or tuning.DETECT_THRESHOLD)]
                      if strict else
                      [tuning.TEMPLATE_THRESHOLD,
                       tuning.TEMPLATE_RETRY_THRESHOLD])
        for threshold in thresholds:
            try:
                found = self.find_one(category, box=box, threshold=threshold,
                                      frame=frame)
            except Exception as e:  # noqa: BLE001
                # 类目不在 COCO 里(标注没保存 / 名字打错)时框架会抛。**不要**
                # 在这里静默回退到别的匹配方式 -- 那正是 2026-10-07「卡在
                # 步骤 0 两天」的成因: 默认路径出问题却没人看见。直接报出来。
                self.log_error(
                    f"    框架匹配 '{category}' 失败: {e}；"
                    f"请确认这个类目在 {COCO_ANNOTATIONS} 里")
                return None
            if found is not None:
                if verbose and threshold != thresholds[0]:
                    self.log_info(f"    匹配成功（宽松阈值 {threshold}）'{keyword}'")
                return found

        if verbose:
            self.log_debug(f"    匹配未达标 '{keyword}'（阈值 {thresholds}）")
        return None

    def _search_box_for(self, screen_path: str, keyword: str, cfg: dict, frame):
        """把标注框换算到当前帧坐标, 再外扩成搜索区."""
        annotation = self._annotations.get(category_name(screen_path, keyword))
        if annotation is None:
            return None
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = frame_box(
            annotation.bbox, annotation.image_size, (frame_w, frame_h),
            tuning.TEMPLATE_SEARCH_EXPAND,
        )
        if w <= 0 or h <= 0:
            self.log_debug(f"    搜索区为空 '{screen_path}/{keyword}'")
            return None
        return Box(x, y, w, h, name=f"search_{keyword}")
    # ---------------------------------------------------------------
    # 截图
    # ---------------------------------------------------------------

    def _grab(self):
        """抓一帧新画面.

        必须主动刷新: TaskExecutor.frame 只在内部缓存为空时抓帧, 任务运行期间
        主循环不会并发更新, 一直读 self.frame 只会拿到进入任务时的旧帧,
        轮询等待页面就永远不会生效.

        bench 挂钩(默认关闭, 见 src/instrumentation.py): 只计时/计数,
        拿不到新帧(异常回退缓存 / next_frame 返回 None)由 mark.fail() 记一次
        失败, 不改变任何原有行为.
        """
        with bench.grab() as mark:
            try:
                frame = self.next_frame()
            except Exception as e:
                mark.fail()
                self.log_warning(f"刷新画面失败，回退缓存帧: {e}")
                return self.frame
            if frame is None:
                mark.fail()
            return frame if frame is not None else self.frame

    # ---------------------------------------------------------------
    # 响应"停止"  (2026-10-06 加: 点了停止却停不下来)
    # ---------------------------------------------------------------

    def _should_yield(self) -> bool:
        """用户点了停止/暂停/退出就返回 True, 调用方应当立刻收工返回.

        **框架有三套互相独立的停止标志, 少看一个就会漏** (2026-10-06 实测:
        只看 paused 时, 用户点任务卡上的"停止"完全没反应 -- 因为"停止"
        会顺手把 paused 清回 False; 用户原话是"点了停止后再开始任务,
        似乎是从上一次任务后继续的").

        | UI 动作              | 底层调用                          | 被打的标志          |
        |---------------------|----------------------------------|-------------------|
        | 主界面 Start/Pause  | `executor.pause()`               | `executor.paused`  |
        | 任务卡"暂停"          | `task.pause()`                   | `executor.paused`  |
        | **任务卡"停止"**      | `task.disable()` + `unpause()`   | **`self._enabled`**|
        | 关闭程序            | `executor.stop()`                | `exit_event`       |

        "停止"那行是最阴的: `TaskCard.stop_clicked` 先 `disable()` 再
        `unpause()`, 而 `unpause()` 内部会调 `executor.start()` 把
        `paused` 设回 False. 所以点完停止后 `executor.paused` 是 **False**,
        只盯着它的写法永远等不到信号.

        另一个坑: `executor.current_task` 在 disable 时会被框架置 None,
        所以不能靠"当前任务是不是我"来判断, 得看自己的 `_enabled`.

        **为什么要自己写这个**: 框架的停止标志不会自动打断任务代码.

        - `TaskExecutor.execute()` 只在**任务之间**看 `self.paused`;
        - `executor.sleep()` 内部虽然会发现 paused, 但它只是原地短轮询等待,
          **醒来后 while 循环照跑下一轮**;
        - `next_frame()` 既不看 paused 也不看 exit_event;
        - 唯一能真正中止任务代码的是 `executor.check_enabled()`:
          paused 时它 sleep, 任务被 disable 时它抛 `TaskDisabledException`.

        我们原来的主循环是 `while time.time() < deadline` -- 只看时钟, 什么
        停止信号都不看, 于是点了停止会一直跑到一局结束.
        框架其实留了钩子 `Task.sleep_check()`, 但那也只覆盖 sleep 期间,
        干活期间同样漏.

        所以**每个循环体里都要显式问一次**. 这里统一封装, 免得漏.

        **先读标志再调 check_enabled**: `check_enabled()` 在 paused 时会
        `sleep(1)` -- 如果每轮轮询都先调它, 正常跑的时候也白白慢1 秒.
        先看标志, 只在真的被点了停止时才走那条会 sleep 的路径.

        返回 True 时同时把 `_stopped` 置上: 调用方常需要区分"这一轮是因为
        用户喊停才结束的"和"真的超时了", 否则会把主动停止报成任务失败.
        """
        # 1) 关闭程序(executor.stop 只设 exit_event, 什么都不设)
        exit_event = getattr(getattr(self, "executor", None), "exit_event", None)
        if exit_event is not None and exit_event.is_set():
            self._stopped = True
            return True

        # 2) 被 disable(任务卡上的"停止"). 必须在 paused 之前判: "停止"会
        #    调unpause() 把 paused 清回 False, 只看 paused 会漏掉这一路.
        if not self.enabled:
            self._stopped = True
            return True

        # 3) 被暂停(主界面/任务卡的 Pause). 这一路要调 check_enabled() 让
        #    框架走它的等待/中止流程 -- 用户可能只是暂停, 之后还会继续.
        executor = getattr(self, "executor", None)
        if executor is None or not executor.paused:
            return False
        self.check_enabled()
        self._stopped = True
        return True

    @property
    def stopped(self) -> bool:
        """本轮任务是不是因为用户点了停止而收工的."""
        return self._stopped

    # ---------------------------------------------------------------
    # 调试落盘(默认关闭, 避免仓库堆积中间文件)
    # ---------------------------------------------------------------

    def _save_click_artifact(self, point, cfg: dict) -> None:
        """保存点击位置附近的裁剪图, 方便核对点到了哪里. 默认关闭."""
        frame = self._grab()
        if frame is None:
            return
        try:
            DEBUG_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            px, py = int(point[0]), int(point[1])
            x1 = max(0, px - 100)
            y1 = max(0, py - 100)
            x2 = min(frame.shape[1], px + 100)
            y2 = min(frame.shape[0], py + 100)
            crop = frame[y1:y2, x1:x2].copy()
            cv2.circle(crop, (px - x1, py - y1), 10, (0, 0, 255), -1)
            cv2.putText(
                crop, f"CLICK ({px},{py})", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
            out = DEBUG_ARTIFACTS_DIR / f"click_{px}_{py}.png"
            imwrite_unicode(out, crop)
            self.log_info(f"    [诊断] 点击位置图: {out}")
        except Exception as e:
            self.log_warning(f"    [诊断] 保存点击位置图失败: {e}")

    def _save_unknown_page(self, frame, hint: str) -> None:
        """认不出来的页面存成待标注截图, 交给人工框几个区域.

        直接存进**标注目录**(带 `待标注_` 前缀) -- 打开 GUI 的「模板」页签就能
        在网格里看到它并补框, 不需要额外一步"导入"。它不在 COCO 里, 所以不会
        被当成页面特征参与匹配。

        对局内认不出状态、或按钮等不到时都会走这里。
        """
        if frame is None:
            return
        try:
            TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%m%d_%H%M%S")
            base = f"{UNANNOTATED_PREFIX}{_safe_filename(hint)}_{ts}"
            path = TEMPLATES_DIR / f"{base}.png"
            imwrite_unicode(path, frame)
            self.log_info(
                f"已存待标注截图: {path}（在「模板」页签里找到它并框区域）")
            self._record({"event": "unknown_page", "file": str(path)})
        except Exception as e:  # noqa: BLE001
            self.log_warning(f"存待标注截图失败: {e}")

    def _record(self, payload: dict) -> None:
        """记录钩子. 基类什么都不记; 打牌任务覆写成写 JSONL."""

    # ---------------------------------------------------------------
    # 标注
    # ---------------------------------------------------------------

    def _load_annotations(self, coco_path: Path) -> None:
        """读 COCO 标注, 建 {类目名: Annotation} 索引.

        这是**唯一**的标注来源: 框架匹配读的就是同一个文件
        (config 的 template_matching.coco_feature_json), 中间没有任何
        导出/转换环节, 所以不存在"标注变了但匹配用的还是旧副本"这种事
        (历史教训见 docs/ARCHITECTURE.md 的「标注」一节)。
        """
        try:
            self._annotations = load_annotations(coco_path)
        except Exception as e:  # noqa: BLE001
            self._annotations = {}
            self.log_error(f"读取标注失败 {coco_path}: {e}")
            return

        if not self._annotations:
            self.log_warning(
                f"还没有任何标注: {coco_path} 不存在或为空。"
                f"请在游戏界面里用「模板」页签截图并框区域。")
            return
        for category in sorted(self._annotations):
            annotation = self._annotations[category]
            self.log_info(
                f"已加载标注: {category} "
                f"rect=({annotation.bbox[0]},{annotation.bbox[1]}) "
                f"{annotation.bbox[2]}x{annotation.bbox[3]}"
                f" @ {annotation.image}")

    def _load_task_config(self) -> dict:
        """从 configs/<任务名>.json 加载用户配置, 合并默认值."""
        cfg = self._default_config()
        cfg_path = Path("configs") / f"{self.__class__.__name__}.json"
        if cfg_path.exists():
            try:
                user_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                cfg.update(user_cfg)
                self.log_info(f"已加载用户配置: {cfg_path}")
            except Exception as e:
                self.log_warning(f"读取配置失败 {cfg_path}: {e}")
        self._jitter_sigma = tuning.JITTER_SIGMA
        return cfg
