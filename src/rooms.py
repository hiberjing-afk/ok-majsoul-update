"""房间规格与开房流程(纯逻辑, 不依赖 ok-script, 可单测).

为什么单独一个文件: 要开的房间不止"友人场三人场一局", 后面还要"银之间四人场
半庄", 而且还要**循环连打**. 这些流程大同小异 -- 换的是页面名和按钮名.
把所有可变的部分收进 `RoomSpec`, 流程由 `build_room_flow()` 生成,
加一个新房间类型 = 加一个常量, 不用复制粘贴整个任务.

分工:

    RoomSpec      描述"要开什么房间"(场次 / 人数 / 时长 / 要不要加电脑)
    RoomScreens   描述"这个场次的页面和按钮各叫什么"(标注里的类目名 = "页名/区域描述",
                  必须和 ok_templates/coco_annotations.json 里的 name 完全一致)
    build_room_flow(spec)  生成从登录到准备页的步骤
    start_step(spec)       准备页上"开始"那一步
    required_categories(spec)  这个规格需要哪些标注类目

标注还没齐的规格(比如银之间)照样可以声明: `required_categories()` 会列出需要
的类目名, 任务据此提示"先去标注", 而不是跑到一半才炸.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

# ---- 界面下拉框的选项(必须是字符串, 见 CreateFriendsRoomTask.config_type 的说明)----
#
# 为什么值和按钮名不一样: 「三人」是人话, 标注里的按钮叫「三人麻将」.
# 两者分开, 于是加一个别名不会牵动标注.
PLAYER_OPTIONS: tuple[str, ...] = ("三人", "四人")
LENGTH_OPTIONS: tuple[str, ...] = ("一局", "东风局")

# ---- 段位场的两段选择(2026-10-25 用户要求细化)----
#
# 流程是「大厅 -> 段位场 -> 段位(tier) -> 对局(game) -> 自动匹配」:
#   tier  段位场选择段位页上的卡片: 铜之间/银之间/金之间/玉之间...
#   game  选中段位后那一页(段位场选择对局页)上的卡片: 四人东/四人南/三人东/三人南
# 两页都是**可滚动列表**: 靠下的卡片可能只露半个(三人南在 1080p 上就在屏幕外),
# 所以点它们不能位置匹配, 要"找 -> 滚 -> 再找"(见 Step.scroll).
RANKED_TIER_OPTIONS: tuple[str, ...] = ("银之间", "金之间")
RANKED_GAME_OPTIONS: tuple[str, ...] = ("四人东", "四人南", "三人东", "三人南")

#: 段位场两张列表页的页名(标注里的 "页名/" 前缀, 必须逐字一致).
RANKED_TIER_SCREEN = "段位场选择段位页"
RANKED_GAME_SCREEN = "段位场选择对局页"

#: 段位/对局卡片上"这一档不开放"时的标题文字(红字).
#: 出现它就说明**当前账号的段位打不了这一档**(例子: 三人场段位过高, 银之间的
#: 三人场那格就显示「段位不符」), 不是找不到按钮.
RANKED_LOCKED = "段位不符"

#: 中文选项 -> 规格 key 里的英文段, 只影响配置文件的写法, 不影响界面.
_TIER_KEYS = {"银之间": "silver", "金之间": "gold"}
_GAME_KEYS = {"四人东": "4p_east", "四人南": "4p_south",
              "三人东": "3p_east", "三人南": "3p_south"}


@dataclass(frozen=True)
class Step:
    """流程里的一步.

    screen:  页面名(类目名的前半, 如 "大厅页")
    target:  要点击的区域描述(类目名的后半, 如 "友人场")
    detect:  可选. 先用这个区域探测当前页面, 命中才执行本步; 不命中则跳过.
    toggle:  这一步是"开关型按钮"(选中/未选中外观不同), 用颜色判断而非模板匹配.
    scroll:  目标在**可滚动列表**里, 首屏可能看不到(三人南就在屏幕外).
             这类步骤走"找 -> 滚一格 -> 再找", 不能用固定位置匹配;
             搜索区也要放大到整列(见 tuning.LIST_SEARCH_BOX).
    """

    screen: str
    target: str
    detect: Optional[str] = None
    toggle: bool = False
    scroll: bool = False


# 登录后的入口判定: 命中哪个就在哪个页面, 都命中不了说明还没进入游戏界面.
# 这里只用来"等画面就绪", 真正的动作在流程第一步(带 detect).
LOGIN_ENTRY: list[tuple[str, str]] = [
    ("活动页", "活动标题"),
    ("大厅页", "友人场"),
]


@dataclass(frozen=True)
class RoomScreens:
    """一个场次的页面名与按钮名. 必须和 ok_templates/coco_annotations.json
    里的类目名一致(类目名 = "页名/区域描述")."""

    lobby: str = "大厅页"
    entry: str = "友人场"                 # 大厅页上进入该场次的按钮
    room_list: str = "友人场选择房间页"    # 房间列表页
    create: str = "创建房间"              # 列表页上的"创建房间"
    settings: str = "友人场创建房间页"     # 人数/时长设置页
    player_count: str = "三人麻将"        # 人数按钮(开关型)
    length: str = "一局"                  # 时长按钮(开关型)
    confirm: str = "创建房间"             # 设置页上的确认按钮
    ready: str = "友人场房间内准备页"      # 准备页(点开始 / 加电脑)
    # 准备页上**判定"我已在准备页"用的特征**, 和要点的按钮分开.
    #
    # 为什么不能直接用 start 当特征(2026-10-06 实跑踩的坑): "开始"是会变状态的
    # 控件 -- 人数不够时是灰白未激活态(实测标注版平均亮度 149.8, 同页其他按钮
    # 只有 48~55), 加上电脑后转激活态, 鼠标悬停还会再变一次. 同一张标注图上
    # 自匹配 1.000, 但实跑跨帧稳定在 **0.836**, 而页面探测阈值是 0.88 --
    # **这个检查注定永远失败**. 实跑表现就是每次都白等满 page_wait_timeout,
    # 然后报"未检测到准备页的'开始'按钮", 而按钮其实好好地在那儿.
    #
    # 判据: 页面特征必须是**不会随交互变化的静态元素**, 且与其他页面区分度足够.
    #
    # **别去"修正"这个框让它盖住文字**(2026-10-07 实测, 代价很大):
    # 它框住的其实是规则面板那一列的**背景纹理**, 没盖住"三人麻将/四人麻将"
    # 那几行字(框只有 92px 宽, 文字在 x=1680~1800, 完全在框外). 看着像个
    # 标注错误, 但把框挪到文字上之后**区分度反而变差**:
    #     现有背景框  三人页 -> 四人页 0.592/ 裕度 0.408
    #     挪到文字    三人页 -> 四人页 0.843 / 裕度 0.157
    # 原因是"三人麻将"和"四人麻将"只差第一个字, 后三个字完全一样, 框越小
    # 那个字占的像素比例越低. 背景纹理反而更独特.
    # (在真实搜索区 =标注框 ±50% 内实测, 8 个页面互比.)
    #
    # 代价: **这个标识分不出时长**. 它只能区分"三人准备页/四人准备页",
    # 分不出"一局/东风局". 时长正确性靠创建房间页上点对时长按钮保证
    # (开关型 + 颜色判定, 见 CreateFriendsRoomTask._click_toggle),
    # 不靠准备页 -- 所以 4 种组合能共用这 2 份准备页标注.
    ready_marker: str = "准备页标识"
    # 准备页上"开始对局"那个按钮, 只用来点, 不用来判定页面.
    start: str = "开始"
    bots: tuple[str, ...] = ("添加电脑1", "添加电脑2")  # 空元组 = 不用加电脑


@dataclass(frozen=True)
class RoomSpec:
    """要开什么房间."""

    key: str                 # 配置里引用的 key
    label: str               # 给人看的名字
    player_count: int        # 3 / 4 -- 决策层要用(三人麻将去掉 2m-8m)
    screens: RoomScreens = field(default_factory=RoomScreens)
    # 一局打完回到准备页后要不要再来一局(循环连打时由任务配置决定局数)
    restartable: bool = True
    # 额外要点的开关型按钮, 在人数之后, 时长之前.
    #
    # **四人东风局是"两个独立开关"**(2026-10-06 用户新增标注后确认):
    #   四人麻将 -- 选人数(与三人麻将同一个位置的互斥按钮)
    #   东风局   -- 选时长(与"一局"互斥, 位于"一局"右侧)
    # 也就是说"四人东风"要连点两个, 漏掉任何一个都开不出想要的局.
    # 一局/东风局 的模板都只能覆盖未选中态, 所以都必须按 toggle 处理.
    extra_toggles: tuple[tuple[str, str], ...] = ()
    # 段位场流程(2026-10-09): 与友人场的差别不在按钮名而在**流程形状** --
    #   友人场: 选人数/时长(开关) -> 创建 -> 准备页点"开始" -> 开局.
    #   段位场: 大厅 -> 段位场 -> 银之间 -> 四人东 -> **自动匹配开局**,
    #           没有准备页、没有"开始"按钮、没有任何开关型按钮.
    # flow 生成与"打完一局怎么开下一局"都按这个字段分支(见
    # build_room_flow 与 PlayOneGameTask._after_room_ready).
    ranked: bool = False


# ---- 已注册的房间规格 ----
FRIENDS_3P_EAST = RoomSpec(
    key="friends_3p",
    label="友人场 三人麻将 一局",
    player_count=3,
    screens=RoomScreens(ready="友人场三人场房间内准备页"),
)

# 友人场四人场东风局(2026-10-06 用户新增标注后开放).
#
# 与三人场的差异, 全在准备页页名 / 加电脑个数 / 两个开关上:
#   - 准备页是独立的标注页("友人场四人场房间内准备页"), 准备页标识取的是
#     右上角规则块的上半部分(175px), 与三人那份整块(330px)高度不同,
#     两人模板不会互相命中(实测: 四人模板在三人页上最高 0.756).
#   - 要加**三个**电脑(四人局满员 4 = 自己 + 3).
#   - 人数选"四人麻将", 时长选"东风局"(不是"一局").
FRIENDS_4P_EAST = RoomSpec(
    key="friends_4p",
    label="友人场 四人麻将 东风局",
    player_count=4,
    screens=RoomScreens(
        ready="友人场四人场房间内准备页",
        player_count="四人麻将",
        length="东风局",
        bots=("添加电脑1", "添加电脑2", "添加电脑3"),
    ),
)

# 段位场: 段位(tier) x 对局(game) 两张表相乘, 8 个规格.
#
# 页名约定(2026-10-09 标注、2026-10-25 补齐金之间与另外三个对局):
#   点「段位场」-> 段位场选择段位页(铜/银/金/玉之间)
#   点「银之间」-> 段位场选择对局页(四人东/四人南/三人东/三人南)
#   点「四人东」-> 自动匹配开局(无准备页、无"开始"按钮)
#
# 两张列表页都可能要滚动: 段位页最下面的玉之间、对局页最下面的三人南在
# 1080p 上都看不全, 所以这两步都带 scroll=True.
def _ranked_spec(tier: str, game: str) -> RoomSpec:
    """拼一个段位场规格. key 形如 ranked_silver_4p_east."""
    if tier not in RANKED_TIER_OPTIONS:
        raise KeyError(f"没有段位 '{tier}'（可选: {list(RANKED_TIER_OPTIONS)}）")
    if game not in RANKED_GAME_OPTIONS:
        raise KeyError(f"没有对局 '{game}'（可选: {list(RANKED_GAME_OPTIONS)}）")
    return RoomSpec(
        key=f"ranked_{_TIER_KEYS[tier]}_{_GAME_KEYS[game]}",
        label=f"段位场 {tier} {game}",
        # 三人麻将去掉 2m-8m, 牌桌标注也分三人/四人两套, 所以人数必须从
        # 对局名推出来, 不能写死.
        player_count=4 if game.startswith("四人") else 3,
        ranked=True,
        screens=RoomScreens(
            entry="段位场",
            room_list=RANKED_TIER_SCREEN,
            create=tier,
            settings=RANKED_GAME_SCREEN,
            player_count=game,      # 这页上不是开关, 是直接点就匹配的入口
            length="",              # 段位场没有时长选择
            confirm="",             # 也没有确认按钮
            ready="对局开始页",      # 匹配成功 -> 胜负横幅(当 ready_marker 用)
            ready_marker="胜负",
            start="",               # 匹配成功自动开局, 没有"开始"可点
            bots=(),                # 段位场是真人对局, 不用加电脑
        ),
    )


RANKED_SPECS: tuple[RoomSpec, ...] = tuple(
    _ranked_spec(tier, game)
    for tier in RANKED_TIER_OPTIONS
    for game in RANKED_GAME_OPTIONS
)

#: 老配置里的 key(只有一个「银之间四人东」)。保留它是因为 configs/*.json 里
#: 可能写着 room="ranked_silver_4p"; 新 key 从它派生, 指向同一个规格.
LEGACY_RANKED_KEYS: dict[str, str] = {"ranked_silver_4p": "ranked_silver_4p_east"}

ROOM_SPECS: tuple[RoomSpec, ...] = (FRIENDS_3P_EAST, FRIENDS_4P_EAST,
                                    *RANKED_SPECS)

# 标注已经齐, 可以真正跑起来的规格.
# 没列进来的规格在测试里只断言"缺标注被报出来", 不会因为没标注而失败.
READY_SPECS: tuple[str, ...] = (
    "friends_3p", "friends_4p",
    *(spec.key for spec in RANKED_SPECS),
)

# ---- 界面下拉框 -> 房间规格 ----
#
# 用户要的是「四人 or 三人」「一局 or 东风」分开选(2026-10-07), 所以这里
# 把两个下拉框的组合映射到规格. 官方规则里"三人东风"是存在的
# (模式: 一局胜负 / 四人. 三人东风 / 四人. 三人半庄), 所以四人种组合都合法.
#
# **关键约束: 准备页标注只跟"人数"有关, 与"时长"无关**(2026-10-07 实测):
# 两个准备页的「准备页标识」区域框住的其实是规则面板那一列的**背景纹理**,
# 没盖住"三人麻将"/"四人麻将"那几行字(框只有 92px 宽, 文字在其右侧).
# 而准备页上除了"开始""添加电脑N"和这块规则面板, 其余完全一样, 规则面板上
# 唯一随房型变化的字(人数/时长)又恰好落在框外.
# 后果有两面:
#   - 好的一面: 同一份准备页标注能覆盖"三人一局"和"三人东风",
#     "四人一局"和"四人东风" -- 4 种组合只需要 2 份准备页标注.
#   - 坏的一面: **准备页标识无法区分时长**. 所以时长选错不会被准备页拦住.
#     时长正确性靠创建房间页那一步保证(那里点的是"东风局"按钮本身,
#     开关型+颜色判定, 见 _click_toggle), 不靠准备页.
#
# 因此每个组合必须列出**明确要点的时长按钮名**, 而不是共用一个.


def _with_length(spec: RoomSpec, length: str) -> RoomSpec:
    """换个时长按钮, 其余(人数/页面名/电脑位/准备页)全沿用.

    存在的意义: 「三人一局」和「三人东风」在**标注层面完全一样** -- 同一个
    准备页, 同样两个电脑位, 只有创建房间页上点的那个时长按钮不同. 所以不必
    为每个组合各写一份 RoomSpec, 换个 length 派生一份就行.
    """
    return replace(spec, key=f"{spec.key}_{length}",
                   screens=replace(spec.screens, length=length))


_COMBINATION_TABLE: dict[tuple[str, str], RoomSpec] = {
    ("三人", "一局"): FRIENDS_3P_EAST,
    ("四人", "一局"): _with_length(FRIENDS_4P_EAST, "一局"),
    ("四人", "东风局"): FRIENDS_4P_EAST,
    # 三人东风局: 人数/加电脑位与三人一局完全相同, 只有时长按钮不同.
    ("三人", "东风局"): _with_length(FRIENDS_3P_EAST, "东风局"),
}


def available_combinations() -> list[tuple[str, str]]:
    """下拉框允许的组合, 按 (人数, 时长) 排. 界面的报错信息会用到."""
    return list(_COMBINATION_TABLE)


def spec_by_combination(players: str, length: str) -> RoomSpec:
    """按 (人数, 时长) 取规格.

    组合非法要**立刻报错**, 不能静默回退到默认 -- 开出一个错的局比崩掉更糟.
    """
    try:
        return _COMBINATION_TABLE[(players, length)]
    except KeyError:
        valid = "、".join(f"{p}{ln}" for p, ln in available_combinations())
        raise KeyError(
            f"没有「{players} + {length}」这个组合（可选: {valid}）"
        ) from None


def spec_by_key(key: str) -> RoomSpec:
    """按 key 取规格; 找不到就抛(配置写错要立刻发现, 不要静默回退)."""
    key = LEGACY_RANKED_KEYS.get(key, key)
    for spec in ROOM_SPECS:
        if spec.key == key:
            return spec
    available = ", ".join(spec.key for spec in ROOM_SPECS)
    raise KeyError(f"没有 key 为 '{key}' 的房间规格（已有: {available}）")


def ranked_spec(tier: str, game: str) -> RoomSpec:
    """按界面上的两个选项取段位场规格.

    非法组合**立刻报错**: 开出一个错的局(比如想打三人南却点了四人东)比崩掉更糟.
    """
    return _ranked_spec(tier, game)


def build_room_flow(spec: RoomSpec) -> list[Step]:
    """生成"从登录到停在准备页"的步骤(不含点开始).

    第一步是活动页的返回大厅: 雀魂每天第一次登录会先进活动页, 用 detect
    探测, 不在活动页就自动跳过.

    开关型按钮的顺序**不能随意改**: 人数和时长在创建房间页上是两组互斥按钮,
    点人数会让时长那组的未选中/选中态刷新, 所以先点人数再点时长.

    段位场(spec.ranked)是另一条流程: 点完"四人东"就自动匹配开局,
    没有开关、没有确认、没有准备页 -- 见 RoomSpec.ranked 的说明.
    """
    screens = spec.screens
    if spec.ranked:
        return [
            Step("活动页", "返回大厅", detect="活动标题"),
            Step(screens.lobby, screens.entry),
            # 段位页与对局页都是可滚动列表: 目标可能在屏幕外, 走"找->滚->找".
            Step(screens.room_list, screens.create, scroll=True),
            Step(screens.settings, screens.player_count, scroll=True),
        ]
    flow = [
        Step("活动页", "返回大厅", detect="活动标题"),
        Step(screens.lobby, screens.entry),
    ]
    if screens.create:
        flow.append(Step(screens.room_list, screens.create))
    flow.append(Step(screens.settings, screens.player_count, toggle=True))
    for screen, target in spec.extra_toggles:
        flow.append(Step(screen, target, toggle=True))
    flow.append(Step(screens.settings, screens.length, toggle=True))
    if screens.confirm:
        flow.append(Step(screens.settings, screens.confirm))
    for bot in screens.bots:
        flow.append(Step(screens.ready, bot))
    return flow


def start_step(spec: RoomSpec) -> Step:
    """准备页上"开始对局"那一步."""
    return Step(spec.screens.ready, spec.screens.start)


def ready_marker(spec: RoomSpec) -> tuple[str, str]:
    """判定"当前已在准备页"的 (页名, 特征名).

    返回的**不是**要点的开始按钮, 而是页面上不会变的静态特征 -- 理由见
    RoomScreens.ready_marker 的说明. 调用方判断"回到准备页了没"一律用这个,
    绝不能拿 start 按钮去判.

    段位场没有准备页: 一场的"结束" = **回到大厅**(下一轮排队的起点, 与
    _back_to_lobby 的判据同源). 绝不能用 screens.ready 的「对局开始页/
    胜负」判局结束 -- 那是**开局**页面, 匹配成功的同一秒就会命中
    (2026-10-09 实跑实锤: "匹配成功" -> "第 1 局结束, 共 0 轮" -> 回大厅
    重排队, 无限循环永远进不了对局).
    """
    if spec.ranked:
        return ("大厅页", spec.screens.entry)
    return spec.screens.ready, spec.screens.ready_marker


def required_categories(spec: RoomSpec) -> list[str]:
    """这个规格需要哪些标注类目名("页名/区域描述").

    不含 detect 用的条件特征(活动页/活动标题) -- 那个缺了只影响"每天首次登录",
    单独在任务里警告.
    """
    need: list[str] = []
    steps = build_room_flow(spec)
    if not spec.ranked:
        steps = steps + [start_step(spec)]
    for step in steps:
        if step.screen == "活动页":
            continue
        category = f"{step.screen}/{step.target}"
        if category not in need:
            need.append(category)
    # 页面特征也要标: 缺了它任务一样跑, 但会一直判不出"回到准备页了没",
    # 表现为每局都要空转到 game_timeout -- 这种错非常难查, 必须在启动前报出来.
    # 段位场的"页面特征"是回大厅判据(大厅页/段位场, 见 ready_marker), 与
    # flow 第二步同名; 而「对局开始页/胜负」是匹配成功的信号(_wait_matchmaking
    # 在用), 不经过 ready_marker, 必须单独列 -- 缺了它匹配永远等不到.
    for category in (f"{ready_marker(spec)[0]}/{ready_marker(spec)[1]}",
                     f"{spec.screens.ready}/{spec.screens.ready_marker}"
                     if spec.ranked else ()):
        if category and category not in need:
            need.append(category)
    return need
