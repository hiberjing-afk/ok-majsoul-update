"""对局内"这一轮该干什么"的决策(纯逻辑, 可单测, 不碰截图也不碰点击).

输入是感知层给的两样东西:

    tiles    认出来的手牌, 从左到右
    buttons  认出来的按钮, {类别名: 置信度}(有名字就代表画面上出现了)

输出一个 `TurnDecision`, 由任务去执行(点按钮 or 点手牌 or 什么都不做等一等).

按钮优先级按"纯牌效进攻流"排(docs/STRATEGY.md): 能和就和, 能立直就立直,
吃碰杠默认**不碰**(保持门清, 门清 + 立直是进攻流最典型的一路,
而且不用判断"碰了之后向听降不降"), 有跳过就跳过.

**拔北要拔**(2026-10-06 查证后改, 推翻同一天早些的结论): 三人麻将的拔北
有两种触发, 我早先只认了"别人打出的北"那一种, 于是把手里的北当成不能拔:

    拔北的程序等同加杠, **必须于摸牌后进行**. 玩家从其**手牌中**将北拔出来,
    摊开并放到自己的右边, 然后从牌山补一张牌, 接着亦可自摸/ 加杠 / 暗杠或
    继续拔, 否则打一张牌出来.

也就是说**轮到自己出牌时, 手里攥着北就可以拔**, 这跟"是不是别人打的"
无关. 早先那句"拔北出现时不是我们的出牌回合(北是别人打出来的)"是错的,
实跑证据也支持纠正: 那一局有 3 轮认出了 babei 按钮, 每轮手里都攥着
3~4 张 4z(北) 且都是 14 张(刚摸完牌), 位置也固定在 x≈727 -- 那不是吃
的槽位(三人麻将不能吃), 就是拔北本身. 因为被归进 IGNORED_BUTTONS,
这 3 次真实的拔北机会全被放过了.

见北就拔: 拔一张北等于"多摸一张牌 + 多一翻", 而我们本来也多半会把那张
北打掉. 例外是留着北能和牌(国士/四喜)或者正在听牌想赶紧走, 那种情况
留在手里才有价值 -- 由 babei_policy 控制.

**立直后只能打高亮牌**(同日实跑, 用户纠正): 立直一旦点下去, 雀魂进入
选牌态 -- **能打的牌高亮, 其余变暗**, 此时点一张高亮牌即完成立直宣言.
早先我以为是"雀魂指定了宣言牌, 把它单独渲染得更矮(118px)放在分离位",
那是**错的**: 复核实跑坐标发现 118px 量到的是**刚摸到的那张牌**(只出现在
刚摸牌那一轮), 而"分离位左边距"来自统计脚本取 max(gaps) 的 bug. 于是
t78~t82 连续 5 轮按牌效点同一张**变暗的**牌, 雀魂根本不响应(手牌数一直
是 14); t83 打出高亮的 2p 才成功, 手牌随即变成已排序(见 log t76~t83).

判据交给调用方: 传入 allowed_indices(高亮牌的下标集合)就只在这些牌里
按牌效挑最好的那张; 传空集合表示"这帧没认出高亮牌", 那一轮不动 --
猜错就是点在变暗的牌上, 纯白等.

**轮到我出牌时不点"跳过"**(同一天实跑后改): 手里 14 张说明刚摸完牌, 轮到自己,
雀魂此时不可能给吃碰杠的机会, 画面上那个"跳过"是上一轮的残留或误检. 实测一局
点了 8 次跳过, 其中 5 次是这种白点, 每次白花 2~3 秒. 判据见 decide_turn 的注释.

轮次判断完全靠手牌张数, 不靠画面上的高亮提示:

    暗牌数 % 3 == 2  ->  刚摸完牌, 该我出牌(14 / 11 / 8 / 5 / 2)
    否则             ->  要么在等别人, 要么识别漏了牌

这个判据的好处是不需要额外标注"轮到我了"的提示元素, 而且副露之后同样成立
(每多一个副露, 暗牌少 3 张).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from src.mahjong.brain import choose_discard, is_tenpai_hand, rank_discards, robust_discard
from src.mahjong.tiles import out_of_play_indices

# ---- 动作 ----
ACTION_WIN = "win"          # 和牌 / 自摸
ACTION_RIICHI = "riichi"    # 立直
ACTION_BABEI = "babei"      # 拔北(三人麻将特有)
ACTION_MELD = "meld"        # 吃 / 碰 / 杠
ACTION_SKIP = "skip"        # 跳过这次吃碰杠机会
ACTION_CONFIRM = "confirm"  # 确认(流局 / 结算)
ACTION_DISCARD = "discard"  # 出牌
ACTION_WAIT = "wait"        # 什么都不做, 等下一轮

# ---- UI 模型的类别名 -> 动作 ----
BUTTON_WIN = ("zimo", "he")
BUTTON_RIICHI = "lizhi"
BUTTON_BABEI = "babei"
BUTTON_GANG = "gang"                # 杠要单独判(暗杠不破门清), 见 decide_turn 第 5 步
BUTTON_PENG = "peng"
BUTTON_CHI = "chi"                  # 吃可能弹组合选择条(没建模), 见 decide_turn 第 5 步
BUTTON_MELD = (BUTTON_PENG, BUTTON_CHI, BUTTON_GANG)
BUTTON_SKIP = "tiaoguo"
BUTTON_CONFIRM = ("confirm", "liuju")

#: 协议的鸣牌种类 -> 要点的按钮名. 大明杠和暗杠/加杠共用「杠」这一个按钮
#: (雀魂按当前机会决定点击语义), 所以 minggang 对应 BUTTON_GANG.
KIND_TO_BUTTON = {"chi": BUTTON_CHI, "peng": BUTTON_PENG,
                  "minggang": BUTTON_GANG}

# 认得出来但先不动的按钮: match 是大厅的匹配(不该出现在对局里),
# silver 和 3p-east 语义还没确认 -- 只记进日志, 不点.
# babei(拔北)也在这一列: 三人麻将里拔北是可选的, 不点会自动跳过(见模块说明).
# babei(拔北)曾经也在这列, 2026-10-06 查证后移出去了: 拔自己的北也是拔北,
# 手里有北就能拔(见模块说明). 现在它是一个真动作, 由 babei_policy 控制.
IGNORED_BUTTONS = ("match", "silver", "3p-east")


@dataclass(frozen=True)
class TurnDecision:
    """一轮的决策结果.

    action:  win / riichi / babei / meld / skip / confirm / discard / wait
    button:  要点的按钮类别名(动作为 wait / discard 时是 None)
    tile:    要打的牌(只有 discard 有)
    index:   要打的手牌位置(只有 discard 有)
    reason:  为什么这么决定, 记进日志用
    suspect: 这个 wait 是不是"感知出了问题"(而不是正常的在等别人).
             任务靠它区分"等对手出牌"(每局几十轮, 完全正常)和"这画面我
             认不出来"(该截图留证让人标注) -- 两者都是 wait, 但只有前者
             不该触发留证.
    """

    action: str
    button: str | None = None
    tile: str | None = None
    index: int | None = None
    reason: str = ""
    suspect: bool = False

    @property
    def needs_click(self) -> bool:
        return self.action != ACTION_WAIT


def valid_concealed_counts(max_melds: int = 4) -> frozenset[int]:
    """合法的暗牌张数: 13 张起, 每多一个副露少 3 张, 摸牌后 +1."""
    counts = set()
    for melds in range(max_melds + 1):
        base = 13 - 3 * melds
        counts.add(base)          # 摸牌前
        counts.add(base + 1)      # 摸牌后
    return frozenset(counts)


# 低于这个张数就当"认漏了", 不当合法牌型.
#
# **为什么不能只靠 valid_concealed_counts**(2026-10-06 20:59 实跑踩的坑):
# 那函数纯粹按"13 - 3×副露 + 摸牌"推, 于是 4 副露时 base=1, **1 张和 2 张
# 都在合法集合里**. 而牌面模型在**手牌区是空的**时候(开局立绘动画期间手牌还
# 没发出来)必然幻觉出 1~2 张 -- 实跑 15 轮里turn 5 认出 1 张 7z, turn 8~15
# 一直认 1 张 3m, 全是把头像框花纹当成了牌.
# 后果很隐蔽: 1 张是合法张数 -> sane=True -> 第 9 步直接返回"在等别人"
# (reason 不含 suspect) -> 上层既不计入 unknown_streak 也不截图, 只是一轮
# 轮空转, 看起来"任务在跑但一张牌不打".
#
# 为什么下限取4 而不是 1: 三人麻将里4 副露(全副露)几乎不存在, 真出现时
# 暗牌也只有 1~2 张且**画面上明摆着 4 个副露区**; 而认漏的典型症状恰恰是
# "只剩零星几张". 见到 <4 张一律当感知有问题, 宁可多等一轮.
MIN_CONCEALED = 4


def is_plausible_concealed(concealed: int) -> bool:
    """手牌张数是否像"真的认全了"的样子.

    比valid_concealed_counts 多一道下限: 1~3 张虽然数学上对应 4 副露, 但
    实战里几乎只见于"手牌区空着, 模型幻觉出几张"的情况(见 MIN_CONCEALED).
    """
    return concealed in valid_concealed_counts() and concealed >= MIN_CONCEALED


def is_my_discard_turn(concealed: int) -> bool:
    """暗牌数 % 3 == 2 说明刚摸完牌, 该我出牌(14 / 11 / 8 / 5 / 2)."""
    return concealed % 3 == 2


def meld_count_of(concealed: int) -> int:
    """由暗牌数反推副露数. 只在张数合法时才有意义."""
    return (14 - concealed) // 3


def expected_concealed_counts(n_melds: int) -> frozenset[int]:
    """给定**记住的**副露组数, 这一手暗牌数只可能是两个值.

    13 - 3n(还没摸牌) 与 14 - 3n(摸牌后, 含刚鸣牌后待打的那张 —— 鸣牌
    拿进来的牌替代了这次摸牌, 张数数学完全一样). 副露组数是任务层自己
    记账的(点碰/吃/杠按钮一次记一组, 见 PlayOneGameTask._meld_count),
    不是从张数反推的 —— 所以这个集合是**独立的硬校验**: 读数张数不在
    集合里, 说明读数被污染了(副露混进手牌 / 漏认 / 幻觉), 这一轮不能
    按它出牌. 2026-10-11 三碰局实锤这个洞: 副露被读进手牌后张数 8 恰好
    %3==2, 老的 is_my_discard_turn 判成"该我出牌", 反复想打已碰掉的牌.

    与 valid_concealed_counts 的区别: 那个是"任何副露数下的合法张数全集"
    (给不出"该是几张"), 这个把副露数**钉死**到记账值, 两个候选之外全是错.

    n 钳到 0..4: 4 副露全副露时 1/2 张数学上合法, 但调用方另有
    MIN_CONCEALED 下限守着幻觉(见 is_plausible_concealed), 这里只管数学.
    """
    n = max(0, min(4, int(n_melds)))
    return frozenset((13 - 3 * n, 14 - 3 * n))


#: 三元牌(白/发/中): 任何场合都是役. 场风也是役, 由调用方按房型传入
#: (见 tuning.ROUND_WIND: 友人场只有东场, 场风恒为东); 自风要读座位,
#: 没读, 不碰, 宁可少碰也不碰死.
YAKU_HONORS = ("5z", "6z", "7z")


def yaku_pair_kinds(tiles: list[str], missing_slots: int = 0,
                    round_wind: str = "1z") -> frozenset[str]:
    """手里握着的役牌对子(及以上)的种类集合.

    has_yaku_pair 的"是/否"版, 出牌裁决(keep_kinds: 平分时不优先打役对)
    与决策日志都要用种类而非布尔. missing_slots > 0 = 读数不可信, 返回空.
    """
    if missing_slots:
        return frozenset()
    return frozenset(t for t in (*YAKU_HONORS, round_wind)
                     if tiles.count(t) >= 2)


def meld_taken_kinds(pre: list[str], post: list[str]) -> list[str]:
    """从"点碰/杠前手牌"与"之后稳定读数"的差集里反推副露拿走的牌种.

    我们读不到副露区, 但点碰必然从手里拿走一对(d>=2), 明杠拿走三张
    (d>=3), 暗杠四张(d>=4) —— 共同点是该种类在手里的数量**下降 >= 2**.
    之后自己摸牌只会增加数量, 自己出牌只让一个种类 -1, 都不会伪造出
    "-2". 用于任务层维护 melded 列表(加杠支持判据的输入).
    """
    return [t for t in sorted(set(pre))
            if pre.count(t) - post.count(t) >= 2]


def has_yaku_pair(tiles: list[str], missing_slots: int = 0,
                  round_wind: str = "1z") -> bool:
    """手里是否握着役牌对子 -- "有役才碰"的判据.

    背景(2026-10-07 用户拍板): 有碰就点会把没役的手碰死 -- 副露没役 +
    立直没了 = 这手胡不了. 役牌对子碰出去这副露自带役, 才值得鸣.
    2026-10-08 用户提议再开: 点名的东风对子就是**场风役** -- 友人场只有
    东场, 东(1z)恒为役, 所以把场风并进役牌集合.

    盲区: 我们读不到别人打出了什么, 只能看到"有碰的机会". 握着白板对子
    时, 碰的机会也可能是手里别的对子被打出第三张 -> 碰进没役的一副
    (用户已认可接受这个风险). missing_slots > 0 说明读数不可信, 一律不碰.
    """
    if missing_slots:
        return False
    return bool(yaku_pair_kinds(tiles, missing_slots, round_wind))


#: 发牌发齐时手牌应有的张数(还没摸牌). 见 deal_is_done.
OPENING_CONCEALED = 13

#: 连续多少帧"看起来发齐了"才算发牌结束. 见 deal_is_done.
DEAL_CONFIRM_FRAMES = 2


def opening_hand_complete(concealed: int, missing_slots: int = 0) -> bool:
    """这一帧的手牌**看起来**发齐了吗(单帧判据, 不锁存任何状态).

    两个条件缺一不可:

    - `concealed >= OPENING_CONCEALED`: 开局不可能有副露(副露得等别人先打牌),
      所以"手牌发齐"只可能落在 13(或摸牌后的 14)张上.
    - `missing_slots == 0`: 间距判出来有空档时**张数本身就不可信** —— 这时的
      "13 张"很可能是"16 张里只认出 13 张", 拿它当"发齐了"是自欺.

    单独暴露这个单帧判据只为可测: 真正决定要不要出牌的是 deal_is_done.
    """
    return concealed >= OPENING_CONCEALED and missing_slots == 0


def deal_streak(seen: int, *, complete: bool) -> int:
    """把"连续几帧看起来发齐了"推进一格. 断一帧就清零(纯函数, 状态在调用方).

    为什么要计数而不是信单帧: 见 deal_is_done.
    """
    return seen + 1 if complete else 0


def deal_is_done(already_done: bool, streak: int, *, grace_over: bool) -> bool:
    """这一局的发牌结束了没有. **没结束就不许出牌.**

    为什么要单独判这个: 发牌动画期间手牌是**逐张出现**的, 而它路上会经过一串
    "看起来合法"的张数. 三人麻将 2 副露时暗牌正好 8 张, 于是 `is_my_discard_turn(8)`
    为真, 决策层就照着一手还没发完的牌算出了"该打哪张". 实跑 2026-10-07 t18:
    手牌才 8 张, 它决定打 1m, 点了两下都没反应(那时候雀魂根本不收), 还把出牌
    模式误判成"选中再确认". 翻更早的记录, 234158 那局 5 张就出牌、023623 那局
    11 张就出牌 —— **几乎每一局的第一手都是在发牌动画里打出去的**.

    判据为什么是"见过 >=13 张": 开局不可能有副露(副露得等别人先打牌), 所以
    "发齐"这件事只可能落在 13(或摸牌后的 14)张上; 而副露到只剩 8 张一定发生在
    这之后. 所以见过一次 13 之后, 8 张才是真的 8 张.

    `streak` 是"连续几帧看起来发齐了"(见 opening_hand_complete / deal_streak).
    **为什么不能信单帧**(2026-10-07 第二轮实跑 t20~t23 踩的): 那一局 t20 单帧
    读出 13 张(其实是加载页残影), 闸门立刻打开; 紧接着 t21 只读出 11 张, 而
    11 % 3 == 2 恰好是"刚摸完牌, 轮到我", 于是它在一手其实有 13~14 张的牌上
    算出了 1m 并打了出去 —— 点了两下没反应(手牌从 11 读到 13), 于是
    `_need_confirm_click` 被永久置位, 后面**每一次**出牌都变成补两下.
    单帧的错误读数不该有这种后果, 所以要求连续 DEAL_CONFIRM_FRAMES 帧成立:
    真发齐了的话下一帧照样成立, 而残影/误读撑不过下一帧.

    为什么不用"点完开始等 N 秒": 写死秒数要么不够(加载慢), 要么太久 —— 太久会
    撞上雀魂自己的出牌计时被**托管**, 那比打错一张牌严重得多.

    `grace_over`(开局宽限期过了)是**兜底**: 万一牌面识别彻底坏掉、永远认不到 13 张,
    这里不能把任务锁死成"一张牌不打"(那就变成另一个"打着打着不打了"). 25 秒还
    没见过发齐, 说明本来就有别的毛病, 退回旧行为、让日志去暴露问题.

    一旦为真就永远为真(调用方把返回值存回去), 所以这个函数是纯的, 状态在调用方.
    """
    return bool(already_done or grace_over
                or streak >= DEAL_CONFIRM_FRAMES)


@dataclass(frozen=True)
class DealState:
    """`DealWatch.see()` 的结论(纯数据, 便于断言)."""

    #: 现在能不能按正常逻辑打(出牌 / 立直 / 副露 / 拔北).
    done: bool
    #: 这一帧刚确认发齐 = 这一手正式开始(用来打一条日志).
    just_opened: bool = False
    #: 这一帧手牌区刚被清空 = 上一手结束了, 闸门重新上膛.
    rearmed: bool = False
    #: 和了之后等到宽限期都没等到结算, 退回旧行为(只在这一帧报一次).
    gave_up: bool = False


@dataclass
class DealWatch:
    """按**手**管理的"发牌闸门 + 和了封锁"(纯逻辑, 状态存这里而不是调用方).

    为什么要有这个类(两个实测 bug, 都是"把一手牌的状态当成了整场任务的状态"):

    1. **闸门必须按"手"重臂, 不能按"任务"(2026-10-07 t158)**. 一次
       `_play_game` 跑的不是一只手, 而是**整场**(东风局 / 半庄): 中间和了、
       流局、结算之后会接着开下一手, 而主循环并不知道"新的一手开始了".
       于是 `deal_is_done` 的 `already_done` 一旦为真就**永远是真相**(见它的
       docstring), 发牌闸门从第二手起彻底失效 —— 实跑 t158 在东 2 局发到
       **8 张**时就出了牌, 又是"点了两下没反应" -> `_need_confirm_click`
       再次被永久置位. 宽限期同理: 它是按 `_play_game` 起点算的, 25 秒之后就
       **恒为过期**, 也就是闸门的兜底分支被永久打开.

       重臂用的判据是**"手牌区从有牌变成空"这个边沿**, 不是"点了确认"之类的
       动作: 那个 confirm 按钮的实测置信度只有 0.151~0.156, 就压在 UI_CONF
       (0.15)上, 漏掉一次就永远不重臂; 而"手牌区空了"是画面事实, 漏不掉.

    2. **和了之后必须封锁(2026-10-07 t189~t195)**. 和了之后雀魂把**我们自己的
       和了牌正面朝上**摆在原来的手牌位置上, 摊牌动画期间牌面模型会读出一手
       干净的 13~14 张(实跑 t190 读 14 张、`missing_slots=0`), **只看张数和
       正常打牌一模一样** —— 于是它在摊牌动画里又打了 3 次牌(还都是补两下).
       唯一可靠的判据是"我们刚点过和了", 所以 `agari()` 要显式喂进来, 并且
       **粘到下一次手牌区清空为止**(下一手发牌时手牌区必然先空, 实测 t156/t157
       读到 2 张 / 3 张).

    两个方向都要有兜底(项目铁律: 页面门控宁可漏判不可锁死): 和了封锁超过
    `grace_period` 就退回旧行为并报一次警告, 免得判据失灵时把任务锁成
    "一张牌不打"(那是另一种"打着打着不打了", 比出一手错牌更难查).
    """

    grace_period: float = 0.0
    grace_until: float = 0.0
    #: 这一手的发牌已经确认过(>=13 张连续两帧 / 宽限兜底).
    opened: bool = False
    #: 这一手的发牌**刚确认齐**的时刻(see() 里记录). 发牌后的整理动画期间
    #: 牌在滑动, 低置信存帧按它静音(见 tuning.LOWCONF_MUTE_AFTER_DEAL).
    opened_at: float = 0.0
    #: 这一帧手牌区里读得出可信的手牌(用来抓"从有到无"的边沿).
    seen_hand: bool = False
    #: 我们刚和了, 正在等摊牌/结算过完. 粘到下一次手牌区清空.
    #: (名字刻意不叫 `agari`: 那会和下面的 `agari()` 方法撞名, 而 dataclass
    #:  会把同名的类属性当成字段默认值 —— 于是 `self.agari` 变成**函数对象**,
    #:  `if self.agari:` 恒为真, 整个状态机静悄悄地只走和了分支. 写完先跑
    #:  一次单测才发现的.)
    settling: bool = False
    #: "连续几帧看起来发齐了"(见 deal_streak).
    streak: int = 0
    #: `force_rearm()` 置位, 由下一帧 `see()` 消费一次并返回 rearmed=True.
    pending_rearm: bool = False

    def see(self, count: int, missing_slots: int, now: float,
            melded: bool = False, round_open: bool | None = None) -> DealState:
        """喂一帧手牌读数, 返回这一帧能不能按正常逻辑打.

        melded: 这一手里我们自己鸣过牌(暗杠实锤过). 暗杠的 4 张牌**竖着**
        摆在手牌行右端, 和暗牌一样无法从外观区分, 会被一起读进来 ——
        count+missing 因此可以到 17, 正好撞上下面的摊牌幻觉判据(2026-10-08
        20:47 局实锤: 点完暗杠后 11+6=17 被当成摊牌 -> 闸门重臂 -> 杠后的
        手牌永远到不了 13 张 -> 死循环). 自己回合里摊牌不可能出现
        (结算画面没有吃碰杠按钮), 所以 melded=True 时跳过该判据.

        round_open: 协议说"这一手还在打吗"(NetAssist.round_open; 不可用时
        None). **清空边沿的协议否决**(2026-10-17 东二局实锤): 点完吃/碰的
        滑入动画帧手牌可能只读出 1 张(< MIN_CONCEALED), 视觉上与"上一手
        结束了"一模一样, 于是闸门在**一手打中间**被重臂 —— hand_end 误记、
        副露记账清零、之后 25s 宽限期里闸门不开, 8 张手牌一张不出. 协议说
        round_open=True 时, "手牌区空了"必然是动画帧不是收局(收局必有
        Hule/NoTile/LiuJu 事件先到), 直接跳过重臂; None(抓包没开/断了)时
        回退纯视觉判断(老行为, 误重臂也会在宽限兜底后恢复).
        """
        # force_rearm 是**一次性信号**: 点掉一手结束的「(N)确认」后画面可能
        # 直接切到下一手发牌, "手牌区清空"的边沿根本不出现(2026-10-08 实跑:
        # 东 2 每轮决策 discard 却 30 秒一张不打) —— 所以用显式信号代替边沿,
        # 在下一帧无论读到什么都报一次 rearmed, 让调用方的 rearmed 分支
        # (复位 riichi_active / 自动开关 / 日志去重)统一走原路.
        force = self.pending_rearm
        self.pending_rearm = False
        # 摊牌/牌背画面的幻觉读数(2026-10-07 19:15 实跑): 流局摊牌时整行
        # 牌亮出来、别人的和了摊牌时我们的手牌区是牌背, 牌面模型会幻觉出
        # 几张低置信读数 —— 实测 count 5~8 + missing 11~14, **count+missing
        # 远超 14**. 真实手牌一行最多 14 张, 这是"这一行根本不是手牌"的
        # 硬信号, 按"手牌区空了"处理(重臂闸门 + 封锁), 别拿幻觉牌去出牌.
        # 阈值 16 留了 2 张的余量: 正常一手最多 14, 间距估算误差 +1~2.
        # melded=True 时跳过: 那是暗杠牌挤进读数, 不是摊牌(见 docstring).
        if count + missing_slots >= 16 and not melded:
            count = 0
        if count < MIN_CONCEALED:
            # 协议否决(见 docstring): 这一手协议说还在打, "空"必然是鸣牌
            # 滑入动画之类的过渡帧, 不是收局 —— 状态一律不动(尤其不能动
            # settling: 协议的 Hule 事件可能比摊牌动画晚到, 不能拿旧协议
            # 值盖掉自己点过和了的事实), 闸门保持原样. force_rearm 是
            # 显式信号, 优先级高于协议, 不能被这里吞掉.
            if (round_open is True and not self.settling
                    and not self.pending_rearm):
                return DealState(done=self.opened)
            # 手牌区空着. 有两种情况, 处理一样: 上一手已经结束了(和了摊牌 /
            # 结算页 / 下一手还没发牌), 或者这一手还没开始.
            was = self.seen_hand or self.settling or force
            self.seen_hand = False
            self.settling = False
            self.opened = False
            self.streak = 0
            if was:
                # 只在**边沿**上重臂宽限. 每帧都重臂的话, 识别彻底坏掉
                # (count 恒为 0)时宽限会被无限往后推, 又变成永久锁死.
                self.grace_until = now + self.grace_period
            return DealState(done=False, rearmed=was)

        self.seen_hand = True
        if self.settling:
            if now >= self.grace_until:
                self.settling = False
                return DealState(done=True, gave_up=True)
            return DealState(done=False)

        self.streak = deal_streak(
            self.streak,
            complete=opening_hand_complete(count, missing_slots))
        opened = deal_is_done(self.opened, self.streak,
                              grace_over=now >= self.grace_until)
        just = opened and not self.opened
        self.opened = opened
        if just:
            # 这一手"发齐"的时刻. 发牌确认后雀魂还会跑 ~1s 的整理动画(牌在
            # 滑动), 那段低置信是常态不是现场 —— 低置信存帧按它静音.
            self.opened_at = now
        return DealState(done=opened, just_opened=just, rearmed=force)

    def force_rearm(self, now: float) -> None:
        """显式宣告"这一手已经结束了", 下一帧 see() 报一次 rearmed.

        为什么需要(2026-10-08 实跑): 一手结束的顺位结算画面点掉「(N)确认」
        会**加速过场**, 画面可能直接切到下一手的发牌, 手牌行从未出现
        "count < MIN_CONCEALED"的清空帧 —— 默认的重臂边沿根本不出现,
        上一手的 riichi_active / 自动开关标志 / 日志去重表全部滞留, 下一手
        每轮都决策出牌却一张不打(静默, 因为"立直生效中"的日志也被滞留的
        去重表吞了), 用户看到的就是"第二轮直接不打了".

        点掉确认这个**动作**本身就是可靠的"一手结束"信号(与 agari 同理:
        画面事实被加速过场跳掉时, 只能靠动作信号). 状态复位不在本方法里做
        —— 统一走调用方 see() 返回 rearmed=True 之后的那个分支, 一条路.
        """
        self.pending_rearm = True
        self.opened = False
        self.streak = 0
        self.settling = False
        self.grace_until = now + self.grace_period

    def agari(self, now: float) -> None:
        """我们点了和了(自摸 / 荣和): 这一手立刻结束, 封锁到下一次手牌区清空.

        见类注释第 2 条: 摊牌画面上我们自己的和了牌是正面朝上摆着的, 张数
        看起来完全正常, 所以只能靠这个显式信号.
        """
        self.settling = True
        self.opened = False
        self.streak = 0
        self.grace_until = now + self.grace_period


# ---- 牌桌特征在不在(页面门控的状态机) ----
#
# 返回值是四个字符串常量, 不要写成裸字面量, 免得两边拼错还没人发现.
TABLE_OK = "ok"                # 牌桌在(状态已复位)
TABLE_WAITING = "waiting"      # 还没出来, 继续等下一帧
TABLE_GAVE_UP = "gave_up"      # 刚超过上限, 记一次警告后照旧往下走
TABLE_IGNORING = "ignoring"    # 之前就放弃过了, 照旧往下走且不必再记警告


@dataclass
class TableWatch:
    """牌桌特征"还在不在"的状态机(纯逻辑, 状态存这里而不是调用方).

    为什么要单独一个状态机: 这是**页面门控** —— 判"不在牌桌"时整轮跳过感知和
    决策, 所以两个方向都必须管住:

    - 认不出来要有**上限**: 特征万一失灵(分辨率变了 / 标注被误删), 不能把任务
      锁死成"一张牌不打"(那是另一种"打着打着不打了", 比出一手错牌更难查).
      超过 `tuning.TABLE_WAIT_LIMIT` 就放弃等待、退回旧逻辑, 让日志去暴露问题.
    - 重新认出来要能**复位**: 不然下一局的加载阶段会直接沿用"已放弃", 等于这个
      守卫只在第一局有效.
    - 警告只能报**一次**: 主循环每 0.3s 一轮, 不限流就是几百条刷屏 —— 项目里
      踩过同样的坑(`FeatureSet` 那条每 0.5s 一刷的 WARNING 把真正的错误淹了).
    """

    #: 从哪一刻开始认不出来的(0.0 = 此刻认得出来).
    gone_since: float = 0.0
    #: 是否已经放弃等待.
    gave_up: bool = False

    def see(self, present: bool, now: float, *, limit: float) -> str:
        """喂一帧的判定结果, 返回上面四个常量之一."""
        if present:
            self.gone_since = 0.0
            self.gave_up = False
            return TABLE_OK
        if self.gone_since == 0.0:
            self.gone_since = now
            return TABLE_WAITING
        if self.gave_up:
            return TABLE_IGNORING
        if now - self.gone_since >= limit:
            self.gave_up = True
            return TABLE_GAVE_UP
        return TABLE_WAITING

    def gone_for(self, now: float) -> float:
        """认不出来已经多久了(秒). 牌桌在时是 0."""
        return 0.0 if self.gone_since == 0.0 else now - self.gone_since


def can_declare_riichi(tiles: list[str], meld_count: int = 0,
                       *, player_count: int = 3) -> bool:
    """摸牌后的手牌(14 张)能不能立直: 存在一种打法, 打完就听牌.

    **坑(实测)**: jansou 的 `shanten()` 对 14 张手牌返回 -1 -- 它把 14 张
    当成"已经和了", `is_tenpai()` 同理会返回 False. 所以**不能**用
    `is_tenpai_hand(14 张)` 来判断能不能立直, 必须看"打掉一张之后"的向听数.
    (`tests/TestMahjong.py` 里有一条反例守着这个行为.)
    """
    try:
        plans = rank_discards(tiles, meld_count, player_count=player_count)
    except Exception:  # noqa: BLE001
        return False
    return bool(plans) and plans[0].shanten == 0


def conf_of(buttons: Mapping[str, object], name: str) -> float:
    """取按钮的置信度.

    buttons 的 value 可以是数字(置信度), 也可以是 Detection(有 .conf).
    没给置信度的(比如测试里只传了名字)按 1.0 处理, 当作"确定出现了".
    """
    value = buttons.get(name)
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    conf = getattr(value, "conf", None)
    return float(conf) if conf is not None else 1.0


def pick_tile_index(tiles: list[str], tile: str,
                    rng: random.Random | None = None,
                    allowed_indices: list[int] | None = None,
                    drawn_tile: str | None = None) -> int | None:
    """在手牌里找这张牌的位置; **刚摸的那张**(末位)匹配时优先打它.

    `drawn_tile` 是协议给的"刚摸到的那一张"(带赤记号: 三麻是 "0p"/"0s"),
    只在这一手真的握着赤 5 时才起作用 —— 见下面"保留赤 5"那段. 拿不到
    协议读数(None)时整段跳过, 行为与从前完全一致.

    **为什么优先摸牌位**(2026-10-10, 8筒误读局): 旧行为是打最左那张 ——
    手里有重复牌时会点进顺子中间, 视觉上"拆顺"(两张 8p 等价无损, 但用户
    看 789p 中间被抽走一张就是观感异常). 雀魂把刚摸的牌摆在行末独立槽,
    人类玩家打刚摸的那张是压倒性习惯; 服务端看到的"打了什么"两者毫无
    区别, 只有点击落点不同. 判据用"张数 %3==2"(出牌回合手牌 = 暗牌+
    摸牌, 见 is_my_discard_turn 的同一约定), 13 张(应答/副露间隙)不触发.

    **2026-10-08 用户拍板删掉随机出牌**: 出哪张牌由牌效唯一决定, rng 参数
    保留只为兼容旧调用/测试, 已不使用.

    allowed_indices 非None 时只在这些下标里找(立直选牌态): 同一种牌可能
    既有高亮实例又有变暗实例, 必须确保点下去的那张是**高亮**的那张;
    摸牌位优先同样只在允许集合内生效.

    **保留赤 5**(2026-10-18 用户拍板, 只做平局裁决): 视觉是 34 类, 认不出
    赤(屏幕上手牌按花色点数排序, 赤 5 摆在普通 5 **左边**, 唯一例外是刚
    摸到的那张, 它永远在最右边的独立槽). 于是同种牌有多个槽位时:
        刚摸的正是赤 5 -> 最右那张才是赤 -> 打左边那张;
        否则          -> 最左那张是赤   -> 打右边那张.
    哪一种都还是"打一张 5p", 服务端看到的完全一致, 只有赤留不留的差别.

    这条排在摸牌位优先**之前**: 刚摸到赤 5 时摸切会直接把它切掉, 那正是
    要避免的. 拿不到 `drawn_tile` 时按老规矩打, 不猜 —— 白送一翻可以接受,
    打错牌种不行.
    """
    positions = [i for i, t in enumerate(tiles) if t == tile]
    if allowed_indices is not None:
        allow = set(allowed_indices)
        positions = [i for i in positions if i in allow]
    if not positions:
        return None
    if len(positions) > 1 and drawn_tile is not None:
        if drawn_tile == "0" + tile[1]:
            return positions[0]       # 最右是刚摸到的赤 -> 打左边那张
        return positions[-1]          # 最左是赤 -> 打右边那张
    if len(tiles) % 3 == 2 and positions[-1] == len(tiles) - 1:
        return positions[-1]      # 刚摸的那张就是它 -> 直接打摸到的牌
    return positions[0]


def ev_tile_of(row) -> str | None:
    """EV 候选里那一张牌的名字. 接受 `EvDiscard` 对象或裸字符串.

    只 duck-typing 取 `.tile` 而不 import `ev_eval` —— 决策层要能在**没有
    jansou** 的环境里被单独测(整个 tests/TestTable.py 就是这么跑的).
    """
    if isinstance(row, str):
        return row
    tile = getattr(row, "tile", None)
    return tile if isinstance(tile, str) else None


def pick_by_ev_rank(tiles: list[str], ev_rank: Sequence | None,
                    allowed_tiles: set[str] | None = None):
    """按 EV 排序(最优在前)挑**第一张落地得了的牌**, 返回那个候选(或 None).

    只做落地过滤, 不重新排序(排序是引擎的事):
    - 候选不在这一帧认出的手牌里 -> 跳过(协议读数与视觉读数不可能逐张重合);
    - 给了 allowed_tiles(立直选牌态的高亮集合)就只在这个集合里挑.
    全都落不了地时返回 None, 调用方回退到牌效 —— **宁可回退, 不硬打**.
    """
    have = set(tiles)
    for row in ev_rank or ():
        tile = ev_tile_of(row)
        if not tile or tile not in have:
            continue
        if allowed_tiles is not None and tile not in allowed_tiles:
            continue
        return row
    return None


def ev_row_note(row) -> str:
    """EV 候选取一截可读说明(填进 reason). 拿不到就空串."""
    summary = getattr(row, "summary", None)
    if callable(summary):
        try:
            return f"（{summary()}）"
        except Exception:  # noqa: BLE001
            return ""
    return ""


def _button_conf(buttons: Mapping[str, object], name: str) -> float:
    """按钮的置信度; value 是检测框时取框的 conf, 缺按钮时 0."""
    value = buttons.get(name)
    if value is None:
        return 0.0
    conf = getattr(value, "conf", value)
    try:
        return float(conf)
    except (TypeError, ValueError):
        return 0.0


#: 两个**不同名**的吃/碰/杠按钮框中心距离低于该值(像素) = 同一个按钮被
#: UI 模型认成了两个类别(双标签), 这一帧的按钮读数不可信.
#:
#: 数据依据(2026-10-08 对 data/debug/game_*.jsonl 全量 button_centers 统计):
#:   - 「吃」的 3 次检出全部和同帧的 gang 框中心只差 7~8px —— 两个按钮
#:     不可能叠在一起, 那是同一个按钮(位置与 gang 的常驻位 1025 吻合)被
#:     多贴了一个 chi 标签;
#:   - peng+gang 同帧也出现过中心只差 3~5px 的记录, 同理;
#:   - 真正同时出现的两个按钮中心距 >= 280px.
#: 60 取在两者正中, 两端裕度都够. 触发时**连"跳过"都不点**: 重叠帧上
#: 分不清哪个标签是真的, 点跳过可能把真的鸣牌机会一起跳掉, 等下一帧
#: (机会窗口有几秒, 等得起; 画面静止时重叠不会自己消失, 雀魂会自动跳过).
MELD_LABEL_OVERLAP_TOL = 60.0


def meld_label_conflict(centers: Mapping[str, Sequence[float]], *,
                        tol: float = MELD_LABEL_OVERLAP_TOL) -> bool:
    """吃/碰/杠里是否有两个不同标签的框中心几乎重合(见常量说明)."""
    names = [n for n in BUTTON_MELD if n in centers]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ax, ay = centers[a][0], centers[a][1]
            bx, by = centers[b][0], centers[b][1]
            if math.hypot(ax - bx, ay - by) < tol:
                return True
    return False


#: 动作前双帧确认允许的按钮中心位移(像素, 见 action_confirmed).
#: 滑入动画每 0.4s 移动几百像素, 静止按钮的帧间抖动只有几个像素,
#: 24 取在中间, 两端裕度都够.
BUTTON_STABLE_TOL = 24.0


def action_confirmed(first: TurnDecision, second: TurnDecision,
                     first_center: Sequence[float] | None = None,
                     second_center: Sequence[float] | None = None, *,
                     tol: float = BUTTON_STABLE_TOL) -> tuple[bool, str]:
    """动作前的双帧一致性确认(2026-10-08 用户提议"多截几次取最稳"落地).

    背景: 21:45 局的乱出牌/漏碰全发生在"牌桌还没稳定下来"的时候 ——
    过场画面的幻觉读数、滑入动画里的低置信按钮, 都撑不过第二帧.
    判据: 两次决策**同动作、同按钮**; meld 还要求按钮框中心位移 <= tol
    (还在动 = 还在滑入 = 先别点); discard 还要求要打的牌相同.

    第二帧怎么来的(调用方负责): 等 ~0.4s 重抓一帧, 用同参数重跑
    decide_turn. 不一致就返回 False, 调用方本轮不动、回主循环重判 ——
    主循环本来就每 0.3s 一轮, 天然重试, 不需要超时兜底也不会锁死;
    真正稳定的画面下一轮必然通过, 代价只是每动作多 ~0.55s.
    """
    if first.action != second.action:
        return False, f"动作变了（{first.action} -> {second.action}）"
    if first.button != second.button:
        return False, f"按钮变了（{first.button} -> {second.button}）"
    if (first.action == ACTION_MELD and first_center is not None
            and second_center is not None):
        dist = math.hypot(first_center[0] - second_center[0],
                          first_center[1] - second_center[1])
        if dist > tol:
            return False, (f"按钮还在动（位移 {dist:.0f}px > {tol:.0f}px），"
                           f"多半在滑入动画里")
    if first.action == ACTION_DISCARD and first.tile != second.tile:
        # 摸切例外: 两帧都决定打"最右那张刚摸的牌"(同下标 = 物理上同一张),
        # 标签可以不一致 —— 摸切守卫(8.0.5)的本意就是不信任那张牌的标签,
        # 再拿标签做一致性判据就自相矛盾了(误读牌两帧读成不同的名字很常见).
        if (first.index == second.index and first.index is not None
                and "摸切" in first.reason and "摸切" in second.reason):
            return True, "两帧同位置摸切（标签不同但物理同张）"
        return False, f"要打的牌变了（{first.tile} -> {second.tile}）"
    return True, "两帧一致"


#: 暗杠/加杠的"形中性"半径. 与要杠的牌相距这么多以内的同花色牌, 都可能
#: 和它组成顺子: 7 的 ±2 正好覆盖 567 / 678 / 789 三种含 7 的顺子.
_KAN_NEIGHBOR_REACH = 2


def kan_keeps_shape(tile: str, tiles: Sequence[str]) -> bool:
    """杠掉这张牌会不会拆散手里的搭子.

    照抄 jansou 的 `_shape_neutral`(game/agents.py, 私有) 的判据, 不 import
    它 —— 那个模块会把 game 子系统(action/event/flow)整个拖进来, 而本项目
    只依赖 analysis 与 core 两层. 三麻只有 1m/9m, 半径判定照样成立(1m 的
    邻域里只可能有 9m, |9-1|=8 > 2, 恒为中性, 等价于"1m 不成顺").

    字牌恒为 True: 字牌组不成顺子, 杠掉它拆不了任何搭子.

    `tiles` 里那几张要杠的牌自己会被跳过 —— 否则 4 张 7m 会自己判自己
    "形不中性", 暗杠永远做不成.
    """
    if not tile or len(tile) != 2 or tile[1] == "z":
        return True
    suit = tile[1]
    rank = 5 if tile[0] == "0" else int(tile[0])   # 赤 5 折叠
    for other in tiles:
        if len(other) != 2 or other[1] != suit:
            continue
        other_rank = 5 if other[0] == "0" else int(other[0])
        # 按**折叠后的点数**判是不是同一张, 不能用 `other == tile`: 协议写
        # 赤 5 作 "0p" 而普通 5 是 "5p", 字符串不相等 —— 于是手里那张赤 5p
        # 会把自己判成"形不中性", 暗杠 5p 永远做不成(2026-10-18 探针实锤).
        if other_rank == rank:
            continue
        if abs(other_rank - rank) <= _KAN_NEIGHBOR_REACH:
            return False
    return True


def pick_meld_button(tiles: list[str], buttons: Mapping[str, object], *,
                     meld_policy: str, missing_slots: int = 0,
                     meld_min_conf: float = 0.0,
                     melded: Sequence[str] = ()) -> str | None:
    """第 5 步"点哪个吃/碰/杠按钮"的纯选择器: 返回按钮名或 None.

    单独抽出来是因为双标签重叠守卫(见 decide_turn 第 5.0 步)必须回答
    "若没有重叠, 这一手会不会点鸣牌" —— 判据若两处各写一份, 迟早漂移.
    kan_only 的暗杠分支判据不同(四张 + 我方出牌回合), 不在这里.

    这里只会选**暗杠/加杠**(张数 %3==2 的那一刻), 大明杠恒不选: 它是"别人
    打出的牌 + 手里三张", 破门清且让这一手立刻无役 —— 我们只碰役牌、靠门清
    立直, 这条禁令比任何置信度都硬(见下面循环里的说明).

    melded 是**我方本手副露里的牌**(由任务层从手牌差集反推, 见
    meld_taken_kinds): 杠的手牌支持判据必须覆盖**加杠** —— 副露里已有
    3 张同种 + 手里 1 张(刚摸到第 4 张). 只看暗杠判据(tiles.count>=3)
    时加杠轮的 conf(0.718) 会被 0.72 兜底门槛误杀(2026-10-09 東2局实锤:
    碰了 9s 后摸到第 4 张, 加杠机会白白放过还把它打了出去).
    """
    if meld_policy not in ("none", "kan_only"):
        names = (BUTTON_MELD if meld_policy == "all"
                 else (BUTTON_PENG, BUTTON_GANG))
        for name in names:
            if name not in buttons:
                continue
            if name == BUTTON_GANG and not is_my_discard_turn(len(tiles)):
                # **大明杠一律不做**(2026-10-18 实跑: 别家打 2z、我们手里三张,
                # 点了 gang —— 客风杠出去门清没了、这一手直接变成无役, 后面
                # 听了 1m/4m 也荣和不了, 白打一手).
                # 判据是**张数**: 暗杠/加杠只可能发生在"自己刚摸完牌"那一刻
                # (14/11/8/5/2 张, %3==2); 大明杠是"别人打出的牌 + 手里三张",
                # 那时暗牌是 13/10/7(%3==1), 两个集合不相交.
                # 放在 supported/conf 判断**之前**: 大明杠是策略禁令, 不是
                # "置信度不够就先放过"—— conf 再高也不点.
                continue
            if name == BUTTON_PENG:
                supported = (not missing_slots
                             and any(tiles.count(t) >= 2 for t in set(tiles)))
            else:
                # 只剩暗杠与加杠(大明杠上面已经拦掉).
                # 暗杠: 手里四张同牌, 且不能拆了手里的搭子(见 kan_keeps_shape).
                #   手里 7777 而旁边还搁着 89m 时, 暗杠会把 789m 一并拆掉 ——
                #   为翻数退向听, 不值.
                # 加杠: 副露里已碰的三张 + 手里刚摸到的第四张. **不需要形中性
                #   那一关** —— 加杠的对象必然是碰过的牌, 而我们只碰役牌, 役牌
                #   全是字牌, 字牌组不成顺子, 自然拆不了任何搭子.
                #   (顺带: 三麻的 1m/9m 同样不成顺, 所以三麻里形中性只对
                #   筒子索子真正生效, 万子与字牌恒为中性.)
                supported = (not missing_slots and (
                    any(tiles.count(t) >= 3 and kan_keeps_shape(t, tiles)
                        for t in set(tiles))
                    or any(melded.count(t) >= 3 and tiles.count(t) >= 1
                           for t in set(melded))))
            if not supported and _button_conf(buttons, name) < meld_min_conf:
                continue    # 手牌不支持 + 低置信 = 多半是误检, 宁可跳过
            if (name == BUTTON_PENG and meld_policy != "all"
                    and not has_yaku_pair(tiles, missing_slots)):
                continue    # 没役的碰会把这手碰死, 落到第 6 步跳过
            return name
    return None


def would_click_meld(tiles: list[str], buttons: Mapping[str, object], *,
                     meld_policy: str, missing_slots: int = 0,
                     meld_min_conf: float = 0.0,
                     melded: Sequence[str] = ()) -> bool:
    """重叠帧专用: 若没有重叠, 这一手会不会点吃/碰/杠?(判据在 pick_meld_button)"""
    if meld_policy == "none":
        return False
    if meld_policy == "kan_only":
        return (BUTTON_GANG in buttons
                and is_my_discard_turn(len(tiles)) and not missing_slots
                and any(tiles.count(t) >= 4 and kan_keeps_shape(t, tiles)
                        for t in set(tiles))
                and _button_conf(buttons, BUTTON_GANG) >= meld_min_conf)
    return pick_meld_button(tiles, buttons, meld_policy=meld_policy,
                            missing_slots=missing_slots,
                            meld_min_conf=meld_min_conf,
                            melded=melded) is not None


def decide_turn(
    tiles: list[str],
    buttons: Mapping[str, object],
    *,
    player_count: int = 3,
    meld_policy: str = "none",
    allow_riichi: bool = True,
    riichi_min_conf: float = 0.60,
    meld_min_conf: float = 0.0,
    babei_policy: str = "auto",
    allowed_indices: list[int] | None = None,
    blurry_indices: list[int] | None = None,
    robust_discard_enabled: bool = True,
    missing_slots: int = 0,
    button_centers: Mapping[str, Sequence[float]] | None = None,
    blurry_draw: bool = False,
    melded: Sequence[str] = (),
    my_turn: bool | None = None,
    proto_turn: bool | None = None,
    proto_meld: bool | None = None,
    round_open: bool | None = None,
    ev_rank: Sequence | None = None,
    rng: random.Random | None = None,
    drawn_tile: str | None = None,
) -> TurnDecision:
    """决定这一轮干什么.

    buttons 只要有 key 就算"画面上出现了", value 可以是置信度或检测框.
    任何一步算不出来(手牌张数不对 / 认出了三人麻将里不该有的牌)都不会抛,
    而是退化成 wait 并把原因写进 reason -- 对局里宁可多等一轮也不要乱点.

    blurry_indices 是"牌面模型认不准的槽位下标", 由调用方按置信度给.
    非空时第 8 步会走稳健出牌(见 robust_discard 的说明): 枚举这些牌可能的
    真实取值,挑一张"怎么认都不会错"的打. 置 robust_discard_enabled=False
    可以关掉, 恢复成普通的纯牌效最优.

    missing_slots 是"按间距看出来漏认了几张"(src/detection.missing_slots).
    它和置信度无关: 认出来的每张都很确定, 但行里有一段空档 -> 张数就是错的.
    **这个数非 0 时不能按张数下任何结论**, 尤其不能把它当成"在等别人":
    10 张是"1 副露"的合法张数, 于是 `is_my_discard_turn` 为假, 决策层会
    输出一句不含 suspect 的 "在等别人", 上层既不计数也不存帧, 一路静默
    空转到卡死兜底(2026-10-07 实跑, 用户报的"打着打着就不打了").

    button_centers 是各按钮框中心的像素坐标(调用方从检测框算好传进来),
    只用于第 5.0 步的双标签重叠守卫(见 meld_label_conflict); 不传就跳过
    那道守卫(测试与旧调用兼容).

    blurry_draw 是"刚摸的那张牌(最右槽)曾经低置信过"的粘性标志(调用方
    在本轮摸牌期间任何一帧见到就置位, 打出这张牌之前不清), 用于第 8 步的
    听牌摸切守卫: 听牌时误读刚摸的牌会算出"打别的牌也听"的假象, 实际
    退向(2026-10-09 实锤: 刚摸的 8p 被读成 4p, 牌效按幻影 4p 算出打 6m
    "保持听牌", 真手牌直接退向). 摸切是唯一**位置无关**的安全打 —— 不管
    那张牌实际是什么, 打出去都不改变听牌状态.

    my_turn 是**协议对"现在是不是我该动"的权威说法**(见
    net.state.protocol_says_my_turn): 只有接管档会传, 且桥不新鲜时调用方传
    None(不参战). 传 False = 服务端明确"不轮到你" -> 这一轮一律不动, 挡的是
    "过场画面里牌河被读成手牌、张数刚好合法, 于是点出去一张牌"那类事故
    (见 hand_count_ok 的说明). 传 True **不改变**任何决策 —— 协议只做否决,
    不做指令: 它说"轮到你"也仍要看牌效算出来的结果.

    proto_turn 是同一个协议说法的**窄口径**(调用方**每个档位都传**), 只用来
    过滤**立直按钮**: 立直只可能出现在"我自己摸牌后正要出牌"那一刻, 协议说
    这一刻没有我的出牌操作(type 1)就说明那个按钮是误读(2026-10-17 东三局:
    假 lizhi 0.714 出现在协议给"碰"的那一帧)。与 my_turn 分开的理由是**代价
    不对称**: 漏立直只少一番, 而把"过期的操作列表"当否决去挡出牌会把整手牌
    拖进托管 —— 所以整轮否决仍然只在接管档, 这道只挡立直。

    proto_meld 是协议说"此刻开着**给我的**鸣牌窗口"(offered 里有吃/碰/大明杠)。
    它存在的唯一理由是给 my_turn 那道整轮否决开一个口子: 鸣牌窗口里协议本来
    就只给鸣牌操作、不给打牌操作, my_turn=False 是**正常读数**而不是"不轮到你"
    (见第 1.5 步)。传 None(拿不到协议/关着) = 不开口子, 否决照旧。

    ev_rank 是**出牌的新目标函数**(见 tuning.EV_EVAL_TAKEOVER): 由调用方把
    `ev_eval.evaluate_discards()` 的结果原样传进来, 最优在前. 有它时, "出牌
    挑哪一张"由它决定(期望打点 = 和牌率 x 和牌打点, 扣着牌河算进张、带着
    宝牌算打点); 没有 / 落不了地 / 引擎给不出结论 -> 回退到原来的牌效
    (`choose_discard`). 它**只替换"挑哪一张"** —— 稳健出牌、听牌摸切守卫、
    `allowed_indices`、张数与可信度闸门全部在它之前, 坏输入根本到不了这一步;
    立直选牌态里也只用它在高亮集合内挑(见第 3 步).

    """
    rng = rng or random

    # 1) 和牌 / 自摸: 永远第一优先
    for name in BUTTON_WIN:
        if name in buttons:
            return TurnDecision(ACTION_WIN, button=name, reason="能和就和")

    # 1.5) 协议权威闸(见 my_turn 的说明): 服务端明确说"不轮到你" -> 不动.
    #
    # **必须放在和牌之后**: 桥可能滞后半拍(操作列表还没推过来), 而"漏和"的
    # 代价(整手白打)远大于"少鸣一次", 所以和牌/自摸永远优先, 协议无权否决它.
    # 这一闸挡的不是"时机判断不准", 而是**幻觉读数点出去一张牌**: 过场画面里
    # 牌河被读成手牌、张数刚好落在合法集合的那类事故, 视觉侧所有闸门都会放行,
    # 而协议说"不是你"是当时能拿到的最硬证据.
    if my_turn is False and not (proto_meld
                                 and not is_my_discard_turn(len(tiles))):
        # 例外(2026-10-17 真机实锤, 2026-10-21 改挂在协议上): **鸣牌窗口本身
        # 就是"有吃/碰/杠操作、无出牌操作"** —— 夹具 `liqi_sanma_2rounds`
        # 里两次 `ActionDiscardTile` 给的 operation_list 都是 `[3]`(只有碰),
        # 没有 type 1, 所以 my_turn=False 恰恰是"这是个鸣牌窗口"的**正常**读数,
        # 拿它去否决就把整个窗口拦成 wait/skip(那一局 23 次想碰, 一嘴都没点
        # 出去). `proto_meld` 是协议对"此刻开着给我的鸣牌窗口"的正面说法
        # (offered 里有 2 吃 / 3 碰 / 5 大明杠), 与 my_turn 同源同硬; 两者同时
        # 成立 = 协议确认这是鸣牌窗口 -> 不否决, 交给第 5 步的规则判"点不点"
        # (它自己会因为该我出牌/没役对/按钮不在画面而不鸣). 其余情形(残留
        # 按钮/幻觉读数, proto_meld 为 None 或 False)照旧一律不动.
        #
        # 视觉侧还要**独立同意**(`not is_my_discard_turn`): 14 张是我刚摸完牌
        # 的形状, 那时协议还挂着鸣牌机会说明桥滞后了一拍 —— 那种时候不值得
        # 为它放开整轮否决.
        #
        # 二次实锤(2026-10-17 17:59 局): 决定"不鸣"时老逻辑在窗口里干等
        # 整段倒计时(~10s, jsonl 连续 20+ 条 wait) —— 但倒计时走完的终点
        # 本来就是"自动不鸣", 等待只是白耗全桌时间. 这道闸的前提取自
        # _proto_clock_read: offered 非空且无出牌 = 协议**确认**此刻开着
        # 一个操作窗口, 视觉里的「跳过」按钮是真按钮, 点它 = 立刻 Decline,
        # 游戏马上往下走. 护栏: 拔北窗口不走这里(拔北永远拔, 见下)、
        # 张数必须还是 13(鸣牌窗口的形状; 幻觉读数的张数五花八门, 让它
        # 继续吃 WAIT)、按钮必须真的在画面. 顺序 bug(不鸣被误判成可鸣)
        # 的方向是"多等 10 秒", 不会误点 —— 与"误立直"不同, 没有状态机
        # 代价, 点击走按钮动作的双帧确认.
        if (BUTTON_SKIP in buttons and BUTTON_BABEI not in buttons
                and not is_my_discard_turn(len(tiles))):
            return TurnDecision(
                ACTION_SKIP, button=BUTTON_SKIP,
                reason="协议确认是鸣牌窗口且决定不鸣, 点跳过立刻过")
        return TurnDecision(
            ACTION_WAIT,
            reason="协议说现在不轮到我，这一轮不动（防幻觉读数点出去）")

    concealed = len(tiles)
    # sane = "张数像真的认全了". 比纯数学的合法张数多一道下限, 见
    # is_plausible_concealed 的说明: 1~2 张在数学上合法(4 副露), 但实战里
    # 只见于"手牌区空着, 模型幻觉出几张", 而那种情况**不能**被当成
    # "在等别人" -- 那会让上层既不计数也不截图, 静默空转.
    sane = is_plausible_concealed(concealed)

    # 出牌理由里带一句"这手牌其实没认全", 便于事后对着留证帧复核.
    # 注意: 空档时不拒绝出牌 —— 拒绝就等于空转, 而空转正是要修的那个 bug;
    # 稳健出牌只能枚举低置信**槽位**的取值, 覆盖不了"整个槽位都没认出来",
    # 所以这里只能如实标注, 不能假装它可靠.
    gap_note = f"；行里还有约 {missing_slots} 张的空档" if missing_slots else ""

    # 2) 立直: **以按钮为准**.
    #
    # 雀魂只在"真的能立直"时才画这个按钮, 它替我们判断了一大堆 jansou 看不到
    # 的条件: 门清 / 点数够 1000 / 牌山还剩 4 张以上 / 振听 / 形式听牌(没役的
    # 听牌不能立直). 这些从我们认出来的手牌里**全都算不出来** -- 我们连自己
    # 有没有副露都是从暗牌数反推的, 认漏一张就推错.
    #
    # 反过来手牌判断只能当"交叉验证": 如果日志里常出现"按钮说能立直但手牌
    # 算出来不听", 那是在提示手牌识别不准(或副露数推错), 应该去修感知层,
    # **不是拿它否决按钮** -- 那等于用更不可靠的信号否决更可靠的信号.
    #
    # 唯一的硬保护是置信度: 立直不可撤销, 按钮认得不够确定就宁可不点.
    if allow_riichi and BUTTON_RIICHI in buttons:
        # 立直按钮**只可能出现在"我自己摸牌后、正要挑一张打出去"的那一刻**.
        # 这条前提有两个独立、都不贵的检查, 任一不成立就是误检:
        #   ① 张数必须是"摸牌后"那一档(14/11/8/5/2);
        #   ② 协议(可用时)必须说这一刻有我的出牌操作(type 1).
        # 2026-10-17 东三局: 那一刻手里 **13 张**、协议给的可用操作是
        # **碰**(type 3) —— 两条任意一条都能拦住那个 lizhi(conf 0.714);
        # 而它是从"上家立直"的宣告画面里被误读出来的(用户核对回放指出).
        #
        # 为什么单独给立直加这两道: 立直误判的代价**不对称** ——
        # 漏立直只少一番, 这手照打, 下一帧按钮还在; 误立直则把状态机带进
        # "我在挑宣言牌", 整手牌报废(见 PlayOneGameTask 立直状态机).
        # 因为只否决立直、否决不了任何别的动作, 所以两条都可以不看档位,
        # 不会有"把整手牌拖进托管"的反向风险.
        #
        # 张数那道只在**读数够多**时才敢当反证(concealed >= MIN_CONCEALED):
        # 手牌区几乎空着时读数本身就是幻觉, 拿它去否决一个真按钮违背了
        # "按钮比手牌判断可靠"这条既定原则(见上面 2) 的说明).
        if concealed >= MIN_CONCEALED and not is_my_discard_turn(concealed):
            return TurnDecision(
                ACTION_WAIT,
                reason=f"立直按钮出现在手牌 {concealed} 张这一刻"
                       f"（不是摸牌后的张数），当误检")
        if proto_turn is False:
            return TurnDecision(
                ACTION_WAIT,
                reason="协议说这一刻没有我的出牌操作，立直按钮当误检")
        if round_open is False:
            # 协议说这一手已经结束了(胡牌/流局看过收尾报文) —— 结算画面上
            # 残留的手牌行读出的"立直按钮"只能是误读(2026-10-17 真机:
            # 别家胡牌后 12s, 我们在结算画面上"稳健打出"了一张 9p).
            return TurnDecision(
                ACTION_WAIT,
                reason="协议说这一手已经结束，立直按钮当误检")
        conf = conf_of(buttons, BUTTON_RIICHI)
        if conf < riichi_min_conf:
            return TurnDecision(
                ACTION_WAIT,
                reason=f"立直按钮置信度 {conf:.2f} 低于 {riichi_min_conf}，当误检")
        note = "手牌没认全，以按钮为准"
        if sane and is_my_discard_turn(concealed):
            note = ("手牌算出来打一张就听牌"
                    if can_declare_riichi(tiles, meld_count_of(concealed),
                                          player_count=player_count)
                    else "手牌算出来打完不听，以按钮为准")
        return TurnDecision(ACTION_RIICHI, button=BUTTON_RIICHI,
                            reason=f"雀魂给了立直按钮(conf={conf:.2f})；{note}")

    # 3) 立直选牌态: **直接出牌**, 牌效只在高亮牌里挑. 优先级高于拔北/
    #    吃碰杠/跳过 -- 选牌态下点别的按钮都是无效操作, 而点高亮牌就完成
    #    了宣言, 这是当前唯一能推进局面的动作.
    #
    # 前提是"**摸牌后**、手里 14/11/8/5/2 张, 要挑一张当宣言牌". 张数不是那
    # 一档就说明这个"选牌态"是误判, 当场作废、落回普通判断 —— 2026-10-17
    # 东三局实锤: 一个假立直按钮(conf 0.714, 协议当帧就判为误检)把状态机
    # 带了进来, 之后整整 26 秒拿着 **13 张牌** 反复走下面的牌效, 每次都抛
    # `needs a post-draw hand` —— 而那段没有出口, 只能等摸到第 14 张才自己
    # 解开; 摸到之后又把一张普通牌当宣言牌打了出去, 置上"立直生效", 整局
    # 不再主动出牌, 出牌权全被雀魂托管接管.
    if allowed_indices is not None and not is_my_discard_turn(concealed):
        allowed_indices = None
    if allowed_indices is not None:
        allowed = [i for i in allowed_indices if 0 <= i < len(tiles)]
        if not allowed:
            # 认不出高亮牌就空等一轮, 不猜. 猜错点在变暗的牌上, 雀魂
            # 根本不响应, 那是纯白等(实跑里白等过 5 轮).
            return TurnDecision(ACTION_WAIT,
                                reason="立直选牌态但没认出高亮牌, 重新认一轮")
        if not sane:
            return TurnDecision(
                ACTION_WAIT, suspect=True,
                reason=(f"只认出 {concealed} 张牌（低于下限 {MIN_CONCEALED}）"
                        if concealed < MIN_CONCEALED else
                        f"手牌 {concealed} 张不是合法张数，可能认漏了"))
        # 立直选牌态里**不能**走稳健出牌: 那里已经限定"只打高亮牌", 再枚举
        # 槽位取值没有意义(高亮是雀魂给的确定集合). 所以这里保守停手,
        # 但 reason 要点名槽位, 便于对照留证帧排查.
        # 2026-10-06 23:21 实跑的卡死不在这里(普通出牌回合, 见第 8 步).
        bad_idx = out_of_play_indices(tiles, player_count)
        if bad_idx:
            return TurnDecision(
                ACTION_WAIT, suspect=True,
                reason=f"立直选牌态, 但认出了 {player_count} 人麻将里没有的牌 "
                       f"{[tiles[i] for i in bad_idx]}（槽位 {bad_idx}），"
                       f"先不动(选牌态不走稳健出牌)")
        allowed_tiles = {tiles[i] for i in allowed}
        # 宣言牌优先用 EV 排序(与高亮集合取交), 引擎给不出结论就回退牌效.
        row = pick_by_ev_rank(tiles, ev_rank, allowed_tiles)
        if row is not None:
            tile = ev_tile_of(row)
            index = pick_tile_index(tiles, tile, rng, allowed_indices=allowed,
                                    drawn_tile=drawn_tile)
            if index is not None:
                return TurnDecision(
                    ACTION_DISCARD, tile=tile, index=index,
                    reason=f"立直宣言牌(按EV排序{ev_row_note(row)}，只打高亮牌)")
        try:
            plan = choose_discard(tiles, meld_count_of(concealed), rng=rng,
                                  player_count=player_count,
                                  allowed_tiles=allowed_tiles)
        except Exception as e:  # noqa: BLE001
            return TurnDecision(ACTION_WAIT, suspect=True,
                                reason=f"牌效计算失败({e})")
        index = pick_tile_index(tiles, plan.tile, rng,
                                allowed_indices=allowed,
                                drawn_tile=drawn_tile)
        if index is None:
            return TurnDecision(ACTION_WAIT, suspect=True,
                                reason=f"要打的 {plan.tile} 不在认出的高亮牌里")
        return TurnDecision(ACTION_DISCARD, tile=plan.tile, index=index,
                            reason=f"立直宣言牌({plan.picked} 向听{plan.shanten} "
                                   f"进张{plan.acceptance}，只打高亮牌)")
    allowed = None

    # 4) 拔北: 见北就拔(见模块说明).
    #
    # 判据"手里有没有北"而不是"这是不是我的回合": 拔北按钮本身就只在能拔的
    # 时候才画出来, 雀魂已经替我们判过一遍了(轮到自己 + 手里有北).
    # 早先的错判是把babei 归进 IGNORED_BUTTONS, 理由是"拔北出现时北是别人
    # 打出来的" -- 实跑证明那是错的, 3 次 babei 全都发生在手里有 3~4 张北
    # 的出牌回合.
    babei_offered = BUTTON_BABEI in buttons
    if babei_offered and babei_policy != "never":
        # babei_policy="auto" 只在"轮到自己出牌"时拔. 万一这个按钮是别人
        # 打北时的应答机会(那种情况我们手牌是 13 张), 拔了也合法, 但那样
        # 就不是在给自己换牌而是纯赚一翻, 行为上更可疑 -- 而且早先那 3 次
        # 误判的教训是别在看不懂的按钮上乱动手.
        if babei_policy == "always" or is_my_discard_turn(concealed):
            return TurnDecision(ACTION_BABEI, button=BUTTON_BABEI,
                                reason="手里有北, 拔一张换牌(等同加杠)")

    # 5) 吃碰杠: 现行档 `MELD_POLICY="peng"` —— 吃永不点(会破门清且要选组合条),
    #    碰只在手里有役牌对子时点, 杠见下.
    #
    # 5.0) 双标签重叠守卫(2026-10-08 新增, 同晚实跑后**条件化**): 同一个
    #      按钮被认成两类时(见 MELD_LABEL_OVERLAP_TOL 的数据依据), 标签
    #      真假分不清 —— **若这一手本来就会点其中之一**, 只能等(滑入动画
    #      的双标签一两帧就散); 若本来就不会点(典型: 吃这类我们从不做的
    #      机会), 直接落到第 6 步跳过. **无条件 WAIT 是错的**, 22:57 局
    #      实锤: 「吃」被双标成 gang+chi 且静态不动, 跳过明明在按钮字典里,
    #      守卫却让我们在机会窗口里挂了 16 秒+, 直到托管自动吃牌 —— 用户
    #      报的"还在吃牌"就是它.
    if (button_centers and meld_label_conflict(button_centers)
            and would_click_meld(tiles, buttons, meld_policy=meld_policy,
                                 missing_slots=missing_slots,
                                 meld_min_conf=meld_min_conf,
                                 melded=melded)):
        return TurnDecision(
            ACTION_WAIT, suspect=True,
            reason="吃/碰/杠按钮标签互相重叠（同一按钮被认成两类），"
                   "而这一手本来会点其中之一，真假分不清，先不动")
    #
    # 杠要单独放出来: 吃碰会破门清, 而**暗杠不破门清**(日麻里暗杠/ 加杠
    # 都不影响门清, 立直后也可以杠), 所以策略上接受杠但仍然不碰不吃.
    # 实跑踩过: 手里 4 张北时雀魂给了 gang(conf=0.842), 因为
    # meld_policy="none" 被白白放过, 直接打了张北出去.
    if meld_policy == "kan_only" and BUTTON_GANG in buttons:
        # gang 按钮有两种来源, 必须区分(2026-10-07 19:13 实跑翻车):
        #   明杠 = 别人打牌我们手里有三张 -> **破门清**, 杠了就立直不了、
        #          门清和了也没了. 那一局手牌 13 张、9m x3, 旧代码不分青红
        #          皂白按"手里四张"点了杠, 一手门清听牌直接拆掉.
        #   暗杠 = 自己的出牌回合(张数 % 3 == 2)且手里真有四张同种
        #          -> 不影响门清, 杠.
        # 判据以张数为准; 张数没认全(missing_slots > 0)时四张也可能是数错
        # 的, 宁可放过一次暗杠也不冒明杠的险.
        kan = (is_my_discard_turn(concealed) and not missing_slots
               and any(tiles.count(t) >= 4 for t in set(tiles)))
        if kan and _button_conf(buttons, BUTTON_GANG) >= meld_min_conf:
            return TurnDecision(ACTION_MELD, button=BUTTON_GANG,
                                reason="手里四张, 暗杠不破门清")
        # 明杠 / 看不懂的杠机会: 落到第 6 步点"跳过", 保持门清.
    if meld_policy != "none":
        # "peng" = 碰只在手里有役牌对子时点(见 has_yaku_pair); 杠只点**暗杠
        # 与加杠**(不破门清), **大明杠不做**(破门清 + 立刻无役, 见
        # pick_meld_button 里的张数判据); 吃永不点 —— 有多种吃法时雀魂会弹
        # 组合选择条, 那个界面没有建模, 点了吃按钮弹出来就卡死. 碰/杠永远是
        # 单一选项, 点按钮直接生效.
        # "all" = 不设条件全点(仅在明确要求时启用).
        # **按钮真实性 = 手牌支持**(2026-10-08 两次实锤合订):
        #   - 20:47 局: 「吃」被 UI 模型误报成 gang(0.52), "杠都点"点了,
        #     实际执行了吃 —— 那次手里**没有三张同牌**, 杠机会不可能存在.
        #   - 21:45 局: 手握 5z 对子, 真的弹了 peng 按钮, 但滑入动画期
        #     conf 只有 0.25/0.43, 被旧的"conf>=0.72"门槛误杀, 白板没碰成.
        # 雀魂只在真的可碰/可杠时才画按钮, 所以**手牌支持才是按钮真实性的
        # 最强证据**: 碰需要手里有对子、杠需要三张以上同种. 手牌支持就不看
        # conf(动画帧低置信无妨); 手牌不支持时 conf 再高也多半是认错了按钮
        # (保留 conf 门槛兜底, 防的是"读数本身不可信"的场景).
        # 选择器抽成 pick_meld_button(见其说明): 双标签守卫要回答
        # "没有重叠时会不会点", 判据必须和这里同源, 不能各写一份.
        name = pick_meld_button(tiles, buttons, meld_policy=meld_policy,
                                missing_slots=missing_slots,
                                meld_min_conf=meld_min_conf, melded=melded)
        if name:
            if name == BUTTON_PENG:
                reason = f"有役牌对子, {meld_policy}"
            else:
                reason = f"meld_policy={meld_policy}"
            return TurnDecision(ACTION_MELD, button=name, reason=reason)

    # 6) 跳过: 有吃碰杠机会但我们不要(拔北的取消键除外, 见上)
    #
    # **该我出牌时一律不点跳过**(2026-10-06 实跑发现): 手里有 14 张说明刚摸完
    # 牌, 轮到自己, 雀魂此时不会给我们吃碰杠的机会 -- 画面上那个"跳过"要么是
    # 上一轮残留, 要么是 UI 模型误检. 实测一局里点了 8 次跳过, 其中 5 次是
    # 这种白点, 每次白花 2~3 秒(拟人延时 + 点击 + 抓帧), 而且让节奏变得很奇怪:
    # "想一下 -> 点跳过 -> 再想一下 -> 才打牌".
    # 雀魂不可能在 14 张暗牌时给碰/吃, 所以这个判断是安全的.
    if BUTTON_SKIP in buttons and not babei_offered and not is_my_discard_turn(concealed):
        return TurnDecision(ACTION_SKIP, button=BUTTON_SKIP, reason="跳过吃碰杠")

    # 7) 确认: 流局 / 结算
    for name in BUTTON_CONFIRM:
        if name in buttons:
            return TurnDecision(ACTION_CONFIRM, button=name, reason="确认")

    # 8) 出牌
    if is_my_discard_turn(concealed) and sane:
        if round_open is False:
            # 收局守卫(2026-10-17 真机实锤): 别家胡牌后, 和了演出/结算画面
            # 底下的手牌行还在, 视觉照样能读出一手"稳定"的牌并决定出牌 ——
            # 那一刻桥已报"这一手结束"(round_open=False), 协议其余闸门都因
            # available()=False 而不参战, 这里是唯一还开着的手电筒. 胡牌之后
            # 不存在任何合法出牌, 这张读数必然是画面残留, 不出.
            return TurnDecision(
                ACTION_WAIT,
                reason="协议说这一手已经结束（有人胡牌/流局），结算画面的手牌"
                       "是残留读数，不出牌")
        # 8.0) 读数可信度硬闸(2026-10-08 21:45 局实锤): "东2局0本场"过场画面
        # 上手牌行位置摆的是牌桌残留牌, 模型读出 8 张全低置信(conf 0.15-0.72),
        # 张数 8 恰好是"2 副露"的合法张数 -- 闸门和张数判据全被绕过, 稳健
        # 枚举 0 种可能却照样打出 5z, 连着两局各乱打一张. 低置信槽位过半
        # = 这行读数多半不是手牌, 宁可等一轮(StuckDetector 有兜底)也别拿
        # 幻觉牌下决策.
        if blurry_indices and len(blurry_indices) > concealed // 2:
            return TurnDecision(
                ACTION_WAIT, suspect=True,
                reason=f"{len(blurry_indices)}/{concealed} 张低置信超过手牌一半，"
                       f"读数不可信，先不动")
        # "这局不可能的牌"(三人麻将的 2m-8m)是**独立于置信度**的第二类误识
        # 信号, 必须和低置信槽位一起喂给稳健出牌.
        # 2026-10-06 23:21 实跑: 8s 被认成 4m, conf=0.871 远超 min_tile_conf,
        # 所以它压根不会进 blurry_indices, 而旧代码在算出 bad 后直接 return -
        # 于是 robust_discard 那条路永远走不到, 一张错牌把整局卡死 70 轮.
        bad_idx = out_of_play_indices(tiles, player_count)
        if bad_idx and not robust_discard_enabled:
            # 稳健出牌被关掉时, 含错牌的手牌**不能**去算牌效 -- 那是拿着
            # 已知错误的输入做决策, 算出来的"最优"恰恰是最该避免的一打.
            # 保守停手(测试 test_out_of_play_tile_still_waits_when_robust_disabled
            # 锁的就是这条).
            return TurnDecision(
                ACTION_WAIT, suspect=True,
                reason=f"认出了 {player_count} 人麻将里没有的牌 "
                       f"{[tiles[i] for i in bad_idx]}（槽位 {bad_idx}），"
                       f"且稳健出牌已关闭，先不动")
        # 8.0.5) 听牌摸切守卫(2026-10-09 实锤): 听牌 + 刚摸的牌没认准过 ->
        #      直接摸切. 稳健出牌枚举的是"低置信槽位可能是什么", 但误读
        #        也可能是**高置信**的(g2 t164: 刚摸的 8p 一帧 0.34 一帧
        #        0.95, 高置信帧被当成真的 4p, 牌效按幻影 4p 挑了"打 6m",
        #        真手牌(8p 对)直接从听牌退向). 摸切是唯一位置无关的安全打:
        #        不管刚摸的牌实际是什么, 打出去都不改变听牌状态. 代价是
        #        放弃"打别的牌升级多面听"(那需要可信的读数), 保听优先.
        #      前提: 张数没有空档(missing_slots=0, 否则"听牌"本身就不可信)、
        #      也没有"不可能的牌"(bad_idx 为空, 否则整手读数存疑).
        if (blurry_draw and not missing_slots and not bad_idx
                and concealed % 3 == 2
                and is_tenpai_hand(tiles[:-1], meld_count_of(concealed))):
            return TurnDecision(
                ACTION_DISCARD, tile=tiles[-1], index=concealed - 1,
                reason="听牌且刚摸的牌没认准过，摸切保听"
                       "（摸切不改变听牌，不怕误读）")
        # 立直选牌态在第 3 步就返回了, 走到这里一定不是选牌态(allowed=None).
        assert allowed is None

        # 8a) 手牌里有认不准的牌 -> 稳健出牌.
        #
        # 为什么不是"别点那张低置信的牌": 实测 6 个已证实误识的轮次里, 只有
        # 1 个真的点了低置信那张, 其余 5 个点的是**置信度正常**的牌 -- 错牌
        # 混进手牌后把"哪张该打"的全局判断带偏了(实跑 turn 185: 真手牌 789p
        # 被读成 779p, 牌效于是"正确地"算出打 9p 最优, 打出去直接退向).
        # 所以防护必须做在决策层: 枚举低置信牌的可能取值, 挑一张怎么认都不
        # 会错的. 详见 src/mahjong/brain.py::robust_discard.
        #
        # 稳健模式下**不注入随机性**: 稳健解本身已经是"在多个假设下都不踩坑"
        # 的妥协, 再随机就等于随便挑一个可能踩坑的, 把稳健性丢掉了. 反检测
        # 的随机性在"全部牌都认准"时照常生效, 占比本来就高(实测只有 1~2 张
        # 认不准的轮次才走这条路).
        #
        # suspect = 不参与稳健的槽位有哪些, 用于把原因写进 reason(排查用).
        suspect_idx = sorted(set(blurry_indices or ()) | set(bad_idx))
        if robust_discard_enabled and suspect_idx:
            try:
                robust = robust_discard(tiles, suspect_idx,
                                        meld_count_of(concealed),
                                        player_count=player_count)
            except Exception:  # noqa: BLE001
                # 稳健算法出任何问题都不能连累这一轮出牌, 退回普通逻辑.
                # 不抛也不wait: 这里的异常只可能来自 jansou 内部, 而普通
                # 路径用的是同一个函数, 它要是能算, 下面就一定能算.
                pass
            else:
                # alternatives == 0 = 所有假设都凑不出合法手牌(或最优全指向
                # 低置信槽位自己) -- 读数已经不可信到"怎么枚举都是垃圾",
                # 绝不能把 fallback 的普通最优解当真打出去(21:45 局实锤:
                # 8 张幻觉牌枚举 216 假设 0 种成立, 却打出了一张 5z).
                if robust.alternatives == 0:
                    return TurnDecision(
                        ACTION_WAIT, suspect=True,
                        reason=f"稳健枚举 {robust.cases} 种假设没有一种凑出"
                               f"合法手牌（{len(suspect_idx)} 张没认准），"
                               f"读数不可信，先不动")
                index = pick_tile_index(tiles, robust.tile, rng,
                                        drawn_tile=drawn_tile)
                if index is None:
                    return TurnDecision(
                        ACTION_WAIT, suspect=True,
                        reason=f"稳健选出的 {robust.tile} 不在认出的手牌里")
                how = ("怎么认都是最优" if robust.stable
                       else f"{robust.alternatives} 种可能下最稳的一张")
                extra = ""
                if bad_idx:
                    extra = (f"；其中槽位 {bad_idx} 是"
                             f"{player_count} 人麻将里不存在的牌"
                             f"（{[tiles[i] for i in bad_idx]}），"
                             f"已当误识枚举")
                return TurnDecision(
                    ACTION_DISCARD, tile=robust.tile, index=index,
                    reason=f"稳健({how})，向听{robust.shanten}；"
                           f"有 {len(suspect_idx)} 张牌没认准"
                           f"（槽位 {suspect_idx}），"
                           f"枚举了 {robust.cases} 种可能{extra}{gap_note}")

        # 8b) 出牌: 有 EV 排序就用它(**新目标函数**), 引擎给不出结论就回退牌效.
        #
        # 走到这里手牌已经是"干净"的: 低置信槽位 / 这个人数下不存在的牌要么
        # 在上面被拦下, 要么走了稳健出牌并已返回 —— 读数可以全信, 正好交给
        # 估值引擎. keep_kinds(役对)在这条路上**不单独施加**: 它是给牌效那把
        # 尺子打的补丁(牌效看不见"役"), 而 EV 引擎带着宝牌/役种算期望打点,
        # "这役对该不该留"由它自己回答. 回退路径仍然带着 keep_kinds, 一行没动.
        row = pick_by_ev_rank(tiles, ev_rank)
        if row is not None:
            tile = ev_tile_of(row)
            index = pick_tile_index(tiles, tile, rng,
                                    drawn_tile=drawn_tile)
            if index is not None:
                return TurnDecision(
                    ACTION_DISCARD, tile=tile, index=index,
                    reason=f"EV最优{ev_row_note(row)}{gap_note}")
        try:
            # keep_kinds(2026-10-09 東2局实锤): meld_policy=peng 时手里的
            # 役牌对子是碰的资本(碰出去自带役), 出牌平分时不能优先打掉它 --
            # 实锤手里 1z1z+7p7p+散张, 打 1z/7p/6s 一阶进张完全平分(24),
            # 裁决却选中打 1z, 把役对拆了, 之后的碰机会全部作废.
            plan = choose_discard(tiles, meld_count_of(concealed), rng=rng,
                                  player_count=player_count,
                                  keep_kinds=(yaku_pair_kinds(tiles, missing_slots)
                                              if meld_policy in ("peng", "all")
                                              else frozenset()))
        except Exception as e:  # noqa: BLE001
            return TurnDecision(ACTION_WAIT, suspect=True,
                                reason=f"牌效计算失败({e})")
        index = pick_tile_index(tiles, plan.tile, rng,
                                drawn_tile=drawn_tile)
        if index is None:
            return TurnDecision(ACTION_WAIT, suspect=True,
                                reason=f"要打的 {plan.tile} 不在认出的手牌里")
        return TurnDecision(ACTION_DISCARD, tile=plan.tile, index=index,
                            reason=f"{plan.picked} 向听{plan.shanten} "
                                   f"进张{plan.acceptance}{gap_note}")

    # 9) 等
    if not sane:
        # 两种"不合法"要分开说, 因为上层据此判断是不是该截图留证:
        #   张数压根不在合法集合里-> 认漏了/认错了
        #   张数在集合里但低于下限   -> 手牌区多半是空的, 模型幻觉出的零星几张
        if concealed < MIN_CONCEALED:
            return TurnDecision(
                ACTION_WAIT, suspect=True,
                reason=f"只认出 {concealed} 张牌（低于下限 {MIN_CONCEALED}），"
                       f"多半是手牌区还没发出来或在转场画面，先不动")
        return TurnDecision(ACTION_WAIT, suspect=True,
                            reason=f"手牌 {concealed} 张不是合法张数，可能认漏了")
    if missing_slots > 0:
        # 张数"合法"但**和间距对不上**: 行里有空档, 说明漏认了 missing_slots 张.
        # 这时候 concealed 恰好落进合法集合完全是巧合(10 张 = 1 副露), 用它去
        # 判断"是不是轮到我"就会得出"在等别人" —— 而那是错的, 而且不含 suspect,
        # 上层不会存帧也不会报警. 所以这里必须显式报异常.
        return TurnDecision(
            ACTION_WAIT, suspect=True,
            reason=f"手牌行里有约 {missing_slots} 张牌的空档（间距对不上），"
                   f"只认出 {concealed} 张，张数不可信，不按“在等别人”处理")
    if my_turn is True:
        # 协议**明确说轮到我**了, 而视觉这一帧没有"该我打牌"的张数(13/10/7
        # 而不是 14/11/8). 差别不只是文案: "在等别人"会让上层按正常空转处理,
        # 而真相是"协议已经把我这一动推过来了, 视觉晚了一帧" —— 那种情况下
        # 根本不该等, 应该立刻重看(2026-10-18 13:48 局: 每次摸牌后都撞上这个,
        # 日志写"在等别人", 看上去像是时机判断错了, 其实只是摸的那张还没画完).
        # 不给 suspect: 它是一帧的过渡, 不是感知故障, 不该进留证/卡死统计.
        return TurnDecision(
            ACTION_WAIT,
            reason=f"协议说轮到我打牌，但这一帧视觉只读出 {concealed} 张"
                   f"（该是摸牌后的张数），等下一帧重看")
    return TurnDecision(ACTION_WAIT, reason="在等别人")
