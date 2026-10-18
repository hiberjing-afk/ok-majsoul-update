"""协议读数 -> 决策输入的胶水层(从 PlayOneGameTask 搬出来的模块级函数).

**这一层只干一件事**: 把 `NetAssist` / `GameState` 的读数翻译成决策层
(`table.decide_turn`) 与执行层认得的几个值, 并统一实现"协议不可用就返回
None = 不参战"这条纪律. 它不碰截屏、不碰鼠标、不碰任务状态机, 所以可以脱离
任务单独测.

**为什么是模块级函数而不是方法**: `_execute` / `_confirm_stable` 在单测里会被
挂在假任务对象(SimpleNamespace)上跑, 那些对象只挂自己真正需要的几个方法 ——
为一个"提示"属性把整条执行路径炸掉不值得(与"网络层永不拖垮主循环"同一条
原则).

返回值的约定(全层一致):
    None      协议不可用 / 桥不新鲜 / 算不动 -> 调用方按"没有协议"处理
    True/False 协议明确给出了答案
任何异常都在这里吃掉并退化成 None, 绝不向主循环抛.
"""

from __future__ import annotations

from src.mahjong.chi_panel import AKA_RED_RATIO, aka_ratio
from src.mahjong.ev_eval import evaluate_discards, evaluate_efficiency, has_win_signal
from src.mahjong.table import ACTION_DISCARD, ACTION_MELD
from src.tasks import tuning


def _urgent_now(obj) -> bool:
    """这一轮要不要压缩节拍(协议倒计时临近, 见 tuning.NET_TURN_CLOCK).

    写成模块级函数而不是方法: `_execute` / `_confirm_stable` 在单测里会被挂在
    假任务对象(SimpleNamespace)上跑, 那些对象只挂自己真正需要的几个方法 ——
    为一个"提示"属性把整条执行路径炸掉不值得, 与"网络层永不拖垮主循环"同一条
    原则. 真任务上这个字段由 _play_one_game 每轮写.
    """
    return bool(getattr(obj, "_urgent", False))


def _proto_clock_read(net, mode) -> bool | None:
    """按协议读"这一刻有没有我的出牌操作"; 拿不到就是 None(不参战).

    做成模块级函数而不是方法, 理由与其余包装一致: 单测把 `_proto_turn` 绑到
    SimpleNamespace 假对象上跑, 除了 `_net`/`_cfg` 什么都没有.

    **必须先看 offered**: 协议的操作列表为空只说明"服务端还没给我选项"(刚摸牌 /
    过场 / 上一窗口已作废), 那时 turn_clock 里的 my_turn=False 是没有信息的保守
    填充. 拿它去否决出牌就是"拿沉默当反对" —— 一次误否就把整手牌拖进托管.
    只有"给我的列表非空、里面确实没有打牌(type 1)"才算协议说了"不是你".
    """
    if net is None or not tuning.NET_TURN_CLOCK:
        return None
    if not mode or mode == tuning.NET_READ_OFF:
        return None
    try:
        if not net.available():
            return None
        clock = net.turn_clock()
        if not clock.get("offered"):
            return None
        return bool(clock.get("my_turn"))
    except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
        return None


def _proto_meld_window_of(obj) -> bool | None:
    """协议说"此刻开着给我的鸣牌窗口"(吃/碰/大明杠), 拿不到就是 None.

    与 `_proto_turn_of` 配成一对: 那个说"有没有我的**出牌**操作", 这个说
    "有没有我的**鸣牌**操作". 两件事在鸣牌窗口里**必然相反**(协议只给吃碰杠,
    不给打牌), 所以拿前者去否决整轮就会把每一个鸣牌机会都拦死 —— 2026-10-17
    真机 23 次想碰一嘴没点出去, 就是这个原因.

    看 `meld_options()` 而不是自己翻 offered: 那是协议 → 鸣牌语义的唯一翻译
    点(吃/碰/大明杠的 type 映射与组合解析都在那里), 两处各写一份迟早漂移.
    对假对象/异常一律 None(不参战), 与其余包装同一条纪律.
    """
    fn = getattr(obj, "_proto_meld_window", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
        return None


def _proto_turn_of(obj) -> bool | None:
    """协议对"轮到我"的权威说法, 拿不到就是 None(不参战, 见 _proto_turn).

    模块级包装的理由和 _urgent_now 一样: `_confirm_stable` 在单测里被绑到假
    任务对象上跑, 假对象没这个方法 —— 那种情况下"不参战"是唯一正确语义,
    绝不该抛(与"网络层永不拖垮主循环"同一条原则).
    """
    fn = getattr(obj, "_proto_turn", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _proto_clock_turn_of(obj) -> bool | None:
    """协议对"这一刻有没有我的出牌操作"的窄口径说法(见 _proto_clock_turn).

    与 `_proto_turn_of` 分开: 这个看档位、那个不看. 同样对假对象返回 None.
    """
    fn = getattr(obj, "_proto_clock_turn", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _proto_round_seq_of(obj) -> int | None:
    """协议开过几手了(单调递增, 见 assist.NetAssist.round_seq).

    模块级包装的理由与其余几个一致: 单测把主循环绑到只有少数几个方法的假
    对象上跑, 那种情况下"不参战"是唯一正确语义.
    """
    fn = getattr(obj, "_proto_round_seq", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _draw_settle_pending(net) -> bool:
    """协议刚给自己发了牌、渐入动画还没落定 -> 这一轮先不睁眼(见
    tuning.NET_DRAW_SETTLE).

    依据(2026-10-17, 用户逐帧判定): 4 张 lowconf 帧全是**自己摸牌的渐入
    动画**与开局整理动画 —— 摸到那张落定前读手牌, 低置信、协议对账
    对不上(少了那张)、存帧全是动画噪声. 协议知道发牌时刻, 等这零点几秒
    再读, 读到的就是落定画面. 开局直接轮到自己时系统多给 2s, 等得起;
    中局拟人延时 0.6~1.3s, 这点等待也在预算内.

    拿不到协议(没抓包/状态冻着/异常)一律 False = 老行为, 多看一眼永远
    安全. 唯一例外: 倒计时已临期时不等 —— 托管比一张低置信帧贵得多.
    做成模块级函数与 _proto_clock_read 同一个理由: 单测绑假对象跑.
    """
    if net is None or tuning.NET_DRAW_SETTLE <= 0:
        return False
    try:
        secs = net.seconds_since_self_draw()
    except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
        return False
    if secs is None or secs >= tuning.NET_DRAW_SETTLE:
        return False
    try:
        left = net.turn_clock().get("left")
    except Exception:  # noqa: BLE001
        left = None
    if isinstance(left, (int, float)) and left <= tuning.NET_URGENT_SECONDS:
        return False
    return True


def _prefer_non_aka_box_impl(net, grab, tile: str, hand, box):
    """同牌种里有赤宝牌在场时, 出牌**避开画着红牌的那个槽**.

    决策(EV/牌效)按**牌种**算, 赤 5 和普通 5 牌效完全一样; 但留哪张打哪张
    有讲究: 赤宝牌和了多一番, 该**留赤打普通**. 视觉模型没有赤类(34 类全部
    折叠, 见 src/detector.py 模块说明), 槽位层面谁是赤只有协议知道 ——
    协议手牌里同时有 "0x" 与 "x" 时, 用红色像素把普通的那张挑出来点.
    读不到协议 / 帧拿不到 / 候选不足两张: 原样返回(不猜).
    """
    if tile not in ("5m", "5p", "5s"):
        return box
    try:
        view = net.ev_inputs()
        proto_hand = list((view or {}).get("hand") or [])
    except Exception:  # noqa: BLE001 - 网络层永不拖垮主循环
        return box
    aka = "0" + tile[-1]
    if aka not in proto_hand or tile not in proto_hand:
        return box
    candidates = [i for i, t in enumerate(hand.tiles) if t == tile]
    if len(candidates) < 2:
        return box
    frame = grab()
    if frame is None:
        return box
    for i in candidates:
        b = hand.box_of(i)
        if b is not None and aka_ratio(frame, b) < AKA_RED_RATIO:
            return b
    return box


def _proto_round_open_of(obj) -> bool | None:
    """协议说"这一手还在打吗"(见 assist.NetAssist.round_open).

    **不要求 available()**: 收局后 available() 已经 False, 其余协议闸门全部
    哑火 —— round_open=False 恰恰是那段画面上唯一还活着的协议信号, 拿它拦
    "结算画面残留手牌被读成稳定出牌"(2026-10-17 真机: 别家胡牌后 12s,
    我们在结算画面上"稳健打出"了一张 9p). 对假对象返回 None(不参战).
    """
    fn = getattr(obj, "_proto_round_open", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _hand_conflict_veto(trust: str, decision) -> bool:
    """协议张数与视觉张数互相矛盾时, 本轮要不要**什么都不做**.

    用户拍板的规矩是"对不上就以协议为准, 同时保留截图"(截图由调用方的
    `_save_proto_mismatch` 存, 与这里无关). 张数矛盾这一档能怎么"以协议为准":

    - 协议给的是**牌是什么 + 几张**, 视觉给的才是**哪张牌在哪个槽**(点击坐标).
      张数都不一样时两边根本对不齐: 照协议那 14 张算得出牌效, 却点不下去;
      照视觉那 11 张点得下去, 但那手牌已经确定是假的.
    - 2026-10-18 13:19 局 turn 3 就是后者: 视觉 11 张、协议 14 张, 程序照 11 张
      算出"打 5m"点了出去 —— 后面拆对子/拆顺子/无役振听一路从这里来.

    所以这一档的正确表达是"两手都不信, 本轮不动", 而不是硬按哪一边打.
    只否决**打出牌/鸣牌**这类"会改变局面的动作"; 跳过/等待本来就没事可做.

    **没有超时放行(2026-10-18 20:21 局用户拍板, 删掉 NET_CONFLICT_MAX_WAIT)**:
    老设计"超 2s 退回按视觉继续打(宁可漏判不可锁死)"在退回去的那一刻, 视觉
    可能还是假牌或动画中的框 —— 按它打就是 18:56 局"点 1z 打出 9s"的复制,
    而且日志明晃晃写着"按视觉手牌继续打", 正是用户反复反对的模式. 视觉坏了
    就等: 主循环每 0.3s 重读一次, 恢复的第一时间就能按协议正确出牌(实测开局
    4 张假帧 6s 内恢复); 真等成托管, 也比主动点错牌强 —— 托管至少不点动画
    中的框.
    """
    if trust != "conflict":
        return False
    return getattr(decision, "action", None) in (ACTION_DISCARD, ACTION_MELD)


def _trust_skips_confirm(trust: str, action: str, declare_turn: bool,
                         pending_riichi: bool, proto_meld: bool = False) -> bool:
    """信任档能否跳过动作前双帧确认直接执行(纯逻辑, 可单测/注入).

    - **普通出牌**: 手牌标签已是协议真值, 协议值不会因第二帧变化, 重算唯一
      能产出的就是"低置信槽位两帧跳类"引发的"要打的牌变了"空转(真机
      9m<->7z 折腾 3 轮).
    - **鸣牌窗口里的按钮动作**(`proto_meld=True`, 协议说此刻开着给我的吃碰杠
      窗口): 动作**种类**是服务端给的 —— 面板只会画它允许的那几种, 视觉在这
      里只负责**坐标**。再花 0.4s 重算一遍决策, 等于让视觉用第二帧重新回答
      "该不该碰": 那一帧按钮漏检(2026-10-18 14:58 局, 连「跳过」都一起漏)、
      或者第 11 槽读数抖一下没有役牌对子, 就判成"动作变了（meld -> skip）",
      于是**每一轮都卡在同一堵墙上**, 整整 24 秒不出手, 最后靠协议倒计时
      临期(`_urgent_now`)才逃出去 —— 用户在桌上看到的就是"有役牌能碰、
      一直等着不碰".
    - **立直宣言 / 立直按钮 / 拔北**: 一律不跳. 立直不可撤销, 按钮仍要靠双帧
      过滤单帧误检(东三局 lizhi 0.714); 加杠/暗杠点错无收益, 不值得为它开门.
    """
    if not tuning.NET_TRUST_PROTO or trust != "trusted":
        return False
    if action == ACTION_DISCARD:
        return not (declare_turn or pending_riichi)
    if action == ACTION_MELD:
        return bool(proto_meld) and not (declare_turn or pending_riichi)
    return False


def _plan_ev(net, cfg, *, urgent: bool = False):
    """跑一次出牌 EV 估值: 返回 `(候选, 牌山)`, 拿不到就是 `(None, None)`.

    输入**全部来自协议**(手牌精确到张、牌河、副露、宝牌指示牌、场风/自风、
    牌山), 不掺视觉读数. 只在"我该打牌"的张数档(暗牌 %3==2)算 —— 别的时候
    `discard_outlooks` 本来就拒收.

    **一次计算供两处用**: 出牌记录(影子)与出牌决策(接管, 见
    tuning.EV_EVAL_TAKEOVER)—— 接管之后影子那份就是决策那份, 不重复跑.

    做成模块级函数而不是方法, 理由与 `_proto_clock_read` 一样: 单测把
    `_ev_shadow` 绑到 SimpleNamespace 假对象上跑, 那些对象只有 `_net`/`_cfg`.
    任何异常都返回 `(None, None)` —— 估值永不拖垮主循环.
    """
    if net is None or urgent:
        # 倒计时危急时不跑: 最坏一次估价几百毫秒, 而这一轮剩下的时间本来
        # 就不够用了 —— 那一轮直接回退牌效, 先把牌打出去.
        return None, None
    mode = (cfg or {}).get("net_read") or tuning.NET_READ
    if mode == tuning.NET_READ_OFF:
        return None, None
    try:
        if not net.available():
            return None, None
        view = net.ev_inputs()
        if not view:
            return None, None
        hand = view.get("hand") or []
        # 只在"摸完牌该我打"的张数档(14/11/8/5/2)算. 引擎自己也会拒收别的档
        # (它校验得更严: 只认 sum 恰好在那几个值上), 这里先拦一道是为了不做
        # 无用功 —— 一轮里 2/3 的轮询都不是我的回合.
        if len(hand) % 3 != 2:
            return None, None
        # 两档共用同一批协议事实, 产出同一个类型(见各自的 docstring):
        #   EV_EVAL_POINTS —— 带 EV 展望与打点, 按"期望打点"排序(30~几百 ms);
        #   关掉           —— 纯牌效, 按"向听 + 真实进张"排序(0.09 ms).
        # 关掉那档每轮都算得起, 不必再为倒计时让路.
        if bool(tuning.EV_EVAL_POINTS):
            rows = evaluate_discards(
                hand,
                melds=view.get("my_melds") or (),
                visible=view.get("visible"),
                draws=int(view.get("draws") or 0),
                player_count=int(view.get("player_count") or 3),
                doras=view.get("doras") or (),
                round_wind=view.get("round_wind") or "1z",
                seat_wind=view.get("seat_wind"),
                my_seat=int(view.get("my_seat") or 0),
                with_points=True,
            )
        else:
            rows = evaluate_efficiency(
                hand,
                melds=view.get("my_melds") or (),
                visible=view.get("visible"),
                player_count=int(view.get("player_count") or 3),
                my_seat=int(view.get("my_seat") or 0),
            )
    except Exception:  # noqa: BLE001 - 估值永不拖垮主循环
        return None, None
    return (list(rows) if rows else None), view.get("wall")


def _ev_record_of(rows, wall) -> dict | None:
    """`_plan_ev` 的候选 -> jsonl 的一行(前几名 + 牌山 + 信号标志).

    记的是**前几名**而不只是第一名: 事后要对的是"我们实际打的那张排第几、
    比第一名少多少期望", 只看第一名答不了这个. 三张够覆盖"最优 + 两个近似"
    这种常见局面, 全量进 jsonl 会把每行撑大好几倍.
    """
    if not rows:
        return None
    top = rows[0]
    record = top.as_record()
    record["summary"] = top.summary()
    record["wall"] = wall
    #: 引擎这手牌**有没有给出可用信号**. `win` 全零时排序已自动退回牌效
    #: (见 ev_eval.evaluate_discards), 但那一列"最优出牌"就不再是估值结论了
    #: —— 事后复盘必须能区分这两种情况, 否则会把兜底当成建议.
    record["signal"] = has_win_signal(rows)
    record["candidates"] = [row.as_record() for row in rows[:3]]
    return record
