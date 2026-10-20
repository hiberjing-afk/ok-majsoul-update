"""出牌估值: 从协议事实算 EV 候选(纯函数, 可单测).

输入全部来自协议(手牌、牌河、副露、宝牌指示牌、场风/自风、牌山), 不掺视觉
读数. 一次计算同时供引擎决策与记录使用; 任何异常都返回 `(None, None)`, 估值
永不拖垮对局.
"""

from __future__ import annotations

from src.mahjong.ev_eval import evaluate_discards, evaluate_efficiency
from src.tasks import tuning


def _plan_ev(net, cfg):
    """跑一次出牌 EV 估值: 返回 `(候选, 牌山)`, 拿不到就是 `(None, None)`."""
    if net is None:
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
        # 只在摸完牌该我打的张数档(14/11/8/5/2)算, 别的档引擎也拒收.
        if len(hand) % 3 != 2:
            return None, None
        if tuning.EV_EVAL_POINTS:
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
