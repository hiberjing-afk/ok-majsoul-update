"""操作耗时的随机分布(纯逻辑, 可单测).

为什么单独一个文件: 出牌速度是最容易被服务端统计出来的特征. 真人打牌的
间隔是一个有肥尾的分布 -- 大部分 1~4 秒, 偶尔长考五六秒, 偶尔秒打.
如果写成"固定 2 秒 ± 0.1", 标准差接近 0, 一眼就是脚本.

三条硬约束(都写成了单测):

- 均值落在真人区间;
- 标准差 > 0.8 秒(docs/ANTI_DETECTION.md 里给的判定线是 < 0.3s 必判脚本);
- 有小概率的长尾(长考), 不能永远挤在均值附近.

**为什么均值一降再降**(2026-10-06 4.0→2.4, 2026-10-07 2.4→1.8,
2026-10-08 1.8→1.3→0.9, 2026-10-09 0.9→0.6): 第一版出牌均值 4.0s + 5% 概率
长考 8~15s, 单看每个动作都很像真人, 但一局三人场只有 ~17 次决策, 串起来
就成了"每一步都很慎重"的机器人节奏; 叠加轮询/抓帧/复核的固定开销后,
一次出牌平均要 8~10 秒, 容易撞上雀魂自己的回合计时(超时会被托管自动打牌,
行为特征更明显). 2026-10-07 四人场实跑用户再报"思考过长": 雀魂每回合只有
5s 基础时间, 超出就吃全局 bank. 2026-10-08 晚双帧确认闸门上线后用户再报
"经常超时, 还被托管吃牌": 实测决策->点击中位 2.86s / 均值 4.28s, 于是
降到 N(0.9, 1.1) + 1% 长考 2.5~3.5s, 点击前总耗时中位 ~2.4s. 2026-10-09
晚用户实跑仍嫌久, 再降到 N(0.6, 1.3)/min0.2, 总耗时中位 ~2.1s.

分布用 random 而不是 numpy: 这只是几个标量抽样, 不值得为一个分布函数
拖进整个 numpy, 而且 random.Random 可以注入种子, 测试才能确定.
"""

from __future__ import annotations

import random

# 出牌: N(0.4, 1.3), 下限 0.2 秒, 1% 概率长考 2.5~3.5 秒
# (2026-10-09 降到 N(0.6,1.3)/min0.2, 总耗时中位 ~2.1s. 2026-10-17 深夜
# 用户再拍板"摸牌等 0.3s 够了, 模拟思考的时间也相应降低一点" -> mu
# 0.6 -> 0.4; sigma 保持 1.3 —— floor 0.2 截掉的左尾更多了, 再动 sigma
# 会跌破 TestTiming 的 std>0.8 反脚本线, 实测 std=0.83 仍在线上,
# P(>2.5s)≈5%, 最差尾部仍 <5s 不撞雀魂 5s 基础时间.)
DISCARD_MU = 0.4
DISCARD_SIGMA = 1.3
DISCARD_MIN = 0.2
LONG_THINK_RATE = 0.01
LONG_THINK_RANGE = (2.5, 3.5)

# 其它操作, 都比出牌快一些(同样已扣双帧确认的固定开销).
# meld 是应答窗口, 慢了整个窗口都被托管接管, 压到 ~0.5s.
MELD_MU, MELD_SIGMA, MELD_MIN = 0.5, 0.4, 0.3
RIICHI_MU, RIICHI_SIGMA, RIICHI_MIN = 1.0, 0.6, 0.5
WIN_MU, WIN_SIGMA, WIN_MIN = 0.4, 0.3, 0.25
IDLE_MU, IDLE_SIGMA, IDLE_MIN = 0.3, 0.25, 0.2


def _normal(rng: random.Random, mu: float, sigma: float, floor: float) -> float:
    return max(floor, rng.gauss(mu, sigma))


def discard_delay(rng: random.Random | None = None) -> float:
    """出牌前想多久."""
    rng = rng if rng is not None else random
    if rng.random() < LONG_THINK_RATE:
        return rng.uniform(*LONG_THINK_RANGE)
    return _normal(rng, DISCARD_MU, DISCARD_SIGMA, DISCARD_MIN)


def meld_delay(rng: random.Random | None = None) -> float:
    """碰 / 吃 / 杠 前想多久."""
    return _normal(rng or random, MELD_MU, MELD_SIGMA, MELD_MIN)


def riichi_delay(rng: random.Random | None = None) -> float:
    """宣布立直前想多久."""
    return _normal(rng or random, RIICHI_MU, RIICHI_SIGMA, RIICHI_MIN)


def win_delay(rng: random.Random | None = None) -> float:
    """和牌 / 自摸前想多久."""
    return _normal(rng or random, WIN_MU, WIN_SIGMA, WIN_MIN)


def idle_delay(rng: random.Random | None = None) -> float:
    """放弃这次机会(不碰不吃)前想多久."""
    return _normal(rng or random, IDLE_MU, IDLE_SIGMA, IDLE_MIN)
