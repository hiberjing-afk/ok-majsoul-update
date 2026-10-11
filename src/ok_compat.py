"""ok-script 2.0.8 的兼容补丁.

**必须在 import ok-script 之前调用 `apply()`**(`src/config.py` 顶部已经在调)。

目前只有一个补丁, 但它不是可选的:

    `ok.feature.FeatureSet` 模块里的文本 `open()` 都没传 encoding, 于是按
    locale 默认编码解码。中文 Windows 上是 cp936(GBK), 而框架自己的标注工具
    `ok.core.template_store.CocoTemplateStore` 写文件时明确用的是
    UTF-8 + ensure_ascii=False —— **读自己写的文件就会炸**:

        UnicodeDecodeError: 'gbk' codec can't decode byte 0x80 in position ...
        (0x80 是「一」的 UTF-8 续字节)

    受影响的路径: `FeatureSet.load_json()` -> `read_from_json()` ->
    `find_one_feature()`, 也就是**任务里每一次模板匹配**。

实测(2026-10-07, 本仓库「标注层收敛为 COCO」之后):

    PYTHONUTF8=1 的 shell 里一切正常(UTF-8 模式下 `open()` 默认就是 UTF-8),
    双击 `run_admin.bat` 却必炸 —— **环境差异把这个问题掩盖了**,
    和当初"卡在步骤 0 两天"是同一个教训: 开发/测试环境和真实启动环境不一致。

所以补丁不能靠在启动脚本里设 `PYTHONUTF8=1` 了事(换个启动方式: IDE、快捷方式、
另一台机器就复现)。这里在代码里堵死, 与启动方式无关。

补丁方式刻意做到最小: 只给**那一个模块的命名空间**塞一个默认 UTF-8 的 `open`,
不改 `builtins`, 不动其它模块; 二进制模式(`'b'`)完全不受影响。

副作用(有意为之): 该模块里另外两处写 COCO 的 `open(..., 'w')` 也因此写 UTF-8,
与 `CocoTemplateStore` 一致 —— 这两个 writer 属于框架的打包/复制辅助函数,
我们并不调用, 但保持一致总是更好。

框架若在后续版本自己修好, 这个补丁在新版上退化成等价行为(UTF-8 读 UTF-8),
**不会破坏任何东西**, 但也就不再需要了。
"""

from __future__ import annotations

import builtins
import logging

logger = logging.getLogger(__name__)

#: 框架标注文件的编码. 与 `ok.core.template_store.CocoTemplateStore` 的写入
#: 编码必须一致, 否则就是上面那个"读自己写的文件炸掉"的 bug。
COCO_ENCODING = "utf-8"

_applied = False


def _utf8_open(file, mode: str = "r", *args, **kwargs):
    """默认 UTF-8 的 open: 只补文本模式缺省的 encoding, 其余原样透传."""
    if "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = COCO_ENCODING
    return builtins.open(file, mode, *args, **kwargs)


def apply() -> bool:
    """打上补丁; 返回是否是**本次**打上的(重复调用无副作用)."""
    global _applied
    if _applied:
        return False

    try:
        import ok.feature.FeatureSet as feature_set
    except Exception as e:  # noqa: BLE001 - 没装框架时不该让 import 崩掉
        logger.warning(f"ok-script 未安装或导入失败, 跳过兼容补丁: {e}")
        return False

    if getattr(feature_set, "_ok_majsoul_utf8_open", False):
        _applied = True
        return False

    feature_set.open = _utf8_open
    feature_set._ok_majsoul_utf8_open = True
    _applied = True
    logger.debug(
        "已给 ok.feature.FeatureSet 打上 UTF-8 open 补丁"
        "(框架的 open 未传 encoding, 中文 Windows 下读不了自己写的 COCO)")
    return True


def is_applied() -> bool:
    return _applied
