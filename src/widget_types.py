"""ok-script 界面控件能渲染哪些配置值类型, 以及把不能渲染的标成 hidden.

**为什么单独抽一个模块**: 这条规则是"程序能不能开起来"的生死线
(见 tools/check_gui_startup.py 的说明), 必须能被单元测试直接覆盖,
而测试规范禁止 import ok-script(会在仓库里写 logs/ok-script.log).
所以这里只放纯逻辑, 不 import 框架任何东西.

**规则本身**(2026-10-07 逐行核对
`.venv/Lib/site-packages/ok/ui/qt/tasks/ConfigItemFactory.py:52-71`):

    if   isinstance(value, bool):  SwitchButton
    elif isinstance(value, list):  ModifyListItem
    elif isinstance(value, int):   SpinBox
    elif isinstance(value, float): DoubleSpinBox
    elif isinstance(value, str):   TextEdit / LineEdit(按长度自动选)
    else: raise ValueError(f"invalid type {type(value)}, value {value}")

也就是说 **tuple / dict / None 全部会 raise**. 而这个异常发生在
TaskCard 构造链里(MainWindow -> OneTimeTaskTab.refresh_ui -> TaskCard
-> ConfigCard.__addConfig -> config_widget), 一路冒泡到
`ok.OK.start()` 的 show_main_window, 结果是**整个程序打不开**,
日志里只有一行 "ValueError: invalid type <class 'tuple'>".

**为什么以前一直没炸**: 这些 tuple 型配置项一直是"任务内部参数",
从没在界面上渲染过. 一旦为了让下拉框出现而把 `default_config` 填上,
框架就开始渲染**每一个**配置项, 雷就炸了. 见 MEMORY.md
「ok-script 的下拉框机制」一节.

**所以界面该长什么样, 由 default_config 决定, 不由这个模块决定**:
ok-script 没有"隐藏但保留配置"的机制(每个配置项都会生成控件, `hidden`
只给派生字段设计). 想让界面干净, 唯一的办法是把内部调参从
`default_config` 里**拿出去**成模块常量(见 `src/tasks/tuning.py`,
这是 ok-nte 的做法). 这个模块只做兜底: 万一以后又有人往配置里塞了
tuple, 至少程序还能开起来, 而不是开不了机.
"""

from __future__ import annotations

#: 框架能渲染成控件的值类型. 注意 isinstance 的语义: bool 是 int 的子类,
#: 但框架把 bool 分支写在 int 之前, 所以顺序是对的, 这里照抄即可.
WIDGET_SAFE_TYPES: tuple[type, ...] = (bool, int, float, str, list)


def is_widget_safe(value: object) -> bool:
    """这个值能不能直接交给框架渲染成控件."""
    return isinstance(value, WIDGET_SAFE_TYPES)


def mark_unrenderable_hidden(config_type: dict, defaults: dict) -> list[str]:
    """把 ``defaults`` 里框架渲染不了的键在 ``config_type`` 上标成 hidden.

    已经在 ``config_type`` 里显式声明过的键**不动** -- 那些声明
    (下拉框等)是任务自己写的, 值得信任, 不要覆盖掉.

    返回被新标 hidden 的键名(按 ``defaults`` 的顺序), 便于调用方打日志
    以及测试断言.
    """
    marked: list[str] = []
    for key, value in defaults.items():
        if key.startswith("_") or key in config_type:
            continue
        if not is_widget_safe(value):
            config_type[key] = {"hidden": True}
            marked.append(key)
    return marked