"""全局单例, 存放跨任务共享的状态.

通过 src/config.py 的 `my_app` 注册, 任务里用 `self.og.my_app` 拿到实例.
当前只保留必要状态; 不引入模型加载(识别走 ok-script 的模板匹配 / OCR).
"""


class Globals:
    """跨任务共享状态."""

    def __init__(self, exit_event=None):
        self.exit_event = exit_event
        # 已加载的标注索引: {screen_path: 标注 dict}
        self.annotations: dict = {}
        # 当前场景名(由任务维护, 便于日志与调试)
        self.scene: str | None = None

    def reset(self) -> None:
        self.annotations.clear()
        self.scene = None
