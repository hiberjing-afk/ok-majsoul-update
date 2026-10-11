"""ok-script 配置: 窗口/截图/识别/任务注册.

参考 ok-nte 的组织方式: 任务统一注册在 `onetime_tasks` 里,
不要在仓库里依赖 `ok_tasks/` 目录扫描(那个目录留给用户自定义任务, 且不进版本库).
"""

import os

from src import ok_compat

# 必须赶在 import ok-script 之前: 框架读标注文件时不传 encoding, 中文 Windows
# 下按 cp936 解码它自己写的 UTF-8 COCO, 会抛 UnicodeDecodeError。详见模块说明。
ok_compat.apply()

from src.interactions import HumanLikePynputInteraction  # noqa: E402

version = "dev"

gui_config = {
    "type": "qt",
    "window_size": {
        "width": 1200,
        "height": 800,
        "min_width": 600,
        "min_height": 450,
    },
}

config = {
    # 是否允许用户在 ok_tasks/ 目录里放自定义任务(该目录不进版本库)
    "custom_tasks": True,
    "debug": False,
    "gui": gui_config,
    "config_folder": "configs",
    "global_configs": [],
    "screenshot_processor": None,
    "gui_icon": "icons/icon.png",
    "wait_until_before_delay": 0,
    "wait_until_check_delay": 0,
    "wait_until_settle_time": 0,
    "ocr": {
        "lib": "onnxocr",
        "auto_simplify": True,
        "params": {
            "use_openvino": True,
        },
    },
    "windows": {
        "exe": ["Jantama_MahjongSoul.exe"],
        "start_exe": False,
        # 雀魂是 Unity(uGUI) 游戏: 点击命中用的是真实系统光标位置,
        # 不消费 WM_LBUTTONDOWN 的 lParam 坐标 -> PostMessage 系全部无效
        # (实测: 裸 PostMessage / WM_ACTIVATE+PostMessage 点击都无反应).
        # 所以必须走真实鼠标事件, 且窗口要在前台(Pynput 的 clickable 要求).
        #
        # HumanLikePynputInteraction 是 Pynput 的子类, 只是把"瞬移到目标再按下"
        # 换成贝塞尔轨迹滑过去(见 src/interactions.py / src/mouse_path.py).
        #
        # 注意列表里**故意不再放** "Pynput": DeviceManager 会用 configs/devices.json
        # 里存的名字回查这个列表, 命中就用存的那个. 把 "Pynput" 留在列表里,
        # 老配置里残留的 "Pynput" 会继续命中, 于是悄悄退回瞬移点击; 去掉之后
        # 匹配不上就回落到列表首项(也就是拟人版), 升级不会静默失效.
        "interaction": [HumanLikePynputInteraction, "PyDirect", "PostMessage"],
        "capture_method": ["WGC", "BitBlt_RenderFull", "BitBlt"],
        "check_hdr": False,
        "force_no_hdr": False,
    },
    "start_timeout": 120,
    # 雀魂在 WGC 截图下 client area == 渲染分辨率.
    #
    # 重要: 这里**不要**再用 min_size 把分辨率锁死. 这台机器上雀魂窗口处在 150% 的
    # DPI 虚拟化上下文(GetDpiForWindow=144)而显示器是 100%(GetDpiForMonitor=96),
    # 于是同一个窗口在不同 DPI 上下文里量出来可以是 1920x1080 也可以是 1280x720,
    # 锁死 min_size 会让脚本在其中一个上下文里直接拒绝启动.
    #
    # 正确做法是让识别端做换算(见 src.annotations.scale_factor): 标注按制作时的
    # 分辨率存 image_size, 运行时按帧尺寸缩放区域坐标和模板, 任何分辨率都能跑.
    # min_size 只保留"低于 720p 就没法看"这个兜底下限, resize_to 留空 = 不自动拉伸窗口.
    "supported_resolution": {
        "ratio": "16:9",
        "min_size": (1280, 720),
        "resize_to": [],
    },
    "links": {
        "default": {
            "github": "https://github.com/yourname/ok-majsoul",
        },
    },
    "screenshots_folder": "screenshots",
    "gui_title": "ok-majsoul 雀魂助手",
    "template_matching": {
        # COCO 标注 == 唯一真源, 由 ok-script 自带「模板」页签产出
        # (截图 + 画框, 见 ok/ui/qt/tasks/TemplateTab.py 与 MarkUpWindow.py)。
        # 我们不再有自研标注格式, 也没有"导出成 COCO"这一层。
        #
        # 页签要求 `custom_tasks` 为真才会出现在界面上(见 MainWindow 里
        # `if og.task_manager.has_custom or debug`), 所以下面那个开关别关。
        "coco_feature_json": "ok_templates/coco_annotations.json",
        # 搜索区外扩倍数(相对标注框): 框架自带的"按 variance 外扩"只有几个
        # 像素(见 FeatureSet.find_one_feature), 对我们的场景太窄, 所以由
        # src/annotations.frame_box 自己算好搜索区再传给 find_one。
        "default_horizontal_variance": 0.002,
        "default_vertical_variance": 0.002,
        "default_threshold": 0.8,
    },
    "version": version,
    # 全局单例: 任务里用 self.og.my_app 访问
    "my_app": ["src.globals", "Globals"],
    # 项目自带任务(用户点击触发)
    # "创建友人场房间"卡片已下架(2026-10-08 用户拍板: 只用打牌任务, 单独开房
    # 没用了). **类不能删** —— PlayOneGameTask 继承它复用整套开房/匹配/点击
    # 流程, 这里只是不再把它注册成独立任务卡片.
    "onetime_tasks": [
        ["src.tasks.PlayOneGameTask", "PlayOneGameTask"],
    ],
}
