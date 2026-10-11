# ok-majsoul

基于图像识别的雀魂（Mahjong Soul）自动对局工具，基于 [ok-script](https://github.com/ok-oldking/ok-wuthering-waves) 框架开发。通过屏幕截图 + YOLO 目标识别读取牌局，用牌效引擎（向听/进张计算）决定出牌，以普通键鼠事件与游戏交互；不读写游戏内存、不修改游戏文件。

> ⚠️ 仅供学习研究使用。使用第三方工具可能违反游戏用户协议，由此产生的一切问题与后果由使用者自行承担。

## 功能

- **段位场连打**（默认主任务）：大厅 → 段位场 → 银之间 → 四人东 → 自动匹配进入对局，打完一场自动回大厅重排队，循环进行。
- **友人场**（测试用）：自动建房（三人/四人，一局/东风），自动开始。
- 自动打完整场：自动和了 / 立直 / 副露（仅碰役对与杠） / 自动拒绝无役鸣牌 / 结算画面自动确认。
- 自动开关点亮（「和」「鸣」等），开局自动就位，发牌闸门防止在过场动画期间误操作。
- 思考节奏拟人化：随机化思考延迟、1% 概率长考，防节奏特征。

## 运行要求

- Windows 10/11
- 任意 DirectX 12 显卡可加速（ONNX DirectML，N/A/I 卡通用；CPU 也能跑但每帧推理明显变慢）
- 雀魂客户端窗口保持前台、完整在屏内、分辨率 1280×720 或更高（推荐 1080p）
- 运行期间不要移动鼠标

## 从源码运行

需要 Python 3.12（必须 3.12，ok-script 依赖约束）。运行时依赖约 300MB（无 torch）。

```bat
git clone <本仓库>
cd majsoul-helper

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

:: 下载模型权重（ONNX 由 Release 包自带；.pt 仅重训/导出时需要）
.venv\Scripts\python tools\fetch_models.py

:: 以管理员身份运行（键鼠注入需要）
run_admin.bat
```

命令行可临时覆盖配置（值按 JSON 解析，`--once` 只影响本次运行）：

```bat
run_admin.bat --once max_games 1 save_frames false
run_admin.bat players "三人" length "一局"
```

常用配置（GUI 下拉框）：`mode`（段位场/友人场）、`players`（三人/四人）、`length`（一局/东风局）、`max_games`、`save_frames`。阈值与节奏类参数在 `src/tasks/tuning.py`。

## 打包 / 更新

见 [docs/RELEASE.md](docs/RELEASE.md)：打包成绿色版 zip 发布到 GitHub Releases，启动器启动时自动检查更新。

## 项目结构

```
├── main.py                    # 生产入口（配置解析 + 启动）
├── run_admin.bat              # 管理员启动器（转发命令行参数）
├── src/
│   ├── detection.py           # 检测结果数据结构（手牌行/按钮/空档语义）
│   ├── detector.py            # YOLO 封装（L1 牌面 / L2 UI 双权重）
│   ├── vision.py              # 组合识别：手牌行裁切、投票、稳健模式
│   ├── mahjong/
│   │   ├── table.py           # 对局状态机：decide_turn 决策链、闸门、守卫
│   │   ├── brain.py           # 牌效：向听/进张排序、稳健出牌、平分裁决
│   │   ├── timing.py          # 思考延迟分布（拟人节奏）
│   │   └── tiles.py           # 牌种定义
│   ├── tasks/
│   │   ├── PlayOneGameTask.py # 主任务：连打循环、自动开关、点击执行
│   │   ├── CreateFriendsRoomTask.py  # 建房流程（两种房型共用）
│   │   └── tuning.py          # 全部阈值/延迟常量
│   ├── rooms.py               # 房型 → 屏幕特征/操作流程（RoomSpec）
│   ├── ok_compat.py           # ok-script 兼容层（编码等）
│   └── imageio.py             # 图像读写（Windows 非 ASCII 路径）
├── ok_templates/              # 模板真源（coco_annotations.json + 截图），运行时加载
├── models/                    # 权重（fetch_models.py 下载）
├── tools/                     # 维护与训练工具（标注/重训/自检/清理）
├── tests/                     # 单元测试（unittest）
└── docs/                      # 架构文档、策略说明、发布流程
```

## 文档

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 架构与设计决策（识别体系、决策链、参数、红线）
- [docs/STRATEGY.md](docs/STRATEGY.md) — 出牌与鸣牌策略说明
- [docs/ANTI_DETECTION.md](docs/ANTI_DETECTION.md) — 拟人化设计说明
- [docs/RELEASE.md](docs/RELEASE.md) — 打包与发布流程

## 开发

```bat
:: 全部单元测试
.venv\Scripts\python -m unittest discover -s tests -p "Test*.py"

:: GUI 启动自检（改配置/标注后必跑）
.venv\Scripts\python tools\check_gui_startup.py

:: 清理运行时产物（默认 dry-run）
.venv\Scripts\python tools\cleanup.py --apply
```

## 声明

本项目为开源免费工具，仅供学习交流。通过现有 UI 界面与普通键鼠事件与游戏交互，不修改游戏文件与代码。使用第三方工具可能违反游戏用户协议，由此产生的一切问题与后果由使用者自行承担。请遵守相关法律法规与游戏公平性条款。
