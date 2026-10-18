"""生产入口: 加载配置并启动 ok-script 应用.

支持用命令行覆盖任务配置, 方便不改代码就临时调开关(通过 run_admin.bat
也能传, 它会把参数原样转发过来):

    run_admin.bat --once save_frames false
    run_admin.bat save_frames false max_games 3

**参数一律用空格分隔, 不用 `--set key=value`**:
实测 cmd 的单个参数展开 `%1` / `%~1` / `for %%A` 在 token 含 `=` 时会在
`=` 处截断(`--set=save_frames=false` 变成 `--set`), shift 也救不了. `%*`
虽然完整但没法逐个判断开关, 一旦为了判断去遍历就踩同一个坑. 空格分隔形式
实测处处正确.

    --oncekey value    只影响本次运行(不写盘)
    key value          持久化写进 configs/<任务名>.json

`--once` 之后的键值对都不写盘; 不带 `--once` 的会持久化.
值按 JSON 解析(能写 true/false/数字/列表), 不是合法 JSON 时按字符串处理.
不认识的键会打警告, 避免拼错键名却以为生效了.
"""

import json
import sys
from pathlib import Path

# 内嵌运行时(python312._pth 隔离模式)不会把脚本所在目录加进 sys.path,
# `from src.config import config` 会 ModuleNotFoundError. 这里自愈:
# 无论哪种启动方式(bat / 安装器快捷方式 / 直接 python main.py)都保证仓库根在 path 上.
# 与 tools/check_gui_startup.py:39 同一处理(2026-10-11 v1.2.0 实测踩坑).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 允许覆盖的键(取自 PlayOneGameTask._default_config).
# 故意显式列出: 拼错键名要立刻警告, 而不是静默写进去以为生效了.
#
# 房型现在是界面上两个下拉框: players(三人/四人) + length(一局/东风局),
# 组合由 src/rooms.py 的组合表解析; room 是**派生字段**, 一般不用手填,
# 但保留在白名单里以便临时强制指定(留空 = 由 players+length 推导).
ALLOWED_KEYS = {
    # 只列**用户真正常用的**配置项(2026-10-07 用户决定), 与
    # src/tasks/PlayOneGameTask.py 的 _DEFAULT_CONFIG 一一对应.
    # 其余二十多项阈值/延迟/比例已经搬去 src/tasks/tuning.py 做模块常量,
    # 它们不再是配置项, 所以这里也不接受 --set: 给了也无处可去,
    # 静默写进 configs/*.json 只会造成"改了却没生效"的错觉.
    # 要调那些参数请直接改 tuning.py 里的常量.
    "PlayOneGameTask": {
        "players", "length", "room",       # 房型(前两个是界面下拉框)
        "max_games", "save_frames", "robust_discard",
    },
}

TARGET_TASK = "PlayOneGameTask"


def _parse_value(text: str):
    """按 JSON 解析; 失败则当普通字符串."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_args(argv: list[str]) -> tuple[bool, list[tuple[str, object]]]:
    """拆出 (是否只影响本次运行, [(键, 值), ...]).

    两种写法混用:
        --once save_frames false      -> once=True, save_frames=False
        save_frames false             -> once=False, save_frames=False
    `--once` 之后的键值对不写盘; 不带 `--once` 的持久化.
    """
    once = False
    pairs: list[tuple[str, object]] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--once":
            once = True
            i += 1
            continue
        if arg.startswith("--once="):
            # 兼容 --once=key value: 开关与键值对连写
            once = True
            pairs.append((arg[len("--once="):].strip(), ""))
            i += 1
            continue
        if arg == "--":
            once = True
            i += 1
            continue
        key, value = arg, ""
        if "=" in arg and not arg.startswith("-"):
            # 兼容 key=value 形式(直接调python 时方便, 经 bat 转发会被截断)
            key, _, raw = arg.partition("=")
            pairs.append((key.strip(), _parse_value(raw.strip())))
            i += 1
            continue
        # 键 + 下一个 token 是值; 没有下一个就当"打开该开关"(值为 True)
        if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            value = _parse_value(argv[i + 1])
            i += 2
        else:
            value = True
            i += 1
        pairs.append((key, value))
    return once, pairs


def apply_overrides(pairs: list[tuple[str, object]], once: bool) -> None:
    """把 (键, 值) 写进 configs/<任务名>.json; once=True 时只打印."""
    if not pairs:
        return

    values: dict[str, object] = {}
    allowed = ALLOWED_KEYS[TARGET_TASK]
    for key, value in pairs:
        if not key:
            print(f"[WARN] 忽略空的配置键")
            continue
        if key not in allowed:
            print(f"[WARN] 未知配置键 {key!r}, 已忽略. "
                  f"可用键: {', '.join(sorted(allowed))}")
            continue
        values[key] = value

    if not values:
        return

    shown = ", ".join(f"{k}={v!r}" for k, v in values.items())
    if once:
        print(f"[OK] 本次运行覆盖: {shown} (不写盘)")
        return

    cfg_path = Path("configs") / f"{TARGET_TASK}.json"
    data = {}
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] 读取 {cfg_path} 失败, 将覆盖: {e}")
    data.update(values)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8"
    )
    print(f"[OK] 已写入 {cfg_path}: {shown}")


def check_update_quietly() -> None:
    """启动前检查更新：失败/离线/未发布一律放行，绝不阻塞启动.

    跳过方式：命令行加 --no-update，或 configs/app.json 写 {"check_update": false}。
    """
    try:
        from src.update import check_for_update
        from src.version import APP_VERSION, UPDATE_REPO

        flag = Path("configs") / "app.json"
        if flag.exists():
            try:
                if json.loads(flag.read_text(encoding="utf-8")).get("check_update") is False:
                    return
            except Exception:
                pass

        release = check_for_update(UPDATE_REPO, APP_VERSION)
        if release:
            print(f"[UPDATE] 发现新版本 {release.tag}（当前 {APP_VERSION}）")
            print(f"[UPDATE] 下载页: https://github.com/{UPDATE_REPO}/releases/latest")
            print("[UPDATE] 可运行 `.runtime\\python\\python -m src.update --apply` 自动更新")
    except Exception:
        pass  # 更新检查永不阻塞启动


def bootstrap_resources_quietly() -> None:
    """首次运行资源引导（启动器渠道）：缺 ok_templates/models 时自动下载.

    失败/离线/无令牌一律放行，绝不阻塞启动；跳过条件同更新检查。
    """
    try:
        from src.update import bootstrap_resources_quietly as _boot
        _boot()
    except Exception:
        pass  # 资源引导永不阻塞启动


def main() -> None:
    argv = sys.argv[1:]

    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return

    if "--no-update" not in argv:
        check_update_quietly()
        bootstrap_resources_quietly()

    once, pairs = parse_args(argv)
    apply_overrides(pairs, once)

    import multiprocessing

    multiprocessing.freeze_support()

    from src.config import config

    import ok

    ok_instance = ok.OK(config)
    ok_instance.start()


if __name__ == "__main__":
    main()