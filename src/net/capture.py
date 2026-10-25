"""抓包链路自动起停(mitmdump + mihomo TUN) —— 不用再手动开两个窗口.

为什么要有这个模块
------------------
网络读牌要抓包, 而抓包要满足两个条件: ① `mitmdump` 在 127.0.0.1:23410 上监听;
② mihomo 的 TUN 把雀魂进程流量导进来. 少任何一个, 整条链路都读不到东西, 而
症状只是 `NetAssist.status()` 报 `no-spool` —— 用户得自己去翻 docs/NET_CAPTURE.md
猜是漏了哪一步 (2026-10-16 实机就踩过: 没开抓包, 日志只说"不可用").

主程序本来就是管理员权限 (`pyappify.yml` 的 `uac: true` / `admin: true`,
`run_admin.bat` 也走 `Start-Process -Verb RunAs`), 所以建 TUN 网卡的权限已经具备,
只差把两个进程拉起来. 这个模块就是干这个的.

安全边界(与 addon 一致, 不变)
----------------------------
- **只读**: addon 从不修改任何流量; 出牌仍走真实鼠标点击.
- **只关自己起的**: 若 23410 已经在监听 / `mihomo-tun` 网卡已存在, 一律认为是
  用户自己起的, 全程不动它, 退出时也不关(别人的进程不归我们管).
- **失败不阻塞**: 任何一步起不来都只记 errors, 由调用方决定是否继续 —— 网络
  读牌永远是"增强", 不是"依赖"(见 AGENTS.md 铁律 3 的门控兜底要求).

为什么要"先探测再起"而不是无脑起
--------------------------------
无脑起会有两个后果: ① 端口被占时 mitmdump 起不来, 白等超时; ② 用户手动起过
的抓包会被我们退出时误杀. 探测一次的成本是 0.3s 的连接尝试, 换来幂等.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from src import runtime_paths

ROOT = Path(__file__).resolve().parents[2]
MITM_DIR = ROOT / "tools" / "mitm"
ADDON = MITM_DIR / "liqi_addon.py"
TUN_YAML = MITM_DIR / "mihomo-tun.yaml"
LOG_PATH = ROOT / "logs" / "mitm_capture.log"

#: Windows: 子进程不弹控制台窗口(否则任务跑起来会蹦出黑框).
CREATE_NO_WINDOW = 0x08000000

# 运行时资产的路径与查找全部来自 runtime_paths(bootstrap 装、这里用).
find_mihomo = runtime_paths.find_mihomo
ca_cert_path = runtime_paths.ca_cert_path


def is_admin() -> bool:
    """当前进程是否管理员(TUN 建网卡需要). 非 Windows / 取不到一律 False."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:  # noqa: BLE001 - 探测失败按"不是"处理, 走降级路径
        return False


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    """127.0.0.1:port 上有没有人在监听. 起进程前先问一次, 避免端口冲突."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, int(port))) == 0
    except Exception:  # noqa: BLE001
        return False


def tun_device_present(name: str = "mihomo-tun") -> bool:
    """`name` 网卡是否已存在(说明 TUN 已经在跑).

    用 `netsh` 而不是 ipconfig: netsh 的输出里"接口名"是 ASCII 的设备名,
    不会随系统语言变(`ipconfig` 的适配器标题是本地化的, grep 不稳).
    """
    if os.name != "nt":
        return False
    try:
        out = subprocess.run(
            ["netsh", "interface", "show", "interface"],
            capture_output=True, timeout=6,
        ).stdout or b""
    except Exception:  # noqa: BLE001
        return False
    return name.encode("ascii", "ignore").lower() in out.lower()


def ca_trusted(cert: Path | None = None) -> bool:
    """mitmproxy 的 CA 是否已在受信任的根里(当前用户 或 本地计算机).

    **为什么起 TUN 前必须问这个**: TUN 是网卡层分流, 它把雀魂的 **HTTPS**
    (登录 / 大厅 API / 资源)也一并导进 mitmdump. 受信任根里没有这个 CA 时,
    客户端会拒绝 mitmproxy 签的证书 -> **游戏直接登录不上**, 这比"读不到协议"
    严重得多(协议读不到只是本局停手, 游戏连不上是整件事都干不了).
    ws 网关是明文、不校验证书, 所以"只看 ws 能不能抓到"发现不了这个坑.

    只做**检测**: 往证书库里写东西是系统级副作用, 默认不干(见 install_ca).
    """
    if os.name != "nt":
        return False
    for scope in (["-user"], []):        # 当前用户 -> 本地计算机
        try:
            out = subprocess.run(
                ["certutil", *scope, "-store", "Root"],
                capture_output=True, timeout=20,
            ).stdout or b""
        except Exception:  # noqa: BLE001
            continue
        if b"mitmproxy" in out:
            return True
    return False


def install_ca(cert: Path | None = None) -> bool:
    """把 mitmproxy CA 装进**当前用户**的受信任根(不需要管理员).

    只在 tuning.CAPTURE_INSTALL_CA 显式打开时才会被调用 —— 默认只检测不安装,
    因为往系统证书库写东西的影响面比"起一个本地代理进程"大得多, 而且这一步
    是一次性的(装完永久有效), 没有必须自动化的理由.
    """
    if os.name != "nt":
        return False
    path = Path(cert) if cert else ca_cert_path()
    if not path.is_file():
        return False
    try:
        done = subprocess.run(
            ["certutil", "-user", "-addstore", "-f", "Root", str(path)],
            capture_output=True, timeout=30,
        )
    except Exception:  # noqa: BLE001
        return False
    return done.returncode == 0


def mitm_cmd(prefix: list[str], port: int, addon: Path = ADDON,
             upstream: str = "", confdir: Path = runtime_paths.MITM_CONF) -> list[str]:
    """mitmdump 命令行(纯函数, 便于单测锁死参数顺序).

    `prefix` 是 runtime_paths.mitmdump_cmd() 给的启动前缀(解释器 + 入口函数).
    `confdir` 指进项目内: CA 与配置跟着目录走, 换机器不用重新签发证书.
    """
    cmd = [*prefix, "--set", f"confdir={confdir}", "-p", str(port),
           "--mode", "regular", "-s", str(addon)]
    if upstream:
        cmd += ["--mode", f"upstream:{upstream}"]
    return cmd


def tun_cmd(mihomo: str, config_dir: Path = MITM_DIR,
            config: Path = TUN_YAML) -> list[str]:
    """mihomo TUN 命令行(与 tools/mitm/start_tun.bat 等价)."""
    return [mihomo, "-d", str(config_dir), "-f", str(config)]


@dataclass
class CaptureStatus:
    """一次 start() 的结果. 调用方只需要看 ok / spool / notes / errors."""

    ok: bool = False                  # 抓包链路已就位(mitmdump 在监听)
    spool: Path | None = None         # 本次 spool 路径, 交给 NetAssist 显式使用
    mitm_started: bool = False        # mitmdump 是我们起的吗(决定退出关不关)
    tun_started: bool = False         # mihomo 是我们起的吗
    reused_mitm: bool = False         # 本来就在跑, 复用(退出时不动)
    reused_tun: bool = False
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """一行中文摘要, 直接进日志."""
        bits = []
        if self.mitm_started:
            bits.append("已启动 mitmdump")
        elif self.reused_mitm:
            bits.append("复用已在监听的 mitmdump")
        if self.tun_started:
            bits.append("已启动 mihomo TUN")
        elif self.reused_tun:
            bits.append("复用已在跑的 TUN")
        if self.spool is not None:
            bits.append(f"spool={self.spool.name}")
        body = "; ".join(bits) or "未就位"
        if self.errors:
            body += " | 问题: " + "; ".join(self.errors)
        return body


class CaptureSupervisor:
    """抓包链路的生命周期. `start()` 幂等, `stop()` 只关自己起的."""

    def __init__(self, *, port: int = 23410, spool: Path | str | None = None,
                 upstream: str = "", mitm_bin: str = "", mihomo_bin: str = "",
                 start_timeout: float = 8.0, tun_device: str = "mihomo-tun",
                 install_ca: bool = False, log_path: Path | str | None = None,
                 runner=None, port_probe=None, tun_probe=None,
                 admin_probe=None, ca_probe=None):
        self.port = int(port)
        self.spool = Path(spool) if spool else None
        self.upstream = upstream
        self.mitm_bin = mitm_bin
        self.mihomo_bin = mihomo_bin
        self.start_timeout = float(start_timeout)
        self.tun_device = tun_device
        self.install_ca = bool(install_ca)
        self.log_path = Path(log_path) if log_path else LOG_PATH
        # 注入口子(测试用): 不注入就用模块级真实现.
        self._runner = runner or subprocess.Popen
        self._port_probe = port_probe or port_open
        self._tun_probe = tun_probe or tun_device_present
        self._admin_probe = admin_probe or is_admin
        self._ca_probe = ca_probe or ca_trusted
        self._mitm_proc = None
        self._tun_proc = None
        self._status: CaptureStatus | None = None

    # ------------------------------------------------------------- 生命周期

    def start(self) -> CaptureStatus:
        """确保抓包链路就位. 幂等: 已在跑的一律复用. 绝不抛异常."""
        if self._status is not None and self._status.ok:
            return self._status
        status = CaptureStatus()
        try:
            self._start_mitm(status)
            self._start_tun(status)
        except Exception as exc:  # noqa: BLE001 - 抓包永不拖垮主循环
            status.errors.append(f"起抓包时异常: {type(exc).__name__}: {exc}")
        self._status = status
        return status

    def _start_mitm(self, status: CaptureStatus) -> None:
        if self._port_probe(self.port):
            status.reused_mitm = True
            status.ok = True
            status.notes.append(f":{self.port} 已在监听, 复用(退出时不动)")
            return
        prefix = [self.mitm_bin] if self.mitm_bin else runtime_paths.mitmdump_cmd()
        if not prefix:
            status.errors.append("运行时里没有 mitmproxy —— 先跑 setup.bat")
            return
        if self.spool is None:
            from src.net.bridge import spool_path_for

            self.spool = spool_path_for()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["LIQI_SPOOL"] = str(self.spool)
        runtime_paths.MITM_CONF.mkdir(parents=True, exist_ok=True)
        cmd = mitm_cmd(prefix, self.port, upstream=self.upstream)
        self._mitm_proc = self._spawn(cmd, env=env)
        if self._mitm_proc is None:
            status.errors.append("mitmdump 启动失败(见 logs/mitm_capture.log)")
            return
        if not self._wait_port():
            status.errors.append(f"mitmdump 起了但 :{self.port} 一直没就绪(超时)")
            return
        status.mitm_started = True
        status.ok = True
        status.spool = self.spool
        status.notes.append(f"mitmdump 已监听 :{self.port}, spool={self.spool.name}")

    def _start_tun(self, status: CaptureStatus) -> None:
        if self._tun_probe(self.tun_device):
            status.reused_tun = True
            status.notes.append(f"TUN 网卡 {self.tun_device} 已在, 复用")
            return
        mihomo = self.mihomo_bin or find_mihomo()
        if not mihomo:
            status.errors.append(
                "没找到 mihomo, 流量导不进来. 修法: 双击 setup.bat(会放一份到 "
                ".runtime\\bin\\mihomo.exe), 或用 OKMAJSOUL_MIHOMO 指定路径")
            return
        if not self._admin_probe():
            status.errors.append("非管理员权限, 建不了 TUN 网卡(请用 run_admin.bat 启动)")
            return
        # 起 TUN 前必须确认 CA 已受信任: TUN 会把游戏的 HTTPS(登录/大厅)也
        # 导进 mitmdump, CA 不在信任库里客户端会直接连不上 —— 那比"读不到
        # 协议"严重得多. 宁可不起 TUN(只少读协议, 游戏照常), 不可把游戏搞挂.
        if not self._ca_probe():
            if self.install_ca and install_ca():
                status.notes.append("已把 mitmproxy CA 装进当前用户受信任根")
            else:
                status.errors.append(
                    "mitmproxy CA 没装进受信任根, 已跳过 TUN(否则游戏 HTTPS 会"
                    "连不上). 修法: setup.bat --ca, 或双击 "
                    f"{ca_cert_path()} -> 装到「受信任的根证书颁发机构」")
                return
        self._tun_proc = self._spawn(tun_cmd(mihomo), env=None)
        if self._tun_proc is None:
            status.errors.append("mihomo 启动失败(见 logs/mitm_capture.log)")
            return
        status.tun_started = True
        status.notes.append("mihomo TUN 已启动")

    def _spawn(self, cmd: list[str], env=None):
        """起一个后台子进程, stdout/stderr 追加到日志文件(不留管道, 防阻塞)."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(self.log_path, "ab", buffering=0)
            owned = True
        except Exception:  # noqa: BLE001
            handle, owned = subprocess.DEVNULL, False
        flags = CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            proc = self._runner(
                cmd, stdout=handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=env, creationflags=flags,
            )
        except Exception:  # noqa: BLE001 - 起不来就是起不来, 交给调用方降级
            proc = None
        # 父进程这份句柄要立刻还回去: Popen 已经把 fd 交给子进程, 我们留着
        # 只是泄漏(而且子进程活着时日志文件也删不掉).
        if owned:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass
        if proc is None:
            return None
        # 立刻 poll 一下: 有些失败(找不到 DLL/地址被占)是瞬间退出的,
        # 不问一声就会得到一个已经死掉还当它活着的句柄.
        time.sleep(0.1)
        if proc.poll() is not None:
            return None
        return proc

    def _wait_port(self) -> bool:
        """等 mitmdump 真正开始监听(进程可能先起来再 bind)."""
        deadline = time.time() + self.start_timeout
        while time.time() < deadline:
            if self._port_probe(self.port):
                return True
            if self._mitm_proc is not None and self._mitm_proc.poll() is not None:
                return False     # 进程已经退了, 再等也没用
            time.sleep(0.2)
        return False

    def alive(self) -> bool | None:
        """我们自己起的抓包进程都还活着吗.

        True  = 起过的都还活着（含"只起了一个、那个还活着"）;
        False = **至少一个已经退了** —— 链路废了，协议读数从此不可信;
        None  = 我们一个都没起（复用外部实例 / 本来就没启用）.

        None 与 True 要分开：那是用户的选择（复用外部 mitmdump、或干脆没开
        抓包），不是故障，归不到"链路坏了"那一类。

        为什么不等桥的 `stale_after` 超时再反推"没新事件 = 坏了": 这条链路是
        **本进程 spawn 的**，进程退了本来就该立刻知道。等 10s 超时那 10 秒里，
        桥喂的是旧状态，`protocol_says_my_turn()` 之类会冻结在最后一帧的真值
        上 —— 那 10 秒里决策层是在拿一份死掉的协议当真的用（2026-10-18 核到
        全项目只有 1 处用了 `fresh()`，三条关键读数路径全部裸奔）。
        """
        procs = [p for p in (self._mitm_proc, self._tun_proc) if p is not None]
        if not procs:
            return None
        # poll() 只查退出码、不等待，微秒级 —— 每轮都问得起.
        return all(p.poll() is None for p in procs)

    def stop(self) -> list[str]:
        """关掉**我们自己起的**进程. 复用来的一个都不碰. 返回动作描述."""
        done: list[str] = []
        for name, attr in (("mihomo TUN", "_tun_proc"), ("mitmdump", "_mitm_proc")):
            proc = getattr(self, attr)
            if proc is None or proc.poll() is not None:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 - 关不掉就强杀, 别卡住收工
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                setattr(self, attr, None)
            done.append(f"已关闭 {name}")
        return done

    @property
    def status(self) -> CaptureStatus | None:
        return self._status
