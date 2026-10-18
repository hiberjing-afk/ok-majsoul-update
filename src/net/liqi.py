"""liqi 协议 wire 层解析: WebSocket 帧 -> 语义事件(纯逻辑, 不依赖 ok-script).

报文结构(参照 majsoul_wrapper / MahjongCopilot 两个上游实现):

    Notify  0x01                      服务端通知, 无 msg_id
    Req     0x02 + uint16 小端 id      客户端请求
    Res     0x03 + uint16 小端 id      服务端对对应 req 的回复

后随 protobuf, 固定两段: field 1 = 方法名(如 ".lq.ActionPrototype"),
field 2 = 该方法的数据. 对局内的动作全部走 ".lq.ActionPrototype" 通知,
它的载荷是二级消息 ActionPrototype{step, name, data}: name 是动作名
(如 "ActionDiscardTile"), data 才是动作本身的 protobuf.

两个必须记住的坑:
1. **二级 data 的 XOR 混淆随协议版本变**: majsoul_wrapper 直接解析二级
   data, MahjongCopilot 先按 XOR_KEYS 解码 -- 两边都能跑, 说明不同
   版本混淆不同. 这里两手都试(先明文, 失败/空再 XOR), 都失败不抛,
   交给上层记 protocol_suspect(宁可漏判不可锁死).
2. **protobuf 7.x 的 Message.FromString() 返回新对象, 不保证原地修改**
   (实测: m2.FromString(b) 之后 m2 仍是空的). 本模块一律用
   ParseFromString().

Res 的方法名不在报文里(field 1 是空串), 必须靠 req 时记下的 msg_id
关联补全 -- 所以 parser 是**按 WebSocket 连接**实例化的(msg_id 按连接
滑动), 大厅和牌桌两条连接各持一个实例, 不会串.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import IntEnum

from google.protobuf.descriptor import FieldDescriptor

from .proto import liqi_pb2
from .tiles import all_tiles_valid, is_tile


class MsgType(IntEnum):
    NOTIFY = 1
    REQ = 2
    RES = 3


# MahjongCopilot liqi.py(2024)用过的二级 data XOR keys. 协议再变就改这里,
# 配合 tools/sync_liqi_proto.py 重同步.
XOR_KEYS = (0x84, 0x5E, 0x4E, 0x42, 0x39, 0xA2, 0x1F, 0x60, 0x1C)

ACTION_METHOD = ".lq.ActionPrototype"
GAME_METHOD_PREFIX = ".lq."

# 需要解码 req/res 数据的少数方法(避免运行时依赖 liqi.json 全量方法表).
# 其余 req/res 只记方法名, data 留 None.
REQUEST_TYPES = {
    ".lq.FastTest.authGame": "ReqAuthGame",
}
RESPONSE_TYPES = {
    ".lq.FastTest.authGame": "ResAuthGame",
    ".lq.FastTest.enterGame": "ResEnterGame",
    ".lq.FastTest.syncGame": "ResSyncGame",
}
NOTIFY_TYPES = {
    ".lq.NotifyGameEndResult": "NotifyGameEndResult",
    # 对局被掐(被踢/断线超时/服务端异常): 唯一带 reason 的消息, 解出来
    # 才能把"为什么这局没了"写进日志(见 state.py 的 NotifyGameTerminate 分支).
    ".lq.NotifyGameTerminate": "NotifyGameTerminate",
}

# req->res 关联表上限: msg_id 是 uint16 且服务端未必条条回 res
_PENDING_MAX = 4096


class LiqiError(Exception):
    """wire 层解析失败(流量不是 liqi, 或报文残缺)."""


def xor_decode(data: bytes) -> bytes:
    """二级 data 的 XOR 解码(与 encode 互为逆运算)."""
    if not data:
        return data
    n = len(data)
    out = bytearray(n)
    for i in range(n):
        u = ((23 ^ n) + 5 * i + XOR_KEYS[i % len(XOR_KEYS)]) & 0xFF
        out[i] = data[i] ^ u
    return bytes(out)


def _read_varint(buf: bytes, p: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while p < len(buf):
        b = buf[p]
        p += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, p
        shift += 7
        if shift > 63:
            raise LiqiError("varint too long")
    raise LiqiError("truncated varint")


def _iter_wire_fields(buf: bytes):
    """遍历 protobuf wire 格式字段, yield (field_number, wire_type, value).

    value: wire_type 2 给 bytes, 其余给 None(调用方不关心标量值).
    未知 wire type 直接抛 LiqiError -- 不是 liqi 流量.
    """
    p = 0
    while p < len(buf):
        tag, p = _read_varint(buf, p)
        fn, wt = tag >> 3, tag & 7
        if fn == 0:
            raise LiqiError("field number 0")
        if wt == 0:
            _, p = _read_varint(buf, p)
            yield fn, wt, None
        elif wt == 1:
            if p + 8 > len(buf):
                raise LiqiError("truncated 64bit field")
            yield fn, wt, buf[p:p + 8]
            p += 8
        elif wt == 2:
            ln, p = _read_varint(buf, p)
            if p + ln > len(buf):
                raise LiqiError("truncated length-delimited field")
            yield fn, wt, buf[p:p + ln]
            p += ln
        elif wt == 5:
            if p + 4 > len(buf):
                raise LiqiError("truncated 32bit field")
            yield fn, wt, buf[p:p + 4]
            p += 4
        else:
            raise LiqiError(f"unsupported wire type {wt}")


def _is_repeated(fd) -> bool:
    """字段是否 repeated.

    坑: protobuf 7.x(upb)起 FieldDescriptor.label / .cpp_type 被移除,
    只有 is_repeated 可用; 老版本没有 is_repeated 时才回退 label.
    """
    if hasattr(fd, "is_repeated"):
        return bool(fd.is_repeated)
    return fd.label == FieldDescriptor.LABEL_REPEATED


def _field_value(fd, value):
    if _is_repeated(fd):
        if fd.type == FieldDescriptor.TYPE_MESSAGE:
            return [proto_to_dict(v) for v in value]
        if fd.type == FieldDescriptor.TYPE_BYTES:
            return [base64.b64encode(v).decode("ascii") for v in value]
        return list(value)
    if fd.type == FieldDescriptor.TYPE_MESSAGE:
        return proto_to_dict(value)
    if fd.type == FieldDescriptor.TYPE_BYTES:
        return base64.b64encode(value).decode("ascii")
    return value


def proto_to_dict(msg) -> dict:
    """protobuf 消息 -> dict( snake_case 字段名, 只含被设置过的字段 ).

    不用 json_format.MessageToDict: 它会把字段名转 camelCase(accountId),
    并把 int64 转成字符串, 都与协议字段名对不上, 平白多一层映射.
    bytes 一律 base64(与上游实现一致, 便于 JSONL spool).
    """
    out: dict = {}
    for fd, value in msg.ListFields():
        out[fd.name] = _field_value(fd, value)
    return out


def _parse_dual(cls_name: str, raw: bytes):
    """按类名解析 protobuf; 先明文, 明文失败或解出空消息再试 XOR.

    返回消息对象或 None(都失败). 空消息也返回 -- 调用方按需判断.
    """
    cls = getattr(liqi_pb2, cls_name, None)
    if cls is None:
        return None
    fallback = None
    for candidate in (raw, xor_decode(raw)):
        msg = cls()
        try:
            msg.ParseFromString(candidate)
        except Exception:  # noqa: BLE001 - protobuf 对畸形输入各版本抛法不同
            continue
        if msg.ListFields():
            return msg
        if fallback is None:
            fallback = msg
    return fallback


def decode_nested_action(name: str, raw: bytes) -> dict | None:
    """二级动作 data -> dict(自动处理 XOR). name 不是已知消息时返回 None."""
    if not name:
        return None
    msg = _parse_dual(name, raw)
    if msg is None:
        return None
    return proto_to_dict(msg)


@dataclass
class LiqiMessage:
    """一条解码后的 liqi 报文."""

    msg_type: MsgType
    msg_id: int                 # notify 恒为 -1
    method: str                 # 规范方法名; res 经 req 关联补全
    payload: bytes              # field-2 原始字节
    from_client: bool
    action: str | None = None   # ActionPrototype 的二级动作名
    data: dict | None = None    # 解码后的数据(未解码/解不动为 None)
    step: int = 0               # ActionPrototype.step(单调递增, 用于去重)

    def to_dict(self) -> dict:
        return {
            "type": int(self.msg_type),
            "id": self.msg_id,
            "method": self.method,
            "action": self.action,
            "step": self.step,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LiqiMessage":
        return cls(
            msg_type=MsgType(d.get("type", 1)),
            msg_id=int(d.get("id", -1)),
            method=str(d.get("method", "")),
            payload=b"",
            from_client=False,
            action=d.get("action"),
            data=d.get("data"),
            step=int(d.get("step", 0) or 0),
        )


class LiqiParser:
    """一个 WebSocket 连接持有一个实例(req/res 的 msg_id 按连接滑动)."""

    def __init__(self) -> None:
        self._pending: dict[int, str] = {}   # msg_id -> req 方法名(等它的 res)
        self.stats = {"parsed": 0, "skipped": 0, "undecoded": 0}

    def parse(self, buf: bytes, from_client: bool) -> LiqiMessage | None:
        """ws 报文 -> LiqiMessage; 不是 liqi 流量返回 None. 绝不抛."""
        if not isinstance(buf, (bytes, bytearray)) or len(buf) < 2:
            return None
        buf = bytes(buf)
        try:
            head = buf[0]
            if head == MsgType.NOTIFY:
                msg_type, msg_id, body = MsgType.NOTIFY, -1, buf[1:]
            elif head in (MsgType.REQ, MsgType.RES):
                if len(buf) < 4:
                    return None
                msg_type = MsgType.REQ if head == MsgType.REQ else MsgType.RES
                msg_id = int.from_bytes(buf[1:3], "little")
                body = buf[3:]
            else:
                return None

            method = ""
            payload = b""
            for fn, wt, value in _iter_wire_fields(body):
                if fn == 1 and wt == 2 and not method:
                    method = value.decode("utf-8", "replace")
                elif fn == 2 and wt == 2:
                    payload = value

            if msg_type == MsgType.RES:
                # res 的 field 1 是空串, 方法名靠 req 时记下的关联补全
                method = self._pending.pop(msg_id, "")
                if not method:
                    return None
            elif msg_type == MsgType.REQ:
                # 只有 req 记关联; 别的连接流量(非 .lq. 方法)不进 pending
                if not method.startswith(GAME_METHOD_PREFIX):
                    return None
                if len(self._pending) >= _PENDING_MAX:
                    # 没等到 res 的 req(协议变更/异常路径): 丢最旧的, 防泄漏
                    self._pending.pop(next(iter(self._pending)))
                self._pending[msg_id] = method
            elif not method.startswith(GAME_METHOD_PREFIX):
                # notify 同样只关心 liqi 方法
                return None

            if not method.startswith(GAME_METHOD_PREFIX):
                return None

            msg = LiqiMessage(msg_type, msg_id, method, payload, from_client)
            self._decode(msg)
            self.stats["parsed"] += 1
            if msg.data is None:
                self.stats["undecoded"] += 1
            return msg
        except LiqiError:
            self.stats["skipped"] += 1
            return None
        except Exception:  # noqa: BLE001 - 解析器永不把异常抛给捕获线程
            self.stats["skipped"] += 1
            return None

    def _decode(self, msg: LiqiMessage) -> None:
        """按方法名解 data. 解不出来留 None(不抛)."""
        try:
            if msg.msg_type == MsgType.NOTIFY:
                if msg.method == ACTION_METHOD:
                    outer = _parse_dual("ActionPrototype", msg.payload)
                    if outer is None or not outer.name:
                        return
                    msg.action = outer.name
                    msg.step = int(outer.step)
                    msg.data = decode_nested_action(outer.name, outer.data)
                    return
                cls_name = NOTIFY_TYPES.get(msg.method)
            elif msg.msg_type == MsgType.REQ:
                cls_name = REQUEST_TYPES.get(msg.method)
            else:
                cls_name = RESPONSE_TYPES.get(msg.method)
            if cls_name is None:
                return
            parsed = _parse_dual(cls_name, msg.payload)
            if parsed is not None:
                msg.data = proto_to_dict(parsed)
        except Exception:  # noqa: BLE001
            return


def build_notify(method: str, payload: bytes) -> bytes:
    """合成 notify 帧(测试/回放工具用, 生产链路只读不用)."""
    body = b"\x0a" + _varint(len(method.encode())) + method.encode()
    body += b"\x12" + _varint(len(payload)) + payload
    return b"\x01" + body


def build_req(msg_id: int, method: str, payload: bytes) -> bytes:
    body = b"\x0a" + _varint(len(method.encode())) + method.encode()
    body += b"\x12" + _varint(len(payload)) + payload
    return b"\x02" + msg_id.to_bytes(2, "little") + body


def build_res(msg_id: int, payload: bytes) -> bytes:
    """res 的 field 1 是空方法名(协议如此)."""
    body = b"\x0a\x00"
    body += b"\x12" + _varint(len(payload)) + payload
    return b"\x03" + msg_id.to_bytes(2, "little") + body


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def looks_like_action_payload(data: dict | None) -> bool:
    """解出来的动作数据像不像真的(协议漂移/误解析的第一道筛子)."""
    if not data:
        return False
    if "tile" in data and not is_tile(data["tile"]):
        return False
    if "tiles" in data and not all_tiles_valid(data["tiles"]):
        return False
    return True
