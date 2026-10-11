"""图像读写: 绕开 OpenCV 在 Windows 下不认非 ASCII 路径的问题.

cv2.imread / cv2.imwrite 走的是 ANSI 接口, 中文路径会直接失败.
统一改用「读字节 -> imdecode / imencode -> 写字节」.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path) -> np.ndarray | None:
    """读取图像, 支持任意路径(含中文). 失败返回 None."""
    try:
        data = Path(path).read_bytes()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """写入图像, 支持任意路径(含中文). 成功返回 True."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(p.suffix or ".png", image)
        if not ok:
            return False
        p.write_bytes(buf.tobytes())
        return True
    except Exception:
        return False
