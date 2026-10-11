from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, Hashable, Iterable, Optional

logger = logging.getLogger("langgraph_jev_router")

END = "__end__"
END_DESCRIPTION = "Nothing more needs to be done for this request."

_warned: set[str] = set()
_warn_lock = threading.Lock()


def warn_once(key: str, message: str) -> None:
    with _warn_lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.warning(message)


def reset_warnings() -> None:
    with _warn_lock:
        _warned.clear()


def label_base(name: str) -> str:
    """Turn a route name into a plain label Jev can return (``__end__`` -> ``end``)."""
    if name == END:
        return "end"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")


def make_labels(names: Iterable[str]) -> Dict[str, str]:
    """Map unique, API-safe labels back to the original route names."""
    out: Dict[str, str] = {}
    for i, name in enumerate(names):
        base = label_base(name) or f"route_{i}"
        label, n = base, 2
        while label in out:
            label, n = f"{base}_{n}", n + 1
        out[label] = name
    return out


def humanize(name: str) -> str:
    return re.sub(r"[_\-]+", " ", str(name)).strip()


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class LRUCache:
    def __init__(self) -> None:
        self._data: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: Hashable, value: Any, max_size: int) -> None:
        if max_size <= 0:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


decision_cache = LRUCache()
