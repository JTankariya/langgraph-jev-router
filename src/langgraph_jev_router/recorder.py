"""Every routing decision becomes one event. Events power the savings and shadow reports.

Privacy: by default the text sent to Jev is NOT stored, only a short hash. Turn on
``configure(log_inputs=True)`` (or JEV_ROUTER_LOG_INPUTS=1) if you want to read the
disagreements yourself. Nothing is ever sent anywhere except the Jev call itself.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .settings import get_settings

Event = Dict[str, Any]


class Recorder:
    def __init__(self, maxlen: int = 20_000) -> None:
        self._events: Deque[Event] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, event: Event) -> None:
        settings = get_settings()
        event = {k: v for k, v in event.items() if v is not None}
        if not settings.log_inputs:
            event.pop("input", None)
        event.setdefault("ts", round(time.time(), 3))
        with self._lock:
            self._events.append(event)
            if settings.log_path:
                path = Path(settings.log_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, default=str) + "\n")

    def events(self, router: Optional[str] = None) -> List[Event]:
        with self._lock:
            items = list(self._events)
        return [e for e in items if router is None or e.get("router") == router]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


_recorder = Recorder()


def get_recorder() -> Recorder:
    return _recorder


def load_events(path: str) -> List[Event]:
    """Read a JSONL decision log written via ``configure(log_path=...)`` / JEV_ROUTER_LOG."""
    events: List[Event] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
