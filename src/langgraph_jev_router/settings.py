"""Global settings.

Precedence, highest first:
  1. keyword arguments on ``@jev_first(...)`` / ``install_shortcut(...)``
  2. ``configure(...)`` in code
  3. environment variables
  4. built-in defaults

Two environment switches always win, because they are operational safety valves:
``JEV_ROUTER_DISABLED=1`` (your original routers handle everything) and
``JEV_ROUTER_SHADOW=1`` (Jev only observes; your routers keep deciding).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, fields, replace
from typing import Any, Optional

# Pinned rather than "jev-latest": the alias moves when TypeSafe ships a release, and the
# confidence values your threshold was tuned on can move with it. Override with
# configure(model=...) or TYPESAFE_DEFAULT_MODEL.
DEFAULT_MODEL = "jev-1.13.0"
DEFAULT_THRESHOLD = 0.7

# Published input price, September 2026. Output tokens are free. Used only for estimates.
JEV_PRICE_PER_MILLION_INPUT_TOKENS = 0.042


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    backend: Any = None
    """Object with choose()/achoose(). None means the real TypeSafe backend."""
    model: Optional[str] = None
    threshold: float = DEFAULT_THRESHOLD
    shadow: bool = False
    disabled: bool = False
    log_path: Optional[str] = None
    """Append every decision as a JSON line to this file (for `jev-router report`)."""
    log_inputs: bool = False
    """Store the text sent to Jev in the log. Off by default: only a hash is kept."""
    cache_size: int = 1024


_overrides: dict[str, Any] = {}
_lock = threading.Lock()
_FIELDS = {f.name for f in fields(Settings)}


def configure(**kwargs: Any) -> None:
    """Set global defaults, e.g. ``configure(threshold=0.8, log_path="jev.jsonl")``."""
    unknown = set(kwargs) - _FIELDS
    if unknown:
        raise TypeError(f"Unknown setting(s): {', '.join(sorted(unknown))}. Valid: {', '.join(sorted(_FIELDS))}")
    if "threshold" in kwargs and not 0.0 <= float(kwargs["threshold"]) <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    with _lock:
        _overrides.update(kwargs)


def reset_configuration() -> None:
    """Forget everything passed to configure(). Mainly for tests."""
    with _lock:
        _overrides.clear()


def get_settings() -> Settings:
    base = Settings(
        model=os.getenv("TYPESAFE_DEFAULT_MODEL") or DEFAULT_MODEL,
        threshold=float(os.getenv("JEV_ROUTER_THRESHOLD", DEFAULT_THRESHOLD)),
        log_path=os.getenv("JEV_ROUTER_LOG") or None,
        log_inputs=env_flag("JEV_ROUTER_LOG_INPUTS"),
    )
    with _lock:
        merged = replace(base, **_overrides)
    if env_flag("JEV_ROUTER_DISABLED"):
        merged.disabled = True
    if env_flag("JEV_ROUTER_SHADOW"):
        merged.shadow = True
    return merged
