"""Jev-first routing for LangGraph: keep your router, let Jev take the easy decisions.

    from langgraph_jev_router import jev_first

    @jev_first
    def route(state) -> Literal["billing", "tech"]:
        ...  # your existing LLM router, unchanged; it now runs only when Jev is unsure
"""

from ._version import __version__
from .backends import FakeJev, TypeSafeBackend
from .compat import JevConditionalEdge
from .decorator import jev_first
from .discovery import attach_routes
from .engine import RouterEngine
from .recorder import get_recorder, load_events
from .report import build_report
from .settings import configure, reset_configuration
from .shortcut import install_shortcut
from .types import ChoiceRequest, Decision

__all__ = [
    "__version__",
    "jev_first",
    "attach_routes",
    "install_shortcut",
    "configure",
    "reset_configuration",
    "build_report",
    "load_events",
    "get_recorder",
    "TypeSafeBackend",
    "FakeJev",
    "Decision",
    "ChoiceRequest",
    "RouterEngine",
    "JevConditionalEdge",
]
