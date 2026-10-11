"""Backwards compatibility with 0.1.x."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from .engine import RouterEngine


class JevConditionalEdge:
    """Deprecated 0.1.x API. Use ``@jev_first`` on your existing router instead.

    Kept so ``JevConditionalEdge(question, routes).route`` keeps working. Unlike 0.1.x it calls
    the real Jev API. When Jev is unsure or unavailable it returns ``fallback_route``
    (default: the first route) instead of crashing.
    """

    def __init__(self, question: str, routes: Dict[str, str], *, fallback_route: Optional[str] = None,
                 threshold: float = 0.0, **options: Any) -> None:
        warnings.warn(
            "JevConditionalEdge is deprecated; decorate your existing router with @jev_first instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not routes:
            raise ValueError("routes must not be empty")
        fallback = fallback_route or next(iter(routes))

        def _fallback(state: Any) -> str:
            return fallback

        self._engine = RouterEngine(_fallback, question=question, routes=routes, threshold=threshold,
                                    name=options.pop("name", "JevConditionalEdge"), **options)
        self.question = question
        self.routes = routes

    def route(self, state: Any) -> str:
        return self._engine.call((state,), {})

    async def aroute(self, state: Any) -> str:
        return await self._engine.acall((state,), {})
