from __future__ import annotations

import functools
from typing import Any, Callable, Optional, TypeVar

from .engine import RouterEngine, RoutesArg
from .types import Decision

F = TypeVar("F", bound=Callable[..., Any])


def jev_first(
    fn: Optional[F] = None,
    *,
    routes: RoutesArg = None,
    threshold: Optional[float] = None,
    shadow: Optional[bool] = None,
    question: Optional[str] = None,
    state_from: Optional[Callable[[Any], Any]] = None,
    model: Optional[str] = None,
    include_end: bool = False,
    on_jev: Optional[Callable[[Any, Decision], Any]] = None,
    backend: Any = None,
    name: Optional[str] = None,
    kind: Optional[str] = None,
) -> Any:
    """Let Jev make this routing decision first; your function becomes the fallback.

    Works on:
      * a conditional-edge function that returns a route name, and
      * a node that returns ``Command(goto=...)`` (detected from a ``-> Command[...]`` annotation,
        or force it with ``kind="command"``).

    Sync and async functions are both supported. With no arguments it needs nothing else
    if the function is annotated ``-> Literal[...]`` (or you call ``attach_routes(builder)``).

    Options (all optional):
      routes       names, or {name: description}; overrides discovery
      threshold    minimum Jev confidence to act on (default 0.7)
      shadow       True: your function always decides; Jev is only measured
      question     the instruction Jev gets; refer to fields in backticks, e.g. `request`
      state_from   state -> str | dict | None: what Jev sees (default: newest user message)
      model        Jev model version (default pinned, see settings.DEFAULT_MODEL)
      include_end  let Jev pick END (off by default: ending a run is left to your router)
      on_jev       (state, decision) -> state update, for Command nodes when Jev decides
      backend      decision backend (default: real Jev). FakeJev for tests.
      name         name used in reports (default: function's qualified name)
      kind         "edge" or "command"; normally detected
    """

    def decorate(f: F) -> F:
        engine = RouterEngine(
            f,
            routes=routes,
            threshold=threshold,
            shadow=shadow,
            question=question,
            state_from=state_from,
            model=model,
            include_end=include_end,
            on_jev=on_jev,
            backend=backend,
            name=name,
            kind=kind,
        )
        if engine.is_async:

            @functools.wraps(f)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await engine.acall(args, kwargs, kwargs.get("config"))

            wrapper: Any = async_wrapper
        else:

            @functools.wraps(f)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return engine.call(args, kwargs, kwargs.get("config"))

            wrapper = sync_wrapper
        wrapper.jev = engine
        wrapper.__jev_router__ = engine
        return wrapper  # type: ignore[return-value]

    return decorate(fn) if fn is not None else decorate
