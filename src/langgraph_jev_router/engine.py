"""The decision engine shared by @jev_first and install_shortcut.

Live mode:   Jev answers first. Confident and valid -> use it, your router is never called.
             Anything else (low confidence, error, mid-conversation, unknown routes) -> your
             router runs exactly as before. The graph never breaks because of Jev.
Shadow mode: your router always decides. Jev runs concurrently and the two are compared.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from ._util import END, END_DESCRIPTION, decision_cache, logger, make_labels, sha12, warn_once
from .backends import default_backend
from .discovery import annotation_is_command, routes_from_annotation
from .extract import default_state_from, normalize_payload
from .recorder import get_recorder
from .settings import env_flag, get_settings
from .types import ChoiceRequest, Decision

DEFAULT_QUESTION = "Which option should handle the user's latest request in `request`?"

RoutesArg = Union[None, Sequence[str], Mapping[str, Optional[str]]]

_executor: Optional[ThreadPoolExecutor] = None


def _shadow_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="jev-shadow")
    return _executor


def _normalize_routes(routes: RoutesArg) -> Optional[Dict[str, Optional[str]]]:
    if routes is None:
        return None
    if isinstance(routes, Mapping):
        return {str(k): v for k, v in routes.items()}
    return {str(name): None for name in routes}


@dataclass
class _Resolved:
    threshold: float
    model: str
    shadow: bool
    disabled: bool
    backend: Any
    cache_size: int


@dataclass
class Prepared:
    request: Optional[ChoiceRequest]
    label_map: Dict[str, str]
    reason: Optional[str]
    input_hash: Optional[str] = None
    input_text: Optional[str] = None


class RouterEngine:
    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: Optional[str] = None,
        routes: RoutesArg = None,
        question: Optional[str] = None,
        threshold: Optional[float] = None,
        model: Optional[str] = None,
        shadow: Optional[bool] = None,
        state_from: Optional[Callable[[Any], Any]] = None,
        include_end: bool = False,
        on_jev: Optional[Callable[[Any, Decision], Any]] = None,
        backend: Any = None,
        kind: Optional[str] = None,
    ) -> None:
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if kind not in (None, "edge", "command"):
            raise ValueError("kind must be 'edge' or 'command'")
        self.fn = fn
        self.name = name or getattr(fn, "__qualname__", "router")
        self.is_async = inspect.iscoroutinefunction(fn)
        self.kind = kind or ("command" if annotation_is_command(fn) else "edge")
        self.question = question or DEFAULT_QUESTION
        self.threshold = threshold
        self.model = model
        self.shadow = shadow
        self.state_from = state_from
        self.include_end = include_end
        self.on_jev = on_jev
        self.backend = backend
        self._explicit = _normalize_routes(routes)
        self._annotated = routes_from_annotation(fn)
        self._attached: Optional[Dict[str, Optional[str]]] = None
        # Set by install_shortcut: shadow outcomes are observed later, downstream, and the
        # "fallback" is only a hand-off, so timing it would understate your router's latency.
        self.shadow_sink: Optional[Callable[..., None]] = None
        self.measure_original = True

    # ---- routes ------------------------------------------------------------------------
    def set_attached_routes(self, mapping: Mapping[str, Optional[str]]) -> None:
        self._attached = dict(mapping)

    @property
    def routes(self) -> Dict[str, Optional[str]]:
        """Route name -> description, as Jev will see them."""
        attached = self._attached or {}
        if self._explicit is not None:
            return {k: (v or attached.get(k) or (END_DESCRIPTION if k == END else None)) for k, v in self._explicit.items()}
        if self._attached is not None:
            names: Dict[str, Optional[str]] = dict(self._attached)
        elif self._annotated:
            names = {n: None for n in self._annotated}
        else:
            return {}
        out = {k: v for k, v in names.items() if self.include_end or k != END}
        if END in out and not out[END]:
            out[END] = END_DESCRIPTION
        return out

    # ---- settings ----------------------------------------------------------------------
    def _resolve(self) -> _Resolved:
        s = get_settings()
        shadow = env_flag("JEV_ROUTER_SHADOW") or (self.shadow if self.shadow is not None else s.shadow)
        return _Resolved(
            threshold=self.threshold if self.threshold is not None else s.threshold,
            model=self.model or s.model,
            shadow=shadow,
            disabled=s.disabled,
            backend=self.backend or s.backend or default_backend(),
            cache_size=s.cache_size,
        )

    # ---- request -----------------------------------------------------------------------
    def prepare(self, state: Any, cfg: _Resolved) -> Prepared:
        routes = self.routes
        if len(routes) < 2:
            warn_once(
                f"no_routes:{self.name}",
                f"jev: router '{self.name}' has no known routes, so your router is handling every decision. "
                "Fix: annotate it -> Literal[...], call attach_routes(builder) before compile(), "
                "or pass @jev_first(routes=[...]).",
            )
            return Prepared(None, {}, "no_routes")
        if self.state_from is None:
            payload, reason = default_state_from(state)
        else:
            try:
                payload, reason = normalize_payload(self.state_from(state)), None
            except Exception as exc:
                warn_once(f"state_from:{self.name}", f"jev: state_from for '{self.name}' raised {exc!r}; using your router.")
                return Prepared(None, {}, "state_from_error")
            if payload is None:
                reason = "no_input"
        if payload is None:
            return Prepared(None, {}, reason)
        label_map = make_labels(routes)
        options = tuple((label, routes[name]) for label, name in label_map.items())
        text = json.dumps(payload, sort_keys=True, default=str)
        request = ChoiceRequest(question=self.question, options=options, state=payload, model=cfg.model)
        return Prepared(request, label_map, None, sha12(text), text)

    # ---- deciding ----------------------------------------------------------------------
    def _cache_key(self, request: ChoiceRequest, cfg: _Resolved) -> tuple:
        return (id(cfg.backend), request.cache_key())

    def decide(self, request: ChoiceRequest, cfg: _Resolved) -> Decision:
        key = self._cache_key(request, cfg)
        hit = decision_cache.get(key) if cfg.cache_size else None
        if hit is not None:
            return replace(hit, cached=True, latency_ms=0.0)
        start = time.perf_counter()
        decision = cfg.backend.choose(request)
        decision = replace(decision, latency_ms=(time.perf_counter() - start) * 1000)
        decision_cache.put(key, decision, cfg.cache_size)
        return decision

    async def adecide(self, request: ChoiceRequest, cfg: _Resolved) -> Decision:
        key = self._cache_key(request, cfg)
        hit = decision_cache.get(key) if cfg.cache_size else None
        if hit is not None:
            return replace(hit, cached=True, latency_ms=0.0)
        start = time.perf_counter()
        decision = await cfg.backend.achoose(request)
        decision = replace(decision, latency_ms=(time.perf_counter() - start) * 1000)
        decision_cache.put(key, decision, cfg.cache_size)
        return decision

    def _warn_backend(self, exc: BaseException) -> None:
        warn_once(
            f"backend:{type(exc).__name__}",
            f"jev: Jev is unavailable ({type(exc).__name__}: {exc}). Your routers are handling these decisions. "
            "Check TYPESAFE_API_KEY / network, or run `jev-router check`.",
        )

    def accept(self, decision: Decision, prep: Prepared, cfg: _Resolved) -> Optional[str]:
        """Return the route to take if Jev's answer is usable, else the reason it is not."""
        if decision.route not in prep.label_map:
            return None
        if decision.confidence < cfg.threshold:
            return None
        return prep.label_map[decision.route]

    def _rejection_reason(self, decision: Decision, prep: Prepared) -> str:
        return "invalid_choice" if decision.route not in prep.label_map else "low_confidence"

    # ---- output ------------------------------------------------------------------------
    def emit(self, route: str, decision: Decision, state: Any) -> Any:
        if self.kind == "edge":
            return route
        from langgraph.types import Command

        update = self.on_jev(state, decision) if self.on_jev is not None else None
        return Command(goto=route, update=update)

    def route_of(self, result: Any) -> Optional[str]:
        """What your router chose, for shadow comparisons."""
        if isinstance(result, str):
            return result
        goto = getattr(result, "goto", None)
        if isinstance(goto, str):
            return goto
        if isinstance(goto, (list, tuple)) and len(goto) == 1 and isinstance(goto[0], str):
            return goto[0]
        return None

    # ---- recording ---------------------------------------------------------------------
    def record(
        self,
        *,
        mode: str,
        decided_by: str,
        prep: Prepared,
        reason: Optional[str] = None,
        decision: Optional[Decision] = None,
        jev_route: Optional[str] = None,
        original_route: Optional[str] = None,
        original_ms: Optional[float] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        if decision is not None and jev_route is None:
            jev_route = prep.label_map.get(decision.route)  # None if Jev answered outside the routes
        agree = None
        if mode == "shadow" and jev_route is not None and original_route is not None:
            agree = jev_route == original_route
        get_recorder().record(
            {
                "router": self.name,
                "kind": self.kind,
                "mode": mode,
                "decided_by": decided_by,
                "reason": reason,
                "jev_route": jev_route,
                "jev_confidence": round(decision.confidence, 4) if decision else None,
                "original_route": original_route,
                "agree": agree,
                "jev_ms": round(decision.latency_ms, 2) if decision else None,
                "original_ms": round(original_ms, 2) if original_ms is not None else None,
                "model": decision.model if decision else None,
                "request_id": decision.request_id if decision else None,
                "input_tokens": decision.input_tokens if decision and not decision.cached else None,
                "cached": True if decision and decision.cached else None,
                "input_hash": prep.input_hash,
                "input": prep.input_text,
                "error": f"{type(error).__name__}: {error}"[:300] if error else None,
            }
        )

    # ---- sync path ---------------------------------------------------------------------
    def _run_original(self, args, kwargs, prep, *, reason, decision=None, error=None) -> Any:
        start = time.perf_counter()
        result = self.fn(*args, **kwargs)
        ms = (time.perf_counter() - start) * 1000
        self.record(mode="live", decided_by="router", prep=prep, reason=reason, decision=decision,
                    original_route=self.route_of(result), original_ms=ms if self.measure_original else None, error=error)
        return result

    def call(self, args: tuple, kwargs: dict, config: Any = None) -> Any:
        cfg = self._resolve()
        if cfg.disabled:
            return self.fn(*args, **kwargs)
        state = args[0] if args else kwargs.get("state")
        prep = self.prepare(state, cfg)
        if cfg.shadow:
            return self._shadow_sync(prep, args, kwargs, cfg, state, config)
        if prep.request is None:
            return self._run_original(args, kwargs, prep, reason=prep.reason)
        try:
            decision = self.decide(prep.request, cfg)
        except Exception as exc:
            self._warn_backend(exc)
            return self._run_original(args, kwargs, prep, reason="error", error=exc)
        route = self.accept(decision, prep, cfg)
        if route is None:
            return self._run_original(args, kwargs, prep, reason=self._rejection_reason(decision, prep), decision=decision)
        decision = replace(decision, route=route)
        self.record(mode="live", decided_by="jev", prep=prep, decision=decision, jev_route=route)
        return self.emit(route, decision, state)

    def _shadow_sync(self, prep, args, kwargs, cfg, state, config) -> Any:
        future = _shadow_executor().submit(self.decide, prep.request, cfg) if prep.request else None
        start = time.perf_counter()
        result = self.fn(*args, **kwargs)
        original_ms = (time.perf_counter() - start) * 1000
        decision, error = None, None
        if future is not None:
            try:
                decision = future.result(timeout=60)
            except Exception as exc:
                error = exc
                self._warn_backend(exc)
        self._finish_shadow(prep, decision, error, result, original_ms, state, config)
        return result

    # ---- async path --------------------------------------------------------------------
    async def _arun_original(self, args, kwargs, prep, *, reason, decision=None, error=None) -> Any:
        start = time.perf_counter()
        result = await self._acall_fn(args, kwargs)
        ms = (time.perf_counter() - start) * 1000
        self.record(mode="live", decided_by="router", prep=prep, reason=reason, decision=decision,
                    original_route=self.route_of(result), original_ms=ms if self.measure_original else None, error=error)
        return result

    async def _acall_fn(self, args: tuple, kwargs: dict) -> Any:
        result = self.fn(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def acall(self, args: tuple, kwargs: dict, config: Any = None) -> Any:
        cfg = self._resolve()
        if cfg.disabled:
            return await self._acall_fn(args, kwargs)
        state = args[0] if args else kwargs.get("state")
        prep = self.prepare(state, cfg)
        if cfg.shadow:
            return await self._shadow_async(prep, args, kwargs, cfg, state, config)
        if prep.request is None:
            return await self._arun_original(args, kwargs, prep, reason=prep.reason)
        try:
            decision = await self.adecide(prep.request, cfg)
        except Exception as exc:
            self._warn_backend(exc)
            return await self._arun_original(args, kwargs, prep, reason="error", error=exc)
        route = self.accept(decision, prep, cfg)
        if route is None:
            return await self._arun_original(args, kwargs, prep, reason=self._rejection_reason(decision, prep), decision=decision)
        decision = replace(decision, route=route)
        self.record(mode="live", decided_by="jev", prep=prep, decision=decision, jev_route=route)
        return self.emit(route, decision, state)

    async def _shadow_async(self, prep, args, kwargs, cfg, state, config) -> Any:
        task = asyncio.ensure_future(self.adecide(prep.request, cfg)) if prep.request else None
        try:
            start = time.perf_counter()
            result = await self._acall_fn(args, kwargs)
            original_ms = (time.perf_counter() - start) * 1000
        except BaseException:
            if task is not None:
                task.cancel()
            raise
        decision, error = None, None
        if task is not None:
            try:
                decision = await task
            except Exception as exc:
                error = exc
                self._warn_backend(exc)
        self._finish_shadow(prep, decision, error, result, original_ms, state, config)
        return result

    # ---- shadow bookkeeping ------------------------------------------------------------
    def _finish_shadow(self, prep, decision, error, result, original_ms, state, config) -> None:
        if self.shadow_sink is not None:
            self.shadow_sink(prep=prep, decision=decision, error=error, state=state, config=config)
            return
        jev_route = prep.label_map.get(decision.route) if decision is not None else None
        reason = prep.reason
        if error is not None:
            reason = "error"
        elif decision is not None and jev_route is None:
            reason = "invalid_choice"
        self.record(mode="shadow", decided_by="router", prep=prep, reason=reason, decision=decision,
                    jev_route=jev_route, original_route=self.route_of(result), original_ms=original_ms, error=error)

    # ---- user-facing helpers -----------------------------------------------------------
    def events(self) -> List[Dict[str, Any]]:
        return get_recorder().events(router=self.name)

    def report(self, **kwargs: Any):
        """Savings / shadow report for this router. kwargs go to build_report()."""
        from .report import build_report

        return build_report(self.events(), router=self.name, **kwargs)

    def __repr__(self) -> str:
        return f"<jev router {self.name!r} kind={self.kind} routes={list(self.routes)}>"
