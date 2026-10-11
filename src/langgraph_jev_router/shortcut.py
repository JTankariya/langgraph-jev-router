"""For routing you can't (or don't want to) edit: prebuilt supervisors, compiled subgraphs,
an LLM node whose decision lives inside a tool call.

``install_shortcut(builder, before="supervisor")`` adds one node in front of ``supervisor``:

    START -> jev_shortcut --(Jev confident)--> chosen agent
                          \\-(otherwise)-------> supervisor  (unchanged, decides as before)

Only the entry edge (START -> before) is rewired; edges that loop back to ``before`` after an
agent replies are left alone, so "what next / are we done?" stays with your supervisor.

Shadow mode needs to know what the supervisor actually chose. For that, a tiny pass-through
observer node is placed on each destination's outgoing edges; it records which agent ran.
Observers are added only when shadow mode is on at install time and do nothing else.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

from langchain_core.runnables import RunnableConfig

from ._util import END, logger, sha12, warn_once
from .discovery import node_description
from .engine import RouterEngine
from .extract import last_human_text
from .settings import env_flag, get_settings
from .types import Decision

START = "__start__"
_PENDING_MAX = 2_000


def _config_thread(config: Any) -> str:
    try:
        return str((config or {}).get("configurable", {}).get("thread_id") or "")
    except Exception:
        return ""


class _Pending:
    """Shadow decisions waiting to see which agent the supervisor picks."""

    def __init__(self) -> None:
        self._data: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(state: Any, config: Any) -> Optional[str]:
        text = last_human_text(state)
        return None if text is None else f"{_config_thread(config)}:{sha12(text)}"

    def put(self, key: str, value: tuple) -> None:
        with self._lock:
            self._data[key] = value
            while len(self._data) > _PENDING_MAX:
                self._data.popitem(last=False)

    def pop(self, key: Optional[str]) -> Optional[tuple]:
        if key is None:
            return None
        with self._lock:
            return self._data.pop(key, None)


def install_shortcut(
    builder: Any,
    before: str,
    *,
    routes: Union[None, Sequence[str], Mapping[str, Optional[str]]] = None,
    threshold: Optional[float] = None,
    shadow: Optional[bool] = None,
    question: Optional[str] = None,
    state_from: Optional[Callable[[Any], Any]] = None,
    model: Optional[str] = None,
    include_end: bool = False,
    backend: Any = None,
    node_name: str = "jev_shortcut",
    name: Optional[str] = None,
) -> RouterEngine:
    """Put a Jev shortcut in front of node ``before``. Call before ``compile()``.

    Destinations default to wherever ``before`` can go (e.g. a prebuilt supervisor's agents).
    Returns the router handle: ``handle.report()`` prints savings / shadow results.
    """
    nodes = getattr(builder, "nodes", {})
    if before not in nodes:
        raise ValueError(f"install_shortcut: no node named {before!r} in this graph (have: {', '.join(nodes)})")
    if node_name in nodes:
        raise ValueError(f"install_shortcut: a node named {node_name!r} already exists; pass node_name=...")

    # ---- destinations -------------------------------------------------------------------
    if routes is None:
        dests = list(getattr(nodes[before], "ends", ()) or ())
        for branch in getattr(builder, "branches", {}).get(before, {}).values():
            dests.extend((getattr(branch, "ends", None) or {}).values())
        dests = [d for d in dict.fromkeys(dests) if d != before and (include_end or d != END)]
        if not dests:
            raise ValueError(
                f"install_shortcut: can't tell where {before!r} routes to. Pass routes=[...] "
                "(or {name: description})."
            )
        route_map: Dict[str, Optional[str]] = {d: node_description(builder, d) for d in dests}
    elif isinstance(routes, Mapping):
        route_map = {k: (v or node_description(builder, k)) for k, v in routes.items()}
    else:
        route_map = {r: node_description(builder, r) for r in routes}
    missing = [r for r in route_map if r != END and r not in nodes]
    if missing:
        raise ValueError(f"install_shortcut: routes not found as nodes: {', '.join(missing)}")
    undescribed = [r for r, d in route_map.items() if not d and r != END]
    if undescribed:
        logger.info(
            "jev: shortcut routes without descriptions (%s) will be judged by name. "
            "Pass routes={'agent': 'what it handles', ...} for better accuracy.",
            ", ".join(undescribed),
        )

    # ---- engine: "your router" here simply means: hand over to `before` ----------------
    from langgraph.types import Command

    def go_to_before(state: Any) -> Any:
        return Command(goto=before)

    engine = RouterEngine(
        go_to_before,
        name=name or f"shortcut->{before}",
        routes=route_map,
        threshold=threshold,
        shadow=shadow,
        question=question,
        state_from=state_from,
        model=model,
        include_end=include_end,
        backend=backend,
        kind="command",
    )

    pending = _Pending()

    def sink(*, prep, decision: Optional[Decision], error, state, config) -> None:
        key = pending.key(state, config)
        if key is None:  # no user message to match on later; this decision can't be scored
            return
        if decision is None or error is not None:
            reason = "error" if error is not None else prep.reason
            pending.put(key, (prep, None, reason, error))
        elif decision.route not in prep.label_map:
            pending.put(key, (prep, None, "invalid_choice", None))
        else:
            pending.put(key, (prep, decision, None, None))

    engine.shadow_sink = sink
    engine.measure_original = False

    def shortcut_node(state: Any, config: RunnableConfig) -> Any:
        return engine.call((state,), {}, config)

    async def ashortcut_node(state: Any, config: RunnableConfig) -> Any:
        return await engine.acall((state,), {}, config)

    shortcut_node.__jev_internal__ = True  # type: ignore[attr-defined]

    from langchain_core.runnables import RunnableLambda

    builder.add_node(
        node_name,
        RunnableLambda(shortcut_node, afunc=ashortcut_node, name=node_name),
        destinations=tuple(dict.fromkeys([*route_map, before])),
    )

    # ---- rewire the entry edge -----------------------------------------------------------
    edges = getattr(builder, "edges", None)
    if edges is None or (START, before) not in edges:
        raise ValueError(
            f"install_shortcut: expected an edge START -> {before!r}. If your entry point is conditional, "
            f"add the shortcut yourself: builder.add_edge(START, {node_name!r})."
        )
    edges.discard((START, before))
    builder.add_edge(START, node_name)

    # ---- shadow observers ----------------------------------------------------------------
    shadow_on = env_flag("JEV_ROUTER_SHADOW") or (shadow if shadow is not None else get_settings().shadow)
    if shadow_on:
        _install_observers(builder, engine, pending, list(route_map), node_name)
    return engine


def _install_observers(builder: Any, engine: RouterEngine, pending: _Pending, dests, node_name: str) -> None:
    for dest in dests:
        if dest == END:
            continue
        outgoing = [(a, b) for (a, b) in list(builder.edges) if a == dest]
        if not outgoing:
            warn_once(
                f"observe:{engine.name}:{dest}",
                f"jev: shadow mode can't observe '{dest}' (it has no plain outgoing edge), so decisions "
                f"routed there won't be scored.",
            )
            continue
        observer = f"{node_name}__seen__{dest}"

        def make_observer(chosen: str) -> Callable[..., Any]:
            def observe(state: Any, config: RunnableConfig) -> Dict[str, Any]:
                item = pending.pop(pending.key(state, config))
                if item is not None:
                    prep, decision, reason, error = item
                    engine.record(mode="shadow", decided_by="router", prep=prep, reason=reason,
                                  decision=decision, original_route=chosen, error=error)
                return {}

            observe.__jev_internal__ = True  # type: ignore[attr-defined]
            return observe

        builder.add_node(observer, make_observer(dest))
        for edge in outgoing:
            builder.edges.discard(edge)
            builder.add_edge(observer, edge[1])
        builder.add_edge(dest, observer)
