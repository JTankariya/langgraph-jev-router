"""Zero-config route discovery.

Route names come from (first match wins):
  1. ``routes=`` you pass explicitly
  2. the graph itself, via ``attach_routes(builder)``: conditional-edge destinations, or the
     ``destinations=`` / ``Command[Literal[...]]`` of a node
  3. the router's return annotation: ``-> Literal["a", "b"]`` or ``-> Command[Literal["a", "b"]]``

Route descriptions come from ``routes=`` or the destination node's docstring. A route with
no description is judged by its name alone, which works but is less accurate.
"""

from __future__ import annotations

import functools
import inspect
import re
import typing
from typing import Any, Dict, List, Literal, Optional, Union

from ._util import END, logger, warn_once

_MAX_DESCRIPTION = 400


def _command_type():
    try:
        from langgraph.types import Command

        return Command
    except Exception:  # pragma: no cover
        return None


def _literal_values(tp: Any) -> Optional[List[str]]:
    origin = typing.get_origin(tp)
    if origin is Literal:
        return [str(v) for v in typing.get_args(tp)]
    command = _command_type()
    if command is not None and (tp is command or origin is command):
        args = typing.get_args(tp)
        return _literal_values(args[0]) if args else None
    if origin is Union or (origin is not None and str(origin) == "<class 'types.UnionType'>"):
        values: List[str] = []
        for arg in typing.get_args(tp):
            found = _literal_values(arg)
            if found:
                values.extend(found)
        return values or None
    return None


def _return_hint(fn: Any) -> Any:
    try:
        return typing.get_type_hints(fn).get("return")
    except Exception:
        return getattr(fn, "__annotations__", {}).get("return")


def routes_from_annotation(fn: Any) -> Optional[List[str]]:
    hint = _return_hint(fn)
    if hint is None or isinstance(hint, str):
        return None
    values = _literal_values(hint)
    return list(dict.fromkeys(values)) if values else None


def annotation_is_command(fn: Any) -> bool:
    hint = _return_hint(fn)
    if hint is None:
        return False
    if isinstance(hint, str):
        return hint.strip().startswith("Command")
    command = _command_type()
    return command is not None and (hint is command or typing.get_origin(hint) is command)


def _clean_doc(doc: str) -> Optional[str]:
    first_paragraph = doc.strip().split("\n\n")[0]
    text = re.sub(r"\s+", " ", first_paragraph).strip()
    return text[:_MAX_DESCRIPTION] or None


def node_description(builder: Any, node: str) -> Optional[str]:
    """Docstring of the function behind a graph node, if it has one."""
    spec = getattr(builder, "nodes", {}).get(node)
    if spec is None:
        return None
    runnable = getattr(spec, "runnable", None)
    for candidate in (getattr(runnable, "func", None), getattr(runnable, "afunc", None)):
        while isinstance(candidate, functools.partial):
            candidate = next((a for a in candidate.args if callable(a)), candidate.func)
        if candidate is not None and (inspect.isfunction(candidate) or inspect.ismethod(candidate)):
            if getattr(candidate, "__jev_internal__", False):
                return None
            doc = candidate.__doc__
            if doc:
                return _clean_doc(doc)
    return None


def find_engine(obj: Any, _depth: int = 0) -> Any:
    """Find the @jev_first engine behind whatever LangGraph wrapped our function in."""
    if obj is None or _depth > 4:
        return None
    engine = getattr(obj, "__jev_router__", None)
    if engine is not None and hasattr(engine, "set_attached_routes"):
        return engine
    if isinstance(obj, functools.partial):
        for inner in (obj.func, *obj.args):
            found = find_engine(inner, _depth + 1)
            if found is not None:
                return found
        return None
    for attr in ("func", "afunc"):
        inner = getattr(obj, attr, None)
        if inner is not None and inner is not obj:
            found = find_engine(inner, _depth + 1)
            if found is not None:
                return found
    return None


def attach_routes(builder: Any, *, descriptions: Optional[Dict[str, str]] = None) -> Any:
    """Fill every @jev_first router in this graph with its destinations and their docstrings.

    Call once, before ``compile()``. Returns the builder so you can chain:
    ``graph = attach_routes(builder).compile()``
    """
    descriptions = dict(descriptions or {})
    engines = []

    for source, branches in getattr(builder, "branches", {}).items():
        for branch in branches.values():
            engine = find_engine(getattr(branch, "path", None))
            if engine is None:
                continue
            ends = getattr(branch, "ends", None)
            if not ends:
                warn_once(
                    f"no_ends:{engine.name}",
                    f"jev: can't see the destinations of router '{engine.name}' (edge from '{source}'). "
                    "Pass a list/dict as path_map to add_conditional_edges, annotate the router with "
                    "-> Literal[...], or give @jev_first(routes=...). Your router handles it until then.",
                )
                continue
            engine.set_attached_routes(
                {key: descriptions.get(key) or node_description(builder, node) for key, node in ends.items()}
            )
            engines.append(engine)

    for node, spec in getattr(builder, "nodes", {}).items():
        engine = find_engine(getattr(spec, "runnable", None))
        if engine is None:
            continue
        ends = tuple(getattr(spec, "ends", ()) or ())
        if not ends:
            warn_once(
                f"no_ends:{engine.name}",
                f"jev: node '{node}' uses @jev_first but its destinations are unknown. Annotate it "
                "-> Command[Literal[...]] or add_node(..., destinations=(...)).",
            )
            continue
        engine.set_attached_routes({dest: descriptions.get(dest) or node_description(builder, dest) for dest in ends})
        engines.append(engine)

    if not engines:
        warn_once("attach_none", "jev: attach_routes() found no @jev_first routers in this graph.")
    for engine in engines:
        undescribed = [name for name, desc in engine.routes.items() if not desc and name != END]
        if undescribed:
            logger.info(
                "jev: router '%s' has routes without descriptions (%s); Jev will judge them by name. "
                "Add a docstring to those nodes or pass routes={...} for better accuracy.",
                engine.name,
                ", ".join(undescribed),
            )
    return builder
