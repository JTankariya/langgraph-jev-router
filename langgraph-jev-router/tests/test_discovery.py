import logging
from typing import Literal, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from langgraph_jev_router import attach_routes, jev_first
from langgraph_jev_router._util import make_labels
from langgraph_jev_router.discovery import annotation_is_command, routes_from_annotation

from conftest import ListState


def test_literal_annotation():
    def r(s) -> Literal["a", "b"]: ...
    assert routes_from_annotation(r) == ["a", "b"]


def test_optional_and_command_annotations():
    def r(s) -> Optional[Literal["a", "b"]]: ...
    def c(s) -> Command[Literal["x", "y"]]: ...
    assert routes_from_annotation(r) == ["a", "b"]
    assert routes_from_annotation(c) == ["x", "y"]
    assert annotation_is_command(c) and not annotation_is_command(r)


def test_no_annotation():
    def r(s): ...
    assert routes_from_annotation(r) is None


def test_labels_are_safe_and_unique():
    labels = make_labels(["billing agent", "billing-agent", "__end__", "!!"])
    assert labels == {"billing_agent": "billing agent", "billing-agent": "billing-agent", "end": "__end__", "route_3": "!!"}


def _graph(router, path_map=None):
    def billing(s):
        """Charges, refunds and invoices.

        Second paragraph is ignored."""
        return {"messages": ["b"]}

    def tech(s):
        return {"messages": ["t"]}

    g = StateGraph(ListState)
    g.add_node("billing_node", billing)
    g.add_node("tech_node", tech)
    g.add_conditional_edges(START, router, path_map)
    g.add_edge("billing_node", END)
    g.add_edge("tech_node", END)
    return g


def test_attach_uses_path_map_keys_and_docstrings():
    @jev_first
    def router(s):
        return "pay"

    attach_routes(_graph(router, {"pay": "billing_node", "it": "tech_node"}))
    assert router.jev.routes == {"pay": "Charges, refunds and invoices.", "it": None}


def test_explicit_routes_win_and_get_filled():
    @jev_first(routes={"billing_node": "Custom text", "tech_node": None})
    def router(s) -> Literal["billing_node", "tech_node"]:
        return "tech_node"

    attach_routes(_graph(router))
    assert router.jev.routes == {"billing_node": "Custom text", "tech_node": None}


def test_end_excluded_by_default_included_on_request():
    @jev_first
    def a(s) -> Literal["billing_node", "__end__"]: ...

    @jev_first(include_end=True)
    def b(s) -> Literal["billing_node", "__end__"]: ...

    assert list(a.jev.routes) == ["billing_node"]
    assert list(b.jev.routes) == ["billing_node", "__end__"] and b.jev.routes["__end__"]


def test_command_node_destinations():
    @jev_first
    def sup(s) -> Command[Literal["billing_node", "tech_node"]]:
        return Command(goto="tech_node")

    g = _graph(lambda s: "billing_node", ["billing_node", "tech_node"])
    g.add_node("sup", sup)
    attach_routes(g)
    assert sup.jev.kind == "command"
    assert sup.jev.routes["billing_node"] == "Charges, refunds and invoices."


def test_attach_warns_when_destinations_unknown(caplog):
    @jev_first
    def router(s):
        return "billing_node"

    g = StateGraph(ListState)
    g.add_node("billing_node", lambda s: {"messages": []})
    g.add_conditional_edges(START, router)  # no path_map, no Literal
    with caplog.at_level(logging.WARNING):
        attach_routes(g)
    assert "can't see the destinations" in caplog.text
