import asyncio
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from langgraph_jev_router import FakeJev, attach_routes, configure, jev_first

from conftest import ListState, fixed


def test_async_router_live_and_fallback():
    calls = []

    @jev_first
    async def router(state) -> Literal["billing", "tech"]:
        calls.append(1)
        await asyncio.sleep(0)
        return "tech"

    configure(backend=FakeJev(decide=fixed("billing", 0.95)))
    assert asyncio.run(router({"messages": [HumanMessage("a")]})) == "billing"
    configure(backend=FakeJev(decide=fixed("billing", 0.2)))
    assert asyncio.run(router({"messages": [HumanMessage("b")]})) == "tech"
    assert len(calls) == 1


def test_async_shadow():
    configure(backend=FakeJev(decide=fixed("billing", 0.95), latency_ms=50))

    @jev_first(shadow=True)
    async def router(state) -> Literal["billing", "tech"]:
        return "billing"

    assert asyncio.run(router({"messages": [HumanMessage("a")]})) == "billing"
    assert router.jev.events()[0]["agree"] is True


def test_async_graph_with_sync_router():
    configure(backend=FakeJev(decide=fixed("tech", 0.95)))

    @jev_first
    def router(state) -> Literal["billing", "tech"]:
        return "billing"

    g = StateGraph(ListState)
    g.add_node("billing", lambda s: {"messages": ["B"]})
    g.add_node("tech", lambda s: {"messages": ["T"]})
    g.add_conditional_edges(START, router)
    g.add_edge("billing", END)
    g.add_edge("tech", END)
    out = asyncio.run(g.compile().ainvoke({"messages": [HumanMessage("x")]}))
    assert out["messages"][-1] == "T"


def _command_graph(supervisor):
    def billing(s):
        """Charges and refunds."""
        return {"messages": [AIMessage("billing")]}

    def tech(s):
        """Errors and outages."""
        return {"messages": [AIMessage("tech")]}

    g = StateGraph(ListState)
    g.add_node("supervisor", supervisor)
    g.add_node("billing", billing)
    g.add_node("tech", tech)
    g.add_edge(START, "supervisor")
    g.add_edge("billing", END)
    g.add_edge("tech", END)
    return attach_routes(g).compile()


def test_command_node_live_with_on_jev_update():
    configure(backend=FakeJev(decide=fixed("billing", 0.95)))
    calls = []

    @jev_first(on_jev=lambda state, d: {"messages": [AIMessage(f"[jev -> {d.route} @ {d.confidence:.2f}]")]})
    def supervisor(state) -> Command[Literal["billing", "tech"]]:
        calls.append(1)
        return Command(goto="tech", update={"messages": [AIMessage("llm supervisor")]})

    out = _command_graph(supervisor).invoke({"messages": [HumanMessage("refund")]})
    contents = [m.content for m in out["messages"]]
    assert contents == ["refund", "[jev -> billing @ 0.95]", "billing"]
    assert calls == []
    assert supervisor.jev.routes == {"billing": "Charges and refunds.", "tech": "Errors and outages."}


def test_command_node_fallback_keeps_original_update():
    configure(backend=FakeJev(decide=fixed("billing", 0.3)))

    @jev_first
    def supervisor(state) -> Command[Literal["billing", "tech"]]:
        return Command(goto="tech", update={"messages": [AIMessage("llm supervisor")]})

    out = _command_graph(supervisor).invoke({"messages": [HumanMessage("refund")]})
    assert [m.content for m in out["messages"]] == ["refund", "llm supervisor", "tech"]


def test_command_node_shadow_compares_goto():
    configure(backend=FakeJev(decide=fixed("billing", 0.95)))

    @jev_first(shadow=True)
    def supervisor(state) -> Command[Literal["billing", "tech"]]:
        return Command(goto="tech")

    _command_graph(supervisor).invoke({"messages": [HumanMessage("x")]})
    e = supervisor.jev.events()[0]
    assert (e["jev_route"], e["original_route"], e["agree"]) == ("billing", "tech", False)
