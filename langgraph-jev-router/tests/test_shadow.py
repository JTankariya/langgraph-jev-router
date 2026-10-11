import time
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from langgraph_jev_router import FakeJev, configure, jev_first

from conftest import ListState, fixed


def test_shadow_never_changes_behaviour_and_scores_agreement():
    fake = FakeJev(decide=lambda text, labels: ("billing", 0.9) if "refund" in text else ("tech", 0.8))
    configure(backend=fake)
    calls = []

    @jev_first(shadow=True)
    def router(state) -> Literal["billing", "tech"]:
        calls.append(1)
        return "billing"

    for text in ["refund please", "refund again", "my screen is broken"]:
        assert router({"messages": [HumanMessage(text)]}) == "billing"
    assert len(calls) == 3 and fake.calls == 3
    events = router.jev.events()
    assert [e["agree"] for e in events] == [True, True, False]
    assert all(e["mode"] == "shadow" for e in events)


def test_shadow_runs_jev_concurrently():
    configure(backend=FakeJev(decide=fixed("billing", 0.9), latency_ms=300))

    @jev_first(shadow=True)
    def router(state) -> Literal["billing", "tech"]:
        time.sleep(0.3)
        return "billing"

    start = time.perf_counter()
    router({"messages": [HumanMessage("x")]})
    assert time.perf_counter() - start < 0.5  # ~0.3s, not 0.6s


def test_shadow_env_switch_overrides_code(monkeypatch):
    configure(backend=FakeJev(decide=fixed("billing", 0.99)))
    calls = []

    @jev_first(shadow=False)
    def router(state) -> Literal["billing", "tech"]:
        calls.append(1)
        return "tech"

    monkeypatch.setenv("JEV_ROUTER_SHADOW", "1")
    assert router({"messages": [HumanMessage("x")]}) == "tech" and calls == [1]


def test_shadow_records_errors_and_skips():
    configure(backend=FakeJev(fail_with=TimeoutError("slow")))

    @jev_first(shadow=True)
    def router(state) -> Literal["billing", "tech"]:
        return "tech"

    router({"messages": [HumanMessage("x")]})
    router({"messages": [HumanMessage("x"), ("ai", "reply")]})
    reasons = [e.get("reason") for e in router.jev.events()]
    assert reasons == ["error", "mid_conversation"]


def test_shadow_inside_graph():
    configure(backend=FakeJev(decide=fixed("tech", 0.9)))

    @jev_first(shadow=True)
    def router(state) -> Literal["billing", "tech"]:
        return "billing"

    g = StateGraph(ListState)
    g.add_node("billing", lambda s: {"messages": ["B"]})
    g.add_node("tech", lambda s: {"messages": ["T"]})
    g.add_conditional_edges(START, router)
    g.add_edge("billing", END)
    g.add_edge("tech", END)
    out = g.compile().invoke({"messages": [HumanMessage("x")]})
    assert out["messages"][-1] == "B"
    assert router.jev.events()[0]["agree"] is False
