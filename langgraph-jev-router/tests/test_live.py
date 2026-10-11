import logging
from typing import Literal

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from langgraph_jev_router import FakeJev, attach_routes, configure, jev_first

from conftest import ListState, fixed


def build(router):
    g = StateGraph(ListState)
    g.add_node("billing", lambda s: {"messages": [AIMessage("billing")]})
    g.add_node("tech", lambda s: {"messages": [AIMessage("tech")]})
    g.add_conditional_edges(START, router)
    g.add_edge("billing", END)
    g.add_edge("tech", END)
    return attach_routes(g).compile()


def run(graph, text):
    return graph.invoke({"messages": [HumanMessage(text)]})["messages"][-1].content


def make_router(calls, answer="tech", **options):
    @jev_first(**options)
    def router(state) -> Literal["billing", "tech"]:
        calls.append(state)
        return answer

    return router


def test_confident_jev_skips_your_router():
    calls = []
    configure(backend=FakeJev(decide=fixed("billing", 0.93)))
    router = make_router(calls)
    assert run(build(router), "refund me") == "billing"
    assert calls == []
    (event,) = router.jev.events()
    assert event["decided_by"] == "jev" and event["jev_route"] == "billing" and event["mode"] == "live"


def test_low_confidence_falls_back():
    calls = []
    configure(backend=FakeJev(decide=fixed("billing", 0.4)))
    router = make_router(calls)
    assert run(build(router), "hmm") == "tech"
    assert len(calls) == 1
    (event,) = router.jev.events()
    assert event["reason"] == "low_confidence" and event["original_route"] == "tech"


def test_threshold_is_configurable_per_router():
    calls = []
    configure(backend=FakeJev(decide=fixed("billing", 0.6)))
    router = make_router(calls, threshold=0.5)
    assert run(build(router), "x") == "billing" and calls == []


def test_invalid_answer_falls_back():
    calls = []
    configure(backend=FakeJev(decide=fixed("not_a_route", 0.99)))
    router = make_router(calls)
    assert run(build(router), "x") == "tech"
    assert router.jev.events()[0]["reason"] == "invalid_choice"


def test_backend_error_falls_back_and_warns_once(caplog):
    calls = []
    configure(backend=FakeJev(fail_with=ConnectionError("down")))
    router = make_router(calls)
    graph = build(router)
    with caplog.at_level(logging.WARNING):
        assert run(graph, "a") == "tech"
        assert run(graph, "b") == "tech"
    assert len(calls) == 2
    assert caplog.text.count("Jev is unavailable") == 1
    assert all(e["reason"] == "error" for e in router.jev.events())


def test_missing_api_key_falls_back(monkeypatch):
    """Real backend, no key: the SDK refuses before sending anything; the graph keeps working."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    calls = []
    router = make_router(calls)
    assert run(build(router), "refund") == "tech"
    assert router.jev.events()[0]["reason"] == "error"
    assert "TypeSafe" in router.jev.events()[0]["error"]


def test_mid_conversation_goes_to_your_router():
    calls = []
    fake = FakeJev(decide=fixed("billing", 0.99))
    configure(backend=fake)
    router = make_router(calls)
    graph = build(router)
    out = graph.invoke({"messages": [HumanMessage("refund"), AIMessage("previous agent reply")]})
    assert out["messages"][-1].content == "tech"
    assert fake.calls == 0 and len(calls) == 1
    assert router.jev.events()[0]["reason"] == "mid_conversation"


def test_kill_switch_env(monkeypatch):
    calls = []
    fake = FakeJev(decide=fixed("billing", 0.99))
    configure(backend=fake)
    router = make_router(calls)
    monkeypatch.setenv("JEV_ROUTER_DISABLED", "1")
    assert run(build(router), "refund") == "tech"
    assert fake.calls == 0 and router.jev.events() == []


def test_no_routes_warns_and_falls_back(caplog):
    configure(backend=FakeJev(decide=fixed("billing", 0.99)))

    @jev_first
    def router(state):
        return "tech"

    with caplog.at_level(logging.WARNING):
        assert router({"messages": [HumanMessage("x")]}) == "tech"
    assert "has no known routes" in caplog.text


def test_cache_avoids_repeat_calls():
    fake = FakeJev(decide=fixed("billing", 0.95))
    configure(backend=fake)
    router = make_router([])
    graph = build(router)
    for _ in range(3):
        run(graph, "same message")
    assert fake.calls == 1
    assert [e.get("cached", False) for e in router.jev.events()] == [False, True, True]


def test_request_shape_sent_to_jev():
    fake = FakeJev(decide=fixed("billing", 0.95))
    configure(backend=fake, model="jev-test-1")
    router = make_router([])
    run(build(router), "hello")
    req = fake.requests[0]
    assert req.state == {"request": "hello"} and req.model == "jev-test-1"
    assert req.labels == ["billing", "tech"] and "`request`" in req.question


def test_custom_state_from_and_question():
    fake = FakeJev(decide=fixed("tech", 0.95))
    configure(backend=fake)

    @jev_first(state_from=lambda s: {"ticket": s["messages"][-1].content, "tier": "gold"},
               question="Which queue fits `ticket` for a `tier` customer?")
    def router(state) -> Literal["billing", "tech"]:
        return "billing"

    assert run(build(router), "screen flickers") == "tech"
    assert fake.requests[0].state == {"ticket": "screen flickers", "tier": "gold"}


def test_router_receives_config_when_it_asks_for_it():
    seen = {}
    configure(backend=FakeJev(decide=fixed("billing", 0.1)))

    @jev_first
    def router(state, config) -> Literal["billing", "tech"]:
        seen["config"] = config
        return "tech"

    build(router).invoke({"messages": [HumanMessage("x")]}, {"configurable": {"thread_id": "t1"}})
    assert seen["config"]["configurable"]["thread_id"] == "t1"


def test_inputs_not_logged_by_default_but_can_be(tmp_path):
    configure(backend=FakeJev(decide=fixed("billing", 0.95)), log_path=str(tmp_path / "log.jsonl"))
    router = make_router([])
    run(build(router), "secret customer text")
    event = router.jev.events()[0]
    assert "input" not in event and len(event["input_hash"]) == 12
    assert "secret" not in (tmp_path / "log.jsonl").read_text()

    configure(log_inputs=True)
    run(build(router), "visible text")
    assert "visible text" in router.jev.events()[-1]["input"]


def test_bad_settings_rejected():
    with pytest.raises(TypeError):
        configure(treshold=0.5)
    with pytest.raises(ValueError):
        configure(threshold=2)
