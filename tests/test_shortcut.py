import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from langgraph_jev_router import FakeJev, configure, install_shortcut

from conftest import fixed


# ---------------------------------------------------------------- hand-built supervisor graph
def handmade(llm_calls):
    def supervisor(state) -> Command:
        llm_calls.append(1)  # stands in for an expensive LLM call
        last = state["messages"][-1]
        if getattr(last, "name", None):  # an agent already answered
            return Command(goto=END)
        return Command(goto="tech_agent")

    def billing_agent(state):
        """Handles charges, refunds and invoices."""
        return {"messages": [AIMessage("refunded", name="billing_agent")]}

    def tech_agent(state):
        """Handles errors, crashes and outages."""
        return {"messages": [AIMessage("restarted", name="tech_agent")]}

    g = StateGraph(MessagesState)
    g.add_node("supervisor", supervisor, destinations=("billing_agent", "tech_agent", END))
    g.add_node("billing_agent", billing_agent)
    g.add_node("tech_agent", tech_agent)
    g.add_edge(START, "supervisor")
    g.add_edge("billing_agent", "supervisor")
    g.add_edge("tech_agent", "supervisor")
    return g


def test_shortcut_live_skips_first_supervisor_call():
    configure(backend=FakeJev(decide=fixed("billing_agent", 0.95)))
    calls = []
    g = handmade(calls)
    handle = install_shortcut(g, before="supervisor")
    assert handle.routes == {"billing_agent": "Handles charges, refunds and invoices.",
                             "tech_agent": "Handles errors, crashes and outages."}
    out = g.compile().invoke({"messages": [HumanMessage("refund my charge")]})
    assert [m.content for m in out["messages"]] == ["refund my charge", "refunded"]
    assert len(calls) == 1  # only the "are we done?" call; the routing call was skipped


def test_shortcut_unsure_hands_over_to_supervisor():
    configure(backend=FakeJev(decide=fixed("billing_agent", 0.3)))
    calls = []
    g = handmade(calls)
    handle = install_shortcut(g, before="supervisor")
    out = g.compile().invoke({"messages": [HumanMessage("hmm")]})
    assert out["messages"][-1].content == "restarted" and len(calls) == 2
    assert handle.events()[0]["reason"] == "low_confidence"


def test_shortcut_shadow_scores_what_supervisor_chose():
    configure(backend=FakeJev(decide=lambda text, labels: ("billing_agent", 0.9)))
    calls = []
    g = handmade(calls)
    handle = install_shortcut(g, before="supervisor", shadow=True)
    app = g.compile()
    out = app.invoke({"messages": [HumanMessage("refund")]}, {"configurable": {"thread_id": "1"}})
    assert out["messages"][-1].content == "restarted"  # behaviour unchanged: supervisor picked tech
    assert len(calls) == 2
    (event,) = handle.events()
    assert (event["mode"], event["jev_route"], event["original_route"], event["agree"]) == (
        "shadow", "billing_agent", "tech_agent", False)


def test_shortcut_validation_errors():
    g = handmade([])
    with pytest.raises(ValueError, match="no node named"):
        install_shortcut(g, before="nope")
    with pytest.raises(ValueError, match="not found as nodes"):
        install_shortcut(g, before="supervisor", routes=["ghost"])

    g2 = StateGraph(MessagesState)
    g2.add_node("supervisor", lambda s: {})
    g2.add_edge(START, "supervisor")
    with pytest.raises(ValueError, match="can't tell where"):
        install_shortcut(g2, before="supervisor")


# ------------------------------------------------------------ real prebuilt: create_supervisor
langgraph_supervisor = pytest.importorskip("langgraph_supervisor")


class CountingFakeModel(GenericFakeChatModel):
    calls: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        object.__setattr__(self, "calls", self.calls + 1)
        return super()._generate(*args, **kwargs)


def _agent(name, reply):
    g = StateGraph(MessagesState)
    g.add_node("run", lambda s: {"messages": [AIMessage(reply, name=name)]})
    g.add_edge(START, "run")
    return g.compile(name=name)


def _prebuilt(responses):
    model = CountingFakeModel(messages=iter(responses))
    workflow = langgraph_supervisor.create_supervisor(
        [_agent("billing_agent", "Refund issued"), _agent("tech_agent", "Service restarted")], model=model
    )
    return workflow, model


def test_prebuilt_supervisor_live():
    configure(backend=FakeJev({"billing_agent": ["charge", "refund"], "tech_agent": ["crash", "error"]}))
    # Without the shortcut this run needs 2 supervisor LLM calls (route, then finish).
    workflow, model = _prebuilt([AIMessage("All sorted.")])
    install_shortcut(workflow, before="supervisor", routes={
        "billing_agent": "Charges, refunds, invoices",
        "tech_agent": "Errors, crashes, outages",
    })
    out = workflow.compile().invoke({"messages": [HumanMessage("I was charged twice, refund one charge")]})
    texts = [m.content for m in out["messages"]]
    assert "Refund issued" in texts and texts[-1] == "All sorted."
    assert model.calls == 1


def test_prebuilt_supervisor_shadow():
    configure(backend=FakeJev({"billing_agent": ["charge", "refund"], "tech_agent": ["crash", "error"]}))
    responses = [
        AIMessage("", tool_calls=[{"name": "transfer_to_billing_agent", "args": {}, "id": "t1"}]),
        AIMessage("All sorted."),
    ]
    workflow, model = _prebuilt(responses)
    handle = install_shortcut(workflow, before="supervisor", shadow=True)
    out = workflow.compile().invoke({"messages": [HumanMessage("refund my double charge")]})
    assert out["messages"][-1].content == "All sorted." and model.calls == 2
    (event,) = handle.events()
    assert event["original_route"] == "billing_agent" and event["agree"] is True
