"""Routing you can't edit: a prebuilt langgraph-supervisor graph.

install_shortcut() adds a Jev node in front of the supervisor. Confident -> straight to the
agent, skipping the supervisor's routing LLM call. Unsure -> the supervisor decides as before.

Needs:  pip install langgraph-supervisor   (included in the [dev] extra)
Run:    python examples/04_prebuilt_supervisor.py
"""

import sys

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, MessagesState, StateGraph

import _demo
from langgraph_jev_router import configure, install_shortcut

try:
    from langgraph_supervisor import create_supervisor
except ImportError:
    sys.exit("This example needs langgraph-supervisor: pip install langgraph-supervisor")

configure(backend=_demo.pick_backend(rules={
    "billing_agent": _demo.FAKE_RULES["billing"],
    "tech_agent": _demo.FAKE_RULES["tech"],
}))


class CountingModel(GenericFakeChatModel):
    """Stand-in for the supervisor's LLM. Counts calls; always hands to billing, then finishes."""

    calls: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        object.__setattr__(self, "calls", self.calls + 1)
        last = messages[-1]
        if getattr(last, "type", "") == "human":
            msg = AIMessage("", tool_calls=[{"name": "transfer_to_billing_agent", "args": {}, "id": f"c{self.calls}"}])
        else:
            msg = AIMessage("All done - anything else?")
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(generations=[ChatGeneration(message=msg)])


def agent(name):
    g = StateGraph(MessagesState)
    g.add_node("run", lambda s: {"messages": [AIMessage(f"[{name}] handled", name=name)]})
    g.add_edge(START, "run")
    return g.compile(name=name)


def build(shadow):
    model = CountingModel(messages=iter([]))
    workflow = create_supervisor([agent("billing_agent"), agent("tech_agent")], model=model)
    handle = install_shortcut(workflow, before="supervisor", shadow=shadow,
                              name="shortcut (shadow)" if shadow else "shortcut (live)", routes={
        "billing_agent": "Charges, refunds, invoices, plans and payment methods.",
        "tech_agent": "Errors, crashes, outages and anything technically broken.",
    })
    return workflow.compile(), model, handle


if __name__ == "__main__":
    tickets = ["Please refund the duplicate charge from March.", "Our invoice lists the wrong company name.",
               "The app shows an error when saving settings.", "I need help."]
    for shadow in (False, True):
        app, model, handle = build(shadow)
        print(f"--- {'SHADOW' if shadow else 'LIVE'} mode ---")
        for text in tickets:
            out = app.invoke({"messages": [HumanMessage(text)]})
            agents = [m.name for m in out["messages"] if getattr(m, "name", None) in ("billing_agent", "tech_agent")]
            print(f"  {text[:50]:<52} -> {agents[0] if agents else '-'}")
        print(f"  Supervisor LLM calls: {model.calls} (without the shortcut: {2 * len(tickets)})\n")
        print(handle.report(), "\n")
