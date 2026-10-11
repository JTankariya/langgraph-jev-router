"""A supervisor node that returns Command(goto=...) - the other common routing shape.

Run:  python examples/03_command_node.py
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

import _demo
from langgraph_jev_router import attach_routes, configure, jev_first

configure(backend=_demo.pick_backend())


@jev_first(on_jev=lambda state, d: {"messages": [AIMessage(f"(routed by Jev to {d.route}, {d.confidence:.0%})")]})
def supervisor(state: MessagesState) -> Command[Literal["billing", "tech", "account", "general"]]:
    """Your LLM supervisor (simulated): writes a note and hands off."""
    route = _demo.fake_llm_router(state["messages"][-1].content)
    return Command(goto=route, update={"messages": [AIMessage(f"(LLM supervisor chose {route})")]})


def make_agent(name, doc):
    def agent(state):
        return {"messages": [AIMessage(f"[{name} agent] handling it")]}

    agent.__doc__ = doc
    return agent


builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor)
for name, doc in {
    "billing": "Charges, refunds, invoices, plans, pricing and payment methods.",
    "tech": "Errors, crashes, outages, slowness and anything technically broken.",
    "account": "Login, passwords, 2FA, account settings, users and ownership.",
    "general": "Everything else: company questions, feedback, feature ideas, greetings.",
}.items():
    builder.add_node(name, make_agent(name, doc))
    builder.add_edge(name, END)
builder.add_edge(START, "supervisor")
graph = attach_routes(builder).compile()

if __name__ == "__main__":
    for text, _ in _demo.LIVE_SET[:5]:
        out = graph.invoke({"messages": [HumanMessage(text)]})
        print(text)
        for m in out["messages"][1:]:
            print("   ", m.content)
    print()
    print(supervisor.jev.report())
