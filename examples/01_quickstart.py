"""Quickstart: add Jev to an existing LangGraph router with one decorator.

Run:  python examples/01_quickstart.py
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

import _demo
from langgraph_jev_router import attach_routes, configure, jev_first

configure(backend=_demo.pick_backend())  # in your app: nothing to do, real Jev is the default


# ----- Your existing code. The only change is the decorator line. -------------------------
@jev_first
def route_ticket(state: MessagesState) -> Literal["billing", "tech", "account", "general"]:
    """Your existing LLM-based router (simulated). Now it only runs when Jev is unsure."""
    return _demo.fake_llm_router(state["messages"][-1].content)


def billing(state):
    """Charges, refunds, invoices, plans, pricing and payment methods."""
    return {"messages": [AIMessage("[billing agent] on it")]}


def tech(state):
    """Errors, crashes, outages, slowness and anything technically broken."""
    return {"messages": [AIMessage("[tech agent] on it")]}


def account(state):
    """Login, passwords, 2FA, account settings, users and ownership."""
    return {"messages": [AIMessage("[account agent] on it")]}


def general(state):
    """Everything else: questions about the company, feedback, feature ideas, greetings."""
    return {"messages": [AIMessage("[general agent] on it")]}


builder = StateGraph(MessagesState)
for fn in (billing, tech, account, general):
    builder.add_node(fn.__name__, fn)
    builder.add_edge(fn.__name__, END)
builder.add_conditional_edges(START, route_ticket)
graph = attach_routes(builder).compile()  # optional: uses the node docstrings above as route descriptions
# -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    tickets = _demo.LIVE_SET
    for text, _ in tickets:
        out = graph.invoke({"messages": [HumanMessage(text)]})
        print(f"{_demo.who_decided(route_ticket):>10}  ->  {out['messages'][-1].content:<22}  {text[:60]}")
    print(f"\nLLM router calls: {_demo.llm_calls} of {len(tickets)}\n")
    print(route_ticket.jev.report(llm_cost_per_call=0.002))
