"""The recommended rollout: measure in shadow mode, pick a threshold, then go live.

Shadow mode changes nothing: your router keeps deciding, Jev's answer is only compared.
Run:  python examples/02_shadow_then_live.py
"""

from typing import Literal

from langchain_core.messages import HumanMessage

import _demo
from langgraph_jev_router import configure, jev_first

LOG = "jev_decisions.jsonl"
configure(backend=_demo.pick_backend(), log_path=LOG)


@jev_first(routes={
    "billing": "Charges, refunds, invoices, plans, pricing and payment methods.",
    "tech": "Errors, crashes, outages, slowness and anything technically broken.",
    "account": "Login, passwords, 2FA, account settings, users and ownership.",
    "general": "Everything else: company questions, feedback, feature ideas, greetings.",
})
def route_ticket(state) -> Literal["billing", "tech", "account", "general"]:
    return _demo.fake_llm_router(state["messages"][-1].content)


if __name__ == "__main__":
    print(f"PHASE 1 - shadow mode on {len(_demo.SHADOW_SET)} tickets (your router decides every one)")
    configure(shadow=True)
    for text, _ in _demo.SHADOW_SET:
        route_ticket({"messages": [HumanMessage(text)]})
    shadow_report = route_ticket.jev.report(target=0.95)
    print(shadow_report)

    threshold = shadow_report.routers[0].recommended_threshold
    if threshold is None:
        print("\nNo safe threshold yet - stay in shadow mode and improve route descriptions.")
        raise SystemExit(0)

    print(f"\nPHASE 2 - live mode at threshold {threshold} on {len(_demo.LIVE_SET)} new tickets")
    configure(shadow=False, threshold=threshold)
    calls_before = _demo.llm_calls
    for text, expected in _demo.LIVE_SET:
        route = route_ticket({"messages": [HumanMessage(text)]})
        flag = "ok " if route == expected else "MISS"
        print(f"  {flag} {_demo.who_decided(route_ticket):>10} -> {route:<8} {text[:58]}")
    print(f"\nLLM router calls in live phase: {_demo.llm_calls - calls_before} of {len(_demo.LIVE_SET)}")
    print(f"Every decision was also written to {LOG}. Re-read it any time with:\n  jev-router report {LOG}")
