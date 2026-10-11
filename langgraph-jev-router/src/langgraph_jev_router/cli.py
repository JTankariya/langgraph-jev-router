"""Command line: `jev-router check` and `jev-router report`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional


def _cmd_check(args: argparse.Namespace) -> int:
    from .backends import TypeSafeBackend
    from .settings import get_settings
    from .types import ChoiceRequest

    model = args.model or get_settings().model
    request = ChoiceRequest(
        question="Which team should handle `request`?",
        options=(("billing", "Charges, refunds, invoices, payment methods"), ("tech", "Errors, outages, bugs")),
        state={"request": "I was charged twice for my plan this month, please refund one charge."},
        model=model,
    )
    print(f"Calling Jev ({model}) with one test question...")
    start = time.perf_counter()
    try:
        decision = TypeSafeBackend(base_url=args.base_url).choose(request)
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print("Check TYPESAFE_API_KEY, network access to the TypeSafe API, and that the model name is valid.")
        return 1
    ms = (time.perf_counter() - start) * 1000
    print(f"OK  model={decision.model}  answer={decision.route}  confidence={decision.confidence:.3f}  "
          f"latency={ms:.0f} ms  request_id={decision.request_id}")
    if decision.route != "billing":
        print("Note: expected 'billing' for this message; the call worked, but double-check the model.")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .recorder import load_events
    from .report import build_report

    events = load_events(args.path)
    report = build_report(events, router=args.router, target=args.target, llm_cost_per_call=args.llm_cost_per_call,
                          min_samples=args.min_samples)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.to_text())
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-router", description="langgraph-jev-router tools")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="make one real Jev call to verify key, network and model")
    check.add_argument("--model", default=None)
    check.add_argument("--base-url", default=None, help="override the API base URL")
    check.set_defaults(func=_cmd_check)

    report = sub.add_parser("report", help="summarize a decision log (configure(log_path=...) / JEV_ROUTER_LOG)")
    report.add_argument("path")
    report.add_argument("--router", default=None, help="only this router")
    report.add_argument("--target", type=float, default=0.95, help="agreement target for the recommendation")
    report.add_argument("--llm-cost-per-call", type=float, default=None, help="your LLM router's cost per call, USD")
    report.add_argument("--min-samples", type=int, default=20)
    report.add_argument("--json", action="store_true")
    report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:  # e.g. `jev-router report log.jsonl | head`
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
