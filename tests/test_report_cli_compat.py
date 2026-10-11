import json
import warnings
from typing import Literal

import pytest
from langchain_core.messages import HumanMessage

from langgraph_jev_router import FakeJev, JevConditionalEdge, build_report, configure, jev_first, load_events
from langgraph_jev_router.cli import main

from conftest import fixed
from fake_jev_server import FakeJevServer


def shadow_events(n_agree_high, n_disagree_low, n_agree_low=0):
    events = []
    for _ in range(n_agree_high):
        events.append({"router": "r", "mode": "shadow", "jev_route": "a", "original_route": "a", "agree": True, "jev_confidence": 0.92})
    for _ in range(n_agree_low):
        events.append({"router": "r", "mode": "shadow", "jev_route": "a", "original_route": "a", "agree": True, "jev_confidence": 0.55})
    for _ in range(n_disagree_low):
        events.append({"router": "r", "mode": "shadow", "jev_route": "b", "original_route": "a", "agree": False, "jev_confidence": 0.62})
    return events


def test_recommends_lowest_threshold_meeting_target():
    report = build_report(shadow_events(40, 10), target=0.95).routers[0]
    row_06 = next(r for r in report.rows if r.threshold == 0.6)
    row_07 = next(r for r in report.rows if r.threshold == 0.7)
    assert (row_06.covered, row_06.agreement) == (50, 0.8)
    assert (row_07.covered, row_07.agreement, round(row_07.coverage, 2)) == (40, 1.0, 0.8)
    assert report.recommended_threshold == 0.7
    assert report.disagreements == [("b", "a", 10)]


def test_not_enough_data_and_no_threshold_cases():
    assert "Not enough data" in build_report(shadow_events(5, 0)).routers[0].recommendation
    bad = build_report(shadow_events(0, 30)).routers[0]
    assert bad.recommended_threshold is None and "No threshold reaches" in bad.recommendation


def test_live_savings_numbers():
    events = [{"router": "r", "mode": "live", "decided_by": "jev", "jev_ms": 100, "input_tokens": 1000}] * 8
    events += [{"router": "r", "mode": "live", "decided_by": "router", "reason": "low_confidence",
                "jev_ms": 100, "original_ms": 1100, "input_tokens": 1000}] * 2
    r = build_report(events, llm_cost_per_call=0.01).routers[0]
    assert (r.jev_decided, r.fallback_reasons) == (8, {"low_confidence": 2})
    assert r.time_saved_s == pytest.approx(8.0) and r.llm_cost_saved_usd == pytest.approx(0.08)
    text = build_report(events, llm_cost_per_call=0.01).to_text()
    assert "LLM routing calls avoided" in text and text.isascii()


def test_report_from_real_decisions_and_jsonl(tmp_path, capsys):
    log = tmp_path / "jev.jsonl"
    configure(backend=FakeJev(decide=lambda t, labels: ("billing", 0.9) if "refund" in t else ("tech", 0.65)),
              log_path=str(log))

    @jev_first(shadow=True)
    def router(state) -> Literal["billing", "tech"]:
        return "billing" if "refund" in state["messages"][-1].content else "billing"

    for i in range(25):
        router({"messages": [HumanMessage(f"refund {i}")]})
    for i in range(5):
        router({"messages": [HumanMessage(f"other {i}")]})

    assert len(load_events(str(log))) == 30
    assert main(["report", str(log)]) == 0
    out = capsys.readouterr().out
    assert "<- recommended" in out and "Use threshold 0.70" in out
    assert main(["report", str(log), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["routers"][0]["recommended_threshold"] == 0.7


def test_cli_check_success_and_failure(capsys, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    with FakeJevServer() as srv:
        assert main(["check", "--base-url", srv.url]) == 0
    assert "OK  model=jev-1.13.0  answer=billing" in capsys.readouterr().out
    with FakeJevServer(status=401) as srv:
        assert main(["check", "--base-url", srv.url]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_deprecated_0_1_api_still_works():
    configure(backend=FakeJev(decide=fixed("analytics", 0.9)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        edge = JevConditionalEdge("Which agent?", {"jira_docs": "Jira", "analytics": "Data", "support": "General"})
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert edge.route({"messages": ["Write a SQL query"]}) == "analytics"
    configure(backend=FakeJev(fail_with=RuntimeError("down")))
    assert edge.route({"messages": ["anything else"]}) == "jira_docs"  # falls back, no crash
