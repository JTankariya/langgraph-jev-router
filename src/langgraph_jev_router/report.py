"""Turn decision events into answers a team can act on.

Live mode answers: how many LLM routing calls did Jev replace, and what did that save?
Shadow mode answers: if we switched Jev on, at which threshold, and how often would it agree
with the router we trust today?

Note that shadow "agreement" is measured against your current router, not against ground
truth. Where they disagree, your router may be the one that is wrong; read a sample.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .settings import JEV_PRICE_PER_MILLION_INPUT_TOKENS

DEFAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
REASON_LABELS = {
    "low_confidence": "low confidence",
    "mid_conversation": "mid-conversation (agent/tool spoke last)",
    "no_input": "no user text found",
    "no_routes": "routes unknown",
    "invalid_choice": "unexpected answer",
    "error": "Jev error/unavailable",
    "state_from_error": "state_from raised",
}


@dataclass
class ThresholdRow:
    threshold: float
    covered: int
    total: int
    agreed: int

    @property
    def coverage(self) -> float:
        return self.covered / self.total if self.total else 0.0

    @property
    def agreement(self) -> Optional[float]:
        return self.agreed / self.covered if self.covered else None


@dataclass
class RouterReport:
    router: str
    # live
    live_total: int = 0
    jev_decided: int = 0
    fallback_reasons: Dict[str, int] = field(default_factory=dict)
    jev_median_ms: Optional[float] = None
    router_median_ms: Optional[float] = None
    jev_input_tokens: int = 0
    jev_cost_usd: float = 0.0
    time_saved_s: Optional[float] = None
    llm_cost_saved_usd: Optional[float] = None
    # shadow
    shadow_total: int = 0
    shadow_skipped: Dict[str, int] = field(default_factory=dict)
    shadow_errors: int = 0
    rows: List[ThresholdRow] = field(default_factory=list)
    recommended_threshold: Optional[float] = None
    recommendation: str = ""
    disagreements: List[Tuple[str, str, int]] = field(default_factory=list)


@dataclass
class Report:
    routers: List[RouterReport]
    target: float

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        for r_dict, r in zip(out["routers"], self.routers):
            for row_dict, row in zip(r_dict["rows"], r.rows):
                row_dict["coverage"] = row.coverage
                row_dict["agreement"] = row.agreement
        return out

    def to_text(self) -> str:
        if not self.routers:
            return "No routing decisions recorded yet."
        return "\n\n".join(_format_router(r, self.target) for r in self.routers)

    def __str__(self) -> str:
        return self.to_text()


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return statistics.median(vals) if vals else None


def build_report(
    events: List[Dict[str, Any]],
    *,
    router: Optional[str] = None,
    target: float = 0.95,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    llm_cost_per_call: Optional[float] = None,
    min_samples: int = 20,
) -> Report:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        if router is None or event.get("router") == router:
            grouped[event.get("router", "?")].append(event)
    reports = [
        _router_report(name, evs, target, tuple(thresholds), llm_cost_per_call, min_samples)
        for name, evs in grouped.items()
    ]
    return Report(routers=reports, target=target)


def _router_report(name, events, target, thresholds, llm_cost_per_call, min_samples) -> RouterReport:
    rep = RouterReport(router=name)
    live = [e for e in events if e.get("mode") == "live"]
    shadow = [e for e in events if e.get("mode") == "shadow"]

    # ---- live --------------------------------------------------------------------------
    rep.live_total = len(live)
    jev_live = [e for e in live if e.get("decided_by") == "jev"]
    rep.jev_decided = len(jev_live)
    rep.fallback_reasons = dict(Counter(e.get("reason", "unknown") for e in live if e.get("decided_by") != "jev"))
    rep.jev_median_ms = _median(e.get("jev_ms") for e in events if not e.get("cached"))
    rep.router_median_ms = _median(e.get("original_ms") for e in events)
    rep.jev_input_tokens = sum(int(e.get("input_tokens", 0) or 0) for e in events if not e.get("cached"))
    rep.jev_cost_usd = rep.jev_input_tokens * JEV_PRICE_PER_MILLION_INPUT_TOKENS / 1_000_000
    if rep.jev_decided and rep.router_median_ms is not None:
        jev_ms = _median(e.get("jev_ms") for e in jev_live) or 0.0
        rep.time_saved_s = max(rep.router_median_ms - jev_ms, 0.0) * rep.jev_decided / 1000
    if llm_cost_per_call is not None:
        rep.llm_cost_saved_usd = rep.jev_decided * llm_cost_per_call

    # ---- shadow ------------------------------------------------------------------------
    labeled = [e for e in shadow if e.get("original_route") is not None]
    rep.shadow_total = len(labeled)
    rep.shadow_skipped = dict(Counter(e.get("reason") for e in labeled if e.get("jev_route") is None and e.get("reason") != "error"))
    rep.shadow_errors = sum(1 for e in labeled if e.get("reason") == "error")
    answered = [e for e in labeled if e.get("jev_route") is not None]
    for t in thresholds:
        covered = [e for e in answered if float(e.get("jev_confidence", 0)) >= t]
        agreed = sum(1 for e in covered if e.get("agree"))
        rep.rows.append(ThresholdRow(threshold=t, covered=len(covered), total=len(labeled), agreed=agreed))
    rep.disagreements = [
        (jev, yours, n)
        for (jev, yours), n in Counter(
            (e["jev_route"], e["original_route"]) for e in answered if not e.get("agree")
        ).most_common(5)
    ]
    if labeled:
        good = [r for r in rep.rows if r.covered >= min_samples and (r.agreement or 0) >= target]
        if good:
            best = min(good, key=lambda r: r.threshold)
            rep.recommended_threshold = best.threshold
            rep.recommendation = (
                f"Use threshold {best.threshold:.2f}: Jev would decide {best.coverage:.0%} of routing calls "
                f"and agree with your router {best.agreement:.1%} of the time (n={best.covered})."
            )
        elif max((r.covered for r in rep.rows), default=0) < min_samples:
            rep.recommendation = (
                f"Not enough data yet: collect at least {min_samples} shadow decisions that Jev answers "
                "confidently before switching to live mode."
            )
        else:
            rep.recommendation = (
                f"No threshold reaches {target:.0%} agreement yet. Improve the descriptions of the routes in "
                "the disagreement list (node docstrings or routes=...), then re-run shadow mode."
            )
    return rep


def _fmt_ms(ms: Optional[float]) -> str:
    return "n/a" if ms is None else (f"{ms:,.0f} ms" if ms >= 10 else f"{ms:.1f} ms")


def _format_router(r: RouterReport, target: float) -> str:
    lines = [f"Jev router report: {r.router}", "-" * (19 + len(r.router))]
    if r.live_total:
        fallbacks = r.live_total - r.jev_decided
        lines.append(f"LIVE   {r.live_total} routing decisions")
        lines.append(f"  Decided by Jev ........ {r.jev_decided:>5} ({r.jev_decided / r.live_total:.0%})   <- LLM routing calls avoided")
        lines.append(f"  Sent to your router ... {fallbacks:>5} ({fallbacks / r.live_total:.0%})")
        for reason, n in sorted(r.fallback_reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {REASON_LABELS.get(reason, reason)}: {n}")
        yours = _fmt_ms(r.router_median_ms) if r.router_median_ms is not None else "n/a (none recorded)"
        lines.append(f"  Median latency: Jev {_fmt_ms(r.jev_median_ms)}  vs  your router {yours}")
        if r.time_saved_s is not None:
            saved = f"{r.time_saved_s * 1000:,.0f} ms" if r.time_saved_s < 1 else f"{r.time_saved_s:,.1f} s"
            lines.append(f"  Estimated routing time saved: {saved}")
        cost = f"  Jev cost (est.): ${r.jev_cost_usd:,.6f}"
        if r.llm_cost_saved_usd is not None:
            cost += f"   LLM routing cost avoided (est.): ${r.llm_cost_saved_usd:,.4f}"
        else:
            cost += "   (pass llm_cost_per_call to estimate LLM spend avoided)"
        lines.append(cost)
    if r.shadow_total:
        if r.live_total:
            lines.append("")
        lines.append(f"SHADOW {r.shadow_total} decisions compared (your router decided; Jev only observed)")
        if r.shadow_skipped:
            skipped = ", ".join(f"{REASON_LABELS.get(k, k)}: {v}" for k, v in r.shadow_skipped.items())
            lines.append(f"  Jev stepped aside on: {skipped}")
        if r.shadow_errors:
            lines.append(f"  Jev errors: {r.shadow_errors}")
        lines.append("  Threshold | Jev would decide | Agrees with your router")
        for row in r.rows:
            agree = "    -" if row.agreement is None else f"{row.agreement:6.1%}"
            mark = "  <- recommended" if r.recommended_threshold == row.threshold else ""
            lines.append(f"     {row.threshold:.2f}  |  {row.coverage:6.1%} (n={row.covered:<4}) |  {agree}{mark}")
        lines.append(f"  {r.recommendation}")
        if r.disagreements:
            lines.append("  Most common disagreements (fix these descriptions first):")
            for jev, yours, n in r.disagreements:
                lines.append(f"      Jev said '{jev}', your router said '{yours}': {n}x")
        lines.append(f"  (Agreement target {target:.0%}; measured against your router, not ground truth.)")
    return "\n".join(lines)
