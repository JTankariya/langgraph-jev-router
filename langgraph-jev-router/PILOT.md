# Pilot plan: does this solve a real problem?

Before building more features, find out whether real LangGraph teams install this, understand it
without help, and **keep it switched on**. Two weeks, 3–4 developers, a decision at the end.

## Who to recruit

3–4 developers who already have a LangGraph graph that routes with an LLM, ideally covering the
three supported shapes:

| Developer | Routing shape | Integration they'll use |
|---|---|---|
| A | Conditional-edge function calling an LLM | `@jev_first` |
| B | Supervisor node returning `Command(goto=...)` | `@jev_first` |
| C | Prebuilt `create_supervisor` (or similar code they don't own) | `install_shortcut` |
| D (optional) | Anything else | Find out what breaks |

Developer D matters: their "this didn't fit my graph" is the most valuable feedback you'll get.

## Timeline

**Day 1: setup session, 30 minutes, you watch and stay quiet.** Hand them only the README. Note:

- minutes from `pip install` to their first shadow report,
- lines of their code they changed,
- every question they ask and every place they get stuck (each one is a README or API fix).

They finish with shadow mode on and logging enabled:

```bash
export TYPESAFE_API_KEY=...
export JEV_ROUTER_SHADOW=1
export JEV_ROUTER_LOG=jev_decisions.jsonl
```

**Days 2–5: shadow on real or staging traffic.** They do nothing. Behavior is unchanged.

**Day 5: review, 20 minutes.** Read the report together:

```bash
jev-router report jev_decisions.jsonl --llm-cost-per-call <their cost>
```

If it recommends a threshold, they switch to live with it. If not, look at the disagreement list, improve
two or three route descriptions, and give shadow mode a few more days.

**Days 6–14: live.** Ask them to note any misroute they notice. Check the fallback reasons in the live report.

**Day 14: exit interview, 20 minutes.**

## What to measure

| Metric | Where it comes from | Target that means "it works" |
|---|---|---|
| Setup time to first shadow report | Day 1 observation | ≤ 15 minutes |
| Lines of their code changed | Day 1 observation | ≤ 3 |
| Share of routing Jev handles at ≥ 95% agreement | Shadow report | ≥ 60% |
| Misroutes noticed in live mode | Their notes | Fewer than they tolerate from their LLM router today |
| Still enabled at day 14 | Ask | Yes for at least 2 of 3 |

`jev-router report ... --json` contains no message text by default, so it is usually easy for teams to share.

## Exit interview questions

1. Walk me through the first ten minutes. Where did you hesitate?
2. Was there a moment you didn't trust it? What would have fixed that?
3. Did the report tell you what you needed to decide whether to go live? What was missing?
4. Is it still on? If you turned it off, why?
5. What would you need before using this on a second graph?
6. If this package disappeared tomorrow, what would you do instead?

Question 6 tells you whether you're competing with "nothing" or with something they already like
(for example `langchain-typesafe` middleware, or their own prompt-based classifier).

## Decision rules

- **Go (invest further):** at least 2 of 3 keep it on at day 14, and setup took ≤ 15 minutes without help.
- **Fix, then re-pilot:** they liked the result but setup was painful or confusing. Fix those specific
  points and run it again with one new developer.
- **Rethink:** coverage stays below ~40% even after improving descriptions, or teams turn it off because
  of misroutes. That points at Jev's fit for routing in their domain, not at the package.

## Before piloting on company traffic

Shadow mode sends each new user message to TypeSafe's API. If the pilot runs on internal or customer
data, confirm with your data-classification / security team that sending that content to an external AI
vendor is allowed, or run the pilot on synthetic or staging traffic.

## Collecting feedback

Use the "Pilot feedback" GitHub issue template (`.github/ISSUE_TEMPLATE/pilot_feedback.md`), so every
participant's answers arrive in the same shape.
