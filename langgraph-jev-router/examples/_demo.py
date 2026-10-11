"""Shared bits for the examples: a labeled support-ticket dataset, a stand-in "LLM router",
and backend selection (real Jev if TYPESAFE_API_KEY is set, otherwise an offline FakeJev).
"""

import os
import time

from langgraph_jev_router import FakeJev, TypeSafeBackend

ROUTES = ("billing", "tech", "account", "general")

# (message, the route a good LLM router would pick)
TICKETS = [
    ("I was charged twice for my subscription this month, please refund one.", "billing"),
    ("Can I get an invoice for last quarter with our VAT number on it?", "billing"),
    ("Why did my plan price go up without notice?", "billing"),
    ("My card was declined but the payment still shows as pending.", "billing"),
    ("How do I switch from monthly to annual billing?", "billing"),
    ("I cancelled last week but was still charged. I want my money back.", "billing"),
    ("The refund you promised never arrived on my card.", "billing"),
    ("Please update the billing address on our invoices.", "billing"),
    ("Is there a discount for nonprofits on the Pro plan?", "billing"),
    ("You billed us for 12 seats but we only have 9 users.", "billing"),
    ("Do you accept payment by bank transfer?", "billing"),
    ("The dashboard shows a 500 error every time I open reports.", "tech"),
    ("The app crashes as soon as I upload a photo.", "tech"),
    ("CSV export has been timing out since Monday.", "tech"),
    ("Pages are extremely slow to load today, is there an outage?", "tech"),
    ("The API returns an error 429 even though we're under our limit.", "tech"),
    ("Notifications stopped working on Android after the update.", "tech"),
    ("Search results are empty even for documents I know exist, looks like a bug.", "tech"),
    ("The integration with Slack is broken, messages don't come through.", "tech"),
    ("Charts render blank in Safari but fine in Chrome.", "tech"),
    ("Webhook deliveries keep failing with a timeout error.", "tech"),
    ("The mobile app freezes on the login screen after the latest update.", "tech"),
    ("I forgot my password and the reset email never arrives.", "account"),
    ("How do I enable two-factor authentication (2FA)?", "account"),
    ("My account is locked after too many sign in attempts.", "account"),
    ("Please change the email address on my account.", "account"),
    ("I want to delete my account and all my data.", "account"),
    ("Can I transfer account ownership to a colleague?", "account"),
    ("I can't log in with Google sign in anymore.", "account"),
    ("How do I change my username?", "account"),
    ("Add a new admin user to our workspace account please.", "account"),
    ("I lost my 2FA device and can't get into my account.", "account"),
    ("What are your support hours?", "general"),
    ("Just wanted to say the new onboarding is fantastic, thanks!", "general"),
    ("Would love a dark mode for the dashboard.", "general"),
    ("Do you have an office in Europe?", "general"),
    ("Is there a feature roadmap I can look at?", "general"),
    ("Hello, who should I talk to about a partnership?", "general"),
    ("Can you recommend best practices for organizing projects?", "general"),
    ("Thanks for the quick help yesterday!", "general"),
    # Harder ones: mixed signals or vague wording.
    ("The payment page crashes when I try to update my card.", "tech"),
    ("I got logged out and now I'm being charged for a second account.", "billing"),
    ("Something is wrong with my account, the invoice page shows an error.", "tech"),
    ("I need help.", "general"),
    # Held-out set, used for the live phase in 02_shadow_then_live.py
    ("Please refund the duplicate charge from March.", "billing"),
    ("Our invoice lists the wrong company name.", "billing"),
    ("The app shows an error when saving settings.", "tech"),
    ("Reports page crashes on large files.", "tech"),
    ("Reset my password please, the link expired.", "account"),
    ("How do I turn on 2FA for my team?", "account"),
    ("Any plans for a public feature roadmap?", "general"),
    ("Is my subscription renewal going to fail because my card expired and the app logged me out?", "billing"),
]
SHADOW_SET, LIVE_SET = TICKETS[:44], TICKETS[44:]
_TRUTH = dict(TICKETS)

FAKE_RULES = {
    "billing": ["charge", "refund", "invoice", "bill", "payment", "card", "subscription", "plan", "price",
                "discount", "seats", "money"],
    "tech": ["error", "crash", "bug", "broken", "outage", "slow", "timeout", "timing out", "freezes", "blank",
             "stopped working", "failing", "500", "429"],
    "account": ["password", "log in", "logged", "sign in", "2fa", "two-factor", "locked", "email address",
                "username", "account", "admin user", "ownership"],
    "general": ["hours", "thanks", "dark mode", "roadmap", "office", "partnership", "best practices",
                "hello", "feature"],
}

LLM_LATENCY_S = float(os.getenv("DEMO_LLM_LATENCY", "0.25"))
llm_calls = 0


def fake_llm_router(text: str) -> str:
    """Stands in for your existing LLM routing call: slow, costs money, usually right."""
    global llm_calls
    llm_calls += 1
    time.sleep(LLM_LATENCY_S)
    return _TRUTH.get(text, "general")


def pick_backend(rules=None):
    if os.getenv("TYPESAFE_API_KEY"):
        print(">> Using REAL Jev (TYPESAFE_API_KEY is set).\n")
        return TypeSafeBackend()
    print(">> Offline demo: FakeJev stands in for Jev (keyword-based, deterministic).")
    print(">> Set TYPESAFE_API_KEY to run the same script against real Jev.\n")
    return FakeJev(rules or FAKE_RULES, latency_ms=30)


def who_decided(router) -> str:
    event = router.jev.events()[-1]
    if event.get("decided_by") == "jev":
        return f"Jev {event['jev_confidence']:.2f}"
    return "your LLM"
