"""The real TypeSafeBackend + official typesafe-sdk, against a local server speaking Jev's wire format."""

import asyncio
from typing import Literal

from langchain_core.messages import HumanMessage

from langgraph_jev_router import TypeSafeBackend, configure, jev_first
from langgraph_jev_router.types import ChoiceRequest

from fake_jev_server import FakeJevServer

REQ = ChoiceRequest(question="Which team handles `request`?",
                    options=(("billing", "Charges"), ("tech", None)),
                    state={"request": "charged twice"}, model="jev-1.13.0")


def test_sync_request_and_response_mapping():
    with FakeJevServer() as srv:
        d = TypeSafeBackend(api_key="test-key", base_url=srv.url).choose(REQ)
    assert (d.route, d.confidence, d.model, d.request_id, d.input_tokens) == ("billing", 0.93, "jev-1.13.0", "req_test_123", 87)
    (sent,) = srv.requests
    assert sent["path"] == "/v1/systemone"
    assert sent["body"] == {
        "state": {"request": "charged twice"},
        "model": "jev-1.13.0",
        "questions": {"route": {"type": "choice", "instructions": "Which team handles `request`?",
                                "criteria": {"billing": "Charges", "tech": None}}},
    }
    auth = {k.lower(): v for k, v in sent["headers"].items()}
    assert "test-key" in auth.get("authorization", "") + auth.get("x-api-key", "")


def test_async_client():
    async def go():
        backend = TypeSafeBackend(api_key="k", base_url=srv.url)
        return await backend.achoose(REQ), await backend.achoose(REQ)

    with FakeJevServer(choice="tech", confidence=0.81) as srv:
        a, b = asyncio.run(go())
    assert a.route == b.route == "tech" and len(srv.requests) == 2


def test_end_to_end_router_with_real_sdk():
    calls = []

    @jev_first
    def router(state) -> Literal["billing", "tech"]:
        calls.append(1)
        return "tech"

    with FakeJevServer(choice="billing", confidence=0.93) as srv:
        configure(backend=TypeSafeBackend(api_key="k", base_url=srv.url))
        assert router({"messages": [HumanMessage("charged twice")]}) == "billing"
    assert calls == []


def test_server_error_falls_back_fast():
    calls = []

    @jev_first
    def router(state) -> Literal["billing", "tech"]:
        calls.append(1)
        return "tech"

    with FakeJevServer(status=500) as srv:
        configure(backend=TypeSafeBackend(api_key="k", base_url=srv.url, max_retries=0))
        assert router({"messages": [HumanMessage("x")]}) == "tech"
    assert calls == [1]
    assert "InternalServerError" in router.jev.events()[0]["error"]


def test_unknown_label_from_api_is_rejected():
    @jev_first
    def router(state) -> Literal["billing", "tech"]:
        return "tech"

    with FakeJevServer(choice="sales", confidence=0.99) as srv:
        configure(backend=TypeSafeBackend(api_key="k", base_url=srv.url))
        assert router({"messages": [HumanMessage("x")]}) == "tech"
    assert router.jev.events()[0]["reason"] == "invalid_choice"
