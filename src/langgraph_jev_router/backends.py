"""Decision backends.

``TypeSafeBackend`` calls Jev through the official ``typesafe-sdk``.
``FakeJev`` is a deterministic stand-in for tests and offline demos. The library never
falls back to it on its own: you only get it if you pass it explicitly.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
import zlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ._util import label_base
from .types import ChoiceRequest, Decision

_QUESTION_ID = "route"


class TypeSafeBackend:
    """Jev via the official SDK, with routing-friendly defaults.

    Routing sits on the hot path and always has a fallback, so the retry budget is short:
    if Jev is slow or down, your own router takes over quickly instead of the graph hanging.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 3.0,
        max_retries: int = 1,
        total_timeout: float = 5.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._total_timeout = total_timeout
        self._client = None
        self._lock = threading.Lock()
        self._async_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, object]" = weakref.WeakKeyDictionary()

    # -- clients ---------------------------------------------------------------------------
    def _sdk(self):
        try:
            import typesafe_sdk
        except ImportError as exc:  # pragma: no cover - it is a declared dependency
            raise RuntimeError("typesafe-sdk is not installed: pip install typesafe-sdk") from exc
        return typesafe_sdk

    def _client_kwargs(self) -> dict:
        sdk = self._sdk()
        kwargs = dict(
            timeout=self._timeout,
            retry=sdk.RetryPolicy(max_retries=self._max_retries, timeout=self._total_timeout),
        )
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return kwargs

    def _sync_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._sdk().TypeSafeClient(**self._client_kwargs())
        return self._client

    def _async_client(self):
        loop = asyncio.get_running_loop()
        client = self._async_clients.get(loop)
        if client is None:
            client = self._sdk().AsyncTypeSafeClient(**self._client_kwargs())
            self._async_clients[loop] = client
        return client

    # -- request/response ------------------------------------------------------------------
    def _questions(self, request: ChoiceRequest) -> dict:
        Choice = self._sdk().Choice
        return {_QUESTION_ID: Choice(instructions=request.question, criteria=request.criteria)}

    @staticmethod
    def _to_decision(response) -> Decision:
        answer = response.choices.get(_QUESTION_ID)
        if answer is None:
            raise RuntimeError("Jev response did not include the routing answer")
        try:
            request_id = response.request_id
        except Exception:
            request_id = None
        return Decision(
            route=answer.choice,
            confidence=float(answer.confidence),
            probabilities={k: float(v) for k, v in answer.probabilities.items()},
            model=response.model,
            request_id=request_id,
            input_tokens=int(response.usage.input_tokens or 0),
        )

    def choose(self, request: ChoiceRequest) -> Decision:
        response = self._sync_client().system_one(
            state=request.state, questions=self._questions(request), model=request.model
        )
        return self._to_decision(response)

    async def achoose(self, request: ChoiceRequest) -> Decision:
        response = await self._async_client().system_one(
            state=request.state, questions=self._questions(request), model=request.model
        )
        return self._to_decision(response)


_default_backend: Optional[TypeSafeBackend] = None
_default_lock = threading.Lock()


def default_backend() -> TypeSafeBackend:
    global _default_backend
    if _default_backend is None:
        with _default_lock:
            if _default_backend is None:
                _default_backend = TypeSafeBackend()
    return _default_backend


class FakeJev:
    """Deterministic stand-in for Jev. For unit tests and offline demos only.

    Either give keyword ``rules`` ({route: [keywords]}) or a ``decide(text, routes)``
    callable returning ``(route, confidence)``.
    """

    def __init__(
        self,
        rules: Optional[Dict[str, Sequence[str]]] = None,
        *,
        decide: Optional[Callable[[str, List[str]], Tuple[str, float]]] = None,
        latency_ms: float = 0.0,
        fail_with: Optional[BaseException] = None,
    ) -> None:
        self.rules = {label_base(k): [w.lower() for w in v] for k, v in (rules or {}).items()}
        self._decide = decide
        self.latency_ms = latency_ms
        self.fail_with = fail_with
        self.calls = 0
        self.requests: List[ChoiceRequest] = []

    def _answer(self, request: ChoiceRequest) -> Decision:
        self.calls += 1
        self.requests.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        text = str(request.state.get("request", request.state))
        labels = request.labels
        if self._decide is not None:
            route, conf = self._decide(text, labels)
        else:
            route, conf = self._keyword_decide(text, labels)
        probs = {lbl: (conf if lbl == route else (1 - conf) / max(len(labels) - 1, 1)) for lbl in labels}
        return Decision(route=route, confidence=conf, probabilities=probs, model="fake-jev",
                        input_tokens=len(text.split()) + 40)

    def _keyword_decide(self, text: str, labels: List[str]) -> Tuple[str, float]:
        lowered = text.lower()
        hits = {lbl: sum(1 for w in self.rules.get(lbl, []) if w in lowered) for lbl in labels}
        jitter = (zlib.crc32(text.encode()) % 6) / 100  # deterministic spread, 0.00-0.05
        top = max(hits, key=lambda k: hits[k])
        if hits[top] == 0:
            return labels[0], round(0.30 + jitter, 3)
        share = hits[top] / sum(hits.values())  # 1.0 when only one route matched
        conf = share * (0.70 + 0.08 * (min(hits[top], 3) - 1)) + jitter
        return top, round(min(conf, 0.99), 3)

    def choose(self, request: ChoiceRequest) -> Decision:
        if self.latency_ms:
            import time

            time.sleep(self.latency_ms / 1000)
        return self._answer(request)

    async def achoose(self, request: ChoiceRequest) -> Decision:
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        return self._answer(request)
