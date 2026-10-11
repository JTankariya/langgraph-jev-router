from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable


@dataclass(frozen=True)
class ChoiceRequest:
    """One routing question for a decision backend."""

    question: str
    options: Tuple[Tuple[str, Optional[str]], ...]
    """(label, description) pairs. A None description means "judge by the label alone"."""
    state: Dict[str, Any]
    """What Jev looks at, e.g. {"request": "I was charged twice"}."""
    model: str

    @property
    def criteria(self) -> Dict[str, Optional[str]]:
        return dict(self.options)

    @property
    def labels(self) -> list[str]:
        return [label for label, _ in self.options]

    def cache_key(self) -> tuple:
        return (self.model, self.question, self.options, json.dumps(self.state, sort_keys=True, default=str))


@dataclass(frozen=True)
class Decision:
    """A routing decision, with everything needed to audit it."""

    route: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    model: Optional[str] = None
    request_id: Optional[str] = None
    input_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Backend(Protocol):
    """Anything that can answer a ChoiceRequest. TypeSafeBackend and FakeJev implement this."""

    def choose(self, request: ChoiceRequest) -> Decision: ...

    async def achoose(self, request: ChoiceRequest) -> Decision: ...
