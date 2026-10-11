import operator
from typing import Annotated, TypedDict

import pytest

from langgraph_jev_router import get_recorder, reset_configuration
from langgraph_jev_router._util import decision_cache, reset_warnings

ENV_VARS = ["JEV_ROUTER_SHADOW", "JEV_ROUTER_DISABLED", "JEV_ROUTER_LOG", "JEV_ROUTER_THRESHOLD",
            "JEV_ROUTER_LOG_INPUTS", "TYPESAFE_DEFAULT_MODEL"]


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    reset_configuration()
    get_recorder().clear()
    decision_cache.clear()
    reset_warnings()
    yield
    reset_configuration()
    get_recorder().clear()
    decision_cache.clear()


class ListState(TypedDict):
    messages: Annotated[list, operator.add]


def fixed(route, confidence):
    """FakeJev decide() that always gives the same answer."""
    return lambda text, labels: (route, confidence)
