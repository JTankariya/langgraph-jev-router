"""Decide what Jev gets to see.

Default rule: Jev routes a *fresh user message*. If the newest message is not from the user
(an agent or tool just replied and the graph is looping back), the question has changed to
"what next / are we done?", which needs the conversation, not just the user's last line. In
that case Jev steps aside and your router decides. This is what stops a Jev-first router from
sending the same request back to the same agent forever.

Pass ``state_from=`` to change this, e.g. to let Jev see more of the state.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

_TEXT_KEYS = ("input", "question", "query", "request", "text", "message", "prompt")
_HUMAN_ROLES = {"human", "user"}


def _get(state: Any, key: str) -> Any:
    if isinstance(state, Mapping):
        return state.get(key)
    return getattr(state, key, None)


def is_human(message: Any) -> bool:
    if isinstance(message, str):
        return True
    if isinstance(message, (tuple, list)) and len(message) == 2 and isinstance(message[0], str):
        return message[0].lower() in _HUMAN_ROLES
    if isinstance(message, Mapping):
        role = message.get("role") or message.get("type")
        return isinstance(role, str) and role.lower() in _HUMAN_ROLES
    return getattr(message, "type", None) == "human"


def message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, (tuple, list)) and len(message) == 2:
        content = message[1]
    elif isinstance(message, Mapping):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content)


def last_human_text(state: Any) -> Optional[str]:
    """Most recent user message anywhere in the history (used by shadow observers)."""
    messages = _get(state, "messages")
    if not messages:
        return None
    for message in reversed(list(messages)):
        if is_human(message):
            text = message_text(message).strip()
            return text or None
    return None


def default_state_from(state: Any) -> Tuple[Optional[dict], Optional[str]]:
    """Return (payload for Jev, None) or (None, reason Jev should step aside)."""
    messages = _get(state, "messages")
    if messages is not None:
        messages = list(messages)
        if not messages:
            return None, "no_input"
        last = messages[-1]
        if not is_human(last):
            return None, "mid_conversation"
        text = message_text(last).strip()
        return ({"request": text}, None) if text else (None, "no_input")
    for key in _TEXT_KEYS:
        value = _get(state, key)
        if isinstance(value, str) and value.strip():
            return {"request": value.strip()}, None
    return None, "no_input"


def normalize_payload(raw: Any) -> Optional[dict]:
    """Accept what a custom state_from returns: None, a string, or a JSON-able dict."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return {"request": raw} if raw.strip() else None
    if isinstance(raw, Mapping):
        return dict(raw)
    return {"request": str(raw)}
