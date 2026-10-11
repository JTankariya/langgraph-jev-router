from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_jev_router.extract import default_state_from, last_human_text, normalize_payload


def test_last_human_message_object():
    payload, reason = default_state_from({"messages": [HumanMessage("refund please")]})
    assert payload == {"request": "refund please"} and reason is None


def test_mid_conversation_steps_aside():
    state = {"messages": [HumanMessage("refund please"), AIMessage("done, refunded")]}
    assert default_state_from(state) == (None, "mid_conversation")
    state = {"messages": [HumanMessage("x"), ToolMessage("ok", tool_call_id="1")]}
    assert default_state_from(state) == (None, "mid_conversation")


def test_tuple_dict_and_string_messages():
    assert default_state_from({"messages": [("user", "hi")]})[0] == {"request": "hi"}
    assert default_state_from({"messages": [{"role": "user", "content": "hi"}]})[0] == {"request": "hi"}
    assert default_state_from({"messages": ["plain string"]})[0] == {"request": "plain string"}
    assert default_state_from({"messages": [("assistant", "hi")]}) == (None, "mid_conversation")


def test_content_blocks():
    msg = HumanMessage(content=[{"type": "text", "text": "part one"}, {"type": "image_url", "image_url": "x"},
                                {"type": "text", "text": "part two"}])
    assert default_state_from({"messages": [msg]})[0] == {"request": "part one\npart two"}


def test_text_keys_and_objects():
    assert default_state_from({"question": "where is my order"})[0] == {"request": "where is my order"}

    class Obj:
        query = "  spaced  "

    assert default_state_from(Obj())[0] == {"request": "spaced"}
    assert default_state_from({"other": 1}) == (None, "no_input")
    assert default_state_from({"messages": []}) == (None, "no_input")
    assert default_state_from({"messages": [HumanMessage("   ")]}) == (None, "no_input")


def test_last_human_text_scans_history():
    state = {"messages": [HumanMessage("first"), AIMessage("a"), HumanMessage("second"), AIMessage("b")]}
    assert last_human_text(state) == "second"


def test_normalize_payload():
    assert normalize_payload(None) is None
    assert normalize_payload("x") == {"request": "x"}
    assert normalize_payload("  ") is None
    assert normalize_payload({"a": 1}) == {"a": 1}
