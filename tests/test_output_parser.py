from qwenforge.mlx_engine import QwenOutputParser, _normalize_messages_for_qwen


def test_parser_routes_thinking_and_final_answer_separately_across_chunks():
    parser = QwenOutputParser(thinking_enabled=True)
    events = []
    events += parser.feed("Let me inspect the question. ")
    events += parser.feed("Now I know.</think>The answer is ")
    events += parser.feed("42.")
    events += parser.finish()
    reasoning = "".join(value for kind, value in events if kind == "reasoning_content")
    content = "".join(value for kind, value in events if kind == "content")
    assert reasoning == "Let me inspect the question. Now I know."
    assert content == "The answer is 42."
    assert "<think>" not in reasoning + content
    assert "</think>" not in reasoning + content
    assert parser.in_thinking is False


def test_parser_strips_explicit_think_open_and_close_markers():
    parser = QwenOutputParser(thinking_enabled=False)
    events = parser.feed("<think>private thoughts</think>visible answer")
    events += parser.finish()
    assert ("reasoning_content", "private thoughts") in events
    assert ("content", "visible answer") in events


def test_parser_handles_qwen_native_tool_call_and_typed_parameters_across_chunks():
    tools = [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    }]
    parser = QwenOutputParser(thinking_enabled=False, tools=tools)
    events = []
    events += parser.feed("Need inspect. <tool_")
    events += parser.feed("call>\n<function=bash>\n<parameter=command>\nls -la /tmp\n</parameter>\n")
    events += parser.feed("<parameter=timeout>\n12\n</parameter>\n</function>\n</tool_call>")
    events += parser.finish()
    assert ("content", "Need inspect. ") in events
    calls = [value for kind, value in events if kind == "tool_call"]
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "bash"
    assert calls[0]["function"]["arguments"] == {"command": "ls -la /tmp", "timeout": 12}


def test_unknown_or_incomplete_tool_call_is_not_executed():
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {"properties": {}}}}]
    parser = QwenOutputParser(thinking_enabled=False, tools=tools)
    events = parser.feed("<tool_call><function=unknown><parameter=x>y</parameter></function>")
    events += parser.finish()
    assert not any(kind == "tool_call" for kind, _ in events)
    assert any(kind == "content" and "<tool_call>" in value for kind, value in events)


def test_normalizer_preserves_reasoning_and_prior_tool_calls():
    messages = _normalize_messages_for_qwen([
        {"role": "assistant", "content": None, "reasoning_content": "planning", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"pwd\"}"}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "/tmp"},
    ])
    assert messages[0]["reasoning_content"] == "planning"
    assert messages[0]["tool_calls"] == [{"name": "bash", "arguments": {"command": "pwd"}}]
    assert messages[1]["role"] == "tool"
    assert messages[1]["content"] == "/tmp"
