from __future__ import annotations

from types import SimpleNamespace

from qwenforge.mlx_engine import (
    _TOOL_CALL_HEARTBEAT_SECONDS,
    _TOOL_CALL_HEARTBEAT_TEXT,
    _maybe_send_tool_call_heartbeat,
)


def test_heartbeat_emitted_when_tool_call_buffer_is_idle():
    parser = SimpleNamespace(tool_buffer="<tool_call><function=write>")
    emitted = []
    next_time = _maybe_send_tool_call_heartbeat(
        parser, 10.0, emitted.append, now=10.0 + _TOOL_CALL_HEARTBEAT_SECONDS + 0.1
    )
    assert emitted == [{"reasoning_content": _TOOL_CALL_HEARTBEAT_TEXT}]
    assert next_time == 10.0 + _TOOL_CALL_HEARTBEAT_SECONDS + 0.1


def test_heartbeat_not_emitted_before_interval():
    parser = SimpleNamespace(tool_buffer="<tool_call>")
    emitted = []
    next_time = _maybe_send_tool_call_heartbeat(parser, 10.0, emitted.append, now=20.0)
    assert emitted == []
    assert next_time == 10.0


def test_heartbeat_timer_resets_outside_tool_call():
    parser = SimpleNamespace(tool_buffer=None)
    emitted = []
    next_time = _maybe_send_tool_call_heartbeat(parser, 1.0, emitted.append, now=99.0)
    assert emitted == []
    assert next_time == 99.0
