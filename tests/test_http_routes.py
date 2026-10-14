"""HTTP dashboard/API route smoke tests with a fake inference engine."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer


def test_dashboard_api_routes(monkeypatch):
    import qwenforge.mlx_engine as module

    holder = {}

    class FakeEngine:
        model_name = "Qwen3.8-27B-4bit"

        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            self.snapshot_dir = kwargs.get("snapshot_dir")
            self._lock = threading.Lock()
            self.last_generation_kwargs = None
            self.snapshot_saves = 0
            holder["engine"] = self

        def health(self):
            return {
                "status": "ok", "engine": "mlx-lm", "model": self.model_id,
                "context_window": 262144, "kv_bits": 4, "kv_group_size": 64,
                "active_memory_gb": 1.2, "peak_memory_gb": 1.4,
                "last_generation": {},
            }

        def save_snapshot(self):
            self.snapshot_saves += 1
            return None

        def stream_responses(self, messages, **kwargs):
            assert messages and messages[-1]["role"] == "user"
            self.last_generation_kwargs = kwargs
            if kwargs.get("tools"):
                yield SimpleNamespace(
                    text="<tool_call>\n<function=bash>\n<parameter=command>\nls -la /tmp\n</parameter>\n</function>\n</tool_call>",
                    finish_reason="stop",
                ), 3, 1
                return
            if kwargs.get("chat_template_kwargs", {}).get("enable_thinking", True):
                yield SimpleNamespace(text="Let me think this through.", finish_reason=None), 3, 1
                yield SimpleNamespace(text="</think>The answer is 42.", finish_reason="stop"), 3, 2
                return
            yield SimpleNamespace(text="Hello", finish_reason=None), 3, 1
            yield SimpleNamespace(text=" streamed", finish_reason="stop"), 3, 2

    class TrackingServer(ThreadingHTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["server"] = self

    monkeypatch.setattr(module, "NativeMLXEngine", FakeEngine)
    monkeypatch.setattr(module, "ThreadingHTTPServer", TrackingServer)

    import tempfile
    snapshot_dir = tempfile.mkdtemp(prefix="qwenforge-snapshot-test-")
    runner = threading.Thread(
        target=module.serve_http,
        kwargs={"model_id": "/fake/model", "host": "127.0.0.1", "port": 0, "snapshot_dir": snapshot_dir},
        daemon=True,
    )
    runner.start()
    try:
        deadline = time.time() + 3
        while "server" not in holder and time.time() < deadline:
            time.sleep(0.01)
        assert "server" in holder
        server = holder["server"]
        base = f"http://127.0.0.1:{server.server_port}"

        with urllib.request.urlopen(base + "/", timeout=2) as response:
            assert response.status == 200
            assert "text/html" in response.headers["Content-Type"]
            page = response.read()
            assert b"QwenForge" in page
            assert b'max="262144"' in page
            assert b"Math.min(262144" in page
            assert b"repetition guard stopped a repeated-token loop" in page
            assert b"possible early-EOS issue" in page
            assert b"event.qwenforge?.stop_reason" in page

        with urllib.request.urlopen(base + "/admin/dashboard?tab=models&modelsTab=manager", timeout=2) as response:
            assert response.status == 200
            assert b"Quick chat test" in response.read()

        with urllib.request.urlopen(base + "/v1", timeout=2) as response:
            assert response.status == 200
            payload = json.loads(response.read())
            assert payload["web_ui"] == "/"
            assert "chat_completions" in payload["endpoints"]

        with urllib.request.urlopen(base + "/v1/models", timeout=2) as response:
            assert json.loads(response.read())["data"][0]["id"] == "Qwen3.8-27B-4bit"


        stream_req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({
                "model": "Qwen3.8-27B-4bit",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 8,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream_options": {"include_usage": True},
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(stream_req, timeout=2) as response:
            assert response.status == 200
            assert "text/event-stream" in response.headers["Content-Type"]
            stream_body = response.read().decode()
            assert '"content":"Hello"' in stream_body
            assert '"content":" streamed"' in stream_body
            assert '"completion_tokens":2' in stream_body
            assert "data: [DONE]" in stream_body
        # The first request explicitly disabled thinking, so use instruct-mode defaults.
        assert holder["engine"].last_generation_kwargs["temperature"] == 0.7
        assert holder["engine"].last_generation_kwargs["top_p"] == 0.80
        assert holder["engine"].last_generation_kwargs["top_k"] == 20
        assert holder["engine"].snapshot_saves == 1, "successful streamed turns should checkpoint to SSD"

        thinking_req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({
                "model": "Qwen3.8-27B-4bit",
                "messages": [{"role": "user", "content": "Answer 6 times 7."}],
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
                "stream": True,
                "max_tokens": 32,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(thinking_req, timeout=2) as response:
            body = response.read().decode()
        assert '"reasoning_content":"Let me think this through."' in body
        assert '"content":"The answer is 42."' in body
        assert '"thinking_incomplete":false' in body
        assert holder["engine"].snapshot_saves == 2

        tools_req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({
                "model": "Qwen3.8-27B-4bit",
                "messages": [{"role": "user", "content": "List /tmp."}],
                "chat_template_kwargs": {"enable_thinking": False},
                "tools": [{"type": "function", "function": {
                    "name": "bash", "description": "Run a shell command",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
                }}],
                "stream": True,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(tools_req, timeout=2) as response:
            body = response.read().decode()
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
        call_events = [event for event in events if event.get("choices") and event["choices"][0].get("delta", {}).get("tool_calls")]
        assert call_events
        tool_call = call_events[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["function"]["name"] == "bash"
        assert json.loads(tool_call["function"]["arguments"]) == {"command": "ls -la /tmp"}
        assert any(event.get("choices", [{}])[0].get("finish_reason") == "tool_calls" for event in events if event.get("choices"))
        assert holder["engine"].last_generation_kwargs["tools"][0]["function"]["name"] == "bash"
        assert holder["engine"].snapshot_saves == 3, "assistant tool-call responses should be checkpointed too"

        # Explicitly non-thinking requests get the matching instruct preset.
        nonthinking_req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({
                "model": "Qwen3.8-27B-4bit",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 8,
                "chat_template_kwargs": {"enable_thinking": False},
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(nonthinking_req, timeout=2) as response:
            assert response.status == 200
        assert holder["engine"].last_generation_kwargs["temperature"] == 0.7
        assert holder["engine"].last_generation_kwargs["top_p"] == 0.80
        assert holder["engine"].last_generation_kwargs["top_k"] == 20
        deadline = time.time() + 1
        while holder["engine"].snapshot_saves < 4 and time.time() < deadline:
            time.sleep(0.005)
        assert holder["engine"].snapshot_saves == 4

        with urllib.request.urlopen(base + "/health", timeout=2) as response:
            assert json.loads(response.read())["status"] == "ok"

        with urllib.request.urlopen(base + "/favicon.ico", timeout=2) as response:
            assert response.status == 204

        try:
            urllib.request.urlopen(base + "/does-not-exist", timeout=2)
        except urllib.error.HTTPError as error:
            assert error.code == 404
            assert b"Open / for the QwenForge dashboard" in error.read()
        else:
            raise AssertionError("unknown route should return 404")
    finally:
        if "server" in holder:
            holder["server"].shutdown()
        runner.join(timeout=3)
