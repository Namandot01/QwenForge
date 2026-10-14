import json

from qwenforge import bench


class FakeHTTPResponse:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self._lines)


def test_benchmark_parses_sse_usage_output_and_template_kwargs(monkeypatch):
    captured = {}
    payload_lines = [
        b'data: {"choices":[{"delta":{"content":"OK"}}]}\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":123,"completion_tokens":1,"total_tokens":124}}\n',
        b'data: [DONE]\n',
    ]

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return FakeHTTPResponse(payload_lines)

    monkeypatch.setattr(bench.urllib.request, "urlopen", fake_urlopen)
    result = bench.stream_chat_completion(
        base_url="http://127.0.0.1:8000",
        model="qwen",
        prompt="hello",
        max_tokens=8,
        chat_template_kwargs={"enable_thinking": False},
    )

    assert result.output_text == "OK"
    assert result.output_chars == 2
    assert result.prompt_tokens == 123
    assert result.completion_tokens == 1
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["timeout"] == 3600
