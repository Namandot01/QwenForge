"""Simple OpenAI-compatible streaming benchmark for a local QwenForge/oMLX endpoint."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class BenchResult:
    elapsed_s: float
    ttft_s: float | None
    completion_tokens: int | None
    prompt_tokens: int | None
    output_chars: int
    output_text: str
    decode_tok_s: float | None
    approx_char_s: float


def stream_chat_completion(
    *, base_url: str, model: str, prompt: str, max_tokens: int = 256,
    temperature: float = 0.0, api_key: str | None = None, timeout: float = 3600,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> BenchResult:
    base_url = base_url.rstrip("/")
    url = base_url if base_url.endswith("/v1/chat/completions") else base_url + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    start = time.perf_counter()
    first_content: float | None = None
    output_chars = 0
    output_parts: list[str] = []
    completion_tokens = None
    prompt_tokens = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk: dict[str, Any] = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    if isinstance(usage.get("completion_tokens"), int):
                        completion_tokens = usage["completion_tokens"]
                    if isinstance(usage.get("prompt_tokens"), int):
                        prompt_tokens = usage["prompt_tokens"]
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    # Qwen may stream reasoning separately from final visible content.
                    # Count the first generated text field so TTFT/decode duration do not
                    # accidentally exclude a long reasoning phase.
                    content = next(
                        (delta.get(key) for key in ("reasoning_content", "reasoning", "content")
                         if isinstance(delta.get(key), str) and delta.get(key)),
                        None,
                    )
                    if isinstance(content, str) and content:
                        if first_content is None:
                            first_content = time.perf_counter()
                        output_chars += len(content)
                        output_parts.append(content)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Server returned HTTP {exc.code}: {body[:1500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc

    end = time.perf_counter()
    elapsed = end - start
    decode_elapsed = end - first_content if first_content is not None else None
    decode_tok_s = (
        completion_tokens / decode_elapsed
        if completion_tokens is not None and decode_elapsed and decode_elapsed > 0 else None
    )
    return BenchResult(
        elapsed_s=elapsed,
        ttft_s=(first_content - start) if first_content is not None else None,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        output_chars=output_chars,
        output_text="".join(output_parts),
        decode_tok_s=decode_tok_s,
        approx_char_s=(output_chars / decode_elapsed) if decode_elapsed and decode_elapsed > 0 else 0.0,
    )
