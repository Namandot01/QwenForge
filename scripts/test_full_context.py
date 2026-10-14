#!/usr/bin/env python3
"""Safely probe QwenForge near its configured full context using the local tokenizer.

This loads the tokenizer only (not model weights), builds one synthetic user prompt
whose chat-template token count is as close as practical to the context limit while
reserving a few output tokens, then streams a tiny response from QwenForge.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TASK = (
    "\nTask: This is a synthetic full-context test. Recall the capacity marker at the very beginning of the context. "
    "Reply with exactly QF-FULL-262K-ALPHA and nothing else."
)
PAD_TEXT = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega. "
)


def http_json(url: str, timeout: float = 15.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/Volumes/New Volume/QwenForge/Qwen3.8-27B-4bit",
                        help="Local MLX model directory (tokenizer only is loaded).")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                        help="QwenForge server root URL (not /v1).")
    parser.add_argument("--target-context", type=int, default=262144,
                        help="Desired total context window to probe (default: 262144).")
    parser.add_argument("--output-tokens", type=int, default=16,
                        help="Small output budget so most of the context is prompt (default: 16).")
    parser.add_argument("--safety-tokens", type=int, default=16,
                        help="Unused token margin below the context limit (default: 16).")
    args = parser.parse_args()

    if args.output_tokens < 1 or args.safety_tokens < 0:
        parser.error("--output-tokens must be >= 1 and --safety-tokens must be >= 0")

    base = args.base_url.rstrip("/")
    try:
        health = http_json(base + "/health")
        models = http_json(base + "/v1/models")
    except Exception as exc:
        print(f"ERROR: Cannot reach QwenForge at {base}: {exc}", file=sys.stderr)
        print("Start QwenForge first with --context 262144.", file=sys.stderr)
        return 2

    server_context = int(health.get("context_window", 0) or 0)
    context_limit = min(args.target_context, server_context)
    if server_context < args.target_context:
        print(f"WARNING: server context is {server_context:,}, not {args.target_context:,}; testing the server limit.")
    if context_limit < args.output_tokens + args.safety_tokens + 1024:
        print(f"ERROR: reported server context is too small: {context_limit}", file=sys.stderr)
        return 2

    model_items = models.get("data", [])
    model_id = model_items[0].get("id") if model_items else None
    if not model_id:
        model_id = "Qwen3.8-27B-4bit"

    # Same tokenizer API and chat-template construction as QwenForge's inference path.
    try:
        from mlx_lm.tokenizer_utils import load as load_tokenizer  # type: ignore
        tokenizer = load_tokenizer(Path(args.model))
    except Exception as exc:
        print(f"ERROR: Could not load the local tokenizer from {args.model}: {exc}", file=sys.stderr)
        print("Run this script inside QwenForge's activated .venv; this loads tokenizer files only, not model weights.", file=sys.stderr)
        return 2

    target_prompt_tokens = context_limit - args.output_tokens - args.safety_tokens
    template_kwargs = {"enable_thinking": False}

    def count_prompt_tokens(content: str) -> int:
        messages = [{"role": "user", "content": content}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    def build_content(line_count: int, pad_chars: int = 0) -> str:
        # Numbered records make the payload non-identical line-to-line and easy to reproduce.
        records = "".join(
            f"Context record {i:06d}: synthetic test material for measuring long-context memory. Keep as plain text.\n"
            for i in range(line_count)
        )
        padding = "Padding fragment: " + PAD_TEXT[:pad_chars] + "\n"
        return records + padding + TASK

    try:
        empty_count = count_prompt_tokens(build_content(0, 0))
        probe_count = count_prompt_tokens(build_content(100, 0))
        per_record = max((probe_count - empty_count) / 100.0, 1.0)
        estimate = max(0, int((target_prompt_tokens - empty_count) / per_record))

        # Find the largest number of complete records under the exact token budget.
        lo = 0
        hi = max(estimate + 512, 1024)
        while count_prompt_tokens(build_content(hi, 0)) <= target_prompt_tokens:
            lo = hi
            hi *= 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if count_prompt_tokens(build_content(mid, 0)) <= target_prompt_tokens:
                lo = mid
            else:
                hi = mid

        # Use a partial padding tail to fill the remaining few tokens as closely as possible.
        best_text = build_content(lo, 0)
        best_tokens = count_prompt_tokens(best_text)
        p_lo, p_hi = 0, len(PAD_TEXT)
        while p_lo <= p_hi:
            mid = (p_lo + p_hi) // 2
            candidate = build_content(lo, mid)
            n_tokens = count_prompt_tokens(candidate)
            if n_tokens <= target_prompt_tokens:
                if n_tokens >= best_tokens:
                    best_text, best_tokens = candidate, n_tokens
                p_lo = mid + 1
            else:
                p_hi = mid - 1
    except Exception as exc:
        print(f"ERROR while counting tokens: {exc}", file=sys.stderr)
        return 2

    requested_total = best_tokens + args.output_tokens
    unused = context_limit - requested_total
    utilization = requested_total / context_limit * 100.0
    print("\nQWENFORGE FULL-CONTEXT PREFILL TEST")
    print(f"Server:             {base}")
    print(f"Model ID:           {model_id}")
    print(f"Server context:     {server_context:,}")
    print(f"Target context:     {context_limit:,}")
    print(f"Rendered prompt:    {best_tokens:,} tokens (measured with the local tokenizer)")
    print(f"Requested output:   {args.output_tokens:,} tokens")
    print(f"Planned utilization:{utilization:9.4f}%")
    print(f"Safety left:        {unused:,} tokens")
    print("Thinking:           disabled for this prefill-capacity test")
    print("\nThis is a genuine near-full-context request. MLX/Metal may OOM; if that happens, preserve the QwenForge Terminal log.\n")
    input("Press Enter to send the request, or Ctrl+C to cancel... ")

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": best_text}],
        "max_tokens": args.output_tokens,
        "temperature": 0.7,
        "top_p": 0.80,
        "top_k": 20,
        "min_p": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.monotonic()
    seen_usage = None
    response_text = []
    first_chunk_time = None
    server_error = None
    try:
        # No client read timeout: this test intentionally waits for long prefill.
        with urllib.request.urlopen(req, timeout=None) as resp:
            print(f"HTTP status: {resp.status}")
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    print("\nNon-JSON SSE data:", data[:500])
                    continue
                if "error" in event:
                    server_error = event["error"]
                    print("\nSERVER ERROR:", json.dumps(server_error, ensure_ascii=False, indent=2))
                    continue
                if "usage" in event:
                    seen_usage = event["usage"]
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        if first_chunk_time is None:
                            first_chunk_time = time.monotonic() - started
                        response_text.append(piece)
                        print(piece, end="", flush=True)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"\nHTTP ERROR {exc.code}:\n{error_body}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nREQUEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - started
    print("\n\nRESULT")
    print(f"Elapsed:             {elapsed:.1f} seconds")
    print(f"Time to first text:  {first_chunk_time:.1f} seconds" if first_chunk_time is not None else "Time to first text:  no visible text received")
    print(f"Output:              {''.join(response_text)!r}")
    print(f"Server usage:        {json.dumps(seen_usage) if seen_usage else 'not returned'}")
    if seen_usage:
        actual = int(seen_usage.get("prompt_tokens", 0)) + int(seen_usage.get("completion_tokens", 0))
        print(f"Actual total tokens: {actual:,}/{context_limit:,} ({actual/context_limit*100:.4f}%)")
        if int(seen_usage.get("prompt_tokens", 0)) != best_tokens:
            print("WARNING: server prompt token count differs from local estimate; use server-reported usage as authoritative.")
    print("Check the QwenForge Terminal for adaptive-prefill progress, checkpoint-save messages, and any Metal OOM.")
    if server_error or not seen_usage:
        print("CAPACITY TEST: FAIL (server error or missing usage record)")
        return 1
    if int(seen_usage.get("prompt_tokens", 0)) < target_prompt_tokens - 32:
        print("CAPACITY TEST: INCONCLUSIVE (server accepted fewer prompt tokens than planned)")
        return 1
    print("CAPACITY TEST: PASS (server processed a near-full-context prompt without reporting a server error)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130)
