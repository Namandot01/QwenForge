"""Adaptive prefill pass sizing for memory-constrained Apple Silicon.

This is a heuristic scheduler, not a Metal memory guarantee. It caps MLX-LM
prompt passes more aggressively as both the request length and current prefix
length grow, helping bound temporary activation/workspace allocations.
"""
from __future__ import annotations


def adaptive_prefill_chunk_size(
    context_tokens_before_pass: int,
    total_prompt_tokens: int,
    max_chunk_tokens: int = 512,
) -> int:
    """Choose a conservative prefill chunk size for a Qwen3.8-27B request.

    The schedule is intentionally powers-of-two and never below 32 tokens. The
    request-length cap starts large prompts cautiously; the position cap shrinks
    additional passes as the live attention cache grows.
    """
    if context_tokens_before_pass < 0:
        raise ValueError("context_tokens_before_pass cannot be negative")
    if total_prompt_tokens < 1:
        raise ValueError("total_prompt_tokens must be positive")
    if max_chunk_tokens < 32:
        raise ValueError("max_chunk_tokens must be at least 32")

    request_cap = max_chunk_tokens
    if total_prompt_tokens >= 131_072:
        request_cap = min(request_cap, 128)
    elif total_prompt_tokens >= 65_536:
        request_cap = min(request_cap, 256)

    position_cap = 512
    if context_tokens_before_pass >= 131_072:
        position_cap = 32
    elif context_tokens_before_pass >= 65_536:
        position_cap = 64
    elif context_tokens_before_pass >= 32_768:
        position_cap = 128
    elif context_tokens_before_pass >= 8_192:
        position_cap = 256

    return max(32, min(max_chunk_tokens, request_cap, position_cap))
