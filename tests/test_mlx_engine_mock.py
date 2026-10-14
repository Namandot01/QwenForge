"""Exercise prompt-prefix reuse without requiring Apple Silicon/MLX wheels."""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass

from qwenforge.mlx_engine import NativeMLXEngine


@dataclass
class FakeResponse:
    text: str
    token: int
    finish_reason: str | None = None


class FakeCache:
    def __init__(self):
        self.offset = 0

    @property
    def state(self):
        return {"offset": self.offset}


class FakeArray(list):
    @property
    def size(self):
        return len(self)

    def __getitem__(self, key):
        if key is None:
            return FakeArray([list(self)])
        result = super().__getitem__(key)
        return FakeArray(result) if isinstance(result, list) else result


class FakeModel:
    def __init__(self, chunk_sizes=None, fail_above=None):
        self.chunk_sizes = chunk_sizes if chunk_sizes is not None else []
        self.fail_above = fail_above

    def __call__(self, tokens, cache=None):
        rows = tokens[0] if tokens and isinstance(tokens[0], list) else tokens
        count = len(rows)
        self.chunk_sizes.append(count)
        for item in cache or []:
            item.offset += count
        if self.fail_above is not None and count > self.fail_above:
            raise RuntimeError("Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)")
        return None


def _install_fake_mlx(monkeypatch):
    import qwenforge.mlx_engine as engine_module

    monkeypatch.setattr(engine_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(engine_module.platform, "machine", lambda: "arm64")

    mx = types.ModuleType("mlx.core")
    mx.uint32 = "uint32"
    mx.array = lambda values, dtype=None: FakeArray(list(values))
    mx.eval = lambda values: None
    mx.clear_cache = lambda: None
    mx.get_peak_memory = lambda: 123_000_000
    mx.get_active_memory = lambda: 100_000_000
    mlx = types.ModuleType("mlx")
    mlx.__path__ = []
    mlx.core = mx

    cache_module = types.ModuleType("mlx_lm.models.cache")
    made_caches = []
    chunk_sizes = []

    def make_prompt_cache(model):
        cache = FakeCache()
        made_caches.append(cache)
        return [cache]

    cache_module.make_prompt_cache = make_prompt_cache
    cache_module.save_prompt_cache = lambda path, cache: None
    cache_module.load_prompt_cache = lambda path: [FakeCache()]

    generate_module = types.ModuleType("mlx_lm.generate")
    generate_module.generation_stream = object()

    def stream_generate(model, tokenizer, prompt, *, max_tokens, prompt_cache, sampler, **kwargs):
        cache = prompt_cache[0]
        cache.offset += len(prompt)
        tokens = [65, 66][:max_tokens]
        for index, token in enumerate(tokens):
            yield FakeResponse(chr(token), token, "length" if index == len(tokens) - 1 else None)
            if index < len(tokens) - 1:
                # Model processes each yielded token except the final token into the cache.
                cache.offset += 1

    generate_module.stream_generate = stream_generate
    sample_module = types.ModuleType("mlx_lm.sample_utils")
    sample_module.make_sampler = lambda **kwargs: kwargs

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True, **kwargs):
            text = "".join(f"{message['role']}:{message['content']}\n" for message in messages)
            return text + "assistant:"

        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.__path__ = []
    mlx_lm.load = lambda model_id: (FakeModel(chunk_sizes), FakeTokenizer())
    models_module = types.ModuleType("mlx_lm.models")
    models_module.__path__ = []
    models_module.cache = cache_module

    for name, module in {
        "mlx": mlx,
        "mlx.core": mx,
        "mlx_lm": mlx_lm,
        "mlx_lm.models": models_module,
        "mlx_lm.models.cache": cache_module,
        "mlx_lm.generate": generate_module,
        "mlx_lm.sample_utils": sample_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return made_caches, chunk_sizes


def test_prompt_cache_reuses_exact_conversation_prefix(monkeypatch):
    created, chunk_sizes = _install_fake_mlx(monkeypatch)
    engine = NativeMLXEngine("/fake/model", context_window=2048, kv_bits=4)

    first_messages = [{"role": "user", "content": "hello"}]
    first = list(engine.stream_responses(first_messages, max_tokens=2))
    assert len(first) == 2
    first_cache_tokens = engine._cache_token_ids.copy()

    # Match a normal follow-up request: caller resends the conversation history.
    second_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "AB"},
        {"role": "user", "content": "continue"},
    ]
    rendered = engine.tokenizer.apply_chat_template(second_messages, tokenize=False, add_generation_prompt=True)
    second_prompt_ids = engine.tokenizer.encode(rendered, add_special_tokens=False)
    assert second_prompt_ids[: len(first_cache_tokens)] == first_cache_tokens

    second = list(engine.stream_responses(second_messages, max_tokens=2))
    assert len(second) == 2
    assert len(created) == 1, "a matching prefix should reuse, not allocate, the model cache"
    assert engine.health()["last_generation"]["prompt_tokens"] == len(second_prompt_ids)


def test_adaptive_prefill_uses_smaller_chunks_as_context_grows(monkeypatch):
    from qwenforge.prefill_schedule import adaptive_prefill_chunk_size

    assert adaptive_prefill_chunk_size(0, 20_000, 512) == 512
    assert adaptive_prefill_chunk_size(8_192, 20_000, 512) == 256
    assert adaptive_prefill_chunk_size(32_768, 70_000, 512) == 128
    assert adaptive_prefill_chunk_size(65_536, 70_000, 512) == 64
    assert adaptive_prefill_chunk_size(131_072, 168_960, 512) == 32
    assert adaptive_prefill_chunk_size(0, 168_960, 512) == 128
    assert adaptive_prefill_chunk_size(0, 168_960, 64) == 64


def test_adaptive_prefill_executes_smaller_passes_for_long_request(monkeypatch):
    created, chunk_sizes = _install_fake_mlx(monkeypatch)
    engine = NativeMLXEngine(
        "/fake/model", context_window=100_000, kv_bits=4,
        prefill_step_size=512, prefill_strategy="adaptive",
    )
    cache = [FakeCache()]
    suffix = list(range(70_000))
    prefill = engine._adaptive_prefill(suffix, cache, len(suffix))
    while True:
        try:
            event = next(prefill)
            assert event is None  # in production this becomes an SSE keepalive
        except StopIteration as completed:
            final_prompt_token, elapsed = completed.value
            break

    # All prompt tokens except the final one are already in cache; the normal
    # MLX-LM stream generator must process the final token exactly once.
    assert len(final_prompt_token) == 1
    assert final_prompt_token[0] == suffix[-1]
    assert cache[0].offset == len(suffix) - 1
    assert elapsed >= 0
    assert max(chunk_sizes) <= 256
    assert 256 in chunk_sizes
    assert 128 in chunk_sizes
    assert 64 in chunk_sizes


def test_adaptive_prefill_retries_smaller_chunks_after_metal_oom(monkeypatch):
    _created, chunk_sizes = _install_fake_mlx(monkeypatch)
    engine = NativeMLXEngine(
        "/fake/model", context_window=2048, kv_bits=4, prefill_step_size=512,
        prefill_strategy="adaptive",
    )
    engine.model = FakeModel(chunk_sizes, fail_above=8)
    prompt = list(range(100))
    cache = engine._make_prompt_cache(engine.model)
    prefill = engine._adaptive_prefill(prompt, cache, len(prompt), full_prompt_ids=prompt)
    while True:
        try:
            next(prefill)
        except StopIteration as completed:
            final_token, _elapsed = completed.value
            break
    assert final_token[0] == prompt[-1]
    assert max(chunk_sizes) == 99  # first attempt fails, then the retry cap decreases
    successful_or_failed_sizes = chunk_sizes[1:]
    assert any(size <= 8 for size in successful_or_failed_sizes)
    assert engine._cache is not None or cache


def test_partial_ssd_prefix_wins_over_longer_unrelated_ram_cache(monkeypatch, tmp_path):
    _created, _chunk_sizes = _install_fake_mlx(monkeypatch)
    engine = NativeMLXEngine(
        "/fake/model", context_window=2048, kv_bits=4,
        snapshot_dir=tmp_path, snapshot_min_tokens=1,
    )
    requested = [1, 2, 3, 4, 5, 6]
    # Simulate a longer cache left over from a different conversation.
    engine._cache = [FakeCache()]
    engine._cache_token_ids = [90, 91, 92, 93, 94, 95, 96, 97]
    partial_cache = [FakeCache()]
    partial_tokens = requested[:3]
    monkeypatch.setattr(engine._prefill_store, "matching_token_ids", lambda _prompt: partial_tokens)
    monkeypatch.setattr(
        engine._prefill_store,
        "load_for_prefix",
        lambda _prompt, **_kwargs: (partial_cache, partial_tokens),
    )

    assert engine._restore_partial_prefill_if_better(requested)
    assert engine._cache is partial_cache
    assert engine._cache_token_ids == partial_tokens


def test_one_token_oom_does_not_leave_corrupt_cache_reusable(monkeypatch):
    _created, chunk_sizes = _install_fake_mlx(monkeypatch)
    engine = NativeMLXEngine(
        "/fake/model", context_window=2048, kv_bits=4, prefill_step_size=32,
        prefill_strategy="adaptive",
    )
    engine.model = FakeModel(chunk_sizes, fail_above=0)
    prompt = list(range(100))
    cache = engine._make_prompt_cache(engine.model)
    engine._cache = cache
    engine._cache_token_ids = prompt[:10]
    prefill = engine._adaptive_prefill(prompt, cache, len(prompt), full_prompt_ids=prompt)
    caught = None
    try:
        list(prefill)
    except RuntimeError as exc:
        caught = exc
    assert caught is not None
    assert "1-token prefill pass" in str(caught)
    assert engine._cache is None
    assert engine._cache_token_ids == []
    assert cache == []
