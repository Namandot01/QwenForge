from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np

from qwenforge.prefill_checkpoint import IncrementalPrefillStore


class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    @property
    def state(self):
        return self.keys, self.values, self.offset

    @state.setter
    def state(self, value):
        self.keys, self.values, self.offset = value

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.state = state
        return obj


class QuantizedKVCache:
    def __init__(self, group_size=64, bits=4):
        self.keys = None
        self.values = None
        self.offset = 0
        self.group_size = group_size
        self.bits = bits

    @property
    def state(self):
        return self.keys, self.values, self.offset, self.group_size, self.bits

    @state.setter
    def state(self, value):
        self.keys, self.values, self.offset, self.group_size, self.bits = value

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.state = state
        return obj


class ArraysCache:
    def __init__(self, value=0):
        self.cache = [np.array([value], dtype=np.int32), np.array([value + 1], dtype=np.int32)]
        self.left_padding = None
        self.lengths = None

    @property
    def state(self):
        return self.cache, self.left_padding, self.lengths

    @state.setter
    def state(self, value):
        self.cache, self.left_padding, self.lengths = value

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.state = state
        return obj


class UnknownCache:
    def __init__(self):
        self.value = 1

    @property
    def state(self):
        return (self.value,)

    @state.setter
    def state(self, state):
        self.value = state[0]

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.state = state
        return obj


class FakeMX:
    contiguous = staticmethod(np.ascontiguousarray)
    concatenate = staticmethod(np.concatenate)
    eval = staticmethod(lambda value: None)
    clear_cache = staticmethod(lambda: None)


def fake_save(path, caches):
    Path(path).write_bytes(pickle.dumps(caches, protocol=pickle.HIGHEST_PROTOCOL))


def fake_load(path):
    return copy.deepcopy(pickle.loads(Path(path).read_bytes()))


def _quant_cache(tokens: int, start: int = 0) -> QuantizedKVCache:
    cache = QuantizedKVCache(group_size=64, bits=4)
    # Mimic the three packed arrays for each of keys and values. In the real
    # MLX cache, all packed arrays have sequence at axis -2.
    a = np.arange(tokens * 2, dtype=np.uint32).reshape(1, 1, tokens, 2) + start
    b = np.arange(tokens, dtype=np.float16).reshape(1, 1, tokens, 1) + start
    c = np.zeros_like(b)
    cache.keys = (a.copy(), b.copy(), c.copy())
    cache.values = (a.copy() + 10, b.copy() + 10, c.copy())
    cache.offset = tokens
    return cache


def test_incremental_pages_restore_kv_and_latest_recurrent_state(tmp_path: Path):
    identity = {
        "model_id": "/fake/qwen",
        "model_fingerprint": "abc123",
        "context_window": 262144,
        "kv_bits": 4,
        "kv_group_size": 64,
        "quantized_kv_start": 0,
    }
    store = IncrementalPrefillStore(tmp_path, identity, min_tokens=2)
    mx = FakeMX()

    base = [_quant_cache(4), ArraysCache(4)]
    assert store.save(base, [1, 2, 3, 4], mx=mx, save_prompt_cache=fake_save)

    # Extend the full-attention cache by four tokens and advance recurrent state.
    extended = [_quant_cache(8), ArraysCache(8)]
    assert store.save(extended, list(range(1, 9)), mx=mx, save_prompt_cache=fake_save)

    manifest = store._read_manifest()
    assert manifest is not None
    assert manifest["mode"] == "incremental-kv-pages"
    assert len(manifest["pages"]) == 1
    page_file = tmp_path / "prefill-pages" / manifest["pages"][0]["file"]
    page = fake_load(page_file)
    assert page[0].offset == 4  # only the delta, not all eight tokens

    restored = store.load_for_prefix(list(range(1, 13)), mx=mx, load_prompt_cache=fake_load)
    assert restored is not None
    cache, token_ids = restored
    assert token_ids == list(range(1, 9))
    assert cache[0].offset == 8
    for expected, actual in zip(extended[0].keys, cache[0].keys):
        np.testing.assert_array_equal(expected, actual)
    assert int(cache[1].cache[0][0]) == 8


def test_partial_checkpoint_is_only_used_for_matching_prefix(tmp_path: Path):
    store = IncrementalPrefillStore(
        tmp_path,
        {
            "model_id": "qwen",
            "model_fingerprint": "f",
            "context_window": 16384,
            "kv_bits": 4,
            "kv_group_size": 64,
            "quantized_kv_start": 0,
        },
        min_tokens=1,
    )
    cache = [_quant_cache(3), ArraysCache(3)]
    mx = FakeMX()
    assert store.save(cache, [7, 8, 9], mx=mx, save_prompt_cache=fake_save)
    assert store.matching_token_ids([7, 8, 9, 10]) == [7, 8, 9]
    assert store.matching_token_ids([7, 8, 0, 10]) is None


def test_unknown_cache_type_uses_safe_full_snapshot_fallback(tmp_path: Path):
    store = IncrementalPrefillStore(
        tmp_path,
        {
            "model_id": "qwen",
            "model_fingerprint": "f",
            "context_window": 16384,
            "kv_bits": 4,
            "kv_group_size": 64,
            "quantized_kv_start": 0,
        },
        min_tokens=1,
    )
    mx = FakeMX()
    cache = [UnknownCache()]
    assert store.save(cache, [1, 2], mx=mx, save_prompt_cache=fake_save)
    manifest = store._read_manifest()
    assert manifest is not None
    assert manifest["mode"] == "full-base-fallback"
    assert manifest["pages"] == []


def test_atomic_cache_save_keeps_safetensors_suffix_and_renames(tmp_path: Path):
    target = tmp_path / "prefill-pages" / "base-test.safetensors"
    target.parent.mkdir(parents=True)
    observed = {}

    def suffix_sensitive_save(path, cache):
        observed["path"] = Path(path)
        # Mirrors MLX safetensors writers: the supplied filename must retain
        # the .safetensors suffix.
        assert Path(path).suffix == ".safetensors"
        Path(path).write_bytes(b"cache-payload")

    IncrementalPrefillStore._atomic_save_cache(target, [object()], suffix_sensitive_save)

    assert observed["path"] != target
    assert target.read_bytes() == b"cache-payload"
    assert not observed["path"].exists()
    assert sorted(item.name for item in target.parent.iterdir()) == ["base-test.safetensors"]


def test_atomic_cache_save_cleans_temp_when_writer_fails(tmp_path: Path):
    target = tmp_path / "base-test.safetensors"

    def bad_save(path, cache):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated writer failure")

    try:
        IncrementalPrefillStore._atomic_save_cache(target, [object()], bad_save)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected writer failure to propagate")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
