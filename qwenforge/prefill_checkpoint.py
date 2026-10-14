"""Incremental SSD checkpoints for partially prefilling MLX-LM caches.

This store is for *restart/resume* after a failed or interrupted prefill. It does
not page live K/V tensors out of Metal memory while attention is running; MLX's
attention kernels still require the active cache in unified memory.

For Qwen hybrid caches, each transaction stores one complete base cache. Later
checkpoints append only new tokens' KV slices and atomically replace the latest
fixed-size recurrent-state snapshot (ArraysCache). The manifest is committed
last, so an interrupted disk write leaves the previous checkpoint usable.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

STORE_VERSION = 1
_KV_TYPES = {"KVCache", "QuantizedKVCache"}
_RECURRENT_TYPES = {"ArraysCache"}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _tree_slice_sequence(tree: Any, start: int, end: int, mx: Any) -> Any:
    if tree is None:
        return None
    if isinstance(tree, tuple):
        return tuple(_tree_slice_sequence(item, start, end, mx) for item in tree)
    if isinstance(tree, list):
        return [_tree_slice_sequence(item, start, end, mx) for item in tree]
    # Both KVCache and every packed tensor inside QuantizedKVCache place the
    # sequence axis at -2: [batch, heads, tokens, feature/packed-feature].
    result = tree[..., start:end, :]
    contiguous = getattr(mx, "contiguous", None)
    return contiguous(result) if callable(contiguous) else result


def _tree_trim_sequence(tree: Any, end: int, mx: Any) -> Any:
    return _tree_slice_sequence(tree, 0, end, mx)


def _tree_concat_sequence(left: Any, right: Any, mx: Any) -> Any:
    if left is None:
        return right
    if right is None:
        return left
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            raise ValueError("KV cache tuple state shape changed between SSD pages")
        return tuple(_tree_concat_sequence(a, b, mx) for a, b in zip(left, right))
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("KV cache list state shape changed between SSD pages")
        return [_tree_concat_sequence(a, b, mx) for a, b in zip(left, right)]
    return mx.concatenate([left, right], axis=-2)


def _cache_offset(cache: Any) -> int | None:
    if type(cache).__name__ in _KV_TYPES and hasattr(cache, "offset"):
        try:
            return int(cache.offset)
        except (TypeError, ValueError):
            return None
    return None


def _valid_cache_types(cache: list[Any]) -> bool:
    # The target Qwen3.8-27B MLX model exposes ArraysCache for recurrent layers
    # and KVCache/QuantizedKVCache for full-attention layers. Unknown cache
    # types deliberately fall back to full snapshots rather than risk corruption.
    return all(type(layer).__name__ in (_KV_TYPES | _RECURRENT_TYPES) for layer in cache)


class IncrementalPrefillStore:
    """Persist resumable partial-prefill state using base snapshots + KV deltas."""

    def __init__(self, root: str | Path | None, identity: dict[str, Any], min_tokens: int = 2048):
        self.root = Path(root).expanduser().resolve() if root else None
        self.identity = dict(identity)
        self.min_tokens = int(min_tokens)

    @property
    def manifest_path(self) -> Path | None:
        return self.root / "prefill-active.json" if self.root else None

    @property
    def pages_dir(self) -> Path | None:
        return self.root / "prefill-pages" if self.root else None

    def matching_token_ids(self, prompt_ids: list[int]) -> list[int] | None:
        """Return the durable prefix if it matches this prompt and can be resumed."""
        path = self.manifest_path
        if path is None or not path.is_file():
            return None
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("version") != STORE_VERSION:
                return None
            if any(manifest.get(key) != value for key, value in self.identity.items()):
                return None
            raw = manifest.get("cache_token_ids")
            if not isinstance(raw, list) or not raw or len(raw) >= len(prompt_ids):
                return None
            tokens = [int(item) for item in raw]
            if prompt_ids[: len(tokens)] != tokens:
                return None
            base = manifest.get("base_cache_file")
            if not isinstance(base, str) or Path(base).name != base:
                return None
            if self.pages_dir is None or not (self.pages_dir / base).is_file():
                return None
            for page in manifest.get("pages", []):
                name = page.get("file") if isinstance(page, dict) else None
                if not isinstance(name, str) or Path(name).name != name or not (self.pages_dir / name).is_file():
                    return None
            recurrent = manifest.get("recurrent_state_file")
            if recurrent is not None:
                if not isinstance(recurrent, str) or Path(recurrent).name != recurrent:
                    return None
                if not (self.pages_dir / recurrent).is_file():
                    return None
            return tokens
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _atomic_save_cache(path: Path, cache: list[Any], save_prompt_cache: Callable[..., Any]) -> None:
        # Keep the .safetensors extension on the temporary path. MLX-LM's
        # save_prompt_cache/mx.save_safetensors path handling expects this
        # extension; using `name.safetensors.tmp` may cause the writer to emit
        # a differently-suffixed file, leaving the path passed to os.replace
        # missing. Unique temp names also avoid collisions between concurrent
        # checkpoint attempts.
        tmp = path.with_name(f"{path.stem}.tmp-{uuid.uuid4().hex}{path.suffix}")
        try:
            save_prompt_cache(str(tmp), cache)
            if not tmp.is_file():
                raise FileNotFoundError(
                    f"cache writer returned without creating its temporary file: {tmp}"
                )
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _read_manifest(self) -> dict[str, Any] | None:
        path = self.manifest_path
        if path is None or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("version") != STORE_VERSION:
                return None
            if any(value.get(key) != val for key, val in self.identity.items()):
                return None
            return value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def save(
        self,
        cache: list[Any],
        token_ids: list[int],
        *,
        mx: Any,
        save_prompt_cache: Callable[..., Any],
    ) -> bool:
        if self.root is None or self.pages_dir is None or self.manifest_path is None:
            return False
        if len(token_ids) < self.min_tokens:
            return False
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        cache_types = [type(layer).__name__ for layer in cache]
        current_offsets = [_cache_offset(layer) for layer in cache]
        old = self._read_manifest()
        old_tokens = [int(x) for x in old.get("cache_token_ids", [])] if old else []
        old_types = old.get("cache_types") if old else None
        can_append = bool(
            old
            and old_types == cache_types
            and len(old_tokens) < len(token_ids)
            and token_ids[: len(old_tokens)] == old_tokens
            and isinstance(old.get("kv_offsets"), list)
            and len(old["kv_offsets"]) == len(cache)
            and _valid_cache_types(cache)
        )
        if old and old_tokens == token_ids and old_types == cache_types:
            return True

        # A model with a cache class we don't explicitly understand still gets
        # recoverable prefill checkpoints, but as full snapshots instead of deltas.
        if not can_append:
            stamp = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
            base_name = f"base-{stamp}.safetensors"
            try:
                self._atomic_save_cache(self.pages_dir / base_name, cache, save_prompt_cache)
                manifest = {
                    "version": STORE_VERSION,
                    **self.identity,
                    "base_cache_file": base_name,
                    "base_token_ids": list(token_ids),
                    "cache_token_ids": list(token_ids),
                    "cache_types": cache_types,
                    "kv_offsets": current_offsets,
                    "pages": [],
                    "recurrent_layer_indices": [],
                    "recurrent_state_file": None,
                    "saved_at": time.time(),
                    "mode": "full-base-fallback" if not _valid_cache_types(cache) else "incremental-base",
                }
                _atomic_json(self.manifest_path, manifest)
                self._prune(manifest)
                if not _valid_cache_types(cache):
                    print(
                        "WARNING: this MLX model uses an unknown cache type; partial-prefill recovery is using "
                        "full snapshots instead of incremental KV pages.",
                        flush=True,
                    )
                else:
                    print(
                        f"Saved initial partial-prefill base at {len(token_ids):,} tokens; subsequent KV writes are incremental.",
                        flush=True,
                    )
                return True
            except Exception as exc:
                print(f"WARNING: partial-prefill base checkpoint failed: {exc}", flush=True)
                return False

        old_offsets = old["kv_offsets"]
        # If any offset regresses or a supported KV layer changed implementation,
        # start a new base from the current known-good cache instead of appending
        # incompatible delta pages.
        reset = False
        kv_layers: list[int] = []
        delta_caches: list[Any] = []
        starts: dict[str, int] = {}
        ends: dict[str, int] = {}
        for index, layer in enumerate(cache):
            name = type(layer).__name__
            if name not in _KV_TYPES:
                continue
            start = old_offsets[index]
            end = _cache_offset(layer)
            if start is None or end is None or end < int(start):
                reset = True
                break
            start = int(start)
            end = int(end)
            if end == start:
                continue
            if name == "QuantizedKVCache":
                state = layer.state
                keys, values, _offset, group_size, bits = state
                delta_keys = _tree_slice_sequence(keys, start, end, mx)
                delta_values = _tree_slice_sequence(values, start, end, mx)
                delta_state = (delta_keys, delta_values, end - start, group_size, bits)
            else:
                keys, values, _offset = layer.state
                delta_keys = _tree_slice_sequence(keys, start, end, mx)
                delta_values = _tree_slice_sequence(values, start, end, mx)
                delta_state = (delta_keys, delta_values, end - start)
            delta_caches.append(type(layer).from_state(delta_state))
            kv_layers.append(index)
            starts[str(index)] = start
            ends[str(index)] = end

        if reset:
            # Rare cache-type/offset transition: safely reset to a new full base.
            return self._save_forced_base(cache, token_ids, cache_types, current_offsets, save_prompt_cache)

        stamp = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        new_page_name: str | None = None
        new_recurrent_name: str | None = None
        try:
            pages = list(old.get("pages", []))
            if delta_caches:
                new_page_name = f"kv-page-{stamp}.safetensors"
                self._atomic_save_cache(self.pages_dir / new_page_name, delta_caches, save_prompt_cache)
                pages.append({
                    "file": new_page_name,
                    "layer_indices": kv_layers,
                    "start_offsets": starts,
                    "end_offsets": ends,
                })

            recurrent_indices = [i for i, layer in enumerate(cache) if type(layer).__name__ in _RECURRENT_TYPES]
            recurrent_name = old.get("recurrent_state_file")
            if recurrent_indices:
                new_recurrent_name = f"recurrent-{stamp}.safetensors"
                recurrent_caches = [cache[i] for i in recurrent_indices]
                self._atomic_save_cache(self.pages_dir / new_recurrent_name, recurrent_caches, save_prompt_cache)
                recurrent_name = new_recurrent_name
            elif not recurrent_name:
                recurrent_indices = []

            manifest = {
                "version": STORE_VERSION,
                **self.identity,
                "base_cache_file": old["base_cache_file"],
                "base_token_ids": old.get("base_token_ids", []),
                "cache_token_ids": list(token_ids),
                "cache_types": cache_types,
                "kv_offsets": current_offsets,
                "pages": pages,
                "recurrent_layer_indices": recurrent_indices or old.get("recurrent_layer_indices", []),
                "recurrent_state_file": recurrent_name,
                "saved_at": time.time(),
                "mode": "incremental-kv-pages",
            }
            _atomic_json(self.manifest_path, manifest)
            self._prune(manifest)
            print(
                f"Saved partial-prefill checkpoint at {len(token_ids):,} tokens "
                f"({len(pages)} incremental KV page(s)); only new KV slices were written.",
                flush=True,
            )
            return True
        except Exception as exc:
            for name in (new_page_name, new_recurrent_name):
                if name:
                    try:
                        (self.pages_dir / name).unlink(missing_ok=True)
                    except OSError:
                        pass
            print(f"WARNING: incremental partial-prefill checkpoint failed: {exc}", flush=True)
            return False

    def _save_forced_base(
        self,
        cache: list[Any],
        token_ids: list[int],
        cache_types: list[str],
        offsets: list[int | None],
        save_prompt_cache: Callable[..., Any],
    ) -> bool:
        assert self.pages_dir is not None and self.manifest_path is not None
        stamp = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        base_name = f"base-{stamp}.safetensors"
        try:
            self._atomic_save_cache(self.pages_dir / base_name, cache, save_prompt_cache)
            manifest = {
                "version": STORE_VERSION,
                **self.identity,
                "base_cache_file": base_name,
                "base_token_ids": list(token_ids),
                "cache_token_ids": list(token_ids),
                "cache_types": cache_types,
                "kv_offsets": offsets,
                "pages": [],
                "recurrent_layer_indices": [],
                "recurrent_state_file": None,
                "saved_at": time.time(),
                "mode": "incremental-reset-base",
            }
            _atomic_json(self.manifest_path, manifest)
            self._prune(manifest)
            print("Cache layout changed; started a fresh SSD partial-prefill base snapshot.", flush=True)
            return True
        except Exception as exc:
            print(f"WARNING: partial-prefill base checkpoint failed: {exc}", flush=True)
            return False

    def load_for_prefix(
        self,
        prompt_ids: list[int],
        *,
        mx: Any,
        load_prompt_cache: Callable[..., Any],
    ) -> tuple[list[Any], list[int]] | None:
        manifest_path = self.manifest_path
        pages_dir = self.pages_dir
        tokens = self.matching_token_ids(prompt_ids)
        if manifest_path is None or pages_dir is None or tokens is None:
            return None
        try:
            manifest = self._read_manifest()
            if manifest is None:
                return None
            base_file = manifest["base_cache_file"]
            cache = load_prompt_cache(str(pages_dir / base_file))
            for page in manifest.get("pages", []):
                delta_caches = load_prompt_cache(str(pages_dir / page["file"]))
                layer_indices = page.get("layer_indices", [])
                if len(delta_caches) != len(layer_indices):
                    raise ValueError("KV page layer index count does not match page cache count")
                for index, delta in zip(layer_indices, delta_caches):
                    index = int(index)
                    current = cache[index]
                    expected_start = int(page["start_offsets"][str(index)])
                    expected_end = int(page["end_offsets"][str(index)])
                    if _cache_offset(current) != expected_start:
                        raise ValueError(
                            f"KV page offset mismatch at layer {index}: cache has {_cache_offset(current)}, "
                            f"page expects {expected_start}"
                        )
                    self._append_delta(current, delta, mx)
                    if _cache_offset(current) != expected_end:
                        raise ValueError(f"KV page restore ended at the wrong offset for layer {index}")
                    # Evaluate one layer at a time so restoration does not keep a
                    # graph of every concatenation alive on the unified-memory heap.
                    mx.eval(current.state)
                    mx.clear_cache()

            recurrent_file = manifest.get("recurrent_state_file")
            recurrent_indices = manifest.get("recurrent_layer_indices", [])
            if recurrent_file:
                recurrent_caches = load_prompt_cache(str(pages_dir / recurrent_file))
                if len(recurrent_caches) != len(recurrent_indices):
                    raise ValueError("Recurrent-state checkpoint layer index count mismatch")
                for index, recurrent_cache in zip(recurrent_indices, recurrent_caches):
                    cache[int(index)].state = recurrent_cache.state
                mx.eval([cache[int(index)].state for index in recurrent_indices])
                mx.clear_cache()

            expected_offsets = manifest.get("kv_offsets", [])
            for index, expected in enumerate(expected_offsets):
                if expected is not None and _cache_offset(cache[index]) != int(expected):
                    raise ValueError(f"restored KV cache offset mismatch at layer {index}")
            return cache, tokens
        except Exception as exc:
            print(f"WARNING: ignoring incomplete/corrupt partial-prefill checkpoint: {exc}", flush=True)
            return None

    @staticmethod
    def _append_delta(current: Any, delta: Any, mx: Any) -> None:
        if type(current).__name__ != type(delta).__name__:
            raise ValueError("KV cache class changed between base and incremental page")
        name = type(current).__name__
        if name == "QuantizedKVCache":
            ck, cv, co, cgs, cbits = current.state
            dk, dv, do, dgs, dbits = delta.state
            if (cgs, cbits) != (dgs, dbits):
                raise ValueError("Quantized KV cache settings differ across pages")
            keys = _tree_concat_sequence(_tree_trim_sequence(ck, int(co), mx), dk, mx)
            values = _tree_concat_sequence(_tree_trim_sequence(cv, int(co), mx), dv, mx)
            current.state = (keys, values, int(co) + int(do), cgs, cbits)
            return
        if name == "KVCache":
            ck, cv, co = current.state
            dk, dv, do = delta.state
            keys = _tree_concat_sequence(_tree_trim_sequence(ck, int(co), mx), dk, mx)
            values = _tree_concat_sequence(_tree_trim_sequence(cv, int(co), mx), dv, mx)
            current.state = (keys, values, int(co) + int(do))
            return
        raise ValueError(f"Unsupported incremental KV cache class: {name}")

    def _prune(self, manifest: dict[str, Any]) -> None:
        if self.pages_dir is None:
            return
        keep = {manifest.get("base_cache_file"), manifest.get("recurrent_state_file")}
        keep.update(page.get("file") for page in manifest.get("pages", []) if isinstance(page, dict))
        for path in self.pages_dir.glob("*.safetensors"):
            if path.name not in keep:
                try:
                    path.unlink()
                except OSError:
                    pass

    def clear(self) -> None:
        if self.pages_dir:
            for path in self.pages_dir.glob("*.safetensors"):
                try:
                    path.unlink()
                except OSError:
                    pass
            for path in self.pages_dir.glob("*.tmp"):
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                self.pages_dir.rmdir()
            except OSError:
                pass
        if self.manifest_path:
            try:
                self.manifest_path.unlink(missing_ok=True)
                self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp").unlink(missing_ok=True)
            except OSError:
                pass
