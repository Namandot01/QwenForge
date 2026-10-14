"""Native MLX-LM Q4-KV local server for Qwen3.8 text-only checkpoints.

This module intentionally uses upstream mlx-lm's generation/cache APIs. It is
not a custom Metal attention kernel; the engine owns request handling, cache
reuse/persistence, context enforcement, and diagnostics around that runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import threading
import time
import uuid
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from contextlib import nullcontext
from urllib.parse import urlparse

from .prefill_schedule import adaptive_prefill_chunk_size
from .prefill_checkpoint import IncrementalPrefillStore


def _has_repeated_token_cycle(token_ids: list[int]) -> bool:
    """Return True only for a sustained exact token cycle.

    The old three-repeat / eight-token threshold caused false positives in
    repetitive HTML, CSS, lists, and other structured output. Require at least
    128 identical repeated token positions, a block of >=16 tokens, and 6-8
    consecutive copies depending on block size. This still detects sustained
    degenerate loops while allowing ordinary repeated markup to continue.
    """
    minimum_repeated_tokens = 128
    for block_size in (16, 20, 24, 28, 32, 40, 48, 56, 64):
        repetitions = max(6, (minimum_repeated_tokens + block_size - 1) // block_size)
        needed = block_size * repetitions
        if len(token_ids) < needed:
            continue
        tail = token_ids[-needed:]
        first = tail[:block_size]
        if all(tail[i : i + block_size] == first for i in range(block_size, needed, block_size)):
            return True
    return False


class EngineConfigurationError(RuntimeError):
    """Raised for a model/runtime configuration that cannot be started."""


def _apply_request_reasoning_effort(request: dict[str, Any], template_kwargs: dict[str, Any]) -> None:
    """Normalize common OpenAI/OpenCode reasoning-effort request shapes.

    Older OpenCode releases using @ai-sdk/openai-compatible can serialize a
    selected model variant as `options: {reasoning_effort: ...}` instead of a
    top-level OpenAI-style field. Read both forms and pass the selected tier to
    the model's chat template. Explicit chat_template_kwargs take precedence.
    """
    keys = ("reasoning_effort", "reasoningEffort")
    effort: Any = None
    found = False

    for key in keys:
        if key in template_kwargs:
            effort = template_kwargs[key]
            found = True
            break

    if not found:
        for key in keys:
            if key in request:
                effort = request[key]
                found = True
                break

    if not found:
        for container_key in ("options", "providerOptions", "provider_options"):
            container = request.get(container_key)
            if not isinstance(container, dict):
                continue
            for key in keys:
                if key in container:
                    effort = container[key]
                    found = True
                    break
            if found:
                break
            reasoning = container.get("reasoning")
            if isinstance(reasoning, dict) and "effort" in reasoning:
                effort = reasoning["effort"]
                found = True
                break

    if not found:
        return

    if not isinstance(effort, str):
        raise ValueError("reasoning_effort must be one of: low, medium, xhigh")
    effort = effort.lower()
    if effort not in {"low", "medium", "xhigh"}:
        raise ValueError(
            "Qwen3.8 reasoning_effort must be one of: low, medium, xhigh; "
            "OpenCode's generic 'high' tier is not supported by this model template."
        )
    template_kwargs["reasoning_effort"] = effort



def _reasoning_effort_from_model_alias(requested_model: Any, base_model: str) -> str | None:
    """Return an effort encoded in a QwenForge model alias.

    This gives OpenCode clients a schema-safe alternative to model variants: use
    <base>-low, <base>-medium, or <base>-xhigh as ordinary model IDs.
    """
    if not isinstance(requested_model, str):
        return None
    prefix = f"{base_model}-"
    if not requested_model.startswith(prefix):
        return None
    suffix = requested_model[len(prefix):].lower()
    return suffix if suffix in {"low", "medium", "xhigh"} else None

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"


def _tool_definitions_by_name(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index OpenAI-style function tool definitions by function name."""
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", tool)
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            result[name] = function
    return result


def _normalize_messages_for_qwen(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve tool calls and reasoning while adapting OpenAI messages to Qwen's template."""
    normalized: list[dict[str, Any]] = []
    valid_roles = {"system", "user", "assistant", "tool"}
    for item in messages:
        if not isinstance(item, dict) or item.get("role") not in valid_roles:
            raise ValueError("messages must contain system, user, assistant, or tool role objects")
        role = item["role"]
        content = item.get("content", "")
        if content is None and role == "assistant":
            content = ""
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, str):
                    chunks.append(part)
                elif isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}:
                    value = part.get("text", "")
                    if isinstance(value, str):
                        chunks.append(value)
                else:
                    raise ValueError("QwenForge's current checkpoint is text-only; unsupported message content part")
            content = "".join(chunks)
        if not isinstance(content, str):
            raise ValueError("message content must be text (or text-only content parts)")
        message: dict[str, Any] = {"role": role, "content": content}
        reasoning = item.get("reasoning_content", item.get("reasoning"))
        if isinstance(reasoning, str):
            message["reasoning_content"] = reasoning
        if role == "tool":
            if isinstance(item.get("name"), str):
                message["name"] = item["name"]
            if isinstance(item.get("tool_call_id"), str):
                message["tool_call_id"] = item["tool_call_id"]
        if role == "assistant" and isinstance(item.get("tool_calls"), list):
            calls = []
            for call in item["tool_calls"]:
                if not isinstance(call, dict):
                    continue
                function = call.get("function", call)
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    continue
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    except json.JSONDecodeError:
                        arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                # Qwen's chat template expects {name, arguments: dict}, not the
                # OpenAI wrapper {id, type, function: {name, arguments: JSON-string}}.
                calls.append({"name": function["name"], "arguments": arguments})
            if calls:
                message["tool_calls"] = calls
        normalized.append(message)
    return normalized


def _coerce_tool_argument(raw: str, schema: dict[str, Any] | None) -> Any:
    """Convert the Qwen XML-ish parameter payload to the schema's JSON type."""
    value = raw.strip()
    expected = schema.get("type") if isinstance(schema, dict) else None
    if expected == "string":
        return value
    if expected in {"object", "array"}:
        try:
            parsed = json.loads(value)
            if expected == "object" and isinstance(parsed, dict):
                return parsed
            if expected == "array" and isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return {} if expected == "object" else []
    if expected == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    if expected == "number":
        try:
            return float(value)
        except ValueError:
            return value
    if expected == "boolean":
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        return value
    # When schemas omit a type, preserve strings except for unambiguous JSON objects/arrays.
    if value.startswith(("{", "[")):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass
    return value


def _parse_qwen_tool_call(block: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Parse one Qwen <tool_call><function=...> block into OpenAI's tool_call shape."""
    match = re.search(r"<function=([^>\s]+)>(.*?)</function>", block, flags=re.DOTALL)
    if not match:
        return None
    name, body = match.group(1).strip(), match.group(2)
    definitions = _tool_definitions_by_name(tools)
    definition = definitions.get(name)
    if definition is None:
        return None
    parameters = definition.get("parameters", {})
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    arguments: dict[str, Any] = {}
    for param in re.finditer(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", body, flags=re.DOTALL):
        key = param.group(1).strip()
        schema = properties.get(key) if isinstance(properties, dict) else None
        arguments[key] = _coerce_tool_argument(param.group(2), schema if isinstance(schema, dict) else None)
    return {
        "id": "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class QwenOutputParser:
    """Split Qwen's emitted <think> region and parse its native tool-call blocks."""

    def __init__(self, *, thinking_enabled: bool, tools: list[dict[str, Any]] | None = None) -> None:
        self.in_thinking = bool(thinking_enabled)
        self.tools = tools or []
        self.buffer = ""
        self.tool_buffer: str | None = None

    def feed(self, text: str) -> list[tuple[str, Any]]:
        self.buffer += text
        return self._drain(final=False)

    def finish(self) -> list[tuple[str, Any]]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[tuple[str, Any]]:
        events: list[tuple[str, Any]] = []
        markers = [_THINK_OPEN, _THINK_CLOSE]
        if self.tools:
            markers.append(_TOOL_OPEN)
        while True:
            if self.tool_buffer is not None:
                close_at = self.buffer.find(_TOOL_CLOSE)
                if close_at < 0:
                    if final:
                        # Do not fabricate/execute an incomplete tool call. Expose it for diagnosis.
                        raw = self.tool_buffer + self.buffer
                        events.append(("reasoning_content" if self.in_thinking else "content", raw))
                        self.tool_buffer = None
                        self.buffer = ""
                    else:
                        self.tool_buffer += self.buffer
                        self.buffer = ""
                    break
                end = close_at + len(_TOOL_CLOSE)
                self.tool_buffer += self.buffer[:end]
                self.buffer = self.buffer[end:]
                call = _parse_qwen_tool_call(self.tool_buffer, self.tools)
                if call is None:
                    events.append(("reasoning_content" if self.in_thinking else "content", self.tool_buffer))
                else:
                    events.append(("tool_call", call))
                self.tool_buffer = None
                continue

            positions = [(self.buffer.find(marker), marker) for marker in markers]
            positions = [(pos, marker) for pos, marker in positions if pos >= 0]
            if positions:
                pos, marker = min(positions, key=lambda pair: pair[0])
                prefix = self.buffer[:pos]
                if prefix:
                    events.append(("reasoning_content" if self.in_thinking else "content", prefix))
                self.buffer = self.buffer[pos + len(marker):]
                if marker == _THINK_OPEN:
                    self.in_thinking = True
                elif marker == _THINK_CLOSE:
                    self.in_thinking = False
                elif marker == _TOOL_OPEN:
                    self.tool_buffer = marker
                continue

            if final:
                if self.buffer:
                    events.append(("reasoning_content" if self.in_thinking else "content", self.buffer))
                self.buffer = ""
                break

            # Retain only a suffix that could be the beginning of a marker arriving next chunk.
            hold = 0
            for marker in markers:
                max_test = min(len(marker) - 1, len(self.buffer))
                for size in range(max_test, 0, -1):
                    if self.buffer.endswith(marker[:size]):
                        hold = max(hold, size)
                        break
            safe_len = len(self.buffer) - hold
            if safe_len > 0:
                safe = self.buffer[:safe_len]
                events.append(("reasoning_content" if self.in_thinking else "content", safe))
                self.buffer = self.buffer[safe_len:]
            break
        return events


# OpenCode has an internal stream-idle watchdog independent of provider chunkTimeout.
# Qwen tool-call XML is buffered until </tool_call>, so very large write/edit arguments
# can otherwise produce no semantic stream events for >60 seconds. Emit an invisible
# reasoning delta periodically only while that block is being assembled.
_TOOL_CALL_HEARTBEAT_SECONDS = 15.0
_TOOL_CALL_HEARTBEAT_TEXT = "\u2063"  # INVISIBLE SEPARATOR: non-empty SSE delta, no visible text

def _maybe_send_tool_call_heartbeat(
    parser: "QwenOutputParser",
    last_sent: float,
    send_delta: Any,
    *,
    now: float | None = None,
) -> float:
    """Keep OpenCode's semantic stream alive while a native tool call is buffered.

    Returns the timestamp to use for the next heartbeat interval. When the parser is
    outside a tool call, the timer is reset so ordinary generation never gets heartbeats.
    """
    current = time.monotonic() if now is None else now
    if parser.tool_buffer is None:
        return current
    if current - last_sent >= _TOOL_CALL_HEARTBEAT_SECONDS:
        send_delta({"reasoning_content": _TOOL_CALL_HEARTBEAT_TEXT})
        return current
    return last_sent


def _model_fingerprint(model_id: str) -> str:
    """Fingerprint local checkpoint metadata without hashing tens of GB of weights."""
    path = Path(model_id).expanduser()
    payload: dict[str, Any] = {"model_id": model_id}
    if path.is_dir():
        files = []
        for item in sorted(path.glob("*.safetensors")):
            try:
                stat = item.stat()
                files.append((item.name, stat.st_size, stat.st_mtime_ns))
            except OSError:
                continue
        try:
            stat = (path / "config.json").stat()
            config_stamp = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            config_stamp = None
        payload.update({"files": files, "config": config_stamp})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class NativeMLXEngine:
    """Single-session native MLX-LM engine with Q4/Q8 KV cache and prefix reuse."""

    def __init__(
        self,
        model_id: str,
        *,
        context_window: int = 262_144,
        kv_bits: int = 4,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        prefill_step_size: int = 1024,
        prefill_strategy: str = "adaptive",
        snapshot_dir: str | Path | None = None,
        snapshot_min_tokens: int = 2048,
        prefill_checkpoint_interval: int = 8192,
    ) -> None:
        if platform.system() != "Darwin":
            raise EngineConfigurationError("Native MLX inference requires macOS on Apple Silicon.")
        if platform.machine().lower() not in {"arm64", "aarch64"}:
            raise EngineConfigurationError("Native MLX inference requires an Apple Silicon CPU (arm64).")
        if context_window < 1 or context_window > 262_144:
            raise ValueError("context_window must be between 1 and the model-native 262,144-token limit")
        if kv_bits not in (4, 8):
            raise ValueError("kv_bits must be 4 or 8 for this build")
        if kv_group_size not in (32, 64, 128):
            raise ValueError("kv_group_size must be 32, 64, or 128")
        if quantized_kv_start < 0:
            raise ValueError("quantized_kv_start cannot be negative")
        if prefill_step_size < 32:
            raise ValueError("prefill_step_size must be at least 32 tokens")
        if prefill_strategy not in {"adaptive", "fixed"}:
            raise ValueError("prefill_strategy must be adaptive or fixed")
        if snapshot_min_tokens < 1:
            raise ValueError("snapshot_min_tokens must be at least 1 token")
        if prefill_checkpoint_interval < 256:
            raise ValueError("prefill_checkpoint_interval must be at least 256 tokens")

        try:
            import mlx.core as mx  # type: ignore
            from mlx_lm import load  # type: ignore
            from mlx_lm.models.cache import make_prompt_cache  # type: ignore
        except Exception as exc:
            raise EngineConfigurationError(
                "Could not import mlx/ mlx-lm. Activate QwenForge's virtual environment and install mlx-lm."
            ) from exc

        self.mx = mx
        self._make_prompt_cache = make_prompt_cache
        self.model_id = model_id
        self.model_fingerprint = _model_fingerprint(model_id)
        self.context_window = context_window
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        self.prefill_step_size = prefill_step_size
        self.prefill_strategy = prefill_strategy
        self.snapshot_dir = Path(snapshot_dir).expanduser().resolve() if snapshot_dir else None
        self.snapshot_min_tokens = int(snapshot_min_tokens)
        self.prefill_checkpoint_interval = int(prefill_checkpoint_interval)
        checkpoint_identity = {
            "model_id": self.model_id,
            "model_fingerprint": self.model_fingerprint,
            "context_window": self.context_window,
            "kv_bits": self.kv_bits,
            "kv_group_size": self.kv_group_size,
            "quantized_kv_start": self.quantized_kv_start,
        }
        self._prefill_store = IncrementalPrefillStore(
            self.snapshot_dir, checkpoint_identity, min_tokens=self.snapshot_min_tokens
        )
        self._snapshot_skip_notice_printed = False
        self._lock = threading.RLock()
        self._cache: Any | None = None
        self._cache_token_ids: list[int] = []
        self._last_stats: dict[str, Any] = {}

        print(f"Loading MLX-LM model: {model_id}", flush=True)
        self.model, self.tokenizer = load(model_id)
        print("Model loaded.", flush=True)
        if self.snapshot_dir:
            self._restore_snapshot()

    @property
    def model_name(self) -> str:
        return Path(self.model_id).expanduser().name or self.model_id.rsplit("/", 1)[-1]

    @property
    def snapshot_file(self) -> Path | None:
        return self.snapshot_dir / "active.json" if self.snapshot_dir else None

    def _restore_snapshot(self) -> None:
        if self.snapshot_dir is None:
            return
        active = self.snapshot_file
        if active is None or not active.exists():
            return
        try:
            from mlx_lm.models.cache import load_prompt_cache  # type: ignore
            metadata = json.loads(active.read_text(encoding="utf-8"))
            expected = {
                "model_id": self.model_id,
                "model_fingerprint": self.model_fingerprint,
                "context_window": self.context_window,
                "kv_bits": self.kv_bits,
                "kv_group_size": self.kv_group_size,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    print(f"Ignoring SSD cache snapshot: {key} does not match this run.", flush=True)
                    return
            cache_name = metadata.get("cache_file")
            if not isinstance(cache_name, str) or Path(cache_name).name != cache_name:
                print("Ignoring SSD cache snapshot: invalid cache filename.", flush=True)
                return
            cache_path = self.snapshot_dir / cache_name
            tokens = metadata.get("cache_token_ids")
            if not cache_path.is_file() or not isinstance(tokens, list):
                print("Ignoring SSD cache snapshot: cache data or token manifest is missing.", flush=True)
                return
            token_ids = [int(token) for token in tokens]
            if not token_ids or len(token_ids) > self.context_window:
                print("Ignoring SSD cache snapshot: token count is invalid.", flush=True)
                return
            self._cache = load_prompt_cache(str(cache_path))
            self._cache_token_ids = token_ids
            print(f"Restored {len(token_ids):,} cached prefix tokens from SSD.", flush=True)
        except Exception as exc:
            self._cache = None
            self._cache_token_ids = []
            print(f"Could not restore SSD cache snapshot (continuing without it): {exc}", flush=True)

    def save_snapshot(self) -> None:
        """Persist the latest usable prefix snapshot, safely serialized against inference."""
        if self.snapshot_dir is None:
            return
        # Generation and cache mutation use the same re-entrant lock. Do not serialize
        # cache arrays while another request could be extending them.
        with self._lock:
            self._save_snapshot_locked()

    def _save_snapshot_locked(self) -> None:
        if self.snapshot_dir is None or self._cache is None or not self._cache_token_ids:
            return
        token_count = len(self._cache_token_ids)
        if token_count < self.snapshot_min_tokens:
            if not self._snapshot_skip_notice_printed:
                print(
                    f"SSD prefix checkpoint deferred: {token_count:,} cached tokens is below "
                    f"--snapshot-min-tokens {self.snapshot_min_tokens:,}; no cache file is expected yet.",
                    flush=True,
                )
                self._snapshot_skip_notice_printed = True
            return
        from mlx_lm.models.cache import save_prompt_cache  # type: ignore

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.time_ns()}"
        cache_name = f"cache-{stamp}.safetensors"
        cache_path = self.snapshot_dir / cache_name
        tmp_cache = self.snapshot_dir / f".{cache_name}.tmp"
        active = self.snapshot_file
        tmp_active = self.snapshot_dir / ".active.json.tmp"
        try:
            print(
                f"Checkpointing {token_count:,} cached prefix tokens to SSD after completed response; "
                "writing a full MLX cache snapshot (not an incremental delta)...",
                flush=True,
            )
            save_prompt_cache(str(tmp_cache), self._cache)
            os.replace(tmp_cache, cache_path)
            metadata = {
                "version": 2,
                "model_id": self.model_id,
                "model_fingerprint": self.model_fingerprint,
                "context_window": self.context_window,
                "kv_bits": self.kv_bits,
                "kv_group_size": self.kv_group_size,
                "cache_file": cache_name,
                "cache_token_ids": self._cache_token_ids,
                "saved_at": time.time(),
            }
            with tmp_active.open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_active, active)
            for old in self.snapshot_dir.glob("cache-*.safetensors"):
                if old != cache_path:
                    try:
                        old.unlink()
                    except OSError:
                        pass
            # A committed full response supersedes any interrupted-prefill chain.
            self._prefill_store.clear()
            self._snapshot_skip_notice_printed = False
            print(f"Saved persistent prefix checkpoint: {cache_path}", flush=True)
        except Exception as exc:
            for leftover in (tmp_cache, tmp_active):
                try:
                    leftover.unlink(missing_ok=True)
                except OSError:
                    pass
            print(f"WARNING: SSD prefix checkpoint was not saved: {exc}", flush=True)

    @staticmethod
    def _is_memory_allocation_error(exc: BaseException) -> bool:
        text = str(exc).lower()
        return any(marker in text for marker in (
            "outofmemory", "out of memory", "insufficient memory", "failed to allocate",
            "allocation failed", "memory allocation", "kio gpu command buffer callback error outofmemory",
        ))

    def _restore_partial_prefill_if_better(self, full_prompt_ids: list[int]) -> bool:
        """Restore the best matching partial-prefill prefix if RAM's cache is unsuitable.

        A longer in-memory cache is only preferable when it is itself a prefix of
        this exact prompt. A stale cache from a different conversation must not
        prevent restoring a shorter but compatible SSD checkpoint.
        """
        if self.snapshot_dir is None:
            return False
        tokens = self._prefill_store.matching_token_ids(full_prompt_ids)
        if tokens is None:
            return False
        current_tokens = list(self._cache_token_ids)
        current_cache_matches = bool(
            self._cache is not None
            and current_tokens
            and len(current_tokens) <= len(full_prompt_ids)
            and full_prompt_ids[:len(current_tokens)] == current_tokens
        )
        if current_cache_matches and len(tokens) <= len(current_tokens):
            return False
        # The SSD cache may be larger than the old RAM-resident prefix. Release the
        # old cache before loading it, otherwise restoration itself can peak at ~2x.
        self._cache = None
        self._cache_token_ids = []
        try:
            self.mx.clear_cache()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            from mlx_lm.models.cache import load_prompt_cache  # type: ignore
            restored = self._prefill_store.load_for_prefix(
                full_prompt_ids, mx=self.mx, load_prompt_cache=load_prompt_cache
            )
            if restored is None:
                self._restore_snapshot()
                return False
            cache, token_ids = restored
            self._cache = cache
            self._cache_token_ids = token_ids
            print(
                f"Resumed partial prefill from SSD at {len(token_ids):,}/{len(full_prompt_ids):,} prompt tokens.",
                flush=True,
            )
            return True
        except Exception as exc:
            print(f"WARNING: could not restore partial-prefill state: {exc}", flush=True)
            self._restore_snapshot()
            return False

    def _restore_best_saved_prefix(self, full_prompt_ids: list[int]) -> tuple[list[Any], list[int]]:
        """Drop a possibly corrupted in-memory cache and reload the longest disk prefix."""
        self._cache = None
        self._cache_token_ids = []
        try:
            self.mx.clear_cache()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass

        # First consider the completed-turn snapshot. It is loaded without assuming
        # it belongs to this prompt; the token-prefix check below decides whether it
        # can be reused.
        self._restore_snapshot()
        active_cache = self._cache
        active_tokens = list(self._cache_token_ids)
        if not active_tokens or len(active_tokens) >= len(full_prompt_ids) or full_prompt_ids[:len(active_tokens)] != active_tokens:
            active_cache, active_tokens = None, []
            self._cache, self._cache_token_ids = None, []

        partial_tokens = self._prefill_store.matching_token_ids(full_prompt_ids) if self.snapshot_dir else None
        if partial_tokens is not None and len(partial_tokens) > len(active_tokens):
            # Release the shorter completed snapshot before rebuilding the longer
            # incremental chain, to avoid doubling the entire cache during restore.
            active_cache = None
            self._cache = None
            self._cache_token_ids = []
            try:
                self.mx.clear_cache()
            except Exception:
                pass
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            try:
                from mlx_lm.models.cache import load_prompt_cache  # type: ignore
                restored = self._prefill_store.load_for_prefix(
                    full_prompt_ids, mx=self.mx, load_prompt_cache=load_prompt_cache
                )
                if restored is not None:
                    active_cache, active_tokens = restored
                    print(
                        f"Recovered {len(active_tokens):,} durable partial-prefill tokens from SSD after allocation failure.",
                        flush=True,
                    )
            except Exception as exc:
                print(f"WARNING: partial-prefill recovery failed: {exc}", flush=True)

        if active_cache is None:
            active_cache = self._make_prompt_cache(self.model)
            active_tokens = []
        self._cache = active_cache
        self._cache_token_ids = list(active_tokens)
        return active_cache, list(active_tokens)

    def _adaptive_prefill(
        self,
        suffix_ids: list[int],
        cache: Any,
        total_prompt_tokens: int,
        full_prompt_ids: list[int] | None = None,
    ):
        """Memory-adaptive prefill with resumable SSD checkpoints.

        Successful passes update the in-memory MLX cache. At configurable token
        intervals, the cache is committed to disk as one base cache plus incremental
        K/V deltas and the latest recurrent state. On a Metal OOM, QwenForge restores
        the latest durable prefix and retries the remaining suffix with smaller
        chunks, down to one token if necessary.

        The disk pages make interrupted prefill resumable; active attention still
        uses MLX's in-memory cache (this is not live disk-paged attention).
        """
        if not suffix_ids:
            raise ValueError("Cannot prefill an empty prompt suffix")
        if full_prompt_ids is None:
            full_prompt_ids = list(suffix_ids)
        if len(suffix_ids) == 1:
            return self.mx.array(suffix_ids, dtype=self.mx.uint32), 0.0

        from mlx_lm.generate import generation_stream as mlx_generation_stream  # type: ignore

        # base_context marks the exact prefix represented by `cache`; suffix_ids is
        # the not-yet-cached part of full_prompt_ids.
        base_context = total_prompt_tokens - len(suffix_ids)
        prefill_ids = suffix_ids[:-1]
        processed = 0
        started = time.perf_counter()
        last_reported = 0
        last_heartbeat = time.monotonic()
        last_checkpointed = base_context
        last_checkpoint_attempt = base_context
        retry_chunk_cap = self.prefill_step_size
        stream_context = self.mx.stream(mlx_generation_stream) if hasattr(self.mx, "stream") else nullcontext()
        with stream_context:
            while processed < len(prefill_ids):
                context_before = base_context + processed
                scheduled_size = adaptive_prefill_chunk_size(
                    context_before, total_prompt_tokens, self.prefill_step_size
                )
                # Normal schedule floors at 32, but an actual OOM triggers retries
                # at 16/8/4/2/1 tokens until it finds a pass size the GPU can handle.
                chunk_size = max(1, min(scheduled_size, retry_chunk_cap))
                chunk_ids = prefill_ids[processed : processed + chunk_size]
                chunk = self.mx.array(chunk_ids, dtype=self.mx.uint32)
                try:
                    self.model(chunk[None], cache=cache)
                    for index, layer_cache in enumerate(cache):
                        if (
                            hasattr(layer_cache, "to_quantized")
                            and getattr(layer_cache, "offset", 0) >= self.quantized_kv_start
                        ):
                            cache[index] = layer_cache.to_quantized(
                                group_size=self.kv_group_size,
                                bits=self.kv_bits,
                            )
                    self.mx.eval([layer_cache.state for layer_cache in cache])
                    self.mx.clear_cache()
                except Exception as exc:
                    if not self._is_memory_allocation_error(exc):
                        raise RuntimeError(
                            f"Adaptive prefill failed at {context_before:,}/{total_prompt_tokens:,} tokens "
                            f"using a {len(chunk_ids)}-token pass: {exc}"
                        ) from exc
                    if chunk_size <= 1:
                        # The failed MLX op may have partially mutated this cache.
                        # Do not leave it reachable as if it still matched the last
                        # committed token prefix; a later request could reuse corrupt state.
                        cache.clear()
                        self._cache = None
                        self._cache_token_ids = []
                        try:
                            self.mx.clear_cache()
                        except Exception:
                            pass
                        try:
                            import gc
                            gc.collect()
                        except Exception:
                            pass
                        raise RuntimeError(
                            f"Metal ran out of memory even with a 1-token prefill pass at "
                            f"{context_before:,}/{total_prompt_tokens:,} prompt tokens. "
                            "The latest durable SSD checkpoint was preserved if one was available. "
                            "This suggests the active weights/cache/attention workspace no longer fits; "
                            "smaller chunks cannot reduce the resident cache itself."
                        ) from exc

                    retry_chunk_cap = max(1, min(retry_chunk_cap - 1, chunk_size // 2))
                    print(
                        f"Metal OOM at {context_before:,}/{total_prompt_tokens:,} prompt tokens "
                        f"with a {chunk_size}-token prefill pass; restoring the last SSD checkpoint "
                        f"and retrying at <= {retry_chunk_cap} tokens.",
                        flush=True,
                    )
                    # The local `cache` list remains a Python reference even if
                    # self._cache is cleared. Empty it before restoration so a failed
                    # Metal allocation cannot leave the old large cache resident while
                    # the disk checkpoint is loaded (which could otherwise peak near 2x).
                    cache.clear()
                    self._cache = None
                    self._cache_token_ids = []
                    try:
                        self.mx.clear_cache()
                    except Exception:
                        pass
                    try:
                        import gc
                        gc.collect()
                    except Exception:
                        pass
                    # Rebuild from a known-good disk prefix, not from a cache that
                    # may have been partially mutated by the failed MLX command.
                    restored_cache, restored_tokens = self._restore_best_saved_prefix(full_prompt_ids)
                    # Mutate the original list passed by stream_responses, preserving the
                    # caller's reference while replacing potentially corrupted cache objects.
                    cache[:] = restored_cache
                    base_context = len(restored_tokens)
                    suffix_ids = full_prompt_ids[base_context:]
                    if len(suffix_ids) <= 1:
                        return self.mx.array(suffix_ids or [full_prompt_ids[-1]], dtype=self.mx.uint32), time.perf_counter() - started
                    prefill_ids = suffix_ids[:-1]
                    processed = 0
                    last_checkpointed = base_context
                    last_checkpoint_attempt = base_context
                    last_reported = 0
                    self._cache = cache
                    self._cache_token_ids = list(restored_tokens)
                    continue

                processed += len(chunk_ids)
                absolute_done = base_context + processed

                if self.snapshot_dir is not None and absolute_done >= self.snapshot_min_tokens:
                    interval_due = absolute_done - last_checkpoint_attempt >= self.prefill_checkpoint_interval
                    first_checkpoint_due = last_checkpointed < self.snapshot_min_tokens <= absolute_done
                    final_checkpoint_due = processed == len(prefill_ids) and absolute_done > last_checkpointed
                    if interval_due or first_checkpoint_due or final_checkpoint_due:
                        try:
                            from mlx_lm.models.cache import save_prompt_cache  # type: ignore
                            did_save = self._prefill_store.save(
                                cache,
                                full_prompt_ids[:absolute_done],
                                mx=self.mx,
                                save_prompt_cache=save_prompt_cache,
                            )
                        except Exception as exc:
                            did_save = False
                            print(f"WARNING: partial-prefill checkpoint failed: {exc}", flush=True)
                        last_checkpoint_attempt = absolute_done
                        if did_save:
                            last_checkpointed = absolute_done
                            self._cache = cache
                            self._cache_token_ids = list(full_prompt_ids[:absolute_done])

                if last_reported == 0 or processed - last_reported >= 8192 or processed == len(prefill_ids):
                    elapsed = max(time.perf_counter() - started, 1e-6)
                    try:
                        peak_gb = float(self.mx.get_peak_memory()) / 1e9
                        memory = f"; MLX peak {peak_gb:.2f} GB"
                    except Exception:
                        memory = ""
                    next_size = min(
                        adaptive_prefill_chunk_size(absolute_done, total_prompt_tokens, self.prefill_step_size),
                        retry_chunk_cap,
                    )
                    print(
                        f"Adaptive prefill: {absolute_done:,}/{total_prompt_tokens:,} tokens; "
                        f"next pass <= {next_size} tokens; {elapsed:.1f}s{memory}",
                        flush=True,
                    )
                    last_reported = processed

                if time.monotonic() - last_heartbeat >= 10.0:
                    last_heartbeat = time.monotonic()
                    yield None

        elapsed = time.perf_counter() - started
        # The final token is intentionally processed by stream_generate, as in
        # MLX-LM's regular prompt-cache contract.
        if self.snapshot_dir is not None:
            absolute_done = base_context + len(prefill_ids)
            if absolute_done >= self.snapshot_min_tokens and absolute_done > last_checkpointed:
                try:
                    from mlx_lm.models.cache import save_prompt_cache  # type: ignore
                    did_save = self._prefill_store.save(
                        cache, full_prompt_ids[:absolute_done], mx=self.mx, save_prompt_cache=save_prompt_cache
                    )
                    if did_save:
                        self._cache = cache
                        self._cache_token_ids = list(full_prompt_ids[:absolute_done])
                except Exception as exc:
                    print(f"WARNING: final partial-prefill checkpoint failed: {exc}", flush=True)
        return self.mx.array([suffix_ids[-1]], dtype=self.mx.uint32), elapsed

    def stream_responses(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        chat_template_kwargs: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[tuple[Any, int, int]]:
        """Yield (MLX-LM GenerationResponse, full prompt token count, output token count)."""
        try:
            from mlx_lm.generate import stream_generate, generation_stream as mlx_generation_stream  # type: ignore
            from mlx_lm.sample_utils import make_sampler  # type: ignore
        except Exception as exc:
            raise EngineConfigurationError("Your installed mlx-lm is missing the expected generation APIs.") from exc

        normalized_messages = _normalize_messages_for_qwen(messages)
        if not normalized_messages:
            raise ValueError("messages cannot be empty")

        template_kwargs = dict(chat_template_kwargs or {})
        if "tokenize" in template_kwargs or "add_generation_prompt" in template_kwargs or "tools" in template_kwargs:
            raise ValueError("Do not set tokenize/add_generation_prompt/tools inside chat_template_kwargs; tools belong in the request's tools field")
        template_args: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            **template_kwargs,
        }
        if tools:
            template_args["tools"] = tools
        prompt = self.tokenizer.apply_chat_template(normalized_messages, **template_args)
        input_ids = [int(token) for token in self.tokenizer.encode(prompt, add_special_tokens=False)]
        if not input_ids:
            raise ValueError("The rendered prompt contains no tokens")
        available = self.context_window - len(input_ids)
        if available <= 0:
            raise ValueError(
                f"Prompt uses {len(input_ids):,} tokens, leaving no room in the {self.context_window:,}-token context"
            )
        max_tokens = max(1, min(int(max_tokens), available))

        with self._lock:
            # If a prior request failed (for example a Metal OOM), recover the most
            # recently committed SSD prefix before deciding to prefill from scratch.
            if self._cache is None and self.snapshot_dir is not None and self.snapshot_file and self.snapshot_file.is_file():
                self._restore_snapshot()
            # A newer, matching partial-prefill checkpoint can be longer than the
            # last completed-turn snapshot. Prefer it when its token IDs match exactly.
            self._restore_partial_prefill_if_better(input_ids)
            reuse = (
                self._cache is not None
                and 0 < len(self._cache_token_ids) < len(input_ids)
                and input_ids[: len(self._cache_token_ids)] == self._cache_token_ids
            )
            if reuse:
                cache = self._cache
                suffix = input_ids[len(self._cache_token_ids) :]
                print(
                    f"Reusing cached prefix: {len(self._cache_token_ids):,} tokens; "
                    f"prefilling {len(suffix):,} new tokens.",
                    flush=True,
                )
            else:
                # Release an unrelated previous context before allocating a fresh one.
                self._cache = None
                self._cache_token_ids = []
                try:
                    self.mx.clear_cache()
                except Exception:
                    pass
                cache = self._make_prompt_cache(self.model)
                suffix = input_ids
                print(f"Fresh prompt: {len(input_ids):,} tokens.", flush=True)
            self._cache = cache

            try:
                sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
                if self.prefill_strategy == "adaptive":
                    prefill_iterator = self._adaptive_prefill(
                        suffix, cache, len(input_ids), full_prompt_ids=input_ids
                    )
                    while True:
                        try:
                            prefill_event = next(prefill_iterator)
                        except StopIteration as completed_prefill:
                            prompt_array, adaptive_prefill_seconds = completed_prefill.value
                            break
                        if prefill_event is None:
                            yield SimpleNamespace(text="", token=None, finish_reason=None,
                                                  prefill_heartbeat=True), len(input_ids), 0
                else:
                    prompt_array = self.mx.array(suffix, dtype=self.mx.uint32)
                    adaptive_prefill_seconds = 0.0
                generated_ids: list[int] = []
                response_count = 0
                loop_detected = False
                # Keep the model lock during generation so shared MLX model/cache state
                # cannot be mutated concurrently by two HTTP requests.
                generation_stream = stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt_array,
                    max_tokens=max_tokens,
                    stream=mlx_generation_stream,
                    prompt_cache=cache,
                    sampler=sampler,
                    kv_bits=self.kv_bits,
                    kv_group_size=self.kv_group_size,
                    quantized_kv_start=self.quantized_kv_start,
                    prefill_step_size=self.prefill_step_size,
                )
                for response in generation_stream:
                    if self.prefill_strategy == "adaptive" and adaptive_prefill_seconds > 0 and response_count == 0:
                        # stream_generate sees only the final uncached prompt token;
                        # present full-request statistics to API clients instead.
                        try:
                            response.prompt_tokens = len(input_ids)
                            response.prompt_tps = len(input_ids) / adaptive_prefill_seconds
                        except Exception:
                            pass
                    token = getattr(response, "token", None)
                    token_id = None
                    if token is not None:
                        try:
                            token_id = int(token)
                            generated_ids.append(token_id)
                        except (TypeError, ValueError):
                            pass
                    response_count += 1
                    guard_mode = os.environ.get("QWENFORGE_REPETITION_GUARD", "conservative").strip().lower()
                    guard_enabled = guard_mode not in {"off", "false", "0", "disabled"}
                    if token_id is not None and guard_enabled and _has_repeated_token_cycle(generated_ids):
                        loop_detected = True
                        print(
                            "Repetition guard stopped generation after a sustained exact token cycle "
                            f"(mode={guard_mode}; temperature={temperature:g}, top_p={top_p:g}, "
                            f"top_k={top_k}, min_p={min_p:g}). Structured code/markup can still cause "
                            "false positives. For diagnosis only, set QWENFORGE_REPETITION_GUARD=off to "
                            "disable automatic stopping; max_tokens will remain the upper bound.",
                            flush=True,
                        )
                        # The repeated tail token is counted, but we don't emit another
                        # copy of its text. The API adds a diagnostic stop reason.
                        yield SimpleNamespace(text="", token=token_id, finish_reason="stop",
                                              stop_reason="repetition_guard"), len(input_ids), len(generated_ids)
                        close_stream = getattr(generation_stream, "close", None)
                        if callable(close_stream):
                            close_stream()
                        break
                    yield response, len(input_ids), len(generated_ids)

                if loop_detected:
                    # Do not reuse a cache produced by a runaway/repetitive generation.
                    self._cache = None
                    self._cache_token_ids = []
                    try:
                        self.mx.clear_cache()
                    except Exception:
                        pass
                    self._last_stats = {
                        "prompt_tokens": len(input_ids),
                        "completion_tokens": len(generated_ids),
                        "peak_memory_gb": float(self.mx.get_peak_memory()) / 1e9,
                        "responses": response_count,
                        "stop_reason": "repetition_guard",
                    }
                    return

                # mlx-lm computes logits for the final yielded token but does not need
                # to feed that final token through the model until the next request.
                # Cache token IDs therefore intentionally omit the final generated token.
                if generated_ids:
                    self._cache_token_ids = input_ids + generated_ids[:-1]
                else:
                    self._cache_token_ids = input_ids
                self._cache = cache
                self._last_stats = {
                    "prompt_tokens": len(input_ids),
                    "completion_tokens": len(generated_ids),
                    "peak_memory_gb": float(self.mx.get_peak_memory()) / 1e9,
                    "responses": response_count,
                }
            except GeneratorExit:
                self._cache = None
                self._cache_token_ids = []
                try:
                    self.mx.clear_cache()
                except Exception:
                    pass
                raise
            except Exception:
                self._cache = None
                self._cache_token_ids = []
                try:
                    self.mx.clear_cache()
                except Exception:
                    pass
                raise

    def health(self) -> dict[str, Any]:
        partial_manifest: dict[str, Any] = {}
        if self._prefill_store.manifest_path is not None:
            try:
                candidate = json.loads(self._prefill_store.manifest_path.read_text(encoding="utf-8"))
                if isinstance(candidate, dict):
                    partial_manifest = candidate
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                partial_manifest = {}
        result = {
            "status": "ok",
            "engine": "mlx-lm",
            "model": self.model_id,
            "context_window": self.context_window,
            "kv_bits": self.kv_bits,
            "kv_group_size": self.kv_group_size,
            "prefill_strategy": self.prefill_strategy,
            "prefill_max_chunk_tokens": self.prefill_step_size,
            "cached_prefix_tokens": len(self._cache_token_ids),
            "ssd_snapshot_enabled": self.snapshot_dir is not None,
            "ssd_snapshot_path": str(self.snapshot_dir) if self.snapshot_dir else None,
            "ssd_snapshot_present": bool(self.snapshot_file and self.snapshot_file.is_file()),
            "ssd_snapshot_min_tokens": self.snapshot_min_tokens,
            "ssd_snapshot_mode": "completed-response-full-snapshot" if self.snapshot_dir else "disabled",
            "prefill_checkpoint_enabled": self.snapshot_dir is not None,
            "prefill_checkpoint_interval": self.prefill_checkpoint_interval,
            "prefill_checkpoint_path": str(self._prefill_store.manifest_path) if self._prefill_store.manifest_path else None,
            "prefill_checkpoint_present": bool(self._prefill_store.manifest_path and self._prefill_store.manifest_path.is_file() and partial_manifest),
            "prefill_checkpoint_tokens": len(partial_manifest.get("cache_token_ids", [])) if isinstance(partial_manifest.get("cache_token_ids", []), list) else 0,
            "prefill_checkpoint_mode": partial_manifest.get("mode", "enabled-no-checkpoint-yet") if self.snapshot_dir else "disabled",
            "prefill_checkpoint_saved_at": partial_manifest.get("saved_at"),
            "last_generation": self._last_stats,
        }
        # MLX memory APIs vary slightly by version; never fail /health over telemetry.
        for field, method_name in (("active_memory_gb", "get_active_memory"), ("peak_memory_gb", "get_peak_memory")):
            try:
                method = getattr(self.mx, method_name)
                result[field] = float(method()) / 1e9
            except Exception:
                pass
        return result



DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QwenForge</title>
<style>
:root{color-scheme:dark;--bg:#0c1018;--panel:#141b27;--line:#283448;--text:#e8edf7;--muted:#9baac1;--accent:#90e0c0;--bad:#ff9b9b}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 15% 0%,#182b32 0,transparent 42%),var(--bg);color:var(--text);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1050px;margin:0 auto;padding:34px 22px 60px}.top{display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap;margin-bottom:24px}.brand{display:flex;align-items:center;gap:14px}.mark{width:46px;height:46px;border:1px solid #3a685e;border-radius:13px;display:grid;place-items:center;color:var(--accent);font-weight:800;font-size:22px;background:#132b2b}.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}h1{font-size:27px;letter-spacing:-.04em;margin:1px 0 0}h2{font-size:17px;margin:0 0 14px}.sub{color:var(--muted);margin:3px 0 0}.status{display:flex;align-items:center;gap:9px;border:1px solid var(--line);border-radius:999px;padding:8px 13px;color:var(--muted);font-size:13px}.dot{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px #90e0c060}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:18px}.card,.panel{background:linear-gradient(145deg,#171f2d,#111722);border:1px solid var(--line);border-radius:16px}.card{padding:16px;min-width:0}.label{color:var(--muted);font-size:12px}.value{font-size:21px;font-weight:700;letter-spacing:-.03em;margin-top:6px;overflow-wrap:anywhere}.panel{padding:21px;margin-top:14px}.modelpath{color:var(--muted);font-size:13px;overflow-wrap:anywhere;margin-top:8px}.layout{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(280px,.7fr);gap:14px}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.row .spacer{flex:1}textarea,input,select{font:inherit;color:var(--text);background:#0c121c;border:1px solid #344157;border-radius:10px;padding:11px 12px;width:100%;outline:none}textarea:focus,input:focus,select:focus{border-color:var(--accent)}textarea{resize:vertical;min-height:140px}button{background:var(--accent);border:0;border-radius:10px;padding:11px 16px;color:#092019;font-weight:750;cursor:pointer}button:disabled{opacity:.5;cursor:wait}.secondary{background:#202b3b;color:var(--text);border:1px solid #344157}.field{display:grid;gap:7px;margin:12px 0}.field label{color:var(--muted);font-size:13px}.check{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px}.check input{width:auto}.thinking-panel summary{cursor:pointer;color:var(--accent);font-weight:650;margin-bottom:8px}.output{white-space:pre-wrap;overflow-wrap:anywhere;min-height:108px;max-height:480px;overflow:auto;background:#0b1119;border:1px solid var(--line);border-radius:11px;padding:14px;color:#d8e5f4}.small{font-size:12px;color:var(--muted)}.api-list{display:grid;gap:11px}.endpoint{display:flex;gap:9px;align-items:flex-start;font-size:13px}.verb{font:700 10px/1.7 ui-monospace,monospace;border-radius:5px;padding:1px 6px;background:#20382f;color:var(--accent);min-width:42px;text-align:center}.endpoint code{color:#e3eafa;overflow-wrap:anywhere}.note{margin-top:16px;color:var(--muted);font-size:12px}.error{color:var(--bad)}@media(max-width:720px){main{padding:22px 14px 40px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.layout{grid-template-columns:1fr}.value{font-size:18px}}
</style></head><body><main>
<header class="top"><div class="brand"><div class="mark">Q</div><div><div class="eyebrow">Local inference</div><h1>QwenForge</h1><p class="sub">Qwen3.8 · MLX-LM · Apple Silicon</p></div></div><div class="status"><span class="dot" id="dot"></span><span id="status">Checking server…</span></div></header>
<section class="grid"><div class="card"><div class="label">Active memory</div><div class="value" id="active">—</div></div><div class="card"><div class="label">Peak memory</div><div class="value" id="peak">—</div></div><div class="card"><div class="label">Context limit</div><div class="value" id="context">—</div></div><div class="card"><div class="label">KV cache</div><div class="value" id="kv">—</div></div></section>
<section class="panel"><h2>Loaded model</h2><div id="model">Reading model information…</div><div class="modelpath" id="path"></div><div class="modelpath" id="last"></div></section>
<div class="layout"><section class="panel"><h2>Quick chat test</h2><div class="field"><label for="prompt">Message</label><textarea id="prompt">Reply with exactly: QwenForge works</textarea></div><div class="row"><div style="width:130px"><div class="field"><label for="tokens">Max tokens</label><input id="tokens" type="number" min="1" max="262144" value="64"></div></div><label class="check"><input id="thinking" type="checkbox"> Enable thinking</label><div class="field" style="width:150px;margin:0"><label for="effort">Reasoning effort</label><select id="effort"><option value="low">Low</option><option value="medium" selected>Medium</option><option value="xhigh">Extra high</option></select></div><div class="spacer"></div><button id="stop" class="secondary" type="button" hidden>Stop generation</button><button id="send" type="button">Generate response</button></div><details class="field thinking-panel" id="thinkingPanel" open><summary>Thinking / reasoning</summary><div class="output" id="reasoningOutput">Thinking will appear here when enabled.</div></details><div class="field"><label>Final response</label><div class="output" id="output">Your model response will appear here.</div></div><div class="small" id="timing"></div></section>
<aside class="panel"><h2>API endpoints</h2><div class="api-list"><div class="endpoint"><span class="verb">GET</span><div><code>/health</code><div class="small">Engine state and memory stats</div></div></div><div class="endpoint"><span class="verb">GET</span><div><code>/v1/models</code><div class="small">OpenAI-compatible model list</div></div></div><div class="endpoint"><span class="verb">POST</span><div><code>/v1/chat/completions</code><div class="small">Generate a completion</div></div></div></div><p class="note">For an app such as OpenCode, use <code>http://127.0.0.1:8000/v1</code> as the API base URL. For this page, open <code>http://127.0.0.1:8000/</code>.</p></aside></div>
<p class="note">QwenForge is experimental. A configured 262,144-token limit does not guarantee the full context will fit in available memory.</p></main>
<script>
const $=id=>document.getElementById(id);const fmtBytes=n=>Number.isFinite(Number(n))?(Number(n).toFixed(2)+' GB'):'—';
async function refresh(){try{const r=await fetch('/health',{cache:'no-store'});if(!r.ok)throw new Error('HTTP '+r.status);const h=await r.json();$('status').textContent='Server online';$('dot').style.background='var(--accent)';$('active').textContent=fmtBytes(h.active_memory_gb);$('peak').textContent=fmtBytes(h.peak_memory_gb);$('context').textContent=Number(h.context_window||0).toLocaleString();$('kv').textContent='Q'+(h.kv_bits??'?')+' · group '+(h.kv_group_size??'?');$('path').textContent=h.model||'';const m=await fetch('/v1/models',{cache:'no-store'}).then(x=>x.json());$('model').textContent=(m.data&&m.data[0]&&m.data[0].id)||'Model loaded';const s=h.last_generation||{};$('last').textContent=Object.keys(s).length?('Last generation: '+JSON.stringify(s)):'No generation statistics recorded yet.';}catch(e){$('status').textContent='Server unavailable';$('dot').style.background='var(--bad)';$('model').textContent=String(e);}}
let activeController=null;
$('stop').addEventListener('click',()=>{if(activeController)activeController.abort();});
$('send').addEventListener('click',async()=>{
  const btn=$('send'), stopBtn=$('stop'), output=$('output'), reasoningOutput=$('reasoningOutput');
  btn.disabled=true;btn.textContent='Generating…';stopBtn.hidden=false;stopBtn.disabled=false;
  output.classList.remove('error');output.textContent='';reasoningOutput.textContent='';$('timing').textContent='Connecting…';
  const controller=new AbortController();activeController=controller;const start=performance.now();
  let usage=null,finished=false,receivedChars=0,finishReason=null,stopReason=null,thinkingIncomplete=false;
  try{
    const modelRes=await fetch('/v1/models',{cache:'no-store',signal:controller.signal});
    if(!modelRes.ok)throw new Error('Could not read model list (HTTP '+modelRes.status+')');
    const model=await modelRes.json();const modelId=model.data?.[0]?.id||'Qwen3.8-27B-4bit';
    const res=await fetch('/v1/chat/completions',{
      method:'POST',headers:{'Content-Type':'application/json'},signal:controller.signal,
      body:(()=>{const thinking=$('thinking').checked;const kwargs={enable_thinking:thinking};if(thinking)kwargs.reasoning_effort=$('effort').value;return JSON.stringify({model:modelId,messages:[{role:'user',content:$('prompt').value}],max_tokens:Math.max(1,Math.min(262144,Number($('tokens').value)||64)),temperature:thinking?1.0:0.7,top_p:thinking?0.95:0.80,top_k:20,min_p:0.0,stream:true,stream_options:{include_usage:true},chat_template_kwargs:kwargs});})()
    });
    if(!res.ok){let detail='HTTP '+res.status;try{const body=await res.json();detail=body.error?.message||detail;}catch{}throw new Error(detail);}
    if(!res.body)throw new Error('This browser does not expose a readable response stream.');
    const reader=res.body.getReader(), decoder=new TextDecoder();let buffer='';
    while(!finished){
      const part=await reader.read();
      if(part.done)break;
      buffer+=decoder.decode(part.value,{stream:true});
      let boundary;
      while((boundary=buffer.indexOf('\n\n'))>=0){
        const block=buffer.slice(0,boundary);buffer=buffer.slice(boundary+2);
        const data=block.split('\n').filter(line=>line.startsWith('data:')).map(line=>line.slice(5).trimStart()).join('\n');
        if(!data)continue;
        if(data==='[DONE]'){finished=true;break;}
        let event;try{event=JSON.parse(data);}catch{continue;}
        if(event.error)throw new Error(event.error.message||'Generation failed');
        if(event.usage)usage=event.usage;
        if(event.qwenforge?.stop_reason)stopReason=event.qwenforge.stop_reason;if(event.qwenforge?.thinking_incomplete)thinkingIncomplete=true;
        for(const choice of event.choices||[]){
          const reasoning=choice.delta?.reasoning_content??choice.delta?.reasoning;
          const text=choice.delta?.content;
          if(typeof reasoning==='string'&&reasoning.length){reasoningOutput.textContent+=reasoning;receivedChars+=reasoning.length;reasoningOutput.scrollTop=reasoningOutput.scrollHeight;}
          if(typeof text==='string'&&text.length){output.textContent+=text;receivedChars+=text.length;output.scrollTop=output.scrollHeight;}
          if(typeof reasoning==='string'||typeof text==='string'){const elapsed=(performance.now()-start)/1000;$('timing').textContent=`Streaming · ${receivedChars.toLocaleString()} characters · ${elapsed.toFixed(1)} s`;}
          if(choice.finish_reason)finishReason=choice.finish_reason;
        }
      }
    }
    if(!finished&&buffer.trim()){
      const data=buffer.split('\n').find(line=>line.startsWith('data:'))?.slice(5).trimStart();
      if(data&&data!=='[DONE]'){try{const event=JSON.parse(data);if(event.error)throw new Error(event.error.message||'Generation failed');if(event.usage)usage=event.usage;for(const choice of event.choices||[]){const reasoning=choice.delta?.reasoning_content??choice.delta?.reasoning;const text=choice.delta?.content;if(typeof reasoning==='string'&&reasoning){reasoningOutput.textContent+=reasoning;receivedChars+=reasoning.length;}if(typeof text==='string'&&text){output.textContent+=text;receivedChars+=text.length;}if(choice.finish_reason)finishReason=choice.finish_reason;}}catch(e){if(e instanceof Error&&e.message!=='Unexpected end of JSON input')throw e;}}
    }
    const ms=performance.now()-start;
    let reasonLabel=finishReason||'stream ended';
    if(controller.signal.aborted) reasonLabel='user stopped generation';
    else if(stopReason==='repetition_guard') reasonLabel='repetition guard stopped a repeated-token loop';
    else if(finishReason==='length') reasonLabel='output-token limit reached';
    else if(finishReason==='stop') {
      reasonLabel=thinkingIncomplete
        ? 'model stopped before closing its thinking block (possible early-EOS issue)'
        : 'model emitted end-of-turn token';
    }
    $('timing').textContent=`${controller.signal.aborted?'Stopped':'Completed'} in ${(ms/1000).toFixed(2)} s · ${usage?.prompt_tokens??'?'} prompt tokens · ${usage?.completion_tokens??(receivedChars?`${receivedChars} characters`:'?')} output · ${reasonLabel}`;
  }catch(e){
    if(e.name==='AbortError'){$('timing').textContent=`Generation stopped · ${( (performance.now()-start)/1000).toFixed(2)} s`}
    else{if(output.textContent)output.textContent+='\n\n[Generation error: '+e.message+']';else output.textContent='Request failed: '+e.message;output.classList.add('error');$('timing').textContent='Generation failed';}
  }finally{
    activeController=null;btn.disabled=false;btn.textContent='Generate response';stopBtn.hidden=true;stopBtn.disabled=true;await refresh();
  }
});
refresh();setInterval(refresh,5000);
</script></body></html>"""

def serve_http(
    *,
    model_id: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    context_window: int = 262_144,
    kv_bits: int = 4,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prefill_step_size: int = 1024,
    prefill_strategy: str = "adaptive",
    snapshot_dir: str | Path | None = None,
    snapshot_min_tokens: int = 2048,
    prefill_checkpoint_interval: int = 8192,
) -> int:
    engine = NativeMLXEngine(
        model_id,
        context_window=context_window,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
        prefill_step_size=prefill_step_size,
        prefill_strategy=prefill_strategy,
        snapshot_dir=snapshot_dir,
        snapshot_min_tokens=snapshot_min_tokens,
        prefill_checkpoint_interval=prefill_checkpoint_interval,
    )

    class Handler(BaseHTTPRequestHandler):
        server_version = "QwenForge/0.3.3"
        protocol_version = "HTTP/1.1"

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _html(self, code: int, page: str) -> None:
            raw = page.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in {"/", "/admin/dashboard"}:
                self._html(200, DASHBOARD_HTML)
                return
            if path == "/v1":
                self._json(200, {
                    "name": "QwenForge",
                    "version": "0.3.3",
                    "message": "This is an OpenAI-compatible API base URL, not a chat page.",
                    "web_ui": "/",
                    "endpoints": {
                        "health": "GET /health",
                        "models": "GET /v1/models",
                        "chat_completions": "POST /v1/chat/completions",
                    },
                })
                return
            if path in {"/favicon.ico", "/apple-touch-icon.png", "/apple-touch-icon-precomposed.png"}:
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if path == "/health":
                self._json(200, engine.health())
                return
            if path == "/v1/models":
                now = int(time.time())
                model_ids = [engine.model_name] + [
                    f"{engine.model_name}-{effort}" for effort in ("low", "medium", "xhigh")
                ]
                self._json(200, {
                    "object": "list",
                    "data": [{"id": model_id, "object": "model", "created": now, "owned_by": "qwenforge"}
                             for model_id in model_ids],
                })
                return
            self._json(404, {"error": {"message": f"not found: {path}. Open / for the QwenForge dashboard or /v1 for API information.", "type": "not_found"}})

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/v1/chat/completions":
                self._json(404, {"error": {"message": "not found", "type": "not_found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 128 * 1024 * 1024:
                    raise ValueError("request body must be between 1 byte and 128 MiB")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("request body must be a JSON object")
                messages = request.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a non-empty JSON array")
                for message in messages:
                    if not isinstance(message, dict):
                        raise ValueError("each message must be an object")
                    if message.get("role") not in {"system", "user", "assistant", "tool"}:
                        raise ValueError("message role must be system, user, assistant, or tool")
                    content = message.get("content", "")
                    if content is not None and not isinstance(content, (str, list)):
                        raise ValueError("message content must be text or text-only content parts")
                tools = request.get("tools") or []
                if not isinstance(tools, list) or any(not isinstance(item, dict) for item in tools):
                    raise ValueError("tools must be an array of OpenAI-compatible function tool objects")
                if request.get("tool_choice") == "none":
                    tools = []
                max_tokens = int(request.get("max_tokens", request.get("max_completion_tokens", 256)))
                if max_tokens < 1:
                    raise ValueError("max_tokens must be >= 1")
                template_kwargs = request.get("chat_template_kwargs") or {}
                if not isinstance(template_kwargs, dict):
                    raise ValueError("chat_template_kwargs must be an object")
                # Normalize Qwen/OpenCode reasoning tiers. Older OpenCode releases
                # may put selected variant parameters inside request["options"].
                template_kwargs = dict(template_kwargs)
                _apply_request_reasoning_effort(request, template_kwargs)
                # Schema-safe OpenCode fallback: a model ID ending in -low, -medium,
                # or -xhigh selects an effort when no explicit request option exists.
                # Explicit chat_template_kwargs / request effort always takes precedence.
                if "reasoning_effort" not in template_kwargs:
                    alias_effort = _reasoning_effort_from_model_alias(
                        request.get("model"), engine.model_name
                    )
                    if alias_effort:
                        template_kwargs["reasoning_effort"] = alias_effort
                # Qwen3.8's published defaults differ by mode. A hard-coded greedy
                # temperature=0 is particularly prone to repetitive thinking loops.
                thinking_enabled = bool(template_kwargs.get("enable_thinking", True))
                temperature = float(request["temperature"]) if "temperature" in request else (1.0 if thinking_enabled else 0.7)
                top_p = float(request["top_p"]) if "top_p" in request else (0.95 if thinking_enabled else 0.80)
                top_k = int(request["top_k"] or 0) if "top_k" in request else 20
                min_p = float(request.get("min_p", 0.0) or 0.0)
                print(
                    f"[QwenForge sampling] thinking={thinking_enabled} temperature={temperature:g} "
                    f"top_p={top_p:g} top_k={top_k} min_p={min_p:g}",
                    flush=True,
                )
                want_stream = bool(request.get("stream", False))
                stream_options = request.get("stream_options") or {}
                if not isinstance(stream_options, dict):
                    raise ValueError("stream_options must be an object")
                include_usage = bool(stream_options.get("include_usage", False))
                stop_value = request.get("stop")
                if isinstance(stop_value, str):
                    stop_sequences = [stop_value] if stop_value else []
                elif isinstance(stop_value, list) and all(isinstance(x, str) for x in stop_value):
                    stop_sequences = [x for x in stop_value if x]
                elif stop_value is None:
                    stop_sequences = []
                else:
                    raise ValueError("stop must be a string or an array of strings")
                requested_model = request.get("model")
                model_name = (requested_model if requested_model == engine.model_name or
                              _reasoning_effort_from_model_alias(requested_model, engine.model_name)
                              else engine.model_name)
                completion_id = f"chatcmpl-{time.time_ns()}"
                created = int(time.time())

                responses = engine.stream_responses(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    chat_template_kwargs=template_kwargs,
                    tools=tools,
                )
                parser = QwenOutputParser(thinking_enabled=thinking_enabled, tools=tools)
                if want_stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    prompt_tokens = 0
                    completion_tokens = 0
                    finish_reason = "length"
                    raw_pending = ""
                    stopped = False
                    stop_reason = None
                    tool_calls: list[dict[str, Any]] = []
                    last_tool_heartbeat = time.monotonic()

                    def send_delta(delta: dict[str, Any]) -> None:
                        self._sse({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                                   "model": model_name, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})

                    def emit_events(events: list[tuple[str, Any]]) -> None:
                        for kind, value in events:
                            if kind == "reasoning_content":
                                if value:
                                    send_delta({"reasoning_content": value})
                            elif kind == "content":
                                if value:
                                    send_delta({"content": value})
                            elif kind == "tool_call":
                                index = len(tool_calls)
                                call = value
                                tool_calls.append(call)
                                send_delta({"tool_calls": [{
                                    "index": index,
                                    "id": call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": call["function"]["name"],
                                        "arguments": json.dumps(call["function"]["arguments"], ensure_ascii=False),
                                    },
                                }]})

                    try:
                        self._sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model_name,
                                   "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                        for response, prompt_count, output_count in responses:
                            prompt_tokens = prompt_count
                            completion_tokens = max(completion_tokens, output_count)
                            if getattr(response, "prefill_heartbeat", False):
                                # Send a valid, empty SSE delta to reset client stream-idle
                                # watchdogs while MLX processes a long prompt prefill.
                                send_delta({})
                                continue
                            raw_pending += getattr(response, "text", "") or ""
                            stop_at = None
                            for marker in stop_sequences:
                                pos = raw_pending.find(marker)
                                if pos >= 0 and (stop_at is None or pos < stop_at):
                                    stop_at = pos
                            if stop_at is not None:
                                emit_events(parser.feed(raw_pending[:stop_at]))
                                finish_reason = "stop"
                                stopped = True
                                break
                            keep = max((len(marker) for marker in stop_sequences), default=1) - 1
                            if keep > 0 and len(raw_pending) > keep:
                                visible, raw_pending = raw_pending[:-keep], raw_pending[-keep:]
                            elif not stop_sequences:
                                visible, raw_pending = raw_pending, ""
                            else:
                                visible = ""
                            emit_events(parser.feed(visible))
                            # The parser intentionally waits for the closing Qwen tool tag
                            # before publishing structured tool_calls. During a large file
                            # write this may take longer than OpenCode's internal 60s idle
                            # watchdog, so provide an invisible reasoning delta every 15s.
                            last_tool_heartbeat = _maybe_send_tool_call_heartbeat(
                                parser, last_tool_heartbeat, send_delta
                            )
                            if getattr(response, "finish_reason", None):
                                finish_reason = str(response.finish_reason)
                            if getattr(response, "stop_reason", None):
                                stop_reason = str(response.stop_reason)
                        if not stopped and raw_pending:
                            emit_events(parser.feed(raw_pending))
                        emit_events(parser.finish())
                        if stopped:
                            responses.close()
                        if tool_calls:
                            finish_reason = "tool_calls"
                        final_chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model_name,
                                       "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                                       "qwenforge": {"thinking_incomplete": bool(parser.in_thinking)}}
                        if stop_reason:
                            final_chunk["qwenforge"]["stop_reason"] = stop_reason
                        self._sse(final_chunk)
                        if include_usage:
                            self._sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model_name,
                                       "choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                                                    "total_tokens": prompt_tokens + completion_tokens}})
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        # Commit the completed turn only after the protocol's terminal marker
                        # has been sent, so saving the snapshot does not delay generated text.
                        if snapshot_dir is not None:
                            engine.save_snapshot()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        try:
                            responses.close()
                        except Exception:
                            pass
                    except Exception as exc:
                        try:
                            self._sse({"error": {"message": str(exc), "type": "server_error"}})
                            self.wfile.write(b"data: [DONE]\n\n")
                        except Exception:
                            pass
                    return

                reasoning_pieces: list[str] = []
                content_pieces: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                prompt_tokens = 0
                completion_tokens = 0
                finish_reason = "length"
                stopped_by_marker = False
                stop_reason = None
                raw_pending = ""
                def collect_events(events: list[tuple[str, Any]]) -> None:
                    for kind, value in events:
                        if kind == "reasoning_content":
                            reasoning_pieces.append(value)
                        elif kind == "content":
                            content_pieces.append(value)
                        elif kind == "tool_call":
                            tool_calls.append(value)

                for response, prompt_count, output_count in responses:
                    prompt_tokens = prompt_count
                    completion_tokens = output_count
                    raw_pending += getattr(response, "text", "") or ""
                    stop_at = None
                    for marker in stop_sequences:
                        pos = raw_pending.find(marker)
                        if pos >= 0 and (stop_at is None or pos < stop_at):
                            stop_at = pos
                    if stop_at is not None:
                        collect_events(parser.feed(raw_pending[:stop_at]))
                        responses.close()
                        stopped_by_marker = True
                        finish_reason = "stop"
                        break
                    keep = max((len(marker) for marker in stop_sequences), default=1) - 1
                    if keep > 0 and len(raw_pending) > keep:
                        visible, raw_pending = raw_pending[:-keep], raw_pending[-keep:]
                    elif not stop_sequences:
                        visible, raw_pending = raw_pending, ""
                    else:
                        visible = ""
                    collect_events(parser.feed(visible))
                    if getattr(response, "finish_reason", None):
                        finish_reason = str(response.finish_reason)
                    if getattr(response, "stop_reason", None):
                        stop_reason = str(response.stop_reason)
                if raw_pending and not stopped_by_marker:
                    collect_events(parser.feed(raw_pending))
                collect_events(parser.finish())
                content = "".join(content_pieces)
                reasoning_content = "".join(reasoning_pieces)
                if tool_calls:
                    finish_reason = "tool_calls"
                message_payload: dict[str, Any] = {"role": "assistant", "content": content if content else None}
                if reasoning_content:
                    message_payload["reasoning_content"] = reasoning_content
                if tool_calls:
                    message_payload["tool_calls"] = [
                        {**call, "function": {"name": call["function"]["name"],
                                                "arguments": json.dumps(call["function"]["arguments"], ensure_ascii=False)}}
                        for call in tool_calls
                    ]
                completion_payload: dict[str, Any] = {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "message": message_payload, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
                    "qwenforge": {"thinking_incomplete": bool(parser.in_thinking)},
                }
                if stop_reason:
                    completion_payload["qwenforge"]["stop_reason"] = stop_reason
                self._json(200, completion_payload)
                if snapshot_dir is not None:
                    engine.save_snapshot()
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except Exception as exc:
                self._json(500, {"error": {"message": str(exc), "type": "server_error"}})

        def _sse(self, data: dict[str, Any]) -> None:
            encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.wfile.write(b"data: " + encoded + b"\n\n")
            self.wfile.flush()

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[QwenForge API] {self.address_string()} - {fmt % args}", flush=True)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print(f"QwenForge native MLX-LM server listening at http://{host}:{port}/v1", flush=True)
    print(f"  Browser chat: http://{host}:{port}/", flush=True)
    print(f"  Model: {model_id}", flush=True)
    print(f"  Context window: {context_window:,}; KV cache: Q{kv_bits}; group size: {kv_group_size}; quantize from token {quantized_kv_start}", flush=True)
    print(f"  Prefill step size: {prefill_step_size}", flush=True)
    if snapshot_dir:
        print(f"  Persistent SSD prefix cache: {Path(snapshot_dir).expanduser().resolve()}", flush=True)
        print(f"  Auto-checkpoint: after each completed response at >= {snapshot_min_tokens:,} cached tokens", flush=True)
        print(f"  Partial-prefill checkpoint interval: {prefill_checkpoint_interval:,} tokens", flush=True)
        print("  Partial-prefill storage: base cache + incremental KV pages + latest recurrent state (resume/checkpoint only; live attention remains in unified memory).", flush=True)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print("WARNING: server is not bound to loopback; this minimal server has no API-key authentication.", flush=True)
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nShutting down QwenForge...", flush=True)
    finally:
        # serve_forever runs on this thread; shutdown() here would deadlock.
        # server_close() is sufficient once serve_forever has returned/interrupted.
        server.server_close()
        with engine._lock:
            engine.save_snapshot()
    return 0
