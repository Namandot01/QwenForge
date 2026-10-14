"""Small crash-safe SSD page store prototype for future live-KV integration.

This module is intentionally runtime-agnostic. The current engine does NOT
plug these pages into MLX attention; oMLX itself handles persisted prefix
snapshots. This implementation is a tested persistence primitive for future
work on block-addressable KV/recurrent-state snapshots.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import zlib
from pathlib import Path
from typing import Any

MAGIC = b"QWENFORGE_PAGE_V1\n"


class PageCorruptionError(RuntimeError):
    pass


class SSDPageStore:
    def __init__(self, root: str | Path, max_bytes: int = 50 * 1024**3):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._lock = threading.RLock()

    @staticmethod
    def _safe_name(key: str) -> str:
        if not key:
            raise ValueError("key cannot be empty")
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _page_path(self, key: str) -> Path:
        return self.root / (self._safe_name(key) + ".qfp")

    def put(self, key: str, payload: bytes, metadata: dict[str, Any] | None = None) -> Path:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        record = {
            "key": key,
            "size": len(payload),
            "crc32": f"{zlib.crc32(payload) & 0xffffffff:08x}",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "metadata": metadata or {},
            "written_at": time.time(),
        }
        header = json.dumps(record, separators=(",", ":")).encode("utf-8")
        body = MAGIC + str(len(header)).encode("ascii") + b"\n" + header + payload
        if len(body) > self.max_bytes:
            raise ValueError(f"Serialized page is {len(body)} bytes, larger than store max_bytes={self.max_bytes}")
        path = self._page_path(key)
        with self._lock:
            temp = path.with_suffix(".qfp.tmp")
            with temp.open("wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
            self._evict_unlocked(exclude=path)
        return path

    def get(self, key: str) -> tuple[bytes, dict[str, Any]] | None:
        path = self._page_path(key)
        with self._lock:
            if not path.is_file():
                return None
            try:
                raw = path.read_bytes()
                if not raw.startswith(MAGIC):
                    raise PageCorruptionError(f"Invalid page magic: {path.name}")
                offset = len(MAGIC)
                end_len = raw.index(b"\n", offset)
                header_len = int(raw[offset:end_len].decode("ascii"))
                h0, h1 = end_len + 1, end_len + 1 + header_len
                record = json.loads(raw[h0:h1].decode("utf-8"))
                payload = raw[h1:]
                if record.get("key") != key:
                    raise PageCorruptionError("Page key mismatch")
                if len(payload) != record.get("size"):
                    raise PageCorruptionError("Page size mismatch")
                if f"{zlib.crc32(payload) & 0xffffffff:08x}" != record.get("crc32"):
                    raise PageCorruptionError("Page CRC mismatch")
                if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
                    raise PageCorruptionError("Page SHA-256 mismatch")
                os.utime(path, None)
                return payload, record.get("metadata", {})
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
                if isinstance(exc, PageCorruptionError):
                    raise
                raise PageCorruptionError(f"Unable to read page {path.name}: {exc}") from exc

    def delete(self, key: str) -> bool:
        path = self._page_path(key)
        with self._lock:
            try:
                path.unlink()
                return True
            except FileNotFoundError:
                return False

    def stats(self) -> dict[str, int]:
        with self._lock:
            files = list(self.root.glob("*.qfp"))
            return {"pages": len(files), "bytes": sum(p.stat().st_size for p in files), "max_bytes": self.max_bytes}

    def _evict_unlocked(self, exclude: Path | None = None) -> None:
        pages = [p for p in self.root.glob("*.qfp") if p != exclude]
        current = sum(p.stat().st_size for p in pages)
        if exclude is not None and exclude.exists():
            current += exclude.stat().st_size
        if current <= self.max_bytes:
            return
        pages.sort(key=lambda p: p.stat().st_atime)
        for page in pages:
            try:
                size = page.stat().st_size
                page.unlink()
                current -= size
            except FileNotFoundError:
                continue
            if current <= self.max_bytes:
                break
