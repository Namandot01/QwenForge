from pathlib import Path

import pytest

from qwenforge.ssd_pages import PageCorruptionError, SSDPageStore


def test_page_round_trip(tmp_path: Path):
    store = SSDPageStore(tmp_path, max_bytes=1024 * 1024)
    store.put("model-a:layer-3:block-1", b"KV-page-data", {"layer": 3, "block": 1})
    result = store.get("model-a:layer-3:block-1")
    assert result == (b"KV-page-data", {"layer": 3, "block": 1})


def test_missing_page_returns_none(tmp_path: Path):
    assert SSDPageStore(tmp_path).get("missing") is None


def test_corrupt_page_is_rejected(tmp_path: Path):
    store = SSDPageStore(tmp_path)
    path = store.put("test", b"original")
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(PageCorruptionError):
        store.get("test")


def test_delete_is_idempotent(tmp_path: Path):
    store = SSDPageStore(tmp_path)
    store.put("page", b"x")
    assert store.delete("page") is True
    assert store.delete("page") is False


def test_oversized_page_is_rejected_without_breaking_store(tmp_path: Path):
    store = SSDPageStore(tmp_path, max_bytes=64)
    with pytest.raises(ValueError, match="larger than store max_bytes"):
        store.put("too-large", b"x" * 100)
    assert store.stats()["pages"] == 0
