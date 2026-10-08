"""Unit tests for core abstractions and manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpurunner.core import JobHandle, JobStatus, manifest


@pytest.fixture(autouse=True)
def _isolated_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the manifest at a per-test tmp file."""
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path))


def _make_handle(**overrides: object) -> JobHandle:
    defaults: dict[str, object] = dict(
        backend="kaggle",
        remote_id="alter/test-kernel-1234",
        job_name="paddleocr",
        params={"lang": "ru"},
        gpu="T4",
    )
    defaults.update(overrides)
    return JobHandle(**defaults)  # type: ignore[arg-type]


def test_manifest_empty() -> None:
    assert manifest.load() == []


def test_manifest_round_trip() -> None:
    h = _make_handle()
    manifest.add(h)
    loaded = manifest.load()
    assert len(loaded) == 1
    assert loaded[0].id == h.id
    assert loaded[0].backend == "kaggle"
    assert loaded[0].status == JobStatus.QUEUED


def test_manifest_update_status() -> None:
    h = _make_handle()
    manifest.add(h)
    h.status = JobStatus.RUNNING
    manifest.update(h)
    loaded = manifest.load()
    assert len(loaded) == 1
    assert loaded[0].status == JobStatus.RUNNING


def test_manifest_update_stamps_updated_at() -> None:
    """updated_at мусить рухатись сам, без участі того, хто кличе update().

    Кожен писар (status --ping, watch, fetch, sweep) свого часу забув його
    проставити, тож на 284 хендлах поспіль updated_at дорівнював created_at і
    тривалість прогону з маніфесту не виводилась узагалі.
    """
    h = _make_handle()
    manifest.add(h)
    before = manifest.load()[0].updated_at
    h.status = JobStatus.RUNNING
    manifest.update(h)
    assert manifest.load()[0].updated_at > before


def test_manifest_keeps_error() -> None:
    h = _make_handle()
    manifest.add(h)
    h.status = JobStatus.FAILED
    h.error = "hit its timeout of 3600s"
    manifest.update(h)
    loaded = manifest.load()[0]
    assert loaded.status == JobStatus.FAILED
    assert loaded.error == "hit its timeout of 3600s"


def test_manifest_get_by_prefix() -> None:
    h = _make_handle()
    manifest.add(h)
    got = manifest.get(h.id[:8])
    assert got is not None
    assert got.id == h.id


def test_manifest_get_by_remote_id() -> None:
    h = _make_handle()
    manifest.add(h)
    got = manifest.get("alter/test-kernel-1234")
    assert got is not None
    assert got.id == h.id


def test_manifest_remove() -> None:
    h = _make_handle()
    manifest.add(h)
    assert manifest.remove(h.id) is True
    assert manifest.load() == []
    assert manifest.remove(h.id) is False
