"""Manifest on SQLite: migration, the event journal, and the race it exists for."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from gpurunner.core import JobHandle, JobStatus, manifest


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path))


def _handle(**over: object) -> JobHandle:
    data: dict[str, object] = dict(
        backend="kaggle", remote_id="me/kernel-1", job_name="parseq_train",
        params={"dataset": "me/crops"}, gpu="T4",
    )
    data.update(over)
    return JobHandle(**data)  # type: ignore[arg-type]


# ---- the reason this is a database ----------------------------------------


def test_concurrent_updates_do_not_lose_each_other(tmp_path: Path) -> None:
    """The bug that forced the move off JSON.

    `load() → mutate → save()` has no locking: two writers read the same list and
    the later write drops the earlier one's change. `sweep --max-concurrent 10`
    does exactly this from ten processes, with a `watch` polling alongside.
    """
    handles = [_handle(remote_id=f"me/k{i}") for i in range(20)]
    for h in handles:
        manifest.add(h)

    errors: list[BaseException] = []

    def worker(subset: list[JobHandle]) -> None:
        try:
            for h in subset:
                h.status = JobStatus.COMPLETED
                h.output_dir = f"/out/{h.id[:8]}"
                manifest.update(h)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(handles[i::4],)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    stored = manifest.load()
    assert len(stored) == 20
    # every single one must carry its update — this is what got lost before
    assert all(h.status is JobStatus.COMPLETED for h in stored)
    assert all(h.output_dir for h in stored)


# ---- migration -------------------------------------------------------------


def test_legacy_json_is_imported_once(tmp_path: Path) -> None:
    old, kept = _handle(), _handle(remote_id="me/kernel-2")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"handles": [json.loads(h.model_dump_json()) for h in (old, kept)]}),
        encoding="utf-8",
    )

    assert {h.id for h in manifest.load()} == {old.id, kept.id}

    # a removed handle must not come back on the next open
    manifest.remove(old.id)
    assert {h.id for h in manifest.load()} == {kept.id}


def test_handle_only_in_legacy_json_is_still_reachable(tmp_path: Path) -> None:
    """A live run must not be stranded by a migration that did not cover it."""
    manifest.add(_handle())                     # DB now non-empty → no auto-import
    stray = _handle(remote_id="me/stray")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"handles": [json.loads(stray.model_dump_json())]}), encoding="utf-8"
    )

    found = manifest.get(stray.id)
    assert found is not None and found.remote_id == "me/stray"
    assert manifest.get(stray.id[:8]) is not None


# ---- journal ---------------------------------------------------------------


def test_transitions_are_journalled_but_repeats_are_not() -> None:
    h = _handle()
    manifest.add(h)
    h.status = JobStatus.RUNNING
    manifest.update(h)
    manifest.update(h)          # a 30 s watch poll seeing the same state
    manifest.update(h)
    h.status = JobStatus.FAILED
    h.error = "hit its timeout of 3600s"
    manifest.update(h)

    seen = [(e["status"], e["error"]) for e in manifest.events(h.id)]
    assert seen == [
        ("queued", None),
        ("running", None),
        ("failed", "hit its timeout of 3600s"),
    ]


def test_journal_lets_you_time_a_run() -> None:
    """created→terminal is derivable, which the old single-row manifest could not do."""
    h = _handle()
    manifest.add(h)
    h.status = JobStatus.RUNNING
    manifest.update(h)
    h.status = JobStatus.COMPLETED
    manifest.update(h)

    ev = manifest.events(h.id)
    assert ev[0]["ts"] <= ev[-1]["ts"]
    assert [e["status"] for e in ev] == ["queued", "running", "completed"]


def test_removing_a_handle_drops_its_events() -> None:
    h = _handle()
    manifest.add(h)
    h.status = JobStatus.COMPLETED
    manifest.update(h)
    assert manifest.events(h.id)

    assert manifest.remove(h.id) is True
    assert manifest.events(h.id) == []
    assert manifest.remove(h.id) is False


# ---- unchanged contract ----------------------------------------------------


def test_lookup_by_prefix_and_remote_id() -> None:
    h = _handle()
    manifest.add(h)
    assert manifest.get(h.id) is not None
    assert manifest.get(h.id[:8]) is not None
    assert manifest.get("me/kernel-1") is not None
    assert manifest.get("nope") is None


def test_update_stamps_updated_at() -> None:
    h = _handle()
    manifest.add(h)
    before = manifest.get(h.id).updated_at  # type: ignore[union-attr]
    h.status = JobStatus.RUNNING
    manifest.update(h)
    assert manifest.get(h.id).updated_at > before  # type: ignore[union-attr]


def test_params_survive_the_round_trip() -> None:
    h = _handle(params={"dataset": "me/crops", "charset_extra": "'''", "epochs": 3})
    manifest.add(h)
    got = manifest.get(h.id)
    assert got is not None
    assert got.params == {"dataset": "me/crops", "charset_extra": "'''", "epochs": 3}


# ---- спільний реєстр і чужі прогони ----------------------------------------


def test_ambiguous_prefix_raises_instead_of_guessing(data_dir) -> None:
    """🔴 `cancel <короткий префікс>` мовчки брав НАЙСТАРІШИЙ збіг.

    На спільній базі (дві сесії на одній машині) це прямий шлях знищити
    чужий живий бокс замість свого. Тепер неоднозначність — помилка.
    """
    from gpurunner.core.manifest import AmbiguousHandle

    for suffix in ("aaa", "bbb"):
        h = JobHandle(backend="vast", remote_id=f"r-{suffix}", job_name="htr_case", gpu="any")
        h.id = "deadbeef" + suffix + "0" * (32 - 8 - len(suffix))
        manifest.add(h)
    assert manifest.get("deadbeefaaa") is not None
    with pytest.raises(AmbiguousHandle, match="збігається з 2"):
        manifest.get("deadbeef")


def test_owner_is_recorded_and_defaults_to_env(data_dir, monkeypatch) -> None:
    monkeypatch.setenv("GPURUNNER_OWNER", "session-A")
    h = JobHandle(backend="vast", remote_id="42", job_name="htr_case", gpu="any")
    manifest.add(h)
    assert manifest.get(h.id).owner == "session-A"


def test_old_rows_without_owner_are_readable(data_dir) -> None:
    """Міграція не має ламати те, що вже лежить у базі."""
    h = JobHandle(backend="vast", remote_id="43", job_name="htr_case", gpu="any")
    manifest.add(h)
    conn = sqlite3.connect(str(manifest.db_path()))
    try:
        conn.execute("UPDATE runs SET owner = NULL WHERE id = ?", (h.id,))
        conn.commit()
    finally:
        conn.close()
    assert manifest.get(h.id).owner == ""
