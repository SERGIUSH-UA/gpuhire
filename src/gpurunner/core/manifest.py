"""Local DB of submitted job handles, plus the history of what happened to them.

Was a single JSON document rewritten in full on every change. Two things forced
the move to SQLite:

**Lost updates.** ``load() → mutate → save()`` has no locking, so two writers
racing (``sweep --max-concurrent 10`` submits from ten processes; a ``watch`` in
another shell polls every 30 s) both read the same list and the second write
silently drops the first one's change. ``core/budget.py`` already made this
argument for money; handles have the same shape of bug and no ceiling to catch
it. Each write here is now one transaction, and the file is opened WAL so a
reader never blocks a writer.

**No history.** A handle carried only its *latest* status, so every transition
overwrote the previous one: the manifest could say a run FAILED but never when
it started running, or how long it took. Reconstructing "what went wrong in the
last 50 runs" meant grepping ``.log`` files scattered across output dirs. Every
status change now appends to ``run_events``, which makes that a query.

The JSON file is still read once, to import what it holds, and is then left
alone as a backup — ``load()``/``get()`` fall back to it for anything the DB has
never heard of, so a half-finished migration cannot strand a live run.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gpurunner.config import data_dir, manifest_path
from gpurunner.core.models import JobHandle, _utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    backend     TEXT NOT NULL,
    remote_id   TEXT,
    job_name    TEXT NOT NULL,
    params      TEXT,
    gpu         TEXT,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    output_dir  TEXT,
    volume_name TEXT,
    error       TEXT,
    owner       TEXT
);
CREATE INDEX IF NOT EXISTS runs_created ON runs (created_at);
CREATE INDEX IF NOT EXISTS runs_remote  ON runs (remote_id);

-- One row per observed transition. Nothing is ever updated here: this is the
-- only place that can answer "when did it start running / how long did it take".
CREATE TABLE IF NOT EXISTS run_events (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  TEXT NOT NULL,
    ts      TEXT NOT NULL,
    status  TEXT,
    error   TEXT,
    note    TEXT
);
CREATE INDEX IF NOT EXISTS run_events_run ON run_events (run_id, seq);
"""

_COLUMNS = (
    "id", "backend", "remote_id", "job_name", "params", "gpu", "status",
    "created_at", "updated_at", "output_dir", "volume_name", "error", "owner",
)


class AmbiguousHandle(LookupError):
    """Префікс хендла збігся з кількома прогонами."""


def current_owner() -> str:
    """Хто «я» для реєстру прогонів.

    🔴 База `runs.sqlite3` спільна на всю машину, тож дві паралельні сесії
    бачать хендли одна одної як свої. Мітка власника — єдине, що дозволяє
    `cancel --all-running` не вбити чужий живий бокс.

    Порожньо — «невідомо»; такі прогони масові операції НЕ чіпають.
    """
    return str(os.environ.get("GPURUNNER_OWNER") or "").strip()


def db_path() -> Path:
    return data_dir() / "runs.sqlite3"


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None → transactions are driven explicitly below.
    # busy_timeout: a concurrent writer should wait for the lock, not fail a submit.
    conn = sqlite3.connect(str(path), timeout=15, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.executescript(_SCHEMA)
        _migrate(conn)
        _import_legacy_once(conn)
        yield conn
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Долити колонки, яких немає у старій базі.

    `CREATE TABLE IF NOT EXISTS` мовчки лишає стару схему, тож нове поле
    треба додавати явно — інакше запис падає на «no column named owner»
    рівно там, де реєструється щойно орендований (і вже оплачуваний) бокс.
    """
    have = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    for column, ddl in (("owner", "ALTER TABLE runs ADD COLUMN owner TEXT"),):
        if column not in have:
            conn.execute(ddl)


# ---- legacy JSON import ----------------------------------------------------


def _import_legacy_once(conn: sqlite3.Connection) -> None:
    """Copy manifest.json into the DB the first time, then never again.

    Guarded by emptiness rather than a flag file: if the table has rows, the
    import already happened (or the user started fresh), and re-importing would
    resurrect handles that were deliberately removed.
    """
    if conn.execute("SELECT 1 FROM runs LIMIT 1").fetchone():
        return
    legacy = manifest_path()
    if not legacy.exists():
        return
    try:
        payload = json.loads(legacy.read_text(encoding="utf-8"))
        handles = [JobHandle.model_validate(h) for h in payload.get("handles", [])]
    except Exception:
        return  # a corrupt legacy file must not make the DB unusable
    conn.execute("BEGIN IMMEDIATE")
    try:
        for handle in handles:
            _upsert(conn, handle)
            conn.execute(
                "INSERT INTO run_events (run_id, ts, status, error, note)"
                " VALUES (?, ?, ?, ?, 'imported from manifest.json')",
                (handle.id, handle.updated_at.isoformat(),
                 _status_str(handle.status), handle.error),
            )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _legacy_handles() -> list[JobHandle]:
    """Whatever the old JSON still holds — used only as a read-through fallback."""
    legacy = manifest_path()
    if not legacy.exists():
        return []
    try:
        payload = json.loads(legacy.read_text(encoding="utf-8"))
        return [JobHandle.model_validate(h) for h in payload.get("handles", [])]
    except Exception:
        return []


# ---- row <-> model ---------------------------------------------------------


def _status_str(status: Any) -> str:
    return str(getattr(status, "value", status))


def _to_row(handle: JobHandle) -> tuple:
    return (
        handle.id,
        handle.backend,
        handle.remote_id,
        handle.job_name,
        json.dumps(handle.params, ensure_ascii=False, default=str),
        handle.gpu,
        _status_str(handle.status),
        handle.created_at.isoformat(),
        handle.updated_at.isoformat(),
        handle.output_dir,
        handle.volume_name,
        handle.error,
        handle.owner or None,
    )


def _from_row(row: tuple) -> JobHandle:
    data = dict(zip(_COLUMNS, row, strict=True))
    # Старі рядки не мають власника — це «невідомо», а не помилка.
    data["owner"] = data.get("owner") or ""
    try:
        data["params"] = json.loads(data["params"]) if data["params"] else {}
    except json.JSONDecodeError:
        data["params"] = {}
    return JobHandle.model_validate(data)


def _upsert(conn: sqlite3.Connection, handle: JobHandle) -> None:
    placeholders = ", ".join("?" * len(_COLUMNS))
    conn.execute(
        f"INSERT OR REPLACE INTO runs ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
        _to_row(handle),
    )


# ---- public API (unchanged for callers) ------------------------------------


def load() -> list[JobHandle]:
    """All known handles, oldest first — same order the JSON list had."""
    with _db() as conn:
        rows = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM runs ORDER BY created_at"
        ).fetchall()
    return [_from_row(r) for r in rows]


def save(handles: list[JobHandle]) -> None:
    """Replace the whole set. Kept for API compatibility; prefer add/update."""
    with _db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM runs")
            for handle in handles:
                _upsert(conn, handle)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


def add(handle: JobHandle) -> None:
    if not handle.owner:
        handle.owner = current_owner()
    with _db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _upsert(conn, handle)
            conn.execute(
                "INSERT INTO run_events (run_id, ts, status, error, note)"
                " VALUES (?, ?, ?, NULL, 'submitted')",
                (handle.id, handle.created_at.isoformat(), _status_str(handle.status)),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


def update(handle: JobHandle) -> None:
    """Persist a handle, stamping ``updated_at`` and journalling any transition.

    ``updated_at`` is set here rather than at each call site because every writer
    (status --ping, watch, fetch, sweep) forgot to: across 287 handles the field
    still equalled ``created_at``, so run durations were not derivable at all.
    """
    handle.updated_at = _utcnow()
    # 🔴 Доштампувати власника, якщо його немає. Раніше це робив лише `add()`,
    # тож хендли, створені до появи поля (або чужим шляхом), лишались «нічиїми»
    # НАЗАВЖДИ — і назавжди невидимими для масових операцій із owner-фільтром.
    if not handle.owner:
        handle.owner = current_owner()
    with _db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            prev = conn.execute(
                "SELECT status, error FROM runs WHERE id = ?", (handle.id,)
            ).fetchone()
            _upsert(conn, handle)
            now_status, now_error = _status_str(handle.status), handle.error
            # Journal only real transitions — a 30-second `watch` poll that sees
            # the same RUNNING must not write 1400 identical rows per 12-h train.
            if prev is None or prev[0] != now_status or (prev[1] or None) != (now_error or None):
                conn.execute(
                    "INSERT INTO run_events (run_id, ts, status, error, note)"
                    " VALUES (?, ?, ?, ?, NULL)",
                    (handle.id, handle.updated_at.isoformat(), now_status, now_error),
                )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


def get(handle_id: str) -> JobHandle | None:
    """Full id, 8-char (or any) prefix, or remote_id — same contract as before."""
    with _db() as conn:
        row = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM runs WHERE id = ? OR remote_id = ?",
            (handle_id, handle_id),
        ).fetchone()
        if row is None:
            # 🔴 Неоднозначний префікс — це ПОМИЛКА, а не «візьму перший».
            # Раніше тут стояло `fetchone()` з сортуванням за датою, тобто
            # `cancel 47` мовчки обирав найстаріший збіг. На спільній базі
            # (дві сесії на одній машині) це шлях знищити чужий живий бокс.
            rows = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM runs WHERE id LIKE ? ORDER BY created_at",
                (handle_id.replace("%", "") + "%",),
            ).fetchall()
            if len(rows) > 1:
                ids = ", ".join(r[0][:8] for r in rows[:6])
                raise AmbiguousHandle(
                    f"префікс {handle_id!r} збігається з {len(rows)} прогонами "
                    f"({ids}{'…' if len(rows) > 6 else ''}) — назви довший id"
                )
            row = rows[0] if rows else None
    if row is not None:
        return _from_row(row)
    # Read-through fallback: anything the DB has never seen but the old file
    # still holds stays reachable, so a partial migration cannot strand a run.
    for handle in _legacy_handles():
        if handle.id == handle_id or handle.id.startswith(handle_id) or handle.remote_id == handle_id:
            return handle
    return None


def remove(handle_id: str) -> bool:
    with _db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute("DELETE FROM runs WHERE id = ?", (handle_id,))
            conn.execute("DELETE FROM run_events WHERE run_id = ?", (handle_id,))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return cur.rowcount > 0


# ---- history ---------------------------------------------------------------


def events(handle_id: str) -> list[dict[str, Any]]:
    """Recorded transitions for one run, oldest first."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT ts, status, error, note FROM run_events WHERE run_id = ? ORDER BY seq",
            (handle_id,),
        ).fetchall()
    return [{"ts": ts, "status": st, "error": err, "note": note} for ts, st, err, note in rows]


def events_bulk(handle_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Transitions for many runs at once, keyed by run id.

    ``events()`` per handle means one connection *per run*, and each open also
    runs the schema script and the legacy-import probe. Anything that walks the
    whole manifest (the dashboard's usage figures do) would pay that ~300 times
    for a single page load. Ids are chunked because SQLite caps a statement at
    999 bound parameters by default.
    """
    if not handle_ids:
        return {}
    out: dict[str, list[dict[str, Any]]] = {hid: [] for hid in handle_ids}
    with _db() as conn:
        for i in range(0, len(handle_ids), 500):
            chunk = handle_ids[i : i + 500]
            placeholders = ", ".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT run_id, ts, status, error, note FROM run_events"
                f" WHERE run_id IN ({placeholders}) ORDER BY seq",
                chunk,
            ).fetchall()
            for run_id, ts, st, err, note in rows:
                out[run_id].append({"ts": ts, "status": st, "error": err, "note": note})
    return out
