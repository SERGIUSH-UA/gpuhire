"""Local spend ledger — what stands between a typo and a real charge.

Providers that bill money differ in what they will tell you. Modal exposes usage,
Beam exposes **nothing**: no credit-balance endpoint, no usage endpoint. Once a card
is attached, the only thing that can stop a mistyped ``--gpu H100 -p max_hours=12``
from becoming a $43 charge is this file.

So the ledger is deliberately pessimistic:

- a run is booked at its **worst case** (``timeout × rate``), not at its estimate — a
  job that hangs until the timeout costs exactly that;
- the booking happens **before** the task can start doing anything, and is rolled
  back if the submit then fails;
- when the run ends it is *settled* with the wall time the runner reported, so the
  month's budget frees up again;
- anything that cannot be priced is refused rather than assumed cheap.

**Why SQLite and not JSON** (the rest of gpurunner's state is a JSON file): the
check-then-write here must be atomic. With a JSON document, two shells submitting at
once both read "committed = $28", both decide there is room, and both write — one
reservation is lost and the cap is silently breached. ``reserve_within_cap()`` does
the sum and the insert inside a single ``BEGIN IMMEDIATE``, so the second caller
either sees the first one's booking or waits for it.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpurunner.config import data_dir


class BudgetExceeded(RuntimeError):
    """A reservation would breach the monthly cap. Nothing was written."""

    def __init__(self, committed: float, requested: float, cap: float) -> None:
        self.committed = committed
        self.requested = requested
        self.cap = cap
        super().__init__(
            f"monthly cap ${cap:.2f} would be breached: ${committed:.2f} committed "
            f"+ ${requested:.2f} requested"
        )


def ledger_path() -> Path:
    return data_dir() / "spend.sqlite3"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    handle    TEXT PRIMARY KEY,
    backend   TEXT NOT NULL,
    month     TEXT NOT NULL,
    ts        TEXT NOT NULL,
    reserved  REAL NOT NULL,
    actual    REAL,
    meta      TEXT
);
CREATE INDEX IF NOT EXISTS spend_backend_month ON spend (backend, month);
"""


@contextmanager
def _db():
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None → we drive transactions by hand (BEGIN IMMEDIATE below).
    # timeout: a concurrent submit should wait for the lock, not fail the run.
    conn = sqlite3.connect(str(path), timeout=15, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.executescript(_SCHEMA)
        yield conn
    finally:
        conn.close()


def _month_key(when: datetime | None = None) -> str:
    return (when or datetime.now(tz=UTC)).strftime("%Y-%m")


def reserve_within_cap(
    backend: str,
    handle_id: str,
    amount: float,
    *,
    cap: float,
    **meta: Any,
) -> float:
    """Book ``amount`` for this month unless it would breach ``cap``.

    The sum and the insert share one ``BEGIN IMMEDIATE`` transaction — that is the
    whole point of this module being a database. Returns the month's committed total
    *including* the new booking; raises ``BudgetExceeded`` (having written nothing)
    otherwise.
    """
    month = _month_key()
    with _db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            committed = _committed(conn, backend, month)
            if committed + amount > cap:
                conn.execute("ROLLBACK")
                raise BudgetExceeded(committed, amount, cap)
            conn.execute(
                "INSERT OR REPLACE INTO spend (handle, backend, month, ts, reserved, actual, meta)"
                " VALUES (?, ?, ?, ?, ?, NULL, ?)",
                (
                    handle_id,
                    backend,
                    month,
                    datetime.now(tz=UTC).isoformat(),
                    round(float(amount), 4),
                    json.dumps(meta, ensure_ascii=False, default=str),
                ),
            )
            conn.execute("COMMIT")
        except BaseException:
            with_suppressed_rollback(conn)
            raise
        return committed + amount


def with_suppressed_rollback(conn: sqlite3.Connection) -> None:
    """Roll back if a transaction is open; a closed one is not an error here."""
    with suppress(sqlite3.Error):
        conn.execute("ROLLBACK")


def release(backend: str, handle_id: str) -> None:
    """Drop a booking — the submit it was made for never started."""
    with _db() as conn:
        conn.execute(
            "DELETE FROM spend WHERE handle = ? AND backend = ?", (handle_id, backend)
        )


def settle(backend: str, handle_id: str, actual: float) -> None:
    """Replace a booking's worst case with what the run really cost. Idempotent."""
    with _db() as conn:
        conn.execute(
            "UPDATE spend SET actual = ? WHERE handle = ? AND backend = ?",
            (round(float(actual), 4), handle_id, backend),
        )


def is_settled(backend: str, handle_id: str) -> bool:
    with _db() as conn:
        row = conn.execute(
            "SELECT actual FROM spend WHERE handle = ? AND backend = ?", (handle_id, backend)
        ).fetchone()
    return bool(row) and row[0] is not None


def _committed(conn: sqlite3.Connection, backend: str, month: str) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(COALESCE(actual, reserved)), 0) FROM spend"
        " WHERE backend = ? AND month = ?",
        (backend, month),
    ).fetchone()
    return float(row[0] or 0.0)


def month_committed(backend: str, when: datetime | None = None) -> float:
    """$ committed this month: settled runs at their real cost, live ones at worst case."""
    with _db() as conn:
        return _committed(conn, backend, _month_key(when))


def month_entries(backend: str, when: datetime | None = None) -> list[dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT handle, ts, reserved, actual, meta FROM spend"
            " WHERE backend = ? AND month = ? ORDER BY ts",
            (backend, _month_key(when)),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for handle, ts, reserved, actual, meta in rows:
        try:
            parsed = json.loads(meta) if meta else {}
        except json.JSONDecodeError:
            parsed = {}
        out.append(
            {"handle": handle, "ts": ts, "reserved": reserved, "actual": actual, **parsed}
        )
    return out
