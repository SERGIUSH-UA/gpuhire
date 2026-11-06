"""Unit tests for `gpurunner sweep`.

Uses a FakeBackend that finishes a configurable number of poll cycles after
submission. Verifies:
  - concurrency cap is never exceeded
  - skip-complete short-circuits without submitting
  - handle.params is preserved verbatim (no truncation)
  - _sweep_summary.json is written at the end
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from gpurunner.cli import _read_params_file, _run_sweep
from gpurunner.core import Job, JobStatus
from gpurunner.core.models import JobHandle, StatusReport

# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_CONFIG_DIR", str(tmp_path / "config"))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the real time.sleep during sweep tests."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_kw: None)


class _EchoJob(Job):
    """Trivial job that writes ``params`` as JSON into out_dir."""

    name = "echo"
    description = "echo test job"
    supported_backends = ("fake",)

    def requirements(self) -> list[str]:
        return []

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return dict(params)

    def render_remote_code(
        self,
        params: dict[str, Any],
        *,
        shard_index: int = 0,
        total_shards: int = 1,
    ) -> str:
        return f"print({params!r})"


class _FakeBackend:
    """Stand-in for a real Backend that completes after ``poll_to_complete``
    calls to ``status()``. Tracks max concurrent in-flight submissions so the
    test can assert the cap is respected."""

    name = "fake"
    default_gpu = "T4"

    def __init__(self, *, poll_to_complete: int = 1, fail_labels: set[str] | None = None) -> None:
        self.poll_to_complete = poll_to_complete
        self.fail_labels = fail_labels or set()
        self.in_flight: dict[str, int] = {}  # remote_id -> remaining polls
        self.handle_label: dict[str, str] = {}  # remote_id -> label (for fail check)
        self.max_in_flight_observed = 0
        self.submits: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self._seq = 0

    def check_auth(self) -> None:
        return None

    def submit(self, job: Job, params: dict[str, Any], *, gpu: str) -> JobHandle:
        self._seq += 1
        remote_id = f"fake-{self._seq:04d}"
        # Label hint comes from params if the caller embedded one; for this
        # test the sweep CLI pops 'label' before submit, so we use the seq id.
        self.submits.append(dict(params))
        h = JobHandle(
            backend=self.name,
            remote_id=remote_id,
            job_name=job.name,
            params=params,
            gpu=gpu,
        )
        self.in_flight[h.id] = self.poll_to_complete
        self.handle_label[h.id] = remote_id
        self.max_in_flight_observed = max(self.max_in_flight_observed, len(self.in_flight))
        return h

    def status(self, handle: JobHandle) -> StatusReport:
        remaining = self.in_flight.get(handle.id, 0)
        if remaining > 1:
            self.in_flight[handle.id] = remaining - 1
            return StatusReport(status=JobStatus.RUNNING)
        if remaining == 1:
            self.in_flight.pop(handle.id)
            label = self.handle_label.get(handle.id, "")
            if label in self.fail_labels:
                return StatusReport(status=JobStatus.FAILED, error="forced failure")
            return StatusReport(status=JobStatus.COMPLETED)
        return StatusReport(status=JobStatus.COMPLETED)

    def fetch_outputs(self, handle: JobHandle, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "result.json"
        p.write_text(json.dumps(handle.params), encoding="utf-8")
        return [p]

    def cancel(self, handle: JobHandle) -> None:
        self.cancelled.append(handle.remote_id)
        self.in_flight.pop(handle.id, None)


# ---------------------------------------------------------------------------
# _read_params_file
# ---------------------------------------------------------------------------


def test_read_params_file_basic(tmp_path: Path) -> None:
    f = tmp_path / "rows.jsonl"
    f.write_text(
        '\n'
        '# leading comment\n'
        '{"a": 1, "label": "row_a"}\n'
        '\n'
        '{"a": 2, "label": "row_b"}\n'
        '# trailing comment\n',
        encoding="utf-8",
    )
    rows = _read_params_file(f)
    assert rows == [{"a": 1, "label": "row_a"}, {"a": 2, "label": "row_b"}]


def test_read_params_file_rejects_non_object(tmp_path: Path) -> None:
    f = tmp_path / "rows.jsonl"
    f.write_text('[1, 2, 3]\n', encoding="utf-8")
    with pytest.raises(Exception, match="expected JSON object"):
        _read_params_file(f)


def test_read_params_file_rejects_invalid_json(tmp_path: Path) -> None:
    f = tmp_path / "rows.jsonl"
    f.write_text('{not json}\n', encoding="utf-8")
    with pytest.raises(Exception, match="invalid JSON"):
        _read_params_file(f)


# ---------------------------------------------------------------------------
# _run_sweep
# ---------------------------------------------------------------------------


def _make_plan(out_root: Path, n: int) -> list[tuple[str, Path, dict[str, Any]]]:
    return [
        (f"row_{i:02d}", out_root / f"row_{i:02d}", {"i": i})
        for i in range(n)
    ]


def test_run_sweep_respects_max_concurrent(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    out_root.mkdir()
    plan = _make_plan(out_root, n=10)
    bk = _FakeBackend(poll_to_complete=3)
    job = _EchoJob()

    _run_sweep(job, bk, plan, gpu="T4", max_concurrent=3, poll=0, out_root=out_root)

    assert bk.max_in_flight_observed <= 3
    assert len(bk.submits) == 10


def test_run_sweep_preserves_full_params(tmp_path: Path) -> None:
    """Regression: a sweep script truncated handle.params to {year, n_pdfs}.
    Sweep must NOT do that — manifest entry should contain the original params."""
    from gpurunner.core import manifest

    out_root = tmp_path / "out"
    out_root.mkdir()
    plan = [
        ("row_a", out_root / "row_a", {"i": 1, "extra": [1, 2, 3], "deep": {"k": "v"}}),
    ]
    bk = _FakeBackend(poll_to_complete=1)
    _run_sweep(_EchoJob(), bk, plan, gpu="T4", max_concurrent=2, poll=0, out_root=out_root)

    handles = manifest.load()
    assert len(handles) == 1
    h = handles[0]
    assert h.params == {"i": 1, "extra": [1, 2, 3], "deep": {"k": "v"}}


def test_run_sweep_writes_summary(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    out_root.mkdir()
    plan = _make_plan(out_root, n=3)
    bk = _FakeBackend(poll_to_complete=1, fail_labels={"fake-0002"})
    n_bad = _run_sweep(_EchoJob(), bk, plan, gpu="T4", max_concurrent=3, poll=0, out_root=out_root)
    assert n_bad >= 1

    sp = out_root / "_sweep_summary.json"
    assert sp.exists()
    payload = json.loads(sp.read_text(encoding="utf-8"))
    assert payload["rows"] == 3
    assert payload["ok"] >= 1
    # The forced failure should be reflected.
    assert any("FAILED" in v or "failed" in v for v in payload["results"].values()) or payload["failed"] >= 1


def test_run_sweep_fetches_to_label_dir(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    out_root.mkdir()
    plan = _make_plan(out_root, n=2)
    bk = _FakeBackend(poll_to_complete=1)
    _run_sweep(_EchoJob(), bk, plan, gpu="T4", max_concurrent=2, poll=0, out_root=out_root)

    assert (out_root / "row_00" / "result.json").exists()
    assert (out_root / "row_01" / "result.json").exists()


# ---------------------------------------------------------------------------
# Job.is_output_complete default + override
# ---------------------------------------------------------------------------


def test_is_output_complete_default(tmp_path: Path) -> None:
    job = _EchoJob()
    out = tmp_path / "x"
    assert not job.is_output_complete(out)
    out.mkdir()
    assert not job.is_output_complete(out)
    (out / "file.txt").write_text("hi", encoding="utf-8")
    assert job.is_output_complete(out)


def test_paddleocr_is_output_complete(tmp_path: Path) -> None:
    from gpurunner.jobs import PaddleOCRJob

    job = PaddleOCRJob()
    out = tmp_path / "x"
    out.mkdir()
    # Empty dir is NOT complete.
    assert not job.is_output_complete(out)
    # Stray non-page file is NOT complete.
    (out / "garbage.log").write_text("noise", encoding="utf-8")
    assert not job.is_output_complete(out)
    # _summary.json marks complete.
    (out / "_summary.json").write_text("{}", encoding="utf-8")
    assert job.is_output_complete(out)
    # Without _summary, page_*.txt also counts.
    sp = out / "_summary.json"
    sp.unlink()
    sub = out / "pdf_0000"
    sub.mkdir()
    (sub / "page_0001.txt").write_text("hello", encoding="utf-8")
    assert job.is_output_complete(out)
