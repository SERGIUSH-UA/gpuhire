"""Статус прогону в маніфесті — той, що бачить людина в `gpurunner status`.

15.10.2026, робота 8a133851: бокс пів години читав, а список показував «queued»
(наглядач писав у маніфест лише оренду й гасіння), а після гасіння ручний
`status <id>` переписав `cancelled` на `unknown`, бо погашеного боксу «немає».
"""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpurunner import cli
from gpurunner.core import manifest
from gpurunner.core.models import JobHandle, JobStatus, StatusReport


class _GoneBackend:
    """Бекенд, для якого бокс уже знищено."""

    def status(self, handle: JobHandle) -> StatusReport:
        return StatusReport(status=JobStatus.UNKNOWN,
                            message=f"instance {handle.remote_id} no longer exists (destroyed?)")


@pytest.mark.parametrize("terminal", [JobStatus.CANCELLED, JobStatus.COMPLETED, JobStatus.FAILED])
def test_a_gone_box_does_not_rewrite_a_terminal_status(
        data_dir: Path, monkeypatch: pytest.MonkeyPatch, terminal: JobStatus) -> None:
    h = JobHandle(backend="vast", remote_id="56052819", job_name="htr_case", params={}, gpu="any")
    h.status = terminal
    h.error = "причина" if terminal == JobStatus.FAILED else None
    manifest.add(h)
    monkeypatch.setattr(cli, "get_backend", lambda name: _GoneBackend)
    res = CliRunner().invoke(cli.app, ["status", h.id[:8]])
    assert res.exit_code == 0, res.output
    got = manifest.get(h.id)
    assert got is not None and got.status == terminal
    assert got.error == h.error


def test_a_gone_box_of_a_live_job_is_still_reported_unknown(
        data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Не термінальний статус — бекенд кажe правду: доля справді невідома."""
    h = JobHandle(backend="vast", remote_id="1", job_name="htr_case", params={}, gpu="any")
    h.status = JobStatus.RUNNING
    manifest.add(h)
    monkeypatch.setattr(cli, "get_backend", lambda name: _GoneBackend)
    CliRunner().invoke(cli.app, ["status", h.id[:8]])
    got = manifest.get(h.id)
    assert got is not None and got.status == JobStatus.UNKNOWN


def test_the_supervisor_mirrors_running_into_the_manifest(
        data_dir: Path, tmp_path: Path) -> None:
    from gpurunner.supervise.htr import Supervisor
    from gpurunner.supervise.plan import CasePlan, Plan

    plan = Plan(assets_url="https://r2/a.tgz",
                cases=[CasePlan(case="spr-1", pages_url="https://r2/p.tar", n_pages=20,
                                out_dir=str(tmp_path / "spr-1"))],
                budget_usd=1.0, max_hours=1.0)
    sup = Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]
    h = JobHandle(backend="vast", remote_id="7", job_name="htr_case", params={}, gpu="any")
    manifest.add(h)                                   # оренда: queued
    sup._handle = h
    sup._absorb_queue({"phase": "setup", "case": "spr-1", "case_index": 0}, "тік")
    assert manifest.get(h.id).status == JobStatus.QUEUED, "до читання — ще не running"
    sup._absorb_queue({"phase": "running", "case": "spr-1", "case_index": 0,
                       "pages_done": 3}, "тік")
    assert manifest.get(h.id).status == JobStatus.RUNNING
