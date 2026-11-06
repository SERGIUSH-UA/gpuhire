"""`wrap-up` наглядача: теплий бокс без другого забору, заливка до зупинки, SFTP з короткою стелею.

Інцидент 28.10.2026: чергу забрано штатно через R2, бокс у теплому очікуванні;
`wrap-up` вбив раннер разом із `curl -X PUT` (заливка чекпоінта потім
тайм-аутилась), пішов у другий забір і висів на SFTP 30 хв. Машину гасили 36 хв
замість хвилини. Без мережі й оренди.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from gpurunner.core.models import JobHandle
from gpurunner.supervise import htr as htr_mod
from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import CasePlan, Plan


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.delenv("GPURUNNER_OWNER", raising=False)
    monkeypatch.setattr(htr_mod, "_staging_root", lambda: tmp_path / "staging")


class _Backend:
    """Фейк: журнал викликів, мертвий SFTP-обхід із лічильником."""

    def __init__(self, walk_sec: float = 0.0, phase: str = "idle") -> None:
        self.calls: list[str] = []
        self.walk_sec = walk_sec
        self.phase = phase

    def _ssh(self, handle: Any, timeout: int = 30) -> Any:
        raise OSError("ssh недоступний")

    def request_checkpoint(self, handle: Any, *, wait_sec: int = 120) -> str:
        self.calls.append("flush")
        return "done"

    def fetch_outputs(self, handle: Any, dest: Path, **kw: Any) -> None:
        self.calls.append("sftp")
        time.sleep(self.walk_sec)

    def cancel(self, handle: Any) -> None:
        self.calls.append("cancel")


def _sup(tmp_path: Path, backend: Any) -> Supervisor:
    plan = Plan(assets_url="", transport="box", assets_path="/x/a.tgz",
                cases=[CasePlan(case="sprava", pages_url="", n_pages=4,
                                out_dir=str(tmp_path / "out" / "sprava"),
                                pages_path="/x/s.tar")],
                budget_usd=1.0, max_hours=4.0)
    sup = Supervisor(plan, backend=backend, session="S")  # type: ignore[arg-type]
    sup._handle = JobHandle(backend="vast", remote_id="1", job_name="htr_case", gpu="any")
    return sup


def _spy(sup: Supervisor, monkeypatch: pytest.MonkeyPatch, log: list[str]) -> None:
    monkeypatch.setattr(sup, "_flush_checkpoints", lambda: log.append("flush"))
    monkeypatch.setattr(sup, "_quiesce_runner", lambda: log.append("quiesce"))
    monkeypatch.setattr(sup, "_destroy", lambda why: log.append("destroy"))
    monkeypatch.setattr(sup, "_record_box", lambda *a, **k: log.append("record"))

    def fetch(**kw: Any) -> None:
        log.append("fetch:" + ",".join(f"{k}={v}" for k, v in sorted(kw.items())))
    monkeypatch.setattr(sup, "_fetch_queue", fetch)


def test_warm_box_with_queue_home_is_destroyed_without_second_fetch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sup = _sup(tmp_path, _Backend())
    log: list[str] = []
    _spy(sup, monkeypatch, log)
    sup._fetched_idle = True
    sup._last_progress = {"phase": "idle"}
    sup.state.cases[0].status = "done"
    sup.state.cases[0].complete = True

    sup._wrap_up("теплий бокс: довішувати нічого")

    assert log == ["destroy", "record"], "ні заливки, ні зупинки, ні забору"
    assert sup.state.verdict == "ok"
    assert sup.state.exit_code == 0 and not sup.state.human_action_required


def test_warm_box_with_unfetched_case_still_fetches(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Довісок після забору: справа не звірена — «вдома» не вважається."""
    sup = _sup(tmp_path, _Backend())
    log: list[str] = []
    _spy(sup, monkeypatch, log)
    sup._fetched_idle = True
    sup._last_progress = {"phase": "idle"}
    sup.state.cases[0].complete = False

    sup._wrap_up("x")

    assert any(c.startswith("fetch:") for c in log)
    assert sup.state.verdict == "cancelled"


def test_working_box_flushes_checkpoint_before_quiesce_then_fetches_r2_first(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sup = _sup(tmp_path, _Backend())
    log: list[str] = []
    _spy(sup, monkeypatch, log)
    sup._last_progress = {"phase": "running"}

    sup._wrap_up("зупинити")

    assert log[:2] == ["flush", "quiesce"], "заливку не вбивають до її кінця"
    # забір без повторного «залий зараз» (вже залили), через ту саму _fetch_queue
    # (R2 перш за SFTP), на аварійному режимі
    assert log[2] == "fetch:best_effort=True,flush=False"
    assert log[3] == "destroy"
    assert sup.state.verdict == "cancelled"


def test_wrapup_sftp_fallback_has_a_short_ceiling(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Штатна стеля 30 хв; у wrap-up — коротка, решту дістане fetch-ckpt."""
    monkeypatch.setattr(htr_mod, "FETCH_PHASE_MAX_SEC", 30.0)
    monkeypatch.setattr(htr_mod, "WRAPUP_FETCH_MAX_SEC", 0.2)
    backend = _Backend(walk_sec=2.0)
    sup = _sup(tmp_path, backend)
    monkeypatch.setattr(sup, "_flush_checkpoints", lambda: None)
    monkeypatch.setattr(sup, "_quiesce_runner", lambda: None)
    monkeypatch.setattr(sup, "_destroy", lambda why: None)
    monkeypatch.setattr(sup, "_fetch_from_store", lambda staging: False)
    sup._last_progress = {"phase": "running"}

    t0 = time.monotonic()
    sup._wrap_up("x")

    assert time.monotonic() - t0 < 1.5, "SFTP-запас не чекав свого 2-секундного обходу"
    assert "fetch_timeout" in [i.kind for i in sup.state.incidents]
    assert backend.calls.count("sftp") == 1
