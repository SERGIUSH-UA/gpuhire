"""Подвійний старт неможливий, стан не бреше про фазу, гроші й теку кадрів.

Усе тут куплено одним днем (10.09.2026, справа 315-1-8591):

- відчеплена задача ставила тригер «+1 хв» поруч із `/run` — черга FS
  стартувала двічі, і замок справи пускав другу копію тієї ж сесії як «свою»;
- фаза весь захід показувала `renting`, гроші — `$0.00` під час підйому боксу;
- після забору в меті лишився `case_dir` боксу, і теку лагодили руками.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from gpurunner.core import locks
from gpurunner.core.locks import LockBusy
from gpurunner.supervise import state as state_mod
from gpurunner.supervise.htr import Supervisor, _stamp_case_key
from gpurunner.supervise.state import EXIT_FAILED
from tests.test_supervise_flow import DONE, OFFER, FakeBackend, make_plan


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.setenv("GPURUNNER_BOXES_OVERRIDES", str(tmp_path / "over.json"))


def _plant_lock(resource: str, *, owner: str, session: str, pid: int) -> None:
    path = locks._path(resource)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "resource": resource, "owner": owner, "session": session,
        "pid": pid, "ts": time.time(), "ttl_sec": 3600,
    }), encoding="utf-8")


# ---- замок --------------------------------------------------------------


def test_same_session_from_another_live_process_is_refused() -> None:
    """🔴 Та сама сесія з ІНШОГО живого процесу — подвійний старт, не оновлення."""
    _plant_lock("case:spr-8591", owner="htr-A", session="A1", pid=os.getppid())
    with pytest.raises(LockBusy):
        locks.acquire("case:spr-8591", owner="htr-A", session="A1")


def test_same_session_after_its_process_died_is_taken_over() -> None:
    """Перезапуск після смерті наглядача — законний: мертвий pid не тримає."""
    _plant_lock("case:spr-8591", owner="htr-A", session="A1", pid=2_000_000_000)
    info = locks.acquire("case:spr-8591", owner="htr-A", session="A1")
    assert info.pid == os.getpid()


def test_second_supervisor_of_a_session_does_not_touch_the_live_state(
        tmp_path: Path) -> None:
    """Другий наглядач тієї ж сесії виходить ДО першого запису стану й оренди."""
    backend = FakeBackend(pages=100, texts=100)
    sup = Supervisor(make_plan(tmp_path), session="htr-dup",
                     backend=backend, tick_sec=0)  # type: ignore[arg-type]
    _plant_lock("session:htr-dup", owner=sup._owner, session="htr-dup",
                pid=os.getppid())

    assert sup.run() == EXIT_FAILED
    assert backend.calls == []
    assert not state_mod.state_path("htr-dup").exists()


# ---- фаза, гроші, серцебиття ------------------------------------------------


def test_phase_leaves_renting_once_the_box_is_up(tmp_path: Path) -> None:
    backend = FakeBackend(pages=100, texts=100)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    seen: list[str] = []

    def read() -> tuple[dict, bool]:
        seen.append(sup.state.phase)
        return DONE, True

    sup.backend = backend  # type: ignore[assignment]
    sup._read_progress = read  # type: ignore[method-assign]
    sup._progress_age = lambda p: 1.0  # type: ignore[method-assign]
    sup._instance_state = lambda: "running"  # type: ignore[method-assign]

    assert sup.run() == 0
    assert seen and seen[0] == "setup"
    assert "renting" not in seen
    assert sup.state.phase == "finished"


def test_heartbeat_counts_money_while_the_box_is_booting(tmp_path: Path) -> None:
    """🔴 Інстанс створено годину тому, основний цикл ще чекає SSH — серцебиття
    мусить показати гроші, а не `$0.00`."""
    sup = Supervisor(make_plan(tmp_path), backend=FakeBackend(pages=1, texts=1),  # type: ignore[arg-type]
                     tick_sec=0)
    sup._pending = {"instance_id": "1", "dph": 0.30, "since": time.monotonic() - 3600}
    sup._beat()

    saved = json.loads(state_mod.state_path(sup.state.session).read_text(encoding="utf-8"))
    assert saved["budget"]["spent_usd"] == pytest.approx(0.30, abs=0.01)
    assert saved["heartbeat"]


def test_billing_start_accrues_immediately(tmp_path: Path) -> None:
    sup = Supervisor(make_plan(tmp_path), backend=FakeBackend(pages=1, texts=1),  # type: ignore[arg-type]
                     tick_sec=0)
    sup.settled_usd = 0.05
    sup._note_billing_started("77", OFFER)
    assert sup.state.budget["spent_usd"] == pytest.approx(0.05, abs=0.001)
    assert sup.state.box and sup.state.box["billing"] is True


def test_live_state_always_says_the_supervisor_is_alive() -> None:
    """Поле, яке з'являється лише в біді, агент не відрізнить від «не питали»."""
    fresh = state_mod.SupervisorState(session="s").to_dict()
    out = state_mod.diagnose(fresh)
    assert out["supervisor_alive"] is True
    assert out["quiet_min"] == 0.0


# ---- тека кадрів після забору ---------------------------------------------------


def _meta(d: Path, **fields: object) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    mp = d / "_htr_meta.json"
    mp.write_text(json.dumps({"pages": {"0001.jpg": {}}, **fields}), encoding="utf-8")
    return mp


def test_box_path_in_meta_is_replaced_by_the_local_frames_dir(tmp_path: Path) -> None:
    frames = tmp_path / "raw" / "spr-8591"
    frames.mkdir(parents=True)
    out = tmp_path / "reports" / "spr-8591"
    mp = _meta(out, case_dir="/tmp/htrcase/pages_dl_01")
    voice = _meta(tmp_path / "reports" / "spr-8591-diak_v4", case_dir="/tmp/htrcase/pages_dl_01")

    assert _stamp_case_key(out, "DAHMO/315/8591", local_dir=str(frames)) == 2
    for path in (mp, voice):
        meta = json.loads(path.read_text(encoding="utf-8"))
        assert meta["case_dir"] == str(frames).replace("\\", "/")
        assert meta["case_key"] == "DAHMO/315/8591"


def test_live_case_dir_is_left_alone(tmp_path: Path) -> None:
    """Жива тека — рішення людини або ремонту; автомат її не перебиває."""
    mine = tmp_path / "chosen"
    mine.mkdir()
    other = tmp_path / "plan_dir"
    other.mkdir()
    out = tmp_path / "reports" / "x"
    mp = _meta(out, case_dir=str(mine), case_key="K/1/2")

    assert _stamp_case_key(out, "K/1/2", local_dir=str(other)) == 0
    assert json.loads(mp.read_text(encoding="utf-8"))["case_dir"] == str(mine)


def test_missing_local_dir_does_not_overwrite_anything(tmp_path: Path) -> None:
    out = tmp_path / "reports" / "x"
    mp = _meta(out, case_dir="/tmp/htrcase/p")
    assert _stamp_case_key(out, "", local_dir=str(tmp_path / "nope")) == 0
    assert json.loads(mp.read_text(encoding="utf-8"))["case_dir"] == "/tmp/htrcase/p"
