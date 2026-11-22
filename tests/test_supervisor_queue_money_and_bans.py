"""Наглядач: провал тому видно одразу, гроші не двояться, бан — лише справжній.

Аудит десяти заходів 12–14.09.2026:

- irnbuv1 q23: 8 томів із 23 впали на боксі, а стан показував їх `running` з
  нулем сторінок до самого забору;
- spr-2461 і cdiak1040-1-2: фініш писав «прогнано повністю за $0.58» при $0.29
  у бюджеті — серцебиття рахувало поточну оренду вдруге, поки йшло гасіння;
- spr-11652: V100 33283 з трьома успіхами дістав `ssh_auth_denied`, інцидент
  пообіцяв «чорний список» (реєстр дав лише попередження), машину викинули на
  весь захід, і справу дочитала GTX 1080 утричі дорожче за сторінку.
"""

from __future__ import annotations

import time
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from gpurunner.core import boxes
from gpurunner.core.backend import SshAuthRejected
from gpurunner.core.boxes import BoxObservation
from gpurunner.supervise import htr as htr_mod
from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import CasePlan, Plan


@pytest.fixture
def sup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Supervisor:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.setenv("GPURUNNER_BOXES_OVERRIDES", str(tmp_path / "boxes.overrides.json"))
    monkeypatch.setattr(htr_mod, "RETRY_SLEEP_SEC", 0)
    plan = Plan(assets_url="https://r2/a.tgz", budget_usd=3.0, max_hours=8.0, max_rents=3,
                cases=[CasePlan(case=c, pages_url=f"https://r2/{c}.tar", n_pages=200,
                                out_dir=str(tmp_path / c))
                       for c in ("t01", "t02", "t03")])
    s = Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]
    monkeypatch.setattr(s, "_destroy_orphan", lambda e: None)
    return s


def _incidents(s: Supervisor) -> list[tuple[str, str]]:
    out = []
    for inc in s.state.incidents:
        get = inc.get if isinstance(inc, dict) else lambda k, i=inc: getattr(i, k, "")
        out.append((get("kind"), get("action") or ""))
    return out


# ---- провал тому видно до забору -------------------------------------------------


def test_box_results_show_a_failed_volume_before_the_fetch(sup: Supervisor) -> None:
    progress = {
        "phase": "running", "case_index": 3, "pages_done": 40,
        "results": [
            {"index": 1, "case": "t01", "complete": True, "n_pages_txt": 200, "attempt": 1},
            {"index": 2, "case": "t02", "complete": False, "attempt": 1, "retry_pending": True,
             "error": "RuntimeError('curl https://r2/t02.tar провалився (rc=28)')"},
        ],
    }
    sup._absorb_queue(progress, "йде")
    t01, t02, t03 = sup.state.cases
    assert t01.pages_done == 200 and "повна" in t01.detail
    assert t02.status == "failed" and "rc=28" in t02.detail
    assert t03.status == "running" and t03.pages_done == 40
    failed = [a for k, a in _incidents(sup) if k == "case_failed"]
    assert len(failed) == 1 and "повторить" in failed[0]

    sup._absorb_queue(progress, "йде")          # той самий провал удруге — без нового інциденту
    assert [k for k, _ in _incidents(sup)].count("case_failed") == 1

    progress["results"][1] = {"index": 2, "case": "t02", "complete": True,
                              "n_pages_txt": 200, "attempt": 2}
    sup._absorb_queue(progress, "йде")
    assert sup.state.cases[1].status == "running", "повтор удався — том більше не failed"


def test_a_fetched_verdict_is_not_overwritten_by_box_results(sup: Supervisor) -> None:
    sup.state.cases[0].status = "done"
    sup._absorb_queue({"phase": "running", "case_index": 2, "results": [
        {"index": 1, "case": "t01", "complete": False, "attempt": 1, "error": "x"}]}, "йде")
    assert sup.state.cases[0].status == "done"


# ---- гроші під час гасіння ---------------------------------------------------------


def test_money_is_not_counted_twice_while_the_box_is_being_destroyed(
    sup: Supervisor, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Backend:
        def cancel(self, handle) -> None:
            sup._accrue()      # серцебиття з окремого потоку саме в цю мить

    sup.backend = _Backend()  # type: ignore[assignment]
    sup._handle = types.SimpleNamespace(id="h1", remote_id="1", status=None)  # type: ignore[assignment]
    sup._offer = {"machine_id": 1, "dph_total": 0.36}
    sup.settled_usd = 0.01
    sup._rented_at = time.monotonic() - 3600
    monkeypatch.setattr(htr_mod.manifest, "update", lambda handle: None)

    sup._destroy("черга завершена")

    assert sup.spent_usd == pytest.approx(0.37, abs=0.005)
    sup._accrue()
    assert sup.spent_usd == pytest.approx(0.37, abs=0.005)


# ---- що реєстр насправді зробив із машиною ---------------------------------------


def _candidate(mid: int = 33283) -> types.SimpleNamespace:
    return types.SimpleNamespace(offer={"machine_id": mid, "dph_total": 0.215}, machine_id=mid)


def _auth_denied() -> SshAuthRejected:
    e = SshAuthRejected("хост ssh1.vast.ai:30578 відхилив наш ключ (Authentication failed.)")
    e.instance_id = "51030578"  # type: ignore[attr-defined]
    return e


def test_a_rejected_key_on_a_proven_machine_gets_one_more_try(sup: Supervisor) -> None:
    boxes.record(BoxObservation(machine_id=33283, outcome="ok",
                                ts=(datetime.now(tz=UTC) - timedelta(days=1)).isoformat()))

    assert sup._after_failed_submit(_auth_denied(), _candidate(), time.monotonic()) == "retry"
    action = _incidents(sup)[-1][1]
    assert "попередження" in action and "ще одна спроба" in action
    assert "чорний список" not in action
    assert 33283 not in sup._failed_machines

    # друга відмова в тому самому заході — без поблажок
    assert sup._after_failed_submit(_auth_denied(), _candidate(), time.monotonic()) == "next"
    assert 33283 in sup._failed_machines


def test_a_rejected_key_on_an_unknown_machine_is_skipped_but_not_banned(sup: Supervisor) -> None:
    """Невідомій машині повтору в тому ж заході немає, але бан — лише з другого удару."""
    assert sup._after_failed_submit(_auth_denied(), _candidate(4711), time.monotonic()) == "next"
    assert "попередження" in _incidents(sup)[-1][1]
    assert 4711 in sup._failed_machines
    assert 4711 not in boxes.banned_ids()


def test_a_failed_attempt_counts_as_one_wasted_rent(sup: Supervisor) -> None:
    sup._after_failed_submit(_auth_denied(), _candidate(4711), time.monotonic())
    assert (sup.rents, sup.rents_wasted) == (1, 1)
    # успішний сабміт закриває попередню ВІДКРИТУ оренду — невдала такою не була
    sup._close_rent()
    assert sup.rents_wasted == 1
