"""Збій сторінки, що йде за справою, не запускає коло переоренд.

21-22.11.2026: кадри пачки з іменами понад 255 байт давали на машині
`Errno 36: File name too long`. Флот сипався, наглядач переорендовував, нова
машина починала з тієї самої справи — і 1437 спроб пройшли колом, а кожна
машина діставала удар у реєстрі ні за що.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gpurunner.supervise.decide import Obs

LONG = "0001.jpg: OSError: [Errno 36] File name too long: '/workspace/data/derived/htr_seg/x'"


def _sup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from gpurunner.supervise.htr import Supervisor
    from gpurunner.supervise.plan import CasePlan, Plan

    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    plan = Plan(assets_url="https://r2/a.tgz",
                cases=[CasePlan(case="c", pages_url="https://r2/c.tar", n_pages=10, out_dir="")],
                budget_usd=3.0, max_hours=8.0)
    return Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]


def _failing(failed: int = 367, done: int = 0) -> Obs:
    return Obs(progress={"phase": "running", "pages_done": done, "pages_failed": failed})


def _errs(*texts: str) -> list[str]:
    return [t.split(": ", 1)[-1] for t in texts]


def test_a_dominant_name_too_long_under_a_failing_fleet_follows_the_case(
        tmp_path: Path, monkeypatch) -> None:
    sup = _sup(tmp_path, monkeypatch)
    assert sup._page_error_follows_the_case(_failing(), _errs(LONG, LONG, "x: інше"))


def test_one_old_long_name_does_not_stop_an_unrelated_fleet_death(
        tmp_path: Path, monkeypatch) -> None:
    """`last_error` шарда не скидається: давнє задовге ім'я на одному шарді не
    робить причиною смерть флоту від CUDA на решті."""
    sup = _sup(tmp_path, monkeypatch)
    errs = _errs(LONG, "a: CUDA error: illegal memory access",
                 "b: CUDA error: illegal memory access")
    assert not sup._page_error_follows_the_case(_failing(), errs)


def test_without_the_fail_ratio_rule_firing_the_case_is_not_blamed(
        tmp_path: Path, monkeypatch) -> None:
    sup = _sup(tmp_path, monkeypatch)
    assert not sup._page_error_follows_the_case(_failing(failed=3, done=400), _errs(LONG))
    assert not sup._page_error_follows_the_case(_failing(), [])


def test_the_loop_stops_with_the_case_named_instead_of_renting_again(
        tmp_path: Path, monkeypatch) -> None:
    """На рівні циклу: замість нової оренди — забір, гасіння, `incomplete`."""
    from gpurunner.core import boxes
    from gpurunner.supervise.htr import Supervisor
    from tests.test_supervise_flow import FakeBackend, make_plan, run

    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.setenv("GPURUNNER_BOXES_OVERRIDES", str(tmp_path / "over.json"))
    backend = FakeBackend(pages=100, texts=0, complete=False)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    recorded: list[str] = []
    sup._record_box = lambda outcome, why: recorded.append(outcome)  # type: ignore[method-assign]
    progress = {"phase": "running", "case": "pachka-уманські", "n_pages_expected": 100,
                "pages_done": 0, "pages_failed": 367, "pages_per_hour": 0,
                "shards": [{"alive": True, "rc": None, "last_error": LONG}] * 4}

    run(sup, backend, progress)

    assert backend.calls.count("rent") == 1, "нової оренди бути не мусить"
    assert sup.state.verdict == "incomplete" and sup.state.human_action_required
    assert "pachka-уманські" in (sup.state.human_action or "")
    assert recorded and all(o in boxes.NEUTRAL for o in recorded)
    assert any(i.kind == "page_error_follows_case" for i in sup.state.incidents)
