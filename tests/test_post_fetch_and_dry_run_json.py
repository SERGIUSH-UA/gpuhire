"""Облік після забору запускає сам наглядач; кошторис — одним JSON (10.09.2026).

Обгортка «одна команда» вирішує про автозапуск за кошторисом сухого прогону, а
розбирати таблицю rich рядками означало б ламати автозапуск кожною новою
колонкою. Облік після заходу (реєстр справ, індекс тексту) жив на пам'яті
агента й після 8591 робився руками.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import load_plan
from tests.test_supervise_flow import DONE, FakeBackend, make_plan


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.setenv("GPURUNNER_BOXES_OVERRIDES", str(tmp_path / "over.json"))


def _hook(marker: Path) -> dict:
    return {"cmd": [sys.executable, "-c",
                    f"open(r'{marker}', 'w', encoding='utf-8').write('ok')"]}


def _run(sup: Supervisor, backend: FakeBackend) -> int:
    sup.backend = backend  # type: ignore[assignment]
    sup._read_progress = lambda: (DONE, True)  # type: ignore[method-assign]
    sup._progress_age = lambda p: 1.0  # type: ignore[method-assign]
    sup._instance_state = lambda: "running"  # type: ignore[method-assign]
    return sup.run()


def test_post_fetch_runs_after_a_complete_run(tmp_path: Path) -> None:
    marker = tmp_path / "cases_built.txt"
    plan = replace(make_plan(tmp_path), post_fetch=[_hook(marker)])
    backend = FakeBackend(pages=100, texts=100)
    assert _run(Supervisor(plan, backend=backend, tick_sec=0), backend) == 0  # type: ignore[arg-type]
    assert marker.read_text(encoding="utf-8") == "ok"


def test_post_fetch_does_not_run_on_an_incomplete_case(tmp_path: Path) -> None:
    """Неповна справа в реєстрі — застарілий зріз, що виглядає як відповідь."""
    marker = tmp_path / "cases_built.txt"
    plan = replace(make_plan(tmp_path), post_fetch=[_hook(marker)])
    backend = FakeBackend(pages=100, texts=99, complete=False)
    _run(Supervisor(plan, backend=backend, tick_sec=0), backend)  # type: ignore[arg-type]
    assert not marker.exists()


def test_a_failing_hook_is_an_incident_not_a_failed_run(tmp_path: Path) -> None:
    plan = replace(make_plan(tmp_path),
                   post_fetch=[{"cmd": [sys.executable, "-c", "raise SystemExit(4)"]}])
    backend = FakeBackend(pages=100, texts=100)
    sup = Supervisor(plan, backend=backend, tick_sec=0)  # type: ignore[arg-type]
    assert _run(sup, backend) == 0
    assert any(i.kind == "post_fetch_failed" for i in sup.state.incidents)


def test_plan_loader_knows_post_fetch(tmp_path: Path) -> None:
    raw = {
        "assets_url": "https://r2/assets.tgz", "budget_usd": 1.0, "max_hours": 4.0,
        "cases": [{"case": "spr-1", "pages_url": "https://r2/x.tar", "n_pages": 3,
                   "out_dir": str(tmp_path)}],
        "post_fetch": [{"cmd": ["consumer.exe", "cases", "build"], "cwd": str(tmp_path)},
                       {"cwd": "без команди — відкидається"}],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    plan = load_plan(path)
    assert plan.post_fetch == [raw["post_fetch"][0]]
    assert not [w for w in plan.warnings if "post_fetch" in w]


def test_dry_run_json_is_one_parseable_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    from gpurunner import cli
    from gpurunner.backends import vast

    monkeypatch.setattr(vast, "VastBackend",
                        lambda: FakeBackend(pages=100, texts=100, credit=6.5))
    cli._htr_dry_run(make_plan(tmp_path), json_out=True)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    data = json.loads(out[0])
    assert data["credit"] == 6.5 and data["pages"] == 100 and not data["empty"]
    assert data["best"]["cost"] == pytest.approx(0.20)
    assert data["best"]["shards"] >= 1
