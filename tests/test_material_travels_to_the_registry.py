"""Матеріал заходу доходить до стану, реєстру й плану (06.09.2026).

- `pages_per_hour_case` без рядків не давав перевірити «чи темп від матеріалу»
  (`pages_per_hour_mpx` за добу назбирав нуль рядків);
- поле `BoxObservation.case` існувало, але жоден виклик його не заповнював;
- стан сесії не знав шифри — зшивка йшла лише через `out_dir`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpurunner.core import boxes
from gpurunner.htr.plan_build import lines_per_page_from_meta
from gpurunner.supervise.htr import Supervisor, need_from_plan, plan_material
from gpurunner.supervise.plan import CasePlan, Plan


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))


def _meta(out_dir: Path, lines: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_htr_meta.json").write_text(json.dumps({
        "version": 1, "pages": {f"{i:04d}.jpg": {"lines": n} for i, n in enumerate(lines, 1)}}),
        encoding="utf-8")


def _plan(tmp_path: Path, *, lines: float = 0.0, params: dict | None = None) -> Plan:
    return Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case="std160", pages_url="https://r2/p.tar", n_pages=160,
                        out_dir=str(tmp_path / "out" / "std160"), case_key="DAHmO/230/7",
                        frame_mpx_median=7.2, lines_per_page_median=lines)],
        budget_usd=1.0, max_hours=1.5, params=params or {},
    )


def test_lines_from_a_previous_run_are_read_from_the_meta(tmp_path: Path) -> None:
    """spr-8248: 112 рядків/стор і 247 стор/год проти 4594 на 230-1-24 із 35
    рядками — той самий бокс, та сама модель; план про рядки не знав."""
    out = tmp_path / "out" / "spr-8248"
    # 250 = сторінка з піднятою стелею: з 10.09.2026 РАХУЄТЬСЯ (доти відкидалась,
    # і з вибірки зникали саме найгустіші аркуші); 0 = порожня, не рахується
    _meta(out, [110, 112, 115, 250, 0])
    assert lines_per_page_from_meta(out) == 113.5
    assert lines_per_page_from_meta(tmp_path / "немає") == 0.0


def test_plan_material_prefers_the_knob_over_the_meta(tmp_path: Path) -> None:
    plan = _plan(tmp_path, lines=70)
    assert plan_material(plan) == (7.2, 7.2, 70)
    knob = _plan(tmp_path, lines=70, params={"lines_per_page": "40"})
    assert plan_material(knob) == (7.2, 7.2, 40)
    need = need_from_plan(plan, pages=160)
    assert need.lines_per_page == 70 and need.frame_mpx == 7.2
    assert not need.lines_assumed


def test_unknown_density_is_assumed_and_says_so(tmp_path: Path) -> None:
    """🔴 Невідома густина — не якір кривої 123.6, на якому ціль 5 000 недосяжна
    для будь-якої машини ринку (spr-43: прогноз 2 347 при факті 6 517), а
    припущення, ПОЗНАЧЕНЕ як припущення."""
    from gpurunner.core.htr_sizing import LINES_PER_PAGE_ASSUMED

    need = need_from_plan(_plan(tmp_path), pages=160)
    assert need.lines_assumed
    assert need.lines_per_page == LINES_PER_PAGE_ASSUMED


def test_one_heavy_case_does_not_decide_the_whole_queue(tmp_path: Path) -> None:
    """🔴🔴 23.09.2026: одна справа на 36 Мпікс і 251 рядок із 194 поставила
    6 ГБ на шард і густину 251 УСІЙ черзі — 1 шард на 8-ГБ карту, час заходу
    завищено в 3.7 раза. Матеріал черги — зважений сторінками."""
    typical = [CasePlan(case=f"c{i}", pages_url="https://r2/p.tar", n_pages=25,
                        out_dir=str(tmp_path / f"c{i}"), frame_mpx_median=12.3,
                        frame_mpx_p95=12.5, lines_per_page_median=39)
               for i in range(20)]
    heavy = CasePlan(case="spr-2351", pages_url="https://r2/p.tar", n_pages=20,
                     out_dir=str(tmp_path / "heavy"), frame_mpx_median=30.0,
                     frame_mpx_p95=36.0, lines_per_page_median=251)
    plan = Plan(assets_url="https://r2/a.tgz", cases=[*typical, heavy],
                budget_usd=3.0, max_hours=8.0)
    mpx_vram, mpx_typ, lines = plan_material(plan)
    assert mpx_vram == 12.5, "VRAM — від p90 сторінок черги, а не від найважчої справи"
    assert lines < 50, f"густина черги {lines:.0f} — дві справи не вирішують за всіх"
    assert mpx_typ < 14


def test_state_carries_the_case_key(tmp_path: Path) -> None:
    sup = Supervisor(_plan(tmp_path), backend=object(), session="S")  # type: ignore[arg-type]
    assert sup.state.cases[0].case_key == "DAHmO/230/7"


def test_observation_carries_lines_and_case(tmp_path: Path, monkeypatch) -> None:
    """Після забору мета лежить на диску — рядки читаються з неї; поле `case`
    у реєстрі перестає бути порожнім."""
    plan = _plan(tmp_path)
    _meta(Path(plan.cases[0].out_dir), [68, 70, 72])
    sup = Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]
    sup._probe = {"vram_total_gb": 24.0, "n_gpus": 1, "cores": 24.0}
    sup._measured_pph = 4600.0
    recorded: list = []
    monkeypatch.setattr(boxes, "record", lambda obs: recorded.append(obs))
    sup._record_offer({"id": 1, "machine_id": 14096, "gpu_name": "RTX 3090",
                       "num_gpus": 1, "dph_total": 0.15}, "ok", "тест")
    assert recorded, "запис у реєстр не зроблено"
    obs = recorded[0]
    assert obs.case == "std160"
    assert obs.measured["pages_per_hour_lines"] == 70
    assert obs.measured["pages_per_hour_case"] == "std160"


def test_lines_fall_back_to_the_plan_before_the_fetch(tmp_path: Path) -> None:
    sup = Supervisor(_plan(tmp_path, lines=70), backend=object(), session="S")  # type: ignore[arg-type]
    assert sup._lines_per_page_done() == 70


def test_vram_per_shard_follows_the_p95_frame_not_the_median(tmp_path: Path) -> None:
    """Справа з медіаною 7 Мпікс і розворотами по 16 серед сторінок дала б OOM
    саме на розворотах, якби флот розкладали під 1.9 ГБ від медіани."""
    from gpurunner.core.htr_sizing import gb_per_shard_for

    mixed = Plan(
        assets_url="https://r2/a.tgz", budget_usd=1.0, max_hours=1.0,
        cases=[CasePlan(case="mixed", pages_url="https://r2/p.tar", n_pages=100,
                        out_dir=str(tmp_path / "mixed"),
                        frame_mpx_median=7.2, frame_mpx_p95=16.0)])
    assert plan_material(mixed)[0] == 16.0
    assert need_from_plan(mixed, pages=100).gb_per_shard == gb_per_shard_for(16.0)
    old_plan = Plan(
        assets_url="https://r2/a.tgz", budget_usd=1.0, max_hours=1.0,
        cases=[CasePlan(case="old", pages_url="https://r2/p.tar", n_pages=100,
                        out_dir=str(tmp_path / "old"), frame_mpx_median=7.2)])
    assert plan_material(old_plan)[0] == 7.2
