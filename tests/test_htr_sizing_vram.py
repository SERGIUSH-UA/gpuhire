"""VRAM на шард: одне виміряне число, і чому воно НЕ виводиться з геометрії.

🔴 На цьому вже помилялись у ДВА протилежні боки:

- **замало.** Дефолтні 2.5 стоять нижче базового споживання шарда (3.2-3.6 ГБ
  завжди), тож 8 шардів × 3.3 = 26 ГБ на 24-ГБ карті дали 62% сторінок у збоях;
- **забагато.** Поріг 4.5, порахований «за площею розвороту», відправив захід на
  RTX 4060 Ti з трьома шардами замість RTX 3090 із сімома — утричі повільніше за
  ті самі гроші. Завищений поріг ріже не лише флот на обраній машині: скоринг
  ділить на нього ще на РАНЖУВАННІ РИНКУ.
"""

from __future__ import annotations

from pathlib import Path

from gpurunner.core.htr_sizing import GB_PER_SHARD, gb_per_shard_for, plan_sizing
from gpurunner.supervise.htr import need_from_plan
from gpurunner.supervise.plan import CasePlan, Plan


def _plan(*cases: CasePlan) -> Plan:
    return Plan(assets_url="https://r2/a.tgz", cases=list(cases),
                budget_usd=3.0, max_hours=8.0)


def _case(name: str, *, mpx: float = 0.0, aspect: float = 0.0,
          pages: int = 100) -> CasePlan:
    return CasePlan(case=name, pages_url="https://r2/x.tar", n_pages=pages,
                    out_dir=f"E:/prostir/reports/htr/{name}",
                    frame_mpx_median=mpx, frame_aspect_median=aspect)


def test_the_default_is_above_the_base_consumption_of_a_shard() -> None:
    """🔴 Шард тримає 3.2-3.6 ГБ ЗАВЖДИ — це ваги PARSeq + kraken, а не функція
    матеріалу. Поріг, нижчий за це, не «ризикований компроміс», а арифметика,
    яка не сходиться: місця не вистачить навіть на порожній сторінці."""
    assert GB_PER_SHARD >= 3.2


def test_area_lowers_the_threshold_but_aspect_never_does() -> None:
    """🔴 Урок 19.08.2026 лишається: поріг «4.5 за площею РОЗВОРОТУ» відправив
    захід на слабшу карту, бо рядки з форми кадру не виводяться. Але 05.09.2026
    std160 показав інше: на 7-Мпікс кадрах шард тримає 1.1–1.25 ГБ, і 3.3 там —
    половина флоту даремно. Тому площа ОПУСКАЄ поріг до заміряного, aspect не
    діє, а невідомий матеріал лишається важким."""
    assert gb_per_shard_for() == GB_PER_SHARD                      # не міряли
    assert gb_per_shard_for(7.2, 0.75) == gb_per_shard_for(7.2, 1.26)
    assert gb_per_shard_for(7.2, 0.75) < GB_PER_SHARD
    # 16-Мпікс розвороти: заміряне зайняття карти після 04.09 — до ~1.7 ГБ на
    # шард (див. `GB_PER_MPX`); прогноз лежить над ним, а не на старих 3.3.
    assert 1.7 <= gb_per_shard_for(16.0, 1.2) < GB_PER_SHARD


def test_the_default_keeps_a_24gb_card_competitive() -> None:
    """Приймач із заміру 19.08.2026: на RTX 3090 поріг мусить лишати флот, який
    б'є дрібну карту. При 4.5 виходило 4 шарди й програш RTX 4060 Ti."""
    assert plan_sizing(cores=64, vram_gb=24.0).shards >= 6


def test_geometry_still_travels_in_the_plan() -> None:
    """Геометрію міряємо й возимо — вона потрібна у звіті й у калібруванні.
    Не потрібна вона рівно в одному місці: у виборі порога."""
    case = _case("spr-1649", mpx=7.1, aspect=1.26)
    assert case.frame_mpx_median == 7.1
    need = need_from_plan(_plan(case), pages=100)
    assert need.gb_per_shard == gb_per_shard_for(7.1)
    assert need.frame_mpx == 7.1


def test_an_explicit_knob_still_wins() -> None:
    """Дослідник, що поставив число руками, знає про матеріал більше за нас."""
    plan = _plan(_case("spr-1"))
    object.__setattr__(plan, "gb_per_shard", 6.0)
    assert need_from_plan(plan, pages=100).gb_per_shard == 6.0


def test_the_calibration_registry_records_what_it_cost(tmp_path: Path,
                                                       monkeypatch) -> None:
    """🔴 Числа тут стоять на замірах, кожен з яких оплачений падінням прогону, а
    записаний був у чиюсь пам'ять сесії. Реєстр робить наступне уточнення
    питанням до ДАНИХ — і саме він мав би спіймати «4.5 за площею» раніше."""
    import json

    from gpurunner.supervise import htr as htr_mod
    from gpurunner.supervise.decide import Obs

    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(htr_mod, "registry_dir", lambda: tmp_path / "repo")

    plan = _plan(_case("spr-1739", mpx=7.1, aspect=1.26))
    object.__setattr__(plan, "gb_per_shard", 3.3)
    sup = htr_mod.Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]
    sup._probe = {"vram_total_gb": 24.0, "n_gpus": 1}
    sup._bump_gb_per_shard(Obs(progress={"oom_events": 59, "pages_failed": 94}))

    rows = [json.loads(line) for line
            in (tmp_path / "repo" / "shard_calibration.jsonl").read_text(
                encoding="utf-8").splitlines()]
    assert rows[0]["frame_aspect"] == 1.26
    assert rows[0]["gb_per_shard"] == 3.3
    assert rows[0]["oom_events"] == 59
    assert rows[0]["ok"] is False


def test_the_step_up_after_oom_is_measured_not_preemptive(tmp_path: Path,
                                                          monkeypatch) -> None:
    """🔴 Запас наперед коштує грошей на КОЖНІЙ оренді, а крок після ФАКТИЧНОГО
    OOM — лише один цикл. Тому підіймаємось за фактом, а не про всяк випадок."""
    from gpurunner.supervise import htr as htr_mod
    from gpurunner.supervise.decide import Obs

    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(htr_mod, "registry_dir", lambda: tmp_path / "repo")

    plan = _plan(_case("spr-1"))
    sup = htr_mod.Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]
    sup._probe = {"vram_total_gb": 24.0, "n_gpus": 1}

    # до OOM ручки немає, геометрії теж → дефолт невідомого матеріалу, і саме
    # він їде на бокс (06.09.2026), а не нуль, який раннер тлумачив по-своєму
    assert sup._gb_per_shard() == GB_PER_SHARD
    sup._bump_gb_per_shard(Obs(progress={"oom_events": 59, "pages_failed": 94}))
    assert sup._gb_per_shard() > GB_PER_SHARD
