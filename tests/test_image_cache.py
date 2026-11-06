"""Кеш образу на хості — у виборі машини.

Docker тримає раз стягнутий образ: хост, що вже піднімав наш образ, стартує
за хвилину (V100 у Техасі — ~55 с), а новий тягне його 2–7 хв або не дотягує
зовсім (перепис 04.10.2026: на RTX 3090 піднялось 3 з 7). Реєстр пам'ятає
підйом кожної машини, але доти — без образу: швидкий старт зі СТАРИМ образом
давав машині бали й для нового, якого в неї немає.

Тут — чотири обіцянки:
- запис оренди називає образ, а згортка тримає підйом окремо на кожен образ;
- записи без образу — старий образ Vast за замовчуванням;
- оцінка оферту бере підйом лише для образу заходу й позначає «у кеші»;
- потреба заходу знає свій образ так само, як оренда.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gpurunner.backends import vast
from gpurunner.core import boxes
from gpurunner.core.boxes import BoxObservation, BoxVerdict
from gpurunner.core.htr_sizing import OVERHEAD_SEC_COLD
from gpurunner.core.offer_score import Need, boot_for, score_offer

T0 = datetime(2026, 10, 4, 12, tzinfo=UTC)
V100 = {"id": 1, "machine_id": 30970, "gpu_name": "Tesla V100", "num_gpus": 1,
        "cpu_cores_effective": 20, "gpu_ram": 16 * 1024, "cpu_ram": 128 * 1024,
        "disk_space": 200, "dph_total": 0.125, "reliability2": 0.99,
        "inet_down": 900, "geolocation": "Texas, US"}
NEED = Need(pages=500, max_hours=4, budget_usd=2, image=vast.ENGINE_IMAGE)


def _obs(minutes: int, boot: float, image: str | None) -> tuple[datetime, BoxObservation]:
    measured: dict = {"boot_sec": boot}
    if image:
        measured["image"] = image
    ts = T0 + timedelta(minutes=minutes)
    return ts, BoxObservation(machine_id=30970, outcome=boxes.OK, ts=ts.isoformat(),
                              measured=measured)


def test_legacy_records_belong_to_the_old_default_image() -> None:
    """Записи до 0.6.2 образу не називали — тоді бокс завжди піднімався образом за
    замовчуванням, і саме йому належить їхній підйом."""
    assert boxes.LEGACY_IMAGE == vast.DEFAULT_IMAGE


def test_boot_is_folded_per_image_latest_wins() -> None:
    got = boxes._boot_by_image([_obs(0, 40, None), _obs(10, 300, vast.ENGINE_IMAGE),
                                _obs(20, 55, vast.ENGINE_IMAGE)])
    assert got == {vast.DEFAULT_IMAGE: 40.0, vast.ENGINE_IMAGE: 55.0}


def test_registry_verdict_carries_boots_per_image(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.setenv("GPURUNNER_BOXES_OVERRIDES", str(tmp_path / "overrides.json"))
    for _ts, obs in (_obs(0, 40, None), _obs(30, 55, vast.ENGINE_IMAGE)):
        boxes.record(obs)
    v = boxes.verdicts(now=T0 + timedelta(hours=1))[30970]
    assert v.best_measured["boot_by_image"] == {vast.DEFAULT_IMAGE: 40.0,
                                                vast.ENGINE_IMAGE: 55.0}


def _verdict(boots: dict[str, float]) -> BoxVerdict:
    return BoxVerdict(machine_id=30970, state="starred", reason="1 успішних",
                      best_measured={"boot_sec": 40.0, "boot_by_image": boots})


def test_cached_image_starts_on_its_measured_boot() -> None:
    assert boot_for(_verdict({vast.ENGINE_IMAGE: 55.0}), vast.ENGINE_IMAGE) == (55.0, True)


def test_a_fast_boot_with_another_image_says_nothing_about_ours() -> None:
    """🔴 Машина швидко піднімала СТАРИЙ образ — новий вона тягне з нуля:
    холодний старт, без позначки кешу."""
    assert boot_for(_verdict({vast.DEFAULT_IMAGE: 40.0}), vast.ENGINE_IMAGE) == (0.0, False)
    assert boot_for(None, vast.ENGINE_IMAGE) == (0.0, False)


def test_need_without_image_keeps_the_old_reading() -> None:
    assert boot_for(_verdict({}), "") == (40.0, True)


def test_cached_host_wins_the_cold_start_in_hours_and_money() -> None:
    """Та сама машина за ту саму ціну: з образом у кеші захід коротший рівно на
    різницю холодного старту й заміряного підйому — і це бали у виборі."""
    cached = score_offer(V100, NEED, _verdict({vast.ENGINE_IMAGE: 55.0}))
    cold = score_offer(V100, NEED, _verdict({vast.DEFAULT_IMAGE: 40.0}))
    assert cached.image_cached and not cold.image_cached
    assert cold.hours - cached.hours == pytest.approx((OVERHEAD_SEC_COLD - 55) / 3600, abs=1e-6)
    assert cached.cost < cold.cost


def test_need_from_plan_knows_the_image_the_box_will_run() -> None:
    from gpurunner.supervise.htr import need_from_plan
    from gpurunner.supervise.plan import CasePlan, Plan

    case = CasePlan(case="spr-1", pages_url="https://r2/x.tar", n_pages=100,
                    out_dir="E:/prostir/reports/htr/spr-1")
    engine = Plan(assets_url="https://r2/a.tgz", cases=[case],
                  params={"engine_requirements": "kraken==7.1.1"})
    old = Plan(assets_url="https://r2/a.tgz", cases=[case])
    assert need_from_plan(engine, pages=100).image == vast.ENGINE_IMAGE
    assert need_from_plan(old, pages=100).image == vast.DEFAULT_IMAGE
