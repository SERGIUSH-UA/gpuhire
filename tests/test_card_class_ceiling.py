"""Стеля карти за класом: слабка карта насичується раніше за RTX 3090.

30.09.2026, ф.315: ворота пустили P100 і двокарткову RTX 3060 під ціль ~3600
стор/год, бокси дали 1736–2121 і були погашені як `slow_run`. Модель мала одну
стелю карти на всі класи.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpurunner.core import boxes
from gpurunner.core.boxes import BoxObservation
from gpurunner.core.htr_sizing import (
    CARD_LINES_BY_VRAM,
    CARD_LINES_PER_HOUR,
    CARD_MAX_LINES_PER_HOUR,
    card_class,
    card_max_lines_for,
    gb_per_shard_for,
    plan_sizing,
)


@pytest.mark.parametrize(("name", "cls"), [
    ("RTX 4060 Ti", "rtx 4060 ti"),
    ("NVIDIA GeForce RTX 4060 Ti", "rtx 4060 ti"),
    ("Tesla V100", "v100"),
    ("Tesla V100-SXM2-32GB", "v100"),
    ("Tesla V100-PCIE-16GB", "v100"),
    ("Q RTX 8000", "rtx 8000"),
    ("Quadro RTX 8000", "rtx 8000"),
    ("RTX A4000", "rtx a4000"),
    ("NVIDIA RTX A4000", "rtx a4000"),
    ("RTX 4000Ada", "rtx 4000 ada"),
    ("NVIDIA RTX 4000 Ada Generation", "rtx 4000 ada"),
    ("GTX 1660 S", "gtx 1660 super"),
    ("NVIDIA GeForce GTX 1660 SUPER", "gtx 1660 super"),
    ("RTX 3060 laptop", "rtx 3060 laptop"),
    ("", ""),
])
def test_the_offer_and_the_probe_name_the_same_class(name: str, cls: str) -> None:
    assert card_class(name) == cls


def test_every_class_in_the_table_is_written_the_way_names_normalise() -> None:
    for cls in CARD_LINES_PER_HOUR:
        assert card_class(cls) == cls


def test_without_a_name_the_ceiling_is_the_old_one() -> None:
    assert card_max_lines_for() == CARD_MAX_LINES_PER_HOUR
    assert card_max_lines_for("", 11.0) == CARD_MAX_LINES_PER_HOUR
    kw = {"cores": 64.0, "vram_gb": 24.0, "gb_per_shard": 1.2,
          "lines_per_page": 40.0, "frame_mpx": 6.0}
    assert plan_sizing(**kw).pages_per_hour == plan_sizing(**kw, gpu_name="").pages_per_hour


def test_an_unmeasured_card_takes_the_ceiling_of_its_memory() -> None:
    small, mid, big = (lines for _, lines in CARD_LINES_BY_VRAM)
    assert card_max_lines_for("GTX 1070", 8.0) == small
    assert card_max_lines_for("RTX A4500", 20.0) == big
    assert card_max_lines_for("Tesla P100", 16.0) == mid
    assert card_max_lines_for("Tesla P40", 0.0) == big, "пам'ять невідома — останній щабель"
    assert small < mid < big < CARD_MAX_LINES_PER_HOUR


def _on(gpu: str, *, cards: int, vram: float, cores: float, lines: float,
        mpx: float = 7.21) -> float:
    return plan_sizing(cores=cores, vram_gb=vram, num_gpus=cards,
                       gb_per_shard=gb_per_shard_for(mpx), lines_per_page=lines,
                       frame_mpx=mpx, gpu_name=gpu).pages_per_hour


def test_the_boxes_of_30_09_are_forecast_near_what_they_gave() -> None:
    """Факти реєстру боксів: 2×RTX 3060 — 2100 і 3080 стор/год, 2×GTX 1080 Ti на
    тому ж матеріалі ф.315 — 1570. Без класу модель обіцяла вдвічі більше."""
    rtx3060 = _on("RTX 3060", cards=2, vram=23.2, cores=26.9, lines=113.0)
    assert 0.8 <= 2100 / rtx3060 <= 1.25, rtx3060
    blind = _on("", cards=2, vram=23.2, cores=26.9, lines=113.0)
    assert 2100 / blind < 0.6, "без класу — те саме завищення, що було"
    gtx = _on("GTX 1080 Ti", cards=2, vram=22.0, cores=23.0, lines=131.0, mpx=12.59)
    assert 0.8 <= 1570 / gtx <= 1.25, gtx


def test_a_strong_card_is_not_cut_below_what_it_gave() -> None:
    """Q RTX 8000 на 19.2 ядра, spr-43 (39 рядків, 6.84 Мпікс): факт 7258."""
    got = _on("Q RTX 8000", cards=1, vram=47.0, cores=19.2, lines=39.0, mpx=6.84)
    assert got >= 5500, got


def test_more_cards_of_a_class_never_forecast_less() -> None:
    one = _on("RTX 3060", cards=1, vram=11.6, cores=32.0, lines=60.0)
    two = _on("RTX 3060", cards=2, vram=23.2, cores=32.0, lines=60.0)
    assert two >= one


def test_the_probe_name_and_the_offer_name_forecast_the_same() -> None:
    a = _on("Tesla V100", cards=1, vram=31.4, cores=46.0, lines=49.0)
    b = _on("Tesla V100-SXM2-32GB", cards=1, vram=31.4, cores=46.0, lines=49.0)
    assert a == b


# ---- замір машини в реєстрі ------------------------------------------------------


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.setenv("GPURUNNER_BOXES_OVERRIDES", str(tmp_path / "boxes.overrides.json"))
    return tmp_path


def _obs(outcome: str, ts: str, **measured: float) -> BoxObservation:
    return BoxObservation(machine_id=27243, outcome=outcome, ts=ts, measured=dict(measured))


def test_a_later_success_without_a_pace_does_not_erase_the_measurement(registry: Path) -> None:
    from datetime import UTC, datetime, timedelta

    now = datetime.now(tz=UTC)
    boxes.record(_obs("ok", (now - timedelta(days=3)).isoformat(),
                      pages_per_hour=2347, boot_sec=40.0))
    boxes.record(_obs("ok", (now - timedelta(days=1)).isoformat(), boot_sec=260.0))
    got = boxes.verdicts()[27243].best_measured
    assert got["pages_per_hour"] == 2347
    assert got["boot_sec"] == 260.0, "підйом — з останнього успіху"


def test_a_slow_run_pace_reaches_the_selection(registry: Path) -> None:
    from datetime import UTC, datetime, timedelta

    now = datetime.now(tz=UTC)
    boxes.record(_obs("ok", (now - timedelta(days=5)).isoformat(), pages_per_hour=4800))
    boxes.record(_obs("slow_run", (now - timedelta(hours=2)).isoformat(),
                      pages_per_hour=2100, pages_per_hour_lines=113.0))
    got = boxes.verdicts()[27243].best_measured
    assert got["pages_per_hour"] == 2100, "свіжіший власний замір б'є старий успіх"


# ---- розріз за картою в калібровці ---------------------------------------------


def test_by_card_reads_only_a_box_own_pace(tmp_path: Path) -> None:
    from gpurunner.htr.calibrate import card_runs, card_table

    def row(outcome: str, case: str, pace_case: str, lines: float, pph: float) -> dict:
        return {"ts": "2026-09-30T12:00:00+00:00", "outcome": outcome, "case": case,
                "gpu_name": "RTX 3060", "num_gpus": 2,
                "measured": {"pages_per_hour": pph, "pages_per_hour_case": pace_case,
                             "pages_per_hour_mpx": 7.21, "pages_per_hour_lines": lines,
                             "cores_quota": 26.9, "vram_free_gb": 23.2, "n_gpus": 2.0}}

    rows = [
        row("slow_run", "spr-6816", "spr-6816", 113.0, 2100),
        row("slow_for_data", "spr-7021", "spr-7021", 131.0, 1561),   # чужий темп
        row("ok", "spr-1", "spr-0", 80.0, 3000),                     # замір іншої справи
        row("slow_run", "spr-2", "spr-2", 0.0, 1736),                # без густини рядків
        row("ok", "spr-3", "spr-3", 13.0, 900),                      # збій заміру рядків
    ]
    (tmp_path / "boxes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    runs = card_runs(tmp_path, tmp_path / "no-states")
    assert [r.case for r in runs] == ["spr-6816"]
    table = card_table(runs)
    assert table[0]["gpu"] == "rtx 3060" and table[0]["n"] == 1
    assert table[0]["ratio_before"] < 0.6 < table[0]["ratio_after"]


# ---- клас доходить до вибору й до воріт ---------------------------------------


def _offer(gpu: str, cards: int, vram_per_card_gb: float, cores: float) -> dict:
    return {"id": 1, "machine_id": 27243, "gpu_name": gpu, "num_gpus": cards,
            "cpu_cores_effective": cores, "gpu_ram": vram_per_card_gb * 1024,
            "cpu_ram": 64 * 1024, "disk_space": 200.0, "dph_total": 0.1375,
            "reliability2": 0.99, "inet_down": 755.0}


def _f315_need(**kw: float):
    from gpurunner.core.offer_score import Need

    base = dict(pages=2480, max_hours=4.0, budget_usd=0.5, disk_gb=40,
                lines_per_page=113.0, frame_mpx=7.21, gb_per_shard=gb_per_shard_for(7.21))
    base.update(kw)
    return Need(**base)


def test_selection_no_longer_promises_the_target_on_a_weak_pair() -> None:
    """2×RTX 3060 на ф.315: ціль 5000 (тут ~2400) бокс не дає й не береться."""
    from gpurunner.core.offer_score import score_offer

    weak = score_offer(_offer("RTX 3060", 2, 12.0, 28.0), _f315_need(target_pph=5000.0))
    assert not weak.ok
    assert 1700 <= weak.sizing.pages_per_hour <= 2600, weak.sizing.pages_per_hour
    strong = score_offer(_offer("RTX 4060 Ti", 2, 16.0, 32.0), _f315_need(target_pph=5000.0))
    assert strong.ok, strong.explain


def test_the_gate_forecasts_by_the_class_of_the_offer() -> None:
    from gpurunner.supervise.gate import evaluate

    probe = {"cores": 28.0, "cores_all": 28.0, "cores_quota": 26.9, "ram_gb": 31.0,
             "gpu": "NVIDIA GeForce RTX 3060", "n_gpus": 2.0,
             "vram_total_mb": 24576.0, "vram_free_mb": 23771.0, "vram_free_min_mb": 11876.0,
             "disk_free_gb": 190.0, "net_bps": 50_000_000.0, "py": "3.11"}
    offer = _offer("RTX 3060", 2, 12.0, 28.0)
    res = evaluate(probe, offer, _f315_need())
    assert res.ok and res.sizing is not None
    assert 1700 <= res.sizing.pages_per_hour <= 2600, res.sizing.pages_per_hour
    # Оффер без назви — клас із проби бокса, а не стеля «класу 3090».
    nameless = evaluate(probe, {**offer, "gpu_name": ""}, _f315_need())
    assert nameless.sizing is not None
    assert nameless.sizing.pages_per_hour == res.sizing.pages_per_hour


def test_a_registry_pace_is_rescaled_within_the_class() -> None:
    """Замір 2×RTX 3060 на 27 ядрах, оффер тієї ж машини на 14: стеля карти та
    сама, тож замір лишається своїм, а не росте за ядрами."""
    from gpurunner.core.offer_score import measured_pph_for

    measured = {"pages_per_hour": 2100, "cores_quota": 26.9, "vram_total_gb": 24.0,
                "pages_per_hour_mpx": 7.21, "pages_per_hour_lines": 113.0, "n_gpus": 2.0}
    same = measured_pph_for(measured, _f315_need(), cores=26.9, vram_gb=24.0,
                            num_gpus=2, gpu_name="RTX 3060")
    assert same == pytest.approx(2100, rel=0.02)
    more_cores = measured_pph_for(measured, _f315_need(), cores=54.0, vram_gb=24.0,
                                  num_gpus=2, gpu_name="RTX 3060")
    assert more_cores == pytest.approx(2100, rel=0.02), "ядра понад стелю карти темпу не дають"
