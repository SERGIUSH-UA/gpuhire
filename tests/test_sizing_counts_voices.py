"""Ворота й вибір машини рахують ГОЛОСИ: кожен додатковий голос читає кожен рядок ще раз.

09.10.2026 план на трьох голосах (Писар XVIII + Літописець + Скриба PPv3) обіцяв
A4000 4 300 стор/год, бокс давав 1 150–1 700, і наглядач гасив захід за
бюджетом посеред черги: модель не знала, скільки голосів читає рядок.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpurunner.core.htr_sizing import (
    CARD_GB_PER_SHARD,
    page_cost,
    plan_sizing,
    seg_peak_gb,
    voice_weight,
)
from gpurunner.core.offer_score import Need, measured_pph_for
from gpurunner.supervise.htr import plan_voice_weight

THREE = "diak_cyr_v6.safetensors,skryba_pp_v3.safetensors"


@pytest.mark.parametrize(("voices", "weight"), [
    ("", 1.0),
    (None, 1.0),
    ("diak_cyr_v4.mlmodel", 1.0),                 # kraken уже в сталих
    ("diak_cyr_v6.safetensors", 1.5),             # Літописець — заміряний
    (THREE, 3.0),
    (" diak_cyr_v6.safetensors , skryba_pp_v3.safetensors ", 3.0),
    (["models/skryba_pp_v3.safetensors"], 2.5),
    ("pysar_cyr_v19.pt", 2.0),                    # другий PARSeq
    ("dyvna_model.onnx", 2.5),                    # невідомий рушій — важкий
])
def test_voice_weight(voices, weight):
    assert voice_weight(voices) == pytest.approx(weight)


def test_segmentation_is_paid_once_lines_per_voice():
    one, three = page_cost(6.0, 100.0, 1.0), page_cost(6.0, 100.0, 3.0)
    assert three - one == pytest.approx(200.0)


def _a4000(voice_w: float, **kw) -> float:
    return plan_sizing(cores=9.6, vram_gb=15.6, gb_per_shard=CARD_GB_PER_SHARD,
                       seg_peak=seg_peak_gb(1.37), lines_per_page=83.5, frame_mpx=6.87,
                       gpu_name="RTX A4000", voice_w=voice_w, **kw).pages_per_hour


def test_one_voice_forecast_is_unchanged():
    assert _a4000(1.0) == pytest.approx(plan_sizing(
        cores=9.6, vram_gb=15.6, gb_per_shard=CARD_GB_PER_SHARD, seg_peak=seg_peak_gb(1.37),
        lines_per_page=83.5, frame_mpx=6.87, gpu_name="RTX A4000").pages_per_hour)


def test_three_voices_forecast_matches_live_box():
    """A4000 14016, cdiak_224-spr-49, 09.10.2026: 1 574 стор/год на трьох голосах."""
    one, three = _a4000(1.0), _a4000(voice_weight(THREE))
    assert one > 2 * 1574                     # стара модель — понад удвічі
    assert 0.8 <= 1574 / three <= 1.3         # нова — у межах заміру


def test_target_shrinks_with_voices():
    """Ціль — потужність машини: три голоси на тій самій машині дають менше сторінок."""
    base = dict(pages=1000, max_hours=6, budget_usd=1.0, target_pph=5000.0,
                frame_mpx=6.0, lines_per_page=100.0)
    one, three = Need(**base).target_here, Need(**base, voice_w=3.0).target_here
    assert three == pytest.approx(one * page_cost(6.0, 100.0, 1.0) / page_cost(6.0, 100.0, 3.0))


def _measured(**extra):
    return dict(pages_per_hour=3000.0, cores_quota=16.0, vram_total_gb=16.0,
                pages_per_hour_mpx=6.0, pages_per_hour_lines=100.0, n_gpus=1,
                ts="2026-10-09T10:00:00", **extra)


def _need(voice_w: float) -> Need:
    return Need(pages=1000, max_hours=6, budget_usd=1.0, frame_mpx=6.0,
                lines_per_page=100.0, voice_w=voice_w,
                gb_per_shard=CARD_GB_PER_SHARD, seg_peak=seg_peak_gb(1.37))


def _transfer(measured, voice_w):
    return measured_pph_for(measured, _need(voice_w), cores=16.0, vram_gb=16.0,
                            num_gpus=1, gpu_name="RTX A4000")


def test_measurement_on_one_voice_is_not_promised_to_three():
    pph = _transfer(_measured(), 3.0)
    assert pph == pytest.approx(3000.0 * page_cost(6.0, 100.0, 1.0) / page_cost(6.0, 100.0, 3.0),
                                rel=0.02)


def test_measurement_on_three_voices_carries_over_to_three():
    assert _transfer(_measured(pages_per_hour_voice_w=3.0), 3.0) == pytest.approx(3000.0)


def test_measurement_without_conditions_still_counts_voices():
    bare = dict(pages_per_hour=3000.0, ts="2026-10-09T10:00:00")
    assert _transfer(bare, 3.0) < 1500.0
    assert _transfer(dict(bare, pages_per_hour_voice_w=3.0), 3.0) == pytest.approx(3000.0)


def test_plan_voice_weight_reads_params():
    assert plan_voice_weight(SimpleNamespace(params={"voices": THREE})) == pytest.approx(3.0)
    assert plan_voice_weight(SimpleNamespace(params={})) == 1.0
    assert plan_voice_weight(SimpleNamespace(params=None)) == 1.0
