"""Підпис «що тисне» у таблиці ринку покриває всі значення `limited_by`.

До 06.09.2026 мапа в `cli.py` знала ключ `max_shards`, якого `plan_sizing`
ніколи не повертає, — і стеля шардів друкувалась сирим `cap`.
"""
from __future__ import annotations

from gpurunner.cli import _SQUEEZE
from gpurunner.core.htr_sizing import plan_sizing


def test_every_limited_by_value_has_a_label() -> None:
    seen = {
        plan_sizing(cores=4, vram_gb=32.0).limited_by,          # cpu
        plan_sizing(cores=64, vram_gb=8.0).limited_by,          # vram
        plan_sizing(cores=64, vram_gb=64.0, gb_per_shard=1.0, max_shards=4).limited_by,  # cap
    }
    assert seen == {"cpu", "vram", "cap"}
    for value in seen:
        assert value in _SQUEEZE, f"limited_by={value!r} друкувався б сирим"
