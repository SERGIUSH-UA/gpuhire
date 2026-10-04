"""Батч kraken-голосу доходить до шарда на боксі.

Замір 05.09.2026 на RTX 3090 (std160, 160 стор. по 70 рядків): `--voice-batch 32`
дає +10% темпу флоту і на 8, і на 12 шардах (4600 → 5084, 5040 → 5573). Але
ручка жила лише в самому раннері: план її не знав, job не нормалізував, і в
base-команді шарда її не було — тобто в хмарі батч був недосяжний.
"""
from __future__ import annotations

import re
from pathlib import Path

from gpurunner.jobs import get_job

RUNNER_SRC = (Path(__file__).resolve().parents[1] / "src" / "gpurunner"
              / "_embedded" / "htr_case_runner.py").read_text(encoding="utf-8")


def test_job_normalises_voice_batch_and_defaults_to_the_runner() -> None:
    job = get_job("htr_case")()
    assert job.validate_params({"dataset": "o/s", "voice_batch": 16})["voice_batch"] == 16
    assert job.validate_params({"dataset": "o/s"})["voice_batch"] == 0, (
        "без ручки дефолт вирішує раннер, а не gpurunner")


def test_shard_base_command_carries_the_flag_only_above_one() -> None:
    """0/1 не сміють просочуватись у команду: `--voice-batch 1` виглядав би як
    свідомий вибір, а не як відсутність вибору."""
    m = re.search(r"if voice_batch > 1:\s*\n\s*base \+= \[\"--voice-batch\", str\(voice_batch\)\]",
                  RUNNER_SRC)
    assert m, "у base-команді шарда немає --voice-batch за умовою > 1"
    assert 'voice_batch = int(params.get("voice_batch") or 0)' in RUNNER_SRC
