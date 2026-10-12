"""Кожен ключ, який читає бокс-раннер, мусить проходити крізь опис job'а.

`HTRCaseJob.validate_params` збирає словник для боксу з переліку ключів, і
ключ поза переліком губиться мовчки. Так `pysar_fp16`, `seg_max_mpx`,
`pipeline` і `engine_patches` не діяли ніколи: пара заміру 01.10.2026 «з
патчами / без» вийшла двома однаковими заходами.
"""

from __future__ import annotations

import re
from pathlib import Path

from gpurunner._embedded import htr_case_runner as runner
from gpurunner.jobs.htr_case import HTRCaseJob

#: Ключі, які бокс читає не з параметрів job'а: внутрішні (`_…`) ставить сам
#: раннер, а ці живуть у записі справи черги (`cases[]`).
_PER_CASE = {"frame_mpx_p95"}


def _box_reads() -> set[str]:
    text = Path(runner.__file__).read_text(encoding="utf-8")
    keys = set(re.findall(r"params(?:\.get\(|\[)\s*[\"']([a-z][a-z_0-9]*)[\"']", text))
    return keys - _PER_CASE


def test_every_key_the_box_reads_survives_validation() -> None:
    out = HTRCaseJob().validate_params({"pages_dataset": "u/pages", "models_dataset": "u/models",
                                        "scripts_dataset": "u/scripts"})
    lost = sorted(_box_reads() - set(out))
    assert not lost, f"бокс читає, а опис job'а не пропускає: {lost}"


def test_the_knobs_arrive_as_given() -> None:
    out = HTRCaseJob().validate_params({
        "pages_dataset": "u/pages", "models_dataset": "u/models", "scripts_dataset": "u/scripts",
        "pysar_fp16": "off", "seg_max_mpx": "6", "engine_patches": "off", "pipeline": "false"})
    assert out["pysar_fp16"] == "off" and out["engine_patches"] == "off"
    assert float(out["seg_max_mpx"]) == 6.0 and out["pipeline"] is False
