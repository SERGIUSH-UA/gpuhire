"""Відчеплений запуск для справи з кирилицею в назві.

Наші справи звуться `spr-47а` і `spr-84г`. Доти пускач WSH писався в ASCII,
і такий захід падав `UnicodeEncodeError` ще до планувальника — тобто оренди
не було, роботи не було, а на диску лишався порожній `.vbs`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpurunner.supervise import detach


@pytest.mark.skipif(os.name != "nt", reason="schtasks і WSH лише на Windows")
@pytest.mark.parametrize("session", ["htr-spr-47а-0921-2200", "htr-spr-84г-q2-0921-0048"])
def test_hidden_task_survives_cyrillic_case_names(tmp_path: Path, monkeypatch, session: str) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(detach.subprocess, "run",
                        lambda argv, **kw: calls.append(list(argv)))
    script = tmp_path / f"{session}.cmd"

    detach.schedule_hidden(session, script, ['echo "тест"'])

    launcher = script.with_suffix(".vbs")
    assert launcher.is_file() and launcher.stat().st_size > 0
    # Пускач мусить нести шлях цілим, а не в теоретичному ASCII.
    assert str(script) in launcher.read_text(encoding="utf-16")
    assert calls, "задача планувальника не створювалась"
