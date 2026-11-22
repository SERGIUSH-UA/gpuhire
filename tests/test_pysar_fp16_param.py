"""`-p pysar_fp16=` доходить до раннера шарда лише заданим явно.

Раннер Нишпорки сам вмикає fp16 Писаря на картах із тензорними ядрами
(`--pysar-fp16 auto`). Ключ плану — вимикач; без нього прапорця немає, щоб
старий раннер в ассетах не впав на незнайомому аргументі.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpurunner._embedded import htr_case_runner as runner


@pytest.mark.parametrize(("value", "flag"), [
    ("off", ["--pysar-fp16", "off"]),
    ("ON", ["--pysar-fp16", "on"]),
    ("auto", ["--pysar-fp16", "auto"]),
    (None, []),
    ("", []),
    ("half", []),
])
def test_flag_only_when_set(value: str | None, flag: list[str]) -> None:
    params = {} if value is None else {"pysar_fp16": value}
    assert runner._pysar_fp16_flag(params) == flag


def test_both_command_builders_pass_it() -> None:
    text = Path(runner.__file__).read_text(encoding="utf-8")
    assert text.count("base += _pysar_fp16_flag(params)") == 2
