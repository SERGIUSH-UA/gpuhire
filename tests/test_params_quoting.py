"""PARAMS мусить переживати будь-яке значення, яке користувач може ввести.

Інжект робився як ``json.loads(r'''<json>''')`` — конструкція, що стає
SyntaxError, щойно в значенні трапиться ``'''``. Дістатись до неї можна було
звичайним `-p`: `charset_extra` (parseq) і `spec` (kraken) — вільний текст.
Падало це локально, при рендері, але з повідомленням про рядок 2 згенерованого
модуля, за яким походження проблеми не вгадується.
"""
from __future__ import annotations

import ast
import json

import pytest

from gpurunner.core.job import Job

NASTY = [
    "'''",                      # той самий термінатор
    'a\'\'\'b',
    '"""',
    '\\',                       # зворотний слеш
    'x\\',                      # слеш у кінці — ламає raw-рядки
    '"quoted"',
    "line\nbreak",
    "\t\r",
    "кирилиця й emoji 🔴",
    "'; import os; os.system('rm -rf /')  #",   # ін'єкція коду
]


@pytest.mark.parametrize("value", NASTY)
def test_params_block_survives_any_value(value):
    block = Job._params_block({"charset_extra": value, "n": 1})
    tree = ast.parse(block)          # мусить бути валідним Python
    assign = tree.body[0]
    assert isinstance(assign, ast.Assign)
    payload = ast.literal_eval(assign.value.args[0])  # type: ignore[attr-defined]
    assert json.loads(payload)["charset_extra"] == value


@pytest.mark.parametrize("value", NASTY)
def test_rendered_job_survives_any_value(value):
    """Наскрізно: job → рендер → парсинг → те саме значення назад."""
    from gpurunner.jobs.parseq_train import ParseqTrainJob

    code = ParseqTrainJob().render_remote_code(
        {"dataset": "owner/slug", "charset_extra": value}
    )
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets
        ):
            payload = ast.literal_eval(node.value.args[0])  # type: ignore[attr-defined]
            assert json.loads(payload)["charset_extra"] == value
            return
    raise AssertionError("PARAMS не знайдено у згенерованому коді")


def test_kraken_spec_with_quotes_does_not_break_render():
    """`spec` — друге вільне поле, через яке баг був досяжний."""
    from gpurunner.jobs.kraken_train import KrakenTrainJob

    code = KrakenTrainJob().render_remote_code(
        {"dataset": "owner/arrows", "spec": "[1,120 '''] # \\"}
    )
    ast.parse(code)
