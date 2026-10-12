"""Прапорці `gpurunner htr plan`, які обіцяє контракт, мусять існувати.

`--key-not-in-library` стояв у `docs/contract.md` і його слала Нишпорка на
теку без шифри, а CLI його не знав: план не складався зовсім.
"""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from gpurunner.cli import app

ROOT = Path(__file__).resolve().parents[1]


def test_every_plan_flag_of_the_contract_is_known() -> None:
    contract = (ROOT / "docs" / "contract.md").read_text(encoding="utf-8")
    block = contract[contract.index("gpurunner htr plan"):contract.index("gpurunner htr preflight")]
    flags = set(re.findall(r"--[a-z][a-z0-9-]+", block))
    out = CliRunner().invoke(app, ["htr", "plan", "--help"], terminal_width=200).output
    missing = sorted(f for f in flags if f not in out)
    assert not missing, f"контракт обіцяє, а CLI не знає: {missing}"
