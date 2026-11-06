"""Калібровка знає, скількома голосами читали захід.

22.10.2026 ковзний сторож дрейфу показав медіану 1.32: свіжі заходи читались
трьома голосами, а рядок калібровки міряв їх проти прогнозу для одного.
"""
from __future__ import annotations

import json
from pathlib import Path

from gpurunner.htr import calibrate as C


def _state(tmp: Path, *, box: dict, log_head: str = "") -> dict:
    log = tmp / "s.log"
    log.write_text(log_head, encoding="utf-8")
    return {"log_path": str(log), "box": box}


def test_the_state_says_it_first(tmp_path: Path) -> None:
    assert C._voice_w_of(_state(tmp_path, box={"voice_w": 2.85}), {"voice_w": 2.85}) == 2.85


def test_an_old_state_reads_the_plan_named_in_its_log(tmp_path: Path) -> None:
    plan = tmp_path / "plan dir" / "plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"params": {
        "voices": "diak_cyr_v10.safetensors,skryba_pp_v3.safetensors"}}), encoding="utf-8")
    head = (f'[blackbox]    запуск: gpurunner htr supervise --plan "{plan}" '
            f"--session s --budget 0.5\n")
    assert C._voice_w_of(_state(tmp_path, box={}, log_head=head), {}) == 2.85


def test_without_a_plan_the_run_stays_single_voice(tmp_path: Path) -> None:
    head = f"--plan {tmp_path / 'gone.json'} --session s\n"
    assert C._voice_w_of(_state(tmp_path, box={}, log_head=head), {}) == 1.0
    assert C._voice_w_of({"box": {}}, {}) == 1.0
