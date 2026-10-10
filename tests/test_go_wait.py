"""Очікування даних на боксі — справжнім bash, із секундами замість годин.

🔴 9.10.2026 раннер чекав `GO` рівно пів години, а доставка 25 ГБ тривала 77
хвилин: на 30-й він вийшов з `inputs never arrived`, і дані доїхали в
порожнечу. Тепер бокс чекає до строку, який назвав наглядач (`GO_UNTIL`).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import gpurunner.backends.vast as vast


def _posix_bash() -> str | None:
    """bash із POSIX-утилітами. На Windows — той, що з Git: `bash` у PATH там
    часто веде у WSL, де шляхи й оточення інші."""
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    if not git:
        return None
    for up in Path(git).resolve().parents[:3]:
        cand = up / "bin" / "bash.exe"
        if cand.exists():
            return str(cand)
    return None


BASH = _posix_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="немає POSIX bash")


def _sh(p: Path) -> str:
    s = p.resolve().as_posix()
    if os.name == "nt" and len(s) > 1 and s[1] == ":":
        return f"/{s[0].lower()}{s[2:]}"
    return s


def _start(tmp_path: Path, floor: int) -> tuple[subprocess.Popen[bytes], Path]:
    root = tmp_path / "box"
    root.mkdir()
    script = vast._render_go_wait(root=_sh(root), floor_seconds=floor, poll_seconds=1)
    proc = subprocess.Popen([BASH, "-c", script + "echo started > " + _sh(root / "ran") + "\n"],  # type: ignore[list-item]
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, root


def test_without_a_promise_the_floor_holds_and_failure_is_written(tmp_path: Path) -> None:
    proc, root = _start(tmp_path, floor=1)

    assert proc.wait(timeout=20) == 1
    status = json.loads((root / "_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "inputs never arrived" in status["error"]
    assert not (root / "ran").exists()


def test_a_promise_keeps_the_box_waiting_past_the_floor(tmp_path: Path) -> None:
    """🔴 Саме той випадок: підлога вичерпалась, а доставка ще йде."""
    proc, root = _start(tmp_path, floor=1)
    (root / "GO_UNTIL").write_text(str(int(time.time()) + 30), encoding="utf-8")
    time.sleep(4)                          # підлога вже минула
    assert proc.poll() is None, "бокс здався, хоч наглядач пообіцяв дані"
    (root / "GO").write_text("", encoding="utf-8")

    assert proc.wait(timeout=20) == 0
    assert (root / "ran").exists()
    assert not (root / "_status.json").exists()


def test_a_stale_or_garbled_promise_does_not_extend_the_wait(tmp_path: Path) -> None:
    proc, root = _start(tmp_path, floor=1)
    (root / "GO_UNTIL").write_text("сміття", encoding="utf-8")

    assert proc.wait(timeout=20) == 1


def test_the_onstart_waits_by_the_promise() -> None:
    script = vast.VastBackend()._render_onstart(max_hours=1, autodestroy_hours=0.5)
    assert "GO_UNTIL" in script
    assert "seq 1 900" not in script, "стала пів години повернулась"
