"""`gpurunner bg`: одна копія, стан із PID-ами, зупинка за PID (10.09.2026).

Черга FS того дня стартувала двічі (тригер «+1 хв» поруч із `/run`), а
зупиняючи стару, агент вбив власну оболонку фільтром за командним рядком.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gpurunner.core import locks
from gpurunner.core.locks import LockBusy
from gpurunner.supervise import bg


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))


def test_run_records_pids_and_exit_code(tmp_path: Path) -> None:
    rc = bg.run("fs-11900", [sys.executable, "-c", "raise SystemExit(7)"], cwd=tmp_path)
    assert rc == 7
    st = bg.load("fs-11900")
    assert st and st["rc"] == 7 and st["pid_runner"] == os.getpid() and st["pid_child"]
    assert st["finished"] and not bg.is_running(st)
    assert locks.read("bg:fs-11900") is None, "замок знято після кінця"


def test_second_copy_from_another_live_process_is_refused(tmp_path: Path) -> None:
    """🔴 Та сама задача з іншого живого pid — подвійний старт."""
    path = locks._path("bg:fs-11900")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"resource": "bg:fs-11900", "owner": "bg-fs-11900",
                                "session": "fs-11900", "pid": os.getppid(),
                                "ts": time.time(), "ttl_sec": 3600}), encoding="utf-8")
    with pytest.raises(LockBusy):
        bg.run("fs-11900", [sys.executable, "-c", "pass"], cwd=tmp_path)


def test_start_refuses_while_the_same_task_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    bg._save("fs-11900", {"name": "fs-11900", "pid_runner": os.getppid(), "finished": None})
    with pytest.raises(bg.BgBusy):
        bg.start("fs-11900", ["python", "x.py"])


def test_stop_kills_by_pid_from_state_never_by_command_text(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class _Done:
        returncode = 0

    monkeypatch.setattr(bg.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _Done())
    monkeypatch.setattr(bg.locks, "_alive", lambda pid: True)
    bg._save("fs-11900", {"name": "fs-11900", "pid_runner": 4242, "finished": None,
                          "cmd": ["pwsh", "-File", "fs_chain_1875.ps1"]})
    done = bg.stop("fs-11900")
    kills = [c for c in calls if c and c[0] in ("taskkill", "kill")]
    if os.name == "nt":
        assert kills == [["taskkill", "/T", "/F", "/PID", "4242"]]
        assert "4242" in done[0]
    assert not any("fs_chain" in " ".join(c) for c in calls)
    assert bg.load("fs-11900")["finished"]


@pytest.mark.skipif(os.name != "nt", reason="schtasks лише на Windows")
def test_start_writes_a_hidden_self_deleting_task(monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:
    calls: list[list[str]] = []

    class _Done:
        returncode = 0
        stdout = stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _Done())
    how = bg.start("fs-11900", ["pwsh", "-File", "dl one.ps1"], cwd=tmp_path)
    assert "gpurunner-bg-fs-11900" in how
    body = [ln for ln in (bg.bg_dir() / "fs-11900.cmd").read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    assert body[1].startswith("schtasks /delete /f /tn")
    assert "bg _run fs-11900" in body[-1] and '"dl one.ps1"' in body[-1]
    launcher = (bg.bg_dir() / "fs-11900.vbs").read_text(encoding="utf-16")
    assert launcher.rstrip().endswith(", 0, False")
    assert calls[0][:2] == ["schtasks", "/create"] and calls[1][:2] == ["schtasks", "/run"]
