"""Наглядач мусить пережити сесію, яка його запустила, і прибрати за собою.

🔴 Виміряно 19.08.2026 тричі поспіль: `Start-Process -WindowStyle Hidden`,
`cmd /c` через WMI і `nohup … &` — усі три гинуть разом із зачисткою сесії
харнеса. Пережив лише планувальник завдань. А забута задача `/sc once` одного
разу підняла НОВУ оренду о 23:59 без нагляду, тож прибирання за собою тут така
сама умова роботи, як і саме відчеплення.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gpurunner.supervise import detach


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))


def test_task_name_is_per_session_not_per_machine() -> None:
    """Дві кампанії на машині — дві задачі. Спільне ім'я означало б, що друга
    затирає першу, а перша лишається без нагляду."""
    assert detach.task_name("htr-a") != detach.task_name("htr-b")
    assert detach.task_name("htr-a").startswith(detach.TASK_PREFIX)


@pytest.mark.skipif(os.name != "nt", reason="schtasks лише на Windows")
def test_spawn_creates_a_one_shot_task_and_runs_it(monkeypatch) -> None:
    """🔴 `/sc once`, а не `/sc daily`: задача мусить бути одноразовою навіть
    тоді, коли її не встигли прибрати."""
    calls: list[list[str]] = []

    class _Done:
        returncode = 0
        stdout = stderr = ""

    monkeypatch.setattr(detach.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or _Done())
    monkeypatch.setattr(detach.shutil, "which", lambda _n: "C:/bin/gpurunner.exe")

    how = detach.spawn(["htr", "supervise", "--plan", "p.json"],
                       session="htr-x", owner="agent-1")

    assert "htr-x" in how
    create = calls[0]
    assert "/sc" in create and create[create.index("/sc") + 1] == "once"
    assert calls[1][:2] == ["schtasks", "/run"]

    # 🔴🔴 У `/tr` їде ОДИН ШЛЯХ, а не команда з лапками. Спроба зібрати
    # `cmd /c "…"` провалилась на першому ж бойовому запуску: у задачі опинилось
    # `\"…gpurunner.EXE\"` — так екранує C-рантайм, а `cmd` бачить саме
    # скісну. Задача падала за секунду з кодом 1, і причини не було ніде.
    target = create[create.index("/tr") + 1]
    assert target.endswith(".vbs"), "у задачі — пускач, який ховає вікно"
    assert '\\"' not in target

    # 🔴 Вікна бути не повинно: задача виконується в інтерактивній сесії, і
    # `cmd` тримав би чорну консоль відкритою на весь захід, тобто години.
    # Штатний `/ru … /np` тут заборонено політикою (перевірено 04.09.2026),
    # тому вікно ховає стиль 0 у пускачі WSH.
    launcher = detach._launcher_path("htr-x").read_text(encoding="ascii")
    assert launcher.rstrip().endswith(", 0, False")
    assert "htr-x.cmd" in launcher

    body = detach._script_path("htr-x").read_text(encoding="utf-8")
    # 🔴🔴 Задача прибирає СЕБЕ першим ділом: тригер «+1 хв» поруч із `/run`
    # інакше піднімав другу копію (черга FS, 10.09.2026).
    first_cmd = [ln for ln in body.splitlines() if ln.strip()][1]
    assert first_cmd.startswith("schtasks /delete /f /tn")
    assert detach.task_name("htr-x") in first_cmd
    assert "set GPURUNNER_OWNER=agent-1" in body
    assert "htr supervise" in body
    # Вивід МУСИТЬ кудись іти: усе, що падає до власного лога наглядача,
    # інакше зникає безслідно.
    assert "htr-x.spawn.log" in body and "2>&1" in body


def _no_path(monkeypatch) -> None:
    monkeypatch.setattr(detach.shutil, "which", lambda _n: None)


def test_self_argv_takes_the_running_gpurunner_when_path_is_empty(monkeypatch,
                                                                  tmp_path: Path) -> None:
    """🔴 10.09.2026: без gpurunner у PATH задача кликала `python.exe htr supervise`."""
    _no_path(monkeypatch)
    exe = tmp_path / ("gpurunner.exe" if os.name == "nt" else "gpurunner")
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(detach.sys, "argv", [str(exe), "htr", "supervise"])
    assert detach.self_argv() == [str(exe.resolve())]


def test_self_argv_finds_the_venv_script_even_with_a_base_interpreter(
        monkeypatch, tmp_path: Path) -> None:
    """`sys.executable` буває базовим Python312 — `sys.prefix` усе одно venv."""
    _no_path(monkeypatch)
    scripts = tmp_path / "venv" / ("Scripts" if os.name == "nt" else "bin")
    scripts.mkdir(parents=True)
    exe = scripts / ("gpurunner.exe" if os.name == "nt" else "gpurunner")
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(detach.sys, "argv", ["python.exe"])
    monkeypatch.setattr(detach.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(detach.sys, "executable", str(tmp_path / "Python312" / "python.exe"))
    assert detach.self_argv() == [str(exe)]


def test_self_argv_last_resort_is_a_runnable_module(monkeypatch, tmp_path: Path) -> None:
    import importlib.util

    _no_path(monkeypatch)
    monkeypatch.setattr(detach.sys, "argv", ["python.exe"])
    monkeypatch.setattr(detach.sys, "prefix", str(tmp_path / "nothing"))
    monkeypatch.setattr(detach.sys, "executable", str(tmp_path / "py" / "python.exe"))
    argv = detach.self_argv()
    assert argv[1:] == ["-m", "gpurunner"]
    assert importlib.util.find_spec("gpurunner.__main__") is not None


def _plan_file(tmp_path: Path) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "assets_url": "https://r2/assets.tgz", "budget_usd": 1.0, "max_hours": 4.0,
        "cases": [{"case": "spr-1", "pages_url": "https://r2/x.tar", "n_pages": 3,
                   "out_dir": str(tmp_path)}]}), encoding="utf-8")
    return path


@pytest.mark.skipif(os.name != "nt", reason="шлях --detach через планувальник")
def test_detach_refuses_to_report_a_supervisor_that_never_started(monkeypatch,
                                                                  tmp_path: Path) -> None:
    """🔴 «Пішов у фон» без стану — саме та брехня, що коштувала запуску 10.09."""
    from typer.testing import CliRunner

    from gpurunner import cli

    monkeypatch.setattr(cli, "DETACH_START_WAIT_SEC", 0.2)
    monkeypatch.setattr(cli, "DETACH_POLL_SEC", 0.05)
    monkeypatch.setattr(detach, "spawn", lambda argv, **kw: "задача (тест)")
    monkeypatch.setattr(detach, "cleanup", lambda session: True)
    res = CliRunner().invoke(cli.app, ["htr", "supervise", "--plan", str(_plan_file(tmp_path)),
                                       "--detach", "--session", "htr-dead"])
    assert res.exit_code == 3
    assert "НЕ стартував" in res.output


@pytest.mark.skipif(os.name != "nt", reason="шлях --detach через планувальник")
def test_detach_reports_success_once_the_state_appears(monkeypatch, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from gpurunner import cli
    from gpurunner.supervise.state import state_path

    def spawn(argv, **kw):
        state_path("htr-alive").parent.mkdir(parents=True, exist_ok=True)
        state_path("htr-alive").write_text("{}", encoding="utf-8")
        return "задача (тест)"

    monkeypatch.setattr(cli, "DETACH_START_WAIT_SEC", 1.0)
    monkeypatch.setattr(cli, "DETACH_POLL_SEC", 0.05)
    monkeypatch.setattr(detach, "spawn", spawn)
    res = CliRunner().invoke(cli.app, ["htr", "supervise", "--plan", str(_plan_file(tmp_path)),
                                       "--detach", "--session", "htr-alive"])
    assert res.exit_code == 0, res.output
    assert "пішов у фон" in res.output


@pytest.mark.skipif(os.name != "nt", reason="schtasks лише на Windows")
def test_cleanup_is_silent_when_there_is_no_task(monkeypatch) -> None:
    """🔴 Наглядача могли запустити й руками. Падати на прибиранні того, чого
    не заводили, — це втратити вердикт уже завершеного заходу."""
    class _Missing:
        returncode = 1
        stdout = stderr = "ERROR: The system cannot find the file specified."

    monkeypatch.setattr(detach.subprocess, "run", lambda cmd, **kw: _Missing())
    assert detach.cleanup("htr-none") is False


def test_release_locks_frees_only_this_session(tmp_path: Path) -> None:
    """Чужий замок не чіпаємо — те саме правило, що з інстансами."""
    from gpurunner.core import locks

    directory = locks.locks_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for session, resource in (("mine", "case:spr-1"), ("alien", "case:spr-2")):
        payload = {"resource": resource, "owner": "me", "session": session, "pid": 1}
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in resource)[:80]
        (directory / f"{safe}.lock").write_text(json.dumps(payload), encoding="utf-8")

    freed = detach.release_locks("mine", owner="me")

    assert freed == ["case:spr-1"]
    assert locks.read("case:spr-2") is not None


def test_lock_resource_comes_from_the_file_not_from_its_name(tmp_path: Path) -> None:
    """🔴 Ім'я файла санітизоване: `market:rent` і `market_rent` дають однакове
    ім'я, тож відновити ресурс з нього не можна."""
    from gpurunner.core import locks

    directory = locks.locks_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "market_rent.lock").write_text(
        json.dumps({"resource": "market:rent", "owner": "me", "session": "s", "pid": 1}),
        encoding="utf-8")

    assert detach.release_locks("s", owner="me") == ["market:rent"]


def test_posix_detach_does_not_need_the_setsid_binary(monkeypatch) -> None:
    """🔴 `_spawn_posix` відчіплює системним викликом (`start_new_session`), а
    умова вимагала БІНАРНИКА `setsid` — якого немає на macOS. `--detach` там
    відхилявся, хоча все потрібне було на місці."""
    monkeypatch.setattr(detach.os, "name", "posix")
    monkeypatch.setattr(detach.shutil, "which", lambda _n: None)
    assert detach.supported() is True


def test_posix_spawn_writes_startup_output_to_the_spawn_log(monkeypatch) -> None:
    """Усе, що падає ДО власного лога наглядача (битий план, немає ключа), на
    POSIX ішло в DEVNULL — і `spawn_log()` завжди був порожній."""
    seen: dict = {}

    class _Popen:
        def __init__(self, argv, **kw) -> None:
            seen.update(kw, argv=argv)
            kw["stdout"].write(b"boom: no plan\n")

    monkeypatch.setattr(detach.subprocess, "Popen", _Popen)
    monkeypatch.setattr(detach.shutil, "which", lambda _n: "/usr/bin/gpurunner")

    how = detach._spawn_posix(["htr", "supervise", "--plan", "p.json"],
                              session="htr-px", owner="agent-1")

    assert "відчеплений" in how
    assert seen["argv"] == ["/usr/bin/gpurunner", "htr", "supervise", "--plan", "p.json"]
    assert seen["start_new_session"] is True
    assert seen["stderr"] == detach.subprocess.STDOUT
    assert seen["stdout"] is not detach.subprocess.DEVNULL
    assert seen["env"]["GPURUNNER_OWNER"] == "agent-1"
    assert detach.spawn_log("htr-px") == "boom: no plan"


def test_kill_removes_the_task_even_without_a_pid(monkeypatch) -> None:
    """Задача мусить зникнути й тоді, коли процес уже помер сам: інакше вона
    підніме нову оренду за розкладом."""
    monkeypatch.setattr(detach, "cleanup", lambda _s: True)
    done = detach.kill("htr-x", pid=0)
    assert any("задач" in line for line in done)
