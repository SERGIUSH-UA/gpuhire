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
import types
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
    xml: list[str] = []

    class _Done:
        returncode = 0
        stdout = stderr = ""

    def run(cmd, **kw):
        calls.append(cmd)
        if "/xml" in cmd:   # файл визначення прибирається одразу після реєстрації
            xml.append(Path(cmd[cmd.index("/xml") + 1]).read_text(encoding="utf-16"))
        return _Done()

    monkeypatch.setattr(detach.subprocess, "run", run)
    monkeypatch.setattr(detach.shutil, "which", lambda _n: "C:/bin/gpurunner.exe")

    how = detach.spawn(["htr", "supervise", "--plan", "p.json"],
                       session="htr-x", owner="agent-1")

    assert "htr-x" in how
    assert calls[0][:2] == ["schtasks", "/create"]
    assert calls[1][:2] == ["schtasks", "/run"]
    # 🔴 Одноразова: один `TimeTrigger`, без повтору — навіть тоді, коли
    # задачу не встигли прибрати.
    assert xml[0].count("<TimeTrigger>") == 1 and "<Repetition>" not in xml[0]
    # 🔴🔴 Без умов живлення. `/sc once` давав «лише від мережі», і на ноутбуці
    # від батареї `/run` казав SUCCESS, не запускаючи нічого (10.10.2026).
    assert "<DisallowStartIfOnBatteries>false<" in xml[0]
    assert "<StopIfGoingOnBatteries>false<" in xml[0]

    # 🔴🔴 Задача кличе пускач ОДНИМ ШЛЯХОМ у лапках, а не команду з лапками.
    # Спроба зібрати `cmd /c "…"` провалилась на першому ж бойовому запуску:
    # у задачі опинилось `\"…gpurunner.EXE\"` — так екранує C-рантайм, а `cmd`
    # бачить саме скісну. Задача падала за секунду з кодом 1.
    assert "wscript.exe</Command>" in xml[0]
    assert f'"{detach._launcher_path("htr-x")}"</Arguments>' in xml[0]
    assert '\\"' not in xml[0]
    assert not detach._script_path("htr-x").with_suffix(".xml").exists()

    # 🔴 Вікна бути не повинно: задача виконується в інтерактивній сесії, і
    # `cmd` тримав би чорну консоль відкритою на весь захід, тобто години.
    # Штатний `/ru … /np` тут заборонено політикою (перевірено 04.09.2026),
    # тому вікно ховає стиль 0 у пускачі WSH.
    launcher = detach._launcher_path("htr-x").read_text(encoding="utf-16")
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


@pytest.mark.skipif(os.name != "nt", reason="батник cmd лише на Windows")
def test_batch_switches_cmd_to_utf8_before_any_path(monkeypatch, tmp_path: Path) -> None:
    """🔴🔴 Файл у UTF-8, а `cmd` читає батник сторінкою консолі (OEM 866):
    `Documents\\Нишпорка\\…\\plan.json` доїжджав кашею `╨Э╨╕╤И…` — «[Errno 2]»
    (звіт 10.10.2026; відтворено на консолі `chcp 866`)."""
    monkeypatch.setattr(detach.subprocess, "run", lambda cmd, **kw: None)
    script = tmp_path / "Нишпорка" / "s.cmd"
    script.parent.mkdir()
    detach.schedule_hidden("t", script, [f'"{script.parent}\\plan.json"'])
    raw = script.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "BOM зламав би перший рядок cmd"
    lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    assert "chcp 65001" in lines[0]
    assert all(ln.isascii() for ln in lines[:2]), "до chcp — лише ASCII"
    # Вивід Python у файл без цього пишеться ANSI-сторінкою: `💵` → UnicodeEncodeError.
    assert 'set "PYTHONIOENCODING=utf-8"' in raw.decode("utf-8")


@pytest.mark.skipif(os.name != "nt", reason="батник cmd лише на Windows")
def test_detached_task_gets_the_callers_gpurunner_env(monkeypatch) -> None:
    """Задача стартує з оточенням ВХОДУ, а не викликача: Нишпорчин
    `GPURUNNER_REPO_DATA_DIR` губився, і наглядач брав порожній реєстр боксів."""
    monkeypatch.setattr(detach.subprocess, "run", lambda cmd, **kw: None)
    monkeypatch.setattr(detach.shutil, "which", lambda _n: "C:/bin/gpurunner.exe")
    monkeypatch.setenv("GPURUNNER_REPO_DATA_DIR", r"C:\Простір\data\gpurunner")
    monkeypatch.setenv("GPURUNNER_PCT", "100%")
    monkeypatch.setenv("GPURUNNER_OWNER", "з-оточення")
    detach.spawn(["htr", "supervise"], session="htr-env", owner="agent-1")
    body = detach._script_path("htr-env").read_text(encoding="utf-8")
    assert r'set "GPURUNNER_REPO_DATA_DIR=C:\Простір\data\gpurunner"' in body
    assert 'set "GPURUNNER_PCT=100%%"' in body
    assert "з-оточення" not in body, "власника дає викликач, а не оточення"


@pytest.mark.skipif(os.name != "nt", reason="schtasks лише на Windows")
@pytest.mark.parametrize(("rc", "stdout", "expect"), [
    (0, '"PC","\\gpurunner-htr-x","N/A","Готово","Interactive","30.11.1999 0:00:00",'
        '"267011","me","wscript.exe"\r\n', "так і не запустив"),
    (0, '"PC","\\gpurunner-htr-x","N/A","Готово","Interactive","10.10.2026 1:00:00",'
        '"-2147020576","me","wscript.exe"\r\n', "відмовив у запуску"),
    (1, "", "вже немає"),
])
def test_task_report_names_why_the_task_did_not_start(monkeypatch, rc, stdout,
                                                      expect) -> None:
    """🔴 «НЕ стартував» без причини коштував користувачці годин: вона шукала
    задачу в Планувальнику й журналі подій, а її вже прибрали. Заголовки й
    статуси локалізовані — читаємо позицію й число."""
    class _Done:
        returncode = rc
        stderr = ""

    _Done.stdout = stdout
    monkeypatch.setattr(detach.subprocess, "run", lambda cmd, **kw: _Done())
    assert expect in detach.task_report("gpurunner-htr-x")


@pytest.mark.skipif(os.name != "nt", reason="шлях --detach через планувальник")
def test_detach_with_dry_run_estimates_here_instead_of_waiting(monkeypatch,
                                                               tmp_path: Path) -> None:
    """🔴 Кошторис стану не пише — відчеплений `--dry-run` завжди «НЕ стартував»
    за 120 с і видавався за відтворення справжнього збою (10.10.2026)."""
    from typer.testing import CliRunner

    from gpurunner import cli

    seen: list[bool] = []

    def spawn(argv, **kw):
        raise AssertionError("кошторис не мав відчіплятись")

    monkeypatch.setattr(detach, "spawn", spawn)
    monkeypatch.setattr(cli, "_htr_dry_run", lambda plan, json_out: seen.append(json_out))
    res = CliRunner().invoke(cli.app, ["htr", "supervise", "--plan", str(_plan_file(tmp_path)),
                                       "--detach", "--dry-run", "--session", "htr-dry"])
    assert res.exit_code == 0, res.output
    assert seen == [False]


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
    order: list[str] = []
    monkeypatch.setattr(detach, "spawn", lambda argv, **kw: "задача (тест)")
    monkeypatch.setattr(detach, "cleanup", lambda session: order.append("cleanup") or True)
    monkeypatch.setattr(detach, "task_report", lambda name: order.append("report")
                        or "задача лишилась: планувальник її так і не запустив")
    res = CliRunner().invoke(cli.app, ["htr", "supervise", "--plan", str(_plan_file(tmp_path)),
                                       "--detach", "--session", "htr-dead"])
    assert res.exit_code == 3
    assert "НЕ стартував" in res.output
    # Причина від планувальника — у виводі, і спитана ДО прибирання задачі.
    assert "так і не запустив" in res.output
    assert order == ["report", "cleanup"]


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
    # Без цього на Linux CI із живим systemd пішов би справжній `systemd-run`.
    monkeypatch.setattr(detach, "_systemd_user_ok", lambda: False)

    how = detach._spawn_posix(["htr", "supervise", "--plan", "p.json"],
                              session="htr-px", owner="agent-1")

    assert "відчеплено setsid" in how
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


# ── Linux: systemd замість setsid ───────────────────────────────────────────
#
# 🔴🔴 14.10.2026: п'ять наглядачів, відчеплених setsid, зникли одночасно разом
# із cgroup запуску (systemd, пісочниця агента), а п'ять боксів горіли далі.


class _Run:
    """Підміна `subprocess.run`: відповідає за сценарієм і пише виклики."""

    def __init__(self, answers: dict[str, tuple[int, str]]) -> None:
        self.answers = answers
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        key = " ".join(cmd[:3])
        rc, out = next((v for k, v in self.answers.items() if key.startswith(k)), (0, ""))
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="")


def _posix(monkeypatch, *, systemd: bool, linger: str = "yes") -> _Run:
    monkeypatch.setattr(detach, "_windows", lambda: False)
    monkeypatch.setattr(detach.shutil, "which",
                        lambda n: f"/usr/bin/{n}" if systemd or n == "gpurunner" else None)
    run = _Run({"systemctl --user is-system-running": (1, "degraded\n"),
                "loginctl show-user": (0, f"{linger}\n")})
    monkeypatch.setattr(detach.subprocess, "run", run)
    return run


def test_with_a_user_systemd_the_supervisor_becomes_its_own_unit(monkeypatch) -> None:
    run = _posix(monkeypatch, systemd=True)
    monkeypatch.setenv("VAST_API_KEY", 'se"cr\\et')
    monkeypatch.setattr(detach.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("setsid при живому systemd"))

    how = detach._spawn_posix(["htr", "supervise", "--plan", "p.json"],
                              session="htr-юніт", owner="agent-1")

    cmd = next(c for c in run.calls if c[0] == "systemd-run")
    unit = detach.unit_name("htr-юніт")
    assert cmd[:4] == ["systemd-run", "--user", "--quiet", "--collect"]
    assert f"--unit={unit}" in cmd
    assert unit.isascii() and unit.endswith(".service"), "кирилиця екранована"
    assert "Restart" not in " ".join(cmd), "без справжнього resume перезапуск небезпечний"
    assert cmd[cmd.index("--") + 1:] == ["/usr/bin/gpurunner", "htr", "supervise",
                                         "--plan", "p.json"]
    log = detach._spawn_log_path("htr-юніт")
    assert f"--property=StandardOutput=append:{log}" in cmd
    env_file = detach._env_file_path("htr-юніт")
    assert f"--property=EnvironmentFile={env_file}" in cmd
    assert not any(c.startswith("--setenv") for c in cmd), "ключі не в властивості юніта"
    body = env_file.read_text(encoding="utf-8")
    assert 'GPURUNNER_OWNER="agent-1"' in body
    assert 'VAST_API_KEY="se\\"cr\\\\et"' in body
    assert f"{detach.ENV_DETACHED_HOW}=" in body
    if os.name != "nt":
        assert env_file.stat().st_mode & 0o077 == 0, "файл з ключами — лише власнику"
    assert "systemd" in how and "lingering" not in how

    detach.cleanup("htr-юніт")
    assert not env_file.exists(), "файл з ключами не лишається після наглядача"


def test_without_lingering_the_description_warns_and_nothing_is_enabled(monkeypatch) -> None:
    run = _posix(monkeypatch, systemd=True, linger="no")

    how = detach._spawn_posix(["htr", "supervise"], session="htr-l", owner="")

    assert "lingering вимкнено" in how and "enable-linger" in how
    assert not any("enable-linger" in " ".join(c) for c in run.calls), \
        "lingering вмикає людина, не ми"


def test_without_systemd_setsid_says_it_will_not_survive_a_sandbox(monkeypatch) -> None:
    _posix(monkeypatch, systemd=False)
    seen: dict = {}

    class _Popen:
        def __init__(self, argv, **kw) -> None:
            seen.update(kw, argv=argv)

    monkeypatch.setattr(detach.subprocess, "Popen", _Popen)

    how = detach._spawn_posix(["htr", "supervise"], session="htr-s", owner="")

    assert seen["start_new_session"] is True
    assert "setsid" in how and "пісочниц" in how
    assert seen["env"][detach.ENV_DETACHED_HOW] == how


def test_a_failed_systemd_run_is_an_error_not_a_quiet_setsid(monkeypatch) -> None:
    run = _posix(monkeypatch, systemd=True)
    run.answers["systemd-run --user --quiet"] = (1, "Failed to connect to bus")

    with pytest.raises(OSError, match="systemd-run"):
        detach._spawn_posix(["htr", "supervise"], session="htr-f", owner="")
    assert not detach._env_file_path("htr-f").exists()


def test_unit_report_speaks_for_a_dead_unit(monkeypatch) -> None:
    run = _posix(monkeypatch, systemd=True)
    run.answers["systemctl --user show"] = (
        0, "LoadState=loaded\nActiveState=failed\nSubState=failed\n"
           "Result=exit-code\nExecMainStatus=2\n")
    assert "failed" in detach.unit_report("htr-d") and "код=2" in detach.unit_report("htr-d")

    run.answers["systemctl --user show"] = (0, "LoadState=not-found\n")
    assert "уже немає" in detach.unit_report("htr-d")


def test_kill_stops_the_unit_instead_of_signalling_a_group(monkeypatch) -> None:
    run = _posix(monkeypatch, systemd=True)
    monkeypatch.setattr(detach.os, "killpg", lambda *a: pytest.fail("killpg при юніті"),
                        raising=False)

    done = detach.kill("htr-k", pid=4242)

    assert ["systemctl", "--user", "stop", detach.unit_name("htr-k")] in run.calls
    assert any("юніт" in line for line in done)


def test_posix_dead_start_reports_the_unit_and_the_spawn_log(monkeypatch, tmp_path: Path) -> None:
    """CLI-шлях мертвого старту на POSIX досі не був покритий взагалі."""
    from typer.testing import CliRunner

    from gpurunner import cli

    plan = _plan_file(tmp_path)
    monkeypatch.setattr(cli, "DETACH_START_WAIT_SEC", 0.2)
    monkeypatch.setattr(cli, "DETACH_POLL_SEC", 0.05)

    def spawn(argv, **kw):
        detach._spawn_log_path("htr-pdead").write_text("Traceback: boom", encoding="utf-8")
        return "user-юніт systemd (тест)"

    monkeypatch.setattr(detach, "spawn", spawn)
    monkeypatch.setattr(detach, "task_report", lambda name: "")
    monkeypatch.setattr(detach, "unit_report", lambda s: "юніт: failed/failed (тест)")
    monkeypatch.setattr(detach, "cleanup", lambda s: False)

    out = CliRunner().invoke(cli.app, ["htr", "supervise", "--plan", str(plan),
                                       "--detach", "--session", "htr-pdead"])

    assert out.exit_code == 3
    assert "НЕ стартував" in out.output
    assert "failed/failed" in out.output and "boom" in out.output


def test_the_supervisor_records_how_it_was_detached(monkeypatch) -> None:
    from gpurunner.supervise.state import SupervisorState

    state = SupervisorState(session="htr-h", detached_how="user-юніт systemd x")
    assert state.to_dict()["detached_how"] == "user-юніт systemd x"
