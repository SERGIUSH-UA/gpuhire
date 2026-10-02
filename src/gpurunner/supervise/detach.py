"""Запустити наглядача так, щоб він пережив сесію, яка його запустила.

🔴🔴 Це не зручність, а умова роботи. Наглядач живе годинами, а запускає його
агент із сесії, яку зачищають; виміряно 19.08.2026 тричі поспіль:

| спосіб | що сталось |
|---|---|
| `Start-Process -WindowStyle Hidden` | помер разом із зачисткою сесії |
| `cmd /c` через WMI | те саме |
| `nohup … &` | те саме (на Windows `nohup` не рятує взагалі) |
| `.bat` у власній консолі | пережив, але лишав вікно |
| **Планувальник завдань** | **пережив** |

Тобто відв'язати процес від батька на Windows надійно вміє лише служба
планувальника: вона стартує задачу від себе, а не від нашого дерева процесів.

🔴 Задача створюється як `/sc once` і ВИДАЛЯЄТЬСЯ, щойно наглядач завершився.
Забута задача одного разу підняла НОВУ оренду о 23:59 без жодного нагляду —
саме тому прибирання тут не «охайність», а запобіжник від витрат.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

TASK_PREFIX = "gpurunner-htr"


def task_name(session: str) -> str:
    return f"{TASK_PREFIX}-{session}"


def self_argv() -> list[str]:
    """Чим кликати САМ gpurunner із задачі планувальника, де PATH порожній.

    🔴🔴 Було `which("gpurunner") or sys.executable` — і без gpurunner у PATH
    задача кликала `python.exe htr supervise …`: «can't open file …\\htr».
    Наглядач не стартував, стану не було, а викликач (обгортка над
    `gpurunner htr …`) рапортував «пішов у фон» (10.09.2026, перший бойовий запуск). До того ж `sys.executable` тут
    буває БАЗОВИМ інтерпретатором (Python312), а не venv — тож і «сусід
    python.exe» не рятує. Тому порядок: PATH → сам запущений `gpurunner.exe`
    (`sys.argv[0]`) → `Scripts` у `sys.prefix` (у venv це venv, навіть коли
    `sys.executable` базовий) → поруч із `sys.executable` → `python -m gpurunner`.
    Спільне для наглядача й `gpurunner bg`: дві копії пошуку вже розійшлись раз.
    """
    name = "gpurunner.exe" if os.name == "nt" else "gpurunner"
    found = shutil.which("gpurunner")
    if found:
        return [found]
    me = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if me is not None and me.stem.lower() == "gpurunner" and me.is_file():
        return [str(me.resolve())]
    for cand in (Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin") / name,
                 Path(sys.executable).with_name(name)):
        if cand.is_file():
            return [str(cand)]
    return [sys.executable, "-m", "gpurunner"]


def supported() -> bool:
    """Чи вміємо відчепити процес на цій системі.

    🔴 На POSIX бінарник `setsid` НЕ потрібен: `_spawn_posix` відчіплює через
    `Popen(start_new_session=True)`, тобто системним викликом. Стара умова
    вимагала саме бінарника, а його немає на macOS — і `--detach` там
    відхилявся, хоча все потрібне для нього було на місці.
    """
    return os.name in ("nt", "posix")


def spawn(argv: list[str], *, session: str, owner: str = "") -> str:
    """Запустити `gpurunner htr supervise …` відчепленим. Повертає опис способу.

    `argv` — аргументи ПІСЛЯ `gpurunner`, тобто `["htr", "supervise", ...]`.
    """
    if os.name == "nt":
        return _spawn_schtasks(argv, session=session, owner=owner)
    return _spawn_posix(argv, session=session, owner=owner)


def _script_path(session: str) -> Path:
    from gpurunner.config import data_dir

    path = data_dir() / "htr" / f"{session}.cmd"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _launcher_path(session: str) -> Path:
    """Скрипт WSH, який ховає вікно консолі. Живе поруч із самою обгорткою."""
    return _script_path(session).with_suffix(".vbs")


def _spawn_schtasks(argv: list[str], *, session: str, owner: str) -> str:
    """Windows: задача планувальника на «через хвилину», яку сама себе прибере.

    🔴🔴 Команда їде ФАЙЛОМ-ОБГОРТКОЮ, а не рядком у `/tr`, і це не смак.
    Спроба зібрати `cmd /c "…"` з екранованими лапками провалилась на першому ж
    бойовому запуску 04.09.2026: у задачі опинялась зворотна скісна перед
    кожною лапкою навколо шляху. Так екранує C-рантайм, а `cmd` такого не
    розуміє — він бачить саме скісну. Задача відпрацьовувала за секунду з кодом
    1, стан не створювався, і причини не було ніде.

    З файлом вкладених лапок немає взагалі: у `/tr` їде один шлях, а всі лапки
    лишаються всередині скрипта, де їх читає той самий `cmd`, що їх і написав.
    Побічний виграш: файл можна прочитати очима й запустити руками.
    """
    name = task_name(session)
    command = " ".join(f'"{a}"' if (" " in a or not a) else a for a in [*self_argv(), *argv])
    script = _script_path(session)
    lines: list[str] = []
    # 🔴 Робочий каталог. Задача планувальника не має його взагалі й дістає
    # `C:\Windows\System32`, де створити теку не можна. Наглядач тепер усі свої
    # шляхи будує абсолютно, але другий рубіж дешевий: будь-який відносний
    # шлях, який колись знову з'явиться, впаде тут у видиме місце, а не в
    # системну теку. Спіймано бойовим експериментом 04.09.2026 —
    # `PermissionError: 'out'` посеред рятувального забору.
    workdir = _spawn_log_path(session).parent
    lines.append(f'cd /d "{workdir}"')
    if owner:
        lines.append(f"set GPURUNNER_OWNER={owner}")
    # 🔴 Вивід МУСИТЬ кудись іти. Власний лог наглядач заводить лише ПІСЛЯ того,
    # як прочитав план і створив стан; усе, що падає раніше, інакше зникає
    # безслідно — лишається код повернення в планувальнику й нічого більше.
    lines.append(f'{command} > "{_spawn_log_path(session)}" 2>&1')
    schedule_hidden(name, script, lines)
    return f"задача планувальника {name} (без вікна)"


def schedule_hidden(name: str, script: Path, lines: list[str]) -> None:
    """Задача планувальника БЕЗ ВІКНА, що видаляє СЕБЕ першим рядком.

    Спільне для наглядача й `gpurunner bg`: обидві пастки нижче куплені
    інцидентами, і тримати їх у двох копіях означало б полагодити одну.
    """
    body = ["@echo off",
            # 🔴🔴 Задача прибирає СЕБЕ першим рядком, а не в `finally`.
            # Тригер `/sc once` на «+1 хв» стоїть поруч із негайним `/run`, а
            # пускач WSH виходить за мить — тож для планувальника задача вже
            # завершена, і за хвилину тригер піднімав ДРУГУ копію. 10.09.2026
            # саме так двічі стартувала черга FS, і дві копії писали в одну
            # теку кадрів. Видалення визначення не чіпає процесу, що вже біжить.
            f'schtasks /delete /f /tn "{name}" >nul 2>&1',
            *lines]
    script.write_text("\r\n".join(body) + "\r\n", encoding="utf-8")

    # 🔴 БЕЗ ВІКНА. Задача планувальника виконується в ІНТЕРАКТИВНІЙ сесії, і
    # `cmd` відкриває чорне вікно консолі на весь захід — тобто на години.
    #
    # ⚠ Штатний спосіб це прибрати (`/ru <користувач> /np`, «виконувати, коли
    # користувач не ввійшов») на цій машині ЗАБОРОНЕНО політикою — перевірено
    # 04.09.2026, `schtasks` відмовляє. Тому вікно ховаємо інакше: задача
    # запускає крихітний скрипт WSH, а той стартує наш `cmd` зі стилем вікна 0.
    # Прав не треба, вікна немає, процес переживає і сам скрипт, і сесію.
    launcher = script.with_suffix(".vbs")
    # 🔴 UTF-16, а не ASCII. У шлях скрипта входить ІМ'Я СПРАВИ, а наші справи
    # звуться `spr-47а` і `spr-84г` — з українськими літерами. На ASCII такий
    # запис падає `UnicodeEncodeError` ще до планувальника: захід не стартує
    # взагалі, а на диску лишається порожній `.vbs` — 21.09.2026 так тихо
    # загинув `htr-spr-84г-q2-0921-0048` (0 байт), а потім і `spr-47а`.
    # WSH читає VBS у UTF-16 за BOM — перевірено живим запуском.
    launcher.write_text(
        f'CreateObject("WScript.Shell").Run """{script}""", 0, False\r\n',
        encoding="utf-16")

    start_at = (datetime.now() + timedelta(minutes=1)).strftime("%H:%M")
    subprocess.run(
        ["schtasks", "/create", "/f", "/tn", name, "/sc", "once",
         "/st", start_at, "/tr", str(launcher)],
        check=True, capture_output=True, text=True)
    subprocess.run(["schtasks", "/run", "/tn", name],
                   check=True, capture_output=True, text=True)


def _spawn_posix(argv: list[str], *, session: str, owner: str) -> str:
    env = dict(os.environ)
    if owner:
        env["GPURUNNER_OWNER"] = owner
    # 🔴 Вивід — у той самий `spawn.log`, що й на Windows, а не в DEVNULL.
    # Власний лог наглядач заводить лише ПІСЛЯ того, як прочитав план; усе, що
    # падає раніше (битий план, немає ключа), на POSIX зникало безслідно, і
    # `spawn_log()` — єдина діагностика невдалого старту — завжди був порожній.
    log_path = _spawn_log_path(session)
    # Дескриптор у батька закриваємо одразу: дитина тримає свою копію, а наш
    # процес за мить виходить.
    with open(log_path, "wb") as log:
        subprocess.Popen(
            [*self_argv(), *argv], env=env, start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    return "відчеплений процес (setsid)"


def _spawn_log_path(session: str) -> Path:
    """Куди відчеплена задача пише все, що сказала ДО власного лога."""
    from gpurunner.config import data_dir

    path = data_dir() / "htr" / f"{session}.spawn.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def spawn_log(session: str) -> str:
    """Що сказала відчеплена задача на старті. Порожньо — не сказала нічого."""
    try:
        return _spawn_log_path(session).read_text(
            encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def cleanup(session: str) -> bool:
    """Прибрати задачу планувальника. Викликається наглядачем у `finally`.

    🔴 Мовчазний no-op, якщо задачі немає: наглядача могли запустити й руками,
    а падати на прибиранні того, чого не заводили, — це втратити вердикт уже
    завершеного заходу.
    """
    if os.name != "nt":
        return False
    try:
        done = subprocess.run(
            ["schtasks", "/delete", "/f", "/tn", task_name(session)],
            capture_output=True, text=True, check=False)
    except OSError:
        return False
    # Обгортку прибираємо разом із задачею; лог старту лишаємо — він може
    # бути єдиним, що пояснює, чому заходу не сталося.
    with contextlib.suppress(OSError):
        _script_path(session).unlink(missing_ok=True)
        _launcher_path(session).unlink(missing_ok=True)
    return done.returncode == 0


def kill(session: str, pid: int = 0) -> list[str]:
    """Убити наглядача сесії й прибрати за ним. Повертає, що саме зроблено.

    🔴 `Stop-Process` по pid `uv.exe` наглядача НЕ вбиває: під ним живе окреме
    дерево, і головний процес лишається (17.08.2026 довелось шукати `python.exe`
    за рядком `--plan` руками). Тому вбиваємо ДЕРЕВО.
    """
    done: list[str] = []
    if pid > 0:
        if os.name == "nt":
            result = subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                                    capture_output=True, text=True, check=False)
            if result.returncode == 0:
                done.append(f"убито дерево процесів pid {pid}")
        else:
            # `killpg`/`getpgid` існують лише на POSIX, і mypy на Windows про
            # це знає — беремо через getattr, щоб не тягти ignore на весь файл.
            killpg = getattr(os, "killpg", None)
            getpgid = getattr(os, "getpgid", None)
            if killpg and getpgid:
                try:
                    killpg(getpgid(pid), 15)
                    done.append(f"SIGTERM групі pid {pid}")
                except (OSError, ProcessLookupError):
                    pass
    if cleanup(session):
        done.append(f"видалено задачу {task_name(session)}")
    return done


def release_locks(session: str, *, owner: str) -> list[str]:
    """Зняти замки справ, які тримала ця сесія.

    Без цього наступний захід відмовляється брати справу («тримає замок іншої
    сесії»), хоч тієї сесії давно немає — а мертві замки накопичувались до
    шести десятків віком 10 000+ хвилин.
    """
    import json

    from gpurunner.core import locks

    freed: list[str] = []
    directory = locks.locks_dir()
    if not directory.is_dir():
        return freed
    # 🔴 Ім'я файла — САНІТИЗОВАНИЙ ресурс (`market:rent` → `market_rent`), тож
    # відновити ресурс з імені не можна: двокрапка й підкреслення дають однакове
    # ім'я. Справжнє ім'я лежить усередині, тому читаємо вміст.
    for path in sorted(directory.glob("*.lock")):
        try:
            holder = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(holder.get("session") or "") != session:
            continue
        resource = str(holder.get("resource") or "")
        if resource and locks.release(
                resource, owner=owner or str(holder.get("owner") or "")):
            freed.append(resource)
    return freed


def state_pid(session: str) -> int:
    """pid наглядача зі стану сесії (0 — невідомо)."""
    from gpurunner.supervise import state as state_mod

    data = state_mod.load(session) or {}
    try:
        return int(data.get("pid") or 0)
    except (TypeError, ValueError):
        return 0


def log_path_for(session: str) -> Path | None:
    from gpurunner.supervise import state as state_mod

    data = state_mod.load(session) or {}
    raw = data.get("log_path")
    return Path(str(raw)) if raw else None
