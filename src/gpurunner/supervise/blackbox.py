"""Чорна скринька наглядача: що з ним сталося, коли він зник без вердикту.

🔴🔴 Причина існування. 22.09.2026 два наглядачі поспіль зникли посеред заходу,
лишивши бокс горіти: `htr-spr-576-q9` прожив 54 хв, `htr-hvist-adopt2` — 6 хв.
В обох випадках на диску не було НІЧОГО: ні трасування, ні вердикту, ні події
в журналі Windows, ні рядка в лозі. Розслідування впиралось у те, що стан
обривався на звичайному тіку, і відрізнити «впав сам» від «убили ззовні» не
було чим.

Тут інструментуються всі шляхи смерті, які процес може зафіксувати САМ:

| шлях                          | що лишиться                    |
|-------------------------------|--------------------------------|
| необроблений виняток          | трасування в `sys.excepthook`  |
| виняток у потоці              | трасування в `threading`-хуку  |
| segfault / abort / MemoryError| `faulthandler`                 |
| `taskkill` без `/F`, Ctrl+C   | рядок із номером сигналу       |
| штатний вихід                 | рядок `atexit` із кодом        |
| смерть батька (сесія/шелл)    | запис «батько зник» до смерті  |

🔴 І головне — ТИША ТЕЖ Є ДОКАЗОМ. `TerminateProcess` (це `taskkill /F`,
`Stop-Process -Force` і вбивство дерева процесів) не дає Python жодного шансу:
жоден хук не спрацює. Тому якщо після смерті в скриньці лежать хлібні крихти
аж до останньої секунди й ЖОДНОГО запису про причину — це не «невідомо», а
точний діагноз: процес убили ззовні жорстко. Без цього модуля той самий факт
виглядав як «просто зник».

Крихти пишуться в окремий файл (`<сесія>.blackbox.log`), а не в лог наглядача:
раз на 15 с вони б утопили інциденти, заради яких лог читають.
"""

from __future__ import annotations

import atexit
import contextlib
import faulthandler
import os
import signal
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

#: Як часто лишати хлібну крихту. 15 с — компроміс: точність визначення
#: моменту смерті проти розміру файла за 14-годинний захід (~3400 рядків).
CRUMB_SEC = 15.0

#: Скільки крихт тримати у файлі. Кільце, бо цінні лише останні перед смертю.
CRUMB_KEEP = 400


def _now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def _psutil() -> Any | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def _parent_chain(pid: int, depth: int = 4) -> str:
    """Ланцюг батьків процесу — ХТО його насправді запустив.

    🔴 Це відповідь на питання, яке двічі коштувало оренди: наглядача пустили
    відчеплено (планувальник → `wscript` → `cmd`) чи він дитина шелла агента?
    У другому випадку він приречений, і знати це треба з першого рядка лога, а
    не з розтину через годину.
    """
    ps = _psutil()
    if ps is None:
        return f"pid {pid} (psutil немає — ланцюг невідомий)"
    out = []
    try:
        proc = ps.Process(pid)
        for _ in range(depth):
            proc = proc.parent()
            if proc is None:
                break
            out.append(f"{proc.name()}[{proc.pid}]")
    except Exception:  # розтин не має права впасти
        pass
    return " ← ".join(out) or "батьків не видно"


class Blackbox:
    """Реєстратор смерті. Створюється один на процес наглядача."""

    def __init__(self, path: Path, session: str, log_fh: IO[str] | None = None) -> None:
        self.path = path
        self.session = session
        self.log_fh = log_fh
        self.pid = os.getpid()
        self.ppid = os.getppid()
        self._stop = threading.Event()
        self._crumbs: list[str] = []
        self._lock = threading.Lock()
        self._fault_fh: IO[str] | None = None

    # ---- запис ----

    def _write(self, line: str, *, crumb: bool = False) -> None:
        stamped = f"{_now()} {line}"
        with self._lock:
            if crumb:
                self._crumbs.append(stamped)
                if len(self._crumbs) > CRUMB_KEEP:
                    del self._crumbs[: len(self._crumbs) - CRUMB_KEEP]
            else:
                self._crumbs.append(stamped)
            with contextlib.suppress(OSError):
                self.path.write_text("\n".join(self._crumbs) + "\n", encoding="utf-8")
        # Причини смерті дублюються в лог наглядача: його читають першим.
        if not crumb and self.log_fh is not None:
            with contextlib.suppress(OSError, ValueError):
                print(f"[blackbox] {line}", file=self.log_fh, flush=True)

    # ---- встановлення ----

    def install(self) -> Blackbox:
        self._write(
            f"НАРОДЖЕННЯ сесії {self.session}: pid {self.pid}, ppid {self.ppid}, "
            f"батьки: {_parent_chain(self.pid)}"
        )
        self._write(f"   запуск: {sys.executable} {' '.join(sys.argv)}")
        self._write(f"   тека: {os.getcwd()}")
        self._install_faulthandler()
        self._install_excepthooks()
        self._install_signals()
        atexit.register(self._on_exit)
        self._start_crumbs()
        return self

    def _install_faulthandler(self) -> None:
        """Ловить segfault, abort і збій у C-розширенні (paramiko, cryptography).

        Файл окремий і тримається ВІДКРИТИМ: faulthandler пише в нього вже з
        розваленого інтерпретатора, коли відкривати щось пізно.
        """
        try:
            self._fault_fh = self.path.with_suffix(".fault.log").open("a", encoding="utf-8")
            faulthandler.enable(file=self._fault_fh, all_threads=True)
        except (OSError, ValueError):
            self._fault_fh = None

    def _install_excepthooks(self) -> None:
        prev = sys.excepthook

        def hook(exc_type, exc, tb):  # type: ignore[no-untyped-def]
            self._write("СМЕРТЬ: необроблений виняток у головному потоці\n"
                        + "".join(traceback.format_exception(exc_type, exc, tb)))
            prev(exc_type, exc, tb)

        sys.excepthook = hook

        # 🔴 Потоки — окремий хук. Наглядач тримає серцебиття й опитування боксу
        # у потоках, і виняток там до `sys.excepthook` не доходить узагалі.
        prev_thread = threading.excepthook

        def thook(args):  # type: ignore[no-untyped-def]
            self._write(
                f"ВИНЯТОК У ПОТОЦІ {getattr(args.thread, 'name', '?')}\n"
                + "".join(traceback.format_exception(
                    args.exc_type, args.exc_value, args.exc_traceback))
            )
            prev_thread(args)

        threading.excepthook = thook

    def _install_signals(self) -> None:
        """Сигнали, які Windows таки доставляє: м'який `taskkill`, Ctrl+C/Break.

        🔴 `taskkill /F` сюди НЕ потрапляє — і саме тому відсутність цього
        рядка в скриньці є доказом жорсткого вбивства, а не браком приладу.
        """
        names = ["SIGTERM", "SIGINT"]
        if os.name == "nt":
            names.append("SIGBREAK")
        for name in names:
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            prev = signal.getsignal(sig)

            def handler(signum, frame, _name=name, _prev=prev):  # type: ignore[no-untyped-def]
                self._write(f"СМЕРТЬ: сигнал {_name} ({signum}); стек:\n"
                            + "".join(traceback.format_stack(frame)))
                if callable(_prev):
                    _prev(signum, frame)
                else:
                    raise SystemExit(128 + int(signum))

            with contextlib.suppress(OSError, ValueError):
                signal.signal(sig, handler)

    def _on_exit(self) -> None:
        self._stop.set()
        self._write("ШТАТНИЙ ВИХІД: інтерпретатор завершується (atexit)")
        if self._fault_fh is not None:
            with contextlib.suppress(OSError):
                self._fault_fh.close()

    # ---- хлібні крихти ----

    def _start_crumbs(self) -> None:
        t = threading.Thread(target=self._crumb_loop, name="blackbox", daemon=True)
        t.start()

    def _crumb_loop(self) -> None:
        ps = _psutil()
        proc = None
        if ps is not None:
            with contextlib.suppress(Exception):
                proc = ps.Process(self.pid)
        parent_seen = True
        while not self._stop.wait(CRUMB_SEC):
            bits = [f"живий {int(time.monotonic())}"]
            if proc is not None:
                with contextlib.suppress(Exception):
                    bits.append(f"RSS {proc.memory_info().rss / 2 ** 20:.0f} МБ")
                    bits.append(f"потоків {proc.num_threads()}")
                    bits.append(f"дескрипторів {proc.num_handles()}"
                                if hasattr(proc, "num_handles") else "")
            # 🔴 Смерть БАТЬКА фіксується окремо: якщо наглядач зник слідом за
            # ним, це вбивство дерева процесів, а не власна біда наглядача.
            if ps is not None and parent_seen and not ps.pid_exists(self.ppid):
                parent_seen = False
                self._write(f"⚠ БАТЬКО ЗНИК: ppid {self.ppid} більше не існує "
                            f"(якщо слідом зникну і я — це вбивство дерева)")
            self._write(" · ".join(b for b in bits if b), crumb=True)


def install(session: str, log_fh: IO[str] | None = None,
            state_dir: Path | None = None) -> Blackbox | None:
    """Поставити скриньку. Не має права завалити захід, тому все під `suppress`."""
    try:
        from gpurunner.supervise.state import state_dir as default_dir

        base = state_dir or default_dir()
        base.mkdir(parents=True, exist_ok=True)
        return Blackbox(base / f"{session}.blackbox.log", session, log_fh).install()
    except Exception as exc:
        print(f"[blackbox] ⚠ не встановилась: {exc!r}", flush=True)
        return None


def postmortem(session: str, state_dir_: Path | None = None) -> str:
    """Розтин: що скринька каже про смерть цієї сесії.

    Читається людиною або агентом після `orphaned` — щоб замість «просто зник»
    був один із названих вище діагнозів.
    """
    from gpurunner.supervise.state import state_dir as default_dir

    base = state_dir_ or default_dir()
    path = base / f"{session}.blackbox.log"
    if not path.is_file():
        return f"скриньки немає ({path}) — захід стартував до її появи"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"скриньку не прочитати: {exc!r}"

    reasons = [ln for ln in lines
               if "СМЕРТЬ" in ln or "ШТАТНИЙ ВИХІД" in ln or "ВИНЯТОК" in ln
               or "БАТЬКО ЗНИК" in ln]
    fault = path.with_suffix(".fault.log")
    if fault.is_file() and fault.stat().st_size > 0:
        reasons.append(f"faulthandler лишив запис: {fault}")
    last = lines[-1] if lines else "(порожньо)"
    if not reasons:
        return ("🔴 ПРИЧИНИ НЕМАЄ, А КРИХТИ Є — процес убито ЖОРСТКО ззовні "
                "(`taskkill /F`, `Stop-Process -Force` або вбивство дерева). "
                "Жоден хук Python при `TerminateProcess` не спрацьовує.\n"
                f"   остання крихта: {last}")
    return "\n".join(["причини, які скринька зафіксувала:", *reasons,
                      f"   остання крихта: {last}"])
