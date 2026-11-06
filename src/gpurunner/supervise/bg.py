"""Довга задача у фоні: без вікна, ОДНА копія, зупинка лише за PID.

    gpurunner bg start <ім'я> [--cwd ТЕКА] -- <команда> <аргументи…>
    gpurunner bg status [<ім'я>]
    gpurunner bg stop <ім'я>

🔴 Навіщо окремо від наглядача. Черга FS-завантаження 10.09.2026 заводилась
руками через `schtasks`, і за один захід наступила на три пастки поспіль:

- видиме вікно на години (задача виконується в інтерактивній сесії);
- ДВІ копії: тригер «+1 хв» поруч із `/run` підняв другу, і обидві писали в
  одну теку кадрів;
- зупиняючи стару, агент добивав процеси фільтром за текстом командного рядка
  — і вбив ВЛАСНУ оболонку, в якій той текст стояв.

Тут кожна закрита кодом: задача без вікна й видаляє себе першим рядком
(`detach.schedule_hidden`); усередині неї біжить `bg _run`, що тримає замок
`bg:<ім'я>` (друга копія з іншого живого pid дістає `LockBusy`) і пише PID-и в
стан; `bg stop` вбиває дерево ЗА PID зі стану, а не за текстом.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpurunner.config import data_dir
from gpurunner.core import locks

TASK_PREFIX = "gpurunner-bg"


class BgBusy(RuntimeError):
    """Задача з цим ім'ям уже біжить."""


def bg_dir() -> Path:
    path = data_dir() / "bg"
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(name: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_." else "-" for c in name).strip("-")
    if not out:
        raise ValueError(f"порожнє ім'я фонової задачі: {name!r}")
    return out


def state_path(name: str) -> Path:
    return bg_dir() / f"{safe_name(name)}.json"


def log_path(name: str) -> Path:
    return bg_dir() / f"{safe_name(name)}.log"


def load(name: str) -> dict[str, Any] | None:
    try:
        data = json.loads(state_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _save(name: str, data: dict[str, Any]) -> None:
    path = state_path(name)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def is_running(state: dict[str, Any] | None) -> bool:
    """Біжить — якщо не записано кінця і процес-тримач живий."""
    if not state or state.get("finished"):
        return False
    pid = int(state.get("pid_runner") or 0)
    return bool(pid) and locks._alive(pid)


def _self_exe() -> list[str]:
    """Чим кликати `gpurunner` із задачі планувальника — спільний пошук detach."""
    from gpurunner.supervise.detach import self_argv

    return self_argv()


def start(name: str, argv: list[str], *, cwd: Path | None = None) -> str:
    """Поставити задачу у фон. Відмова, якщо задача з цим ім'ям уже біжить."""
    name = safe_name(name)
    if not argv:
        raise ValueError("немає команди: `gpurunner bg start <ім'я> -- <команда>`")
    if is_running(load(name)):
        raise BgBusy(f"фонова задача «{name}» уже біжить — `gpurunner bg status {name}`")
    workdir = Path(cwd).resolve() if cwd else bg_dir()
    _save(name, {"name": name, "cmd": argv, "cwd": str(workdir), "log": str(log_path(name)),
                 "queued": _now()})
    inner = [*_self_exe(), "bg", "_run", name, "--cwd", str(workdir), "--", *argv]
    if os.name == "nt":
        from gpurunner.supervise import detach

        quoted = " ".join(f'"{a}"' if (" " in a or not a) else a for a in inner)
        detach.schedule_hidden(f"{TASK_PREFIX}-{name}", bg_dir() / f"{name}.cmd",
                               [f'cd /d "{workdir}"', f'{quoted} > "{log_path(name)}" 2>&1'])
        return f"задача планувальника {TASK_PREFIX}-{name} (без вікна)"
    with log_path(name).open("ab") as log:
        subprocess.Popen(inner, cwd=workdir, stdin=subprocess.DEVNULL, stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
    return "відчеплений процес (setsid)"


def run(name: str, argv: list[str], *, cwd: Path | None = None) -> int:
    """Тіло фонової задачі: замок, дитина, PID-и в стан, код виходу в стан.

    Кличе її сам `bg start` через планувальник; руками — лише для відлагодження.
    """
    name = safe_name(name)
    with locks.hold(f"bg:{name}", owner=f"bg-{name}", session=name,
                    ttl_sec=14 * 24 * 3600, note="фонова задача"):
        state = load(name) or {"name": name}
        child = subprocess.Popen(argv, cwd=str(cwd) if cwd else None)
        state.update({"cmd": argv, "cwd": str(cwd or ""), "pid_runner": os.getpid(),
                      "pid_child": child.pid, "started": _now(), "finished": None,
                      "rc": None})
        _save(name, state)
        rc = child.wait()
        state.update({"finished": _now(), "rc": rc})
        _save(name, state)
    return rc


def stop(name: str) -> list[str]:
    """Убити задачу ЗА PID зі стану — деревом, разом із дитиною."""
    name = safe_name(name)
    state = load(name)
    done: list[str] = []
    if not state:
        return [f"задачі «{name}» немає"]
    pid = int(state.get("pid_runner") or 0)
    if pid and is_running(state):
        if os.name == "nt":
            rc = subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                                capture_output=True, text=True, check=False).returncode
            if rc == 0:
                done.append(f"убито дерево pid {pid}")
        else:
            killpg = getattr(os, "killpg", None)
            getpgid = getattr(os, "getpgid", None)
            if killpg and getpgid:
                try:
                    killpg(getpgid(pid), 15)
                    done.append(f"SIGTERM групі pid {pid}")
                except (OSError, ProcessLookupError):
                    pass
    if os.name == "nt":
        subprocess.run(["schtasks", "/delete", "/f", "/tn", f"{TASK_PREFIX}-{name}"],
                       capture_output=True, text=True, check=False)
    locks.release(f"bg:{name}", owner=f"bg-{name}")
    state.update({"finished": state.get("finished") or _now(),
                  "rc": state.get("rc") if state.get("rc") is not None else "stopped"})
    _save(name, state)
    return done or [f"«{name}» не бігла — стан позначено завершеним"]


def status(name: str | None = None) -> list[dict[str, Any]]:
    names = [safe_name(name)] if name else sorted(p.stem for p in bg_dir().glob("*.json"))
    out = []
    for n in names:
        st = load(n)
        if st is None:
            continue
        out.append({**st, "running": is_running(st)})
    return out
