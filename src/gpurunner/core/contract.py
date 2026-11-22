"""Машинний вивід CLI: те, що читають інші програми.

🔴 Навіщо окремий модуль. Споживачі (`nyshporka`, обгортки дослідницьких
репозиторіїв) роками читали ЛЮДСЬКИЙ вивід: регекс на `submitted <id>`, пошук
слів `running`/`failed` у панелі статусу. Такий стик ламається від правки
кольору чи перекладу фрази — і ламається мовчки, під час платної оренди.
Тут форма відповіді описана один раз, і тест тримає її незмінною.

Правила для споживача:

* прапорець `--json` → у stdout рівно ОДИН рядок JSON-об'єкта, і він ОСТАННІЙ
  рядок, що починається з `{`. Брати саме останній: SDK бекендів подекуди
  друкують у stdout власний прогрес, і заборонити їм це не можна;
* усе людське (таблиці, попередження про тарифікацію) з `--json` іде в stderr;
* код виходу той самий, що й без `--json`; при відмові об'єкт теж друкується —
  з `"ok": false` і полем `error`;
* `schema` росте лише тоді, коли поле зникає або міняє зміст. Нове поле —
  не привід: споживач зайві поля ігнорує.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Iterator
from typing import Any

from rich.console import Console

SCHEMA = 1

#: Стани, які бачить споживач, — значення `JobStatus`, без синонімів.
STATES = ("queued", "running", "completed", "failed", "cancelled", "unknown")
TERMINAL = ("completed", "failed", "cancelled")


def state_of(status: Any) -> str:
    """`JobStatus` або рядок → один зі `STATES`."""
    value = str(getattr(status, "value", status) or "").lower()
    return value if value in STATES else "unknown"


def emit(payload: dict[str, Any]) -> None:
    """Один рядок у справжній stdout — повз rich, щоб той не переніс і не розфарбував."""
    line = json.dumps({"schema": SCHEMA, **payload}, ensure_ascii=False, default=str)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


@contextlib.contextmanager
def human_to_stderr(console: Console, enabled: bool) -> Iterator[None]:
    """На час машинного виводу людські повідомлення консолі йдуть у stderr."""
    if not enabled:
        yield
        return
    # 🔴 Зберігається саме `_file`, а не `console.file`. Властивість `file` у rich
    # лінива: без явного потоку вона щоразу бере ПОТОЧНИЙ `sys.stdout`. Повернути
    # після себе її значення означало б прибити консоль до потоку, який був
    # stdout на ту мить, — і весь подальший людський вивід пішов би в нього
    # (у тестах — у вже закритий буфер попереднього виклику).
    previous = getattr(console, "_file", None)
    console.file = sys.stderr
    try:
        yield
    finally:
        console._file = previous


def handle_payload(handle: Any) -> dict[str, Any]:
    """Спільна частина для `run` і `status`: хто це і де воно."""
    return {
        "handle": handle.id,
        "short": handle.id[:8],
        "backend": handle.backend,
        "job": handle.job_name,
        "gpu": handle.gpu,
        "remote_id": handle.remote_id,
        "state": state_of(handle.status),
        "output_dir": handle.output_dir or "",
    }
