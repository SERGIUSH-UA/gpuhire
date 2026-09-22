"""Замки між паралельними сесіями на одній машині.

Дві сесії gpurunner ділять усе: акаунт Vast, реєстр прогонів, реєстр боксів,
теки результатів. 2026-08-11 це коштувало п'яти взаємно знищених живих боксів
і кількох гонок за той самий оффер. Власність (`owner`) відповідає на питання
«чиє це»; замки — на питання «чи можна за це братись просто зараз».

Навмисно **файлові й наївні**, без сторонніх залежностей:

- один файл на ресурс, у ньому `{owner, session, pid, ts, note}` — тобто замок
  сам себе пояснює, і людині видно, хто тримає;
- створення через `O_EXCL`, тож перегони на створенні виграє рівно один;
- **протухання за часом і за pid**: сесія, яку вбили, не має блокувати роботу
  назавжди. Мертвий власник — гучний рядок у звіті, а не тиха перехопка.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from gpurunner.config import data_dir

#: Скільки замок лишається чинним без оновлення. Довше за найдовший захід не
#: треба: наглядач і так має власну стелю годин.
DEFAULT_TTL_SEC = 6 * 3600


class LockBusy(RuntimeError):
    """Ресурс уже тримає інша сесія."""

    def __init__(self, resource: str, holder: dict) -> None:
        who = holder.get("owner") or "невідомий"
        session = holder.get("session") or "?"
        age = int(time.time() - float(holder.get("ts") or 0))
        super().__init__(
            f"{resource} уже тримає {who} (сесія {session}, {age // 60} хв тому). "
            f"Це не помилка — це інша сесія робить ту саму роботу."
        )
        self.resource = resource
        self.holder = holder


@dataclass(frozen=True)
class LockInfo:
    resource: str
    owner: str
    session: str
    pid: int
    ts: float


def locks_dir() -> Path:
    return data_dir() / "locks"


def _path(resource: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in resource)[:80]
    return locks_dir() / f"{safe}.lock"


def _alive(pid: int) -> bool:
    """Чи живий процес. Помилятись безпечніше в бік «живий»."""
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        pass
    # 🔴 На Windows `os.kill(pid, 0)` — це `TerminateProcess`, а не перевірка:
    # він або бреше, або вбиває. Без psutil там не питаємо взагалі; протухання
    # замка тоді ловиться лише за TTL, і це правильний бік помилки.
    if sys.platform == "win32":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def read(resource: str) -> dict | None:
    path = _path(resource)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def stale_reason(holder: dict, ttl_sec: int = DEFAULT_TTL_SEC) -> str:
    """Публічна обгортка: чому замок можна перейняти (порожньо — не можна).

    Потрібна поза модулем, щоб відрізнити ЖИВОГО утримувача від покинутого —
    напр. перш ніж «підхопити осиротілий» бокс, який насправді ще рахує.
    """
    return _stale(holder, ttl_sec)


def _stale(holder: dict, ttl_sec: int) -> str:
    """Чому замок можна перейняти — або порожньо, якщо не можна.

    🔴 Порядок перевірок ЗНАЧУЩИЙ: спершу пряма ознака життя (pid), і лише
    потім вік запису. Живий процес не буває «протухлим» — мітка часу ставиться
    раз при взятті й не оновлюється, тож на заході, довшому за TTL, власний
    замок оголошував себе покинутим.

    🔴 Строк судиться ЗА ТИМ, ХТО ТРИМАЄ, а не за тим, хто питає. Раніше читач
    підставляв свій TTL: сесія з `--max-hours 2` вважала протухлим замок
    сесії з `--max-hours 10` уже через 2.5 години. Далі спрацьовувало
    підхоплення «осиротілого» боксу — і A забирала результат та гасила
    машину, на якій B ще рахує. Рівно той інцидент (htr-olhopil, 2026-08-12),
    проти якого й писалась перевірка `_is_adoptable`.
    """
    pid = int(holder.get("pid") or 0)
    if pid and not _alive(pid):
        return f"процес власника (pid {pid}) мертвий"
    if pid and _alive(pid):
        return ""            # живий тримач — жодного TTL не досить
    own_ttl = holder.get("ttl_sec")
    limit = float(own_ttl) if own_ttl else float(ttl_sec)
    age = time.time() - float(holder.get("ts") or 0)
    if age > limit:
        return f"замок протух ({int(age // 60)} хв без оновлення)"
    return ""


def acquire(
    resource: str,
    *,
    owner: str,
    session: str = "",
    ttl_sec: int = DEFAULT_TTL_SEC,
    note: str = "",
) -> LockInfo:
    """Узяти замок або кинути `LockBusy`. Перехоплення протухлого — гучне."""
    path = _path(resource)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    payload = {
        "resource": resource, "owner": owner, "session": session,
        "pid": os.getpid(), "ts": ts, "note": note,
        # Строк, який попросив САМ утримувач: читач не має права судити чужий
        # замок за своєю міркою (див. `_stale`).
        "ttl_sec": int(ttl_sec),
    }
    blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = read(resource) or {}
            if holder.get("owner") == owner and holder.get("session") == session:
                # 🔴 «Своя» сесія з ІНШОГО живого процесу — це не оновлення, а
                # подвійний старт. Відчеплена задача планувальника ставила тригер
                # «+1 хв» і одразу `/run`: другий наглядач тієї ж сесії проходив
                # цей замок як свій і брався за ту саму справу (черга FS,
                # 10.09.2026, — дві копії писали в одну теку кадрів).
                other = int(holder.get("pid") or 0)
                if other and other != os.getpid() and _alive(other):
                    raise LockBusy(resource, holder) from None
                path.write_bytes(blob)  # свій же замок — оновлюємо мітку часу
                break
            why = _stale(holder, ttl_sec)
            if not why:
                raise LockBusy(resource, holder) from None
            print(f"[lock] переймаю {resource}: {why}; тримав "
                  f"{holder.get('owner')} / {holder.get('session')}", flush=True)
            path.unlink(missing_ok=True)
            continue
        else:
            with os.fdopen(fd, "wb") as fh:
                fh.write(blob)
            break
    else:
        raise LockBusy(resource, read(resource) or {})

    return LockInfo(resource=resource, owner=owner, session=session,
                    pid=os.getpid(), ts=ts)


def release(resource: str, *, owner: str) -> bool:
    """Зняти СВІЙ замок. Чужий не чіпаємо — це те саме правило, що з інстансами."""
    holder = read(resource)
    if holder is None:
        return False
    if holder.get("owner") != owner:
        print(f"[lock] {resource} тримає {holder.get('owner')}, не знімаю", flush=True)
        return False
    _path(resource).unlink(missing_ok=True)
    return True


@contextmanager
def hold(resource: str, *, owner: str, session: str = "", ttl_sec: int = DEFAULT_TTL_SEC,
         note: str = ""):
    """`with hold(...)` — узяти й гарантовано віддати."""
    info = acquire(resource, owner=owner, session=session, ttl_sec=ttl_sec, note=note)
    try:
        yield info
    finally:
        release(resource, owner=owner)
