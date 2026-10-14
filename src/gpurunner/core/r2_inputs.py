"""Входи задачі через R2: залити теку один раз за ключем вмісту, бокс тягне presigned-посиланням.

Навіщо: бекенд Vast заливав теки входів (корпус, пакет заміру) системним `scp`
ІЗ ДОМУ на кожну оренду — коли машина вже взята й тарифікується. Корпус трену й
пакет заміру разом важать 2–3 ГБ, тобто 5–15 хв оплачуваного простою на кожен
трен і ще стільки ж на кожен окремий замір тих самих даних.

Тепер та сама тека пакується в той самий tar (файли верхнього рівня, як і в
`scp`-шляху), але кладеться в R2 під ключем, який залежить від ВМІСТУ: ім'я,
розмір і час зміни кожного файла. Друга задача з тими самими входами не заливає
нічого — бокс бере архів із R2 за presigned GET на швидкості дата-центру.
Ключі R2 на чужу машину не потрапляють: лише посилання на один об'єкт, що
спливає за `URL_HOURS`.

Режими (`-p inputs_via=…`): `auto` — R2, коли він налаштований і входу ≥ `MIN_MB`;
`r2` — завжди R2; `scp` — старий шлях. Будь-який збій R2 в `auto` повертає до `scp`.

Приймач на боксі той самий, що в `scp`: розмір архіву, потім кількість
розпакованих файлів. Прибирання — з Python: `gpurunner.htr.r2.prune(["inputs/"], yes=True)`
(окремої команди CLI немає); об'єкти під цим префіксом не тримають нічого, крім часу
наступної заливки.
"""

from __future__ import annotations

import hashlib
import shlex
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

PREFIX = "inputs"
#: Нижче цього `scp` не повільніший за похід у R2, а дрібні тести й службові
#: входи не мусять залежати від мережі до бакета.
MIN_MB = 20
URL_HOURS = 12.0


def content_key(slug: str, files: list[Path]) -> str:
    """Ключ об'єкта: однаковий для того самого вмісту, інший — для будь-якої зміни."""
    h = hashlib.sha256(slug.encode("utf-8"))
    for p in sorted(files, key=lambda x: x.name):
        st = p.stat()
        h.update(f"\0{p.name}\0{st.st_size}\0{st.st_mtime_ns}".encode())
    return f"{PREFIX}/{slug}-{h.hexdigest()[:16]}.tar"


def wanted(via: str, inputs: dict[str, Path]) -> bool:
    """Чи везти входи через R2 у цьому режимі."""
    via = (via or "auto").strip().lower()
    if via == "scp":
        return False
    from gpurunner.htr import r2

    if not r2.configured():
        if via == "r2":
            raise RuntimeError("inputs_via=r2, але R2 не налаштований (gpurunner r2 setup)")
        return False
    if via == "r2":
        return True
    total = sum(p.stat().st_size for d in inputs.values() for p in d.iterdir() if p.is_file())
    return total >= MIN_MB * 1024 * 1024


def stage(slug: str, files: list[Path], *, log: Callable[[str], None] = print,
          s3: Any = None) -> tuple[str, int]:
    """Покласти tar теки в R2, якщо його там ще немає. → (presigned GET, розмір архіву)."""
    from gpurunner.htr import r2

    key = content_key(slug, files)
    have = r2.head(key, s3=s3)
    if have and have["size"] > 0:
        log(f"[inputs] {slug}: уже в R2 ({have['size'] / 2**20:.0f} МБ, {key}) — не заливаю")
        return r2.get_url(key, hours=URL_HOURS, s3=s3), int(have["size"])
    with tempfile.TemporaryDirectory(prefix="grinput-") as tmp:
        ball = Path(tmp) / Path(key).name
        with tarfile.open(ball, "w") as tf:
            for p in files:
                tf.add(p, arcname=p.name)
        size = ball.stat().st_size
        log(f"[inputs] {slug}: заливаю в R2 {size / 2**20:.0f} МБ → {key}")
        got = r2.put(ball, prefix=PREFIX, s3=s3, quiet=True)
    if got != key:  # pragma: no cover — r2.put складає ключ із префікса й імені
        raise RuntimeError(f"R2 поклав {got}, а чекали {key}")
    return r2.get_url(key, hours=URL_HOURS, s3=s3), size


def fetch_cmd(url: str, remote_tar: str) -> str:
    """Команда для боксу: стягнути архів і надрукувати його розмір (для звірки)."""
    return (f"curl -fsSL --retry 5 --retry-delay 3 -o {shlex.quote(remote_tar)} {shlex.quote(url)} "
            f"&& stat -c %s {shlex.quote(remote_tar)}")
