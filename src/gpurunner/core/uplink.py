"""Заміри каналу дім → машина: журнал на цьому комп'ютері.

Канал до орендованої машини обмежує здебільшого НАШ аплінк, а не хост: заміри
на різних машинах того самого дому лягають поруч (1.08–1.2 МБ/с, 23.09.2026).
Тому минулі доставки дають прогноз наступної ще ДО оренди, і захід, що
витратив би години оплаченої машини на перевезення, спиняється безкоштовно.

Журнал лежить у `data_dir()`, а не в реєстрі машин: реєстр переносять між
комп'ютерами (`GPURUNNER_REPO_DATA_DIR`), а аплінк — властивість мережі, у
якій стоїть саме цей комп'ютер.
"""
from __future__ import annotations

import contextlib
import json
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path

from gpurunner.config import data_dir

#: Замір на меншому обсязі — шум: вісім потоків не встигають розігнатись.
MIN_BYTES = 20_000_000
#: Скільки останніх замірів дивитись і наскільки давніх.
RECENT = 5
MAX_AGE_DAYS = 30


def path() -> Path:
    return data_dir() / "uplink.jsonl"


def record(mbs: float, nbytes: int, *, via: str = "",
           machine_id: int | None = None) -> None:
    """Дописати замір. Збій запису — не привід валити доставку."""
    if mbs <= 0 or nbytes < MIN_BYTES:
        return
    row = {"ts": datetime.now(tz=UTC).isoformat(timespec="seconds"),
           "mbs": round(mbs, 3), "bytes": int(nbytes), "via": via,
           "machine_id": machine_id}
    with contextlib.suppress(OSError):
        target = path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def typical(*, now: datetime | None = None) -> float | None:
    """Медіана останніх замірів, МБ/с; `None` — свіжих замірів немає."""
    try:
        lines = path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    since = (now or datetime.now(tz=UTC)) - timedelta(days=MAX_AGE_DAYS)
    rates: list[float] = []
    for line in lines:
        try:
            row = json.loads(line)
            ts = datetime.fromisoformat(str(row["ts"]))
            mbs = float(row["mbs"])
        except (ValueError, KeyError, TypeError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if ts >= since and mbs > 0 and int(row.get("bytes") or 0) >= MIN_BYTES:
            rates.append(mbs)
    recent = rates[-RECENT:]
    return statistics.median(recent) if recent else None
