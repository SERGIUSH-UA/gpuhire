"""Запобіжник дубля: той самий job із тими самими даними не запускається вдруге, поки живий перший.

🔴 26.10.2026: цикл очікування ринку і ручний запуск одночасно подали той самий
`kraken_train` (той самий корпус, ті самі параметри) — два A100 натренували одне
й те саме, $2.2 зайвих. Хто саме подав другий — сесія, цикл, інший агент —
не важить: перевіряти має сам `run`, а не уважність того, хто його кличе.

Ключ прогону — job, бекенд, нормалізовані параметри job'а і вхідна тека
(`input_root` — ручка бекенда, у нормалізованих параметрах її немає, а два
плечі A/B часто відрізняються саме корпусом). Ручки ціни й оренди в ключ не
входять: дубль на дорожчій машині — так само дубль.

Два шари, бо `submit` триває хвилини (оренда, заливка), а хендл потрапляє в
реєстр лише після нього:
- реєстр прогонів — живий хендл із тим самим ключем;
- заявка — файл, що існує на час `submit`; другий запуск у це вікно відмовляє.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gpurunner.config import data_dir
from gpurunner.core import manifest
from gpurunner.core.models import JobHandle, JobStatus

#: Ключ у `JobHandle.params`, під яким лежить ключ прогону.
RUN_KEY = "_run_key"

#: Стани, у яких прогін ще може тарифікуватись. `unknown` — сюди ж: бекенд не
#: дав сигналу, а не сказав «згасло».
_ALIVE = {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.UNKNOWN}

#: Хендл, старший за це, дублем не вважається: реєстр не знає, що бокс згас,
#: доки хтось не спитав `status`, і давній «running» блокував би назавжди.
MAX_AGE_HOURS = 24.0

#: Заявка без живого `submit` старша за це — покинута (процес убили).
CLAIM_STALE_SEC = 3 * 3600


class DuplicateRun(RuntimeError):
    """Такий самий прогін уже живий або саме подається."""


def run_key(job_name: str, backend: str, normalized: dict[str, Any], input_root: Any) -> str:
    payload = {
        "job": job_name,
        "backend": backend,
        "params": {k: v for k, v in normalized.items() if not str(k).startswith("_")},
        "input_root": str(Path(str(input_root)).resolve()).lower() if input_root else "",
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def find_alive(key: str, *, now: datetime | None = None) -> JobHandle | None:
    """Живий хендл із тим самим ключем, найсвіжіший."""
    now = now or datetime.now(UTC)
    floor = now - timedelta(hours=MAX_AGE_HOURS)
    hits = [
        h for h in manifest.load()
        if (h.params or {}).get(RUN_KEY) == key
        and h.status in _ALIVE
        and _aware(h.created_at) >= floor
    ]
    return max(hits, key=lambda h: _aware(h.created_at)) if hits else None


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def _claim_path(key: str) -> Path:
    return data_dir() / "run_claims" / f"{key}.json"


@contextmanager
def claim(key: str, *, owner: str = "") -> Iterator[None]:
    """Зайняти ключ на час `submit`. Зайнятий свіжою заявкою — `DuplicateRun`."""
    path = _claim_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"pid": os.getpid(), "owner": owner, "at": time.time()})
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < CLAIM_STALE_SEC:
                try:
                    who = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    who = {}
                raise DuplicateRun(
                    f"такий самий прогін саме подається (pid {who.get('pid', '?')}, "
                    f"власник «{who.get('owner') or '?'}», {int(age)} с тому)") from None
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        break
    else:
        raise DuplicateRun("заявку на прогін не вдалося зайняти")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
