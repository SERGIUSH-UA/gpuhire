"""Пошук на ринку Vast: денна квота рядків, журнал запитів, спільний кеш видачі.

🔴 Vast рахує не лише запити на секунду (це тримає `_throttle_vast_api`), а й
РЯДКИ пошукової видачі за добу — на акаунт: 20 000. Один добір наглядача з
ціллю темпу — це два проходи ринку по 100 рядків і до восьми адресних запитів
зірок по 4, тобто ~230 рядків; чекання ринку повторювало добір щохвилини, а
обхідний «швидше, ніж чекати» — ще раз. Кампанія 06.10.2026 (42 справи, кілька
одночасних наглядачів) вичерпала квоту за вечір, і відповідь

    429 {"error":"search_quota_exceeded","retry_after":8181,"limit":20000,"remaining":0}

кінчалась так: 8181 с різались до 30, три повтори, помилку ковтав добір, і
людина бачила «ринок порожній» — хоча ринок ніхто не дивився.

Тут три речі, і всі спільні для процесів одного користувача (лежать під
`data_dir()`, як і дросель):

- **запобіжник квоти** — після такої відповіді пошук до `reset_at` не ходить
  у мережу зовсім; оренда, статус і гасіння працюють як працювали;
- **журнал** `vast_search_audit.jsonl` — хто, навіщо, скільки рядків попросив і
  отримав; без нього розкласти 20 000 рядків між командами неможливо;
- **кеш видачі** на `CACHE_TTL_SEC` — dry-run, живий старт і паралельні
  наглядачі з тим самим запитом беруть одну видачу.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from gpurunner.core.backend import SearchQuotaExceeded

#: Скільки секунд видача ринку вважається свіжою. Довше — і наглядач бере
#: оффер, якого вже нема (оренда тоді впаде й скине кеш); коротше — і кеш не
#: перекриває навіть dry-run → старт. Перекривається `GPURUNNER_MARKET_CACHE_SEC`
#: (0 — вимкнути).
CACHE_TTL_SEC = 120.0

#: `retry_after` понад це — не короткий rate-limit, а вичерпаний ліміт: чекати
#: всередині запиту не можна (наглядач спав би годинами з відкритим сокетом).
LONG_RETRY_AFTER_SEC = 60.0

_purpose: contextvars.ContextVar[str] = contextvars.ContextVar("vast_search_purpose",
                                                               default="")


@contextlib.contextmanager
def purpose(name: str) -> Iterator[None]:
    """Позначити, НАВІЩО йдуть пошуки всередині блоку (`estimate`, `pick`, …)."""
    token = _purpose.set(name)
    try:
        yield
    finally:
        _purpose.reset(token)


def current_purpose() -> str:
    return _purpose.get() or "search"


def _dir() -> Path:
    from gpurunner.config import data_dir

    return data_dir()


# ---- квота -------------------------------------------------------------------


def quota_from_body(body: str) -> SearchQuotaExceeded | None:
    """Відповідь 429 — це вичерпана квота? Тоді виняток із її числами, інакше None.

    Ознаки — будь-яка з трьох: код помилки Vast, нуль залишку, або
    `retry_after`, якого всередині запиту не перечекати.
    """
    try:
        data = json.loads(body or "{}")
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    error = str(data.get("error") or "")
    try:
        retry_after = float(data.get("retry_after") or 0)
    except (TypeError, ValueError):
        retry_after = 0.0
    remaining = data.get("remaining")
    quota = ("quota" in error) or (remaining is not None and _num(remaining) <= 0) \
        or retry_after > LONG_RETRY_AFTER_SEC
    if not quota:
        return None
    return SearchQuotaExceeded(
        str(data.get("msg") or error or "квота пошуку Vast вичерпана"),
        limit=int(_num(data.get("limit"))),
        remaining=int(_num(remaining)),
        retry_after=retry_after,
        reset_at=time.time() + retry_after,
    )


def _num(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _block_path() -> Path:
    return _dir() / "vast_search_block.json"


def block(err: SearchQuotaExceeded) -> None:
    """Закрити пошук для всіх процесів до `err.reset_at`."""
    path = _block_path()
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "reset_at": err.reset_at, "limit": err.limit, "remaining": err.remaining,
            "retry_after": err.retry_after, "message": str(err), "since": time.time(),
        }), encoding="utf-8")


def blocked(now: float | None = None) -> SearchQuotaExceeded | None:
    """Чинний запобіжник — виняток, яким відмовити пошуку, або None."""
    try:
        data = json.loads(_block_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    reset_at = _num(data.get("reset_at"))
    now = time.time() if now is None else now
    if reset_at <= now:
        return None
    return SearchQuotaExceeded(
        str(data.get("message") or "квота пошуку Vast вичерпана"),
        limit=int(_num(data.get("limit"))), remaining=int(_num(data.get("remaining"))),
        retry_after=reset_at - now, reset_at=reset_at,
    )


def reset_label(reset_at: float) -> str:
    """Час скидання квоти за МІСЦЕВИМ годинником — те, що людина звіряє з годинником."""
    return datetime.fromtimestamp(reset_at).strftime("%H:%M")


# ---- журнал ------------------------------------------------------------------


def _audit_path() -> Path:
    return _dir() / "vast_search_audit.jsonl"


def audit(*, limit: int, rows: int, status: str, cached: bool = False,
          remaining: int | None = None) -> None:
    """Рядок журналу пошуку. Жодних заголовків і ключів — лише облік рядків."""
    rec: dict[str, Any] = {
        "ts": round(time.time(), 3), "pid": os.getpid(),
        "owner": os.environ.get("GPURUNNER_OWNER", ""),
        "purpose": _purpose.get() or "search",
        "limit": int(limit), "rows": int(rows), "status": status, "cached": cached,
    }
    if remaining is not None:
        rec["remaining"] = remaining
    path = _audit_path()
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def usage(since: float) -> dict[str, Any]:
    """Скільки рядків ринку взято з `since` (епоха) — з журналу, по призначенню."""
    out: dict[str, Any] = {"rows": 0, "requests": 0, "cached": 0, "quota_hits": 0,
                           "by_purpose": {}}
    try:
        lines = _audit_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if _num(rec.get("ts")) < since:
            continue
        if rec.get("cached"):
            out["cached"] += 1
            continue
        if rec.get("status") == "quota":
            out["quota_hits"] += 1
        out["requests"] += 1
        out["rows"] += int(_num(rec.get("rows")))
        p = str(rec.get("purpose") or "search")
        out["by_purpose"][p] = out["by_purpose"].get(p, 0) + int(_num(rec.get("rows")))
    return out


# ---- кеш видачі --------------------------------------------------------------


def _ttl() -> float:
    raw = os.environ.get("GPURUNNER_MARKET_CACHE_SEC")
    if raw is None or raw == "":
        return CACHE_TTL_SEC
    return max(0.0, _num(raw))


def _cache_dir() -> Path:
    return _dir() / "vast_market_cache"


def _key(query: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()[:32]


def cached(query: dict[str, Any], now: float | None = None) -> list[dict[str, Any]] | None:
    """Свіжа видача на ТОЧНО такий самий запит — або None."""
    ttl = _ttl()
    if ttl <= 0:
        return None
    try:
        data = json.loads((_cache_dir() / f"{_key(query)}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    now = time.time() if now is None else now
    if now - _num(data.get("ts")) > ttl:
        return None
    offers = data.get("offers")
    return list(offers) if isinstance(offers, list) else None


def store(query: dict[str, Any], offers: list[dict[str, Any]]) -> None:
    if _ttl() <= 0:
        return
    d = _cache_dir()
    with contextlib.suppress(OSError):
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{_key(query)}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"ts": time.time(), "offers": offers}), encoding="utf-8")
        os.replace(tmp, d / f"{_key(query)}.json")


def invalidate() -> None:
    """Скинути кеш: оренду взято чи відхилено — видача вже не та."""
    d = _cache_dir()
    if not d.is_dir():
        return
    for f in d.glob("*.json"):
        with contextlib.suppress(OSError):
            f.unlink()
