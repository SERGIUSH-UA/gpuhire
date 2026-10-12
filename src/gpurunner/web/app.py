"""FastAPI за дашбордом.

Сторінка статична й тягне все через ``/api/*``; сервер лише збирає дані. Три
рішення, які тут важать більше за решту коду:

**Мережа — на вимогу.** ``balance()`` кожного бекенда — це синхронний виклик
чужого SDK, іноді на десятки секунд. Сім таких при кожному відкритті сторінки
зробили б дашборд повільнішим за ``gpurunner balance``, заради якого його й
писали. Тому є кеш на 5 хвилин, а сторінка показує вік даних і кнопку оновлення.

**Кожен бекенд — у своєму треді, з таймаутом.** Інакше один провайдер, що
підвис на TCP-з'єднанні, вішає всю сторінку.

**Треди — демони, і це не оптимізація.** ``asyncio.to_thread`` кладе роботу в
``ThreadPoolExecutor``, чиї треди не-демони: підвислий SDK не дасть процесу
завершитись після Ctrl+C, і на Windows це виглядає як намертво зависла консоль
(див. CLAUDE.md). Тому блокуючі виклики йдуть у власні daemon-треди.

Слухає тільки 127.0.0.1 і не має авторизації — сторінка показує баланси й імена
прогонів, назовні цього віддавати не можна.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from gpurunner.backends import BACKEND_NAMES
from gpurunner.core import balances, manifest, quota, usage
from gpurunner.core.backend import AuthError

STATIC_DIR = Path(__file__).parent / "static"

#: Скільки кеш вважається свіжим. П'ять хвилин — компроміс: баланс за цей час
#: помітно не змінюється, а API-ліміти провайдерів не палляться перезавантаженнями.
CACHE_TTL_S = 300.0

#: Стеля на один бекенд. Vast та Lightning живими доходили до ~15 с; 25 дає запас
#: і при цьому не дає одному мертвому провайдеру тримати сторінку хвилину.
BACKEND_TIMEOUT_S = 25.0

# TypeVar, а не `def f[T]`: синтаксис PEP 695 з 3.12, а пакет ставиться з 3.11.
T = TypeVar("T")


async def run_blocking(fn: Callable[[], T], *, timeout: float) -> T:
    """Виконати синхронний виклик у daemon-треді з таймаутом.

    Свідомо не ``asyncio.to_thread``: його executor тримає не-демон-треди, які
    інтерпретатор join'ить на виході. Один підвислий SDK — і процес не вмирає
    після Ctrl+C. Тут тред кинутий напризволяще: він або відповість, або згине
    разом із процесом.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()

    def _settle(setter: Callable[[Any], None], value: Any) -> None:
        if not future.done():
            setter(value)

    def target() -> None:
        try:
            result = fn()
        except BaseException as exc:
            loop.call_soon_threadsafe(_settle, future.set_exception, exc)
        else:
            loop.call_soon_threadsafe(_settle, future.set_result, result)

    threading.Thread(target=target, daemon=True, name="gpurunner-dash").start()
    return await asyncio.wait_for(future, timeout)


@dataclass
class _Cache:
    """Останній зібраний огляд плюс замок, щоб не збирати його вп'ятьох одразу."""

    payload: dict[str, Any] | None = None
    fetched_at: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def age(self, loop_time: float) -> float | None:
        return None if self.payload is None else loop_time - self.fetched_at

    def fresh(self, loop_time: float) -> bool:
        age = self.age(loop_time)
        return age is not None and age < CACHE_TTL_S


def _row(backend: str, detail: str) -> dict[str, Any]:
    return {"backend": backend, "available": None, "unit": "", "spent": None,
            "detail": detail, "url": ""}


async def _gather_overview(names: list[str]) -> dict[str, Any]:
    """Опитати бекенди паралельно й зібрати картки."""
    now = datetime.now(tz=UTC)

    async def one(name: str) -> dict[str, Any]:
        try:
            rows = await run_blocking(
                lambda: balances.collect_reports([name]), timeout=BACKEND_TIMEOUT_S
            )
        except TimeoutError:
            return _row(name, f"помилка запиту: не відповів за {BACKEND_TIMEOUT_S:.0f} с")
        except AuthError as e:
            # ``collect_reports`` це вже ловить сам, але вимикати тут захист не
            # можна: бекенд без креденшлів — це не «зламався», і плутати ці два
            # стани означає слати користувача чинити те, що просто не налаштоване.
            return _row(name, f"не налаштовано: {str(e).splitlines()[0]}")
        except Exception as e:
            return _row(name, f"помилка запиту: {str(e).splitlines()[0][:90]}")
        return rows[0]

    rows = await asyncio.gather(*(one(n) for n in names))

    def build() -> balances.Overview:
        handles = manifest.load()
        overrides = quota.gpu_factors()
        views = [balances.build_view(r, overrides, now=now, handles=handles) for r in rows]
        result = balances.Overview(generated_at=now, services=views)
        for view in views:
            if view.available is not None:
                balances.record_snapshot(view, ts=now)
        return result

    # Побудова картки читає manifest, журнал і теки виводу — це диск, не мережа,
    # але на кількох сотнях прогонів усе одно варте окремого треда.
    view_obj = await run_blocking(build, timeout=BACKEND_TIMEOUT_S * 2)
    return view_obj.as_dict()


def create_app() -> FastAPI:
    cache = _Cache()
    app = FastAPI(title="gpurunner dashboard", docs_url=None, redoc_url=None)

    async def overview(force: bool = False) -> dict[str, Any]:
        loop_time = asyncio.get_running_loop().time()
        if not force and cache.fresh(loop_time) and cache.payload is not None:
            return {**cache.payload, "cache_age_s": round(cache.age(loop_time) or 0.0, 1),
                    "cached": True}
        async with cache.lock:
            loop_time = asyncio.get_running_loop().time()
            # Поки чекали на замок, сусідній запит міг усе зібрати.
            if not force and cache.fresh(loop_time) and cache.payload is not None:
                return {**cache.payload, "cache_age_s": round(cache.age(loop_time) or 0.0, 1),
                        "cached": True}
            payload = await _gather_overview(list(BACKEND_NAMES))
            cache.payload = payload
            cache.fetched_at = asyncio.get_running_loop().time()
        return {**payload, "cache_age_s": 0.0, "cached": False}

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/overview")
    async def api_overview() -> dict[str, Any]:
        return await overview()

    @app.post("/api/refresh")
    async def api_refresh() -> dict[str, Any]:
        return await overview(force=True)

    @app.get("/api/runs")
    async def api_runs(limit: int = 25) -> dict[str, Any]:
        def build() -> dict[str, Any]:
            handles = manifest.load()
            journal = manifest.events_bulk([h.id for h in handles])
            now = datetime.now(tz=UTC)
            active = [
                usage.run_usage(h, now=now, events=journal.get(h.id, []))
                for h in usage.active_runs(handles=handles)
            ]
            recent = sorted(handles, key=lambda h: h.created_at, reverse=True)[:limit]
            # Нетермінальний статус ще не означає «виконується». У manifest
            # накопичуються хендли, які лишились у ``queued`` назавжди: подали й
            # більше не опитували. Показувати їх як активні («іде 1658 год»)
            # означає ховати справді живі прогони серед мерців, тому вони
            # їдуть окремим списком.
            def row(r: usage.RunUsage) -> dict[str, Any]:
                return {"handle_id": r.handle_id[:8], "backend": r.backend, "job": r.job_name,
                        "gpu": r.gpu, "status": r.status, "hours": round(r.hours, 2),
                        "stale": r.stale,
                        "last_seen": r.ended.isoformat() if r.stale and r.ended else None,
                        "started": r.started.isoformat() if r.started else None}

            by_hours = sorted(active, key=lambda r: r.hours, reverse=True)
            return {
                "active": [row(r) for r in by_hours if not r.stale],
                "lost": [row(r) for r in by_hours if r.stale],
                "recent": [
                    {"handle_id": h.id[:8], "backend": h.backend, "job": h.job_name,
                     "gpu": h.gpu,
                     "status": str(getattr(h.status, "value", h.status)),
                     "created": h.created_at.isoformat(),
                     "error": (h.error or "").replace("\n", " ")[:120] or None}
                    for h in recent
                ],
            }

        return await run_blocking(build, timeout=BACKEND_TIMEOUT_S)

    @app.get("/api/usage/{backend}")
    async def api_usage(backend: str) -> dict[str, Any]:
        _known(backend)

        def build() -> dict[str, Any]:
            state = quota.compute(backend)
            return {
                **state.as_dict(),
                "rows": [c.as_dict() for c in
                         sorted(state.charges, key=lambda c: c.run.started or datetime.min.replace(tzinfo=UTC),
                                reverse=True)],
                "anchors": quota.anchor_history(backend),
            }

        return await run_blocking(build, timeout=BACKEND_TIMEOUT_S)

    @app.get("/api/history/{backend}")
    async def api_history(backend: str, days: int = 30) -> dict[str, Any]:
        _known(backend)
        return {"backend": backend, "points": quota.snapshots(backend, days=days)}

    @app.post("/api/quota/{backend}")
    async def api_set_quota(backend: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        _known(backend)
        try:
            if any(body.get(k) is not None for k in
                   ("allowance", "period", "reset_weekday", "unit", "plan")):
                quota.set_config(
                    backend,
                    allowance=_opt_float(body.get("allowance")),
                    unit=body.get("unit"),
                    period=body.get("period"),
                    reset_weekday=_opt_int(body.get("reset_weekday")),
                    plan=body.get("plan") or None,
                )
            if body.get("remaining") is not None:
                quota.set_anchor(backend, float(body["remaining"]),
                                 note=str(body.get("note") or ""))
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        cache.payload = None      # наступний запит має побачити нову цифру одразу
        return quota.compute(backend).as_dict()

    @app.delete("/api/quota/{backend}/anchor")
    async def api_clear_anchor(backend: str) -> dict[str, Any]:
        _known(backend)
        cfg = quota.get_config(backend)
        start, _ = quota.period_bounds(cfg)
        removed = quota.clear_anchor(backend, start)
        cache.payload = None
        return {"removed": removed, **quota.compute(backend).as_dict()}

    @app.get("/api/factors")
    async def api_factors() -> dict[str, Any]:
        from gpurunner.core import gpu_equiv

        manual = quota.gpu_factors()
        known = {}
        for gpu in gpu_equiv.known_gpus(manual):
            eq = gpu_equiv.equivalence(gpu, manual)
            if eq:
                known[gpu] = {"factor": round(eq.factor, 3), "basis": eq.basis, "note": eq.note}
        return {"known": known, "manual": manual, "missing": gpu_equiv.needs_override()}

    @app.post("/api/factors/{gpu}")
    async def api_set_factor(gpu: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        try:
            quota.set_gpu_factor(gpu, _opt_float(body.get("factor")),
                                 note=str(body.get("note") or ""))
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        cache.payload = None
        return {"manual": quota.gpu_factors()}

    @app.exception_handler(Exception)
    async def _unhandled(_request: Any, exc: Exception) -> JSONResponse:
        # Дашборд читає чужі SDK і власні БД; будь-який виняток має стати
        # видимим рядком у сторінці, а не порожнім вікном без пояснень.
        return JSONResponse(status_code=500,
                            content={"error": f"{type(exc).__name__}: {exc}"})

    return app


def _known(backend: str) -> None:
    if backend not in BACKEND_NAMES:
        raise HTTPException(status_code=404, detail=f"невідомий бекенд: {backend}")


def _opt_float(value: Any) -> float | None:
    return None if value is None or value == "" else float(value)


def _opt_int(value: Any) -> int | None:
    return None if value is None or value == "" else int(value)
