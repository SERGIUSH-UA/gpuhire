"""Скільки GPU-годин уже витрачено — з того, що бачить локальний manifest.

Провайдери, у яких квота найважливіша, її не віддають: Kaggle тримає свої
30 год/тиждень тільки у вебі, Colab — compute units тільки в UI. Єдине джерело,
яке в нас є, — власний журнал прогонів. Цей модуль перетворює його на години.

**Головне обмеження, і воно не технічне, а принципове**: сюди потрапляють лише
прогони, подані через gpurunner. Сесія, відкрита руками в браузері Kaggle, для
нас не існує. Тому будь-яка цифра звідси — **нижня межа витрат**, а отже верхня
межа залишку. Саме тому в дашборді є ручна корекція: вона не аварійний люк, а
штатний спосіб звести автопідрахунок із реальністю.

Друге обмеження — точність самої тривалості. Вона береться з чотирьох джерел
різної якості, і кожен рядок несе позначку, звідки саме, щоб «18.4 год» можна
було не тільки прочитати, а й перевірити.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gpurunner.core import manifest
from gpurunner.core.models import JobHandle, JobStatus

#: Ключі, під якими ранери пишуть свій час у summary. Історично прижилися два.
_ELAPSED_KEYS = ("elapsed_s", "wall_sec")

#: Скільки json-файлів у теці виводу переглядати. Тека прогону може містити
#: десятки тисяч файлів (посторінковий OCR), і повний обхід зробив би відкриття
#: дашборду хвилинним.
_MAX_JSON_SCAN = 60

_TERMINAL = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}

# Джерела тривалості, від найточнішого до найгіршого.
SRC_SUMMARY = "summary"      # ранер сам записав свій час
SRC_JOURNAL = "журнал"       # running → термінальна подія
SRC_QUEUED = "з черги"       # старту не бачили: рахуємо від подання, тобто із чергою
SRC_RUNNING = "триває"       # ще не завершився, і ми бачили його живим щойно
SRC_STALE = "загублений"     # нетермінальний, але давно не спостерігався

#: Після якого мовчання нетермінальний хендл перестає вважатись живим.
#:
#: Це не косметика. У manifest накопичуються прогони, які лишились у ``queued``
#: назавжди: користувач подав їх, більше не опитував, а кернел давно помер. На
#: живих даних таких було 39 штук із травня-червня, і рахунок «від подання до
#: зараз» дав **1106 годин витрат за тиждень**, у якому їх фізично 168.
#:
#: Правильна межа — не «зараз», а ``updated_at``: останній момент, коли ми
#: справді бачили прогін. Далі йдуть уже не дані, а припущення. Шість годин —
#: щоб трен, який поллять хоча б двічі на добу, лишався точним.
STALE_AFTER = timedelta(hours=6)

#: Скільки максимум може тривати ОДИН прогін у провайдера, який обриває сесію
#: сам. Kaggle вбиває кернел на 12-й годині — довший прогін там неможливий, тож
#: будь-яке більше число означає зіпсовану мітку, а не витрату. Провайдери без
#: жорсткого ліміту сесії сюди не входять: вигадувати їм стелю нема з чого.
SESSION_CAP_H: dict[str, float] = {"kaggle": 12.0, "colab": 12.0}


@dataclass(frozen=True)
class RunUsage:
    """Один прогін і скільки годин він з'їв."""

    handle_id: str
    backend: str
    job_name: str
    gpu: str
    status: str
    hours: float                 # години, що потрапили у вікно підрахунку
    source: str
    started: datetime | None
    ended: datetime | None
    full_hours: float | None = None   # уся тривалість, якщо вікно її обрізало

    @property
    def exact(self) -> bool:
        """Чи можна на цю цифру спиратись без застережень."""
        return self.source in (SRC_SUMMARY, SRC_JOURNAL)

    @property
    def stale(self) -> bool:
        """Нетермінальний прогін, якого давно не бачили — швидше за все, мертвий."""
        return self.source == SRC_STALE

    @property
    def clipped(self) -> bool:
        """Чи прогін перетнув межу вікна (якір або початок періоду)."""
        return self.full_hours is not None and abs(self.full_hours - self.hours) > 1e-6


def _parse(ts: Any) -> datetime | None:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _elapsed_from_summary(output_dir: str | None) -> float | None:
    """Час, який ранер сам заміряв, у годинах. ``None`` — немає або не читається.

    Це найточніше джерело, але воно міряє **обчислення**, а не сесію: черга й
    підняття контейнера сюди не входять. Для квоти Kaggle, яка тарифікує сесію,
    цифра трохи занижена — ще одна причина, чому автопідрахунок є нижньою межею.
    """
    if not output_dir:
        return None
    root = Path(output_dir)
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    try:
        for pattern in ("*.json", "*/*.json"):
            for path in root.glob(pattern):
                candidates.append(path)
                if len(candidates) >= _MAX_JSON_SCAN:
                    break
            if len(candidates) >= _MAX_JSON_SCAN:
                break
    except OSError:
        return None
    # summary-файли спершу: у них час усього прогону, а в _meta.json — однієї частини
    candidates.sort(key=lambda p: (0 if "summary" in p.name else 1, p.name))
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for key in _ELAPSED_KEYS:
            value = payload.get(key)
            if isinstance(value, int | float) and value > 0:
                return float(value) / 3600.0
    return None


def run_usage(
    handle: JobHandle,
    *,
    now: datetime | None = None,
    events: list[dict[str, Any]] | None = None,
) -> RunUsage:
    """Тривалість одного прогону з позначкою, звідки вона взята.

    ``events`` можна передати вже прочитаними (``manifest.events_bulk``), щоб обхід
    усього manifest не відкривав по з'єднанню на прогін.
    """
    now = now or datetime.now(tz=UTC)
    if events is None:
        events = manifest.events(handle.id)
    started = next((_parse(e["ts"]) for e in events if e["status"] == JobStatus.RUNNING.value), None)
    ended = next((_parse(e["ts"]) for e in events if e["status"] in {s.value for s in _TERMINAL}), None)
    terminal = handle.status in _TERMINAL

    # Для незавершеного прогону межа — не «зараз», а останній момент, коли ми
    # його справді бачили. Далі йдуть уже не дані, а припущення (див. STALE_AFTER).
    last_seen = max(handle.updated_at, started or handle.created_at)
    alive = (now - handle.updated_at) < STALE_AFTER
    open_end, open_src = (now, SRC_RUNNING) if alive else (last_seen, SRC_STALE)

    hours = _elapsed_from_summary(handle.output_dir)
    if hours is not None:
        source = SRC_SUMMARY
    elif started is not None and ended is not None:
        hours, source = (ended - started).total_seconds() / 3600.0, SRC_JOURNAL
    elif started is not None:
        hours, source = (open_end - started).total_seconds() / 3600.0, open_src
    elif terminal and ended is not None:
        # Прогін жодного разу не опитали в стані RUNNING — журнал знає тільки
        # «подали» і «скінчилось». Рахуємо від подання, тобто разом із чергою:
        # це завищення, і воно позначене саме тому, що на нього не можна спиратись.
        hours, source = (ended - handle.created_at).total_seconds() / 3600.0, SRC_QUEUED
    elif terminal:
        hours, source = (handle.updated_at - handle.created_at).total_seconds() / 3600.0, SRC_QUEUED
    else:
        hours, source = (open_end - handle.created_at).total_seconds() / 3600.0, open_src

    cap = SESSION_CAP_H.get(handle.backend)
    if cap is not None and hours > cap:
        # Провайдер обриває сесію сам, тож більше — це зіпсована мітка, а не витрата.
        hours, source = cap, SRC_QUEUED

    return RunUsage(
        handle_id=handle.id,
        backend=handle.backend,
        job_name=handle.job_name,
        gpu=handle.gpu,
        status=str(getattr(handle.status, "value", handle.status)),
        hours=max(0.0, hours),
        source=source,
        started=started or handle.created_at,
        ended=ended if ended is not None else (None if alive else last_seen),
    )


def clip_to_window(run: RunUsage, start: datetime, end: datetime,
                   now: datetime) -> RunUsage | None:
    """Обрізати прогін до вікна ``[start, end)``. ``None`` — не перетинається.

    Ріжеться саме перетин, а не «включений/виключений за стартом». Причина —
    ручний якір: коли користувач звіряє залишок посеред 10-годинного трену, у
    його числі вже враховано те, що трен спалив ДО звіряння, і не враховано те,
    що спалить після. Викинути такий прогін цілком означало б недорахувати
    години, зарахувати цілком — порахувати частину двічі.

    Для прогонів із ``summary`` заміряний час не збігається з проміжком
    ``старт…кінець`` (черга й підняття контейнера в нього не входять), тому
    частка вікна застосовується пропорційно, а не як різниця міток.
    """
    started = run.started
    if started is None:
        return None
    ended = run.ended or now
    lo, hi = max(started, start), min(ended, end)
    if hi <= lo:
        # Виродження: миттєвий прогін точно на межі — рахуємо за стартом.
        return run if start <= started < end else None
    span = (ended - started).total_seconds()
    share = 1.0 if span <= 0 else (hi - lo).total_seconds() / span
    if share >= 0.999999:
        return run
    return replace(run, hours=run.hours * share, full_hours=run.hours)


def usage_in_window(
    backend: str,
    start: datetime,
    end: datetime,
    *,
    handles: list[JobHandle] | None = None,
    now: datetime | None = None,
) -> list[RunUsage]:
    """Прогони бекенда, обрізані до вікна ``[start, end)``."""
    now = now or datetime.now(tz=UTC)
    pool = [h for h in (handles if handles is not None else manifest.load()) if h.backend == backend]
    journal = manifest.events_bulk([h.id for h in pool])
    out = (run_usage(h, now=now, events=journal.get(h.id, [])) for h in pool)
    clipped = (clip_to_window(u, start, end, now) for u in out)
    return [u for u in clipped if u is not None and u.hours > 0]


def total_hours(rows: list[RunUsage]) -> float:
    return round(sum(r.hours for r in rows), 3)


def active_runs(*, handles: list[JobHandle] | None = None) -> list[JobHandle]:
    """Прогони, які ще не завершились — те, що показує панель «зараз крутиться»."""
    pool = handles if handles is not None else manifest.load()
    live = {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.UNKNOWN}
    return [h for h in pool if h.status in live]
