"""Квоти, які провайдер не віддає: автопідрахунок + ручна корекція.

Kaggle тримає свої 30 GPU-год/тиждень тільки у вебі, Colab — compute units тільки
в UI, Saturn узагалі не публікує ліміту. Для них залишок доводиться рахувати
самим: ``ліміт − витрачено``, де «витрачено» бере ``core/usage.py`` з журналу
прогонів.

Цей підрахунок систематично **занижує витрати**: він бачить лише те, що подали
через gpurunner, а сесія, відкрита руками в браузері, для нас не існує. Тому тут
є другий механізм — **якір**: користувач звіряє залишок із вебом провайдера і
вписує його. Якір стає новою точкою відліку, і все, що gpurunner запустив після
нього, віднімається далі автоматично:

    є якір у поточному періоді:  залишок = якір − витрати ПІСЛЯ якоря
    немає:                       залишок = ліміт − витрати від початку періоду

Якір із минулого періоду ігнорується: на скиданні квоти все повертається до ліміту.

Стан живе в окремій ``quota.sqlite3`` — той самий патерн, що й ``core/budget.py``
(кожна турбота володіє своєю БД, WAL, ``busy_timeout``). Причина та сама, що й у
manifest: між вкладкою дашборду й паралельним ``gpurunner run`` немає жодної
синхронізації, а JSON-документ тут дав би ті самі загублені записи.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gpurunner.config import data_dir
from gpurunner.core import gpu_equiv, usage
from gpurunner.core.usage import RunUsage

PERIOD_WEEK = "week"
PERIOD_MONTH = "month"
PERIOD_NONE = "none"
PERIODS = (PERIOD_WEEK, PERIOD_MONTH, PERIOD_NONE)

_SCHEMA = """
-- «Станом на ts лишалось remaining» — точка відліку, вписана руками.
-- Рядки не перезаписуються: історія звірянь сама по собі корисна, а арифметика
-- і так дивиться лише на найсвіжіший якір у поточному періоді.
CREATE TABLE IF NOT EXISTS quota_anchor (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    backend   TEXT NOT NULL,
    ts        TEXT NOT NULL,
    remaining REAL NOT NULL,
    note      TEXT
);
CREATE INDEX IF NOT EXISTS quota_anchor_backend ON quota_anchor (backend, ts);

-- Перекриття дефолтів: ліміт, період, день скидання. Провайдери свої правила
-- міняють без попередження, і це має бути один POST, а не реліз.
CREATE TABLE IF NOT EXISTS quota_config (
    backend       TEXT PRIMARY KEY,
    allowance     REAL,
    unit          TEXT,
    period        TEXT,
    reset_weekday INTEGER,
    plan          TEXT,
    updated_at    TEXT
);

-- Множники «у скільки разів карта швидша за T4», вписані руками. Потрібні там,
-- де паспортний проксі ``core/gpu_equiv.py`` свідомо мовчить — насамперед для
-- GeForce, чиї TFLOPS не описують реальної швидкості трену.
CREATE TABLE IF NOT EXISTS gpu_factor (
    gpu        TEXT PRIMARY KEY,
    factor     REAL NOT NULL,
    note       TEXT,
    updated_at TEXT
);

-- Знімок балансу на момент оновлення — з них малюється графік «як танув залишок».
CREATE TABLE IF NOT EXISTS balance_snapshot (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    backend   TEXT NOT NULL,
    ts        TEXT NOT NULL,
    available REAL,
    unit      TEXT,
    t4_hours  REAL
);
CREATE INDEX IF NOT EXISTS balance_snapshot_backend ON balance_snapshot (backend, ts);
"""


def db_path() -> Path:
    return data_dir() / "quota.sqlite3"


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.executescript(_SCHEMA)
        _migrate(conn)
        yield conn
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Добудувати колонки, яких немає у вже створеній БД.

    ``CREATE TABLE IF NOT EXISTS`` мовчки нічого не робить, якщо таблиця вже є,
    тож нове поле у схемі не доїхало б до тих, хто відкривав дашборд раніше, і
    падало б на першому ж SELECT.
    """
    have = {row[1] for row in conn.execute("PRAGMA table_info(quota_config)")}
    if "plan" not in have:
        conn.execute("ALTER TABLE quota_config ADD COLUMN plan TEXT")


# ---- конфігурація ----------------------------------------------------------


#: Як провайдер списує квоту. Різниця принципова: годину на ``T4x2`` Kaggle
#: списує як ОДНУ годину сесії (картки йому байдужі), а Colab за ту саму годину
#: зніме вдвічі більше units, бо тарифікує обчислення.
CHARGE_SESSION = "session"      # 1 год сесії = 1 одиниця, незалежно від карти
CHARGE_T4_UNITS = "t4_units"    # одиниці = T4-години × rate
CHARGE_HOURS = "hours"          # просто години роботи інстансу


@dataclass(frozen=True)
class QuotaConfig:
    """Правила квоти одного бекенда."""

    backend: str
    allowance: float | None      # ліміт на період; None = провайдер його не публікує
    unit: str
    period: str
    reset_weekday: int = 5       # 0 = понеділок … 5 = субота; лише для period="week"
    assumed: bool = False        # ліміт — наше припущення, а не факт від провайдера
    charge: str = CHARGE_SESSION
    rate: float = 1.0            # одиниць за одну T4-годину; лише для CHARGE_T4_UNITS
    plan: str = ""               # обраний тариф, якщо в бекенда їх кілька
    note: str = ""


#: Тарифні плани, між якими правила квоти міняються не кількісно, а якісно.
#:
#: Colab — єдиний такий випадок і водночас найпідступніший: **compute units
#: існують тільки на Pro / Pay-as-you-go**. На безкоштовному тарифі їх немає
#: взагалі, доступ дається за залишковим принципом, і Google ніде не публікує
#: ні годин, ні одиниць. Тому «free» — це не «менший ліміт», а *інша одиниця
#: виміру і відсутність ліміту як такого*.
PLANS: dict[str, dict[str, QuotaConfig]] = {
    "colab": {
        "free": QuotaConfig(
            backend="colab", allowance=None, unit="год", period=PERIOD_MONTH,
            charge=CHARGE_HOURS, plan="free",
            note="безкоштовний тариф: compute units не нараховуються взагалі, а "
                 "ліміту в годинах Google не публікує — рахуємо лише витрачене. "
                 "Маєш Pro? Переключи план, і з'явиться ліміт в units",
        ),
        "pro": QuotaConfig(
            backend="colab", allowance=100.0, unit="units", period=PERIOD_MONTH,
            assumed=True, charge=CHARGE_T4_UNITS, rate=100.0 / 57.0, plan="pro",
            note="тариф Pro: ≈100 units/міс ≈ 57 год T4 — звідси курс 1.75 units за "
                 "T4-годину. На Pro+ ліміт інший — постав своє число",
        ),
    },
}

#: Дефолти. ``assumed=True`` означає: число взяте з плану/документації, а не з API,
#: і користувач має право його виправити — воно й показується в UI як припущення.
DEFAULTS: dict[str, QuotaConfig] = {
    "kaggle": QuotaConfig(
        backend="kaggle", allowance=30.0, unit="год", period=PERIOD_WEEK,
        reset_weekday=5, assumed=True, charge=CHARGE_SESSION,
        note="30 GPU-год/тиждень; тарифікується сесія, тож T4x2 списує стільки ж, "
             "скільки T4. День скидання через API не дізнатись — дефолт «субота "
             "00:00 UTC», поправ, якщо у веб-квоті інакше",
    ),
    # Дефолт — безкоштовний тариф, і це не обережність, а факт: compute units
    # існують лише на Pro / Pay-as-you-go. Приписати безкоштовному акаунту
    # «100 units/міс» означало б додати в підсумок ~57 T4-годин, яких немає.
    "colab": PLANS["colab"]["free"],
    "saturn": QuotaConfig(
        backend="saturn", allowance=None, unit="год", period=PERIOD_MONTH,
        charge=CHARGE_HOURS,
        note="Saturn місячного ліміту не публікує ніде — впиши свій, якщо знаєш",
    ),
    # Нижче — бекенди з робочим API балансу. Ліміт тут не потрібен, але період
    # лишається: за ним рахується «витрачено» і дата наступного скидання.
    "modal": QuotaConfig(backend="modal", allowance=None, unit="$", period=PERIOD_MONTH,
                         charge=CHARGE_HOURS,
                         note="залишок рахує сам бекенд ($30/міс мінус витрати)"),
    "beam": QuotaConfig(backend="beam", allowance=None, unit="$", period=PERIOD_MONTH,
                        charge=CHARGE_HOURS,
                        note="залишок веде локальний леджер core/budget.py"),
    "lightning": QuotaConfig(backend="lightning", allowance=None, unit="credits",
                             period=PERIOD_MONTH, charge=CHARGE_HOURS,
                             note="залишок віддає API teamspace"),
    "vast": QuotaConfig(backend="vast", allowance=None, unit="$", period=PERIOD_NONE,
                        charge=CHARGE_HOURS, note="передоплата, не скидається"),
}


def plans(backend: str) -> list[str]:
    """Назви тарифних планів бекенда; порожньо — планів немає."""
    return list(PLANS.get(backend, {}))


def get_config(backend: str) -> QuotaConfig:
    """Дефолт бекенда (з урахуванням обраного плану), перекритий правками."""
    fallback = QuotaConfig(backend=backend, allowance=None, unit="", period=PERIOD_MONTH)
    with _db() as conn:
        row = conn.execute(
            "SELECT allowance, unit, period, reset_weekday, plan FROM quota_config"
            " WHERE backend = ?",
            (backend,),
        ).fetchone()
    plan = row[4] if row else None
    base = PLANS.get(backend, {}).get(plan or "", DEFAULTS.get(backend, fallback))
    if row is None:
        return base
    allowance, unit, period, weekday, _ = row
    return QuotaConfig(
        backend=backend,
        allowance=base.allowance if allowance is None else float(allowance),
        unit=unit or base.unit,
        period=period or base.period,
        reset_weekday=base.reset_weekday if weekday is None else int(weekday),
        # Щойно користувач вписав ліміт своїми руками, це вже не наше припущення.
        assumed=base.assumed and allowance is None,
        charge=base.charge,
        rate=base.rate,
        plan=plan or base.plan,
        note=base.note,
    )


def set_config(
    backend: str,
    *,
    allowance: float | None = None,
    unit: str | None = None,
    period: str | None = None,
    reset_weekday: int | None = None,
    plan: str | None = None,
) -> QuotaConfig:
    """Перекрити правила квоти. Передані ``None`` лишають поточне значення."""
    if period is not None and period not in PERIODS:
        raise ValueError(f"period має бути одним з {PERIODS}, а не {period!r}")
    if reset_weekday is not None and not 0 <= reset_weekday <= 6:
        raise ValueError("reset_weekday — 0 (понеділок) … 6 (неділя)")
    if allowance is not None and allowance < 0:
        raise ValueError("allowance не може бути від'ємним")
    if plan is not None and plan not in PLANS.get(backend, {}):
        raise ValueError(
            f"у бекенда {backend!r} немає плану {plan!r}; доступні: {plans(backend) or '—'}"
        )
    current = get_config(backend)
    # Зміна плану скидає ручні перекриття: у нового плану інша одиниця виміру,
    # і залишити від старого «100» з units поверх годин означало б зліпити
    # число з однієї шкали з підписом з іншої.
    reset_overrides = plan is not None and plan != current.plan
    with _db() as conn:
        conn.execute(
            "INSERT INTO quota_config"
            " (backend, allowance, unit, period, reset_weekday, plan, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(backend) DO UPDATE SET"
            "   allowance = excluded.allowance, unit = excluded.unit,"
            "   period = excluded.period, reset_weekday = excluded.reset_weekday,"
            "   plan = excluded.plan, updated_at = excluded.updated_at",
            (
                backend,
                None if reset_overrides and allowance is None
                else (current.allowance if allowance is None else float(allowance)),
                None if reset_overrides and unit is None else (current.unit if unit is None else unit),
                current.period if period is None else period,
                current.reset_weekday if reset_weekday is None else int(reset_weekday),
                current.plan if plan is None else plan,
                datetime.now(tz=UTC).isoformat(),
            ),
        )
    return get_config(backend)


# ---- ручні множники карт ---------------------------------------------------


def gpu_factors() -> dict[str, float]:
    """Множники до T4, вписані користувачем. Ключ — канонічне ім'я карти."""
    with _db() as conn:
        rows = conn.execute("SELECT gpu, factor FROM gpu_factor").fetchall()
    return {gpu_equiv.normalize(gpu)[0]: float(factor) for gpu, factor in rows}


def set_gpu_factor(gpu: str, factor: float | None, *, note: str = "") -> dict[str, float]:
    """Вписати (або прибрати, передавши ``None``) множник карти до T4."""
    name, _ = gpu_equiv.normalize(gpu)
    if not name:
        raise ValueError("порожня назва карти")
    with _db() as conn:
        if factor is None:
            conn.execute("DELETE FROM gpu_factor WHERE gpu = ?", (name,))
        else:
            if factor <= 0:
                raise ValueError("множник має бути додатним")
            conn.execute(
                "INSERT INTO gpu_factor (gpu, factor, note, updated_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(gpu) DO UPDATE SET factor = excluded.factor,"
                "   note = excluded.note, updated_at = excluded.updated_at",
                (name, float(factor), note or None, datetime.now(tz=UTC).isoformat()),
            )
    return gpu_factors()


def equivalence(gpu: str) -> gpu_equiv.Equivalence | None:
    """``gpu_equiv.equivalence`` з урахуванням ручних множників."""
    return gpu_equiv.equivalence(gpu, gpu_factors())


# ---- межі періоду ----------------------------------------------------------


def period_bounds(cfg: QuotaConfig, now: datetime | None = None) -> tuple[datetime, datetime]:
    """``[початок, кінець)`` поточного періоду квоти."""
    now = now or datetime.now(tz=UTC)
    if cfg.period == PERIOD_NONE:
        # Передоплата не скидається: «період» — уся історія.
        return datetime(1970, 1, 1, tzinfo=UTC), now + timedelta(days=365 * 100)
    if cfg.period == PERIOD_WEEK:
        midnight = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        back = (midnight.weekday() - cfg.reset_weekday) % 7
        start = midnight - timedelta(days=back)
        return start, start + timedelta(days=7)
    start = now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = (start + timedelta(days=32)).replace(day=1)
    return start, end


def next_reset(cfg: QuotaConfig, now: datetime | None = None) -> datetime | None:
    """Коли квота відновиться. ``None`` — не скидається взагалі."""
    if cfg.period == PERIOD_NONE:
        return None
    return period_bounds(cfg, now)[1]


# ---- якорі -----------------------------------------------------------------


def set_anchor(backend: str, remaining: float, *, note: str = "",
               ts: datetime | None = None) -> dict[str, Any]:
    """Зафіксувати звірений залишок. Стає новою точкою відліку."""
    if remaining < 0:
        raise ValueError("залишок не може бути від'ємним")
    stamp = (ts or datetime.now(tz=UTC)).isoformat()
    with _db() as conn:
        conn.execute(
            "INSERT INTO quota_anchor (backend, ts, remaining, note) VALUES (?, ?, ?, ?)",
            (backend, stamp, float(remaining), note or None),
        )
    return {"backend": backend, "ts": stamp, "remaining": float(remaining), "note": note or None}


def current_anchor(backend: str, since: datetime) -> dict[str, Any] | None:
    """Найсвіжіший якір, поставлений не раніше ``since`` (початку періоду)."""
    with _db() as conn:
        row = conn.execute(
            "SELECT ts, remaining, note FROM quota_anchor"
            " WHERE backend = ? AND ts >= ? ORDER BY ts DESC, seq DESC LIMIT 1",
            (backend, since.isoformat()),
        ).fetchone()
    if row is None:
        return None
    return {"ts": row[0], "remaining": float(row[1]), "note": row[2]}


def clear_anchor(backend: str, since: datetime) -> int:
    """Прибрати якорі поточного періоду — повернутись до чистого автопідрахунку.

    Старіші рядки лишаються: арифметика їх і так не бачить, а історія звірянь —
    єдиний спосіб потім зрозуміти, наскільки автопідрахунок розходився з вебом.
    """
    with _db() as conn:
        cur = conn.execute(
            "DELETE FROM quota_anchor WHERE backend = ? AND ts >= ?",
            (backend, since.isoformat()),
        )
    return int(cur.rowcount or 0)


def anchor_history(backend: str, limit: int = 20) -> list[dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT ts, remaining, note FROM quota_anchor WHERE backend = ?"
            " ORDER BY ts DESC, seq DESC LIMIT ?",
            (backend, int(limit)),
        ).fetchall()
    return [{"ts": ts, "remaining": float(rem), "note": note} for ts, rem, note in rows]


# ---- знімки балансу --------------------------------------------------------


def add_snapshot(backend: str, available: float | None, unit: str,
                 t4_hours: float | None, *, ts: datetime | None = None) -> None:
    with _db() as conn:
        conn.execute(
            "INSERT INTO balance_snapshot (backend, ts, available, unit, t4_hours)"
            " VALUES (?, ?, ?, ?, ?)",
            ((backend), (ts or datetime.now(tz=UTC)).isoformat(),
             None if available is None else float(available), unit,
             None if t4_hours is None else float(t4_hours)),
        )


def snapshots(backend: str, *, days: int = 30) -> list[dict[str, Any]]:
    since = (datetime.now(tz=UTC) - timedelta(days=days)).isoformat()
    with _db() as conn:
        rows = conn.execute(
            "SELECT ts, available, unit, t4_hours FROM balance_snapshot"
            " WHERE backend = ? AND ts >= ? ORDER BY ts",
            (backend, since),
        ).fetchall()
    return [{"ts": ts, "available": av, "unit": unit, "t4_hours": t4} for ts, av, unit, t4 in rows]


def prune_snapshots(*, keep_days: int = 180) -> int:
    cutoff = (datetime.now(tz=UTC) - timedelta(days=keep_days)).isoformat()
    with _db() as conn:
        cur = conn.execute("DELETE FROM balance_snapshot WHERE ts < ?", (cutoff,))
    return int(cur.rowcount or 0)


# ---- власне підрахунок -----------------------------------------------------


@dataclass(frozen=True)
class RunCharge:
    """Прогін і скільки одиниць квоти він списав."""

    run: RunUsage
    charged: float
    converted: bool     # False = карту не вдалось перевести, списання не пораховане
    basis: str = ""     # "вимір" / "паспорт" — для CHARGE_T4_UNITS

    def as_dict(self) -> dict[str, Any]:
        return {
            "handle_id": self.run.handle_id[:8],
            "job": self.run.job_name,
            "gpu": self.run.gpu,
            "status": self.run.status,
            "hours": round(self.run.hours, 3),
            "source": self.run.source,
            "exact": self.run.exact,
            "charged": round(self.charged, 3),
            "converted": self.converted,
            "basis": self.basis,
            "started": self.run.started.isoformat() if self.run.started else None,
        }


def charge_run(run: RunUsage, cfg: QuotaConfig,
               overrides: dict[str, float] | None = None) -> RunCharge:
    """Скільки одиниць квоти зняв один прогін — за правилами цього провайдера."""
    if cfg.charge == CHARGE_SESSION:
        # Kaggle рахує години сесії: карта й їх кількість не впливають ні на що.
        return RunCharge(run=run, charged=run.hours, converted=True)
    if cfg.charge == CHARGE_T4_UNITS:
        converted = gpu_equiv.t4_hours(run.gpu, run.hours, overrides=overrides)
        if converted is None:
            # Карти немає в таблиці коефіцієнтів. Списати «нуль» означало б
            # збрехати в бік більшого залишку — тож рахуємо окремо і кажемо вголос.
            return RunCharge(run=run, charged=0.0, converted=False)
        t4h, basis = converted
        return RunCharge(run=run, charged=t4h * cfg.rate, converted=True, basis=basis)
    return RunCharge(run=run, charged=run.hours, converted=True)


@dataclass
class QuotaState:
    """Усе, що дашборд знає про квоту одного бекенда."""

    backend: str
    unit: str
    period: str
    plan: str
    plans: list[str]
    rate: float
    period_start: datetime
    next_reset: datetime | None
    allowance: float | None
    allowance_assumed: bool
    anchor: dict[str, Any] | None
    counted_since: datetime
    used: float
    remaining: float | None
    charges: list[RunCharge] = field(default_factory=list)
    note: str = ""

    @property
    def runs(self) -> list[RunUsage]:
        return [c.run for c in self.charges]

    @property
    def exact_used(self) -> bool:
        """Чи всі прогони у витраті мають надійне джерело часу й переводяться."""
        return all(c.run.exact and c.converted for c in self.charges)

    @property
    def unconverted(self) -> list[str]:
        """Карти, чиї години не вдалось перевести в одиниці квоти."""
        return sorted({c.run.gpu for c in self.charges if not c.converted})

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "unit": self.unit,
            "period": self.period,
            "plan": self.plan,
            "plans": self.plans,
            "period_start": self.period_start.isoformat(),
            "next_reset": self.next_reset.isoformat() if self.next_reset else None,
            "allowance": self.allowance,
            "allowance_assumed": self.allowance_assumed,
            "anchor": self.anchor,
            "counted_since": self.counted_since.isoformat(),
            "used": round(self.used, 3),
            "remaining": None if self.remaining is None else round(self.remaining, 3),
            "runs_counted": len(self.charges),
            "exact_used": self.exact_used,
            "unconverted": self.unconverted,
            "note": self.note,
        }


def compute(backend: str, *, now: datetime | None = None,
            handles: list[Any] | None = None) -> QuotaState:
    """Порахувати залишок бекенда за правилом «якір або ліміт, мінус витрати».

    ``remaining is None`` означає «ліміту немає з чого віднімати» — або провайдер
    його не публікує, або залишок і так віддає API. Це НЕ нуль і не помилка.
    """
    now = now or datetime.now(tz=UTC)
    cfg = get_config(backend)
    start, _ = period_bounds(cfg, now)
    anchor = current_anchor(backend, start)

    counted_since = datetime.fromisoformat(anchor["ts"]) if anchor else start
    if counted_since.tzinfo is None:
        counted_since = counted_since.replace(tzinfo=UTC)

    runs = usage.usage_in_window(backend, counted_since, now, handles=handles, now=now)
    overrides = gpu_factors() if cfg.charge == CHARGE_T4_UNITS else None
    charges = [charge_run(r, cfg, overrides) for r in runs]
    used = round(sum(c.charged for c in charges), 3)

    base = anchor["remaining"] if anchor else cfg.allowance
    remaining = None if base is None else base - used

    return QuotaState(
        backend=backend,
        unit=cfg.unit,
        period=cfg.period,
        plan=cfg.plan,
        plans=plans(backend),
        rate=cfg.rate,
        period_start=start,
        next_reset=next_reset(cfg, now),
        allowance=cfg.allowance,
        allowance_assumed=cfg.assumed,
        anchor=anchor,
        counted_since=counted_since,
        used=used,
        remaining=remaining,
        charges=charges,
        note=cfg.note,
    )
