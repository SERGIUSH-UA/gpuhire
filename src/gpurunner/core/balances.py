"""Зведення балансів усіх бекендів в одну картину — і в одну одиницю.

Три речі, які поодинці вже вміє код, але разом ніде не збиралися:

1. **Опитування бекендів** із правильною обробкою збоїв (одна зламана інтеграція
   не має ховати таблицю) — жило всередині ``cli.balance`` і було доступне лише
   як друк у термінал.
2. **Локальний підрахунок квоти** (``core/quota.py``) для Kaggle/Colab/Saturn, де
   API залишку не віддає взагалі.
3. **Переведення в T4-години** (``core/gpu_equiv.py``), щоб «15 credits», «$13.05»
   і «18 год» можна було скласти.

Правило вибору головного числа: там, де провайдер віддає залишок сам, показуємо
його. Там, де ні (Kaggle, Colab, Saturn), головним стає локальний розрахунок, а
те, що API все-таки віддає, з'їжджає в деталі. Для Kaggle це принципово: його
``balance()`` повертає **AI-квоту в доларах**, і сплутати її з GPU-годинами —
рівно та помилка, від якої тут стоїть окремий бар'єр.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from gpurunner.backends import BACKEND_NAMES, get_backend
from gpurunner.core import gpu_equiv, quota
from gpurunner.core.backend import AuthError

#: Бекенди, де головне число — наш розрахунок, а не відповідь API.
QUOTA_IS_PRIMARY = ("kaggle", "colab", "saturn")

#: Курси, узяті з планів провайдерів, а не заміряні нами. Кожен — рівно те, що
#: провайдер сам пише про свій тариф; звідси й позначка «план» замість «вимір».
#: Lightning: 15 credits ≈ 22 год T4. Це курс **вартості кредита**, а не місячна
#: видача: щомісячні 15 кредитів скасовано з серпня 2026 (липень — останній
#: місяць), лишився разовий грант 25 кредитів за прив'язану картку.
_CREDITS_PER_T4_HOUR = 15.0 / 22.0
_COLAB_UNITS_PER_T4_HOUR = 100.0 / 57.0  # Colab Pro: ≈100 units/міс ≈ 57 год T4

#: Скільки офферів дивитись на ринку Vast, щоб дізнатись поточну ціну A100.
_VAST_OFFER_SAMPLE = 5

#: Скільки останніх замірів їде в картку на спарклайн.
_SPARK_POINTS = 24


@dataclass(frozen=True)
class T4Value:
    """Залишок, переведений у T4-години, і чесна довідка, звідки взявся курс."""

    hours: float
    basis: str          # "вимір" | "тариф" | "ринок" | "план" | "паспорт" | "вручну"
    explain: str

    @property
    def confident(self) -> bool:
        """● якщо спирається на вимір або на опубліковану ціну; ◐ якщо на оцінку."""
        return self.basis in ("вимір", "тариф", "ринок", "вручну")

    def as_dict(self) -> dict[str, Any]:
        return {"hours": round(self.hours, 2), "basis": self.basis,
                "confident": self.confident, "explain": self.explain}


@dataclass
class ServiceView:
    """Одна картка дашборду."""

    backend: str
    auth: str                      # "ok" | "не налаштовано" | "помилка"
    available: float | None
    unit: str
    source: str                    # "api" | "розрахунок" | "—"
    spent: float | None = None
    detail: str = ""
    url: str = ""
    t4: T4Value | None = None
    quota: quota.QuotaState | None = None
    api_note: str = ""             # що сказав API, коли головним є розрахунок
    spark: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "auth": self.auth,
            "available": None if self.available is None else round(self.available, 3),
            "unit": self.unit,
            "source": self.source,
            "spent": None if self.spent is None else round(self.spent, 3),
            "detail": self.detail,
            "url": self.url,
            "api_note": self.api_note,
            "t4": self.t4.as_dict() if self.t4 else None,
            "quota": self.quota.as_dict() if self.quota else None,
            "spark": self.spark,
        }


# ---- крок 1: опитати бекенди -----------------------------------------------


def collect_reports(names: list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """``BalanceReport`` кожного бекенда у вигляді рядків-словників.

    Форма рядка збережена від ``cli.balance``, який цю логіку й ніс: ``AuthError``
    стає «не налаштовано», будь-який інший виняток — «помилка запиту», і жоден із
    них не зриває опитування решти.
    """
    rows: list[dict[str, Any]] = []
    for name in names or BACKEND_NAMES:
        try:
            bk = get_backend(name)()
        except KeyError as e:
            rows.append(_error_row(name, f"невідомий бекенд: {e}"))
            continue
        try:
            rows.append(bk.balance().model_dump())
        except AuthError as e:
            rows.append(_error_row(name, f"не налаштовано: {str(e).splitlines()[0]}"))
        except Exception as e:  # один зламаний бекенд не має ховати решту таблиці
            rows.append(_error_row(name, f"помилка запиту: {str(e).splitlines()[0][:90]}"))
    return rows


def _error_row(name: str, detail: str) -> dict[str, Any]:
    return {"backend": name, "available": None, "unit": "", "spent": None,
            "detail": detail, "url": ""}


def _auth_state(row: dict[str, Any]) -> str:
    detail = str(row.get("detail") or "")
    if detail.startswith("не налаштовано"):
        return "не налаштовано"
    if detail.startswith(("помилка запиту", "невідомий бекенд")):
        return "помилка"
    return "ok"


# ---- крок 2: перевести в T4-години -----------------------------------------


def _vast_t4_rate() -> tuple[float, str] | None:
    """($/год за одну T4-годину на ринку Vast, пояснення) або ``None``.

    Рахується через A100 — карту, яку ми **міряли самі** (3.68× T4), тож із живої
    ринкової ціни виходить найтвердіший курс із усіх тут: ціна справжня, множник
    виміряний. Один запит до маркетплейсу; не вийшов — краще не переводити взагалі.
    """
    try:
        from gpurunner.backends.vast import VastBackend

        offers = VastBackend().search_offers(gpu="A100", limit=_VAST_OFFER_SAMPLE)
    except Exception:
        return None
    prices = sorted(float(o.get("dph_total") or 0) for o in offers if o.get("dph_total"))
    if not prices:
        return None
    a100_price = prices[0]
    eq = quota.equivalence("A100")
    if eq is None:
        return None
    return a100_price / eq.factor, f"найдешевша A100 на ринку ${a100_price:.3f}/год ÷ {eq.factor:.2f}"


def _best_value_gpu(rates: dict[str, float],
                    overrides: dict[str, float]) -> tuple[str, float, str] | None:
    """Карта з найкращим «T4-годин за долар» серед тих, що мають і ціну, і множник."""
    best: tuple[str, float, str] | None = None
    for gpu, price in rates.items():
        if gpu == "none" or price <= 0:
            continue
        eq = gpu_equiv.equivalence(gpu, overrides)
        if eq is None:
            continue
        per_dollar = eq.factor / price
        if best is None or per_dollar > best[1]:
            best = (gpu, per_dollar, eq.basis)
    return best


def _t4_from_dollars_modal(usd: float) -> T4Value | None:
    from gpurunner.backends.modal import _GPU_HOURLY

    rate = _GPU_HOURLY.get("T4")
    if not rate:
        return None
    return T4Value(usd / rate, "тариф", f"опублікований тариф Modal T4 ${rate:.2f}/год")


def _t4_from_dollars_beam(usd: float, overrides: dict[str, float]) -> T4Value | None:
    from gpurunner.backends.beam import _GPU_HOURLY, _gpu_hourly

    # Beam тарифу на T4 не публікує взагалі, тож питання переформульовується
    # чесно: скільки T4-годин ці гроші можуть КУПИТИ, якщо взяти найвигіднішу
    # карту з тих, що Beam і тарифікує, і ми вміємо перевести.
    #
    # Ставки беруться через `_gpu_hourly`, а не з прайсу: рахунок за RTX4090
    # виявився в 2.2 раза більшим за прайсовий, і рахувати куплені години за
    # прайсом означало б показати на дашборді втричі більше, ніж є.
    rates = {gpu: rate for gpu in _GPU_HOURLY if (rate := _gpu_hourly(gpu)) is not None}
    best = _best_value_gpu(rates, overrides)
    if best is None:
        return None
    gpu, per_dollar, basis = best
    return T4Value(usd * per_dollar, basis,
                   f"T4-тарифу немає; за найвигіднішою картою бекенда — {gpu}")


def _t4_from_vast(usd: float) -> T4Value | None:
    rate = _vast_t4_rate()
    if rate is None:
        return None
    per_hour, explain = rate
    return T4Value(usd / per_hour, "ринок", explain)


def to_t4(view: ServiceView, overrides: dict[str, float]) -> T4Value | None:
    """Скільки T4-годин «важить» залишок цього сервісу. ``None`` — не переводиться.

    Порожній або від'ємний баланс (Lightning на вичерпаному free-тарифі показує
    ``-0.12 credits``) — це нуль годин, а не «не переводиться»: курс відомий, і
    ховати такий сервіс у списку неперекладених було б брехнею про причину.
    """
    value = view.available
    if value is None:
        return None
    if value <= 0:
        return T4Value(0.0, "вимір", "залишку немає")
    backend, unit = view.backend, view.unit
    # Розбір іде ПО ОДИНИЦІ, а не по бекенду: у Colab вона залежить від тарифу
    # (на Pro — units, на безкоштовному — години), і зашивати «colab завжди в
    # units» означало б ділити години на курс одиниць після зміни плану.
    if unit in ("год", "hours"):
        return T4Value(value, "вимір", "квота вже номінована в годинах T4-сесії")
    if unit == "units" and backend == "colab":
        rate = view.quota.rate if view.quota and view.quota.rate else _COLAB_UNITS_PER_T4_HOUR
        return T4Value(value / rate, "план", f"курс тарифу Pro: {rate:.2f} units за T4-годину")
    if unit == "credits" and backend == "lightning":
        return T4Value(value / _CREDITS_PER_T4_HOUR, "план",
                       f"курс тарифу: {_CREDITS_PER_T4_HOUR:.2f} credits за T4-годину")
    if unit == "$":
        if backend == "modal":
            return _t4_from_dollars_modal(value)
        if backend == "beam":
            return _t4_from_dollars_beam(value, overrides)
        if backend == "vast":
            return _t4_from_vast(value)
    return None


# ---- крок 3: зібрати картку -------------------------------------------------


def build_view(row: dict[str, Any], overrides: dict[str, float],
               *, now: datetime | None = None,
               handles: list[Any] | None = None) -> ServiceView:
    backend = str(row["backend"])
    view = ServiceView(
        backend=backend,
        auth=_auth_state(row),
        available=row.get("available"),
        unit=str(row.get("unit") or ""),
        source="api" if row.get("available") is not None else "—",
        spent=row.get("spent"),
        detail=str(row.get("detail") or ""),
        url=str(row.get("url") or ""),
    )
    try:
        state = quota.compute(backend, now=now, handles=handles)
    except Exception as e:  # квота — допоміжна, вона не має ламати картку
        state = None
        view.detail = f"{view.detail} · квоту порахувати не вдалось ({type(e).__name__})".strip(" ·")
    view.quota = state

    if backend in QUOTA_IS_PRIMARY and state is not None:
        # API або мовчить про залишок, або говорить не про той (Kaggle віддає
        # AI-квоту в доларах). Головним стає розрахунок; відповідь API — у деталі.
        view.api_note = view.detail
        if state.remaining is not None:
            view.available, view.unit, view.source = state.remaining, state.unit, "розрахунок"
        else:
            view.available, view.unit, view.source = None, state.unit, "—"
        view.detail = state.note

    view.t4 = to_t4(view, overrides)
    # Міні-історія їде разом з оглядом, а не окремим запитом на кожну картку:
    # це локальний SQLite, і сім дешевих читань дешевші за сім HTTP-раундтрипів
    # із браузера, кожен з яких знову відкривав би ту саму БД.
    view.spark = [
        {"ts": p["ts"], "v": p["available"]}
        for p in quota.snapshots(backend, days=45)
        if p["available"] is not None
    ][-_SPARK_POINTS:]
    return view


@dataclass
class Overview:
    """Те, що бачить сторінка цілком."""

    generated_at: datetime
    services: list[ServiceView] = field(default_factory=list)

    @property
    def total_t4(self) -> float:
        return sum(s.t4.hours for s in self.services if s.t4)

    @property
    def estimated_t4(self) -> float:
        """Яка частина підсумку спирається на оцінку, а не на вимір чи ціну."""
        return sum(s.t4.hours for s in self.services if s.t4 and not s.t4.confident)

    @property
    def unconverted(self) -> list[dict[str, Any]]:
        """Сервіси з ненульовим залишком, який перевести не вдалось.

        Мовчазне нехтування ними означало б, що підсумок повний, коли він неповний.
        """
        return [
            {"backend": s.backend, "available": s.available, "unit": s.unit}
            for s in self.services
            if s.t4 is None and s.available is not None and s.available > 0
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "services": [s.as_dict() for s in self.services],
            "total": {
                "t4_hours": round(self.total_t4, 2),
                "estimated_t4_hours": round(self.estimated_t4, 2),
                "unconverted": self.unconverted,
                "needs_factor": gpu_equiv.needs_override(),
            },
        }


def overview(names: list[str] | tuple[str, ...] | None = None,
             *, now: datetime | None = None) -> Overview:
    """Опитати бекенди й зібрати повну картину. Мережевий виклик."""
    from gpurunner.core import manifest

    now = now or datetime.now(tz=UTC)
    overrides = quota.gpu_factors()
    handles = manifest.load()
    views = [
        build_view(row, overrides, now=now, handles=handles)
        for row in collect_reports(names)
    ]
    return Overview(generated_at=now, services=views)


def record_snapshot(view: ServiceView, *, ts: datetime | None = None) -> None:
    """Зберегти точку для графіка історії."""
    quota.add_snapshot(
        view.backend, view.available, view.unit,
        view.t4.hours if view.t4 else None, ts=ts,
    )
