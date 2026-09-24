"""Вибір машини: ЦІЛЬ темпу — вимога, а серед придатних — найдешевша робота.

🔴🔴 Переписано 23.09.2026. Доти вибір мав підлоги, тіри поступок і скор
«сторінки ÷ ((ціна години + $0.10) × години)», але не мав ЦІЛІ. Наслідки,
заміряні на 54 орендах 18–23.09:

- 19-ядерна Q RTX 8000 дала 6 517 стор/год за $0.060/тис, 10-ядерна та сама
  карта тієї ж ночі — 1 068 за $0.23. Скор бачив їх майже однаковими: модель
  не знала густини (123.6 рядка замість 39) і пакувала шарди з суми VRAM.
- Тіри пускали машину «що не добирає до порогів рівня 0» (`degraded`), а
  очікування на ядра закінчувалось `settled_for_less` — тобто вимога щоразу
  програвала ринку.

Тепер правило одне й перевірюване:

1. **Придатна** машина — та, чий ОБЕРЕЖНИЙ прогноз (`pph × GUARANTEE_FACTOR`)
   не нижчий за `Need.target_pph`, і тисяча сторінок на ній не дорожча за
   стелю. Плюс бан, майнінгові карти, диск, бюджет і строк.
2. Серед придатних виграє найменша **очікувана повна ціна** заходу: підйом,
   накладні черги й читання, поділені на ризик невдалої оренди (надійність і
   пам'ять реєстру).
3. Придатних немає — порожньо, і в `reason` названо найкраще, що ринок має.
   Слабшої машини вибір не дає НІКОЛИ: чекання на ринок безкоштовне, а слабка
   оренда — ні.

`target_pph = 0` — старий режим без цілі (для не-HTR викликів і старих
планів): тоді діє лише `min_pages_per_hour`, як і доти.

🔴 Заявлений `inet_down` у вибір не входить. Оффер із «755 Mbps» віддавав 0.5
і з'їв годину; канал вирішує `probe_box` на живому боксі.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from gpurunner.core.boxes import BoxVerdict
from gpurunner.core.htr_sizing import (
    CORE_LINES_PER_HOUR,
    GB_PER_SHARD,
    LINES_PER_PAGE_DEFAULT,
    MIN_CORES_PER_SHARD,
    Sizing,
    fixed_cost_for,
    gb_per_shard_for,
    plan_sizing,
    predict_cost,
    predict_hours,
)

#: 🔴 ГОЛОВНА вимога за замовчуванням, стор/год ЕТАЛОННОГО матеріалу. Доведено
#: можливою: spr-43, Q RTX 8000 / 19.2 ядра / 14 шардів — 6 517 стор/год за
#: $0.060 за тисячу, на кадрах 5.5 Мпікс і 39 рядках на сторінку.
DEFAULT_TARGET_PPH = 5000.0

#: 🔴🔴 ЦІЛЬ — ЦЕ ПОТУЖНІСТЬ МАШИНИ, а не стала в сторінках. Сторінка коштує
#: `L0(Мпікс) + рядки` рядко-еквівалентів, і на важчому матеріалі та сама
#: машина дає менше сторінок: 5 000 на 251 рядку фізично недосяжні ні на
#: якому залізі ринку (стеля ядер 64 × 20 000 / 291 = 4 400). Стала ціль у
#: сторінках перетворила б кожну щільну чергу на вічний `market_empty`.
#:
#: Тому «5 000» означає: машина, що на ЕТАЛОННОМУ матеріалі (spr-43, де ціль
#: доведено) дає 5 000 стор/год. На черзі черга переводиться у свої сторінки
#: (`Need.target_here`): важчий матеріал — пропорційно менше сторінок, легший
#: — не більше цілі (вимогу не підвищуємо там, де сторінки дешевші).
REFERENCE_MPX = 5.5
REFERENCE_LINES = 39.0

#: 🔴 Скільки від прогнозу моделі вважати ГАРАНТОВАНИМ — p20 історії.
#:
#: Зшивка 517 заходів (`htr.calibrate.stitch_runs`), лише великі — від 500
#: сторінок, 57 штук, бо ціль про стійкий темп, а не про мікросправи: факт /
#: прогноз має медіану 1.12, p25 0.93, **p20 0.86**. Тобто на прогнозі × 0.86
#: чотири заходи з п'яти не повільніші. Звірка з доведеною машиною: Q8000 /
#: 19.2 ядра / 5.5 Мпікс / 39 рядків — модель 5 818, × 1.12 = 6 516, факт 6 517.
#:
#: Дрібні заходи гірші (p20 0.71): їх з'їдають накладні справи в раннері, і
#: це лікується раннером, а не множником.
GUARANTEE_FACTOR = 0.86

#: 🔴 Стеля тисячі сторінок, коли ціль задано. $0.10 — удвічі від доведеного
#: ($0.049–0.060 на V100 142447 і Q8000 115469): запас на ринок, але не згода
#: на машину, що бере грошима за швидкість, яка не окупається.
MAX_USD_PER_1000_WITH_TARGET = 0.10

#: Стеля тисячі сторінок у старому режимі без цілі. Виміряно:
#:
#:     V100 Колорадо   2278 стор/год · $0.222/год → $0.097 / 1000
#:     RTX 3090        1564 · $0.196              → $0.125 / 1000
#:     3×L40 (08-11)   3966 · $1.200              → $0.303 / 1000  ← так не треба
MAX_USD_PER_1000_PAGES = 0.20

#: Стеля ціни ядро-години, $. Див. `Need.max_usd_per_core_h`: 4× медіани по
#: 384 орендах нашої історії, відсіває 1% машин.
MAX_USD_PER_CORE_H = 0.020

#: Нижче цієї надійності хоста не беремо. Сигнал слабкий (заміри 18–23.09:
#: частка вдалих оренд 23% нижче 0.98 і ~45% вище), тож поріг лише відсікає
#: явно хворих; решту розрізняє реєстр за `machine_id`.
MIN_RELIABILITY = 0.90

#: Надійність, яку приписуємо офферу без поля.
UNKNOWN_RELIABILITY = 0.90

#: 🔴 Майнінгові карти (CMP, P102/P104/P106) — не бере. 06.09.2026 бойовий захід
#: P2 узяв NVIDIA CMP 170HX: 20 шардів, 4327 стор/год на першій справі, а тоді
#: інстанс перейшов у `offline` посеред другої — робота на боксі втрачена.
MINING_GPU_PREFIXES = ("CMP", "P102", "P104", "P106")


def is_mining_card(offer: dict[str, Any]) -> bool:
    name = str(offer.get("gpu_name") or "").strip().upper()
    return any(name.startswith(pfx) or f" {pfx}" in name for pfx in MINING_GPU_PREFIXES)


@dataclass(frozen=True)
class Need:
    """Чого ми хочемо від машини на цей захід."""

    pages: int
    max_hours: float
    budget_usd: float
    #: Скільки СПРАВ у черзі: кожна має фіксовану ціну (`PER_CASE_SEC`).
    cases: int = 1
    disk_gb: int = 120
    #: Явна межа каналу, Мбіт/с. 0 = рахувати від обсягу даних.
    min_net_mbps: float = 0.0
    #: Скільки МБ кадрів на сторінку качає бокс. Нуль = обсяг невідомий.
    data_mb_per_page: float = 0.0
    #: Ядер на шард (`-p cores_per_shard`); 0 = `MIN_CORES_PER_SHARD`.
    cores_per_shard: float = 0.0
    gb_per_shard: float = GB_PER_SHARD
    warm: bool = False
    #: Матеріал заходу: ТИПОВА площа кадру (Мпікс) і рядків на сторінку.
    frame_mpx: float = 0.0
    lines_per_page: float = 0.0
    #: Густину не міряли — `lines_per_page` припущена, а не виміряна. Вибір
    #: однаково відбувається (інакше перший прогін справи не мав би машини
    #: взагалі), а звірку з обіцянкою під час заходу робить наглядач.
    lines_assumed: bool = False
    #: Стеля вартості однієї справи (середньої по черзі); `None` — лише бюджет.
    max_cost_per_case: float | None = None
    #: Стеля вартості тисячі сторінок, $. `None` = дефолт режиму.
    max_usd_per_1000_pages: float | None = None
    #: 🔴🔴 ЦІЛЬОВИЙ ТЕМП, стор/год. Машина, чий обережний прогноз нижчий,
    #: не береться НІКОЛИ. 0 = старий режим без цілі.
    target_pph: float = 0.0
    #: Підлога темпу старого режиму (без цілі). З ціллю не потрібна.
    min_pages_per_hour: float = 0.0
    #: Обсяг найбільшого архіву черги, МБ — фіксована ціна входу.
    max_archive_mb: float = 0.0
    #: 🔴🔴 СТЕЛЯ ЦІНИ ЯДРО-ГОДИНИ, $. Ядра — пропускна здатність; 23.09.2026
    #: та сама карта за $0.0270 за ядро-годину виявилась на 78% дорожчою за
    #: роботу, ніж за $0.0152. `None` = `MAX_USD_PER_CORE_H`.
    max_usd_per_core_h: float | None = None

    @property
    def target_here(self) -> float:
        """Ціль у сторінках ЦІЄЇ черги: потужність еталонної машини на цьому матеріалі."""
        if self.target_pph <= 0:
            return 0.0
        ref = fixed_cost_for(REFERENCE_MPX) + REFERENCE_LINES
        here = fixed_cost_for(self.frame_mpx) + (self.lines_per_page or LINES_PER_PAGE_DEFAULT)
        return self.target_pph * min(1.0, ref / here)

    @property
    def usd_per_1000_ceiling(self) -> float:
        if self.max_usd_per_1000_pages is not None:
            return float(self.max_usd_per_1000_pages)
        return MAX_USD_PER_1000_WITH_TARGET if self.target_pph > 0 else MAX_USD_PER_1000_PAGES

    @property
    def max_dph(self) -> float:
        """Найдорожча година, за яку ціль узагалі може вкластись у стелю тисячі.

        Похідна, а не стала: машина, що ДАЄ ціль, може коштувати
        `ціль × стеля / 1000` на годину — і ні центом більше. 0 — ціль не задана.
        """
        if self.target_pph <= 0:
            return 0.0
        return self.target_here / GUARANTEE_FACTOR * self.usd_per_1000_ceiling / 1000.0

    @property
    def data_gb(self) -> float:
        """Скільки ГБ кадрів бокс звантажить на цю чергу; 0 — обсяг невідомий."""
        return self.pages * self.data_mb_per_page / 1024.0


@dataclass(frozen=True)
class ScoredOffer:
    """Оффер із прорахованою вартістю рішення."""

    offer: dict[str, Any]
    machine_id: int
    sizing: Sizing
    hours: float
    cost: float
    score: float
    verdict: BoxVerdict | None = None
    rejects: tuple[str, ...] = ()
    #: Обережний темп: `sizing.pages_per_hour × GUARANTEE_FACTOR`.
    pph_sure: float = 0.0
    #: Очікувана повна ціна заходу з поправкою на ризик невдалої оренди.
    expected_cost: float = math.inf
    #: Плата хосту за вхідний трафік черги, $ (уже в `cost`).
    traffic_usd: float = 0.0
    #: Та сама плата на тисячу сторінок, $.
    traffic_per_1000: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.rejects

    @property
    def usd_per_1000(self) -> float:
        """Скільки коштує тисяча сторінок на цій машині: читання плюс трафік."""
        if not self.sizing.usable:
            return math.inf
        return 1000.0 * _dph(self.offer) / self.sizing.pages_per_hour + self.traffic_per_1000

    @property
    def explain(self) -> str:
        o = self.offer
        head = (
            f"offer {o.get('id')} · machine {self.machine_id} · "
            f"{o.get('gpu_name')}×{o.get('num_gpus') or 1} · "
            f"{_cores(o):.0f} ядер · {_vram_gb(o):.0f} ГБ · "
            f"${_dph(o):.3f}/год · {o.get('geolocation') or '?'}"
        )
        if self.rejects:
            return f"{head} — ✗ {', '.join(self.rejects)}"
        tail = (
            f" → {self.sizing.shards} шардів × {self.sizing.threads_per_shard} потоків, "
            f"{self.sizing.pages_per_hour:.0f} стор/год (гарантовано {self.pph_sure:.0f}), "
            f"{self.hours:.1f} год, ${self.cost:.2f} "
            f"(${self.usd_per_1000:.3f}/1000 стор)"
        )
        if self.traffic_usd >= 0.01:
            tail += (f"; у т.ч. трафік ${self.traffic_usd:.2f} "
                     f"(${_down_cost(o) * 1024:.0f}/ТБ)")
        if self.sizing.wasted_cores >= 8:
            tail += (
                f"; оплачено {_cores(o):.0f} ядер, задіяно {self.sizing.cores_used:.0f}"
                f" (обмежувач — {self.sizing.limited_by})"
            )
        if self.verdict is not None and self.verdict.state != "unknown":
            tail += f"; реєстр: {self.verdict.state} — {self.verdict.reason}"
        return head + tail


# ---- витяг полів оффера ----------------------------------------------------


def _num(offer: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = offer.get(key)
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _cores(offer: dict[str, Any]) -> float:
    """Ефективні ядра. 🔴 Саме `cpu_cores_effective`, а не `cpu_cores`: бокси,
    що рапортували 96 «ядер», видавали менше за 64-ядерний V100 — контейнеру
    дістається квота, а не весь хост."""
    return _num(offer, "cpu_cores_effective") or _num(offer, "cpu_cores")


def _vram_gb(offer: dict[str, Any]) -> float:
    """Уся VRAM оффера в ГБ: `gpu_ram` (на карту) × число карт. Розкладку по
    картах робить `plan_sizing` — шард не ділиться між картами."""
    return _num(offer, "gpu_ram") / 1024.0 * max(1.0, _num(offer, "num_gpus", 1.0))


def _dph(offer: dict[str, Any]) -> float:
    return _num(offer, "dph_total")


def _down_cost(offer: dict[str, Any]) -> float:
    """Плата хосту за вхідний трафік, $/ГБ (`inet_down_cost`); 0 — не вказано.

    🔴🔴 У ціні за годину її НЕМАЄ, а платиться вона за кожен гігабайт кадрів.
    24.09.2026, захід spr-97-q4: рахунок Vast — GPU $0.46, вхідний трафік
    $0.47. Естонська A4000 за $0.14/год брала $0.026/ГБ (медіана ринку
    $0.0026) і за 10 ГБ томів 102+105 виставила $0.27 трафіку проти $0.10 за
    саму карту. Вибір бачив лише години й вважав її дешевою.
    """
    return _num(offer, "inet_down_cost")


def traffic_usd(offer: dict[str, Any], need: Need) -> float:
    """Скільки хост візьме за звантаження кадрів цієї черги, $."""
    return need.data_gb * _down_cost(offer)


def traffic_per_1000(offer: dict[str, Any], need: Need) -> float:
    """Плата за трафік на тисячу сторінок цієї черги, $."""
    return need.data_mb_per_page * 1000.0 / 1024.0 * _down_cost(offer)


def _reliability(offer: dict[str, Any]) -> float:
    return _num(offer, "reliability2") or _num(offer, "reliability")


def _num_gpus(offer: dict[str, Any]) -> int:
    return int(_num(offer, "num_gpus", 1.0) or 1)


def machine_id_of(offer: dict[str, Any]) -> int:
    return int(_num(offer, "machine_id"))


#: Публічні імена для тих самих витягів — щоб ворота заліза й CLI не лізли в
#: приватні функції модуля.
offer_cores = _cores
offer_vram_gb = _vram_gb
offer_dph = _dph
offer_down_cost = _down_cost
offer_reliability = _reliability


def target_cores_floor(need: Need) -> float:
    """Скільки ядер квоти треба, щоб ОБЕРЕЖНИЙ прогноз узагалі сягнув цілі.

    Стеля за ядрами — `CORE_LINES_PER_HOUR × ядра / (L0 + рядки)`, і жодна
    карта її не підніме. Тож машина з меншою квотою не пройде вибору за
    жодних умов — її нема сенсу тягнути з ринку. 0 — ціль не задана.
    """
    if need.target_pph <= 0:
        return 0.0
    lines = need.lines_per_page or LINES_PER_PAGE_DEFAULT
    per_core = CORE_LINES_PER_HOUR / (fixed_cost_for(need.frame_mpx) + lines)
    return math.ceil(need.target_here / GUARANTEE_FACTOR / per_core)


def offer_sizing(offer: dict[str, Any], need: Need) -> Sizing:
    """Модельна розкладка цього оффера під цей захід (без заміру реєстру)."""
    return plan_sizing(
        cores=_cores(offer), vram_gb=_vram_gb(offer), gb_per_shard=need.gb_per_shard,
        num_gpus=_num_gpus(offer), lines_per_page=need.lines_per_page,
        cores_per_shard=need.cores_per_shard or MIN_CORES_PER_SHARD,
        frame_mpx=need.frame_mpx,
    )


# ---- скоринг ---------------------------------------------------------------


def measured_pph_for(
    measured: dict[str, Any] | None,
    need: Need,
    *,
    cores: float,
    vram_gb: float,
    num_gpus: int,
    gpu_name: str = "",
) -> float:
    """Виміряний темп машини, ПЕРЕРАХОВАНИЙ на умови цього оффера й цієї черги.

    🔴🔴 Замір б'є модель лише там, де умови ті самі. Реєстр пам'ятає темп по
    `machine_id`, а машина продає кілька офферів (у 140182 — 12- і 6-ядерний на
    тій самій карті). Тому замір масштабується відношенням МОДЕЛЬНИХ темпів
    «тоді» і «тепер»; де умови збігаються, відношення — одиниця.
    Бракує умов заміру — замір застосовується як є.
    """
    if not measured:
        return 0.0
    pph = float(measured.get("pages_per_hour") or 0)
    if pph <= 0:
        return 0.0
    then_cores = float(measured.get("cores_quota") or measured.get("cores") or 0)
    then_vram = float(measured.get("vram_total_gb") or 0)
    then_mpx = float(measured.get("pages_per_hour_mpx") or 0)
    then_lines = float(measured.get("pages_per_hour_lines") or 0)
    then_gpus = int(float(measured.get("n_gpus") or 1) or 1)
    if not (then_cores > 0 and then_vram > 0 and then_mpx > 0 and then_lines > 0):
        return pph
    # Пам'ять шарда «тоді» — те саме правило, що «тепер», перенесене за площею
    # кадру. Інакше різниця між явною ручкою плану й формулою масштабувала б
    # замір там, де умови однакові.
    now_mpx = need.frame_mpx or then_mpx
    gb_then = need.gb_per_shard * gb_per_shard_for(then_mpx) / gb_per_shard_for(now_mpx)
    then = plan_sizing(
        cores=then_cores, vram_gb=then_vram,
        gb_per_shard=gb_then, num_gpus=then_gpus,
        lines_per_page=then_lines, frame_mpx=then_mpx,
        cores_per_shard=need.cores_per_shard or MIN_CORES_PER_SHARD,
    )
    # Невідома площа черги означає «така сама, як у заміру», а не «типові
    # 12 Мпікс»: інакше замір масштабувався б не через умови, а через різницю
    # наших припущень.
    now = plan_sizing(
        cores=cores, vram_gb=vram_gb, gb_per_shard=need.gb_per_shard,
        num_gpus=num_gpus, lines_per_page=need.lines_per_page,
        frame_mpx=now_mpx,
        cores_per_shard=need.cores_per_shard or MIN_CORES_PER_SHARD,
    )
    if then.pages_per_hour <= 0 or now.pages_per_hour <= 0:
        return pph
    return pph * (now.pages_per_hour / then.pages_per_hour)


def score_offer(
    offer: dict[str, Any],
    need: Need,
    verdict: BoxVerdict | None = None,
) -> ScoredOffer:
    """Один оффер: розкладка, час, ціна — і чому відкинуто, якщо відкинуто."""
    cores, vram, dph = _cores(offer), _vram_gb(offer), _dph(offer)
    sizing = offer_sizing(offer, need)
    # 🔴 ЗАМІР Б'Є МОДЕЛЬ: власне число машини точніше за прогноз із картки.
    measured_pph = measured_pph_for(
        verdict.best_measured if verdict is not None else None, need,
        cores=cores, vram_gb=vram, num_gpus=_num_gpus(offer),
        gpu_name=str(offer.get("gpu_name") or ""),
    )
    if measured_pph > 0:
        sizing = replace(sizing, pages_per_hour=measured_pph)
    # Те саме для ПІДЙОМУ хоста: реєстр пам'ятає, скільки ця машина стартує.
    boot_sec = float((verdict.best_measured or {}).get("boot_sec") or 0
                     ) if verdict is not None else 0.0
    hours = predict_hours(need.pages, sizing, warm=need.warm, cases=need.cases,
                          boot_sec=boot_sec)
    traffic = traffic_usd(offer, need)
    per_1000_traffic = traffic_per_1000(offer, need)
    cost = predict_cost(need.pages, sizing, dph, warm=need.warm, cases=need.cases,
                        boot_sec=boot_sec) + traffic
    pph_sure = sizing.pages_per_hour * GUARANTEE_FACTOR if sizing.usable else 0.0

    rejects: list[str] = []
    if verdict is not None and verdict.banned:
        rejects.append(f"у чорному списку: {verdict.reason}")
    if is_mining_card(offer):
        rejects.append(f"майнінгова карта {offer.get('gpu_name')}: без гарантій PCIe/VRAM")
    if not sizing.usable:
        rejects.append(f"не тримає жодного шарда (обмежувач — {sizing.limited_by})")
    if _num(offer, "disk_space") and _num(offer, "disk_space") < need.disk_gb:
        rejects.append(f"диск {_num(offer, 'disk_space'):.0f} < {need.disk_gb} ГБ")
    if _reliability(offer) and _reliability(offer) < MIN_RELIABILITY:
        rejects.append(f"надійність {_reliability(offer):.2f} < {MIN_RELIABILITY:.2f}")
    if hours > need.max_hours:
        rejects.append(f"{hours:.1f} год > {need.max_hours:.1f}")
    if cost > need.budget_usd:
        rejects.append(f"${cost:.2f} > ${need.budget_usd:.2f}")
    if need.max_cost_per_case is not None and cost / max(1, need.cases) > need.max_cost_per_case:
        rejects.append(f"${cost / max(1, need.cases):.2f} на справу > стелі "
                       f"${need.max_cost_per_case:.2f}")
    # 🔴🔴 ЦІЛЬ. Не тір і не штраф: машина, що обережно не дає цілі, не
    # варіант. Порожній ринок має давати чесне «порожньо», а не слабкий захід.
    if need.target_pph > 0 and sizing.usable and pph_sure < need.target_here:
        rejects.append(
            f"гарантовано {pph_sure:.0f} стор/год < цілі {need.target_here:.0f} "
            f"({sizing.shards} шардів, {cores:.0f} ядер, обмежувач — "
            f"{'ядра' if sizing.cpu_capped else 'карта' if sizing.card_capped else sizing.limited_by})"
        )
    if (need.target_pph <= 0 and need.min_pages_per_hour > 0 and sizing.usable
            and sizing.pages_per_hour < need.min_pages_per_hour):
        rejects.append(
            f"{sizing.pages_per_hour:.0f} стор/год < підлоги "
            f"{need.min_pages_per_hour:.0f} ({sizing.shards} шардів, {cores:.0f} ядер)"
        )
    # 🔴🔴 Ціна ЯДРО-ГОДИНИ — пряма ціна роботи.
    core_ceiling = (need.max_usd_per_core_h if need.max_usd_per_core_h is not None
                    else MAX_USD_PER_CORE_H)
    if core_ceiling > 0 and cores > 0 and dph > 0 and dph / cores > core_ceiling:
        rejects.append(
            f"${dph / cores:.4f} за ядро-годину > стелі ${core_ceiling:.4f} "
            f"(${dph:.3f}/год ÷ {cores:.0f} ядер)"
        )
    # Стеля тисячі: з ціллю — на ГАРАНТОВАНОМУ темпі, без цілі — на модельному.
    ceiling = need.usd_per_1000_ceiling
    basis = pph_sure if need.target_pph > 0 else sizing.pages_per_hour
    if basis > 0:
        per_1000 = 1000.0 * dph / basis + per_1000_traffic
        if sizing.usable and ceiling > 0 and per_1000 > ceiling:
            rejects.append(
                f"${per_1000:.3f} за 1000 стор. > стелі ${ceiling:.2f} "
                f"({basis:.0f} стор/год за ${dph:.3f}/год"
                + (f", трафік ${per_1000_traffic:.3f}/1000" if per_1000_traffic >= 0.001
                   else "") + ")"
            )

    # Очікувана ціна: невдала оренда коштує підйому й повтору, тож повна ціна
    # ділиться на ймовірність дійти до кінця — надійність хоста і пам'ять
    # реєстру про ЦЮ машину (зірка стартує, попереджена — ні).
    expected = math.inf
    score = 0.0
    if not rejects and dph > 0 and cost > 0:
        reliability = _reliability(offer) or UNKNOWN_RELIABILITY
        registry = verdict.score_factor if verdict is not None else 1.0
        expected = cost / max(1e-6, (reliability ** 2) * registry)
        score = need.pages / expected

    return ScoredOffer(
        offer=offer,
        machine_id=machine_id_of(offer),
        sizing=sizing,
        hours=hours,
        cost=cost,
        score=score,
        verdict=verdict,
        rejects=tuple(rejects),
        pph_sure=pph_sure,
        expected_cost=expected,
        traffic_usd=traffic,
        traffic_per_1000=per_1000_traffic,
    )


def rank_offers(
    offers: list[dict[str, Any]],
    need: Need,
    verdicts: dict[int, BoxVerdict] | None = None,
) -> list[ScoredOffer]:
    """Придатні оффери, найдешевша очікувана робота першою."""
    verdicts = verdicts or {}
    scored = [score_offer(o, need, verdicts.get(machine_id_of(o))) for o in offers]
    ok = [s for s in scored if s.ok]
    ok.sort(key=lambda s: (s.expected_cost, -s.sizing.pages_per_hour))
    return ok


@dataclass(frozen=True)
class Selection:
    """Результат вибору: придатні кандидати, відкинуті й чесне «чому порожньо»."""

    candidates: list[ScoredOffer] = field(default_factory=list)
    rejected: list[ScoredOffer] = field(default_factory=list)
    reason: str = ""

    @property
    def best(self) -> ScoredOffer | None:
        return self.candidates[0] if self.candidates else None

    @property
    def empty(self) -> bool:
        return not self.candidates


def _fastest_rejected(rejected: list[ScoredOffer]) -> ScoredOffer | None:
    usable = [r for r in rejected if r.sizing.usable and not r.offer.get("_banned")]
    return max(usable, key=lambda r: r.pph_sure, default=None)


def select_offers(
    offers: list[dict[str, Any]],
    need: Need,
    verdicts: dict[int, BoxVerdict] | None = None,
) -> Selection:
    """Придатні машини за очікуваною ціною; порожньо — з назвою найкращого.

    Нічого не послаблюється: ні ціль, ні бюджет, ні строк. Порожній результат
    означає «не орендуємо» — наглядач чекає ринку.
    """
    verdicts = verdicts or {}
    scored = [score_offer(o, need, verdicts.get(machine_id_of(o))) for o in offers]
    candidates = sorted((s for s in scored if s.ok),
                        key=lambda s: (s.expected_cost, -s.sizing.pages_per_hour))
    rejected = [s for s in scored if not s.ok]
    reason = ""
    if not candidates:
        best = _fastest_rejected(rejected)
        if best is None:
            reason = ("ринок не дав жодної машини, що вкладається в бюджет і строк"
                      if not offers else
                      "ринок не дав жодної машини, що тримає бодай один шард")
        else:
            what = ("не дає цілі" if need.target_pph > 0
                    else "не вкладається у вимоги (бюджет, строк, стелі)")
            reason = (
                f"жодна машина {what} — найкраща на ринку: "
                f"{best.offer.get('gpu_name')}×{best.offer.get('num_gpus') or 1}, "
                f"{_cores(best.offer):.0f} ядер, ${_dph(best.offer):.3f}/год → "
                f"гарантовано {best.pph_sure:.0f} стор/год "
                f"(${best.usd_per_1000:.3f}/1000); ✗ {', '.join(best.rejects)}"
            )
    return Selection(candidates=candidates, rejected=rejected, reason=reason)
