"""Ранжування офферів: скільки сторінок за долар дасть ця машина.

Раніше вибір був `offers[0]` — найдешевший оффер із потрібною назвою карти.
Це і є джерело половини інцидентів: назва карти не каже ні про ядра (а вони
визначають швидкість), ні про VRAM (а вона визначає, скільки ядер узагалі
працюватиме), ні про те, чи ця машина вже підводила.

Тут три речі, яких там не було:

1. **Скор на виміряних числах**, а не на картці: `стор/год ÷ $/год`, де
   `стор/год` рахує `htr_sizing` із ядер і VRAM.
2. **Пам'ять** — вердикт `core.boxes` множить скор, а бан відкидає взагалі.
3. **Деградація порогів тірами**, коли ринок порожній: спершу послаблюємо
   ціну, потім VRAM, потім ядра. Бюджет і строк не послаблюються ніколи —
   інакше «взяли хоч щось» означає «заплатили і не встигли».

🔴 Заявлений `inet_down` у скор не входить взагалі. Оффер із «755 Mbps»
віддавав 0.5 і з'їв годину; це число не інформація, а шум. Канал вирішує
`probe_box` на живому боксі — і його вирок іде в реєстр.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from gpurunner.core.boxes import BoxVerdict
from gpurunner.core.htr_sizing import (
    GB_PER_SHARD,
    MIN_CORES_PER_SHARD,
    Sizing,
    gb_per_shard_for,
    plan_sizing,
    predict_cost,
    predict_hours,
)

#: Еталон для чесного `slowdown_x`, коли на ринку взагалі нема з чим порівняти:
#: Tesla V100, 64 ефективні ядра, 32 ГБ — 2078-2486 стор/год (виміряно).
REFERENCE_CORES = 64.0
REFERENCE_VRAM_GB = 32.0

#: 🔴 Скільки коштує ГОДИНА ОЧІКУВАННЯ, окрім оренди.
#:
#: Без цього доданка скор «сторінок за долар» обирає найдешевшу машину, і це
#: не помилка арифметики — це помилка постановки. Замір сесії 2026-08-11:
#: 1665 сторінок на V100 ($0.222/год) = 0.76 год = $0.17; на RTX 4060 Ti
#: ($0.098/год) = 1.67 год = $0.16. **Грошей однаково, часу вдвічі більше.**
#: Тобто на цих сумах гроші майже нічого не розрізняють, а години —
#: розрізняють усе: поки бокс молотить, захід стоїть.
#:
#: Число взяте як типова ставка оренди в наших заходах ($0.10-0.34/год):
#: «година очікування коштує приблизно як година оренди». Підняти — і
#: наглядач ганятиметься за швидкістю; опустити до 0 — повернеться стара
#: поведінка «найдешевше за будь-яку ціну в часі».
#: 🔴 Знижено з 0.30 до 0.10 після 2026-08-11. Вихідне число робило дорогі
#: машини конкурентними: 3×L40 за $1.36/год «перемагала», а на ділі дала
#: $0.30 за тисячу сторінок проти $0.10 у V100 — тобто швидкість НЕ окупилась.
#: Час коштує грошей, але не стільки.
TIME_VALUE_USD_PER_HOUR = 0.10

#: 🔴 ГОЛОВНИЙ поріг вибору — скільки коштує тисяча сторінок.
#:
#: Це єдине число, у якому ціна й швидкість уже зведені разом, і саме його
#: треба тримати, а не «$/год» окремо. Виміряно на наших прогонах:
#:
#:     V100 Колорадо   2278 стор/год · $0.222/год → $0.097 / 1000
#:     RTX 3090        1564 · $0.196              → $0.125 / 1000
#:     RTX 4060 Ti     1673 · $0.338              → $0.202 / 1000
#:     3×L40 (08-11)   3966 · $1.200              → $0.303 / 1000  ← так не треба
#:
#: Стеля 0.20 лишає весь звичний ринок і відрізає машини, що беруть грошима
#: за швидкість, яка не окупається.
#: 🔴 Поріг УЗГОДЖЕНИЙ зі стелею за годину, і це не збіг: $0.365/год при
#: типових 2000 стор/год — це $0.18 за тисячу. Тобто все, що дорожче за 0.20,
#: суперечило б власній стелі користувача. Перевірено на живому флоті
#: 2026-08-12: RTX 3090 з 64 ядрами — $0.083, Tesla P40 — $0.118, і обидві
#: проходять; а RTX 3090 з 26 ядрами дає $0.233 і НЕ проходить — це саме той
#: випадок, заради якого поріг існує: машина дешева за годину, але повільна за
#: свої гроші (виміряний двійник такої машини дав 703 стор/год проти 2548 на
#: 64-ядерній тій самій карті).
MAX_USD_PER_1000_PAGES = 0.20

#: 🔴 Майнінгові карти (CMP, P102/P104/P106) — не бере. 06.09.2026 бойовий захід
#: P2 узяв NVIDIA CMP 170HX: 20 шардів, 4327 стор/год на першій справі, а тоді
#: інстанс перейшов у `offline` посеред другої — робота на боксі втрачена,
#: $0.10 за нуль, переоренда. У картки Vast таких карт VRAM і PCIe виглядають
#: нормально, тож відсіяти їх може лише ім'я.
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
    disk_gb: int = 120
    #: Явна межа каналу, Мбіт/с. 0 = рахувати від обсягу даних
    #: (`data_mb_per_page`), а коли й він невідомий — стара стала межа воріт.
    min_net_mbps: float = 0.0
    #: Скільки МБ кадрів на сторінку качає бокс (сума `pages_bytes` плану ÷
    #: сторінки). Нуль = план старий, обсяг невідомий.
    data_mb_per_page: float = 0.0
    #: Ядер на шард (`-p cores_per_shard`); 0 = `MIN_CORES_PER_SHARD`.
    cores_per_shard: float = 0.0
    gb_per_shard: float = GB_PER_SHARD
    warm: bool = False
    #: Матеріал заходу: медіана площі кадру (Мпікс) і рядків на сторінку.
    #: Нуль = не міряли; тоді прогноз рівно такий, як був до члена за матеріалом.
    frame_mpx: float = 0.0
    lines_per_page: float = 0.0
    #: Стеля вартості однієї справи; `None` — лише спільний бюджет заходу.
    max_cost_per_case: float | None = None
    #: Скільки коштує година очікування. `None` = дефолт модуля.
    time_value_usd_per_hour: float | None = None
    #: Стеля вартості тисячі сторінок, $. `None` = дефолт модуля.
    max_usd_per_1000_pages: float | None = None
    #: 🔴 ПІДЛОГА ТЕМПУ, стор/год: нижче неї машина не береться взагалі.
    #:
    #: Окремий поріг від стелі ціни, бо вони про різне. `max_usd_per_1000`
    #: пропускає скільки завгодно повільну машину, аби дешеву: 6-ядерна
    #: TITAN X за $0.051/год дає $0.234 за тисячу сторінок і проходить будь-яку
    #: стелю ціни — при 218 стор/год, тобто 3.7 години на чергу, яку
    #: 32-ядерна машина читає за 27 хвилин.
    #:
    #: 🔴 Тірами НЕ послаблюється, як бюджет і строк: сенс підлоги саме в
    #: тому, щоб порожній ринок не перетворювався на згоду взяти будь-що.
    #: Замість цього наглядач чекає ринку (`wait_for_cores_min`) і, якщо не
    #: дочекався, закінчує захід чесним `market_empty`.
    #: 0 = підлоги немає (стара поведінка).
    min_pages_per_hour: float = 0.0
    #: 🔴 Обсяг НАЙБІЛЬШОГО архіву черги, МБ — фіксована ціна входу.
    #:
    #: `data_mb_per_page` міряє ТЕМП, яким флот споживає кадри, і повільний
    #: канал там лише знижує прогноз. Але перший архів мусить лягти ЦІЛКОМ,
    #: перш ніж прочитається бодай одна сторінка, і цього ворота не бачили
    #: взагалі. 22.09.2026, spr-160: архів 2.25 ГБ на боксі з виміряними
    #: 21.6 Мбіт/с — це 14 хвилин до першої сторінки, і кожен обрив качання
    #: починався заново. Бокс пройшов усі ворота, бо і ядра, і ціна, і темп
    #: у нього були добрі; не годився саме канал під ЦЕЙ обсяг.
    max_archive_mb: float = 0.0


@dataclass(frozen=True)
class Tier:
    """Рівень поблажливості до ринку. `0` — як хочемо, далі — як доводиться."""

    level: int
    label: str
    min_reliability: float
    hours_frac: float          # частка `max_hours`, у яку маємо вкластись
    cost_frac: float           # частка бюджету
    min_shards: int
    min_cores: float


#: Невеликий штраф за кожен рівень поступки. Він не має перекривати різницю в
#: ціні (заради цього все й переписано), лише розводити машини з однаковим
#: скором на користь тієї, що відповідає вимогам повністю.
TIER_PENALTY = {0: 1.00, 1: 0.97, 2: 0.94, 3: 0.90}

#: Порядок послаблень: ціна → VRAM → ядра. Ядра тримаємо найдовше, бо саме
#: вони — пропускна здатність (73.8% часу сторінки — CPU-геометрія kraken).
#: 🔴 Планка ядер 8, а не 32 (06.09.2026): шард бере ~одне ядро, тож 8 шардів
#: «як треба» потребують 8 ядер, а не 32. std160 на 12 ядрах дав 5040
#: стор/год, а з планкою 32 цей бокс ішов у тір 3 зі штрафом 0.90 і підписом
#: «у 1.14× повільніше за еталон» при прогнозі 1100 і факті 3600. 32 було
#: спадком моделі «ядра купують темп», знятої дуеллю 05.09.
TIERS: tuple[Tier, ...] = (
    Tier(0, "як треба", 0.98, 0.60, 0.50, 8, 8),
    Tier(1, "дорожче", 0.95, 0.80, 1.00, 8, 8),
    Tier(2, "менше VRAM", 0.95, 0.85, 1.00, 4, 8),
    Tier(3, "менше ядер", 0.90, 1.00, 1.00, 1, 4),
)


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
    tier: int = 0
    degraded: bool = False
    slowdown_x: float = 1.0

    @property
    def ok(self) -> bool:
        return not self.rejects

    @property
    def usd_per_1000(self) -> float:
        """Головне число вибору: скільки коштує тисяча сторінок."""
        if not self.sizing.usable:
            return math.inf
        return 1000.0 * _dph(self.offer) / self.sizing.pages_per_hour

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
            f"{self.sizing.pages_per_hour:.0f} стор/год, "
            f"{self.hours:.1f} год, ${self.cost:.2f} "
            f"(${self.usd_per_1000:.3f}/1000 стор)"
        )
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
    """Уся VRAM оффера в ГБ: `gpu_ram` (на карту) × число карт.

    🔴 Це число чесне РІВНО ТОМУ, що шарди розкладаються по всіх картах
    (`_start_shard` дає шарду k пристрій `cuda:(k % N)`). Поки всі шарди
    сиділи на `cuda:0`, множення на `num_gpus` було брехнею: бюджет рахувався
    по машині, а працювала одна карта — вдвічі-увосьмеро більше шардів, ніж
    влазить, і гарантований OOM.

    Тобто ці дві речі мусять мінятись ЛИШЕ разом. Якщо колись доведеться
    повернути прив'язку до однієї карти — прибрати й множник.
    """
    return _num(offer, "gpu_ram") / 1024.0 * max(1.0, _num(offer, "num_gpus", 1.0))


def _dph(offer: dict[str, Any]) -> float:
    return _num(offer, "dph_total")


def _reliability(offer: dict[str, Any]) -> float:
    return _num(offer, "reliability2") or _num(offer, "reliability")


def machine_id_of(offer: dict[str, Any]) -> int:
    return int(_num(offer, "machine_id"))


#: Публічні імена для тих самих витягів — щоб ворота заліза й CLI не лізли в
#: приватні функції модуля.
offer_cores = _cores
offer_vram_gb = _vram_gb
offer_dph = _dph
offer_reliability = _reliability


# ---- скоринг ---------------------------------------------------------------


def measured_pph_for(
    measured: dict[str, Any] | None,
    need: Need,
    *,
    cores: float,
    vram_gb: float,
    num_gpus: int,
) -> float:
    """Виміряний темп машини, ПЕРЕРАХОВАНИЙ на умови цього оффера й цієї черги.

    🔴🔴 Замір б'є модель лише там, де умови ті самі. Реєстр пам'ятає темп по
    `machine_id`, а машина продає кілька офферів: у 140182 їх два — 12-ядерний
    і 6-ядерний, на тій самій карті. 21.09.2026 замір 918 стор/год, зроблений
    на 12 ядрах і на метриках (66 рядків на сторінку), приписався 6-ядерному
    офферу під протоколи консисторії (138 рядків) — і той виграв скор у
    машини, що була вдвічі швидшою. Захід отримав 1513 стор/год там, де ринок
    давав 16-, 28- і 32-ядерні машини за ті самі гроші.

    Тому замір масштабується відношенням МОДЕЛЬНИХ темпів «тоді» і «тепер».
    Там, де умови збігаються, відношення дорівнює одиниці й замір іде як є —
    тобто стара поведінка зберігається рівно там, де вона була правильною.

    Умови заміру беруться з того ж `measured`, що й сам темп: `cores_quota`
    (наша частка ядер), `vram_total_gb`, `n_gpus`, `pages_per_hour_mpx`
    (площа кадру) і `pages_per_hour_lines` (щільність рядків). Бракує
    будь-чого з них — масштабувати нічим, і замір застосовується як є: так
    само, як до появи цих полів у реєстрі.
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
    then = plan_sizing(
        cores=then_cores, vram_gb=then_vram,
        gb_per_shard=gb_per_shard_for(then_mpx), num_gpus=then_gpus,
        lines_per_page=then_lines,
        cores_per_shard=need.cores_per_shard or MIN_CORES_PER_SHARD,
    )
    now = plan_sizing(
        cores=cores, vram_gb=vram_gb, gb_per_shard=need.gb_per_shard,
        num_gpus=num_gpus, lines_per_page=need.lines_per_page,
        cores_per_shard=need.cores_per_shard or MIN_CORES_PER_SHARD,
    )
    if then.pages_per_hour <= 0 or now.pages_per_hour <= 0:
        return pph
    return pph * (now.pages_per_hour / then.pages_per_hour)


def score_offer(
    offer: dict[str, Any],
    need: Need,
    verdict: BoxVerdict | None = None,
    *,
    tier: Tier = TIERS[0],
) -> ScoredOffer:
    """Один оффер: розкладка, час, ціна, скор — і чому відкинуто, якщо відкинуто."""
    cores, vram, dph = _cores(offer), _vram_gb(offer), _dph(offer)
    sizing = plan_sizing(cores=cores, vram_gb=vram, gb_per_shard=need.gb_per_shard,
                         num_gpus=int(_num(offer, "num_gpus", 1.0) or 1),
                         lines_per_page=need.lines_per_page,
                         cores_per_shard=need.cores_per_shard or MIN_CORES_PER_SHARD)
    # 🔴 ЗАМІР Б'Є МОДЕЛЬ. Якщо ця сама машина вже щось для нас порахувала,
    # її власне число точніше за будь-який прогноз із картки оффера: модель
    # не знає ні реального розподілу ядер (V100 із 128 ядрами віддала нам 32),
    # ні щільності конкретної справи. Доти реєстр пам'ятав темп, показував
    # його в поясненні — і НЕ використовував у розрахунку.
    measured_pph = measured_pph_for(
        verdict.best_measured if verdict is not None else None, need,
        cores=cores, vram_gb=vram, num_gpus=int(_num(offer, "num_gpus", 1.0) or 1),
    )
    if measured_pph > 0:
        sizing = replace(sizing, pages_per_hour=measured_pph)
    hours = predict_hours(need.pages, sizing, warm=need.warm)
    cost = predict_cost(need.pages, sizing, dph, warm=need.warm)

    rejects: list[str] = []
    if verdict is not None and verdict.banned:
        rejects.append(f"у чорному списку: {verdict.reason}")
    if is_mining_card(offer):
        rejects.append(f"майнінгова карта {offer.get('gpu_name')}: без гарантій PCIe/VRAM, "
                       f"06.09.2026 CMP 170HX помер offline посеред справи")
    if not sizing.usable:
        rejects.append(f"не тримає жодного шарда (обмежувач — {sizing.limited_by})")
    if sizing.shards < tier.min_shards:
        rejects.append(f"{sizing.shards} шардів < {tier.min_shards}")
    if cores < tier.min_cores:
        rejects.append(f"{cores:.0f} ядер < {tier.min_cores:.0f}")
    if _num(offer, "disk_space") and _num(offer, "disk_space") < need.disk_gb:
        rejects.append(f"диск {_num(offer, 'disk_space'):.0f} < {need.disk_gb} ГБ")
    if _reliability(offer) and _reliability(offer) < tier.min_reliability:
        rejects.append(f"надійність {_reliability(offer):.2f} < {tier.min_reliability:.2f}")
    if hours > need.max_hours * tier.hours_frac:
        rejects.append(f"{hours:.1f} год > {need.max_hours * tier.hours_frac:.1f}")
    if cost > need.budget_usd * tier.cost_frac:
        rejects.append(f"${cost:.2f} > ${need.budget_usd * tier.cost_frac:.2f}")
    if need.max_cost_per_case is not None and cost > need.max_cost_per_case:
        rejects.append(f"${cost:.2f} > стелі справи ${need.max_cost_per_case:.2f}")
    # 🔴 Підлога темпу — ПЕРЕД стелею ціни, бо це різні питання: дешева
    # повільна машина проходить будь-яку стелю ціни й забирає в заходу години.
    # Тірами не послаблюється (див. `Need.min_pages_per_hour`).
    if (need.min_pages_per_hour > 0 and sizing.usable
            and sizing.pages_per_hour < need.min_pages_per_hour):
        rejects.append(
            f"{sizing.pages_per_hour:.0f} стор/год < підлоги "
            f"{need.min_pages_per_hour:.0f} ({sizing.shards} шардів, "
            f"{cores:.0f} ядер)"
        )
    # 🔴 Головний поріг: скільки коштує тисяча сторінок саме на цій машині.
    # Рахується з ЧИСТОЇ швидкості, без накладних на підйом — інакше дрібна
    # справа відкидала б будь-яку машину.
    ceiling = (need.max_usd_per_1000_pages if need.max_usd_per_1000_pages is not None
               else MAX_USD_PER_1000_PAGES)
    if sizing.usable and ceiling > 0:
        per_1000 = 1000.0 * dph / sizing.pages_per_hour
        if per_1000 > ceiling:
            rejects.append(
                f"${per_1000:.3f} за 1000 стор. > стелі ${ceiling:.2f} "
                f"({sizing.pages_per_hour:.0f} стор/год за ${dph:.3f}/год)"
            )

    score = 0.0
    if not rejects and dph > 0:
        reliability = _reliability(offer) or 0.90
        registry = verdict.score_factor if verdict is not None else 1.0
        # Не впритул до стелі: машина, що ледве встигає, не лишає запасу на
        # догінний прохід і на повільний старт.
        fit = max(0.0, min(1.0, (need.max_hours - hours) / max(1e-9, need.max_hours * 0.4)))
        # Сторінок за долар, де долар включає ціну години очікування —
        # інакше найдешевша машина виграє завжди, і захід стоїть удвічі довше
        # за ті самі гроші (див. TIME_VALUE_USD_PER_HOUR).
        time_value = (need.time_value_usd_per_hour
                      if need.time_value_usd_per_hour is not None
                      else TIME_VALUE_USD_PER_HOUR)
        effective = (dph + time_value) * hours
        score = (need.pages / effective) * (reliability**2) * registry * fit

    return ScoredOffer(
        offer=offer,
        machine_id=machine_id_of(offer),
        sizing=sizing,
        hours=hours,
        cost=cost,
        score=score,
        verdict=verdict,
        rejects=tuple(rejects),
        tier=tier.level,
        degraded=tier.level > 0,
    )


def rank_offers(
    offers: list[dict[str, Any]],
    need: Need,
    verdicts: dict[int, BoxVerdict] | None = None,
    *,
    tier: Tier = TIERS[0],
) -> list[ScoredOffer]:
    """Кандидати цього тіру, найкращий першим. Відкинуті не повертаються."""
    verdicts = verdicts or {}
    scored = [
        score_offer(o, need, verdicts.get(machine_id_of(o)), tier=tier) for o in offers
    ]
    ok = [s for s in scored if s.ok]
    ok.sort(key=lambda s: s.score, reverse=True)
    return ok


@dataclass(frozen=True)
class Selection:
    """Результат вибору: кандидати, тір і чесне «наскільки гірше»."""

    candidates: list[ScoredOffer] = field(default_factory=list)
    tier: Tier = TIERS[0]
    rejected: list[ScoredOffer] = field(default_factory=list)
    reason: str = ""

    @property
    def best(self) -> ScoredOffer | None:
        return self.candidates[0] if self.candidates else None

    @property
    def empty(self) -> bool:
        return not self.candidates


#: Наскільки зірка може програвати найкращому скору й однаково йти першою.
STAR_PRIORITY_MARGIN = 0.30


def stars_first(ranked: list[ScoredOffer]) -> list[ScoredOffer]:
    """Зірки в межах `STAR_PRIORITY_MARGIN` від найкращого скору — на початок черги.

    🔴 Зірка давала лише ×1.25 до скору, тож вільна перевірена машина раз у раз
    програвала невідомій: за 14 заходів 12–15.09.2026 перша спроба на
    невідомій машині падала на підйомі (145248 `slow_boot`, 49534 SSH), на
    каналі (39565) чи на маршруті до R2 (55752 — 8 томів непрочитані), поки
    зірки 56491 і 45392 стояли вільні. Невдала спроба коштує 2–13 хв оренди;
    зірка вже довела, що піднімається й качає. Порядок усередині груп — за скором.
    """
    if not ranked:
        return ranked
    floor = ranked[0].score * (1.0 - STAR_PRIORITY_MARGIN)
    stars = [c for c in ranked
             if c.verdict is not None and c.verdict.state == "starred" and c.score >= floor]
    picked = {id(c) for c in stars}
    return stars + [c for c in ranked if id(c) not in picked]


def select_offers(
    offers: list[dict[str, Any]],
    need: Need,
    verdicts: dict[int, BoxVerdict] | None = None,
) -> Selection:
    """Пройти тіри до першого непорожнього — і сказати, чого це коштувало.

    Порожньо на останньому тірі — це `Selection.empty`, а не «візьмемо хоч
    щось»: бюджет і строк не послаблюються ніколи.
    """
    verdicts = verdicts or {}
    ideal_hours, from_reference = _ideal_hours(offers, need)
    last_rejected: list[ScoredOffer] = []

    # 🔴 Тіри — це «наскільки довелось поступитись», а НЕ послідовний відсів.
    #
    # Було: беремо перший непорожній тір і далі не дивимось. Ціна інциденту
    # 2026-08-11: RTX 3090 за $0.19 має 28 ефективних ядер, тобто на чотири
    # менше за поріг tier 0 (32) — і вилітала повністю. У tier 0 лишалась сама
    # 3×L40 за **$1.36/год** і «перемагала» без суперників. При чесному
    # порівнянні на 1297 сторінках виграла б саме 3090: скор 2533 проти 2279,
    # ціна $0.19 проти $0.46. Тобто ціна в рішенні не брала участі взагалі.
    #
    # Тепер: збираємо кандидатів З УСІХ тірів і обираємо за скором глобально —
    # у ньому ціна вже врахована (сторінок за долар + година очікування).
    # Тір лишається міткою для звіту й невеликим штрафом, щоб за рівного скору
    # перемагала машина, що відповідає вимогам повністю.
    pooled: dict[int, ScoredOffer] = {}
    for tier in TIERS:
        for candidate in rank_offers(offers, need, verdicts, tier=tier):
            key = int(candidate.offer.get("id") or id(candidate.offer))
            if key not in pooled:
                pooled[key] = replace(candidate, score=candidate.score * TIER_PENALTY[tier.level])
    if pooled:
        best_first = stars_first(sorted(pooled.values(), key=lambda c: c.score, reverse=True))
        tier = TIERS[min(c.tier for c in best_first[:1])]
        candidates = best_first
        if any(c.tier > 0 for c in candidates[:1]):
            candidates = [
                replace(c, slowdown_x=(c.hours / ideal_hours) if ideal_hours > 0 else 1.0)
                for c in candidates
            ]
        best = candidates[0]
        tier = next(t for t in TIERS if t.level == best.tier)
        reason = ""
        if best.tier > 0:
            against = ("еталон V100 64 ядра (живих повноцінних не було)"
                       if from_reference else "найкраще на ринку")
            reason = (
                f"tier {best.tier} ({tier.label}): за скором виграла машина, що не "
                f"добирає до порогів рівня 0 — {best.offer.get('gpu_name')} "
                f"{_vram_gb(best.offer):.0f} ГБ / {_cores(best.offer):.0f} ядер, "
                f"${_dph(best.offer):.3f}/год · ${best.usd_per_1000:.3f} за 1000 стор.; "
                f"у {best.slowdown_x:.2f}× повільніше за {against}"
            )
        # 🔴 Відкинутих несемо й тоді, коли вибір ВІДБУВСЯ. Доти вони
        # зберігались лише у вироку «ринок порожній», тож питання «скільки
        # машин з'їла моя підлога і чи лишився запас» не мало відповіді доти,
        # доки не ставало пізно. Беремо найм'якший тір: те, що не пройшло там,
        # не пройшло ніде.
        rejected = [c for c in (score_offer(o, need, verdicts.get(machine_id_of(o)),
                                            tier=TIERS[-1]) for o in offers)
                    if c.rejects]
        return Selection(candidates=candidates, tier=tier, rejected=rejected,
                         reason=reason)

    for tier in TIERS:
        candidates = rank_offers(offers, need, verdicts, tier=tier)
        if not candidates:
            last_rejected = [
                score_offer(o, need, verdicts.get(machine_id_of(o)), tier=tier) for o in offers
            ]
            continue
        if tier.level > 0:
            candidates = [
                replace(c, slowdown_x=(c.hours / ideal_hours) if ideal_hours > 0 else 1.0)
                for c in candidates
            ]
        best = candidates[0]
        reason = ""
        if tier.level > 0:
            against = "еталон V100 64 ядра (живих повноцінних не було)" if from_reference \
                else "найкраще на ринку"
            reason = (
                f"tier {tier.level} ({tier.label}): порогів рівня 0 ринок не дав; "
                f"узято {best.offer.get('gpu_name')} "
                f"{_vram_gb(best.offer):.0f} ГБ / {_cores(best.offer):.0f} ядер "
                f"— у {best.slowdown_x:.2f}× повільніше за {against}"
            )
        return Selection(candidates=candidates, tier=tier, rejected=[], reason=reason)

    return Selection(
        candidates=[],
        tier=TIERS[-1],
        rejected=sorted(last_rejected, key=lambda s: s.hours)[:10],
        reason="ринок не дав жодної машини, що вкладається в бюджет і строк",
    )


def _ideal_hours(offers: list[dict[str, Any]], need: Need) -> tuple[float, bool]:
    """Скільки тривала б справа на ПОВНОЦІННІЙ машині. `(години, це_еталон)`.

    Знаменник для `slowdown_x`. Рахується лише по офферах, що проходять
    вимоги до заліза рівня 0 — інакше єдина пропозиція на ринку виявляється
    сама собі еталоном, `slowdown_x` дорівнює 1.00 і деградація виглядає як
    норма. Якщо повноцінних немає взагалі — беремо еталонний V100 із
    бенчмарку й так і кажемо: це еталон, а не жива пропозиція.
    """
    top = TIERS[0]
    best = math.inf
    for offer in offers:
        sizing = plan_sizing(
            cores=_cores(offer), vram_gb=_vram_gb(offer), gb_per_shard=need.gb_per_shard,
            num_gpus=int(_num(offer, "num_gpus", 1.0) or 1), lines_per_page=need.lines_per_page,
        )
        if sizing.shards < top.min_shards or _cores(offer) < top.min_cores:
            continue
        best = min(best, predict_hours(need.pages, sizing, warm=need.warm))
    if not math.isinf(best):
        return best, False
    reference = plan_sizing(
        cores=REFERENCE_CORES, vram_gb=REFERENCE_VRAM_GB, gb_per_shard=need.gb_per_shard,
        lines_per_page=need.lines_per_page,
    )
    return predict_hours(need.pages, reference, warm=need.warm), True
