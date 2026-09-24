"""Ворота заліза: чи ця машина справді така, як обіцяла картка оффера.

Викликається одразу після SSH і **до** заливки даних та pip. Порядок не
косметичний: раніше єдиний замір (канал) жив усередині job'а, тобто після
восьми хвилин встановлення kraken/torch — і за «оффер обіцяв 755 Мбіт/с,
віддає 0.5» ми платили спершу цими вісьмома хвилинами, а потім годиною
діагностики. Тут провал коштує ~30 секунд.

Ворота нічого не гасять і нічого не пишуть — вони лише виносять вирок.
Гасіння, запис у реєстр і перехід до наступного кандидата робить наглядач:
так цю логіку можна ганяти в тестах без жодного інстансу.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from gpurunner.core.boxes import DEAD_NET_MBPS
from gpurunner.core.htr_sizing import (
    MAX_SHARDS,
    MIN_CORES_PER_SHARD,
    VRAM_HEADROOM,
    Sizing,
    plan_sizing,
    predict_cost,
    predict_hours,
)
from gpurunner.core.offer_score import (
    GUARANTEE_FACTOR,
    Need,
    offer_cores,
    offer_dph,
    offer_vram_gb,
    traffic_per_1000,
    traffic_usd,
)

#: Наскільки менше обіцяного ще вважається «в межах похибки». Ядра
#: коливаються сильніше (квота контейнера пливе), VRAM — майже ні.
CORES_TOLERANCE = 0.80
VRAM_TOLERANCE = 0.90

#: Яку частку СВОЄЇ карти ми маємо бачити вільною, щоб вважати її своєю.
#: Нижче — на карті сидить чужий орендар, і винен у повільності не хост.
#: Заміряно 2026-08-19 на машині 139040: 5.1 з 24.0 ГБ (21%) → 1 шард замість
#: 8, темп 165-249 замість 2592 стор/год, ціна тисячі сторінок ×14 до стелі.
CARD_MINE_FRACTION = 0.60

#: Наскільки вище стелі ціни ще НЕ привід відмовитись від машини.
#:
#: 🔴 Ворота рахують ціну з МОДЕЛІ швидкості, а її власний розкид — ±20%
#: (`PAGES_PER_HOUR_PER_*` калібровані по нижній межі заміряного). Відмовляти
#: за перевищення на пів відсотка означає міряти шум і на тонкому ринку не
#: орендувати нічого: 2026-08-19 так відпали дві машини поспіль — $0.201 і
#: $0.206 при стелі $0.200, і захід лишився без боксів узагалі.
#:
#: 1.15 лишає з того боку все, заради чого поріг існує: бокс із чужим процесом
#: на карті давав $0.644 при стелі $0.20 — це 3.2×, а не 3%.
PRICE_TOLERANCE = 1.15

#: Нижче цього ядер сторінка коштує вдвічі дорожче (замір: 2 ядра/шард —
#: 38.3 с/стор проти 17.5 на 6). Такий бокс не рятує жодна ціна.
MIN_USABLE_CORES = 4.0

#: Нижче цього канал МЕРТВИЙ, і це вада хоста: бан (`slow_net`).
#:
#: 🔴 Живий повільний канал — не вада. Доти ворота мали одну сталу межу 20
#: Мбіт/с на все, і 15.09.2026 перечитування 14 томів (1.23 ГБ кадрів) відкинуло
#: машини з 6.1 і 2.1 Мбіт/с, забанивши обидві на 14 днів. На 6.1 Мбіт/с ці
#: кадри з передзавантаженням встигали б за флотом повністю, на 2.1 — прогін
#: затягнувся б на ~35 хв (+$0.14); натомість $0.08 пішло на невдалі оренди, і
#: захід не дав нічого.
#: Сама межа — `core.boxes.DEAD_NET_MBPS`: реєстр складає за нею й старі вироки.
#: Межа для планів без обсягу даних (`pages_bytes`) — стара стала. Нижче неї
#: машина відкидається, але НЕ банується: обсяг невідомий, вина не доведена.
LEGACY_MIN_NET_MBPS = 20.0
#: Запас над темпом, яким флот споживає кадри.
NET_HEADROOM = 1.2
#: Яку частку прогнозованого заходу дозволено з'їсти качанню першого архіву.
FIRST_PULL_SHARE = 0.25
#: Підлога стелі, хв: на короткому заході частка дає надто мало, і машину
#: відкидало б за цілком нормальне качання.
FIRST_PULL_FLOOR_MIN = 10.0
#: 🔴 Абсолютна стеля качання, хв — вона ж головна.
#:
#: Частка від годин сама по собі не рятує: на прогнозованій годині 14 хвилин
#: качання це «лише 23%», а насправді саме на них захід і загинув. Довге
#: качання небезпечне НЕ часткою, а тим, що `curl` рве з'єднання після 60 с
#: нижче підлоги швидкості: що довше тягнеться архів, то певніше десь
#: трапиться така хвилина. 12 хв лишає запас під звичайні архіви 300-500 МБ на
#: будь-якому пристойному каналі й відрізає ті, де обрив став питанням часу.
FIRST_PULL_HARD_MAX_MIN = 12.0


@dataclass(frozen=True)
class GateResult:
    """Вирок про машину. `outcome` — саме той рядок, що піде в реєстр боксів."""

    ok: bool
    outcome: str
    detail: str
    measured: dict[str, Any] = field(default_factory=dict)
    sizing: Sizing | None = None
    hours: float | None = None
    cost: float | None = None


def evaluate(probe: dict[str, Any], offer: dict[str, Any], need: Need) -> GateResult:
    """Звірити виміряне з обіцяним і перерахувати план на РЕАЛЬНОМУ залізі.

    Перша перевірка, що впала, дає вирок: далі рахувати нема сенсу, бо
    кожна наступна секунда — оплачена.
    """
    measured = _measured(probe)
    claimed_cores = offer_cores(offer)
    # 🔴🔴 `nproc` — це ХОСТ, а не наша частка. Замір 2026-08-19 на машині
    # 39565: контейнер рапортував 192 ядра при проданих 48, і план поїхав на
    # вчетверо неіснуючому залізі — 16 шардів × 8 потоків = 128 потоків на 48
    # куплених ядер. Те саме з `free -g`: 251 ГБ при проданих 62.9.
    #
    # Плануємо на `min(nproc, квота cgroup)` — ТІЛЬКИ на жорстких фактах.
    #
    # 🔴🔴 Продане число (`cpu_cores_effective`) сюди НЕ входить, і це не
    # недогляд. Vast продає ЧАСТКУ, але не ріже нею контейнер: машина 139040 з
    # 40 проданими ядер на 80 видимих видала 2592 стор/год — 65 сторінок на
    # продане ядро при калібруванні 35. Тобто `PAGES_PER_HOUR_PER_CORE` знято з
    # ВИДИМИХ ядер, і стеля $/1000 калібрована проти таких самих оцінок.
    # Обмежити план проданим означало б удвічі занизити прогноз на КОЖНОМУ
    # боксі (у нас типово nproc = 2× продане) — і власна ж стеля ціни
    # відкидала б машини, які досі возили черги без нарікань.
    #
    # Квота cgroup — інша річ: це не комерційна частка, а межа, яку ядро
    # справді не дасть перейти. Її й беремо, коли вона є.
    cores_seen = measured["cores_seen"]
    cores = cores_seen
    measured["cores_eff"] = cores
    vram_free = measured["vram_free_gb"]
    vram_total = measured["vram_total_gb"]
    claimed_vram = offer_vram_gb(offer)

    if not measured["gpu"] or vram_total <= 0:
        return GateResult(
            False, "never_booted",
            "у контейнері немає карти: `nvidia-smi` не віддав нічого",
            measured,
        )

    if claimed_cores and cores_seen < claimed_cores * CORES_TOLERANCE:
        return GateResult(
            False, "cpu_lie",
            f"ядер {cores_seen:.0f} проти обіцяних {claimed_cores:.0f} "
            f"({cores_seen / claimed_cores:.0%} від картки)",
            measured,
        )
    if cores < MIN_USABLE_CORES:
        return GateResult(
            False, "cpu_lie",
            f"ядер {cores:.0f} — повільніше за домашній ноутбук за будь-яку ціну",
            measured,
        )

    if claimed_vram and vram_total < claimed_vram * VRAM_TOLERANCE:
        return GateResult(
            False, "vram_lie",
            f"VRAM {vram_total:.1f} ГБ проти обіцяних {claimed_vram:.0f}",
            measured,
        )
    if (measured.get("vram_free_min_gb") or vram_free) < need.gb_per_shard:
        # Не «карта мала», а «карта зайнята»: сусід на тій самій машині.
        return GateResult(
            False, "vram_lie",
            f"вільної VRAM {vram_free:.1f} ГБ — не влазить навіть один шард "
            f"({need.gb_per_shard:.1f} ГБ); карта зайнята сусідом",
            measured,
        )

    if measured["disk_measured"] and measured["disk_free_gb"] < need.disk_gb:
        # 🔴 Нуль тут — теж вирок, а не «не міряли»: `df` не відповів на боксі,
        # де ми збираємось розпакувати десятки гігабайтів кадрів. Непідтверджене
        # місце дорожче за зайву переоренду.
        return GateResult(
            False, "disk_short",
            f"вільного диску {measured['disk_free_gb']:.0f} ГБ із замовлених {need.disk_gb}"
            + (" (df не відповів)" if measured["disk_free_gb"] <= 0 else ""),
            measured,
        )

    net = measured["net_mbps"]
    http = str(measured.get("net_http") or "").strip()
    if http == "000":
        # 🔴 `000` — це НЕ код відповіді, а «curl не довіз». Проба качає 32 МБ
        # із `--max-time 12`, тож усе повільніше за ~2.8 МБ/с обривається саме
        # так. Діагноз тут протилежний до 403: посилання може бути бездоганне
        # (перевірено `GET` з `Range` — 206), а канал хоста до R2 — мертвий.
        # Класифікувати це як `our_bug` означає спинити захід на справному
        # плані; правильна реакція та сама, що на `slow_net` — узяти інший бокс.
        return GateResult(
            False, "slow_net",
            "проба каналу не довезла 32 МБ за 12 с (curl 000) — канал хоста до "
            "сховища мертвий або надто повільний",
            measured,
        )
    if http and http not in ("200", "206"):
        # Наше посилання, а не хост. Машину НЕ звинувачуємо — інакше протухла
        # presigned URL мовчки виб'є з ринку всіх кандидатів поспіль: канал
        # міряється качанням саме цього файла, і 403 виглядає як «0.2 Мбіт/с».
        return GateResult(
            False, "our_bug",
            f"перевірка каналу дістала HTTP {http} — це наше посилання на дані "
            f"протухло або зіпсоване, а не канал хоста",
            measured,
        )
    claimed_net = offer.get("inet_down")
    promised = f" (в картці — {float(claimed_net):.0f})" if claimed_net else ""
    if measured["net_measured"] and net < DEAD_NET_MBPS:
        return GateResult(
            False, "slow_net",
            f"канал {net:.1f} Мбіт/с — мертвий (нижче {DEAD_NET_MBPS:.0f}){promised}",
            measured,
        )
    flat = need.min_net_mbps or (0.0 if need.data_mb_per_page > 0 else LEGACY_MIN_NET_MBPS)
    if flat > 0 and measured["net_measured"] and net < flat:
        return GateResult(
            False, "slow_for_data",
            f"канал {net:.1f} Мбіт/с при потрібних {flat:.0f}{promised} — живий, "
            f"машину не звинувачую",
            measured,
        )


    # План перераховується на ВИМІРЯНОМУ, а не на обіцяному — саме тут
    # 16 ГБ під 8 шардів перетворюються на 5 шардів, а не на OOM.
    # 🔴 Ємність рахується від НАЙЗАЙНЯТІШОЇ карти, а не від суми. Шарди
    # розкладаються круговою чергою `cuda:(k % N)`, тож карта з сусідом обмежує
    # всю машину: 3 картки по 11 ГБ вільних дають 11 шардів із суми, з них
    # чотири сядуть на карту, де влазить три — і це тихий OOM, за який догін
    # потім заплатить удруге. `vram_free_min_gb` тепер справді мінімум по
    # картах (awk у пробі рахував ПЕРШУ карту й називав це мінімумом).
    n_gpus = max(1, int(measured.get("n_gpus") or 1))
    per_card = float(measured.get("vram_free_min_gb") or (vram_free / n_gpus))
    capacity = per_card * n_gpus if per_card > 0 else vram_free
    # 🔴🔴 Поділ робиться ПОКАРТКОВО, а не від суми. `int(сума × 0.9 / gb)`
    # округлює ОДИН раз і тому щедріший за чесний покартковий підрахунок: дві
    # карти по 8 ГБ дають int(16×0.9/2.5) = 5 шардів, тоді як покартково це
    # int(8×0.9/2.5) = 2 на карту, тобто 4. А шарди розкладаються круговою
    # чергою, отже на кожну карту сідає ceil(шардів / карт) — і саме ця
    # величина мусить влазити, бо пам'ять не спільна.
    #
    # Ціна виміряна 2026-08-12 на кліровій справі: бокс віддав дві GTX 1080 по
    # 8 ГБ, стеля порахувалась із «16 ГБ», на cuda:0 сіли два шарди — і
    # «Tried to allocate 1.51 GiB, GPU 0 has 7.92 GiB»: 152 збої на 48 готових.
    per_card_shards = int(per_card * VRAM_HEADROOM // max(0.1, need.gb_per_shard))         if per_card > 0 else 0
    fit_cap = max(1, per_card_shards * n_gpus) if per_card_shards else MAX_SHARDS
    sizing = plan_sizing(cores=cores, vram_gb=capacity, gb_per_shard=need.gb_per_shard,
                         max_shards=min(MAX_SHARDS, fit_cap), num_gpus=n_gpus,
                         lines_per_page=need.lines_per_page,
                         frame_mpx=need.frame_mpx,
                         # 🔴 Число шардів звідси стає СТЕЛЕЮ регулятора на боксі:
                         # без ручки перечитування Скрибою стояло на 14 шардах
                         # при 0.7 ГБ карти на шард (15.09.2026).
                         cores_per_shard=need.cores_per_shard or MIN_CORES_PER_SHARD)
    hours = predict_hours(need.pages, sizing, warm=need.warm, cases=need.cases)
    # Трафік — та сама плата, що й у виборі: гігабайти черги × тариф хоста.
    traffic = traffic_usd(offer, need)
    cost = predict_cost(need.pages, sizing, offer_dph(offer), warm=need.warm,
                         cases=need.cases) + traffic

    # 🌐 Чи встигає канал за флотом. Кадри наступного тому качаються, поки
    # читається поточний, тож вузьке місце — не обсяг, а ТЕМП: флот споживає
    # `МБ/стор × стор/год`, і повільніший канал просто знижує темп. Машину
    # це саме по собі не відкидає — ціну й години вирішують ворота нижче.
    # Кадри качаються кількома з'єднаннями й кількома справами наперед, тож
    # канал для темпу — більший із двох замірів, як і для першого архіву нижче.
    flow_net = max(net, float(measured.get("net_par_mbps") or 0))
    slowdown, need_mbps = 1.0, 0.0
    base_hours, base_cost, base_pph = hours, cost, sizing.pages_per_hour
    if (need.data_mb_per_page > 0 and measured["net_measured"] and flow_net > 0
            and sizing.usable):
        need_mbps = need.data_mb_per_page * sizing.pages_per_hour / 3600.0 * 8 * NET_HEADROOM
        if flow_net < need_mbps:
            slowdown = need_mbps / flow_net
            sizing = replace(sizing, pages_per_hour=sizing.pages_per_hour / slowdown)
            hours = predict_hours(need.pages, sizing, warm=need.warm, cases=need.cases)
            cost = predict_cost(need.pages, sizing, offer_dph(offer), warm=need.warm,
                         cases=need.cases) + traffic
    net_why = (f"канал {flow_net:.1f} Мбіт/с не встигає за флотом (треба ~{need_mbps:.1f} на "
               f"{need.data_mb_per_page:.2f} МБ/стор) — темп ×{1 / slowdown:.2f}; "
               if slowdown > 1 else "")

    # 🔴🔴 ЦІНА ВХОДУ: перший архів мусить лягти ЦІЛКОМ до першої сторінки.
    #
    # Перевірка нижче міряє ТЕМП споживання кадрів — і повільний канал там лише
    # знижує прогноз. Але обсяг однієї справи каналом не ділиться: 2.25 ГБ на
    # 21.6 Мбіт/с — це 14 хвилин, протягом яких флот не читає нічого, а бокс
    # тарифікується. І що довше качання, то певніший обрив: curl рве на 60 с
    # нижче підлоги швидкості.
    #
    # Виміряно 22.09.2026 (spr-160, RTX A4000x4, Japan): офер обіцяв 1821
    # Мбіт/с, проба дала 21.6 — у 84 рази менше. Машина пройшла ВСІ ворота
    # (ядра, ціна, темп), бо жодні з них про мережу під цей обсяг не питають.
    if need.max_archive_mb > 0 and measured["net_measured"] and net > 0:
        # 🔴 Тут рахується ЧАС КАЧАННЯ, а качає транспорт діапазонами — до
        # восьми з'єднань. Міряти однопотоково, а платити багатопотоково
        # означало відсікати машини, які насправді встигають: проба занижує
        # рівно там, де транспорт виграє. Беремо більше з двох замірів, бо
        # саме стільки канал і дає під нашим навантаженням.
        pull_net = max(net, float(measured.get("net_par_mbps") or 0))
        pull_min = need.max_archive_mb * 8 / pull_net / 60.0
        limit_min = min(FIRST_PULL_HARD_MAX_MIN,
                        max(FIRST_PULL_FLOOR_MIN, FIRST_PULL_SHARE * hours * 60.0))
        if pull_min > limit_min:
            return GateResult(
                False, "slow_for_data",
                f"найбільший архів черги ({need.max_archive_mb / 1024:.1f} ГБ) на каналі "
                f"{pull_net:.1f} Мбіт/с тягнеться {pull_min:.0f} хв при стелі {limit_min:.0f}"
                f"{promised} — до першої сторінки бокс лише тарифікується",
                measured,
            )

    if not sizing.usable:
        return GateResult(
            False, "vram_lie",
            f"на виміряному залізі не тримається жодного шарда "
            f"(обмежувач — {sizing.limited_by})",
            measured, sizing, hours, cost,
        )

    # 🔴🔴 ЦІНА ТИСЯЧІ СТОРІНОК НА ВИМІРЯНОМУ ЗАЛІЗІ.
    #
    # Решта вироків абсолютна — «влазить у бюджет» і «встигає до строку», — і
    # саме тому бокс, ВДЕСЯТЕРО повільніший за той, по що ми йшли, їх проходив:
    # 2026-08-19, машина 139040, справа на 709 сторінок — 3.0 год < стелі 8 і
    # $0.48 < бюджету $2, тобто формально все гаразд, а фактично тисяча
    # сторінок коштувала $0.976 при стелі $0.20. На довгій черзі цих самих
    # грошей вистачило б на десяток справ.
    #
    # Стеля береться ТА САМА, що при доборі оффера (`need.max_usd_per_1000_pages`
    # уже враховує тір деградації) — інакше ворота відкидали б рівно те, що
    # добір свідомо дозволив на порожньому ринку.
    ceiling = need.usd_per_1000_ceiling
    per_traffic = traffic_per_1000(offer, need)
    per_1000 = 1000.0 * offer_dph(offer) / sizing.pages_per_hour + per_traffic
    base_per_1000 = 1000.0 * offer_dph(offer) / base_pph + per_traffic
    if (slowdown > 1 and ceiling > 0 and per_1000 > ceiling * PRICE_TOLERANCE
            and base_per_1000 <= ceiling * PRICE_TOLERANCE):
        return GateResult(
            False, "slow_for_data",
            f"{net_why}${per_1000:.3f} за 1000 стор. при стелі ${ceiling:.2f}",
            measured, sizing, hours, cost,
        )
    if slowdown > 1 and (hours > need.max_hours >= base_hours
                         or cost > need.budget_usd >= base_cost):
        return GateResult(
            False, "slow_for_data",
            f"{net_why}{hours:.1f} год / ${cost:.2f} при стелі {need.max_hours:.1f} год / "
            f"бюджеті ${need.budget_usd:.2f}",
            measured, sizing, hours, cost,
        )
    # 🔴🔴 ЦІЛЬ НА ВИМІРЯНОМУ ЗАЛІЗІ. Вибір перевіряв ціль за КАРТКОЮ
    # оффера; тут те саме число рахується з квоти cgroup і вільної VRAM кожної
    # карти. Машина, що на ділі не дає цілі, гаситься тут — за хвилину оренди,
    # а не після години слабкого заходу.
    if need.target_pph > 0 and sizing.pages_per_hour * GUARANTEE_FACTOR < need.target_here:
        # Ціль зрізав канал, а не залізо: це вирок мережі, а не заліза.
        verdict = ("slow_for_data" if slowdown > 1
                   and base_pph * GUARANTEE_FACTOR >= need.target_here else "below_target")
        return GateResult(
            False, verdict,
            f"{net_why}на виміряному залізі гарантовано "
            f"{sizing.pages_per_hour * GUARANTEE_FACTOR:.0f} стор/год < цілі "
            f"{need.target_here:.0f} ({sizing.shards} шардів, {cores:.0f} ядер квоти, "
            f"{per_card:.1f} ГБ вільно на карту)",
            measured, sizing, hours, cost,
        )
    if ceiling > 0 and per_1000 > ceiling * PRICE_TOLERANCE:
        # Причина розділяється, бо вироки різні за суттю: карту тримає сусід
        # (хост справний, забути через 3 дні) — це не те саме, що машина, яка
        # просто повільна за свої гроші.
        per_card_total = vram_total / max(1, int(measured["n_gpus"]))
        per_card_free = float(measured.get("vram_free_min_gb") or vram_free)
        busy = per_card_total > 0 and per_card_free < per_card_total * CARD_MINE_FRACTION
        why = (
            f"вільно {per_card_free:.1f} з {per_card_total:.1f} ГБ карти — її тримає "
            f"чужий орендар; " if busy else ""
        )
        return GateResult(
            False, "card_busy" if busy else "overpriced",
            f"{why}${per_1000:.3f} за 1000 стор. — це {per_1000 / ceiling:.1f}× "
            f"стелі ${ceiling:.2f} ({sizing.pages_per_hour:.0f} стор/год на "
            f"${offer_dph(offer):.3f}/год, {sizing.shards} шардів)",
            measured, sizing, hours, cost,
        )
    if hours > need.max_hours:
        return GateResult(
            False, "cpu_lie" if sizing.limited_by == "cpu" else "vram_lie",
            f"на виміряному залізі це {hours:.1f} год при стелі {need.max_hours:.1f} "
            f"({sizing.shards} шардів, {sizing.pages_per_hour:.0f} стор/год)",
            measured, sizing, hours, cost,
        )
    if cost > need.budget_usd:
        return GateResult(
            False, "cpu_lie" if sizing.limited_by == "cpu" else "vram_lie",
            f"на виміряному залізі це ${cost:.2f} при бюджеті ${need.budget_usd:.2f}",
            measured, sizing, hours, cost,
        )

    cores_note = (
        f"{cores:.0f} ядер (видно {measured['cores_all']:.0f})"
        if measured["cores_all"] and measured["cores_all"] > cores + 0.5
        else f"{cores:.0f} ядер"
    )
    detail = (
        f"{cores_note} · {vram_free:.1f} з {vram_total:.1f} ГБ вільно · "
        f"{net:.0f} Мбіт/с → {sizing.shards} шардів × {sizing.threads_per_shard} потоків, "
        f"{sizing.pages_per_hour:.0f} стор/год, {hours:.1f} год, ${cost:.2f} "
        f"(${per_1000:.3f}/1000 стор)"
    )
    if measured["ram_limit_gb"]:
        detail += f"; RAM контейнера {measured['ram_limit_gb']:.0f} ГБ"
    if sizing.wasted_cores >= 8:
        detail += f"; простоює {sizing.wasted_cores:.0f} ядер (обмежувач — {sizing.limited_by})"
    if slowdown > 1:
        detail += f"; {net_why.rstrip('; ')}"
    return GateResult(True, "ok", detail, measured, sizing, hours, cost)


def _measured(probe: dict[str, Any]) -> dict[str, Any]:
    """Нормалізувати сирий вивід проби. Відсутнє число — нуль, а не виняток."""

    def num(key: str) -> float:
        try:
            return float(probe.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    net_raw = probe.get("net_bps")
    disk_raw = probe.get("disk_free_gb")
    # 🔴 Квота контейнера, а не хост. `nproc` і `free -g` показують МАШИНУ:
    # 192 ядра там, де продано 48, і 251 ГБ там, де продано 62.9. Квоту знає
    # лише cgroup, і нуль тут означає «не прочитали», а не «нуль ядер», тож
    # відсутнє значення просто не бере участі в `min`.
    quota = num("cores_quota")
    nproc = num("cores")
    cores_seen = min(nproc, quota) if quota > 0 else nproc
    return {
        "disk_measured": disk_raw not in (None, ""),
        "cores": num("cores"),
        "cores_all": num("cores_all") or nproc,
        "cores_quota": quota or None,
        "cores_seen": cores_seen,
        "ram_gb": num("ram_gb"),
        "ram_limit_gb": num("ram_limit_gb") or None,
        "gpu": str(probe.get("gpu") or ""),
        "vram_total_gb": num("vram_total_gb") or num("vram_total_mb") / 1024.0,
        "vram_free_gb": num("vram_free_gb") or num("vram_free_mb") / 1024.0,
        "disk_free_gb": num("disk_free_gb"),
        # Скільки карт бачить контейнер і скільки вільно на НАЙМЕНШ вільній.
        # Шарди розкладаються по всіх картах, тож бюджет — сумарний, але
        # жодна окрема карта не має бути забита сусідом.
        "n_gpus": max(1, int(num("n_gpus") or 1)),
        "vram_free_min_gb": (num("vram_free_min_mb") / 1024.0) or None,
        "net_mbps": num("net_mbps") or num("net_bps") * 8 / 1_000_000,
        # 🔴 Друге число каналу: той самий обсяг ВІСЬМОМА діапазонами — тобто
        # так, як качає сам транспорт. Нуль означає «паралельно не міряли»
        # (стара проба), і тоді рішення лишається на однопотоковому.
        "net_par_mbps": num("net_par_mbps") or num("net_par_bps") * 8 / 1_000_000,
        # Порожній рядок = проби не було (не дали URL). Нуль = міряли й нуль:
        # 🔴 різниця критична, інакше «не міряли» читається як «мертвий канал»
        # і ми банимо здорову машину.
        "net_measured": net_raw not in (None, ""),
        "net_http": probe.get("net_http") or "",
        "py": str(probe.get("py") or ""),
    }
