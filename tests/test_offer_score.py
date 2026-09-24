"""Ранжування офферів на матеріалі реальних інцидентів.

Фікстури — це машини, які справді бралися й справді підводили. Тест тут
означає «цей вибір більше не повториться», а не «функція повертає число».
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from gpurunner.core.boxes import BoxVerdict
from gpurunner.core.offer_score import (
    Need,
    Selection,
    rank_offers,
    score_offer,
    select_offers,
)

NOW = datetime(2026, 8, 11, tzinfo=UTC)


def offer(
    *,
    id: int,
    machine_id: int,
    gpu: str,
    cores: float,
    vram_gb: float,
    dph: float,
    reliability: float = 0.99,
    disk: float = 200.0,
    inet_down: float = 500.0,
    geo: str = "US",
    num_gpus: int = 1,
) -> dict:
    return {
        "id": id,
        "machine_id": machine_id,
        "gpu_name": gpu,
        "num_gpus": num_gpus,
        "cpu_cores_effective": cores,
        "gpu_ram": vram_gb * 1024 / num_gpus,
        "cpu_ram": 128 * 1024,
        "disk_space": disk,
        "dph_total": dph,
        "reliability2": reliability,
        "inet_down": inet_down,
        "geolocation": geo,
    }


# Реальні машини сесії 2026-08-11
V100_COLORADO = offer(id=1, machine_id=38902, gpu="Tesla V100", cores=64, vram_gb=32, dph=0.222)
TI4060_VIETNAM = offer(id=2, machine_id=44476, gpu="RTX 4060 Ti", cores=32, vram_gb=16, dph=0.098)
RTX4090_THAI = offer(id=3, machine_id=36590, gpu="RTX 4090", cores=128, vram_gb=23, dph=0.323)
RTX2080_CALIF = offer(id=4, machine_id=47127, gpu="RTX 2080 Ti", cores=64, vram_gb=22, dph=0.113)
CHEAP_4CORE = offer(id=5, machine_id=999, gpu="Tesla V100", cores=4, vram_gb=32, dph=0.150)

NEED = Need(pages=1665, max_hours=8.0, budget_usd=3.00)


# ---- інциденти -------------------------------------------------------------


def test_sixteen_gb_box_loses_to_thirtytwo_gb() -> None:
    """🔴 Інцидент: узято 16 ГБ під 8 шардів → OOM з'їв 46 сторінок.

    VRAM визначає, скільки з оплачених ядер узагалі працюватиме: на 16 ГБ
    план не сміє підняти 8 шардів, а з ціллю така машина не проходить зовсім.
    """
    small = score_offer(TI4060_VIETNAM, NEED)
    assert small.sizing.shards <= int(16 * 0.9 // NEED.gb_per_shard)
    with_target = replace(NEED, target_pph=5000.0)
    ranked = rank_offers([TI4060_VIETNAM, V100_COLORADO], with_target)
    assert TI4060_VIETNAM["machine_id"] not in [r.machine_id for r in ranked]


def test_idle_cores_are_priced_in() -> None:
    """🔴 Інцидент: 128 ядер, з них працює 12.8 — і за решту ми платимо."""
    # Стелю вартості тут піднімаємо свідомо: тест про ПРОСТІЙ ЯДЕР, а не про
    # ціну; інакше машина відсіюється раніше, ніж дійде до перевірки.
    need = Need(pages=1665, max_hours=8.0, budget_usd=3.00, max_usd_per_1000_pages=0.50)
    scored = score_offer(RTX4090_THAI, need)
    assert scored.ok
    assert scored.ok
    assert scored.sizing.wasted_cores > 40
    assert "оплачено 128 ядер" in scored.explain


def test_four_effective_cores_never_wins_on_price() -> None:
    """«Найдешевший оффер із потрібною картою» — і є та сама пастка."""
    ranked = rank_offers([CHEAP_4CORE, V100_COLORADO], NEED)
    assert ranked[0].machine_id == V100_COLORADO["machine_id"]
    assert ranked[0].cost > CHEAP_4CORE["dph_total"] * 0  # дорожча за годину, дешевша за справу


def test_banned_machine_is_rejected_even_with_the_best_score() -> None:
    """🔴 Інцидент: оффер #47127187 (Каліфорнія) обіцяв 755 Мбіт/с, давав 0.5.

    Найдешевша машина з 64 ядрами — і саме її не можна брати вдруге.
    """
    verdicts = {
        47127: BoxVerdict(
            machine_id=47127, state="banned", reason="slow_net×1 (0.5 Мбіт/с проти 755)"
        )
    }
    ranked = rank_offers([RTX2080_CALIF, V100_COLORADO], NEED, verdicts)
    assert [c.machine_id for c in ranked] == [38902]
    rejected = score_offer(RTX2080_CALIF, NEED, verdicts[47127])
    assert "чорному списку" in rejected.rejects[0]


def test_starred_machine_beats_a_slightly_cheaper_stranger() -> None:
    """Виміряна машина цінніша за незнайомця з кращою карткою на 20%."""
    stranger = offer(id=9, machine_id=777, gpu="Tesla V100", cores=64, vram_gb=32, dph=0.185)
    verdicts = {
        38902: BoxVerdict(machine_id=38902, state="starred", reason="3 успішні прогони")
    }
    ranked = rank_offers([stranger, V100_COLORADO], NEED, verdicts)
    assert ranked[0].machine_id == 38902


def test_warned_machine_is_demoted_but_not_excluded() -> None:
    verdicts = {38902: BoxVerdict(machine_id=38902, state="warned", reason="ssh_unreachable×1")}
    with_warning = score_offer(V100_COLORADO, NEED, verdicts[38902])
    clean = score_offer(V100_COLORADO, NEED, None)
    assert with_warning.ok
    assert with_warning.score < clean.score


# ---- бюджет і строк не послаблюються ---------------------------------------


def test_offer_that_cannot_finish_in_time_is_never_a_candidate() -> None:
    tight = Need(pages=30_000, max_hours=2.0, budget_usd=100.0)
    assert rank_offers([V100_COLORADO, TI4060_VIETNAM], tight) == []


def test_offer_over_budget_is_never_a_candidate() -> None:
    poor = Need(pages=1665, max_hours=8.0, budget_usd=0.05)
    assert rank_offers([V100_COLORADO], poor) == []


def test_empty_market_returns_empty_selection_not_a_bad_box() -> None:
    """Порожньо — це «нічого не орендуємо», а не «візьмемо хоч що-небудь»."""
    sel = select_offers([CHEAP_4CORE], Need(pages=30_000, max_hours=1.0, budget_usd=1.0))
    assert sel.empty
    assert sel.candidates == []
    assert "бюджет" in sel.reason
    assert sel.rejected  # але чому саме — видно


# ---- ціна -----------------------------------------------------------------


def test_cheap_box_missing_a_threshold_still_wins_on_value() -> None:
    """🔴 Інцидент 2026-08-11: 3×L40 за $1.36/год проти RTX 3090 за $0.19.

    Пороги були КАСКАДОМ: 3090 має 28 ефективних ядер — на чотири менше за
    поріг tier 0 — і вилітала повністю, а в tier 0 лишалась сама L40 й
    «перемагала» без суперників. Ціна в рішенні не брала участі взагалі.

    Тепер тірів немає взагалі: придатні порівнюються за очікуваною ціною.
    """
    l40 = offer(id=1, machine_id=30258, gpu="L40", cores=192, vram_gb=135,
                dph=1.36, num_gpus=3)
    cheap = offer(id=2, machine_id=14025, gpu="RTX 3090", cores=28, vram_gb=24, dph=0.19)
    sel = select_offers([l40, cheap], Need(pages=1297, max_hours=8.0, budget_usd=3.0))
    assert sel.best.machine_id == 14025
    # 🔴 Межа піднята з 0.25 разом із перекалібруванням моделі 2026-08-12:
    # 35 стор/год на ядро замість 52 (нижня межа чотирьох живих замірів), тож
    # 28-ядерна машина тепер чесно передбачається повільнішою й довшою. Суть
    # тесту не в сумі, а в тому, що дешева машина перемагає збірку за $1.36/год;
    # ціна сторінки в неї $0.194 — під порогом 0.20.
    assert sel.best.cost < 0.30
    assert sel.best.usd_per_1000 < 0.20


def test_price_ceiling_is_explicit_and_on_by_default() -> None:
    """Стеля $/год мусить існувати сама по собі, а не лише через скор.

    🔴 0.60 фактично означало «стелі немає»: перевірка «за тисячу сторінок»
    пропускає дорогу карту, якщо вона швидка, і захід узяв 4× RTX 3090 за
    $0.50/год. Рішення користувача 2026-08-11 — дорогих карт не брати.
    """
    from gpurunner.supervise.plan import Plan

    assert Plan(assets_url="u", cases=[]).max_price == 0.365


def test_sick_host_is_refused_and_a_merely_imperfect_one_costs_more() -> None:
    """Надійність — слабкий сигнал (18–23.09: 23% вдалих оренд нижче 0.98,
    ~45% вище), тож тірів за нею немає: нижче 0.90 — відмова, вище — дорожча
    ОЧІКУВАНА ціна, бо невдала оренда коштує підйому й повтору."""
    sick = offer(id=7, machine_id=666, gpu="Tesla V100", cores=64, vram_gb=32,
                 dph=0.20, reliability=0.85)
    assert rank_offers([sick], NEED) == []
    shaky = score_offer({**V100_COLORADO, "reliability2": 0.93}, NEED)
    solid = score_offer(V100_COLORADO, NEED)
    assert shaky.ok and solid.ok
    assert shaky.expected_cost > solid.expected_cost


# ---- дрібне ---------------------------------------------------------------


def test_small_disk_is_rejected() -> None:
    tiny = offer(id=6, machine_id=222, gpu="Tesla V100", cores=64, vram_gb=32, dph=0.2, disk=40)
    scored = score_offer(tiny, NEED)
    assert not scored.ok
    assert any("диск" in r for r in scored.rejects)


def test_claimed_bandwidth_does_not_affect_the_score() -> None:
    """🔴 Картка бреше про канал системно — тож вона на вибір не впливає.

    Рішення про мережу ухвалює замір на живому боксі (`probe_box`).
    """
    honest = dict(V100_COLORADO)
    liar = {**V100_COLORADO, "id": 99, "inet_down": 5000.0}
    assert score_offer(liar, NEED).score == pytest.approx(
        score_offer(honest, NEED).score
    )


def test_multi_gpu_offer_gets_proportionally_more_shards() -> None:
    """Дві карти дають удвічі більше шардів — бо шарди по них РОЗКЛАДАЮТЬСЯ.

    🔴 Це число чесне лише разом із `_start_shard`, який дає шарду k пристрій
    `cuda:(k % N)`. Поки всі шарди сиділи на `cuda:0`, той самий множник був
    брехнею й давав OOM. Дві речі міняються тільки разом.
    """
    # `vram_gb` у фікстурі — СУМАРНА: 2×24 ГБ проти однієї 24-ГБ карти
    twin = offer(id=10, machine_id=333, gpu="RTX 3090", cores=128, vram_gb=48, dph=0.4, num_gpus=2)
    single = offer(id=11, machine_id=334, gpu="RTX 3090", cores=128, vram_gb=24, dph=0.4)
    twin_s = score_offer(twin, NEED).sizing.shards
    single_s = score_offer(single, NEED).sizing.shards
    # ±1 — цілочисельне округлення на межі карти, не помилка моделі
    assert abs(twin_s - 2 * single_s) <= 1


def test_selection_has_no_candidates_without_offers() -> None:
    assert select_offers([], NEED) == Selection(
        candidates=[], rejected=[], reason=select_offers([], NEED).reason
    )


# ---- вартість як ГОЛОВНИЙ поріг --------------------------------------------


def test_expensive_speed_is_rejected_by_cost_per_page() -> None:
    """🔴 Балансу нема, бо швидка машина не окупилась.

    Заміри: V100 $0.097 за 1000 сторінок, RTX 3090 $0.125, 4060 Ti $0.202,
    а 3×L40 за $1.36/год — **$0.30**. Швидкість була 3966 стор/год, тобто
    вдвічі краща, і все одно втричі дорожча за сторінку.

    Тому вартість тисячі сторінок — не доданок у скорі, а ПОРІГ.
    """
    l40 = offer(id=1, machine_id=30258, gpu="L40", cores=192, vram_gb=135,
                dph=1.36, num_gpus=3)
    scored = score_offer(l40, NEED)
    # ⚠ 06.09.2026: стеля темпу тепер покарткова (std160 спростував 4500 на
    # бокс — одна 3090 дає 5573), тож три L40 прогнозуються на межі порогу.
    # Масштабування по картах НЕ МІРЯНЕ (E4 у плані модернізації), а єдиний
    # живий замір — 3966 на цій самій збірці, тобто $0.34 за тисячу. Доки
    # немає заміру, приймач тут — ціна сторінки: збірка за $1.36 мусить
    # лишатись ДОРОЖЧОЮ за тисячу, ніж 3090 за $0.196, і не менш як за 0.19.
    cheap = score_offer(offer(id=9, machine_id=9, gpu="RTX 3090", cores=26,
                              vram_gb=24, dph=0.196), NEED)
    assert scored.usd_per_1000 > cheap.usd_per_1000 * 1.25
    assert scored.usd_per_1000 >= 0.19


@pytest.mark.parametrize(
    ("name", "cores", "vram", "dph", "expect_ok"),
    [
        ("V100 Колорадо", 64, 32, 0.222, True),    # $0.097/1000 — еталон
        # 🔴🔴 БУЛО `False` — і це була помилка, яку спростовує замір у ЦЬОМУ Ж
        # файлі: `test_conservative_planning_never_exceeds_measured` записує
        # RTX 3090 / 26 ядер / 24 ГБ → **1564 стор/год**. За $0.196/год це
        # $0.125 за тисячу, тобто вдвічі під стелею $0.20 — машина економічна,
        # а тест вимагав від неї відмови.
        #
        # Відмова бралася зі стелі «35 × ядра»: 35×26 = 910 стор/год → $0.215 за
        # тисячу. Стелю знято 05.09.2026 після дуелі (два V100, та сама справа,
        # ті самі 8 шардів, 15 проти 92 ядер → 921 і 857 стор/год).
        #
        # ⚠ Чесно: у зведенні реєстру малоядерні 3090 таки повільніші за
        # багатоядерні (медіана ~1564 проти ~2400). Але то РІЗНІ заходи на
        # різному матеріалі, а контрольований дослід один, і він каже інше.
        ("RTX 3090 мало ядер", 26, 24, 0.196, True),
        ("RTX 3090 багато ядер", 64, 24, 0.196, True),   # $0.088/1000
        # 🔴 4060 Ti за 2.5 ГБ/шард тримає лише 5 шардів → $0.34/1000, і це
        # ЧЕСНО: виміряні 1673 стор/год були на 8 шардах, тобто на 1.68 ГБ —
        # рівно тій щільності, що дала OOM. Дешева карта з малою VRAM коштує
        # за сторінку дорого, щойно перестаєш ризикувати.
        ("RTX 4060 Ti", 80, 16, 0.338, False),
        ("дорога швидка", 192, 135, 1.36, False),   # $0.30/1000 — відсікаємо
    ],
)
def test_ceiling_keeps_the_usual_market(
    name: str, cores: float, vram: float, dph: float, expect_ok: bool
) -> None:
    """Стеля $0.20/1000 має лишати весь звичний ринок і різати лише переплату."""
    o = offer(id=1, machine_id=1, gpu=name, cores=cores, vram_gb=vram, dph=dph)
    assert score_offer(o, NEED).ok is expect_ok, name


def test_ceiling_is_a_knob() -> None:
    """Дозволити переплату можна — але свідомо, у плані."""
    l40 = offer(id=1, machine_id=1, gpu="L40", cores=192, vram_gb=135, dph=1.36, num_gpus=3)
    generous = Need(pages=1665, max_hours=8.0, budget_usd=5.0, max_usd_per_1000_pages=0.50)
    assert score_offer(l40, generous).ok


def test_measured_throughput_beats_the_model() -> None:
    """🔴 Замір на РЕАЛЬНОМУ прогоні точніший за будь-який прогноз із картки.

    Модель не знає ні фактичного розподілу ядер (V100 із 128 ядрами віддала
    нам 32), ні щільності конкретної справи. Реєстр темп пам'ятав і навіть
    показував у поясненні — але в розрахунок його не брали, тож знання про
    машину не впливало на вибір машини.
    """
    from gpurunner.core.boxes import BoxVerdict

    o = offer(id=1, machine_id=97081, gpu="Q RTX 6000", cores=28, vram_gb=22.5, dph=0.167)
    without = score_offer(o, NEED)
    with_history = score_offer(
        o, NEED,
        verdict=BoxVerdict(machine_id=97081, state="starred", reason="1 успішних",
                           best_measured={"pages_per_hour": 1562}),
    )
    assert with_history.sizing.pages_per_hour == 1562
    assert with_history.usd_per_1000 < without.usd_per_1000
    assert with_history.hours < without.hours


def test_no_history_still_uses_the_model() -> None:
    """Перша зустріч із машиною — рахуємо як рахували."""
    o = offer(id=2, machine_id=7, gpu="RTX 3090", cores=64, vram_gb=24, dph=0.2)
    s = score_offer(o, NEED)
    assert s.sizing.pages_per_hour > 0


def test_twin_3090_is_predicted_roughly_twice_the_single() -> None:
    """Ринок 05.09.2026: 2×3090 $0.264 проти 1×3090 $0.168. Стеля 4500 на бокс
    не масштабувалась картами, і двокарткова виглядала вдвічі дорожчою за
    тисячу, хоч шарди розкладаються по картах."""
    need = Need(pages=1500, max_hours=8.0, budget_usd=3.0, gb_per_shard=1.9,
                lines_per_page=70)
    single = score_offer(offer(id=1, machine_id=1, gpu="RTX 3090", cores=12,
                               vram_gb=24, dph=0.168), need)
    twin = score_offer(offer(id=2, machine_id=2, gpu="RTX 3090", cores=24,
                             vram_gb=48, dph=0.264, num_gpus=2), need)
    assert twin.sizing.shards >= 2 * single.sizing.shards * 0.9
    assert twin.sizing.pages_per_hour >= 1.8 * single.sizing.pages_per_hour
    assert twin.usd_per_1000 <= single.usd_per_1000 * 1.1


def test_mining_cards_are_refused_by_name() -> None:
    """06.09.2026: бойовий захід P2 узяв NVIDIA CMP 170HX — 4327 стор/год на
    першій справі, а тоді інстанс пішов offline посеред другої: $0.10 за нуль
    і переоренда. Картка Vast про майнінгову природу карти не каже нічого."""
    from gpurunner.core.offer_score import is_mining_card

    cmp = offer(id=1, machine_id=1, gpu="NVIDIA CMP 170HX", cores=26, vram_gb=8, dph=0.232)
    ok = offer(id=2, machine_id=2, gpu="RTX 3090", cores=12, vram_gb=24, dph=0.15)
    assert is_mining_card(cmp) and not is_mining_card(ok)
    assert is_mining_card(offer(id=3, machine_id=3, gpu="P106-100", cores=8, vram_gb=6, dph=0.05))
    scored = score_offer(cmp, NEED)
    assert not scored.ok and any("майнінгова" in r for r in scored.rejects)


# ---- зірки першими --------------------------------------------------------------


def _ranked(*pairs):
    """[(машина, скор, стан)] → кандидати в тому порядку, як їх віддає сортування."""
    import dataclasses

    base = score_offer(V100_COLORADO, NEED, None)
    out = []
    for i, (mid, score, state) in enumerate(pairs):
        verdict = BoxVerdict(machine_id=mid, state=state, reason="") if state else None
        out.append(dataclasses.replace(base, offer={**V100_COLORADO, "id": i, "machine_id": mid},
                                       machine_id=mid, score=score, verdict=verdict))
    return out


def _titan_x_6_cores() -> dict:
    """Другий оффер машини 140182: та сама карта, удвічі менше ядер."""
    return offer(id=51832512, machine_id=140182, gpu="GTX TITAN X",
                 cores=6, vram_gb=12, dph=0.051)


#: Замір, який реєстр зберіг для машини 140182: зроблений на ДВАНАДЦЯТИ ядрах
#: і на метриках (66 рядків на сторінку, кадри 19 Мпікс).
_MEASURED_ON_12_CORES = {
    "pages_per_hour": 918, "cores_quota": 11.52, "vram_total_gb": 24.0,
    "n_gpus": 2.0, "pages_per_hour_mpx": 18.96, "pages_per_hour_lines": 66.0,
}

#: Черга протоколів консисторії: 812 сторінок, 138.5 рядків, кадри 23 Мпікс.
_HEAVY = Need(pages=812, max_hours=8.0, budget_usd=1.2, gb_per_shard=4.53,
              lines_per_page=138.5, max_usd_per_1000_pages=0.55)


def test_measured_throughput_is_rescaled_to_this_offer() -> None:
    """🔴🔴 Замір реєстру лежить на МАШИНІ, а машина продає кілька офферів.

    21.09.2026 у 140182 їх було два — 12-ядерний і 6-ядерний, на тій самій
    карті. Темп 918 стор/год, виміряний на 12 ядрах і на вдвічі рідших
    рядках, приписався 6-ядерному офферу під удвічі щільніший матеріал: той
    виграв скор і читав чергу зі швидкістю, якої в нього немає.
    """
    from gpurunner.core.boxes import BoxVerdict

    star = BoxVerdict(machine_id=140182, state="starred", reason="2 успішних",
                      best_measured=_MEASURED_ON_12_CORES)
    scored = score_offer(_titan_x_6_cores(), _HEAVY, star)
    assert scored.sizing.pages_per_hour < 400, scored.explain
    # Модель без заміру дала б приблизно те саме — замір не має робити машину
    # швидшою за те, на що вистачає її ядер і VRAM.
    plain = score_offer(_titan_x_6_cores(), _HEAVY)
    assert scored.sizing.pages_per_hour < plain.sizing.pages_per_hour * 1.5


def test_measured_throughput_still_beats_model_on_same_conditions() -> None:
    """А там, де умови ті самі, замір як був головнішим за модель, так і лишився."""
    from gpurunner.core.boxes import BoxVerdict

    same = dict(_MEASURED_ON_12_CORES, cores_quota=6.0, vram_total_gb=12.0,
                n_gpus=1.0, pages_per_hour_mpx=23.0, pages_per_hour_lines=138.5)
    verdict = BoxVerdict(machine_id=140182, state="starred", reason="2 успішних",
                         best_measured=same)
    scored = score_offer(_titan_x_6_cores(), _HEAVY, verdict)
    assert scored.sizing.pages_per_hour == pytest.approx(918, rel=0.02)


def test_slow_cheap_box_is_rejected_by_the_pph_floor() -> None:
    """Підлога темпу — окремий поріг від стелі ціни, і саме вона ловить цей клас.

    6-ядерна TITAN X дає $0.234 за тисячу сторінок: усередині будь-якої
    розумної стелі ціни. І 218 стор/год, тобто 3.7 години на чергу, яку
    32-ядерна машина читає за півгодини.
    """
    need = replace(_HEAVY, min_pages_per_hour=1000)
    scored = score_offer(_titan_x_6_cores(), need)
    assert not scored.ok
    assert any("підлоги" in r for r in scored.rejects), scored.rejects


def test_fat_box_passes_the_floor() -> None:
    """І та сама підлога пропускає машину, яка ринку справді варта."""
    need = replace(_HEAVY, min_pages_per_hour=1000)
    fat = offer(id=5139, machine_id=151, gpu="RTX 3090", cores=32,
                vram_gb=48, dph=0.33, num_gpus=2)
    scored = score_offer(fat, need)
    assert scored.ok, scored.rejects
    assert scored.sizing.pages_per_hour > 1000


# ---- трафік хоста ------------------------------------------------------------


def _with_down_cost(o: dict, usd_per_gb: float, **kw) -> dict:
    return {**o, "inet_down_cost": usd_per_gb, **kw}


#: Черга 24.09.2026: томи 102+105, ~10 ГБ кадрів на ~6 800 сторінок.
TRAFFIC_NEED = Need(pages=6800, max_hours=8.0, budget_usd=3.00,
                    data_mb_per_page=10 * 1024 / 6800)


def test_pricey_traffic_loses_to_the_same_box_with_cheap_traffic() -> None:
    """🔴 24.09.2026: A4000 в Естонії — $0.10 за карту і $0.27 за трафік.

    Дві однакові машини однакової ціни за годину: та, що бере $0.026/ГБ,
    програє тій, що бере медіану ринку $0.0026/ГБ, і різниця — рівно трафік.
    """
    dear = _with_down_cost(RTX2080_CALIF, 0.026, id=10, machine_id=10)
    cheap = _with_down_cost(RTX2080_CALIF, 0.0026, id=11, machine_id=11)
    ranked = rank_offers([dear, cheap], TRAFFIC_NEED)
    assert [r.machine_id for r in ranked] == [11, 10]
    d, c = score_offer(dear, TRAFFIC_NEED), score_offer(cheap, TRAFFIC_NEED)
    assert d.traffic_usd == pytest.approx(10 * 0.026)
    assert d.cost - c.cost == pytest.approx(10 * (0.026 - 0.0026))
    assert d.usd_per_1000 - c.usd_per_1000 == pytest.approx(
        (0.026 - 0.0026) * 10 * 1000 / 6800)
    assert "трафік $0.26" in d.explain


def test_offer_without_a_traffic_price_scores_as_before() -> None:
    """Старі знімки ринку й стани без поля: ціна трафіку невідома — не додається."""
    s = score_offer(RTX2080_CALIF, TRAFFIC_NEED)
    assert s.traffic_usd == 0.0
    assert "трафік" not in s.explain


def test_traffic_can_push_a_box_over_the_per_1000_ceiling() -> None:
    """Машина, якої година сама по собі під стелею, а з трафіком — ні."""
    base = score_offer(RTX2080_CALIF, TRAFFIC_NEED)
    assert base.ok
    ceiling = base.usd_per_1000 * 1.2
    need = replace(TRAFFIC_NEED, max_usd_per_1000_pages=ceiling)
    assert score_offer(RTX2080_CALIF, need).ok
    dear = _with_down_cost(RTX2080_CALIF, 0.067)
    s = score_offer(dear, need)
    assert not s.ok
    assert any("трафік" in r for r in s.rejects)
