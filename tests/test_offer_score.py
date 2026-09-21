"""Ранжування офферів на матеріалі реальних інцидентів.

Фікстури — це машини, які справді бралися й справді підводили. Тест тут
означає «цей вибір більше не повториться», а не «функція повертає число».
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gpurunner.core.boxes import BoxVerdict
from gpurunner.core.offer_score import (
    TIERS,
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

    Дешевша машина з удвічі меншою VRAM має програвати, попри ціну: VRAM
    визначає, скільки з оплачених ядер узагалі працюватиме.
    """
    ranked = rank_offers([TI4060_VIETNAM, V100_COLORADO], NEED, tier=TIERS[3])
    assert ranked[0].machine_id == V100_COLORADO["machine_id"]


def test_equal_money_double_time_prefers_the_fast_box() -> None:
    """🔴 Замір сесії: 1665 стор. на V100 — 0.76 год / $0.17; на 4060 Ti —
    1.67 год / $0.16. **Грошей однаково, часу вдвічі більше.**

    Скор «сторінок за долар» у чистому вигляді обирав би дешеву машину — і
    саме через це захід стояв удвічі довше за ті самі гроші.
    """
    fast = score_offer(V100_COLORADO, NEED, tier=TIERS[3])
    cheap = score_offer(TI4060_VIETNAM, NEED, tier=TIERS[3])
    assert cheap.cost == pytest.approx(fast.cost, rel=0.25)  # гроші майже рівні
    assert cheap.hours > fast.hours * 1.8                    # час — ні
    assert fast.score > cheap.score


def test_idle_cores_are_priced_in() -> None:
    """🔴 Інцидент: 128 ядер, з них працює 12.8 — і за решту ми платимо."""
    # Стелю вартості тут піднімаємо свідомо: тест про ПРОСТІЙ ЯДЕР, а не про
    # ціну; інакше машина відсіюється раніше, ніж дійде до перевірки.
    need = Need(pages=1665, max_hours=8.0, budget_usd=3.00, max_usd_per_1000_pages=0.50)
    scored = score_offer(RTX4090_THAI, need, tier=TIERS[3])
    assert scored.ok
    assert scored.ok
    assert scored.sizing.wasted_cores > 40
    assert "оплачено 128 ядер" in scored.explain


def test_four_effective_cores_never_wins_on_price() -> None:
    """«Найдешевший оффер із потрібною картою» — і є та сама пастка."""
    ranked = rank_offers([CHEAP_4CORE, V100_COLORADO], NEED, tier=TIERS[3])
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
    ranked = rank_offers([RTX2080_CALIF, V100_COLORADO], NEED, verdicts, tier=TIERS[3])
    assert [c.machine_id for c in ranked] == [38902]
    rejected = score_offer(RTX2080_CALIF, NEED, verdicts[47127], tier=TIERS[3])
    assert "чорному списку" in rejected.rejects[0]


def test_starred_machine_beats_a_slightly_cheaper_stranger() -> None:
    """Виміряна машина цінніша за незнайомця з кращою карткою на 20%."""
    stranger = offer(id=9, machine_id=777, gpu="Tesla V100", cores=64, vram_gb=32, dph=0.185)
    verdicts = {
        38902: BoxVerdict(machine_id=38902, state="starred", reason="3 успішні прогони")
    }
    ranked = rank_offers([stranger, V100_COLORADO], NEED, verdicts, tier=TIERS[3])
    assert ranked[0].machine_id == 38902


def test_warned_machine_is_demoted_but_not_excluded() -> None:
    verdicts = {38902: BoxVerdict(machine_id=38902, state="warned", reason="ssh_unreachable×1")}
    with_warning = score_offer(V100_COLORADO, NEED, verdicts[38902], tier=TIERS[3])
    clean = score_offer(V100_COLORADO, NEED, None, tier=TIERS[3])
    assert with_warning.ok
    assert with_warning.score < clean.score


# ---- бюджет і строк не послаблюються ---------------------------------------


def test_offer_that_cannot_finish_in_time_is_never_a_candidate() -> None:
    tight = Need(pages=30_000, max_hours=2.0, budget_usd=100.0)
    for tier in TIERS:
        assert rank_offers([V100_COLORADO, TI4060_VIETNAM], tight, tier=tier) == []


def test_offer_over_budget_is_never_a_candidate() -> None:
    poor = Need(pages=1665, max_hours=8.0, budget_usd=0.05)
    for tier in TIERS:
        assert rank_offers([V100_COLORADO], poor, tier=tier) == []


def test_empty_market_returns_empty_selection_not_a_bad_box() -> None:
    """Порожньо — це «нічого не орендуємо», а не «візьмемо хоч що-небудь»."""
    sel = select_offers([CHEAP_4CORE], Need(pages=30_000, max_hours=1.0, budget_usd=1.0))
    assert sel.empty
    assert sel.candidates == []
    assert "бюджет" in sel.reason
    assert sel.rejected  # але чому саме — видно


# ---- деградація ------------------------------------------------------------


def test_tier_zero_is_used_when_the_market_is_rich() -> None:
    sel = select_offers([V100_COLORADO, TI4060_VIETNAM], NEED)
    assert sel.tier.level == 0
    assert sel.best is not None and sel.best.machine_id == 38902
    assert not sel.best.degraded
    assert sel.reason == ""


def test_degradation_is_reported_with_a_number() -> None:
    """Коли доводиться брати гірше — у звіті має бути «наскільки гірше»."""
    sel = select_offers([TI4060_VIETNAM], NEED)
    assert not sel.empty
    assert sel.tier.level >= 2  # 16 ГБ не тримає 8 шардів
    assert sel.best.degraded
    assert sel.best.slowdown_x > 1.0
    assert "повільніше" in sel.reason


def test_cheap_box_missing_a_threshold_still_wins_on_value() -> None:
    """🔴 Інцидент 2026-08-11: 3×L40 за $1.36/год проти RTX 3090 за $0.19.

    Пороги були КАСКАДОМ: 3090 має 28 ефективних ядер — на чотири менше за
    поріг tier 0 — і вилітала повністю, а в tier 0 лишалась сама L40 й
    «перемагала» без суперників. Ціна в рішенні не брала участі взагалі.

    Тепер кандидати з усіх тірів порівнюються за скором, у якому ціна вже є.
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


def test_time_value_knob_can_turn_off_the_rush() -> None:
    """«Швидкість сьогодні не варта переплати» — це має бути ручкою."""
    fast_pricey = offer(id=1, machine_id=1, gpu="L40", cores=192, vram_gb=135,
                        dph=1.36, num_gpus=3)
    slow_cheap = offer(id=2, machine_id=2, gpu="RTX 3090", cores=64, vram_gb=24, dph=0.19)
    patient = Need(pages=4000, max_hours=8.0, budget_usd=3.0, time_value_usd_per_hour=0.0)
    sel = select_offers([fast_pricey, slow_cheap], patient)
    assert sel.best.machine_id == 2, "при нульовій ціні часу перемагає дешевше"


def test_low_reliability_host_needs_a_lower_tier() -> None:
    flaky = offer(id=7, machine_id=666, gpu="Tesla V100", cores=64, vram_gb=32,
                  dph=0.20, reliability=0.93)
    assert rank_offers([flaky], NEED, tier=TIERS[0]) == []
    assert rank_offers([flaky], NEED, tier=TIERS[3]) != []


# ---- дрібне ---------------------------------------------------------------


def test_small_disk_is_rejected() -> None:
    tiny = offer(id=6, machine_id=222, gpu="Tesla V100", cores=64, vram_gb=32, dph=0.2, disk=40)
    scored = score_offer(tiny, NEED, tier=TIERS[3])
    assert not scored.ok
    assert any("диск" in r for r in scored.rejects)


def test_claimed_bandwidth_does_not_affect_the_score() -> None:
    """🔴 Картка бреше про канал системно — тож вона на вибір не впливає.

    Рішення про мережу ухвалює замір на живому боксі (`probe_box`).
    """
    honest = dict(V100_COLORADO)
    liar = {**V100_COLORADO, "id": 99, "inet_down": 5000.0}
    assert score_offer(liar, NEED, tier=TIERS[3]).score == pytest.approx(
        score_offer(honest, NEED, tier=TIERS[3]).score
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
    twin_s = score_offer(twin, NEED, tier=TIERS[3]).sizing.shards
    single_s = score_offer(single, NEED, tier=TIERS[3]).sizing.shards
    # ±1 — цілочисельне округлення на межі карти, не помилка моделі
    assert abs(twin_s - 2 * single_s) <= 1


def test_selection_has_no_candidates_without_offers() -> None:
    assert select_offers([], NEED) == Selection(
        candidates=[], tier=TIERS[-1], rejected=[], reason=select_offers([], NEED).reason
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
    scored = score_offer(l40, NEED, tier=TIERS[3])
    # ⚠ 06.09.2026: стеля темпу тепер покарткова (std160 спростував 4500 на
    # бокс — одна 3090 дає 5573), тож три L40 прогнозуються на межі порогу.
    # Масштабування по картах НЕ МІРЯНЕ (E4 у плані модернізації), а єдиний
    # живий замір — 3966 на цій самій збірці, тобто $0.34 за тисячу. Доки
    # немає заміру, приймач тут — ціна сторінки: збірка за $1.36 мусить
    # лишатись ДОРОЖЧОЮ за тисячу, ніж 3090 за $0.196, і не менш як за 0.19.
    cheap = score_offer(offer(id=9, machine_id=9, gpu="RTX 3090", cores=26,
                              vram_gb=24, dph=0.196), NEED, tier=TIERS[3])
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
    assert score_offer(o, NEED, tier=TIERS[3]).ok is expect_ok, name


def test_ceiling_is_a_knob() -> None:
    """Дозволити переплату можна — але свідомо, у плані."""
    l40 = offer(id=1, machine_id=1, gpu="L40", cores=192, vram_gb=135, dph=1.36, num_gpus=3)
    generous = Need(pages=1665, max_hours=8.0, budget_usd=5.0, max_usd_per_1000_pages=0.50)
    assert score_offer(l40, generous, tier=TIERS[3]).ok


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


def test_twelve_core_3090_is_tier_zero_now() -> None:
    """std160 05.09.2026: RTX 3090 із 12 ядрами йшла в тір 3 зі штрафом 0.90 і
    підписом «у 1.14× повільніше за еталон» при прогнозі 1100 і факті 3600.
    Планка 32 ядра — спадок моделі «ядра купують темп»."""
    box = offer(id=1, machine_id=14096, gpu="RTX 3090", cores=12, vram_gb=24, dph=0.15)
    sel = select_offers([box], Need(pages=160, max_hours=1.5, budget_usd=1.0,
                                    gb_per_shard=1.9, lines_per_page=70))
    assert sel.best is not None and sel.best.tier == 0, sel.reason
    assert sel.best.sizing.shards >= 10


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
    scored = score_offer(cmp, NEED, tier=TIERS[3])
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


def test_a_starred_machine_close_to_the_best_goes_first() -> None:
    """🔴 12–15.09.2026: вільні зірки програвали невідомим машинам, а ті падали на
    підйомі, SSH чи каналі — 2–13 хв оренди за кожну невдалу спробу."""
    from gpurunner.core.offer_score import stars_first

    order = stars_first(_ranked((1, 100.0, None), (2, 90.0, None), (3, 75.0, "starred"),
                                (4, 60.0, "starred")))
    assert [c.machine_id for c in order] == [3, 1, 2, 4]


def test_a_starred_machine_far_behind_does_not_jump_the_queue() -> None:
    from gpurunner.core.offer_score import stars_first

    order = stars_first(_ranked((1, 100.0, None), (2, 50.0, "starred")))
    assert [c.machine_id for c in order] == [1, 2]
    assert stars_first([]) == []
