"""Алгоритм вибору машини — на СПРАВЖНЬОМУ ринку, а не на фікстурах.

🔴 Навіщо окремий рід тестів. 23.09.2026 наглядач узяв бокс за $0.0270 за
ядро-годину тієї самої ночі, коли інший захід читав за $0.0152: на 11%
дешевше за годину й на 78% дорожче за роботу. Усі тести були зелені — бо в
них лежали ті оффери, про які вже подумали.

Знімок ринку приносить те, про що не подумали: у файлі `tests/data/market/`
розкид ціни ядро-години **36×** (від $0.0026 до $0.0938), машини з 96
«ядрами» й квотою в одиниці, карти Pascal за копійки, хости з нульовою
надійністю. Мережі тут немає — лише JSON, знятий `tools/market_snapshot.py`.

Перевіряємо не «обрано саме цю машину» (ринок змінюється щодня, такий тест
жив би добу), а ВЛАСТИВОСТІ вибору, які не мають права порушуватись ніколи.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpurunner.core.offer_score import (
    GUARANTEE_FACTOR,
    MAX_USD_PER_1000_WITH_TARGET,
    MAX_USD_PER_CORE_H,
    Need,
    select_offers,
)

MARKET_DIR = Path(__file__).resolve().parent / "data" / "market"


def _snapshots() -> list[Path]:
    return sorted(MARKET_DIR.glob("*.json"))


def _offers(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["offers"]


def _cores(o: dict[str, Any]) -> float:
    return float(o.get("cpu_cores_effective") or o.get("cpu_cores") or 0)


def _per_core_h(o: dict[str, Any]) -> float:
    c = _cores(o)
    return float(o.get("dph_total") or 0) / c if c else float("inf")


#: Матеріал ночі 23.09 (spr-43/spr-54): медіана 39–40 рядків, кадри 5.5 Мпікс.
LINES, MPX = 39.0, 5.5


def _need(**kw: Any) -> Need:
    from gpurunner.core.htr_sizing import gb_per_shard_for

    base: dict[str, Any] = dict(pages=5000, max_hours=8.0, budget_usd=3.0,
                                disk_gb=40, lines_per_page=LINES, frame_mpx=MPX,
                                gb_per_shard=gb_per_shard_for(MPX), target_pph=5000.0)
    base.update(kw)
    return Need(**base)


@pytest.fixture(params=_snapshots(), ids=lambda p: p.stem)
def market(request: pytest.FixtureRequest) -> list[dict[str, Any]]:
    """Один знімок ринку, уже без карт, під які немає коліс torch.

    🔴 Саме так робить бекенд: `find_candidates` відсіває за `min_compute_cap`
    ДО скору. Без цього стенд перевіряв би `select_offers` на офферах, яких
    той ніколи не бачить, — і перший же прогін «знайшов ваду», якої немає:
    алгоритм нібито взяв GTX TITAN X (Maxwell 5.2), де torch дає «CUDA error:
    no kernel image» і нуль сторінок за оплачений підйом.
    """
    return [o for o in _offers(request.param)
            if int(o.get("compute_cap") or 0) >= ORACLE_MIN_COMPUTE_CAP]


def test_the_snapshot_contains_machines_we_must_not_take(
        market: list[dict[str, Any]]) -> None:
    """Приймач самого знімка: у ньому мусять бути й ПОГАНІ машини.

    Знімок, у якому всі оффери прохідні, нічого не доводить — він перевіряв би
    ворота на матеріалі, де їм нема що ловити.
    """
    dear = [o for o in market if _per_core_h(o) > MAX_USD_PER_CORE_H]
    assert dear, "у знімку немає жодної машини, дорожчої за стелю ядро-години"
    assert len(dear) < len(market), "у знімку немає жодної прохідної машини"


def test_never_takes_a_box_dearer_per_core_hour_than_the_ceiling(
        market: list[dict[str, Any]]) -> None:
    """🔴🔴 Аксіома: недогодовану машину взяти не можна ні на якому тірі."""
    got = select_offers(market, _need())
    if got.empty:
        pytest.skip("ринок цього знімка порожній під цю потребу")
    per_core = _per_core_h(got.best.offer)
    assert per_core <= MAX_USD_PER_CORE_H, (
        f"обрано ${per_core:.4f} за ядро-годину при стелі ${MAX_USD_PER_CORE_H:.4f}: "
        f"{got.best.offer.get('gpu_name')}, {_cores(got.best.offer):.0f} ядер за "
        f"${got.best.offer.get('dph_total'):.3f}/год"
    )


def test_never_takes_a_slower_box_for_more_money(
        market: list[dict[str, Any]]) -> None:
    """🔴 Домінування — у ТЕМПІ, а не в ядрах.

    ⚠ Наївний критерій «більше ядер за меншу ціну» тут не працює, і перший же
    справжній знімок це показав: серед прохідних були 48 ядер за $0.144 проти
    обраних 36.6 за $0.208 — але в тих 48 ядер VRAM 22 ГБ проти 44, тобто
    вдвічі менше шардів. Флот обмежує `min(VRAM, ядра)`, тож «більше ядер»
    саме по собі не є більшою машиною.

    Порушенням є інше: оффер, який дає НЕ МЕНШИЙ прогноз темпу за НЕ БІЛЬШУ
    ціну за годину. Саме це сталось 23.09: 10 ядер за $0.259 проти 20 ядер за
    $0.292 — половина темпу за 89% ціни.
    """
    need = _need()
    got = select_offers(market, need)
    if got.empty:
        pytest.skip("ринок цього знімка порожній під цю потребу")
    pick = got.best
    p_pph, p_dph = pick.sizing.pages_per_hour, float(pick.offer.get("dph_total") or 0)

    better = [
        c for c in got.candidates
        if c.offer.get("id") != pick.offer.get("id")
        and c.sizing.pages_per_hour >= p_pph
        and float(c.offer.get("dph_total") or 0) <= p_dph
        and (c.sizing.pages_per_hour > p_pph
             or float(c.offer.get("dph_total") or 0) < p_dph)
    ]
    assert not better, (
        f"обрано {p_pph:.0f} стор/год за ${p_dph:.3f}/год, хоча серед прохідних "
        f"було "
        + "; ".join(f"{c.sizing.pages_per_hour:.0f} стор/год за "
                    f"${float(c.offer['dph_total']):.3f}/год" for c in better[:3])
    )


def test_the_same_market_gives_the_same_answer(
        market: list[dict[str, Any]]) -> None:
    """Вибір детермінований: інакше розбір «чому взяли цю» неможливий."""
    a = select_offers(market, _need())
    b = select_offers(list(reversed(market)), _need())
    if a.empty or b.empty:
        pytest.skip("ринок цього знімка порожній під цю потребу")
    assert (a.best.offer.get("id") == b.best.offer.get("id")), (
        "порядок офферів у відповіді ринку не має впливати на вибір"
    )


#: Дві машини тієї ночі — сирі числа з реєстру боксів, не вигадані.
NIGHT_23_09 = [
    {"id": 900001, "machine_id": 15167, "gpu_name": "Q RTX 8000", "num_gpus": 1,
     "gpu_ram": 46080, "cpu_cores": 80, "cpu_cores_effective": 19.2,
     "cpu_ram": 48000, "dph_total": 0.292, "disk_space": 200.0,
     "inet_down": 805.1, "inet_up": 854.0, "reliability2": 0.9976,
     "geolocation": "Indiana, US", "compute_cap": 750, "rentable": True},
    {"id": 900002, "machine_id": 99999, "gpu_name": "Q RTX 8000", "num_gpus": 1,
     "gpu_ram": 46080, "cpu_cores": 80, "cpu_cores_effective": 9.6,
     "cpu_ram": 48000, "dph_total": 0.259, "disk_space": 200.0,
     "inet_down": 805.1, "inet_up": 854.0, "reliability2": 0.9976,
     "geolocation": "Indiana, US", "compute_cap": 750, "rentable": True},
]


def test_the_night_of_23_09_would_now_pick_the_other_box() -> None:
    """🔴🔴 Той самий вибір, що коштував $1.19 замість $0.78.

    Обидві машини — Q RTX 8000, однакова карта, однакова надійність, однаковий
    канал. Різниця лише в квоті ядер і ціні за годину. Матеріал теж однаковий:
    щільність, заміряна по ВСІХ текстах обох прогонів, дала медіану 36 і 40
    рядків на сторінку.
    """
    got = select_offers(NIGHT_23_09, _need())

    assert not got.empty, "ринок із двох машин не має бути порожнім"
    assert got.best.offer["id"] == 900001, (
        f"взято {_cores(got.best.offer):.0f}-ядерну за "
        f"${got.best.offer['dph_total']:.3f}/год — ту саму, що коштувала $1.19 "
        f"за 5115 сторінок"
    )
    dear = [c for c in got.candidates if c.offer["id"] == 900002]
    assert not dear, "10-ядерна не має лишатись навіть у кандидатах"


def test_the_bad_box_alone_leaves_the_market_empty() -> None:
    """🔴 І головне: коли доброї машини НЕМАЄ, відповідь — порожньо.

    Не «візьмемо гіршу, бо іншої немає»: чекання безкоштовне, оренда — ні.
    """
    got = select_offers([NIGHT_23_09[1]], _need())
    assert got.empty, (
        "єдина машина на ринку дорожча за роботу вдвічі — її все одно не беремо"
    )

# ── незалежна правильна відповідь ───────────────────────────────────────────
#
# 🔴 Тест без правильної відповіді — не тест. Перевірка «ніщо не домінує над
# обраним» міряє вибір ВІДНОСНО НЬОГО САМОГО. Тому нижче — оракул: чи дає
# машина ціль і скільки коштує ВЕСЬ захід на ній, порахований тут-таки, явно.
# Фізичні сталі імпортуються (заміряні величини, не алгоритм), формула
# написана тут: якщо алгоритм розійдеться з нею, ми це побачимо.
#
#     шарди = min(⌊VRAM карти × 0.9 / ГБ на шард⌋ × карти, ⌊ядра⌋, 32)
#     темп  = min(шарди × A, ядра × A_ядро, карти × A_карта) / (L0(Мпікс) + рядки)
#     придатна ⇔ темп × 0.86 ≥ ціль, $/1000 на цьому темпі ≤ стелі, $/ядро-год ≤ стелі
#     години = 300 с підйому + справ × 27 с + сторінок × 0.138 с + сторінки / темп

ORACLE_COLD_SEC = 300.0
ORACLE_PER_CASE_SEC = 27.0
ORACLE_PER_PAGE_SEC = 0.138

#: Найстаріша архітектура, під яку є колеса torch. Нижче — «CUDA error: no
#: kernel image» і нуль сторінок за оплачений підйом (справжня оренда 19.09).
ORACLE_MIN_COMPUTE_CAP = 700


def _oracle(offer: dict[str, Any], pages: int, cases: int,
            target: float = 5000.0) -> tuple[bool, float]:
    """(чи дає машина ціль, ціна всього заходу на ній, $)."""
    import math

    from gpurunner.core.htr_sizing import (
        CARD_MAX_LINES_PER_HOUR,
        CORE_LINES_PER_HOUR,
        MAX_SHARDS,
        SHARD_LINES_PER_HOUR,
        VRAM_HEADROOM,
        fixed_cost_for,
        gb_per_shard_for,
    )

    cores = _cores(offer)
    dph = float(offer.get("dph_total") or 0)
    cards = int(offer.get("num_gpus") or 1)
    per_card = float(offer.get("gpu_ram") or 0) / 1024.0
    if not cores or not dph or not per_card:
        return False, math.inf
    if int(offer.get("compute_cap") or 0) < ORACLE_MIN_COMPUTE_CAP:
        return False, math.inf
    shards = min(int(per_card * VRAM_HEADROOM // gb_per_shard_for(MPX)) * cards,
                 int(cores), MAX_SHARDS)
    if shards < 1:
        return False, math.inf
    lines_h = min(SHARD_LINES_PER_HOUR * shards, CORE_LINES_PER_HOUR * cores,
                  CARD_MAX_LINES_PER_HOUR * cards)
    rate = lines_h / (fixed_cost_for(MPX) + LINES)
    sure = rate * GUARANTEE_FACTOR
    fits = (sure >= target
            and 1000.0 * dph / sure <= MAX_USD_PER_1000_WITH_TARGET
            and dph / cores <= MAX_USD_PER_CORE_H)
    hours = (pages / rate
             + (ORACLE_COLD_SEC + cases * ORACLE_PER_CASE_SEC
                + pages * ORACLE_PER_PAGE_SEC) / 3600.0)
    return fits, hours * dph


#: Наскільки дорожчим за оптимум може бути вибір алгоритму. Він зважує ще
#: надійність хоста й пам'ять реєстру, тож точного збігу не вимагаємо — але
#: «вдвічі дорожче» (ніч 23.09) має падати.
ORACLE_TOLERANCE = 1.25


@pytest.mark.parametrize(("pages", "cases"), [(5000, 237), (13000, 13), (500, 1)])
def test_the_pick_meets_the_target_and_is_close_to_the_cheapest_such_run(
        market: list[dict[str, Any]], pages: int, cases: int) -> None:
    """🔴🔴 Головний приймач: обрана машина ДАЄ ціль, і серед тих, що дають, —
    майже найдешевша. А коли жодна не дає — вибір порожній, а не поступка."""
    got = select_offers(market, _need(pages=pages, cases=cases))
    verdicts = [(_oracle(o, pages, cases), o) for o in market]
    fitting = sorted(((cost, o) for (ok, cost), o in verdicts if ok),
                     key=lambda pair: pair[0])

    if not fitting:
        assert got.empty, (
            f"оракул не бачить жодної машини, що дає ціль, а алгоритм узяв "
            f"{got.best.offer.get('gpu_name')} ({_cores(got.best.offer):.0f} ядер)")
        return
    assert not got.empty, (
        f"оракул бачить {len(fitting)} придатних (найдешевша — "
        f"{fitting[0][1].get('gpu_name')}), а алгоритм каже «порожньо»: {got.reason}")
    ok, pick_cost = _oracle(got.best.offer, pages, cases)
    assert ok, (
        f"алгоритм узяв машину, що за оракулом НЕ дає цілі: "
        f"{got.best.offer.get('gpu_name')} ({_cores(got.best.offer):.0f} ядер, "
        f"${got.best.offer.get('dph_total'):.3f}/год)")
    best_cost, best_offer = fitting[0]
    assert pick_cost <= best_cost * ORACLE_TOLERANCE, (
        f"{pages} стор. у {cases} справах: алгоритм узяв "
        f"{got.best.offer.get('gpu_name')} за ${pick_cost:.3f}, а найдешевший "
        f"придатний захід — {best_offer.get('gpu_name')} за ${best_cost:.3f}")


def test_every_candidate_meets_the_target(market: list[dict[str, Any]]) -> None:
    """Не лише перший: будь-який кандидат, до якого дійде переоренда, теж дає ціль."""
    got = select_offers(market, _need())
    for c in got.candidates:
        assert c.pph_sure >= 5000.0, c.explain


def test_the_night_of_23_09_oracle_names_the_box_we_should_have_taken() -> None:
    """Оракул на двох машинах тієї ночі — незалежно від алгоритму."""
    good, _ = _oracle(NIGHT_23_09[0], 5115, 237)    # 19.2 ядра
    bad, _ = _oracle(NIGHT_23_09[1], 5115, 237)     # 9.6 ядра
    assert good and not bad
