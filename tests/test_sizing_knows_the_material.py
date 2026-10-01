"""Модель добору знає матеріал і карту, а не лише число шардів (06.09.2026).

Замір-причина — std160 на RTX 3090 (24 ГБ, квота 11.52 ядра, кадри 7.2 Мпікс,
70 рядків/стор, $0.15/год):

    5 шардів (те, що дала модель)  3600 стор/год   карта 64%
    8 шардів                       4600            86%
    12 шардів                      5040            94%
    16 шардів                      4997            96%   ← стеля карти

Модель на цьому боксі бачила 5 шардів і 1100 стор/год — тобто ціну тисячі
$0.136 там, де факт $0.03. Три причини, кожна тут закрита окремим тестом:
`MIN_CORES_PER_SHARD = 2` (шард бере ~1 ядро), `GB_PER_SHARD = 3.3` на кадрах,
де шард тримає 1.2 ГБ, і темп «220 на шард» незалежно від щільності рядків.
"""
from __future__ import annotations

import pytest

from gpurunner.core import htr_sizing as hs
from gpurunner.core.htr_sizing import gb_per_shard_for, plan_sizing


def test_std160_box_gets_the_fleet_that_was_measured() -> None:
    """RTX 3090 / 11.52 ядра / 7.2 Мпікс / 70 рядків → 10–12 шардів.

    🪤 ЦІНА РІШЕННЯ 21.09.2026, записана тут навмисно. Сам замір std160 —
    5040 стор/год, тобто 48 000 рядко-годин на ЯДРО. Медіана тієї самої смуги
    (менше двох ядер на шард, 113 заходів) — 20 883, і std160 лежить між її p75
    і p90. Кадри там легкі (7.2 Мпікс), а CPU-геометрія kraken дешевшає саме з
    площею кадру, тож це верхній хвіст, а не спростування стелі за ядрами.

    Ми плануємо по МЕДІАНІ 113 заходів, а не по найкращому боксу, і тому
    недооцінюємо такий бокс приблизно втричі (прогноз ~1885 проти 5040). Ціна
    цієї консервативності — впущена дешева швидка машина; ціна протилежного
    вибору — захід, що не вклався в бюджет, бо модель завищила темп усім
    флотам із малим числом ядер (саме це й тривало з 06.09 по 21.09).

    Розсудити це може лише замір: повторити std160 із метою прогону (щоб у
    реєстрі з'явились `lines`) і подивитись, куди він ляже наступною
    калібровкою.
    """
    s = plan_sizing(cores=11.52, vram_gb=24.0, gb_per_shard=gb_per_shard_for(7.2),
                    lines_per_page=70)
    assert 10 <= s.shards <= 12, s
    # 11.5 ядер тримають 11 шардів; пам'яті (24×0.9/1.18) вистачає на 18 —
    # обмежувач чесно CPU, головне, що жоден не дав п'ять
    assert s.limited_by in ("cpu", "vram")
    assert s.cpu_capped, "на 11.5 ядрах і 11 шардах мусить тиснути CPU"
    assert 1800 <= s.pages_per_hour <= 5040, (
        f"прогноз {s.pages_per_hour:.0f}: ворота ціни бачили б не ту тисячу")


def test_light_frames_get_a_lower_vram_threshold() -> None:
    """7-Мпікс кадри std160 тримали 1.1–1.25 ГБ на шард; 3.3 на них — 6 шардів
    замість 12. Прогноз лежить НАД заміряним зайняттям карти (23.09.2026:
    0.4 + 0.08 × Мпікс як верхня межа), а невідомий матеріал і далі важкий."""
    assert 1.1 <= gb_per_shard_for(7.2) <= 1.4
    assert 0.4 + 0.08 * 16.0 < gb_per_shard_for(16.0) < hs.GB_PER_SHARD
    assert gb_per_shard_for() == hs.GB_PER_SHARD
    assert gb_per_shard_for(0.0) == hs.GB_PER_SHARD
    # aspect не діє — рядки з форми кадру не виводяться (урок 19.08.2026)
    assert gb_per_shard_for(7.1, 1.26) == gb_per_shard_for(7.1, 0.75)
    # монотонність і стеля
    assert gb_per_shard_for(12.7) < gb_per_shard_for(16.0) <= hs.GB_PER_SHARD_CEILING
    assert gb_per_shard_for(100.0) == hs.GB_PER_SHARD_CEILING


def test_two_cards_double_the_pipeline_ceiling() -> None:
    """Стеля 4500 стояла на БОКС: 2×3090 прогнозувались як одна карта.

    ⚠ Ядра ростуть разом із картами (21.09.2026). Доти обидва бокси мали по 32
    ядра, і дослід міняв ДВІ величини одразу: друга карта подвоювала флот, але
    ядер на шард ставало вдвічі менше — і стеля за ядрами чесно різала темп,
    через що тест «про карту» падав через CPU. Щоб питання лишилось про карту,
    друга карта приходить зі своїми ядрами.
    """
    one = plan_sizing(cores=32, vram_gb=24.0, gb_per_shard=1.9, lines_per_page=70)
    two = plan_sizing(cores=64, vram_gb=48.0, gb_per_shard=1.9, num_gpus=2,
                      lines_per_page=70)
    assert two.shards == 2 * one.shards or two.limited_by == "cpu"
    assert two.pages_per_hour == pytest.approx(2 * one.pages_per_hour, rel=0.10)


def test_the_card_ceiling_is_reported_not_hidden() -> None:
    """16 шардів на 3090 не дали більше за 12 — і модель мусить це сказати
    (`card_capped`), а не малювати лінійне зростання."""
    s = plan_sizing(cores=64, vram_gb=48.0, gb_per_shard=1.5, lines_per_page=70)
    assert s.shards >= 20
    assert s.card_capped
    assert s.pages_per_hour == pytest.approx(hs.card_cap_pages_per_hour(70), rel=1e-6)
    small = plan_sizing(cores=4, vram_gb=24.0, gb_per_shard=1.9, lines_per_page=70)
    assert not small.card_capped


def test_dense_pages_are_predicted_slower_than_sparse_ones() -> None:
    """230-1-24 (35 рядків) — 4594 стор/год проти spr-7049 (114 рядків) — 3195
    на тому самому класі боксу. Без члена за матеріалом обидва прогнозувались
    одним числом."""
    sparse = plan_sizing(cores=64, vram_gb=24.0, lines_per_page=35)
    dense = plan_sizing(cores=64, vram_gb=24.0, lines_per_page=114)
    assert sparse.shards == dense.shards
    assert sparse.pages_per_hour > 1.5 * dense.pages_per_hour


def test_unknown_material_is_predicted_exactly_as_before() -> None:
    """Перекалібровка 05.09.2026 дала 220 на шард; невідомий матеріал не сміє
    зрушити жоден старий приймач."""
    s = plan_sizing(cores=64, vram_gb=32.0)
    assert s.pages_per_hour == pytest.approx(s.shards * hs.PAGES_PER_HOUR_PER_SHARD, rel=1e-6)
    assert hs.per_shard_pages_per_hour(0) == pytest.approx(hs.PAGES_PER_HOUR_PER_SHARD)
    assert hs.per_shard_pages_per_hour(hs.LINES_PER_PAGE_DEFAULT) == pytest.approx(
        hs.PAGES_PER_HOUR_PER_SHARD)


def test_material_curve_matches_the_registry_bins() -> None:
    """Крива `A/(L0+рядки)` проти медіан зшивки (стор/год на шард).

    Освіжено 21.09.2026: точки взято з бінів, де ЯДЕР ВИСТАЧАЄ (від двох на
    шард) — саме там крива й міряється, бо в решті темп ріже стеля за ядрами, а
    не матеріал. Допуск ±15%: це підбір, не апроксимація кожного біну.

    ⚠ Найрідший бін (<40 рядків) навмисно НЕ приймач: крива дає там 486 проти
    заміряних 583, тобто занижує на 17%. Це консервативний бік, і рідкі
    сторінки — найменша частина матеріалу; підганяти під них означало б
    завищити прогноз на щільних, де й крутиться робота.
    """
    for lines, measured in ((50, 395), (114, 248), (149, 191)):
        assert hs.per_shard_pages_per_hour(lines) == pytest.approx(measured, rel=0.15), lines
    assert hs.per_shard_pages_per_hour(30) < 583, "рідкий бін мусить лишатись заниженим"


def test_one_core_per_shard_is_the_new_floor() -> None:
    """4 ядра = 4 шарди (а не 2), і обмежувач — ядра."""
    s = plan_sizing(cores=4, vram_gb=32.0)
    assert s.shards == 4 and s.limited_by == "cpu"
    assert s.pages_per_hour < plan_sizing(cores=12, vram_gb=32.0).pages_per_hour
