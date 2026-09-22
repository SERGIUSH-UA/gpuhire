"""Розкладка прогону на залізі — проти виміряних таблиць, а не проти інтуїції.

Золоті числа — заміри хмарних прогонів. Модель свідомо
консервативна (планує по ПІКОВІЙ VRAM на шард), тож нижня межа допуску
широка, а верхня — вузька: завищений прогноз означає взяту не ту машину.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from gpurunner.core import htr_sizing as hs
from gpurunner.core.htr_sizing import plan_sizing, predict_cost, predict_hours

# ---- виміряні машини -------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "cores", "vram", "measured_pph"),
    [
        # карта · ефективні ядра · VRAM ГБ · виміряні стор/год
        ("Tesla V100", 64, 32.0, 2278),   # 2078-2486, 12 шардів
        ("RTX 3090", 26, 24.0, 1564),     # 10 шардів
        ("RTX 4060 Ti", 80, 16.0, 1673),  # 8 шардів
        ("GTX 1650 (дім)", 6, 4.0, 350),  # 3 шарди
    ],
)
def test_conservative_planning_never_exceeds_measured(
    name: str, cores: float, vram: float, measured_pph: float
) -> None:
    """У КОНСЕРВАТИВНОМУ режимі (2.6 ГБ/шард — пік) прогноз не оптимістичніший
    за живий замір.

    2.6 — це реально спостережена щільність на найгустішій книзі
    (31.6 ГБ / 12 шардів V100). Плануючи по ній, ми не можемо пообіцяти
    більше, ніж машина колись видала.
    """
    sizing = plan_sizing(cores=cores, vram_gb=vram, gb_per_shard=2.6)
    assert sizing.usable, name
    assert sizing.pages_per_hour <= measured_pph * 1.15, (
        f"{name}: прогноз {sizing.pages_per_hour:.0f} > заміру {measured_pph}"
    )
    # і не втричі песимістичніший — інакше модель нічого не ранжує
    assert sizing.pages_per_hour >= measured_pph * 0.45, name


def test_default_covers_the_base_consumption_of_a_shard() -> None:
    """🔴 Дефолт мусить покривати БАЗОВЕ споживання шарда, а не «типове».

    Шард тримає 3.2-3.6 ГБ ЗАВЖДИ: це ваги PARSeq + kraken і буфери, вони не
    залежать ні від матеріалу, ні від стелі сегментації. Тому все, що нижче, —
    не «ризикований компроміс», а арифметика, яка не сходиться:

      · 2.0 → Царевка, RTX 3090×2: 1478 записів «out of memory», 740 збоїв із
        1488 сторінок, тобто більше збоїв, ніж зробленого;
      · 2.5 → 8 шардів × 3.3 = 26 ГБ на 24-ГБ карті, 62% сторінок у збоях.

    🔴 Але й вище брати не можна: 4.5 «за площею розвороту» відправило захід на
    RTX 4060 Ti з трьома шардами замість RTX 3090 із сімома — утричі повільніше
    за ті самі гроші, бо скоринг ділить на цей поріг ще на ранжуванні ринку.
    """
    assert 3.2 <= hs.GB_PER_SHARD <= 3.6, "між базовим споживанням і запасом"
    safe = plan_sizing(cores=64, vram_gb=24.0)
    assert safe.shards * hs.GB_PER_SHARD <= 24.0 * hs.VRAM_HEADROOM
    # і при цьому не обіцяє нічого понад стелю шарда
    assert safe.pages_per_hour <= hs.PAGES_PER_HOUR_PER_SHARD * safe.shards


@pytest.mark.parametrize("cores", [26, 32, 64, 128])
def test_more_shards_never_predicts_less_work(cores: int) -> None:
    """🔴 Модель мусить бути монотонною за числом шардів.

    Через `floor` у двох ролях одразу (стеля потоків І оцінка зайнятості)
    вона передбачала, що 10 шардів на 26 ядрах дадуть МЕНШЕ, ніж 8 — і
    планувальник обирав би гіршу розкладку через власну арифметику.
    """
    seen = [plan_sizing(cores=cores, vram_gb=48.0, gb_per_shard=g).pages_per_hour
            for g in (3.0, 2.6, 2.4, 2.0, 1.8)]
    assert seen == sorted(seen), f"{cores} ядер: {seen}"


def test_v100_reference_box() -> None:
    """Еталон із бенчмарку: V100, 64 ядра, 32 ГБ.

    Виміряно було 12 шардів — і саме там стався OOM, тобто 12 це вже за межею.
    На робочих 3.3 виходить 8, і це той бік межі, з якого сторінки не губляться.
    """
    s = plan_sizing(cores=64, vram_gb=32.0)
    assert 8 <= s.shards <= 10
    assert s.limited_by == "vram"
    # Занижений поріг і далі дає більший флот — саме так і губили сторінки.
    assert plan_sizing(cores=64, vram_gb=32.0, gb_per_shard=2.0).shards == 14


def test_shard_count_ordering_matches_benchmark() -> None:
    """Більше VRAM — більше шардів; це той порядок, у якому міряли."""
    v100 = plan_sizing(cores=64, vram_gb=32.0)
    p3090 = plan_sizing(cores=26, vram_gb=24.0)
    ti4060 = plan_sizing(cores=80, vram_gb=16.0)
    assert v100.shards > p3090.shards > ti4060.shards


# ---- інциденти, які модель мусить робити неможливими -----------------------


def test_16gb_box_never_gets_eight_shards() -> None:
    """🔴 Інцидент: 8 шардів на 16 ГБ → CUDA OOM з'їв 46 сторінок мовчки.

    Саме вісім більше не пропонуються за жодного налаштування, а запланована
    VRAM ніколи не перевищує наявну.
    """
    s = plan_sizing(cores=80, vram_gb=16.0)
    assert s.shards < 8
    assert s.shards * hs.GB_PER_SHARD <= 16.0


def test_idle_cores_are_counted_not_hidden() -> None:
    """🔴 Інцидент: 128 ядер, 8 шардів — 9/10 машини не робить нічого.

    Формула це не «лікує», але називає числом, і саме воно потім знижує скор.
    """
    s = plan_sizing(cores=128, vram_gb=23.0)
    assert s.limited_by == "vram"
    assert s.wasted_cores > 40
    # і не обіцяє 128×52 = 6656 стор/год: стеля шарда тримає прогноз чесним
    assert s.pages_per_hour <= hs.PAGES_PER_HOUR_PER_SHARD * s.shards


def test_cores_alone_do_not_buy_throughput() -> None:
    """🔴🔴 Регресія на ДУЕЛЬ 05.09.2026.

    Тут стояв зворотний тест — він вимагав, щоб 128 ядер прогнозувались швидше
    за 16 на тій самій карті. Твердження спростовано контрольованим дослідом:
    два Tesla V100 32 ГБ, та сама справа, ті самі 8 шардів, холодна сегментація
    в обох, різниця тільки в ядрах — 15 проти 92 — дали 921 і 857 стор/год.
    Ушестеро більше ядер = мінус 7% темпу.

    Модель на тих самих числах обіцяла 525 і 2000, тобто різницю в 3.8 раза, і
    ворота ціни через це відкидали дешевший бокс.
    """
    s_few = plan_sizing(cores=16, vram_gb=32.0)
    s_many = plan_sizing(cores=128, vram_gb=32.0)

    assert s_few.shards == s_many.shards, "та сама карта — той самий флот"
    assert s_many.pages_per_hour == s_few.pages_per_hour, (
        "ядра знову купують темп повз число шардів — саме це спростувала дуель")
    # Ядра лишились видимими як ПРОСТІЙ, а не як швидкість.
    assert s_many.wasted_cores > s_few.wasted_cores


def test_the_duel_is_now_predicted_the_right_way_round() -> None:
    """💰 Грошовий приймач: дешевий малоядерний бокс мусить виходити ДЕШЕВШИМ.

    Обидві машини з дуелі — реальні оффери Vast 05.09.2026. Заміряно: 921
    стор/год за $0.219/год ($0.238 за тисячу) проти 857 за $0.324 ($0.378).
    Модель не зобов'язана вгадати темп, але зобов'язана не переплутати, який
    бокс дешевший за тисячу сторінок.
    """
    cheap = plan_sizing(cores=15, vram_gb=32.0)
    dear = plan_sizing(cores=92, vram_gb=32.0)
    pages = 1000

    cheap_k = predict_cost(pages, cheap, 0.219)
    dear_k = predict_cost(pages, dear, 0.324)

    assert cheap_k < dear_k, (
        f"модель знову вважає дорожчий бокс вигіднішим: "
        f"${cheap_k:.3f} проти ${dear_k:.3f}")
    # Заміряне відношення 0.378/0.238 = 1.59; модель має бути в тому ж боці й
    # порядку, але точності від неї тут не вимагаємо — матеріалу вона не знає.
    assert 1.2 <= dear_k / cheap_k <= 2.2


def test_cores_still_limit_the_fleet() -> None:
    """Ядра не зникли з моделі — вони лишились там, де їхній вплив справжній:
    ріжуть ЧИСЛО ШАРДІВ. Інакше 4-ядерний бокс отримав би повний флот."""
    starved = plan_sizing(cores=4, vram_gb=32.0)
    roomy = plan_sizing(cores=64, vram_gb=32.0)
    assert starved.shards < roomy.shards
    assert starved.limited_by == "cpu"
    assert starved.pages_per_hour < roomy.pages_per_hour


def test_cores_beyond_two_per_shard_buy_nothing() -> None:
    """🔴 Сторож проти спростованої стелі — тепер поведінковий, а не текстовий.

    Стеля «стор/год × ядра» поверталась би природно, бо виглядає як
    обережність; дуель 05.09.2026 показала, що вона бреше (15 і 92 ядра на тій
    самій карті — 921 і 857 стор/год, а модель обіцяла 525 і 2000, тобто ворота
    ціни брали дорожчий бокс). Спростована саме ЗРОСТАЮЧА стеля: ядра понад
    дві на шард темпу не купують.

    ⚠ Текстовий сторож («cores» немає у вихідному коді формули) стояв тут до
    21.09.2026 і став непридатним: тоді в модель увійшла СПАДНА стеля за
    ядрами — вона ловить протилежний випадок, коли шардів підняли більше, ніж
    є кому годувати. Тому стережемо саме поведінку, а не текст: текст
    забороняв би й те, що виміряно.
    """
    src = (Path(hs.__file__)).read_text(encoding="utf-8")
    formula = src[src.index("budget = min("):]
    formula = formula[:formula.index("return Sizing(")]
    assert "lines" in formula, "з формули темпу зник матеріал:\n" + formula
    assert not hasattr(hs, "PAGES_PER_HOUR_PER_CORE"), (
        "стала повернулась — разом із нею повернеться й подвійний рахунок")

    # ядра понад дві на шард не купують нічого: 8 шардів за VRAM в обох
    rich = plan_sizing(cores=128, vram_gb=32.0)
    enough = plan_sizing(cores=16, vram_gb=32.0)
    assert rich.shards == enough.shards == 8
    assert rich.pages_per_hour == enough.pages_per_hour, (
        "ядра знову купують темп: 128 ядер обіцяють більше за 16 на тому "
        "самому флоті")
    assert not rich.cpu_capped and not enough.cpu_capped

    # 🔴 Головне: стеля не сміє ПЕРЕВЕРНУТИ вирок дуелі. Обидва її бокси мали
    # ≥1.875 ядра на шард, тобто стояли біля самого зламу: на 15 ядрах стеля
    # забирає 6%, на 92 — нічого. Спростована стеля давала там різницю в 3.8
    # раза (525 проти 2000) і саме через це ворота ціни брали дорожчий бокс.
    cheap = plan_sizing(cores=15.0, vram_gb=32.0, max_shards=8)
    dear = plan_sizing(cores=92.0, vram_gb=32.0, max_shards=8)
    assert cheap.pages_per_hour >= 0.9 * dear.pages_per_hour, (
        f"дешевий бокс знову занижений: {cheap.pages_per_hour:.0f} проти "
        f"{dear.pages_per_hour:.0f} — саме так він і провалював ворота ціни")
    assert not dear.cpu_capped, "92 ядра на 8 шардах: годувати флот є чим"


def test_shard_ceiling_is_computed_per_card_not_from_the_sum() -> None:
    """🔴🔴 Пам'ять карт НЕ спільна, тож ділити треба покартково.

    `int(сума × 0.9 / gb)` округлює ОДИН раз і тому щедріший за чесний
    покартковий підрахунок: дві карти по 8 ГБ дають int(16×0.9/2.5) = 5
    шардів, тоді як покартково це int(8×0.9/2.5) = 2 на карту, тобто 4.
    А шарди розкладаються круговою чергою `cuda:(k % N)`, отже на кожну карту
    сідає ceil(шардів / карт) — і саме ця величина мусить влазити.

    Ціна 2026-08-12: бокс віддав дві GTX 1080 по 8 ГБ, стеля порахувалась із
    «16 ГБ», на cuda:0 сіли ТРИ шарди (7.5 ГБ на карті 7.92) — і
    «Tried to allocate 1.51 GiB»: 152 збої на 48 готових сторінок.
    """
    import math

    from gpurunner.core.htr_sizing import VRAM_HEADROOM

    per_card, n_gpus, gb = 8.0, 2, 2.5
    naive = plan_sizing(cores=16, vram_gb=per_card * n_gpus, gb_per_shard=gb).shards
    assert math.ceil(naive / n_gpus) * gb > per_card * VRAM_HEADROOM, (
        "саме так виглядала вада: три шарди по 2.5 на карті 8.0"
    )

    fit_cap = int(per_card * VRAM_HEADROOM // gb) * n_gpus
    fixed = plan_sizing(cores=16, vram_gb=per_card * n_gpus, gb_per_shard=gb,
                        max_shards=fit_cap).shards
    assert math.ceil(fixed / n_gpus) * gb <= per_card * VRAM_HEADROOM


def test_single_card_box_is_unaffected_by_the_per_card_rule() -> None:
    """На одній карті сума і є карта — правило нічого не міняє."""
    from gpurunner.core.htr_sizing import VRAM_HEADROOM

    gb = 2.5
    plain = plan_sizing(cores=32, vram_gb=24.0, gb_per_shard=gb).shards
    capped = plan_sizing(cores=32, vram_gb=24.0, gb_per_shard=gb,
                         max_shards=int(24.0 * VRAM_HEADROOM // gb)).shards
    assert plain == capped


def test_four_core_box_is_all_but_useless() -> None:
    """Найдешевший оффер із потрібною картою регулярно має 4 ядра.

    ⚠ Переписано 06.09.2026: шард бере ~1 ядро (std160 — 12 шардів на квоті
    11.5, loadavg 10.2), тож 4 ядра = 4 шарди, а не 2. Бокс усе одно слабкий:
    обмежувач — ядра, і він повільніший за 12-ядерний удвічі-втричі.
    """
    s = plan_sizing(cores=4, vram_gb=32.0)
    assert s.shards == 4
    assert s.limited_by == "cpu"
    assert s.pages_per_hour <= plan_sizing(cores=12, vram_gb=32.0).pages_per_hour / 2


# ---- непридатне залізо -----------------------------------------------------


@pytest.mark.parametrize(
    ("cores", "vram", "limited"),
    [(64, 2.0, "vram"), (0.5, 32.0, "cpu"), (0, 0, "vram")],
)
def test_unusable_box_returns_zero_shards(cores: float, vram: float, limited: str) -> None:
    """Непридатне повертається нулем шардів, а не винятком — щоб ранжування
    просто його відкинуло, не обростаючи try/except."""
    s = plan_sizing(cores=cores, vram_gb=vram)
    assert not s.usable
    assert s.shards == 0
    assert s.limited_by == limited
    assert math.isinf(predict_hours(1000, s))
    assert math.isinf(predict_cost(1000, s, 0.2))


def test_max_shards_cap_is_reported() -> None:
    s = plan_sizing(cores=256, vram_gb=180.0, max_shards=12)
    assert s.shards == 12
    assert s.limited_by == "cap"


# ---- час і гроші -----------------------------------------------------------


def test_cold_start_overhead_is_charged() -> None:
    """Холодний бокс коштує ~8 хв ще до першої сторінки — на дрібній справі це
    більше за саму роботу, і саме тому потрібна тепла черга."""
    s = plan_sizing(cores=64, vram_gb=32.0)
    cold = predict_hours(100, s, warm=False)
    warm = predict_hours(100, s, warm=True)
    assert cold - warm == pytest.approx(
        (hs.OVERHEAD_SEC_COLD - hs.OVERHEAD_SEC_WARM) / 3600.0, rel=1e-6
    )
    assert cold > 2 * warm


def test_big_case_on_reference_box_matches_measured_wall_clock() -> None:
    """1665 сторінок на V100 — ~45-55 хв разом зі стартом (замір: 41 хв на 1700
    без накладних)."""
    s = plan_sizing(cores=64, vram_gb=32.0)
    hours = predict_hours(1665, s)
    assert 0.7 <= hours <= 1.1
    assert 0.15 <= predict_cost(1665, s, 0.222) <= 0.30


def test_vram_per_shard_knob_changes_shards() -> None:
    """Ручка існує, щоб ризикувати свідомо, а не правити константу в коді."""
    cautious = plan_sizing(cores=64, vram_gb=32.0, gb_per_shard=2.6)
    greedy = plan_sizing(cores=64, vram_gb=32.0, gb_per_shard=1.8)
    assert greedy.shards > cautious.shards


# ---- домовленість між планувальником і раннером -----------------------------


def test_multi_gpu_vram_is_summed_only_when_shards_are_distributed() -> None:
    """🔴🔴 Два виміри одного дня, і обидва треба тримати в голові.

    БЕЗ розкладки (стан коду до 2026-08-11): Царевка на RTX 3090×2 дала
    «47.2 ГБ вільно» → 21 шард → усі на `cuda:0` з 24 ГБ → 1478 записів
    «out of memory», 740 сторінок із 1488 у збоях.

    З розкладкою (`_start_shard` → `cuda:(k % N)`): ділити суму означало б
    утричі занизити 8-карткову машину без причини.

    Тому прапорець явний. Цей тест — місце, де домовленість зафіксована: якщо
    хтось змінить раннер, тест має впасти разом із ним.
    """
    kw = {"cores": 128, "vram_gb": 48.0, "gb_per_shard": 2.0}
    spread = plan_sizing(**kw, num_gpus=2, shards_distributed=True)
    pinned = plan_sizing(**kw, num_gpus=2, shards_distributed=False)
    # ±1 — цілочисельне округлення на межі карти (21.6 → 21 проти 10.8 → 10)
    assert abs(spread.shards - 2 * pinned.shards) <= 1
    assert spread.shards > pinned.shards
    # одна карта — прапорець нічого не міняє
    single = plan_sizing(cores=128, vram_gb=24.0, gb_per_shard=2.0)
    assert single.shards == plan_sizing(
        cores=128, vram_gb=24.0, gb_per_shard=2.0, num_gpus=1,
        shards_distributed=False,
    ).shards


def test_runner_actually_distributes_shards_across_cards() -> None:
    """Друга половина тієї ж домовленості — з боку раннера.

    Якщо цей тест упаде, `shards_distributed=True` у планувальнику стане
    брехнею, і повернеться рівно той OOM, що з'їв 740 сторінок.
    """
    import types

    from gpurunner._embedded import htr_case_runner as runner

    launched: list[list[str]] = []

    class _Proc:
        pid = 1
        stdout = None

    original_popen, original_thread = runner.subprocess.Popen, runner.threading.Thread
    runner.subprocess.Popen = lambda cmd, **kw: (launched.append(list(cmd)), _Proc())[1]
    runner.threading.Thread = lambda **kw: types.SimpleNamespace(start=lambda: None)
    try:
        fleet = {"shards": {}, "n_gpus": 2}
        base = ["py", "htr_case_run.py", "--device", "cuda:0", "--gpu-lock", "/tmp/gpu.lock"]
        for k in range(4):
            runner._start_shard(k, 4, base, {}, Path("."), fleet)
    finally:
        runner.subprocess.Popen, runner.threading.Thread = original_popen, original_thread

    devices = [c[c.index("--device") + 1] for c in launched]
    locks = [c[c.index("--gpu-lock") + 1] for c in launched]
    assert devices == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    # лок теж пер-картковий — спільний звів би дві карти назад до однієї
    assert len(set(locks)) == 2


# ---- калібровка проти РЕАЛЬНИХ заходів -------------------------------------


def _registry_rows() -> list[dict]:
    """Завершені заходи із заміряним темпом І рядками на сторінку.

    Джерело — стан сесій наглядача, зшитий із метою (`htr.calibrate.stitch_runs`):
    у `data/boxes.jsonl` темп лежить без матеріалу, а модель тепер від нього
    залежить. Це єдине чесне джерело калібровки: власні досліди систематично
    занижені (бойовий прогін тієї самої справи дав 15.9 с/стор на шард,
    дослідний збирач — 31.3).

    🔴 Мікрозаходи відсіяні (`MIN_PAGES_FOR_RATE`): вони міряють не темп, а
    фіксовану ціну справи, яку модель рахує окремо (`OVERHEAD_SEC_*`), і в
    калібровці вона враховувалась би вдруге — 59 таких заходів давали медіану
    4.29 проти 1.05 на решті.
    """
    from gpurunner.htr.calibrate import rated, stitch_runs

    out = []
    for r in rated(stitch_runs()):
        if r.vram <= 0 or r.cores <= 0:
            continue
        out.append({"cores": r.cores, "vram": r.vram, "ngpu": r.n_gpus,
                    "lines": r.lines, "shards": r.shards, "actual": r.actual})
    return out


def _on_measured_fleet(r: dict) -> float:
    """Прогноз темпу на ЗАПИСАНОМУ флоті, а не на тому, який обрали б ми.

    🔴 Інакше одне число перевіряє дві різні речі — правило VRAM і правило
    темпу, — і вони гасять похибку одне одного: модель планує флот на чверть
    менший за той, що реально працював (дефолтні 3.3 ГБ на шард проти
    заміряних ~1.95 у наглядача), тож завищений темп виглядає нормальним.
    Саме так зміщення 1.13 прожило два тижні непоміченим.
    """
    return plan_sizing(
        cores=r["cores"],
        vram_gb=max(r["vram"] * r["ngpu"],
                    r["shards"] * hs.GB_PER_SHARD / hs.VRAM_HEADROOM),
        num_gpus=r["ngpu"], lines_per_page=r["lines"],
        max_shards=max(1, r["shards"]),
    ).pages_per_hour


def test_model_is_slightly_conservative_against_recorded_runs() -> None:
    """🔴 Приймач самої калібровки: на записаних заходах модель мусить трохи
    ЗАНИЖУВАТИ, а не завищувати.

    Правило проєкту: завищений прогноз гірший за занижений, бо він проходить
    ворота, а потім захід не встигає ні в бюджет, ні в строк. Крива
    `36 000 / (40 + рядки)` зі стелею за ядрами обрана саме за цим критерієм
    (медіана 0.92 на 457 чистих заходах); попередня трійка без стелі давала
    1.06, і то було СЕРЕДНЄ по двох режимах — див. сторожа нижче.

    ⚠ Точності тут і далі не вимагаємо: член за матеріалом (06.09.2026) знімає
    розкид за рядками, але не за жанром, станом кадрів і стелею сегментації.
    Стережеться тільки ЗМІЩЕННЯ.
    """
    rows = _registry_rows()
    if len(rows) < 20:
        pytest.skip(f"замало записаних заходів: {len(rows)}")

    ratios = sorted(_on_measured_fleet(r) / r["actual"] for r in rows)
    median = ratios[len(ratios) // 2]
    assert 0.85 <= median <= 1.05, (
        f"модель зміщена: медіана прогноз/факт {median:.2f} на {len(ratios)} заходах")


def test_the_two_runners_are_predicted_the_same_way() -> None:
    """🔴🔴 Сторож, якого бракувало: спільна медіана ховає ДВА режими.

    07.09–21.09.2026 модель мала медіану 1.06 — майже в межах, — а за нею
    стояв розрив: старий раннер 0.96, новий 1.27. Тобто вона завищувала рівно
    на тих флотах, які ми купуємо тепер (14–20 шардів на 19–23 ядрах), і
    кошторис був занижений там, де це коштує грошей. Одне число цього не
    показувало, бо зміщення в різні боки гасились.

    Колонки, що розходяться, означають, що моделі бракує ЧЛЕНА, а не підбору
    сталої. Межа 0.20 — з запасом: зараз розрив 0.04.
    """
    from gpurunner.htr.calibrate import rated, stitch_runs

    rows = [r for r in rated(stitch_runs()) if r.vram > 0 and r.cores > 0]
    old = [r for r in rows if not r.runner_new]
    new = [r for r in rows if r.runner_new]
    if len(old) < 20 or len(new) < 20:
        pytest.skip(f"замало заходів для порівняння: {len(old)} / {len(new)}")

    def median_of(rs: list) -> float:
        v = sorted(_on_measured_fleet(
            {"cores": r.cores, "vram": r.vram, "ngpu": r.n_gpus,
             "lines": r.lines, "shards": r.shards, "actual": r.actual}) / r.actual
            for r in rs)
        return v[len(v) // 2]

    gap = abs(median_of(old) - median_of(new))
    assert gap <= 0.20, (
        f"раннери прогнозуються по-різному: {median_of(old):.2f} проти "
        f"{median_of(new):.2f} — моделі бракує члена, а не калібровки")


def test_the_recent_runs_have_not_drifted() -> None:
    """⏱ Ковзний сторож: останні 200 чистих заходів.

    Ловить дрейф раніше, ніж це зробить людина, і раніше, ніж він розчиниться
    в повному наборі. Межі ширші за головний приймач навмисно — коротке вікно
    шумне (на 100 заходах 0.89, на 250 — 0.91).
    """
    rows = _registry_rows()[-200:]
    if len(rows) < 50:
        pytest.skip(f"замало свіжих заходів: {len(rows)}")

    ratios = sorted(_on_measured_fleet(r) / r["actual"] for r in rows)
    median = ratios[len(ratios) // 2]
    assert 0.75 <= median <= 1.15, (
        f"свіжі заходи поїхали: медіана {median:.2f} на {len(ratios)} останніх")


def test_low_core_boxes_are_no_longer_written_off() -> None:
    """💰 Саме через це відкидались дешеві машини: стеля «35 × ядра» різала
    прогноз малоядерних утричі, і ворота ціни бачили їх дорожчими за тисячу
    сторінок, ніж вони є.

    Замір, що це спростував: бокс на 15 ядер дав $0.238 за тисячу проти $0.378
    у 92-ядерного.
    """
    rows = [r for r in _registry_rows() if r["cores"] <= 24]
    if len(rows) < 5:
        pytest.skip(f"замало малоядерних заходів: {len(rows)}")

    ratios = sorted(plan_sizing(cores=r["cores"], vram_gb=r["vram"] * r["ngpu"],
                                num_gpus=r["ngpu"], lines_per_page=r["lines"]).pages_per_hour
                    / r["actual"] for r in rows)
    median = ratios[len(ratios) // 2]
    assert median >= 0.85, (
        f"малоядерні знову занижені: медіана {median:.2f} — саме так вони й "
        f"провалювали ворота ціни")
