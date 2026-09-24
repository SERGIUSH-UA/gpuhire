"""Ворота заліза: що саме заводить бокс у чорний список, а що пропускає.

Кожен провал тут коштував грошей у сесії 2026-08-11.
"""

from __future__ import annotations

import pytest

from gpurunner.core.htr_sizing import gb_per_shard_for
from gpurunner.core.offer_score import Need
from gpurunner.supervise.gate import evaluate

NEED = Need(pages=1665, max_hours=8.0, budget_usd=3.00, disk_gb=120, min_net_mbps=20.0)

V100 = {
    "id": 1, "machine_id": 38902, "gpu_name": "Tesla V100", "num_gpus": 1,
    "cpu_cores_effective": 64.0, "gpu_ram": 32 * 1024, "cpu_ram": 128 * 1024,
    "disk_space": 200.0, "dph_total": 0.222, "reliability2": 0.99, "inet_down": 755.0,
}


def probe(**kw) -> dict:
    """Здорова проба, у яку тест підмінює одне поле."""
    base = {
        "cores": 64.0, "cores_all": 64.0, "ram_gb": 128.0,
        "gpu": "Tesla V100-SXM2-32GB",
        "vram_total_mb": 32510.0, "vram_free_mb": 32200.0,
        "disk_free_gb": 190.0, "net_bps": 50_000_000.0, "py": "3.10",
    }
    base.update(kw)
    return base


# ---- пропуск ---------------------------------------------------------------


def test_healthy_box_passes_and_plans_on_measured_hardware() -> None:
    res = evaluate(probe(), V100, NEED)
    assert res.ok and res.outcome == "ok"
    assert res.sizing is not None and res.sizing.shards >= 8
    assert res.hours is not None and res.hours < 1.5
    assert "стор/год" in res.detail


def test_missing_net_probe_is_not_a_dead_channel() -> None:
    """🔴 «Не міряли» ≠ «нуль». Інакше ворота банять здорову машину лише за те,
    що їм не дали URL для заміру."""
    res = evaluate(probe(net_bps=""), V100, NEED)
    assert res.ok


# ---- інциденти -------------------------------------------------------------


def test_slow_channel_is_caught_before_pip() -> None:
    """🔴 Оффер обіцяв 755 Мбіт/с, віддавав 0.5 — і це коштувало години.

    Тепер вирок виноситься до заливки даних, тобто за секунди.
    """
    res = evaluate(probe(net_bps=62_500.0), V100, NEED)  # 0.5 Мбіт/с
    assert not res.ok and res.outcome == "slow_net"
    assert "0.5 Мбіт/с" in res.detail
    assert "755" in res.detail  # і обіцянка теж у вироку


def test_container_without_a_gpu_is_never_booted() -> None:
    """Симптом «Template not found» / CDI: інстанс живий, карти в ньому немає."""
    res = evaluate(probe(gpu="", vram_total_mb=0, vram_free_mb=0), V100, NEED)
    assert not res.ok and res.outcome == "never_booted"


def test_core_count_lie_is_caught() -> None:
    """🔴 Бокси, що рапортували 96 «ядер», видавали менше за 64-ядерний V100."""
    res = evaluate(probe(cores=8.0), V100, NEED)
    assert not res.ok and res.outcome == "cpu_lie"
    assert "8 проти обіцяних 64" in res.detail


def test_vram_lie_is_caught() -> None:
    res = evaluate(probe(vram_total_mb=16 * 1024, vram_free_mb=16 * 1024), V100, NEED)
    assert not res.ok and res.outcome == "vram_lie"


def test_card_busy_with_a_neighbour_is_rejected() -> None:
    """Карта є, вона навіть заявленого розміру — але вільного місця немає."""
    res = evaluate(probe(vram_free_mb=900.0), V100, NEED)
    assert not res.ok and res.outcome == "vram_lie"
    assert "сусід" in res.detail


def test_short_disk_is_rejected() -> None:
    res = evaluate(probe(disk_free_gb=40.0), V100, NEED)
    assert not res.ok and res.outcome == "disk_short"


def test_honest_but_too_slow_box_is_rejected_by_the_deadline() -> None:
    """Машина не бреше — просто не встигає. Це теж підстава не орендувати."""
    small = {**V100, "cpu_cores_effective": 8.0, "gpu_ram": 8 * 1024, "dph_total": 0.05}
    res = evaluate(
        probe(cores=8.0, vram_total_mb=8 * 1024, vram_free_mb=8 * 1024), small,
        Need(pages=30_000, max_hours=4.0, budget_usd=10.0),
    )
    assert not res.ok
    assert "год при стелі" in res.detail


def test_over_budget_on_measured_hardware_is_rejected() -> None:
    res = evaluate(probe(), V100, Need(pages=1665, max_hours=8.0, budget_usd=0.05))
    assert not res.ok
    assert "при бюджеті" in res.detail


# ---- порядок перевірок -----------------------------------------------------


def test_gpu_check_comes_first() -> None:
    """Найдешевша перевірка й найважливіший вирок — перед усіма іншими."""
    res = evaluate(
        probe(gpu="", vram_total_mb=0, cores=2.0, disk_free_gb=1.0, net_bps=1.0), V100, NEED
    )
    assert res.outcome == "never_booted"


def test_16gb_box_never_gets_the_eight_shards_that_ate_46_pages() -> None:
    """🔴 Той самий бокс, що з'їв 46 сторінок, тепер отримує стільки шардів,
    скільки справді влазить: 7 на дефолтних 2.0 ГБ, 5 на консервативних 2.6 —
    але ніколи вісім."""
    ti = {**V100, "gpu_name": "RTX 4060 Ti", "gpu_ram": 16 * 1024,
          "cpu_cores_effective": 32.0, "dph_total": 0.098}
    res = evaluate(
        probe(cores=32.0, vram_total_mb=16 * 1024, vram_free_mb=15.9 * 1024), ti, NEED
    )
    assert res.ok
    assert res.sizing is not None and res.sizing.shards < 8


def test_idle_cores_are_named_in_the_verdict() -> None:
    many = {**V100, "gpu_name": "RTX 4090", "gpu_ram": 23 * 1024, "cpu_cores_effective": 128.0}
    res = evaluate(
        probe(cores=128.0, vram_total_mb=23 * 1024, vram_free_mb=22.8 * 1024), many, NEED
    )
    assert res.ok
    assert "простоює" in res.detail


def test_probe_without_claims_still_works() -> None:
    """Оффер без `cpu_cores_effective` (буває) не має валити ворота."""
    bare = {"id": 5, "machine_id": 7, "gpu_name": "?", "dph_total": 0.2}
    res = evaluate(probe(), bare, NEED)
    assert res.ok


def test_garbage_probe_values_do_not_raise() -> None:
    res = evaluate({"cores": "нема", "gpu": "", "vram_total_mb": "?"}, V100, NEED)
    assert not res.ok and res.outcome == "never_booted"


@pytest.mark.parametrize("field", ["cores", "vram_free_mb", "disk_free_gb"])
def test_zero_everything_is_rejected(field: str) -> None:
    res = evaluate(probe(**{field: 0.0}), V100, NEED)
    assert not res.ok


# ---- частка машини, яка справді наша (розбір 2026-08-19) -------------------

#: Оффер машини 139040: RTX 3090, 40 проданих ядер, реєстр знав 2592 стор/год.
RTX3090 = {
    "id": 2, "machine_id": 139040, "gpu_name": "RTX 3090", "num_gpus": 1,
    "cpu_cores_effective": 40.0, "gpu_ram": 24 * 1024, "cpu_ram": 64 * 1024,
    "disk_space": 200.0, "dph_total": 0.161, "reliability2": 0.99, "inet_down": 500.0,
}

#: Оффер машини 39565: дві RTX 3090, продано 48 ядер, `nproc` показував 192.
RTX3090X2 = {
    "id": 3, "machine_id": 39565, "gpu_name": "RTX 3090", "num_gpus": 2,
    "cpu_cores_effective": 48.0, "gpu_ram": 24 * 1024, "cpu_ram": 64 * 1024,
    "disk_space": 780.0, "dph_total": 0.270, "reliability2": 0.98, "inet_down": 448.0,
}


def test_busy_card_is_rejected_before_the_first_page() -> None:
    """🔴 Машина 139040, 2026-08-19: чужий процес тримав 19 з 24 ГБ.

    Ворота пропускали її, бо всі їхні пороги абсолютні: 3 год < стелі 8 і
    $0.48 < бюджету $2. А тисяча сторінок коштувала $0.976 при стелі $0.20 —
    удесятеро дорожче за те, по що ми йшли.
    """
    res = evaluate(
        probe(cores=80.0, cores_all=80.0, cores_quota=40.0,
              gpu="NVIDIA GeForce RTX 3090",
              vram_total_mb=24576.0, vram_free_mb=5248.0, vram_free_min_mb=5248.0),
        RTX3090, Need(pages=709, max_hours=8.0, budget_usd=2.0, min_net_mbps=20.0),
    )
    assert not res.ok
    # Не `vram_lie`: карта не мала, вона зараз не наша — і хост здоровий.
    assert res.outcome == "card_busy"
    assert "чужий орендар" in res.detail


def test_slow_for_its_money_is_not_blamed_on_a_neighbour() -> None:
    """Карта вільна, а машина все одно не окупається — це `overpriced`."""
    res = evaluate(
        probe(cores=6.0, cores_all=6.0, gpu="NVIDIA GeForce RTX 3090",
              vram_total_mb=24576.0, vram_free_mb=24000.0, vram_free_min_mb=24000.0),
        {**RTX3090, "cpu_cores_effective": 6.0, "dph_total": 0.35},
        Need(pages=709, max_hours=8.0, budget_usd=5.0, min_net_mbps=20.0),
    )
    assert not res.ok and res.outcome == "overpriced"
    assert "чужий орендар" not in res.detail


def test_ceiling_follows_the_one_used_when_picking_the_offer() -> None:
    """Стеля береться з `need`, інакше ворота ріжуть те, що добір дозволив."""
    busy = probe(cores=80.0, cores_all=80.0, gpu="NVIDIA GeForce RTX 3090",
                 vram_total_mb=24576.0, vram_free_mb=5248.0, vram_free_min_mb=5248.0)
    relaxed = Need(pages=709, max_hours=8.0, budget_usd=2.0, min_net_mbps=20.0,
                   max_usd_per_1000_pages=1.50)
    assert evaluate(busy, RTX3090, relaxed).ok


def test_two_card_box_still_gets_both_cards() -> None:
    """🔴 РЕГРЕСІЯ. Дво-картковий бокс НЕ винен у падінні 2026-08-19: шарди
    розкладаються `cuda:(k % N)`, і машина 51342 з тим самим профілем
    («47.1 з 48 ГБ») того ж тижня прогнала чергу повністю.

    Тобто лагодження не сміє забрати половину темпу: 8 шардів на карту × 2.
    """
    res = evaluate(
        probe(cores=192.0, cores_all=192.0, cores_quota=48.0, ram_gb=251.0,
              gpu="NVIDIA GeForce RTX 3090", n_gpus=2.0,
              vram_total_mb=49152.0, vram_free_mb=48248.0, vram_free_min_mb=24124.0),
        RTX3090X2, Need(pages=2230, max_hours=8.0, budget_usd=3.0, min_net_mbps=20.0),
    )
    assert res.ok, res.detail
    assert res.sizing is not None and res.sizing.shards == 12


def test_plan_uses_the_cores_we_bought_not_the_ones_visible() -> None:
    """🔴 `nproc` = 192 при проданих 48 — це ХОСТ.

    Розкладка на неіснуючих ядрах давала 16 шардів × 8 потоків = 128 потоків
    на 48 куплених. Квота cgroup — єдине місце, де видно нашу частку.
    """
    seen = probe(cores=192.0, cores_all=192.0, cores_quota=48.0,
                 gpu="NVIDIA GeForce RTX 3090", n_gpus=2.0,
                 vram_total_mb=49152.0, vram_free_mb=48248.0, vram_free_min_mb=24124.0)
    res = evaluate(seen, RTX3090X2,
                   Need(pages=2230, max_hours=8.0, budget_usd=3.0, min_net_mbps=20.0))
    assert res.sizing is not None
    # 48 куплених ядер діляться на ФАКТИЧНИЙ флот, а не на 192 видимих:
    # 12 шардів × 4 потоки = 48, і жодне ядро не пропадає.
    assert res.sizing.threads_per_shard == 4
    assert res.sizing.cores_used == 48.0
    assert res.measured["cores_eff"] == 48.0
    assert res.measured["cores_all"] == 192.0
    assert "видно 192" in res.detail


def test_quota_below_the_card_is_a_cpu_lie() -> None:
    """Якщо квота МЕНША за продане — це вже брехня картки, а не наша обачність."""
    res = evaluate(probe(cores=64.0, cores_all=64.0, cores_quota=8.0), V100, NEED)
    assert not res.ok and res.outcome == "cpu_lie"


def test_missing_quota_falls_back_to_nproc() -> None:
    """Голий образ без cgroup-файлів не має ставати вироком."""
    assert evaluate(probe(), V100, NEED).ok


def test_sold_share_does_not_cap_the_plan() -> None:
    """🔴 Продане число — не межа, а комерція: машина 139040 з 40 проданими
    ядрами на 80 видимих видала 2592 стор/год, тобто 65 на продане ядро при
    калібруванні 35. Обмежити план проданим = удвічі занизити прогноз на
    кожному боксі, і власна ж стеля $/1000 почала б різати робочі машини."""
    res = evaluate(probe(cores=128.0, cores_all=128.0), V100, NEED)
    assert res.sizing is not None
    assert res.measured["cores_eff"] == 128.0  # а не 64 з картки


def test_two_card_box_on_light_frames_gets_a_fleet_per_card() -> None:
    """З порогом від площі кадру (7 Мпікс → 1.9 ГБ) двокарткова 3090 дає
    ~11 шардів на карту, і ядра (квота 48) не ріжуть: 22–24 шарди."""
    res = evaluate(
        probe(cores=192.0, cores_all=192.0, cores_quota=48.0, ram_gb=251.0,
              gpu="NVIDIA GeForce RTX 3090", n_gpus=2.0,
              vram_total_mb=49152.0, vram_free_mb=48248.0, vram_free_min_mb=24124.0),
        RTX3090X2, Need(pages=2230, max_hours=8.0, budget_usd=3.0, min_net_mbps=20.0,
                        gb_per_shard=1.9, lines_per_page=70),
    )
    assert res.ok, res.detail
    assert res.sizing is not None and 22 <= res.sizing.shards <= 24


# ---- канал проти обсягу даних ------------------------------------------------
#
# 🔴 15.09.2026: перечитування 14 томів (1.23 ГБ кадрів, 0.33 МБ/стор) відкинуло
# машини з 6.1 і 2.1 Мбіт/с за сталою межею 20 і забанило обидві на 14 днів.


def _bps(mbps: float) -> float:
    return mbps * 1_000_000 / 8


def test_a_slow_but_live_channel_is_enough_for_a_small_volume() -> None:
    need = Need(pages=3720, max_hours=8.0, budget_usd=3.0, data_mb_per_page=0.33)
    res = evaluate(probe(net_bps=_bps(6.1)), V100, need)
    assert res.ok, res.detail
    assert "не встигає" not in res.detail


def test_a_channel_slower_than_the_fleet_slows_the_plan_instead_of_rejecting() -> None:
    need = Need(pages=3720, max_hours=12.0, budget_usd=3.0, data_mb_per_page=2.0,
                max_usd_per_1000_pages=1.0)
    fast = evaluate(probe(), V100, need)
    slow = evaluate(probe(net_bps=_bps(3.0)), V100, need)
    assert fast.ok and slow.ok, slow.detail
    assert slow.sizing.pages_per_hour < fast.sizing.pages_per_hour
    assert slow.hours > fast.hours
    assert "не встигає" in slow.detail


def test_when_the_channel_makes_it_too_expensive_the_verdict_is_neutral() -> None:
    need = Need(pages=3720, max_hours=12.0, budget_usd=3.0, data_mb_per_page=2.0)
    res = evaluate(probe(net_bps=_bps(3.0)), V100, need)
    assert not res.ok and res.outcome == "slow_for_data"
    assert "не встигає" in res.detail


def test_a_dead_channel_is_still_a_ban_even_with_a_known_volume() -> None:
    need = Need(pages=3720, max_hours=12.0, budget_usd=3.0, data_mb_per_page=0.33)
    assert evaluate(probe(net_bps=_bps(0.5)), V100, need).outcome == "slow_net"


def test_without_a_volume_the_old_limit_rejects_but_does_not_ban() -> None:
    res = evaluate(probe(net_bps=_bps(15.0)), V100,
                   Need(pages=1665, max_hours=8.0, budget_usd=3.0))
    assert not res.ok and res.outcome == "slow_for_data"


def test_megabytes_per_page_come_from_cases_that_know_their_volume() -> None:
    import types

    from gpurunner.supervise.htr import plan_mb_per_page

    plan = types.SimpleNamespace(cases=[
        types.SimpleNamespace(n_pages=400, pages_bytes=128_000_000),
        types.SimpleNamespace(n_pages=200, pages_bytes=0),
    ])
    assert plan_mb_per_page(plan) == pytest.approx(0.32)
    assert plan_mb_per_page(types.SimpleNamespace(cases=[])) == 0.0


def test_a_reread_plan_lifts_the_shard_ceiling() -> None:
    """🔴 Число шардів воріт стає стелею регулятора: перечитування Скрибою на
    кеші стояло на 14 шардах при 0.70 ГБ карти на шард (15.09.2026)."""
    base = evaluate(probe(), V100, Need(pages=3720, max_hours=8.0, budget_usd=3.0))
    reread = evaluate(probe(), V100, Need(pages=3720, max_hours=8.0, budget_usd=3.0,
                                          gb_per_shard=1.0, cores_per_shard=0.5))
    assert reread.ok and reread.sizing.shards > base.sizing.shards


def test_plan_sizing_counts_cores_with_the_given_appetite() -> None:
    from gpurunner.core.htr_sizing import plan_sizing

    assert plan_sizing(cores=16, vram_gb=48, gb_per_shard=1.0).shards == 16
    assert plan_sizing(cores=16, vram_gb=48, gb_per_shard=1.0, cores_per_shard=0.5).shards == 32


# ---- ціна входу: качання першого архіву -------------------------------------


def test_a_fat_archive_on_a_thin_channel_is_rejected_before_the_upload() -> None:
    """22.09.2026, spr-160: офер обіцяв 1821 Мбіт/с, проба дала 21.6.

    Машина мала добрі ядра, добру ціну й добрий темп — і пройшла ВСІ ворота.
    А архів 2.25 ГБ на такому каналі лягає 14 хвилин, протягом яких флот не
    читає нічого. Тричі поспіль качання обривалось на `rc=28`, і справа так і
    не почалась.
    """
    need = Need(pages=1753, max_hours=5.0, budget_usd=0.50, disk_gb=60,
                data_mb_per_page=1.28, max_archive_mb=2252.0)
    # `net_bps` — байти за секунду: 2.7 МБ/с = ті самі 21.6 Мбіт/с проби
    res = evaluate(probe(net_bps=2_700_000.0), V100, need)
    assert not res.ok
    assert res.outcome == "slow_for_data"
    assert "тягнеться" in res.detail and "2.2 ГБ" in res.detail


def test_the_same_archive_on_a_fat_channel_passes() -> None:
    """Той самий обсяг на здоровому каналі — не привід відмовляти."""
    need = Need(pages=1753, max_hours=5.0, budget_usd=3.00, disk_gb=60,
                data_mb_per_page=1.28, max_archive_mb=2252.0)
    res = evaluate(probe(net_bps=50_000_000.0), V100, need)
    assert res.ok, res.detail


def test_a_plan_without_archive_sizes_keeps_the_old_behaviour() -> None:
    """`max_archive_mb=0` — обсяг невідомий, і ворота про нього не питають."""
    need = Need(pages=1753, max_hours=5.0, budget_usd=3.00, disk_gb=60,
                data_mb_per_page=1.28)
    assert evaluate(probe(net_bps=2_700_000.0), V100, need).ok


def test_a_short_run_still_gets_the_absolute_floor() -> None:
    """На короткому заході частка від годин дає надто мало — тримає стеля 10 хв.

    Архів 120 МБ на 2.7 МБ/с — це 6 хвилин; захід прогнозується у хвилини, і
    25% від нього були б секундами. Машину відкидати не треба.
    """
    need = Need(pages=60, max_hours=5.0, budget_usd=3.00, disk_gb=60,
                data_mb_per_page=2.0, max_archive_mb=120.0)
    assert evaluate(probe(net_bps=2_700_000.0), V100, need).ok

def test_a_box_that_is_dear_per_core_hour_is_never_taken() -> None:
    """🔴🔴 Аксіома: недогодована машина не «гірший варіант», а не варіант.

    Дві оренди тієї самої ночі, та сама карта (Q RTX 8000) і той самий за
    вагою матеріал — щільність, заміряна по ВСІХ текстах обох прогонів, дала
    медіану 36 і 40 рядків на сторінку:

      · 20 ядер за $0.292/год = $0.0152 за ядро-годину → 13 343 стор. за $0.78
      · 10 ядер за $0.259/год = $0.0270 за ядро-годину →  5 115 стор. за $1.19

    Машина, на 11% дешевша за ГОДИНУ, коштувала на 78% дорожче за РОБОТУ. Її
    пропустили обидві наявні ворота: підлогу темпу вона проходила (2469
    стор/год прогнозу), стелю тисячі — теж ($0.105 проти дозволених $0.35),
    бо обидві виставлені з запасом під найважчий матеріал.

    Поріг $0.020 узятий не зі стелі: по 384 орендах нашої історії медіана
    ціни ядро-години $0.0050, третій квартиль $0.0068. $0.020 — це 4× медіани,
    він відсіває 1% машин і лягає рівно між двома прикладами вище.
    """
    from gpurunner.core.offer_score import MAX_USD_PER_CORE_H

    dear = 0.259 / 9.6
    fine = 0.292 / 19.2
    assert dear > MAX_USD_PER_CORE_H > fine, (
        f"поріг ${MAX_USD_PER_CORE_H} мусить лежати між ${fine:.4f} (взяли й не "
        f"пошкодували) і ${dear:.4f} (взяли й переплатили вдвічі)"
    )


def test_the_core_hour_ceiling_is_not_weakened_by_tiers() -> None:
    """🔴 Як і підлога темпу: порожній ринок має давати чесний `market_empty`,
    а не згоду взяти вдвічі дорожчу за роботу машину. Послаблення по тірах
    стосується бюджету, строку, VRAM і ядер-як-кількості — але не ціни
    одиниці роботи."""
    import inspect

    from gpurunner.core import offer_score as OS

    src = inspect.getsource(OS._score_offer if hasattr(OS, "_score_offer") else OS)
    i = src.index("за ядро-годину")
    window = src[max(0, i - 900):i]
    assert "tier." not in window.split("core_ceiling")[-1], (
        "стеля ядро-години не має множитись на поступку тіра"
    )

def test_a_queue_of_many_small_cases_is_forecast_dearer_than_one_big() -> None:
    """🔴🔴 Найдорожча вада ночі 23.09, і вона була саме в ПРОГНОЗІ.

    `predict_hours`/`predict_cost` умiли брати `cases`, але його не передавав
    ЖОДЕН із восьми викликів, а `Need` поля не мав — тож черга з 237 мікросправ
    важила для воріт і скору точно як одна справа. Насправді 237 × 22 сторінки
    віддали накладним 55% рахунку.

    Заміряно на тому самому залізі (10 ядер, $0.259/год, щільність 41):
    5115 сторінок як одна справа — $0.63, як 237 справ — $1.09, факт — $1.19.
    """
    from gpurunner.core.htr_sizing import plan_sizing, predict_cost

    s = plan_sizing(vram_gb=45.0, cores=9.6, num_gpus=1, lines_per_page=41.0)
    one = predict_cost(5115, s, 0.259, cases=1)
    many = predict_cost(5115, s, 0.259, cases=237)

    assert many > one * 1.5, (
        f"черга з 237 справ прогнозується як ${many:.3f} проти ${one:.3f} за одну "
        f"— різниця мусить бути відчутною, інакше ворота її не побачать"
    )
    # І не «як завгодно дорожче»: факт тієї ночі $1.190, тримаємось у ±20%.
    assert 0.8 * 1.190 <= many <= 1.2 * 1.190, f"прогноз ${many:.3f} проти факту $1.190"


def test_every_forecast_call_passes_the_queue_size() -> None:
    """🔴 Приймач від повернення вади: поле є, а не передається.

    Саме так вона й виглядала — `PER_CASE_SEC` жив у моделі, `cases` був у
    підписі, і все це множилось на одиницю, бо викликачі про нього не знали.
    Сторож розбирає ДЖЕРЕЛО обох модулів: кожен виклик прогнозу мусить нести
    `cases`.
    """
    import ast
    import inspect

    from gpurunner.core import offer_score as OS
    from gpurunner.supervise import gate as GT

    bad: list[str] = []
    for mod in (OS, GT):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
            if name not in ("predict_hours", "predict_cost"):
                continue
            if not any(kw.arg == "cases" for kw in node.keywords):
                bad.append(f"{mod.__name__}:{node.lineno} {name}(…) без cases")
    assert not bad, "прогноз кличеться без розміру черги:\n  " + "\n  ".join(bad)


def test_a_machine_known_to_boot_slowly_is_counted_as_slow() -> None:
    """🔴🔴 Стала «5 хвилин холодного старту» однакова для всіх, а хост тягне
    образ СВОЇМ каналом.

    23.09.2026 бокс у Кореї піднімався 27 хвилин — у п'ять разів понад сталу, і
    це оплачений `docker pull`, а не робота. Реєстр цей час пам'ятав і навіть
    пояснював («єдиний сигнал про закешованість образу»), але в РОЗРАХУНОК він
    не йшов: машина, про яку ми знаємо, що вона підніматиметься пів години,
    важила стільки ж, скільки та, що піднімається за 40 с.

    Це те саме правило, яке вже діє для темпу: замір цієї машини б'є модель.
    """
    from gpurunner.core.htr_sizing import OVERHEAD_SEC_COLD, Sizing, predict_hours

    sizing = Sizing(shards=4, threads_per_shard=3, cores_used=12.0,
                    wasted_cores=0.0, limited_by="cpu", pages_per_hour=1000.0)
    blind = predict_hours(1000, sizing, cases=1)
    slow = predict_hours(1000, sizing, cases=1, boot_sec=1620)
    fast = predict_hours(1000, sizing, cases=1, boot_sec=40)

    assert slow > blind > fast, (blind, slow, fast)
    assert (slow - fast) * 3600 == pytest.approx(1580, abs=1), "різниця — це підйом"
    assert blind * 3600 == pytest.approx(
        1000 / 1000 * 3600 + OVERHEAD_SEC_COLD + 27 + 138, abs=2), (
        "без заміру лишається стала — незнання не має ставати нулем")


def test_the_scorer_takes_the_boot_time_from_the_registry() -> None:
    """Число мусить дійти саме до вибору машини, а не лишитись у поясненні."""
    from pathlib import Path

    from tests.srcprobe import method_body

    src = (Path(__file__).resolve().parents[1] / "src" / "gpurunner"
           / "core" / "offer_score.py").read_text(encoding="utf-8")
    body = method_body(src, "score_offer")
    assert 'best_measured or {}).get("boot_sec")' in body
    # 🪤 Саме КІЛЬКІСТЬ, а не наявність: спершу тут стояв пошук підрядка, і
    # мутація «прибрати `boot_sec` з розрахунку годин» лишилась зеленою — бо
    # той самий підрядок стояв у розрахунку ЦІНИ. Час і гроші беруть підйом
    # обидва, тож і перевіряти треба обидва.
    assert body.count("boot_sec=boot_sec") == 2, (
        "підйом мусить дійти і в години, і в ціну — "
        f"знайдено {body.count('boot_sec=boot_sec')}")


# ---- ціль темпу на виміряному залізі ----------------------------------------


def test_a_box_whose_real_quota_misses_the_target_is_turned_back_at_the_gate() -> None:
    """🔴🔴 Вибір перевіряв ціль за КАРТКОЮ; ворота — за квотою cgroup.

    Картка каже 64 ядра, контейнер дав 12: на такій квоті ціль недосяжна, і
    бокс гаситься тут, за хвилину оренди, а не після години слабкого заходу.
    """
    from dataclasses import replace

    need = replace(NEED, target_pph=5000.0, lines_per_page=39.0, frame_mpx=5.5,
                   gb_per_shard=gb_per_shard_for(5.5), max_usd_per_1000_pages=0.40)
    ok = evaluate(probe(), V100, need)
    assert ok.ok, ok.detail
    starved = evaluate(probe(cores_quota=12.0), V100, need)
    assert not starved.ok
    assert starved.outcome in ("below_target", "cpu_lie")


def test_below_target_names_the_numbers() -> None:
    from dataclasses import replace

    need = replace(NEED, target_pph=5000.0, lines_per_page=39.0, frame_mpx=5.5,
                   gb_per_shard=gb_per_shard_for(5.5), max_usd_per_1000_pages=0.40)
    # 64 ядра видно, але пам'яті карти — на один шард: ціль недосяжна.
    res = evaluate(probe(vram_total_mb=8192.0, vram_free_mb=8000.0),
                   {**V100, "gpu_ram": 8 * 1024}, need)
    assert not res.ok
    assert res.outcome == "below_target", res.detail
    assert "цілі" in res.detail


def test_the_pace_uses_the_parallel_channel_the_transport_actually_gets() -> None:
    """Кадри качаються кількома з'єднаннями: темп мусить сповільнювати
    паралельний замір, а не однопотоковий. Машина 45760 (23.09.2026): 10.4 Мбіт/с
    одним потоком, 22.3 паралельно — за першим її відкинуто як «нижче цілі»."""
    need = Need(pages=3720, max_hours=12.0, budget_usd=3.0, data_mb_per_page=2.0,
                max_usd_per_1000_pages=1.0)
    single = evaluate(probe(net_bps=_bps(3.0)), V100, need)
    parallel = evaluate(probe(net_bps=_bps(3.0), net_par_bps=_bps(30.0)), V100, need)
    assert parallel.sizing.pages_per_hour > single.sizing.pages_per_hour


def test_a_target_missed_because_of_the_channel_is_a_network_verdict() -> None:
    """Ціль зрізав канал, а залізо її тримає: вирок про мережу, з причиною."""
    from gpurunner.supervise.gate import GUARANTEE_FACTOR

    need = Need(pages=3720, max_hours=12.0, budget_usd=3.0, data_mb_per_page=2.0,
                max_usd_per_1000_pages=1.0, target_pph=2000)
    fast = evaluate(probe(), V100, need)
    assert fast.ok, fast.detail
    slow = evaluate(probe(net_bps=_bps(3.0)), V100, need)
    base = fast.sizing.pages_per_hour * GUARANTEE_FACTOR
    assert base >= need.target_here, "залізо мусить тримати ціль без каналу"
    assert not slow.ok
    assert slow.outcome == "slow_for_data", slow.outcome
    assert "не встигає" in slow.detail


def test_the_gate_prices_the_hosts_traffic_like_the_selection() -> None:
    """🔴 24.09.2026: трафік хоста в рахунку Vast дорівнював ціні карти.

    Ворота рахують ціну на виміряному залізі тією самою формулою, що й вибір:
    інакше машина, яку вибір оцінив би дорожче, проходила б тут як дешева.
    """
    need = Need(pages=6800, max_hours=8.0, budget_usd=3.00, disk_gb=120,
                min_net_mbps=20.0, data_mb_per_page=10 * 1024 / 6800)
    base = evaluate(probe(), V100, need)
    dear = evaluate(probe(), {**V100, "inet_down_cost": 0.026}, need)
    assert base.ok and dear.ok
    assert base.cost is not None and dear.cost is not None
    assert dear.cost - base.cost == pytest.approx(10 * 0.026)
    # Стеля тисячі, під якою сама година ще проходить, а з трафіком — ні.
    per_1000_hour = 1000 * 0.222 / base.sizing.pages_per_hour
    tight = Need(**{**need.__dict__, "max_usd_per_1000_pages": per_1000_hour * 1.05})
    assert evaluate(probe(), V100, tight).ok
    res = evaluate(probe(), {**V100, "inet_down_cost": 0.067}, tight)
    assert not res.ok and res.outcome == "overpriced"
