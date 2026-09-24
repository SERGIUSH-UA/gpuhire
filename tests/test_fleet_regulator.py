"""Флот сам знаходить коліно на ЦІЙ справі й на ЦЬОМУ залізі (10.09.2026).

Щільність справи наперед не вгадується: план рахував флот із площі кадру, і
8591 дістав 28 шардів на 2×V100 при насиченні карти значно раніше — 4850
стор/год проти обіцяних 6160. Регулятор стартує обережно, росте кроками,
міряє темп і пам'ять і зупиняється там, де приріст закінчився або пам'ять
скінчилась. Тут перевіряється рішення (чиста функція) і поведінка на
симульованому флоті, без жодного процесу й карти.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from gpurunner._embedded import htr_case_runner as runner
from gpurunner.jobs import get_job

BASE = {
    "n_gpus": 1, "ceiling": 28, "knee": None, "oom_new": 0,
    "card_total_mb": 32000.0, "card_used_mb": 8000.0, "epoch_rate": None,
    "rates": {}, "pages_left": 5000, "vram_per_shard_mb": 2000.0,
    "rss_per_shard_mb": 1500.0, "ram_limit_mb": 256000.0, "cores": 64.0,
}


def decide(**kw):
    return runner._regulate_decision({**BASE, **kw})


# ---- рішення ------------------------------------------------------------------


def test_no_decision_until_the_epoch_has_measured_something() -> None:
    assert decide(n=4) == (4, None, None)


def test_grows_by_two_per_card_while_throughput_and_memory_allow() -> None:
    target, why, knee = decide(n=4, epoch_rate=1200.0, rates={2: 600})
    assert target == 6 and knee is None
    assert "пробую 6" in why


def test_rolls_back_to_the_smaller_fleet_when_growth_bought_nothing() -> None:
    """🔴 Саме той випадок 8591: більше шардів, той самий темп — лише більше VRAM."""
    target, why, knee = decide(n=10, epoch_rate=2400.0, rates={8: 2400})
    assert (target, knee) == (8, 8)
    assert "коліно" in why


def test_keeps_the_bigger_fleet_when_it_is_still_a_bit_faster() -> None:
    target, _, knee = decide(n=10, epoch_rate=2500.0, rates={8: 2400})
    assert (target, knee) == (10, 10)


def test_after_the_knee_it_does_not_grow_again() -> None:
    assert decide(n=8, knee=8, epoch_rate=2400.0, rates={6: 1800, 8: 2400})[0] == 8


@pytest.mark.parametrize(("kw", "limit"), [
    ({"n": 28, "ceiling": 28}, "стеля"),
    ({"n": 8, "pages_left": 20}, "хвіст"),
    ({"n": 8, "card_total_mb": 0.0}, "VRAM карти невідома"),
    ({"n": 8, "vram_per_shard_mb": 3000.0}, "межа VRAM"),
    ({"n": 4, "rss_per_shard_mb": 10000.0, "ram_limit_mb": 64000.0}, "межа RAM"),
    ({"n": 4, "cores": 6.0}, "межа ядер"),
])
def test_growth_stops_at_every_measured_limit(kw: dict, limit: str) -> None:
    target, why, knee = decide(epoch_rate=5000.0, **kw)
    assert target == kw["n"] and knee is None
    assert limit in why


def test_vram_is_projected_from_the_card_not_from_summed_peaks() -> None:
    """🔴 316-1-39, 10.09.2026: пік шарда 4.0 ГБ × 9 на карту «не влазив», а
    карта з 8 шардами була зайнята на 11.3 ГБ із 32 — рости було куди."""
    target, why, _ = decide(n=16, n_gpus=2, ceiling=18, epoch_rate=1924.0,
                            rates={14: 1700}, card_total_mb=32768.0,
                            card_per_shard_mb=11319 / 8, peak_shard_mb=4024.0,
                            vram_per_shard_mb=4024.0)
    assert target == 18, why


def test_card_projection_still_stops_a_fleet_that_really_does_not_fit() -> None:
    target, why, _ = decide(n=8, epoch_rate=3000.0, rates={6: 2000},
                            card_total_mb=32768.0, card_per_shard_mb=3500.0,
                            peak_shard_mb=4000.0)
    assert target == 8
    assert "виміряно на карті" in why


def test_oom_shrinks_immediately_and_forbids_growing_back() -> None:
    target, why, knee = decide(n=6, oom_new=2)
    assert (target, knee) == (5, 5)
    assert "OOM" in why


def test_a_card_near_full_is_drained_before_it_ooms() -> None:
    target, why, knee = decide(n=6, card_used_mb=31000.0)
    assert (target, knee) == (5, 5)
    assert "зайнята" in why


def test_steps_follow_the_number_of_cards() -> None:
    assert decide(n=8, n_gpus=2, epoch_rate=3000.0, rates={6: 2000})[0] == 12
    assert decide(n=8, n_gpus=2, oom_new=1)[0] == 6


def test_never_shrinks_below_one_shard_per_card() -> None:
    assert decide(n=1, oom_new=3)[0] == 1
    assert decide(n=2, n_gpus=2, oom_new=3)[0] == 2


# ---- симуляція флоту ------------------------------------------------------------


def _simulate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, pph_of, start: int,
              ceiling: int, per_shard_mb: float, card_mb: float = 32000.0,
              hours: float = 3.0, n_gpus: int = 1, memory: dict | None = None,
              page_sec: float = 12.0):
    fleet: dict = {"shards": {}, "n_gpus": n_gpus, "n_pages_expected": 100000,
                   "resumed_pages": 0, "denom": runner.REG_DENOM}

    def fake_start(k, denom, base, env, logs_dir, fl) -> None:
        fl["shards"][k + 1] = {"k": k + 1, "pid": 0, "rc": None, "done": 0, "failed": 0,
                               "skipped": 0, "secs": [page_sec], "acc": 0.0}

    def live() -> int:
        return sum(1 for s in fleet["shards"].values() if s["rc"] is None)

    monkeypatch.setattr(runner, "_start_shard", fake_start)
    monkeypatch.setattr(runner, "_gpu_mem_per_card",
                        lambda: [(live() * per_shard_mb / n_gpus, card_mb)] * n_gpus)
    monkeypatch.setattr(runner, "_group_rss_mb", lambda pgid, proc_root="/proc": 1500.0)
    monkeypatch.setattr(runner, "_ram_limit_mb", lambda *a, **k: 256000.0)
    for k in range(start):
        fake_start(k, runner.REG_DENOM, None, None, None, fleet)
    reg = runner._FleetRegulator(fleet, out_dir=tmp_path, base=[], env={}, logs_dir=tmp_path,
                                 denom=runner.REG_DENOM, ceiling=ceiling, cores=64, t0=0.0,
                                 memory=memory)
    reg.t_change = 0.0
    t, dt = 0.0, 5.0
    while t < hours * 3600:
        t += dt
        act = [s for s in fleet["shards"].values() if s["rc"] is None]
        rate = pph_of(len(act))
        for s in act:
            s["acc"] += rate / len(act) * dt / 3600.0
            done = int(s["acc"])
            if s.get("draining") and done > s["done"]:
                s["rc"] = 0            # дочитав сторінку й вийшов
            s["done"] = done
        reg.tick(t)
    return fleet, reg


def test_fleet_finds_the_knee_where_the_card_saturates(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Карта насичується на 8 шардах: флот доходить до 10, бачить той самий
    темп і повертається на 8 — зливом, без убитих сторінок."""
    fleet, reg = _simulate(tmp_path, monkeypatch, pph_of=lambda n: 300.0 * min(n, 8),
                           start=2, ceiling=28, per_shard_mb=2000.0)
    assert reg.knee == 8
    assert len(reg.active()) == 8
    assert {2, 4, 6, 8, 10} <= set(reg.rates)
    drained = [k for k, s in fleet["shards"].items() if s.get("draining")]
    assert len(drained) == 2 and all(fleet["shards"][k]["rc"] == 0 for k in drained)
    assert all((tmp_path / "_drain" / str(k)).exists() for k in drained)


def test_fleet_stops_at_the_vram_it_measured_not_at_the_plan(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """Темп росте без меж, але кожен шард тримає 3 ГБ: 32-гігабайтна карта
    вміщає 8 (з запасом 15%), і стеля плану в 28 нічого не важить."""
    _, reg = _simulate(tmp_path, monkeypatch, pph_of=lambda n: 300.0 * n,
                       start=2, ceiling=28, per_shard_mb=3000.0)
    assert len(reg.active()) == 8
    assert reg.knee is None
    assert "межа VRAM" in (reg.note or "")


def test_two_cards_grow_evenly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, reg = _simulate(tmp_path, monkeypatch, pph_of=lambda n: 300.0 * min(n, 12),
                       start=4, ceiling=28, per_shard_mb=2000.0, n_gpus=2)
    per_card: dict[int, int] = {}
    for k in reg.active():
        per_card[reg._card_of(k)] = per_card.get(reg._card_of(k), 0) + 1
    assert per_card[0] == per_card[1]


def test_no_decisions_while_a_drained_shard_is_still_reading(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """Злитий шард тримає пам'ять, доки дочитує сторінку: рішення за OOM у цей
    час злили б каскадом пів флоту."""
    fleet, reg = _simulate(tmp_path, monkeypatch, pph_of=lambda n: 300.0 * n,
                           start=4, ceiling=28, per_shard_mb=2000.0, hours=0.01)
    fleet["shards"][4]["draining"] = True
    fleet["oom_events_total"] = 5
    reg.tick(10_000.0)
    assert len(reg.active()) == 3
    assert not any(s.get("draining") for k, s in fleet["shards"].items() if k != 4)


# ---- пам'ять флоту між томами черги ----------------------------------------------
#
# 🔴 irnbuv1 q23, 14.09.2026: 23 томи на одному боксі, кожен стартував з 10 шардів
# при стелі 20, бо регулятор створювався заново на кожен том і дозрівав 4–5 хв,
# а том жив 3–6. Коліно, знайдене на spr-2467 і cdiak1040-1-3, губилось на
# наступному ж томі.


def _material(model: str = "pysar_cyr_v17.pt", voices: tuple = ("diak_cyr_v4.mlmodel",),
              mpx: float = 1.0) -> dict:
    return runner._material_key(model, list(voices), mpx)


def _memory(**kw) -> dict:
    mem = {"n_stable": 16, "knee": None, "card_per_shard": 900, "peak_shard": 2000,
           "rss_per_shard": 1900, "material": _material()}
    mem.update(kw)
    return mem


def test_memory_carries_to_a_volume_of_the_same_material() -> None:
    assert runner._carried_fleet(_memory(), _material(mpx=1.2)) is not None


def test_unknown_frame_area_does_not_block_the_carry() -> None:
    assert runner._carried_fleet(_memory(), _material(mpx=0.0)) is not None


@pytest.mark.parametrize("material", [
    _material(model="skryba_f792_v6.mlmodel"),
    _material(voices=()),
    _material(mpx=2.0),
])
def test_memory_does_not_carry_to_other_material(material: dict) -> None:
    assert runner._carried_fleet(_memory(), material) is None


def test_nothing_is_carried_before_the_first_volume() -> None:
    assert runner._carried_fleet({}, _material()) is None
    assert runner._carried_fleet(None, _material()) is None


def test_the_next_volume_starts_from_what_the_fleet_reached(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    _, first = _simulate(tmp_path, monkeypatch, pph_of=lambda n: 300.0 * n,
                         start=8, ceiling=20, per_shard_mb=1000.0, hours=0.5)
    mem = first.remember()
    assert mem["n_stable"] == len(first.active()) > 8
    assert mem["card_per_shard"] > 0 and mem["rss_per_shard"] > 0

    _, second = _simulate(tmp_path, monkeypatch, pph_of=lambda n: 300.0 * n,
                          start=mem["n_stable"], ceiling=20, per_shard_mb=1000.0,
                          hours=0.001, memory=mem)
    assert second.carried == {"n_stable": mem["n_stable"], "knee": None}
    assert second.card_per_shard >= mem["card_per_shard"]
    assert second.view()["carried"]["n_stable"] == mem["n_stable"]


def test_a_knee_found_on_one_volume_holds_on_the_next(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    _, first = _simulate(tmp_path, monkeypatch, pph_of=lambda n: 300.0 * min(n, 8),
                         start=2, ceiling=28, per_shard_mb=2000.0)
    mem = first.remember()
    assert (mem["knee"], mem["n_stable"]) == (8, 8)

    # наступний том легший, темп росте без меж — але коліно карти вже знайдено
    _, second = _simulate(tmp_path, monkeypatch, pph_of=lambda n: 300.0 * n,
                          start=mem["n_stable"], ceiling=28, per_shard_mb=2000.0,
                          hours=1.0, memory=mem)
    assert len(second.active()) == 8
    assert second.history == []


def test_a_shard_leaving_on_its_own_voids_the_rate_window(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """🔴 spr-2461, 13.09.2026: `rates {"1": 4008}` — на хвості шарди виходили
    самі, і сторінки чотирнадцяти шардів записувались одному. Такий «замір»
    і перенесений на наступний том розмір флоту були б хибні."""
    fleet, reg = _simulate(tmp_path, monkeypatch, pph_of=lambda n: 300.0 * n,
                           start=4, ceiling=4, per_shard_mb=1000.0, hours=0.001)
    reg.window = (0.0, 0, 4)
    fleet["shards"][4]["rc"] = 0          # дочитав свою частку й вийшов
    reg.tick(1000.0)
    assert reg.window is None
    assert 3 not in reg.rates
    assert reg.remember()["n_stable"] == 4


def test_the_start_follows_the_supervisor_estimate_not_the_heavy_default() -> None:
    """🔴 Старт від `max(3.3, оцінка)` ігнорував оцінку наглядача на легких
    кадрах: A10 (spr-2474) стартував із 6 шардів при 592 МБ карти на шард."""
    src = Path(runner.__file__).read_text(encoding="utf-8")
    body = src.split("def _run_case(")[1]
    assert "max(REG_START_GB" not in body
    assert 'float(params.get("vram_gb_per_shard") or 0) or REG_START_GB' in body
    assert "cores_cap" in body.split("start_gb =")[1].split("print(")[0]


# ---- вимір пам'яті ---------------------------------------------------------------


def _proc(root: Path, pid: int, pgrp: int, rss_kb: int, comm: str = "python x") -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "stat").write_text(f"{pid} ({comm}) S 1 {pgrp} {pgrp} 0 -1", encoding="ascii")
    (d / "status").write_text(f"Name:\tpython\nVmRSS:\t{rss_kb} kB\n", encoding="ascii")


def test_rss_is_summed_over_the_whole_process_group(tmp_path: Path) -> None:
    """Шард — це голова `--supervise` і дитина-воркер: рахувати лише голову
    означало б бачити десяту частку пам'яті."""
    root = tmp_path / "proc"
    _proc(root, 700, 700, 100 * 1024)
    _proc(root, 701, 700, 2000 * 1024)
    _proc(root, 800, 800, 9999 * 1024)
    (root / "self").mkdir()
    assert runner._group_rss_mb(700, proc_root=str(root)) == pytest.approx(2100.0)
    assert runner._group_rss_mb(999, proc_root=str(root)) == 0.0


def test_ram_limit_comes_from_cgroup_not_from_the_host(tmp_path: Path) -> None:
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.max").write_text("8589934592\n", encoding="ascii")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       527000000 kB\n", encoding="ascii")
    assert runner._ram_limit_mb(str(cg), str(meminfo)) == pytest.approx(8192.0)
    (cg / "memory.max").write_text("max\n", encoding="ascii")
    assert runner._ram_limit_mb(str(cg), str(meminfo)) == pytest.approx(527000000 / 1024)


def test_cpu_memory_errors_count_as_oom_too() -> None:
    assert runner._is_oom_line("torch.OutOfMemoryError: CUDA out of memory. Tried …")
    assert runner._is_oom_line("✗ 0042.jpg: MemoryError: ")
    assert not runner._is_oom_line("[htr-run] ✓ готово: 12 розпізнано")


# ---- злив і слоти ------------------------------------------------------------------


def test_drain_files_are_written_and_cleared(tmp_path: Path) -> None:
    runner._request_drain(tmp_path, 3)
    runner._request_drain(tmp_path, 5)
    assert runner._clear_drains(tmp_path, 3) == 1
    assert (tmp_path / "_drain" / "5").exists()
    assert runner._clear_drains(tmp_path) == 1
    assert runner._clear_drains(tmp_path / "немає") == 0


def test_a_restarted_shard_is_not_born_drained(tmp_path: Path) -> None:
    runner._request_drain(tmp_path, 2)

    class _Proc:
        pid = 1
        stdout = None

    popen, thread = runner.subprocess.Popen, runner.threading.Thread
    runner.subprocess.Popen = lambda cmd, **kw: _Proc()
    runner.threading.Thread = lambda **kw: types.SimpleNamespace(start=lambda: None)
    try:
        runner._start_shard(1, 64, ["py", "r.py", "--out-dir", str(tmp_path)], {}, tmp_path,
                            {"shards": {}, "n_gpus": 1, "dynamic": True})
    finally:
        runner.subprocess.Popen, runner.threading.Thread = popen, thread
    assert not (tmp_path / "_drain" / "2").exists()


def test_new_shards_never_reuse_a_finished_shard_number(tmp_path: Path) -> None:
    """Номер шарда — ім'я його парту мети: новий власник того самого номера
    переписав би сторінки злитого."""
    fleet = {"shards": {1: {"rc": None}, 2: {"rc": 0}, 3: {"rc": None}}, "n_gpus": 2}
    reg = runner._FleetRegulator.__new__(runner._FleetRegulator)
    reg.fleet, reg.n_gpus, reg.denom = fleet, 2, 64
    k = reg._free_slot()
    assert k not in fleet["shards"]
    assert reg._card_of(k) == 1          # карта 1 тримає лише злитий №2
    assert reg._victims(1, [1, 3, 5]) == [5]


def test_a_drained_shard_is_not_a_fallen_one(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """🚰 904-24-70, 11.09.2026: раннер nyshporka повертав злитому шарду rc=3, і
    справа падала з «шарди впали: [13, 14, 15, 16]», коли флот звужувався 16 → 12."""
    class _P:
        def __init__(self, rc: int) -> None:
            self.returncode = rc

        def poll(self) -> int:
            return self.returncode

    fleet: dict = {"shards": {
        1: {"k": 1, "rc": None, "_proc": _P(0), "last_out": 0.0, "done": 5},
        2: {"k": 2, "rc": None, "_proc": _P(3), "last_out": 0.0, "done": 2, "draining": True},
        3: {"k": 3, "rc": None, "_proc": _P(3), "last_out": 0.0, "done": 1},
    }}
    monkeypatch.setattr(runner, "_publish", lambda *a, **k: None)
    rcs = runner._watch_fleet(fleet, tmp_path, shards=3, stall_sec=900, restart_max=2,
                              base=[], env={}, logs_dir=tmp_path, t0=0.0)
    assert rcs == {1: 0, 2: 0, 3: 3}
    assert fleet["shards"][2]["rc"] == 3          # у стані — як повернув шард


def test_job_regulates_by_default_and_can_be_switched_off() -> None:
    job = get_job("htr_case")()
    assert job.validate_params({"dataset": "o/s"})["regulate"] is True
    assert job.validate_params({"dataset": "o/s", "regulate": "false"})["regulate"] is False


# ---- перечитування з готовою сегментацією ------------------------------------------


def test_cores_per_shard_from_the_plan_lifts_the_core_limit() -> None:
    """На готовій сегментації шард не платить за геометрію й sato — ядер на
    шард треба менше, і стеля за ядрами мусить це знати."""
    assert decide(n=8, epoch_rate=5000.0, rates={6: 4000}, cores=12.0)[0] == 8
    assert decide(n=8, epoch_rate=5000.0, rates={6: 4000}, cores=12.0,
                  cores_per_shard=0.5)[0] == 10


def test_auto_shards_counts_cores_with_the_given_appetite(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_free_vram_per_card", lambda: [32.0])
    monkeypatch.setattr(runner, "_usable_cores", lambda: 8)
    assert runner._auto_shards(1.5) == 8
    assert runner._auto_shards(1.5, cores_per_shard=0.5) == 16


def test_the_box_keeps_the_segmentation_cache_in_a_stable_folder() -> None:
    """🔴 Без явного шляху тека кешу виходила з хешу `/tmp/htrcase/pages_NN` —
    засіяний кеш влучав лише на тому самому місці в черзі."""
    src = Path(runner.__file__).read_text(encoding="utf-8")
    assert '"--seg-cache-dir", str(work / SEG_CACHE_ARC)' in src


def test_job_booleans_and_rerun_knobs_survive_validation() -> None:
    job = get_job("htr_case")()
    out = job.validate_params({"dataset": "o/s", "seg_cache": "false",
                               "cores_per_shard": "0.5", "gpu_lock": "false"})
    assert out["seg_cache"] is False and out["gpu_lock"] is False
    assert out["cores_per_shard"] == 0.5
    assert out["queue_retry_passes"] == 1
    assert job.validate_params({"dataset": "o/s"})["seg_cache"] is True


# ---- швидкі сторінки: регулятор мусить встигати на короткому томі ------------------


def test_slow_pages_keep_the_epochs_the_knees_were_found_with() -> None:
    assert runner._epoch_timing(12.0) == (60.0, 180.0)
    assert runner._epoch_timing(40.0) == (60.0, 180.0)
    assert runner._epoch_timing(0.0) == (60.0, 180.0)
    assert runner._epoch_timing(2.0) == (20.0, 45.0)


def test_fast_pages_let_the_fleet_grow_within_a_short_volume(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """🔴 Скриба на готовій сегментації (15.09.2026): том у 400 сторінок живе ~70 с,
    а регулятор дозрівав 60 + 180 с — за всю чергу флот не зробив жодного кроку."""
    _, reg = _simulate(tmp_path, monkeypatch, pph_of=lambda n: 1500.0 * n, start=4,
                       ceiling=28, per_shard_mb=700.0, hours=0.1, page_sec=2.0)
    assert len(reg.active()) >= 10
    _, slow = _simulate(tmp_path, monkeypatch, pph_of=lambda n: 1500.0 * n, start=4,
                        ceiling=28, per_shard_mb=700.0, hours=0.1, page_sec=12.0)
    assert len(slow.active()) <= 6
