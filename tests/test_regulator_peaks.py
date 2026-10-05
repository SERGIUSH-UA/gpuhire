"""Пік чи тіснота: флот після пікового OOM росте знову, а вільні ядра стають
сегментаторами.

🔴 Інцидент (05.10.2026, DAZHO 1-78-1037, 1×V100 16 ГБ, квота 17 ядер): ОДИН рядок
«out of memory» від піку одного шарда — і регулятор назавжди зупинив флот на 6
шардах. Карта в середньому була зайнята на 5–9 ГБ із 16 (nvidia-smi по процесах:
спокій ~0.8 ГБ, пік 2.2–3.3 ГБ), 11 ядер стояли, а `card_per_shard` — максимум за
весь захід (14 757 МБ / 7 = 2108) — і без порогу не пускав сьомий шард.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import ClassVar

import pytest

from gpurunner._embedded import htr_case_runner as runner

BASE = {
    "n_gpus": 1, "ceiling": 13, "knee": None, "oom_new": 0,
    "card_total_mb": 16384.0, "card_used_mb": 6000.0, "epoch_rate": None,
    "rates": {}, "pages_left": 5000, "vram_per_shard_mb": 2600.0,
    "rss_per_shard_mb": 2800.0, "ram_limit_mb": 30839.0, "cores": 17.0,
}


# ---- рядок раннера --------------------------------------------------------------


def test_oom_line_of_the_runner_is_parsed() -> None:
    got = runner._parse_oom_line("[htr-oom] stage=seg_net page=0002.jpg outcome=retry "
                                 "alloc_mb=812 free_mb=310 total_mb=16160 wmax=1547")
    assert got["stage"] == "seg_net" and got["outcome"] == "retry"
    assert got["free_mb"] == "310"
    assert runner._parse_oom_line("[htr-run] ✗ 0002.jpg: boom") is None


class _Proc:
    def __init__(self, lines: list[str]):
        self.stdout = io.StringIO("".join(ln + "\n" for ln in lines))


def _pump(lines: list[str], tmp_path: Path, state: dict | None = None) -> tuple[dict, dict]:
    fleet = {"oom_events": 0}
    state = state if state is not None else {"done": 0, "failed": 0, "skipped": 0}
    runner._pump(_Proc(lines), state, fleet, tmp_path / "shard.log")
    return fleet, state


def test_a_recovered_peak_does_not_count_as_a_lost_page(tmp_path: Path) -> None:
    fleet, _ = _pump(["[htr-oom] stage=seg_net page=1.jpg outcome=retry",
                      "[htr-oom] stage=recog_pysar page=2.jpg outcome=split"], tmp_path)
    assert fleet.get("oom_events_total", 0) == 0
    assert fleet["oom_recovered"] == 2


def test_a_lost_page_is_counted_once_and_as_a_peak(tmp_path: Path) -> None:
    """Сторінка, що впала й після повтору: `[htr-oom] … fail`, а слідом — текст
    винятку з «out of memory». Подія одна, і вона пікова."""
    fleet, _ = _pump([
        "[htr-oom] stage=page page=3.jpg outcome=retry",
        "[htr-oom] stage=page page=3.jpg outcome=fail",
        "[htr-run] ✗ 3.jpg: OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB",
    ], tmp_path)
    assert fleet["oom_events_total"] == 1
    assert fleet["oom_peak_total"] == 1


def test_an_old_runner_oom_still_counts_as_before(tmp_path: Path) -> None:
    fleet, _ = _pump(["[htr-run] ✗ 3.jpg: OutOfMemoryError: CUDA out of memory."],
                     tmp_path)
    assert fleet["oom_events_total"] == 1
    assert fleet.get("oom_peak_total", 0) == 0          # без етапу — тіснота


def test_processor_memory_error_is_never_a_peak(tmp_path: Path) -> None:
    fleet, _ = _pump(["[htr-oom] stage=seg_net page=1.jpg outcome=retry",
                      "[htr-run] ✗ 4.jpg: MemoryError: "], tmp_path)
    assert fleet["oom_events_total"] == 1
    assert fleet.get("oom_peak_total", 0) == 0


# ---- пам'ять процесів -----------------------------------------------------------


def test_rest_and_peak_come_from_the_process_distribution() -> None:
    """Числа з живого боксу 05.10.2026: 6 процесів, спокій ~800 МБ, один-два у піку."""
    samples = [
        [802, 1312, 1116, 1328, 818, 802],
        [802, 576, 2280, 818, 818, 802],
        [802, 3310, 802, 802, 592, 802],
        [1116, 2640, 802, 802, 818, 2634],
        [2418, 804, 802, 2204, 818, 802],
        [802, 804, 802, 818, 818, 802],
        [802, 804, 2848, 818, 1198, 802],
        [802, 1116, 2652, 818, 2502, 1114],
    ]
    rest, excess, load = runner._proc_mem_model(samples)
    assert 790 <= rest <= 830
    assert 1900 <= excess <= 2600
    # найгірша вибірка — два-три процеси в піку на шість: ~0.65 ГБ на процес
    assert 600 <= load <= 700


def test_too_few_samples_give_no_model() -> None:
    assert runner._proc_mem_model([[800, 900]] * 3) is None


def test_projection_from_processes_lets_a_16gb_card_hold_more_than_six() -> None:
    """Та сама картка: спокій 0.8 ГБ, пік +1.6 ГБ, у піку разом ~1/6 процесів —
    13 читачів влазять, 14 — ні."""
    s = dict(BASE, proc_rest_mb=800.0, proc_peak_excess_mb=1600.0, proc_peak_load_mb=270.0,
             ram_limit_mb=60000.0, cores=64.0)
    assert runner._growth_limit(s, 13, 1, 16384.0) is None
    why = runner._growth_limit(s, 14, 1, 16384.0)
    assert why and "спокій" in why


def test_a_small_fleet_still_reserves_one_whole_peak() -> None:
    """Два процеси, рідкий пік: навантаження на процес мале, але один пік
    цілком однаково має влізти."""
    s = dict(BASE, proc_rest_mb=800.0, proc_peak_excess_mb=7000.0, proc_peak_load_mb=100.0,
             ram_limit_mb=60000.0, cores=64.0)
    why = runner._growth_limit(s, 10, 1, 16384.0)
    assert why and "один пік 7000" in why


def test_peaks_are_not_stacked_at_their_worst() -> None:
    """🔴 05.10.2026, 1×V100 16 ГБ, 1037: 5 читачів (спокій ~0.6 ГБ) і 4
    сегментатори (~0.4 ГБ), карта зайнята на 7–10 ГБ, піки до +2.4 ГБ, разом у
    піку — до двох процесів. Стара модель («частка в піку × найбільший пік»)
    дала 19.6 ГБ на десять процесів і не пустила п'ятий сегментатор."""
    rest = [594] * 5 + [422] * 4
    samples = []
    for i in range(40):
        s = list(rest)
        if i % 3:                                    # у двох вибірках із трьох — пік
            s[i % 9] += 2360
        if i % 10 == 0:                              # зрідка — два піки разом
            s[(i + 4) % 9] += 2300
        samples.append(s)
    used = sorted(sum(s) for s in samples)
    assert 7000 <= used[-1] <= 10000                 # те, що бачив nvidia-smi
    _rest, excess, load = runner._proc_mem_model(samples)
    s = dict(BASE, readers=5, segs=5, proc_rest_mb=_rest, proc_peak_excess_mb=excess,
             proc_peak_load_mb=load, rest_reader_mb=594.0, rest_seg_mb=422.0,
             ram_limit_mb=60000.0, rss_per_seg_mb=1500.0)
    assert runner._growth_limit(s, 5, 1, 16384.0) is None


def test_container_ram_caps_readers_on_that_box() -> None:
    """Той самий бокс: RSS читача 2.8 ГБ, ліміт контейнера 30 ГБ — понад 9
    читачів не пускає вже RAM, хоч би скільки було VRAM і ядер."""
    s = dict(BASE, proc_rest_mb=800.0, proc_peak_excess_mb=1600.0, proc_peak_load_mb=270.0,
             rss_per_shard_mb=2821.0, ram_limit_mb=30839.0)
    assert runner._growth_limit(s, 9, 1, 16384.0) is None
    assert "межа RAM" in runner._growth_limit(s, 10, 1, 16384.0)


# ---- пік чи тіснота -------------------------------------------------------------


@pytest.mark.parametrize(("args", "kind"), [
    ((1, 0, 4800.0, 16384.0, False), "peak"),     # сторінка впала на піку, спокій 30%
    ((1, 1, 4800.0, 16384.0, False), "rest"),     # серед нових — тіснота / старий раннер
    ((0, 1, 4800.0, 16384.0, False), "rest"),     # SIGKILL / MemoryError
    ((1, 0, 14000.0, 16384.0, False), "rest"),    # карта повна вже у спокої
    ((1, 0, 4800.0, 16384.0, True), "rest"),      # другий пік на тому самому флоті
    ((1, 0, 0.0, 0.0, False), "peak"),            # спокою не виміряно — віримо етапу
])
def test_oom_kind(args: tuple, kind: str) -> None:
    assert runner._oom_kind(*args) == kind


def test_a_full_card_in_a_peak_does_not_drain_readers() -> None:
    """05.10.2026, 1×V100 з сегментаторами: вибірки nvidia-smi по 98% карти при
    спокої флоту 5.6 ГБ злили читачів 5 → 3. Пік переживає повтор сторінки в
    раннері — зливати за ним означає лише втратити темп."""
    target, why, knee = runner._regulate_decision(dict(BASE, n=5, card_used_mb=16000.0,
                                                       pressure_kind="peak"))
    assert (target, knee) == (5, None)
    assert "у піку" in why and "не зливаю" in why


def test_a_card_full_at_rest_still_drains() -> None:
    target, _, knee = runner._regulate_decision(dict(BASE, n=9, card_used_mb=15800.0,
                                                     pressure_kind="rest"))
    assert (target, knee) == (8, 8)


def test_a_peak_oom_shrinks_but_says_the_knee_is_temporary() -> None:
    target, why, knee = runner._regulate_decision(dict(BASE, n=7, oom_new=1,
                                                       oom_kind="peak"))
    assert (target, knee) == (6, 6)
    assert "пік" in why and "хв" in why


# ---- симуляція: поріг піку знімається -------------------------------------------


def _sim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, start: int, pph_of,
         events, procs, hours: float = 1.5, seg_mode: str = "off", hits=None):
    """Флот без процесів. ``events(t, fleet)`` — підкинути OOM; ``procs(n)`` —
    вибірка nvidia-smi по процесах; ``hits(t)`` — чи читач узяв нарізку з кешу."""
    fleet: dict = {"shards": {}, "n_gpus": 1, "n_pages_expected": 100000,
                   "resumed_pages": 0, "denom": runner.REG_DENOM, "oom_events": 0}

    def fake_start(k, denom, base, env, logs_dir, fl, role=None) -> None:
        fl["shards"][k + 1] = {"k": k + 1, "pid": 0, "rc": None, "done": 0, "failed": 0,
                               "skipped": 0, "secs": [11.0], "acc": 0.0,
                               "role": role or "reader"}

    def live() -> int:
        return sum(1 for s in fleet["shards"].values() if s["rc"] is None)

    monkeypatch.setattr(runner, "_start_shard", fake_start)
    monkeypatch.setattr(runner, "_gpu_mem_per_card", lambda: [(live() * 900.0, 16384.0)])
    monkeypatch.setattr(runner, "_gpu_proc_mem", lambda: procs(live()))
    monkeypatch.setattr(runner, "_group_rss_mb", lambda pgid, proc_root="/proc": 2800.0)
    monkeypatch.setattr(runner, "_ram_limit_mb", lambda *a, **k: 60000.0)
    monkeypatch.setattr(runner, "_request_drain", lambda out_dir, k: None)
    for k in range(start):
        fake_start(k, runner.REG_DENOM, None, None, None, fleet)
    reg = runner._FleetRegulator(fleet, out_dir=tmp_path, base=[], env={}, logs_dir=tmp_path,
                                 denom=runner.REG_DENOM, ceiling=13, cores=17, t0=0.0,
                                 seg_mode=seg_mode)
    reg.t_change = 0.0
    t, dt = 0.0, 5.0
    sizes = []
    while t < hours * 3600:
        t += dt
        events(t, fleet)
        readers = [s for s in fleet["shards"].values()
                   if s["rc"] is None and s.get("role") != "seg"]
        rate = pph_of(len(readers))
        for s in readers:
            s["acc"] += rate / len(readers) * dt / 3600.0
            done = int(s["acc"])
            if done > s["done"] and hits is not None:
                s.setdefault("seg_hits", []).append(hits(t))
                del s["seg_hits"][:-runner.REG_SEG_HITS_KEEP]
            if s.get("draining") and done > s["done"]:
                s["rc"] = 0
            s["done"] = done
        for s in fleet["shards"].values():
            if s.get("role") == "seg" and s.get("draining") and s["rc"] is None:
                s["rc"] = 0
        reg.tick(t)
        sizes.append(len(reg.active()))
    return fleet, reg, sizes


def _calm(n: int) -> list[float]:
    return [800.0] * max(1, n - 1) + [2400.0]       # один у піку, решта в спокої


def test_after_a_peak_oom_the_fleet_tries_to_grow_again(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    fired = {"done": False}

    def one_peak(t, fleet):
        if t >= 1200 and not fired["done"]:
            fleet["oom_events_total"] = fleet.get("oom_events_total", 0) + 1
            fleet["oom_peak_total"] = fleet.get("oom_peak_total", 0) + 1
            fired["done"] = True

    _, reg, sizes = _sim(tmp_path, monkeypatch, start=6, pph_of=lambda n: 330.0 * n,
                         events=one_peak, procs=_calm)
    at_oom = sizes[int(1200 / 5)]
    assert max(sizes[int(1200 / 5):]) > at_oom, "флот так і не спробував рости після піку"
    assert not reg.knee_temp


def test_a_second_peak_on_the_same_fleet_is_crowding(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Коли пік на тому самому розмірі повторюється — це вже не випадковий збіг
    піків, а тіснота: поріг стає постійним."""
    hits = {"n": 0}

    def peaks(t, fleet):
        n = sum(1 for s in fleet["shards"].values() if s["rc"] is None)
        if n >= 8 and hits["n"] < 5 and int(t) % 300 == 0:
            fleet["oom_events_total"] = fleet.get("oom_events_total", 0) + 1
            fleet["oom_peak_total"] = fleet.get("oom_peak_total", 0) + 1
            hits["n"] += 1

    _, reg, sizes = _sim(tmp_path, monkeypatch, start=6, pph_of=lambda n: 330.0 * n,
                         events=peaks, procs=_calm, hours=3.0)
    assert reg.knee is not None and not reg.knee_temp
    assert max(sizes[-200:]) < 8


def test_a_temporary_knee_is_not_carried_to_the_next_volume(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch
                                                            ) -> None:
    def one_peak(t, fleet):
        if t == 1200:
            fleet["oom_events_total"] = 1
            fleet["oom_peak_total"] = 1

    _, reg, _ = _sim(tmp_path, monkeypatch, start=6, pph_of=lambda n: 330.0 * n,
                     events=one_peak, procs=_calm, hours=0.4)
    assert reg.knee_temp and reg.knee is not None
    assert reg.remember()["knee"] is None


# ---- сегментатори ---------------------------------------------------------------


SEG = dict(BASE, mode="auto", segs=0, readers=6, miss_frac=0.9, since_change=999.0,
           proc_rest_mb=800.0, proc_peak_excess_mb=1600.0, proc_peak_load_mb=270.0,
           ram_limit_mb=60000.0, rss_per_seg_mb=1500.0)


def test_segmenters_off_means_none() -> None:
    assert runner._segmenter_decision(dict(SEG, mode="off")) == (0, None)


def test_auto_adds_a_segmenter_while_readers_still_cut_pages() -> None:
    want, why = runner._segmenter_decision(SEG)
    assert want == 1 and "сегментаторів 0 → 1" in why


@pytest.mark.parametrize("kw", [
    {"miss_frac": 0.1},                          # читачі вже беруть нарізку з кешу
    {"since_change": 10.0},                      # щойно додали — нехай розженеться
    {"pages_left": 5},                           # хвіст
])
def test_auto_holds(kw: dict) -> None:
    assert runner._segmenter_decision(dict(SEG, **kw))[0] == 0


def test_segmenters_never_take_the_readers_cores() -> None:
    """17 ядер: 6 читачів × 1.25 = 7.5 — сегментаторів влазить 9, десятий — ні."""
    light = dict(SEG, segs=9, rest_reader_mb=600.0, rest_seg_mb=300.0,
                 proc_peak_excess_mb=800.0)
    want, why = runner._segmenter_decision(light)
    assert want == 9 and "ядер" in why


def test_a_light_segmenter_is_counted_by_its_own_rest() -> None:
    """Сегментатор без моделей розпізнавання: його спокій менший за читацький,
    і саме з нього рахується, скільки їх влізе."""
    heavy = runner._segmenter_decision(dict(SEG, mode="9", rest_reader_mb=1200.0,
                                            rest_seg_mb=1200.0))[0]
    light = runner._segmenter_decision(dict(SEG, mode="9", rest_reader_mb=1200.0,
                                            rest_seg_mb=500.0))[0]
    assert light > heavy


def test_a_fixed_number_of_segmenters_is_started_at_once() -> None:
    assert runner._segmenter_decision(dict(SEG, mode="4", rest_seg_mb=500.0))[0] == 4


def test_a_fixed_number_is_a_ceiling_not_all_or_nothing() -> None:
    want, _ = runner._segmenter_decision(dict(SEG, mode="12", rest_reader_mb=1200.0,
                                              rest_seg_mb=500.0))
    assert 0 < want < 12


def test_regulator_grows_segmenters_while_readers_miss(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    fleet, reg, _ = _sim(tmp_path, monkeypatch, start=6, pph_of=lambda n: 330.0 * n,
                         events=lambda t, f: None, procs=_calm, hours=0.5,
                         seg_mode="auto", hits=lambda t: False)
    assert len(reg.segs_active()) >= 3
    assert all(s.get("role") in ("reader", "seg") for s in fleet["shards"].values())


# ---- роль процесу ---------------------------------------------------------------


class _FakePopen:
    calls: ClassVar[list[list[str]]] = []

    def __init__(self, cmd, **kw):
        _FakePopen.calls.append(list(cmd))
        self.pid = 4242
        self.stdout = io.StringIO("")


def test_a_segmenter_is_restarted_as_a_segmenter(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    _FakePopen.calls = []
    monkeypatch.setattr(runner.subprocess, "Popen", _FakePopen)
    fleet: dict = {"shards": {}, "dynamic": True}
    base = ["python", "htr_case_run.py", "--out-dir", str(tmp_path)]
    runner._start_shard(2, 64, base, {}, tmp_path, fleet, role="seg")
    assert "--segment-only" in _FakePopen.calls[-1]
    assert fleet["shards"][3]["role"] == "seg"
    runner._start_shard(2, 64, base, {}, tmp_path, fleet)        # вотчдог піднімає знову
    assert "--segment-only" in _FakePopen.calls[-1]
    runner._start_shard(3, 64, base, {}, tmp_path, fleet)        # новий номер — читач
    assert "--segment-only" not in _FakePopen.calls[-1]
    assert fleet["shards"][4]["role"] == "reader"


def test_segmenters_stay_off_with_an_old_runner(tmp_path: Path) -> None:
    old = tmp_path / "htr_case_run.py"
    old.write_text('ap.add_argument("--claim")\n', encoding="utf-8")
    assert runner._seg_mode_of({"segmenters": "auto"}, old) == "off"
    new = tmp_path / "new_run.py"
    new.write_text('ap.add_argument("--segment-only")\n', encoding="utf-8")
    assert runner._seg_mode_of({"segmenters": "auto"}, new) == "auto"
    assert runner._seg_mode_of({"segmenters": 3}, new) == "3"
    assert runner._seg_mode_of({}, new) == "auto"          # типово — увімкнено
    assert runner._seg_mode_of({"segmenters": "off"}, new) == "off"


def test_stale_segmenter_claims_are_cleared_with_the_readers(tmp_path: Path) -> None:
    for d in (tmp_path / "_claims", tmp_path / "_segq" / "_claims"):
        d.mkdir(parents=True)
        (d / "0001.claim").write_text("123 1/8\n", encoding="utf-8")
    assert runner._clear_claims(tmp_path) == 2


def test_a_segmenter_exit_is_never_a_failed_shard() -> None:
    """05.10.2026: п'ять сегментаторів вийшли з rc=3 («неповно»), наглядач
    прочитав це як смерть флоту й погасив бокс, що дочитував справу."""
    seg = {"role": "seg"}
    assert runner._shard_rc(seg, 3) == 0 and seg["rc_raw"] == 3
    assert runner._shard_rc({"role": "seg"}, None) is None
    assert runner._shard_rc({"role": "reader"}, 3) == 3
    assert runner._shard_rc({}, 1) == 1
