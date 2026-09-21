"""Черга томів на одному боксі: передзавантаження, повтор, чесний прогрес.

🔴 irnbuv1 q23 (14.09.2026, RTX 2080 Ti×2, канал 30 Мбіт/с): 23 томи йшли
чергою, і між кожними двома бокс стояв на качанні кадрів; 8 томів упали без
жодного чекпоінта, провал лише писався в лог, а новий том успадковував «11 збоїв»
і темп попереднього. Наглядач до кінця бачив їх `running` з нулем сторінок.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpurunner._embedded import htr_case_runner as runner

GIB = 2 ** 30


@pytest.fixture
def append_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "_append.jsonl"
    monkeypatch.setattr(runner, "APPEND_FILE", str(path))
    monkeypatch.setattr(runner, "IDLE_POLL_SEC", 0)
    return path


# ---- передзавантаження ----------------------------------------------------------


def test_prefetched_frames_are_handed_over(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_disk_free_bytes", lambda p="/tmp": 100 * GIB)
    monkeypatch.setattr(runner, "_set_phase", lambda *a, **kw: None)
    seen: list = []

    def fake_download(url, local, name, tries=2, heartbeat=True, min_free_bytes=0):
        seen.append((heartbeat, min_free_bytes))
        Path(local).write_bytes(b"tar")
        return 0

    monkeypatch.setattr(runner, "_download_with_heartbeat", fake_download)
    local = tmp_path / "prefetch_02" / "t02.tar"
    pf = runner._Prefetch("https://r2/t02.tar", local)
    assert pf.start()
    assert pf.take("https://r2/t02.tar") == local
    # фонове качання не публікує фазу й стереже диск поточного тому
    assert seen == [(False, runner.PREFETCH_ABORT_FREE_BYTES)]


def test_a_failed_prefetch_falls_back_to_a_normal_download(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_disk_free_bytes", lambda p="/tmp": 100 * GIB)
    monkeypatch.setattr(runner, "_set_phase", lambda *a, **kw: None)

    def fake_download(url, local, name, tries=2, heartbeat=True, min_free_bytes=0):
        Path(local).write_bytes(b"half")
        return 28

    monkeypatch.setattr(runner, "_download_with_heartbeat", fake_download)
    local = tmp_path / "prefetch_02" / "t02.tar"
    pf = runner._Prefetch("https://r2/t02.tar", local)
    assert pf.start()
    assert pf.take("https://r2/t02.tar") is None
    assert not local.exists(), "огризок невдалого качання не лишається на диску"


def test_a_prefetch_for_another_url_is_not_used(tmp_path: Path, monkeypatch) -> None:
    pf = runner._Prefetch("https://r2/t02.tar", tmp_path / "t02.tar")
    assert pf.take("https://r2/t03.tar") is None


def test_no_prefetch_when_the_disk_is_short(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_disk_free_bytes", lambda p="/tmp": 3 * GIB)
    started: list = []
    monkeypatch.setattr(runner, "_download_with_heartbeat",
                        lambda *a, **kw: started.append(a) or 0)
    prefetches: dict = {}
    runner._start_prefetch([{"case": "а"}, {"case": "б", "pages_url": "https://r2/b.tar"}],
                           2, prefetches)
    assert prefetches == {} and started == []


def test_prefetch_skips_the_end_of_the_queue_and_volumes_without_url() -> None:
    prefetches: dict = {}
    runner._start_prefetch([{"case": "а"}], 2, prefetches)
    runner._start_prefetch([{"case": "а"}, {"case": "б"}], 2, prefetches)
    assert prefetches == {}


def test_a_download_is_cut_when_the_disk_runs_low(tmp_path: Path, monkeypatch) -> None:
    phases: list = []
    monkeypatch.setattr(runner, "_set_phase", lambda *a, **kw: phases.append(a))
    monkeypatch.setattr(runner, "DOWNLOAD_HEARTBEAT_SEC", 0.01)
    monkeypatch.setattr(runner, "_disk_free_bytes", lambda p="/tmp": 1 * GIB)
    local = tmp_path / "t02.tar"

    class _Proc:
        killed = False

        def wait(self, timeout=None):
            if self.killed:
                return -9
            local.write_bytes(b"x" * 10)
            raise runner.subprocess.TimeoutExpired("curl", timeout)

        def kill(self):
            self.killed = True

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: _Proc())
    rc = runner._download_with_heartbeat("https://r2/t02.tar", local, "t02.tar",
                                         heartbeat=False, min_free_bytes=2 * GIB)
    assert rc == -2
    assert not local.exists()
    assert phases == [], "фонове качання не перебиває фазу поточного тому"


def test_retries_use_a_softer_speed_floor(tmp_path: Path, monkeypatch) -> None:
    """🔴 irnbuv1 q23: маршрут до R2 просідав нижче 1 МБ/с, обидві спроби
    обривались на `rc=28`, і том на 128 МБ падав, не прочитавши жодного кадру.
    Перша спроба строга (ловить мертвий канал), повтори — м'якші."""
    monkeypatch.setattr(runner, "_set_phase", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "DOWNLOAD_RETRY_SLEEP_SEC", 0)
    monkeypatch.setattr(runner, "_DOWNLOAD_SEEN_BPS", [])
    floors: list = []
    rcs = iter([28, 28, 0])

    class _Proc:
        def __init__(self, cmd):
            floors.append(int(cmd[cmd.index("--speed-limit") + 1]))

        def wait(self, timeout=None):
            return next(rcs)

    monkeypatch.setattr(runner.subprocess, "Popen", lambda cmd, **kw: _Proc(cmd))
    assert runner._download_with_heartbeat("https://r2/t03.tar", tmp_path / "t03.tar",
                                           "t03.tar") == 0
    assert floors == [runner.DOWNLOAD_RETRY_MIN_BPS, runner.DOWNLOAD_DEAD_BPS,
                      runner.DOWNLOAD_DEAD_BPS]


def test_the_floor_follows_the_measured_channel(monkeypatch) -> None:
    """🔴 Стала 1 МіБ/с обривала кожен том на каналі 6 Мбіт/с (0.76 МБ/с) —
    до 4 хв на том. Підлога від виміряного тим самим маршрутом."""
    monkeypatch.setattr(runner, "_DOWNLOAD_SEEN_BPS", [760_000.0])      # ~6.1 Мбіт/с
    assert runner._download_floor(1) == 228_000
    assert runner._download_floor(2) == runner.DOWNLOAD_DEAD_BPS
    monkeypatch.setattr(runner, "_DOWNLOAD_SEEN_BPS", [50_000_000.0])
    assert runner._download_floor(1) == runner.DOWNLOAD_MIN_BPS


def test_the_download_loop_takes_prefetched_frames_before_curl() -> None:
    body = Path(runner.__file__).read_text(encoding="utf-8").split("def _run_case(")[1]
    take = body.index("prefetched.take(url)")
    assert take < body.index("_download_with_heartbeat(url, local, name)")
    assert body.index("on_ready()") < body.index('forced_pages = params.get("_pages_root")')


# ---- черга --------------------------------------------------------------------


def _run_ok(merged):
    return {"case": merged["case"], "complete": True, "n_pages_txt": 5}


def test_a_failed_volume_is_retried_once_at_the_end(append_file: Path, monkeypatch) -> None:
    calls: list = []
    published: dict = {}

    def fake(merged):
        calls.append((merged["case"], merged["_case_index"], merged.get("_on_frames_ready")))
        if merged["case"] == "б" and sum(1 for c in calls if c[0] == "б") == 1:
            raise RuntimeError("curl https://r2/b.tar провалився (rc=28)")
        if merged["case"] == "в":
            published["results"] = [dict(e) for e in runner._QUEUE_RESULTS]
        return _run_ok(merged)

    monkeypatch.setattr(runner, "_run_case", fake)
    monkeypatch.setattr(runner, "_set_phase", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_drop_case_frames", lambda merged: None)

    out = runner._main_inner({"cases": [{"case": "а"}, {"case": "б"}, {"case": "в"}]})

    assert [(c, i) for c, i, _ in calls] == [("а", 1), ("б", 2), ("в", 3), ("б", 2)]
    assert calls[-1][2] is None, "повтор не тягне кадри наступного тому — він уже прочитаний"
    assert all(r["complete"] for r in out["results"])
    # поки йшов том «в», наглядач уже бачив провал «б» і знав, що буде повтор
    failed = next(e for e in published["results"] if e["index"] == 2)
    assert failed["complete"] is False and failed["retry_pending"] is True
    assert "rc=28" in failed["error"]
    assert [e["attempt"] for e in runner._QUEUE_RESULTS] == [1, 2, 1]
    assert runner._LAST_PROGRESS["results"] == runner._QUEUE_RESULTS


def test_retry_can_be_switched_off(append_file: Path, monkeypatch) -> None:
    calls: list = []

    def fake(merged):
        calls.append(merged["case"])
        return {"case": merged["case"], "complete": merged["case"] != "б"}

    monkeypatch.setattr(runner, "_run_case", fake)
    monkeypatch.setattr(runner, "_set_phase", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_drop_case_frames", lambda merged: None)
    with pytest.raises(RuntimeError, match="неповні"):
        runner._main_inner({"cases": [{"case": "а"}, {"case": "б"}], "queue_retry_passes": 0})
    assert calls == ["а", "б"]
    assert runner._QUEUE_RESULTS[1]["retry_pending"] is False


def test_a_new_volume_does_not_inherit_the_previous_counters(append_file: Path,
                                                             monkeypatch) -> None:
    starting: list = []
    monkeypatch.setattr(runner, "_run_case", _run_ok)
    monkeypatch.setattr(runner, "_drop_case_frames", lambda merged: None)
    monkeypatch.setattr(runner, "_set_phase",
                        lambda phase, **kw: phase == "starting" and starting.append(kw))
    runner._main_inner({"cases": [{"case": "а"}, {"case": "б"}]})
    assert len(starting) == 2
    for kw in starting:
        assert kw["pages_failed"] == 0 and kw["pages_per_hour"] == 0
        assert kw["oom_events"] == 0 and kw["eta_sec"] is None


def test_fleet_memory_and_prefetch_travel_along_the_queue(append_file: Path,
                                                           monkeypatch) -> None:
    memories: list = []
    prefetched: list = []

    def fake(merged):
        memories.append(merged["_fleet_memory"])
        merged["_fleet_memory"][merged["case"]] = True
        merged["_on_frames_ready"]()
        return _run_ok(merged)

    monkeypatch.setattr(runner, "_run_case", fake)
    monkeypatch.setattr(runner, "_set_phase", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_drop_case_frames", lambda merged: None)
    monkeypatch.setattr(runner, "_start_prefetch",
                        lambda queue, n, prefetches: prefetched.append(n))
    runner._main_inner({"cases": [{"case": "а"}, {"case": "б"}]})
    assert memories[0] is memories[1]
    assert memories[1] == {"а": True, "б": True}
    assert prefetched == [2, 3]


def test_frames_of_a_closed_volume_are_dropped(append_file: Path, tmp_path: Path,
                                               monkeypatch) -> None:
    real_drop = runner._drop_case_frames
    dropped: list = []
    monkeypatch.setattr(runner, "_run_case", _run_ok)
    monkeypatch.setattr(runner, "_set_phase", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_drop_case_frames",
                        lambda merged: dropped.append(merged["_pages_dl"]))
    runner._main_inner({"cases": [{"case": "а"}, {"case": "б"}]})
    assert dropped == ["/tmp/htrcase/pages_dl_01", "/tmp/htrcase/pages_dl_02"]

    dirs = {k: tmp_path / k for k in ("dl", "unpack", "prefetch")}
    for d in dirs.values():
        (d / "sub").mkdir(parents=True)
        (d / "sub" / "0001_L.jpg").write_bytes(b"jpg")
    real_drop({"_pages_dl": str(dirs["dl"]), "_pages_unpack": str(dirs["unpack"]),
               "_prefetch_dir": str(dirs["prefetch"])})
    assert not any(d.exists() for d in dirs.values())


# ---- кеш сегментації зі старих чекпоінтів ------------------------------------


def test_legacy_seg_cache_is_adopted_into_the_stable_folder(tmp_path: Path) -> None:
    """🔴 irnbuv1 q9 (15.09.2026): засів лежав у `pages_dl_01__085f51d2`, а раннер
    шукав `case` — t03, t07, t08 сегментувались наново з кешем на тому ж диску."""
    seg = tmp_path / "data" / "derived" / "htr_seg"
    legacy = seg / "pages_dl_01__085f51d2"
    legacy.mkdir(parents=True)
    for stem in ("0001", "0002"):
        (legacy / f"{stem}.o0.c400.seg.json.gz").write_bytes(b"old")
    (legacy / "note.txt").write_text("не кеш", encoding="utf-8")
    stable = tmp_path / runner.SEG_CACHE_ARC
    stable.mkdir(parents=True)
    (stable / "0002.o0.c400.seg.json.gz").write_bytes(b"new")

    assert runner._adopt_legacy_seg_cache(tmp_path) == 1
    assert (stable / "0001.o0.c400.seg.json.gz").read_bytes() == b"old"
    assert (stable / "0002.o0.c400.seg.json.gz").read_bytes() == b"new", "наявний не затирається"
    assert runner._adopt_legacy_seg_cache(tmp_path) == 0


def test_no_seg_cache_folder_means_nothing_to_adopt(tmp_path: Path) -> None:
    assert runner._adopt_legacy_seg_cache(tmp_path) == 0


def test_adoption_runs_after_resume_and_before_the_fleet() -> None:
    body = Path(runner.__file__).read_text(encoding="utf-8").split("def _run_case(")[1]
    adopt = body.index("_adopt_legacy_seg_cache(work)")
    assert body.index("з хмари підхоплено") < adopt < body.index("_start_shard(k, denom")
