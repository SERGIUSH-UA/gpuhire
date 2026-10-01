"""🧩 Архів кадрів частинами: складач плану, план, наглядач і рішення раннера.

30.09.2026, ф.315: справа їхала одним tar на 4–5 ГБ. Після пересадки нова
машина тягла його цілком (14–18 хв простою), а прогноз догону рахував трафік і
час лише від залишку сторінок. Наскрізна поведінка на боксі —
`test_frame_parts_on_the_box.py` (Linux).
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpurunner._embedded import htr_case_runner as runner
from gpurunner.htr import plan_build
from gpurunner.htr.plan_build import (
    BuildOptions,
    build_plan,
    expected_tar_size,
    part_name,
    split_frames,
)
from gpurunner.supervise.decide import RESEAT_BASE_SEC, Cfg, Obs, decide
from gpurunner.supervise.htr import Supervisor, case_archives, case_bytes_to_pull
from gpurunner.supervise.plan import load_plan

GIB = 2 ** 30


class FakeS3:
    """Мінімальний R2: пам'ятає, що залито, і роздає передбачувані посилання."""

    def __init__(self) -> None:
        self.uploaded: list[str] = []
        self.sizes: dict[str, int] = {}

    def upload_file(self, path: str, bucket: str, key: str, **kw: Any) -> None:
        self.uploaded.append(key)
        self.sizes[key] = Path(path).stat().st_size

    def generate_presigned_url(self, op: str, *, Params: dict, ExpiresIn: int) -> str:
        return f"https://r2/{Params['Key']}?op={op}"


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "frames" / "spr-6816"
    directory.mkdir(parents=True)
    for i in range(1, 13):
        (directory / f"{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + bytes([i]) * (20 + i))
    return directory


def _opts(tmp_path: Path, **kw: Any) -> BuildOptions:
    base: dict[str, Any] = {"out_root": (tmp_path / "prostir" / "htr").resolve(),
                            "model": "pysar_cyr_v17.pt", "part_pages": 5}
    base.update(kw)
    return BuildOptions(**base)


def _build(case_dir: Path, opts: BuildOptions, monkeypatch: pytest.MonkeyPatch,
           s3: FakeS3 | None = None, **kw: Any) -> dict[str, Any]:
    from gpurunner.htr import r2

    s3 = s3 or FakeS3()
    monkeypatch.setattr(r2, "client", lambda env=None: s3)
    monkeypatch.setattr(plan_build.r2, "client", lambda env=None: s3)
    monkeypatch.setattr(plan_build.r2, "head", lambda *a, **k: None)
    if "assets" not in kw:
        kw["assets_key"] = "assets/a.tgz"
    return build_plan([case_dir], opts, case_keys=["DAHMO/315/6816"],
                      log=lambda _line: None, **kw)


# ---- нарізка ------------------------------------------------------------------


def test_a_case_that_fits_one_part_is_not_split(tmp_path: Path) -> None:
    frames = [tmp_path / f"{i:04d}.jpg" for i in range(500)]
    assert split_frames(frames, 500) == [frames]
    assert split_frames(frames, 0) == [frames], "0 — не різати"


def test_a_short_tail_joins_the_previous_part(tmp_path: Path) -> None:
    frames = [tmp_path / f"{i:04d}.jpg" for i in range(1079)]
    sizes = [len(p) for p in split_frames(frames, 500)]
    assert sizes == [500, 579], "79 кадрів — не окремий архів"
    longer = [tmp_path / f"{i:04d}.jpg" for i in range(1200)]
    assert [len(p) for p in split_frames(longer, 500)] == [500, 500, 200]
    assert sum(len(p) for p in split_frames(frames, 500)) == 1079


def test_part_names_sort_in_order() -> None:
    names = [part_name("spr-6816", i, 12) for i in (1, 2, 10, 12)]
    assert names == sorted(names)
    assert names[0] == "spr-6816.part-001-of-012.tar"


# ---- складач плану -------------------------------------------------------------


def test_a_big_case_goes_to_the_bucket_in_parts(case_dir, tmp_path, monkeypatch) -> None:
    s3 = FakeS3()
    case = _build(case_dir, _opts(tmp_path), monkeypatch, s3)["cases"][0]
    parts = case["pages_parts"]
    assert [p["n_pages"] for p in parts] == [5, 5, 2]
    assert [f for p in parts for f in p["frames"]] == [f"{i:04d}.jpg" for i in range(1, 13)]
    folder = "cases/DAHMO-315-6816/spr-6816.p5"
    assert [k for k in s3.uploaded if k.startswith("cases/")] == [
        f"{folder}/spr-6816.part-{n:03d}-of-003.tar" for n in (1, 2, 3)]
    assert all(f"/{key}?" in p["url"] for key, p in zip(s3.uploaded, parts, strict=True))
    assert case["pages_url"] == parts[0]["url"], "перша частина — і є `pages_url`"
    assert case["n_pages"] == 12
    assert sum(p["bytes"] for p in parts) == case["pages_bytes"]


def test_each_part_is_the_archive_its_size_was_predicted_for(
        case_dir, tmp_path, monkeypatch) -> None:
    """«Вже залито» питає бакет про розмір КОЖНОЇ частини окремо."""
    s3 = FakeS3()
    _build(case_dir, _opts(tmp_path), monkeypatch, s3)
    frames = plan_build.frames_of(case_dir)
    for key, chunk in zip(s3.uploaded, split_frames(frames, 5), strict=True):
        assert s3.sizes[key] == expected_tar_size(chunk)


def test_parts_already_in_the_bucket_are_not_uploaded_again(
        case_dir, tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime, timedelta

    s3 = FakeS3()
    _build(case_dir, _opts(tmp_path), monkeypatch, s3)
    first, sizes = list(s3.uploaded), dict(s3.sizes)
    later = datetime.now(tz=UTC) + timedelta(hours=1)
    s3.uploaded.clear()
    from gpurunner.htr import r2

    monkeypatch.setattr(r2, "client", lambda env=None: s3)
    monkeypatch.setattr(plan_build.r2, "client", lambda env=None: s3)
    monkeypatch.setattr(plan_build.r2, "head", lambda key, **k: (
        {"size": sizes[key], "modified": later} if key in sizes else None))
    # Друга частина «змінилась»: розмір у бакеті інший — заливається лише вона.
    sizes[first[1]] += 512
    build_plan([case_dir], _opts(tmp_path), case_keys=["DAHMO/315/6816"],
               assets_key="assets/a.tgz", log=lambda _line: None)
    assert s3.uploaded == [first[1]]


def test_a_small_case_keeps_the_single_archive_and_its_old_key(
        case_dir, tmp_path, monkeypatch) -> None:
    """Залите до нарізки лишається чинним: адреса малої справи не змінилась."""
    s3 = FakeS3()
    case = _build(case_dir, _opts(tmp_path, part_pages=500), monkeypatch, s3)["cases"][0]
    assert "pages_parts" not in case
    assert "cases/DAHMO-315-6816/spr-6816.tar" in s3.uploaded
    assert "/cases/DAHMO-315-6816/spr-6816.tar?" in case["pages_url"]


def test_a_change_of_part_size_does_not_reuse_the_old_parts(
        case_dir, tmp_path, monkeypatch) -> None:
    a, b = FakeS3(), FakeS3()
    _build(case_dir, _opts(tmp_path, part_pages=5), monkeypatch, a)
    _build(case_dir, _opts(tmp_path, part_pages=4), monkeypatch, b)
    assert not set(a.uploaded) & set(b.uploaded)


def _box_plan(case_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    assets = tmp_path / "assets.tgz"
    with tarfile.open(assets, "w:gz"):
        pass
    out = tmp_path / "run" / "plan.json"
    out.parent.mkdir(parents=True)
    _build(case_dir, _opts(tmp_path, transport="box"), monkeypatch,
           assets=assets, out_path=out)
    return out


def test_box_transport_packs_every_part_at_home(case_dir, tmp_path, monkeypatch) -> None:
    path = _box_plan(case_dir, tmp_path, monkeypatch)
    case = json.loads(path.read_text(encoding="utf-8"))["cases"][0]
    assert case["pages_url"] == "" and all(p["url"] == "" for p in case["pages_parts"])
    files = [Path(p["path"]) for p in case["pages_parts"]]
    assert all(f.is_file() for f in files) and case["pages_path"] == str(files[0])
    frames = plan_build.frames_of(case_dir)
    for file, chunk in zip(files, split_frames(frames, 5), strict=True):
        assert file.stat().st_size == expected_tar_size(chunk)


def test_a_stale_home_archive_is_packed_again(case_dir, tmp_path, monkeypatch) -> None:
    """🔴 «Архів уже спакований» означало лише «файл є». Архів, зібраний до
    заміни кадрів, їхав на машину як свіжий."""
    path = _box_plan(case_dir, tmp_path, monkeypatch)
    part = Path(json.loads(path.read_text(encoding="utf-8"))["cases"][0]
                ["pages_parts"][0]["path"])
    part.write_bytes(b"old")
    _build(case_dir, _opts(tmp_path, transport="box"), monkeypatch,
           assets=tmp_path / "assets.tgz", out_path=path)
    assert part.stat().st_size == expected_tar_size(plan_build.frames_of(case_dir)[:5])


# ---- план ---------------------------------------------------------------------


def _plan_file(tmp_path: Path, parts: list[dict[str, Any]], n_pages: int = 4,
               **case: Any) -> Path:
    raw = {"assets_url": "https://r2/a.tgz", "budget_usd": 1.0, "max_hours": 4.0,
           "cases": [{"case": "spr-1", "pages_url": parts[0].get("url", ""),
                      "n_pages": n_pages, "out_dir": str(tmp_path / "out" / "spr-1"),
                      "pages_bytes": sum(int(p.get("bytes") or 0) for p in parts),
                      "pages_parts": parts, **case}]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return path


def _two_parts() -> list[dict[str, Any]]:
    return [{"name": "spr-1.part-001-of-002.tar", "url": "https://r2/p1.tar", "n_pages": 2,
             "bytes": 3_000_000, "frames": ["0001.jpg", "0002.jpg"]},
            {"name": "spr-1.part-002-of-002.tar", "url": "https://r2/p2.tar", "n_pages": 2,
             "bytes": 1_000_000, "frames": ["0003.jpg", "0004.jpg"]}]


def test_the_plan_reads_parts_without_a_warning(tmp_path: Path) -> None:
    plan = load_plan(_plan_file(tmp_path, _two_parts()))
    assert [p["n_pages"] for p in plan.cases[0].pages_parts] == [2, 2]
    assert not [w for w in plan.warnings if "pages_parts" in w]


def test_parts_that_do_not_add_up_to_the_case_are_refused(tmp_path: Path) -> None:
    """🔴 Знаменник справи — сума частин; розбіжність вилізла б уже після
    оплаченого заходу, на домашній звірці."""
    with pytest.raises(ValueError, match="сумі частин"):
        load_plan(_plan_file(tmp_path, _two_parts(), n_pages=5))


def test_a_frame_listed_in_two_parts_is_refused(tmp_path: Path) -> None:
    parts = _two_parts()
    parts[1]["frames"] = ["0002.jpg", "0004.jpg"]
    with pytest.raises(ValueError, match="двох частинах"):
        load_plan(_plan_file(tmp_path, parts))


def test_a_part_without_a_link_is_refused(tmp_path: Path) -> None:
    parts = _two_parts()
    parts[1]["url"] = ""
    with pytest.raises(ValueError, match="немає посилання"):
        load_plan(_plan_file(tmp_path, parts))


def test_a_part_that_miscounts_its_frames_is_refused(tmp_path: Path) -> None:
    parts = _two_parts()
    parts[0]["n_pages"] = 3
    with pytest.raises(ValueError, match="кадрів у ній названо"):
        load_plan(_plan_file(tmp_path, parts))


# ---- наглядач -----------------------------------------------------------------


def _sup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parts=None, **case) -> Supervisor:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    plan = load_plan(_plan_file(tmp_path, parts or _two_parts(), **case))
    return Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]


def test_the_box_gets_every_part_with_its_frames(tmp_path, monkeypatch) -> None:
    sup = _sup(tmp_path, monkeypatch)
    case = sup.plan.cases[0]
    params = sup._params_for(case, sup._need_for(4), resume=False, queue=[case])
    assert params["cases"][0]["pages_parts"] == [
        {"url": "https://r2/p1.tar", "frames": ["0001.jpg", "0002.jpg"]},
        {"url": "https://r2/p2.tar", "frames": ["0003.jpg", "0004.jpg"]}]
    # 🔴 На верхньому рівні частин немає: звідти їх успадкувала б наступна справа
    # черги, що їде одним архівом (30.09.2026, spr-6750 ← частини spr-7298).
    assert "pages_parts" not in params
    single = sup._params_for(case, sup._need_for(4), resume=False)
    assert single["pages_parts"] == params["cases"][0]["pages_parts"],         "без черги справа несе частини на верхньому рівні"
    assert params["pages_url"] == "https://r2/p1.tar"
    from gpurunner.jobs.htr_case import HTRCaseJob

    normalized = HTRCaseJob().validate_params(params)
    assert normalized["cases"][0]["pages_parts"] == params["cases"][0]["pages_parts"],         "job не губить частин"


def test_a_reseat_is_priced_by_what_the_box_will_really_pull(tmp_path, monkeypatch) -> None:
    """🔴 Прогноз догону рахував трафік від залишку сторінок. Удома вже лежать
    тексти першої частини: качати лишилось другу — 1 МБ на дві сторінки."""
    sup = _sup(tmp_path, monkeypatch)
    fresh = sup._need_for(4)
    assert fresh.data_mb_per_page == pytest.approx(1.0) and fresh.max_archive_mb == 4.0
    out = Path(sup.plan.cases[0].out_dir)
    out.mkdir(parents=True)
    for stem in ("0001", "0002"):
        (out / f"{stem}.txt").write_text("текст", encoding="utf-8")
    need = sup._need_for(2)
    assert need.max_archive_mb == pytest.approx(1.0)
    assert need.data_mb_per_page == pytest.approx(0.5)


def test_a_case_in_one_archive_is_pulled_whole_until_it_is_done() -> None:
    from types import SimpleNamespace

    whole = SimpleNamespace(pages_parts=[], pages_bytes=4_000_000)
    assert case_bytes_to_pull(whole, {"0001", "0002", "0003"}) == 4_000_000
    parts = SimpleNamespace(pages_parts=_two_parts(), pages_bytes=4_000_000)
    assert case_bytes_to_pull(parts, set()) == 4_000_000
    assert case_bytes_to_pull(parts, {"0001", "0002"}) == 1_000_000
    assert case_bytes_to_pull(parts, {"0001", "0003", "0004"}) == 3_000_000
    assert case_bytes_to_pull(parts, {"0001", "0002", "0003", "0004"}) == 0


def test_the_reseat_cost_follows_the_frames_of_the_current_case(
        tmp_path, monkeypatch) -> None:
    parts = _two_parts()
    parts[0]["bytes"], parts[1]["bytes"] = 2_000_000_000, 2_000_000_000
    sup = _sup(tmp_path, monkeypatch, parts)
    sup._rent_queue = list(sup.plan.cases)
    sup._probe = {"net_par_mbps": 40.0}
    early = sup._reseat_sec({"case_index": 1, "pages_done": 0})
    late = sup._reseat_sec({"case_index": 1, "pages_done": 3})
    assert early == pytest.approx(RESEAT_BASE_SEC + 4e9 * 8 / 1e6 / 40.0)
    assert late < early, "під кінець справи частин із недочитаним менше"
    whole = _sup(tmp_path, monkeypatch, parts)
    whole.plan.cases[0] = replace(whole.plan.cases[0], pages_parts=[])
    whole._rent_queue = list(whole.plan.cases)
    whole._probe = {"net_par_mbps": 40.0}
    assert whole._reseat_sec({"case_index": 1, "pages_done": 3}) == pytest.approx(early), \
        "справа одним архівом качається цілком, скільки б не лишилось"
    assert sup._reseat_sec({"pipeline": True}) == 0.0


def test_a_long_pull_keeps_a_slow_box_reading() -> None:
    """Том на 4.5 ГБ на каналі 38 Мбіт/с — це 16 хв до першої сторінки нової
    машини: пересадка, вигідна за сталих 15 хв, із чесною ціною вже ні."""
    cfg = replace(Cfg(budget_usd=0.5, max_hours=1.0), target_pph=3573.0)
    progress = {"phase": "running", "n_pages_expected": 900, "pages_done": 0,
                "pages_per_hour": 1736, "wall_sec": 1800.0}
    base = dict(progress=progress, box_pages=469, box_sec=960.0, underdeliver_streak=3)
    assert decide(Obs(**base), cfg)[0] == "underdelivers"
    assert decide(Obs(**base, reseat_sec=300 + 4.5e9 * 8 / 1e6 / 38.0), cfg)[0] != "underdelivers"


def test_box_transport_names_the_parts_the_same_for_delivery_and_links(
        case_dir, tmp_path, monkeypatch) -> None:
    from gpurunner.htr import box_transport as bt

    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    plan = load_plan(_box_plan(case_dir, tmp_path, monkeypatch))
    sup = Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]
    case = plan.cases[0]
    links = [p["url"] for p in sup._case_urls(case)["pages_parts"]]
    base = bt.base_url(token=sup._origin_token)
    assert links == [f"{base}/{rel}" for _local, rel in case_archives(case)]
    assert all(Path(local).is_file() for local, _rel in case_archives(case))
    assert sup._case_urls(case)["pages_url"] == links[0]


def test_a_finished_case_is_not_delivered_to_the_next_box(
        case_dir, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    plan = load_plan(_box_plan(case_dir, tmp_path, monkeypatch))
    sup = Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]
    assert sup._pages_bytes_left() > 0
    sup.state.cases[0].status = "done"
    assert sup._cases_to_deliver() == [] and sup._pages_bytes_left() == 0


# ---- рішення раннера ------------------------------------------------------------


def _read_state(work: Path, stems: list[str], *, meta: list[str] | None = None,
                voices: dict[str, list[str]] | None = None) -> Path:
    out = work / "out"
    out.mkdir(parents=True)
    for stem in stems:
        (out / f"{stem}.txt").write_text("текст", encoding="utf-8")
    known = stems if meta is None else meta
    (out / "_htr_meta.part1.json").write_text(
        json.dumps({"pages": {f"{s}.jpg": {} for s in known[:1]}}), encoding="utf-8")
    (out / "_htr_meta.part2.json").write_text(
        json.dumps({"pages": {f"{s}.jpg": {} for s in known[1:]}}), encoding="utf-8")
    for voice, have in (voices or {}).items():
        side = work / f"out-{voice}"
        side.mkdir()
        for stem in have:
            (side / f"{stem}.txt").write_text("голос", encoding="utf-8")
    return out


FRAMES = ["0001.jpg", "0002.jpg", "0003.jpg"]
STEMS = ["0001", "0002", "0003"]


def test_a_part_is_read_only_when_every_frame_has_text_and_a_meta_entry(tmp_path) -> None:
    out = _read_state(tmp_path / "a", STEMS)
    assert runner._part_is_read(out, FRAMES)
    assert not runner._part_is_read(_read_state(tmp_path / "b", STEMS[:2]), FRAMES)
    assert not runner._part_is_read(_read_state(tmp_path / "c", STEMS, meta=STEMS[:2]), FRAMES), \
        "текст без запису в меті шард перечитав би"
    assert not runner._part_is_read(out, [])
    assert not runner._part_is_read(tmp_path / "немає" / "out", FRAMES)


def test_a_part_with_a_voice_missing_is_not_read(tmp_path) -> None:
    """🔴 Шард на голос не дивиться, і сторінка без тексту голосу лишилась би
    без нього назавжди: кадру, щоб дочитати, на боксі не було б."""
    full = _read_state(tmp_path / "a", STEMS, voices={"diak": STEMS, "skryba": STEMS})
    assert runner._part_is_read(full, FRAMES, n_voices=2)
    gap = _read_state(tmp_path / "b", STEMS, voices={"diak": STEMS, "skryba": STEMS[:2]})
    assert not runner._part_is_read(gap, FRAMES, n_voices=2)
    one = _read_state(tmp_path / "c", STEMS, voices={"diak": STEMS})
    assert not runner._part_is_read(one, FRAMES, n_voices=2), "теки голосу ще немає"
    assert runner._part_is_read(one, FRAMES, n_voices=1)


def test_frame_parts_fall_back_to_the_single_archive() -> None:
    assert runner._frame_parts({"pages_url": "https://r2/a.tar"}) == [("https://r2/a.tar", None)]
    assert runner._frame_parts({}) == []
    parts = [{"url": "https://r2/p1.tar", "frames": ["0001.jpg"]},
             {"url": "https://r2/p2.tar", "frames": ["0002.jpg"]}]
    assert runner._frame_parts({"pages_url": "https://r2/p1.tar", "pages_parts": parts}) == [
        ("https://r2/p1.tar", ["0001.jpg"]), ("https://r2/p2.tar", ["0002.jpg"])]


def test_placeholders_keep_the_denominator_and_spare_real_frames(tmp_path) -> None:
    root = tmp_path / "pages"
    root.mkdir()
    (root / "0002.jpg").write_bytes(b"\xff\xd8real")
    made = runner._touch_placeholders(root, [("https://r2/p1.tar", FRAMES)])
    assert made == 2
    assert [p.name for p in runner._script_pages(root)] == FRAMES
    assert (root / "0001.jpg").stat().st_size == 0
    assert (root / "0002.jpg").read_bytes() == b"\xff\xd8real", "справжній кадр не затирається"


def test_the_next_volume_is_prefetched_in_all_its_parts(tmp_path, monkeypatch) -> None:
    """Том частинами мусить приїхати наперед увесь, як приїздив один архів:
    інакше решту частин бокс качав би на старті тому, з флотом у простої."""
    monkeypatch.setattr(runner, "_disk_free_bytes", lambda p="/tmp": 100 * GIB)
    monkeypatch.setattr(runner, "_set_phase", lambda *a, **kw: None)
    got: list[str] = []

    def fake_download(url, local, name, tries=2, heartbeat=True, min_free_bytes=0):
        got.append(name)
        Path(local).write_bytes(b"tar")
        return 0

    monkeypatch.setattr(runner, "_download_with_heartbeat", fake_download)
    urls = ["https://r2/b.part-001-of-002.tar", "https://r2/b.part-002-of-002.tar"]
    pf = runner._PrefetchSet(urls, tmp_path / "prefetch_02")
    assert pf.start()
    assert pf.take(urls[1]).name == "b.part-002-of-002.tar"
    assert pf.take(urls[0]).name == "b.part-001-of-002.tar"
    assert got == ["b.part-001-of-002.tar", "b.part-002-of-002.tar"]
    assert pf.take("https://r2/other.tar") is None


def test_a_prefetch_cut_by_the_disk_leaves_the_rest_to_the_volume(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_disk_free_bytes", lambda p="/tmp": 100 * GIB)
    monkeypatch.setattr(runner, "_set_phase", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_download_with_heartbeat", lambda *a, **kw: -2)
    urls = ["https://r2/b.part-001-of-002.tar", "https://r2/b.part-002-of-002.tar"]
    pf = runner._PrefetchSet(urls, tmp_path / "prefetch_02")
    assert pf.start()
    assert pf.take(urls[0]) is None and pf.take(urls[1]) is None
    assert pf.items[1].rc is None, "другу частину після обриву по диску не починали"


def test_start_prefetch_takes_the_whole_set_for_a_volume_in_parts(monkeypatch) -> None:
    started: list[Any] = []
    monkeypatch.setattr(runner._PrefetchSet, "start", lambda self: started.append(self) or True)
    monkeypatch.setattr(runner._Prefetch, "start", lambda self: started.append(self) or True)
    parts = [{"url": "https://r2/b.part-001-of-002.tar", "frames": ["0001.jpg"]},
             {"url": "https://r2/b.part-002-of-002.tar", "frames": ["0002.jpg"]}]
    queue = [{"case": "а"}, {"case": "б", "pages_url": parts[0]["url"], "pages_parts": parts},
             {"case": "в", "pages_url": "https://r2/c.tar"}]
    prefetches: dict = {}
    runner._start_prefetch(queue, 2, prefetches)
    runner._start_prefetch(queue, 3, prefetches)
    assert isinstance(prefetches[2], runner._PrefetchSet) and len(prefetches[2].items) == 2
    assert isinstance(prefetches[3], runner._Prefetch)


def test_a_case_in_parts_stays_out_of_the_small_case_pipeline() -> None:
    """Конвеєр дрібних справ качає один архів на справу: справа частинами пішла б
    у нього з першою частиною замість усіх."""
    body = Path(runner.__file__).read_text(encoding="utf-8").split("def _main_inner(")[1]
    small = body[body.index("small = ["):body.index("use_pipeline =")]
    assert 'len(c.get("pages_parts") or []) <= 1' in small
