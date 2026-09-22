"""Забір декоду з чекпоінтів: куди лягає, в якому порядку, що вважається доказом.

🔴🔴 Тека призначення береться з `out_dir` ПЛАНУ й нізвідки більше. Доти забір
мав власний корінь, і 95 сторінок обома голосами лягли в чужий простір, при
цьому чесно відзвітувавши «на диску Писар 95 · Дяк 95» — саме тому помилку було
легко не помітити.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from gpurunner.htr.fetch_ckpt import (
    cases_from_plan,
    ckpt_keys,
    count_on_disk,
    stamp_meta,
    unpack,
)


class FakeS3:
    """Паginator-сумісний фейк: віддає перелік ключів як справжній R2."""

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def get_paginator(self, _op: str):
        keys = self.keys

        class _P:
            def paginate(self, **kw):
                prefix = kw.get("Prefix") or ""
                yield {"Contents": [{"Key": k, "Size": 10}
                                    for k in keys if k.startswith(prefix)]}

        return _P()


def _make_ckpt(path: Path, *, branch: str, names: list[str]) -> None:
    """Чекпоінт як його пише раннер: гілки голосів окремими теками."""
    stage = path.parent / f"stage-{path.stem}"
    (stage / branch).mkdir(parents=True, exist_ok=True)
    for name in names:
        (stage / branch / name).write_text("текст", encoding="utf-8")
    with tarfile.open(path, "w:gz") as tf:
        tf.add(stage / branch, arcname=branch)


def test_checkpoints_are_ordered_by_number_not_by_name() -> None:
    """🔴 Лексичне сортування зламалось би на `ckpt_0010` проти `ckpt_0009`
    лише з появою десятого — тобто саме тоді, коли справа велика."""
    s3 = FakeS3([f"ckpt/spr-1/m/ckpt_{i:04d}.tgz" for i in (9, 10, 1, 2)])
    keys = ckpt_keys("spr-1", s3=s3)
    assert [k[-8:-4] for k in keys] == ["0001", "0002", "0009", "0010"]


def test_voices_land_in_sibling_directories(tmp_path: Path) -> None:
    """🔴 Без сестринських тек пошук по прогону дасть нуль хітів БЕЗ помилки —
    успішний дорогий захід прочитається як негативний результат."""
    out_root = tmp_path / "prostir" / "reports" / "htr" / "spr-1739"
    tgz = tmp_path / "ckpt_0001.tgz"
    _make_ckpt(tgz, branch="out", names=["0001.txt"])
    unpack(tgz, "spr-1739", out_root)

    second = tmp_path / "ckpt_0002.tgz"
    _make_ckpt(second, branch="out-diak_v4", names=["0001.txt"])
    unpack(second, "spr-1739", out_root)

    assert (out_root / "0001.txt").is_file()
    assert (out_root.parent / "spr-1739-diak_v4" / "0001.txt").is_file()


def test_any_voice_branch_lands_not_only_the_known_ones(tmp_path: Path) -> None:
    """Третій голос і нова версія ваг — теж гілки: перелік їх мовчки викидав."""
    out_root = tmp_path / "reports" / "htr" / "spr-7"
    tgz = tmp_path / "ckpt_0001.tgz"
    _make_ckpt(tgz, branch="out-skryba_v7", names=["0001.txt"])
    assert unpack(tgz, "spr-7", out_root) == {"out-skryba_v7": 1}
    assert (out_root.parent / "spr-7-skryba_v7" / "0001.txt").is_file()
    assert count_on_disk(out_root, ["out", "out-skryba_v7"]) == {"out": 0, "out-skryba_v7": 1}


def test_unpack_goes_where_the_plan_said_not_next_to_the_tool(tmp_path: Path) -> None:
    """Тека виводу — абсолютний шлях у простір ДОСЛІДЖЕННЯ."""
    out_root = tmp_path / "other-space" / "reports" / "htr" / "spr-1283"
    tgz = tmp_path / "ckpt_0001.tgz"
    _make_ckpt(tgz, branch="out", names=["0001.txt", "0002.txt"])
    unpack(tgz, "spr-1283", out_root)
    assert count_on_disk(out_root)["out"] == 2


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    out_root = tmp_path / "out" / "spr-1"
    tgz = tmp_path / "ckpt_0001.tgz"
    _make_ckpt(tgz, branch="out", names=["0001.txt"])
    added = unpack(tgz, "spr-1", out_root, dry=True)
    assert added == {"out": 1}
    assert not out_root.exists()


def test_a_plan_without_out_dir_is_refused(tmp_path: Path) -> None:
    """🔴 Без `out_dir` забір не знає, ЯКОМУ простору належить справа, і
    покладе декод поруч із собою — саме так 95 сторінок опинились у чужому
    проєкті, а звіт при цьому сказав «на диску Писар 95 · Дяк 95»."""
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"cases": [{"case": "spr-1", "n_pages": 95}]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="out_dir"):
        cases_from_plan(plan)


def test_plan_carries_what_the_run_name_cannot(tmp_path: Path) -> None:
    """План несе знаменник, локальну теку кадрів і шифру справи — без них
    прогін приїжджає нічиїм."""
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"cases": [{
        "case": "spr-1739", "n_pages": 95, "out_dir": "E:/prostir/reports/htr/spr-1739",
        "case_dir": "D:/raw/cdiak/spr-1739", "case_key": "CDIAK/127/1078/1739",
        "resume_urls": ["https://r2/a", "https://r2/b"],
    }]}), encoding="utf-8")
    rec = cases_from_plan(plan)[0]
    assert rec["case_key"] == "CDIAK/127/1078/1739"
    assert rec["n_pages"] == 95
    assert len(rec["resume_urls"]) == 2


def test_meta_gets_the_local_frames_dir_not_the_container_path(tmp_path: Path) -> None:
    """🔴 `case_dir` у меті з хмари — шлях КОНТЕЙНЕРА. Локально це означає
    «кропів немає», тобто очний крок, заради якого прогін і робиться, стає
    неможливим (480 метів зі 511 несли шлях боксу)."""
    out_root = tmp_path / "out" / "spr-1"
    out_root.mkdir(parents=True)
    (out_root / "_htr_meta.json").write_text(
        json.dumps({"case_dir": "/tmp/htrcase/pages_dl_07"}), encoding="utf-8")
    frames = tmp_path / "raw" / "spr-1"
    frames.mkdir(parents=True)

    stamp_meta(out_root, case_key="CDIAK/127/1078/1739", frames_dir=frames)

    meta = json.loads((out_root / "_htr_meta.json").read_text(encoding="utf-8"))
    assert meta["case_dir"] == str(frames)
    assert meta["case_dir_cloud"] == "/tmp/htrcase/pages_dl_07"
    assert meta["case_key"] == "CDIAK/127/1078/1739"


def test_an_existing_case_key_is_not_overwritten(tmp_path: Path) -> None:
    out_root = tmp_path / "out" / "spr-1"
    out_root.mkdir(parents=True)
    (out_root / "_htr_meta.json").write_text(
        json.dumps({"case_key": "DAVO/904/24"}), encoding="utf-8")
    stamp_meta(out_root, case_key="CDIAK/127/1078/1739")
    meta = json.loads((out_root / "_htr_meta.json").read_text(encoding="utf-8"))
    assert meta["case_key"] == "DAVO/904/24"


def test_the_proof_is_the_disk_not_the_sum_of_archives(tmp_path: Path) -> None:
    """🔴 Пізніші чекпоінти перекривають ранні тими самими іменами, тож сума
    доданого завищує. Рахувати треба те, що лежить."""
    out_root = tmp_path / "out" / "spr-1"
    first = tmp_path / "ckpt_0001.tgz"
    second = tmp_path / "ckpt_0002.tgz"
    _make_ckpt(first, branch="out", names=["0001.txt", "0002.txt"])
    _make_ckpt(second, branch="out", names=["0002.txt", "0003.txt"])

    added = sum(unpack(t, "spr-1", out_root).get("out", 0) for t in (first, second))
    assert added == 4, "сума з архівів рахує 0002.txt двічі"
    assert count_on_disk(out_root)["out"] == 3
