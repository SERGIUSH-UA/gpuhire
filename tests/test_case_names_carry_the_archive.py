"""Адреси справи несуть шифру: номер справи не унікальний між архівами.

🔴🔴 23.09.2026: теки результату, кадри в сховищі, чекпоінти й службовий
архів звались лише номером справи (`spr-66`). Паралельна сесія ДАХмО ф.315
мала справи з тими самими номерами — у теки наших книг 66–72 приїхали 588
чужих сторінок, а книга 69 ледь не пішла як «уже прочитана» чужим декодом.
"""
from __future__ import annotations

import json
from pathlib import Path

from gpurunner.htr import fetch_ckpt
from gpurunner.htr.plan_build import key_tag, pick_out_dir


def _meta(d: Path, case_key: str) -> None:
    d.mkdir(parents=True)
    (d / "_htr_meta.json").write_text(json.dumps({"case_key": case_key}), encoding="utf-8")


def test_the_tag_is_the_archive_path() -> None:
    assert key_tag("DAHmO/315/66") == "DAHmO-315-66"
    assert key_tag("") == ""


def test_a_folder_of_another_archive_is_never_reused(tmp_path: Path) -> None:
    _meta(tmp_path / "spr-66", "DAHmO/315/66")
    got = pick_out_dir(tmp_path, "spr-66", "CDIAK/127/66")
    assert got.name == "spr-66__CDIAK-127-66"


def test_our_own_folder_is_kept_for_a_rerun(tmp_path: Path) -> None:
    """Дочитування й перепрогін тієї самої справи лягають у звичну теку."""
    _meta(tmp_path / "spr-66", "CDIAK/127/66")
    assert pick_out_dir(tmp_path, "spr-66", "CDIAK/127/66").name == "spr-66"
    assert pick_out_dir(tmp_path, "spr-67", "CDIAK/127/67").name == "spr-67"


def test_checkpoints_are_listed_by_the_exact_prefix(monkeypatch) -> None:
    """`ckpt/spr-66` без роздільника підхоплював чужий архів, іншу модель і `spr-660`."""
    store = [
        "ckpt/CDIAK-127-66/spr-66/pysar_cyr_v17.pt/ckpt_0001.tgz",
        "ckpt/DAHmO-315-66/spr-66/pysar_cyr_v17.pt/ckpt_0001.tgz",
        "ckpt/CDIAK-127-66/spr-66/skryba_f792_v6.mlmodel/ckpt_0001.tgz",
        "ckpt/spr-660/pysar_cyr_v17.pt/ckpt_0001.tgz",
    ]
    asked: list[str] = []

    def fake_ls(prefix="", **kw):
        asked.append(prefix)
        return [{"key": k} for k in store if k.startswith(prefix)]

    monkeypatch.setattr(fetch_ckpt.r2, "ls", fake_ls)
    keys = fetch_ckpt.ckpt_keys("spr-66", prefix="ckpt/CDIAK-127-66/spr-66/pysar_cyr_v17.pt")
    assert keys == [store[0]]
    assert asked[-1].endswith("/")
    # і без плану — не чіпляє `spr-660`
    assert all("spr-660" not in k for k in fetch_ckpt.ckpt_keys("spr-66"))
