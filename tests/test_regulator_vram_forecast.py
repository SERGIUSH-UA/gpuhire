"""Запас VRAM регулятора — з кадрів, що ще попереду, а не з найбільшого піку за захід.

ANRM 134-249 (01.10.2026): перший кадр — етикетка плівки FamilySearch
7461×1802 — узяв 6.8 ГБ проти 2.2 ГБ звичайної сторінки, і регулятор до кінця
заходу тримав під нього запас: 12 шардів на 2×16 ГБ замість 16.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from gpurunner._embedded import htr_case_runner as runner


def _frames(root: Path, sizes: list[tuple[str, tuple[int, int]]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, wh in sizes:
        Image.new("L", wh).save(root / name)


def _regulator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames_dir: Path,
               out_dir: Path, peaks: list[int], *, source: bool = True):
    monkeypatch.setattr(runner, "_ram_limit_mb", lambda *a, **k: 256000.0)
    fleet: dict = {"shards": {1: {"rc": None, "page_peaks": peaks, "vram_peak_mb": max(peaks)}},
                   "n_gpus": 1, "n_pages_expected": 100, "resumed_pages": 0}
    return runner._FleetRegulator(
        fleet, out_dir=out_dir, base=[], env={}, logs_dir=tmp_path,
        denom=runner.REG_DENOM, ceiling=20, cores=64, t0=0.0,
        frames_source=(lambda: [(frames_dir, out_dir)]) if source else None,
        seg_cap_mpx=0.0)


def _case(tmp_path: Path) -> tuple[Path, Path]:
    frames, out = tmp_path / "frames", tmp_path / "out"
    _frames(frames, [("0001.jpg", (7461, 1802))]
            + [(f"{i:04d}.jpg", (3600, 2700)) for i in range(2, 42)])
    out.mkdir()
    return frames, out


def test_seg_input_mpx_follows_the_aspect_and_the_cap(tmp_path: Path) -> None:
    _frames(tmp_path, [("strip.jpg", (7461, 1802)), ("page.jpg", (3600, 2700))])
    assert runner._seg_input_mpx(tmp_path / "strip.jpg") == pytest.approx(13.41, abs=0.01)
    assert runner._seg_input_mpx(tmp_path / "page.jpg") == pytest.approx(4.32, abs=0.01)
    assert runner._seg_input_mpx(tmp_path / "strip.jpg", cap=8.0) == 8.0
    (tmp_path / "empty.jpg").write_bytes(b"")          # заглушка прочитаної частини
    assert runner._seg_input_mpx(tmp_path / "empty.jpg") is None


def test_unread_giant_frame_keeps_the_reserve(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    frames, out = _case(tmp_path)
    reg = _regulator(tmp_path, monkeypatch, frames, out, [2200] * 30)
    got = reg._predicted_peak()
    assert got is not None and got > 6000      # 13.4 Мп проти типових 4.3


def test_once_the_giant_is_read_the_reserve_drops(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    frames, out = _case(tmp_path)
    (out / "0001.txt").write_text("", encoding="utf-8")
    # пік самої етикетки є серед виміряних — і не тримає запас
    reg = _regulator(tmp_path, monkeypatch, frames, out, [2200] * 30 + [6800])
    got = reg._predicted_peak()
    assert got is not None and got < 2500
    reg.sample(1e9)
    assert reg.peak_shard < 3000


def test_too_few_pages_or_no_frames_fall_back_to_the_largest_peak(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames, out = _case(tmp_path)
    (out / "0001.txt").write_text("", encoding="utf-8")
    assert _regulator(tmp_path, monkeypatch, frames, out, [2200] * 5)._predicted_peak() is None
    reg = _regulator(tmp_path, monkeypatch, frames, out, [2200] * 30 + [6800], source=False)
    assert reg._predicted_peak() is None
    reg.sample(1e9)
    assert reg.peak_shard == 6800 + runner.REG_CTX_MB


def test_reserve_lets_eight_shards_on_a_16gb_card_after_the_giant(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Числа живого заходу: 1492 МБ на шард на карті, звичайна сторінка 2.2 ГБ."""
    frames, out = _case(tmp_path)
    (out / "0001.txt").write_text("", encoding="utf-8")
    reg = _regulator(tmp_path, monkeypatch, frames, out, [2222] * 30 + [6792])
    reg.sample(1e9)
    common = {"n_gpus": 2, "ceiling": 21, "knee": None, "oom_new": 0,
              "card_total_mb": 16384.0, "card_used_mb": 9000.0, "epoch_rate": 4883.0,
              "rates": {8: 3528}, "pages_left": 500, "card_per_shard_mb": 1492.0,
              "rss_per_shard_mb": 1500.0, "ram_limit_mb": 120000.0, "cores": 26.0}
    t, why, _ = runner._regulate_decision({**common, "n": 12, "peak_shard_mb": reg.peak_shard})
    assert t == 16, why            # крок — два шарди на карту
    t, why, _ = runner._regulate_decision({**common, "n": 12, "peak_shard_mb": 6792 + 400})
    assert t == 12 and "межа VRAM" in (why or "")
