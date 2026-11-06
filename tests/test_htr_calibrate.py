"""Зшивка заходів із матеріалом для калібровки (06.09.2026).

До цього калібровку робили в чиємусь scratchpad: `data/boxes.jsonl` знає темп,
але не знає рядків на сторінку, а мета на диску знає рядки, але не знає темпу.
`stitch_runs` зводить обидва через стан сесій наглядача — і саме цим шляхом
знайдено форму `A/(29+рядки)`.
"""
from __future__ import annotations

import json
from pathlib import Path

from gpurunner.htr.calibrate import Row, bins_table, fit_table, rated, stitch_runs


def _meta(out_dir: Path, lines: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_htr_meta.json").write_text(json.dumps({
        "version": 1, "pages": {f"{i:04d}.jpg": {"lines": n, "sec": 3.0}
                                for i, n in enumerate(lines, 1)}}), encoding="utf-8")


def _state(path: Path, *, session: str, updated: str, cases: list[dict],
           gpu: str = "RTX 3090", shards: int = 8) -> None:
    path.write_text(json.dumps({
        "session": session, "updated": updated, "phase": "done",
        "box": {"gpu": gpu, "num_gpus": 1,
                "sizing": {"shards": shards},
                "measured": {"cores": 24.0, "cores_quota": 11.5, "vram_total_gb": 24.0}},
        "cases": cases,
    }), encoding="utf-8")


def test_stitch_dedups_latest_snapshots_of_the_same_run(tmp_path: Path) -> None:
    """`latest-*.json` дублює кожен захід: 987 записів проти 408 унікальних.
    Дубль рахувався б двічі й тягнув би медіану до себе."""
    out = tmp_path / "out" / "std160"
    _meta(out, [70] * 10)
    case = {"case": "std160", "status": "done", "out_dir": str(out),
            "pages_per_hour": 4600, "pages_done": 160}
    states = tmp_path / "htr"
    states.mkdir()
    _state(states / "s1.json", session="s1", updated="2026-09-05T20:00:00", cases=[case])
    _state(states / "latest-me.json", session="s1", updated="2026-09-05T20:05:00",
           cases=[dict(case, pages_per_hour=4700)])
    rows = stitch_runs(states)
    assert len(rows) == 1
    assert rows[0].actual == 4700, "перемагає свіжіший знімок того самого заходу"
    assert rows[0].lines == 70 and rows[0].shards == 8
    assert rows[0].runner_new


def test_running_cases_are_not_calibration_points(tmp_path: Path) -> None:
    """Темп справи, що ще рахується, — це темп на розгоні, а не замір."""
    states = tmp_path / "htr"
    states.mkdir()
    _state(states / "s.json", session="s", updated="2026-08-01T10:00:00", cases=[
        {"case": "a", "status": "running", "out_dir": "", "pages_per_hour": 3000},
        {"case": "b", "status": "failed", "out_dir": "", "pages_per_hour": 3000},
        {"case": "c", "status": "incomplete", "out_dir": "", "pages_per_hour": 900},
    ])
    rows = stitch_runs(states)
    assert [r.case for r in rows] == ["c"]
    assert rows[0].lines == 0.0 and not rows[0].runner_new


def test_missing_state_dir_is_empty_not_an_error(tmp_path: Path) -> None:
    assert stitch_runs(tmp_path / "немає") == []


def test_bins_and_fit_use_only_rows_with_material() -> None:
    rows = [
        Row("a", "", "RTX 3090", 1, 12, 24, 8, 35, 100, 4600, "2026-09-05"),
        Row("b", "", "RTX 3090", 1, 12, 24, 8, 118, 100, 1400, "2026-09-05"),
        Row("c", "", "RTX 3090", 1, 12, 24, 8, 0, 100, 2000, "2026-09-05"),   # без мети
    ]
    labels = [b[0] for b in bins_table(rows)]
    assert labels == ["<40", "100–130"]
    grid = fit_table(rows)
    assert grid and all(len(t) == 9 for t in grid)
    assert fit_table([rows[2]]) == []
    # стеля за ядрами — вимір сітки, а не зашите число: без неї теж рахуємо
    assert {t[2] for t in grid} == {None, 16_000.0, 18_000.0, 20_000.0}


def test_micro_runs_never_reach_the_calibration() -> None:
    """🔴 Захід на 10 сторінок міряє не темп, а фіксовану ціну справи.

    Замір 21.09.2026: 59 таких заходів (проби, догони, два записи з нулем
    сторінок при додатному темпі) давали медіану прогноз/факт 4.29 проти 1.05
    на решті — і тягли калібровку вгору, тобто змушували модель завищувати
    темп. Ту саму ціну модель рахує окремо, через `OVERHEAD_SEC_*`.
    """
    rows = [
        Row("велика", "", "RTX 3090", 1, 12, 24, 8, 70, 400, 2000, "2026-09-05"),
        Row("проба", "", "RTX 3090", 1, 12, 24, 8, 70, 1, 5, "2026-09-05"),
        Row("догін", "", "RTX 3090", 1, 12, 24, 8, 70, 19, 40, "2026-09-05"),
    ]
    assert [r.case for r in rated(rows)] == ["велика"]
    assert [b[1] for b in bins_table(rows)] == [1]
