"""`-p form_cells=1`: бланк поклітинно на боксі — той самий раннер, ще голос `-cells`.

Сторінка бланка без сітки тексту голосу клітинок не має, лише позначку; звірка
частини не мусить через це вважати її непрочитаною.
"""

from __future__ import annotations

import json
from pathlib import Path

from gpurunner._embedded import htr_case_runner as runner
from gpurunner.jobs import htr_case


def _code(tmp_path: Path, runner_text: str, *, modules: bool = True) -> Path:
    p = tmp_path / "htr_case_run.py"
    p.write_text(runner_text, encoding="utf-8")
    if modules:
        (tmp_path / "cells_read.py").write_text("", encoding="utf-8")
        (tmp_path / "form_cells.py").write_text("", encoding="utf-8")
    return p


def test_flag_only_when_asked_and_the_runner_can(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    full = _code(tmp_path / "a", '"--form-cells"')
    assert runner._form_cells_flag({"form_cells": "1"}, full) == ["--form-cells"]
    assert runner._form_cells_flag({"form_cells": True}, full) == ["--form-cells"]
    for params in ({}, {"form_cells": ""}, {"form_cells": "false"}):
        assert runner._form_cells_flag(params, full) == []
    (tmp_path / "b").mkdir()
    old = _code(tmp_path / "b", '"--fast-geom"')
    assert runner._form_cells_flag({"form_cells": "1"}, old) == [], "старий раннер упав би"
    (tmp_path / "c").mkdir()
    bare = _code(tmp_path / "c", '"--form-cells"', modules=False)
    assert runner._form_cells_flag({"form_cells": "1"}, bare) == [], "модулів клітинок немає"


def test_both_command_builders_pass_it() -> None:
    text = Path(runner.__file__).read_text(encoding="utf-8")
    assert text.count("base += _form_cells_flag(params") == 2


def test_the_param_reaches_the_box() -> None:
    src = Path(htr_case.__file__).read_text(encoding="utf-8")
    assert '"form_cells": _as_bool(params.get("form_cells"), False)' in src


def test_a_page_without_grid_counts_as_read(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    frames = ["0001.jpg", "0002.jpg"]
    for s in ("0001", "0002"):
        (out / f"{s}.txt").write_text("текст", encoding="utf-8")
    (out / "_htr_meta.json").write_text(json.dumps({"pages": {f: {} for f in frames}}),
                                        encoding="utf-8")
    cells = tmp_path / "out-cells"
    cells.mkdir()
    (cells / "0001.txt").write_text("клітинка", encoding="utf-8")
    assert not runner._part_is_read(out, frames, n_voices=1)
    (cells / "0002.nogrid").write_text("", encoding="utf-8")
    assert runner._part_is_read(out, frames, n_voices=1)
    # звичайний голос позначкою не обходиться
    diak = tmp_path / "out-diak_v10"
    diak.mkdir()
    (diak / "0001.txt").write_text("голос", encoding="utf-8")
    (diak / "0002.nogrid").write_text("", encoding="utf-8")
    assert not runner._part_is_read(out, frames, n_voices=2)


def test_a_mixed_queue_carries_the_case_decision_to_the_pipeline() -> None:
    """Мішана черга: справа актів — з клітинками, метрика — без. Конвеєр має один
    процес раннера на всі справи, тож рішення справи їде рядком черги."""
    text = Path(runner.__file__).read_text(encoding="utf-8")
    assert 'if "form_cells" in cases[i - 1]:' in text
    assert 'entry["form_cells"] = bool(_form_cells_flag(' in text
    assert "{**params, **cases[i - 1]}" in text


def test_the_case_decision_beats_the_job_knob(tmp_path: Path) -> None:
    """Справа перекриває захід в обидва боки (так зливає їх `run_one` і конвеєр)."""
    (tmp_path / "a").mkdir()
    full = _code(tmp_path / "a", '"--form-cells"')
    job_on, job_off = {"form_cells": True}, {}
    assert runner._form_cells_flag({**job_on, "form_cells": "0"}, full) == []
    assert runner._form_cells_flag({**job_off, "form_cells": "1"}, full) == ["--form-cells"]


def test_the_plan_says_the_supervisor_can_do_it() -> None:
    from gpurunner.htr import plan_build
    from gpurunner.supervise import plan as P

    assert 'plan["form_cells_case"] = 1' in Path(plan_build.__file__).read_text(encoding="utf-8")
    assert "form_cells_case" in P._KNOWN_TOP_KEYS
