"""`-p engine_patches=off`: рушій «до 0.21» на тій самій машині.

Чесний замір прискорення — та сама справа на тому самому боксі з патчами й
без. Вимикачі є в раннері Нишпорки, а наглядач їх не прокидав.
"""

from __future__ import annotations

from pathlib import Path

from gpurunner._embedded import htr_case_runner as runner

_ALL = ["--no-fast-geom", "--no-fast-order", "--no-fast-seam", "--no-seg-resize",
        "--no-fast-clahe"]


def _runner(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "htr_case_run.py"
    p.write_text(text, encoding="utf-8")
    return p


def test_off_switches_every_patch_the_runner_knows(tmp_path: Path) -> None:
    full = _runner(tmp_path, " ".join(f'"{s}"' for s in runner.ENGINE_PATCH_SWITCHES))
    assert runner._engine_patch_flags({"engine_patches": "OFF"}, full) == _ALL


def test_old_runner_gets_only_what_it_knows(tmp_path: Path) -> None:
    old = _runner(tmp_path, '"--fast-geom" "--seg-resize"')
    got = runner._engine_patch_flags({"engine_patches": "off"}, old)
    assert got == ["--no-fast-geom", "--no-seg-resize"]


def test_default_keeps_patches_on(tmp_path: Path) -> None:
    full = _runner(tmp_path, " ".join(f'"{s}"' for s in runner.ENGINE_PATCH_SWITCHES))
    for params in ({}, {"engine_patches": ""}, {"engine_patches": "on"}):
        assert runner._engine_patch_flags(params, full) == []


def test_both_command_builders_pass_it() -> None:
    text = Path(runner.__file__).read_text(encoding="utf-8")
    assert text.count("base += _engine_patch_flags(params") == 2
