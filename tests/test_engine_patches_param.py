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


def test_summary_versions_come_from_distribution_metadata(monkeypatch) -> None:
    """🔴 kraken 7.x не має `__version__` — у підсумках справ стояло «?», і з
    підсумку не було видно, яким kraken читано."""
    import sys
    import types

    fake = types.ModuleType("kraken")            # без `__version__`, як 7.x
    monkeypatch.setitem(sys.modules, "kraken", fake)
    monkeypatch.setattr(runner, "_dist_version",
                        lambda dist, fresh=False: {"kraken": "7.1.1"}.get(dist, ""))
    assert runner._versions()["kraken"] == "7.1.1"


def test_summary_carries_model_hashes(tmp_path: Path) -> None:
    """Ім'я ваг не каже, які ваги читали: перезібрана під тим самим іменем модель
    у підсумку виглядала б тією самою."""
    import hashlib

    a = tmp_path / "pysar_cyr_v19.pt"
    a.write_bytes(b"weights-a")
    got = runner._models_sha256([a, ""])
    assert got == {"pysar_cyr_v19.pt": hashlib.sha256(b"weights-a").hexdigest()}
    a.write_bytes(b"weights-b, other size")
    assert runner._models_sha256([a])["pysar_cyr_v19.pt"] == \
        hashlib.sha256(b"weights-b, other size").hexdigest()
    assert runner._models_sha256([tmp_path / "gone.pt"]) == {"gone.pt": "ERR FileNotFoundError"}


def test_engine_switches_say_what_was_asked(tmp_path: Path) -> None:
    full = _runner(tmp_path, " ".join(f'"{s}"' for s in runner.ENGINE_PATCH_SWITCHES))
    on = runner._engine_switches({}, full)
    assert on == {"fast-geom": "on", "fast-order": "on", "fast-seam": "on",
                  "seg-resize": "on", "fast-clahe": "on", "pysar-fp16": "default"}
    off = runner._engine_switches({"engine_patches": "off", "pysar_fp16": "on"}, full)
    assert off == {"fast-geom": "off", "fast-order": "off", "fast-seam": "off",
                   "seg-resize": "off", "fast-clahe": "off", "pysar-fp16": "on"}


def test_old_runner_lists_no_patches_rather_than_off(tmp_path: Path) -> None:
    old = _runner(tmp_path, '"--fast-geom"')
    assert runner._engine_switches({"engine_patches": "off"}, old) == {
        "fast-geom": "off", "pysar-fp16": "default"}


def test_both_summaries_carry_the_fingerprint() -> None:
    """Підсумок черги (`_finalize_case`) версій не мав зовсім — а керований шлях
    Нишпорки йде саме чергою."""
    text = Path(runner.__file__).read_text(encoding="utf-8")
    assert text.count("**_fingerprint(params") == 2
    assert 'params["_runner_path"] = str(code / "htr_case_run.py")' in text


def test_summary_versions_name_every_gpu_arch(monkeypatch) -> None:
    import sys
    import types

    cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                 get_device_capability=lambda i: (7, 0))
    fake = types.ModuleType("torch")
    fake.version = types.SimpleNamespace(cuda="12.6")
    fake.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(runner, "_dist_version", lambda dist, fresh=False: "")
    got = runner._versions()
    assert got["cuda"] == "12.6" and got["gpu_arch"] == "sm_70,sm_70"
