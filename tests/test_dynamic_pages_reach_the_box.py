"""Динамічний розподіл сторінок (клейми) доходить до флоту на боксі (06.09.2026).

Хвіст статичного зрізу `[k::n]` на std160 05.09.2026 — 12% роботи (шарди
фінішують у вікні 110–125 с), на довгих справах ~20%. Раннер уміє `--claim`;
тут перевіряється, що бокс-раннер його ставить, чистить сироти-клейми перед
кожним флотом і не дає клеймам просочитись у догін.
"""
from __future__ import annotations

import types
from pathlib import Path

from gpurunner._embedded import htr_case_runner as runner
from gpurunner.jobs import get_job


def _launch(fleet: dict, shards: int) -> list[list[str]]:
    launched: list[list[str]] = []

    class _Proc:
        pid = 1
        stdout = None

    original_popen, original_thread = runner.subprocess.Popen, runner.threading.Thread
    runner.subprocess.Popen = lambda cmd, **kw: (launched.append(list(cmd)), _Proc())[1]
    runner.threading.Thread = lambda **kw: types.SimpleNamespace(start=lambda: None)
    try:
        base = ["py", "htr_case_run.py", "--device", "cuda:0", "--gpu-lock", "/tmp/gpu.lock"]
        for k in range(shards):
            runner._start_shard(k, shards, base, {}, Path("."), fleet)
    finally:
        runner.subprocess.Popen, runner.threading.Thread = original_popen, original_thread
    return launched


def test_dynamic_fleet_adds_claim_to_every_shard() -> None:
    cmds = _launch({"shards": {}, "n_gpus": 1, "dynamic": True}, 3)
    assert all("--claim" in c and "--shard" in c for c in cmds)


def test_static_fleet_and_single_process_carry_no_claim() -> None:
    assert all("--claim" not in c for c in _launch({"shards": {}, "n_gpus": 1, "dynamic": False}, 3))
    assert all("--claim" not in c for c in _launch({"shards": {}, "n_gpus": 1, "dynamic": True}, 1))


def test_job_defaults_to_dynamic_and_keeps_the_static_fallback() -> None:
    job = get_job("htr_case")()
    assert job.validate_params({"dataset": "o/s"})["dynamic_pages"] is True
    assert job.validate_params({"dataset": "o/s", "dynamic_pages": False})["dynamic_pages"] is False


def test_clear_claims_leaves_texts_and_meta_alone(tmp_path: Path) -> None:
    """Клейми з попереднього флоту несуть чужі pid-и — перевірка життя на
    новому боксі збрехала б. Чистимо їх, і тільки їх."""
    out = tmp_path / "out"
    (out / "_claims").mkdir(parents=True)
    (out / "_claims" / "0001.claim").write_text("1 1/2\n", encoding="utf-8")
    (out / "0001.txt").write_text("текст\n", encoding="utf-8")
    (out / "_htr_meta.json").write_text("{}", encoding="utf-8")
    assert runner._clear_claims(out) == 1
    assert not list((out / "_claims").glob("*.claim"))
    assert (out / "0001.txt").exists() and (out / "_htr_meta.json").exists()
    assert runner._clear_claims(tmp_path / "немає") == 0
    assert runner._clear_claims(None) == 0


def test_catchup_pass_never_carries_claim_or_shard() -> None:
    src = Path(runner.__file__).read_text(encoding="utf-8")
    body = src.split("def _catchup_pass(")[1].split("\ndef ")[0]
    assert '"--claim"' not in body and '"--shard"' not in body
    assert "_clear_claims(" in body


def test_claim_is_not_passed_to_a_runner_that_does_not_know_it(tmp_path: Path) -> None:
    """06.09.2026: бокс-раннер додав `--claim` до старої копії раннера — усі
    шарди впали на argparse до першої сторінки, захід пішов у догін з нулем."""
    old = tmp_path / "old.py"
    old.write_text('ap.add_argument("--shard")\n', encoding="utf-8")
    new = tmp_path / "new.py"
    new.write_text('ap.add_argument("--claim", action="store_true")\n', encoding="utf-8")
    assert not runner._runner_supports(old, "--claim")
    assert runner._runner_supports(new, "--claim")
    assert not runner._runner_supports(tmp_path / "немає.py", "--claim")
