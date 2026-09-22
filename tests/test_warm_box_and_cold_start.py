"""Холодний старт і теплий бокс (06.09.2026).

Замір std160 05.09.2026: оренда → перша сторінка 5 хв, робота над 160
сторінками — 2.5 хв. На другій справі черги `startup_sec` був 232 с при
пропущених pip і ассетах — бо цикл `resume_urls` робив по одному curl на
кожне з 392 посилань (`підхоплено 2/392`, захід live2 04.09.2026). А щоб
довісити 160 сторінок, доводилось платити другий холодний старт: `done`
публікувався одразу після останньої справи.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import gpurunner._embedded.htr_case_runner as runner
from gpurunner.supervise.decide import Cfg, Obs, decide

CFG = Cfg(budget_usd=3.0, max_hours=8.0)


@pytest.fixture
def append_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "_append.jsonl"
    monkeypatch.setattr(runner, "APPEND_FILE", str(path))
    monkeypatch.setattr(runner, "IDLE_POLL_SEC", 0)
    return path


def _fake_run(calls: list):
    def _run(merged):
        calls.append(merged.get("case"))
        return {"case": merged.get("case"), "complete": True, "n_pages_txt": 1}
    return _run


# ---- теплий бокс ------------------------------------------------------------


def test_an_idle_box_picks_up_a_case_appended_after_the_queue_drained(
    append_file: Path, monkeypatch
) -> None:
    """Черга вичерпана, бокс теплий — довісок, що прилетів у тиші, рахується
    на тому самому боксі, і лише потім публікується `done`."""
    calls: list = []
    phases: list = []
    monkeypatch.setattr(runner, "_run_case", _fake_run(calls))
    monkeypatch.setattr(runner, "_set_phase",
                        lambda phase, **kw: phases.append(phase))
    polls = {"n": 0}

    def _appended(seen):
        polls["n"] += 1
        if polls["n"] == 2:
            seen.add("б")
            return [{"case": "б"}]
        return []
    monkeypatch.setattr(runner, "_read_appended", _appended)

    out = runner._main_inner({"cases": [{"case": "а"}], "keep_warm_sec": 30})

    assert calls == ["а", "б"]
    assert out["queue"] == 2
    assert "idle" in phases and phases[-1] == "done"
    assert phases.index("idle") < phases.index("done")


def test_idle_box_shuts_down_after_silence(append_file: Path, monkeypatch) -> None:
    """Тиша `keep_warm_sec` → `done`; без ключа фази `idle` не буває взагалі."""
    calls: list = []
    phases: list = []
    monkeypatch.setattr(runner, "_run_case", _fake_run(calls))
    monkeypatch.setattr(runner, "_set_phase", lambda phase, **kw: phases.append(phase))
    clock = {"t": 1000.0}
    monkeypatch.setattr(runner.time, "time", lambda: clock.__setitem__("t", clock["t"] + 7) or clock["t"])
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)

    out = runner._main_inner({"cases": [{"case": "а"}], "keep_warm_sec": 20})
    assert calls == ["а"] and out["queue"] == 1
    assert phases.count("idle") >= 1 and phases[-1] == "done"

    phases.clear()
    runner._main_inner({"cases": [{"case": "а"}]})
    assert "idle" not in phases, "без keep_warm_sec поведінка як була"


def test_idle_phase_is_a_wait_not_a_dead_runner() -> None:
    """Фаза не `running`, прогрес старіє — правило 5 оголосило б `rerent`."""
    action, why = decide(Obs(progress={"phase": "idle", "idle_left_sec": 300},
                             progress_age_sec=2000, dph=0.15), CFG)
    assert action == "wait"
    assert "append" in why


def test_idle_still_obeys_budget_and_deadline() -> None:
    action, _ = decide(Obs(progress={"phase": "idle"}, spent_usd=5.0, dph=0.15,
                           elapsed_h=0.1), Cfg(budget_usd=1.0, max_hours=8.0))
    assert action != "wait"


# ---- холодний старт ---------------------------------------------------------


def test_resume_probe_stops_after_three_misses_in_a_row(monkeypatch) -> None:
    """`підхоплено 2/392`: 390 curl у 404 по ~0.5 с = 3 хв на кожну справу."""
    src = Path(runner.__file__).read_text(encoding="utf-8")
    assert "misses >= RESUME_MISSES_TO_STOP" in src
    assert runner.RESUME_MISSES_TO_STOP == 3


def test_runner_refuses_scripts_with_the_wrong_hash(tmp_path: Path) -> None:
    """05.09.2026 на боксі лежала стара копія `htr_case_run.py` без ліміту
    потоків; перевірялись лише імена (`NEEDED_SCRIPTS`)."""
    import hashlib

    code = tmp_path / "scripts"
    code.mkdir()
    (code / "htr_case_run.py").write_text("print('new')\n", encoding="utf-8")
    good = hashlib.sha256((code / "htr_case_run.py").read_bytes()).hexdigest()

    got = runner._verify_scripts(code, {"htr_case_run.py": good})
    assert got["htr_case_run.py"] == good

    with pytest.raises(RuntimeError, match="застарілі скрипти"):
        runner._verify_scripts(code, {"htr_case_run.py": "0" * 64})
    with pytest.raises(RuntimeError, match="немає"):
        runner._verify_scripts(code, {"seg_resize.py": good})
    assert runner._verify_scripts(code, {}) == {"htr_case_run.py": good}


def test_a_stale_runner_inside_assets_is_refused_before_renting(tmp_path: Path) -> None:
    import hashlib
    import tarfile

    from gpurunner.htr.plan_build import scripts_sha256_in

    root = tmp_path / "a"
    (root / "scripts").mkdir(parents=True)
    # байти, не текст: на Windows `write_text` підставив би CRLF і зсунув sha
    (root / "scripts" / "htr_case_run.py").write_bytes(b"old\n")
    (root / "scripts" / "seg_resize.py").write_bytes(b"patch\n")
    tgz = tmp_path / "assets.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(root / "scripts", arcname="scripts")
    sha = scripts_sha256_in(tgz)
    assert set(sha) == {"htr_case_run.py", "seg_resize.py"}
    assert sha["htr_case_run.py"] == hashlib.sha256(b"old\n").hexdigest()


def test_plan_carries_keep_warm_and_script_hashes(tmp_path: Path) -> None:
    from gpurunner.supervise.plan import load_plan

    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "assets_url": "https://r2/a.tgz", "budget_usd": 1.0, "max_hours": 1.0,
        "keep_warm_min": 8, "scripts_sha256": {"htr_case_run.py": "ab" * 32},
        "cases": [{"case": "x", "pages_url": "https://r2/x.tar", "n_pages": 10,
                   "out_dir": "E:/out/x"}],
    }), encoding="utf-8")
    plan = load_plan(path)
    assert plan.keep_warm_min == 8.0
    assert plan.scripts_sha256 == {"htr_case_run.py": "ab" * 32}
    assert not plan.warnings
