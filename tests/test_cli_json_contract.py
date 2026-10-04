"""Замок на машинний вивід CLI (`--json`), який читають інші програми.

🔴 Ці тести тримають ФОРМУ, а не зміст. Споживач (`nyshporka`, обгортки
дослідницьких репозиторіїв) живе в іншому репозиторії й не впаде в нашому CI,
коли поле зникне, — він упаде в людини під час платної оренди. Тому перелік
полів звіряється тут дослівно: прибрати поле можна лише разом із `SCHEMA`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gpurunner.cli import app
from gpurunner.core import contract, manifest
from gpurunner.core.models import JobHandle, JobStatus

HANDLE_FIELDS = {"handle", "short", "backend", "job", "gpu", "remote_id", "state", "output_dir"}


def _last_json(stdout: str) -> dict[str, Any]:
    """Так, як це робить споживач: останній рядок, що починається з `{`."""
    line = next(ln for ln in reversed(stdout.splitlines()) if ln.startswith("{"))
    return json.loads(line)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def handle(data_dir: Path) -> JobHandle:
    h = JobHandle(backend="kaggle", remote_id="owner/kernel-1", job_name="net-probe",
                  params={}, gpu="T4")
    h.status = JobStatus.RUNNING
    manifest.add(h)
    return h


def test_dry_run_gives_one_json_object_with_the_validated_params(runner: CliRunner) -> None:
    res = runner.invoke(app, ["run", "paddleocr", "-b", "kaggle", "--dry-run", "--json",
                              "-p", 'urls=["https://example.org/a.pdf"]'])
    assert res.exit_code == 0
    # людське — у stderr: у stdout рівно один рядок, і це об'єкт
    assert [ln for ln in res.stdout.splitlines() if ln.strip()] == [res.stdout.strip()]
    payload = _last_json(res.stdout)
    assert payload["schema"] == contract.SCHEMA == 1
    assert payload["ok"] is True and payload["dry_run"] is True
    assert payload["command"] == "run" and payload["job"] == "paddleocr"
    assert payload["params"]["items"][0]["url"] == "https://example.org/a.pdf"


def test_a_refused_run_still_answers_in_json_and_keeps_the_exit_code(runner: CliRunner) -> None:
    res = runner.invoke(app, ["run", "htr_eval", "-b", "kaggle", "--dry-run", "--json",
                              "-p", "dataset=owner/slug"])
    assert res.exit_code == 2
    payload = _last_json(res.stdout)
    assert payload["ok"] is False and payload["exit_code"] == 2
    assert payload["error"].startswith("validation:")


def test_without_the_flag_the_human_line_is_untouched(runner: CliRunner) -> None:
    """Старі споживачі читають текст — він мусить лишитись, де був."""
    res = runner.invoke(app, ["run", "net-probe", "-b", "kaggle", "--dry-run"])
    assert res.exit_code == 0
    assert "Validated params" in res.stdout
    assert not any(ln.startswith('{"schema"') for ln in res.stdout.splitlines())


def test_status_of_one_handle(runner: CliRunner, handle: JobHandle) -> None:
    res = runner.invoke(app, ["status", handle.id[:8], "--no-ping", "--json"])
    assert res.exit_code == 0
    payload = _last_json(res.stdout)
    assert HANDLE_FIELDS | {"schema", "ok", "command", "refreshed", "terminal",
                            "message", "error"} <= set(payload)
    assert payload["handle"] == handle.id and payload["short"] == handle.id[:8]
    assert payload["state"] == "running" and payload["terminal"] is False
    # --no-ping: стан із маніфесту, і споживач мусить це бачити
    assert payload["refreshed"] is False


def test_status_list_and_unknown_handle(runner: CliRunner, handle: JobHandle) -> None:
    listed = _last_json(runner.invoke(app, ["status", "--no-ping", "--json"]).stdout)
    assert [h["handle"] for h in listed["handles"]] == [handle.id]
    assert set(listed["handles"][0]) == HANDLE_FIELDS

    res = runner.invoke(app, ["status", "nosuchid", "--json"])
    assert res.exit_code == 2
    assert _last_json(res.stdout) == {"schema": 1, "ok": False, "command": "status",
                                      "handle": "nosuchid", "error": "unknown handle",
                                      "exit_code": 2}


def test_states_are_the_job_status_values_and_nothing_else() -> None:
    assert set(contract.STATES) == {s.value for s in JobStatus}
    assert contract.state_of(JobStatus.COMPLETED) == "completed"
    assert contract.state_of("щось нове від бекенда") == "unknown"
    assert set(contract.TERMINAL) < set(contract.STATES)
