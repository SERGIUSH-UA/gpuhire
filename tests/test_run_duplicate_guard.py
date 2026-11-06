"""`run` не подає вдруге той самий job із тими самими даними, поки живий перший (26.10.2026).

Цикл очікування ринку і ручний запуск подали той самий `kraken_train` разом —
два A100 натренували одне й те саме. Перевірка в самому `run`, а не в уважності
того, хто його кличе.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gpurunner import cli
from gpurunner.core import manifest, run_guard
from gpurunner.core.models import JobHandle, JobStatus


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_OWNER", "test-owner")


class FakeJob:
    name = "fake_train"

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        # Як справжні job'и: ручки бекенда (ціна, вхідна тека) сюди не потрапляють.
        return {k: v for k, v in params.items()
                if k not in {"input_root", "max_price", "datacenter_only"}}


submitted: list[dict[str, Any]] = []


class FakeBackend:
    name = "fake"
    default_gpu = "A100"

    def check_auth(self) -> None:
        pass

    def submit(self, job: Any, params: dict[str, Any], *, gpu: str) -> JobHandle:
        submitted.append(dict(params))
        return JobHandle(backend=self.name, remote_id=f"box-{len(submitted)}",
                         job_name=job.name, params=job.validate_params(params), gpu=gpu,
                         status=JobStatus.RUNNING, owner=manifest.current_owner())


@pytest.fixture(autouse=True)
def _fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted.clear()
    monkeypatch.setattr(cli, "get_job", lambda name: FakeJob)
    monkeypatch.setattr(cli, "get_backend", lambda name: FakeBackend)


def _run(*extra: str) -> Any:
    return CliRunner().invoke(cli.app, ["run", "fake_train", "-b", "fake",
                                        "-p", "epochs=12", "-p", "input_root=T:/corpus_a", *extra])


def test_second_identical_run_is_refused_before_renting() -> None:
    assert _run().exit_code == 0
    second = _run("-p", "max_price=1.2", "-p", "datacenter_only=true")
    assert second.exit_code == 2, second.output
    assert len(submitted) == 1, "дубль на дорожчій машині — так само дубль"


def test_other_corpus_or_params_is_not_a_duplicate() -> None:
    assert _run().exit_code == 0
    other_corpus = CliRunner().invoke(cli.app, ["run", "fake_train", "-b", "fake", "-p", "epochs=12",
                                                "-p", "input_root=T:/corpus_b"])
    other_params = CliRunner().invoke(cli.app, ["run", "fake_train", "-b", "fake", "-p", "epochs=6",
                                                "-p", "input_root=T:/corpus_a"])
    assert other_corpus.exit_code == 0 and other_params.exit_code == 0
    assert len(submitted) == 3


def test_allow_duplicate_submits_anyway() -> None:
    assert _run().exit_code == 0
    assert _run("--allow-duplicate").exit_code == 0
    assert len(submitted) == 2


def test_finished_run_does_not_block() -> None:
    assert _run().exit_code == 0
    (h,) = manifest.load()
    h.status = JobStatus.COMPLETED
    manifest.update(h)
    assert _run().exit_code == 0


def test_claim_blocks_a_run_while_the_first_is_still_submitting() -> None:
    """`submit` триває хвилини, а хендл з ключем з'являється лише після нього."""
    key = run_guard.run_key("fake_train", "fake", {"epochs": 12}, "T:/corpus_a")
    with run_guard.claim(key, owner="перший"):
        res = _run()
    assert res.exit_code == 2, res.output
    assert submitted == []
    assert _run().exit_code == 0, "заявка знімається після submit"


def test_failed_submit_releases_the_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.core import BackendError

    def boom(self: Any, job: Any, params: dict[str, Any], *, gpu: str) -> JobHandle:
        raise BackendError("ринок порожній")

    real_submit = FakeBackend.submit
    monkeypatch.setattr(FakeBackend, "submit", boom)
    assert _run().exit_code == 3
    # Не `monkeypatch.undo()`: той зняв би й ізоляцію теки даних.
    monkeypatch.setattr(FakeBackend, "submit", real_submit)
    assert _run().exit_code == 0
