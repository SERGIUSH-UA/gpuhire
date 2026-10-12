"""Unit tests for BeamBackend.

No network and no beam-client import: a fake ``beam`` module (so the generated
entry module can be imported) and an in-memory volume store stand in for the SDK.
What stays real: the generated ``entry.py`` — it is compiled and executed for
every test, so a syntax error or a bad ``task_queue`` argument fails here — plus
the status state machine, input resolution and the fetch/logs paths.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pytest

from gpurunner.backends.beam import (
    DEFAULT_DATA_VOLUME,
    BeamBackend,
    SpendRefused,
    _backend_opts,
    _memory_to_gib,
    _render_entry_module,
    estimate_cost,
    worst_case_cost,
)
from gpurunner.core import Job, JobStatus
from gpurunner.core.backend import BackendError


@pytest.fixture(autouse=True)
def _isolated_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_CONFIG_DIR", str(tmp_path / "config"))


class _DummyJob(Job):
    name = "dummy"
    description = "test job"

    def requirements(self) -> list[str]:
        return ["numpy"]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"echo": str(params.get("echo", "hello")), "dataset": params.get("dataset", "")}

    def render_remote_code(
        self, params: dict[str, Any], *, shard_index: int = 0, total_shards: int = 1
    ) -> str:
        return f"print({params['echo']!r})"

    def colab_input_dirs(self, params: dict[str, Any]) -> dict[str, str]:
        ds = self.validate_params(params)["dataset"]
        return {ds: ds} if ds else {}


class _FakeTask:
    def __init__(self) -> None:
        self.id = "task-123"


class _FakeQueue:
    """Stands in for the object ``@task_queue`` returns."""

    # ClassVar, бо це саме клас-рівневі слоти: тест читає їх через тип, а не
    # через екземпляр (`type(self).last_kwargs = ...` нижче)
    last_kwargs: ClassVar[dict[str, Any]] = {}
    last_cwd: ClassVar[str] = ""

    def __init__(self, func: Any, kwargs: dict[str, Any]) -> None:
        self.func = func
        type(self).last_kwargs = kwargs

    def put(self) -> _FakeTask:
        # Beam resolves the handler relative to the *current* working directory,
        # so submit() must have chdir'd into the temp dir before calling put().
        type(self).last_cwd = os.getcwd()
        return _FakeTask()


def _fake_beam_module() -> types.ModuleType:
    module = types.ModuleType("beam")

    class Image:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.commands: list[str] = []

        def add_commands(self, commands: list[str]) -> Image:
            self.commands.extend(commands)
            return self

    class Volume:
        def __init__(self, name: str, mount_path: str) -> None:
            self.name = name
            self.mount_path = mount_path

    def task_queue(**kwargs: Any) -> Any:
        def decorate(func: Any) -> _FakeQueue:
            return _FakeQueue(func, kwargs)

        return decorate

    module.Image = Image  # type: ignore[attr-defined]
    module.Volume = Volume  # type: ignore[attr-defined]
    module.task_queue = task_queue  # type: ignore[attr-defined]
    return module


@pytest.fixture(autouse=True)
def _fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.auth import beam as beam_auth

    module = _fake_beam_module()
    monkeypatch.setitem(sys.modules, "beam", module)
    monkeypatch.setattr(
        beam_auth,
        "discover_credentials",
        lambda: types.SimpleNamespace(token="tok", source="test"),
    )
    monkeypatch.setattr(beam_auth, "import_sdk", lambda: module)
    _FakeQueue.last_kwargs = {}
    _FakeQueue.last_cwd = ""


class _TestBackend(BeamBackend):
    """BeamBackend with the volume I/O replaced by an in-memory store."""

    def __init__(self) -> None:
        super().__init__()
        self.files: dict[str, bytes] = {}   # "<volume>/<path>" -> bytes
        self.task_state: str | None = "running"
        self.stopped: list[str] = []

    def _upload_dir(self, local: Path, volume: str, slug: str) -> int:
        files = [p for p in sorted(local.rglob("*")) if p.is_file()]
        if not files:
            raise BackendError(f"input dir {local} has no files")
        for p in files:
            rel = p.relative_to(local).as_posix()
            self.files[f"{volume}/{slug}/{rel}"] = p.read_bytes()
        return len(files)

    def _list_volume(self, volume: str, prefix: str = "") -> list[str]:
        head = f"{volume}/{prefix}".rstrip("/") + "/"
        return [k[len(f"{volume}/"):] for k in self.files if k.startswith(head)]

    def _download_bytes(self, volume: str, rel_path: str) -> bytes | None:
        return self.files.get(f"{volume}/{rel_path}")

    def _task_status(self, handle: Any) -> str | None:
        return self.task_state

    def cancel(self, handle: Any) -> None:
        self.stopped.append(handle.remote_id)


def _submit(bk: _TestBackend, gpu: str = "RTX4090", **params: Any) -> Any:
    # Every submit goes through the spend guard, so the shared helper states a
    # ceiling explicitly; the guard itself is exercised in its own section below.
    params.setdefault("max_cost", 100)
    return bk.submit(_DummyJob(), params, gpu=gpu)


def _entry_kwargs() -> dict[str, Any]:
    return _FakeQueue.last_kwargs


# ---------------------------------------------------------------------------


def test_submit_records_task_id_and_output_volume() -> None:
    bk = _TestBackend()
    handle = _submit(bk)

    assert handle.remote_id == "task-123"
    assert handle.volume_name == f"gpurunner-out-{handle.id[:24]}"
    assert handle.status is JobStatus.QUEUED


def test_submit_maps_gpu_and_runtime_cap_onto_task_queue() -> None:
    bk = _TestBackend()
    _submit(bk, max_hours=3, cpu=4, memory="16Gi")

    kw = _entry_kwargs()
    assert kw["gpu"] == "RTX4090" and kw["gpu_count"] == 1
    assert kw["timeout"] == 3 * 3600
    assert kw["cpu"] == 4.0 and kw["memory"] == "16Gi"
    # retries default to 0: a crashing 12-hour train must not be re-run 3× on credit
    assert kw["retries"] == 0


def test_a100_maps_to_beams_own_sku_name() -> None:
    """Beam has no bare "A100" — it is A100-40 / A100-80."""
    bk = _TestBackend()
    # A100-40 has no published serverless rate either, so the guard needs one.
    bk.submit(_DummyJob(), {"price_per_hour": 1.0, "max_hours": 1, "max_cost": 5}, gpu="A100")
    assert _entry_kwargs()["gpu"] == "A100-40"


def test_cpu_only_run_requests_no_gpu() -> None:
    bk = _TestBackend()
    bk.submit(_DummyJob(), {"max_cost": 5}, gpu="none")
    kw = _entry_kwargs()
    assert kw["gpu"] == "" and kw["gpu_count"] == 0


def test_submit_rejects_unknown_gpu() -> None:
    with pytest.raises(ValueError, match="Unsupported gpu"):
        _TestBackend().submit(_DummyJob(), {"max_cost": 5}, gpu="RTX3090")


def test_entry_module_mounts_both_volumes() -> None:
    bk = _TestBackend()
    handle = _submit(bk)
    volumes = {v.name: v.mount_path for v in _entry_kwargs()["volumes"]}
    # Relative, not absolute: Beam silently ignores an absolute mount_path and the
    # volume then stays empty while the task reports success (smoke run 99058bb9).
    assert volumes == {
        f"gpurunner-out-{handle.id[:24]}": "./outputs",
        DEFAULT_DATA_VOLUME: "./inputs",
    }


def test_put_is_called_from_the_temp_dir_not_the_project_root() -> None:
    """Beam derives the handler from the module path *relative to CWD* and syncs
    that directory into the container — submitting from the repo root would ship
    the whole checkout (and produce a ``../..``-shaped module name)."""
    bk = _TestBackend()
    before = os.getcwd()
    _submit(bk)

    assert _FakeQueue.last_cwd != before
    assert "gpurunner-beam-" in _FakeQueue.last_cwd
    assert os.getcwd() == before          # and it is restored afterwards


def test_entry_module_carries_the_job_body_and_kaggle_emulation() -> None:
    source = _render_entry_module(
        "print('hello')",
        job_name="dummy",
        handle_id="0" * 32,
        gpu="T4",
        image_spec=_DummyJob().modal_image_spec(),
        inputs=["train-v42"],
        out_volume="gpurunner-out-x",
        data_volume=DEFAULT_DATA_VOLUME,
        opts=_backend_opts({}),
    )
    compile(source, "<entry>", "exec")                   # must be valid python
    assert json.dumps("print('hello')") in source        # job body travels inside
    assert 'kaggle_root = Path("/kaggle")' in source     # Kaggle-FS emulation
    assert 'working = kaggle_root / "working"' in source
    assert "train-v42" in source                         # declared inputs reach the wrapper
    assert "finalized" in source                         # terminal-status latch present
    assert 'src.replace("/kaggle/"' in source            # path rewrite for the fallback root


def test_extra_index_url_becomes_a_pip_command() -> None:
    """``Image`` has no extra_index_url parameter (Modal's does), so a job that
    needs one must get an explicit pip install instead of silently losing it."""

    class _PaddleJob(_DummyJob):
        def requirements(self) -> list[str]:
            return ["--extra-index-url", "https://example.org/cu126/", "paddlepaddle-gpu==3.2.0"]

    bk = _TestBackend()
    bk.submit(_PaddleJob(), {"max_hours": 1, "max_cost": 5}, gpu="RTX4090")
    image = _entry_kwargs()["image"]
    assert image.kwargs["python_packages"] == []
    assert any("--extra-index-url https://example.org/cu126/" in c for c in image.commands)
    assert any("paddlepaddle-gpu==3.2.0" in c for c in image.commands)


def test_inputs_are_uploaded_to_the_data_volume(tmp_path: Path) -> None:
    data = tmp_path / "train-v42"
    data.mkdir()
    (data / "train.tgz").write_bytes(b"payload")

    bk = _TestBackend()
    _submit(bk, dataset="train-v42", input_root=str(tmp_path))

    assert bk.files[f"{DEFAULT_DATA_VOLUME}/train-v42/train.tgz"] == b"payload"


def test_missing_input_location_fails_before_spending_anything() -> None:
    bk = _TestBackend()
    with pytest.raises(BackendError, match="input_root"):
        _submit(bk, dataset="train-v42")
    assert _FakeQueue.last_kwargs == {}  # nothing was submitted


def test_status_without_status_file_reports_task_state() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.task_state = "pending"
    report = bk.status(h)
    assert report.status is JobStatus.QUEUED
    assert "pending" in (report.message or "")


def test_status_reads_the_volume_status_file() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.files[f"{h.volume_name}/_status.json"] = json.dumps(
        {"status": "completed", "message": "5 files in out/"}
    ).encode("utf-8")
    report = bk.status(h)
    assert report.status is JobStatus.COMPLETED
    assert "5 files" in (report.message or "")


def test_status_flags_stale_heartbeat_as_unknown() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    old = (datetime.now(tz=UTC) - timedelta(hours=1)).isoformat()
    bk.files[f"{h.volume_name}/_status.json"] = json.dumps(
        {"status": "running", "heartbeat": old}
    ).encode("utf-8")
    assert bk.status(h).status is JobStatus.UNKNOWN


def test_status_running_status_file_but_dead_task_is_failed() -> None:
    """A container killed mid-run leaves a stale 'running' file behind; Beam's own
    task state is what disambiguates it."""
    bk = _TestBackend()
    h = _submit(bk)
    bk.files[f"{h.volume_name}/_status.json"] = json.dumps(
        {"status": "running", "heartbeat": datetime.now(tz=UTC).isoformat()}
    ).encode("utf-8")
    bk.task_state = "error"
    assert bk.status(h).status is JobStatus.FAILED


def test_status_task_finished_without_writing_status_is_failed() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.task_state = "complete"
    report = bk.status(h)
    assert report.status is JobStatus.FAILED
    assert "without writing" in (report.error or "")


def test_fetch_outputs_pulls_out_tree_and_sidecars(tmp_path: Path) -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.files[f"{h.volume_name}/out/result.txt"] = b"hello"
    bk.files[f"{h.volume_name}/out/runs/best.pt"] = b"\x00\x01"
    bk.files[f"{h.volume_name}/_runner.log"] = b"line\n"

    written = bk.fetch_outputs(h, tmp_path / "out")
    names = {p.relative_to(tmp_path / "out").as_posix() for p in written}
    assert names == {"result.txt", "runs/best.pt", "_runner.log"}
    assert (tmp_path / "out" / "runs" / "best.pt").read_bytes() == b"\x00\x01"


def test_logs_read_the_volume_log() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.files[f"{h.volume_name}/_runner.log"] = b"a\nb\n"
    assert bk.logs(h) == ["a", "b"]


def test_logs_say_so_when_there_is_nothing_yet() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    assert "no logs yet" in bk.logs(h)[0]


# ---- cost estimate --------------------------------------------------------


def test_estimate_cost_sums_gpu_cpu_and_ram() -> None:
    """_DummyJob inherits the ABC's 1-hour estimate, so the numbers are checkable:
    RTX4090's *measured* $2.2395/h + 2 cores × $0.0000125/s + 8 GiB × $0.0000021/s.

    The GPU term is the billed one, not the $0.69 the price page advertises —
    see `_GPU_MEASURED_HOURLY`."""
    est = estimate_cost(_DummyJob(), {}, "RTX4090")
    assert est is not None
    expected = 2.2395 + 2 * 0.0000125 * 3600 + 8 * 0.0000021 * 3600
    assert est["hours"] == pytest.approx(1.0)
    assert est["total"] == pytest.approx(expected, rel=1e-6)
    assert est["per_epoch"] == pytest.approx(expected, rel=1e-6)


def test_rate_reproduces_the_actual_beam_invoice() -> None:
    """The one configuration we have a real invoice for must come out exact.

    Beam's usage page for taskqueue/entry:execute (RTX4090, 8 cores, 32 GB,
    21m32s, 2026-08-09) reported $2.8414/h and charged $1.020 of credit. The
    published price list yields $1.29/h for the same container — 2.2× less —
    so this test is what keeps the estimate anchored to the invoice rather
    than to the marketing page.
    """
    from gpurunner.backends.beam import hourly_rate

    rate = hourly_rate("RTX4090", {"cpu": 8, "memory": "32Gi"})
    assert rate == pytest.approx(2.8414, abs=0.001)
    billed = rate * (21 * 60 + 32) / 3600  # 21m32s — the invoiced duration
    assert billed == pytest.approx(1.020, abs=0.005)


def test_unmeasured_cards_are_marked_up_rather_than_taken_at_list_price() -> None:
    """Only RTX4090 has an invoice. For everything else the list price is known
    to be optimistic, and under-quoting the ceiling is the failure that costs
    money — so the guard books the marked-up figure."""
    from gpurunner.backends.beam import _GPU_HOURLY, hourly_rate

    listed = _GPU_HOURLY["H100"] + (2 * 0.0000125 + 8 * 0.0000021) * 3600
    rate = hourly_rate("H100", {})
    assert rate is not None and rate > listed


def test_estimate_cost_is_silent_for_cards_beam_does_not_price() -> None:
    """T4/L4/A10G/A100-40 exist in the SDK's GpuType but not in the serverless
    price list — better no estimate than an invented one."""
    assert estimate_cost(_DummyJob(), {}, "T4") is None
    assert estimate_cost(_DummyJob(), {}, "A100") is None


def test_estimate_cost_divides_by_epochs() -> None:
    est = estimate_cost(_DummyJob(), {"epochs": 4}, "H100")
    assert est is not None
    assert est["per_epoch"] == pytest.approx(est["total"] / 4)


@pytest.mark.parametrize(
    ("value", "gib"),
    [("16Gi", 16.0), ("512Mi", 0.5), (16384, 16.0), ("8", 8 / 1024), ("nonsense", 0.0)],
)
def test_memory_parsing(value: Any, gib: float) -> None:
    assert _memory_to_gib(value) == pytest.approx(gib)


# ---- spend guard ----------------------------------------------------------
#
# Beam is the one backend where a card is attached and the API tells us nothing:
# no balance, no usage. These tests are the guard rails around real money.


def test_worst_case_is_the_timeout_not_the_estimate() -> None:
    """A job that hangs bills until the timeout, so that — not the runtime
    estimate — is what the ceiling must be computed from."""
    worst = worst_case_cost("RTX4090", {"max_hours": 12})
    rate = 2.2395 + (2 * 0.0000125 + 8 * 0.0000021) * 3600
    assert worst == pytest.approx(rate * 12)


def test_expensive_run_is_refused_before_anything_is_uploaded() -> None:
    bk = _TestBackend()
    with pytest.raises(SpendRefused, match=r"РЕАЛЬНІ ГРОШІ"):
        bk.submit(_DummyJob(), {"max_hours": 12}, gpu="H100")
    assert _FakeQueue.last_kwargs == {}      # never reached the SDK
    assert bk.files == {}                    # nothing uploaded


def test_allow_cost_unlocks_exactly_that_much() -> None:
    bk = _TestBackend()
    worst = worst_case_cost("RTX4090", {"max_hours": 2})
    with pytest.raises(SpendRefused):
        bk.submit(_DummyJob(), {"max_hours": 2, "max_cost": worst - 0.01}, gpu="RTX4090")
    handle = bk.submit(_DummyJob(), {"max_hours": 2, "max_cost": worst + 0.01}, gpu="RTX4090")
    assert handle.remote_id == "task-123"


def test_unpriced_gpu_is_refused_rather_than_assumed_cheap() -> None:
    """T4 exists in Beam's GpuType but has no published serverless rate — with no
    rate there is no ceiling, so the submit must not happen."""
    bk = _TestBackend()
    with pytest.raises(SpendRefused, match="no serverless price"):
        bk.submit(_DummyJob(), {}, gpu="T4")


def test_explicit_rate_makes_an_unpriced_gpu_usable() -> None:
    bk = _TestBackend()
    handle = bk.submit(
        _DummyJob(), {"price_per_hour": 0.1, "max_hours": 1, "max_cost": 0.5}, gpu="T4"
    )
    assert handle.remote_id == "task-123"


def test_monthly_cap_blocks_accumulation(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.core import budget

    run = worst_case_cost("RTX4090", {"max_hours": 1})
    assert run is not None
    # Стеля між одним і двома прогонами — рахуємо від ставки, а не від
    # константи: ставка калібрується за рахунками і буде мінятись.
    monkeypatch.setenv("GPURUNNER_BEAM_MONTHLY_BUDGET", f"{run * 1.5:.4f}")
    bk = _TestBackend()
    bk.submit(_DummyJob(), {"max_hours": 1, "max_cost": 10}, gpu="RTX4090")
    with pytest.raises(SpendRefused, match="місячна стеля"):
        bk.submit(_DummyJob(), {"max_hours": 1, "max_cost": 10}, gpu="RTX4090")
    assert budget.month_committed("beam") == pytest.approx(run, rel=1e-3)


def test_failed_submit_releases_its_booking(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reservation that never became a running task must not eat the budget."""
    from gpurunner.core import budget

    class _Boom(_TestBackend):
        def _upload_dir(self, local: Path, volume: str, slug: str) -> int:
            raise BackendError("upload exploded")

    bk = _Boom()
    data = Path(tempfile.mkdtemp()) / "train-v42"
    data.mkdir(parents=True)
    (data / "f.bin").write_bytes(b"x")
    with pytest.raises(BackendError, match="upload exploded"):
        bk.submit(_DummyJob(), {"dataset": "train-v42", "inputs": {"train-v42": str(data)},
                                "max_hours": 1, "max_cost": 5}, gpu="RTX4090")
    assert budget.month_committed("beam") == 0.0


def test_finished_run_is_settled_down_to_real_wall_time() -> None:
    """While a run is alive it is booked at the timeout; once it reports how long it
    actually took, the month's budget gets the difference back."""
    from gpurunner.core import budget

    bk = _TestBackend()
    h = bk.submit(_DummyJob(), {"max_hours": 2, "max_cost": 5}, gpu="RTX4090")
    booked = budget.month_committed("beam")

    bk.files[f"{h.volume_name}/_status.json"] = json.dumps(
        {"status": "completed", "message": "done", "elapsed_s": 180}
    ).encode("utf-8")
    bk.task_state = "complete"
    assert bk.status(h).status is JobStatus.COMPLETED

    settled = budget.month_committed("beam")
    assert settled < booked
    rate = worst_case_cost("RTX4090", {"max_hours": 1})
    assert settled == pytest.approx(rate * 180 / 3600, rel=1e-3)


def test_terminal_run_without_elapsed_keeps_its_worst_case() -> None:
    """Killed before the runner wrote anything → we do not know what it cost, so the
    pessimistic booking stands. Over-counting is the safe direction."""
    from gpurunner.core import budget

    bk = _TestBackend()
    h = bk.submit(_DummyJob(), {"max_hours": 2, "max_cost": 5}, gpu="RTX4090")
    booked = budget.month_committed("beam")
    bk.task_state = "error"
    bk.status(h)
    assert budget.month_committed("beam") == pytest.approx(booked)
