"""Unit tests for LightningBackend.

No network and no lightning-sdk import: a fake Teamspace (in-memory "drive") and
a fake Job class stand in for the SDK. What stays real: the drive layout, command
rendering (incl. the base64-shipped wrapper), the status state machine, input
resolution and the fetch/logs paths.
"""

from __future__ import annotations

import base64
import json
import shlex
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pytest

from gpurunner.backends.lightning import DRIVE_ROOT, LightningBackend
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


class _FakeTeamspace:
    """In-memory teamspace drive: {remote path -> bytes}."""

    def __init__(self, name: str = "ts") -> None:
        self.name = name
        self.id = "ts-id"
        self.default_cloud_account = "cloud-1"
        self.owner = types.SimpleNamespace(name="tester")
        self.files: dict[str, bytes] = {}
        self.uploads: list[tuple[str, str]] = []
        self._teamspace_api = _FakeTeamspaceApi(self)

    def upload_folder(self, folder_path: str, remote_path: str, progress_bar: bool = True) -> None:
        # The real SDK silently transfers NOTHING here (verified live). Modelling
        # that keeps the backend honest about using upload_file instead.
        self.uploads.append((folder_path, remote_path))

    def upload_file(self, file_path: str, remote_path: str, progress_bar: bool = True,
                    cloud_account: str | None = None) -> None:
        # The real helper runs remote_path through os.path.normpath — on Windows
        # that mangles "a/b.txt" into "a\b.txt" and the blob is lost. Model it so
        # the backend can't regress to using this entry point.
        import os as _os

        self.files[_os.path.normpath(remote_path)] = Path(file_path).read_bytes()

    def download_folder(self, remote_path: str, target_path: str | None = None) -> None:
        # Real SDK writes zero-byte files here; nothing should rely on it.
        raise AssertionError("download_folder is broken upstream — use list_files + download_file")

    def download_file(self, remote_path: str, file_path: str | None = None) -> None:
        if remote_path not in self.files:
            raise FileNotFoundError(remote_path)
        target = Path(file_path or remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[remote_path])


class _FakeTeamspaceApi:
    """Mirrors the private API gpurunner uses to enumerate a run folder."""

    def __init__(self, ts: _FakeTeamspace) -> None:
        self._ts = ts

    def upload_file(self, teamspace_id: str, cloud_account: str, file_path: str,
                    remote_path: str, progress_bar: bool = False) -> None:
        self._ts.files[remote_path] = Path(file_path).read_bytes()   # verbatim key

    def list_files(self, teamspace_id: str, path: str = "") -> list[dict[str, Any]]:
        prefix = path.rstrip("/") + "/"
        return [
            {"path": k[len(prefix):], "type": "blob", "size": len(v)}
            for k, v in self._ts.files.items()
            if k.startswith(prefix)
        ]


class _FakeSDKJob:
    """Stands in for lightning_sdk.Job — records the last run() call."""

    last_kwargs: ClassVar[dict[str, Any]] = {}
    state: str = "running"
    cost: float | None = 0.37
    stopped: bool = False

    def __init__(self, name: str, teamspace: Any = None, **kw: Any) -> None:
        self.name = name

    @classmethod
    def run(cls, **kwargs: Any) -> _FakeSDKJob:
        cls.last_kwargs = kwargs
        return cls(name=kwargs["name"])

    @property
    def status(self) -> Any:
        import types

        return types.SimpleNamespace(name=self.state)

    @property
    def total_cost(self) -> float | None:
        return self.cost

    @property
    def logs(self) -> str:
        return "sdk log line"

    def stop(self) -> None:
        type(self).stopped = True


class _FakeMachine:
    CPU = "cpu"
    T4 = "t4"
    T4_X_2 = "t4x2"
    L4 = "l4"
    L40S = "l40s"
    A100 = "a100"
    H100 = "h100"


class _FakeSDK:
    Job = _FakeSDKJob
    Machine = _FakeMachine


class _TestBackend(LightningBackend):
    def __init__(self) -> None:
        super().__init__()
        self.ts = _FakeTeamspace()

    def _teamspace(self, name: str | None = None) -> Any:
        return self.ts


@pytest.fixture(autouse=True)
def _fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.auth import lightning as lit_auth

    monkeypatch.setattr(lit_auth, "import_sdk", lambda: _FakeSDK)
    _FakeSDKJob.last_kwargs = {}
    _FakeSDKJob.state = "running"
    _FakeSDKJob.stopped = False


def _submit(bk: _TestBackend, **params: Any) -> Any:
    params.setdefault("studio", "my-studio")
    return bk.submit(_DummyJob(), params, gpu="T4")


# ---------------------------------------------------------------------------


def test_submit_passes_machine_studio_and_runtime_cap() -> None:
    bk = _TestBackend()
    handle = _submit(bk, max_hours=3)

    kw = _FakeSDKJob.last_kwargs
    assert kw["machine"] == _FakeMachine.T4
    assert kw["studio"] == "my-studio"
    assert kw["max_runtime"] == 3 * 3600
    assert kw["interruptible"] is False
    assert handle.remote_id == kw["name"] and handle.remote_id.startswith("gpurunner-dummy-")
    assert handle.volume_name == f"{DRIVE_ROOT}/runs/{handle.id}"
    assert handle.status is JobStatus.QUEUED


def test_submit_requires_a_studio() -> None:
    bk = _TestBackend()
    with pytest.raises(BackendError, match="studio"):
        bk.submit(_DummyJob(), {}, gpu="T4")


def test_submit_rejects_unknown_gpu() -> None:
    with pytest.raises(ValueError, match="Unsupported gpu"):
        _TestBackend().submit(_DummyJob(), {"studio": "s"}, gpu="RTX4090")


def test_command_ships_a_valid_python_wrapper() -> None:
    bk = _TestBackend()
    handle = _submit(bk)
    command = _FakeSDKJob.last_kwargs["command"]

    blob = shlex.split(command.split("echo ", 1)[1].split(" | base64", 1)[0] + " ")[0]
    wrapper = base64.b64decode(blob).decode("utf-8")
    compile(wrapper, "<wrapper>", "exec")  # must be valid python, not just a string

    assert json.dumps("print('hello')") in wrapper       # the job body travels inside
    assert f"{DRIVE_ROOT}/runs/{handle.id}" in wrapper
    assert "/kaggle/working" in wrapper and "/kaggle/input" in wrapper
    assert "_FINALIZED" in wrapper                        # terminal-status latch present


def test_inputs_are_uploaded_to_the_drive_and_declared_to_the_wrapper(tmp_path: Path) -> None:
    data = tmp_path / "train-v42"
    data.mkdir()
    (data / "train.tgz").write_bytes(b"payload")

    bk = _TestBackend()
    _submit(bk, dataset="train-v42", input_root=str(tmp_path))

    assert bk.ts.files[f"{DRIVE_ROOT}/data/train-v42/train.tgz"] == b"payload"
    wrapper = base64.b64decode(
        shlex.split(
            _FakeSDKJob.last_kwargs["command"].split("echo ", 1)[1].split(" | base64", 1)[0] + " "
        )[0]
    ).decode("utf-8")
    assert "train-v42" in wrapper


def test_missing_input_location_fails_before_spending_anything() -> None:
    bk = _TestBackend()
    with pytest.raises(BackendError, match="input_root"):
        _submit(bk, dataset="train-v42")
    assert _FakeSDKJob.last_kwargs == {}  # nothing was submitted


def test_empty_download_counts_as_missing() -> None:
    """The real SDK writes an EMPTY local file for a missing remote one instead of
    raising — so bytes, not the absence of an exception, decide 'is it there'.
    Regression: a 0-byte _status.json was read as a valid (empty) status."""

    class _SilentTeamspace(_FakeTeamspace):
        def download_file(self, remote_path: str, file_path: str | None = None) -> None:
            target = Path(file_path or remote_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.files.get(remote_path, b""))  # empty, no raise

    bk = _TestBackend()
    bk.ts = _SilentTeamspace()
    h = _submit(bk)
    _FakeSDKJob.state = "pending"

    assert bk.status(h).status is JobStatus.QUEUED       # not COMPLETED/garbage
    assert bk.logs(h) == ["sdk log line"]                # fell through to the SDK


def test_wrapper_moves_data_over_the_api_not_the_filesystem(tmp_path: Path) -> None:
    """Verified on a live account: every /teamspace mount is read-only inside a job
    and the studio home is ephemeral. So the wrapper must use the Teamspace API for
    inputs AND for pushing status/log/outputs back."""
    data = tmp_path / "train-v42"
    data.mkdir()
    (data / "a.bin").write_bytes(b"x")
    bk = _TestBackend()
    _submit(bk, dataset="train-v42", inputs={"train-v42": str(data)})
    wrapper = base64.b64decode(
        shlex.split(
            _FakeSDKJob.last_kwargs["command"].split("echo ", 1)[1].split(" | base64", 1)[0] + " "
        )[0]
    ).decode("utf-8")

    assert "from lightning_sdk import Teamspace" in wrapper
    # folder-level SDK helpers are broken upstream: everything is file-by-file
    assert "TS.download_file" in wrapper         # inputs come down through the API
    assert "TS.upload_file" in wrapper           # status/log/outputs go up
    assert "TS.download_folder" not in wrapper
    assert "TS.upload_folder" not in wrapper
    assert "/teamspace/studios/this_studio" not in wrapper  # never trust the mount


def test_wrapper_falls_back_when_kaggle_root_is_not_writable() -> None:
    """Lightning jobs run unprivileged, so /kaggle can't be created. The wrapper
    must relocate and rewrite the job's hardcoded paths instead of dying."""
    bk = _TestBackend()
    _submit(bk)
    wrapper = base64.b64decode(
        shlex.split(
            _FakeSDKJob.last_kwargs["command"].split("echo ", 1)[1].split(" | base64", 1)[0] + " "
        )[0]
    ).decode("utf-8")
    assert "PermissionError" in wrapper
    assert "gpurunner_kaggle" in wrapper
    assert 'RUNNER_SRC.replace("/kaggle/"' in wrapper


def test_status_without_status_file_reports_sdk_state() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    _FakeSDKJob.state = "pending"
    report = bk.status(h)
    assert report.status is JobStatus.QUEUED
    assert "pending" in (report.message or "")


def test_status_reads_the_drive_status_file_and_shows_cost() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.ts.files[f"{h.volume_name}/_status.json"] = json.dumps(
        {"status": "completed", "message": "5 files in out/"}
    ).encode("utf-8")
    report = bk.status(h)
    assert report.status is JobStatus.COMPLETED
    assert "5 files" in (report.message or "")
    assert "$0.37" in (report.message or "")


def test_status_flags_stale_heartbeat_as_unknown() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    old = (datetime.now(tz=UTC) - timedelta(hours=1)).isoformat()
    bk.ts.files[f"{h.volume_name}/_status.json"] = json.dumps(
        {"status": "running", "heartbeat": old}
    ).encode("utf-8")
    assert bk.status(h).status is JobStatus.UNKNOWN


def test_status_running_with_fresh_heartbeat() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.ts.files[f"{h.volume_name}/_status.json"] = json.dumps(
        {"status": "running", "heartbeat": datetime.now(tz=UTC).isoformat()}
    ).encode("utf-8")
    assert bk.status(h).status is JobStatus.RUNNING


def test_status_job_finished_without_writing_status_is_failed() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    _FakeSDKJob.state = "completed"
    report = bk.status(h)
    assert report.status is JobStatus.FAILED
    assert "without writing" in (report.error or "")


def test_fetch_outputs_pulls_out_tree_and_sidecars(tmp_path: Path) -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.ts.files[f"{h.volume_name}/out/result.txt"] = b"hello"
    bk.ts.files[f"{h.volume_name}/out/runs/best.pt"] = b"\x00\x01"
    bk.ts.files[f"{h.volume_name}/_runner.log"] = b"line\n"

    written = bk.fetch_outputs(h, tmp_path / "out")
    names = {p.relative_to(tmp_path / "out").as_posix() for p in written}
    assert names == {"result.txt", "runs/best.pt", "_runner.log"}
    assert (tmp_path / "out" / "runs" / "best.pt").read_bytes() == b"\x00\x01"


def test_logs_prefer_the_drive_log_over_the_sdk() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.ts.files[f"{h.volume_name}/_runner.log"] = b"a\nb\n"
    assert bk.logs(h) == ["a", "b"]


def test_logs_fall_back_to_sdk_when_the_drive_log_is_absent() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    assert bk.logs(h) == ["sdk log line"]


def test_cancel_stops_the_job() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.cancel(h)
    assert _FakeSDKJob.stopped is True
