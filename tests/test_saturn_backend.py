"""Unit tests for SaturnBackend.

No network and no saturn-client import: a fake ``SaturnConnection`` and an
in-memory ``sfs://`` filesystem stand in for the SDK. What stays real: the recipe
that gets applied, the bootstrap command, the wrapper source (compiled here), the
instance-type resolution, the status state machine and the fetch/logs paths.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gpurunner.backends.saturn import SaturnBackend, estimate_cost, pick_instance_type
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


_SIZES = [
    {"name": "large", "cores": 8, "memory": "32Gi", "gpu": 0, "display_name": "Large - CPU"},
    {"name": "medium", "cores": 4, "memory": "16Gi", "gpu": 0, "display_name": "Medium - CPU"},
    {"name": "g4dnxlarge", "cores": 4, "memory": "16Gi", "gpu": 1, "display_name": "T4-XLarge"},
    {"name": "g4dn12xlarge", "cores": 48, "memory": "192Gi", "gpu": 4, "display_name": "T4-12XLarge"},
    {"name": "p3xlarge", "cores": 8, "memory": "61Gi", "gpu": 1, "display_name": "V100-XLarge"},
]


class _FakeFS:
    """In-memory sfs: {"org/user/…" -> bytes}. fsspec strips the protocol."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    @staticmethod
    def _key(path: str) -> str:
        return path.split("://", 1)[-1]

    def put_file(self, lpath: str, rpath: str, **kwargs: Any) -> None:
        self.files[self._key(rpath)] = Path(lpath).read_bytes()

    def get_file(self, rpath: str, lpath: str, **kwargs: Any) -> None:
        key = self._key(rpath)
        if key not in self.files:
            raise FileNotFoundError(rpath)
        target = Path(lpath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[key])

    def cat_file(self, path: str, **kwargs: Any) -> bytes:
        key = self._key(path)
        if key not in self.files:
            raise FileNotFoundError(path)
        return self.files[key]

    def find(self, path: str, **kwargs: Any) -> list[str]:
        prefix = self._key(path).rstrip("/") + "/"
        return sorted(k for k in self.files if k.startswith(prefix))


class _FakeConn:
    url = "https://app.example.saturnenterprise.io"

    def __init__(self) -> None:
        self.applied: list[dict[str, Any]] = []
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.resource_status = "pending"
        self.logs_text = "container log line"

    # -- identity
    @property
    def current_user(self) -> dict[str, Any]:
        return {"username": "tester", "id": "user-1"}

    @property
    def primary_org(self) -> dict[str, Any]:
        return {"name": "acme", "id": "org-1"}

    # -- catalogue
    def list_options(self, option_type: str, glob: str | None = None) -> list[dict[str, Any]]:
        assert option_type == "sizes"
        return list(_SIZES)

    # -- resources
    def apply(self, recipe: dict[str, Any]) -> dict[str, Any]:
        self.applied.append(recipe)
        return {**recipe, "state": {"id": "res-1", "status": self.resource_status}}

    def start(self, resource_type: str, resource_id: str, debug_mode: bool = False) -> None:
        self.started.append(resource_id)

    def stop(self, resource_type: str, resource_id: str) -> None:
        self.stopped.append(resource_id)

    def get_resource(self, resource_type: str, resource_name: str, **kwargs: Any) -> dict[str, Any]:
        if not self.applied:
            raise KeyError(resource_name)
        return {**self.applied[-1], "state": {"id": "res-1", "status": self.resource_status}}

    def get_logs(self, resource_type: str, resource_name: str, **kwargs: Any) -> str:
        return self.logs_text


class _TestBackend(SaturnBackend):
    def __init__(self) -> None:
        super().__init__()
        self.conn = _FakeConn()
        self.fs = _FakeFS()

    def _connection(self) -> Any:
        return self.conn

    def _filesystem(self) -> Any:
        return self.fs


def _submit(bk: _TestBackend, **params: Any) -> Any:
    params.setdefault("image", "saturncloud/saturn-python-pytorch:2026.01.01")
    return bk.submit(_DummyJob(), params, gpu="T4")


def _spec(bk: _TestBackend) -> dict[str, Any]:
    return bk.conn.applied[-1]["spec"]


# ---------------------------------------------------------------------------


def test_submit_applies_a_job_recipe_and_starts_it() -> None:
    bk = _TestBackend()
    handle = _submit(bk)

    recipe = bk.conn.applied[-1]
    assert recipe["type"] == "job"
    spec = recipe["spec"]
    assert spec["name"] == handle.remote_id
    assert handle.remote_id.startswith("gpurunner-dummy-")
    assert spec["owner"] == "tester"
    assert spec["image"] == "saturncloud/saturn-python-pytorch:2026.01.01"
    assert bk.conn.started == ["res-1"]
    assert handle.volume_name == f"sfs://acme/tester/gpurunner/runs/{handle.id}"
    assert handle.status is JobStatus.QUEUED


def test_submit_requires_an_image() -> None:
    bk = _TestBackend()
    with pytest.raises(BackendError, match="image"):
        bk.submit(_DummyJob(), {}, gpu="T4")
    assert bk.conn.applied == []


def test_submit_rejects_unknown_gpu() -> None:
    with pytest.raises(ValueError, match="Unsupported gpu"):
        _TestBackend().submit(_DummyJob(), {"image": "img"}, gpu="RTX4090")


def test_instance_type_is_resolved_from_the_accounts_own_catalogue() -> None:
    bk = _TestBackend()
    _submit(bk)
    # smallest T4 box, not the 4-GPU one
    assert _spec(bk)["instance_type"] == "g4dnxlarge"


def test_explicit_instance_type_wins() -> None:
    bk = _TestBackend()
    _submit(bk, instance_type="g4dn12xlarge")
    assert _spec(bk)["instance_type"] == "g4dn12xlarge"


def test_unavailable_gpu_lists_what_the_account_has() -> None:
    with pytest.raises(BackendError, match="g4dnxlarge"):
        pick_instance_type(_SIZES, "H100")


def test_cpu_only_picks_the_smallest_cpu_box() -> None:
    assert pick_instance_type(_SIZES, "") == "medium"


def test_command_downloads_the_wrapper_off_sfs_and_runs_it() -> None:
    bk = _TestBackend()
    handle = _submit(bk)
    command = _spec(bk)["command"]

    assert f"sfs://acme/tester/gpurunner/runs/{handle.id}/job.py" in command
    assert command.startswith("pip install -q saturnfs && ")
    assert command.endswith("python -u /tmp/gpurunner_job.py")


def test_wrapper_uploaded_to_sfs_is_valid_python_and_carries_the_job_body() -> None:
    bk = _TestBackend()
    handle = _submit(bk)

    key = f"acme/tester/gpurunner/runs/{handle.id}/job.py"
    wrapper = bk.fs.files[key].decode("utf-8")
    compile(wrapper, "<wrapper>", "exec")

    assert json.dumps("print('hello')") in wrapper
    assert 'KAGGLE_ROOT = Path("/kaggle")' in wrapper    # Kaggle-FS emulation
    assert 'WORKING = KAGGLE_ROOT / "working"' in wrapper
    assert "_FINALIZED" in wrapper                       # terminal-status latch present
    assert "gpurunner_kaggle" in wrapper                 # unprivileged-user fallback
    assert 'RUNNER_SRC.replace("/kaggle/"' in wrapper


def test_wrapper_fails_loudly_when_saturn_credentials_are_absent_in_the_job() -> None:
    """saturnfs is the only channel back — a job that cannot reach it must say so
    in its first seconds, not fail deep inside fsspec halfway through a train."""
    bk = _TestBackend()
    handle = _submit(bk)
    wrapper = bk.fs.files[f"acme/tester/gpurunner/runs/{handle.id}/job.py"].decode("utf-8")
    assert "SATURN_TOKEN" in wrapper and "SATURN_BASE_URL" in wrapper


def test_inputs_are_uploaded_to_sfs(tmp_path: Path) -> None:
    data = tmp_path / "train-v42"
    data.mkdir()
    (data / "train.tgz").write_bytes(b"payload")

    bk = _TestBackend()
    _submit(bk, dataset="train-v42", input_root=str(tmp_path))

    assert bk.fs.files["acme/tester/gpurunner/data/train-v42/train.tgz"] == b"payload"


def test_missing_input_location_fails_before_spending_anything() -> None:
    bk = _TestBackend()
    with pytest.raises(BackendError, match="input_root"):
        _submit(bk, dataset="train-v42")
    assert bk.conn.applied == []


def test_status_without_status_file_reports_resource_state() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    report = bk.status(h)
    assert report.status is JobStatus.QUEUED
    assert "pending" in (report.message or "")


def test_status_reads_the_sfs_status_file() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.fs.files[f"acme/tester/gpurunner/runs/{h.id}/_status.json"] = json.dumps(
        {"status": "completed", "message": "5 files in out/"}
    ).encode("utf-8")
    report = bk.status(h)
    assert report.status is JobStatus.COMPLETED
    assert "5 files" in (report.message or "")


def test_status_flags_stale_heartbeat_as_unknown() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    old = (datetime.now(tz=UTC) - timedelta(hours=1)).isoformat()
    bk.fs.files[f"acme/tester/gpurunner/runs/{h.id}/_status.json"] = json.dumps(
        {"status": "running", "heartbeat": old}
    ).encode("utf-8")
    assert bk.status(h).status is JobStatus.UNKNOWN


def test_status_errored_resource_without_status_file_is_failed() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.conn.resource_status = "error"
    report = bk.status(h)
    assert report.status is JobStatus.FAILED
    assert "errored" in (report.error or "")


def test_fetch_outputs_pulls_out_tree_and_skips_the_wrapper(tmp_path: Path) -> None:
    bk = _TestBackend()
    h = _submit(bk)
    run = f"acme/tester/gpurunner/runs/{h.id}"
    bk.fs.files[f"{run}/out/result.txt"] = b"hello"
    bk.fs.files[f"{run}/out/runs/best.pt"] = b"\x00\x01"
    bk.fs.files[f"{run}/_runner.log"] = b"line\n"

    written = bk.fetch_outputs(h, tmp_path / "out")
    names = {p.relative_to(tmp_path / "out").as_posix() for p in written}
    assert names == {"result.txt", "runs/best.pt", "_runner.log"}   # job.py is not an output
    assert (tmp_path / "out" / "runs" / "best.pt").read_bytes() == b"\x00\x01"


def test_logs_prefer_the_sfs_log_over_container_logs() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.fs.files[f"acme/tester/gpurunner/runs/{h.id}/_runner.log"] = b"a\nb\n"
    assert bk.logs(h) == ["a", "b"]


def test_logs_fall_back_to_container_logs_before_the_runner_starts() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    assert bk.logs(h) == ["container log line"]


def test_cancel_stops_the_job() -> None:
    bk = _TestBackend()
    h = _submit(bk)
    bk.cancel(h)
    assert bk.conn.stopped == ["res-1"]


# ---- catalogue matching + cost estimate -----------------------------------


def test_l4_request_does_not_match_an_l40s() -> None:
    """Substring matching would rent an L40S for an L4 request — the price and the
    free-tier burn differ by ~2×."""
    sizes = [
        *_SIZES,
        {"name": "shadeform-l40s", "cores": 12, "gpu": 1, "display_name": "Shadeform L40S"},
        {"name": "gpu-l4-1x", "cores": 8, "gpu": 1, "display_name": "L4-XLarge"},
    ]
    assert pick_instance_type(sizes, "L4") == "gpu-l4-1x"
    assert pick_instance_type(sizes, "L40S") == "shadeform-l40s"


def test_gpu_glued_to_a_count_still_matches() -> None:
    """Real community-catalogue names look like `nebius-1xh100`, so a strict left
    word boundary would miss them."""
    sizes = [{"name": "nebius/nebius-1xh100", "cores": 16, "gpu": 1,
              "display_name": "nebius-1xh100"}]
    assert pick_instance_type(sizes, "H100") == "nebius/nebius-1xh100"


def test_estimate_reports_hours_when_the_plan_has_no_prices() -> None:
    """The community catalogue returns price_per_hour = null — the free tier is
    denominated in hours, so that is what gets shown."""
    est = estimate_cost(_DummyJob(), {}, "T4", [{**s, "price_per_hour": None} for s in _SIZES])
    assert est is not None
    assert est["hours"] == pytest.approx(1.0)      # ABC default estimate
    assert est["instance_type"] == "g4dnxlarge"
    assert est["total"] is None


def test_estimate_adds_dollars_when_the_catalogue_prices_the_box() -> None:
    priced = [{**s, "price_per_hour": 0.5 if s["name"] == "g4dnxlarge" else None} for s in _SIZES]
    est = estimate_cost(_DummyJob(), {}, "T4", priced)
    assert est is not None and est["total"] == pytest.approx(0.5)


def test_estimate_is_silent_when_the_card_is_absent() -> None:
    assert estimate_cost(_DummyJob(), {}, "H100", _SIZES) is None
