"""Unit tests for ColabBackend.

No network and no googleapiclient import: a fake in-memory Drive stands in for
``service.files()``, and the three byte-moving helpers (upload stream/path,
download) are overridden in a test subclass. Everything else — query building,
folder bookkeeping, notebook rendering, status mapping, md5 verification — is
the real code.
"""

from __future__ import annotations

import json
import re
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest

from gpurunner.backends.colab import ColabBackend
from gpurunner.core import Job, JobStatus
from gpurunner.core.models import JobHandle

_FOLDER_MIME = "application/vnd.google-apps.folder"


@pytest.fixture(autouse=True)
def _isolated_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_CONFIG_DIR", str(tmp_path / "config"))


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _DummyJob(Job):
    name = "dummy"
    description = "test job"
    supported_backends = ("kaggle", "colab")

    def requirements(self) -> list[str]:
        return ["numpy"]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"echo": str(params.get("echo", "hello")), "dataset": params.get("dataset", "")}

    def render_remote_code(
        self,
        params: dict[str, Any],
        *,
        shard_index: int = 0,
        total_shards: int = 1,
    ) -> str:
        return f"from pathlib import Path\nPath('/kaggle/working/o.txt').write_text({params['echo']!r})"

    def colab_input_dirs(self, params: dict[str, Any]) -> dict[str, str]:
        ds = self.validate_params(params)["dataset"]
        return {ds: ds} if ds else {}


class _Exec:
    """Mimics the googleapiclient request object: build now, .execute() later."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def execute(self) -> Any:
        return self._fn()


class _FakeFiles:
    _Q_CHILD = re.compile(r"^name = '(?P<name>.*)' and '(?P<parent>[^']*)' in parents")
    _Q_LIST = re.compile(r"^'(?P<parent>[^']*)' in parents")

    def __init__(self, store: dict[str, dict[str, Any]]) -> None:
        self.store = store
        self._next = 1

    def _new_id(self) -> str:
        self._next += 1
        return f"id{self._next}"

    def list(self, q: str = "", fields: str = "", pageSize: int = 100,
             pageToken: str | None = None) -> _Exec:
        def run() -> dict[str, Any]:
            m = self._Q_CHILD.match(q)
            if m:
                name, parent = m.group("name"), m.group("parent")
                hits = [
                    f for f in self.store.values()
                    if f["name"] == name and parent in f["parents"]
                ]
            else:
                m2 = self._Q_LIST.match(q)
                assert m2, f"unsupported query: {q}"
                parent = m2.group("parent")
                hits = [f for f in self.store.values() if parent in f["parents"]]
            out = []
            for f in hits:
                item = {"id": f["id"], "name": f["name"], "mimeType": f["mimeType"]}
                if f["mimeType"] != _FOLDER_MIME:
                    item["size"] = str(len(f["content"]))
                    item["md5Checksum"] = _md5(f["content"])
                out.append(item)
            return {"files": out}

        return _Exec(run)

    def create(self, body: dict[str, Any], fields: str = "", media_body: Any = None) -> _Exec:
        def run() -> dict[str, Any]:
            fid = self._new_id()
            self.store[fid] = {
                "id": fid,
                "name": body["name"],
                "parents": list(body.get("parents") or []),
                "mimeType": body.get("mimeType", "application/octet-stream"),
                "content": b"",
            }
            return {"id": fid}

        return _Exec(run)

    def delete(self, fileId: str) -> _Exec:
        return _Exec(lambda: self.store.pop(fileId, None))

    def get(self, fileId: str, fields: str = "") -> _Exec:
        return _Exec(lambda: {"md5Checksum": _md5(self.store[fileId]["content"])})


class _FakeDrive:
    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}
        self._files = _FakeFiles(self.store)

    def files(self) -> _FakeFiles:
        return self._files


class _TestBackend(ColabBackend):
    """ColabBackend wired to the fake Drive, with byte-moving stubbed out."""

    def __init__(self) -> None:
        super().__init__()
        self.drive = _FakeDrive()

    def _service(self) -> Any:
        return self.drive

    def _upload_stream(self, stream: Any, *, name: str, parent: str, mime: str) -> str:
        fid = self.drive.files().create({"name": name, "parents": [parent]}).execute()["id"]
        self.drive.store[fid]["content"] = stream.getvalue()
        self.drive.store[fid]["mimeType"] = mime
        return fid

    def _upload_path(self, path: Path, *, parent: str) -> str:
        fid = self.drive.files().create({"name": path.name, "parents": [parent]}).execute()["id"]
        self.drive.store[fid]["content"] = path.read_bytes()
        return fid

    def _download_to(self, file_id: str, sink: Any) -> None:
        sink.write(self.drive.store[file_id]["content"])

    # -- test helpers ----------------------------------------------------
    def put(self, parent: str, name: str, content: bytes) -> str:
        fid = self.drive.files().create({"name": name, "parents": [parent]}).execute()["id"]
        self.drive.store[fid]["content"] = content
        return fid


def _sources(nb: dict[str, Any]) -> str:
    """Concatenate cell sources — nbformat serializes them as line lists."""
    out = []
    for cell in nb["cells"]:
        src = cell["source"]
        out.append(src if isinstance(src, str) else "".join(src))
    return "\n".join(out)


def _md5(data: bytes) -> str:
    import hashlib

    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_submit_renders_notebook_with_gpu_metadata_and_kaggle_shim() -> None:
    bk = _TestBackend()
    handle = bk.submit(_DummyJob(), {"echo": "hi"}, gpu="T4")

    assert handle.backend == "colab"
    assert handle.status is JobStatus.QUEUED
    assert handle.volume_name  # run folder id recorded
    assert bk.notebook_url(handle).endswith(handle.remote_id)

    nb = json.loads(bk.drive.store[handle.remote_id]["content"].decode("utf-8"))
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["colab"]["gpuType"] == "T4"

    source = _sources(nb)
    assert "drive.mount" in source
    assert "/kaggle/working" in source
    assert "nvidia-smi" in source
    assert "!pip install numpy" in source
    # The job body is embedded as data for exec(), not pasted inline.
    assert "RUNNER_SRC" in source
    assert json.dumps(_DummyJob().render_remote_code({"echo": "hi"})) in source


def test_generated_cells_are_valid_python() -> None:
    """Guards the `.format()` templates: an unescaped brace would break at runtime,
    three minutes into a Colab session, and only for whoever clicked Run all."""
    bk = _TestBackend()
    data_root = bk._child(bk._root(), "data", create=True)
    bk._child(data_root, "train-v42", create=True)
    handle = bk.submit(_DummyJob(), {"echo": "hi", "dataset": "train-v42"}, gpu="T4")
    nb = json.loads(bk.drive.store[handle.remote_id]["content"].decode("utf-8"))

    for i, cell in enumerate(nb["cells"]):
        src = cell["source"]
        src = src if isinstance(src, str) else "".join(src)
        if src.lstrip().startswith("print('[gpurunner] setup start'"):
            continue  # the !pip cell is IPython, not plain Python
        compile(src, f"<cell{i}>", "exec")


def _exec_bookkeeping_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Actually run the notebook's first cell locally, with google.colab faked.

    Lets us test the runtime behaviour that lives inside the notebook (status
    latching) instead of only asserting on its source text.
    """
    import sys
    import types

    from gpurunner.backends import colab as colab_mod

    fake_drive = types.SimpleNamespace(mount=lambda path: None)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    colab_pkg = types.ModuleType("google.colab")
    colab_pkg.drive = fake_drive  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google.colab", colab_pkg)
    monkeypatch.setattr(colab_mod, "DRIVE_PREFIX", str(tmp_path).replace("\\", "/"))

    bk = _TestBackend()
    handle = bk.submit(_DummyJob(), {}, gpu="T4")
    nb = json.loads(bk.drive.store[handle.remote_id]["content"].decode("utf-8"))
    src = nb["cells"][0]["source"]
    src = src if isinstance(src, str) else "".join(src)

    scope: dict[str, Any] = {"__name__": "__main__"}
    exec(compile(src, "<cell0>", "exec"), scope)
    return scope


def test_notebook_status_is_latched_once_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late heartbeat must not resurrect 'running' over 'completed'.

    Without the latch the run would sit in RUNNING until its heartbeat went
    stale, and `watch` would report UNKNOWN for a job that actually succeeded.
    """
    scope = _exec_bookkeeping_cell(tmp_path, monkeypatch)
    write_status = scope["write_status"]
    status_file = Path(scope["RUN_DIR"]) / "_status.json"

    write_status("running", message="mid-flight")
    assert json.loads(status_file.read_text(encoding="utf-8"))["status"] == "running"

    write_status("completed", message="done")
    write_status("running", message="late heartbeat")  # the race we're guarding
    payload = json.loads(status_file.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["message"] == "done"


def test_notebook_cell_writes_starting_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = _exec_bookkeeping_cell(tmp_path, monkeypatch)
    payload = json.loads((Path(scope["RUN_DIR"]) / "_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "starting"
    assert payload["heartbeat"]
    assert (Path(scope["RUN_DIR"]) / "out").is_dir()


def test_submit_without_gpu_marks_notebook_cpu() -> None:
    bk = _TestBackend()
    handle = bk.submit(_DummyJob(), {}, gpu="none")
    nb = json.loads(bk.drive.store[handle.remote_id]["content"].decode("utf-8"))
    assert nb["metadata"]["accelerator"] == "None"
    assert "gpuType" not in nb["metadata"]["colab"]


def test_submit_rejects_unknown_gpu_and_unsupported_job() -> None:
    bk = _TestBackend()
    with pytest.raises(ValueError, match="Unsupported gpu"):
        bk.submit(_DummyJob(), {}, gpu="H100")

    class _ModalOnly(_DummyJob):
        supported_backends = ("modal",)

    from gpurunner.core.backend import BackendError

    with pytest.raises(BackendError, match="supported_backends"):
        bk.submit(_ModalOnly(), {}, gpu="T4")


def test_submit_fails_early_when_input_missing_from_drive() -> None:
    from gpurunner.core.backend import BackendError

    bk = _TestBackend()
    with pytest.raises(BackendError, match="drive push"):
        bk.submit(_DummyJob(), {"dataset": "train-v42"}, gpu="T4")


def test_submit_stages_declared_input_when_present() -> None:
    bk = _TestBackend()
    data_root = bk._child(bk._root(), "data", create=True)
    bk._child(data_root, "train-v42", create=True)

    handle = bk.submit(_DummyJob(), {"dataset": "train-v42"}, gpu="T4")
    nb = json.loads(bk.drive.store[handle.remote_id]["content"].decode("utf-8"))
    source = _sources(nb)
    assert '"train-v42": "train-v42"' in source or "'train-v42': 'train-v42'" in source


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "completed"}, JobStatus.COMPLETED),
        ({"status": "failed", "error": "boom"}, JobStatus.FAILED),
        ({"status": "running", "heartbeat": "2026-07-21T12:00:00+00:00"}, JobStatus.UNKNOWN),
        ({"status": "weird"}, JobStatus.UNKNOWN),
    ],
)
def test_status_mapping(payload: dict[str, Any], expected: JobStatus) -> None:
    bk = _TestBackend()
    handle = bk.submit(_DummyJob(), {}, gpu="T4")
    assert handle.volume_name is not None
    bk.put(handle.volume_name, "_status.json", json.dumps(payload).encode("utf-8"))
    assert bk.status(handle).status is expected


def test_status_running_with_fresh_heartbeat() -> None:
    from datetime import datetime

    bk = _TestBackend()
    handle = bk.submit(_DummyJob(), {}, gpu="T4")
    assert handle.volume_name is not None
    fresh = datetime.now(tz=UTC).isoformat()
    bk.put(
        handle.volume_name,
        "_status.json",
        json.dumps({"status": "running", "heartbeat": fresh}).encode("utf-8"),
    )
    assert bk.status(handle).status is JobStatus.RUNNING


def test_status_without_file_is_queued_and_tells_you_to_press_run() -> None:
    bk = _TestBackend()
    handle = bk.submit(_DummyJob(), {}, gpu="T4")
    report = bk.status(handle)
    assert report.status is JobStatus.QUEUED
    assert "Run all" in (report.message or "")


def test_fetch_outputs_pulls_out_tree_plus_log_and_status(tmp_path: Path) -> None:
    bk = _TestBackend()
    handle = bk.submit(_DummyJob(), {}, gpu="T4")
    assert handle.volume_name is not None
    out_folder = bk._child(handle.volume_name, "out", create=True)
    assert out_folder is not None
    bk.put(out_folder, "result.txt", b"hello")
    nested = bk._child(out_folder, "weights", create=True)
    assert nested is not None
    bk.put(nested, "best.pt", b"\x00\x01")
    bk.put(handle.volume_name, "_runner.log", b"line1\nline2\n")
    bk.put(handle.volume_name, "_status.json", b'{"status": "completed"}')

    written = bk.fetch_outputs(handle, tmp_path / "out")
    names = {p.relative_to(tmp_path / "out").as_posix() for p in written}
    assert names == {"result.txt", "weights/best.pt", "_runner.log", "_status.json"}
    assert (tmp_path / "out" / "result.txt").read_bytes() == b"hello"
    assert (tmp_path / "out" / "weights" / "best.pt").read_bytes() == b"\x00\x01"


def test_logs_reads_drive_synced_file() -> None:
    bk = _TestBackend()
    handle = bk.submit(_DummyJob(), {}, gpu="T4")
    assert handle.volume_name is not None
    bk.put(handle.volume_name, "_runner.log", b"a\nb\n")
    assert bk.logs(handle) == ["a", "b"]


def test_logs_without_file_explains_rather_than_raising() -> None:
    bk = _TestBackend()
    handle = bk.submit(_DummyJob(), {}, gpu="T4")
    assert "hasn't started" in bk.logs(handle)[0]


def test_cancel_points_at_the_browser() -> None:
    from gpurunner.core.backend import BackendError

    bk = _TestBackend()
    handle = bk.submit(_DummyJob(), {}, gpu="T4")
    with pytest.raises(BackendError, match="no programmatic cancel"):
        bk.cancel(handle)


def test_drive_push_verifies_md5(tmp_path: Path) -> None:
    bk = _TestBackend()
    src = tmp_path / "stage"
    src.mkdir()
    (src / "train.tgz").write_bytes(b"payload" * 100)

    pushed = bk.drive_push(src, name="train-v42")
    assert [p["name"] for p in pushed] == ["train.tgz"]
    assert pushed[0]["md5"] == _md5(b"payload" * 100)

    listing = bk.drive_files("train-v42")
    assert listing == [{"name": "train.tgz", "size": 700, "md5": _md5(b"payload" * 100)}]


def test_drive_push_detects_truncated_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.core.backend import BackendError

    bk = _TestBackend()
    src = tmp_path / "stage"
    src.mkdir()
    (src / "big.tgz").write_bytes(b"x" * 1000)

    def _truncating_upload(path: Path, *, parent: str) -> str:
        fid = bk.drive.files().create({"name": path.name, "parents": [parent]}).execute()["id"]
        bk.drive.store[fid]["content"] = path.read_bytes()[:10]  # simulate a cut-off upload
        return fid

    monkeypatch.setattr(bk, "_upload_path", _truncating_upload)
    with pytest.raises(BackendError, match="md5 mismatch"):
        bk.drive_push(src, name="train-v42")
    # the bad file must not linger on Drive
    assert not any(f["name"] == "big.tgz" for f in bk.drive.store.values())


def test_drive_push_replaces_existing_file(tmp_path: Path) -> None:
    bk = _TestBackend()
    src = tmp_path / "stage"
    src.mkdir()
    (src / "a.bin").write_bytes(b"v1")
    bk.drive_push(src, name="ds")
    (src / "a.bin").write_bytes(b"v2")
    bk.drive_push(src, name="ds")

    files = bk.drive_files("ds")
    assert len(files) == 1
    assert files[0]["md5"] == _md5(b"v2")


def test_run_folder_falls_back_to_lookup_for_legacy_handles() -> None:
    bk = _TestBackend()
    handle = bk.submit(_DummyJob(), {}, gpu="T4")
    folder_id = handle.volume_name
    legacy = JobHandle(
        id=handle.id,
        backend="colab",
        remote_id=handle.remote_id,
        job_name="dummy",
        params={},
        gpu="T4",
    )
    assert bk._run_folder(legacy) == folder_id


def test_default_colab_input_dirs_derives_from_dataset_sources() -> None:
    class _KaggleStyle(_DummyJob):
        def dataset_sources(self, params: dict[str, Any]) -> list[str]:
            return ["owner/train-v42/versions/3"]

        colab_input_dirs = Job.colab_input_dirs  # use the ABC default

    assert _KaggleStyle().colab_input_dirs({}) == {"train-v42": "train-v42"}
