"""Unit tests for KaggleBackend.

Real network is touched only by ``test_real_auth``, gated on credentials.
Submit/fetch are exercised with a stubbed API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpurunner.auth import kaggle as kaggle_auth
from gpurunner.backends.kaggle import KaggleBackend
from gpurunner.core import Job, JobStatus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_CONFIG_DIR", str(tmp_path / "config"))


class _DummyJob(Job):
    name = "dummy"
    description = "test job"
    supported_backends = ("kaggle",)

    def requirements(self) -> list[str]:
        return ["numpy"]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"echo": str(params.get("echo", "hello"))}

    def render_remote_code(
        self,
        params: dict[str, Any],
        *,
        shard_index: int = 0,
        total_shards: int = 1,
    ) -> str:
        return f"print({params['echo']!r})"


class _StubKaggleApi:
    """Minimal stub matching the methods KaggleBackend calls."""

    def __init__(self) -> None:
        self.push_calls: list[dict[str, Any]] = []
        self.config: dict[str, str] = {"username": "stubuser"}
        # (file names, next_page_token) — two pages so paging is actually exercised.
        self.output_pages: list[tuple[list[str], str | None]] = [
            (["fake_output.txt"], "page2"),
            (["sub/nested.txt"], None),
        ]

    def authenticate(self) -> None:
        pass

    def get_config_value(self, key: str) -> str | None:
        return self.config.get(key)

    def kernels_push(self, folder: str, timeout: str | None = None, acc: str | None = None) -> Any:
        # capture what was uploaded
        folder_path = Path(folder)
        meta = json.loads((folder_path / "kernel-metadata.json").read_text(encoding="utf-8"))
        nb = (folder_path / "notebook.ipynb").read_text(encoding="utf-8")
        self.push_calls.append({"folder": folder, "acc": acc, "metadata": meta, "notebook": nb})
        return type("Resp", (), {"ref": meta["id"], "versionNumber": 1})()

    def kernels_status(self, kernel: str) -> Any:
        return type("S", (), {"status": "running", "failure_message": None})()

    def kernels_output(
        self,
        kernel: str,
        path: str,
        file_pattern: str | None = None,
        force: bool = False,
        quiet: bool = True,
    ) -> tuple[list[str], str]:
        out = Path(path) / "fake_output.txt"
        out.write_text("hello from stub", encoding="utf-8")
        return [str(out)], ""

    # fetch_outputs no longer uses kernels_output: the SDK helper silently drops
    # next_page_token and truncates at ~500 files. It drives the low-level
    # list_kernel_session_output loop instead, so the stub must model paging.
    def build_kaggle_client(self) -> _StubKaggleClient:
        return _StubKaggleClient(self.output_pages)


class _StubOutputFile:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        self.url = f"https://stub.invalid/{file_name}"


class _StubOutputResponse:
    def __init__(self, names: list[str], next_page_token: str | None) -> None:
        self.files = [_StubOutputFile(n) for n in names]
        self.next_page_token = next_page_token


class _StubKernelsApiClient:
    def __init__(self, pages: list[tuple[list[str], str | None]]) -> None:
        self._pages = pages
        self.requests: list[Any] = []

    def list_kernel_session_output(self, req: Any) -> _StubOutputResponse:
        self.requests.append(req)
        token = getattr(req, "page_token", None)
        idx = 0 if not token else next(
            i for i, (_, tok) in enumerate(self._pages) if tok == token
        ) + 1
        names, next_token = self._pages[idx]
        return _StubOutputResponse(names, next_token)


class _StubKernels:
    def __init__(self, pages: list[tuple[list[str], str | None]]) -> None:
        self.kernels_api_client = _StubKernelsApiClient(pages)


class _StubKaggleClient:
    def __init__(self, pages: list[tuple[list[str], str | None]]) -> None:
        self.kernels = _StubKernels(pages)

    def __enter__(self) -> _StubKaggleClient:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_submit_renders_notebook_and_metadata() -> None:
    backend = KaggleBackend()
    stub = _StubKaggleApi()
    backend._api = stub
    backend._username = "stubuser"

    handle = backend.submit(_DummyJob(), {"echo": "hi"}, gpu="T4")

    assert handle.backend == "kaggle"
    assert handle.remote_id.startswith("stubuser/dummy-")
    assert handle.job_name == "dummy"
    assert handle.gpu == "T4"
    assert handle.status == JobStatus.QUEUED

    assert len(stub.push_calls) == 1
    call = stub.push_calls[0]
    # GPU is selected explicitly via machine_shape (``acc``); relying on
    # enable_gpu alone gave whatever Kaggle defaulted to. See _GPU_TO_ACC.
    assert call["acc"] == "NvidiaTeslaT4"

    meta = call["metadata"]
    assert meta["id"] == handle.remote_id
    assert meta["enable_gpu"] == "true"
    assert meta["enable_internet"] == "true"
    assert meta["kernel_type"] == "notebook"

    nb = call["notebook"]
    # Install is no longer quiet: the setup markers and pip output are what
    # `run --expect-log` polls to confirm the kernel picked up the right data.
    assert "!pip install numpy" in nb
    assert "[gpurunner] setup start" in nb
    assert "[gpurunner] setup done" in nb
    assert "print('hi')" in nb


def test_submit_rejects_unsupported_gpu() -> None:
    backend = KaggleBackend()
    backend._api = _StubKaggleApi()
    backend._username = "stubuser"

    with pytest.raises(ValueError, match="Unsupported gpu"):
        backend.submit(_DummyJob(), {}, gpu="A100")


def test_submit_rejects_unsupported_backend() -> None:
    class _Pinned(_DummyJob):
        supported_backends = ("modal",)

    backend = KaggleBackend()
    backend._api = _StubKaggleApi()
    backend._username = "stubuser"

    with pytest.raises(Exception, match="does not declare 'kaggle'"):
        backend.submit(_Pinned(), {}, gpu="T4")


def test_status_maps_kaggle_states() -> None:
    backend = KaggleBackend()
    stub = _StubKaggleApi()
    backend._api = stub
    backend._username = "stubuser"
    handle = backend.submit(_DummyJob(), {}, gpu="T4")

    # default stub returns "running"
    report = backend.status(handle)
    assert report.status == JobStatus.RUNNING

    # patch stub to return "complete"
    stub.kernels_status = lambda k: type(  # type: ignore[method-assign]
        "S", (), {"status": "complete", "failure_message": None}
    )()
    assert backend.status(handle).status == JobStatus.COMPLETED

    stub.kernels_status = lambda k: type(  # type: ignore[method-assign]
        "S", (), {"status": "error", "failure_message": "boom"}
    )()
    rep = backend.status(handle)
    assert rep.status == JobStatus.FAILED
    assert rep.error == "boom"


class _StubHTTPResponse:
    """Minimal stand-in for requests.Response used as a streaming download."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int = 1) -> Any:
        yield self._payload


def test_fetch_outputs_writes_files(tmp_path: Path, monkeypatch: Any) -> None:
    import requests

    def fake_get(url: str, **kwargs: Any) -> _StubHTTPResponse:
        return _StubHTTPResponse(b"hello from stub")

    monkeypatch.setattr(requests, "get", fake_get)

    backend = KaggleBackend()
    stub = _StubKaggleApi()
    backend._api = stub
    backend._username = "stubuser"
    handle = backend.submit(_DummyJob(), {}, gpu="T4")

    out_dir = tmp_path / "out"
    files = backend.fetch_outputs(handle, out_dir)

    # Both pages must be walked — truncating at page 1 is the exact bug the
    # paginated loop exists to prevent.
    assert [f.name for f in files] == ["fake_output.txt", "nested.txt"]
    assert files[0].read_text(encoding="utf-8") == "hello from stub"
    # nested paths from the kernel output are recreated under out_dir
    assert (out_dir / "sub" / "nested.txt").exists()


# ---------------------------------------------------------------------------
# Live auth test — only runs if local Kaggle credentials exist.
# ---------------------------------------------------------------------------


def _has_kaggle_creds() -> bool:
    try:
        kaggle_auth.discover_credentials()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_kaggle_creds(), reason="no kaggle credentials on this machine")
def test_real_auth_verifies() -> None:
    """End-to-end auth verification — talks to Kaggle, no submit."""
    username = kaggle_auth.verify()
    assert username
    assert username != "<unknown>"
