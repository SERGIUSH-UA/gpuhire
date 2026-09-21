"""Unit tests for ModalBackend.

We never hit Modal infra. Tests focus on:
  - the wrapper string compiles to valid Python and round-trips small payloads
  - cancel() returns BackendError when modal isn't reachable
  - the new volume-aware fetch path falls back to legacy when volume_name=None
  - _volume_name_for produces DNS-safe names within Modal's 63-char limit

Real submit/fetch would require modal auth + image build; those are exercised
by examples/paddleocr_modal.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from gpurunner.backends.modal import _REMOTE_WRAPPER, _volume_name_for
from gpurunner.core.backend import AuthError, BackendError
from gpurunner.core.models import JobHandle

# ---------------------------------------------------------------------------
# _volume_name_for
# ---------------------------------------------------------------------------


def test_volume_name_is_under_modal_limit() -> None:
    # Modal volume names: ≤ 64 chars, lowercase alnum + hyphens.
    name = _volume_name_for("a" * 32)
    assert len(name) <= 63
    assert name.startswith("gpurunner-out-")
    assert all(c.isalnum() or c == "-" for c in name)


def test_volume_name_deterministic_from_handle_id() -> None:
    assert _volume_name_for("abc123") == _volume_name_for("abc123")
    assert _volume_name_for("abc123") != _volume_name_for("xyz789")


# ---------------------------------------------------------------------------
# _REMOTE_WRAPPER — exec'd in the local interpreter to catch syntax errors
# and verify the file-collection + tee logic without touching Modal.
# ---------------------------------------------------------------------------


def test_remote_wrapper_compiles() -> None:
    scope: dict[str, Any] = {}
    exec(compile(_REMOTE_WRAPPER, "<wrapper>", "exec"), scope)
    assert "execute" in scope
    assert callable(scope["execute"])


def test_remote_wrapper_calls_main_and_counts_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the wrapper locally against a sandboxed output dir and a fake modal."""
    output_root = tmp_path / "mnt_outputs"

    # Patch Path("/mnt/outputs") by intercepting via monkeypatching the
    # wrapper's globals after compile. Simplest: build a modified wrapper that
    # reads output_root from a global.
    wrapper_src = _REMOTE_WRAPPER.replace('/mnt/outputs', str(output_root).replace("\\", "/"))

    # Fake a modal.Volume.from_name(...).commit() so the import inside the
    # wrapper doesn't blow up.
    class _FakeVolume:
        committed = False
        def commit(self) -> None:
            _FakeVolume.committed = True

    class _FakeModal:
        class Volume:
            @staticmethod
            def from_name(name: str) -> _FakeVolume:
                return _FakeVolume()

    monkeypatch.setitem(sys.modules, "modal", _FakeModal)

    scope: dict[str, Any] = {}
    exec(compile(wrapper_src, "<wrapper>", "exec"), scope)
    execute = scope["execute"]

    runner_src = (
        "def main(params):\n"
        "    from pathlib import Path\n"
        "    out = Path(params['output_root'])\n"
        "    (out / 'a.txt').write_text('hello', encoding='utf-8')\n"
        "    (out / 'sub').mkdir(parents=True, exist_ok=True)\n"
        "    (out / 'sub' / 'b.bin').write_bytes(b'\\x00\\x01\\x02')\n"
        "    return {'note': 'ok'}\n"
    )

    result = execute({}, runner_src, "test-volume")
    assert result["summary"] == {"note": "ok"}
    # 2 user files + _runner.log = 3
    assert result["file_count"] == 3
    assert result["volume_name"] == "test-volume"
    assert (output_root / "a.txt").read_text(encoding="utf-8") == "hello"
    assert (output_root / "sub" / "b.bin").read_bytes() == b"\x00\x01\x02"
    assert (output_root / "_runner.log").exists()
    assert _FakeVolume.committed is True


def test_remote_wrapper_propagates_runner_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "mnt_outputs"
    wrapper_src = _REMOTE_WRAPPER.replace('/mnt/outputs', str(output_root).replace("\\", "/"))

    class _FakeModal:
        class Volume:
            @staticmethod
            def from_name(name: str) -> Any:
                class _V:
                    def commit(self) -> None:
                        pass
                return _V()

    monkeypatch.setitem(sys.modules, "modal", _FakeModal)

    scope: dict[str, Any] = {}
    exec(compile(wrapper_src, "<wrapper>", "exec"), scope)
    execute = scope["execute"]

    runner_src = (
        "def main(params):\n"
        "    raise ValueError('something went wrong inside runner')\n"
    )
    with pytest.raises(RuntimeError, match="something went wrong"):
        execute({}, runner_src, "test-volume")

    # Log file should still exist with the traceback.
    log_path = output_root / "_runner.log"
    assert log_path.exists()
    assert "something went wrong" in log_path.read_text(encoding="utf-8")


def test_remote_wrapper_requires_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "mnt_outputs"
    wrapper_src = _REMOTE_WRAPPER.replace('/mnt/outputs', str(output_root).replace("\\", "/"))

    class _FakeModal:
        class Volume:
            @staticmethod
            def from_name(name: str) -> Any:
                class _V:
                    def commit(self) -> None:
                        pass
                return _V()

    monkeypatch.setitem(sys.modules, "modal", _FakeModal)

    scope: dict[str, Any] = {}
    exec(compile(wrapper_src, "<wrapper>", "exec"), scope)
    execute = scope["execute"]
    with pytest.raises(RuntimeError, match="main"):
        execute({}, "x = 1\n", "test-volume")


# ---------------------------------------------------------------------------
# cancel — error surface
# ---------------------------------------------------------------------------


def test_cancel_wraps_modal_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.backends.modal import ModalBackend

    class _FakeFC:
        @staticmethod
        def from_id(remote_id: str) -> _FakeFC:
            raise RuntimeError("simulated SDK failure")

    class _FakeModal:
        FunctionCall = _FakeFC

    monkeypatch.setitem(sys.modules, "modal", _FakeModal)

    bk = ModalBackend()
    h = JobHandle(backend="modal", remote_id="ap-broken", job_name="x", gpu="T4")
    with pytest.raises(BackendError, match="from_id failed"):
        bk.cancel(h)


def test_cancel_calls_function_call_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.backends.modal import ModalBackend

    calls: list[str] = []

    class _FakeCall:
        def __init__(self, rid: str) -> None:
            self._rid = rid
        def cancel(self) -> None:
            calls.append(self._rid)

    class _FakeFC:
        @staticmethod
        def from_id(remote_id: str) -> _FakeCall:
            return _FakeCall(remote_id)

    class _FakeModal:
        FunctionCall = _FakeFC

    monkeypatch.setitem(sys.modules, "modal", _FakeModal)

    bk = ModalBackend()
    h = JobHandle(backend="modal", remote_id="ap-12345", job_name="x", gpu="T4")
    bk.cancel(h)
    assert calls == ["ap-12345"]


def test_cancel_no_modal_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.backends.modal import ModalBackend

    # Simulate ImportError by removing modal from sys.modules and blocking import.
    monkeypatch.setitem(sys.modules, "modal", None)
    bk = ModalBackend()
    h = JobHandle(backend="modal", remote_id="ap-x", job_name="x", gpu="T4")
    with pytest.raises(AuthError):
        bk.cancel(h)


# ---------------------------------------------------------------------------
# fetch_outputs branches on handle.volume_name
# ---------------------------------------------------------------------------


def test_submit_passes_image_spec_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """PaddleOCRJob sets spec['timeout']=14400 for heavy PEV years; the backend
    must propagate it to @app.function instead of clamping at 3600."""
    from gpurunner.backends.modal import ModalBackend
    from gpurunner.core import Job

    captured: dict[str, Any] = {}

    class _FakeImage:
        def apt_install(self, *a: str) -> _FakeImage:
            return self
        def pip_install(self, *a: str, **kw: Any) -> _FakeImage:
            return self

    class _FakeImageBuilder:
        @staticmethod
        def debian_slim(python_version: str = "3.12") -> _FakeImage:
            return _FakeImage()

    class _FakeVolume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> _FakeVolume:
            return _FakeVolume()

    class _FakeCall:
        object_id = "ap-fake-001"

    class _FakeDecorated:
        def spawn(self, *args: Any, **kwargs: Any) -> _FakeCall:
            return _FakeCall()

    class _FakeApp:
        def __init__(self, name: str) -> None:
            pass
        def function(self, **kwargs: Any) -> Any:
            captured["fn_kwargs"] = dict(kwargs)
            def _wrap(fn: Any) -> _FakeDecorated:
                return _FakeDecorated()
            return _wrap
        def run(self, detach: bool = False) -> Any:
            from contextlib import nullcontext
            return nullcontext()

    class _FakeModal:
        App = _FakeApp
        Image = _FakeImageBuilder
        Volume = _FakeVolume
        @staticmethod
        def enable_output() -> Any:
            from contextlib import nullcontext
            return nullcontext()

    monkeypatch.setitem(sys.modules, "modal", _FakeModal)

    class _TimedJob(Job):
        name = "timed"
        supported_backends = ("modal",)
        def requirements(self) -> list[str]:
            return ["numpy"]
        def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
            return dict(params)
        def render_remote_code(self, params: dict[str, Any], **_: Any) -> str:
            return ""
        def render_runner_module(self) -> str:
            return "def main(p): return {}"
        def modal_image_spec(self) -> dict[str, Any]:
            spec = super().modal_image_spec()
            spec["timeout"] = 14400
            return spec

    bk = ModalBackend()
    handle = bk.submit(_TimedJob(), {}, gpu="T4")
    assert captured["fn_kwargs"]["timeout"] == 14400
    assert handle.remote_id == "ap-fake-001"


def test_submit_default_timeout_when_spec_omits_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Jobs that don't set spec['timeout'] get the backend default."""
    from gpurunner.backends.modal import ModalBackend
    from gpurunner.core import Job

    captured: dict[str, Any] = {}

    class _FakeImage:
        def apt_install(self, *a: str) -> _FakeImage:
            return self
        def pip_install(self, *a: str, **kw: Any) -> _FakeImage:
            return self

    class _FakeImageBuilder:
        @staticmethod
        def debian_slim(python_version: str = "3.12") -> _FakeImage:
            return _FakeImage()

    class _FakeVolume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> _FakeVolume:
            return _FakeVolume()

    class _FakeCall:
        object_id = "ap-fake-002"

    class _FakeDecorated:
        def spawn(self, *args: Any, **kwargs: Any) -> _FakeCall:
            return _FakeCall()

    class _FakeApp:
        def __init__(self, name: str) -> None:
            pass
        def function(self, **kwargs: Any) -> Any:
            captured["fn_kwargs"] = dict(kwargs)
            def _wrap(fn: Any) -> _FakeDecorated:
                return _FakeDecorated()
            return _wrap
        def run(self, detach: bool = False) -> Any:
            from contextlib import nullcontext
            return nullcontext()

    class _FakeModal:
        App = _FakeApp
        Image = _FakeImageBuilder
        Volume = _FakeVolume
        @staticmethod
        def enable_output() -> Any:
            from contextlib import nullcontext
            return nullcontext()

    monkeypatch.setitem(sys.modules, "modal", _FakeModal)

    class _MinimalJob(Job):
        name = "minimal"
        supported_backends = ("modal",)
        def requirements(self) -> list[str]:
            return []
        def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
            return dict(params)
        def render_remote_code(self, params: dict[str, Any], **_: Any) -> str:
            return ""
        def render_runner_module(self) -> str:
            return "def main(p): return {}"

    bk = ModalBackend()
    bk.submit(_MinimalJob(), {}, gpu="T4")
    # Backend default (currently 7200) — exact number is implementation detail,
    # but it must be a positive int and NOT 14400 (which is the override).
    assert isinstance(captured["fn_kwargs"]["timeout"], int)
    assert captured["fn_kwargs"]["timeout"] > 0
    assert captured["fn_kwargs"]["timeout"] != 14400


def test_fetch_outputs_uses_legacy_path_when_volume_name_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from gpurunner.backends.modal import ModalBackend

    class _FakeCall:
        def get(self) -> dict[str, Any]:
            return {
                "summary": {"note": "legacy"},
                "files": {"a.txt": "hello", "b.bin": b"\x00\x01"},
                "files_skipped": [],
            }

    class _FakeFC:
        @staticmethod
        def from_id(remote_id: str) -> _FakeCall:
            return _FakeCall()

    class _FakeModal:
        FunctionCall = _FakeFC

    monkeypatch.setitem(sys.modules, "modal", _FakeModal)

    bk = ModalBackend()
    h = JobHandle(backend="modal", remote_id="ap-legacy", job_name="x", gpu="T4")
    # volume_name is None → legacy path
    assert h.volume_name is None

    out = tmp_path / "out"
    files = bk.fetch_outputs(h, out)
    written_names = {p.name for p in files}
    assert "a.txt" in written_names
    assert "b.bin" in written_names
    assert "_summary.json" in written_names
    assert (out / "a.txt").read_text(encoding="utf-8") == "hello"
    assert (out / "b.bin").read_bytes() == b"\x00\x01"


def test_a_python_mismatch_is_refused_before_anything_is_built() -> None:
    """🔴 Modal серіалізує обгортку між інтерпретаторами: пакет ставиться з 3.11,
    а образи зібрані під 3.12. Розбіжність мусить спинити подачу ДО збирання
    образу — інакше вона спливає помилкою десеріалізації вже в контейнері."""
    from gpurunner.backends.modal import ModalBackend
    from gpurunner.core.backend import BackendError
    from gpurunner.core.job import Job

    class _OtherPythonJob(Job):
        name = "other-python"
        supported_backends = ("modal",)
        def requirements(self) -> list[str]:
            return []
        def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
            return dict(params)
        def render_remote_code(self, params: dict[str, Any], **_: Any) -> str:
            return ""
        def render_runner_module(self) -> str:
            return "def main(p): return {}"
        def modal_image_spec(self) -> dict[str, Any]:
            return {**super().modal_image_spec(), "python_version": "3.99"}

    with pytest.raises(BackendError, match=r"потребує локального Python 3\.99"):
        ModalBackend().submit(_OtherPythonJob(), {}, gpu="T4")
