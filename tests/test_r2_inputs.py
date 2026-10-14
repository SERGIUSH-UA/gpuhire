"""Входи задачі через R2: ключ вмісту, вибір шляху, доставка на бокс і повернення до scp."""
from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from typing import Any

import pytest

from gpurunner.backends.vast import BackendError, VastBackend
from gpurunner.core import r2_inputs as RI


def _dir(tmp_path: Path, name: str = "train-v1", files: dict[str, bytes] | None = None) -> Path:
    d = tmp_path / name
    d.mkdir()
    for n, b in (files or {"train.tgz": b"payload", "val.tgz": b"vv"}).items():
        (d / n).write_bytes(b)
    return d


# ── ключ вмісту ───────────────────────────────────────────────────────────────
def test_content_key_is_stable_and_follows_any_change(tmp_path: Path) -> None:
    d = _dir(tmp_path)
    files = sorted(d.iterdir())
    k1 = RI.content_key("train-v1", files)
    assert k1 == RI.content_key("train-v1", list(reversed(files))), "порядок файлів не важить"
    assert k1.startswith(f"{RI.PREFIX}/train-v1-") and k1.endswith(".tar")
    assert RI.content_key("other", files) != k1, "інший вхід — інший об'єкт"
    (d / "train.tgz").write_bytes(b"payload2")            # інший розмір
    assert RI.content_key("train-v1", sorted(d.iterdir())) != k1
    st = (d / "val.tgz").stat()
    k2 = RI.content_key("train-v1", sorted(d.iterdir()))
    os.utime(d / "val.tgz", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))   # той самий розмір, інший час
    assert RI.content_key("train-v1", sorted(d.iterdir())) != k2


# ── вибір шляху ───────────────────────────────────────────────────────────────
def test_wanted_respects_mode_config_and_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.htr import r2

    d = _dir(tmp_path)
    monkeypatch.setattr(r2, "configured", lambda: True)
    assert RI.wanted("scp", {"x": d}) is False
    assert RI.wanted("r2", {"x": d}) is True
    assert RI.wanted("auto", {"x": d}) is False, "дрібне — scp"
    monkeypatch.setattr(RI, "MIN_MB", 0)
    assert RI.wanted("auto", {"x": d}) is True
    monkeypatch.setattr(r2, "configured", lambda: False)
    assert RI.wanted("auto", {"x": d}) is False, "без бакета auto мовчки бере scp"
    with pytest.raises(RuntimeError, match="не налаштований"):
        RI.wanted("r2", {"x": d})


def test_stage_reuses_an_object_already_in_r2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.htr import r2

    d = _dir(tmp_path)
    put: list[Path] = []
    monkeypatch.setattr(r2, "head", lambda key, **kw: {"size": 123, "modified": None})
    monkeypatch.setattr(r2, "put", lambda path, **kw: put.append(path) or "never")
    monkeypatch.setattr(r2, "get_url", lambda key, **kw: f"https://r2/{key}?sig")
    url, size = RI.stage("train-v1", sorted(d.iterdir()), log=lambda m: None)
    assert size == 123 and url.startswith("https://r2/inputs/train-v1-") and not put, \
        "той самий вміст не заливається вдруге"


def test_stage_uploads_once_under_the_content_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.htr import r2

    d = _dir(tmp_path)
    seen: dict[str, Any] = {}

    def fake_put(path: Path, *, prefix: str = "", **kw: Any) -> str:
        with tarfile.open(path) as tf:
            seen["names"] = sorted(m.name for m in tf.getmembers())
        return f"{prefix}/{path.name}"

    monkeypatch.setattr(r2, "head", lambda key, **kw: None)
    monkeypatch.setattr(r2, "put", fake_put)
    monkeypatch.setattr(r2, "get_url", lambda key, **kw: f"https://r2/{key}")
    url, size = RI.stage("train-v1", sorted(d.iterdir()), log=lambda m: None)
    assert seen["names"] == ["train.tgz", "val.tgz"], "той самий плоский tar, що й у scp"
    assert size > 0 and url == f"https://r2/{RI.content_key('train-v1', sorted(d.iterdir()))}"


# ── доставка на бокс ──────────────────────────────────────────────────────────
class _Client:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.files: dict[str, bytes] = {}


def _backend(monkeypatch: pytest.MonkeyPatch, *, curl_size_delta: int = 0) -> tuple[VastBackend, list[str]]:
    """Бекенд, чий «бокс» уміє `curl` (кладе архів, що лежить у фейковому R2) і `tar -xf`."""
    from gpurunner.htr import box_transport as bt

    bk = VastBackend()
    pushed: list[str] = []
    r2_store: dict[str, bytes] = {}

    def fake_stage(slug: str, files: list[Path], **kw: Any) -> tuple[str, int]:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for p in files:
                tf.add(p, arcname=p.name)
        r2_store["https://r2/x"] = buf.getvalue()
        return "https://r2/x", len(buf.getvalue())

    def fake_exec(client: _Client, command: str) -> str:
        client.commands.append(command)
        if command.startswith("curl "):
            remote = command.split(" -o ", 1)[1].split()[0]
            blob = r2_store["https://r2/x"]
            client.files[remote] = blob
            return str(len(blob) + curl_size_delta)
        if "tar -xf" in command:
            remote = command.split("tar -xf", 1)[1].split()[0]
            blob = client.files.get(remote)
            if blob is None:
                return "0"
            with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
                return str(sum(1 for m in tf.getmembers() if m.isfile()))
        return ""

    def fake_push(ep: Any, local: Path, remote: str, **kw: Any) -> int:
        pushed.append(remote)
        return 0

    monkeypatch.setattr(RI, "stage", fake_stage)
    monkeypatch.setattr(bk, "_exec", fake_exec)
    monkeypatch.setattr(bt, "endpoint_of", lambda backend, handle: object())
    monkeypatch.setattr(bt, "push", fake_push)
    return bk, pushed


def test_inputs_ride_r2_and_scp_is_not_touched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _dir(tmp_path)
    bk, pushed = _backend(monkeypatch)
    monkeypatch.setattr(RI, "wanted", lambda via, inputs: True)
    client = _Client()
    bk._upload_inputs(client, {"train-v1": d}, handle=object(), via="r2")
    assert not pushed, "через R2 scp не потрібен"
    assert any(c.startswith("curl ") for c in client.commands)
    assert any("tar -xf" in c for c in client.commands), "приймач кількості файлів лишився"


def test_short_r2_download_is_refused_in_r2_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _dir(tmp_path)
    bk, pushed = _backend(monkeypatch, curl_size_delta=-1)
    monkeypatch.setattr(RI, "wanted", lambda via, inputs: True)
    with pytest.raises(BackendError, match="R2"):
        bk._upload_inputs(_Client(), {"train-v1": d}, handle=object(), via="r2")
    assert not pushed


def test_auto_falls_back_to_scp_when_r2_breaks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _dir(tmp_path)
    bk, pushed = _backend(monkeypatch, curl_size_delta=-1)
    monkeypatch.setattr(RI, "wanted", lambda via, inputs: True)
    client = _Client()
    bk._upload_inputs(client, {"train-v1": d}, handle=object(), via="auto")
    assert pushed == ["/tmp/_in_train-v1.tar"], "у режимі auto збій R2 повертає до scp"
    assert client.commands[-1].startswith("tar -xf"), "після scp — той самий приймач кількості"
