"""Архів догону з машини — HTTP, а SFTP лише запасний.

paramiko SFTP везе архів одним вікном, ~0.48 МБ/с (замір 20.09.2026); на
тисячах бракуючих сторінок це хвилини оренди. Тут — підроблені SSH і HTTP:
перевіряється, яким каналом поїхав архів і що відмова HTTP не губить його.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any

import pytest

from gpurunner.backends.vast import VastBackend
from gpurunner.core.models import JobHandle


def _tgz(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


ARCHIVE = _tgz({"0001.txt": "перша", "0002.txt": "друга"})


class _Out:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> bytes:
        return self._text.encode("utf-8")


class FakeBox:
    """SSH-клієнт машини: журнал команд, `curl` вдається або ні."""

    def __init__(self, *, curl_ok: bool = True) -> None:
        self.cmds: list[str] = []
        self.curl_ok = curl_ok
        self.sftp_got: list[str] = []

    def exec_command(self, cmd: str, timeout: int = 0) -> tuple[Any, Any, Any]:
        self.cmds.append(cmd)
        if cmd.startswith("curl"):
            return None, _Out("rc=0" if self.curl_ok else "rc=22"), None
        return None, _Out("rc=0"), None

    def open_sftp(self) -> Any:
        box = self

        class _Sftp:
            def open(self, path: str, mode: str = "r") -> Any:
                return io.BytesIO()

            def putfo(self, fh: Any, path: str) -> None:
                pass

            def get(self, remote: str, local: str) -> None:
                box.sftp_got.append(remote)
                Path(local).write_bytes(ARCHIVE)

            def remove(self, path: str) -> None:
                pass

            def close(self) -> None:
                pass

        return _Sftp()

    def close(self) -> None:
        pass


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """`fetch_url` додому: віддає архів; журнал адрес."""
    from gpurunner.htr import fetch_ckpt

    got: list[str] = []

    def fake(url: str, dest: Path, *, tries: int = 4) -> None:
        got.append(url)
        dest.write_bytes(ARCHIVE)

    monkeypatch.setattr(fetch_ckpt, "fetch_url", fake)
    return got


def _fetch(monkeypatch, box: FakeBox, dest: Path, **kw: str) -> tuple[VastBackend, int]:
    monkeypatch.setattr(VastBackend, "_ssh", lambda self, h, **k: box)
    backend = VastBackend()
    handle = JobHandle(backend="vast", remote_id="1", job_name="htr_case", gpu="any")
    got = backend.fetch_case_files(handle, "spr-1", ["0001.txt", "0002.txt"], dest, **kw)
    return backend, got


def test_the_archive_goes_through_the_bucket_not_sftp(tmp_path, monkeypatch, http) -> None:
    box = FakeBox()
    backend, got = _fetch(monkeypatch, box, tmp_path / "spr-1",
                          put_url="https://r2/put?sig", get_url="https://r2/get?sig")
    assert got == 2 and backend.last_delta_via == "http"
    assert box.sftp_got == [], "архів не мав їхати SFTP"
    assert any(c.startswith("curl") and "https://r2/put?sig" in c for c in box.cmds)
    assert http == ["https://r2/get?sig"]
    assert (tmp_path / "spr-1" / "0002.txt").read_text(encoding="utf-8") == "друга"


def test_a_failed_upload_from_the_box_falls_back_to_sftp(tmp_path, monkeypatch, http) -> None:
    box = FakeBox(curl_ok=False)
    backend, got = _fetch(monkeypatch, box, tmp_path / "spr-1",
                          put_url="https://r2/put?sig", get_url="https://r2/get?sig")
    assert got == 2 and backend.last_delta_via == "sftp"
    assert box.sftp_got, "запасний шлях мав забрати архів"
    assert http == [], "додому не качали те, чого бокс не залив"


def test_the_box_store_serves_a_copy_so_sftp_still_has_the_original(
        tmp_path, monkeypatch) -> None:
    """Склад на машині: архів КОПІЮЄТЬСЯ в теку складу — якщо HTTP впаде,
    запасному SFTP лишається що брати."""
    from gpurunner.htr import fetch_ckpt

    def broken(url: str, dest: Path, *, tries: int = 4) -> None:
        raise OSError("обрив")

    monkeypatch.setattr(fetch_ckpt, "fetch_url", broken)
    box = FakeBox()
    backend, got = _fetch(monkeypatch, box, tmp_path / "spr-1",
                          serve_at="/workspace/_origin/delta/s/spr-1.tgz",
                          get_url="http://box:1/tok/delta/s/spr-1.tgz")
    assert any(c.startswith("mkdir -p") and " cp -f " in c for c in box.cmds)
    assert not any(" mv " in c for c in box.cmds)
    assert backend.last_delta_via == "sftp" and got == 2
    assert any(c.startswith("rm -f /workspace/_origin/delta") for c in box.cmds), \
        "копію в складі прибрано"


def test_without_an_http_point_it_is_sftp_as_before(tmp_path, monkeypatch, http) -> None:
    box = FakeBox()
    backend, got = _fetch(monkeypatch, box, tmp_path / "spr-1")
    assert got == 2
    assert backend.last_delta_via == "sftp" and http == []


# ── наглядач: куди класти архів ──────────────────────────────────────────────
def _sup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport: str = "r2") -> Any:
    from gpurunner.supervise.htr import Supervisor
    from gpurunner.supervise.plan import CasePlan, Plan

    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    plan = Plan(assets_url="https://r2/a.tgz",
                cases=[CasePlan(case="spr-1", pages_url="https://r2/p.tar", n_pages=3,
                                out_dir=str(tmp_path / "spr-1"))],
                budget_usd=1.0, max_hours=1.0, transport=transport)
    return Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]


def test_the_supervisor_points_the_archive_at_the_bucket(tmp_path, monkeypatch) -> None:
    from gpurunner.htr import r2

    sup = _sup(tmp_path, monkeypatch)
    monkeypatch.setattr(r2, "configured", lambda: True)
    monkeypatch.setattr(r2, "put_url", lambda key, **kw: f"PUT:{key}")
    monkeypatch.setattr(r2, "get_url", lambda key, **kw: f"GET:{key}")
    http, key = sup._delta_http("spr-1")
    assert http == {"put_url": "PUT:delta/S/spr-1.tgz", "get_url": "GET:delta/S/spr-1.tgz"}
    assert key == "delta/S/spr-1.tgz", "ключ прибирається після забору"


def test_the_box_store_is_used_when_the_box_transport_runs(tmp_path, monkeypatch) -> None:
    sup = _sup(tmp_path, monkeypatch, transport="box")
    sup._origin_outside = "http://1.2.3.4:5/tok"
    http, key = sup._delta_http("spr-1")
    assert http == {"serve_at": "/workspace/_origin/delta/S/spr-1.tgz",
                    "get_url": "http://1.2.3.4:5/tok/delta/S/spr-1.tgz"}
    assert key == ""
    sup._origin_outside = ""
    assert sup._delta_http("spr-1") == ({}, ""), "склад мовчить — SFTP, як доти"
