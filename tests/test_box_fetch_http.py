"""Забір зі складу на машині, коли `scp` мертвий, — без оренди й мережі.

01.10.2026, партія 33 справ на Windows-машині дослідника: `scp.exe` падав на
кожному тіку всіх 13 боксів (`rc=255 Connection closed`), хоч доставка тим
самим складом ішла HTTP. Чекпоінтів удома не було, забір падав у пофайловий
SFTP, висів до стелі фази 30 хв, на `WinError 32` наглядач падав і рятунок ішов
у той самий SFTP ще на 30 хв. $2.05 із $5.89 партії — оренда після прочитаного.
"""
from __future__ import annotations

import io
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest

from gpurunner.htr import origin as origin_mod
from gpurunner.supervise import htr as htr_mod
from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import CasePlan, Plan


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.delenv("GPURUNNER_OWNER", raising=False)
    monkeypatch.setattr(htr_mod, "_staging_root", lambda: tmp_path / "staging")


def _tgz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, blob in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tf.addfile(info, io.BytesIO(blob))
    return buf.getvalue()


class _DeadScp:
    """Бекенд, у якого SSH/scp не працює, а SFTP-обхід рахується."""

    def __init__(self, walk_sec: float = 0.0) -> None:
        self.walks = 0
        self.walk_sec = walk_sec

    def _ssh(self, handle: Any, timeout: int = 30) -> Any:
        raise OSError("scp.EXE rc=255 Connection closed")

    def fetch_outputs(self, handle: Any, dest: Path) -> None:
        self.walks += 1
        time.sleep(self.walk_sec)

    def cancel(self, handle: Any) -> None:
        pass


def _sup(tmp_path: Path, backend: Any) -> Supervisor:
    from gpurunner.core.models import JobHandle

    plan = Plan(assets_url="", transport="box", assets_path="/x/a.tgz",
                cases=[CasePlan(case="sprava", pages_url="", n_pages=4,
                                out_dir=str(tmp_path / "out" / "sprava"),
                                pages_path="/x/s.tar")],
                budget_usd=1.0, max_hours=4.0)
    sup = Supervisor(plan, backend=backend, session="S")  # type: ignore[arg-type]
    sup._handle = JobHandle(backend="vast", remote_id="1", job_name="htr_case", gpu="any")
    return sup


@pytest.fixture
def store(tmp_path: Path):
    """Справжній склад машини на петлі: (корінь, адреса)."""
    root = tmp_path / "origin"
    (root / "ckpt" / "sprava").mkdir(parents=True)
    httpd = origin_mod.serve(root, port=0, host="127.0.0.1")
    try:
        yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


def test_checkpoints_come_home_over_http_when_scp_is_dead(tmp_path: Path, store) -> None:
    root, base = store
    for i in (1, 2):
        (root / "ckpt" / "sprava" / f"ckpt_{i:04d}.tgz").write_bytes(
            _tgz({f"out/{i}.txt": b"x"}))
    sup = _sup(tmp_path, _DeadScp())
    sup._origin_outside = base

    assert sup.sync_box_checkpoints() == 2
    home = sup._box_ckpt_dir()
    assert sorted(p.relative_to(home).as_posix() for p in home.rglob("ckpt_*.tgz")) == \
        ["sprava/ckpt_0001.tgz", "sprava/ckpt_0002.tgz"], \
        "розкладка та сама, що дає scp: її читає забір і рятунок"
    assert "box_ckpt_sync_failed" not in [i.kind for i in sup.state.incidents]

    (root / "ckpt" / "sprava" / "ckpt_0003.tgz").write_bytes(_tgz({"out/3.txt": b"x"}))
    assert sup.sync_box_checkpoints() == 1, "їде лише нове"
    assert sup.sync_box_checkpoints() == 0


def test_a_torn_checkpoint_is_never_kept(tmp_path: Path, store) -> None:
    """Половина архіву вдома — це «привезено» назавжди: наступний тік її
    пропустить, а забір розпакує биту точку відновлення."""
    root, base = store
    (root / "ckpt" / "sprava" / "ckpt_0001.tgz").write_bytes(b"not a tarball")
    sup = _sup(tmp_path, _DeadScp())
    sup._origin_outside = base

    assert sup.sync_box_checkpoints() == 0
    assert not list(sup._box_ckpt_dir().rglob("ckpt_*"))
    kinds = [i.kind for i in sup.state.incidents]
    assert "box_ckpt_http_failed" in kinds, "HTTP не дав — сказано й пробувано scp"


def test_fetch_does_not_walk_sftp_when_the_store_answers(
        tmp_path: Path, store, monkeypatch) -> None:
    """🔴 Головне: чекпоінти вдома — пофайлового SFTP-обходу немає взагалі."""
    root, base = store
    (root / "ckpt" / "sprava" / "ckpt_0001.tgz").write_bytes(
        _tgz({"out/0001.txt": b"x"}))
    backend = _DeadScp()
    sup = _sup(tmp_path, backend)
    sup._origin_outside = base
    monkeypatch.setattr(sup, "_flush_checkpoints", lambda: None)
    monkeypatch.setattr(sup, "_top_up_from_box", lambda staging: None)

    sup._fetch_queue()
    assert backend.walks == 0
    assert "fetch_timeout" not in [i.kind for i in sup.state.incidents]
    assert "fetched_via_box" in [i.kind for i in sup.state.incidents]


def test_sftp_walk_is_not_repeated_after_it_hit_the_ceiling(
        tmp_path: Path, monkeypatch) -> None:
    """Рятунок після збою йшов у той самий обхід на тому самому каналі —
    ще 30 хв оренди (п'ять заходів партії: 61 хв замість 30)."""
    monkeypatch.setattr(htr_mod, "FETCH_PHASE_MAX_SEC", 0.2)
    backend = _DeadScp(walk_sec=1.0)
    sup = _sup(tmp_path, backend)
    monkeypatch.setattr(sup, "_flush_checkpoints", lambda: None)

    sup._fetch_queue()
    assert "fetch_timeout" in [i.kind for i in sup.state.incidents]
    sup._fetch_queue(best_effort=True)
    assert backend.walks == 1, "другого обходу на тій самій машині немає"


def test_new_box_gets_a_fresh_sftp_chance(tmp_path: Path) -> None:
    sup = _sup(tmp_path, _DeadScp())
    sup._sftp_timed_out = True
    sup.backend.http_endpoint = lambda h, port: {}  # type: ignore[attr-defined]
    sup._origin_outside_url(sup._handle)
    assert sup._sftp_timed_out is False


def test_a_locked_file_does_not_crash_placement(tmp_path: Path, monkeypatch) -> None:
    """🔴 `WinError 32` на одному файлі, який тримав покинутий потік SFTP,
    валив наглядача. `_claims` — замки шардів, удома їм не місце."""
    sup = _sup(tmp_path, _DeadScp())
    staging = tmp_path / "stg"
    src = staging / "sprava" / "out"
    (src / "_claims").mkdir(parents=True)
    (src / "_claims" / "1173.claim").write_text("x")
    (src / "0001.txt").write_text("a")
    (src / "0002.txt").write_text("b")

    real = htr_mod.shutil.move

    def move(a: str, b: str) -> Any:
        if a.endswith("0002.txt"):
            raise PermissionError(32, "being used by another process")
        return real(a, b)

    monkeypatch.setattr(htr_mod.shutil, "move", move)
    sup._place_case(sup.plan.cases[0], sup.state.cases[0], staging)

    out = Path(sup.state.cases[0].out_dir)
    assert not list(out.rglob("*.claim"))
    assert list(out.rglob("0001.txt")), "решта справи розкладена"
    assert "place_locked" in [i.kind for i in sup.state.incidents]


def test_service_bundles_come_over_http(tmp_path: Path, store, monkeypatch) -> None:
    root, base = store
    (root / "service").mkdir()
    (root / "service" / "sprava.tgz").write_bytes(_tgz({"logs/shard_0.log": b"ok"}))
    sup = _sup(tmp_path, _DeadScp())
    sup._origin_outside = base
    got = sup._pull_box_service(tmp_path / "stg")
    assert got, "службове приїхало HTTP, хоч scp мертвий"
    assert "service_pull_failed" not in [i.kind for i in sup.state.incidents]


def test_scp_fallback_survives_when_the_store_is_silent(tmp_path: Path) -> None:
    """Склад не відповів — лишається `scp`, як і досі, і збій там — нотатка."""
    sup = _sup(tmp_path, _DeadScp())
    sup._origin_outside = "http://127.0.0.1:9"   # нікого
    assert sup.sync_box_checkpoints() == 0
    kinds = [i.kind for i in sup.state.incidents]
    assert "box_ckpt_http_failed" in kinds and "box_ckpt_sync_failed" in kinds
