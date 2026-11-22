"""Ваги епох трену в R2 по ходу трену: посилання, вивантажувач на боксі, забір і прибирання.

Мережі немає ніде: R2 — підставний клієнт S3 у пам'яті, PUT боксу — підставна
функція або локальний HTTP-сервер на 127.0.0.1.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from gpurunner._embedded import _train_out as box
from gpurunner.core import train_out as TO
from gpurunner.jobs import get_job

# ── підставний R2 ─────────────────────────────────────────────────────────────


class FakeS3:
    """Клієнт S3 у пам'яті: presigned-посилання, список, get, download, delete."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.presigned: list[tuple[str, str, int]] = []
        self.deleted: list[str] = []

    def generate_presigned_url(self, op: str, Params: dict[str, str], ExpiresIn: int) -> str:
        self.presigned.append((op, Params["Key"], ExpiresIn))
        return f"https://r2.example/{Params['Key']}?op={op}&exp={ExpiresIn}"

    def get_paginator(self, name: str) -> Any:
        store = self.objects

        class _P:
            def paginate(self, **kw: Any) -> list[dict[str, Any]]:
                pre = kw.get("Prefix", "")
                return [{"Contents": [{"Key": k, "Size": len(v)}
                                      for k, v in sorted(store.items()) if k.startswith(pre)]}]
        return _P()

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def download_file(self, bucket: str, key: str, dst: str) -> None:
        Path(dst).write_bytes(self.objects[key])

    def delete_object(self, Bucket: str, Key: str) -> None:
        self.deleted.append(Key)
        self.objects.pop(Key, None)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _spec(named: tuple[str, ...] = ("ktrain.log",), slots: int = 4, job_id: str = "JOB") -> dict[str, Any]:
    base = f"train_out/{job_id}/"
    return {
        "job_id": job_id, "prefix": base,
        "named": {n: {"key": base + n, "url": f"put://{base}{n}"} for n in named},
        "slots": [{"key": f"{base}slots/slot_{i:03d}", "url": f"put://{base}slots/slot_{i:03d}"}
                  for i in range(slots)],
        "manifest": {"key": base + "_manifest.json", "url": f"put://{base}_manifest.json"},
    }


class Bucket:
    """PUT боксу в «R2»: url → вміст. Уміє падати задану кількість разів."""

    def __init__(self, fail_times: int = 0, during: Any = None) -> None:
        self.store: dict[str, bytes] = {}
        self.calls: list[str] = []
        self.fail_times = fail_times
        self.during = during

    def __call__(self, url: str, body: Any, size: int) -> None:
        self.calls.append(url)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OSError("connection reset")
        data = bytes(body) if isinstance(body, (bytes, bytearray)) else Path(body).read_bytes()
        assert len(data) == size, "Content-Length мусить збігатися з тілом"
        if self.during is not None:
            self.during(url)
        self.store[url] = data

    def manifest(self, job_id: str = "JOB") -> dict[str, Any]:
        return json.loads(self.store[f"put://train_out/{job_id}/_manifest.json"])


def _uploader(spec: dict[str, Any], bucket: Bucket, **kw: Any) -> tuple[box._TrainOutUploader, list[str], list[float]]:
    logs: list[str] = []
    sleeps: list[float] = []
    up = box._TrainOutUploader(spec, put=bucket, log=logs.append, sleep=sleeps.append, **kw)
    return up, logs, sleeps


def _age(path: Path, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


# ── обидва боки знають той самий шлях ────────────────────────────────────────


def test_spec_path_agrees_between_backend_and_box() -> None:
    from gpurunner.backends.vast import REMOTE_ROOT

    assert f"{REMOTE_ROOT}/{TO.SPEC_NAME}" == box._TO_SPEC_PATH


def test_without_a_spec_the_uploader_is_a_noop(tmp_path: Path) -> None:
    logs: list[str] = []
    up = box._to_start(log=logs.append, spec_path=tmp_path / "missing.json")
    assert not up.enabled
    up.watch(tmp_path, "*.ckpt")
    up.offer(tmp_path / "x")
    up.start()
    up.finish(globs=[(tmp_path, "*")])
    assert up._thread is None and any("вимкнено" in m for m in logs)


def test_broken_spec_does_not_break_training(tmp_path: Path) -> None:
    bad = tmp_path / "train_out.json"
    bad.write_text("{not json", encoding="utf-8")
    logs: list[str] = []
    up = box._to_start(log=logs.append, spec_path=bad)
    assert not up.enabled and any("не прочитано" in m for m in logs)


# ── «завершений» файл ────────────────────────────────────────────────────────


def test_file_still_being_written_is_not_uploaded(tmp_path: Path) -> None:
    bucket = Bucket()
    up, _, _ = _uploader(_spec(), bucket, stable_sec=20)
    up.watch(tmp_path, "checkpoint_*.ckpt")
    f = tmp_path / "checkpoint_00-0.5000.ckpt"
    f.write_bytes(b"partial")
    up.poll_once()
    up.drain()
    assert not bucket.store, "перший погляд — ще не відомо, чи дописано"
    f.write_bytes(b"partial-and-more")            # росте між обходами
    _age(f, 60)
    up.poll_once()
    up.drain()
    assert not bucket.store, "розмір змінився — ще пишеться"
    up.poll_once()                                 # той самий розмір, старший за stable_sec
    up.drain()
    assert bucket.store["put://train_out/JOB/slots/slot_000"] == b"partial-and-more"
    assert set(bucket.manifest()["files"]) == {"checkpoint_00-0.5000.ckpt"}


def test_fresh_file_with_stable_size_waits_for_stable_sec(tmp_path: Path) -> None:
    bucket = Bucket()
    up, _, _ = _uploader(_spec(), bucket, stable_sec=3600)
    up.watch(tmp_path, "*.ckpt")
    (tmp_path / "checkpoint_00-0.1.ckpt").write_bytes(b"x")
    for _ in range(3):
        up.poll_once()
        up.drain()
    assert not bucket.store, "щойно записаний файл чекає stable_sec, хоч розмір і не міняється"


def test_newer_checkpoint_marks_the_previous_one_complete(tmp_path: Path) -> None:
    bucket = Bucket()
    up, _, _ = _uploader(_spec(), bucket, stable_sec=3600)
    up.watch(tmp_path, "checkpoint_*.ckpt")
    a = tmp_path / "checkpoint_00-0.5000.ckpt"
    a.write_bytes(b"epoch0")
    _age(a, 30)
    b = tmp_path / "checkpoint_01-0.6000.ckpt"
    b.write_bytes(b"epoch1-in-progress")
    up.poll_once()
    up.drain()
    assert set(bucket.manifest()["files"]) == {"checkpoint_00-0.5000.ckpt"}, \
        "старіша епоха — готова, новіша ще пишеться"


def test_unknown_names_take_slots_and_the_manifest_maps_them(tmp_path: Path) -> None:
    bucket = Bucket()
    up, _, _ = _uploader(_spec(named=("ktrain.log",)), bucket)
    for name, data in (("checkpoint_00-0.5.ckpt", b"c0"), ("ktrain.log", b"log"),
                       ("checkpoint_01-0.6.ckpt", b"c1")):
        (tmp_path / name).write_bytes(data)
        up.offer(tmp_path / name)
    up.drain()
    m = bucket.manifest()
    assert m["files"]["checkpoint_00-0.5.ckpt"]["key"] == "train_out/JOB/slots/slot_000"
    assert m["files"]["checkpoint_01-0.6.ckpt"]["key"] == "train_out/JOB/slots/slot_001"
    assert m["files"]["ktrain.log"]["key"] == "train_out/JOB/ktrain.log", "відоме ім'я — свій ключ"
    assert m["files"]["checkpoint_01-0.6.ckpt"]["sha256"] == _sha(b"c1")
    assert m["files"]["checkpoint_01-0.6.ckpt"]["size"] == 2
    assert bucket.store["put://train_out/JOB/ktrain.log"] == b"log"


def test_same_file_unchanged_is_not_uploaded_twice_and_reuses_its_slot(tmp_path: Path) -> None:
    bucket = Bucket()
    up, _, _ = _uploader(_spec(), bucket)
    f = tmp_path / "parseq_last.pt"
    f.write_bytes(b"state1")
    up.offer(f)
    up.drain()
    up.offer(f)
    up.drain()
    data_puts = [c for c in bucket.calls if "_manifest" not in c]
    assert data_puts == ["put://train_out/JOB/slots/slot_000"]
    f.write_bytes(b"state2-bigger")                 # наступна епоха переписала стан
    up.offer(f)
    up.drain()
    assert bucket.store["put://train_out/JOB/slots/slot_000"] == b"state2-bigger", \
        "переписаний файл іде в ТОЙ САМИЙ слот, а не з'їдає новий"


# ── збої ─────────────────────────────────────────────────────────────────────


def test_transient_errors_are_retried_with_backoff(tmp_path: Path) -> None:
    bucket = Bucket(fail_times=2)
    up, _, sleeps = _uploader(_spec(), bucket, backoff_sec=5)
    f = tmp_path / "checkpoint_00-0.5.ckpt"
    f.write_bytes(b"w")
    up.offer(f)
    up.drain()
    assert bucket.store["put://train_out/JOB/slots/slot_000"] == b"w"
    assert sleeps == [5.0, 10.0], "пауза росте вдвічі"


def test_upload_failure_never_raises_and_is_recorded(tmp_path: Path) -> None:
    bucket = Bucket(fail_times=10**6)
    up, logs, _ = _uploader(_spec(), bucket, retries=2, backoff_sec=0)
    f = tmp_path / "checkpoint_00-0.5.ckpt"
    f.write_bytes(b"w")
    up.offer(f)
    up.drain()                                      # не кидає
    up.finish()                                     # і фінал не кидає
    assert "checkpoint_00-0.5.ckpt" in up._failed
    assert any("трен іде далі" in m for m in logs)


def test_put_function_raising_anything_does_not_escape(tmp_path: Path) -> None:
    def boom(url: str, body: Any, size: int) -> None:
        raise ValueError("weird")
    logs: list[str] = []
    up = box._TrainOutUploader(_spec(), put=boom, log=logs.append, sleep=lambda s: None,
                               retries=0)
    f = tmp_path / "a.ckpt"
    f.write_bytes(b"x")
    up.offer(f)
    up.drain()
    assert "a.ckpt" in up._failed


def test_file_rewritten_during_put_is_not_counted(tmp_path: Path) -> None:
    f = tmp_path / "parseq_last.pt"
    f.write_bytes(b"old")
    rewrites = [0]

    def rewrite(url: str) -> None:
        if rewrites[0] == 0 and "_manifest" not in url:
            rewrites[0] += 1
            f.write_bytes(b"new-longer-state")
    bucket = Bucket(during=rewrite)
    up, _, _ = _uploader(_spec(), bucket)
    up.offer(f)
    up.drain()
    m = bucket.manifest()
    assert m["files"]["parseq_last.pt"]["sha256"] == _sha(b"new-longer-state"), \
        "суміш не зараховано — файл перезалито в новому стані"


def test_exhausted_slots_warn_once_and_keep_going(tmp_path: Path) -> None:
    bucket = Bucket()
    up, logs, _ = _uploader(_spec(named=(), slots=1), bucket)
    for i in range(3):
        f = tmp_path / f"checkpoint_{i:02d}-0.1.ckpt"
        f.write_bytes(b"x")
        up.offer(f)
    up.drain()
    assert sum("слоти вичерпано" in m for m in logs) == 1
    assert len(bucket.manifest()["files"]) == 1


# ── потік і фінал ───────────────────────────────────────────────────────────


def test_background_thread_uploads_offered_files_and_finish_flushes_the_rest(tmp_path: Path) -> None:
    bucket = Bucket()
    up, _, _ = _uploader(_spec(named=("ptrain.log", "parseq_ep01.pt")), bucket, poll_sec=0.05)
    log = tmp_path / "ptrain.log"
    log.write_bytes(b"start\n")
    up.live(log)
    up.start()
    ep = tmp_path / "parseq_ep01.pt"
    ep.write_bytes(b"weights")
    up.offer(ep)
    deadline = time.time() + 10
    while "put://train_out/JOB/parseq_ep01.pt" not in bucket.store and time.time() < deadline:
        time.sleep(0.02)
    assert bucket.store["put://train_out/JOB/parseq_ep01.pt"] == b"weights"
    log.write_bytes(b"start\ndone\n")
    (tmp_path / "ptrain_summary.json").write_text("{}", encoding="utf-8")
    up.finish(globs=[(tmp_path, "ptrain_summary.json")])
    assert not up._thread.is_alive()
    assert bucket.store["put://train_out/JOB/ptrain.log"] == b"start\ndone\n", "фінальний лог"
    m = bucket.manifest()
    assert m["final"] is True and "ptrain_summary.json" in m["files"]


def test_real_http_put_streams_file_and_bytes(tmp_path: Path) -> None:
    got: dict[str, bytes] = {}

    class H(BaseHTTPRequestHandler):
        def do_PUT(self) -> None:
            n = int(self.headers["Content-Length"])
            got[self.path] = self.rfile.read(n)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a: Any) -> None:
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        f = tmp_path / "w.bin"
        f.write_bytes(os.urandom(300_000))
        box._to_http_put(url + "/file?sig=1", str(f), f.stat().st_size, timeout=10)
        box._to_http_put(url + "/manifest", b"{}", 2, timeout=10)
    finally:
        srv.shutdown()
        srv.server_close()
    assert got["/file?sig=1"] == f.read_bytes()
    assert got["/manifest"] == b"{}"


# ── раннери: фінал у finally ────────────────────────────────────────────────


def _rendered_namespace(job: str, params: dict[str, Any]) -> dict[str, Any]:
    code = get_job(job)().render_remote_code(params)
    code = code.rsplit("# --- entry point ---", 1)[0]
    ns: dict[str, Any] = {"__name__": "rendered"}
    exec(compile(code, f"<{job}>", "exec"), ns)
    return ns


def test_kraken_runner_ships_checkpoints_and_log_even_when_training_fails(tmp_path: Path) -> None:
    ns = _rendered_namespace("kraken_train", {"dataset": "me/arrows"})
    work, out = tmp_path / "ktrain", tmp_path / "working"
    (work / "ckpt").mkdir(parents=True)
    out.mkdir()
    ns["WORK"], ns["KAGGLE_WORKING"] = work, out
    spec_file = tmp_path / "train_out.json"
    spec_file.write_text(json.dumps(_spec(named=("ktrain.log",), slots=8)), encoding="utf-8")
    bucket = Bucket()

    def fake_impl(params: dict[str, Any]) -> None:
        up = ns["_to_active"](spec_path=spec_file, put=bucket, sleep=lambda s: None,
                              poll_sec=3600)
        up.watch(work / "ckpt", "checkpoint_*.ckpt")
        up.live(out / "ktrain.log")
        (work / "ckpt" / "checkpoint_00-0.5000.ckpt").write_bytes(b"c0")
        (out / "ktrain.log").write_text("ketos died\n", encoding="utf-8")
        (out / "best_inprogress.ckpt").write_bytes(b"copy")
        raise RuntimeError("ketos train exited 1")

    ns["_main_impl"] = fake_impl
    with pytest.raises(RuntimeError, match="exited 1"):
        ns["main"]({})
    files = bucket.manifest()["files"]
    assert set(files) == {"checkpoint_00-0.5000.ckpt", "ktrain.log"}, \
        "чекпойнт і лог — у R2 навіть після падіння; проміжна копія best — ні"


def test_parseq_runner_main_returns_the_summary_and_flushes(tmp_path: Path) -> None:
    ns = _rendered_namespace("parseq_train", {"dataset": "me/crops"})
    ns["KAGGLE_WORKING"] = tmp_path
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(_spec(named=("parseq_best.pt",), slots=4)), encoding="utf-8")
    bucket = Bucket()

    def fake_impl(params: dict[str, Any]) -> dict[str, Any]:
        ns["_to_active"](spec_path=spec_file, put=bucket, sleep=lambda s: None, poll_sec=3600)
        (tmp_path / "parseq_best.pt").write_bytes(b"best")
        (tmp_path / "ptrain_summary.json").write_text("{}", encoding="utf-8")
        return {"best_val_cer": 0.1}

    ns["_main_impl"] = fake_impl
    assert ns["main"]({}) == {"best_val_cer": 0.1}, "Modal бере повернене значення main"
    assert {"parseq_best.pt", "ptrain_summary.json"} <= set(bucket.manifest()["files"])


def test_rendered_training_code_carries_the_uploader() -> None:
    for job, params in (("kraken_train", {"dataset": "me/a"}), ("parseq_train", {"dataset": "me/b"})):
        code = get_job(job)().render_remote_code(params)
        assert "class _TrainOutUploader" in code and "_to_finish_active" in code, job
        modal = get_job(job)().render_runner_module()
        assert "class _TrainOutUploader" in modal, f"{job}: Modal-модуль теж мусить знати імена"


# ── локальний бік: режим, план, посилання ────────────────────────────────────


@pytest.mark.parametrize(("raw", "want"), [
    (None, "auto"), ("", "auto"), ("auto", "auto"), (False, "off"), ("false", "off"),
    ("0", "off"), (True, "on"), ("yes", "on")])
def test_mode(raw: Any, want: str) -> None:
    assert TO.mode({} if raw is None else {"r2_epochs": raw}) == want


def test_mode_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        TO.mode({"r2_epochs": "maybe"})


def test_wanted_defaults_to_configured_r2(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.htr import r2

    monkeypatch.setattr(r2, "configured", lambda: True)
    assert TO.wanted("kraken_train", {}) is True
    assert TO.wanted("parseq_train", {"r2_epochs": False}) is False
    assert TO.wanted("htr_case", {}) is False, "не трен — не стосується"
    monkeypatch.setattr(r2, "configured", lambda: False)
    assert TO.wanted("kraken_train", {}) is False
    with pytest.raises(RuntimeError, match="не налаштований"):
        TO.wanted("kraken_train", {"r2_epochs": True})


def test_url_hours_are_long_but_within_sigv4_limit() -> None:
    assert TO.url_hours(12, 2) == 48.0
    assert TO.url_hours(100, 2) == 114.0
    assert TO.url_hours(500, 2) == 168.0


def test_plan_kraken_unlimited_epochs_uses_cap() -> None:
    named, slots = TO.plan("kraken_train", {"epochs": -1})
    assert "ktrain.log" in named and "kraken_churro_best.mlmodel" in named
    assert slots == 3 * (TO.KRAKEN_EPOCHS_CAP + 1) + 10
    _, slots20 = TO.plan("kraken_train", {"epochs": 20})
    assert slots20 == 3 * 21 + 10
    _, custom = TO.plan("kraken_train", {"epochs": -1}, {"r2_max_epochs": 5})
    assert custom == 3 * 6 + 10


def test_plan_parseq_names_every_epoch() -> None:
    named, slots = TO.plan("parseq_train", {"epochs": 12})
    assert {"parseq_ep01.pt", "parseq_ep12.pt", "parseq_last.pt", "ptrain.log"} <= set(named)
    assert slots == 32


def test_build_spec_presigns_put_on_every_key(monkeypatch: pytest.MonkeyPatch) -> None:
    s3 = FakeS3()
    spec = TO.build_spec("parseq_train", {"epochs": 2}, {}, "abc123", hours=48, s3=s3)
    assert spec["prefix"] == "train_out/abc123/"
    assert spec["named"]["parseq_ep02.pt"]["key"] == "train_out/abc123/parseq_ep02.pt"
    assert spec["slots"][0]["key"] == "train_out/abc123/slots/slot_000"
    assert spec["manifest"]["key"] == "train_out/abc123/_manifest.json"
    assert {op for op, _, _ in s3.presigned} == {"put_object"}, "лише PUT — читати бокс не може"
    assert {exp for _, _, exp in s3.presigned} == {48 * 3600}
    assert len(s3.presigned) == len(spec["named"]) + len(spec["slots"]) + 1


# ── забір із R2 ──────────────────────────────────────────────────────────────


def _r2_with_run(job_id: str = "JOB") -> FakeS3:
    s3 = FakeS3()
    base = f"train_out/{job_id}/"
    s3.objects[base + "slots/slot_000"] = b"ck0"
    s3.objects[base + "slots/slot_001"] = b"ck1"
    s3.objects[base + "slots/slot_002"] = b"orphan"          # маніфест не встиг
    s3.objects[base + "ktrain.log"] = b"log"
    s3.objects[base + "_manifest.json"] = json.dumps({"files": {
        "checkpoint_00-0.5000.ckpt": {"key": base + "slots/slot_000", "size": 3, "sha256": _sha(b"ck0")},
        "checkpoint_01-0.6000.ckpt": {"key": base + "slots/slot_001", "size": 3, "sha256": _sha(b"ck1")},
        "ktrain.log": {"key": base + "ktrain.log", "size": 3, "sha256": _sha(b"log")},
    }}).encode()
    return s3


@pytest.fixture
def fake_r2(monkeypatch: pytest.MonkeyPatch) -> FakeS3:
    from gpurunner.htr import r2

    s3 = _r2_with_run()
    monkeypatch.setattr(r2, "client", lambda *a, **k: s3)
    monkeypatch.setattr(r2, "bucket_name", lambda explicit="": "bucket")
    return s3


def test_fetch_restores_real_names_and_verifies_hashes(tmp_path: Path, fake_r2: FakeS3) -> None:
    rep = TO.fetch("JOB", tmp_path, s3=fake_r2, log=lambda m: None)
    assert sorted(rep["downloaded"]) == ["checkpoint_00-0.5000.ckpt", "checkpoint_01-0.6000.ckpt",
                                         "ktrain.log"]
    assert (tmp_path / "checkpoint_01-0.6000.ckpt").read_bytes() == b"ck1"
    assert rep["unnamed"] == ["slots/slot_002"], "слот без імені не перейменовується навздогад"
    assert (tmp_path / "train_out_manifest.json").is_file()
    assert not list(tmp_path.glob("*.r2part"))


def test_fetch_never_overwrites_what_the_box_gave(tmp_path: Path, fake_r2: FakeS3) -> None:
    (tmp_path / "ktrain.log").write_bytes(b"newer log from the box")
    rep = TO.fetch("JOB", tmp_path, s3=fake_r2, log=lambda m: None)
    assert "ktrain.log" in rep["present"]
    assert (tmp_path / "ktrain.log").read_bytes() == b"newer log from the box"


def test_fetch_drops_a_corrupted_object(tmp_path: Path, fake_r2: FakeS3) -> None:
    fake_r2.objects["train_out/JOB/slots/slot_001"] = b"XXX"           # той самий розмір, інший вміст
    rep = TO.fetch("JOB", tmp_path, s3=fake_r2, log=lambda m: None)
    assert any("sha256" in f for f in rep["failed"])
    assert not (tmp_path / "checkpoint_01-0.6000.ckpt").exists()


def test_fetch_rejects_path_tricks_from_the_box(tmp_path: Path, fake_r2: FakeS3) -> None:
    base = "train_out/JOB/"
    m = json.loads(fake_r2.objects[base + "_manifest.json"])
    m["files"]["../../evil.py"] = {"key": base + "slots/slot_002", "size": 6}
    fake_r2.objects[base + "_manifest.json"] = json.dumps(m).encode()
    rep = TO.fetch("JOB", tmp_path / "out", s3=fake_r2, log=lambda m: None)
    assert "../../evil.py" in rep["unnamed"]
    assert not (tmp_path / "evil.py").exists()


def test_after_a_good_box_fetch_kraken_checkpoints_are_not_pulled_again(tmp_path: Path,
                                                                         fake_r2: FakeS3) -> None:
    (tmp_path / "epoch_00.mlmodel").write_bytes(b"m")
    rep = TO.fetch("JOB", tmp_path, s3=fake_r2, skip=TO.skip_after_box("kraken_train", tmp_path),
                   log=lambda m: None)
    assert rep["skipped"] == ["checkpoint_00-0.5000.ckpt"]
    assert "checkpoint_01-0.6000.ckpt" in rep["downloaded"], "епохи без ваг — догружаються"


def test_prune_shows_first_and_deletes_only_with_yes(fake_r2: FakeS3) -> None:
    fake_r2.objects["train_out/JOB2/x"] = b"other run"
    shown = TO.prune("JOB", yes=False, s3=fake_r2)
    assert len(shown) == 5 and not fake_r2.deleted
    TO.prune("JOB", yes=True, s3=fake_r2)
    assert all(k.startswith("train_out/JOB/") for k in fake_r2.deleted)
    assert "train_out/JOB2/x" in fake_r2.objects, "чужий захід із префіксом-підрядком цілий"
    with pytest.raises(ValueError):
        TO.prune("", yes=True, s3=fake_r2)


# ── бекенд Vast ──────────────────────────────────────────────────────────────


def test_forced_r2_epochs_without_r2_fails_before_renting(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.auth import vast as vast_auth
    from gpurunner.backends.vast import BackendError, VastBackend
    from gpurunner.htr import r2

    monkeypatch.setattr(vast_auth, "require_ssh_key", lambda: (Path("k"), Path("k.pub")))
    monkeypatch.setattr(r2, "configured", lambda: False)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "train.arrow").write_bytes(b"x")
    bk = VastBackend()
    rented: list[Any] = []
    monkeypatch.setattr(bk, "_create_instance", lambda *a, **k: rented.append(a) or "1")
    with pytest.raises(BackendError, match="r2_epochs"):
        bk.submit_to_offer(get_job("kraken_train")(),
                           {"dataset": "me/a", "r2_epochs": True, "input_root": str(tmp_path)},
                           gpu="RTX_3090", offer={"id": 1})
    assert not rented, "до оренди справа не дійшла"


def test_stage_train_out_puts_the_spec_where_the_box_reads_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.backends.vast import VastBackend
    from gpurunner.core.models import JobHandle

    bk = VastBackend()
    written: dict[str, str] = {}
    monkeypatch.setattr(bk, "_upload_text", lambda c, path, text: written.__setitem__(path, text))
    monkeypatch.setattr(TO, "build_spec", lambda job, norm, raw, job_id, *, hours, s3=None:
                        {**_spec(job_id=job_id), "hours": hours})
    h = JobHandle(backend="vast", remote_id="1", job_name="kraken_train", params={}, gpu="x")
    bk._stage_train_out(object(), get_job("kraken_train")(), {"epochs": -1}, {}, h,
                        {"max_hours": 12, "autodestroy_hours": 2.0})
    spec = json.loads(written[box._TO_SPEC_PATH])
    assert spec["job_id"] == h.id and spec["hours"] == 48.0


def test_stage_failure_is_a_warning_in_auto_and_an_error_when_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.backends.vast import BackendError, VastBackend
    from gpurunner.core.models import JobHandle

    bk = VastBackend()

    def boom(*a: Any, **k: Any) -> dict[str, Any]:
        raise RuntimeError("R2 down")
    monkeypatch.setattr(TO, "build_spec", boom)
    h = JobHandle(backend="vast", remote_id="1", job_name="kraken_train", params={}, gpu="x")
    opts = {"max_hours": 12, "autodestroy_hours": 2.0}
    bk._stage_train_out(object(), get_job("kraken_train")(), {}, {}, h, opts)    # не кидає
    with pytest.raises(BackendError, match="R2 down"):
        bk._stage_train_out(object(), get_job("kraken_train")(), {}, {"r2_epochs": "true"}, h, opts)


# ── CLI fetch ────────────────────────────────────────────────────────────────


def _cli_setup(monkeypatch: pytest.MonkeyPatch, *, box_dead: bool, rep: dict[str, Any],
               final: bool = False) -> list[str]:
    from gpurunner import cli
    from gpurunner.core import manifest
    from gpurunner.htr import r2

    calls: list[str] = []

    class Bk:
        def fetch_outputs(self, h: Any, target: Path, *, skip_present: bool = False) -> list[Path]:
            calls.append("box:skip_present" if skip_present else "box")
            if box_dead:
                raise cli.BackendError("instance 1 no longer exists")
            return [target / "ktrain.log"]
    monkeypatch.setattr(cli, "get_backend", lambda name: Bk)
    monkeypatch.setattr(manifest, "update", lambda h: None)
    monkeypatch.setattr(r2, "configured", lambda: True)
    monkeypatch.setattr(TO, "read_manifest", lambda job_id, s3=None: {"final": final, "files": {}})
    monkeypatch.setattr(TO, "fetch", lambda job_id, out, **kw: calls.append(
        "skip" if kw.get("skip") else "all") or rep)
    monkeypatch.setattr(TO, "prune", lambda job_id, yes=False, s3=None: calls.append(
        f"prune:{yes}") or [("k", 1)])
    return calls


def _r2(calls: list[str]) -> list[str]:
    return [c for c in calls if not c.startswith("box")]


def _handle() -> Any:
    from gpurunner.core.models import JobHandle

    return JobHandle(backend="vast", remote_id="1", job_name="kraken_train", params={}, gpu="x")


def _rep(**kw: Any) -> dict[str, Any]:
    return {"downloaded": [], "present": [], "skipped": [], "failed": [], "unnamed": [],
            "manifest": {"files": {}}, **kw}


def test_fetch_falls_back_to_r2_when_the_box_is_gone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner import cli

    calls = _cli_setup(monkeypatch, box_dead=True, rep=_rep(downloaded=["a", "b"]))
    label, n = cli._fetch_one(_handle(), out=tmp_path, resume=False)
    assert (label, n) == ("ok", 2)
    assert _r2(calls) == ["all"], "бокс знищено — з R2 тягнеться все"


def test_fetch_without_box_and_without_r2_data_is_still_an_error(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner import cli

    _cli_setup(monkeypatch, box_dead=True, rep=_rep(manifest=None))
    label, _ = cli._fetch_one(_handle(), out=tmp_path, resume=False)
    assert label.startswith("error:")


def test_fetch_with_live_box_only_tops_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner import cli

    calls = _cli_setup(monkeypatch, box_dead=False, rep=_rep())
    assert cli._fetch_one(_handle(), out=tmp_path, resume=False)[0] == "ok"
    assert _r2(calls) == ["skip"]
    calls.clear()
    cli._fetch_one(_handle(), out=tmp_path, resume=False, r2_mode="never")
    assert _r2(calls) == []


def test_prune_after_fetch_needs_confirmation_and_no_failures(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    from gpurunner import cli

    calls = _cli_setup(monkeypatch, box_dead=False, rep=_rep(failed=["x: boom"]))
    cli._fetch_one(_handle(), out=tmp_path, resume=False, prune_r2=True, yes=True)
    assert not any(c.startswith("prune") for c in calls), "недотягнуте з R2 не прибирається"

    calls = _cli_setup(monkeypatch, box_dead=False, rep=_rep())
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
    cli._fetch_one(_handle(), out=tmp_path, resume=False, prune_r2=True)
    assert _r2(calls) == ["skip", "prune:False"], "без --yes — лише показ і питання"
    calls.clear()
    cli._fetch_one(_handle(), out=tmp_path, resume=False, prune_r2=True, yes=True)
    assert _r2(calls) == ["skip", "prune:False", "prune:True"]


def test_fetch_takes_weights_from_r2_first_when_the_train_is_final(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """Фінальний маніфест — ваги з R2, бокс лише дотягує, чого вдома нема (10-17: бокс
    із повільним каналом віддавав ~1 ГБ годину, і весь цей час тарифікувався)."""
    from gpurunner import cli

    calls = _cli_setup(monkeypatch, box_dead=False, rep=_rep(downloaded=["parseq_ep01.pt"]), final=True)
    assert cli._fetch_one(_handle(), out=tmp_path, resume=False)[0] == "ok"
    assert calls == ["all", "box:skip_present"], "спершу R2 цілком, потім бокс без наявного"


# ── конвертація чекпойнтів kraken удома ──────────────────────────────────────


def test_kraken_convert_mirrors_the_box(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    from gpurunner.core import kraken_ckpt_convert as KC

    class Net:
        def save_model(self, path: str) -> None:
            Path(path).write_bytes(b"coreml")

    class VGSL:
        @staticmethod
        def load_from_checkpoint(p: str, weights_only: bool = False) -> Any:
            if "02-" in p:
                raise RuntimeError("not VGSL (PP-OCRv6)")
            return types.SimpleNamespace(net=Net())

    mod = types.ModuleType("kraken.train")
    mod.VGSLRecognitionModel = VGSL  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kraken", types.ModuleType("kraken"))
    monkeypatch.setitem(sys.modules, "kraken.train", mod)
    converted: list[tuple[str, str]] = []

    def fake_ketos(src: Path, dst: Path) -> int:
        converted.append((src.name, dst.name))
        dst.write_bytes(b"st")
        return 0
    monkeypatch.setattr(KC, "_ketos_convert", fake_ketos)
    for n in ("checkpoint_01-0.5000.ckpt", "checkpoint_02-0.6000.ckpt", "checkpoint_03-0.7.ckpt"):
        (tmp_path / n).write_bytes(b"c")
    (tmp_path / "epoch_03.safetensors").write_bytes(b"done before")
    made = KC.convert(tmp_path)
    assert made == ["epoch_01.mlmodel", "epoch_02.safetensors"]
    assert converted == [("checkpoint_02-0.6000.ckpt", "epoch_02.safetensors")]
    assert KC.convert(tmp_path, fmt="safetensors") == [], "наявні епохи не переписуються"


def test_skip_superseded_drops_ckpt_only_when_its_epoch_weight_is_in_r2() -> None:
    """10-17: забір Дяка з R2 тягнув по 333 МБ чекпоінтів поряд із готовими epoch_NN."""
    names = {"checkpoint_00-0.91.ckpt", "checkpoint_01-0.92.ckpt", "epoch_00.safetensors", "ktrain.log"}
    skip = TO.skip_superseded(names)
    assert skip("checkpoint_00-0.91.ckpt")
    assert not skip("checkpoint_01-0.92.ckpt"), "епохи 01 у R2 нема — чекпоінт єдине джерело"
    assert not skip("epoch_00.safetensors") and not skip("ktrain.log")
