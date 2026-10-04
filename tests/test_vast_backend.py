"""Unit tests for VastBackend.

No network, no paramiko: `_request` is stubbed with a tiny fake of the Vast REST
API, and the SSH layer is replaced by an in-memory fake filesystem. What stays
real: offer-query construction, onstart rendering, the status state machine
(instance state × job state × cost), input resolution, and the fetch walk.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pytest

from gpurunner.backends.vast import REMOTE_ROOT, REMOTE_WORKING, VastBackend
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


class _FakeChannel:
    def __init__(self) -> None:
        self.timeout: float | None = None

    def settimeout(self, sec: float) -> None:
        self.timeout = sec


class _FakeSFTP:
    """In-memory stand-in for paramiko's SFTPClient (only what we call)."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.uploaded: list[tuple[str, str]] = []
        self.channel = _FakeChannel()

    def get_channel(self):
        return self.channel

    def put(self, local: str, remote: str) -> None:
        self.files[remote] = Path(local).read_bytes()
        self.uploaded.append((local, remote))

    def stat(self, remote: str) -> Any:
        import types

        return types.SimpleNamespace(st_size=len(self.files[remote]))

    def get(self, remote: str, local: str) -> None:
        if remote not in self.files:
            raise OSError(f"no such file: {remote}")
        Path(local).write_bytes(self.files[remote])

    def listdir_attr(self, remote: str) -> list[Any]:
        import stat as stat_mod
        import types

        prefix = remote.rstrip("/") + "/"
        names: dict[str, bool] = {}
        for path in self.files:
            if not path.startswith(prefix):
                continue
            rest = path[len(prefix):]
            head, _, tail = rest.partition("/")
            names[head] = bool(tail)
        if not names and not any(p.startswith(prefix) for p in self.files):
            raise OSError(f"no such dir: {remote}")
        return [
            types.SimpleNamespace(
                filename=name,
                st_mode=(stat_mod.S_IFDIR | 0o755) if is_dir else (stat_mod.S_IFREG | 0o644),
            )
            for name, is_dir in sorted(names.items())
        ]

    def open(self, remote: str, mode: str = "r") -> Any:
        if "w" in mode:
            return _WriteCM(self, remote)
        if remote not in self.files:
            raise OSError(f"no such file: {remote}")
        import io

        return _CM(io.BytesIO(self.files[remote]))

    def close(self) -> None:
        pass


class _WriteCM:
    """Writable half of the fake: `sftp.open(path, "wb")` must land bytes that a
    following `sftp.stat()` can size up — that round trip is how `_upload_text`
    detects a truncated upload."""

    def __init__(self, sftp: _FakeSFTP, remote: str) -> None:
        self.sftp = sftp
        self.remote = remote
        self.sftp.files[remote] = b""
        self.pipelined = False

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def set_pipelined(self, value: bool = True) -> None:
        """🔴 Фейк мусить уміти те саме, що справжній paramiko.

        Без конвеєра paramiko пише СИНХРОННО шматками по 32 КБ: ціна файла —
        не смуга, а `розмір / 32 КБ × круговий оберт`. Заміряно 23.09.2026:
        `job.py` на 48 МБ поїхав до Кореї 25 хвилин, і в лозі це була тиша.
        Якби фейк цього методу не мав, тест просто не пустив би виправлення.
        """
        self.pipelined = bool(value)

    def write(self, data: bytes) -> None:
        self.sftp.files[self.remote] += data


class _CM:
    def __init__(self, buf: Any) -> None:
        self.buf = buf

    def __enter__(self) -> Any:
        return self.buf

    def __exit__(self, *exc: Any) -> bool:
        return False

    def read(self) -> bytes:
        return self.buf.read()


class _FakeTransport:
    def __init__(self) -> None:
        self.keepalive: int | None = None

    def set_keepalive(self, sec: int) -> None:
        self.keepalive = sec


class _FakeSSH:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.commands: list[str] = []
        self.closed = False

    def open_sftp(self) -> _FakeSFTP:
        return _FakeSFTP(self.files)

    def get_transport(self):
        # 🔴 Keepalive й таймаути читання — не косметика: без них завмерле
        # з'єднання вішало забір НАЗАВЖДИ (двічі за ніч 2026-08-12, по 4.4 і
        # 4.8 год оренди за нуль роботи). Фейк мусить це підтримувати, інакше
        # тест перестане покривати саме ту гілку, що зламалась.
        return _FakeTransport()

    def close(self) -> None:
        self.closed = True


class _TestBackend(VastBackend):
    """VastBackend with the REST API and SSH replaced by fakes."""

    OFFERS: ClassVar[list[dict[str, Any]]] = [
        {"id": 111, "gpu_name": "RTX 3090", "num_gpus": 1, "dph_total": 0.21,
         "disk_space": 200.0, "inet_down": 500.0, "reliability2": 0.99},
        {"id": 222, "gpu_name": "RTX 3090", "num_gpus": 1, "dph_total": 0.35,
         "disk_space": 400.0, "inet_down": 900.0, "reliability2": 0.99},
    ]

    def __init__(self, *, instance: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.remote_files: dict[str, bytes] = {}
        self.ssh_ok = True
        self.instance_row: dict[str, Any] = instance or {
            "actual_status": "running",
            "ssh_host": "ssh1.vast.ai",
            "ssh_port": 40000,
            "dph_total": 0.21,
            "gpu_name": "RTX 3090",
            "start_date": (datetime.now(tz=UTC) - timedelta(hours=2)).timestamp(),
        }
        self.destroyed = False

    def _api_key(self) -> str:
        return "test-key"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.requests.append((method, path, kwargs))
        if path == "/bundles/":
            return {"offers": list(self.OFFERS)}
        if re.fullmatch(r"/asks/\d+/", path):
            return {"success": True, "new_contract": 987654}
        if re.fullmatch(r"/instances/\d+/", path) and method == "GET":
            return {"instances": dict(self.instance_row)}
        if re.fullmatch(r"/instances/\d+/", path) and method == "DELETE":
            self.destroyed = True
            return {"success": True}
        if path.endswith("/ssh/"):
            return {"success": True}
        if "request_logs" in path:
            return {"result_url": "https://example.invalid/logs"}
        return {}

    def _ssh(self, handle: Any, *, timeout: int = 30) -> Any:
        if not self.ssh_ok:
            raise BackendError("ssh refused (fake)")
        return _FakeSSH(self.remote_files)

    def _wait_for_ssh(self, handle: Any, *, timeout: int = 900) -> Any:
        return self._ssh(handle)

    @staticmethod
    def _exec(client: Any, command: str) -> str:
        """Виконує на «машині» те, від чого залежить приймач заливки.

        🔴 Фейк мусить РОЗПАКУВАТИ архів, а не вдавати. Інакше тест не помітить,
        що заливка привезла порожнє: саме перевірка «скільки файлів
        розпакувалось» і є тут приймачем.
        """
        client.commands.append(command)
        if "tar -xf" in command:
            import io
            import tarfile

            remote_tar = command.split("tar -xf", 1)[1].split()[0]
            dest = command.split("-C", 1)[1].split()[0]
            blob = client.files.get(remote_tar) or _FAKE_SCP_SINK.get(remote_tar)
            if blob is None:
                return "0"
            with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    src = tf.extractfile(member)
                    if src is not None:
                        client.files[f"{dest}/{member.name}"] = src.read()
            client.files.pop(remote_tar, None)
            return str(sum(1 for k in client.files if k.startswith(dest + "/")))
        return ""


@pytest.fixture()
def _fake_scp(monkeypatch: pytest.MonkeyPatch):
    """`scp` без мережі: кладе байти туди, куди поклала б справжня доставка.

    Підміняється саме `box_transport.push` — той самий шов, яким заливка
    користується в бою. Список викликів повертається тесту, щоб він міг
    перевірити, що архів ОДИН на теку, а не файл на файл.
    """
    from gpurunner.htr import box_transport as bt

    calls: list[tuple[Path, str]] = []

    def fake_push(ep: Any, local: Path, remote: str, **kw: Any) -> int:
        calls.append((Path(local), remote))
        _FAKE_SCP_SINK[remote] = Path(local).read_bytes()
        return Path(local).stat().st_size

    monkeypatch.setattr(bt, "push", fake_push)
    monkeypatch.setattr(bt, "endpoint_of", lambda backend, handle: object())
    _FAKE_SCP_SINK.clear()
    return calls


#: Куди «доїхали» файли підробленого scp — фейк SSH читає звідси.
_FAKE_SCP_SINK: dict[str, bytes] = {}


@pytest.fixture()
def _ssh_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    priv = tmp_path / "id_ed25519"
    priv.write_text("PRIVATE", encoding="utf-8")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test", encoding="utf-8")
    monkeypatch.setenv("GPURUNNER_VAST_SSH_KEY", str(priv))
    return priv


# ---------------------------------------------------------------------------


def test_search_offers_query_filters_and_sorts_by_price(_ssh_key: Path) -> None:
    bk = _TestBackend()
    bk.search_offers(gpu="RTX3090", max_price=0.4, disk_gb=100, num_gpus=1)
    _, path, kwargs = bk.requests[-1]
    assert path == "/bundles/"
    q = kwargs["json"]
    assert q["gpu_name"] == {"eq": "RTX 3090"}
    assert q["rentable"] == {"eq": True} and q["rented"] == {"eq": False}
    # 🔴 Стеля ринку — `min(запит, ABSOLUTE_MAX_DPH)`: запит на 0.4 зрізається
    # до 0.365. Тверда межа живе в КОДІ, щоб її не обійшов ані забутий
    # параметр, ані план, згенерований до її появи.
    assert q["dph_total"] == {"lte": 0.365}
    assert q["disk_space"] == {"gte": 100.0}
    assert q["order"] == [["dph_total", "asc"]]
    assert q["type"] == "on-demand"


def test_pick_offer_takes_the_cheapest(_ssh_key: Path) -> None:
    bk = _TestBackend()
    offer = bk.pick_offer(gpu="RTX3090", max_price=None, disk_gb=60, num_gpus=1)
    assert offer["id"] == 111


def test_pick_offer_raises_with_actionable_message(_ssh_key: Path) -> None:
    class _Empty(_TestBackend):
        OFFERS: ClassVar[list[dict[str, Any]]] = []

    with pytest.raises(BackendError, match="max_price"):
        _Empty().pick_offer(gpu="T4", max_price=0.05, disk_gb=60, num_gpus=1)


def test_submit_creates_instance_uploads_inputs_and_releases_go(
    tmp_path: Path, _ssh_key: Path, _fake_scp: list
) -> None:
    data = tmp_path / "train-v42"
    data.mkdir()
    (data / "train.tgz").write_bytes(b"payload")

    bk = _TestBackend()
    handle = bk.submit(
        _DummyJob(),
        {"echo": "hi", "dataset": "train-v42", "input_root": str(tmp_path)},
        gpu="RTX3090",
    )

    assert handle.remote_id == "987654"
    assert handle.status is JobStatus.RUNNING
    assert "RTX 3090" in (handle.volume_name or "") and "0.210" in (handle.volume_name or "")

    put = next(r for r in bk.requests if r[0] == "PUT" and r[1].startswith("/asks/"))
    payload = put[2]["json"]
    assert payload["runtype"] == "ssh_direc ssh_proxy"
    assert payload["disk"] == 60.0

    # dataset landed where the Kaggle-shaped job will look for it
    assert "/kaggle/input/train-v42/train.tgz" in bk.remote_files
    assert bk.remote_files["/kaggle/input/train-v42/train.tgz"] == b"payload"

    # 🔴🔴 І ОДНИМ архівом, а не файл за файлом. Доти тут стояв paramiko SFTP:
    # заміряно 0.48 МБ/с проти 7.56 у системного `scp`, плюс ДВА обміни на
    # кожен файл (`put` і `stat`) — тека з тисячі кадрів платила дві тисячі
    # кругових обертів через океан.
    assert len(_fake_scp) == 1, f"доставка пішла {len(_fake_scp)} разів замість одного"
    local, remote = _fake_scp[0]
    assert local.name.endswith(".tar") and remote.endswith(".tar")

    # the SSH key was attached to the live instance
    assert any(r[1].endswith("/ssh/") for r in bk.requests)


def test_job_py_is_uploaded_and_onstart_waits_for_the_go_sentinel(_ssh_key: Path) -> None:
    """The job source travels over SFTP, not inside the onstart script.

    It used to be base64-inlined into the onstart (a JSON payload the container
    shell re-quotes). Now it goes over the same SFTP session as the inputs, so
    the onstart's only job is to wait for the `GO` the upload touches.
    """
    bk = _TestBackend()
    bk.submit(_DummyJob(), {"echo": "hi"}, gpu="RTX3090")
    put = next(r for r in bk.requests if r[0] == "PUT" and r[1].startswith("/asks/"))
    onstart = put[2]["json"]["onstart"]

    assert f"{REMOTE_ROOT}/GO" in onstart  # handshake present
    assert REMOTE_WORKING in onstart
    assert "GPURUNNER_EOF" not in onstart  # no inlined source any more

    wrapper = bk.remote_files[f"{REMOTE_ROOT}/job.py"].decode("utf-8")
    compile(wrapper, "<wrapper>", "exec")  # the shipped python must be valid
    assert json.dumps("print('hi')") in wrapper
    assert "write_status" in wrapper and "_FINALIZED" in wrapper


def test_onstart_autodestroy_is_opt_in(_ssh_key: Path) -> None:
    bk = _TestBackend()
    bk.submit(_DummyJob(), {}, gpu="RTX3090")
    plain = next(r for r in bk.requests if r[1].startswith("/asks/"))[2]["json"]["onstart"]
    assert "self-destruct" not in plain

    bk2 = _TestBackend()
    bk2.submit(_DummyJob(), {"autodestroy_hours": 2}, gpu="RTX3090")
    armed = next(r for r in bk2.requests if r[1].startswith("/asks/"))[2]["json"]["onstart"]
    assert "self-destruct in 7200s" in armed
    assert "CONTAINER_API_KEY" in armed


def test_submit_requires_input_location_when_job_declares_one(_ssh_key: Path) -> None:
    bk = _TestBackend()
    with pytest.raises(BackendError, match="input_root"):
        bk.submit(_DummyJob(), {"dataset": "train-v42"}, gpu="RTX3090")


def test_submit_rejects_missing_input_dir(tmp_path: Path, _ssh_key: Path) -> None:
    bk = _TestBackend()
    with pytest.raises(BackendError, match="not a directory"):
        bk.submit(
            _DummyJob(),
            {"dataset": "nope", "input_root": str(tmp_path)},
            gpu="RTX3090",
        )


def test_submit_rejects_unknown_gpu(_ssh_key: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported gpu"):
        _TestBackend().submit(_DummyJob(), {}, gpu="TPU")


def _handle(bk: _TestBackend, **params: Any) -> Any:
    return bk.submit(_DummyJob(), params, gpu="RTX3090")


def test_status_reports_queued_while_the_box_boots(_ssh_key: Path) -> None:
    bk = _TestBackend()
    h = _handle(bk)
    bk.instance_row["actual_status"] = "loading"
    report = bk.status(h)
    assert report.status is JobStatus.QUEUED
    assert "loading" in (report.message or "")


def test_status_includes_accrued_cost_and_destroy_reminder(_ssh_key: Path) -> None:
    bk = _TestBackend()
    h = _handle(bk)
    bk.remote_files[f"{REMOTE_ROOT}/_status.json"] = json.dumps(
        {"status": "completed", "message": "3 files"}
    ).encode("utf-8")
    report = bk.status(h)
    assert report.status is JobStatus.COMPLETED
    # 2 h at $0.21/h — the user must see money, and that it keeps ticking
    assert "$0.42" in (report.message or "")
    assert "cancel" in (report.message or "")


def test_status_flags_stale_heartbeat(_ssh_key: Path) -> None:
    bk = _TestBackend()
    h = _handle(bk)
    old = (datetime.now(tz=UTC) - timedelta(hours=1)).isoformat()
    bk.remote_files[f"{REMOTE_ROOT}/_status.json"] = json.dumps(
        {"status": "running", "heartbeat": old}
    ).encode("utf-8")
    assert bk.status(h).status is JobStatus.UNKNOWN


def test_status_running_with_fresh_heartbeat(_ssh_key: Path) -> None:
    bk = _TestBackend()
    h = _handle(bk)
    bk.remote_files[f"{REMOTE_ROOT}/_status.json"] = json.dumps(
        {"status": "running", "heartbeat": datetime.now(tz=UTC).isoformat()}
    ).encode("utf-8")
    assert bk.status(h).status is JobStatus.RUNNING


def test_status_dead_instance_without_job_status_is_failed(_ssh_key: Path) -> None:
    bk = _TestBackend()
    h = _handle(bk)
    bk.instance_row["actual_status"] = "exited"
    bk.ssh_ok = False
    report = bk.status(h)
    assert report.status is JobStatus.FAILED
    assert "exited" in (report.error or "")


def test_fetch_outputs_walks_working_dir(tmp_path: Path, _ssh_key: Path) -> None:
    bk = _TestBackend()
    h = _handle(bk)
    bk.remote_files[f"{REMOTE_WORKING}/result.txt"] = b"hello"
    bk.remote_files[f"{REMOTE_WORKING}/runs/weights/best.pt"] = b"\x00\x01"
    bk.remote_files[f"{REMOTE_ROOT}/_runner.log"] = b"log line\n"

    written = bk.fetch_outputs(h, tmp_path / "out")
    names = {p.relative_to(tmp_path / "out").as_posix() for p in written}
    assert names == {"result.txt", "runs/weights/best.pt", "_runner.log"}
    assert (tmp_path / "out" / "runs" / "weights" / "best.pt").read_bytes() == b"\x00\x01"


def test_cancel_destroys_the_instance(_ssh_key: Path) -> None:
    bk = _TestBackend()
    h = _handle(bk)
    bk.cancel(h)
    assert bk.destroyed is True


def test_logs_prefer_the_remote_runner_log(_ssh_key: Path) -> None:
    bk = _TestBackend()
    h = _handle(bk)
    bk.remote_files[f"{REMOTE_ROOT}/_runner.log"] = b"a\nb\n"
    assert bk.logs(h) == ["a", "b"]


def test_explicit_inputs_param_overrides_job_declaration(
    tmp_path: Path, _ssh_key: Path, _fake_scp: list
) -> None:
    custom = tmp_path / "elsewhere"
    custom.mkdir()
    (custom / "x.bin").write_bytes(b"z")

    bk = _TestBackend()
    bk.submit(
        _DummyJob(),
        {"dataset": "train-v42", "inputs": {"train-v42": str(custom)}},
        gpu="RTX3090",
    )
    assert bk.remote_files["/kaggle/input/train-v42/x.bin"] == b"z"


# ---- спільна стеля API: 5 запитів/с на КЛЮЧ, а не на процес (2026-08-19) ---


def test_429_is_retried_with_the_delay_vast_asks_for(monkeypatch) -> None:
    """🔴 Дві сесії ділять стелю ключа, і 429 сипле в ОБИДВА логи, хоча кожна
    окремо поводиться чемно. Раніше запит просто падав: добір лишався без
    ринку й ходив по колу («⚠ запит ринку не вдався»), а на оренді той самий
    429 читався як `market_busy` і витрачав спробу.
    """
    from gpurunner.backends import vast as vast_mod
    from gpurunner.core.backend import BackendError

    calls = []
    slept = []

    def fake_once(self, method, path, *, api_version="v0", **kw):
        calls.append(path)
        if len(calls) < 3:
            err = BackendError("429")
            err.status_code = 429
            err.body = '{"error":"HTTPTooManyRequests","retry_after":7}'
            raise err
        return {"ok": True}

    monkeypatch.setattr(vast_mod.VastBackend, "_request_once", fake_once)
    monkeypatch.setattr(vast_mod, "_throttle_vast_api", lambda: None)
    monkeypatch.setattr(vast_mod.time, "sleep", slept.append)

    assert vast_mod.VastBackend()._request("POST", "/bundles/") == {"ok": True}
    assert len(calls) == 3
    # Чекаємо СТІЛЬКИ, СКІЛЬКИ СКАЗАВ Vast, плюс доважок проти локстепу двох сесій.
    assert slept and all(s >= 7 for s in slept)


def test_429_gives_up_after_the_retries(monkeypatch) -> None:
    """Нескінченно не перепитуємо: борг може бути й не наш."""
    from gpurunner.backends import vast as vast_mod
    from gpurunner.core.backend import BackendError

    def always_429(self, method, path, *, api_version="v0", **kw):
        err = BackendError("429")
        err.status_code = 429
        err.body = "{}"
        raise err

    monkeypatch.setattr(vast_mod.VastBackend, "_request_once", always_429)
    monkeypatch.setattr(vast_mod, "_throttle_vast_api", lambda: None)
    monkeypatch.setattr(vast_mod.time, "sleep", lambda _s: None)

    with pytest.raises(BackendError):
        vast_mod.VastBackend()._request("POST", "/bundles/")


def test_throttle_is_shared_through_a_file_not_process_memory(tmp_path, monkeypatch) -> None:
    """Позначка часу мусить лежати в СПІЛЬНОМУ файлі — інакше друга сесія про
    першу не знає, і стеля тримається лише всередині процесу."""
    import time as _time

    from gpurunner.backends import vast as vast_mod

    monkeypatch.setattr(vast_mod, "_rate_state_path", lambda: tmp_path / "rl")
    t0 = _time.monotonic()
    vast_mod._throttle_vast_api()
    vast_mod._throttle_vast_api()
    assert _time.monotonic() - t0 >= vast_mod._RATE_MIN_INTERVAL_SEC * 0.9
    assert (tmp_path / "rl").exists()


def test_price_ceiling_scales_with_the_number_of_cards(_ssh_key: Path) -> None:
    """Стеля $0.365 стояла на БОКС: 2×3090 за $0.40 не потрапляли у видачу,
    хоч на карту це $0.20 (ринок 05.09.2026: 2×3090 $0.264–0.355)."""
    bk = _TestBackend()
    bk.search_offers(gpu="RTX3090", max_price=0.4, disk_gb=100, num_gpus=2)
    q = bk.requests[-1][2]["json"]
    assert q["dph_total"] == {"lte": pytest.approx(0.73)}
    assert q["num_gpus"] == {"gte": 2}


def test_inputs_never_travel_by_paramiko_sftp() -> None:
    """🔴🔴 Сторож проти повернення найгіршого каналу з усіх наявних.

    Заміряно на орендованому боксі 20.09.2026, 150 МБ: paramiko SFTP **0.48
    МБ/с**, системний `scp` 7.56, HTTP 20.07. paramiko жене дані одним вікном.
    Тут він стояв до 23.09.2026, і я двічі звітував, що «прибрано з передачі
    даних», маючи на увазі лише забір і службові файли — а заливка входів
    лишалась на ньому.

    Дивимось у ДЖЕРЕЛО функції, бо поведінковий тест цього не ловить: фейк
    підміняє і те, і те, і обидва «працюють».
    """
    from tests.srcprobe import method_body

    src = (Path(__file__).resolve().parents[1] / "src" / "gpurunner"
           / "backends" / "vast.py").read_text(encoding="utf-8")
    body = method_body(src, "_upload_inputs")
    assert "open_sftp" not in body and "sftp.put" not in body, (
        "paramiko повернувся в заливку входів — це 0.48 МБ/с проти 7.56")
    assert "bt.push(" in body, "канал мусить бути системним `scp`"
    assert "tarfile.open" in body, (
        "без архіву кожен файл платить два оберти через океан")
    assert "wc -l" in body, (
        "приймач — КІЛЬКІСТЬ розпакованих файлів; без нього обірвана заливка "
        "лишає теку, що виглядає повною")


def test_an_upload_that_arrived_empty_is_a_refusal(
    tmp_path: Path, _ssh_key: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """🔴 Тека входу, що виглядає повною, — найдорожча з можливих помилок.

    `scp` міг віддати нуль (обрив, повний диск на боксі), а `tar -xf` мовчки
    розпакувати нічого. Без перевірки КІЛЬКОСТІ захід їде далі й падає вже на
    оплачуваній карті — там, де діагностика коштує оренди.
    """
    from gpurunner.htr import box_transport as bt

    data = tmp_path / "train-v42"
    data.mkdir()
    (data / "train.tgz").write_bytes(b"payload")

    # Доставка «вдалась», але на машину нічого не поклала.
    monkeypatch.setattr(bt, "endpoint_of", lambda backend, handle: object())
    monkeypatch.setattr(bt, "push", lambda ep, local, remote, **kw: 0)
    _FAKE_SCP_SINK.clear()

    bk = _TestBackend()
    with pytest.raises(BackendError, match="розпакувалось"):
        bk.submit(_DummyJob(),
                  {"echo": "hi", "dataset": "train-v42", "input_root": str(tmp_path)},
                  gpu="RTX3090")


def test_a_big_text_file_is_not_written_synchronously(_ssh_key: Path) -> None:
    """🔴 Без конвеєра paramiko пише шматками по 32 КБ і чекає кожен.

    Ціна файла тоді — не смуга, а `розмір / 32 КБ × круговий оберт`: `job.py`
    на 48 МБ поїхав до Кореї 25 хвилин, і в лозі це була тиша. Один рядок
    (`set_pipelined`) робить запис потоковим.
    """
    bk = _TestBackend()
    client = bk._ssh(object())
    bk._upload_text(client, "/workspace/gpurunner/job.py", "x" * 100_000)

    assert bk.remote_files["/workspace/gpurunner/job.py"] == b"x" * 100_000
    src = (Path(__file__).resolve().parents[1] / "src" / "gpurunner"
           / "backends" / "vast.py").read_text(encoding="utf-8")
    i = src.index("def _upload_text")
    body = src[i:src.index("def _upload_inputs")]
    # 🪤 Прив'язка саме до ВИКЛИКУ (`fh.`), а не до імені методу: у докстрінгу
    # тієї ж функції воно згадане словами, і перша редакція сторожа ловила
    # власний коментар — мутація «вимкнути конвеєр» лишалась зеленою. Третій
    # такий випадок за день.
    assert "fh.set_pipelined(True)" in body, (
        "синхронний запис повернувся — наступний великий файл заплатить хвилинами")
