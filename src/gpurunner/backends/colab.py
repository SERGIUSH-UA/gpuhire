"""ColabBackend — run jobs on Google Colab, mediated through Google Drive.

Why it looks like this
----------------------
Free/Pro Colab has **no** API for headless submission (Colab Enterprise on Vertex
AI does, but that's paid GCP at Modal-ish prices). The only officially supported,
non-fragile path is:

  1. gpurunner renders a notebook and uploads it to ``MyDrive/gpurunner/runs/<id>/``
  2. **you** open the printed URL and hit *Runtime → Run all* (one click per run)
  3. the notebook writes ``_status.json`` / ``_runner.log`` / ``out/`` back to that
     same Drive folder while it runs
  4. ``status`` / ``logs`` / ``fetch`` poll Drive — no browser involvement

The rendered notebook **emulates the Kaggle filesystem** (``/kaggle/working``,
``/kaggle/input/<slug>``) and then executes ``job.render_remote_code(params)``
verbatim. That is the whole trick: every job that runs on Kaggle runs here with
no job-side changes.

Drive layout::

    MyDrive/gpurunner/
      data/<name>/…              inputs, uploaded once (`gpurunner drive push`)
      runs/<handle_id>/
        notebook.ipynb           the thing you open in Colab
        _status.json             {status, ts, heartbeat, error}
        _runner.log              teed stdout/stderr
        out/                     artifacts (= contents of /kaggle/working)

Limits to keep in mind:
  - the browser tab must stay open (Pro+ adds background execution up to 24 h)
  - Colab disconnects an idle tab after ~90 min — hence the periodic out/ sync
  - ``cancel`` is not available programmatically (stop the runtime in the UI)
"""

from __future__ import annotations

import io
import json
import shlex
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import nbformat
from nbformat.v4 import new_code_cell, new_notebook

from gpurunner.core.backend import AuthError, Backend, BackendError
from gpurunner.core.job import Job
from gpurunner.core.models import BalanceReport, JobHandle, JobStatus, StatusReport

#: Where the notebook mounts Drive, and the resulting prefix of every run folder.
DRIVE_MOUNT = "/content/drive"
DRIVE_PREFIX = f"{DRIVE_MOUNT}/MyDrive/gpurunner"

#: Colab GPU names as they appear in the runtime picker / notebook metadata.
_GPU_TO_COLAB: dict[str, str | None] = {
    "none": None,
    "T4": "T4",
    "L4": "L4",
    "A100": "A100",
}

#: A run whose heartbeat is older than this is presumed disconnected.
_HEARTBEAT_STALE = timedelta(minutes=10)
#: Grace period after submit during which "no _status.json yet" means "not started".
_START_GRACE = timedelta(minutes=10)

_FOLDER_MIME = "application/vnd.google-apps.folder"


class ColabBackend(Backend):
    """Backend that runs jobs as Google Colab notebooks staged on Drive."""

    name: ClassVar[str] = "colab"
    gpu_choices: ClassVar[tuple[str, ...]] = tuple(_GPU_TO_COLAB.keys())
    default_gpu: ClassVar[str] = "T4"

    def __init__(self) -> None:
        self._svc: Any | None = None
        self._root_id: str | None = None

    # ---- public API -------------------------------------------------------

    def check_auth(self) -> None:
        from gpurunner.auth import google as google_auth

        google_auth.discover_credentials()
        self._service()  # forces a token load/refresh
        self._root()  # forces a Drive round-trip + creates MyDrive/gpurunner

    def submit(self, job: Job, params: dict[str, Any], *, gpu: str) -> JobHandle:
        if gpu not in self.gpu_choices:
            raise ValueError(
                f"Unsupported gpu '{gpu}' for colab backend. Choose from: {self.gpu_choices}"
            )
        if self.name not in job.supported_backends:
            raise BackendError(f"Job {job.name!r} does not declare 'colab' in supported_backends")

        normalized = job.validate_params(params)
        inputs = job.colab_input_dirs(normalized)

        # Fail here, locally, rather than 3 minutes into a runtime that can't find data.
        data_root = self._folder(self._root(), "data")
        for slug, drive_name in inputs.items():
            if self._child(data_root, drive_name) is None:
                raise BackendError(
                    f"input {slug!r} expects MyDrive/gpurunner/data/{drive_name} — not found. "
                    f"Upload it first: gpurunner drive push <folder> --name {drive_name}"
                )

        handle = JobHandle(
            backend=self.name,
            remote_id="<pending>",
            job_name=job.name,
            params=normalized,
            gpu=gpu,
        )

        runs_root = self._folder(self._root(), "runs")
        run_folder = self._folder(runs_root, handle.id)

        notebook = self._build_notebook(job, normalized, gpu=gpu, handle_id=handle.id, inputs=inputs)
        buf = io.BytesIO(nbformat.writes(notebook).encode("utf-8"))
        file_id = self._upload_stream(
            buf,
            name="notebook.ipynb",
            parent=run_folder,
            # NOT 'application/vnd.google.colaboratory' — that's a Google-native type
            # you cannot create with an upload. A plain .ipynb opens in Colab fine
            # via colab.research.google.com/drive/<id>.
            mime="application/x-ipynb+json",
        )

        handle.remote_id = file_id
        # `volume_name` is the ABC's generic "backend-managed durable storage" slot;
        # for Colab it holds the Drive folder id of this run.
        handle.volume_name = run_folder
        handle.status = JobStatus.QUEUED
        handle.updated_at = datetime.now(tz=UTC)
        return handle

    def notebook_url(self, handle: JobHandle) -> str:
        return f"https://colab.research.google.com/drive/{handle.remote_id}"

    def status(self, handle: JobHandle) -> StatusReport:
        raw = self._read_status(handle)
        if raw is None:
            age = datetime.now(tz=UTC) - handle.created_at
            if age < _START_GRACE:
                return StatusReport(
                    status=JobStatus.QUEUED,
                    message=f"not started yet — open {self.notebook_url(handle)} → Runtime → Run all",
                )
            return StatusReport(
                status=JobStatus.QUEUED,
                message=(
                    f"still no _status.json after {int(age.total_seconds() // 60)} min. "
                    f"Did you run the notebook? {self.notebook_url(handle)}"
                ),
            )

        state = str(raw.get("status", "")).lower()
        error = raw.get("error")
        if state == "completed":
            return StatusReport(status=JobStatus.COMPLETED, message=raw.get("message"))
        if state == "failed":
            return StatusReport(status=JobStatus.FAILED, message=raw.get("message"), error=error)
        if state in ("running", "starting"):
            beat = _parse_ts(raw.get("heartbeat") or raw.get("ts"))
            if beat is not None and datetime.now(tz=UTC) - beat > _HEARTBEAT_STALE:
                stale_min = int((datetime.now(tz=UTC) - beat).total_seconds() // 60)
                return StatusReport(
                    status=JobStatus.UNKNOWN,
                    message=(
                        f"heartbeat stale for {stale_min} min — the Colab tab was probably "
                        f"disconnected. Re-open {self.notebook_url(handle)} and Run all "
                        f"(partial output already synced is fetchable)."
                    ),
                )
            return StatusReport(status=JobStatus.RUNNING, message=raw.get("message"))
        return StatusReport(status=JobStatus.UNKNOWN, message=f"unrecognized status {state!r}")

    def fetch_outputs(self, handle: JobHandle, out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        folder = self._run_folder(handle)

        written: list[Path] = []
        out_folder = self._child(folder, "out")
        if out_folder is not None:
            written.extend(self._download_tree(out_folder, out_dir))

        # The log and the status file are small and always worth having locally.
        for name in ("_runner.log", "_status.json"):
            fid = self._child(folder, name)
            if fid is None:
                continue
            target = out_dir / name
            self._download_file(fid, target)
            written.append(target)
        return written

    def balance(self) -> BalanceReport:
        # Colab compute units live only in the notebook UI ("Compute units" panel);
        # there is no API for them — the same reason submission needs a click.
        #
        # The wording must not assume Pro: compute units are a Pro / pay-as-you-go
        # concept and **do not exist on the free tier at all**, where access is
        # best-effort with limits Google publishes nowhere. Saying "≈100 units/міс"
        # to a free account credits it with roughly 57 T4-hours it does not have.
        return BalanceReport(
            backend=self.name,
            unit="",
            detail="залишок Colab через API не віддає взагалі · безкоштовний тариф "
                   "compute units не нараховує (і ліміту не публікує), на Pro — ≈100 "
                   "units/міс (~57 год T4)",
            url="https://colab.research.google.com/signup",
        )

    def cancel(self, handle: JobHandle) -> None:
        raise BackendError(
            "Colab has no programmatic cancel — stop the runtime in the browser tab "
            f"(Runtime → Interrupt/Disconnect): {self.notebook_url(handle)}"
        )

    def logs(self, handle: JobHandle) -> list[str]:
        fid = self._child(self._run_folder(handle), "_runner.log")
        if fid is None:
            return ["(no _runner.log on Drive yet — the notebook hasn't started)"]
        buf = io.BytesIO()
        self._download_to(fid, buf)
        return buf.getvalue().decode("utf-8", errors="replace").splitlines()

    def wait_for_log_line(
        self,
        handle: JobHandle,
        needle: str,
        *,
        timeout: int = 1800,
        poll: int = 60,
    ) -> str | None:
        """Poll the Drive-synced log until a line contains ``needle``. None on timeout.

        Mirrors ``KaggleBackend.wait_for_log_line`` — used by ``run --expect-log`` to
        verify the runtime really picked up the dataset version you just uploaded.
        """
        start = time.monotonic()
        while True:
            try:
                for line in self.logs(handle):
                    if needle in line:
                        return line
            except BackendError:
                pass
            if time.monotonic() - start > timeout:
                return None
            time.sleep(poll)

    def wait(
        self,
        handle: JobHandle,
        *,
        poll_interval: int = 30,
        timeout: int | None = None,
    ) -> StatusReport:
        """Poll until status is terminal. Returns the final StatusReport."""
        terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        start = time.monotonic()
        while True:
            report = self.status(handle)
            if report.status in terminal:
                return report
            if timeout is not None and time.monotonic() - start > timeout:
                return report
            time.sleep(poll_interval)

    # ---- input staging ("datasets" on Drive) -------------------------------

    def drive_push(self, folder: Path, *, name: str) -> list[dict[str, Any]]:
        """Upload every file of ``folder`` into ``MyDrive/gpurunner/data/<name>/``.

        Existing files with the same name are replaced. Each upload is verified
        against the md5 Drive computed server-side — Drive has no Kaggle-style
        version race, but a truncated upload must not pass silently.
        """
        folder = Path(folder).resolve()
        files = [p for p in folder.iterdir() if p.is_file()]
        if not files:
            raise BackendError(f"no files to upload in {folder}")

        data_root = self._folder(self._root(), "data")
        target = self._folder(data_root, name)

        pushed: list[dict[str, Any]] = []
        for p in files:
            existing = self._child(target, p.name)
            if existing is not None:
                self._delete(existing)
            fid = self._upload_path(p, parent=target)
            remote_md5 = self._file_md5(fid)
            local_md5 = _md5_of(p)
            if remote_md5 and remote_md5 != local_md5:
                self._delete(fid)
                raise BackendError(
                    f"{p.name}: md5 mismatch after upload (local {local_md5}, "
                    f"Drive {remote_md5}) — upload truncated, file removed. Retry."
                )
            pushed.append({"name": p.name, "size": p.stat().st_size, "id": fid, "md5": local_md5})
        return pushed

    def drive_files(self, name: str) -> list[dict[str, Any]]:
        """List files in ``MyDrive/gpurunner/data/<name>/`` as ``[{name, size, md5}]``."""
        data_root = self._folder(self._root(), "data")
        target = self._child(data_root, name)
        if target is None:
            raise BackendError(f"MyDrive/gpurunner/data/{name} does not exist")
        out = []
        for f in self._list(target):
            size = f.get("size")
            out.append(
                {
                    "name": f["name"],
                    "size": int(size) if size is not None else None,
                    "md5": f.get("md5Checksum"),
                }
            )
        return out

    # ---- notebook rendering -----------------------------------------------

    def _build_notebook(
        self,
        job: Job,
        params: dict[str, Any],
        *,
        gpu: str,
        handle_id: str,
        inputs: dict[str, str],
    ) -> nbformat.NotebookNode:
        nb = new_notebook()
        nb.metadata.setdefault("kernelspec", {})
        nb.metadata["kernelspec"].update(
            {"display_name": "Python 3", "language": "python", "name": "python3"}
        )
        nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
        # Colab reads these back when opening a notebook from Drive. If it ever
        # stops honouring them the GPU-guard cell below fails loudly rather than
        # letting a 12-hour train crawl on CPU.
        colab_gpu = _GPU_TO_COLAB[gpu]
        nb.metadata["accelerator"] = "GPU" if colab_gpu else "None"
        nb.metadata["colab"] = {"provenance": [], "name": f"{job.name}-{handle_id[:8]}"}
        if colab_gpu:
            nb.metadata["colab"]["gpuType"] = colab_gpu

        run_dir = f"{DRIVE_PREFIX}/runs/{handle_id}"

        nb.cells.append(new_code_cell(_CELL_MOUNT.format(run_dir=run_dir)))
        nb.cells.append(new_code_cell(_CELL_GPU_GUARD.format(want_gpu=repr(colab_gpu))))
        nb.cells.append(
            new_code_cell(
                _CELL_SHIM.format(
                    inputs_json=json.dumps(inputs, ensure_ascii=False),
                    data_root=f"{DRIVE_PREFIX}/data",
                )
            )
        )

        reqs = job.requirements()
        if reqs:
            req_text = " ".join(shlex.quote(r) for r in reqs)
            nb.cells.append(
                new_code_cell(
                    "print('[gpurunner] setup start', flush=True)\n"
                    f"!pip install {req_text}\n"
                    "print('[gpurunner] setup done', flush=True)"
                )
            )

        body = job.render_remote_code(params)
        nb.cells.append(new_code_cell(_CELL_RUN.format(runner_src=json.dumps(body, ensure_ascii=False))))
        return nb

    # ---- Drive plumbing ----------------------------------------------------

    def _service(self) -> Any:
        if self._svc is None:
            from gpurunner.auth import google as google_auth

            self._svc = google_auth.build_drive()
        return self._svc

    def _root(self) -> str:
        if self._root_id is None:
            from gpurunner.auth.google import DRIVE_ROOT_FOLDER

            self._root_id = self._folder("root", DRIVE_ROOT_FOLDER)
        assert self._root_id is not None
        return self._root_id

    def _run_folder(self, handle: JobHandle) -> str:
        if handle.volume_name:
            return handle.volume_name
        # Handles created before volume_name was recorded, or hand-edited manifests.
        runs_root = self._folder(self._root(), "runs")
        fid = self._child(runs_root, handle.id)
        if fid is None:
            raise BackendError(f"no Drive folder for handle {handle.id} under gpurunner/runs/")
        return fid

    def _folder(self, parent: str, name: str) -> str:
        """Find-or-create a folder. Same as ``_child(..., create=True)``, non-optional."""
        fid = self._child(parent, name, create=True)
        assert fid is not None  # create=True never returns None
        return fid

    def _child(self, parent: str, name: str, *, create: bool = False) -> str | None:
        """Find a direct child by exact name. Optionally create it as a folder."""
        q = (
            f"name = {_q(name)} and {_q(parent)} in parents and trashed = false"
        )
        try:
            resp = (
                self._service()
                .files()
                .list(q=q, fields="files(id,name,mimeType)", pageSize=10)
                .execute()
            )
        except Exception as e:
            raise self._wrap(e, f"Drive lookup of {name!r} failed") from e
        files = resp.get("files") or []
        if files:
            return str(files[0]["id"])
        if not create:
            return None
        try:
            created = (
                self._service()
                .files()
                .create(
                    body={"name": name, "mimeType": _FOLDER_MIME, "parents": [parent]},
                    fields="id",
                )
                .execute()
            )
        except Exception as e:
            raise self._wrap(e, f"Drive folder create {name!r} failed") from e
        return str(created["id"])

    def _list(self, parent: str) -> list[dict[str, Any]]:
        svc = self._service()
        out: list[dict[str, Any]] = []
        token: str | None = None
        while True:
            try:
                resp = (
                    svc.files()
                    .list(
                        q=f"{_q(parent)} in parents and trashed = false",
                        fields="nextPageToken, files(id,name,mimeType,size,md5Checksum)",
                        pageSize=1000,
                        pageToken=token,
                    )
                    .execute()
                )
            except Exception as e:
                raise self._wrap(e, "Drive listing failed") from e
            out.extend(resp.get("files") or [])
            token = resp.get("nextPageToken")
            if not token:
                return out

    def _delete(self, file_id: str) -> None:
        try:
            self._service().files().delete(fileId=file_id).execute()
        except Exception as e:
            raise self._wrap(e, "Drive delete failed") from e

    def _file_md5(self, file_id: str) -> str | None:
        try:
            meta = self._service().files().get(fileId=file_id, fields="md5Checksum").execute()
        except Exception as e:
            raise self._wrap(e, "Drive metadata read failed") from e
        return meta.get("md5Checksum")

    def _upload_stream(self, stream: io.BytesIO, *, name: str, parent: str, mime: str) -> str:
        from googleapiclient.http import MediaIoBaseUpload

        media = MediaIoBaseUpload(stream, mimetype=mime, resumable=False)
        try:
            created = (
                self._service()
                .files()
                .create(body={"name": name, "parents": [parent]}, media_body=media, fields="id")
                .execute()
            )
        except Exception as e:
            raise self._wrap(e, f"Drive upload of {name!r} failed") from e
        return str(created["id"])

    def _upload_path(self, path: Path, *, parent: str) -> str:
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(path), resumable=True, chunksize=8 * 1024 * 1024)
        try:
            request = (
                self._service()
                .files()
                .create(body={"name": path.name, "parents": [parent]}, media_body=media, fields="id")
            )
            response = None
            while response is None:
                _, response = request.next_chunk()
        except Exception as e:
            raise self._wrap(e, f"Drive upload of {path.name} failed") from e
        return str(response["id"])

    def _download_tree(self, folder: str, out_dir: Path) -> list[Path]:
        written: list[Path] = []
        for entry in self._list(folder):
            target = out_dir / entry["name"]
            if entry.get("mimeType") == _FOLDER_MIME:
                target.mkdir(parents=True, exist_ok=True)
                written.extend(self._download_tree(entry["id"], target))
                continue
            self._download_file(entry["id"], target)
            written.append(target)
        return written

    def _download_file(self, file_id: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            self._download_to(file_id, fh)

    def _download_to(self, file_id: str, sink: Any) -> None:
        from googleapiclient.http import MediaIoBaseDownload

        try:
            request = self._service().files().get_media(fileId=file_id)
            downloader = MediaIoBaseDownload(sink, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        except Exception as e:
            raise self._wrap(e, f"Drive download of {file_id} failed") from e

    def _read_status(self, handle: JobHandle) -> dict[str, Any] | None:
        fid = self._child(self._run_folder(handle), "_status.json")
        if fid is None:
            return None
        buf = io.BytesIO()
        self._download_to(fid, buf)
        try:
            data = json.loads(buf.getvalue().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # A half-written status file — treat as "nothing new" rather than crashing.
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _wrap(e: Exception, msg: str) -> Exception:
        text = str(e)
        if "invalid_grant" in text or "401" in text:
            return AuthError(
                f"{msg}: {e}\nToken expired or revoked — run: gpurunner auth google --login"
            )
        return BackendError(f"{msg}: {e}")


# ---- helpers ---------------------------------------------------------------


def _q(value: str) -> str:
    """Quote a value for a Drive query string."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _md5_of(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# ---- notebook cell templates ----------------------------------------------
#
# These run INSIDE Colab. Keep them dependency-free (stdlib + google.colab) and
# `.format()`-safe: literal braces must be doubled.

_CELL_MOUNT = '''\
# --- gpurunner: mount Drive + run-folder bookkeeping -------------------------
import json, os, sys, threading, time, shutil, traceback
from datetime import datetime, timezone
from pathlib import Path

from google.colab import drive
drive.mount("/content/drive")

RUN_DIR = Path({run_dir!r})
RUN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = RUN_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WORKING = Path("/kaggle/working")
LOG_PATH = RUN_DIR / "_runner.log"


def _now():
    return datetime.now(tz=timezone.utc).isoformat()


_STATUS_LOCK = threading.Lock()
_FINALIZED = []


def write_status(status, **extra):
    """Atomically publish state for `gpurunner status` to poll.

    Terminal states latch: the heartbeat thread must never be able to overwrite
    a 'completed' with a stale 'running' if it wakes up during the finalizer.
    """
    with _STATUS_LOCK:
        if _FINALIZED:
            return
        if status in ("completed", "failed"):
            _FINALIZED.append(status)
        payload = {{"status": status, "ts": _now(), "heartbeat": _now()}}
        payload.update(extra)
        tmp = RUN_DIR / "_status.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, RUN_DIR / "_status.json")


write_status("starting", message="notebook started, setting up")
print("[gpurunner] run dir:", RUN_DIR, flush=True)
'''

_CELL_GPU_GUARD = '''\
# --- gpurunner: GPU guard ----------------------------------------------------
WANT_GPU = {want_gpu}

_smi = os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").read().strip()
print("[gpurunner] nvidia-smi:", _smi or "(no GPU)", flush=True)

if WANT_GPU and not _smi:
    write_status(
        "failed",
        error="no GPU attached",
        message=(
            "This run was submitted with --gpu %s but the runtime has no GPU. "
            "Runtime -> Change runtime type -> %s, then Run all again."
        ) % (WANT_GPU, WANT_GPU),
    )
    raise SystemExit(
        "[gpurunner] no GPU in this runtime — set Runtime > Change runtime type > %s and re-run. "
        "Refusing to continue on CPU." % WANT_GPU
    )
if WANT_GPU and WANT_GPU.lower() not in _smi.lower():
    print(
        "[gpurunner] WARNING: asked for %s, runtime reports %r — continuing anyway."
        % (WANT_GPU, _smi),
        flush=True,
    )
'''

_CELL_SHIM = '''\
# --- gpurunner: emulate the Kaggle filesystem --------------------------------
# Jobs render code that reads /kaggle/input and writes /kaggle/working. We are
# root here, so we just create those paths: inputs are copied off the Drive FUSE
# mount (reading gigabytes through it is slow), outputs live on local disk and
# get synced back to Drive periodically.
INPUTS = json.loads({inputs_json!r})
DATA_ROOT = Path({data_root!r})

WORKING.mkdir(parents=True, exist_ok=True)
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_INPUT.mkdir(parents=True, exist_ok=True)

for _slug, _drive_name in INPUTS.items():
    _src = DATA_ROOT / _drive_name
    if not _src.exists():
        raise RuntimeError("input %r missing on Drive: %s" % (_slug, _src))
    _dst = KAGGLE_INPUT / _slug
    _dst.mkdir(parents=True, exist_ok=True)
    for _f in sorted(_src.iterdir()):
        if not _f.is_file():
            continue
        _target = _dst / _f.name
        print("[gpurunner] staging %s (%.1f MB) ..." % (_f.name, _f.stat().st_size / 1e6), flush=True)
        shutil.copy2(_f, _target)
print("[gpurunner] /kaggle/input ready:", [str(p) for p in KAGGLE_INPUT.rglob("*")][:20], flush=True)
'''

_CELL_RUN = '''\
# --- gpurunner: run the job --------------------------------------------------
# The job body is executed here (not pasted inline) so that a failure still runs
# the finalizer below: status + log + partial outputs must reach Drive even when
# the job raises or the tab is about to be disconnected.
RUNNER_SRC = {runner_src}
SYNC_EVERY_SEC = 600

_stop = threading.Event()


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _sync_outputs():
    """Copy /kaggle/working -> Drive out/. Cheap enough to repeat; skips same-size files."""
    for src in WORKING.rglob("*"):
        if not src.is_file():
            continue
        dst = OUT_DIR / src.relative_to(WORKING)
        try:
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except Exception as e:
            print("[gpurunner] sync skipped %s: %s" % (src, e), flush=True)


def _heartbeat():
    """Keep _status.json fresh and flush partial results while the job runs."""
    last_sync = time.time()
    while not _stop.wait(60):
        try:
            write_status("running", message="in progress")
            if time.time() - last_sync >= SYNC_EVERY_SEC:
                _sync_outputs()
                last_sync = time.time()
        except Exception:
            pass


_beat = threading.Thread(target=_heartbeat, daemon=True)
_beat.start()

write_status("running", message="job started")
_real_out, _real_err = sys.stdout, sys.stderr
_summary, _err = None, None
with open(LOG_PATH, "w", encoding="utf-8", buffering=1) as _log:
    sys.stdout = _Tee(_real_out, _log)
    sys.stderr = _Tee(_real_err, _log)
    try:
        _scope = {{"__name__": "__main__"}}
        exec(compile(RUNNER_SRC, "<gpurunner_job>", "exec"), _scope)
    except BaseException as e:
        traceback.print_exc(file=sys.stderr)
        _err = repr(e)
    finally:
        sys.stdout, sys.stderr = _real_out, _real_err
        _stop.set()

_beat.join(timeout=90)  # let the heartbeat finish its current pass before the final sync
_sync_outputs()
_n = sum(1 for p in OUT_DIR.rglob("*") if p.is_file())
if _err is None:
    write_status("completed", message="%d files in out/" % _n)
    print("[gpurunner] DONE — %d files synced to %s" % (_n, OUT_DIR), flush=True)
else:
    write_status("failed", error=_err, message="%d files in out/ (partial)" % _n)
    print("[gpurunner] FAILED: %s (see _runner.log)" % _err, flush=True)
    raise SystemExit(1)
'''
