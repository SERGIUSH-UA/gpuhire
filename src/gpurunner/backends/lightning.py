"""LightningBackend — headless batch jobs on Lightning AI (lightning.ai).

The only backend besides Modal with a real fire-and-forget API *and* a free tier
(~15 credits/month ≈ 22 h of T4). No browser click like Colab, no rented box that
keeps billing like Vast: a Job runs, costs what it ran, and stops.

Shape of a run::

    upload inputs  →  Teamspace drive: gpurunner/data/<slug>/
    Job.run(studio=…, machine=…, command=<base64 bootstrap>)
    the job:  drive → /kaggle/input/<slug>,  writes /kaggle/working,
              syncs  /kaggle/working → drive: gpurunner/runs/<handle>/out/
              plus   _status.json + _runner.log next to it
    status/logs/fetch  ←  download those files from the drive

Everything moves over the **Teamspace API in both directions** — that is not a
stylistic choice, it is the only thing that works. Verified on a live free-tier
account (2026-07-21):

- every ``/teamspace/...`` mount inside a job is **read-only** (``/teamspace/uploads``,
  ``/teamspace/jobs/<name>/artifacts``, the studio home) — a job cannot write there;
- the studio home a job sees is an **ephemeral copy**: files written to it vanish
  with the job and never reach the drive;
- files uploaded from the laptop are **not visible** in the job's mount either;
- but ``LIGHTNING_API_KEY``/``LIGHTNING_USER_ID`` are injected into every job and
  ``lightning_sdk`` is preinstalled, so the job can call the same API we do.

Hence: we upload inputs via the API, the job downloads them via the API, and the
job uploads status/log/outputs back via the API.

We also do not use ``job.artifact_path`` (``None`` for image jobs, read-only for
studio jobs). ``job.logs`` *raises* while a job runs — but that is the SDK wrapper
refusing, not the platform: underneath, ``jobs_service_get_job_logs`` hands out a
WebSocket follow URL that streams the live tail. ``logs()`` falls back to it, so a
multi-hour run is no longer a black box until it terminates (verified live on a
kraken training run, 2026-07-26).

As with the Colab and Vast backends, the remote side emulates the Kaggle
filesystem (``/kaggle/input/<slug>``, ``/kaggle/working``) and then executes
``job.render_remote_code(params)`` verbatim — jobs need no Lightning-specific code.

Backend-only params (read from RAW params, before ``validate_params`` drops
unknown keys): ``studio`` ``teamspace`` ``inputs`` ``input_root`` ``max_hours``
``interruptible``.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
import tempfile
import time
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

from gpurunner.core.backend import Backend, BackendError
from gpurunner.core.inputs import resolve_inputs
from gpurunner.core.job import Job
from gpurunner.core.models import BalanceReport, JobHandle, JobStatus, StatusReport

#: Root of our tree inside the teamspace drive (relative paths, as the SDK wants).
#: 🔴 Must live under ``uploads/`` (2026-07-29). The drive root is now a fixed set of
#: system trees — ``studios``, ``uploads``, ``artifacts``, ``lightning_storage``, the
#: connection folders — and nothing else may be created there. Writing to a top-level
#: ``gpurunner/…`` key still returned HTTP success while storing nothing: the upload
#: verification then failed with "reported success but N file(s) are not on the drive",
#: and a plain ``list_files`` of the drive root showed no ``gpurunner`` tree at all.
#: ``lightning_storage/…`` is likewise refused (404 on the upload-URL request).
DRIVE_ROOT = "uploads/gpurunner"

#: gpurunner GPU name → ``lightning_sdk.Machine`` attribute.
#: (verified against lightning-sdk 2026.7.21 — there is no A10G any more; the
#: mid-tier cards are L4 / L40S / RTXP_6000.)
_GPU_TO_MACHINE: dict[str, str] = {
    "none": "CPU",
    "T4": "T4",
    "T4x2": "T4_X_2",
    "L4": "L4",
    "L40S": "L40S",
    "A100": "A100",
    "H100": "H100",
}

DEFAULT_MAX_HOURS = 12
_HEARTBEAT_STALE = timedelta(minutes=10)


class LightningBackend(Backend):
    """Run jobs as Lightning AI batch jobs against a Studio environment."""

    name: ClassVar[str] = "lightning"
    gpu_choices: ClassVar[tuple[str, ...]] = tuple(_GPU_TO_MACHINE.keys())
    default_gpu: ClassVar[str] = "T4"

    def __init__(self) -> None:
        self._ts: Any | None = None
        self._ts_name: str | None = None

    # ---- public API -------------------------------------------------------

    def check_auth(self) -> None:
        from gpurunner.auth import lightning as lit_auth

        lit_auth.discover_credentials()
        lit_auth.import_sdk()
        self._teamspace()

    def submit(self, job: Job, params: dict[str, Any], *, gpu: str) -> JobHandle:
        from gpurunner.auth import lightning as lit_auth

        if gpu not in self.gpu_choices:
            raise ValueError(
                f"Unsupported gpu '{gpu}' for lightning backend. Choose from: {self.gpu_choices}"
            )
        if self.name not in job.supported_backends:
            raise BackendError(
                f"Job {job.name!r} does not declare 'lightning' in supported_backends"
            )

        opts = _backend_opts(params)
        if not opts["studio"]:
            raise BackendError(
                "lightning needs a Studio to borrow the environment from: "
                "-p studio=<name>. Create one once in the web UI (any Studio works; "
                "its installed packages become the job's environment)."
            )
        inputs = _resolve_inputs(job, params, opts)

        normalized = job.validate_params(params)
        body = job.render_remote_code(normalized)

        handle = JobHandle(
            backend=self.name,
            remote_id="<pending>",
            job_name=job.name,
            params=normalized,
            gpu=gpu,
        )
        run_dir = f"{DRIVE_ROOT}/runs/{handle.id}"
        # Lightning нормалізує "_" → "-" в іменах job'ів; робимо це самі,
        # інакше remote_id не збігається і status/cancel/logs не знаходять job
        job_name = f"gpurunner-{job.name}-{handle.id[:8]}"[:60].replace("_", "-")

        ts = self._teamspace(opts["teamspace"])
        for slug, local in inputs.items():
            self._upload_dir(local, f"{DRIVE_ROOT}/data/{slug}")

        command = self._render_command(
            body,
            run_dir=run_dir,
            inputs=inputs,
            teamspace_name=str(getattr(ts, "name", "")),
            username=self._owner_name(ts),
        )
        sdk = lit_auth.import_sdk()
        machine = getattr(sdk.Machine, _GPU_TO_MACHINE[gpu], None)  # type: ignore[attr-defined]
        if machine is None:
            raise BackendError(
                f"lightning-sdk has no Machine.{_GPU_TO_MACHINE[gpu]} — SDK version mismatch"
            )
        try:
            sdk.Job.run(  # type: ignore[attr-defined]
                name=job_name,
                machine=machine,
                command=command,
                studio=opts["studio"],
                teamspace=ts,
                max_runtime=int(opts["max_hours"] * 3600),
                interruptible=bool(opts["interruptible"]),
            )
        except Exception as e:
            raise BackendError(f"Lightning Job.run failed: {e}") from e

        handle.remote_id = job_name
        # ABC's generic durable-storage slot: for Lightning it's the drive folder
        # that carries status, logs and outputs for this run.
        handle.volume_name = run_dir
        handle.status = JobStatus.QUEUED
        handle.updated_at = datetime.now(tz=UTC)
        return handle

    def status(self, handle: JobHandle) -> StatusReport:
        raw = self._read_status(handle)
        if raw is None:
            sdk_state, cost = self._sdk_state(handle)
            if sdk_state in ("failed", "stopped"):
                return StatusReport(
                    status=JobStatus.FAILED,
                    error=f"job {sdk_state} before writing any status{cost}",
                )
            if sdk_state == "completed":
                return StatusReport(
                    status=JobStatus.FAILED,
                    error=f"job finished without writing _status.json — see logs{cost}",
                )
            return StatusReport(
                status=JobStatus.QUEUED, message=f"job {sdk_state or 'pending'}{cost}"
            )

        _, cost = self._sdk_state(handle)
        state = str(raw.get("status", "")).lower()
        if state == "completed":
            return StatusReport(status=JobStatus.COMPLETED, message=f"{raw.get('message') or 'done'}{cost}")
        if state == "failed":
            return StatusReport(
                status=JobStatus.FAILED, message=f"{raw.get('message') or ''}{cost}", error=raw.get("error")
            )
        if state in ("running", "starting"):
            beat = _parse_ts(raw.get("heartbeat") or raw.get("ts"))
            if beat is not None and datetime.now(tz=UTC) - beat > _HEARTBEAT_STALE:
                return StatusReport(
                    status=JobStatus.UNKNOWN,
                    message=f"heartbeat stale — job may have been preempted{cost}",
                )
            return StatusReport(status=JobStatus.RUNNING, message=f"{raw.get('message') or ''}{cost}")
        return StatusReport(status=JobStatus.UNKNOWN, message=f"unrecognized job status {state!r}")

    def fetch_outputs(self, handle: JobHandle, out_dir: Path) -> list[Path]:
        """Download the run folder file-by-file.

        ``Teamspace.download_folder`` is **broken**: it creates the right file names
        locally but every one of them is 0 bytes (verified on a live account). Only
        ``download_file`` returns real content, so we enumerate the remote tree and
        pull each blob individually.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = self._run_dir(handle)

        written: list[Path] = []
        for rel, _size in self._list_remote(run_dir):
            # `out/foo.txt` → <out_dir>/foo.txt; sidecars land next to them.
            local_rel = rel[len("out/"):] if rel.startswith("out/") else rel
            blob = self._download_bytes(f"{run_dir}/{rel}")
            if blob is None:
                continue
            target = out_dir / local_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            written.append(target)
        return written

    def _upload_dir(self, local: Path, remote_prefix: str) -> int:
        """Upload a directory file-by-file, then verify it actually landed.

        ``Teamspace.upload_folder`` is a **silent no-op** (verified live: the target
        prefix stayed empty afterwards, and the job found no inputs). Only
        ``upload_file`` transfers anything, so we walk the tree ourselves and then
        re-list the prefix — an input that never arrived must fail here, on the
        laptop, not three minutes into a paid GPU job.
        """
        import contextlib
        import io

        ts = self._teamspace()
        api = getattr(ts, "_teamspace_api", None)
        if api is None or not hasattr(api, "upload_file"):
            raise BackendError("lightning-sdk exposes no teamspace upload API — version mismatch")
        files = [p for p in sorted(local.rglob("*")) if p.is_file()]
        if not files:
            raise BackendError(f"input dir {local} has no files")
        noise = io.StringIO()
        for p in files:
            rel = p.relative_to(local).as_posix()
            try:
                with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
                    # NOT ts.upload_file(): it runs the remote path through
                    # os.path.normpath, which on Windows turns "a/b/c.txt" into
                    # "a\b\c.txt" — the blob lands under a mangled key, the upload
                    # reports success, and the job then finds no inputs. The
                    # lower-level API takes the path verbatim.
                    api.upload_file(
                        teamspace_id=ts.id,
                        cloud_account=ts.default_cloud_account,
                        file_path=str(p),
                        remote_path=f"{remote_prefix}/{rel}",
                        progress_bar=False,
                    )
            except Exception as e:
                raise BackendError(f"upload of {p} → {remote_prefix}/{rel} failed: {e}") from e

        landed = {rel for rel, _ in self._list_remote(remote_prefix)}
        missing = {p.relative_to(local).as_posix() for p in files} - landed
        if missing:
            raise BackendError(
                f"upload to {remote_prefix} reported success but "
                f"{len(missing)} file(s) are not on the drive: {sorted(missing)[:5]}"
            )
        return len(files)

    def _list_remote(self, prefix: str) -> list[tuple[str, int]]:
        """``[(path relative to prefix, size)]`` for every blob under ``prefix``.

        Uses the teamspace API's ``list_files`` (recursive). It is reached through
        the SDK's private ``_teamspace_api`` because ``Teamspace`` exposes no public
        listing method.
        """
        ts = self._teamspace()
        api = getattr(ts, "_teamspace_api", None)
        if api is None or not hasattr(api, "list_files"):
            raise BackendError(
                "lightning-sdk exposes no list_files on the teamspace API — "
                "SDK version mismatch; cannot enumerate outputs"
            )
        try:
            entries = api.list_files(teamspace_id=ts.id, path=prefix)
        except Exception as e:
            raise BackendError(f"listing {prefix} failed: {e}") from e
        out: list[tuple[str, int]] = []
        for entry in entries or []:
            if isinstance(entry, dict) and entry.get("type") == "blob" and entry.get("path"):
                out.append((str(entry["path"]), int(entry.get("size") or 0)))
        return out

    def balance(self) -> BalanceReport:
        """Credits left in the teamspace.

        The **project** balance is the one that matters: the free tier's monthly
        credits live there. ``billing_service_get_user_balance`` only knows about
        purchased credit (0.0 on a free account) — reported as ``spent`` context.
        """
        from gpurunner.auth.lightning import import_sdk

        import_sdk()
        from lightning_sdk.lightning_cloud.rest_client import LightningClient

        ts = self._teamspace()
        client = LightningClient(retry=False)
        try:
            project = client.billing_service_get_project_balance(project_id=ts.id).to_dict()
        except Exception as e:
            raise BackendError(f"Lightning balance lookup failed: {e}") from e
        spent = None
        with suppress(Exception):
            spent = float(client.billing_service_get_user_balance().to_dict().get("total_spent") or 0)
        return BalanceReport(
            backend=self.name,
            available=float(project.get("balance") or 0),
            unit="credits",
            spent=spent,
            detail=f"teamspace {getattr(ts, 'name', '?')} · free tier ≈ 15 credits/month (~22 h T4)",
            url="https://lightning.ai/billing",
        )

    def cancel(self, handle: JobHandle) -> None:
        job = self._job(handle)
        if job is None:
            raise BackendError(f"job {handle.remote_id} not found in the teamspace")
        try:
            job.stop()
        except Exception as e:
            raise BackendError(f"Lightning stop failed: {e}") from e

    def logs(self, handle: JobHandle) -> list[str]:
        blob = self._download_bytes(f"{self._run_dir(handle)}/_runner.log")
        if blob is not None:
            return blob.decode("utf-8", errors="replace").splitlines()
        job = self._job(handle)
        if job is None:
            return [f"(job {handle.remote_id} not found)"]
        try:
            return str(job.logs).splitlines()
        except Exception:
            # ``Job.logs`` refuses to serve anything while the job is Running --
            # but that's a limitation of the SDK wrapper, not of the platform.
            # Underneath sits ``jobs_service_get_job_logs``, which hands out a
            # WebSocket follow URL that streams the live tail. Without this a
            # multi-hour training run is a black box until it terminates.
            live = self._live_logs(handle, job)
            if live:
                return live
            return ["(no logs yet)"]

    def _live_logs(
        self, handle: JobHandle, job=None, *, tail: int = 400, seconds: float = 20.0
    ) -> list[str]:
        """Tail of a *running* job via the log WebSocket. Empty list if unavailable."""
        try:
            import websocket  # websocket-client, ships with lightning-sdk
            from lightning_sdk.lightning_cloud.login import Auth
            from websocket import (
                WebSocketConnectionClosedException,
                WebSocketTimeoutException,
            )
        except ImportError:
            return []
        job = job or self._job(handle)
        if job is None:
            return []
        try:
            ts = job.teamspace
            resp = job._job_api._client.jobs_service_get_job_logs(
                project_id=ts.id, id=job._guaranteed_job.id
            )
            url = re.sub(r"lineCounter=0", f"lineCounter=0&tail={tail}",
                         resp.follow_url or "")
            if not url:
                return []
            ws = websocket.create_connection(
                url, header=[f"Authorization: {Auth().authenticate()}"], timeout=8
            )
        except Exception:
            return []

        lines: list[str] = []
        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline and len(lines) < tail * 4:
                try:
                    msg = ws.recv()
                except WebSocketTimeoutException:
                    continue          # тиша в лозі — не помилка, просто чекаємо
                except WebSocketConnectionClosedException:
                    break
                if not msg:
                    continue
                try:
                    payload = json.loads(msg)
                except json.JSONDecodeError:
                    lines.append(str(msg).rstrip())
                    continue
                for entry in payload if isinstance(payload, list) else [payload]:
                    text = entry.get("message") if isinstance(entry, dict) else str(entry)
                    if text and text.strip():
                        lines.append(text.rstrip())
        finally:
            with suppress(Exception):
                ws.close()
        return lines

    def wait_for_log_line(
        self, handle: JobHandle, needle: str, *, timeout: int = 1800, poll: int = 60
    ) -> str | None:
        """Poll the drive-synced log until a line contains ``needle``. None on timeout."""
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

    # ---- remote side ------------------------------------------------------

    def _render_command(
        self,
        body: str,
        *,
        run_dir: str,
        inputs: dict[str, Path],
        teamspace_name: str,
        username: str,
    ) -> str:
        """Shell one-liner-ish bootstrap: decode the wrapper, run it.

        The wrapper travels base64-encoded — the command string is passed through
        ``sh -c`` by Lightning, so embedding arbitrary Python source directly is a
        quoting minefield (same reason as the Vast backend).
        """
        wrapper = (
            _RUNNER_WRAPPER.replace("__RUNNER_SRC__", json.dumps(body, ensure_ascii=False))
            .replace("__RUN_DIR__", json.dumps(run_dir))
            .replace("__INPUTS__", json.dumps(json.dumps({s: s for s in inputs})))
            .replace("__DATA_ROOT__", json.dumps(f"{DRIVE_ROOT}/data"))
            .replace("__TEAMSPACE__", json.dumps(teamspace_name))
            .replace("__USERNAME__", json.dumps(username))
        )
        blob = base64.b64encode(wrapper.encode("utf-8")).decode("ascii")
        return (
            "set -e; mkdir -p /tmp/gpurunner; "
            f"echo {shlex.quote(blob)} | base64 -d > /tmp/gpurunner/job.py; "
            "python3 -u /tmp/gpurunner/job.py"
        )

    # ---- plumbing ---------------------------------------------------------

    def _teamspace(self, name: str | None = None) -> Any:
        from gpurunner.auth import lightning as lit_auth

        if self._ts is None or (name and name != self._ts_name):
            self._ts = lit_auth.resolve_teamspace(name)
            self._ts_name = name
        return self._ts

    @staticmethod
    def _owner_name(ts: Any) -> str:
        """Teamspace owner's username — the job needs it to re-open the teamspace.

        Taken off the resolved Teamspace object; only falls back to an extra API
        round-trip when the SDK doesn't expose it.
        """
        owner = getattr(ts, "owner", None)
        name = getattr(owner, "name", None)
        if name:
            return str(name)
        from gpurunner.auth.lightning import current_username

        return current_username() or ""

    def _run_dir(self, handle: JobHandle) -> str:
        return handle.volume_name or f"{DRIVE_ROOT}/runs/{handle.id}"

    def _job(self, handle: JobHandle) -> Any | None:
        from gpurunner.auth import lightning as lit_auth

        sdk = lit_auth.import_sdk()
        try:
            # "_"→"-": старі handle'и могли зберегти ненормалізоване ім'я
            return sdk.Job(  # type: ignore[attr-defined]
                name=handle.remote_id.replace("_", "-"),
                teamspace=self._teamspace())
        except Exception:
            return None

    def _sdk_state(self, handle: JobHandle) -> tuple[str | None, str]:
        """``(status name lowercased, ' · $x.xx' suffix)``. Never raises."""
        job = self._job(handle)
        if job is None:
            return None, ""
        try:
            state = str(getattr(job.status, "name", job.status)).lower()
        except Exception:
            state = None
        cost = ""
        try:
            total = job.total_cost
            if total:
                cost = f" · ${float(total):.2f}"
        except Exception:
            cost = ""
        return state, cost

    def _download_bytes(self, remote_path: str) -> bytes | None:
        """Fetch a single small file from the drive. None when it isn't there yet.

        ``Teamspace.download_file`` does **not** raise for a missing remote file —
        it happily creates an empty local one (and prints a bogus progress bar).
        So "no bytes" is the real signal for "not there", and the SDK's chatter is
        redirected away to keep `status`/`logs` output clean.
        """
        import contextlib
        import io

        ts = self._teamspace()
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / Path(remote_path).name
            try:
                noise = io.StringIO()
                with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
                    ts.download_file(remote_path, file_path=str(local))
            except Exception:
                return None
            if not local.exists() or local.stat().st_size == 0:
                return None
            return local.read_bytes()

    def _read_status(self, handle: JobHandle) -> dict[str, Any] | None:
        blob = self._download_bytes(f"{self._run_dir(handle)}/_status.json")
        if blob is None:
            return None
        try:
            data = json.loads(blob.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None  # caught mid-write
        return data if isinstance(data, dict) else None


# ---- helpers ---------------------------------------------------------------


def _backend_opts(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "studio": (str(params["studio"]).strip() if params.get("studio") else None),
        "teamspace": (str(params["teamspace"]).strip() if params.get("teamspace") else None),
        "inputs": params.get("inputs") or {},
        "input_root": params.get("input_root"),
        "max_hours": float(params.get("max_hours") or DEFAULT_MAX_HOURS),
        "interruptible": bool(params.get("interruptible", False)),
    }


def _resolve_inputs(job: Job, params: dict[str, Any], opts: dict[str, Any]) -> dict[str, Path]:
    """``{slug: local dir}`` to upload. Same contract as every Kaggle-FS backend."""
    return resolve_inputs(job, params, inputs=opts["inputs"], input_root=opts["input_root"])


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


#: Runs inside the Lightning job. Same ``_status.json`` contract (and terminal
#: latch) as the Colab and Vast backends, so `status` reads all three alike.
_RUNNER_WRAPPER = '''\
import json, os, shutil, sys, threading, time, traceback
from datetime import datetime, timezone
from pathlib import Path

from lightning_sdk import Teamspace

RUN_DIR = __RUN_DIR__          # drive-relative, e.g. "gpurunner/runs/<handle>"
DATA_ROOT = __DATA_ROOT__      # drive-relative, e.g. "gpurunner/data"
INPUTS = json.loads(__INPUTS__)

# Every /teamspace mount is read-only in a job and the studio home is ephemeral,
# so the drive is reachable ONLY through the API. Credentials are injected by
# Lightning, so this needs no secrets from us.
TS = Teamspace(name=__TEAMSPACE__, user=__USERNAME__)

LOCAL = Path("/tmp/gpurunner")
LOCAL.mkdir(parents=True, exist_ok=True)
STATUS_PATH = LOCAL / "_status.json"
LOG_PATH = LOCAL / "_runner.log"
WORKING = None  # set below, once we know where we are allowed to write

_LOCK = threading.Lock()
_FINALIZED = []
_SYNCED = {}  # rel path -> size already pushed, so periodic syncs stay incremental


def _now():
    return datetime.now(tz=timezone.utc).isoformat()


def _push(local_path, remote_name):
    """Upload one small sidecar file to the run folder on the drive."""
    try:
        TS.upload_file(str(local_path), remote_path=RUN_DIR + "/" + remote_name,
                       progress_bar=False)
    except Exception as e:
        print("[gpurunner] upload of %s failed: %s" % (remote_name, e), flush=True)


def _pull_dir(remote_prefix, target):
    """Download a drive folder file-by-file.

    upload_folder/download_folder are broken in lightning-sdk (silent no-op /
    zero-byte files), so both directions go one file at a time.
    """
    entries = TS._teamspace_api.list_files(teamspace_id=TS.id, path=remote_prefix)
    blobs = [e for e in (entries or []) if e.get("type") == "blob" and e.get("path")]
    for e in blobs:
        dst = Path(target) / e["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        TS.download_file(remote_prefix + "/" + e["path"], file_path=str(dst))
    return len(blobs)


def write_status(status, **extra):
    """Publish job state; terminal states latch so a late heartbeat can't undo them."""
    with _LOCK:
        if _FINALIZED:
            return
        if status in ("completed", "failed"):
            _FINALIZED.append(status)
        payload = {"status": status, "ts": _now(), "heartbeat": _now()}
        payload.update(extra)
        STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _push(STATUS_PATH, "_status.json")


write_status("starting", message="preparing filesystem")

# Jobs render code that hardcodes /kaggle/input and /kaggle/working. Lightning
# jobs run as an unprivileged user ("zeus"), so creating /kaggle at the root of
# the filesystem raises PermissionError. Fall back to a writable home-relative
# tree and rewrite the job source to match — the alternative (failing outright)
# would make the whole backend unusable for every existing job.
KAGGLE_ROOT = Path("/kaggle")
try:
    KAGGLE_ROOT.mkdir(parents=True, exist_ok=True)
    (KAGGLE_ROOT / "probe").mkdir(exist_ok=True)
    (KAGGLE_ROOT / "probe").rmdir()
except (PermissionError, OSError):
    KAGGLE_ROOT = Path(os.path.expanduser("~")) / "gpurunner_kaggle"
    KAGGLE_ROOT.mkdir(parents=True, exist_ok=True)
    print("[gpurunner] /kaggle not writable — using %s" % KAGGLE_ROOT, flush=True)

WORKING = KAGGLE_ROOT / "working"
WORKING.mkdir(parents=True, exist_ok=True)
(KAGGLE_ROOT / "input").mkdir(parents=True, exist_ok=True)

for _slug in INPUTS:
    _dst = KAGGLE_ROOT / "input" / _slug
    _dst.mkdir(parents=True, exist_ok=True)
    _remote = DATA_ROOT + "/" + _slug
    print("[gpurunner] downloading input %s" % _remote, flush=True)
    try:
        _pull_dir(_remote, _dst)
    except Exception as e:
        write_status("failed", error=repr(e), message="input %s not on the drive" % _remote)
        raise
    _got = [p for p in _dst.rglob("*") if p.is_file()]
    if not _got:
        write_status("failed", error="empty input",
                     message="%s downloaded 0 files — was it uploaded?" % _remote)
        raise RuntimeError("input %r came back empty from %s" % (_slug, _remote))
    print("[gpurunner] staged %d file(s), %.1f MB"
          % (len(_got), sum(p.stat().st_size for p in _got) / 1e6), flush=True)

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
    """Push the working dir to the drive so partial results survive a preemption.

    File-by-file (upload_folder is a no-op), skipping anything already uploaded at
    the same size so the periodic sync stays cheap.
    """
    for src in sorted(WORKING.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(WORKING).as_posix()
        size = src.stat().st_size
        if _SYNCED.get(rel) == size:
            continue
        try:
            TS.upload_file(str(src), remote_path=RUN_DIR + "/out/" + rel, progress_bar=False)
            _SYNCED[rel] = size
        except Exception as e:
            print("[gpurunner] sync of %s failed: %s" % (rel, e), flush=True)


def _heartbeat():
    last = time.time()
    while not _stop.wait(60):
        try:
            write_status("running", message="in progress")
            _push(LOG_PATH, "_runner.log")
            if time.time() - last >= 600:
                _sync_outputs()
                last = time.time()
        except Exception:
            pass


_beat = threading.Thread(target=_heartbeat, daemon=True)
_beat.start()

write_status("running", message="job started")
RUNNER_SRC = __RUNNER_SRC__
if str(KAGGLE_ROOT) != "/kaggle":
    # Point the job's hardcoded absolute paths at the writable tree we just made.
    RUNNER_SRC = RUNNER_SRC.replace("/kaggle/", str(KAGGLE_ROOT) + "/")
_real_out, _real_err = sys.stdout, sys.stderr
_err = None
with open(LOG_PATH, "w", encoding="utf-8", buffering=1) as _log:
    sys.stdout = _Tee(_real_out, _log)
    sys.stderr = _Tee(_real_err, _log)
    try:
        exec(compile(RUNNER_SRC, "<gpurunner_job>", "exec"), {"__name__": "__main__"})
    except BaseException as e:
        traceback.print_exc(file=sys.stderr)
        _err = repr(e)
    finally:
        sys.stdout, sys.stderr = _real_out, _real_err
        _stop.set()

_beat.join(timeout=90)
_sync_outputs()
_push(LOG_PATH, "_runner.log")
_n = sum(1 for p in WORKING.rglob("*") if p.is_file())
if _err is None:
    write_status("completed", message="%d files in out/" % _n)
    print("[gpurunner] DONE — %d files synced" % _n, flush=True)
else:
    write_status("failed", error=_err, message="%d files in out/ (partial)" % _n)
    print("[gpurunner] FAILED: %s" % _err, flush=True)
    sys.exit(1)
'''
