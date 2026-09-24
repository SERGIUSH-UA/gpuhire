"""SaturnBackend — headless batch jobs on Saturn Cloud (saturncloud.io).

Saturn Cloud is a hosted JupyterLab/jobs platform with a free tier that includes GPU
instances. **How much** it includes is not published anywhere the API or the site will
tell you (checked 2026-08-02: no balance, no quota endpoint, no figure in the UI) —
second-hand blog numbers are not repeated here as fact. Unlike Colab it has a real API: a *job*
is a first-class resource created from a **recipe**, started over HTTP, and
polled for logs and pod status — no browser click anywhere.

Shape of a run::

    upload inputs   →  sfs://<org>/<user>/gpurunner/data/<slug>/
    upload wrapper  →  sfs://<org>/<user>/gpurunner/runs/<handle>/job.py
    apply(recipe) + start()   →  the job container runs:
        saturnfs cp …/job.py  →  python job.py
        the wrapper: sfs data → /kaggle/input/<slug>, writes /kaggle/working,
                     syncs /kaggle/working → runs/<handle>/out/ every 10 min,
                     plus _status.json + _runner.log next to it
    status/logs/fetch  ←  resource state + those files over sfs

**Files move over ``saturnfs`` in both directions.** It is Saturn's own fsspec
filesystem (``sfs://`` paths), symmetric on the laptop and inside the job, which
makes it the one channel that does not depend on the container's disk surviving.

As with Colab, Vast, Lightning and Beam the remote side emulates the Kaggle
filesystem (``/kaggle/input/<slug>``, ``/kaggle/working``) and then executes
``job.render_remote_code(params)`` verbatim, so no job needs Saturn-specific code.

Backend-only params (read from RAW params, before ``validate_params`` drops
unknown keys): ``image`` ``instance_type`` ``inputs`` ``input_root``
``disk_space`` ``owner`` ``pip``.
"""

from __future__ import annotations

import contextlib
import json
import re
import shlex
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

from gpurunner.core.backend import Backend, BackendError
from gpurunner.core.inputs import resolve_inputs
from gpurunner.core.job import Job
from gpurunner.core.models import BalanceReport, JobHandle, JobStatus, StatusReport

#: Root of our tree inside Saturn's shared file storage, relative to the user's
#: own prefix (``sfs://<org>/<username>/``).
SFS_SUBDIR = "gpurunner"

#: gpurunner GPU name → substring searched for in Saturn's *sizes* catalogue.
#: Saturn does not publish a stable enum of instance types (they differ per
#: deployment), so ``submit`` resolves the actual ``instance_type`` at run time
#: from ``list_options("sizes")`` — see ``pick_instance_type``. ``-p instance_type=…``
#: always wins.
_GPU_TO_SATURN: dict[str, str] = {
    "none": "",
    "T4": "T4",
    "L4": "L4",
    "V100": "V100",
    "A10G": "A10G",
    "A100": "A100",
    "H100": "H100",
}

DEFAULT_WORKING_DIR = "/home/jovyan/workspace"
_HEARTBEAT_STALE = timedelta(minutes=10)


class SaturnBackend(Backend):
    """Run jobs as Saturn Cloud job resources."""

    name: ClassVar[str] = "saturn"
    gpu_choices: ClassVar[tuple[str, ...]] = tuple(_GPU_TO_SATURN.keys())
    default_gpu: ClassVar[str] = "T4"

    def __init__(self) -> None:
        self._conn: Any | None = None
        self._fs: Any | None = None

    # ---- public API -------------------------------------------------------

    def check_auth(self) -> None:
        from gpurunner.auth import saturn as saturn_auth

        saturn_auth.discover_credentials()
        self._connection()

    def submit(self, job: Job, params: dict[str, Any], *, gpu: str) -> JobHandle:
        if gpu not in self.gpu_choices:
            raise ValueError(
                f"Unsupported gpu '{gpu}' for saturn backend. Choose from: {self.gpu_choices}"
            )
        if self.name not in job.supported_backends:
            raise BackendError(f"Job {job.name!r} does not declare 'saturn' in supported_backends")

        opts = _backend_opts(params)
        if not opts["image"]:
            raise BackendError(
                "saturn needs an image to run in: -p image=<name-or-reference>. "
                "Take it from any existing resource in the web UI — Saturn images are "
                "per-deployment, there is no portable default. "
                "(`gpurunner saturn sizes` lists the instance types, not the images.)"
            )
        inputs = resolve_inputs(job, params, inputs=opts["inputs"], input_root=opts["input_root"])

        normalized = job.validate_params(params)
        body = job.render_remote_code(normalized)

        conn = self._connection()
        instance_type = opts["instance_type"] or pick_instance_type(
            self.list_sizes(), _GPU_TO_SATURN[gpu]
        )

        handle = JobHandle(
            backend=self.name,
            remote_id="<pending>",
            job_name=job.name,
            params=normalized,
            gpu=gpu,
        )
        run_dir = f"{self._sfs_root()}/runs/{handle.id}"
        job_name = f"gpurunner-{job.name}-{handle.id[:8]}"[:60].replace("_", "-")

        for slug, local in inputs.items():
            self._upload_dir(local, f"{self._sfs_root()}/data/{slug}")

        wrapper = _render_wrapper(
            body,
            run_dir=run_dir,
            data_root=f"{self._sfs_root()}/data",
            inputs=sorted(inputs),
        )
        self._write_file(f"{run_dir}/job.py", wrapper.encode("utf-8"))

        recipe: dict[str, Any] = {
            "type": "job",
            "spec": {
                "name": job_name,
                "owner": opts["owner"] or self._username(),
                "image": opts["image"],
                "instance_type": instance_type,
                "description": f"gpurunner {job.name} ({handle.id[:8]})",
                "command": _bootstrap_command(f"{run_dir}/job.py", pip=opts["pip"]),
                "working_directory": DEFAULT_WORKING_DIR,
                "start_dind": False,
            },
        }
        if opts["disk_space"]:
            recipe["spec"]["disk_space"] = opts["disk_space"]

        try:
            result = conn.apply(recipe)
        except Exception as e:
            raise BackendError(f"Saturn apply(recipe) failed: {e}") from e
        resource_id = _resource_id(result)
        if not resource_id:
            raise BackendError(f"Saturn accepted the recipe but returned no resource id: {result}")
        try:
            conn.start("job", resource_id)
        except Exception as e:
            # `apply` already created the resource, so a refused start would leave a
            # dead job sitting in the UI for every attempt. Clean it up, then explain.
            with contextlib.suppress(Exception):
                conn.delete("job", resource_id)
            text = str(e)
            if "UPGRADE_TO_PRO" in text or "exceeded the limits" in text:
                raise BackendError(
                    f"Saturn відмовив у запуску: акаунту не виділено годин compute.\n"
                    f"  {text.strip()}\n"
                    "Free tier на community-хмарі може мати ліміт 0 год — тоді жоден job не "
                    "стартує, поки не ввімкнено Pro (перевірено 2026-08-02). Ресурс видалено, "
                    "нічого не залишилось висіти."
                ) from None
            raise BackendError(f"Saturn start({job_name}) failed: {e}") from e

        handle.remote_id = job_name
        # The ABC's durable-storage slot: for Saturn it's the sfs folder carrying
        # status, logs and outputs for this run.
        handle.volume_name = run_dir
        handle.status = JobStatus.QUEUED
        handle.updated_at = datetime.now(tz=UTC)
        return handle

    def status(self, handle: JobHandle) -> StatusReport:
        resource_state = self._resource_status(handle)
        raw = self._read_status(handle)

        if raw is None:
            if resource_state == "error":
                return StatusReport(
                    status=JobStatus.FAILED, error="job errored before writing any status"
                )
            if resource_state == "stopped":
                return StatusReport(
                    status=JobStatus.FAILED,
                    error="job stopped without writing _status.json — see logs",
                )
            return StatusReport(
                status=JobStatus.QUEUED, message=f"job {resource_state or 'pending'}"
            )

        state = str(raw.get("status", "")).lower()
        message = str(raw.get("message") or "")
        if state == "completed":
            return StatusReport(status=JobStatus.COMPLETED, message=message or "done")
        if state == "failed":
            return StatusReport(status=JobStatus.FAILED, message=message, error=raw.get("error"))
        if state in ("running", "starting"):
            if resource_state == "error":
                return StatusReport(
                    status=JobStatus.FAILED,
                    error="job errored while the runner was still working",
                )
            beat = _parse_ts(raw.get("heartbeat") or raw.get("ts"))
            if beat is not None and datetime.now(tz=UTC) - beat > _HEARTBEAT_STALE:
                return StatusReport(
                    status=JobStatus.UNKNOWN,
                    message="heartbeat stale — the pod may have been evicted",
                )
            return StatusReport(status=JobStatus.RUNNING, message=message)
        return StatusReport(status=JobStatus.UNKNOWN, message=f"unrecognized job status {state!r}")

    def fetch_outputs(self, handle: JobHandle, out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = self._run_dir(handle)
        fs = self._filesystem()

        written: list[Path] = []
        for remote in self._list_remote(run_dir):
            rel = remote[len(run_dir):].lstrip("/")
            if rel == "job.py":
                continue  # the wrapper we uploaded, not an output
            local_rel = rel[len("out/"):] if rel.startswith("out/") else rel
            target = out_dir / local_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                fs.get_file(remote, str(target))
            except Exception as e:
                raise BackendError(f"download of {remote} failed: {e}") from e
            written.append(target)
        return written

    def logs(self, handle: JobHandle) -> list[str]:
        blob = self._read_bytes(f"{self._run_dir(handle)}/_runner.log")
        if blob is not None:
            return blob.decode("utf-8", errors="replace").splitlines()
        # Before the wrapper starts logging, the container's own output is all
        # there is — image pull, pip install, or the reason it never got going.
        try:
            text = self._connection().get_logs("job", handle.remote_id)
        except Exception as e:
            return [f"(no logs yet: {e})"]
        return str(text or "").splitlines() or ["(no logs yet)"]

    def cancel(self, handle: JobHandle) -> None:
        conn = self._connection()
        resource = self._resource(handle)
        if resource is None:
            raise BackendError(f"job {handle.remote_id} not found on Saturn Cloud")
        resource_id = _resource_id(resource)
        try:
            conn.stop("job", resource_id)
        except Exception as e:
            raise BackendError(f"Saturn stop failed: {e}") from e

    def balance(self) -> BalanceReport:
        """Compute used this month, from the org's daily-usage endpoint.

        Saturn exposes *usage*, never a remaining-hours figure — the free tier's
        monthly allowance is not published through the API. So this reports what
        was consumed and links to the billing page rather than inventing a
        remainder (compare the Modal backend, which can at least subtract from a
        documented constant).
        """
        conn = self._connection()
        now = datetime.now(tz=UTC)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        detail = "Saturn не віддає залишок годин через API"
        spent: float | None = None
        try:
            org = conn.primary_org
            user = conn.current_user
            usage = conn.get_user_usage(org["id"], user["id"], start, now)
            spent = _sum_usage_hours(usage)
            if spent is not None:
                detail = f"витрачено з {start:%Y-%m-%d} · {detail}"
            else:
                detail = f"формат usage-відповіді не розпізнано ({_usage_keys(usage)}) · {detail}"
        except Exception as e:
            detail = f"usage-запит не вдався ({type(e).__name__}) · {detail}"
        return BalanceReport(
            backend=self.name,
            available=None,
            unit="год",
            spent=spent,
            detail=detail,
            url="https://saturncloud.io/",
        )

    def wait_for_log_line(
        self, handle: JobHandle, needle: str, *, timeout: int = 1800, poll: int = 60
    ) -> str | None:
        """Poll the sfs-synced log until a line contains ``needle``. None on timeout."""
        start = time.monotonic()
        while True:
            for line in self.logs(handle):
                if needle in line:
                    return line
            if time.monotonic() - start > timeout:
                return None
            time.sleep(poll)

    # ---- catalogue --------------------------------------------------------

    def list_sizes(self) -> list[dict[str, Any]]:
        """Instance sizes this Saturn deployment offers (name, cores, gpu, …)."""
        try:
            return list(self._connection().list_options("sizes"))
        except Exception as e:
            raise BackendError(f"Saturn size catalogue unavailable: {e}") from e

    # ---- plumbing ---------------------------------------------------------

    def _connection(self) -> Any:
        if self._conn is None:
            from gpurunner.auth import saturn as saturn_auth

            self._conn = saturn_auth.connect()
        return self._conn

    def _filesystem(self) -> Any:
        if self._fs is None:
            from gpurunner.auth import saturn as saturn_auth

            self._fs = saturn_auth.filesystem()
        return self._fs

    def _username(self) -> str:
        user = self._connection().current_user or {}
        name = user.get("username")
        if not name:
            raise BackendError("Saturn API returned no username for the current user")
        return str(name)

    def _sfs_root(self) -> str:
        conn = self._connection()
        org = (conn.primary_org or {}).get("name")
        if not org:
            raise BackendError("Saturn API returned no primary organization")
        return f"sfs://{org}/{self._username()}/{SFS_SUBDIR}"

    def _run_dir(self, handle: JobHandle) -> str:
        if not handle.volume_name:
            raise BackendError(f"handle {handle.id[:8]} has no Saturn run folder recorded")
        return handle.volume_name

    def _resource(self, handle: JobHandle) -> dict[str, Any] | None:
        try:
            return self._connection().get_resource("job", handle.remote_id)
        except Exception:
            return None

    def _resource_status(self, handle: JobHandle) -> str | None:
        resource = self._resource(handle)
        if not resource:
            return None
        state = (resource.get("state") or {}).get("status")
        return str(state).lower() if state else None

    def _upload_dir(self, local: Path, remote_prefix: str) -> int:
        """Upload a directory to sfs, then verify it actually landed."""
        fs = self._filesystem()
        files = [p for p in sorted(local.rglob("*")) if p.is_file()]
        if not files:
            raise BackendError(f"input dir {local} has no files")
        for p in files:
            rel = p.relative_to(local).as_posix()
            try:
                fs.put_file(str(p), f"{remote_prefix}/{rel}")
            except Exception as e:
                raise BackendError(f"upload of {p} → {remote_prefix}/{rel} failed: {e}") from e

        landed = {
            remote[len(remote_prefix):].lstrip("/") for remote in self._list_remote(remote_prefix)
        }
        missing = {p.relative_to(local).as_posix() for p in files} - landed
        if missing:
            raise BackendError(
                f"upload to {remote_prefix} reported success but "
                f"{len(missing)} file(s) are not on sfs: {sorted(missing)[:5]}"
            )
        return len(files)

    def _list_remote(self, prefix: str) -> list[str]:
        """Absolute ``sfs://`` paths of every file under ``prefix``."""
        fs = self._filesystem()
        try:
            found = fs.find(prefix)
        except FileNotFoundError:
            return []
        except Exception as e:
            raise BackendError(f"listing {prefix} failed: {e}") from e
        scheme = prefix.split("://", 1)[0] + "://"
        return [p if "://" in p else scheme + p.lstrip("/") for p in (found or [])]

    def _write_file(self, remote_path: str, blob: bytes) -> None:
        fs = self._filesystem()
        with tempfile.TemporaryDirectory(prefix="gpurunner-saturn-") as tmp:
            local = Path(tmp) / Path(remote_path).name
            local.write_bytes(blob)
            try:
                fs.put_file(str(local), remote_path)
            except Exception as e:
                raise BackendError(f"upload of {remote_path} failed: {e}") from e

    def _read_bytes(self, remote_path: str) -> bytes | None:
        fs = self._filesystem()
        try:
            return bytes(fs.cat_file(remote_path))
        except Exception:
            return None

    def _read_status(self, handle: JobHandle) -> dict[str, Any] | None:
        blob = self._read_bytes(f"{self._run_dir(handle)}/_status.json")
        if blob is None:
            return None
        try:
            data = json.loads(blob.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None  # caught mid-write
        return data if isinstance(data, dict) else None


# ---- helpers ---------------------------------------------------------------


def _backend_opts(params: dict[str, Any]) -> dict[str, Any]:
    """Backend-only knobs, read from RAW params (validate_params drops unknown keys)."""
    return {
        "image": (str(params["image"]).strip() if params.get("image") else None),
        "instance_type": (
            str(params["instance_type"]).strip() if params.get("instance_type") else None
        ),
        "owner": (str(params["owner"]).strip() if params.get("owner") else None),
        "disk_space": (str(params["disk_space"]).strip() if params.get("disk_space") else None),
        "inputs": params.get("inputs") or {},
        "input_root": params.get("input_root"),
        "pip": bool(params.get("pip", True)),
    }


def _size_cores(size: dict[str, Any]) -> float:
    try:
        return float(size.get("cores") or 0)
    except (TypeError, ValueError):
        return 0.0


def _size_gpus(size: dict[str, Any]) -> float:
    try:
        return float(size.get("gpu") or 0)
    except (TypeError, ValueError):
        return 0.0


def _mentions_gpu(size: dict[str, Any], needle: str) -> bool:
    """Does this catalogue entry name the requested card?

    Substring matching is not enough: ``"l4" in "Shadeform L40S"`` is true, so a
    request for an L4 would quietly rent an L40S. Two passes instead — a strict
    token match first (``T4-XLarge`` → ``t4``), then a looser one that only bans a
    trailing digit, because real entries like ``nebius-1xh100`` glue the model name
    to the GPU count and would fail a left word boundary. ``gpu_type`` is useless
    here: the community catalogue reports it as plain ``"NVIDIA"``.
    """
    values = [str(v).lower() for v in size.values() if isinstance(v, str)]
    esc = re.escape(needle.lower())
    for pattern in (rf"(?<![a-z0-9]){esc}(?![a-z0-9])", rf"{esc}(?![0-9])"):
        if any(re.search(pattern, v) for v in values):
            return True
    return False


def pick_instance_type(sizes: list[dict[str, Any]], gpu_needle: str) -> str:
    """Smallest instance in the catalogue that carries the requested GPU.

    Saturn's ``sizes`` catalogue is per-deployment: the names (``g4dnxlarge``,
    ``nebius/nebius-1xh100``, ``k0rdent/shadeform-…``) differ between the community
    cloud, an enterprise install and whatever they rename things to next. So instead
    of hardcoding a mapping that silently rots, the requested card is matched against
    the catalogue the account actually has, and an unmatched request lists what *is*
    available.
    """
    if not sizes:
        raise BackendError("Saturn returned an empty size catalogue")

    if not gpu_needle:
        cpu_only = [s for s in sizes if _size_gpus(s) == 0]
        if not cpu_only:
            raise BackendError("Saturn catalogue has no CPU-only size")
        return str(min(cpu_only, key=_size_cores)["name"])

    matches = [s for s in sizes if _size_gpus(s) >= 1 and _mentions_gpu(s, gpu_needle)]
    if not matches:
        available = ", ".join(
            sorted(str(s.get("name")) for s in sizes if _size_gpus(s) >= 1)
        ) or "—"
        raise BackendError(
            f"no Saturn instance type carries a {gpu_needle}. GPU sizes on this account: "
            f"{available}. Pick one with -p instance_type=<name>."
        )
    return str(min(matches, key=lambda s: (_size_gpus(s), _size_cores(s)))["name"])


def estimate_cost(job: Any, params: dict, gpu: str, sizes: list[dict[str, Any]]) -> dict | None:
    """Наперед-прорахунок Saturn-run.

    Free-план не віддає тарифів (``price_per_hour`` = ``null`` у community-каталозі),
    тож основна валюта тут — **години**. Якщо каталог усе ж має ціну
    (enterprise-деплой), додається і сума в $. ``None``, коли job не вміє оцінити час
    або потрібної карти в каталозі немає.
    """
    try:
        secs = job.estimate_runtime({**params, "_gpu": gpu}).total_seconds()
    except Exception:
        return None
    try:
        name = params.get("instance_type") or pick_instance_type(sizes, _GPU_TO_SATURN.get(gpu, ""))
    except BackendError:
        return None
    size = next((s for s in sizes if str(s.get("name")) == str(name)), None)
    price = (size or {}).get("price_per_hour")
    hours = secs / 3600.0
    out: dict[str, Any] = {"hours": hours, "gpu": gpu, "instance_type": name, "total": None}
    try:
        if price is not None:
            out["total"] = float(price) * hours
    except (TypeError, ValueError):
        out["total"] = None
    return out


def _bootstrap_command(job_path: str, *, pip: bool) -> str:
    """Shell command the job container runs: fetch the wrapper off sfs, run it."""
    prefix = "pip install -q saturnfs && " if pip else ""
    return (
        f"{prefix}saturnfs cp {shlex.quote(job_path)} /tmp/gpurunner_job.py && "
        "python -u /tmp/gpurunner_job.py"
    )


def _resource_id(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    state = payload.get("state")
    if isinstance(state, dict) and state.get("id"):
        return str(state["id"])
    return str(payload["id"]) if payload.get("id") else None


def _usage_keys(usage: Any) -> str:
    if isinstance(usage, list) and usage and isinstance(usage[0], dict):
        return ", ".join(sorted(usage[0]))
    return type(usage).__name__


def _sum_usage_hours(usage: Any) -> float | None:
    """Total hours in a daily-usage payload, or ``None`` if the shape is unknown.

    The endpoint is undocumented, so this only trusts a key that literally names
    hours — anything else is reported as "unrecognised" rather than guessed at.
    """
    if not isinstance(usage, list):
        return None
    total = 0.0
    seen = False
    for row in usage:
        if not isinstance(row, dict):
            return None
        for key, value in row.items():
            if "hour" in key.lower() and isinstance(value, (int, float)):
                total += float(value)
                seen = True
    return total if seen else None


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _render_wrapper(body: str, *, run_dir: str, data_root: str, inputs: list[str]) -> str:
    """Source of the module the Saturn job downloads off sfs and executes.

    ``__RUNNER_SRC__`` is substituted last: the job body is arbitrary source that
    could itself contain one of the other placeholders.
    """
    return (
        _RUNNER_WRAPPER.replace("__RUN_DIR__", json.dumps(run_dir))
        .replace("__DATA_ROOT__", json.dumps(data_root))
        .replace("__INPUTS__", json.dumps(json.dumps(inputs)))
        .replace("__RUNNER_SRC__", json.dumps(body, ensure_ascii=False))
    )


#: Runs inside the Saturn job. Same ``_status.json`` contract (and terminal latch)
#: as the Colab, Vast, Lightning and Beam backends, so `status` reads them alike.
_RUNNER_WRAPPER = '''\
import json, os, sys, threading, time, traceback
from datetime import datetime, timezone
from pathlib import Path

from saturnfs import SaturnFS

RUN_DIR = __RUN_DIR__
DATA_ROOT = __DATA_ROOT__
INPUTS = json.loads(__INPUTS__)

# saturnfs authenticates off SATURN_TOKEN/SATURN_BASE_URL, which Saturn injects
# into its own resources. Fail loudly here rather than three minutes into a GPU
# run with an unreadable stack trace from deep inside fsspec.
for _var in ("SATURN_TOKEN", "SATURN_BASE_URL"):
    if not os.environ.get(_var):
        raise RuntimeError(
            "%s is not set inside the job — saturnfs cannot reach the run folder" % _var
        )

FS = SaturnFS()

LOCAL = Path("/tmp/gpurunner")
LOCAL.mkdir(parents=True, exist_ok=True)
STATUS_PATH = LOCAL / "_status.json"
LOG_PATH = LOCAL / "_runner.log"

_LOCK = threading.Lock()
_FINALIZED = []
_SYNCED = {}


def _now():
    return datetime.now(tz=timezone.utc).isoformat()


def _push(local_path, remote_name):
    try:
        FS.put_file(str(local_path), RUN_DIR + "/" + remote_name)
    except Exception as e:
        print("[gpurunner] upload of %s failed: %s" % (remote_name, e), flush=True)


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

# Saturn jobs run as an unprivileged user (jovyan), so /kaggle usually cannot be
# created at the filesystem root. Fall back to a writable home-relative tree and
# rewrite the job source to match (same trick as the Lightning backend).
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
        _blobs = FS.find(_remote)
    except Exception as e:
        write_status("failed", error=repr(e), message="input %s unreadable" % _remote)
        raise
    # fsspec strips the protocol from results, so compare on the bare key.
    _prefix = _remote.split("://", 1)[1].rstrip("/")
    for _blob in _blobs or []:
        _key = _blob.split("://", 1)[-1]
        _rel = _key[len(_prefix):].lstrip("/") if _key.startswith(_prefix) else Path(_key).name
        _target = _dst / _rel
        _target.parent.mkdir(parents=True, exist_ok=True)
        FS.get_file(_blob, str(_target))
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
    """Push the working dir to sfs so partial results survive a lost pod."""
    for src in sorted(WORKING.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(WORKING).as_posix()
        size = src.stat().st_size
        if _SYNCED.get(rel) == size:
            continue
        try:
            FS.put_file(str(src), RUN_DIR + "/out/" + rel)
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
