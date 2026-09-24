"""BeamBackend — headless serverless jobs on Beam (beam.cloud).

Beam is the closest analogue to Modal: a Python-native serverless GPU platform
with per-second billing and — unlike Vast or RunPod — a *recurring* free tier
($30 of credit refreshed monthly on the Developer plan, no card required).

Shape of a run::

    upload inputs   →  Beam Volume "gpurunner-data"  (<slug>/…)
    task_queue(...).put()  →  task id
    the task:  /inputs/<slug> → /kaggle/input/<slug>,  writes /kaggle/working,
               syncs  /kaggle/working → /outputs/out/  every 10 min and at the end,
               plus   /outputs/_status.json + /outputs/_runner.log
    status/logs/fetch  ←  Task API + those files on the output Volume

As with Colab, Vast and Lightning the remote side emulates the Kaggle filesystem
(``/kaggle/input/<slug>``, ``/kaggle/working``) and then executes
``job.render_remote_code(params)`` verbatim, so no job needs Beam-specific code.

**Why a generated entry module and not ``exec()`` like Modal.** Beam does not ship
a callable across the wire: ``prepare_runtime`` resolves the decorated function to
``<module file relative to CWD>:<name>`` and syncs the *current working directory*
into the container. So the backend writes a self-contained ``entry.py`` (job body
embedded as a string literal) into a temp dir, chdir's there for the duration of
the submit, and imports it. That keeps the sync payload to a few files instead of
the whole gpurunner checkout.

Backend-only params (read from RAW params, before ``validate_params`` drops
unknown keys): ``inputs`` ``input_root`` ``max_hours`` ``cpu`` ``memory``
``retries`` ``data_volume``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

from gpurunner.core.backend import Backend, BackendError
from gpurunner.core.inputs import resolve_inputs
from gpurunner.core.job import Job
from gpurunner.core.models import BalanceReport, JobHandle, JobStatus, StatusReport

#: gpurunner GPU name → Beam ``GpuType`` literal (verified against beam-client's
#: ``beta9.type.GpuType`` — Beam's card list is much wider than Modal's, but the
#: names differ: A100 comes in ``A100-40``/``A100-80``, and there is no bare
#: "A100"). ``""`` means CPU-only.
_GPU_TO_BEAM: dict[str, str] = {
    "none": "",
    "T4": "T4",
    "L4": "L4",
    "A10G": "A10G",
    "A100": "A100-40",
    "RTX4090": "RTX4090",
    "A6000": "A6000",
    "RTX5090": "RTX5090",
    "L40S": "L40S",
    "A100-80GB": "A100-80",
    "RTXPro6000": "RTXPro6000",
    "H100": "H100",
    "H200": "H200",
    "B200": "B200",
}

#: $/год **опублікованого** serverless-прайсу Beam (beam.cloud/pricing, звірено
#: 2026-08-02, перезвірено 2026-08-09 — числа ті самі; на сторінці ціни в $/сек,
#: тут ×3600). Карти, яких у прайсі НЕМА (T4, L4, A10G, A100-40), свідомо
#: відсутні: `estimate_cost` для них поверне None замість вигаданої цифри.
#: Маркетплейс дедикованих інстансів («Compute» у дашборді) тарифікується інакше —
#: ці числа саме про serverless-таски, які запускає цей бекенд.
#:
#: 🔴 ПРАЙС НЕ ДОРІВНЮЄ РАХУНКУ. Ці числа лишились тут як опубліковані, але
#: рахувати вартість треба через `_gpu_hourly()` — див. `_GPU_MEASURED_HOURLY`.
_GPU_HOURLY: dict[str, float] = {
    "none": 0.0,
    "RTX4090": 0.000191667 * 3600,   # $0.69
    "A6000": 0.000227778 * 3600,     # $0.82
    "RTX5090": 0.000303 * 3600,      # $1.09
    "L40S": 0.000486 * 3600,         # $1.75
    "A100-80GB": 0.000625 * 3600,    # $2.25
    "RTXPro6000": 0.000758 * 3600,   # $2.73
    "H100": 0.000986 * 3600,         # $3.55
    "H200": 0.001136 * 3600,         # $4.09
    "B200": 0.001561 * 3600,         # $5.62
}
#: CPU тарифікується за **фізичне ядро** (= 2 vCPU), RAM — за GiB, обидва per-second.
_CPU_CORE_PER_S: float = 0.0000125
_RAM_GIB_PER_S: float = 0.0000021

#: 🔴 GPU-складова $/год, ВИМІРЯНА ЗА РАХУНКОМ, а не взята з прайсу.
#:
#: Рахунок Beam за taskqueue/entry:execute (RTX4090, 8 ядер, 32 ГБ, 21 хв 32 с,
#: 2026-08-09) показав ставку **$2.8414/год** і списав $1.020 кредиту
#: (2.8414 × 1292/3600 = 1.0199 — сходиться до цента). Формула за прайсом дає
#: для тієї ж конфігурації $1.2919/год, тобто рахунок **у 2.2 раза більший**.
#:
#: Чому — з опублікованих чисел не виводиться: сторінка тарифів того дня
#: показувала ті самі $0.000191667/сек, а docs.beam.cloud/v2/resources/
#: pricing-and-billing механіку не розкриває (жодного слова про мінімальну
#: тарифіковану конфігурацію, округлення чи бандл CPU/RAM у ціну карти).
#: Тому тут НЕ модель, а факт: різниця віднесена на GPU-складову за
#: припущення, що опубліковані ставки CPU/RAM правильні —
#: 2.8414 − 8×0.045 − 32×0.00756 = 2.2395.
#: Кожна нова карта, за яку прийде рахунок, має отримати тут свій рядок.
_GPU_MEASURED_HOURLY: dict[str, float] = {
    "RTX4090": 2.2395,   # рахунок 2026-08-09; прайс обіцяв 0.69
}

#: Множник для карт, за які рахунку ще не було. Не «модель ціноутворення», а
#: страховка стелі витрат: єдиний наявний вимір показав, що GPU-складова
#: коштує в 3.25 раза (2.2395 / 0.69) більше за прайс, і немає підстав вважати
#: RTX4090 винятком. Асиметрія навмисна — заниження стелі означає списані
#: гроші, завищення означає відмову в сабміті, яку знімає `--allow-cost`.
_UNVERIFIED_MARKUP: float = 2.2395 / (0.000191667 * 3600)


def _gpu_hourly(gpu: str) -> float | None:
    """$/год GPU-складової: вимір, якщо він є, інакше прайс зі страховкою."""
    measured = _GPU_MEASURED_HOURLY.get(gpu)
    if measured is not None:
        return measured
    listed = _GPU_HOURLY.get(gpu)
    if listed is None:
        return None
    return listed * _UNVERIFIED_MARKUP

#: Volume mount points inside the container. **Relative, not absolute** — Beam
#: mounts volumes under the container's working dir (``/mnt/code``) and quietly
#: ignores an absolute path: with ``mount_path="/outputs"`` the task still ran and
#: still reported "1 file written", but the volume stayed empty because the wrapper
#: had `mkdir`'d a plain local directory (smoke run 99058bb9, 2026-08-02).
_OUT_MOUNT = "./outputs"
_IN_MOUNT = "./inputs"

#: Shared volume holding staged inputs, one sub-directory per slug.
DEFAULT_DATA_VOLUME = "gpurunner-data"

DEFAULT_MAX_HOURS = 12
DEFAULT_CPU = 2.0
DEFAULT_MEMORY = "8Gi"
_HEARTBEAT_STALE = timedelta(minutes=10)

#: Стелі витрат. Beam — єдиний бекенд, де прив'язана картка означає, що після
#: вичерпання безкоштовних $30 списуються справжні гроші, і при цьому API не
#: віддає ні балансу, ні витрат. Тому обидві стелі перевіряються локально й
#: **до** будь-якої заливки даних (див. ``check_spend_allowed``).
DEFAULT_RUN_CEILING = 1.0          # $ на один run без явного --allow-cost
DEFAULT_MONTHLY_BUDGET = 30.0      # $ на місяць = рівно безкоштовний кредит


def _out_volume_name(handle_id: str) -> str:
    return f"gpurunner-out-{handle_id[:24]}"


def estimate_cost(job: Any, params: dict, gpu: str) -> dict | None:
    """Наперед-прорахунок вартості Beam-run: $/епоха, $/всього, год.

    ``None``, коли для карти немає опублікованого serverless-тарифу або job не вміє
    оцінити свій час — краще не показати нічого, ніж показати вигадане число.
    Beam білить per-second і не має мінімального часу, тож fudge-фактор Modal тут
    не потрібен; лишається лише CPU/RAM поверх GPU.

    GPU-складова береться з `_gpu_hourly()`, а не напряму з прайсу: рахунок за
    RTX4090 виявився вдвічі більшим за прайсовий (див. `_GPU_MEASURED_HOURLY`).
    """
    gpu_hr = _gpu_hourly(gpu)
    if gpu_hr is None:
        return None
    try:
        secs = job.estimate_runtime({**params, "_gpu": gpu}).total_seconds()
    except Exception:
        return None
    cores = float(params.get("cpu") or DEFAULT_CPU)
    ram_gib = _memory_to_gib(params.get("memory") or DEFAULT_MEMORY)
    per_s = gpu_hr / 3600.0 + cores * _CPU_CORE_PER_S + ram_gib * _RAM_GIB_PER_S
    total = per_s * secs
    try:
        epochs = max(1, int(params.get("epochs", 1)))
    except (TypeError, ValueError):
        epochs = 1
    return {
        "hours": secs / 3600.0,
        "total": total,
        "per_epoch": total / epochs,
        "gpu": gpu,
        "cores": cores,
        "ram_gib": ram_gib,
    }


class SpendRefused(BackendError):
    """Submit blocked by the spend guard — nothing was uploaded or started."""


def hourly_rate(gpu: str, params: dict[str, Any]) -> float | None:
    """$/год цього конкретного контейнера (GPU + CPU + RAM), або ``None``.

    ``None`` означає «Beam не публікує serverless-тариф для цієї карти» — для
    guard'а це не «безкоштовно», а «стелю порахувати неможливо».
    """
    override = params.get("price_per_hour")
    gpu_hr = _gpu_hourly(gpu) if override is None or override == "" else float(override)
    if gpu_hr is None:
        return None
    cores = float(params.get("cpu") or DEFAULT_CPU)
    ram_gib = _memory_to_gib(params.get("memory") or DEFAULT_MEMORY)
    return gpu_hr + (cores * _CPU_CORE_PER_S + ram_gib * _RAM_GIB_PER_S) * 3600.0


def worst_case_cost(gpu: str, params: dict[str, Any]) -> float | None:
    """Максимум, який цей run може списати: timeout × ставка.

    Саме timeout, а не оцінка часу: job, що завис, коштуватиме рівно стільки, і
    тільки ця цифра має право керувати блокуванням.
    """
    rate = hourly_rate(gpu, params)
    if rate is None:
        return None
    return rate * float(params.get("max_hours") or DEFAULT_MAX_HOURS)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        raise BackendError(f"{name}={raw!r} is not a number") from None


def check_run_ceiling(gpu: str, params: dict[str, Any]) -> float:
    """Raise ``SpendRefused`` unless this run fits the **per-run** ceiling.

    ``--allow-cost`` / ``-p max_cost`` / ``GPURUNNER_BEAM_MAX_RUN_COST`` (default $1).
    A single mistyped flag is what this catches. Returns the worst case in $.
    """
    worst = worst_case_cost(gpu, params)
    if worst is None:
        raise SpendRefused(
            f"Beam publishes no serverless price for {gpu!r}, so the spending ceiling for this "
            f"run cannot be computed — refusing to submit. Pick a priced card "
            f"({', '.join(k for k in _GPU_HOURLY if k != 'none')}) or state the rate yourself "
            f"with -p price_per_hour=<$/h>."
        )

    per_run = (
        float(params["max_cost"])
        if params.get("max_cost") not in (None, "")
        else _env_float("GPURUNNER_BEAM_MAX_RUN_COST", DEFAULT_RUN_CEILING)
    )
    hours = float(params.get("max_hours") or DEFAULT_MAX_HOURS)
    if worst > per_run:
        raise SpendRefused(
            f"РЕАЛЬНІ ГРОШІ: цей run може списати до ${worst:.2f} "
            f"({gpu}, timeout {hours:g} год × ${worst / hours:.2f}/год), "
            f"а стеля — ${per_run:.2f}.\n"
            f"  дешевше:   --gpu RTX4090 -p max_hours=<менше>\n"
            f"  дозволити: gpurunner run … --allow-cost {worst:.2f}\n"
            f"  назавжди:  GPURUNNER_BEAM_MAX_RUN_COST=<$>"
        )
    return worst


def monthly_budget() -> float:
    """$ дозволених на місяць. Дефолт = рівно безкоштовний кредит Beam."""
    return _env_float("GPURUNNER_BEAM_MONTHLY_BUDGET", DEFAULT_MONTHLY_BUDGET)


def _memory_to_gib(value: Any) -> float:
    """``"16Gi"`` / ``"512Mi"`` / ``16384`` (MiB, як у Beam) → GiB."""
    if isinstance(value, (int, float)):
        return float(value) / 1024.0
    text = str(value).strip()
    try:
        if text.lower().endswith("gi"):
            return float(text[:-2])
        if text.lower().endswith("mi"):
            return float(text[:-2]) / 1024.0
        return float(text) / 1024.0
    except ValueError:
        return 0.0


class BeamBackend(Backend):
    """Run jobs as Beam task-queue tasks."""

    name: ClassVar[str] = "beam"
    gpu_choices: ClassVar[tuple[str, ...]] = tuple(_GPU_TO_BEAM.keys())
    #: RTX4090, не T4: у serverless-прайсі Beam T4/L4/A10G/A100-40 взагалі немає
    #: (лише в маркетплейсі дедикованих інстансів), а 4090 — найдешевша з тих, що є,
    #: і при цьому 24 ГБ проти 16 у T4.
    default_gpu: ClassVar[str] = "RTX4090"

    # ---- public API -------------------------------------------------------

    def check_auth(self) -> None:
        from gpurunner.auth import beam as beam_auth

        beam_auth.discover_credentials()
        beam_auth.import_sdk()

    def submit(self, job: Job, params: dict[str, Any], *, gpu: str) -> JobHandle:
        from gpurunner.auth import beam as beam_auth

        if gpu not in self.gpu_choices:
            raise ValueError(
                f"Unsupported gpu '{gpu}' for beam backend. Choose from: {self.gpu_choices}"
            )
        if self.name not in job.supported_backends:
            raise BackendError(f"Job {job.name!r} does not declare 'beam' in supported_backends")

        # Spend guard FIRST — before inputs are uploaded, before an image is built,
        # before anything that could start the meter. Raises SpendRefused.
        worst = check_run_ceiling(gpu, params)

        opts = _backend_opts(params)
        inputs = resolve_inputs(
            job, params, inputs=opts["inputs"], input_root=opts["input_root"]
        )

        normalized = job.validate_params(params)
        body = job.render_remote_code(normalized)

        beam_auth.discover_credentials()
        beam_auth.import_sdk()

        handle = JobHandle(
            backend=self.name,
            remote_id="<pending>",
            job_name=job.name,
            params=normalized,
            gpu=gpu,
        )
        out_volume = _out_volume_name(handle.id)

        # Book the worst case against the month BEFORE anything can start the meter.
        # The check and the write share one transaction, so two shells submitting at
        # once cannot both slip under the same remaining budget. Rolled back below if
        # the submit does not reach a running task.
        from gpurunner.core import budget

        try:
            committed = budget.reserve_within_cap(
                self.name,
                handle.id,
                worst,
                cap=monthly_budget(),
                gpu=gpu,
                job=job.name,
                max_hours=opts["max_hours"],
                rate=hourly_rate(gpu, params),
            )
        except budget.BudgetExceeded as e:
            raise SpendRefused(
                f"РЕАЛЬНІ ГРОШІ: місячна стеля ${e.cap:.2f} вичерпується — вже заброньовано "
                f"${e.committed:.2f}, цей run додає до ${e.requested:.2f}.\n"
                f"  (стеля = безкоштовний кредит Beam; облік локальний, бо API витрат не "
                f"віддає: gpurunner balance -b beam)\n"
                f"  підняти: GPURUNNER_BEAM_MONTHLY_BUDGET=<$>"
            ) from None

        try:
            if inputs:
                for slug, local in inputs.items():
                    self._upload_dir(local, opts["data_volume"], slug)

            entry_src = _render_entry_module(
                body,
                job_name=job.name,
                handle_id=handle.id,
                gpu=_GPU_TO_BEAM[gpu],
                image_spec=job.modal_image_spec(),
                inputs=sorted(inputs),
                out_volume=out_volume,
                data_volume=opts["data_volume"],
                opts=opts,
            )

            with tempfile.TemporaryDirectory(prefix="gpurunner-beam-") as tmp:
                entry_path = Path(tmp) / "entry.py"
                entry_path.write_text(entry_src, encoding="utf-8")
                with _chdir(tmp):
                    module = _import_module(entry_path)
                    try:
                        task = module.execute.put()
                    except Exception as e:
                        raise BackendError(f"Beam put() failed: {e}") from e
                if not task:
                    raise BackendError(
                        "Beam refused the task (image build or file sync failed) — "
                        "see the output above."
                    )
        except BaseException:
            # Nothing is running, so the money must not stay booked.
            budget.release(self.name, handle.id)
            raise

        handle.remote_id = str(getattr(task, "id", task))
        # The ABC's durable-storage slot: for Beam it's the per-run output volume
        # carrying status, logs and results.
        handle.volume_name = out_volume
        handle.status = JobStatus.QUEUED
        handle.updated_at = datetime.now(tz=UTC)
        handle.params = {**normalized, "_worst_case_cost": round(worst, 4),
                         "_month_committed": round(committed, 4)}
        return handle

    def status(self, handle: JobHandle) -> StatusReport:
        task_state = self._task_status(handle)
        raw = self._read_status(handle)
        self._settle_budget(handle, raw, task_state)

        if raw is None:
            if task_state in ("error", "timeout"):
                return StatusReport(
                    status=JobStatus.FAILED,
                    error=f"task {task_state} before writing any status",
                )
            if task_state == "cancelled":
                return StatusReport(status=JobStatus.CANCELLED, message="task cancelled")
            if task_state == "complete":
                return StatusReport(
                    status=JobStatus.FAILED,
                    error="task finished without writing _status.json — see logs",
                )
            return StatusReport(status=JobStatus.QUEUED, message=f"task {task_state or 'pending'}")

        state = str(raw.get("status", "")).lower()
        message = str(raw.get("message") or "")
        if state == "completed":
            return StatusReport(status=JobStatus.COMPLETED, message=message or "done")
        if state == "failed":
            return StatusReport(status=JobStatus.FAILED, message=message, error=raw.get("error"))
        if state in ("running", "starting"):
            if task_state == "cancelled":
                return StatusReport(status=JobStatus.CANCELLED, message="task cancelled")
            if task_state in ("error", "timeout"):
                return StatusReport(
                    status=JobStatus.FAILED,
                    error=f"task {task_state} while the runner was still working",
                )
            beat = _parse_ts(raw.get("heartbeat") or raw.get("ts"))
            if beat is not None and datetime.now(tz=UTC) - beat > _HEARTBEAT_STALE:
                return StatusReport(
                    status=JobStatus.UNKNOWN,
                    message="heartbeat stale — the container may have been evicted",
                )
            return StatusReport(status=JobStatus.RUNNING, message=message)
        return StatusReport(status=JobStatus.UNKNOWN, message=f"unrecognized job status {state!r}")

    def fetch_outputs(self, handle: JobHandle, out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        volume = self._volume_of(handle)

        written: list[Path] = []
        for rel in self._list_volume(volume):
            # `out/foo.txt` → <out_dir>/foo.txt; sidecars land next to them.
            local_rel = rel[len("out/"):] if rel.startswith("out/") else rel
            blob = self._download_bytes(volume, rel)
            if blob is None:
                continue
            target = out_dir / local_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            written.append(target)
        return written

    def logs(self, handle: JobHandle) -> list[str]:
        blob = self._download_bytes(self._volume_of(handle), "_runner.log")
        if blob is None:
            return ["(no logs yet — the container may still be pulling the image)"]
        return blob.decode("utf-8", errors="replace").splitlines()

    def cancel(self, handle: JobHandle) -> None:
        from beta9.clients.gateway import StopTasksRequest

        from gpurunner.auth import beam as beam_auth

        with beam_auth.service_client() as service:
            try:
                res = service.gateway.stop_tasks(StopTasksRequest(task_ids=[handle.remote_id]))
            except Exception as e:
                raise BackendError(f"Beam stop_tasks failed: {e}") from e
        if not getattr(res, "ok", False):
            raise BackendError(f"Beam refused to stop {handle.remote_id}: {getattr(res, 'err_msg', '')}")

    def balance(self) -> BalanceReport:
        """What the **local ledger** says — Beam exposes no usage or balance API.

        This number is ours, not Beam's: runs are booked at their worst case and
        settled at the wall time the runner reported. It is what the spend guard
        actually enforces, so it is the number worth showing — clearly labelled as
        a local estimate, with the dashboard linked for the authoritative figure.
        """
        from gpurunner.auth import beam as beam_auth
        from gpurunner.core import budget

        beam_auth.discover_credentials()  # raises AuthError → CLI shows "не налаштовано"
        cap = monthly_budget()
        committed = budget.month_committed(self.name)
        entries = budget.month_entries(self.name)
        live = sum(1 for e in entries if e.get("actual") is None)
        detail = (
            f"ЛОКАЛЬНИЙ облік ({len(entries)} run(-ів) цього місяця"
            + (f", {live} ще не завершено — рахуються по стелі" if live else "")
            + f") · стеля ${cap:.2f} = безкоштовний кредит · API витрат Beam не віддає"
        )
        return BalanceReport(
            backend=self.name,
            available=max(0.0, cap - committed),
            unit="$",
            spent=committed,
            detail=detail,
            url="https://platform.beam.cloud/settings/billing",
        )

    def wait_for_log_line(
        self, handle: JobHandle, needle: str, *, timeout: int = 1800, poll: int = 60
    ) -> str | None:
        """Poll the volume-synced log until a line contains ``needle``. None on timeout."""
        start = time.monotonic()
        while True:
            with contextlib.suppress(BackendError):
                for line in self.logs(handle):
                    if needle in line:
                        return line
            if time.monotonic() - start > timeout:
                return None
            time.sleep(poll)

    # ---- plumbing ---------------------------------------------------------

    def _settle_budget(
        self, handle: JobHandle, raw: dict[str, Any] | None, task_state: str | None
    ) -> None:
        """Turn a finished run's booking into what it actually cost.

        Until this runs, the month's budget carries the run at ``timeout × rate`` —
        which is the point while it is alive, and pure over-counting once it is not.
        A terminal task with no ``elapsed_s`` (killed before the runner wrote one)
        keeps its worst case: better to over-count than to under-count real money.
        """
        from gpurunner.core import budget

        terminal_task = task_state in ("complete", "error", "timeout", "cancelled")
        state = str((raw or {}).get("status", "")).lower()
        if not terminal_task and state not in ("completed", "failed"):
            return
        elapsed = (raw or {}).get("elapsed_s")
        if elapsed is None:
            return
        rate = handle.params.get("_rate") or hourly_rate(handle.gpu, handle.params)
        if rate is None:
            return
        with contextlib.suppress(Exception):
            if not budget.is_settled(self.name, handle.id):
                budget.settle(self.name, handle.id, float(rate) * float(elapsed) / 3600.0)

    @staticmethod
    def _volume_of(handle: JobHandle) -> str:
        if not handle.volume_name:
            raise BackendError(f"handle {handle.id[:8]} has no Beam volume recorded")
        return handle.volume_name

    def _task_status(self, handle: JobHandle) -> str | None:
        """Beam's own view of the task: ``pending|running|complete|error|…``."""
        from gpurunner.auth import beam as beam_auth

        creds = beam_auth.discover_credentials()
        beam = beam_auth.import_sdk()
        try:
            task = beam.Client(token=creds.token).get_task_by_id(handle.remote_id)
            state = task.status()
        except Exception:
            return None
        return str(getattr(state, "value", state) or "").lower() or None

    def _upload_dir(self, local: Path, volume: str, slug: str) -> int:
        """Upload a directory into ``<volume>/<slug>/``, then verify it landed.

        Beam's own ``Beta9Handler.upload`` rewrites remote paths based on file
        suffixes and probes ``is_dir`` per file; we drive the multipart upload
        directly so the remote key is exactly ``<slug>/<relative path>``. As with
        the Lightning backend the prefix is re-listed afterwards — an input that
        silently never arrived must fail here, not three minutes into a GPU run.
        """
        from beta9.multipart import RemotePath, beta9_upload

        from gpurunner.auth import beam as beam_auth

        files = [p for p in sorted(local.rglob("*")) if p.is_file()]
        if not files:
            raise BackendError(f"input dir {local} has no files")

        self._ensure_volume(volume)
        with beam_auth.service_client() as service:
            for p in files:
                rel = p.relative_to(local).as_posix()
                try:
                    beta9_upload(
                        service=service.volume,
                        file_path=p,
                        remote_path=RemotePath("beam", volume, f"{slug}/{rel}"),
                    )
                except Exception as e:
                    raise BackendError(f"upload of {p} → {volume}/{slug}/{rel} failed: {e}") from e

        landed = {
            rel[len(f"{slug}/"):]
            for rel in self._list_volume(volume, prefix=slug)
            if rel.startswith(f"{slug}/")
        }
        missing = {p.relative_to(local).as_posix() for p in files} - landed
        if missing:
            raise BackendError(
                f"upload to {volume}/{slug} reported success but "
                f"{len(missing)} file(s) are not on the volume: {sorted(missing)[:5]}"
            )
        return len(files)

    def _ensure_volume(self, name: str) -> None:
        from gpurunner.auth import beam as beam_auth

        beam = beam_auth.import_sdk()
        try:
            vol = beam.Volume(name=name, mount_path=_IN_MOUNT)
            if not vol.get_or_create():
                raise BackendError(f"Beam volume {name!r} could not be created")
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"Beam volume {name!r} could not be created: {e}") from e

    def _list_volume(self, volume: str, prefix: str = "") -> list[str]:
        """Volume-relative paths of every file under ``prefix``, walked recursively.

        ``list_path`` lists **one directory**; it does not glob. Both ``…/**`` and
        ``…/*`` answer ``ok=True`` with an empty list, so the obvious one-shot
        listing silently reports an empty volume — which is exactly how the first
        smoke run fetched 0 files out of a volume that had three (2026-08-02).
        Beam's own ``Beta9Handler.list_dir`` passes ``/**``; do not copy it.
        """
        from beta9.clients.volume import ListPathRequest

        from gpurunner.auth import beam as beam_auth

        out: list[str] = []
        with beam_auth.service_client() as service:
            pending = [f"{volume}/{prefix}".rstrip("/")]
            seen: set[str] = set()
            while pending:
                path = pending.pop()
                if path in seen:
                    continue
                seen.add(path)
                try:
                    res = service.volume.list_path(ListPathRequest(path=path))
                except Exception as e:
                    raise BackendError(f"listing {path} failed: {e}") from e
                if not getattr(res, "ok", False):
                    # An empty/not-yet-created volume is not an error for our callers.
                    continue
                for entry in res.path_infos or []:
                    if entry.is_dir:
                        pending.append(f"{volume}/{entry.path}")
                    else:
                        out.append(entry.path)
        return sorted(out)

    def _download_bytes(self, volume: str, rel_path: str) -> bytes | None:
        """Fetch one file off a volume. ``None`` when it isn't there (yet)."""
        from beta9.multipart import RemotePath, beta9_download

        from gpurunner.auth import beam as beam_auth

        with tempfile.TemporaryDirectory(prefix="gpurunner-beam-dl-") as tmp:
            local = Path(tmp) / Path(rel_path).name
            with beam_auth.service_client() as service:
                try:
                    beta9_download(
                        service=service.volume,
                        remote_path=RemotePath("beam", volume, rel_path),
                        file_path=local,
                    )
                except Exception:
                    return None
            if not local.exists():
                return None
            return local.read_bytes()

    def _read_status(self, handle: JobHandle) -> dict[str, Any] | None:
        blob = self._download_bytes(self._volume_of(handle), "_status.json")
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
        "inputs": params.get("inputs") or {},
        "input_root": params.get("input_root"),
        "max_hours": float(params.get("max_hours") or DEFAULT_MAX_HOURS),
        "cpu": float(params.get("cpu") or DEFAULT_CPU),
        "memory": params.get("memory") or DEFAULT_MEMORY,
        "retries": int(params.get("retries") or 0),
        "data_volume": str(params.get("data_volume") or DEFAULT_DATA_VOLUME),
    }


@contextlib.contextmanager
def _chdir(path: str | Path):
    """Temporarily switch CWD — Beam resolves the handler relative to it."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _import_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("gpurunner_beam_entry", path)
    if spec is None or spec.loader is None:
        raise BackendError(f"could not load generated entry module {path}")
    module = importlib.util.module_from_spec(spec)
    # The module must be importable by name inside the container too.
    sys.modules["gpurunner_beam_entry"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise BackendError(f"generated Beam entry module failed to import: {e}") from e
    return module


def _image_spec_to_beam(image_spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    """``(python_packages, shell commands)`` for Beam's ``Image``.

    ``Image`` has no ``extra_index_url`` parameter (Modal's does), so a job that
    needs one — PaddleOCR pulls paddle from the CN index — gets its packages
    installed through an explicit ``pip install`` command instead. Beam's base
    image is debian-slim-like, so ``apt_packages`` are installed too (Kaggle's
    base image ships them; a fresh container does not).
    """
    packages = list(image_spec.get("pip_packages") or [])
    extra_index = image_spec.get("extra_index_url")
    commands: list[str] = []
    apt = list(image_spec.get("apt_packages") or [])
    if apt:
        commands.append("apt-get update && apt-get install -y " + " ".join(apt))
    if extra_index and packages:
        commands.append("pip install --extra-index-url " + extra_index + " " + " ".join(packages))
        packages = []
    return packages, commands


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _render_entry_module(
    body: str,
    *,
    job_name: str,
    handle_id: str,
    gpu: str,
    image_spec: dict[str, Any],
    inputs: list[str],
    out_volume: str,
    data_volume: str,
    opts: dict[str, Any],
) -> str:
    """Source of the temp module Beam imports, syncs and runs."""
    packages, commands = _image_spec_to_beam(image_spec)
    python_version = str(image_spec.get("python_version") or "3.12")
    if not python_version.startswith("python"):
        python_version = f"python{python_version}"

    # ``__RUNNER_SRC__`` is substituted LAST: the job body is arbitrary source and
    # could itself contain a ``__GPU__``-looking token, which an earlier
    # substitution would then mangle.
    return (
        _ENTRY_TEMPLATE.replace("__INPUTS__", json.dumps(json.dumps(inputs)))
        .replace("__APP_NAME__", json.dumps(f"gpurunner-{job_name}"[:60]))
        .replace("__TASK_NAME__", json.dumps(f"gpurunner-{job_name}-{handle_id[:8]}"[:60]))
        .replace("__PYTHON_VERSION__", json.dumps(python_version))
        .replace("__PACKAGES__", json.dumps(packages))
        .replace("__COMMANDS__", json.dumps(commands))
        .replace("__GPU__", json.dumps(gpu))
        .replace("__GPU_COUNT__", json.dumps(0 if not gpu else 1))
        .replace("__CPU__", json.dumps(opts["cpu"]))
        .replace("__MEMORY__", json.dumps(opts["memory"]))
        .replace("__TIMEOUT__", json.dumps(int(opts["max_hours"] * 3600)))
        .replace("__RETRIES__", json.dumps(int(opts["retries"])))
        .replace("__OUT_VOLUME__", json.dumps(out_volume))
        .replace("__DATA_VOLUME__", json.dumps(data_volume))
        .replace("__OUT_MOUNT__", json.dumps(_OUT_MOUNT))
        .replace("__IN_MOUNT__", json.dumps(_IN_MOUNT))
        # 🔴 ensure_ascii=False обов'язковий: json.dumps екранує символ поза BMP
        # сурогатною парою ("\\ud83d\\udcbe"), а Python-ЛІТЕРАЛ, на відміну від
        # json.loads, назад її не склеює — у контейнері лишаються два непарні
        # сурогати, і перший же print із таким рядком валить run на
        # UnicodeEncodeError('surrogates not allowed'). Ціна помилки — повний
        # цикл сабміту (htr_case, 9 хв), а причина в трейсбеку виглядає як
        # проблема раннера, а не бекенда.
        .replace("__RUNNER_SRC__", json.dumps(body, ensure_ascii=False))
    )


#: Generated into a temp dir and imported by ``submit()``; Beam syncs the file and
#: runs ``entry:execute`` in the container. The heartbeat/status contract matches
#: the Colab, Vast and Lightning backends, so `status` reads them all alike.
_ENTRY_TEMPLATE = '''\
"""Generated by gpurunner — do not edit."""

import json
import os
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from beam import Image, Volume, task_queue

INPUTS = json.loads(__INPUTS__)
RUNNER_SRC = __RUNNER_SRC__
OUT_MOUNT = __OUT_MOUNT__
IN_MOUNT = __IN_MOUNT__

image = (
    Image(python_version=__PYTHON_VERSION__, python_packages=__PACKAGES__)
    .add_commands(__COMMANDS__)
)


@task_queue(
    app=__APP_NAME__,
    name=__TASK_NAME__,
    cpu=__CPU__,
    memory=__MEMORY__,
    gpu=__GPU__,
    gpu_count=__GPU_COUNT__,
    image=image,
    timeout=__TIMEOUT__,
    retries=__RETRIES__,
    volumes=[
        Volume(name=__OUT_VOLUME__, mount_path=OUT_MOUNT),
        Volume(name=__DATA_VOLUME__, mount_path=IN_MOUNT),
    ],
)
def execute():
    started = time.time()
    # Resolved once, up front: the mount is relative to the container's working
    # dir, and the job body is free to chdir wherever it likes afterwards.
    out_root = Path(OUT_MOUNT).resolve()
    in_root = Path(IN_MOUNT).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print("[gpurunner] outputs → %s" % out_root, flush=True)
    status_path = out_root / "_status.json"
    log_path = out_root / "_runner.log"
    out_dir = out_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    finalized = []
    synced = {}

    def now():
        return datetime.now(tz=timezone.utc).isoformat()

    def write_status(status, **extra):
        """Publish job state; terminal states latch so a late heartbeat can't undo them.

        `elapsed_s` is what the local spend ledger settles the run's booking with —
        without it every run would keep costing its worst case on paper.
        """
        with lock:
            if finalized:
                return
            if status in ("completed", "failed"):
                finalized.append(status)
            payload = {"status": status, "ts": now(), "heartbeat": now(),
                       "elapsed_s": round(time.time() - started, 1)}
            payload.update(extra)
            status_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    write_status("starting", message="preparing filesystem")

    # Jobs render code that hardcodes /kaggle/input and /kaggle/working. Beam
    # containers run as root, so the real paths are available; fall back to a
    # home-relative tree (and rewrite the job source) if that ever changes.
    kaggle_root = Path("/kaggle")
    try:
        kaggle_root.mkdir(parents=True, exist_ok=True)
        (kaggle_root / "probe").mkdir(exist_ok=True)
        (kaggle_root / "probe").rmdir()
    except (PermissionError, OSError):
        kaggle_root = Path(os.path.expanduser("~")) / "gpurunner_kaggle"
        kaggle_root.mkdir(parents=True, exist_ok=True)
        print("[gpurunner] /kaggle not writable — using %s" % kaggle_root, flush=True)

    working = kaggle_root / "working"
    working.mkdir(parents=True, exist_ok=True)
    (kaggle_root / "input").mkdir(parents=True, exist_ok=True)

    # Inputs live on a network volume; jobs read them repeatedly, so copy to the
    # container's local disk first (same reason the Colab backend does it).
    for slug in INPUTS:
        src = in_root / slug
        dst = kaggle_root / "input" / slug
        if not src.is_dir():
            write_status("failed", error="missing input",
                         message="%s not found on the data volume" % src)
            raise RuntimeError("input %r is not on the data volume" % slug)
        dst.mkdir(parents=True, exist_ok=True)
        count = 0
        for item in sorted(src.rglob("*")):
            if not item.is_file():
                continue
            target = dst / item.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            count += 1
        if count == 0:
            write_status("failed", error="empty input",
                         message="%s has no files — was it uploaded?" % src)
            raise RuntimeError("input %r came back empty" % slug)
        print("[gpurunner] staged %d file(s) for %s" % (count, slug), flush=True)

    def sync_outputs():
        """Copy the working dir onto the output volume so partial results survive."""
        for src in sorted(working.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(working).as_posix()
            size = src.stat().st_size
            if synced.get(rel) == size:
                continue
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, target)
                synced[rel] = size
            except OSError as e:
                print("[gpurunner] sync of %s failed: %s" % (rel, e), flush=True)

    stop = threading.Event()

    class Tee:
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

    def heartbeat():
        last = time.time()
        while not stop.wait(60):
            try:
                write_status("running", message="in progress")
                if time.time() - last >= 600:
                    sync_outputs()
                    last = time.time()
            except Exception:
                pass

    beat = threading.Thread(target=heartbeat, daemon=True)
    beat.start()

    write_status("running", message="job started")
    src = RUNNER_SRC
    if str(kaggle_root) != "/kaggle":
        src = src.replace("/kaggle/", str(kaggle_root) + "/")

    real_out, real_err = sys.stdout, sys.stderr
    err = None
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        sys.stdout = Tee(real_out, log)
        sys.stderr = Tee(real_err, log)
        try:
            exec(compile(src, "<gpurunner_job>", "exec"), {"__name__": "__main__"})
        except BaseException as e:
            traceback.print_exc(file=sys.stderr)
            err = repr(e)
        finally:
            sys.stdout, sys.stderr = real_out, real_err
            stop.set()

    beat.join(timeout=90)
    sync_outputs()
    n = sum(1 for p in out_dir.rglob("*") if p.is_file())
    if err is None:
        write_status("completed", message="%d files in out/" % n)
        print("[gpurunner] DONE — %d files synced" % n, flush=True)
        return {"status": "completed", "files": n}

    write_status("failed", error=err, message="%d files in out/ (partial)" % n)
    print("[gpurunner] FAILED: %s" % err, flush=True)
    raise RuntimeError("gpurunner job failed: %s" % err)
'''
