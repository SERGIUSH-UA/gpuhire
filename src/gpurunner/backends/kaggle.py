"""KaggleBackend: submit/poll/fetch via the official ``kaggle`` Python API.

A Kaggle "kernel" is a notebook/script running on Kaggle infrastructure. The
flow for our purposes is:

  1. Build a folder with ``notebook.ipynb`` + ``kernel-metadata.json``.
  2. ``kaggle.api.kernels_push(folder, acc=<gpu>)`` uploads and queues a run.
  3. Poll ``kernels_status(kernel_ref)`` until status terminal.
  4. ``kernels_output(kernel_ref, path)`` downloads ``/kaggle/working/`` contents.

Limits to keep in mind (v0.1 doesn't auto-shard around them):
  - 12 h max per kernel run
  - 30 h GPU/week per account
  - Output cap ~20 GB
"""

from __future__ import annotations

import json
import re
import shlex
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import nbformat
from nbformat.v4 import new_code_cell, new_notebook

from gpurunner.config import data_dir
from gpurunner.core.backend import AuthError, Backend, BackendError
from gpurunner.core.job import Job
from gpurunner.core.models import BalanceReport, JobHandle, JobStatus, StatusReport

_STATUS_MAP: dict[str, JobStatus] = {
    # KernelWorkerStatus enum names (Kaggle SDK returns these as .name).
    "queued": JobStatus.QUEUED,
    "queueing": JobStatus.QUEUED,
    "running": JobStatus.RUNNING,
    "complete": JobStatus.COMPLETED,
    "completed": JobStatus.COMPLETED,
    "success": JobStatus.COMPLETED,
    "error": JobStatus.FAILED,
    "failed": JobStatus.FAILED,
    "cancel_acknowledged": JobStatus.CANCELLED,
    "cancelacknowledged": JobStatus.CANCELLED,
    "cancelled": JobStatus.CANCELLED,
    "cancel_requested": JobStatus.RUNNING,
    "cancelrequested": JobStatus.RUNNING,
}

_TERMINAL = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


class KaggleBackend(Backend):
    """Backend that runs jobs as Kaggle notebooks."""

    name: ClassVar[str] = "kaggle"
    gpu_choices: ClassVar[tuple[str, ...]] = ("none", "T4", "T4x2", "P100")
    default_gpu: ClassVar[str] = "T4"

    # Kaggle's ``machine_shape`` (passed as ``acc`` to kernels_push → request.machine_shape).
    # Valid values are a SERVER-side enum matching the CLI ``--accelerator`` flag:
    # NvidiaTeslaT4 / NvidiaTeslaP100 / NvidiaTeslaA100 / NvidiaL4 / NvidiaH100 / ...
    # ⚠️ Invalid strings (e.g. the old "GPU T4 x2") are silently rejected → kernel falls
    # back to the DEFAULT accelerator, which is P100 (long single-card queue). And acc=None
    # ALSO defaults to P100 — so we must pass an explicit valid name to actually get T4.
    # ⚠️ Dual T4 (T4 x2) is NOT available via API/CLI (kaggle-cli #589/#821 open) — only in
    # the interactive web editor. So "T4x2" maps to single T4 (still newer + larger pool +
    # shorter queue than P100).
    _GPU_TO_ACC: ClassVar[dict[str, str | None]] = {
        "none": None,
        "T4": "NvidiaTeslaT4",
        "T4x2": "NvidiaTeslaT4",   # dual T4 недоступний через API → single T4
        "P100": "NvidiaTeslaP100",
    }

    def __init__(self) -> None:
        self._api: Any | None = None
        self._username: str | None = None

    # ---- public API -------------------------------------------------------

    def check_auth(self) -> None:
        self._get_api()
        self._get_username()

    def submit(self, job: Job, params: dict[str, Any], *, gpu: str) -> JobHandle:
        if gpu not in self.gpu_choices:
            raise ValueError(
                f"Unsupported gpu '{gpu}' for kaggle backend. Choose from: {self.gpu_choices}"
            )
        if self.name not in job.supported_backends:
            raise BackendError(
                f"Job {job.name!r} does not declare 'kaggle' in supported_backends"
            )

        normalized = job.validate_params(params)
        handle = JobHandle(
            backend=self.name,
            remote_id="<pending>",  # filled in after push succeeds
            job_name=job.name,
            params=normalized,
            gpu=gpu,
        )

        slug = self._make_slug(job.name, handle.id)
        username = self._get_username()
        kernel_ref = f"{username}/{slug}"

        work_dir = self._submission_dir(handle.id)
        work_dir.mkdir(parents=True, exist_ok=True)

        notebook = self._build_notebook(job, normalized)
        nb_path = work_dir / "notebook.ipynb"
        with nb_path.open("w", encoding="utf-8") as f:
            nbformat.write(notebook, f)

        metadata = self._build_kernel_metadata(
            kernel_ref=kernel_ref,
            title=self._title(job, handle.id),
            gpu=gpu,
            code_file="notebook.ipynb",
            dataset_sources=job.dataset_sources(normalized),
            kernel_sources=job.kernel_sources(normalized),
        )
        (work_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        api = self._get_api()
        try:
            acc = self._GPU_TO_ACC[gpu]
            api.kernels_push(str(work_dir), acc=acc)
        except Exception as e:
            raise BackendError(f"Kaggle kernels_push failed: {e}") from e

        handle.remote_id = kernel_ref
        handle.status = JobStatus.QUEUED
        handle.updated_at = datetime.now(tz=UTC)
        return handle

    def status(self, handle: JobHandle) -> StatusReport:
        api = self._get_api()
        try:
            resp = api.kernels_status(handle.remote_id)
        except Exception as e:
            raise BackendError(f"Kaggle kernels_status failed: {e}") from e

        status_raw = getattr(resp, "status", None)
        # Kaggle SDK returns an enum (KernelWorkerStatus.RUNNING). Pull .name when
        # available, fall back to str(). Either way, normalize to lowercase.
        enum_name = getattr(status_raw, "name", None)
        raw = str(enum_name).lower() if enum_name is not None else str(status_raw or "").lower()

        message = (
            getattr(resp, "failure_message", None)
            or getattr(resp, "failureMessage", None)
            or getattr(resp, "message", None)
        )
        mapped = _STATUS_MAP.get(raw, JobStatus.UNKNOWN)
        return StatusReport(
            status=mapped,
            message=message,
            error=message if mapped == JobStatus.FAILED else None,
        )

    def fetch_outputs(self, handle: JobHandle, out_dir: Path) -> list[Path]:
        """Download ALL of /kaggle/working, paginating over the kernel-output API.

        The SDK's high-level ``kernels_output`` fetches only the FIRST page
        (~500 files) and silently drops the ``next_page_token`` — so any kernel
        producing >500 output files (e.g. a PaddleOCR shard of >250 pages, which
        emits 2 files/page) gets truncated. We loop the low-level
        ``list_kernel_session_output`` over its page token and download each file
        with retries (kaggleusercontent signed-URL GETs are flaky)."""
        import requests
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiListKernelSessionOutputRequest,
        )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        api = self._get_api()
        owner_slug, _, kernel_slug = handle.remote_id.partition("/")

        downloaded: list[Path] = []
        seen_tokens: set[str] = set()
        token: str | None = None
        try:
            with api.build_kaggle_client() as client:
                while True:
                    req = ApiListKernelSessionOutputRequest()
                    req.user_name = owner_slug
                    req.kernel_slug = kernel_slug
                    if token:
                        req.page_token = token
                    resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
                    for item in (resp.files or []):
                        dest = out_dir / item.file_name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        last_err: Exception | None = None
                        for attempt in range(1, 5):
                            try:
                                r = requests.get(item.url, stream=True, timeout=120)
                                r.raise_for_status()
                                with open(dest, "wb") as fh:
                                    for chunk in r.iter_content(1 << 16):
                                        fh.write(chunk)
                                downloaded.append(dest)
                                last_err = None
                                break
                            except Exception as e:
                                last_err = e
                                time.sleep(2 ** attempt)
                        if last_err is not None:
                            raise BackendError(
                                f"failed to download {item.file_name} after retries: {last_err}"
                            )
                    token = getattr(resp, "next_page_token", None)
                    # Guard against an API that echoes the same token forever.
                    if not token or token in seen_tokens:
                        break
                    seen_tokens.add(token)
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"Kaggle kernels_output (paginated) failed: {e}") from e
        return downloaded

    def balance(self) -> BalanceReport:
        """Kaggle's *AI* quota — the only quota the SDK exposes.

        The weekly **GPU** quota (30 h) exists nowhere in kagglesdk; it is rendered
        by the web UI only. What is available is the model-proxy / benchmarks daily
        quota in USD (``/api/v1/benchmarks/tasks/quota``), which is what the vision
        HTR runs burn — so it is worth reporting, clearly labelled as *not* GPU.
        """
        detail = "GPU-квота (30 год/тиждень) через API НЕ доступна — лише у вебі"
        available: float | None = None
        spent: float | None = None
        unit = ""
        try:
            from kagglesdk.benchmarks.types.benchmark_tasks_api_service import (
                ApiGetBenchmarkTaskQuotaRequest,
            )

            api = self._get_api()
            with api.build_kaggle_client() as client:
                resp = client.benchmarks.benchmark_tasks_api_client.get_benchmark_task_quota(
                    ApiGetBenchmarkTaskQuotaRequest()
                )
            used = float(getattr(resp, "daily_quota_used", 0) or 0)
            allowed = float(getattr(resp, "total_daily_quota_allowed", 0) or 0)
            available = max(0.0, allowed - used)
            spent = used
            unit = "$"
            detail = f"AI-квота (модель-проксі, НЕ GPU): {used:.2f}/{allowed:.2f} $ за добу · " + detail
        except Exception as e:  # old kagglesdk, no access, network — stay informative
            detail = f"AI-квоту дістати не вдалось ({type(e).__name__}) · " + detail
        return BalanceReport(
            backend=self.name,
            available=available,
            unit=unit,
            spent=spent,
            detail=detail,
            url="https://www.kaggle.com/code",
        )

    def cancel(self, handle: JobHandle) -> None:
        """Not possible with an API token — verified, not assumed (2026-07-21).

        ``kagglesdk`` *does* ship ``cancel_kernel_session``, but it needs a
        ``kernel_session_id`` that **no** endpoint returns for a kernel we pushed
        (``kernels_push`` gives ``kernel_id``, status gives only a status enum).
        Probing the endpoint with the kernel id and the version number both answer
        **403 Forbidden**, so it appears to be gated to the browser session anyway.
        """
        raise BackendError(
            "Kaggle не дає скасувати кернел через API "
            "(cancel_kernel_session вимагає kernel_session_id, якого жоден ендпоїнт "
            "не віддає; проба з kernel_id → 403). Зупини вручну: "
            f"https://kaggle.com/code/{handle.remote_id} → Stop session."
        )

    def logs(self, handle: JobHandle) -> list[str]:
        api = self._get_api()
        try:
            raw = api.kernels_logs(handle.remote_id)
        except Exception as e:
            raise BackendError(f"Kaggle kernels_logs failed: {e}") from e
        if raw is None:
            return []
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return str(raw).splitlines()

    # ---- dataset staging (анти version-race, інцидент із 41-ю версією датасету) ----
    #
    # Kaggle рапортує «Upload successful» ДО того, як нова версія датасету стає
    # current; кернел, засабмічений у це вікно, ПРИВ'ЯЗУЄТЬСЯ ДО СТАРОЇ версії і
    # мовчки тренується на старих даних (детермінований дубль попередніх ваг).
    # Тому staging = push + БЛОКУЮЧЕ чекання, доки listing поточної версії не
    # покаже залиті файли (ім'я + байт-розмір).

    def dataset_files(self, dataset_id: str) -> list[dict[str, Any]]:
        """Файли ПОТОЧНОЇ версії датасету: [{name, size}] (= `kaggle datasets files`).

        🔴 Лістинг ПАГІНОВАНИЙ, і дефолт `page_size=20` віддає рівно перші
        двадцять імен. Без обходу сторінок це не «неповний список», а тихо
        НЕПРАВИЛЬНА відповідь на питання «чи доїхав файл»: `dataset_wait_current`
        звіряв очікуване з першою сторінкою, не знаходив там нічого й чекав до
        самого таймауту. Заміряно 2026-08-13 на датасеті кадрів справи
        (362 JPG): заливка пройшла успішно, а виклик упав через 3600 с із
        «нова версія не стала current», перелічивши двадцять файлів із 362.
        Симптом брехливий удвічі — виглядає як проблема на боці Kaggle.
        """
        api = self._get_api()
        out: list[dict[str, Any]] = []
        token = None
        # Стеля сторінок — запобіжник від нескінченного циклу, якщо API почне
        # вертати той самий токен. 200×500 = 100k файлів, більше за будь-який
        # наш датасет (найбільший — 49 422 кропи корпусу).
        for _ in range(200):
            try:
                resp = (api.dataset_list_files(dataset_id, page_token=token, page_size=500)
                        if token else api.dataset_list_files(dataset_id, page_size=500))
            except TypeError:
                # Старий kaggle-клієнт без пагінації в сигнатурі — беремо що дає.
                resp = api.dataset_list_files(dataset_id)
                token = None
            except Exception as e:
                raise BackendError(f"Kaggle dataset_list_files failed: {e}") from e
            for f in getattr(resp, "files", None) or []:
                name = getattr(f, "name", None) or getattr(f, "ref", None) or str(f)
                size = None
                for attr in ("total_bytes", "totalBytes", "size"):
                    v = getattr(f, attr, None)
                    if isinstance(v, int):
                        size = v
                        break
                    if v is not None and str(v).isdigit():
                        size = int(v)
                        break
                out.append({"name": str(name), "size": size})
            token = (getattr(resp, "next_page_token", None)
                     or getattr(resp, "nextPageToken", None))
            if not token:
                break
        return out

    def _dataset_exists(self, dataset_id: str) -> bool:
        """Чи існує датасет (і чи маємо до нього доступ). 403/404 → вважаємо, що нема."""
        try:
            self._get_api().dataset_list_files(dataset_id)
            return True
        except Exception:
            return False

    def dataset_push(self, folder: Path, *, notes: str) -> str:
        """Залити теку як датасет (мусить містити dataset-metadata.json з id).

        Якщо датасету ще НЕМА — створює його (`dataset_create_new`), інакше додає нову
        версію. Повертає dataset id '<owner>/<slug>'. Заливає БЕЗ dir-mode розпакування:
        архіви (*.tgz) лишаються файлами (runner'и самі розпаковують).

        ⚠ Спостереження 2026-08-13, що суперечить рядку вище для **`.tar`**:
        залитий `spr22.tar` (362 JPG) з'явився в датасеті РОЗПАКОВАНИМ —
        `0001.JPG … 0362.JPG`, самого архіву в лістингу немає. Тобто платформа
        розпаковує принаймні цей формат сама, і звірка за іменем архіву
        (`_stage_dataset` будує `expect` з локальних імен) не справдиться
        ніколи. Для `.tgz` це НЕ переміряно — наші тренові джоби на ньому й досі
        покладаються на «лишається файлом», тому логіку тут не чіпаю: буде
        привід — міряти окремо, а не міняти наосліп.
        """
        folder = Path(folder)
        meta_p = folder / "dataset-metadata.json"
        if not meta_p.exists():
            raise BackendError(f"нема {meta_p} — тека не схожа на kaggle-датасет")
        ds_id = json.loads(meta_p.read_text(encoding="utf-8")).get("id")
        if not ds_id or "/" not in str(ds_id):
            raise BackendError("dataset-metadata.json без валідного id '<owner>/<slug>'")

        # 🔴 `dir_mode="skip"` (нижче) МОВЧКИ викидає підтеки: kaggle друкує
        # «Skipping folder: X» серед прогрес-барів і закінчує бадьорим
        # «upload прийнято», перелічивши лише файли верхнього рівня. Інцидент
        # 2026-08-02: корпус HTR (49 422 кропи в `images/` + два .txt) поїхав
        # на Kaggle як два текстові файли без жодної картинки, версія стала
        # current, і це виявилось би аж на треновому кернелі — після заливки,
        # черги і розпакування. Тому мовчазний пропуск перетворено на відмову:
        # хочеш залити дерево — спакуй його в архів (так і роблять усі наші
        # джоби, вони самі розпаковують *.tgz).
        subdirs = sorted(p.name for p in folder.iterdir()
                         if p.is_dir() and not p.name.startswith("."))
        if subdirs:
            raise BackendError(
                f"у теці датасету є підтеки, які Kaggle МОВЧКИ пропустить: "
                f"{', '.join(subdirs)}. Спакуй їх в архів "
                f"(tar -czf <ім'я>.tgz -C {folder} {' '.join(subdirs)} …) і "
                f"залий архів — runner'и розпаковують *.tgz самі."
            )
        api = self._get_api()

        # kaggle ≤2.1.2 будує sidecar-шлях резюмованої заливки як
        #   <temp>/.kaggle/uploads/<abs_path з os.sep→'_' та ':'→'_'>.json
        # (kaggle_api_extended.py, ResumableUploadContext._get_upload_file_path).
        # Замінюється ЛИШЕ os.path.sep — тож шлях зі скісними рисками ('E:/a/b')
        # на Windows лишає їх як є, і клієнт лізе в неіснуючу підтеку
        # 'uploads/E_/a/' → [Errno 2]. Рідні роздільники дають плоске ім'я.
        folder_arg = str(folder.resolve())

        if self._dataset_exists(str(ds_id)):
            try:
                api.dataset_create_version(
                    folder_arg, version_notes=notes, quiet=False, dir_mode="skip"
                )
            except Exception as e:
                raise BackendError(f"Kaggle dataset_create_version failed: {e}") from e
        else:
            # Новий slug: create_version віддав би 403 Forbidden (виглядає як брак
            # прав, а насправді «датасету нема») — тому створюємо явно.
            try:
                api.dataset_create_new(folder_arg, public=False, quiet=False, dir_mode="skip")
            except Exception as e:
                raise BackendError(
                    f"Kaggle dataset_create_new failed для нового датасету {ds_id}: {e}"
                ) from e
        return str(ds_id)

    def dataset_wait_current(
        self,
        dataset_id: str,
        expect: dict[str, int | None],
        *,
        timeout: int = 3600,
        poll: int = 30,
        on_tick: Any | None = None,
    ) -> None:
        """Блокується, доки current-версія датасету не міститиме ВСІ expect-файли.

        expect = {ім'я: байт-розмір | None}. Розмір звіряється, лише якщо заданий і
        listing віддає числовий. TimeoutError-еквівалент → BackendError."""
        start = time.monotonic()
        while True:
            try:
                files = {f["name"]: f["size"] for f in self.dataset_files(dataset_id)}
            except BackendError:
                files = {}
            ok = bool(expect) and all(
                n in files and (sz is None or files[n] is None or files[n] == sz)
                for n, sz in expect.items()
            )
            if ok:
                return
            if time.monotonic() - start > timeout:
                raise BackendError(
                    f"dataset {dataset_id}: нова версія не стала current за {timeout}s "
                    f"(очікував {sorted(expect)}, listing: {sorted(files)})"
                )
            if on_tick is not None:
                on_tick(files)
            time.sleep(poll)

    def wait_for_log_line(
        self,
        handle: JobHandle,
        needle: str,
        *,
        timeout: int = 1800,
        poll: int = 60,
    ) -> str | None:
        """Полить логи кернела, доки не з'явиться рядок із підрядком `needle`.

        Повертає знайдений рядок, або None по таймауту (логи в queued-фазі
        недоступні — BackendError ковтається і полінг триває)."""
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

    # ---- internals --------------------------------------------------------

    def _get_api(self) -> Any:
        if self._api is not None:
            return self._api
        try:
            from kaggle import KaggleApi
        except ImportError as e:
            raise AuthError(f"kaggle PyPI package not installed: {e}") from e
        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as e:
            raise AuthError(f"Kaggle authentication failed: {e}") from e
        self._api = api
        return api

    def _get_username(self) -> str:
        if self._username:
            return self._username
        api = self._get_api()
        # kaggle.json path: username is stored locally
        name = api.get_config_value("username")
        if not name:
            # KGAT path: derive from a kernels_list response. Empty for fresh
            # accounts, so fall back to the API client's user object.
            try:
                kernels = api.kernels_list(mine=True, page_size=1) or []
                if kernels:
                    ref = getattr(kernels[0], "ref", "") or ""
                    if "/" in ref:
                        name = ref.split("/", 1)[0]
            except Exception:
                name = None
        if not name:
            # Last-resort: ask the API client for whoami via competitions list,
            # which always echoes the authenticated user.
            try:
                comps = api.competitions_list(page=1) or []
                _ = comps  # not used; we just needed the auth probe
                name = api.get_config_value("username") or None
            except Exception:
                name = None
        if not name:
            raise AuthError(
                "Could not determine Kaggle username from auth. "
                "Submit a kernel via the Kaggle UI once, or use legacy "
                "kaggle.json which embeds the username."
            )
        self._username = name
        return name

    def _build_notebook(self, job: Job, params: dict[str, Any]) -> nbformat.NotebookNode:
        nb = new_notebook()
        nb.metadata.setdefault("kernelspec", {})
        nb.metadata["kernelspec"].update(
            {"display_name": "Python 3", "language": "python", "name": "python3"}
        )
        nb.metadata["language_info"] = {"name": "python", "version": "3.11"}

        reqs = job.requirements()
        if reqs:
            req_text = " ".join(shlex.quote(r) for r in reqs)
            cmd = (
                "print('[gpurunner] setup start', flush=True)\n"
                f"!pip install {req_text}\n"
                "print('[gpurunner] setup done', flush=True)"
            )
            nb.cells.append(new_code_cell(cmd))

        body = job.render_remote_code(params)
        body = "print('[gpurunner] runner start', flush=True)\n" + body
        nb.cells.append(new_code_cell(body))

        return nb

    def _build_kernel_metadata(
        self,
        *,
        kernel_ref: str,
        title: str,
        gpu: str,
        code_file: str,
        dataset_sources: list[str] | None = None,
        kernel_sources: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": kernel_ref,
            "title": title,
            "code_file": code_file,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true" if gpu != "none" else "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": list(dataset_sources or []),
            "competition_sources": [],
            "kernel_sources": list(kernel_sources or []),
            "model_sources": [],
        }

    def _make_slug(self, job_name: str, handle_id: str) -> str:
        # Kaggle slug rules: lowercase, hyphens, max 50 chars.
        norm = unicodedata.normalize("NFKD", job_name).encode("ascii", "ignore").decode("ascii")
        norm = re.sub(r"[^a-zA-Z0-9]+", "-", norm).strip("-").lower()
        prefix = norm[:32] or "gpurunner-job"
        return f"{prefix}-{handle_id[:8]}"

    def _title(self, job: Job, handle_id: str) -> str:
        # Kaggle warns (and may reject) when the title slugifies to something
        # different than the kernel slug. Easiest fix: make the title BE the
        # slug, so its own slugify is identity. Kaggle requires title length >= 5.
        return self._make_slug(job.name, handle_id)

    def _submission_dir(self, handle_id: str) -> Path:
        return data_dir() / "submissions" / handle_id

    def wait(
        self,
        handle: JobHandle,
        *,
        poll_interval: int = 30,
        timeout: int | None = None,
    ) -> StatusReport:
        """Poll until status is terminal. Returns the final StatusReport."""
        start = time.monotonic()
        while True:
            report = self.status(handle)
            if report.status in _TERMINAL:
                return report
            if timeout is not None and time.monotonic() - start > timeout:
                return report
            time.sleep(poll_interval)
