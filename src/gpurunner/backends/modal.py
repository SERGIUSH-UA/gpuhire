"""ModalBackend — submit/poll/fetch jobs as Modal Function calls.

Flow:
    submit():
      1. Build a ``modal.Image`` from ``job.modal_image_spec()``.
      2. Create (or reuse) a ``modal.Volume`` for the handle's outputs.
      3. Define an ephemeral ``modal.App`` with one function ``execute`` that
         mounts the volume at ``/mnt/outputs``.
      4. ``with app.run(detach=True): call = execute.spawn(params, runner_src, volume_name)``.
      5. ``call.object_id`` and ``volume_name`` are stored on the handle.

    status():
      ``modal.FunctionCall.from_id(remote_id).get(timeout=0)`` non-blocking.

    fetch_outputs():
      Read every file from the volume into ``out_dir`` (binary-safe, via
      ``volume.read_file``). Falls back to the legacy in-payload format for
      handles created before volume support (``handle.volume_name is None``).

    logs():
      Streams ``_runner.log`` from the volume (the wrapper tees runner stdout
      to that file).

Modal advantages over Kaggle for us:
  - no phone-verify gating internet egress
  - full network access in runtime → paddle 3.x / CN-hosted wheels work
  - $30/month free credit, pay-as-go ($0.59/h T4) after
  - cold start ~10 s vs Kaggle's ~minute of queue + ~3 min pip install
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any, ClassVar

from gpurunner.core.backend import AuthError, Backend, BackendError
from gpurunner.core.job import Job
from gpurunner.core.models import BalanceReport, JobHandle, JobStatus, StatusReport

# Modal accelerator-spec strings.  None ⇒ CPU container.
_GPU_MAP: dict[str, str | None] = {
    "none": None,
    "T4": "T4",
    "L4": "L4",
    "A10G": "A10G",
    "A100": "A100-40GB",
    "A100-80GB": "A100-80GB",
    "H100": "H100",
}

# $/год GPU (Modal sticker-тариф). Для наперед-прорахунку вартості run.
_GPU_HOURLY: dict[str, float] = {
    "none": 0.0, "T4": 0.59, "L4": 0.80, "A10G": 1.10,
    "A100": 2.10, "A100-80GB": 2.50, "H100": 3.95,
}
# CPU/RAM тарифи (per-second). Modal в'яже RAM до cores (~4 GiB/core, якщо memory не
# заданий). Реальний рахунок виходив на ~10-15% вище sticker (startup/overhead) → _FUDGE.
_CPU_CORE_PER_S: float = 0.0000131       # $/фіз.ядро/с
_RAM_GIB_PER_S: float = 0.00000222       # $/GiB/с
_RAM_GIB_PER_CORE: float = 4.0
_MODAL_DEFAULT_CORES: float = 2.0        # коли cpu не заданий
_BILLING_FUDGE: float = 1.12


def estimate_cost(job: Any, params: dict, gpu: str) -> dict | None:
    """Наперед-прорахунок вартості Modal-run: $/епоха, $/всього, год. None якщо
    немає тарифу/оцінки часу. Час бере з job.estimate_runtime (per-GPU it/s)."""
    gpu_hr = _GPU_HOURLY.get(gpu)
    if gpu_hr is None:
        return None
    try:
        secs = job.estimate_runtime({**params, "_gpu": gpu}).total_seconds()
    except Exception:
        return None
    cores = float(params.get("cpu") or _MODAL_DEFAULT_CORES)
    ram_gib = float(params.get("memory") or 0) / 1024.0 or cores * _RAM_GIB_PER_CORE
    per_s = gpu_hr / 3600.0 + cores * _CPU_CORE_PER_S + ram_gib * _RAM_GIB_PER_S
    total = per_s * secs * _BILLING_FUDGE
    try:
        epochs = max(1, int(params.get("epochs", 1)))
    except (TypeError, ValueError):
        epochs = 1
    return {"hours": secs / 3600.0, "total": total, "per_epoch": total / epochs,
            "gpu": gpu, "cores": cores, "ram_gib": ram_gib}


# Wrapper executed inside the Modal container. It:
#   1. exec()'s the runner source provided by the Job
#   2. points params["output_root"] at /mnt/outputs (the mounted volume)
#   3. tees runner stdout/stderr into /mnt/outputs/_runner.log (so logs survive)
#   4. invokes the runner's main(params) -> summary dict
#   5. commits the volume so fetch_outputs() sees the writes immediately
#   6. returns a tiny payload — files live on the volume, not in the return value
_REMOTE_WRAPPER = '''
def execute(params: dict, runner_src: str, volume_name: str) -> dict:
    import sys, threading, traceback
    from pathlib import Path

    output_root = Path("/mnt/outputs")
    output_root.mkdir(parents=True, exist_ok=True)

    params = dict(params)
    params["output_root"] = str(output_root)

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

    log_path = output_root / "_runner.log"
    real_out, real_err = sys.stdout, sys.stderr
    summary = None
    err_repr = None

    # 🔴 Періодичний commit тому — інакше `gpurunner logs` МОВЧИТЬ увесь прогін.
    # Том стає видимим клієнту лише після коміту, а той за замовчуванням
    # трапляється раз — на виході функції. Тобто протягом усього трену
    # `gpurunner logs` віддавав перший рядок і тишу, що зовні невідрізнимо від
    # зависання, і саме тоді лог найпотрібніший. Коміт кожні 45 с — інкрементний
    # і дешевий; демон-нитка, щоб не тримати shutdown інтерпретатора.
    # ⚠ Побічний ефект: між комітами на томі може лежати НЕДОПИСАНИЙ чекпойнт
    # (torch.save у процесі). Це нікому не шкодить — фінальний коміт на виході
    # кладе цілий файл, а `fetch` до завершення все одно не викликають.
    _stop_commit = threading.Event()

    def _commit_loop():
        try:
            import modal
            vol = modal.Volume.from_name(volume_name)
        except Exception:
            return
        while not _stop_commit.wait(45):
            try:
                vol.commit()
            except Exception:
                pass

    threading.Thread(target=_commit_loop, daemon=True).start()

    with open(log_path, "w", encoding="utf-8") as log:
        sys.stdout = _Tee(real_out, log)
        sys.stderr = _Tee(real_err, log)
        try:
            scope: dict = {}
            exec(compile(runner_src, "<gpurunner_runner>", "exec"), scope)
            if "main" not in scope:
                raise RuntimeError("runner_src must define main(params: dict) -> dict")
            summary = scope["main"](params)
        except BaseException as e:
            traceback.print_exc(file=sys.stderr)
            err_repr = repr(e)
        finally:
            sys.stdout = real_out
            sys.stderr = real_err

    _stop_commit.set()          # періодичний коміт більше не потрібен

    file_count = sum(1 for p in output_root.rglob("*") if p.is_file())

    try:
        import modal
        vol = modal.Volume.from_name(volume_name)
        vol.commit()
    except Exception:
        # Modal auto-commits on clean container exit; explicit commit is best-effort.
        pass

    if err_repr is not None:
        raise RuntimeError("gpurunner runner failed: " + err_repr)

    return {"summary": summary, "file_count": file_count, "volume_name": volume_name}
'''


def _volume_name_for(handle_id: str) -> str:
    """Modal volume name from a gpurunner handle id. Max 63 chars, DNS-safe."""
    # Handle ids are uuid4 hex (32 chars). Truncate generously to leave room for prefix.
    return f"gpurunner-out-{handle_id[:24]}"


class ModalBackend(Backend):
    """Run jobs on Modal Labs.

    Each ``submit()`` builds an ephemeral ``modal.App`` containing a single
    function. The function is spawned async; the ``object_id`` is stored as
    the handle's ``remote_id``. Outputs go to a per-handle ``modal.Volume``
    whose name is stored as ``handle.volume_name``.
    """

    name: ClassVar[str] = "modal"
    gpu_choices: ClassVar[tuple[str, ...]] = tuple(_GPU_MAP.keys())
    default_gpu: ClassVar[str] = "T4"

    # ---- public API -------------------------------------------------------

    def check_auth(self) -> None:
        config_path = Path.home() / ".modal.toml"
        if not config_path.exists():
            raise AuthError(
                f"No Modal config at {config_path}. "
                "Run: uv sync --extra modal && uv run modal token new"
            )
        try:
            import modal  # noqa: F401
        except ImportError as e:
            raise AuthError(
                f"modal package not installed. Run: uv sync --extra modal ({e})"
            ) from e

    def submit(self, job: Job, params: dict[str, Any], *, gpu: str) -> JobHandle:
        if gpu not in self.gpu_choices:
            raise ValueError(
                f"Unsupported gpu '{gpu}' for modal backend. Choose from: {self.gpu_choices}"
            )
        if self.name not in job.supported_backends:
            raise BackendError(
                f"Job {job.name!r} does not declare 'modal' in supported_backends"
            )

        normalized = job.validate_params(params)
        runner_src = job.render_runner_module()
        image_spec = job.modal_image_spec()

        # 🔴 Modal серіалізує обгортку між локальним і віддаленим інтерпретатором,
        # тож мінорні версії мусять збігатися. Пакет ставиться з 3.11, а образи
        # job'ів зібрані під 3.12 (paddle не має коліс під 3.13) — без цієї
        # перевірки розбіжність спливала б незрозумілою помилкою десеріалізації
        # вже в контейнері, тобто після збирання образу й старту білінгу.
        remote_py = str(image_spec.get("python_version", "3.12"))
        local_py = f"{sys.version_info.major}.{sys.version_info.minor}"
        if local_py != remote_py:
            raise BackendError(
                f"бекенд modal потребує локального Python {remote_py}, а зараз {local_py}: "
                f"Modal серіалізує обгортку між інтерпретаторами. Запустіть gpurunner з "
                f"оточення на {remote_py} (uv venv --python {remote_py}) або оберіть інший бекенд."
            )

        try:
            import modal
        except ImportError as e:
            raise AuthError(f"modal package not installed: {e}") from e

        # Pre-generate handle so we can name the volume from it.
        handle = JobHandle(
            backend=self.name,
            remote_id="<pending>",
            job_name=job.name,
            params=normalized,
            gpu=gpu,
        )
        volume_name = _volume_name_for(handle.id)

        # ---- build image ---------------------------------------------------
        try:
            image = modal.Image.debian_slim(
                python_version=remote_py
            )
            apt = image_spec.get("apt_packages") or []
            if apt:
                image = image.apt_install(*apt)
            pip_pkgs = image_spec.get("pip_packages") or []
            if pip_pkgs:
                kwargs: dict[str, Any] = {}
                extra_index = image_spec.get("extra_index_url")
                if extra_index:
                    kwargs["extra_index_url"] = extra_index
                image = image.pip_install(*pip_pkgs, **kwargs)
        except Exception as e:
            raise BackendError(f"failed to build Modal image: {e}") from e

        # ---- volume --------------------------------------------------------
        try:
            volume = modal.Volume.from_name(volume_name, create_if_missing=True)
        except Exception as e:
            raise BackendError(f"failed to create Modal volume {volume_name!r}: {e}") from e

        # ---- ephemeral app + function -------------------------------------
        app_name = f"gpurunner-{job.name}"[:63]
        app = modal.App(app_name)

        modal_gpu = _GPU_MAP[gpu]

        wrapper_scope: dict[str, Any] = {}
        exec(compile(_REMOTE_WRAPPER, "<gpurunner_wrapper>", "exec"), wrapper_scope)
        execute_fn = wrapper_scope["execute"]

        volumes: dict[str, Any] = {"/mnt/outputs": volume}
        for mount_path, vol_name in (job.modal_input_volumes(normalized) or {}).items():
            try:
                volumes[mount_path] = modal.Volume.from_name(vol_name)
            except Exception as e:
                raise BackendError(
                    f"failed to attach Modal input volume {vol_name!r} → {mount_path}: {e}"
                ) from e

        fn_kwargs: dict[str, Any] = {
            "image": image,
            "timeout": int(image_spec.get("timeout", 7200)),
            "volumes": volumes,
        }
        if modal_gpu is not None:
            fn_kwargs["gpu"] = modal_gpu
        # Явний запит CPU/RAM контейнера. Корисно лише для швидких GPU (A100+), де
        # dataloader-augmentation голодує GPU; на T4/L4 cpu↑ марний і дорожчає run
        # (тягне RAM). Тому OPT-IN: per-run params (`-p cpu=8`) мають пріоритет над
        # job-дефолтом image_spec; без жодного — Modal-авто (дешево).
        cpu = normalized.get("cpu") or image_spec.get("cpu")
        if cpu:
            fn_kwargs["cpu"] = float(cpu)
        memory = normalized.get("memory") or image_spec.get("memory")
        if memory:
            fn_kwargs["memory"] = int(memory)

        secret_names = job.modal_secrets(normalized)
        if secret_names:
            try:
                fn_kwargs["secrets"] = [modal.Secret.from_name(s) for s in secret_names]
            except Exception as e:
                raise BackendError(
                    f"failed to resolve Modal secrets {secret_names}: {e}"
                ) from e

        decorated = app.function(**fn_kwargs)(execute_fn)

        # ---- spawn ---------------------------------------------------------
        try:
            output_cm: Any = modal.enable_output()
        except Exception:
            from contextlib import nullcontext

            output_cm = nullcontext()
        try:
            with output_cm, app.run(detach=True):
                call = decorated.spawn(normalized, runner_src, volume_name)
                call_id = call.object_id
        except Exception as e:
            raise BackendError(f"Modal spawn failed: {e}") from e

        handle.remote_id = call_id
        handle.volume_name = volume_name
        return handle

    def status(self, handle: JobHandle) -> StatusReport:
        try:
            import modal
            import modal.exception as mexc
        except ImportError as e:
            raise AuthError(f"modal package not installed: {e}") from e

        try:
            call = modal.FunctionCall.from_id(handle.remote_id)
        except Exception as e:
            return StatusReport(status=JobStatus.UNKNOWN, error=f"from_id failed: {e}")

        try:
            call.get(timeout=0)
            return StatusReport(status=JobStatus.COMPLETED)
        except getattr(mexc, "OutputExpiredError", Exception) as e:
            return StatusReport(
                status=JobStatus.FAILED,
                error=f"output expired (Modal retains ~24h): {e}",
            )
        except TimeoutError:
            return StatusReport(status=JobStatus.RUNNING)
        except Exception as e:
            return StatusReport(status=JobStatus.FAILED, error=str(e))

    def balance(self) -> BalanceReport:
        """Month-to-date spend from ``modal.billing.workspace_billing_report``.

        Modal has no *remaining credit* endpoint, but it does expose a billing
        report (docs claim Team/Enterprise; it works on a personal workspace too).
        Spend-so-far against the monthly free credit is the actionable number.
        """
        import os
        from datetime import UTC, datetime

        # Modal has no remaining-credit endpoint, so the free monthly allowance is
        # a *constant* we subtract from: $30 on Starter, $100 on Team (modal.com/pricing,
        # checked 2026-07-21). Override when the plan changes — the number is an
        # estimate by construction and labelled as such.
        try:
            monthly_credit = float(os.environ.get("GPURUNNER_MODAL_MONTHLY_CREDIT", "30"))
        except ValueError:
            monthly_credit = 30.0

        detail = "залишку кредитів Modal API не віддає"
        spent: float | None = None
        available: float | None = None
        try:
            from modal.billing import workspace_billing_report

            now = datetime.now(tz=UTC)
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            rows = workspace_billing_report(start=start, end=now, resolution="d")
            # rows are plain dicts with Decimal costs
            spent = float(sum(float(r.get("cost") or 0) for r in rows))
            top = ""
            by_app: dict[str, float] = {}
            for r in rows:
                by_app[str(r.get("description") or "?")] = by_app.get(
                    str(r.get("description") or "?"), 0.0
                ) + float(r.get("cost") or 0)
            if by_app:
                name, cost = max(by_app.items(), key=lambda kv: kv[1])
                top = f" · найбільше: {name} ${cost:.2f}"
            available = max(0.0, monthly_credit - spent)
            detail = (
                f"ОЦІНКА = ${monthly_credit:.0f} безкоштовних/міс − витрати з початку місяця"
                f"{top} · точний залишок лише в дашборді"
            )
            if spent >= monthly_credit:
                detail = f"⚠ безкоштовний ліміт ${monthly_credit:.0f} вичерпано{top} · " + detail
        except Exception as e:
            detail = f"білінг-звіт недоступний ({type(e).__name__}) · " + detail
        return BalanceReport(
            backend=self.name,
            available=available,
            unit="$",
            spent=spent,
            detail=detail,
            url="https://modal.com/settings/usage",
        )

    def fetch_outputs(self, handle: JobHandle, out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if handle.volume_name:
            return self._fetch_from_volume(handle, out_dir)
        return self._fetch_from_payload(handle, out_dir)

    def cancel(self, handle: JobHandle) -> None:
        try:
            import modal
        except ImportError as e:
            raise AuthError(f"modal package not installed: {e}") from e

        try:
            call = modal.FunctionCall.from_id(handle.remote_id)
        except Exception as e:
            raise BackendError(f"Modal cancel: from_id failed for {handle.remote_id!r}: {e}") from e
        try:
            call.cancel()
        except Exception as e:
            raise BackendError(f"Modal cancel failed: {e}") from e

    def logs(self, handle: JobHandle) -> list[str]:
        if not handle.volume_name:
            return [
                "(no log volume associated with this handle — see Modal dashboard: "
                "https://modal.com/apps)"
            ]
        try:
            import modal
        except ImportError as e:
            raise AuthError(f"modal package not installed: {e}") from e
        try:
            vol = modal.Volume.from_name(handle.volume_name)
            with contextlib.suppress(Exception):
                vol.reload()  # runtime-only API; best-effort on client
            buf = bytearray()
            for chunk in vol.read_file("_runner.log"):
                buf.extend(chunk)
        except Exception as e:
            return [f"(could not read _runner.log from volume {handle.volume_name!r}: {e})"]
        return buf.decode("utf-8", errors="replace").splitlines()

    # ---- internals --------------------------------------------------------

    def _fetch_from_volume(self, handle: JobHandle, out_dir: Path) -> list[Path]:
        try:
            import modal
        except ImportError as e:
            raise AuthError(f"modal package not installed: {e}") from e

        try:
            vol = modal.Volume.from_name(handle.volume_name)  # type: ignore[arg-type]
        except Exception as e:
            raise BackendError(
                f"Modal volume {handle.volume_name!r} not accessible: {e}"
            ) from e
        # vol.reload() is a runtime-only API and raises on the client. We don't
        # need it for read-only fetches — Modal already commits the volume on
        # function exit, so listdir/read_file from the client see the latest
        # state.
        with contextlib.suppress(Exception):
            vol.reload()

        # Pull the function-call return value for the summary metadata. May
        # block briefly if the caller didn't wait; status() should be terminal
        # before calling this.
        summary_payload: dict[str, Any] = {}
        try:
            call = modal.FunctionCall.from_id(handle.remote_id)
            payload = call.get()
            if isinstance(payload, dict):
                summary_payload = payload
        except Exception:
            # The function may have raised; we still try to recover whatever
            # landed on the volume (partial output, _runner.log).
            pass

        written: list[Path] = []
        try:
            entries = vol.listdir("/", recursive=True)
        except Exception as e:
            raise BackendError(f"Modal volume listdir failed: {e}") from e

        import time
        skipped_files: list[tuple[str, str]] = []
        for entry in entries:
            # FileEntryType.FILE == 1 in modal_proto; safest check is `.type` name.
            type_name = getattr(getattr(entry, "type", None), "name", "").upper()
            if type_name and type_name != "FILE":
                continue
            rel = entry.path.lstrip("/")
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size > 0:
                # already on disk from a previous fetch attempt — skip
                written.append(target)
                continue
            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    with target.open("wb") as f:
                        for chunk in vol.read_file(entry.path):
                            f.write(chunk)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    # Transient connect/timeout errors — back off and retry.
                    if target.exists():
                        target.unlink(missing_ok=True)
                    time.sleep(2 ** attempt)
            if last_err is not None:
                skipped_files.append((entry.path, str(last_err)))
                continue
            written.append(target)
        if skipped_files:
            sp = out_dir / "_fetch_skipped.json"
            sp.write_text(
                json.dumps(
                    [{"path": p, "error": e} for p, e in skipped_files],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            written.append(sp)

        summary = summary_payload.get("summary")
        if summary is not None and not (out_dir / "_summary.json").exists():
            sp = out_dir / "_summary.json"
            sp.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            written.append(sp)

        return written

    def _fetch_from_payload(self, handle: JobHandle, out_dir: Path) -> list[Path]:
        """Legacy path: read files embedded in the FunctionCall return value.

        Used for handles created before volume support. Kept so existing
        manifest entries can still be fetched.
        """
        try:
            import modal
        except ImportError as e:
            raise AuthError(f"modal package not installed: {e}") from e

        try:
            call = modal.FunctionCall.from_id(handle.remote_id)
            payload = call.get()
        except Exception as e:
            raise BackendError(f"Modal fetch failed: {e}") from e

        if not isinstance(payload, dict):
            raise BackendError(
                f"Modal function returned {type(payload).__name__}, "
                "expected dict with keys 'summary' / 'files'."
            )

        written: list[Path] = []
        files = payload.get("files") or {}
        for rel_path, content in files.items():
            target = out_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, str):
                target.write_text(content, encoding="utf-8")
            elif isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(str(content), encoding="utf-8")
            written.append(target)

        summary = payload.get("summary")
        if summary is not None:
            sp = out_dir / "_summary.json"
            sp.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            written.append(sp)

        skipped = payload.get("files_skipped")
        if skipped:
            sp2 = out_dir / "_files_skipped.json"
            sp2.write_text(
                json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            written.append(sp2)

        return written

    def delete_volume(self, handle: JobHandle) -> None:
        """Drop the Modal volume associated with this handle.

        Not auto-invoked by ``fetch_outputs``; call explicitly when you're
        confident the local copy is intact. No-op for legacy handles without
        ``volume_name``.
        """
        if not handle.volume_name:
            return
        try:
            import modal
        except ImportError as e:
            raise AuthError(f"modal package not installed: {e}") from e
        try:
            modal.Volume.delete(handle.volume_name)
        except Exception as e:
            raise BackendError(
                f"failed to delete Modal volume {handle.volume_name!r}: {e}"
            ) from e
