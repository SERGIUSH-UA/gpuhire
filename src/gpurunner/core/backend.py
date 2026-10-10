"""Backend ABC: a remote-execution target (Kaggle, Modal, RunPod, …)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from .job import Job
from .models import BalanceReport, JobHandle, StatusReport


class AuthError(RuntimeError):
    """Credentials missing or invalid."""


class BackendError(RuntimeError):
    """Backend-side failure (API down, kernel push rejected, etc.)."""

    #: Класифікація для реєстру боксів (`core.boxes`). Заповнюється там, де
    #: причина відома напевно; порожньо — «не знаємо, кого звинувачувати».
    outcome: str = ""
    #: HTTP-статус і початок тіла відповіді. Їх дописує на екземпляр
    #: `VastBackend._request`; дефолти ті самі, з якими читачі беруть їх через
    #: `getattr(e, "status_code", None)` / `getattr(e, "body", "")`.
    status_code: int | None = None
    body: str = ""


class SshAuthRejected(BackendError):
    """Хост відхилив ключ.

    🔴 Окремий клас потрібен рівно заради одного: це НЕ «бокс ще
    вантажиться». Раніше обидва випадки згорталися в `BackendError`, і
    `_wait_for_ssh` чесно ретраїв відмову автентифікації 15 хвилин поспіль —
    усі 15 оплачених. Відмова ключа не розсмокчеться сама; це властивість
    хоста або образу, і єдина правильна реакція — гасити й брати інший.
    """

    outcome = "ssh_auth_denied"


class SearchQuotaExceeded(BackendError):
    """Денна квота рядків пошукової видачі провайдера вичерпана.

    🔴 Це не короткий rate-limit і не порожній ринок. Ринок ніхто не дивився:
    пошук відмовлено до `reset_at`, і чесна відповідь людині — «квота до HH:MM»,
    а не «придатних машин немає». Уже орендовані машини це не зачіпає.
    """

    outcome = "search_quota"

    def __init__(self, message: str, *, limit: int = 0, remaining: int = 0,
                 retry_after: float = 0.0, reset_at: float = 0.0) -> None:
        super().__init__(message)
        self.limit = limit
        self.remaining = remaining
        self.retry_after = retry_after
        self.reset_at = reset_at
        self.status_code = 429

    def as_dict(self) -> dict[str, float | int | str]:
        return {"error": "search_quota_exceeded", "message": str(self),
                "limit": self.limit, "remaining": self.remaining,
                "retry_after": round(self.retry_after, 1), "reset_at": round(self.reset_at, 1)}


class Backend(ABC):
    """Abstract execution target."""

    name: ClassVar[str]
    gpu_choices: ClassVar[tuple[str, ...]] = ()
    default_gpu: ClassVar[str] = ""

    @abstractmethod
    def check_auth(self) -> None:
        """Raise AuthError if credentials are missing/invalid."""

    @abstractmethod
    def submit(self, job: Job, params: dict[str, Any], *, gpu: str) -> JobHandle:
        """Package the job and start it on the remote. Returns one handle per shard.

        For multi-shard jobs, the caller submits N times and stitches results;
        this method always corresponds to one remote run.
        """

    @abstractmethod
    def status(self, handle: JobHandle) -> StatusReport:
        """Current status. Should be cheap to call repeatedly."""

    @abstractmethod
    def fetch_outputs(self, handle: JobHandle, out_dir: Path) -> list[Path]:
        """Download artifacts produced by the remote run. Returns local paths."""

    def balance(self) -> BalanceReport:
        """Credits / money left on this backend.

        Default reports "unknown" — override where the provider actually exposes it.
        Implementations may raise AuthError when credentials are missing; the CLI
        renders that as "not configured" rather than failing the whole table.
        """
        return BalanceReport(backend=self.name, detail="провайдер не віддає баланс через API")

    def cancel(self, handle: JobHandle) -> None:
        """Stop a running job. Default no-op (override if backend supports it)."""
        _ = handle

    def logs(self, handle: JobHandle) -> Iterable[str]:
        """Iterate over log lines (newest first or oldest first — backend's choice).

        Default returns nothing; override if backend exposes a log API.
        """
        _ = handle
        return ()
