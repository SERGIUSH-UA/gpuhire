"""Pydantic models shared by Job, Backend and the local manifest."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(StrEnum):
    """Lifecycle states for a remote job."""

    QUEUED = "queued"          # accepted by backend, not yet running
    RUNNING = "running"        # executing on remote worker
    COMPLETED = "completed"    # finished successfully
    FAILED = "failed"          # finished with error
    CANCELLED = "cancelled"    # manually stopped
    UNKNOWN = "unknown"        # transient or backend gave us no signal


def _new_id() -> str:
    """Short, sortable, URL-safe handle id. 22 chars of base62-ish uuid4."""
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class JobHandle(BaseModel):
    """Local reference to a single remote run.

    A logical job that needs to be sharded into multiple kernel runs (e.g.,
    a Kaggle OCR job that exceeds 12h limit) will produce multiple JobHandles,
    each with the same `parent_id`. Fetching/watching a `parent_id` aggregates
    children.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(default_factory=_new_id, description="Local handle id.")
    parent_id: str | None = Field(
        default=None,
        description="If this handle is a shard of a larger job, points to logical parent id.",
    )
    backend: str = Field(description="Backend name, e.g. 'kaggle', 'modal'.")
    remote_id: str = Field(
        description=(
            "Backend-specific identifier: kaggle kernel slug "
            "('user/slug-1234'), modal FunctionCall id, etc."
        )
    )
    job_name: str = Field(description="Name of the Job that produced this handle.")
    owner: str = Field(
        default="",
        description=(
            "Хто створив цей прогін. 🔴 Реєстр СПІЛЬНИЙ для всіх сесій на "
            "машині, і без цього поля «мій інстанс» не відрізнити від чужого: "
            "2026-08-11 паралельні агенти п'ять разів гасили одне одному живі "
            "бокси. Заповнюється з GPURUNNER_OWNER."
        ),
    )
    params: dict[str, Any] = Field(default_factory=dict, description="Params the job was submitted with.")
    gpu: str = Field(description="GPU/accelerator requested, e.g. 'T4', 'T4x2', 'A100'.")
    status: JobStatus = Field(default=JobStatus.QUEUED, description="Last-known status.")
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    output_dir: str | None = Field(
        default=None,
        description="Local dir where fetched outputs were written (set on fetch).",
    )
    volume_name: str | None = Field(
        default=None,
        description=(
            "Backend-managed durable-storage name (Modal: modal.Volume name). "
            "Absent for backends/handles that embed outputs in the return payload."
        ),
    )
    error: str | None = Field(default=None, description="Last-known error message, if any.")


class BalanceReport(BaseModel):
    """What a backend can tell us about money/quota left.

    ``available is None`` means the provider exposes no API for it (Kaggle's weekly
    GPU quota, Colab compute units, Modal credits) — the row still carries a URL so
    the answer is one click away instead of silently missing.
    """

    backend: str
    available: float | None = None          # credits / dollars left, if knowable
    unit: str = ""                          # "credits", "$", …
    spent: float | None = None              # lifetime or cycle spend, if exposed
    detail: str = ""                        # free-form note (plan, quota rules)
    url: str = ""                           # dashboard to check by hand


class StatusReport(BaseModel):
    """Snapshot of a job's current state, returned by Backend.status()."""

    status: JobStatus
    message: str | None = None
    progress_pct: float | None = None  # 0..100, if backend exposes it
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
