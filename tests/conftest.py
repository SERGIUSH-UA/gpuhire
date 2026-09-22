"""Спільні помічники для тестів дашборду.

Прогони з керованим часом інакше не зробити: ``manifest.update`` штампує мітки
сам (і правильно робить — саме тому, що жоден виклик їх не проставляв, 287
хендлів мали ``updated_at == created_at``). Тож тест спершу створює прогін
штатним шляхом, а потім переписує мітки в журналі напряму.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from gpurunner.core import manifest
from gpurunner.core.models import JobHandle, JobStatus


@pytest.fixture
def host_cores(monkeypatch: pytest.MonkeyPatch):
    """Скільки ядер «видно» раннерові — обома джерелами одразу.

    🔴 Підставити саме `os.cpu_count()` мало. `_usable_cores()` бере ще й
    `len(os.sched_getaffinity(0))`, а він існує лише на POSIX: на Windows
    гілка не виконується й підставлене число доживає до кінця, на Linux
    справжні два ядра раннера CI затирають його мовчки. Через це вісім
    тестів розкладки були зеленими на машині розробника й червоними в CI —
    і спіймалось це аж на тезі v0.2.0.

    Фікстура ставить обидва джерела разом і додає `sched_getaffinity` там,
    де його в `os` немає, — тобто гілка прив'язки перевіряється й на Windows.
    """
    def apply(n: int) -> None:
        monkeypatch.setattr(os, "cpu_count", lambda: n)
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(n)),
                            raising=False)
    return apply


@pytest.fixture(autouse=True)
def _never_write_into_the_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """🔴 Жоден тест не сміє писати в `data/` РЕПОЗИТОРІЮ.

    Там лежить знання, оплачене грішми: реєстр боксів і калібрування шардів.
    Спіймано одразу після появи другого — прогін тестів дописав туди 38 рядків
    про неіснуючу справу `spr-6671`, і калібрувальна вибірка почала брехати.
    Ізоляція стоїть тут, а не в кожному файлі, саме тому, що наступний, хто
    додасть запис у `registry_dir()`, про це правило не знатиме.
    """
    monkeypatch.setenv("GPURUNNER_REPO_DATA_DIR", str(tmp_path / "repo-data"))


@pytest.fixture
def data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Ізольовані ``runs.sqlite3`` / ``quota.sqlite3`` на один тест."""
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def now() -> datetime:
    """Справжній «зараз». ⚠ Тести, що рахують ПЕРІОДИ квот, мусять перекривати
    цю фікстуру прибитим моментом — інакше вони залежать від дня тижня; так і
    сталося в `test_quota.py`, див. коментар там."""
    return datetime.now(tz=UTC)


@pytest.fixture
def make_run(now: datetime):
    """Створити завершений прогін із заданими стартом і тривалістю."""
    counter = {"n": 0}

    def _make(
        *,
        starts_h_ago: float,
        duration_h: float,
        backend: str = "kaggle",
        gpu: str = "T4",
        job: str = "parseq_train",
        status: JobStatus = JobStatus.COMPLETED,
        output_dir: str | None = None,
    ) -> JobHandle:
        counter["n"] += 1
        handle = JobHandle(
            backend=backend, remote_id=f"me/kernel-{counter['n']}", job_name=job, gpu=gpu,
        )
        handle.created_at = now - timedelta(hours=starts_h_ago)
        handle.output_dir = output_dir
        manifest.add(handle)
        handle.status = JobStatus.RUNNING
        manifest.update(handle)
        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            handle.status = status
            manifest.update(handle)

        started = now - timedelta(hours=starts_h_ago)
        ended = started + timedelta(hours=duration_h)
        conn = sqlite3.connect(str(manifest.db_path()))
        try:
            rows = conn.execute(
                "SELECT seq FROM run_events WHERE run_id = ? ORDER BY seq", (handle.id,)
            ).fetchall()
            conn.execute("UPDATE run_events SET ts = ? WHERE seq = ?",
                         (started.isoformat(), rows[1][0]))
            if len(rows) > 2:
                conn.execute("UPDATE run_events SET ts = ? WHERE seq = ?",
                             (ended.isoformat(), rows[2][0]))
            conn.commit()
        finally:
            conn.close()
        return handle

    return _make
