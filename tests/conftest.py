"""Спільні помічники для тестів дашборду.

Прогони з керованим часом інакше не зробити: ``manifest.update`` штампує мітки
сам (і правильно робить — саме тому, що жоден виклик їх не проставляв, 287
хендлів мали ``updated_at == created_at``). Тож тест спершу створює прогін
штатним шляхом, а потім переписує мітки в журналі напряму.
"""

from __future__ import annotations

import os
import sqlite3
import time
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
    # 🔴 І до теки користувача (`%LOCALAPPDATA%\gpurunner`) тести не дістають:
    # там живе справжній наглядач (замки, журнал прогонів, стан сесій), а тест,
    # що тихо на неї спирається, зеленіє чи червоніє від даних машини, а не від
    # коду. Заміряно 23.09.2026: п'ять тестів її читали. Хто читає навмисно —
    # знімає змінну сам (`test_htr_sizing.py`). Фікстура `data_dir` ставить ту
    # саму змінну пізніше й перекриває цю.
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "user-data"))


class _NoSleepTime:
    """`time` без сну: решта атрибутів — справжні."""

    def __getattr__(self, name: str):
        return getattr(time, name)

    @staticmethod
    def sleep(_sec: float) -> None:
        return None


@pytest.fixture(autouse=True)
def _no_retry_pauses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Паузи між спробами спляться по-справжньому: R2 — 2+4+8+16 с, наглядач
    після відкинутої машини — 20 с. Сім тестів наскрізного шляху так і стояли
    ~155 с із 196 усього прогону.

    Підміна — лише `time` цих двох модулів, а не глобальний `time.sleep`:
    потоки й інші модулі сплять як раніше. Тест, якому потрібен свій `sleep`,
    ставить його поверх, як і досі (`monkeypatch.setattr(mod.time, "sleep", …)`).
    """
    from gpurunner.htr import r2
    from gpurunner.supervise import htr

    for mod in (r2, htr):
        monkeypatch.setattr(mod, "time", _NoSleepTime())


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


try:
    import xdist  # noqa: F401
except ImportError:  # xdist не стоїть — хук не оголошується, інакше pytest його відкидає
    pass
else:
    def pytest_xdist_auto_num_workers(config) -> int | None:
        """`-n auto` лише для пака: вузол чи один файл ідуть в одному процесі.

        Старт воркера — ще один інтерпретатор з імпортами (~6-8 с на Windows);
        для одного тесту це чисте очікування. `None` — хай xdist рахує ядра сам.
        """
        paths = [a for a in config.invocation_params.args if not a.startswith("-")]
        if any("::" in a for a in paths):
            return 0
        if len(paths) == 1 and paths[0].endswith(".py"):
            return 0
        return None
