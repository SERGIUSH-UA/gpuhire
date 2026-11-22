"""Розкладка на боксі рахується від ПРОДАНИХ ядер, а не від видимих.

`os.cpu_count()` у контейнері показує хост. Замір 04.09.2026 на Tesla V100:
96 видимих ядер проти квоти 46.08. Рахувати на видимих означає дві тихі
помилки одразу — більший флот, ніж машина тягне, і більше потоків на шард, ніж
є ядер, — і жодна з них не подає себе як збій. Видно лише «повільно».

Раніше стелю `min(8, …)` це маскувало: 96÷8 = 12 обрізалось до 8 при бюджеті 5.
На боксі з меншою квотою маска зникає.
"""

from __future__ import annotations

import re
from pathlib import Path

import gpurunner._embedded.htr_case_runner as runner

SRC = Path(runner.__file__)


def test_the_quota_line_from_the_box_is_read_as_46_cores(tmp_path: Path) -> None:
    """Рівно той рядок, що лежав на боксі: 4608000/100000."""
    (tmp_path / "cpu.max").write_text("4608000 100000")
    assert runner._cgroup_quota_cores(tmp_path) == 46.08


def test_cgroup_v1_layout_is_read_too(tmp_path: Path) -> None:
    v1 = tmp_path / "cpu"
    v1.mkdir()
    (v1 / "cpu.cfs_quota_us").write_text("800000")
    (v1 / "cpu.cfs_period_us").write_text("100000")
    assert runner._cgroup_quota_cores(tmp_path) == 8.0


def test_no_quota_reads_as_no_quota(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("max 100000")
    assert runner._cgroup_quota_cores(tmp_path) is None
    assert runner._cgroup_quota_cores(tmp_path / "немає") is None


def test_usable_cores_never_exceeds_the_quota(monkeypatch, host_cores) -> None:
    host_cores(96)
    monkeypatch.setattr(runner, "_cgroup_quota_cores", lambda *a: 46.08)
    assert runner._usable_cores() == 46


def test_usable_cores_never_returns_zero(monkeypatch, host_cores) -> None:
    """Нуль ядер зупиняє прогін — це дорожче за будь-яку неточність."""
    host_cores(96)
    monkeypatch.setattr(runner, "_cgroup_quota_cores", lambda *a: 0.3)
    assert runner._usable_cores() == 1


def test_the_shard_fleet_is_not_planned_on_host_cores(monkeypatch, host_cores) -> None:
    """Ворота флоту: 96 видимих ядер не сміють купити більше шардів, ніж
    дозволяє квота на 46."""
    host_cores(96)
    monkeypatch.setattr(runner, "_free_vram_per_card", lambda: [400.0])  # VRAM не обмежує
    monkeypatch.setattr(runner, "_cgroup_quota_cores", lambda *a: 4.0)

    tight = runner._auto_shards(gb_per_shard=1.0)

    monkeypatch.setattr(runner, "_cgroup_quota_cores", lambda *a: 46.08)
    roomy = runner._auto_shards(gb_per_shard=1.0)

    assert tight < roomy, "квота не впливає на число шардів — рахується хост"
    assert tight <= 4 // runner.CORES_PER_SHARD + 1


#: Присвоєння видимих ядер у змінну — саме так стара вада й виглядала:
#: `cores = os.cpu_count() or 1`, а ділення стояло рядком нижче.
_ASSIGNED = re.compile(r"=\s*\(?\s*os\.cpu_count\(\)")

#: Арифметика чи порівняння прямо на видимих ядрах.
_ARITHMETIC = re.compile(r"os\.cpu_count\(\)\s*(?:or\s*\d+\s*)?\)?\s*(?://|/|\*|[<>]|-)")


def _sizing_uses_host_cores(line: str) -> bool:
    return bool(_ASSIGNED.search(line) or _ARITHMETIC.search(line))


def _usable_cores_span(src: str) -> tuple[int, int]:
    """Рядки самої `_usable_cores` — єдине місце, де видимі ядра читати можна."""
    lines = src.splitlines()
    start = next(i for i, s in enumerate(lines) if s.startswith("def _usable_cores"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("def "))
    return start, end


def test_no_sizing_decision_is_taken_on_raw_cpu_count() -> None:
    """🔴 Приймач від повернення вади. `os.cpu_count()` лишається дозволеним
    лише як ДІАГНОСТИКА (надрукувати, що видно хосту) і всередині самого
    `_usable_cores`. Будь-яка арифметика на ньому — та сама помилка, написана
    вдруге, і вона знову не подасть себе як збій."""
    src = SRC.read_text(encoding="utf-8")
    lo, hi = _usable_cores_span(src)
    bad = [f"{i}: {line.strip()[:90]}"
           for i, line in enumerate(src.splitlines())
           if not (lo <= i < hi) and _sizing_uses_host_cores(line)]
    assert not bad, (
        "розкладка рахується на видимих ядрах хоста; брати `_usable_cores()`:\n  "
        + "\n  ".join(bad))


def test_the_guard_catches_both_historical_forms_of_the_defect() -> None:
    """🔴 Перевірка самої перевірки. Перший варіант цього сторожа ловив лише
    ділення в один рядок і пропускав `cores = os.cpu_count() or 1` — тобто саме
    ту форму, що стояла в `_auto_shards`. Зелений сторож на живій ваді гірший
    за відсутній: він переконує, що дивитись більше нема куди."""
    was_in_auto_shards = "    cores = os.cpu_count() or 1"
    was_in_threads = "        threads_per_shard = max(1, min(8, (os.cpu_count() or 1) // shards))"
    fixed_shards = "    cores = _usable_cores()"
    fixed_threads = "        threads_per_shard = max(1, min(8, _usable_cores() // shards))"
    diagnostics = '          % (n_pages, shards, _gpu_name(), os.cpu_count()), flush=True)'
    report_field = '        "cpu_count": os.cpu_count(),'

    assert _sizing_uses_host_cores(was_in_auto_shards), "присвоєння мусить ловитись"
    assert _sizing_uses_host_cores(was_in_threads), "ділення мусить ловитись"
    assert not _sizing_uses_host_cores(fixed_shards)
    assert not _sizing_uses_host_cores(fixed_threads)
    assert not _sizing_uses_host_cores(diagnostics), "друк видимих ядер — не рішення"
    assert not _sizing_uses_host_cores(report_field), "поле звіту — не рішення"


def test_runner_constants_match_the_planner() -> None:
    """🔴 Дубль у боксі розійшовся з ядром: 2.5 ГБ / 2 ядра проти 3.3 / 1.
    Резервний шлях (`shards=auto` без ручки) рахував інший флот, ніж ворота."""
    from gpurunner.core import htr_sizing as hs

    assert runner.CORES_PER_SHARD == hs.MIN_CORES_PER_SHARD
    assert runner.GB_PER_SHARD == hs.GB_PER_SHARD
    assert runner.VRAM_HEADROOM == hs.VRAM_HEADROOM
    assert runner.MAX_SHARDS == hs.MAX_SHARDS
