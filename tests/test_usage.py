"""Тривалість прогону: чотири джерела, їхній пріоритет і різання по вікну."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gpurunner.core import manifest, usage
from gpurunner.core.models import JobHandle, JobStatus

pytestmark = pytest.mark.usefixtures("data_dir")


# ---- джерела ---------------------------------------------------------------


def test_journal_gives_the_duration(make_run, now: datetime) -> None:
    handle = make_run(starts_h_ago=5, duration_h=2.0)
    result = usage.run_usage(handle, now=now)
    assert result.source == usage.SRC_JOURNAL
    assert result.hours == pytest.approx(2.0, abs=0.01)
    assert result.exact


def test_summary_beats_the_journal(make_run, now: datetime, tmp_path: Path) -> None:
    """Ранер міряє себе точніше, ніж наше опитування ззовні.

    Журнал знає лише моменти, коли ми ПОБАЧИЛИ зміну статусу; при опитуванні раз
    на 30 с це похибка до півхвилини на кінець. ``elapsed_s`` із summary — те, що
    ранер заміряв усередині.
    """
    out = tmp_path / "out"
    out.mkdir()
    (out / "ptrain_summary.json").write_text(json.dumps({"wall_sec": 5400}), encoding="utf-8")
    handle = make_run(starts_h_ago=5, duration_h=2.0, output_dir=str(out))

    result = usage.run_usage(handle, now=now)
    assert result.source == usage.SRC_SUMMARY
    assert result.hours == pytest.approx(1.5, abs=0.01)   # 5400 с, а не 2 год із журналу


def test_summary_accepts_both_key_spellings(make_run, now: datetime, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "metrics.json").write_text(json.dumps({"elapsed_s": 3600}), encoding="utf-8")
    handle = make_run(starts_h_ago=5, duration_h=2.0, output_dir=str(out))
    assert usage.run_usage(handle, now=now).hours == pytest.approx(1.0, abs=0.01)


def test_never_polled_while_running_is_flagged(now: datetime) -> None:
    """Прогін, який жодного разу не бачили в стані RUNNING.

    Тоді єдине, що є, — «подали» і «скінчилось», а між ними ще й черга. Цифру
    рахуємо, але позначаємо: спиратись на неї не можна.
    """
    handle = JobHandle(backend="kaggle", remote_id="me/k", job_name="j", gpu="T4")
    handle.created_at = now - timedelta(hours=4)
    manifest.add(handle)
    handle.status = JobStatus.COMPLETED       # RUNNING ніхто не спостерігав
    manifest.update(handle)

    result = usage.run_usage(handle, now=now)
    assert result.source == usage.SRC_QUEUED
    assert not result.exact
    assert result.hours > 0


def test_unfinished_run_counts_up_to_now(make_run, now: datetime) -> None:
    handle = make_run(starts_h_ago=3, duration_h=0, status=JobStatus.RUNNING)
    result = usage.run_usage(handle, now=now)
    assert result.source == usage.SRC_RUNNING
    assert result.hours == pytest.approx(3.0, abs=0.01)


def test_abandoned_handle_is_not_counted_as_still_running(now: datetime) -> None:
    """Хендл, що застряг у ``queued`` два місяці тому, не «йде» 1600 годин.

    Знайдено на живих даних: 39 таких хендлів давали Kaggle **1106 витрачених
    годин за тиждень**, у якому їх фізично 168. Ознака смерті — не сам статус, а
    те, що ``updated_at`` давно не рухався: далі за останнє спостереження в нас
    немає даних, лише припущення.
    """
    handle = JobHandle(backend="kaggle", remote_id="me/zombie", job_name="j", gpu="T4")
    handle.created_at = handle.updated_at = now - timedelta(days=60)
    manifest.add(handle)

    result = usage.run_usage(handle, now=now)
    assert result.source == usage.SRC_STALE
    assert result.stale
    assert result.hours == pytest.approx(0.0, abs=0.01)   # а не 1440


def test_session_cap_bounds_a_corrupt_timestamp(now: datetime) -> None:
    """Kaggle обриває кернел на 12-й годині — більше означає зіпсовану мітку.

    Термінальний хендл, у якого між ``created_at`` і ``updated_at`` тижні
    (наприклад, скасований гуртом через місяць), інакше вніс би ці тижні у
    тижневу квоту.
    """
    handle = JobHandle(backend="kaggle", remote_id="me/old", job_name="j", gpu="T4")
    handle.created_at = now - timedelta(days=30)
    manifest.add(handle)
    handle.status = JobStatus.CANCELLED
    manifest.update(handle)

    assert usage.run_usage(handle, now=now).hours == usage.SESSION_CAP_H["kaggle"]


# ---- вікно -----------------------------------------------------------------


def test_run_outside_the_window_is_not_counted(make_run, now: datetime) -> None:
    make_run(starts_h_ago=100, duration_h=2.0)
    rows = usage.usage_in_window("kaggle", now - timedelta(hours=10), now)
    assert rows == []


def test_other_backends_are_not_counted(make_run, now: datetime) -> None:
    make_run(starts_h_ago=2, duration_h=1.0, backend="modal")
    assert usage.usage_in_window("kaggle", now - timedelta(hours=10), now) == []


def test_run_straddling_the_window_edge_is_clipped(make_run, now: datetime) -> None:
    """Половина трену до межі, половина після — зараховується лише друга.

    Це найважливіший випадок ручного звіряння: користувач дивиться у веб
    провайдера посеред 10-годинного трену. У побаченій ним цифрі вже враховано
    те, що трен спалив ДО цього моменту. Викинути прогін цілком — недорахувати;
    зарахувати цілком — відняти ті самі години двічі.
    """
    make_run(starts_h_ago=4, duration_h=4.0)
    rows = usage.usage_in_window("kaggle", now - timedelta(hours=2), now)
    assert len(rows) == 1
    assert rows[0].hours == pytest.approx(2.0, abs=0.02)
    assert rows[0].full_hours == pytest.approx(4.0, abs=0.02)
    assert rows[0].clipped


def test_clipping_scales_summary_time_proportionally(
    make_run, now: datetime, tmp_path: Path
) -> None:
    """У summary час менший за проміжок старт…кінець (черга й підняття туди не входять).

    Тому частка вікна застосовується пропорційно, а не як різниця міток — інакше
    обрізаний прогін давав би більше годин, ніж він узагалі тривав.
    """
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"elapsed_s": 3600}), encoding="utf-8")
    make_run(starts_h_ago=4, duration_h=4.0, output_dir=str(out))

    rows = usage.usage_in_window("kaggle", now - timedelta(hours=2), now)
    assert rows[0].hours == pytest.approx(0.5, abs=0.02)   # половина від 1 год, не 2


def test_totals_and_active_runs(make_run, now: datetime) -> None:
    make_run(starts_h_ago=3, duration_h=1.0)
    make_run(starts_h_ago=2, duration_h=0.5)
    live = make_run(starts_h_ago=1, duration_h=0, status=JobStatus.RUNNING)

    rows = usage.usage_in_window("kaggle", now - timedelta(hours=5), now)
    assert usage.total_hours(rows) == pytest.approx(2.5, abs=0.02)
    assert [h.id for h in usage.active_runs()] == [live.id]
