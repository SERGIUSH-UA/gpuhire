"""Ворота повноти: неповний результат не має жодного шансу виглядати повним.

Числа в тестах — з реальних інцидентів: 203 з 323, 157 з 209, 16 і 46
сторінок, з'їдених CUDA OOM при `rc=0` і `failed_shards: []`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpurunner._embedded import htr_case_runner as runner
from gpurunner.jobs.htr_case import HTRCaseJob
from gpurunner.supervise.verify import verify_case


@pytest.fixture(autouse=True)
def _runner_prelude(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_embedded/_common.py` приклеюється до раннера тільки під час рендеру.

    У контейнері `_utc_iso` приїжджає звідти; при прямому імпорті модуля для
    тестів його немає. Підставляємо ту саму функцію, а не дублюємо її в
    раннері — інакше в контейнері жило б два визначення одного імені.
    """
    from gpurunner._embedded._common import _utc_iso

    monkeypatch.setattr(runner, "_utc_iso", _utc_iso, raising=False)


def make_out(tmp_path: Path, *, texts: int, summary: dict | None) -> Path:
    out = tmp_path / "fetched"
    (out / "out").mkdir(parents=True)
    for i in range(texts):
        (out / "out" / f"{i:04d}.txt").write_text("текст", encoding="utf-8")
    if summary is not None:
        (out / "htr_case_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False), encoding="utf-8"
        )
    return out


FULL = {
    "n_pages_input": 1665, "n_pages_total": 1665, "n_pages_expected": 1665,
    "n_pages_txt": 1665, "missing_pages": [], "quarantined_pages": [],
    "complete": True, "failed_shards": [], "oom_events": 0,
}


# ---- шар B: fetch не бере неповне ------------------------------------------


def test_complete_result_is_accepted(tmp_path: Path) -> None:
    out = make_out(tmp_path, texts=1665, summary=FULL)
    assert HTRCaseJob().is_output_complete(out)
    assert verify_case(out).complete


def test_one_missing_page_is_enough_to_reject(tmp_path: Path) -> None:
    """1664 з 1665 — це неповно. Без цього правила зникали десятки."""
    out = make_out(tmp_path, texts=1664, summary={**FULL, "n_pages_txt": 1664})
    assert not HTRCaseJob().is_output_complete(out)
    assert not verify_case(out).complete


def test_the_oom_signature_is_caught_even_without_new_fields(tmp_path: Path) -> None:
    """🔴 Точна ознака інциденту: шарди rc=0, `failed_shards` порожній,
    і єдина розбіжність — `n_pages_input` ≠ `n_pages_total`.

    Такий підсумок пише СТАРИЙ раннер, у якого полів повноти ще немає.
    """
    old_summary = {
        "n_pages_input": 209, "n_pages_total": 193,
        "failed_shards": [], "shard_rc": {1: 0, 2: 0, 3: 0},
    }
    out = make_out(tmp_path, texts=193, summary=old_summary)
    assert not HTRCaseJob().is_output_complete(out)
    res = verify_case(out)
    assert not res.complete
    assert "193" in res.detail


def test_runner_verdict_alone_is_enough_to_reject(tmp_path: Path) -> None:
    out = make_out(tmp_path, texts=1665, summary={**FULL, "complete": False})
    assert not HTRCaseJob().is_output_complete(out)


def test_named_missing_pages_reject(tmp_path: Path) -> None:
    out = make_out(tmp_path, texts=1663, summary={
        **FULL, "missing_pages": ["0042", "1301"], "complete": False,
    })
    res = verify_case(out)
    assert not res.complete
    assert res.missing == ["0042", "1301"]


def test_quarantine_without_text_is_a_hole_not_an_excuse(tmp_path: Path) -> None:
    """🔴🔴 Карантин НЕ доводить, що сторінка нечитабельна.

    Доти карантиновані віднімались від знаменника, тобто справа з дірою
    віддавала `complete: true`. Замір 18.08.2026 (ДАВіО): карантин з'їв п'ять
    сторінок, вердикт прийшов `ok`, а локально всі п'ять узялись З ПЕРШОГО
    РАЗУ. Карантин каже лише, що ЦЕЙ бокс не впорався під цим навантаженням.
    """
    out = make_out(tmp_path, texts=1664, summary={
        **FULL, "n_pages_txt": 1664, "quarantined_pages": ["9999"],
    })
    res = verify_case(out)

    assert not res.complete
    assert res.quarantined_empty == ["9999"]
    assert res.missing_count >= 1
    assert "БЕЗ тексту" in res.detail
    assert not HTRCaseJob().is_output_complete(out)


def test_quarantine_with_text_on_disk_is_not_a_loss(tmp_path: Path) -> None:
    """Зворотний бік: сторінка потрапила в карантин, але догінний прохід її
    таки прочитав. Доказ — текст на диску, і тоді діри немає."""
    out = make_out(tmp_path, texts=1665, summary={
        **FULL, "n_pages_txt": 1665, "quarantined_pages": ["0433"],
    })
    res = verify_case(out)

    assert res.complete, res.detail
    assert res.quarantined == ["0433"] and res.quarantined_empty == []
    assert HTRCaseJob().is_output_complete(out)


def test_failed_shard_rejects_even_with_all_texts(tmp_path: Path) -> None:
    out = make_out(tmp_path, texts=1665, summary={**FULL, "failed_shards": [3]})
    assert not verify_case(out).complete


def test_missing_summary_without_a_denominator_is_not_success(tmp_path: Path) -> None:
    """🔴 Нуль знаменника — це «нема з чим звіряти», а не «повно»."""
    out = make_out(tmp_path, texts=100, summary=None)
    res = verify_case(out)
    assert not res.complete
    assert res.source == "none"
    assert "НЕМА ЧИМ" in res.detail
    assert not HTRCaseJob().is_output_complete(out)


def test_plan_hint_works_when_the_summary_is_lost(tmp_path: Path) -> None:
    out = make_out(tmp_path, texts=323, summary=None)
    assert verify_case(out, expected_hint=323).complete
    assert not verify_case(out, expected_hint=400).complete


def test_truncated_fetch_is_caught(tmp_path: Path) -> None:
    """🔴 Інцидент: конвеєр забрав 203 сторінки з 323 і назвав це готовим."""
    out = make_out(tmp_path, texts=203, summary={
        **FULL, "n_pages_expected": 323, "n_pages_input": 323, "n_pages_total": 323,
        "n_pages_txt": 323,
    })
    res = verify_case(out)
    assert not res.complete
    assert "203" in res.detail
    assert res.missing_count == 120


def test_oom_events_are_surfaced_in_the_detail(tmp_path: Path) -> None:
    out = make_out(tmp_path, texts=1665, summary={**FULL, "oom_events": 3})
    assert verify_case(out).complete  # повно…
    assert "OOM-подій 3" in verify_case(out).detail  # …але слід видно


def test_broken_summary_is_not_complete(tmp_path: Path) -> None:
    out = make_out(tmp_path, texts=10, summary=None)
    (out / "htr_case_summary.json").write_text("{обірваний", encoding="utf-8")
    assert not HTRCaseJob().is_output_complete(out)


# ---- шар A: знаменник у самому раннері -------------------------------------


def test_runner_counts_missing_from_disk_not_from_meta(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    for name in ("0001", "0002", "0004"):
        (out / f"{name}.txt").write_text("x", encoding="utf-8")
    missing = runner._missing_pages(out, ["0001", "0002", "0003", "0004", "0005"])
    assert missing == ["0003", "0005"]


def test_runner_subtracts_quarantine(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "0001.txt").write_text("x", encoding="utf-8")
    runner._quarantine_add(out, "0002.jpg", "вотчдог: 10 хв тиші ×2")
    assert runner._missing_pages(out, ["0001", "0002"]) == []


def test_runner_page_order_matches_the_local_script(tmp_path: Path) -> None:
    """🔴 Індекси для догону мусять рахуватись у порядку самого скрипта:
    він бере `iterdir()` (без вкладених тек) і лише jpg/jpeg/png."""
    case = tmp_path / "case"
    (case / "sub").mkdir(parents=True)
    for name in ("0002.jpg", "0001.png", "0003.tif"):
        (case / name).write_text("", encoding="utf-8")
    (case / "sub" / "0004.jpg").write_text("", encoding="utf-8")
    assert [p.name for p in runner._script_pages(case)] == ["0001.png", "0002.jpg"]


def test_quarantine_file_is_readable_by_the_local_runner(tmp_path: Path) -> None:
    """Формат мусить збігатися з тим, що читає `htr_case_run.load_quarantine`."""
    out = tmp_path / "out"
    out.mkdir()
    runner._quarantine_add(out, "0042.jpg", "вотчдог")
    data = json.loads((out / "_htr_quarantine.json").read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert "0042.jpg" in data["pages"]
    assert set(data["pages"]["0042.jpg"]) == {"reason", "at"}


def test_done_sentinel_is_unique_per_run() -> None:
    """Маркер завершення не має збігатися з тим, що друкує кожен шард."""
    assert runner.DONE_SENTINEL.startswith("@@")
    assert "готово" not in runner.DONE_SENTINEL


@pytest.mark.parametrize(
    ("cards", "cores", "expected"),
    [
        # 06.09.2026: дефолт невідомого матеріалу 3.3 ГБ (як у ядрі, а не 2.5)
        ([32.0], 64, 8),           # одна карта 32 ГБ: 28.8 / 3.3
        ([16.0], 32, 4),
        ([23.0], 128, 6),
        ([4.0], 6, 1),
        ([24.0, 24.0], 128, 12),   # 🔴 дві карти — удвічі більше шардів
        ([24.0] * 4, 128, 24),     # чотири карти — учетверо, стеля 32 ще не тисне
        ([48.0] * 4, 128, 32),     # а тут уже тисне
    ],
)
def test_auto_shards_counts_every_card(
    monkeypatch: pytest.MonkeyPatch, host_cores, cards: list[float],
    cores: int, expected: int,
) -> None:
    """Формула в раннері дублює `core.htr_sizing` і мусить бачити ВСІ карти.

    🔴 Чесно це лише тому, що шарди розкладаються по картах (`cuda:(k % N)`).
    Поки всі сиділи на `cuda:0`, той самий підрахунок давав OOM.
    """
    monkeypatch.setattr(runner, "_free_vram_per_card", lambda: cards)
    host_cores(cores)
    assert runner._auto_shards() == expected
