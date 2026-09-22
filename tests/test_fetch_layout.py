"""Розкладка забраного результату: що куди лягає і що переживає переоренду."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpurunner.supervise.htr import _is_refreshable, _remap


def test_flatten_puts_texts_where_the_consumer_looks() -> None:
    """🔴 Раннер кладе `out/`, споживач шукає ПЛАСКУ теку прогону.

    Без розпластання `clan_hunt` бачить нуль хітів БЕЗ помилки — успішний
    дорогий захід виглядає як негативний результат.
    """
    out = Path("/repo/reports/htr/spr-114")
    assert _remap(Path("out/0001.txt"), out, flatten=True) == Path("0001.txt")
    assert _remap(Path("out/_htr_meta.json"), out, flatten=True) == Path("_htr_meta.json")


def test_voice_goes_to_a_sibling_directory() -> None:
    """Побічний голос — сестринська `<прогін>-diak_v4/`, а не підтека."""
    out = Path("/repo/reports/htr/spr-114")
    got = _remap(Path("out-diak_v4/0001.txt"), out, flatten=True)
    assert got == Path("..") / "spr-114-diak_v4" / "0001.txt"


def test_logs_stay_inside_the_case() -> None:
    out = Path("/repo/reports/htr/spr-114")
    assert _remap(Path("logs/shard0.log"), out, flatten=True) == Path("logs/shard0.log")


def test_without_the_flag_nothing_moves() -> None:
    """Розкладка — угода з ЗАМОВНИКОМ, а не поведінка за замовчуванням."""
    out = Path("/x")
    assert _remap(Path("out/0001.txt"), out, flatten=False) == Path("out/0001.txt")


@pytest.mark.parametrize("name", ["htr_case_summary.json", "_htr_meta.part01.json"])
def test_mutable_files_are_refreshed_by_a_later_fetch(name: str) -> None:
    """🔴 Аварійний забір кладе ЧАСТКОВИЙ підсумок (`complete: false`).

    Правило «перший запис виграє» слушне для незмінних `*.txt` і хибне тут:
    старий підсумок заслоняв фінальний, і доведена до кінця справа
    оголошувалась неповною.
    """
    assert _is_refreshable(Path("/out") / name)


def test_page_texts_are_not_rewritten() -> None:
    assert not _is_refreshable(Path("/out/0001.txt"))


# ---- звідки забирати результат ------------------------------------------------


def test_fetch_prefers_r2_over_per_file_sftp() -> None:
    """🔴🔴 Вузьке місце забору — не смуга, а КРУГОВІ ОБЕРТИ.

    `_download_tree` тягне пофайлово, а одна справа це 2379 файлів на 30 МБ
    (тексти + рамки рядків + голос Дяка). На черзі з восьми це ~13 тисяч
    файлів: при 50-100 мс на обмін через океан самі лише оберти дають 11-22
    хвилини, скільки б не було мегабайтів. Заміряно 2026-08-12: «на диску нуль
    за 15 хвилин фази fetching», тоді як ручний забір тих самих даних через R2
    зайняв секунди.
    """
    from pathlib import Path as P

    src = (P(__file__).resolve().parents[1]
           / "src" / "gpurunner" / "supervise" / "htr.py").read_text(encoding="utf-8")
    i = src.index("def _fetch_queue")
    body = src[i:i + 2500]
    # Збірка з чекпоінтів винесена в `_fetch_from_store`: джерел тепер два
    # (бакет і склад на самій машині), а намір той самий — спершу один тарбол,
    # і лише потім пофайловий SFTP.
    assert "self._fetch_from_store(staging)" in body
    assert body.index("_fetch_from_store") < body.index("fetch_outputs"), (
        "збірка з чекпоінтів мусить іти ПЕРШОЮ, SFTP — запасним"
    )
    assert "only_meta=True" in body, "після неї по SFTP лишається лише службове"

    j = src.index("def _fetch_from_store")
    chooser = src[j:j + 800]
    assert "_fetch_via_r2" in chooser and "_fetch_via_box" in chooser, (
        "обидва джерела чекпоінтів мусять лишитись: бакет і склад на машині"
    )


def test_r2_failure_falls_back_to_sftp() -> None:
    """Якщо чекпоінтів немає — штатний забір мусить лишитись робочим."""
    from pathlib import Path as P

    src = (P(__file__).resolve().parents[1]
           / "src" / "gpurunner" / "supervise" / "htr.py").read_text(encoding="utf-8")
    i = src.index("def _fetch_queue")
    body = src[i:i + 2500]
    # 🔴 Перевіряємо НАМІР, а не форму рядка: гілка без `only_meta` мусить
    # лишитись, бо саме вона тягне ВСЕ, коли чекпоінтів у R2 немає.
    assert "fetch_outputs(handle, staging)" in body, (
        "запасна гілка SFTP зникла — без чекпоінтів забирати стало нічим"
    )
    assert "FETCH_PHASE_MAX_SEC" in body, (
        "забір мусить мати стелю часу: бокс зникав посеред обміну, і наглядач "
        "висів на мертвому сокеті пів години"
    )
