"""Довісок роботи на ЖИВИЙ бокс: черга росте, доки бокс іще наш.

Навіщо (ДОРОБКА 48): холодний старт коштує ≈8 хв оренди плюс очікування ринку,
а справа на 100 сторінок при 1700 стор/год — 3.5 хв роботи. Тобто накладні у
2.5 раза більші за саму роботу, і платяться щоразу, коли справа згадалась
пізніше.

🔴🔴 Найтонше місце тут — НЕ читання файла, а момент публікації терміналу.
Наглядач вважає `phase in ("done", "failed")` кінцем заходу: забирає результат і
гасить бокс. Доки останню справу черги позначали `_final_case=True`, довісок був
неможливий саме в найтиповішому випадку — «докинути малу справу, поки рахується
велика», бо велика і є остання. Тому термінал тепер публікує сам цикл, і лише
коли черга вичерпалась ПІСЛЯ чергового перечитування довіска.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gpurunner._embedded.htr_case_runner as runner


@pytest.fixture
def append_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "_append.jsonl"
    monkeypatch.setattr(runner, "APPEND_FILE", str(path))
    return path


def _write(path: Path, *cases: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")


# ---- саме читання довіска --------------------------------------------------


def test_missing_file_is_normal_not_an_error(append_file: Path) -> None:
    """Довіска немає в переважній більшості заходів — це тиша, а не збій."""
    assert not append_file.exists()
    assert runner._read_appended(set()) == []


def test_new_cases_are_picked_up(append_file: Path) -> None:
    _write(append_file, {"case": "230-1-50", "pages_url": "https://x/1"},
           {"case": "230-1-51", "pages_url": "https://x/2"})
    seen: set[str] = set()
    got = runner._read_appended(seen)
    assert [c["case"] for c in got] == ["230-1-50", "230-1-51"]
    assert seen == {"230-1-50", "230-1-51"}


def test_the_same_case_is_never_taken_twice(append_file: Path) -> None:
    """🔴 Ідемпотентність — це прямі гроші: повторно прочитана справа
    оплачується вдруге. Наглядач може перезапуститись, людина — натиснути
    двічі, тож дубль у файлі неминучий і мусить бути безпечним."""
    _write(append_file, {"case": "230-1-50"})
    seen: set[str] = set()
    assert len(runner._read_appended(seen)) == 1
    # той самий файл, друге читання між наступними справами
    assert runner._read_appended(seen) == []
    # і дубль, дописаний ще раз
    _write(append_file, {"case": "230-1-50"})
    assert runner._read_appended(seen) == []


def test_cases_already_in_the_plan_are_not_re_read(append_file: Path) -> None:
    """Довісок справи, яка вже в початковій черзі, теж дубль."""
    _write(append_file, {"case": "вже-в-плані"})
    assert runner._read_appended({"вже-в-плані"}) == []


def test_a_broken_line_does_not_kill_the_queue(append_file: Path, capsys) -> None:
    """⚠ Решта черги вже оплачена — губити її через зіпсовану кому безглуздо.
    Але й мовчати не можна: рядок мусить бути в лозі."""
    append_file.write_text(
        '{"case": "добра-1"}\n'
        "{зіпсований json\n"
        '{"case": ""}\n'
        "[1, 2, 3]\n"
        "\n"
        '{"case": "добра-2"}\n',
        encoding="utf-8")
    got = runner._read_appended(set())
    assert [c["case"] for c in got] == ["добра-1", "добра-2"]
    said = capsys.readouterr().out
    assert "не JSON" in said and "без імені справи" in said


# ---- поведінка черги -------------------------------------------------------


def _fake_case_run(calls: list, appends: dict):
    """Підміна `_run_case`: нічого не рахує, лише фіксує виклик і за потреби
    імітує довісок, що прилетів САМЕ ПІД ЧАС цієї справи."""
    def _run(merged):
        name = merged.get("case")
        calls.append({"case": name, "final": merged.get("_final_case"),
                      "index": merged.get("_case_index"),
                      "total": merged.get("_cases_total")})
        for case in appends.pop(name, []):
            _write(Path(runner.APPEND_FILE), case)
        return {"case": name, "complete": True, "n_pages_txt": 1}
    return _run


def test_a_case_appended_during_the_last_one_is_still_read(
    append_file: Path, monkeypatch
) -> None:
    """🔴🔴 Головний сценарій ДОРОБКИ 48 і найтиповіший у житті: докинути малу
    справу, поки рахується велика. Велика і є остання відома, тож саме тут
    старий код публікував `done`, наглядач гасив бокс, і довісок не встигав."""
    calls: list = []
    phases: list = []
    monkeypatch.setattr(runner, "_run_case", _fake_case_run(
        calls, {"велика": [{"case": "мала", "pages_url": "https://x/мала"}]}))
    monkeypatch.setattr(runner, "_set_phase",
                        lambda phase, **kw: phases.append((phase, kw)))

    out = runner._main_inner({"cases": [{"case": "велика"}]})

    assert [c["case"] for c in calls] == ["велика", "мала"], (
        "довісок, що прилетів під час останньої справи, не потрапив у чергу")
    assert out["queue"] == 2
    assert ("done", {"case_index": 2, "cases_total": 2}) in phases


def test_no_case_in_a_queue_declares_itself_final(
    append_file: Path, monkeypatch
) -> None:
    """Термінал публікує ЦИКЛ. Якщо його знову віддати останній справі, бокс
    гаситимуть до того, як довісок можна прочитати."""
    calls: list = []
    monkeypatch.setattr(runner, "_run_case", _fake_case_run(calls, {}))
    monkeypatch.setattr(runner, "_set_phase", lambda *a, **kw: None)

    runner._main_inner({"cases": [{"case": "а"}, {"case": "б"}]})

    assert [c["final"] for c in calls] == [False, False]


def test_terminal_phase_says_failed_when_a_case_is_incomplete(
    append_file: Path, monkeypatch
) -> None:
    """Неповна справа мусить лишити захід `failed` — інакше наглядач визнає
    неповний результат успіхом, а це той самий мовчазний клас втрат."""
    phases: list = []
    monkeypatch.setattr(runner, "_run_case",
                        lambda merged: {"case": merged.get("case"), "complete": False})
    monkeypatch.setattr(runner, "_set_phase",
                        lambda phase, **kw: phases.append(phase))

    with pytest.raises(RuntimeError, match="неповні"):
        runner._main_inner({"cases": [{"case": "а"}]})
    assert phases[-1] == "failed"


def test_single_case_run_is_untouched(monkeypatch) -> None:
    """Захід без черги йде старим шляхом і сам публікує термінал — довісок там
    не при чому, і ламати цю гілку не можна."""
    seen: list = []
    monkeypatch.setattr(runner, "_run_case", lambda p: seen.append(p) or {"complete": True})
    runner._main_inner({"case": "одна"})
    assert len(seen) == 1
    assert "_final_case" not in seen[0], "одиночний захід лишається фінальним за замовчуванням"
