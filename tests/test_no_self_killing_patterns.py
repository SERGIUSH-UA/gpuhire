"""Жоден шаблон пошуку процесів не сміє збігатися сам із собою.

Убивство за шаблоном командного рядка знаходить і ту оболонку, у якій саме
виконується, тож вбиває СЕБЕ разом із ціллю: 15 із 16 спроб. Пастка
задокументована в проєкті давно — і це не врятувало: 04.09.2026 я наступив на
неї ДВІЧІ за одну сесію, маючи попередження перед очима й написавши поруч
власний коментар про неї.

Звідси й цей тест. Запис у пам'яті не є запобіжником: запобіжник — це або код,
який неможливо написати неправильно, або перевірка, яка падає. Тут друге.

Безпечна форма — символ у дужках: власний командний рядок тоді містить
`[h]tr…`, а регекс шукає `htr…` і в собі його не знаходить.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

#: Оболонкова форма: прапорець, а ЗА НИМ одразу лапка з шаблоном. Проза, що
#: згадує пастку, під це не підпадає — там після прапорця йде голий текст.
_SHELL = re.compile(r"""(?:pkill|pgrep)\s+-[a-zA-Z]*f\s+(["'])(.{0,40})""")

#: Спискова форма виклику з Python: ["pkill", "-f", "<шаблон>"].
_LIST = re.compile(
    r"""["'](?:pkill|pgrep)["']\s*,\s*["']-[a-zA-Z]*f["']\s*,\s*(["'])(.{0,40})""")

_BRACKETED = re.compile(r"\[[A-Za-z]\]")


def _offenders() -> list[str]:
    bad: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            for rx in (_SHELL, _LIST):
                for _, pattern in rx.findall(line):
                    if not _BRACKETED.search(pattern):
                        bad.append(f"{path.relative_to(SRC)}:{i}: {line.strip()[:90]}")
    return bad


def test_every_pattern_kill_is_written_so_it_cannot_match_itself() -> None:
    """Якщо колись з'явиться вбивство за голим іменем — тест упаде тут, а не на
    боксі посеред оплаченого прогону."""
    bad = _offenders()
    assert not bad, (
        "шаблон збігається сам із собою й уб'є власну оболонку; писати символ у "
        "дужках, напр. \"[h]tr_case_run.py\":\n  " + "\n  ".join(bad))


def test_the_guard_actually_catches_both_dangerous_forms() -> None:
    """🔴 Перевірка самої перевірки. Перший варіант цього тесту не ловив
    спискового виклику — тобто найімовірнішої форми в Python-коді, — і був би
    зеленим на справжній ваді."""
    unsafe_shell = 'run(\'pgrep -f "htr_case_run.py"\')'
    unsafe_list = 'subprocess.run(["pkill", "-f", "htr_case_run.py"])'
    safe_shell = 'run(\'pgrep -f "[h]tr_case_run.py"\')'
    safe_list = 'subprocess.run(["pkill", "-f", "[h]tr_case_run.py"])'
    prose = "Пастка: pkill -f htr_case_run вбиває власну оболонку."

    def flagged(line: str) -> bool:
        return any(not _BRACKETED.search(p)
                   for rx in (_SHELL, _LIST) for _, p in rx.findall(line))

    assert flagged(unsafe_shell), "оболонкова форма мусить ловитись"
    assert flagged(unsafe_list), "спискова форма мусить ловитись"
    assert not flagged(safe_shell)
    assert not flagged(safe_list)
    assert not flagged(prose), "проза про пастку — не сама пастка"


def test_a_supported_way_to_stop_the_runner_exists() -> None:
    """Приймач іншого боку: команда зупинки МУСИТЬ існувати. Інакше її знову
    напишуть на місці — і напишуть неправильно, як уже двічі й було."""
    cli = (SRC / "gpurunner" / "cli.py").read_text(encoding="utf-8")

    assert 'htr_app.command("quiesce")' in cli, (
        "немає штатної зупинки раннера на боксі — саме тому її пишуть руками")
    assert "[h]tr_case_run.py" in cli, "зупинка мусить уживати безпечний шаблон"
