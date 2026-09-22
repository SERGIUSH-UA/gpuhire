"""Приймачі релізного скрипта.

Реліз проходять рідко й під тиском, тож помилка в ньому не ловиться
повторенням. Тут перевіряється те, що не видно оком у момент випуску.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import release as rel

SRC = Path(rel.__file__)


def test_version_is_read_from_the_single_source() -> None:
    assert rel.version_of('__version__ = "1.2.3"\n') == "1.2.3"
    assert rel.version_of("# немає версії\n") == ""


@pytest.mark.parametrize(("version", "good"), [
    ("0.2.5", True), ("10.0.1", True),
    ("v0.2.5", False), ("0.2", False), ("0.2.5rc1", False), ("", False),
])
def test_only_plain_semver_is_accepted(version: str, good: bool) -> None:
    """`v` у номері — найчастіша описка: тег і версія пишуться по-різному."""
    assert rel.valid_version(version) is good


def test_changelog_section_matches_on_a_boundary() -> None:
    """🔴 Без межі `0.2.1` знайшлося б усередині `0.2.10`, і реліз поїхав би
    з чужими нотатками — а сторінку релізу читають як опис саме цієї версії."""
    text = "## [0.2.10] — 2026-09-30\n\n- щось\n"
    assert rel.changelog_has(text, "0.2.10")
    assert not rel.changelog_has(text, "0.2.1"), "збіг префікса не є розділом"

    assert rel.changelog_has("## [0.2.1] — 2026-09-21\n", "0.2.1")
    assert not rel.changelog_has("## [Unreleased]\n", "0.2.1")


def test_tag_and_version_differ_by_exactly_one_letter() -> None:
    assert rel.tag_of("0.2.5") == "v0.2.5"


def test_asking_pypi_can_answer_i_do_not_know(monkeypatch: pytest.MonkeyPatch) -> None:
    """🔴 Три стани, не два. «Спитати не вдалось», показане як «версії там
    немає», коштувало б спаленого номера: колесо з PyPI не видаляється."""
    def boom(*_a: object, **_k: object) -> None:
        raise OSError("мережі немає")

    monkeypatch.setattr(rel.urllib.request, "urlopen", boom)
    assert rel.on_pypi("0.2.5") is None


def test_dirty_paths_survive_cyrillic_names_and_renames() -> None:
    """🔴 Розбір іде по `-z`, бо звичайний `--porcelain` бере не-ASCII шлях у
    лапки з екрануванням (`"data/\320\260"`), і зліпок робочої теки дістав
    би неіснуючий файл. Справи тут звуться `spr-47а`, `spr-84г`."""
    z = " M src/a.py\0?? src/spr-47а.py\0 D src/gone.py\0R  new.py\0old.py\0"
    copy, drop = rel.dirty_paths(z)

    assert copy == ["src/a.py", "src/spr-47а.py", "new.py"]
    assert drop == ["src/gone.py", "old.py"], "стара назва перейменованого теж знімається"


def test_dirty_snapshot_is_read_only_for_the_repository() -> None:
    """🔴 Зліпок робиться копіюванням ФАЙЛІВ, а не через індекс.

    `git add`/`stash` у спільному дереві забрали б чужу незакомічену роботу —
    саме цього сторож не пускає в реліз, і тут те саме правило.
    """
    import inspect

    src = inspect.getsource(rel.apply_dirty)
    assert "shutil.copy2" in src
    assert not any(_destructive(args) for args in _git_calls(src))


def _git_calls(source: str) -> list[tuple[str, ...]]:
    """Аргументи-літерали кожного виклику `git(...)` у модулі.

    🔴 Розбір саме по AST, а не пошуком підрядка: цей файл і сам скрипт
    ЗГАДУЮТЬ заборонені команди в поясненнях, і сторож на `in text` червонів
    би від власного докстрінга. Сторож, який червоніє завжди, вимикають.
    """
    calls: list[tuple[str, ...]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Name) and fn.id == "git"):
            continue
        calls.append(tuple(a.value for a in node.args
                           if isinstance(a, ast.Constant) and isinstance(a.value, str)))
    return calls


def _destructive(args: tuple[str, ...]) -> str:
    """Чи чіпає цей виклик РОБОЧУ ТЕКУ. Порожньо — ні.

    ⚠ Дивимось на ПІДКОМАНДУ (перший аргумент), а не на набір слів: `git
    worktree add` теж містить «add», але створює окрему теку й нічого не
    забирає. Сторож, який ловить і його, змусив би себе послабити цілком.
    """
    if not args:
        return ""
    sub, rest = args[0], set(args[1:])
    if sub == "checkout" and "--orphan" in rest:
        return "checkout --orphan"
    if sub == "add":
        return "add"
    if sub == "reset" and "--hard" in rest:
        return "reset --hard"
    if sub in {"stash", "clean"}:
        return sub
    return ""


def test_release_never_touches_the_working_tree() -> None:
    """🔴 Приймач від повернення аварії 21.09.2026.

    Звичний спосіб зібрати orphan-гілку — `checkout --orphan` плюс `add -A` —
    забирає в коміт усе, що лежить у теці. Двічі за вечір це забрало чужу
    незакомічену роботу в сусідньому репозиторії, і двічі її діставали
    `reset`ом. Тут дерево береться `commit-tree <ref>^{tree}`, і робоча тека
    не читається взагалі.
    """
    bad = [f"git{args} → {why}"
           for args in _git_calls(SRC.read_text(encoding="utf-8"))
           if (why := _destructive(args))]
    assert not bad, "релізний скрипт чіпає робочу теку:\n  " + "\n  ".join(bad)


def test_public_is_built_from_a_commit_tree() -> None:
    """Зворотний бік попереднього: спосіб, який МАЄ бути, справді на місці."""
    src = SRC.read_text(encoding="utf-8")
    assert "commit-tree" in src
    assert "^{tree}" in src


def test_the_guard_would_catch_the_historical_form() -> None:
    """🔴 Перевірка самої перевірки: зелений сторож на живій ваді гірший за
    відсутній, бо переконує, що дивитись більше нема куди."""
    was = 'git("checkout", "--orphan", "public")\ngit("add", "-A")'
    whys = {_destructive(args) for args in _git_calls(was)}
    assert whys == {"checkout --orphan", "add"}

    now = 'git("commit-tree", tree, "-F", path)\ngit("branch", "-f", "public", c)'
    assert not any(_destructive(args) for args in _git_calls(now))


def test_distribution_name_is_not_the_command_name() -> None:
    """⚠ На PyPI пакет зветься інакше, ніж команда. Скрипт питає майданчик
    саме про дистрибутив; сплутавши їх, він щоразу казав би «версії немає»."""
    assert rel.DIST == "gpuhire"
    assert rel.MIRROR.endswith("/gpuhire")
