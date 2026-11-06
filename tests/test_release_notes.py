"""Ворота релізу: журнал версії мусить існувати ДО збирання колеса."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("release_notes", ROOT / "tools" / "release_notes.py")
assert _spec is not None and _spec.loader is not None
notes = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = notes
_spec.loader.exec_module(notes)

LOG = """# Журнал змін

## [Unreleased]

## [0.3.0] — 2027-01-01

### Додано

- нове

## [0.2.0] — 2026-09-19

- старе
"""


def test_the_section_stops_at_the_next_version() -> None:
    body = notes.section("0.3.0", LOG)
    assert "нове" in body and "старе" not in body
    # тег приходить із `v`, журнал пишеться без нього
    assert notes.section("v0.2.0", LOG) == "- старе"


def test_a_missing_or_empty_section_fails_the_release(capsys) -> None:  # type: ignore[no-untyped-def]
    assert notes.section("9.9.9", LOG) == ""
    assert notes.section("Unreleased", LOG) == ""
    assert notes.main(["release_notes.py", "9.9.9"]) == 1
    assert "немає розділу" in capsys.readouterr().err


def test_the_current_version_has_its_section() -> None:
    """Версія пакета без розділу в журналі — реліз, який впаде на воротах."""
    from gpurunner import __version__

    assert notes.section(__version__, notes.CHANGELOG.read_text(encoding="utf-8"))
