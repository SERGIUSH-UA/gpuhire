#!/usr/bin/env python3
"""Розділ версії з CHANGELOG.md — текст сторінки релізу на GitHub.

    python tools/release_notes.py 0.2.0     # або v0.2.0

🔴 Розділу немає — код виходу 1, а не порожній текст. Журнал забувають саме
тоді, коли реліз збирають поспіхом; порожня сторінка релізу виглядала б як
«нічого не змінилось». Тому цей скрипт стоїть воротами в `release.yml` ДО
збирання колеса: колесо на PyPI не видаляється.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"


def section(version: str, text: str) -> str:
    """Тіло розділу `## [версія] …` до наступного заголовка другого рівня."""
    version = version.removeprefix("v")
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", text,
                  flags=re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    body = section(argv[1], CHANGELOG.read_text(encoding="utf-8"))
    if not body:
        print(f"у CHANGELOG.md немає розділу [{argv[1].removeprefix('v')}] — "
              "реліз без журналу не збирається", file=sys.stderr)
        return 1
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
