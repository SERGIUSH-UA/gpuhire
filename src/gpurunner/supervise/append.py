"""Черга довіска: справи, дописані до ЖИВОГО заходу (ДОРОБКА 48).

Навіщо: холодний старт коштує ≈8 хв оренди плюс очікування ринку, а справа на
100 сторінок — 3.5 хв роботи. Накладні у 2.5 раза більші за саму роботу, і
платяться щоразу, коли справа згадалась пізніше.

## Чому саме файл-черга, а не прямий запис на бокс

Спокуса — дати команді `htr append` самій дописати рядок на бокс по SSH. Так
робити НЕ МОЖНА, і причина не в акуратності:

🔴 Забір результату йде `zip(plan.cases, state.cases)`. Справа, про яку знає
бокс, але не знає наглядач, порахується, приїде в стейджинг і **нікуди не
розкладеться** — оплачена й тихо втрачена. Рівно той клас мовчазних втрат, проти
якого писалась решта запобіжників.

Тому єдиний писар на бокс — наглядач. Команда лише кладе справу в цю чергу, а
наглядач на своєму тіку забирає її, вносить у план і в облік, і аж тоді штовхає
на бокс. Черга при цьому переживає і перезапуск команди, і смерть наглядача.

## Формат

JSONL, по справі на рядок. Дописування короткого рядка з ``O_APPEND`` атомарне,
тож читач ніколи не побачить половини запису; спільний JSON-масив довелося б
перечитувати-переписувати, і гонка з'явилася б на рівному місці.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpurunner.supervise.state import state_dir


def queue_path(session: str) -> Path:
    """Файл-черга заходу. Лежить поруч зі станом — те саме життя, той самий власник."""
    return state_dir() / f"{session}.append.jsonl"


def enqueue(session: str, case: dict[str, Any]) -> Path:
    """Покласти справу в чергу довіска.

    ⚠ Тут НЕМАЄ перевірки бюджету й строку. Вона свідомо лишається за
    наглядачем: між дописуванням і тіком минає час, за який захід міг з'їсти
    решту бюджету, тож остаточне слово мусить казати той, хто бачить стан у
    момент штовхання. Команда перевіряє те саме наперед — щоб людина почула
    відмову одразу, а не через хвилину.
    """
    name = str(case.get("case") or "").strip()
    if not name:
        raise ValueError("справа без імені: у черзі довіска вона нічого не означає")
    path = queue_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    return path


def drain(session: str, seen: set[str]) -> list[dict[str, Any]]:
    """Справи з черги, яких наглядач ще не бачив.

    🔴 Ідемпотентність за ІМЕНЕМ: повторний запис (людина натиснула двічі,
    команду перезапустили) не сміє додати справу вдруге — на боксі це прямі
    гроші за вже зроблене.

    ⚠ Файл НЕ чиститься: він і є журнал того, що просили довісити. Чистка
    зробила б чергу залежною від того, чи вижив наглядач між читанням і
    записом, — а це та сама втрата, лише в іншому місці.
    """
    path = queue_path(session)
    if not path.is_file():
        return []
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    fresh: list[dict[str, Any]] = []
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            case = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(case, dict):
            continue
        name = str(case.get("case") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        fresh.append(case)
    return fresh
