"""Прохання до наглядача ЗГОРНУТИСЬ: спинити роботу, забрати, погасити.

Навіщо окремий сигнал, коли є `htr stop`. `stop` убиває сам наглядач — і
машина лишається горіти без нікого, хто її погасить: забору не було, звірки не
було, прочитане лежить на боксі, а лічильник іде до сторожа провайдера. Це
найдорожчий спосіб «зупинити захід», і саме він виглядає найприроднішим.

`quiesce` теж не згортання: він спиняє раннер НА БОКСІ, а наглядач про це не
знає й далі чекає прогресу — тобто машина тарифікується, доки не спрацює
детектор застрягання. Виміряно 20.09.2026: після `quiesce` захід лишався в
фазі «читає, прогрес свіжий» і не рухався.

Тому прохання кладеться туди, куди наглядач дивиться сам: файл поруч зі
станом. На черговому тіку він бачить його й іде тим самим шляхом, яким
завершує захід на стелі грошей — зупинка роботи, забір, звірка, гасіння,
вердикт. Жодного нового шляху виходу не з'являється; з'являється лише ще одна
причина піти вже наявним.

Файл, а не сигнал процесу: сигнал не переживає перезапуску наглядача й не
працює однаково на Windows і POSIX, а прохання мусить діяти й тоді, коли
наглядач саме перепідіймається після переоренди.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from gpurunner.supervise.state import state_dir


def request_path(session: str) -> Path:
    """Файл-прохання заходу. Поруч зі станом — те саме життя, той самий власник."""
    return state_dir() / f"{session}.wrapup.json"


def request(session: str, *, why: str = "") -> Path:
    """Попросити наглядача згорнути захід. Повертає шлях прохання.

    Повторне прохання нешкідливе: наглядач читає його раз і прибирає.
    """
    path = request_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"asked_at": round(time.time(), 3),
               "why": why or "попросили згорнути захід"}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def requested(session: str) -> dict[str, Any] | None:
    """Прохання, якщо воно є. Побите — теж прохання: людина його клала."""
    path = request_path(session)
    if not path.is_file():
        return None
    try:
        got = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"why": "попросили згорнути захід"}
    return got if isinstance(got, dict) else {"why": "попросили згорнути захід"}


def clear(session: str) -> None:
    """Прибрати прохання — щоб наступний захід тієї ж сесії його не побачив."""
    request_path(session).unlink(missing_ok=True)
