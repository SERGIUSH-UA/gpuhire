"""Чи є мережа В НАС — окремо від того, чи живий бокс і чи відповідає Vast.

🔴 Обрив домашнього інтернету наглядач читав як смерть машини або як відмову
Vast. 18.10.2026 (відгук користувачки, 0.33.1): година без мережі — і
наглядачі вийшли `finished · incomplete` з `getaddrinfo failed` на DELETE
інстансу, а три бокси лишились тарифікуватись. Гасіння, яке впало через НАШУ
мережу, не провал Vast: його треба повторити, щойно мережа повернеться.

Проба — розв'язати ім'я API. Це єдине, що відрізняє «лежимо ми» від «лежить
бокс»: SSH до боксу мовчить в обох випадках, а DNS — лише в першому.

`reachable`, `sleep` і `clock` беруться з модуля в момент виклику: їх
підміняють тести.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Callable
from typing import Any, TypeVar

#: Чиє ім'я розв'язувати. Саме його й не розв'язало в інциденті.
PROBE_HOST = "console.vast.ai"
PROBE_PORT = 443
#: Пауза між пробами й повторами росте від першої до стелі.
FIRST_PAUSE_SEC = 5.0
MAX_PAUSE_SEC = 60.0

sleep = time.sleep
clock = time.monotonic

T = TypeVar("T")


def reachable(host: str = PROBE_HOST, port: int = PROBE_PORT) -> bool:
    """Чи розв'язується ім'я API. Без запиту до самого API: він коштує квоту."""
    try:
        socket.getaddrinfo(host, port)
    except OSError:
        return False
    return True


def wait_for_network(deadline: float) -> bool:
    """Чекати, доки мережа повернеться, але не довше `deadline` (за `clock()`).

    True — мережа є; False — стеля вичерпана, а мережі так і нема.
    """
    pause = FIRST_PAUSE_SEC
    while True:
        if reachable():
            return True
        left = deadline - clock()
        if left <= 0:
            return False
        sleep(min(pause, left))
        pause = min(pause * 2, MAX_PAUSE_SEC)


def retry_through_outage(call: Callable[[], T], *, max_wait_sec: float,
                         on_wait: Callable[[BaseException], Any] | None = None) -> T:
    """`call()`, що переживає обрив НАШОЇ мережі.

    На `NetworkError` — чекати мережу й повторювати з паузою, що росте, доки
    не мине `max_wait_sec`. `on_wait` кличеться один раз, на першому обриві.
    Будь-яка інша помилка, як і вичерпана стеля, летить нагору як була.
    """
    from gpurunner.core.backend import NetworkError

    deadline = clock() + max_wait_sec
    pause = FIRST_PAUSE_SEC
    waited = False
    while True:
        try:
            return call()
        except NetworkError as e:
            if not waited:
                waited = True
                if on_wait is not None:
                    on_wait(e)
            if clock() >= deadline or not wait_for_network(deadline):
                raise
            # DNS відповідає, а запит усе одно не дійшов: мережа «блимає».
            # Пауза потрібна, щоб не молотити API впритул.
            sleep(min(pause, max(0.0, deadline - clock())))
            pause = min(pause * 2, MAX_PAUSE_SEC)
