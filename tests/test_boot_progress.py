"""Пороги підйому мусять дивитись на ПОСУВАННЯ, а не на секундомір.

🔴 Vast увесь час розповідає, що робить хост (`status_msg`: «Pulling from
pytorch/pytorch», «Downloading», «Extracting», «success»). Ми читали це лише
щоб надрукувати, а рішення ухвалювали за голим таймером — тож бокс, який
СУМЛІННО тягне образ, помирав за тим самим порогом, що й намертво завислий.
"""

from __future__ import annotations

from pathlib import Path

SRC = (Path(__file__).resolve().parents[1]
       / "src" / "gpurunner" / "backends" / "vast.py").read_text(encoding="utf-8")


def test_status_message_is_part_of_the_signature() -> None:
    assert "def _status_signature" in SRC
    assert 'inst.get("status_msg")' in SRC


def test_stuck_timer_restarts_when_the_host_moves() -> None:
    """Секундомір рахується від ОСТАННЬОЇ ЗМІНИ підпису, не від оренди."""
    assert "sig, sig_since = now_sig, time.monotonic()" in SRC
    assert "time.monotonic() - sig_since > _BOOT_STUCK_SEC" in SRC
    assert "time.monotonic() - start > _BOOT_STUCK_SEC" not in SRC


def test_the_verdict_quotes_what_vast_last_said() -> None:
    """Діагноз мусить нести останнє слово хоста — інакше причину не з'ясувати."""
    assert "останнє " in SRC and "повідомлення Vast" in SRC


def test_boot_and_ssh_thresholds_are_six_minutes() -> None:
    """Рішення користувача 2026-08-11 після того, як 180 с убили дві робочі
    машини: успішний бокс на спр.114 піднявся за 208 с."""
    assert "_BOOT_STUCK_SEC = 360" in SRC
    assert "_SSH_AFTER_RUNNING_SEC = 360" in SRC


def test_ssh_wait_also_watches_progress_not_just_the_clock() -> None:
    """🔴🔴 Найдорожча помилка дня, і вона була в ДРУГІЙ гілці таймера.

    Прогрес-обізнаність спершу додали лише для фази образу. Але Vast ставить
    `running`, щойно контейнер створено, а `onstart` у цей час ще ставить
    пакети — тобто найдовша частина підйому відбувається ВЖЕ в стані
    `running`, і секундомір `_SSH_AFTER_RUNNING_SEC` її не бачив.

    Ціна: убито машину 135791 — RTX 3090 з 80 ядрами за $0.017/год, тобто
    найкраще, що ринок дав за день (≈$0.006 за тисячу сторінок) — на 511-й
    секунді, коли в статусі йшов живий лічильник apt (`Get:85 …libstemmer0d`).
    """
    i = SRC.index("_SSH_AFTER_RUNNING_SEC:")
    window = SRC[max(0, i - 1500):i]
    assert "running_since = now" in window
    assert "_status_signature(handle)) != sig" in window, (
        "гілка очікування SSH мусить скидати секундомір на зміну статусу"
    )


def test_both_timers_share_one_progress_signal() -> None:
    """Обидві фази підйому міряються тим самим підписом — інакше одну з них
    неминуче забудуть при наступній правці."""
    assert SRC.count("_status_signature(handle)") >= 2


# ---- забір --------------------------------------------------------------------

FETCH_SRC = SRC  # той самий модуль


def test_fetch_has_a_read_timeout_and_keepalive() -> None:
    """🔴🔴 Найтихіша пастка системи: забір без таймаутів.

    Завмерле TCP-з'єднання посеред качання блокує `sftp.get()` НАЗАВЖДИ.
    Наглядач лишається живий, але стоїть у читанні: стан не оновлюється, бокс
    тарифікується, і зовні це виглядає як «працює». Заміряно 2026-08-12 ДВІЧІ
    за ніч у двох незалежних кампаніях — обидва наглядачі завмерли саме на
    `fetching` і простояли 4.4 і 4.8 години.
    """
    assert "set_keepalive(30)" in FETCH_SRC
    assert "chan.settimeout(FETCH_READ_TIMEOUT_SEC)" in FETCH_SRC


def test_fetch_has_an_overall_deadline_checked_between_files() -> None:
    """Стеля на весь забір, і перевіряється МІЖ ФАЙЛАМИ: краще віддати те, що
    встигли, ніж стояти в читанні, поки йде тарифікація."""
    assert "FETCH_DEADLINE_SEC" in FETCH_SRC
    assert "вичерпано стелю часу на забір" in FETCH_SRC


def test_the_brake_expires_by_itself() -> None:
    """🔴🔴 Стоп-кран мусить протухати, інакше він гірший за проблему.

    Наглядач ставить `NOSTOP` перед забором, щоб бокс не погасив себе посеред
    качання (самознищення рахує голий час: 30 хв після job'а, а стеля забору —
    година). Але наглядач тричі за добу помирав саме на заборі — і вічний кран
    означав би бокс, який не погасить себе НІКОЛИ («STILL BILLING» у самому
    шаблоні). Свіжий файл = «зараз хтось качає»; протухлий = «той, хто його
    поставив, більше не з нами».
    """
    assert SRC.count("-mmin -30") == 2, "обидва сторожі мусять зважати на ВІК крана"
    assert "if [ -f {root}/NOSTOP ]" not in SRC


def test_fetch_raises_and_drops_the_brake() -> None:
    """Кран ставиться на початку забору і знімається ЗАВЖДИ, навіть на збої."""
    i = SRC.index("def fetch_outputs")
    body = SRC[i:i + 4000]
    assert 'sftp.open(f"{REMOTE_ROOT}/NOSTOP", "w")' in body
    assert 'sftp.remove(f"{REMOTE_ROOT}/NOSTOP")' in body
    assert body.index("sftp.remove") > body.index("finally:")
