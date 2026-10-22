"""Замки між сесіями: чужого не чіпаємо, мертвого не чекаємо."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from gpurunner.core import locks
from gpurunner.core.locks import LockBusy


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path))


def test_second_session_is_refused() -> None:
    """🔴 Дві сесії не мають братись за ту саму справу.

    Інакше вони пишуть в одну теку `reports/htr/<...>` і псують одна одній
    знаменник повноти.
    """
    locks.acquire("case:spr-6671", owner="htr-A", session="A1")
    with pytest.raises(LockBusy, match="htr-A"):
        locks.acquire("case:spr-6671", owner="htr-B", session="B1")


def test_same_session_can_refresh_its_own_lock() -> None:
    first = locks.acquire("case:x", owner="htr-A", session="A1")
    time.sleep(0.01)
    again = locks.acquire("case:x", owner="htr-A", session="A1")
    assert again.ts >= first.ts


def test_different_resources_do_not_collide() -> None:
    locks.acquire("case:a", owner="htr-A", session="A1")
    locks.acquire("case:b", owner="htr-B", session="B1")  # не кидає


def test_expired_lock_is_taken_over_loudly(capsys: pytest.CaptureFixture[str]) -> None:
    """Убита сесія не має блокувати роботу назавжди — але перехопка гучна.

    🔴 Строк діє лише там, де ЖИТТЯ утримувача невідоме (немає pid). Живий
    процес не буває «протухлим»: мітка часу ставиться раз при взятті й не
    оновлюється, тож заходи, довші за TTL, оголошували б самі себе покинутими,
    а сусідня сесія «підхоплювала» б їхній бокс — саме той інцидент, проти
    якого писався `_is_adoptable`.
    """
    locks.acquire("case:x", owner="htr-A", session="A1", ttl_sec=0)
    path = locks._path("case:x")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pid"] = 0                     # життя утримувача невідоме
    path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(0.01)

    locks.acquire("case:x", owner="htr-B", session="B1", ttl_sec=0)
    assert "переймаю" in capsys.readouterr().out


def test_a_live_holder_survives_its_own_ttl() -> None:
    """Живий процес тримає замок, хоч би скільки минуло."""
    locks.acquire("case:long", owner="htr-A", session="A1", ttl_sec=0)
    time.sleep(0.01)
    with pytest.raises(LockBusy):
        locks.acquire("case:long", owner="htr-B", session="B1", ttl_sec=0)


def test_holder_ttl_beats_the_readers_ttl() -> None:
    """🔴 Строк судить ТОЙ, ХТО ТРИМАЄ.

    Сесія з `--max-hours 2` вважала протухлим замок сесії з `--max-hours 10`
    уже через 2.5 години — і далі підхоплювала її живий бокс.
    """
    locks.acquire("case:y", owner="htr-A", session="A1", ttl_sec=36_000)
    path = locks._path("case:y")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pid"] = 0                     # життя невідоме → судимо за строком
    data["ts"] = time.time() - 10_000   # ~2.8 год тому
    path.write_text(json.dumps(data), encoding="utf-8")

    # Читач із власним коротким строком НЕ має права перейняти чужий довгий.
    with pytest.raises(LockBusy):
        locks.acquire("case:y", owner="htr-B", session="B1", ttl_sec=9_000)


def test_dead_owner_process_frees_the_lock(capsys: pytest.CaptureFixture[str]) -> None:
    """Живий за часом, але мертвий за pid — теж підстава перейняти."""
    locks.acquire("case:x", owner="htr-A", session="A1")
    path = locks._path("case:x")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pid"] = 999_999  # такого процесу немає
    path.write_text(json.dumps(data), encoding="utf-8")

    locks.acquire("case:x", owner="htr-B", session="B1")
    assert "мертвий" in capsys.readouterr().out


def test_release_refuses_to_drop_a_foreign_lock() -> None:
    """Те саме правило, що з інстансами: чуже не чіпаємо."""
    locks.acquire("case:x", owner="htr-A", session="A1")
    assert locks.release("case:x", owner="htr-B") is False
    assert locks.read("case:x")["owner"] == "htr-A"
    assert locks.release("case:x", owner="htr-A") is True


def test_hold_releases_even_on_error() -> None:
    with pytest.raises(ValueError), locks.hold("case:x", owner="htr-A", session="A1"):
        raise ValueError("щось пішло не так")
    assert locks.read("case:x") is None


def test_holder_is_readable_by_a_human() -> None:
    """Замок мусить сам пояснювати, хто його тримає."""
    locks.acquire("case:spr-6671", owner="htr-myastkivka", session="all9", note="черга з 9")
    holder = locks.read("case:spr-6671")
    assert holder["owner"] == "htr-myastkivka"
    assert holder["note"] == "черга з 9"
    assert holder["pid"] == os.getpid()
