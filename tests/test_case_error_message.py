"""Повідомлення про впалу справу мусить нести ПРИЧИНУ, а не адресу.

21.09.2026 три справи черги впали поспіль, і в стані лишилось по 300 символів
підписаного посилання — без коду помилки й без тексту. Діагноз за таким
рядком ставиться навмання: перше, що спало на думку, — «посилання протухло»,
хоча в ньому ж було написано, що живе воно добу.
"""

from __future__ import annotations

from gpurunner.supervise.htr import short_error

#: Підписане посилання складається так само, як справжнє, але всі значення
#: вигадані: справжній приклад тягне за собою адресу бакета й креденшел, а це
#: приватні дані (ворота `tools/scan_private.py` ловлять їх і в тестах).
#: Важлива тут лише ДОВЖИНА: близько 370 символів, тобто більша за ліміт.
ПІДПИС = (
    "https://example-account.r2.cloudflarestorage.com/example-bucket"
    "/cases/spr-576.tar?X-Amz-Algo=AWS4-EXAMPLE"
    "&X-Amz-Cred=" + "c" * 64
    + "&X-Amz-Date=20260921T195647Z&X-Amz-Exp=86400"
    + "&X-Amz-Headers=host&X-Amz-Sig=" + "f" * 96
)
СПРАВЖНЯ = f"curl {ПІДПИС} -> rc=23: Failure writing output to destination: No space left on device"


def test_the_signature_is_long_enough_to_matter() -> None:
    """Сторож самого тесту: без довгого посилання він нічого не доводить."""
    assert len(ПІДПИС) > 300


def test_the_reason_survives_a_presigned_url() -> None:
    короткий = short_error(СПРАВЖНЯ)
    assert "No space left on device" in короткий
    assert "rc=23" in короткий
    assert "spr-576.tar" in короткий          # яка саме справа — теж потрібно
    assert "X-Amz-Sig" not in короткий        # а підпис не несе змісту
    assert len(короткий) <= 300


def test_short_messages_are_left_alone() -> None:
    assert short_error("неповна") == "неповна"


def test_a_long_message_without_urls_keeps_both_ends() -> None:
    текст = "ПОЧАТОК " + "х" * 500 + " КІНЕЦЬ"
    короткий = short_error(текст)
    assert короткий.startswith("ПОЧАТОК")
    assert короткий.endswith("КІНЕЦЬ")
    assert len(короткий) <= 300


def test_url_without_query_is_folded_too() -> None:
    короткий = short_error("curl https://example.com/a/b/c.tar -> rc=7")
    assert "…/c.tar" in короткий and "rc=7" in короткий
