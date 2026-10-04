"""R2: ретрай на відмову, префікс як ТЕКА, посилання без ключів.

🔴 Обидві вади цього файла коштували чужої роботи: `prune` за підрядком мало не
зніс чекпоінти паралельної сесії, а відсутність ретраю губила ВЕСЬ план через
останню справу в черзі.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpurunner.htr import r2


class FakeS3:
    def __init__(self, keys: list[str] | None = None, *, fail_times: int = 0,
                 error: str = "ServiceUnavailable") -> None:
        self.keys = list(keys or [])
        self.deleted: list[str] = []
        self.uploaded: list[str] = []
        self.fail_times = fail_times
        self.error = error
        self.attempts = 0

    def upload_file(self, path: str, bucket: str, key: str, **kw: Any) -> None:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError(f"{self.error}: Reduce your concurrent request rate")
        self.uploaded.append(key)

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        self.deleted.append(Key)

    def get_paginator(self, _op: str):
        keys = self.keys

        class _P:
            def paginate(self, **kw):
                prefix = kw.get("Prefix") or ""
                yield {"Contents": [{"Key": k, "Size": 1_000_000}
                                    for k in keys if k.startswith(prefix)]}

        return _P()

    def generate_presigned_url(self, op: str, *, Params: dict, ExpiresIn: int) -> str:
        return f"https://r2/{Params['Key']}?op={op}"


@pytest.fixture
def blob(tmp_path: Path) -> Path:
    path = tmp_path / "spr-1.tar"
    path.write_bytes(b"x" * 1024)
    return path


def test_upload_retries_a_rate_limit_instead_of_losing_the_whole_plan(
        blob, monkeypatch) -> None:
    """🔴 На черзі з 22 справ R2 відповів `ServiceUnavailable: Reduce your
    concurrent request rate`, і план не створився ВЗАГАЛІ — файл лишився
    нульовим, тобто втратилась і вже залита половина (30.08.2026)."""
    monkeypatch.setattr(r2.time, "sleep", lambda _s: None)
    s3 = FakeS3(fail_times=2)
    assert r2.put(blob, prefix="cases", s3=s3, quiet=True) == "cases/spr-1.tar"
    assert s3.attempts == 3


def test_a_real_error_is_not_retried(blob, monkeypatch) -> None:
    """Ретраїмо лише те, що минеться саме. Брак прав чекання не вилікує."""
    monkeypatch.setattr(r2.time, "sleep", lambda _s: None)
    s3 = FakeS3(fail_times=99, error="AccessDenied")
    with pytest.raises(RuntimeError, match="AccessDenied"):
        r2.put(blob, s3=s3, quiet=True)
    assert s3.attempts == 1


def test_prune_matches_a_folder_not_a_substring() -> None:
    """🔴🔴 Доти збіг ішов по підрядку, і `ckpt/230-1-12` зносив
    `ckpt/230-1-129` — чекпоінти ЧУЖОГО завершеного заходу (17.08.2026)."""
    s3 = FakeS3(["ckpt/230-1-12/m/ckpt_0001.tgz", "ckpt/230-1-129/m/ckpt_0001.tgz"])
    victims = r2.prune(["ckpt/230-1-12"], yes=True, s3=s3)
    assert [k for k, _ in victims] == ["ckpt/230-1-12/m/ckpt_0001.tgz"]
    assert s3.deleted == ["ckpt/230-1-12/m/ckpt_0001.tgz"]


def test_prune_without_yes_deletes_nothing() -> None:
    s3 = FakeS3(["ckpt/spr-1/m/ckpt_0001.tgz"])
    victims = r2.prune(["ckpt/spr-1"], s3=s3)
    assert len(victims) == 1
    assert s3.deleted == []


def test_put_urls_and_resume_urls_address_the_same_keys() -> None:
    """🔴 Без дзеркальних GET чекпоінти пишуться, але прочитати їх нема чим:
    коли бокс гине, наглядач переорендовує й починає з нуля."""
    s3 = FakeS3()
    writes = r2.put_urls("ckpt/spr-1/m", 3, s3=s3)
    reads = r2.ckpt_get_urls("ckpt/spr-1/m", 3, s3=s3)
    assert len(writes) == len(reads) == 3
    assert [u.split("?")[0] for u in writes] == [u.split("?")[0] for u in reads]
    assert writes[0].endswith("op=put_object")
    assert reads[0].endswith("op=get_object")
