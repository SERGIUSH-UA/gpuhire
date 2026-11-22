"""Доступ до R2 для людини, що бакета ще не має (`storage_login`).

Чому: дешеві машини обриваються, і без проміжного сховища кадри після
переоренди їдуть з дому вдруге, а чекпоінти наглядач тягне додому раз на 5 хв.
Доти `r2.json` писався лише руками — споживач Нишпорки до нього не доходив.

Обіцянки:
- доступ дописується у `r2.json`, порожнє не затирає наявного, стара адреса
  сховища прибирається, коли названо новий акаунт;
- `describe` не несе секретів;
- `verify` ловить окремо: немає бакета, токен без запису, посилання не відкривається;
- плагін зберігає доступ навіть тоді, коли перевірка не пройшла.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from gpurunner.htr import r2
from gpurunner.plugins import nyshporka_cloud as plugin

KEY, SECRET = "AKIA-test-key", "sekret-sekret-123"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_CONFIG_DIR", str(tmp_path / "config"))
    for k in r2.CONFIG_KEYS:
        monkeypatch.delenv(k, raising=False)


def _stored() -> dict[str, str]:
    return json.loads(r2.config_path().read_text(encoding="utf-8"))


def test_save_merges_and_keeps_what_was_not_named() -> None:
    r2.save_config({"R2_ACCESS_KEY_ID": KEY, "R2_SECRET_ACCESS_KEY": SECRET,
                    "R2_ENDPOINT": "https://old.example", "R2_BUCKET": "b1"})
    r2.save_config({"R2_BUCKET": "b2", "R2_ACCESS_KEY_ID": ""})
    got = _stored()
    assert got["R2_BUCKET"] == "b2" and got["R2_ACCESS_KEY_ID"] == KEY
    r2.save_config({"CLOUDFLARE_ACCOUNT_ID": "acc"}, drop=("R2_ENDPOINT",))
    got = _stored()
    assert "R2_ENDPOINT" not in got and got["CLOUDFLARE_ACCOUNT_ID"] == "acc"
    assert r2.describe()["endpoint"] == "https://acc.r2.cloudflarestorage.com"


def test_save_refuses_unknown_keys() -> None:
    with pytest.raises(r2.R2Error, match="AWS_SECRET"):
        r2.save_config({"AWS_SECRET": "x"})


def test_describe_carries_no_secrets() -> None:
    r2.save_config({"R2_ACCESS_KEY_ID": KEY, "R2_SECRET_ACCESS_KEY": SECRET,
                    "CLOUDFLARE_ACCOUNT_ID": "acc", "R2_BUCKET": "mine"})
    view = r2.describe()
    assert view["bucket"] == "mine"
    assert KEY not in repr(view) and SECRET not in repr(view)


class _S3:
    def __init__(self, *, bucket: bool = True, write: bool = True) -> None:
        self.bucket, self.write = bucket, write
        self.objects: dict[str, bytes] = {}

    def head_bucket(self, Bucket: str) -> None:
        if not self.bucket:
            raise RuntimeError("NoSuchBucket")

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:
        if not self.write:
            raise RuntimeError("AccessDenied")
        self.objects[Key] = Body

    def generate_presigned_url(self, op: str, Params: dict[str, str],
                               ExpiresIn: int) -> str:
        return "presigned://" + Params["Key"]

    def delete_object(self, Bucket: str, Key: str) -> None:
        self.objects.pop(Key, None)


def _with(monkeypatch: pytest.MonkeyPatch, s3: _S3, *, link_ok: bool = True) -> None:
    import urllib.request

    monkeypatch.setattr(r2, "client", lambda *a, **k: s3)

    def urlopen(url: str, timeout: float = 0) -> Any:
        if not link_ok:
            raise OSError("403 Forbidden")
        return io.BytesIO(s3.objects[url.removeprefix("presigned://")])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


def test_verify_passes_and_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    s3 = _S3()
    _with(monkeypatch, s3)
    assert r2.verify(bucket="mine") == []
    assert s3.objects == {}, "пробний об'єкт мусить зникнути"


@pytest.mark.parametrize(("s3", "link_ok", "word"), [
    (_S3(bucket=False), True, "Create bucket"),
    (_S3(write=False), True, "Read & Write"),
    (_S3(), False, "підписане посилання"),
])
def test_verify_names_what_exactly_is_broken(monkeypatch: pytest.MonkeyPatch, s3: _S3,
                                             link_ok: bool, word: str) -> None:
    _with(monkeypatch, s3, link_ok=link_ok)
    problems = r2.verify(bucket="mine")
    assert len(problems) == 1 and word in problems[0], problems


def test_plugin_keeps_the_access_even_when_the_bucket_is_missing(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _with(monkeypatch, _S3(bucket=False))
    monkeypatch.setattr(r2, "configured", lambda: True)
    out = plugin.VastRent().storage_login(KEY, SECRET, account="acc", bucket="mine")
    assert out["ready"] is False and out["problems"]
    assert _stored()["R2_ACCESS_KEY_ID"] == KEY      # не вводити вдруге
    assert SECRET not in repr(out)


def test_plugin_refuses_without_an_address() -> None:
    with pytest.raises(plugin.contract().AuthError, match="--account"):
        plugin.VastRent().storage_login(KEY, SECRET)
