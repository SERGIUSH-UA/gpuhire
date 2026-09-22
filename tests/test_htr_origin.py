"""Склад на самій машині: чи справді він замінює бакет для раннера.

Раннер уміє рівно два рухи — `GET` посиланням і `PUT` посиланням, — тож склад
перевіряється саме ними, а не своїм внутрішнім устроєм. Окремо перевіряється
те, за що доведеться платити помилкою: вихід за корінь (склад роздавав би чужі
файли боксу), напівзаписаний чекпоінт під справжнім іменем (ламає відновлення
мовчки) і `Range` (ним міряють канал і доганяють обірване качання).
"""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pytest

from gpurunner.htr import origin


@pytest.fixture
def store(tmp_path: Path):
    """Піднятий склад: (база url, тека)."""
    root = tmp_path / "store"
    root.mkdir()
    httpd = origin.serve(root, port=0)
    yield f"http://127.0.0.1:{httpd.server_address[1]}", root
    httpd.shutdown()


def _get(url: str, *, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, resp.read()


def _put(url: str, body: bytes) -> int:
    req = urllib.request.Request(url, data=body, method="PUT")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status


def test_get_returns_the_file(store) -> None:
    base, root = store
    (root / "cases").mkdir()
    (root / "cases" / "sprava.tar").write_bytes(
        "кадри".encode().ljust(64, b"\x00"))

    code, body = _get(f"{base}/cases/sprava.tar")
    assert code == 200
    assert body == (root / "cases" / "sprava.tar").read_bytes()


def test_range_is_supported(store) -> None:
    """Ним міряють канал перед заливкою й доганяють обірване качання."""
    base, root = store
    (root / "a.bin").write_bytes(bytes(range(256)))

    code, body = _get(f"{base}/a.bin", headers={"Range": "bytes=10-19"})
    assert code == 206 and body == bytes(range(10, 20))


def test_missing_file_is_404_not_a_crash(store) -> None:
    base, _ = store
    with pytest.raises(urllib.error.HTTPError) as got:
        _get(f"{base}/cases/nemaye.tar")
    assert got.value.code == 404


def test_put_lands_whole_or_not_at_all(store) -> None:
    """🔴 Півчекпоінта під справжнім іменем виглядає як цілий і ламає
    відновлення мовчки — тому запис іде через `.part`."""
    base, root = store
    body = "чекпоінт".encode() * 100

    assert _put(f"{base}/ckpt/sprava/model_v1.pt/ckpt_0001.tgz", body) == 201
    landed = root / "ckpt" / "sprava" / "model_v1.pt" / "ckpt_0001.tgz"
    assert landed.read_bytes() == body
    assert not list(root.rglob("*.part")), "часткових файлів не лишається"


def test_put_then_get_round_trip(store) -> None:
    """Саме цим циклом живе відновлення: бокс поклав, бокс же й забрав."""
    base, _ = store
    _put(f"{base}/ckpt/x/ckpt_0002.tgz", "дані".encode())
    code, body = _get(f"{base}/ckpt/x/ckpt_0002.tgz")
    assert code == 200 and body == "дані".encode()


def test_path_cannot_escape_the_root(store, tmp_path: Path) -> None:
    """🔴 Вихід за корінь означав би роздачу чужих файлів боксу через наш склад."""
    base, _ = store
    secret = tmp_path / "secret.txt"
    secret.write_text("ключі", encoding="utf-8")

    for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/cases/../../secret.txt"):
        with pytest.raises(urllib.error.HTTPError) as got:
            _get(base + path)
        assert got.value.code in (403, 404), path
    assert secret.read_text(encoding="utf-8") == "ключі"


def test_put_cannot_escape_the_root(store, tmp_path: Path) -> None:
    base, _ = store
    with pytest.raises(urllib.error.HTTPError) as got:
        _put(f"{base}/../written.bin", b"x")
    assert got.value.code in (403, 404)
    assert not (tmp_path / "written.bin").exists()


def test_listens_on_loopback_only(tmp_path: Path) -> None:
    """🔴 Бокс стоїть у чужому дата-центрі з публічною адресою.

    Перевіряємо САМ сокет, а не рядок, який склав тест: підміна `host` на
    `0.0.0.0` лишила б рядкову перевірку зеленою, а склад — відкритим назовні.
    """
    root = tmp_path / "store"
    root.mkdir()
    httpd = origin.serve(root, port=0)
    try:
        assert httpd.server_address[0] == "127.0.0.1"
        # Адреса, на якій склад НЕ мусить слухати, — будь-яка інша локальна.
        import socket

        probe = socket.socket()
        probe.settimeout(1.0)
        with pytest.raises(OSError):
            probe.connect(("127.0.0.2", httpd.server_address[1]))
        probe.close()
    finally:
        httpd.shutdown()
