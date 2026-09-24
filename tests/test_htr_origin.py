"""Склад на самій машині: чи справді він замінює бакет для раннера.

Раннер уміє рівно два рухи — `GET` посиланням і `PUT` посиланням, — тож склад
перевіряється саме ними, а не своїм внутрішнім устроєм. Окремо перевіряється
те, за що доведеться платити помилкою: вихід за корінь (склад роздавав би чужі
файли боксу), напівзаписаний чекпоінт під справжнім іменем (ламає відновлення
мовчки) і `Range` (ним міряють канал і доганяють обірване качання).
"""
from __future__ import annotations

import secrets
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

# ── склад назовні: секрет замість SSH ───────────────────────────────────────

# 🔴 Секрет ГЕНЕРУЄТЬСЯ, а не пишеться літералом, і не заради ворот
# приватних даних (хоч вони його й упіймали). Літерал у тесті — це
# завжди той самий рядок, тож тест не помітив би, якби склад порівнював
# секрети з точністю до префікса чи довжини. Тут він щоразу інший — і
# такий самий, який видасть захід.
TOKEN = secrets.token_urlsafe(24)


@pytest.fixture
def guarded(tmp_path: Path):
    """Склад із секретом: (база url без секрету, тека)."""
    root = tmp_path / "guarded"
    root.mkdir()
    (root / "cases").mkdir()
    (root / "cases" / "a.tar").write_bytes(b"x" * 1000)
    # host лишаємо петлею навмисно: тест про СЕКРЕТ, а не про адресу, і
    # відкривати порт назовні на машині розробника ні до чого.
    httpd = origin.serve(root, port=0, host="127.0.0.1", token=TOKEN)
    yield f"http://127.0.0.1:{httpd.server_address[1]}", root
    httpd.shutdown()


def _code(url: str, *, method: str = "GET", body: bytes | None = None) -> int:
    req = urllib.request.Request(url, data=body, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_without_the_token_nothing_is_served(guarded) -> None:
    """Секрет у шляху — єдиний ключ; без нього складу наче й немає."""
    base, _ = guarded
    assert _code(f"{base}/cases/a.tar") == 404


def test_a_wrong_token_is_404_and_never_403(guarded) -> None:
    """🔴 Різниця не косметична.

    403 означав би «секрет не той, але склад тут є» — тобто підтверджував би
    ціль тому, хто просто сканує порти орендованої машини. 404 не каже нічого.
    """
    base, _ = guarded
    assert _code(f"{base}/nope/cases/a.tar") == 404
    assert _code(f"{base}/{TOKEN[:-1]}/cases/a.tar") == 404, "префікс теж не ключ"


def test_the_right_token_serves_the_file(guarded) -> None:
    base, _ = guarded
    status, body = _get(f"{base}/{TOKEN}/cases/a.tar")
    assert status == 200
    assert len(body) == 1000


def test_the_token_does_not_open_the_way_out_of_the_root(guarded, tmp_path: Path) -> None:
    """Секрет дає доступ до СКЛАДУ, а не до машини."""
    secret = tmp_path / "passwd"
    secret.write_text("не для складу", encoding="utf-8")
    base, _ = guarded
    assert _code(f"{base}/{TOKEN}/../passwd") == 404
    assert _code(f"{base}/{TOKEN}/%2e%2e/passwd") == 404


def test_put_needs_the_token_too(guarded) -> None:
    """🔴 Саме PUT робить відкритий склад небезпечним: без ключа будь-хто
    писав би на нашу машину що завгодно."""
    base, root = guarded
    assert _code(f"{base}/ckpt/evil.tgz", method="PUT", body=b"zzz") == 404
    assert not (root / "ckpt" / "evil.tgz").exists()

    assert _code(f"{base}/{TOKEN}/ckpt/ok.tgz", method="PUT", body=b"zzz") == 201
    assert (root / "ckpt" / "ok.tgz").read_bytes() == b"zzz"


def test_open_to_the_world_only_together_with_a_secret(tmp_path: Path) -> None:
    """🔴 Адреса й секрет — ОДНЕ рішення, а не два прапорці.

    Два незалежні прапорці рано чи пізно розходяться: хтось відкриває адресу
    «щоб перевірити» й забуває секрет. Тут забути неможливо — адресу вибирає
    наявність секрету.
    """
    root = tmp_path / "store"
    root.mkdir()

    plain = origin.serve(root, port=0)
    try:
        assert plain.server_address[0] == "127.0.0.1"
    finally:
        plain.shutdown()

    guarded_srv = origin.serve(root, port=0, token=TOKEN)
    try:
        assert guarded_srv.server_address[0] == "0.0.0.0"
    finally:
        guarded_srv.shutdown()


def test_the_secret_never_travels_in_the_command_line() -> None:
    """🔴 Рядок запуску видно в `ps` сусідам по хосту, а бокс багатоквартирний.

    Тому `main` бере секрет з оточення. Сторож дивиться джерело: поява
    `argv` як джерела секрету — це витік, якого не видно в жодному тесті
    поводження.
    """
    import inspect

    src = inspect.getsource(origin.main)
    assert "GPURUNNER_ORIGIN_TOKEN" in src
    i = src.index("token")
    assert "argv" not in src[i - 120:i + 120], "секрет не має приходити аргументом"


def _raw_put(base: str, path: str, *, declared: int, send: int) -> bytes:
    """PUT сирим сокетом: оголосити `declared` байтів, надіслати `send`, читати.

    Так поводиться справжній клієнт складу (`curl`) — він читає відповідь, не
    чекаючи кінця віддачі. `urlopen` навпаки дописує тіло до кінця, тож на
    великому тілі він діставав `BrokenPipeError` там, де сервер чесно
    відповів; на Windows це ховалось у буфері сокета, а Linux CI падав.
    """
    import socket
    from urllib.parse import urlsplit

    crlf = b"\r\n"
    parts = urlsplit(base)
    request = (b"PUT " + path.encode() + b" HTTP/1.1" + crlf
               + b"Host: " + parts.netloc.encode() + crlf
               + b"Content-Length: " + str(declared).encode() + crlf + crlf)
    head = b""
    with socket.create_connection((parts.hostname, parts.port), timeout=10) as sock:
        sock.sendall(request)
        if send:
            sock.sendall(b"z" * send)
        while crlf + crlf not in head:
            chunk = sock.recv(4096)
            if not chunk:
                break
            head += chunk
    return head


def test_a_rejected_put_answers_instead_of_dropping_the_connection(guarded) -> None:
    """🔴 Відмова мусить ДОЇХАТИ, а не обірвати з'єднання мовчки.

    Відповісти й закрити, поки клієнт ще шле тіло, — це збій сокета на його
    боці: наглядач бачить обрив і не знає, що справа в секреті. На дрібному
    тілі вада не відтворюється взагалі (воно влазить у буфер), тож виглядає
    як рідкісний глюк мережі.

    🪤 Клієнт тут САМЕ СИРИЙ СОКЕТ, і це не ускладнення заради ускладнення.
    Перша редакція тесту слала 16 МБ через `urlopen`, який дописує тіло до
    кінця й аж потім читає відповідь, — на Windows усе влазило в буфер і тест
    був зелений, а на Linux CI падав `BrokenPipeError`. Тобто тест ловив не
    поведінку складу, а розмір буфера ОС. Справжній клієнт складу — `curl`, і
    він відповідь під час віддачі читає; тут ми поводимось так само: оголосили
    велике тіло, надіслали шматок, прочитали відповідь.
    """
    base, root = guarded
    answer = _raw_put(base, "/nope/ckpt/big.tgz", declared=16 << 20,
                      send=64 << 10)

    assert answer.startswith(b"HTTP/1.1 404"), answer[:80]
    assert not (root / "ckpt" / "big.tgz").exists()


def test_a_rejected_put_that_was_not_read_to_the_end_closes_the_connection(
        guarded) -> None:
    """🔴 Недочитаний хвіст не має читатись як наступні запити.

    Слив обмежений навмисно — інакше будь-хто займає наш потік стільки,
    скільки захоче надіслати. Але з'єднання з підтримкою життя після цього
    лишалося б зі сміттям у сокеті: сотні мегабайтів чужого чекпоінта, які
    `HTTP/1.1` розбирає як запити. Тому відмова, що не дочитала тіло, несе
    `Connection: close`.
    """
    base, _ = guarded
    head = _raw_put(base, "/nope/ckpt/big.tgz", declared=16 << 20,
                    send=64 << 10).lower()

    assert b"connection: close" in head, head[:160]


def test_a_rejected_put_that_fits_the_drain_keeps_the_connection(guarded) -> None:
    """А дрібну відмову з'єднання переживає: закривати його ні до чого.

    Раннер шле чекпоінти підряд тим самим `curl`, і зайвий розрив коштував би
    рукостискання на кожну помилку.
    """
    base, _ = guarded
    head = _raw_put(base, "/nope/ckpt/small.tgz", declared=5, send=5).lower()

    assert head.startswith(b"http/1.1 404"), head[:80]
    assert b"connection: close" not in head, head[:160]
