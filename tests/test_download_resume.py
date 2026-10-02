"""Качання кадрів докачує привезене, а не починає з нуля — на справжньому curl.

🔴 24.09.2026, RTX 3090 (Мічиган): маршрут до R2 просів, одна з восьми частин
падала нижче підлоги — і раннер стирав усі вісім, потім качав том з нуля, потім
невдале передзавантаження стирало свій огризок. Том spr-102 (5.6 ГБ) шість разів
поспіль, ~2.3 ГБ викинуто, ~30 хв оплаченого простою.

Сервер тут — локальний, із діапазонами й ETag, і вміє рвати з'єднання посеред
відповіді. Приймач — скільки байтів сервер віддав: докачування коштує трохи
більше за розмір файла, качання з нуля — вдвічі й більше.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gpurunner._embedded import htr_case_runner as runner

pytestmark = pytest.mark.skipif(shutil.which("curl") is None, reason="потрібен curl")

BLOB = hashlib.sha256(b"frames").digest() * (4 * 2 ** 20 // 32)   # 4 МіБ


class _Origin:
    """Сховище з одним об'єктом: віддає діапазони, рахує байти, рве на вимогу."""

    def __init__(self) -> None:
        self.etag = '"v1"'
        self.body = BLOB
        self.served = 0
        #: Початки діапазонів (або "full"), на яких рвати ОДИН раз після `cut_at` байтів.
        self.cut: set = set()
        self.cut_at = 256 * 1024
        #: Рвати КОЖНУ відповідь, довшу за `cut_at` (хворий маршрут).
        self.always = False
        self.lock = threading.Lock()
        origin = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a) -> None:
                pass

            def do_GET(self) -> None:
                rng = self.headers.get("Range")
                body = origin.body
                if rng:
                    a, b = rng.split("=", 1)[1].split("-")
                    start, end = int(a), int(b) if b else len(body) - 1
                    key = start
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
                else:
                    start, end, key = 0, len(body) - 1, "full"
                    self.send_response(200)
                chunk = body[start:end + 1]
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("ETag", origin.etag)
                self.end_headers()
                with origin.lock:
                    cut = (origin.always or key in origin.cut) and len(chunk) > origin.cut_at
                    if cut:
                        origin.cut.discard(key)
                if cut:
                    chunk = chunk[:origin.cut_at]
                    self.close_connection = True
                self.wfile.write(chunk)
                self.wfile.flush()
                if rng != "bytes=0-0":        # проба діапазону — не качання
                    with origin.lock:
                        origin.served += len(chunk)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/t102.tar"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def origin(monkeypatch):
    monkeypatch.setattr(runner, "_set_phase", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "DOWNLOAD_RETRY_SLEEP_SEC", 0)
    monkeypatch.setattr(runner, "DOWNLOAD_HEARTBEAT_SEC", 0.05)
    monkeypatch.setattr(runner, "DOWNLOAD_PARALLEL_MIN_BYTES", 1024)
    monkeypatch.setattr(runner, "DOWNLOAD_MIN_PART_BYTES", 256 * 1024)
    monkeypatch.setattr(runner, "_disk_free_bytes", lambda p="/tmp": 100 * 2 ** 30)
    monkeypatch.setattr(runner, "_DOWNLOAD_SEEN_BPS", [])
    o = _Origin()
    yield o
    o.server.shutdown()


def _starts(total: int, parts: int, begin: int = 0) -> list[int]:
    step = (total - begin) // parts
    return [begin + i * step for i in range(parts)]


def _sick_first_call(origin, local, **kw) -> int:
    """Два заходи по хворому маршруту: кожна відповідь рветься на 200 КіБ."""
    origin.always, origin.cut_at = True, 200 * 1024
    rc = runner._download_with_heartbeat(origin.url, local, local.name, tries=2, **kw)
    assert rc != 0
    origin.always, origin.served = False, 0
    return rc


def test_a_broken_part_does_not_throw_away_the_finished_ones(origin, tmp_path) -> None:
    local = tmp_path / "t102.tar"
    total = runner._ranged_plan(origin.url, local)
    assert total == len(BLOB)
    origin.cut = {_starts(total, 8)[3]}              # четверта частина рветься
    rc = runner._download_ranged(origin.url, local, local.name, total, 8, 16384,
                                 False, 0, 1)
    assert rc != 0
    first = origin.served
    # Сім частин лишились цілими, четверта — префіксом свого діапазону.
    assert runner._parts_bytes(local) == first
    rc = runner._download_ranged(origin.url, local, local.name, total, 8, 16384,
                                 False, 0, 2)
    assert rc == 0
    assert local.read_bytes() == BLOB
    # Друга спроба привезла лише залишок четвертої частини.
    assert origin.served == total
    assert not runner._ledger_path(local).exists() or \
        runner._read_ledger(local).get("n") is None


def test_the_whole_loop_resumes_after_a_broken_stream_and_a_broken_part(origin,
                                                                         tmp_path) -> None:
    local = tmp_path / "t102.tar"
    origin.cut_at = 1024 * 1024
    origin.cut = {"full"}                            # один потік рветься на 1 МіБ
    # і одна з частин продовження — теж
    total = len(BLOB)
    origin.cut.add(_starts(total, 8, begin=1024 * 1024)[5])
    assert runner._download_with_heartbeat(origin.url, local, local.name) == 0
    assert local.read_bytes() == BLOB
    assert origin.served <= total * 1.1, "повтор мав докачувати, а не качати заново"


def test_leftovers_survive_to_the_next_call_for_the_same_object(origin, tmp_path) -> None:
    """Повтор тому в кінці черги продовжує з того, що привезла невдала спроба."""
    local = tmp_path / "t102.tar"
    total = len(BLOB)
    _sick_first_call(origin, local)
    kept = runner._have_bytes(local)
    assert kept > total // 2
    assert runner._download_with_heartbeat(origin.url, local, local.name) == 0
    assert local.read_bytes() == BLOB
    assert origin.served == total - kept


def test_a_changed_object_is_downloaded_from_scratch(origin, tmp_path) -> None:
    """Інший ETag — привезене чуже: стерти й качати заново, а не склеювати."""
    local = tmp_path / "t102.tar"
    _sick_first_call(origin, local)
    assert runner._have_bytes(local) > 0
    origin.etag = '"v2"'
    origin.body = bytes(reversed(BLOB))
    assert runner._download_with_heartbeat(origin.url, local, local.name) == 0
    assert local.read_bytes() == origin.body


def test_a_failed_prefetch_hands_its_bytes_to_the_normal_download(origin, tmp_path) -> None:
    pre = tmp_path / "prefetch_02" / "t102.tar"
    main = tmp_path / "t102.tar"
    total = len(BLOB)
    pre.parent.mkdir()
    pf = runner._Prefetch(origin.url, pre)
    pf.rc = _sick_first_call(origin, pre, heartbeat=False)
    pf._thread = threading.Thread(target=lambda: None)
    pf._thread.start()
    assert pf.take(origin.url, resume_to=main) is None
    assert not pre.exists()
    kept = runner._have_bytes(main)
    assert kept > total // 2
    assert runner._download_with_heartbeat(origin.url, main, main.name) == 0
    assert main.read_bytes() == BLOB
    assert origin.served == total - kept
