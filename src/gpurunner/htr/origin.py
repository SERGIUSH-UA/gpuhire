"""Сховище на самій машині: крихітний HTTP-склад замість хмарного бакета.

Раннер на боксі знає рівно один спосіб брати й віддавати дані — `curl` по
посиланню: кадри й ассети він качає `GET`, чекпоінти віддає `PUT`. Доти таке
посилання вмів видати лише R2, тобто хмарний прогін вимагав свого бакета S3
навіть тому, хто просто орендував машину на годину.

Тут те саме посилання дає сама машина. Перед запуском роботи наглядач кладе
файли на бокс `scp`, піднімає цей склад на `127.0.0.1` і підставляє в
параметри роботи адреси виду `http://127.0.0.1:<порт>/cases/<справа>.tar`.
Раннер не змінюється жодним рядком — він як качав `curl`, так і качає.

🔴 Слухаємо ТІЛЬКИ петлю (`127.0.0.1`). Орендований бокс стоїть у чужому
дата-центрі з публічною адресою, і склад, відкритий назовні, роздавав би чужим
кадри архівної справи, а `PUT` дозволяв би писати на нашу машину що завгодно.
Вихід назовні тут не потрібен узагалі: і раннер, і склад живуть на одному
боксі, а додому файли забирає наглядач по SSH.

🔴 Чекпоінти, що лежать на боксі, НЕ рятують від смерті боксу — на відміну від
чекпоінтів у бакеті. Тому наглядач мусить періодично забирати їх додому; сам
склад цього не робить і робити не може.

Запускається на боксі як самостійний скрипт (`python3 origin.py <тека>
[порт]`), без жодної залежності поза стандартною бібліотекою: на машині є лише
той python, що прийшов з образом.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

#: Порт за замовчуванням. Високий і непримітний: на боксі вже живуть чужі
#: служби образу, а 8000 і 8080 зайняті частіше за все.
DEFAULT_PORT = 8752

#: Скільки читати за раз. Чекпоінт справи — десятки мегабайтів, і читати його
#: одним шматком у пам'ять на боксі з 8 ГБ не варто.
CHUNK = 1 << 20


def _safe_target(root: Path, raw_path: str) -> Path | None:
    """Шлях усередині складу — або `None`, якщо він звідти виводить.

    🔴 `..` у запиті — класичний вихід за корінь, і тут він означав би читання
    чужих файлів боксу через наш же склад. Перевіряємо ПІСЛЯ нормалізації, а не
    пошуком підрядка: `%2e%2e` і подвійні слеші пошук підрядка не ловить.
    """
    from urllib.parse import unquote, urlparse

    rel = unquote(urlparse(raw_path).path).lstrip("/")
    if not rel:
        return None
    target = (root / rel).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    return target


class Handler(BaseHTTPRequestHandler):
    """GET віддає файл, PUT приймає. Більше складові вміти нічого не треба."""

    root = Path(".")
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        # Журнал складу нікому не потрібен, а ось лог самої роботи він
        # затопить: раннер робить сотні запитів на кожну справу.
        return

    def _fail(self, code: int, why: str = "") -> None:
        body = (why or "").encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self) -> None:
        self._serve(head_only=True)

    def do_GET(self) -> None:
        self._serve(head_only=False)

    def _serve(self, *, head_only: bool) -> None:
        """Віддати файл цілком або шматком (`Range`).

        `Range` підтримується не для краси: наглядач міряє ним канал до
        заливки даних, а `curl --retry` доганяє обірване качання з середини.
        """
        target = _safe_target(self.root, self.path)
        if target is None or not target.is_file():
            self._fail(404, "немає такого файла")
            return
        size = target.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range", "")
        partial = False
        if rng.startswith("bytes="):
            head, _, tail = rng[len("bytes="):].partition("-")
            try:
                start = int(head) if head else 0
                end = int(tail) if tail else size - 1
            except ValueError:
                self._fail(416, "не розібрав Range")
                return
            if start >= size or start > end:
                self._fail(416, "Range поза файлом")
                return
            end = min(end, size - 1)
            partial = True
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        with target.open("rb") as fh:
            fh.seek(start)
            left = length
            while left > 0:
                chunk = fh.read(min(CHUNK, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def do_PUT(self) -> None:
        """Прийняти чекпоінт.

        🔴 Спершу у `.part`, і лише цілий файл стає на місце. Раннер може
        вмерти посеред віддачі, а наглядач у цей самий час забирає теку додому:
        півчекпоінта під справжнім іменем виглядає як цілий і ламає відновлення
        мовчки.
        """
        target = _safe_target(self.root, self.path)
        if target is None:
            self._fail(403, "шлях виводить за межі складу")
            return
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            # 🔴 Без довжини тіло могло приїхати шматками (`chunked`), і нуль
            # тут читався б як «нічого не передали» — тобто на місце цілого
            # чекпоінта лягав би ПОРОЖНІЙ файл із кодом 201. Мовчазна заміна
            # цілого на порожнє — рівно та вада, від якої стоїть запис через
            # `.part`, тільки з іншого боку.
            self._fail(411, "потрібен Content-Length (chunked не підтримуємо)")
            return
        try:
            length = int(raw_len)
        except ValueError:
            self._fail(411, "Content-Length не число")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        left = length
        try:
            with tmp.open("wb") as fh:
                while left > 0:
                    chunk = self.rfile.read(min(CHUNK, left))
                    if not chunk:
                        break
                    fh.write(chunk)
                    left -= len(chunk)
            if left:
                tmp.unlink(missing_ok=True)
                self._fail(400, "тіло коротше за Content-Length")
                return
            os.replace(tmp, target)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            self._fail(507, f"не записалось: {exc}")
            return
        self._fail(201, "")


def serve(root: Path, port: int = DEFAULT_PORT, *,
          host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Підняти склад і повернути сервер (зупиняється `shutdown()`)."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    handler = type("BoundHandler", (Handler,), {"root": root})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main(argv: list[str]) -> int:
    if not argv:
        print("вжиток: origin.py <тека> [порт]", file=sys.stderr)
        return 2
    root = Path(argv[0])
    port = int(argv[1]) if len(argv) > 1 else DEFAULT_PORT
    if shutil.disk_usage(root.parent if root.parent.exists() else Path(".")).free < CHUNK:
        print("[origin] на диску немає місця", file=sys.stderr)
        return 1
    httpd = serve(root, port)
    print(f"[origin] {root} на 127.0.0.1:{port}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
