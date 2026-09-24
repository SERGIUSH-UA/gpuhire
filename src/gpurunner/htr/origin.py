"""Сховище на самій машині: крихітний HTTP-склад замість хмарного бакета.

Раннер на боксі знає рівно один спосіб брати й віддавати дані — `curl` по
посиланню: кадри й ассети він качає `GET`, чекпоінти віддає `PUT`. Доти таке
посилання вмів видати лише R2, тобто хмарний прогін вимагав свого бакета S3
навіть тому, хто просто орендував машину на годину.

Тут те саме посилання дає сама машина. Перед запуском роботи наглядач кладе
файли на бокс `scp`, піднімає цей склад на `127.0.0.1` і підставляє в
параметри роботи адреси виду `http://127.0.0.1:<порт>/cases/<справа>.tar`.
Раннер не змінюється жодним рядком — він як качав `curl`, так і качає.

🔴 Орендований бокс стоїть у чужому дата-центрі з публічною адресою, тож склад,
відкритий назовні без нічого, роздавав би чужим кадри архівної справи, а `PUT`
дозволяв би писати на нашу машину що завгодно. Тому адреса прив'язана до
СЕКРЕТУ, і це одне рішення, а не два прапорці:

- секрету немає → слухаємо тільки петлю (`127.0.0.1`), як було;
- секрет є → слухаємо назовні, і кожен шлях мусить починатись із нього.

Секрет живе в ШЛЯХУ — рівно як у presigned-посиланні R2, тож раннер качає тим
самим `curl` і не змінюється жодним рядком. Чужий секрет дістає 404, а не 403:
403 підтверджував би, що за цією адресою щось є.

Навіщо взагалі вихід назовні: інакше додому файли доводиться везти по SSH, а
це заміряно й погано — SFTP paramiko 0.48 МБ/с і зворотний тунель 0.49 проти
20.07 МБ/с у того самого HTTP (`box_transport.py`). Секрет передається
оточенням (`GPURUNNER_ORIGIN_TOKEN`), а не аргументом: рядок запуску видно в
`ps` сусідам по хосту.

🔴 Чекпоінти, що лежать на боксі, НЕ рятують від смерті боксу — на відміну від
чекпоінтів у бакеті. Тому наглядач мусить періодично забирати їх додому; сам
склад цього не робить і робити не може.

Запускається на боксі як самостійний скрипт (`python3 origin.py <тека>
[порт]`), без жодної залежності поза стандартною бібліотекою: на машині є лише
той python, що прийшов з образом.
"""
from __future__ import annotations

import hmac
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

#: Шлях відповіді «склад живий». Під секретом, як і все інше.
HEALTH_PATH = "/healthz"

#: Скільки тіла зливати перед ВІДМОВОЮ. Досить, щоб клієнт дістав чесний
#: код замість обриву, і мало, щоб чужий запит не займав наш потік.
REJECT_DRAIN_MAX = 1 << 16


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
    #: Секрет заходу. Порожній — склад слухає ЛИШЕ петлю (стара поведінка).
    token = ""
    protocol_version = "HTTP/1.1"
    #: 🔴 Стеля на ОДНЕ читання з сокета. Без неї клієнт, який оголосив тіло
    #: більше, ніж надіслав, тримає наш потік і свій `.part` доти, доки TCP не
    #: здасться сам — а це десятки хвилин на машині, за яку платять погодинно.
    #: Читання цілого чекпоінта вона не ріже: стеля на шматок, не на файл.
    timeout = 120

    def _target(self) -> Path | None:
        """Файл під запитом — або `None`, якщо шлях чи секрет не ті.

        🔴 Секрет живе в ШЛЯХУ, а не в заголовку, і це не лінощі: раннер качає
        звичайним `curl <url>`, і будь-який заголовок довелось би проводити
        через нього, через `--retry`, через діапазонні з'єднання й через
        `fetch-ckpt`. Секрет у шляху — рівно модель presigned-посилання R2,
        яку раннер уже споживає, тож він не змінюється жодним рядком.

        🔴 Відповідь на чужий секрет — 404, а не 403: 403 підтверджує, що за
        цією адресою щось є, і перетворює сканування портів на пошук цілі.
        Порівняння постійним часом — щоб відповідь не підказувала префікс.
        """
        raw = self._own_path()
        return None if raw is None else _safe_target(self.root, raw)

    def _own_path(self) -> str | None:
        """Шлях без секрету — або `None`, якщо секрет не той."""
        raw = self.path
        if not self.token:
            return raw
        head, _, rest = raw.lstrip("/").partition("/")
        if not hmac.compare_digest(head, self.token):
            return None
        return "/" + rest

    def log_message(self, fmt: str, *args: object) -> None:
        # Журнал складу нікому не потрібен, а ось лог самої роботи він
        # затопить: раннер робить сотні запитів на кожну справу.
        return

    def _drain(self, limit: int) -> bool:
        """Прочитати й викинути до `limit` байтів тіла. True — дочитали ВСЕ.

        Потрібно лише перед ВІДМОВОЮ: інакше ми закриваємо з'єднання, поки
        клієнт ще шле, і він бачить обрив замість коду відповіді.

        🔴 Межа обов'язкова, і через неї «дочитали все» не гарантоване:
        зливати чуже тіло до кінця означало б дати будь-кому займати наш потік
        стільки, скільки він захоче надіслати. Тому відповідь тут і вирішує,
        чи можна лишити з'єднання живим.
        """
        try:
            left = total = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return False
        left = min(left, limit)
        read = 0
        while left > 0:
            chunk = self.rfile.read(min(CHUNK, left))
            if not chunk:
                return False
            left -= len(chunk)
            read += len(chunk)
        return read >= total

    def _fail(self, code: int, why: str = "", *, close: bool = False) -> None:
        """Відповідь-відмова. `close` — розірвати з'єднання після неї.

        🔴 `close` не косметика: коли ми відмовляємо, НЕ дочитавши тіло, у
        сокеті лишаються сотні мегабайтів, і на з'єднанні з підтримкою життя
        (`HTTP/1.1`) сервер почне читати їх як наступний запит. Вийшла б каша:
        чужий чекпоінт, розібраний на «запити», і жодного чесного коду.
        """
        body = (why or "").encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        if close:
            # `send_header` сам ставить `close_connection` на це значення.
            self.send_header("Connection", "close")
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
        own = self._own_path()
        if own is not None and own.split("?")[0].rstrip("/") == HEALTH_PATH:
            # 🔴 Окрема відповідь «живий», бо питання інше: не «чи є файл», а
            # «чи дістає нас цей склад узагалі». Доти наглядач питав про
            # `_origin.py` — файл, який існує лише тому, що ми його щойно
            # поклали; проба на побічній обставині ламається щоразу, коли
            # обставина змінюється.
            #
            # 🔴 І вона теж ПІД СЕКРЕТОМ: інакше це маячок, який підтверджує
            # чужому сканеру, що за цим портом є склад, — рівно те, від чого
            # стоїть 404 замість 403.
            self._fail(200, "ok")
            return
        target = self._target()
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
        target = self._target()
        if target is None:
            # 🔴 404, а не 403: із секретом у шляху 403 означав би «секрет не
            # той, але склад тут є» — тобто підтверджував би ціль тому, хто
            # просто сканує порти.
            # 🔴 Перед відмовою ЗЛИВАЄМО трохи тіла: відповісти й закрити,
            # поки клієнт ще шле, означає обрив на його боці — наглядач
            # дістав би збій сокета замість чесного коду й не знав би, що
            # справа в секреті. Межа є, бо зливати чуже тіло до кінця
            # означало б дати будь-кому займати наш потік скільки завгодно.
            #
            # 🔴 А якщо тіло за межу НЕ влізло — з'єднання мусить померти
            # разом із відповіддю. Недочитаний хвіст на живому з'єднанні
            # `HTTP/1.1` сервер читав би як наступні запити. Клієнт, що встиг
            # прочитати відповідь (так робить `curl`), бачить чесні 404;
            # той, хто шле не дивлячись, — розрив, і це найбільше, що тут
            # можна дати, не роздавши власний потік кожному охочому.
            self._fail(404, "немає такого шляху",
                       close=not self._drain(REJECT_DRAIN_MAX))
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


class _QuietServer(ThreadingHTTPServer):
    """Той самий сервер, але без трасувань у лог на кожен обрив.

    🔴 Розрив — штатний стан цього складу, а не подія. Ми самі рвемо
    з'єднання, коли відмовляємо, не дочитавши чужого тіла; те саме робить
    `curl` раннера, коли качання перервано й повтор іде діапазонами. Базовий
    клас друкує на кожен такий випадок повне трасування — а на боксі це той
    самий лог, у якому потім шукають, ЧОМУ впала справа. Топити його
    очікуваним шумом означає зробити його непридатним саме тоді, коли він
    потрібен.
    """

    def handle_error(self, request: object, client_address: object) -> None:
        import traceback

        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError, BrokenPipeError)):
            return
        # Решта — наша помилка, і про неї мовчати не можна: склад один на
        # весь захід, і тихий збій тут виглядає як мертва мережа.
        print("[origin] ⚠ ", "".join(traceback.format_exc())[-500:],
              file=sys.stderr, flush=True)


def serve(root: Path, port: int = DEFAULT_PORT, *,
          host: str = "", token: str = "") -> ThreadingHTTPServer:
    """Підняти склад і повернути сервер (зупиняється `shutdown()`).

    🔴 Адреса вибирається САМА і за одним правилом: є секрет — слухаємо
    назовні, немає — тільки петлю. Тобто відкрити склад світові, забувши
    секрет, неможливо: це не два незалежні прапорці, а один.
    Явний `host` лишається для тестів і для того, хто свідомо знає інакше.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bind = host or ("0.0.0.0" if token else "127.0.0.1")
    handler = type("BoundHandler", (Handler,), {"root": root, "token": token})
    httpd = _QuietServer((bind, port), handler)
    httpd.daemon_threads = True
    # `shutdown()` чекає до кінця поточного інтервалу опитування; типові 0.5 с
    # набігали в тестах на кожен сервер.
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05},
                     daemon=True).start()
    return httpd


def main(argv: list[str]) -> int:
    if not argv:
        print("вжиток: origin.py <тека> [порт]", file=sys.stderr)
        return 2
    root = Path(argv[0])
    port = int(argv[1]) if len(argv) > 1 else DEFAULT_PORT
    # 🔴 Секрет приходить оточенням, а не аргументом: рядок запуску видно в
    # `ps` кожному, хто має доступ до машини, зокрема сусідам по хосту.
    token = os.environ.get("GPURUNNER_ORIGIN_TOKEN", "").strip()
    if shutil.disk_usage(root.parent if root.parent.exists() else Path(".")).free < CHUNK:
        print("[origin] на диску немає місця", file=sys.stderr)
        return 1
    httpd = serve(root, port, token=token)
    where = "0.0.0.0" if token else "127.0.0.1"
    print(f"[origin] {root} на {where}:{port}"
          f"{' (із секретом)' if token else ' (лише петля)'}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
