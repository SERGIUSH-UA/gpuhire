"""Доставка без бакета: файли везе HTTP, посилання видає сама машина.

Половина, що живе вдома. Друга половина — `htr/origin.py`, крихітний склад,
який ми кладемо на машину; раннер качає з нього тим самим `curl`, що й з
бакета, і не знає про підміну нічого.

🔴 ОДНА технологія на дані. Заміряно на орендованому боксі 20.09.2026, 150 МБ:

| чим                        | МБ/с |
|----------------------------|------|
| SFTP paramiko              | 0.48 |
| зворотний тунель paramiko  | 0.49 |
| системний `scp`            | 7.56 |
| HTTP одним потоком         | 20.07|

Розкид у 42 рази на тому самому залізі вирішує протокол, а не смуга — але
🔴 ЦЯ ТАБЛИЦЯ ПРО РІЗНІ ПЛЕЧІ, і плутати їх дорого. 20.07 МБ/с — це бокс, що
качає зі СХОВИЩА; 0.48 у SFTP — це забір ДОДОМУ. А 7.56 у `scp` не може
описувати плече дім→бокс: 7.56 МБ/с це 60 Мбіт/с, тоді як наш домашній аплінк
заміряно 23.09.2026 — **9 Мбіт/с** (1.08 МБ/с до бакета).

Тому чесно про вигоду: HTTP замінив SFTP там, де плече справді було вузьким
(забір додому й службові файли — 42 рази й 2600 обертів проти двох). А на
плечі дім→бокс стеля НАША: живий замір 23.09.2026 на орендованому боксі дав
1.2 МБ/с, тобто `PUT` вибрав увесь домашній аплінк і жоден протокол більше з
нього не візьме. Виграш там дає не протокол, а НЕ ВЕЗТИ: 121 МБ ассетів із
135 — те, що вже лежить у бакеті.

`scp` лишився запасним плечем на випадок, коли порт складу знадвору
недосяжний, і перехід на нього називається вголос, а не мовчки.

🔴 Чекпоінти, складені на цю машину, гинуть разом із нею — на відміну від
чекпоінтів у бакеті. Тому `pull_checkpoints` тут не зручність, а єдине, що
відрізняє «бокс помер, дочитаємо з місця» від «бокс помер, платимо все
наново»; наглядач кличе його періодично, а не в кінці.
"""
from __future__ import annotations

import os
import posixpath
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurunner.core.backend import BackendError

#: Де на машині живе склад. Під `/workspace`, бо саме він на Vast лежить на
#: великому диску інстансу, а не в образі.
REMOTE_ROOT = "/workspace/_origin"

#: Порт складу на петлі машини.
ORIGIN_PORT = 8752

#: Скільки чекати `scp`. Рахується від обсягу: 0.5 МБ/с — свідомо песимістична
#: оцінка (справжні заміри вп'ятнадцятеро кращі), щоб таймаут ловив зависання,
#: а не повільний, але живий канал.
_SCP_MIN_SEC = 120
_SCP_MB_PER_SEC = 0.5

#: Те саме для HTTP. Вище за `scp`-ову, бо й вимір вищий, але однаково
#: песимістична: це стеля АВАРІЇ, а не очікуваний темп.
#: ⚠ Стала тимчасова. Правильне — рахувати від ЗАМІРЯНОГО каналу до цієї
#: машини; поки що заміряти його нічим до першої ж заливки, і саме цю
#: курку-яйце розв'язує наступний крок (проба складу по `Range`).
_HTTP_MB_PER_SEC = 1.0

#: Скільки чекати відповіді складу на пробу «ти живий і назовні».
PROBE_TIMEOUT_SEC = 8

#: На скільки потоків різати заливку й від якого розміру це має сенс.
#:
#: 🔴 Що тут заміряно, а що ні — важливо не переплутати.
#:
#: ЗАМІРЯНО на живому боксі: доставка 135 МБ одним потоком у Корею — 1.2 МБ/с.
#: ЗАМІРЯНО вдома: те саме плече до бакета гуляє 1.05-8.94 МБ/с між прогонами,
#: медіана ~2.2 — розкид у вісім разів. Тому одиночне порівняння «один потік
#: проти восьми» на ньому НЕ доводить нічого: чергований замір 23.09.2026 дав
#: відношення медіан 0.97, тобто різниця методів менша за шум каналу.
#:
#: ⚠ Виграш саме на плечі дім→бокс ще НЕ заміряний. Змінено це з іншої
#: причини — та сама, з якої транспорт на боксі вже ескалює до восьми
#: діапазонів: на великому колі один потік обмежений добутком смуги й
#: затримки, і поки летить підтвердження, вікно мовчить. Доказом буде A/B на
#: ОДНІЙ машині: 135 МБ одним потоком проти восьми.
#:
#: Це та сама вада, від якої стоїть двопотокова проба каналу, — тільки у
#: зворотний бік: там ми МІРЯЛИ одним, а качали вісьмома; тут міряли вісьмома,
#: а ВЕЗЛИ одним.
PUSH_PARTS = 8
PUSH_PARALLEL_MIN_BYTES = 8 << 20


@dataclass(frozen=True)
class Endpoint:
    """Куди й чим стукати `scp`."""

    host: str
    port: int
    key: Path
    user: str = "root"

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def endpoint_of(backend: Any, handle: Any) -> Endpoint:
    """SSH-точка машини — або чесна відмова.

    Береться в бекенда через необов'язковий `ssh_endpoint(handle)`: транспорт
    `box` має сенс лише там, де в машини взагалі є SSH, і бекенд без цього
    методу чесно каже про це замість того, щоб вдавати доставку.
    """
    fn = getattr(backend, "ssh_endpoint", None)
    if not callable(fn):
        raise BackendError(
            f"бекенд «{getattr(backend, 'name', backend)}» не дає SSH-доступу до "
            f"машини, а транспорт `box` возить файли саме ним. Тут потрібен "
            f"бакет: `--transport r2`.")
    raw = fn(handle) or {}
    host, port, key = raw.get("host"), raw.get("port"), raw.get("key")
    if not host or not port or not key:
        raise BackendError("машина ще не назвала SSH-точки — доставку починати нема куди")
    return Endpoint(host=str(host), port=int(port), key=Path(str(key)),
                    user=str(raw.get("user") or "root"))


def _scp_base(ep: Endpoint, *, known_hosts: Path | None = None) -> list[str]:
    exe = shutil.which("scp")
    if not exe:
        raise BackendError(
            "немає системного `scp` — транспорт `box` возить файли ним. "
            "Поставте OpenSSH-клієнт або використайте `--transport r2`.")
    cmd = [exe, "-P", str(ep.port), "-i", str(ep.key),
           "-o", "BatchMode=yes",
           # Машина щоразу нова й живе годину — постійного відбитка в неї немає,
           # звіряти нема з чим. Пишемо у СВІЙ файл, щоб не засмічувати
           # користувацький `known_hosts` ефемерними ключами.
           "-o", "StrictHostKeyChecking=accept-new"]
    if known_hosts is not None:
        known_hosts.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["-o", f"UserKnownHostsFile={known_hosts}"]
    return cmd


def _timeout_for(size_bytes: int) -> int:
    return int(max(_SCP_MIN_SEC, size_bytes / 1e6 / _SCP_MB_PER_SEC))


def _run(cmd: list[str], *, timeout: int, what: str) -> None:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise BackendError(f"{what}: не вклалось у {timeout} с") from None
    if done.returncode:
        tail = (done.stderr or done.stdout or "").strip().splitlines()[-1:]
        tool = posixpath.basename(str(cmd[0])).replace(".exe", "") or "команда"
        raise BackendError(
            f"{what}: {tool} rc={done.returncode} {' '.join(tail)}".strip())


def push(ep: Endpoint, local: Path, remote: str, *,
         known_hosts: Path | None = None, verify_with: Any = None) -> int:
    """Покласти файл на машину. Повертає його розмір.

    `verify_with(cmd) -> str` виконує команду НА МАШИНІ; коли він переданий,
    розмір звіряється там-таки. 🔴 Обірвана заливка лишає файл, який виглядає
    покладеним, і виявляється це вже на оплачуваній карті — тим, що раннер не
    може розпакувати архів.
    """
    local = Path(local)
    if not local.is_file():
        raise BackendError(f"немає файла для доставки: {local}")
    size = local.stat().st_size
    cmd = [*_scp_base(ep, known_hosts=known_hosts), str(local),
           f"{ep.target}:{remote}"]
    _run(cmd, timeout=_timeout_for(size), what=f"доставка {local.name}")
    if callable(verify_with):
        said = str(verify_with(f"stat -c %s {remote} 2>/dev/null || echo ?")).strip()
        landed = int(said) if said.isdigit() else -1
        if landed != size:
            raise BackendError(
                f"{local.name}: доїхало {landed if landed >= 0 else '?'} байтів "
                f"із {size} — заливка обірвалась. На машині це виглядало б як "
                f"битий архів уже під час роботи.")
    return size


def pull(ep: Endpoint, remote: str, local: Path, *,
         known_hosts: Path | None = None, timeout: int = _SCP_MIN_SEC) -> None:
    """Забрати файл із машини."""
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    cmd = [*_scp_base(ep, known_hosts=known_hosts), f"{ep.target}:{remote}",
           str(local)]
    _run(cmd, timeout=timeout, what=f"забір {posixpath.basename(remote)}")


def base_url(port: int = ORIGIN_PORT, *, host: str = "127.0.0.1",
             token: str = "") -> str:
    """Адреса складу. Типово — з погляду самої машини (петля).

    🔴 Секрет живе в ШЛЯХУ, а не в заголовку: раннер качає звичайним
    `curl <url>`, і будь-який заголовок довелось би проводити крізь нього,
    крізь `--retry`, крізь діапазонні з'єднання й крізь `fetch-ckpt`. Це рівно
    модель presigned-посилання R2, яку раннер уже споживає.

    Адреса для ДОМУ — та сама функція з публічним `host`: склад один, шляхи
    однакові, різна лише точка входу.
    """
    root = f"http://{host}:{port}"
    return f"{root}/{token.strip('/')}" if token else root


def start_origin(exec_fn: Any, upload_text: Any, *, port: int = ORIGIN_PORT,
                 root: str = REMOTE_ROOT, token: str = "") -> str:
    """Покласти склад на машину й підняти його. Повертає базову адресу (петля).

    `exec_fn(cmd) -> str` виконує команду на машині, `upload_text(path, text)`
    кладе туди текстовий файл — обидва дає бекенд: свого SSH тут немає
    навмисно, бо з'єднання вже підняте й друге було б другою автентифікацією.

    🔴 Секрет їде ФАЙЛОМ-запускачем, а не в рядку команди. Рядок запуску видно
    в `ps` кожному сусідові по хосту, а бокс багатоквартирний; префікс
    `VAR=... python3` теж видно — його бачить оболонка, і саме її командний
    рядок лишається в `ps`. Файл із правами 600 читає лише той, хто вже має
    машину.
    """
    from gpurunner.htr import origin as origin_mod

    code = Path(origin_mod.__file__).read_text(encoding="utf-8")
    remote_py = posixpath.join(root, "_origin.py")
    launcher = posixpath.join(root, "_origin.sh")
    exec_fn(f"mkdir -p {root}")
    upload_text(remote_py, code)
    upload_text(launcher,
                "#!/bin/sh\n"
                + (f"GPURUNNER_ORIGIN_TOKEN='{token}'\nexport GPURUNNER_ORIGIN_TOKEN\n"
                   if token else "")
                + f"exec python3 {remote_py} {root} {port}\n")
    exec_fn(f"chmod 600 {launcher}")
    # 🔴 `nohup` і `setsid`: склад мусить пережити закриття цієї SSH-сесії,
    # інакше він помре раніше за роботу, яку обслуговує, і раннер почне діставати
    # «connection refused» на першому ж чекпоінті.
    exec_fn(f"cd {root} && (setsid nohup sh {launcher} "
            f"> {root}/_origin.log 2>&1 &) ; sleep 1")
    said = exec_fn(f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 5 "
                   f"{base_url(port, token=token)}{origin_mod.HEALTH_PATH} || true")
    if "200" not in str(said):
        tail = exec_fn(f"tail -n 20 {root}/_origin.log 2>/dev/null || true")
        raise BackendError(f"склад на машині не піднявся: {str(tail).strip()[-300:]}")
    return base_url(port, token=token)


# ── плече дім → бокс по HTTP ────────────────────────────────────────────────
def _curl() -> str:
    exe = shutil.which("curl")
    if not exe:
        raise BackendError(
            "немає системного `curl` — ним їде заливка на машину. Постав curl "
            "або використай `--transport r2`.")
    return exe


def reachable(base: str, *, timeout: int = PROBE_TIMEOUT_SEC) -> bool:
    """Чи відповідає склад ЗВІДСИ. Це вимір, а не припущення.

    🔴 Питання не риторичне: порт складу мусить бути прокинутий назовні
    орендою, а чи прокинувся він — знає лише сама машина. Тому HTTP-заливка
    вмикається за ВІДПОВІДДЮ складу, а не за здогадом про хостера; не
    відповів — кажемо вголос і веземо `scp`.
    """
    from gpurunner.htr.origin import HEALTH_PATH

    cmd = [_curl(), "-s", "-o", os.devnull, "-w", "%{http_code}",
           "--max-time", str(timeout), f"{base}{HEALTH_PATH}"]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout + 5, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return "200" in (done.stdout or "")


def _put_slice(url: str, local: Path, offset: int, length: int,
               timeout: int) -> None:
    """Залити ШМАТОК файла окремим з'єднанням."""
    import urllib.error
    import urllib.request

    with local.open("rb") as fh:
        fh.seek(offset)
        body = fh.read(length)
    req = urllib.request.Request(url, data=body, method="PUT")
    req.add_header("Content-Length", str(len(body)))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status not in (200, 201, 204):
                raise BackendError(f"шматок {offset}: код {resp.status}")
    except urllib.error.HTTPError as exc:
        raise BackendError(f"шматок {offset}: код {exc.code}") from None
    except OSError as exc:
        raise BackendError(f"шматок {offset}: {exc}") from None


def push_http_parallel(base: str, local: Path, rel: str, *, assemble: Any,
                       parts: int = PUSH_PARTS) -> int:
    """Залити файл `parts` потоками й зібрати його на машині. Повертає розмір.

    🔴 Навіщо. Одне з'єднання на великому колі не вибирає каналу: заміряно
    23.09.2026 — 40 МБ одним потоком 3.83 МБ/с, вісьмома 8.94; а жива доставка
    135 МБ у Корею одним потоком дала 1.2 МБ/с при каналі 72 Мбіт/с, тобто
    всемеро менше за можливе. Протокол тут ні до чого — річ у кількості вікон.

    `assemble(rel, names) -> None` склеює шматки НА МАШИНІ (у наглядача для
    цього вже є відкрите з'єднання; свого SSH тут немає навмисно). Після
    склейки розмір звіряється тим самим каналом — обірваний шматок інакше дав
    би файл, який виглядає цілим.
    """
    local = Path(local)
    if not local.is_file():
        raise BackendError(f"немає файла для доставки: {local}")
    size = local.stat().st_size
    parts = max(1, int(parts))
    if parts == 1 or size < PUSH_PARALLEL_MIN_BYTES:
        # Дрібне не варте восьми рукостискань.
        return push_http(base, local, rel)

    from concurrent.futures import ThreadPoolExecutor

    chunk = -(-size // parts)
    timeout = int(max(_SCP_MIN_SEC, chunk / 1e6 / _HTTP_MB_PER_SEC))
    names = [f"{rel}.p{i:02d}" for i in range(parts)]
    jobs = [(f"{base.rstrip('/')}/{name}", i * chunk,
             min(chunk, size - i * chunk), timeout)
            for i, name in enumerate(names)]
    with ThreadPoolExecutor(max_workers=parts) as pool:
        futures = [pool.submit(_put_slice, url, local, off, length, tmo)
                   for url, off, length, tmo in jobs if length > 0]
        for fut in futures:
            fut.result()          # перша ж невдача валить доставку
    assemble(rel, [n for n, (_, _, length, _) in zip(names, jobs, strict=False)
                   if length > 0])
    landed = head_size(f"{base.rstrip('/')}/{rel.lstrip('/')}",
                       timeout=PROBE_TIMEOUT_SEC)
    if landed != size:
        raise BackendError(
            f"{local.name}: після склейки на машині {landed if landed >= 0 else '?'} "
            f"байтів із {size} — шматки не зійшлись. Далі це виглядало б як "
            f"битий архів уже під час роботи.")
    return size


def push_http(base: str, local: Path, rel: str) -> int:
    """Покласти файл у склад через `PUT`. Повертає розмір.

    🔴 Звірка обов'язкова й робиться ТИМ САМИМ каналом: обірвана заливка лишає
    файл, який виглядає покладеним, і виявляється це вже на оплачуваній карті —
    тим, що раннер не може розпакувати архів. Склад пише через `.part` і
    перейменовує лише ціле, тож `HEAD` на повний розмір — чесна відповідь
    «доїхало», а не «щось там лежить».
    """
    local = Path(local)
    if not local.is_file():
        raise BackendError(f"немає файла для доставки: {local}")
    size = local.stat().st_size
    url = f"{base.rstrip('/')}/{rel.lstrip('/')}"
    timeout = int(max(_SCP_MIN_SEC, size / 1e6 / _HTTP_MB_PER_SEC))
    _run([_curl(), "-fsS", "--retry", "2", "--retry-connrefused",
          "--max-time", str(timeout), "-X", "PUT", "--upload-file", str(local), url],
         timeout=timeout + 30, what=f"доставка {local.name}")
    landed = head_size(url, timeout=PROBE_TIMEOUT_SEC)
    if landed != size:
        raise BackendError(
            f"{local.name}: доїхало {landed if landed >= 0 else '?'} байтів "
            f"із {size} — заливка обірвалась. На машині це виглядало б як "
            f"битий архів уже під час роботи.")
    return size


def head_size(url: str, *, timeout: int = PROBE_TIMEOUT_SEC) -> int:
    """Розмір об'єкта за `HEAD`. -1 — немає або не відповіли."""
    try:
        done = subprocess.run(
            [_curl(), "-fsS", "-I", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 5, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return -1
    if done.returncode:
        return -1
    for line in (done.stdout or "").splitlines():
        name, _, value = line.partition(":")
        if name.strip().lower() == "content-length" and value.strip().isdigit():
            return int(value.strip())
    return -1


def ckpt_names(ckpt_prefix: str, slots: int) -> list[str]:
    """Імена чекпоінтів під префіксом — ОДНЕ джерело правди про них.

    Ними користуються і посилання складу, і рятунок (`htr fetch-ckpt`). Два
    списки, складені окремо, розійшлись би тихо — і помітно це стало б рівно в
    той день, коли точка відновлення знадобилась.
    """
    root = ckpt_prefix.strip("/")
    return [f"{root}/ckpt_{i:04d}.tgz" for i in range(1, max(1, slots) + 1)]


def urls_for(case_slug: str, ckpt_prefix: str, slots: int, *,
             port: int = ORIGIN_PORT, token: str = "") -> dict[str, Any]:
    """Адреси складу для однієї справи — у тій самій формі, що й у бакета.

    Раннер розрізняє «взяти» й «покласти» лише методом запиту, тож той самий
    шлях годиться і для `GET` чекпоінта, і для `PUT` — на відміну від
    presigned-посилань, де кожен напрям підписується окремо.
    """
    base = base_url(port, token=token)
    ckpt = [f"{base}/{name}" for name in ckpt_names(ckpt_prefix, slots)]
    return {
        "pages_url": f"{base}/cases/{case_slug}.tar",
        "ckpt_urls": list(ckpt),
        "resume_urls": list(ckpt),
        # Службовий архів лягає у той самий склад, тобто додому їде разом із
        # чекпоінтами — одним tar і одним з'єднанням, а не обходом тек.
        "service_put_url": f"{base}/{SERVICE_DIR}/{case_slug}.tgz",
    }


#: Тека службових архівів у складі — і на машині, і в бакеті однаково.
SERVICE_DIR = "service"


def remote_paths(case_slug: str, *, root: str = REMOTE_ROOT) -> dict[str, str]:
    """Куди на машині лягають файли складу."""
    return {
        "assets": posixpath.join(root, "assets.tgz"),
        "pages": posixpath.join(root, "cases", f"{case_slug}.tar"),
    }
