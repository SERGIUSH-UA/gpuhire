"""Доставка без бакета: файли везе `scp`, посилання видає сама машина.

Половина, що живе вдома. Друга половина — `htr/origin.py`, крихітний склад,
який ми кладемо на машину й піднімаємо на її петлі; раннер качає з нього тим
самим `curl`, що й з бакета, і не знає про підміну нічого.

Чому `scp`, а не SFTP того ж з'єднання, яким наглядач і так керує машиною:
виміряно на орендованому боксі 20.09.2026, 150 МБ — SFTP paramiko 0.48 МБ/с,
зворотний тунель paramiko 0.49 МБ/с, системний `scp` 7.56 МБ/с. Різниця не в
смузі, а в тому, що paramiko жене дані одним вікном; для архіву ассетів на
105 МБ це три з половиною хвилини оплаченого простою проти чверті хвилини.

🔴 Чекпоінти, складені на цю машину, гинуть разом із нею — на відміну від
чекпоінтів у бакеті. Тому `pull_checkpoints` тут не зручність, а єдине, що
відрізняє «бокс помер, дочитаємо з місця» від «бокс помер, платимо все
наново»; наглядач кличе його періодично, а не в кінці.
"""
from __future__ import annotations

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
        raise BackendError(f"{what}: scp rc={done.returncode} {' '.join(tail)}".strip())


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


def base_url(port: int = ORIGIN_PORT) -> str:
    """Адреса складу з погляду самої машини. Лише петля — назовні він закритий."""
    return f"http://127.0.0.1:{port}"


def start_origin(exec_fn: Any, upload_text: Any, *, port: int = ORIGIN_PORT,
                 root: str = REMOTE_ROOT) -> str:
    """Покласти склад на машину й підняти його. Повертає базову адресу.

    `exec_fn(cmd) -> str` виконує команду на машині, `upload_text(path, text)`
    кладе туди текстовий файл — обидва дає бекенд: свого SSH тут немає
    навмисно, бо з'єднання вже підняте й друге було б другою автентифікацією.
    """
    from gpurunner.htr import origin as origin_mod

    code = Path(origin_mod.__file__).read_text(encoding="utf-8")
    remote_py = posixpath.join(root, "_origin.py")
    exec_fn(f"mkdir -p {root}")
    upload_text(remote_py, code)
    # 🔴 `nohup` і `setsid`: склад мусить пережити закриття цієї SSH-сесії,
    # інакше він помре раніше за роботу, яку обслуговує, і раннер почне діставати
    # «connection refused» на першому ж чекпоінті.
    exec_fn(f"cd {root} && (setsid nohup python3 {remote_py} {root} {port} "
            f"> {root}/_origin.log 2>&1 &) ; sleep 1")
    said = exec_fn(f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 5 "
                   f"{base_url(port)}/_origin.py || true")
    if "200" not in str(said):
        tail = exec_fn(f"tail -n 20 {root}/_origin.log 2>/dev/null || true")
        raise BackendError(f"склад на машині не піднявся: {str(tail).strip()[-300:]}")
    return base_url(port)


def urls_for(case_slug: str, ckpt_prefix: str, slots: int, *,
             port: int = ORIGIN_PORT) -> dict[str, Any]:
    """Адреси складу для однієї справи — у тій самій формі, що й у бакета.

    Раннер розрізняє «взяти» й «покласти» лише методом запиту, тож той самий
    шлях годиться і для `GET` чекпоінта, і для `PUT` — на відміну від
    presigned-посилань, де кожен напрям підписується окремо.
    """
    base = base_url(port)
    ckpt = [f"{base}/{ckpt_prefix.strip('/')}/ckpt_{i:04d}.tgz"
            for i in range(1, max(1, slots) + 1)]
    return {
        "pages_url": f"{base}/cases/{case_slug}.tar",
        "ckpt_urls": list(ckpt),
        "resume_urls": list(ckpt),
    }


def remote_paths(case_slug: str, *, root: str = REMOTE_ROOT) -> dict[str, str]:
    """Куди на машині лягають файли складу."""
    return {
        "assets": posixpath.join(root, "assets.tgz"),
        "pages": posixpath.join(root, "cases", f"{case_slug}.tar"),
    }
