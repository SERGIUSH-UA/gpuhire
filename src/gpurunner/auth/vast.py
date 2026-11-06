"""Vast.ai credential discovery — API key + the SSH key used to reach instances.

Vast.ai is a GPU *marketplace*: you rent someone's box by the hour, it boots a
Docker container, and you reach it over SSH. So two secrets matter:

  - **API key** — drives the REST API (search offers, create/destroy instances).
    Discovered, in order: ``VAST_API_KEY`` env, our own ``config_dir()/vast_api_key``,
    the official CLI's locations (``%APPDATA%/vastai/vast_api_key``,
    ``~/.config/vastai/vast_api_key``, legacy ``~/.vast_api_key``).
  - **SSH key pair** — how outputs come back (SFTP). We only ever read the public
    half and hand it to Vast; the private half stays with your ssh-agent/openssh.
    Default ``~/.ssh/id_ed25519[.pub]``, then ``~/.ssh/id_rsa[.pub]``; override with
    ``GPURUNNER_VAST_SSH_KEY`` (path to the PRIVATE key). When none of those exist,
    ``ensure_ssh_key`` generates a dedicated pair under ``config_dir()/vast_ssh/`` —
    the rental path uses it so that an API key is the only thing a person brings.

⚠️ Unlike Kaggle/Colab this backend spends real money the moment an instance
boots, and it keeps spending until the instance is **destroyed** (not just when
the job ends). Everything here is written with that in mind.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

from gpurunner.config import config_dir
from gpurunner.core.backend import AuthError

API_BASE = "https://console.vast.ai/api/v0"


def _candidate_key_paths() -> list[Path]:
    paths = [config_dir() / "vast_api_key"]
    appdata = os.environ.get("APPDATA")
    if appdata:
        paths.append(Path(appdata) / "vastai" / "vast_api_key")
    paths.append(Path.home() / ".config" / "vastai" / "vast_api_key")
    paths.append(Path.home() / ".vast_api_key")  # legacy CLI location
    return paths


@dataclass
class VastCredentials:
    api_key: str
    source: str
    ssh_private: Path | None
    ssh_public: Path | None


_SETUP_HINT = (
    "Setup (once):\n"
    "  1. create an account at https://cloud.vast.ai/ and add credit (billing is per-hour)\n"
    "  2. Account → API Keys → copy the key\n"
    "  3. save it (no quotes, single line) to: {path}\n"
    "     or set the VAST_API_KEY environment variable\n"
    "  4. make sure you have an SSH key pair (~/.ssh/id_ed25519) — outputs come back over SFTP"
)


def find_api_key() -> tuple[str, str]:
    """Return ``(api_key, source)``. Raises AuthError when nothing is configured."""
    env = os.environ.get("VAST_API_KEY")
    if env and env.strip():
        return env.strip(), "VAST_API_KEY env"
    for p in _candidate_key_paths():
        if p.exists():
            try:
                key = p.read_text(encoding="utf-8").strip()
            except OSError as e:
                raise AuthError(f"Failed to read {p}: {e}") from e
            if key:
                return key, str(p)
    raise AuthError(
        "No Vast.ai API key found.\n" + _SETUP_HINT.format(path=config_dir() / "vast_api_key")
    )


def generated_ssh_key_path() -> Path:
    """Where ``ensure_ssh_key`` keeps the pair it made (the PRIVATE half)."""
    return config_dir() / "vast_ssh" / "id_ed25519"


def _pub_of(priv: Path) -> Path:
    return priv.with_suffix(priv.suffix + ".pub") if priv.suffix else Path(str(priv) + ".pub")


def find_ssh_key() -> tuple[Path | None, Path | None]:
    """Return ``(private, public)`` paths for the SSH key, or ``(None, None)``."""
    override = os.environ.get("GPURUNNER_VAST_SSH_KEY")
    candidates = [Path(override)] if override else []
    candidates += [Path.home() / ".ssh" / "id_ed25519", Path.home() / ".ssh" / "id_rsa"]
    # 🔴 Згенерована пара — ОСТАННЬОЮ. Власний ключ людини вже може бути
    # прописаний в акаунті Vast і в її `ssh_config`; підмінити його своїм
    # означало б зламати їй ручний `ssh` на бокс без жодного повідомлення.
    candidates.append(generated_ssh_key_path())
    for priv in candidates:
        pub = _pub_of(priv)
        if priv.exists() and pub.exists():
            return priv, pub
    return None, None


def ensure_ssh_key() -> tuple[Path, Path]:
    """``find_ssh_key``, а коли ключа немає ніде — згенерувати власну пару.

    Навіщо: сторонній людині, яка принесла лише ключ API, порада «запусти
    ``ssh-keygen``» — це зупинка на рівному місці (на Windows його може не бути
    взагалі). Пара лягає в ``config_dir()/vast_ssh/`` і далі знаходиться
    ``find_ssh_key`` нарівні з рештою, тож увесь код бекенда (``_ssh``,
    ``_attach_ssh_key``) бачить її без жодної правки.

    🔴 Наявний файл НЕ перезаписується ніколи: приватний ключ пишеться з
    ``O_EXCL``. Ключ, затертий новим, — це живий орендований бокс, до якого
    більше нема чим зайти, тобто гроші без результату.
    """
    priv, pub = find_ssh_key()
    if priv is not None and pub is not None:
        return priv, pub

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as e:  # cryptography приходить разом із paramiko
        raise AuthError(
            f"cannot generate an SSH key: {e}\nRun: uv sync --extra vast "
            f"(or: pip install 'gpuhire[vast]')"
        ) from e

    priv = generated_ssh_key_path()
    pub = _pub_of(priv)
    priv.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(priv.parent, 0o700)

    if priv.exists():
        # Приватна половина є, публічної немає (стерли руками, обірваний запис).
        # Публічну відновлюємо З приватної — генерувати нову пару означало б
        # затерти ключ, яким, можливо, вже відкрито живий бокс.
        try:
            loaded = serialization.load_ssh_private_key(priv.read_bytes(), password=None)
        except Exception as e:
            raise AuthError(
                f"{priv} exists but is not a readable OpenSSH private key ({e}). "
                f"Move it away (gpurunner never overwrites a key) and retry."
            ) from e
        public = loaded.public_key()
    else:
        if pub.exists():
            raise AuthError(
                f"{pub} exists without its private half. Move it away "
                f"(gpurunner never overwrites a key) and retry."
            )
        key = Ed25519PrivateKey.generate()
        blob = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption(),
        )
        _write_new(priv, blob, mode=0o600)
        public = key.public_key()

    line = public.public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    )
    _write_new(pub, line + b" gpurunner-vast\n", mode=0o644)
    return priv, pub


def _write_new(path: Path, blob: bytes, *, mode: int) -> None:
    """Створити файл, якого ще немає. Наявний — помилка, а не перезапис.

    ``O_EXCL`` робить перевірку й створення однією дією: дві сесії, що
    стартували разом, не затруть ключ одна одній. Права 0600 діють на POSIX;
    на Windows ``mode`` зводиться до прапорця «лише читання», тож там ключ
    захищає ACL профілю користувача, у якому лежить ``config_dir()``.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags, mode)
    except FileExistsError as e:
        raise AuthError(f"{path} already exists — gpurunner never overwrites a key") from e
    with os.fdopen(fd, "wb") as fh:
        fh.write(blob)


def save_api_key(key: str) -> Path:
    """Зберегти ключ API у ``config_dir()/vast_api_key``. Повертає шлях.

    🔴 Сам ключ звідси нікуди не йде: ні в лог, ні у виняток, ні у значення, що
    повертається. Перевірка форми навмисно мінімальна (один рядок без
    пробілів) — формат ключа належить Vast і вже мінявся.
    """
    clean = (key or "").strip()
    if not clean or any(ch.isspace() for ch in clean):
        raise AuthError("the Vast.ai API key must be a single non-empty token without spaces")
    path = config_dir() / "vast_api_key"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Через тимчасовий файл: обірваний запис не має лишити половину ключа там,
    # де попередній працював.
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    _write_new(tmp, clean.encode("utf-8") + b"\n", mode=0o600)
    os.replace(tmp, path)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    return path


def discover_credentials() -> VastCredentials:
    key, source = find_api_key()
    priv, pub = find_ssh_key()
    return VastCredentials(api_key=key, source=source, ssh_private=priv, ssh_public=pub)


def require_ssh_key() -> tuple[Path, Path]:
    """Same as ``find_ssh_key`` but raises with instructions when absent."""
    priv, pub = find_ssh_key()
    if priv is None or pub is None:
        raise AuthError(
            "No SSH key pair found (~/.ssh/id_ed25519[.pub] or ~/.ssh/id_rsa[.pub]).\n"
            "Vast.ai instances are reached over SSH — outputs are pulled back with SFTP.\n"
            "Create one:  ssh-keygen -t ed25519 -C gpurunner\n"
            "Or point GPURUNNER_VAST_SSH_KEY at an existing private key."
        )
    return priv, pub


def verify() -> dict[str, object]:
    """Probe ``GET /users/current``. Returns ``{email, balance, ssh_key_count}``."""
    import httpx

    key, _ = find_api_key()
    try:
        resp = httpx.get(
            f"{API_BASE}/users/current/",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        raise AuthError(
            f"Vast.ai rejected the API key ({e.response.status_code}). "
            "Check Account → API Keys."
        ) from e
    except httpx.HTTPError as e:
        raise AuthError(f"Vast.ai probe failed: {e}") from e
    return {
        "email": data.get("email"),
        "balance": data.get("credit", data.get("balance")),
        "ssh_key_count": len(data.get("ssh_keys") or []),
    }
