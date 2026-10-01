"""SSH-ключ без рук і збереження ключа API.

Сторонній людині, яка принесла лише ключ API Vast, порада «запусти ssh-keygen»
— зупинка на рівному місці. Тому оренда заводить пару сама; але чужого ключа
вона не чіпає НІКОЛИ: затертий ключ — це живий бокс, до якого нема чим зайти.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpurunner.auth import vast as vast_auth
from gpurunner.core.backend import AuthError


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Порожній дім: ні справжнього `~/.ssh`, ні справжнього ключа API."""
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("GPURUNNER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("GPURUNNER_VAST_SSH_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    return home


def test_key_is_generated_in_the_config_dir(tmp_path: Path) -> None:
    assert vast_auth.find_ssh_key() == (None, None)
    priv, pub = vast_auth.ensure_ssh_key()

    assert priv == tmp_path / "config" / "vast_ssh" / "id_ed25519"
    assert pub == tmp_path / "config" / "vast_ssh" / "id_ed25519.pub"
    assert priv.read_text(encoding="ascii").startswith("-----BEGIN OPENSSH PRIVATE KEY-----")
    assert pub.read_text(encoding="ascii").startswith("ssh-ed25519 ")
    assert vast_auth.find_ssh_key() == (priv, pub), "далі знаходиться нарівні з рештою"
    assert vast_auth.require_ssh_key() == (priv, pub)
    if os.name == "posix":
        assert (priv.stat().st_mode & 0o777) == 0o600


def test_second_call_does_not_overwrite() -> None:
    priv, pub = vast_auth.ensure_ssh_key()
    before = (priv.read_bytes(), pub.read_bytes())
    assert vast_auth.ensure_ssh_key() == (priv, pub)
    assert (priv.read_bytes(), pub.read_bytes()) == before


def test_lost_public_half_is_rebuilt_from_the_private_one() -> None:
    """Нова пара тут затерла б ключ, яким уже може бути відкритий живий бокс."""
    priv, pub = vast_auth.ensure_ssh_key()
    secret, line = priv.read_bytes(), pub.read_bytes()
    pub.unlink()
    assert vast_auth.ensure_ssh_key() == (priv, pub)
    assert priv.read_bytes() == secret and pub.read_bytes() == line


def test_orphan_public_half_is_not_overwritten(tmp_path: Path) -> None:
    pub = tmp_path / "config" / "vast_ssh" / "id_ed25519.pub"
    pub.parent.mkdir(parents=True)
    pub.write_text("ssh-ed25519 AAAA чужий", encoding="utf-8")
    with pytest.raises(AuthError):
        vast_auth.ensure_ssh_key()
    assert pub.read_text(encoding="utf-8") == "ssh-ed25519 AAAA чужий"


def test_generated_key_loads_with_the_backend_loader() -> None:
    """Тим самим кодом, яким бекенд відкриває ключ перед `connect`."""
    paramiko = pytest.importorskip("paramiko")
    from gpurunner.backends.vast import _load_private_key

    priv, pub = vast_auth.ensure_ssh_key()
    key = _load_private_key(priv)
    assert isinstance(key, paramiko.Ed25519Key)
    kind, blob = pub.read_text(encoding="ascii").split()[:2]
    assert kind == key.get_name() == "ssh-ed25519"
    assert blob == key.get_base64(), "публічна половина — від ЦЬОГО приватного"


def test_generated_key_loads_with_the_nyshporka_loader() -> None:
    """…і тим, яким його відкриє транспорт Нишпорки."""
    pytest.importorskip("paramiko")
    ssh = pytest.importorskip("nyshporka.cloud.ssh")

    priv, _ = vast_auth.ensure_ssh_key()
    assert ssh._load_key(priv).get_name() == "ssh-ed25519"


def test_own_key_in_home_wins(_isolated: Path, tmp_path: Path) -> None:
    mine = _isolated / ".ssh" / "id_rsa"
    mine.write_text("PRIVATE", encoding="utf-8")
    (_isolated / ".ssh" / "id_rsa.pub").write_text("ssh-rsa AAAA me", encoding="utf-8")

    assert vast_auth.ensure_ssh_key() == (mine, _isolated / ".ssh" / "id_rsa.pub")
    assert not (tmp_path / "config" / "vast_ssh").exists(), "свого не заводимо без потреби"


def test_own_key_still_wins_after_one_was_generated(_isolated: Path) -> None:
    generated, _ = vast_auth.ensure_ssh_key()
    mine = _isolated / ".ssh" / "id_ed25519"
    mine.write_text("PRIVATE", encoding="utf-8")
    (_isolated / ".ssh" / "id_ed25519.pub").write_text("ssh-ed25519 AAAA me", encoding="utf-8")
    assert vast_auth.find_ssh_key()[0] == mine != generated


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    priv = tmp_path / "elsewhere" / "vast_key"
    priv.parent.mkdir()
    priv.write_text("PRIVATE", encoding="utf-8")
    Path(str(priv) + ".pub").write_text("ssh-ed25519 AAAA env", encoding="utf-8")
    monkeypatch.setenv("GPURUNNER_VAST_SSH_KEY", str(priv))
    assert vast_auth.ensure_ssh_key()[0] == priv


# ---- ключ API --------------------------------------------------------------


def test_save_api_key_round_trip(tmp_path: Path) -> None:
    path = vast_auth.save_api_key("  abc123  \n")
    assert path == tmp_path / "config" / "vast_api_key"
    assert vast_auth.find_api_key() == ("abc123", str(path))
    vast_auth.save_api_key("newer")            # новий ключ заміщає старий
    assert vast_auth.find_api_key()[0] == "newer"
    if os.name == "posix":
        assert (path.stat().st_mode & 0o777) == 0o600


@pytest.mark.parametrize("bad", ["", "   ", "two words", "line\nbreak"])
def test_save_api_key_rejects_garbage(bad: str, tmp_path: Path) -> None:
    with pytest.raises(AuthError) as exc:
        vast_auth.save_api_key(bad)
    assert not (tmp_path / "config" / "vast_api_key").exists()
    assert bad.strip() == "" or bad.strip() not in str(exc.value)


def test_cli_auth_vast_key_saves_and_never_prints_it(tmp_path: Path) -> None:
    from gpurunner.cli import app

    result = CliRunner().invoke(app, ["auth", "vast", "--key", "sekret-key-987"])
    assert result.exit_code == 0, result.output
    saved = tmp_path / "config" / "vast_api_key"
    assert saved.read_text(encoding="utf-8").strip() == "sekret-key-987"
    assert "sekret-key-987" not in result.output, "🔴 ключ не друкується ніколи"
    assert "vast_api_key" in result.output.replace("\n", "")


def test_cli_auth_vast_key_from_stdin(tmp_path: Path) -> None:
    from gpurunner.cli import app

    result = CliRunner().invoke(app, ["auth", "vast", "--key", "-"], input="from-stdin-55\n")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "config" / "vast_api_key").read_text(encoding="utf-8").strip() \
        == "from-stdin-55"
    assert "from-stdin-55" not in result.output
