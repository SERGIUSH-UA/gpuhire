"""Сторожі самознищення на боксі — справжнім bash, а не пошуком рядків у шаблоні.

🔴🔴 Обидва сторожі шляху з наглядачем (дедлайн і «по результат ніхто не
прийшов») дивились на стоп-кран `NOSTOP` ОДИН раз: свіжий — `exit 0`, і більше
не поверталися. Кран свіжий саме під час забору, тож дедлайн посеред забору плюс
смерть наглядача (чи вимкнений ПК) давали бокс, який уже ніщо не погасить. Те,
що кран «сам протухає за 30 хвилин», нічого не давало: на нього ніхто не
дивився. До того ж знищення там слалось один раз, без повторів.

Тут шаблон рендериться з секундами замість годин і виконується bash'ем; `curl`
підмінено записувачем, тож у мережу не йде нічого.
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import time
import types
from pathlib import Path

import pytest

import gpurunner.backends.vast as vast


def _posix_bash() -> str | None:
    """bash із POSIX-утилітами (`find -mmin`, `seq`). На Windows — той, що з Git:
    `bash` у PATH там часто веде у WSL, де шляхи й оточення інші."""
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    if not git:
        return None
    for up in Path(git).resolve().parents[:3]:
        cand = up / "bin" / "bash.exe"
        if cand.exists():
            return str(cand)
    return None


BASH = _posix_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="немає POSIX bash")


def _sh(p: Path) -> str:
    """Шлях, як його бачить bash: `C:\\x` → `/c/x` у Git Bash."""
    s = p.resolve().as_posix()
    if os.name == "nt" and len(s) > 1 and s[1] == ":":
        return f"/{s[0].lower()}{s[2:]}"
    return s


def _start(script: str, tmp_path: Path) -> tuple[subprocess.Popen[bytes], Path]:
    fake = tmp_path / "fakebin"
    fake.mkdir()
    calls = tmp_path / "curl_calls.txt"
    (fake / "curl").write_text(f'#!/bin/sh\necho "$@" >> "{_sh(calls)}"\n', newline="\n")
    (fake / "curl").chmod(0o755)
    body = f'export PATH="{_sh(fake)}:$PATH"\n{script}\nwait\n'
    env = {**os.environ, "CONTAINER_API_KEY": "k", "CONTAINER_ID": "42"}
    proc = subprocess.Popen([BASH, "-c", body], env=env,  # type: ignore[list-item]
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, calls


def _age(path: Path, minutes: float) -> None:
    old = time.time() - minutes * 60
    os.utime(path, (old, old))


GUARDS = {
    "deadline": lambda root: vast._render_deadline_guard(
        deadline_seconds=1, root=root, api_base="http://127.0.0.1:9"),
    "nobody_came": lambda root: vast._render_autodestroy(
        grace_seconds=1, root=root, api_base="http://127.0.0.1:9"),
}


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vast, "_DESTROY_TRIES", 3)
    monkeypatch.setattr(vast, "_DESTROY_PAUSE_SEC", 1)


@pytest.mark.parametrize("guard", list(GUARDS))
def test_fresh_brake_postpones_destruction_but_never_cancels_it(
        guard: str, tmp_path: Path, fast: None) -> None:
    root = tmp_path / "box"
    root.mkdir()
    brake = root / "NOSTOP"
    brake.write_text("")
    proc, calls = _start(GUARDS[guard](_sh(root)), tmp_path)
    try:
        time.sleep(4)
        assert not calls.exists(), "кран свіжий — іде забір, гасити не можна"
        assert proc.poll() is None, "🔴 сторож вийшов на свіжому крані — бокс без сторожа"

        _age(brake, 31)   # наглядач помер: кран більше ніхто не підновлює
        proc.wait(timeout=30)
    finally:
        proc.kill()

    sent = calls.read_text().splitlines()
    assert len(sent) == 3, "знищення повторюється, а не шлеться один раз"
    assert all("-X DELETE" in s and "/instances/42/" in s for s in sent)
    log = (root / "_runner.log").read_text(encoding="utf-8")
    assert log.count("чекаю, доки протухне") == 1, "про очікування — один рядок, не щохвилини"


@pytest.mark.parametrize("guard", list(GUARDS))
def test_no_brake_destroys_with_retries(guard: str, tmp_path: Path, fast: None) -> None:
    root = tmp_path / "box"
    root.mkdir()
    proc, calls = _start(GUARDS[guard](_sh(root)), tmp_path)
    try:
        proc.wait(timeout=30)
    finally:
        proc.kill()
    assert len(calls.read_text().splitlines()) == 3


def test_stale_brake_does_not_hold_the_box(tmp_path: Path, fast: None) -> None:
    """Кран, забутий мертвим наглядачем, не тримає бокс ані хвилини."""
    root = tmp_path / "box"
    root.mkdir()
    brake = root / "NOSTOP"
    brake.write_text("")
    _age(brake, 45)
    proc, calls = _start(GUARDS["deadline"](_sh(root)), tmp_path)
    try:
        proc.wait(timeout=30)
    finally:
        proc.kill()
    assert len(calls.read_text().splitlines()) == 3
    assert "чекаю" not in (root / "_runner.log").read_text(encoding="utf-8")


def test_long_fetch_keeps_the_brake_fresh(monkeypatch: pytest.MonkeyPatch,
                                          tmp_path: Path) -> None:
    """Сторож тепер ЧЕКАЄ на свіжому крані й гасить на протухлому — отже забір,
    довший за 30 хв, мусить кран підновлювати, інакше бокс погасне посеред
    качання."""
    from tests.test_vast_backend import _FakeSFTP, _handle, _TestBackend

    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_CONFIG_DIR", str(tmp_path / "config"))
    key = tmp_path / "id_ed25519"
    key.write_text("PRIVATE", encoding="utf-8")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test", encoding="utf-8")
    monkeypatch.setenv("GPURUNNER_VAST_SSH_KEY", str(key))
    bk = _TestBackend()
    h = _handle(bk)   # оренда — ДО підміни годинника: її цикли чекання на ньому
    for i in range(6):
        bk.remote_files[f"{vast.REMOTE_WORKING}/c/{i:04d}.txt"] = b"x"

    now = [1000.0]
    monkeypatch.setattr(vast, "time", types.SimpleNamespace(
        monotonic=lambda: now[0], time=time.time, sleep=time.sleep))

    real_get, real_open = _FakeSFTP.get, _FakeSFTP.open
    brake_writes: list[float] = []

    def slow_get(self, remote: str, local: str) -> None:
        now[0] += 400.0          # кожен файл — майже 7 хвилин
        real_get(self, remote, local)

    def spy_open(self, remote: str, mode: str = "r"):
        if remote.endswith("/NOSTOP") and "w" in mode:
            brake_writes.append(now[0])
        return real_open(self, remote, mode)

    monkeypatch.setattr(_FakeSFTP, "get", slow_get)
    monkeypatch.setattr(_FakeSFTP, "open", spy_open)

    written = bk.fetch_outputs(h, tmp_path / "out")

    assert len([p for p in written if p.suffix == ".txt"]) == 6
    assert len(brake_writes) >= 1 + 5, "кран підновлюється під час довгого забору"
    gaps = [b - a for a, b in itertools.pairwise(brake_writes)]
    assert max(gaps) < vast._NOSTOP_FRESH_MIN * 60, "між підновленнями кран не встигає протухнути"
