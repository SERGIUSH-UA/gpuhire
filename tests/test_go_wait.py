"""Очікування даних на боксі — справжнім bash, із секундами замість годин.

🔴 9.10.2026 раннер чекав `GO` рівно пів години, а доставка 25 ГБ тривала 77
хвилин: на 30-й він вийшов з `inputs never arrived`, і дані доїхали в
порожнечу. Тепер бокс чекає до строку, який назвав наглядач (`GO_UNTIL`).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import gpurunner.backends.vast as vast


def _posix_bash() -> str | None:
    """bash із POSIX-утилітами. На Windows — той, що з Git: `bash` у PATH там
    часто веде у WSL, де шляхи й оточення інші."""
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
    s = p.resolve().as_posix()
    if os.name == "nt" and len(s) > 1 and s[1] == ":":
        return f"/{s[0].lower()}{s[2:]}"
    return s


def _start(tmp_path: Path, floor: int) -> tuple[subprocess.Popen[bytes], Path]:
    root = tmp_path / "box"
    root.mkdir()
    script = vast._render_go_wait(root=_sh(root), floor_seconds=floor, poll_seconds=1)
    proc = subprocess.Popen([BASH, "-c", script + "echo started > " + _sh(root / "ran") + "\n"],  # type: ignore[list-item]
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, root


def test_without_a_promise_the_floor_holds_and_failure_is_written(tmp_path: Path) -> None:
    proc, root = _start(tmp_path, floor=1)

    assert proc.wait(timeout=20) == 1
    status = json.loads((root / "_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "inputs never arrived" in status["error"]
    assert not (root / "ran").exists()


def test_a_promise_keeps_the_box_waiting_past_the_floor(tmp_path: Path) -> None:
    """🔴 Саме той випадок: підлога вичерпалась, а доставка ще йде."""
    proc, root = _start(tmp_path, floor=1)
    (root / "GO_UNTIL").write_text(str(int(time.time()) + 30), encoding="utf-8")
    time.sleep(4)                          # підлога вже минула
    assert proc.poll() is None, "бокс здався, хоч наглядач пообіцяв дані"
    (root / "GO").write_text("", encoding="utf-8")

    assert proc.wait(timeout=20) == 0
    assert (root / "ran").exists()
    assert not (root / "_status.json").exists()


def test_a_stale_or_garbled_promise_does_not_extend_the_wait(tmp_path: Path) -> None:
    proc, root = _start(tmp_path, floor=1)
    (root / "GO_UNTIL").write_text("сміття", encoding="utf-8")

    assert proc.wait(timeout=20) == 1


def test_the_onstart_waits_by_the_promise() -> None:
    script = vast.VastBackend()._render_onstart(max_hours=1, autodestroy_hours=0.5)
    assert "GO_UNTIL" in script
    assert "seq 1 900" not in script, "стала пів години повернулась"


# ── Мертвий наглядач: бокс без даних гасить себе сам ────────────────────────
#
# 🔴 14.10.2026 пісочниця Codex убила наглядача одразу після `nysh cloud go`.
# Бокси чекали `GO` до обіцяного строку, потім робили `exit 1` і покладались на
# фонового дедлайн-сторожа — тобто горіли без роботи до `max_hours` + 30 хв.


def _start_armed(tmp_path: Path, *, floor: int, stall: int,
                 env_box: bool = True) -> tuple[subprocess.Popen[bytes], Path, Path, Path]:
    root = tmp_path / "box"
    root.mkdir()
    origin = tmp_path / "origin"
    origin.mkdir()
    fake = tmp_path / "fakebin"
    fake.mkdir()
    calls = tmp_path / "curl_calls.txt"
    (fake / "curl").write_text(f'#!/bin/sh\necho "$@" >> "{_sh(calls)}"\n', newline="\n")
    (fake / "curl").chmod(0o755)
    script = vast._render_go_wait(
        root=_sh(root), floor_seconds=floor, poll_seconds=1, stall_seconds=stall,
        pulse=(_sh(origin), _sh(root / "GO_UNTIL")), tmp_dir=_sh(tmp_path / "tmp"),
        api_base="http://127.0.0.1:9")
    body = f'export PATH="{_sh(fake)}:$PATH"\n{script}echo started > {_sh(root / "ran")}\n'
    env = {k: v for k, v in os.environ.items()
           if k not in ("CONTAINER_API_KEY", "CONTAINER_ID")}
    if env_box:
        env.update(CONTAINER_API_KEY="k", CONTAINER_ID="42")
    proc = subprocess.Popen([BASH, "-c", body], env=env,  # type: ignore[list-item]
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, root, origin, calls


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vast, "_DESTROY_TRIES", 2)
    monkeypatch.setattr(vast, "_DESTROY_PAUSE_SEC", 1)


def _old(path: Path, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


def test_no_go_destroys_the_box_in_the_foreground(tmp_path: Path, fast: None) -> None:
    proc, root, _, calls = _start_armed(tmp_path, floor=1, stall=60)
    try:
        assert proc.wait(timeout=30) == 1
    finally:
        proc.kill()
    sent = calls.read_text().splitlines()
    assert len(sent) == 2, "знищення шлеться з повтором, а не покладається на фон"
    assert all("-X DELETE" in s and "/instances/42/" in s for s in sent)
    assert "inputs never arrived" in (root / "_status.json").read_text(encoding="utf-8")
    assert not (root / "ran").exists()


def test_a_dead_supervisor_s_promise_does_not_hold_the_box(tmp_path: Path, fast: None) -> None:
    """Обіцянка на годину вперед, але доставка мовчить довше за вікно — гасимо."""
    proc, root, _, calls = _start_armed(tmp_path, floor=1, stall=3)
    until = root / "GO_UNTIL"
    until.write_text(str(int(time.time()) + 3600), encoding="utf-8")
    _old(until, 30)                   # наглядач поклав строк давно й замовк
    try:
        assert proc.wait(timeout=30) == 1
    finally:
        proc.kill()
    assert len(calls.read_text().splitlines()) == 2
    assert "доставка мовчить" in (root / "_runner.log").read_text(encoding="utf-8")


def test_a_live_delivery_keeps_the_box_waiting_past_the_floor(tmp_path: Path, fast: None) -> None:
    """Повільна, але жива доставка (росте `.part` складу) бокс не гасить."""
    proc, root, origin, calls = _start_armed(tmp_path, floor=1, stall=3)
    until = root / "GO_UNTIL"
    until.write_text(str(int(time.time()) + 3600), encoding="utf-8")
    _old(until, 30)
    part = origin / "cases.tar.part"
    try:
        for _ in range(6):            # 6 с — удвічі довше за вікно застою
            with part.open("ab") as fh:
                fh.write(b"x" * 1024)
            time.sleep(1)
        assert proc.poll() is None, "бокс здався посеред живої доставки"
        (root / "GO").write_text("", encoding="utf-8")
        assert proc.wait(timeout=20) == 0
    finally:
        proc.kill()
    assert (root / "ran").exists()
    assert not calls.exists()


def test_a_fresh_brake_postpones_destruction(tmp_path: Path, fast: None) -> None:
    proc, root, _, calls = _start_armed(tmp_path, floor=1, stall=60)
    brake = root / "NOSTOP"
    brake.write_text("")
    try:
        time.sleep(4)
        assert proc.poll() is None and not calls.exists(), "свіжий NOSTOP — гасити не можна"
        _old(brake, 31 * 60)
        assert proc.wait(timeout=30) == 1
    finally:
        proc.kill()
    assert len(calls.read_text().splitlines()) == 2


def test_without_container_keys_the_box_says_it_is_still_billing(tmp_path: Path) -> None:
    proc, root, _, calls = _start_armed(tmp_path, floor=1, stall=60, env_box=False)
    try:
        assert proc.wait(timeout=20) == 1
    finally:
        proc.kill()
    assert not calls.exists()
    assert "ТАРИФІКАЦІЯ ЙДЕ" in (root / "_runner.log").read_text(encoding="utf-8")


def test_the_onstart_go_wait_watches_the_delivery_paths() -> None:
    from gpurunner.htr import box_transport as bt

    script = vast.VastBackend()._render_onstart(max_hours=1, autodestroy_hours=0)
    wait = script[script.index("waiting for GO"):script.index("cd /workspace/gpurunner")]
    assert bt.REMOTE_ROOT in wait and vast.REMOTE_INPUT in wait
    assert "/instances/$CONTAINER_ID/" in wait, "знищення — у самій гілці, не лише у фоні"


# ── 14.10.2026: дірки самознищення поза наглядачем ─────────────────────────


def test_the_onstart_waits_for_its_background_guard() -> None:
    """Фоновий дедлайн-сторож не мусить пережити кінець onstart: скрипт чекає."""
    script = vast.VastBackend()._render_onstart(max_hours=1, autodestroy_hours=0)
    lines = [ln.strip() for ln in script.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert lines[-1] == "wait"
    assert script.index("job.py") < script.rindex("wait")


@pytest.mark.parametrize(("given", "want"), [
    ({}, vast.DEFAULT_AUTODESTROY_HOURS), ({"autodestroy_hours": 0}, 0.0),
    ({"autodestroy_hours": "0.5"}, 0.5)])
def test_plain_vast_runs_destroy_themselves_after_two_hours_by_default(given, want) -> None:
    from gpurunner.jobs import get_job

    job = get_job("htr_case")
    assert vast._backend_opts(given, job)["autodestroy_hours"] == want
    assert vast.DEFAULT_AUTODESTROY_HOURS == 2.0


class _Box:
    def __init__(self, log: str) -> None:
        self.log = log


def _checker(monkeypatch, log: str) -> vast.VastBackend:
    bk = vast.VastBackend()
    monkeypatch.setattr(vast.VastBackend, "_exec", staticmethod(lambda client, cmd: client.log))
    return bk


def test_a_box_that_cannot_destroy_itself_is_rejected_for_the_supervisor(monkeypatch) -> None:
    bk = _checker(monkeypatch, "")
    box = _Box("[gpurunner] ⚠ немає CONTAINER_API_KEY/CONTAINER_ID — дедлайн-кілера НЕМАЄ\n")
    with pytest.raises(vast.BackendError) as exc:
        bk._check_self_destruct(box, required=True)
    assert exc.value.outcome == "no_self_destruct"


def test_plain_run_only_warns_about_such_a_box(monkeypatch, capsys) -> None:
    bk = _checker(monkeypatch, "")
    box = _Box("[gpurunner] ⚠ немає CONTAINER_API_KEY/CONTAINER_ID — дедлайн-кілера НЕМАЄ\n")
    bk._check_self_destruct(box, required=False)
    assert "gpurunner cancel" in capsys.readouterr().out


def test_an_armed_box_passes_and_a_silent_one_is_not_rejected(monkeypatch, capsys) -> None:
    bk = _checker(monkeypatch, "")
    bk._check_self_destruct(_Box("[gpurunner] дедлайн-кілер зведено на 45000s\n"), required=True)
    bk._check_self_destruct(_Box(""), required=True, wait_sec=0)
    assert "не вдалось" in capsys.readouterr().out
