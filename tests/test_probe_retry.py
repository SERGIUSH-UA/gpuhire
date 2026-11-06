"""Проба заліза переживає один обрив SSH свіжого боксу.

52409272 (24.09.2026, ssh3.vast.ai): SSH уже відповів, наступне з'єднання для
проби один раз обірвалось — і машину погасили за 68 с як `ssh_unreachable`.
"""
from __future__ import annotations

import io

import pytest

from gpurunner.backends import vast as vast_mod
from gpurunner.backends.vast import VastBackend
from gpurunner.core.backend import BackendError, SshAuthRejected
from gpurunner.core.models import JobHandle

PROBE_OUT = "cores=16\ngpu=RTX 3060\nvram_total_mb=12000\nvram_free_mb=11900\nnet_bps=1000000\n"


class _Chan:
    def recv_exit_status(self):
        return 0


class _Out(io.BytesIO):
    channel = _Chan()


class _Client:
    def exec_command(self, script, timeout=None):
        return None, _Out(PROBE_OUT.encode()), None

    def close(self):
        pass


def _handle() -> JobHandle:
    return JobHandle(backend="vast", remote_id="1", job_name="htr_case", gpu="any")


def test_one_dropped_connection_does_not_kill_the_box(monkeypatch) -> None:
    monkeypatch.setattr(vast_mod, "_PROBE_RETRY_SEC", 0)
    calls = []

    def ssh(self, handle, timeout=30):
        calls.append(1)
        if len(calls) == 1:
            raise BackendError("SSH to ssh3.vast.ai:19272 failed: ")
        return _Client()

    monkeypatch.setattr(VastBackend, "_ssh", ssh)
    probe = VastBackend().probe_box(_handle())
    assert probe["cores"] == 16 and len(calls) == 2


def test_a_rejected_key_is_not_retried_here(monkeypatch) -> None:
    monkeypatch.setattr(vast_mod, "_PROBE_RETRY_SEC", 0)
    calls = []

    def ssh(self, handle, timeout=30):
        calls.append(1)
        raise SshAuthRejected("хост відхилив наш ключ")

    monkeypatch.setattr(VastBackend, "_ssh", ssh)
    with pytest.raises(SshAuthRejected):
        VastBackend().probe_box(_handle())
    assert len(calls) == 1


def test_a_box_that_never_answers_is_still_unreachable(monkeypatch) -> None:
    monkeypatch.setattr(vast_mod, "_PROBE_RETRY_SEC", 0)

    def ssh(self, handle, timeout=30):
        e = BackendError("SSH to ssh9.vast.ai:1 failed: banner")
        e.outcome = "ssh_unreachable"
        raise e

    monkeypatch.setattr(VastBackend, "_ssh", ssh)
    with pytest.raises(BackendError) as info:
        VastBackend().probe_box(_handle())
    assert getattr(info.value, "outcome", "") == "ssh_unreachable"
