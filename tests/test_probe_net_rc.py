"""Проба каналу пише справжню швидкість і тоді, коли curl упирається в `--max-time`.

curl друкує `-w` і на rc=28, а розбір бере ОСТАННІЙ ключ. Старий хвіст
`|| echo "net_bps=0"` перетирав виміряне нулем, і ворота казали «канал 0.0
Мбіт/с» про канал, що давав ~15 (11.09.2026, машини 31846 і 138981).
"""
from __future__ import annotations

import pytest

from gpurunner.backends.vast import _PROBE_SCRIPT, VastBackend


def test_probe_script_does_not_overwrite_the_measured_speed_with_zero() -> None:
    code = "\n".join(line for line in _PROBE_SCRIPT.splitlines()
                     if not line.lstrip().startswith("#"))
    assert '|| echo "net_bps=0"' not in code
    assert "net_rc=" in code


def test_a_timed_out_download_keeps_its_speed() -> None:
    probe = VastBackend.parse_probe("net_bps=1800000.000\nnet_http=206\nnet_rc=28\n")
    assert probe["net_mbps"] == pytest.approx(14.4)
    assert probe["net_http"] == "206"


def test_no_url_still_means_not_measured() -> None:
    probe = VastBackend.parse_probe("cores=8\nnet_bps=\n")
    assert probe["net_bps"] == ""


# ── паралельна проба рахує взяті байти (04.10.2026) ──────────────────────────
def test_the_parallel_probe_counts_bytes_actually_taken() -> None:
    """Обірваний діапазон ішов у суму як повні 4 МБ — паралель виходила
    «швидкою» саме на хворому маршруті."""
    code = "\n".join(line for line in _PROBE_SCRIPT.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "size_download" in code and "net_par_got=" in code
    assert "33554432*1000000000/D" not in code, "замовлений обсяг замість взятого"


def test_a_parallel_probe_that_took_too_little_measured_nothing() -> None:
    probe = VastBackend.parse_probe(
        "net_bps=2000000\nnet_rc=0\nnet_par_got=4194304\nnet_par_bps=90000000\n")
    assert probe["net_par_mbps"] == 0.0 and probe["net_par_bps"] == 0.0
    whole = VastBackend.parse_probe(
        "net_bps=2000000\nnet_rc=0\nnet_par_got=33554432\nnet_par_bps=6000000\n")
    assert whole["net_par_mbps"] == pytest.approx(48.0)


def test_an_old_probe_without_the_count_is_read_as_before() -> None:
    probe = VastBackend.parse_probe("net_bps=2000000\nnet_rc=0\nnet_par_bps=6000000\n")
    assert probe["net_par_mbps"] == pytest.approx(48.0)
