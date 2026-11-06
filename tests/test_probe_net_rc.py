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


# ── склеєні числа паралельної проби (15.10.2026) ─────────────────────────────
def test_the_parallel_probe_sums_per_file_not_a_glued_string() -> None:
    """`-w '%{size_download}'` без `\n`: `cat` склеював 8 чисел в одне, і
    net_par_got ставав 1.8e19 (а net_par_bps — -1)."""
    import shutil
    import subprocess
    import tempfile

    code = "\n".join(line for line in _PROBE_SCRIPT.splitlines()
                     if not line.lstrip().startswith("#"))
    line = next(ln for ln in code.splitlines() if ln.strip().startswith("G=$("))
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash недоступний")
    with tempfile.TemporaryDirectory() as tmp:
        # рядок скрипта читає /tmp/_np_*; підміняємо лише теку
        script = (f"cd '{tmp}'; printf 4194304 > _np_0; printf 4194304 > _np_1; "
                  f"printf 0 > _np_2; " + line.strip().replace("/tmp/_np_", "_np_")
                  + '; echo "$G"')
        out = subprocess.run([bash, "-c", script], capture_output=True, text=True,
                             timeout=30).stdout.strip()
    assert out == "8388608", out


@pytest.mark.parametrize("fields", [
    "net_par_got=18446744073709551616\nnet_par_bps=-1\n",      # те, що було 15.10
    "net_par_got=18446744073709551616\nnet_par_bps=90000000\n",  # склеєне число, швидкість «пристойна»
    "net_par_got=33554432\nnet_par_bps=-1\n",                # від'ємна швидкість
])
def test_a_garbage_parallel_probe_measures_nothing(fields: str) -> None:
    probe = VastBackend.parse_probe("net_bps=2000000\nnet_rc=0\n" + fields)
    assert probe["net_par_mbps"] == 0.0 and probe["net_par_bps"] == 0.0
