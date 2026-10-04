"""Середовище рушіїв на боксі — з маніфесту Нишпорки, за версіями.

Доти бокс ставив власний перелік наглядача (`kraken==7.0.2`, PARSeq без коміту)
і пропускав пакет, якщо той просто імпортувався. Тобто хмара читала іншим
середовищем, ніж машина людини, а теплий бокс зі старим kraken так на ньому й
лишався. Тепер вимоги їдуть в архіві ассетів поруч із раннером, план везе їх на
бокс, а образ і драйвер хоста обираються під них.

Тут — чотири обіцянки:
- план читає вимоги з архіву, а архів без них лишає старий шлях;
- образ і мінімальна CUDA хоста залежать від того, що ставитиме бокс;
- job пропускає вимоги до раннера (`validate_params` викидає невідоме);
- раннер ставить їх одним `uv pip` і звіряє версію kraken ПІСЛЯ установки.
"""
from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest

from gpurunner.backends import vast
from gpurunner.htr import plan_build as PB

EMB = Path(__file__).resolve().parents[1] / "src" / "gpurunner" / "_embedded"
REQS = ["kraken==7.1.1", "timm>=1.0",
        "strhub @ git+https://github.com/baudm/parseq.git@1902db043c029a7e03a3818c616c06600af574be"]


def _assets(tmp_path: Path, reqs: list[str] | None) -> Path:
    out = tmp_path / "assets.tgz"
    with tarfile.open(out, "w:gz") as tf:
        body = b"# runner\n"
        info = tarfile.TarInfo("scripts/htr_case_run.py")
        info.size = len(body)
        tf.addfile(info, io.BytesIO(body))
        if reqs is not None:
            text = ("# середовище рушіїв\n" + "\n".join(reqs) + "\n").encode("utf-8")
            info = tarfile.TarInfo(PB.ENGINE_REQUIREMENTS)
            info.size = len(text)
            tf.addfile(info, io.BytesIO(text))
    return out


def test_plan_reads_requirements_from_assets(tmp_path: Path) -> None:
    assert PB.engine_requirements_in(_assets(tmp_path, REQS)) == REQS


def test_assets_of_an_old_app_keep_the_old_path(tmp_path: Path) -> None:
    """Архів старого nyshporka вимог не має — і план їх не вигадує."""
    assert PB.engine_requirements_in(_assets(tmp_path, None)) == []


def test_image_and_host_cuda_follow_the_engine() -> None:
    """🔴 Образ cu12.6 на драйвері 12.1 не підніметься, а оренда вже йде."""
    new = {"engine_requirements": "\n".join(REQS)}
    assert vast.image_for(new) == vast.ENGINE_IMAGE
    assert vast.min_cuda_for(new) == pytest.approx(12.6)
    # план без вимог — старий образ і старий драйвер: новий gpuhire не ламає старий застосунок
    assert vast.image_for({}) == vast.DEFAULT_IMAGE
    assert vast.min_cuda_for({}) == pytest.approx(12.1)
    # явний образ і явна CUDA — сильніші за будь-яке правило
    assert vast.image_for({**new, "image": "x/y:1"}) == "x/y:1"
    assert vast.min_cuda_for({**new, "min_cuda": "13.0"}) == pytest.approx(13.0)


def test_engine_image_keeps_the_torch_of_the_manifest() -> None:
    """Образ із тим самим torch, що ставить `nysh htr install` під kraken 7.1.1
    (2.14.0), і cu126 — у колесах cu128 немає sm_70 (V100)."""
    assert "2.14.0" in vast.ENGINE_IMAGE and "cuda12.6" in vast.ENGINE_IMAGE


def test_job_passes_requirements_through_to_the_box() -> None:
    from gpurunner.jobs.htr_case import HTRCaseJob

    got = HTRCaseJob().validate_params({"engine_requirements": "\n".join(REQS)})
    assert got["engine_requirements"].splitlines() == REQS
    assert HTRCaseJob().validate_params({})["engine_requirements"] == ""


# ── раннер боксу ─────────────────────────────────────────────────────────────
def _runner_ns() -> dict:
    src = (EMB / "_common.py").read_text(encoding="utf-8") + "\n" + (
        EMB / "htr_case_runner.py").read_text(encoding="utf-8").replace(
        'if __name__ == "__main__":\n    main({})', "")
    ns: dict = {"__name__": "runner_under_test"}
    exec(compile(src, "htr_case_runner.py", "exec"), ns)
    return ns


@pytest.mark.skipif(sys.platform == "win32", reason="бокс-раннер живе на Linux-шляхах")
def test_box_installs_the_manifest_by_version_not_by_import(monkeypatch) -> None:
    """🔴 Теплий бокс зі старим kraken імпортується так само — тож рішення
    «імпорт пройшов, пропускаю» лишило б його на неперевіреній версії."""
    ns = _runner_ns()
    calls: list[list[str]] = []
    monkeypatch.setattr(__import__("shutil"), "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(__import__("subprocess"), "call",
                        lambda cmd, **k: calls.append([str(c) for c in cmd]) or 0)

    class _Out:
        stdout = "7.1.1\n"

    monkeypatch.setattr(__import__("subprocess"), "run", lambda *a, **k: _Out())
    got = ns["_ensure_deps"]({"engine_requirements": "\n".join(REQS)})
    assert got == REQS
    uv = [c for c in calls if c[:3] == ["/usr/bin/uv", "pip", "install"]]
    assert len(uv) == 1, "вимоги мусять іти ОДНИМ резолвом, не по пакету"
    for spec in REQS:
        assert spec in uv[0]


@pytest.mark.skipif(sys.platform == "win32", reason="бокс-раннер живе на Linux-шляхах")
def test_box_refuses_when_kraken_did_not_land(monkeypatch) -> None:
    """Приймач — версія kraken після установки, а не код повернення `uv`."""
    ns = _runner_ns()
    monkeypatch.setattr(__import__("shutil"), "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(__import__("subprocess"), "call", lambda cmd, **k: 0)

    class _Out:
        stdout = "7.0.2\n"

    monkeypatch.setattr(__import__("subprocess"), "run", lambda *a, **k: _Out())
    with pytest.raises(RuntimeError, match=r"7\.0\.2"):
        ns["_ensure_deps"]({"engine_requirements": "\n".join(REQS)})


def test_pin_is_read_from_the_requirements() -> None:
    ns = _runner_ns()
    assert ns["_engine_pin"](REQS, "kraken") == "7.1.1"
    assert ns["_engine_pin"](["timm>=1.0"], "kraken") == ""
