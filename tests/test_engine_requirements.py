"""Середовище рушіїв на боксі — з маніфесту Нишпорки, за версіями.

Доти бокс ставив власний перелік наглядача (`kraken==7.0.2`, PARSeq без коміту)
і пропускав пакет, якщо той просто імпортувався. Тобто хмара читала іншим
середовищем, ніж машина людини, а теплий бокс зі старим kraken так на ньому й
лишався. Тепер вимоги їдуть в архіві ассетів поруч із раннером, план везе їх на
бокс, а torch ставиться колесом з індексу під карту поверх закешованого образу.

Тут — чотири обіцянки:
- план читає вимоги з архіву, а архів без них лишає старий шлях;
- образ лишається закешованим, torch — з індексу під карту (cu126 / cu128);
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


def test_a_new_engine_rides_the_cached_vast_image() -> None:
    """🔴 Образ під нове середовище — базовий образ Vast із torch 2.14 cu126
    (закешований на хостах: SSH за 41–55 с, kraken за 7 с). Свіжий
    pytorch/pytorch:2.14.0-cuda12.6 хост 04.10.2026 не стягнув за 360 с, а
    torch колесом на старому образі йшов 790 с."""
    new = {"engine_requirements": "\n".join(REQS)}
    assert vast.image_for(new) == vast.ENGINE_IMAGE
    assert vast.ENGINE_IMAGE.startswith("vastai/pytorch:cuda-12.6")
    assert vast.min_cuda_for(new) == pytest.approx(12.6)
    # план без вимог — старий образ і старий драйвер: новий gpuhire не ламає старий застосунок
    assert vast.image_for({}) == vast.DEFAULT_IMAGE
    assert vast.min_cuda_for({}) == pytest.approx(12.1)
    # явний образ і явна CUDA — сильніші за будь-яке правило
    assert vast.image_for({**new, "image": "x/y:1"}) == "x/y:1"
    assert vast.min_cuda_for({**new, "min_cuda": "13.0"}) == pytest.approx(13.0)


def test_the_job_runs_in_the_python_that_has_torch() -> None:
    """Образ Vast тримає torch у /venv/main, а `python3` там системний, без
    torch; job мусить іти тим інтерпретатором, де середовище є."""
    src = Path(vast.__file__).read_text(encoding="utf-8")
    assert 'PY=/venv/main/bin/python' in src and '[ -x "$PY" ] || PY=python3' in src
    assert '"$PY" -u job.py' in src


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
        returncode = 0

    monkeypatch.setattr(__import__("subprocess"), "run", lambda *a, **k: _Out())
    # коментар у файлі вимог — не вимога: `uv pip install "# …"` падає rc=2
    got = ns["_ensure_deps"]({"engine_requirements": "# рушії\n" + "\n".join(REQS)})
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
        returncode = 0

    monkeypatch.setattr(__import__("subprocess"), "run", lambda *a, **k: _Out())
    with pytest.raises(RuntimeError, match=r"7\.0\.2"):
        ns["_ensure_deps"]({"engine_requirements": "\n".join(REQS)})


TORCH_PINS = ["torch==2.14.0", "torchvision==0.29.1"]
LANDED = {"kraken": "7.1.1", "torch": "2.14.0+cu126", "torchvision": "0.29.1+cu126"}


def _box(monkeypatch, *, cap: str, have: dict, probe_rc: int = 0, runs: bool = True):
    """Раннер на фейковому боксі: `have` — версії в образі до установки,
    після установки — `LANDED`; `probe_rc` — код проби імпорту; `runs` — чи
    torch образу рахує на карті."""
    ns = _runner_ns()
    calls: list[list[str]] = []
    monkeypatch.setattr(__import__("shutil"), "which", lambda name, **k: "/usr/bin/uv")
    monkeypatch.setattr(__import__("subprocess"), "call",
                        lambda cmd, **k: calls.append([str(c) for c in cmd]) or 0)

    class _Out:
        def __init__(self, text: str, rc: int = 0) -> None:
            self.stdout, self.returncode = text, rc
            self.stderr = "Traceback …\nRuntimeError: operator torchvision::nms does not exist"

    def fake_run(cmd, **k):
        if cmd[0] == "nvidia-smi":
            return _Out(cap + "\n")
        if "torch.ones" in cmd[-1]:
            return _Out("8.0\n" if runs else "", 0 if runs else 1)
        return _Out("", probe_rc)

    monkeypatch.setattr(__import__("subprocess"), "run", fake_run)
    monkeypatch.setitem(ns, "_dist_version",
                        lambda d, fresh=False: (LANDED if fresh else have).get(d, ""))
    return ns, calls


def _uv(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c[1:3] == ["pip", "install"]]


@pytest.mark.skipif(sys.platform == "win32", reason="бокс-раннер живе на Linux-шляхах")
@pytest.mark.parametrize("cap, tag", [("7.0", "cu126"), ("8.6", "cu126"),
                                      ("8.9", "cu126"), ("9.0", "cu128"), ("12.0", "cu128")])
def test_torch_comes_from_the_index_of_the_card(monkeypatch, cap, tag) -> None:
    """🔴 Колесо torch з PyPI на Linux — cu130, а в ньому немає sm_70: V100
    отримала б torch, що не бачить її ядер. Тож torch — окремим кроком з
    індексу під карту (той самий поділ, що в маніфесті Нишпорки), до решти."""
    ns, calls = _box(monkeypatch, cap=cap, have={"torch": "2.4.1", "torchvision": "0.19.1"})
    ns["_ensure_deps"]({"engine_requirements": "\n".join([*REQS, *TORCH_PINS])})
    uv = _uv(calls)
    assert len(uv) == 2, uv
    assert "torch==2.14.0" in uv[0] and uv[0][-1].endswith("/" + tag), uv[0]
    assert "kraken==7.1.1" in uv[1], "решта ставиться другим кроком"


@pytest.mark.skipif(sys.platform == "win32", reason="бокс-раннер живе на Linux-шляхах")
def test_torchvision_follows_the_torch_already_in_the_image(monkeypatch) -> None:
    """🔴 Образ Vast уже має torch 2.14.0+cu126, а torchvision іншої версії.
    Колесо torchvision з PyPI зібране під іншу CUDA — імпорт падав на
    «operator torchvision::nms does not exist» (V100, 04.10.2026). Тож
    torchvision іде з індексу ЗБІРКИ torch, а не карти: на 9.0 теж cu126."""
    ns, calls = _box(monkeypatch, cap="9.0",
                     have={"torch": "2.14.0+cu126", "torchvision": "0.30.0+cu126"})
    ns["_ensure_deps"]({"engine_requirements": "\n".join([*REQS, *TORCH_PINS])})
    uv = _uv(calls)
    assert len(uv) == 2, uv
    assert "torchvision==0.29.1" in uv[0] and uv[0][-1].endswith("/cu126"), uv[0]


@pytest.mark.skipif(sys.platform == "win32", reason="бокс-раннер живе на Linux-шляхах")
def test_matching_torch_is_not_touched(monkeypatch) -> None:
    ns, calls = _box(monkeypatch, cap="8.6",
                     have={"torch": "2.14.0+cu126", "torchvision": "0.29.1+cu126"})
    ns["_ensure_deps"]({"engine_requirements": "\n".join([*REQS, *TORCH_PINS])})
    uv = _uv(calls)
    assert len(uv) == 1, "torch і torchvision уже ті — лише загальний крок"
    assert not any("--reinstall-package" in c for c in uv)


@pytest.mark.skipif(sys.platform == "win32", reason="бокс-раннер живе на Linux-шляхах")
def test_torch_without_kernels_for_the_card_is_rebuilt(monkeypatch) -> None:
    """🔴 Образ cu126 на Blackwell (sm_120): версія та сама, `is_available()`
    True, а ядер під карту немає. Перезбирається та сама версія з індексу
    карти — і з `--reinstall-package`, бо для uv `2.14.0+cu126` уже задовольняє
    `torch==2.14.0`."""
    ns, calls = _box(monkeypatch, cap="12.0", runs=False,
                     have={"torch": "2.14.0+cu126", "torchvision": "0.29.1+cu126"})
    ns["_ensure_deps"]({"engine_requirements": "\n".join([*REQS, *TORCH_PINS])})
    uv = _uv(calls)
    assert len(uv) == 2, uv
    assert uv[0][-1].endswith("/cu128") and "--reinstall-package" in uv[0], uv[0]


@pytest.mark.skipif(sys.platform == "win32", reason="бокс-раннер живе на Linux-шляхах")
def test_box_refuses_when_the_stack_does_not_import(monkeypatch) -> None:
    """Версії збігаються, а torchvision під іншу збірку torch не імпортується —
    відмова тут, до першої сторінки, з останнім рядком помилки."""
    ns, _ = _box(monkeypatch, cap="8.6", probe_rc=1,
                 have={"torch": "2.14.0+cu126", "torchvision": "0.29.1+cu126"})
    with pytest.raises(RuntimeError, match="torchvision::nms"):
        ns["_ensure_deps"]({"engine_requirements": "\n".join([*REQS, *TORCH_PINS])})


def test_pin_is_read_from_the_requirements() -> None:
    ns = _runner_ns()
    assert ns["_engine_pin"](REQS, "kraken") == "7.1.1"
    assert ns["_engine_pin"](["timm>=1.0"], "kraken") == ""


@pytest.mark.skipif(sys.platform == "win32", reason="бокс-раннер живе на Linux-шляхах")
def test_slow_wheel_hosts_get_a_longer_uv_timeout(monkeypatch, tmp_path) -> None:
    """🔴 Дефолтні 30 с `uv` обірвали колесо `nvidia-nvjitlink` з
    download.pytorch.org на RTX 3090 (04.10.2026): torch не став, а наступний
    крок притяг з PyPI cu130, якому замалий драйвер. Явне значення сильніше."""
    ns = _runner_ns()
    seen: list[dict] = []
    monkeypatch.setattr(__import__("subprocess"), "call",
                        lambda cmd, **k: seen.append(k.get("env") or {}) or 0)
    monkeypatch.delenv("UV_HTTP_TIMEOUT", raising=False)
    ns["_uv_install"]("/usr/bin/uv", ["x"], str(tmp_path / "uv.log"), "проба")
    assert int(seen[-1]["UV_HTTP_TIMEOUT"]) >= 300
    monkeypatch.setenv("UV_HTTP_TIMEOUT", "60")
    ns["_uv_install"]("/usr/bin/uv", ["x"], str(tmp_path / "uv.log"), "проба")
    assert seen[-1]["UV_HTTP_TIMEOUT"] == "60"
