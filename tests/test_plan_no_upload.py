"""План для кошторису (`htr plan --no-upload`): нічого не залито — і оренди на ньому немає.

🔴 Навіщо. Кошторис рахується з числа сторінок, байтів, геометрії кадрів і
щільності; посилань йому не треба. Доти сухий прогін партії ДАЖО 1-78 (три
справи, 2 ГБ) заливав кадри й ассети в бакет і лише потім називав ціну — 95 с
аплінка перед числом, яке людина просила (05.10.2026).
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gpurunner.htr.plan_build import BuildOptions, build_plan
from gpurunner.supervise.plan import load_plan


class NoTransferS3:
    """Сховище, яке падає на будь-якій передачі: підписувати посилання можна."""

    def upload_file(self, *a: Any, **kw: Any) -> None:
        pytest.fail("план для кошторису нічого не заливає")

    def head_object(self, *a: Any, **kw: Any) -> None:
        pytest.fail("план для кошторису не питає бакет")

    def generate_presigned_url(self, op: str, *, Params: dict, ExpiresIn: int) -> str:
        return f"https://r2/{Params['Key']}?op={op}"


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    d = tmp_path / "frames" / "spr-965"
    d.mkdir(parents=True)
    for i in range(3):
        (d / f"{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xe0jpeg")
    return d


@pytest.fixture
def assets(tmp_path: Path) -> Path:
    src = tmp_path / "htr_case_run.py"
    src.write_text("print('runner')\n", encoding="utf-8")
    out = tmp_path / "assets_x.tgz"
    with tarfile.open(out, "w:gz") as tf:
        tf.add(src, arcname="htr_case_run.py")
    return out


def _build(case_dir: Path, assets: Path, tmp_path: Path, monkeypatch, *,
           transport: str = "r2", seed: str = "") -> dict[str, Any]:
    from gpurunner.htr import plan_build, r2

    s3 = NoTransferS3()
    monkeypatch.setattr(r2, "client", lambda env=None, **kw: s3)
    monkeypatch.setattr(plan_build.r2, "client", lambda env=None, **kw: s3)
    monkeypatch.setattr(r2, "put", lambda *a, **kw: pytest.fail("заливка в плані для кошторису"))
    opts = BuildOptions(out_root=(tmp_path / "out").resolve(), model="pysar.pt",
                        transport=transport, no_upload=True,
                        staging_dir=tmp_path / "_box",
                        seed_seg=[seed] if seed else [])
    out = tmp_path / "plan.json"
    build_plan([case_dir], opts, assets=assets, out_path=out, log=lambda _l: None)
    return json.loads(out.read_text(encoding="utf-8"))


def test_nothing_is_uploaded_and_the_plan_says_so(case_dir, assets, tmp_path,
                                                  monkeypatch) -> None:
    seg = tmp_path / "seg"
    seg.mkdir()
    (seg / "0000.seg.json.gz").write_bytes(b"x")
    plan = _build(case_dir, assets, tmp_path, monkeypatch, seed=str(seg))
    assert plan["staged"] is False
    case = plan["cases"][0]
    # Усе, з чого рахується кошторис, — на місці.
    assert case["n_pages"] == 3 and case["pages_bytes"] > 0
    assert case["pages_url"].startswith("https://")
    assert load_plan(tmp_path / "plan.json").staged is False


def test_a_box_plan_for_the_estimate_packs_nothing(case_dir, assets, tmp_path,
                                                   monkeypatch) -> None:
    """Склад на машині: тар пакувався б удома ще до кошторису — теж ні."""
    plan = _build(case_dir, assets, tmp_path, monkeypatch, transport="box")
    assert plan["staged"] is False
    assert not (tmp_path / "_box").exists() or not any((tmp_path / "_box").rglob("*.tar"))
    assert load_plan(tmp_path / "plan.json").staged is False


def test_ordinary_plans_stay_staged(tmp_path: Path) -> None:
    raw = {"assets_url": "https://r2/a.tgz", "budget_usd": 1, "max_hours": 1,
           "cases": [{"case": "spr-1", "pages_url": "https://r2/c.tar", "n_pages": 3}]}
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    plan = load_plan(p)
    assert plan.staged is True
    assert not plan.warnings, "ключ `staged` відомий — без попереджень"


def _unstaged(tmp_path: Path) -> Path:
    raw = {"assets_url": "https://r2/a.tgz", "budget_usd": 1, "max_hours": 1,
           "staged": False,
           "cases": [{"case": "spr-1", "pages_url": "https://r2/c.tar", "n_pages": 3}]}
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    return p


def test_supervise_refuses_to_rent_on_an_estimate_plan(tmp_path: Path, monkeypatch) -> None:
    """🔴 Наглядач і відчеплення ПІДМІНЕНІ. Без цього тест, у якого зламався
    сторож, орендує справжню машину: 05.10.2026 мутант цього тесту взяв два
    V100 на Vast із планом-заглушкою, і гасити їх довелось руками."""
    from gpurunner.cli import app
    from gpurunner.supervise import detach as detach_mod
    from gpurunner.supervise import htr as sup_mod

    class NoRent:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pytest.fail("наглядач на плані для кошторису — це оренда")

    monkeypatch.setattr(sup_mod, "Supervisor", NoRent)
    monkeypatch.setattr(detach_mod, "spawn",
                        lambda *a, **kw: pytest.fail("відчеплення на плані для кошторису"))
    monkeypatch.setattr(detach_mod, "supported", lambda: True)
    p = _unstaged(tmp_path)
    for extra in ([], ["--detach"]):
        res = CliRunner().invoke(app, ["htr", "supervise", "--plan", str(p), *extra])
        assert res.exit_code == 2, res.output
        assert "кошторису" in res.output


def test_preflight_names_the_estimate_plan(tmp_path: Path) -> None:
    from gpurunner.htr.preflight import check_plan

    rep = check_plan(_unstaged(tmp_path))
    assert rep["problems"] and "кошторису" in rep["problems"][0]
