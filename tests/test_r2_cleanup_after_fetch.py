"""Прибирання бакета після повного забору — лише за ручкою `-p r2_cleanup=true`.

19.10.2026: наглядач після забору прибирав із бакета тільки архів догону, а
архіви кадрів `cases/` і чекпоінти лишав. Ручне `prune` після кожного заходу ніхто
не робив, і за кілька тижнів у бакеті лягло 239 ГБ уже прочитаних справ.
Дослідник погодив прибирання, але «по команді»: без ручки бакет не чіпається.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpurunner.htr import r2
from gpurunner.supervise import htr as htr_mod
from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import CasePlan, Plan

_HOST = "https://acc.r2.cloudflarestorage.com/bucket-x"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.delenv("GPURUNNER_OWNER", raising=False)
    monkeypatch.setattr(htr_mod, "_staging_root", lambda: tmp_path / "staging")


class FakeS3:
    def __init__(self, keys: list[str]) -> None:
        self.keys = list(keys)
        self.deleted: list[str] = []

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        self.deleted.append(Key)

    def get_paginator(self, _op: str):
        keys = self.keys

        class _P:
            def paginate(self, *, Bucket: str, Prefix: str):
                yield {"Contents": [{"Key": k, "Size": 1} for k in keys if k.startswith(Prefix)]}

        return _P()


BUCKET = [
    "cases/DAHMO-230-1-227/227_full.p500/p1.tar",
    "cases/DAHMO-230-1-227/227_full.p500/p2.tar",
    "cases/DAHMO-230-1-227/230-1-227.tar",          # інший прогін тієї самої шифри
    "ckpt/DAHMO-230-1-227/227_full/pysar_cyr_v28.pt/ckpt_0001.tgz",
    "ckpt/DAHMO-230-1-227/227_full/diak_cyr_v10/ckpt_0001.tgz",
    "ckpt/DAHMO-230-1-227/230-1-227/pysar_cyr_v28.pt/ckpt_0001.tgz",
]


def _sup(tmp_path: Path, params: dict[str, Any], monkeypatch) -> tuple[Supervisor, FakeS3]:
    s3 = FakeS3(BUCKET)
    monkeypatch.setattr(r2, "client", lambda *a, **k: s3)
    monkeypatch.setattr(r2, "bucket_name", lambda explicit="": "bucket-x")
    case = CasePlan(
        case="227_full", n_pages=2, out_dir=str(tmp_path / "out" / "227_full"),
        pages_url=f"{_HOST}/cases/DAHMO-230-1-227/227_full.p500/p1.tar?sig",
        pages_parts=[{"name": "p1.tar", "url": f"{_HOST}/cases/DAHMO-230-1-227/227_full.p500/p1.tar?s"},
                     {"name": "p2.tar", "url": f"{_HOST}/cases/DAHMO-230-1-227/227_full.p500/p2.tar?s"}],
        ckpt_prefix="ckpt/DAHMO-230-1-227/227_full/pysar_cyr_v28.pt")
    plan = Plan(assets_url="", cases=[case], budget_usd=1.0, max_hours=4.0, params=params)
    return Supervisor(plan, session="S"), s3  # type: ignore[arg-type]


def _staging(tmp_path: Path, pages: int) -> Path:
    staging = tmp_path / "stg"
    out = staging / "227_full" / "out"
    out.mkdir(parents=True)
    for i in range(1, pages + 1):
        (out / f"{i:04d}.txt").write_text("текст", encoding="utf-8")
    return staging


def test_with_the_flag_a_complete_case_leaves_the_bucket(tmp_path: Path, monkeypatch) -> None:
    sup, s3 = _sup(tmp_path, {"r2_cleanup": "true"}, monkeypatch)
    sup._place_case(sup.plan.cases[0], sup.state.cases[0], _staging(tmp_path, 2))
    assert sup.state.cases[0].complete
    assert sorted(s3.deleted) == sorted([
        "cases/DAHMO-230-1-227/227_full.p500/p1.tar",
        "cases/DAHMO-230-1-227/227_full.p500/p2.tar",
        "ckpt/DAHMO-230-1-227/227_full/pysar_cyr_v28.pt/ckpt_0001.tgz",
        "ckpt/DAHMO-230-1-227/227_full/diak_cyr_v10/ckpt_0001.tgz",
    ]), "лише цей прогін: архів іншого прогону тієї самої шифри лишається"


def test_without_the_flag_the_bucket_is_untouched(tmp_path: Path, monkeypatch) -> None:
    sup, s3 = _sup(tmp_path, {}, monkeypatch)
    sup._place_case(sup.plan.cases[0], sup.state.cases[0], _staging(tmp_path, 2))
    assert sup.state.cases[0].complete
    assert s3.deleted == []


def test_an_incomplete_case_keeps_its_archive_for_the_catch_up(tmp_path: Path, monkeypatch) -> None:
    """Неповна справа ще піде на догін: архів і чекпоінти їй потрібні."""
    sup, s3 = _sup(tmp_path, {"r2_cleanup": "true"}, monkeypatch)
    sup._place_case(sup.plan.cases[0], sup.state.cases[0], _staging(tmp_path, 1))
    assert not sup.state.cases[0].complete
    assert s3.deleted == []
