"""Прочитане не пропадає разом із машиною: чекпоінти не вичерпуються, забір добирає.

23.09.2026 книга ЦДІАК spr-68 (1 308 сторінок): план дав 16 слотів чекпоінтів,
раннер спалив їх за перші ~32 хв, і до кінця заходу чекпоінтів не було. Забір
побачив 16 архівів у сховищі (719 сторінок), вирішив, що текст удома, і не
торкнувся машини; її знищили, 589 прочитаних сторінок пропали.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gpurunner.core.models import JobHandle
from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import CasePlan, Plan


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.delenv("GPURUNNER_OWNER", raising=False)


# ---- наглядач ----------------------------------------------------------------

class _Box:
    """Машина, на диску якої лежить ВСЯ прочитана черга."""

    def __init__(self, pages: dict[str, int]) -> None:
        self.pages = pages
        self.asked: list[tuple[str, ...]] = []

    def fetch_outputs(self, handle, dest, *, only=()):
        self.asked.append(tuple(only))
        for slug in only or self.pages:
            out = Path(dest) / slug / "out"
            out.mkdir(parents=True, exist_ok=True)
            for i in range(1, self.pages[slug] + 1):
                (out / f"{i:04d}.txt").write_text("текст", encoding="utf-8")
        return []

    def cancel(self, handle):
        pass


def _two_case_plan(tmp_path: Path) -> Plan:
    return Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case="spr-68", pages_url="https://r2/68.tar", n_pages=30,
                        out_dir=str(tmp_path / "o68"), flatten_out=True),
               CasePlan(case="spr-69", pages_url="https://r2/69.tar", n_pages=6,
                        out_dir=str(tmp_path / "o69"), flatten_out=True)],
        budget_usd=1.0, max_hours=1.0,
    )


def test_store_short_is_topped_up_from_the_live_box(tmp_path: Path, monkeypatch) -> None:
    """Сховище дало 17 із 30 сторінок книги: решту бере з машини ДО її знищення."""
    box = _Box({"spr-68": 30, "spr-69": 6})
    sup = Supervisor(_two_case_plan(tmp_path), backend=box, session="S")  # type: ignore[arg-type]
    sup._handle = JobHandle(backend="vast", remote_id="1", job_name="htr_case", gpu="any")

    def store(staging: Path) -> bool:
        for slug, have in (("spr-68", 17), ("spr-69", 6)):
            out = staging / slug / "out"
            out.mkdir(parents=True, exist_ok=True)
            for i in range(1, have + 1):
                (out / f"{i:04d}.txt").write_text("текст", encoding="utf-8")
        return True

    monkeypatch.setattr(sup, "_fetch_from_store", store)
    monkeypatch.setattr(sup, "_fetch_service_bundles", lambda staging: [])
    sup._fetch_queue()

    assert box.asked == [("spr-68",)], f"з машини треба лише неповне, а просили {box.asked}"
    assert all(c.complete for c in sup.state.cases), [c.detail for c in sup.state.cases]
    assert len(list((tmp_path / "o68").glob("*.txt"))) == 30
    assert any(i.kind == "store_short" for i in sup.state.incidents)


def test_complete_store_does_not_touch_the_box(tmp_path: Path, monkeypatch) -> None:
    box = _Box({"spr-68": 30, "spr-69": 6})
    sup = Supervisor(_two_case_plan(tmp_path), backend=box, session="S")  # type: ignore[arg-type]
    sup._handle = JobHandle(backend="vast", remote_id="1", job_name="htr_case", gpu="any")

    def store(staging: Path) -> bool:
        for slug, have in (("spr-68", 30), ("spr-69", 6)):
            out = staging / slug / "out"
            out.mkdir(parents=True, exist_ok=True)
            for i in range(1, have + 1):
                (out / f"{i:04d}.txt").write_text("текст", encoding="utf-8")
        return True

    monkeypatch.setattr(sup, "_fetch_from_store", store)
    monkeypatch.setattr(sup, "_fetch_service_bundles", lambda staging: [])
    sup._fetch_queue()

    assert box.asked == []
    assert all(c.complete for c in sup.state.cases)


class _DeltaBox(_Box):
    """Машина, що вміє віддати дельту: список файлів і архів названих."""

    def __init__(self, pages: dict[str, int]) -> None:
        super().__init__(pages)
        self.fetched: dict[str, list[str]] = {}

    def case_files(self, handle, slug):
        return {f"out/{i:04d}.txt": len("текст".encode()) for i in range(1, self.pages[slug] + 1)}

    def fetch_case_files(self, handle, slug, rels, dest):
        self.fetched[slug] = sorted(rels)
        for rel in rels:
            target = Path(dest) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("текст", encoding="utf-8")
        return len(rels)


def test_the_top_up_takes_only_what_the_store_did_not_bring(tmp_path: Path,
                                                            monkeypatch) -> None:
    """Дельтою: зі 30 сторінок книги сховище дало 17 — з машини їдуть рівно 13,
    а не вся тека справи пофайлово."""
    box = _DeltaBox({"spr-68": 30, "spr-69": 6})
    sup = Supervisor(_two_case_plan(tmp_path), backend=box, session="S")  # type: ignore[arg-type]
    sup._handle = JobHandle(backend="vast", remote_id="1", job_name="htr_case", gpu="any")

    def store(staging: Path) -> bool:
        for slug, have in (("spr-68", 17), ("spr-69", 6)):
            out = staging / slug / "out"
            out.mkdir(parents=True, exist_ok=True)
            for i in range(1, have + 1):
                (out / f"{i:04d}.txt").write_text("текст", encoding="utf-8")
        return True

    monkeypatch.setattr(sup, "_fetch_from_store", store)
    monkeypatch.setattr(sup, "_fetch_service_bundles", lambda staging: [])
    sup._fetch_queue()

    assert box.asked == [], "пофайловий забір не потрібен"
    assert box.fetched == {"spr-68": [f"out/{i:04d}.txt" for i in range(18, 31)]}
    assert all(c.complete for c in sup.state.cases), [c.detail for c in sup.state.cases]


def test_the_supervisor_asks_for_a_last_checkpoint_before_the_store_fetch(
        tmp_path: Path, monkeypatch) -> None:
    """Прочитане йде в сховище HTTP-заливкою раннера, а не пофайловим SFTP:
    наглядач просить останній чекпоінт ДО того, як забирати зі сховища."""
    calls: list[str] = []

    class _FlushBox(_DeltaBox):
        def request_checkpoint(self, handle, *, wait_sec=120):
            calls.append("flush")
            return "done"

    box = _FlushBox({"spr-68": 30, "spr-69": 6})
    sup = Supervisor(_two_case_plan(tmp_path), backend=box, session="S")  # type: ignore[arg-type]
    sup._handle = JobHandle(backend="vast", remote_id="1", job_name="htr_case", gpu="any")

    def store(staging: Path) -> bool:
        calls.append("store")
        for slug, have in (("spr-68", 30), ("spr-69", 6)):
            out = staging / slug / "out"
            out.mkdir(parents=True, exist_ok=True)
            for i in range(1, have + 1):
                (out / f"{i:04d}.txt").write_text("текст", encoding="utf-8")
        return True

    monkeypatch.setattr(sup, "_fetch_from_store", store)
    monkeypatch.setattr(sup, "_fetch_service_bundles", lambda staging: [])
    sup._fetch_queue()
    assert calls == ["flush", "store"]
    assert box.fetched == {}, "після чекпоінта з машини брати нічого"

