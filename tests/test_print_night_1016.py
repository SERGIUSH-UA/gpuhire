"""Захід газети 16.10.2026: 1 606 номерів по 2 сторінки на чужому R2.

Наглядач гасив здорові бокси власними запобіжниками: стеля ціни рукопису на
друці, «0 стор. за 51 хв» на боксі, що прочитав 300, дотяг із машини сотень
непочатих справ, розкладка, якої не було після зупинки за грошима.
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path

from gpurunner.core import print_engine
from gpurunner.htr import fetch_ckpt
from gpurunner.supervise.decide import Cfg, Obs, decide
from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import CasePlan
from tests.test_supervise_flow import (
    FakeBackend,
    _isolated,  # noqa: F401  (фікстура)
    make_plan,
)

CFG = Cfg(budget_usd=3.00, max_hours=8.0)
PRINT = {"engine": print_engine.ENGINE, "script": "print", "model": "PaddleOCR-VL-1.6-0.9B"}


def _print_plan(tmp_path: Path, **kw):
    return dataclasses.replace(make_plan(tmp_path), params=dict(PRINT), **kw)


# ── стеля ціни ───────────────────────────────────────────────────────────────
def test_print_plan_without_a_knob_gets_the_print_price_ceiling(tmp_path: Path) -> None:
    sup = Supervisor(_print_plan(tmp_path), backend=FakeBackend(pages=100, texts=100),
                     tick_sec=0)  # type: ignore[arg-type]
    assert sup.cfg.max_usd_per_1000 == print_engine.USD_PER_1000


def test_explicit_knob_still_wins_on_print(tmp_path: Path) -> None:
    plan = dataclasses.replace(_print_plan(tmp_path), max_usd_per_1000_pages=1.2)
    sup = Supervisor(plan, backend=FakeBackend(pages=100, texts=100), tick_sec=0)  # type: ignore[arg-type]
    assert sup.cfg.max_usd_per_1000 == 1.2


def test_handwriting_keeps_its_own_ceiling(tmp_path: Path) -> None:
    sup = Supervisor(make_plan(tmp_path), backend=FakeBackend(pages=100, texts=100),
                     tick_sec=0)  # type: ignore[arg-type]
    assert sup.cfg.max_usd_per_1000 == Cfg(budget_usd=1, max_hours=1).max_usd_per_1000


# ── гроші проти сторінок (правило 3-тер) ─────────────────────────────────────
def _running(**kw) -> dict:
    base = {"phase": "running", "n_pages_expected": 2000, "pages_done": 600,
            "pages_per_hour": 700, "shards": [{"k": 1, "alive": True, "rc": None}]}
    base.update(kw)
    return base


def test_an_idle_warm_box_is_not_rerented_for_money_without_pages() -> None:
    """Теплий бокс чекає довіску: гроші йдуть, сторінок нема — це простій."""
    idle = Obs(progress=_running(phase="idle"), progress_age_sec=30, ssh_ok=True,
               dph=0.33, rent_age_sec=80 * 60, rent_settled_sec=60 * 60, rent_usd=0.30,
               rent_pages=0, box_best_usd_per_1000=0.5)
    assert decide(idle, CFG)[0] != "rerent"


def test_a_busy_box_that_stalls_is_still_rerented() -> None:
    hung = Obs(progress=_running(), progress_age_sec=30, ssh_ok=True, dph=0.33,
               rent_age_sec=80 * 60, rent_settled_sec=60 * 60, rent_usd=0.30,
               rent_pages=0, box_best_usd_per_1000=0.5)
    assert decide(hung, CFG)[0] == "rerent"


def test_resumed_pages_count_when_asked(tmp_path: Path) -> None:
    sup = Supervisor(make_plan(tmp_path), backend=FakeBackend(pages=100, texts=100),
                     tick_sec=0)  # type: ignore[arg-type]
    p = {"box_pages_done": 10, "box_pages_resumed": 5}
    assert sup._box_pages_total(p, count_resumed=True) == 15
    assert sup._box_pages_total(p) == 10


def test_a_box_counter_that_restarts_rebases_instead_of_reading_zero(tmp_path: Path) -> None:
    """База застигла на 300; новий процес раннера лічить з нуля — доти це
    було «0 стор. за 51 хв» і погашений здоровий бокс."""
    sup = Supervisor(make_plan(tmp_path), backend=FakeBackend(pages=100, texts=100),
                     tick_sec=0)  # type: ignore[arg-type]
    sup._current_rent_usd = lambda: 0.40  # type: ignore[method-assign]
    now = time.monotonic()
    sup._rent_base = (0.10, 300, now - 3000)
    _, pages, _ = sup._rent_money({"box_pages_done": 50}, settled=True, now=now)
    assert pages == 0 and sup._rent_base[1] == 50, "база переставлена, а не застаріла"
    _, pages, _ = sup._rent_money({"box_pages_done": 120}, settled=True, now=now + 60)
    assert pages == 70


# ── забір після зупинки ──────────────────────────────────────────────────────
class _ListingBackend(FakeBackend):
    """Рахує SSH-обміни «що в теці справи на машині»."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.listed: list[str] = []

    def case_files(self, handle, slug):
        self.listed.append(slug)
        return []

    def fetch_case_files(self, handle, slug, names, dest, get_url=None):
        return 0


def test_top_up_asks_the_box_only_about_cases_it_reached(tmp_path: Path) -> None:
    cases = [CasePlan(case=f"n{i:03d}", pages_url="https://r2/x.tar", n_pages=2,
                      out_dir=str(tmp_path / f"n{i:03d}")) for i in range(200)]
    plan = dataclasses.replace(make_plan(tmp_path), cases=cases)
    backend = _ListingBackend(pages=2, texts=0)
    sup = Supervisor(plan, backend=backend, tick_sec=0)  # type: ignore[arg-type]
    sup._handle = object()  # type: ignore[assignment]
    sup._last_progress = {"results": [{"case": c.case} for c in cases[:10]]}
    sup._top_up_from_box(tmp_path / "staging")
    assert 0 < len(backend.listed) <= 50, len(backend.listed)
    assert "n199" not in backend.listed


def test_quiesce_stops_the_print_worker_too(tmp_path: Path) -> None:
    sup = Supervisor(make_plan(tmp_path), backend=FakeBackend(pages=100, texts=100),
                     tick_sec=0)  # type: ignore[arg-type]
    sent: list[str] = []

    class _Client:
        def close(self) -> None:
            pass

    sup._handle = object()  # type: ignore[assignment]
    sup.backend._ssh = lambda h, timeout=20: _Client()  # type: ignore[attr-defined]
    sup._exec_on = lambda client, cmd, **kw: sent.append(cmd)  # type: ignore[method-assign]
    sup._quiesce_runner()
    assert any("[p]rint_vl_worker.py" in c for c in sent)


def test_incomplete_cases_are_filled_from_store_keys(tmp_path: Path, monkeypatch) -> None:
    """Прочитане на боксі лежить у чекпоінтах R2, а presigned-посилання плану
    протухли — доти справа лишалась поза диском, і її добирали руками."""
    from gpurunner.htr import r2

    cases = [CasePlan(case="n001", pages_url="https://r2/x.tar", n_pages=2,
                      out_dir=str(tmp_path / "n001"),
                      ckpt_prefix="ckpt/VYD-X-1937/n001/PaddleOCR-VL-1.6-0.9B"),
             CasePlan(case="n002", pages_url="https://r2/y.tar", n_pages=2,
                      out_dir=str(tmp_path / "n002"))]          # без чекпоінтів — не чіпати
    plan = dataclasses.replace(make_plan(tmp_path), cases=cases)
    sup = Supervisor(plan, backend=FakeBackend(pages=2, texts=0), tick_sec=0)  # type: ignore[arg-type]
    asked: list[str] = []

    def fake_fetch(rec, *, s3=None, log=None, **kw):
        asked.append(rec["case"])
        out = Path(rec["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        for i in (1, 2):
            (out / f"{i:04d}.txt").write_text("текст", encoding="utf-8")
        return {"case": rec["case"], "ckpt": 2, "ok": True}

    monkeypatch.setattr(r2, "configured", lambda: True)
    monkeypatch.setattr(r2, "client", lambda *a, **kw: object())
    monkeypatch.setattr(fetch_ckpt, "fetch_case", fake_fetch)
    sup._fill_from_store_keys()

    assert asked == ["n001"]
    assert sup.state.cases[0].complete and sup.state.cases[0].status == "done"
    assert not sup.state.cases[1].complete
    assert any(i.kind == "store_filled" for i in sup.state.incidents)


# ── тексти ───────────────────────────────────────────────────────────────────
def test_fetch_ckpt_names_the_print_branch() -> None:
    assert fetch_ckpt.branch_name("out", "PaddleOCR-VL-1.6-0.9B") == "Друк"
    assert fetch_ckpt.branch_name("out", "pysar_cyr_v21.pt") == "Писар"
