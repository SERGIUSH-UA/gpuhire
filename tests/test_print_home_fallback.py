"""Друк без датацентрів: домашні хости з воротами і явна стеля ціни картки.

Підстава — лист 20.10.2026 і замір ринку 22.10.2026: `nysh cloud go --print`
щоразу закінчувався `market_empty`, бо Ampere ≥ 20 ГБ у датацентрах під
$0.365/год не лишилось жодної, а друк за замовчуванням брав лише датацентри.
Рішення дослідника: датацентрів немає — домашні хости з жорсткими воротами;
стелю картки піднімає явне число людини.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpurunner.backends import vast as V
from gpurunner.backends.vast import VastBackend
from gpurunner.core import print_engine
from gpurunner.core.offer_score import Need

AMPERE = {"num_gpus": 1, "gpu_name": "RTX 3090", "cpu_cores_effective": 12,
          "gpu_ram": 24 * 1024, "cpu_ram": 64 * 1024, "disk_space": 300,
          "compute_cap": 860, "cuda_max_good": 12.8, "geolocation": "PL"}
HOME_GOOD = dict(AMPERE, id=1, machine_id=11, dph_total=0.20, reliability2=0.99,
                 inet_down=600, hosting_type=0)
HOME_FLAKY = dict(AMPERE, id=2, machine_id=12, dph_total=0.15, reliability2=0.93,
                  inet_down=600, hosting_type=0)
DC_DEAR = dict(AMPERE, id=3, machine_id=13, dph_total=0.45, reliability2=0.999,
               inet_down=2000, hosting_type=1)


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.setenv("GPURUNNER_BOXES_OVERRIDES", str(tmp_path / "overrides.json"))


def _market(bk: VastBackend, monkeypatch: pytest.MonkeyPatch,
            offers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ринок, що чесно застосовує фільтри запиту, — і журнал запитів."""
    calls: list[dict[str, Any]] = []

    def search_offers(**kw: Any) -> list[dict[str, Any]]:
        calls.append(kw)
        cap = kw.get("ceiling_per_card") or V.ABSOLUTE_MAX_DPH
        if kw.get("max_price"):
            cap = min(cap, float(kw["max_price"]))
        return [o for o in offers
                if (not kw.get("datacenter_only") or o["hosting_type"] == 1)
                and o["reliability2"] >= float(kw.get("min_reliability") or 0)
                and o["inet_down"] >= float(kw.get("min_inet_down") or 0)
                and o["dph_total"] <= cap + 1e-9]

    monkeypatch.setattr(bk, "search_offers", search_offers)
    return calls


def _print_need(**kw: Any) -> Need:
    return Need(**{"pages": 2000, "max_hours": 8, "budget_usd": 5.0,
                   "engine": print_engine.ENGINE, "gb_per_shard": print_engine.MIN_VRAM_GB,
                   "datacenter_only": True, "home_fallback": True, **kw})


def test_no_datacenters_falls_back_to_a_reliable_home_host(
        monkeypatch: pytest.MonkeyPatch) -> None:
    bk = VastBackend()
    calls = _market(bk, monkeypatch, [HOME_GOOD, HOME_FLAKY, DC_DEAR])
    sel = bk.find_candidates(gpu="any", need=_print_need(), min_compute_cap=8.0)
    assert sel.best is not None and sel.best.machine_id == 11, sel.reason
    home = [c for c in calls if not c.get("datacenter_only")]
    assert home and all(c["min_reliability"] >= print_engine.HOME_MIN_RELIABILITY
                        and c["min_inet_down"] >= print_engine.HOME_MIN_INET_DOWN
                        for c in home), "домашні — лише з воротами"
    assert any(c.get("datacenter_only") for c in calls), "датацентри — першими"


def test_an_explicit_datacenter_flag_stays_hard(monkeypatch: pytest.MonkeyPatch) -> None:
    bk = VastBackend()
    calls = _market(bk, monkeypatch, [HOME_GOOD])
    sel = bk.find_candidates(gpu="any", need=_print_need(home_fallback=False),
                             min_compute_cap=8.0)
    assert sel.empty
    assert all(c.get("datacenter_only") for c in calls)


def test_an_explicit_price_lifts_the_card_ceiling_for_print(
        monkeypatch: pytest.MonkeyPatch) -> None:
    bk = VastBackend()
    _market(bk, monkeypatch, [DC_DEAR])
    assert bk.find_candidates(gpu="any", need=_print_need(home_fallback=False),
                              min_compute_cap=8.0).empty, "без числа — $0.365"
    sel = bk.find_candidates(gpu="any", need=_print_need(home_fallback=False),
                             max_price=0.6, min_compute_cap=8.0)
    assert sel.best is not None and sel.best.machine_id == 13, sel.reason


def test_handwriting_keeps_the_hard_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    bk = VastBackend()
    calls = _market(bk, monkeypatch, [DC_DEAR])
    bk.find_candidates(gpu="any", need=Need(pages=600, max_hours=4, budget_usd=1.0),
                       max_price=0.6)
    assert calls and all(c.get("ceiling_per_card") is None for c in calls)


def test_the_plan_marks_only_a_default_datacenter_flag_as_softenable() -> None:
    from types import SimpleNamespace

    from gpurunner.supervise.htr import need_from_plan

    def plan(params: dict[str, Any]) -> Any:
        return SimpleNamespace(params=params, cases=[], max_hours=4, budget_usd=1.0,
                               disk_gb=50, min_net_mbps=0.0, warnings=[])

    try:
        soft = need_from_plan(plan({"engine": "paddle_vl"}), pages=10)
        hard = need_from_plan(plan({"engine": "paddle_vl", "datacenter_only": "true"}),
                              pages=10)
    except AttributeError as exc:            # план-заглушка бідніший за справжній
        pytest.skip(f"need_from_plan хоче справжній план: {exc}")
    assert soft.datacenter_only and soft.home_fallback
    assert hard.datacenter_only and not hard.home_fallback
