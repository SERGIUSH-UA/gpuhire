"""Лише датацентри (`-p datacenter_only=true`).

Запит людей з дешевих машин (05.10.2026): «відчув смак обривів і відвалів
дешевих машин». Vast розрізняє домашній хост і датацентр полем `hosting_type`
(0 / 1); під стелею ціни датацентрів зараз 13 із 366 офферів.

Обіцянки:
- фільтр іде в ЗАПИТ до ринку й дублюється клієнтським відсівом;
- ним шукаються і зірки реєстру, і ринок; закріплена машина — ні;
- ручка читається з плану, хибне значення — помилка до оренди.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpurunner.backends.vast import VastBackend
from gpurunner.core.offer_score import Need
from gpurunner.supervise.htr import need_from_plan
from gpurunner.supervise.plan import load_plan, parse_flag

HOME = {"id": 1, "machine_id": 11, "gpu_name": "Tesla V100", "num_gpus": 1,
        "cpu_cores_effective": 20, "gpu_ram": 16 * 1024, "cpu_ram": 64 * 1024,
        "disk_space": 300, "dph_total": 0.097, "reliability2": 0.97,
        "inet_down": 900, "geolocation": "Texas, US", "compute_cap": 700,
        "hosting_type": 0}
DC = dict(HOME, id=2, machine_id=22, dph_total=0.178, hosting_type=1)


class _Recorder(VastBackend):
    def __init__(self, offers: list[dict[str, Any]]) -> None:
        super().__init__()
        self.offers = offers
        self.queries: list[dict[str, Any]] = []

    def _request(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        self.queries.append(kw.get("json") or {})
        return {"offers": [dict(o) for o in self.offers]}


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.setenv("GPURUNNER_BOXES_OVERRIDES", str(tmp_path / "overrides.json"))


def test_query_asks_the_market_for_datacenters_only() -> None:
    bk = _Recorder([HOME, DC])
    got = bk.search_offers(gpu="any", datacenter_only=True)
    assert bk.queries[-1]["hosting_type"] == {"eq": 1}
    # ринок міг проігнорувати фільтр — домашній хост однаково не проходить
    assert [o["id"] for o in got] == [2]


def test_without_the_flag_the_query_is_unchanged() -> None:
    bk = _Recorder([HOME, DC])
    got = bk.search_offers(gpu="any")
    assert "hosting_type" not in bk.queries[-1]
    assert {o["id"] for o in got} == {1, 2}


def test_market_search_passes_the_flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    bk = VastBackend()
    calls: list[dict[str, Any]] = []

    def search_offers(**kw: Any) -> list[dict[str, Any]]:
        calls.append(kw)
        return [o for o in (HOME, DC)
                if not kw.get("datacenter_only") or o["hosting_type"] == 1]

    monkeypatch.setattr(bk, "search_offers", search_offers)
    need = Need(pages=600, max_hours=4, budget_usd=1.0, datacenter_only=True)
    sel = bk.find_candidates(gpu="any", need=need)
    assert calls and all(c["datacenter_only"] for c in calls)
    assert sel.best is not None and sel.best.machine_id == 22, sel.reason


def test_pinned_machine_ignores_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Людина назвала машину сама — фільтр ринку її не відсікає."""
    bk = VastBackend()
    calls: list[dict[str, Any]] = []

    def search_offers(**kw: Any) -> list[dict[str, Any]]:
        calls.append(kw)
        return [HOME]

    monkeypatch.setattr(bk, "search_offers", search_offers)
    need = Need(pages=600, max_hours=4, budget_usd=1.0, datacenter_only=True)
    sel = bk.find_candidates(gpu="any", need=need, machine_ids=[11])
    assert not calls[0].get("datacenter_only")
    assert sel.best is not None and sel.best.machine_id == 11


def _plan(tmp_path: Path, params: dict[str, Any]) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "assets_url": "https://example.invalid/assets.tgz",
        "params": params, "budget_usd": 0.5, "max_hours": 2,
        "cases": [{"case": "c1", "pages_url": "https://example.invalid/c1.tar",
                   "n_pages": 10, "out_dir": str(tmp_path / "out")}],
    }), encoding="utf-8")
    return path


@pytest.mark.parametrize(("value", "want"), [
    ("true", True), ("1", True), ("так", True),
    ("false", False), ("0", False), (None, False),
])
def test_plan_knob_reaches_the_need(tmp_path: Path, value: Any, want: bool) -> None:
    params = {} if value is None else {"datacenter_only": value}
    plan = load_plan(_plan(tmp_path, params))
    assert need_from_plan(plan, pages=10).datacenter_only is want


def test_unclear_value_fails_before_renting(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="datacenter_only"):
        load_plan(_plan(tmp_path, {"datacenter_only": "yes,please"}))
    with pytest.raises(ValueError):
        parse_flag("maybe", "datacenter_only")
