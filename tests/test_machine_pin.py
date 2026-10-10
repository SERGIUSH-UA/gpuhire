"""Закріплена машина (`--machine` / `machine_id` у плані).

Захід інколи мусить іти на конкретну машину: порівняти два прогони на тому
самому залізі, взяти хост із образом у кеші. Без закріплення наглядач брав
найвигіднішу за ринком — 04.10.2026 це був хост, що 7 хв не міг стягнути
образ, тоді як потрібна машина стояла вільна.

Обіцянки:
- кандидати — лише оферти закріпленої машини, ринку не питаємо;
- ціль темпу ринку на неї не діє (машину обрала людина), стелі грошей — діють;
- план і команда наглядача знають ключ.
"""
from __future__ import annotations

import json
from pathlib import Path

from gpurunner.backends.vast import VastBackend
from gpurunner.core.offer_score import Need
from gpurunner.supervise.htr import need_target_pph
from gpurunner.supervise.plan import check_plan_keys, load_plan

V100X2 = {"id": 32257938, "machine_id": 20242, "gpu_name": "Tesla V100", "num_gpus": 2,
          "cpu_cores_effective": 32, "gpu_ram": 16 * 1024, "cpu_ram": 128 * 1024,
          "disk_space": 3000, "dph_total": 0.244, "reliability2": 0.97,
          "inet_down": 900, "geolocation": "Sweden, SE", "compute_cap": 700}


def _backend(monkeypatch, tmp_path, offers):
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.setenv("GPURUNNER_BOXES_OVERRIDES", str(tmp_path / "overrides.json"))
    calls: list[dict] = []
    bk = VastBackend()

    def search_offers(**kw):
        calls.append(kw)
        ids = list(kw.get("machine_ids") or [])
        return [o for o in offers if not ids or o["machine_id"] in ids]

    monkeypatch.setattr(bk, "search_offers", search_offers)
    return bk, calls


def test_pinned_search_asks_only_for_that_machine(monkeypatch, tmp_path) -> None:
    other = dict(V100X2, id=1, machine_id=148664, dph_total=0.138)
    bk, calls = _backend(monkeypatch, tmp_path, [V100X2, other])
    need = Need(pages=606, max_hours=4, budget_usd=0.5, target_pph=50_000)
    sel = bk.find_candidates(gpu="V100", need=need, num_gpus=2, machine_ids=[20242])
    assert len(calls) == 1 and calls[0]["machine_ids"] == [20242], calls
    assert sel.best is not None and sel.best.machine_id == 20242, sel.reason
    # ціль 50 000 стор/год не дає жодна V100 — закріплену машину це не зупиняє


def test_pinned_machine_still_obeys_the_money(monkeypatch, tmp_path) -> None:
    bk, _ = _backend(monkeypatch, tmp_path, [V100X2])
    need = Need(pages=606, max_hours=4, budget_usd=0.0001)
    assert bk.find_candidates(gpu="V100", need=need, num_gpus=2,
                              machine_ids=[20242]).best is None


def test_plan_carries_the_pin_and_drops_the_market_target(tmp_path) -> None:
    raw = {"assets_url": "https://r2/a.tgz", "machine_id": 20242, "target_pph": 5000,
           "budget_usd": 0.5, "max_hours": 2,
           "cases": [{"case": "spr-1", "pages_url": "https://r2/x.tar", "n_pages": 10,
                      "out_dir": "E:/prostir/reports/htr/spr-1"}]}
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    assert not [w for w in check_plan_keys(raw) if "machine_id" in w]
    plan = load_plan(p)
    assert plan.machine_id == 20242
    assert need_target_pph(plan) == 0.0


def test_supervisor_cli_has_the_machine_flag() -> None:
    src = (Path(__file__).resolve().parents[1] / "src" / "gpurunner" / "cli.py").read_text(
        encoding="utf-8")
    assert '"--machine"' in src and "plan = replace(plan, machine_id=machine)" in src


def test_a_wrapper_pins_the_machine_through_params(tmp_path) -> None:
    """`nysh cloud go --machine` кладе машину в `params` (`-p machine_id=`)."""
    raw = {"assets_url": "https://r2/a.tgz", "budget_usd": 0.5, "max_hours": 2,
           "params": {"machine_id": "150661"},
           "cases": [{"case": "spr-1", "pages_url": "https://r2/x.tar", "n_pages": 10,
                      "out_dir": "E:/prostir/reports/htr/spr-1"}]}
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    plan = load_plan(p)
    assert plan.machine_id == 150661
    assert need_target_pph(plan) == 0.0
