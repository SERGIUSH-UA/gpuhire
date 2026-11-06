"""Гроші мусять бути видні З МОМЕНТУ СТВОРЕННЯ інстансу, а не після успіху.

🔴 Найпідступніший стан заходу: фаза `renting`, `оренд: 0`, `$0.00` — а на Vast
уже висить машина й тарифікується. Агент читає стан, бачить
`human_action_required: false` і спокійно чекає, поки горять гроші. Так було
кілька разів 2026-08-11.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import CasePlan, Plan


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("GPURUNNER_OWNER", raising=False)


def _sup(tmp_path: Path) -> Supervisor:
    plan = Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case="c1", pages_url="https://r2/p.tar", n_pages=100,
                        out_dir=str(tmp_path / "c1"))],
        budget_usd=1.0, max_hours=1.0,
    )
    return Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]


def test_creation_marks_the_box_as_billing(tmp_path: Path) -> None:
    sup = _sup(tmp_path)
    sup._note_billing_started("47494016", {"dph_total": 0.5})

    assert sup.state.box["instance_id"] == "47494016"
    assert sup.state.box["billing"] is True
    assert "ТАРИФІКАЦІЯ" in sup.state.why


def test_money_accrues_before_the_rental_succeeds(tmp_path: Path) -> None:
    """Саме те, чого бракувало: $0.00 при живому оплачуваному інстансі."""
    sup = _sup(tmp_path)
    sup._note_billing_started("1", {"dph_total": 3600.0})  # $1/с — щоб видно було
    time.sleep(0.05)
    sup._accrue()
    assert sup.spent_usd > 0


def test_boot_time_is_not_lost_when_the_rental_succeeds(tmp_path: Path) -> None:
    """Хвилини, за які хост тягнув образ, теж оплачені — вони не мають зникати."""
    sup = _sup(tmp_path)
    sup._note_billing_started("1", {"dph_total": 3600.0})
    time.sleep(0.05)
    booted = sup._pending_usd()
    sup.settled_usd += sup._pending_usd()
    sup._pending = None
    sup._accrue()
    assert sup.spent_usd >= booted


def test_no_pending_means_no_phantom_money(tmp_path: Path) -> None:
    sup = _sup(tmp_path)
    assert sup._pending_usd() == 0.0
