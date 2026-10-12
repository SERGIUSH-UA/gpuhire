"""Підлога темпу мусить говорити вголос — і тоді, коли вибір є.

Мовчазна ручка невідрізнима від зламаної: людина бачить «наглядач нічого не
обирає» і не знає, це ринок просів чи її власний поріг.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpurunner.core.offer_score import Need, select_offers
from gpurunner.supervise.htr import Supervisor


class ЗаписникСтану:
    def __init__(self) -> None:
        self.notes: list[tuple[str, str, str]] = []

    def note(self, kind: str, detail: str, action: str = "") -> None:
        self.notes.append((kind, detail, action))


def наглядач() -> tuple[Any, ЗаписникСтану]:
    sup = Supervisor.__new__(Supervisor)
    state = ЗаписникСтану()
    sup.state = state
    sup._say = lambda рядок: None
    return sup, state


def відсіяні(n: int) -> list[Any]:
    return [type("R", (), {"rejects": ("218 стор/год < підлоги 1000",),
                           "sizing": type("S", (), {"pages_per_hour": 218})()})()
            for _ in range(n)]


def вибір(cut: int, left: int) -> Any:
    return type("Sel", (), {"rejected": відсіяні(cut),
                            "candidates": list(range(left))})()


def test_floor_says_how_much_it_cut() -> None:
    sup, state = наглядач()
    sup._warn_pph_floor(вибір(cut=24, left=1), Need(pages=1, max_hours=1,
                                                    budget_usd=1,
                                                    min_pages_per_hour=1000))
    (kind, detail, action), = state.notes
    assert kind == "pph_floor"
    assert "відсікла 24 машини" in detail and "лишилось 1" in detail
    assert "market_empty" in action


def test_roomy_market_is_reported_as_roomy() -> None:
    sup, state = наглядач()
    sup._warn_pph_floor(вибір(cut=9, left=72), Need(pages=1, max_hours=1,
                                                    budget_usd=1,
                                                    min_pages_per_hour=1000))
    assert state.notes[0][2] == "запас є"


def test_silent_when_no_floor_is_set() -> None:
    """Без підлоги ручки немає — і попередження теж."""
    sup, state = наглядач()
    sup._warn_pph_floor(вибір(cut=24, left=1), Need(pages=1, max_hours=1,
                                                    budget_usd=1))
    assert not state.notes


@pytest.mark.parametrize("cut,слово", [(1, "машину"), (2, "машини"), (5, "машин"),
                                       (11, "машин"), (22, "машини")])
def test_counts_are_declined(cut: int, слово: str) -> None:
    sup, state = наглядач()
    sup._warn_pph_floor(вибір(cut=cut, left=9), Need(pages=1, max_hours=1,
                                                     budget_usd=1,
                                                     min_pages_per_hour=1000))
    assert f"{cut} {слово}" in state.notes[0][1]


def test_empty_market_names_the_reachable_rate() -> None:
    """Вирок мусить казати, ДО ЯКОГО числа знижувати підлогу.

    🔴 І не радити `--max-hours`: причина підлоги звучить «… стор/год <
    підлоги 1000», тобто містить «год», і класифікатор відносив її до стелі
    годин — порада була про іншу ручку.
    """
    sel = type("Sel", (), {
        "rejected": [type("R", (), {
            "rejects": ("1815 стор/год < підлоги 2500",),
            "sizing": type("S", (), {"pages_per_hour": 1815})()})()],
        "reason": "ринок не дав жодної машини",
    })()
    why, action = Supervisor._explain_empty_market(sel)
    assert "підлога темпу" in why
    assert "1815" in action and "--min-pph" in action
    assert "max-hours" not in action


def test_selection_carries_rejects_even_when_it_succeeds() -> None:
    """Без цього попередження нема з чого рахувати."""
    need = Need(pages=100, max_hours=8.0, budget_usd=5.0, gb_per_shard=1.9,
                lines_per_page=70, min_pages_per_hour=1000)
    швидка = {"id": 1, "machine_id": 1, "gpu_name": "RTX 3090", "num_gpus": 1,
              "cpu_cores_effective": 32, "gpu_ram": 24 * 1024, "dph_total": 0.2,
              "reliability2": 0.99, "disk_space": 200}
    повільна = dict(швидка, id=2, machine_id=2, cpu_cores_effective=2,
                    gpu_ram=8 * 1024)
    sel = select_offers([швидка, повільна], need)
    assert sel.best is not None and sel.best.machine_id == 1
    assert any(any("підлоги" in w for w in r.rejects) for r in sel.rejected)
