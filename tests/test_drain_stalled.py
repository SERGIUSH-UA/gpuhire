"""Усе прочитано, а бокс не каже `done`: заливка в R2 не йде (15.10.2026).

Сесія `…-15a9`: 288/288 сторінок о 06:20, бокс 9 Мбіт/с, `curl` PUT у R2
отримував 0 байт. Раннер публікує `done` лише після фінальних заливок
(`--retry 2 --max-time 120+`), тож наглядач чекав у `running` з
«прогрес свіжий» і не забирав, поки людина не викликала `wrap-up`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gpurunner.supervise.decide import Cfg, Obs, decide
from gpurunner.supervise.htr import Supervisor
from tests.test_supervise_flow import (  # noqa: F401  (фікстура _isolated)
    FakeBackend,
    _isolated,
    make_plan,
    run,
)

CFG = Cfg(budget_usd=3.00, max_hours=8.0)


def finished_fleet(**kw) -> dict:
    base = {
        "phase": "running", "n_pages_expected": 288, "pages_done": 288,
        "pages_per_hour": 2944, "missing_count": 0, "wall_sec": 352.0,
        "shards": [{"k": 1, "alive": False, "rc": 0}, {"k": 2, "alive": False, "rc": 0}],
        "ckpt": {"ok": 2, "exhausted": True},
    }
    base.update(kw)
    return base


def test_read_complete_but_silent_past_grace_is_drained() -> None:
    action, why = decide(Obs(progress=finished_fleet(), progress_age_sec=400, dph=0.17), CFG)
    assert action == "drain_stalled"
    assert "288" in why and "забираю" in why


def test_inside_the_grace_a_healthy_final_upload_is_left_alone() -> None:
    action, _ = decide(Obs(progress=finished_fleet(), progress_age_sec=120, dph=0.17), CFG)
    assert action == "wait"


@pytest.mark.parametrize("override", [
    {"pages_done": 250},                                         # ще є що читати
    {"missing_count": 3},                                        # догін — справа раннера
    {"shards": [{"k": 1, "alive": True, "rc": None}]},           # шард ще працює
    {"shards": [{"k": 1, "alive": False, "rc": 1}]},             # впав — інша гілка
    {"phase": "idle"},                                           # теплий бокс чекає довіска
])
def test_only_a_fully_read_dead_fleet_is_drained(override: dict) -> None:
    action, _ = decide(Obs(progress=finished_fleet(**override), progress_age_sec=900,
                           dph=0.17), CFG)
    assert action != "drain_stalled"


def test_drain_wins_over_the_stale_progress_rerent() -> None:
    """Без правила 20 хв тиші дали б `rerent` — викинули б готовий результат."""
    action, _ = decide(Obs(progress=finished_fleet(), progress_age_sec=1500, dph=0.17), CFG)
    assert action == "drain_stalled"


def test_supervisor_quiesces_fetches_then_destroys(tmp_path: Path) -> None:
    backend = FakeBackend(pages=100, texts=100)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    quiesced: list[bool] = []
    sup._quiesce_runner = lambda: quiesced.append(True)  # type: ignore[method-assign]
    progress = finished_fleet(n_pages_expected=100, pages_done=100)
    sup.backend = backend  # type: ignore[assignment]
    sup._read_progress = lambda: (progress, True)  # type: ignore[method-assign]
    sup._progress_age = lambda p: 600.0            # type: ignore[method-assign]
    sup._instance_state = lambda: "running"        # type: ignore[method-assign]
    asked: list[bool] = []
    backend.request_checkpoint = lambda h: asked.append(True) or "timeout"  # type: ignore[attr-defined]
    code = sup.run()

    assert asked == [], "заливка вже не йде — просити її ще раз означає чекати тайм-аут"

    order = [c.split(":")[0] for c in backend.calls]
    assert quiesced == [True]
    assert order.index("fetch") < order.index("destroy")
    assert order.count("fetch") == 1, "паралельних/повторних забірів бути не повинно"
    assert code == 0 and sup.state.verdict == "ok"
    assert any(i.kind == "drain_stalled" for i in sup.state.incidents)


def test_a_normal_done_still_asks_for_the_last_checkpoint(tmp_path: Path) -> None:
    """Штатний `done` — заливка йде, тож прохання «залий зараз» лишається: без
    нього забір тягнув би з машини те, що за секунду лягло б у сховище."""
    backend = FakeBackend(pages=100, texts=100)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    progress = finished_fleet(phase="done", n_pages_expected=100, pages_done=100)
    sup._read_progress = lambda: (progress, True)  # type: ignore[method-assign]
    sup._progress_age = lambda p: 5.0              # type: ignore[method-assign]
    sup._instance_state = lambda: "running"        # type: ignore[method-assign]
    asked: list[bool] = []
    backend.request_checkpoint = lambda h: asked.append(True) or "done"  # type: ignore[attr-defined]
    sup.run()
    assert asked == [True]
