"""План і стан: те, що агент читає замість логів, і те, що ловить погані плани."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpurunner.supervise import state as state_mod
from gpurunner.supervise.plan import load_plan
from gpurunner.supervise.state import CaseState, SupervisorState

GOOD = {
    "assets_url": "https://r2/assets.tgz",
    "budget_usd": 3.0,
    "max_hours": 8.0,
    "cases": [{"case": "spr-6671", "pages_url": "https://r2/spr-6671.tar", "n_pages": 1665}],
}


def write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---- план ------------------------------------------------------------------


def test_good_plan_loads(tmp_path: Path) -> None:
    plan = load_plan(write(tmp_path, GOOD))
    assert plan.total_pages == 1665
    assert plan.cases[0].case == "spr-6671"


def test_plan_without_page_count_is_rejected(tmp_path: Path) -> None:
    """🔴 Знаменник обов'язковий: без нього «нуль пропущених» — не факт, а
    відсутність факту, і неповний результат виглядає повним."""
    bad = {**GOOD, "cases": [{"case": "x", "pages_url": "https://r2/x.tar", "n_pages": 0}]}
    with pytest.raises(ValueError, match="Знаменник обов'язковий"):
        load_plan(write(tmp_path, bad))


def test_plan_without_assets_is_rejected(tmp_path: Path) -> None:
    """Без моделей і скриптів бокс підніметься й упаде — на оплачуваній карті."""
    with pytest.raises(ValueError, match="assets_url"):
        load_plan(write(tmp_path, {**GOOD, "assets_url": ""}))


def test_plan_without_limits_is_rejected(tmp_path: Path) -> None:
    """Бюджет і строк — єдині межі, всередині яких наглядач діє без питань."""
    with pytest.raises(ValueError, match="budget_usd"):
        load_plan(write(tmp_path, {**GOOD, "budget_usd": 0}))


def test_empty_case_list_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="жодної справи"):
        load_plan(write(tmp_path, {**GOOD, "cases": []}))


# ---- стан ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path))


def test_state_round_trips_through_latest() -> None:
    """Агент має читати стан, не знаючи імені сесії."""
    st = SupervisorState(session="sup-test")
    st.cases = [CaseState(case="spr-6671", n_pages_expected=1665)]
    st.save()
    loaded = state_mod.load()
    assert loaded is not None
    assert loaded["session"] == "sup-test"
    assert loaded["cases"][0]["case"] == "spr-6671"


def test_incident_carries_a_price() -> None:
    st = SupervisorState(session="s")
    st.note("ssh_auth_denied", "хост відхилив ключ", action="знищено за 41 с",
            machine_id=4711, cost_usd=0.003)
    assert st.incidents[0].cost_usd == 0.003
    assert st.incidents[0].machine_id == 4711


@pytest.mark.parametrize(
    ("verdict", "code", "human"),
    [
        ("ok", 0, False),
        ("failed", 3, False),
        ("incomplete", 4, True),
        ("budget_stop", 5, True),
        ("market_empty", 6, True),
        ("deadline", 7, False),
    ],
)
def test_exit_code_and_human_flag_agree(verdict: str, code: int, human: bool) -> None:
    """🔴 Прапорець і код виходу виводяться з ОДНОГО вердикту.

    Якби їх виставляли окремо, вони б розійшлись — і агент або смикав би
    людину дарма, або мовчки закривав захід, який стоїть.
    """
    st = SupervisorState(session="s")
    st.finish(verdict, "чому", human_action="що робити")
    assert st.exit_code == code
    assert st.human_action_required is human
    assert (st.human_action is not None) is human


def test_deadline_does_not_need_a_human() -> None:
    """Час скінчився — наглядач сам погасив і сам звітував; питати нема чого."""
    st = SupervisorState(session="s")
    st.finish("deadline", "8.5 год при стелі 8")
    assert not st.human_action_required


def test_missing_state_reads_as_none() -> None:
    assert state_mod.load("немає-такої-сесії") is None


# ---- план як файл із валідацією, а не рядок команди -------------------------


def test_yaml_plan_is_accepted(tmp_path: Path) -> None:
    """План правлять руками — YAML дозволяє коментарі й не падає через кому."""
    p = tmp_path / "plan.yml"
    p.write_text(
        "# захід на дві справи\n"
        "assets_url: https://r2/assets.tgz\n"
        "budget_usd: 3.0\n"
        "max_hours: 8\n"
        "cases:\n"
        "  - case: spr-6671      # метрика 1863\n"
        "    pages_url: https://r2/spr-6671.tar\n"
        "    n_pages: 1665\n",
        encoding="utf-8",
    )
    plan = load_plan(p)
    assert plan.total_pages == 1665


def test_empty_url_is_named_as_a_broken_shell_variable(tmp_path: Path) -> None:
    """🔴 Двічі за сесію порожня shell-змінна давала живий оплачуваний бокс,
    який стоїть і нічого не рахує. Валідація файла ловить це за нуль коштів —
    і головне, називає причину, а не просто «порожньо»."""
    bad = {**GOOD, "cases": [{"case": "x", "pages_url": "", "n_pages": 10}]}
    with pytest.raises(ValueError, match="shell-змінна"):
        load_plan(write(tmp_path, bad))


def test_non_url_is_rejected(tmp_path: Path) -> None:
    """`$(...)`, що не спрацював, лишає по собі текст, а не адресу."""
    bad = {**GOOD, "cases": [{"case": "x", "pages_url": "cases/x.tar", "n_pages": 10}]}
    with pytest.raises(ValueError, match="не схожий на URL"):
        load_plan(write(tmp_path, bad))


# ---- живий процес ≠ сирота (розбір 2026-08-19) -----------------------------


def _quiet(pid: int) -> dict:
    """Стан, який давно не оновлювався, без термінального вердикту."""
    from datetime import UTC, datetime, timedelta

    old = (datetime.now(tz=UTC) - timedelta(minutes=9)).isoformat(timespec="seconds")
    return {"session": "s", "phase": "renting", "verdict": None, "pid": pid,
            "updated": old, "next_poll_sec": 30}


def test_live_supervisor_is_never_orphaned(monkeypatch) -> None:
    """🔴🔴 Вердикт рахувався з ВІКУ ЗАПИСУ, а тихі фази законні: замок ринку,
    качання образу, очікування SSH (360 с).

    2026-08-19 це дало п'ять хибних «ЗАХІД ОБІРВАНО» за один захід — на заході,
    який у ту саму секунду качав образ на щойно оплаченому боксі. Кожна тривога
    підказувала `cancel --all-running`, тобто вбити свою ж роботу й заплатити
    за неї вдруге.
    """
    from gpurunner.supervise import state as state_mod

    monkeypatch.setattr(state_mod, "_pid_alive", lambda _pid: True)
    out = state_mod.diagnose(_quiet(4242))
    assert out.get("verdict") is None
    assert out["supervisor_alive"] is True
    assert not out.get("human_action_required")
    assert "ЖИВИЙ" in out["why"] and "9 хв" in out["why"]
    assert out["quiet_min"] == pytest.approx(9, abs=0.5)


def test_dead_supervisor_is_still_orphaned(monkeypatch) -> None:
    """Сирота — це ВІДСУТНІЙ процес, і цей сигнал мусить лишитись гучним."""
    from gpurunner.supervise import state as state_mod

    monkeypatch.setattr(state_mod, "_pid_alive", lambda _pid: False)
    out = state_mod.diagnose(_quiet(4242))
    assert out["verdict"] == "orphaned"
    assert out["human_action_required"] is True
    assert "не існує" in out["why"]


def test_finished_run_is_left_alone(monkeypatch) -> None:
    from gpurunner.supervise import state as state_mod

    monkeypatch.setattr(state_mod, "_pid_alive", lambda _pid: False)
    data = _quiet(4242) | {"verdict": "ok"}
    assert state_mod.diagnose(data)["verdict"] == "ok"
