"""Дві сесії на одній машині не мають шкодити одна одній.

Кожен тест названий інцидентом 2026-08-11: того дня паралельні агенти п'ять
разів погасили одне одному живі бокси, ділили один `latest.json` і бралися за
ту саму справу.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpurunner.core import locks, manifest
from gpurunner.core.models import JobHandle, JobStatus
from gpurunner.supervise import state as state_mod
from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import CasePlan, Plan


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.setenv("GPURUNNER_BOXES_OVERRIDES", str(tmp_path / "over.json"))
    monkeypatch.delenv("GPURUNNER_OWNER", raising=False)


def make_plan(tmp_path: Path, case: str = "spr-6671") -> Plan:
    return Plan(
        assets_url="https://r2/assets.tgz",
        cases=[CasePlan(case=case, pages_url="https://r2/x.tar", n_pages=100,
                        out_dir=str(tmp_path / case))],
        budget_usd=1.0, max_hours=2.0,
    )


class _Backend:
    """Порожній бекенд: тести тут про власність і замки, не про оренду."""

    def find_candidates(self, **kw):  # pragma: no cover — не доходить
        raise AssertionError("оренди в цих тестах бути не має")


# ---- власник ----------------------------------------------------------------


def test_two_supervisors_without_session_get_different_owners(tmp_path: Path) -> None:
    """🔴 Головна дірка: `setdefault("GPURUNNER_OWNER", f"htr-{session or 'sup'}")`.

    Без `--session` ОБИДВІ сесії діставали однакового власника `htr-sup`, і
    owner-фільтр, заради якого все й робилось, переставав їх розрізняти.
    """
    import os

    a = Supervisor(make_plan(tmp_path, "a"), backend=_Backend())  # type: ignore[arg-type]
    owner_a = os.environ.pop("GPURUNNER_OWNER")
    b = Supervisor(make_plan(tmp_path, "b"), backend=_Backend())  # type: ignore[arg-type]
    owner_b = os.environ["GPURUNNER_OWNER"]

    assert owner_a != owner_b, "два наглядачі не можуть бути одним власником"
    assert a.state.session != b.state.session
    assert owner_a.endswith(a.state.session)


# ---- стан --------------------------------------------------------------------


def test_each_session_reads_its_own_latest(tmp_path: Path, monkeypatch) -> None:
    """🔴 `latest.json` був спільним, і `htr state --json` без `--session`
    показував ЧУЖИЙ захід — тобто агент керувався не своїм прогоном."""
    monkeypatch.setenv("GPURUNNER_OWNER", "htr-A")
    st_a = state_mod.SupervisorState(session="A")
    st_a.why = "захід А"
    st_a.save()

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-B")
    st_b = state_mod.SupervisorState(session="B")
    st_b.why = "захід Б"
    st_b.save()

    assert state_mod.load()["why"] == "захід Б"
    monkeypatch.setenv("GPURUNNER_OWNER", "htr-A")
    assert state_mod.load()["why"] == "захід А"


# ---- замки -------------------------------------------------------------------


def test_second_session_refuses_the_same_case(tmp_path: Path, monkeypatch) -> None:
    """Дві сесії на одну справу пишуть в одну теку й псують знаменник повноти."""
    monkeypatch.setenv("GPURUNNER_OWNER", "htr-A")
    locks.acquire("case:spr-6671", owner="htr-A", session="A")

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-B")
    sup = Supervisor(make_plan(tmp_path), backend=_Backend(),  # type: ignore[arg-type]
                     session="B", tick_sec=0)
    sup.run()

    assert sup.state.verdict == "failed"
    assert "htr-A" in sup.state.why
    assert [i.kind for i in sup.state.incidents] == ["locked"]


def test_different_cases_run_side_by_side(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GPURUNNER_OWNER", "htr-A")
    locks.acquire("case:alpha", owner="htr-A", session="A")
    monkeypatch.setenv("GPURUNNER_OWNER", "htr-B")
    locks.acquire("case:beta", owner="htr-B", session="B")  # не кидає


# ---- реєстр ------------------------------------------------------------------


def test_update_stamps_owner_on_ownerless_handles(monkeypatch) -> None:
    """«Нічиї» хендли лишались невидимими для масових операцій НАЗАВЖДИ."""
    h = JobHandle(backend="vast", remote_id="1", job_name="htr_case", gpu="any")
    manifest.add(h)
    assert manifest.get(h.id).owner == ""

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-A")
    h.status = JobStatus.RUNNING
    manifest.update(h)
    assert manifest.get(h.id).owner == "htr-A"


def test_foreign_handle_is_visible_as_foreign(monkeypatch) -> None:
    """Мітка інстансу не доказ власності — доказ лежить у реєстрі."""
    monkeypatch.setenv("GPURUNNER_OWNER", "htr-A")
    mine = JobHandle(backend="vast", remote_id="1", job_name="htr_case", gpu="any")
    manifest.add(mine)

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-B")
    theirs = JobHandle(backend="vast", remote_id="2", job_name="htr_case", gpu="any")
    manifest.add(theirs)

    assert manifest.get(mine.id).owner == "htr-A"
    assert manifest.get(theirs.id).owner == "htr-B"


# ---- гроші -------------------------------------------------------------------


def test_spend_is_visible_across_sessions(tmp_path: Path, monkeypatch) -> None:
    """🔴 Дві сесії з `--budget 3.00` спалювали $6, не бачачи одна одної."""
    from gpurunner.core import budget as budget_mod

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-A")
    budget_mod.reserve_within_cap("vast", "handle-a", 1.25, cap=1e9)

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-B")
    sup = Supervisor(make_plan(tmp_path), backend=_Backend(),  # type: ignore[arg-type]
                     session="B")
    assert sup.machine_wide_spend() >= 1.25


def test_settled_rent_is_not_forgotten_after_destroy(tmp_path: Path) -> None:
    """🔴 Найдорожча вада дня: `_destroy` занулював оффер, `_dph()` віддавав 0,
    і витрачене ставало НУЛЕМ — тобто бюджетний запобіжник не міг спинити
    серію переоренд за побудовою."""
    sup = Supervisor(make_plan(tmp_path), backend=_Backend(),  # type: ignore[arg-type]
                     session="S")
    sup.settled_usd = 0.42
    sup._offer = None
    sup._handle = None
    sup._accrue()
    assert sup.spent_usd == pytest.approx(0.42)


# ---- ручні вердикти ----------------------------------------------------------


def test_two_bans_do_not_erase_each_other(monkeypatch, tmp_path: Path) -> None:
    """🔴 Два одночасні `boxes ban` — класичний загублений апдейт.

    Стирається саме РУЧНЕ рішення людини, тобто єдине, чого автоматика не
    відтворить із заміру.
    """
    import json

    from gpurunner.core import boxes

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-A")
    with boxes.mutate_overrides() as data:
        data["never"]["111"] = "битий SSH"

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-B")
    with boxes.mutate_overrides() as data:
        data["never"]["222"] = "нема диска"

    saved = json.loads(boxes.overrides_path().read_text(encoding="utf-8"))
    assert saved["never"] == {"111": "битий SSH", "222": "нема диска"}


def test_ban_waits_for_a_live_holder(monkeypatch) -> None:
    """Поки хтось живий редагує — другий не лізе всередину файла."""
    from gpurunner.core import boxes, locks

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-B")
    locks.acquire("boxes:overrides", owner="htr-A", session="A")
    with pytest.raises(locks.LockBusy), boxes.mutate_overrides():
        pass  # pragma: no cover


def test_star_and_never_are_mutually_exclusive(monkeypatch) -> None:
    """Машина не може бути одночасно зіркою і в бані — інакше добір недетермінований."""
    from gpurunner.core import boxes

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-A")
    with boxes.mutate_overrides() as data:
        data.setdefault("star", {})["77"] = "швидка"
    with boxes.mutate_overrides() as data:
        data.setdefault("never", {})["77"] = "виявилась битою"
        data.get("star", {}).pop("77", None)

    saved = boxes.load_overrides()
    assert saved["never"]["77"] and "77" not in saved["star"]


# ---- гасіння -----------------------------------------------------------------


def _cancel(args: list[str], owner: str, monkeypatch) -> object:
    from typer.testing import CliRunner

    from gpurunner.cli import app

    monkeypatch.setenv("GPURUNNER_OWNER", owner)
    return CliRunner().invoke(app, ["cancel", *args])


def test_from_file_refuses_foreign_ids(tmp_path: Path, monkeypatch) -> None:
    """🔴 Сценарій `to_stop.txt`: у корені репо лежить готовий список id.

    Гілка `--from-file` власного owner-фільтра НЕ має — її накриває спільна
    охорона нижче. Тест існує саме тому, що це неочевидно з коду і легко
    зламати рефактором.
    """
    monkeypatch.setenv("GPURUNNER_OWNER", "htr-A")
    theirs = JobHandle(backend="vast", remote_id="777", job_name="htr_case", gpu="any")
    manifest.add(theirs)

    listing = tmp_path / "to_stop.txt"
    listing.write_text(f"# чужі\n{theirs.id}\n", encoding="utf-8")

    res = _cancel(["--from-file", str(listing)], "htr-B", monkeypatch)
    assert res.exit_code == 2
    assert "htr-A" in res.output


def test_explicit_id_refuses_foreign_handle(monkeypatch) -> None:
    """Найчастіший спосіб зробити боляче: `cancel <id>` з чужим id."""
    monkeypatch.setenv("GPURUNNER_OWNER", "htr-A")
    theirs = JobHandle(backend="vast", remote_id="778", job_name="htr_case", gpu="any")
    manifest.add(theirs)

    res = _cancel([theirs.id], "htr-B", monkeypatch)
    assert res.exit_code == 2
    assert "--force" in res.output


def test_all_running_without_owner_is_refused(monkeypatch) -> None:
    """Масове гасіння без власника = «вб'ю все на машині»."""
    monkeypatch.delenv("GPURUNNER_OWNER", raising=False)
    from typer.testing import CliRunner

    from gpurunner.cli import app

    res = CliRunner().invoke(app, ["cancel", "--all-running"])
    assert res.exit_code == 2
    assert "GPURUNNER_OWNER" in res.output


def test_ambiguous_state_refuses_instead_of_guessing(monkeypatch, tmp_path: Path) -> None:
    """🔴 Резерв «найсвіжіший latest-*.json» = «той, чий наглядач писав останнім».

    При двох живих заходах це з імовірністю ~50% ЧУЖИЙ, і агент переказав би
    людині результат не свого прогону («повно, оренду погашено», поки власний
    бокс горить). Змінні оточення не переживають між викликами оболонки, тож
    безіменний читач — типовий випадок, а не екзотика.
    """
    for name, why in (("A", "захід А"), ("B", "захід Б")):
        monkeypatch.setenv("GPURUNNER_OWNER", f"htr-{name}")
        st = state_mod.SupervisorState(session=name)
        st.why = why
        st.save()

    monkeypatch.delenv("GPURUNNER_OWNER", raising=False)
    got = state_mod.load()
    assert got and got.get("ambiguous") is True
    assert {s["session"] for s in got["sessions"]} == {"A", "B"}
    assert "GPURUNNER_OWNER" in got["why"]


def test_single_session_is_still_readable_without_owner(monkeypatch) -> None:
    """Один захід на машині — резерв мусить працювати, інакше зламався б
    звичайний однокористувацький сценарій заради багатокористувацького."""
    monkeypatch.setenv("GPURUNNER_OWNER", "htr-only")
    st = state_mod.SupervisorState(session="only")
    st.why = "єдиний захід"
    st.save()

    monkeypatch.delenv("GPURUNNER_OWNER", raising=False)
    got = state_mod.load()
    assert got and got.get("why") == "єдиний захід"


def test_killed_supervisor_is_visible_in_its_own_state(monkeypatch) -> None:
    """🔴 Агент читає `htr state --json` і НЕ МОЖЕ відрізнити «захід іде» від
    «наглядача вбито, а бокс лишився горіти»: в обох випадках у стані стоїть
    `phase: renting, verdict: null, human_action_required: false`.

    Заміряно 2026-08-11: перерваний захід лишив саме таку картину, агент
    доповів «усе гаразд, чекаю», а на Vast тарифікувались дві машини.
    """
    monkeypatch.setenv("GPURUNNER_OWNER", "htr-dead")
    st = state_mod.SupervisorState(session="dead")
    st.phase = "renting"
    st.pid = 999_999          # такого процесу немає
    st.save()

    got = state_mod.load("dead")
    assert got and got["supervisor_alive"] is False
    assert got["human_action_required"] is True
    assert got["verdict"] == "orphaned"
    assert got["exit_code"] != 0
    assert "ОБІРВАНО" in got["why"]
    assert "reconcile" in (got["human_action"] or "")


def test_a_live_supervisor_is_not_slandered(monkeypatch) -> None:
    """Живий наглядач мусить лишатись живим у звіті — інакше агент гасив би
    робочі заходи через хибну тривогу."""
    monkeypatch.setenv("GPURUNNER_OWNER", "htr-live")
    st = state_mod.SupervisorState(session="live")
    st.phase = "running"
    st.save()

    got = state_mod.load("live")
    assert got and got.get("supervisor_alive") is not False
    assert got.get("verdict") is None


def test_finished_run_is_never_called_orphaned(monkeypatch) -> None:
    """Захід, що дійшов до вердикту сам, діагностиці не підлягає."""
    monkeypatch.setenv("GPURUNNER_OWNER", "htr-done")
    st = state_mod.SupervisorState(session="done")
    st.pid = 999_999
    st.finish("ok", "усе прогнано")
    st.save()

    got = state_mod.load("done")
    assert got and got["verdict"] == "ok" and got["exit_code"] == 0
