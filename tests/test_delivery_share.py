"""Доставка кадрів з дому: частка оренди, згода людини і два різні діагнози.

10.10.2026 сторонній користувач на каналі ~1 МБ/с бачив одне й те саме:
наглядач брав машину, заливав ассети, казав «беру іншу машину» — і так машина
за машиною. Межа рахувалась від стелі заходу, а винна завжди виявлялась
машина, хоча вузьким місцем був домашній канал.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpurunner.core import uplink
from gpurunner.core.backend import BackendError
from gpurunner.supervise import delivery as dm
from gpurunner.supervise.decide import Cfg, Obs, decide
from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import CasePlan, Plan
from gpurunner.supervise.state import EXIT_BY_VERDICT, SupervisorState

GB = 1_000_000_000


@pytest.fixture(autouse=True)
def _own_data_dir(tmp_path: Path, monkeypatch) -> None:
    """Журнал каналу — у тимчасовій теці, а не в справжньому `data_dir()`."""
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))


def _sup(*, n_pages: int = 4000, max_hours: float = 8.0,
         params: dict[str, Any] | None = None) -> Supervisor:
    sup = Supervisor.__new__(Supervisor)
    sup.plan = Plan(assets_url="", transport="box", assets_path="/немає/a.tgz",
                    cases=[CasePlan(case="sprava", pages_url="", n_pages=n_pages,
                                    out_dir="/o", pages_path="/немає/pages.tar")],
                    budget_usd=5.0, max_hours=max_hours, params=dict(params or {}))
    sup.state = SupervisorState(session="s")
    sup._origin_outside = "http://box"
    sup._uplink_mbs = 0.0
    sup._uplink_via = ""
    sup._gate_sizing = None
    sup._delivery_told = set()
    sup._delivery_refusal = None
    return sup


def _offer(pph: float = 2000.0, dph: float = 0.15) -> Any:
    return SimpleNamespace(offer={"dph_total": dph, "machine_id": 77, "id": 1},
                           sizing=SimpleNamespace(usable=True, pages_per_hour=pph))


def _history(*rates: float) -> None:
    for rate in rates:
        uplink.record(rate, 200_000_000, via="http", machine_id=1)


def _kinds(sup: Supervisor) -> list[str]:
    return [i.kind for i in sup.state.incidents]


# ── прогноз ──────────────────────────────────────────────────────────────────
def test_the_share_is_of_the_rent_not_of_the_ceiling() -> None:
    """🔴 Межа — частка оренди. Двадцять хвилин читання й година перевезення —
    це 75% грошей за перевезення, хоч би якою широкою була стеля заходу."""
    fc = dm.Forecast(nbytes=int(3.6 * GB), mbs=1.0, read_h=1 / 3, dph=0.15)

    assert fc.hours == pytest.approx(1.0)
    assert fc.share == pytest.approx(0.75)
    assert fc.too_costly
    assert fc.usd == pytest.approx(0.15)
    assert "75%" in fc.human() and "$0.15" in fc.human()


def test_an_unknown_channel_is_not_a_verdict() -> None:
    fc = dm.Forecast(nbytes=8 * GB, mbs=0.0, read_h=2.0)

    assert not fc.too_costly, "без заміру прогнозу немає — і відмови теж"
    assert "не міряний" in fc.human()


def test_the_advice_names_the_portion_that_fits() -> None:
    """Порада конкретна: скільки ГБ за цього каналу вкладається в межу."""
    fc = dm.Forecast(nbytes=25 * GB, mbs=1.0, read_h=3.0)

    # 3 год читання → доставці можна 1 год → 3.6 ГБ на 1 МБ/с.
    assert fc.fits_bytes() == pytest.approx(3.6 * GB, rel=0.01)
    advice = fc.advice()
    assert "R2" in advice and "3.6 ГБ" in advice
    assert f"-p {dm.ACCEPT_FLAG}=true" in advice


# ── журнал каналу ────────────────────────────────────────────────────────────
def test_the_uplink_journal_gives_the_median_of_recent() -> None:
    assert uplink.typical() is None
    _history(1.0, 1.2, 9.0)
    assert uplink.typical() == pytest.approx(1.2)


def test_a_tiny_transfer_is_not_a_measurement() -> None:
    """Вісім потоків на 5 МБ не розганяються — такий замір лише шум."""
    uplink.record(0.1, 5_000_000)
    assert uplink.typical() is None


# ── ворота доставки на орендованій машині ───────────────────────────────────
def test_a_slow_home_channel_stops_the_run_instead_of_renting_more() -> None:
    """🔴 Канал такий, як завжди, — вузьке місце дім. Наступна машина дала б
    те саме, тож вердикт `slow_uplink`, а не «беру іншу машину»."""
    sup = _sup()

    with pytest.raises(BackendError) as caught:
        # 100 МБ за 100 с = 1 МБ/с; 8 ГБ — 2.2 год проти 2 год читання.
        sup._gate_delivery_speed(100_000_000, 100.0, left_bytes=8 * GB,
                                 candidate=_offer())

    assert caught.value.outcome == "slow_uplink"
    assert "іншу машину" not in str(caught.value)
    why, advice = sup._delivery_refusal or ("", "")
    assert "канал дому" in why and "R2" in advice
    assert "delivery_rate" in _kinds(sup), "замір лишається в журналі заходу"
    assert uplink.typical() == pytest.approx(1.0), "замір іде в журнал каналу"


def test_a_machine_far_below_the_usual_channel_is_its_own_fault() -> None:
    """Звичний канал дому 8 МБ/с, а до цієї машини 1 — вада її вхідного
    каналу: брак без бану (`slow_for_data`) і наступний кандидат."""
    _history(8.0, 8.0, 8.0)
    sup = _sup()

    with pytest.raises(BackendError) as caught:
        sup._gate_delivery_speed(100_000_000, 100.0, left_bytes=8 * GB,
                                 candidate=_offer())

    assert caught.value.outcome == "slow_for_data"
    assert "іншу машину" in str(caught.value)


def test_consent_lets_the_delivery_go_and_says_so() -> None:
    sup = _sup(params={dm.ACCEPT_FLAG: "true"})

    sup._gate_delivery_speed(100_000_000, 100.0, left_bytes=8 * GB, candidate=_offer())

    assert "delivery_accepted" in _kinds(sup)
    assert sup.state.verdict is None


def test_consent_does_not_lift_the_hours_ceiling() -> None:
    """🔴 Згода знімає межу частки, а не строк: 2.2 год доставки й 2 год
    читання при стелі 4 год — захід спинився б посеред читання, заплативши за
    перевезення."""
    sup = _sup(max_hours=4.0, params={dm.ACCEPT_FLAG: "true"})

    with pytest.raises(BackendError) as caught:
        sup._gate_delivery_speed(100_000_000, 100.0, left_bytes=8 * GB,
                                 candidate=_offer())

    assert caught.value.outcome == "slow_uplink"
    why, advice = sup._delivery_refusal or ("", "")
    assert "стелю заходу 4 год" in why
    assert "`--max-hours` до 6" in advice


def test_the_hours_ceiling_is_checked_before_the_rent(monkeypatch) -> None:
    _history(1.0)
    sup = _sup(max_hours=4.0, params={dm.ACCEPT_FLAG: "true"})
    monkeypatch.setattr(Supervisor, "_pages_bytes_left", lambda self: 8 * GB)

    assert sup._delivery_fits_before_rent(_offer()) is False
    assert sup.state.verdict == "slow_uplink"
    assert "--max-hours" in (sup.state.human_action or "")


def test_a_fast_channel_passes() -> None:
    sup = _sup()

    # 8 МБ/с: 8 ГБ за ~17 хв проти 2 год читання — 12% оренди.
    sup._gate_delivery_speed(100_000_000, 12.5, left_bytes=8 * GB, candidate=_offer())

    note = next(i for i in sup.state.incidents if i.kind == "delivery_rate")
    assert "МБ/с" in note.detail and "оренди" in note.detail


def test_a_wide_ceiling_does_not_buy_hours_of_transport() -> None:
    """Доти стеля 8 год давала доставці 2 години навіть на 10 хвилинах читання."""
    sup = _sup(n_pages=330, max_hours=8.0)

    with pytest.raises(BackendError) as caught:
        # ~1 год доставки (3.6 ГБ на 1 МБ/с) на ~10 хв читання.
        sup._gate_delivery_speed(100_000_000, 100.0, left_bytes=int(3.6 * GB),
                                 candidate=_offer())
    assert caught.value.outcome == "slow_uplink"


# ── до оренди ────────────────────────────────────────────────────────────────
def test_a_known_slow_channel_is_refused_before_any_rent(monkeypatch) -> None:
    """Канал відомий із попередніх доставок — рішення безкоштовне, до оренди."""
    _history(1.0, 1.0)
    sup = _sup()
    monkeypatch.setattr(Supervisor, "_pages_bytes_left", lambda self: 8 * GB)

    assert sup._delivery_fits_before_rent(_offer()) is False
    assert sup.state.verdict == "slow_uplink"
    assert "оренди не було" in (sup.state.why or "")
    assert "R2" in (sup.state.human_action or "")
    assert sup.state.human_action_required


def test_an_unmeasured_channel_rents_and_measures(monkeypatch) -> None:
    sup = _sup()
    monkeypatch.setattr(Supervisor, "_pages_bytes_left", lambda self: 8 * GB)

    assert sup._delivery_fits_before_rent(_offer()) is True
    note = next(i for i in sup.state.incidents if i.kind == "delivery_forecast")
    assert "не міряний" in note.detail


def test_consent_rents_despite_a_known_slow_channel(monkeypatch) -> None:
    _history(1.0)
    sup = _sup(params={dm.ACCEPT_FLAG: "true"})
    monkeypatch.setattr(Supervisor, "_pages_bytes_left", lambda self: 8 * GB)

    assert sup._delivery_fits_before_rent(_offer()) is True
    assert "delivery_accepted" in _kinds(sup)


def test_the_bucket_is_not_judged() -> None:
    sup = _sup()
    sup.plan = Plan(assets_url="https://r2/a", transport="r2", cases=sup.plan.cases,
                    budget_usd=5.0, max_hours=8.0)
    _history(0.2)

    assert sup._delivery_fits_before_rent(_offer()) is True
    assert not sup.state.incidents


# ── відмова після оренди ─────────────────────────────────────────────────────
def test_a_slow_uplink_refusal_stops_without_blaming_the_machine() -> None:
    sup = _sup()
    sup._delivery_refusal = ("доставка з'їла б 70% оренди", "бакет R2 …")
    sup.settled_usd, sup._pending, sup.rents, sup.rents_wasted = 0.0, None, 0, 0
    sup._destroy_orphan = lambda e: None            # type: ignore[method-assign]
    sup._accrue = lambda: None                       # type: ignore[method-assign]
    sup._close_rent = lambda: None                   # type: ignore[method-assign]

    def no_registry(*a: Any, **k: Any) -> None:
        raise AssertionError("машина не винна — у реєстр її не пишемо")

    sup._record_offer = no_registry                  # type: ignore[method-assign]
    err = BackendError("instance 9 is RUNNING AND BILLING but setup failed: …")
    err.outcome = "slow_uplink"
    err.instance_id = "9"                            # type: ignore[attr-defined]

    verdict = sup._after_failed_submit(err, _offer(), time.monotonic())

    assert verdict == "stop"
    assert sup.state.verdict == "slow_uplink"
    assert sup.state.why == "доставка з'їла б 70% оренди"
    assert sup.state.human_action == "бакет R2 …"


def test_slow_uplink_is_a_human_decision_with_its_own_exit_code() -> None:
    st = SupervisorState(session="s")
    st.finish("slow_uplink", "довго", human_action="R2")

    assert st.human_action_required
    assert EXIT_BY_VERDICT["slow_uplink"] not in (
        v for k, v in EXIT_BY_VERDICT.items() if k != "slow_uplink")


# ── строк, до якого бокс чекає дані ──────────────────────────────────────────
def test_the_box_is_told_how_long_to_wait() -> None:
    sup = _sup(max_hours=8.0)
    sent: list[str] = []
    sup._exec_on = lambda client, cmd: sent.append(cmd) or ""   # type: ignore[method-assign]

    sup._promise_delivery(object(), 8 * GB, 1.0)       # 8000 с ×3 + 600 → стеля 8 год
    sup._promise_delivery(object(), 10_000_000, 8.0)   # секунди → підлога 30 хв

    assert all("/workspace/gpurunner/GO_UNTIL" in c for c in sent)
    assert "$(date +%s) + 24600 " in sent[0]
    assert "$(date +%s) + 1800 " in sent[1]


# ── раннер, що впав ──────────────────────────────────────────────────────────
def _obs(**kw: Any) -> Obs:
    return Obs(**{"rent_age_sec": 1000.0, "elapsed_h": 0.2, **kw})


def test_lost_inputs_take_another_machine_at_once() -> None:
    action, why = decide(_obs(runner_error="inputs never arrived (no GO sentinel)"),
                         Cfg(budget_usd=5.0, max_hours=8.0))
    assert action == "runner_failed"
    assert "inputs never arrived" in why


def test_a_runner_crash_before_work_is_not_the_machine() -> None:
    action, _ = decide(_obs(runner_error="ModuleNotFoundError('kraken')"),
                       Cfg(budget_usd=5.0, max_hours=8.0))
    assert action == "startup_failed"


def test_a_runner_crash_mid_run_takes_another_machine() -> None:
    progress = {"phase": "running", "pages_done": 10, "ts": "2026-10-10T10:00:00+00:00"}
    action, _ = decide(_obs(progress=progress, runner_error="RuntimeError('cuda')"),
                       Cfg(budget_usd=5.0, max_hours=8.0))
    assert action == "runner_failed"


def test_a_finished_case_is_fetched_not_rerented() -> None:
    action, _ = decide(_obs(progress={"phase": "done"}, last_phase="done",
                            runner_error="RuntimeError('після done')"),
                       Cfg(budget_usd=5.0, max_hours=8.0))
    assert action != "runner_failed"


def test_money_still_comes_first() -> None:
    action, _ = decide(_obs(runner_error="inputs never arrived", elapsed_h=9.0),
                       Cfg(budget_usd=5.0, max_hours=8.0))
    assert action == "destroy_deadline"


class _Sftp:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def open(self, path: str, mode: str = "r") -> Any:
        import io

        if path not in self.files:
            raise FileNotFoundError(path)
        return io.BytesIO(self.files[path])


def test_the_runner_error_is_read_from_the_box() -> None:
    status = "/workspace/gpurunner/_status.json"
    failed = json.dumps({"status": "failed", "error": "inputs never arrived"}).encode()
    running = json.dumps({"status": "running"}).encode()

    assert Supervisor._read_runner_error(_Sftp({status: failed})) == "inputs never arrived"
    assert Supervisor._read_runner_error(_Sftp({status: running})) == ""
    assert Supervisor._read_runner_error(_Sftp({})) == ""
    assert Supervisor._read_runner_error(_Sftp({status: "{битий".encode()})) == ""


# ── проводка: нові частини справді викликаються ─────────────────────────────
def _src() -> str:
    return (Path(__file__).resolve().parents[1] / "src" / "gpurunner" / "supervise"
            / "htr.py").read_text(encoding="utf-8")


def test_the_supervisor_reads_the_runner_status_every_tick() -> None:
    from tests.srcprobe import method_body

    src = _src()
    assert "self._runner_error = self._read_runner_error(sftp)" in method_body(
        src, "_read_progress")
    queue = method_body(src, "_run_queue")
    assert "runner_error=self._runner_error" in queue
    handler = queue[queue.index('if action == "runner_failed"'):]
    handler = handler[:handler.index("continue")]
    assert '"setup_failed"' in handler, "провина машини не доведена — не died_under_load"
    assert "_rent_the_rest" in handler


def test_the_forecast_is_asked_before_the_first_candidate() -> None:
    from tests.srcprobe import method_body

    body = method_body(_src(), "_rent_once")
    assert body.index("_delivery_fits_before_rent") < body.index("for attempt, candidate")


def test_the_box_is_promised_a_deadline_before_the_first_byte() -> None:
    from tests.srcprobe import method_body

    body = method_body(_src(), "_deliver_to_box")
    first_promise = body.index("_promise_delivery")
    assert first_promise < body.index("deliver(Path(self.plan.assets_path)")
    assert body.index("_promise_delivery", first_promise + 1) > body.index(
        "_gate_delivery_speed"), "після заміру строк звужується до чесного"
