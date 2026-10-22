"""Обрив НАШОЇ мережі не залишає бокс тарифікуватись і не вбиває живу машину.

18.10.2026 (відгук користувачки, 0.33.1): година без інтернету — наглядачі
вийшли `finished · incomplete` з `getaddrinfo failed` на DELETE інстансу, три
бокси горіли далі. Тут — те саме на підробленому бекенді: гасіння чекає мережу
й повторюється, а тік без мережі не рахується тиком мертвого боксу.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from gpurunner.core import net
from gpurunner.core.backend import BackendError, NetworkError
from gpurunner.supervise.htr import Supervisor
from tests.test_supervise_flow import (  # noqa: F401  (фікстура _isolated)
    DONE,
    FakeBackend,
    _isolated,
    make_plan,
)


class _Clock:
    """Годинник, який крутить лише `sleep`: стеля очікування без справжніх годин."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, sec: float) -> None:
        self.slept.append(sec)
        self.now += sec


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(net, "clock", c.clock)
    monkeypatch.setattr(net, "sleep", c.sleep)
    return c


class _FlakyCancel(FakeBackend):
    """`cancel` падає заданими помилками, потім гасить."""

    def __init__(self, errors: list[Exception], **kw) -> None:
        super().__init__(pages=100, texts=100, **kw)
        self.errors = list(errors)

    def cancel(self, handle, *, force: bool = False) -> None:
        self.calls.append("cancel_try")
        if self.errors:
            raise self.errors.pop(0)
        self.calls.append("destroy")


def _run(sup: Supervisor, backend: FakeBackend, ticks: list) -> int:
    seq = iter(ticks)
    last = ticks[-1]
    sup.backend = backend  # type: ignore[assignment]
    sup._read_progress = lambda: next(seq, last)  # type: ignore[method-assign]
    sup._progress_age = lambda p: 1.0              # type: ignore[method-assign]
    sup._instance_state = lambda: "running"        # type: ignore[method-assign]
    return sup.run()


def _dns_down_for(monkeypatch: pytest.MonkeyPatch, n: int) -> list[int]:
    """Перші `n` проб DNS — «мережі нема», далі — є."""
    asked = [0]

    def probe(*a, **k) -> bool:
        asked[0] += 1
        return asked[0] > n

    monkeypatch.setattr(net, "reachable", probe)
    return asked


def test_destroy_waits_for_the_network_and_puts_the_box_out(
        tmp_path: Path, clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    _dns_down_for(monkeypatch, 3)
    backend = _FlakyCancel([NetworkError("getaddrinfo failed"),
                            NetworkError("getaddrinfo failed")])
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]

    code = _run(sup, backend, [(DONE, True)])

    assert "destroy" in backend.calls
    assert code == 0
    assert sup.state.verdict == "ok"
    assert not sup.state.human_action_required
    kinds = [i.kind for i in sup.state.incidents]
    assert "destroy_waiting_network" in kinds
    assert "network_back" in kinds
    assert "destroy_failed" not in kinds


def test_a_refusal_from_vast_is_still_terminal(tmp_path: Path, clock: _Clock) -> None:
    """Відповідь Vast (не мережа) повторами не лікується — кличемо людину, як було."""
    refusal = BackendError("Vast.ai DELETE failed: 400")
    refusal.status_code = 400
    backend = _FlakyCancel([refusal] * 10)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]

    _run(sup, backend, [(DONE, True)])

    assert "destroy" not in backend.calls
    # Без очікувань і без циклу повторів: по одній спробі на кожне гасіння.
    assert backend.calls.count("cancel_try") <= 2
    assert sup.state.verdict == "incomplete"
    assert sup.state.human_action_required
    assert clock.slept == []


def test_the_wait_has_a_ceiling(tmp_path: Path, clock: _Clock,
                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(net, "reachable", lambda *a, **k: False)
    backend = _FlakyCancel([NetworkError("getaddrinfo failed")] * 10_000)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]

    _run(sup, backend, [(DONE, True)])

    assert "destroy" not in backend.calls
    assert sup.state.verdict == "incomplete"
    assert sup.state.human_action_required
    assert sum(clock.slept) == pytest.approx(Supervisor.NETWORK_WAIT_MAX_SEC)


def test_an_offline_tick_is_not_a_dead_box(tmp_path: Path, clock: _Clock,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """Без мережі SSH мовчить — але бокс живий: не гасити й не переорендовувати."""
    _dns_down_for(monkeypatch, 40)
    backend = FakeBackend(pages=100, texts=100)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    sup._progress_age = lambda p: 10 ** 6  # type: ignore[method-assign]

    working = {"phase": "work", "n_pages_expected": 100, "pages_done": 40,
               "pages_per_hour": 2000}
    code = _run(sup, backend, [(working, True)] + [(None, False)] * 40 + [(DONE, True)])

    assert backend.calls.count("rent") == 1
    assert backend.calls.count("destroy") == 1
    assert code == 0
    assert sup.state.verdict == "ok"
    assert "network_down" in [i.kind for i in sup.state.incidents]


def test_a_transport_failure_is_a_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpurunner.backends.vast import VastBackend

    def boom(*a, **k):
        raise httpx.ConnectError("[Errno 11001] getaddrinfo failed")

    monkeypatch.setattr(httpx, "request", boom)
    monkeypatch.setattr(VastBackend, "_api_key", lambda self: "k")
    with pytest.raises(NetworkError):
        VastBackend()._request_once("DELETE", "/instances/1/")


def test_retry_through_outage_passes_other_errors_at_once(clock: _Clock) -> None:
    calls = [0]

    def call() -> None:
        calls[0] += 1
        raise BackendError("400")

    with pytest.raises(BackendError):
        net.retry_through_outage(call, max_wait_sec=3600)
    assert calls[0] == 1
    assert clock.slept == []


def test_an_endless_outage_ends_the_run_instead_of_hanging(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Стеля й у головному циклі: без мережі наглядач не висить вічно."""
    monkeypatch.setattr(net, "reachable", lambda *a, **k: False)
    backend = _FlakyCancel([NetworkError("getaddrinfo failed")] * 10)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    sup._progress_age = lambda p: 1.0  # type: ignore[method-assign]
    clock = iter(range(0, 10 ** 9, 3600))
    import gpurunner.supervise.htr as H

    monkeypatch.setattr(H.time, "time", lambda: float(next(clock)))
    working = {"phase": "work", "n_pages_expected": 100, "pages_done": 40,
               "pages_per_hour": 2000}
    _run(sup, backend, [(working, True)] + [(None, False)] * 100)

    assert sup.state.verdict == "incomplete" and sup.state.human_action_required
    assert backend.calls.count("cancel_try") <= 2, "фінальне гасіння — одна спроба"


def test_a_network_failure_while_renting_bans_no_machine(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Обрив нашої мережі під час оренди — не вада машини й не наш конфіг."""

    _dns_down_for(monkeypatch, 2)
    monkeypatch.setattr(net, "sleep", lambda s: None)
    backend = FakeBackend(pages=100, texts=100)
    backend.submit_errors = [NetworkError("getaddrinfo failed")]
    backend.n_candidates = 2
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]

    code = _run(sup, backend, [(DONE, True)])

    assert code == 0 and sup.state.verdict == "ok"
    kinds = [i.kind for i in sup.state.incidents]
    assert "network_down" in kinds and "our_bug" not in kinds
    assert not sup._failed_machines, "справну машину не забанено"


def test_an_outage_turns_missing_pages_into_a_repeat_not_a_frame_review(
        tmp_path: Path, clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Лист 20.10.2026: ноутбук вийшов з мережі, черга закрилась «переглянути
    пропущені кадри» при 0 сторінок у більшості справ, а повтор тієї самої
    команди все дочитав. Нестача після обриву нашої мережі — справа забору."""
    _dns_down_for(monkeypatch, 5)
    backend = FakeBackend(pages=323, texts=203, complete=False)
    sup = Supervisor(make_plan(tmp_path, pages=323), backend=backend,  # type: ignore[arg-type]
                     tick_sec=0)
    working = {"phase": "work", "n_pages_expected": 323, "pages_done": 100,
               "pages_per_hour": 2000}
    _run(sup, backend, [(working, True)] + [(None, False)] * 5
         + [({**DONE, "n_pages_expected": 323, "pages_done": 323}, True)])

    assert sup.state.verdict == "incomplete" and sup.state.human_action_required
    assert "ту саму команду" in (sup.state.human_action or "")
    assert "переглянути пропущені кадри" not in (sup.state.human_action or "")


def test_without_an_outage_missing_pages_still_ask_for_the_frames(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(net, "reachable", lambda *a, **k: True)
    backend = FakeBackend(pages=323, texts=203, complete=False)
    sup = Supervisor(make_plan(tmp_path, pages=323), backend=backend,  # type: ignore[arg-type]
                     tick_sec=0)
    _run(sup, backend, [({**DONE, "n_pages_expected": 323, "pages_done": 323}, True)])
    assert "переглянути пропущені кадри" in (sup.state.human_action or "")
