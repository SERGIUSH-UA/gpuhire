"""Довісок, який бокс не взяв, повторюється; рядок їде через stdin, а не аргументом.

🔴 Відгук 21.11.2026: справа-пачка на 8 403 сторінки (17 частин `pages_parts` зі
списками кадрів) не влізла в командний рядок `printf '%s\\n' '<JSON>'`, дозапис
упав з `EOFError()`, а повтору не було: черга довіска віддає справу один раз,
повторний `htr append` відмовляв «уже в заході», і захід згортався б із нею
недочитаною.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpurunner.supervise import append as append_mod

pytestmark = pytest.mark.usefixtures("data_dir")


class _Chan:
    def __init__(self, sink: list[bytes]) -> None:
        self.sink = sink
        self.closed_write = False

    def shutdown_write(self) -> None:
        self.closed_write = True


class _Stdin:
    def __init__(self, sink: list[bytes]) -> None:
        self.sink = sink
        self.channel = _Chan(sink)

    def write(self, data: bytes) -> None:
        self.sink.append(data)

    def flush(self) -> None:
        pass


class _Out:
    def __init__(self, text: str) -> None:
        self.text = text

    def read(self) -> bytes:
        return self.text.encode()


class _Backend:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.stdin: list[bytes] = []
        self.fail = 0

    def _ssh(self, handle: Any, timeout: int = 0) -> Any:
        backend = self

        class _Client:
            def exec_command(self, cmd: str, timeout: int = 0) -> Any:
                if backend.fail:
                    backend.fail -= 1
                    raise EOFError()
                backend.commands.append(cmd)
                return _Stdin(backend.stdin), _Out("1\n"), _Out("")

            def close(self) -> None:
                pass

        return _Client()


def _sup(monkeypatch, tmp_path: Path | None = None) -> Any:
    from gpurunner.supervise import htr as htr_mod
    from gpurunner.supervise.htr import Supervisor
    from gpurunner.supervise.plan import CasePlan, Plan
    from gpurunner.supervise.state import CaseState

    box = tmp_path is not None
    assets = ""
    if tmp_path is not None:
        (tmp_path / "a.tgz").write_bytes(b"a")
        assets = str(tmp_path / "a.tgz")
    plan = Plan(assets_url="" if box else "https://r2/a.tgz", assets_path=assets,
                transport="box" if box else "r2",
                cases=[CasePlan(case="стара", pages_url="https://r2/s.tar", n_pages=10,
                                out_dir="")],
                budget_usd=3.0, max_hours=8.0)
    sup = Supervisor(plan, backend=_Backend(), session="S")  # type: ignore[arg-type]
    sup._handle = object()  # type: ignore[assignment]
    sup.state.cases = [CaseState(case="стара", n_pages_expected=10)]
    sup._appended_seen = {"стара"}
    sup._accrue = lambda: None  # type: ignore[method-assign]
    sup._say = lambda line: None  # type: ignore[method-assign]
    monkeypatch.setattr(htr_mod.locks, "acquire",
                        lambda *a, **kw: type("L", (), {"resource": "r"})())
    return sup


def _velyka(name: str = "пачка") -> dict[str, Any]:
    parts = [{"url": f"https://r2/p{i}.tar",
              "frames": [f"кадр_{i:02d}_{k:05d}.jpg" for k in range(500)]}
             for i in range(17)]
    return {"case": name, "n_pages": 8403, "pages_url": "", "pages_parts": parts,
            "transport": "r2"}


def test_a_huge_append_rides_stdin_not_the_command_line(monkeypatch) -> None:
    sup = _sup(monkeypatch)
    append_mod.enqueue("S", _velyka("пачка'з лапкою"))

    sup._absorb_appended()

    cmd = sup.backend.commands[0]
    assert len(cmd) < 400, "команда мусить бути сталою, без тіла довіска"
    assert "пачка" not in cmd, "ім'я справи не стає частиною оболонки"
    sent = b"".join(sup.backend.stdin).decode("utf-8")
    assert sent.endswith("\n") and sent.count("\n") == 1
    row = json.loads(sent)
    assert row["case"] == "пачка'з лапкою"
    assert len(row["pages_parts"]) == 17 and len(row["pages_parts"][16]["frames"]) == 500
    assert not sup._append_retry


def test_a_failed_push_is_retried_on_the_next_tick(monkeypatch) -> None:
    sup = _sup(monkeypatch)
    sup.backend.fail = 1
    append_mod.enqueue("S", _velyka())

    sup._absorb_appended()
    assert "пачка" in sup._append_retry
    assert [c.case for c in sup.state.cases] == ["стара", "пачка"]
    assert any(i.kind == "append_push_failed" for i in sup.state.incidents)

    sup._absorb_appended()           # наступний тік: черга вже порожня, а повтор іде
    assert not sup._append_retry
    assert json.loads(b"".join(sup.backend.stdin))["case"] == "пачка"
    assert [c.case for c in sup.state.cases] == ["стара", "пачка"], "не вдруге в облік"


def test_a_retry_is_dropped_when_the_box_changed(monkeypatch) -> None:
    """Нова оренда бере справу з плану сама: дописати її ще й рядком — двічі."""
    sup = _sup(monkeypatch)
    sup.backend.fail = 1
    append_mod.enqueue("S", _velyka())
    sup._absorb_appended()
    assert "пачка" in sup._append_retry

    sup._handle = object()  # type: ignore[assignment]  # переоренда
    sup._absorb_appended()
    assert not sup._append_retry
    assert sup.backend.commands == []


def test_a_box_case_whose_frames_failed_is_redelivered_then_pushed(
        tmp_path: Path, monkeypatch) -> None:
    from gpurunner.htr import box_transport as bt
    from gpurunner.supervise import htr as htr_mod

    sup = _sup(monkeypatch, tmp_path)
    monkeypatch.setattr(bt, "endpoint_of", lambda backend, handle: "ep")
    monkeypatch.setattr(htr_mod.Supervisor, "_exec_on",
                        staticmethod(lambda client, cmd: ""))
    tries = {"n": 0}

    def deliverer(outside, ep, known, client):
        def deliver(local: Path, rel: str, remote: str) -> int:
            tries["n"] += 1
            if tries["n"] == 1:
                raise bt.BackendError("шматок 0: reset")
            return 1
        return deliver

    sup._deliverer = deliverer
    pushed: list[dict[str, Any]] = []
    sup._push_append = lambda payload: pushed.append(payload) or True
    pages = tmp_path / "nova.tar"
    pages.write_bytes(b"x")
    append_mod.enqueue("S", {"case": "нова", "n_pages": 4, "transport": "box",
                             "pages_path": str(pages), "pages_url": "",
                             "ckpt_prefix": "ckpt/нова", "ckpt_slots": 1})

    sup._absorb_appended()
    assert not pushed and "нова" in sup._append_retry
    sup._absorb_appended()
    assert [p["case"] for p in pushed] == ["нова"] and tries["n"] == 2
    assert not sup._append_retry


def test_an_unreachable_box_costs_one_try_per_tick_and_one_note_per_case(monkeypatch) -> None:
    """Бокс не відповідає: тік не мусить висіти на SSH до кожної з N справ."""
    sup = _sup(monkeypatch)
    sup.backend.fail = 100
    for i in range(5):
        append_mod.enqueue("S", _velyka(f"с{i}"))
    sup._absorb_appended()           # перший тік: кожна нова справа — своя спроба
    assert len(sup._append_retry) == 5
    notes = [i for i in sup.state.incidents if i.kind == "append_push_failed"]
    assert len(notes) == 5
    left = sup.backend.fail
    sup._absorb_appended()           # повтор: одна спроба, далі стоп
    assert left - sup.backend.fail == 1
    assert len([i for i in sup.state.incidents if i.kind == "append_push_failed"]) == 5


def test_a_case_failing_for_its_own_reason_does_not_block_the_others(monkeypatch) -> None:
    sup = _sup(monkeypatch)
    sup.backend.fail = 2
    append_mod.enqueue("S", _velyka("a"))
    append_mod.enqueue("S", _velyka("b"))
    sup._absorb_appended()                     # обидві не доїхали
    assert list(sup._append_retry) == ["a", "b"]
    sup.backend.fail = 1
    sup._absorb_appended()                     # «a» знову впала — у кінець
    assert list(sup._append_retry) == ["b", "a"]
    sup._absorb_appended()                     # тепер «b» перша й доїжджає, далі «a»
    assert not sup._append_retry
