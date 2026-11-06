"""Довісок на живий бокс при транспорті `box`: кадри везе наглядач.

🔴 Відгук користувача 07.10.2026: наступні малі партії з `transport=box` не
можна було довісити на прогрітий бокс. Посилання в такій справі немає — архів
лежить у нас на диску, а команда передавала лише `pages_url`, тож рядок довіска
вів раннер у порожнечу. Кожна партія платила новий холодний старт: оренду,
підйом, заливку, добір ринку.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpurunner.supervise import append as append_mod

pytestmark = pytest.mark.usefixtures("data_dir")


class _Backend:
    def __init__(self) -> None:
        self.closed = 0

    def _ssh(self, handle: Any, timeout: int = 0) -> Any:
        backend = self

        class _Client:
            def close(self) -> None:
                backend.closed += 1

        return _Client()


def _sup(tmp_path: Path, monkeypatch, *, transport: str) -> Any:
    from gpurunner.supervise import htr as htr_mod
    from gpurunner.supervise.htr import Supervisor
    from gpurunner.supervise.plan import CasePlan, Plan
    from gpurunner.supervise.state import CaseState

    assets = tmp_path / "a.tgz"
    assets.write_bytes(b"a")
    plan = Plan(assets_url="" if transport == "box" else "https://r2/a.tgz",
                assets_path=str(assets) if transport == "box" else "",
                transport=transport,
                cases=[CasePlan(case="стара", pages_url="", n_pages=10, out_dir="")],
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


def _box_case(tmp_path: Path) -> dict[str, Any]:
    pages = tmp_path / "nova.tar"
    pages.write_bytes(b"x" * 100)
    return {"case": "нова", "n_pages": 40, "transport": "box",
            "pages_path": str(pages), "pages_url": "",
            "ckpt_prefix": "ckpt/нова", "ckpt_slots": 4, "params": {"model": "m.pt"}}


def test_a_box_case_lands_on_the_store_before_the_box_hears_of_it(
        tmp_path: Path, monkeypatch) -> None:
    from gpurunner.htr import box_transport as bt
    from gpurunner.supervise import htr as htr_mod

    sup = _sup(tmp_path, monkeypatch, transport="box")
    order: list[str] = []
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(bt, "endpoint_of", lambda backend, handle: "ep")
    monkeypatch.setattr(htr_mod.Supervisor, "_exec_on",
                        staticmethod(lambda client, cmd: order.append(cmd) or ""))

    def deliverer(outside, ep, known, client):
        def deliver(local: Path, rel: str, remote: str) -> int:
            order.append("deliver")
            sent.append((local.name, rel))
            return local.stat().st_size
        return deliver

    sup._deliverer = deliverer
    pushed: list[dict[str, Any]] = []
    sup._push_append = lambda payload: order.append("push") or pushed.append(payload) or True
    append_mod.enqueue("S", _box_case(tmp_path))

    sup._absorb_appended()

    assert order[0].startswith("mkdir -p ")
    assert order[1:] == ["deliver", "push"], "рядок довіска раніше за архів — 404 на боксі"
    assert sent == [("nova.tar", bt.pages_rels("нова")[0])]
    url = pushed[0]["pages_url"]
    assert url.startswith(bt.base_url(token=sup._origin_token)) and url.endswith(sent[0][1])
    assert pushed[0]["ckpt_urls"] and pushed[0]["model"] == "m.pt"
    added = sup.plan.cases[-1]
    assert (added.pages_path, added.ckpt_prefix, added.ckpt_slots) == (
        str(tmp_path / "nova.tar"), "ckpt/нова", 4), "пересадка мусить знати, що везти"
    assert [c.case for c in sup.state.cases] == ["стара", "нова"]
    assert sup.backend.closed == 1


def test_a_box_case_that_did_not_arrive_is_not_written_to_the_box(
        tmp_path: Path, monkeypatch) -> None:
    from gpurunner.htr import box_transport as bt
    from gpurunner.supervise import htr as htr_mod

    sup = _sup(tmp_path, monkeypatch, transport="box")
    monkeypatch.setattr(bt, "endpoint_of", lambda backend, handle: "ep")
    monkeypatch.setattr(htr_mod.Supervisor, "_exec_on",
                        staticmethod(lambda client, cmd: ""))

    def deliverer(outside, ep, known, client):
        def deliver(local: Path, rel: str, remote: str) -> int:
            raise bt.BackendError("шматок 0: reset")
        return deliver

    sup._deliverer = deliverer
    sup._push_append = lambda payload: pytest.fail("архіву на складі немає")
    append_mod.enqueue("S", _box_case(tmp_path))

    sup._absorb_appended()

    # Справа лишається в плані й обліку: забір покаже її недочитаною, а не загубить.
    assert [c.case for c in sup.plan.cases] == ["стара", "нова"]
    assert [c.case for c in sup.state.cases] == ["стара", "нова"]
    failed = [i for i in sup.state.incidents if i.kind == "append_push_failed"]
    assert failed and "кадри не доїхали" in failed[0].detail
    assert sup.backend.closed == 1


@pytest.mark.parametrize(("session", "case_transport"), [("r2", "box"), ("box", "r2")])
def test_a_case_from_a_plan_of_the_other_transport_is_refused(
        tmp_path: Path, monkeypatch, session: str, case_transport: str) -> None:
    """Справу `box` у заході через бакет нікому везти; справа з бакета в заході
    `box` зірвала б доставку всієї черги на пересадці (її `pages_path` порожній)."""
    from gpurunner.supervise import htr as htr_mod

    sup = _sup(tmp_path, monkeypatch, transport=session)
    raw = _box_case(tmp_path) if case_transport == "box" else {
        "case": "нова", "n_pages": 40, "pages_url": "https://r2/n.tar"}
    append_mod.enqueue("S", raw)
    sup._push_append = lambda payload: pytest.fail("не той транспорт")
    sup._deliver_appended = lambda case: pytest.fail("не той транспорт")
    monkeypatch.setattr(htr_mod.locks, "acquire",
                        lambda *a, **kw: pytest.fail("не сміє брати замок"))

    sup._absorb_appended()

    assert [c.case for c in sup.plan.cases] == ["стара"]
    refused = [i for i in sup.state.incidents if i.kind == "append_refused"]
    assert refused and f"`{case_transport}`" in refused[0].detail


def test_the_command_hands_the_supervisor_what_a_box_case_needs(
        tmp_path: Path, monkeypatch) -> None:
    import json

    from typer.testing import CliRunner

    from gpurunner.cli import app
    from gpurunner.supervise.state import SupervisorState

    SupervisorState(session="S").save()
    assets = tmp_path / "a.tgz"
    assets.write_bytes(b"a")
    pages = tmp_path / "nova.tar"
    pages.write_bytes(b"x")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "transport": "box", "assets_path": str(assets), "budget_usd": 1.0, "max_hours": 1.0,
        "cases": [{"case": "нова", "n_pages": 40, "out_dir": str(tmp_path / "o"),
                   "pages_path": str(pages), "ckpt_prefix": "ckpt/нова", "ckpt_slots": 4,
                   "flatten_out": True, "frame_mpx_median": 8.9}],
    }), encoding="utf-8")

    res = CliRunner().invoke(app, ["htr", "append", "--plan", str(plan),
                                   "--case", "нова", "--session", "S"])

    assert res.exit_code == 0, res.output
    got = append_mod.drain("S", set())[0]
    assert (got["transport"], got["pages_path"], got["ckpt_prefix"], got["ckpt_slots"]) == (
        "box", str(pages), "ckpt/нова", 4)
    whole = got["plan_case"]
    assert whole["flatten_out"] is True and whole["frame_mpx_median"] == 8.9


def test_the_supervisor_rebuilds_the_whole_case_from_its_plan(
        tmp_path: Path, monkeypatch) -> None:
    """🔴 08.10.2026: довісок переносив поля вибірково й загубив `flatten_out` —
    довішені справи лягли сирою розкладкою раннера (`out/`, `out-diak_v6/`),
    і для конвеєра споживача були порожнечею без помилки."""
    sup = _sup(tmp_path, monkeypatch, transport="box")
    sup._deliver_appended = lambda case: True
    sup._push_append = lambda payload: True
    raw = _box_case(tmp_path)
    raw["plan_case"] = {"case": "нова", "pages_url": "", "n_pages": 40,
                        "out_dir": str(tmp_path / "o"), "flatten_out": True,
                        "local_dir": str(tmp_path / "кадри"), "frame_mpx_median": 8.9,
                        "pages_path": raw["pages_path"], "ckpt_prefix": "ckpt/нова",
                        "ckpt_slots": 4, "поле_з_майбутнього": 1}
    append_mod.enqueue("S", raw)

    sup._absorb_appended()

    case = sup.plan.cases[-1]
    assert case.flatten_out is True
    assert (case.local_dir, case.frame_mpx_median) == (str(tmp_path / "кадри"), 8.9)
    assert (case.pages_path, case.ckpt_slots) == (raw["pages_path"], 4)


def test_a_queue_line_without_the_whole_case_still_reads(tmp_path: Path, monkeypatch) -> None:
    """Черга, записана старшою командою, не має `plan_case` — читається як раніше."""
    sup = _sup(tmp_path, monkeypatch, transport="box")
    sup._deliver_appended = lambda case: True
    sup._push_append = lambda payload: True
    append_mod.enqueue("S", _box_case(tmp_path))

    sup._absorb_appended()

    case = sup.plan.cases[-1]
    assert (case.case, case.n_pages, case.ckpt_prefix) == ("нова", 40, "ckpt/нова")
    assert case.flatten_out is False
