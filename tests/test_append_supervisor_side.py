"""Наглядач мусить ДІЗНАТИСЬ про довісок, а не лише бокс (ДОРОБКА 48).

🔴 Чому це не косметика: забір результату йде `zip(plan.cases, state.cases)`.
Справа, про яку знає бокс, але не знає наглядач, порахується, приїде в стейджинг
і **нікуди не розкладеться** — оплачена й тихо втрачена. Тому єдиний писар на
бокс — наглядач: команда лише кладе справу в чергу-файл.
"""

from __future__ import annotations

import json

import pytest

from gpurunner.supervise import append as append_mod

pytestmark = pytest.mark.usefixtures("data_dir")


# ---- черга-файл ------------------------------------------------------------


def test_enqueue_then_drain_returns_the_case() -> None:
    append_mod.enqueue("s1", {"case": "230-1-50", "n_pages": 100})
    got = append_mod.drain("s1", set())
    assert [c["case"] for c in got] == ["230-1-50"]


def test_a_case_is_never_drained_twice() -> None:
    """🔴 Ідемпотентність — прямі гроші: на боксі повторна справа читається
    вдруге за наші кошти. Дубль у черзі неминучий (наглядач перезапустився,
    людина натиснула двічі), тож він мусить бути безпечним."""
    append_mod.enqueue("s1", {"case": "230-1-50"})
    seen: set[str] = set()
    assert len(append_mod.drain("s1", seen)) == 1
    assert append_mod.drain("s1", seen) == []

    append_mod.enqueue("s1", {"case": "230-1-50"})
    assert append_mod.drain("s1", seen) == []


def test_cases_already_in_the_plan_are_skipped() -> None:
    append_mod.enqueue("s1", {"case": "вже-в-плані"})
    assert append_mod.drain("s1", {"вже-в-плані"}) == []


def test_a_case_without_a_name_is_refused_at_the_door() -> None:
    """Безіменна справа в черзі нічого не означає: за нею не можна ні перевірити
    ідемпотентність, ні знайти теку результату."""
    with pytest.raises(ValueError, match="без імені"):
        append_mod.enqueue("s1", {"n_pages": 10})


def test_queues_of_different_sessions_do_not_mix() -> None:
    append_mod.enqueue("s1", {"case": "а"})
    append_mod.enqueue("s2", {"case": "б"})
    assert [c["case"] for c in append_mod.drain("s1", set())] == ["а"]
    assert [c["case"] for c in append_mod.drain("s2", set())] == ["б"]


def test_broken_lines_do_not_kill_the_queue() -> None:
    """Решта заходу вже оплачена — губити її через зіпсовану кому безглуздо."""
    path = append_mod.queue_path("s1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"case": "добра"}\n{зіпсовано\n[1,2]\n\n', encoding="utf-8")
    assert [c["case"] for c in append_mod.drain("s1", set())] == ["добра"]


def test_the_queue_file_is_a_journal_not_a_mailbox() -> None:
    """⚠ Файл НЕ чиститься читанням. Чистка зробила б чергу залежною від того,
    чи вижив наглядач між читанням і записом, — тобто перенесла б втрату в інше
    місце, а не прибрала її."""
    append_mod.enqueue("s1", {"case": "а"})
    append_mod.drain("s1", set())
    assert append_mod.queue_path("s1").is_file()
    assert json.loads(append_mod.queue_path("s1").read_text(
        encoding="utf-8").splitlines()[0])["case"] == "а"


# ---- вбирання наглядачем ---------------------------------------------------


class _FakeSupervisor:
    """Тільки те, чого торкається `_absorb_appended`. Справжній наглядач тягне
    за собою бекенд, замки й SSH — а перевіряється тут порядок кроків."""

    from gpurunner.supervise.htr import Supervisor

    _absorb_appended = Supervisor._absorb_appended
    _push_append = staticmethod(lambda payload: True)


def test_supervisor_puts_the_case_into_plan_and_state(monkeypatch) -> None:
    """🔴 План І облік — обидва. Лише план означав би, що справи немає у звірці
    повноти; лише облік — що її не буде в заборі."""
    from gpurunner.supervise import htr as htr_mod
    from gpurunner.supervise.plan import CasePlan, Plan
    from gpurunner.supervise.state import CaseState, SupervisorState

    append_mod.enqueue("s1", {"case": "нова", "n_pages": 50, "pages_url": "https://x/n"})

    sup = _FakeSupervisor()
    sup._handle = object()
    sup.plan = Plan(assets_url="", cases=[CasePlan(
        case="стара", pages_url="", n_pages=100, out_dir="")],
        budget_usd=3.0, max_hours=8.0)
    sup.state = SupervisorState(session="s1")
    sup.state.cases = [CaseState(case="стара", n_pages_expected=100)]
    sup._appended_seen = {"стара"}
    sup._locked = []
    sup._owner = "тест"
    sup.spent_usd = 0.5
    sup.settled_usd = 0.0
    sup.started = htr_mod.time.monotonic()
    sup._accrue = lambda: None
    sup._say = lambda line: None
    pushed: list = []
    sup._push_append = lambda payload: pushed.append(payload) or True
    monkeypatch.setattr(htr_mod.locks, "acquire",
                        lambda *a, **kw: type("L", (), {"resource": "r"})())

    sup._absorb_appended()

    assert [c.case for c in sup.plan.cases] == ["стара", "нова"]
    assert [c.case for c in sup.state.cases] == ["стара", "нова"]
    assert pushed and pushed[0]["case"] == "нова"
    assert pushed[0]["estimated_n_pages"] == 50


def test_supervisor_refuses_when_the_budget_is_gone(monkeypatch) -> None:
    """🔴 Бюджет і строк перевіряються ПРИ ДОПИСУВАННІ — інакше довісок тихо
    винесе захід за межу, яку задала людина."""
    from gpurunner.supervise import htr as htr_mod
    from gpurunner.supervise.plan import Plan
    from gpurunner.supervise.state import SupervisorState

    append_mod.enqueue("s1", {"case": "нова", "n_pages": 50})

    sup = _FakeSupervisor()
    sup._handle = object()
    sup.plan = Plan(assets_url="", cases=[], budget_usd=1.0, max_hours=8.0)
    sup.state = SupervisorState(session="s1")
    sup._appended_seen = set()
    sup._locked = []
    sup._owner = "тест"
    sup.spent_usd = 1.0          # бюджет вичерпано рівно
    sup.settled_usd = 1.0
    sup.started = htr_mod.time.monotonic()
    sup._accrue = lambda: None
    sup._say = lambda line: None
    sup._push_append = lambda payload: pytest.fail("не сміє штовхати на бокс")
    monkeypatch.setattr(htr_mod.locks, "acquire",
                        lambda *a, **kw: pytest.fail("не сміє брати замок"))

    sup._absorb_appended()

    assert sup.plan.cases == []
    kinds = [i.kind for i in sup.state.incidents]
    assert "append_refused" in kinds
