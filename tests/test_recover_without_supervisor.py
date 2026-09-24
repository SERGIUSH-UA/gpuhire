"""Наглядач помер, бокс працював далі: гроші поза наглядом і підсумок у стані.

24.09.2026: наглядач обірвався близько 02:05 разом із сесією, бокс читав і писав
чекпоінти до 03:41 — ~1.2 год оренди ніде не порахована.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from gpurunner.htr.recover import patch_state, unseen_spend

DIED = "2026-09-23T23:28:50+00:00"
DIED_T = datetime.fromisoformat(DIED).timestamp()
STATE = {"updated": DIED, "box": {"dph_total": 0.1356, "instance_id": "52309609"},
         "budget": {"spent_usd": 0.283}}


def test_a_finished_box_is_billed_up_to_its_self_destruct() -> None:
    beat = {"t": DIED_T + 3600, "final": True}
    usd, how = unseen_spend(STATE, beat, box_alive=False, autodestroy_hours=0.5)
    assert abs(usd - 1.5 * 0.1356) < 1e-6
    assert "верхня межа" in how


def test_a_box_that_went_silent_is_billed_to_its_last_beat() -> None:
    beat = {"t": DIED_T + 3600, "final": False}
    usd, _ = unseen_spend(STATE, beat, box_alive=False, autodestroy_hours=0.5)
    assert abs(usd - (3600 + 300) / 3600 * 0.1356) < 1e-6


def test_a_live_box_is_billed_until_now() -> None:
    usd, how = unseen_spend(STATE, None, box_alive=True, autodestroy_hours=0.5,
                            now=DIED_T + 7200)
    assert abs(usd - 2 * 0.1356) < 1e-6 and "живий" in how


def test_without_a_heartbeat_there_is_no_made_up_number() -> None:
    usd, how = unseen_spend(STATE, None, box_alive=False, autodestroy_hours=0.5)
    assert usd == 0.0 and "серцебиття немає" in how


def test_the_state_gets_the_unseen_money_and_a_verdict(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps(dict(STATE, phase="running", verdict=None, incidents=[])),
                    encoding="utf-8")
    patch_state(path, extra_usd=0.17, how="тест", verdict="ok", why="4 справи повні")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["budget"]["spent_usd"] == 0.453 and data["budget"]["unseen_usd"] == 0.17
    assert data["phase"] == "finished" and data["verdict"] == "ok"
    assert data["incidents"][-1]["kind"] == "recovered"
    assert data["updated"] > DIED


def test_the_plan_carries_the_heartbeat_links_without_warnings(tmp_path: Path) -> None:
    from gpurunner.supervise.plan import load_plan

    raw = {"assets_url": "https://r2/a.tgz", "budget_usd": 1, "max_hours": 2,
           "plan_id": "20260924-000000-abc", "heartbeat_put_url": "https://r2/hb?put",
           "heartbeat_url": "https://r2/hb?get",
           "cases": [{"case": "spr-1", "pages_url": "https://r2/p.tar", "n_pages": 3,
                      "out_dir": str(tmp_path / "o")}]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    plan = load_plan(path)
    assert plan.heartbeat_put_url.endswith("put") and plan.heartbeat_url.endswith("get")


def test_without_a_heartbeat_the_last_checkpoint_marks_the_end() -> None:
    from datetime import timedelta

    from gpurunner.htr.recover import last_store_write

    base = datetime(2026, 9, 24, 0, 41, tzinfo=UTC)
    objs = {"ckpt/a/": [{"modified": base - timedelta(hours=1)}, {"modified": base}],
            "ckpt/b/": [{"modified": base - timedelta(hours=2)}]}
    got = last_store_write(["ckpt/a", "ckpt/b", ""], list_fn=lambda p: objs.get(p, []))
    assert got == base.timestamp()


def test_the_session_is_billed_by_vast_charges_of_its_own_machines(tmp_path: Path) -> None:
    """Рахунок Vast — правда: у ньому й трафік, якого немає в ціні за годину
    (52309609: GPU $0.32 + диск $0.03 + завантаження $0.05 = $0.40)."""
    from gpurunner.htr.recover import billed, session_instances

    log = tmp_path / "s.log"
    log.write_text("[supervise] 💵 інстанс 52300655 створено — гроші пішли\n"
                   "[supervise] 💵 інстанс 52309609 створено — гроші пішли\n", encoding="utf-8")
    ids = session_instances(STATE, log)
    assert ids == ["52309609", "52300655"]
    charges = [
        {"instance": "52309609", "amount": 0.403,
         "items": {"gpu": 0.32, "disk": 0.03, "bwd": 0.053, "bwu": 0.0}},
        {"instance": "52300655", "amount": 0.18, "items": {"gpu": 0.17, "disk": 0.01}},
        {"instance": "99999999", "amount": 5.0, "items": {"gpu": 5.0}},
    ]
    bill = billed(charges, ids)
    assert abs(bill["total"] - 0.583) < 1e-9
    assert abs(bill["parts"]["bwd"] - 0.053) < 1e-9


def test_a_bill_replaces_the_estimate_in_the_state(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps(dict(STATE, incidents=[])), encoding="utf-8")
    patch_state(path, extra_usd=0.23, how="оцінка", verdict="ok", why="x", billed_usd=0.583)
    budget = json.loads(path.read_text(encoding="utf-8"))["budget"]
    assert budget["spent_usd"] == 0.583 and budget["billed_usd"] == 0.583
    assert budget["unseen_usd"] == 0.3


def test_finished_sessions_get_the_late_bill(tmp_path: Path) -> None:
    """Пост фактум: рахунок Vast приходить пізніше за фініш — звірка дописує його."""
    from gpurunner.htr.recover import reconcile_sessions

    now = DIED_T + 3600
    done = dict(STATE, session="s1", phase="finished", instances=["52309609"])
    running = dict(STATE, session="s2", phase="running", instances=["52300655"])
    (tmp_path / "s1.json").write_text(json.dumps(done), encoding="utf-8")
    (tmp_path / "s2.json").write_text(json.dumps(running), encoding="utf-8")
    (tmp_path / "latest-s1.json").write_text(json.dumps(done), encoding="utf-8")
    charges = [{"instance": "52309609", "amount": 0.403, "items": {}},
               {"instance": "52300655", "amount": 0.18, "items": {}}]
    got = reconcile_sessions(tmp_path, charges, days=3, now=now)
    assert got == [("s1", 0.283, 0.403)]
    data = json.loads((tmp_path / "s1.json").read_text(encoding="utf-8"))
    assert data["budget"]["spent_usd"] == 0.403 and data["budget"]["billed_usd"] == 0.403
    assert reconcile_sessions(tmp_path, charges, days=3, now=now) == [], "вдруге — без змін"


def test_the_ledger_matches_the_vast_bill_month_by_month(tmp_path: Path, monkeypatch) -> None:
    """Журнал витрат ≠ рахунок: наглядач писав «ціна × час», прогони без
    наглядача не писались зовсім. Після звірки місяць дорівнює рахунку."""
    from types import SimpleNamespace

    from gpurunner.core import budget
    from gpurunner.htr.recover import ledger_from_bill

    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path))
    budget.reserve_within_cap("vast", "h-sup", 0.50, cap=1e9)
    budget.settle("vast", "h-sup", 0.283)
    aug = datetime(2026, 8, 10, tzinfo=UTC).timestamp()
    sep = datetime(2026, 9, 23, tzinfo=UTC).timestamp()
    charges = [
        {"instance": "52309609", "amount": 0.403, "day": sep},
        {"instance": "47350818", "amount": 3.10, "day": aug},     # прогін без наглядача
        {"instance": "11111111", "amount": 0.05, "day": aug},     # машина без прогону
    ]
    handles = [SimpleNamespace(id="h-sup", remote_id="52309609", created_at="2026-09-23T21:00:00+00:00"),
               SimpleNamespace(id="h-old", remote_id="47350818", created_at="2026-08-10T08:14:00+00:00")]
    rows = ledger_from_bill(charges, handles)
    assert ("vast-instance-11111111", 0.05, "2026-08-10T00:00:00+00:00") in rows
    months = budget.reconcile_with_bill("vast", rows)
    assert abs(months["2026-09"]["after"] - 0.403) < 1e-9
    assert abs(months["2026-08"]["after"] - 3.15) < 1e-9
    again = budget.reconcile_with_bill("vast", rows)
    assert abs(again["2026-08"]["after"] - 3.15) < 1e-9, "повтор не подвоює"
