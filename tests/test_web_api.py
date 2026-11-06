"""API дашборду: збірка огляду, кеш і те, що один зламаний бекенд нічого не валить."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="потрібна екстра web (uv sync --extra web)")

from fastapi.testclient import TestClient

from gpurunner.core import balances, quota
from gpurunner.core.backend import AuthError

pytestmark = pytest.mark.usefixtures("data_dir")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Клієнт із підміненим опитуванням бекендів — жодної мережі в тестах."""
    def fake_reports(names: Any = None) -> list[dict[str, Any]]:
        table = {
            # віддає баланс через API
            "modal": {"backend": "modal", "available": 11.8, "unit": "$", "spent": 18.2,
                      "detail": "ОЦІНКА", "url": "https://modal.com"},
            # API мовчить про GPU-квоту — головним стане розрахунок
            "kaggle": {"backend": "kaggle", "available": 4.0, "unit": "$", "spent": 1.0,
                       "detail": "AI-квота (НЕ GPU)", "url": "https://kaggle.com"},
            "colab": {"backend": "colab", "available": None, "unit": "compute units",
                      "spent": None, "detail": "залишок Colab через API не віддає", "url": ""},
        }
        out = []
        for name in names or ["kaggle", "modal", "colab"]:
            if name == "vast":
                raise AuthError("VAST_API_KEY не знайдено")
            if name == "lightning":
                raise RuntimeError("teamspace lookup failed")
            out.append(table.get(name, {"backend": name, "available": None, "unit": "",
                                        "spent": None, "detail": "", "url": ""}))
        return out

    monkeypatch.setattr(balances, "collect_reports", fake_reports)
    monkeypatch.setattr("gpurunner.web.app.BACKEND_NAMES",
                        ("kaggle", "modal", "colab", "vast", "lightning"))

    from gpurunner.web.app import create_app

    return TestClient(create_app())


# ---- огляд -----------------------------------------------------------------


def test_overview_survives_a_broken_backend(client: TestClient) -> None:
    """Ані відсутні креденшли, ані падіння SDK не мають ховати решту таблиці."""
    body = client.get("/api/overview").json()
    assert body["services"], body
    by_name = {s["backend"]: s for s in body["services"]}

    assert by_name["vast"]["auth"] == "не налаштовано"
    assert by_name["lightning"]["auth"] == "помилка"
    assert by_name["modal"]["auth"] == "ok" and by_name["modal"]["available"] == 11.8


def test_kaggle_dollar_quota_is_not_mistaken_for_gpu_hours(client: TestClient) -> None:
    """``KaggleBackend.balance()`` віддає AI-квоту в ДОЛАРАХ, а не GPU-години.

    Показати її як залишок означало б написати «лишилось 4», маючи на увазі
    долари, там, де користувач читає години. Головним для Kaggle мусить бути
    наш розрахунок, а відповідь API — з'їхати в примітку.
    """
    kaggle = {s["backend"]: s for s in client.get("/api/overview").json()["services"]}["kaggle"]
    assert kaggle["source"] == "розрахунок"
    assert kaggle["unit"] == "год"
    assert kaggle["available"] == pytest.approx(30.0)
    assert "AI-квота" in kaggle["api_note"]


def test_total_separates_measured_from_estimated(client: TestClient) -> None:
    quota.set_config("colab", plan="pro")     # на free одиниць немає, тож і оцінки теж
    total = client.post("/api/refresh").json()["total"]
    assert total["t4_hours"] > 0
    # Colab Pro переводиться за курсом плану — це оцінка, і вона видима окремо
    assert total["estimated_t4_hours"] > 0
    assert total["estimated_t4_hours"] < total["t4_hours"]   # Kaggle/Modal — вимір
    assert "RTX4090" in total["needs_factor"]


def test_free_colab_adds_nothing_to_the_total(client: TestClient) -> None:
    """Головне, заради чого free став дефолтом: не роздувати підсумок.

    На безкоштовному тарифі compute units не нараховуються взагалі, тож Colab не
    має внеску в T4-години — ані виміряного, ані оціненого.
    """
    total = client.get("/api/overview").json()["total"]
    assert total["estimated_t4_hours"] == 0
    colab = {s["backend"]: s for s in client.get("/api/overview").json()["services"]}["colab"]
    assert colab["t4"] is None and colab["available"] is None
    assert colab["quota"]["plan"] == "free"


# ---- кеш -------------------------------------------------------------------


def test_second_request_is_served_from_cache(client: TestClient, monkeypatch) -> None:
    calls: list[int] = []
    original = balances.collect_reports

    def counting(names: Any = None) -> list[dict[str, Any]]:
        calls.append(1)
        return original(names)

    monkeypatch.setattr(balances, "collect_reports", counting)
    client.get("/api/overview")
    first = len(calls)
    assert client.get("/api/overview").json()["cached"] is True
    assert len(calls) == first          # бекенди вдруге не опитувались

    assert client.post("/api/refresh").json()["cached"] is False
    assert len(calls) > first           # а на форс — опитувались


# ---- ручне керування -------------------------------------------------------


def test_anchor_changes_the_next_overview(client: TestClient) -> None:
    assert client.post("/api/quota/kaggle", json={"remaining": 12.5, "note": "звірив"}).status_code == 200
    kaggle = {s["backend"]: s for s in client.get("/api/overview").json()["services"]}["kaggle"]
    assert kaggle["available"] == pytest.approx(12.5)
    assert kaggle["quota"]["anchor"]["note"] == "звірив"

    assert client.delete("/api/quota/kaggle/anchor").json()["removed"] == 1
    kaggle = {s["backend"]: s for s in client.get("/api/overview").json()["services"]}["kaggle"]
    assert kaggle["available"] == pytest.approx(30.0)


def test_allowance_can_be_edited(client: TestClient) -> None:
    client.post("/api/quota/kaggle", json={"allowance": 20})
    assert quota.get_config("kaggle").allowance == 20.0
    assert client.get("/api/usage/kaggle").json()["allowance"] == 20.0


def test_bad_input_is_a_400_not_a_500(client: TestClient) -> None:
    assert client.post("/api/quota/kaggle", json={"remaining": -3}).status_code == 400
    assert client.post("/api/quota/kaggle", json={"period": "fortnight"}).status_code == 400
    assert client.post("/api/factors/T4", json={"factor": 0}).status_code == 400


def test_unknown_backend_is_404(client: TestClient) -> None:
    assert client.get("/api/usage/nope").status_code == 404
    assert client.get("/api/history/nope").status_code == 404
    assert client.post("/api/quota/nope", json={"remaining": 1}).status_code == 404
    assert client.delete("/api/quota/nope/anchor").status_code == 404


# ---- решта ендпойнтів ------------------------------------------------------


def test_usage_breakdown_shows_where_each_number_came_from(
    client: TestClient, make_run, now: datetime
) -> None:
    make_run(starts_h_ago=3, duration_h=1.5)
    body = client.get("/api/usage/kaggle").json()
    assert body["runs_counted"] == 1
    row = body["rows"][0]
    assert row["hours"] == pytest.approx(1.5, abs=0.02)
    assert row["source"] == "журнал" and row["exact"] is True


def test_runs_endpoint_splits_active_lost_and_recent(client: TestClient, make_run) -> None:
    """Живі прогони не мають тонути серед хендлів, які застрягли в черзі назавжди."""
    from gpurunner.core import manifest
    from gpurunner.core.models import JobHandle, JobStatus

    make_run(starts_h_ago=2, duration_h=1.0)                       # завершений
    make_run(starts_h_ago=1, duration_h=0, status=JobStatus.RUNNING)  # живий
    zombie = JobHandle(backend="kaggle", remote_id="me/zombie", job_name="j", gpu="T4")
    zombie.created_at = zombie.updated_at = datetime.now(tz=UTC) - timedelta(days=60)
    manifest.add(zombie)

    body = client.get("/api/runs").json()
    assert len(body["active"]) == 1
    assert len(body["lost"]) == 1 and body["lost"][0]["stale"] is True
    assert len(body["recent"]) == 3


def test_history_endpoint_returns_snapshots(client: TestClient) -> None:
    quota.add_snapshot("kaggle", 20.0, "год", 20.0,
                       ts=datetime.now(tz=UTC) - timedelta(days=1))
    quota.add_snapshot("kaggle", 18.0, "год", 18.0)
    points = client.get("/api/history/kaggle").json()["points"]
    assert [p["available"] for p in points] == [20.0, 18.0]


def test_index_page_is_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "T4-годин" in page.text
