"""Tests for `gpurunner balance`.

The point of the command is honesty: report a number where the provider actually
has an API for it, and say so plainly (with a link) where it does not — instead of
printing a confident zero.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpurunner.backends import get_backend
from gpurunner.backends.kaggle import KaggleBackend
from gpurunner.backends.lightning import LightningBackend
from gpurunner.backends.vast import VastBackend
from gpurunner.core.backend import AuthError, BackendError
from gpurunner.core.models import BalanceReport


@pytest.fixture(autouse=True)
def _isolated_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_CONFIG_DIR", str(tmp_path / "config"))


def test_colab_has_no_balance_api_and_says_so_with_a_link() -> None:
    rep = get_backend("colab")().balance()
    assert isinstance(rep, BalanceReport)
    assert rep.available is None           # never fake a number
    assert rep.url.startswith("https://")  # but always point somewhere useful
    assert rep.detail


def test_kaggle_balance_reports_ai_quota_and_flags_that_gpu_quota_is_unavailable() -> None:
    """kagglesdk exposes the model-proxy daily quota (USD) but NOTHING for the
    weekly GPU quota — the report must not let those two be confused."""

    class _Resp:
        daily_quota_used = 1.5
        total_daily_quota_allowed = 10.0

    class _Client:
        # a real class, not SimpleNamespace: `with` looks up __enter__ on the TYPE
        def __enter__(self) -> Any:
            import types

            return types.SimpleNamespace(
                benchmarks=types.SimpleNamespace(
                    benchmark_tasks_api_client=types.SimpleNamespace(
                        get_benchmark_task_quota=lambda req: _Resp()
                    )
                )
            )

        def __exit__(self, *exc: Any) -> bool:
            return False

    class _Bk(KaggleBackend):
        def _get_api(self) -> Any:
            import types

            return types.SimpleNamespace(build_kaggle_client=_Client)

    rep = _Bk().balance()
    assert rep.available == 8.5
    assert rep.spent == 1.5
    assert "НЕ GPU" in rep.detail          # labelled, so it can't be misread
    assert "GPU-квота" in rep.detail       # and the real gap is stated


def test_kaggle_balance_degrades_gracefully_when_the_quota_call_fails() -> None:
    class _Bk(KaggleBackend):
        def _get_api(self) -> Any:
            raise RuntimeError("no creds")

    rep = _Bk().balance()
    assert rep.available is None
    assert "GPU-квота" in rep.detail       # still tells you where to look


def test_modal_balance_reports_month_to_date_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Modal has no 'remaining credit' endpoint, but the billing report works even
    on a personal workspace — spend-so-far is the actionable number."""
    import sys
    import types
    from decimal import Decimal

    fake = types.ModuleType("modal.billing")
    fake.workspace_billing_report = lambda **kw: [  # type: ignore[attr-defined]
        {"description": "gpurunner-yolo_spotter", "cost": Decimal("20.00")},
        {"description": "gpurunner-yolo_spotter", "cost": Decimal("9.13")},
        {"description": "something-else", "cost": Decimal("1.00")},
    ]
    monkeypatch.setitem(sys.modules, "modal.billing", fake)

    monkeypatch.setenv("GPURUNNER_MODAL_MONTHLY_CREDIT", "30")
    rep = get_backend("modal")().balance()
    assert rep.spent == pytest.approx(30.13)
    assert "gpurunner-yolo_spotter $29.13" in rep.detail
    # spend passed the free allowance → 0 left, and the row must SAY it is an estimate
    assert rep.available == pytest.approx(0.0)
    assert "вичерпано" in rep.detail


def test_modal_remaining_is_an_estimate_from_the_monthly_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types
    from decimal import Decimal

    fake = types.ModuleType("modal.billing")
    fake.workspace_billing_report = lambda **kw: [{"description": "app", "cost": Decimal("4.00")}]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modal.billing", fake)
    monkeypatch.setenv("GPURUNNER_MODAL_MONTHLY_CREDIT", "30")

    rep = get_backend("modal")().balance()
    assert rep.available == pytest.approx(26.0)
    assert "ОЦІНКА" in rep.detail          # never presented as an exact balance


def test_modal_monthly_allowance_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Team plan gets $100/month — the constant must not be hardcoded."""
    import sys
    import types
    from decimal import Decimal

    fake = types.ModuleType("modal.billing")
    fake.workspace_billing_report = lambda **kw: [{"description": "app", "cost": Decimal("10.00")}]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modal.billing", fake)
    monkeypatch.setenv("GPURUNNER_MODAL_MONTHLY_CREDIT", "100")

    assert get_backend("modal")().balance().available == pytest.approx(90.0)


def test_vast_balance_reads_credit_and_warns_about_live_burn() -> None:
    class _Bk(VastBackend):
        def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/users/current":
                return {"credit": 12.5}
            return {
                "instances": [
                    {"actual_status": "running", "dph_total": 0.21},
                    {"actual_status": "exited", "dph_total": 9.99},  # must not count
                ]
            }

    rep = _Bk().balance()
    assert rep.available == 12.5
    assert rep.unit == "$"
    assert "0.210" in rep.detail  # the live burn rate is surfaced, not hidden


def test_vast_balance_survives_an_instance_listing_failure() -> None:
    class _Bk(VastBackend):
        def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/users/current":
                return {"credit": 3.0}
            raise BackendError("listing down")

    rep = _Bk().balance()
    assert rep.available == 3.0  # the balance itself still comes back


def test_lightning_balance_prefers_the_project_credits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Free-tier credits live on the *project* balance; the user balance is 0.0 on a
    free account, so reading that one would report 'no credits left' every time."""
    import types

    class _Resp:
        def __init__(self, d: dict[str, Any]) -> None:
            self._d = d

        def to_dict(self) -> dict[str, Any]:
            return self._d

    class _Client:
        def __init__(self, retry: bool = False) -> None:
            pass

        def billing_service_get_project_balance(self, project_id: str) -> _Resp:
            return _Resp({"balance": 14.8, "project_id": project_id})

        def billing_service_get_user_balance(self) -> _Resp:
            return _Resp({"balance": 0.0, "total_spent": 0.21})

    fake_module = types.ModuleType("lightning_sdk.lightning_cloud.rest_client")
    fake_module.LightningClient = _Client  # type: ignore[attr-defined]
    import sys

    monkeypatch.setitem(sys.modules, "lightning_sdk.lightning_cloud.rest_client", fake_module)

    from gpurunner.auth import lightning as lit_auth

    monkeypatch.setattr(lit_auth, "import_sdk", lambda: types.ModuleType("lightning_sdk"))

    class _Bk(LightningBackend):
        def _teamspace(self, name: str | None = None) -> Any:
            return types.SimpleNamespace(id="proj-1", name="general")

    rep = _Bk().balance()
    assert rep.available == 14.8
    assert rep.spent == 0.21
    assert rep.unit == "credits"
    assert "general" in rep.detail


def test_missing_credentials_surface_as_autherror_not_a_crash() -> None:
    """The CLI turns this into a 'not configured' row; the important part is that
    the backend raises AuthError rather than returning a bogus zero."""
    with pytest.raises(AuthError):
        VastBackend().balance()
