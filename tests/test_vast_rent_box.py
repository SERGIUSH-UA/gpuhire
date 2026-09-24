"""`rent_box` / `destroy_box` / `box_info` — оренда без job'а.

Примітив для викликача, що працює на боксі сам (плагін Нишпорки). Наглядача в
цього шляху немає, тож усе, що береже гроші, мусить бути в самому примітиві:
таймер самознищення в onstart, гасіння інстансу при будь-якому збої після його
створення, незнімна стеля ціни.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from gpurunner.backends import vast as vast_mod
from gpurunner.backends.vast import ABSOLUTE_MAX_DPH, ForeignInstance, VastBackend
from gpurunner.core.backend import BackendError

OFFER = {"id": 111, "machine_id": 501, "host_id": 7501, "gpu_name": "RTX 3090",
         "num_gpus": 1, "gpu_ram": 24576.0, "dph_total": 0.21, "cpu_cores_effective": 32.0,
         "cpu_ram": 65536.0, "disk_space": 200.0, "geolocation": "Poland, PL"}


class _Client:
    closed = False

    def close(self) -> None:
        self.closed = True


class _Backend(VastBackend):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.row: dict[str, Any] | None = {
            "actual_status": "running", "ssh_host": "ssh4.vast.ai", "ssh_port": 40022,
            "label": "nysh-rent-abc", "gpu_name": "RTX 3090"}
        self.wait_raises: BaseException | None = None
        self.delete_404 = False

    def _api_key(self) -> str:
        return "test-key"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.requests.append((method, path, kwargs))
        if re.fullmatch(r"/asks/\d+/", path):
            return {"success": True, "new_contract": 424242}
        if re.fullmatch(r"/instances/\d+/", path) and method == "DELETE":
            if self.delete_404:
                err = BackendError("404 (fake)")
                err.status_code = 404
                raise err
            self.row = None
            return {"success": True}
        if re.fullmatch(r"/instances/\d+/", path):
            return {"instances": dict(self.row) if self.row else None}
        return {"success": True}

    def _wait_for_ssh(self, handle: Any, *, timeout: int = 900) -> Any:
        if self.wait_raises is not None:
            raise self.wait_raises
        return _Client()

    def calls(self, method: str) -> list[str]:
        return [p for m, p, _ in self.requests if m == method]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_CONFIG_DIR", str(tmp_path / "config"))
    priv = tmp_path / "id_ed25519"
    priv.write_text("PRIVATE", encoding="utf-8")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test", encoding="utf-8")
    monkeypatch.setenv("GPURUNNER_VAST_SSH_KEY", str(priv))


def test_rent_box_creates_attaches_key_waits_and_describes_the_box() -> None:
    bk = _Backend()
    seen: list[str] = []
    info = bk.rent_box(OFFER, disk_gb=50, label="nysh-rent-abc", autodestroy_hours=6,
                       on_created=lambda iid, offer: seen.append(iid))

    assert seen == ["424242"], "гроші пішли — викликач дізнається одразу"
    assert bk.calls("PUT") == ["/asks/111/"]
    assert "/instances/424242/ssh/" in bk.calls("POST")
    payload = bk.requests[0][2]["json"]
    assert payload["label"] == "nysh-rent-abc" and payload["disk"] == 50.0
    assert payload["image"] == vast_mod.DEFAULT_IMAGE
    assert payload["runtype"] == "ssh_direc ssh_proxy" and payload["num_gpus"] == 1

    assert {k: info[k] for k in ("instance_id", "machine_id", "host_id", "ssh_host",
                                 "ssh_port", "ssh_user", "gpu_name", "num_gpus")} == {
        "instance_id": "424242", "machine_id": 501, "host_id": 7501,
        "ssh_host": "ssh4.vast.ai", "ssh_port": 40022, "ssh_user": "root",
        "gpu_name": "RTX 3090", "num_gpus": 1}
    assert info["cores"] == 32.0 and info["ram_gb"] == 64.0 and info["vram_gb"] == 24.0
    assert info["disk_gb"] == 50.0 and info["dph_total"] == 0.21
    assert info["geolocation"] == "Poland, PL"
    assert bk.calls("DELETE") == []


def test_rent_onstart_is_only_the_self_destruct_timer() -> None:
    script = VastBackend()._render_rent_onstart(autodestroy_hours=6)
    assert "sleep 21600" in script
    assert "CONTAINER_API_KEY" in script and "instances/$CONTAINER_ID/" in script
    assert "-X DELETE" in script
    # job'а немає: ні сентинела, ні job.py, ні стоп-крана, який нема кому зняти
    assert "GO" not in script and "job.py" not in script and "NOSTOP" not in script
    assert not re.search(r"\{[a-z_]+\}", script), "шаблон відформатовано до кінця"
    assert "-d '{}'" in script, "тіло DELETE пережило `str.format`"


def test_rent_box_never_goes_without_the_timer() -> None:
    bk = _Backend()
    with pytest.raises(ValueError):
        bk.rent_box(OFFER, disk_gb=50, label="x", autodestroy_hours=0)
    assert bk.requests == []
    bk.rent_box(OFFER, disk_gb=50, label="x")            # None → дефолт, не «без таймера»
    assert "sleep 32400" in bk.requests[0][2]["json"]["onstart"]


def test_absolute_price_ceiling_holds_here_too() -> None:
    bk = _Backend()
    with pytest.raises(BackendError):
        bk.rent_box({**OFFER, "dph_total": ABSOLUTE_MAX_DPH + 0.01}, disk_gb=50, label="x")
    assert bk.requests == [], "🔴 дорожчий за стелю оффер не доходить до PUT"
    # стеля — на карту: двокартковий за подвійну ціну проходить
    bk.rent_box({**OFFER, "num_gpus": 2, "dph_total": 0.50}, disk_gb=50, label="x")


def test_failure_after_creation_destroys_the_instance() -> None:
    bk = _Backend()
    dead = BackendError("SSH не відповідає (fake)")
    dead.outcome = "ssh_unreachable"
    bk.wait_raises = dead
    with pytest.raises(BackendError) as exc:
        bk.rent_box(OFFER, disk_gb=50, label="x")
    assert bk.calls("DELETE") == ["/instances/424242/"]
    assert exc.value.outcome == "ssh_unreachable"
    assert exc.value.instance_id == "424242" and exc.value.destroyed is True


def test_ctrl_c_while_waiting_for_ssh_destroys_the_instance() -> None:
    """Очікування SSH триває хвилини — саме тоді людина й перериває."""
    bk = _Backend()
    bk.wait_raises = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        bk.rent_box(OFFER, disk_gb=50, label="x")
    assert bk.calls("DELETE") == ["/instances/424242/"]


def test_taken_offer_creates_nothing_and_destroys_nothing() -> None:
    bk = _Backend()

    def refuse(method: str, path: str, **kw: Any) -> dict[str, Any]:
        err = BackendError("400 no_such_ask")
        err.status_code = 400
        err.body = ""
        raise err

    bk._request = refuse  # type: ignore[method-assign]
    with pytest.raises(BackendError) as exc:
        bk.rent_box(OFFER, disk_gb=50, label="x")
    assert exc.value.outcome == "offer_taken"
    assert not getattr(exc.value, "instance_id", "")


def test_destroy_box_is_idempotent() -> None:
    bk = _Backend()
    bk.destroy_box(424242)
    assert bk.row is None
    bk.delete_404 = True
    bk.destroy_box(424242)                     # уже немає — не помилка
    bk.destroy_box("424242", label_prefix="nysh-rent")


def test_destroy_box_refuses_a_foreign_label() -> None:
    bk = _Backend()
    assert bk.row is not None
    bk.row["label"] = "gpurunner-htr_case-spr-7-deadbeef"
    with pytest.raises(ForeignInstance):
        bk.destroy_box(424242, label_prefix="nysh-rent")
    assert bk.calls("DELETE") == []


def test_box_info_none_for_a_missing_instance() -> None:
    bk = _Backend()
    assert bk.box_info(424242)["ssh_host"] == "ssh4.vast.ai"  # type: ignore[index]
    bk.row = None
    assert bk.box_info(424242) is None
