"""Плагін `nyshporka.cloud`: gpurunner орендує й гасить бокс для Нишпорки.

Без мережі й без грошей: REST Vast підмінено крихітним фейком (той самий
прийом, що в `test_vast_backend.py`), SSH не відкривається взагалі. Справжнім
лишається все, заради чого плагін існує: перевірка ключа й балансу, скоринг
офферів, реєстр боксів, гасіння інстансу при збої, форма `Box.meta`.

Більшість тестів працює і БЕЗ Нишпорки (у CI gpurunner її немає): типи
контракту беруться через `plugin.contract()` — справжні, якщо пакет є, і
двійники, якщо ні. Лише звірка з самим протоколом і `connect` її вимагають.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpurunner.backends.vast import VastBackend
from gpurunner.core import boxes
from gpurunner.core.backend import BackendError
from gpurunner.core.htr_sizing import LINES_PER_PAGE_ASSUMED
from gpurunner.plugins import nyshporka_cloud as plugin


def _offer(offer_id: int, machine_id: int, *, dph: float = 0.15, cores: float = 32.0,
           gpu: str = "RTX 3090") -> dict[str, Any]:
    return {"id": offer_id, "machine_id": machine_id, "host_id": 7000 + machine_id,
            "gpu_name": gpu, "num_gpus": 1, "gpu_ram": 24576.0, "dph_total": dph,
            "cpu_cores_effective": cores, "cpu_ram": 65536.0, "disk_space": 200.0,
            "reliability2": 0.99, "inet_down": 500.0, "geolocation": "Poland, PL"}


class _FakeClient:
    def close(self) -> None:
        pass


class _FakeVast(VastBackend):
    """Vast без мережі: ринок, баланс, інстанси — у пам'яті."""

    def __init__(self) -> None:
        super().__init__()
        self.offers: list[dict[str, Any]] = [_offer(111, 501), _offer(222, 502, dph=0.18)]
        self.credit: float | None = 12.5
        self.instances: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []
        self.created: list[dict[str, Any]] = []
        self.ssh_fails_on: set[int] = set()       # offer id → SSH не піднявся
        self.taken: set[int] = set()              # offer id → 400 «зайняли»
        self.requests: list[tuple[str, str]] = []
        self._next = 900000

    def _api_key(self) -> str:
        return "test-key"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.requests.append((method, path))
        if path == "/users/current":
            return {} if self.credit is None else {"credit": self.credit}
        if path == "/instances/" and method == "GET":
            return {"instances": list(self.instances.values())}
        if path == "/bundles/":
            want = (kwargs.get("json") or {}).get("machine_id") or {}
            rows = list(self.offers)
            if "in" in want:
                rows = [o for o in rows if o["machine_id"] in want["in"]]
            return {"offers": rows}
        if m := re.fullmatch(r"/asks/(\d+)/", path):
            offer_id = int(m.group(1))
            if offer_id in self.taken:
                err = BackendError("Vast.ai PUT failed: 400 — no_such_ask")
                err.status_code = 400
                err.body = '{"error": "no_such_ask"}'
                raise err
            self._next += 1
            iid = str(self._next)
            payload = kwargs["json"]
            self.created.append({"offer_id": offer_id, "instance_id": iid, **payload})
            self.instances[iid] = {
                "id": int(iid), "label": payload["label"], "actual_status": "running",
                "ssh_host": "ssh9.vast.ai", "ssh_port": 41234, "gpu_name": "RTX 3090",
                "num_gpus": 1, "dph_total": 0.15, "machine_id": 501, "host_id": 7501,
                "cpu_cores_effective": 32.0, "cpu_ram": 65536.0, "gpu_ram": 24576.0,
                "disk_space": 60.0, "geolocation": "Poland, PL",
                "start_date": (datetime.now(tz=UTC) - timedelta(minutes=30)).timestamp(),
                "_offer": offer_id,
            }
            return {"success": True, "new_contract": int(iid)}
        if m := re.fullmatch(r"/instances/(\d+)/", path):
            iid = m.group(1)
            if method == "DELETE":
                self.deleted.append(iid)
                self.instances.pop(iid, None)
                return {"success": True}
            row = self.instances.get(iid)
            return {"instances": dict(row) if row else None}
        if path.endswith("/ssh/"):
            return {"success": True}
        return {}

    def _wait_for_ssh(self, handle: Any, *, timeout: int = 900) -> Any:
        row = self.instances.get(str(handle.remote_id)) or {}
        if row.get("_offer") in self.ssh_fails_on:
            dead = BackendError("контейнер працює, але SSH не відповідає (fake)")
            dead.outcome = "ssh_unreachable"
            raise dead
        return _FakeClient()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ні справжнього ключа API, ні справжнього `~/.ssh`, ні справжнього реєстру."""
    from gpurunner.auth import vast as vast_auth

    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("VAST_API_KEY", "test-key")
    monkeypatch.setattr(vast_auth, "_candidate_key_paths",
                        lambda: [tmp_path / "config" / "vast_api_key"])
    priv = tmp_path / "id_ed25519"
    priv.write_text("PRIVATE", encoding="utf-8")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test", encoding="utf-8")
    monkeypatch.setenv("GPURUNNER_VAST_SSH_KEY", str(priv))
    monkeypatch.delenv(plugin.IMAGE_ENV, raising=False)


@pytest.fixture
def fake() -> _FakeVast:
    return _FakeVast()


@pytest.fixture
def rent(fake: _FakeVast) -> plugin.VastRent:
    backend = plugin.VastRent(backend_factory=lambda: fake)
    backend.retry_sleep_sec = 0.0
    return backend


def _need(**over: Any) -> Any:
    base = {"pages": 1200, "bytes_in": 3_000_000_000, "gb_per_shard": 3.3, "disk_gb": 0,
            "max_hours": None, "budget_usd": None, "max_price_usd_h": None,
            "prefer_cores": 0,
            # Ці тести — про механіку оренди, а не про ціль темпу: фейкові
            # оффери (32 ядра, 24 ГБ) цілі 5 000 не дають, і вибір правильно
            # їх відкидає. Ціль на шляху плагіна — окремим тестом нижче.
            "target_pph": 0}
    base.update(over)
    return SimpleNamespace(**base)


# ---- acquire ---------------------------------------------------------------


def test_acquire_without_api_key_says_what_to_do(rent, fake, monkeypatch) -> None:
    monkeypatch.delenv("VAST_API_KEY")
    with pytest.raises(plugin.contract().AuthError) as exc:
        rent.acquire(_need())
    text = str(exc.value)
    assert "nysh cloud rent login" in text
    assert "gpurunner auth vast --key" in text
    assert fake.requests == [], "без ключа до Vast не ходимо взагалі"


def test_acquire_with_empty_balance_asks_to_top_up(rent, fake) -> None:
    fake.credit = 0.12
    with pytest.raises(plugin.contract().CloudError) as exc:
        rent.acquire(_need())
    assert "поповніть баланс Vast" in str(exc.value)
    assert "$0.12" in str(exc.value)
    assert fake.created == [], "🔴 при порожньому балансі інстанс не створюється"
    assert ("POST", "/bundles/") not in fake.requests, "і ринок не питаємо"


def test_unknown_balance_does_not_block_the_rent(rent, fake) -> None:
    """Невідомий баланс ≠ нульовий: вигадувати число не можна, спиняти — теж."""
    fake.credit = None
    assert rent.acquire(_need()).id


def test_acquire_on_empty_market_gives_the_reason(rent, fake) -> None:
    fake.offers = []
    with pytest.raises(plugin.contract().CloudError) as exc:
        rent.acquire(_need())
    assert "немає придатної машини" in str(exc.value)
    assert "бюджет і строк" in str(exc.value), "причина — із `selection.reason`"
    assert fake.created == []


def test_successful_rent_returns_a_box_the_ssh_backend_can_use(rent, fake, tmp_path) -> None:
    before = datetime.now(tz=UTC)
    box = rent.acquire(_need(max_hours=5.0))

    assert box.backend == "vast" and box.id == fake.created[0]["instance_id"]
    assert box.price_usd_h == pytest.approx(0.15)
    assert box.cores == 32.0 and box.vram_gb == pytest.approx(24.0) and box.gpus == 1
    assert "RTX 3090" in box.label and "$0.150/год" in box.label and "Poland" in box.label

    host = box.meta["host"]
    assert host["user"] == "root" and host["host"] == "ssh9.vast.ai" and host["port"] == 41234
    assert host["key"] == str(tmp_path / "id_ed25519"), "шлях ПРИВАТНОГО ключа, не вміст"
    assert host["workdir"] == "/workspace/nysh-run" and host["python"] == "python3"

    rented = datetime.fromisoformat(box.meta["rented_at"])
    kill_at = datetime.fromisoformat(box.meta["autodestroy_at"])
    assert rented.tzinfo is not None and rented >= before - timedelta(seconds=1)
    # max_hours 5 + година на забір і звірку
    assert box.meta["autodestroy_hours"] == 6.0
    assert timedelta(hours=5.9) < kill_at - rented < timedelta(hours=6.1)
    assert box.meta["machine_id"] == 501 and box.meta["offer_id"] == 111
    assert box.meta["gpu_name"] == "RTX 3090" and box.meta["geolocation"] == "Poland, PL"

    created = fake.created[0]
    assert created["label"].startswith("nysh-rent-")
    assert "sleep 21600" in created["onstart"], "таймер самознищення = 6 год"
    assert "GO" not in created["onstart"], "job'а немає — чекати сентинела нема чого"
    assert box.as_dict()["meta"]["host"]["port"] == 41234, "їде у стан заходу як є"


def test_default_hours_when_need_has_no_ceiling(rent, fake) -> None:
    box = rent.acquire(_need())
    assert box.meta["autodestroy_hours"] == 9.0, "(не задано → 8) + 1"


def test_instance_is_destroyed_when_the_box_never_comes_up(rent, fake) -> None:
    """🔴 Гроші: інстанс, до якого не пустив SSH, гаситься ДО наступної спроби."""
    fake.ssh_fails_on = {111}
    box = rent.acquire(_need())

    first = fake.created[0]["instance_id"]
    assert first in fake.deleted and first not in fake.instances
    assert box.id == fake.created[1]["instance_id"], "узято наступного кандидата"
    rows = boxes.read_all()
    assert [(r.machine_id, r.outcome) for r in rows] == [(501, "ssh_unreachable")]
    assert rows[0].run_id == f"nysh-{first}"


def test_rents_are_capped(rent, fake) -> None:
    fake.offers = [_offer(100 + i, 600 + i) for i in range(6)]
    fake.ssh_fails_on = {o["id"] for o in fake.offers}
    with pytest.raises(plugin.contract().CloudError) as exc:
        rent.acquire(_need())
    assert len(fake.created) == rent.max_rents == 3
    assert sorted(fake.deleted) == sorted(c["instance_id"] for c in fake.created)
    assert "створено й погашено інстансів: 3" in str(exc.value)


def test_a_taken_offer_is_the_market_not_the_machine(rent, fake) -> None:
    fake.taken = {111}
    box = rent.acquire(_need())
    assert box.meta["offer_id"] == 222
    assert boxes.read_all() == [], "перехоплений оффер у реєстр не пишеться"


def test_banned_machine_is_not_rented(rent, fake) -> None:
    boxes.record(boxes.BoxObservation(machine_id=501, outcome="cpu_lie", detail="4 з 32"))
    box = rent.acquire(_need())
    assert box.meta["machine_id"] == 502 and fake.created[0]["offer_id"] == 222


def test_target_picks_the_offer_or_the_machine(rent, fake) -> None:
    assert rent.acquire(_need(), target="222").meta["offer_id"] == 222
    assert rent.acquire(_need(), target="machine:502").meta["offer_id"] == 222
    with pytest.raises(plugin.contract().CloudError):
        rent.acquire(_need(), target="333")
    with pytest.raises(plugin.contract().CloudError):
        rent.acquire(_need(), target="my-home-server")


# ---- release / find --------------------------------------------------------


def test_release_twice_is_quiet_and_writes_one_observation(rent, fake) -> None:
    box = rent.acquire(_need())
    rent.release(box, why="ok pph=1840")
    rent.release(box, why="ok pph=1840")      # `finally` кличе вдруге — це норма

    assert box.id not in fake.instances
    rows = boxes.read_all()
    assert len(rows) == 1
    assert rows[0].outcome == "ok" and rows[0].machine_id == 501
    assert rows[0].measured["pages_per_hour"] == 1840
    assert rows[0].claimed["dph_total"] == pytest.approx(0.15)


def test_release_of_a_dead_instance_does_not_raise(rent, fake) -> None:
    box = rent.acquire(_need())
    fake.instances.clear()                     # хост знищив сам
    rent.release(box, why="збій під час підготовки")
    assert boxes.read_all() == []


def test_failed_release_writes_a_bad_observation(rent, fake) -> None:
    box = rent.acquire(_need())
    old = (datetime.now(tz=UTC) - timedelta(hours=2)).isoformat()
    box.meta["rented_at"] = old
    rent.release(box, why="failed: 40 сторінок із 1200, решта в OOM")

    (row,) = boxes.read_all()
    assert row.verdict == "bad" and row.outcome == "setup_failed"
    assert "40 сторінок" in row.detail
    assert 7100 <= row.billed_sec <= 7300
    assert row.cost_usd == pytest.approx(0.30, abs=0.01), "2 год × $0.15"


def test_slow_and_neutral_release(rent, fake) -> None:
    a = rent.acquire(_need())
    rent.release(a, why="slow: 300 стор/год замість 1500")
    b = rent.acquire(_need())
    rent.release(b, why="захід завершено")
    assert [r.outcome for r in boxes.read_all()] == ["slow_run", "user_stop"]
    assert boxes.read_all()[1].verdict == "neutral"


def test_release_refuses_a_foreign_instance(rent, fake) -> None:
    box = rent.acquire(_need())
    fake.instances[box.id]["label"] = "gpurunner-htr_case-spr-12-ab12cd34"
    with pytest.raises(plugin.contract().CloudError):
        rent.release(box, why="ok")
    assert fake.deleted == [], "чужий бокс на тому самому акаунті не гаситься"


def test_find_reads_the_live_api(rent, fake) -> None:
    box = rent.acquire(_need())
    found = rent.find(box.id)
    assert found is not None and found.id == box.id
    assert found.meta["host"]["host"] == "ssh9.vast.ai"
    assert found.price_usd_h == pytest.approx(0.15)

    rent.release(box, why="ok")
    assert rent.find(box.id) is None, "зниклий інстанс — None, а не виняток"
    assert rent.find("user@host:22") is None


# ---- status / login / estimate ---------------------------------------------


def test_status_never_rents(rent, fake) -> None:
    fake.instances["555"] = {"id": 555, "label": "gpurunner-x", "dph_total": 0.2,
                             "actual_status": "running", "gpu_name": "V100"}
    st = rent.status()
    assert st["ready"] is True and st["problems"] == []
    assert st["api_key"] is True and st["ssh_key"].endswith("id_ed25519")
    assert st["balance_usd"] == 12.5
    assert st["burning"] == [{"instance_id": "555", "dph_total": 0.2, "label": "gpurunner-x",
                              "gpu_name": "V100", "status": "running"}]
    assert not any(m == "PUT" for m, _ in fake.requests)


def test_status_without_key_does_not_invent_a_balance(rent, fake, monkeypatch) -> None:
    monkeypatch.delenv("VAST_API_KEY")
    st = rent.status()
    assert st["ready"] is False and st["api_key"] is False
    assert st["balance_usd"] is None and st["burning"] is None
    assert fake.requests == []


def test_login_saves_the_key_and_never_echoes_it(rent, fake, monkeypatch, tmp_path) -> None:
    from gpurunner.auth import vast as vast_auth

    monkeypatch.delenv("VAST_API_KEY")
    monkeypatch.setattr(vast_auth, "verify", lambda: {"email": "x", "balance": 12.5})
    st = rent.login("  sekret-key-123  ")
    saved = tmp_path / "config" / "vast_api_key"
    assert saved.read_text(encoding="utf-8").strip() == "sekret-key-123"
    assert st["ready"] is True and st["api_key"] is True
    assert "sekret-key-123" not in repr(st)


def test_estimate_on_an_empty_market_has_no_numbers(rent, fake) -> None:
    fake.offers = []
    est = rent.estimate(_need())
    assert est["empty"] is True and est["candidates"] == 0 and est["reason"]
    assert est["balance_usd"] == 12.5
    for key in ("gpu", "price_usd_h", "pages_per_hour", "hours", "cost_usd", "usd_per_1000"):
        assert key not in est, f"🔴 {key}: на порожньому ринку числа взяти нема з чого"


def test_estimate_gives_the_dry_run_numbers_without_renting(rent, fake) -> None:
    est = rent.estimate(_need())
    assert est["empty"] is False and est["candidates"] == 2
    assert est["gpu"] == "RTX 3090" and est["num_gpus"] == 1 and est["cores"] == 32.0
    assert est["price_usd_h"] == pytest.approx(0.15)
    assert est["pages_per_hour"] > 0 and est["hours"] > 0
    assert est["usd_per_1000"] == pytest.approx(
        1000 * est["price_usd_h"] / est["pages_per_hour"], rel=0.01)
    assert fake.created == []


# ---- форма плагіна ---------------------------------------------------------


def test_module_works_without_nyshporka(rent, fake, monkeypatch) -> None:
    """У CI gpurunner Нишпорки немає; локально її відсутність імітуємо. Оренда й
    гасіння мусять працювати на двійниках контракту, а `connect` — чесно сказати,
    що транспорт веде Нишпорка."""
    import sys

    # Усі три: підмодуль, уже завантажений сусіднім тестом, імпортується повз
    # батьківський пакет — і `connect` пішов би справжнім SSH у мережу.
    for name in ("nyshporka.cloud", "nyshporka.cloud.base", "nyshporka.cloud.ssh"):
        monkeypatch.setitem(sys.modules, name, None)
    c = plugin.contract()
    assert c.Box is plugin._Box and issubclass(c.AuthError, c.CloudError)

    box = rent.acquire(_need())
    assert isinstance(box, plugin._Box) and box.as_dict()["meta"]["host"]["user"] == "root"
    with pytest.raises(c.CloudError):
        rent.connect(box)
    rent.release(box, why="ok")
    assert box.id in fake.deleted


def test_plugin_has_the_shape_the_registry_checks() -> None:
    """Те саме, що `nyshporka.cloud.registry._from_entry_points`: фабрика без
    аргументів, непорожній `id`, чотири callable."""
    backend = plugin.VastRent()
    assert backend.id == "vast" and backend.label
    assert backend.caps == frozenset({"rent", "cancel", "market"})
    for name in ("acquire", "connect", "release", "find", "status", "login", "estimate"):
        assert callable(getattr(backend, name, None)), name


def test_entry_point_is_declared() -> None:
    from importlib.metadata import entry_points

    eps = {e.name: e.value for e in entry_points(group="nyshporka.cloud")}
    assert eps.get("vast") == "gpurunner.plugins.nyshporka_cloud:VastRent"


def test_plugin_satisfies_the_real_protocol() -> None:
    pytest.importorskip("nyshporka")
    from nyshporka.cloud.base import CloudBackend, bills

    backend = plugin.VastRent()
    assert isinstance(backend, CloudBackend)
    assert bills(backend), "`rent` у caps → `release` для конвеєра обов'язковий"


def test_real_registry_loads_the_plugin() -> None:
    pytest.importorskip("nyshporka")
    from nyshporka.cloud import registry

    reg = registry.load()
    assert "vast" in reg.backends and isinstance(reg.backends["vast"], plugin.VastRent)
    assert not [b for b in reg.broken if b[0] == "vast"]


def test_not_ready_is_not_gone(rent, fake, monkeypatch) -> None:
    """🔴 Плутанина цих двох станів коштувала чотирьох оренд поспіль."""
    pytest.importorskip("nyshporka")
    pytest.importorskip("paramiko")
    from nyshporka.cloud import ssh as nysh_ssh
    from nyshporka.cloud.base import BoxGone, BoxNotReady

    def refuse(self: Any, box: Any) -> Any:
        raise BoxNotReady("ssh9.vast.ai не відповідає (fake)")

    monkeypatch.setattr(nysh_ssh.SshBackend, "connect", refuse)
    box = rent.acquire(_need())

    with pytest.raises(BoxNotReady):           # інстанс живий — просто чекати
        rent.connect(box)

    fake.instances.clear()
    with pytest.raises(BoxNotReady):           # свіжий: в API його ще може не бути
        rent.connect(box)

    box.meta["rented_at"] = (datetime.now(tz=UTC) - timedelta(hours=1)).isoformat()
    with pytest.raises(BoxGone):
        rent.connect(box)


def test_page_density_reaches_the_scoring_and_is_echoed_only_then() -> None:
    """Вилку кошторису споживач звужує лише за відлунням: воно означає, що
    щільність справді пішла в прогноз темпу, а не просто була відома комусь."""
    from types import SimpleNamespace

    from gpurunner.plugins.nyshporka_cloud import VastRent

    base = {"pages": 100, "bytes_in": 10_000_000, "gb_per_shard": 3.3, "disk_gb": 0,
            "max_hours": 4.0, "budget_usd": 1.0, "max_price_usd_h": None}
    dense, _ = VastRent._score_need(SimpleNamespace(**base, lines_per_page=164))
    unknown, _ = VastRent._score_need(SimpleNamespace(**base))
    assert dense.lines_per_page == 164.0 and not dense.lines_assumed
    # Невідома густина — ПРИПУЩЕНА для вибору, і так і позначена: відлуння
    # споживачеві йде лише з виміряної.
    assert unknown.lines_assumed
    assert unknown.lines_per_page == pytest.approx(LINES_PER_PAGE_ASSUMED)


def test_the_plugin_path_also_refuses_a_box_that_misses_the_target(rent, fake) -> None:
    """🔴 Ціль темпу діє і на шляху Нишпорки, не лише в наглядачі.

    Фейковий ринок — 32 ядра / 24 ГБ: на такому залізі модель обіцяє близько
    2 200 стор/год, тобто цілі 5 000 машина не дає. Без явного `target_pph`
    плагін бере дефолтну ціль і НЕ орендує — порожньо з причиною, а не слабкий
    захід.
    """
    with pytest.raises(Exception) as err:
        rent.acquire(_need(target_pph=None))
    assert "цілі" in str(err.value)
    assert not fake.created
