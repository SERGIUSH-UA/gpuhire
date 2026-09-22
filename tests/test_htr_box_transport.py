"""Доставка без бакета: план, передполіт і саме везіння — без мережі.

Перевіряється те, чим цей транспорт відрізняється від бакета, і тільки воно:

* план `box` НЕ торкається бакета взагалі — саме його відсутність і є причина,
  з якої транспорт існує;
* замість посилань, яких ще немає (їх видасть машина, якої поки немає), план
  несе шляхи на нашому диску, і передполіт перевіряє саме їх;
* доставка йде ПІСЛЯ воріт: везти гігабайти на машину, яку ми ще можемо
  забракувати, означало б платити заливкою за кожного невдалого кандидата;
* посилання, які дістає раннер, дивляться на петлю самої машини.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpurunner.htr import box_transport as bt
from gpurunner.htr import plan_build as pb


def _case(tmp_path: Path, name: str = "sprava", n: int = 3) -> Path:
    from PIL import Image

    case = tmp_path / name
    case.mkdir()
    for i in range(n):
        Image.new("L", (40, 60), 200).save(case / f"{i:04d}.jpg", "JPEG")
    return case


def _assets(tmp_path: Path) -> Path:
    import tarfile

    src = tmp_path / "runner.py"
    src.write_text("# раннер", encoding="utf-8")
    out = tmp_path / "assets_test.tgz"
    with tarfile.open(out, "w:gz") as tf:
        tf.add(src, arcname="scripts/htr_case_run.py")
    return out


def _opts(tmp_path: Path, **kw: Any) -> pb.BuildOptions:
    return pb.BuildOptions(out_root=(tmp_path / "out").resolve(),
                           model="model_v1.pt", transport="box",
                           staging_dir=tmp_path / "_box", **kw)


# ── план ─────────────────────────────────────────────────────────────────────
def test_box_plan_never_touches_the_bucket(tmp_path: Path, monkeypatch) -> None:
    """🔴 Відсутність бакета — це причина існування транспорту, а не дрібниця."""
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("план `box` пішов у бакет — саме цього й не можна")

    for name in ("client", "put", "get_url", "put_urls", "ckpt_get_urls"):
        monkeypatch.setattr(pb.r2, name, boom)

    case = _case(tmp_path)
    plan = pb.build_plan([case], _opts(tmp_path), assets=_assets(tmp_path),
                         out_path=tmp_path / "plan.json", log=lambda s: None)

    assert plan["transport"] == "box"
    assert plan["assets_url"] == "" and Path(plan["assets_path"]).is_file()
    entry = plan["cases"][0]
    assert entry["pages_url"] == "" and Path(entry["pages_path"]).is_file()
    assert entry["ckpt_urls"] == [] and entry["resume_urls"] == []
    assert entry["ckpt_slots"] > 0 and entry["ckpt_prefix"]


def test_box_plan_packs_frames_once(tmp_path: Path, monkeypatch) -> None:
    """Переоренда не пакує все вдруге: архів лежить і чекає машини."""
    monkeypatch.setattr(pb.r2, "client", lambda *a, **k: None)
    case = _case(tmp_path)
    opts = _opts(tmp_path)
    assets = _assets(tmp_path)

    first = pb.build_plan([case], opts, assets=assets, log=lambda s: None)
    archive = Path(first["cases"][0]["pages_path"])
    stamp = archive.stat().st_mtime_ns
    pb.build_plan([case], _opts(tmp_path), assets=assets, log=lambda s: None)

    assert archive.stat().st_mtime_ns == stamp


def test_box_refuses_assets_key_and_skip_upload(tmp_path: Path, monkeypatch) -> None:
    """Обидва прапорці кажуть «воно вже в бакеті» — тому, хто бакета не має."""
    monkeypatch.setattr(pb.r2, "client", lambda *a, **k: None)
    case = _case(tmp_path)

    with pytest.raises(ValueError, match="assets"):
        pb.build_plan([case], _opts(tmp_path), assets_key="assets/x.tgz",
                      log=lambda s: None)
    with pytest.raises(ValueError, match=r"skip-upload|бакет"):
        pb.build_plan([case], _opts(tmp_path, skip_upload=True),
                      assets=_assets(tmp_path), log=lambda s: None)


def test_unknown_transport_is_a_refusal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="транспорт"):
        pb.build_plan([_case(tmp_path)],
                      pb.BuildOptions(out_root=(tmp_path / "o").resolve(),
                                      model="m.pt", transport="карго"),
                      assets=_assets(tmp_path), log=lambda s: None)


# ── читання плану наглядачем ────────────────────────────────────────────────
def _written_plan(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(pb.r2, "client", lambda *a, **k: None)
    out = tmp_path / "plan.json"
    pb.build_plan([_case(tmp_path)], _opts(tmp_path), assets=_assets(tmp_path),
                  out_path=out, log=lambda s: None)
    return out


def test_supervisor_reads_the_box_plan(tmp_path: Path, monkeypatch) -> None:
    from gpurunner.supervise.plan import load_plan

    plan = load_plan(_written_plan(tmp_path, monkeypatch))
    assert plan.transport == "box"
    assert Path(plan.assets_path).is_file()
    assert Path(plan.cases[0].pages_path).is_file()
    assert plan.cases[0].ckpt_slots > 0


def test_missing_archive_is_caught_before_the_rent(tmp_path: Path, monkeypatch) -> None:
    """🔴 Ті самі слова, що й про порожнє посилання: без архіву машина
    підніметься, поставить рушій і впаде — уже на оплачуваній карті."""
    from gpurunner.supervise.plan import load_plan

    path = _written_plan(tmp_path, monkeypatch)
    raw = json.loads(path.read_text(encoding="utf-8"))
    Path(raw["cases"][0]["pages_path"]).unlink()
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="архів"):
        load_plan(path)


# ── передполіт ──────────────────────────────────────────────────────────────
def test_preflight_checks_the_disk_not_the_network(tmp_path: Path, monkeypatch) -> None:
    from gpurunner.htr import preflight

    monkeypatch.setattr(preflight, "probe", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("передполіт `box` пішов у мережу")))
    path = _written_plan(tmp_path, monkeypatch)

    report = preflight.check_plan(path)
    assert not report["problems"], report["problems"]
    assert report["assets"]["ok"]
    assert any("склад на самій машині" in n for n in report["notes"])
    assert any("чекпоінт" in n.lower() for n in report["notes"]), \
        "людині треба сказати, що чекпоінти лежать на машині, яка може вмерти"


def test_preflight_names_the_missing_file(tmp_path: Path, monkeypatch) -> None:
    from gpurunner.htr import preflight

    path = _written_plan(tmp_path, monkeypatch)
    raw = json.loads(path.read_text(encoding="utf-8"))
    Path(raw["assets_path"]).unlink()
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    report = preflight.check_plan(path)
    assert report["problems"] and "ассет" in report["problems"][0].lower()


# ── посилання складу ────────────────────────────────────────────────────────
def test_urls_point_at_the_loopback_of_the_machine() -> None:
    """🔴 Назовні склад не дивиться: машина стоїть у чужому дата-центрі."""
    urls = bt.urls_for("sprava", "ckpt/sprava/model_v1.pt", 3)

    assert urls["pages_url"] == f"{bt.base_url()}/cases/sprava.tar"
    assert urls["pages_url"].startswith("http://127.0.0.1:")
    assert len(urls["ckpt_urls"]) == 3
    assert urls["ckpt_urls"] == urls["resume_urls"], \
        "той самий шлях годиться і покласти, і взяти — напрям задає метод запиту"
    assert urls["ckpt_urls"][0].endswith("/ckpt/sprava/model_v1.pt/ckpt_0001.tgz")


def test_backend_without_ssh_is_told_to_use_the_bucket() -> None:
    from gpurunner.core.backend import BackendError

    with pytest.raises(BackendError, match="r2"):
        bt.endpoint_of(object(), handle=None)


def test_endpoint_needs_a_live_machine() -> None:
    from gpurunner.core.backend import BackendError

    class NotReady:
        def ssh_endpoint(self, handle: Any) -> dict[str, Any] | None:
            return None

    with pytest.raises(BackendError, match="SSH"):
        bt.endpoint_of(NotReady(), handle=None)


# ── посилання мусять бути ДО оренди ─────────────────────────────────────────
class _ParamsOnly:
    """Лише те, чого торкається складання параметрів роботи."""

    from gpurunner.supervise.htr import Supervisor

    _params_for = Supervisor._params_for
    _case_urls = Supervisor._case_urls
    _assets_url = Supervisor._assets_url
    _box_transport = Supervisor._box_transport
    _gb_per_shard = staticmethod(lambda: 0.0)
    _requested_shards = staticmethod(lambda: 0)


def test_box_urls_are_in_the_params_before_any_rent() -> None:
    """🔴 Параметри роботи перевіряються ДО оренди, і порожній `pages_url` там
    законно вважається обірваною змінною — захід падає ще до машини (так і
    сталось на першій пробі). Тому адреси складу підставляються одразу:
    порт і шляхи фіксовані, тож вони відомі наперед.
    """
    from gpurunner.core.offer_score import Need
    from gpurunner.jobs.htr_case import HTRCaseJob
    from gpurunner.supervise.plan import CasePlan, Plan

    case = CasePlan(case="sprava", pages_url="", n_pages=4, out_dir="/tmp/out",
                    pages_path="/home/sprava.tar", ckpt_prefix="ckpt/sprava/m.pt",
                    ckpt_slots=8)
    sup = _ParamsOnly()
    sup.plan = Plan(assets_url="", assets_path="/home/assets.tgz", transport="box",
                    cases=[case], budget_usd=1.0, max_hours=4.0)

    params = sup._params_for(case, Need(pages=4, max_hours=4.0, budget_usd=1.0),
                             resume=False, queue=[case])

    assert params["pages_url"] == f"{bt.base_url()}/cases/sprava.tar"
    assert params["assets_url"].endswith("/assets.tgz")
    assert len(params["ckpt_urls"]) == 8 and params["resume_urls"]
    assert params["cases"][0]["pages_url"] == params["pages_url"]
    # Та сама перевірка, яка впала на живій пробі, — тепер проходить.
    HTRCaseJob().validate_params(params)


def test_first_checkpoint_is_not_waited_five_minutes() -> None:
    """🔴 Порожня тека чекпоінтів у перші хвилини — штатний стан (раннер ще
    сегментує). З'їдений на ній п'ятихвилинний слот лишає найуразливіший
    відрізок заходу — початок — без жодної точки відновлення вдома."""
    from gpurunner.supervise import htr as htr_mod

    assert htr_mod.BOX_CKPT_FIRST_SEC < htr_mod.BOX_CKPT_SYNC_SEC
    src = Path(htr_mod.__file__).read_text(encoding="utf-8")
    i = src.index("period = (BOX_CKPT_SYNC_SEC")
    body = src[i:i + 400]
    assert "_ckpt_home_seen" in body, \
        "частота мусить залежати від того, чи приїхав хоч один чекпоінт"
    assert "if self.sync_box_checkpoints():" in body, \
        "успіх забору — це і є ознака, що можна питати рідше"


def test_home_checkpoints_are_keyed_by_case_not_session() -> None:
    """🔴 Точка відновлення потрібна саме тоді, коли попередній захід не дожив.

    Сесія щоразу нова, тож тека, названа по ній, зробила б привезені чекпоінти
    невидимими рівно для того заходу, який мав би ними скористатись, — і справа
    читалась би з нуля за повні гроші, хоч усе лежить на диску.
    """
    from gpurunner.supervise.htr import Supervisor
    from gpurunner.supervise.plan import CasePlan, Plan
    from gpurunner.supervise.state import SupervisorState

    class Fake:
        _box_ckpt_dir = Supervisor._box_ckpt_dir

    plan = Plan(assets_url="", transport="box", assets_path="/x/a.tgz",
                cases=[CasePlan(case="sprava", pages_url="", n_pages=1,
                                out_dir="/o", pages_path="/x/s.tar")],
                budget_usd=1.0, max_hours=4.0)
    first, second = Fake(), Fake()
    first.plan, first.state = plan, SupervisorState(session="htr-sprava-1111-aaaa")
    second.plan, second.state = plan, SupervisorState(session="htr-sprava-2222-bbbb")

    assert first._box_ckpt_dir() == second._box_ckpt_dir()
    assert "sprava" in first._box_ckpt_dir().name


def test_chunked_put_cannot_replace_a_checkpoint_with_nothing(tmp_path: Path) -> None:
    """🔴 Без `Content-Length` довжина читалась як нуль — і на місце цілого
    чекпоінта лягав порожній файл із кодом 201."""
    import http.client

    from gpurunner.htr import origin as origin_mod

    root = tmp_path / "store"
    root.mkdir()
    (root / "ckpt_0001.tgz").write_bytes(b"GOOD" * 100)
    httpd = origin_mod.serve(root, port=0)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1],
                                          timeout=10)
        conn.putrequest("PUT", "/ckpt_0001.tgz")
        conn.putheader("Transfer-Encoding", "chunked")
        conn.endheaders()
        conn.send(b"0\r\n\r\n")
        code = conn.getresponse().status
        conn.close()
    finally:
        httpd.shutdown()

    assert code == 411, "без довжини тіла склад мусить відмовити"
    assert (root / "ckpt_0001.tgz").read_bytes() == b"GOOD" * 100, \
        "цілий чекпоінт лишається на місці"


# ── забір чекпоінтів: поведінка, а не форма ─────────────────────────────────
class _SyncOnly:
    """Лише те, чого торкається забір чекпоінтів зі складу на машині."""

    from gpurunner.supervise.htr import Supervisor

    sync_box_checkpoints = Supervisor.sync_box_checkpoints
    _box_ckpt_dir = Supervisor._box_ckpt_dir
    _box_transport = Supervisor._box_transport


def _sync_rig(tmp_path: Path, monkeypatch, *, on_box: list[str]):
    """Наглядач із підробленими машиною й каналом: (об'єкт, журнал команд)."""
    import tarfile

    from gpurunner.supervise import htr as htr_mod
    from gpurunner.supervise.plan import CasePlan, Plan
    from gpurunner.supervise.state import SupervisorState

    home = tmp_path / "staging"
    monkeypatch.setattr(htr_mod, "_staging_root", lambda: home)
    said: list[str] = []

    class FakeClient:
        def close(self) -> None:
            pass

    class FakeBackend:
        def _ssh(self, handle: Any, timeout: int = 30) -> Any:
            return FakeClient()

    def fake_exec(client: Any, cmd: str) -> str:
        said.append(cmd)
        if "find ." in cmd:
            return "\n".join(on_box)
        return "yes"

    def fake_pull(ep: Any, remote: str, local: Path, **kw: Any) -> None:
        # Машина віддає рівно те, що ми в неї попросили запакувати.
        asked = [n for n in on_box if n in said[-1]]
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(local, "w:gz") as tf:
            for name in asked:
                blob = tmp_path / "blob"
                blob.write_bytes(b"ckpt")
                tf.add(blob, arcname=name)

    monkeypatch.setattr(bt, "endpoint_of", lambda b, h: object())
    monkeypatch.setattr(bt, "pull", fake_pull)

    sup = _SyncOnly()
    sup.backend = FakeBackend()
    sup._handle = object()
    sup._exec_on = staticmethod(fake_exec)
    sup.plan = Plan(assets_url="", transport="box", assets_path="/x/a.tgz",
                    cases=[CasePlan(case="sprava", pages_url="", n_pages=4,
                                    out_dir="/o", pages_path="/x/s.tar")],
                    budget_usd=1.0, max_hours=4.0)
    sup.state = SupervisorState(session="htr-sprava-0920-zzzz")
    sup._say = staticmethod(lambda _s: None)
    return sup, said


def test_checkpoints_come_home_and_only_the_new_ones_travel(
        tmp_path: Path, monkeypatch) -> None:
    """🔴 Чекпоінти накопичуються, і тар усієї теки щоп'ять хвилин означає, що
    ті самі мегабайти переїжджають додому знову й знову — а пакує їх машина, за
    яку платять погодинно."""
    box = ["sprava/model.pt/ckpt_0001.tgz", "sprava/model.pt/ckpt_0002.tgz"]
    sup, said = _sync_rig(tmp_path, monkeypatch, on_box=box)

    assert sup.sync_box_checkpoints() == 2
    home = sorted(p.name for p in sup._box_ckpt_dir().rglob("ckpt_*.tgz"))
    assert home == ["ckpt_0001.tgz", "ckpt_0002.tgz"], "обидва приїхали додому"

    # Другий захід: на машині з'явився третій — їде ЛИШЕ він.
    box.append("sprava/model.pt/ckpt_0003.tgz")
    said.clear()
    assert sup.sync_box_checkpoints() == 1
    packed = next(c for c in said if "tar czf" in c)
    assert "ckpt_0003.tgz" in packed
    assert "ckpt_0001.tgz" not in packed and "ckpt_0002.tgz" not in packed

    # Нічого нового — машину не турбуємо взагалі.
    said.clear()
    assert sup.sync_box_checkpoints() == 0
    assert not [c for c in said if "tar czf" in c]


def test_sync_failure_is_a_note_not_a_stop(tmp_path: Path, monkeypatch) -> None:
    """Робота на машині від невдалого забору не постраждала — спинити захід
    через це означало б викинути те, що вже прочитано."""
    sup, _ = _sync_rig(tmp_path, monkeypatch,
                       on_box=["sprava/model.pt/ckpt_0001.tgz"])
    monkeypatch.setattr(bt, "pull", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("канал обірвався")))

    assert sup.sync_box_checkpoints() == 0
    kinds = [i.kind for i in sup.state.incidents]
    assert "box_ckpt_sync_failed" in kinds


# ── згортання на прохання ───────────────────────────────────────────────────
def test_wrapup_request_is_seen_and_cleared(tmp_path: Path, monkeypatch) -> None:
    """🔴 Прохання кладеться туди, куди наглядач дивиться сам, і діє РАЗ:
    інакше наступний захід тієї ж сесії згорнувся б, не почавшись."""
    from gpurunner.supervise import state as state_mod
    from gpurunner.supervise import wrapup

    monkeypatch.setattr(state_mod, "state_dir", lambda: tmp_path)
    monkeypatch.setattr(wrapup, "state_dir", lambda: tmp_path)

    assert wrapup.requested("s1") is None
    wrapup.request("s1", why="людина передумала")
    asked = wrapup.requested("s1")
    assert asked is not None and "передумала" in asked["why"]

    wrapup.clear("s1")
    assert wrapup.requested("s1") is None


def test_wrapup_goes_the_same_way_as_a_money_ceiling() -> None:
    """🔴 Той самий шлях, яким захід завершується на стелі грошей: забір ДО
    гасіння, машина гаситься завжди, вердикт окремий від збою. Новий вихід із
    циклу означав би ще одне місце, де щось забувається."""
    from gpurunner.supervise import htr as htr_mod

    src = Path(htr_mod.__file__).read_text(encoding="utf-8")
    body = src[src.index("def _wrap_up"):src.index("def _quiesce_runner")]
    order = [body.index("_fetch_queue"), body.index("_destroy"),
             body.index("finish(")]
    assert order == sorted(order), "забір → гасіння → вирок, саме в такому порядку"
    assert "cancelled" in body, "згортання — не збій: текст на диску справжній"
    assert "wrapup_mod.clear" in body, "прохання прибирається, щоб не спрацювати двічі"


# ── канал до машини: гвард, якого проба воріт не дає ─────────────────────────
class _DeliveryOnly:
    """Лише замір швидкості доставки."""

    from gpurunner.supervise.htr import Supervisor

    _gate_delivery_speed = Supervisor._gate_delivery_speed


def _delivery_rig(*, max_hours: float):
    from gpurunner.supervise.plan import CasePlan, Plan
    from gpurunner.supervise.state import SupervisorState

    sup = _DeliveryOnly()
    sup.plan = Plan(assets_url="", transport="box", assets_path="/x/a.tgz",
                    cases=[CasePlan(case="sprava", pages_url="", n_pages=100,
                                    out_dir="/o", pages_path="/x/pages.tar")],
                    budget_usd=1.0, max_hours=max_hours)
    sup.state = SupervisorState(session="s")
    return sup


def test_slow_uplink_rejects_the_machine_before_the_hours_are_paid() -> None:
    """🔴 Проба каналу воріт міряє шлях ХОСТА до бакета, а при складі на машині
    веземо ми — тож єдине число про наш аплінк дає перший же архів. Без нього
    8 ГБ на 0.5 МБ/с — це 4.5 год, за які машина не прочитає жодної сторінки."""
    from gpurunner.core.backend import BackendError

    sup = _delivery_rig(max_hours=4.0)

    # 100 МБ за 200 с = 0.5 МБ/с; справа на 8 ГБ їхала б ~4.4 год при стелі 4.
    with pytest.raises(BackendError, match="іншу машину"):
        sup._gate_delivery_speed(100_000_000, 200.0, left_bytes=8_000_000_000)
    assert any(i.kind == "delivery_rate" for i in sup.state.incidents), \
        "замір мусить лишитись у журналі, навіть коли машину забраковано"


def test_fast_uplink_passes_and_is_recorded() -> None:
    sup = _delivery_rig(max_hours=4.0)

    # ~7.7 МБ/с — заміряна норма; ті самі 8 ГБ доїдуть за ~17 хв.
    sup._gate_delivery_speed(100_000_000, 13.0, left_bytes=8_000_000_000)

    note = next(i for i in sup.state.incidents if i.kind == "delivery_rate")
    assert "МБ/с" in note.detail
