"""Наглядач не має права звітувати успіх, якого не було.

Кожен тест названий конкретною вадою, знайденою на аудиті 2026-08-11.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpurunner.core.backend import BackendError
from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import CasePlan, Plan


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.delenv("GPURUNNER_OWNER", raising=False)


def _need(pages: int):
    """Потреба заходу — стільки, скільки треба тесту про ЯДРА."""
    from gpurunner.core.offer_score import Need
    return Need(pages=pages, max_hours=8.0, budget_usd=3.0)


def _plan(tmp_path: Path) -> Plan:
    return Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case="spr-114", pages_url="https://r2/p.tar", n_pages=128,
                        out_dir=str(tmp_path / "spr-114"))],
        budget_usd=1.0, max_hours=1.0,
    )


class _Backend:
    def __init__(self, fail_fetch: bool = True) -> None:
        self.fail_fetch = fail_fetch

        self.cancelled = []

    def fetch_outputs(self, handle, dest):
        if self.fail_fetch:
            raise BackendError("ssh помер саме на заборі")

    def cancel(self, handle):
        self.cancelled.append(handle)


def test_failed_fetch_is_never_reported_as_success(tmp_path: Path) -> None:
    """🔴🔴 Найгірша вада наглядача: збій забору давав `ok` і код 0.

    `_fetch_queue` виходив ДО циклу звірки, справи лишались `running`, і
    `_settle` (що рахує лише done/incomplete/failed) падав у гілку `else`.
    Бокс на той момент уже знищено — тобто при НУЛІ сторінок на диску агент,
    який за контрактом читає лише цей JSON, звітував людині успіх.
    """
    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="S")  # type: ignore[arg-type]
    from gpurunner.core.models import JobHandle
    sup._handle = JobHandle(backend="vast", remote_id="1", job_name="htr_case", gpu="any")
    sup._fetch_queue()
    sup._settle()

    assert sup.state.verdict != "ok"
    assert sup.state.exit_code != 0
    assert all(c.status != "done" for c in sup.state.cases)


def test_nothing_done_is_not_success(tmp_path: Path) -> None:
    """Захід без жодної завершеної справи — не успіх, навіть якщо ніхто не скаржився."""
    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="S")  # type: ignore[arg-type]
    sup._settle()
    assert sup.state.verdict == "incomplete"
    assert sup.state.human_action_required


def test_failed_rent_money_survives_the_next_tick(tmp_path: Path) -> None:
    """🔴 Гроші невдалих оренд стирались першим же `_accrue()`.

    `_rent_for` додавав вартість у `spent_usd`, а `_accrue` присвоює
    `settled_usd + поточна оренда` — і серія згорілих спроб коштувала реальних
    доларів, яких бюджетний запобіжник не бачив.
    """
    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="S")  # type: ignore[arg-type]
    sup.settled_usd += 0.11   # так тепер робить гілка невдалої оренди
    sup._accrue()
    assert sup.spent_usd == pytest.approx(0.11)
    sup._accrue()
    assert sup.spent_usd == pytest.approx(0.11)


# ---- другий раунд аудиту: регресії, внесені виправленнями першого ------------


def test_queue_progress_does_not_inflate_pages_left(tmp_path: Path) -> None:
    """🔴 «Уже пройдено» рахувалось за статусом `done`, який виставляється лише
    ПІСЛЯ всієї черги — тобто весь захід дорівнював нулю, і до залишку
    додавались усі вже прочитані справи. На черзі з п'яти це давало вчетверо
    завищений прогноз і `destroy_budget` за хвилини до кінця оплаченої роботи.
    """
    plan = Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case=f"c{i}", pages_url="https://r2/p.tar", n_pages=3000,
                        out_dir=str(tmp_path / f"c{i}")) for i in range(1, 6)],
        budget_usd=3.0, max_hours=8.0,
    )
    sup = Supervisor(plan, backend=_Backend(), session="S")  # type: ignore[arg-type]
    # Іде четверта справа — три позаду, хоч жодна ще не забрана з боксу.
    assert sup._queue_pages_before({"case_index": 4}) == 9000
    assert sup._queue_pages_before({"case_index": 1}) == 0
    assert sup._queue_pages_before(None) == 0


def test_unkillable_box_verdict_is_not_overwritten(tmp_path: Path) -> None:
    """🔴 `_destroy` чесно ставив «негайно погасити руками», а гілка, що його
    покликала, одразу робила свій `finish("failed")` — і `human_action_required`
    падав у **false**. Агент розгалужується рівно на цьому булевому, тож він
    мовчав, поки оренда горіла до дедлайн-кілера.
    """
    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="S")  # type: ignore[arg-type]
    sup.state.finish("incomplete", "бокс НЕ погашено", human_action="gpurunner cancel …",
                     sticky=True)
    sup.state.finish("failed", "щось інше")

    assert sup.state.verdict == "incomplete"
    assert sup.state.human_action_required is True
    assert "cancel" in (sup.state.human_action or "")


def test_sticky_flag_never_leaks_into_the_contract(tmp_path: Path) -> None:
    """Внутрішній прапорець не має з'являтись у JSON, який читає агент."""
    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="S")  # type: ignore[arg-type]
    sup.state.finish("incomplete", "x", sticky=True)
    assert "_sticky" not in sup.state.to_dict()


def test_checkpoint_alarm_does_not_fire_on_the_second_case(tmp_path: Path) -> None:
    """🔴 Хибна тривога `no_recovery_point` при робочих чекпоінтах.

    Лічильник `ok` раннер створює ПОСПРАВНО, а вік мірявся від ОРЕНДИ — тож на
    другій справі черги (коли від оренди вже минуло пів години, а лічильник
    свіжий і нульовий) тривога спрацьовувала гарантовано. Виміряно 2026-08-12:
    п'ять справ уже було закрито, чекпоінти писались, а наглядач кричав, що
    точок відновлення немає.
    """
    import time as _t

    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="S")  # type: ignore[arg-type]
    sup._rented_at = _t.monotonic() - 4000        # оренда давно

    sup._watch_checkpoints({"ckpt": {"ok": 3, "last_ok": "2026-08-12T10:00:00"}})
    assert not sup.state.incidents                # справа 1: усе добре

    sup._watch_checkpoints({"ckpt": {"ok": 0, "last_ok": None}})   # справа 2 почалась
    assert not sup.state.incidents, "лічильник обнулився — це не привід кричати"


def test_checkpoint_alarm_still_fires_when_nothing_ever_worked(tmp_path: Path) -> None:
    """Але справжню відсутність точок відновлення мусить ловити."""
    import time as _t

    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="S")  # type: ignore[arg-type]
    sup._rented_at = _t.monotonic() - 4000
    sup._watch_checkpoints({"ckpt": {"ok": 0, "last_ok": None}})
    assert [i.kind for i in sup.state.incidents] == ["no_recovery_point"]


def test_waits_for_a_faster_box_before_settling(tmp_path: Path, monkeypatch) -> None:
    """🔴 Ядра — головна ручка швидкості, а ринок багатоядерних машин тонкий:
    під стелею $0.365/год їх було рівно ТРИ, і одна вже працювала на нас.

    Але ринок оновлюється щохвилини, а чекання БЕЗКОШТОВНЕ — оренди ще немає.
    Тож замість «беремо, що дають» наглядач деякий час перепитує.
    """
    import gpurunner.supervise.htr as htr_mod

    plan = _plan(tmp_path)
    object.__setattr__(plan, "prefer_min_cores", 64.0)
    object.__setattr__(plan, "wait_for_cores_min", 1.0)

    calls = {"n": 0}

    class _Sel:
        def __init__(self, cores: float) -> None:
            self.best = type("O", (), {"offer": {"cpu_cores_effective": cores}})()
            self.empty = False

    class _Market:
        def find_candidates(self, **kw):
            calls["n"] += 1
            return _Sel(32.0 if calls["n"] < 3 else 64.0)

    sup = Supervisor(plan, backend=_Market(), session="S")  # type: ignore[arg-type]
    monkeypatch.setattr(htr_mod.time, "sleep", lambda *_: None)

    # 🔴 Обсяг тут не декорація: планка ядер масштабується від роботи, і
    # чекати на 64-ядерну машину має сенс лише під велику чергу.
    sel = sup._find_with_patience(_need(pages=5000))
    assert calls["n"] >= 3, "мусив перепитати ринок, а не взяти першу-ліпшу"
    assert sel.best.offer["cpu_cores_effective"] == 64.0


def test_settling_for_less_is_said_out_loud(tmp_path: Path, monkeypatch) -> None:
    """Терпіння не безмежне — але поступка мусить бути названа, а не тиха."""
    import gpurunner.supervise.htr as htr_mod

    plan = _plan(tmp_path)
    object.__setattr__(plan, "prefer_min_cores", 64.0)
    object.__setattr__(plan, "wait_for_cores_min", 0.001)

    class _Sel:
        def __init__(self) -> None:
            self.best = type("O", (), {"offer": {"cpu_cores_effective": 32.0}})()
            self.empty = False

    class _Market:
        def find_candidates(self, **kw):
            return _Sel()

    sup = Supervisor(plan, backend=_Market(), session="S")  # type: ignore[arg-type]
    monkeypatch.setattr(htr_mod.time, "sleep", lambda *_: None)
    sup._find_with_patience(_need(pages=5000))
    assert [i.kind for i in sup.state.incidents] == ["settled_for_less"]


def test_a_short_queue_does_not_wait_an_hour_for_64_cores(tmp_path: Path, monkeypatch) -> None:
    """🔴 План, зроблений під велику чергу, ніс `prefer_min_cores: 64`. На
    добивці 283 сторінок наглядач через це ГОДИНУ стояв у циклі «найкраще на
    ринку — 24 ядер, хочемо від 64», не орендуючи нічого (31.08.2026).
    Формально працював, фактично не рухався: при $0.08/год 24-ядерний бокс
    дорахує за ніч і коштуватиме менше за годину очікування."""
    import gpurunner.supervise.htr as htr_mod

    plan = _plan(tmp_path)
    object.__setattr__(plan, "prefer_min_cores", 64.0)
    object.__setattr__(plan, "wait_for_cores_min", 60.0)

    calls = {"n": 0}

    class _Sel:
        def __init__(self) -> None:
            self.best = type("O", (), {"offer": {"cpu_cores_effective": 24.0}})()
            self.empty = False

    class _Market:
        def find_candidates(self, **kw):
            calls["n"] += 1
            return _Sel()

    sup = Supervisor(plan, backend=_Market(), session="S")  # type: ignore[arg-type]
    monkeypatch.setattr(htr_mod.time, "sleep", lambda *_: None)
    sup._find_with_patience(_need(pages=283))

    assert calls["n"] == 1, "на 283 сторінках 24 ядра доречні — чекати нема чого"
    assert not sup.state.incidents


def test_adopting_a_slow_box_is_said_out_loud(tmp_path: Path, monkeypatch) -> None:
    """🔴 «Підхопив» ≠ «схвалив».

    Наглядач брав БУДЬ-ЯКИЙ живий бокс і більше не питав, чи він вартий
    роботи: перезапуск на 32-ядерній машині мовчки продовжував платити за
    повільність, хоч ринок міг давати вдвічі більше ядер. Гасити чужу вже
    зроблену роботу не можна — але сказати з числами треба.
    """
    from gpurunner.core.models import JobHandle, JobStatus

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-slow")
    plan = _plan(tmp_path)
    object.__setattr__(plan, "prefer_min_cores", 64.0)

    h = JobHandle(backend="vast", remote_id="777", job_name="htr_case", gpu="any")
    h.status = JobStatus.RUNNING
    from gpurunner.core import manifest

    manifest.add(h)

    class _Backend:
        def _instance(self, handle):
            return {"actual_status": "running", "cpu_cores_effective": 32,
                    "dph_total": 0.2, "machine_id": 4711}

    sup = Supervisor(plan, backend=_Backend(), session="S")  # type: ignore[arg-type]
    assert sup._adopt_live_box() is True
    kinds = [i.kind for i in sup.state.incidents]
    assert kinds == ["adopted", "adopted_slow"]
    assert "32" in sup.state.incidents[1].detail


def test_checkpoints_are_used_on_a_fresh_campaign_too(tmp_path: Path) -> None:
    """🔴🔴 Відновлення не працювало на практиці НІКОЛИ.

    `resume_urls` передавались лише коли `resume=True`, а цей прапорець
    ставиться тільки при переоренді ВСЕРЕДИНІ того самого заходу. Новий захід
    ігнорував чекпоінти повністю — і саме він буває після смерті наглядача,
    тобто рівно тоді, коли відновлення й потрібне. Плюс `htr_cloud_plan.py`
    завжди робить ЧЕРГУ, а дірка була саме в гілці черги.

    Ціна одного такого перезапуску: 769 сторінок наново, ~45 хв, ~$0.13 — при
    тому, що передполітна перевірка чесно казала «точки відновлення на місці».
    """
    plan = Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case="c1", pages_url="https://r2/p.tar", n_pages=100,
                        out_dir=str(tmp_path / "c1"),
                        resume_urls=["https://r2/ckpt_0001.tgz"],
                        ckpt_urls=["https://r2/put_0001"])],
        budget_usd=1.0, max_hours=1.0,
    )
    sup = Supervisor(plan, backend=_Backend(), session="S")  # type: ignore[arg-type]
    need = sup._need_for(100)

    fresh = sup._params_for(plan.cases[0], need, resume=False, queue=plan.cases)
    assert fresh["resume_urls"] == ["https://r2/ckpt_0001.tgz"]
    assert fresh["cases"][0]["resume_urls"] == ["https://r2/ckpt_0001.tgz"], (
        "гілка ЧЕРГИ викидала посилання — а черга це бойовий режим"
    )


def test_box_destroys_itself_after_the_job_when_nobody_comes(tmp_path: Path) -> None:
    """🔴 Робота ЗРОБЛЕНА, наглядач мертвий — бокс мусить закритися сам.

    `_AUTODESTROY` стоїть у onstart одразу після команди job'а, тобто це саме
    сторож «job скінчився, по результат ніхто не прийшов». Але наглядач ніколи
    не передавав `autodestroy_hours`, тож блок не встановлювався взагалі, і
    єдиним запобіжником лишався жорсткий дедлайн `max_hours + 30хв`.

    Ціна 2026-08-12: наглядач помер у фазі забору, черга вже дорахувалась, а
    бокс горів іще 4.44 год — $0.93 намарно.
    """
    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="S")  # type: ignore[arg-type]
    params = sup._params_for(sup.plan.cases[0], sup._need_for(100), resume=False)
    assert params["autodestroy_hours"] == 0.5


def _handle_running(cases: list[str]):
    from gpurunner.core.models import JobHandle, JobStatus

    h = JobHandle(backend="vast", remote_id="900", job_name="htr_case", gpu="any",
                  params={"cases": [{"case": c} for c in cases]})
    h.status = JobStatus.RUNNING
    return h


def test_never_adopts_a_box_running_someone_elses_cases(tmp_path: Path, monkeypatch) -> None:
    """🔴🔴 Спільного власника НЕ ДОСИТЬ, щоб вважати бокс осиротілим.

    Дві половини одної кампанії часто йдуть під одним `GPURUNNER_OWNER`. Тоді
    наглядач A бачив ЖИВИЙ бокс наглядача B, вважав його покинутим і
    «підхоплював»: забрав би результат і погасив машину, поки B на ній ще
    рахує. Ледь не сталось 2026-08-12 (кампанія htr-olhopil) — спинили руками
    за півхвилини.
    """
    from gpurunner.core import manifest

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-olhopil")
    theirs = _handle_running(["spov1876-315-12011"])   # справи ІНШОЇ половини
    manifest.add(theirs)

    class _Backend:
        def _instance(self, handle):
            return {"actual_status": "running", "cpu_cores_effective": 64, "dph_total": 0.14}

    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="A")  # type: ignore[arg-type]
    assert sup._adopt_live_box() is False
    assert [i.kind for i in sup.state.incidents] == ["not_adopting"]


def test_never_adopts_a_case_held_by_a_live_session(tmp_path: Path, monkeypatch) -> None:
    """Навіть на СВОЇХ справах: живий замок означає живого наглядача."""
    from gpurunner.core import locks, manifest

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-olhopil")
    mine = _handle_running(["spr-114"])
    manifest.add(mine)
    locks.acquire("case:spr-114", owner="htr-olhopil", session="B")   # B ще працює

    class _Backend:
        def _instance(self, handle):
            return {"actual_status": "running", "cpu_cores_effective": 64, "dph_total": 0.14}

    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="A")  # type: ignore[arg-type]
    assert sup._adopt_live_box() is False


def test_still_adopts_a_genuinely_orphaned_box(tmp_path: Path, monkeypatch) -> None:
    """А справді покинутий бокс мусить підхоплюватись — заради цього все й є."""
    from gpurunner.core import manifest

    monkeypatch.setenv("GPURUNNER_OWNER", "htr-solo")
    mine = _handle_running(["spr-114"])
    manifest.add(mine)

    class _Backend:
        def _instance(self, handle):
            return {"actual_status": "running", "cpu_cores_effective": 64, "dph_total": 0.14}

    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="A")  # type: ignore[arg-type]
    assert sup._adopt_live_box() is True


def test_vram_per_shard_knob_actually_reaches_the_gate(tmp_path: Path) -> None:
    """🔴🔴 Ручка `vram_gb_per_shard` не діяла НІКОЛИ.

    Вона доходила до раннера, але той її не читав: число шардів приходило вже
    готовим (`_auto_shards` викликається лише при `shards < 1`), а рахували
    його ворота — зі СВОГО `need.gb_per_shard`, куди параметр не потрапляв.
    Тобто ручка міняла те, чим раннер рахував би шарди САМ, а сам він ніколи
    не рахує.

    Ціна на сповідках 2026-08-12: попросили 2.8 ГБ на шард, дістали 8 шардів
    (з дефолтних 2.5), споживання 1.5-4.3 ГБ, сумарно 23.3 з 24.6 ГБ —
    **2045 збоїв проти 549 готових сторінок**.
    """
    plan = Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case="spov", pages_url="https://r2/p.tar", n_pages=100,
                        out_dir=str(tmp_path / "spov"))],
        budget_usd=1.0, max_hours=1.0,
        params={"vram_gb_per_shard": 2.8},
    )
    sup = Supervisor(plan, backend=_Backend(), session="S")  # type: ignore[arg-type]

    assert sup._gb_per_shard() == 2.8
    assert sup._need_for(100).gb_per_shard == 2.8, "ворота мусять рахувати шарди цим числом"
    assert sup._params_for(plan.cases[0], sup._need_for(100),
                           resume=False)["vram_gb_per_shard"] == 2.8


def test_per_case_vram_knob_is_honoured_too(tmp_path: Path) -> None:
    """Задане на СПРАВІ теж має діяти — сповідки щільніші за метрики."""
    plan = Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case="spov", pages_url="https://r2/p.tar", n_pages=100,
                        out_dir=str(tmp_path / "spov"),
                        params={"vram_gb_per_shard": 4.5})],
        budget_usd=1.0, max_hours=1.0,
    )
    sup = Supervisor(plan, backend=_Backend(), session="S")  # type: ignore[arg-type]
    assert sup._need_for(100).gb_per_shard == 4.5


def test_without_the_knob_the_default_still_applies(tmp_path: Path) -> None:
    from gpurunner.core.htr_sizing import GB_PER_SHARD

    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="S")  # type: ignore[arg-type]
    # 06.09.2026: дефолт більше не мовчазний — без ручки й без геометрії
    # наглядач сам називає 3.3 і везе це число на бокс, щоб раннер не рахував
    # флот від власної копії сталої
    assert sup._gb_per_shard() == GB_PER_SHARD
    assert sup._need_for(100).gb_per_shard == GB_PER_SHARD


def _plan_with(params: dict, tmp_path: Path) -> Plan:
    return Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case="spov", pages_url="https://r2/p.tar", n_pages=100,
                        out_dir=str(tmp_path / "spov"))],
        budget_usd=1.0, max_hours=1.0, params=params,
    )


def test_explicit_shards_are_not_overridden_by_the_gate(tmp_path: Path) -> None:
    """🔴🔴 Ворота перекривали ЯВНО ЗАДАНЕ число шардів.

    Вони повертали своє число безумовно, тож `-p shards=5` доходив до раннера
    й там-таки затирався пробою боксу: на сповідках просили 5, дістали 8,
    VRAM 23.0 з 24.6 ГБ — і 84% сторінок у збоях. Проба знає про залізо, але
    НЕ знає про щільність справи; дослідник, який уже бачив пік 4.3 ГБ на
    шард, знає більше.
    """
    sup = Supervisor(_plan_with({"shards": 5}, tmp_path),
                     backend=_Backend(), session="S")  # type: ignore[arg-type]
    assert sup._requested_shards() == 5


def test_auto_is_not_a_request(tmp_path: Path) -> None:
    """`shards: "auto"` означає «вирішуй сам» — це не прохання про число."""
    sup = Supervisor(_plan_with({"shards": "auto"}, tmp_path),
                     backend=_Backend(), session="S")  # type: ignore[arg-type]
    assert sup._requested_shards() == 0


def test_request_above_what_fits_is_capped_out_loud(tmp_path: Path) -> None:
    """Стеля VRAM лишається за нами — але про обрізання треба сказати."""
    from gpurunner.core.htr_sizing import plan_sizing
    from gpurunner.supervise import gate as gate_mod

    sizing = plan_sizing(cores=28, vram_gb=22.5)          # тримає 8
    sup = Supervisor(_plan_with({"shards": 99}, tmp_path),
                     backend=_Backend(), session="S")  # type: ignore[arg-type]

    class _Res:
        ok = True
        detail = ""

    res = _Res()
    res.sizing = sizing
    sup.backend.probe_box = lambda *a, **k: {}            # type: ignore[attr-defined]
    gate_mod.evaluate = lambda *a, **k: res               # type: ignore[assignment]

    out = sup._gate(object(), None, type("C", (), {"offer": {}})(), sup._need_for(100))
    assert out["shards"] == sizing.shards
    assert [i.kind for i in sup.state.incidents] == ["shards_capped"]


def test_unset_price_ceiling_never_crashes_the_supervisor(tmp_path: Path) -> None:
    """🔴🔴 `float > None` валив наглядача в найгіршу мить.

    `Plan.max_usd_per_1000_pages` за замовчуванням None («стелю не задавали»),
    і воно передавалось просто так у `Cfg`, затираючи дефолт 0.20. Перевірка
    ціни падала TypeError'ом рівно тоді, коли вимір дозрів і треба було
    ухвалювати рішення — тобто захід гинув не від параметрів, а від коду.
    Заміряно 2026-08-12 на кліровій кампанії: результат урятовано з 28
    чекпоінтів, оренду погашено, але вердикт `failed` замість нової машини.
    """
    from gpurunner.supervise.decide import Cfg, Obs, decide

    plan = Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case="c", pages_url="https://r2/p.tar", n_pages=10,
                        out_dir=str(tmp_path / "c"))],
        budget_usd=1.0, max_hours=1.0,
    )
    assert plan.max_usd_per_1000_pages is None
    sup = Supervisor(plan, backend=_Backend(), session="S")  # type: ignore[arg-type]
    assert sup.cfg.max_usd_per_1000 == Cfg(budget_usd=1.0, max_hours=1.0).max_usd_per_1000

    obs = Obs(
        progress={"phase": "running", "n_pages_expected": 1000, "pages_done": 900,
                  "pages_per_hour": 800, "wall_sec": 3600.0, "missing_count": 0},
        spent_usd=0.1, dph=0.2,
    )
    decide(obs, sup.cfg)          # не має кидати


def test_explicit_price_ceiling_is_honoured(tmp_path: Path) -> None:
    """Задану стелю беремо як задано — сповідки об'єктивно дорожчі за сторінку."""
    plan = Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case="c", pages_url="https://r2/p.tar", n_pages=10,
                        out_dir=str(tmp_path / "c"))],
        budget_usd=1.0, max_hours=1.0, max_usd_per_1000_pages=0.35,
    )
    sup = Supervisor(plan, backend=_Backend(), session="S")  # type: ignore[arg-type]
    assert sup.cfg.max_usd_per_1000 == 0.35
