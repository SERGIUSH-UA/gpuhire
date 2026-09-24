"""Наглядач не має права звітувати успіх, якого не було.

Кожен тест названий конкретною вадою, знайденою на аудиті 2026-08-11.
"""

from __future__ import annotations

import pathlib
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


def test_a_failed_fetch_does_not_condemn_pages_that_are_on_disk(tmp_path: Path) -> None:
    """🔴🔴 Дзеркальна вада до попередньої, і теж дорога.

    Лікування «збій забору → усі справи failed» поверталось ДО розкладання, а
    коли текст уже приїхав із R2, на диску лежали ПОВНІ справи. Заміряно
    23.09.2026 на черзі з 237 справ: 5115/5115 сторінок на диску, усі 237
    позначені `failed` із причиною «результат не забрано з боксу». Реєстр
    вважав їх непрочитаними, тож наступний захід заплатив би за ту саму роботу
    вдруге — при $0.233 за тисячу це $1.19 на рівному місці.

    Вирок ухвалює ЗВІРКА ПО ДИСКУ, а причина падіння лишається пояснювати
    лише те, чого диск не підтвердив.
    """
    from gpurunner.core.models import JobHandle

    plan = _plan(tmp_path)
    case = plan.cases[0]
    out = pathlib.Path(case.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, case.n_pages + 1):
        (out / f"{i:04d}.txt").write_text("текст", encoding="utf-8")

    sup = Supervisor(plan, backend=_Backend(), session="S")  # type: ignore[arg-type]
    sup._handle = JobHandle(backend="vast", remote_id="1", job_name="htr_case", gpu="any")
    sup._fetch_queue()
    sup._settle()

    one = sup.state.cases[0]
    assert one.complete, one.detail
    assert one.status == "done", f"диск каже повно, а вердикт {one.status}: {one.detail}"
    assert sup.state.verdict == "ok"
    assert any(i.kind == "fetch_failed" for i in sup.state.incidents),         "падіння забору мусить лишитись інцидентом, навіть коли робота ціла"


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


def _target_need(pages: int = 5000):
    from gpurunner.core.offer_score import Need
    return Need(pages=pages, max_hours=8.0, budget_usd=3.0, target_pph=5000.0)


class _Sel:
    """Вибір ринку для тестів очікування: порожній або з одним кандидатом."""

    def __init__(self, found: bool) -> None:
        self.empty = not found
        self.best = type("O", (), {"offer": {"cpu_cores_effective": 20.0}})() if found else None
        self.candidates = [self.best] if found else []
        self.rejected = []
        self.reason = "" if found else "жодна машина не дає цілі — найкраща на ринку: …"


def test_no_box_meets_the_target_so_it_waits_for_the_market(tmp_path: Path,
                                                             monkeypatch) -> None:
    """🔴🔴 Придатної машини немає — наглядач ЧЕКАЄ ринку, а не бере найкращу з
    поганих. Чекання безкоштовне: оренди ще немає.

    Доти після очікування йшло `settled_for_less` — і 23.09.2026 так узяли
    10-ядерну Q8000, що коштувала вдвічі дорожче за сторінку.
    """
    import gpurunner.supervise.htr as htr_mod

    plan = _plan(tmp_path)
    object.__setattr__(plan, "max_wait_min", 10.0)
    calls = {"n": 0}

    class _Market:
        def find_candidates(self, **kw):
            calls["n"] += 1
            return _Sel(found=calls["n"] >= 3)

    sup = Supervisor(plan, backend=_Market(), session="S")  # type: ignore[arg-type]
    monkeypatch.setattr(htr_mod.time, "sleep", lambda *_: None)
    sel = sup._find_with_patience(_target_need())
    assert calls["n"] == 3, "мусив перепитувати ринок, доки не з'явиться придатна"
    assert not sel.empty
    assert "settled_for_less" not in [i.kind for i in sup.state.incidents]


def test_after_the_wait_it_is_market_empty_not_a_weaker_box(tmp_path: Path,
                                                            monkeypatch) -> None:
    """Терпіння не безмежне — але по його кінці вибір ПОРОЖНІЙ, а не поступка."""
    import gpurunner.supervise.htr as htr_mod

    plan = _plan(tmp_path)
    object.__setattr__(plan, "max_wait_min", 0.0001)

    class _Market:
        def find_candidates(self, **kw):
            return _Sel(found=False)

    sup = Supervisor(plan, backend=_Market(), session="S")  # type: ignore[arg-type]
    monkeypatch.setattr(htr_mod.time, "sleep", lambda *_: None)
    sel = sup._find_with_patience(_target_need())
    assert sel.empty
    assert "цілі" in sel.reason


def test_without_a_target_there_is_nothing_to_wait_for(tmp_path: Path, monkeypatch) -> None:
    """Старий режим (`target_pph=0`) не чекає: один запит і що є."""
    import gpurunner.supervise.htr as htr_mod
    from gpurunner.core.offer_score import Need

    plan = _plan(tmp_path)
    object.__setattr__(plan, "max_wait_min", 60.0)
    calls = {"n": 0}

    class _Market:
        def find_candidates(self, **kw):
            calls["n"] += 1
            return _Sel(found=False)

    sup = Supervisor(plan, backend=_Market(), session="S")  # type: ignore[arg-type]
    monkeypatch.setattr(htr_mod.time, "sleep", lambda *_: None)
    sup._find_with_patience(Need(pages=283, max_hours=8.0, budget_usd=3.0))
    assert calls["n"] == 1


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


def test_request_above_what_fits_is_capped_out_loud(tmp_path: Path,
                                                     monkeypatch) -> None:
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
    res.measured = {}
    sup.backend.probe_box = lambda *a, **k: {}            # type: ignore[attr-defined]
    # Через monkeypatch: пряма підміна тікала в інші тести того ж воркера xdist.
    monkeypatch.setattr(gate_mod, "evaluate", lambda *a, **k: res)

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


def test_fleet_ceiling_lets_the_regulator_grow_past_the_start() -> None:
    """🔴🔴 spr-66 (23.09, RTX 3090, 61 ядро квоти, 24 ГБ): раннер діставав
    `shards_max = shards = 7` і стояв на 7 шардах при 6.2 з 24 ГБ карти —
    регулятор, що росте за виміряною пам'яттю, не мав куди рости."""
    from gpurunner.supervise.htr import fleet_ceiling

    measured = {"cores_eff": 61.44, "cores": 128.0, "n_gpus": 1,
                "vram_free_min_gb": 23.6, "vram_free_gb": 23.6}
    ceiling = fleet_ceiling(measured, 7)
    assert ceiling > 7
    assert ceiling <= int(61.44 // 1.25)
    # стеля — від КВОТИ, а не від `nproc` хоста (128)
    assert fleet_ceiling({**measured, "cores_eff": 12.0}, 4) <= 12 // 1.25 + 1


def test_fleet_ceiling_never_orders_the_fleet_to_shrink() -> None:
    from gpurunner.supervise.htr import fleet_ceiling

    assert fleet_ceiling({"cores_eff": 4.0, "n_gpus": 1, "vram_free_min_gb": 2.0}, 6) == 6


def _three_case_plan(tmp_path: Path) -> Plan:
    return Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case=f"spr-{n}", pages_url="https://r2/p.tar", n_pages=20,
                        out_dir=str(tmp_path / f"spr-{n}")) for n in (1, 2, 3)],
        budget_usd=1.0, max_hours=1.0,
    )


def test_catchup_progress_lands_on_the_right_case(tmp_path: Path) -> None:
    """🔴 На догоні бокс отримує лише недочитані справи, і його `case_index` —
    позиція в ЦІЙ черзі. За індексом у повному плані прогрес spr-3 ліг би на
    spr-1, а сторінки «позаду» рахувались би від чужих справ."""
    plan = _three_case_plan(tmp_path)
    sup = Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]
    sup._rent_queue = [plan.cases[2]]            # догін: лише spr-3
    sup._absorb_queue({"case": "spr-3", "case_index": 1, "pages_done": 7}, "тік")
    by_name = {c.case: c for c in sup.state.cases}
    assert by_name["spr-3"].pages_done == 7
    assert by_name["spr-1"].pages_done == 0
    assert sup._queue_pages_before({"case_index": 1}) == 0
    # Справи різного обсягу: на догоні [spr-2, spr-3] «позаду» у другої — spr-2.
    sized = Plan(assets_url="https://r2/a.tgz",
                 cases=[CasePlan(case=f"spr-{n}", pages_url="https://r2/p.tar",
                                 n_pages=10 * n, out_dir=str(tmp_path / f"s{n}"))
                        for n in (1, 2, 3)], budget_usd=1.0, max_hours=1.0)
    sup2 = Supervisor(sized, backend=object(), session="S2")  # type: ignore[arg-type]
    sup2._rent_queue = sized.cases[1:]
    assert sup2._queue_pages_before({"case_index": 2}) == 20, "позаду spr-2, а не spr-1"


def test_box_pace_survives_the_case_boundary(tmp_path: Path) -> None:
    """Посправний `pages_done` падає до нуля на кожній новій справі; темп
    бокса — ні, інакше правило обіцянки на черзі дрібних справ не дозріває."""
    plan = _three_case_plan(tmp_path)
    sup = Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]
    assert sup._box_pace({"case_index": 1, "pages_done": 1}, now=100.0) == (0, 0.0)
    pages, sec = sup._box_pace({"case_index": 1, "pages_done": 20}, now=200.0)
    assert (pages, sec) == (19, 100.0)
    pages, sec = sup._box_pace({"case_index": 2, "pages_done": 3}, now=260.0)
    assert (pages, sec) == (22, 160.0), "перехід на справу 2 не обнуляє сторінок бокса"


def test_an_old_plan_does_not_ship_a_48_mb_job(tmp_path: Path) -> None:
    """🔴🔴 23.09.2026: догін на старому плані віз 314 слотів чекпоінта на
    справу — 121 832 посилання, `job.py` 48 МБ, 25 хвилин заливки. Слоти
    обрізаються від розміру справи в наглядачі, а не лише в новому плані."""
    urls = [f"https://r2/ckpt/c/ckpt_{i:04d}.tgz?sig={'x' * 300}" for i in range(314)]
    plan = Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case="spr-1", pages_url="https://r2/p.tar", n_pages=24,
                        out_dir=str(tmp_path / "spr-1"), ckpt_urls=urls, resume_urls=urls)],
        budget_usd=1.0, max_hours=14.0,
    )
    sup = Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]
    got = sup._case_urls(plan.cases[0])
    assert 1 <= len(got["ckpt_urls"]) <= 20, len(got["ckpt_urls"])
    assert got["ckpt_urls"] == urls[: len(got["ckpt_urls"])], "перші слоти — ті самі"
    assert len(got["resume_urls"]) == len(got["ckpt_urls"])


def test_the_promise_counts_the_fixed_price_of_each_case() -> None:
    """🔴🔴 Темп бокса міряється разом із фіксованою ціною справи. Порівнювати
    його з ЧИСТИМ темпом читання — гасити кожну машину на черзі мікросправ:
    194 × 24 сторінки при 5 000 стор/год — 55 хв читання і 87 хв накладних."""
    from gpurunner.supervise.htr import queue_pace

    micro = [CasePlan(case=f"c{i}", pages_url="u", n_pages=24, out_dir="o")
             for i in range(194)]
    big = [CasePlan(case="b", pages_url="u", n_pages=4656, out_dir="o")]
    assert queue_pace(5000.0, micro) < 2500, "накладні 194 справ з'їдають більше половини"
    assert queue_pace(5000.0, big) > 4900


def test_a_new_box_starts_its_own_setup_clock(tmp_path: Path, monkeypatch) -> None:
    """B1 (24.09.2026): лог нової машини починається з нуля й не перевищував
    розміру логу попередньої, тож «сетап не посувається» рахувався від першої
    оренди — і кожну наступну машину гасили за хвилину після підйому."""
    import time as _time

    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="S")  # type: ignore[arg-type]

    class _Sftp:
        size = 0

        def stat(self, path):
            return type("S", (), {"st_size": self.size})()

    sftp = _Sftp()
    clock = [1000.0]
    monkeypatch.setattr(_time, "monotonic", lambda: clock[0])
    sftp.size = 50_000
    sup._note_setup_movement(sftp)
    clock[0] += 34 * 60                       # перша машина давно стоїть
    assert sup._setup_stall_sec() >= 34 * 60
    sup._close_rent()                         # її погасили, береться нова
    clock[0] += 60
    sftp.size = 1_000                         # лог нової машини — з нуля
    sup._note_setup_movement(sftp)
    assert sup._setup_stall_sec() < 60


def test_nothing_left_to_read_means_no_new_rental_and_an_ok_verdict(tmp_path: Path) -> None:
    """B1 (24.09.2026): забір після загибелі машини довіз усе, а порожній список
    недочитаного підмінявся всім планом — наглядач ішов орендувати на прочитане,
    упирався в стелю оренд і ставив `failed` при 7/7 повних справах."""
    sup = Supervisor(_plan(tmp_path), backend=_Backend(), session="S")  # type: ignore[arg-type]
    rented = []
    sup._rent_for = lambda *a, **k: rented.append(a) or True  # type: ignore[method-assign]
    for cs in sup.state.cases:
        cs.status, cs.complete = "done", True
    assert sup._rent_the_rest(None) is False
    assert rented == []
    sup._settle()
    assert sup.state.verdict == "ok"


def test_rent_pages_do_not_collapse_from_the_pipeline_to_a_big_case(tmp_path: Path) -> None:
    """spr-80 (24.09.2026): після конвеєра дрібних справ раннер узяв велику книгу
    з `case_index=1` і власним `pages_done` — сума «справи позаду + поточна»
    впала з ~1500 до нуля, і правило «гроші проти сторінок» погасило машину,
    що читала («$0.08 за 0 стор. за 34 хв»)."""
    from gpurunner.supervise.plan import CasePlan, Plan

    plan = Plan(assets_url="https://r2/a.tgz", budget_usd=1.0, max_hours=4.0, cases=[
        CasePlan(case="spr-80", pages_url="https://r2/80.tar", n_pages=1867,
                 out_dir=str(tmp_path / "80")),
        *[CasePlan(case=f"spr-{n}", pages_url=f"https://r2/{n}.tar", n_pages=200,
                   out_dir=str(tmp_path / str(n))) for n in range(89, 97)],
    ])
    sup = Supervisor(plan, backend=_Backend(), session="S")  # type: ignore[arg-type]
    sup._rent_queue = list(plan.cases)
    sup._current_rent_usd = lambda: 0.0  # type: ignore[method-assign]
    piped = {"pipeline": True, "case_index": 1, "pages_done": 1500,
             "pages_resumed": 100, "box_pages_done": 1400, "box_pages_resumed": 100}
    big = {"case": "spr-80", "case_index": 1, "pages_done": 60, "pages_resumed": 0,
           "box_pages_done": 1460, "box_pages_resumed": 100}
    sup._rent_money(piped, settled=True, now=0.0)          # база — у конвеєрі
    _usd, pages, _sec = sup._rent_money(big, settled=True, now=600.0)
    assert pages == 60, "лічильник оренди мусить іти далі, а не обвалитись"
    left = sup._pages_left_in_queue(big)
    assert left == 1867 + 8 * 200 - 1560


def test_the_vast_bill_raises_the_spend_and_the_hourly_price(tmp_path: Path, monkeypatch) -> None:
    """Рахунок Vast містить трафік і диск, яких немає в «ціна × час»: витрачене
    береться більшим із двох, а прогноз множить ціну за годину на їхнє
    відношення. Рахунок питається не частіше за раз на 15 хв."""
    import time as _time

    class _BillBackend(_Backend):
        calls = 0

        def instance_charges(self, start, end):
            _BillBackend.calls += 1
            return [{"instance": "52309609", "amount": 0.60},
                    {"instance": "99999999", "amount": 9.0}]

    sup = Supervisor(_plan(tmp_path), backend=_BillBackend(), session="S")  # type: ignore[arg-type]
    sup.state.instances = ["52309609"]
    sup.settled_usd = 0.40
    clock = [10_000.0]
    monkeypatch.setattr(_time, "monotonic", lambda: clock[0])
    sup._refresh_bill()
    sup._accrue()
    assert sup.spent_usd == 0.60, "витрачене — рахунок Vast, не чужі машини"
    assert abs(sup._bill_factor - 1.5) < 1e-9
    assert sup._budget_view()["billed_usd"] == 0.6
    clock[0] += 60
    sup._refresh_bill()
    assert _BillBackend.calls == 1, "рахунок не частіше за раз на 15 хв"
    clock[0] += 900
    sup._refresh_bill()
    assert _BillBackend.calls == 2
