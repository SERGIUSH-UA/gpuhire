"""Наскрізний прогін наглядача на підробленому бекенді — без жодної оренди.

Перевіряється не арифметика (для неї є окремі тести), а ПОРЯДОК дій. Саме
порядок і був зламаний: бокс гасили до звірки повноти, а неповний результат
рахували успіхом.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from pathlib import Path
from typing import Any

import pytest

from gpurunner.core.backend import BackendError
from gpurunner.core.htr_sizing import plan_sizing
from gpurunner.core.models import JobHandle
from gpurunner.core.offer_score import ScoredOffer, Selection
from gpurunner.supervise import plan as plan_mod
from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import CasePlan, Plan

OFFER = {
    "id": 1, "machine_id": 38902, "gpu_name": "Tesla V100", "num_gpus": 1,
    "cpu_cores_effective": 64.0, "gpu_ram": 32 * 1024, "cpu_ram": 128 * 1024,
    "disk_space": 200.0, "dph_total": 0.222, "reliability2": 0.99, "inet_down": 500.0,
}

HEALTHY_PROBE = {
    "cores": 64.0, "gpu": "Tesla V100-SXM2-32GB",
    "vram_total_gb": 31.7, "vram_free_gb": 31.4,
    "disk_free_gb": 190.0, "net_mbps": 400.0, "net_bps": 50_000_000.0,
}


class FakeBackend:
    """Бекенд, який нічого не орендує, але веде журнал того, що з ним робили."""

    def __init__(self, *, pages: int, texts: int, complete: bool = True,
                 probe: dict | None = None, credit: float | None = 50.0) -> None:
        self.calls: list[str] = []
        self.pages = pages
        self.texts = texts
        self.complete = complete
        self.probe = probe or HEALTHY_PROBE
        self.gate_raised: Exception | None = None
        self.credit = credit
        self.submit_errors: list[Exception] = []
        #: Скільки кандидатів віддає ринок. Один — окремий випадок: перша ж
        #: невдача вичерпує пошук, і вердикт приходить раніше за будь-який стрік.
        self.n_candidates = 1

    def balance(self) -> Any:
        self.calls.append("balance")
        from gpurunner.core.models import BalanceReport
        return BalanceReport(backend="vast", available=self.credit, unit="$")

    # --- те, що кличе наглядач ---

    def find_candidates(self, **kw: Any) -> Selection:
        self.calls.append("find_candidates")
        sizing = plan_sizing(cores=64, vram_gb=32.0)
        scored = [
            ScoredOffer(
                offer={**OFFER, "id": i + 1}, machine_id=38902 + i, sizing=sizing,
                hours=0.9, cost=0.20, score=1000.0 - i,
            )
            for i in range(self.n_candidates)
        ]
        return Selection(candidates=scored)

    def submit_to_offer(self, job, params, *, gpu, offer, verify=None,
                        on_created=None) -> JobHandle:
        self.calls.append("rent")
        if self.submit_errors:
            raise self.submit_errors.pop(0)
        if verify is not None:
            overrides = verify(JobHandle(backend="vast", remote_id="42", job_name="htr_case",
                                         gpu=gpu), object())
            self.calls.append(f"gate:{sorted(overrides or {})}")
        self.calls.append("upload")
        return JobHandle(backend="vast", remote_id="42", job_name="htr_case", gpu=gpu)

    def probe_box(self, handle, *, net_probe_url: str = "") -> dict:
        self.calls.append("probe")
        return self.probe

    def fetch_outputs(self, handle, out_dir: Path) -> list[Path]:
        """Раннер кладе кожну справу у власну підтеку — це й відтворюємо."""
        self.calls.append("fetch")
        out = Path(out_dir) / "spr-6671"
        (out / "out").mkdir(parents=True, exist_ok=True)
        for i in range(self.texts):
            (out / "out" / f"{i:04d}.txt").write_text("текст", encoding="utf-8")
        (out / "htr_case_summary.json").write_text(json.dumps({
            "n_pages_expected": self.pages, "n_pages_input": self.pages,
            "n_pages_total": self.texts, "n_pages_txt": self.texts,
            "complete": self.complete, "missing_pages": [] if self.complete else ["0099"],
            "quarantined_pages": [], "failed_shards": [],
        }), encoding="utf-8")
        return []

    def cancel(self, handle, *, force: bool = False) -> None:
        self.calls.append("destroy")

    def _ssh(self, handle, timeout: int = 20):
        raise BackendError("no ssh in tests")

    def _instance(self, handle):
        return {"actual_status": "running", "dph_total": 0.222}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.setenv("GPURUNNER_BOXES_OVERRIDES", str(tmp_path / "over.json"))


def make_plan(tmp_path: Path, pages: int = 100) -> Plan:
    return Plan(
        assets_url="https://r2/assets.tgz",
        cases=[CasePlan(case="spr-6671", pages_url="https://r2/x.tar", n_pages=pages,
                        out_dir=str(tmp_path / "out"))],
        budget_usd=3.0, max_hours=8.0,
        # Ці тести — про ПОРЯДОК потоку (оренда → проба → заливка → забір), а
        # фейковий ринок віддає кандидата в обхід вибору; ціль і чекання ринку
        # перевіряють `test_market_replay` і `test_supervisor_honesty`.
        target_pph=0.0,
        max_wait_min=0.0,
    )


def run(sup: Supervisor, backend: FakeBackend, progress: dict) -> int:
    """Прогнати наглядача, підмінивши читання прогресу на готовий словник."""
    sup.backend = backend  # type: ignore[assignment]
    sup._read_progress = lambda: (progress, True)  # type: ignore[method-assign]
    sup._progress_age = lambda p: 1.0              # type: ignore[method-assign]
    sup._instance_state = lambda: "running"        # type: ignore[method-assign]
    return sup.run()


DONE = {"phase": "done", "n_pages_expected": 100, "pages_done": 100,
        "pages_per_hour": 2000, "missing_count": 0}


def test_happy_path_order_is_rent_probe_upload_fetch_verify_destroy(tmp_path: Path) -> None:
    """🔴 Порядок і є суть: проба ДО заливки, гасіння ПІСЛЯ звірки."""
    backend = FakeBackend(pages=100, texts=100)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    code = run(sup, backend, DONE)

    assert code == 0
    order = [c.split(":")[0] for c in backend.calls]
    assert order.index("probe") < order.index("upload")
    assert order.index("fetch") < order.index("destroy")
    assert sup.state.verdict == "ok"
    assert not sup.state.human_action_required


def test_shard_count_comes_from_the_probe_not_the_offer(tmp_path: Path) -> None:
    backend = FakeBackend(pages=100, texts=100)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    run(sup, backend, DONE)
    gate_call = next(c for c in backend.calls if c.startswith("gate:"))
    assert "shards" in gate_call
    assert "threads_per_shard" in gate_call


def test_incomplete_result_does_not_pass_as_success(tmp_path: Path) -> None:
    """🔴 Інцидент 203 з 323: неповне більше не може завершитись кодом 0."""
    backend = FakeBackend(pages=323, texts=203, complete=False)
    sup = Supervisor(make_plan(tmp_path, pages=323), backend=backend,  # type: ignore[arg-type]
                     tick_sec=0)
    code = run(sup, backend, {**DONE, "n_pages_expected": 323, "pages_done": 323})

    assert code == 4  # EXIT_INCOMPLETE
    assert sup.state.verdict == "incomplete"
    assert sup.state.human_action_required
    assert sup.state.cases[0].missing_count > 0
    # і все одно забрали те, що є, і погасили — гроші не горять
    assert "fetch" in backend.calls
    assert "destroy" in backend.calls


def test_bad_hardware_is_rejected_before_upload(tmp_path: Path) -> None:
    """Машина з 8 ядрами замість 64: заливки не буде взагалі."""
    backend = FakeBackend(pages=100, texts=100, probe={**HEALTHY_PROBE, "cores": 8.0})
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    code = run(sup, backend, DONE)

    assert "upload" not in backend.calls
    assert code != 0
    kinds = [i.kind for i in sup.state.incidents]
    assert "cpu_lie" in kinds


def test_rejected_box_lands_in_the_registry(tmp_path: Path) -> None:
    from gpurunner.core import boxes

    backend = FakeBackend(pages=100, texts=100, probe={**HEALTHY_PROBE, "net_bps": 62500.0,
                                                       "net_mbps": 0.5})
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    run(sup, backend, DONE)

    assert 38902 in boxes.banned_ids()
    assert "slow_net" in boxes.explain(38902)


def test_empty_market_asks_the_human_and_spends_nothing(tmp_path: Path) -> None:
    backend = FakeBackend(pages=100, texts=100)
    backend.find_candidates = lambda **kw: Selection(  # type: ignore[method-assign]
        candidates=[], reason="ринок порожній"
    )
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    code = run(sup, backend, DONE)

    assert code == 6
    assert sup.state.human_action_required
    assert "rent" not in backend.calls


def test_our_own_config_error_never_blames_the_machine(tmp_path: Path) -> None:
    """🔴 Замір 2026-08-11: один забутий `input_root` оббрехав три здорові бокси
    поспіль, серед них найшвидшу пропозицію дня (RTX 6000Ada 48 ГБ).

    Помилка без класифікації — наша: битий план, забутий параметр, зламаний
    архів. Машина за неї не відповідає, і в реєстр не потрапляє.
    """
    from gpurunner.core import boxes

    backend = FakeBackend(pages=100, texts=100)

    def boom(*a: Any, **kw: Any):
        raise BackendError("job needs inputs ['models', 'pages', 'scripts']")

    backend.submit_to_offer = boom  # type: ignore[method-assign]
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    code = run(sup, backend, DONE)

    assert code == 3                       # провал, але не «ринок порожній»
    assert boxes.banned_ids() == set()     # ЖОДНОЇ забаненої машини
    assert [i.kind for i in sup.state.incidents] == ["our_bug"]
    assert "НЕ звинувачую" in sup.state.incidents[0].action


def test_machine_fault_still_lands_in_the_registry(tmp_path: Path) -> None:
    """Контроль до попереднього: класифікована провина машини таки записується."""
    from gpurunner.core import boxes
    from gpurunner.core.backend import SshAuthRejected

    backend = FakeBackend(pages=100, texts=100)

    def boom(*a: Any, **kw: Any):
        raise SshAuthRejected("хост відхилив ключ")

    backend.submit_to_offer = boom  # type: ignore[method-assign]
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    run(sup, backend, DONE)

    # Один відхилений ключ — підозра, а не бан (здебільшого збій прив'язки на
    # боці Vast, 15.09.2026); головне тут — що вирок записано.
    assert boxes.verdicts()[38902].state == "warned"
    assert "ssh_auth_denied" in boxes.explain(38902)


def test_orphan_cleanup_never_raises(tmp_path: Path) -> None:
    """🔴 Найдорожчий баг сесії: `_destroy_orphan` будував `JobHandle` без
    обов'язкового `gpu` і падав із `ValidationError` — тобто рівно там, де
    треба спинити лічильник, наглядач помирав, а бокс лишався горіти
    (інстанс 47458730 у стані `created`, $0.256/год).

    Заразом виняток маскував СПРАВЖНЮ причину: у лозі лишалась помилка
    валідації замість «контейнер не створився».
    """
    backend = FakeBackend(pages=100, texts=100)
    killed: list[str] = []
    backend.cancel = lambda h, **kw: killed.append(h.remote_id)  # type: ignore[method-assign]

    def boom(*a: Any, **kw: Any):
        err = BackendError("instance 47458730 is RUNNING AND BILLING but setup failed")
        err.outcome = "never_booted"
        err.instance_id = "47458730"
        raise err

    backend.submit_to_offer = boom  # type: ignore[method-assign]
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    run(sup, backend, DONE)

    assert "47458730" in killed, "сирота мусить бути погашений"
    assert "ValidationError" not in " ".join(i.detail for i in sup.state.incidents)
    assert any(i.kind == "never_booted" for i in sup.state.incidents), \
        "справжня причина мусить лишитись у звіті"


def test_progress_age_restarts_with_each_rental(tmp_path: Path) -> None:
    """🔴 Живий баг, знайдений паралельною сесією 2026-08-11.

    Вік прогресу рахувався від старту НАГЛЯДАЧА. Через півтори години заходу
    свіжоорендований бокс миттєво «мовчав 107 хвилин», оголошувався мертвим
    за три хвилини після підйому — і наглядач крутився в циклі
    «взяв → убив → взяв», не прочитавши жодної сторінки.
    """
    backend = FakeBackend(pages=100, texts=100)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    sup.started -= 6000            # захід іде вже 100 хвилин
    sup._rented_at = sup.started

    assert sup._progress_age(None) > 5000, "до оренди відлік іде від заходу"

    sup._rented_at = __import__("time").monotonic()   # щойно орендували
    assert sup._progress_age(None) < 5, "після оренди відлік починається наново"


def test_assets_url_reaches_the_box(tmp_path: Path) -> None:
    """🔴 Скарга паралельної сесії: «раннер не знайшов ваг, значить план не
    передає assets_url наглядачу».

    Перевіряємо не міркуванням, а фактом: проганяємо параметри тим самим
    шляхом, яким вони їдуть на бокс, і шукаємо посилання у ЗГЕНЕРОВАНОМУ
    коді. Заразом фіксуємо ідемпотентність валідації — повторний виклик на
    власному ж виводі падав на `shards=0`.
    """
    from gpurunner.supervise.htr import Supervisor

    plan = make_plan(tmp_path)
    sup = Supervisor(plan, backend=FakeBackend(pages=100, texts=100),  # type: ignore[arg-type]
                     session="probe")
    params = sup._params_for(plan.cases[0], sup._need_for(100), resume=False,
                             queue=plan.cases)
    norm = sup.job.validate_params(params)
    assert norm["assets_url"] == plan.assets_url
    assert sup.job.validate_params(norm) == norm, "валідація мусить бути ідемпотентною"

    code = sup.job.render_remote_code(norm)
    assert plan.assets_url in code
    assert plan.cases[0].pages_url in code


# ---- ручки плану доходять до підбору, воріт і наглядача (2026-08-19) -------


class _Knobbed:
    """Мінімальний план: лише те, з чого будується `Need`."""

    max_hours = 8.0
    budget_usd = 3.0
    disk_gb = 40
    min_net_mbps = 20.0
    time_value_usd_per_hour = None
    gb_per_shard = 0.0
    max_usd_per_1000_pages = None
    max_cost_per_case = None
    total_pages = 700

    def __init__(self, params=None, case_params=None):
        self.params = params or {}
        self.cases = [type("C", (), {"params": case_params or {}})()]


def test_knobs_given_via_dash_p_reach_the_need() -> None:
    """🔴🔴 `-p max_usd_per_1000_pages=0.30` лягає в `params`, а поле `Plan`
    лишається порожнім — і наглядач читав ТІЛЬКИ поле.

    Доки ворота ціну не перевіряли, це було невидимо. З 2026-08-19 воно означає,
    що на кадрі-РОЗВОРОТІ (чесні $0.34-0.60 за тисячу) дефолтна стеля $0.20
    відсікла б увесь ринок і захід не поїхав би взагалі.
    """
    from gpurunner.supervise.htr import need_from_plan

    need = need_from_plan(
        _Knobbed(params={"vram_gb_per_shard": 4.5, "max_usd_per_1000_pages": 0.30,
                         "max_cost_per_case": 1.5}),
        pages=700,
    )
    assert need.gb_per_shard == 4.5
    assert need.max_usd_per_1000_pages == 0.30
    assert need.max_cost_per_case == 1.5


def test_case_params_win_over_empty_plan_fields() -> None:
    from gpurunner.supervise.htr import need_from_plan

    need = need_from_plan(_Knobbed(case_params={"vram_gb_per_shard": 3.5}), pages=700)
    assert need.gb_per_shard == 3.5


def test_plan_field_still_works_when_no_params() -> None:
    from gpurunner.supervise.htr import need_from_plan

    plan = _Knobbed()
    plan.gb_per_shard = 2.8
    plan.max_usd_per_1000_pages = 0.25
    need = need_from_plan(plan, pages=700)
    assert need.gb_per_shard == 2.8 and need.max_usd_per_1000_pages == 0.25


# ---- гроші: порожній баланс не сміє читатись як «ринок зайнятий» (04.09.2026) ---


def test_empty_vast_balance_stops_before_any_rent(tmp_path: Path) -> None:
    """🔴🔴 Вісімнадцять запусків наглядача поспіль пішли в нікуди тому, що
    `vast · 0.00 $` виглядав як `market_empty` з порадою дивитись реєстр банів.
    Баланс питається ПЕРШИМ — до пошуку офферів."""
    backend = FakeBackend(pages=100, texts=100, credit=0.0)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    code = run(sup, backend, DONE)

    assert sup.state.verdict == "no_credit"
    assert code == 8  # EXIT_NO_CREDIT — НЕ 6 (market_empty) і не 5 (budget_stop)
    assert "rent" not in backend.calls
    assert "find_candidates" not in backend.calls
    assert sup.state.human_action_required
    assert "billing" in (sup.state.human_action or "")


def test_unknown_balance_does_not_block_the_run(tmp_path: Path) -> None:
    """Невідомий баланс ≠ порожній. Бекенд без `balance()` мусить працювати."""
    backend = FakeBackend(pages=100, texts=100, credit=None)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    assert run(sup, backend, DONE) == 0


def test_second_offer_taken_in_a_row_asks_the_balance(tmp_path: Path) -> None:
    """🔴 Ринок так не поводиться: чотири різні оффери не бувають зайняті
    поспіль. На ДРУГОМУ 400 питаємо гроші прямо — один дешевий запит проти
    годин розслідування справного механізму."""
    backend = FakeBackend(pages=100, texts=100, credit=0.0)
    backend.n_candidates = 4
    for _ in range(2):
        err = BackendError("оффер уже зайняли (HTTP 400)")
        err.outcome = "offer_taken"  # type: ignore[attr-defined]
        backend.submit_errors.append(err)
    # Баланс на старті цю сесію пропускає: вдаємо, що на той момент він ще був.
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    sup._preflight_credit = lambda: True  # type: ignore[method-assign]
    code = run(sup, backend, DONE)

    assert sup.state.verdict == "no_credit"
    assert code == 8
    assert backend.calls.count("rent") == 2  # третьої спроби не було


def test_no_credit_from_the_body_of_400_is_terminal(tmp_path: Path) -> None:
    """Коли Vast сам сказав про кошти — наступного кандидата не пробуємо."""
    backend = FakeBackend(pages=100, texts=100)
    err = BackendError("Vast відмовив у створенні інстансу (HTTP 400) через КОШТИ")
    err.outcome = "no_credit"  # type: ignore[attr-defined]
    backend.submit_errors.append(err)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    code = run(sup, backend, DONE)

    assert sup.state.verdict == "no_credit"
    assert code == 8
    assert backend.calls.count("rent") == 1


# ---- OOM: наступний бокс не сміє брати те саме число (31.08.2026) -----------


def test_oom_raises_vram_per_shard_instead_of_circling(tmp_path: Path) -> None:
    """🔴🔴 Наглядач на `fleet_dying` з OOM гасив бокс і брав наступний З ТИМ
    САМИМ `gb_per_shard`, тобто ходив по колу, поки не вичерпає стелю оренд.
    Пакет 1900 (8 томів, 2900 стор.) за ніч пройшов ЧОТИРИ бокси з однаковим
    наслідком; поріг довелось піднімати руками 3.5 → 2.5 → 4.0 → 6.0."""
    from gpurunner.supervise.decide import Obs

    backend = FakeBackend(pages=100, texts=100)
    plan = make_plan(tmp_path)
    object.__setattr__(plan, "gb_per_shard", 3.0)
    sup = Supervisor(plan, backend=backend, tick_sec=0)  # type: ignore[arg-type]
    sup._probe = {"vram_total_gb": 24.0, "n_gpus": 1}

    assert sup._gb_per_shard() == 3.0
    sup._bump_gb_per_shard(Obs(progress={"oom_events": 40, "pages_failed": 45}))
    assert sup._gb_per_shard() == 3.7, "крок 0.7 — мінус шард, а не пів флоту"
    sup._bump_gb_per_shard(Obs(progress={"oom_events": 30, "pages_failed": 33}))
    assert round(sup._gb_per_shard(), 1) == 4.4
    kinds = [i.kind for i in sup.state.incidents]
    assert kinds == ["gb_per_shard_up", "gb_per_shard_up"]


def test_vram_bump_never_exceeds_half_the_card(tmp_path: Path) -> None:
    """Більше половини карти на шард — це вже один шард на бокс; далі підіймати
    нема куди, справа просто важка."""
    from gpurunner.supervise.decide import Obs

    backend = FakeBackend(pages=100, texts=100)
    plan = make_plan(tmp_path)
    object.__setattr__(plan, "gb_per_shard", 5.0)
    sup = Supervisor(plan, backend=backend, tick_sec=0)  # type: ignore[arg-type]
    sup._probe = {"vram_total_gb": 11.0, "n_gpus": 1}  # GTX 1080 Ti з нічного ринку
    sup._bump_gb_per_shard(Obs(progress={"oom_events": 100, "pages_failed": 115}))
    assert sup._gb_per_shard() == 5.5


# ---- «ринок порожній» мусить казати, ЩО саме відсіяло (31.08.2026) ----------


def test_market_empty_names_our_own_ceilings_not_just_the_market(tmp_path: Path) -> None:
    """🔴 Шість заходів поспіль закрились як `market_empty` — «машин немає».
    Машини були: їх відсікала `--max-hours 3` («GTX 1080 Ti · $0.074/год —
    ✗ 4.4 год > 3.0»). Після підняття стелі той самий бокс дав $0.117 за 1000
    сторінок, дешевше за всі денні RTX 3090."""
    from gpurunner.core.offer_score import Selection

    rejected = [
        ScoredOffer(offer=OFFER, machine_id=1, sizing=plan_sizing(cores=24, vram_gb=11.0),
                    hours=4.4, cost=0.33, score=0.0, rejects=["4.4 год > 3.0"]),
        ScoredOffer(offer=OFFER, machine_id=2, sizing=plan_sizing(cores=16, vram_gb=11.0),
                    hours=5.6, cost=0.58, score=0.0, rejects=["5.6 год > 3.0"]),
    ]
    why, action = Supervisor._explain_empty_market(
        Selection(candidates=[], rejected=rejected, reason="ринок не дав машини"))

    assert "стеля годин" in why
    assert "max-hours" in action
    assert "boxes ls" not in action, "порада про реєстр банів веде хибним слідом"


def test_boxes_ls_is_advised_only_when_the_registry_did_the_filtering() -> None:
    """Порада мусить іти від того, що НАСПРАВДІ відсіяло."""
    from collections import Counter

    sup = Supervisor.__new__(Supervisor)
    assert "boxes ls" in sup._action_for_rejects(Counter({"banned": 3}))
    assert "boxes ls" not in sup._action_for_rejects(Counter({"slow_net": 3}))
    assert "preflight" in sup._action_for_rejects(Counter({"our_bug": 2}))
    assert "balance" in sup._action_for_rejects(Counter({"offer_taken": 4}))


# ---- фаза забору не сміє висіти на мертвому сокеті (17.08.2026) -------------


def test_a_hanging_fetch_is_cut_off_instead_of_burning_the_rental(
        tmp_path: Path) -> None:
    """🔴 Бокс віддав 404 під час забору, і наглядач висів на мертвому сокеті
    ПІВГОДИНИ. Стан увесь цей час показував фазу `fetching` і не старів, бо
    процес був живий і чемно чекав, — тобто жоден приймач біди не бачив."""
    import threading

    backend = FakeBackend(pages=100, texts=100)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]

    started = threading.Event()

    def _never_returns() -> None:
        started.set()
        threading.Event().wait()   # саме те, що робить мертвий сокет

    with pytest.raises(BackendError, match="не вклався"):
        sup._with_deadline(_never_returns, seconds=0.2, what="забір по SFTP")

    assert started.is_set(), "виклик мусив початись, а не бути пропущеним"
    assert [i.kind for i in sup.state.incidents] == ["fetch_timeout"]


def test_a_deadline_passes_the_result_through(tmp_path: Path) -> None:
    backend = FakeBackend(pages=100, texts=100)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    assert sup._with_deadline(lambda: 42, seconds=5, what="проба") == 42
    assert not sup.state.incidents


def test_a_deadline_does_not_swallow_the_real_error(tmp_path: Path) -> None:
    """Помилка виклику мусить дійти як є: інакше «не вклався» приховає причину."""
    backend = FakeBackend(pages=100, texts=100)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]

    def _boom() -> None:
        raise BackendError("SFTP: no such file")

    with pytest.raises(BackendError, match="no such file"):
        sup._with_deadline(_boom, seconds=5, what="проба")


# ---- шляхи наглядача мусять бути АБСОЛЮТНІ (бойовий експеримент 04.09.2026) --


def test_staging_does_not_depend_on_the_current_directory(tmp_path: Path,
                                                          monkeypatch) -> None:
    """🔴🔴 Тут стояв `Path("out")` — тека відносно поточного каталогу процесу.
    Поки наглядача запускали руками з каталогу репозиторію, воно працювало;
    відчеплена задача планувальника робочого каталогу не має взагалі й дістає
    системну теку, де створити нічого не можна.

    Спіймано навмисним убивством боксу: наглядач правильно виявив смерть — і
    ВПАВ на `PermissionError: 'out'`, а рятувальний забір упав слідом. Сто
    готових сторінок урятували лише чекпоінти в R2.
    """
    from gpurunner.supervise.htr import _fallback_out_root, _staging_root

    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    for root in (_staging_root(), _fallback_out_root()):
        assert root.is_absolute(), f"{root} відносний — розкладеться від cwd"
        assert str(tmp_path) in str(root)


def test_a_box_we_cancelled_ourselves_is_not_blamed(tmp_path: Path) -> None:
    """🔴 Інстанс у стані «gone» виглядає однаково, хто б його не прибрав, і доти
    це БЕЗУМОВНО писалось у провину машині. Виміряно 04.09.2026 навмисним
    експериментом: справний V100, що читав 1714 стор/год, дістав
    `died_under_load` за те, що його вбили ми. Ще один такий випадок — і
    найдешевша швидка карта йде з ринку на 21 день ні за що."""
    from gpurunner.core import manifest
    from gpurunner.core.models import JobStatus

    backend = FakeBackend(pages=100, texts=100)
    sup = Supervisor(make_plan(tmp_path), backend=backend, tick_sec=0)  # type: ignore[arg-type]
    handle = JobHandle(backend="vast", remote_id="42", job_name="htr_case", gpu="V100")
    manifest.add(handle)
    sup._handle = handle

    assert sup._death_outcome() == "died_under_load", "невідома смерть — на хост"

    handle.status = JobStatus.CANCELLED
    manifest.update(handle)
    assert sup._death_outcome() == "user_stop", "нашe гасіння машині не провина"

def test_the_allowlist_and_the_parser_agree_about_top_level_keys() -> None:
    """🔴 Перелік дозволених ключів і РОЗБІР мусять знати одне й те саме.

    Розійтись вони можуть у два боки, і обидва вже траплялись:

    · ключ РОЗБИРАЄТЬСЯ, але його немає в переліку → сторож на кожному заході
      каже «ключ верхнього рівня невідомий — НЕ ДІЄ» про налаштування, яке
      діє. 23.09.2026 саме через це переробили робочий план, вирішивши, що
      `transport` не застосовується;
    · ключ є в переліку, але не розбирається → сторож мовчить, а налаштування
      мовчки не існує. Так сталося з `max_usd_per_core_h` наступного ж дня
      після його появи.

    Попередження, яке бреше, дорожче за відсутнє: воно змушує міняти те, що
    працює, і привчає відмахуватись від справжнього.
    """
    import re

    src = pathlib.Path(plan_mod.__file__).read_text(encoding="utf-8")
    parsed = set(re.findall(r'raw\.get\(\s*"([a-z_0-9]+)"', src))
    parsed |= set(re.findall(r'raw\[\s*"([a-z_0-9]+)"\s*\]', src))
    allowed = set(plan_mod._KNOWN_TOP_KEYS)

    silent = sorted(parsed - allowed)
    assert not silent, (
        "план читає ці ключі, а сторож зве їх невідомими: " + ", ".join(silent))

    # Зворотний бік: усе, що дозволено, має бути або полем `Plan`, або
    # розбиратись явно. `params` і `cases` — контейнери, вони окремо.
    fields = {f.name for f in dataclasses.fields(plan_mod.Plan)}
    containers = {"params", "cases"}
    dead = sorted(allowed - parsed - fields - containers)
    assert not dead, (
        "ці ключі дозволені, але нікуди не потрапляють: " + ", ".join(dead))
