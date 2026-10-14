"""Квоти без API: автопідрахунок, ручний якір, скидання періоду."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gpurunner.core import quota

pytestmark = pytest.mark.usefixtures("data_dir")


@pytest.fixture
def now() -> datetime:
    """🔴 Момент «зараз» тут ПРИБИТИЙ, а не взятий із годинника.

    Спільна фікстура віддає `datetime.now()`, і через це два тести падали
    **щосуботи**. Квота Kaggle тижнева, скидається в суботу 00:00 UTC, а
    фікстури кладуть прогони «20 годин тому». У суботу до 20:00 UTC такий
    прогін потрапляє в МИНУЛИЙ тиждень, не рахується, і `remaining` лишається
    30.0 замість 28.0. Тобто перевірка була справна шість днів на тиждень і
    брехлива на сьомий — найгірший різновид тесту, бо червоне на CI виглядає
    як регресія коду.

    Середа 16.09.2026 12:00 UTC з запасом усередині обох періодів, якими
    користуються ці тести: 108 годин від скидання тижня Kaggle і 372 від
    початку місяця (colab, modal) при максимальному `starts_h_ago=100`.

    ⚠ Перекривається САМЕ ТУТ, а не в `conftest`: `test_usage.py` міряє живі
    прогони проти справжнього годинника, і прибитий момент перетворює їх на
    «загублені» (перевірено — три падіння).
    """
    return datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


# ---- сценарій із формулювання задачі ---------------------------------------


def test_the_whole_manual_correction_story(make_run, now: datetime) -> None:
    """Kaggle: 30 год на тиждень, витратили 2, звірили 26, витратили ще 1.

    Рівно те, заради чого ручна корекція існує: автопідрахунок бачить лише наші
    прогони, і коли він розійшовся з вебом провайдера, користувач вписує правду,
    а далі віднімання продовжується від неї.
    """
    make_run(starts_h_ago=20, duration_h=2.0)
    assert quota.compute("kaggle", now=now).remaining == pytest.approx(28.0, abs=0.02)

    quota.set_anchor("kaggle", 26.0, note="звірив у вебі", ts=now - timedelta(hours=10))
    state = quota.compute("kaggle", now=now)
    assert state.remaining == pytest.approx(26.0, abs=0.02)
    assert state.anchor is not None and state.anchor["remaining"] == 26.0

    make_run(starts_h_ago=5, duration_h=1.0)
    assert quota.compute("kaggle", now=now).remaining == pytest.approx(25.0, abs=0.02)


def test_runs_before_the_anchor_are_not_double_counted(make_run, now: datetime) -> None:
    """Прогін до звіряння вже сидить у цифрі, яку вписав користувач."""
    make_run(starts_h_ago=20, duration_h=5.0)
    quota.set_anchor("kaggle", 26.0, ts=now - timedelta(hours=10))
    state = quota.compute("kaggle", now=now)
    assert state.used == 0
    assert state.remaining == pytest.approx(26.0)


def test_clearing_the_anchor_returns_to_pure_auto(make_run, now: datetime) -> None:
    make_run(starts_h_ago=20, duration_h=2.0)
    quota.set_anchor("kaggle", 26.0, ts=now - timedelta(hours=10))
    assert quota.compute("kaggle", now=now).remaining == pytest.approx(26.0)

    start, _ = quota.period_bounds(quota.get_config("kaggle"), now)
    assert quota.clear_anchor("kaggle", start) == 1
    state = quota.compute("kaggle", now=now)
    assert state.anchor is None
    assert state.remaining == pytest.approx(28.0, abs=0.02)


# ---- періоди ---------------------------------------------------------------


def test_weekly_period_starts_on_the_configured_weekday() -> None:
    cfg = quota.get_config("kaggle")
    assert cfg.period == quota.PERIOD_WEEK
    moment = datetime(2026, 8, 5, 13, 0, tzinfo=UTC)      # середа
    start, end = quota.period_bounds(cfg, moment)
    assert start == datetime(2026, 8, 1, tzinfo=UTC)      # попередня субота
    assert end == datetime(2026, 8, 8, tzinfo=UTC)
    assert start.weekday() == cfg.reset_weekday


def test_anchor_from_a_previous_period_is_ignored(make_run, now: datetime) -> None:
    """Після скидання квоти старе звіряння більше нічого не означає."""
    quota.set_anchor("kaggle", 3.0, ts=now - timedelta(days=30))
    state = quota.compute("kaggle", now=now)
    assert state.anchor is None
    assert state.remaining == pytest.approx(30.0)         # повний ліміт, а не 3


def test_monthly_period_covers_the_calendar_month() -> None:
    cfg = quota.get_config("colab")
    start, end = quota.period_bounds(cfg, datetime(2026, 8, 15, tzinfo=UTC))
    assert (start, end) == (datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC))


def test_prepaid_balance_never_resets() -> None:
    cfg = quota.get_config("vast")
    assert cfg.period == quota.PERIOD_NONE
    assert quota.next_reset(cfg) is None


# ---- режими списання -------------------------------------------------------


def test_kaggle_charges_session_hours_not_compute(make_run, now: datetime) -> None:
    """Година на T4x2 списує з тижневих 30 рівно годину, хоч обчислень дає дві."""
    make_run(starts_h_ago=5, duration_h=1.0, gpu="T4x2")
    assert quota.compute("kaggle", now=now).used == pytest.approx(1.0, abs=0.02)


def test_free_colab_has_no_units_and_therefore_no_allowance() -> None:
    """Compute units існують ТІЛЬКИ на Pro — на безкоштовному тарифі їх немає.

    Дефолт «100 units/міс» приписував би безкоштовному акаунту ліміт неіснуючого
    плану, а через курс 1.75 units за T4-годину — ще й ~57 T4-годин у загальному
    підсумку. Тому дефолт — free: одиниця «год», ліміту немає, залишок ``None``.
    """
    cfg = quota.get_config("colab")
    assert cfg.plan == "free"
    assert cfg.allowance is None and cfg.unit == "год"
    assert quota.compute("colab").remaining is None


def test_switching_to_pro_brings_units_back(make_run, now: datetime) -> None:
    cfg = quota.set_config("colab", plan="pro")
    assert (cfg.allowance, cfg.unit) == (100.0, "units")
    make_run(starts_h_ago=5, duration_h=1.0, backend="colab", gpu="T4")
    assert quota.compute("colab", now=now).used == pytest.approx(100 / 57, abs=0.02)


def test_unknown_plan_is_rejected() -> None:
    with pytest.raises(ValueError, match="плану"):
        quota.set_config("colab", plan="enterprise")
    with pytest.raises(ValueError, match="плану"):
        quota.set_config("kaggle", plan="pro")      # у Kaggle планів немає


def test_changing_plan_drops_stale_overrides(make_run) -> None:
    """У нового тарифу інша одиниця виміру, тож старі перекриття не переносяться.

    Інакше «100», вписане в units на Pro, лишилося б лімітом у годинах на free —
    число з однієї шкали під підписом з іншої.
    """
    quota.set_config("colab", plan="pro", allowance=250.0)
    assert quota.get_config("colab").allowance == 250.0
    cfg = quota.set_config("colab", plan="free")
    assert cfg.allowance is None and cfg.unit == "год"


def test_colab_pro_charges_units_scaled_by_the_card(make_run, now: datetime) -> None:
    """Colab тарифікує обчислення: A100 з'їдає units помітно швидше за T4."""
    quota.set_config("colab", plan="pro")
    make_run(starts_h_ago=5, duration_h=1.0, backend="colab", gpu="T4")
    t4_used = quota.compute("colab", now=now).used
    assert t4_used == pytest.approx(100 / 57, abs=0.02)

    make_run(starts_h_ago=4, duration_h=1.0, backend="colab", gpu="A100")
    total = quota.compute("colab", now=now).used
    assert (total - t4_used) == pytest.approx(t4_used * 3.68, rel=0.02)


def test_unconvertible_card_is_reported_not_charged_as_zero(make_run, now: datetime) -> None:
    """RTX4090 у Colab Pro неможливо перевести в units — це має бути видно.

    Списати нуль означало б завищити залишок і промовчати про це; тому прогін
    лишається в списку, але з ``converted=False``, і карта потрапляє в
    ``unconverted``.
    """
    quota.set_config("colab", plan="pro")
    make_run(starts_h_ago=5, duration_h=2.0, backend="colab", gpu="RTX4090")
    state = quota.compute("colab", now=now)
    assert state.used == 0
    assert state.unconverted == ["RTX4090"]
    assert not state.exact_used


# ---- конфігурація і ручні множники ------------------------------------------


def test_editing_the_allowance_stops_it_being_an_assumption() -> None:
    """30 год — наше припущення, доки користувач не вписав своє число.

    Різниця не косметична: у дашборді припущення підписане окремо, щоб залишок,
    порахований від вигаданого ліміту, не читався як факт від провайдера.
    """
    assert quota.get_config("kaggle").assumed is True
    cfg = quota.set_config("kaggle", allowance=20.0)
    assert cfg.allowance == 20.0 and cfg.assumed is False


def test_config_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="period"):
        quota.set_config("kaggle", period="fortnight")
    with pytest.raises(ValueError, match="reset_weekday"):
        quota.set_config("kaggle", reset_weekday=9)
    with pytest.raises(ValueError, match="від'ємним"):
        quota.set_config("kaggle", allowance=-1)
    with pytest.raises(ValueError, match="від'ємним"):
        quota.set_anchor("kaggle", -5)


def test_manual_gpu_factor_makes_a_card_convertible(make_run, now: datetime) -> None:
    quota.set_config("colab", plan="pro")
    make_run(starts_h_ago=5, duration_h=2.0, backend="colab", gpu="RTX4090")
    assert quota.compute("colab", now=now).used == 0

    quota.set_gpu_factor("RTX4090", 3.2, note="власний бенч")
    state = quota.compute("colab", now=now)
    assert state.unconverted == []
    assert state.used == pytest.approx(2.0 * 3.2 * (100 / 57), rel=0.02)

    quota.set_gpu_factor("RTX4090", None)
    assert quota.compute("colab", now=now).unconverted == ["RTX4090"]


def test_gpu_factor_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="додатним"):
        quota.set_gpu_factor("T4", -1)
    with pytest.raises(ValueError, match="порожня"):
        quota.set_gpu_factor("", 2.0)


# ---- знімки ----------------------------------------------------------------


def test_snapshots_round_trip_and_prune() -> None:
    old = datetime.now(tz=UTC) - timedelta(days=400)
    quota.add_snapshot("kaggle", 12.0, "год", 12.0, ts=old)
    quota.add_snapshot("kaggle", 10.0, "год", 10.0)
    assert len(quota.snapshots("kaggle", days=30)) == 1
    assert len(quota.snapshots("kaggle", days=500)) == 2
    assert quota.prune_snapshots(keep_days=180) == 1
    assert len(quota.snapshots("kaggle", days=500)) == 1


def test_period_boundary_is_specified_not_accidental(make_run, now: datetime) -> None:
    """🔴 Те, що раніше проявлялось як «тест падає щосуботи», тут задано ЯВНО.

    Прибитий момент — середа 16.09.2026 12:00 UTC, тиждень Kaggle почався в
    суботу 12.09 00:00, тобто 108 годин тому. Отже прогін «100 годин тому» —
    усередині періоду, а «120 годин тому» — до скидання й не рахується.

    Це не вада обліку, а межа тижня; саме на неї й натикався тест, прибитий до
    годинника, — і виглядало це як регресія коду, хоч код був справний.
    """
    start, _ = quota.period_bounds(quota.get_config("kaggle"), now)
    assert (now - start).total_seconds() / 3600 == pytest.approx(108.0), (
        "прибитий момент з'їхав — решта тверджень тут перестала щось означати")

    make_run(starts_h_ago=120, duration_h=2.0)   # до скидання
    assert quota.compute("kaggle", now=now).remaining == pytest.approx(30.0), (
        "прогін із МИНУЛОГО тижня не сміє списуватись із нової квоти")

    make_run(starts_h_ago=100, duration_h=2.0)   # після скидання
    assert quota.compute("kaggle", now=now).remaining == pytest.approx(28.0, abs=0.02), (
        "прогін усередині періоду мусить списуватись")
