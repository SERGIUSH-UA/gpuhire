"""Таблиця сценаріїв → дія. Це те, що раніше вирішував агент очима з логів."""

from __future__ import annotations

import pytest

from gpurunner.supervise.decide import Cfg, Obs, decide

CFG = Cfg(budget_usd=3.00, max_hours=8.0)


def progress(**kw) -> dict:
    base = {
        "phase": "running", "n_pages_expected": 1665, "pages_done": 800,
        "pages_per_hour": 2100, "missing_count": 0,
        # Зрілий прогін за замовчуванням: шарди працюють півгодини. Молодий
        # замір — окрема поведінка, у неї власні тести нижче.
        "wall_sec": 1800.0,
    }
    base.update(kw)
    return base


# ---- усе гаразд ------------------------------------------------------------


def test_fresh_progress_means_wait() -> None:
    action, why = decide(Obs(progress=progress(), progress_age_sec=12, dph=0.22), CFG)
    assert action == "wait"
    assert "865 сторінок" in why  # і в причині одразу видно, чого чекаємо


def test_wait_reports_eta() -> None:
    _, why = decide(Obs(progress=progress(), progress_age_sec=5, dph=0.22), CFG)
    assert "хв" in why


# ---- завершення ------------------------------------------------------------


def test_done_with_empty_queue_fetches_and_destroys() -> None:
    """🔴 Інцидент: 1.5 год простою вже завершеної справи, бо ніхто не дивився."""
    action, why = decide(Obs(progress=progress(phase="done", pages_done=1665)), CFG)
    assert action == "fetch_and_finish"
    assert "гашу" in why


def test_done_with_queue_keeps_the_box_warm() -> None:
    action, why = decide(
        Obs(progress=progress(phase="done", pages_done=1665), queue_left=3), CFG
    )
    assert action == "next_case"
    assert "8 хв" in why  # ціна холодного старту названа


def test_missing_pages_trigger_catchup_before_anything_else() -> None:
    action, _ = decide(
        Obs(progress=progress(phase="done", pages_done=1650, missing_count=15),
            queue_left=3),
        CFG,
    )
    assert action == "catchup"


def test_catchup_is_not_endless() -> None:
    action, _ = decide(
        Obs(progress=progress(phase="done", missing_count=15),
            catchup_rounds_done=2, catchup_passes=2),
        CFG,
    )
    assert action == "fetch_and_finish"


def test_failed_phase_still_fetches_what_there_is() -> None:
    action, _ = decide(Obs(progress=progress(phase="failed", pages_done=900)), CFG)
    assert action == "fetch_and_finish"


# ---- смерть боксу ----------------------------------------------------------


@pytest.mark.parametrize("state", ["exited", "offline", "stopped", "gone"])
def test_dead_instance_is_destroyed(state: str) -> None:
    action, why = decide(Obs(progress=progress(), instance_state=state), CFG)
    assert action == "destroy_dead_box"
    assert "платимо" in why


def test_ssh_streak_means_the_box_is_lost() -> None:
    action, _ = decide(Obs(progress=progress(), ssh_fail_streak=3), CFG)
    assert action == "destroy_dead_box"


def test_two_ssh_failures_are_not_enough() -> None:
    action, _ = decide(Obs(progress=progress(), ssh_fail_streak=2, progress_age_sec=30), CFG)
    assert action == "wait"


def test_dead_runner_on_a_live_box_is_relaunched() -> None:
    """Шарди піднімає вотчдог у самому раннері; якщо мовчить сам раннер —
    рятує лише переоренда з чекпоінтів."""
    action, why = decide(Obs(progress=progress(), progress_age_sec=1500), CFG)
    assert action == "rerent"
    assert "чекпоінт" in why


def test_runner_that_never_started_is_not_relaunched() -> None:
    """Переоренда рятує лише те, що вже працювало. Прогін, який не стартував
    жодного разу, на новій машині не стартує так само."""
    action, _ = decide(
        Obs(progress=None, progress_age_sec=1500, progress_miss_streak=3), CFG)
    assert action == "startup_failed"


# ---- гроші й час -----------------------------------------------------------


def test_budget_spent_stops_everything() -> None:
    action, _ = decide(Obs(progress=progress(), spent_usd=3.10, dph=0.22), CFG)
    assert action == "destroy_budget"


def test_projection_stops_before_the_money_runs_out() -> None:
    """🔴 Спиняємось, поки результат ще можна забрати, а не коли рахунок порожній."""
    obs = Obs(
        progress=progress(pages_done=100, pages_per_hour=200),  # 1565 стор. по 200/год
        spent_usd=2.00, dph=0.50,
    )
    action, why = decide(obs, CFG)
    assert action == "destroy_budget"
    assert "виміряним темпом" in why


def test_projection_uses_measured_rate_not_the_offer_promise() -> None:
    fast = Obs(progress=progress(pages_done=100, pages_per_hour=2400), spent_usd=0.2, dph=0.22)
    slow = Obs(progress=progress(pages_done=100, pages_per_hour=200), spent_usd=0.2, dph=0.22)
    assert decide(fast, CFG)[0] == "wait"
    assert slow.projected_total_usd() > fast.projected_total_usd()


def test_deadline_stops_the_run() -> None:
    action, why = decide(Obs(progress=progress(), elapsed_h=8.5), CFG)
    assert action == "destroy_deadline"
    assert "чекпоінт" in why


# ---- пріоритет правил ------------------------------------------------------


def test_dead_box_outranks_a_finished_case() -> None:
    """Гроші зупиняються першими: забирати нема звідки, якщо боксу немає."""
    action, _ = decide(
        Obs(progress=progress(phase="done"), instance_state="exited"), CFG
    )
    assert action == "destroy_dead_box"


def test_deadline_outranks_budget_projection() -> None:
    action, _ = decide(
        Obs(progress=progress(pages_per_hour=10), elapsed_h=9.0, spent_usd=0.5, dph=0.22), CFG
    )
    assert action == "destroy_deadline"


def test_no_progress_yet_is_not_a_reason_to_panic() -> None:
    """Перші хвилини (pip, ваги, кадри) прогресу ще немає — це норма."""
    action, _ = decide(Obs(progress=None, progress_age_sec=240), CFG)
    assert action == "wait"


def test_box_that_never_starts_working_is_not_re_rented() -> None:
    """🔴 Бокс живий, SSH бездоганний, залізо здорове — і нуль сторінок.

    Причина завжди в тому, що МИ на нього поклали: не доїхали моделі
    (порожня shell-змінна в посиланні), не розпакувався архів, упав pip.
    Інша машина поводитиметься так само, тож переоренда лише подвоїть
    рахунок.
    """
    action, why = decide(
        Obs(progress=None, progress_age_sec=800, progress_miss_streak=3), CFG)
    assert action == "startup_failed"
    assert "не допоможе" in why


def test_one_unreadable_progress_file_does_not_kill_the_box() -> None:
    """🔴 Термінальний вердикт від ОДНОГО збою читання файла.

    `_progress.json` пишеться атомарно через `os.replace`; читання може
    впасти на гонці, на транзієнтному OSError у sftp, на чому завгодно. Для
    SSH-збоїв лічильник серії був, а тут — ні: один чих давав «за 13 хв бокс
    не порахував жодної сторінки», знищення боксу й пораду шукати неіснуючу
    помилку в `assets_url`.
    """
    action, _ = decide(
        Obs(progress=None, progress_age_sec=800, progress_miss_streak=1), CFG)
    assert action != "startup_failed"


def test_first_progress_window_is_wider_than_a_cold_start() -> None:
    """Pip + assets + кадри законно тривають хвилини — до 12 не панікуємо."""
    assert decide(Obs(progress=None, progress_age_sec=600), CFG)[0] == "wait"


def test_dead_runner_after_real_work_is_still_re_rented() -> None:
    """А от якщо сторінки вже йшли — це смерть раннера, і чекпоінти рятують."""
    action, _ = decide(Obs(progress=progress(), progress_age_sec=1500), CFG)
    assert action == "rerent"


# ---- грейс свіжої оренди ----------------------------------------------------


def test_fresh_rental_is_not_declared_dead() -> None:
    """🔴 Весь цикл 2026-08-11 стояв на цьому.

    `GET /instances/<id>/` віддає порожньо ще кілька хвилин після оренди, наш
    код читав це як `gone` — і на ПЕРШОМУ Ж тіку казав `destroy_dead_box`.
    Чотири інстанси за п'ять хвилин, кожен прожив ~1 хвилину.
    """
    fresh = Obs(progress=None, instance_state="gone", rent_age_sec=30)
    assert decide(fresh, CFG)[0] == "wait"

    fresh_no_ssh = Obs(progress=None, instance_state="unknown",
                       ssh_fail_streak=5, rent_age_sec=60)
    assert decide(fresh_no_ssh, CFG)[0] == "wait"


def test_grace_expires_and_dead_box_is_caught() -> None:
    """Грейс не має ховати справді мертвий бокс — лише відкласти вирок."""
    old = Obs(progress=progress(), instance_state="exited", rent_age_sec=400)
    assert decide(old, CFG)[0] == "destroy_dead_box"


def test_grace_does_not_block_budget_or_deadline() -> None:
    """Гроші й строк — понад грейсом: вони не про здоров'я машини."""
    assert decide(Obs(progress=progress(), elapsed_h=9.0, rent_age_sec=10), CFG)[0] \
        == "destroy_deadline"
    assert decide(Obs(progress=progress(), spent_usd=5.0, dph=0.2, rent_age_sec=10), CFG)[0] \
        == "destroy_budget"


def test_setup_silence_is_not_a_dead_runner() -> None:
    """🔴 Тиша під час СЕТАПУ законна: pip, ассети, кілька ГБ кадрів,
    розпакування — публікувати нема чого за побудовою.

    Справа на 3.4 ГБ при пороговому каналі 20 Мбіт/с качається 20+ хвилин;
    зі стелею `stale_progress_sec` наглядач гасив би бокс, який щойно ДОКАЧАВ
    дані, а другий помирав би так само — дві оренди, нуль сторінок.
    """
    obs = Obs(progress={"phase": "starting", "pages_done": 0}, progress_age_sec=1500)
    action, _ = decide(obs, CFG)
    assert action != "rerent"


def test_silence_while_actually_running_is_a_dead_runner() -> None:
    """А от у фазі `running` та сама тиша означає саме те, що означала."""
    obs = Obs(progress={"phase": "running", "pages_done": 10}, progress_age_sec=1500)
    action, _ = decide(obs, CFG)
    assert action == "rerent"


def test_queue_pages_are_counted_across_the_whole_run() -> None:
    """Прогноз витрат на черзі мусить бачити ВСЮ чергу, а не хвіст справи."""
    obs = Obs(
        progress={"phase": "running", "n_pages_expected": 3000, "pages_done": 2900},
        queue_pages_total=15000, queue_pages_before=9000,
    )
    assert obs.pages_left == 3100


def test_finished_work_is_not_thrown_away_on_a_brief_ssh_flap() -> None:
    """🔴 Найдорожча з можливих помилок: гасити бокс із ГОТОВИМ результатом.

    Наглядач не пам'ятав останній прогрес — при мертвому SSH `progress` стає
    None, і те, що фаза була `done`, забувалось. Ssh-проксі Vast флапнув 90 с
    → бокс знищено, аварійний забір без SSH не взяв нічого, усе рахується з
    нуля за нові гроші.
    """
    obs = Obs(progress=None, ssh_fail_streak=3, last_phase="done", rent_age_sec=9999)
    action, _ = decide(obs, CFG)
    assert action != "destroy_dead_box"


def test_a_truly_lost_box_is_still_released() -> None:
    """Терпіння не безмежне: довга серія збоїв — це справді втрачений бокс."""
    obs = Obs(progress=None, ssh_fail_streak=25, last_phase="done", rent_age_sec=9999)
    action, _ = decide(obs, CFG)
    assert action == "destroy_dead_box"


def test_unfinished_work_keeps_the_short_ssh_patience() -> None:
    """Поки робота НЕ завершена, рятувати нема чого — терпіння коротке."""
    obs = Obs(progress=None, ssh_fail_streak=3, last_phase="running", rent_age_sec=9999)
    action, _ = decide(obs, CFG)
    assert action == "destroy_dead_box"


def test_a_young_measurement_never_triggers_the_budget_projection() -> None:
    """🔴🔴 Темп у перші хвилини — це НЕ темп.

    Бокс тягне тригігабайтний архів кадрів, розпаковує його, ставить пакети —
    і «сторінок за годину» майже нуль. Прогноз від такого числа сміттєвий, а
    запобіжник рахує від того, що виміряв, і сам себе спиняє. Тричі поспіль
    2026-08-12; найгірший випадок — Quadro RTX 6000 (64 ядра, $0.142), бокс
    піднявся за 169 с і був знищений з вироком «за виміряним темпом 120
    стор/год вийде $6.23», не прочитавши жодного аркуша.
    """
    obs = Obs(
        progress=progress(pages_done=3, pages_per_hour=120, wall_sec=120.0),
        spent_usd=0.07, dph=0.142,
    )
    action, _ = decide(obs, CFG)
    assert action != "destroy_budget"


def test_a_mature_measurement_still_stops_a_runaway() -> None:
    """Але дозрілий замір мусить спиняти захід — інакше запобіжника немає."""
    obs = Obs(
        progress=progress(pages_done=400, pages_per_hour=200, wall_sec=7200.0),
        spent_usd=2.00, dph=0.50,
    )
    action, why = decide(obs, CFG)
    assert action == "destroy_budget" and "виміряним темпом" in why


def test_hard_budget_ceiling_ignores_warmup_entirely() -> None:
    """Жорсткий поріг «витрачено ≥ бюджет» не залежить від зрілості заміру —
    інакше молодий бокс міг би палити гроші без стелі."""
    obs = Obs(
        progress=progress(pages_done=1, pages_per_hour=0, wall_sec=30.0),
        spent_usd=CFG.budget_usd + 0.01, dph=0.2,
    )
    action, why = decide(obs, CFG)
    assert action == "destroy_budget" and "бюджет вичерпано" in why


def test_resumed_pages_do_not_count_as_proof_of_work() -> None:
    """🔴 Ворота дозрілості мусять рахувати роботу ЦЬОГО боксу.

    `pages_done` несе ще й сторінки, підняті з чекпоінтів. На відновленій
    справі половина воріт («досить прочитаного, щоб вірити середньому») була
    виконана з першого ж тіку: після переоренди з 2000 відновлених сторінок
    вистачало трьох нових за 15 хвилин, щоб темп «дозрів» на рівні 12 стор/год,
    і бюджетний прогноз гасив бокс саме тоді, коли вотчдог піднімав завислий
    шард. Оренда знищена по вибірці з трьох сторінок.
    """
    obs = Obs(
        progress=progress(pages_done=2003, pages_resumed=2000,
                          pages_per_hour=12, wall_sec=1800.0),
        spent_usd=0.2, dph=0.2,
    )
    assert obs.throughput_settled is False
    action, _ = decide(obs, CFG)
    assert action != "destroy_budget"


def test_real_work_after_a_resume_does_mature() -> None:
    """А коли новий бокс справді прочитав своє — вимір знову дійсний."""
    obs = Obs(
        progress=progress(pages_done=2100, pages_resumed=2000,
                          pages_per_hour=200, wall_sec=1800.0),
        spent_usd=2.0, dph=0.5,
    )
    assert obs.throughput_settled is True


def test_price_per_page_is_checked_on_the_LIVE_rate() -> None:
    """🔴🔴 Стеля $/1000 перевірялась ЛИШЕ при виборі оффера — на ПРОГНОЗІ.

    А прогноз не знає, що за матеріал: сповідки з їхніми щільними графами
    йдуть удвічі повільніше за метрики на тій самій машині. Заміряно
    2026-08-12: Q RTX 6000 за $0.211/год мала в реєстрі замір 1562 стор/год
    (зроблений на МЕТРИЦІ) і модельний прогноз 1600 — обидва обіцяли ~$0.135
    за тисячу. Реальність на сповідках: 882 стор/год і $0.239, тобто на 20%
    вище стелі — і ніхто цього не помічав до кінця заходу.
    """
    obs = Obs(
        progress=progress(pages_done=900, pages_per_hour=882, wall_sec=3600.0),
        spent_usd=0.5, dph=0.211,
    )
    assert round(obs.usd_per_1000, 3) == 0.239
    action, why = decide(obs, CFG)
    assert action == "price_alert" and "вище стелі" in why


def test_cheap_enough_run_is_not_alerted() -> None:
    """Прогін у межах стелі не має турбувати нікого."""
    obs = Obs(
        progress=progress(pages_done=900, pages_per_hour=2200, wall_sec=3600.0),
        spent_usd=0.5, dph=0.185,
    )
    action, _ = decide(obs, CFG)
    assert action != "price_alert"


def test_price_alert_waits_for_a_mature_measurement() -> None:
    """На молодому темпі ціна сторінки завжди «жахлива» — це не привід кричати."""
    obs = Obs(
        progress=progress(pages_done=3, pages_per_hour=12, wall_sec=120.0),
        spent_usd=0.05, dph=0.2,
    )
    action, _ = decide(obs, CFG)
    assert action != "price_alert"


# ---- ціна сторінки: гасимо лише те, що ВИСТОЯЛОСЬ (розбір 2026-08-19) ------

#: Машина 139040 з чужим процесом на карті: 165 стор/год за $0.161/год —
#: $0.976 за тисячу при стелі $0.20.
SLOW = {"pages_per_hour": 165, "pages_done": 120, "wall_sec": 3600.0}


def test_first_price_breach_only_warns() -> None:
    """Одна просадка темпу — не стан машини: важкий аркуш, качання наступної
    справи, вотчдог піднімає шард. Гасити по ній означає викинути оплачене."""
    action, _ = decide(
        Obs(progress=progress(**SLOW), dph=0.161, price_breach_streak=1,
            price_breach_sec=30.0),
        CFG,
    )
    assert action == "price_alert"


def test_sustained_price_breach_destroys() -> None:
    """🔴 А коли ціна ТРИМАЄТЬСЯ пів години — це вже інша арифметика заходу:
    за ті самі гроші деінде проходить удесятеро більше справ."""
    action, why = decide(
        Obs(progress=progress(**SLOW), dph=0.161, price_breach_streak=30,
            price_breach_sec=1800.0),
        CFG,
    )
    assert action == "destroy_overpriced"
    assert "чекпоінт" in why


def test_price_breach_that_recovers_is_not_a_verdict() -> None:
    """Темп повернувся в норму — лічильники обнуляє наглядач, і дії немає."""
    action, _ = decide(
        Obs(progress=progress(), dph=0.161, price_breach_streak=0,
            price_breach_sec=0.0),
        CFG,
    )
    assert action == "wait"


def test_long_breach_but_few_ticks_still_only_warns() -> None:
    """Обидві умови разом: час без числа замірів — це один довгий провал
    зв'язку, а не тридцять послідовних вимірів."""
    action, _ = decide(
        Obs(progress=progress(**SLOW), dph=0.161, price_breach_streak=1,
            price_breach_sec=3600.0),
        CFG,
    )
    assert action == "price_alert"


# ---- флот сиплеться --------------------------------------------------------


def shards(alive: int, total: int, *, rc: int = 1) -> list[dict]:
    """Флот, де `alive` шардів ще працює, а решта вийшла з кодом `rc`."""
    return [{"k": k + 1, "alive": k < alive, "rc": None if k < alive else rc}
            for k in range(total)]


def test_finished_shards_are_not_a_dying_fleet() -> None:
    """🔴🔴 РЕГРЕСІЯ, що коштувала шести оренд поспіль 2026-08-19.

    `alive` у раннері означає рівно «процес ще живий». Шард, який дочитав свою
    частину справи, виходить із `rc=0` — і перша версія правила читала це як
    смерть: «живих шардів 1 із 2 — флот обсипався (збоїв 0)», бокс гасився,
    брався наступний, і так шість разів, доки не втрутилась людина.
    """
    for alive, total in ((2, 4), (1, 2), (0, 4)):
        obs = Obs(progress=progress(shards=shards(alive, total, rc=0), pages_failed=0),
                  dph=0.27, dead_streak=9)
        assert decide(obs, CFG)[0] == "wait", (alive, total)


def test_half_the_fleet_dead_is_not_a_wait() -> None:
    """🔴 Машина 39565: 15 із 16 шардів мертві, 69 збоїв зі 196 спроб — і
    наглядач півгодини казав «лишилось 2125 сторінок, ще ~102 хв».

    Смерть — це НЕНУЛЬОВИЙ rc: вотчдог раннера підіймає впалий шард сам і
    скидає rc, тож сюди доходить лише те, що він уже здав.
    """
    action, why = decide(
        Obs(progress=progress(shards=shards(1, 16), pages_failed=69, oom_events=12),
            dph=0.27, dead_streak=2),
        CFG,
    )
    assert action == "rerent"
    assert "15 шардів із 16" in why and "OOM" in why


def test_dead_shards_need_their_own_two_ticks() -> None:
    """Один тік — не вирок: шард міг щойно впасти, і вотчдог його підніме."""
    obs = Obs(progress=progress(shards=shards(1, 16), pages_failed=69), dph=0.27,
              dead_streak=1)
    assert decide(obs, CFG)[0] == "wait"


def test_drained_shards_are_not_a_dying_fleet() -> None:
    """🚰 Регулятор зливає зайві шарди, а раннер nyshporka до 12.09.2026 при
    цьому повертав rc=3. Злиті — не мертві, хоч би з яким rc вийшли."""
    fleet = shards(8, 16, rc=3)
    for sh in fleet[8:]:
        sh["draining"] = True
    obs = Obs(progress=progress(shards=fleet, pages_failed=0), dph=0.27, dead_streak=5)
    assert obs.shards_dead == 0
    assert decide(obs, CFG)[0] == "wait"


def test_high_failure_ratio_needs_two_ticks() -> None:
    p = progress(shards=shards(16, 16), pages_done=100, pages_failed=60)
    assert decide(Obs(progress=p, dph=0.27, fail_streak=1), CFG)[0] == "wait"
    action, why = decide(Obs(progress=p, dph=0.27, fail_streak=2), CFG)
    assert action == "rerent" and "збоях" in why


def test_healthy_fleet_with_a_few_failures_keeps_working() -> None:
    """Кілька збоїв на сотні сторінок — норма, а не привід платити двічі."""
    action, _ = decide(
        Obs(progress=progress(shards=shards(16, 16), pages_failed=3), dph=0.27,
            fail_streak=9),
        CFG,
    )
    assert action == "wait"


def test_one_quarantined_page_does_not_kill_a_healthy_box() -> None:
    """🔴 РЕГРЕСІЯ 2026-08-22: наглядач гасив СПРАВНИЙ бокс через одну сторінку.

    На відновленій справі `fresh` дорівнює нулю (усе підняте з чекпоінтів), тож
    єдина невдала сторінка дає рівно 100% — і `fleet_dying` спрацьовував тричі
    поспіль на ЦДІАК ф.224, хоча в лозі раннера обидва шарди щоразу віддавали
    `rc=0` за 10 с. Знаменник у частки має бути значущий.
    """
    obs = Obs(
        progress=progress(shards=shards(2, 2), pages_done=91, pages_resumed=91,
                          pages_failed=1),
        dph=0.09, fail_streak=7,
    )
    assert obs.fail_ratio == 1.0          # частка справді 100% …
    assert obs.fail_attempts == 1         # … але спроба всього одна
    assert decide(obs, CFG)[0] != "rerent"


def test_quarantined_pages_leave_the_numerator() -> None:
    """Карантин ≠ хворий флот: цю сторінку раннер більше НЕ пробує ніколи.

    Інакше вона назавжди сидить у чисельнику й тримає частку на стелі — навіть
    коли решту справи вже дораховано здоровим флотом.
    """
    obs = Obs(
        progress=progress(shards=shards(8, 8), pages_done=140, pages_resumed=100,
                          pages_failed=12, quarantined=12),
        dph=0.27, fail_streak=5,
    )
    assert obs.pages_failed_active == 0
    assert obs.fail_ratio == 0.0
    assert decide(obs, CFG)[0] != "rerent"


def test_real_fleet_collapse_still_triggers() -> None:
    """А справжній колапс мусить ловитись, як і ловився (машина 39565, 08-19)."""
    obs = Obs(
        progress=progress(shards=shards(16, 16), pages_done=127, pages_resumed=0,
                          pages_failed=69),
        dph=0.27, fail_streak=2,
    )
    assert obs.fail_attempts >= CFG.fail_min_attempts
    assert decide(obs, CFG)[0] == "rerent"


def test_resumed_pages_do_not_dilute_the_failure_ratio() -> None:
    """🔴 Знаменник — СВІЖА робота. З `pages_done` разом із піднятим із
    чекпоінтів будь-яка частка виглядає мізерною саме тоді, коли флот сиплеться."""
    obs = Obs(
        progress=progress(shards=shards(16, 16), pages_done=2000,
                          pages_resumed=1980, pages_failed=40),
        dph=0.27, fail_streak=2,
    )
    assert obs.fail_ratio > 0.6
    assert decide(obs, CFG)[0] == "rerent"


# ---- сетап, який СТОЇТЬ (17.08.2026: 50 хв простою при ssh_ok) --------------


def test_a_stalled_setup_is_caught_by_movement_not_by_the_clock() -> None:
    """🔴 Бокс завис на качанні 412-МБ архіву й простояв 50 хвилин, а стан весь
    цей час казав `ssh_ok: true`, `human_action_required: false`. Годинна стеля
    його не спіймала б узагалі, бо 50 менше за 60."""
    obs = Obs(progress=None, ssh_ok=True, rent_age_sec=3000,
              setup_stall_sec_seen=50 * 60)
    action, why = decide(obs, CFG)
    assert action == "rerent"
    assert "не посувається" in why


def test_a_slow_but_moving_setup_is_left_alone() -> None:
    """Справа на 3.4 ГБ при пороговому каналі законно качається 20+ хвилин.
    Поки лог росте — це робота, а не зависання."""
    obs = Obs(progress=None, ssh_ok=True, rent_age_sec=1500,
              setup_stall_sec_seen=120)
    assert decide(obs, CFG)[0] != "rerent"


def test_setup_movement_is_not_judged_before_anything_was_seen() -> None:
    """Нуль означає «ще не бачили логу», а не «лог не росте»."""
    obs = Obs(progress=None, ssh_ok=True, rent_age_sec=600,
              setup_stall_sec_seen=0.0)
    assert decide(obs, CFG)[0] != "rerent"


# ---- пропущені сторінки — це ЗРОБЛЕНІ сторінки (звід із п'яти випадків) -----


def test_skipped_pages_are_not_counted_as_work_left() -> None:
    """🔴 Пропущена сторінка — це та, у якої текст на диску ВЖЕ Є: шард побачив
    його й пішов далі. Пропуски з'являються щоразу, коли вотчдог перепідіймає
    завислий шард, тобто в нормальному житті прогону.

    Доти залишок завищувався на число пропущених, прогноз витрат роздувався
    слідом, і `budget_stop` гасив ЗДОРОВИЙ прогін."""
    obs = Obs(progress={"n_pages_expected": 1000, "pages_done": 300,
                        "pages_skipped": 600})
    assert obs.pages_left == 100


def test_double_counting_can_only_understate_the_tail() -> None:
    """Підняте з чекпоінтів сидить і в `pages_done`, і пізніше в пропущених.
    Сума обмежена знаменником: помилка можлива лише в безпечний бік, бо
    витрати однаково стереже фактична сплачена сума."""
    obs = Obs(progress={"n_pages_expected": 1000, "pages_done": 900,
                        "pages_skipped": 900})
    assert obs.pages_left == 0


def test_a_run_without_skips_behaves_exactly_as_before() -> None:
    obs = Obs(progress={"n_pages_expected": 1000, "pages_done": 300})
    assert obs.pages_left == 700


def test_an_inflated_tail_no_longer_forecasts_a_false_budget_stop() -> None:
    """Той самий прогін, що доти гасився за бюджетом: 600 сторінок уже мають
    текст, але лічильник їх не бачив."""
    common = {"n_pages_expected": 1000, "pages_done": 300, "pages_per_hour": 300.0}
    honest = Obs(progress={**common, "pages_skipped": 600}, dph=0.20, spent_usd=0.50)
    blind = Obs(progress=common, dph=0.20, spent_usd=0.50)

    assert honest.projected_total_usd() < blind.projected_total_usd()
    assert honest.projected_total_usd() < 0.60


# ---- 21.09.2026: захід спинився за темпом, якого в нього не було ------------


def _spr_203b_progress(**kw) -> dict:
    """Знімок того самого тіку, на якому наглядач погасив здоровий бокс.

    96 сторінок за 932.8 с — це 371 стор/год «у середньому», разом зі стартом
    семи процесів і завантаженням ваг. Регулятор у тому ж знімку міряв сталий
    хід: `rates {"7": 758}`.
    """
    base = {
        "phase": "running", "n_pages_expected": 217, "pages_done": 96,
        "pages_per_hour": 371, "wall_sec": 932.8, "missing_count": 121,
        "case_index": 1, "cases_total": 5,
        "regulator": {"active": 7, "rates": {"7": 758}},
    }
    base.update(kw)
    return base


def _queue_obs(**kw) -> Obs:
    """Черга з п'яти справ: 812 сторінок, $0.256/год, витрачено $0.0785."""
    return Obs(
        progress=_spr_203b_progress(), spent_usd=0.0785, dph=0.25611,
        queue_pages_total=812, queue_pages_before=0, **kw,
    )


def test_projection_uses_the_steady_rate_not_the_warmup() -> None:
    """🔴🔴 Розгін не є темпом, і прогноз на ньому будувати не можна.

    За середніми 371 стор/год виходило $0.57 при бюджеті $0.50 — бокс
    погасили. Перезапуск на ТІЙ САМІЙ машині прочитав усю чергу за 32
    хвилини і $0.151.
    """
    obs = _queue_obs()
    assert obs.steady_pages_per_hour == 758
    naive = obs.spent_usd + (obs.pages_left / 371) * obs.dph
    assert naive > 0.50          # так рахували — і гасили
    assert obs.projected_total_usd() < 0.50


def test_projection_still_counts_the_warmup_of_cases_ahead() -> None:
    """Розгін не зникає з прогнозу — він переїжджає туди, де справді буде.

    Кожна наступна справа черги заплатить старт флоту й качання кадрів знову,
    тож прогноз мусить додати його стільки разів, скільки справ попереду, а
    не розмазати розгін першої справи на всі 812 сторінок.
    """
    obs = _queue_obs()
    assert obs.case_overhead_sec > 300          # ~477 с на цьому заході
    bare = obs.spent_usd + (obs.pages_left / 758) * obs.dph
    assert obs.projected_total_usd() > bare


def test_healthy_box_is_not_killed_by_the_budget_rule() -> None:
    """Той самий знімок — і наглядач більше не гасить бокс."""
    action, _ = decide(_queue_obs(), CFG)
    assert action != "destroy_budget"


def test_without_regulator_the_old_average_still_works() -> None:
    """Старий раннер не публікує `rates` — там усе як було."""
    obs = Obs(progress=_spr_203b_progress(regulator=None),
              spent_usd=2.00, dph=0.50, queue_pages_total=812)
    assert obs.steady_pages_per_hour == 0
    assert obs.projected_total_usd() > obs.spent_usd
