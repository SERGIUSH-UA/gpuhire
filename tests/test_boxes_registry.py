"""Реєстр боксів: фолд, TTL, двоударність і паралельний допис.

Кожен тест названий інцидентом, який він робить неповторюваним.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from gpurunner.core import boxes
from gpurunner.core.boxes import BoxObservation

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))
    monkeypatch.setenv("GPURUNNER_BOXES_OVERRIDES", str(tmp_path / "boxes.overrides.json"))
    # Амністія зламаних таймерів — окрема поведінка з власними тестами нижче.
    # Решта правил реєстру від неї не залежить, тож відсуваємо епоху в минуле,
    # щоб вона не з'їдала їхні заміри.
    monkeypatch.setattr(boxes, "BROKEN_TIMER_EPOCH", datetime(2000, 1, 1, tzinfo=UTC))
    return tmp_path


def add(outcome: str, *, machine_id: int = 4711, days_ago: float = 0, **kw) -> None:
    boxes.record(
        BoxObservation(
            machine_id=machine_id,
            outcome=outcome,
            ts=(NOW - timedelta(days=days_ago)).isoformat(),
            **kw,
        )
    )


# ---- двоударність ----------------------------------------------------------


def test_single_flake_does_not_ban() -> None:
    """Один недосяжний SSH міг бути мережевим збоєм — банити рано.

    Той самий принцип, що в локальній черзі: перше зависання не карантинить.
    """
    add("ssh_unreachable")
    assert boxes.verdicts(now=NOW)[4711].state == "warned"
    assert boxes.banned_ids(now=NOW) == set()


def test_second_flake_bans() -> None:
    add("ssh_unreachable", days_ago=1)
    add("ssh_unreachable")
    assert boxes.banned_ids(now=NOW) == {4711}


def test_slow_boot_is_not_a_permanent_verdict() -> None:
    """🔴 Машина, що довго тягне образ, — не зламана.

    Один `never_booted` більше не банить (треба два), вирок протухає за 14
    днів, а пізніший успіх знімає його одразу: це прямий доказ, що контейнер
    таки створюється.
    """
    add("never_booted", detail="застряг у created 10 хв")
    assert boxes.banned_ids(now=NOW) == set()
    add("never_booted", days_ago=0.1)
    assert boxes.banned_ids(now=NOW) == {4711}
    add("ok")
    assert boxes.banned_ids(now=NOW) == set()


def test_auth_denial_needs_a_second_hit() -> None:
    """🔴 Відхилений ключ виявився здебільшого збоєм прив'язки на боці Vast:
    138268 працювала за 20 хв до і через 20 хв після відмови (03.09.2026),
    12728 пройшла з другої спроби (15.09). Один удар — підозра, два — бан."""
    add("ssh_auth_denied", days_ago=0.1, detail="Permission denied by authenticating user root")
    assert boxes.banned_ids(now=NOW) == set()
    assert boxes.verdicts(now=NOW)[4711].state == "warned"
    add("ssh_auth_denied")
    assert boxes.banned_ids(now=NOW) == {4711}


def test_measured_lies_ban_on_first_hit() -> None:
    """Замір однозначний: канал 0.5 Мбіт/с проти обіцяних 755 — не флуктуація."""
    for kind in ("slow_net", "cpu_lie", "vram_lie"):
        assert boxes.HITS_TO_BAN[kind] == 1, kind
    add("slow_net", detail="0.5 Мбіт/с проти обіцяних 755")
    assert boxes.banned_ids(now=NOW) == {4711}


def test_disk_short_never_blames_the_machine() -> None:
    """🔴 `disk_short` — це розбіжність НАШОЇ вимоги з машиною, а не поломка.

    Замір 2026-08-11: план просив 120 ГБ (реально треба ~40 — кадри 8.3 ГБ,
    розпаковані 9.4, моделі 105 МБ), машина дала 104 і поїхала в чорний список
    на два тижні за наш власний прорахунок. Здорові хости так вибувають з
    ринку один за одним, і виглядає це як «ринок порожній».
    """
    assert "disk_short" in boxes.NEUTRAL
    add("disk_short", detail="104 ГБ із замовлених 120")
    assert boxes.banned_ids(now=NOW) == set()


# ---- TTL -------------------------------------------------------------------


def test_a_live_channel_never_bans_whatever_the_old_limit_said() -> None:
    """🔴 35928 (15.1 Мбіт/с після повної черги, 13.09.2026), 39565 і 43503 (2.1 і
    6.1, 15.09) — забанені сталою межею 20. Живий канал — не вада хоста."""
    add("slow_net", days_ago=0.1, measured={"net_mbps": 15.1})
    add("slow_net", measured={"net_mbps": 2.1})
    assert boxes.banned_ids(now=NOW) == set()
    assert boxes.verdicts(now=NOW)[4711].state != "warned"


def test_a_barely_alive_channel_after_a_success_needs_two_hits() -> None:
    add("ok", days_ago=0.2)
    add("slow_net", days_ago=0.1, measured={"net_mbps": 0.6})
    verdict = boxes.verdicts(now=NOW)[4711]
    assert verdict.state == "warned" and "slow_net_after_ok" in verdict.reason
    add("slow_net", measured={"net_mbps": 0.4})
    assert boxes.banned_ids(now=NOW) == {4711}


def test_a_dead_channel_bans_even_after_a_success() -> None:
    """`000` / нуль — канал мертвий, а не повільний: це вада хоста зараз."""
    add("ok", days_ago=0.2)
    add("slow_net", measured={"net_mbps": 0.0})
    assert boxes.banned_ids(now=NOW) == {4711}


def test_a_dead_channel_without_any_success_bans_on_first_hit() -> None:
    add("slow_net", measured={"net_mbps": 0.5})
    assert boxes.banned_ids(now=NOW) == {4711}


def test_zero_speed_from_the_broken_probe_is_void(monkeypatch) -> None:
    """🔴 До 11.09.2026 проба перетирала виміряну швидкість нулем: HTTP 206 (дані
    йшли) і «0.0 Мбіт/с» — так забанено 140184, що мала 208 Мбіт/с."""
    monkeypatch.setattr(boxes, "BROKEN_PROBE_EPOCH", NOW)
    add("slow_net", days_ago=1, measured={"net_mbps": 0.0, "net_http": "206"})
    assert boxes.banned_ids(now=NOW) == set()
    add("slow_net", machine_id=4712, days_ago=1, measured={"net_mbps": 0.0, "net_http": "000"})
    assert boxes.banned_ids(now=NOW) == {4712}, "`000` — канал не відповів, амністія не діє"


def test_ban_expires_so_a_repaired_box_gets_another_chance() -> None:
    add("slow_net", days_ago=13)
    assert boxes.banned_ids(now=NOW) == {4711}
    add("slow_net", machine_id=4712, days_ago=15)
    assert 4712 not in boxes.banned_ids(now=NOW)


def test_flake_ttl_is_short() -> None:
    """Недосяжність забувається за 3 дні, брехня про залізо — за 14."""
    add("ssh_unreachable", days_ago=4)
    add("ssh_unreachable", days_ago=3.5)
    assert boxes.banned_ids(now=NOW) == set()


# ---- успіх знімає (і не знімає) --------------------------------------------


def test_success_clears_a_transient_verdict() -> None:
    """Машина зникала під навантаженням, потім двічі відпрацювала — беремо."""
    add("died_under_load", days_ago=5)
    add("died_under_load", days_ago=4)
    assert boxes.banned_ids(now=NOW) == {4711}
    add("ok", days_ago=1)
    verdict = boxes.verdicts(now=NOW)[4711]
    assert verdict.state == "starred"
    assert 4711 not in boxes.banned_ids(now=NOW)


def test_auth_denial_is_forgotten_in_three_days() -> None:
    """Два удари банять, але ненадовго: збій прив'язки ключа не властивість хоста."""
    add("ssh_auth_denied", days_ago=1)
    add("ssh_auth_denied", days_ago=0.5)
    assert boxes.banned_ids(now=NOW) == {4711}
    add("ssh_auth_denied", machine_id=4712, days_ago=5)
    add("ssh_auth_denied", machine_id=4712, days_ago=4)
    assert 4712 not in boxes.banned_ids(now=NOW)


def test_later_success_clears_auth_denial() -> None:
    """Ключ прийнявся — отже твердження «не приймає» більше не діє."""
    add("ssh_auth_denied", days_ago=5)
    add("ok", days_ago=1)
    assert boxes.banned_ids(now=NOW) == set()


def test_denial_after_a_success_is_judged_as_transient() -> None:
    """🔴 Живий замір 2026-08-11, який спростував саме припущення.

    Машина 144417 о 20:03 прогнала справу ПОВНІСТЮ (`ok`), а о 20:19 відхилила
    той самий ключ — і поїхала в бан на 30 днів із поясненням «успіху тут не
    буває за побудовою», яке спростовував запис у тому ж реєстрі за 16 хвилин
    до того. Тобто назавжди викидалась із ринку єдина машина, яка щойно
    довела, що працює. Одна така відмова — підозра, не вирок.
    """
    add("ok", days_ago=2)
    add("ssh_auth_denied", days_ago=1)
    assert boxes.banned_ids(now=NOW) == set()

    add("ssh_auth_denied", days_ago=1)  # друга поспіль — уже досить
    assert boxes.banned_ids(now=NOW) == {4711}


def test_bad_after_success_counts_again() -> None:
    add("ok", days_ago=6)
    add("slow_net", days_ago=1)
    assert boxes.banned_ids(now=NOW) == {4711}


# ---- зірки -----------------------------------------------------------------


def test_starred_sorted_by_measured_pages_per_dollar() -> None:
    """«Добрий дешевий бокс» — це не найдешевший, а найбільше сторінок за долар,
    і обидва числа беруться з ЗАМІРУ, а не з картки оффера."""
    add("ok", machine_id=100, claimed={"dph_total": 0.22},
        measured={"pages_per_hour": 2278}, gpu_name="Tesla V100")
    add("ok", machine_id=200, claimed={"dph_total": 0.10},
        measured={"pages_per_hour": 400}, gpu_name="RTX 3090")
    top = boxes.starred(now=NOW)
    assert [v.machine_id for v in top] == [100, 200]  # 10355 проти 4000 стор/$


def test_star_expires_without_fresh_success() -> None:
    add("ok", days_ago=45)
    assert boxes.verdicts(now=NOW)[4711].state == "unknown"


def test_starred_filters_by_gpu() -> None:
    add("ok", machine_id=100, gpu_name="Tesla V100", measured={"pages_per_hour": 2278})
    add("ok", machine_id=200, gpu_name="RTX 3090", measured={"pages_per_hour": 1564})
    assert [v.machine_id for v in boxes.starred(gpu="V100", now=NOW)] == [100]


# ---- ручні рішення ---------------------------------------------------------


def test_manual_never_outranks_everything(registry: Path) -> None:
    (registry / "boxes.overrides.json").write_text(
        json.dumps({"never": {"4711": "жере дані, двічі"}, "star": {}}), encoding="utf-8"
    )
    add("ok")
    verdict = boxes.verdicts(now=NOW)[4711]
    assert verdict.state == "banned"
    assert "жере дані" in verdict.reason
    assert verdict.expires is None  # ручне не протухає


def test_manual_star_shows_up_without_any_run(registry: Path) -> None:
    (registry / "boxes.overrides.json").write_text(
        json.dumps({"never": {}, "star": {"38902": "V100 64 ядра, $0.22"}}), encoding="utf-8"
    )
    assert boxes.verdicts(now=NOW)[38902].state == "starred"


def test_broken_overrides_file_does_not_break_selection(registry: Path) -> None:
    (registry / "boxes.overrides.json").write_text("{не json", encoding="utf-8")
    add("ok")
    assert boxes.verdicts(now=NOW)[4711].state == "starred"


# ---- журнал ----------------------------------------------------------------


def test_parallel_appends_do_not_lose_rows(registry: Path) -> None:
    """Два наглядачі можуть писати одночасно; append-only тим і обраний."""
    for i in range(50):
        add("ok", machine_id=1000 + i % 7)
    lines = (registry / "boxes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 50
    assert len(boxes.read_all()) == 50


def test_corrupt_line_is_skipped_not_fatal(registry: Path) -> None:
    add("ok")
    with open(registry / "boxes.jsonl", "a", encoding="utf-8") as fh:
        fh.write("обірваний рядок без json\n")
    add("slow_net")
    assert len(boxes.read_all()) == 2


def test_unknown_outcome_is_rejected_at_the_door() -> None:
    """Закритий перелік — бо на ньому тримаються TTL і кількість ударів."""
    with pytest.raises(ValueError, match="невідомий outcome"):
        BoxObservation(machine_id=1, outcome="щось_нове")


def test_neutral_outcomes_never_ban() -> None:
    """OOM калібрує константи, а не звинувачує машину."""
    for _ in range(5):
        add("oom_pages")
    assert boxes.banned_ids(now=NOW) == set()


def test_missing_journal_is_empty_not_an_error() -> None:
    assert boxes.read_all() == []
    assert boxes.banned_ids(now=NOW) == set()


def test_claimed_from_offer_normalizes_units() -> None:
    """Vast віддає VRAM і RAM у мегабайтах — звірка з `nvidia-smi` була б
    порівнянням гігабайтів із мегабайтами, і брехня хоста лишилась би непоміченою."""
    claimed = boxes.claimed_from_offer(
        {"gpu_ram": 32768, "cpu_ram": 102400, "cpu_cores_effective": 64.0, "dph_total": 0.222}
    )
    assert claimed["vram_gb"] == pytest.approx(32.0)
    assert claimed["ram_gb"] == pytest.approx(100.0)
    assert claimed["cores"] == 64.0


def test_explain_reads_like_a_sentence() -> None:
    add("slow_net", days_ago=2, detail="0.5 Мбіт/с проти обіцяних 755", cost_usd=0.11)
    text = boxes.explain(4711, now=NOW)
    assert "BANNED" in text
    assert "0.5 Мбіт/с" in text
    assert "$0.11" in text


def test_verdicts_from_the_broken_timers_are_void(monkeypatch) -> None:
    """🔴🔴 Замір міряльником із доведеною похибкою — не доказ.

    До 2026-08-12 усі три таймери підйому міряли ГОЛИЙ ЧАС, не дивлячись, чи
    хост посувається: 180 с на образ, 180 с на SSH після `running`, 90 с на
    прив'язку ключа. Реальні бокси піднімаються по 204 і 511 с — вони ще
    ставлять пакети, а SSH-демон уже відповідає, поки ключ туди не прокинуто.
    Система звинувачувала СПРАВНІ машини: 6 банів і 12 підозр за дві доби.

    Доказ, що вада наша: машини 144417 і 97081 мали УСПІШНІ прогони
    напередодні («391 з 391 сторінок, 0 OOM»), а назавтра дістали «хост
    відхилив наш ключ».
    """
    monkeypatch.setattr(boxes, "BROKEN_TIMER_EPOCH", NOW)
    add("ssh_auth_denied", days_ago=1)          # вироки ДО полагодження
    add("ssh_auth_denied", days_ago=0.5)
    assert boxes.banned_ids(now=NOW) == set(), "старий вирок таймера має бути недійсним"


def test_verdicts_after_the_fix_still_count(monkeypatch) -> None:
    """Скасування стосується ЛИШЕ старих замірів — інакше ми осліпли б назовсім."""
    monkeypatch.setattr(boxes, "BROKEN_TIMER_EPOCH", NOW - timedelta(days=2))
    add("ssh_auth_denied", days_ago=1)          # вироки ПІСЛЯ полагодження
    add("ssh_auth_denied", days_ago=0.5)
    assert boxes.banned_ids(now=NOW) == {4711}


def test_hardware_lies_are_not_amnestied(monkeypatch) -> None:
    """Брехня картки про залізо таймерів не стосується — вона лишається в силі."""
    monkeypatch.setattr(boxes, "BROKEN_TIMER_EPOCH", NOW)
    add("vram_lie", days_ago=1)                 # брехня заліза амністії не підлягає
    assert boxes.banned_ids(now=NOW) == {4711}


# ---- зайнята карта і невигідна машина (розбір 2026-08-19) ------------------


def test_busy_card_bans_from_the_first_hit_but_forgets_in_three_days() -> None:
    """🔴 Машина 139040 лишалась `starred` із п'ятьма успіхами, поки чужий
    процес тримав 19 з 24 ГБ її карти — і добір ніс її першою ж зіркою.

    Бан із першого удару (замір однозначний і зроблений до першої сторінки),
    але короткий: сусід піде, а хост і далі справний.
    """
    add("card_busy", machine_id=139040)
    assert boxes.verdicts(now=NOW)[139040].banned
    assert 139040 not in boxes.banned_ids(now=NOW + timedelta(days=4))


def test_overpriced_records_but_never_bans() -> None:
    """🔴 Ціна — рішення ПРО ЗАХІД, а не вирок машині: вона залежить від
    матеріалу (кадр-розворот тримає 4 шарди замість 8) і від стелі дослідника.

    Заміряно 2026-08-19: машина 14563 поїхала в бан на 3 дні за $0.201 проти
    стелі $0.200 — за пів відсотка, при власному розкиді моделі ±20%.
    """
    add("overpriced", machine_id=4711)
    assert not boxes.verdicts(now=NOW)[4711].banned


def test_later_success_clears_a_busy_card() -> None:
    """Карта звільнилась — вирок знімається, як і для решти нелипких."""
    add("card_busy", machine_id=4711, days_ago=1)
    add("ok", machine_id=4711)
    assert not boxes.verdicts(now=NOW)[4711].banned


def test_slow_for_data_never_bans() -> None:
    """Живий канал, повільний для ЦЬОГО обсягу кадрів, — не вада хоста."""
    add("slow_for_data", days_ago=0.1)
    add("slow_for_data")
    assert boxes.banned_ids(now=NOW) == set()
    assert boxes.verdicts(now=NOW)[4711].state != "warned"


def test_a_verdict_caused_by_our_code_can_be_absolved(tmp_path: Path) -> None:
    """Машину 45760 (24.09.2026) забанили як `below_target` за однопотоковий
    замір каналу — ваду воріт, а не хоста. Скасований запис не б'є, а журнал
    лишається як був."""
    from typer.testing import CliRunner

    from gpurunner.cli import app

    boxes.record(BoxObservation(machine_id=45760, outcome="below_target",
                                ts="2026-09-23T21:28:00+00:00", detail="однопотоковий замір"))
    now = datetime(2026, 9, 24, tzinfo=UTC)
    assert 45760 in boxes.banned_ids(now=now)
    journal = (tmp_path / "boxes.jsonl").read_text(encoding="utf-8")
    res = CliRunner().invoke(app, ["boxes", "absolve", "45760", "--since",
                                   "2026-09-23T00:00", "--why", "вада воріт ea7c05b"])
    assert res.exit_code == 0, res.output
    assert 45760 not in boxes.banned_ids(now=now)
    assert (tmp_path / "boxes.jsonl").read_text(encoding="utf-8") == journal
