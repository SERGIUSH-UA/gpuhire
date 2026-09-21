"""Бокс, що завис МІЖ справами, і стеля оренд, що рахує змарноване (06.09.2026).

Бойовий захід ф.230, сесія P2: перший бокс (CMP 170HX) пішов offline посеред
другої справи; другий дочитав першу справу з чекпоінтів R2 і завис на качанні
кадрів другої. Наглядач:

- годину не реагував — правило «сетап не посувається» діяло лише ДО першого
  прогресу, а прогрес від першої справи вже існував, тож чекав `setup_max_sec`;
- потім порахував продуктивний бокс як битий і вибив захід по стелі 2 оренд —
  сім справ (~3300 сторінок) лишились непрочитаними, доки людина не подивилась.

Третя вада — на боксі: `curl --speed-limit 10240` (10 КБ/с) не є підлогою
швидкості, і в фазі качання раннер не публікував жодного серцебиття.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import gpurunner._embedded.htr_case_runner as runner
from gpurunner.supervise.decide import SETUP_PHASES, Cfg, Obs, decide
from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import CasePlan, Plan

CFG = Cfg(budget_usd=3.0, max_hours=8.0)


# ---- наглядач: зависання між справами -----------------------------------------


@pytest.mark.parametrize("phase", ["starting", "downloading", "unpacking"])
def test_a_stalled_setup_between_cases_is_rerented_not_waited_out(phase: str) -> None:
    """Прогрес від попередньої справи є, фаза — сетап наступної, лог не росте
    16 хв → переоренда, а не година очікування."""
    action, why = decide(Obs(progress={"phase": phase, "pages_done": 1179},
                             progress_age_sec=200, ssh_ok=True,
                             setup_stall_sec_seen=16 * 60, dph=0.3), CFG)
    assert action == "rerent"
    assert "не посувається" in why


def test_a_running_fleet_is_not_judged_by_the_setup_log() -> None:
    """Під час читання лог раннера може мовчати годинами — шарди пишуть у свої
    логи. Правило сетапу на `running` не поширюється."""
    action, _ = decide(Obs(progress={"phase": "running", "pages_done": 10,
                                     "n_pages_expected": 100, "pages_per_hour": 500,
                                     "wall_sec": 1800.0},
                           progress_age_sec=30, ssh_ok=True,
                           setup_stall_sec_seen=3 * 3600, dph=0.3), CFG)
    assert action == "wait"


def test_setup_phases_cover_everything_before_the_shards() -> None:
    assert set(SETUP_PHASES) == {"", "starting", "downloading", "unpacking"}


# ---- наглядач: стеля рахує лише змарновані оренди ---------------------------


def _plan(tmp_path: Path) -> Plan:
    return Plan(assets_url="https://r2/a.tgz", budget_usd=3.0, max_hours=8.0, max_rents=2,
                cases=[CasePlan(case="230-1-18", pages_url="https://r2/p.tar", n_pages=1185,
                                out_dir=str(tmp_path / "230-1-18"))])


def test_a_productive_box_does_not_count_against_the_rent_cap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    sup = Supervisor(_plan(tmp_path), backend=object(), session="S")  # type: ignore[arg-type]
    # оренда 1: CMP помер, нуль сторінок
    sup.rents = 1
    sup._rent_open = True
    sup._box_pages_done = 0
    sup._close_rent()
    assert sup.rents_wasted == 1
    # оренда 2: дочитала справу з чекпоінтів (1179 сторінок) і зависла
    sup.rents = 2
    sup._rent_open = True
    sup._box_pages_done = 1179
    sup._close_rent()
    assert sup.rents_wasted == 1, "продуктивний бокс — не змарнована оренда"
    assert not sup._rent_cap_hit(), "третя оренда мусить бути дозволена"
    # але грошова межа лишається: 2 × max_rents усього
    sup.rents = 4
    assert sup._rent_cap_hit()


def test_two_dead_boxes_still_hit_the_cap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    sup = Supervisor(_plan(tmp_path), backend=object(), session="S")  # type: ignore[arg-type]
    for n in (1, 2):
        sup.rents = n
        sup._rent_open = True
        sup._box_pages_done = 5          # менше за PRODUCTIVE_PAGES
        sup._close_rent()
    assert sup.rents_wasted == 2 and sup._rent_cap_hit()
    assert sup._budget_view()["rents_wasted"] == 2


def test_closing_twice_does_not_count_the_same_rent_twice(tmp_path: Path, monkeypatch) -> None:
    """🔴 spr-11652, 14.09.2026: невдала спроба (V100, відхилений ключ) рахувала
    себе сама, а наступний успішний сабміт закривав її ВДРУГЕ — `rents_wasted 2`
    при одній реальній втраті й стелі 3. Закривається лише відкрита оренда."""
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    sup = Supervisor(_plan(tmp_path), backend=object(), session="S")  # type: ignore[arg-type]
    sup._box_pages_done = 0
    sup._close_rent()                    # немає відкритої оренди — нічого
    assert sup.rents_wasted == 0
    sup._rent_open = True
    sup._close_rent()
    sup._close_rent()                    # та сама оренда вдруге — нічого
    assert sup.rents_wasted == 1


# ---- бокс: підлога швидкості й серцебиття качання ----------------------------


def test_download_floor_never_drops_to_ten_kilobytes() -> None:
    """Підлога тепер від виміряного каналу, але нижче «мертвого» ~1 Мбіт/с не
    опускається: 10 КБ/с були не підлогою, а її відсутністю."""
    assert runner.DOWNLOAD_DEAD_BPS >= 100_000
    assert runner.DOWNLOAD_MIN_BPS >= 1_000_000
    src = Path(runner.__file__).read_text(encoding="utf-8")
    body = src.split("def _download_with_heartbeat(")[1].split("\ndef ")[0]
    assert "_download_floor(attempt)" in body and '"downloading"' in body


def test_download_publishes_a_heartbeat_while_curl_runs(tmp_path: Path, monkeypatch) -> None:
    """Доки curl качає, прогрес мусить оновлюватись — інакше для наглядача
    качання невідрізненне від мертвого раннера."""
    phases: list = []
    monkeypatch.setattr(runner, "_set_phase", lambda phase, **kw: phases.append((phase, kw)))
    monkeypatch.setattr(runner, "DOWNLOAD_HEARTBEAT_SEC", 0.01)
    local = tmp_path / "pages.tar"

    class _Proc:
        def __init__(self):
            self.polls = 0

        def wait(self, timeout=None):
            self.polls += 1
            if self.polls < 3:
                local.write_bytes(b"x" * (self.polls * 100))
                raise runner.subprocess.TimeoutExpired("curl", timeout)
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: _Proc())
    assert runner._download_with_heartbeat("https://r2/p.tar", local, "pages.tar") == 0
    assert [p for p, _ in phases] == ["downloading", "downloading"]
    assert phases[0][1]["bytes"] == 100 and phases[1][1]["bytes"] == 200
    assert phases[1][1]["stalled"] is False


def test_a_failed_download_is_retried_once_from_scratch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_set_phase", lambda phase, **kw: None)
    monkeypatch.setattr(runner, "DOWNLOAD_RETRY_SLEEP_SEC", 0)
    local = tmp_path / "pages.tar"
    calls = {"n": 0}

    class _Proc:
        def wait(self, timeout=None):
            calls["n"] += 1
            local.write_bytes(b"half")
            return 28 if calls["n"] == 1 else 0     # 28 = speed-limit / timeout

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: _Proc())
    assert runner._download_with_heartbeat("https://r2/p.tar", local, "pages.tar") == 0
    assert calls["n"] == 2


# ---- наглядач: гроші на цю оренду проти сторінок, які вона дала -------------


def _running(**kw) -> dict:
    base = {"phase": "running", "n_pages_expected": 4477, "pages_done": 1179,
            "pages_per_hour": 5600, "missing_count": 0, "wall_sec": 1800.0}
    base.update(kw)
    return base


def test_a_box_that_costs_twice_what_it_showed_is_rerented() -> None:
    """P2 06.09.2026: бокс дав 1179 сторінок за ~14 хв ($0.07), потім завис —
    прогрес завмер на 5600 стор/год, самозвітна ціна лишалась гарною, а гроші
    йшли. Через годину оренда коштувала $0.37 за ті самі 1179 сторінок."""
    good = Obs(progress=_running(), progress_age_sec=30, ssh_ok=True, dph=0.297,
               rent_age_sec=35 * 60, rent_settled_sec=25 * 60, rent_usd=0.12,
               rent_pages=1179, box_best_usd_per_1000=0.053)
    assert decide(good, CFG)[0] == "wait"
    hung = Obs(progress=_running(), progress_age_sec=30, ssh_ok=True, dph=0.297,
               rent_age_sec=85 * 60, rent_settled_sec=75 * 60, rent_usd=0.37,
               rent_pages=1179, box_best_usd_per_1000=0.053)
    action, why = decide(hung, CFG)
    assert action == "rerent"
    assert "за 1000 сторінок" in why and "дорожче" in why


def test_the_money_rule_waits_for_the_rent_to_mature() -> None:
    """20 хв від дозрівання темпу — одна важка справа ще не переплата."""
    young = Obs(progress=_running(pages_done=40), progress_age_sec=30, ssh_ok=True,
                dph=0.297, rent_age_sec=30 * 60, rent_settled_sec=10 * 60,
                rent_usd=0.05, rent_pages=40, box_best_usd_per_1000=0.05)
    assert decide(young, CFG)[0] == "wait"


def test_without_its_own_measured_price_the_rule_is_silent() -> None:
    """🔴 904-24-70, 11.09.2026, бокс №1: живий темп 3759 стор/год ($0.078 за
    1000 при прогнозі $0.067), але 20 хв оренди разом із сетапом дали 705 стор.
    за $0.10 — «$0.140, 2.1× дорожче» проти прогнозу, і здоровий бокс погасили.
    Поки темп не дозрів, власної ціни немає, і правило мовчить."""
    obs = Obs(progress=_running(pages_done=705, pages_per_hour=3759, wall_sec=700.0),
              progress_age_sec=30, ssh_ok=True, dph=0.294, rent_age_sec=20 * 60)
    assert decide(obs, CFG)[0] == "wait"


def test_a_total_hang_after_maturity_is_still_caught() -> None:
    """Жодної нової сторінки після дозрівання — ціна тисячі йде вгору без меж."""
    obs = Obs(progress=_running(), progress_age_sec=30, ssh_ok=True, dph=0.297,
              rent_age_sec=60 * 60, rent_settled_sec=30 * 60, rent_usd=0.15,
              rent_pages=0, box_best_usd_per_1000=0.053)
    assert decide(obs, CFG)[0] == "rerent"


def test_supervisor_counts_the_money_from_the_matured_tempo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    plan = Plan(assets_url="https://r2/a.tgz", budget_usd=3.0, max_hours=8.0, max_rents=2,
                cases=[CasePlan(case=c, pages_url="https://r2/p.tar", n_pages=n,
                                out_dir=str(tmp_path / c))
                       for c, n in (("904-24-70", 1091), ("904-24-182", 521))])
    sup = Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]
    usd = {"v": 0.04}
    monkeypatch.setattr(sup, "_current_rent_usd", lambda: usd["v"])
    # сетап і розгін: темп не дозрів — нулі, бази немає
    assert sup._rent_money({"case_index": 1, "pages_done": 300}, settled=False, now=100.0) \
        == (0.0, 0, 0.0)
    assert sup._rent_base is None
    # дозрів: тут стає база
    assert sup._rent_money({"case_index": 1, "pages_done": 300}, settled=True, now=200.0) \
        == (0.0, 0, 0.0)
    usd["v"] = 0.10
    got = sup._rent_money({"case_index": 1, "pages_done": 900}, settled=True, now=1400.0)
    assert got[0] == pytest.approx(0.06) and got[1:] == (600, 1200.0)
    # друга справа черги: посправний лічильник упав до 10, сторінки оренди — ні
    got = sup._rent_money({"case_index": 2, "pages_done": 10}, settled=False, now=1500.0)
    assert got[1] == 1091 + 10 - 300
    sup._box_best_usd_per_1000 = 0.05
    sup._box_pages_done = 500
    sup._close_rent()
    assert sup._rent_base is None
    assert sup._box_best_usd_per_1000 == 0.0 and sup._box_pages_done == 0
