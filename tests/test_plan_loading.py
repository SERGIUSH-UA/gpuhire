"""Що план каже про себе — і про що він мовчав.

🔴 Клас вади один на всі тести цього файла: ключ, покладений не туди, не давав
ні помилки, ні попередження. Захід їхав із дефолтом, і видно це було лише за
наслідками — OOM, перевитрата, година очікування ринку. Найгірше, що поруч у
тому самому плані `max_usd_per_1000_pages` працював, тож перевірка «стеля
підхопилась, значить план читається» давала хибну впевненість.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpurunner.supervise.plan import check_plan_keys, load_plan

BASE = {
    "assets_url": "https://r2/assets.tgz",
    "budget_usd": 3.0,
    "max_hours": 8.0,
    "cases": [{"case": "spr-1", "pages_url": "https://r2/x.tar", "n_pages": 100,
               "out_dir": "E:/prostir/reports/htr/spr-1"}],
}


def _write(tmp_path: Path, **extra) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({**BASE, **extra}, ensure_ascii=False), encoding="utf-8")
    return path


def test_vram_gb_per_shard_is_read_from_the_top_level_too(tmp_path: Path) -> None:
    """🔴🔴 Скіл називає ручку `vram_gb_per_shard` і показує її «в плані», а
    план читав ЛИШЕ `gb_per_shard`. Три заходи поспіль упали з OOM (34 і 53
    події), ручку двічі виставили «як у скілі» й двічі дістали «авто»."""
    plan = load_plan(_write(tmp_path, vram_gb_per_shard=4.5))
    assert plan.gb_per_shard == 4.5


def test_gb_per_shard_still_wins_when_both_names_are_given(tmp_path: Path) -> None:
    plan = load_plan(_write(tmp_path, gb_per_shard=3.0, vram_gb_per_shard=4.5))
    assert plan.gb_per_shard == 3.0


def test_shards_from_the_top_level_is_not_ignored(tmp_path: Path) -> None:
    """`shards` із верхнього рівня не читався ВЗАГАЛІ — ані планом, ані через
    params."""
    assert load_plan(_write(tmp_path, shards=6)).shards == 6


def test_unknown_top_level_key_is_named_out_loud(tmp_path: Path) -> None:
    plan = load_plan(_write(tmp_path, vram_per_shard=4.5))
    assert any("vram_per_shard" in w and "НЕ ДІЄ" in w for w in plan.warnings)


def test_plan_key_put_into_params_is_named_out_loud() -> None:
    """🔴 `-p max_rents=8` кладе ключ у `params`, звідки він не читається
    ніколи (17.08.2026). Те саме з `-p max_usd_per_1000_pages` (19.08.2026)."""
    warnings = check_plan_keys({**BASE, "params": {"max_rents": 8}})
    assert any("max_rents" in w and "params" in w for w in warnings)


def test_a_real_param_knob_in_params_is_not_flagged() -> None:
    """`vram_gb_per_shard` у `params` — законне місце: саме так його задає `-p`."""
    warnings = check_plan_keys({**BASE, "params": {"vram_gb_per_shard": 4.5}})
    assert not warnings


def test_unknown_case_key_is_named_with_the_case(tmp_path: Path) -> None:
    plan = load_plan(_write(tmp_path, cases=[{**BASE["cases"][0], "out_root": "x"}]))
    assert any("spr-1" in w and "out_root" in w for w in plan.warnings)


def test_a_clean_plan_warns_about_nothing(tmp_path: Path) -> None:
    assert load_plan(_write(tmp_path)).warnings == []


def test_unknown_key_is_a_warning_not_a_refusal(tmp_path: Path) -> None:
    """Падати не можна: план могли зробити новішим планувальником, і невідомий
    ключ не привід не орендувати бокс. Але промовчати теж не можна."""
    plan = load_plan(_write(tmp_path, something_new=1))
    assert plan.warnings and plan.total_pages == 100


def test_a_broken_plan_still_refuses_before_any_money(tmp_path: Path) -> None:
    """Попередження — не заміна перевіркам, які коштують нуль тут і гроші на боксі."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({**BASE, "cases": [
        {"case": "spr-1", "pages_url": "https://r2/x.tar", "n_pages": 0, "out_dir": "x"}]}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="Знаменник"):
        load_plan(path)


# ---- режим заходу: швидко чи дешево ---------------------------------------------


def test_a_target_named_with_p_beats_the_one_the_builder_wrote(tmp_path: Path) -> None:
    """🔴🔴 29.09.2026: три заходи пущено з `-p target_pph=0`, складач при цьому
    записав зверху свої 5000 — і діяли вони. Наглядач гасив бокси «за недобір
    цілі», якої людина не ставила."""
    plan = load_plan(_write(tmp_path, target_pph=5000.0, params={"target_pph": "0"}))
    assert plan.target_pph == 0.0


def test_pace_cheap_means_no_target(tmp_path: Path) -> None:
    from_builder = load_plan(_write(tmp_path, pace="cheap", target_pph=0.0))
    assert (from_builder.pace, from_builder.target_pph) == ("cheap", 0.0)
    # Так режим кладе обгортка: складач пише свій дефолт, людина — `-p pace=cheap`.
    from_p = load_plan(_write(tmp_path, pace="fast", target_pph=5000.0,
                              params={"pace": "cheap"}))
    assert (from_p.pace, from_p.target_pph) == ("cheap", 0.0)
    bare = load_plan(_write(tmp_path, pace="cheap"))
    assert bare.target_pph == 0.0


def test_pace_fast_is_the_default_and_keeps_the_target(tmp_path: Path) -> None:
    from gpurunner.core.offer_score import DEFAULT_TARGET_PPH

    plan = load_plan(_write(tmp_path))
    assert (plan.pace, plan.target_pph) == ("fast", DEFAULT_TARGET_PPH)
    assert not [w for w in load_plan(_write(tmp_path, pace="fast")).warnings if "pace" in w]


def test_an_explicit_target_survives_pace_cheap(tmp_path: Path) -> None:
    plan = load_plan(_write(tmp_path, params={"pace": "cheap", "target_pph": "3000"}))
    assert (plan.pace, plan.target_pph) == ("cheap", 3000.0)


def test_an_unknown_pace_is_refused_before_the_rent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pace"):
        load_plan(_write(tmp_path, params={"pace": "швидко"}))
