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
