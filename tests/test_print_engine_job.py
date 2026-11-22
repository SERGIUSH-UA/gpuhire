"""Рушій друку в job'і `htr_case`: опис, рендер і розвилка бокс-раннера.

Друк живе ЛИШЕ на орендованій машині: робітник (`_embedded/print_vl_worker.py`)
їде всередині job.py, а не в ассетах Нишпорки, і читає лише конвеєром.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gpurunner._embedded import htr_case_runner as runner
from gpurunner.jobs.htr_case import ENGINE_PRINT, PRINT_MODEL, HTRCaseJob

BASE = {"pages_dataset": "u/pages", "models_dataset": "u/models", "scripts_dataset": "u/scripts"}


def test_print_engine_defaults_to_print_script_and_model() -> None:
    out = HTRCaseJob().validate_params({**BASE, "engine": ENGINE_PRINT})

    assert (out["engine"], out["script"], out["model"]) == (ENGINE_PRINT, "print", PRINT_MODEL)
    assert HTRCaseJob().validate_params(out) == out, "валідація ідемпотентна"


def test_handwriting_is_untouched() -> None:
    out = HTRCaseJob().validate_params(BASE)

    assert (out["engine"], out["script"]) == ("", "cyrillic")
    assert out["model"] == "pysar_cyr_v17.pt"


@pytest.mark.parametrize("params, msg", [
    ({"engine": "tesseract"}, "engine must be"),
    ({"engine": ENGINE_PRINT, "script": "cyrillic"}, "reads print only"),
    ({"script": "print"}, "script must be"),
])
def test_wrong_engine_or_script_is_refused(params: dict, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        HTRCaseJob().validate_params({**BASE, **params})


def test_worker_rides_inside_job_py_only_for_print() -> None:
    printing = HTRCaseJob().render_remote_code({**BASE, "engine": ENGINE_PRINT})
    handwriting = HTRCaseJob().render_remote_code(BASE)

    marker = "Робітник друку: PaddleOCR-VL через vLLM-сервер"
    assert marker in printing, "текст робітника їде в параметрах job.py"
    assert marker not in handwriting


def test_box_reads_print_from_the_validated_engine() -> None:
    out = HTRCaseJob().validate_params({**BASE, "engine": ENGINE_PRINT})

    assert runner._is_print(out) and not runner._is_print(HTRCaseJob().validate_params(BASE))
    assert runner.PRINT_MODEL == PRINT_MODEL, "модель мети робітника = модель плану"


def test_single_case_path_refuses_print_out_loud() -> None:
    with pytest.raises(RuntimeError, match="лише конвеєром"):
        runner._run_case({"engine": ENGINE_PRINT})


# ── кошторис і ворота друку ──────────────────────────────────────────────────
def test_print_sizing_is_one_worker_per_24gb_card() -> None:
    from gpurunner.core import htr_sizing as hs

    good = hs.plan_sizing(cores=12, vram_gb=24, gpu_name="RTX 3090", engine=ENGINE_PRINT)
    small = hs.plan_sizing(cores=12, vram_gb=16, gpu_name="RTX 4080S", engine=ENGINE_PRINT)

    assert (good.shards, good.pages_per_hour) == (1, 900.0) and good.usable
    assert not small.usable, "vLLM із моделлю не влазить у 16 ГБ"
    assert hs.print_pph_for("NVIDIA GeForce RTX 4090") == 1300.0
    assert hs.print_pph_for("Quadro P6000") == hs.PRINT_PPH_DEFAULT


def test_print_estimate_for_the_newspaper_batch() -> None:
    """1 789 номерів / 4 064 сторінки на 3090 по $0.19: години й долари, а не доба."""
    from gpurunner.core import htr_sizing as hs

    sz = hs.plan_sizing(cores=12, vram_gb=24, gpu_name="RTX 3090", engine=ENGINE_PRINT)
    hours = hs.predict_hours(4064, sz, cases=1789)
    kraken_overhead_h = 1789 * hs.PER_CASE_SEC / 3600

    assert 5.0 < hours < 8.0
    assert hs.predict_cost(4064, sz, 0.19, cases=1789) < 2.0
    assert hours < kraken_overhead_h, "накладні друку не рахуються 27 с на справу"


def test_print_ignores_kraken_measurements_of_the_machine() -> None:
    from gpurunner.core.offer_score import Need, measured_pph_for

    measured = {"pages_per_hour": 4000, "cores": 12, "vram_total_gb": 24,
                "pages_per_hour_mpx": 9, "pages_per_hour_lines": 40}
    need = Need(pages=10, max_hours=5, budget_usd=1, engine=ENGINE_PRINT)

    assert measured_pph_for(measured, need, cores=12, vram_gb=24, num_gpus=1) == 0.0


def test_print_plan_rides_the_measured_image() -> None:
    from gpurunner.backends.vast import PRINT_IMAGE, image_for, min_cuda_for

    assert image_for({"engine": ENGINE_PRINT}) == PRINT_IMAGE
    assert min_cuda_for({"engine": ENGINE_PRINT}) == 12.6
    assert image_for({"engine": ENGINE_PRINT, "image": "x/y:1"}) == "x/y:1"


def test_print_plan_has_no_handwriting_pace_target() -> None:
    """Ціль темпу міряна kraken'ом: друк її не дасть, і захід чекав би ринку вічно."""
    from gpurunner.supervise.plan import DEFAULT_TARGET_PPH, _target_pph

    assert _target_pph({"params": {"engine": ENGINE_PRINT}}) == 0.0
    assert _target_pph({"params": {"engine": ENGINE_PRINT, "target_pph": 500}}) == 500.0
    assert _target_pph({"params": {}}) == DEFAULT_TARGET_PPH


def test_print_has_its_own_price_ceiling_per_1000_pages() -> None:
    """Стеля рукопису ($0.20) відсікала весь ринок датацентрів для друку."""
    from gpurunner.core.offer_score import MAX_USD_PER_1000_PRINT, Need

    printing = Need(pages=10, max_hours=5, budget_usd=1, engine=ENGINE_PRINT)
    explicit = Need(pages=10, max_hours=5, budget_usd=1, engine=ENGINE_PRINT,
                    max_usd_per_1000_pages=0.3)

    assert printing.usd_per_1000_ceiling == MAX_USD_PER_1000_PRINT
    assert explicit.usd_per_1000_ceiling == 0.3
    assert Need(pages=10, max_hours=5, budget_usd=1).usd_per_1000_ceiling < MAX_USD_PER_1000_PRINT


def test_print_market_search_does_not_demand_kraken_cores(monkeypatch) -> None:
    """Датацентри продають 3090/A5000 із 6 ядрами — підлога 8 ядер їх відсікала."""
    from gpurunner.backends import vast as V
    from gpurunner.core import boxes
    from gpurunner.core.offer_score import Need

    asked: list[float] = []

    def fake_search(self, **kw):
        asked.append(float(kw.get("min_cpu") or 0))
        return []

    monkeypatch.setattr(V.VastBackend, "search_offers", fake_search)
    monkeypatch.setattr(V.VastBackend, "__init__", lambda self: None)
    monkeypatch.setattr(boxes, "verdicts", lambda now=None: {})
    monkeypatch.setattr(boxes, "starred", lambda **kw: [])
    b = V.VastBackend()
    b.find_candidates(gpu="any", need=Need(pages=4, max_hours=4, budget_usd=1,
                                           engine=ENGINE_PRINT))
    printing = list(asked)
    asked.clear()
    b.find_candidates(gpu="any", need=Need(pages=4, max_hours=4, budget_usd=1))

    assert printing and max(printing) == 4.0
    assert asked and min(asked) >= 8.0


def test_print_probes_the_network_with_a_real_archive() -> None:
    """Ассети друку крихітні (ваг немає): проба на них міряла рукостискання."""
    from types import SimpleNamespace

    from gpurunner.supervise.htr import net_probe_url

    cases = [SimpleNamespace(pages_url="https://r2/a.tar", pages_bytes=5_000_000),
             SimpleNamespace(pages_url="https://r2/b.tar", pages_bytes=19_000_000)]
    printing = SimpleNamespace(params={"engine": ENGINE_PRINT}, assets_url="https://r2/as.tgz",
                               cases=cases)
    handwriting = SimpleNamespace(params={}, assets_url="https://r2/as.tgz", cases=cases)

    assert net_probe_url(printing) == "https://r2/b.tar"
    assert net_probe_url(handwriting) == "https://r2/as.tgz"


# ── граблі першого живого заходу друку (12.10.2026) — кожна закрита тут ──────
def test_print_env_installs_in_the_measured_order_not_one_resolution() -> None:
    """Paddle 3.3.1 пінить CUDA 12.6, torch 2.8 — 12.8: спільне розв'язання `uv`
    каже «unsatisfiable». Заміряне середовище ставилось почергово."""
    stages = [what for what, _, _ in runner.PRINT_STAGES]
    reqs = [r for _, rr, _ in runner.PRINT_STAGES for r in rr]

    assert stages[0] == "Paddle" and stages[-1].startswith("vLLM")
    assert runner.PRINT_STAGES[0][1] == ("paddlepaddle-gpu==3.3.1",)
    assert any(r.startswith("paddlex[genai-vllm-server]") for r in runner.PRINT_STAGES[-1][1])
    assert all(len(rr) < len(reqs) for _, rr, _ in runner.PRINT_STAGES), "жодного кроку на все"
    src = Path(runner.__file__).read_text(encoding="utf-8")
    assert "for k, (what, reqs, paddle_index) in enumerate(PRINT_STAGES, 1)" in src


def test_print_rides_the_image_cached_on_hosts() -> None:
    """`pytorch/pytorch:2.6.0` хости тягнули по 6 хв — `slow_boot` на кожній оренді."""
    from gpurunner.backends.vast import ENGINE_IMAGE, PRINT_IMAGE

    assert PRINT_IMAGE == ENGINE_IMAGE


def test_print_is_not_rejected_for_the_price_of_a_core() -> None:
    """Датацентрова 3090 за $0.30 з 6 ядрами — $0.05/ядро-год, а стеля kraken $0.02."""
    from gpurunner.core.offer_score import Need, score_offer

    offer = {"id": 1, "machine_id": 7, "gpu_name": "RTX 3090", "num_gpus": 1,
             "gpu_ram": 24576, "cpu_cores_effective": 6.0, "dph_total": 0.30,
             "reliability2": 0.99, "inet_down": 5000, "inet_down_cost": 0.0}
    printing = score_offer(offer, Need(pages=4000, max_hours=24, budget_usd=5,
                                       engine=ENGINE_PRINT))
    handwriting = score_offer(offer, Need(pages=4000, max_hours=24, budget_usd=5))

    assert not any("ядро-годину" in r for r in printing.rejects), printing.rejects
    assert any("ядро-годину" in r for r in handwriting.rejects)


def test_print_plan_rents_disk_for_the_print_environment() -> None:
    """План на 15 ГБ під kraken: установка друку впала з «No space left on device»."""
    from gpurunner.supervise.plan import PRINT_DISK_GB, _disk_gb

    assert _disk_gb({"disk_gb": 15, "params": {"engine": ENGINE_PRINT}}) == PRINT_DISK_GB
    assert _disk_gb({"disk_gb": 80, "params": {"engine": ENGINE_PRINT}}) == 80
    assert _disk_gb({"disk_gb": 15, "params": {}}) == 15
    src = Path(runner.__file__).read_text(encoding="utf-8")
    assert '"pip", "install", "--no-cache"' in src, "колеса друку не мають лежати двічі"
