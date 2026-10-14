"""ParseqTrainJob: параметри адаптації під залізо + цілісність рендера.

Ці перевірки існують, бо всі три відмови тут ТИХІ: невірний `amp_dtype`
проковтнувся б і трен поїхав би у fp32 удвічі повільніше; загублений при
редагуванні `_train_entry` дав би NameError аж на віддаленому кернелі, коли
GPU-квота вже витрачена на розпакування корпусу.
"""
from __future__ import annotations

import ast

import pytest

from gpurunner._embedded import parseq_train_runner as runner
from gpurunner.jobs.parseq_train import ParseqTrainJob


def _p(**kw):
    return ParseqTrainJob().validate_params({"dataset": "owner/slug", **kw})


def test_hardware_defaults_are_auto():
    p = _p()
    assert p["workers"] == 0          # 0 = визначити на місці
    assert p["ddp"] == "auto"         # усі видимі карти
    assert p["amp_dtype"] == "auto"   # bf16 на Ampere+, fp16 на T4
    assert p["lr_scale"] == "sqrt"
    assert p["compile"] is False


@pytest.mark.parametrize("bad", [
    {"amp_dtype": "int8"},
    {"lr_scale": "cosine"},
    {"ddp": "both"},
])
def test_bad_hardware_params_rejected(bad):
    with pytest.raises(ValueError):
        _p(**bad)


@pytest.mark.parametrize("val", ["auto", "off", "2"])
def test_ddp_accepts_auto_off_and_count(val):
    assert _p(ddp=val)["ddp"] == val


def test_workers_minus_one_means_none():
    """0 — автовизначення, тож «без воркерів» мусить мати окреме значення."""
    assert runner._auto_workers(-1, 1) == 0
    assert runner._auto_workers(3, 2) == 3
    auto_one = runner._auto_workers(0, 1)
    auto_two = runner._auto_workers(0, 2)
    assert auto_one >= 1 and auto_two >= 1
    # при DDP CPU ділиться між процесами, інакше воркери б'ються за ядра
    assert auto_two <= auto_one


def test_rendered_code_is_valid_and_complete():
    src = ParseqTrainJob().render_remote_code(_p(epochs=3))
    ast.parse(src)
    for needle in ("_train_entry", "DistributedDataParallel", "_StepWrap",
                   "broadcast", "max_memory_allocated"):
        assert needle in src, needle


@pytest.mark.parametrize("raw, want", [
    (True, True), (False, False),
    ("true", True), ("false", False),
    # 🔴 саме ці форми раніше давали True через bool("no"): спроба вимкнути
    # amp/profile мовчки вмикала їх, і побачити це можна було лише в лозі трену
    ("no", False), ("off", False), ("0", False), (0, False),
    ("yes", True), ("on", True), ("1", True),
])
def test_bool_params_accept_cli_spellings(raw, want):
    assert _p(amp=raw)["amp"] is want
    assert _p(profile=raw)["profile"] is want


@pytest.mark.parametrize("name", ["amp", "profile", "compile", "ddp_find_unused"])
def test_bool_params_reject_garbage(name):
    with pytest.raises(ValueError):
        _p(**{name: "maybe"})


@pytest.mark.parametrize("bad", [
    # від'ємний ліміт вмикав guard і зупиняв трен після першої ж епохи
    {"wall_limit_h": -1},
    {"patience": -1},
    {"warmup_pct": 1.5},
    {"charset_min_freq": 0},
    {"ddp_timeout_min": 0},
    {"lr": 0},
    {"min_delta": -0.1},
    {"limit": -5},
    {"max_steps": -1},
])
def test_out_of_range_params_rejected(bad):
    with pytest.raises(ValueError):
        _p(**bad)


def test_augmentation_is_not_pinned_to_sample_index():
    """Аугментація мусить різнитись між епохами.

    Раніше генератор сідувався індексом зразка (`Random(seed*k + i)`), тож у
    кожній епосі той самий кроп діставав ТУ САМУ трансформацію — 20 епох бачили
    один варіант замість двадцяти, і аугментація як регуляризатор не працювала.
    """
    src = ast.unparse(ast.parse(_runner_source()))
    assert "random.Random(" not in _getitem_source(), _getitem_source()
    assert "self.seed" not in src


def test_resume_requires_its_dataset():
    """Без датасету з parseq_last.pt продовжувати нічого — краще впасти локально."""
    with pytest.raises(ValueError):
        _p(resume=True)


def test_resume_dataset_is_mounted_as_input():
    p = _p(resume=True, resume_dataset="owner/prev-run")
    job = ParseqTrainJob()
    assert "owner/prev-run" in job.dataset_sources(
        {"dataset": "owner/slug", "resume": True, "resume_dataset": "owner/prev-run"})
    dirs = job.colab_input_dirs(
        {"dataset": "owner/slug", "resume": True, "resume_dataset": "owner/prev-run"})
    assert "prev-run" in dirs
    assert p["resume"] is True


def test_last_checkpoint_is_an_expected_output():
    """Інакше fetch --resume вважав би прогін забраним без файлу, потрібного далі."""
    assert "parseq_last.pt" in ParseqTrainJob().expected_outputs({"dataset": "o/s"})


def test_fingerprint_catches_changed_schedule():
    """Fingerprint мусить ловити все, що змінює форму ваг або розклад lr."""
    cfg = {"img_height": 32, "img_width": 128, "max_label_length": 25}
    base = runner._fingerprint(_p(), "абв", cfg, 1000)
    assert base == runner._fingerprint(_p(), "абв", cfg, 1000)   # детермінований
    for changed in (
        runner._fingerprint(_p(batch=32), "абв", cfg, 1000),
        runner._fingerprint(_p(epochs=30), "абв", cfg, 1000),
        runner._fingerprint(_p(lr=1e-4), "абв", cfg, 1000),
        runner._fingerprint(_p(), "абвг", cfg, 1000),            # інший charset
        runner._fingerprint(_p(), "абв", cfg, 999),              # інший обсяг train
    ):
        assert changed != base


def test_rendered_code_carries_resume_machinery():
    src = ParseqTrainJob().render_remote_code(_p())
    for needle in ("parseq_last.pt", "_load_resume", "_fingerprint",
                   'opt.load_state_dict', 'sched.load_state_dict', "start_epoch"):
        assert needle in src, needle


def test_xeval_runs_before_training_on_fresh_start():
    """Пакет заміру читається й ДО першої епохи (`xeval_ep00.tsv` — бойові ваги при
    дотренуванні), і лише на свіжому старті: при відновленні ep00 уже є з минулої сесії."""
    import inspect
    src = inspect.getsource(runner._train_entry)
    first_xeval = src.index("_xeval(")
    loop = src.index("for epoch in range(start_epoch")
    assert first_xeval < loop, "замір до трену мусить стояти перед циклом епох"
    head = src[src.rindex("if ", 0, first_xeval):first_xeval]
    assert "resume is None" in head and 'ctx.get("xeval")' in head, head
    assert src[first_xeval:src.index("\n", first_xeval)].rstrip().endswith(", 0)")


def _runner_source() -> str:
    import inspect
    return inspect.getsource(runner.LineDataset)


def _getitem_source() -> str:
    import inspect
    return inspect.getsource(runner.LineDataset.__getitem__)


def test_xeval_reads_with_workers_and_a_large_batch():
    """Замір — прямий прохід, але авторегресивний: з `num_workers=0` і батчем трену
    75 тис. кропів ішли 516 с на RTX 3090, повільніше за епоху трену на рядок."""
    import inspect
    src = inspect.getsource(runner._xeval)
    assert "num_workers=0" not in src, "кропи заміру не мають читатись у головному потоці"
    assert "num_workers=workers" in src and "_cpu_count()" in src
    assert "batch_size=max(256, batch)" in src


def test_eval_only_needs_models_not_a_corpus():
    p = ParseqTrainJob().validate_params({"eval_only": True, "model_dataset": "models",
                                          "eval_dataset": "xeval"})
    assert p["eval_only"] is True and p["dataset"] == ""
    assert set(ParseqTrainJob().colab_input_dirs({"eval_only": True, "model_dataset": "models",
                                                  "eval_dataset": "xeval"})) == {"models", "xeval"}
    with pytest.raises(ValueError, match="model_dataset"):
        ParseqTrainJob().validate_params({"eval_only": True})
    with pytest.raises(ValueError, match="dataset"):
        ParseqTrainJob().validate_params({})


def test_eval_only_branch_runs_before_any_corpus_reading():
    """eval_only не шукає маніфестів трену: інакше впав би на «не знайдено жодного маніфесту»."""
    import inspect
    src = inspect.getsource(runner.main)
    branch = src.index('if params.get("eval_only")')
    assert branch < src.index("_collect_manifests(roots)")
    ev = inspect.getsource(runner._eval_only)
    assert 'parseq_ep(\d+)\.pt$' in ev and '"parseq_last.pt", "parseq_best.pt"' in ev
    assert "_xeval(model, items" in ev
