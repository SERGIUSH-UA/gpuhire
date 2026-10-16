"""KrakenTrainJob: валідація параметрів + цілісність рендера.

Job не мав жодного тесту, хоча саме тут відмови найдорожчі: ketos приймає
`--precision garbage` і `-d gpu9` мовчки, тож помилка в одному символі
виявлялася аж на кернелі — після того, як GPU-сесія вже стартувала.
"""
from __future__ import annotations

import ast

import pytest

from gpurunner.jobs.kraken_train import KrakenTrainJob


def _p(**kw):
    return KrakenTrainJob().validate_params({"dataset": "owner/arrows", **kw})


def test_defaults_are_the_tuned_ones():
    p = _p()
    # lag 3 + min_delta 0.002 — подвійна засувка проти трену на плато
    assert p["lag"] == 3
    assert p["min_delta"] == 0.002
    # wall-limit як страховка, коли early stopping не спрацював
    assert p["wall_limit_h"] == 3.0
    # auto → `-d cuda:0,cuda:1`, тобто обидві карти Kaggle
    assert p["devices"] == "auto"
    # без model_dataset job мусить підставити VGSL-топологію для трену з нуля
    assert p["spec"].startswith("[1,120,0,1")


def test_spec_not_injected_when_base_model_given():
    assert _p(model_dataset="owner/mlmodel")["spec"] == ""


@pytest.mark.parametrize("bad", [
    {"resize": "grow"},
    {"batch": 0},
    {"precision": "garbage"},   # ketos ковтає і падає вже на кернелі
    {"workers": -3},
    {"epochs": 0},
    {"min_epochs": -1},
    {"lag": -1},
    {"wall_limit_h": -1},       # раніше вмикало guard і різало трен одразу
    {"lr": 0},
    {"min_delta": -0.5},
])
def test_bad_params_rejected(bad):
    with pytest.raises(ValueError):
        _p(**bad)


@pytest.mark.parametrize("val", ["32", "16-mixed", "bf16-mixed"])
def test_precision_accepts_lightning_spellings(val):
    assert _p(precision=val)["precision"] == val


def test_epochs_minus_one_means_unbounded():
    assert _p(epochs=-1)["epochs"] == -1


def test_dataset_is_required():
    with pytest.raises(ValueError):
        KrakenTrainJob().validate_params({})


def test_rendered_code_is_valid_and_complete():
    src = KrakenTrainJob().render_remote_code(_p())
    ast.parse(src)
    assert "if __name__" not in src        # __main__-блок мусить бути зрізаний
    for needle in (
        "_run_ketos",          # спільний цикл замість дубля під фолбек
        "_watchdog",           # guard тредом: ловить і зависання ketos
        "ktrain_ddp_attempt",  # лог невдалої DDP-спроби не затирається
        "_ketos_entry.py",     # файл-лаунчер замість python -c
    ):
        assert needle in src, needle


def test_min_delta_has_single_source_of_truth():
    """Раннер мусить брати min_delta з PARAMS, а не мати власний дефолт.

    Два джерела істини вже розійшлися були (0.001 у раннері проти 0.002 у job),
    і виграв би той, хто мовчки підставиться при відсутньому ключі.
    """
    src = KrakenTrainJob().render_remote_code(_p())
    assert 'params["min_delta"]' in src
    assert 'params.get("min_delta"' not in src


def test_kraken_version_is_pinned_and_reaches_the_runner():
    """Vast ставить kraken у порожній образ: без піна `>=5.0` взяв би будь-яку
    новішу за перевірену 7.1.1 (PP-OCRv6), і трен змінився б мовчки."""
    assert _p()["kraken_version"] == "7.1.1"
    src = KrakenTrainJob().render_remote_code(_p())
    assert 'f"kraken=={version}"' in src
    assert 'params.get("kraken_version")' in src


# ── пакет заміру на боксі ─────────────────────────────────────────────────────
def _bundle_tgz(where, name):
    import io
    import tarfile
    where.mkdir(parents=True, exist_ok=True)
    with tarfile.open(where / name, "w:gz") as tf:
        data = b"g-1/0001/line_000.png\tg-1\t0001\t0\n"
        ti = tarfile.TarInfo("xeval/xeval.index")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))


def _fake_tar(monkeypatch, runner):
    """Системний `tar` на Windows не однаковий; розпаковуємо тим самим архівом через tarfile."""
    import tarfile

    def run(cmd, check=False, **kw):
        assert cmd[:2] == ["tar", "xzf"], cmd
        with tarfile.open(cmd[2]) as tf:
            tf.extractall(cmd[4])
    monkeypatch.setattr(runner.subprocess, "run", run)


def test_xeval_bundle_found_under_any_archive_name(tmp_path, monkeypatch):
    """Пакет MeGen зветься `_xeval_xix.tgz`: маска `xeval*.tgz` його не бачила,
    і замір мовчки пропускався — трен віддавав епохи без жодного xeval_epNN.tsv."""
    from gpurunner._embedded import kraken_train_runner as runner
    monkeypatch.setattr(runner, "KAGGLE_INPUT", tmp_path / "input")
    monkeypatch.setattr(runner, "WORK", tmp_path / "work")
    _bundle_tgz(tmp_path / "input" / "xeval", "_xeval_xix.tgz")
    _fake_tar(monkeypatch, runner)
    idx = runner._xeval_index()
    assert idx is not None and idx.name == "xeval.index"
    assert idx.read_text(encoding="utf-8").startswith("g-1/")


def test_xeval_reads_the_base_before_the_epochs(tmp_path, monkeypatch):
    """База дотренування міряється на тому самому боксі: `xeval_base.tsv` поруч з епохами."""
    from gpurunner._embedded import kraken_train_runner as runner
    work = tmp_path / "working"
    work.mkdir()
    for n in ("epoch_00.safetensors", "epoch_01.safetensors"):
        (work / n).write_bytes(b"w")
    base = tmp_path / "input" / "diak_cyr_v6.safetensors"
    monkeypatch.setattr(runner, "KAGGLE_WORKING", work)
    got = [(p.name, out) for p, out in runner._xeval_targets(base)]
    assert got == [("diak_cyr_v6.safetensors", "xeval_base.tsv"),
                   ("epoch_00.safetensors", "xeval_ep00.tsv"),
                   ("epoch_01.safetensors", "xeval_ep01.tsv")]
    assert [out for _, out in runner._xeval_targets()] == ["xeval_ep00.tsv", "xeval_ep01.tsv"], \
        "без бази (eval_only, трен з нуля) — лише епохи"


def test_training_passes_the_base_to_xeval():
    import inspect

    from gpurunner._embedded import kraken_train_runner as runner
    src = inspect.getsource(runner.main)
    assert "_xeval_epochs(int(params.get(\"xeval_batch\", 1)), base=base_model)" in src


def test_xeval_pages_are_split_evenly_and_completely():
    """Сторінки розкладено по процесах: кожна рівно раз, найбільші першими по колу."""
    from gpurunner._embedded import kraken_train_runner as runner
    pages = {("s", f"{i:03d}"): [(f"/x/{i}_{k}.png", k) for k in range(n)]
             for i, n in enumerate([90, 80, 10, 10, 10, 5, 5, 1])}
    shards = runner._xeval_shards(pages, 3)
    assert len(shards) == 3
    seen = [(st, pg) for s in shards for st, pg, _ in s]
    assert sorted(seen) == sorted(pages), "кожна сторінка — рівно в одній частині"
    sizes = [sum(len(lines) for _, _, lines in s) for s in shards]
    assert max(sizes) - min(sizes) <= 90, sizes
    assert runner._xeval_shards(pages, 3) == shards, "розклад детермінований"
    assert len(runner._xeval_shards(dict(list(pages.items())[:2]), 8)) == 2, "порожніх частин немає"


def test_xeval_worker_is_a_guarded_script():
    """Воркер — окремий скрипт із запобіжником: kraken у `predict` заводить дочірні
    процеси, і під spawn без запобіжника кожен перезапускав би читання."""
    import ast

    from gpurunner._embedded import kraken_train_runner as runner
    tree = ast.parse(runner._XEVAL_WORKER)
    guards = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "_RUN"]
    assert '_RUN = __name__ == "__main__"' in runner._XEVAL_WORKER
    assert guards, "немає запобіжника від дочірніх процесів kraken"
    top_calls = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    assert not top_calls, "робота на верхньому рівні виконалась би в кожному дочірньому процесі"
    assert "num_line_workers=0" in runner._XEVAL_WORKER,         "пул kraken пересилає все полотно на кожен рядок і плутає порядок батча"


def test_xeval_runs_workers_as_subprocesses():
    import inspect

    from gpurunner._embedded import kraken_train_runner as runner
    src = inspect.getsource(runner._xeval_epochs)
    assert "subprocess.Popen" in src and "_xeval_shards(" in src and "_xeval_procs(" in src
    assert "multiprocessing" not in src


def test_eval_only_measures_the_base_too():
    """eval_only: ваги не `epoch_*` у вході моделей — база, її замір лягає в xeval_base.tsv."""
    import inspect

    from gpurunner._embedded import kraken_train_runner as runner
    src = inspect.getsource(runner.main)
    head = src[src.index('if params.get("eval_only")'):src.index("return", src.index('if params.get("eval_only")'))]
    assert 'startswith(("epoch_", "best_"))' in head and "base=bases[0]" in head


def test_xeval_workers_get_a_fixed_gpu_share():
    """Частка пам'яті на процес: повтор меншим батчем мусить залежати від сторінки, а не
    від того, що читають сусіди, інакше різні ваги читають ті самі сторінки по-різному."""
    import inspect

    from gpurunner._embedded import kraken_train_runner as runner
    assert "set_per_process_memory_fraction" in runner._XEVAL_WORKER
    assert "str(len(shards))" in inspect.getsource(runner._xeval_epochs)
