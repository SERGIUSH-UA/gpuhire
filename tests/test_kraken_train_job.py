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
