"""T4-еквіваленти: що переводиться, що ні, і чому калібрування саме таке."""

from __future__ import annotations

import pytest

from gpurunner.core import gpu_equiv

# ---- розбір назви ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("T4", ("T4", 1)),
        ("T4x2", ("T4", 2)),
        ("T4_X_2", ("T4", 2)),      # так пише Lightning
        ("2xT4", ("T4", 2)),
        ("A100-40", ("A100", 1)),   # так пише Beam
        ("A100-80GB", ("A100", 1)),
        ("", ("", 0)),
    ],
)
def test_normalize(raw: str, expected: tuple[str, int]) -> None:
    assert gpu_equiv.normalize(raw) == expected


@pytest.mark.parametrize("raw", ["RTX3090", "RTX4090", "RTX5090"])
def test_model_number_is_not_a_multiplier(raw: str) -> None:
    """``RTX3090`` не має читатись як «RT × 3090».

    ``X3090`` виглядає точно як суфікс-множник, і без верхньої межі на кількість
    карт розбір ламався мовчки: карта ставала невідомою ``RT``, коефіцієнта не
    знаходилось, і прогін тихо випадав із підрахунку.
    """
    assert gpu_equiv.normalize(raw) == (raw, 1)


# ---- коефіцієнти -----------------------------------------------------------


def test_measured_cards_come_from_our_own_benchmarks() -> None:
    for gpu in ("T4", "L4", "A100"):
        eq = gpu_equiv.equivalence(gpu)
        assert eq is not None and eq.basis == "вимір" and eq.confident

    assert gpu_equiv.equivalence("T4").factor == pytest.approx(1.0)      # type: ignore[union-attr]
    assert gpu_equiv.equivalence("L4").factor == pytest.approx(1.41, abs=0.01)   # type: ignore[union-attr]
    assert gpu_equiv.equivalence("A100").factor == pytest.approx(3.68, abs=0.01)  # type: ignore[union-attr]


def test_spec_calibration_reproduces_the_measurements() -> None:
    """Калібрувальний коефіцієнт виведений із вимірів — перевіряємо, що сходиться.

    Якби ``_SPEC_CALIBRATION`` хтось «підкрутив», паспортна формула перестала б
    відтворювати ті самі L4 і A100, з яких її й отримали, і всі паспортні
    коефіцієнти поїхали б разом із нею.
    """
    t4 = gpu_equiv._FP16_DENSE_TFLOPS["T4"]
    for gpu, measured in (("L4", 1.409), ("A100", 3.682)):
        spec = gpu_equiv._FP16_DENSE_TFLOPS.get(gpu)
        if spec is None:      # L4 і A100 є у вимірах, спец-числа беремо для звірки
            spec = {"L4": 121.0, "A100": 312.0}[gpu]
        predicted = spec / t4 * gpu_equiv._SPEC_CALIBRATION
        assert predicted == pytest.approx(measured, rel=0.02)


def test_datacentre_cards_get_a_spec_factor() -> None:
    eq = gpu_equiv.equivalence("H100")
    assert eq is not None and eq.basis == "паспорт" and not eq.confident
    assert eq.factor > gpu_equiv.equivalence("A100").factor  # type: ignore[union-attr]


@pytest.mark.parametrize("gpu", ["RTX3090", "RTX4090", "P100", "B200"])
def test_cards_where_the_proxy_fails_have_no_factor(gpu: str) -> None:
    """Мовчазний нуль тут був би гіршим за відсутність числа.

    Для GeForce паспортна формула дає «повільніше за T4», для P100 — «вчетверо
    повільніше», і жодне з цих тверджень не витримує польової перевірки. Такі
    карти мусять чесно не переводитись, а не потрапляти в підсумок заниженими.
    """
    assert gpu_equiv.equivalence(gpu) is None


def test_missing_cards_are_reported_not_swallowed() -> None:
    missing = gpu_equiv.needs_override()
    assert "RTX4090" in missing
    # «any» (Vast — будь-яка карта) і «none» — режими, а не моделі
    assert not {"any", "none", ""} & set(missing)


# ---- ручні множники --------------------------------------------------------


def test_manual_override_wins_over_everything() -> None:
    eq = gpu_equiv.equivalence("RTX4090", {"RTX4090": 3.2})
    assert eq is not None and eq.factor == 3.2 and eq.basis == "вручну" and eq.confident
    # навіть там, де в нас є власний вимір
    assert gpu_equiv.equivalence("T4", {"T4": 2.0}).factor == 2.0  # type: ignore[union-attr]


# ---- години ----------------------------------------------------------------


def test_multiple_cards_multiply_compute_but_not_session_time() -> None:
    """Дві T4 дають удвічі більше обчислень — але годину сесії Kaggle однаково.

    Саме тому ``scale_by_count`` існує окремим прапорцем: квота Kaggle тарифікує
    сесію, і зарахувати ``T4x2`` як дві години було б помилкою вдвічі.
    """
    assert gpu_equiv.t4_hours("T4x2", 1.0) == (2.0, "вимір")
    assert gpu_equiv.t4_hours("T4x2", 1.0, scale_by_count=False) == (1.0, "вимір")


def test_unknown_card_returns_none_not_zero() -> None:
    assert gpu_equiv.t4_hours("RTX4090", 5.0) is None
