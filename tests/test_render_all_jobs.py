"""Кожен job мусить рендерити синтаксично валідний, повний код.

Помилка рендеру не має де виявитись, окрім віддаленого кернела: `submit` шле
рядок, Kaggle запускає його через papermill і повертає PapermillExecutionError
через 3 хвилини після старту GPU-сесії. Тому рендер кожного job перевіряється
локально й на кожному job — включно з тими, яких давно не запускали.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from gpurunner.jobs import get_job, list_jobs

# Мінімум, щоб validate_params пропустив. Job, який не задовольняється
# 'dataset', мусить бути тут явно — інакше тест впаде і змусить додати запис,
# замість того щоб мовчки не перевіряти новий job.
MINIMAL_PARAMS: dict[str, dict] = {
    "paddleocr": {"urls": ["https://example.org/a.pdf"]},
    # шукане слово — дані користувача, дефолту для нього немає
    "htr_eval": {"dataset": "owner/slug", "target": "слово"},
}
_DEFAULT_PARAMS = {"dataset": "owner/slug"}

# dino ЄДИНИЙ інжектить sys.argv і навмисно лишає __main__ як точку входу:
# його раннер — argparse-CLI, а не main(params). Решта мусить зрізати блок.
_ARGV_BRIDGE_JOBS = {"dino_surname_verifier"}

# Раннери, що навмисно перекривають функцію зі спільного прологу. Порядок
# склеювання (_common → раннер) робить перекриття передбачуваним, але кожен
# випадок мусить мати причину — інакше це просто забута копія.
_ALLOWED_OVERRIDES = {
    # один раннер обслуговує і Kaggle, і Modal; на Modal спільного прологу
    # немає взагалі, тож власний _utc_iso мусить лишатись при ньому
    "paddleocr": {"_utc_iso"},
    # у ViT інша сигнатура: _dataset_root(slug) шукає train_labels.csv,
    # спільна — _dataset_root(slug, gt_file)
    "vit_classifier": {"_dataset_root"},
    "crop_verifier": {"_dataset_root"},
}

JOB_NAMES = sorted(j.name for j in list_jobs())


def _params(name: str) -> dict:
    return MINIMAL_PARAMS.get(name, _DEFAULT_PARAMS)


def _render(name: str) -> str:
    return get_job(name)().render_remote_code(_params(name))


def test_every_job_is_covered():
    assert JOB_NAMES, "реєстр job-ів порожній"


@pytest.mark.parametrize("name", JOB_NAMES)
def test_rendered_code_parses(name):
    ast.parse(_render(name))


@pytest.mark.parametrize("name", JOB_NAMES)
def test_rendered_code_defines_params_and_entry(name):
    src = _render(name)
    tree = ast.parse(src)
    assigned = {
        t.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    # job без жодного параметра (net-probe — інлайн-діагностика) не має чого
    # інжектити; для решти відсутній PARAMS означав би NameError на кернелі
    if get_job(name)().validate_params(_params(name)):
        assert "PARAMS" in assigned, "PARAMS не інжектнуто"
    # рівно одне визначення main на верхньому рівні — дубль означав би, що
    # зріз __main__ не спрацював і хвіст раннера приклеївся вдруге
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert len(mains) <= 1, f"{len(mains)} визначень main"


@pytest.mark.parametrize("name", JOB_NAMES)
def test_no_duplicate_top_level_definitions(name):
    """Жодне ім'я не має визначатись двічі на верхньому рівні.

    Спільний пролог (_common.py) вклеюється перед раннером, тож дубль означав
    би, що раннер досі несе власну копію винесеної функції — вона перекриє
    спільну і правка спільної до нього не доїде. Саме так розійшлись були
    чотири копії _dataset_root, з яких лише одна навчилась нової схеми
    монтування Kaggle.
    """
    tree = ast.parse(_render(name))
    seen: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seen[node.name] = seen.get(node.name, 0) + 1
    allowed = _ALLOWED_OVERRIDES.get(name, ())
    dupes = {n: c for n, c in seen.items() if c > 1 and n not in allowed}
    assert not dupes, f"визначені двічі: {dupes}"


@pytest.mark.parametrize("name", JOB_NAMES)
def test_rendered_code_has_no_undefined_names(name):
    """У згенерованому модулі не має бути невизначених імен.

    ``ast.parse`` цього не ловить, а саме тут з'явився новий спосіб помилитись:
    відколи спільні хелпери живуть у ``_common.py``, окремий раннер більше не
    самодостатній — забути вклеїти пролог або прибрати з раннера потрібну йому
    функцію тепер можна беззвучно, і NameError вилізе аж на кернелі.
    """
    ruff = Path(sys.executable).with_name("ruff.exe")
    if not ruff.exists():
        ruff = Path(sys.executable).with_name("ruff")
    if not ruff.exists():
        pytest.skip("ruff недоступний")
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "rendered.py"
        target.write_text(_render(name), encoding="utf-8")
        proc = subprocess.run(
            [str(ruff), "check", "--isolated", "--select", "F821",
             "--output-format", "concise", str(target)],
            capture_output=True, text=True,
        )
    assert proc.returncode == 0, f"невизначені імена:\n{proc.stdout}"


@pytest.mark.parametrize("name", sorted(set(JOB_NAMES) - _ARGV_BRIDGE_JOBS))
def test_main_guard_is_stripped(name):
    """__main__-блок раннера мусить бути зрізаний.

    У Kaggle-клітинці `__name__` дорівнює "__main__", тож незрізаний блок
    виконався б ДО інжектнутого `main(PARAMS)` — зі своїми жорстко зашитими
    параметрами.
    """
    assert "if __name__" not in _render(name)
