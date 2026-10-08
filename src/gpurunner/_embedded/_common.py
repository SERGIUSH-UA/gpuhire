"""Спільний пролог для embedded-ранерів; вклеюється рендером ПЕРЕД раннером.

Це не імпортований модуль (як і самі ранери) — ``Job.render_kaggle_code``
конкатенує цей файл із раннером в один текст, який їде на кернел. Звідси
обмеження ті самі: лише stdlib на верхньому рівні, жодних імпортів gpurunner,
без ``from __future__ import annotations``.

🔴 Порядок склеювання (спершу _common, потім раннер) навмисний: якщо раннер
визначає функцію з тим самим іменем, перемагає ЙОГО версія. Тому додавання
цього файлу нічого не ламає само по собі, а прибирати локальні копії можна
поступово, по одному раннеру.

Що сюди варто виносити: дрібні чисті функції, які вже скопійовані дослівно в
кілька ранерів і де розходження копій — це баг. Що НЕ варто: усе, що ранери
свідомо роблять по-різному (parseq розпаковує tgz системним ``tar``, решта —
``tarfile``), і все, що потрібне раннерам, які їздять ще й на Modal
(``paddleocr``, ``vit_classifier``, ``yolo_spotter``): Modal бере лише сам
модуль раннера, без цього прологу, тож їхні копії мусять лишатись при них.
"""
from datetime import datetime, timezone
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def _utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _dataset_root(slug: str, gt_file: str) -> Path:
    """Тека змонтованого датасету, знайдена за характерним файлом.

    🔴 Точка монтування Kaggle не одна: буває ``/kaggle/input/<slug>``, буває
    ``/kaggle/input/<owner>-<slug>``, а з якогось моменту — ще й
    ``/kaggle/input/datasets/<owner>/<slug>/``. Через це job падав із «не
    знайдено», хоча дані лежали поруч. Тому: спершу очевидні кандидати, далі
    рекурсивний пошук, і аж тоді помилка — але з деревом того, що реально
    змонтовано (найчастіша справжня причина — датасет ще не розпакувався).
    """
    slug_short = slug.split("/")[-1]
    candidates = [KAGGLE_INPUT / slug_short, KAGGLE_INPUT / slug.replace("/", "-")]
    candidates += [d for d in KAGGLE_INPUT.iterdir() if d.is_dir()] if KAGGLE_INPUT.exists() else []
    for c in candidates:
        if (c / gt_file).exists():
            return c
    hits = sorted(KAGGLE_INPUT.glob(f"**/{gt_file}")) if KAGGLE_INPUT.exists() else []
    if hits:
        print(f"[diag] dataset root via rglob: {hits[0].parent}", flush=True)
        return hits[0].parent
    tree = []
    if KAGGLE_INPUT.exists():
        for d in sorted(KAGGLE_INPUT.iterdir()):
            tree.append(str(d))
            if d.is_dir():
                tree += [f"  {x.name}" for x in sorted(d.iterdir())[:8]]
    raise RuntimeError(
        f"dataset root with {gt_file} not found under {KAGGLE_INPUT}; mounted:\n"
        + ("\n".join(tree) or "<порожньо>"))


def _norm(s: str) -> str:
    """Нормалізація для нечіткого порівняння дореформеної кирилиці.

    ѣ→е, і→и, ъ викидається — інакше та сама лексема в тексті XIX ст. і в
    сучасному GT не збігається. Правило мусить бути одне на всі бенчмарки:
    коли копій три, вони розходяться, і числа з різних job-ів перестають бути
    порівнюваними — а вони існують саме заради порівняння.
    """
    s = s.lower().replace("ё", "е").replace("ъ", "").replace("ѣ", "е").replace("і", "и")
    return "".join(ch for ch in s if ch.isalpha() or ch == " ").strip()
