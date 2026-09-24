"""Переведення різних карт в одну одиницю — **T4-години**.

Баланси семи бекендів номіновані в чотирьох різних одиницях: години (Kaggle),
кредити (Lightning), compute units (Colab), долари (Modal/Vast/Beam). Питання
«де в мене більше обчислень» без спільної шкали відповіді не має, а всі бекенди
вміють T4 (або щось, що з T4 порівнюється), тож T4-година — природний знаменник.

Коефіцієнти двох сортів, і різниця між ними НЕ косметична:

**Вимір** — з наших власних бенчів (``jobs/yolo_spotter.py::MEASURED_IT_S``,
train imgsz1280, 2026-06-30): T4 2.2 it/s, L4 3.1, A100 8.1. Це те, що реально
показали ці карти на нашому навантаженні.

**Паспорт** — відношення FP16 tensor (dense, FP32-accumulate) до T4, помножене на
калібрувальний коефіцієнт. Коефіцієнт не вигаданий, а **виведений із тих самих
двох вимірів**:

    L4:   спец 121/65 = 1.862,  вимір 1.409  →  0.757
    A100: спец 312/65 = 4.800,  вимір 3.682  →  0.767

Дві незалежні точки сходяться в межах 1.3%: паспорт стабільно оптимістичніший за
реальність приблизно на чверть (пам'ять, dataloader, дрібні ядра — все, чого в
TFLOPS немає). Звідси ``_SPEC_CALIBRATION``.

Застереження, яке мусить їхати разом із числом: калібрування зняте з **однієї**
задачі — YOLO-трену з великим imgsz. На пам'яте-зв'язаному або дрібно-батчевому
навантаженні воно поїде, причому в бік ще більшого відриву паспорта від реальності.
Тому все, що не «вимір», позначене ``basis="паспорт"`` і в UI має свій маркер.

Карта, якої тут немає, **не рахується як нуль** — вона повертає ``None`` і
потрапляє в список «не переведено». Мовчазний нуль у підсумку означав би
«обчислень немає», що прямо протилежне до «невідомо, скільки їх».
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: it/s на batch=8 з ``yolo_spotter.MEASURED_IT_S`` — єдині карти, які ми міряли самі.
_MEASURED_IT_S: dict[str, float] = {"T4": 2.2, "L4": 3.1, "A100": 8.1}

#: FP16 tensor-core, **dense** (без structured sparsity), TFLOPS. Для споживчих
#: GeForce взято шлях із FP32-акумуляцією — тренування йде саме ним, а на GeForce
#: він удвічі повільніший за FP16-акумуляцію (архітектурне обмеження, не драйверне).
#: Числа звірені 2026-08-02; у дужках — що каже маркетинг «зі sparsity», бо саме
#: його друкують на слайдах і саме через нього тут легко помилитись удвічі.
_FP16_DENSE_TFLOPS: dict[str, float] = {
    "T4": 65.0,      # Turing, sparsity не підтримує
    "V100": 125.0,   # Volta, tensor 1-го покоління
    "A10G": 125.0,   # (250 зі sparsity)
    "L4": 121.0,     # (242 зі sparsity)
    "A100": 312.0,   # (624 зі sparsity); 40 і 80 GB мають однаковий обчислювач
    "A6000": 155.0,  # Ampere professional — FP32-акумуляція тут НЕ ділиться навпіл
    "L40S": 181.0,   # (362 зі sparsity — це число і стоїть у даташиті)
    "H100": 989.0,   # SXM (1979 зі sparsity); PCIe-версія помітно нижча — 756
    "H200": 989.0,   # той самий обчислювач, що H100 SXM; відрізняється пам'яттю
}
# СВІДОМО ВІДСУТНІ — і це не недогляд:
#
# GeForce (RTX 3090/4090/5090). Тут паспортний проксі ДОВЕДЕНО не працює. NVIDIA
# ріже на GeForce шлях «FP16 з FP32-акумуляцією» вдвічі (саме ним іде mixed-precision
# трен), тож формула дає RTX3090 = 0.42…0.83 від T4 — тобто «повільніший за T4»,
# що суперечить будь-якому польовому виміру. Причина розходження в тому, чого в
# TFLOPS немає: у 3090 936 GB/s пам'яті проти 320 у T4 і 350 Вт проти 70. Двох
# наших калібрувальних точок (обидві — датацентрові карти) не вистачає, щоб
# підігнати двофакторну модель, а підганяти дві точки двома параметрами — це не
# модель, а переписування вимірів. Тому для GeForce коефіцієнта немає; хто має
# власний вимір, вписує його руками (див. ``overrides``).
#
# P100 — той самий клас помилки, з іншого боку: у Pascal tensor-ядер немає
# взагалі, тож її 19 TFLOPS — це звичайний векторний FP16, який ні з чиїмись
# tensor-TFLOPS не порівнюється. Формула дає 0.22× T4, тобто «вчетверо повільніша»,
# чого польові дані не підтверджують. Для квоти Kaggle це й не потрібно: вона
# тарифікує години сесії, а не обчислення (див. ``CHARGE_SESSION`` у quota.py).
#
# RTXPro6000, B200, RTX5090: джерела на їхню dense FP16 з FP32-акумуляцією
# розходяться між собою вдвічі. Додавати — лише звіривши по даташиту NVIDIA.

#: Наскільки паспорт бреше. Виведений з двох вимірів (див. докстринг модуля).
_SPEC_CALIBRATION: float = 0.76


@dataclass(frozen=True)
class Equivalence:
    """У скільки разів карта швидша за T4 і звідки це відомо."""

    factor: float
    basis: str          # "вимір" | "паспорт"
    note: str = ""

    @property
    def confident(self) -> bool:
        """Чи спирається множник на вимір (наш або користувача), а не на паспорт."""
        return self.basis in ("вимір", "вручну")


def _measured_factor(gpu: str) -> float | None:
    base = _MEASURED_IT_S.get(gpu)
    if base is None:
        return None
    return base / _MEASURED_IT_S["T4"]


#: Аліаси до канонічних імен. Бекенди називають ту саму карту по-різному:
#: Lightning пише ``T4_X_2``, Beam — ``A100-40``, Saturn віддає назви інстансів.
_ALIASES: dict[str, str] = {
    "A100-40": "A100",
    "A100-40GB": "A100",
    "A100-80": "A100",
    "A100-80GB": "A100",
    "A10": "A10G",
    "RTX A6000": "A6000",
    "RTX 3090": "RTX3090",
    "RTX 4090": "RTX4090",
}


#: Скільки карт максимум може бути в одному прогоні. Верхня межа тут не про
#: залізо, а про розбір рядка: без неї ``RTX3090`` читається як «RT × 3090»,
#: бо ``X3090`` виглядає точно як множник.
_MAX_GPU_COUNT = 16


def normalize(gpu: str) -> tuple[str, int]:
    """``"T4x2"`` → ``("T4", 2)``. Повертає канонічне ім'я карти і їх кількість.

    Множники пишуть трьома способами (``T4x2`` у нас, ``T4_X_2`` у Lightning,
    ``2xT4`` у прайсах), і всі три зустрічаються в параметрах прогонів.
    """
    raw = (gpu or "").strip()
    if not raw:
        return "", 0
    count = 1
    m = re.search(r"(?:_X_|[xX*])\s*(\d+)$", raw)
    if m and 1 <= int(m.group(1)) <= _MAX_GPU_COUNT:
        count = int(m.group(1))
        raw = raw[: m.start()].rstrip("_xX* ")
    else:
        m = re.match(r"^(\d+)\s*[xX*]\s*(.+)$", raw)
        if m and 1 <= int(m.group(1)) <= _MAX_GPU_COUNT:
            count = int(m.group(1))
            raw = m.group(2)
    name = raw.strip()
    canonical = _ALIASES.get(name, _ALIASES.get(name.upper(), name))
    return canonical, count


def equivalence(gpu: str, overrides: dict[str, float] | None = None) -> Equivalence | None:
    """Скільки T4 «варта» одна така карта. ``None`` — коефіцієнта немає.

    Кількість карт тут НЕ враховується: множник — це властивість моделі. Скільки
    їх у прогоні, вирішує ``t4_hours``, бо для квоти Kaggle (яка рахує години
    сесії, а не карт) це різні питання.

    ``overrides`` — множники, вписані користувачем (зберігає ``core/quota.py``).
    Вони мають найвищий пріоритет: власний вимір завжди кращий за наш паспортний
    проксі, а для GeForce це взагалі єдиний спосіб отримати число.
    """
    name, _ = normalize(gpu)
    if not name:
        return None
    if overrides and (manual := overrides.get(name)) is not None:
        return Equivalence(factor=float(manual), basis="вручну", note="множник вписано в дашборді")
    if (measured := _measured_factor(name)) is not None:
        return Equivalence(factor=measured, basis="вимір", note="бенч yolo_spotter, 2026-06-30")
    tflops = _FP16_DENSE_TFLOPS.get(name)
    if tflops is None:
        return None
    ratio = tflops / _FP16_DENSE_TFLOPS["T4"]
    return Equivalence(
        factor=ratio * _SPEC_CALIBRATION,
        basis="паспорт",
        note=f"FP16 dense {tflops:g} TFLOPS ÷ T4, калібр. ×{_SPEC_CALIBRATION}",
    )


def t4_hours(
    gpu: str,
    hours: float,
    *,
    scale_by_count: bool = True,
    overrides: dict[str, float] | None = None,
) -> tuple[float, str] | None:
    """Години на ``gpu`` → T4-години. ``None``, якщо карта не переводиться.

    ``scale_by_count=False`` для квот, які тарифікують **сесію**, а не карти:
    година на ``T4x2`` списує з тижневих 30 годин Kaggle рівно одну годину,
    хоч обчислень дає дві.
    """
    eq = equivalence(gpu, overrides)
    if eq is None:
        return None
    _, count = normalize(gpu)
    total = hours * eq.factor * (count if scale_by_count else 1)
    return total, eq.basis


def known_gpus(overrides: dict[str, float] | None = None) -> list[str]:
    """Усі карти з коефіцієнтом — виміряні спершу."""
    names = set(_FP16_DENSE_TFLOPS) | set(overrides or {})
    return sorted(_MEASURED_IT_S) + sorted(names - set(_MEASURED_IT_S))


def needs_override() -> list[str]:
    """Карти, які бекенди вміють замовити, але коефіцієнта для них немає.

    Це не «список невідомих карт узагалі», а рівно ті, через які підсумок у
    T4-годинах виходить неповним — і які варто запропонувати користувачу
    заповнити руками.
    """
    from gpurunner.backends import BACKEND_NAMES, get_backend

    # «any» (Vast — будь-яка карта), «none»/«cpu» — не моделі GPU, а режими.
    not_a_card = {"none", "cpu", "any", ""}
    missing: set[str] = set()
    for name in BACKEND_NAMES:
        try:
            choices = get_backend(name).gpu_choices
        except KeyError:
            continue
        for gpu in choices:
            base, _ = normalize(gpu)
            if base.lower() not in not_a_card and equivalence(gpu) is None:
                missing.add(base)
    return sorted(missing)
