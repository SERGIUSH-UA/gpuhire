"""Ворота повноти на боці наглядача: чи забране справді ціле.

Раннер уже перевіряє себе на боксі, але його вирок — самозвіт. Тут той самий
факт звіряється ще раз і **на локальному диску**, бо між боксом і диском є
обірваний SFTP, недокачаний архів і рука людини.

🔴 Звіряються ТРИ незалежні числа, і розбіжність будь-якої пари — це «неповно»:

1. `n_pages_expected` із підсумку — скільки кадрів було на вході;
2. фактичні `*.txt` у забраній теці — скільки їх лежить тут і зараз;
3. пара `n_pages_input` / `n_pages_total` — стара ознака CUDA OOM, чинна для
   результатів попередніх версій раннера, де полів повноти ще не було.

Саме на браку такої звірки конвеєр забрав 203 сторінки з 323 і 157 з 209.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Completeness:
    """Вирок про забраний результат."""

    complete: bool
    expected: int
    got: int
    missing: list[str] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    #: Карантиновані сторінки, у яких НЕМАЄ тексту на диску. Саме вони — дірки.
    #: Решта карантину нешкідлива: сторінка потрапила в карантин, але пізніший
    #: прохід (чи чекпоінт) її все-таки прочитав.
    quarantined_empty: list[str] = field(default_factory=list)
    source: str = "summary"       # звідки взявся знаменник
    detail: str = ""

    @property
    def missing_count(self) -> int:
        """🔴🔴 КАРАНТИН НЕ ЗМЕНШУЄ ЗНАМЕННИКА.

        Доти карантиновані віднімались від очікуваних, тобто справа з дірою
        віддавала `complete: true`. Замір 18.08.2026 (ДАВіО): карантин з'їв
        п'ять сторінок, вердикт прийшов `ok`, а локально всі п'ять узялись
        З ПЕРШОГО РАЗУ. Тобто карантин не доводить, що сторінка нечитабельна;
        він доводить лише, що ЦЕЙ бокс не впорався з нею під цим навантаженням.

        Доказ читабельності один — текст на диску. Тому дірка = карантинована
        сторінка без тексту.
        """
        return max(len(self.missing), len(self.quarantined_empty),
                   max(0, self.expected - self.got))


def verify_case(out_dir: Path, *, expected_hint: int | None = None) -> Completeness:
    """Звірити забрану теку. `expected_hint` — знаменник із плану, якщо він є."""
    out_dir = Path(out_dir)
    texts = _texts(out_dir)
    got = len(texts)
    summary = _summary(out_dir)

    if summary is None:
        expected = int(expected_hint or 0)
        if not expected:
            # 🔴 Нуль знаменника — це не «повно», це «нема з чим звіряти».
            return Completeness(
                False, 0, got, source="none",
                detail="немає htr_case_summary.json і немає знаменника з плану — "
                       "повноту підтвердити НЕМА ЧИМ",
            )
        missing = max(0, expected - got)
        return Completeness(
            missing == 0, expected, got, source="hint",
            detail=f"підсумку немає; за планом мало бути {expected}, на диску {got}",
        )

    quarantined = [str(x) for x in (summary.get("quarantined_pages") or [])]
    # 🔴 Карантинована сторінка — не «списана», а ПІДОЗРІЛА. Питаємо диск: якщо
    # текст є (його міг дати догінний прохід чи чекпоінт), діри немає.
    have_text = {t.stem for t in texts}
    quarantined_empty = [q for q in quarantined if _stem(q) not in have_text]
    from_box = int(summary.get("n_pages_expected") or summary.get("n_pages_input") or 0)
    expected = from_box or int(expected_hint or 0)
    missing_named = [str(x) for x in (summary.get("missing_pages") or [])]

    problems: list[str] = []
    # 🔴 Знаменник із плану мовчки поступався знаменнику з боксу. Замовник рахує
    # кадри з реальних файлів справи саме для того, щоб їх не вигадувати, — а
    # використовувався він ЛИШЕ коли підсумку нема взагалі. Сценарій, у якому
    # це коштує справи: у R2 лежить тарбол минулого заходу на 3000 кадрів, на
    # диску вже 3390. Бокс чесно робить 3000 і каже `complete: true`; наглядач
    # пише поруч свої 3390 — і жоден рядок стану не називає розходження.
    # Тепер розходження саме по собі є проблемою: мовчки перемагати не має
    # право жодне з двох чисел.
    if expected_hint and from_box and int(expected_hint) != from_box:
        problems.append(
            f"знаменники розійшлись: у плані {int(expected_hint)} кадрів, "
            f"бокс рахував {from_box} — на боксі був НЕ ТОЙ набір кадрів"
        )
        expected = max(from_box, int(expected_hint))
    if summary.get("complete") is False:
        problems.append("раннер сам позначив прогін неповним")
    if missing_named:
        problems.append(f"раннер назвав {len(missing_named)} кадрів без тексту")
    if expected and got < expected:
        problems.append(f"на диску {got} текстів із очікуваних {expected}")
    if quarantined_empty:
        # 🔴 Називаємо ПОІМЕННО: ці сторінки майже завжди читаються вдома з
        # першого разу, тобто дірка закривається безкоштовно — але лише якщо
        # агент про неї дізнався.
        shown = ", ".join(quarantined_empty[:8])
        problems.append(
            f"{len(quarantined_empty)} карантинованих сторінок БЕЗ тексту "
            f"({shown}{'…' if len(quarantined_empty) > 8 else ''}) — "
            f"це дірки, і локально вони зазвичай беруться з першого разу")

    total = summary.get("n_pages_total")
    if expected and total is not None and int(total) < expected - len(quarantined_empty):
        # Та сама ознака, що ловила OOM до появи полів повноти.
        problems.append(f"мета знає {total} сторінок із {expected}")

    if failed := (summary.get("failed_shards") or []):
        problems.append(f"шарди впали: {failed}")

    detail = "; ".join(problems) if problems else (
        f"{got} текстів із {expected}"
        + (f", {len(quarantined)} у карантині (всі з текстом)" if quarantined else "")
    )
    if oom := int(summary.get("oom_events") or 0):
        detail += f"; OOM-подій {oom}"

    return Completeness(
        complete=not problems,
        expected=expected,
        got=got,
        missing=missing_named,
        quarantined=quarantined,
        quarantined_empty=quarantined_empty,
        source="summary",
        detail=detail,
    )


def _stem(page: str) -> str:
    """Ім'я сторінки без розширення. Карантин пише то `0042`, то `0042.jpg`."""
    return pathlib.PurePath(page).stem


def _summary(out_dir: Path) -> dict | None:
    for candidate in (out_dir / "htr_case_summary.json", *out_dir.rglob("htr_case_summary.json")):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            return data if isinstance(data, dict) else None
    return None


def _texts(out_dir: Path) -> list[Path]:
    """Тексти сторінок. `out/` — штатне місце, але забір міг розкласти інакше.

    Службові файли конвеєра (`_htr_*.json`) сюди не потрапляють за
    розширенням, а от `.txt` із логів — потрапили б, тому беремо лише те, що
    лежить у теках виводу.
    """
    direct = sorted((out_dir / "out").glob("*.txt"))
    if direct:
        return direct
    return [p for p in sorted(out_dir.rglob("*.txt")) if p.parent.name != "logs"]
