"""Калібровка моделі добору на ЗАВЕРШЕНИХ заходах, зшитих із матеріалом.

Джерело — не `boxes.jsonl` (там темп без рядків), а стан сесій наглядача
`state_dir()/*.json`: у кожній справі є `case`, `out_dir`, `pages_per_hour`, а в
`box` — карта, число карт, флот і проба заліза. Рядки на сторінку беруться з
`out_dir/_htr_meta.json` — того самого файла, що лежить на диску після забору.

Навіщо окремий модуль: до 06.09.2026 калібровку робили в чиємусь scratchpad, і
кожен наступний перерахунок починався з нуля. Тепер це одна функція
(`stitch_runs`) для тестів і одна команда (`gpurunner htr calibrate`) для
людини — і обидві читають ті самі рядки.

🔴 Зшивка дедуплікує за `(case, out_dir)`: `latest-*.json` дублює кожен захід,
і без цього 408 унікальних заходів виглядали б як 987.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurunner.htr.plan_build import lines_per_page_from_meta
from gpurunner.supervise.state import state_dir

#: З цієї дати заходи їдуть новим раннером (ліміт потоків, лок GPU, seg_resize)
#: — їхній темп на шард помітно вищий, і калібрувати їх разом зі старими
#: означало б усереднити дві різні машини.
NEW_RUNNER_SINCE = "2026-09-04"

#: Статуси, що означають «справу дорахували й темп заміряний».
_DONE = {"done", "incomplete"}

#: 🔴 Менші заходи міряють НЕ ТЕМП, а фіксовану ціну справи — і псують
#: калібровку вдвічі, бо ту ціну модель уже рахує окремо (`OVERHEAD_SEC_*`).
#: Замір 21.09.2026: 59 заходів із `pages_done` 0–19 (проби, догони, два записи
#: з нулем сторінок при додатному темпі) дають медіану прогноз/факт **4.29**
#: проти 1.05 на решті. «Непояснений» час — 19 с у справ до 30 сторінок і 0±25 с
#: у всіх більших бінів аж до 800+; для справи на 10 сторінок ці 19 с і є вся
#: справа. Поріг не підганяється: результат однаковий для 20, 50, 100, 200, 300.
#: ⚙ З 21.09.2026 `CaseState` зберігає `pages_this_run` і `wall_sec`, тож для
#: НОВИХ заходів поріг б'є саме по сторінках цього прогону — догін, що підняв
#: 380 сторінок і прочитав 20, більше не виглядає як захід на 400. Старі
#: заходи цих полів не мають і йдуть за `pages_done`, як і доти (`rated_pages`).
MIN_PAGES_FOR_RATE = 20
#: Те саме правило на рівні ШАРДА: кожен шард справи платить старт (процес,
#: моделі, розгін, ~15–20 с), а читає лише свою частку сторінок. Коли на шард
#: припадає менше п'яти сторінок, старт дорівнює читанню або більший за нього,
#: і темп справи міряє старт. Замір 23.09.2026: 61 такий захід (черги справ на
#: 20–60 сторінок на 8–9 шардах) — медіана прогноз/факт 1.49 проти 0.92 на решті.
MIN_PAGES_PER_SHARD_FOR_RATE = 5


@dataclass(frozen=True)
class Row:
    case: str
    out_dir: str
    gpu: str
    n_gpus: int
    cores: float
    vram: float
    shards: int
    lines: float
    pages: int
    actual: float
    updated: str
    #: Сторінки, прочитані САМЕ В ЦЬОМУ прогоні (21.09.2026). Нуль — захід
    #: старіший за це поле, і тоді обсяг доводиться брати з `pages`, у якому
    #: сидить і підняте з чекпоінтів.
    pages_this_run: int = 0
    #: Стінний час прогону, секунд; 0 — стан його не зберіг.
    wall_sec: float = 0.0

    @property
    def runner_new(self) -> bool:
        return self.updated[:10] >= NEW_RUNNER_SINCE

    @property
    def rated_pages(self) -> int:
        """Обсяг, за яким судимо, чи замір щось означає.

        🔴 Для свіжих заходів це сторінки ЦЬОГО прогону — саме вони стоять у
        чисельнику темпу. Для старих лишається `pages_done`: гірше, але це
        єдине, що про них записано.
        """
        return self.pages_this_run or self.pages

    @property
    def per_shard(self) -> float:
        return self.actual / self.shards if self.shards else 0.0


def _num(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def stitch_runs(states: Path | None = None, *, since: str = "") -> list[Row]:
    """Заходи з темпом, флотом і рядками на сторінку. Нема мети — рядки 0."""
    root = states or state_dir()
    if not root.is_dir():
        return []
    seen: dict[tuple[str, str], Row] = {}
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        box = data.get("box") or {}
        measured = box.get("measured") or {}
        sizing = box.get("sizing") or {}
        updated = str(data.get("updated") or "")
        if since and updated[:10] < since:
            continue
        for case in data.get("cases") or []:
            if not isinstance(case, dict) or case.get("status") not in _DONE:
                continue
            pph = _num(case.get("pages_per_hour"))
            out_dir = str(case.get("out_dir") or "")
            name = str(case.get("case") or "")
            if pph <= 0 or not name:
                continue
            key = (name, out_dir)
            prev = seen.get(key)
            if prev is not None and prev.updated >= updated:
                continue
            cores = _num(measured.get("cores_quota")) or _num(measured.get("cores"))
            row = Row(
                case=name, out_dir=out_dir,
                gpu=str(box.get("gpu") or ""),
                n_gpus=int(_num(box.get("num_gpus")) or 1),
                cores=cores,
                vram=_num(measured.get("vram_total_gb")),
                shards=int(_num(sizing.get("shards"))),
                lines=lines_per_page_from_meta(Path(out_dir)) if out_dir else 0.0,
                pages=int(_num(case.get("pages_done"))),
                actual=pph,
                updated=updated,
                pages_this_run=int(_num(case.get("pages_this_run"))),
                wall_sec=_num(case.get("wall_sec")),
            )
            seen[key] = row
    return sorted(seen.values(), key=lambda r: r.updated)


def rated(rows: list[Row]) -> list[Row]:
    """Заходи, чий темп узагалі щось міряє: є флот, є матеріал, є обсяг."""
    return [r for r in rows
            if r.shards and r.lines and r.rated_pages >= MIN_PAGES_FOR_RATE
            and r.rated_pages >= MIN_PAGES_PER_SHARD_FOR_RATE * r.shards]


def bins_table(rows: list[Row]) -> list[tuple[str, int, float, float, float]]:
    """(бін рядків, n, медіана на шард, p25, p75) — лише заходи з `rated`."""
    edges = [(0, 40), (40, 60), (60, 80), (80, 100), (100, 130), (130, 10_000)]
    rows = rated(rows)
    out = []
    for lo, hi in edges:
        vals = sorted(r.per_shard for r in rows if lo <= r.lines < hi)
        if not vals:
            continue
        q = statistics.quantiles(vals, n=4) if len(vals) >= 4 else [vals[0], vals[-1], vals[-1]]
        label = f"<{hi}" if lo == 0 else (f"{lo}+" if hi >= 10_000 else f"{lo}–{hi}")
        out.append((label, len(vals), statistics.median(vals), q[0], q[2]))
    return out


#: Стелі за ядрами в сітці: `None` — модель без неї (як до 21.09.2026).
_A_CORE_GRID = (None, 16_000.0, 18_000.0, 20_000.0)


def fit_table(
    rows: list[Row],
) -> list[tuple[float, float, float | None, float, float, float, float, float, float]]:
    """Сітка (A, L0, A_core, A_card) → медіана прогноз/факт, медіани окремо по
    СТАРОМУ й НОВОМУ раннеру, частка занижених >20% і завищених >25%.

    Формула та сама, що в `plan_sizing`, але флот береться ЗАМІРЯНИЙ: тут
    калібрується крива темпу, а не правила VRAM.

    🔴 Медіани раннерів — окремими колонками, і це не подробиця. Саме розрив
    між ними (0.96 проти 1.27) був ранньою ознакою того, що модель поїхала, а
    одне спільне число (1.06) його ховало: зміщення в різні боки взаємно
    гасились. Сходження двох колонок і є доказ, що член структурний, а не
    підігнаний.
    """
    grid: list[tuple[float, float, float | None, float, float, float, float, float, float]] = []
    usable = rated(rows)
    if not usable:
        return grid

    def median_of(rs: list[Row], a: float, l0: float, a_core: float | None,
                  a_card: float) -> float:
        out = []
        for r in rs:
            budget = min(a * r.shards, a_card * r.n_gpus)
            if a_core is not None and r.cores > 0:
                budget = min(budget, a_core * r.cores)
            out.append(budget / (l0 + r.lines) / r.actual)
        return statistics.median(out) if out else 0.0

    old = [r for r in usable if not r.runner_new]
    new = [r for r in usable if r.runner_new]
    for a in (30_000.0, 33_700.0, 36_000.0, 40_000.0, 50_000.0, 60_000.0):
        for l0 in (20.0, 29.0, 40.0):
            for a_core in _A_CORE_GRID:
                for a_card in (400_000.0, 495_000.0, 600_000.0):
                    ratios = []
                    for r in usable:
                        budget = min(a * r.shards, a_card * r.n_gpus)
                        if a_core is not None and r.cores > 0:
                            budget = min(budget, a_core * r.cores)
                        ratios.append(budget / (l0 + r.lines) / r.actual)
                    med = statistics.median(ratios)
                    under = sum(1 for x in ratios if x < 0.8) / len(ratios)
                    over = sum(1 for x in ratios if x > 1.25) / len(ratios)
                    grid.append((a, l0, a_core, a_card, med,
                                 median_of(old, a, l0, a_core, a_card),
                                 median_of(new, a, l0, a_core, a_card),
                                 under, over))
    return grid
