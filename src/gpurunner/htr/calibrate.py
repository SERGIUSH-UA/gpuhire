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

from gpurunner.config import registry_dir
from gpurunner.core.htr_sizing import (
    card_class,
    fixed_cost_for,
    gb_per_shard_for,
    plan_sizing,
    runner_of,
)
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
#: Від якого обсягу темп справи — сталий хід, а не розгін флоту: те саме число,
#: на якому міряно `GUARANTEE_FACTOR`.
MIN_PAGES_FOR_STEADY = 500


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


# ---- розріз за картою --------------------------------------------------------

#: Запис із меншою густиною — не матеріал, а збій заміру (30.09.2026: P100×2 з
#: `lines = 13` дав факт/прогноз 0.12).
MIN_LINES_FOR_CARD = 20.0
#: Вироки, чий темп міряв САМ бокс. На решті (`slow_for_data`, `below_target`,
#: `setup_failed`…) до 30.09.2026 лежав темп попередньої машини заходу.
_OWN_PACE = ("ok", "slow_run")
#: Нижче цієї частки некарткового бюджету (шарди, ядра) бокс уперся в карту.
CARD_BOUND_SHARE = 0.9


@dataclass(frozen=True)
class CardRun:
    """Один захід із власним заміром темпу й усім, від чого залежить прогноз."""

    source: str          # "boxes" | "state"
    case: str
    gpu: str             # клас карти (`card_class`)
    n_gpus: int
    cores: float
    vram: float          # вільна VRAM усіх карт, ГБ
    mpx: float
    lines: float
    pph: float
    outcome: str
    ts: str

    @property
    def card_lines(self) -> float:
        """Рядко-еквівалентів за годину на ОДНУ карту — те, що стеля обмежує."""
        return self.pph * (fixed_cost_for(self.mpx) + self.lines) / self.n_gpus

    def forecast(self, *, by_card: bool) -> float:
        sizing = plan_sizing(
            cores=self.cores, vram_gb=self.vram, num_gpus=self.n_gpus,
            gb_per_shard=gb_per_shard_for(self.mpx), lines_per_page=self.lines,
            frame_mpx=self.mpx, gpu_name=self.gpu if by_card else "",
            runner=runner_of(self.ts))
        return sizing.pages_per_hour

    @property
    def card_bound(self) -> bool:
        """Чи впирався бокс у карту, а не в ядра чи число шардів."""
        free = plan_sizing(
            cores=self.cores, vram_gb=self.vram, num_gpus=self.n_gpus,
            gb_per_shard=gb_per_shard_for(self.mpx), lines_per_page=self.lines,
            frame_mpx=self.mpx, card_lines_per_hour=float("inf"),
            runner=runner_of(self.ts))
        return self.pph < CARD_BOUND_SHARE * free.pages_per_hour


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return [r for r in rows if isinstance(r, dict)]


def card_runs(registry: Path | None = None, states: Path | None = None, *,
              since: str = "") -> list[CardRun]:
    """Заходи з власним заміром темпу: реєстр боксів і стан сесій наглядача.

    Реєстр несе умови заміру в самому записі. Стан сесії площі кадру не знає —
    вона береться за справою з реєстру й `shard_calibration.jsonl`; справа без
    площі в розріз не йде. Із сесій беруться лише ті, де машина була одна:
    `box` у стані — остання машина, і справи, прочитані до пересадки,
    приписались би їй.
    """
    root = registry or registry_dir()
    runs: list[CardRun] = []
    mpx_of: dict[str, float] = {}
    for row in _jsonl(root / "shard_calibration.jsonl"):
        if _num(row.get("frame_mpx")) > 0:
            mpx_of[str(row.get("case") or "")] = _num(row.get("frame_mpx"))
    for row in _jsonl(root / "boxes.jsonl"):
        m = row.get("measured") or {}
        case = str(row.get("case") or "")
        if _num(m.get("pages_per_hour_mpx")) > 0:
            mpx_of.setdefault(case, _num(m.get("pages_per_hour_mpx")))
        ts = str(row.get("ts") or "")
        if since and ts[:10] < since:
            continue
        if row.get("outcome") not in _OWN_PACE or m.get("pages_per_hour_case") != case:
            continue
        run = CardRun(
            source="boxes", case=case, gpu=card_class(str(row.get("gpu_name") or "")),
            n_gpus=int(_num(m.get("n_gpus")) or _num(row.get("num_gpus")) or 1),
            cores=_num(m.get("cores_quota")) or _num(m.get("cores")),
            vram=_num(m.get("vram_free_gb")) or _num(m.get("vram_total_gb")),
            mpx=_num(m.get("pages_per_hour_mpx")), lines=_num(m.get("pages_per_hour_lines")),
            pph=_num(m.get("pages_per_hour")), outcome=str(row.get("outcome")), ts=ts)
        if run.pph > 0 and run.mpx > 0 and run.cores > 0 and run.vram > 0 \
                and run.lines >= MIN_LINES_FOR_CARD and run.gpu:
            runs.append(run)

    sroot = states or state_dir()
    seen: dict[tuple[str, str], CardRun] = {}
    for path in sorted(sroot.glob("*.json")) if sroot.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or len(data.get("instances") or []) != 1:
            continue
        updated = str(data.get("updated") or "")
        if since and updated[:10] < since:
            continue
        box = data.get("box") or {}
        m = box.get("measured") or {}
        for case in data.get("cases") or []:
            if not isinstance(case, dict) or case.get("status") not in _DONE:
                continue
            name, out_dir = str(case.get("case") or ""), str(case.get("out_dir") or "")
            pages = int(_num(case.get("pages_this_run")) or _num(case.get("pages_done")))
            run = CardRun(
                source="state", case=name, gpu=card_class(str(box.get("gpu") or "")),
                n_gpus=int(_num(box.get("num_gpus")) or 1),
                cores=_num(m.get("cores_quota")) or _num(m.get("cores")),
                vram=_num(m.get("vram_free_gb")) or _num(m.get("vram_total_gb")),
                mpx=mpx_of.get(name, 0.0),
                lines=lines_per_page_from_meta(Path(out_dir)) if out_dir else 0.0,
                pph=_num(case.get("pages_per_hour")), outcome="state", ts=updated)
            if run.pph > 0 and run.mpx > 0 and run.cores > 0 and run.vram > 0 \
                    and run.lines >= MIN_LINES_FOR_CARD and run.gpu \
                    and pages >= MIN_PAGES_FOR_STEADY:
                prev = seen.get((name, out_dir))
                if prev is None or prev.ts < updated:
                    seen[(name, out_dir)] = run
    return runs + sorted(seen.values(), key=lambda r: r.ts)


def _quartiles(vals: list[float]) -> tuple[float, float, float]:
    vals = sorted(vals)
    if len(vals) >= 4:
        q = statistics.quantiles(vals, n=4)
        return q[0], statistics.median(vals), q[2]
    return vals[0], statistics.median(vals), vals[-1]


def card_table(runs: list[CardRun]) -> list[dict[str, Any]]:
    """На клас карти: скільки заходів, рядко-год на карту, факт/прогноз до й після."""
    by: dict[str, list[CardRun]] = {}
    for r in runs:
        by.setdefault(r.gpu, []).append(r)
    out = []
    for gpu, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        bound = [r.card_lines for r in rs if r.card_bound]
        p25, med, p75 = _quartiles([r.card_lines for r in rs])
        out.append({
            "gpu": gpu, "n": len(rs), "n_bound": len(bound),
            "vram_per_card": statistics.median(r.vram / r.n_gpus for r in rs),
            "card_lines": med, "card_lines_p25": p25, "card_lines_p75": p75,
            "card_lines_bound": statistics.median(bound) if bound else 0.0,
            "ratio_before": statistics.median(r.pph / r.forecast(by_card=False) for r in rs),
            "ratio_after": statistics.median(r.pph / r.forecast(by_card=True) for r in rs),
        })
    return out


def card_summary(runs: list[CardRun], *, by_card: bool) -> dict[str, float]:
    """Приймачі калібровки: медіана факт/прогноз, p20 і частка недобору."""
    ratios = sorted(r.pph / r.forecast(by_card=by_card) for r in runs)
    if not ratios:
        return {}
    return {
        "n": len(ratios), "median": statistics.median(ratios),
        "p20": ratios[int(0.2 * (len(ratios) - 1))],
        "under_08": sum(1 for x in ratios if x < 0.8) / len(ratios),
    }


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
                # Флот, який регулятор справді тримав; старі стани його не
                # мають — тоді флот воріт, як і доти.
                shards=int(_num((box.get("fleet") or {}).get("n_stable"))
                           or _num(sizing.get("shards"))),
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
