"""Передпольотна перевірка плану — ПЕРЕД орендою.

Дві речі, які інакше з'ясовуються вже на оплачуваній машині:

1. **Чи живі посилання й яка з них швидкість.** Presigned URL має строк
   придатності; протермінований віддає 403 за мілісекунди, і на боксі це
   виглядає як «мертвий канал хоста» — тобто здорові машини їдуть у чорний
   список одна за одною, а захід закінчується діагнозом «ринок порожній».
2. **Чи є в R2 чекпоінти цієї справи.** Це відповідь на питання «якщо бокс
   помре, ми почнемо з нуля чи продовжимо».

🔴 Перевіряти треба саме GET із `Range`, і ніколи `HEAD`: підпис виданий під
GET, тож HEAD на справному посиланні віддає 403 — і бездоганний план виглядав
би протухлим.

🔴 Ціна перевірки не мусить рости з числом справ швидше, ніж сама робота.
Доти кожна справа тягнула 8 МБ на замір швидкості й до восьми запитів по
чекпоінти — послідовно. Пачка з 1 789 газетних номерів по 4 сторінки
(12.10.2026) за кілька годин дійшла до 137-ї справи і так і не орендувала
машину. Тепер швидкість міряється один раз на план, чекпоінти питаються до
першого відсутнього слота, а справи перевіряються паралельно.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

#: Скільки байтів тягнемо для заміру швидкості. 8 МБ вистачає, щоб число було
#: осмисленим, і мало, щоб перевірка лишалась дешевою.
PROBE_BYTES = 8 * 1024 * 1024
PROBE_TIMEOUT = 20

#: Скільки справ перевіряється одночасно. Кожна перевірка — це `curl` у
#: власному процесі, що чекає на мережу, тож потоків може бути більше за ядра.
PARALLEL = 16

#: Від якого розміру плану друкувати хід перевірки (у stderr: stdout у `--json`
#: читає програма).
PROGRESS_FROM = 200

#: Нижче цього канал зробить заливку кадрів довшою за саму роботу.
MIN_MBPS = 20.0


def probe(url: str, *, want_bytes: int = PROBE_BYTES) -> dict[str, Any]:
    """Код відповіді й швидкість. Порожній URL — окремий діагноз, не помилка."""
    if not url:
        return {"ok": False, "http": "", "mbps": 0.0, "why": "посилання порожнє", "sec": 0.0}
    t0 = time.monotonic()
    proc = subprocess.run(
        ["curl", "-o", os.devnull, "-s", "--max-time", str(PROBE_TIMEOUT),
         "-r", f"0-{want_bytes - 1}",
         # Кожне значення окремим рядком: спільний рядок уже давав «HTTP 2060»,
         # коли швидкість потрапляла в поле коду.
         "-w", "http=%{http_code}\nbps=%{speed_download}\n", url],
        capture_output=True, text=True, check=False,
    )
    fields = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
    http = (fields.get("http") or "").strip()
    try:
        mbps = float(fields.get("bps") or 0) * 8 / 1e6
    except ValueError:
        mbps = 0.0
    ok = http in ("200", "206")
    why = ""
    if not ok:
        why = {
            "403": "ПРОТЕРМІНОВАНО або підпис не збігається",
            "404": "об'єкта немає в бакеті",
            "000": "не достукались (мережа/DNS) — це бік ХОСТА, не наш конфіг",
        }.get(http, f"HTTP {http or '—'}")
    return {"ok": ok, "http": http, "mbps": round(mbps, 1), "why": why,
            "sec": round(time.monotonic() - t0, 1)}


def exists(url: str) -> bool:
    """Чи існує об'єкт. Один байт GET — не HEAD (підпис виданий під GET)."""
    if not url:
        return False
    proc = subprocess.run(
        ["curl", "-o", os.devnull, "-s", "--max-time", "15", "-r", "0-0",
         "-w", "http=%{http_code}\n", url],
        capture_output=True, text=True, check=False,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("http="):
            return line.split("=", 1)[1].strip() in ("200", "206")
    return False


def alive(url: str) -> dict[str, Any]:
    """Живе посилання чи ні — один байт, без заміру швидкості (`mbps` = None)."""
    got = probe(url, want_bytes=1)
    got["mbps"] = None
    return got


def ckpt_found(resume: list[str], limit: int) -> tuple[int, int]:
    """(скільки чекпоінтів є, скільки слотів спитали).

    Раннер заповнює слоти суцільно з першого: невдала заливка не зсуває номер,
    а повторюється в той самий слот наступним раундом (`_CkptSink.push`). Тож
    перший відсутній слот означає, що далі теж порожньо, — і свіжа справа
    коштує один запит замість восьми.
    """
    found = checked = 0
    for url in resume[: max(0, limit)]:
        checked += 1
        if not exists(url):
            break
        found += 1
    return found, checked


def _check_case(case: dict[str, Any], *, ckpt_probe: int,
                measure: bool) -> tuple[dict[str, Any], list[str], list[str]]:
    """Перевірка однієї справи плану з R2: (запис звіту, проблеми, нотатки)."""
    problems: list[str] = []
    notes: list[str] = []
    name = case.get("case")
    url = str(case.get("pages_url") or "")
    pages = probe(url) if measure else alive(url)
    resume = [str(u) for u in (case.get("resume_urls") or [])]
    found, checked = ckpt_found(resume, ckpt_probe)
    out_dir = str(case.get("out_dir") or "")
    entry = {
        "case": name,
        "n_pages": case.get("n_pages"),
        "out_dir": out_dir,
        "pages_url": pages,
        "ckpt_urls": len(case.get("ckpt_urls") or []),
        "resume_urls": len(resume),
        "ckpt_found": found,
        "ckpt_checked": checked,
    }
    # Справа частинами: `pages_url` — перша, решту питаємо дешево (один
    # байт), бо бокс качатиме кожну, де є недочитане.
    dead = [str(p.get("name") or f"№{n}")
            for n, p in enumerate(case.get("pages_parts") or [], 1)
            if isinstance(p, dict) and n > 1 and not exists(str(p.get("url") or ""))]
    entry["pages_parts"] = len(case.get("pages_parts") or [])
    if dead:
        problems.append(
            f"{name}: у сховищі немає частин архіву кадрів: {', '.join(dead[:4])}"
            + ("…" if len(dead) > 4 else ""))
    if not pages["ok"]:
        problems.append(f"{name}: кадри — {pages['why']}")
    elif pages["mbps"] is not None and pages["mbps"] < MIN_MBPS:
        # 🔴 Швидкість тут — це НАШ домашній аплінк до R2, а не канал боксу.
        # Предмет рішення — скільки качатиме орендована машина, і його
        # міряють в іншому місці (ворота на живому боксі: 153-787 Мбіт/с).
        notes.append(
            f"{name}: наш канал до R2 {pages['mbps']:.0f} Мбіт/с — це домашній "
            f"аплінк, не канал боксу; на оренду не впливає")
    # 🔴 Без `out_dir` результат ляже в теку запуску, тобто в чужий простір.
    if not out_dir:
        problems.append(
            f"{name}: у плані немає `out_dir` — забір не знатиме, якому "
            f"простору належить справа")
    elif not Path(out_dir).is_absolute():
        problems.append(f"{name}: `out_dir` не абсолютний: {out_dir}")
    if not resume:
        problems.append(
            f"{name}: НЕМАЄ resume-посилань — при смерті боксу справа піде з нуля")
    # Справа без жодного чекпоінта описується в зведенні (`check_plan`): на
    # тисячі свіжих справ тисяча однакових рядків сховала б справжні вади.
    return entry, problems, notes


#: Скільки справ без чекпоінтів називати поіменно; далі — одним рядком.
FRESH_BY_NAME = 5


def _check_box_plan(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Передполіт для складу на машині: чи все, що поїде, лежить на диску."""
    assets_path = str(plan.get("assets_path") or "")
    size = Path(assets_path).stat().st_size if Path(assets_path).is_file() else 0
    report["assets"] = {"ok": bool(size), "http": None, "mbps": 0.0,
                        "why": "" if size else f"немає архіву ассетів: {assets_path}"}
    if not size:
        report["problems"].append(report["assets"]["why"])
    total = size
    for case in plan.get("cases") or []:
        name = case.get("case")
        pages_path = Path(str(case.get("pages_path") or ""))
        # Справа частинами: на диску мусить лежати КОЖНА, а не лише перша.
        archives = [Path(str(p.get("path") or "")) for p in case.get("pages_parts") or []
                    if isinstance(p, dict)] or [pages_path]
        absent = [a for a in archives if not a.is_file()]
        ok = not absent
        if absent:
            pages_path = absent[0]
        total += sum(a.stat().st_size for a in archives if a.is_file())
        out_dir = str(case.get("out_dir") or "")
        report["cases"].append({
            "case": name, "n_pages": case.get("n_pages"), "out_dir": out_dir,
            "pages_url": {"ok": ok, "http": None, "mbps": 0.0,
                          "why": "" if ok else f"немає архіву кадрів: {pages_path}"},
            "ckpt_urls": int(case.get("ckpt_slots") or 0),
            "resume_urls": 0, "ckpt_found": 0, "ckpt_checked": 0,
        })
        if not ok:
            report["problems"].append(f"{name}: немає архіву кадрів {pages_path}")
        if not out_dir:
            report["problems"].append(
                f"{name}: немає `out_dir` — результат ляже в теку запуску, "
                f"тобто в чужий простір")
    # 🔴 `scp` — єдине, чим ми веземо дані при цьому транспорті, і його може
    # просто не бути (Windows без OpenSSH-клієнта). Без цієї перевірки відмова
    # спливає ПІСЛЯ оренди, у воротах: бокс беруть, доставка падає, бокс
    # знищують — і так до стелі переоренд, за детерміновану локальну помилку.
    if not shutil.which("scp"):
        report["problems"].append(
            "немає системного `scp` — саме ним транспорт `box` везе дані на "
            "машину. Поставте OpenSSH-клієнт (Windows: «Додаткові компоненти» "
            "→ «Клієнт OpenSSH») або використайте `--transport r2`")
    report["notes"].append(
        f"склад на самій машині: повеземо {total / 1e6:.0f} МБ по SSH. "
        f"🔴 Чекпоінти лежатимуть на тій самій машині, тож наглядач забирає їх "
        f"додому сам — смерть машини інакше з'їла б усю роботу")
    return report


def check_plan(plan_path: Path, *, ckpt_probe: int = 8) -> dict[str, Any]:
    """Звіт про придатність плану. `problems` порожній — можна орендувати."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {"plan": str(plan_path), "cases": [], "problems": [],
                              "notes": []}
    if plan.get("staged") is False:
        report["problems"].append(
            "план складено лише для кошторису (`htr plan --no-upload`): кадри й "
            "ассети не залиті — складіть план заново без `--no-upload`")
        return report
    if str(plan.get("transport") or "r2") == "box":
        # 🔴 Перевіряти нічого «на тому кінці»: посилань ще не існує, бо їх
        # видасть машина, якої поки немає. Питання, на яке передполіт відповідає
        # тут, інше — чи є на диску те, що ми збираємось везти.
        return _check_box_plan(plan, report)

    assets = probe(str(plan.get("assets_url") or ""))
    report["assets"] = assets
    if not assets["ok"]:
        report["problems"].append(f"assets_url: {assets['why']}")

    cases = list(plan.get("cases") or [])
    # Швидкість міряється на ПЕРШІЙ справі: це наш домашній канал до того самого
    # бакета, і на кожній наступній він був би тим самим числом.
    results: list[tuple[dict[str, Any], list[str], list[str]] | None] = [None] * len(cases)

    def one(i: int) -> None:
        results[i] = _check_case(cases[i], ckpt_probe=ckpt_probe, measure=i == 0)

    with ThreadPoolExecutor(max_workers=max(1, min(PARALLEL, len(cases)))) as pool:
        for done, _ in enumerate(pool.map(one, range(len(cases))), 1):
            if len(cases) >= PROGRESS_FROM and (done % PROGRESS_FROM == 0
                                                or done == len(cases)):
                print(f"передполіт: {done}/{len(cases)} справ", file=sys.stderr,
                      flush=True)

    fresh: list[str] = []
    for got in results:
        assert got is not None
        entry, problems, notes = got
        report["cases"].append(entry)
        report["problems"].extend(problems)
        report["notes"].extend(notes)
        if entry["resume_urls"] and not entry["ckpt_found"]:
            fresh.append(str(entry["case"]))
    # 🔴 «Посилання є» ≠ «є що відновлювати». Звіт друкував «точки відновлення
    # на місці» при 0/8 знайдених — тобто давав письмове запевнення про
    # відновлюваність, якої немає.
    if 0 < len(fresh) <= FRESH_BY_NAME:
        report["notes"].extend(
            f"{name}: чекпоінтів ЩЕ НЕМАЄ — при смерті боксу справа піде з нуля, "
            f"поки не з'явиться перший" for name in fresh)
    elif fresh:
        report["notes"].append(
            f"чекпоінтів ЩЕ НЕМАЄ у {len(fresh)} з {len(cases)} справ "
            f"({', '.join(fresh[:3])}…) — при смерті боксу такі справи підуть з нуля, "
            f"поки не з'явиться перший")

    report["ok"] = not report["problems"]
    return report
