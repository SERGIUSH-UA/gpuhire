"""Зібрати декод справи з ЧЕКПОІНТІВ у R2 — без оренди й без наглядача.

Коли наглядач помирає ПІСЛЯ роботи, але ДО забору, готовий текст лишається жити
в `ckpt/<справа>/<модель>/ckpt_NNNN.tgz`, і більше ніде: у `results/` його
немає, на диску немає, а бокс уже погашено. Переорендувати заради цього означає
заплатити вдруге за вже пораховане.

Ціна незнання виміряна: в одному з просторів так тихо загубились 16
васильківських справ. Півтора тижня вони стояли у звітах як невиконана робота —
забір повернув 11 справ повністю, 5538 сторінок, за кілька хвилин і $0.00.

🔴 ЧЕКПОІНТИ ІНКРЕМЕНТАЛЬНІ: кожен несе лише сторінки, ДОДАНІ від попереднього.
Тому розпаковувати треба ВСІ й СТРОГО за номером — інакше в теці опиниться
випадкова підмножина, і вона виглядатиме як повний декод.

🔴🔴 Тека призначення береться з `out_dir` ПЛАНУ й нізвідки більше. Доти скрипт
мав власний корінь, і 95 сторінок обома голосами лягли в чужий простір, при
цьому чесно відзвітувавши «на диску Писар 95 · Дяк 95» — саме тому помилку було
легко не помітити.
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from gpurunner.htr import r2

#: людські імена гілок для звіту (ланцюжок `.replace` тут давав «ПисарДяк»)
BRANCH_NAME = {"out": "Писар", "out-diak_v4": "Дяк", "out-skryba_v6": "Скриба"}


def branch_suffix(branch: str) -> str | None:
    """Тека всередині чекпоінта → суфікс теки прогону; не декод — None.

    🔴 Правило, а не перелік: голосів у прогоні стільки, скільки дали `voices`,
    і зашитий перелік мовчки викидав би будь-яку нову версію (`out-skryba_v7`).
    """
    if branch == "out":
        return ""
    if branch.startswith("out-") and len(branch) > 4:
        return branch[3:]
    return None


def branch_name(branch: str) -> str:
    return BRANCH_NAME.get(branch) or branch[4:] or branch


def _branches_on_disk(out_root: Path) -> list[str]:
    """Гілки, що вже лежать поруч: `out` + сестринські `<прогін>-<тег>`."""
    found = ["out"]
    if out_root.parent.is_dir():
        found += sorted(f"out-{d.name[len(out_root.name) + 1:]}"
                        for d in out_root.parent.glob(f"{out_root.name}-*") if d.is_dir())
    return found

_RE_CKPT = re.compile(r"/ckpt_(\d+)\.tgz$")


def ckpt_keys(case: str, *, bucket: str = "", s3: Any = None) -> list[str]:
    """Ключі чекпоінтів справи, ВІДСОРТОВАНІ за номером (не за іменем).

    🔴 Лексичне сортування зламалось би на `ckpt_0010` проти `ckpt_0009` лише з
    появою десятого — тобто саме тоді, коли справа велика й помилка дорога.
    """
    found: list[tuple[int, str]] = []
    for obj in r2.ls(f"ckpt/{case}", bucket=bucket, s3=s3):
        match = _RE_CKPT.search(obj["key"])
        if match:
            found.append((int(match.group(1)), obj["key"]))
    return [key for _, key in sorted(found)]


def count_ckpt_via_urls(resume_urls: list[str]) -> int:
    """Скільки чекпоінтів реально лежить у R2 — БЕЗ ключів, самими посиланнями.

    Потрібно там, де ключів немає (наглядач їх не має принципово): один
    `Range: 0-0` на посилання, 404 означає, що ряд скінчився. Дешево й чесно.
    """
    seen = 0
    for url in resume_urls:
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        try:
            with urllib.request.urlopen(request, timeout=20) as resp:
                if resp.status in (200, 206):
                    seen += 1
                    continue
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                break
            continue
        except OSError:
            continue
        break
    return seen


def fetch_url(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=180) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def unpack(tgz: Path, case: str, out_root: Path, *, dry: bool = False) -> dict[str, int]:
    """Розкласти один чекпоінт по гілках. Повертає {гілка: скільки .txt}.

    🔴 Гілки голосів ідуть у СЕСТРИНСЬКІ теки (`<прогін>-diak_v4`), як їх шукає
    споживач декоду. Без цього пошук по прогону дасть нуль хітів БЕЗ помилки —
    тобто успішний дорогий захід прочитається як негативний результат.
    """
    added: dict[str, int] = {}
    with tarfile.open(tgz, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            suffix = branch_suffix(parts[0]) if len(parts) >= 2 else None
            if suffix is None:
                continue  # data/, logs/ — не декод
            branch = parts[0]
            target = out_root.parent / f"{out_root.name}{suffix}" / Path(
                *parts[1:])
            if member.name.endswith(".txt"):
                added[branch] = added.get(branch, 0) + 1
            if dry:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, target.open("wb") as fh:
                shutil.copyfileobj(src, fh)
    return added


def stamp_meta(out_root: Path, *, case_key: str = "", frames_dir: Path | None = None,
               branches: list[str] | None = None) -> None:
    """Проставити в меті шифру справи й ЛОКАЛЬНУ теку кадрів.

    🔴 `case_dir` у меті з хмари — шлях КОНТЕЙНЕРА (`/tmp/htrcase/pages_dl_NN`),
    а `case_key` раннер лишає порожнім. Локально це означає «кропів немає» —
    тобто очний крок, заради якого прогін і робиться, стає неможливим. Стара
    адреса лишається в `case_dir_cloud` як слід походження.
    """
    for branch in branches or _branches_on_disk(out_root):
        meta_path = out_root.parent / f"{out_root.name}{branch_suffix(branch)}" / "_htr_meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        changed = False
        # Шифра головніша за шлях: тека кадрів може переїхати, ключ лишиться.
        if case_key and not str(meta.get("case_key") or "").strip():
            meta["case_key"] = case_key
            changed = True
        if frames_dir is not None and not Path(str(meta.get("case_dir") or "")).is_dir():
            meta["case_dir_cloud"] = meta.get("case_dir")
            meta["case_dir"] = str(frames_dir)
            changed = True
        if changed:
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def count_on_disk(out_root: Path, branches: list[str] | None = None) -> dict[str, int]:
    """Скільки `.txt` НА ДИСКУ по гілках.

    🔴 Приймач — саме диск, а не сума з архівів: пізніші чекпоінти перекривають
    ранні тими самими іменами, тож сума завищує.
    """
    disk: dict[str, int] = {}
    for branch in branches or ["out"]:
        directory = out_root.parent / f"{out_root.name}{branch_suffix(branch)}"
        disk[branch] = len(list(directory.glob("*.txt"))) if directory.is_dir() else 0
    return disk


def cases_from_plan(plan_path: Path) -> list[dict[str, Any]]:
    """Справи з плану разом із тим, чого з самої назви прогону не відновити.

    🔴 План несе ТРИ речі: знаменник, локальну теку кадрів і шифру справи. Доки
    забір їх не читав, він шукав теку здогадом за суфіксом імені, а шифру не
    ставив узагалі — і прогін приїжджав нічиїм.
    """
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cases = []
    for case in plan.get("cases", []):
        out_dir = str(case.get("out_dir") or "")
        if not out_dir:
            raise ValueError(
                f"{case.get('case')}: у плані немає `out_dir`. Без нього забір не "
                f"знає, ЯКОМУ простору належить справа, і покладе декод поруч із "
                f"собою — саме так 95 сторінок опинились у чужому проєкті.")
        cases.append({
            "case": str(case.get("case") or ""),
            "n_pages": int(case.get("n_pages") or 0),
            "out_dir": out_dir,
            "case_dir": str(case.get("case_dir") or ""),
            "case_key": str(case.get("case_key") or ""),
            "resume_urls": list(case.get("resume_urls") or []),
        })
    return cases


def fetch_case(rec: dict[str, Any], *, bucket: str = "", hours: float = 3.0,
               dry: bool = False, s3: Any = None, log: Any = None) -> dict[str, Any]:
    """Забрати одну справу. Повертає підсумок із числами З ДИСКА."""
    log = log or (lambda line: print(line, flush=True))
    case = str(rec["case"])
    out_root = Path(str(rec["out_dir"]))
    expected = int(rec.get("n_pages") or 0)
    keys = ckpt_keys(case, bucket=bucket, s3=s3)
    if not keys:
        log(f"❌ {case}: чекпоінтів у R2 немає")
        return {"case": case, "ckpt": 0, "disk": count_on_disk(out_root), "ok": False}

    # 🔴 Приймач — ті гілки, що приїхали в ЦЬОМУ прогоні, а не всі сестринські
    # теки: окреме перечитування тієї ж справи іншою моделлю тут чуже
    branches: list[str] = ["out"]
    with tempfile.TemporaryDirectory(prefix="htrckpt-") as tmp:
        for key in keys:
            url = r2.get_url(key, hours=hours, bucket=bucket, s3=s3)
            local = Path(tmp) / Path(key).name
            fetch_url(url, local)
            for branch in unpack(local, case, out_root, dry=dry):
                if branch not in branches:
                    branches.append(branch)
            local.unlink(missing_ok=True)

    frames = Path(str(rec.get("case_dir") or ""))
    if not dry:
        stamp_meta(out_root, case_key=str(rec.get("case_key") or ""),
                   frames_dir=frames if str(frames) and frames.is_dir() else None,
                   branches=branches)

    disk = count_on_disk(out_root, branches)
    ok = expected == 0 or all(n >= expected for n in disk.values())
    # 🔴 Абсолютний шлях у звіті. Рядок «на диску N» без нього читається як
    # успіх навіть тоді, коли N лягло в чужий простір.
    log(f"{'✅' if ok else '⚠'} {case}: чекпоінтів {len(keys)} · "
        + " · ".join(f"{branch_name(b)} {n}" for b, n in disk.items())
        + (f" · очікувано {expected}" if expected else "")
        + f"\n   → {out_root}")
    return {"case": case, "ckpt": len(keys), "disk": disk, "ok": ok,
            "out_dir": str(out_root)}
