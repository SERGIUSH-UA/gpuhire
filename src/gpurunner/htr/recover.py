"""Довести захід до кінця, коли наглядач помер: стан бокса, забір, повнота, гроші.

Наглядач живе на домашній машині, бокс — у хмарі. Коли домашня машина вимкнулась
(сесія обірвалась, ноутбук закрили), бокс працює далі: читає, пише чекпоінти й
серцебиття в сховище, потім гасить себе сам. Повертаючись, агент мусить мати
одну команду, яка скаже, що сталося, забере прочитане й порахує гроші, яких
наглядач уже не бачив.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Скільки рядків хвоста логу бокса показувати.
LOG_TAIL = 15


def read_beat(url: str) -> dict[str, Any] | None:
    """Серцебиття бокса зі сховища: `beat.json`, прогрес і хвіст логу. None — немає."""
    if not url:
        return None
    with tempfile.TemporaryDirectory(prefix="beat-") as tmp:
        local = Path(tmp) / "beat.tgz"
        rc = subprocess.call(["curl", "-fsSL", "--max-time", "120", "-o", str(local), url],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rc != 0 or not local.is_file():
            return None
        try:
            with tarfile.open(local) as tf:
                files = {m.name: tf.extractfile(m).read()  # type: ignore[union-attr]
                         for m in tf.getmembers() if m.isfile()}
        except (tarfile.TarError, OSError):
            return None
    beat: dict[str, Any] = {}
    if "beat.json" in files:
        beat = json.loads(files["beat.json"].decode("utf-8", "replace"))
    if "_progress.json" in files:
        with contextlib.suppress(ValueError):
            beat["progress"] = json.loads(files["_progress.json"].decode("utf-8", "replace"))
    log = files.get("_runner.log", b"").decode("utf-8", "replace").splitlines()
    beat["log_tail"] = log[-LOG_TAIL:]
    return beat


def _iso_to_t(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


def unseen_spend(state: dict[str, Any], beat: dict[str, Any] | None, *,
                 box_alive: bool, autodestroy_hours: float,
                 now: float | None = None) -> tuple[float, str]:
    """Гроші, яких наглядач уже не бачив: ціна за годину × час після його смерті.

    Кінець тарифікації: живий бокс — зараз; бокс, що доробив (`final`), — не
    пізніше за останнє серцебиття плюс очікування самознищення; бокс, що замовк
    без `final`, — останнє серцебиття (плюс до одного інтервалу).
    Повертає (долари, як пораховано).
    """
    now = now or time.time()
    box = state.get("box") or {}
    dph = float(box.get("dph_total") or 0)
    died = _iso_to_t(str(state.get("updated") or ""))
    if dph <= 0 or died <= 0:
        return 0.0, "ціни бокса або часу смерті наглядача в стані немає"
    if box_alive:
        end, how = now, "бокс досі живий — до цієї миті"
    elif beat and beat.get("t"):
        last = float(beat["t"])
        if beat.get("final"):
            end = last + autodestroy_hours * 3600.0
            how = (f"робота закінчилась {datetime.fromtimestamp(last, UTC):%H:%M} UTC, "
                   f"плюс не більше {autodestroy_hours:g} год очікування самознищення (верхня межа)")
        else:
            end = last + 300.0
            how = (f"останнє серцебиття {datetime.fromtimestamp(last, UTC):%H:%M} UTC "
                   f"без завершення, плюс до 5 хв")
    else:
        return 0.0, "серцебиття немає (план старший за нього) — оцінити нема з чого"
    return max(0.0, end - died) / 3600.0 * dph, how


def session_instances(state: dict[str, Any], log_path: Path | None) -> list[str]:
    """Усі інстанси сесії: поле стану, а для старих сесій — рядки лога наглядача."""
    import re

    ids = [str(i) for i in state.get("instances") or []]
    box_id = str((state.get("box") or {}).get("instance_id") or "")
    if box_id:
        ids.append(box_id)
    if log_path is not None and log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        ids += re.findall(r"інстанс (\d{6,}) створено", text)
    return list(dict.fromkeys(i for i in ids if i))


def billed(charges: list[dict[str, Any]], instances: list[str]) -> dict[str, Any]:
    """Рахунок Vast за інстанси сесії: разом і за статтями."""
    wanted = set(instances)
    total = 0.0
    parts: dict[str, float] = {}
    seen: set[str] = set()
    for row in charges:
        if row.get("instance") not in wanted:
            continue
        seen.add(str(row["instance"]))
        total += float(row.get("amount") or 0)
        for kind, amount in (row.get("items") or {}).items():
            parts[kind] = parts.get(kind, 0.0) + float(amount)
    return {"total": total, "parts": parts, "instances": sorted(seen)}


def reconcile_sessions(state_root: Path, charges: list[dict[str, Any]], *,
                       days: float = 3.0, apply: bool = True,
                       now: float | None = None) -> list[tuple[str, float, float]]:
    """Звірити завершені сесії з рахунком Vast: (сесія, було, стало).

    Рахунок приходить із запізненням, тож цифра, записана наглядачем на фініші,
    часто неповна. Тут витрачене сесії стає більшим із записаного й рахунку за
    її машини.
    """
    now = now or time.time()
    out: list[tuple[str, float, float]] = []
    for path in sorted(state_root.glob("*.json")):
        if path.name.startswith("latest-"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("phase") != "finished":
            continue
        if now - _iso_to_t(str(data.get("updated") or "")) > days * 86400:
            continue
        ids = session_instances(data, path.with_suffix(".log"))
        bill = billed(charges, ids)
        if not bill["instances"]:
            continue
        budget = dict(data.get("budget") or {})
        old = float(budget.get("spent_usd") or 0)
        new = round(max(old, bill["total"]), 4)
        if abs(float(budget.get("billed_usd") or -1) - bill["total"]) < 1e-4:
            continue
        out.append((str(data.get("session") or path.stem), old, new))
        if apply:
            budget["billed_usd"] = round(bill["total"], 4)
            budget["spent_usd"] = new
            data["budget"] = budget
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, path)
    return out


def ledger_from_bill(charges: list[dict[str, Any]],
                     handles: list[Any]) -> list[tuple[str, float, str]]:
    """Рахунок по машинах → рядки журналу витрат (прогін, $, час).

    Машина прив'язується до прогону за номером інстансу; машина без прогону
    (напр. знищений реєстр) лягає окремим рядком `vast-instance-<id>`, щоб
    місячна сума журналу дорівнювала рахунку.
    """
    per: dict[str, float] = {}
    day: dict[str, str] = {}
    for row in charges:
        inst = str(row.get("instance") or "")
        if not inst:
            continue
        per[inst] = per.get(inst, 0.0) + float(row.get("amount") or 0)
        if row.get("day") and inst not in day:
            day[inst] = datetime.fromtimestamp(float(row["day"]), UTC).isoformat(timespec="seconds")
    by_remote = {str(getattr(h, "remote_id", "") or ""): h for h in handles}
    out = []
    for inst, usd in per.items():
        h = by_remote.get(inst)
        if h is not None:
            when = str(getattr(h, "created_at", "") or day.get(inst, ""))
            out.append((str(h.id), usd, when))
        else:
            out.append((f"vast-instance-{inst}", usd, day.get(inst, "")))
    return out


def patch_state(path: Path, *, extra_usd: float, how: str, verdict: str, why: str,
                billed_usd: float | None = None) -> None:
    """Дописати підсумок у стан сесії — атомарно, не змінюючи решти полів.

    `billed_usd` — рахунок Vast: тоді він і є витратою сесії, а не оцінка.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    budget = dict(data.get("budget") or {})
    seen = float(budget.get("spent_usd") or 0)
    if billed_usd is not None:
        budget["spent_usd"] = round(billed_usd, 4)
        budget["billed_usd"] = round(billed_usd, 4)
        extra_usd = max(0.0, billed_usd - seen)
    else:
        budget["spent_usd"] = round(seen + extra_usd, 4)
    budget["unseen_usd"] = round(extra_usd, 4)
    data["budget"] = budget
    data.setdefault("incidents", []).append({
        "kind": "recovered",
        "detail": f"наглядача не було; добрано без нього, гроші поза наглядом ${extra_usd:.3f} ({how})",
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    data["phase"] = "finished"
    data["verdict"] = verdict
    data["why"] = why
    data["updated"] = datetime.now(UTC).isoformat(timespec="seconds")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def last_store_write(prefixes: list[str], *, list_fn: Any = None) -> float:
    """Коли бокс востаннє щось записав у сховище: найпізніший об'єкт під префіксами.

    Запасна мірка для планів без серцебиття: кожен чекпоінт — це слід живого бокса.
    0 — нічого не знайдено.
    """
    if list_fn is None:
        from gpurunner.htr import r2

        list_fn = r2.ls
    last = 0.0
    for prefix in prefixes:
        if not prefix:
            continue
        for obj in list_fn(prefix.rstrip("/") + "/"):
            mod = obj.get("modified")
            t = mod.timestamp() if hasattr(mod, "timestamp") else _iso_to_t(str(mod or ""))
            last = max(last, t)
    return last


def beat_summary(beat: dict[str, Any] | None) -> list[str]:
    """Людські рядки про серцебиття."""
    if not beat:
        return ["серцебиття в сховищі немає"]
    lines = []
    t = beat.get("t")
    if t:
        lines.append(f"останнє серцебиття: {datetime.fromtimestamp(float(t), UTC):%Y-%m-%d %H:%M} UTC"
                     + (" · робота ЗАКІНЧИЛАСЬ" if beat.get("final") else " · роботу не закінчено"))
    lines.append(f"прочитано на боксі: {beat.get('box_pages_done', '?')} стор. "
                 f"(підхоплено з чекпоінтів {beat.get('box_pages_resumed', '?')})")
    done = [r for r in beat.get("results") or [] if r.get("complete")]
    if beat.get("results"):
        lines.append(f"справ закрито боксом: {len(done)} повних із {len(beat['results'])}")
    return lines

