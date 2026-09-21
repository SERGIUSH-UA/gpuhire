"""HTRCaseRunner — прогін справи НАШИМ ЖЕ ``htr_case_run.py``, лише на чужій карті.

Чому subprocess, а не порт логіки в раннер: ``htr_case_run.py`` (раннер справи
nyshporka) — це 2200 рядків, у яких сидять два патчі гарячих функцій kraken, детектор
орієнтації, посторінковий CLAHE, автопідйом стелі сегментації та kraken-голос із
``pad=16``. Порт розійшовся б із локальним конвеєром ТИХО — і замір швидкості
порівнював би вже не карти, а два різні конвеєри. Тут їде той самий файл із тими
самими прапорцями; різниця тільки в залізі.

‼ РЕЗУЛЬТАТ НЕ МОЖНА ВТРАТИТИ. Довгий прогін гине трьома способами, і кожен
обробляється окремо:
  · виняток у раннері      → бекенд синхронізує ``/kaggle/working`` у ``finally``;
  · вбитий контейнер       (таймаут, витиснення, стеля витрат) — ``finally`` НЕ
    відпрацює, тому власний тред кожні ``checkpoint_sec`` копіює новий текст на
    змонтований том; вікно втрати = інтервал, а не весь прогін;
  · кінець грошей/часу між запусками → ``state_slug`` тримає копію на ВХІДНОМУ
    томі, і наступний запуск бачить її як стартовий стан.
Сам ``htr_case_run.py`` пише посторінково і при рестарті пропускає готові
сторінки (``known_pages``), тож чекпоінт — це просто його ж тека, донесена до
наступного запуску.

Вхід ``/kaggle/input/``:
  ``<pages_slug>/``   — кадри справи (*.jpg / *.png), рекурсивно
  ``<models_slug>/``  — ваги: ``*.pt`` (Писар) + ``*.mlmodel`` (Дяк-голос)
  ``<scripts_slug>/`` — htr_case_run.py, pysar_lines_infer.py, gpu_sato.py, fast_geom.py,
  seg_ceiling.py, seg_resize.py

Вихід ``/kaggle/working/``: ``out*/`` (тексти, рамки рядків, мета) і
``htr_case_summary.json`` — саме останній і є замір: с/стор, рядків/с, паралелізм.

⚠ БЕЗ ``from __future__ import annotations`` — код інжектиться після PARAMS.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

#: Скрипти, без яких прогін не поїде. Патчі теж обов'язкові: без них сегментація
#: іде старим CPU-шляхом (30-43 с/стор при util 0%) і замір збрехав би вдвічі.
#: `seg_ceiling` імпортується з `main` БЕЗУМОВНО — його брак валить прогін уже
#: після завантаження моделей, тобто на оплачуваній карті.
#: ⚠ `clan_anchors` / `clan_review` / `htr_clan_scan` сюди НЕ входять: вони
#: потрібні лише рятувальному проходу (`--rescue`) і тягнуть приватний пакет замовника, якого
#: в контейнері немає. Тому рятунок тут недоступний — він і не потрібен, це вже
#: пошук по готовому тексту, а не декод.
NEEDED_SCRIPTS = ("htr_case_run.py", "pysar_lines_infer.py", "gpu_sato.py",
                  "fast_geom.py", "seg_ceiling.py", "seg_resize.py")

PAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

#: Куди бекенд монтує вихідний том. Відносний шлях — не помилка: Beam монтує
#: томи під робочою текою контейнера і АБСОЛЮТНИЙ ігнорує мовчки.
OUT_MOUNTS = ("./outputs", "/mnt/outputs", "outputs")
IN_MOUNTS = ("./inputs", "/mnt/inputs", "inputs")


def _slug_dir(slug):
    """Тека слага під /kaggle/input — або перша, що містить його ім'я."""
    direct = KAGGLE_INPUT / slug
    if direct.is_dir():
        return direct
    for cand in sorted(KAGGLE_INPUT.rglob(slug)):
        if cand.is_dir():
            return cand
    raise RuntimeError("немає теки входу %r під %s" % (slug, KAGGLE_INPUT))


def _mount(candidates):
    for c in candidates:
        p = Path(c)
        if p.is_dir():
            return p.resolve()
    return None


def _pages_of(root):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in PAGE_SUFFIXES)


def _script_pages(case_dir):
    """Сторінки РІВНО так, як їх бачить сам ``htr_case_run.py``.

    🔴 Не те саме, що ``_pages_of``: скрипт бере ``iterdir()`` (без вкладених
    тек) і лише jpg/jpeg/png. Індекси для ``--pages`` мусять рахуватись у
    ЙОГО порядку, інакше догінний прохід перечитає не ті кадри — і зробить це
    мовчки, бо номери будуть валідні.
    """
    return sorted(p for p in Path(case_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))


#: Канал прогресу самого ``htr_case_run.py`` (`--progress-json`).
PROGRESS_PREFIX = "@@PROGRESS@@ "
#: Куди наглядач дивиться замість логів.
PROGRESS_PATH = Path("/workspace/gpurunner/_progress.json")
#: 🔴 Один рядок на ПРОГІН — на відміну від «✓ готово:», яке друкує КОЖЕН шард.
#: Саме на тій фразі конвеєр вирішив, що справа готова, і забрав 203 з 323
#: сторінок; шукати треба те, чого не може бути більше одного.
DONE_SENTINEL = "@@HTRCASE_DONE@@ "

#: Тиша шарда, після якої він вважається завислим. Здоровий HTR емітить подію
#: на кожній сторінці (найповільніша — десятки секунд), тож 10 хвилин мовчання
#: means OOM або дедлок. Те саме число, що в локальній черзі Нишпорки.
STALL_SEC_DEFAULT = 600
WATCH_TICK = 5
#: Скільки разів піднімати шард, що не дає прогресу. Лічильник — саме
#: «поспіль БЕЗ прогресу»: будь-яка нова сторінка його обнуляє.
SHARD_RESTART_MAX = 3
#: Скільки разів та сама сторінка має підвісити шард, щоб піти в карантин.
#: Перший раз не карантинить: сусідній OOM не має викидати справний аркуш.
QUARANTINE_HITS = 2


def _quarantine_add(out_dir, page, reason):
    """Дописати сторінку в карантин — його читає сам ``htr_case_run.py``."""
    path = Path(out_dir) / "_htr_quarantine.json"
    data = {"version": 1, "pages": {}}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("pages"), dict):
                data = loaded
        except Exception:
            pass
    data.setdefault("pages", {})[str(page)] = {"reason": reason, "at": _utc_iso()}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _kill_tree(proc):
    """Убити ГРУПУ процесів шарда, а не голову.

    ``htr_case_run.py`` тримає CUDA-контекст; ``terminate()`` голови лишає
    дітей живими, і вони далі тримають VRAM — тобто перезапуск одразу впаде
    в OOM.
    """
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _start_shard(k, shards, base, env, logs_dir, fleet):
    """Підняти шард k і повісити на нього помпу прогресу.

    ``start_new_session=True`` — щоб вотчдог міг убити ГРУПУ (див. _kill_tree).
    stdout іде через PIPE, а не одразу у файл: помпа і в лог пише, і читає з
    нього події прогресу. Без цього тиша шарда невідрізненна від роботи.
    """
    cmd = list(base)
    if shards > 1:
        cmd += ["--shard", "%d/%d" % (k + 1, shards)]
        if fleet.get("dynamic"):
            # 🧲 клейми: шард бере кадри з повного списку, а не свій зріз
            cmd.append("--claim")
    # 🔴 Шард k їде на карту k % N. Без цього рядка ВСІ шарди сідали на
    # `cuda:0`, і бокс із 2/4/8 картами працював рівно як однокартковий:
    # решта карт простоювала, а бюджет VRAM рахувався по всій машині.
    n_gpus = max(1, int(fleet.get("n_gpus") or 1))
    if n_gpus > 1:
        device = "cuda:%d" % (k % n_gpus)
        for i, arg in enumerate(cmd):
            if arg == "--device":
                cmd[i + 1] = device
                break
        fleet["shards_by_gpu"] = fleet.get("shards_by_gpu") or {}
        fleet["shards_by_gpu"][str(k + 1)] = device
        # Лок — свій на кожну карту: спільний звів би 8 карт до однієї.
        for i, arg in enumerate(cmd):
            if arg == "--gpu-lock":
                cmd[i + 1] = "%s.%d" % (cmd[i + 1], k % n_gpus)
                break
    # Шард із цим номером стартує НЕзлитим: прапорець від попереднього власника
    # номера (або від минулого флоту) інакше випустив би його після першої ж
    # сторінки.
    out_arg = next((cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == "--out-dir"), None)
    if out_arg:
        _clear_drains(Path(out_arg), k + 1)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, cwd="/tmp/htrcase", text=True, encoding="utf-8", errors="replace",
        bufsize=1, start_new_session=True,
    )
    prev = fleet["shards"].get(k + 1) or {}
    state = {
        "k": k + 1, "pid": proc.pid, "alive": True, "rc": None,
        "i": 0, "n": 0,
        "done": prev.get("done", 0), "failed": prev.get("failed", 0),
        "skipped": prev.get("skipped", 0),
        "inflight": None, "last_out": time.time(),
        "restarts": prev.get("restarts", 0),
        "done_at_restart": prev.get("done", 0),
        "result": None,
    }
    fleet["shards"][k + 1] = state
    thread = threading.Thread(
        target=_pump, args=(proc, state, fleet, logs_dir / ("shard%02d.log" % (k + 1))),
        daemon=True,
    )
    thread.start()
    state["_proc"] = proc
    state["_thread"] = thread
    return proc


def _is_oom_line(line):
    """Слід нестачі пам'яті у виводі шарда: CUDA OOM і `MemoryError` процесора."""
    low = line.lower()
    return "out of memory" in low or "memoryerror" in low


def _pump(proc, state, fleet, log_path):
    """Читати вивід шарда: у лог — усе, у стан — події прогресу.

    Дзеркало ``HtrManager._pump`` із локальної черги. Саме звідси береться
    ``inflight`` — сторінка, яку шард молотить ЗАРАЗ; без неї вотчдог знає,
    що хтось завис, але не знає, на чому.
    """
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            for raw in proc.stdout:
                log.write(raw)
                log.flush()
                state["last_out"] = time.time()
                line = raw.strip()
                if line.startswith(PROGRESS_PREFIX):
                    try:
                        ev = json.loads(line[len(PROGRESS_PREFIX):])
                    except Exception:
                        continue
                    phase = ev.get("phase")
                    if phase == "page_start":
                        state["inflight"] = ev.get("page")
                        state["i"] = ev.get("i") or state["i"]
                        state["n"] = ev.get("n") or state["n"]
                    elif phase == "htr":
                        state["i"] = ev.get("i") or state["i"]
                        state["n"] = ev.get("n") or state["n"]
                        if ev.get("error"):
                            state["failed"] += 1
                            # 🔴 Текст — у стан, а не лише в лог шарда. 11.09.2026
                            # (904-24-198) 34 збої за 45 с поклали флот, наглядач
                            # переорендував бокс, а причина лишилась у логах, які
                            # згоріли разом із ним.
                            state["last_error"] = ("%s: %s" % (
                                ev.get("page") or "?", ev.get("error")))[:300]
                        elif ev.get("skipped"):
                            state["skipped"] += 1
                        else:
                            state["done"] += 1
                        state["inflight"] = None
                        # Час і пам'ять сторінки — те, з чого регулятор флоту
                        # міряє темп і межу VRAM (раннер пише їх у подію `htr`).
                        sec = ev.get("sec")
                        if isinstance(sec, (int, float)) and not ev.get("skipped"):
                            secs = state.setdefault("secs", [])
                            secs.append(float(sec))
                            del secs[:-30]
                        for key in ("vram_peak_mb", "rss_peak_mb"):
                            v = ev.get(key)
                            if isinstance(v, (int, float)):
                                state[key] = max(int(state.get(key) or 0), int(v))
                    elif phase == "done":
                        state["result"] = ev
                elif _is_oom_line(line):
                    # 🔴 Єдиний слід OOM, який узагалі існує: шард ловить його
                    # як звичайний виняток сторінки, лічить у «збої» власного
                    # лога і завершується з rc=0. На рівні job'а це виглядало
                    # як повний успіх — саме так зникли 46 сторінок.
                    fleet["oom_events"] += 1
                    # Сумарний — не обнуляється між раундами звуження флоту:
                    # саме він іде в підсумок і в калібрування.
                    fleet["oom_events_total"] = fleet.get("oom_events_total", 0) + 1
    except Exception:
        pass
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass


def _watch_fleet(fleet, out_dir, *, shards, stall_sec, restart_max, base, env, logs_dir, t0,
                 regulator=None):
    """Вотчдог шардів: тиша → карантин → перезапуск, поки є прогрес.

    Три речі, кожна куплена інцидентом:

    1. **Тиша рахується пошардово.** Спільний лічильник ховав завислий шард
       за виводом сусідів — той міг мовчати годину, поки решта працювала.
    2. **Карантин з другого удару.** Перше зависання може бути наслідком
       OOM сусіда, а не отруйної сторінки; викидати аркуш одразу — втрачати
       справні дані.
    3. **Стеля на перезапуски БЕЗ прогресу, а не на перезапуски.** Шард, що
       після підйому дочитав хоч сторінку, отримує лічильник наново; шард,
       який тричі підряд не зрушив, лишається мертвим і йде у звіт.
    """
    rcs = {}
    last_publish = 0.0
    # Знаменник `--shard k/N` для перезапуску: у режимі регулятора він більший
    # за число живих шардів (номери — лише мітки, див. REG_DENOM).
    denom = int(fleet.get("denom") or shards)
    while True:
        alive = 0
        now = time.time()
        # Набір шардів ЖИВИЙ: регулятор додає й зливає їх на ходу.
        for k in sorted(fleet["shards"]):
            state = fleet["shards"].get(k)
            if state is None or state.get("rc") is not None:
                continue
            proc = state["_proc"]
            rc = proc.poll()
            if rc is not None:
                state["rc"] = rc
                state["alive"] = False
                state["finished_at"] = round(time.time(), 1)
                thread = state.get("_thread")
                if thread is not None:
                    thread.join(timeout=10)
                # 🚰 Злитий шард вийшов СВІДОМО: регулятор сам попросив його не
                # брати нових сторінок. Раннер nyshporka до 12.09.2026 при цьому
                # звітував rc=3 («неповно», бо сусіди ще не дочитали), і справа
                # падала з «шарди впали» саме тоді, коли флот звужувався
                # (904-24-70, 11.09.2026: [13, 14, 15, 16] при 16 → 12). Злив не
                # є збоєм, хоч би що шард повернув: недочитане ловлять ворота
                # повноти з диска.
                rcs[k] = 0 if state.get("draining") else rc
                print("[htr-case] шард %d/%d rc=%s (%.0f с)%s"
                      % (k, denom, rc, time.time() - t0,
                         " — злито" if state.get("draining") else ""), flush=True)
                if rc in (137, -9) and not state.get("draining"):
                    # SIGKILL, якого ми не посилали, — майже завжди OOM-кілер
                    # ядра: процесорна пам'ять скінчилась, і рядка «out of
                    # memory» у лозі не буде взагалі.
                    fleet["oom_events"] = fleet.get("oom_events", 0) + 1
                    fleet["oom_events_total"] = fleet.get("oom_events_total", 0) + 1
                    fleet["oom_kills"] = fleet.get("oom_kills", 0) + 1
                    print("[htr-case] ☠ шард %d убито SIGKILL — схоже на OOM-кілер "
                          "(RAM контейнера)" % k, flush=True)
                continue

            alive += 1
            silent = now - state["last_out"]
            if silent <= stall_sec:
                continue
            if state.get("draining"):
                # Злитий шард завис на останній сторінці — піднімати його
                # немає сенсу: регулятор уже вирішив, що він зайвий.
                _kill_tree(proc)
                proc.wait(timeout=60)
                state["rc"] = proc.returncode if proc.returncode is not None else -9
                state["alive"] = False
                state["finished_at"] = round(time.time(), 1)
                rcs[k] = 0
                print("[htr-case] ⏱ злитий шард %d мовчав %.0f хв — добиваю"
                      % (k, silent / 60), flush=True)
                continue

            page = state.get("inflight") or "?"
            suspects = fleet.setdefault("suspects", {})
            suspects[page] = suspects.get(page, 0) + 1
            note = ""
            if page != "?" and suspects[page] >= QUARANTINE_HITS:
                _quarantine_add(out_dir, page, "вотчдог: %d хв тиші ×%d"
                                % (stall_sec // 60, suspects[page]))
                if page not in fleet["quarantine"]:
                    fleet["quarantine"].append(page)
                note = "; у карантин"
            print("[htr-case] ⏱ вотчдог: шард %d мовчить %.0f хв на %s — вбиваю%s"
                  % (k, silent / 60, page, note), flush=True)
            _kill_tree(proc)
            proc.wait(timeout=60)

            advanced = state["done"] > state.get("done_at_restart", 0)
            restarts = 0 if advanced else state.get("restarts", 0) + 1
            if restarts > restart_max:
                state["rc"] = proc.returncode if proc.returncode is not None else -9
                state["alive"] = False
                state["finished_at"] = round(time.time(), 1)
                rcs[k] = state["rc"]
                print("[htr-case] ✗ шард %d не дає прогресу %d рестарти поспіль — здаюсь"
                      % (k, restart_max), flush=True)
                continue
            state["restarts"] = restarts
            print("[htr-case] ↻ шард %d піднімаю наново (спроба %d, зроблено %d стор.)"
                  % (k, restarts, state["done"]), flush=True)
            _start_shard(k - 1, denom, base, env, logs_dir, fleet)
            fleet["shards"][k]["restarts"] = restarts

        if regulator is not None:
            # Регулятор ніколи не має права покласти флот: його збій — рядок у
            # лозі, а флот їде далі тим, що є.
            try:
                regulator.tick(now)
            except Exception as exc:
                print("[htr-case] ⚠ регулятор: %r — флот їде як є" % (exc,), flush=True)
        if now - last_publish >= 10:
            _publish(fleet, t0)
            last_publish = now
        if alive == 0 and all(s.get("rc") is not None for s in fleet["shards"].values()):
            break
        time.sleep(WATCH_TICK)

    _publish(fleet, t0)
    return rcs


def _publish(fleet, t0):
    """Зібрати стан флоту у файл, який читає наглядач."""
    shards = []
    done = failed = skipped = 0
    now = time.time()
    for k in sorted(fleet["shards"]):
        s = fleet["shards"][k]
        done += s["done"]
        failed += s["failed"]
        skipped += s["skipped"]
        shards.append({
            "k": k, "pid": s.get("pid"), "alive": s.get("rc") is None, "rc": s.get("rc"),
            "i": s.get("i"), "n": s.get("n"), "done": s["done"], "failed": s["failed"],
            "skipped": s["skipped"], "inflight": s.get("inflight"),
            "silent_sec": round(now - s.get("last_out", now)),
            "restarts": s.get("restarts", 0),
            # коли шард закінчив — метрика хвоста: (max − min)/span по флоту
            "finished_at": s.get("finished_at"),
            "draining": bool(s.get("draining")),
            "vram_peak_mb": s.get("vram_peak_mb"),
            "rss_peak_mb": s.get("rss_peak_mb"),
            "last_error": s.get("last_error"),
        })
    wall = max(1e-6, now - t0)
    expected = fleet.get("n_pages_expected") or 0
    total_done = done + fleet.get("resumed_pages", 0)
    pph = 3600.0 * done / wall if done else 0.0
    left = max(0, expected - total_done)
    fleet_out = {
        "schema": fleet.get("schema", 1),
        "ts": _utc_iso(),
        "case": fleet.get("case"),
        "phase": fleet.get("phase"),
        "n_pages_expected": expected,
        "pages_done": total_done,
        # 🔴 Скільки з них ПІДНЯТО з чекпоінтів, а не прочитано зараз. Без
        # цього числа наглядач не може відрізнити «бокс працює» від «бокс
        # успадкував чужу роботу»: ворота дозрілості темпу рахують сторінки, і
        # на відновленій справі вони виконані з першого ж тіку.
        "pages_resumed": fleet.get("resumed_pages", 0),
        "pages_failed": failed,
        "pages_skipped": skipped,
        "pages_per_hour": round(pph),
        "eta_sec": round(3600.0 * left / pph) if pph and left else (0 if not left else None),
        "oom_events": fleet.get("oom_events", 0),
        "oom_events_total": fleet.get("oom_events_total", 0),
        "regulator": fleet.get("regulator"),
        "quarantine": list(fleet.get("quarantine", [])),
        "threads_per_shard": fleet.get("threads_per_shard"),
        "started": fleet.get("started"),
        "wall_sec": round(wall, 1),
        "shards": shards,
    }
    # 🔴 `case_index` губився саме тут: `_set_phase` клав його у прогрес між
    # справами, а перший же `_publish` від флоту затирав увесь payload — і
    # наглядач до кінця черги писав сторінки п'ятої справи у стан ПЕРШОЇ.
    # Агент бачив «справа 1 застрягла», решта лишалась `pending`.
    for key in ("missing", "missing_count", "complete", "catchup_rounds", "result_url",
                "case_index", "cases_total", "final_case", "ckpt"):
        if key in fleet:
            fleet_out[key] = fleet[key]
    if _QUEUE_RESULTS:
        fleet_out["results"] = list(_QUEUE_RESULTS)
    _write_progress(fleet_out)


#: Скільки VRAM тримає один шард. Типове споживання ~1.8, пік на щільній
#: сторінці — 2.63. 🔴 Планування по 2.0 дало на Царевці 1478 записів
#: «out of memory» і 740 збоїв із 1488 сторінок — тобто більше збоїв, ніж
#: зробленого. Тримаємо 2.5: майже пік, але без надмірної обережності 2.6.
#: Скільки записів «out of memory» уважати не випадковістю, а тіснотою.
#: Один OOM буває на найщільнішій сторінці; кілька означають, що флот просто
#: не влазить у карту.
OOM_SHRINK_TRIGGER = 3
#: Скільки разів звужувати. Дві спроби доводять флот з 8 до 4, далі шукати
#: щастя дорожче, ніж дорахувати догоном.
OOM_SHRINK_MAX = 2

#: 🔴 Дубль сталих `core/htr_sizing.py` — цей файл їде на бокс сам, без пакета.
#: Розбіжність ловить `test_runner_constants_match_the_planner`: до 06.09.2026
#: тут стояло 2.5 ГБ проти 3.3 у ядрі, тобто резервний шлях рахував інший флот.
#: Дефолт невідомого матеріалу; наглядач передає `vram_gb_per_shard` від
#: площі кадру (`gb_per_shard_for`), і тоді ця стала не діє.
GB_PER_SHARD = 3.3
VRAM_HEADROOM = 0.90
#: Шард бере ~одне ядро (std160 05.09.2026: 12 шардів на квоті 11.5 — loadavg
#: 10.2 і 5040 стор/год). Стара «2 ядра/шард = 38.3 с/стор» мірялась без
#: ліміту потоків BLAS і описувала оверсабскрайб, а не апетит шарда.
CORES_PER_SHARD = 1.0
MAX_SHARDS = 32
#: Тека кешу сегментації всередині робочої теки справи. Дублює
#: `htr/plan_build.SEG_CACHE_ARC`: засів із плану лягає саме сюди.
SEG_CACHE_ARC = "data/derived/htr_seg/case"

# ── 🎛 регулятор флоту ────────────────────────────────────────────────────────
# 🔴 Щільність справи (рядків на сторінку, розмір кадру) наперед не вгадується,
# а від неї залежить і пам'ять шарда, і те, скільки шардів карта взагалі тягне.
# План рахував флот із площі кадру: 8591 (10.09.2026) дістав 28 шардів по
# 1.95 ГБ на 2×V100 і дав 4850 стор/год проти обіцяних 6160 — карта насичується
# раніше (заміри 04.09: 8→16 шардів = +0% темпу, ×2 VRAM). Тому флот стартує
# обережно, росте кроками й сам міряє, де коліно: більше шардів, ніж дає темп,
# і більше пам'яті, ніж є, він не бере.
#: Стартова оцінка VRAM на шард, коли наглядач не передав своєї (`vram_gb_per_shard`
#: від площі кадру) — виміряна база важких розворотів (3.2–3.6 ГБ).
REG_START_GB = 3.3
#: Не більше стількох шардів на карту на старті — далі лише за заміром темпу.
REG_START_PER_CARD_MAX = 8
#: Мінімальне вікно заміру темпу після зміни флоту, с (без розгону шардів).
REG_MIN_EPOCH_SEC = 180
#: І щонайменше стільки готових сторінок на шард у вікні.
REG_PAGES_PER_SHARD = 3
#: Крок угору вартий, лише якщо темп виріс хоча б на стільки.
REG_GAIN_MIN = 1.08
#: Більший флот лишається, лише якщо він помітно швидший за менший.
REG_KEEP_BIGGER = 1.02
#: Запас на пік VRAM понад виміряне, і частка RAM контейнера, яку беремо.
REG_VRAM_SAFETY = 1.15
REG_RAM_FRACTION = 0.85
#: Ядер на шард. Заміряно 04.09: 8 шардів — 11.7 ядра, 16 — 17.0 (1.5 → 1.1).
REG_CORES_PER_SHARD = 1.25
#: Карта зайнята понад цю частку — зливаємо шард, не чекаючи OOM.
REG_VRAM_PRESSURE = 0.95
#: Як часто міряти пам'ять (nvidia-smi + /proc), с.
REG_SAMPLE_SEC = 15
#: CUDA-контекст процесу поверх `max_memory_reserved`, МБ.
REG_CTX_MB = 400
#: Знаменник `--shard k/N` у режимі регулятора: номер шарда — лише мітка
#: (клейми роздають сторінки), а номери не перевикористовуються, щоб новий шард
#: не переписав парт мети злитого.
REG_DENOM = 64
#: 🎛 Пам'ять флоту між томами черги переноситься, лише коли кадри схожі за
#: площею: не більше ніж у стільки разів різниці p95 мегапікселів.
FLEET_MEMORY_MPX_RATIO = 1.3


def _free_vram_per_card():
    """Вільна VRAM КОЖНОЇ видимої карти, у ГБ: `[cuda:0, cuda:1, …]`.

    🔴 Раніше тут стояла СУМА по всіх картах, а всі шарди йшли на `cuda:0` —
    тобто бюджет рахувався по всій машині, а працювала одна карта. На боксі з
    2/4/8 картами це давало вдвічі-увосьмеро більше шардів, ніж влазить, і
    гарантований CUDA OOM.

    Тепер список повертається покартково, а шарди РОЗКЛАДАЮТЬСЯ по картах
    (`_start_shard` дає шарду k пристрій `cuda:(k % N)`), тож обидва числа
    знову чесні. Ці дві речі міняються лише разом.

    Вільну, а не загальну: сусід на карті реальний.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=30,
        )
        return [float(v.strip()) / 1024.0 for v in out.split("\n") if v.strip()]
    except Exception:
        return []


def _cgroup_quota_cores(root="/sys/fs/cgroup"):
    """Квота cgroup у ядрах, або ``None`` якщо ліміту немає.

    ``root`` параметром, щоб перевірку можна було написати: на Windows цих
    файлів немає, і без нього тест довів би лише те, що функція не падає.
    """
    root = Path(root)
    try:  # cgroup v2: рядок "<квота> <період>" або "max <період>"
        raw = (root / "cpu.max").read_text().split()
        if raw and raw[0] != "max":
            period = float(raw[1]) if len(raw) > 1 else 100000.0
            if period > 0:
                return float(raw[0]) / period
    except (OSError, ValueError, IndexError):
        pass
    try:  # cgroup v1
        quota = float((root / "cpu" / "cpu.cfs_quota_us").read_text().strip())
        period = float((root / "cpu" / "cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def _usable_cores():
    """Скільки ядер продано НАМ, а не скільки їх у хоста.

    🔴 ``os.cpu_count()`` показує машину цілком, і на орендованому боксі це
    інше число: 04.09.2026 — 96 видимих проти квоти 46.08. Рахувати флот і
    потоки на видимих означає двічі перебрати: більший флот, ніж машина тягне,
    і більше потоків на шард, ніж є ядер. Обидві помилки тихі — з'являються не
    як збій, а як просто повільний прогін.
    """
    cores = float(os.cpu_count() or 1)
    try:
        allowed = len(os.sched_getaffinity(0))
        if allowed:
            cores = min(cores, float(allowed))
    except (AttributeError, OSError):
        pass
    quota = _cgroup_quota_cores()
    if quota and quota > 0:
        cores = min(cores, quota)
    return max(1, int(cores))


def _auto_shards(gb_per_shard=GB_PER_SHARD, cores_per_shard=CORES_PER_SHARD):
    """Скільки шардів тримає ЦЕЙ бокс — з УСІМА його картами.

    Дублює формулу з ``core/htr_sizing.py`` свідомо: ``_embedded`` не імпортує
    gpurunner (код інжектиться в контейнер окремим файлом). Дублювання
    маленьке, а альтернатива — вгадування числа на іншому континенті.

    Кожна карта дає свій запас шардів; шарди потім розкладаються по картах
    круговим порядком, тож 4 карти по 24 ГБ — це вчетверо більше шардів, а не
    вчетверо більший бюджет однієї карти.
    """
    cards = _free_vram_per_card()
    cores = _usable_cores()
    by_vram = sum(int(v * VRAM_HEADROOM // max(0.1, gb_per_shard)) for v in cards)
    by_cpu = int(cores // max(0.1, cores_per_shard))
    shards = max(1, min(by_vram, by_cpu, MAX_SHARDS))
    limiter = "VRAM" if by_vram <= by_cpu else "ядра"
    print("[htr-case] шардів рахую сам: %d · карт %d (%s ГБ вільно) · %d ядер · "
          "обмежувач — %s" % (shards, len(cards),
                              ", ".join("%.1f" % v for v in cards) or "?",
                              cores, limiter), flush=True)
    return shards


def _gpu_mem_per_card():
    """``[(used_mb, total_mb), …]`` по картах; ``[]`` — nvidia-smi недоступний.

    Покарткове ``memory.used``, а не пер-процесне: у контейнері Vast
    ``--query-compute-apps`` віддає pid-и хоста або нічого.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=20)
    except Exception:
        return []
    cards = []
    for row in out.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) != 2:
            continue
        try:
            cards.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return cards


def _group_rss_mb(pgid, proc_root="/proc"):
    """Сума VmRSS усіх процесів групи шарда, МБ.

    Шард — це голова (`--supervise`) і дитина-воркер; обидва в одній групі,
    бо ``_start_shard`` піднімає їх із ``start_new_session``.
    """
    total = 0.0
    try:
        entries = list(Path(proc_root).iterdir())
    except OSError:
        return 0.0
    for d in entries:
        if not d.name.isdigit():
            continue
        try:
            # "pid (comm) state ppid pgrp …" — comm буває з пробілами
            fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != int(pgid):
                continue
            for line in (d / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) / 1024.0
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total


def _ram_limit_mb(root="/sys/fs/cgroup", meminfo="/proc/meminfo"):
    """Скільки RAM продано НАМ (cgroup), МБ; інакше вся пам'ять машини; None — невідомо.

    🔴 ``free`` показує хост, а не нашу частку (див. `_usable_cores` про ядра).
    """
    root = Path(root)
    for f in (root / "memory.max", root / "memory" / "memory.limit_in_bytes"):
        try:
            raw = f.read_text().strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        if 0 < v < 2 ** 60:        # v1 «без ліміту» — 9223372036854771712
            return v / 2 ** 20
    try:
        for line in Path(meminfo).read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _drain_dir(out_dir):
    return Path(out_dir) / "_drain"


def _request_drain(out_dir, k):
    """Попросити шард k (1-based) дочитати сторінку й вийти — див. раннер."""
    d = _drain_dir(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / str(int(k))).write_text(str(time.time()), encoding="utf-8")


def _clear_drains(out_dir, k=None):
    d = _drain_dir(out_dir) if out_dir else None
    if d is None or not d.is_dir():
        return 0
    n = 0
    for f in d.iterdir():
        if k is not None and f.name != str(int(k)):
            continue
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n


#: Нижні межі заспокоєння й вікна заміру для ШВИДКИХ сторінок, с.
REG_FAST_SETTLE_SEC = 20.0
REG_FAST_EPOCH_SEC = 45.0


def _epoch_timing(med):
    """(заспокоєння, вікно заміру) під тривалість сторінки `med`, с.

    🔴 Скриба на готовій сегментації (15.09.2026) читав том у 400 сторінок за
    ~70 с, а регулятор дозрівав 60 + 180 с — флот не робив жодного кроку, і
    стеля в 14 шардів лишалась стартом на всю чергу. Межі масштабуються з
    тривалістю сторінки: від ~12 с і повільніше вони ті самі 60 і 180 с, на
    яких знайдено коліна; швидше — до 20 і 45 с. Невідома тривалість — старі.
    """
    if not med:
        return 60.0, float(REG_MIN_EPOCH_SEC)
    settle = max(1.5 * med, min(60.0, max(REG_FAST_SETTLE_SEC, 5.0 * med)))
    epoch = max(2.5 * med, min(float(REG_MIN_EPOCH_SEC), max(REG_FAST_EPOCH_SEC, 15.0 * med)))
    return settle, epoch


def _regulate_decision(s):
    """Скільки шардів тримати далі — і чому. Чиста функція, без жодного I/O.

    ``s``: n, n_gpus, ceiling, knee, oom_new, card_total_mb, card_used_mb,
    epoch_rate (None — вікно заміру не дозріло), rates {n: стор/год},
    pages_left, vram_per_shard_mb, rss_per_shard_mb, ram_limit_mb, cores.
    Повертає ``(target, reason | None, knee)``; ``knee`` не None — вище не ростемо.
    """
    n = int(s["n"])
    g = max(1, int(s.get("n_gpus") or 1))
    up, down = 2 * g, g
    floor = max(1, min(g, n))
    ceiling = max(1, int(s.get("ceiling") or n))
    knee = s.get("knee")

    oom = int(s.get("oom_new") or 0)
    if oom and n > floor:
        t = max(floor, n - down)
        return t, "OOM ×%d → %d → %d шардів, вище не ростемо" % (oom, n, t), t
    total = float(s.get("card_total_mb") or 0)
    used = float(s.get("card_used_mb") or 0)
    if total and used > REG_VRAM_PRESSURE * total and n > floor:
        t = max(floor, n - down)
        return t, ("карта зайнята на %.0f%% → %d → %d шардів, вище не ростемо"
                   % (100.0 * used / total, n, t)), t

    rate = s.get("epoch_rate")
    if rate is None:
        return n, None, knee
    rates = dict(s.get("rates") or {})
    lower = sorted(m for m in rates if m < n)
    if lower and knee is None:
        p = lower[-1]
        if rate < rates[p] * REG_GAIN_MIN:
            t = n if rate > rates[p] * REG_KEEP_BIGGER else p
            return t, ("коліно: %d шардів — %.0f стор/год, %d — %.0f (приріст < %d%%) → "
                       "тримаю %d" % (n, rate, p, rates[p],
                                      round(100 * (REG_GAIN_MIN - 1)), t)), t
    if knee is not None and n >= knee:
        return n, None, knee

    nxt = min(ceiling, n + up)
    if nxt <= n:
        return n, "стеля %d шардів" % ceiling, knee
    if int(s.get("pages_left") or 0) < 3 * nxt:
        return n, "хвіст справи — рости вже не окупиться", knee
    if not total:
        return n, "VRAM карти невідома — наосліп не ростемо", knee
    per_card = -(-nxt // g)
    card_per_shard = float(s.get("card_per_shard_mb") or 0)
    if card_per_shard:
        # 🔴 Прогноз — із ЗАЙНЯТОСТІ КАРТИ, а не з суми піків шардів: піки різних
        # шардів у часі не збігаються. 316-1-39 (10.09.2026): пік шарда 4.0 ГБ ×
        # 9 на карту «не влазив» у 32 ГБ, а карта з 8 шардами була зайнята
        # максимум на 11.3 ГБ. Запас — один пік понад середнє (найгірший шард).
        spike = max(0.0, float(s.get("peak_shard_mb") or 0) - card_per_shard)
        projected = (card_per_shard * per_card + spike) * REG_VRAM_SAFETY
        if projected > total:
            return n, ("межа VRAM: %d на карту × %.0f МБ (виміряно на карті) + пік "
                       "%.0f → %.0f > %.0f МБ"
                       % (per_card, card_per_shard, spike, projected, total)), knee
    else:
        need = float(s.get("vram_per_shard_mb") or REG_START_GB * 1024)
        if per_card * need * REG_VRAM_SAFETY > total:
            return n, ("межа VRAM: %d на карту × %.0f МБ × %.2f > %.0f МБ"
                       % (per_card, need, REG_VRAM_SAFETY, total)), knee
    ram = s.get("ram_limit_mb")
    rss = float(s.get("rss_per_shard_mb") or 0)
    if ram and rss and nxt * rss > REG_RAM_FRACTION * float(ram):
        return n, ("межа RAM: %d × %.0f МБ > %.0f%% з %.0f МБ"
                   % (nxt, rss, 100 * REG_RAM_FRACTION, float(ram))), knee
    cores = float(s.get("cores") or 0)
    per_shard = float(s.get("cores_per_shard") or REG_CORES_PER_SHARD)
    if cores and nxt * per_shard > cores:
        return n, ("межа ядер: %d × %.2f > %.1f" % (nxt, per_shard, cores)), knee
    return nxt, ("темп %.0f стор/год на %d шардах, пам'яті вистачає → пробую %d"
                 % (rate, n, nxt)), knee


class _FleetRegulator:
    """Флот сам знаходить коліно: скільки шардів дає найбільший темп на ЦІЙ
    справі й на ЦЬОМУ залізі — у межах ВИМІРЯНОЇ пам'яті.

    Росте кроками по два шарди на карту (клейми роздають сторінки новим без
    перезапуску флоту), звужується зливом (`_request_drain`): шард дочитує
    сторінку й виходить із rc=0, тож нічого не губиться і не рахується збоєм.
    """

    def __init__(self, fleet, *, out_dir, base, env, logs_dir, denom, ceiling, cores, t0,
                 memory=None, cores_per_shard=None):
        self.fleet = fleet
        self.out_dir = Path(out_dir)
        self.base, self.env, self.logs_dir = base, env, logs_dir
        self.denom = int(denom)
        self.ceiling = int(ceiling)
        self.cores = float(cores)
        self.cores_per_shard = float(cores_per_shard or REG_CORES_PER_SHARD)
        self.t0 = t0
        self.n_gpus = max(1, int(fleet.get("n_gpus") or 1))
        self.rates = {}
        self.knee = None
        self.t_change = time.time()
        self.window = None
        self.oom_seen = int(fleet.get("oom_events_total") or 0)
        self.last_sample = 0.0
        self.card_total = 0.0
        self.card_used = 0.0
        self.card_used_max = 0.0
        self.vram_per_shard = 0.0
        #: Зайнятість карти на шард (nvidia-smi / живі шарди на карті), максимум.
        self.card_per_shard = 0.0
        #: Пік одного шарда (подія `htr` раннера + контекст CUDA), максимум.
        self.peak_shard = 0.0
        self.rss_per_shard = 0.0
        self.ram_limit = _ram_limit_mb()
        self.note = None
        self.history = []
        #: Найбільший флот, який тримали без OOM і без коліна, — з нього стартує
        #: наступний том черги.
        self.n_stable = len(self.active())
        #: Що прийшло з попереднього тому (None — перший том або інший матеріал).
        self.carried = None
        if memory:
            # Темп (`rates`) НЕ переноситься: абсолютне число сторінок на годину
            # різниться від тому до тому (1856–5395 у черзі spr-2461), тож коліно
            # на новому томі шукається від перенесеного розміру заново.
            self.knee = memory.get("knee")
            self.card_per_shard = float(memory.get("card_per_shard") or 0)
            self.peak_shard = float(memory.get("peak_shard") or 0)
            self.rss_per_shard = float(memory.get("rss_per_shard") or 0)
            self.vram_per_shard = max(self.card_per_shard, self.peak_shard)
            self.carried = {"n_stable": memory.get("n_stable"), "knee": self.knee}
        fleet["regulator"] = self.view()

    # ---- що є ----

    def _card_of(self, k):
        return (int(k) - 1) % self.n_gpus

    def _live(self):
        return [k for k, s in self.fleet["shards"].items() if s.get("rc") is None]

    def active(self):
        return sorted(k for k, s in self.fleet["shards"].items()
                      if s.get("rc") is None and not s.get("draining"))

    def _done_total(self):
        return sum(int(s.get("done") or 0) for s in self.fleet["shards"].values())

    def _page_sec_median(self):
        secs = sorted(x for s in self.fleet["shards"].values() for x in (s.get("secs") or []))
        return secs[len(secs) // 2] if secs else 0.0

    def sample(self, now):
        """Пам'ять: карта (nvidia-smi), процеси (/proc), пік із подій раннера."""
        if now - self.last_sample < REG_SAMPLE_SEC:
            return
        self.last_sample = now
        live = self._live()
        cards = _gpu_mem_per_card()
        if cards:
            self.card_total = min(t for _, t in cards)
            self.card_used = max(u for u, _ in cards)
            self.card_used_max = max(self.card_used_max, self.card_used)
            per_card = {}
            for k in live:
                c = self._card_of(k)
                per_card[c] = per_card.get(c, 0) + 1
            for c, (used, _total) in enumerate(cards):
                if per_card.get(c):
                    # алокатор торча пам'ять не віддає й росте від найбільшого
                    # кадру, тож беремо максимум за весь захід
                    self.card_per_shard = max(self.card_per_shard, used / per_card[c])
        peaks = [int(s.get("vram_peak_mb") or 0) for s in self.fleet["shards"].values()]
        if peaks and max(peaks):
            self.peak_shard = max(self.peak_shard, max(peaks) + REG_CTX_MB)
        self.vram_per_shard = max(self.card_per_shard, self.peak_shard)
        rss = [_group_rss_mb(self.fleet["shards"][k].get("pid") or 0) for k in live]
        rss = [r for r in rss if r > 0]
        if rss:
            self.rss_per_shard = max(self.rss_per_shard, max(rss))

    # ---- рішення ----

    def tick(self, now):
        self.sample(now)
        act = self.active()
        n = len(act)
        if not n:
            return
        # 🔴 Поки злитий шард дочитує сторінку, він тримає пам'ять і може ще раз
        # упасти в OOM: рішення за тими самими ознаками злили б каскадом
        # пів флоту. Чекаємо, доки злиті вийдуть.
        if any(s.get("draining") and s.get("rc") is None
               for s in self.fleet["shards"].values()):
            return
        oom_total = int(self.fleet.get("oom_events_total") or 0)
        oom_new = oom_total - self.oom_seen
        self.oom_seen = oom_total
        med = self._page_sec_median()
        epoch_rate = None
        if self.window is not None and self.window[2] != n:
            # 🔴 Шард вийшов сам (кінець його частки, хвіст справи): темп цього
            # вікна вже не про n шардів. Так у spr-2461 з'явилось
            # `rates {"1": 4008}` — сторінки чотирнадцяти шардів, записані одному.
            self.window = None
            self.t_change = now
        settle, epoch = _epoch_timing(med)
        if self.window is None:
            if now - self.t_change >= settle:
                self.window = (now, self._done_total(), n)
        else:
            t_w, d_w, _ = self.window
            span, pages = now - t_w, self._done_total() - d_w
            if span >= epoch and pages >= REG_PAGES_PER_SHARD * n:
                epoch_rate = 3600.0 * pages / span
        expected = int(self.fleet.get("n_pages_expected") or 0)
        left = max(0, expected - self._done_total() - int(self.fleet.get("resumed_pages") or 0))
        target, why, knee = _regulate_decision({
            "n": n, "n_gpus": self.n_gpus, "ceiling": self.ceiling, "knee": self.knee,
            "oom_new": oom_new, "card_total_mb": self.card_total,
            "card_used_mb": self.card_used, "epoch_rate": epoch_rate, "rates": self.rates,
            "pages_left": left, "vram_per_shard_mb": self.vram_per_shard,
            "card_per_shard_mb": self.card_per_shard, "peak_shard_mb": self.peak_shard,
            "rss_per_shard_mb": self.rss_per_shard, "ram_limit_mb": self.ram_limit,
            "cores": self.cores, "cores_per_shard": self.cores_per_shard,
        })
        if epoch_rate is not None:
            self.rates[n] = round(epoch_rate)
            self.window = None
            self.t_change = now
        if knee is not None and knee != self.knee:
            self.n_stable = int(knee)
        self.knee = knee
        if target != n:
            self.apply(target, act, now)
            self.t_change = now
            self.window = None
        if why and why != self.note:
            print("[htr-case] 🎛 регулятор: %s" % why, flush=True)
            self.note = why
        self.fleet["regulator"] = self.view()

    def apply(self, target, act, now):
        n = len(act)
        if target > n:
            for _ in range(target - n):
                k = self._free_slot()
                if k is None:
                    break
                _start_shard(k - 1, self.denom, self.base, self.env, self.logs_dir, self.fleet)
        else:
            for k in self._victims(n - target, act):
                _request_drain(self.out_dir, k)
                self.fleet["shards"][k]["draining"] = True
        got = len(self.active())
        if got > n and self.knee is None:
            self.n_stable = max(self.n_stable, got)
        print("[htr-case] 🎛 флот %d → %d шардів" % (n, got), flush=True)
        self.history.append({"t": round(now - self.t0), "from": n, "to": got})

    def _free_slot(self):
        """Номер для нового шарда на НАЙМЕНШ зайнятій карті; номери не повторюються."""
        loads = {c: 0 for c in range(self.n_gpus)}
        for k in self._live():
            loads[self._card_of(k)] += 1
        card = min(loads, key=lambda c: (loads[c], c))
        for k in range(1, self.denom + 1):
            if k not in self.fleet["shards"] and self._card_of(k) == card:
                return k
        return None

    def _victims(self, count, act):
        """Кого зливати: найновіші шарди на найзавантаженіших картах."""
        by_card = {}
        for k in act:
            by_card.setdefault(self._card_of(k), []).append(k)
        out = []
        for _ in range(count):
            card = max(by_card, key=lambda c: (len(by_card[c]), c))
            if not by_card[card]:
                break
            k = max(by_card[card])
            by_card[card].remove(k)
            out.append(k)
        return out

    def remember(self):
        """Що передати наступному тому черги (див. `_carried_fleet`)."""
        return {
            "n_stable": int(self.n_stable or len(self.active()) or 1),
            "knee": self.knee,
            "card_per_shard": round(self.card_per_shard),
            "peak_shard": round(self.peak_shard),
            "rss_per_shard": round(self.rss_per_shard),
        }

    def view(self):
        return {
            "active": len(self.active()), "ceiling": self.ceiling, "knee": self.knee,
            "n_stable": getattr(self, "n_stable", None),
            "carried": getattr(self, "carried", None),
            "rates": {str(k): v for k, v in sorted(self.rates.items())},
            "vram_per_shard_mb": round(self.vram_per_shard),
            "card_per_shard_mb": round(self.card_per_shard),
            "peak_shard_mb": round(self.peak_shard),
            "card_total_mb": round(self.card_total),
            "card_used_max_mb": round(self.card_used_max),
            "rss_per_shard_mb": round(self.rss_per_shard),
            "ram_limit_mb": round(self.ram_limit) if self.ram_limit else None,
            "note": self.note,
            "history": self.history[-12:],
        }


def _quarantined_stems(out_dir):
    path = Path(out_dir) / "_htr_quarantine.json"
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {Path(name).stem for name in (data.get("pages") or {})}
    except Exception:
        return set()


def _missing_pages(out_dir, expected_stems):
    """Кадри, для яких немає тексту. Знаменник — з ДИСКА, не з мети.

    Мета — це самозвіт конвеєра, а ми саме його й перевіряємо. Карантин
    віднімається: ці сторінки визнані нечитаними свідомо й задокументовано.
    """
    produced = {p.stem for p in Path(out_dir).glob("*.txt")}
    skip = _quarantined_stems(out_dir)
    return [s for s in expected_stems if s not in produced and s not in skip]


def _runner_supports(script, flag):
    """Чи знає раннер прапорець — за текстом файла, без запуску."""
    try:
        return ('"%s"' % flag) in Path(script).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _clear_claims(out_dir):
    """Зняти всі клейми в теці виводу. Тексти й мету не чіпає."""
    if not out_dir:
        return 0
    d = Path(out_dir) / "_claims"
    if not d.is_dir():
        return 0
    n = 0
    for c in d.glob("*.claim"):
        try:
            c.unlink()
            n += 1
        except OSError:
            pass
    return n


def _catchup_pass(base, indices, env, logs_dir, round_no):
    """Перечитати пропущені сторінки ОДНИМ процесом.

    🔴 Без ``--shard`` не лише заради VRAM. ``htr_case_run.py`` іменує файл
    мети за номером шарда (``_htr_meta.part1..N``), тож догін на шардах
    ПЕРЕЗАПИСАВ би мету основного прогону — і знаменник поїхав би тихо.
    Без ``--shard`` він пише в ``_htr_meta.json``, який ``merge_meta`` бере
    базовим шаром: колізії немає за побудовою.
    """
    cmd = [a for a in base if a != "--progress-json"]
    cmd += ["--pages", ",".join(str(i) for i in indices)]
    # Догін іде без `--shard`, отже й без `--claim`: клеймів він не читає, а
    # сироти флоту йому не заважають за побудовою. Чистимо лише для порядку.
    out_dir = next((base[i + 1] for i, a in enumerate(base[:-1]) if a == "--out-dir"), None)
    _clear_claims(Path(out_dir) if out_dir else None)
    log_path = logs_dir / ("catchup%02d.log" % round_no)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd="/tmp/htrcase", start_new_session=True)
        return proc.wait()


#: Останнє опубліковане тіло прогресу — його освіжає серцебиття.
_LAST_PROGRESS = {}
#: Підсумки закритих томів черги: `{index, case, complete, n_pages_txt, attempt, error?}`.
#: Їде в кожну публікацію прогресу, щоб наглядач бачив провал тому одразу, а не
#: на заборі (irnbuv1 q23, 14.09.2026: 8 томів із 23 впали мовчки).
_QUEUE_RESULTS = []
_HEARTBEAT_STOP = threading.Event()


def _heartbeat(period=10):
    """Оновлювати мітку часу в `_progress.json`, поки живий процес.

    🔴 Без цього між справами черги настає повна тиша: шарди першої вже
    завершились, кадри другої ще качаються, мету зводимо — і ніхто не пише
    прогрес. Наглядач бачить «20 хвилин без оновлень при живому SSH»,
    оголошує раннер мертвим і **переорендовує бокс посеред успішної
    роботи** (виміряно 2026-08-11: перша справа зроблена повністю, 10 шардів
    rc=0, і саме тоді захід перезапустився).

    Серцебиття належить ПРОЦЕСУ, а не циклу спостереження за шардами: воно
    мусить битись і тоді, коли шардів немає взагалі.
    """
    while not _HEARTBEAT_STOP.wait(period):
        if not _LAST_PROGRESS:
            continue
        try:
            payload = dict(_LAST_PROGRESS)
            payload["ts"] = _utc_iso()
            # 🔴 Серцебиття доводить, що ЖИВИЙ ПРОЦЕС, а не що йде РОБОТА.
            # Освіжаючи мітку часу, воно вимикало обидва детектори застрягання
            # у наглядача: завислий `curl` без `--max-time` або pip у мертве
            # дзеркало тарифікувались до самої стелі годин, бо «прогрес
            # свіжий». Тому окремо публікуємо мітку РОБОТИ: наглядач міряє вік
            # саме її, а `ts` лишає для «чи живий процес».
            payload["heartbeat"] = True
            payload["work_ts"] = _LAST_WORK_TS[0] or payload.get("ts")
            _write_progress(payload, remember=False)
        except Exception:
            pass


#: Коли востаннє публікувався РЕАЛЬНИЙ поступ (не серцебиття).
#: Порожній до першої публікації — щоб не залежати від порядку склейки
#: раннера з `_common.py` на боксі.
_LAST_WORK_TS = [""]


def _set_phase(phase, **extra):
    """Позначити фазу поза шардами (качання, злиття, наступна справа)."""
    payload = dict(_LAST_PROGRESS)
    payload.update(extra)
    payload["phase"] = phase
    payload["ts"] = _utc_iso()
    _write_progress(payload)


def _write_progress(state, remember=True):
    """Атомарно опублікувати стан. Це і є те, що читає наглядач."""
    if remember:
        # Реальна публікація (не серцебиття) = робота зрушила.
        _LAST_WORK_TS[0] = _utc_iso()
        state.setdefault("work_ts", _LAST_WORK_TS[0])
        _LAST_PROGRESS.clear()
        _LAST_PROGRESS.update(state)
    try:
        PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PROGRESS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, PROGRESS_PATH)
    except Exception:
        pass  # прогрес — не привід валити прогін


def _find_model(root, name):
    """Ваги за іменем файла; шлях приймається як є, якщо він абсолютний."""
    cand = Path(name)
    if cand.is_absolute() and cand.is_file():
        return cand
    hits = [p for p in root.rglob(cand.name) if p.is_file()]
    if not hits:
        raise RuntimeError("ваг %r немає під %s" % (cand.name, root))
    return hits[0]


def _gpu_name():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "?"


def _versions():
    out = {}
    for mod in ("torch", "kraken", "skimage", "shapely", "PIL"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "?"))
        except Exception as exc:
            out[mod] = "ERR %s" % type(exc).__name__
    return out


#: Що НЕ їде у проміжний чекпоінт.
#:
#: 🔴 Тут був `.lines.json` — «важать сотні КБ і потрібні лише в кінці». Це
#: помилка: після смерті боксу нова оренда бачить сторінку в меті як зроблену
#: й пропускає її, тож рамки для вже зробленої половини справи не породжуються
#: НІКОЛИ. Ворота повноти рахують лише `*.txt` і кажуть «повно», а на ділі для
#: половини справи не працює підсвітка рядка у в'ювері й немає з чого різати
#: кропи для доказ-файлів. Замір: 17 КБ на кадр (не сотні), тобто на справі в
#: 3400 кадрів це ~58 МБ — дешевше за повторний прохід.
CKPT_SKIP_SUFFIXES = ()


def _new_files(src_root, seen):
    """Нові/дорослі файли під src_root. Ключ — розмір, не mtime.

    `_htr_meta.part*.json` переписується цілком і росте, тож розміру досить, щоб
    не ганяти те саме двічі.
    """
    out = []
    if not src_root.is_dir():
        return out
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root).as_posix()
        if rel.endswith(CKPT_SKIP_SUFFIXES) or Path(rel).name.startswith("ckpt_"):
            continue
        # 🧲 Клейми — стан ЦЬОГО флоту (pid-и цієї машини); у чекпоінт їм не
        # можна: на іншому боксі перевірка життя власника збрехала б.
        if "/_claims/" in "/" + rel:
            continue
        try:
            size = src.stat().st_size
        except OSError:
            continue
        if seen.get(rel) == size:
            continue
        out.append((src, rel, size))
    return out


def _mirror(src_root, dst_root, seen):
    """Скопіювати нові файли src→dst ПОФАЙЛОВО. Повертає, скільки скопійовано.

    ‼ Лише для фіналу. На мережевому томі один дрібний файл коштує ~1 секунду
    (замір Beam 2026-08-09: +60 файлів за 62 с), тож проміжний чекпоінт так
    робити НЕ МОЖНА — він з'їдає рівно той час, який мав би рятувати.
    """
    n = 0
    for src, rel, size in _new_files(src_root, seen):
        dst = dst_root / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            seen[rel] = size
            n += 1
        except OSError as exc:
            print("[htr-case] ⚠ чекпоінт %s: %s" % (rel, exc), flush=True)
    return n


def _mirror_archive(src_root, dst_root, seen, index):
    """Інкрементальний чекпоінт ОДНИМ архівом: 1 запис на том замість N.

    Архіви накопичуються (`ckpt_0001.tgz`, `ckpt_0002.tgz`, …) і разом дають повний
    стан: наступний запуск розпаковує їх по порядку, пізніший перекриває раніший.
    """
    fresh = _new_files(src_root, seen)
    if not fresh:
        return 0
    dst_root.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/htrcase/ckpt_%04d.tgz" % index)
    with tarfile.open(tmp, "w:gz", compresslevel=1) as tf:
        for src, rel, _ in fresh:
            tf.add(src, arcname=rel)
    shutil.copy2(tmp, dst_root / tmp.name)
    tmp.unlink(missing_ok=True)
    for _, rel, size in fresh:
        seen[rel] = size
    return len(fresh)


#: Що конвеєр вимагає в контейнері. На Beam/Modal це запікається в образ, а на
#: Vast орендується гола машина з pytorch-образом — там ставить сам раннер.
PIP_DEPS = (
    "kraken==7.0.2",          # пін із KRAKEN_PATCHES.md: патчі б'ють приватні функції
    "timm>=0.9",
    "pytorch-lightning>=2.0",
    "nltk",
    "git+https://github.com/baudm/parseq.git",   # strhub — код PARSeq під ваги Писаря
)


def _ensure_deps():
    """Поставити те, чого бракує — ПО ОДНОМУ пакету за раз.

    ‼ Одним списком не можна: ``pip install kraken==7.0.2 timm nltk
    pytorch-lightning>=2.0 git+…/parseq`` падає резолвером, бо kraken тягне свій
    ``lightning``, і вимога поруч конфліктує. Кожен пакет ОКРЕМО ставиться чисто
    (перевірено на боксі 2026-08-09: kraken rc=0, timm rc=0, strhub rc=0), а
    разом — rc≠0. Ціна помилки: оренда, доведена до самого прогону, і падіння
    за секунду до роботи.

    ‼ І БЕЗ ``-q``: перший раз причина падіння була невідома саме тому, що вивід
    приглушили. Лог кожного пакета лягає у /tmp, хвіст друкується при помилці.
    """
    plan = [("kraken", PIP_DEPS[0]), ("timm", PIP_DEPS[1]),
            ("pytorch_lightning", PIP_DEPS[2]), ("nltk", PIP_DEPS[3]),
            ("strhub", PIP_DEPS[4])]
    installed = []
    for i, (mod, spec) in enumerate(plan):
        try:
            __import__(mod)
            continue
        except Exception:
            pass
        log = "/tmp/pip_%02d.log" % i
        print("[htr-case] pip install %s …" % spec, flush=True)
        t = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.call([sys.executable, "-m", "pip", "install", spec],
                                 stdout=fh, stderr=subprocess.STDOUT)
        print("[htr-case] %s: rc=%s за %.0f с" % (spec, rc, time.time() - t), flush=True)
        if rc != 0:
            try:
                tail = Path(log).read_text(encoding="utf-8",
                                           errors="replace")[-2000:]
            except OSError:
                tail = "(лог недоступний)"
            print("[htr-case] --- хвіст %s ---\n%s" % (log, tail), flush=True)
            raise RuntimeError("pip install %s провалився (rc=%s)" % (spec, rc))
        installed.append(spec)
    return installed


def _ckpt_to_url(src_root, seen, index, urls):
    """Інкрементальний чекпоінт у хмарне сховище через presigned PUT.

    Навіщо не ключами: бокс орендований і чужий, а наявні S3-ключі бачать ще й
    бакет із бекапами. Presigned PUT дає право записати РІВНО один заздалегідь
    названий об'єкт — ні прочитати, ні перелічити, ні торкнутись чогось іншого.

    Повертає (скільки файлів, скільки секунд) або (0, 0), якщо нема чого лити.
    """
    fresh = _new_files(src_root, seen)
    if not fresh:
        return 0, 0.0
    if index > len(urls):
        print("[htr-case] ⚠ presigned PUT-посилання скінчились (%d) — чекпоінт "
              "пропущено; наступні прогони брати з більшим --count" % len(urls), flush=True)
        return 0, 0.0
    tmp = Path("/tmp/htrcase/ckpt_%04d.tgz" % index)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tmp, "w:gz", compresslevel=1) as tf:
        for src, rel, _ in fresh:
            tf.add(src, arcname=rel)
    t = time.time()
    rc = subprocess.call(["curl", "-fsS", "--retry", "2", "-X", "PUT",
                          "--upload-file", str(tmp), urls[index - 1]])
    dt = time.time() - t
    size = tmp.stat().st_size
    tmp.unlink(missing_ok=True)
    if rc != 0:
        print("[htr-case] ⚠ чекпоінт №%d не залився (curl rc=%s) — файли лишаються "
              "в позначених, спробуємо наступним раундом" % (index, rc), flush=True)
        return 0, dt
    for _, rel, sz in fresh:
        seen[rel] = sz
    print("[htr-case] чекпоінт №%d → хмара: %d файл(ів), %.1f МБ за %.1f с"
          % (index, len(fresh), size / 1e6, dt), flush=True)
    return len(fresh), dt


def main(params):
    """Точка входу. Одна справа — або ЧЕРГА справ на одному боксі.

    🔴 Черга тут не оптимізація, а виправлення дурості. Холодний старт коштує
    ~8 хвилин (pip по одному пакету, 105 МБ моделей, завантаження ваг), і на
    семи дрібних справах це майже година оренди, за яку не прочитано жодної
    сторінки. Обходити це сімома паралельними орендами — платити всемеро за
    ту саму ваду. Тому бокс піднімається ОДИН раз, а справи йдуть чергою:
    залежності, моделі й скрипти вже на місці, і кожна наступна коштує лише
    завантаження власних кадрів.
    """
    beat = threading.Thread(target=_heartbeat, daemon=True)
    beat.start()
    try:
        return _main_inner(params)
    finally:
        _HEARTBEAT_STOP.set()


#: Куди наглядач дописує справи ЖИВОМУ боксу. Формат — JSONL, по справі на
#: рядок, і саме тому JSONL: дописування одного короткого рядка з `O_APPEND`
#: атомарне, тож читач ніколи не побачить половини запису. Спільний JSON-масив
#: довелося б перечитувати-переписувати, і гонка з'явилась би на рівному місці.
APPEND_FILE = "/tmp/htrcase/_append.jsonl"
#: Скільки 404 поспіль у серії resume-посилань означають кінець серії.
#: Три, а не один — запас на транзієнтний 4xx від сховища.
RESUME_MISSES_TO_STOP = 3
#: Як часто теплий бокс перечитує довісок у фазі `idle`, с.
IDLE_POLL_SEC = 10


def _read_appended(seen):
    """Справи, дописані ззовні, поки бокс уже працює.

    🔴 Ідемпотентність за ІМЕНЕМ справи: повторний довісок того самого запису
    (наглядач перезапустився, людина натиснула двічі) не сміє перечитати справу
    вдруге — це прямі гроші за вже зроблене.

    ⚠ Битий рядок пропускається з голосним рядком у лозі, а не валить чергу:
    решта справ уже оплачена, і губити її через одну зіпсовану кому безглуздо.
    """
    path = Path(APPEND_FILE)
    if not path.is_file():
        return []
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print("[htr-case] ⚠ довісок не прочитався: %r" % (exc,), flush=True)
        return []

    fresh = []
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            case = json.loads(raw)
        except ValueError:
            print("[htr-case] ⚠ довісок: рядок не JSON, пропускаю: %.80s" % raw, flush=True)
            continue
        if not isinstance(case, dict) or not str(case.get("case") or "").strip():
            print("[htr-case] ⚠ довісок: запис без імені справи, пропускаю", flush=True)
            continue
        key = str(case["case"]).strip()
        if key in seen:
            continue
        seen.add(key)
        fresh.append(case)
    if fresh:
        print("[htr-case] 🧊 довісок: +%d справ(и) — %s" % (
            len(fresh), ", ".join(str(c.get("case")) for c in fresh)), flush=True)
    return fresh


#: Підлога швидкості качання: нижче за 1 МБ/с протягом хвилини — обрив і
#: повтор. 🔴 Було 10 КБ/с за 120 с — це не підлога, а її відсутність: архів
#: кадрів на 400 МБ при 20 КБ/с проходив перевірку і качався б годинами.
#: 06.09.2026 (P2, ф.230) друга справа черги простояла на качанні годину.
DOWNLOAD_MIN_BPS = 1048576
#: 🔴 Підлога ПОВТОРНИХ спроб. irnbuv1 q23 (14.09.2026, бокс у Таїланді): маршрут
#: до R2 просідав нижче 1 МБ/с, обидві спроби обривались на `rc=28`, і 8 томів
#: із 23 впали, не прочитавши жодного кадру. Архіви при цьому були 60–130 МБ — на
#: 0.3 МБ/с це кілька хвилин, а сусідні томи тієї ж хвилини качались на 2–9 МБ/с.
#: Перша спроба лишає строгу підлогу (ловить завислий канал), наступні — м'якшу.
DOWNLOAD_RETRY_MIN_BPS = 262144
#: Абсолютна підлога — канал мертвий (~1 Мбіт/с, як `gate.DEAD_NET_MBPS`).
DOWNLOAD_DEAD_BPS = 131072
#: Частка вже виміряної швидкості качання, нижче якої спроба вважається завислою.
#:
#: 🔴 Стала 1 МіБ/с на першій спробі обривала кожен том на боксі з каналом 6
#: Мбіт/с (0.76 МБ/с): `--retry 3` × 60 с — до 4 хв на том, і лише потім м'якша
#: підлога. Підлога від ВИМІРЯНОГО тим самим маршрутом ловить зависання, не
#: караючи повільний, але живий канал.
DOWNLOAD_FLOOR_SHARE = 0.3
DOWNLOAD_TRIES = 3
#: Швидкості успішних качань цього боксу, Б/с.
_DOWNLOAD_SEEN_BPS = []


def _download_floor(attempt):
    """Підлога швидкості для спроби: від виміряного каналу, у межах [мертвий, 1 МіБ/с]."""
    seen = sorted(_DOWNLOAD_SEEN_BPS)
    base = seen[len(seen) // 2] * DOWNLOAD_FLOOR_SHARE if seen else DOWNLOAD_RETRY_MIN_BPS
    if attempt > 1:
        base /= 2
    return int(max(DOWNLOAD_DEAD_BPS, min(DOWNLOAD_MIN_BPS, base)))
DOWNLOAD_RETRY_SLEEP_SEC = 10
DOWNLOAD_HEARTBEAT_SEC = 15


def _download_with_heartbeat(url, local, name, tries=DOWNLOAD_TRIES, heartbeat=True,
                             min_free_bytes=0):
    """curl із серцебиттям: доки качається, прогрес публікує фазу
    `downloading` з розміром файла, і наглядач бачить рух, а не тишу.
    `--max-time` лишається стелею на випадок, коли й серцебиття бреше.
    Повтори — з нуля, з м'якшою підлогою швидкості (`DOWNLOAD_RETRY_MIN_BPS`);
    rc curl повертається як є.

    ``heartbeat=False`` — фонове передзавантаження: фазу публікує той, хто
    зараз працює (флот попереднього тому), і перебивати її не можна.
    ``min_free_bytes`` — обрив, коли диск спускається нижче: передзавантаження
    не сміє забрати місце в тому, що читається зараз. Тоді rc=-2 без повтору.
    """
    rc = 1
    for attempt in range(1, tries + 1):
        floor = str(_download_floor(attempt))
        t_attempt = time.time()
        proc = subprocess.Popen(["curl", "-fsSL", "--retry", "3", "--max-time", "7200",
                                 "--speed-limit", floor, "--speed-time", "60",
                                 "-o", str(local), url])
        last = -1
        while True:
            try:
                rc = proc.wait(timeout=DOWNLOAD_HEARTBEAT_SEC)
                break
            except subprocess.TimeoutExpired:
                size = local.stat().st_size if local.exists() else 0
                if heartbeat:
                    _set_phase("downloading", download=name, bytes=size, attempt=attempt,
                               stalled=(size == last))
                if min_free_bytes:
                    free = _disk_free_bytes(local.parent)
                    if free is not None and free < min_free_bytes:
                        proc.kill()
                        proc.wait(timeout=60)
                        print("[htr-case] ⚠ качання %s обірвано: вільного диска %.1f ГБ"
                              % (name, free / 2 ** 30), flush=True)
                        try:
                            local.unlink()
                        except OSError:
                            pass
                        return -2
                last = size
        if rc == 0:
            try:
                size = local.stat().st_size
                if size >= 8 * 2 ** 20:    # малий файл швидкості каналу не показує
                    _DOWNLOAD_SEEN_BPS.append(size / max(0.001, time.time() - t_attempt))
            except OSError:
                pass
            return 0
        print("[htr-case] ⚠ качання %s: curl rc=%s (спроба %d/%d)" % (name, rc, attempt, tries),
              flush=True)
        try:
            local.unlink()
        except OSError:
            pass
        if attempt < tries:
            time.sleep(DOWNLOAD_RETRY_SLEEP_SEC)
    return rc


#: ⏩ Передзавантаження кадрів наступного тому стартує лише за стількох вільних
#: байтів і обривається, коли диск спускається нижче другої межі. Бокси Vast
#: приходять із 15 ГБ, а архів тому — до 2 ГБ плюс стільки ж розпакованого.
PREFETCH_MIN_FREE_BYTES = 4 * 2 ** 30
PREFETCH_ABORT_FREE_BYTES = 2 * 2 ** 30
#: Скільки разів у кінці черги повторювати томи, що впали або лишились неповні.
QUEUE_RETRY_PASSES = 1


def _disk_free_bytes(path="/tmp"):
    """Вільне місце на диску теки (або найближчої наявної вище); None — невідомо."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(str(p)).free
    except OSError:
        return None


def _url_name(url, default="download.tgz"):
    # ‼ Ім'я — БЕЗ query. Presigned URL несе підпис (`?X-Amz-Signature=…`), і
    # взяте цілком воно стає недопустимим іменем файла: curl падає з rc=23
    # (write error), що читається як «мережа», хоча мережа ні до чого.
    return url.split("?", 1)[0].rsplit("/", 1)[-1] or default


class _Prefetch:
    """⏩ Кадри НАСТУПНОГО тому качаються, поки флот читає поточний.

    🔴 Між томами черги бокс стояв на качанні: spr-2462 — 913 МБ за 417 с
    (13.09.2026), а на боксі з каналом 30 Мбіт/с (irnbuv1 q23, 23 томи) качання
    з'їдало помітну частку оренди. Карта й ядра в цей час простоювали.
    """

    def __init__(self, url, local):
        self.url = url
        self.local = Path(local)
        self.name = self.local.name
        self.rc = None
        self.sec = None
        self._thread = None

    def start(self):
        free = _disk_free_bytes(self.local.parent)
        if free is not None and free < PREFETCH_MIN_FREE_BYTES:
            print("[htr-case] ⏩ передзавантаження %s пропущено: вільного диска %.1f ГБ"
                  % (self.name, free / 2 ** 30), flush=True)
            return False
        self.local.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="prefetch", daemon=True)
        self._thread.start()
        print("[htr-case] ⏩ передзавантажую кадри наступного тому: %s" % self.name, flush=True)
        return True

    def _run(self):
        t = time.time()
        try:
            self.rc = _download_with_heartbeat(self.url, self.local, self.name, heartbeat=False,
                                               min_free_bytes=PREFETCH_ABORT_FREE_BYTES)
        except Exception as exc:
            print("[htr-case] ⚠ передзавантаження %s: %r" % (self.name, exc), flush=True)
            self.rc = -3
        self.sec = time.time() - t

    def take(self, url):
        """Готовий файл (дочекавшись, якщо ще качається) або None — качати звичайно."""
        if url != self.url or self._thread is None:
            return None
        while self._thread.is_alive():
            size = self.local.stat().st_size if self.local.exists() else 0
            _set_phase("downloading", download=self.name, bytes=size, prefetch=True)
            self._thread.join(timeout=DOWNLOAD_HEARTBEAT_SEC)
        if self.rc == 0 and self.local.is_file():
            return self.local
        print("[htr-case] ⚠ передзавантаження %s не вдалось (rc=%s) — качаю звичайно"
              % (self.name, self.rc), flush=True)
        try:
            self.local.unlink()
        except OSError:
            pass
        return None


def _start_prefetch(queue, n, prefetches):
    """Почати качати кадри тому №n (1-based), якщо він є і ще не качається."""
    if n > len(queue) or n in prefetches:
        return
    url = str(queue[n - 1].get("pages_url") or "")
    if not url:
        return
    pf = _Prefetch(url, Path("/tmp/htrcase/prefetch_%02d" % n) / _url_name(url))
    if pf.start():
        prefetches[n] = pf


def _adopt_legacy_seg_cache(work):
    """Перенести кеш сегментації зі старих тек у стабільну `SEG_CACHE_ARC`.

    🔴 Чекпоінти й засіви, зроблені до 15.09.2026, несуть кеш у теці з хешу
    шляху `/tmp/htrcase/pages_dl_NN__<хеш>`, а раннер тепер шукає його в
    `case`. Без перенесення такий кеш мовчки не влучає: irnbuv1 q9 (Скриба,
    засів прототипом) сегментував t03, t07, t08 наново, хоча сегментація
    лежала на боксі поруч. Наявний у `case` файл не перезаписується.
    """
    root = Path(work) / "data" / "derived" / "htr_seg"
    target = Path(work) / SEG_CACHE_ARC
    if not root.is_dir():
        return 0
    moved = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.resolve() == target.resolve():
            continue
        for f in sorted(d.glob("*.seg.json.gz")):
            dst = target / f.name
            if dst.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst))
            moved += 1
    return moved


def _drop_case_frames(merged):
    """Прибрати кадри закритого тому: вивід лежить окремо, а диск боксу — 15 ГБ.

    Досі кадри кожного тому лишались у `/tmp/htrcase/pages_dl_NN` до кінця
    черги — на 23 томах це кілька гігабайт, яких бракує передзавантаженню.
    """
    for key in ("_pages_dl", "_pages_unpack", "_prefetch_dir"):
        d = merged.get(key)
        if d:
            shutil.rmtree(str(d), ignore_errors=True)


def _record_result(results, n, res, attempt, retry_pending):
    """Підсумок тому №n — у список черги й у прогрес для наглядача."""
    while len(results) < n:
        results.append({"complete": False})
    results[n - 1] = res
    entry = {"index": n, "case": res.get("case"), "complete": bool(res.get("complete")),
             "n_pages_txt": res.get("n_pages_txt"), "attempt": attempt}
    if not entry["complete"]:
        entry["error"] = str(res.get("error") or "неповна")[:400]
        entry["retry_pending"] = bool(retry_pending)
    for j, old in enumerate(_QUEUE_RESULTS):
        if old.get("index") == n:
            _QUEUE_RESULTS[j] = entry
            break
    else:
        _QUEUE_RESULTS.append(entry)
    # Термінальна фаза публікується через `_set_phase`, який копіює останній
    # прогрес — підсумок останнього тому мусить бути вже в ньому.
    _LAST_PROGRESS["results"] = list(_QUEUE_RESULTS)


def _material_key(model_name, voices, frame_mpx_p95):
    return {"model": str(model_name), "voices": sorted(str(v) for v in voices or []),
            "mpx": float(frame_mpx_p95 or 0)}


def _carried_fleet(memory, material):
    """Що флот попереднього тому може передати цьому — або None.

    Переноситься лише на ту саму модель із тими самими голосами й на кадри,
    близькі за площею: від цього залежить і пам'ять шарда, і коліно карти.
    Невідома площа (0) з будь-якого боку переносу не забороняє — черга йде
    одним планом по одному жанру.
    """
    if not memory or not memory.get("n_stable"):
        return None
    old = memory.get("material") or {}
    if old.get("model") != material["model"] or old.get("voices") != material["voices"]:
        return None
    a, b = float(old.get("mpx") or 0), float(material.get("mpx") or 0)
    if a > 0 and b > 0 and max(a, b) / min(a, b) > FLEET_MEMORY_MPX_RATIO:
        return None
    return memory


def _wait_for_appended(queue, seen, keep_warm_sec, done_count):
    """Фаза `idle`: чекати довісок до `keep_warm_sec` тиші. True = черга виросла.

    Публікує `idle` кожні `IDLE_POLL_SEC` — це і серцебиття для наглядача
    (інакше старий прогрес виглядав би як мертвий раннер), і момент, коли він
    може забрати вже готове, не чекаючи гасіння.
    """
    if keep_warm_sec <= 0:
        return False
    t0 = time.time()
    while time.time() - t0 < keep_warm_sec:
        _set_phase("idle", idle_since_epoch=int(t0),
                   keep_warm_sec=keep_warm_sec,
                   idle_left_sec=int(keep_warm_sec - (time.time() - t0)),
                   case_index=done_count, cases_total=done_count)
        fresh = _read_appended(seen)
        if fresh:
            queue.extend(fresh)
            return True
        time.sleep(IDLE_POLL_SEC)
    print("[htr-case] теплий бокс: тиша %d с — завершую" % keep_warm_sec, flush=True)
    return False


def _verify_scripts(code_dir, expected):
    """sha256 скриптів на боксі; розбіжність з очікуваними — RuntimeError.

    Повертає {ім'я: sha256} усіх `*.py` у теці — це йде в підсумок заходу
    (`summary["scripts_sha256"]`), щоб реєстр знав, ЯКИМ раннером зняте число.
    """
    import hashlib

    actual = {}
    for path in sorted(Path(code_dir).glob("*.py")):
        actual[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    stale = {name: (sha, actual.get(name))
             for name, sha in (expected or {}).items()
             if sha and actual.get(name) != sha}
    if stale:
        lines = ["%s: очікувано %s…, у ассетах %s…" % (n, e[:12], (a or "немає")[:12])
                 for n, (e, a) in sorted(stale.items())]
        raise RuntimeError("застарілі скрипти в ассетах — перепакуй архів: "
                           + "; ".join(lines))
    return actual


def _main_inner(params):
    cases = list(params.get("cases") or [])
    if not cases:
        return _run_case(params)

    print("[htr-case] черга з %d справ на одному боксі" % len(cases), flush=True)
    results = []
    # 🔴 Черга РОСТЕ: між справами читається довісок. Тому не `for` по списку.
    queue = list(cases)
    seen = {str(c.get("case") or "").strip() for c in queue}
    # 🔥 Теплий бокс: після вичерпання черги бокс лишається живим ще
    # `keep_warm_sec` і приймає довісок. Холодний старт коштує ~5 хв оренди
    # (std160 05.09.2026: оренда → перша сторінка), а справа на 160 сторінок —
    # 2.5 хв роботи; довісити на теплий бокс — лише дані.
    keep_warm_sec = int(params.get("keep_warm_sec") or 0)
    retry_passes = max(0, int(params.get("queue_retry_passes", QUEUE_RETRY_PASSES) or 0))
    # 🎛 Пам'ять флоту живе ВСЮ чергу: том передає наступному розмір флоту,
    # коліно й виміряну пам'ять шарда (`_FleetRegulator.remember`).
    fleet_memory = {}
    # ⏩ Кадри наступного тому, що качаються, поки читається поточний.
    prefetches = {}
    del _QUEUE_RESULTS[:]

    def run_one(i, attempt):
        case = queue[i - 1]
        merged = dict(params)
        merged.update(case)
        merged["_case_index"] = i
        merged["_cases_total"] = len(queue)
        # 🔴🔴 ЖОДНА справа в черзі не оголошує себе останньою. `_final_case`
        # публікує `phase="done"`, а наглядач вважає це терміналом: забирає
        # результат і ГАСИТЬ бокс. Доки останню справу позначали фінальною,
        # довісок був неможливий саме в найтиповішому випадку — «докинути малу
        # справу, поки рахується велика»: велика і є остання. Термінал тепер
        # публікує сам цикл, після фінального перечитування довіска.
        merged["_final_case"] = False
        # Кожна справа — своя тека виводу й свої кадри: інакше друга побачила б
        # тексти першої як «уже зроблені» й тихо оголосила себе повною.
        merged["_out_root"] = str(KAGGLE_WORKING / _case_slug(case.get("case") or f"case{i}"))
        merged["_pages_dl"] = "/tmp/htrcase/pages_dl_%02d" % i
        # 🔴 І тека РОЗПАКУВАННЯ теж посправна. Тека завантаження була
        # посправною з самого початку, а розпаковувались усі справи в одну
        # константу `/tmp/htrcase/pages` — при тому, що `htr_cloud_plan.py`
        # пакує кадри ПЛАСКО (`arcname=frame.name`), а імена кадрів у різних
        # справах однакові (`0001_L.jpg`). Тобто справа 2 перезаписувала
        # частину кадрів справи 1 і успадковувала її хвіст як свій: тексти
        # чужої справи лягали в її теку під її ж іменами. Знаменник рахується
        # з тієї самої теки, тож усі три ворота повноти казали «повно».
        # Ціна помилки тут не грошова, а доказова — хибна прив'язка аркуша.
        merged["_pages_unpack"] = "/tmp/htrcase/pages_%02d" % i
        merged["_prefetch_dir"] = "/tmp/htrcase/prefetch_%02d" % i
        # Ассети качаються лише першою справою — далі вони вже в /kaggle/input.
        merged["_skip_assets"] = i > 1
        merged["_fleet_memory"] = fleet_memory
        merged["_prefetch"] = prefetches.pop(i, None)
        if attempt == 1:
            # Наступний том ще не читався — його кадри можна тягнути вже зараз.
            merged["_on_frames_ready"] = lambda: _start_prefetch(queue, i + 1, prefetches)
        print("[htr-case] ── справа %d/%d: %s%s" % (
            i, len(queue), case.get("case"),
            "" if attempt == 1 else " (повтор №%d)" % (attempt - 1)), flush=True)
        # 🔴 Лічильники попереднього тому НЕ успадковуються: `_set_phase` копіює
        # останній прогрес, і новий том показував «11 збоїв» і чужий темп
        # (irnbuv1_opys1_t21–t23, 14.09.2026).
        _set_phase("starting", case=case.get("case"), case_index=i, cases_total=len(queue),
                   n_pages_expected=int(case.get("estimated_n_pages") or 0),
                   pages_done=0, shards=[], attempt=attempt,
                   pages_failed=0, pages_skipped=0, pages_resumed=0, pages_per_hour=0,
                   eta_sec=None, oom_events=0, complete=None, missing=[], missing_count=0)
        try:
            return _run_case(merged)
        except Exception as exc:
            # Провал однієї справи не має ховати решту черги: наступні можуть
            # пройти, а звіт мусить назвати саме ту, що впала.
            print("[htr-case] ✗ справа %s впала: %r" % (case.get("case"), exc), flush=True)
            return {"case": case.get("case"), "complete": False, "error": repr(exc)}
        finally:
            _drop_case_frames(merged)

    i = 0
    retries_left = retry_passes
    while True:
        while i < len(queue):
            i += 1
            _record_result(results, i, run_one(i, 1), 1, retry_passes > 0)
            # МІЖ справами, а не всередині: напівстан посеред справи рятувати нема
            # чим, а тут черга в чистому місці — попередня закрита, наступна ще не
            # почалась.
            #
            # 🔴 Читається ПІСЛЯ кожної справи, зокрема після останньої відомої.
            # Саме це й закриває гонку: доки внутрішній цикл не вичерпав чергу,
            # `done` ще не опубліковано, і бокс живий.
            queue.extend(_read_appended(seen))
        # 🔁 Повтор томів, що впали або лишились неповні, — поки бокс теплий.
        # Досі провал тому лише писався в лог: irnbuv1 q23 (14.09.2026) закрив
        # чергу з 8 томами без жодного чекпоінта, і дочитати їх можна було лише
        # новою орендою.
        failed = [n for n, r in enumerate(results, 1) if not r.get("complete")]
        if failed and retries_left > 0:
            retries_left -= 1
            attempt = retry_passes - retries_left + 1
            print("[htr-case] 🔁 повтор неповних томів черги (%d): %s" % (
                len(failed), ", ".join(str(queue[n - 1].get("case")) for n in failed)),
                flush=True)
            for n in failed:
                _record_result(results, n, run_one(n, attempt), attempt, retries_left > 0)
            queue.extend(_read_appended(seen))
            continue
        if _wait_for_appended(queue, seen, keep_warm_sec, len(queue)):
            continue
        break

    ok = sum(1 for r in results if r.get("complete"))
    # 🔴 Термінал публікує ЦИКЛ, а не остання справа: доки це робила справа,
    # наглядач гасив бокс до того, як довісок міг бути прочитаний.
    _set_phase("done" if ok == len(queue) else "failed",
               case_index=len(queue), cases_total=len(queue))
    print("%s%s" % (DONE_SENTINEL, json.dumps(
        {"queue": len(queue), "complete": ok,
         "cases": [{"case": r.get("case"), "complete": r.get("complete"),
                    "n_pages_txt": r.get("n_pages_txt"),
                    "missing": len(r.get("missing_pages") or [])} for r in results]},
        ensure_ascii=False)), flush=True)
    if ok < len(queue):
        raise RuntimeError("черга: %d із %d справ неповні" % (len(queue) - ok, len(queue)))
    return {"queue": len(queue), "results": results}


def _case_slug(name):
    keep = "".join(ch for ch in str(name) if ch.isalnum() or ch in "-_.")
    return keep[:64] or "case"


def _run_case(params):
    t_start = time.time()
    _ensure_deps()
    pages_slug = str(params.get("pages_slug") or "pages")
    models_slug = str(params.get("models_slug") or "models")
    scripts_slug = str(params.get("scripts_slug") or "scripts")
    state_slug = str(params.get("state_slug") or "").strip()
    model_name = str(params.get("model") or "pysar_cyr_v17.pt")
    voices = [v.strip() for v in str(params.get("voices") or "").split(",") if v.strip()]
    shards = int(params.get("shards") or 0)
    enhance = str(params.get("enhance") or "auto")
    script = str(params.get("script") or "cyrillic")
    limit = int(params.get("limit") or 0)
    batch = int(params.get("batch") or 0)
    voice_batch = int(params.get("voice_batch") or 0)
    gpu_lock = bool(params.get("gpu_lock") or False)
    seg_cache = bool(params.get("seg_cache") or False)
    checkpoint_sec = max(30, int(params.get("checkpoint_sec") or 120))
    make_bundle = bool(params.get("bundle") or False)
    threads_per_shard = int(params.get("threads_per_shard") or 0)
    ckpt_urls = list(params.get("ckpt_urls") or [])      # presigned PUT, по одному на раунд
    resume_urls = list(params.get("resume_urls") or [])  # presigned GET на попередні ckpt
    case_label = str(params.get("case") or "")
    stall_sec = max(60, int(params.get("stall_sec") or STALL_SEC_DEFAULT))
    restart_max = max(0, int(params.get("shard_restart_max") or SHARD_RESTART_MAX))
    # Скільки разів доганяти сторінки, які шарди не зробили. Нуль — свідома
    # відмова від догону (для тестів воріт повноти).
    catchup_passes = int(params.get("catchup_passes", 2) or 0)
    # 🔴 Неповний результат за замовчуванням = ПРОВАЛ job'а. Саме тому, що
    # раніше він виглядав як успіх: `rc=0`, `failed_shards: []`, і 46
    # сторінок просто не існувало.
    allow_incomplete = bool(params.get("allow_incomplete") or False)

    # ── вхід по HTTP замість SFTP ───────────────────────────────────────────
    # ‼ SFTP бекенда дає ~0.4 МБ/с (замір Vast 2026-08-09: 302 МБ за 20 хв), тобто
    # корпус на 31 ГБ їхав би ~22 години — довше за саму роботу. Тому великі
    # архіви качає САМ бокс зі свого каналу (755 Mbps), а SFTP лишається лише для
    # дрібниць. Джерело — будь-який HTTP: власний VPS, S3, що завгодно.
    # ‼ Замір каналу ПЕРЕД тим, як на нього покластися. Заявлена швидкість хоста
    # у маркетплейсі не означає нічого: оффер #47127187 обіцяв 755/797 Mbps, а
    # віддавав 0.5 Mbps (2026-08-09) — 302 МБ їхали 20 хв, pip падав із
    # «versions: none» (обрив індексу), і діагностувалось це годину. Десять
    # секунд тут рятують годину там.
    # ‼ НЕ `or 5.0`: нуль у Python falsy, тож «вимкнути перевірку» (0) мовчки
    # перетворювалось назад на 5 — і job падав на порозі, який щойно зняли.
    _nm = params.get("min_net_mbps")
    net_min = 5.0 if _nm is None or _nm == "" else float(_nm)
    probe_url = str(params.get("assets_url") or params.get("pages_url") or "")
    if probe_url and net_min > 0:
        t = time.time()
        rc = subprocess.call(["curl", "-s", "-o", "/dev/null", "--max-time", "15",
                              "-r", "0-8000000", probe_url])
        dt = max(0.001, time.time() - t)
        mbps = (8.0 / dt) * 8  # 8 МБ → Мбіт/с
        print("[htr-case] канал до джерела: ~%.1f Мбіт/с (%.1f с на 8 МБ, rc=%s)"
              % (mbps, dt, rc), flush=True)
        # rc=28 — це `--max-time`, тобто ЗАМІР дійшов до стелі часу, а не збій:
        # на повільному каналі 8 МБ просто не влазять у вікно, і швидкість при
        # цьому виміряна правильно. Все інше (DNS, 404, обрив) — справжня помилка.
        if rc not in (0, 28) or (net_min > 0 and mbps < net_min):
            raise RuntimeError(
                "мережа бокса непридатна: ~%.1f Мбіт/с при порозі %.1f. "
                "Це хост, а не наш код — беріть інший оффер "
                "(-p min_net_mbps=0 щоб ігнорувати)." % (mbps, net_min))

    pages_dl = Path(params.get("_pages_dl") or "/tmp/htrcase/pages_dl")
    assets_url = "" if params.get("_skip_assets") else str(params.get("assets_url") or "")
    if params.get("_skip_assets"):
        print("[htr-case] моделі й скрипти вже на боксі — качаю лише кадри", flush=True)
    for url, dest in ((assets_url, KAGGLE_INPUT),
                      (str(params.get("pages_url") or ""), pages_dl)):
        if not url:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        name = _url_name(url)
        local = Path("/tmp/htrcase") / name
        local.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        prefetched = params.get("_prefetch") if dest == pages_dl else None
        ready = prefetched.take(url) if prefetched is not None else None
        if ready is not None:
            local, rc = ready, 0
        else:
            rc = _download_with_heartbeat(url, local, name)
        if rc != 0:
            raise RuntimeError("curl %s провалився (rc=%s)" % (url, rc))
        size = local.stat().st_size
        dt = max(0.001, time.time() - t)
        if ready is not None:
            print("[htr-case] ⏩ кадри %s уже передзавантажено: %.0f МБ (дочікування %.0f с)"
                  % (name, size / 1e6, dt), flush=True)
        else:
            print("[htr-case] завантажено %s: %.0f МБ за %.0f с (%.1f МБ/с)"
                  % (name, size / 1e6, dt, size / 1e6 / dt), flush=True)
        _set_phase("unpacking", download=name, bytes=size)
        with tarfile.open(local) as tf:
            tf.extractall(dest)
        local.unlink(missing_ok=True)
        if dest == pages_dl:
            params = dict(params)
            params["_pages_root"] = str(dest)

    # ⏩ Свої кадри на місці — канал вільний для кадрів наступного тому черги.
    on_ready = params.get("_on_frames_ready")
    if on_ready is not None:
        try:
            on_ready()
        except Exception as exc:
            print("[htr-case] ⚠ передзавантаження не стартувало: %r" % (exc,), flush=True)

    forced_pages = params.get("_pages_root")
    pages_root = Path(forced_pages) if forced_pages else _slug_dir(pages_slug)
    models_root = _slug_dir(models_slug)
    scripts_root = _slug_dir(scripts_slug)

    # Архів кадрів замість голих файлів: бекенд заливає том ФАЙЛ ЗА ФАЙЛОМ, і на
    # справі в тисячі сканів це години на самій лише заливці. Один tgz їде як один
    # файл; розпакування в контейнері коштує секунди.
    tarballs = [p for p in sorted(pages_root.rglob("*")) if p.suffix.lower() in (".tgz", ".tar")
                or p.name.lower().endswith(".tar.gz")]
    if tarballs:
        unpacked = Path(params.get("_pages_unpack") or "/tmp/htrcase/pages")
        # Прибрати перед розпакуванням: тека могла лишитись від попередньої
        # справи (повторний запуск, resume з чекпоінта), і тоді до кадрів цієї
        # справи домішались би чужі — мовчки, бо знаменник береться звідси ж.
        if unpacked.exists():
            shutil.rmtree(unpacked, ignore_errors=True)
        unpacked.mkdir(parents=True, exist_ok=True)
        for tb in tarballs:
            t = time.time()
            with tarfile.open(tb) as tf:
                tf.extractall(unpacked)
            print("[htr-case] розпаковано %s за %.0f с" % (tb.name, time.time() - t), flush=True)
        pages_root = unpacked

    pages = _pages_of(pages_root)
    if not pages:
        raise RuntimeError("під %s немає кадрів %s" % (pages_root, PAGE_SUFFIXES))
    n_pages = min(limit, len(pages)) if limit else len(pages)

    # Робоча тека — у /kaggle/working: бекенд синхронізує саме її, тож навіть
    # аварійне завершення раннера лишає текст на томі. У черзі кожна справа
    # має ВЛАСНУ підтеку: спільна означала б, що друга справа бачить тексти
    # першої як свої вже готові й оголошує себе повною, не прочитавши нічого.
    work = Path(params.get("_out_root") or KAGGLE_WORKING)
    work.mkdir(parents=True, exist_ok=True)
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = work / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Скрипти — на локальний диск: htr_case_run.py кладе теку свого файла в sys.path
    # і звідти бере gpu_sato / fast_geom / pysar_lines_infer.
    code = Path("/tmp/htrcase/scripts")
    code.mkdir(parents=True, exist_ok=True)
    staged = []
    for src in scripts_root.rglob("*.py"):
        shutil.copy2(src, code / src.name)
        staged.append(src.name)
    # 🔴 Свіжість раннера: 05.09.2026 в ассетах лежала копія без ліміту
    # потоків, і флот ішов утричі повільніше — перевірялись лише ІМЕНА.
    # Хеші приходять із плану (`htr plan` рахує їх з архіву); розбіжність
    # валить старт до завантаження моделей, тобто до оплачуваної роботи.
    scripts_sha = _verify_scripts(code, params.get("scripts_sha256") or {})
    missing = [n for n in NEEDED_SCRIPTS if n not in staged]
    if missing:
        raise RuntimeError("у слагу %r бракує скриптів: %s" % (scripts_slug, missing))

    model_path = _find_model(models_root, model_name)
    voice_paths = [str(_find_model(models_root, v)) for v in voices]

    # ── стан із попереднього запуску ────────────────────────────────────────
    state_root = None
    resumed = 0
    if state_slug:
        in_mount = _mount(IN_MOUNTS)
        if in_mount is None:
            print("[htr-case] ⚠ state_slug задано, але вхідного тому не видно — "
                  "чекпоінт МІЖ запусками вимкнено", flush=True)
        else:
            state_root = in_mount / state_slug
            state_root.mkdir(parents=True, exist_ok=True)
            # Архіви розпаковуються ПО ПОРЯДКУ: пізніший чекпоінт перекриває
            # раніший, бо `_htr_meta.part*.json` у ньому повніший.
            ckpts = sorted(state_root.glob("ckpt_*.tgz"))
            for ck in ckpts:
                try:
                    with tarfile.open(ck) as tf:
                        tf.extractall(work)
                    resumed += 1
                except Exception as exc:
                    print("[htr-case] ⚠ чекпоінт %s не розпакувався: %r" % (ck.name, exc),
                          flush=True)
            loose = _mirror(state_root, work, {})
            print("[htr-case] стан із %s: %d архів(ів) + %d файл(ів)"
                  % (state_root, resumed, loose), flush=True)
    last_present = 0  # найбільший номер архіву, який РЕАЛЬНО існує в хмарі
    misses = 0        # 404 поспіль — серія чекпоінтів без дірок, далі порожньо
    for i, url in enumerate(resume_urls, 1):
        dst = Path("/tmp/htrcase/resume_%04d.tgz" % i)
        rc = subprocess.call(["curl", "-fsSL", "--retry", "2", "--max-time", "1800",
                              "--speed-limit", "10240", "--speed-time", "120",
                              "-o", str(dst), url])
        if rc != 0:
            # rc=22 — HTTP 4xx, тобто «такого чекпоінта ще немає». Це НОРМА:
            # посилання нарізаються з запасом на всі можливі раунди, і на
            # першому запуску не існує жодного. Шуміти тут означало б виводити
            # шість десятків «помилок» на здоровому старті.
            if rc != 22:
                print("[htr-case] ⚠ resume-архів №%d не завантажився (rc=%s)" % (i, rc),
                      flush=True)
                continue
            # 🔴 Серія чекпоінтів іде без дірок (порожній раунд слот не палить),
            # тож три 404 поспіль означають «далі нічого немає». Доти цикл
            # проходив УСІ посилання: `max_hours=8` = 392 curl по ~0.5 с на
            # КОЖНУ справу черги — `startup_sec` 232 с на другій справі, де
            # pip і ассети вже пропущено (захід live2, 04.09.2026).
            misses += 1
            if misses >= RESUME_MISSES_TO_STOP:
                print("[htr-case] resume: спинився на №%d — %d порожніх поспіль"
                      % (i, misses), flush=True)
                break
            continue
        misses = 0
        last_present = i
        try:
            with tarfile.open(dst) as tf:
                tf.extractall(work)
            resumed += 1
        except Exception as exc:
            print("[htr-case] ⚠ resume-архів №%d биті дані: %r" % (i, exc), flush=True)
        dst.unlink(missing_ok=True)
    if resume_urls:
        print("[htr-case] з хмари підхоплено %d/%d архів(ів) стану"
              % (resumed, len(resume_urls)), flush=True)
    if seg_cache:
        adopted = _adopt_legacy_seg_cache(work)
        if adopted:
            print("[htr-case] 💾 кеш сегментації зі старої теки → %s: %d кадрів"
                  % (SEG_CACHE_ARC, adopted), flush=True)
    if last_present:
        params = dict(params)
        params["_ckpt_start"] = last_present
        print("[htr-case] нумерація чекпоінтів продовжиться з №%d — щоб не "
              "затерти серію попередньої оренди" % (last_present + 1), flush=True)

    already = len(list(out_dir.glob("*.txt")))
    if already:
        print("[htr-case] ▶ у теці вже %d сторінок — вони будуть пропущені" % already,
              flush=True)

    lock_path = Path("/tmp/htrcase/gpu.lock")
    base = [
        sys.executable, str(code / "htr_case_run.py"),
        "--case-dir", str(pages_root),
        "--out-dir", str(out_dir),
        "--model", str(model_path),
        "--device", "cuda:0",
        "--script", script,
        "--enhance", enhance,
        # Машинний канал прогресу: без нього тиша шарда невідрізненна від
        # роботи, і вотчдогу нема на що дивитись.
        "--progress-json",
    ]
    if voice_paths:
        base += ["--models", ",".join(voice_paths)]
    if limit:
        base += ["--limit", str(limit)]
    if batch:
        base += ["--batch", str(batch)]
    if voice_batch > 1:
        base += ["--voice-batch", str(voice_batch)]
    if not seg_cache:
        base += ["--no-seg-cache"]
    else:
        # 💾 Кеш сегментації — у СТАБІЛЬНІЙ теці справи. Без явного шляху раннер
        # рахує її з хешу `/tmp/htrcase/pages_NN`, тобто з місця справи в черзі:
        # кеш, засіяний для перечитування іншою моделлю, влучав лише тоді, коли
        # справа стояла в черзі на тому самому місці, що й першого разу.
        base += ["--seg-cache-dir", str(work / SEG_CACHE_ARC)]
    # Стеля рядків на сторінку. Дефолти локального скрипта (400 + автоперепуск
    # 1600) працюють і тут, але щільний формуляр дешевше пустити одразу з
    # піднятою стелею: перепуск коштує ДРУГОГО проходу сторінки (48 с проти
    # 23), а на шлюбній метриці в стелю впирається більш як половина аркушів.
    if params.get("max_endpoints"):
        base += ["--max-endpoints", str(int(params["max_endpoints"]))]
    if params.get("ceiling_retry") is not None:
        base += ["--ceiling-retry", str(int(params["ceiling_retry"]))]
    # 🔴 Перевірка орієнтації кадру. Локально її вмикають звично, а в хмару
    # прапорець не доходив ЗОВСІМ — і перевернутий аркуш давав псевдокириличний
    # шум, невідрізненний від поганого письма. На костелі ф.685 це 109 сторінок
    # і два раунди підозр за маркерами формуляра, поки не прогнали локально.
    if params.get("orient_check"):
        base += ["--orient-check"]
    # 🔴 Куди рахувати sato: на карту чи на процесор. Дефолт раннера — карта, і
    # сам він попереджає, що на CPU «сторінка коштує вдвічі». Але це замір
    # ОДНОГО процесу; під шардингом картина інша, бо шарди б'ються за карту, а
    # ядра простоюють. Заміряно на живому боксі 04.09.2026: GPU 91%, при цьому
    # з 46 ядер квоти працює 11.7. Перевірити це можна лише прапорцем, і доти
    # він у хмару не доходив узагалі.
    if params.get("no_gpu_sato"):
        base += ["--no-gpu-sato"]
    if gpu_lock:
        # ⚠ Лок серіалізує GPU-фазу. На кількох картах спільний лок звів би
        # усю багатокартковість нанівець, тож ім'я лока робиться пер-карту
        # нижче, у `_start_shard`; тут лише позначаємо, що лок потрібен.
        base += ["--gpu-lock", str(lock_path)]

    n_gpus = max(1, len(_free_vram_per_card()))
    # 🎛 Регулятор — лише коли раннер в ассетах уміє і клейми, і злив: старий
    # раннер без зливу не вийшов би з флоту за проханням, і звужувати флот
    # довелося б убивством посеред сторінки.
    runner_script = code / "htr_case_run.py"
    regulate = (bool(params.get("regulate"))
                and bool(params.get("dynamic_pages", True))
                and _runner_supports(runner_script, "--claim")
                and _runner_supports(runner_script, "_drain"))
    if params.get("regulate") and not regulate:
        print("[htr-case] ⚠ регулятор вимкнено: раннер в ассетах не знає клеймів або "
              "зливу — флот фіксований (перепакуй архів новим раннером)", flush=True)
    shards_max = 0
    fleet_memory = params.get("_fleet_memory")
    material = _material_key(model_name, voices, params.get("frame_mpx_p95"))
    carried = None
    # 🎛 Ядер на шард. Перечитування іншою моделлю з готовою сегментацією не
    # платить за геометрію й sato (~74% процесора сторінки) — там план кладе
    # менше число, і регулятор може вирости понад звичну стелю за ядрами.
    reg_cores = float(params.get("cores_per_shard") or 0) or REG_CORES_PER_SHARD
    if regulate:
        # Число від воріт (або з рук) — СТЕЛЯ.
        shards_max = (int(params.get("shards_max") or shards or 0)
                      or _auto_shards(1.5, cores_per_shard=min(CORES_PER_SHARD, reg_cores)))
        shards_max = max(1, min(MAX_SHARDS, shards_max))
        # 🔴 Старт і за ядрами: вище `ядра / ядер на шард` регулятор однаково
        # не виросте, а стартувати там — одразу оверсабскрайб.
        cores_cap = max(1, int(_usable_cores() // reg_cores))
        carried = _carried_fleet(fleet_memory, material)
        if carried:
            # 🎛 Том черги стартує з того, до чого флот дійшов на попередньому.
            # Досі кожен том починав з нуля: 23 томи irnbuv1 (14.09.2026) — усі
            # від 10 шардів при стелі 20, бо регулятор дозріває 4–5 хв, а том
            # живе 3–6; коліно, знайдене на spr-2467 і cdiak1040-1-3, губилось
            # на наступному ж томі.
            shards = max(1, min(shards_max, int(carried["n_stable"]), cores_cap))
            knee = carried.get("knee")
            print("[htr-case] 🎛 регулятор: старт %d шардів — з попереднього тому (%s), "
                  "стеля %d" % (shards, "коліно %s" % knee if knee is not None
                               else "коліна ще не знайдено", shards_max), flush=True)
        else:
            # 🔴 Старт — від оцінки наглядача (площа кадру), а не від
            # `max(3.3, оцінка)`: на легких кадрах шард займає 0.6–1.3 ГБ карти
            # (A10, spr-2474: 592 МБ), і 3.3 ГБ садили флот удвічі нижче, ніж
            # влазить. Від важкого матеріалу стережуть OOM і тиск карти.
            start_gb = float(params.get("vram_gb_per_shard") or 0) or REG_START_GB
            shards = max(1, min(shards_max, _auto_shards(start_gb),
                                REG_START_PER_CARD_MAX * n_gpus, cores_cap))
            print("[htr-case] 🎛 регулятор: старт %d шардів (%.1f ГБ на шард), стеля %d — "
                  "далі флот міряє темп і пам'ять сам" % (shards, start_gb, shards_max),
                  flush=True)
    elif shards < 1:
        shards = _auto_shards(float(params.get("vram_gb_per_shard") or GB_PER_SHARD))
    if n_gpus > 1:
        print("[htr-case] карт %d — шарди розкладаю по всіх (cuda:0..%d)"
              % (n_gpus, n_gpus - 1), flush=True)
    if threads_per_shard <= 0:
        # під регулятором флот може вирости до стелі — потоки ділимо на неї
        threads_per_shard = max(1, min(8, _usable_cores() // max(shards, shards_max)))
    print("[htr-case] %d кадрів, %d шард(ів), карта=%s, ядер %d (видно %s)"
          % (n_pages, shards, _gpu_name(), _usable_cores(), os.cpu_count()),
          flush=True)
    print("[htr-case] модель=%s, голоси=%s" % (model_path.name,
                                               [Path(v).name for v in voice_paths]), flush=True)

    # ── чекпоінт-тред ───────────────────────────────────────────────────────
    # ЄДИНА ціль — `state_slug` на вхідному томі, і тільки архівами. Вихідний том
    # сюди не входить навмисно: бекенд синхронізує `working` сам (періодично і в
    # `finally`), а другий пофайловий обхід того самого дерева коштував би ще
    # хвилину на раунд — на мережевому томі один дрібний файл ≈ 1 с.
    stop = threading.Event()
    # 🔴 Нумерація ПРОДОВЖУЄТЬСЯ після відновлення, а не починається з одиниці.
    # Інакше нова оренда пише `ckpt_0001.tgz` поверх першого архіву попередньої
    # серії. Поки resume спрацював, це безпечно (новий перший чекпоінт містить
    # усе відновлене, тобто є надмножиною) — але саме тоді, коли resume впав
    # ЧАСТКОВО, ми затираємо базу попереднього прогону остаточно. Тобто
    # рятувальний механізм ламався рівно у випадку, заради якого існує.
    # Здоров'я чекпоінтів. Окремий словник, а не поле `fleet`, СВІДОМО: потік
    # чекпоінтів стартує РАНІШЕ, ніж будується `fleet`, і пряме звертання
    # працювало б лише завдяки тому, що перший раунд настає через 120 с. Той
    # самий словник кладеться у `fleet` за посиланням, тож усі оновлення
    # видно в кожній публікації без жодної залежності від порядку.
    ck_health = {"ok": 0, "files": 0, "last_ok": None, "exhausted": False}
    ck_state = {"rounds": int(params.get("_ckpt_start") or 0), "files": 0, "last_sec": None}
    ck_seen = {}
    if state_root is None and not ckpt_urls:
        print("[htr-case] ⚠ ні state_slug, ні ckpt_urls — чекпоінт лише штатним "
              "синком бекенда (а на Vast томів немає взагалі: урвана робота "
              "триматиметься на диску інстансу)", flush=True)

    def checkpoint(tag=""):
        if state_root is None and not ckpt_urls:
            return
        nxt = ck_state["rounds"] + 1
        t = time.time()
        if ckpt_urls:
            n, _ = _ckpt_to_url(work, ck_seen, nxt, ckpt_urls)
        else:
            n = _mirror_archive(work, state_root, ck_seen, nxt)
        # Порожній раунд НЕ палить слот посилання: нема чого лити — нема чого
        # й нумерувати. Інакше на довгому заході слоти вичерпувались тишею.
        if n:
            ck_state["rounds"] = nxt
            ck_state["ok"] = ck_state.get("ok", 0) + 1
            ck_state["last_ok"] = _utc_iso()
        elif ckpt_urls and nxt > len(ckpt_urls):
            ck_state["exhausted"] = True
        ck_state["files"] += n
        # 🔴 Здоров'я чекпоінтів мусить бути ВИДНИМ наглядачу. Досі вичерпані
        # чи протерміновані посилання лишали слід лише в лозі на боксі — у
        # який агенту за контрактом дивитись не треба, — і захід міг годинами
        # їхати без жодної точки відновлення, не знаючи про це.
        ck_health["ok"] = ck_state.get("ok", 0)
        ck_health["files"] = ck_state["files"]
        ck_health["last_ok"] = ck_state.get("last_ok")
        ck_health["exhausted"] = bool(ck_state.get("exhausted"))
        ck_state["last_sec"] = round(time.time() - t, 1)
        if n:
            print("[htr-case] чекпоінт%s → ckpt_%04d.tgz: %d файл(ів) за %.1f с"
                  % (tag, ck_state["rounds"], n, ck_state["last_sec"]), flush=True)

    def checkpointer():
        while not stop.wait(checkpoint_sec):
            try:
                checkpoint()
            except Exception as exc:
                print("[htr-case] ⚠ чекпоінт упав: %r" % (exc,), flush=True)

    ck_thread = threading.Thread(target=checkpointer, daemon=True)
    if state_root is not None or ckpt_urls:
        ck_thread.start()

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # ‼ Без цього кожен шард вважає машину своєю: на Beam-хості з 255 ядрами
    # чотири шарди розгорнули BLAS/OpenMP на 182 ядра сумарно (дашборд, 2026-08-09),
    # тоді як контейнеру замовлено 8. Це не швидкість, а контекст-світчі й тротлінг
    # до квоти. Дефолт рахується від КВОТИ (`_usable_cores`), а не від видимих:
    # до 04.09.2026 тут стояв `os.cpu_count()`, тобто хост, і рятував лише
    # стеля 8 — на боксі з 96 видимими проти 46 квоти виходило 8 потоків там,
    # де бюджет 5.
    if threads_per_shard > 0:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[var] = str(threads_per_shard)
        print("[htr-case] потоків на шард: %d · ядер %d із %s видимих"
              % (threads_per_shard, _usable_cores(), os.cpu_count()), flush=True)

    t0 = time.time()
    rcs = {}
    fleet = {
        "ckpt": ck_health,   # той самий об'єкт — оновлення видно без копіювання
        "schema": 1,
        "case": case_label,
        "case_index": params.get("_case_index") or 1,
        "cases_total": params.get("_cases_total") or 1,
        "phase": "running",
        "n_pages_expected": n_pages,
        "started": _utc_iso(),
        "shards": {},
        "oom_events": 0,
        "n_gpus": n_gpus,
        "quarantine": [],
        "suspects": {},
        "resumed_pages": already,
        "threads_per_shard": threads_per_shard,
        # 🔴 Лише якщо раннер в ассетах прапорець знає: 06.09.2026 бокс-раннер
        # додав `--claim` до копії без нього — усі шарди впали на argparse ще
        # до першої сторінки, і захід пішов у догін з нулем. Старий раннер
        # мовчки їде статичним зрізом, і це видно в лозі.
        "dynamic": bool(params.get("dynamic_pages", True))
        and _runner_supports(code / "htr_case_run.py", "--claim"),
    }
    if params.get("dynamic_pages", True) and not fleet["dynamic"]:
        print("[htr-case] ⚠ раннер в ассетах не знає --claim — статичний зріз "
              "сторінок (перепакуй архів новим раннером)", flush=True)
    try:
        # 🔴🔴 САМОЛІКУВАННЯ ПРИ OOM: звужуємо флот, а не доліковуємо потім.
        #
        # Доти OOM лише РАХУВАВСЯ. Шард, який уперся у VRAM, падав на кожній
        # наступній сторінці до кінця своєї частки — і «лікуванням» ставав
        # догінний прохід, що переганяв усе пропущене ОДНИМ шардом. На справі
        # 2026-08-12 це означало: 70% сторінок у збоях, а потім ті самі 70%
        # серіалізовано — тобто найдорожчий можливий спосіб.
        #
        # Тепер: помітили OOM у кількох шардів — гасимо флот, зменшуємо його на
        # третину й піднімаємо знову. Уже прочитані сторінки лежать на диску й
        # пропускаються самим скриптом, тож ніщо не рахується двічі. Дві спроби
        # звуження, далі працюємо тим, що є, і лишаємо решту догону.
        oom_shrinks = 0
        while True:
            fleet["shards"] = {}
            fleet["oom_events"] = 0
            # 🧲 Клейми з попереднього флоту (переоренда, OOM-звуження) несуть
            # pid-и, які на цій машині можуть належати іншим процесам —
            # перевірка життя збрехала б. Чистий старт: без txt = без клейму.
            _clear_claims(out_dir)
            _clear_drains(out_dir)
            denom = REG_DENOM if regulate else shards
            fleet["denom"] = denom
            for k in range(shards):
                _start_shard(k, denom, base, env, logs_dir, fleet)
                print("[htr-case] шард %d/%d стартував" % (k + 1, shards), flush=True)
            regulator = (_FleetRegulator(fleet, out_dir=out_dir, base=base, env=env,
                                         logs_dir=logs_dir, denom=denom,
                                         ceiling=shards_max, cores=_usable_cores(), t0=t0,
                                         memory=carried, cores_per_shard=reg_cores)
                         if regulate else None)

            rcs = _watch_fleet(fleet, out_dir, shards=shards, stall_sec=stall_sec,
                               restart_max=restart_max, base=base, env=env,
                               logs_dir=logs_dir, t0=t0, regulator=regulator)
            if regulator is not None and fleet_memory is not None:
                fleet_memory.clear()
                fleet_memory.update(regulator.remember())
                fleet_memory["material"] = material
                carried = fleet_memory

            oom = int(fleet.get("oom_events") or 0)
            left = len(_missing_pages(out_dir, [pp.stem for pp in _script_pages(pages_root)]))
            if oom < OOM_SHRINK_TRIGGER or shards <= 1 or oom_shrinks >= OOM_SHRINK_MAX                     or not left:
                if oom >= OOM_SHRINK_TRIGGER and shards <= 1:
                    print("[htr-case] ⚠ OOM на ОДНОМУ шарді — звужувати нікуди; "
                          "справі потрібна карта з більшою VRAM або більший "
                          "vram_gb_per_shard", flush=True)
                break
            oom_shrinks += 1
            smaller = max(1, shards - max(1, shards // 3))
            print("[htr-case] 🔧 OOM ×%d → звужую флот %d → %d шардів "
                  "(лишилось %d сторінок; зроблене не переробляється)"
                  % (oom, shards, smaller, left), flush=True)
            fleet["oom_shrinks"] = oom_shrinks
            fleet["shards_now"] = smaller
            shards = smaller
            _set_phase("running", shards=[])
    finally:
        # Ловить і виняток, і KeyboardInterrupt: усе, що вже прочитано, мусить
        # доїхати на том до того, як процес помре.
        stop.set()
        try:
            checkpoint(" (фінальний)")
        except Exception as exc:
            print("[htr-case] ⚠ фінальний чекпоінт упав: %r" % (exc,), flush=True)
    # ── ворота повноти + догін ──────────────────────────────────────────────
    # 🔴 Тут і тільки тут з'ясовується, чи справа справді зроблена. Усе решта
    # (rc шардів, `failed_shards`, «✓ готово:» у логах) уже показувало успіх
    # на прогоні, де 46 сторінок з'їв CUDA OOM. Єдина правда — файли на диску.
    expected_paths = _script_pages(pages_root)
    if limit:
        expected_paths = expected_paths[:limit]
    expected_stems = [p.stem for p in expected_paths]
    catchup_rounds = 0
    missing = _missing_pages(out_dir, expected_stems)
    while missing and catchup_rounds < catchup_passes:
        catchup_rounds += 1
        fleet["phase"] = "catchup"
        _publish(fleet, t0)
        idx = [i for i, stem in enumerate(expected_stems, 1) if stem in set(missing)]
        print("[htr-case] 🩹 догін №%d: %d сторінок (%s)"
              % (catchup_rounds, len(idx), ", ".join(missing[:8])
                 + ("…" if len(missing) > 8 else "")), flush=True)
        rc = _catchup_pass(base, idx, env, logs_dir, catchup_rounds)
        print("[htr-case] 🩹 догін №%d rc=%s" % (catchup_rounds, rc), flush=True)
        before, missing = missing, _missing_pages(out_dir, expected_stems)
        if len(missing) >= len(before):
            # Догін нічого не дав — це вже не OOM, а отруйні кадри.
            print("[htr-case] 🩹 догін не зрушив (%d → %d) — далі немає сенсу"
                  % (len(before), len(missing)), flush=True)
            break

    wall = time.time() - t0
    bad = [k for k, rc in rcs.items() if rc != 0]

    # ── замір ───────────────────────────────────────────────────────────────
    meta_pages = {}
    whole = out_dir / "_htr_meta.json"
    if whole.is_file():
        try:
            meta_pages.update(json.loads(whole.read_text(encoding="utf-8")).get("pages", {}))
        except Exception:
            pass
    for part in sorted(out_dir.glob("_htr_meta.part*.json")):
        try:
            meta_pages.update(json.loads(part.read_text(encoding="utf-8")).get("pages", {}))
        except Exception:
            pass

    secs = [v.get("sec") for v in meta_pages.values() if isinstance(v.get("sec"), (int, float))]
    lines = [v.get("lines") for v in meta_pages.values() if isinstance(v.get("lines"), int)]
    chars = [v.get("chars") for v in meta_pages.values() if isinstance(v.get("chars"), int)]
    enhanced = sum(1 for v in meta_pages.values() if v.get("enhanced") not in (None, "", "none"))

    done_now = max(0, len(meta_pages) - already)
    summary = {
        "n_pages_input": n_pages,
        "n_pages_total": len(meta_pages),
        "n_pages_this_run": done_now,
        "resumed_pages": already,
        "shards": shards,
        "gpu": _gpu_name(),
        "cpu_count": os.cpu_count(),
        "gpu_lock": gpu_lock,
        "n_gpus": n_gpus,
        "shards_by_gpu": fleet.get("shards_by_gpu") or {},
        "seg_cache": seg_cache,
        "enhance": enhance,
        "model": model_path.name,
        "voices": [Path(v).name for v in voice_paths],
        "wall_sec": round(wall, 1),
        # ‼ Головне число заміру: wall / зроблені В ЦЬОМУ запуску сторінки. Саме
        # воно, а не сума посторінкових `sec`: у шардах вони йдуть паралельно, і
        # їхня сума більша за wall рівно в стільки разів, скільки паралелізму
        # реально вийшло — цю величину видно нижче як `parallel_gain`.
        "sec_per_page": round(wall / done_now, 2) if done_now else None,
        "pages_per_hour": round(3600.0 * done_now / wall) if wall > 0 and done_now else None,
        "sum_page_sec": round(sum(secs), 1) if secs else None,
        "parallel_gain": round(sum(secs) / wall, 2) if secs and wall > 0 else None,
        "lines_total": sum(lines) if lines else 0,
        "chars_total": sum(chars) if chars else 0,
        "lines_per_page_median": (sorted(lines)[len(lines) // 2] if lines else None),
        "enhanced_pages": enhanced,
        "shard_rc": rcs,
        "failed_shards": bad,
        # ── повнота ─────────────────────────────────────────────────────────
        # Ці п'ять полів і є ворота: `complete` рахується з ФАЙЛІВ на диску, а
        # не з лічильників конвеєра, і саме його дивиться `fetch`.
        "n_pages_expected": len(expected_stems),
        "n_pages_txt": len(list(out_dir.glob("*.txt"))),
        "missing_pages": missing,
        "quarantined_pages": sorted(_quarantined_stems(out_dir)),
        "complete": not missing,
        "catchup_rounds": catchup_rounds,
        "scripts_sha256": scripts_sha,
        "oom_events": fleet.get("oom_events", 0),
        "oom_events_total": fleet.get("oom_events_total", 0),
        "oom_kills": fleet.get("oom_kills", 0),
        # 🎛 що флот виміряв сам: коліно, темп на кожному рівні, пам'ять шарда.
        # Це калібрування на наступні плани замість формули з площі кадру.
        "regulator": fleet.get("regulator"),
        "shards_max": shards_max,
        "shard_restarts": {k: v.get("restarts", 0) for k, v in fleet["shards"].items()
                           if v.get("restarts")},
        "stall_sec": stall_sec,
        "checkpoint_sec": checkpoint_sec,
        "checkpoint_rounds": ck_state["rounds"],
        "checkpoint_files": ck_state["files"],
        "checkpoint_state": str(state_root) if state_root else None,
        "checkpoint_last_sec": ck_state["last_sec"],
        "versions": _versions(),
        "startup_sec": round(t0 - t_start, 1),
        "finished": _utc_iso(),
    }

    if make_bundle:
        with tarfile.open(work / "htr_case_bundle.tgz", "w:gz") as tf:
            for d in sorted(work.glob("out*")):
                if d.is_dir():
                    tf.add(d, arcname=d.name)
            tf.add(logs_dir, arcname="logs")
    (work / "htr_case_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        checkpoint(" (підсумок)")
    except Exception:
        pass

    # 🔴 `done` означає «черга скінчилась», а не «ця справа скінчилась».
    # Інакше наглядач забере результат після ПЕРШОЇ справи й погасить бокс,
    # на якому ще мали рахуватись решта.
    final = bool(params.get("_final_case", True))
    # 🔴 `_final_case` мусить керувати ОБОМА гілками. Гілка `done` його
    # перевіряла, а `failed` — ні: неповна справа №1 із трьох публікувала
    # `phase="failed"`, наглядач трактує `failed` як термінал, забирає
    # результат і гасить бокс ПОСЕРЕД черги — при тому, що решта справ ще
    # мали рахуватись і вже оплачені.
    if not summary["complete"]:
        fleet["phase"] = "failed" if final else "running"
    else:
        fleet["phase"] = "done" if final else "running"
    fleet["complete"] = summary["complete"]
    fleet["missing"] = missing[:20]
    fleet["missing_count"] = len(missing)
    fleet["catchup_rounds"] = catchup_rounds
    _publish(fleet, t0)

    # 🔴 Сентинел, а не фраза. «✓ готово:» друкує КОЖЕН шард у свій лог, і
    # саме на ньому конвеєр вирішив, що справа завершена, забравши 203 з 323
    # сторінок. Цей рядок з'являється рівно один раз на прогін.
    if final:
        print("%s%s" % (DONE_SENTINEL, json.dumps(
        {k: summary[k] for k in ("n_pages_expected", "n_pages_txt", "n_pages_this_run",
                                 "complete", "missing_pages", "catchup_rounds",
                                 "oom_events", "wall_sec", "sec_per_page",
                                 "pages_per_hour", "parallel_gain", "failed_shards")},
            ensure_ascii=False)), flush=True)
    summary["case"] = case_label

    if bad:
        raise RuntimeError("шарди впали: %s — див. logs/ у виході" % bad)
    if missing and not allow_incomplete:
        raise RuntimeError(
            "НЕПОВНО: %d із %d сторінок без тексту після %d догонів — %s%s. "
            "Це саме той випадок, коли раніше job звітував успіх "
            "(-p allow_incomplete=true щоб прийняти як є)."
            % (len(missing), len(expected_stems), catchup_rounds,
               ", ".join(missing[:10]), "…" if len(missing) > 10 else ""))
    return summary


if __name__ == "__main__":
    main({})
