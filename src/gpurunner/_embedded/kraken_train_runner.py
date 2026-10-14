"""Ketos fine-tune (McCATMuS → дистиляційна база CHURRO) на GPU.

Вхід /kaggle/input/**: train*.arrow + val*.arrow (прекомпільовані ketos compile
--force-type baseline) + базова модель *.mlmodel (окремий датасет/тека).
Вихід /kaggle/working/:
  kraken_churro_best.mlmodel  — конвертована найкраща модель (load_any-сумісна)
  best_*.safetensors          — сирі ваги найкращої епохи
  ktrain.log                  — повний лог ketos
  ktrain_summary.json         — крива val_accuracy по епохах + best

⚠ kraken 7: train БЕРЕ лише -f binary (path-режим у train зламаний), тому
arrow-файли компілюються локально перед сабмітом. Конвертація у .mlmodel —
kraken.models.load_models → TorchVGSLModel.save_model (deprecated, але єдиний
шлях до coreml, який їсть load_any у .venv_kraken 7.0.2).
⚠ БЕЗ `from __future__ import annotations` — код інжектиться після PARAMS.
"""
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
WORK = Path("/tmp/ktrain")


def _ensure_kraken(version: str = "") -> None:
    """Lightning не ставить requirements() — Studio-оточення позичене, тож
    бутстрапимось самі; на Kaggle це no-op (пакет уже стоїть)."""
    try:
        import kraken  # noqa: F401
    except Exception:
        print("[ktrain] pip install kraken…", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             f"kraken=={version}" if version else "kraken>=5.0"],
            check=True,
        )


def _drop_conflicting_transformers() -> None:
    """kraken 7.1 тримає `safetensors~=0.7.0`, а образ Kaggle несе `transformers`,
    якому треба ≥0.8. Сам kraken `transformers` не вживає, але torchmetrics
    імпортує його, щойно пакет є в середовищі, — і `ketos train` падав на
    імпорті Lightning, ще до першої епохи (job 1a20fbe7, 2026-10-03). Контейнер
    одноразовий, тож пакет просто прибираємо."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            version("transformers")
        except PackageNotFoundError:
            return
        st = version("safetensors")
        if tuple(int(x) for x in st.split(".")[:2]) >= (0, 8):
            return
        print(f"[ktrain] safetensors {st} < 0.8 при встановленому transformers — "
              f"прибираю transformers (kraken його не вживає)", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                        "transformers"], check=False)
    except Exception as exc:
        print(f"[ktrain] перевірку transformers пропущено: {type(exc).__name__}: {exc}",
              flush=True)


def _xeval_index():
    """Індекс пакета заміру: готовий `xeval.index` у входах або з розпакованого архіву."""
    idx_files = _find("xeval.index")
    if idx_files:
        return idx_files[0]
    # Kaggle-датасет не несе підтек (gpurunner вимагає архів), тож пакет
    # приходить архівом — розпаковуємо самі. 🔴 Маска із зірочкою спереду:
    # пакет MeGen зветься `_xeval_xix.tgz`, і `xeval*.tgz` його не бачив —
    # замір мовчки пропускався, а трен віддавав ваги без жодного xeval_epNN.tsv.
    for tgz in _find("*xeval*.tgz"):
        dst = WORK / "xeval_in"
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "xzf", str(tgz), "-C", str(dst)], check=False)
        idx_files = sorted(dst.rglob("xeval.index"))
        if idx_files:
            return idx_files[0]
    return None


def _xeval_targets(base=None) -> list:
    """[(ваги, ім'я виходу)]: база → `xeval_base.tsv`, епохи → `xeval_epNN.tsv`.

    Базу читає той самий бокс тим самим батчем: при дотренуванні це бойова модель,
    і порівнювати епохи треба саме з нею. Окрема оренда під `eval_only` з базою
    коштувала б ще один старт машини й розпакування пакета.
    """
    out = []
    if base is not None:
        out.append((base, "xeval_base.tsv"))
    for mp in sorted(KAGGLE_WORKING.glob("epoch_*.mlmodel")) + \
            sorted(KAGGLE_WORKING.glob("epoch_*.safetensors")):
        m = re.search(r"epoch_(\d+)", mp.name)
        out.append((mp, f"xeval_ep{int(m.group(1)):02d}.tsv"))
    return out


#: Воркер заміру: окремий процес зі своєю моделлю читає свою частину сторінок.
#: Окремим скриптом, а не `multiprocessing`: kraken усередині `predict` заводить
#: дочірні процеси (демонічним процесам пулу це заборонено), а `job.py` без
#: `__main__`-запобіжника під `spawn` перезапустив би весь трен у кожному воркері.
_XEVAL_WORKER = r'''
import json
import sys
from pathlib import Path

import torch
import torch._dynamo
torch._dynamo.config.disable = True   # компіляція під кожну ширину полотна — дорожча за читання
from kraken.configs import RecognitionInferenceConfig
from kraken.containers import BBoxLine, Segmentation
from kraken.tasks import RecognitionTaskModel
from PIL import Image


def main():
    torch.set_num_threads(1)
    shard, model_path, batch, dst = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    if torch.cuda.is_available() and procs > 1:
        # Своя частка пам'яті карти: без неї нестача пам'яті залежала від того, що в цю мить
        # читають сусіди, і повтор меншим батчем (інше прочитане) падав на різні сторінки
        # для різних ваг (22 з 969 сторінок, 10-12). З часткою він залежить лише від сторінки.
        torch.cuda.set_per_process_memory_fraction(min(0.95, 0.92 / procs))
    pages = json.loads(Path(shard).read_text(encoding="utf-8"))
    cuda = torch.cuda.is_available()

    def config(b):
        return RecognitionInferenceConfig(accelerator="gpu" if cuda else "cpu",
                                          device=[0] if cuda else "auto", batch_size=b, num_workers=0,
                                      # 🔴 рядки вирізати в цьому процесі: пул kraken (дефолт 2)
                                      # пересилає ВСЕ полотно сторінки на кожен рядок і віддає рядки
                                      # у довільному порядку — замір упирався в CPU (~30 рядків/с
                                      # при 0% A100), а склад батча, отже й прочитане, гуляв
                                      num_line_workers=0)

    full, small = config(batch), config(min(batch, 4))
    model = RecognitionTaskModel.load_model(model_path)
    out = []
    for st, pg, lines in pages:
        ims = [Image.open(p).convert("L") for p, _ in lines]
        width = max(im.width for im in ims)
        # Білий проміжок між кропами: kraken вирізає рамку з відступом і без
        # нього захоплював краї сусідніх рядків (3.4% розбіжності символів
        # проти читання кропа окремо, звірено на holdout 10-03).
        gap = 64
        canvas = Image.new("L", (width + 2 * gap, sum(im.height + gap for im in ims) + gap), 255)
        boxes, y = [], gap
        for (_, k), im in zip(lines, ims):
            canvas.paste(im, (gap, y))
            boxes.append(BBoxLine(id=f"l{k}", bbox=(gap, y, gap + im.width, y + im.height)))
            y += im.height + gap
        seg = Segmentation(type="bbox", imagename=f"{st}_{pg}", text_direction="horizontal-lr",
                           script_detection=False, lines=boxes)
        preds = []
        for cfg in (full, small):
            try:
                preds = list(model.predict(im=canvas, segmentation=seg, config=cfg))
                break
            except Exception as exc:
                if cuda:
                    torch.cuda.empty_cache()
                if cfg is full and "out of memory" in str(exc):
                    # менший батч читає трохи інакше (паддінг до спільної ширини) — тому в лог
                    print(f"[xeval] OOM-повтор {st}/{pg} батчем {min(batch, 4)}", flush=True)
                    continue
                print(f"[xeval] {Path(model_path).name} {st}/{pg}: {type(exc).__name__}: {exc}", flush=True)
                preds = []
                break
        # kraken віддає рядки в порядку рамок; інша довжина = збій сторінки,
        # тоді краще порожні рядки, ніж зсунутий на рядок текст
        if len(preds) != len(lines):
            preds = [None] * len(lines)
        for (_, k), r in zip(lines, preds):
            out.append(f"{st}\t{pg}\t{k}\t{(r.prediction if r is not None else '').strip()}")
    Path(dst).write_text("".join(r + "\n" for r in out), encoding="utf-8")


# kraken у `predict` заводить дочірні процеси; під spawn вони імпортують цей файл
# заново (як `__mp_main__`), і без запобіжника кожен перезапускав би читання.
# Запобіжник записано через змінну: `Job._runner_source` відрізає раннер по
# першому запобіжнику на `__name__` і відрізав би тут, посеред рядка.
_RUN = __name__ == "__main__"
if _RUN:
    main()
'''


def _xeval_procs(batch: int) -> int:
    """Скільки процесів читає пакет: ядра мінус одне, пам'ять карти, не більше 8.

    Без пулу вирізання рядків (`num_line_workers=0`) замір упирається в карту: 7
    процесів на RTX 3090 — 100% GPU, 75 тис. кропів за 536 с (10-12). Пам'яті процес
    на широких рядках бере понад 3 ГБ на батчі 16 — запас 2.5 + 0.15·батч ГБ.
    """
    import os
    n = min(8, max(1, (os.cpu_count() or 2) - 1))
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            n = min(n, max(1, int(free / ((2.5 + 0.15 * batch) * 2 ** 30))))
    except Exception:
        pass
    return n


def _xeval_shards(pages: dict, k: int) -> list:
    """Сторінки → до k частин: найбільші першими й по колу, щоб процеси закінчували разом."""
    order = sorted(pages.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    shards = [[] for _ in range(max(1, k))]
    for i, ((st, pg), lines) in enumerate(order):
        shards[i % len(shards)].append([st, pg, [[str(p), n] for p, n in lines]])
    return [s for s in shards if s]


def _xeval_epochs(batch: int = 1, base=None) -> list:
    """Замір кожної епохи (і бази, якщо дана) на пакеті заміру → xeval_epNN.tsv.

    Епоху обирає замір на holdout, а не val_accuracy. Досі його рахували вдома:
    ~12 хв на епоху Дяка на GTX 1650 (6 тис. рядків), тоді як ця карта вільна
    одразу після трену. Кропи сторінки складаються в одне полотно з рамками
    рядків — так модель вантажиться раз, а сторінка читається одним викликом.
    Сторінки розкладено по кількох процесах (`_XEVAL_WORKER`): одним процесом
    пакет у 75 тис. кропів ішов ~45 хв на кожні ваги.
    🔴 Батч міняє прочитане: він доповнює рядки до спільної ширини, і CTC від
    цього читає трохи інакше — 3.4% розбіжності символів проти читання кропа
    окремо; з батчем 1 тотожно 255/255 (звірено 10-03). Порівнювати лише голоси,
    прочитані тим самим батчем.
    Потрібен kraken ≥ 7.1 (`kraken.tasks`); на старішому замір пропускається.
    """
    idx = _xeval_index()
    if idx is None:
        return []
    try:
        import kraken.tasks  # noqa: F401
    except Exception as exc:
        print(f"[xeval] пропущено: {type(exc).__name__}: {exc}", flush=True)
        return []
    pages: dict = {}
    for raw in idx.read_text(encoding="utf-8").splitlines():
        parts = raw.split("\t")
        if len(parts) == 4:
            pages.setdefault((parts[1], parts[2]), []).append((idx.parent / parts[0], int(parts[3])))
    shards = _xeval_shards(pages, _xeval_procs(batch))
    wdir = WORK / "xeval_run"
    wdir.mkdir(parents=True, exist_ok=True)
    worker = wdir / "worker.py"
    worker.write_text(_XEVAL_WORKER, encoding="utf-8")
    shard_files = []
    for i, s in enumerate(shards):
        f = wdir / f"shard{i}.json"
        f.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
        shard_files.append(f)
    print(f"[xeval] {sum(len(v) for v in pages.values())} кропів, {len(pages)} сторінок → "
          f"{len(shards)} процесів, батч {batch}", flush=True)
    import os
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    written = []
    for mp, name in _xeval_targets(base):
        t0 = time.time()
        parts_out = [wdir / f"{name}.part{i}" for i in range(len(shards))]
        for p in parts_out:
            p.unlink(missing_ok=True)
        procs = [subprocess.Popen([sys.executable, str(worker), str(sf), str(mp), str(batch), str(po),
                                   str(len(shards))], env=env)
                 for sf, po in zip(shard_files, parts_out)]
        rcs = [p.wait() for p in procs]
        failed = [i for i, rc in enumerate(rcs) if rc != 0]
        if failed:
            print(f"[xeval] {mp.name}: процеси {failed} впали (rc {[rcs[i] for i in failed]}) — "
                  f"їхніх сторінок у виході немає", flush=True)
        if len(failed) == len(procs):
            continue
        rows = []
        for po in parts_out:
            if po.is_file():
                rows += [r for r in po.read_text(encoding="utf-8").splitlines() if r]
        rows.sort(key=lambda r: (r.split("\t")[0], r.split("\t")[1], int(r.split("\t")[2])))
        dst = KAGGLE_WORKING / name
        dst.write_text("\n".join(rows) + "\n", encoding="utf-8")
        written.append(dst.name)
        print(f"[xeval] {mp.name}: {len(rows)} рядків за {time.time() - t0:.0f}s → {dst.name}",
              flush=True)
    return written


def _find(pattern: str) -> list:
    return sorted(KAGGLE_INPUT.rglob(pattern))


def _best_ckpt(ckpt_dir: Path):
    """Найкращий checkpoint_NN-0.xxxx.ckpt за val_accuracy з імені → (acc, epoch, path)."""
    scored = []
    for p in ckpt_dir.glob("checkpoint_*.ckpt"):
        m = re.match(r"checkpoint_(\d+)-([\d.]+)\.ckpt$", p.name)
        if m:
            scored.append((float(m.group(2)), int(m.group(1)), p))
    scored.sort()
    return scored[-1] if scored else None


def _run_ketos(args: list, log_path: Path, deadline: float, ckpt_dir: Path) -> tuple:
    """Запустити ketos, стрімлячи stdout у log_path. → (rc, зупинено_за_годинником).

    ``deadline`` — АБСОЛЮТНИЙ час (time.time()), спільний для всіх спроб: після
    відкоту з DDP на одну карту друга спроба отримує залишок бюджету, а не
    повний ліміт заново. Раніше тут переприсвоювався t0, тож дві спроби могли
    сумарно з'їсти подвійний ліміт і бути вбитими по ліміту сесії — рівно те,
    від чого guard і рятує.
    """
    proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            errors="replace")
    stopped = []

    def _watchdog():
        # 🔴 Окремий тред, а не перевірка всередині `for line in proc.stdout`.
        # Той цикл блокується на читанні: якщо ketos замовк (зависання, довга
        # епоха без жодного рядка), guard не спрацював би саме тоді, коли він
        # найпотрібніший.
        while proc.poll() is None:
            left = deadline - time.time()
            if left <= 0:
                stopped.append(True)
                print("[ktrain] ⏱ wall-limit вичерпано — зупиняю ketos, "
                      "конвертую найкращу епоху", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=120)
                except Exception:
                    proc.kill()
                return
            time.sleep(min(30.0, max(1.0, left)))

    if deadline:
        threading.Thread(target=_watchdog, daemon=True).start()

    last_sync = 0.0
    with log_path.open("w", encoding="utf-8") as logf:
        for line in proc.stdout:
            logf.write(line)
            stripped = line.strip()
            if stripped:
                print(f"[ketos] {stripped}", flush=True)
            # Проміжний best у вихідну теку. Без цього ваги живуть ЛИШЕ на
            # віддаленій машині до кінця тренування: зупинити job на плато
            # (або втратити його на preemption) = втратити всі витрачені години.
            if time.time() - last_sync > 120:
                last_sync = time.time()
                try:
                    # ⚠ `best_*.safetensors` під час трену НЕ ІСНУЄ (ketos пише
                    # його лише в кінці) — снапшот мусить брати чекпойнти,
                    # інакше він мовчки не робить нічого, як було до 2026-07-29.
                    snap = sorted(ckpt_dir.glob("best_*.safetensors"))
                    if snap:
                        shutil.copyfile(snap[-1],
                                        KAGGLE_WORKING / "best_inprogress.safetensors")
                        logf.flush()
                    else:
                        ck = _best_ckpt(ckpt_dir)
                        if ck:
                            shutil.copyfile(ck[2],
                                            KAGGLE_WORKING / "best_inprogress.ckpt")
                            logf.flush()
                except Exception as exc:  # проміжний зліпок не варт падіння трену
                    print(f"[ktrain] snapshot skipped: {type(exc).__name__}", flush=True)
        rc = proc.wait()
    if stopped:
        rc = 0                      # зупинка за планом, не збій
    return rc, bool(stopped)


def _bind_roots(params: dict[str, Any]) -> None:
    """Modal не емулює /kaggle/{input,working}: вхід — том (/vol, за потреби
    його підтека `modal_subdir`, щоб кілька заходів жили в одному томі), вихід —
    `params["output_root"]`. Без прив'язки трен відпрацював би, а ваги лягли б
    туди, звідки Modal нічого не забирає."""
    global KAGGLE_INPUT, KAGGLE_WORKING
    out = params.get("output_root")
    if out:
        KAGGLE_WORKING = Path(out)
    if not KAGGLE_INPUT.is_dir() and Path("/vol").is_dir():
        sub = str(params.get("modal_subdir") or "").strip("/")
        KAGGLE_INPUT = Path("/vol") / sub if sub else Path("/vol")
    print(f"[ktrain][roots] вхід={KAGGLE_INPUT} вихід={KAGGLE_WORKING}", flush=True)


def main(params: dict[str, Any]) -> None:
    _bind_roots(params)
    _ensure_kraken(str(params.get("kraken_version") or "").strip())
    _drop_conflicting_transformers()
    import torch
    if params.get("eval_only"):
        # Лише замір: ваги епох (epoch_NN.mlmodel/.safetensors) приходять у
        # model_dataset, кропи — в eval_dataset. Трену немає.
        KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)
        for p in _find("epoch_*.mlmodel") + _find("epoch_*.safetensors"):
            dst = KAGGLE_WORKING / p.name
            if not dst.exists():
                shutil.copyfile(p, dst)
        # ваги з іменем не `epoch_*` — база дотренування: міряється тим самим
        # прогоном у `xeval_base.tsv`, щоб порівнювати епохи з нею одним батчем
        bases = [p for p in _find("*.mlmodel") + _find("*.safetensors")
                 if not p.name.startswith(("epoch_", "best_"))]
        if len(bases) > 1:
            print(f"[xeval] базою беру {bases[0].name}; решту ({[b.name for b in bases[1:]]}) — ні",
                  flush=True)
        out = _xeval_epochs(int(params.get("xeval_batch", 1)), base=bases[0] if bases else None)
        (KAGGLE_WORKING / "ktrain_summary.json").write_text(
            json.dumps({"eval_only": True, "xeval": out}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"[xeval] eval_only: {len(out)} епох", flush=True)
        return

    train_arrows = [p for p in _find("*.arrow") if p.name.startswith("train")]
    val_arrows = [p for p in _find("*.arrow") if p.name.startswith("val")]
    # База kraken 7.1 буває лише `.safetensors` (PP-OCRv6 у coreml не
    # зберігається взагалі), тож шукаємо обидва формати; `.mlmodel` має
    # перевагу, щоб старі датасети з двома файлами поводились як раніше.
    models = _find("*.mlmodel") or _find("*.safetensors")
    if not train_arrows:
        raise RuntimeError(f"no train*.arrow under {KAGGLE_INPUT}")
    if not val_arrows:
        raise RuntimeError(f"no val*.arrow under {KAGGLE_INPUT}")
    # Базова модель НЕобов'язкова: `spec` дозволяє тренувати з нуля. Для чужого
    # алфавіту це часто чистіше за fine-tune — латинська база тягне за собою
    # ваги для символів, яких у цільовому письмі нема взагалі.
    base_model = models[0] if models else None
    spec = str(params.get("spec", "")).strip()
    if base_model is None and not spec:
        raise RuntimeError(
            f"no *.mlmodel under {KAGGLE_INPUT} — додай model-датасет "
            f"або задай -p spec='<VGSL>' для тренування з нуля")
    # 🔴 ketos УМІЄ кілька карт — попередній коментар тут стверджував протилежне
    # і коштував нам половини заліза на кожному трені. `-d` не має валідатора
    # (звичайний рядок), а `kraken.ketos.util.to_ptl_device` розбирає його через
    # `device.split(",")` і віддає Lightning список `devices=[0, 1]`. Тобто
    # `-d cuda:0,cuda:1` проходить наскрізь і вмикає DDP. Ketos ми запускаємо
    # окремим ПРОЦЕСОМ (не в ноутбуці), тож це звичайний ddp, а не ddp_notebook.
    _ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    _want = str(params.get("devices", "auto")).strip()
    if not torch.cuda.is_available():
        device = "cpu"
    elif _want and _want != "auto":
        device = _want                      # явне «cuda:0» щоб вимкнути DDP
    else:
        device = ",".join(f"cuda:{i}" for i in range(max(1, _ngpu)))
    try:
        import os as _os
        _ncpu = len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") \
            else _os.cpu_count()
        print(f"[ktrain][env] CPU={_ncpu} workers={params['workers']} threads=2 "
              f"batch={params['batch']} | GPU видно={_ngpu} → -d {device}",
              flush=True)
        if _ngpu > 1 and "," in device:
            print(f"[ktrain][env] DDP на {_ngpu} картах: ефективний батч "
                  f"{params['batch'] * _ngpu}, кроків за епоху вдвічі менше",
                  flush=True)
        if params["workers"] < max(1, (_ncpu or 2) - 1):
            print(f"[ktrain][env] ⚠ workers={params['workers']} < CPU-1 — "
                  f"імовірне голодування dataloader'а", flush=True)
    except Exception as exc:
        print(f"[ktrain][env] діагностику пропущено: {type(exc).__name__}: {exc}",
              flush=True)
    print(f"[ktrain] train={[p.name for p in train_arrows]} "
          f"val={[p.name for p in val_arrows]} "
          f"{'load=' + base_model.name if base_model else 'FROM SCRATCH: ' + spec[:60]} "
          f"dev={device}", flush=True)

    WORK.mkdir(parents=True, exist_ok=True)
    ckpt_dir = WORK / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    val_list = WORK / "val_files.txt"
    val_list.write_text("\n".join(str(p) for p in val_arrows) + "\n",
                        encoding="utf-8")

    # 🔴 ketos запускається через ФАЙЛ-лаунчер, а не `python -c`. Lightning для
    # DDP піднімає дочірні ранги, ПОВТОРЮЮЧИ команду батька з `sys.argv`, і в
    # `-c`-режимі argv[0] дорівнює рядку "-c" — дитина намагається відкрити файл
    # `/kaggle/working/-c` і падає з кодом 2, тягнучи за собою весь трен:
    #   «/usr/bin/python3: can't open file '/kaggle/working/-c'»
    #   «[rank: 1] Child process terminated with code 2»
    # З реальним файлом argv[0] — валідний шлях, і перезапуск рангів працює.
    launcher = WORK / "_ketos_entry.py"
    launcher.write_text("from kraken.ketos import cli\ncli()\n", encoding="utf-8")
    args = [
        sys.executable, str(launcher),
        "-d", device, "--workers", str(params["workers"]), "--threads", "2",
        "--precision", params["precision"],
        "train", "-f", "binary", "-u", "NFD",
    ]
    if base_model:
        args += ["--load", str(base_model), "--resize", params["resize"]]
    else:
        args += ["-s", spec]
        if params.get("resize") not in (None, "", "union"):
            print(f"[ktrain] ⚠ resize={params['resize']} не діє при тренуванні "
                  f"з нуля (-s spec) — параметр стосується лише --load", flush=True)
    args += [
        "-B", str(params["batch"]), "-r", str(params["lr"]),
        "--lag", str(params["lag"]), "--min-epochs", str(params["min_epochs"]),
        # без дубльованого дефолту: KrakenTrainJob.validate_params завжди його
        # передає, а два джерела істини вже розійшлись були (0.001 vs 0.002)
        "--min-delta", str(params["min_delta"]),
        "-N", str(params["epochs"]),
        "-e", str(val_list),
        "-o", str(ckpt_dir),
    ] + [str(p) for p in train_arrows]

    # 🔴 WALL-CLOCK GUARD. `-N` у ketos — НЕ ліміт епох, зупиняє лише `--lag`;
    # з дефолтним lag=10 трен на 96k рядків ішов 27 епох і 11 год, упритул до
    # 12-годинного ліміту сесії Kaggle. А обрив по ліміту вбиває кернел ПОСЕРЕД
    # роботи: конвертації в .mlmodel не буде, і чи вціліє /kaggle/working — не
    # гарантовано. Тому зупиняємось самі, лишаючи запас на конвертацію й запис.
    # Дедлайн абсолютний і СПІЛЬНИЙ для обох спроб (див. _run_ketos).
    wall_limit = float(params.get("wall_limit_h", 0)) * 3600
    t_start = time.time()
    deadline = (t_start + wall_limit) if wall_limit > 0 else 0.0
    if deadline:
        print(f"[ktrain] wall-limit {wall_limit / 3600:.1f} год", flush=True)
    log_path = KAGGLE_WORKING / "ktrain.log"
    rc, _stopped = _run_ketos(args, log_path, deadline, ckpt_dir)

    # 🔴 Відкіт із DDP на одну карту. Багатокартковий режим тут уперше, і якщо
    # він не підніметься (Lightning не зібрав процеси, NCCL не стартував), збій
    # станеться НА СТАРТІ — до першого чекпойнта. Тоді дешевше перезапуститись
    # самим, ніж віддати слот Kaggle під traceback: слот однаково вже наш, а
    # трен на одній карті лишається робочим, просто вдвічі довшим.
    if rc != 0 and "," in device and not list(ckpt_dir.glob("checkpoint_*.ckpt")):
        tail = "".join(log_path.read_text(encoding="utf-8").splitlines(True)[-15:])
        print(f"[ktrain] ⚠ DDP не піднявся — перезапуск на cuda:0\n{tail}", flush=True)
        # Лог невдалої спроби ЗБЕРІГАЄМО: друга спроба відкриває ktrain.log на
        # запис і без цього затерла б саме ту діагностику, заради якої фолбек
        # і логується («чому не піднявся DDP» лишалось би тільки в stdout).
        try:
            shutil.copyfile(log_path, KAGGLE_WORKING / "ktrain_ddp_attempt.log")
        except Exception as exc:
            print(f"[ktrain] лог спроби не збережено: {type(exc).__name__}", flush=True)
        device = "cuda:0"
        args[args.index("-d") + 1] = device
        rc, _stopped = _run_ketos(args, log_path, deadline, ckpt_dir)

    if rc != 0:
        tail = "".join(log_path.read_text(encoding="utf-8").splitlines(True)[-40:])
        raise RuntimeError(f"ketos train exited {rc}; log tail:\n{tail}")

    # checkpoint_{epoch:02d}-{val_accuracy:.4f}.ckpt → крива точності
    history = []
    for p in sorted(ckpt_dir.glob("checkpoint_*.ckpt")):
        m = re.match(r"checkpoint_(\d+)-([\d.]+)\.ckpt$", p.name)
        if m:
            history.append({"epoch": int(m.group(1)),
                            "val_accuracy": float(m.group(2))})
    # 🔴 `best_*.safetensors` ketos пише ЛИШЕ при нормальному завершенні трену.
    # При перериванні (наш wall-limit, preemption, вбивство по ліміту сесії)
    # його НЕМА — є тільки `checkpoint_NN-0.7171.ckpt`, які пишуться щоепохи.
    # Через це двічі згоріли ваги: раннер шукав файл, якого під час трену не
    # існує, і падав із «no best_*.safetensors», хоча 23 епохи лежали поруч.
    # Тепер fallback на найкращий чекпойнт за val_accuracy з імені.
    bests = sorted(ckpt_dir.glob("best_*.safetensors"))
    if bests:
        best = bests[-1]
    else:
        scored = _best_ckpt(ckpt_dir)
        if scored is None:
            raise RuntimeError(
                f"ні best_*.safetensors, ні checkpoint_*.ckpt у {ckpt_dir} "
                f"(вміст: {[p.name for p in ckpt_dir.iterdir()]})")
        best = scored[2]
        print(f"[ktrain] best_*.safetensors нема (трен перервано) — беру "
              f"найкращий чекпойнт {best.name} (val_acc={scored[0]:.4f})",
              flush=True)
    shutil.copyfile(best, KAGGLE_WORKING / best.name)

    # 🔴 Дві РІЗНІ гілки конвертації, бо fallback вище віддає інший формат.
    # `km.load_models()` перебирає лоадери з entry-point `kraken.loaders`
    # (safetensors, coreml) — для Lightning-чекпойнта `.ckpt` лоадера НЕМА, і
    # він падає «No loader found». Тобто без цієї гілки спрацьований wall-limit
    # лишав нас із чекпойнтом, який нічим відкрити: ваги ніби є, моделі нема.
    out_model = KAGGLE_WORKING / "kraken_churro_best.mlmodel"
    try:
        if best.suffix == ".ckpt":
            from kraken.train import VGSLRecognitionModel
            m = VGSLRecognitionModel.load_from_checkpoint(str(best),
                                                          weights_only=False)
            m.net.save_model(str(out_model))     # net = TorchVGSLModel
        else:
            from kraken import models as km
            km.load_models(str(best))[0].save_model(str(out_model))
        print(f"[ktrain] converted {best.name} → {out_model.name}", flush=True)
    except Exception as exc:
        # Не падати: сам файл ваг уже скопійований у working вище, тож його
        # можна забрати `fetch` і сконвертувати локально. Падіння тут коштувало
        # б усього трену через останній крок.
        print(f"[ktrain] ⚠ конвертація {best.name} не вдалась "
              f"({type(exc).__name__}: {exc}) — ваги лишаються як {best.name}",
              flush=True)

    # 🔴 ВАГИ КОЖНОЇ ЕПОХИ — щоб епоху обирав HOLDOUT, а не `val_accuracy`.
    # Ketos і так пише `checkpoint_NN-<acc>.ckpt` щоепохи, але раннер віз додому
    # лише найкращу за val — тобто вибір робився метрикою, яку ми самі визнали
    # ненадійною (на Писарі локальний відбір двічі дав +1.4 і +2.3 пп recall
    # проти того, що віддав би трен; у Дяка val — 143 рядки, і 46 символів
    # корпусу в ньому не трапляються взагалі).
    # Конвертуємо у `.mlmodel` (≈16 МБ), а не веземо `.ckpt` (≈49 МБ втричі
    # більше й потребує kraken.train для відкриття).
    epochs_out = []
    if params.get("all_epochs", True):
        for p in sorted(ckpt_dir.glob("checkpoint_*.ckpt")):
            m = re.match(r"checkpoint_(\d+)-([\d.]+)\.ckpt$", p.name)
            if not m:
                continue
            dst = KAGGLE_WORKING / f"epoch_{int(m.group(1)):02d}.mlmodel"
            if dst.exists():
                continue
            try:
                from kraken.train import VGSLRecognitionModel
                mm = VGSLRecognitionModel.load_from_checkpoint(str(p),
                                                               weights_only=False)
                mm.net.save_model(str(dst))
                epochs_out.append(dst.name)
            except Exception as exc:
                # Не-VGSL архітектура (PP-OCRv6, kraken ≥ 7.1) у coreml не
                # зберігається і VGSL-класом не відкривається — тоді епоху
                # віддає сам ketos у safetensors.
                st = dst.with_suffix(".safetensors")
                rc = subprocess.run([sys.executable, str(launcher), "convert",
                                     "-o", str(st), str(p)],
                                    capture_output=True, text=True).returncode
                if rc == 0 and st.exists():
                    epochs_out.append(st.name)
                else:
                    # Не падати: найкраща епоха вже лежить у working, і втрата
                    # однієї проміжної не варта всього трену.
                    print(f"[ktrain] ⚠ епоха {p.name} не сконвертувалась "
                          f"({type(exc).__name__}: {exc}; ketos convert rc={rc})",
                          flush=True)
        print(f"[ktrain] ваг епох у виході: {len(epochs_out)}", flush=True)

    try:
        xeval_out = _xeval_epochs(int(params.get("xeval_batch", 1)), base=base_model)
    except Exception as exc:   # замір не має права забрати ваги
        print(f"[xeval] не вдався: {type(exc).__name__}: {exc}", flush=True)
        xeval_out = []

    summary = {
        "epoch_models": epochs_out,
        "xeval": xeval_out,
        "base_model": base_model.name if base_model else f"scratch: {spec}",
        "train_arrows": [p.name for p in train_arrows],
        "val_arrows": [p.name for p in val_arrows],
        "device": device,
        "params": {k: params[k] for k in sorted(params)},
        "epochs_run": len(history),
        "history": history,
        "best_weights": best.name,
        "best_val_accuracy": max((h["val_accuracy"] for h in history),
                                 default=None),
        # від старту ПЕРШОЇ спроби: після відкоту з DDP тут стояв переприсвоєний
        # t0 і summary звітував лише про час другої спроби
        "wall_sec": round(time.time() - t_start, 1),
        "stopped_by_wall_limit": bool(_stopped),
        # Продовження в ketos неможливе (немає --resume), тож наступний прогін
        # робить теплий старт із цієї моделі. Стан оптимізатора при цьому
        # втрачається — це fine-tune з ваг, а не продовження того самого трену.
        "continue_with": (
            f"gpurunner run kraken_train ... -p model_dataset=<slug з "
            f"{out_model.name}> -p dataset={params['dataset']}"
        ),
        "finished": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    (KAGGLE_WORKING / "ktrain_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[ktrain] done: best={summary['best_val_accuracy']} "
          f"epochs={summary['epochs_run']} wall={summary['wall_sec']}s",
          flush=True)
    if _stopped:
        print(f"[ktrain] трен зупинено за годинником, не за збіжністю. Щоб "
              f"доучити: залий {out_model.name} датасетом і запусти з "
              f"-p model_dataset=<slug> (теплий старт; стан оптимізатора "
              f"ketos не зберігає)", flush=True)


if __name__ == "__main__":
    main({})
