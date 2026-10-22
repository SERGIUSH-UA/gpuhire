"""Rukopys-OCR-4B: сторінка → JSON регіонів (bbox + тип + текст) → рядки тексту.

Самодостатній: stdlib + torch/transformers/pillow. Контракт:
вхід  /kaggle/input/**   — сторінки *.jpg|*.png (flat чи в підтеках)
вихід /kaggle/working/
        pages/<stem>.txt       — текст регіонів у порядку моделі, рядок = рядок
        raw/<stem>.txt         — сира відповідь моделі (для розбору провалів)
        rukopys_results.json   — per-page sec/токени/помітки провалу

Модель: ebinan92/Rukopys-OCR-4B (повний тюн Qwen3.5-4B, 3-тє місце Kaggle
Handwritten to Data, 2026). Промпт — ДОСЛІВНО з картки моделі: автор пише, що
саме на ньому вчив, і будь-яка «покращена» інструкція міряла б іншу модель.

## Чому текст ріжеться на рядки саме так

Вимірювач (``clan_probe.py``) судить токени по РЯДКАХ і шукає прізвище,
розірване переносом, парою «останній токен рядка + перший наступного». Тому
``text`` регіону ділиться за ``\\n`` (модель ставить їх там, де рядок в
оригіналі), а регіони йдуть підряд у порядку відповіді. Таблиця (``a | b``)
лишається одним рядком із розділювачами-пробілами.

## Сита провалу — ті самі чотири, що на CHURRO

HTR_HISTORY §2.2: VLM провалюється так, що вихід виглядає повним. Тут кожна
сторінка несе помітки: ``hit_cap`` (генерація дійшла до стелі токенів — тиха
стеля, сигнатура №1), ``parse_ok`` (JSON не закрився — те саме з іншого боку),
``refused``, ``low_diversity`` / ``trimmed`` (зациклення). Вигадану прозу
детектор не ловить — її видно лише в порівнянні з іншими рушіями.

⚠ БЕЗ `from __future__ import annotations` — код інжектиться ПІСЛЯ параметрів
у кернел-cell, future-import посеред файлу = SyntaxError.
"""
import json
import math
import re
import threading
import time
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

PROMPT = (
    "Detect every text region in this Ukrainian handwritten document and "
    "return a JSON array of regions. Each region has bbox (x1 y1 x2 y2 in "
    "0..1000 normalized image coordinates), type (handwritten | printed | "
    "formula | table | annotation | image | graph), and text (transcription; "
    "empty for image/graph; LaTeX for formula; pipe-separated for table)."
)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")
_TEXT_RE = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"')
# Роздільники порожньої сітки таблиці: пробіли, «|» і ЕКРАНОВАНИЙ «\n» з JSON
# (двосимвольний, а не літера n).
_SEP_RE = re.compile(r'(?:\\n|[\s|"])+')
_REFUSAL_RE = re.compile(
    r"I'm sorry|I am sorry|I am unable|I cannot|I can't assist|I can't process|"
    r"unable to process|as an AI", re.IGNORECASE)


def _trim_repeat_tail(text, min_reps=6):
    """Відрізати хвіст-луп із короткого юніта (як у churro_runner)."""
    for unit_len in range(2, 41):
        unit = text[-unit_len:]
        if len(unit) < unit_len or not unit.strip():
            continue
        reps = 1
        while reps * unit_len < 4000 and text.endswith(unit * (reps + 1)):
            reps += 1
        if reps >= min_reps:
            return text[: len(text) - unit_len * (reps - 1)].rstrip(), True
    return text, False


def _low_diversity(text, min_words=30, thresh=0.35):
    words = text.split()
    if len(words) < min_words:
        return False
    return len(set(w.lower() for w in words)) / len(words) < thresh


def _regions(raw):
    """JSON-масив регіонів; на обірваному JSON — витягти всі закриті ``text``.

    Повертає (regions, parse_ok). Обірваний масив — нормальний наслідок стелі
    токенів, і викидати вже прочитані регіони через незакриту дужку в кінці
    означало б рахувати сторінку порожньою там, де вона прочитана на 90%.
    """
    s = _FENCE_RE.sub("", raw.strip())
    try:
        # strict=False: сирі переноси всередині рядка JSON модель теж видає
        data = json.loads(s, strict=False)
        if isinstance(data, dict):
            data = data.get("regions") or [data]
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)], True
    except Exception:
        pass
    out = []
    end = 0
    for m in _TEXT_RE.finditer(s):
        end = m.end()
        out.append({"type": "?", "text": _unescape(m.group(1))})
    # Незакритий ОСТАННІЙ рядок — це зазвичай таблиця на всю сторінку, обірвана
    # стелею: у пробі 10f05736 саме в ньому лежав увесь текст аркуша.
    tail = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)$', s[end:])
    if tail:
        out.append({"type": "?", "text": _unescape(tail.group(1).rstrip("\\"))})
    return out, False


def _unescape(body):
    try:
        return json.loads('"' + body + '"', strict=False)
    except Exception:
        return body.replace("\\n", "\n").replace('\\"', '"')


def _region_lines(regions):
    """Регіони → рядки тексту (таблиця — клітинки через пробіл) + лічильник типів."""
    lines = []
    types = {}
    for r in regions:
        t = str(r.get("type") or r.get("category") or "?")
        types[t] = types.get(t, 0) + 1
        txt = r.get("text")
        if not isinstance(txt, str) or not txt.strip():
            continue
        if t == "table" or "|" in txt:
            txt = txt.replace("|", " ")
        for ln in txt.split("\n"):
            ln = " ".join(ln.split())
            if ln:
                lines.append(ln)
    return lines, types


def _repeat_block(lines, k=8, min_chars=60):
    """Останні k рядків уже стояли ПІДРЯД раніше — модель пішла по колу записами.

    Проба 10f05736, спр.134 скан 00513: блок із трьох актів («Акантій … Шимко …
    Цагаріевъ») повторено тричі до стелі токенів, решта аркуша не прочитана.
    Токени й клітинки там різноманітні, тож інші сита мовчать. Формула акту
    («Священникъ …», «И.д. Псаломщика …») повторюється легітимно, але не вісім
    рядків підряд разом з іменами батьків; ``min_chars`` відсікає вікна з
    самих номерів і складів переносу.

    🔴 Потрібні ДВА попередні входження, не одне: скан — розворот із двох
    сторінок, і друкована шапка графи («ГОДѢ, ЧАСТЬ ПЕРВЯЯ О РОДИВШИХСЯ» + назви
    колонок) законно стоїть на ньому двічі. З одним входженням стоп обрізав
    24 сторінки c6c3acdd на шапці правої сторінки.
    """
    if len(lines) < 3 * k:
        return False
    key = [re.sub(r"^\d+[\s.]*", "", ln) for ln in lines]
    tail = key[-k:]
    if sum(len(x) for x in tail) < min_chars:
        return False
    earlier = [i for i in range(len(key) - 2 * k + 1) if key[i:i + k] == tail]
    return len(earlier) >= 2


def _postprocess(raw):
    regions, parse_ok = _regions(raw)
    lines, types = _region_lines(regions)
    text = "\n".join(lines)
    text, trimmed = _trim_repeat_tail(text)
    return {"text": text, "parse_ok": parse_ok, "n_regions": len(regions),
            "types": types, "trimmed": trimmed,
            "refused": bool(_REFUSAL_RE.search(raw)),
            "low_diversity": _low_diversity(text),
            "repeat_block": _repeat_block(lines)}


def _ensure_deps(accel=True):
    """transformers ≥5.8.1 (клас Qwen3.5) + за ``accel`` flash-linear-attention.

    Без `fla` 24 з 32 шарів Qwen3.5 (лінійна увага) рахуються довідковим
    torch-шляхом: ~9 ток/с на T4, до 14 хв на сторінку (проба 10f05736, у лозі
    «falling back to its reference PyTorch implementation»). transformers шукає
    пакет у момент ІМПОРТУ модуля моделі й при будь-якій помилці мовчки бере
    torch-шлях — тому ставимо ДО завантаження моделі і друкуємо, чи імпорт вдався.
    Повертає статус для зведення.
    """
    import importlib
    import subprocess
    import sys
    need_tf = True
    try:
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        need_tf = (major, minor) < (5, 8)
    except Exception:
        pass
    if need_tf:
        print("[rukopys] pip install transformers>=5.8.1 …", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                        "transformers>=5.8.1", "accelerate>=1.0", "pillow>=10.0"],
                       check=True)
    if not accel:
        return "off"
    # Голий пакет (без extra [cuda]) з v0.5 НЕ тягне torch — образ Kaggle не чіпаємо.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "flash-linear-attention"], check=False)
    try:
        m = importlib.import_module("fla.ops.gated_delta_rule")
        status = "ok chunk={} recurrent={}".format(
            hasattr(m, "chunk_gated_delta_rule"),
            hasattr(m, "fused_recurrent_gated_delta_rule"))
    except Exception as exc:
        status = "FAIL {}: {}".format(type(exc).__name__, exc)
    print("[rukopys] fla: {}".format(status), flush=True)
    return status


def _collect():
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    seen = {}
    for pat in exts:
        for p in KAGGLE_INPUT.rglob(pat):
            seen.setdefault(p.stem.lower(), p)
    return [seen[k] for k in sorted(seen)]


def _main_vllm(params):
    """vLLM-шлях: усі сторінки одним ``generate``, налаштування ДОСЛІВНО з картки.

    bf16, ``max_model_len`` 16384, min/max_pixels 256/4096·factor², ``max_tokens``
    8192, ``temperature`` 0. Власних стопів тут НЕМАЄ навмисно: у c6c3acdd саме
    вони зіпсували замір (обрізали порожню сітку бланка й шапку другої сторінки
    розвороту), а на сучасній карті з батчингом луп до стелі коштує секунди, а
    не 14 хвилин T4. Лупи позначаються ПІСЛЯ генерації тими самими ситами
    (``hit_cap`` = finish_reason «length», ``repeat_block``, ``low_diversity``).

    Образ: ``vllm/vllm-openai:v0.29.0`` (CUDA 13.0) — transformers ≥5.10 уже
    всередині, нічого не ставимо.
    """
    import time
    t_install = time.time()
    try:
        import vllm  # noqa: F401
    except ImportError:
        # Базовий образ gpurunner (pytorch) замість vllm/vllm-openai: на образі
        # vLLM із власним ENTRYPOINT Vast 15.09 відхилив наш SSH-ключ (інстанс
        # 51113811), а на pytorch-образі SSH перевірений десятками прогонів.
        # vllm==0.29.0 тягне torch 2.13 під CUDA 13 — хост із cuda_max_good ≥13.
        import subprocess
        import sys
        print("[rukopys] pip install vllm==0.29.0 …", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.29.0"],
                       check=True)
    # Triton компілює ядра vLLM на ходу й вимагає C-компілятор. Образ
    # pytorch/*-runtime його не має: 15.09 (821589c2) EngineCore упав із
    # «Failed to find C compiler» уже після установки vLLM, тобто на оплаченій карті.
    import shutil
    if not (shutil.which("gcc") or shutil.which("cc")):
        import subprocess
        print("[rukopys] apt-get install gcc …", flush=True)
        subprocess.run("apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
                       "apt-get install -y -qq --no-install-recommends gcc libc6-dev",
                       shell=True, check=True)
    install_sec = round(time.time() - t_install, 1)
    print("[rukopys] vllm ready ({}s install, gcc={})".format(
        install_sec, shutil.which("gcc")), flush=True)
    # Рушій V1 за замовчуванням живе в ОКРЕМОМУ процесі, стартованому через spawn,
    # а spawn заново імпортує головний модуль. Раннер gpurunner виконується через
    # exec без запобіжника головного модуля, тож дочірній процес повторював увесь
    # job і падав на «start a new process before … bootstrapping phase»
    # (b0933fc0, 15.09). Рушій у тому самому процесі — дочірнього немає.
    import os
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Семплер FlashInfer (top_k_top_p_sampling_from_logits) збирає CUDA-ядро на
    # ходу і шукає nvcc, якого в pytorch-runtime нема (b3cc034f, 15.09: рушій
    # пройшов завантаження й прогрів і впав на першому семплі). У greedy-режимі
    # (temperature 0) нативний torch-семплер нічого не втрачає.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    model_id = params.get("model", "ebinan92/Rukopys-OCR-4B")
    max_new = int(params.get("max_new_tokens", 8192))
    want = {s.strip().lower() for s in str(params.get("pages", "")).split(",") if s.strip()}
    pages = _collect()
    if want:
        pages = [p for p in pages if p.stem.lower() in want]
    if not pages:
        raise RuntimeError("no images under {} (pages={!r})".format(KAGGLE_INPUT, want))
    pages_out = KAGGLE_WORKING / "pages"
    raw_out = KAGGLE_WORKING / "raw"
    pages_out.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_id)
    factor = proc.image_processor.patch_size * proc.image_processor.merge_size
    llm = LLM(
        model=model_id, dtype="bfloat16", max_model_len=16384,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": 256 * factor * factor,
                             "max_pixels": 4096 * factor * factor},
        gpu_memory_utilization=float(params.get("gpu_mem_util", 0.9)),
        max_num_seqs=int(params.get("max_num_seqs", 16)),
    )
    cold_start = round(time.time() - t0, 1)
    print("[rukopys] vLLM loaded in {}s".format(cold_start), flush=True)

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = proc.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
    reqs = [{"prompt": prompt, "multi_modal_data": {"image": Image.open(p).convert("RGB")}}
            for p in pages]
    sp = SamplingParams(max_tokens=max_new, temperature=0.0)
    t1 = time.time()
    outs = llm.generate(reqs, sp)
    gen_sec = round(time.time() - t1, 1)

    results = []
    for path, o in zip(pages, outs):
        c = o.outputs[0]
        raw = c.text
        post = _postprocess(raw)
        (raw_out / (path.stem + ".txt")).write_text(raw, encoding="utf-8")
        (pages_out / (path.stem + ".txt")).write_text(post["text"], encoding="utf-8")
        n_new = len(c.token_ids)
        results.append({
            "page": path.stem, "sec": None, "device": "vllm", "error": "",
            "in_tokens": len(o.prompt_token_ids or []), "new_tokens": n_new,
            "tok_per_sec": None, "hit_cap": c.finish_reason == "length",
            "stop": c.finish_reason, "chars": len(post["text"]),
            "lines": post["text"].count("\n") + 1 if post["text"] else 0,
            **{k: post[k] for k in ("parse_ok", "n_regions", "types", "trimmed",
                                    "refused", "low_diversity", "repeat_block")}})
        print("[rukopys] {}: {} chars, {} regions, new {} tok, {}".format(
            path.stem, len(post["text"]), post["n_regions"], n_new, c.finish_reason),
            flush=True)

    total_new = sum(r["new_tokens"] for r in results)
    summary = {
        "engine": "vllm", "model": model_id, "dtype": "bfloat16",
        "max_new_tokens": max_new, "cold_start_sec": cold_start,
        "gen_sec": gen_sec, "total_new_tokens": total_new,
        "throughput_tok_s": round(total_new / gen_sec, 1) if gen_sec else None,
        "n_pages": len(results),
        "n_hit_cap": sum(1 for r in results if r["hit_cap"]),
        "n_repeat_block": sum(1 for r in results if r["repeat_block"]),
        "n_parse_fail": sum(1 for r in results if r["parse_ok"] is False),
        "n_error": 0, "wall_sec": round(time.time() - t0, 1), "finished": _utc_iso(),
        "items": results,
    }
    (KAGGLE_WORKING / "rukopys_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[rukopys] done: {} pages in {}s generate, {} tok/s, cap {}, repeat_block {}".format(
        summary["n_pages"], gen_sec, summary["throughput_tok_s"], summary["n_hit_cap"],
        summary["n_repeat_block"]), flush=True)
    del llm


def main(params):
    if str(params.get("engine", "hf")).lower() == "vllm":
        return _main_vllm(params)
    fla_status = _ensure_deps(bool(params.get("accel", True)))
    import torch
    from PIL import Image
    from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList
    try:
        from transformers import AutoModelForMultimodalLM as AutoVLM
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoVLM

    model_id = params.get("model", "ebinan92/Rukopys-OCR-4B")
    max_pixels = int(params.get("max_pixels", 4096 * 32 * 32))
    max_new = int(params.get("max_new_tokens", 8192))
    skip_existing = bool(params.get("skip_existing", True))
    want = {s.strip().lower() for s in str(params.get("pages", "")).split(",") if s.strip()}
    dtype_name = str(params.get("dtype", "auto")).strip().lower()

    if dtype_name == "auto":
        # T4 (sm_75) не має нативного bf16; fp16 для моделі, вченої в bf16, —
        # ризик переповнення, тому пробний прогін міряє обидва явно.
        dtype_name = "bfloat16" if torch.cuda.is_bf16_supported(including_emulation=False) \
            else "float16"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[dtype_name]

    class DegenerationStop(StoppingCriteria):
        """Стоп на лупі: (1) жменька унікальних токенів у хвості; (2) повтор
        КЛІТИНОК таблиці.

        (2) з'явився після проби 10f05736: метричний розворот модель віддає однією
        таблицею і крутить рядок «Мѣстѣ Микиты Солодка | …» до стелі 8192 токени —
        14 хв карти на сторінку. Токени там різноманітні (номери рядків ростуть),
        тож (1) мовчить; ловить лише те, що з 32 останніх непорожніх клітинок
        унікальних ≤ 8. Жива таблиця метрики так не виглядає: імена, дати, села.
        """

        def __init__(self, tok, prompt_len, win=160, min_unique=12,
                     cells=32, max_cells_unique=8, every=24, grid_max=1200):
            self.tok, self.prompt_len = tok, prompt_len
            self.win, self.min_unique = win, min_unique
            self.cells, self.max_cells_unique, self.every = cells, max_cells_unique, every
            self.grid_max = grid_max
            self.fired = ""

        def __call__(self, input_ids, scores, **kw):
            gen_len = input_ids.shape[1] - self.prompt_len
            if gen_len < self.win:
                return False
            if input_ids[0, -self.win:].unique().numel() <= self.min_unique:
                # Порожня сітка друкованого бланка (« | | | \n» × десятки рядків)
                # теж дає жменьку токенів, але це легітимний вихід: далі модель
                # переходить до рукописних регіонів. У c6c3acdd цей стоп обрізав
                # так 39 із 72 сторінок. Сітку пускаємо до grid_max токенів.
                # Дивимось лише на ОСТАННІ 48 токенів: у вікні 160 на момент
                # першого спрацювання ще сидить хвіст шапки («родія. | вѣданія.»),
                # і вся сітка після неї рахувалась би «змістом».
                recent = self.tok.decode(input_ids[0, -48:], skip_special_tokens=True)
                if len(_SEP_RE.sub("", recent)) > 8:
                    self.fired = "low_diversity"
                    return True
                if gen_len >= self.grid_max:
                    longer = self.tok.decode(input_ids[0, -self.grid_max:],
                                             skip_special_tokens=True)
                    # ≤40 символів змісту на 1200 токенів — сітка з поодинокими
                    # цифрами, а не рукописний текст
                    if len(_SEP_RE.sub("", longer)) <= 40:
                        self.fired = "empty_grid"
                        return True
            if gen_len % self.every:
                return False
            full = self.tok.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
            cells = [c.strip() for c in re.split(r"\||\\n|\n", full[-2000:])]
            cells = [re.sub(r"^\d+[\s.]*", "", c) for c in cells]
            cells = [c for c in cells if len(c) > 2][:-1]
            if len(cells) >= self.cells and \
                    len(set(cells[-self.cells:])) <= self.max_cells_unique:
                self.fired = "repeat_cells"
                return True
            # (3) луп цілими записами — див. _repeat_block
            if _repeat_block(_region_lines(_regions(full)[0])[0]):
                self.fired = "repeat_block"
                return True
            return False

    pages = _collect()
    if want:
        pages = [p for p in pages if p.stem.lower() in want]
    if not pages:
        raise RuntimeError("no images under {} (pages={!r})".format(KAGGLE_INPUT, want))

    pages_out = KAGGLE_WORKING / "pages"
    raw_out = KAGGLE_WORKING / "raw"
    pages_out.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)
    if skip_existing:
        n0 = len(pages)
        pages = [p for p in pages if not (pages_out / (p.stem + ".txt")).exists()]
        if n0 - len(pages):
            print("[rukopys] resume: -{} pages already done".format(n0 - len(pages)), flush=True)

    # fp32 не влазить у 16 ГБ однієї T4 — тоді одна репліка на обидві карти.
    n_gpu = max(1, torch.cuda.device_count())
    split = dtype_name == "float32" and n_gpu > 1
    n_rep = 1 if split else n_gpu
    t0 = time.time()
    replicas = []
    for gi in range(n_rep):
        proc = AutoProcessor.from_pretrained(model_id)
        mdl = AutoVLM.from_pretrained(
            model_id, dtype=dtype,
            device_map="auto" if split else "cuda:{}".format(gi))
        mdl.eval()
        dev = "cuda:0" if split else "cuda:{}".format(gi)
        replicas.append({"device": dev, "proc": proc, "model": mdl})
    cold_start = round(time.time() - t0, 1)
    print("[rukopys] {} replica(s) {} loaded in {}s; transformers {}".format(
        n_rep, dtype_name, cold_start, __import__("transformers").__version__), flush=True)

    lock = threading.Lock()
    queue = list(reversed(pages))
    total = len(pages)
    done_n = [0]

    def run_page(rep, path):
        device, proc, mdl = rep["device"], rep["proc"], rep["model"]
        t1 = time.time()
        err, raw, n_in, n_new, fired = "", "", 0, 0, ""
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            if w * h > max_pixels:
                s = math.sqrt(max_pixels / (w * h))
                img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": PROMPT}]}]
            text = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
            n_in = int(inputs["input_ids"].shape[1])
            stop = DegenerationStop(proc.tokenizer, n_in)
            with torch.inference_mode():
                out_ids = mdl.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                       stopping_criteria=StoppingCriteriaList([stop]))
            gen = out_ids[:, n_in:]
            n_new = int(gen.shape[1])
            fired = stop.fired
            raw = proc.batch_decode(gen, skip_special_tokens=True)[0]
            post = _postprocess(raw)
        except Exception as exc:  # одна сторінка не валить прогін
            err = "{}: {}".format(type(exc).__name__, exc)
            post = {"text": "", "parse_ok": None, "n_regions": 0, "types": {},
                    "trimmed": False, "refused": None, "low_diversity": None,
                    "repeat_block": None}
        secs = round(time.time() - t1, 1)
        (raw_out / (path.stem + ".txt")).write_text(raw, encoding="utf-8")
        (pages_out / (path.stem + ".txt")).write_text(post["text"], encoding="utf-8")
        return {"page": path.stem, "sec": secs, "device": device, "error": err,
                "in_tokens": n_in, "new_tokens": n_new,
                "tok_per_sec": round(n_new / secs, 2) if secs else None,
                "hit_cap": n_new >= max_new, "stop": fired,
                "chars": len(post["text"]), "lines": post["text"].count("\n") + 1 if post["text"] else 0,
                **{k: post[k] for k in ("parse_ok", "n_regions", "types", "trimmed",
                                        "refused", "low_diversity", "repeat_block")}}

    def worker(rep):
        local = []
        while True:
            with lock:
                if not queue:
                    break
                path = queue.pop()
            r = run_page(rep, path)
            local.append(r)
            with lock:
                done_n[0] += 1
                n = done_n[0]
            print("[rukopys] {}/{} {}: {} chars, {} regions, in {} / new {} tok, {}s "
                  "({} tok/s){}{}".format(
                      n, total, r["page"], r["chars"], r["n_regions"], r["in_tokens"],
                      r["new_tokens"], r["sec"], r["tok_per_sec"],
                      " HIT_CAP" if r["hit_cap"] else "",
                      " ERROR " + r["error"] if r["error"] else ""), flush=True)
        return local

    if n_rep == 1:
        results = worker(replicas[0])
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_rep) as ex:
            futs = [ex.submit(worker, rep) for rep in replicas]
            results = [r for f in futs for r in f.result()]

    results.sort(key=lambda r: r["page"])
    summary = {
        "model": model_id, "dtype": dtype_name, "fla": fla_status, "max_pixels": max_pixels,
        "max_new_tokens": max_new, "n_gpu": n_gpu, "replicas": n_rep,
        "cold_start_sec": cold_start, "n_pages": len(results),
        "n_hit_cap": sum(1 for r in results if r["hit_cap"]),
        "n_parse_fail": sum(1 for r in results if r["parse_ok"] is False),
        "n_error": sum(1 for r in results if r["error"]),
        "wall_sec": round(time.time() - t0, 1), "finished": _utc_iso(),
        "items": results,
    }
    (KAGGLE_WORKING / "rukopys_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[rukopys] done: {} pages, cap {}, parse-fail {}, errors {}, wall {}s".format(
        summary["n_pages"], summary["n_hit_cap"], summary["n_parse_fail"],
        summary["n_error"], summary["wall_sec"]), flush=True)


if __name__ == "__main__":
    main({})
