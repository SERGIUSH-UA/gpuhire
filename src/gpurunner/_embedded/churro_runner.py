"""CHURRO-3B транскрипція: сторінки І рядкові кропи в ОДНІЙ GPU-сесії.

Самодостатній: stdlib + torch/transformers/qwen-vl-utils/pillow. Контракт:
вхід  /kaggle/input/**   — сторінки *.jpg|*.png та/або кропи <page>/line_NNN.png
вихід /kaggle/working/
        pages/<stem>.txt      — повний текст сторінки (режим ``page``)
        lines/<page>.txt      — рядок на seg-індекс (режим ``line``)
        churro_results.json   — per-item chars/sec/device/помітки провалу

Модель: stanford-oval/churro-3B (base Qwen2.5-VL-3B-Instruct, EMNLP 2025).
Промпт сторінки — рідний з churro-ocr (prompts/ocr.py). T4: fp16 (bf16 на sm_75
нема), max_pixels обмежує вхідні токени, щоб 3.9-Мп скани не висадили 16 ГБ.

## Чому два режими в одному прогоні

Холодний старт (завантаження 3B на кожну карту) — десятки секунд, і платити його
двічі лише щоб порівняти «сторінка проти кропа» безглуздо. Важливіше інше:
обидві гілки міряються на ІДЕНТИЧНОМУ стані моделі, тож різниця в результаті —
це різниця контексту, а не випадковість завантаження.

Вихід режиму ``line`` навмисне має той самий вигляд, що в ``kraken_lines_infer.py``
і ``paddle_lines_infer.py`` (``<page>.txt``, рядок на seg-індекс, діри = порожні
рядки) — лягає в ``htr_phrase_recall.py`` без жодного перехідника.

## Що тут оптимізовано (і чому саме це)

1. **Батчі для кропів.** Рядок — це ~200×60 px і ~20 токенів відповіді. Гнати їх
   по одному означає тримати 3B-модель заради мікрозадачі: майже весь час іде на
   накладні, а не на обчислення. Батч на 16 кропів дає кратне прискорення.
   Сторінки лишаються батчем 1 — там і візуальних токенів на порядок більше,
   і генерація довга, тож пам'ять важливіша за пропускну здатність.
2. **Стеля токенів на режим.** Для кропа ``max_new=3072`` — не запобіжник, а
   запрошення до лупа на 3000 токенів. 64 вистачає з запасом (медіана рядка —
   17 символів) і водночас робить дегенерацію дешевою: вона обривається сама.
3. **Спільна черга замість суміжних шматків.** Ділити список навпіл можна лише
   коли складність однорідна. Вона не однорідна: сторінка з лупом іде 200 с,
   сусідня — 20 с, і карта, якій дістався «важкий» шматок, доганяє наприкінці
   на самоті. Черга з блокуванням прибирає цей хвіст повністю.
4. **Сортування за шириною перед батчингом.** У батчі всі елементи доповнюються
   до найширшого; змішати кроп на 3 слова з кропом на 15 — це прогнати половину
   батча по порожнечі. Сортування зводить доповнення нанівець.
5. **Резюмування.** Готові виходи пропускаються, тож обірваний job продовжується
   з місця зупинки, а не з нуля (наступали двічі за сесію 2026-07-26).
6. **Усі чотири сита провалу рахуються ТУТ.** ``gap_loop`` і стеля токенів були
   вбудовані, а відмова моделі й чужомовний цикл ловились постфактум окремим
   скриптом — і 57 сторінок спр.13 проскочили як валідні. Тепер помітка йде
   в ``churro_results.json`` одразу, разом із текстом.

⚠ Кропи називаються ``line_000.png`` У КОЖНІЙ теці сторінки. Старий пошук
дедуплікував за ІМ'ЯМ файлу — на 52 сторінках це лишило б 192 кропи з 6000
і тихо. Ключ дедуплікації — відносний ШЛЯХ.

⚠ БЕЗ `from __future__ import annotations` — код інжектиться ПІСЛЯ параметрів
у кернел-cell, future-import посеред файлу = SyntaxError.
"""
import json
import re
import tarfile
import threading
import time
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

SYSTEM_PROMPT = (
    "You are an expert in diplomatic transcription of historical documents "
    "from various languages. Your task is to extract the full text from a given page."
)

USER_PROMPT = (
    "Follow these instructions:\n\n"
    "1. You will be provided with a scanned document page.\n\n"
    "2. Perform transcription on the entirety of the page, converting all visible "
    "text into the following format. Include handwritten and print text, if any. "
    "Include tables, captions, headers, main text and all other visible text.\n\n"
    "3. If you encounter any non-text elements, simply skip them without attempting "
    "to describe them.\n\n"
    "4. Do not modernize or standardize the text. For example, if the transcription "
    "is using \"ſ\" instead of \"s\" or \"а\" instead of \"a\", keep it that way.\n\n"
    "5. When you come across text in languages other than English, transcribe it as "
    "accurately as possible without translation.\n\n"
    "6. Output the OCR result in the following format:\n\n"
    "<output>\nextracted text here\n</output>\n\n"
    "Remember, your goal is to accurately transcribe the text from the scanned page "
    "as much as possible. Process the entire page, even if it contains a large amount "
    "of text, and provide clear, well-formatted output. Pay attention to the "
    "appropriate reading order and layout of the text."
)

# Промпт рядка навмисно короткий: інструкції про таблиці, заголовки й порядок
# читання до одного рядка не стосуються, а кожне зайве речення — це токени
# в КОЖНОМУ елементі батча.
LINE_SYSTEM_PROMPT = (
    "You are an expert in diplomatic transcription of historical manuscripts."
)

LINE_USER_PROMPT = (
    "This image is a single cropped line from a handwritten historical document.\n\n"
    "1. Transcribe exactly what is written, character by character.\n"
    "2. Do not modernize or standardize spelling: keep archaic letters such as "
    "\"ѣ\", \"ъ\", \"і\", \"ѳ\" exactly as they appear.\n"
    "3. Do not translate. Do not add words that are not visible.\n"
    "4. If the line is empty or completely illegible, output nothing.\n\n"
    "Output only the transcription, in this format:\n\n"
    "<output>\ntranscription here\n</output>"
)

_OUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)
_GAP_RE = re.compile(r"(?:<Gap\b[^>]*/>\s*){3,}")
_LINE_RE = re.compile(r"^line_(\d+)$", re.IGNORECASE)

# Відмови моделі. Легітимного архівного тексту 1800-х з цими фразами не буває,
# тож детектор стовідсотковий; без нього 49/478 сторінок спр.13 пройшли як
# «валідні» (churro-refusal-and-repetition-blindspot).
_REFUSAL_RE = re.compile(
    r"I'm sorry|I am sorry|I am unable|I cannot|I can't assist|I can't process|"
    r"unable to process|as an AI", re.IGNORECASE)


def _collapse_gaps(text):
    """Зрізати дегенеративні серії <Gap …/> (луп-фейл пілота 2026-07-21).

    Серія ≥3 підряд — не транскрипція, а зациклення: всередині тексту стискаємо
    до одного маркера, хвостову серію відрізаємо повністю. Повертає (text, looped).
    """
    looped = bool(_GAP_RE.search(text))
    text = _GAP_RE.sub('<Gap reason="illegible" extent="run"/> ', text)
    text = re.sub(r"(?:<Gap\b[^>]*/>\s*)+$", "", text)
    text = re.sub(r"<Gap[^>]*$", "", text).rstrip()   # обірваний тег після раннього стопу
    return text, looped


def _trim_repeat_tail(text, min_reps=6):
    """Відрізати хвіст-луп із короткого юніта («yſſyſſyſſ…», верифікація f4683783).

    Пробуємо юніти 2..40 симв.: якщо хвіст = юніт, повторений ≥min_reps разів —
    зрізаємо всі повтори (один екземпляр лишаємо). Легітимний текст так не
    закінчується; формули актів довші за 40 симв. і не йдуть 6+ разів упритул.
    """
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
    """Чужомовний цикл-повторення: слів багато, унікальних мало.

    Ловить провал, який НЕ ловлять gap_loop і стеля токенів (8/478 спр.13):
    модель генерує зв'язний текст не з документа, крутячись навколо теми.
    """
    words = text.split()
    if len(words) < min_words:
        return False
    return len(set(w.lower() for w in words)) / len(words) < thresh


def _postprocess(raw):
    """Розбір відповіді моделі + УСІ чотири сита провалу.

    На рівні модуля навмисне: це єдина частина раннера, яку можна (і треба)
    перевірити без GPU, а помилка тут коштує дорого — саме непозначений провал
    отруює псевдо-GT, бо виглядає як звичайна транскрипція.
    """
    m = _OUT_RE.search(raw)
    # без закривального </output> (ранній стоп) — зрізати відкривальний тег
    text = m.group(1) if m else re.sub(r"^\s*<output>\s*", "", raw).strip()
    text, collapsed = _collapse_gaps(text)
    text, trimmed = _trim_repeat_tail(text)
    return {"text": text, "truncated": m is None, "collapsed": collapsed,
            "trimmed": trimmed, "refused": bool(_REFUSAL_RE.search(text)),
            "low_diversity": _low_diversity(text)}


def _blank(n):
    """Заглушки на впалий елемент: провал одного не валить прогін."""
    return [{"text": "", "truncated": None, "collapsed": False, "trimmed": False,
             "refused": None, "low_diversity": None} for _ in range(n)]


def _ensure_deps():
    """Доставити transformers на місці, якщо оточення їх не має.

    Lightning НЕ виконує ``Job.requirements()`` — Studio-оточення позичене як є
    (та сама пастка, що в ``kraken_train_runner._ensure_kraken``). На Kaggle це
    no-op: пакет уже стоїть. Qwen2_5_VL* класи з'явились у transformers 4.49.
    """
    try:
        import transformers  # noqa: F401
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
        return
    except Exception:
        pass
    import subprocess
    import sys
    print("[churro] pip install transformers/accelerate…", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "transformers>=4.49", "accelerate>=0.30", "pillow>=10.0"],
        check=True)


def _extract_inputs():
    """Розпакувати всі *.tgz під входом; повернути корені для пошуку.

    Датасет з тисячами дрібних файлів ОБОВ'ЯЗКОВО їде одним архівом: і Lightning,
    і Kaggle CLI ллють пофайлово, тож 6000 кропів = години проти хвилин на .tgz.
    """
    tgzs = sorted(KAGGLE_INPUT.rglob("*.tgz")) + sorted(KAGGLE_INPUT.rglob("*.tar.gz"))
    if not tgzs:
        return [KAGGLE_INPUT]
    dest = Path("/tmp/churro_input")
    dest.mkdir(parents=True, exist_ok=True)
    for t in tgzs:
        with tarfile.open(t) as tf:
            tf.extractall(dest)
        print("[churro] extracted {} -> {}".format(t.name, dest), flush=True)
    # Разом з архівом можуть лежати й звичайні картинки — шукаємо в обох коренях.
    return [dest, KAGGLE_INPUT]


def _collect(mode, roots=None):
    """Знайти вхід і рознести на сторінки та рядкові кропи.

    Kaggle монтує датасет під /kaggle/input/datasets/<owner>/<slug>/ (nested) —
    тому ТІЛЬКИ рекурсивний пошук (гоча з yolo_spotter, 2026-05-26).
    Ключ дедуплікації — відносний шлях, а не ім'я: кропи в усіх теках сторінок
    називаються однаково, і дедуп за іменем лишив би по одному кропу на індекс.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    seen = {}
    for root in (roots or [KAGGLE_INPUT]):
        for pat in exts:
            for p in root.rglob(pat):
                seen.setdefault(str(p).lower(), p)
    pages, lines = [], []
    for p in (seen[k] for k in sorted(seen)):
        m = _LINE_RE.match(p.stem)
        if m:
            lines.append({"kind": "line", "path": p, "page": p.parent.name,
                          "idx": int(m.group(1))})
        else:
            pages.append({"kind": "page", "path": p, "page": p.stem, "idx": None})
    if mode == "page":
        lines = []
    elif mode == "line":
        pages = []
    return pages, lines


def main(params):
    _ensure_deps()
    import torch
    from PIL import Image
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    model_id = params.get("model", "stanford-oval/churro-3B")
    max_pixels = int(params.get("max_pixels", 1254400))   # ~1600×784 ≈ 1600 vis-токенів
    max_new = int(params.get("max_new_tokens", 3072))
    line_max_new = int(params.get("line_max_new_tokens", 64))
    line_max_pixels = int(params.get("line_max_pixels", 200704))  # ~448×448 на кроп
    line_batch = int(params.get("line_batch", 16))
    mode = str(params.get("mode", "both")).strip().lower()
    skip_existing = bool(params.get("skip_existing", True))
    pages_filter = {s.strip().lower() for s in str(params.get("pages", "")).split(",")
                    if s.strip()}

    class DegenerationStop(StoppingCriteria):
        """Стоп на дегенерації хвоста (луп-фейли пілота: 206с/стор до стелі токенів).

        Два детектори: (1) серія <Gap …/> у декодованому хвості; (2) низька
        різноманітність токенів — посимвольні лупи типу «yſſyſſ…» (03100) дають
        жменьку унікальних id на 96 токенів, жива транскрипція — 50+. Обидві
        перевірки — копійки проти forward-pass.

        Тільки для сторінок (батч 1): у кропів роль запобіжника грає стеля в
        64 токени, і per-row логіка була б складністю без виграшу.
        """

        def __init__(self, tok, prompt_len, max_gaps=4, win=96, min_unique=16):
            self.tok, self.prompt_len = tok, prompt_len
            self.max_gaps, self.win, self.min_unique = max_gaps, win, min_unique
            self.fired = ""

        def __call__(self, input_ids, scores, **kw):
            gen_len = input_ids.shape[1] - self.prompt_len
            if gen_len < self.win:
                return False
            tail_ids = input_ids[0, -self.win:]
            if tail_ids.unique().numel() <= self.min_unique:
                self.fired = "low_diversity"
                return True
            tail = self.tok.decode(input_ids[0, -160:], skip_special_tokens=True)
            if tail.count("<Gap") >= self.max_gaps:
                self.fired = "gap_run"
                return True
            return False

    pages, lines = _collect(mode, _extract_inputs())
    if pages_filter:
        pages = [t for t in pages if t["page"].lower() in pages_filter]
        lines = [t for t in lines if t["page"].lower() in pages_filter]
    if not pages and not lines:
        raise RuntimeError("no images found under {} (mode={})".format(
            KAGGLE_INPUT, mode))

    pages_out = KAGGLE_WORKING / "pages"
    lines_out = KAGGLE_WORKING / "lines"
    if pages:
        pages_out.mkdir(parents=True, exist_ok=True)
    if lines:
        lines_out.mkdir(parents=True, exist_ok=True)

    # Резюмування. Сторінки — пофайлово; рядки — цілою сторінкою, бо вихід
    # агрегований і дописувати в середину готового файлу нема сенсу.
    if skip_existing:
        n0, n1 = len(pages), len(lines)
        pages = [t for t in pages
                 if not (pages_out / (t["page"] + ".txt")).exists()]
        done_pages = {p.stem for p in lines_out.glob("*.txt")}
        lines = [t for t in lines if t["page"] not in done_pages]
        if n0 - len(pages) or n1 - len(lines):
            print("[churro] resume: -{} pages, -{} line crops already done".format(
                n0 - len(pages), n1 - len(lines)), flush=True)

    # Батчі рядків: сортуємо за шириною, щоб доповнення всередині батча було
    # мінімальним, і ріжемо на шматки по line_batch.
    line_batches = []
    if lines:
        def _width(task):
            try:
                with Image.open(task["path"]) as im:
                    return im.width
            except Exception:
                return 0
        lines.sort(key=_width)
        line_batches = [lines[i:i + line_batch]
                        for i in range(0, len(lines), line_batch)]

    # Одиниця роботи для черги: сторінка = 1 картинка, рядки = батч.
    # Сторінки йдуть ПЕРШИМИ у списку, а черга розбирається з кінця (pop) —
    # тобто дешеві батчі кропів розходяться раніше, а довгі сторінки лишаються
    # на потім і не тримають карту в кінці прогону наодинці.
    work = [{"kind": "page", "items": [t]} for t in pages] + \
           [{"kind": "line", "items": b} for b in line_batches]
    total_items = len(pages) + len(lines)
    print("[churro] mode={}: {} pages + {} line crops ({} batches of {})".format(
        mode, len(pages), len(lines), len(line_batches), line_batch), flush=True)

    # Kaggle-«T4» = машина з ДВОМА T4 (enum NvidiaTeslaT4, вкладка Events показує
    # «GPU T4 x2») — репліка моделі на кожну картку: 2× швидкість за ту саму
    # квоту (списується wall-clock сесії, не GPU-години).
    n_gpu = max(1, torch.cuda.device_count())
    t0 = time.time()
    replicas = []
    for gi in range(n_gpu):
        proc = AutoProcessor.from_pretrained(
            model_id, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
        # Генерація декодером вимагає доповнення ЗЛІВА, інакше в батчі
        # модель продовжує padding, а не текст.
        proc.tokenizer.padding_side = "left"
        mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda:{}".format(gi),
            attn_implementation="sdpa")
        mdl.eval()
        replicas.append({"device": "cuda:{}".format(gi), "proc": proc, "model": mdl})
    cold_start = round(time.time() - t0, 1)
    print("[churro] {} replica(s) loaded in {}s".format(n_gpu, cold_start), flush=True)

    # Кропи ріжуться під інший max_pixels, ніж сторінки: тримаємо окремий
    # процесор на репліку, щоб не перемикати конфіг між викликами в потоці.
    line_procs = {}
    if lines:
        for rep in replicas:
            lp = AutoProcessor.from_pretrained(
                model_id, min_pixels=64 * 28 * 28, max_pixels=line_max_pixels)
            lp.tokenizer.padding_side = "left"
            line_procs[rep["device"]] = lp

    lock = threading.Lock()
    queue = list(work)
    done_n = [0]
    page_lines = {}

    def run_page(rep, task):
        device, proc, mdl = rep["device"], rep["proc"], rep["model"]
        t1 = time.time()
        err, gap_loop = "", None
        try:
            img = Image.open(task["path"]).convert("RGB")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": USER_PROMPT},
                ]},
            ]
            chat_text = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[chat_text], images=[img],
                          return_tensors="pt").to(device)
            stop = DegenerationStop(proc.tokenizer, inputs.input_ids.shape[1])
            with torch.inference_mode():
                out_ids = mdl.generate(
                    **inputs, max_new_tokens=max_new, do_sample=False,
                    stopping_criteria=StoppingCriteriaList([stop]))
            gen = out_ids[:, inputs.input_ids.shape[1]:]
            raw = proc.batch_decode(gen, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]
            post = _postprocess(raw)
            gap_loop = bool(stop.fired) or post["collapsed"] or post["trimmed"]
        except Exception as exc:  # одна сторінка не валить прогін
            err = "{}: {}".format(type(exc).__name__, exc)
            post = _blank(1)[0]
        secs = round(time.time() - t1, 1)
        (pages_out / (task["page"] + ".txt")).write_text(
            post["text"], encoding="utf-8")
        return [{"kind": "page", "page": task["page"], "idx": None,
                 "chars": len(post["text"]), "sec": secs, "device": device,
                 "error": err, "gap_loop": gap_loop, "truncated": post["truncated"],
                 "refused": post["refused"], "low_diversity": post["low_diversity"]}]

    def run_line_batch(rep, tasks):
        device, mdl = rep["device"], rep["model"]
        proc = line_procs[device]
        t1 = time.time()
        err = ""
        try:
            imgs = [Image.open(t["path"]).convert("RGB") for t in tasks]
            chat_texts = []
            for img in imgs:
                messages = [
                    {"role": "system", "content": LINE_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": LINE_USER_PROMPT},
                    ]},
                ]
                chat_texts.append(proc.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True))
            inputs = proc(text=chat_texts, images=imgs, padding=True,
                          return_tensors="pt").to(device)
            with torch.inference_mode():
                out_ids = mdl.generate(**inputs, max_new_tokens=line_max_new,
                                       do_sample=False)
            gen = out_ids[:, inputs.input_ids.shape[1]:]
            raws = proc.batch_decode(gen, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
            posts = [_postprocess(r) for r in raws]
        except Exception as exc:  # один батч не валить прогін
            err = "{}: {}".format(type(exc).__name__, exc)
            posts = _blank(len(tasks))
        secs = round(time.time() - t1, 1)
        per = round(secs / max(1, len(tasks)), 2)
        out = []
        for task, post in zip(tasks, posts):
            # Один кроп — один рядок тексту: переносів усередині рядка не буває.
            line_text = " ".join(post["text"].split())
            with lock:
                page_lines.setdefault(task["page"], {})[task["idx"]] = line_text
            out.append({"kind": "line", "page": task["page"], "idx": task["idx"],
                        "chars": len(line_text), "sec": per, "device": device,
                        "error": err, "gap_loop": post["trimmed"],
                        "truncated": post["truncated"], "refused": post["refused"],
                        "low_diversity": post["low_diversity"]})
        return out

    def worker(rep):
        local = []
        while True:
            with lock:
                if not queue:
                    break
                unit = queue.pop()
            got = (run_page(rep, unit["items"][0]) if unit["kind"] == "page"
                   else run_line_batch(rep, unit["items"]))
            local.extend(got)
            with lock:
                done_n[0] += len(got)
                n = done_n[0]
            head = got[0]
            print("[churro] {}/{} {} {}{}: {} chars, {}s{}".format(
                n, total_items, head["kind"], head["page"],
                "" if head["kind"] == "page" else " x{}".format(len(got)),
                sum(g["chars"] for g in got), head["sec"],
                " ERROR " + head["error"] if head["error"] else ""), flush=True)
        return local

    if n_gpu == 1:
        results = worker(replicas[0])
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_gpu) as ex:
            futs = [ex.submit(worker, rep) for rep in replicas]
            results = [r for f in futs for r in f.result()]

    # Рядки зводимо в per-page файл ТОГО САМОГО формату, що дають kraken/paddle:
    # номер рядка у файлі = seg-індекс кропа, пропуски — порожні рядки.
    for page, by_idx in sorted(page_lines.items()):
        top = max(by_idx) if by_idx else -1
        rows = [by_idx.get(i, "") for i in range(top + 1)]
        (lines_out / (page + ".txt")).write_text("\n".join(rows), encoding="utf-8")

    results.sort(key=lambda r: (r["kind"], r["page"], r.get("idx") or 0))
    n_pages = sum(1 for r in results if r["kind"] == "page")
    n_lines = sum(1 for r in results if r["kind"] == "line")
    flagged = sum(1 for r in results
                  if r.get("refused") or r.get("gap_loop") or r.get("low_diversity"))
    summary = {
        "model": model_id, "mode": mode, "max_pixels": max_pixels,
        "max_new_tokens": max_new, "line_max_new_tokens": line_max_new,
        "line_max_pixels": line_max_pixels, "line_batch": line_batch,
        "n_gpu": n_gpu, "cold_start_sec": cold_start,
        "n_pages": n_pages, "n_lines": n_lines, "n_flagged": flagged,
        "wall_sec": round(time.time() - t0, 1),
        "finished": _utc_iso(), "items": results,
    }
    (KAGGLE_WORKING / "churro_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[churro] done: {} pages + {} lines, {} flagged, wall {}s on {} GPU(s)".format(
        n_pages, n_lines, flagged, summary["wall_sec"], n_gpu), flush=True)


if __name__ == "__main__":
    main({})
