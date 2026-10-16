"""Робітник друку: PaddleOCR-VL через vLLM-сервер — газети, журнали, довідники.

Їде на орендовану машину лише тоді, коли план несе `engine=paddle_vl`, і
працює ЗАМІСТЬ шарда рукописного рушія (`htr_case_run.py`). Рушій друку
існує тільки тут: у Нишпорці локального друку немає свідомо — Paddle з vLLM
важать гігабайти й хочуть карту на 20+ ГБ, а більшості людей друк не потрібен.

Замір 12.10.2026 на «Колективізованому селі» (1936): рукописний рушій клеїв
рядки сусідніх колонок (CER 11–17% на колонці), PaddleOCR-VL 1.6 дав 0,4–0,9%
і статті окремими блоками в порядку читання, ~4 с/стор на RTX 3090.

Контракт той самий, що в шарда рукописного рушія, бо його читає той самий
бокс-раннер (конвеєр, вотчдог, приймач повноти, чекпоінти):
- файл черги `--queue` (JSON-рядки `{case_dir, out_dir, case_key, index}`,
  `{"end": true}` — кінець; файл дописується на ходу);
- на сторінку `<stem>.txt` (плюс `<stem>.md` і `<stem>.blocks.json`),
  `_htr_meta.json` із `pages`, `_htr_quarantine.json`;
- події `@@PROGRESS@@` зі схемою 1: `page_start`, `htr`, `done`, `case_done`,
  `queue_wait`, `queue_done`, поле `qi` — номер справи в черзі.

Лише stdlib; `paddleocr` імпортується в середовищі друку (`/opt/pvl`).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

PROGRESS_PREFIX = "@@PROGRESS@@ "
PROGRESS_SCHEMA = 1
MODEL_NAME = "PaddleOCR-VL-1.6-0.9B"
DEFAULT_SERVER = "http://127.0.0.1:8118/v1"
FRAME_EXT = (".jpg", ".jpeg", ".png")
QUARANTINE = "_htr_quarantine.json"

#: Блоки, що текстом не є.
SKIP_LABELS = frozenset({"image", "header_image", "footer_image", "chart", "seal"})
#: Заголовки — окремим абзацом, щоб стаття починалась із власної назви.
HEADING_LABELS = frozenset({"doc_title", "paragraph_title"})
#: Письма, яких у нашій пресі немає: їхні літери — вигадка VLM на шумі
#: (на фото підшивки 12.10.2026 у кириличному тексті з'явилось «النوالبب»).
_FOREIGN = ("ARABIC", "HEBREW", "CJK", "HIRAGANA", "KATAKANA", "HANGUL", "THAI",
            "DEVANAGARI", "BENGALI", "TAMIL", "TELUGU", "ETHIOPIC", "GEORGIAN",
            "ARMENIAN")
#: Блок у крайній смузі кадру (частка ширини) — сусідня сторінка підшивки.
#: Смуга вузька свідомо: загубити статтю гірше, ніж лишити сміття.
EDGE_STRIP = 0.025
#: Коротший блок на повтор не перевіряється: «Пирятин», «1936» трапляються
#: на сторінці законно по кілька разів.
DUP_MIN_CHARS = 24
#: Скільки сторінок поспіль може впасти, перш ніж робітник визнає, що впав
#: сервер, а не сторінка. Інакше смерть vLLM розклала б карантином усю чергу.
FAIL_STREAK_MAX = 3

_TAG_RE = re.compile(r"<[^>]+>")
_MD_RE = re.compile(r"(\*\*|__|^#+\s*|\$+)", re.MULTILINE)
_QI: dict = {}


def emit(enabled, phase, **kw):
    if enabled:
        print(PROGRESS_PREFIX + json.dumps({"v": PROGRESS_SCHEMA, "phase": phase, **_QI, **kw},
                                           ensure_ascii=False), flush=True)


# ── текст сторінки ───────────────────────────────────────────────────────────
def _foreign_share(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    bad = 0
    for ch in letters:
        name = unicodedata.name(ch, "")
        if any(name.startswith(s) or (" %s" % s) in name for s in _FOREIGN):
            bad += 1
    return bad / len(letters)


def clean_text(text):
    """Текст блоку без розмітки: теги таблиць, `**`, `#`, `$`; без рядків чужим письмом."""
    text = _TAG_RE.sub(" ", text or "")
    text = _MD_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line and _foreign_share(line) <= 0.3:
            lines.append(line)
    return "\n".join(lines)


def _key(text):
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def blocks_of(res_json):
    """`res.json` рушія → (блоки в порядку читання, ширина)."""
    res = res_json.get("res", res_json) if isinstance(res_json, dict) else {}
    width = int(res.get("width") or 0)
    out = []
    for b in res.get("parsing_res_list") or []:
        bbox = [int(v) for v in (b.get("block_bbox") or [0, 0, 0, 0])][:4]
        out.append({"label": str(b.get("block_label") or ""),
                    "text": str(b.get("block_content") or ""),
                    "bbox": bbox + [0] * (4 - len(bbox)),
                    "order": b.get("block_order")})
    return out, width


def page_text(blocks, width):
    """Чистий текст сторінки й прийняті блоки.

    Порядок — той, що дав рушій. Відкидається нетекстове, блок у крайній смузі
    кадру, повтор уже прийнятого блоку (рамки, що перекриваються, рушій читає
    двічі) і рядки чужим письмом.
    """
    paras, kept, seen = [], [], []
    for b in blocks:
        if b["label"] in SKIP_LABELS:
            continue
        x0, _, x1, _ = b["bbox"]
        if width and (x1 <= width * EDGE_STRIP or x0 >= width * (1 - EDGE_STRIP)):
            continue
        text = clean_text(b["text"])
        if not text:
            continue
        key = _key(text)
        if len(key) >= DUP_MIN_CHARS and any(key in s for s in seen):
            continue
        seen.append(key)
        paras.append(text)
        kept.append({"label": b["label"], "bbox": b["bbox"], "order": b["order"],
                     "heading": b["label"] in HEADING_LABELS, "text": text})
    return ("\n\n".join(paras) + "\n") if paras else "", kept


def _markdown(res):
    md = getattr(res, "markdown", None)
    if isinstance(md, dict):
        return str(md.get("markdown_texts") or "")
    return str(md or "")


def write_page(out_dir, stem, res_json, md):
    """Виходи однієї сторінки. txt — останнім: повнота рахує саме його."""
    blocks, width = blocks_of(res_json)
    text, kept = page_text(blocks, width)
    (out_dir / ("%s.md" % stem)).write_text(md, encoding="utf-8")
    (out_dir / ("%s.blocks.json" % stem)).write_text(
        json.dumps({"width": width, "blocks": kept}, ensure_ascii=False), encoding="utf-8")
    (out_dir / ("%s.txt" % stem)).write_text(text, encoding="utf-8")
    return {"lines": sum(1 for ln in text.splitlines() if ln.strip()),
            "chars": len(text), "blocks": len(kept)}


# ── облік справи ─────────────────────────────────────────────────────────────
def _atomic_write(path, text):
    tmp = path.with_name("%s.%d.tmp" % (path.name, os.getpid()))
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_quarantine(out_dir):
    try:
        data = json.loads((Path(out_dir) / QUARANTINE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    pages = data.get("pages")
    return pages if isinstance(pages, dict) else {}


def add_quarantine(out_dir, page, reason):
    pages = load_quarantine(out_dir)
    pages[page] = {"reason": reason[:300], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _atomic_write(Path(out_dir) / QUARANTINE,
                  json.dumps({"pages": pages}, ensure_ascii=False, indent=1))


def frames_of(case_dir):
    """Кадри так, як їх бачить бокс-раннер (`_script_pages`): iterdir, за іменем."""
    return sorted(p for p in Path(case_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in FRAME_EXT)


def server_alive(url, timeout=10):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class ServerDown(RuntimeError):
    """vLLM-сервер не відповідає: шард виходить, рішення — за бокс-раннером."""


#: Скільки сторінок кількох справ іде в рушій ОДНИМ переліком. Номер газети —
#: дві сторінки: окремий `predict` на кожен спорожнював конвеєр Paddle
#: (розкладка ↔ VLM) щоразу, і 16.10.2026 номер коштував 10–22 с замість ~8.
#: Стеля — щоб `case_done` (а з ним чекпоінт і закриття справи) не чекав
#: на довгий пакет.
BATCH_PAGES = 24


class CaseJob:
    """Відкрита справа робітника: що читати, облік і коди виходу."""

    def __init__(self, args, case_dir, out_dir, case_key="", qi=None, gone=False):
        self.args, self.qi, self.gone = args, qi, gone
        self.done = self.skipped = self.failed = 0
        self.todo, self.pages, self.n = [], [], 0
        if gone:
            # Теку справи прибрали (раннер уже закрив її сам): у переліку вона
            # лише тримає своє місце, щоб `case_done` ішли строго по черзі.
            return
        self.case_dir, self.out_dir = Path(case_dir), Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prog = args.progress_json
        pages = frames_of(self.case_dir)
        self.frames_total = len(pages)
        if args.limit:
            pages = pages[:args.limit]
        self.pages = pages
        self.n = len(pages)
        self.meta_path = self.out_dir / "_htr_meta.json"
        self.meta = {"version": 1, "case_dir": str(self.case_dir), "case_key": case_key,
                     "frames_total": self.frames_total, "model": args.model,
                     "device": "vllm", "engine": "print", "script": "print",
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "done": False, "failed": [], "pages": {}}
        if not pages:
            return
        if self.meta_path.exists():
            try:
                old = json.loads(self.meta_path.read_text(encoding="utf-8"))
                self.meta["pages"] = old.get("pages") or {}
                self.meta["failed"] = old.get("failed") or []
                self.meta["started"] = old.get("started") or self.meta["started"]
            except (OSError, ValueError):
                pass
        quarantined = load_quarantine(self.out_dir)
        for i, src in enumerate(pages, 1):
            if (self.out_dir / ("%s.txt" % src.stem)).exists() and src.name in self.meta["pages"]:
                self.skipped += 1
                self.emit("htr", i=i, n=self.n, page=src.name, skipped=True)
            elif src.name in quarantined:
                self.failed += 1
                reason = (quarantined[src.name] or {}).get("reason") or "вішала прогін"
                self.emit("htr", i=i, n=self.n, page=src.name, error="карантин: %s" % reason)
            else:
                self.todo.append((i, src))

    def emit(self, phase, **kw):
        if self.qi is not None:
            kw.setdefault("qi", self.qi)
        emit(self.prog, phase, **kw)

    def save(self):
        self.meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _atomic_write(self.meta_path, json.dumps(self.meta, ensure_ascii=False, indent=1))

    def page_done(self, i, src, res, t):
        rec = write_page(self.out_dir, src.stem, dict(res.json), _markdown(res))
        sec = round(time.time() - t, 2)
        self.meta["pages"][src.name] = dict(sec=sec, **rec)
        self.done += 1
        self.save()
        self.emit("htr", i=i, n=self.n, page=src.name, lines=rec["lines"], sec=sec)

    def page_failed(self, i, src, exc):
        self.failed += 1
        add_quarantine(self.out_dir, src.name, "print-vl: %s" % exc)
        if src.name not in self.meta["failed"]:
            self.meta["failed"].append(src.name)
        self.emit("htr", i=i, n=self.n, page=src.name, error=str(exc)[:300])
        print("[print-vl] ✖ %s: %s" % (src.name, exc), flush=True)

    def finish(self):
        """Підсумок справи. 0 — повно, 3 — неповно (догонить повтор), 1 — немає кадрів."""
        if self.gone:
            return 0
        if not self.pages:
            print("[print-vl] у теці немає кадрів jpg/jpeg/png", flush=True)
            self.emit("done", pages=0, skipped=0, failed=0, error="немає сторінок")
            return 1
        quar = load_quarantine(self.out_dir)
        gone = [p.name for p in self.pages
                if not (self.out_dir / ("%s.txt" % p.stem)).exists() and p.name not in quar]
        self.meta["missing"] = gone
        self.meta["quarantined"] = sorted(quar)
        self.meta["done"] = not gone
        self.save()
        self.emit("done", pages=self.done, skipped=self.skipped, failed=self.failed,
                  missing=len(gone))
        if gone:
            print("[print-vl] ⚠ неповно: без тексту %d з %d кадрів" % (len(gone), self.n),
                  flush=True)
            return 3
        return 0


def read_jobs(args, pipe_box, jobs, on_finish=None):
    """Сторінки КІЛЬКОХ справ одним переліком у рушій; справа закривається, щойно
    записано її останню сторінку. Повертає коди справ у тому самому порядку.

    Розкладка наступної сторінки рахується, поки VLM читає блоки попередньої —
    і тепер це діє й через межу справ, а не лише всередині двох сторінок номера.
    """
    work = [(job, i, src) for job in jobs for i, src in job.todo]
    left = {id(job): len(job.todo) for job in jobs}
    codes = {}
    head = [0]

    def close_ready():
        # 🔴 Строго з голови переліку: раннер бере `qi_done` як МАКСИМУМ, тож
        # `case_done` справи 3 раніше за 1 і 2 оголосив би закритими справи,
        # які ще читаються.
        while head[0] < len(jobs) and not left[id(jobs[head[0]])]:
            job = jobs[head[0]]
            codes[id(job)] = job.finish()
            if on_finish is not None:
                on_finish(job, codes[id(job)])
            head[0] += 1

    close_ready()                          # справи, яким нічого читати, на початку
    if work:
        if pipe_box.get("pipe") is None:
            pipe_box["pipe"] = pipe_box["make"]()
        pipe = pipe_box["pipe"]
        it = iter(pipe.predict([str(src) for _, _, src in work], use_queues=True))
        streak = 0
        for k, (job, i, src) in enumerate(work):
            job.emit("page_start", i=i, n=job.n, page=src.name)
            t = time.time()
            try:
                res = next(it)
                job.page_done(i, src, res, t)
                streak = 0
            except StopIteration:
                job.failed += 1
                job.emit("htr", i=i, n=job.n, page=src.name, error="рушій не віддав сторінку")
            except Exception as exc:
                streak += 1
                if streak >= FAIL_STREAK_MAX or not server_alive(args.vl_server):
                    for j in jobs:
                        if id(j) not in codes:
                            j.save()
                    raise ServerDown("vLLM не відповідає після %d збоїв поспіль: %s"
                                     % (streak, exc)) from exc
                job.page_failed(i, src, exc)
                # Генератор після винятку мертвий: читаємо СТРОГО решту переліку.
                rest = [str(s) for _, _, s in work[k + 1:]]
                it = iter(pipe.predict(rest, use_queues=True) if rest else [])
            left[id(job)] -= 1
            close_ready()
    close_ready()
    return [codes[id(job)] for job in jobs]


def run_case(args, pipe_box, case_dir, out_dir, case_key=""):
    """Одна справа. Коди: 0 — повно, 3 — неповно (догонить повтор), 1 — немає кадрів."""
    job = CaseJob(args, case_dir, out_dir, case_key, qi=_QI.get("qi"))
    return read_jobs(args, pipe_box, [job])[0]


def make_pipeline(server_url):
    """Конвеєр PaddleOCR-VL, що читає блоки через vLLM-сервер."""
    from paddleocr import PaddleOCRVL

    return PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url=server_url,
                       use_doc_orientation_classify=False, use_doc_unwarping=False)


def run_queue(args, pipe_box=None):
    """Справи з файла черги одним процесом: конвеєр Paddle вантажиться раз.

    Усі справи, що вже чекають у черзі, беруться пакетом до `BATCH_PAGES`
    сторінок і йдуть у рушій одним переліком (`read_jobs`); `case_done` кожної —
    щойно записано її останню сторінку, тож бокс-раннер закриває справи в тому
    самому порядку й темпі, що й доти, лише без простою між ними.
    """
    qfile = Path(args.queue)
    pipe_box = pipe_box or {"pipe": None, "make": lambda: make_pipeline(args.vl_server)}
    pos, worst, last_beat, ended = 0, 0, 0.0, False
    while not ended:
        try:
            lines = [ln for ln in qfile.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except OSError:
            lines = []
        if pos >= len(lines):
            if time.time() - last_beat >= 20:
                emit(args.progress_json, "queue_wait", done_cases=pos)
                last_beat = time.time()
            time.sleep(0.5)
            continue
        jobs, pages = [], 0
        while pos < len(lines) and pages < BATCH_PAGES:
            try:
                entry = json.loads(lines[pos])
            except ValueError:
                break                      # рядок ще дописується — заберемо наступним колом
            pos += 1
            if entry.get("end"):
                ended = True
                break
            qi = int(entry.get("index") or pos)
            job = CaseJob(args, entry["case_dir"], entry["out_dir"],
                          str(entry.get("case_key") or ""), qi=qi,
                          gone=not Path(str(entry["case_dir"])).is_dir())
            jobs.append(job)
            pages += len(job.todo)
        if not jobs:
            if not ended and pos < len(lines):
                time.sleep(0.2)
            continue

        def finished(job, rc):
            emit(args.progress_json, "case_done", qi=job.qi, rc=rc)

        for rc in read_jobs(args, pipe_box, jobs, on_finish=finished):
            if rc not in (0, 3):
                worst = rc
    emit(args.progress_json, "queue_done", cases=pos, drained=False)
    return worst


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="робітник друку (PaddleOCR-VL через vLLM)")
    ap.add_argument("--queue", required=True)
    ap.add_argument("--progress-json", action="store_true")
    ap.add_argument("--vl-server", default=DEFAULT_SERVER)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    try:
        return run_queue(args)
    except ServerDown as exc:
        print("[print-vl] 🔴 %s" % exc, flush=True)
        return 4


if __name__ == "__main__":
    sys.exit(main())
