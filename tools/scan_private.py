#!/usr/bin/env python3
"""🔒 Ворота проти приватних даних у публічному репозиторії.

Цей інструмент виріс поруч із приватним дослідницьким репо однієї родини й
роками ганяв саме його справи. Небезпека тут не в тому, що хтось свідомо
закомітить приватне, а в тому, що це станеться ВИПАДКОВО: один `git add -A` у
розгоні, приклад у докстрінгу зі шляхом `E:\\Projects\\MeGen\\...`, дефолтне
ім'я бакета з чужого акаунта, реєстр орендованих машин поруч із кодом.

Ворота перенесено з пакета `nyshporka` (`tools/scan_private.py`) — правила ті
самі, винятки свої.

🔴 Головне: git не забуває. Файл, закомічений і видалений наступним комітом,
лишається в історії назавжди, і «прибрати» його означає переписати історію вже
опублікованого репозиторію. Тому перевірка мусить стояти ДО коміту, а не після.

    python tools/scan_private.py                # робоче дерево
    python tools/scan_private.py --staged       # те, що зараз у git add (pre-commit)
    python tools/scan_private.py --history      # УСЯ історія (перед першим push)

Вихід 0 — чисто, 1 — знайдено, 2 — помилка запуску.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Розширення, які взагалі має сенс читати як текст.
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini",
    ".html", ".css", ".js", ".ts", ".sh", ".ps1", ".jsonl", ".tsv", ".csv", "",
}
#: Каталоги, у які не заходимо ніколи.
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
             ".ruff_cache", ".pytest_cache", "dist", "build", ".idea", ".vscode"}
#: 🔴 Імена, які бувають ФАЙЛОМ, а не текою. `SKIP_DIRS` підрізає лише обхід
#: каталогів, тож `.git` у git-worktree і в субмодулі (там це файл із рядком
#: `gitdir: <абсолютний шлях>`) проходив повз і давав ХИБНУ тривогу на правило
#: абсолютного шляху. Ворота, які кричать на службовий файл git, привчають
#: розробника відмахуватись від них — а це рівно те, чого вони мають не
#: допустити. Розширення тут не рятує: `Path(".git").suffix` порожній, а
#: порожній суфікс у `TEXT_SUFFIXES` є навмисно (LICENSE, Dockerfile).
SKIP_NAMES = {".git"}
MAX_BYTES = 2_000_000


@dataclass(frozen=True)
class Rule:
    id: str
    why: str
    pattern: re.Pattern[str]


def _rx(p: str) -> re.Pattern[str]:
    return re.compile(p, re.IGNORECASE)


RULES: tuple[Rule, ...] = (
    # ── ідентифікатори канону ────────────────────────────────────────────────
    # Особи/родини/місця приватного дослідження. Саме вони найлегше приїжджають
    # разом зі скопійованим прикладом чи тестовою фікстурою.
    Rule("canon-person", "ID особи з приватного канону (I0123)",
         re.compile(r"\bI0\d{3}\b")),
    Rule("canon-family", "ID родини з приватного канону (F0123)",
         re.compile(r"\bF0\d{3}\b")),
    Rule("canon-place", "ID місця з приватного канону (PL0123)",
         re.compile(r"\bPL0\d{3}\b")),
    Rule("canon-source", "ID джерела з приватного канону (S_DAHMO_F315_...)",
         re.compile(r"\bS_[A-Z]{2,}_F\d+")),

    # ── прізвище роду в усіх написаннях ──────────────────────────────────────
    # 🔴 Не одна форма, а корінь із варіантами: декод і транслітерація дають
    # десяток написань, і фільтр на єдине «Долищинський» пропустив би більшість.
    # ⚠ Перша редакція вимагала `c`/`s` після кореня (`doli[sș][cs]`) і через це
    # не бачила голого румунського `doliş` — тобто саме тієї форми, заради якої
    # варіанти й перелічуються.
    Rule("clan-surname", "прізвище роду з приватного дослідження",
         _rx(r"д[оаі]л[иіы]щ[иі]н|d[oa]li[sșş]")),

    # 🔴 СПОТВОРЕНІ ФОРМИ того самого прізвища — окреме правило, бо в жодній з
    # них немає складу, за яким ловить попереднє: рушій калічить саме СЕРЕДИНУ
    # слова. Ці форми виглядають як шум машинного читання, а насправді це
    # перелік того, що шукає одна родина, — і саме тому вони найлегше проїдуть
    # повз ока в скопійованому прикладі чи в тексті інструкції для агента.
    # ⚠ Свідомо НЕ ловляться «Доманський» і «Долинський»: обидва — поширені
    # самостійні прізвища, і правило на них заважало б чужому дослідженню в
    # пакеті, яким користуються інші люди. Ціна пропуску тут менша за ціну
    # правила, що бореться з користувачем.
    Rule("clan-misread", "спотворена форма прізвища роду (як її калічить рушій)",
         _rx(r"дом[иі]н[сc]к|дем[иі]ц[иі]н|дон[иі]ц[иі]н|домб[иі]н|дом[іи]ан[сc]к")),

    # ── особистий контакт ────────────────────────────────────────────────────
    # 🔴 Пошта дослідника — не «майже те саме», що прізвище: вона їде В КОЖНОМУ
    # HTTP-ЗАПИТІ, якщо потрапила в User-Agent, і осідає в логах чужих сайтів,
    # звідки її вже не прибрати. Знайдено при перенесенні завантажувачів: два
    # скрипти несли адресу автора в UA, і жодне з наявних правил її не бачило —
    # перевірено підкладеним файлом, ворота пройшли повз.
    # ⚠ Ловиться будь-яка адреса, а не одна конкретна: у пакеті, який ставлять
    # чужі люди, зашита особиста пошта — завжди питання, чия вона й навіщо там.
    # Законний виняток — файли, де адреса є ДАНИМИ (контакт архіву в доці).
    Rule("contact-email", "особиста поштова адреса в коді",
         _rx(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b")),

    # ── локальні шляхи ───────────────────────────────────────────────────────
    # Абсолютний шлях машини автора — не секрет, але він робить код непереносним
    # і видає структуру приватного архіву.
    # 🔴 Перелік тек тут НЕ вичерпний, і саме тому друга альтернатива ловить
    # будь-який `<літера>:/megen*`. Спіймано перед першим push: приклади в
    # докстрінгах несли `E:/megen_stage/…` і `T:/megen_spotter_out/…` — обидва
    # проходили повз, бо після літери диска стояла не «Projects» і не «Temp».
    # Правило, яке перелічує ЗНАЙОМІ випадки, ловить лише знайомі.
    Rule("abs-path-win", "абсолютний шлях Windows із машини автора",
         _rx(r"[A-Z]:[\\/](?:Projects|Users|Temp|megen[\w-]*)[\\/]")),
    Rule("abs-path-nix", "абсолютний шлях із чужої машини/VPS",
         re.compile(r"(?:^|[\"'\s=])/(?:root|home)/[a-z0-9_.-]+/")),
    # 🔴 Ловиться АДРЕСА, а не назва. Рішення 21.09.2026: саме слово «MeGen» —
    # ім'я проєкту, з якого цей інструмент виріс, і воно нічого не видає.
    # Ловити його означало б вичистити 405 місць у докстрінгах, де «Кейс
    # MeGen: …» пояснює ГОЛОВНЕ — чому код саме такий; пояснення без імені
    # кейса знецінюється, а користі від заміни немає жодної.
    # Приватне тут — адреса чужого сховища й імена приватних просторів: за
    # ними людину знаходять, за назвою проєкту — ні.
    # ⚠ Але ІДЕНТИФІКАТОР із цим коренем (`megen-spotter`, `megen_archive`,
    # `megen_htr_cloud`) лишається приватним: у публічному пакеті дефолтне ім'я
    # тому чи бакета читається як «так і треба», і чужа людина заводить у себе
    # том із назвою чужого роду. Тому ловимо `megen` із дефісом чи
    # підкресленням, а голе слово в реченні — ні.
    Rule("private-repo", "адреса приватного сховища дослідження",
         _rx(r"megen[-_][a-z0-9][\w-]*|SERGIUSH-UA/domus|gdrive:MeGen")),
    # Простори сусідніх приватних досліджень: у докстрінгах вони стояли як
    # приклад «куди лягає вивід», а назва простору — це назва чийогось роду.
    Rule("private-space", "ім'я приватного простору дослідження",
         _rx(r"rodovid-shupyky|knyha-rodu")),

    # ── секрети ──────────────────────────────────────────────────────────────
    Rule("aws-presigned", "presigned-URL з креденшелом (X-Amz-...)",
         _rx(r"X-Amz-(?:Credential|Signature|Security-Token)")),
    Rule("bearer", "захардкоджений токен/ключ",
         _rx(r"(?:api[_-]?key|secret|password|token)\s*[:=]\s*[\"'][A-Za-z0-9_\-]{16,}")),
    Rule("private-host", "приватний хост/тунель автора",
         _rx(r"easykey-backup|itdeo\.tech")),
)

#: Свідомі винятки: файл (glob) → які правила там дозволені.
#: 🔴 Виняток завжди ТОЧКОВИЙ — правило × шлях. Глобальне «ігнорувати цей файл»
#: перетворює ворота на декорацію: наступна людина допише туди що завгодно.
ALLOW: tuple[tuple[str, str], ...] = (
    # Сам сканер містить усі патерни за визначенням.
    ("tools/scan_private.py", "*"),
    # Тест воріт мусить містити зразки того, що вони ловлять.
    ("tests/test_scan_private.py", "*"),
    # Адреса для повідомлень про вразливості — публічна за призначенням.
    ("SECURITY.md", "contact-email"),
    # 🔴 Атрибуція автора — НЕ приватні дані, хоч і збігається з прізвищем, яке
    # шукає сусіднє дослідження. Межа проходить по ролі, а не по слову: ім'я в
    # метаданих пакета й у ліцензії публічне за призначенням, те саме прізвище
    # всередині КОДУ (дефолт параметра, ім'я класу детектора, фікстура) —
    # ознака, що сюди приїхав шматок приватного конвеєра.
    ("pyproject.toml", "clan-surname"),
    ("LICENSE", "clan-surname"),
    # `jovyan` — стандартний користувач образів Jupyter, під яким Saturn Cloud
    # піднімає ресурс; це адреса на ЇХНІЙ машині, а не на чиїйсь особистій.
    ("src/gpurunner/backends/saturn.py", "abs-path-nix"),
    # Коментар пояснює, ЧОМУ ім'я файла береться без query: там назва параметра
    # підпису, а не сам підпис. Значення після `=` у рядку немає.
    ("src/gpurunner/_embedded/htr_case_runner.py", "aws-presigned"),
)


def _allowed(rel: str, rule_id: str) -> bool:
    rel = rel.replace("\\", "/")
    for pat, allowed in ALLOW:
        if (rel == pat or Path(rel).match(pat)) and allowed in ("*", rule_id):
            return True
    return False


@dataclass
class Finding:
    path: str
    line_no: int
    rule: Rule
    excerpt: str


def scan_text(rel: str, text: str, where: str | None = None) -> list[Finding]:
    """`rel` — шлях ДЛЯ ЗВІРКИ з винятками, `where` — підпис для показу.

    🔴 Ці дві речі мусять бути розділені, і це не педантизм. Перша редакція
    режиму `--history` віддавала шлях у вигляді `tests/foo.py @26d20e22`, щоб у
    виводі було видно коміт, — і через приліплений sha жоден виняток не
    збігався. Ворота при цьому не мовчали, а навпаки: сипали 27 «знахідками» у
    власних тестах, тобто в CI стояли б вічно червоними. Ворота, які завжди
    червоні, вимикають — і тоді вони не ловлять уже нічого.
    """
    out: list[Finding] = []
    label = where or rel
    # 🔴 Винятки залежать від ФАЙЛА, не від рядка, тож рахуються один раз. У
    # першій редакції `_allowed` стояв усередині подвійного циклу й конструював
    # `Path` для glob-матчингу на кожну пару (рядок × правило): на історії це
    # 1.65 млн викликів і 49 секунд там, де роботи на секунду.
    rules = [r for r in RULES if not _allowed(rel, r.id)]
    if not rules:
        return out
    for i, line in enumerate(text.splitlines(), 1):
        for rule in rules:
            m = rule.pattern.search(line)
            if m:
                frag = line.strip()
                if len(frag) > 120:
                    lo = max(0, m.start() - 50)
                    frag = ("…" if lo else "") + line[lo:lo + 120].strip() + "…"
                out.append(Finding(label, i, rule, frag))
    return out


def _git(*args: str) -> str:
    res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {res.stderr.strip()}")
    return res.stdout


#: (шлях для звірки з винятками, текст, підпис для показу)
Item = tuple[str, str, str]


def _ignored() -> set[str]:
    """Шляхи, які git ігнорує — тобто ті, що в репозиторій не потраплять НІКОЛИ.

    ⚠ Ворота від цього не слабшають: те, що справді може поїхати в коміт,
    перевіряють `iter_staged()` (індекс) і режим `--history` (уся історія), а
    ігнорований файл не проходить ні там, ні там. Натомість без цього кроку
    особистий чернетковий файл у корені робить прогін воріт вічно червоним — а
    ворота, які завжди червоні, перестають означати «щось не так».

    Порожня множина, якщо git недоступний або це не репозиторій: не пропускаємо
    нічого, тобто помиляємось у бік перевірки.
    """
    try:
        out = _git("ls-files", "--others", "--ignored", "--exclude-standard", "-z")
    except (RuntimeError, OSError):
        return set()
    return {n for n in out.split("\0") if n}


def iter_worktree() -> list[Item]:
    """🔴 `SKIP_DIRS` підрізає ОБХІД, а не відсіює результат.

    `rglob("*")` із перевіркою після факту все одно заходить усередину `.venv` і
    `node_modules` — а це десятки тисяч файлів, тобто 34 секунди на кожен запуск
    воріт, які стоять у pre-commit. Тут дешевше не «не читати», а «не заходити».
    """
    out: list[Item] = []
    skip = _ignored()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        base = Path(dirpath)
        for fn in filenames:
            if fn in SKIP_NAMES:
                continue
            p = base / fn
            if p.relative_to(ROOT).as_posix() in skip:
                continue
            if p.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if p.stat().st_size > MAX_BYTES:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = p.relative_to(ROOT).as_posix()
            out.append((rel, text, rel))
    return out


def iter_staged() -> list[Item]:
    """Вміст, який ЗАРАЗ у індексі — саме він потрапить у коміт.

    Читаємо з `git show :file`, а не з диска: інакше перевірка дивилась би на
    робоче дерево, тоді як закомітиться індекс, і `git add -p` пройшов би повз.
    """
    names = [n for n in _git("diff", "--cached", "--name-only", "-z").split("\0") if n]
    out: list[Item] = []
    for rel in names:
        if Path(rel).suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            out.append((rel, _git("show", f":{rel}"), rel))
        except RuntimeError:
            continue  # видалений файл
    return out


def _git_bytes(*args: str, stdin: bytes = b"") -> bytes:
    res = subprocess.run(["git", *args], cwd=ROOT, input=stdin, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {res.stderr.decode(errors='replace')}")
    return res.stdout


def _batch_read(shas: list[str]) -> dict[str, bytes]:
    """Вміст блобів одним викликом `git cat-file --batch`.

    🔴 Не мікрооптимізація. Наївний варіант (`cat-file -s` + `cat-file -p` на
    кожен об'єкт) — це два процеси на блоб, і на Windows, де запуск процесу
    коштує дорого, весь режим `--history` займав 171 секунду. Тест, який стільки
    думає, розробник вимикає — а це рівно ті ворота, ціна пропуску яких
    незворотна.

    Формат `--batch`: рядок `<sha> <type> <size>`, далі рівно `size` байтів
    вмісту і `\\n`. Читаємо саме за лічильником, а не за роздільником: у
    текстовому файлі трапляється будь-що, і розбір «до наступного порожнього
    рядка» тихо з'їхав би на першому ж файлі з такою послідовністю.
    """
    if not shas:
        return {}
    buf = _git_bytes("cat-file", "--batch", stdin=("\n".join(shas) + "\n").encode())
    out: dict[str, bytes] = {}
    i = 0
    while i < len(buf):
        nl = buf.find(b"\n", i)
        if nl < 0:
            break
        head = buf[i:nl].decode(errors="replace").split()
        i = nl + 1
        if len(head) != 3:          # `<sha> missing` — об'єкт зник між викликами
            continue
        sha, size = head[0], int(head[2])
        out[sha] = buf[i:i + size]
        i += size + 1               # +1 — перевід рядка після вмісту
    return out


#: Гілка, яку публікують. Її історію й перевіряємо, коли питають про публікацію:
#: приватна історія розробки лишається приватною за рішенням автора
#: (21.09.2026), і вимагати від неї чистоти означало б або переписувати 136
#: комітів від самого кореня, або тримати збірку вічно червоною.
PUBLIC_REF_ENV = "GPURUNNER_PUBLIC_REF"
PUBLIC_REF_DEFAULT = "public"


def public_ref() -> str:
    """Реф публікації, якщо він у цьому клоні є; інакше порожньо."""
    named = os.environ.get(PUBLIC_REF_ENV, "").strip() or PUBLIC_REF_DEFAULT
    try:
        out = subprocess.run(["git", "rev-parse", "--verify", "--quiet",
                              f"{named}^{{commit}}"], cwd=ROOT,
                             capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return named if out.returncode == 0 else ""


def iter_history(ref: str = "") -> list[Item]:
    """Усі версії всіх текстових файлів в історії.

    Потрібно перед кожним `push`: після нього прибрати знахідку означає
    переписати опубліковану історію. Один об'єкт може лежати під кількома
    іменами — беремо перше, бо виняток звіряється зі шляхом.

    `ref` — гілка, чию історію дивимось; порожньо — УСЕ, що є в репозиторії
    (`--all`), найсуворіший режим і дефолт для виклику руками.
    """
    scope = [ref] if ref else ["--all"]
    names: dict[str, str] = {}
    for line in _git("rev-list", "--objects", *scope).splitlines():
        sha, _, name = line.partition(" ")
        if not name or Path(name).suffix.lower() not in TEXT_SUFFIXES:
            continue
        names.setdefault(sha, name)
    # Розміри — окремим пакетом: `--batch-check` віддає лише заголовки, тож
    # величезний блоб не доводиться тягти в пам'ять, щоб дізнатись, що він завеликий.
    small: list[str] = []
    head = _git_bytes("cat-file", "--batch-check",
                      stdin=("\n".join(names) + "\n").encode()).decode(errors="replace")
    for line in head.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "blob" and int(parts[2]) <= MAX_BYTES:
            small.append(parts[0])
    # Шлях для звірки — чистий; коміт-об'єкт іде лише в підпис. Змішавши їх, ми
    # зробили б винятки недієвими саме тут (див. `scan_text`).
    return [(names[sha], blob.decode("utf-8", errors="replace"), f"{names[sha]} @{sha[:8]}")
            for sha, blob in _batch_read(small).items()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--staged", action="store_true", help="лише те, що в git add")
    g.add_argument("--history", action="store_true", help="уся історія (перед першим push)")
    g.add_argument("--public", action="store_true",
                   help="історія ГІЛКИ ПУБЛІКАЦІЇ — тієї, що поїде людям")
    ap.add_argument("--list-rules", action="store_true", help="показати правила й вийти")
    a = ap.parse_args()

    if a.list_rules:
        for r in RULES:
            print(f"  {r.id:16s} {r.why}")
        return 0

    try:
        if a.staged:
            items, what = iter_staged(), "індекс"
        elif a.public:
            ref = public_ref()
            if not ref:
                print(f"немає гілки публікації (`{PUBLIC_REF_DEFAULT}` або "
                      f"${PUBLIC_REF_ENV}) — нічого перевіряти")
                return 0
            items, what = iter_history(ref), f"історія гілки {ref}"
        elif a.history:
            items, what = iter_history(), "історія"
        else:
            items, what = iter_worktree(), "робоче дерево"
    except RuntimeError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2

    findings: list[Finding] = []
    for rel, text, where in items:
        findings += scan_text(rel, text, where)

    if not findings:
        print(f"✅ приватних даних не знайдено ({what}: {len(items)} файлів, "
              f"{len(RULES)} правил)")
        return 0

    print(f"🔴 ЗНАЙДЕНО ПРИВАТНІ ДАНІ ({what}) — {len(findings)} збігів:\n")
    by_rule: dict[str, list[Finding]] = {}
    for f in findings:
        by_rule.setdefault(f.rule.id, []).append(f)
    for rid, group in sorted(by_rule.items()):
        print(f"  ▸ {rid} — {group[0].rule.why}  ({len(group)})")
        for f in group[:8]:
            print(f"      {f.path}:{f.line_no}  {f.excerpt}")
        if len(group) > 8:
            print(f"      … ще {len(group) - 8}")
    print("\nЩо робити: прибрати дані або, якщо це свідомий приклад, додати "
          "ТОЧКОВИЙ виняток у `ALLOW` (правило × шлях, не «ігнорувати файл»).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
