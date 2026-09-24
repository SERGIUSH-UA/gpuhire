"""Скласти план хмарного заходу: спакувати кадри, залити в R2, нарізати посилання.

🔴🔴 Тека виводу приходить ЗЗОВНІ й лягає в план абсолютним шляхом. Доти
складач жив скриптом у репозиторії одного дослідження й рахував її від СВОГО
кореня, тож справи із сусідніх просторів розкладались у той репозиторій. За одну
кампанію це спрацювало п'ять разів; двічі виглядало як «робота втрачена», хоч
на диску лежало 1101 і 441 готова сторінка — просто в чужому проєкті. Тут
кореня немає взагалі, і `out_root` без значення означає відмову, а не здогад.
"""

from __future__ import annotations

import io
import json
import re
import statistics
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpurunner.core.offer_score import DEFAULT_TARGET_PPH
from gpurunner.htr import r2

IMG_SUFFIXES = (".jpg", ".jpeg", ".png")

#: Скільки чекпоінт-посилань нарізати на справу. Кожне придатне лише для свого
#: ключа, тож це стеля числа чекпоінтів; вичерпання = мовчазна втрата хвоста.
#:
#: 🔴 Число мусить покривати ВСЮ стелю часу, а не абстрактну «достатню»
#: кількість. При чекпоінті раз на 120 с шістдесят посилань вичерпуються рівно
#: за 2 години — тобто на 8-годинному заході останні шість годин не мали точки
#: відновлення ЗОВСІМ, і дізнатись про це можна було лише з лога на боксі.
CKPT_SEC_DEFAULT = 120
CKPT_URLS_MIN = 60

#: Скільки сторінок покриває ОДИН чекпоінт і яка підлога слотів на справу.
#:
#: 🔴🔴 Найдорожча дрібниця всього заходу. Слоти рахувались від стелі часу
#: ЗАХОДУ (14 год ⇒ 314) і так — НА КОЖНУ справу. На черзі з 194 справ це
#: 121 832 presigned-посилання, вшитих у `job.py`: **48 МБ** джерела, яке потім
#: їде на машину paramiko-SFTP синхронними шматками по 32 КБ — півтори тисячі
#: кругових обертів через океан. Заміряно 23.09.2026: підйом боксу 1644 с, із
#: них 25 хвилин пішло саме на цей файл, і в лозі це виглядало як тиша.
#:
#: Правильна міра — РОЗМІР СПРАВИ: справа на 24 сторінки читається хвилини й
#: потребує двох-трьох точок, а не трьохсот. Підлога лишається щедрою, бо
#: вичерпані слоти означають мовчазну втрату хвоста при смерті боксу.
CKPT_PAGES_PER_SLOT = 100
CKPT_URLS_MIN_PER_CASE = 8

#: Транслітерація для імені справи. 🔴 Кирилицю саме ПЕРЕКЛАДАЄМО, а не
#: викидаємо: доти кожна кирилична літера ставала дефісом, і `230-1-2`,
#: `230-1-2а`, `230-1-2б` (три РІЗНІ томи) давали один слуг, тобто спільний
#: префікс у R2 і спільну теку виводу. Три декоди злились би в один — і це не
#: впало б, а тихо дало б чужий текст під правильним шифром.
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e",
    "є": "ie", "ж": "zh", "з": "z", "и": "y", "і": "i", "ї": "i", "й": "i",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "shch", "ь": "", "ю": "iu", "я": "ia", "ы": "y", "э": "e",
    "ъ": "", "ё": "e",
}


def ckpt_urls_for(max_hours: float, ckpt_sec: int = CKPT_SEC_DEFAULT,
                  pages: int = 0) -> int:
    """Скільки посилань нарізати справі.

    Без `pages` — як було: від стелі часу ЗАХОДУ (× запас 1.3). Це правильно для
    одної справи й катастрофічно для черги: кожна справа діставала слоти на весь
    захід, і `job.py` розпухав до 48 МБ (194 справи × 628 посилань), а потім їхав
    на машину 25 хвилин.

    З `pages` міра стає власною: одна точка на ~100 сторінок плюс запас, але не
    менше за підлогу й не більше за часову стелю. Вичерпані слоти = мовчазна
    втрата хвоста при смерті боксу, тому підлога щедра, а стеля часу лишається
    межею зверху.
    """
    by_time = int(max_hours * 3600 / max(30, ckpt_sec) * 1.3) + 2
    if pages <= 0:
        return max(CKPT_URLS_MIN, by_time)
    by_pages = -(-int(pages) // CKPT_PAGES_PER_SLOT) + 2
    return max(CKPT_URLS_MIN_PER_CASE, min(by_pages, max(CKPT_URLS_MIN, by_time)))


#: Імена тек, які НЕ називають справу. 🔴 Кадри часто лежать у підтеці
#: `<справа>/pages`, і саме її подають на вхід (рахунок кадрів не рекурсивний).
#: Без цього списку дві різні справи дають ОДИН слуг `pages` — спільний префікс
#: у R2 і спільну теку виводу, тобто два декоди зливаються в один. Спіймано
#: сухим прогоном на ЦДІАК ф.127 (spr-1649/pages і spr-1655/pages).
_SERVICE_DIR_NAMES = frozenset({"pages", "frames", "images", "img", "scans", "raw"})


def case_slug(case_dir: Path) -> str:
    """Ім'я справи для ключів R2 і тек виводу: без пробілів і кирилиці."""
    raw = case_dir.name or case_dir.parent.name
    if raw.lower() in _SERVICE_DIR_NAMES and case_dir.parent.name:
        raw = case_dir.parent.name
    out = []
    for ch in raw:
        lower = ch.lower()
        if lower in _TRANSLIT:
            translit = _TRANSLIT[lower]
            out.append(translit.upper() if ch.isupper() and translit else translit)
        else:
            out.append(ch)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", "".join(out)).strip("-")
    return slug or "case"


def run_slug(case_dir: Path, name: str = "") -> str:
    """Ім'я прогону: явне (`--name`) або з теки кадрів.

    🔴 Перечитування тієї самої справи ІНШОЮ моделлю мусить мати власне ім'я
    (`spr-2461-skryba_v6`): воно ж префікс чекпоінтів у R2 і тека результату, а
    в теці справи вже лежать тексти першої моделі, яких забір не перезаписує.
    """
    return case_slug(Path(name.strip())) if name and name.strip() else case_slug(case_dir)


#: Де раннер на боксі тримає кеш сегментації справи — стабільно, без номера місця
#: в черзі. Дублюється в `_embedded/htr_case_runner.SEG_CACHE_ARC`.
SEG_CACHE_ARC = "data/derived/htr_seg/case"


def seed_seg_cache(seg_dir: Path, ckpt_prefix: str, opts: BuildOptions, *,
                   s3: Any, log: Any) -> int:
    """Покласти готову сегментацію першим чекпоінтом прогону. Повертає кількість кадрів.

    Бокс розпаковує чекпоінти в робочу теку ДО старту раннера, тож кеш лягає
    саме туди, де раннер його шукає (`SEG_CACHE_ARC`), і `blla` не вантажиться
    зовсім: сторінка коштує розпізнавання, а не сегментацію (18.4 → 9.1 с/стор
    на повторному прогоні іншою моделлю, TOOLS_HTR_PIPELINE §2).

    🔴 Лише в ПОРОЖНІЙ префікс: якщо прогін уже має чекпоінти, це відновлення
    після обриву, і ті архіви несуть і кеш, і тексти — `ckpt_0001` поверх них
    знищив би базу серії.
    """
    files = sorted(Path(seg_dir).glob("*.seg.json.gz"))
    if not files:
        log(f"[план] {ckpt_prefix}: у {seg_dir} немає *.seg.json.gz — без засіву")
        return 0
    have = r2.ls(ckpt_prefix.rstrip("/") + "/", bucket=opts.bucket, s3=s3)
    if have:
        log(f"[план] {ckpt_prefix}: уже є {len(have)} чекпоінт(ів) — кеш не засіваю, "
            f"відновлення підхопить своє")
        return 0
    with tempfile.TemporaryDirectory(prefix="htrseed-") as tmp:
        tgz = Path(tmp) / "ckpt_0001.tgz"
        with tarfile.open(tgz, "w:gz") as tf:
            for f in files:
                tf.add(f, arcname=f"{SEG_CACHE_ARC}/{f.name}")
        r2.put(tgz, bucket=opts.bucket, prefix=ckpt_prefix.rstrip("/"), s3=s3, quiet=True)
    log(f"[план] {ckpt_prefix}: засіяно кеш сегментації — {len(files)} кадрів")
    return len(files)


def assert_unique_slugs(case_dirs: list[Path], names: list[str] | None = None) -> None:
    """🔴 Приймач: дві справи НЕ можуть мати спільне ім'я.

    Колізія не падає сама собою — вона тихо зливає кадри в один префікс R2 і
    перезаписує чужий декод. Тому перевіряємо ДО заливки.
    """
    seen: dict[str, Path] = {}
    for i, case_dir in enumerate(case_dirs):
        slug = run_slug(case_dir, names[i] if names else "")
        if slug in seen:
            raise ValueError(
                f"дві справи дають однакове ім'я «{slug}»:\n"
                f"   {seen[slug]}\n   {case_dir}\n"
                "   Перейменуй теку кадрів — інакше декоди зіллються в один.")
        seen[slug] = case_dir


def frames_of(case_dir: Path) -> list[Path]:
    """Кадри справи РІВНО так, як їх побачить раннер: пласко, jpg/jpeg/png.

    🔴 Не рекурсивно: раннер бере `iterdir()`, і якщо порахувати тут інакше,
    знаменник розійдеться з тим, що прогін реально прочитає — а саме на
    знаменнику стоять ворота повноти.
    """
    return sorted(p for p in case_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_SUFFIXES)


@dataclass
class Geometry:
    """Геометрія кадрів справи — вхід до розрахунку VRAM на шард."""

    mpx_median: float = 0.0
    mpx_p95: float = 0.0
    aspect_median: float = 0.0
    sampled: int = 0

    @property
    def spread(self) -> bool:
        """Чи це РОЗВОРОТ (ширина більша за висоту)."""
        return self.aspect_median > 1.0


def measure_frames(frames: list[Path], *, sample: int = 60) -> Geometry:
    """Розміри кадрів із заголовків — без декодування пікселів.

    🔴 Пік пам'яті тримає ПЛОЩА кадру, а не кількість каналів. Хибна гіпотеза
    «сирі RGB → втричі більший пік → сірий дозволить менший поріг» коштувала
    двох перерваних прогонів (04.09.2026): на тому самому матеріалі сирі RGB на
    4.5 ГБ/шард дали НУЛЬ OOM, а препнуті сірі на 2.5 — 141 збій зі 191 спроби.
    Тому міряємо саме площу й пропорції.
    """
    if not frames:
        return Geometry()
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        # 🔴 Мовчазний нуль тут читався б як «кадри звичайні»: геометрія
        # визначає VRAM на шард, і «не міряли» — це інший стан, ніж «сторінка».
        print("[план] ⚠ немає Pillow — геометрію кадрів НЕ ЗМІРЯНО, VRAM/шард "
              "лишиться дефолтним. Ставити ВСІ extras разом (один --extra видаляє "
              "решту): uv sync --extra vast --extra modal … --extra r2",
              file=sys.stderr, flush=True)
        return Geometry()

    step = max(1, len(frames) // sample)
    mpx: list[float] = []
    aspects: list[float] = []
    for frame in frames[::step][:sample]:
        try:
            with Image.open(frame) as im:
                width, height = im.size
        except Exception:
            continue
        if width and height:
            mpx.append(width * height / 1e6)
            aspects.append(width / height)
    if not mpx:
        return Geometry()
    ordered = sorted(mpx)
    return Geometry(
        mpx_median=round(statistics.median(mpx), 2),
        mpx_p95=round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 2),
        aspect_median=round(statistics.median(aspects), 3),
        sampled=len(mpx),
    )


#: Блок і запис tar. Стандартні, але потрібні тут явно: розмір архіву ми
#: рахуємо ДО пакування, і робиться це саме блоками.
_TAR_BLOCK = tarfile.BLOCKSIZE
_TAR_RECORD = tarfile.RECORDSIZE


def expected_tar_size(frames: list[Path]) -> int:
    """Скільки байтів дасть `pack` на цих кадрах — не пакуючи їх.

    Не оцінка: заголовки складає той самий `tarfile` тим самим форматом, тіла
    ми лише рахуємо блоками. Тому довге ім'я, кирилиця в імені й розширений
    заголовок PAX ураховуються точно так, як у справжньому архіві. Приймач —
    `test_the_predicted_archive_size_is_exact`: спаковане й передбачене мусять
    зійтись до байта.

    Потрібне це для одного питання: чи лежить у бакеті ТОЙ САМИЙ архів. Розмір
    відповідає на нього без читання 24 ГБ і без довіри до самої назви ключа.
    """
    total = 0
    with tarfile.open(fileobj=io.BytesIO(), mode="w") as probe:
        for frame in frames:
            info = probe.gettarinfo(str(frame), arcname=frame.name)
            total += len(info.tobuf(probe.format, probe.encoding, probe.errors))
            total += -(-info.size // _TAR_BLOCK) * _TAR_BLOCK
    total += 2 * _TAR_BLOCK                       # два нульові блоки = кінець
    tail = total % _TAR_RECORD                    # добивка до запису
    return total + (_TAR_RECORD - tail if tail else 0)


def already_in_bucket(key: str, *, size: int, mtime: float, opts: BuildOptions,
                      s3: Any, log: Any, what: str) -> bool:
    """Чи лежить у бакеті рівно цей файл — тобто чи можна не заливати.

    🔴🔴 Навіщо взагалі. Ключ архіву справи — `cases/<слуг>.tar`, і він СТАЛИЙ:
    не несе ні дати, ні номера заходу. Тому кадри, залиті першого разу, лежать
    там і далі, а кожен наступний захід пакував і заливав їх заново. Заміряно
    23.09.2026: 237 справ, 24 ГБ у бакеті — і 10-15 хвилин домашнього аплінка
    перед КОЖНИМ прогоном, який сам триває хвилини.

    Прапорець на це вже був (`--skip-upload`), і саме тому вада прожила довго:
    можливість була, а зробити її дією мусила людина — причому людина, яка не
    може знати напевно, чи об'єкт на місці. Помилка в бік «пропустити» дає
    посилання в порожнечу: раннер дістає 404 на пробі каналу, і оренда
    змарнована (06.09.2026). Тому питаємо бакет, а не пам'ять.

    Два умови, і обидві потрібні:

    · РОЗМІР той самий — інакше кадри додали, прибрали чи перезняли;
    · об'єкт НОВІШИЙ за найсвіжіший кадр — інакше кадр підмінили після
      заливки, а розмір випадково зійшовся.

    Будь-яка невизначеність (немає об'єкта, не спитались, не збіглось) означає
    заливку: зайва заливка коштує часу, захід зі чужими кадрами — усього
    прогону. І кожна відмова називає ПРИЧИНУ, інакше повторна заливка
    виглядала б як те, що ця перевірка й мала прибрати.
    """
    from datetime import UTC, datetime

    found = r2.head(key, bucket=opts.bucket, s3=s3)
    if found is None:
        return False
    if int(found.get("size") or 0) != int(size):
        log(f"[план] {what}: у бакеті {int(found.get('size') or 0) / 1e6:.0f} МБ, "
            f"а треба {size / 1e6:.0f} МБ — заливаю заново")
        return False
    stamp = found.get("modified")
    if not isinstance(stamp, datetime):
        # Часу немає — підміни за розміром ми не побачимо, тож не ризикуємо.
        log(f"[план] {what}: бакет не сказав дати об'єкта — заливаю заново")
        return False
    when = stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)
    if mtime > when.timestamp():
        log(f"[план] {what}: файл новіший за об'єкт у бакеті — заливаю заново")
        return False
    return True


def pack(case_dir: Path, frames: list[Path], dest: Path, slug: str = "") -> Path:
    """Один tar на справу, БЕЗ gzip: JPEG уже стиснутий, а gzip коштує хвилини.

    🔴 Ім'я архіву = ключ у R2 (`r2.put` бере ім'я файла), тож воно мусить бути
    тим самим `slug`, з якого нарізано `pages_url`. Доти архів завжди звався за
    текою кадрів, а посилання — за `--name`: перечитування іншою моделлю
    (`178-53-36-skryba_v6`) заливало `cases/178-53-36.tar` і посилалось на
    `cases/178-53-36-skryba_v6.tar` — передполіт 404 (15.09.2026).
    """
    archive = dest / f"{slug or case_slug(case_dir)}.tar"
    with tarfile.open(archive, "w") as tf:
        for frame in frames:
            tf.add(frame, arcname=frame.name)
    return archive


@dataclass
class BuildOptions:
    """Що саме будуємо. `out_root` обов'язковий і абсолютний."""

    out_root: Path
    model: str
    voices: str = ""
    prefix: str = "cases"
    bucket: str = ""
    budget_usd: float = 3.0
    max_hours: float = 8.0
    max_price: float = 0.365
    #: ⚠ З 23.09.2026 не діють і в план не пишуться (див. `Plan.prefer_min_cores`);
    #: лишились, щоб старі виклики не падали.
    prefer_cores: float = 0.0
    wait_min: float = 0.0
    #: 🔴🔴 Ціль темпу, стор/год: машина, що обережно її не дає, не береться.
    target_pph: float = DEFAULT_TARGET_PPH
    #: Скільки хвилин чекати ринку, коли жодна машина не дає цілі.
    max_wait_min: float = 60.0
    disk_gb: int = 40
    gpu: str = "any"
    max_usd_per_1000: float | None = None
    gb_per_shard: float = 0.0
    url_hours: float = 24.0
    skip_upload: bool = False
    params: dict[str, str] = field(default_factory=dict)
    #: {ім'я скрипта: sha256, який МУСИТЬ лежати в архіві}. Розбіжність — відмова
    #: ще до заливки; порожньо = не перевіряти.
    expect_scripts: dict[str, str] = field(default_factory=dict)
    #: Хвилин тримати бокс теплим після черги (див. `Plan.keep_warm_min`).
    keep_warm_min: float = 0.0
    #: Ім'я прогону на кожну справу, у тому ж порядку ("" = з теки кадрів).
    names: list[str] = field(default_factory=list)
    #: Тека готової сегментації на кожну справу ("" = без засіву).
    seed_seg: list[str] = field(default_factory=list)
    #: Чим доставляти дані на машину: `r2` — бакет S3 з presigned-посиланнями,
    #: `box` — склад на самій машині (`htr/origin.py`), куди файли кладе scp.
    #: 🔴 Бакет швидший і переживає смерть машини, але він мусить БУТИ: тому,
    #: хто просто орендував бокс на годину, вимагати ще й S3 — це вимагати
    #: другого акаунта заради однієї справи. `box` знімає цю вимогу ціною
    #: того, що чекпоінти лежать на тій самій машині, яка може вмерти, —
    #: і тому наглядач забирає їх додому сам.
    transport: str = "r2"
    #: Куди класти спаковані кадри при `transport="box"` (порожньо — поруч із
    #: планом). Тека переживає захід: переоренда не пакує все вдруге.
    staging_dir: Path | None = None


def scripts_sha256_in(assets: Path) -> dict[str, str]:
    """sha256 кожного `*.py` в архіві ассетів (за базовим іменем).

    Саме ці файли раннер розкладає в `/tmp/htrcase/scripts` і звіряє з планом
    до старту — тож хеш береться з того, що ПОЇДЕ, а не з робочого дерева.
    """
    import hashlib

    out: dict[str, str] = {}
    with tarfile.open(assets) as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".py"):
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            out[Path(member.name).name] = hashlib.sha256(fh.read()).hexdigest()
    return out


def sha256_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines_per_page_from_meta(out_dir: Path) -> float:
    """Медіана рядків на сторінку з мети ПОПЕРЕДНЬОГО прогону цієї справи.

    Найдешевше джерело щільності матеріалу: перепрогін (нова модель, догін,
    другий голос) завжди має за собою `_htr_meta.json`, і в ньому `lines` на
    кожну сторінку. Порожні сторінки не рахуються. Нуль = мети немає або вона
    порожня.

    🔴 Сторінки в стелі сегментації (200 рядків) і з піднятою стелею (понад
    200) РАХУЮТЬСЯ. Доти їх відкидали як «обрізані» — тобто з вибірки зникали
    саме найгустіші аркуші, і щільність сповідки виходила заниженою там, де
    вона найважливіша для прогнозу. 200 — нижня межа справжнього числа, і це
    ближче до правди, ніж пропуск.
    """
    path = out_dir / "_htr_meta.json"
    if not path.is_file():
        return 0.0
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0.0
    pages = meta.get("pages") if isinstance(meta, dict) else None
    items = pages.values() if isinstance(pages, dict) else (pages or [])
    lines = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        n = entry.get("lines")
        if isinstance(n, (int, float)) and n > 0:
            lines.append(float(n))
    return round(statistics.median(lines), 1) if lines else 0.0


def key_tag(case_key: str) -> str:
    """Шифра справи як частина адреси: `DAHmO/315/66` → `DAHmO-315-66`; "" — немає."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(case_key or "")).strip("-")


def _meta_case_key(out_dir: Path) -> str:
    """Шифра справи, чий текст уже лежить у теці (з `_htr_meta.json`)."""
    try:
        meta = json.loads((out_dir / "_htr_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(meta.get("case_key") or "") if isinstance(meta, dict) else ""


def pick_out_dir(out_root: Path, slug: str, case_key: str) -> Path:
    """Тека результату, у якій немає ЧУЖОЇ справи.

    🔴🔴 Тека звалась лише номером справи (`spr-66`), без архіву. 23.09.2026
    паралельна сесія ДАХмО ф.315 мала справи з тими самими номерами: у теки
    наших книг 66–72 приїхали 588 чужих сторінок, а книга 69 ледь не пішла як
    «уже прочитана» чужим декодом. Лишаємо звичне ім'я, якщо тека вільна або
    належить ТІЙ САМІЙ шифрі (дочитування, перепрогін), — інакше ім'я несе
    шифру.
    """
    base = out_root / slug
    tag = key_tag(case_key)
    if not tag:
        return base
    owner = _meta_case_key(base)
    if not owner or key_tag(owner) == tag:
        return base
    return out_root / f"{slug}__{tag}"


def _case_entry(case_dir: Path, opts: BuildOptions, *, case_key: str,
                s3: Any, log: Any, name: str = "", seed_seg: str = "") -> dict[str, Any]:
    """Одна справа: спакувати, залити, нарізати посилання, зміряти геометрію."""
    frames = frames_of(case_dir)
    if not frames:
        raise ValueError(f"{case_dir}: немає кадрів {IMG_SUFFIXES}")
    slug = run_slug(case_dir, name)
    tag = key_tag(case_key)
    # 🔴🔴 Адреси в СХОВИЩІ — зі шифрою. Кадри (`cases/…`), чекпоінти й
    # службовий архів мали ключ лише з номера справи, тож дві сесії з різних
    # архівів ділили їх: чекпоінти однієї розпаковувались у теку іншої (588
    # чужих сторінок, 23.09.2026), а кадри могли перезаписатись.
    store = f"{tag}/{slug}" if tag else slug
    key = f"{opts.prefix.rstrip('/')}/{store}.tar"
    geometry = measure_frames(frames)

    pages_path = ""
    if opts.transport == "box":
        # Нікуди не заливаємо: архів лежить удома, доки не буде машини, на яку
        # його класти. Пакуємо однаково зараз — щоб відмова (немає місця,
        # битий кадр) сталася ДО оренди, а не на оплачуваній машині.
        staging = Path(opts.staging_dir or Path.cwd())
        staging.mkdir(parents=True, exist_ok=True)
        archive = staging / f"{store.replace('/', '__')}.tar"
        if archive.is_file() and archive.stat().st_size:
            log(f"[план] {slug}: {len(frames)} кадрів — архів уже спакований")
        else:
            packed = pack(case_dir, frames, staging, slug)
            if packed != archive:
                packed.replace(archive)
            log(f"[план] {slug}: {len(frames)} кадрів, "
                f"{archive.stat().st_size / 1e6:.0f} МБ → {archive}")
        pages_path = str(archive.resolve())
    elif opts.skip_upload:
        log(f"[план] {slug}: {len(frames)} кадрів (заливку пропущено — сказано)")
    elif already_in_bucket(key, size=expected_tar_size(frames),
                           mtime=max(f.stat().st_mtime for f in frames),
                           opts=opts, s3=s3, log=log, what=slug):
        log(f"[план] {slug}: {len(frames)} кадрів уже в бакеті — не заливаю")
    else:
        with tempfile.TemporaryDirectory(prefix="htrplan-") as tmp:
            archive = pack(case_dir, frames, Path(tmp), slug)
            log(f"[план] {slug}: {len(frames)} кадрів, "
                f"{archive.stat().st_size / 1e6:.0f} МБ → R2")
            # `r2.put` складає ключ із префікса й ІМЕНІ файла, а ключ плану несе
            # шифру (`cases/<шифра>/<справа>.tar`) — тож заливається під ним
            # самим, і невідповідність валить план до оренди.
            prefix, _, name = key.rpartition("/")
            if archive.name != name:
                archive = archive.replace(archive.with_name(name))
            got = r2.put(archive, bucket=opts.bucket, prefix=prefix, s3=s3)
            if got != key:
                raise RuntimeError(f"{slug}: кадри залито під `{got}`, а план "
                                   f"посилається на `{key}`")

    # 🔴 Слоти — від розміру ЦІЄЇ справи. Див. `CKPT_PAGES_PER_SLOT`: на черзі
    # стеля заходу давала кожній справі по 314 посилань, і `job.py` важив 48 МБ.
    n_ckpt = ckpt_urls_for(opts.max_hours, pages=len(frames))
    # 🔴 Ключ чекпоінтів несе МОДЕЛЬ. Без неї прогін новою моделлю підхоплював
    # чужі тексти: раннер бачить сторінку в стані як зроблену й пропускає її, а
    # забір не перезаписує наявні `.txt`. Тобто платимо за v17, отримуємо v16 —
    # і підсумок при цьому чесно каже «модель v17, complete: true».
    model_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", opts.model or "model").strip("-") or "model"
    ckpt_prefix = f"ckpt/{store}/{model_tag}"
    if seed_seg:
        seed_seg_cache(Path(seed_seg), ckpt_prefix, opts, s3=s3, log=log)
    out_dir = pick_out_dir(opts.out_root, slug, case_key).resolve()
    log(f"[план] {slug}: результат ляже в {out_dir}")
    lines = lines_per_page_from_meta(out_dir)
    if lines:
        log(f"[план] {slug}: попередній прогін — медіана {lines:.0f} рядків/стор")
    box = opts.transport == "box"
    return {
        "case": slug,
        # 🔴 При складі на машині посилання ще НЕ ІСНУЄ: його адреса — це порт
        # машини, якої поки немає. Наглядач підставить його, коли бокс
        # підніметься; тут лишається шлях до архіву вдома.
        "pages_url": "" if box else r2.get_url(key, hours=opts.url_hours,
                                               bucket=opts.bucket, s3=s3),
        "pages_path": pages_path,
        # 🔴 Службовий архів (лог раннера, стан, прогрес, логи шардів) їде ТИМ
        # САМИМ шляхом, що й чекпоінти, — одним об'єктом. Доти дім забирав ці
        # файли по SFTP пофайлово, і на черзі це вироджувалось у ~2600 обертів
        # через океан: забір висів понад 30 хвилин і лишив 237 справ без
        # вердикту при цілому результаті в бакеті (23.09.2026).
        "service_put_url": "" if box else r2.put_url(
            f"service/{store}.tgz", hours=opts.url_hours, bucket=opts.bucket, s3=s3),
        "service_url": "" if box else r2.get_url(
            f"service/{store}.tgz", hours=opts.url_hours, bucket=opts.bucket, s3=s3),
        "case_key": case_key,
        "case_dir": str(case_dir.resolve()),
        "n_pages": len(frames),
        "out_dir": str(out_dir),
        "flatten_out": True,
        "frame_mpx_median": geometry.mpx_median,
        # 🔴 VRAM на шард рахується від p95, не від медіани: справа з медіаною
        # 7 Мпікс і десятком розворотів по 16 дасть OOM саме на них, а флот
        # уже розкладено під 1.9 ГБ. Пік алокатора йде від НАЙБІЛЬШОГО кадру.
        "frame_mpx_p95": geometry.mpx_p95,
        "frame_aspect_median": geometry.aspect_median,
        "lines_per_page_median": lines,
        # Обсяг, який бокс качатиме: ворота звіряють із ним виміряний канал.
        "pages_bytes": sum(f.stat().st_size for f in frames),
        "ckpt_prefix": ckpt_prefix,
        "ckpt_slots": n_ckpt,
        "ckpt_urls": [] if box else r2.put_urls(ckpt_prefix, n_ckpt, hours=opts.url_hours,
                                                bucket=opts.bucket, s3=s3),
        "resume_urls": [] if box else r2.ckpt_get_urls(
            ckpt_prefix, n_ckpt, hours=opts.url_hours, bucket=opts.bucket, s3=s3),
    }


def build_plan(case_dirs: list[Path], opts: BuildOptions, *,
               case_keys: list[str] | None = None,
               assets: Path | None = None, assets_key: str = "",
               out_path: Path | None = None,
               log: Any = None) -> dict[str, Any]:
    """Скласти план. Пише його на диск ПІСЛЯ КОЖНОЇ справи, якщо дано `out_path`.

    🔴 Інкрементальність тут не зручність. На черзі з 22 справ R2 відмовив на
    останній, і план не створився ВЗАГАЛІ — файл лишився нульовим, тобто
    втратилась і вже залита частина (30.08.2026). Тепер кожна залита справа вже
    записана, а падіння забирає з собою лише себе.
    """
    log = log or (lambda line: print(line, file=sys.stderr, flush=True))
    if not opts.out_root.is_absolute():
        raise ValueError(
            f"`out_root` мусить бути АБСОЛЮТНИМ: {opts.out_root}. Відносний "
            f"шлях розкладеться від теки, з якої запущено наглядача, а це "
            f"майже завжди інший простір, ніж той, якому належить справа.")
    if case_keys and len(case_keys) != len(case_dirs):
        raise ValueError(
            f"шифр задано {len(case_keys)} на {len(case_dirs)} справ. Треба або "
            f"жодної, або по одній на кожну справу в тому ж порядку: порядок — "
            f"єдине, що зв'язує ключ зі справою.")
    if opts.transport not in ("r2", "box", "auto"):
        raise ValueError(f"невідомий транспорт «{opts.transport}»: буває `r2` "
                         f"(бакет S3), `box` (склад на самій машині) або `auto`")
    if opts.transport == "auto":
        # Бакет швидший і переживає смерть машини, тож коли він є — беремо його.
        opts.transport = "r2" if r2.configured() else "box"
        log(f"[план] транспорт: {opts.transport}"
            + ("" if opts.transport == "r2" else " (бакет не налаштований)"))
    if not assets and not assets_key:
        raise ValueError("треба або `assets` (локальний архів), або `assets_key`")
    if opts.transport == "box":
        if not assets:
            # Ключ у бакеті нічого не означає для машини, яка до бакета не
            # ходить, — а мовчазний відкат на R2 тут означав би вимагати
            # ключів після того, як людина сказала, що їх немає.
            raise ValueError("транспорт `box` возить архів ассетів із диска — "
                             "дайте `--assets <файл>`, а не `--assets-key`")
        if opts.skip_upload:
            raise ValueError("`--skip-upload` каже «кадри вже в бакеті», а "
                             "транспорт `box` бакета не знає")
        if any(opts.seed_seg):
            # Засів сегментації кладеться першим чекпоінтом У БАКЕТ; на складі
            # машини його поки нема куди покласти — вона ще не орендована.
            raise ValueError(
                "`--seed-seg` поки працює лише з бакетом (`--transport r2`): "
                "готова сегментація їде першим чекпоінтом у сховище, а склад "
                "на машині з'являється аж після оренди")
    for label, values in (("імен прогону", opts.names), ("тек сегментації", opts.seed_seg)):
        if values and len(values) != len(case_dirs):
            raise ValueError(f"{label} задано {len(values)} на {len(case_dirs)} справ — "
                             f"треба по одному на кожну справу в тому ж порядку")

    assert_unique_slugs(case_dirs, names=opts.names or None)
    for case_dir in case_dirs:
        if not case_dir.is_dir():
            raise ValueError(f"немає теки справи: {case_dir}")

    # 🔴 Presigned-посилання мусять пережити ВЕСЬ захід разом із холодним
    # стартом і переорендою. Протермінований GET віддає 403, а раннер читає це
    # як «чекпоінта ще немає» і тихо стартує з нуля — тобто «відновлення після
    # обриву» перетворюється на повний повторний прогін за повні гроші.
    need_hours = opts.max_hours * 1.5 + 2
    if opts.url_hours < need_hours:
        log(f"[план] термін дії посилань {opts.url_hours:.0f} год замалий під "
            f"стелю {opts.max_hours:.1f} год — піднімаю до {need_hours:.0f}")
        opts.url_hours = need_hours

    box = opts.transport == "box"
    if box and opts.staging_dir is None and out_path is not None:
        opts.staging_dir = out_path.parent / "_box"
    # 🔴 Клієнта бакета не створюємо взагалі: саме його відсутність (немає
    # ключів, немає boto3) і є причина, з якої транспорт `box` існує.
    s3 = None if box else r2.client()
    scripts_sha: dict[str, str] = {}
    if assets is not None:
        if not assets.is_file():
            raise ValueError(f"немає архіву assets: {assets}")
        scripts_sha = scripts_sha256_in(assets)
        stale = {name: (expected, scripts_sha.get(name))
                 for name, expected in (opts.expect_scripts or {}).items()
                 if scripts_sha.get(name) != expected}
        if stale:
            # 🔴 До оренди, а не на оплачуваній карті: 05.09.2026 в архіві лежав
            # старий раннер без ліміту потоків, і флот ішов утричі повільніше.
            raise ValueError(
                "assets несе застарілі скрипти — перепакуй архів: "
                + "; ".join(f"{n}: очікувано {e[:12]}…, в архіві {(a or 'немає')[:12]}…"
                            for n, (e, a) in sorted(stale.items())))
        # 🔴 `--skip-upload` на ассети НЕ поширюється: він каже «кадри вже в
        # R2», а не «і ассети теж» — 06.09.2026 план зі свіжими ассетами й
        # `--skip-upload` поїхав на бокс із посиланням у порожнечу (HTTP 404 на
        # пробі каналу, оренда змарнована). Пропуск тут вирішує САМ БАКЕТ, а
        # не прапорець: він відповідає про цей конкретний об'єкт, тож сказати
        # «уже там» про те, чого там немає, нема як.
        if box:
            # Ассети теж лишаються вдома до появи машини.
            log(f"[план] ассети поїдуть із {assets}")
        elif already_in_bucket(f"assets/{assets.name}",
                               size=assets.stat().st_size,
                               mtime=assets.stat().st_mtime,
                               opts=opts, s3=s3, log=log,
                               what=f"assets {assets.name}"):
            # Ваги й скрипти між заходами майже не міняються, а важать сотні
            # мегабайтів — це другий за розміром шматок домашнього аплінка
            # після самих кадрів.
            log(f"[план] assets {assets.name} уже в бакеті — не заливаю")
            assets_key = f"assets/{assets.name}"
        else:
            log(f"[план] заливаю assets {assets.name}")
            r2.put(assets, bucket=opts.bucket, prefix="assets", s3=s3)
            assets_key = f"assets/{assets.name}"

    plan: dict[str, Any] = {
        "scripts_sha256": scripts_sha,
        "keep_warm_min": opts.keep_warm_min,
        "transport": opts.transport,
        "assets_url": "" if box else r2.get_url(assets_key, hours=opts.url_hours,
                                                bucket=opts.bucket, s3=s3),
        "assets_path": str(assets.resolve()) if (box and assets) else "",
        "gpu": opts.gpu,
        "budget_usd": opts.budget_usd,
        "max_hours": opts.max_hours,
        "max_price": opts.max_price,
        "target_pph": opts.target_pph,
        "max_wait_min": opts.max_wait_min,
        "disk_gb": opts.disk_gb,
        "cases": [],
    }
    if not box:
        # 💓 Серцебиття бокса: один об'єкт, який бокс перезаписує кожні кілька
        # хвилин (лог раннера, прогрес, мітка часу). Без наглядача з нього видно,
        # що бокс робив і коли востаннє жив — а отже, скільки він коштував.
        import secrets

        plan_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        beat_key = f"heartbeat/{plan_id}.tgz"
        plan["plan_id"] = plan_id
        plan["heartbeat_put_url"] = r2.put_url(beat_key, hours=opts.url_hours,
                                               bucket=opts.bucket, s3=s3)
        plan["heartbeat_url"] = r2.get_url(beat_key, hours=opts.url_hours,
                                           bucket=opts.bucket, s3=s3)
    if opts.max_usd_per_1000 is not None:
        plan["max_usd_per_1000_pages"] = opts.max_usd_per_1000
    if opts.gb_per_shard:
        plan["gb_per_shard"] = opts.gb_per_shard
    # 🔴 Бойові моделі кладуться в план ЗАВЖДИ, а не лишаються на дефолт job'а.
    # Ціна мовчазного дефолту виміряна 2026-08-11: `voices` дефолтиться в
    # порожній рядок, тож ГІЛКА ДРУГОГО ГОЛОСУ просто не запускалась — 2578
    # сторінок пройшли одним голосом, і виявилось це аж на звірці чекпоінтів.
    params: dict[str, str] = {"model": opts.model}
    if opts.voices:
        params["voices"] = opts.voices
    params.update(opts.params)
    plan["params"] = params

    for i, case_dir in enumerate(case_dirs):
        entry = _case_entry(
            case_dir, opts, s3=s3, log=log,
            case_key=(case_keys[i].strip() if case_keys else ""),
            name=(opts.names[i] if opts.names else ""),
            seed_seg=(opts.seed_seg[i] if opts.seed_seg else ""))
        plan["cases"].append(entry)
        if out_path is not None:
            out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1),
                                encoding="utf-8")

    total = sum(c["n_pages"] for c in plan["cases"])
    log(f"[план] {len(plan['cases'])} справ, {total} сторінок · "
        f"бюджет ${opts.budget_usd:.2f} · стеля {opts.max_hours:.1f} год")
    return plan
