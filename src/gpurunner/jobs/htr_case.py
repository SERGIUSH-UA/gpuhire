"""HTRCaseJob — прогін справи HTR-конвеєром nyshporka (Писар + Дяк) на хмарній карті.

🔴 Потребує пакета `nyshporka`: раннер справи й моделі приходять звідти, сам
gpurunner лише доставляє їх на бокс і стежить за прогоном.

Не бенчмарк і не трен: це та сама робота, яку локально робить
``htr_case_run.py`` (раннер справи nyshporka), винесена на чужу карту, бо на GTX 1650 корпус
сповідних розписів (≈35.6 тис. сторінок) іде 90-120 годин.

Раннер НЕ переписує конвеєр, а запускає той самий файл субпроцесом — інакше
хмарний прогін порівнювався б із локальним не по залізу, а по двох різних
конвеєрах. Тому серед входів є слаг зі скриптами раннера.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path, PurePath
from typing import Any, ClassVar

from gpurunner.config import modal_volume_name
from gpurunner.core.job import Job

#: с/стор на ОДИН шард, за картами. GTX 1650 — виміряно (сповідка ДАХмО 315-1-7864,
#: Писар v17 + Дяк-голос, 95 рядків/стор медіани). Решта — ОЦІНКА за співвідношенням
#: карт, і саме її має замінити перший smoke: числа тут керують лише прогнозом
#: вартості, а не самим прогоном.
SEC_PER_PAGE_1SHARD: dict[str, float] = {
    "GTX1650": 13.5,
    "RTX4090": 3.5,
    "RTX5090": 3.0,
    "A6000": 4.5,
    "L40S": 3.2,
    "A100-40": 3.8,
    "A100-80": 3.6,
    "H100": 2.8,
    "H200": 2.6,
    "B200": 2.4,
}
DEFAULT_SEC_PER_PAGE = 4.0

#: Скільки шардів реально дає прискорення. Вище цього впирається CPU-геометрія
#: сегментації, а не карта, тож масштабування пласке. Число НЕ виміряне — smoke
#: із двома значеннями `shards` його і уточнює.
PARALLEL_CEILING = 6.0


def _as_bool(value: Any, default: bool) -> bool:
    """`-p dynamic_pages=false` приходить рядком, а `bool("false")` — True."""
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "так")
    return bool(value)


class HTRCaseJob(Job):
    """Транскрипція справи: kraken-сегментація + PARSeq (Писар) + kraken-голос (Дяк)."""

    name: ClassVar[str] = "htr_case"
    description: ClassVar[str] = (
        "Run the nyshporka case runner over a case folder: kraken blla segmentation "
        "(gpu_sato + fast_geom patches), PARSeq recognition, optional kraken voice."
    )
    supported_backends: ClassVar[tuple[str, ...]] = (
        "beam",
        "modal",
        "vast",
        "lightning",
        "saturn",
        "colab",
        # 🔴 Kaggle тут НЕ рівноцінний решті, і різниця не в коді, а в режимі
        # роботи: наглядача (`htr supervise`) на ньому немає — той побудований
        # навколо `VastBackend` (оренда, проба заліза, гасіння). Тобто доганяння
        # пропущених сторінок і звірку повноти робить лише сам раннер усередині
        # кернела (`catchup_passes`), а зовнішнього ока над ним немає.
        # Плюс дві жорсткі стелі платформи: 12 год на один кернел і 30 год
        # GPU-квоти на тиждень — тому справу треба різати на порції ЗАЗДАЛЕГІДЬ,
        # а не сподіватись, що велика доїде.
        "kaggle",
    )

    def requirements(self) -> list[str]:
        return ["kraken==7.0.2", "pillow>=10.0", "numpy"]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        raw_shards: Any = params.get("shards")
        # 🔴 Нуль тут — це вже НОРМАЛІЗОВАНЕ «auto», а не помилка. Валідація
        # мусить бути ідемпотентною: `validate(validate(x)) == validate(x)`.
        # Інакше будь-який повторний виклик (а їх кілька — `colab_input_dirs`,
        # `estimate_runtime`, накладання перекриттів після проби заліза)
        # падає на власному ж виводі.
        if raw_shards in (None, "", "auto", 0):
            # 🔴 `auto` = 0 = «порахуй на місці». Число шардів мусить братися з
            # ВИМІРЯНОГО заліза (вільна VRAM ÷ 2.6 ГБ, ядра ÷ 2), а не з рук і
            # не з картки оффера: саме з рук колись узялись 8 шардів на 16 ГБ,
            # і CUDA OOM тихо з'їв 46 сторінок. Рахує сам раннер уже в
            # контейнері — тоді це працює й без наглядача.
            shards = 0
        else:
            shards = int(raw_shards)
            if shards < 1:
                raise ValueError("shards must be >= 1 or 'auto'")
        enhance = str(params.get("enhance") or "auto")
        if enhance not in ("auto", "none", "clahe", "clahew", "clahesmooth"):
            raise ValueError(f"enhance must be one of auto/none/clahe/clahew/clahesmooth, got {enhance!r}")
        script = str(params.get("script") or "cyrillic")
        if script not in ("auto", "latin", "cyrillic", "mixed"):
            raise ValueError(f"script must be auto/latin/cyrillic/mixed, got {script!r}")
        out: dict[str, Any] = {
            "pages_slug": str(params.get("pages_slug") or "pages"),
            "models_slug": str(params.get("models_slug") or "models"),
            "scripts_slug": str(params.get("scripts_slug") or "scripts"),
            # Тека на ВХІДНОМУ томі, куди чекпоінт дзеркалить прочитане. Наступний
            # запуск бачить її як стартовий стан і доганяє справу, а не починає з
            # нуля. Не входить у `colab_input_dirs`: локальної теки під неї немає,
            # її створює сам раннер уже в контейнері.
            "state_slug": str(params.get("state_slug") or ""),
            # HTTP-джерела великих архівів. Бокс качає їх сам своїм каналом —
            # SFTP бекенда дає ~0.4 МБ/с, і 31 ГБ їхали б довше, ніж рахуються.
            # Поріг швидкості каналу бокса, Мбіт/с. 0 = не перевіряти. Заявленій
            # у маркетплейсі вірити не можна — див. коментар у раннері.
            "min_net_mbps": (5.0 if params.get("min_net_mbps") in (None, "")
                             else float(params["min_net_mbps"])),
            # Чекпоінти в хмару БЕЗ ключів: presigned PUT — по одному посиланню на
            # раунд (`r2_put.py puturl --count N`), presigned GET — на архіви
            # попереднього запуску. Ключі сховища на орендований бокс не їдуть.
            # Перевірка орієнтації кадру: один перевернутий аркуш дає
            # псевдокириличний шум, який читається як погане письмо, а не як
            # збій. У хмару прапорець доти не доходив узагалі.
            "orient_check": _as_bool(params.get("orient_check"), False),
            # Рахувати sato на процесорі, а не на карті. Під шардингом це не
            # очевидний програш: карта — спільний ресурс шардів, ядра — ні.
            "no_gpu_sato": _as_bool(params.get("no_gpu_sato"), False),
            "ckpt_urls": list(params.get("ckpt_urls") or []),
            "resume_urls": list(params.get("resume_urls") or []),
            "assets_url": str(params.get("assets_url") or ""),
            "pages_url": str(params.get("pages_url") or ""),
            # Kaggle-канал даних: три ОКРЕМІ датасети `<owner>/<slug>`, які
            # платформа монтує в `/kaggle/input/<slug>/`. Потрібні там, де
            # немає R2: presigned-URL' без ключів сховища не зробити, а SFTP
            # бекенда дає ~0.4 МБ/с.
            #
            # 🔴 Чому саме ТРИ, а не один датасет із теками pages/ models/
            # scripts/: Kaggle при заливці МОВЧКИ викидає підтеки (`dir_mode=
            # "skip"`), тому `dataset_push` їх взагалі відмовляється брати.
            # Пакувати все в один архів теж не можна — розпаковується лише
            # tarball кадрів, а ваги й скрипти раннер шукає файлами.
            # Побічний виграш: моделі й скрипти заливаються ОДИН раз і
            # перевикористовуються всіма наступними порціями справи, а
            # мінятись на кожну порцію мусять лише кадри.
            "pages_dataset": str(params.get("pages_dataset") or ""),
            "models_dataset": str(params.get("models_dataset") or ""),
            "scripts_dataset": str(params.get("scripts_dataset") or ""),
            "checkpoint_sec": int(params.get("checkpoint_sec") or 120),
            # Стеля BLAS/OpenMP-потоків на шард. 0 = не чіпати (кожен процес бачить
            # усі ядра хоста і розгортається на них — на Beam це дало 182 ядра
            # сумарно при квоті 8). Розумний дефолт: замовлені ядра ÷ шарди.
            "threads_per_shard": int(params.get("threads_per_shard") or 0),
            "bundle": _as_bool(params.get("bundle"), False),
            "model": str(params.get("model") or "pysar_cyr_v17.pt"),
            "voices": str(params.get("voices") or ""),
            "shards": shards,
            "enhance": enhance,
            "script": script,
            "limit": int(params.get("limit") or 0),
            "batch": int(params.get("batch") or 0),
            # Батч kraken-голосу (Дяк): 0/1 = дефолт раннера. Замір 05.09.2026 на
            # RTX 3090: +10% темпу флоту на 8 і 12 шардах; ціна — ~1% символів у
            # теці голосу, тому дефолт лишається за раннером, а не тут.
            "voice_batch": int(params.get("voice_batch") or 0),
            # На 4 ГБ лок обов'язковий, на 24 ГБ він рівно те, що з'їдає виграш від
            # шардів: GPU-фаза серіалізується. Дефолт OFF — вмикати лише коли карта
            # мала або шарди почали падати з OOM.
            "gpu_lock": _as_bool(params.get("gpu_lock"), False),
            # 🔴 Дефолт ON, як і локально. Був OFF, і кожен догінний прохід та
            # будь-який повторний захід тією ж справою платив повні 7.3 с/кадр
            # сегментації замість 0.23 — при тому, що кеш був доведений і
            # ввімкнений за замовчуванням у самому скрипті.
            # ⚠ `_as_bool`, а не `bool`: `-p seg_cache=false` приходить рядком, і
            # `bool("false")` лишав кеш увімкненим.
            "seg_cache": _as_bool(params.get("seg_cache"), True),
            # 🎛 Ядер на шард для регулятора (0 = дефолт раннера 1.25). Перечитування
            # іншою моделлю з готовою сегментацією не платить за геометрію й sato
            # (~74% процесора сторінки), тож стеля шардів за ядрами там нижча.
            "cores_per_shard": float(params.get("cores_per_shard") or 0),
            # Скільки разів повторювати томи черги, що впали або лишились неповні.
            "queue_retry_passes": int(params.get("queue_retry_passes", 1) or 0),
            "max_endpoints": int(params.get("max_endpoints") or 0),
            "ceiling_retry": params.get("ceiling_retry"),
            "estimated_n_pages": int(params.get("estimated_n_pages") or 0),
            # Ім'я справи — щоб наглядач у своєму стані називав її, а не
            # «job 7e3bc736».
            "case": str(params.get("case") or ""),
            # ── повнота й нагляд ────────────────────────────────────────────
            # Скільки VRAM закладати на шард при `shards=auto`.
            "vram_gb_per_shard": float(params.get("vram_gb_per_shard") or 0),
            # Скільки разів доганяти сторінки, яких не вистачає. 0 — не
            # доганяти (тоді неповнота просто фіксується й валить job).
            "catchup_passes": int(params.get("catchup_passes", 2) or 0),
            # 🧲 Динамічний розподіл сторінок (клейми) замість зрізу [k::n]:
            # хвіст статичного зрізу — 12–20% роботи (std160 05.09.2026).
            # False = старий зріз, для відкату й звірок.
            "dynamic_pages": _as_bool(params.get("dynamic_pages"), True),
            # 🎛 Регулятор флоту на боксі: старт обережний (3.3 ГБ/шард), далі
            # флот міряє темп і пам'ять і сам шукає коліно; `shards` стає СТЕЛЕЮ.
            # Діє лише з раннером, що вміє злив (`_drain`); `false` — фіксований
            # флот, як до 10.09.2026.
            "regulate": _as_bool(params.get("regulate"), True),
            "shards_max": int(params.get("shards_max") or 0),
            # 🔥 Скільки секунд бокс лишається теплим після черги й приймає
            # довісок (`htr append`). 0 = гасити одразу, як і було.
            "keep_warm_sec": int(params.get("keep_warm_sec") or 0),
            # sha256 скриптів із плану: раннер звіряє копію в ассетах ДО старту.
            "scripts_sha256": dict(params.get("scripts_sha256") or {}),
            # 🔴 Прийняти неповний результат можна лише свідомо. Дефолт
            # `False`, бо саме мовчазна згода з неповнотою і коштувала
            # 46 сторінок, яких ніхто не помітив.
            "allow_incomplete": _as_bool(params.get("allow_incomplete"), False),
            # Скільки секунд тиші шарда вважати зависанням.
            "stall_sec": int(params.get("stall_sec") or 600),
            "shard_restart_max": int(params.get("shard_restart_max") or 3),
            # Черга справ на ОДНОМУ боксі: [{case, pages_url, n_pages, ckpt_urls}].
            # Порожньо — стара однокейсова поведінка.
            "cases": [dict(c) for c in (params.get("cases") or [])],
        }

        # 🔴 Слаг ВИВОДИТЬСЯ з датасету, а не задається окремо. Kaggle монтує
        # `<owner>/<slug>` рівно в `/kaggle/input/<slug>/`, а раннер шукає теку
        # за іменем слага — тож будь-яка розбіжність між `-p pages_dataset` і
        # `-p pages_slug` дає «немає теки входу» вже НА КЕРНЕЛІ, тобто після
        # черги, установки kraken і витраченої квоти. Виводимо самі, щоб цієї
        # пари не існувало.
        for kind in ("pages", "models", "scripts"):
            ds = out[f"{kind}_dataset"]
            if ds:
                out[f"{kind}_slug"] = ds.split("/")[-1]

        # 🔴 Порожній URL — це НЕ «беремо з локальних тек», це майже завжди
        # обірвана shell-змінна. Двічі за одну сесію (2026-08-11) посилання
        # генерувалось командою, запущеною не з того каталогу, підстановка
        # давала порожній рядок — і бокс піднімався, качав кадри, не отримував
        # моделей і **стояв живий та оплачуваний**, нічого не рахуючи.
        # Порожнеча помітна лише тут, до оренди; на боксі вона виглядає як
        # тиша.
        if out["cases"]:
            missing = [c.get("case") for c in out["cases"] if not c.get("pages_url")]
            if missing:
                raise ValueError(f"у черзі є справи без pages_url: {missing}")
            if not out["assets_url"]:
                raise ValueError("черга задана, а assets_url порожній — моделі не приїдуть")
        if not out["assets_url"] and out["pages_url"]:
            raise ValueError(
                "pages_url задано, а assets_url порожній — моделі й скрипти не "
                "приїдуть, і бокс стоятиме живим без роботи. Найчастіша причина: "
                "порожня shell-змінна (команду запущено не з того каталогу). "
                "Перевір посилання ДО оренди."
            )
        if not out["pages_url"] and out["assets_url"]:
            raise ValueError(
                "assets_url задано, а pages_url порожній — кадри не приїдуть"
            )
        return out

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        normalized = self.validate_params(params)
        n_pages = normalized["estimated_n_pages"] or normalized["limit"] or 200
        gpu = str(params.get("_gpu") or "")
        per_page = SEC_PER_PAGE_1SHARD.get(gpu, DEFAULT_SEC_PER_PAGE)
        speedup = min(float(normalized["shards"]), PARALLEL_CEILING)
        # +8 хв: підняття контейнера, копіювання входів з тому і завантаження ваг.
        return timedelta(seconds=n_pages * per_page / speedup + 480)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        # Бандла в списку немає навмисно: тексти й так лежать на томі посторінково
        # (щоб урвана робота не пропала), і пакувати їх удруге означало б качати
        # той самий корпус двічі. `-p bundle=true` — для випадку «хочу один файл».
        return ["htr_case_summary.json"]

    def is_output_complete(self, out_dir: Path) -> bool:
        """Чи можна вважати цю теку завершеним результатом справи.

        🔴 Дефолт `Job.is_output_complete` — «є хоч один файл». Саме він
        дозволив `fetch --resume` вважати справу забраною, коли на диску
        лежало 203 сторінки з 323. Тут перевіряються ТРИ незалежні числа:

        1. `complete` — вирок самого раннера (він рахував файли на боксі);
        2. `n_pages_expected` проти реальних `*.txt` У ЗАБРАНІЙ теці — бо
           обірватись міг і сам забір;
        3. пара `n_pages_input`/`n_pages_total` — стара ознака CUDA OOM,
           яка лишається чинною для результатів попередніх версій раннера.
        """
        summary = Path(out_dir) / "htr_case_summary.json"
        if not summary.is_file():
            return False
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False

        if data.get("complete") is False:
            return False
        if data.get("missing_pages"):
            return False

        expected = int(data.get("n_pages_expected") or data.get("n_pages_input") or 0)
        # 🔴🔴 КАРАНТИН НЕ ЗМЕНШУЄ ЗНАМЕННИКА. Доти карантиновані віднімались
        # від очікуваних, тобто справа з дірою проходила ворота. Замір
        # 18.08.2026 (ДАВіО): карантин з'їв п'ять сторінок, вердикт прийшов
        # `ok`, а локально всі п'ять узялись З ПЕРШОГО РАЗУ. Карантин не
        # доводить, що сторінка нечитабельна, — лише що ЦЕЙ бокс не впорався з
        # нею під цим навантаженням. Доказ читабельності один: текст на диску.
        texts = list((Path(out_dir) / "out").glob("*.txt"))
        if not texts:  # тексти могли лежати поруч, а не в out/
            texts = list(Path(out_dir).rglob("*.txt"))
        have = {t.stem for t in texts}
        empty_quarantine = [q for q in (data.get("quarantined_pages") or [])
                            if PurePath(str(q)).stem not in have]
        if empty_quarantine:
            return False
        if expected and len(texts) < expected:
            return False

        total = data.get("n_pages_total")
        return not (expected and total is not None and int(total) < expected)

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        """Kaggle-датасети, які платформа монтує в ``/kaggle/input/``.

        🔴 Базовий `Job.dataset_sources` повертає порожньо, і саме це, а не
        відсутність бекенда в `supported_backends`, було справжньою прогалиною:
        кернел піднявся б, поставив kraken, а тоді впав би на `немає теки
        входу 'pages'` — тобто витратив би квоту на нічого. Тут порожнеча
        стає помилкою ДО сабміту.

        Кадри можуть їхати й по HTTP (`pages_url` + `assets_url`) — тоді
        датасет не потрібен, бо кернел має інтернет і качає сам.
        """
        normalized = self.validate_params(params)
        sources = [normalized[f"{k}_dataset"] for k in ("pages", "models", "scripts")]
        if all(sources):
            # dict.fromkeys, а не set: порядок монтування лишається передбачуваним,
            # а спільний датасет (напр. ваги і скрипти в одному) не дублюється.
            return list(dict.fromkeys(sources))
        if normalized["pages_url"] and normalized["assets_url"]:
            return []
        if any(sources):
            missing = [k for k in ("pages", "models", "scripts")
                       if not normalized[f"{k}_dataset"]]
            raise ValueError(
                f"backend=kaggle: задано не всі датасети, бракує {missing}. "
                f"Часткового набору не буває: без ваг чи скриптів кернел падає "
                f"вже після установки kraken, витративши квоту."
            )
        raise ValueError(
            "backend=kaggle: задай -p pages_dataset/models_dataset/"
            "scripts_dataset=<owner>/<slug> або пару pages_url+assets_url. "
            "Без жодного з двох кернел не побачить ні кадрів, ні ваг."
        )

    def colab_input_dirs(self, params: dict[str, Any]) -> dict[str, str]:
        """Локальні теки, які бекенд має залити в ``/kaggle/input/<slug>/``.

        🔴 Порожньо, коли все їде по HTTP. Раніше сабміт вимагав локальні
        слаги НАВІТЬ тоді: доводилось робити теки `models/ pages/ scripts/` із
        файлом-заглушкою, інакше `run` падав із `job needs inputs [...]`. Це
        була чиста бюрократія — дані вже їхали з R2 власним каналом боксу,
        а SFTP-заливка (0.4 МБ/с) саме для того й обходилась.

        Умова саме на `assets_url` + `pages_url`: перший везе моделі й
        скрипти, другий — кадри. Якщо їде лише щось одне, решта все ще
        мусить приїхати локальними теками.
        """
        normalized = self.validate_params(params)
        if normalized["assets_url"] and normalized["pages_url"]:
            return {}
        return {normalized[k]: normalized[k]
                for k in ("pages_slug", "models_slug", "scripts_slug")}

    def modal_input_volumes(self, params: dict[str, Any]) -> dict[str, str]:
        return {"/vol": str(params.get("modal_volume")
                            or modal_volume_name(self.name, "gpurunner-htr"))}

    def modal_image_spec(self) -> dict[str, Any]:
        return {
            # 3.12 — та сама версія, під якою живе сам gpurunner. kraken 7.0.2 на
            # ній ставиться (перевірено `uv pip install --dry-run`); локальний
            # .venv_kraken на 3.11 — історія, а не вимога.
            "python_version": "3.12",
            "pip_packages": [
                # 🔴 Пін kraken з KRAKEN_PATCHES.md: обидва патчі підміняють ПРИВАТНІ
                # функції (`_calc_roi`, `boundary_tracing`, `sato`-виклик). Інша
                # версія змінить їх семантику ТИХО — інші полігони рядків, тобто
                # інший текст, без жодної помилки в лозі.
                "kraken==7.0.2",
                "torch",
                "torchvision",
                "timm>=0.9",
                "pytorch-lightning>=2.0",
                "nltk",
                "pillow>=10.0",
                "numpy",
                "scikit-image",
                "shapely",
                # strhub — код PARSeq; ваги Писаря без нього не інстанціюються
                "git+https://github.com/baudm/parseq.git",
            ],
            "extra_index_url": None,
            "apt_packages": ["git"],
            "timeout": 12 * 60 * 60,
            # Сегментація тримає ~1.6 ядра на шард (KRAKEN_PATCHES.md), тож ядер
            # треба не менше, ніж 2×shards, інакше шарди б'ються за CPU і карта
            # простоює.
            "cpu": 8,
            "memory": "32Gi",
        }

    def render_remote_code(self, params: dict[str, Any], *, shard_index: int = 0,
                           total_shards: int = 1) -> str:
        normalized = self.validate_params(params)
        if total_shards > 1:
            raise ValueError(
                "htr_case шардить сторінки ВСЕРЕДИНІ контейнера (-p shards=N), "
                "а не між контейнерами бекенда"
            )
        return self.render_kaggle_code("htr_case_runner.py", normalized)
