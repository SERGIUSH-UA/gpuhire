"""ParseqTrainJob — fine-tune PARSeq-S (strhub) на кириличному line-GT.

Кейс: власний кириличний HTR-движок. Вчителі (фінська TrOCR-large,
Hukyl/trocr-large-uk-handwritten-real) дають псевдо-GT рядків; PARSeq-S у 20×
швидша за них при порівнянній якості → після тюну стає і вчителем, і прод-
движком (заміна kraken-учня для кирилиці).

Вхід — тека/датасет з кропами рядків + TSV-маніфестом (``gt*.txt``:
``relpath<TAB>label``) або JSONL (``{"image": ..., "text": ...}``);
опційно окремий датасет із базовим чекпойнтом ``*.pt``.

⚠ **Заливати кропи ОДНИМ ``.tgz``**, не текою з тисячами файлів: Lightning
(``_upload_dir``) і Kaggle CLI ллють пофайлово, тож 24k кропів = години проти
хвилин на один архів. Runner сам розпакує будь-який ``*.tgz`` із входу —
пакувати треба разом із маніфестом (``tar czf x.tgz -C <тека> .``), бо шляхи
в ``gt*.txt`` резолвляться відносно теки самого маніфесту.
Вихід — ``parseq_best.pt`` у форматі Hukyl (model_state + charset + config),
тобто одразу придатний для job'а ``htr_lines_eval``, і ``parseq_last.pt`` —
повний стан для продовження.

**Продовження перерваного трену.** Сесія Kaggle живе 12 год, трен на 100k
рядків — довше, тож обрив колись обходився в усі витрачені години. Тепер
кожна епоха дописує ``parseq_last.pt`` (ваги + optimizer + scheduler + scaler
+ лічильники ранньої зупинки + крива). Щоб доучити::

    gpurunner fetch <handle>                       # забрати parseq_last.pt
    gpurunner dataset push <тека> -m "run N"       # залити його датасетом
    gpurunner run parseq_train ... -p resume=true -p resume_dataset=<slug>

Параметри мусять збігтися з тими, з якими трен починався: раннер звіряє
fingerprint (charset, розміри, batch, epochs, lr, обсяг train) і відмовляється
продовжувати при розбіжності. Це навмисно — мовчазний рестарт lr-циклу
виглядав би як раптове погіршення моделі без сліду в лозі.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.config import modal_volume_name
from gpurunner.core.job import Job

DEFAULT_PRETRAINED = "Hukyl/parseq-s-cyrillic-handwritten"

_TRUE = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSE = frozenset({"0", "false", "no", "off", "n", "f", ""})


def _as_bool(value: Any, name: str) -> bool:
    """Булевий параметр із CLI.

    🔴 Не `bool(value)`. CLI JSON-декодує `-p amp=false` у `False`, але `-p amp=no`
    лишається РЯДКОМ "no", а `bool("no")` — це `True`. Тобто спроба вимкнути amp
    чи profile мовчки вмикала їх, і дізнатися про це можна було лише з логу
    трену, який уже їхав. Невідоме значення краще відхилити локально.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"{name} must be true|false (got {value!r})")


class ParseqTrainJob(Job):
    """PARSeq-S recognition fine-tune over line crops + TSV/JSONL labels."""

    name: ClassVar[str] = "parseq_train"
    description: ClassVar[str] = (
        "Fine-tune PARSeq-S (strhub) on line crops + gt.txt manifest; charset "
        "extension for pre-reform Cyrillic; outputs Hukyl-format best.pt."
    )
    supported_backends: ClassVar[tuple[str, ...]] = (
        "kaggle",
        "colab",
        "vast",
        "lightning",
        "beam",
        "saturn",
        "modal",
    )

    def requirements(self) -> list[str]:
        return ["timm>=0.9", "pytorch-lightning>=1.9", "nltk", "pillow>=10.0"]

    # ---- Modal ------------------------------------------------------------
    # 🔴 Три речі, якими Modal відрізняється від решти бекендів, і кожна вміє
    # спалити карту намарне:
    #  1. /kaggle/{input,working} НЕ емулюються — раннер прив'язує корені сам
    #     (_bind_roots), вхід чекає у томі, вихід у params["output_root"];
    #  2. хвилина контейнера тарифікується РАЗОМ із GPU, тож `pip install` на
    #     старті — це гроші за простій карти. Тому всі залежності, включно з
    #     strhub (parseq), запікаються в образ: збірка образу йде на CPU-білдері
    #     Modal і кешується шарами, а _ensure_deps() у раннері тоді нічого не
    #     ставить;
    #  3. вхідний том підключається БЕЗ create_if_missing — якщо його нема,
    #     submit падає локально, до спавну контейнера. Це навмисно.

    def modal_image_spec(self) -> dict[str, Any]:
        return {
            # 🔴 Мусить збігатися з Python, під яким живе САМ gpurunner (3.12).
            # Modal серіалізує функцію-обгортку (`serialized=True`) і при
            # розбіжності версій відмовляється її спавнити:
            #   «'execute' was defined with Python 3.12, but its Image has 3.11».
            # Перша редакція ставила 3.11 «під локальний .venv_kraken», щоб ваги
            # читались тим самим pickle — але це хибна тривога: torch.save пише
            # zip-формат, який 3.11 читає незалежно від версії писаря, а от
            # спавн через розбіжність падає гарантовано. Ціна помилки — збірка
            # образу (~83 с) і жодного контейнера, тобто грошей 0, але сабміт
            # не проходить.
            "python_version": "3.12",
            "pip_packages": [
                "torch>=2.2",
                "torchvision",
                "timm>=0.9",
                "pytorch-lightning>=2.0",
                "nltk",
                "pillow>=10.0",
                "numpy",
                # strhub — сам PARSeq; без нього раннер тягнув би його з git
                # уже на оплачуваній карті (~2-3 хв × ціна GPU щоразу)
                "git+https://github.com/baudm/parseq.git",
            ],
            "extra_index_url": None,
            "apt_packages": ["git"],
            "timeout": 12 * 60 * 60,
            # Ядра для dataloader'а. Modal без запиту дає 2 → раннер бере
            # воркерів як (CPU−1)÷карт, тобто ОДИН воркер.
            # ⚠ Чесно про підставу: замір v11 на 2×T4 показав `data_pct` **0.1%**
            # при одному воркері — на T4 dataloader НЕ був вузьким місцем, і
            # твердження «карта чекає на PNG» до цієї конфігурації не належить.
            # Але 0.1% виміряно при 97 рядках/с; A100 просить утричі більше, і
            # чи витягне це один воркер — невідомо. 4 ядра (~$0.32/год проти
            # $2.10 за карту) — дешева страховка, а не доведена потреба.
            # 🔴 Перевіряти в `smoke` по `data_pct`: >40% і раннер сам скаже.
            "cpu": 4,
        }

    def modal_input_volumes(self, params: dict[str, Any]) -> dict[str, str]:
        """Кропи + маніфест лежать у томі, залитому ЗАЗДАЛЕГІДЬ, out-of-band:

            modal volume create <том>           # один раз
            modal volume put <том> <corpus>.tgz /

        Ім'я тому — `-p modal_volume=<том>` або `GPURUNNER_MODAL_VOLUME`.

        Раннер розпакує будь-який ``*.tgz`` під цим коренем сам.
        """
        return {"/vol": str(params.get("modal_volume")
                            or modal_volume_name(self.name, "gpurunner-htr"))}

    def modal_secrets(self, params: dict[str, Any]) -> list[str]:
        return []

    def render_runner_module(self) -> str:
        """Modal: той самий раннер, але БЕЗ інжекції PARAMS і хвостового виклику.

        Kaggle-шлях (`render_remote_code`) склеює нотбук: блок PARAMS + `_common`
        + раннер + `main(PARAMS)`. Обгортка Modal натомість сама викликає
        `main(params)`, тож тут потрібне рівно те саме тіло без обрамлення.
        `_common` лишається — щоб на обох бекендах виконувався ідентичний код, а
        не «майже ідентичний».
        """
        return "\n\n".join([
            "# --- gpurunner common prelude ---",
            self._runner_source("_common.py", strip_main=False),
            "# --- embedded runner ---",
            self._runner_source("parseq_train_runner.py", strip_main=True),
        ])

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if "dataset" not in params:
            raise ValueError(
                "parseq_train needs 'dataset=<тека|owner/slug>' з кропами + gt.txt"
            )
        charset_mode = str(params.get("charset_mode", "extend")).strip()
        if charset_mode not in ("keep", "extend"):
            raise ValueError(f"charset_mode must be keep|extend, got {charset_mode!r}")
        augment = str(params.get("augment", "basic")).strip()
        if augment not in ("basic", "none"):
            raise ValueError(f"augment must be basic|none, got {augment!r}")
        out: dict[str, Any] = {
            "dataset": str(params["dataset"]).strip(),
            "pretrained_dataset": str(params.get("pretrained_dataset", "")).strip(),
            "pretrained": str(params.get("pretrained", DEFAULT_PRETRAINED)).strip(),
            # 🔴 Продовження перерваного трену. GPU-сесія Kaggle живе 12 год, а
            # трен на 100k рядків — довше; без цього обрив означав почати все
            # спочатку. Раннер шукає parseq_last.pt у входах, звіряє fingerprint
            # (charset, розміри, batch, epochs, lr, n_train) і доучує з тієї
            # епохи, на якій зупинився — разом зі станом optimizer і OneCycle.
            "resume": _as_bool(params.get("resume", False), "resume"),
            "resume_dataset": str(params.get("resume_dataset", "")).strip(),
            "charset_mode": charset_mode,
            "charset_extra": str(params.get("charset_extra", "")),
            "charset_min_freq": int(params.get("charset_min_freq", 20)),
            "epochs": int(params.get("epochs", 20)),
            "batch": int(params.get("batch", 64)),
            "lr": float(params.get("lr", 3e-4)),
            "warmup_pct": float(params.get("warmup_pct", 0.075)),
            "weight_decay": float(params.get("weight_decay", 0.0)),
            "val_frac": float(params.get("val_frac", 0.08)),
            "img_h": int(params.get("img_h", 0)),
            "img_w": int(params.get("img_w", 0)),
            "max_label_length": int(params.get("max_label_length", 0)),
            "augment": augment,
            # 0 = визначити на місці ((CPU−1) ÷ карт), −1 = без воркерів.
            # Жорстке число тут було кагл-специфічним: на Modal/Vast із 16
            # ядрами воно лишало dataloader на двох воркерах, і карта чекала б.
            "workers": int(params.get("workers", 0)),
            "amp": _as_bool(params.get("amp", True), "amp"),
            # auto → bf16 на Ampere+ (Modal A10G/A100, Vast 3090/4090),
            # fp16 на T4 (Kaggle/Lightning — bf16 там немає апаратно)
            "amp_dtype": str(params.get("amp_dtype", "auto")).strip().lower(),
            # auto = всі видимі карти (Kaggle віддає 2×T4), off = одна, N = рівно N
            "ddp": str(params.get("ddp", "auto")).strip().lower(),
            "ddp_find_unused": _as_bool(
                params.get("ddp_find_unused", False), "ddp_find_unused"),
            # 15, не 5. Валідація тепер шардована по рангах, тож довгого
            # односіннього очікування бути не має — але таймаут накриває і
            # найповільнішу колективну операцію на холодному старті NCCL, а
            # ціна зайвих 10 хв тут нульова проти вбитого на рівному місці трену.
            "ddp_timeout_min": float(params.get("ddp_timeout_min", 15)),
            # як правити lr під більший ефективний batch: sqrt|linear|none
            "lr_scale": str(params.get("lr_scale", "sqrt")).strip().lower(),
            "compile": _as_bool(params.get("compile", False), "compile"),
            # patience 3, не 5: на v5b чотири з семи епох пішли в нікуди після
            # того, як модель вийшла на плато (best був ep3, трен ішов до ep7).
            "patience": int(params.get("patience", 3)),
            # 🔴 ПІДЛОГА ранньої зупинки. Терпіння дивиться на val_cer, а val у
            # нашого збірника — 155 рядків із ТИХ САМИХ справ, що й train; наш
            # критерій (recall прізвищ на архівному рукописі, інші справи)
            # міряється ПОТІМ, локально, по всіх збережених епохах.
            # Тобто рання зупинка може зрізати епоху, яка за val посередня, а
            # за нашим критерієм найкраща — і дізнатися про це вже нема з чого.
            # На кривій v11 найкращий val був на ep9 ПІСЛЯ провалу на ep8.
            # 0 = без підлоги (стара поведінка).
            "min_epochs": int(params.get("min_epochs", 0)),
            # Поріг значущості покращення val_cer. Було зашито 1e-4 (0.01 пп) —
            # це стільки ж, скільки нічого: 3 символи з 4500 скидали лічильник.
            "min_delta": float(params.get("min_delta", 0.002)),
            # Ваги КОЖНОЇ епохи окремим файлом (~95 МБ × epochs). Дефолт ON:
            # епоху ми обираємо локально по повному holdout, а не за val_cer,
            # і без цих файлів вибирати нема з чого — контейнер уже згорів.
            "save_epochs": _as_bool(params.get("save_epochs", True), "save_epochs"),
            # Modal: назва тому з кропами (див. modal_input_volumes).
            "modal_volume": str(params.get("modal_volume")
                                or modal_volume_name(self.name, "gpurunner-htr")).strip(),
            # cpu / memory контейнера. Мусять пройти крізь validate_params, бо
            # бекенд читає їх саме з НОРМАЛІЗОВАНИХ params (`-p cpu=8` інакше
            # мовчки не діяв би, і замість більшого dataloader'а ми отримали б
            # ту саму двоядерну машину — без жодного сліду в лозі).
            **({"cpu": float(params["cpu"])} if params.get("cpu") else {}),
            **({"memory": int(params["memory"])} if params.get("memory") else {}),
            "seed": int(params.get("seed", 42)),
            "limit": int(params.get("limit", 0)),
            "val_limit": int(params.get("val_limit", 0)),
            # профіль data-vs-compute у логу кожні 50 кроків і в summary
            "profile": _as_bool(params.get("profile", True), "profile"),
            # обрізати епоху на N кроках — для швидкого профільного прогону
            # (ціла епоха на 100k рядків це 33 хв, 200 кроків — дві)
            "max_steps": int(params.get("max_steps", 0)),
            # 🔴 САМОЗУПИНКА за годинником. GPU-сесія Kaggle живе 12 год;
            # обрив по ліміту вбиває кернел посеред роботи, і чи вціліє
            # /kaggle/working — НЕ гарантовано (перевірено 2026-07-29:
            # після cancel у виході лишився лише .log, ваги пропали разом
            # із 12 год квоти). 11 год лишає запас на запис і summary.
            "wall_limit_h": float(params.get("wall_limit_h", 11.0)),
        }
        if out["batch"] < 1:
            raise ValueError("batch must be >= 1")
        if out["epochs"] < 1:
            raise ValueError("epochs must be >= 1")
        if not 0.0 <= out["val_frac"] < 0.9:
            raise ValueError("val_frac must be in [0, 0.9)")
        if out["amp_dtype"] not in ("auto", "fp16", "bf16", "fp32"):
            raise ValueError("amp_dtype must be auto|fp16|bf16|fp32")
        if out["lr_scale"] not in ("sqrt", "linear", "none"):
            raise ValueError("lr_scale must be sqrt|linear|none")
        if out["ddp"] not in ("auto", "on", "off") and not out["ddp"].isdigit():
            raise ValueError("ddp must be auto|on|off|<кількість карт>")
        # Від'ємний ліміт вмикав wall-guard і зупиняв трен одразу після першої
        # епохи: «менше нуля» тихо означало «зупинись негайно». 0 = вимкнено.
        if out["wall_limit_h"] < 0:
            raise ValueError("wall_limit_h must be >= 0 (0 = вимкнено)")
        if out["patience"] < 0:
            raise ValueError("patience must be >= 0 (0 = без ранньої зупинки)")
        if out["min_epochs"] < 0:
            raise ValueError("min_epochs must be >= 0 (0 = без підлоги)")
        if out["min_epochs"] > out["epochs"]:
            # Мовчазне обрізання тут означало б, що замовлена підлога вища за
            # стелю і жодна з них не діє так, як написано в команді.
            raise ValueError(
                f"min_epochs ({out['min_epochs']}) > epochs ({out['epochs']})")
        if not 0.0 <= out["warmup_pct"] <= 1.0:
            # йде в pct_start OneCycleLR, який поза [0,1] кидає незрозуміле
            raise ValueError("warmup_pct must be in [0, 1]")
        if out["charset_min_freq"] < 1:
            raise ValueError("charset_min_freq must be >= 1")
        if out["ddp_timeout_min"] <= 0:
            raise ValueError("ddp_timeout_min must be > 0")
        if out["lr"] <= 0:
            raise ValueError("lr must be > 0")
        if out["min_delta"] < 0:
            raise ValueError("min_delta must be >= 0")
        if out["limit"] < 0 or out["val_limit"] < 0 or out["max_steps"] < 0:
            raise ValueError("limit / val_limit / max_steps must be >= 0 (0 = без обмеження)")
        if out["pretrained"] == "local" and not out["pretrained_dataset"]:
            raise ValueError(
                "pretrained=local потребує pretrained_dataset=<тека з *.pt>"
            )
        if out["resume"] and not out["resume_dataset"]:
            raise ValueError(
                "resume=true потребує resume_dataset=<slug з parseq_last.pt> — "
                "залий вихід попереднього прогону датасетом і вкажи його тут"
            )
        return out

    # Хвилин на епоху зі 100k рядків, ОДНА карта, parseq-s / batch 64 / amp.
    #
    # 🔴 Точка відліку — ВИМІРЯНА, не виведена: `ptrain_summary.json` прогону
    # v11 (`pysar_v11_full`) — 101 169 рядків, **17,4 хв/епоха на 2×T4 DDP**,
    # 97 рядків/с, 12 епох = 3,5 год. При вимiряному прискоренні DDP ×1.88 це
    # ~33 хв на ОДНІЙ T4.
    # ⚠ Перша редакція цієї таблиці стояла на 62 хв: 33 хв узялось із
    # коментаря в коді й було ПОМНОЖЕНО на 1.88, хоча 33 вже й було числом
    # для однієї карти. Уся таблиця виходила вдвічі песимістичною, а з нею і
    # вибір карти. Тому тут тепер стоїть посилання на конкретний файл заміру.
    #
    # Решта — від співвідношення пікових TFLOPS, свідомо занижена (батч малий,
    # повний виграш великої карти не реалізується). 🔴 Ці числа лишаються
    # ОЦІНКОЮ: реальну швидкість конкретної карти дає лише димовий прогін
    # (`pysar_modal_train.py smoke`), і після нього таблицю варто підправити.
    _MIN_PER_EPOCH_100K: ClassVar[dict[str, float]] = {
        "none": 600.0,
        "T4": 33.0,        # ← ВИМІРЯНО (17.4 хв на 2×T4 × 1.88 прискорення DDP)
        "L4": 18.0,        # оцінка
        "A10G": 16.0,      # оцінка
        "A100": 6.0,       # ← ВИМІРЯНО (димовий прогін edd08f1a: 331 рядків/с,
        "A100-80GB": 6.0,  #   тобто 5.2 хв на 103k; +margin на val і запис ваг)
        "H100": 4.0,       # оцінка
    }
    # 🔴 Виміряні точки виявились ДАЛІ одна від одної, ніж дає співвідношення
    # TFLOPS: A100 швидший за одну T4 не втричі, а в ~5.5 раза (331 рядків/с
    # проти 97 на ДВОХ T4). Тому L4/A10G/H100 лишаються оцінкою і позначені як
    # оцінка — інтерполювати між двома вимірами по пікових флопсах виявилось
    # ненадійно, і робити вигляд, що ці числа рівноцінні виміряним, не варто.

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        normalized = self.validate_params(params)
        gpu = str(params.get("_gpu") or "T4")
        per_epoch = self._MIN_PER_EPOCH_100K.get(gpu, 62.0)
        # Обсяг корпусу наперед невідомий (він усередині .tgz), тож або явний
        # limit, або поточний робочий корпус ~100k рядків. `-p est_rows=` —
        # щоб прорахунок ціни не брехав на іншому корпусі.
        rows = int(params.get("est_rows") or normalized["limit"] or 100_000)
        minutes = per_epoch * (rows / 100_000.0) * normalized["epochs"]
        # +8 хв на старт: розпакування корпусу, побудова charset, завантаження
        # базових ваг із HF і baseline-валідація. Було 20 — число з Kaggle, де
        # розпакування корпусу з /kaggle/input тривало десятки хвилин; на Modal
        # том локальний, і замір дав 5 с на розпакування й ~1 хв до першого
        # кроку. Різниця йде з того самого оплачуваного вікна, тож брехати в
        # більший бік теж не безкоштовно — на цьому й будується вибір карти.
        return timedelta(minutes=minutes + 8)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        # parseq_last.pt — повний стан для -p resume=true (ваги + optimizer +
        # scheduler); без нього продовжити перерваний трен неможливо
        return ["parseq_best.pt", "parseq_last.pt", "ptrain_summary.json", "ptrain.log"]

    def _input_refs(self, normalized: dict[str, Any]) -> tuple[str, ...]:
        return (
            normalized["dataset"],
            normalized["pretrained_dataset"],
            normalized["resume_dataset"],
        )

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        normalized = self.validate_params(params)
        return [r for r in self._input_refs(normalized) if r and "/" in r]

    def colab_input_dirs(self, params: dict[str, Any]) -> dict[str, str]:
        """Приймає і '<owner>/<slug>' (Kaggle), і голу назву теки (lightning/vast)."""
        normalized = self.validate_params(params)
        out: dict[str, str] = {}
        for ref in self._input_refs(normalized):
            if ref:
                slug = ref.split("/versions/", 1)[0].split("/")[-1]
                out[slug] = slug
        return out

    def render_remote_code(
        self, params: dict[str, Any], *, shard_index: int = 0, total_shards: int = 1
    ) -> str:
        normalized = self.validate_params(params)
        if total_shards > 1:
            raise ValueError("parseq_train doesn't support sharding")
        return self.render_kaggle_code("parseq_train_runner.py", normalized)
