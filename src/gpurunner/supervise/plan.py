"""План заходу: що прогнати, звідки взяти дані, у які межі вкластись.

Формат навмисно тупий — JSON, який породжує `scripts/htr_cloud_plan.py` у
проєкті-замовнику (пакує кадри, кладе в R2, роздає presigned-посилання).
Наглядач його лише читає й перевіряє.

Перевірка тут не формальність: половина хмарних інцидентів починалась із
того, що чогось бракувало ще до оренди, а виявлялось це вже на оплачуваній
машині — presigned-посилання без підпису, порожній список кадрів, забутий
`seg_ceiling.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CasePlan:
    """Одна справа: звідки кадри й скільки їх."""

    case: str
    pages_url: str
    n_pages: int
    out_dir: str
    #: Шифра справи (`DAVO/885/1`) — координата, за якою декод знаходить свою
    #: книгу. Рахує її замовник (`htr_cloud_plan.py`), бо лише в нього тека
    #: кадрів і каталог є одночасно: на боксі `case_dir` — це `/tmp/htrcase/…`,
    #: а забраний прогін лишається з самим ІМЕНЕМ теки. Порожнє поле законне
    #: (тека може не бути архівною справою), але тоді прив'язка тримається на
    #: розборі імені — і «людське» ім'я на кшталт `bershad-678-79` означає, що
    #: для споживача декоду справа лишилась непрочитаною.
    case_key: str = ""
    #: Локальна тека, з якої складач плану пакував кадри. Після забору вона
    #: стає `case_dir` у меті: раннер пише туди шлях БОКСУ (`/tmp/htrcase/…`),
    #: якого вдома не існує, і гортач брав кропи не з тієї сторінки.
    local_dir: str = ""
    ckpt_urls: list[str] = field(default_factory=list)
    resume_urls: list[str] = field(default_factory=list)
    result_put_url: str = ""
    #: Спакований архів кадрів на НАШОМУ диску — транспорт `box`, де посилання
    #: ще не існує: його адреса це порт машини, якої поки немає.
    pages_path: str = ""
    #: Префікс чекпоінтів і скільки їх дозволено. При транспорті `box`
    #: посилання будуються з них уже на місці, коли машина піднялась.
    ckpt_prefix: str = ""
    ckpt_slots: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    #: Геометрія кадрів справи, зміряна складачем плану. Потрібна тут одна
    #: річ: розворот дорожчий за сторінку, і саме на ньому дефолтні 2.5 ГБ на
    #: шард дають OOM. Нуль означає «не міряли» — тоді лишається дефолт.
    frame_mpx_median: float = 0.0
    #: p95 площі кадру — саме від нього рахується VRAM на шард (пік іде від
    #: найбільшого кадру, а не від типового). Нуль = взяти медіану.
    frame_mpx_p95: float = 0.0
    frame_aspect_median: float = 0.0
    #: Щільність матеріалу: медіана рядків на сторінку. Джерело — мета
    #: попереднього прогону тієї самої справи (`out_dir/_htr_meta.json`) або
    #: ручка `-p lines_per_page`; нуль = не знаємо, і модель прогнозує так, як
    #: до члена за матеріалом (06.09.2026). Час сторінки — `A/(29+рядки)`, тож
    #: на 35 рядках і на 118 та сама карта дає 4594 і 1555 стор/год.
    lines_per_page_median: float = 0.0
    #: Скільки байтів кадрів справи качає бокс. Нуль = план старий. Потрібно
    #: воротам: чи встигає виміряний канал за флотом (`gate.py`).
    pages_bytes: int = 0
    #: Розпластати вивід раннера під конвеєр замовника.
    #:
    #: 🔴 Раннер кладе `out/*.txt` і `out-<голос>/*.txt` усередині теки справи,
    #: а споживач шукає прогони як ПЛАСКУ `reports/htr/<прогін>/*.txt` поруч із
    #: `_htr_meta.json`, і побічний голос — як СЕСТРИНСЬКУ `<прогін>-<голос>/`.
    #: Без перекладання успішний дорогий захід виглядає для `clan_hunt` як
    #: порожнеча БЕЗ помилки — той самий клас, що «нуль без знаменника».
    flatten_out: bool = False


@dataclass(frozen=True)
class Plan:
    """Захід цілком."""

    assets_url: str
    cases: list[CasePlan]
    #: `r2` — бакет S3 з presigned-посиланнями; `box` — склад на самій машині,
    #: куди файли кладе scp, а посилання видає її ж петля.
    transport: str = "r2"
    #: Архів ассетів на НАШОМУ диску (транспорт `box`).
    assets_path: str = ""
    gpu: str = "any"
    budget_usd: float = 3.0
    max_hours: float = 8.0
    #: 🔴 Було 120 ГБ — і це відсікало більшість ринку без жодної потреби.
    #: Замір на черзі з восьми справ ДАВО 904-24: кадри 8.3 ГБ, розпаковані
    #: ~9.4, моделі 105 МБ, вихід ~50 МБ. Сорок вистачає з великим запасом, а
    #: сто двадцять давали `disk_short` на здорових машинах (34347: 104 ГБ із
    #: замовлених 120 — і в чорний список за нашу ж завищену вимогу).
    #: Скільки ядер ХОЧЕТЬСЯ (0 — байдуже). Не поріг відсіву, а планка, заради
    #: якої варто зачекати: ядра — головна ручка швидкості (замір 2026-08-12:
    #: та сама RTX 3090, 64 ядра = 2548 стор/год, 32 ядра = 703), а ринок
    #: багатоядерних машин тонкий і оновлюється щохвилини. Чекання нічого не
    #: коштує: оренди ще немає.
    prefer_min_cores: float = 0.0
    #: Скільки хвилин чекати на машину з `prefer_min_cores`, перш ніж узяти те,
    #: що є. Нуль — не чекати зовсім.
    wait_for_cores_min: float = 0.0
    #: 🔴 Скільки годин бокс чекає на нас ПІСЛЯ того, як job завершився, перш
    #: ніж знищити себе сам. Не те саме, що жорсткий дедлайн `max_hours + 30хв`:
    #: той рахується від СТАРТУ й на довгому заході спрацьовує через півдоби.
    #: Замір 2026-08-12: наглядач помер у фазі забору, робота була ЗРОБЛЕНА, а
    #: бокс горів іще 4.44 год — $0.93 намарно, бо єдиним сторожем був жорсткий
    #: дедлайн. Півгодини з запасом вистачає здоровому наглядачу забрати
    #: результат (забір іде хвилини), а мертвий обмежує збиток тими ж
    #: півгодини замість повної стелі заходу.
    autodestroy_hours: float = 0.5
    disk_gb: int = 40
    #: Явна межа каналу, Мбіт/с; 0 = від обсягу кадрів плану (`gate.py`).
    min_net_mbps: float = 0.0
    num_gpus: int = 1
    #: 🔴 Стеля ЦІНИ ЗА ГОДИНУ. Була `None`, тобто стелі не існувало взагалі —
    #: і 2026-08-11 наглядач узяв 3×L40 за $1.36/год там, де вистачало RTX 3090
    #: за $0.19. Дефолт свідомо низький: наші заходи живуть у $0.10-0.35/год,
    #: а дорожча машина мусить бути ЯВНИМ рішенням людини.
    #: 🔴 ПОГОДИННА стеля ціни. Була 0.60 — тобто фактично її не було: стеля
    #: «за тисячу сторінок» пропускає дорогу карту, якщо вона швидка, і захід
    #: брав 4× RTX 3090 за $0.50/год. Рішення користувача 2026-08-11: дорогих
    #: карт не брати. 0.25 покриває весь клас RTX 3090 / P40 / 4070, на яких
    #: усе й рахується.
    max_price: float | None = 0.365
    #: Стеля вартості ОДНІЄЇ справи. Друга ручка, бо дешева година на повільній
    #: машині теж може вийти дорого.
    max_cost_per_case: float | None = None
    #: Скільки коштує година очікування (див. `offer_score.TIME_VALUE_*`).
    #: Підняти — наглядач ганятиметься за швидкістю; 0 — бере найдешевше.
    time_value_usd_per_hour: float | None = None
    #: Стеля вартості тисячі сторінок — головний поріг вибору машини.
    max_usd_per_1000_pages: float | None = None
    #: 🔴 ПІДЛОГА ТЕМПУ, стор/год. Стеля ціни пропускає скільки завгодно
    #: повільну машину, аби дешеву, і 21.09.2026 так і сталось: 6-ядерна
    #: TITAN X за $0.051/год проходила всі пороги при 218 стор/год. Ця ручка
    #: відповідає на інше питання — скільки заход ГОТОВИЙ ЧЕКАТИ. Тірами не
    #: послаблюється: порожній ринок має давати `market_empty`, а не згоду
    #: взяти будь-що. 0 = підлоги немає.
    min_pages_per_hour: float = 0.0
    max_attempts: int = 4
    #: 🔴 Скільки разів захід узагалі може взяти бокс. Не плутати з
    #: `max_attempts` — та стеля діє на ОДИН пошук кандидатів, а ця на весь
    #: захід. Без неї 2026-08-11 наглядач створив і знищив чотири інстанси за
    #: п'ять хвилин, бо кожна невдача вела до нового пошуку.
    #: Дефолт 2 = одна переоренда на випадок справді мертвої машини.
    #: 🔴 3, а не 2 (06.09.2026): бойовий захід на ф.230 чотирма сесіями
    #: одночасно вперся у три биті машини поспіль (V100 — обрив з'єднання під
    #: час сетапу двічі, RTX 3090 — не віддала контейнер за 6 хв), і дві сесії з
    #: чотирьох вийшли по стелі, доки робота ще не почалась; людині довелось
    #: перезапускати їх руками. Ціна невдалої оренди — $0.02–0.07, ціна
    #: перезапуску руками — півгодини простою.
    max_rents: int = 3
    gb_per_shard: float = 0.0          # 0 = дефолт планувальника
    #: Явне число шардів (0 = рахувати із заліза). 🔴 Читалось ЛИШЕ з `params`,
    #: тож `"shards": 6` у верхньому рівні плану не діяв узагалі й без
    #: попередження — 30.08.2026 це коштувало двох заходів поспіль із OOM, бо
    #: лікування «як у скілі» не діяло, і причину шукали не там.
    shards: int = 0
    catchup_passes: int = 2
    #: 🔥 Хвилин тримати бокс теплим після черги, приймаючи довісок. 0 = гасити
    #: одразу (поведінка до 06.09.2026). Простій коштує $0.0025/хв на 3090, а
    #: холодний старт — ~5 хв оренди плюс очікування ринку.
    keep_warm_min: float = 0.0
    #: sha256 скриптів у архіві ассетів (`htr plan` рахує з tgz). Раннер на
    #: боксі звіряє копію до старту: 05.09.2026 там лежав старий раннер.
    scripts_sha256: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    #: Облік замовника після ПОВНОГО забору: `[{"cmd": [...], "cwd": "...",
    #: "timeout_sec": 1800}]`. Наглядач живе відчеплено, тож крок «перебудуй
    #: реєстр / індекс» після заходу інакше лишався на пам'яті агента — і
    #: забувався (8591, 10.09.2026: облік робили руками). Команди — абсолютними
    #: шляхами: у відчепленого процесу PATH порожній.
    post_fetch: list[dict[str, Any]] = field(default_factory=list)
    #: Що в плані виглядає заданим, а насправді не діє. Друкується на старті
    #: заходу поруч із чинними ручками — тобто ДО оренди.
    warnings: list[str] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        return sum(c.n_pages for c in self.cases)


#: Ключі, які план розуміє на ВЕРХНЬОМУ рівні.
_KNOWN_TOP_KEYS = frozenset({
    "assets_url", "cases", "gpu", "budget_usd", "max_hours", "disk_gb",
    "autodestroy_hours", "prefer_min_cores", "wait_for_cores_min", "min_net_mbps",
    "num_gpus", "max_price", "max_cost_per_case", "time_value_usd_per_hour",
    "max_usd_per_1000_pages", "min_pages_per_hour", "max_attempts", "max_rents", "gb_per_shard",
    "vram_gb_per_shard", "shards", "catchup_passes", "params",
    "keep_warm_min", "scripts_sha256", "post_fetch",
})

#: Ключі, які план розуміє всередині справи.
_KNOWN_CASE_KEYS = frozenset({
    "case", "pages_url", "n_pages", "out_dir", "case_key", "case_dir",
    "ckpt_urls", "resume_urls", "result_put_url", "params", "flatten_out",
    "frame_mpx_median", "frame_mpx_p95", "frame_aspect_median", "lines_per_page_median",
    # 🔴 Ці чотири пише той самий складач (`htr/plan_build.py`) і читає цей
    # самий модуль (`pages_path` нижче, `ckpt_prefix`/`ckpt_slots` у
    # `CaseSpec`), а в переліку їх не було — тож сторож кричав «НЕ ДІЄ» про
    # ключі, які діють, на КОЖНОМУ заході. Найгірше тут не шум: попередження,
    # яке завжди бреше, привчає відмахуватись і від справжнього — а воно
    # ловить мовчазні дефолти на кшталт `max_rents`, за які вже платили
    # переоренди. Спіймано живим заходом 21.09.2026.
    "pages_path", "pages_bytes", "ckpt_prefix", "ckpt_slots",
})

#: Ключі плану, які в `params` НЕ діють. 🔴 Кладуться туди постійно, бо саме
#: так їх задає `-p`, а `-p` кладе все в `params`. Мовчазний дефолт замість
#: заданого числа: `-p max_rents=8` (17.08.2026) і `-p max_usd_per_1000_pages`
#: (19.08.2026) обидва не діяли й обидва не сказали про це ні слова.
_PLAN_ONLY_KEYS = frozenset({
    "keep_warm_min", "scripts_sha256",
    "budget_usd", "max_hours", "max_rents", "max_attempts", "disk_gb",
    "prefer_min_cores", "wait_for_cores_min", "max_price", "num_gpus",
    "autodestroy_hours", "catchup_passes", "gpu", "min_net_mbps",
})


def check_plan_keys(raw: dict[str, Any]) -> list[str]:
    """Попередження про ключі, покладені не туди. Помилкою НЕ є.

    🔴 Клас вади один: ключ, заданий не на тому рівні, не дає ні помилки, ні
    попередження — захід просто їде з дефолтом, і видно це лише за наслідками
    (OOM, перевитрата). Найгірше, що поруч у тому самому плані працює
    `max_usd_per_1000_pages`, тож перевірка «стеля підхопилась, значить план
    читається» дає хибну впевненість.

    Падати тут не можна: план могли зробити новішою версією планувальника, і
    невідомий ключ не привід не орендувати бокс. Але промовчати — теж не можна.
    """
    warnings: list[str] = []
    for key in sorted(set(raw) - _KNOWN_TOP_KEYS):
        warnings.append(f"ключ верхнього рівня `{key}` невідомий — НЕ ДІЄ")
    for key in sorted(set(raw.get("params") or {}) & _PLAN_ONLY_KEYS):
        warnings.append(
            f"`{key}` лежить у `params`, а це ключ ВЕРХНЬОГО рівня плану — "
            f"звідти він не читається ніколи")
    for i, case in enumerate(raw.get("cases") or [], 1):
        if not isinstance(case, dict):
            continue
        name = case.get("case") or f"№{i}"
        for key in sorted(set(case) - _KNOWN_CASE_KEYS):
            warnings.append(f"{name}: ключ `{key}` невідомий — НЕ ДІЄ")
    return warnings


def load_plan(path: str | Path) -> Plan:
    """Прочитати й перевірити план. Помилка тут коштує нуль; на боксі — гроші.

    JSON або YAML — за розширенням. YAML тут не косметика: план часто правлять
    руками (замінити справу, підняти бюджет), а `.yml` дозволяє коментарі й не
    падає через кому. Валідація для обох однакова.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ValueError("для YAML-плану потрібен pyyaml: uv add pyyaml") from e
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("план має бути об'єктом (JSON або YAML)")

    warnings = check_plan_keys(raw)

    transport = str(raw.get("transport") or "r2").strip() or "r2"
    if transport not in ("r2", "box"):
        raise ValueError(f"невідомий транспорт «{transport}»: буває `r2` (бакет "
                         f"S3) або `box` (склад на самій машині)")
    box = transport == "box"
    assets_path = str(raw.get("assets_path") or "").strip()
    assets_url = str(raw.get("assets_url") or "").strip()
    if assets_url and not assets_url.lower().startswith(("http://", "https://")):
        raise ValueError(f"`assets_url` не схожий на URL: {assets_url[:60]!r}")
    if box:
        # 🔴 Те саме правило, що й для посилання, лише іншими словами: без
        # архіву машина підніметься, поставить рушій і впаде на відсутньому
        # раннері — уже на оплачуваній карті.
        if not assets_path:
            raise ValueError(
                "транспорт `box`, а `assets_path` порожній — це архів із "
                "моделями й раннером, і саме його ми веземо на машину.")
        if not Path(assets_path).is_file():
            raise ValueError(f"архіву ассетів немає на диску: {assets_path}")
    elif not assets_url:
        raise ValueError(
            "у плані немає `assets_url` — це архів із моделями й скриптами. "
            "Без нього бокс підніметься, поставить kraken і впаде на відсутньому "
            "htr_case_run.py, тобто на оплачуваній карті."
        )

    cases_raw = raw.get("cases") or []
    if not cases_raw:
        raise ValueError("у плані немає жодної справи (`cases`)")

    cases: list[CasePlan] = []
    for i, item in enumerate(cases_raw, 1):
        case = str(item.get("case") or "").strip()
        pages_url = str(item.get("pages_url") or "").strip()
        pages_path = str(item.get("pages_path") or "").strip()
        n_pages = int(item.get("n_pages") or 0)
        if not case:
            raise ValueError(f"справа №{i}: немає імені (`case`)")
        if box:
            if not pages_path:
                raise ValueError(
                    f"{case}: транспорт `box`, а `pages_path` порожній — "
                    f"везти на машину нічого.")
            if not Path(pages_path).is_file():
                raise ValueError(f"{case}: архіву кадрів немає на диску: {pages_path}")
        elif not pages_url:
            # 🔴 Найчастіша причина — обірвана shell-змінна: команду генерації
            # запущено не з того каталогу, підстановка дала порожній рядок.
            # Двічі за одну сесію (2026-08-11) це давало живий оплачуваний бокс,
            # який стоїть і нічого не рахує. Тут це коштує нуль.
            raise ValueError(
                f"{case}: `pages_url` порожній. Найчастіше це обірвана "
                f"shell-змінна при генерації плану — перевір, що команда "
                f"запускалась із каталогу проєкту-замовника."
            )
        if pages_url and not str(pages_url).lower().startswith(("http://", "https://")):
            raise ValueError(f"{case}: `pages_url` не схожий на URL: {pages_url[:60]!r}")
        if n_pages <= 0:
            # 🔴 Знаменник обов'язковий. Без нього ворота повноти нічого не
            # перевіряють, а «нуль пропущених» — не факт, а відсутність факту.
            raise ValueError(
                f"{case}: `n_pages` = {n_pages}. Знаменник обов'язковий: без нього "
                f"повноту не підтвердити, і неповний результат виглядатиме повним."
            )
        cases.append(
            CasePlan(
                case=case,
                pages_url=pages_url,
                n_pages=n_pages,
                out_dir=str(item.get("out_dir") or ""),
                case_key=str(item.get("case_key") or ""),
                local_dir=str(item.get("case_dir") or ""),
                ckpt_urls=[str(u) for u in (item.get("ckpt_urls") or [])],
                resume_urls=[str(u) for u in (item.get("resume_urls") or [])],
                result_put_url=str(item.get("result_put_url") or ""),
                pages_path=pages_path,
                ckpt_prefix=str(item.get("ckpt_prefix") or ""),
                ckpt_slots=int(item.get("ckpt_slots") or 0),
                params=dict(item.get("params") or {}),
                frame_mpx_median=float(item.get("frame_mpx_median") or 0),
                frame_mpx_p95=float(item.get("frame_mpx_p95") or 0),
                frame_aspect_median=float(item.get("frame_aspect_median") or 0),
                lines_per_page_median=float(item.get("lines_per_page_median") or 0),
                pages_bytes=int(item.get("pages_bytes") or 0),
                flatten_out=bool(item.get("flatten_out")),
            )
        )

    budget = float(raw.get("budget_usd") or 0)
    max_hours = float(raw.get("max_hours") or 0)
    if budget <= 0 or max_hours <= 0:
        raise ValueError(
            "план мусить задавати `budget_usd` і `max_hours` — це єдині межі, "
            "усередині яких наглядач діє без питань до людини"
        )

    return Plan(
        assets_url=assets_url,
        cases=cases,
        transport=transport,
        assets_path=assets_path,
        gpu=str(raw.get("gpu") or "any"),
        budget_usd=budget,
        max_hours=max_hours,
        disk_gb=int(raw.get("disk_gb") or 40),
        autodestroy_hours=float(
            raw["autodestroy_hours"] if raw.get("autodestroy_hours") is not None else 0.5
        ),
        prefer_min_cores=float(raw.get("prefer_min_cores") or 0),
        wait_for_cores_min=float(raw.get("wait_for_cores_min") or 0),
        min_net_mbps=float(raw.get("min_net_mbps") or 0.0),
        num_gpus=int(raw.get("num_gpus") or 1),
        max_price=(float(raw["max_price"]) if raw.get("max_price") is not None else 0.365),
        max_cost_per_case=(float(raw["max_cost_per_case"])
                           if raw.get("max_cost_per_case") is not None else None),
        time_value_usd_per_hour=(float(raw["time_value_usd_per_hour"])
                                 if raw.get("time_value_usd_per_hour") is not None else None),
        max_usd_per_1000_pages=(float(raw["max_usd_per_1000_pages"])
                                if raw.get("max_usd_per_1000_pages") is not None else None),
        min_pages_per_hour=float(raw.get("min_pages_per_hour") or 0),
        max_attempts=int(raw.get("max_attempts") or 4),
        max_rents=int(raw.get("max_rents") or 3),
        # 🔴🔴 Два імені однієї ручки. У `-p` вона зветься `vram_gb_per_shard`
        # (так її називає й скіл), а полем плану була тільки `gb_per_shard` — і
        # верхній рівень плану читав ЛИШЕ друге. Мовчки: план валідний, захід
        # їде, у лозі «VRAM/шард авто». Ціна 30.08.2026 — три заходи поспіль із
        # OOM (34 і 53 події) при двічі виставленій «як у скілі» ручці.
        gb_per_shard=float(raw.get("gb_per_shard")
                           or raw.get("vram_gb_per_shard") or 0),
        shards=int(raw.get("shards") or 0),
        catchup_passes=int(raw.get("catchup_passes", 2) or 0),
        keep_warm_min=float(raw.get("keep_warm_min") or 0),
        scripts_sha256={str(k): str(v) for k, v in (raw.get("scripts_sha256") or {}).items()},
        params=dict(raw.get("params") or {}),
        post_fetch=[dict(h) for h in (raw.get("post_fetch") or [])
                    if isinstance(h, dict) and h.get("cmd")],
        warnings=warnings,
    )
