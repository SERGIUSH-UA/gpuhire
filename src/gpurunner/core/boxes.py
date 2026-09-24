"""Реєстр орендних боксів: що ця машина обіцяла і що вона насправді зробила.

Маркетплейс продає **картку оффера**, а не машину. Картка бреше системно:
оффер із «755 Mbps» віддавав 0.5 і з'їв годину; бокс, що рапортував 96 ядер,
видав менше за 64-ядерний V100; хост із валідним ключем відповідав
«Permission denied» 15 хвилин поспіль, і всі 15 хвилин тарифікувались. Жодну з
цих властивостей не видно до оренди — але кожна лишається за машиною й
повторюється.

Тому кожна оренда лишає тут рядок: **обіцяне поруч із виміряним**. З цього
фолдиться вердикт, який керує наступним пошуком: погану машину не пропонувати
(`machine_id.notin`), добру — шукати адресно (`machine_id.eq`).

Два файли, дві ролі, жодного дублювання істини:

- `boxes.jsonl` (у `registry_dir()`) — пише **тільки код**, дописом одного рядка. Append-only
  знімає весь клас lost-update без транзакцій: два наглядачі дописують
  паралельно й нічого не затирають. Теку варто тримати у власному git — у дифі видно, коли
  й **чому** бокс став поганим.
- `boxes.overrides.json` — пише **тільки людина**. Ручне рішення не
  змішується з виміром і не протухає по TTL.

Ключ — `machine_id` (фізична машина). `offer_id` ефемерний і зникає після
оренди; `host_id` зберігаємо для майбутньої агрегації, але банимо саме машину:
у одного хоста бувають і добрі, і гнилі.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gpurunner.config import registry_dir

# ---- таксономія ------------------------------------------------------------

#: Успіх — справа доїхала повністю (ворота повноти сказали «повно»).
OK = "ok"

#: Скільки днів вердикт лишається чинним. Машину могли полагодити, тож вічних
#: банів немає — крім явного ручного `never`.
TTL_DAYS: dict[str, int] = {
    # 🔴 Було 30 днів з одного удару. Відхилений ключ виявився здебільшого
    # збоєм прив'язки на боці Vast: 138268 мала робочі сесії за 20 хв ДО і
    # через 20 хв ПІСЛЯ відмови (03.09.2026), три відмови 10.09 лягли за дві
    # хвилини на три різні машини, а 12728 (15.09) пройшла з другої ж спроби.
    "ssh_auth_denied": 3,
    # 🔴 Було 30 днів з ОДНОГО удару — і це виявилось наклепом: машина,
    # що довго тягне 3-ГБ образ, не зламана. Повільний старт має
    # забуватись, справжня непрацездатність — лишатись.
    "never_booted": 14,      # контейнер не створився (темплейт, CDI, битий образ)
    "died_under_load": 21,   # зник посеред роботи
    "host_destroyed": 21,    # інстанс знищено не нами
    "slow_net": 14,          # канал заміряний, а не заявлений
    "slow_net_after_ok": 3,  # те саме на машині з успіхом — див. SOFT_AFTER_OK
    "cpu_lie": 14,           # ядер менше, ніж у картці
    "vram_lie": 14,          # VRAM менше, ніж у картці
    "ssh_unreachable": 3,    # міг бути мережевий збій — забуваємо швидко
    # 🔴 Карта ЗАЙНЯТА чужим орендарем: `nvidia-smi` дає 5.1 з 24 ГБ вільних, і
    # флот сідає в один шард замість восьми. Це НЕ `vram_lie` — карта не мала,
    # вона зараз не наша, і хост сам по собі справний (139040 мала 5 успіхів
    # поспіль). Тому три дні, а не чотирнадцять: сусід піде.
    "card_busy": 3,
    # Хост без нашого образу в кеші: не зламаний, але дорогий на підйомі
    # («від хвилин до годин», за підказкою Vast). Забувається швидко — образ
    # міг просто оновитись, і наступного разу кеш уже теплий.
    "slow_boot": 3,
    # Сетап зірвався після оренди, а класифікувати нічим (збій заливки входів,
    # незрозуміла помилка з живим інстансом). Провина машини НЕ доведена, тож
    # забуваємо швидко і банимо лише за повторюваністю.
    "setup_failed": 3,
    # Машина читала повільніше, ніж обіцяв прогноз, — за словами ВИКЛИКАЧА, що
    # орендував її без наглядача (плагін `nyshporka.cloud`, `release(why="slow:…")`).
    # Замір не наш і причина невідома (сусід на хості, тротлінг, матеріал), тож
    # це не `cpu_lie` з його баном з одного удару: тиждень і два удари.
    "slow_run": 7,
    # 🔴 Ворота заліза: на ВИМІРЯНОМУ залізі (квота cgroup, вільна VRAM карт)
    # машина не дає цілі темпу, хоч картка оффера обіцяла. Замір однозначний і
    # зроблений до першої сторінки, але ціль — наша вимога, а не вада хоста,
    # тож три дні: машина могла віддати меншу квоту саме цього разу.
    "below_target": 3,
}

#: Скільки разів має статись, щоб забанити. Одиниця там, де замір однозначний;
#: двійка — де могла бути погода (той самий двоударний карантин, що в локальній
#: черзі Нишпорки: перше зависання не карантинить сторінку).
HITS_TO_BAN: dict[str, int] = {
    "ssh_auth_denied": 2,
    "never_booted": 2,
    "slow_net": 1,
    "slow_net_after_ok": 2,
    "cpu_lie": 1,
    "vram_lie": 1,
    "ssh_unreachable": 2,
    "slow_boot": 2,
    # Замір однозначний і зроблений до першої сторінки — другого удару не треба.
    "card_busy": 1,
    "died_under_load": 2,
    "host_destroyed": 2,
    "setup_failed": 3,
    "slow_run": 2,
    "below_target": 1,
}

#: Ці вироки НЕ знімаються пізнішим успіхом — бо успіху на такій машині не
#: буває за побудовою: якщо хост не приймає ключ, ми там ніколи нічого не
#: порахуємо. Решта — знімається: `ok` означає, що те саме залізо цього разу
#: пройшло всі заміри.
#: `never_booted` тут БІЛЬШЕ НЕМАЄ: пізніший успіх на тій самій машині —
#: прямий доказ, що контейнер таки створюється, і старий вирок хибний.
#:
#: 🔴 Липкість діє, ЛИШЕ поки успіху на машині не було ЖОДНОГО (див. `_fold_one`).
#: Заміряно 2026-08-11: машина 144417 о 20:03 прогнала справу повністю (`ok`), а
#: о 20:19 відхилила той самий ключ — і поїхала в бан на 30 днів із поясненням
#: «успіху тут не буває за побудовою», яке спростовував запис у тому самому
#: реєстрі за 16 хвилин до того. Тобто твердження було не про хост, а про
#: транзієнтний збій прив'язки ключа на боці Vast — і воно назавжди викидало
#: з ринку єдину машину, яка щойно довела, що працює.
STICKY: frozenset[str] = frozenset({"ssh_auth_denied"})

#: 🔴 Повільний канал ПІСЛЯ успіху — погода маршруту, а не вада хоста.
#: Машина 35928 (2080 Ti×2) 13.09.2026 о 10:22 дала 21.5 Мбіт/с і прогнала
#: чергу повністю, а о 12:10 той самий замір дав 15.1 — і поїхала в бан на
#: 14 днів з одного удару. Ненульовий канал на машині, яка вже доводила, що
#: працює, судиться як транзієнт: два удари й коротке TTL. Нуль (`000`, канал
#: мертвий) і машини без жодного успіху лишаються за старим правилом.
SOFT_AFTER_OK: dict[str, str] = {"slow_net": "slow_net_after_ok"}

#: 🔴🔴 Вироки, які виносив ЗЛАМАНИЙ ІНСТРУМЕНТ, — недійсні.
#:
#: До 2026-08-12 усі три таймери підйому міряли ГОЛИЙ ЧАС, не дивлячись на те,
#: чи хост посувається: `_BOOT_STUCK_SEC` (180 с), `_SSH_AFTER_RUNNING_SEC`
#: (180 с) і `_SSH_AUTH_GRACE` (90 с). Реальні бокси піднімаються по 204, 511 і
#: більше секунд — вони ще тягнуть образ або ставлять пакети, а SSH-демон уже
#: відповідає, поки наш ключ туди ще не прокинуто. Тобто система звинувачувала
#: СПРАВНІ машини, і робила це системно: 6 банів і 12 підозр за дві доби.
#:
#: Доказ, що це наша вада, а не ринку: машини 144417 і 97081 мали УСПІШНІ
#: прогони напередодні («391 з 391 сторінок, 0 OOM»), а назавтра дістали
#: «хост відхилив наш ключ».
#:
#: Замір, зроблений міряльником із доведеною похибкою, не є доказом. Тому
#: спостереження цих видів, записані до полагодження, ігноруються — так само,
#: як відкидають калібрувальну серію зі збитого приладу.
TIMER_DEPENDENT: frozenset[str] = frozenset({
    "ssh_auth_denied", "ssh_unreachable", "slow_boot", "never_booted",
})
#: Мить, коли всі три таймери стали дивитись на `status_msg` (посування хоста).
BROKEN_TIMER_EPOCH = datetime(2026, 8, 12, 9, 0, tzinfo=UTC)

#: Нижче цього канал МЕРТВИЙ — єдиний `slow_net`, що банить. Ворота беруть межу
#: звідси (`supervise/gate.py`), реєстр складає за нею й старі вироки: живий
#: канал, забанений колишньою сталою межею 20 Мбіт/с, — не вада хоста.
DEAD_NET_MBPS = 1.0

#: 🔴 Мить, коли проба каналу перестала перетирати виміряну швидкість нулем
#: (коміт 9ad70ef, 11.09.2026). Доти curl на `--max-time` друкував справжню
#: швидкість, а хвіст `|| echo "net_bps=0"` писав поверх неї нуль — і ворота
#: банили за «0.0 Мбіт/с» канали на ~15 (31846, 138981) і 208 Мбіт/с (140184).
#: Ознака такого запису: HTTP 206 (дані йшли) і нульова швидкість.
BROKEN_PROBE_EPOCH = datetime(2026, 9, 11, 20, 1, tzinfo=UTC)

#: Спостереження, що не банять нікого: вони калібрують константи, а не машину.
#: 🔴 `offer_taken` і `market_busy` — про РИНОК, а не про машину: перший це
#: гонка двох покупців за ту саму пропозицію, другий — 429/5xx від Vast.
#: Записувати їх у провину хосту означало б банити справні машини за чужу
#: розторопність.
NEUTRAL: frozenset[str] = frozenset({
    "oom_pages", "budget_stop", "user_stop", "deadline_stop",
    "offer_taken", "market_busy",
    # 🔴 `disk_short` — це розбіжність НАШОЇ вимоги з машиною, а не поломка
    # хоста. Замір 2026-08-11: план просив 120 ГБ (реально треба ~40: кадри
    # 8.3 ГБ, розпаковані 9.4, моделі 105 МБ), машина 34347 дала 104 — і
    # поїхала в чорний список на 14 днів за наш власний прорахунок. Хост
    # здоровий; правильна реакція — просити менше або відсіювати за диском ще
    # на ПОШУКУ, а не банити.
    "disk_short",
    # 🔴 `our_bug` — вирок НАМ, а не хосту: битий план, протухле посилання на
    # дані, забутий параметр. Машина тут ні до чого, і банити її означало б
    # мовчки звужувати собі ринок за власну помилку. Ворота заліза видають
    # цей вирок явно, тож він мусить бути легальним у реєстрі, а не давитись
    # `ValueError` під загальним `except`.
    "our_bug",
    # 🔴 `overpriced` — рішення ПРО ЗАХІД, а не вирок машині. Ціна тисячі
    # сторінок залежить від матеріалу (кадр-розворот тримає 4 шарди замість 8 і
    # чесно коштує $0.34-0.60) і від стелі, яку задав дослідник. Та сама машина,
    # невигідна сьогодні, завтра на звичайних сторінках буде найкращою.
    # Заміряно 2026-08-19: машина 14563 поїхала в бан на 3 дні за $0.201 проти
    # стелі $0.200 — за пів відсотка, при тому що власна модель швидкості має
    # розкид ±20%. Оффер відкидаємо, машину — ні.
    "overpriced",
    # 🔴 `slow_for_data` — живий, але повільний для ЦЬОГО заходу канал: на
    # меншому обсягу кадрів та сама машина проходить. Бан — лише `slow_net`
    # (мертвий канал, `gate.DEAD_NET_MBPS`). 15.09.2026 дві машини з 6.1 і 2.1
    # Мбіт/с дістали бан на 14 днів за захід на 1.23 ГБ.
    "slow_for_data",
})

#: Скільки днів «зірка» лишається зіркою без нових успіхів.
STAR_TTL_DAYS = 30

_KNOWN_OUTCOMES = frozenset({OK}) | frozenset(TTL_DAYS) | NEUTRAL


# ---- модель ---------------------------------------------------------------


@dataclass(frozen=True)
class BoxObservation:
    """Одна оренда — один рядок у журналі.

    `claimed` і `measured` навмисно лежать поруч: саме їхня різниця й ловить
    брехню картки, і саме її треба буде побачити людині через місяць.
    """

    machine_id: int
    outcome: str
    ts: str = ""
    host_id: int | None = None
    offer_id: int | None = None
    geolocation: str | None = None
    gpu_name: str = ""
    num_gpus: int = 1
    run_id: str | None = None
    case: str | None = None
    claimed: dict[str, Any] = field(default_factory=dict)
    measured: dict[str, Any] = field(default_factory=dict)
    detail: str = ""
    cost_usd: float = 0.0
    billed_sec: int = 0

    def __post_init__(self) -> None:
        if self.outcome not in _KNOWN_OUTCOMES:
            raise ValueError(
                f"невідомий outcome {self.outcome!r}; дозволені: {sorted(_KNOWN_OUTCOMES)}"
            )
        if not self.ts:
            object.__setattr__(self, "ts", datetime.now(tz=UTC).isoformat())

    @property
    def verdict(self) -> str:
        if self.outcome == OK:
            return "good"
        return "neutral" if self.outcome in NEUTRAL else "bad"


@dataclass(frozen=True)
class BoxVerdict:
    """Згорнута думка про машину на певний момент часу."""

    machine_id: int
    state: str  # "banned" | "starred" | "warned" | "unknown"
    reason: str
    expires: datetime | None = None
    hits: dict[str, int] = field(default_factory=dict)
    best_measured: dict[str, Any] = field(default_factory=dict)
    runs_ok: int = 0
    last_seen: datetime | None = None
    gpu_name: str = ""
    geolocation: str | None = None
    dph_total: float | None = None

    @property
    def banned(self) -> bool:
        return self.state == "banned"

    @property
    def score_factor(self) -> float:
        """Множник до скору оффера. Бан сюди не потрапляє — він відкидається."""
        return {"starred": 1.25, "unknown": 1.0, "warned": 0.6, "banned": 0.0}[self.state]


# ---- файли ----------------------------------------------------------------


def journal_path() -> Path:
    override = os.environ.get("GPURUNNER_BOXES_FILE")
    return Path(override) if override else registry_dir() / "boxes.jsonl"


def overrides_path() -> Path:
    override = os.environ.get("GPURUNNER_BOXES_OVERRIDES")
    return Path(override) if override else registry_dir() / "boxes.overrides.json"


def record(obs: BoxObservation) -> None:
    """Дописати спостереження. Один рядок, одним `write`, з fsync.

    Помилка запису НЕ валить прогін: реєстр — це пам'ять на майбутнє, а не
    умова роботи. Але вона друкується, бо мовчазна втрата пам'яті означала б,
    що ту саму погану машину візьмуть завтра знову.
    """
    path = journal_path()
    line = json.dumps(asdict(obs), ensure_ascii=False, sort_keys=True) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as e:
        print(f"[boxes] ⚠ не вдалось записати спостереження про {obs.machine_id}: {e}", flush=True)


#: Скільки спроб мусить стояти за вироком «флот обсипався», щоб він щось
#: означав. Те саме число, що `decide.Cfg.fail_min_attempts`, і з тієї самої
#: причини: на відновленій справі одна невдала сторінка давала 100% збоїв.
MIN_ATTEMPTS_FOR_LOAD_VERDICT = 10

_RE_FLEET_ATTEMPTS = re.compile(r"збоїв (\d+)")


def thin_evidence(obs: BoxObservation) -> str:
    """Чому цьому спостереженню не можна вірити. Порожньо — можна.

    🔴 Вирок «флот обсипався» виносився за часткою збоїв БЕЗ мінімального
    знаменника, тож дві невдалі сторінки з двох давали 100% і забирали з ринку
    справну машину на 21 день. Правило вже полагоджене (`fail_min_attempts`),
    але записи, зроблені ДО того, лишились у реєстрі й далі відсіюють хости.

    Судимо лише за тим, що записано в самому спостереженні: якщо в деталі
    стоїть число спроб і воно мале — доказу немає. Стан хоста (`gone`,
    `exited`, `offline`) — це інша річ, він про машину, і його не чіпаємо.
    """
    if obs.outcome != "died_under_load":
        return ""
    match = _RE_FLEET_ATTEMPTS.search(obs.detail or "")
    if not match:
        return ""
    attempts = int(match.group(1))
    if attempts >= MIN_ATTEMPTS_FOR_LOAD_VERDICT:
        return ""
    return (f"«флот обсипався» на {attempts} збоях — знаменник менший за "
            f"{MIN_ATTEMPTS_FOR_LOAD_VERDICT}, доказу немає")


def prune_thin(*, apply: bool = False) -> list[tuple[BoxObservation, str]]:
    """Прибрати з журналу вироки без знаменника. Повертає, що саме зникло.

    Без `apply` лише показує: реєстр оплачений грішми, і чистити його наосліп
    дорожче, ніж лишити зайвий бан.
    """
    kept: list[BoxObservation] = []
    dropped: list[tuple[BoxObservation, str]] = []
    for obs in read_all():
        why = thin_evidence(obs)
        if why:
            dropped.append((obs, why))
        else:
            kept.append(obs)
    if apply and dropped:
        path = journal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for obs in kept:
                fh.write(json.dumps(asdict(obs), ensure_ascii=False) + "\n")
    return dropped


def read_all() -> list[BoxObservation]:
    """Усі спостереження з журналу. Битий рядок пропускається зі скаргою."""
    path = journal_path()
    if not path.is_file():
        return []
    out: list[BoxObservation] = []
    bad = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
            out.append(BoxObservation(**data))
        except (ValueError, TypeError):
            bad += 1
    if bad:
        print(f"[boxes] ⚠ {bad} нечитаних рядків у {path}", flush=True)
    return out


@contextmanager
def mutate_overrides(*, note: str = ""):
    """Єдина точка ЗМІНИ ручних вердиктів — під замком і атомарно.

    🔴 Було два незалежні read-modify-write цілого файла (`boxes ban/star` і
    `boxes forget`). Два одночасні `ban` губили один одного мовчки: перший
    прочитав, другий прочитав, перший записав, другий затер. Це найгірший вид
    втрати — стирається саме РУЧНЕ рішення, тобто те єдине, чого автоматика не
    може відтворити.
    """
    from gpurunner.core import locks
    from gpurunner.core.manifest import current_owner

    owner = current_owner() or f"cli-{os.getpid()}"
    with locks.hold("boxes:overrides", owner=owner, ttl_sec=60, note=note):
        data = load_overrides()
        yield data
        path = overrides_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)


def load_overrides() -> dict[str, dict[str, str]]:
    """Ручні рішення: `{"never": {"4711": "чому"}, "star": {"38902": "чому"},
    "absolve": {"4711@<ts>": "чому"}}`.

    `absolve` — скасовані окремі спостереження (машина@мітка часу): вирок, який
    поставила вада НАШОГО коду, а не машина. Виміряна історія лишається в
    журналі, але ударом не рахується.
    """
    path = overrides_path()
    empty: dict[str, dict[str, str]] = {"never": {}, "star": {}, "absolve": {}}
    if not path.is_file():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"[boxes] ⚠ {path} не читається ({e}) — ручні рішення проігноровано", flush=True)
        return empty
    return {
        key: {str(k): str(v) for k, v in (data.get(key) or {}).items()}
        for key in empty
    }


def absolve_key(machine_id: int, ts: str) -> str:
    """Ключ скасованого спостереження."""
    return f"{int(machine_id)}@{ts}"


# ---- фолд ------------------------------------------------------------------


def _ts(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def verdicts(*, now: datetime | None = None) -> dict[int, BoxVerdict]:
    """Згорнути журнал у думку про кожну машину."""
    now = now or datetime.now(tz=UTC)
    overrides = load_overrides()

    by_machine: dict[int, list[tuple[datetime, BoxObservation]]] = {}
    for obs in read_all():
        ts = _ts(obs.ts)
        if ts is None:
            continue
        by_machine.setdefault(int(obs.machine_id), []).append((ts, obs))

    # машина може бути відома лише з ручного файла — її теж треба показати
    for raw_mid in list(overrides["never"]) + list(overrides["star"]):
        try:
            by_machine.setdefault(int(raw_mid), [])
        except ValueError:
            continue

    out: dict[int, BoxVerdict] = {}
    for mid, rows in by_machine.items():
        rows.sort(key=lambda r: r[0])
        out[mid] = _fold_one(mid, rows, now=now, overrides=overrides)
    return out


def _fold_one(
    machine_id: int,
    rows: list[tuple[datetime, BoxObservation]],
    *,
    now: datetime,
    overrides: dict[str, dict[str, str]],
) -> BoxVerdict:
    last_ok_at: datetime | None = None
    last_ok: BoxObservation | None = None
    runs_ok = 0
    for ts, obs in rows:
        if obs.outcome == OK:
            last_ok_at, last_ok, runs_ok = ts, obs, runs_ok + 1

    hits: dict[str, int] = {}
    worst_expiry: datetime | None = None
    banned_kind: str | None = None
    banned_detail = ""
    absolved = overrides.get("absolve") or {}
    for ts, obs in rows:
        kind = obs.outcome
        if kind == OK or kind in NEUTRAL:
            continue
        if absolve_key(machine_id, obs.ts) in absolved:
            continue  # вирок поставила вада нашого коду, не машина
        ttl = TTL_DAYS.get(kind)
        if ttl is None:
            continue
        if kind in TIMER_DEPENDENT and ts < BROKEN_TIMER_EPOCH:
            continue  # вирок зламаного міряльника — не доказ (див. вище)
        if kind == "slow_net":
            net = _measured_net_mbps(obs)
            http = str((obs.measured or {}).get("net_http") or "")
            if ts < BROKEN_PROBE_EPOCH and http not in ("", "000") and net <= 0:
                continue  # нуль зламаної проби: дані йшли, швидкість перетерто
            if net >= DEAD_NET_MBPS:
                continue  # живий канал — не вада хоста (див. DEAD_NET_MBPS)
        expiry = ts + timedelta(days=ttl)
        if expiry <= now:
            continue  # протухло — машину могли полагодити
        # Пізніший успіх знімає вирок; а для липких — будь-який успіх узагалі
        # (він спростовує саме те твердження, на якому липкість стоїть).
        sticky = kind in STICKY and last_ok_at is None
        if not sticky and last_ok_at is not None and ts < last_ok_at:
            continue  # пізніший успіх довів, що це вже не так
        if kind in STICKY and last_ok_at is not None and ts > last_ok_at:
            # Ключ уже приймався на цій машині — отже це збій, а не вирок.
            # Судимо як транзієнт: два удари й коротке TTL, як `ssh_unreachable`.
            kind = "ssh_unreachable"
            expiry = ts + timedelta(days=TTL_DAYS["ssh_unreachable"])
            if expiry <= now:
                continue
        if (kind in SOFT_AFTER_OK and last_ok_at is not None and ts > last_ok_at
                and _measured_net_mbps(obs) > 0):
            kind = SOFT_AFTER_OK[kind]
            expiry = ts + timedelta(days=TTL_DAYS[kind])
            if expiry <= now:
                continue
        hits[kind] = hits.get(kind, 0) + 1
        enough = hits[kind] >= HITS_TO_BAN.get(kind, 1)
        if enough and (worst_expiry is None or expiry > worst_expiry):
            worst_expiry, banned_kind, banned_detail = expiry, kind, obs.detail

    last = rows[-1][1] if rows else None
    shown = last_ok or last
    common: dict[str, Any] = {
        "machine_id": machine_id,
        "hits": hits,
        "runs_ok": runs_ok,
        "last_seen": rows[-1][0] if rows else None,
        "gpu_name": shown.gpu_name if shown else "",
        "geolocation": shown.geolocation if shown else None,
        "dph_total": _claimed_dph(shown),
        "best_measured": dict(last_ok.measured) if last_ok else {},
    }

    manual_never = overrides["never"].get(str(machine_id))
    if manual_never:
        return BoxVerdict(state="banned", reason=f"вручну: {manual_never}", expires=None, **common)

    if banned_kind:
        days_left = max(0, (worst_expiry - now).days) if worst_expiry else 0
        why = f"{banned_kind}×{hits[banned_kind]}"
        if banned_detail:
            why += f" ({banned_detail})"
        return BoxVerdict(
            state="banned", reason=f"{why}; бан ще {days_left} дн.", expires=worst_expiry, **common
        )

    # 🔴 Ручна зірка НЕ перекриває свіжий факт. Інцидент 2026-08-11: машину
    # 57139 зірковано вручну (як хибно забанену), і `find_candidates` брав її
    # ПЕРШОЮ, адресно — а хост щоразу віддавав `failed to inject CDI devices
    # … gpu=2` і `Template not found`. Три оренди поспіль в одну й ту саму
    # стіну: зірка тримала нас у циклі, замість пустити до наступного
    # кандидата. Тому будь-яке невитрачене влучання опускає машину до
    # «підозри» ще до того, як набереться на бан.
    if hits:
        worst = max(hits, key=lambda k: hits[k])
        return BoxVerdict(
            state="warned",
            reason=f"{worst}×{hits[worst]}, до бану {HITS_TO_BAN.get(worst, 1)}",
            **common,
        )

    manual_star = overrides["star"].get(str(machine_id))
    if manual_star:
        return BoxVerdict(state="starred", reason=f"вручну: {manual_star}", expires=None, **common)

    if last_ok_at is not None and now - last_ok_at <= timedelta(days=STAR_TTL_DAYS):
        pph = (last_ok.measured or {}).get("pages_per_hour") if last_ok else None
        tail = f", {float(pph):.0f} стор/год" if pph else ""
        return BoxVerdict(
            state="starred",
            reason=f"{runs_ok} успішних, останній {last_ok_at:%Y-%m-%d}{tail}",
            expires=last_ok_at + timedelta(days=STAR_TTL_DAYS),
            **common,
        )

    return BoxVerdict(state="unknown", reason="без свіжих спостережень", **common)


def _measured_net_mbps(obs: BoxObservation) -> float:
    try:
        return float((obs.measured or {}).get("net_mbps") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _claimed_dph(obs: BoxObservation | None) -> float | None:
    if obs is None:
        return None
    value = (obs.claimed or {}).get("dph_total")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---- запити ----------------------------------------------------------------


def banned_ids(*, now: datetime | None = None) -> set[int]:
    """Машини, які не пропонувати. Йде у фільтр `machine_id.notin`."""
    return {mid for mid, v in verdicts(now=now).items() if v.banned}


def starred(
    *, gpu: str | None = None, limit: int = 20, now: datetime | None = None
) -> list[BoxVerdict]:
    """Добрі машини, найцінніші першими — щоб шукати їх адресно.

    Порядок — сторінок за долар із ОСТАННЬОГО успішного прогону: це єдине
    число, що поєднує швидкість і ціну, і воно виміряне, а не обіцяне.
    """
    rows = [v for v in verdicts(now=now).values() if v.state == "starred"]
    if gpu:
        needle = gpu.lower()
        rows = [v for v in rows if needle in (v.gpu_name or "").lower()]

    def key(v: BoxVerdict) -> float:
        """Сторінок за долар, зі знижкою за повільний підйом.

        🔴 Час підйому — єдиний сигнал про закешованість нашого образу на
        хості: у картці оффера про образи немає жодного поля (перевірено на
        100 полях). Машина, що минулого разу піднялась за 40 с, майже напевно
        тримає образ у кеші; та, що тягла його три хвилини, тягтиме знову — і
        це оплачений `docker pull`, а не робота.
        """
        measured = v.best_measured or {}
        pph = float(measured.get("pages_per_hour") or 0)
        dph = v.dph_total or 0.0
        base = pph / dph if pph and dph else pph
        boot = float(measured.get("boot_sec") or 0)
        if boot > 120:
            base *= 0.7
        elif boot and boot < 60:
            base *= 1.15
        return base

    rows.sort(key=key, reverse=True)
    return rows[: max(0, int(limit))]


def explain(machine_id: int, *, now: datetime | None = None) -> str:
    """Людський переказ історії машини — для CLI і для звіту наглядача."""
    now = now or datetime.now(tz=UTC)
    verdict = verdicts(now=now).get(int(machine_id))
    rows = [o for o in read_all() if int(o.machine_id) == int(machine_id)]
    if verdict is None and not rows:
        return f"машина {machine_id}: жодного спостереження"

    lines = []
    if verdict is not None:
        head = f"машина {machine_id} · {verdict.state.upper()} · {verdict.reason}"
        if verdict.gpu_name:
            head += f" · {verdict.gpu_name}"
        if verdict.geolocation:
            head += f" · {verdict.geolocation}"
        lines.append(head)
    for obs in rows[-12:]:
        mark = {"good": "✓", "bad": "✗", "neutral": "·"}[obs.verdict]
        stamp = (_ts(obs.ts) or now).strftime("%Y-%m-%d %H:%M")
        tail = f" — {obs.detail}" if obs.detail else ""
        cost = f" · ${obs.cost_usd:.2f}" if obs.cost_usd else ""
        lines.append(f"  {mark} {stamp} {obs.outcome}{cost}{tail}")
    return "\n".join(lines)


def claimed_from_offer(offer: dict[str, Any]) -> dict[str, Any]:
    """Витягти з картки оффера рівно те, що потім звірятиметься з виміром."""
    def num(key: str) -> float | None:
        value = offer.get(key)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    gpu_ram_mb = num("gpu_ram") or 0.0
    return {
        "cores": num("cpu_cores_effective"),
        "ram_gb": (num("cpu_ram") or 0.0) / 1024.0 or None,
        "vram_gb": gpu_ram_mb / 1024.0 if gpu_ram_mb else None,
        "disk_gb": num("disk_space"),
        "inet_down": num("inet_down"),
        "inet_up": num("inet_up"),
        "reliability2": num("reliability2") or num("reliability"),
        "dph_total": num("dph_total"),
    }


def observation_from_offer(
    offer: dict[str, Any],
    *,
    outcome: str,
    measured: dict[str, Any] | None = None,
    detail: str = "",
    run_id: str | None = None,
    case: str | None = None,
    cost_usd: float = 0.0,
    billed_sec: int = 0,
) -> BoxObservation:
    """Зібрати спостереження з картки оффера — щоб виклик був в один рядок."""
    return BoxObservation(
        machine_id=int(offer.get("machine_id") or 0),
        host_id=int(offer["host_id"]) if offer.get("host_id") is not None else None,
        offer_id=int(offer["id"]) if offer.get("id") is not None else None,
        geolocation=offer.get("geolocation"),
        gpu_name=str(offer.get("gpu_name") or ""),
        num_gpus=int(offer.get("num_gpus") or 1),
        outcome=outcome,
        claimed=claimed_from_offer(offer),
        measured=dict(measured or {}),
        detail=detail,
        run_id=run_id,
        case=case,
        cost_usd=float(cost_usd),
        billed_sec=int(billed_sec),
    )
