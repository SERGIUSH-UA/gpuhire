"""Контракт наглядача з агентом: один JSON замість читання логів.

Головна вимога до цього файла — щоб агент **ніколи не мусив дивитись у лог**,
щоб ухвалити рішення. Звідси три властивості:

- `human_action_required` — єдине булеве, на якому агент розгалужується. Воно
  `true` рівно у трьох випадках (бюджет вичерпано, ринок порожній, справа
  лишилась неповною), і в усіх інших наглядач розбирається сам;
- `why` — один рядок людською мовою, який можна процитувати користувачеві без
  інтерпретації;
- `incidents[]` — кожна аномалія з ціною в доларах. Не «див. лог», а «інстанс
  знищено за 41 с, машину заблоковано на 30 днів, $0.003».

Термінальність дублюється кодом виходу процесу: агент може взагалі не читати
JSON, якщо йому досить коду.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpurunner.config import data_dir

#: Коди виходу `gpurunner htr supervise`.
EXIT_OK = 0
EXIT_FAILED = 3
EXIT_INCOMPLETE = 4
EXIT_BUDGET = 5
EXIT_MARKET_EMPTY = 6
EXIT_DEADLINE = 7
EXIT_NO_CREDIT = 8

EXIT_BY_VERDICT = {
    "ok": EXIT_OK,
    "failed": EXIT_FAILED,
    "incomplete": EXIT_INCOMPLETE,
    "budget_stop": EXIT_BUDGET,
    "market_empty": EXIT_MARKET_EMPTY,
    "deadline": EXIT_DEADLINE,
    # 🔴 Окремий код, а не `budget_stop`: наш бюджет цілий, скінчились гроші
    # на боці провайдера. Лікується поповненням, а не іншими ручками, і саме
    # тому мусить читатись інакше (04.09.2026: 18 запусків у нікуди, бо
    # порожній баланс приходив як «ринок зайнятий»).
    "no_credit": EXIT_NO_CREDIT,
}

#: Вердикти, за яких рішення справді за людиною — і тільки вони.
HUMAN_VERDICTS = frozenset({"budget_stop", "market_empty", "incomplete", "no_credit"})

_SAVE_LOCK = threading.Lock()


def state_dir() -> Path:
    return data_dir() / "htr"


def state_path(session: str) -> Path:
    return state_dir() / f"{session}.json"


def latest_path(owner: str | None = None) -> Path:
    """«Останній захід» — ПЕР-ВЛАСНИКОМ, а не один на машину.

    🔴 Було `latest.json` спільним, і обидві паралельні сесії писали в нього
    щотіку. `gpurunner htr state --json` без `--session` читає саме його —
    тобто агент бачив ЧУЖИЙ захід і міг ухвалити рішення про не свій прогін.
    Та сама помилка, що з міткою інстансу: спільне ім'я створює ілюзію
    власності.
    """
    from gpurunner.core.manifest import current_owner

    who = (owner if owner is not None else current_owner()).strip()
    safe = "".join(ch for ch in who if ch.isalnum() or ch in "-_.")[:48]
    return state_dir() / (f"latest-{safe}.json" if safe else "latest.json")


@dataclass
class Incident:
    """Аномалія з ціною. Саме це агент переказує користувачеві."""

    ts: str
    kind: str
    detail: str
    action: str = ""
    machine_id: int | None = None
    cost_usd: float = 0.0
    #: 🔴 Знімок флоту НА МОМЕНТ БІДИ. `state.shards` живе лише поки бокс
    #: живий: на фініші останній прогрес приходить без шардів і затирає його
    #: порожнім списком. Через це причину колапсу машини 39565 (2026-08-19)
    #: після смерті боксу встановити було вже нічим — лишились самі числа в
    #: пам'яті агента. Тут вони переживають і бокс, і захід.
    fleet: dict[str, Any] | None = None


@dataclass
class CaseState:
    case: str
    status: str = "pending"      # pending | running | fetching | done | incomplete | failed
    n_pages_expected: int = 0
    pages_done: int = 0
    pages_failed: int = 0
    pages_per_hour: float = 0.0
    eta_sec: int | None = None
    complete: bool | None = None
    missing_count: int = 0
    missing: list[str] = field(default_factory=list)
    out_dir: str | None = None
    detail: str = ""
    #: Шифра справи (`архів/фонд/справа`). Доти стан її не знав, і зшивка
    #: заходу з метою йшла лише через `out_dir` (06.09.2026).
    case_key: str = ""
    #: 🔴 Скільки сторінок прочитано САМЕ В ЦЬОМУ прогоні й за скільки секунд
    #: стінного часу (21.09.2026). Без цих двох чисел `pages_done` і
    #: `pages_per_hour` — про різні речі: перше включає підняте з чекпоінтів,
    #: друге рахується лише від зробленого зараз. Калібровка через це не могла
    #: відсіяти відновлені й догінні заходи В ПРИНЦИПІ й мусила відкидати все
    #: дрібне гуртом (`calibrate.MIN_PAGES_FOR_RATE`). Обидва числа приходять у
    #: прогресі раннера (`pages_resumed`, `wall_sec`) — тут вони просто
    #: перестають губитись.
    pages_this_run: int = 0
    wall_sec: float = 0.0


@dataclass
class SupervisorState:
    """Повний стан заходу. Пишеться атомарно, читається ким завгодно."""

    session: str
    schema: int = 1
    updated: str = ""
    phase: str = "starting"
    verdict: str | None = None
    human_action_required: bool = False
    human_action: str | None = None
    why: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    box: dict[str, Any] | None = None
    cases: list[CaseState] = field(default_factory=list)
    shards: list[dict[str, Any]] = field(default_factory=list)
    incidents: list[Incident] = field(default_factory=list)
    next_poll_sec: int = 30
    #: 🔴 Pid наглядача, що пише цей стан. Без нього читач НЕ МОЖЕ відрізнити
    #: «захід іде» від «наглядача вбито, а бокс лишився горіти»: у стані стоїть
    #: `phase: renting, verdict: null` і в тому, і в тому випадку. Заміряно
    #: 2026-08-11: перерваний захід лишив по собі саме таку картину, і жоден
    #: рядок JSON про це не казав.
    pid: int = 0
    #: Вердикт уже неперезаписний (див. `finish(sticky=True)`). У JSON не йде.
    _sticky: bool = False
    report_path: str | None = None
    #: Лог НАГЛЯДАЧА — завжди він, від старту до кінця.
    log_path: str | None = None
    #: Лог раннера з боксу, врятований після забору. 🔴 Окреме поле: доти він
    #: підміняв `log_path`, і посилання на лог наглядача вело в інший файл.
    runner_log_path: str | None = None
    #: Серцебиття наглядача: пишеться окремим потоком і в тихих фазах (підйом
    #: боксу, очікування SSH), де основний цикл стану не зберігає.
    heartbeat: str = ""

    # ---- зміна ----

    def note(
        self,
        kind: str,
        detail: str,
        *,
        action: str = "",
        machine_id: int | None = None,
        cost_usd: float = 0.0,
        fleet: dict[str, Any] | None = None,
    ) -> None:
        """Записати інцидент. Друкується одразу — щоб хвіст логу теж був чесним."""
        self.incidents.append(
            Incident(
                ts=datetime.now(tz=UTC).isoformat(timespec="seconds"),
                kind=kind, detail=detail, action=action,
                machine_id=machine_id, cost_usd=round(float(cost_usd), 4),
                fleet=fleet,
            )
        )
        price = f" (${cost_usd:.3f})" if cost_usd else ""
        line = f"[supervise] ⚠ {kind}: {detail}{price}" + (f" → {action}" if action else "")
        print(line, flush=True)
        # 🔴 І в ФАЙЛ. Друк у stdout зникає разом із процесом: агент, що пустив
        # наглядача фоном через пайп, при перериванні втрачав увесь вивід — і
        # причину знищених інстансів не було де прочитати.
        sink = getattr(self, "sink", None)
        if sink is not None:
            with contextlib.suppress(OSError, ValueError):
                print(line, file=sink, flush=True)

    def finish(self, verdict: str, why: str, *, human_action: str | None = None,
               sticky: bool = False) -> None:
        """Термінальний вердикт. `human_action_required` виводиться з нього, а
        не виставляється руками — інакше він розійшовся б із кодом виходу.

        🔴 `sticky` захищає вердикт від перезапису. Живий бокс, який не вдалось
        погасити, — саме такий випадок: `_destroy` чесно ставив
        `human_action: «негайно gpurunner cancel …»`, а гілка, що його
        викликала, одразу робила свій `finish("failed")` і збивала
        `human_action_required` у **false**. Агент розгалужується рівно на
        цьому булевому — тобто він казав «прогін провалився» і не робив
        нічого, поки оренда горіла до дедлайн-кілера (~8.5 год).
        """
        if getattr(self, "_sticky", False):
            return
        self.verdict = verdict
        self.phase = "finished"
        self.why = why
        self.human_action_required = verdict in HUMAN_VERDICTS
        self.human_action = human_action if self.human_action_required else None
        if sticky:
            self._sticky = True

    @property
    def exit_code(self) -> int:
        return EXIT_BY_VERDICT.get(self.verdict or "ok", EXIT_FAILED)

    # ---- запис ----

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_sticky", None)
        data["pid"] = data.get("pid") or os.getpid()
        data["updated"] = datetime.now(tz=UTC).isoformat(timespec="seconds")
        data["exit_code"] = self.exit_code if self.verdict else None
        return data

    def save(self) -> Path:
        """Атомарно опублікувати стан у файл сесії і в `latest.json`.

        `latest.json` існує рівно для того, щоб агент міг прочитати стан, не
        знаючи імені сесії: «подивись, що там» має бути одним викликом.
        """
        # Пишуть двоє — основний цикл і серцебиття — у той самий `.json.tmp`;
        # без замка один підміняв би напівзаписаний файл другого.
        with _SAVE_LOCK:
            payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=1)
            state_dir().mkdir(parents=True, exist_ok=True)
            for path in (state_path(self.session), latest_path()):  # свій latest
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(payload, encoding="utf-8")
                os.replace(tmp, path)
        return state_path(self.session)


def diagnose(data: dict[str, Any]) -> dict[str, Any]:
    """Дописати в стан правду про ЖИВІСТЬ наглядача.

    Контракт «агент читає один JSON і все знає» тримається лише доти, доки
    JSON уміє сказати «мене писав процес, якого вже немає». Інакше обірваний
    захід виглядає точнісінько як робочий — і агент спокійно доповідає
    людині, що все йде, поки оренда горить.
    """
    if data.get("verdict"):
        return data          # захід дійшов до кінця сам — питання не стоїть
    pid = int(data.get("pid") or 0)
    alive = _pid_alive(pid) if pid else True
    age_sec = _age_sec(str(data.get("updated") or ""))
    stale = age_sec is not None and age_sec > max(180, 6 * int(data.get("next_poll_sec") or 30))
    if alive and not stale:
        # Живість кажемо ЗАВЖДИ, а не лише коли щось не так: поле, яке
        # з'являється тільки в біді, агент не може відрізнити від «не питали».
        data = dict(data)
        data["supervisor_alive"] = True
        data["quiet_min"] = round((age_sec or 0) / 60, 1)
        return data

    if alive:
        # 🔴🔴 ЖИВИЙ ПРОЦЕС — НЕ СИРОТА, і плутати ці два стани дорого.
        #
        # Вердикт рахувався з ВІКУ ЗАПИСУ стану, а наглядач має законні тихі
        # фази, у яких писати нема чого: очікування замка ринку, `PUT /asks/`,
        # хост тягне образ, чекаємо SSH (вікно 360 с). У кожній із них стан
        # старів — і читач бачив «ЗАХІД ОБІРВАНО … Оренда могла лишитись
        # живою» на заході, який у ту саму секунду качав образ на щойно
        # оплаченому боксі. 2026-08-19: п'ять хибних тривог за один захід, і
        # кожна підказувала дію «cancel --all-running», тобто вбити СВОЮ ж
        # роботу й орендувати вдруге.
        #
        # Тому тут: сказати правду («мовчить N хв»), НЕ ставити термінальний
        # вердикт і НЕ вимагати втручання. Сирота — це відсутній процес.
        data = dict(data)
        data["supervisor_alive"] = True
        data["quiet_min"] = round((age_sec or 0) / 60, 1)
        data["why"] = (
            f"наглядач ЖИВИЙ (pid {pid}), але стан не оновлювався "
            f"{(age_sec or 0) / 60:.0f} хв — фаза «{data.get('phase')}». Це "
            f"нормально для тихих фаз (замок ринку, качання образу, очікування "
            f"SSH). Приймач: процес в ОС + `/api/v1/instances/`, а не вік запису."
        )
        return data

    data = dict(data)
    data["supervisor_alive"] = False
    data["human_action_required"] = True
    data["verdict"] = data.get("verdict") or "orphaned"
    data["why"] = (
        f"ЗАХІД ОБІРВАНО: процес наглядача (pid {pid}) не існує, а термінального "
        f"вердикту немає (фаза «{data.get('phase')}»). Оренда могла лишитись живою."
    )
    data["human_action"] = (
        "перевірити інстанси: `gpurunner reconcile --any-owner`, і якщо серед "
        "них є СВІЙ живий — `gpurunner cancel --all-running --backend vast` під "
        "своїм GPURUNNER_OWNER. Чужих не чіпати."
    )
    # 🔴 Разом із «обірвано» одразу віддаємо РОЗТИН. Доти агент бачив факт
    # смерті й не мав чим відповісти на «чому»: два заходи 22.09.2026 зникли
    # без сліду, і причину довелось шукати по журналах Windows, де її немає.
    with contextlib.suppress(Exception):
        from gpurunner.supervise.blackbox import postmortem

        data["postmortem"] = postmortem(str(data.get("session") or ""))
    data["exit_code"] = EXIT_FAILED
    return data


def _pid_alive(pid: int) -> bool:
    """Чи живий процес. Помилятись — ТІЛЬКИ в бік «живий».

    🔴🔴 На Windows `os.kill(pid, 0)` НЕ є перевіркою: CPython там викликає
    `TerminateProcess`, тобто в кращому разі дає хибний вердикт (OpenProcess
    не пускає — читається як «мертвий»), а в гіршому вбиває чужий процес
    сигналом, який на Unix нешкідливий. Виміряно 2026-08-12: живий наглядач
    (pid 20356, 36.7 с процесорного часу) був оголошений неіснуючим, і агент
    отримав «ЗАХІД ОБІРВАНО» посеред здорової роботи.

    Тому: є `psutil` — питаємо його; немає — на Windows не питаємо ВЗАГАЛІ й
    вважаємо живим, а мертвих ловимо за віком запису (`updated`), який
    брехати не вміє.
    """
    if pid <= 0:
        return True          # немає pid — немає підстав звинувачувати
    if pid == os.getpid():
        return True
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except ImportError:
        pass
    if sys.platform == "win32":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True          # немає прав — не наша справа, вважаємо живим
    return True


def _age_sec(stamp: str) -> float | None:
    if not stamp:
        return None
    try:
        ts = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(tz=UTC) - ts).total_seconds())


def load(session: str | None = None) -> dict[str, Any] | None:
    """Прочитати стан. Без імені сесії — останній.

    Повертає сирий словник, а не дата-клас: читач (агент, CLI, чужий скрипт)
    не має залежати від наших типів.
    """
    path = state_path(session) if session else latest_path()
    if not path.is_file() and not session:
        # 🔴 `latest` став пер-власником заради ізоляції сесій — і цим зламав
        # читача, у якого `GPURUNNER_OWNER` не заданий (інша оболонка, інший
        # агент, людина руками). Такий читач діставав старий спільний
        # `latest.json` від давнього заходу: перевірено на живій машині —
        # `htr state` віддавав чужий вердикт `failed` із кодом виходу 0, тобто
        # контракт «агент читає один JSON» ламався мовчки й у найгірший бік.
        # 🔴 Резерв бере найсвіжіший ЛИШЕ якщо він один. Інакше «найсвіжіший»
        # означає «той, чий наглядач писав останнім» — тобто при двох живих
        # заходах з імовірністю ~50% ЧУЖИЙ, і агент переказав би людині
        # результат не свого прогону («повно, оренду погашено», поки власний
        # бокс горить). Краще чесна відмова з переліком сесій, ніж упевнена
        # неправда: змінні оточення не переживають між викликами оболонки, тож
        # безіменний читач — не екзотика, а типовий випадок.
        candidates = sorted(state_dir().glob("latest-*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if len(candidates) == 1:
            path = candidates[0]
        elif candidates:
            return {
                "ambiguous": True,
                "sessions": [_peek_session(c) for c in candidates],
                "why": (
                    f"на машині {len(candidates)} заходів і не задано, чий читати. "
                    f"Постав GPURUNNER_OWNER або поклич із --session — інакше "
                    f"можна ухвалити рішення про ЧУЖИЙ прогін."
                ),
            }
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return diagnose(data) if isinstance(data, dict) else None


def _peek_session(path: Path) -> dict[str, Any]:
    """Ім'я й вердикт заходу — щоб людині було з чого вибрати."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"file": path.name}
    return {
        "file": path.name,
        "session": data.get("session"),
        "phase": data.get("phase"),
        "verdict": data.get("verdict"),
        "updated": data.get("updated"),
    }


def new_session(prefix: str = "sup") -> str:
    """Унікальне ім'я заходу.

    🔴 Роздільності до секунди НЕ досить: два наглядачі, запущені в ту саму
    секунду (а їх запускають скриптом), дістають однакове ім'я — а з нього
    виводиться власник, тобто вся ізоляція між сесіями тихо вимикається.
    Додаємо pid і чотири випадкові символи.
    """
    import os
    import secrets

    stamp = f"{datetime.now(tz=UTC):%Y-%m-%d-%H%M%S}"
    return f"{prefix}-{stamp}-{os.getpid()}-{secrets.token_hex(2)}"
