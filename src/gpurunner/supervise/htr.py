"""Наглядач хмарного HTR: орендує, стежить, доганяє, звіряє, гасить.

Це і є той контур керування, у якому раніше сидів агент. Він живе весь захід
і має один вихід назовні — `state.json`.

Машина станів однієї справи, і порядок у ній жорсткий:

    вибрати бокс → орендувати → ПРОБА ЗАЛІЗА → залити → рахувати
        → done → fetch → VERIFY → (неповно? догін → fetch → verify) → гасити

Дві точки в цьому ланцюгу непорушні:

- **проба перед заливкою.** Усе до неї коштує секунди, усе після — хвилини й
  гігабайти. Машина, що збрехала про залізо, гаситься раніше, ніж отримає
  перший байт даних, і потрапляє в реєстр.
- **verify перед гасінням.** Бокс не знищується, доки не підтверджено, що
  результат цілий. Саме тому, що конвеєр колись забрав 203 сторінки з 323 і
  погасив машину, на якій лежали решта 120.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import posixpath
import re
import secrets
import shlex
import shutil
import subprocess
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, ClassVar, cast

from gpurunner.backends.vast import VastBackend
from gpurunner.config import registry_dir
from gpurunner.core import boxes, locks, manifest
from gpurunner.core.backend import BackendError
from gpurunner.core.htr_sizing import GB_PER_SHARD as GB_PER_SHARD_DEFAULT
from gpurunner.core.htr_sizing import (
    LINES_PER_PAGE_ASSUMED,
    gb_per_shard_for,
    gb_per_shard_start_for,
)
from gpurunner.core.models import JobHandle, JobStatus
from gpurunner.core.offer_score import (
    Need,
    ScoredOffer,
    machine_id_of,
    offer_down_cost,
    offer_dph,
)
from gpurunner.jobs.htr_case import HTRCaseJob
from gpurunner.supervise import append as append_mod
from gpurunner.supervise import gate as gate_mod
from gpurunner.supervise import wrapup as wrapup_mod
from gpurunner.supervise.decide import (
    SOFT_DEAD_STATES,
    UNDERDELIVER_SHARE,
    UNDERDELIVER_WARMUP_PAGES,
    UNDERDELIVER_WARMUP_SEC,
    Cfg,
    Obs,
    decide,
)
from gpurunner.supervise.plan import CasePlan, Plan
from gpurunner.supervise.state import (
    EXIT_FAILED,
    CaseState,
    SupervisorState,
    new_session,
    state_dir,
)
from gpurunner.supervise.verify import verify_case

#: Як часто опитувати бокс.
TICK_SEC = 30
#: Як часто серцебиття оновлює стан, коли основний цикл мовчить (підйом боксу,
#: очікування SSH, забір). Частіше за тік — щоб «тихо» не читалось як «мертво».
HEARTBEAT_SEC = 30
#: Скільки чекати між невдалою орендою і наступною спробою.
#: Скільки тримається замок ринку. Мусить покривати весь сабміт: пошук,
#: `PUT /asks/`, підняття боксу й очікування SSH (до 900 с).
#: 🔴 Було 1800: TTL мусив покривати ВЕСЬ сабміт разом з очікуванням SSH. Відколи
#: замок віддається одразу після створення інстансу (`_note_billing_started`),
#: вікно тримання — це секунди `PUT /asks/`, і довгий TTL лишився чистою шкодою:
#: сесія, що вмерла з замком у руках, морозила ринок для решти на пів години.
_RENT_LOCK_TTL_SEC = 300
#: Скільки чекаємо на зайнятий ринок, перш ніж перейти до іншого кандидата.
_RENT_LOCK_WAIT_SEC = 600
RETRY_SLEEP_SEC = 20


#: Скільки терпимо фазу забору цілком. Замір: одна справа — 2379 дрібних файлів,
#: черга з восьми — ~13 тисяч; при 50-100 мс на обмін самі лише оберти дають
#: 11-22 хвилини, і це ще НОРМАЛЬНИЙ забір. Півгодини — вже не повільно, а
#: нікуди; те, що приїхало, лишається, решту добирає `htr fetch-ckpt`.
FETCH_PHASE_MAX_SEC = 1800.0

#: Межі стелі на ОДИН перенос, с. Не робочий час, а рамки аварії: нижня — щоб
#: дрібний перенос не різався на моргання мережі, верхня — щоб зависання не
#: коштувало годин оренди.
TRANSFER_CEILING_MIN_SEC = 120
TRANSFER_CEILING_MAX_SEC = 3600

#: Канал, який припускаємо, поки свого заміру ще немає, МБ/с. Песимістично: це
#: найгірша десята частка реєстру, а не типове значення.
_SLOW_CHANNEL_MBS = 0.5


#: Як часто везти чекпоінти зі складу на машині додому. П'ять хвилин — це
#: ціна втрати при раптовій смерті машини, а не технічна межа: сам забір
#: коштує хвилини каналу, тож частіше нема сенсу.
BOX_CKPT_SYNC_SEC = 300

#: Яку частку стелі часу заходу дозволено з'їсти ДОСТАВЦІ даних на машину.
#: Більше означає, що машину взяли не для роботи, а щоб на неї возити: за
#: чверть стелі 4-годинного заходу (одна година) встигає доїхати ~25 ГБ на
#: заміряних 7.5 МБ/с і лише 1.8 ГБ на поганому каналі — і саме другий випадок
#: треба ловити до того, як за нього заплачено.
DELIVERY_MAX_SHARE = 0.25

#: Доки ПЕРШИЙ чекпоінт не приїхав додому, питаємо машину щотіку. Порожня тека
#: чекпоінтів у перші хвилини — штатний стан (раннер ще сегментує), а з'їдений
#: на ній п'ятихвилинний слот означає, що найуразливіший відрізок заходу —
#: початок — лишається без жодної точки відновлення вдома. Сама перевірка
#: коштує один `ls` по вже піднятому з'єднанню.
BOX_CKPT_FIRST_SEC = 0.0


class GateRejected(BackendError):
    """Машина не пройшла ворота заліза. Несе `outcome` для реєстру."""

    def __init__(self, result: gate_mod.GateResult) -> None:
        super().__init__(result.detail)
        self.outcome = result.outcome
        self.result = result


#: Ручки, які дослідник задає через `-p` і які МУСЯТЬ дійти до підбору оффера,
#: воріт і наглядача. Ключ — ім'я в `params`, значення — поле `Plan`.
_PARAM_KNOBS = {
    "vram_gb_per_shard": "gb_per_shard",
    "max_usd_per_1000_pages": "max_usd_per_1000_pages",
    "max_cost_per_case": "max_cost_per_case",
}


def plan_knob(plan: Any, name: str) -> float | None:
    """Значення ручки З УСІХ місць, де її можна задати: `-p`, справа, поле плану.

    🔴🔴 Ручка, задана через `-p`, лягає в `params` — і поле `Plan` лишається
    порожнім. Наглядач читав ТІЛЬКИ поле, тож `-p max_usd_per_1000_pages=0.30`
    не діяв узагалі: стеля лишалась дефолтною $0.20. Доки ворота ціну не
    перевіряли, це було невидимо; з 2026-08-19 воно означає, що на матеріалі,
    який чесно коштує $0.34-0.60 за тисячу (кадр-РОЗВОРОТ, ~4 шарди), відсіявся
    б увесь ринок і захід не поїхав би взагалі.

    Та сама вада вже була у `vram_gb_per_shard` і лагодилась точково; тепер
    читалка спільна, щоб наступна ручка не повторила історію.
    """
    field = _PARAM_KNOBS.get(name, name)
    for src in (getattr(plan, "params", None), *(c.params for c in plan.cases)):
        value = (src or {}).get(name)
        if value:
            return float(value)
    value = getattr(plan, field, None)
    return float(value) if value else None


def plan_knob_str(plan: Any, name: str) -> str:
    """Рядкова ручка плану (`params` плану, потім справ); порожньо — не задано."""
    for src in (getattr(plan, "params", None), *(c.params for c in plan.cases)):
        value = (src or {}).get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _staging_root() -> Path:
    """Куди складати забране ДО розкладання по справах.

    🔴🔴 АБСОЛЮТНИЙ шлях, і це не косметика. Доти тут стояло `Path("out")`, тобто
    тека відносно поточного каталогу процесу. Поки наглядача запускали руками з
    каталогу репозиторію, воно працювало; відчеплена задача планувальника не має
    робочого каталогу взагалі й дістає `C:\\Windows\\System32`, де створити теку
    не можна. Спіймано бойовим експериментом 04.09.2026: бокс убито навмисно,
    наглядач правильно виявив смерть — і ВПАВ на `PermissionError: 'out'`, а
    рятувальний забір упав слідом. Сто готових сторінок урятували тільки
    чекпоінти в R2.

    Той самий клас, що й `out_dir` у плані: відносний шлях розкладається від
    того, ХТО запустив, а не від того, чому належить робота.
    """
    from gpurunner.config import data_dir

    return data_dir() / "htr" / "staging"


def _fallback_out_root() -> Path:
    """Куди класти справу, якщо план не сказав. Теж абсолютно й теж під наглядом."""
    from gpurunner.config import data_dir

    return data_dir() / "htr" / "out"


#: Скільки місця лишаємо тексту помилки в повідомленні про впалу справу.
ERR_LIMIT = 300


def short_error(text: str, *, limit: int = ERR_LIMIT) -> str:
    """Помилка справи так, щоб у неї влізла ПРИЧИНА, а не адреса.

    🔴🔴 Обрізання з початку з'їдало рівно те, заради чого повідомлення й
    існує. Помилка завантаження виглядає як `curl <presigned-URL> → rc=23`, а
    підписане посилання R2 саме по собі довше за 400 символів: у ліміт 300
    влізав шматок підпису й нічого більше. 21.09.2026 три справи поспіль
    упали з однаковим «curl https://…X-Amz-SignedHeader», і причина не
    збереглася НІДЕ — ні в стані, ні в журналі. Діагноз за таким рядком
    ставиться навмання: перше, що спало на думку, — «посилання протухло»,
    хоча в ньому ж написано `X-Amz-Expires=86400`, тобто доба.

    Тому URL згортається до впізнаваного (`…/cases/spr-576.tar?…`), а те, що
    лишилось, обрізається З СЕРЕДИНИ: початок і хвіст однаково потрібні, бо
    причина буває і там, і там.
    """
    import re

    def _fold(m: re.Match[str]) -> str:
        url = m.group(0)
        head, _, query = url.partition("?")
        tail = head.rsplit("/", 1)[-1] or head
        return f"…/{tail}?…" if query else f"…/{tail}"

    text = re.sub(r"https?://\S+", _fold, str(text)).strip()
    if len(text) <= limit:
        return text
    keep = limit - 3
    head = keep * 2 // 3
    return text[:head] + "…" + text[len(text) - (keep - head):]


def need_target_pph(plan: Any) -> float:
    """Ціль темпу заходу, стор/год; 0 — без цілі (старі плани й явний нуль)."""
    return float(getattr(plan, "target_pph", 0.0) or 0.0)


def need_from_plan(plan: Any, *, pages: int, max_hours: float | None = None,
                   budget_usd: float | None = None) -> Need:
    """Потреба заходу — ОДНА на всі входи.

    🔴 `--dry-run` будував свій `Need` окремо й без ручок: показував 8 шардів
    там, де прогін брав 4, і не бачив стелі ціни взагалі. Тобто безкоштовна
    перевірка ринку відповідала не на те питання, заради якого її роблять.
    """
    # 🔴 Матеріал заходу — ТИПОВИЙ по черзі, зважений сторінками, а не
    # найважчий. 23.09.2026 одна справа на 36 Мпікс із 194 поставила 6 ГБ на
    # шард усім (1 шард на 8-ГБ карту), а густина найгустішої справи (251 рядок
    # проти середніх 39) завищувала час заходу в 3.7 раза. Важку справу черги
    # раннер дочитує тим самим флотом: OOM сторінки він повторює, а не валить.
    mpx_vram, mpx_typ, lines = plan_material(plan)
    gb = plan_knob(plan, "vram_gb_per_shard") or gb_per_shard_for(mpx_vram)
    lines_assumed = not lines
    return Need(
        pages=pages,
        # Скільки справ у черзі: кожна має фіксовану ціну, і без цього числа
        # черга з 237 мікросправ важила для воріт як одна.
        cases=max(1, len(list(getattr(plan, "cases", None) or []))),
        max_hours=plan.max_hours if max_hours is None else max_hours,
        budget_usd=plan.budget_usd if budget_usd is None else budget_usd,
        disk_gb=plan.disk_gb,
        min_net_mbps=plan.min_net_mbps,
        max_cost_per_case=plan_knob(plan, "max_cost_per_case"),
        max_usd_per_1000_pages=plan_knob(plan, "max_usd_per_1000_pages"),
        # 🔴🔴 Ціль темпу: машина, що обережно її не дає, не береться НІКОЛИ.
        target_pph=need_target_pph(plan),
        # Підлога старого режиму (без цілі).
        min_pages_per_hour=float(plan_knob(plan, "min_pages_per_hour") or 0.0),
        # Стеля ціни ядро-години: ручка перекриває дефолт, нуль знімає ворота.
        max_usd_per_core_h=plan_knob(plan, "max_usd_per_core_h"),
        gb_per_shard=gb,
        frame_mpx=mpx_typ,
        lines_per_page=lines or LINES_PER_PAGE_ASSUMED,
        lines_assumed=lines_assumed,
        data_mb_per_page=plan_mb_per_page(plan),
        max_archive_mb=plan_max_archive_mb(plan),
        cores_per_shard=plan_knob(plan, "cores_per_shard") or 0.0,
    )


def plan_mb_per_page(plan: Any) -> float:
    """Скільки МБ кадрів на сторінку качає бокс — по справах, що знають свій обсяг."""
    cases = [c for c in (getattr(plan, "cases", None) or [])
             if int(getattr(c, "pages_bytes", 0) or 0) > 0
             and int(getattr(c, "n_pages", 0) or 0) > 0]
    pages = sum(int(c.n_pages) for c in cases)
    return sum(int(c.pages_bytes) for c in cases) / 1e6 / pages if pages else 0.0


def plan_max_archive_mb(plan: Any) -> float:
    """Найбільший архів кадрів у черзі, МБ — фіксована ціна входу на бокс.

    Береться МАКСИМУМ, а не середнє: черга читається по одному тому, і бокс
    мусить витримати найважчий із них. Нуль — жодна справа не знає свого обсягу.
    """
    sizes = [int(getattr(c, "pages_bytes", 0) or 0)
             for c in (getattr(plan, "cases", None) or [])]
    return max(sizes) / 1e6 if sizes else 0.0


def plan_material(plan: Any) -> tuple[float, float, float]:
    """(Мпікс для VRAM, типові Мпікс, типові рядки на сторінку); 0 = не міряли.

    🔴 Усі три — зважені СТОРІНКАМИ, бо флот читає сторінки, а не справи:
    справа на 20 сторінок не має права вирішувати долю черги на 5 000.

    - **VRAM** — p90 сторінок черги за `frame_mpx_p95` справи (старі плани без
      p95 — медіана). Не максимум: 23.09.2026 одна справа на 36 Мпікс із 194
      дала 6 ГБ на шард усім. Сторінки понад p90 раннер дочитує тим самим
      флотом, повторюючи сторінку при OOM.
    - **Темп** — середні зважені Мпікс і рядки. Не максимум: густина двох
      справ зі 237 (251 рядок) завищувала час усієї черги в 3.7 раза.

    Ручка `lines_per_page` перекриває мету: дослідник, що назвав число, знає
    матеріал краще.
    """
    cases = list(getattr(plan, "cases", None) or [])

    def weight(c: Any) -> float:
        return float(max(1, int(getattr(c, "n_pages", 0) or 0)))

    peaks = sorted(
        (float(getattr(c, "frame_mpx_p95", 0) or 0)
         or float(getattr(c, "frame_mpx_median", 0) or 0), weight(c)) for c in cases)
    peaks = [(m, w) for m, w in peaks if m > 0]
    mpx_vram = 0.0
    if peaks:
        total = sum(w for _, w in peaks)
        acc = 0.0
        for m, w in peaks:
            acc += w
            mpx_vram = m
            if acc >= 0.9 * total:
                break

    def weighted(attr: str) -> float:
        pairs = [(float(getattr(c, attr, 0) or 0), weight(c)) for c in cases]
        pairs = [(v, w) for v, w in pairs if v > 0]
        total = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / total if total else 0.0

    knob = plan_knob(plan, "lines_per_page") if hasattr(plan, "params") else 0
    lines = float(knob or 0) or weighted("lines_per_page_median")
    return mpx_vram, weighted("frame_mpx_median"), lines


def _absorb_rate(cs: Any, progress: dict[str, Any]) -> None:
    """Зберегти те, ЧИМ заміряний темп: сторінки цього прогону й стінний час.

    🔴 `pages_done` і `pages_per_hour` — про різні речі: перше включає підняте
    з чекпоінтів, друге рахується лише від зробленого зараз. Доки ці два числа
    лишались єдиними, калібровка не могла відрізнити захід, що прочитав 400
    сторінок, від догону, що підняв 380 і прочитав 20 (`htr.calibrate`).

    Обидва числа монотонні в межах прогону, тож беремо максимум: пізній тік із
    порожнім прогресом не сміє стерти замір.
    """
    resumed = int(progress.get("pages_resumed") or 0)
    done = int(progress.get("pages_done") or 0)
    cs.pages_this_run = max(int(getattr(cs, "pages_this_run", 0) or 0), max(0, done - resumed))
    cs.wall_sec = max(float(getattr(cs, "wall_sec", 0.0) or 0.0),
                      float(progress.get("wall_sec") or 0.0))


#: Ядер на шард, нижче яких регулятор не росте (дубль `REG_CORES_PER_SHARD`
#: раннера, заміряно 04.09: 8 шардів — 11.7 ядра, 16 — 17.0).
FLEET_CORES_PER_SHARD = 1.25
#: Найменша пам'ять шарда, яку дозволяємо як СТЕЛЮ (не старт): нижче заміряного
#: зайняття карти не бувало (spr-43: 0.4 ГБ на шард на 5.5 Мпікс).
FLEET_MIN_GB_PER_SHARD = 1.0


def fleet_ceiling(measured: dict[str, Any], start: int) -> int:
    """Стеля флоту для регулятора: скільки шардів ЦЯ машина потягне фізично.

    Ядра — `квота / 1.25`, пам'ять — покартково по 1 ГБ на шард. Не нижче
    старту: стеля, менша за старт, — це не стеля, а наказ стиснутись.
    """
    from gpurunner.core.htr_sizing import MAX_SHARDS, VRAM_HEADROOM

    # Ядра — ті самі, що бачать ворота (`cores_eff`: min(nproc, квота cgroup)),
    # а не `nproc` хоста: той на Q8000 115469 показує 80 при квоті 19.2.
    cores = float(measured.get("cores_eff") or measured.get("cores_seen")
                  or measured.get("cores") or 0)
    n_gpus = max(1, int(measured.get("n_gpus") or 1))
    per_card = float(measured.get("vram_free_min_gb") or 0) or (
        float(measured.get("vram_free_gb") or 0) / n_gpus)
    by_cores = int(cores // FLEET_CORES_PER_SHARD) if cores > 0 else start
    by_vram = (int(per_card * VRAM_HEADROOM // FLEET_MIN_GB_PER_SHARD) * n_gpus
               if per_card > 0 else start)
    return max(int(start), min(MAX_SHARDS, by_cores, by_vram))


def underdelivering(box_pages: int, box_sec: float, steady_pph: float,
                    target_here: float) -> bool:
    """Чи не дає бокс обіцяного (один тік серії `underdelivers`).

    Середнє від першої сторінки несе розгони флоту й старти справ; сталий
    замір регулятора — те, що машина дає. Вирок машині — за більшим із двох:
    простій нашого конвеєра заміною машини не лікується.
    """
    if target_here <= 0 or box_sec < UNDERDELIVER_WARMUP_SEC             or box_pages < UNDERDELIVER_WARMUP_PAGES:
        return False
    rate = max(box_pages / box_sec * 3600.0, steady_pph)
    return rate < UNDERDELIVER_SHARE * target_here


def queue_pace(pph: float, queue: list[Any]) -> float:
    """Темп БОКСА на цій черзі при темпі читання `pph`: з накладними справ.

    🔴🔴 Звірка обіцянки міряє сторінки бокса за годину від першої сторінки,
    тобто разом із фіксованою ціною кожної справи (розпакування, моделі,
    чекпоінт — `PER_CASE_SEC`). Порівнювати це з чистим темпом читання означало
    б гасити КОЖНУ машину на черзі дрібних справ: 194 справи по 24 сторінки при
    5 000 стор/год — це 55 хв читання й 87 хв накладних, і найкраща машина
    ринку «не дала б» обіцяного. Обіцянка переводиться в ті самі одиниці.
    """
    from gpurunner.core.htr_sizing import PER_CASE_SEC

    pages = sum(int(getattr(c, "n_pages", 0) or 0) for c in queue)
    if pph <= 0 or pages <= 0:
        return max(0.0, pph)
    hours = pages / pph + len(queue) * PER_CASE_SEC / 3600.0
    return pages / hours


def _sizing_view(sizing: Any) -> dict[str, Any]:
    """Розкладка у вигляді, придатному для JSON стану."""
    return {
        "shards": sizing.shards,
        "threads_per_shard": sizing.threads_per_shard,
        "limited_by": sizing.limited_by,
        "wasted_cores": sizing.wasted_cores,
        "pages_per_hour": round(sizing.pages_per_hour),
    }


class Supervisor:
    """Один захід: черга справ, один бокс за раз, спільний бюджет."""

    def __init__(
        self,
        plan: Plan,
        *,
        session: str | None = None,
        backend: VastBackend | None = None,
        tick_sec: int = TICK_SEC,
    ) -> None:
        self.plan = plan
        self.backend = backend or VastBackend()
        self.job = HTRCaseJob()
        self.tick_sec = tick_sec
        self.state = SupervisorState(session=session or new_session())
        # 🔴 Власник виводиться з ФАКТИЧНОГО імені сесії, а не з аргументу.
        #
        # Було: `setdefault("GPURUNNER_OWNER", f"htr-{session or 'sup'}")` —
        # тобто без `--session` ОБИДВІ паралельні сесії діставали однакового
        # власника `htr-sup`, і owner-фільтр, заради якого все й робилось,
        # переставав їх розрізняти. Причому стан брав уже інше, унікальне ім'я
        # (`new_session()`), тож власник і сесія розходились.
        #
        # `state.session` унікальний за побудовою (мітка часу до секунди), тож
        # два наглядачі ніколи не збігаються навіть без явного імені.
        os.environ.setdefault("GPURUNNER_OWNER", f"htr-{self.state.session}")
        self.state.cases = [
            CaseState(case=c.case, n_pages_expected=c.n_pages, out_dir=c.out_dir or None,
                      case_key=getattr(c, "case_key", "") or "")
            for c in plan.cases
        ]
        #: Імена справ, які наглядач уже знає. Черга довіска ідемпотентна саме
        #: за цим набором: справу з плану дописати ще раз не вийде, а повторний
        #: запис у чергу не додасть її вдруге — на боксі це прямі гроші за вже
        #: зроблене.
        self._appended_seen: set[str] = {c.case for c in plan.cases}
        #: Секрет складу на цій машині — один на ВЕСЬ захід.
        #:
        #: 🔴 Народжується тут, а не під час доставки, бо посилання зі секретом
        #: лягають у параметри роботи ще при сабміті — тобто ДО того, як бокс
        #: узагалі підняли. Переоренда бере той самий секрет: склад на новій
        #: машині піднімається заново, а параметри роботи вже поїхали.
        #:
        #: 🔴 Він же й вирішує, чи склад відкривається назовні: `origin.serve`
        #: слухає світ ЛИШЕ разом із секретом, тож забути одне з двох нема як.
        self._origin_token = secrets.token_urlsafe(24)
        #: Зовнішня адреса складу, коли її вдалось намацати; порожньо — доставка
        #: пішла запасним плечем.
        self._origin_outside = ""
        #: Канал ДО ЦІЄЇ машини, МБ/с, і чим саме зміряний. Заповнює
        #: `_gate_delivery_speed` фактом заливки ассетів.
        #:
        #: 🔴 Доти це число жило нотаткою сесії й помирало з нею: наступний
        #: захід про аплінк до тієї самої машини не знав нічого, хоч канал —
        #: єдиний вхід, що гуляє на два з половиною порядки (282 машини
        #: реєстру, медіана факт/обіцянка 0.266, мінімум 0.004). Тепер воно
        #: їде в реєстр разом із темпом, і стелі часу рахуються з нього, а не
        #: зі сталих.
        self._uplink_mbs = 0.0
        self._uplink_via = ""
        # 🔴 `plan.max_usd_per_1000_pages` за замовчуванням None — «стелю не
        # задавали». Передати це просто так означало затерти дефолт `Cfg`
        # і зробити `float > None` у перевірці ціни: наглядач падав
        # TypeError'ом рівно тоді, коли вимір дозрів і треба було
        # ухвалювати рішення. Порожнє значення = беремо дефолт.
        cfg_kwargs: dict[str, Any] = {}
        ceiling = plan_knob(plan, "max_usd_per_1000_pages")
        if ceiling is not None:
            cfg_kwargs["max_usd_per_1000"] = float(ceiling)
        self.cfg = Cfg(budget_usd=plan.budget_usd, max_hours=plan.max_hours,
                       **cfg_kwargs)
        self.spent_usd = 0.0
        #: Рахунок Vast за машини сесії (`charges`): None — ще не читали або
        #: порожній. Множник — у скільки рахунок більший за «ціна × час»
        #: (трафік, диск); ним прогноз множить ціну за годину.
        self._billed_usd: float | None = None
        self._bill_factor = 1.0
        self._bill_read_at = 0.0
        #: Гроші ЗАКРИТИХ оренд. Поточна додається зверху в `_accrue`; без
        #: цього поділу гасіння обнуляло облік (див. `_accrue`).
        self.settled_usd = 0.0
        #: Скільки разів цей захід уже брав бокс. Стеля — `plan.max_rents`.
        self.rents = 0
        #: 🔴 Стеля `max_rents` рахує лише ЗМАРНОВАНІ оренди — бокси, що не
        #: дали жодної готової сторінки. 06.09.2026 (P2, ф.230): другий бокс
        #: дочитав справу з чекпоінтів і завис на качанні наступної; стеля
        #: порахувала його як битий і вибила захід із сімома справами в черзі.
        #: Жорстка межа грошей лишається: не більше `2 × max_rents` оренд.
        self.rents_wasted = 0
        self._box_pages_done = 0
        self._box_best_usd_per_1000 = 0.0
        #: База правила 3-тер: гроші оренди, сторінки черги на боксі й мить,
        #: коли темп цього боксу ДОЗРІВ. None — ще не дозрів. Див. `_rent_money`.
        self._rent_base: tuple[float, int, float] | None = None
        self._destroy_failed = False
        self._ckpt_warned = False
        self._ckpt_seen_ok = False
        self._measured_pph = 0.0
        #: Коли востаннє везли чекпоінти зі складу на машині додому і чи
        #: приїхав хоч один: доти питаємо частіше (див. `BOX_CKPT_FIRST_SEC`).
        self._last_ckpt_sync = 0.0
        self._ckpt_home_seen = False
        self._price_warned = False
        self._fetched_idle = False
        self._pending: dict[str, Any] | None = None
        self.started = time.monotonic()
        # 🔴 Окремий відлік ВІД ОРЕНДИ. Вік прогресу міряти від старту
        # наглядача не можна: після переоренди свіжий бокс одразу «мовчить
        # 107 хвилин» (стільки живе сам захід), оголошується мертвим за три
        # хвилини після підйому — і цикл «взяв → убив → взяв» палить гроші,
        # не рахуючи жодної сторінки. Скидається на КОЖНІЙ успішній оренді.
        #: None — лише між гасінням (`_destroy`) і наступною орендою. Цикл
        #: нагляду в цьому проміжку не крутиться: після гасіння він або
        #: виходить, або спершу бере новий бокс, і той ставить відлік знову.
        self._rented_at: float | None = time.monotonic()
        self._handle: JobHandle | None = None
        self._offer: dict[str, Any] | None = None
        self._probe: dict[str, Any] = {}
        #: Розкладка, яку порахували ВОРОТА на живому залізі. Саме вона поїхала
        #: раннеру; план з картки оффера лишається поруч лише для порівняння.
        self._gate_sizing: Any = None
        self._boot_sec: float | None = None
        #: Скільки їхали НАШІ дані після готовності хоста (не провина машини).
        self._upload_sec: float | None = None
        self._host_ready_at: float | None = None
        self._owner = manifest.current_owner()
        #: Замки, які тримає цей захід, — щоб віддати їх у `finally`.
        self._locked: list[str] = []
        #: Машини, що вже підвели в цьому заході — другого шансу не даємо.
        self._failed_machines: set[int] = set()
        #: Чи є зараз відкрита (успішно здана) оренда — лише її закриває `_close_rent`.
        self._rent_open = False
        #: Машини, яким уже дали другу спробу після `ssh_auth_denied` у цьому заході.
        self._auth_retry_used: set[int] = set()
        #: Провали томів на боксі, про які вже сказано інцидентом: (справа, спроба).
        self._box_failures_seen: set[tuple[str, int]] = set()
        #: Чим саме закінчилась кожна спроба оренди в ЦЬОМУ пошуку. Вирок
        #: «жоден не пройшов ворота» без цього переліку не діагноз, а констатація.
        self._gate_rejects: list[str] = []
        #: Скільки `offer_taken` поспіль. Ринок не буває зайнятий підряд.
        self._taken_streak = 0
        #: Наша поправка до VRAM/шард, здобута ЦІНОЮ OOM у цьому заході.
        self._gb_bump = 0.0
        #: Розмір лога раннера й мить, коли він востаннє РІС. Приймач руху у
        #: фазі сетапу, де `_progress.json` ще не існує за побудовою.
        self._setup_log_size = -1
        self._setup_moved_at = time.monotonic()
        #: Останній прогрес флоту — з нього фінальний рядок калібрування бере
        #: справжні OOM і збої.
        self._last_progress: dict[str, Any] | None = None
        self._hb_stop = threading.Event()
        #: Потреба останньої оренди — те, що вибір ОБІЦЯВ (ціль на цій черзі).
        self._rent_need: Need | None = None
        #: Стеля флоту, передана раннеру на цій оренді (0 — не передавали).
        self._fleet_ceiling = 0
        #: Черга ЦІЄЇ оренди в порядку раннера (на догоні — підмножина плану).
        self._rent_queue: list[CasePlan] = list(plan.cases)
        #: (сторінки черги, мить) першої сторінки ЦЬОГО боксу — база темпу бокса.
        self._box_base: tuple[int, float] | None = None

    # ---- вхідна точка -----------------------------------------------------

    def _echo_knobs(self) -> str:
        """Надрукувати ЧИННІ ручки заходу — ті, з якими цей процес реально працює.

        🔴 Живий наглядач виконує код, завантажений на СТАРТІ: правка в
        `supervise/*.py` посеред заходу нічого в ньому не міняє, і ручка, яку
        фікс мав увімкнути, лишається мовчазно проігнорованою до перезапуску.
        Ознака зовні неочевидна — у `plan.json` параметр є, а в лозі дефолт.
        Тому чинні числа друкуються один раз на початку: розбіжність із планом
        стає видно відразу, без читання коду.
        """
        need = need_from_plan(self.plan, pages=self.plan.total_pages)
        ceiling = need.usd_per_1000_ceiling
        gb = self._gb_per_shard()
        # 🔴 Щільність друкується поруч зі стелею ціни, бо саме їх пара вирішує
        # вибір машини: стелю дивляться на прогноз темпу, а прогноз — на
        # щільність. 23.09.2026 її брак купив бокс, удвічі дорожчий за сторінку,
        # і дізнатись це можна було лише арифметикою по факту.
        lines = need.lines_per_page
        target = (f"ціль {need.target_pph:.0f} стор/год еталонного матеріалу = "
                  f"{need.target_here:.0f} на цій черзі (гарантовано) · "
                  if need.target_pph > 0 else "БЕЗ ЦІЛІ темпу · ")
        line = (
            f"[supervise] чинні ручки: {target}стеля ${ceiling:.2f}/1000 · "
            f"щільність {lines:.0f} рядків/стор"
            f"{' — ПРИПУЩЕНА (справи ще не читались), звірка на ходу' if need.lines_assumed else ''} · "
            f"VRAM/шард {gb:.1f} ГБ"
            f"{'' if plan_knob(self.plan, 'vram_gb_per_shard') else ' (від площі кадру)'} · "
            f"шардів {self._requested_shards() or 'авто'} · "
            f"бюджет ${self.plan.budget_usd:.2f} · стеля {self.plan.max_hours:.1f} год · "
            f"оренд {self.plan.max_rents}"
        )
        # 🔴 Підлога темпу мусить бути ВИДНА в рядку ручок. Ручка, яка мовчки
        # діє, невідрізнима від ручки, яка мовчки НЕ діє: саме так
        # `-p max_usd_per_1000_pages` два тижні не працював і не сказав ні слова.
        floor_pph = float(plan_knob(self.plan, "min_pages_per_hour") or 0)
        if floor_pph and need.target_pph <= 0:
            line += f" · не повільніше за {floor_pph:.0f} стор/год"
        self._say(line)
        for warn in getattr(self.plan, "warnings", []) or []:
            self._say(f"[supervise] ⚠ план: {warn}")
        return line

    def _preflight_credit(self) -> bool:
        """Чи є чим платити — ПЕРЕД пошуком офферів. Один дешевий запит.

        🔴 Пропуск цієї перевірки коштував 04.09.2026 вісімнадцяти запусків
        наглядача поспіль: кожен ранжував ринок, брав кандидата, діставав 400
        і виходив у `market_empty` з порадою дивитись реєстр банів. Причина
        весь час стояла в одному числі — `vast · 0.00 $`.
        """
        credit = self._credit_left()
        if credit is None:
            self._say("[supervise] баланс Vast: невідомо (запит не вдався)")
            return True
        need = min(0.5, self.plan.budget_usd / 4.0)
        self._say(f"[supervise] баланс Vast: ${credit:.2f}")
        if credit < need:
            why = (f"на акаунті Vast ${credit:.2f} при потребі щонайменше "
                   f"${need:.2f} — оренда не почнеться")
            self.state.finish(
                "no_credit", why,
                human_action="поповнити баланс Vast (https://cloud.vast.ai/billing/)",
            )
            return False
        return True

    def _open_log(self) -> None:
        """Наглядач веде ВЛАСНИЙ лог-файл, хоч би як його запустили.

        🔴 Досі все йшло тільки в stdout. Агент, який пустив наглядача фоном
        через пайп (`| tail`), при перериванні втрачав УВЕСЬ вивід: `tail`
        друкує в кінці, а кінця не було. Заміряно 2026-08-11: від обірваного
        заходу лишилось 706 байтів на чотири рядки, і причину двох знищених
        інстансів не було де прочитати. Лог мусить бути на диску завжди —
        інакше «дивись інциденти» перетворюється на «дивись нікуди».
        """
        try:
            path = state_dir() / f"{self.state.session}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh: IO[str] | None = path.open("a", encoding="utf-8", buffering=1)
            self.state.log_path = str(path)
        except OSError:
            self._log_fh = None
        # 🔴 Чорна скринька — ОДРАЗУ за логом. Два наглядачі 22.09.2026 зникли
        # без жодного сліду, і розтин не мав із чим працювати: лог обривався на
        # штатному рядку. Скринька пише причину смерті там, де Python її взагалі
        # може дізнатись, а там, де не може (`taskkill /F`), доказом стає її
        # власна тиша при живих крихтах. Див. `blackbox.postmortem`.
        from gpurunner.supervise import blackbox

        self._blackbox = blackbox.install(self.state.session, self._log_fh)

    def _say(self, line: str) -> None:
        """Надрукувати І записати. Друк без запису вже коштував розслідування."""
        print(line, flush=True)
        if getattr(self, "_log_fh", None):
            try:
                stamp = f"{datetime.now(tz=UTC):%H:%M:%S}"
                print(stamp, line, file=self._log_fh, flush=True)
            except (OSError, ValueError):
                pass

    def run(self) -> int:
        """Прогнати весь план. Повертає код виходу (див. `state.EXIT_*`)."""
        # 🔴🔴 ОДИН наглядач на сесію — ПЕРЕД першим записом стану. Другий
        # (подвійний старт задачі планувальника, повторний запуск агентом) інакше
        # переписав би стан живого своїм `failed` і взявся б за ту саму справу.
        try:
            self._locked.append(locks.acquire(
                f"session:{self.state.session}", owner=self._owner,
                session=self.state.session,
                ttl_sec=int(self.plan.max_hours * 3600) + 1800,
                note="наглядач сесії",
            ).resource)
        except locks.LockBusy as e:
            print(f"[supervise] ⛔ {e}\n[supervise] ⛔ другий наглядач тієї ж сесії "
                  f"не стартує; стан живого не чіпаю", flush=True)
            return EXIT_FAILED
        self.state.budget = self._budget_view()
        self.state.save()
        self._open_log()
        # `sink` свідомо НЕ поле датакласу: `SupervisorState.to_dict` робить
        # `asdict`, а той глибоко копіює кожне поле — файлову ручку скопіювати
        # не можна. Атрибут живе лише на екземплярі; `note` читає його через
        # `getattr(self, "sink", None)`.
        self.state.sink = self._log_fh  # type: ignore[attr-defined]
        self._echo_knobs()
        if not self._preflight_credit():
            self._say(f"[supervise] {self.state.why}")
            self._release_locks()
            self.state.save()
            return self.state.exit_code
        self._start_heartbeat()
        self._reconcile_past_bills()
        try:
            # 🔴 ОДНА оренда на всю чергу. Холодний старт коштує ~8 хв (pip,
            # 105 МБ моделей, ваги), і на семи дрібних справах це майже година
            # оренди без жодної прочитаної сторінки. Раннер бере справи
            # чергою сам; ми лише стежимо й забираємо.
            self._run_queue()
            self._settle()
            self._run_post_fetch()
        except KeyboardInterrupt:
            self.state.note("interrupt", "перервано з клавіатури", action="гашу оренду")
            self._destroy("перервано вручну")
            self.state.finish("failed", "перервано вручну")
        except Exception as e:
            # 🔴 Спершу РЯТУЄМО, потім гасимо. Виміряно 2026-08-11: збій одного
            # файла в SFTP пройшов крізь `fetch`, цей обробник вважав його крахом
            # і знищив бокс, на якому лежали всі 391 готові сторінки. Результат
            # уцілів лише завдяки чекпоінтам у R2 — тобто випадково.
            self.state.note("crash", f"{type(e).__name__}: {e}",
                            action="рятую результат, ПОТІМ гашу")
            self._rescue_before_destroy()
            self._destroy(f"наглядач упав: {e}")
            self.state.finish("failed", f"наглядач упав: {type(e).__name__}: {e}")
        finally:
            # 🔴 Задачу планувальника прибираємо ЗАВЖДИ. Забута `/sc once`
            # одного разу підняла НОВУ оренду о 23:59 без жодного нагляду —
            # тобто прибирання тут запобіжник від витрат, а не охайність.
            with contextlib.suppress(Exception):
                from gpurunner.supervise import detach

                detach.cleanup(self.state.session)
            self._hb_stop.set()
            self._release_locks()
            self.state.budget = self._budget_view()
            self.state.save()
        return self.state.exit_code

    def _run_post_fetch(self) -> None:
        """Облік замовника після ПОВНОГО забору (`plan.post_fetch`).

        Лише коли вердикт `ok`: неповна справа в реєстрі — це саме той
        застарілий зріз, що виглядає як відповідь. Збій команди — інцидент із
        дією для людини, а не зміна вердикту: декод уже на диску, а бокс погашено.
        """
        hooks = list(getattr(self.plan, "post_fetch", []) or [])
        if not hooks or self.state.verdict != "ok":
            return
        for hook in hooks:
            cmd = [str(a) for a in (hook.get("cmd") or [])]
            if not cmd:
                continue
            shown = " ".join(Path(cmd[0]).name if i == 0 else a for i, a in enumerate(cmd))
            try:
                done = subprocess.run(
                    cmd, cwd=hook.get("cwd") or None, capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    timeout=int(hook.get("timeout_sec") or 1800), check=False)
            except (OSError, subprocess.TimeoutExpired) as e:
                self.state.note("post_fetch_failed", f"{shown}: {e}",
                                action="облік доробити руками цією ж командою")
                continue
            if done.returncode:
                tail = (done.stderr or done.stdout or "").strip().splitlines()[-1:]
                self.state.note("post_fetch_failed",
                                f"{shown}: rc={done.returncode} {' '.join(tail)}".strip(),
                                action="облік доробити руками цією ж командою")
            else:
                self._say(f"[supervise] ✓ облік: {shown}")
        self.state.save()

    def _release_locks(self) -> None:
        for resource in self._locked:
            locks.release(resource, owner=self._owner)
        self._locked.clear()

    def _beat(self) -> None:
        """Одне серцебиття: гроші, бюджет, мітка часу — і на диск."""
        self._accrue()
        self.state.budget = self._budget_view()
        self.state.heartbeat = datetime.now(tz=UTC).isoformat(timespec="seconds")
        self.state.save()

    def _start_heartbeat(self) -> None:
        """Окремий потік, що оновлює стан у тихих фазах.

        🔴 Основний цикл пише стан лише між тіками, а підйом боксу (459 с на
        8591) і забір — це одна довга операція без тіків. Увесь цей час стан
        показував `$0.00` і старів, тобто виглядав однаково для «качає образ»
        і «наглядача вбито». Потік — daemon: він не має права тримати процес
        живим після кінця заходу.
        """
        def loop() -> None:
            while not self._hb_stop.wait(HEARTBEAT_SEC):
                with contextlib.suppress(Exception):
                    self._beat()

        threading.Thread(target=loop, name="supervise-heartbeat", daemon=True).start()

    def _run_queue(self) -> None:
        """Прогнати всю чергу на одному боксі."""
        # 🔴 Замок на КОЖНУ справу черги. Дві сесії, що взялись за ту саму
        # справу, пишуть в одну теку результатів і псують одна одній знаменник
        # повноти — а виявляється це вже на звірці, коли гроші витрачені.
        try:
            for case in self.plan.cases:
                self._locked.append(
                    locks.acquire(
                        f"case:{case.case}", owner=self._owner,
                        session=self.state.session,
                        ttl_sec=int(self.plan.max_hours * 3600) + 1800,
                        note=f"черга з {len(self.plan.cases)} справ",
                    ).resource
                )
        except locks.LockBusy as e:
            self.state.note("locked", str(e), action="не беруся за чужу роботу")
            self.state.finish(
                "failed", str(e),
                human_action="дочекатись іншої сесії або прибрати справу з плану",
            )
            return

        first = self.plan.cases[0]
        need = self._need_for(sum(c.n_pages for c in self.plan.cases))
        # 🔴 ПЕРЕД орендою — перевірити, чи не працює вже наш бокс. Наглядач
        # може померти не своєю смертю (закрито термінал, вимкнено ноутбук,
        # SIGKILL), і тоді на Vast лишається ЖИВИЙ інстанс, який рахує далі й
        # тарифікується. Без цієї перевірки наступний запуск орендував би
        # ДРУГУ машину: платили б за дві, а перша доробила б у порожнечу до
        # свого дедлайн-сторожа. Робота при цьому не гине (чекпоінти в R2),
        # але гроші подвоюються, і саме це найлегше не помітити.
        if not self._adopt_live_box():
            self.state.phase = "renting"
            self.state.why = f"черга з {len(self.plan.cases)} справ: шукаю бокс"
            self.state.save()
            if not self._rent_for(first, need=need, queue=self.plan.cases):
                return

        ssh_fail_streak = 0
        progress_miss_streak = 0
        last_phase = ""
        catchup_rounds = 0
        # 🔴 Витримка перевищень. Обидві термінальні гілки — «дорого» і «флот
        # сиплеться» — вимагають, щоб ознака ТРИМАЛАСЬ: одиничну просадку темпу
        # дає важкий аркуш або качання наступної справи, і гасити по ній
        # означає викинути вже оплачену роботу.
        price_breach_streak = 0
        price_breach_since = 0.0
        fail_streak = 0
        dead_streak = 0
        underdeliver_streak = 0
        dead_state_streak = 0
        while True:
            time.sleep(self.tick_sec)
            now_wall = time.time()
            asked = wrapup_mod.requested(self.state.session)
            if asked is not None:
                self._wrap_up(str(asked.get("why") or "попросили згорнути захід"))
                return
            progress, ssh_ok = self._read_progress()
            ssh_fail_streak = 0 if ssh_ok else ssh_fail_streak + 1
            progress_miss_streak = 0 if progress else progress_miss_streak + 1
            # Остання ВІДОМА фаза переживає обрив SSH: без неї наглядач гасив
            # би бокс із уже готовим результатом, бо `progress` при мертвому
            # каналі стає None і «done» забувається.
            if progress and progress.get("phase"):
                last_phase = str(progress["phase"])
            # 🔴 Довісок беремо ЛИШЕ поки бокс справді працює. На `done`/`failed`
            # раннер уже вийшов із циклу справ і нову справу не побачить — а
            # наглядач вніс би її в план, і звірка повноти потім чесно назвала б
            # захід неповним через роботу, якої ніхто не починав.
            if ssh_ok and last_phase not in ("done", "failed"):
                self._absorb_appended()
            # 🔴 Темп для РЕЄСТРУ беремо лише дозрілий (ті самі ворота, що для
            # бюджетного прогнозу): у перші хвилини бокс качає архів кадрів, і
            # число там близьке до нуля. Записати його означало б назавжди
            # оббрехати справну машину — а реєстр саме за цим числом і ранжує.
            if progress and float(progress.get("wall_sec") or 0) >= 900                     and int(progress.get("pages_done") or 0) >= 60:
                self._measured_pph = max(
                    self._measured_pph, float(progress.get("pages_per_hour") or 0)
                )
            self._refresh_bill()
            self._accrue()
            # 🔴 Чекпоінти зі складу на машині забираємо ПІД ЧАС роботи, а не
            # в кінці: вони лежать на тій самій машині, яка може вмерти, і доти
            # захід не має де відновитись. Рідше за тік — сам забір коштує
            # хвилини каналу, а частота тут не потрібна.
            period = (BOX_CKPT_SYNC_SEC if self._ckpt_home_seen
                      else BOX_CKPT_FIRST_SEC)
            if (self._box_transport and last_phase not in ("done", "failed")
                    and now_wall - self._last_ckpt_sync >= period):
                self._last_ckpt_sync = now_wall
                if self.sync_box_checkpoints():
                    self._ckpt_home_seen = True
            # 🔴 Зв'язність мусить бути ВИДНА зовні. Доти `ssh_fail_streak` жив
            # лише в пам'яті процесу: наглядач мовчки втрачав бокс, а стан
            # показував фазу «робота» й `human_action_required: false` — тобто
            # агент бачив здоровий захід рівно тоді, коли зв'язку вже не було.
            self.state.box = dict(self.state.box or {})
            self.state.box["ssh_ok"] = bool(ssh_ok)
            self.state.box["silent_min"] = round(self._progress_age(progress) / 60, 1)
            if not ssh_ok and ssh_fail_streak >= 2:
                self.state.why = (
                    f"SSH не відповідає {ssh_fail_streak} тіки поспіль "
                    f"({ssh_fail_streak * self.tick_sec} с) — бокс може бути втрачений"
                )
            # Лічильники витримки рахуються ТУТ, а не в `decide`: та функція
            # чиста й не має пам'яті між тіками — саме тому її можна ганяти
            # таблицею сценаріїв.
            now = time.monotonic()
            probe_obs = Obs(progress=progress, dph=self._dph() * self._bill_factor)
            if probe_obs.throughput_settled and probe_obs.usd_per_1000 > self.cfg.max_usd_per_1000:
                price_breach_streak += 1
                price_breach_since = price_breach_since or now
            else:
                price_breach_streak = 0
                price_breach_since = 0.0
            # 🔴 Стрік нарощуємо ЛИШЕ на значущій вибірці: на відновленій справі
            # `fresh` дорівнює нулю, і одна невдала сторінка дає рівно 100% —
            # інакше лічильник дозріває на порожньому місці й `decide` бачить
            # «частка тримається» там, де спроб було півтори.
            if (probe_obs.fail_ratio > self.cfg.fail_ratio_max
                    and probe_obs.fail_attempts >= self.cfg.fail_min_attempts):
                fail_streak += 1
            else:
                fail_streak = 0
            if probe_obs.shards_total and probe_obs.shards_dead * 2 >= probe_obs.shards_total:
                dead_streak += 1
            else:
                dead_streak = 0
            rent_usd, rent_pages, rent_settled_sec = self._rent_money(
                progress, settled=probe_obs.throughput_settled, now=now)
            box_pages, box_sec = self._box_pace(progress, now=now)
            target_here = (self._rent_need.target_here
                           if self._rent_need is not None else 0.0)
            if underdelivering(box_pages, box_sec, probe_obs.steady_pages_per_hour,
                               target_here):
                underdeliver_streak += 1
            else:
                underdeliver_streak = 0
            instance_state = self._instance_state()
            if str(instance_state).lower() in SOFT_DEAD_STATES:
                dead_state_streak += 1
            else:
                dead_state_streak = 0
            obs = Obs(
                progress=progress,
                price_breach_streak=price_breach_streak,
                price_breach_sec=(now - price_breach_since) if price_breach_since else 0.0,
                fail_streak=fail_streak,
                dead_streak=dead_streak,
                progress_age_sec=self._progress_age(progress),
                ssh_ok=ssh_ok,
                ssh_fail_streak=ssh_fail_streak,
                progress_miss_streak=progress_miss_streak,
                last_phase=last_phase,
                queue_pages_total=self._rent_queue_pages(),
                queue_pages_before=self._queue_pages_passed(progress),
                instance_state=instance_state,
                box_pages=box_pages,
                box_sec=box_sec,
                underdeliver_streak=underdeliver_streak,
                dead_state_streak=dead_state_streak,
                spent_usd=self.spent_usd,
                dph=self._dph() * self._bill_factor,
                elapsed_h=(time.monotonic() - self.started) / 3600.0,
                queue_left=0,          # чергою керує раннер, не наглядач
                catchup_rounds_done=catchup_rounds,
                catchup_passes=self.plan.catchup_passes,
                rent_age_sec=time.monotonic() - cast(float, self._rented_at),
                setup_stall_sec_seen=self._setup_stall_sec(),
                rent_usd=rent_usd,
                rent_pages=rent_pages,
                rent_settled_sec=rent_settled_sec,
                box_best_usd_per_1000=self._box_best_usd_per_1000,
            )
            # 🔴 Продуктивність оренди — сторінки БОКСА через межі справ, а не
            # максимум однієї справи: на черзі по 20 сторінок кожна оренда
            # інакше рахувалась би змарнованою й з'їдала `max_rents`.
            self._box_pages_done = max(self._box_pages_done, obs.pages_done, box_pages)
            if obs.throughput_settled and obs.usd_per_1000 > 0:
                # найкраща ціна, яку цей бокс показав сам — еталон правила 3-тер
                self._box_best_usd_per_1000 = min(
                    self._box_best_usd_per_1000 or obs.usd_per_1000, obs.usd_per_1000)
            action, why = decide(obs, self.cfg)
            self._absorb_queue(progress, why)

            if action == "wait":
                # 🔥 Теплий бокс: результат черги вже готовий — забираємо ОДИН
                # раз зараз, не чекаючи гасіння, але в реєстр не пишемо: запис
                # робить фінальний забір після `done`, інакше один захід дав би
                # два рядки `ok` і роздвоїв замір темпу.
                if last_phase == "idle" and not self._fetched_idle:
                    self._fetched_idle = True
                    self._fetch_queue(record=False)
                continue
            if action in ("fetch_and_finish", "next_case"):
                self._fetch_queue()
                self._destroy("черга завершена")
                return
            if action == "price_alert":
                # Не гасимо: робота вже йде, а знищити її дорожче за переплату.
                # Але кажемо ОДИН раз і з числами — далі рішення за людиною.
                if not self._price_warned:
                    self._price_warned = True
                    self.state.note("over_price_per_page", why,
                                    action="прогін НЕ спиняю; якщо ціна не влаштовує "
                                           "— спинити руками й узяти іншу машину")
                continue
            if action == "catchup":
                # Догін виконує САМ раннер, ще до публікації `done`. Наглядач
                # тут лише чекає — і має так і казати, а не рапортувати роботу,
                # якої не робить.
                catchup_rounds += 1
                continue
            if action == "startup_failed":
                self.state.note("startup_failed", why,
                                action="гашу; причина в даних, не в боксі",
                                machine_id=self._machine_id())
                self._destroy("робота так і не почалась")
                self.state.finish("failed", why,
                                  human_action="перевірити assets_url / pages_url у плані")
                return
            if action == "destroy_overpriced":
                # Ціна тримається вище стелі — і це вже не переплата, а інша
                # арифметика заходу: за ті самі гроші деінде пройде вдесятеро
                # більше справ. Чекпоінти забираємо, машину називаємо в реєстрі,
                # щоб добір не приніс її наступного разу першою ж зіркою.
                self.state.note("over_price_per_page", why,
                                action="гашу оренду; чекпоінти забрано",
                                machine_id=self._machine_id(),
                                fleet=self._fleet_snapshot(obs))
                self._record_box(self._price_outcome(), why)
                self._fetch_queue(best_effort=True)
                self._destroy(why)
                self.state.finish(
                    "budget_stop", why,
                    human_action="узяти іншу машину (`gpurunner htr run` наново) "
                                 "або підняти --max-usd-per-1000, якщо ця ціна влаштовує",
                )
                return
            if action == "underdelivers":
                pace = box_pages / max(box_sec, 1.0) * 3600.0
                mid = self._machine_id()
                self.state.note("underdelivers", why,
                                action="забираю прочитане, гашу, беру іншу машину",
                                machine_id=mid)
                # У реєстр — ЗАМІР, а не вирок «помер»: машина жива, просто на
                # цьому матеріалі не дає обіцяного; `slow_run` забувається за
                # тиждень і банить лише повтором. Її виміряний темп іде у вибір
                # наступних заходів («замір б'є модель»).
                self._measured_pph = pace
                self._record_box("slow_run", why)
                if mid:
                    self._failed_machines.add(mid)
                self._fetch_queue(best_effort=True)
                self._destroy("не дає обіцяного темпу")
                if self._destroy_failed:
                    return
                if not self._rent_the_rest(progress):
                    return
                ssh_fail_streak = underdeliver_streak = dead_state_streak = 0
                continue
            if action in ("rerent", "destroy_dead_box"):
                # Мертвий бокс і живий бокс із обсипаним флотом — це різні
                # діагнози, і в інциденті вони мусять читатись по-різному.
                fleet_only = bool(action == "rerent" and ssh_ok and obs.shards_total)
                # 🔴 Діагноз — у вирок, а не в логи боксу, що зараз згорять:
                # 11.09.2026 (904-24-198) флот упав на 34 збоях, і причину не
                # знав ніхто. Найчастіший текст збою сторінки (без імені кадру).
                errs = [str(sh["last_error"]).split(": ", 1)[-1]
                        for sh in (obs.progress or {}).get("shards") or []
                        if sh.get("last_error")]
                if errs:
                    why = f"{why}; збій сторінки: {max(set(errs), key=errs.count)}"
                self.state.note("fleet_dying" if fleet_only else "box_dead", why,
                                action="гашу і беру інший",
                                machine_id=self._machine_id(),
                                fleet=self._fleet_snapshot(obs))
                # 🔴 OOM — це НАШ прорахунок розміру проти щільності матеріалу, а
                # не поломка хоста: `oom_pages` калібрує константи й нікого не
                # банить. Інакше кожен кадр-розворот, що не вліз у 2.5 ГБ на
                # шард, вивозив би з ринку справну машину на 21 день.
                oom_driven = bool(fleet_only and obs.oom_events)
                self._record_box(
                    "oom_pages" if oom_driven else self._death_outcome(), why)
                if oom_driven:
                    self._bump_gb_per_shard(obs)
                self._fetch_queue(best_effort=True)
                self._destroy("бокс втрачено")
                if self._destroy_failed:
                    # 🔴 Другу машину при непогашеній першій брати НЕ можна:
                    # горіли б дві одночасно, а ручку від першої ми вже
                    # втратили. Вердикт уже стоїть і не перезаписується.
                    return
                # Потреба перераховується НА ЗАЛИШОК. Старий `need` ніс бюджет
                # і години, зафіксовані на старті заходу, тож ворота пускали
                # машину під «8 год і $3», коли насправді лишалось 2 год і
                # $0.50 — оренда оплачувалась, дані заливались, і перший же
                # тік гасив її за бюджетом.
                # Машина, що впала в ЦЬОМУ заході, не береться в ньому знову:
                # 23.09.2026 ту саму 42769 орендували повторно через п'ять хвилин.
                if (mid := self._machine_id()):
                    self._failed_machines.add(mid)
                if not self._rent_the_rest(progress):
                    return
                ssh_fail_streak = underdeliver_streak = dead_state_streak = 0
                continue
            if action in ("destroy_budget", "destroy_deadline"):
                self._fetch_queue(best_effort=True)
                self._destroy(why)
                self.state.finish(
                    "budget_stop" if action == "destroy_budget" else "deadline", why,
                    human_action="поповнити баланс і перезапустити — чекпоінти в R2"
                    if action == "destroy_budget" else "підняти --max-hours",
                )
                return

    def _wrap_up(self, why: str) -> None:
        """Згорнути захід на прохання: спинити роботу, забрати, погасити.

        🔴 Той самий шлях, яким захід завершується на стелі грошей, — інакше
        з'явився б ще один вихід, на якому щось забувається. Прочитане
        забирається ДО гасіння, машина гаситься завжди, вердикт окремий
        (`cancelled`): текст на диску справжній, і повторний захід дочитає
        решту, а не почне заново.
        """
        self._say(f"[supervise] згортаюсь на прохання: {why}")
        self.state.note("wrapup", why, action="забираю прочитане й гашу оренду")
        wrapup_mod.clear(self.state.session)
        with contextlib.suppress(Exception):
            self._quiesce_runner()
        self._fetch_queue(best_effort=True)
        self._destroy(why)
        self.state.finish(
            "cancelled", f"згорнуто на прохання: {why}",
            human_action="прочитане на диску; щоб дочитати решту — той самий "
                         "захід ще раз, він відновиться з точок відновлення",
        )

    def _quiesce_runner(self) -> None:
        """Спинити раннер на боксі, щоб він не писав під час забору."""
        if self._handle is None:
            return
        client = self.backend._ssh(self._handle, timeout=20)
        try:
            self._exec_on(
                client,
                "pkill -f '[h]tr_case_run.py' >/dev/null 2>&1; sleep 1; true")
        finally:
            client.close()

    @staticmethod
    def _fleet_snapshot(obs: Obs) -> dict[str, Any]:
        """Числа, за якими біду можна буде розібрати ПІСЛЯ смерті боксу.

        `state.shards` живе лише поки бокс живий: останній прогрес приходить
        без шардів і затирає список порожнім. Тому знімок кладеться в інцидент.
        """
        return {
            "shards_alive": obs.shards_alive,
            "shards_total": obs.shards_total,
            "pages_done": obs.pages_done,
            "pages_failed": obs.pages_failed,
            "fail_ratio": round(obs.fail_ratio, 3),
            "oom_events": obs.oom_events,
            "pages_per_hour": round(obs.pages_per_hour),
            "usd_per_1000": round(obs.usd_per_1000, 3),
            "shards": list((obs.progress or {}).get("shards") or []),
        }

    def _price_outcome(self) -> str:
        """Хто винен у ціні: сусід на карті чи сама машина.

        Проба це вже бачила — і саме її число, а не здогад, вирішує, на скільки
        машину пам'ятати.
        """
        m = self._probe or {}
        try:
            per_card = float(m.get("vram_total_mb") or 0) / 1024.0 / max(1, int(float(m.get("n_gpus") or 1)))
            free = float(m.get("vram_free_min_mb") or 0) / 1024.0
        except (TypeError, ValueError):
            return "overpriced"
        return "card_busy" if per_card > 0 and free < per_card * 0.6 else "overpriced"

    def _pages_left_in_queue(self, progress: dict[str, Any] | None = None) -> int:
        """Скільки сторінок черги ще не зроблено — для переоцінки потреби.

        🔴 Рахується за `case_index`, а НЕ за статусом `done`: той виставляється
        лише у `_fetch_queue`, тобто після всієї черги. Та сама вада, що вже
        полагоджена в `_queue_pages_before` — тут вона лишалась, і ціна інша:
        на смерті боксу під час 5-ї справи з 5 переоцінка давала 5000 сторінок
        замість тисячі, ворота міряли кандидатів по роботі, якої вже не треба,
        і відкидали здорові машини → «жоден не пройшов ворота заліза» при
        80% готовності й живих чекпоінтах.
        """
        p = progress or {}
        passed = self._queue_pages_passed(progress) + int(p.get("pages_done") or 0)
        return max(1, self._rent_queue_pages() - passed)

    def _need_for(self, pages: int) -> Need:
        # Години й бюджет — НА ЗАЛИШОК, решта ручок спільною читалкою.
        return need_from_plan(
            self.plan, pages=pages,
            max_hours=self.plan.max_hours - (time.monotonic() - self.started) / 3600.0,
            budget_usd=self.plan.budget_usd - self.spent_usd,
        )

    def _gb_per_shard(self) -> float:
        """Скільки VRAM просити на шард — З УСІХ місць, де це можна задати.

        🔴🔴 Ланцюг рвався посередині. `vram_gb_per_shard` доходив до
        раннера, але той його НЕ ЧИТАВ: число шардів приходило вже готовим
        (`_auto_shards` викликається лише при `shards < 1`), а рахували його
        ворота — зі СВОГО `need.gb_per_shard`, куди параметр не потрапляв.
        Тобто ручка міняла те, чим раннер рахував би шарди САМ, а сам він
        ніколи не рахує.

        Ціна виміряна на сповідках 2026-08-12: попросили 2.8 ГБ на шард,
        дістали 8 шардів (порахованих із дефолтних 2.5), реальне споживання
        1.5-4.3 ГБ, сумарно 23.3 з 24.6 ГБ — карта забита, **2045 збоїв проти
        549 готових сторінок**.
        """
        # Без ручки — від площі кадру (`gb_per_shard_for`): саме це число
        # бачили ворота й скоринг, і саме його мусить побачити раннер, інакше
        # він порахує флот від свого дефолту невідомого матеріалу.
        # 🔴 Раннеру йде СТАРТОВА пам'ять (обережна), а не та, якою вибір
        # рахує потенціал машини: див. `gb_per_shard_start_for`.
        base = (plan_knob(self.plan, "vram_gb_per_shard")
                or gb_per_shard_start_for(plan_material(self.plan)[0]))
        if not self._gb_bump:
            return base
        # Поправка діє й тоді, коли в плані «авто»: там дефолт знає лише
        # `htr_sizing`, а ми вже знаємо, що на ЦЬОМУ матеріалі його замало.
        return max(base, self._gb_bump)

    def _bump_gb_per_shard(self, obs: Obs) -> None:
        """Підняти VRAM/шард після OOM — і не брати наступний бокс із тим числом.

        🔴🔴 Доти наглядач на `fleet_dying` з OOM гасив бокс і брав наступний
        З ТИМ САМИМ `gb_per_shard`, тобто ходив по колу, поки не вичерпає стелю
        оренд. Виміряно 31.08.2026: пакет 1900 (8 томів, 2900 стор.) пройшов за
        ніч ЧОТИРИ бокси з однаковим наслідком, і поріг довелось піднімати
        руками 3.5 → 2.5 → 4.0 → 6.0; цикл спинився лише на 6.0. Три оренди й
        кілька годин — ціна того, що машина не робила очевидного кроку сама.

        🔴 Крок 0.7 ГБ, а не 1.5. Це підйом ЗА ФАКТОМ OOM, тобто вже дорогий, і
        стрибати через нього не можна: завищений поріг ріже флот на КОЖНІЙ
        наступній оренді й міняє вибір машини (скоринг ділить на нього ще при
        ранжуванні ринку — 4.5 замість 3.3 колись відправило захід на RTX 4060 Ti
        з трьома шардами замість RTX 3090 із сімома). З 3.3 сходинки виходять
        4.0 → 4.7 → 5.4: щоразу мінус один-два шарди, а не половина флоту.
        """
        current = self._gb_per_shard() or GB_PER_SHARD_DEFAULT
        ceiling = self._card_vram_gb() or 0.0
        bumped = current + 0.7
        if ceiling and bumped > ceiling / 2:
            # Більше половини карти на шард — це вже не розкладка, а один шард
            # на бокс; далі підіймати нема куди, справа просто важка.
            bumped = min(bumped, ceiling / 2)
        if bumped <= current:
            return
        self._gb_bump = bumped
        self.state.note(
            "gb_per_shard_up",
            f"OOM {obs.oom_events} подій на {obs.pages_failed} збоїв: "
            f"VRAM/шард {current:.1f} → {bumped:.1f} ГБ",
            action="наступний бокс беру з піднятим порогом, а не з тим самим",
        )
        self._say(f"[supervise] ⬆ VRAM/шард {current:.1f} → {bumped:.1f} ГБ (OOM)")
        self._log_calibration(current, ok=False, oom=obs.oom_events,
                              attempts=obs.pages_failed)

    def _log_calibration(self, gb: float, *, ok: bool, oom: int = 0,
                         attempts: int = 0) -> None:
        """Дописати замір «геометрія × поріг × наслідок» у реєстр.

        🔴 Таблиця в `htr_sizing` стоїть на чотирьох точках, дві з яких
        суперечать одна одній, — і кожна з них здобута падінням прогону, а
        записана в чиюсь пам'ять сесії. Реєстр робить наступне уточнення
        питанням до ДАНИХ, а не до спогадів: він у git, тобто його видно в
        дифі й можна прочитати через рік.
        """
        case = self.plan.cases[0] if self.plan.cases else None
        row = {
            "at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "case": getattr(case, "case", ""),
            "frame_mpx": float(getattr(case, "frame_mpx_median", 0) or 0),
            # VRAM/шард рахується від p95 — тож і в реєстрі мусить бути саме він,
            # інакше калібрувати формулу нема з чим.
            "frame_mpx_p95": float(getattr(case, "frame_mpx_p95", 0) or 0),
            "frame_aspect": float(getattr(case, "frame_aspect_median", 0) or 0),
            "gb_per_shard": round(gb, 2),
            "shards": getattr(self._gate_sizing, "shards", 0),
            "card_vram_gb": round(self._card_vram_gb(), 1),
            "ok": ok,
            "oom_events": oom,
            "pages_failed": attempts,
            # 🔴 Режим препроцесингу: 04.09.2026 саме він, а не площа кадру,
            # вирішив OOM (сірі на 2.5 ГБ — 141 збій зі 191, сирі RGB на 4.5 —
            # нуль). Без цього поля калібровка пам'яті шарда не відрізняє двох
            # режимів і не може сказати, яке зниження порога безпечне.
            "enhance": str(plan_knob_str(self.plan, "enhance") or "auto"),
            # Скільки флот справді тримав і скільки йому дозволили — щоб
            # наступна калібровка бачила, чи регулятор ріс до стелі.
            "shards_max": int(getattr(self, "_fleet_ceiling", 0) or 0),
        }
        # 🎛 Замір регулятора — те, чим калібрувати наступні плани замість
        # формули з площі кадру: де коліно і скільки пам'яті шард узяв насправді.
        fleet = (self._last_progress or {}).get("regulator") or {}
        if fleet:
            row.update({
                "knee_shards": fleet.get("knee"),
                "rates_by_shards": fleet.get("rates"),
                "vram_per_shard_mb": fleet.get("vram_per_shard_mb"),
                "rss_per_shard_mb": fleet.get("rss_per_shard_mb"),
                "card_total_mb": fleet.get("card_total_mb"),
                "gpu": str((self._probe or {}).get("gpu") or ""),
            })
        try:
            path = registry_dir() / "shard_calibration.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _note_setup_movement(self, sftp: Any) -> None:
        """Запам'ятати, чи виріс лог раннера. Дешево: один `stat` по вже
        відкритому SFTP.

        🔴 Без цього фаза сетапу не має приймача взагалі. 17.08.2026 бокс завис
        на качанні 412-МБ архіву й простояв 50 хвилин при `ssh_ok: true` і
        `human_action_required: false`; годинна стеля його не спіймала б, бо
        50 менше за 60.
        """
        try:
            size = int(sftp.stat("/workspace/gpurunner/_runner.log").st_size or 0)
        except Exception:
            return
        if size > self._setup_log_size:
            self._setup_log_size = size
            self._setup_moved_at = time.monotonic()

    def _reset_setup_watch(self) -> None:
        """Нова машина — новий лог раннера: мірка руху сетапу починається заново.

        Без цього лог нової машини (з нуля) не перевищував розміру логу
        попередньої, «останній рух» лишався від першої оренди, і кожну
        наступну машину гасили як завислу за хвилину після підйому (B1,
        24.09.2026: «сетап не посувається 34/37/39/43 хв», чотири машини).
        """
        self._setup_log_size = -1
        self._setup_moved_at = time.monotonic()

    def _setup_stall_sec(self) -> float:
        """Скільки лог раннера не росте. Нуль, поки нічого не бачили."""
        if self._setup_log_size < 0:
            return 0.0
        return time.monotonic() - self._setup_moved_at

    def _death_outcome(self) -> str:
        """Хто винен у смерті боксу: хост чи ми самі.

        🔴 Інстанс у стані «gone» виглядає однаково, хто б його не прибрав —
        і доти це БЕЗУМОВНО писалось у провину машині. Але бокс гасять і ззовні:
        людина через `gpurunner cancel`, сам Vast при відкликанні пропозиції,
        сусідня сесія. Виміряно 04.09.2026 навмисним експериментом: справний
        V100, що читав 1714 стор/год, дістав `died_under_load` за те, що його
        вбили ми. Ще один такий випадок — і машина йде з ринку на 21 день ні за
        що, а це найдешевша швидка карта, яку ми знаходили.

        Розрізнити можна: наше власне гасіння лишає слід у реєстрі прогонів.
        """
        handle = self._handle
        if handle is None:
            return "died_under_load"
        try:
            fresh = manifest.get(handle.id)
        except Exception:
            return "died_under_load"
        if fresh is not None and fresh.status == JobStatus.CANCELLED:
            return "user_stop"
        return "died_under_load"

    def _card_vram_gb(self) -> float:
        """Скільки пам'яті має ОДНА карта останньої проби (не сума боксу)."""
        m = self._probe or {}
        try:
            total = float(m.get("vram_total_gb") or 0) or (
                float(m.get("vram_total_mb") or 0) / 1024.0)
            return total / max(1, int(float(m.get("n_gpus") or 1)))
        except (TypeError, ValueError):
            return 0.0

    def _adopt_live_box(self) -> bool:
        """Підхопити СВІЙ бокс, який пережив смерть наглядача.

        Шукається в реєстрі прогонів (він переживає перезапуск) серед
        нетермінальних хендлів ЦЬОГО власника; беремо лише той, що досі живий
        на боці Vast. Чужі не чіпаємо взагалі — це те саме правило, що з
        гасінням: мітка інстансу не є доказом власності, доказ лежить у реєстрі.

        🔴🔴 Спільного власника НЕ ДОСИТЬ. Дві половини одної кампанії часто
        йдуть під одним `GPURUNNER_OWNER` — і тоді наглядач A бачив ЖИВИЙ бокс
        наглядача B, вважав його осиротілим і «підхоплював»: забрав би
        результат і погасив машину, поки B на ній ще рахує. Ледь не сталось
        2026-08-12 (кампанія `htr-olhopil`), спинили руками за півхвилини.

        Тому підхоплення вимагає ДВОХ доказів, що бокс справді нічий:

        1. він рахує САМЕ МОЇ справи — перетин із планом; чужий план означає
           чужу роботу, хоч би який власник стояв у реєстрі;
        2. на ці справи немає ЖИВОГО замка іншої сесії — замок протухає за
           мертвим процесом і TTL, тож живий замок = живий наглядач.
        """
        try:
            from gpurunner.core.models import JobStatus

            alive = [
                h for h in manifest.load()
                if h.backend == "vast" and h.job_name == "htr_case"
                and (h.owner or "") == self._owner
                and h.status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.UNKNOWN)
            ]
        except Exception:
            return False
        for handle in alive:
            try:
                inst = self.backend._instance(handle)
            except Exception:
                continue
            if not inst or str(inst.get("actual_status") or "").lower() in ("exited", "gone"):
                continue
            if not self._is_adoptable(handle):
                continue
            self._handle = handle
            self._offer = inst
            self._rented_at = time.monotonic()
            self._reset_setup_watch()
            if str(handle.remote_id) not in self.state.instances:
                self.state.instances.append(str(handle.remote_id))
            cores = float(inst.get("cpu_cores_effective") or 0)
            dph = offer_dph(inst)
            self.state.note(
                "adopted",
                f"підхопив живий інстанс {handle.remote_id} від перерваного заходу"
                + (f" ({cores:.0f} ядер, ${dph:.3f}/год)" if cores else ""),
                action="не орендую другий бокс — стежу за цим",
            )
            # 🔴 Підхопити — не означає схвалити. Раніше наглядач брав БУДЬ-ЯКИЙ
            # живий бокс і більше не питав, чи він вартий роботи: перезапуск на
            # повільній машині мовчки продовжував платити за повільність.
            # Знищувати чужу вже зроблену роботу не можна, але СКАЗАТИ треба —
            # з числами, щоб рішення ухвалювала людина, а не тиша.
            # Звірка обіцянки з фактом (`underdelivers`) діє й на підхопленому
            # боксі: якщо він не дає цілі, наглядач сам забере зроблене й
            # переорендує, а не лише скаже.
            self.state.phase = "running"
            self.state.save()
            return True
        return False

    def _is_adoptable(self, handle: JobHandle) -> bool:
        """Чи цей живий бокс справді НІЧИЙ (див. `_adopt_live_box`)."""
        params = handle.params or {}
        on_box = {str(params.get("case") or "")}
        on_box |= {str(c.get("case") or "") for c in (params.get("cases") or [])}
        on_box.discard("")
        mine = {c.case for c in self.plan.cases}
        if on_box and not (on_box & mine):
            self.state.note(
                "not_adopting",
                f"інстанс {handle.remote_id} рахує ЧУЖІ справи "
                f"({', '.join(sorted(on_box))[:60]}) — це не осиротілий бокс, "
                f"а робота сусідньої сесії під тим самим власником",
                action="не чіпаю; шукаю власну машину",
            )
            return False
        for case in sorted(on_box & mine) or sorted(mine):
            holder = locks.read(f"case:{case}")
            if not holder or holder.get("session") == self.state.session:
                continue
            if not locks.stale_reason(holder):
                self.state.note(
                    "not_adopting",
                    f"справу {case} тримає ЖИВА сесія {holder.get('session')} — "
                    f"бокс не осиротілий",
                    action="не чіпаю; шукаю власну машину",
                )
                return False
        return True

    def _box_pages_total(self, progress: dict[str, Any] | None, *,
                         count_resumed: bool = False) -> int:
        """Сторінки, прочитані на цьому боксі за захід, — монотонно.

        Раннер публікує `box_pages_done`: лічильник усіх прочитаних сторінок,
        що не скидається між справами й фазами. Сума «справи позаду за
        `case_index` + поточна» на переході з конвеєра до великих справ падала
        до нуля (індекс і черга розходились), і правило «гроші проти сторінок»
        гасило машину, що читала (spr-80, 24.09.2026). Для старого раннера
        лишається колишня сума.
        """
        p = progress or {}
        if p.get("box_pages_done") is not None:
            return int(p.get("box_pages_done") or 0)
        pages = self._queue_pages_before(progress) + int(p.get("pages_done") or 0)
        if not count_resumed:
            pages -= int(p.get("pages_resumed") or 0)
        return pages

    def _rent_queue_pages(self) -> int:
        """Сторінки черги ЦІЄЇ оренди (на догоні — лише недочитані справи)."""
        queue = self._rent_queue or self.plan.cases
        return sum(int(c.n_pages or 0) for c in queue)

    def _queue_pages_passed(self, progress: dict[str, Any] | None) -> int:
        """Скільки сторінок черги пройдено ПОЗА поточним `pages_done`.

        З `box_pages_done` — монотонно: прочитане й підняте на боксі за захід
        мінус поточний `pages_done` (його `Obs` додає сам). Без нього — стара
        сума за `case_index`.
        """
        p = progress or {}
        if p.get("box_pages_done") is not None:
            passed = (int(p.get("box_pages_done") or 0)
                      + int(p.get("box_pages_resumed") or 0))
            return max(0, passed - int(p.get("pages_done") or 0))
        return self._queue_pages_before(progress)

    def _queue_pages_before(self, progress: dict[str, Any] | None) -> int:
        """Скільки сторінок черги вже позаду — за `case_index`, а не за статусом.

        🔴 Статус `done` виставляється лише у `_fetch_queue`, тобто ПІСЛЯ всієї
        черги. Рахуючи «пройдене» по ньому, наглядач весь захід вважав його
        нулем — і до залишку додавались усі вже прочитані справи. На черзі з
        п'яти справ це давало вчетверо завищений прогноз витрат і `destroy_budget`
        за хвилини до кінця роботи, за яку вже заплачено.
        """
        idx = max(1, int((progress or {}).get("case_index") or 1))
        # Черга, яку реально отримав БОКС: на догоні це лише недочитані
        # справи, і індекс раннера показує в неї, а не в повний план.
        return sum(int(c.n_pages or 0) for c in self._rent_queue[: idx - 1])

    def _case_state_for(self, progress: dict[str, Any] | None) -> CaseState | None:
        """Стан справи, про яку зараз звітує раннер — за ІМЕНЕМ, а не індексом.

        🔴 Індекс раннера — позиція в черзі ЦІЄЇ оренди. На догоні черга —
        підмножина плану, і за індексом прогрес лягав би на чужу справу.
        """
        name = str((progress or {}).get("case") or "")
        if not name:
            idx = max(1, int((progress or {}).get("case_index") or 1)) - 1
            if idx < len(self._rent_queue):
                name = self._rent_queue[idx].case
        for cs in self.state.cases:
            if cs.case == name:
                return cs
        return None

    def _absorb_appended(self) -> None:
        """Забрати справи з черги довіска й довісити їх ЖИВОМУ боксу (ДОРОБКА 48).

        Порядок кроків тут не косметичний, і кожен стоїть перед наступним не
        випадково:

        1. **Бюджет і строк — ПЕРШІ.** Довісок мовчки виносив би захід за межу,
           яку задала людина, і це рівно той клас перевитрат, проти якого
           писалась решта запобіжників. Перевіряє саме наглядач, а не команда:
           між дописуванням і цим тіком минає час, за який захід міг з'їсти
           решту бюджету.
        2. **Замок на справу.** Початкова черга бере замок на кожну справу; без
           того самого тут дві сесії взялися б за одну справу, писали б в одну
           теку й зіпсували знаменник повноти — а видно це вже після витрат.
        3. **План і облік ПЕРЕД боксом.** Забір іде `zip(plan.cases,
           state.cases)`: справа, про яку знає бокс, але не знає наглядач,
           порахується, приїде в стейджинг і нікуди не розкладеться. Тому
           спершу вносимо, і лише потім штовхаємо.
        """
        if self._handle is None:
            return
        fresh = append_mod.drain(self.state.session, self._appended_seen)
        for raw in fresh:
            name = str(raw.get("case") or "").strip()
            pages = int(raw.get("n_pages") or raw.get("estimated_n_pages") or 0)

            self._accrue()
            left_usd = self.plan.budget_usd - self.spent_usd
            left_h = self.plan.max_hours - (time.monotonic() - self.started) / 3600.0
            need_h = (pages / max(1.0, self.state.cases[0].pages_per_hour or 0)
                      if self.state.cases and self.state.cases[0].pages_per_hour
                      else 0.0)
            if left_usd <= 0 or left_h <= 0:
                self.state.note(
                    "append_refused",
                    f"{name}: лишилось ${left_usd:.2f} і {left_h:.2f} год — не беру",
                    action="підняти --budget/--max-hours і довісити знову")
                continue
            if need_h and need_h > left_h:
                self.state.note(
                    "append_refused",
                    f"{name}: {pages} стор. ≈ {need_h:.2f} год, а лишилось {left_h:.2f}",
                    action="підняти --max-hours і довісити знову")
                continue

            try:
                self._locked.append(
                    locks.acquire(
                        f"case:{name}", owner=self._owner, session=self.state.session,
                        ttl_sec=int(max(0.1, left_h) * 3600) + 1800,
                        note="довісок на живий бокс",
                    ).resource
                )
            except locks.LockBusy as e:
                self.state.note("append_refused", f"{name}: {e}",
                                action="дочекатись іншої сесії")
                continue

            case = CasePlan(
                case=name,
                pages_url=str(raw.get("pages_url") or ""),
                n_pages=pages,
                out_dir=str(raw.get("out_dir") or ""),
                case_key=str(raw.get("case_key") or ""),
                local_dir=str(raw.get("case_dir") or ""),
                ckpt_urls=list(raw.get("ckpt_urls") or []),
                resume_urls=list(raw.get("resume_urls") or []),
                params=dict(raw.get("params") or {}),
            )
            self.plan.cases.append(case)
            self.state.cases.append(CaseState(
                case=case.case, n_pages_expected=case.n_pages,
                out_dir=case.out_dir or None,
                case_key=getattr(case, "case_key", "") or ""))

            payload = {
                "case": case.case,
                "pages_url": case.pages_url,
                "estimated_n_pages": case.n_pages,
                "ckpt_urls": case.ckpt_urls,
                "resume_urls": case.resume_urls,
                **case.params,
            }
            if self._push_append(payload):
                self.state.note("append", f"{case.case}: {pages} стор. довісено на бокс",
                                action="")
                self._say(f"[supervise] 🧊 довісок: {case.case} → бокс")
            else:
                # Справа лишається в плані: бокс її не взяв, але наглядач про неї
                # знає, і забір/звірка не пропустять її мовчки.
                self.state.note("append_push_failed",
                                f"{case.case}: не дописав на бокс",
                                action="перевірити SSH; справу видно в стані")
            self.state.save()

    def _push_append(self, payload: dict[str, Any]) -> bool:
        """Дописати один рядок у файл-довісок на боксі.

        `>>` з одного короткого рядка — атомарний дозапис; саме тому формат
        JSONL, а не спільний масив, який довелося б перечитувати-переписувати.
        """
        if self._handle is None:
            return False
        # 🔴 Одинарні лапки в тілі JSON закривають лапки оболонки. Ім'я справи
        # приходить від людини, тож екранування тут не педантизм: `230-1-50'` у
        # назві зробило б із дозапису довільну команду.
        line = json.dumps(payload, ensure_ascii=False).replace("'", "'\\''")
        script = (
            "mkdir -p /tmp/htrcase && "
            f"printf '%s\\n' '{line}' >> /tmp/htrcase/_append.jsonl && "
            "wc -l < /tmp/htrcase/_append.jsonl"
        )
        try:
            client = self.backend._ssh(self._handle, timeout=30)
        except Exception as exc:
            self.state.note("append_push_failed", repr(exc))
            return False
        try:
            _, out, _ = client.exec_command(script, timeout=60)
            return bool(out.read().decode("utf-8", "replace").strip())
        except Exception as exc:
            self.state.note("append_push_failed", repr(exc))
            return False
        finally:
            with contextlib.suppress(Exception):
                client.close()

    def _absorb_queue(self, progress: dict[str, Any] | None, why: str) -> None:
        self.state.why = why
        if progress:
            self._last_progress = progress
            # Фаза БОКСУ (`running` / `catchup` / `done` / `idle`) — те, що
            # агент хоче бачити; власні фази наглядача (`fetching`, `finished`)
            # вона не перебиває.
            box_phase = str(progress.get("phase") or "")
            if box_phase and self.state.phase not in ("fetching", "finished"):
                self.state.phase = box_phase
            # 🎛 Що флот виміряв сам: коліно, темп на кожному рівні, пам'ять
            # шарда. Видно в `htr state --json` без читання логів боксу.
            if progress.get("regulator"):
                self.state.box = dict(self.state.box or {})
                self.state.box["fleet"] = progress["regulator"]
            # Спершу закриті томи, потім поточний: повтор проваленого тому
            # мусить повернути йому `running`, а не бути перебитим старим провалом.
            self._absorb_box_results(progress)
            cs_live = self._case_state_for(progress)
            if cs_live is not None:
                cs = cs_live
                cs.status = "running"
                cs.pages_done = int(progress.get("pages_done") or 0)
                cs.pages_failed = int(progress.get("pages_failed") or 0)
                cs.pages_per_hour = float(progress.get("pages_per_hour") or 0)
                cs.eta_sec = progress.get("eta_sec")
                _absorb_rate(cs, progress)
            self.state.shards = list(progress.get("shards") or [])
            self._watch_checkpoints(progress)
        self.state.budget = self._budget_view()
        self.state.save()

    def _absorb_box_results(self, progress: dict[str, Any]) -> None:
        """Томи, які бокс уже закрив: скільки текстів дав кожен і хто впав.

        🔴 irnbuv1 q23, 14.09.2026: 8 томів із 23 впали на боксі без жодного
        чекпоінта, а стан до самого забору показував їх `running` з нулем
        сторінок (або з «11 збоями», успадкованими від попереднього тому) —
        жоден приймач цього не бачив, доки бокс не згас.
        """
        for r in progress.get("results") or []:
            try:
                idx = int(r.get("index") or 0) - 1
            except (TypeError, ValueError):
                continue
            # Індекс — позиція в черзі ОРЕНДИ; ім'я справи — у тій самій черзі.
            name = str(r.get("case") or "") or (
                self._rent_queue[idx].case if 0 <= idx < len(self._rent_queue) else "")
            cs_found = next((c for c in self.state.cases if c.case == name), None)
            if cs_found is None:
                continue
            cs = cs_found
            if cs.status in ("done", "incomplete"):
                continue   # уже звірено забором — правда диска сильніша
            n_txt = int(r.get("n_pages_txt") or 0)
            if r.get("complete"):
                cs.status = "running"
                cs.pages_done = max(cs.pages_done, n_txt)
                cs.pages_failed = 0
                cs.detail = f"на боксі повна: {n_txt} текстів, чекає забору"
                continue
            attempt = int(r.get("attempt") or 1)
            err = short_error(str(r.get("error") or "неповна").splitlines()[0])
            cs.status = "failed"
            cs.pages_done = max(cs.pages_done, n_txt)
            cs.detail = f"на боксі впала (спроба {attempt}): {err}"
            key = (cs.case, attempt)
            if key in self._box_failures_seen:
                continue
            self._box_failures_seen.add(key)
            self.state.note(
                "case_failed", f"{cs.case}: {err}",
                action=("бокс повторить том у кінці черги" if r.get("retry_pending")
                        else "том лишиться непрочитаним — причина в `_runner.log` "
                             "першої справи черги"),
            )

    def _registry_consequence(self, machine_id: int | None,
                              outcome: str) -> tuple[str, Any]:
        """Що реєстр НАСПРАВДІ зробив із машиною — словами для інциденту.

        🔴 Текст мусить збігатися з тим, що сталося: `ssh_auth_denied` на машині
        з успіхами реєстр судить як транзієнт (попередження), а інцидент писав
        «у чорний список» (V100 33283, 14.09.2026).
        """
        if outcome in boxes.NEUTRAL:
            return "НЕ винна, у реєстр іде без бану", None
        try:
            verdict = boxes.verdicts().get(int(machine_id)) if machine_id is not None else None
        except Exception:
            verdict = None
        if verdict is None:
            return "у реєстр (вердикт не прочитався)", None
        if verdict.state == "banned":
            return f"у чорний список ({verdict.reason})", verdict
        return f"попередження, не бан ({verdict.reason})", verdict

    def _box_ckpt_dir(self) -> Path:
        """Куди вдома складаються чекпоінти, привезені з машини.

        🔴 Ключ — СПРАВА, а не сесія. Точка відновлення потрібна саме тоді,
        коли попередній захід не дожив: наглядача вбили, комп'ютер
        перезавантажили, процес упав. Сесія щоразу нова, тож тека, названа по
        ній, робила б привезені чекпоінти невидимими рівно для того заходу,
        який мав би ними скористатись, — і справа читалась би з нуля за повні
        гроші, хоч усе прочитане лежить на диску.
        """
        slug = _slug(self.plan.cases[0].case) if self.plan.cases else "case"
        return _staging_root() / "_boxckpt" / slug

    def sync_box_checkpoints(self) -> int:
        """Привезти з машини чекпоінти, яких удома ще немає. Скільки привезли.

        🔴 Це не оптимізація забору, а єдиний захист роботи при складі на самій
        машині: чекпоінт, який лежить на боксі, гине разом із боксом — на
        відміну від чекпоінта в бакеті. Тому кличеться періодично ПІД ЧАС
        роботи, а не в кінці, і будь-який збій тут — нотатка, а не зупинка:
        робота на машині від цього не постраждала.
        """
        if not self._box_transport or self._handle is None:
            return 0
        from gpurunner.htr import box_transport as bt

        dest = self._box_ckpt_dir()
        dest.mkdir(parents=True, exist_ok=True)
        remote_tar = "/tmp/_ckpt_pull.tgz"
        local_tar = dest / "_pull.tgz"
        # 🔴 Везти лише НОВЕ. Чекпоінти накопичуються, і тар усієї теки щоп'ять
        # хвилин означає, що під кінець великої справи ті самі десятки
        # мегабайтів переїжджають додому знову й знову — а пакує їх машина, за
        # яку платять погодинно.
        home = {p.relative_to(dest).as_posix()
                for p in dest.rglob("ckpt_*.tgz")}
        try:
            ep = bt.endpoint_of(self.backend, self._handle)
            client = self.backend._ssh(self._handle, timeout=30)
            try:
                listing = self._exec_on(
                    client,
                    f"cd {bt.REMOTE_ROOT}/ckpt 2>/dev/null "
                    f"&& find . -name 'ckpt_*.tgz' -printf '%P\n' || true")
                fresh = sorted(set(listing.split()) - home)
                if not fresh:
                    return 0
                # Порожня тека чекпоінтів — штатний стан перших хвилин.
                quoted = " ".join(shlex.quote(name) for name in fresh)
                have = self._exec_on(
                    client,
                    f"cd {bt.REMOTE_ROOT}/ckpt && tar czf {remote_tar} {quoted} "
                    f"&& stat -c %s {remote_tar} || echo no")
            finally:
                client.close()
            said = str(have).strip().splitlines()[-1:] or [""]
            if not said[0].isdigit():
                return 0
            bt.pull(ep, remote_tar, local_tar,
                    timeout=self._transfer_ceiling(int(said[0])))
        except Exception as exc:
            self.state.note("box_ckpt_sync_failed", f"{type(exc).__name__}: {exc}",
                            action="робота на машині триває; повторимо наступним тіком")
            return 0
        import tarfile

        before = len(list(dest.rglob("ckpt_*.tgz")))
        try:
            with tarfile.open(local_tar) as tf:
                tf.extractall(dest, filter="data")
        except Exception as exc:
            self.state.note("box_ckpt_bad", f"{type(exc).__name__}: {exc}")
            return 0
        finally:
            local_tar.unlink(missing_ok=True)
        now = len(list(dest.rglob("ckpt_*.tgz")))
        if now > before:
            self._say(f"[supervise] чекпоінтів удома: {now} (+{now - before})")
        return max(0, now - before)

    def _fetch_from_store(self, staging: Path) -> bool:
        """Зібрати вивід із чекпоінтів — звідки б вони не лежали. True — щось є.

        Бакет і склад на машині відрізняються лише способом дістати той самий
        тарбол: presigned-посиланням чи `scp`. До петлі машини локальний `curl`
        не достукається, тому вибір робиться тут, а не в самому заборі.
        """
        return (self._fetch_via_box(staging) if self._box_transport
                else self._fetch_via_r2(staging))

    def _fetch_via_box(self, staging: Path) -> bool:
        """Зібрати вивід із чекпоінтів, привезених з машини. True — щось є.

        Та сама процедура, що й для бакета: чекпоінти інкрементальні, тож
        розпаковуються ПО ПОРЯДКУ з перезаписом — кожен наступний несе нові й
        дорослі файли.
        """
        import tarfile

        self.sync_box_checkpoints()
        home = self._box_ckpt_dir()
        if not home.is_dir():
            return False
        pulled = 0
        for case in self.plan.cases:
            slug = _slug(case.case)
            balls = sorted(home.rglob(f"{slug}/**/ckpt_*.tgz"))
            if not balls:
                continue
            dest = staging / slug
            dest.mkdir(parents=True, exist_ok=True)
            for ball in balls:
                try:
                    with tarfile.open(ball) as tf:
                        tf.extractall(dest, filter="data")
                    pulled += 1
                except Exception as exc:
                    self.state.note("box_ckpt_bad", f"{ball.name}: {exc!r}")
        if pulled:
            self.state.note("fetched_via_box",
                            f"результат зібрано з {pulled} чекпоінт-архівів, "
                            f"привезених із машини")
        return bool(pulled)

    def _fetch_via_r2(self, staging: Path) -> bool:
        """Зібрати вивід із чекпоінтів у R2. True — щось таки забрали.

        Чекпоінти інкрементальні, тож розпаковувати треба ПО ПОРЯДКУ з
        перезаписом: кожен наступний несе нові й дорослі файли. Це та сама
        процедура, якою 2026-08-12 вручну відновили дві справи (788+788 і
        351+351 файлів) після того, як бокс уже було знищено.

        Помилка тут не фатальна: не вийшло — лишається штатний SFTP.
        """
        import subprocess
        import tarfile

        pulled = 0
        for case in self.plan.cases:
            urls = list(case.resume_urls or [])
            if not urls:
                continue
            dest = staging / _slug(case.case)
            dest.mkdir(parents=True, exist_ok=True)
            tmp = staging / f"_r2_{_slug(case.case)}.tgz"
            for i, url in enumerate(urls, 1):
                rc = subprocess.call(
                    ["curl", "-fsSL", "--max-time", "300", "-o", str(tmp), url],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if rc != 0:
                    # 404 = такого чекпоінта ще немає; далі їх теж не буде.
                    break
                try:
                    with tarfile.open(tmp) as tf:
                        tf.extractall(dest, filter="data")
                    pulled += 1
                except Exception as exc:
                    self.state.note("r2_ckpt_bad", f"{case.case} №{i}: {exc!r}",
                                    action="цей архів пропущено")
                finally:
                    tmp.unlink(missing_ok=True)
        if pulled:
            self.state.note(
                "fetched_via_r2",
                f"результат зібрано з {pulled} чекпоінт-архівів у R2 замість "
                f"пофайлового SFTP",
                action="службові файли теж їдуть архівом, SFTP не задіяний",
            )
        return pulled > 0

    #: Три файли боксу, спільні для всієї черги: лог раннера, стан, прогрес.
    #: Лежать у КОРЕНІ стейджингу — саме звідти їх бере розкладка.
    _SERVICE_ROOT_FILES = ("_runner.log", "_status.json", "_progress.json")

    def _fetch_service_bundles(self, staging: Path) -> list[Path]:
        """Службові файли — одним архівом на справу, тим самим шляхом, що й текст.

        🔴🔴 Чому не SFTP. Тіло результату приїжджає зі сховища одним об'єктом,
        і доти лишався «дотяг службового»: `listdir` на КОЖНУ справу черги плюс
        `get` на КОЖЕН лог шарда. На 236 справах це ~2600 обертів через океан —
        забір висів понад 30 хвилин, з'їв усю стелю фази й лишив 237 справ без
        вердикту, хоч текст уже лежав удома. Заміри одного плеча на
        орендованому боксі (`box_transport.py`): SFTP paramiko 0.48 МБ/с проти
        20.07 у HTTP, тобто в 42 рази. За 99 заходів SFTP не дав повноти НІ
        РАЗУ самостійно і п'ять разів зіпсував вердикт уже привезеному тексту.

        Best-effort за побудовою: це діагностика. Чого немає — того немає, і
        жодна справа через це не міняє вердикту.
        """
        import subprocess

        if self._box_transport:
            return self._pull_box_service(staging)
        got: list[Path] = []
        staging.mkdir(parents=True, exist_ok=True)
        for case in self.plan.cases:
            url = case.service_url
            if not url:
                continue
            tmp = staging / f"_svc_{_slug(case.case)}.tgz"
            rc = subprocess.call(
                ["curl", "-fsSL", "--max-time", "120", "-o", str(tmp), url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if rc != 0:
                # 404 = раннер до цієї справи не дійшов або нічого не записав.
                tmp.unlink(missing_ok=True)
                continue
            got += self._unpack_service(tmp, staging, _slug(case.case))
        if got:
            self.state.note("service_fetched",
                            f"службові файли: {len(got)} з архівів сховища "
                            f"(без пофайлового SFTP)")
        return got

    def _unpack_service(self, ball: Path, staging: Path, slug: str) -> list[Path]:
        """Розпакувати службовий архів справи. Порожній список — не вийшло."""
        import tarfile

        dest = staging / slug
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(ball) as tf:
                tf.extractall(dest, filter="data")
        except Exception as exc:
            self.state.note("service_bundle_bad", f"{slug}: {exc!r}",
                            action="діагностика цієї справи лишиться на боксі")
            return []
        finally:
            ball.unlink(missing_ok=True)
        got: list[Path] = []
        # 🔴 Три файли боксу СПІЛЬНІ для всієї черги, і чим пізніша справа, тим
        # повніший її знімок лога. Тому переносимо в корінь із перезаписом:
        # останній привезений і є найповніший.
        for name in self._SERVICE_ROOT_FILES:
            src = dest / name
            if src.is_file():
                shutil.move(str(src), str(staging / name))
                got.append(staging / name)
        logs = dest / "logs"
        if logs.is_dir():
            got += sorted(logs.glob("*.log"))
        return got

    def _pull_box_service(self, staging: Path) -> list[Path]:
        """Те саме зі складу на машині: ОДИН tar і ОДНЕ з'єднання.

        Склад стоїть на петлі самої машини, тож локальний `curl` до нього не
        достукається — везе його та сама процедура, що й чекпоінти. Різниця з
        колишнім SFTP не в протоколі, а в кількості обертів: два виклики на
        всю чергу замість двох на кожну справу.
        """
        if self._handle is None:
            return []
        from gpurunner.htr import box_transport as bt

        staging.mkdir(parents=True, exist_ok=True)
        ball = staging / "_service_pull.tgz"
        try:
            client = self.backend._ssh(self._handle, timeout=30)
            try:
                said = self._exec_on(
                    client,
                    f"cd {bt.REMOTE_ROOT}/{bt.SERVICE_DIR} 2>/dev/null "
                    f"&& tar czf /tmp/_svc_pull.tgz . && echo yes || echo no")
            finally:
                client.close()
            if "yes" not in str(said):
                return []
            bt.pull(bt.endpoint_of(self.backend, self._handle),
                    "/tmp/_svc_pull.tgz", ball,
                    known_hosts=_staging_root() / "_known_hosts")
        except Exception as exc:
            self.state.note("service_pull_failed", f"{type(exc).__name__}: {exc}",
                            action="діагностика лишиться на боксі")
            return []
        # У складі архіви лежать пласко (`<справа>.tgz`), тож розкладаємо їх
        # кожен у свою теку — так само, як приїхали б із бакета.
        import tarfile

        tray = staging / "_service"
        try:
            with tarfile.open(ball) as tf:
                tf.extractall(tray, filter="data")
        except Exception as exc:
            self.state.note("service_bundle_bad", f"склад: {exc!r}")
            return []
        finally:
            ball.unlink(missing_ok=True)
        got: list[Path] = []
        for item in sorted(tray.glob("*.tgz")):
            got += self._unpack_service(item, staging, _slug(item.stem))
        shutil.rmtree(tray, ignore_errors=True)
        if got:
            self.state.note("service_fetched",
                            f"службові файли: {len(got)} одним архівом зі складу")
        return got

    def _watch_checkpoints(self, progress: dict[str, Any]) -> None:
        """Помітити, що точок відновлення НЕМАЄ, поки це ще можна виправити.

        🔴 Вичерпані або протерміновані presigned-посилання лишали слід лише в
        лозі на боксі — у який агент за контрактом не дивиться. Захід міг
        годинами їхати без жодної точки відновлення й дізнатись про це аж тоді,
        коли бокс помер і рятувати виявилось нічого.
        """
        ck = progress.get("ckpt") or {}
        if not ck or self._ckpt_warned:
            return
        # 🔴 Лічильник `ok` живе ПОСПРАВНО: раннер створює його заново на кожну
        # справу черги. Міряти від ОРЕНДИ означало гарантовану хибну тривогу
        # на другій же справі: до її початку від оренди вже минуло більш як
        # пів години, а свіжий лічильник — нуль. Виміряно 2026-08-12: тривога
        # спрацювала при цілком робочих чекпоінтах (п'ять справ уже закрито,
        # 236 resume-посилань на кожну справу в плані). Тому пам'ятаємо, що
        # хоч ОДИН чекпоінт у цьому заході вдався, і на другій справі мовчимо.
        if ck.get("ok"):
            self._ckpt_seen_ok = True
        if ck.get("last_ok"):
            self._ckpt_seen_ok = True
        stuck = bool(ck.get("exhausted"))
        dry = (not self._ckpt_seen_ok
               and (time.monotonic() - cast(float, self._rented_at)) > 1800)
        if stuck or dry:
            self._ckpt_warned = True
            self.state.note(
                "no_recovery_point",
                "чекпоінти не пишуться (посилання вичерпані або протухли)"
                if stuck else "за 30+ хв від оренди не залито ЖОДНОГО чекпоінта",
                action="робота НЕ переживе смерті боксу — при обриві все з нуля",
            )

    def _fetch_queue(self, *, best_effort: bool = False, record: bool = True) -> None:
        """Забрати ВСЕ одним заходом і розкласти по справах.

        `record=False` — проміжний забір із теплого бокса: результат на диску,
        а рядок реєстру лишається фінальному забору.

        Раннер кладе кожну справу у власну підтеку `/kaggle/working/<slug>/`,
        тож забір один, а звірка — окрема на кожну.
        """
        if best_effort:
            # 🔴 Аварійний забір іде ПІСЛЯ смерті, повільності чи ціни боксу — і
            # записати тут `ok` означало б погасити в реєстрі щойно зафіксований
            # вирок: 23.09.2026 «помер під навантаженням» машини 42769 закрився
            # «успіхом» цього ж забору, і її взяли знову за п'ять хвилин.
            record = False
        if self._handle is None:
            return
        staging = _staging_root() / f"_queue_{self.state.session}"
        self.state.phase = "fetching"
        self.state.save()
        #: Причина падіння забору, якщо воно було. Вирок справам ставить звірка
        #: по диску, а це — лише пояснення для тих, кого вона не підтвердила.
        fetch_error = ""
        # 🔴🔴 СПЕРШУ R2, і лише потім SFTP. Вузьке місце забору — не смуга,
        # а КРУГОВІ ОБЕРТИ: `_download_tree` тягне пофайлово, а одна справа це
        # 2379 файлів на 30 МБ (тексти + рамки рядків + голос). На черзі з
        # восьми це ~13 тисяч файлів: при 50-100 мс на обмін через океан лише
        # оберти дають 11-22 хвилини, скільки б не було мегабайтів. Ті самі
        # 150 МБ одним об'єктом із R2 їдуть секунди — CDN, egress безкоштовний.
        # Заміряно 2026-08-12: «на диску нуль за 15 хвилин фази fetching»,
        # тоді як ручний забір через R2 дав 62 МБ за секунди.
        self._flush_checkpoints()
        got_r2 = self._fetch_from_store(staging)
        try:
            # 🔴 Стеля на фазу цілком. Таймаут сокета ловить мертвий обмін, але
            # не ловить «живий, проте нескінченний» забір: 13 тисяч дрібних
            # файлів через океан можуть тягтися годинами, а лічильник оренди
            # цокає. Понад стелею вважаємо, що дотягувати нічого: те, що вже
            # приїхало (а головне — R2), лишається, решту добере `fetch-ckpt`.
            handle = self._handle   # звужений тип: вище вже перевірено на None
            if got_r2:
                # 🔴 Текст уже вдома, лишилась діагностика — і вона їде тим
                # самим шляхом, одним архівом на справу. SFTP тут більше немає
                # взагалі: саме «дотяг службового» пофайлово вішав забір на
                # 30+ хвилин при цілому результаті (23.09.2026).
                what = "забір службових архівів"

                def fetch() -> Any:
                    return self._fetch_service_bundles(staging)
            else:
                # Сховище не дало нічого — лишається повний обхід теки по SFTP.
                # Це шлях аварії, а не робочий: за 99 заходів сюди доходило 8.
                what = "забір по SFTP"

                def fetch() -> Any:
                    return self.backend.fetch_outputs(handle, staging)
            self._with_deadline(fetch, seconds=FETCH_PHASE_MAX_SEC, what=what)
        except BackendError as e:
            first = str(e).splitlines()[0]
            self.state.note("fetch_failed", first)
            # 🔴 Тут стояв голий `return` — і це був найгірший баг усього
            # наглядача. Справи лишались у статусі `running`, `_settle`
            # рахує лише `done/incomplete/failed`, порожні списки падали в
            # гілку `else` → вердикт **ok**, код виходу 0,
            # `human_action_required: false`. А бокс уже знищено. Тобто при
            # НУЛІ забраних сторінок агент, який за контрактом читає лише
            # цей JSON, звітував людині успіх.
            #
            # 🔴🔴 Лікування було надмірне: усі справи ставились `failed` і
            # виконання поверталось ДО розкладання. А коли текст уже приїхав
            # із R2, на диску лежать ПОВНІ справи — і вердикт однаково казав
            # «результат не забрано з боксу». Заміряно 23.09.2026 на черзі з
            # 237 справ: 5115/5115 сторінок на диску (перевірено вибіркою по
            # всій черзі), усі 237 позначені `failed`. Ціна: реєстр вважає їх
            # непрочитаними, і наступний захід платить за ту саму роботу
            # вдруге — при $0.233 за тисячу це $1.19 на рівному місці.
            #
            # Тому рішення ухвалює ЗВІРКА ПО ДИСКУ, як і в успішному заборі:
            # розкладаємо те, що вже є, і `failed` дістають лише ті справи,
            # які диск НЕ підтвердив. Порожній забір так само дає `failed` —
            # бо `verify_case` на порожній теці повного не покаже.
            if best_effort:
                return
            fetch_error = first
        for case, cs in zip(self.plan.cases, self.state.cases, strict=False):
            self._place_case(case, cs, staging)
        if got_r2 and not fetch_error:
            self._top_up_from_box(staging)
        if fetch_error:
            # Забір упав — але вирок уже поставила звірка. Неповним дописуємо
            # ПРИЧИНУ (щоб було видно, чому бракує), повним не чіпаємо нічого.
            for cs in self.state.cases:
                if not cs.complete:
                    cs.status = "failed"
                    cs.detail = (f"результат не забрано з боксу: {fetch_error}"
                                 + (f"; {cs.detail}" if cs.detail else ""))
            self.state.save()
        self._fetch_queue_tail(staging, record=record, fetch_error=fetch_error)

    def _flush_checkpoints(self) -> None:
        """Перед забором — попросити раннер залити чекпоінт ЗАРАЗ.

        Тоді в сховищі є все прочитане, і забір бере його HTTP, а не тримає
        машину, що тарифікується, на пофайловому SFTP. Не вийшло — лишається
        добір дельтою з машини (`_top_up_from_box`).
        """
        ask = getattr(self.backend, "request_checkpoint", None)
        if ask is None or self._handle is None or self._box_transport:
            return
        try:
            got = ask(self._handle)
        except Exception as exc:
            got = f"error: {type(exc).__name__}: {exc}"
        if got == "done":
            self._say("[supervise] раннер залив останній чекпоінт на прохання — забираю зі сховища")
        elif got != "no_runner":
            self.state.note("ckpt_flush_failed", str(got)[:200],
                            action="чого бракуватиме в сховищі, доберу з машини")

    def _top_up_from_box(self, staging: Path) -> None:
        """Дотягнути з машини справи, яких сховище не дало повністю.

        🔴🔴 Доти «сховище дало хоч щось» означало «текст удома», і SFTP не
        торкався машини взагалі. 23.09.2026 книга ЦДІАК spr-68: у сховищі 16
        чекпоінтів на 719 сторінок із 1 308 (слоти вичерпались), решта лежала
        на диску живої машини — її знищили за шість секунд після забору, і 589
        прочитаних сторінок пропали. Машина ще жива саме тут, тож неповне
        добирається з неї: лише названі справи, не вся тека.
        """
        lacking = [(case, cs) for case, cs in zip(self.plan.cases, self.state.cases,
                                                  strict=False) if not cs.complete]
        if not lacking or self._handle is None:
            return
        short = sum(int(cs.missing_count or 0) for _, cs in lacking)
        self.state.note(
            "store_short",
            f"у сховищі неповно: {len(lacking)} справ, бракує {short} сторінок",
            action="дотягую ці справи з машини, перш ніж її гасити")
        handle = self._handle
        top = staging / "_top_up"
        import inspect

        # Однокейсовий раннер кладе справу просто в корінь робочої теки, а
        # бекенд без вибіркового забору вміє лише всю теку.
        selective = (len(self.plan.cases) > 1 and "only" in
                     inspect.signature(self.backend.fetch_outputs).parameters)
        slugs = [_slug(case.case) for case, _ in lacking] if selective else []

        # Дельтою, коли бекенд уміє: список файлів справи на машині — один
        # обмін, забір відсутнього вдома — один архів. Пофайловий SFTP через
        # океан тягнув тисячі файлів, більшість яких уже приїхала зі сховища.
        delta = (slugs and hasattr(self.backend, "case_files")
                 and hasattr(self.backend, "fetch_case_files"))

        def fetch() -> Any:
            if delta:
                got = 0
                for case, _ in lacking:
                    slug = _slug(case.case)
                    out_dir = self._out_dir_for(case)
                    need = [rel for rel, size in self.backend.case_files(handle, slug).items()
                            if not _have_file(out_dir, rel, size, flatten=case.flatten_out)]
                    got += self.backend.fetch_case_files(handle, slug, need, top / slug)
                return got
            if slugs:
                return self.backend.fetch_outputs(handle, top, only=slugs)
            return self.backend.fetch_outputs(handle, top)
        try:
            self._with_deadline(fetch, seconds=FETCH_PHASE_MAX_SEC,
                                what="дотяг неповних справ з машини")
        except (BackendError, OSError) as e:
            self.state.note("top_up_failed", str(e).splitlines()[0],
                            action="неповне лишається неповним — догін дочитає")
            return
        for name in self._SERVICE_ROOT_FILES:
            if (top / name).is_file() and not (staging / name).is_file():
                shutil.move(str(top / name), str(staging / name))
        for case, cs in lacking:
            self._place_case(case, cs, top)
        still = sum(1 for _, cs in lacking if not cs.complete)
        self._say(f"[supervise] дотяг з машини: {len(lacking) - still}/{len(lacking)} "
                  f"справ тепер повні")
        self.state.save()

    def _place_case(self, case: Any, cs: Any, staging: Path) -> None:
        """Розкласти справу зі стейджингу в її теку й поставити вирок звіркою."""
        src = staging / _slug(case.case)
        if not src.is_dir() and len(self.plan.cases) == 1:
            # Однокейсовий раннер (старіший або ручний запуск) кладе все
            # просто в /kaggle/working, без підтеки на справу.
            src = staging
        out_dir = self._out_dir_for(case)
        self._say(f"[supervise] {case.case} → {out_dir}")
        if src.is_dir():
            out_dir.mkdir(parents=True, exist_ok=True)
            _retire_foreign_model(out_dir, src, self._say)
            # 🔴 Було `for item in src.iterdir()` — тобто звірка на рівні
            # ТЕК. Аварійний забір (`best_effort`) уже створив `out/`, і
            # після переоренди фінальний забір бачив наявну теку й мовчки
            # пропускав УСІ дороблені сторінки разом зі свіжим підсумком:
            # друга оренда оплачена, робота зроблена, результат втрачено, а
            # звіт цитував старий підсумок і звинувачував аркуші.
            for item in sorted(src.rglob("*")):
                if item.is_dir():
                    continue
                target = Path(os.path.normpath(
                    out_dir / _remap(item.relative_to(src), out_dir,
                                     flatten=case.flatten_out)))
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and not _is_refreshable(target):
                    continue
                if target.exists():
                    target.unlink()
                # 🔴 `Path.replace` (os.replace) не працює МІЖ ДИСКАМИ:
                # на Windows це `WinError 17`, і забраний результат
                # лишився б у стейджингу, а справа виглядала б неповною.
                # Стейджинг живе біля gpurunner, вивід — у просторі
                # дослідження, тобто це нормальний випадок, а не екзотика.
                shutil.move(str(item), str(target))
        _stamp_case_key(out_dir, case.case_key,
                        local_dir=getattr(case, "local_dir", "") or "")
        completeness = verify_case(out_dir, expected_hint=case.n_pages,
                                   model=plan_knob_str(self.plan, "model"))
        cs.out_dir = str(out_dir)
        cs.complete = completeness.complete
        cs.missing = completeness.missing[:20]
        cs.missing_count = completeness.missing_count
        cs.detail = completeness.detail
        cs.status = "done" if completeness.complete else "incomplete"

    def _fetch_queue_tail(self, staging: Path, *, record: bool, fetch_error: str) -> None:
        """Реєстр, калібровка й службові файли після розкладки черги."""
        # 🔴 Вирок боксу `ok` не пишеться, коли забір падав: повнота на диску
        # каже про РОБОТУ, а таймаут обміну — про саму машину (тонкий канал,
        # бокс, що зникає). Змішавши їх, ми поставили б зірку машині, з якої
        # результат довелось витягати обхідним шляхом.
        if record and not fetch_error and all(c.complete for c in self.state.cases):
            self._record_box("ok", f"черга з {len(self.plan.cases)} справ повна")
            # 🔴 OOM і збої — з останнього прогресу флоту. Доти рядок `ok` завжди
            # писав нулі, тож «прогін пройшов, але флот двічі звужувався через
            # OOM» у реєстрі не відрізнявся від чистого проходу.
            last = self._last_progress or {}
            self._log_calibration(
                self._gb_per_shard() or GB_PER_SHARD_DEFAULT, ok=True,
                oom=int(last.get("oom_events_total") or last.get("oom_events") or 0),
                attempts=int(last.get("pages_failed") or 0))
        # 🔴 Спершу врятувати лог раннера — він лежить у КОРЕНІ стейджингу
        # (`_runner.log`, `_status.json`), а розкладка бере лише підтеки справ.
        # Без цього єдиний документ, де написано, ЩО саме впало на боксі,
        # завантажувався на диск і одразу стирався разом зі стейджингом, а
        # бокс уже знищено: у агента лишались інциденти без причини й нуль
        # способів її дізнатись, крім нової оренди.
        first_out = Path(self.plan.cases[0].out_dir
                         or (_fallback_out_root() / self.plan.cases[0].case))
        for name in ("_runner.log", "_status.json", "_progress.json"):
            src_file = staging / name
            if src_file.is_file():
                first_out.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_file), str(first_out / name))
                if name == "_runner.log":
                    self.state.runner_log_path = str(first_out / name)
        # Решта стейджингу не потрібна: усе, що вдалось розкласти, перенесено.
        # Без прибирання кожен захід лишав по копії результату біля gpurunner —
        # гігабайти, про які ніхто не знає.
        shutil.rmtree(staging, ignore_errors=True)
        self.state.save()

    # ---- одна справа ------------------------------------------------------

    # ---- оренда -----------------------------------------------------------

    def _find_with_patience(self, need: Need):
        """Пошук машини, що дає ЦІЛЬ, — з чеканням ринку, але без поступок.

        🔴🔴 Доти тут чекали на «машину від N ядер», а по дедлайну брали лідера
        скору з приміткою `settled_for_less`. Планка ядер при цьому не міняла
        ПОРЯДКУ машин — лише момент ринку, — тож захід щоразу закінчувався
        тим самим: слабка машина, бо ринок «не дав кращої».

        Тепер: придатна машина (обережний прогноз ≥ цілі) — береться одразу;
        немає — ринок перепитується щохвилини до `max_wait_min`, і в стані
        видно найкраще, що ринок має, і чому воно не годиться. Дедлайн вийшов —
        повертається порожній вибір, тобто `market_empty`. Чекання безкоштовне
        (оренди ще немає), слабка оренда — ні.
        """
        wait_min = float(getattr(self.plan, "max_wait_min", 0.0) or 0.0)
        if need.target_pph <= 0:
            wait_min = 0.0
        deadline = time.monotonic() + wait_min * 60
        announced = False
        while True:
            selection = self.backend.find_candidates(
                gpu=self.plan.gpu, need=need, num_gpus=self.plan.num_gpus,
                max_price=self.plan.max_price,
            )
            if not selection.empty or time.monotonic() >= deadline:
                return selection
            left_min = max(0.0, (deadline - time.monotonic()) / 60.0)
            self.state.phase = "waiting_market"
            self.state.why = (
                f"{selection.reason or 'придатних машин немає'} — чекаю ринку ще "
                f"{left_min:.0f} хв (оренди немає, чекання безкоштовне)"
            )
            self.state.save()
            if not announced:
                announced = True
                self._say(f"[supervise] ⏳ {self.state.why}")
            time.sleep(min(60.0, max(5.0, deadline - time.monotonic())))

    def _submit(self, candidate, case, need, resume, queue):
        """Оренда під замком ринку.

        Замок існує тому, що дві сесії ранжують ринок ОДНАКОВО і тягнуться до
        тієї самої найкращої пропозиції — звідси частина `400 Bad Request`
        (оффер зайняли, поки ми його ранжували). TTL мусить покривати ВЕСЬ
        сабміт разом із очікуванням SSH: коротший давав сусідній сесії право
        законно перейняти замок посеред чужої оренди, тобто відтворював саме
        ту гонку, від якої захищає.
        """
        with locks.hold("market:rent", owner=self._owner,
                        session=self.state.session, ttl_sec=self._rent_lock_ttl(),
                        note=f"оренда offer {candidate.offer.get('id')}"):
            return self.backend.submit_to_offer(
                self.job,
                self._params_for(case, need, resume=resume, queue=queue),
                gpu=self.plan.gpu,
                offer=candidate.offer,
                verify=lambda h, c, o=candidate, n=need: self._gate(h, c, o, n),
                on_created=lambda iid, off: self._note_billing_started(iid, off),
            )

    def _rent_lock_ttl(self) -> int:
        """Скільки жити замку ринку цього заходу.

        🔴 Типового TTL вистачає, поки під замком лише оренда й підйом SSH. Зі
        складом на машині туди ж потрапляє ДОСТАВКА гігабайтів, і на великій
        справі вона переживає замок: сусідня сесія законно перейме його посеред
        чужої оренди — рівно та гонка, від якої замок і стоїть. Тому при
        транспорті `box` термін рахується з обсягу, який треба привезти.
        """
        if not self._box_transport:
            return _RENT_LOCK_TTL_SEC
        total = 0
        for path in (self.plan.assets_path,
                     *(c.pages_path for c in self.plan.cases)):
            with contextlib.suppress(OSError, TypeError, ValueError):
                total += Path(str(path)).stat().st_size
        # 0.5 МБ/с — свідомо песимістично (заміряно 2-7.5 МБ/с), плюс типовий
        # TTL згори на саму оренду й ворота.
        return int(_RENT_LOCK_TTL_SEC + total / 1e6 / 0.5)

    def _note_billing_started(self, instance_id: str, offer: dict[str, Any]) -> None:
        """Інстанс створено — з цієї секунди йдуть гроші, і це має бути видно.

        🔴 Найпідступніший стан заходу: фаза `renting`, `оренд: 0`, `$0.00` — а
        на Vast уже висить машина й тарифікується. Агент читає стан, бачить
        `human_action_required: false` і чекає. Тепер тарифікація починається
        для обліку саме тут, а не після успішного сабміту.
        """
        self._pending = {"instance_id": str(instance_id),
                         "dph": offer_dph(offer),
                         "since": time.monotonic()}
        self.state.box = dict(self.state.box or {})
        self.state.box.update({
            "instance_id": str(instance_id),
            "dph_total": offer_dph(offer),
            "billing": True,
        })
        self.state.why = (
            f"інстанс {instance_id} створено (${offer_dph(offer):.3f}/год) — "
            f"ТАРИФІКАЦІЯ ЙДЕ; чекаю образ, контейнер і SSH"
        )
        # 🔴 Нарахувати ЗАРАЗ: без цього стан показував $0.00 увесь підйом
        # боксу (459 с на 8591), хоч інстанс уже тарифікувався.
        self._accrue()
        self.state.budget = self._budget_view()
        if str(instance_id) not in self.state.instances:
            self.state.instances.append(str(instance_id))
        self.state.save()
        self._say(f"[supervise] 💵 інстанс {instance_id} створено — гроші пішли "
                  f"(${offer_dph(offer):.3f}/год)")
        # 🔴🔴 ЗАМОК ВІДДАЄМО ТУТ, а не в кінці сабміту. Гонка, від якої він
        # захищає, — дві сесії тягнуться до ТОГО САМОГО оффера — вирішена рівно
        # в цю мить: `PUT /asks/` уже повернув наш інстанс. Усе подальше (хост
        # тягне образ, піднімається контейнер, чекаємо SSH, ворота заліза) — це
        # наша власна справа й нікому не заважає.
        #
        # Ціна старої поведінки виміряна 2026-08-19: три сесії на машині, замок
        # тримався ВЕСЬ цикл разом із 360-секундним очікуванням SSH, і сесія,
        # що не встигла, стояла 7.5 хв у сліпому очікуванні. Пропускна здатність
        # оренди на акаунт дорівнювала одній сесії за цикл.
        with contextlib.suppress(Exception):
            locks.release("market:rent", owner=self._owner)

    def _pending_usd(self) -> float:
        """Вартість інстансу, який ще не став робочим, але вже тарифікується."""
        p = getattr(self, "_pending", None)
        if not p:
            return 0.0
        return (time.monotonic() - p["since"]) / 3600.0 * p["dph"]

    def _wait_for_market(self, candidate) -> bool:
        """Дочекатись, поки сусідня сесія відпустить ринок. True — можна пробувати."""
        deadline = time.monotonic() + _RENT_LOCK_WAIT_SEC
        while time.monotonic() < deadline:
            time.sleep(10)
            # 🔴 Кожен тік — у СТАН, інакше зовні це не відрізнити від смерті:
            # `updated` старіє, і читач бачить «ЗАХІД ОБІРВАНО» на живому
            # наглядачі. 2026-08-19 таких хибних тривог було п'ять за захід,
            # і кожна — привід орендувати вдруге за ту саму роботу.
            holder = locks.read("market:rent")
            if holder is None:
                return True
            left = max(0, deadline - time.monotonic())
            self.state.phase = "waiting_market"
            self.state.why = (
                f"ринок тримає сусідня сесія ({holder.get('owner') or '?'}) — "
                f"чекаю ще {left / 60:.0f} хв, потім беру наступного кандидата"
            )
            self.state.save()
        self.state.note(
            "rent_locked",
            f"ринок зайнятий сусідньою сесією понад {_RENT_LOCK_WAIT_SEC // 60} хв",
            action="беру наступного кандидата",
        )
        return False

    #: Скільки готових сторінок роблять оренду «не змарнованою».
    PRODUCTIVE_PAGES = 30

    def _rent_cap_hit(self) -> bool:
        """Стеля оренд: змарновані ≥ `max_rents` або всіх ≥ 2 × `max_rents`."""
        return (self.rents_wasted >= self.plan.max_rents
                or self.rents >= 2 * self.plan.max_rents)

    def _close_rent(self) -> None:
        """Закрити облік поточної оренди: без готових сторінок — змарнована.

        🔴 Рахується лише ВІДКРИТА оренда (успішний сабміт). Невдала спроба
        рахує себе сама в `_after_failed_submit`, і доти наступний успішний
        сабміт закривав її ВДРУГЕ: «невдала + робоча» давало `rents_wasted 2`
        при `max_rents 3` (заходи spr-11652, spr-2, spr-2461, spr-1460,
        12–14.09.2026) — ще одна невдача, і захід упав би на стелі оренд.
        """
        if self._rent_open and self._box_pages_done < self.PRODUCTIVE_PAGES:
            self.rents_wasted += 1
        self._rent_open = False
        self._box_pages_done = 0
        self._box_best_usd_per_1000 = 0.0
        self._rent_base = None
        self._box_base = None
        self._reset_setup_watch()

    def _box_pace(self, progress: dict[str, Any] | None, *, now: float) -> tuple[int, float]:
        """(сторінки, секунди) ЦЬОГО боксу від його першої прочитаної сторінки.

        Рахується по черзі (справи позаду + поточна), тож перехід між справами
        не обнуляє темп. Час — від першої сторінки, а не від оренди: підйом і
        качання вже враховані вибором (`boot_sec`, накладні справи), а тут
        міряється те, що обіцяно, — темп читання черги.
        """
        pages = self._box_pages_total(progress)
        if self._box_base is None:
            if pages <= 0:
                return 0, 0.0
            self._box_base = (pages, now)
            return 0, 0.0
        base_pages, base_t = self._box_base
        return max(0, pages - base_pages), max(0.0, now - base_t)

    def _rent_the_rest(self, progress: dict[str, Any] | None) -> bool:
        """Взяти машину на недочитане. False — більше не орендувати.

        Догін — лише справи, які звірка по диску НЕ підтвердила. Коли таких
        немає (забір після загибелі машини вже все довіз), нова машина не
        потрібна: доти порожній список підмінявся ВСІМ планом, наглядач ішов
        орендувати на прочитане, упирався в стелю оренд і ставив `failed` при
        повному диску (B1, 24.09.2026).
        """
        left = self._unfinished_cases()
        if not left:
            self._say("[supervise] усі справи вже повні на диску — нова машина не потрібна")
            return False
        need = self._need_for(self._pages_left_in_queue(progress))
        return self._rent_for(left[0], need=need, queue=left, resume=True)

    def _unfinished_cases(self) -> list[CasePlan]:
        """Справи плану, які звірка по диску ще НЕ підтвердила."""
        return [c for c, cs in zip(self.plan.cases, self.state.cases, strict=False)
                if cs.status != "done"]

    def _rent_money(self, progress: dict[str, Any] | None, *, settled: bool,
                    now: float) -> tuple[float, int, float]:
        """Гроші, сторінки й секунди ЦІЄЇ оренди від миті, коли її темп дозрів.

        До дозрівання — нулі: сетап і розгін флоту не рахуються в ціну, а
        правило 3-тер без бази мовчить. Сторінки рахуються по ЧЕРЗІ (справи
        позаду за `case_index` + `pages_done` поточної), а не по справі:
        `pages_done` посправний і на кожній новій справі падає до нуля, тож
        лічильник завмирав би саме на переході — і правило читало б перехід
        як зависання.
        """
        pages = self._box_pages_total(progress, count_resumed=True)
        usd = self._current_rent_usd()
        if self._rent_base is None:
            if not settled:
                return 0.0, 0, 0.0
            self._rent_base = (usd, pages, now)
        base_usd, base_pages, base_t = self._rent_base
        return max(0.0, usd - base_usd), max(0, pages - base_pages), max(0.0, now - base_t)

    def _rent_for(
        self,
        case: CasePlan,
        *,
        resume: bool = False,
        need: Need | None = None,
        queue: list[CasePlan] | None = None,
    ) -> bool:
        """Знайти бокс і зайняти його. Кандидати перебираються по черзі."""
        # 🔴 Стеля на ЗАХІД, а не на пошук. `max_attempts` обмежує перебір
        # кандидатів усередині одного виклику; цей лічильник — скільки разів
        # захід узагалі брав машину. Без нього 2026-08-11 вийшло чотири оренди
        # за п'ять хвилин: кожна невдача вела до нового пошуку, і жодне
        # обмеження цього не бачило.
        if self._rent_cap_hit():
            self.state.finish(
                "failed",
                f"вичерпано стелю оренд ({self.rents_wasted} змарнованих із "
                f"{self.plan.max_rents}, усього {self.rents}) — далі не беру "
                f"машин самостійно",
                human_action=(
                    "подивитись `htr state --json` → `incidents`: якщо це серія "
                    "мертвих хостів, підняти `max_rents` у плані; якщо та сама "
                    "помилка щоразу — вона не в машині"
                ),
            )
            return False

        need = need or self._need_for(case.n_pages)
        # Потреба, під яку орендуємо: проти її цілі наглядач звірятиме факт.
        self._rent_need = need
        self._rent_queue = list(queue or [case])
        self.cfg = dataclasses.replace(
            self.cfg, target_pph=queue_pace(need.target_here, self._rent_queue))
        self._gate_rejects = []
        selection = self._find_with_patience(need)
        if selection.empty:
            # 🔴 «Ринок порожній» і «мої стелі відсікли всіх» — різні діагнози,
            # і доти вони приходили одним рядком. Шість заходів поспіль
            # (31.08.2026, ніч) закрились як `market_empty`, тоді як машини
            # були: їх відсікала `--max-hours 3` («GTX 1080 Ti · $0.074/год —
            # ✗ 4.4 год > 3.0»). Після підняття стелі до 8 год той самий бокс
            # дав $0.117 за 1000 сторінок — дешевше за всі денні RTX 3090.
            why, action = self._explain_empty_market(selection)
            self.state.finish("market_empty", why, human_action=action)
            for rejected in selection.rejected[:3]:
                self.state.note("market", rejected.explain)
            return False

        self._warn_pph_floor(selection, need)

        # 🔴 Дедуп по МАШИНІ, а не по офферу. 2026-08-11 дві спроби поспіль
        # пішли на ту саму machine 95392: `ssh_unreachable` має два удари до
        # бану, тож перша невдача її не відсіювала, а вердикти в межах циклу
        # не перечитуються. Плюс пропускаємо все, що вже впало в ЦЬОМУ заході.
        seen_machines: set[int] = set()
        shortlist: list[ScoredOffer] = []
        for candidate in selection.candidates:
            mid = candidate.machine_id
            if mid in seen_machines or mid in self._failed_machines:
                continue
            seen_machines.add(mid)
            shortlist.append(candidate)
            if len(shortlist) >= self.plan.max_attempts:
                break

        for attempt, candidate in enumerate(shortlist, 1):
            if self._rent_cap_hit():
                # Стеля мусить діяти і всередині перебору: кожен кандидат, що
                # впав після `PUT`, — це створений і оплачений інстанс.
                self.state.note(
                    "rent_cap", f"стеля {self.plan.max_rents} оренд вичерпана "
                                f"на {attempt - 1}-му кандидаті",
                    action="перебір спинено",
                )
                break
            self.state.phase = "renting"
            self.state.why = f"{case.case}: {candidate.explain}"
            self.state.box = self._box_view(candidate, selection)
            self.state.save()
            self._say(f"[supervise] спроба {attempt}: {candidate.explain}")

            t0 = time.monotonic()
            self._host_ready_at = None
            try:
                handle = self._submit(candidate, case, need, resume, queue)
            except locks.LockBusy as e:
                # 🔴 `locks.hold` кидає LockBusy(RuntimeError), а тут ловився
                # лише BackendError — тож замок, зроблений ЗАРАДИ паралельних
                # сесій, сам валив захід із `verdict: failed`. Це нормальна
                # ситуація: сусідня сесія саме зараз орендує.
                self.state.note("rent_locked", str(e).splitlines()[0],
                                action="чекаю й пробую цього ж кандидата ще раз")
                if not self._wait_for_market(candidate):
                    continue
                try:
                    handle = self._submit(candidate, case, need, resume, queue)
                except locks.LockBusy as e2:
                    # Замок перехопила третя сесія — це ринок, не машина.
                    self.state.note("rent_locked", str(e2).splitlines()[0],
                                    action="беру наступного кандидата")
                    continue
                except BackendError as e2:
                    # 🔴 Той самий обробник, що й на першій спробі. Власний
                    # `except` тут лишав створений інстанс горіти (див.
                    # `_after_failed_submit`).
                    verdict = self._after_failed_submit(e2, candidate, t0)
                    if verdict == "stop":
                        return False
                    if verdict == "retry":
                        shortlist.insert(attempt, candidate)
                    continue
            except BackendError as e:
                verdict = self._after_failed_submit(e, candidate, t0)
                if verdict == "stop":
                    return False
                if verdict == "retry":
                    shortlist.insert(attempt, candidate)
                continue
            self._handle = handle
            self._offer = candidate.offer
            self._rented_at = time.monotonic()
            self._reset_setup_watch()
            self._taken_streak = 0
            # 🔴🔴 Замір належить КОНКРЕТНІЙ машині. Лічильник заводився раз на
            # захід і лише зростав, тож після переоренди темп швидкого боксу
            # (2548 стор/год) записувався в реєстр ПОВІЛЬНОМУ, що доробляв
            # чергу (700-980). Далі `score_offer` бере цей «замір» замість
            # моделі, а `starred()` шукає таку машину адресно й першою — тобто
            # вада липка: вона в git-versioned реєстрі й діє на всі наступні
            # заходи. Скидаємо разом з орендою.
            self._measured_pph = 0.0
            # Час підйому вже оплачений: переносимо його в закриті витрати, бо
            # `_current_rent_usd()` рахує від ЦІЄЇ миті. Інакше хвилини, за які
            # хост тягнув образ, зникли б з обліку.
            self.settled_usd += self._pending_usd()
            self._pending = None
            self._close_rent()
            self.rents += 1
            self._rent_open = True
            # Скільки хост піднімався. 🔴 Єдиний доступний нам сигнал про
            # закешованість образу: Vast у картці оффера про це не каже нічого
            # (перевірено — 100 полів, жодного про образи). Машина, що
            # піднялась за 40 с, майже напевно має наш образ у кеші; та, що
            # тягла його 3 хвилини, наступного разу тягтиме знову.
            # 🔴🔴 Підйом ХОСТА окремо від НАШОЇ доставки. Доти `boot_sec`
            # міряв усю здачу, і 23.09.2026 25 хвилин заливки 48-МБ `job.py`
            # лягли в реєстр як «хост піднімається 27 хвилин» — а вибір потім
            # штрафував за це машину (`predict_hours(boot_sec=…)`).
            total = time.monotonic() - t0
            ready = self._host_ready_at
            self._boot_sec = round((ready - t0) if ready else total, 1)
            self._upload_sec = round(max(0.0, total - self._boot_sec), 1)
            self._book_spend(handle, candidate.cost)
            self._say(f"[supervise] бокс піднявся за {self._boot_sec:.0f} с"
                      + (f", доставка наших даних {self._upload_sec:.0f} с"
                         if self._upload_sec >= 1 else ""))
            self.state.box = self._box_view(candidate, selection, probe=self._probe)
            # 🔴 Оренда позаду — фаза мусить це сказати. Доти `running` ставило
            # лише підхоплення чужого живого боксу, і весь звичайний захід
            # (8591: 28 хв роботи) стан показував «renting».
            self.state.phase = "setup"
            self.state.save()
            return True

        # 🔴 Порада «подивитись boxes ls» стояла тут БЕЗУМОВНО і вела хибним
        # слідом щоразу, коли причина зовнішня. 04.09.2026 вона відправила
        # перевіряти реєстр банів при `vast · 0.00 $`. Тепер порада йде від
        # того, що насправді відсіяло, а на серії `offer_taken` перевіряються
        # гроші — ринок не буває зайнятий чотири рази поспіль.
        tried = min(len(shortlist), self.plan.max_attempts)
        kinds = Counter(self._gate_rejects)
        if kinds.get("offer_taken", 0) >= 2 and self._stop_if_broke():
            return False
        detail = ", ".join(f"{k}×{n}" for k, n in kinds.most_common(4))
        self.state.finish(
            "market_empty",
            f"жоден із {tried} кандидатів не пройшов ворота заліза"
            + (f" ({detail})" if detail else ""),
            human_action=self._action_for_rejects(kinds),
        )
        return False

    def _gate(
        self, handle: JobHandle, client: Any, candidate: ScoredOffer, need: Need
    ) -> dict[str, Any]:
        """Проба заліза + ворота. Викликається ПЕРЕД заливкою даних.

        Повертає перекриття параметрів: число шардів рахується з ВИМІРЯНОЇ
        вільної VRAM, а не з картки оффера.
        """
        # 🔴 При складі на самій машині канал хоста до інтернету не важить
        # узагалі: дані везе наглядач по SSH, а не машина по HTTP. Міряти його
        # посиланням, якого ще не існує, означало б забракувати здорову машину
        # за те, чим ми не користуємось.
        # Мить, коли ХОСТ готовий (SSH є, проба йде) — межа між його підйомом
        # і нашою доставкою. Реєстр пам'ятає лише перше: друге — наша провина.
        self._host_ready_at = time.monotonic()
        probe = self.backend.probe_box(
            handle, net_probe_url="" if self._box_transport else self.plan.assets_url)
        self._probe = probe
        result = gate_mod.evaluate(probe, candidate.offer, need)
        self._gate_sizing = result.sizing
        self._say(f"[supervise] проба: {result.detail}")
        if not result.ok:
            raise GateRejected(result)
        if result.sizing is None:
            return {}
        shards = result.sizing.shards
        # 🔴🔴 ЯВНО ЗАДАНЕ КОРИСТУВАЧЕМ НЕ ПЕРЕКРИВАЄМО. Ворота
        # повертали своє число БЕЗУМОВНО, тож `-p shards=5` доходив до раннера
        # й там-таки затирався пробою боксу: на сповідках просили 5, дістали 8,
        # VRAM 23.0 з 24.6 — і 84% сторінок у збоях. Проба знає про залізо, але
        # НЕ знає про щільність справи; дослідник, який уже бачив пік 4.3 ГБ на
        # шард, знає більше. Ми лишаємо за собою тільки СТЕЛЮ: більше, ніж
        # фізично влазить, не дамо.
        wanted = self._requested_shards()
        if wanted:
            if wanted > shards:
                self.state.note(
                    "shards_capped",
                    f"просили {wanted} шардів, залізо тримає {shards} — беру {shards}",
                    action="це стеля VRAM, а не наша думка",
                )
            else:
                shards = wanted
        overrides: dict[str, Any] = {
            "shards": shards,
            "threads_per_shard": result.sizing.threads_per_shard,
        }
        if not wanted:
            # 🔴🔴 СТЕЛЯ регулятора — від ЯДЕР і фізичної пам'яті, а не від
            # стартового числа. Доти раннер діставав `shards_max = shards`, тож
            # `active == ceiling` з першої секунди, і регулятор, який уміє рости
            # за ВИМІРЯНИМ зайняттям карти, не ріс ніколи: spr-66 (23.09, RTX
            # 3090, 61 ядро) простояв на 7 шардах при 6.2 з 24 ГБ карти.
            # Вгору флот іде лише за заміром темпу й пам'яті; на OOM і тиску
            # карти раннер стискає його сам.
            ceiling = fleet_ceiling(result.measured, shards)
            self._fleet_ceiling = ceiling
            overrides["shards_max"] = ceiling
            overrides["threads_per_shard"] = max(
                1, min(result.sizing.threads_per_shard,
                       int(float(result.measured.get("cores_eff") or 0)
                           // max(1, ceiling)) or 1))
        if self._box_transport:
            # Ворота пройдено — машина наша, і тільки тепер є сенс везти на неї
            # гігабайти. Порядок саме такий: доставка до воріт означала б
            # платити заливкою за кожну забраковану машину. Посилання при цьому
            # НЕ чіпаємо: вони відомі наперед (порт і шляхи складу фіксовані) і
            # вже пройшли перевірку параметрів роботи — а вона робиться до
            # оренди, тобто задовго до цієї хвилини.
            self._deliver_to_box(handle, client)
        return overrides

    @property
    def _box_transport(self) -> bool:
        return str(getattr(self.plan, "transport", "r2") or "r2") == "box"

    def _deliver_to_box(self, handle: JobHandle, client: Any) -> None:
        """Привезти дані на машину й підняти на ній склад.

        Після цього посилання, які вже лежать у параметрах роботи, починають
        працювати: вони дивляться на петлю самої машини, тож раннер не
        змінюється жодним рядком — для нього це такий самий `curl`, як і по
        presigned-посиланню.

        🔴 Порядок обернуто: склад піднімається ПЕРШИМ, і дані їдуть у нього
        `PUT`. Доти першим ішов `scp`, а склад — після; заміряно на боксі
        20.09.2026 (150 МБ): `scp` 7.56 МБ/с проти 20.07 у HTTP. На архіві
        2.25 ГБ це різниця між п'ятьма хвилинами й чвертю години оплаченого
        простою, і та чверть години йде ДО першої прочитаної сторінки.
        """
        from gpurunner.htr import box_transport as bt

        ep = bt.endpoint_of(self.backend, handle)
        known = _staging_root() / "_known_hosts"
        paths = bt.remote_paths(self.plan.cases[0].case)
        self._exec_on(client, f"mkdir -p {posixpath.dirname(paths['pages'])}")
        inside = bt.start_origin(
            lambda cmd: self._exec_on(client, cmd),
            lambda path, text: self.backend._upload_text(client, path, text),
            token=self._origin_token)
        self._say(f"[supervise] склад на машині: {inside}")
        outside = self._origin_outside_url(handle)
        t0 = time.time()
        # 🔴 Архів ассетів їде ПЕРШИМ і служить мірилом каналу. Проба каналу
        # воріт при цьому транспорті не працює — вона міряє шлях ХОСТА до
        # бакета, а веземо ми, — тож єдине число про НАШ аплінк до цієї машини
        # ми дістаємо саме тут. Без нього повільний канал обертається годинами
        # оплаченого простою: 8 ГБ на 0.5 МБ/с — це 4.5 год, за які машина не
        # прочитає жодної сторінки.
        deliver = self._deliverer(outside, ep, known, client)
        sent = deliver(Path(self.plan.assets_path), "assets.tgz", paths["assets"])
        self._gate_delivery_speed(sent, time.time() - t0,
                                  left_bytes=self._pages_bytes_left())
        for case in self.plan.cases:
            sent += deliver(
                Path(case.pages_path), f"cases/{case.case}.tar",
                posixpath.join(bt.REMOTE_ROOT, "cases", f"{case.case}.tar"))
        took = max(0.001, time.time() - t0)
        self.state.note(
            "delivered",
            f"привезено {sent / 1e6:.0f} МБ за {took:.0f} с "
            f"({sent / 1e6 / took:.1f} МБ/с) — "
            f"{'HTTP у склад' if outside else 'scp (склад знадвору не відповів)'}")
        self._restore_checkpoints(ep, known)

    def _origin_outside_url(self, handle: JobHandle) -> str:
        """Зовнішня адреса складу — і ЛИШЕ якщо вона справді відповідає.

        🔴 Два окремі питання, і жодне не можна вгадати. Чи Vast прокинув порт
        назовні — читається з інстансу (зовнішній порт випадковий, він не
        дорівнює внутрішньому). Чи він досяжний ЗВІДСИ — перевіряється
        запитом: між нами й боксом бувають і фаєрвол хоста, і NAT, і просто
        інший дата-центр. Здогад тут коштував би найдорожчого: доставка
        «почалась» і зависла на оплачуваній машині.
        """
        from gpurunner.htr import box_transport as bt

        self._origin_outside = ""
        fn = getattr(self.backend, "http_endpoint", None)
        if not callable(fn):
            return ""
        try:
            raw = fn(handle, bt.ORIGIN_PORT) or {}
        except Exception as exc:
            self.state.note("origin_port_unknown", f"{type(exc).__name__}: {exc}")
            return ""
        host, port = str(raw.get("host") or ""), int(raw.get("port") or 0)
        if not host or not port:
            self.state.note(
                "origin_port_missing",
                "машина не прокинула порт складу назовні",
                action="дані поїдуть `scp` — це 7.56 МБ/с проти 20.07 у HTTP")
            return ""
        base = bt.base_url(port, host=host, token=self._origin_token)
        if not bt.reachable(base):
            self.state.note(
                "origin_unreachable",
                f"порт складу {host}:{port} прокинуто, але звідси він мовчить",
                action="дані поїдуть `scp`; склад лишається робочим ДЛЯ РАННЕРА "
                       "(він ходить на петлю машини)")
            return ""
        self._origin_outside = base
        # 🔴 У стан, а не лише в нотатку: саме звідси рятунок дістане адресу,
        # якщо наглядач помре після читання й до забору. Нотатка — для людини,
        # поле — для машини.
        self.state.origin_base = base
        self.state.save()
        self.state.note("origin_outside",
                        f"склад доступний знадвору: {host}:{port}")
        return base

    def _deliverer(self, outside: str, ep: Any, known: Path, client: Any) -> Any:
        """Одна функція доставки, вибрана ОДИН раз і на всі файли.

        🔴 Вибір плеча посеред черги був би найгіршим із можливих: половина
        архівів приїхала б одним темпом, половина іншим, а `delivery_rate` —
        число, за яким наглядач бракує машину, — міряв би середнє двох різних
        каналів і брехав би в обидві сторони.
        """
        from gpurunner.htr import box_transport as bt

        if outside:
            # 🔴 ВІСІМ потоків, а не один. Заміряно 23.09.2026: домашній канал
            # одним потоком 3.83 МБ/с, вісьмома 8.94; а перша жива доставка
            # 135 МБ у Корею одним потоком дала 1.2 МБ/с — усемеро менше за
            # можливе. Одне вікно TCP на великому колі просто мовчить, поки
            # летить підтвердження.
            def assemble(rel: str, names: list[str]) -> None:
                """Склеїти шматки НА МАШИНІ: одна команда на файл."""
                root = bt.REMOTE_ROOT
                target = shlex.quote(f"{root}/{rel}")
                pieces = " ".join(shlex.quote(f"{root}/{n}") for n in names)
                self._exec_on(
                    client,
                    f"cat {pieces} > {target} && rm -f {pieces}")

            return lambda local, rel, _remote: bt.push_http_parallel(
                outside, local, rel, assemble=assemble)

        check = lambda cmd: self._exec_on(client, cmd)  # noqa: E731 — один вираз
        return lambda local, _rel, remote: bt.push(
            ep, local, remote, known_hosts=known, verify_with=check)

    def _restore_checkpoints(self, ep: Any, known: Path) -> None:
        """Повернути на машину чекпоінти, які ми забрали додому.

        🔴 Без цього переоренда читає справу З НУЛЯ. Раннер бере точки
        відновлення тим самим `curl` зі складу — а склад на НОВІЙ машині
        порожній, хоч удома лежить усе прочитане до її смерті. Саме заради
        таких хвилин чекпоінти й возяться додому.
        """
        from gpurunner.htr import box_transport as bt

        home = self._box_ckpt_dir()
        balls = sorted(home.rglob("ckpt_*.tgz")) if home.is_dir() else []
        if not balls:
            return
        sent = 0
        for ball in balls:
            rel = ball.relative_to(home).as_posix()
            remote = posixpath.join(bt.REMOTE_ROOT, "ckpt", rel)
            try:
                if self._origin_outside:
                    # Склад сам заведе теку під об'єкт — SSH тут не потрібен
                    # узагалі, а рукостискання на кожен чекпоінт коштувало
                    # більше за сам чекпоінт.
                    sent += bt.push_http(
                        self._origin_outside, ball,
                        posixpath.join("ckpt", rel))
                else:
                    self._exec_on_box(ep, f"mkdir -p {posixpath.dirname(remote)}")
                    sent += bt.push(ep, ball, remote, known_hosts=known)
            except Exception as exc:
                self.state.note("ckpt_restore_failed", f"{ball.name}: {exc}",
                                action="справа читатиметься з місця, до якого "
                                       "чекпоінти таки доїхали")
                break
        if sent:
            self.state.note("ckpt_restored",
                            f"повернуто на машину {len(balls)} чекпоінтів "
                            f"({sent / 1e6:.0f} МБ) — читання піде з місця, "
                            f"а не з нуля")

    def _transfer_ceiling(self, nbytes: int, *, slack: float = 3.0) -> int:
        """Скільки секунд давати переносу такого обсягу ЦИМ каналом.

        🔴 Стала тут працювала проти нас в обидві сторони. `timeout=1800` на
        швидкому каналі означав пів години на мертвому сокеті — тобто пів
        години оплаченої оренди після того, як усе вже зрозуміло; на повільному
        вона різала живий перенос посередині. Канал — єдиний вхід, що гуляє на
        два з половиною порядки, тож стеля, однакова для 0.004 і 20 МБ/с, не
        може бути правильною для обох.

        Тепер це обсяг, поділений на ЗАМІРЯНИЙ канал, із запасом. Заміру ще
        немає (перший перенос заходу) — беремо песимістичну підлогу, як і
        раніше; інакше ми б вимагали виміру від того, чим його й міряють.
        """
        rate = self._uplink_mbs if self._uplink_mbs > 0 else _SLOW_CHANNEL_MBS
        need = (nbytes / 1e6) / max(0.01, rate) * max(1.0, slack)
        return int(min(TRANSFER_CEILING_MAX_SEC,
                       max(TRANSFER_CEILING_MIN_SEC, need)))

    def _pages_bytes_left(self) -> int:
        """Скільки байтів кадрів ще треба привезти на машину."""
        total = 0
        for case in self.plan.cases:
            with contextlib.suppress(OSError, TypeError, ValueError):
                total += Path(str(case.pages_path)).stat().st_size
        return total

    def _gate_delivery_speed(self, sent_bytes: int, took_sec: float,
                             left_bytes: int | None = None) -> None:
        """Чи встигне решта доїхати за розумний час — за щойно заміряним каналом.

        🔴 Машину бракуємо ДО того, як витратити на неї години: наглядач візьме
        наступного кандидата, а не стоятиме з нею. Поріг не абсолютний
        (повільний канал — не вада машини), а відносний до стелі часу заходу:
        доставка не має з'їдати більшу її частину, ніж `DELIVERY_MAX_SHARE`.
        """
        rate = sent_bytes / 1e6 / max(0.001, took_sec)
        self._uplink_mbs = rate
        self._uplink_via = "http" if self._origin_outside else "scp"
        left = self._pages_bytes_left() if left_bytes is None else left_bytes
        eta_h = (left / 1e6 / max(0.01, rate)) / 3600.0
        budget_h = float(self.plan.max_hours or 0) * DELIVERY_MAX_SHARE
        self.state.note(
            "delivery_rate",
            f"канал до машини {rate:.1f} МБ/с; решта {left / 1e6:.0f} МБ — "
            f"це ~{eta_h * 60:.0f} хв")
        if budget_h > 0 and eta_h > budget_h:
            raise BackendError(
                f"канал до цієї машини {rate:.1f} МБ/с: решта даних "
                f"({left / 1e6:.0f} МБ) їхала б ~{eta_h:.1f} год при стелі "
                f"заходу {self.plan.max_hours:.0f} год. Це оплачений простій — "
                f"беру іншу машину.")

    def _exec_on_box(self, ep: Any, cmd: str) -> None:
        """Команда на машині окремою сесією.

        ⚠ Дорожче, ніж здається: з'єднання, яким щойно везли файли, ще живе, і
        кожен виклик тут — зайве рукостискання на оплачуваній машині. Лишено
        навмисно вузьким (створення тек під чекпоінти при поверненні), щоб не
        тягнути `client` крізь half a dozen викликів; якщо чекпоінтів стане
        багато, правильний хід — передати сюди відкритий `client`.
        """
        if self._handle is None:      # pragma: no cover — сюди не дійти
            return
        client = self.backend._ssh(self._handle, timeout=30)
        try:
            self._exec_on(client, cmd)
        finally:
            client.close()

    @staticmethod
    def _exec_on(client: Any, cmd: str) -> str:
        """Команда на машині тим самим з'єднанням, яким її перевіряли ворота."""
        _, stdout, stderr = client.exec_command(cmd, timeout=180)
        rc = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        if rc != 0:
            err = stderr.read().decode("utf-8", errors="replace")
            raise BackendError(f"команда на машині впала ({rc}): {cmd}\n{err}")
        return out

    def _requested_shards(self) -> int:
        """Скільки шардів попросив дослідник (0 — не просив).

        Береться з тих самих місць, що й `vram_gb_per_shard`: параметри плану
        або конкретної справи.
        """
        for src in (self.plan.params, *(c.params for c in self.plan.cases)):
            value = (src or {}).get("shards")
            if isinstance(value, str) and value.strip().lower() in ("auto", ""):
                continue
            if value:
                try:
                    return max(1, int(value))
                except (TypeError, ValueError):
                    continue
        # 🔴 Верхній рівень плану. Читався ЛИШЕ `params`, тож `"shards": 6` у
        # плані не діяв і про це не казав — та сама вада, що з `gb_per_shard`.
        return max(0, int(getattr(self.plan, "shards", 0) or 0))

    def _case_urls(self, case: CasePlan) -> dict[str, Any]:
        """Посилання справи: з плану (бакет) або складу на машині.

        🔴 При складі вони відомі НАПЕРЕД — порт і шляхи фіксовані, — і саме
        тому підставляються тут, а не після підйому машини: параметри роботи
        перевіряються до оренди, і порожній `pages_url` там законно вважається
        обірваною змінною й валить захід (що й сталось на першій пробі).
        """
        if not self._box_transport:
            # 🔴🔴 Слоти — від РОЗМІРУ справи, і тут, а не лише в складачі плану.
            # c25ad1f виправив формулу для НОВИХ планів, а догін на старому
            # плані (23.09.2026, plan_seed.json) однаково віз 314 слотів на
            # справу × 194 справи = 121 832 посилання, `job.py` на 48 МБ і
            # 25 хвилин оплаченої заливки до першої сторінки.
            from gpurunner.htr.plan_build import ckpt_urls_for

            keep = ckpt_urls_for(self.plan.max_hours, pages=int(case.n_pages or 0))
            return {"pages_url": case.pages_url,
                    "ckpt_urls": list(case.ckpt_urls or [])[:keep],
                    "resume_urls": list(case.resume_urls or [])[:keep],
                    "service_put_url": case.service_put_url}
        from gpurunner.htr import box_transport as bt

        return bt.urls_for(case.case, case.ckpt_prefix or f"ckpt/{case.case}",
                           case.ckpt_slots or 60, token=self._origin_token)

    def _assets_url(self) -> str:
        if not self._box_transport:
            return self.plan.assets_url
        from gpurunner.htr import box_transport as bt

        return f"{bt.base_url(token=self._origin_token)}/assets.tgz"

    def _params_for(self, case: CasePlan, need: Need, *, resume: bool,
                    queue: list[CasePlan] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "assets_url": self._assets_url(),
            **self._case_urls(case),
            "case": case.case,
            "estimated_n_pages": case.n_pages,
            "min_net_mbps": 0,   # канал уже заміряно воротами, удруге не платимо
            "disk": self.plan.disk_gb,
            # 🔴 `int()` ЗРІЗАЄ: при `--max-hours 1.9` контейнерна стеля ставала
            # годиною, і job убивали за 54 хвилини до того, як наглядач узагалі
            # збереться щось робити. Стеля на боксі мусить бути НЕ МЕНШОЮ за
            # нашу власну, інакше вона працює проти нас.
            "max_hours": max(1, math.ceil(self.plan.max_hours)),
            "catchup_passes": self.plan.catchup_passes,
            # 🔴 Сторож «робота скінчилась, а по результат ніхто не прийшов».
            # Наглядач не передавав цього НІКОЛИ, тож блок самознищення просто
            # не встановлювався, і після завершення job'а бокс горів до
            # жорсткого дедлайну: 2026-08-12 це коштувало 4.44 год і $0.93.
            "autodestroy_hours": self.plan.autodestroy_hours,
            "heartbeat_put_url": self.plan.heartbeat_put_url,
            "shards": "auto",
            "keep_warm_sec": int(self.plan.keep_warm_min * 60),
            "scripts_sha256": dict(self.plan.scripts_sha256 or {}),
            # ключ верхнього рівня плану мусить доходити до бекенда (він читає
            # `params`), інакше `num_gpus: 2` у плані не діяв узагалі
            "num_gpus": int(self.plan.num_gpus or 1),
        }
        if self._gb_per_shard():
            params["vram_gb_per_shard"] = self._gb_per_shard()
        if queue:
            # Черга їде В ОДНОМУ job'і: бокс піднімається раз, а справи
            # змінюються всередині раннера.
            params["cases"] = [
                {
                    "case": c.case,
                    "estimated_n_pages": c.n_pages,
                    **self._case_urls(c),
                    # 🔴🔴 Посилання на відновлення передаються ЗАВЖДИ, коли
                    # вони є. Тут стояло `if resume else []`, а `resume=True`
                    # ставиться лише при переоренді ВСЕРЕДИНІ того самого
                    # заходу — тож новий захід ігнорував чекпоінти повністю. А
                    # `htr_cloud_plan.py` завжди робить чергу, тобто відновлення
                    # не працювало на практиці НІКОЛИ. Ціна одного такого
                    # перезапуску (ДАВО, ольгопільська кампанія 2026-08-12):
                    # 769 сторінок наново, ~45 хв і ~$0.13 — при тому, що
                    # передполітна перевірка чесно казала «точки відновлення на
                    # місці». Вони й були на місці; їх просто не брали.
                    # Зайвим це не буває: якщо чекпоінта ще немає, раннер
                    # дістає 404 (rc=22) і мовчки йде далі. Самі посилання —
                    # у `_case_urls` вище: при складі на машині вони інші.
                    # 🎛 Ключ матеріалу для пам'яті флоту між томами: бокс
                    # переносить виміряне на наступний том, лише коли кадри
                    # схожі за площею (VRAM шарда росте з неї).
                    "frame_mpx_p95": float(getattr(c, "frame_mpx_p95", 0) or 0),
                    **c.params,
                }
                for c in queue
            ]
        params.update(self.plan.params)
        params.update(case.params)
        return params


    # ---- забір і звірка ---------------------------------------------------

    # ⚠ Тут був `_fetch_and_verify` — забір ОДНІЄЇ справи з наступною
    # звіркою. З переходом на чергу його місце зайняв `_fetch_queue`
    # (забір усього одним заходом), і метод лишився без жодного
    # викликача: мертвий код у наглядачі читається як запасний шлях,
    # якого насправді немає.

    # ---- бокс -------------------------------------------------------------

    def _rescue_before_destroy(self) -> None:
        """Остання спроба забрати те, що є, перш ніж машина зникне назавжди.

        Свідомо best-effort і свідомо мовчазна на помилках: ми вже в аварійній
        гілці, і другий виняток тут лише завадив би гасінню — тобто лишив би
        бокс горіти.
        """
        if self._handle is None:
            return
        try:
            # 🔴 Було `_fetch_and_verify` ПЕРШОЇ незавершеної справи: воно тягне
            # все `/kaggle/working` — тобто тексти ВСІХ справ черги — у теку
            # однієї, а `verify_case` рахує `rglob("*.txt")` по всьому дереву.
            # Знаменник першої справи перемішувався з чужими текстами. Черговий
            # забір розкладає по справах правильно й так само best-effort.
            self._fetch_queue(best_effort=True)
        except Exception as e:
            self.state.note("rescue_failed", f"{type(e).__name__}: {e}",
                            action="результат лишається лише в чекпоінтах R2")

    def _destroy(self, why: str) -> None:
        """Погасити оренду. Це єдине, що спиняє лічильник."""
        if self._handle is None:
            return
        # 🔴 Спершу зафіксувати витрачене, ПОТІМ гасити: нижче занулюється
        # `self._offer`, і разом із ним зникла б ціна цієї оренди.
        # 🔴🔴 І поточну оренду обнулити В ТУ Ж МИТЬ, а не після `cancel`. Потік
        # серцебиття (`_beat`) кличе `_accrue()` незалежно, і за секунди
        # мережевого гасіння рахував `settled` (уже з цією орендою) + ту саму
        # оренду вдруге. Фініш читав саме це число: «9 справ прогнано повністю
        # за $0.58» при $0.29 у бюджеті (spr-2461, 13.09.2026), $0.54 при $0.285
        # (cdiak1040-1-2, 14.09.2026).
        rent_usd = self._current_rent_usd()
        rented_at, self._rented_at = self._rented_at, None
        self.settled_usd += rent_usd
        self.spent_usd = self.settled_usd
        self._settle_spend(self._handle, rent_usd)
        try:
            self.backend.cancel(self._handle)
            self._say(f"[supervise] 🔥 оренду знищено ({why})")
        except BackendError as e:
            # 🔴 Тут ручка від інстансу занулювалась БЕЗУМОВНО — тобто після
            # невдалого гасіння повторити його було вже нічим, а бокс далі
            # тарифікувався. Вердикт при цьому міг лишитись `ok`. Тепер ручка
            # зберігається, а захід стає термінальним і кличе людину: єдиний
            # обмежувач, що лишився, — дедлайн-сторож у самому контейнері.
            self.state.note(
                "destroy_failed", f"{e}",
                action=f"ГАСИТИ РУКАМИ: gpurunner cancel {self._handle.id[:8]}",
            )
            self._destroy_failed = True
            # Бокс живий і тарифікується далі: лічильник мусить іти від миті,
            # коли ми перестали його рахувати, а не зникнути.
            if rented_at is not None:
                self.settled_usd -= rent_usd
                self._rented_at = rented_at
            self.state.finish(
                "incomplete",
                f"бокс НЕ погашено і далі тарифікується: {str(e).splitlines()[0]}",
                sticky=True,
                human_action=(
                    f"негайно: `gpurunner cancel {self._handle.id[:8]}` (або "
                    f"інстанс {self._handle.remote_id} у веб-консолі Vast). "
                    f"Автоматика вичерпала способи гасіння."
                ),
            )
            self._offer = None
            return
        try:
            self._handle.status = JobStatus.CANCELLED
            manifest.update(self._handle)
        except Exception:
            pass
        self._handle = None
        self._offer = None

    #: Порада на кожну причину відсіву. 🔴 Ключове тут — чого в таблиці НЕМАЄ:
    #: поради «подивитись boxes ls». Вона доречна рівно тоді, коли кандидатів
    #: справді зняв реєстр, а не завжди, як було доти.
    _ACTION_BY_REJECT: ClassVar[dict[str, str]] = {
        "slow_net": "канал хостів повільний — спробувати пізніше або взяти "
                    "машину ближче до бакета",
        "card_busy": "карти зайняті чужими процесами — спробувати пізніше",
        "disk_short": "опустити `disk_gb` у плані: вимога відсікає більшість "
                      "машин, а реально треба ~кадри×2+5 ГБ",
        "overpriced": "підняти `--max-usd-per-1000` або `max_price`, якщо ця "
                      "ціна влаштовує",
        "our_bug": "це НАШ бік: перевірити посилання плану "
                   "(`gpurunner htr preflight`) — інші машини не допоможуть",
        "ssh_auth_denied": "перевірити SSH-ключ (`gpurunner auth vast --verify`)",
        "offer_taken": "ринок розібрав кандидатів; якщо це повторюється — "
                       "спершу `gpurunner balance`, а не реєстр банів",
        "market_busy": "Vast не приймає оренду (429/5xx) — спробувати пізніше",
    }

    def _action_for_rejects(self, kinds: Counter) -> str:
        """Порада від ТОГО, що насправді відсіяло більшість кандидатів."""
        if not kinds:
            return ("подивитись `htr state --json` → `incidents`: причина не "
                    "класифікована")
        top, _ = kinds.most_common(1)[0]
        if top == "banned":
            return "подивитись `gpurunner boxes ls` — забанено надто багато"
        return self._ACTION_BY_REJECT.get(
            top, f"подивитись `htr state --json` → `incidents` (переважає {top})")

    @staticmethod
    def _explain_empty_market(selection: Any) -> tuple[str, str]:
        """Чому ринок «порожній»: немає машин чи наші стелі відсікли всіх.

        Читається з `rejects` відкинутих кандидатів — там уже лежить причина
        по кожному, і доти вона просто не доходила до вироку.
        """
        reasons: Counter = Counter()
        for rejected in getattr(selection, "rejected", []) or []:
            for why in getattr(rejected, "rejects", []) or []:
                # 🔴 Підлога темпу перевіряється ПЕРШОЮ: її причина звучить
                # «… стор/год < підлоги 1000», тобто містить «год» і без цього
                # рядка класифікувалась як стеля ГОДИН — а порада «підніміть
                # --max-hours» на ній не робить нічого взагалі.
                key = ("ціль темпу" if "цілі" in why else
                       "підлога темпу" if "підлоги" in why else
                       "стеля годин" if "год" in why else
                       "ціна" if "$" in why else
                       "диск" if "ГБ" in why and "диск" in why.lower() else
                       why.split(":")[0][:24])
                reasons[key] += 1
        if not reasons:
            return (selection.reason or "ринок не дав жодної машини",
                    "послабити вимоги (--max-hours / --budget) або прогнати локально")
        detail = ", ".join(f"{k} — {n}" for k, n in reasons.most_common(3))
        top, _ = reasons.most_common(1)[0]
        action = {
            "стеля годин": "ПІДНЯТИ `--max-hours` (8-12): це стеля ДОПУСТИМОГО "
                           "часу, а не очікуваного. Уночі на ринку лишаються "
                           "дешеві повільні картки, яким треба вчетверо більше "
                           "годин — і саме вони найдешевші за тисячу сторінок",
            "ціна": "підняти `max_price` або `--max-usd-per-1000`",
            "диск": "опустити `disk_gb` у плані",
        }.get(top, "послабити вимоги (--max-hours / --budget) або прогнати локально")
        if top == "ціль темпу":
            # Ціль не послаблюється: слабкий захід — рівно те, чого ціль не
            # допускає. Людина бачить найкраще, що ринок дає (`reason`), і
            # вирішує сама: чекати чи свідомо знизити ціль.
            return (f"{selection.reason or 'жодна машина не дає цілі'} · {detail}",
                    "чекати ринку й перезапустити пізніше (ціль НЕ послаблюється "
                    "автоматично); свідомо погодитись на нижчий темп — "
                    "`-p target_pph=<число>`")
        if top == "підлога темпу":
            # 🔴 Порада з ЧИСЛОМ: найшвидше, що ринок сьогодні дає на ЦЬОМУ
            # матеріалі. Без нього людина знижує підлогу навмання, а тут вона
            # одразу бачить, чи вона поставила недосяжне (на щільному
            # матеріалі 2000 стор/год не дає жодна машина ринку), чи просто
            # не пощастило з годиною.
            best = max((float(getattr(r.sizing, "pages_per_hour", 0) or 0)
                        for r in (getattr(selection, "rejected", []) or [])),
                       default=0.0)
            action = (
                f"найшвидша машина ринку дає {best:.0f} стор/год на цьому "
                f"матеріалі — опустити `--min-pph` під це число або чекати "
                f"ринку; підлога тірами НЕ послаблюється, і це навмисно"
                if best > 0 else
                "опустити `--min-pph` або чекати ринку")
        return (f"{selection.reason or 'ринок не дав придатної машини'}; "
                f"відсіяли ВЛАСНІ стелі: {detail}", action)

    def _warn_pph_floor(self, selection: Any, need: Need) -> None:
        """Сказати ВГОЛОС, скільки машин з'їла підлога темпу — поки вибір є.

        🔴 Мовчазна ручка невідрізнима від зламаної. Підлога ріже тим сильніше,
        чим щільніший матеріал: на метриках 1000 стор/год відсікає 4 машини з
        78, на протоколах консисторії — 18 із 20 (заміряно на живому ринку
        21.09.2026). Перше — запас, друге — лезо, і людина має бачити, на
        якому вона боці, ДО того, як ринок просяде і захід стане.
        """
        floor = float(getattr(need, "min_pages_per_hour", 0) or 0)
        if floor <= 0:
            return
        cut = sum(1 for r in (getattr(selection, "rejected", []) or [])
                  if any("підлоги" in w for w in (getattr(r, "rejects", []) or [])))
        if not cut:
            return
        left = len(getattr(selection, "candidates", []) or [])
        хвіст = cut % 10
        слово = ("машину" if хвіст == 1 and cut % 100 != 11 else
                 "машини" if хвіст in (2, 3, 4) and cut % 100 not in (12, 13, 14)
                 else "машин")
        detail = f"підлога {floor:.0f} стор/год відсікла {cut} {слово}, лишилось {left}"
        action = ("запас є" if left > 3 else
                  "запасу майже немає: ще одна просадка ринку, і захід стане "
                  "з `market_empty`. Тримати `--min-pph` нижче або чекати")
        self.state.note("pph_floor", detail, action=action)
        self._say(f"[supervise] ⚠ {detail} — {action}")

    def _with_deadline(self, call: Any, *, seconds: float, what: str) -> Any:
        """Виконати виклик зі стелею часу. Понад неї — `BackendError`.

        🔴 Потік ДЕМОНСЬКИЙ і навмисно: перервати заблокований у C-коді сокет
        ми не можемо, тож дочекатись його не пробуємо взагалі. Демон не тримає
        інтерпретатор при виході, а результат такого виклику нам уже не
        потрібен — важливо лише не стояти самим, поки цокає оренда.
        """
        import threading

        box: dict[str, Any] = {}

        def _run() -> None:
            try:
                box["value"] = call()
            except BaseException as exc:   # віддаємо виклику як є
                box["error"] = exc

        worker = threading.Thread(target=_run, daemon=True,
                                  name=f"gpurunner-deadline-{what}")
        worker.start()
        worker.join(seconds)
        if worker.is_alive():
            self.state.note(
                "fetch_timeout",
                f"{what}: понад {seconds / 60:.0f} хв без завершення — кидаю",
                action="лишаю те, що вже приїхало; решту добере htr fetch-ckpt")
            raise BackendError(
                f"{what} не вклався у {seconds / 60:.0f} хв — бокс міг зникнути "
                f"посеред обміну")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def _out_dir_for(self, case: CasePlan) -> Path:
        """Куди РОЗКЛАДЕТЬСЯ ця справа. Абсолютний шлях і нічого крім нього.

        🔴🔴 Пастка спрацювала ПЯТЬ разів за одну кампанію. Наглядач живе в
        своєму репозиторії, а справи належать простору дослідження; відносний
        шлях розкладався від теки ЗАПУСКУ, тобто в чужий проєкт. Двічі це
        виглядало як «прогін нічого не дав»: у теці справи нулі, у лозі
        «вичерпано стелю оренд», а на диску лежало 1101 і 441 готова сторінка.
        У першому випадку вже складався план перезапуску на 1388 сторінок.

        Тому шлях не вгадується: або він у плані й абсолютний, або ми голосно
        кажемо, куди саме поклали, — і кажемо ЗАВЖДИ, бо мовчазний успіх тут
        нічим не відрізняється від мовчазної втрати.
        """
        if case.out_dir:
            out_dir = Path(case.out_dir)
            if not out_dir.is_absolute():
                out_dir = out_dir.resolve()
                self._say(f"[supervise] ⚠ {case.case}: `out_dir` у плані ВІДНОСНИЙ — "
                          f"розклав від теки запуску в {out_dir}")
        else:
            out_dir = (_fallback_out_root() / case.case).resolve()
            self.state.note(
                "no_out_dir",
                f"{case.case}: у плані немає `out_dir` — кладу в {out_dir}",
                action="це тека ЗАПУСКУ наглядача, а не простору справи")
            self._say(f"[supervise] ⚠ {case.case}: у плані немає `out_dir` → {out_dir}")
        return out_dir

    def _credit_left(self) -> float | None:
        """Скільки грошей на акаунті Vast. `None` — дізнатись не вдалось.

        Ніколи не вигадуємо число: невідомий баланс і нульовий баланс — різні
        стани, і плутати їх означає або спиняти справний захід, або пускати
        його в стіну.
        """
        try:
            report = self.backend.balance()
        except Exception:
            return None
        value = getattr(report, "available", None)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _stop_if_broke(self) -> bool:
        """Спитати баланс і спинити захід, якщо платити нічим.

        Повертає True, якщо вердикт уже виставлено.
        """
        credit = self._credit_left()
        if credit is None or credit >= 0.05:
            return False
        why = (f"на акаунті Vast ${credit:.2f} — оренда неможлива; "
               f"HTTP 400 на створення інстансу означав КОШТИ, а не зайнятий оффер")
        self.state.note("no_credit", why, action="спиняю захід")
        self.state.finish(
            "no_credit", why,
            human_action="поповнити баланс Vast (https://cloud.vast.ai/billing/) "
                         "і перезапустити — чекпоінти в R2 лишились",
        )
        return True

    def _after_failed_submit(self, e: BackendError, candidate: Any, t0: float) -> str:
        """Прибрати за невдалою орендою. Повертає «next» або «stop».

        🔴🔴 ЦЕЙ КОД БУВ ПРОДУБЛЬОВАНИЙ НАПОЛОВИНУ, І ПОЛОВИНА КОШТУВАЛА ГРОШЕЙ.
        Повторна спроба після очікування замка ринку мала ВЛАСНИЙ `except`, який
        лише писав інцидент `rent_failed` і брав наступного кандидата — без
        `_destroy_orphan`, без обліку витрат, без лічильника оренд і без запису
        в реєстр. Тобто інстанс, створений на другій спробі, лишався горіти до
        дедлайн-сторожа в контейнері.

        Відтворено 2026-08-19 (захід `htr-f337-volost`, три сесії на машині):
        `rent_locked` → чекання 7.5 хв → повторна спроба → `rent_failed:
        instance 48121450 is RUNNING AND BILLING but setup failed` — і бокс
        горів паралельно з новим, доки його не погасив дослідник руками.
        Верхня межа збитку однієї такої сироти — `max_hours` + 30 хв × ціну
        години.

        Гілка рідкісна за побудовою (потрібна конкуренція за замок), тому й
        прожила довго. Тепер обидва шляхи ведуть СЮДИ, і розійтись знову не
        можуть.
        """
        cost = (time.monotonic() - t0) / 3600.0 * offer_dph(candidate.offer)
        outcome = getattr(e, "outcome", "") or ""
        first_line = str(e).split("\n")[0]
        self._destroy_orphan(e)
        self._pending = None   # вартість цієї спроби йде як `cost` нижче
        # 🔴 Було `self.spent_usd += cost`, а наступний же `_accrue()`
        # робить ПРИСВОЄННЯ `settled_usd + поточна оренда` — тобто
        # гроші невдалих оренд стирались на першому ж тіку. Та сама
        # вада, що лагодилась комітом 1ac316a, тільки в сусідній гілці.
        self.settled_usd += cost
        self._accrue()
        if getattr(e, "instance_id", ""):
            # 🔴 Стеля «дві оренди на захід» рахувала тільки УСПІШНІ
            # сабміти. Інстанс, створений і знищений на пробі чи SSH,
            # лічильника не торкався — а всередині одного пошуку
            # перебирається до `max_attempts` кандидатів. 2 × 4 = вісім
            # створених інстансів при `max_rents: 2`, кожен до 900 с
            # тарифікації. Рівно та картина, яку стеля мала закрити.
            self._close_rent()
            self.rents += 1
            self.rents_wasted += 1          # інстанс помер на пробі — нуль сторінок

        if outcome == "no_credit":
            # Гроші скінчились: наступний кандидат дістане той самий 400.
            self.state.note("no_credit", first_line, action="спиняю захід")
            self.state.finish(
                "no_credit", first_line,
                human_action="поповнити баланс Vast (https://cloud.vast.ai/billing/) "
                             "і перезапустити — чекпоінти в R2 лишились",
            )
            return "stop"

        if outcome in ("offer_taken", "market_busy"):
            # Ринок, а не машина й не наш конфіг. У реєстр не пишемо
            # (інакше банили б справні хости за чужу розторопність),
            # просто беремо наступного кандидата.
            self.state.note(
                outcome, first_line,
                action="беру наступного кандидата — це ринок, не залізо",
            )
            # 🔴🔴 Ринок так не поводиться. Чотири різні оффери, включно з
            # найдешевшим, не можуть бути зайняті поспіль — а от порожній
            # баланс віддає 400 на кожен. Тіло відповіді розрізняє це не
            # завжди, тож на другому поспіль питаємо баланс прямо: один
            # дешевий запит проти вісімнадцяти запусків у нікуди (04.09.2026).
            self._gate_rejects.append(outcome)
            self._taken_streak += 1
            if (outcome == "offer_taken" and self._taken_streak >= 2
                    and self._stop_if_broke()):
                return "stop"
            time.sleep(5 if outcome == "offer_taken" else RETRY_SLEEP_SEC)
            return "next"

        # 🔴 `our_bug` приходить і ЯВНО — з воріт заліза, коли проба
        # каналу дістала не 200/206 (протухле presigned-посилання).
        # Умова нижче вимагала ПОРОЖНЬОГО outcome, тож явний вирок
        # провалювався в загальну гілку: справні машини йшли в чорний
        # список одна за одною, `BoxObservation` давився невідомим
        # outcome під `except`, і захід закінчувався діагнозом
        # «ринок порожній» — при єдиній причині в одному посиланні.
        if outcome == "our_bug" or (
            not outcome and getattr(e, "status_code", None) is None
            and not getattr(e, "instance_id", "")
        ):
            # 🔴 Помилка без класифікації — це НАША помилка, а не машини:
            # битий план, забутий параметр, зламаний архів. Записати її в
            # реєстр означало б забанити справну машину назавжди й тихо
            # звузити ринок собі ж. Замір 2026-08-11: один пропущений
            # `input_root` оббрехав три здорові бокси поспіль, серед них
            # RTX 6000Ada 48 ГБ — найшвидшу пропозицію дня.
            self.state.note(
                "our_bug", first_line,
                action="машину НЕ звинувачую — це наш конфіг; спроби на "
                       "інших боксах не допоможуть, спиняюсь",
                machine_id=machine_id_of(candidate.offer),
                cost_usd=cost,
            )
            self.state.finish(
                "failed", f"помилка конфігурації, а не заліза: {first_line}",
            )
            return "stop"

        # 🔴 Некласифікована провина з живим `instance_id` (напр. збій
        # заливки входів) давала `outcome=""` → `BoxObservation` кидав
        # `ValueError: невідомий outcome ''`, його ковтав загальний
        # `except`, і машина не потрапляла в реєстр ЗОВСІМ. Перебирались
        # усі кандидати з оплаченим інстансом кожен, а захід закінчувався
        # оманливим «жоден не пройшов ворота заліза».
        outcome = outcome or "setup_failed"
        mid = machine_id_of(candidate.offer)
        self._gate_rejects.append(outcome)
        # Спершу запис, потім текст: наслідок береться з вердикту реєстру, який
        # уже бачить цей рядок.
        self._record_offer(candidate.offer, outcome, first_line, cost_usd=cost)
        consequence, verdict = self._registry_consequence(mid, outcome)
        # 🔴 Відхилений ключ на машині, де він уже приймався, — збій прив'язки
        # ключа на боці Vast, а не хост. V100 33283 (3 успіхи) 14.09.2026 дістав
        # `ssh_auth_denied`, був викинутий на весь захід, і справу дочитала
        # GTX 1080 за $0.29 за 1000 сторінок проти $0.10 на V100. Одна повторна
        # спроба — одразу, наступною в черзі.
        retry = (outcome == "ssh_auth_denied" and mid is not None
                 and verdict is not None and verdict.state != "banned"
                 and verdict.runs_ok > 0 and mid not in self._auth_retry_used)
        self.state.note(
            outcome, first_line,
            # 🔴 Текст мусить збігатися з тим, що реально станеться: не всі
            # вироки банять (`overpriced`, `oom_pages`, `our_bug` — нейтральні).
            # Інцидент, який обіцяє чорний список там, де його не буде, змушує
            # людину шукати неіснуючу причину звуження ринку.
            action=f"інстанс знищено за {time.monotonic() - t0:.0f} с; машина "
                   f"{mid} — {consequence}"
                   + ("; ключ тут уже приймався — ще одна спроба на ній" if retry else ""),
            machine_id=mid,
            cost_usd=cost,
        )
        if retry:
            self._auth_retry_used.add(mid)
        else:
            self._failed_machines.add(mid)
        time.sleep(RETRY_SLEEP_SEC)
        return "retry" if retry else "next"

    def _destroy_orphan(self, exc: Exception) -> None:
        """Погасити інстанс, який орендувався, але не доїхав до роботи.

        🔴 Найнебезпечніша функція в файлі, і вона вже підвела: `JobHandle`
        вимагає `gpu`, тут його не передавали, і замість гасіння сироти
        вилітав `ValidationError` — тобто рівно там, де треба було спинити
        лічильник, наглядач падав, а бокс лишався горіти (виміряно 2026-08-11,
        інстанс 47458730 у стані `created`).

        Тому тут ловиться `Exception`, а не лише `BackendError`: будь-яка
        помилка мусить лишити по собі гучний слід із номером інстансу, а не
        тишу й рахунок.
        """
        instance_id = getattr(exc, "instance_id", "")
        if not instance_id:
            return
        try:
            stub = JobHandle(
                backend="vast", remote_id=str(instance_id),
                job_name="htr_case", gpu=self.plan.gpu or "any",
            )
            # `force`: провенанс доведений інакше — цей `instance_id` щойно
            # прилетів із винятку НАШОЇ ж оренди, у реєстрі його ще немає.
            self.backend.cancel(stub, force=True)
            self._say(f"[supervise] 🔥 невдалий інстанс {instance_id} знищено")
        except Exception as e:
            self.state.note(
                "destroy_failed", f"інстанс {instance_id}: {type(e).__name__}: {e}",
                action=f"ГАСИТИ РУКАМИ: DELETE /instances/{instance_id}/",
            )

    def _read_progress(self) -> tuple[dict[str, Any] | None, bool]:
        """`_progress.json` із боксу. Другий елемент — чи живий SSH.

        🔴 Ловиться `Exception`, а не лише `BackendError`. `paramiko.SSHException`
        і `AuthError` (він НЕ підклас `BackendError`) проходили повз лічильник
        SSH-збоїв просто в аварійний обробник `run()` — тобто транзієнтний
        мережевий чих валив увесь захід із `verdict: failed` при живому боксі.
        """
        if self._handle is None:
            return None, False
        try:
            client = self.backend._ssh(self._handle, timeout=20)
        except Exception:
            return None, False
        try:
            sftp = client.open_sftp()
            try:
                with sftp.open("/workspace/gpurunner/_progress.json") as fh:
                    import json

                    data = json.loads(fh.read().decode("utf-8"))
                # 🔴 Ріст логу міряємо й ТУТ, а не лише до першого прогресу:
                # між справами черги прогрес уже є (від попередньої), а бокс
                # так само може зависнути на качанні кадрів — 06.09.2026 (P2,
                # ф.230) саме так простояв годину при живому SSH.
                self._note_setup_movement(sftp)
                return (data if isinstance(data, dict) else None), True
            except Exception:
                # SSH живий, `_progress.json` ще немає — це нормально: триває
                # сетап. Але саме тут бокс і зависає непоміченим, тож міряємо
                # ПОСУВАННЯ логу раннера.
                self._note_setup_movement(sftp)
                return None, True
            finally:
                sftp.close()
        except Exception:
            return None, False
        finally:
            client.close()

    def _progress_age(self, progress: dict[str, Any] | None) -> float:
        """Скільки прогрес не оновлювався. Відлік — від ОРЕНДИ, не від заходу.

        🔴 Вік ОБМЕЖЕНИЙ життям поточної оренди, і це не косметика.
        Після переоренди `progress` підхоплюється з чекпоінтів, а `ts` у ньому —
        від СТАРОГО, уже мертвого боксу. Без стелі свіжа машина успадковувала
        чужий вік і за секунди після підйому оголошувалась мертвою:
        `rerent` → нова оренда → знову «прогресу немає 107 хв» → нова оренда.
        Замір 2026-08-11 (Царевка): цикл крутився з інтервалом ~3 хв, справа
        стояла на 2578/4066 півтори години, а рахунок ішов далі.
        """
        rented_age = time.monotonic() - cast(float, self._rented_at)
        if not progress:
            return rented_age
        # 🔴 Міряємо вік РОБОТИ, а не життя процесу. Серцебиття раннера
        # освіжає `ts` кожні 10 с доти, доки живий процес — і цим вимикало
        # обидва детектори застрягання: завислий `curl` чи pip у мертве
        # дзеркало тарифікувались до самої стелі годин під написом «прогрес
        # свіжий». `work_ts` оновлюється лише на РЕАЛЬНІЙ публікації поступу.
        stamp = progress.get("work_ts") or progress.get("ts")
        if not stamp:
            return rented_age
        try:
            ts = datetime.fromisoformat(str(stamp))
        except ValueError:
            return 0.0
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        age = max(0.0, (datetime.now(tz=UTC) - ts).total_seconds())
        return min(age, rented_age)

    def _instance_state(self) -> str:
        if self._handle is None:
            return "gone"
        try:
            inst = self.backend._instance(self._handle)
        except BackendError:
            return "unknown"
        if inst is None:
            # 🔴 Порожня відповідь ≠ «інстанс зник». Свіжий інстанс кілька
            # хвилин просто не з'являється у видачі, і читати це як `gone`
            # означає вбити бокс, який ще не встиг народитись. Перепитуємо
            # один раз, і лише тоді визнаємо зникнення.
            try:
                inst = self.backend._instance(self._handle)
            except BackendError:
                return "unknown"
            if inst is None:
                return "gone" if self._rented_at and (
                    time.monotonic() - self._rented_at
                ) > self.cfg.fresh_rent_grace_sec else "unknown"
        return str(inst.get("actual_status") or inst.get("cur_state") or "unknown").lower()

    # ---- дрібне -----------------------------------------------------------

    def _dph(self) -> float:
        return offer_dph(self._offer or {})

    def _current_rent_usd(self) -> float:
        """Скільки вже коштує ПОТОЧНА оренда."""
        if self._handle is None or self._rented_at is None:
            return 0.0
        return (time.monotonic() - self._rented_at) / 3600.0 * self._dph()

    def _accrue(self) -> None:
        """Витрачене = закриті оренди + поточна.

        🔴 Тут була найдорожча вада дня. Стояло присвоєння
        `spent_usd = (зараз - старт заходу) × dph(поточного оффера)`, а `_destroy`
        занулював `self._offer` — тож `_dph()` віддавав 0, і після кожного
        гасіння витрачене ставало **нулем**. Гроші, спалені на невдалих орендах,
        не потрапляли в бюджет ніколи, і `destroy_budget` не міг спинити серію
        переоренд **за побудовою**: 2026-08-11 чотири оренди за п'ять хвилин
        пройшли повз єдиний запобіжник, який мав їх спинити.
        """
        by_time = self.settled_usd + self._current_rent_usd() + self._pending_usd()
        # Рахунок Vast — правда (у ньому трафік і диск), але приходить із
        # запізненням; «ціна × час» — нижня межа, що рахується щомиті.
        self.spent_usd = max(by_time, self._billed_usd or 0.0)

    #: Як часто питати рахунок Vast на ходу, с. Рахунок приходить із запізненням
    #: і спільний для всіх сесій акаунта; остаточну цифру дає звірка після
    #: заходу (`htr bill`, і сам наглядач на старті — за попередні сесії).
    BILL_REFRESH_SEC = 900.0
    #: Нижче цих сум відношення «рахунок / ціна × час» — шум, множник не ставимо.
    BILL_FACTOR_MIN_USD = 0.03

    def _reconcile_past_bills(self) -> None:
        """Пост фактум: дописати рахунок Vast у стани сесій, завершених за 3 дні.

        Рахунок приходить із запізненням, тож цифра на фініші часто неповна;
        наступний наглядач виправляє її одним запитом. Помилка тут нічого не ламає.
        """
        charges = getattr(self.backend, "instance_charges", None)
        if charges is None:
            return
        try:
            from gpurunner.htr.recover import reconcile_sessions
            from gpurunner.supervise.state import state_dir

            rows = charges(time.time() - 4 * 86400, time.time())
            changed = reconcile_sessions(state_dir(), rows, days=3.0)
        except Exception:
            return
        for session, old, new in changed:
            self._say(f"[supervise] рахунок Vast: {session} ${old:.3f} → ${new:.3f}")

    def _refresh_bill(self, *, force: bool = False) -> None:
        """Підтягнути списання Vast за машини сесії — не частіше за раз на 15 хв.

        Наглядач рахував гроші як «ціна за годину × час», а рахунок Vast містить
        ще трафік і диск: на нічному spr-80 $0.582 проти $0.283, які бачив
        наглядач, з них $0.098 — завантаження кадрів. Бюджетна стеля тоді
        спрацьовує запізно.
        """
        charges = getattr(self.backend, "instance_charges", None)
        ids = set(self.state.instances)
        if charges is None or not ids:
            return
        now = time.monotonic()
        if not force and now - self._bill_read_at < self.BILL_REFRESH_SEC:
            return
        self._bill_read_at = now
        try:
            rows = charges(time.time() - 3 * 86400, time.time())
        except Exception:
            return
        mine = [r for r in rows if str(r.get("instance")) in ids]
        if not mine:
            return
        billed = sum(float(r.get("amount") or 0) for r in mine)
        self._billed_usd = billed
        by_time = self.settled_usd + self._current_rent_usd()
        if billed >= self.BILL_FACTOR_MIN_USD and by_time >= self.BILL_FACTOR_MIN_USD:
            self._bill_factor = max(1.0, billed / by_time)

    def _out_of_budget(self) -> bool:
        return self.spent_usd >= self.plan.budget_usd

    def _book_spend(self, handle: JobHandle, worst_case: float) -> None:
        """Записати оренду у СПІЛЬНИЙ журнал витрат.

        🔴 Бюджет наглядача жив у пам'яті процесу, тож дві паралельні сесії з
        `--budget 3.00` спокійно спалювали $6 — і жодна не бачила чужих витрат.
        Журнал уже існував (`core/budget.py`, атомарний `reserve_within_cap`),
        ним просто ніхто не користувався.

        Стеля тут навмисно велика: наш власний бюджет уже стереже `decide`, а
        це запис заради ВИДИМОСТІ між сесіями.
        """
        try:
            from gpurunner.core import budget as budget_mod

            budget_mod.reserve_within_cap(
                "vast", handle.id, round(float(worst_case), 4),
                cap=1e9, session=self.state.session, owner=self._owner,
            )
        except Exception as e:
            self._say(f"[supervise] ⚠ витрату не записано у спільний журнал: {e}")

    def _settle_spend(self, handle: JobHandle, actual: float) -> None:
        try:
            from gpurunner.core import budget as budget_mod

            budget_mod.settle("vast", handle.id, round(float(actual), 4))
        except Exception:
            pass

    def machine_wide_spend(self) -> float:
        """Скільки цього місяця вже заброньовано на Vast УСІМА сесіями."""
        try:
            from gpurunner.core import budget as budget_mod

            return float(budget_mod.month_committed("vast"))
        except Exception:
            return 0.0

    def _out_of_time(self) -> bool:
        return (time.monotonic() - self.started) / 3600.0 >= self.plan.max_hours

    def _machine_id(self) -> int | None:
        return machine_id_of(self._offer) if self._offer else None

    def _absorb(self, cs: CaseState, progress: dict[str, Any] | None, why: str) -> None:
        self.state.why = why
        if progress:
            cs.pages_done = int(progress.get("pages_done") or 0)
            cs.pages_failed = int(progress.get("pages_failed") or 0)
            cs.pages_per_hour = float(progress.get("pages_per_hour") or 0)
            cs.eta_sec = progress.get("eta_sec")
            _absorb_rate(cs, progress)
            self.state.shards = list(progress.get("shards") or [])
            if oom := int(progress.get("oom_events") or 0):
                cs.detail = f"OOM-подій {oom}"
        self.state.budget = self._budget_view()
        self.state.save()

    def _budget_view(self) -> dict[str, Any]:
        elapsed = (time.monotonic() - self.started) / 3600.0
        return {
            "cap_usd": round(self.plan.budget_usd, 4),
            "spent_usd": round(self.spent_usd, 4),
            # Рахунок Vast за машини сесії (з трафіком і диском) і множник до
            # «ціна × час», яким прогноз множить ціну за годину.
            "billed_usd": None if self._billed_usd is None else round(self._billed_usd, 4),
            "bill_factor": round(self._bill_factor, 3),
            "max_hours": self.plan.max_hours,
            "elapsed_h": round(elapsed, 3),
            "will_fit": self.spent_usd < self.plan.budget_usd and elapsed < self.plan.max_hours,
            # Скільки на Vast цього місяця заброньовано ВСІМА сесіями — щоб
            # видно було чужі витрати, а не лише свої.
            "machine_wide_month_usd": round(self.machine_wide_spend(), 2),
            "rents": self.rents,
            "rents_wasted": self.rents_wasted,
            "max_rents": self.plan.max_rents,
        }

    def _box_view(
        self,
        candidate: ScoredOffer,
        selection: Any,
        probe: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        offer = candidate.offer
        view = {
            "instance_id": self._handle.remote_id if self._handle else None,
            "machine_id": candidate.machine_id,
            "host_id": offer.get("host_id"),
            "gpu": offer.get("gpu_name"),
            "num_gpus": offer.get("num_gpus"),
            "geolocation": offer.get("geolocation"),
            "dph_total": offer_dph(offer),
            # Плата хосту за вхідний трафік: її немає в ціні за годину, а в
            # рахунку Vast вона буває більшою за саму карту.
            "inet_down_cost": offer_down_cost(offer),
            "traffic_usd": round(candidate.traffic_usd, 4),
            "claimed": boxes.claimed_from_offer(offer),
            # 🔴🔴 У `sizing` — розкладка, що РЕАЛЬНО ПОЇХАЛА на бокс, тобто
            # порахована воротами на живому залізі. Доти сюди клалась розкладка
            # з КАРТКИ ОФФЕРА, і стан систематично казав не те: «8 шардів ×
            # 5 потоків» на машині, де ворота дали 1 шард (139040, 2026-08-19),
            # і «16 × 3» там, де працювало 16 × 8. Агент, що читав стан замість
            # рядка `проба:` в лозі, діагностував по числу, якого не існувало.
            # План із картки лишається поруч: їхня різниця і є діагноз.
            "sizing": _sizing_view(self._gate_sizing or candidate.sizing),
            "sizing_planned": _sizing_view(candidate.sizing),
            # Що вибір ОБІЦЯВ: ціль, гарантований і модельний темп. Проти цього
            # числа наглядач звіряє факт (`underdelivers`).
            "target_pph": need_target_pph(self.plan),
            "target_here": (round(self._rent_need.target_here)
                            if self._rent_need is not None else None),
            "pph_sure": round(candidate.pph_sure),
            "pph_model": round(candidate.sizing.pages_per_hour),
            "registry": (
                {"state": candidate.verdict.state, "reason": candidate.verdict.reason}
                if candidate.verdict else {"state": "unknown", "reason": "перша зустріч"}
            ),
        }
        if probe:
            view["measured"] = probe
        return view

    def _record_offer(
        self, offer: dict[str, Any], outcome: str, detail: str, *, cost_usd: float = 0.0
    ) -> None:
        try:
            measured = dict(self._probe)
            if self._boot_sec is not None:
                measured["boot_sec"] = self._boot_sec
            if self._upload_sec is not None:
                measured["upload_sec"] = self._upload_sec
            # Головне число реєстру: скільки ця машина ДАЛА НАСПРАВДІ. Без нього
            # запис про успіх не має ціни сторінки, і наступного разу ми знову
            # обираємо наосліп за карткою оффера.
            if self._measured_pph > 0:
                measured["pages_per_hour"] = round(self._measured_pph)
                # 🔴 З ЯКОГО МАТЕРІАЛУ це число. Темп залежить від жанру:
                # сповідка з її графами йде вдвічі повільніше за метрику на
                # тій самій машині (1562 проти 882 на Q RTX 6000). Без цієї
                # позначки замір виглядає універсальним, хоч він не такий.
                measured["pages_per_hour_case"] = self.plan.cases[0].case                     if self.plan.cases else ""
                # 🔴 Поруч із темпом кладемо ПЛОЩУ кадру, на якій він виміряний.
                # Саме цього бракувало, щоб перевірити питання «чи залежить темп
                # від розміру кадру» чесно: 04.09.2026 я вивів такий закон з
                # ОДНОГО раннього заміру (986 стор/год на 420-й секунді) і
                # помилився — сталий темп на тому самому боксі виявився 1764,
                # тобто вищий за прогноз моделі. Тепер число накопичується разом
                # із геометрією, і наступна спроба спиратиметься на вибірку.
                measured["pages_per_hour_mpx"] = max(
                    (float(getattr(c, "frame_mpx_median", 0) or 0)
                     for c in self.plan.cases), default=0.0)
                # 🔴 І РЯДКИ НА СТОРІНКУ — саме та величина, від якої темп
                # залежить на 80% (`A/(29+рядки)`). Читаються з мети, що вже
                # лежить на диску після забору; без забору — з плану.
                measured["pages_per_hour_lines"] = self._lines_per_page_done()
            if self._uplink_mbs > 0:
                # 🔴 Канал до машини — окреме число від `net_mbps` проби: та
                # міряє шлях ХОСТА до бакета, а це НАШ аплінк до цієї машини.
                # Різниця між ними й є причина, з якої повільна доставка
                # виглядала як швидка машина.
                measured["uplink_mbs"] = round(self._uplink_mbs, 2)
                measured["uplink_via"] = self._uplink_via
            # ЯКИМ раннером зняте число: sha копії в ассетах, звірена на боксі.
            runner_sha = (self.plan.scripts_sha256 or {}).get("htr_case_run.py")
            if runner_sha:
                measured["runner_sha"] = runner_sha[:12]
            boxes.record(
                boxes.observation_from_offer(
                    offer, outcome=outcome, measured=measured, detail=detail,
                    run_id=self._handle.id if self._handle else None,
                    case=self.plan.cases[0].case if self.plan.cases else None,
                    cost_usd=cost_usd,
                )
            )
        except Exception as e:
            self._say(f"[supervise] ⚠ реєстр не записав {outcome}: {e}")

    def _lines_per_page_done(self) -> float:
        """Медіана рядків на сторінку першої справи: з мети на диску, а коли її
        ще немає — з плану (мета попереднього прогону або ручка)."""
        from gpurunner.htr.plan_build import lines_per_page_from_meta

        if not self.plan.cases:
            return 0.0
        first = self.plan.cases[0]
        out_dir = getattr(first, "out_dir", "") or ""
        measured = lines_per_page_from_meta(Path(out_dir)) if out_dir else 0.0
        return measured or float(getattr(first, "lines_per_page_median", 0) or 0)

    def _record_box(self, outcome: str, detail: str) -> None:
        if self._offer:
            self._record_offer(self._offer, outcome, detail, cost_usd=self.spent_usd)

    def _settle(self) -> None:
        """Підбити захід, якщо його не завершило термінальне рішення."""
        self._destroy("захід завершено")
        # Підсумок — з рахунку Vast (трафік і диск), якщо він уже є.
        self._refresh_bill(force=True)
        self.spent_usd = max(self.spent_usd, self._billed_usd or 0.0)
        self.state.budget = self._budget_view()
        if self.state.verdict:
            return
        incomplete = [c for c in self.state.cases if c.status == "incomplete"]
        failed = [c for c in self.state.cases if c.status == "failed"]
        done = [c for c in self.state.cases if c.status == "done"]
        if incomplete:
            self.state.finish(
                "incomplete",
                f"{len(done)} справ повні, {len(incomplete)} — ні: "
                + "; ".join(f"{c.case} (бракує {c.missing_count})" for c in incomplete),
                human_action="переглянути пропущені кадри — догони їх не взяли, "
                             "тобто це не OOM, а самі аркуші",
            )
        elif failed:
            self.state.finish("failed", f"{len(failed)} справ провалились")
        elif not done:
            # 🔴 Остання затичка проти вердикту «ok» без роботи. Раніше сюди
            # падала будь-яка справа, що не потрапила в жоден кошик (напр.
            # лишилась `running` після провалу забору) — і захід оголошував
            # успіх при нулі забраних сторінок. «Нічого не зроблено» ніколи не
            # є успіхом, навіть коли жодна гілка не поскаржилась.
            stuck = ", ".join(f"{c.case} ({c.status})" for c in self.state.cases) or "—"
            self.state.finish(
                "incomplete",
                f"жодної справи не доведено до кінця; стан справ: {stuck}",
                human_action="подивитись `incidents` у стані: захід завершився, "
                             "не давши результату — це не успіх",
            )
        else:
            self.state.finish(
                "ok",
                f"{len(done)} справ прогнано повністю за ${self.spent_usd:.2f}",
            )


def run_plan(plan: Plan, *, session: str | None = None, tick_sec: int = TICK_SEC) -> int:
    return Supervisor(plan, session=session, tick_sec=tick_sec).run()


#: Файли, які раннер ПЕРЕПИСУЄ по ходу справи. Їх фінальний забір мусить
#: оновити, решту (тексти сторінок) — ні: вони незмінні, і зайве копіювання
#: коштувало б часу на десятках тисяч файлів.
_REFRESHABLE = ("htr_case_summary.json", "_progress.json", "_status.json")


def _stamp_case_key(out_dir: Path, case_key: str, *, local_dir: str = "") -> int:
    """Проставити шифру справи в мету забраного прогону (і в теки голосів).

    🔴 Це остання точка, де шифра ще відома. На боксі `case_dir` — шлях
    орендованого контейнера (`/tmp/htrcase/pages_dl_NN`), і після зачистки від
    справи лишається саме ІМʼЯ ТЕКИ. Доки ключ не ставився, зв'язок декоду з
    книгою тримався на розборі цього імені — а нерозібране ім'я дає не помилку,
    а ТИШУ: справа виглядає непрочитаною. Замір 2026-08-19: 909 прогонів,
    `case_key` мали 7.

    Тека голосу (`<справа>-diak_v4`) — окремий прогін для реєстру, тож ключ їй
    потрібен так само. Наявний непорожній ключ не чіпаємо: рішення людини й
    ремонт сильніші за автомат.

    `local_dir` — тека, з якої план пакував кадри. Вона заміняє `case_dir`,
    лише якщо той мертвий (шлях боксу): 8591 після забору лишився з
    `/tmp/htrcase/pages_dl_01`, і полагодити його довелось руками.
    """
    local = local_dir.replace("\\", "/") if local_dir and Path(local_dir).is_dir() else ""
    if not case_key and not local:
        return 0
    targets = [out_dir, *sorted(out_dir.parent.glob(f"{out_dir.name}-*"))]
    n = 0
    for d in targets:
        mp = d / "_htr_meta.json"
        if not mp.is_file():
            continue
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        changed = False
        if case_key and not str(meta.get("case_key") or "").strip():
            meta["case_key"] = case_key
            changed = True
        cur = str(meta.get("case_dir") or "")
        if local and (not cur or not Path(cur).is_dir()):
            meta["case_dir"] = local
            meta["case_dir_note"] = "тека, з якої план пакував кадри (забір наглядача)"
            changed = True
        if not changed:
            continue
        tmp = mp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + chr(10),
                       encoding="utf-8")
        tmp.replace(mp)
        n += 1
    return n


def _have_file(out_dir: Path, rel: str, size: int, *, flatten: bool) -> bool:
    """Чи лежить файл виводу машини вдома тим самим розміром."""
    target = Path(os.path.normpath(out_dir / _remap(Path(rel), out_dir, flatten=flatten)))
    try:
        return target.stat().st_size == size
    except OSError:
        return False


def _remap(rel: Path, out_dir: Path, *, flatten: bool) -> Path:
    """Куди лягає файл із виводу раннера відносно `out_dir`.

    Без `flatten` — як є. З ним `out/x.txt` стає `x.txt`, а `out-diak_v4/x.txt`
    їде в сестринську теку `<справа>-diak_v4/`: саме такої розкладки чекає
    конвеєр споживача, і без неї готовий результат для нього невидимий.
    """
    if not flatten or not rel.parts:
        return rel
    head, rest = rel.parts[0], Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path()
    if head == "out":
        return rest or Path(rel.name)
    if head.startswith("out-"):
        suffix = head[len("out-"):]
        return Path("..") / f"{out_dir.name}-{suffix}" / (rest or Path(rel.name))
    return rel


def _retire_foreign_model(out_dir: Path, src: Path, say: Any = print) -> None:
    """Текст ІНШОЇ моделі в теці — відкласти в `_prev_<модель>/`, а не змішати.

    🔴🔴 Розкладання пропускає наявні `*.txt` («перший запис виграє»), а мету
    перезаписує. 23.09.2026 у 19 теках так лягла мета Писаря v17 поверх тексту
    Скриби v6: звірка бачила «повно» й «v17», а на диску був старий латинський
    текст. Старий декод не видаляється (з нього, наприклад, беруть засів
    сегментації) — він переїжджає поруч і перестає видавати себе за новий.
    """
    from gpurunner.supervise.verify import meta_model

    old = meta_model(out_dir)
    new = meta_model(src) or meta_model(src / "out") if src.is_dir() else ""
    if not old or not new or old == new:
        return
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", old).strip("-") or "model"
    dest = out_dir / f"_prev_{tag}"
    moved = 0
    for item in sorted(out_dir.rglob("*")):
        if not item.is_file() or dest in item.parents or any(
                p.name.startswith("_prev_") for p in item.parents):
            continue
        if item.suffix == ".txt" or item.name.startswith("_htr_meta")                 or item.name in _REFRESHABLE:
            target = dest / item.relative_to(out_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(target))
            moved += 1
    if moved:
        say(f"[supervise] {out_dir.name}: текст моделі {old} ({moved} файлів) → "
            f"{dest.name}/ — нова модель {new} кладеться начисто")


def _is_refreshable(target: Path) -> bool:
    """Чи можна затерти вже наявний файл свіжішою версією.

    🔴 Правило «перший запис виграє» слушне для `*.txt` і хибне для підсумку:
    аварійний забір кладе ЧАСТКОВИЙ `htr_case_summary.json` (`complete: false`,
    список пропущених), і після переоренди він заслоняв фінальний повний — тож
    наглядач оголошував доведену до кінця справу неповною.
    """
    name = target.name
    return name in _REFRESHABLE or name.startswith("_htr_meta")


def _slug(name: str) -> str:
    """Те саме перетворення імені справи, що й у раннері."""
    keep = "".join(ch for ch in str(name) if ch.isalnum() or ch in "-_.")
    return keep[:64] or "case"
