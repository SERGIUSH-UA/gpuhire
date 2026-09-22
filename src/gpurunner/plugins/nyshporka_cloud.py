"""Бекенд оренди для Нишпорки: gpurunner ОРЕНДУЄ й ГАСИТЬ бокс на Vast.ai.

Нишпорка (`nyshporka`) веде свій конвеєр по SSH сама: заливка, прогін, забір,
звірка. Від бекенда їй потрібні три речі — дати машину, дати до неї транспорт,
звільнити її (`nyshporka.cloud.base.CloudBackend`). Цей модуль закриває першу й
третю тим, що в gpurunner уже оплачене інцидентами: скоринг офферів, реєстр
боксів, перевірка балансу, незнімна стеля ціни й таймер самознищення. Другу —
транспорт — він не пише зовсім, а віддає вбудованому `SshBackend` Нишпорки:
два транспорти поруч розійшлись би на першій же правці таймаутів.

Від людини потрібен ЛИШЕ ключ API Vast і гроші на балансі. SSH-ключ
генерується сам (`auth.vast.ensure_ssh_key`), R2/Cloudflare на цьому шляху
немає взагалі.

🔴 Імпорти `nyshporka` і `paramiko` — усередині методів. Модуль вантажить
Нишпорка через entry point `nyshporka.cloud`, тож у робочому шляху вона є за
побудовою; але сам файл мусить імпортуватись і без неї (тести gpurunner, CI).
Коли її немає, типи контракту підміняються місцевими двійниками тієї самої
форми — цього досить, щоб орендувати й гасити, але не щоб з'єднатись.

🔴 Жодного вигаданого числа. Чого API не дав — того ключа у відповіді немає
або він `None`: «невідомий баланс» і «нульовий баланс» — різні стани.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import os
import re
import time
import types
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime, timedelta
from typing import Any

#: Префікс мітки інстансу. За ним `release`/`find` відрізняють свій бокс від
#: чужого на тому самому акаунті (наглядач gpurunner мітить `gpurunner-…`).
LABEL_PREFIX = "nysh-rent"

#: Де Нишпорка працює на боксі. Абсолютний шлях, а не `~/nysh-run`: на Vast
#: диск, який ми замовили, — це `/workspace`, і середовище рушіїв на кілька
#: гігабайтів мусить лягти саме туди.
REMOTE_WORKDIR = "/workspace/nysh-run"

#: Скільки годин закладати, коли `need.max_hours` не задано, — і запас понад
#: них на забір і звірку. Саме з цієї суми зводиться таймер самознищення.
AUTODESTROY_GRACE_HOURS = 1.0

#: Скільки хвилин «інстансу немає в API» ще НЕ означає, що його знищено:
#: свіжий інстанс у видачі з'являється не одразу (див. `BoxGone` у контракті —
#: плутанина цих двох станів коштувала чотирьох оренд поспіль).
FRESH_GRACE = timedelta(minutes=10)

#: Змінна оточення, якою можна підмінити образ контейнера.
IMAGE_ENV = "GPURUNNER_VAST_RENT_IMAGE"

_PPH_RE = re.compile(r"\bpph=(\d+(?:\.\d+)?)")


# ---- контракт Нишпорки: справжній або двійник -------------------------------


class _CloudError(RuntimeError):
    """Двійник `nyshporka.cloud.base.CloudError` — лише коли Нишпорки немає."""


class _AuthError(_CloudError):
    pass


class _BoxGone(_CloudError):
    pass


class _BoxNotReady(_CloudError):
    pass


@dataclass(frozen=True)
class _Box:
    """Двійник `nyshporka.cloud.base.Box` — ті самі поля, той самий `as_dict`."""

    id: str
    backend: str
    label: str = ""
    cores: float = 0.0
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    disk_gb: float = 0.0
    gpus: int = 1
    price_usd_h: float | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {f.name: (dict(self.meta) if f.name == "meta" else getattr(self, f.name))
                for f in fields(self)}


_STANDALONE = types.SimpleNamespace(
    CloudError=_CloudError, AuthError=_AuthError, BoxGone=_BoxGone,
    BoxNotReady=_BoxNotReady, Box=_Box,
)


def contract() -> Any:
    """Типи контракту: з Нишпорки, а без неї — місцеві двійники.

    Винятки мусять бути саме ЇЇ класами: конвеєр ловить `BoxNotReady` і
    `CloudError` за типом, і двійник із тим самим іменем пройшов би повз
    `except` — тобто повз `release`.
    """
    try:
        from nyshporka.cloud import base
    except ImportError:
        return _STANDALONE
    return base


# ---- дрібне ----------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _parse_ts(value: object) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _f(value: object) -> float | None:
    """Число або `None`. Нуль лишається нулем, а не «невідомо»."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _plan_default(name: str) -> Any:
    """Дефолт заходу з `supervise.plan.Plan` — одне джерело на наглядача й плагін."""
    from gpurunner.supervise.plan import Plan

    for f in fields(Plan):
        if f.name == name:
            return f.default
    raise KeyError(name)


def _min_compute_cap(need: Any) -> float:
    """Найстаріша архітектура карти, яку потягне середовище споживача.

    Необов'язкове поле `Need.min_compute_cap` (7.0 = Volta/Turing і новіше).
    Немає поля або воно порожнє — 0.0: не відсівати, бо невідомо, чим там
    читатимуть.
    """
    try:
        return max(0.0, float(getattr(need, "min_compute_cap", 0) or 0))
    except (TypeError, ValueError):
        return 0.0


class VastRent:
    """`nyshporka.cloud.CloudBackend`: оренда на Vast.ai.

    Обов'язкове — `acquire / connect / release / find`. Понад контракт (Нишпорка
    кличе через `getattr`) — `status / login / estimate`; їхня форма описана в
    `docs/contract.md`.
    """

    id = "vast"
    label = "Оренда на Vast.ai"
    caps: frozenset[str] = frozenset({"rent", "cancel", "market"})

    #: Скільки інстансів один `acquire` може СТВОРИТИ. Те саме число й з тієї
    #: самої причини, що `Plan.max_rents`: невдала оренда коштує центи, але без
    #: стелі чотири інстанси створювались і гинули за п'ять хвилин.
    max_rents = 3
    #: Скільки кандидатів перебрати загалом — разом із тими, кого перехопили
    #: (`offer_taken` інстансу не створює й у `max_rents` не рахується).
    max_attempts = 8
    #: Пауза після перехопленого оффера / зайнятого ринку, секунд.
    retry_sleep_sec = 5.0

    def __init__(self, backend_factory: Callable[[], Any] | None = None) -> None:
        self._backend_factory = backend_factory

    # ---- обов'язкове ------------------------------------------------------

    def acquire(self, need: Any, *, target: str = "") -> Any:
        """Орендувати бокс під `need`. Повертає `Box`, до якого вже пускає SSH.

        `target`: порожньо — шукати самому; `<число>` — саме цей оффер (якщо
        він проходить добір); `machine:<id>` — адресно ця машина.
        """
        from gpurunner.auth import vast as vast_auth
        from gpurunner.core.backend import AuthError as GAuthError
        from gpurunner.core.backend import BackendError

        c = contract()
        self._require_api_key(c)
        if importlib.util.find_spec("paramiko") is None:
            raise c.CloudError(
                "немає `paramiko` — без нього до боксу не зайти, тож і орендувати "
                "нема сенсу. Поставте: pip install 'gpuhire[vast]'")
        try:
            priv, _pub = vast_auth.ensure_ssh_key()
        except GAuthError as e:
            raise c.AuthError(str(e)) from e

        score_need, max_price = self._score_need(need)
        bk = self._backend()
        try:
            # 🔴 Баланс — ПЕРЕД ринком. Порожній акаунт віддає на створенні
            # інстансу той самий HTTP 400, що й перехоплений оффер, і без цієї
            # перевірки «немає грошей» виглядало як «ринок розібрали».
            credit = self._credit(bk)
            floor = min(0.5, float(score_need.budget_usd) / 4.0)
            if credit is not None and credit < floor:
                raise c.CloudError(
                    f"поповніть баланс Vast: на акаунті ${credit:.2f} при потребі "
                    f"щонайменше ${floor:.2f} — оренда не почнеться "
                    f"(https://cloud.vast.ai/billing/)")

            selection = self._select(bk, score_need, max_price, target, c,
                                     min_cap=_min_compute_cap(need))
        except GAuthError as e:
            raise c.AuthError(f"Vast відхилив ключ API: {e}") from e
        except BackendError as e:
            raise c.CloudError(f"Vast не відповів на пошук: {e}") from e
        if selection.empty:
            tail = "".join(f"\n  {r.explain}" for r in selection.rejected[:3])
            raise c.CloudError(
                f"на ринку Vast немає придатної машини: "
                f"{selection.reason or 'причина невідома'}{tail}")

        hours = float(score_need.max_hours) + AUTODESTROY_GRACE_HOURS
        image = os.environ.get(IMAGE_ENV, "").strip() or None
        rents = 0
        notes: list[str] = []
        for cand in selection.candidates[: self.max_attempts]:
            if rents >= self.max_rents:
                break
            label = f"{LABEL_PREFIX}-{uuid.uuid4().hex[:8]}"
            rented_at = _now()
            try:
                info = bk.rent_box(cand.offer, disk_gb=score_need.disk_gb, label=label,
                                   image=image, autodestroy_hours=hours)
            except GAuthError as e:
                raise c.AuthError(f"Vast відхилив ключ API: {e}") from e
            except BackendError as e:
                first = str(e).split("\n")[0]
                outcome = str(getattr(e, "outcome", "") or "")
                instance_id = str(getattr(e, "instance_id", "") or "")
                if outcome == "no_credit":
                    raise c.CloudError(f"поповніть баланс Vast: {first}") from e
                if outcome in ("offer_taken", "market_busy"):
                    # Ринок, а не машина: у реєстр не пишемо й оренд не рахуємо.
                    notes.append(first)
                    time.sleep(self.retry_sleep_sec)
                    continue
                if not instance_id:
                    # Інстансу не було — отже це не хост, а наш запит або
                    # налаштування акаунта (темплейт). Наступний кандидат
                    # дістане те саме, тож не перебираємо ринок у стіну.
                    raise c.CloudError(f"Vast відмовив в оренді: {e}") from e
                rents += 1
                self._record_failed_rent(cand.offer, e, outcome or "setup_failed", first)
                if not getattr(e, "destroyed", False):
                    # 🔴 Найгірший стан: інстанс є, гроші йдуть, а погасити не
                    # вийшло. Брати ще один поверх нього не можна.
                    raise c.CloudError(
                        f"інстанс {instance_id} НЕ знищено й він ТАРИФІКУЄТЬСЯ — "
                        f"погасіть у https://cloud.vast.ai/instances/ ; {first}") from e
                notes.append(first)
                continue
            return self._box_from_rent(c, info, cand.offer, label=label, key=str(priv),
                                       rented_at=rented_at, hours=hours)

        raise c.CloudError(
            f"не вдалось орендувати бокс (створено й погашено інстансів: {rents}): "
            + ("; ".join(notes[-3:]) or "кандидати скінчились"))

    def connect(self, box: Any) -> Any:
        """Транспорт — вбудований `SshBackend` Нишпорки, без власної копії."""
        c = contract()
        try:
            from nyshporka.cloud.ssh import SshBackend
        except ImportError as e:
            raise c.CloudError(
                "з'єднання з боксом веде Нишпорка: pip install 'nyshporka[cloud]'") from e
        try:
            return SshBackend().connect(box)
        except c.BoxNotReady as exc:
            # 🔴 «Не відповідає» — ще не «зникла». `BoxGone` кажемо лише тоді,
            # коли API певен, що інстансу немає, І він не свіжий.
            gone = self._gone(box)
            if gone:
                raise c.BoxGone(f"інстанс {box.id}: {gone}") from exc
            raise

    def release(self, box: Any, *, why: str = "") -> None:
        """Знищити інстанс. Ідемпотентно; на мертвому інстансі не кидає.

        `why` керує записом у реєстр боксів: `ok…` — успішна оренда (у деталі
        можна дати `pph=<стор/год>`), `failed:<деталь>` / `slow:<деталь>` —
        погана, решта — нейтрально. Кидає лише тоді, коли інстанс ЖИВИЙ, а
        погасити не вдалось: мовчати про бокс, що горить, гірше за виняток.
        """
        from gpurunner.backends.vast import ForeignInstance
        from gpurunner.core.backend import AuthError as GAuthError
        from gpurunner.core.backend import BackendError

        c = contract()
        iid = str(box.id or "").strip()
        if not iid:
            return
        bk = self._backend()
        info: dict[str, Any] | None = None
        with contextlib.suppress(BackendError):
            info = bk.box_info(iid)
        label = str((info or {}).get("label") or "")
        if label and not label.startswith(LABEL_PREFIX):
            raise c.CloudError(
                f"інстанс {iid} має мітку «{label}», а не «{LABEL_PREFIX}-…» — не гашу: "
                f"на ньому може йти чужа робота")

        last: Exception | None = None
        for attempt in range(3):
            try:
                bk.destroy_box(iid)
                last = None
                break
            except ForeignInstance as e:          # pragma: no cover — мітку звірено вище
                raise c.CloudError(str(e)) from e
            except GAuthError as e:
                raise c.AuthError(
                    f"Vast відхилив ключ API — інстанс {iid} НЕ погашено: {e}") from e
            except BackendError as e:
                last = e
                if attempt < 2:
                    time.sleep(self.retry_sleep_sec)
        if last is not None:
            raise c.CloudError(
                f"інстанс {iid} НЕ погашено ({last}) — він ТАРИФІКУЄТЬСЯ; повторіть "
                f"або погасіть у https://cloud.vast.ai/instances/") from last
        # Спостереження — лише якщо інстанс був живий: повторний `release`
        # (а він законний — кличеться з кожного `finally`) не має писати другий рядок.
        if info is not None:
            self._record_release(box, why)

    def find(self, box_id: str) -> Any:
        """Бокс із ЖИВОГО API — щоб `stop` пережив перезапуск процесу.

        `None` — інстансу немає або він не наш (мітка не `nysh-rent-…`).
        """
        from gpurunner.auth import vast as vast_auth
        from gpurunner.core.backend import AuthError as GAuthError
        from gpurunner.core.backend import BackendError

        c = contract()
        iid = str(box_id or "").strip()
        if not iid.isdigit():
            return None
        try:
            inst = self._backend().box_info(iid)
        except GAuthError as e:
            raise c.AuthError(str(e)) from e
        except BackendError as e:
            # Не `None`: «не вдалось спитати» ≠ «машини немає».
            raise c.CloudError(f"Vast не відповів про інстанс {iid}: {e}") from e
        if inst is None or not str(inst.get("label") or "").startswith(LABEL_PREFIX):
            return None
        priv, _ = vast_auth.find_ssh_key()
        return self._box_from_instance(c, iid, inst, key=str(priv) if priv else "")

    # ---- понад контракт ---------------------------------------------------

    def configured(self) -> bool:
        """Чи лежить ключ API локально. Без мережі — для `nysh doctor`."""
        from gpurunner.auth import vast as vast_auth
        from gpurunner.core.backend import AuthError as GAuthError

        try:
            vast_auth.find_api_key()
        except GAuthError:
            return False
        return True

    def status(self) -> dict[str, Any]:
        """Чи готові орендувати — без оренди. Див. `docs/contract.md`."""
        from gpurunner.auth import vast as vast_auth
        from gpurunner.core.backend import AuthError as GAuthError
        from gpurunner.core.backend import BackendError

        problems: list[str] = []
        api_key = True
        try:
            vast_auth.find_api_key()
        except GAuthError:
            api_key = False
            problems.append(self._no_key_hint())
        priv, _ = vast_auth.find_ssh_key()
        if importlib.util.find_spec("paramiko") is None:
            problems.append("немає `paramiko`: pip install 'gpuhire[vast]'")

        balance: float | None = None
        burning: list[dict[str, Any]] | None = None
        accepted = api_key
        if api_key:
            bk = self._backend()
            try:
                balance = _f(bk.balance().available)
            except GAuthError:
                accepted = False
                problems.append("Vast відхилив ключ API — перевірте Account → API Keys")
            except BackendError as e:
                problems.append(f"баланс невідомий: Vast не відповів ({str(e)[:120]})")
            if accepted:
                try:
                    burning = [
                        {"instance_id": str(i.get("id")), "dph_total": _f(i.get("dph_total")),
                         "label": str(i.get("label") or ""),
                         "gpu_name": str(i.get("gpu_name") or ""),
                         "status": str(i.get("actual_status") or "")}
                        for i in bk.live_instances()
                    ]
                except (GAuthError, BackendError) as e:
                    problems.append(f"що горить — невідомо: Vast не відповів ({str(e)[:120]})")
            floor = min(0.5, float(_plan_default("budget_usd")) / 4.0)
            if balance is not None and balance < floor:
                problems.append(
                    f"поповніть баланс Vast: ${balance:.2f} при потребі щонайменше "
                    f"${floor:.2f} (https://cloud.vast.ai/billing/)")
        return {
            "ready": not problems,
            "problems": problems,
            "balance_usd": balance,
            "api_key": api_key,
            "ssh_key": str(priv) if priv else "",
            # `None` — спитати не вдалось; `[]` — спитали, нічого не горить.
            "burning": burning,
        }

    def login(self, api_key: str) -> dict[str, Any]:
        """Зберегти ключ API, перевірити його, завести SSH-ключ. Повертає `status()`.

        Кидає `AuthError` лише на порожній/кривий ключ; відхилений Vast ключ —
        це рядок у `problems`, бо збережений він однаково лишається.
        """
        from gpurunner.auth import vast as vast_auth
        from gpurunner.core.backend import AuthError as GAuthError

        c = contract()
        try:
            vast_auth.save_api_key(api_key)
        except GAuthError as e:
            raise c.AuthError(str(e)) from e
        extra: list[str] = []
        try:
            vast_auth.ensure_ssh_key()
        except GAuthError as e:
            extra.append(str(e).split("\n")[0])
        try:
            vast_auth.verify()
        except GAuthError as e:
            extra.append(str(e))
        out = self.status()
        for line in extra:
            if not any(line[:40] in p for p in out["problems"]):
                out["problems"].append(line)
        out["ready"] = not out["problems"]
        return out

    def estimate(self, need: Any, *, target: str = "") -> dict[str, Any]:
        """Кошторис без оренди: безкоштовний пошук офферів і той самий скоринг.

        Ті самі числа, що `gpurunner htr supervise --dry-run --json`. На
        порожньому ринку ключів про машину у відповіді НЕМАЄ — вигадувати
        «типову» ціну нема з чого.
        """
        from gpurunner.core.backend import AuthError as GAuthError
        from gpurunner.core.backend import BackendError

        c = contract()
        self._require_api_key(c)
        score_need, max_price = self._score_need(need)
        bk = self._backend()
        try:
            balance = self._credit(bk)
            selection = self._select(bk, score_need, max_price, target, c,
                                     min_cap=_min_compute_cap(need))
        except GAuthError as e:
            raise c.AuthError(f"Vast відхилив ключ API: {e}") from e
        except BackendError as e:
            raise c.CloudError(f"Vast не відповів на пошук: {e}") from e
        out: dict[str, Any] = {
            "empty": bool(selection.empty),
            "reason": selection.reason or "",
            "candidates": 0 if selection.empty else len(selection.candidates),
            "balance_usd": balance,
        }
        best = selection.best
        if best is None:
            return out
        out.update({
            "offer_id": best.offer.get("id"),
            "machine_id": best.machine_id,
            "gpu": str(best.offer.get("gpu_name") or ""),
            "num_gpus": int(best.offer.get("num_gpus") or 1),
            "cores": float(best.offer.get("cpu_cores_effective") or 0),
            "price_usd_h": float(best.offer.get("dph_total") or 0),
            "shards": int(best.sizing.shards),
            "pages_per_hour": float(round(best.sizing.pages_per_hour)),
            "hours": round(float(best.hours), 3),
            "cost_usd": round(float(best.cost), 4),
            "usd_per_1000": round(float(best.usd_per_1000), 4),
            "tier": int(selection.tier.level),
        })
        # Споживач звужує вилку кошторису лише тоді, коли щільність справді
        # врахована в прогнозі, — тому відлуння, а не його власне знання.
        if score_need.lines_per_page > 0:
            out["lines_per_page"] = float(score_need.lines_per_page)
        return out

    # ---- внутрішнє --------------------------------------------------------

    def _backend(self) -> Any:
        """Свіжий бекенд на кожну дію: він кешує ключ API, а `login` його міняє."""
        if self._backend_factory is not None:
            return self._backend_factory()
        from gpurunner.backends.vast import VastBackend

        return VastBackend()

    @staticmethod
    def _no_key_hint() -> str:
        return ("немає ключа API Vast.ai. Візьміть його на https://cloud.vast.ai/ "
                "(Account → API Keys) і збережіть: `nysh cloud rent login` "
                "або `gpurunner auth vast --key <КЛЮЧ>`")

    def _require_api_key(self, c: Any) -> None:
        from gpurunner.auth import vast as vast_auth
        from gpurunner.core.backend import AuthError as GAuthError

        try:
            vast_auth.find_api_key()
        except GAuthError as e:
            raise c.AuthError(self._no_key_hint()) from e

    @staticmethod
    def _credit(bk: Any) -> float | None:
        """Баланс або `None`. Невідомий баланс оренди НЕ спиняє (як у наглядача):
        відмова Vast на створенні інстансу однаково розпізнається окремо."""
        from gpurunner.core.backend import AuthError as GAuthError

        try:
            return _f(bk.balance().available)
        except GAuthError:
            raise
        except Exception:
            return None

    @staticmethod
    def _score_need(need: Any) -> tuple[Any, float]:
        """`nyshporka.cloud.base.Need` → `core.offer_score.Need` + погодинна стеля.

        `None`/0 у стелях — «не задано»: тоді дефолти заходу gpurunner
        (`supervise.plan.Plan`). Абсолютна стеля `ABSOLUTE_MAX_DPH` діє поверх
        будь-якого числа — вона стоїть у самому запиті до ринку.

        ⚠ `prefer_cores` свідомо не стає ні фільтром, ні порядком. У наглядача
        це планка, заради якої він ЧЕКАЄ на ринок, а бере однаково найкращого за
        скором; чекати всередині блокуючого `acquire` нема сенсу, а ядра скор уже
        рахує — через число шардів, яке вони дозволяють.
        """
        from gpurunner.core.htr_sizing import GB_PER_SHARD
        from gpurunner.core.offer_score import Need as ScoreNeed

        pages = max(1, int(getattr(need, "pages", 0) or 0))
        bytes_in = max(0, int(getattr(need, "bytes_in", 0) or 0))
        disk = int(getattr(need, "disk_gb", 0) or 0)
        if disk <= 0:
            # Кадри × 2 (архів і розпаковане) + середовище рушіїв і ваги; не
            # менше дефолту заходу. Завищувати не можна: диск — фільтр ринку.
            disk = max(int(_plan_default("disk_gb")), math.ceil(bytes_in * 2 / 1e9) + 20)
        score_need = ScoreNeed(
            pages=pages,
            max_hours=float(getattr(need, "max_hours", None) or _plan_default("max_hours")),
            budget_usd=float(getattr(need, "budget_usd", None) or _plan_default("budget_usd")),
            disk_gb=disk,
            gb_per_shard=float(getattr(need, "gb_per_shard", 0) or GB_PER_SHARD),
            data_mb_per_page=(bytes_in / 1e6 / pages) if bytes_in else 0.0,
            # Рядків на сторінку, якщо Нишпорка їх знає (з мети попереднього
            # прогону). Нуль = не міряли: прогноз темпу тоді без члена за
            # матеріалом, і кошторис чесно ширший.
            lines_per_page=max(0.0, float(getattr(need, "lines_per_page", 0) or 0)),
        )
        max_price = float(getattr(need, "max_price_usd_h", None)
                          or _plan_default("max_price"))
        return score_need, max_price

    @staticmethod
    def _select(bk: Any, score_need: Any, max_price: float, target: str, c: Any,
                min_cap: float = 0.0) -> Any:
        """Кандидати: ринок із реєстром боксів, або адресно за `target`."""
        from dataclasses import replace

        from gpurunner.core import boxes
        from gpurunner.core.offer_score import select_offers

        want = (target or "").strip()
        if not want:
            return bk.find_candidates(gpu="any", need=score_need, num_gpus=1,
                                      max_price=max_price, min_compute_cap=min_cap)
        if want.startswith("machine:") and want[8:].strip().isdigit():
            from gpurunner.backends.vast import offer_compute_cap

            offers = bk.search_offers(
                gpu="any", max_price=max_price, disk_gb=score_need.disk_gb,
                machine_ids=[int(want[8:])], limit=8)
            if min_cap > 0:
                offers = [o for o in offers if offer_compute_cap(o) >= min_cap]
            return select_offers(offers, score_need, boxes.verdicts())
        if want.isdigit():
            selection = bk.find_candidates(gpu="any", need=score_need, num_gpus=1,
                                           max_price=max_price, min_compute_cap=min_cap)
            mine = [x for x in selection.candidates if str(x.offer.get("id")) == want]
            if mine:
                return replace(selection, candidates=mine, reason="")
            return replace(
                selection, candidates=[], rejected=[],
                reason=f"оффера {want} немає серед придатних (зайнятий, задорогий "
                       f"або не проходить добір)")
        raise c.CloudError(
            f"не розумію «{target}»: для Vast це номер оффера, `machine:<id>` "
            f"або порожньо — тоді машину підберу сам")

    def _box_from_rent(self, c: Any, info: dict[str, Any], offer: dict[str, Any], *,
                       label: str, key: str, rented_at: datetime, hours: float) -> Any:
        from gpurunner.core import boxes

        dph = float(info["dph_total"])
        gpus = int(info.get("num_gpus") or 1)
        geo = info.get("geolocation")
        gpu = str(info.get("gpu_name") or "")
        shown = " · ".join(x for x in (
            f"{gpu}×{gpus}" if gpus > 1 else gpu, f"${dph:.3f}/год", str(geo or "")) if x)
        meta: dict[str, object] = {
            # Рівно ті ключі, які читає `SshBackend._host_of`.
            "host": {"name": f"vast-{info['instance_id']}", "user": str(info["ssh_user"]),
                     "host": str(info["ssh_host"]), "port": int(info["ssh_port"]),
                     "key": key, "workdir": REMOTE_WORKDIR, "python": "python3",
                     "cores": float(info.get("cores") or 0),
                     "vram_gb": float(info.get("vram_gb") or 0),
                     "ram_gb": float(info.get("ram_gb") or 0), "gpus": gpus},
            "instance_id": str(info["instance_id"]),
            "machine_id": info.get("machine_id"),
            "host_id": info.get("host_id"),
            "offer_id": offer.get("id"),
            "label": label,
            "rented_at": rented_at.isoformat(),
            # Верхня межа: таймер на боксі стартує разом із контейнером, тобто
            # не пізніше цієї миті.
            "autodestroy_at": (_now() + timedelta(hours=hours)).isoformat(),
            "autodestroy_hours": hours,
            "gpu_name": gpu,
            "num_gpus": gpus,
            "geolocation": geo,
            "boot_sec": info.get("boot_sec"),
            # Що обіцяла картка — щоб `release` записав спостереження з тим
            # самим «обіцяне поруч із виміряним», що й наглядач.
            "claimed": boxes.claimed_from_offer(offer),
        }
        return c.Box(
            id=str(info["instance_id"]), backend=self.id, label=shown,
            cores=float(info.get("cores") or 0), vram_gb=float(info.get("vram_gb") or 0),
            ram_gb=float(info.get("ram_gb") or 0), disk_gb=float(info.get("disk_gb") or 0),
            gpus=gpus, price_usd_h=dph, meta=meta)

    def _box_from_instance(self, c: Any, iid: str, inst: dict[str, Any], *, key: str) -> Any:
        """Бокс із рядка API. Кладемо лише те, що API справді віддав."""
        gpus = int(_f(inst.get("num_gpus")) or 1)
        gpu = str(inst.get("gpu_name") or "")
        dph = _f(inst.get("dph_total"))
        geo = inst.get("geolocation")
        cores = _f(inst.get("cpu_cores_effective")) or 0.0
        ram_gb = (_f(inst.get("cpu_ram")) or 0.0) / 1024.0
        vram_gb = (_f(inst.get("gpu_ram")) or 0.0) / 1024.0 * gpus
        meta: dict[str, object] = {
            "instance_id": iid, "label": str(inst.get("label") or ""),
            "gpu_name": gpu, "num_gpus": gpus, "geolocation": geo,
            "machine_id": inst.get("machine_id"), "host_id": inst.get("host_id"),
            "status": str(inst.get("actual_status") or ""),
        }
        if inst.get("ssh_host") and inst.get("ssh_port"):
            meta["host"] = {"name": f"vast-{iid}", "user": "root",
                            "host": str(inst["ssh_host"]), "port": int(inst["ssh_port"]),
                            "key": key, "workdir": REMOTE_WORKDIR, "python": "python3",
                            "cores": cores, "vram_gb": vram_gb, "ram_gb": ram_gb, "gpus": gpus}
        started = _f(inst.get("start_date"))
        if started:
            meta["rented_at"] = datetime.fromtimestamp(started, tz=UTC).isoformat()
        shown = " · ".join(x for x in (
            f"{gpu}×{gpus}" if gpus > 1 else gpu,
            f"${dph:.3f}/год" if dph is not None else "", str(geo or "")) if x)
        return c.Box(id=iid, backend=self.id, label=shown, cores=cores, vram_gb=vram_gb,
                     ram_gb=ram_gb, disk_gb=_f(inst.get("disk_space")) or 0.0, gpus=gpus,
                     price_usd_h=dph, meta=meta)

    def _gone(self, box: Any) -> str:
        """Чому інстансу ТОЧНО немає — або порожньо, якщо певності нема."""
        from gpurunner.core.backend import BackendError

        try:
            info = self._backend().box_info(str(box.id))
        except BackendError:
            return ""           # не вдалось спитати — це не доказ зникнення
        if info is not None:
            return ""
        meta = box.meta if isinstance(getattr(box, "meta", None), dict) else {}
        rented = _parse_ts(meta.get("rented_at"))
        if rented is not None and _now() - rented < FRESH_GRACE:
            return ""           # свіжий інстанс в API ще не видно
        return "в API Vast його більше немає (знищено нами, хостом або таймером)"

    @staticmethod
    def _record_failed_rent(offer: dict[str, Any], exc: Exception, outcome: str,
                            detail: str) -> None:
        from gpurunner.core import boxes

        billed = int(getattr(exc, "billed_sec", 0) or 0)
        dph = float(offer.get("dph_total") or 0)
        try:
            boxes.record(boxes.observation_from_offer(
                offer, outcome=outcome, detail=detail[:300],
                run_id=f"nysh-{getattr(exc, 'instance_id', '')}",
                cost_usd=round(billed / 3600.0 * dph, 4), billed_sec=billed))
        except Exception as e:   # реєстр — пам'ять, а не умова оренди
            print(f"[nysh-rent] ⚠ реєстр не записав {outcome}: {e}", flush=True)

    @staticmethod
    def _record_release(box: Any, why: str) -> None:
        """Спостереження про завершену оренду — за словами викликача."""
        from gpurunner.core import boxes

        meta = box.meta if isinstance(getattr(box, "meta", None), dict) else {}
        machine = _f(meta.get("machine_id"))
        if not machine:
            return               # без машини запис нікому нічого не скаже
        text = (why or "").strip()
        low = text.lower()
        measured: dict[str, Any] = {}
        if low.startswith("ok"):
            outcome = boxes.OK
            pph = _PPH_RE.search(low)
            if pph:
                measured["pages_per_hour"] = round(float(pph.group(1)))
            if meta.get("boot_sec") is not None:
                measured["boot_sec"] = meta.get("boot_sec")
        elif low.startswith("failed:"):
            # Перше слово деталі може назвати вирок точніше (`died_under_load`);
            # інакше — найм'якший: провина машини словами викликача не доведена.
            head = (text[7:].split() or [""])[0].strip(".,;:")
            outcome = head if head in boxes.TTL_DAYS else "setup_failed"
        elif low.startswith("slow:"):
            head = (text[5:].split() or [""])[0].strip(".,;:")
            outcome = head if head in boxes.TTL_DAYS else "slow_run"
        else:
            outcome = "user_stop"

        run_id = f"nysh-{box.id}"
        rented = _parse_ts(meta.get("rented_at"))
        billed = max(0, int((_now() - rented).total_seconds())) if rented else 0
        price = _f(getattr(box, "price_usd_h", None)) or 0.0
        claimed = meta.get("claimed")
        try:
            # Той самий `release` законно приходить двічі (збій → `finally`).
            if any(o.run_id == run_id and o.outcome == outcome for o in boxes.read_all()):
                return
            boxes.record(boxes.BoxObservation(
                machine_id=int(machine), outcome=outcome,
                host_id=int(h) if (h := _f(meta.get("host_id"))) is not None else None,
                offer_id=int(o) if (o := _f(meta.get("offer_id"))) is not None else None,
                geolocation=str(meta["geolocation"]) if meta.get("geolocation") else None,
                gpu_name=str(meta.get("gpu_name") or ""),
                num_gpus=int(_f(meta.get("num_gpus")) or 1),
                run_id=run_id,
                claimed=dict(claimed) if isinstance(claimed, dict) else {},
                measured=measured, detail=text[:300],
                cost_usd=round(billed / 3600.0 * price, 4), billed_sec=billed))
        except Exception as e:   # реєстр — пам'ять, а не умова звільнення
            print(f"[nysh-rent] ⚠ реєстр не записав {outcome}: {e}", flush=True)
