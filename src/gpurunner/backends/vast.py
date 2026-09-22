"""VastBackend — rent a GPU on the Vast.ai marketplace, run the job, pull results over SFTP.

Vast.ai is a marketplace, not a batch service: you *rent a box by the hour*. So
the lifecycle differs from Kaggle/Modal/Colab in one important way — **the meter
keeps running until the instance is destroyed**, not until the job finishes. That
shapes every decision here:

  - ``submit`` blocks until the container is up, uploads inputs over SFTP, then
    drops a ``GO`` sentinel that releases the onstart script. Without that
    handshake the job would start before its data arrived.
  - ``status`` reports the *job's* state (from ``_status.json`` on the box) and
    the accrued cost so far ($/h × uptime).
  - ``fetch_outputs`` pulls ``/kaggle/working`` and, when asked, destroys the box.
  - ``cancel`` destroys the instance — that is the only thing that stops billing.

Like the Colab backend, the remote side **emulates the Kaggle filesystem**
(``/kaggle/input/<slug>``, ``/kaggle/working``) and then executes
``job.render_remote_code(params)`` verbatim, so jobs need no Vast-specific code.

REST API (paths verified against vast-ai/vast-cli ``vast.py``):
  ``POST /bundles/``                    search offers  → ``{"offers": [...]}``
  ``PUT  /asks/<offer_id>/``            create instance → ``{"new_contract": <id>}``
  ``GET  /instances/<id>/?owner=me``    instance detail → ``{"instances": {...}}``
  ``DELETE /instances/<id>/``           destroy (stops billing)
  ``POST /instances/<id>/ssh/``         attach an SSH pubkey to a live instance
  ``PUT  /instances/request_logs/<id>/``container logs → ``{"result_url": ...}``

Backend-only params (read from the RAW params, before ``validate_params`` strips
unknown keys): ``inputs`` ``image`` ``disk`` ``max_price`` ``max_hours``
``autodestroy_hours`` ``num_gpus``.
"""

from __future__ import annotations

import contextlib
import json
import os
import posixpath
import stat
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

from gpurunner.auth.vast import API_BASE
from gpurunner.core import boxes
from gpurunner.core.backend import AuthError, Backend, BackendError, SshAuthRejected
from gpurunner.core.inputs import resolve_inputs
from gpurunner.core.job import Job
from gpurunner.core.models import BalanceReport, JobHandle, JobStatus, StatusReport
from gpurunner.core.offer_score import TIERS, Need, Selection, select_offers

#: Remote paths. ``/kaggle/*`` is the shim every job already knows how to use.
REMOTE_ROOT = "/workspace/gpurunner"
REMOTE_INPUT = "/kaggle/input"
REMOTE_WORKING = "/kaggle/working"

#: gpurunner GPU name → Vast ``gpu_name`` filter value.
_GPU_TO_VAST: dict[str, str | None] = {
    "any": None,
    "T4": "Tesla T4",
    "L4": "L4",
    "RTX2080Ti": "RTX 2080 Ti",
    "RTX3090": "RTX 3090",
    "RTX4090": "RTX 4090",
    "RTX4060Ti": "RTX 4060 Ti",
    "RTX5060Ti": "RTX 5060 Ti",
    "V100": "Tesla V100",
    "A5000": "RTX A5000",
    "A6000": "RTX A6000",
    "A100": "A100 PCIE",
    "H100": "H100 PCIE",
}

#: Torch+CUDA preinstalled, python3/pip present. Overridable with ``-p image=…``.
DEFAULT_IMAGE = "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
DEFAULT_DISK_GB = 60
DEFAULT_MAX_HOURS = 12

#: Груба відсічка мертвих хостів у пошуку. Не приймач: заявленій швидкості в
#: картці вірити не можна (див. ``search_offers``).
DEFAULT_MIN_INET_DOWN = 100.0

#: Понад стільки виключених машин `notin` у запит не кладемо — лишається
#: клієнтський відсів.
_MAX_NOTIN = 50

_HEARTBEAT_STALE = timedelta(minutes=10)
#: Загальна стеля очікування SSH.
#:
#: 🔴 Тут важливо НЕ плутати два різні провали. Відмова ключа розпізнається
#: окремо й виходить негайно — саме вона колись коштувала 15 хвилин оплаченого
#: очікування. А от повільний старт контейнера — це не поломка: хост тягне
#: 3-гігабайтний образ своїм каналом, і на холодній машині це законно триває
#: довше за здорові 40-60 секунд. Спроба зробити тут «швидко» (420 с) одразу
#: дала хибний вирок справній машині 2026-08-11.
_SSH_WAIT_TIMEOUT = 900
#: Скільки терпіти відмову ключа після повторної прив'язки (API інколи
#: доносить ключ із запізненням).
#: 🔴 Підняте 90 → 240 разом із прогрес-обізнаністю. Дев'яносто секунд
#: вистачало лише на миттєву прив'язку ключа; на боксі, що ще ставить пакети,
#: це означало вирок справній машині (див. гілку `SshAuthRejected`).
_SSH_AUTH_GRACE = 240
#: 🔴 Скільки терпіти бокс, який ще не дав контейнера — рахуючи `loading`.
#:
#: Підказка самого Vast: підйом триває «близько 30 секунд, якщо образ повністю
#: закешований, і **від хвилин до годин**, якщо ні». Тобто затяжний `loading`
#: означає рівно одне: на цьому хості нашого образу немає, і ми платимо за
#: його викачування. Замір 2026-08-11: бокс 9 хвилин показував
#: `Pulling from pytorch/pytorch` і не почав роботи.
#:
#: Правило користувача (уточнено 2026-08-11): **шість хвилин** — і гасимо,
#: беремо інший. Спершу стояло три, але це відсіювало й ті хости, що просто
#: тягнуть образ по повільному каналу і за хвилину-другу почали б рахувати:
#: переоренда коштує ВЛАСНОГО холодного старту (~8 хв), тобто дорожча за
#: очікування. Шість — компроміс: чужий `docker pull` на 9+ хвилин ловиться,
#: а нормальний повільний підйом уже ні.
_BOOT_STUCK_SEC = 360
#: 🔴 Скільки чекати SSH ПІСЛЯ того, як Vast сказав `running`.
#:
#: Це окреме число, бо це вже інша хвороба. Поки інстанс `created`/`loading`,
#: чекати законно — хост тягне образ. Але коли статус став `running`, а
#: `status_msg` каже «success, running pytorch/…», контейнер УЖЕ живий: SSH на
#: ньому підіймається за секунди. Замір 2026-08-11: бокс 13 хвилин показував
#: `running, success`, а наглядач мовчки добивав свої 900 с загального
#: таймауту — тринадцять хвилин оренди за нуль роботи.
#: 🔴 Піднято 180 → 360 разом із порогом підйому (рішення користувача,
#: 2026-08-11). Замір того ж дня: контейнер у Чилі працював 261 с, SSH не
#: відповідав — машину знищено й забанено, а це була друга й остання оренда
#: заходу. Для порівняння, успішний бокс на спр.114 узагалі піднявся за 208 с.
#: Тобто стеля відсікала не хворі хости, а повільний проксі Vast — і коштувала
#: цілого заходу.
_SSH_AFTER_RUNNING_SEC = 360
_PROBE_TIMEOUT = 90
#: Запас понад `max_hours` для безумовного дедлайн-кілера: доробити хвіст,
#: віддати результат і дати наглядачу забрати його штатно.
_DEADLINE_GRACE_SEC = 1800
#: Стеля на ВЕСЬ забір і на одне читання. Без них завмерле з'єднання вішає
#: наглядача назавжди (див. `fetch_outputs`). 62 МБ їдуть секунди, кілька ГБ —
#: хвилини, тож година на забір і хвилина на читання — це стеля аварії, а не
#: робочий режим.
FETCH_DEADLINE_SEC = 3600.0
FETCH_READ_TIMEOUT_SEC = 60.0
#: 🔴🔴 Абсолютна стеля ціни картки, $/год. Не параметр і не дефолт: параметри
#: забуваються, дефолти обходяться старими планами. Виміряна причина: заходи
#: раз за разом брали 3-4-карткові збірки по $0.50/год, бо перевірка «за тисячу
#: сторінок» дорогу-швидку карту ПРОПУСКАЄ. Усе, що ми рахуємо (RTX 3090, P40,
#: 4070, 2080), лежить нижче цієї межі з запасом.
ABSOLUTE_MAX_DPH = 0.365

_NUMERIC_PROBE_KEYS = frozenset(
    {
        "cores", "cores_all", "ram_gb", "vram_total_mb", "vram_free_mb",
        "vram_free_min_mb", "n_gpus", "disk_free_gb", "net_bps",
        # Квота КОНТЕЙНЕРА з cgroup: `cores` (nproc) і `ram_gb` (free) — про
        # хост, і на боксах, де продано частку, розходяться в рази.
        "cores_quota", "ram_limit_gb",
    }
)


#: Слова, якими Vast описує брак коштів у тілі 400. Список навмисно вузький:
#: хибне спрацювання тут спиняє захід, тоді як пропуск лише повертає стару
#: поведінку («оффер зайняли») — тобто помилятись дешевше в бік мовчання.
_NO_CREDIT_MARKERS = (
    "insufficient credit",
    "insufficient funds",
    "insufficient balance",
    "not enough credit",
    "no credit",
    "negative balance",
    "add credit",
    "payment method",
    "billing",
)


def _looks_like_no_credit(e: Exception) -> bool:
    """Чи це 400 «немає грошей», а не 400 «оффер уже зайняли».

    Обидва приходять однаковим кодом, і розрізняє їх лише тіло відповіді.
    Дивимось і в `body` (його зберігає `_request_once`), і в текст помилки:
    у тіла є шанс бути порожнім, якщо Vast відповів без JSON.
    """
    haystack = f"{getattr(e, 'body', '') or ''} {e}".lower()
    return any(marker in haystack for marker in _NO_CREDIT_MARKERS)


def offer_compute_cap(offer: dict[str, Any]) -> float:
    """Compute capability карти з оффера Vast: поле `compute_cap` — ціле ×100 (750 = 7.5).

    Немає поля або воно не число — 0.0, тобто «невідомо»: за заданого мінімуму
    такий оффер не проходить.
    """
    try:
        raw = float(offer.get("compute_cap") or 0)
    except (TypeError, ValueError):
        return 0.0
    return raw / 100.0 if raw >= 20 else raw


class VastBackend(Backend):
    """Run jobs on rented Vast.ai instances."""

    name: ClassVar[str] = "vast"
    gpu_choices: ClassVar[tuple[str, ...]] = tuple(_GPU_TO_VAST.keys())
    default_gpu: ClassVar[str] = "RTX3090"

    def __init__(self) -> None:
        self._key: str | None = None

    # ---- public API -------------------------------------------------------

    def check_auth(self) -> None:
        from gpurunner.auth import vast as vast_auth

        vast_auth.require_ssh_key()
        self._api_key()
        vast_auth.verify()

    def submit(self, job: Job, params: dict[str, Any], *, gpu: str) -> JobHandle:

        if gpu not in self.gpu_choices:
            raise ValueError(
                f"Unsupported gpu '{gpu}' for vast backend. Choose from: {self.gpu_choices}"
            )
        if self.name not in job.supported_backends:
            raise BackendError(f"Job {job.name!r} does not declare 'vast' in supported_backends")

        opts = _backend_opts(params)
        offer = self.pick_offer(
            gpu=gpu,
            max_price=opts["max_price"],
            disk_gb=opts["disk"],
            num_gpus=opts["num_gpus"],
            min_cpu=opts["min_cpu"],
            min_ram_gb=opts["min_ram_gb"],
            min_reliability=opts["min_reliability"],
            min_vram_gb=opts["min_vram_gb"],
            min_cuda=opts["min_cuda"],
        )
        return self.submit_to_offer(job, params, gpu=gpu, offer=offer)

    def submit_to_offer(
        self,
        job: Job,
        params: dict[str, Any],
        *,
        gpu: str,
        offer: dict[str, Any],
        verify: Any = None,
        on_created: Any = None,
    ) -> JobHandle:
        """Зайняти КОНКРЕТНИЙ оффер і підготувати бокс до роботи.

        Відділено від `submit` заради наглядача: він ранжує ринок сам і має
        могти піти до наступного кандидата, не повторюючи пошук.

        `verify(handle, client) -> dict | None` — гачок, що виконується
        **після SSH, але до заливки даних і pip**. Саме туди наглядач вішає
        пробу заліза й ворота: якщо машина не така, як обіцяла, ми ще нічого
        на неї не залили (виняток звідти зупиняє підготовку). Повернений
        словник накладається на параметри job'а — так число шардів береться з
        ВИМІРЯНОЇ VRAM, а не з картки оффера.
        """
        from gpurunner.auth import vast as vast_auth

        _, ssh_pub = vast_auth.require_ssh_key()
        opts = _backend_opts(params)
        inputs = _resolve_inputs(job, params, opts)
        # Ранній розбір параметрів: краще впасти на друкарській помилці зараз,
        # ніж на оплачуваній машині через дві хвилини.
        normalized = job.validate_params(params)

        print(
            f"[vast] offer {offer.get('id')}: {offer.get('gpu_name')}×{offer.get('num_gpus')} · "
            f"cpu_eff={float(offer.get('cpu_cores_effective') or 0):.0f} · "
            f"RAM={float(offer.get('cpu_ram') or 0)/1024:.0f}GB · "
            f"${float(offer.get('dph_total') or 0):.3f}/h · {offer.get('geolocation')}",
            flush=True,
        )

        handle = JobHandle(
            backend=self.name,
            remote_id="<pending>",
            job_name=job.name,
            params=normalized,
            gpu=gpu,
        )

        onstart = self._render_onstart(
            max_hours=opts["max_hours"],
            autodestroy_hours=opts["autodestroy_hours"],
        )
        # 🔴 У мітці — СПРАВА, а не лише id хендла. Інцидент 2026-08-11: дві
        # паралельні сесії на одному акаунті ділять і `runs.sqlite3`, і
        # префікс мітки `gpurunner-htr_case-…`, тож чужий інстанс не
        # відрізнити від свого. Наслідок — знищений чужий бокс посеред
        # роботи. Ім'я справи в мітці робить власника видимим одразу і в
        # UI Vast, і в `/api/v1/instances/`.
        payload = _instance_payload(
            offer, image=opts["image"], disk_gb=float(opts["disk"]),
            label=_instance_label(job.name, params, handle.id),
            onstart=onstart, num_gpus=opts["num_gpus"],
        )
        instance_id = self._create_instance(offer, payload)
        # 🔴 Повідомити ВІДРАЗУ: з цієї секунди інстанс існує і ТАРИФІКУЄТЬСЯ,
        # хоча до готовності ще хвилини (образ, контейнер, SSH). Доти наглядач
        # дізнавався про нього лише після УСПІШНОГО сабміту, тож у стані стояло
        # «оренд 0, $0.00» — і агент, читаючи цей стан, спокійно чекав, поки
        # машина горить. Заміряно 2026-08-11 неодноразово.
        if on_created is not None:
            try:
                on_created(str(instance_id), offer)
            except Exception as exc:   # гачок не має валити оренду
                print(f"[vast] ⚠ on_created впав: {exc!r}", flush=True)
        handle.remote_id = str(instance_id)
        # `volume_name` is the ABC's generic durable-storage slot; for Vast it
        # records what we rented, so `status` can price the run without re-querying
        # the marketplace.
        handle.volume_name = (
            f"{offer.get('gpu_name')}×{offer.get('num_gpus')} "
            f"@ ${float(offer.get('dph_total') or 0):.3f}/h"
        )

        # 🔴 Реєструємо handle ДО setup. Лічильник уже цокає, і якщо заливка
        # впаде (а вона падала — SSH, 2026-08-09), `gpurunner cancel <id>` мусить
        # працювати. Раніше handle зберігав тільки CLI ПІСЛЯ успішного submit,
        # тож на єдиному сценарії, де cancel і потрібен, він відповідав
        # «Unknown handle», і інстанс доводилось гасити руками через API.
        from gpurunner.core import manifest as _manifest

        handle.status = JobStatus.QUEUED
        handle.updated_at = datetime.now(tz=UTC)
        try:
            _manifest.add(handle)
        except Exception as e:  # реєстр — не привід не віддати id користувачу
            print(f"[vast] ⚠ не вдалось записати handle у реєстр: {e}\n"
                  f"    інстанс {instance_id} орендовано; гасити: "
                  f"gpurunner cancel {handle.id[:8]} або DELETE /instances/{instance_id}/",
                  flush=True)

        try:
            self._request(
                "POST",
                f"/instances/{instance_id}/ssh/",
                json={"ssh_key": ssh_pub.read_text(encoding="utf-8").strip()},
            )
            client = self._wait_for_ssh(handle)
            try:
                # 🔴 Ворота — ТУТ, між SSH і першим байтом даних. Усе, що нижче
                # (заливка, pip, ваги), коштує хвилини; усе, що вище, — секунди.
                if verify is not None:
                    overrides = verify(handle, client) or {}
                    if overrides:
                        normalized = job.validate_params({**params, **overrides})
                        handle.params = normalized
                job_py = self._render_job_py(job.render_remote_code(normalized))
                self._upload_text(client, f"{REMOTE_ROOT}/job.py", job_py)
                self._upload_inputs(client, inputs)
                self._exec(client, f"mkdir -p {REMOTE_WORKING} && touch {REMOTE_ROOT}/GO")
            finally:
                client.close()
        except Exception as e:
            # 🔴 Причину не губимо: наглядач за нею вирішує, чи ця машина
            # потрапляє в чорний список і за що саме. Раніше все згорталось у
            # безіменний BackendError, і реєстру не було чого записати.
            failed = BackendError(
                f"instance {instance_id} is RUNNING AND BILLING but setup failed: {e}\n"
                f"Destroy it now:  gpurunner cancel {handle.id[:8]}"
            )
            failed.outcome = getattr(e, "outcome", "") or ""
            failed.instance_id = str(instance_id)  # type: ignore[attr-defined]
            raise failed from e

        handle.status = JobStatus.RUNNING
        handle.updated_at = datetime.now(tz=UTC)
        return handle

    def _create_instance(self, offer: dict[str, Any], payload: dict[str, Any]) -> str:
        """`PUT /asks/<offer>/` — і розбір відмови на ті причини, що вимагають
        РІЗНИХ реакцій. Спільне для job'а (`submit_to_offer`) і голої оренди
        (`rent_box`): класифікація 400-х куплена інцидентами, і друга копія
        розійшлася б із першою на першому ж новому тексті помилки Vast.
        """
        try:
            created = self._request("PUT", f"/asks/{offer['id']}/", json=payload)
        except BackendError as e:
            status = getattr(e, "status_code", None)
            if status == 400 and _looks_like_no_credit(e):
                # 🔴🔴 Порожній баланс віддає ТОЙ САМИЙ 400, що й зайнятий
                # оффер, і доти читався як «ринок». Ціна 2026-09-04
                # (таращанська кампанія): вісімнадцять запусків наглядача
                # поспіль у нікуди, дві зміни ручки шарда й години
                # розслідування механізму, який працював справно. `--dry-run`
                # при цьому щоразу обіцяв десять кандидатів — він інстансу не
                # створює й на баланс не дивиться.
                broke = BackendError(
                    f"Vast відмовив у створенні інстансу (HTTP 400) через КОШТИ: "
                    f"{str(e)[:200]} — поповнити на https://cloud.vast.ai/billing/"
                )
                broke.outcome = "no_credit"
                raise broke from e
            if status in (400, 404, 409) and "template" not in str(e).lower():
                # 🔴 Гонка за оффер, а не наша помилка. Дві сесії (або ми й
                # чужий покупець) ранжують ринок одночасно й тягнуться до тієї
                # самої найкращої пропозиції; хто другий — дістає 400/404.
                # Машина ні в чому не винна, у реєстр це НЕ пишеться, і захід
                # мусить просто взяти наступного кандидата.
                taken = BackendError(
                    f"оффер {offer.get('id')} уже зайняли, поки ми його ранжували "
                    f"(HTTP {status}) — беру наступного кандидата"
                )
                taken.outcome = "offer_taken"
                raise taken from e
            if status == 429 or (status and status >= 500):
                busy = BackendError(
                    f"Vast зараз не приймає оренду (HTTP {status}) — це бік ринку, "
                    f"не наш конфіг"
                )
                busy.outcome = "market_busy"
                raise busy from e
            if "template" in str(e).lower():
                # 🔴 Дефолтний темплейт акаунта накладається на все, що
                # створюється, і порожній/зламаний темплейт валить оренду ще
                # до контейнера. Це не наш конфіг — це налаштування в UI, і
                # полагодити його може тільки власник акаунта.
                raise BackendError(
                    f"Vast відхилив оренду через ТЕМПЛЕЙТ АКАУНТА: {e}\n"
                    f"Це налаштування в UI, не параметр gpurunner. Полагодити:\n"
                    f"  cloud.vast.ai → Templates → зробити дефолтним робочий "
                    f"({DEFAULT_IMAGE}) або прибрати дефолтний зовсім.\n"
                    f"Ознака та сама, якщо інстанс створився, але висить у "
                    f"`created` і SSH не піднімається взагалі."
                ) from e
            raise
        instance_id = created.get("new_contract")
        if not instance_id:
            raise BackendError(f"Vast did not return an instance id: {created}")
        return str(instance_id)

    # ---- оренда без job'а --------------------------------------------------

    def rent_box(
        self,
        offer: dict[str, Any],
        *,
        disk_gb: float,
        label: str,
        image: str | None = None,
        autodestroy_hours: float | None = None,
        on_created: Any = None,
    ) -> dict[str, Any]:
        """Орендувати бокс БЕЗ job'а: інстанс, ключ, SSH — і більше нічого.

        Для викликача, який працює на машині сам (плагін Нишпорки веде свій
        конвеєр по SSH). Тому тут немає ні `job.py`, ні сентинела `GO`, ні
        Kaggle-подібної файлової системи: onstart лише зводить таймер
        самознищення.

        🔴 Таймер — не опція. `autodestroy_hours=None` означає дефолт, а не
        «без страховки»: у цього шляху немає наглядача, який погасив би бокс
        за мертвого викликача, тож без таймера закритий ноутбук дорівнює
        рахунку до ручного втручання.

        🔴 Будь-який провал ПІСЛЯ створення інстансу гасить його перед тим, як
        віддати виняток, — включно з Ctrl+C (`BaseException`): очікування SSH
        триває хвилини, і саме тоді людина найчастіше перериває. Виняток несе
        `outcome` (для реєстру боксів), `instance_id` і `destroyed`.

        `on_created(instance_id, offer)` кличеться в мить, коли пішли гроші.
        """
        from gpurunner.auth import vast as vast_auth

        hours = float(_RENT_AUTODESTROY_HOURS if autodestroy_hours is None
                      else autodestroy_hours)
        if hours <= 0:
            raise ValueError("rent_box: autodestroy_hours мусить бути > 0 — без таймера "
                             "самознищення бокс не орендується")
        cards = max(1, int(offer.get("num_gpus") or 1))
        dph = float(offer.get("dph_total") or 0)
        # Стеля вже стоїть у запиті до ринку (`search_offers`), але оффер сюди
        # може прийти й не звідти. Незнімна стеля мусить бути незнімною на
        # кожному вході, інакше вона — лише звичай одного з викликачів.
        if dph > ABSOLUTE_MAX_DPH * cards:
            raise BackendError(
                f"оффер {offer.get('id')}: ${dph:.3f}/год понад абсолютну стелю "
                f"${ABSOLUTE_MAX_DPH:.3f}/год на карту (карт {cards}) — не орендую"
            )

        _, ssh_pub = vast_auth.ensure_ssh_key()
        payload = _instance_payload(
            offer, image=str(image or DEFAULT_IMAGE), disk_gb=float(disk_gb),
            label=label, onstart=self._render_rent_onstart(autodestroy_hours=hours),
            num_gpus=cards,
        )
        print(
            f"[vast] rent offer {offer.get('id')}: {offer.get('gpu_name')}×{cards} · "
            f"${dph:.3f}/h · {offer.get('geolocation')} · самознищення через {hours:g} год",
            flush=True,
        )
        t0 = time.monotonic()
        instance_id = self._create_instance(offer, payload)
        handle = JobHandle(backend=self.name, remote_id=instance_id,
                           job_name="rent_box", gpu="any")
        try:
            if on_created is not None:
                try:
                    on_created(instance_id, offer)
                except Exception as exc:   # гачок не має валити оренду
                    print(f"[vast] ⚠ on_created впав: {exc!r}", flush=True)
            self._request(
                "POST", f"/instances/{instance_id}/ssh/",
                json={"ssh_key": ssh_pub.read_text(encoding="utf-8").strip()},
            )
            self._wait_for_ssh(handle).close()
            inst = self._instance(handle) or {}
            host, port = inst.get("ssh_host"), inst.get("ssh_port")
            if not host or not port:
                raise BackendError(
                    f"instance {instance_id}: SSH відповів, але API не віддає адреси")
        except BaseException as e:
            destroyed = False
            try:
                self.destroy_box(instance_id)
                destroyed = True
                print(f"[vast] 🔥 невдалий інстанс {instance_id} знищено", flush=True)
            except Exception as kill_exc:
                print(f"[vast] ⚠ інстанс {instance_id} НЕ знищено ({kill_exc}) — "
                      f"ГАСИТИ РУКАМИ: DELETE /instances/{instance_id}/", flush=True)
            if not isinstance(e, Exception):
                raise          # Ctrl+C / SystemExit — не перефарбовуємо
            failed = BackendError(
                f"instance {instance_id}: бокс не піднявся — {e}"
                + ("" if destroyed else
                   f"\nІнстанс НЕ знищено й ТАРИФІКУЄТЬСЯ: DELETE /instances/{instance_id}/")
            )
            failed.outcome = getattr(e, "outcome", "") or "setup_failed"
            failed.instance_id = instance_id  # type: ignore[attr-defined]
            failed.destroyed = destroyed  # type: ignore[attr-defined]
            failed.billed_sec = int(time.monotonic() - t0)  # type: ignore[attr-defined]
            raise failed from e

        gpu_ram_mb = float(offer.get("gpu_ram") or 0)
        return {
            "instance_id": instance_id,
            "machine_id": offer.get("machine_id"),
            "host_id": offer.get("host_id"),
            "ssh_host": str(host),
            "ssh_port": int(port),
            "ssh_user": "root",
            "gpu_name": str(offer.get("gpu_name") or inst.get("gpu_name") or ""),
            "num_gpus": cards,
            "cores": float(offer.get("cpu_cores_effective") or 0),
            "ram_gb": float(offer.get("cpu_ram") or 0) / 1024.0,
            "vram_gb": gpu_ram_mb / 1024.0 * cards,
            "disk_gb": float(disk_gb),
            "dph_total": dph,
            "geolocation": offer.get("geolocation"),
            "boot_sec": int(time.monotonic() - t0),
        }

    def destroy_box(self, instance_id: str | int, *, label_prefix: str = "") -> None:
        """Знищити інстанс. Ідемпотентно: якого вже немає — не помилка.

        `DELETE` шлеться ЗАВЖДИ, а не лише коли інстанс видно в API: свіжий
        інстанс кілька хвилин у видачі відсутній, і «не бачу — отже нема чого
        гасити» лишило б горіти саме той бокс, який щойно створили.

        `label_prefix` — доказ власності для викликача без запису в реєстрі
        прогонів: якщо мітку видно і вона НЕ наша, відмовляємось
        (`ForeignInstance`). Номер інстансу приходить зі стану, який пережив
        правку руками, а чужий бокс на тому самому акаунті — це чужа робота.
        """
        if label_prefix:
            info = self.box_info(instance_id)
            label = str((info or {}).get("label") or "")
            if label and not label.startswith(label_prefix):
                raise ForeignInstance(
                    f"інстанс {instance_id} має мітку «{label}», а не «{label_prefix}…» — "
                    f"гасити не буду: на ньому може йти чужа робота"
                )
        stub = JobHandle(backend=self.name, remote_id=str(instance_id),
                         job_name="rent_box", gpu="any")
        # `force`: власність доведена міткою вище або самим викликачем (він
        # щойно створив цей інстанс); запису в реєстрі прогонів у голої оренди
        # немає за побудовою.
        self.cancel(stub, force=True)

    def box_info(self, instance_id: str | int) -> dict[str, Any] | None:
        """Рядок інстансу з API або `None`, якщо його там немає.

        ⚠ `None` — «API його не показує», а не «його знищено»: щойно створений
        інстанс теж деякий час відсутній. Висновок робить викликач, який знає
        вік оренди.
        """
        stub = JobHandle(backend=self.name, remote_id=str(instance_id),
                         job_name="rent_box", gpu="any")
        try:
            return self._instance(stub)
        except BackendError as e:
            if getattr(e, "status_code", None) == 404:
                return None
            raise

    def live_instances(self) -> list[dict[str, Any]]:
        """Що зараз ТАРИФІКУЄТЬСЯ на акаунті — прямим запитом, повз наш реєстр.

        Те саме, що показує `gpurunner burn`. Кидає `BackendError`, якщо API
        не відповів: порожній список мусить означати «нічого не горить», а не
        «не вдалось спитати».
        """
        # v0 віддає 410 deprecated_endpoint; список живе тільки в v1.
        data = self._request("GET", "/instances/", params={"owner": "me"}, api_version="v1")
        return [i for i in (data.get("instances") or [])
                if str(i.get("actual_status") or "").lower() in ("running", "loading", "created")]

    def _render_rent_onstart(self, *, autodestroy_hours: float) -> str:
        return _RENT_ONSTART.format(
            root=REMOTE_ROOT,
            deadline_seconds=int(float(autodestroy_hours) * 3600),
            api_base=API_BASE,
        )

    def status(self, handle: JobHandle) -> StatusReport:
        inst = self._instance(handle)
        if inst is None:
            return StatusReport(
                status=JobStatus.UNKNOWN,
                message=f"instance {handle.remote_id} no longer exists (destroyed?)",
            )

        actual = str(inst.get("actual_status") or inst.get("cur_state") or "").lower()
        cost = _accrued_cost(inst)
        note = f"{inst.get('gpu_name', '?')} · ${inst.get('dph_total', 0):.3f}/h · spent ≈ ${cost:.2f}"

        if actual in ("created", "loading", "scheduling"):
            return StatusReport(
                status=JobStatus.QUEUED,
                message=f"instance {actual} ({inst.get('status_msg') or 'booting'}) · {note}",
            )

        raw = self._read_status(handle)
        if raw is None:
            if actual in ("exited", "offline", "stopped"):
                return StatusReport(
                    status=JobStatus.FAILED,
                    error=f"instance {actual} before the job reported anything · {note}",
                )
            return StatusReport(status=JobStatus.QUEUED, message=f"container up, job starting · {note}")

        state = str(raw.get("status", "")).lower()
        if state == "completed":
            return StatusReport(
                status=JobStatus.COMPLETED,
                message=(
                    f"{raw.get('message') or 'done'} · {note} — "
                    f"⚠ still billing until destroyed: gpurunner cancel {handle.id[:8]}"
                ),
            )
        if state == "failed":
            return StatusReport(
                status=JobStatus.FAILED,
                message=f"{raw.get('message') or ''} · {note}",
                error=raw.get("error"),
            )
        if state in ("running", "starting"):
            beat = _parse_ts(raw.get("heartbeat") or raw.get("ts"))
            if beat is not None and datetime.now(tz=UTC) - beat > _HEARTBEAT_STALE:
                return StatusReport(
                    status=JobStatus.UNKNOWN,
                    message=f"heartbeat stale — job may have died, instance alive · {note}",
                )
            return StatusReport(status=JobStatus.RUNNING, message=f"{raw.get('message') or ''} · {note}")
        return StatusReport(status=JobStatus.UNKNOWN, message=f"unrecognized job status {state!r} · {note}")

    def fetch_outputs(self, handle: JobHandle, out_dir: Path,
                      *, deadline_sec: float = FETCH_DEADLINE_SEC,
                      only_meta: bool = False) -> list[Path]:
        """Забрати вивід. З ТАЙМАУТАМИ — інакше це найтихіша пастка системи.

        🔴🔴 Тут не було ні keepalive, ні таймауту читання, ні спільного
        дедлайну. Завмерле TCP-з'єднання посеред качання блокує `sftp.get()`
        НАЗАВЖДИ: наглядач лишається живий, але стоїть у читанні — стан не
        оновлюється, бокс тарифікується, і зовні це виглядає як «працює».
        Заміряно 2026-08-12 ДВІЧІ за ніч, у двох незалежних кампаніях: обидва
        наглядачі завмерли саме на `fetching` (01:10 і 01:34) і простояли 4.8 і
        4.4 години; в одній із них устигла піднятись друга оренда, тож горіли
        дві машини. `htr state` увесь цей час показував старі числа — не тому,
        що бреше, а тому, що його нема кому оновити.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        client = self._ssh(handle)
        failures: list[str] = []
        stop_at = time.monotonic() + max(60.0, float(deadline_sec))
        try:
            transport = client.get_transport()
            if transport is not None:
                # Мертвого сусіда видно за півхвилини, а не ніколи.
                transport.set_keepalive(30)
            sftp = client.open_sftp()
            # 🔴 СТОП-КРАН на час забору. Самознищення заводиться на голий
            # `sleep` одразу після job'а (дефолт 30 хв), а власна стеля забору —
            # година: черга з десятками тисяч дрібних файлів (`*.txt` і
            # `*.lines.json` по 17 КБ на кадр) законно тягнеться довше. Бокс
            # гасив себе ПОСЕРЕД качання, забір падав, і повністю прорахована
            # оплачена черга лишалась тільки в чекпоінтах. Сторож питав
            # годинник; тепер він питає, чи ми зараз качаємо.
            with contextlib.suppress(Exception):
                sftp.open(f"{REMOTE_ROOT}/NOSTOP", "w").close()
            try:
                chan = sftp.get_channel()
                if chan is not None:
                    chan.settimeout(FETCH_READ_TIMEOUT_SEC)
                # `only_meta`: тіло результату вже забрали з R2 одним
                # об'єктом, лишились службові файли — десяток замість тисяч.
                written = [] if only_meta else self._download_tree(
                    sftp, REMOTE_WORKING, out_dir, failures=failures, stop_at=stop_at)
                if failures:
                    print(f"[vast] ⚠ не забрано {len(failures)} файл(ів): "
                          f"{failures[0]}" + (f" (і ще {len(failures) - 1})"
                                              if len(failures) > 1 else ""), flush=True)
                for name in ("_runner.log", "_status.json", "_progress.json"):
                    target = out_dir / name
                    try:
                        sftp.get(f"{REMOTE_ROOT}/{name}", str(target))
                        written.append(target)
                    except OSError:
                        continue  # job may not have written it (yet)
                if only_meta:
                    # 🔴 Логи шардів — теж. Тіло прийшло з R2, а чекпоінт логів не
                    # везе: 11.09.2026 (904-24-198) причина 34 збоїв згоріла
                    # разом із боксом. Кілька КБ на шард.
                    written += self._fetch_shard_logs(sftp, out_dir)
                return written
            finally:
                # Кран знімаємо ЗАВЖДИ: лишити його означало б бокс, який не
                # погасить себе ніколи (у самому шаблоні це названо
                # «STILL BILLING»).
                with contextlib.suppress(Exception):
                    sftp.remove(f"{REMOTE_ROOT}/NOSTOP")
                sftp.close()
        finally:
            client.close()

    @staticmethod
    def _fetch_shard_logs(sftp: Any, out_dir: Path) -> list[Path]:
        """Логи шардів кожної справи боксу: `<working>/<справа>/logs/*.log`.

        Розкладка та сама, що в `_download_tree` (`REMOTE_WORKING` → `out_dir`).
        Best-effort: чого немає або що не читається — пропускається.
        """
        got: list[Path] = []
        try:
            cases = sftp.listdir(REMOTE_WORKING)
        except OSError:
            return got
        for case in cases:
            remote = f"{REMOTE_WORKING}/{case}/logs"
            try:
                names = sftp.listdir(remote)
            except OSError:
                continue
            for name in names:
                if not name.endswith(".log"):
                    continue
                target = out_dir / case / "logs" / name
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(f"{remote}/{name}", str(target))
                except OSError:
                    continue
                got.append(target)
        return got

    def balance(self) -> BalanceReport:
        """Prepaid credit left on the account, plus what running instances burn."""
        data = self._request("GET", "/users/current")
        credit = data.get("credit", data.get("balance"))
        burn = 0.0
        try:
            # v0 віддає 410 deprecated_endpoint; список живе тільки в v1.
            # Конкретний інстанс (`/instances/<id>/`) у v0 працює й далі.
            instances = self._request(
                "GET", "/instances/", params={"owner": "me"}, api_version="v1"
            )
            rows = instances.get("instances") or []
            burn = sum(
                float(i.get("dph_total") or 0)
                for i in rows
                if str(i.get("actual_status") or "").lower() in ("running", "loading")
            )
        except BackendError:
            pass
        detail = "prepaid credit"
        if burn:
            detail += f" · ⚠ {burn:.3f} $/h витрачається просто зараз (живі інстанси)"
        return BalanceReport(
            backend=self.name,
            available=float(credit) if credit is not None else None,
            unit="$",
            detail=detail,
            url="https://cloud.vast.ai/billing/",
        )

    def cancel(self, handle: JobHandle, *, force: bool = False) -> None:
        """Знищити інстанс. Це ЄДИНЕ, що спиняє тарифікацію.

        🔴🔴 Перевірка власності стоїть ТУТ, а не лише в CLI. Спершу вона жила
        тільки в команді `gpurunner cancel` — і це захист не на тому рівні:
        будь-який код, що збере `JobHandle` руками й покличе цей метод, іде
        повз неї. 2026-08-11 автор цього захисту сам так і зробив, намагаючись
        добити «вцілілий» інстанс (який насправді вже гасився — список Vast
        віддає стан із затримкою). Захист, який можна обійти зсередини на
        одному рядку, захистом не є.

        `force=True` дозволено рівно там, де провенанс доведений іншим
        способом: гасіння сироти, чий `instance_id` ми щойно дістали з винятку
        власної оренди.

        🔴 Фолбек на v1 обов'язковий. `API_BASE` вказує на v0, а Vast уже
        перевів частину ендпоінтів на v1 (список інстансів на v0 віддає
        410 deprecated). 2026-08-11 тричі прилетів 404 на живий інстанс — і
        кожен раз ми лишали бокс горіти, бо запасного шляху не було.
        """
        if not force:
            self._assert_ours(handle)
        last: BackendError | None = None
        for version in ("v0", "v1"):
            try:
                self._request(
                    "DELETE", f"/instances/{handle.remote_id}/",
                    json={}, api_version=version,
                )
                return
            except BackendError as e:
                status = getattr(e, "status_code", None)
                # 404 після успішного знищення — норма (його вже немає).
                if status == 404 and version == "v1":
                    return
                if status in (404, 410) or status is None:
                    last = e
                    continue
                raise
        if last is not None:
            raise last

    def logs(self, handle: JobHandle) -> list[str]:
        try:
            client = self._ssh(handle)
        except (AuthError, BackendError) as e:
            return [f"(no SSH to instance {handle.remote_id}: {e})", *self._container_logs(handle)]
        try:
            sftp = client.open_sftp()
            try:
                with sftp.open(f"{REMOTE_ROOT}/_runner.log") as fh:
                    return fh.read().decode("utf-8", errors="replace").splitlines()
            except OSError:
                return [
                    "(no _runner.log yet — falling back to container logs)",
                    *self._container_logs(handle),
                ]
            finally:
                sftp.close()
        finally:
            client.close()

    def wait_for_log_line(
        self, handle: JobHandle, needle: str, *, timeout: int = 1800, poll: int = 60
    ) -> str | None:
        """Poll the remote log until a line contains ``needle``. None on timeout."""
        start = time.monotonic()
        while True:
            try:
                for line in self.logs(handle):
                    if needle in line:
                        return line
            except BackendError:
                pass
            if time.monotonic() - start > timeout:
                return None
            time.sleep(poll)

    # ---- marketplace ------------------------------------------------------

    def search_offers(
        self,
        *,
        gpu: str,
        max_price: float | None = None,
        disk_gb: int = DEFAULT_DISK_GB,
        num_gpus: int = 1,
        limit: int = 20,
        min_cpu: float = 0.0,
        min_ram_gb: float = 0.0,
        min_reliability: float = 0.0,
        min_vram_gb: float = 0.0,
        min_inet_down: float = DEFAULT_MIN_INET_DOWN,
        machine_ids: Iterable[int] | None = None,
        exclude_machine_ids: Iterable[int] | None = None,
        order: list[list[str]] | None = None,
        min_cuda: float = 12.1,
    ) -> list[dict[str, Any]]:
        """Rentable offers matching the filters, cheapest first by default.

        🔴 ``min_cpu`` — не косметика. Найдешевший оффер із потрібною картою
        регулярно має 4 ефективні ядра, а CPU-bound job (як ``htr_case``: 74%
        часу сторінки — геометрія kraken на процесорі) на такій машині працює
        ПОВІЛЬНІШЕ за домашній ноутбук. Замір 2026-08-09 на Beam: 2 ядра/шард
        дали 38.3 с/стор проти 17.5 с на 6 ядрах. Без цього фільтра «взяли
        найдешевше» означає «взяли найгірше».

        🔴 ``min_vram_gb`` — друга половина тієї самої помилки. VRAM визначає,
        СКІЛЬКИ ШАРДІВ підняти (~2.6 ГБ на шард у піку), тобто скільки з
        оплачених ядер узагалі працюватиме. Фільтра тут не було, і 2026-08-11
        це дало бокс із 16 ГБ під 8 шардів: CUDA OOM тихо з'їв 46 сторінок при
        ``rc=0``. Карту за назвою вгадувати не можна — у RTX 4060 Ti буває 8 і
        16 ГБ, у RTX 4090 — 23 і 24.

        ``machine_ids`` / ``exclude_machine_ids`` — адресний пошук доброї
        машини й чорний список поганої (`core.boxes`). Виключення дублюється
        клієнтським відсівом: серверний фільтр міг бути проігнорований мовчки,
        а мовчазний бан — це той самий поганий бокс удруге.
        """
        exclude = {int(m) for m in (exclude_machine_ids or [])}
        query: dict[str, Any] = {
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            # `gte`, а не `eq`: 2/4/8-карткові бокси тепер придатні — шарди
            # розкладаються по всіх картах. `num_gpus=1` означає «щонайменше
            # одна», а не «рівно одна».
            "num_gpus": {"gte": int(num_gpus)},
            "disk_space": {"gte": float(disk_gb)},
            "cuda_max_good": {"gte": float(min_cuda)},
            "order": order or [["dph_total", "asc"]],
            "type": "on-demand",
            "limit": int(limit),
            "allocated_storage": float(disk_gb),
        }
        if min_inet_down > 0:
            # Груба відсічка мертвих хостів. ⚠ Заявленому числу вірити не можна
            # (оффер із 755 Mbps віддавав 0.5) — канал міряється на самому боксі
            # перед заливкою, `probe_box`.
            query["inet_down"] = {"gte": float(min_inet_down)}
        if min_cpu:
            query["cpu_cores_effective"] = {"gte": float(min_cpu)}
        if min_ram_gb:
            query["cpu_ram"] = {"gte": float(min_ram_gb) * 1024.0}
        if min_vram_gb:
            # `gpu_ram` — мегабайти НА КАРТУ.
            query["gpu_ram"] = {"gte": float(min_vram_gb) * 1024.0}
        if min_reliability:
            # Надійність хоста корелює з тим, чи доживе добовий прогін до кінця.
            query["reliability2"] = {"gte": float(min_reliability)}
        vast_gpu = _GPU_TO_VAST[gpu]
        if vast_gpu:
            query["gpu_name"] = {"eq": vast_gpu}
        # 🔴🔴 ТВЕРДА стеля, яку не обійде жоден план. Дефолт у `Plan` можна
        # перебити (або взяти старий план, згенерований до появи ручки) — і
        # саме так на ринку бралися картки по $0.50/год. Рішення користувача
        # 2026-08-12: жодної дорожчої за $0.365/год, крапка. Стеля живе в
        # ЗАПИТІ ДО РИНКУ, тобто дорогі оффери навіть не потрапляють у добір.
        # 🔴 Стеля — НА КАРТУ (06.09.2026): бокс із двома картами коштує
        # вдвічі, але й дає вдвічі (шарди розкладаються по картах). Стеля на
        # бокс різала 2×3090 за $0.40 із видачі, хоч на карту це $0.20 — і
        # двокартковий замір не міг відбутись узагалі.
        cap = min(float(max_price), ABSOLUTE_MAX_DPH) if max_price else ABSOLUTE_MAX_DPH
        query["dph_total"] = {"lte": cap * max(1, int(num_gpus or 1))}
        if machine_ids:
            query["machine_id"] = {"in": sorted({int(m) for m in machine_ids})}
        elif exclude and len(exclude) <= _MAX_NOTIN:
            # Довгий `notin` ризикує впертись у ліміт запиту; понад стелю
            # покладаємось лише на клієнтський відсів нижче.
            query["machine_id"] = {"notin": sorted(exclude)}
        resp = self._request("POST", "/bundles/", json=query)
        offers = list(resp.get("offers") or [])
        if exclude:
            offers = [o for o in offers if int(o.get("machine_id") or 0) not in exclude]
        return offers

    def pick_offer(
        self,
        *,
        gpu: str,
        max_price: float | None,
        disk_gb: int,
        num_gpus: int,
        min_cpu: float = 0.0,
        min_ram_gb: float = 0.0,
        min_reliability: float = 0.0,
        min_vram_gb: float = 0.0,
        min_cuda: float = 12.1,
    ) -> dict[str, Any]:
        """Найдешевший придатний оффер — простий шлях для не-HTR job'ів.

        ⚠ Для HTR користуйся `find_candidates`: там ранжування за виміряною
        продуктивністю, пам'ять про погані машини й деградація порогів.
        Тут — «найдешевше, що проходить фільтри», і саме ця простота колись
        дала бокс із 4 ядрами.
        """
        offers = self.search_offers(
            gpu=gpu, max_price=max_price, disk_gb=disk_gb, num_gpus=num_gpus,
            min_cpu=min_cpu, min_ram_gb=min_ram_gb, min_reliability=min_reliability,
            min_vram_gb=min_vram_gb, min_cuda=min_cuda,
            exclude_machine_ids=_banned_machines(),
        )
        if not offers:
            raise BackendError(
                f"no rentable {gpu} offer with ≥{disk_gb} GB disk"
                + (f", ≥{min_cpu:g} cpu cores" if min_cpu else "")
                + (f", ≥{min_ram_gb:g} GB RAM" if min_ram_gb else "")
                + (f" under ${max_price}/h" if max_price else "")
                + ". Try another --gpu, a higher -p max_price, or a smaller -p disk/min_cpu."
            )
        return offers[0]

    def find_candidates(
        self,
        *,
        gpu: str,
        need: Need,
        num_gpus: int = 1,
        max_price: float | None = None,
        limit: int = 40,
        now: datetime | None = None,
        min_compute_cap: float = 0.0,
    ) -> Selection:
        """Кандидати на оренду: пам'ять реєстру + ранжування + деградація.

        `min_compute_cap` — найстаріша архітектура карти, яку потягне середовище
        замовника (7.0 = Volta/Turing і новіше); 0 — не відсівати. 🔴 Справжня
        оренда 19.09.2026: найдешевші машини ринку — Pascal і Maxwell (5.2–6.1),
        а колеса torch, які ставить Нишпорка, під них ядер не мають — «CUDA
        error: no kernel image», нуль сторінок за оплачений підйом. Оффер без
        поля `compute_cap` при заданому мінімумі теж відсівається: перевірити
        нічим — не орендуємо.

        Три проходи, у такому порядку:

        1. **Адресно по зірках** — `machine_id.eq` окремим запитом на кожну
           машину, яка вже відпрацювала. Це прямий сенс реєстру: «був добрий
           дешевий бокс — знайди саме його», а не «шукай схожий».
        2. **Ринок без забанених** — звичайний пошук із `machine_id.notin`
           (і клієнтським відсівом, бо серверний фільтр міг бути мовчки
           проігнорований).
        3. **Деградація порогів** — усередині `select_offers`: ціна, далі
           VRAM, далі ядра. Бюджет і строк не послаблюються ніколи.

        Порожній результат означає «не орендуємо нічого» — і це правильна
        відповідь, а не привід узяти будь-що.
        """
        verdicts_map = boxes.verdicts(now=now)
        banned = {mid for mid, v in verdicts_map.items() if v.banned}

        pool: dict[int, dict[str, Any]] = {}
        for star in boxes.starred(gpu=None, limit=8, now=now):
            try:
                found = self.search_offers(
                    gpu=gpu, max_price=max_price, disk_gb=need.disk_gb, num_gpus=num_gpus,
                    limit=4, machine_ids=[star.machine_id],
                )
            except BackendError as e:
                print(f"[vast] ⚠ зірку {star.machine_id} не опитати: {e}", flush=True)
                continue
            for offer in found:
                pool[int(offer.get("id") or 0)] = offer
            if found:
                print(f"[vast] ★ машина {star.machine_id} вільна ({star.reason})", flush=True)

        # 🔴 Два запити, а не один. Видача сортується за ціною ЗРОСТАННЯМ, тож
        # «перші 40» — це сорок НАЙДЕШЕВШИХ машин, тобто рівно той хвіст, який
        # не проходить порогів (замір: із 40 таких проходив один). Перший запит
        # ставить підлогу по залізу на боці API — і повертає сорок придатних;
        # другий, без підлоги, лишається на випадок деградації, коли доводиться
        # брати гірше.
        floor_cores = TIERS[0].min_cores
        floor_vram = TIERS[0].min_shards * need.gb_per_shard
        # 🔴🔴 Три проходи, і ПЕРШИЙ — за ЯДРАМИ СПАДАННЯМ. Видача Vast
        # сортується за ціною зростанням, тобто «перші 40» = сорок
        # НАЙДЕШЕВШИХ, і багатоядерні машини не потрапляли у видачу взагалі.
        # А ядра — це і є головна ручка швидкості: замір 2026-08-12, та сама
        # RTX 3090 і ті самі 8 шардів, 64 ядра дали 2548 стор/год, 32 ядра —
        # 703. Тобто добір дивився на найдешевший хвіст ринку й ніколи не
        # бачив швидких машин, які коштують на копійки більше.
        for min_cpu, min_vram, take, order in (
            (floor_cores, floor_vram, limit, [["cpu_cores_effective", "desc"]]),
            (floor_cores, floor_vram, limit, None),
            (0.0, 0.0, max(10, limit // 2), None),
        ):
            try:
                found = self.search_offers(
                    gpu=gpu, max_price=max_price, disk_gb=need.disk_gb, num_gpus=num_gpus,
                    limit=take, min_cpu=min_cpu, min_vram_gb=min_vram,
                    exclude_machine_ids=banned, order=order,
                )
            except BackendError as e:
                print(f"[vast] ⚠ запит ринку не вдався: {e}", flush=True)
                continue
            for offer in found:
                pool[int(offer.get("id") or 0)] = offer

        offers = list(pool.values())
        if min_compute_cap > 0:
            kept = [o for o in offers if offer_compute_cap(o) >= min_compute_cap]
            if len(kept) < len(offers):
                print(f"[vast] відсіяно {len(offers) - len(kept)} офферів: архітектура "
                      f"карти старша за {min_compute_cap:g}", flush=True)
            offers = kept
        return select_offers(offers, need, verdicts_map)

    # ---- remote side ------------------------------------------------------

    def _render_job_py(self, body: str) -> str:
        """Повний модуль, що поїде на бокс ФАЙЛОМ (не через onstart — див. _ONSTART)."""
        return _RUNNER_WRAPPER.replace("__RUNNER_SRC__", json.dumps(body, ensure_ascii=False))

    def _render_onstart(self, *, max_hours: int, autodestroy_hours: float) -> str:
        """Bash that prepares the Kaggle-shaped FS and waits for its inputs.

        The job source itself is **not** in here any more: an onstart script is
        embedded in a JSON payload and then re-quoted by the container's shell,
        so inlining arbitrary source is a quoting minefield (it used to travel
        base64-encoded for exactly that reason). It now goes over the same SFTP
        session as the inputs — one transport, size-verified — and this script
        only waits for the `GO` sentinel that the upload touches.
        """
        destroy = ""
        if autodestroy_hours > 0:
            destroy = _AUTODESTROY.format(
                grace_seconds=int(autodestroy_hours * 3600),
                api_base=API_BASE,
                root=REMOTE_ROOT,
            )
        return _ONSTART.format(
            root=REMOTE_ROOT,
            working=REMOTE_WORKING,
            inputs=REMOTE_INPUT,
            max_seconds=int(max_hours * 3600),
            deadline_guard=_DEADLINE_GUARD.format(
                deadline_seconds=int(max_hours * 3600) + _DEADLINE_GRACE_SEC,
                api_base=API_BASE,
                root=REMOTE_ROOT,
            ),
            autodestroy=destroy,
        )

    def _upload_text(self, client: Any, remote_path: str, text: str) -> None:
        """Записати текстовий файл на бокс і звірити розмір: мовчки обрізаний
        job.py упав би вже на оплачуваній машині, і то синтаксичною помилкою."""
        blob = text.encode("utf-8")
        self._exec(client, f"mkdir -p {posixpath.dirname(remote_path)}")
        sftp = client.open_sftp()
        try:
            with sftp.open(remote_path, "wb") as fh:
                fh.write(blob)
            landed = sftp.stat(remote_path).st_size
        finally:
            sftp.close()
        if landed != len(blob):
            raise BackendError(f"{remote_path}: uploaded {landed} B, expected {len(blob)} B")

    def _upload_inputs(self, client: Any, inputs: dict[str, Path]) -> None:
        if not inputs:
            return
        sftp = client.open_sftp()
        try:
            for slug, local in inputs.items():
                remote_dir = f"{REMOTE_INPUT}/{slug}"
                self._exec(client, f"mkdir -p {remote_dir}")
                files = [p for p in sorted(local.iterdir()) if p.is_file()]
                if not files:
                    raise BackendError(f"input dir {local} has no files")
                for p in files:
                    sftp.put(str(p), f"{remote_dir}/{p.name}")
                    remote_size = sftp.stat(f"{remote_dir}/{p.name}").st_size
                    if remote_size != p.stat().st_size:
                        raise BackendError(
                            f"{p.name}: uploaded {remote_size} B, expected {p.stat().st_size} B"
                        )
        finally:
            sftp.close()

    def _download_tree(
        self, sftp: Any, remote_dir: str, out_dir: Path, *,
        failures: list[str] | None = None, stop_at: float | None = None,
    ) -> list[Path]:
        """Забрати дерево. Один непрочитаний файл НЕ валить увесь забір.

        🔴 Виміряно 2026-08-11 на живому прогоні: `sftp.get` упав із
        `FileNotFoundError` на п'ятому файлі (запис на боксі ще тривав), виняток
        пройшов крізь увесь `fetch`, наглядач вважав це крахом і **знищив бокс,
        на якому лежали всі 391 готові сторінки**. Робота вціліла лише тому, що
        паралельно писались чекпоінти в R2.

        Тепер збій окремого файла збирається у `failures`, а решта дерева
        доїжджає. Неповноту потім однаково спіймають ворота — але вже маючи на
        руках те, що вдалось забрати.
        """
        written: list[Path] = []
        try:
            entries = sftp.listdir_attr(remote_dir)
        # 🔴 `OSError` мало. Розрив каналу посеред багатогодинного забору
        # paramiko віддає `SSHException`/`EOFError` — жоден не `OSError`, тож
        # виняток проходив НАСКРІЗЬ повз усю цю оборону, і забір валився
        # цілком. Рівно та поломка, проти якої написано коментар вище, тільки
        # для іншого класу винятків.
        except Exception as e:
            if failures is not None:
                failures.append(f"{remote_dir}/: {e}")
            return written
        for entry in entries:
            # Дедлайн перевіряється МІЖ ФАЙЛАМИ: краще віддати наглядачу те, що
            # встигли, ніж стояти в читанні, поки тарифікація йде.
            if stop_at is not None and time.monotonic() > stop_at:
                if failures is not None:
                    failures.append(f"{remote_dir}/: вичерпано стелю часу на забір")
                break
            remote = posixpath.join(remote_dir, entry.filename)
            target = out_dir / entry.filename
            if stat.S_ISDIR(entry.st_mode or 0):
                target.mkdir(parents=True, exist_ok=True)
                written.extend(self._download_tree(sftp, remote, target,
                                                   failures=failures, stop_at=stop_at))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                sftp.get(remote, str(target))
            except Exception as e:
                if failures is not None:
                    failures.append(f"{remote}: {e}")
                target.unlink(missing_ok=True)
                continue
            written.append(target)
        return written

    def _read_status(self, handle: JobHandle) -> dict[str, Any] | None:
        try:
            client = self._ssh(handle)
        except (AuthError, BackendError):
            return None
        try:
            sftp = client.open_sftp()
            try:
                with sftp.open(f"{REMOTE_ROOT}/_status.json") as fh:
                    data = json.loads(fh.read().decode("utf-8"))
                return data if isinstance(data, dict) else None
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return None
            finally:
                sftp.close()
        finally:
            client.close()

    # ---- SSH --------------------------------------------------------------

    def ssh_endpoint(self, handle: JobHandle) -> dict[str, Any] | None:
        """Куди й чим стукати СТОРОННІМ інструментом (`scp`, `rsync`).

        Потрібне транспорту `box`: системний `scp` везе архів у рази швидше за
        SFTP того самого paramiko (7.56 проти 0.48 МБ/с на живому боксі), але
        він окремий процес і мусить дістати адресу з ключем сам. `None` —
        машина ще не назвала SSH-точки.
        """
        from gpurunner.auth import vast as vast_auth

        inst = self._instance(handle)
        if inst is None:
            return None
        host, port = inst.get("ssh_host"), inst.get("ssh_port")
        if not host or not port:
            return None
        priv, _ = vast_auth.require_ssh_key()
        return {"host": str(host), "port": int(port), "key": str(priv),
                "user": "root"}

    def _ssh(self, handle: JobHandle, *, timeout: int = 30) -> Any:
        from gpurunner.auth import vast as vast_auth

        try:
            import paramiko
        except ImportError as e:
            raise AuthError(
                f"paramiko not installed: {e}\nRun: uv sync --extra vast"
            ) from e
        # 🔴 Поки бокс підіймається, кожен стук у ще мертвий sshd paramiko пише
        # у свій логер ПОВНИЙ трейсбек («Error reading SSH protocol banner»), а
        # без обробників логер Python друкує його просто в stderr. Перша оренда
        # через плагін Нишпорки показала людині п'ять таких трейсбеків поспіль
        # за штатного підйому. Повтори ми ведемо самі й кажемо про них словами.
        import logging

        logging.getLogger("paramiko.transport").setLevel(logging.CRITICAL)

        inst = self._instance(handle)
        if inst is None:
            raise BackendError(f"instance {handle.remote_id} no longer exists")
        host = inst.get("ssh_host")
        port = inst.get("ssh_port")
        if not host or not port:
            raise BackendError(f"instance {handle.remote_id} has no SSH endpoint yet")

        priv, _ = vast_auth.require_ssh_key()
        pkey = _load_private_key(priv)
        client = paramiko.SSHClient()
        # Інстанс щоразу новий і живе годину — постійного host key в нього немає,
        # тож звіряти нема з чим. Ключі НЕ пишуться в ~/.ssh/known_hosts
        # (save_host_keys не викликається), щоб не засмічувати його ефемерними.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=str(host),
                port=int(port),
                username="root",
                pkey=pkey,
                # 🔴 Обидва False — і заради надійності, і заради приватності:
                # інакше paramiko сам обходить ~/.ssh і ssh-agent і ПРОПОНУЄ
                # орендованому чужому боксу всі наявні ключі по черзі. Нам
                # потрібен рівно один, і решта сервера не стосується.
                look_for_keys=False,
                allow_agent=False,
                timeout=timeout,
                banner_timeout=timeout,
                auth_timeout=timeout,
            )
        except Exception as e:
            raise _classify_ssh_error(e, host=str(host), port=int(port)) from e
        # 🔴🔴 Таймаут на САМ СОКЕТ, а не лише на з'єднання. `connect(timeout=…)`
        # обмежує рукостискання й більше нічого: після нього читання блокується
        # НАЗАВЖДИ, якщо бокс зник посеред обміну. Заміряно 17.08.2026 — бокс
        # віддав 404 під час забору, і наглядач висів на мертвому сокеті 30
        # хвилин, доки його не вбили руками; стан увесь цей час показував фазу
        # `fetching` і не старів, бо процес був живий і чемно чекав.
        with contextlib.suppress(Exception):
            transport = client.get_transport()
            if transport is not None:
                transport.sock.settimeout(float(_SFTP_IO_TIMEOUT))
                # Keepalive додатково: мовчазний хост розірве канал сам, а не
                # триматиме нас у стані «з'єднання ніби живе».
                transport.set_keepalive(30)
        return client

    def _wait_for_ssh(self, handle: JobHandle, *, timeout: int = _SSH_WAIT_TIMEOUT) -> Any:
        """Дочекатись робочого SSH — або зрозуміти, що чекати нема сенсу.

        Машина станів, а не «спи й пробуй», бо три різні причини вимагають
        трьох різних реакцій, і дві з них — негайних:

        - **ключ відхилено** → одна повторна спроба (ключ, прив'язаний через
          API, інколи доїжджає з запізненням ~30 с), далі вихід. 🔴 Раніше це
          ретраїлось 15 хвилин як «ще вантажиться».
        - **контейнер не створюється** (`created`/`scheduling` довше
          `_BOOT_STUCK_SEC`) → його не буде ніколи: битий образ, темплейт
          акаунта, CDI. Вихід із `never_booted`.
        - **решта** → бокс справді вантажиться, ретраїмо.
        """
        start = time.monotonic()
        auth_since: float | None = None
        running_since: float | None = None
        rebound_key = False
        last: Exception | None = None
        #: Підпис стану й мить, коли він востаннє змінився. Секундомір
        #: «застряг» рахується ВІД ЦІЄЇ МИТІ, а не від оренди.
        sig: tuple[str, str] = ("", "")
        sig_since = start
        #: Коли востаннє друкували рядок стану `loading`. Під час збирання образу
        #: Vast міняє `status_msg` щосекунди (рядки apt і docker), і перша ж
        #: оренда через плагін Нишпорки висипала людині сотню таких рядків.
        #: Секундомір «застряг» від цього не залежить — він рахує зміни, не друк.
        loading_said = 0.0

        while time.monotonic() - start < timeout:
            try:
                return self._ssh(handle, timeout=15)
            except SshAuthRejected as e:
                last = e
                now = time.monotonic()
                if auth_since is None:
                    auth_since = now
                # 🔴🔴 ТРЕТІЙ таймер у цій функції, і найшкідливіший: він
                # виносить `ssh_auth_denied` — найсуворіший вердикт реєстру.
                # Поки контейнер ще ставить пакети, SSH-демон уже відповідає, а
                # наш ключ у `authorized_keys` ще не потрапив: Vast прокидає
                # його не миттєво. Дев'яносто секунд на це — при боксах, що
                # піднімаються по 204 і 511 с, — означало звинувачувати
                # СПРАВНІ машини. Заміряно двічі: 144417 і 97081, обидві з
                # УСПІШНИМ прогоном напередодні («391 з 391 сторінок»), назавтра
                # дістали «хост відхилив наш ключ».
                # Тому: поки хост посувається, лічильник починається заново, і
                # ключ прив'язується ще раз — це дешево й ідемпотентно.
                now_sig = self._status_signature(handle)
                if now_sig != sig:
                    sig, sig_since, auth_since = now_sig, now, now
                    if now_sig[1]:
                        print(f"[vast] {now_sig[0]}: {now_sig[1][:90]}", flush=True)
                    with contextlib.suppress(BackendError):
                        self._attach_ssh_key(handle)
                    time.sleep(10)
                    continue
                if not rebound_key:
                    rebound_key = True
                    self._attach_ssh_key(handle)
                    time.sleep(10)
                    continue
                if now - auth_since > _SSH_AUTH_GRACE:
                    raise
                time.sleep(10)
            except (AuthError, BackendError) as e:
                last = e
                # Контейнер уже піднявся, а SSH не відповідає — це не «ще
                # вантажиться». Даємо йому коротке окреме вікно й виходимо.
                if self._container_running(handle):
                    now = time.monotonic()
                    if running_since is None:
                        running_since = now
                        print("[vast] контейнер піднявся; чекаю SSH ще "
                              f"{_SSH_AFTER_RUNNING_SEC}s", flush=True)
                    # 🔴🔴 І ТУТ теж дивимось на ПОСУВАННЯ. Спершу я зробив це
                    # лише для фази образу — і цього виявилось мало: Vast
                    # ставить `running`, щойно контейнер створено, а `onstart`
                    # у цей час ще ставить пакети (`Get:85 …libstemmer0d`).
                    # Тобто найдовша частина підйому відбувається ВЖЕ в стані
                    # `running`, і секундомір її не бачив. Ціна помилки
                    # виміряна 2026-08-11: убито машину 135791 — RTX 3090 з
                    # **80 ядрами за $0.017/год**, тобто найкраще, що ринок дав
                    # за весь день (≈$0.006 за тисячу сторінок), — на 511-й
                    # секунді, коли в статусі йшов живий лічильник apt.
                    elif (now_sig := self._status_signature(handle)) != sig:
                        if now_sig[1]:
                            print(f"[vast] {now_sig[0]}: {now_sig[1][:90]}", flush=True)
                        sig, sig_since, running_since = now_sig, now, now
                    elif now - running_since > _SSH_AFTER_RUNNING_SEC:
                        dead = BackendError(
                            f"контейнер {handle.remote_id} працює вже "
                            f"{now - running_since:.0f}s, але SSH не відповідає: {e}. "
                            f"Це хост (порт закритий/проксі не піднявся), не наш ключ."
                        )
                        dead.outcome = "ssh_unreachable"
                        raise dead from e
                # Підпис змінився — хост посувається (тягне шари образу,
                # розпаковує). Секундомір починається заново.
                now_sig = self._status_signature(handle)
                if fatal := _fatal_status(now_sig[1]):
                    doomed = BackendError(
                        f"хост не може завантажити образ і сам це не полагодить: "
                        f"«{now_sig[1][:160]}» (ознака: {fatal}). Це мережа ХОСТА до "
                        f"реєстру образів — чекати нема чого, беремо іншу машину."
                    )
                    doomed.outcome = "never_booted"
                    raise doomed from e
                if now_sig != sig:
                    quiet = (now_sig[0] == "loading"
                             and time.monotonic() - loading_said < 30.0)
                    if now_sig[1] and now_sig != ("", "") and not quiet:
                        print(f"[vast] {now_sig[0]}: {now_sig[1][:90]}", flush=True)
                        if now_sig[0] == "loading":
                            loading_said = time.monotonic()
                    sig, sig_since = now_sig, time.monotonic()
                if time.monotonic() - sig_since > _BOOT_STUCK_SEC and self._boot_stuck(handle):
                    state = self._instance_status(handle)
                    stuck = BackendError(
                        f"instance {handle.remote_id} за {_BOOT_STUCK_SEC}s БЕЗ ЖОДНОГО "
                        f"зрушення не дала контейнера (стан «{state}», останнє "
                        f"повідомлення Vast: «{sig[1] or '—'}»). Найчастіше — на хості немає "
                        f"нашого образу в кеші, і ми платимо за його викачування "
                        f"(Vast: «від хвилин до годин, якщо не закешований»). "
                        f"Остання помилка SSH: {e}"
                    )
                    # `loading` — це не поломка машини, а холодний кеш образу;
                    # `created`/`scheduling` — контейнер не створюється взагалі.
                    stuck.outcome = "slow_boot" if state == "loading" else "never_booted"
                    raise stuck from e
                time.sleep(10)

        timed_out = BackendError(f"instance never became SSH-reachable within {timeout}s: {last}")
        timed_out.outcome = "ssh_unreachable"
        raise timed_out

    def _attach_ssh_key(self, handle: JobHandle) -> None:
        """Прив'язати наш публічний ключ до живого інстансу (ідемпотентно)."""
        from gpurunner.auth import vast as vast_auth

        _, ssh_pub = vast_auth.require_ssh_key()
        self._request(
            "POST",
            f"/instances/{handle.remote_id}/ssh/",
            json={"ssh_key": ssh_pub.read_text(encoding="utf-8").strip()},
        )

    def _status_signature(self, handle: JobHandle) -> tuple[str, str]:
        """(стан, повідомлення) від Vast — ознака ПОСУВАННЯ, а не лише стану.

        🔴 Vast увесь час розповідає, що робить хост: `Pulling from
        pytorch/pytorch`, `Downloading`, `Extracting`, `success`. Ми це
        читали лише щоб надрукувати, а рішення ухвалювали за голим
        секундоміром — тож бокс, який СУМЛІННО тягне образ, помирав за тим
        самим порогом, що й намертво завислий. Поки підпис МІНЯЄТЬСЯ, робота
        йде, і вбивати нема за що.
        """
        try:
            inst = self._instance(handle)
        except BackendError:
            return "", ""
        if inst is None:
            return "", ""
        actual = str(inst.get("actual_status") or inst.get("cur_state") or "").lower()
        return actual, str(inst.get("status_msg") or "").strip()

    def _assert_ours(self, handle: JobHandle) -> None:
        """Довести право знищити цей інстанс — або відмовитись.

        Два незалежні докази, і досить будь-якого: запис у реєстрі прогонів на
        наше ім'я, або мітка інстансу, що несе id нашого хендла. Мітка сама по
        собі НЕ доказ власності (обидві сесії пишуть `gpurunner-htr_case-…`),
        тому звіряється саме id, а не префікс.
        """
        from gpurunner.core import manifest

        owner = manifest.current_owner()
        try:
            known = manifest.get(handle.id)
        except Exception:
            known = None
        if known is not None and (known.owner or "") == owner and owner:
            return
        if known is not None and not (known.owner or ""):
            return  # нічий запис із давніх заходів — наш же реєстр
        label = ""
        try:
            inst = self._instance(handle)
            label = str((inst or {}).get("label") or "")
        except BackendError:
            label = ""
        if label and handle.id[:8] and handle.id[:8] in label:
            return
        raise ForeignInstance(
            f"інстанс {handle.remote_id} не підтверджено як наш: у реєстрі "
            f"{'немає запису' if known is None else 'власник ' + (known.owner or '—')}, "
            f"мітка «{label or '—'}», наш власник «{owner or 'не заданий'}». "
            f"Гасити не буду — на тому боксі може йти чужа робота."
        )

    def _instance_status(self, handle: JobHandle) -> str:
        """Сирий стан інстансу одним словом ("" якщо не дізнатись)."""
        try:
            inst = self._instance(handle)
        except BackendError:
            return ""
        if inst is None:
            return ""
        return str(inst.get("actual_status") or inst.get("cur_state") or "").lower()

    def _container_running(self, handle: JobHandle) -> bool:
        """Чи Vast уже вважає контейнер запущеним."""
        try:
            inst = self._instance(handle)
        except BackendError:
            return False
        if inst is None:
            return False
        actual = str(inst.get("actual_status") or inst.get("cur_state") or "").lower()
        return actual == "running"

    def _boot_stuck(self, handle: JobHandle) -> bool:
        """Чи інстанс досі не дав робочого контейнера.

        `loading` включено СВІДОМО: це «хост тягне образ», і саме воно
        коштувало 9 хвилин оренди за нуль роботи. Машина не зламана — просто
        без нашого образу в кеші; інша віддасть SSH за 30 секунд.
        """
        try:
            inst = self._instance(handle)
        except BackendError:
            return False
        if inst is None:
            return False
        actual = str(inst.get("actual_status") or inst.get("cur_state") or "").lower()
        return actual in ("created", "scheduling", "loading")

    def probe_box(self, handle: JobHandle, *, net_probe_url: str = "") -> dict[str, Any]:
        """Заміряти РЕАЛЬНЕ залізо боксу — одним викликом, ~15-20 с.

        🔴 Порядок тут важливіший за вміст. Раніше єдиний замір (канал) жив
        усередині job'а, тобто після встановлення kraken/torch — і за
        «оффер обіцяв 755 Мбіт/с, віддає 0.5» ми платили спершу вісьмома
        хвилинами pip. Тепер це відбувається ДО заливки даних і ДО pip: ціна
        помилкового оффера падає з години до півхвилини.

        Міряється `memory.free`, а не `memory.total`: сусід на тій самій карті
        робить «16 ГБ» ще меншими, а шарди рахуються саме з вільного.

        Канал міряється з ТОГО САМОГО джерела, звідки поїдуть дані — інакше
        це вимір «до Cloudflare», а не «до нашого архіву».
        """
        script = _PROBE_SCRIPT.replace("__NET_URL__", net_probe_url or "")
        client = self._ssh(handle, timeout=60)
        try:
            _, stdout, _ = client.exec_command(script, timeout=_PROBE_TIMEOUT)
            stdout.channel.recv_exit_status()
            raw = stdout.read().decode("utf-8", errors="replace")
        finally:
            client.close()

        return self.parse_probe(raw)

    @staticmethod
    def parse_probe(raw: str) -> dict[str, Any]:
        """`key=value` рядки проби → словник; повторний ключ перемагає."""
        probe: dict[str, Any] = {}
        for line in raw.splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not key:
                continue
            try:
                probe[key] = float(value) if _NUMERIC_PROBE_KEYS & {key} else value
            except ValueError:
                probe[key] = value

        probe["vram_total_gb"] = float(probe.get("vram_total_mb") or 0) / 1024.0
        probe["vram_free_gb"] = float(probe.get("vram_free_mb") or 0) / 1024.0
        probe["net_mbps"] = float(probe.get("net_bps") or 0) * 8 / 1_000_000
        return probe

    @staticmethod
    def _exec(client: Any, command: str) -> str:
        _, stdout, stderr = client.exec_command(command, timeout=120)
        rc = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        if rc != 0:
            err = stderr.read().decode("utf-8", errors="replace")
            raise BackendError(f"remote command failed ({rc}): {command}\n{err}")
        return out

    # ---- REST -------------------------------------------------------------

    def _api_key(self) -> str:
        if self._key is None:
            from gpurunner.auth import vast as vast_auth

            self._key, _ = vast_auth.find_api_key()
        return self._key

    def _request(self, method: str, path: str, *, api_version: str = "v0",
                 **kwargs: Any) -> dict[str, Any]:
        """Запит до Vast із МІЖПРОЦЕСНИМ дроселем і повагою до `retry_after`.

        🔴 Стеля 5 запитів/с — на КЛЮЧ, а не на процес. Дві паралельні сесії
        (а в нас це норма: один захід рахує, другий шукає ринок) ділять її, і
        429 сипле в ОБИДВА логи — при тому, що кожна окремо поводиться чемно.
        Наслідок не косметичний: `POST /bundles/` падає, добір лишається без
        ринку й ходить по колу («⚠ запит ринку не вдався»), а на оренді той
        самий 429 читається як `market_busy` і витрачає спробу.

        Дросель тримає позначку часу в СПІЛЬНОМУ для всіх процесів файлі під
        `data_dir()`, тож інтервал витримується між сесіями, а не всередині
        однієї.
        """

        attempts = 0
        while True:
            attempts += 1
            _throttle_vast_api()
            try:
                return self._request_once(method, path, api_version=api_version, **kwargs)
            except BackendError as e:
                status = getattr(e, "status_code", None)
                if status != 429 or attempts >= _RATE_RETRIES:
                    raise
                # Vast сам каже, скільки чекати («retry_after»: 7) — беремо його
                # число, а не власне: воно єдине знає, наскільки ми в боргу.
                time.sleep(_retry_after_sec(getattr(e, "body", "")))

    def _request_once(self, method: str, path: str, *, api_version: str = "v0",
                      **kwargs: Any) -> dict[str, Any]:
        import httpx

        url = f"{API_BASE}{path}"
        if api_version != "v0":
            url = f"{API_BASE.rsplit('/', 1)[0]}/{api_version}{path}"
        headers = {"Authorization": f"Bearer {self._api_key()}", "Accept": "application/json"}
        try:
            # follow_redirects: Vast answers 301 on some endpoints (`/users/current/`
            # among them), and httpx does NOT follow redirects by default — the call
            # then fails with "Redirect response '301 Moved Permanently'" instead of
            # returning data. `auth/vast.py::verify` already had this; `_request` did
            # not, which is why `gpurunner balance -b vast` broke (caught 2026-08-02).
            resp = httpx.request(
                method, url, headers=headers, timeout=60, follow_redirects=True, **kwargs
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                raise AuthError(f"Vast.ai rejected the API key on {method} {path}: {e}") from e
            err = BackendError(
                f"Vast.ai {method} {path} failed: {e} — {e.response.text[:300]}"
            )
            # 🔴 Машинно-читний код помилки. Без нього єдиний спосіб дізнатись
            # статус — regex по тексту, і саме тому 400 («оффер уже зайняли»)
            # був невідрізненний від битого конфігу й термінально спиняв захід.
            err.status_code = e.response.status_code
            err.body = e.response.text[:500]
            raise err from e
        except httpx.HTTPError as e:
            raise BackendError(f"Vast.ai {method} {path} failed: {e}") from e
        if not resp.content:
            return {}
        try:
            data = resp.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {"result": data}

    def _instance(self, handle: JobHandle) -> dict[str, Any] | None:
        data = self._request(
            "GET", f"/instances/{handle.remote_id}/", params={"owner": "me"}
        )
        inst = data.get("instances")
        if isinstance(inst, list):
            inst = inst[0] if inst else None
        return inst if isinstance(inst, dict) else None

    def _container_logs(self, handle: JobHandle) -> list[str]:
        """Docker logs via the async request_logs endpoint (result lands on a CDN URL)."""
        import httpx

        try:
            resp = self._request(
                "PUT", f"/instances/request_logs/{handle.remote_id}/", json={"tail": "500"}
            )
        except BackendError as e:
            return [f"(container logs unavailable: {e})"]
        url = resp.get("result_url")
        if not url:
            return [f"(container logs unavailable: {resp.get('msg') or resp})"]
        for _ in range(20):
            time.sleep(1)
            try:
                r = httpx.get(str(url), timeout=30)
                if r.status_code == 200:
                    return r.text.splitlines()
            except httpx.HTTPError:
                continue
        return ["(container logs did not materialize in time)"]


# ---- helpers ---------------------------------------------------------------


#: Повідомлення Vast, після яких чекати НЕМА ЧОГО: хост не може дістати образ.
#: `Error response from daemon: Get "https://registry-1.docker.io/v2/": … read:
#: connection reset by peer` — це мережа ХОСТА до Docker Hub, і сама вона не
#: полагодиться. Без цього списку такий бокс висів повну стелю підйому й лише
#: потім класифікувався — тобто шість хвилин оренди за гарантовано нульовий
#: результат, щоразу.
_FATAL_STATUS_PATTERNS = (
    "error response from daemon",
    "connection reset by peer",
    "no space left on device",
    "manifest unknown",
    "unauthorized: authentication required",
    "failed to resolve reference",
)


def _fatal_status(msg: str) -> str:
    """Чому цей стан безнадійний — або порожньо, якщо ще є шанс."""
    low = (msg or "").lower()
    for pat in _FATAL_STATUS_PATTERNS:
        if pat in low:
            return pat
    return ""


class ForeignInstance(BackendError):
    """Інстанс належить іншій сесії. Знищувати його ми не маємо права."""


def _instance_label(job_name: str, params: dict[str, Any], handle_id: str) -> str:
    """Мітка інстансу: чия це робота, видно без бази даних."""
    case = str(params.get("case") or "").strip()
    case = "".join(ch for ch in case if ch.isalnum() or ch in "-_.")[:32]
    parts = ["gpurunner", job_name, case, handle_id[:8]]
    return "-".join(p for p in parts if p)


def _instance_payload(
    offer: dict[str, Any], *, image: str, disk_gb: float, label: str,
    onstart: str, num_gpus: int,
) -> dict[str, Any]:
    """Тіло `PUT /asks/<offer>/` — одне на job і на голу оренду."""
    return {
        "client_id": "me",
        "image": image,
        "disk": float(disk_gb),
        "label": label,
        "onstart": onstart,
        "runtype": "ssh_direc ssh_proxy",
        "env": {},
        # 🔴 Явна кількість карт і порожній темплейт — інакше Vast бере їх із
        # ТЕМПЛЕЙТА АКАУНТА, вибраного в UI. 2026-08-11: у користувача був
        # активний порожній темплейт (`image: hub.docker.com/r/null/`), і два
        # інстанси поспіль (Таїланд, Квебек) не змогли створити контейнер:
        # «Template not found» + «failed to inject CDI devices … gpu=2:
        # unknown» — хост намагався прокинути другу карту на машині з однією.
        # Помилка КОШТУЄ ГРОШЕЙ мовчки: інстанс висить у стані `created`,
        # тарифікація йде, job лишається `queued`, а SSH не піднімається
        # взагалі (підключатись нема до чого).
        # ⚠ Поля темплейта тут НЕ передаються навмисно: Vast відхиляє `null`,
        # а вгадувати чужий hash небезпечніше, ніж скинути темплейт в UI.
        # Орендуємо те, що відскорили: скоринг рахує VRAM × карти оффера,
        # тож і замовляти треба всі його карти, а не «одну» з плану.
        "num_gpus": int(offer.get("num_gpus") or num_gpus or 1),
    }


def _banned_machines() -> set[int]:
    """Чорний список реєстру. Збій реєстру не має валити оренду."""
    try:
        return boxes.banned_ids()
    except Exception as e:
        print(f"[vast] ⚠ реєстр боксів недоступний ({e}) — фільтр не застосовано", flush=True)
        return set()


def _backend_opts(params: dict[str, Any]) -> dict[str, Any]:
    """Backend-only knobs, read from RAW params (validate_params drops unknown keys)."""
    max_price = params.get("max_price")
    return {
        "image": str(params.get("image") or DEFAULT_IMAGE),
        "disk": int(params.get("disk") or DEFAULT_DISK_GB),
        "num_gpus": int(params.get("num_gpus") or 1),
        "max_price": float(max_price) if max_price else None,
        "max_hours": int(params.get("max_hours") or DEFAULT_MAX_HOURS),
        "autodestroy_hours": float(params.get("autodestroy_hours") or 0),
        "inputs": params.get("inputs") or {},
        "input_root": params.get("input_root"),
        # CPU-bound job'и (htr_case) без цього беруть найдешевшу машину з 4 ядрами
        # і працюють повільніше за домашній ноутбук — див. search_offers.
        "min_cpu": float(params.get("min_cpu") or 0),
        "min_ram_gb": float(params.get("min_ram_gb") or 0),
        "min_reliability": float(params.get("min_reliability") or 0),
        # VRAM визначає, скільки шардів підняти, тобто скільки з оплачених
        # ядер узагалі працюватиме. Без цього фільтра беруться 16 ГБ під
        # 8 шардів — і сторінки зникають в OOM.
        "min_vram_gb": float(params.get("min_vram_gb") or 0),
        # Мінімальна CUDA драйвера хоста (`cuda_max_good`). Образ vLLM 0.29 зібрано
        # під CUDA 13.0 (`VLLM_ENABLE_CUDA_COMPATIBILITY=0`) — на хості з 12.x
        # контейнер не підніме модель, а оренда вже тарифікується.
        "min_cuda": float(params.get("min_cuda") or 12.1),
    }


def _resolve_inputs(job: Job, params: dict[str, Any], opts: dict[str, Any]) -> dict[str, Path]:
    """``{slug: local dir}`` to upload into ``/kaggle/input/<slug>/``.

    Explicit ``-p inputs='{"slug": "E:/path"}'`` wins. Otherwise the job's declared
    inputs (``colab_input_dirs``, which most jobs get for free from
    ``dataset_sources``) are looked up under ``-p input_root=<dir>``.
    """
    return resolve_inputs(job, params, inputs=opts["inputs"], input_root=opts["input_root"])


#: Скільки терпимо ОДНУ операцію на сокеті, перш ніж визнати його мертвим. Забір
#: іде пофайлово, і жоден окремий обмін не буває довгим: 300 с — це вже не
#: повільний канал, а хост, якого немає.
#:
#: 🔴 МЕЖА ДЛЯ ТИХ, ХТО ДОДАВАТИМЕ ВИКЛИКИ: таймаут стоїть на транспорті, тож
#: він стосується КОЖНОГО обміну через це з'єднання. Довга команда, що мовчить
#: понад 300 с, буде вбита — і це навмисно, бо мовчазне зависання на мертвому
#: боксі коштувало пів години оренди (17.08.2026).
#:
#: ⚠ Зворотний бік перевірено аудитом 04.09.2026: усі наявні виклики вже
#: обмежені коротше — проба 90 с, службові команди 120 с, читання каналу при
#: заборі 60 с. Тобто жоден продуктовий шлях цим таймаутом не ламається.
#: Зламався лише одноразовий скрипт, який тримав канал відкритим на весь
#: багатохвилинний прогін без жодного виводу. Так робити не можна: довгу роботу
#: пускати відчепленою на боксі, а результат забирати коротким запитом.
_SFTP_IO_TIMEOUT = 300


def _load_private_key(path: Path) -> Any:
    """Приватний ключ, прочитаний САМЕ своїм класом.

    🔴 З ``key_filename=`` paramiko перебирає класи ключів по черзі, і на
    DSA-парсері падає ``ValueError: q must be exactly 160, 224, or 256 bits
    long`` — ЩЕ ДО того, як дійде до RSA. Це не «ключ не підійшов», а обрив усієї
    спроби автентифікації: інстанс так і не стає доступним, а лічильник оренди
    цокає. Замір 2026-08-09: 15 хвилин оплаченого очікування SSH на боксі, куди
    звичайний ``ssh -i id_rsa`` заходить із першого разу.

    DSSKey у списку немає навмисно — SSH-сервери його вже не приймають, а
    єдиний його внесок тут був у вигляді описаного вище падіння.

    Файл читається локально й НІКУДИ не передається: на бокс іде лише публічна
    половина, через API Vast.
    """
    import paramiko

    errors: list[str] = []
    for cls in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
        try:
            return cls.from_private_key_file(str(path))
        except paramiko.PasswordRequiredException as e:
            raise AuthError(
                f"SSH key {path} is passphrase-protected; gpurunner cannot unlock it "
                f"non-interactively. Use an unencrypted key or point "
                f"GPURUNNER_VAST_SSH_KEY at one."
            ) from e
        except Exception as e:  # не той клас — пробуємо наступний
            errors.append(f"{cls.__name__}: {type(e).__name__}")
    raise AuthError(
        f"cannot read SSH private key {path} as ed25519/rsa/ecdsa ({'; '.join(errors)})"
    )


#: Ознаки того, що хост саме ВІДХИЛИВ ключ, а не був недосяжним. paramiko не
#: завжди кидає `AuthenticationException` — інколи це загальний `SSHException`
#: із текстом, тож звіряємо і клас, і слова.
_AUTH_MARKERS = (
    "permission denied",
    "authentication failed",
    "no authentication methods",
    "not a valid",
    "bad authentication type",
)


def _classify_ssh_error(exc: Exception, *, host: str, port: int) -> BackendError:
    """Відмова ключа чи «ще не піднявся»? Від цього залежить, скільки чекати."""
    try:
        import paramiko
    except ImportError:  # pragma: no cover — сюди не дійти без paramiko
        paramiko = None  # type: ignore[assignment]

    text = str(exc).lower()
    is_auth = bool(paramiko) and isinstance(
        exc, (paramiko.AuthenticationException, paramiko.BadAuthenticationType)
    )
    if not is_auth and any(marker in text for marker in _AUTH_MARKERS):
        is_auth = True

    if is_auth:
        return SshAuthRejected(
            f"хост {host}:{port} відхилив наш ключ ({exc}). Це не «ще вантажиться»: "
            f"чекати далі на цьому інстансі марно. Часто це збій прив'язки ключа на "
            f"боці Vast, а не хост — вирок машині виносить реєстр (бан лише з "
            f"другого удару)."
        )
    err = BackendError(f"SSH to {host}:{port} failed: {exc}")
    err.outcome = "ssh_unreachable"
    return err


def _accrued_cost(inst: dict[str, Any]) -> float:
    dph = float(inst.get("dph_total") or 0)
    start = inst.get("start_date")
    if not start:
        return 0.0
    try:
        started = datetime.fromtimestamp(float(start), tz=UTC)
    except (TypeError, ValueError):
        return 0.0
    return dph * max(0.0, (datetime.now(tz=UTC) - started).total_seconds() / 3600.0)


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# ---- стеля запитів до API --------------------------------------------------

#: Мінімальний інтервал між запитами до Vast, секунд. Стеля ринку — 5/с НА
#: КЛЮЧ; беремо ~3.3/с, щоб лишався запас на сторонні виклики (`vastai` з рук,
#: веб-консоль) і на те, що дві сесії прокидаються майже одночасно.
_RATE_MIN_INTERVAL_SEC = 0.30
#: Скільки разів перепитати після 429, перш ніж віддати помилку нагору.
_RATE_RETRIES = 3
#: Скільки чекати на чужий дросель, перш ніж вважати його покинутим. Довго:
#: тримач спить щонайбільше `_RATE_MIN_INTERVAL_SEC`, тож усе, що довше, — це
#: процес, який помер із заблокованим файлом.
_RATE_LOCK_STALE_SEC = 30.0


def _rate_state_path() -> Path:
    """Спільна для ВСІХ процесів позначка часу останнього запиту."""
    from gpurunner.config import data_dir

    return data_dir() / "vast_ratelimit"


def _throttle_vast_api() -> None:
    """Витримати інтервал від останнього запиту — свого чи чужої сесії.

    Лок — `os.mkdir`: єдина атомарна операція, яка однаково працює на Windows і
    POSIX без залежностей. Тримач спить усередині локу навмисно: так черга
    вишиковується, а не збігається знову на виході.
    """
    path = _rate_state_path()
    lock = path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    waited = 0.0
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            if waited > _RATE_LOCK_STALE_SEC:
                # Тримач мертвий: знімаємо лок і йдемо далі. Гірше, що може
                # статись, — один зайвий запит поспіль.
                with contextlib.suppress(OSError):
                    os.rmdir(lock)
                continue
            time.sleep(0.02)
            waited += 0.02
        except OSError:
            return  # немає де писати — краще запит без дроселя, ніж падіння
    try:
        try:
            last = float(path.read_text(encoding="utf-8") or 0)
        except (OSError, ValueError):
            last = 0.0
        now = time.time()
        # Годинник міг стрибнути назад (або файл із майбутнього) — не чекаємо вічність.
        wait = min(_RATE_MIN_INTERVAL_SEC, last + _RATE_MIN_INTERVAL_SEC - now)
        if wait > 0:
            time.sleep(wait)
        with contextlib.suppress(OSError):
            path.write_text(str(time.time()), encoding="utf-8")
    finally:
        with contextlib.suppress(OSError):
            os.rmdir(lock)


def _retry_after_sec(body: str) -> float:
    """Скільки чекати після 429 — з відповіді Vast, із запасом на розсинхрон."""
    try:
        value = float(json.loads(body or "{}").get("retry_after") or 0)
    except (ValueError, TypeError, AttributeError):
        value = 0.0
    # Плюс невеликий доважок: дві сесії, розбуджені тим самим `retry_after`,
    # інакше прокидаються одночасно й дістають 429 знову — тепер удвох.
    return min(30.0, max(1.0, value) + _RATE_MIN_INTERVAL_SEC)


# ---- remote templates ------------------------------------------------------

#: Проба заліза. Виводить `KEY=VALUE`, нічого не встановлює, нічого не лишає.
#: ⚠ Кожен рядок мусить пережити голий образ — тому скрізь `|| echo` і
#: `2>/dev/null`: відсутній `nvidia-smi` має дати порожнє значення (і ворота
#: скажуть «карти немає»), а не обірвати всю пробу.
_PROBE_SCRIPT = r"""
echo "cores=$(nproc 2>/dev/null || echo 0)"
echo "cores_all=$(nproc --all 2>/dev/null || echo 0)"
echo "ram_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo 0)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', *' '{tot+=$2; free+=$3; n++; if(NR==1){name=$1; mn=$3} else if($3<mn){mn=$3}} \
      END{if(n){printf "gpu=%s\nn_gpus=%d\nvram_total_mb=%s\nvram_free_mb=%s\nvram_free_min_mb=%s\n", \
      name,n,tot,free,mn}}'
# 🔴 Квота КОНТЕЙНЕРА, а не хоста. `nproc` рапортував 192 там, де продано 48,
# і план поїхав на вчетверо неіснуючому залізі (машина 39565, 2026-08-19).
# Знає про це лише cgroup. Порожній рядок = «не обмежено або не прочитали»,
# і ворота тоді лишаються на `nproc` — гірше, ніж раніше, не стає.
if [ -r /sys/fs/cgroup/cpu.max ]; then
  awk '{if ($1 != "max" && $2 > 0) print "cores_quota=" $1/$2}' /sys/fs/cgroup/cpu.max 2>/dev/null
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
  _q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1)
  _p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 0)
  awk -v q="$_q" -v p="$_p" 'BEGIN{if (q+0 > 0 && p+0 > 0) print "cores_quota=" q/p}' 2>/dev/null
fi
# Те саме про RAM: `free -g` дає пам'ять машини. Число завбільшки з терабайт
# означає «ліміту немає» — такі не друкуємо, щоб не вдавати вимір.
for _f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  [ -r "$_f" ] || continue
  awk '{if ($1 ~ /^[0-9]+$/ && $1+0 < 1099511627776) print "ram_limit_gb=" $1/1073741824}' "$_f" 2>/dev/null
  break
done
echo "disk_free_gb=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc 0-9 || echo 0)"
echo "py=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo '')"
U="__NET_URL__"
if [ -n "$U" ]; then
  # 🔴 Разом зі швидкістю знімаємо КОД ВІДПОВІДІ. Протухле presigned-посилання
  # віддає 403 з дрібним XML за мілісекунди — і без коду це не відрізнити від
  # «у хоста мертвий канал»: справна машина летіла в бан на 14 днів за НАШУ
  # прострочену URL, і так по черзі всі кандидати.
  # 🔴 Кожне значення ОКРЕМИМ рядком із власною міткою. Спільний рядок через
  # `cut -d' '` уже дав «HTTP 2060» (швидкість потрапила в поле коду) — і це
  # не косметика: гілка «наше посилання протухло» ТЕРМІНАЛЬНА, тож помилка
  # розбору спиняла весь захід із хибним діагнозом.
  # 🔴 Код виходу — окремим рядком, а не `|| echo "net_bps=0"`. curl друкує
  # `-w` і на помилці, а розбір бере ОСТАННІЙ ключ: на `--max-time` (rc=28,
  # канал повільніший ~22 Мбіт/с) справжня швидкість перетиралась нулем, і
  # вирок звучав «канал 0.0 Мбіт/с» (11.09.2026, машини 31846 і 138981).
  # Нуль лишається лише там, де curl немає взагалі.
  if command -v curl >/dev/null 2>&1; then
    curl -o /dev/null -s --max-time 12 -r 0-33554432 -w 'net_bps=%{speed_download}
net_http=%{http_code}
' "$U" 2>/dev/null
    echo "net_rc=$?"
  else
    echo "net_bps=0"
  fi
else
  echo "net_bps="
fi
"""

_ONSTART = """\
#!/bin/bash
# gpurunner onstart — prepare the Kaggle-shaped filesystem, wait for inputs, run.
mkdir -p {root} {working} {inputs}
{deadline_guard}
# gpurunner uploads inputs over SFTP after the box is reachable and then touches
# GO. Starting before that would run the job against an empty /kaggle/input.
echo "[gpurunner] waiting for GO sentinel" >> {root}/_runner.log
for _ in $(seq 1 900); do
    [ -f {root}/GO ] && break
    sleep 2
done
if [ ! -f {root}/GO ]; then
    echo '{{"status": "failed", "error": "inputs never arrived (no GO sentinel)"}}' > {root}/_status.json
    exit 1
fi

cd {root}
timeout {max_seconds}s python3 -u job.py 2>&1 | tee -a {root}/_runner.log
{autodestroy}
"""

#: Скільки годин живе голо орендований бокс, якщо викликач не сказав свого.
#: Те саме, що дефолтна стеля заходу (`Plan.max_hours` = 8) плюс година на забір.
_RENT_AUTODESTROY_HOURS = 9.0

#: onstart голої оренди (`rent_box`): job'а немає, тож від скрипта лишається
#: одне — страховка грошей.
#:
#: 🔴 Таймер тут ЖОРСТКИЙ, без стоп-крана `NOSTOP`. У шляху з наглядачем кран
#: ставить сам наглядач на час забору й він протухає за 30 хвилин; тут його
#: нема кому ні ставити, ні знімати, а забутий кран — це бокс, який не погасить
#: себе ніколи. Запас на забір закладає викликач у саме число годин.
#:
#: 🔴 Скрипт НЕ виходить, а спить на передньому плані: так таймер не залежить
#: від того, чи переживає фоновий процес завершення onstart на конкретному
#: хості. Спроба знищення повторюється — один мережевий збій у мить дедлайну
#: інакше означав би бокс, що горить далі вже без жодного сторожа.
#:
#: ⚠ `curl` доставляється у фоні, якщо його немає: ним користується і сам
#: таймер (із запасним шляхом через python3), і викликач, який ставить на бокс
#: своє середовище. Невдача тут не фатальна й лише пишеться в лог.
_RENT_ONSTART = """\
#!/bin/bash
# gpurunner rent onstart — no job: only the self-destruct timer.
mkdir -p {root}
LOG={root}/_rent.log
if ! command -v curl >/dev/null 2>&1; then
    ( (apt-get update -qq && apt-get install -y -qq curl ca-certificates) >> $LOG 2>&1 \\
        || echo "[gpurunner] curl не доставлено" >> $LOG ) &
fi
if [ -z "$CONTAINER_API_KEY" ] || [ -z "$CONTAINER_ID" ]; then
    echo "[gpurunner] ⚠ немає CONTAINER_API_KEY/CONTAINER_ID — самознищення НЕМАЄ" >> $LOG
    exit 0
fi
echo "[gpurunner] самознищення зведено на {deadline_seconds}s" >> $LOG
sleep {deadline_seconds}
for _ in $(seq 1 30); do
    echo "[gpurunner] дедлайн {deadline_seconds}s — знищую інстанс" >> $LOG
    if command -v curl >/dev/null 2>&1; then
        curl -s -X DELETE "{api_base}/instances/$CONTAINER_ID/" \\
            -H "Authorization: Bearer $CONTAINER_API_KEY" -H "Content-Type: application/json" \\
            -d '{{}}' >> $LOG 2>&1
    else
        python3 -c 'import os,urllib.request as u; u.urlopen(u.Request("{api_base}/instances/"+os.environ["CONTAINER_ID"]+"/", data=b"{{}}", method="DELETE", headers={{"Authorization":"Bearer "+os.environ["CONTAINER_API_KEY"],"Content-Type":"application/json"}}), timeout=60)' >> $LOG 2>&1
    fi
    sleep 60
done
"""

_DEADLINE_GUARD = """
# 🔴 БЕЗУМОВНИЙ дедлайн, і саме ТУТ — на початку та у фоні.
#
# Раніше єдиний запобіжник (`_AUTODESTROY`) стояв ПІСЛЯ `python3 job.py`, тобто
# у послідовному потоці скрипта. Наслідок: гілка «дані так і не приїхали →
# exit 1» його просто оминала, і бокс, на який ми не змогли нічого залити,
# горів до ручного втручання. Так само його оминав будь-який `exit` вище.
#
# Тут таймер живе окремим процесом від першої секунди життя контейнера, тож
# він переживає і провал заливки, і смерть job'а, і смерть наглядача разом із
# ноутбуком. Стоп-кран той самий — файл NOSTOP.
if [ -n "$CONTAINER_API_KEY" ] && [ -n "$CONTAINER_ID" ]; then
    (
        sleep {deadline_seconds}
        if [ -n "$(find {root}/NOSTOP -mmin -30 2>/dev/null)" ]; then
            echo "[gpurunner] дедлайн настав, але є СВІЖИЙ NOSTOP — не гашу, ТАРИФІКАЦІЯ ЙДЕ" \\
                >> {root}/_runner.log
            exit 0
        fi
        echo "[gpurunner] дедлайн {deadline_seconds}s — знищую інстанс" >> {root}/_runner.log
        curl -s -X DELETE "{api_base}/instances/$CONTAINER_ID/" \\
            -H "Authorization: Bearer $CONTAINER_API_KEY" -H "Content-Type: application/json" \\
            -d '{{}}'
    ) &
    echo "[gpurunner] дедлайн-кілер зведено на {deadline_seconds}s" >> {root}/_runner.log
else
    echo "[gpurunner] ⚠ немає CONTAINER_API_KEY/CONTAINER_ID — дедлайн-кілера НЕМАЄ" \\
        >> {root}/_runner.log
fi
"""

_AUTODESTROY = """
# opt-in runaway guard: destroy this instance after a grace period so a finished
# (or hung) job cannot bill forever. Fetch your outputs before it fires.
if [ -n "$CONTAINER_API_KEY" ] && [ -n "$CONTAINER_ID" ]; then
    echo "[gpurunner] self-destruct in {grace_seconds}s (touch {root}/NOSTOP to cancel)" >> {root}/_runner.log
    sleep {grace_seconds}
    # Стоп-кран. Знадобився в перший же день (2026-08-09): job упав на pip,
    # таймер пішов, і поки залежності доставляли руками, він знищив бокс разом
    # із уже залитими 302 МБ. Тепер його ставить сам наглядач на час забору.
    #
    # 🔴🔴 Кран САМ ПРОТУХАЄ за 30 хвилин, і це принципово: якщо наглядач
    # помре посеред качання (сталося тричі за добу), вічний кран означав би
    # бокс, який не погасить себе НІКОЛИ. Свіжий файл = «зараз хтось качає»;
    # протухлий = «той, хто його поставив, більше не з нами».
    if [ -n "$(find {root}/NOSTOP -mmin -30 2>/dev/null)" ]; then
        echo "[gpurunner] NOSTOP present — self-destruct cancelled, STILL BILLING" >> {root}/_runner.log
        exit 0
    fi
    curl -s -X DELETE "{api_base}/instances/$CONTAINER_ID/" \\
        -H "Authorization: Bearer $CONTAINER_API_KEY" -H "Content-Type: application/json" -d '{{}}'
else
    echo "[gpurunner] no CONTAINER_API_KEY/CONTAINER_ID — cannot self-destruct, still billing" \\
        >> {root}/_runner.log
fi
"""

#: Runs on the instance. Same ``_status.json`` contract as the Colab backend
#: (status/ts/heartbeat/error + terminal latch), so status logic reads the same.
_RUNNER_WRAPPER = '''\
import json, os, sys, threading, traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace/gpurunner")
ROOT.mkdir(parents=True, exist_ok=True)
Path("/kaggle/working").mkdir(parents=True, exist_ok=True)

_LOCK = threading.Lock()
_FINALIZED = []


def _now():
    return datetime.now(tz=timezone.utc).isoformat()


def write_status(status, **extra):
    """Publish job state; terminal states latch so a late heartbeat can't undo them."""
    with _LOCK:
        if _FINALIZED:
            return
        if status in ("completed", "failed"):
            _FINALIZED.append(status)
        payload = {"status": status, "ts": _now(), "heartbeat": _now()}
        payload.update(extra)
        tmp = ROOT / "_status.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, ROOT / "_status.json")


write_status("running", message="job started")
_stop = threading.Event()


def _heartbeat():
    while not _stop.wait(60):
        try:
            write_status("running", message="in progress")
        except Exception:
            pass


_beat = threading.Thread(target=_heartbeat, daemon=True)
_beat.start()

RUNNER_SRC = __RUNNER_SRC__
_err = None
try:
    exec(compile(RUNNER_SRC, "<gpurunner_job>", "exec"), {"__name__": "__main__"})
except BaseException as e:
    traceback.print_exc()
    _err = repr(e)
finally:
    _stop.set()

_beat.join(timeout=90)
_n = sum(1 for p in Path("/kaggle/working").rglob("*") if p.is_file())
if _err is None:
    write_status("completed", message="%d files in /kaggle/working" % _n)
    print("[gpurunner] DONE — %d output files" % _n, flush=True)
else:
    write_status("failed", error=_err, message="%d files in /kaggle/working (partial)" % _n)
    print("[gpurunner] FAILED: %s" % _err, flush=True)
    sys.exit(1)
'''
