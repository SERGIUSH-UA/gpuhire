# Контракт для програм, що кличуть gpurunner

gpurunner — окремий процес. Інші програми (пакет `nyshporka`, обгортки
дослідницьких репозиторіїв) запускають його через `subprocess` і читають
відповідь. Тут описано, на що вони мають право спиратись.

## 1. Як знайти

- Виконуваний файл зветься `gpurunner` і шукається в `PATH`
  (`shutil.which("gpurunner")`).
- `gpurunner --version` друкує версію пакета.
- Сумісність тримається в межах мінорної версії: у `0.2.x` нічого з переліченого
  нижче не зникає й не міняє змісту.

## 2. Машинний вивід (`--json`)

1. У stdout з'являється рівно один рядок JSON-об'єкта.
2. Брати **останній рядок stdout, що починається з `{`**: SDK окремих бекендів
   друкують у stdout власний прогрес, і заборонити їм це не можна.
3. Усе людське (таблиці, попередження про тарифікацію) з `--json` іде в stderr.
4. Код виходу той самий, що й без `--json`. При відмові об'єкт теж друкується:
   `"ok": false`, `error`, `exit_code`. Якщо рядка JSON немає взагалі (збій до
   розбору аргументів) — вирішує код виходу.
5. Кожен об'єкт несе `schema` (зараз `1`). Число росте лише тоді, коли поле
   зникає або міняє зміст; нове поле — не привід, зайві поля споживач ігнорує.

### 2.1. Опис прогону

Спільні поля в `run` і `status`:

| поле | зміст |
|---|---|
| `handle` | повний ідентифікатор прогону |
| `short` | перші 8 символів; усі команди приймають і його |
| `backend`, `job`, `gpu` | куди, що і на чому подано |
| `remote_id` | ідентифікатор на боці бекенда |
| `state` | `queued` · `running` · `completed` · `failed` · `cancelled` · `unknown` |
| `output_dir` | куди ляже або вже ліг вихід; порожньо, якщо не задано |

### 2.2. `gpurunner run <job> -b <backend> [--gpu G] [-p k=v …] --json`

```json
{"schema": 1, "ok": true, "command": "run", "job": "parseq_train", "backend": "modal",
 "dry_run": false, "handle": "…", "short": "…", "gpu": "A100", "remote_id": "…",
 "state": "queued", "output_dir": ""}
```

З `--dry-run`: `"dry_run": true` і `params` (параметри після перевірки), без
полів прогону. Відмова: `"ok": false`, `error` з префіксом `validation:` або
`submit:`, `exit_code` 2 (аргументи, перевірка) або 3 (бекенд не прийняв).

### 2.3. `gpurunner status <handle> [--no-ping] --json`

Поля прогону плюс:

| поле | зміст |
|---|---|
| `refreshed` | `true` — стан щойно взято з бекенда; `false` — з локального маніфесту (бекенд не відповів або `--no-ping`) |
| `terminal` | `true` для `completed`, `failed`, `cancelled` |
| `message` | пояснення від бекенда, якщо є |
| `error` | причина падіння, якщо є |

Без `<handle>`: `{"handles": [<опис прогону>, …]}`. Невідомий handle: `"ok": false`,
`"error": "unknown handle"`, код виходу 2.

### 2.4. `gpurunner fetch <handle> [--out DIR] [--resume] --json`

```json
{"schema": 1, "ok": true, "command": "fetch",
 "fetched": [{"handle": "…", "short": "…", "result": "ok", "files": 12,
              "output_dir": "…", "error": ""}]}
```

`result`: `ok` · `skipped` (вихід уже повний, лише з `--resume`) · `error`.
Якщо хоч один `error` — `"ok": false` і код виходу 3.

### 2.5. Команди HTR

`gpurunner htr supervise --plan P --dry-run --json` — кошторис без оренди; поля,
на які спираються споживачі: `empty`, `reason`, `credit`, `pages`,
`best.{cost, hours, gpu, num_gpus, dph, pages_per_hour}`.

`gpurunner htr state --session S --json` — стан відчепленого наглядача, зокрема
`log_path`.

`gpurunner htr preflight P --json`, `gpurunner balance --json`,
`gpurunner boxes ls --json` — так само один об'єкт або список останнім рядком.

## 3. Команди й прапорці, які не зникають у межах мінорної версії

```
gpurunner run <job> -b <backend> [--gpu G] [--dry-run] [--out DIR] -p k=v … [--json]
gpurunner status [<handle>] [--ping/--no-ping] [--json]
gpurunner fetch <handle>… [--out DIR] [--all] [--resume] [--json]
gpurunner cancel <handle>… | --all-running
gpurunner ls | recipes <job> | balance [--json] | burn | reconcile [--any-owner]
gpurunner dataset push|files · drive push|files · vast offers|instances
gpurunner auth <backend> [--verify] · auth vast --key <КЛЮЧ | ->
gpurunner boxes ls|explain|ban|star|prune|forget
gpurunner bg start <ім'я> -- <команда> | bg status | bg stop <ім'я>
gpurunner htr plan --model --voices --case --case-key [--name] [--seed-seg]
                   [--key-not-in-library] --assets --out --max-hours --disk
                   [--max-usd-per-1000] [--transport r2|box|auto] [-p k=v]
gpurunner htr preflight <plan.json> [--json]
gpurunner htr supervise --plan P (--dry-run [--json] | --detach --session S --budget $ --max-hours H)
gpurunner htr state (--session S | --plan P) [--json]
gpurunner htr stop --session S [--cancel-box]
gpurunner htr fetch-ckpt (--plan P | --case K --out-root DIR) [--dry-run]
gpurunner htr calibrate | htr append
```

Людський текст без `--json` контрактом не є, за двома винятками, які лишаються
заради старих споживачів: рядок `submitted <short>` після успішного `run` і
слово стану в панелі `status`.

## 4. Файл плану (`plan.json`)

План пише `gpurunner htr plan`. Поле `transport` каже, звідки машина візьме
дані:

- `r2` — бакет S3: план несе `assets_url` і `cases[].pages_url` (presigned);
- `box` — склад на самій машині: посилань ще немає, натомість план несе
  `assets_path` і `cases[].pages_path` — шляхи на диску того, хто планує, —
  плюс `cases[].ckpt_prefix` і `ckpt_slots`. Файли везе наглядач по SSH, а
  посилання вигляду `http://127.0.0.1:<порт>/…` він підставляє сам.

🔴 Для `box` план прив'язаний до машини, яка його склала: у ньому шляхи, а не
посилання. Переносити такий план на інший комп'ютер не можна.

Споживач може дописати в план:

- `cases[].case_dir` — локальна тека кадрів справи; після забору вона лягає в
  мету прогону замість шляху контейнера;
- `post_fetch` — список `{"cmd": [...], "cwd": "...", "timeout_sec": N}`.

🔴 `post_fetch` — це **виконання команд із файла**: після повного забору
наглядач запускає кожну з них на машині користувача. Так задумано (облік у
репозиторії споживача мусить оновитись без людини), тому план є довіреним
входом — так само, як shell-скрипт. Чужий `plan.json` запускати не слід.

## 5. Оточення

| змінна | зміст |
|---|---|
| `GPURUNNER_CONFIG_DIR` | тека обліковок (`vast_api_key`, `r2.json`, `google_*.json` …) |
| `GPURUNNER_DATA_DIR` | машинний стан: маніфест прогонів, квоти, сесії наглядача |
| `GPURUNNER_REPO_DATA_DIR` | реєстр боксів і калібрування шардів; без змінної — `<data>/registry`. Варто вести у власному git |
| `GPURUNNER_BOXES_FILE`, `GPURUNNER_BOXES_OVERRIDES` | окремі файли реєстру |
| `GPURUNNER_OWNER` | мітка власника сесії — розводить паралельні заходи |
| `GPURUNNER_MODAL_VOLUME`, `GPURUNNER_MODAL_VOLUME_<JOB>` | ім'я тому Modal, якщо не дано `-p modal_volume=` |
| `R2_BUCKET` (або поле в `r2.json`) | бакет R2; без нього — `gpurunner-htr` |
| `VAST_API_KEY` | ключ API Vast.ai; має пріоритет над `<config>/vast_api_key` |
| `GPURUNNER_VAST_SSH_KEY` | шлях до приватного SSH-ключа для боксів Vast |
| `GPURUNNER_VAST_RENT_IMAGE` | образ контейнера для оренди плагіном `nyshporka.cloud` |

## 6. Плагін `nyshporka.cloud`

Другий спосіб спиратись на gpurunner — не процесом, а **плагіном у тому самому
інтерпретаторі**. Пакет оголошує entry point

```toml
[project.entry-points."nyshporka.cloud"]
vast = "gpurunner.plugins.nyshporka_cloud:VastRent"
```

і `VastRent()` (фабрика без аргументів) реалізує `nyshporka.cloud.base.CloudBackend`:
`id = "vast"`, `caps = {"rent", "cancel", "market"}`. Огляд для людини —
[`nyshporka.md`](nyshporka.md). Модуль імпортується й без Нишпорки; винятки, які
він кидає, — класи самої Нишпорки (`CloudError`, `AuthError`, `BoxGone`,
`BoxNotReady`), коли вона встановлена.

### 6.1. Обов'язкові методи

- `acquire(need, *, target="") -> Box` — повертається, коли на бокс уже пускає
  SSH. `target`: порожньо — підібрати самому; `"<число>"` — саме цей оффер, якщо
  він проходить добір; `"machine:<id>"` — адресно ця машина. З `need` читаються
  `pages, bytes_in, gb_per_shard, disk_gb, max_hours, budget_usd,
  max_price_usd_h`; `None`/0 у стелях — дефолти заходу gpurunner (8 год, $3,
  $0.365/год), `disk_gb = 0` — `max(40, кадри × 2 + 20)` ГБ. `prefer_cores` як
  ручка не впливає: ядра модель ураховує сама, і з 21.09.2026 — двічі. Спершу
  вони ріжуть число шардів, а потім, коли шардів підняли більше, ніж є кому
  годувати (менше двох ядер на шард), ставлять стелю самому темпу. Обидва рази
  це рахує скор, тож просити ядра окремо не треба. Помилки: `AuthError` —
  немає ключа API або Vast його відхилив; `CloudError` — «поповніть баланс
  Vast», порожній ринок (із причиною), вичерпано 3 оренди, інстанс не вдалось
  погасити.
- `connect(box) -> Session` — `nyshporka.cloud.ssh.SshBackend().connect(box)`.
  SSH не відповідає → `BoxNotReady`; `BoxGone` — лише коли API Vast інстансу не
  показує **і** від `rented_at` минуло понад 10 хвилин.
- `release(box, *, why="") -> None` — знищує інстанс; ідемпотентно, на мертвому
  інстансі мовчить. `CloudError` — інстанс живий, а погасити не вдалось, або
  мітка інстансу не `nysh-rent-…`. `why` керує записом у реєстр боксів:
  `ok…` (можна з `pph=<стор/год>`) · `failed:<деталь>` · `slow:<деталь>` · решта
  нейтрально.
- `find(box_id) -> Box | None` — з живого API; `None` — інстансу немає або він не
  наш. У такого `Box` немає `offer_id`, `autodestroy_at`, `claimed`.

### 6.2. `Box.meta`

| ключ | зміст |
|---|---|
| `host` | `{name, user: "root", host, port, key, workdir: "/workspace/nysh-run", python: "python3", cores, vram_gb, ram_gb, gpus}` — рівно те, що читає `SshBackend`; `key` — **шлях** до приватного ключа |
| `instance_id` | номер інстансу Vast, рядком (він же `Box.id`) |
| `machine_id`, `host_id`, `offer_id` | ідентифікатори Vast; ключ реєстру боксів — `machine_id` |
| `label` | мітка інстансу, `nysh-rent-<8 hex>` |
| `rented_at` | ISO UTC — мить перед створенням інстансу (звідси рахується вартість) |
| `autodestroy_at`, `autodestroy_hours` | ISO UTC і години: не пізніше цієї миті бокс знищить себе сам |
| `gpu_name`, `num_gpus`, `geolocation` | з картки оффера |
| `boot_sec` | секунд від створення інстансу до робочого SSH |
| `claimed` | що обіцяла картка: `cores, ram_gb, vram_gb, disk_gb, inet_down, inet_up, reliability2, dph_total` |

Числа `Box.cores / vram_gb / ram_gb / disk_gb` — заявка з картки (`vram_gb` — сума
по всіх картах), `price_usd_h` — `dph_total` оффера.

### 6.3. Необов'язкові методи

Нишпорка кличе їх через `getattr`. 🔴 Чого API не дав — того ключа у відповіді
немає або він `None`; вигаданих чисел не буває.

- `status() -> dict` — нічого не орендує:
  `{"ready": bool, "problems": [str], "balance_usd": float | None,
  "api_key": bool, "ssh_key": str, "burning": list | None}`.
  `ssh_key` — шлях до знайденого приватного ключа або `""` (це не проблема: пару
  згенерує перша оренда). `burning` — те саме, що `gpurunner burn`:
  `[{"instance_id": str, "dph_total": float | None, "label": str, "gpu_name": str,
  "status": str}]`; `None` — спитати не вдалось, `[]` — нічого не горить.
  `ready` = `problems` порожній.
- `login(api_key: str) -> dict` — зберігає ключ (`<config>/vast_api_key`),
  генерує SSH-ключ, перевіряє ключ запитом і повертає `status()`. `AuthError` —
  лише на порожній або багатослівний ключ; ключ, який Vast відхилив, лишається
  збереженим і стає рядком у `problems`.
- `estimate(need, *, target="") -> dict` — безкоштовний пошук офферів без оренди:
  завжди `{"empty": bool, "reason": str, "candidates": int,
  "balance_usd": float | None}`; коли ринок не порожній — ще
  `offer_id, machine_id, gpu: str, num_gpus: int, cores: float,
  price_usd_h: float, shards: int, pages_per_hour: float, hours: float,
  cost_usd: float, usd_per_1000: float, tier: int` найкращого кандидата — ті самі
  числа, що `gpurunner htr supervise --dry-run --json`. Кидає `AuthError` /
  `CloudError`, як `acquire`. Якщо `need.lines_per_page > 0`, щільність сторінки
  йде в прогноз темпу, і відповідь несе відлуння `lines_per_page` — лише за ним
  споживач має право звужувати вилку кошторису.
- `configured() -> bool` — чи лежить ключ API локально; без мережі, для
  діагностики (`nysh doctor`).
