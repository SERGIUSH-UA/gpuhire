# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```powershell
uv sync                              # install deps (develop on Python 3.12, see below)
uv sync --extra modal --extra colab --extra vast --extra lightning --extra beam --extra saturn --extra web
                                     # all extras; ⚠ syncing ONE extra REMOVES the others' packages
uv run gpurunner dash                # local balances dashboard (needs --extra web)
uv run gpurunner --help              # CLI entry point
uv run pytest                        # run all tests
uv run pytest tests/test_paddleocr_job.py::test_validate_urls_list_strings  # single test
uv run ruff check src tests          # lint
uv run ruff format src tests         # format
uv run mypy src                      # type-check
```

CLI shape: `gpurunner auth <backend> [--verify]` (Colab: `auth google --login`; Vast: `auth vast`; Lightning: `auth lightning`; Beam: `auth beam --token …`; Saturn: `auth saturn --url … --token …`), `gpurunner ls`, `gpurunner run <job> --backend <kaggle|modal|colab|vast|lightning|beam|saturn> -p k=v [-p k=v ...] --gpu T4 [--dry-run] [--open]`, then `status`, `fetch`, `watch`, `logs` against a handle id (8-char prefix is enough). Colab inputs: `gpurunner drive push <dir> --name <name>` / `drive files <name>`. Vast: `gpurunner vast offers` before renting, `gpurunner cancel <id>` to stop billing. Saturn: `gpurunner saturn sizes` lists the account's own `instance_type` names. `-p` values are JSON-decoded when possible, otherwise treated as strings.

End-to-end smoke (real Kaggle quota): `uv run python examples/paddleocr_kaggle.py <pdf-url>`. Modal variant: `examples/paddleocr_modal.py <pdf-url>`.

## Architecture

Two ABCs in `src/gpurunner/core/` carry the whole framework:

- **`Job`** (`core/job.py`) — backend-agnostic unit of work. Declares `requirements()` (pip pkgs), `validate_params()`, and `render_remote_code()` (returns a Python source string). For Modal it also exposes `render_runner_module()` (a self-contained module with `main(params) -> dict`) and `modal_image_spec()` (overrides the default derived from `requirements()` — useful when Kaggle and Modal need different package versions, see PaddleOCR).
- **`Backend`** (`core/backend.py`) — remote-execution target. `submit/status/fetch_outputs/logs`. Backends never inspect job internals beyond the ABC; one job class runs on multiple backends by branching only via the rendering hooks.

`cli.py` wires `get_job(name)` × `get_backend(name)` and persists results through `core/manifest.py` — a local **SQLite** DB (`runs.sqlite3` in the `platformdirs` user-data dir). Handle id is a uuid4 hex; `manifest.get()` accepts full id, 8-char prefix, or `remote_id`.

Two tables. `runs` is the handle set (what `manifest.load/add/update/get/remove` operate on — the API did not change when this stopped being JSON). `run_events` appends one row per *observed transition*, which is what makes "when did it actually start, how long did it run, why did it die" answerable; `gpurunner status <id>` prints the tail of it. Repeated identical polls do **not** append — otherwise a 30-second `watch` over a 12-hour train would write ~1400 identical rows.

Why it is a database, not the JSON file it used to be:
- **Lost updates.** `load() → mutate → save()` has no locking. `sweep --max-concurrent 10` submits from ten processes while a `watch` polls alongside; measured on the old code path, 4 concurrent writers over 20 handles landed **5 of 20** updates. Each write is now one transaction (WAL, `busy_timeout=15000`). `core/budget.py` had already made this argument for money.
- **No history.** A handle carried only its latest status, so every transition overwrote the last one — `updated_at` never moved and durations were not derivable at all.
- Migration is automatic and one-shot: the first open imports `manifest.json` if `runs` is empty (emptiness, not a flag file — so deliberately removed handles do not come back). The old file is left in place; `get()` still falls back to it for anything the DB never saw, so a partial migration cannot strand a live run.

### Embedded runner pattern (critical)

`src/gpurunner/_embedded/*_runner.py` files are **NOT imported locally** — they are read as source via `importlib.resources` and shipped to the remote runtime. They must be self-contained: stdlib plus whatever `Job.requirements()` declares. Tests for them live in the job's `validate_params` / `render_*` paths, not by importing the runner.

Two injection styles, both produced by the Job:

- **Kaggle**: `render_remote_code()` returns the full notebook-cell body. Jobs do **not** assemble it by hand — `Job.render_kaggle_code(runner_filename, params)` (`core/job.py`) does: injected `PARAMS` + `_embedded/_common.py` + the runner source (with its `if __name__ == "__main__":` guard stripped) + a final `main(PARAMS)`. `KaggleBackend` wraps this in a `nbformat` notebook with a `!pip install` cell and a `kernel-metadata.json` (slug must be lowercase, hyphens, ≤50 chars; the title is forced to equal the slug because Kaggle re-slugifies the title server-side).
  - The template used to be copy-pasted into 13 jobs and had already drifted. It also injected params as `json.loads(r'''…''')`, which is a **SyntaxError for any value containing `'''`** — reachable via `-p charset_extra=` / `-p spec=`. `Job._params_block` now uses `repr`, so no input can break the module (`tests/test_params_quoting.py`).
  - `_embedded/_common.py` is a shared prelude (`_utc_iso`, `_dataset_root`, `_norm`), concatenated **before** the runner, so a runner keeping its own copy simply overrides it — copies can be retired one at a time. Consequence: a runner read on its own is no longer standalone, hence the `F821` per-file-ignore in `pyproject.toml`. The *assembled* code is linted for undefined names in `tests/test_render_all_jobs.py`, which also asserts every job renders, strips `__main__`, and defines nothing twice.
  - Runners that also travel to Modal (`paddleocr`, `vit_classifier`, `yolo_spotter`) must **keep** their own copies: Modal ships only the runner module, without the prelude.
- Two jobs deviate on purpose: `dino_surname_verifier` passes `strip_main=False` + an argv bridge (its runner is an argparse CLI, so `__main__` *is* the entry point), and `paddleocr` passes a `prelude` with its pip fixups.
- **Modal**: `render_runner_module()` returns just the module source. `ModalBackend` `exec()`s the source inside a remote wrapper (`_REMOTE_WRAPPER` in `backends/modal.py`) that calls `main(params)` and returns `{"summary", "files", "files_skipped"}` — files are collected by walking `params["output_root"]` (forced to `/tmp/gpurunner_output`). Return-value cap is ~256 MB; binary files are listed in `files_skipped` rather than embedded (move to `modal.Volume` for v0.2).

### Backend-specific gotchas baked into the code

- **Develop on Python 3.12** (`.python-version` pins it); the package installs on 3.11+ (`requires-python = ">=3.11"`, same floor as `nyshporka`, which pulls it through its `rent` extra). **The Modal backend still needs a local 3.12**: Modal serializes the wrapper across the local/remote boundary, so local and remote Python must match, and job images are built for 3.12 — `ModalBackend.submit` refuses a mismatch before building anything. 3.13 breaks the paddle dependency chain (`paddlex` → `pandas==1.5.3` has no cp313 wheel and fails to build because `pkg_resources` is gone).
- **PaddleOCR uses paddle 3.2 + paddleocr 3.2 on both backends.** `requirements()` emits `--extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ paddlepaddle-gpu==3.2.0 paddleocr==3.2.0` — Kaggle reaches the CN paddle index since the account is phone-verified. `modal_image_spec()` inherits pip_packages + extra_index_url from the base impl and only overrides `apt_packages` (Kaggle's base image ships them pre-installed; debian-slim doesn't). The runner uses paddleocr 3.x's `predict()` API + `rec_texts/rec_scores/rec_polys` result schema.
- **Kaggle GPU strings** map through `KaggleBackend._GPU_TO_ACC`. `"T4"` becomes `acc=None` + `enable_gpu=true` (the default single-GPU), `"T4x2"` → `"GPU T4 x2"`, `"P100"` → `"GPU P100"`. Limits: 12 h/run, 30 h/week, ~20 GB output. v0.1 does **not** auto-shard around them.
- **Kaggle username discovery**: `KGAT_` access tokens don't store the username locally. `_get_username()` probes `kernels_list(mine=True)` first, falls back to `competitions_list`. Fails noisily if both come back empty (fresh KGAT account with no kernels yet → user must submit one via the UI once).
- **Modal status**: there is no clean "running" check — `FunctionCall.get(timeout=0)` is the probe. `TimeoutError` → `RUNNING`, success → `COMPLETED`, any other exception → `FAILED` carrying the exception message. `OutputExpiredError` means we waited >24h (Modal's retention).
- **`app.run(detach=True)`** in `ModalBackend.submit()` is load-bearing: without `detach`, exiting the context manager kills the ephemeral app and the in-flight `spawn()` call dies with it.

### Colab backend (Drive-mediated, semi-manual)

Free/Pro Colab has **no** API for headless submission — the only non-fragile path is
Drive + a click. `ColabBackend` (`backends/colab.py`) therefore:

1. renders a notebook and uploads it to `MyDrive/gpurunner/runs/<handle_id>/notebook.ipynb`
2. prints `https://colab.research.google.com/drive/<fileId>` — **you** hit *Runtime → Run all*
3. the notebook writes `_status.json` (with a 60 s heartbeat), `_runner.log` and `out/`
   back to the same Drive folder
4. `status` / `logs` / `fetch` poll Drive; no browser involved

**The whole trick is Kaggle-FS emulation.** The notebook `mkdir`s `/kaggle/working`, copies
declared inputs into `/kaggle/input/<slug>/`, then `exec()`s `job.render_remote_code(params)`
verbatim — the same string the Kaggle backend puts in a notebook cell. So no job needs a
Colab-specific renderer; `Job.supported_backends` just has to list `"colab"` (now the ABC default).

- `Job.colab_input_dirs(params) -> {slug: drive_folder_name}` declares inputs. Default derives
  from `dataset_sources()` (`<owner>/<slug>` → `<slug>`), so Kaggle-shaped jobs need nothing.
  `submit()` verifies each folder exists on Drive **before** uploading — a missing dataset fails
  locally in a second, not 3 minutes into a runtime.
- Inputs are staged as `gpurunner drive push <folder> --name <name>` (md5-verified; Drive has no
  Kaggle version-race, but truncated uploads are real). `--stage` is Kaggle-only and rejected here.
- `handle.volume_name` holds the run's **Drive folder id**; `remote_id` is the notebook file id.
- Status mapping: no `_status.json` → `QUEUED` ("you haven't pressed Run all"); fresh heartbeat →
  `RUNNING`; heartbeat older than 10 min → `UNKNOWN` with "the tab was disconnected" (never a
  silent forever-RUNNING). `cancel()` raises — stop the runtime in the UI.
- Outputs sync `/kaggle/working → out/` every 10 min **and** at the end, because Colab drops an
  idle tab after ~90 min and an un-synced 3-hour train would otherwise evaporate.
- `write_status` latches terminal states under a lock: the heartbeat thread must not overwrite
  `completed` with a stale `running` (tested in `test_notebook_status_is_latched_once_terminal`).

Gotchas baked into the code:

- **OAuth scope must be the full `.../auth/drive`.** Files the notebook creates via `drive.mount`
  are owned by the *user*, not by our OAuth client, so `drive.file` would hide every result.
- **Consent screen in "Testing" status expires refresh tokens after 7 days** — publish the app.
  `load_credentials()` says exactly this when a refresh fails.
- **Service accounts don't work**: a SA on a personal Google account has zero storage quota, so
  uploads into MyDrive fail. OAuth desktop flow only.
- Notebook metadata carries `accelerator: "GPU"` + `colab.gpuType` so the runtime picks the right
  card. If Colab ever stops honouring it, the GPU-guard cell **stops the run** rather than letting
  a 12-hour train crawl on CPU.
- The notebook is uploaded as `application/x-ipynb+json`, *not* `application/vnd.google.colaboratory`
  (a Google-native type you cannot create by upload).
- Reading gigabytes straight off the Drive FUSE mount is slow → inputs are copied to local disk first.

Limits: tab must stay open (Pro+ adds background execution up to 24 h), Colab Pro ≈ 100 compute
units/month ≈ 57 h of T4.

### Vast.ai backend (rented boxes — the meter runs until you destroy them)

`backends/vast.py`. Vast is a *marketplace*: you rent someone's GPU by the hour, it boots a
Docker container, you reach it over SSH. Cheapest of all backends (~$0.2/h for an RTX 3090)
but the only one that can silently burn money, so the design is defensive:

1. `submit` searches offers (`POST /bundles/`), rents the cheapest match (`PUT /asks/<id>/`),
   attaches your SSH pubkey, **blocks until SSH answers**, uploads inputs over SFTP, then
   touches a `GO` sentinel that releases the onstart script. Without that handshake the job
   would start against an empty `/kaggle/input`.
2. The onstart script ships the runner **base64-encoded** (an onstart lives inside JSON and is
   then re-quoted by the container shell — inlining source is a quoting minefield) and runs it
   under `timeout <max_hours>`.
3. `status` combines the *instance* state (`actual_status`) with the *job* state (`_status.json`
   over SFTP, same schema and terminal-latch as Colab) and always prints **accrued $** plus a
   destroy reminder once the job is done.
4. `cancel` = `DELETE /instances/<id>/` — the only thing that stops billing. `fetch` first.

Remote layout mirrors Kaggle again: `/kaggle/input/<slug>/` (uploaded by us), `/kaggle/working/`
(fetched by us), `/workspace/gpurunner/{job.py,_status.json,_runner.log,GO}`.

- Backend-only params are read from the **raw** params (jobs' `validate_params` drops unknown
  keys): `inputs` `input_root` `image` `disk` `max_price` `max_hours` `autodestroy_hours` `num_gpus`.
- Inputs: `-p input_root=<dir>` (each declared slug is a subdir) or explicit
  `-p inputs='{"slug": "/local/dir"}'`. Uploads are size-verified after transfer.
- `-p autodestroy_hours=N` arms an opt-in self-destruct N hours after the job ends (uses the
  `CONTAINER_API_KEY`/`CONTAINER_ID` env Vast injects). **Off by default** — destroying the box
  before `fetch` would throw the results away.
- `gpurunner vast offers --gpu RTX4090 --max-price 0.4` prices the market before you commit.
- Auth: `VAST_API_KEY` env → `config_dir()/vast_api_key` → the official CLI's paths
  (`%APPDATA%/vastai/vast_api_key`, `~/.config/vastai/vast_api_key`, `~/.vast_api_key`). Plus an
  SSH key pair (`~/.ssh/id_ed25519`, or `GPURUNNER_VAST_SSH_KEY`) — that is how outputs come back.
- API paths are taken from `vast-ai/vast-cli`'s `vast.py`, not from the docs site (which
  describes some endpoints that do not exist). Create returns `{"new_contract": <instance_id>}`.

### Balances (`gpurunner balance`)

`Backend.balance() -> BalanceReport` (`core/models.py`), default "unknown". What each provider
actually exposes (all verified live, 2026-07-21 — the docs are wrong in both directions):

| Backend | Number | Source |
|---|---|---|
| **lightning** | credits left | `billing_service_get_project_balance(project_id=<teamspace id>)` |
| **vast** | prepaid $ + live burn rate | `GET /users/current` → `credit`, plus summed `dph_total` of running instances |
| **kaggle** | **AI** quota $/day (not GPU!) | `benchmarks.get_benchmark_task_quota()` → `daily_quota_used` / `total_daily_quota_allowed` |
| **modal** | month-to-date spend + **estimated** remaining | `modal.billing.workspace_billing_report(start=…, end=…)`, minus `GPURUNNER_MODAL_MONTHLY_CREDIT` (default 30; Team = 100) |
| **colab** | — | nothing; compute units are UI-only |
| **beam** | — | no credit endpoint at all; the row only states the plan ($30/month, refreshed) and links to the dashboard |
| **saturn** | compute *used* this month | `get_user_usage(org_id, user_id, start, end)` — daily rows; only a key literally naming hours is summed |

Traps encoded in the code:

- Lightning: use the **project** balance. `billing_service_get_user_balance()` counts only
  *purchased* credit and reads `0.0` on a free account (its `total_spent` is still worth showing).
- Kaggle: the SDK has **no** weekly-GPU-quota endpoint anywhere. The quota it does expose is the
  model-proxy/benchmarks one in USD — the row labels it "НЕ GPU" so the two can't be conflated.
- Modal: `workspace_billing_report` is documented as Team/Enterprise but works on a personal
  workspace; it returns plain dicts with `Decimal` costs (`r["cost"]`, not `r.cost`). There is no
  remaining-credit endpoint, so "remaining" is *derived* — free monthly allowance minus spend — and
  the row says **ОЦІНКА** so it can't be mistaken for an authoritative balance.
- Beam and Saturn expose **no remaining-credit endpoint**, and unlike Modal there is no documented
  monthly constant to subtract usage from, so both report `available=None`. Saturn's usage payload is
  undocumented: `_sum_usage_hours` only trusts a key whose name literally contains "hour" and says
  "формат не розпізнано" otherwise, instead of guessing at a number.
- Every lookup degrades to a note instead of raising; an unauthenticated backend raises `AuthError`
  and the CLI renders "не налаштовано" rather than failing the whole table. Never invent a number.

**Kaggle cancel — closed as impossible (probed live 2026-07-21, don't retry):** `kagglesdk` ships
`cancel_kernel_session`, but it takes a `kernel_session_id` that **no** endpoint returns —
`kernels_push` gives `kernel_id` (+ `url`, `version_number`), `kernels_status` gives only a status
enum. Calling the endpoint with the kernel id *and* with the version number both answer
**403 Forbidden**, i.e. it is gated to the browser session. `KaggleBackend.cancel` therefore raises
with that explanation and a link. (A `QUEUED` kernel costs no quota, so this is rarely urgent.)

### Dashboard (`gpurunner dash`)

`web/app.py` + `web/static/index.html`, extra `web` (`fastapi`, `uvicorn`). Localhost-only, no
auth. Answers the two questions `balance` could not: *how much is left where the provider has no
API*, and *how much compute is that, in total*. Four new `core/` modules, none of which touch a
backend:

- **`core/balances.py`** — the collect-and-render logic that used to live inside `cli.balance`
  (one broken backend must not hide the table). Also decides which number is the headline: for
  `kaggle`/`colab`/`saturn` the local calculation wins and the API answer moves to a note. That
  is load-bearing for Kaggle — its `balance()` returns the **AI quota in dollars**, and showing
  that where the user reads GPU-hours is the single most confusable failure here.
- **`core/usage.py`** — GPU-hours from the manifest. Duration comes from four sources, ranked and
  labelled in the UI: runner-reported `elapsed_s`/`wall_sec` → `running`→terminal journal events →
  submit→terminal (includes queue, flagged) → still-open.
- **`core/quota.py`** — `quota.sqlite3` (same pattern as `budget.py`). Anchors, per-backend
  allowance/period overrides, manual GPU factors, balance snapshots.
- **`core/gpu_equiv.py`** — T4-equivalence.

**The whole design rests on one admission**: the local count sees only runs submitted through
gpurunner. A session opened by hand in Kaggle's UI does not exist for it, so the computed figure
is a *lower bound on spend* — hence an upper bound on what's left. The **anchor** is the fix, not
a workaround: the user reads the real number off the provider's page, types it in, and it becomes
the new origin. `remaining = anchor − usage(after anchor)`; with no anchor in the current period,
`allowance − usage(since period start)`. An anchor from a previous period is ignored, so a reset
returns to the allowance.

Things live data forced into the code (all found by pointing it at the real 290-handle manifest):

- **A non-terminal status is not a running job.** 39 handles sat in `queued` since May — submitted,
  never polled again. Counting them "from submit to now" gave Kaggle **1106 hours spent in a
  168-hour week**. A run is now counted only up to `updated_at` (the last time we actually observed
  it) unless that is fresher than `STALE_AFTER` (6 h); older ones are `SRC_STALE`, excluded from
  spend, and listed separately in the UI so they don't bury the live runs.
- **`SESSION_CAP_H`** bounds one run by the provider's own session limit (Kaggle/Colab: 12 h).
  Anything longer is a corrupt timestamp, not a cost. Providers without a hard limit get no cap —
  there would be nothing to derive it from.
- **Runs are clipped to the window, not included/excluded by start.** Anchoring mid-train would
  otherwise either drop hours or count them twice. `summary`-sourced time is scaled by the overlap
  *fraction*, because the measured time is shorter than the start→end span (queue and container
  startup are not in it).
- **Charge mode differs per provider** (`CHARGE_SESSION` / `CHARGE_T4_UNITS` / `CHARGE_HOURS`).
  An hour on `T4x2` costs Kaggle's 30-hour quota exactly one hour (it bills the *session*), but
  costs Colab twice the units (it bills *compute*). Conflating them is a 2× error either way.
- **Colab's plan changes the *unit*, not just the number.** Compute units exist only on Pro /
  pay-as-you-go; the free tier has none at all and Google publishes no hour limit either. So
  `quota.PLANS["colab"]` holds two configs — `free` (unit `год`, no allowance, `CHARGE_HOURS`)
  and `pro` (unit `units`, 100/month, `CHARGE_T4_UNITS`) — and **free is the default**. Defaulting
  to Pro credited a free account with ~57 T4-hours it does not have, which was 26% of the
  headline total. Switching plans clears the stored overrides: `100` entered as units must not
  survive as a limit in hours. `POST /api/quota/colab {"plan": "pro"}`, or the buttons on the card.
  `quota_config` gains its `plan` column through `_migrate()` — `CREATE TABLE IF NOT EXISTS` is
  silent on an existing table, so a schema addition would otherwise never reach anyone who had
  already opened the dashboard.

**T4-equivalence — where the numbers come from, and where they deliberately stop.** Measured
factors come from `jobs/yolo_spotter.py::MEASURED_IT_S` (T4 2.2 it/s, L4 3.1, A100 8.1 → 1.00 /
1.41 / 3.68). For unmeasured cards the factor is `FP16-dense-TFLOPS ÷ T4 × 0.76`; the 0.76 is not
a fudge, it is *derived* from those same two measurements (L4: 1.86 spec → 1.41 measured = 0.757;
A100: 4.80 → 3.68 = 0.767) and reproduces both to within 2%, which `tests/test_gpu_equiv.py`
asserts.

🔴 **The spec proxy provably fails outside datacentre tensor cards, so those cards have no factor:**

- **GeForce (RTX 3090/4090/5090)** — NVIDIA halves the FP16-with-FP32-accumulate path on GeForce
  (that is the path mixed-precision training uses), so the formula yields RTX3090 ≈ 0.42…0.83× T4,
  i.e. *slower than a T4*, which no field report supports. What TFLOPS omits is exactly what
  matters here: 936 GB/s vs T4's 320, 350 W vs 70 W. Two calibration points cannot fit a
  two-factor model, and fitting two points with two parameters is not a model.
- **P100** — Pascal has no tensor cores at all, so its 19 TFLOPS is plain vector FP16 and does not
  compare with anyone's tensor figure (the formula says 0.22× T4). Kaggle's quota bills session
  hours anyway, so nothing needs it.
- **RTXPro6000, B200** — sources disagree by 2×.

Those cards return `None`, are listed under "не переведено" / "без множника" instead of silently
contributing zero, and can be given a hand-measured multiplier (`gpu_factor` table,
`POST /api/factors/<gpu>`) which then outranks everything else.

Money converts at a real rate where one exists: Modal via its published `$0.59/h` T4, Vast via a
live `search_offers` query for the cheapest A100 divided by our *measured* 3.68 (the firmest
conversion of the lot), Beam — which publishes no T4 price — via the best T4-hours-per-dollar
among the cards it does price. Colab and Lightning convert at their published plan ratios
(100 units ≈ 57 h; 15 credits ≈ 22 h), which is why they are marked ◐ оцінка, not ● вимір. The
header always shows how much of the total is estimate and what was left out.

Other gotchas:

- `run_blocking()` uses **its own daemon threads**, not `asyncio.to_thread`: the default executor's
  threads are non-daemon and are joined at interpreter exit, so one hung SDK call would leave the
  process un-killable after Ctrl+C (the Windows symptom in this file's last section).
- `manifest.events_bulk()` exists because the dashboard walks every handle; `events()` per handle
  meant one connection *plus* schema script *plus* legacy-import probe ~300 times per page load.
- The cache is 5 min and in-process; any write endpoint drops it so the next read shows the new
  number. Balance snapshots are written on every successful refresh, so the history chart is
  sparse and unevenly spaced by construction — the axis is labelled rather than smoothed.

### Lightning AI backend (headless; free credits discontinued 2026-08)

`backends/lightning.py`. The only backend besides Modal with a real fire-and-forget API.
No click like Colab, no rented box like Vast.

🔴 **Out of service — the account has 0 credits and will not be topped up (2026-08-09).**
Lightning's email of 2026-08 states that **July 2026 was the last month free users got the 15
monthly credits**; what remains is a **one-time grant of 25 credits that requires attaching a
payment method**, and the decision here is *not* to attach one. So Lightning joins Saturn (0 h
granted) and Vast ($0 prepaid) as a backend that is implemented and verified but cannot currently
run anything — don't offer it as a run target and don't count it as available GPU time. The code
stays: it works, and a top-up revives it instantly.

Do **not** describe Lightning as a recurring free tier anywhere. The `15 credits ≈ 22 h of T4`
figure survives *only as a conversion rate* (`core/balances.py::_CREDITS_PER_T4_HOUR`) — it says
what a credit is worth, not what arrives monthly.

Everything moves through the **teamspace drive**, which is symmetric on both ends —
`Teamspace.upload_folder`/`download_folder`/`download_file` locally, `/teamspace/studios/this_studio/…`
inside the job:

```
gpurunner/data/<slug>/        inputs (uploaded at submit)
gpurunner/runs/<handle_id>/
  out/                        outputs (synced from /kaggle/working every 10 min + at the end)
  _status.json  _runner.log   status/log polling — same schema/latch as Colab & Vast
```

**SDK landmines — all four found by running it for real (2026-07-21), all worked around:**

| What looks fine | What actually happens | What we do |
|---|---|---|
| `Teamspace.upload_folder` | **silent no-op** — nothing lands, no error | walk the dir, upload file-by-file |
| `Teamspace.download_folder` | creates correctly-named **0-byte** files | `list_files` + `download_file` per blob |
| `Teamspace.upload_file` | runs the remote path through `os.path.normpath`, so **on Windows** `a/b.txt` becomes `a\b.txt` and the blob is lost under a mangled key — upload still "succeeds" | call `ts._teamspace_api.upload_file(...)`, which takes the path verbatim |
| job writes to `/teamspace/...` | every mount is **read-only** in a job (uploads, jobs/*/artifacts, studio home), and the studio home is an **ephemeral copy** — writes vanish | job pushes results through the API instead |

After every upload the backend **re-lists the prefix and fails if files aren't there** — a silent
no-op must not turn into "job started, found no data" three minutes into a GPU run.

- **Jobs borrow a Studio's environment** (`-p studio=<name>`, required).
- **Jobs run unprivileged** (user `zeus`): `/kaggle` cannot be created. The wrapper probes it and
  falls back to `~/gpurunner_kaggle`, rewriting the job body's hardcoded `/kaggle/` paths to match.
- **We deliberately do not use `job.artifact_path` / `job.logs`**: `artifact_path` is `None` for
  image jobs and read-only for studio jobs, and `job.logs` **raises while the job is running**
  (SDK: "Getting jobs logs while the job is pending or running is not supported yet"). Our own
  `_runner.log` on the drive has neither problem.
- Teamspace lookup needs an owner (`Teamspace(name=…, user=<username>)`) — a bare name raises
  "Neither user or org are specified".
- `Machine` names re-verified against lightning-sdk **2026.7.31**: `CPU T4 T4_X_2 L4 L40S A100
  H100` all still exist and **there is still no `A10G`**, so don't reintroduce it from old docs.
  That release also exposes `H200`, `B200_X_8`, `RTXP_6000`, `A100_80GB` and `_X_4`/`_X_8`
  variants of most cards — our `gpu_choices` is a deliberate subset, not an outdated one.
- Backend-only params (raw): `studio` `teamspace` `inputs` `input_root` `max_hours` `interruptible`.
  `max_hours` maps to `max_runtime` (SDK default is only 3 h — we pass 12 h).
- Auth: `gpurunner auth lightning --user-id … --api-key … --teamspace …` stores everything in
  `config_dir()/lightning.json` and exports it into the env for the SDK; `LIGHTNING_USER_ID` /
  `LIGHTNING_API_KEY` env vars still win if set. Teamspace override:
  `GPURUNNER_LIGHTNING_TEAMSPACE`, org override: `GPURUNNER_LIGHTNING_ORG`.
- `cancel` → `job.stop()`. Nothing keeps billing after a job ends (unlike Vast).

### Beam backend (Modal-shaped, $30/month free credit)

`backends/beam.py`. Beam (beam.cloud) is the closest analogue to Modal — Python-native
serverless, per-second billing — and, since Lightning stopped its monthly credits in 2026-08,
the only *recurring* free tier besides Modal: **$30 of credit refreshed monthly** on the Developer plan, no card
(beam.cloud/pricing, re-checked 2026-08-09 — the page says verbatim "$30 free credit refreshed
monthly"; earlier sources said "one-time ~10-15 h", which is no longer what the pricing page says).
Two things the wording does **not** settle, so don't assume either: whether unused credit **rolls
over** (nothing published; treat it as use-it-or-lose-it), and whether the refresh survives having
a card attached — this account has one. `check_run_ceiling`'s monthly ceiling
(`GPURUNNER_BEAM_MONTHLY_BUDGET`, default $30) is a *local* ledger, not a reading of Beam's
balance, so it cannot detect a refresh that failed to happen. Only the dashboard can.

```
inputs   → Beam Volume "gpurunner-data"  (<slug>/…), mounted at ./inputs
outputs  → Beam Volume "gpurunner-out-<handle>",     mounted at ./outputs
           out/ + _status.json + _runner.log — same schema/latch as Colab, Vast, Lightning
```

- **The submit path is a generated entry module, not `exec()` like Modal.** Beam never ships a
  callable: `prepare_runtime` resolves the decorated function to `<module file relative to
  CWD>:<name>` and syncs the **current working directory** into the container
  (`beta9/sync.py`, `FileSyncer(root_dir=".")`). So `submit()` writes a self-contained
  `entry.py` (job body embedded as a string literal) into a temp dir, `chdir`s there, imports
  it, and calls `.put()`. Submitting from the repo root would upload the whole checkout.
- `@task_queue(...).put()` returns a `Task` (`.id`) — `@function.remote()` does not, so the
  task queue is the only spawn primitive with a retrievable id.
- `retries=0` is deliberate: the SDK default is **3**, i.e. a crashing 12-hour train would be
  re-run three times on free credit.
- **`Image` has no `extra_index_url`** (Modal's does). `_image_spec_to_beam()` folds a job's
  extra index into an explicit `pip install --extra-index-url …` command, and installs
  `apt_packages` too (Kaggle's base image ships them; a fresh Beam container does not).
- GPU names differ from Modal's: `A100-40`/`A100-80`, no bare `A100`. Full list lives in
  `beta9.type.GpuType`; ours maps the gpurunner names onto it.
- Volume I/O from the laptop goes through `beta9.multipart.beta9_upload/beta9_download` +
  `service.volume.list_path`, **not** `Beta9Handler`: the handler rewrites remote paths based
  on file suffixes and calls `local_path.relative_to(Path.cwd())`, which raises for any output
  dir outside the CWD (i.e. most of them on Windows). The gRPC stubs are synchronous.
- `check_auth`/`discover_credentials` export `BEAM_TOKEN` **before** the SDK is touched: with no
  token in sight `get_config_context()` opens an interactive stdin prompt and a non-interactive
  `gpurunner run` would hang there. `import_sdk()` then re-instantiates `SDKSettings` because
  `beta9.config` caches the first snapshot (which would still say "no token").
- **`ServiceClient()` must be given the context explicitly** — `ServiceClient(get_config_context())`.
  The no-arg form ends up in `get_channel(None)`, which *also* prompts ("Context Name [default]:")
  and hangs, even when the token is perfectly valid (hit live on 2026-08-02: `auth beam --verify`
  froze until the tool timeout; `faulthandler` pinned it at `channel.py:165`). Always go through
  `auth/beam.py::service_client()`.
- **Two things only a live run reveals** (smoke runs 99058bb9 / eb380a2b, 2026-08-02):
  - `mount_path` **must be relative** (`./outputs`). An absolute `/outputs` is silently
    ignored: the task runs, the wrapper `mkdir`s a plain local directory, reports "1 file
    written" — and the volume stays empty. Nothing anywhere errors.
  - `list_path` lists **one directory and does not glob**. `…/**` and `…/*` both answer
    `ok=True` with an empty list, so a one-shot listing reports an empty volume and `fetch`
    quietly returns 0 files. `_list_volume` walks the tree instead. Beam's own
    `Beta9Handler.list_dir` passes `/**` — do not copy it.
- **Spend guard (`check_run_ceiling` + `core/budget.py`)**: Beam is the only backend where a
  card is attached *and* the API exposes neither balance nor usage, so the ceiling is enforced
  locally, **before** inputs are uploaded or an image is built. Two limits: per run
  (`--allow-cost` / `-p max_cost` / `GPURUNNER_BEAM_MAX_RUN_COST`, default $1) and per month
  (`GPURUNNER_BEAM_MONTHLY_BUDGET`, default $30 = the free credit). Everything is booked at
  `timeout × rate` — the worst case, because a hung job costs exactly that — and settled down
  to the runner's reported `elapsed_s` once the run ends. A GPU with no published rate is
  **refused**, not assumed cheap (`-p price_per_hour=` overrides). The ledger is SQLite, not
  JSON, so the check-then-write is one `BEGIN IMMEDIATE` and two concurrent submits cannot
  both slip under the same remaining budget.
- 🔴 **The price page is not the invoice.** A real run (taskqueue/entry:execute, RTX4090, 8 cores,
  32 GB, 21m32s, 2026-08-09) billed at **$2.8414/h** and burned $1.020 of credit. The published
  rates give $1.2919/h for that same container — the invoice is **2.2× larger**. Nothing public
  explains the gap: the pricing page still showed $0.000191667/s for the 4090 that day, and
  `docs.beam.cloud/v2/resources/pricing-and-billing` says nothing about minimum billable
  configurations, rounding, or CPU/RAM bundled into the card price. So `_GPU_MEASURED_HOURLY`
  holds the **invoiced** figure (the gap attributed to the GPU term, assuming the published
  CPU/RAM rates hold: 2.8414 − 8×0.045 − 32×0.00756 = 2.2395), and every other card gets that
  same 3.25× markup via `_UNVERIFIED_MARKUP` until an invoice says otherwise. The asymmetry is
  deliberate: an under-quoted ceiling spends real money, an over-quoted one refuses a submit and
  `--allow-cost` lifts it. Give any newly-invoiced card its own row instead of leaning on the
  markup. `core/balances.py::_t4_from_dollars_beam` goes through the same corrected rates —
  costing purchasable T4-hours at list price would overstate the dashboard threefold.
- **Costs**: `estimate_cost()` in `backends/beam.py` mirrors the Modal one and is wired into
  `cli._print_cost_estimate`. Rates in `_GPU_HOURLY` are the **serverless** per-second prices from
  beam.cloud/pricing ×3600 (checked 2026-08-02, re-checked 2026-08-09), plus $0.0000125/s per
  *physical* core and $0.0000021/s per GiB — but read them through `_gpu_hourly()`, never
  directly. No fudge factor (unlike Modal): Beam bills per second with no minimum.
  T4 / L4 / A10G / A100-40 are **absent from the price list** — they stay in the GPU map but get no
  rate, and the estimate stays silent rather than inventing one. `default_gpu` is therefore
  `RTX4090` ($0.69/h, 24 GB) — the cheapest card Beam actually prices for serverless.
  The dashboard's «Compute» tab is the *marketplace* for dedicated instances and is billed
  differently; do not mix those numbers in.
- Backend-only params (raw): `inputs` `input_root` `max_hours` `cpu` `memory` `retries`
  `data_volume`. `max_hours` → the task's `timeout` (SDK default is 1 h).
- `cancel` → `service.gateway.stop_tasks(StopTasksRequest(...))`. Nothing bills after a task ends.

### Saturn Cloud backend (recipe + sfs, free tier)

`backends/saturn.py`. Saturn Cloud is a hosted JupyterLab/jobs platform whose free tier gives
GPU instances. A *job* is a first-class resource created from a
**recipe** (`PUT /api/recipes`), started over HTTP, polled for pod status and logs — no click.

```
sfs://<org>/<user>/gpurunner/data/<slug>/       inputs (uploaded at submit)
sfs://<org>/<user>/gpurunner/runs/<handle>/
  job.py                                        the wrapper, fetched by the container
  out/                                          outputs (synced from /kaggle/working)
  _status.json  _runner.log                     status/log polling — same schema/latch
```

- **Everything moves over `saturnfs`** (Saturn's own fsspec filesystem, `sfs://`), symmetric on
  the laptop and inside the job. The path layout `sfs://<org>/<username>/…` comes from
  `saturn_client.file_syncs.get_default_sfs_base_dir_url`, not from guesswork.
- `saturnfs.settings` reads `SATURN_BASE_URL`/`SATURN_TOKEN` from `os.environ` **at import
  time** and raises `KeyError` if absent — `auth/saturn.py` exports both before importing it.
- The container bootstrap is `pip install -q saturnfs && saturnfs cp <sfs job.py> /tmp && python
  -u …`; the wrapper asserts both env vars exist and dies immediately if Saturn did not inject
  them, rather than failing deep inside fsspec mid-train.
- **`instance_type` is resolved from the account's own catalogue** (`list_options("sizes")`) —
  Saturn's size names are per-deployment and a hardcoded map would silently rot. On the community
  free tier (verified live 2026-08-02, 93 sizes) they look like `g4dnxlarge` (T4), `g52xlarge`
  (A10G), `p3xlarge` (V100), `p4d24xlarge` (A100), `nebius/nebius-1xh100`,
  `k0rdent/shadeform-…` (L40S/H100 via Shadeform). `pick_instance_type()` matches the requested
  card against that catalogue; `-p instance_type=…` overrides.
  - Matching is **two-pass, not substring**: `"l4" in "Shadeform L40S"` is true, so a plain
    substring test would rent an L40S for an L4 request. First a strict token match, then a
    looser one that only forbids a trailing digit — needed because `nebius-1xh100` glues the
    model to the GPU count. `gpu_type` is useless for this: community reports it as `"NVIDIA"`.
- **Costs are denominated in hours, not dollars**: the community catalogue returns
  `price_per_hour: null`, so `estimate_cost()` reports the run's **hours** and only adds `$`
  when a deployment actually prices the box. **Do not quote a monthly allowance**: Saturn
  publishes none — no balance endpoint, no quota endpoint, no figure in the UI (checked
  2026-08-02). Blog posts citing "30 h/month" are second-hand and were removed from the code.
- **`-p image=…` is required** (like Lightning's `-p studio=`): Saturn images are per-deployment,
  there is no portable default. `gpurunner saturn sizes` prints the catalogue.
- Jobs run as `jovyan`, so `/kaggle` usually cannot be created — the wrapper probes and falls
  back to `~/gpurunner_kaggle`, rewriting the job body's hardcoded `/kaggle/` paths (same trick
  as Lightning).
- Backend-only params (raw): `image` `instance_type` `inputs` `input_root` `disk_space` `owner`
  `pip`.
- `cancel` → `conn.stop("job", <resource id>)`. A finished job stops billing; the resource stays
  in the UI until deleted by hand.
- **Confirmed live (2026-08-02, community tier)**: auth, `current_user`/`primary_org`, the
  93-entry size catalogue, and the **recipe schema** — `apply()` accepted
  `name owner image instance_type description command working_directory start_dind` and created
  the job resource. Images are not in `list_options`; the working ones come from
  `list_resource_templates()` → `recipe.spec.image`: `saturncloud/saturn-python:2025.05.01`,
  `…-pytorch:…`, `…-llm:…`, `saturn-r:…`.
- 🔴 **The community free tier may grant zero compute.** `start()` answered
  `OrgErrors.UPGRADE_TO_PRO` — *"Organization limits: Max of 0 hours of resource usage"* — so no
  job runs at all without Pro, whatever the size catalogue advertises. `submit` recognises that
  error, **deletes the resource `apply` just created** (otherwise every attempt leaves a dead job
  in the UI) and says so plainly. Everything up to `start` is therefore exercised; the wrapper
  itself has never run on a Saturn container.

### Auth

`auth/lightning.py` — Lightning AI keys + teamspace resolution. `auth/beam.py` — Beam token
(`BEAM_TOKEN` env → `config_dir()/beam.json` → `~/.beam/config.ini`), exported into the env before
the SDK loads. `auth/saturn.py` — Saturn URL + token (`SATURN_BASE_URL`/`SATURN_TOKEN` env →
`config_dir()/saturn.json`), likewise exported because `saturnfs` reads them at import time.
`auth/vast.py` — API key + SSH key
discovery for Vast. `auth/google.py` — OAuth desktop flow for the Colab backend. `google_client_secret.json` (Desktop-app
OAuth client, put there by hand) and `google_token.json` (written by `gpurunner auth google --login`)
both live in `config_dir()`. `auth/kaggle.py` accepts both `~/.kaggle/access_token` (new KGAT tokens) and `~/.kaggle/kaggle.json` (legacy). `auth/modal.py` expects `~/.modal.toml` from `modal token new`. Both backends' `check_auth()` raise `AuthError` (a `RuntimeError` subclass) which the CLI maps to exit code 2 (auth) / 3 (submit failure).

## Conventions

- Embedded runners live in `_embedded/`. Anything in there is shipped as source; do not add imports of other gpurunner modules. Shared helpers go in `_embedded/_common.py` (see the runner-pattern section), not into a fourth copy.
- New jobs go in `jobs/<name>.py`, registered in `jobs/__init__.py` (`get_job` table + `list_jobs`). Declare `supported_backends` honestly — backends raise `BackendError` if a job omits them. Render via `Job.render_kaggle_code(...)`; a hand-rolled `PARAMS` block is a bug waiting for a value with a quote in it.
- **Long trains must survive the session limit.** Kaggle kills the kernel at 12 h and `/kaggle/working` is not guaranteed to survive it. Two mechanisms, both already in the train jobs: a `wall_limit_h` self-stop that leaves time to write results, and resume — `parseq_train` writes `parseq_last.pt` (weights + optimizer + scheduler + counters) every epoch and continues from it with `-p resume=true -p resume_dataset=<slug>`, guarded by a fingerprint check. `kraken_train` can only warm-start (`ketos` has no `--resume`); its `ktrain_summary.json` carries a ready `continue_with` command.
- New backends go in `backends/<name>.py`, registered in `backends/__init__.py` (class table **and** `BACKEND_NAMES`, which drives `gpurunner ls` / `balance`). Implement `check_auth`, `submit`, `status`, `fetch_outputs`; override `logs`/`cancel` if the API allows.
- Staging local dirs into `/kaggle/input/<slug>` is shared: `core/inputs.py::resolve_inputs` implements the `-p inputs=` / `-p input_root=` contract for every Kaggle-FS backend (Colab, Vast, Lightning, Beam, Saturn). Don't re-implement it per backend.
- Ruff config: line length 100, py312 target, rule sets `E F I B UP SIM RUF` (`E501` ignored — formatter handles it; `F821` ignored for `_embedded/*_runner.py`, see above). The repo is **not** ruff-clean overall — hundreds of `RUF001/002/003` come from Ukrainian comments — so judge a change by the `F`-rules and by whether it added anything new, not by the total count.
- `manifest.update()` stamps `updated_at` itself, and every status writer must also carry `rep.error` into `handle.error`. Both were skipped by every call site for 284 runs, which left the manifest unable to answer "what failed and when".
- pytest uses `asyncio_mode = "auto"` and `--strict-markers`.
- `examples/` are real end-to-end scripts (they spend quota); they are not unit tests.

## Windows-specific

- `Bash` tool can hang past Python's exit if a non-daemon thread is alive (typical with aiosqlite / unclosed asyncio loops). Check `taskkill //F //IM python.exe` if you see zombie pythons accumulate.
- Do NOT pipe long `git diff` / `git log` output through `head`/`tail` — SIGPIPE returns exit 1 here. Use `git log -5` or `2>&1` instead.
