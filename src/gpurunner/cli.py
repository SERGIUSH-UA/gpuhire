"""gpurunner CLI — top-level entry point.

Commands:
    gpurunner auth kaggle [--verify]
    gpurunner auth modal  [--verify]
    gpurunner auth google [--login] [--verify]                  # Colab (через Drive)
    gpurunner auth vast   [--verify]                            # Vast.ai (оренда $/год!)
    gpurunner auth lightning [--verify]                         # Lightning AI (free ~22 год T4/міс)
    gpurunner auth beam   [--token T] [--verify]                # Beam.cloud (free $30/міс)
    gpurunner auth saturn [--url U --token T] [--verify]        # Saturn Cloud (free tier)
    gpurunner ls
    gpurunner balance [--backend B] [--json]                    # скільки лишилось грошей/кредитів
    gpurunner vast offers [--gpu RTX3090] [--max-price 0.3]     # ціни ДО оренди
    gpurunner saturn sizes [--gpu-only]                         # instance_type цього акаунта
    gpurunner run    <job>  [--backend kaggle] [--param k=v ...] [--gpu T4] [--out DIR]
                            [--stage DIR] [--expect-log STR] [--open]
    gpurunner dataset push  <DIR> [-m MSG] [--wait/--no-wait]   # анти version-race
    gpurunner dataset files <owner/slug>
    gpurunner drive   push  <DIR> --name NAME                   # вхідні дані для Colab
    gpurunner drive   files NAME
    gpurunner sweep  <job>  --params-file F  [--max-concurrent N] [--out-root DIR]
    gpurunner status [HANDLE]
    gpurunner fetch  HANDLE... | --all  [--resume] [--backend B]
    gpurunner cancel HANDLE... | --all-running [--backend B] | --from-file FILE
    gpurunner watch  HANDLE [--poll 30]
    gpurunner logs   HANDLE
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gpurunner import __version__
from gpurunner.auth import google as google_auth
from gpurunner.auth import kaggle as kaggle_auth
from gpurunner.auth import modal as modal_auth
from gpurunner.auth import vast as vast_auth
from gpurunner.backends import BACKEND_NAMES, get_backend
from gpurunner.core import AuthError, Backend, BackendError, JobStatus, contract, manifest
from gpurunner.core.models import JobHandle
from gpurunner.jobs import get_job, list_jobs

app = typer.Typer(
    name="gpurunner",
    help="Запуск GPU-задач на Kaggle/Modal тощо через єдиний інтерфейс.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


# ---- top-level options ----------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"gpurunner [bold]{__version__}[/bold]")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Показати версію gpurunner і вийти.",
    ),
) -> None:
    """gpurunner — запуск GPU-задач на віддалених бекендах."""
    return None


# ---- auth -----------------------------------------------------------------


auth_app = typer.Typer(help="Облікові дані бекендів: показати, зберегти, перевірити.")
app.add_typer(auth_app, name="auth")


@auth_app.command("kaggle")
def auth_kaggle(
    verify: bool = typer.Option(False, "--verify", help="Перевірити ключі запитом до Kaggle API."),
) -> None:
    """Показати, де лежать ключі Kaggle; за бажанням — перевірити їх."""
    try:
        creds = kaggle_auth.discover_credentials()
    except AuthError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None

    console.print(f"Source:  [cyan]{creds.source}[/cyan]")
    console.print(f"Path:    {creds.path}")
    if creds.username:
        console.print(f"User:    {creds.username}")

    if verify:
        try:
            username = kaggle_auth.verify()
        except AuthError as e:
            err_console.print(f"[red]Verify failed: {e}[/red]")
            raise typer.Exit(code=3) from None
        console.print(f"[green]Verified — authenticated as[/green] [bold]{username}[/bold]")


@auth_app.command("modal")
def auth_modal(
    verify: bool = typer.Option(False, "--verify", help="Перевірити конфіг Modal."),
) -> None:
    """Показати, де лежить конфіг Modal; за бажанням — перевірити його."""
    path = modal_auth.modal_config_path()
    if not path.exists():
        err_console.print(f"[yellow]No Modal config at {path}.[/yellow]")
        err_console.print("Run: [bold]uv sync --extra modal && uv run modal token new[/bold]")
        raise typer.Exit(code=2)
    console.print(f"Config:  {path}")
    if verify:
        try:
            modal_auth.verify()
        except (AuthError, NotImplementedError) as e:
            err_console.print(f"[yellow]{e}[/yellow]")
            raise typer.Exit(code=2) from None


@auth_app.command("google")
def auth_google(
    login: bool = typer.Option(False, "--login", help="Пройти OAuth у браузері і зберегти токен."),
    verify: bool = typer.Option(False, "--verify", help="Перевірити Google Drive збереженим токеном."),
) -> None:
    """Ключі Google Drive для бекенда Colab: показати шляхи, увійти, перевірити."""
    try:
        creds = google_auth.discover_credentials()
    except AuthError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None

    console.print(f"Client:  {creds.client_secret}")
    console.print(f"Token:   {creds.token}  {'[green]present[/green]' if creds.has_token else '[yellow]absent[/yellow]'}")

    if login:
        try:
            email = google_auth.login()
        except AuthError as e:
            err_console.print(f"[red]Login failed: {e}[/red]")
            raise typer.Exit(code=3) from None
        console.print(f"[green]Logged in as[/green] [bold]{email}[/bold]")
        return

    if verify:
        try:
            email = google_auth.verify()
        except AuthError as e:
            err_console.print(f"[red]Verify failed: {e}[/red]")
            raise typer.Exit(code=3) from None
        console.print(f"[green]Verified — authenticated as[/green] [bold]{email}[/bold]")


@auth_app.command("vast")
def auth_vast(
    verify: bool = typer.Option(False, "--verify", help="Перевірити ключ запитом до Vast.ai API."),
    key: str = typer.Option(
        "", "--key",
        help="Зберегти ключ API у конфіг gpurunner (`-` — прочитати зі stdin, "
             "щоб ключ не лишався в історії оболонки).",
    ),
) -> None:
    """Ключ API і SSH-ключ Vast.ai: показати, де лежать, зберегти, перевірити."""
    if key:
        # 🔴 Сам ключ не друкується ніде — ні тут, ні в помилці: вивід цієї
        # команди потрапляє в логи агентів і в скриншоти звернень по допомогу.
        raw = sys.stdin.readline() if key == "-" else key
        try:
            saved = vast_auth.save_api_key(raw)
        except AuthError as e:
            err_console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=2) from None
        console.print(f"[green]Ключ API збережено:[/green] {saved}")
        if os.environ.get("VAST_API_KEY", "").strip():
            err_console.print(
                "[yellow]Задано змінну оточення VAST_API_KEY — вона має пріоритет "
                "над збереженим файлом.[/yellow]"
            )
    try:
        creds = vast_auth.discover_credentials()
    except AuthError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None

    console.print(f"API key: [cyan]{creds.source}[/cyan]")
    if creds.ssh_private:
        console.print(f"SSH key: {creds.ssh_private} (+ .pub)")
    else:
        err_console.print(
            "[yellow]SSH key: не знайдено — вихідні файли забираються по SFTP.\n"
            "Створи: ssh-keygen -t ed25519 -C gpurunner "
            "(оренда для Нишпорки згенерує власну пару сама)[/yellow]"
        )

    if verify:
        try:
            info = vast_auth.verify()
        except AuthError as e:
            err_console.print(f"[red]Verify failed: {e}[/red]")
            raise typer.Exit(code=3) from None
        console.print(
            f"[green]Verified[/green] — {info['email']} · balance ${info['balance']} · "
            f"{info['ssh_key_count']} SSH key(s) on the account"
        )


@auth_app.command("lightning")
def auth_lightning(
    verify: bool = typer.Option(False, "--verify", help="Перевірити ключі запитом до Lightning AI API."),
    user_id: str = typer.Option("", "--user-id", help="Зберегти USER_ID у конфіг gpurunner."),
    api_key: str = typer.Option("", "--api-key", help="Зберегти API_KEY у конфіг gpurunner."),
    teamspace: str = typer.Option("", "--teamspace", help="Зберегти назву teamspace."),
) -> None:
    """Ключі Lightning AI у конфігу gpurunner: показати, зберегти, перевірити."""
    from gpurunner.auth import lightning as lightning_auth

    if user_id or api_key or teamspace:
        if bool(user_id) != bool(api_key) and not (teamspace and not user_id and not api_key):
            err_console.print("[red]--user-id і --api-key задаються разом[/red]")
            raise typer.Exit(code=2)
        if user_id and api_key:
            path = lightning_auth.save_credentials(
                user_id=user_id, api_key=api_key, teamspace=teamspace or None
            )
        else:
            # only --teamspace: keep the stored keys, update the teamspace
            try:
                stored = lightning_auth.discover_credentials()
            except AuthError as e:
                err_console.print(f"[red]{e}[/red]")
                raise typer.Exit(code=2) from None
            import os as _os

            path = lightning_auth.save_credentials(
                user_id=stored.user_id or _os.environ.get("LIGHTNING_USER_ID", ""),
                api_key=_os.environ.get("LIGHTNING_API_KEY", ""),
                teamspace=teamspace,
            )
        console.print(f"[green]✓ збережено[/green] {path}")

    try:
        creds = lightning_auth.discover_credentials()
    except AuthError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None

    console.print(f"Source:    [cyan]{creds.source}[/cyan]")
    console.print(f"Teamspace: {creds.teamspace or '(default)'}")

    if verify:
        try:
            info = lightning_auth.verify()
        except AuthError as e:
            err_console.print(f"[red]Verify failed: {e}[/red]")
            raise typer.Exit(code=3) from None
        studios: list[Any] = list(info.get("studios") or [])  # type: ignore[call-overload]
        console.print(
            f"[green]Verified[/green] — teamspace [bold]{info['teamspace']}[/bold], "
            f"studios: {', '.join(map(str, studios)) or '(жодної)'}"
        )
        if not studios:
            err_console.print(
                "[yellow]⚠ Потрібна хоч одна Studio — job позичає її оточення.\n"
                "Створи будь-яку в веб-UI, далі: -p studio=<назва>[/yellow]"
            )


@auth_app.command("beam")
def auth_beam(
    verify: bool = typer.Option(False, "--verify", help="Перевірити токен запитом до Beam API."),
    token: str = typer.Option("", "--token", help="Зберегти токен у конфіг gpurunner."),
) -> None:
    """Токен Beam.cloud у конфігу gpurunner: показати, зберегти, перевірити."""
    from gpurunner.auth import beam as beam_auth

    if token:
        path = beam_auth.save_token(token)
        console.print(f"[green]✓ збережено[/green] {path}")

    try:
        creds = beam_auth.discover_credentials()
    except AuthError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None

    console.print(f"Source: [cyan]{creds.source}[/cyan]")

    if verify:
        try:
            info = beam_auth.verify()
        except AuthError as e:
            err_console.print(f"[red]Verify failed: {e}[/red]")
            raise typer.Exit(code=3) from None
        console.print(
            f"[green]Verified[/green] — токен робочий, volume(s) на акаунті: {info['volumes']}"
        )


@auth_app.command("saturn")
def auth_saturn(
    verify: bool = typer.Option(False, "--verify", help="Перевірити ключі запитом до Saturn Cloud API."),
    url: str = typer.Option("", "--url", help="URL інстансу Saturn (https://app...saturnenterprise.io)."),
    token: str = typer.Option("", "--token", help="API-токен Saturn Cloud."),
) -> None:
    """Ключі Saturn Cloud у конфігу gpurunner: показати, зберегти, перевірити."""
    from gpurunner.auth import saturn as saturn_auth

    if url or token:
        if not (url and token):
            err_console.print("[red]--url і --token задаються разом[/red]")
            raise typer.Exit(code=2)
        path = saturn_auth.save_credentials(url=url, token=token)
        console.print(f"[green]✓ збережено[/green] {path}")

    try:
        creds = saturn_auth.discover_credentials()
    except AuthError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None

    console.print(f"Source: [cyan]{creds.source}[/cyan]")
    console.print(f"URL:    {creds.url}")

    if verify:
        try:
            info = saturn_auth.verify()
        except AuthError as e:
            err_console.print(f"[red]Verify failed: {e}[/red]")
            raise typer.Exit(code=3) from None
        console.print(
            f"[green]Verified[/green] — {info['username']} · org [bold]{info['org']}[/bold]"
        )


# ---- ls -------------------------------------------------------------------


@app.command()
def recipes(job: str = typer.Argument(..., help="Назва job (напр. yolo_spotter).")) -> None:
    """Готові конфігурації запуску (job.RECIPES) + наперед-прорахунок вартості кожної."""
    try:
        job_cls = get_job(job)
    except KeyError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None
    recs = getattr(job_cls, "RECIPES", None)
    if not recs:
        console.print(f"[yellow]{job} не має RECIPES.[/yellow]")
        return
    job_inst = job_cls()
    table = Table(title=f"{job} — рецепти запуску")
    table.add_column("preset", style="cyan")
    table.add_column("backend/gpu", style="magenta")
    table.add_column("~час/кошт", style="green")
    table.add_column("команда")
    for name, rec in recs.items():
        params = rec.get("params", {})
        gpu = rec.get("gpu", "")
        bk = rec.get("backend", "")
        cost = ""
        if bk == "modal":
            try:
                from gpurunner.backends.modal import estimate_cost
                est = estimate_cost(job_inst, {"dataset": "_estimate", **params}, gpu)
                if est:
                    cost = f"~{est['hours']:.1f}год / ${est['total']:.2f}"
            except Exception:
                pass
        elif bk == "kaggle":
            cost = "$0 (квота)"
        pstr = " ".join(f"-p {k}={v}" for k, v in params.items())
        cmd = f"gpurunner run {job} -b {bk} --gpu {gpu} {pstr}"
        table.add_row(name, f"{bk}/{gpu}", cost, cmd)
    console.print(table)


@app.command()
def ls() -> None:
    """Перелік доступних job і бекендів."""
    job_table = Table(title="Jobs")
    job_table.add_column("name", style="cyan")
    job_table.add_column("backends", style="magenta")
    job_table.add_column("description")
    for cls in list_jobs():
        job_table.add_row(
            cls.name,
            ", ".join(cls.supported_backends),
            cls.description or "",
        )
    console.print(job_table)

    bk_table = Table(title="Backends")
    bk_table.add_column("name", style="cyan")
    bk_table.add_column("gpus", style="magenta")
    bk_table.add_column("default", style="green")
    for name in BACKEND_NAMES:
        bk_cls = get_backend(name)
        bk_table.add_row(
            bk_cls.name,
            ", ".join(bk_cls.gpu_choices) or "—",
            bk_cls.default_gpu or "—",
        )
    console.print(bk_table)


@app.command()
def balance(
    backend: str = typer.Option("", "--backend", "-b", help="Лише один бекенд. Без нього — усі."),
    json_out: bool = typer.Option(False, "--json", help="Машинний вивід."),
) -> None:
    """Скільки лишилось грошей / кредитів на кожному бекенді."""
    from gpurunner.core import balances

    if backend:
        try:
            get_backend(backend)
        except KeyError as e:
            err_console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=2) from None
    names = [backend] if backend else [
        "kaggle", "lightning", "colab", "vast", "modal", "beam", "saturn",
    ]
    rows = balances.collect_reports(names)

    if json_out:
        console.print_json(json.dumps(rows, ensure_ascii=False))
        return

    table = Table(title="Баланси", show_header=True, header_style="bold")
    table.add_column("backend", style="cyan")
    table.add_column("залишок", justify="right", style="green")
    table.add_column("витрачено", justify="right")
    table.add_column("деталі")
    for r in rows:
        avail = "—" if r["available"] is None else f"{r['available']:.2f} {r['unit']}".strip()
        spent = "—" if r.get("spent") is None else f"{r['spent']:.2f} {r['unit']}".strip()
        detail = r["detail"]
        if r.get("url"):
            detail = f"{detail}\n[dim]{r['url']}[/dim]"
        table.add_row(r["backend"], avail, spent, detail)
    console.print(table)
    console.print(
        "[dim]«—» = провайдер не віддає цифру через API (GPU-квота Kaggle, Colab units) — "
        "дивись за посиланням. Modal — ОЦІНКА ($30/міс мінус витрати; "
        "GPURUNNER_MODAL_MONTHLY_CREDIT=100 для Team).[/dim]"
    )


# ---- run ------------------------------------------------------------------


def _parse_params(items: list[str]) -> dict[str, Any]:
    """Parse repeated --param k=v args. Values are JSON-decoded if possible."""
    out: dict[str, Any] = {}
    for raw in items:
        if "=" not in raw:
            raise typer.BadParameter(f"--param expects k=v, got {raw!r}")
        k, v = raw.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise typer.BadParameter(f"empty key in --param {raw!r}")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


@app.command()
def run(
    job: str = typer.Argument(..., help="Назва job. Див. `gpurunner ls`."),
    backend: str = typer.Option("kaggle", "--backend", "-b", help="Назва бекенда."),
    param: list[str] = typer.Option(
        [], "--param", "-p", help="Параметр job як k=v (повторюваний). Значення — JSON, якщо розбирається."),
    gpu: str = typer.Option("", "--gpu", "-g", help="Прискорювач. Без нього — default_gpu бекенда."),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Локальна тека під результати (для подальшого fetch)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Перевірити параметри й згенерувати код; без сабміту."),
    stage: Path | None = typer.Option(
        None, "--stage",
        help="Kaggle: ПЕРЕД сабмітом залити цю теку (dataset-metadata.json усередині) новою "
             "версією датасету і дочекатись, поки вона стане current (анти version-race)."),
    expect_log: str = typer.Option(
        "", "--expect-log",
        help="Kaggle: після сабміту полити логи кернела, доки не з'явиться цей підрядок "
             "(напр. 'Extracting train-v42.tgz') — верифікація, що кернел узяв правильні дані."),
    expect_log_timeout: int = typer.Option(
        1800, "--expect-log-timeout", help="Скільки секунд чекати на --expect-log рядок."),
    open_browser: bool = typer.Option(
        False, "--open", help="Colab: відкрити нотбук у браузері одразу після заливки."),
    allow_cost: float = typer.Option(
        0.0, "--allow-cost",
        help="Beam: дозволити цьому run'у списати до $N (стеля за замовчуванням — $1). "
             "Без цього дорогий запуск відхиляється ДО заливки даних."),
    json_out: bool = typer.Option(
        False, "--json",
        help="Машинний вивід: останній рядок stdout — JSON із handle (див. docs/contract.md)."),
) -> None:
    """Запустити job на бекенді."""
    result: dict[str, Any] = {"command": "run", "job": job, "backend": backend}
    with contract.human_to_stderr(console, json_out):
        try:
            _run_impl(job, backend, param, gpu, out, dry_run,
                      stage=stage, expect_log=expect_log,
                      expect_log_timeout=expect_log_timeout,
                      open_browser=open_browser, allow_cost=allow_cost, result=result)
        except typer.Exit as e:
            if json_out and e.exit_code:
                contract.emit({"ok": False, **result, "exit_code": e.exit_code})
            raise
    if json_out:
        contract.emit({"ok": True, **result})


def _fmt_hours(h: float) -> str:
    return f"{int(h)}год {int((h * 60) % 60)}хв" if h >= 1 else f"{int(h * 60)}хв"


def _print_spend_ceiling(backend: str, params: dict, gpu: str) -> None:
    """Beam: скільки цей run може списати В НАЙГІРШОМУ разі, і що з місячною стелею.

    Оцінка часу — це прогноз, а списання відбувається за фактом до таймауту, тож
    перед сабмітом показуємо саме стелю: timeout × ставка. Це та цифра, яку
    перевіряє guard у бекенді.
    """
    if backend != "beam":
        return
    try:
        from gpurunner.backends.beam import monthly_budget, worst_case_cost
        from gpurunner.core import budget

        worst = worst_case_cost(gpu, params)
        if worst is None:
            return
        committed = budget.month_committed("beam")
        cap = monthly_budget()
    except Exception:
        return
    console.print(
        f"[yellow]⛔ стеля витрат:[/yellow] до [bold]${worst:.2f}[/bold] за цей run · "
        f"місяць: ${committed:.2f}/${cap:.2f} "
        f"[dim](локальний облік — Beam не віддає витрати через API)[/dim]"
    )


def _print_cost_estimate(backend: str, job_inst: Any, params: dict, gpu: str) -> None:
    """Наперед-прорахунок вартості. Best-effort — мовчить, якщо backend/job не підтримує.

    Modal і Beam тарифікуються погодинно → показуємо $. Saturn на free-плані тарифів
    не віддає (`price_per_hour` = null), тому там валюта — **години** з місячної квоти.
    """
    try:
        if backend == "modal":
            from gpurunner.backends.modal import estimate_cost

            est = estimate_cost(job_inst, params, gpu)
        elif backend == "beam":
            from gpurunner.backends.beam import estimate_cost as beam_estimate

            est = beam_estimate(job_inst, params, gpu)
        elif backend == "saturn":
            from gpurunner.backends.saturn import SaturnBackend
            from gpurunner.backends.saturn import estimate_cost as saturn_estimate

            est = saturn_estimate(job_inst, params, gpu, SaturnBackend().list_sizes())
        else:
            return
    except Exception:
        return
    if not est:
        return

    hh = _fmt_hours(est["hours"])
    if backend == "saturn":
        money = f" · [bold]${est['total']:.2f}[/bold]" if est.get("total") is not None else ""
        console.print(
            f"[dim]≈ прорахунок:[/dim] [bold]{gpu}[/bold] ({est['instance_type']}) · "
            f"~{hh}{money} [dim](Saturn не публікує ні залишку, ні місячного ліміту)[/dim]"
        )
        return

    tail = "±15%" if backend == "modal" else "per-second білінг, без мінімалки"
    console.print(
        f"[dim]≈ прорахунок:[/dim] [bold]{gpu}[/bold] · ~{hh} · "
        f"[bold]${est['total']:.2f}[/bold] всього "
        f"([dim]${est['per_epoch']:.3f}/епоха, {est['cores']:.0f} cores/{est['ram_gib']:.0f}GiB RAM, {tail}[/dim])"
    )


def _run_impl(
    job: str, backend: str, param: list[str], gpu: str, out: Path | None, dry_run: bool,
    *, stage: Path | None = None, expect_log: str = "", expect_log_timeout: int = 1800,
    open_browser: bool = False, allow_cost: float = 0.0,
    result: dict[str, Any] | None = None,
) -> None:
    # `result` — те, що `run --json` віддасть споживачеві; людський вивід не міняється.
    result = result if result is not None else {}
    try:
        job_cls = get_job(job)
        backend_cls = get_backend(backend)
    except KeyError as e:
        err_console.print(f"[red]{e}[/red]")
        result["error"] = str(e)
        raise typer.Exit(code=2) from None

    if (stage or expect_log) and backend not in ("kaggle", "colab"):
        err_console.print("[red]--stage/--expect-log підтримуються лише для backend=kaggle|colab[/red]")
        raise typer.Exit(code=2)
    if stage and backend == "colab":
        err_console.print(
            "[red]для colab заливай дані через `gpurunner drive push <тека> --name <ім'я>` "
            "(у Drive немає Kaggle-version-race, тому --stage не потрібен)[/red]")
        raise typer.Exit(code=2)
    if open_browser and backend != "colab":
        err_console.print("[red]--open має сенс лише для backend=colab[/red]")
        raise typer.Exit(code=2)

    job_inst = job_cls()
    params = _parse_params(param)
    if allow_cost:
        # Backend-only knob; validate_params() drops it before the job ever sees it.
        params["max_cost"] = allow_cost

    if dry_run:
        try:
            normalized = job_inst.validate_params(params)
        except (ValueError, FileNotFoundError) as e:
            err_console.print(f"[red]Validation failed: {e}[/red]")
            result["error"] = f"validation: {e}"
            raise typer.Exit(code=2) from None
        result.update({"dry_run": True, "params": normalized})
        console.print(Panel.fit("Validated params", style="green"))
        console.print(json.dumps(normalized, ensure_ascii=False, indent=2))
        code = job_inst.render_remote_code(params)
        console.print(Panel.fit(f"Remote code preview ({len(code):,} chars)", style="green"))
        first_500 = code[:500] + ("\n…" if len(code) > 500 else "")
        console.print(first_500)
        return

    bk = backend_cls()
    if not gpu:
        gpu = bk.default_gpu or "T4"

    _print_cost_estimate(backend, job_inst, params, gpu)
    _print_spend_ceiling(backend, params, gpu)

    try:
        bk.check_auth()
        if stage is not None:
            _stage_dataset(bk, Path(stage))
        handle = bk.submit(job_inst, params, gpu=gpu)
    except (AuthError, BackendError, ValueError, FileNotFoundError) as e:
        err_console.print(f"[red]Submit failed: {e}[/red]")
        result["error"] = f"submit: {e}"
        raise typer.Exit(code=3) from None

    if out:
        handle.output_dir = str(Path(out).resolve())
    manifest.add(handle)
    result.update({"dry_run": False, **contract.handle_payload(handle)})

    console.print(
        f"[green]✓ submitted[/green]  "
        f"[bold]{handle.id[:8]}[/bold]  →  {handle.remote_id}  ({backend}, {gpu})"
    )
    if backend == "vast":
        console.print(
            f"\n[bold yellow]💸 Оренда {handle.volume_name or ''} тарифікується ПОГОДИННО[/bold yellow] "
            "і не спиняється по завершенню job'а.\n"
            f"  забрати вихід:  [dim]gpurunner fetch {handle.id[:8]} -o <тека>[/dim]\n"
            f"  ЗНИЩИТИ інстанс: [bold]gpurunner cancel {handle.id[:8]}[/bold]\n"
        )
    if backend == "colab":
        from gpurunner.backends.colab import ColabBackend

        assert isinstance(bk, ColabBackend)
        url = bk.notebook_url(handle)
        console.print(
            f"\n[bold yellow]▶ Colab не запускає нотбук сам[/bold yellow] — відкрий і тисни "
            f"[bold]Runtime → Run all[/bold]:\n  [link={url}]{url}[/link]\n"
        )
        if open_browser:
            import webbrowser

            webbrowser.open(url)
    console.print(f"Track:  [dim]gpurunner status {handle.id[:8]}[/dim]")
    console.print(f"Watch:  [dim]gpurunner watch  {handle.id[:8]}[/dim]")

    if expect_log:
        console.print(f"[dim]чекаю в логах кернела рядок «{expect_log}» (до {expect_log_timeout}s)…[/dim]")
        # wait_for_log_line є лише в KaggleBackend — --expect-log і задокументований
        # як Kaggle-only, але без звуження типу mypy бачить голий Backend
        from gpurunner.backends.kaggle import KaggleBackend

        assert isinstance(bk, KaggleBackend)
        line = bk.wait_for_log_line(handle, expect_log, timeout=expect_log_timeout)
        if line is None:
            # 🔴 «рядка нема» і «логів нема» — це РІЗНІ діагнози, а раніше обидва
            # давали ту саму червону помилку «кернел міг узяти стару версію
            # датасету». Kaggle не віддає лог кернела, поки той `running`
            # (перевірено на parseq_train `03a7ad9a` і kraken_train `50adf3c6`:
            # `logs` порожній весь час роботи), тож на довгих тренах перевірка
            # ГАРАНТОВАНО «провалюється» — марно з'їдає expect_log_timeout і
            # валить команду кодом 4, хоча запуск успішний.
            try:
                any_logs = any(True for _ in bk.logs(handle))
            except Exception:
                any_logs = False
            if not any_logs:
                console.print(
                    f"[yellow]⚠ логи кернела ще недоступні (бекенд не віддає їх "
                    f"під час роботи) — рядок «{expect_log}» не перевірено. "
                    f"Звір після завершення: gpurunner logs {handle.id[:8]}[/yellow]")
            else:
                err_console.print(
                    f"[red]⚠ логи є, але рядка «{expect_log}» у них НЕМА за "
                    f"{expect_log_timeout}s — кернел міг узяти стару версію "
                    f"датасету. Перевір: gpurunner logs {handle.id[:8]}[/red]")
                raise typer.Exit(code=4)
        else:
            # ⚠ Саме `else`, а не рядок після `if`. Жовта гілка вище нічого не
            # кидає (це успіх — просто нема чого звіряти), тож без цього
            # виконання доходило сюди з line=None і команда падала
            # `AttributeError: 'NoneType' has no attribute 'strip'` — рівно в
            # тому сценарії, заради якого гілку й додавали.
            console.print(f"[green]✓ лог підтверджено:[/green] {line.strip()}")


def _stage_dataset(bk: Any, folder: Path) -> None:
    """push теки новою версією датасету + блокуюче чекання current (анти version-race)."""
    folder = folder.resolve()
    expect = {p.name: p.stat().st_size for p in folder.iterdir()
              if p.is_file() and p.name != "dataset-metadata.json"}
    if not expect:
        raise BackendError(f"--stage: у {folder} нема файлів даних")
    ds_id = bk.dataset_push(folder, notes=f"gpurunner stage {datetime.now(tz=UTC):%Y-%m-%d %H:%M}")
    console.print(f"[green]✓ upload прийнято[/green] {ds_id}: {', '.join(sorted(expect))}")
    console.print("[dim]чекаю, поки нова версія стане current (Kaggle version-race)…[/dim]")
    bk.dataset_wait_current(ds_id, expect)
    console.print(f"[green]✓ версія current[/green] — сабмічу проти {ds_id}")


# ---- dataset staging (Kaggle) ----------------------------------------------

dataset_app = typer.Typer(
    help="Kaggle-датасети: push із блокуючим чеканням current-версії (анти version-race — "
         "«Upload successful» приходить ДО перемикання версії; кернел у цьому вікні бере СТАРУ).")
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("push")
def dataset_push_cmd(
    folder: Path = typer.Argument(
        ..., help="Тека з dataset-metadata.json (id=<owner>/<slug>) + файлами даних."),
    message: str = typer.Option("gpurunner push", "--message", "-m", help="Нотатка до версії."),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Чекати, поки версія стане current."),
    timeout: int = typer.Option(3600, "--timeout", help="Максимум секунд чекання current."),
    poll: int = typer.Option(30, "--poll", help="Інтервал полінгу listing, сек."),
) -> None:
    """Залити нову версію датасету і дочекатись, поки вона стане current."""
    from gpurunner.backends.kaggle import KaggleBackend
    bk = KaggleBackend()
    try:
        bk.check_auth()
        folder_r = folder.resolve()
        # int | None: dataset_wait_current приймає None як «розмір не звіряти»
        expect: dict[str, int | None] = {
            p.name: p.stat().st_size for p in folder_r.iterdir()
            if p.is_file() and p.name != "dataset-metadata.json"}
        if not expect:
            err_console.print(f"[red]у {folder_r} нема файлів даних[/red]")
            raise typer.Exit(code=2)
        ds_id = bk.dataset_push(folder_r, notes=message)
        console.print(f"[green]✓ upload прийнято[/green] {ds_id}: {', '.join(sorted(expect))}")
        if wait:
            console.print("[dim]чекаю current…[/dim]")
            bk.dataset_wait_current(ds_id, expect, timeout=timeout, poll=poll)
            console.print(f"[green]✓ версія current[/green] — безпечно сабмітити кернели проти {ds_id}")
        else:
            console.print("[yellow]⚠ --no-wait: НЕ сабміть кернели, доки `gpurunner dataset files` "
                          "не покаже нові файли (version-race)[/yellow]")
    except (AuthError, BackendError) as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3) from None


@dataset_app.command("files")
def dataset_files_cmd(
    dataset_id: str = typer.Argument(..., help="<owner>/<slug>"),
) -> None:
    """Файли ПОТОЧНОЇ версії датасету (name + bytes)."""
    from gpurunner.backends.kaggle import KaggleBackend
    bk = KaggleBackend()
    try:
        bk.check_auth()
        files = bk.dataset_files(dataset_id)
    except (AuthError, BackendError) as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3) from None
    t = Table(show_header=True, header_style="bold")
    t.add_column("name")
    t.add_column("bytes", justify="right")
    for f in files:
        t.add_row(str(f["name"]), str(f["size"] if f["size"] is not None else "?"))
    console.print(t)


# ---- drive staging (Colab) -------------------------------------------------

drive_app = typer.Typer(
    help="Вхідні дані для Colab у Google Drive (MyDrive/gpurunner/data/<name>/). "
         "Нотбук копіює їх у /kaggle/input/<name>/ перед стартом job'а.")
app.add_typer(drive_app, name="drive")


@drive_app.command("push")
def drive_push_cmd(
    folder: Path = typer.Argument(..., help="Тека з файлами даних (архіви заливаються як є)."),
    name: str = typer.Option(..., "--name", "-n", help="Ім'я теки у MyDrive/gpurunner/data/."),
) -> None:
    """Залити теку у Drive і звірити md5 кожного файлу (ловить обірвану заливку)."""
    from gpurunner.backends.colab import ColabBackend

    bk = ColabBackend()
    try:
        bk.check_auth()
        pushed = bk.drive_push(folder, name=name)
    except (AuthError, BackendError) as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3) from None
    for f in pushed:
        console.print(f"[green]✓[/green] {f['name']}  {f['size']:,} B  md5={f['md5'][:12]}…")
    console.print(f"[green]✓ у Drive:[/green] MyDrive/gpurunner/data/{name}")


@drive_app.command("files")
def drive_files_cmd(
    name: str = typer.Argument(..., help="Ім'я теки у MyDrive/gpurunner/data/."),
) -> None:
    """Показати вміст MyDrive/gpurunner/data/<name>/."""
    from gpurunner.backends.colab import ColabBackend

    bk = ColabBackend()
    try:
        bk.check_auth()
        files = bk.drive_files(name)
    except (AuthError, BackendError) as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3) from None
    t = Table(show_header=True, header_style="bold")
    t.add_column("name")
    t.add_column("bytes", justify="right")
    t.add_column("md5")
    for f in files:
        t.add_row(str(f["name"]), f"{f['size']:,}" if f["size"] is not None else "?",
                  str(f["md5"] or "-"))
    console.print(t)


# ---- vast marketplace ------------------------------------------------------

vast_app = typer.Typer(
    help="Vast.ai marketplace. ⚠ Оренда тарифікується ПОГОДИННО і не спиняється, "
         "поки інстанс не знищено (`gpurunner cancel <id>`).")
app.add_typer(vast_app, name="vast")


@vast_app.command("instances")
def vast_instances_cmd(
    json_out: bool = typer.Option(False, "--json", help="Машинний вивід."),
) -> None:
    """Що орендовано ПРЯМО ЗАРАЗ — те саме, що `gpurunner burn`.

    🔴 Команду шукали тут у п'яти сесіях і не знаходили: рука тягнеться в групу
    `vast`, а приймач горіння лежав на верхньому рівні. Питання «скільки в мене
    зараз орендовано» після цього щоразу вирішували новим шматком коду з httpx.
    """
    burn(json_out=json_out)


@vast_app.command("offers")
def vast_offers_cmd(
    gpu: str = typer.Option("RTX3090", "--gpu", "-g", help="Модель GPU. Див. `gpurunner ls`."),
    max_price: float = typer.Option(0.0, "--max-price", help="Стеля $/год (0 = без стелі)."),
    disk: int = typer.Option(60, "--disk", help="Мінімум диску, ГБ."),
    limit: int = typer.Option(10, "--limit", help="Скільки пропозицій показати."),
    min_cpu: float = typer.Option(0.0, "--min-cpu", help="Мінімум ЕФЕКТИВНИХ ядер."),
    min_vram: float = typer.Option(0.0, "--min-vram", help="Мінімум VRAM, ГБ."),
    min_ram: float = typer.Option(0.0, "--min-ram", help="Мінімум RAM, ГБ."),
    min_reliability: float = typer.Option(0.0, "--min-reliability", help="0..1."),
    machine: int = typer.Option(0, "--machine", help="Шукати КОНКРЕТНУ машину за machine_id."),
    score: bool = typer.Option(False, "--score", help="Порахувати шарди, години й ціну справи."),
    pages: int = typer.Option(0, "--pages", help="Скільки сторінок у справі (для --score)."),
    budget: float = typer.Option(3.0, "--budget", help="Бюджет заходу, $ (для --score)."),
    max_hours: float = typer.Option(8.0, "--max-hours", help="Стеля годин (для --score)."),
) -> None:
    """Вільні пропозиції — подивитись ціну ДО того, як орендувати.

    🔴 `--min-cpu` і `--min-vram` тут не косметика. Ядра — це швидкість
    (74% часу сторінки — геометрія kraken на процесорі), VRAM — це скільки
    шардів підняти, тобто скільки з оплачених ядер узагалі працюватиме.
    Пошук без них колись дав бокс із 4 ядрами й бокс із 16 ГБ під 8 шардів.

    `--machine` шукає конкретну машину за `machine_id` — так адресно
    знаходиться той самий добрий бокс, а не «схожий на нього».

    З `--score` показує не картку, а РІШЕННЯ: скільки шардів вийде, скільки
    годин і доларів коштуватиме справа й що про машину знає реєстр.
    """
    from gpurunner.backends.vast import VastBackend
    from gpurunner.core import boxes
    from gpurunner.core.offer_score import Need, machine_id_of, score_offer

    bk = VastBackend()
    try:
        offers = bk.search_offers(
            gpu=gpu, max_price=max_price or None, disk_gb=disk, limit=limit,
            min_cpu=min_cpu, min_ram_gb=min_ram, min_reliability=min_reliability,
            min_vram_gb=min_vram,
            machine_ids=[machine] if machine else None,
            exclude_machine_ids=None if machine else boxes.banned_ids(),
        )
    except (AuthError, BackendError, ValueError, KeyError) as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3) from None
    if not offers:
        err_console.print("[yellow]нічого не знайдено — послаб фільтри[/yellow]")
        raise typer.Exit(code=1)

    verdicts = boxes.verdicts()
    t = Table(show_header=True, header_style="bold")
    cols = ["offer", "machine", "gpu", "n", "ядра", "VRAM", "$/год", "інет ↓", "надійн.", "гео"]
    if score:
        cols += ["шардів", "стор/год", "год", "$ справа"]
    cols += ["реєстр"]
    for col in cols:
        t.add_column(col)

    need = Need(pages=pages or 1000, max_hours=max_hours, budget_usd=budget, disk_gb=disk)
    for o in offers[:limit]:
        mid = machine_id_of(o)
        verdict = verdicts.get(mid)
        row = [
            str(o.get("id")),
            str(mid or "—"),
            str(o.get("gpu_name")),
            str(o.get("num_gpus")),
            f"{float(o.get('cpu_cores_effective') or 0):.0f}",
            f"{float(o.get('gpu_ram') or 0) / 1024:.0f}",
            f"{float(o.get('dph_total') or 0):.3f}",
            f"{float(o.get('inet_down') or 0):.0f}",
            f"{float(o.get('reliability2') or o.get('reliability') or 0):.3f}",
            str(o.get("geolocation") or "—"),
        ]
        if score:
            s = score_offer(o, need, verdict)
            row += [
                str(s.sizing.shards),
                f"{s.sizing.pages_per_hour:.0f}",
                "—" if s.hours == float("inf") else f"{s.hours:.1f}",
                "—" if s.cost == float("inf") else f"{s.cost:.2f}",
            ]
        mark = {"starred": "★", "banned": "✗", "warned": "⚠", "unknown": ""}
        row.append(f"{mark.get(verdict.state, '')} {verdict.reason}" if verdict else "")
        t.add_row(*row)
    console.print(t)
    console.print(
        "[dim]Заявленому інет ↓ вірити не можна (оффер із 755 Мбіт/с віддавав 0.5) — "
        "канал міряється на самому боксі перед заливкою.[/dim]"
    )
    console.print(
        "[dim]Прогін справи: gpurunner htr supervise --plan plan.json ; "
        "знищення оренди: gpurunner cancel <id>[/dim]"
    )


# ---- reconcile ------------------------------------------------------------


@app.command()
def reconcile(
    any_owner: bool = typer.Option(False, "--any-owner", help="Чіпати й чужі записи."),
    apply: bool = typer.Option(False, "--apply", help="Справді записати зміни."),
    dry_run: bool = typer.Option(  # синонім поведінки за замовчуванням
        False, "--dry-run",
        help="Явний синонім поведінки за замовчуванням (пише лише --apply).",
    ),
) -> None:
    """Звірити нетермінальні хендли з реальністю й прибрати сміття.

    🔴 У спільному реєстрі накопичуються десятки «живих» прогонів від давніх
    заходів (на 2026-08-11 їх було 52, з них 13 на Vast). Через них будь-яка
    масова операція завжди має справу зі сміттям, а `--all-running` виглядає
    страшніше, ніж є. Тут ми питаємо бекенд, що з них насправді живе.
    """
    from gpurunner.core.models import JobStatus

    me = manifest.current_owner()
    live = [h for h in manifest.load()
            if h.status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.UNKNOWN)]
    mine = [h for h in live if any_owner or (h.owner or "") == me]
    if not mine:
        console.print("[dim]нетермінальних прогонів немає[/dim]")
        return

    stale: list = []
    for h in mine:
        if h.backend != "vast":
            continue
        try:
            from gpurunner.backends.vast import VastBackend

            bk = get_backend(h.backend)()
            assert isinstance(bk, VastBackend)  # вище відсіяно все, крім vast
            if bk._instance(h) is None:
                stale.append(h)
        except (AuthError, BackendError) as e:
            err_console.print(f"[yellow]{h.id[:8]}: не опитати ({e})[/yellow]")

    console.print(f"нетермінальних: {len(mine)} · зниклих на боці Vast: {len(stale)}")
    # 🔴🔴 Ці два числа НЕ є приймачем горіння, хоч і читаються як він.
    # `reconcile` перелічує лише ЗНИКЛІ, а живих не показує зовсім — тож
    # «нетермінальних 1 · зниклих 0» означає «один ГОРИТЬ» (30.08.2026: два
    # осиротілі бокси горіли по $0.228/год, поки не спитали Vast прямо).
    _print_live_instances()
    for h in stale:
        case = (h.params or {}).get("case") or ""
        console.print(f"  [dim]{h.id[:8]}[/dim] інстанс {h.remote_id}"
                      + (f" · {case}" if case else "")
                      + (f" · {h.owner}" if h.owner else ""))
    if not apply:
        console.print("[dim]сухий прогін; --apply щоб позначити їх cancelled[/dim]")
        return
    for h in stale:
        h.status = JobStatus.CANCELLED
        h.error = "reconcile: інстансу немає на боці Vast"
        manifest.update(h)
    console.print(f"[green]позначено cancelled: {len(stale)}[/green]")


def _print_live_instances() -> int:
    """Що ГОРИТЬ на боці Vast просто зараз. Повертає кількість.

    Єдиний приймач, який не бреше: прямий список інстансів. Стан наглядача,
    реєстр прогонів і `reconcile` кажуть про наше уявлення, а гроші тарифікує
    Vast за своїм.
    """
    from gpurunner.backends.vast import VastBackend

    try:
        data = VastBackend()._request(
            "GET", "/instances/", params={"owner": "me"}, api_version="v1")
    except (AuthError, BackendError) as e:
        err_console.print(f"[yellow]список інстансів не опитати: {e}[/yellow]")
        return -1
    rows = [i for i in (data.get("instances") or [])
            if str(i.get("actual_status") or "").lower() in ("running", "loading", "created")]
    if not rows:
        console.print("[green]активних інстансів на Vast: 0[/green]")
        return 0
    burn = sum(float(i.get("dph_total") or 0) for i in rows)
    console.print(
        f"[red bold]активних інстансів на Vast: {len(rows)} · "
        f"${burn:.3f}/год ГОРИТЬ ПРОСТО ЗАРАЗ[/red bold]"
    )
    for i in rows:
        console.print(
            f"  [red]{i.get('id')}[/red] {i.get('gpu_name')}×{i.get('num_gpus') or 1} · "
            f"${float(i.get('dph_total') or 0):.3f}/год · {i.get('actual_status')} · "
            f"{i.get('geolocation') or '?'}"
        )
    console.print("[dim]погасити: gpurunner cancel <id> (гроші спиняє лише знищення)[/dim]")
    return len(rows)


@app.command()
def burn(
    json_out: bool = typer.Option(False, "--json", help="Машинний вивід."),
) -> None:
    """Що зараз ГОРИТЬ на Vast — прямим запитом, повз наш реєстр.

    🔴 Питання «скільки в мене зараз орендовано» ставилось у 36 сесіях, і
    щоразу на нього відповідали новим шматком коду з `httpx`, бо штатної
    команди не було. Реєстр прогонів на це не відповідає: він знає лише те,
    що заводили ми, а бокс міг лишитись від убитого наглядача.
    """
    if not json_out:
        _print_live_instances()
        return
    from gpurunner.backends.vast import VastBackend

    try:
        data = VastBackend()._request(
            "GET", "/instances/", params={"owner": "me"}, api_version="v1")
    except (AuthError, BackendError) as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None
    rows = [i for i in (data.get("instances") or [])
            if str(i.get("actual_status") or "").lower() in ("running", "loading", "created")]
    console.print_json(data=json.dumps({
        "active": len(rows),
        "burn_usd_per_hour": round(sum(float(i.get("dph_total") or 0) for i in rows), 4),
        "instances": [
            {"id": i.get("id"), "gpu": i.get("gpu_name"),
             "num_gpus": i.get("num_gpus"), "dph": i.get("dph_total"),
             "status": i.get("actual_status"), "geo": i.get("geolocation")}
            for i in rows
        ],
    }, ensure_ascii=False))


# ---- htr supervisor --------------------------------------------------------

htr_app = typer.Typer(
    help="Наглядач хмарних HTR-прогонів: орендує, стежить, доганяє, звіряє, гасить.")
app.add_typer(htr_app, name="htr")

# ---- фонові задачі ----------------------------------------------------------

bg_app = typer.Typer(
    help="Довгі задачі у фоні: без вікна, ОДНА копія, зупинка лише за PID.")
app.add_typer(bg_app, name="bg")

_BG_CTX = {"allow_extra_args": True, "ignore_unknown_options": True}


@bg_app.command("start", context_settings=_BG_CTX)
def bg_start_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Ім'я задачі — одна копія на ім'я."),
    cwd: Path = typer.Option(None, "--cwd", help="Робоча тека команди."),
) -> None:
    """Поставити команду після `--` у фон (напр. чергу завантаження).

    Замість ручного `schtasks /create`: без вікна, друга копія неможлива, а
    зупинка — `gpurunner bg stop <ім'я>` за PID, не за текстом команди.
    """
    from gpurunner.supervise import bg

    try:
        how = bg.start(name, list(ctx.args), cwd=cwd)
    except (bg.BgBusy, ValueError) as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None
    console.print(f"[green]{name}: {how}[/green]")
    console.print(f"[dim]стан:    gpurunner bg status {name}\nлог:     {bg.log_path(name)}"
                  f"\nспинити: gpurunner bg stop {name}[/dim]")


@bg_app.command("_run", hidden=True, context_settings=_BG_CTX)
def bg_run_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(...),
    cwd: Path = typer.Option(None, "--cwd"),
) -> None:
    """Тіло фонової задачі (кличе планувальник): замок, дитина, PID-и в стан."""
    from gpurunner.supervise import bg

    raise typer.Exit(code=bg.run(name, list(ctx.args), cwd=cwd))


@bg_app.command("status")
def bg_status_cmd(
    name: str = typer.Argument(None, help="Ім'я задачі (без нього — усі)."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Що біжить у фоні: ▶ біжить · ✓ скінчилась rc=0 · ✗ інакше."""
    from gpurunner.supervise import bg

    rows = bg.status(name)
    if json_out:
        console.print_json(data=json.dumps(rows, ensure_ascii=False))
        return
    if not rows:
        console.print("[dim]фонових задач немає[/dim]")
        return
    for r in rows:
        mark = "▶" if r["running"] else ("✓" if r.get("rc") == 0 else "✗")
        console.print(f"{mark} {r['name']} · rc={r.get('rc')} · "
                      f"старт {r.get('started') or r.get('queued')} · pid {r.get('pid_runner')} · "
                      f"{' '.join(r.get('cmd') or [])[:80]}")


@bg_app.command("stop")
def bg_stop_cmd(name: str = typer.Argument(..., help="Ім'я задачі.")) -> None:
    """Зупинити задачу деревом ЗА PID зі стану — ніколи за текстом команди."""
    from gpurunner.supervise import bg

    for line in bg.stop(name):
        console.print(line)


@htr_app.command("supervise")
def htr_supervise_cmd(
    plan_path: str = typer.Option(..., "--plan", help="JSON-план заходу."),
    budget: float = typer.Option(0.0, "--budget", help="Перекрити бюджет плану, $."),
    max_hours: float = typer.Option(0.0, "--max-hours", help="Перекрити стелю годин."),
    max_attempts: int = typer.Option(0, "--max-attempts", help="Скільки боксів пробувати."),
    max_rents: int = typer.Option(0, "--max-rents", help="Скільки ОРЕНД на захід."),
    gb_per_shard: float = typer.Option(
        0.0, "--gb-per-shard", help="VRAM на шард, ГБ (те саме, що `vram_gb_per_shard`)."),
    shards: int = typer.Option(0, "--shards", help="Явне число шардів (0 = із заліза)."),
    prefer_cores: float = typer.Option(
        0.0, "--prefer-cores", hidden=True, help="Не діє з 23.09.2026 — див. --target-pph."),
    target_pph: float = typer.Option(
        -1.0, "--target-pph",
        help="Перекрити ціль темпу плану, стор/год (0 — без цілі; -1 — як у плані)."),
    max_usd_per_1000: float = typer.Option(
        0.0, "--max-usd-per-1000", help="Стеля вартості тисячі сторінок, $."),
    session: str = typer.Option("", "--session", help="Ім'я сесії (для стану)."),
    tick: int = typer.Option(30, "--tick", help="Період опитування боксу, с."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Порахувати ринок БЕЗ оренди."),
    detach: bool = typer.Option(
        False, "--detach",
        help="Пустити ВІДЧЕПЛЕНО, щоб пережив сесію (Windows: планувальник)."),
    json_out: bool = typer.Option(
        False, "--json",
        help="Лише з --dry-run: кошторис одним JSON (для обгорток, що вирішують самі)."),
) -> None:
    """Прогнати план у хмарі без участі людини.

    Усередині бюджету й строку наглядач сам орендує, переорендовує, доганяє
    пропущені сторінки і гасить оренду. Людину він смикає рівно у трьох
    випадках: скінчились гроші, ринок порожній, справа лишилась неповною
    після всіх догонів.

    Стан — `gpurunner htr state --json`. У лог дивитись не треба.
    """
    from dataclasses import replace

    from gpurunner.supervise.htr import Supervisor
    from gpurunner.supervise.plan import load_plan

    try:
        plan = load_plan(plan_path)
    except (OSError, ValueError) as e:
        err_console.print(f"[red]план не годиться: {e}[/red]")
        raise typer.Exit(code=2) from None

    # 🔴 Реєстр боксів — дані користувача, і адресу його дає змінна оточення.
    # Процес, який її не бачить (інший термінал, планувальник задач), мовчки
    # заводить ПОРОЖНІЙ реєстр поруч — і наглядач орендує вже забанені машини.
    # Порожній реєстр на першому заході законний, тому це попередження, а не
    # відмова; у stderr, щоб не зачепити `--json`.
    from gpurunner.core import boxes as boxes_mod

    if not boxes_mod.journal_path().is_file():
        err_console.print(
            f"[yellow]⚠ реєстр боксів порожній: {boxes_mod.journal_path()}[/yellow]\n"
            "[dim]  перша оренда з цієї машини — так і має бути; якщо ні, задайте "
            "GPURUNNER_REPO_DATA_DIR на теку зі своїм boxes.jsonl[/dim]")

    if detach:
        # 🔴 Відчеплення робиться ПЕРЕД будь-якою роботою: далі процес має жити
        # своїм життям, а цей — завершитись, віддавши сесію.
        from gpurunner.supervise import detach as detach_mod

        if not detach_mod.supported():
            err_console.print("[red]--detach тут не підтримується[/red]")
            raise typer.Exit(code=2)
        run_session = session or f"htr-{datetime.now(tz=UTC):%m%d-%H%M%S}"
        argv = [a for a in sys.argv[1:] if a != "--detach"]
        if "--session" not in argv:
            argv += ["--session", run_session]
        # Старий лог старту цієї сесії дав би хибну тривогу ще до старту нового.
        with contextlib.suppress(OSError):
            detach_mod._spawn_log_path(run_session).unlink(missing_ok=True)
        spawned_at = time.time()
        try:
            how = detach_mod.spawn(argv, session=run_session,
                                   owner=manifest.current_owner())
        except (OSError, subprocess.CalledProcessError) as e:
            err_console.print(f"[red]не вдалось відчепити: {e}[/red]")
            raise typer.Exit(code=3) from None
        # 🔴🔴 «Пішов у фон» — лише коли наглядач СПРАВДІ записав свій стан. Доти
        # рапорт ішов одразу після `schtasks /run`: 10.09.2026 задача кликала
        # `python.exe htr supervise …` і падала за секунду, а викликач (обгортка
        # над `gpurunner htr …`) звітував успіх — стану не було, оренди теж.
        if not _wait_supervisor_started(run_session, since=spawned_at):
            tail = detach_mod.spawn_log(run_session)[-1200:]
            err_console.print(
                f"[red]наглядач НЕ стартував: за {DETACH_START_WAIT_SEC:.0f} с сесія "
                f"{run_session} не записала стану[/red]")
            if tail:
                err_console.print(f"[dim]{tail}[/dim]")
            detach_mod.cleanup(run_session)
            raise typer.Exit(code=3)
        console.print(f"[green]наглядач пішов у фон: {how}[/green]")
        console.print(f"[dim]стежити:  gpurunner htr state --session {run_session}"
                      f"\nспинити:  gpurunner htr stop --session {run_session}[/dim]")
        return

    if budget:
        plan = replace(plan, budget_usd=budget)
    if max_hours:
        plan = replace(plan, max_hours=max_hours)
    if max_attempts:
        plan = replace(plan, max_attempts=max_attempts)
    # 🔴 Ручки прапорцем, а не правкою JSON. Правка файлу плану на ЖИВИЙ захід
    # не діє взагалі (наглядач тримає план у пам'яті), а на новий діяла лише
    # тоді, коли ключ покладено на потрібний рівень — чого якраз і не ставалось.
    if max_rents:
        plan = replace(plan, max_rents=max_rents)
    if gb_per_shard:
        plan = replace(plan, gb_per_shard=gb_per_shard)
    if shards:
        plan = replace(plan, shards=shards)
    if target_pph >= 0:
        plan = replace(plan, target_pph=target_pph)
    if max_usd_per_1000:
        plan = replace(plan, max_usd_per_1000_pages=max_usd_per_1000)
    for warn in plan.warnings:
        err_console.print(f"[yellow]⚠ план: {warn}[/yellow]")

    if dry_run:
        _htr_dry_run(plan, json_out=json_out)
        return

    console.print(
        f"[bold]{len(plan.cases)} справ, {plan.total_pages} сторінок · "
        f"бюджет ${plan.budget_usd:.2f} · стеля {plan.max_hours:.1f} год[/bold]"
    )
    sup = Supervisor(plan, session=session or None, tick_sec=tick)
    code = sup.run()
    st = sup.state
    console.print(f"[bold]{st.verdict}[/bold]: {st.why}")
    if st.human_action_required:
        console.print(f"[yellow]потрібне рішення людини: {st.human_action}[/yellow]")
    console.print(f"[dim]стан: gpurunner htr state --session {st.session} --json[/dim]")
    raise typer.Exit(code=code)


#: Скільки чекати, доки відчеплений наглядач запише свій стан, і як часто дивитись.
DETACH_START_WAIT_SEC = 120.0
DETACH_POLL_SEC = 2.0
#: Сліди в логу старту, після яких чекати далі немає сенсу.
_SPAWN_DEATH_MARKS = ("Traceback", "can't open file", "No module named",
                      "is not recognized", "не є внутрішньою")


def _wait_supervisor_started(session: str, *, since: float) -> bool:
    """Чи відчеплений наглядач сесії записав стан після `since` (unix-час)."""
    from gpurunner.supervise import detach as detach_mod
    from gpurunner.supervise.state import state_path

    path = state_path(session)
    deadline = time.monotonic() + DETACH_START_WAIT_SEC
    while True:
        try:
            if path.stat().st_mtime >= since - 1:
                return True
        except OSError:
            pass
        log = detach_mod.spawn_log(session)
        if any(mark in log for mark in _SPAWN_DEATH_MARKS):
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(DETACH_POLL_SEC)


#: Підпис до `Sizing.limited_by` у таблиці ринку. 🔴 Ключі мусять збігатися з
#: тим, що повертає `plan_sizing`: доти тут стояв неіснуючий `max_shards`, і
#: стеля шардів друкувалась сирим `cap`.
_SQUEEZE = {"cpu": "ядра", "vram": "шарди", "cap": "стеля"}


def _htr_dry_run_json(bk: Any, plan: Any, credit: float | None) -> None:
    """Той самий кошторис одним JSON-рядком: обгортка вирішує за ним сама.

    Розбирати таблицю рядками означало б, що кожна зміна колонки мовчки ламає
    автозапуск. Друкується лише JSON — жодних рядків rich поруч.
    """
    from gpurunner.supervise.htr import need_from_plan

    total = plan.total_pages
    need = need_from_plan(plan, pages=total)
    selection = bk.find_candidates(
        gpu=plan.gpu, need=need, num_gpus=plan.num_gpus, max_price=plan.max_price
    )
    payload: dict[str, Any] = {
        "credit": credit,
        "pages": total,
        "lines_per_page": need.lines_per_page,
        "gb_per_shard": need.gb_per_shard,
        "empty": bool(selection.empty),
        "reason": selection.reason or "",
        "candidates": 0 if selection.empty else len(selection.candidates),
    }
    if not selection.empty:
        best = selection.best
        payload["best"] = {
            "offer": best.offer.get("id"),
            "machine": best.machine_id,
            "gpu": str(best.offer.get("gpu_name") or ""),
            "num_gpus": int(best.offer.get("num_gpus") or 1),
            "dph": float(best.offer.get("dph_total") or 0),
            "shards": best.sizing.shards,
            "pages_per_hour": round(best.sizing.pages_per_hour),
            "hours": round(best.hours, 3),
            "cost": round(best.cost, 4),
            "usd_per_1000": round(best.usd_per_1000, 4),
            "pph_sure": round(best.pph_sure),
            "target_pph": need.target_pph,
        }
    print(json.dumps(payload, ensure_ascii=True), flush=True)
    if selection.empty:
        raise typer.Exit(code=6)


def _htr_dry_run(plan: Any, *, json_out: bool = False) -> None:
    """Ранжування ринку без оренди: безкоштовний `POST /bundles`."""
    from gpurunner.backends.vast import VastBackend

    # 🔴 Той самий будівник, що й у наглядача. Свій окремий `Need` тут показував
    # 8 шардів там, де прогін брав 4 (`-p vram_gb_per_shard` не доходив), і не
    # знав про стелю ціни взагалі — тобто безкоштовна перевірка ринку
    # відповідала не на те питання, заради якого її роблять.
    from gpurunner.supervise.htr import need_from_plan

    bk = VastBackend()
    # 🔴 Баланс — ПЕРШИМ рядком. Сухий прогін інстансу не створює й на гроші не
    # дивиться, тож при $0.00 він щоразу обіцяв десять кандидатів і 3360 стор/год,
    # тоді як жодна оренда не стартувала б (04.09.2026).
    try:
        credit = bk.balance().available
    except Exception:
        credit = None
    if json_out:
        _htr_dry_run_json(bk, plan, credit)
        return
    if credit is None:
        console.print("[dim]баланс Vast: невідомо[/dim]")
    elif credit < 0.5:
        console.print(
            f"[red]баланс Vast ${credit:.2f} — оренди НЕ БУДЕ, хоч би що показала "
            f"таблиця нижче: Vast відмовляє на створенні інстансу[/red]"
        )
    else:
        console.print(f"[dim]баланс Vast: ${credit:.2f}[/dim]")
    total = plan.total_pages
    need = need_from_plan(plan, pages=total)
    selection = bk.find_candidates(
        gpu=plan.gpu, need=need, num_gpus=plan.num_gpus, max_price=plan.max_price
    )
    if selection.empty:
        err_console.print(f"[yellow]{selection.reason}[/yellow]")
        for rejected in selection.rejected[:5]:
            err_console.print(f"  [dim]{rejected.explain}[/dim]")
        raise typer.Exit(code=6)

    t = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    columns: tuple[tuple[str, Literal["left", "right"]], ...] = (
        ("offer/machine", "left"), ("карта", "left"),
        ("ядра", "right"), ("$/год", "right"),
        ("шард", "right"), ("стор/год", "right"), ("на шард", "right"),
        ("що тисне", "left"), ("хв", "right"),
        ("$", "right"), ("$/1000", "right"), ("рег", "left"))
    for col, just in columns:
        t.add_column(col, justify=just, no_wrap=True)
    for c in selection.candidates[:10]:
        mark = {"starred": "★", "banned": "✗", "warned": "⚠"}
        # Що ТИСНЕ НА ШВИДКІСТЬ. Базово це `limited_by` — те, що обрізало
        # ЧИСЛО шардів. Поверх нього дві стелі самого темпу, і порядок тут
        # такий: спершу ядра, потім карта.
        # 🔴 Ядра повернулись у цю колонку 21.09.2026, але інакше, ніж їх
        # звідси прибрали: дуель 05.09 спростувала ЗРОСТАЮЧУ стелю (15 і 92
        # ядра дали 921 і 857 стор/год), а тут спадна — бокс, де шардів
        # підняли більше, ніж є кому годувати. Без цього рядка машина з 20
        # шардами на 20 ядрах показувала «шарди», хоч тисне саме CPU.
        squeeze = _SQUEEZE.get(c.sizing.limited_by, c.sizing.limited_by)
        if c.sizing.cpu_capped:
            squeeze = "ядра"
        if c.sizing.card_capped:
            squeeze = "карта"
        t.add_row(
            f"{c.offer.get('id')}/{c.machine_id}",
            str(c.offer.get("gpu_name"))[:12],
            f"{float(c.offer.get('cpu_cores_effective') or 0):.0f}",
            f"{float(c.offer.get('dph_total') or 0):.3f}",
            str(c.sizing.shards),
            f"{c.sizing.pages_per_hour:.0f}",
            # 🔴 Звідки береться швидкість. Замір 2026-08-12 на чотирьох
            # одночасних заходах: 257 стор/год на шард на Q RTX 6000 і 268 на
            # GTX 1080 — дві карти різного класу дають те саме, бо вузьке місце
            # не GPU, а CPU-геометрія kraken. Карта вирішує лише, скільки
            # шардів улізе за пам'яттю. Без цих двох колонок число «стор/год»
            # виглядає властивістю карти, хоч воно властивість флоту.
            f"{c.sizing.pages_per_hour / max(1, c.sizing.shards):.0f}",
            squeeze,
            f"{c.hours * 60:.0f}",
            f"{c.cost:.2f}",
            f"{c.usd_per_1000:.3f}",
            # Лише позначка: повний текст вердикту з'їдав ширину, і таблиця
            # починала різати числа, заради яких вона й існує.
            # Подробиці — `gpurunner boxes explain <machine>`.
            (mark.get(c.verdict.state, "") if c.verdict else ""),
        )
    console.print(t)
    best = selection.best
    assert best is not None  # порожній вибір вийшов вище кодом 6
    console.print(
        f"[bold]обрав би[/bold] offer {best.offer.get('id')} (machine {best.machine_id}), "
        f"гарантовано {best.pph_sure:.0f} стор/год, ${best.cost:.2f}, {best.hours * 60:.0f} хв "
        f"на {total} сторінок"
    )
    if selection.reason:
        console.print(f"[yellow]{selection.reason}[/yellow]")
    console.print("[dim]оренди не було — це сухий прогін[/dim]")


@htr_app.command("plan")
def htr_plan_cmd(
    case: list[str] = typer.Option(..., "--case", help="Тека справи з кадрами."),
    out_root: Path = typer.Option(
        ..., "--out-root",
        help="Куди КОЖНА справа розкладеться. Абсолютний шлях у простір ДОСЛІДЖЕННЯ."),
    model: str = typer.Option(..., "--model", help="Бойова модель (напр. pysar_cyr_v17.pt)."),
    out: Path = typer.Option(..., "--out", help="Куди записати план."),
    voices: str = typer.Option("", "--voices", help="Другий голос (напр. diak_cyr_v4.mlmodel)."),
    case_key: list[str] = typer.Option(
        [], "--case-key", help="Шифра справи — по одній на кожен --case, у тому ж порядку."),
    name: list[str] = typer.Option(
        [], "--name",
        help="Ім'я прогону — по одному на кожен --case (порожнє = з теки кадрів). "
             "Перечитування іншою моделлю: `<справа>-skryba_v6`, інакше тексти "
             "лягли б у теку першої моделі."),
    seed_seg: list[str] = typer.Option(
        [], "--seed-seg",
        help="Тека з *.seg.json.gz попереднього прогону — по одній на кожен --case "
             "(порожнє = без засіву). Кеш їде першим чекпоінтом, і бокс не "
             "сегментує справу вдруге."),
    assets: Path = typer.Option(None, "--assets", help="Локальний архів моделей."),
    assets_key: str = typer.Option("", "--assets-key", help="Ключ уже залитого архіву."),
    keep_warm_min: float = typer.Option(
        0.0, "--keep-warm-min",
        help="Хвилин тримати бокс теплим після черги, приймаючи `htr append` "
             "(0 = гасити одразу). Простій на 3090 — $0.0025/хв, холодний старт — "
             "~5 хв оренди плюс ринок."),
    expect_script: list[str] = typer.Option(
        [], "--expect-script",
        help="ім'я=шлях: скрипт у архіві мусить побайтно збігатися з локальним "
             "файлом (напр. htr_case_run.py=<шлях до раннера nyshporka>/runner.py). "
             "Розбіжність — відмова ДО оренди."),
    prefix: str = typer.Option("cases", "--prefix", help="Тека в бакеті під кадри."),
    bucket: str = typer.Option("", "--bucket", help="Бакет R2."),
    budget: float = typer.Option(3.0, "--budget", help="Стеля витрат заходу, $."),
    max_hours: float = typer.Option(
        8.0, "--max-hours",
        help="Стеля ДОПУСТИМОГО часу (не очікуваного). Ставити щедро: 8-12."),
    max_price: float = typer.Option(0.365, "--max-price", help="Погодинна стеля, $/год."),
    target_pph: float = typer.Option(
        5000.0, "--target-pph",
        help="Ціль темпу, стор/год: машина, що обережно її не дає, НЕ береться ніколи. "
             "0 — без цілі (старий режим)."),
    max_wait_min: float = typer.Option(
        60.0, "--max-wait-min",
        help="Скільки хвилин чекати ринку, коли жодна машина не дає цілі."),
    prefer_cores: float = typer.Option(0.0, "--prefer-cores", hidden=True,
                                       help="Не діє з 23.09.2026 — див. --target-pph."),
    wait_min: float = typer.Option(0.0, "--wait-min", hidden=True,
                                   help="Не діє з 23.09.2026 — див. --max-wait-min."),
    disk: int = typer.Option(40, "--disk", help="Скільки ГБ диску просити."),
    gpu: str = typer.Option("any", "--gpu"),
    max_usd_per_1000: float = typer.Option(
        0.0, "--max-usd-per-1000", help="Стеля вартості тисячі сторінок, $."),
    gb_per_shard: float = typer.Option(0.0, "--gb-per-shard", help="VRAM на шард, ГБ."),
    url_hours: float = typer.Option(24.0, "--hours", help="Строк дії посилань."),
    skip_upload: bool = typer.Option(
        False, "--skip-upload",
        help="Не питати бакет — вважати, що кадри там. Зазвичай НЕ треба: "
             "складач сам питає про кожен архів і заливає лише те, чого "
             "немає або що змінилось."),
    transport: str = typer.Option(
        "auto", "--transport",
        help="Чим доставляти дані: `r2` — бакет S3 з presigned-посиланнями; "
             "`box` — склад на самій машині, куди файли кладе scp (бакет не "
             "потрібен зовсім, але чекпоінти лежать на машині, і наглядач "
             "забирає їх додому сам); `auto` — бакет, якщо він налаштований."),
    param: list[str] = typer.Option([], "-p", "--param", help="Параметр job, key=value."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Порахувати, нічого не заливаючи."),
) -> None:
    """Скласти план заходу: спакувати кадри, залити в R2, нарізати посилання.

    🔴 `--out-root` обов'язковий і мусить вказувати в простір ДОСЛІДЖЕННЯ, а не
    туди, звідки запускають. Саме тому він тут ручний: доти складач рахував
    теку від власного кореня, і результат прогонів з інших просторів п'ять
    разів за кампанію ліг у чужий проєкт — двічі це виглядало як «робота
    втрачена» при 1101 готовій сторінці на диску.
    """
    from gpurunner.htr.plan_build import (
        BuildOptions,
        assert_unique_slugs,
        build_plan,
        frames_of,
        measure_frames,
        run_slug,
        sha256_of,
    )
    from gpurunner.htr.r2 import R2Error

    case_dirs = [Path(c) for c in case]
    names = list(name)
    for label, values in (("--name", names), ("--seed-seg", list(seed_seg))):
        if values and len(values) != len(case_dirs):
            # Порядок — єдине, що зв'язує значення зі справою: пропуск зсунув би решту.
            err_console.print(f"[red]{label} задано {len(values)} на {len(case_dirs)} "
                              f"справ — треба по одному на кожен --case[/red]")
            raise typer.Exit(code=2)
    # 🔴 Колізію імен ловимо і в СУХОМУ прогоні. Її роблять саме для того, щоб
    # побачити біду до оренди, а дві справи під одним іменем зливають декоди в
    # одну теку без жодної помилки.
    try:
        assert_unique_slugs(case_dirs, names=names or None)
    except ValueError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None
    if dry_run:
        for i, case_dir in enumerate(case_dirs):
            slug = run_slug(case_dir, names[i] if names else "")
            if not case_dir.is_dir():
                err_console.print(f"[red]немає теки: {case_dir}[/red]")
                raise typer.Exit(code=2)
            frames = frames_of(case_dir)
            if not frames:
                # 🔴 Нуль кадрів — це відмова, а не рядок у звіті. Сухий прогін
                # роблять саме для того, щоб побачити проблему до оренди; мовчазне
                # «0 кадрів · 0 МБ» читається як успішна перевірка. Найчастіша
                # причина — кадри лежать у підтеці `pages/`, а рахунок тут
                # НЕ рекурсивний, бо таким його бачить раннер.
                nested = [d for d in case_dir.iterdir() if d.is_dir()
                          and any(d.glob("*.jpg"))]
                hint = (f" Кадри є в підтеці: {nested[0]}" if nested else "")
                err_console.print(
                    f"[red]{case_dir}: немає кадрів (.jpg/.jpeg/.png) просто в "
                    f"теці.{hint}[/red]")
                raise typer.Exit(code=2)
            geometry = measure_frames(frames)
            size_mb = sum(f.stat().st_size for f in frames) / 1e6
            console.print(
                f"{slug}: {len(frames)} кадрів · {size_mb:.0f} МБ · "
                f"медіана {geometry.mpx_median:.1f} Мпікс · "
                f"{'РОЗВОРОТ' if geometry.spread else 'сторінка'} "
                f"(aspect {geometry.aspect_median:.2f})"
            )
            console.print(f"  → {(out_root / slug).resolve()}")
        console.print("[dim]сухий прогін — у R2 нічого не залито[/dim]")
        return

    options = BuildOptions(
        out_root=out_root.resolve(), model=model, voices=voices, prefix=prefix,
        bucket=bucket, budget_usd=budget, max_hours=max_hours, max_price=max_price,
        prefer_cores=prefer_cores, wait_min=wait_min, disk_gb=disk, gpu=gpu,
        target_pph=target_pph, max_wait_min=max_wait_min,
        max_usd_per_1000=max_usd_per_1000 or None, gb_per_shard=gb_per_shard,
        url_hours=url_hours, skip_upload=skip_upload, keep_warm_min=keep_warm_min,
        names=names, seed_seg=list(seed_seg), transport=transport,
        params=dict(item.split("=", 1) for item in param if "=" in item),
        expect_scripts={name: sha256_of(Path(local))
                        for name, local in (item.split("=", 1) for item in expect_script
                                            if "=" in item)},
    )
    try:
        build_plan(case_dirs, options, case_keys=list(case_key) or None,
                   assets=assets, assets_key=assets_key, out_path=out)
    except (R2Error, ValueError) as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None
    console.print(f"[green]план: {out}[/green]")


@htr_app.command("preflight")
def htr_preflight_cmd(
    plan_path: Path = typer.Argument(..., help="JSON-план заходу."),
    json_out: bool = typer.Option(False, "--json"),
    ckpt_probe: int = typer.Option(8, "--ckpt-probe", help="Скільки resume перевірити."),
) -> None:
    """Чи живі посилання плану й чи є в R2 чекпоінти — ПЕРЕД орендою."""
    from gpurunner.htr.preflight import check_plan

    report = check_plan(plan_path, ckpt_probe=ckpt_probe)
    if json_out:
        console.print_json(data=json.dumps(report, ensure_ascii=False))
        raise typer.Exit(code=0 if report["ok"] else 1)

    assets = report["assets"]
    console.print(f"assets: {'✅' if assets['ok'] else '❌'} HTTP {assets['http'] or '—'} · "
                  f"{assets['mbps']:.0f} Мбіт/с"
                  + (f" — {assets['why']}" if assets["why"] else ""))
    for entry in report["cases"]:
        pages = entry["pages_url"]
        ckpt = (f"чекпоінти: {entry['ckpt_found']}/{entry['ckpt_checked']} перевірених є"
                if entry["resume_urls"] else "чекпоінтів НЕМАЄ")
        console.print(
            f"{'✅' if pages['ok'] else '❌'} {entry['case']}: {entry['n_pages']} кадрів · "
            f"HTTP {pages['http'] or '—'} · {pages['mbps']:.0f} Мбіт/с · {ckpt}"
            + (f" — {pages['why']}" if pages["why"] else ""))
        console.print(f"   [dim]→ {entry['out_dir'] or '(немає out_dir!)'}[/dim]")
    if report["problems"]:
        console.print("\n[red]🔴 ЩО НЕ ТАК (оренду не починати):[/red]")
        for line in report["problems"]:
            console.print(f"  · {line}")
        console.print("\n[dim]Найчастіше: план старший за строк дії посилань → "
                      "перегенерувати `gpurunner htr plan` з тими самими --case "
                      "(і --skip-upload, якщо кадри вже в R2).[/dim]")
        raise typer.Exit(code=1)
    for line in report["notes"]:
        console.print(f"[dim]ℹ {line}[/dim]")
    console.print("\n[green]✅ план придатний[/green]")


@htr_app.command("bill")
def htr_bill_cmd(
    days: float = typer.Option(3.0, "--days", help="сесії, завершені за стільки днів"),
    dry_run: bool = typer.Option(False, "--dry-run", help="лише показати"),
) -> None:
    """Звірити завершені сесії з рахунком Vast (трафік, диск — те, чого не бачить «ціна × час»)."""
    from gpurunner.backends.vast import VastBackend
    from gpurunner.core import budget as budget_mod
    from gpurunner.core import manifest
    from gpurunner.htr import recover as rec
    from gpurunner.supervise import state as state_mod

    charges = VastBackend().instance_charges(time.time() - (days + 1) * 86400, time.time())
    console.print(f"рахунок Vast за {days:g} днів: {len(charges)} записів, "
                  f"${sum(float(r.get('amount') or 0) for r in charges):.2f}")
    changed = rec.reconcile_sessions(state_mod.state_dir(), charges, days=days,
                                     apply=not dry_run)
    console.print(f"[bold]сесії[/bold]: змінено {len(changed)}"
                  + (f", ${sum(o for _, o, _ in changed):.2f} → ${sum(n for _, _, n in changed):.2f}"
                     if changed else ""))
    rows = rec.ledger_from_bill(charges, manifest.load())
    months = budget_mod.reconcile_with_bill("vast", rows, apply=not dry_run)
    console.print("[bold]журнал витрат по місяцях[/bold]" + (" (сухо)" if dry_run else ""))
    for m in sorted(months):
        v = months[m]
        console.print(f"   {m}: ${v['before']:.2f} → ${v['after']:.2f}")


@htr_app.command("recover")
def htr_recover_cmd(
    plan_path: Path = typer.Option(..., "--plan", help="План заходу."),
    session: str = typer.Option(
        "", "--session", help="Сесія наглядача: з її стану — бокс, ціна, час смерті; "
                              "у неї ж дописується підсумок."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="лише показати: серцебиття, машину, що в сховищі"),
    cancel_box: bool = typer.Option(False, "--cancel-box",
                                    help="погасити живий бокс сесії після забору"),
) -> None:
    """Наглядач помер, а бокс працював далі: що сталося, забрати, звірити, порахувати гроші.

    Порядок: серцебиття бокса зі сховища → чи горить машина → забір усіх
    чекпоінтів у теки плану → звірка повноти → гроші, яких наглядач не бачив
    (ціна за годину × час після його смерті) → підсумок у стан сесії.
    """
    from rich.markup import escape

    from gpurunner.htr import recover as rec
    from gpurunner.htr.fetch_ckpt import cases_from_plan, fetch_case
    from gpurunner.supervise import state as state_mod
    from gpurunner.supervise.htr import plan_knob_str
    from gpurunner.supervise.plan import load_plan
    from gpurunner.supervise.verify import verify_case

    try:
        plan = load_plan(plan_path)
    except (OSError, ValueError) as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None
    state_file = state_mod.state_path(session) if session else None
    state: dict = {}
    if state_file is not None and state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))

    console.print("[bold]1. серцебиття бокса[/bold]")
    beat = rec.read_beat(plan.heartbeat_url)
    for line in rec.beat_summary(beat):
        console.print(f"   {line}")
    for line in (beat or {}).get("log_tail") or []:
        console.print(f"   [dim]{escape(line)}[/dim]")

    console.print("[bold]2. машина[/bold]")
    inst = str((state.get("box") or {}).get("instance_id") or "")
    alive = False
    if inst:
        from gpurunner.backends.vast import VastBackend

        alive = any(str(i.get("id")) == inst for i in VastBackend().live_instances())
        console.print(f"   бокс {inst}: " + ("[red]ЖИВИЙ, тарифікується[/red]" if alive
                                            else "погашено"))
        if alive and not (beat or {}).get("final"):
            console.print("   [yellow]бокс ще працює: перезапусти наглядача тим самим планом "
                          "і тим самим GPURUNNER_OWNER — він підхопить бокс, а не візьме "
                          "другий[/yellow]")
    else:
        console.print("   [dim]без --session невідомо, який бокс був цієї сесії[/dim]")

    console.print("[bold]3. забір зі сховища[/bold]" + (" (сухий)" if dry_run else ""))
    for rec_case in cases_from_plan(plan_path):
        try:
            fetch_case(rec_case, dry=dry_run)
        except Exception as e:
            err_console.print(f"   [red]{rec_case.get('case')}: {e}[/red]")

    console.print("[bold]4. повнота по диску[/bold]")
    model = plan_knob_str(plan, "model")
    done: list[tuple[str, Any]] = []
    lacking: list[tuple[str, Any]] = []
    for case in plan.cases:
        v = verify_case(Path(case.out_dir), expected_hint=case.n_pages, model=model)
        (done if v.complete else lacking).append((case.case, v))
    console.print(f"   повних {len(done)} із {len(plan.cases)}")
    for name, v in lacking:
        console.print(f"   [yellow]{name}: бракує {v.missing_count}[/yellow]")

    console.print("[bold]5. гроші поза наглядом[/bold]")
    billing_beat = beat
    if not beat:
        # План старший за серцебиття: останній слід бокса — найпізніший
        # чекпоінт його справ у сховищі. Доробив він, чи ні, видно з повноти.
        last = rec.last_store_write([c.ckpt_prefix for c in plan.cases])
        if last:
            billing_beat = {"t": last, "final": not lacking}
            console.print("   [dim]серцебиття немає — кінець роботи за останнім "
                          "чекпоінтом у сховищі[/dim]")
    extra, how = rec.unseen_spend(state, billing_beat, box_alive=alive,
                                  autodestroy_hours=plan.autodestroy_hours)
    seen = float((state.get("budget") or {}).get("spent_usd") or 0)
    billed_usd: float | None = None
    ids = rec.session_instances(
        state, state_mod.state_dir() / f"{session}.log" if session else None)
    if ids:
        try:
            from gpurunner.backends.vast import VastBackend

            since = time.time() - 14 * 86400
            bill = rec.billed(VastBackend().instance_charges(since, time.time()), ids)
            if bill["instances"]:
                billed_usd = bill["total"]
                parts = " · ".join(f"{k} ${v:.3f}" for k, v in sorted(bill["parts"].items()) if v)
                console.print(f"   рахунок Vast за {len(bill['instances'])} машин сесії: "
                              f"[bold]${billed_usd:.3f}[/bold] ({parts})")
        except Exception as e:
            console.print(f"   [yellow]рахунок Vast не прочитався: {e}[/yellow]")
    console.print(f"   наглядач бачив ${seen:.3f}; оцінка після його смерті ≈ ${extra:.3f} ({how})")
    if billed_usd is None:
        console.print(f"   разом ≈ ${seen + extra:.3f} (оцінка — рахунку Vast немає)")

    if cancel_box and alive and inst and not dry_run:
        from gpurunner.backends.vast import VastBackend

        VastBackend().destroy_box(inst)
        console.print(f"   🔥 бокс {inst} погашено")

    if not dry_run and state_file is not None and state_file.is_file():
        verdict = "ok" if not lacking else "incomplete"
        why = (f"{len(done)} справ повні (добрано без наглядача)" if not lacking else
               f"{len(done)} повні, {len(lacking)} — ні (добрано без наглядача)")
        rec.patch_state(state_file, extra_usd=extra, how=how, verdict=verdict, why=why,
                        billed_usd=billed_usd)
        console.print(f"   підсумок записано в стан {session}: {verdict}")
    if lacking:
        raise typer.Exit(code=4)


@htr_app.command("fetch-ckpt")
def htr_fetch_ckpt_cmd(
    plan_path: Path = typer.Option(None, "--plan", help="План заходу."),
    case: list[str] = typer.Option([], "--case", help="Ім'я прогону (без плану)."),
    out_root: Path = typer.Option(
        None, "--out-root", help="Куди класти — ОБОВ'ЯЗКОВО, якщо немає --plan."),
    bucket: str = typer.Option("", "--bucket"),
    hours: float = typer.Option(3.0, "--hours", help="Строк presigned-посилань."),
    origin: str = typer.Option(
        "", "--origin",
        help="Адреса складу на машині з секретом заходу (режим `box`). Без неї "
             "береться зі стану наглядача."),
    session: str = typer.Option(
        "", "--session", help="Чий стан читати по адресу складу."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Зібрати декод із чекпоінтів у R2 — без оренди й без наглядача.

    🔴 Перший крок після КОЖНОГО обірваного заходу: робота може бути ЖИВА й
    лежати в `ckpt/`, тоді як на диску її немає, а реєстр рахує справу
    непрочитаною. В одному з просторів так загубились 16 справ на
    півтора тижня; забір повернув 11 із них, 5538 сторінок, за $0.00.
    """
    from gpurunner.htr.fetch_ckpt import cases_from_plan, fetch_case
    from gpurunner.htr.r2 import R2Error

    if not origin:
        # 🔴 У режимі `box` посилань у плані немає за побудовою: їхня адреса —
        # це порт машини, якої на час складання плану ще не існувало. Знає її
        # лише наглядач, і саме тому він кладе її в стан. Питати цю адресу в
        # людини означало б вимагати того, чого їй нема де взяти.
        from gpurunner.supervise import state as state_mod

        data = state_mod.load(session or None) or {}
        origin = str(data.get("origin_base") or "")
        if origin:
            console.print("[dim]склад машини зі стану наглядача "
                          f"{data.get('session')}[/dim]")

    try:
        if plan_path is not None:
            cases = cases_from_plan(plan_path, origin_base=origin)
        elif case:
            if out_root is None:
                err_console.print(
                    "[red]без --plan потрібен --out-root: інакше забір не знає, "
                    "якому простору належить справа, і покладе декод поруч із "
                    "собою — саме так 95 сторінок опинились у чужому проєкті[/red]")
                raise typer.Exit(code=2)
            cases = [{"case": name, "n_pages": 0,
                      "out_dir": str((out_root / name).resolve())} for name in case]
        else:
            err_console.print("[red]треба --plan або --case[/red]")
            raise typer.Exit(code=2)
    except (OSError, ValueError) as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None

    rc = 0
    for rec in cases:
        try:
            result = fetch_case(rec, bucket=bucket, hours=hours, dry=dry_run,
                                log=console.print)
        except R2Error as e:
            err_console.print(f"[red]{rec['case']}: {e}[/red]")
            rc = 2
            continue
        if not result["ok"]:
            rc = 4
    if dry_run:
        console.print("[dim](сухий прогін — нічого не записано)[/dim]")
    raise typer.Exit(code=rc)


@htr_app.command("calibrate")
def htr_calibrate_cmd(
    since: str = typer.Option("", "--since", help="Лише заходи від дати YYYY-MM-DD."),
) -> None:
    """Калібровка моделі добору на завершених заходах, зшитих із матеріалом.

    Читає стан сесій наглядача і мету кожної справи на диску, друкує медіану
    темпу НА ШАРД за бінами рядків на сторінку і сітку сталих
    `A / L0 / A_core / A_card` з медіаною прогноз/факт.

    Правило вибору: медіана трохи НИЖЧА за 1 (орієнтир 0.95), бо завищений
    прогноз проходить ворота, а потім захід не встигає ні в бюджет, ні в строк.

    🔴 Дивитись не лише на спільну медіану, а й на дві колонки поруч — старий і
    новий раннер. 21.09.2026 спільне число було 1.06 (майже в межах), а за ним
    ховався розрив 0.96 / 1.27: модель завищувала рівно на тих флотах, які ми
    купуємо тепер. Колонки, що розходяться, означають, що бракує ЧЛЕНА, а не
    підбору сталої.
    """
    from gpurunner.htr.calibrate import (
        MIN_PAGES_FOR_RATE,
        NEW_RUNNER_SINCE,
        bins_table,
        fit_table,
        rated,
        stitch_runs,
    )

    rows = stitch_runs(since=since or "")
    if not rows:
        err_console.print("[red]заходів із темпом не знайдено[/red]")
        raise typer.Exit(code=2)
    with_lines = [r for r in rows if r.lines]
    fit_rows = rated(rows)
    console.print(f"заходів {len(rows)} · з рядками на сторінку {len(with_lines)} · "
                  f"придатних до калібровки {len(fit_rows)} "
                  f"(від {MIN_PAGES_FOR_RATE} сторінок) · "
                  f"новим раннером (від {NEW_RUNNER_SINCE}) "
                  f"{sum(1 for r in fit_rows if r.runner_new)}")
    t = Table(title="стор/год на ШАРД за щільністю матеріалу")
    for col in ("рядків/стор", "n", "медіана", "p25", "p75"):
        t.add_column(col, justify="right")
    for label, n, med, p25, p75 in bins_table(rows):
        t.add_row(label, str(n), f"{med:.0f}", f"{p25:.0f}", f"{p75:.0f}")
    console.print(t)
    for title, subset in (("усі заходи", with_lines),
                          ("лише новий раннер", [r for r in with_lines if r.runner_new])):
        grid = fit_table(subset)
        if not grid:
            continue
        g = Table(title=f"сітка сталих — {title} ({len(rated(subset))} заходів)")
        for col in ("A", "L0", "A_core", "A_card", "прогноз/факт",
                    "старий раннер", "новий раннер", "занижує >20%", "завищує >25%"):
            g.add_column(col, justify="right")
        for a, l0, a_core, a_card, med, med_old, med_new, under, over in sorted(
                grid, key=lambda x: abs(x[4] - 0.95))[:8]:
            g.add_row(f"{a:.0f}", f"{l0:.0f}",
                      "—" if a_core is None else f"{a_core:.0f}",
                      f"{a_card:.0f}", f"{med:.2f}",
                      f"{med_old:.2f}" if med_old else "—",
                      f"{med_new:.2f}" if med_new else "—",
                      f"{under:.0%}", f"{over:.0%}")
        console.print(g)


@htr_app.command("append")
def htr_append_cmd(
    plan_path: Path = typer.Option(..., "--plan", help="План, звідки взяти справу."),
    case_name: str = typer.Option(..., "--case", help="Ім'я справи з цього плану."),
    session: str = typer.Option("", "--session", help="Сесія (без неї — остання)."),
) -> None:
    """Довісити справу ЖИВОМУ заходу — щоб не платити холодний старт удруге.

    Холодний старт коштує ≈8 хв оренди плюс очікування ринку, а справа на 100
    сторінок при 1700 стор/год — 3.5 хв роботи. Тобто накладні у 2.5 раза
    більші за саму роботу, і платяться щоразу, коли справа згадалась пізніше.

    🔴 Команда НЕ пише на бокс сама, і це навмисно. Забір результату йде
    `zip(plan.cases, state.cases)`: справа, про яку знає бокс, але не знає
    наглядач, порахується, приїде в стейджинг і **нікуди не розкладеться** —
    оплачена й тихо втрачена. Тому єдиний писар на бокс — наглядач: команда
    кладе справу в чергу, наглядач на своєму тіку вносить її в план і в облік і
    аж тоді штовхає на бокс.

    Отже справа з'явиться на боксі не миттєво, а на найближчому тіку, і
    почнеться вона МІЖ справами — раннер не рве поточну.
    """
    from gpurunner.supervise import append as append_mod
    from gpurunner.supervise import state as state_mod
    from gpurunner.supervise.plan import load_plan

    data = state_mod.load(session or None) or {}
    sess = str(data.get("session") or session or "").strip()
    if not sess:
        err_console.print("[red]не знайшов заходу: вкажіть --session[/red]")
        raise typer.Exit(code=2)
    verdict = data.get("verdict")
    if verdict:
        err_console.print(
            f"[red]захід {sess} уже завершено (вердикт {verdict}) — довішувати нема куди[/red]")
        raise typer.Exit(code=2)

    plan = load_plan(plan_path)
    case = next((c for c in plan.cases if c.case == case_name), None)
    if case is None:
        have = ", ".join(c.case for c in plan.cases[:12]) or "(порожньо)"
        err_console.print(f"[red]у плані немає справи {case_name}[/red]; є: {have}")
        raise typer.Exit(code=2)

    known = {c.get("case") for c in (data.get("cases") or [])}
    if case.case in known:
        console.print(f"[yellow]{case.case} уже в заході — нічого не роблю[/yellow]")
        return

    # Попередня перевірка меж: остаточну зробить наглядач у момент штовхання,
    # але людина мусить почути відмову зараз, а не через хвилину.
    budget = data.get("budget") or {}
    left = budget.get("left_usd")
    if isinstance(left, (int, float)) and left <= 0:
        err_console.print(
            f"[red]бюджет заходу вичерпано (лишилось ${float(left):.2f})[/red]")
        raise typer.Exit(code=2)

    append_mod.enqueue(sess, {
        "case": case.case,
        "pages_url": case.pages_url,
        "n_pages": case.n_pages,
        "out_dir": case.out_dir,
        "case_key": case.case_key,
        "ckpt_urls": case.ckpt_urls,
        "resume_urls": case.resume_urls,
        "params": case.params,
    })
    console.print(f"[green]{case.case} ({case.n_pages} стор.) у черзі довіска "
                  f"заходу {sess}[/green]")
    console.print("[dim]наглядач візьме її на найближчому тіку й почне МІЖ "
                  "справами; стежити — gpurunner htr state --json[/dim]")


@htr_app.command("wrap-up")
def htr_wrapup_cmd(
    session: str = typer.Option("", "--session", help="Сесія (без неї — остання)."),
    why: str = typer.Option("", "--why", help="Причина — піде в журнал заходу."),
) -> None:
    """Попросити наглядача ЗГОРНУТИ захід: забрати, звірити, погасити оренду.

    🔴 Саме так зупиняють захід, коли він ще живий. `htr stop` убиває наглядача
    — і машина лишається горіти без нікого, хто її погасить; `htr quiesce`
    спиняє раннер, але наглядач про це не знає й далі чекає прогресу.

    Прохання кладеться поруч зі станом, і наглядач бачить його на черговому
    тіку. Команда повертається одразу: згортання займає стільки, скільки
    триває забір.
    """
    from gpurunner.supervise import state as state_mod
    from gpurunner.supervise import wrapup as wrapup_mod

    data = state_mod.load(session or None)
    name = str((data or {}).get("session") or session or "")
    if not name:
        err_console.print("[red]немає заходу, який можна згорнути[/red]")
        raise typer.Exit(code=1)
    if data and str(data.get("phase")) in ("done", "failed", "finished"):
        console.print(f"[dim]{name}: захід уже завершено "
                      f"({data.get('verdict') or data.get('phase')})[/dim]")
        raise typer.Exit(code=0)
    path = wrapup_mod.request(name, why=why)
    console.print(f"✅ {name}: попросив згорнути — наглядач забере прочитане, "
                  f"звірить і погасить оренду")
    console.print(f"[dim]прохання: {path}[/dim]")
    console.print(f"[dim]стежити: gpurunner htr state --session {name} "
                  f"--json[/dim]")


@htr_app.command("stop")
def htr_stop_cmd(
    session: str = typer.Option("", "--session", help="Сесія (без неї — остання)."),
    cancel_box: bool = typer.Option(
        False, "--cancel-box", help="Заразом погасити орендований бокс."),
) -> None:
    """Спинити наглядача сесії: процес, задача планувальника, замки.

    🔴 Це НЕ спосіб зупинити живий захід: наглядач помре, а машина лишиться
    горіти без забору й звірки. Для живого заходу — `gpurunner htr wrap-up`.

    🔴 Ручний `Stop-Process` по pid `uv.exe` наглядача НЕ вбиває — під ним живе
    окреме дерево, і 17.08.2026 головний процес довелось шукати за рядком
    `--plan` серед усіх `python.exe`, а замки справ чистити руками.

    ⚠ Бокс за замовчуванням НЕ гаситься: убити наглядача й лишити машину — це
    гроші, але погасити її ДО забору — це втрачена робота. Спершу
    `gpurunner htr fetch-ckpt`, потім `--cancel-box`.
    """
    from gpurunner.supervise import detach as detach_mod
    from gpurunner.supervise import state as state_mod

    data = state_mod.load(session or None)
    if data is None:
        err_console.print("[yellow]стану немає — наглядач не запускався[/yellow]")
        raise typer.Exit(code=1)
    run_session = str(data.get("session") or session)
    pid = int(data.get("pid") or 0)

    done = detach_mod.kill(run_session, pid)
    freed = detach_mod.release_locks(run_session, owner=manifest.current_owner())
    for line in done:
        console.print(f"[green]· {line}[/green]")
    for resource in freed:
        console.print(f"[green]· знято замок {resource}[/green]")
    if not done and not freed:
        console.print("[dim]нічого зупиняти: процесу, задачі й замків немає[/dim]")

    box = data.get("box") or {}
    if box.get("instance_id") or box.get("machine_id"):
        if cancel_box:
            console.print("[yellow]гашу бокс…[/yellow]")
            _print_live_instances()
            console.print("[dim]погасити конкретний: gpurunner cancel <id>[/dim]")
        else:
            console.print(
                f"[bold yellow]⚠ бокс machine {box.get('machine_id')} міг лишитись "
                f"живим — перевірте `gpurunner burn`. Спершу заберіть роботу:\n"
                f"  gpurunner htr fetch-ckpt --plan <план>[/bold yellow]")


@htr_app.command("quiesce")
def htr_quiesce_cmd(
    session: str = typer.Option("", "--session", help="Сесія (без неї — остання)."),
    handle_id: str = typer.Option("", "--handle", help="Хендл прогону замість сесії."),
) -> None:
    """Спинити раннер НА БОКСІ, не гасячи сам бокс.

    Потрібно там, де машина ще потрібна, а робота на ній — ні: заміряти іншу
    розкладку, підмінити модель, розібратися із завислим шардом.

    🔴🔴 Команда існує саме тому, що інакше її пишуть на місці — і пишуть
    неправильно. `pkill -f htr_case_run` збігається з рядком тієї ж оболонки,
    у якій сам `pkill` і виконується, тож вбиває СЕБЕ: 15 із 16 спроб. Пастка
    задокументована в проєкті, і це не врятувало — 04.09.2026 я наступив на неї
    двічі за одну сесію, маючи попередження перед очима.

    Тут шаблон у дужках: власний командний рядок містить `[h]tr_case_run.py`, а
    регекс шукає `htr_case_run.py` — і в собі його не знаходить. Приймач
    друкується завжди: скільки процесів лишилось і що з картою.
    """
    from gpurunner.backends.vast import VastBackend
    from gpurunner.core.models import JobStatus

    handle = None
    if handle_id:
        handle = manifest.get(handle_id)
        if handle is None:
            err_console.print(f"[red]невідомий хендл: {handle_id}[/red]")
            raise typer.Exit(code=2)
    else:
        from gpurunner.supervise import state as state_mod

        data = state_mod.load(session or None) or {}
        box = data.get("box") or {}
        remote = str(box.get("instance_id") or "")
        live = [h for h in manifest.load()
                if h.backend == "vast"
                and h.status in (JobStatus.QUEUED, JobStatus.RUNNING)]
        handle = next((h for h in live if h.remote_id == remote), None) or (
            live[-1] if live else None)
    if handle is None:
        err_console.print("[yellow]немає живого прогону на Vast[/yellow]")
        raise typer.Exit(code=1)

    script = (
        'for p in $(pgrep -f "[h]tr_case_run.py"); do kill -TERM "$p" 2>/dev/null; done\n'
        "sleep 6\n"
        'for p in $(pgrep -f "[h]tr_case_run.py"); do kill -KILL "$p" 2>/dev/null; done\n'
        "sleep 2\n"
        "echo LEFT=$(pgrep -cf '[h]tr_case_run.py' || echo 0)\n"
        "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader\n"
    )
    try:
        client = VastBackend()._ssh(handle, timeout=30)
    except (AuthError, BackendError) as e:
        err_console.print(f"[red]SSH не піднявся: {e}[/red]")
        raise typer.Exit(code=3) from None
    try:
        _, out, _ = client.exec_command(script, timeout=120)
        said = out.read().decode("utf-8", "replace").strip()
    finally:
        client.close()

    console.print(said or "[yellow]бокс промовчав[/yellow]")
    if "LEFT=0" in said:
        console.print(f"[green]раннер спинено, бокс {handle.remote_id} живий[/green]")
    else:
        console.print("[yellow]⚠ процеси лишились — подивіться вручну[/yellow]")
    console.print("[dim]бокс і далі тарифікується: gpurunner cancel "
                  f"{handle.id[:8]}[/dim]")


@htr_app.command("state")
def htr_state_cmd(
    session: str = typer.Option("", "--session", help="Ім'я сесії (без нього — остання)."),
    as_json: bool = typer.Option(False, "--json", help="Сирий JSON."),
    plan_path: Path = typer.Option(
        None, "--plan",
        help="План заходу — щоб перевірити, чи не лежить робота незабрана в R2."),
) -> None:
    """Стан заходу — єдине, що треба читати замість логів."""
    from gpurunner.supervise import state as state_mod

    data = state_mod.load(session or None)
    if data is None:
        err_console.print("[yellow]стану немає — наглядач ще не запускався[/yellow]")
        # 🔴 …або запустився ВІДЧЕПЛЕНО і впав до того, як завів власний лог.
        # Тоді єдиний слід — вивід самої задачі, і без нього причина зникає
        # безслідно: у планувальнику лишається код повернення й нічого більше.
        if session:
            from gpurunner.supervise import detach as detach_mod

            said = detach_mod.spawn_log(session)
            if said:
                err_console.print(
                    "[red]але відчеплена задача щось сказала на старті:[/red]")
                err_console.print(said)
        raise typer.Exit(code=1)
    if data.get("ambiguous"):
        # Чесна відмова замість упевненої неправди про чужий захід.
        if as_json:
            console.print_json(json.dumps(data, ensure_ascii=False))
            raise typer.Exit(code=2)
        err_console.print(f"[red]{data['why']}[/red]")
        for item in data.get("sessions") or []:
            err_console.print(
                f"  · {item.get('session')} — фаза {item.get('phase')}, "
                f"вердикт {item.get('verdict') or '—'}, оновлено {item.get('updated')}"
            )
        raise typer.Exit(code=2)
    if as_json:
        console.print_json(json.dumps(data, ensure_ascii=False))
        return

    console.print(f"[bold]{data.get('session')}[/bold] · фаза {data.get('phase')} · "
                  f"{data.get('why')}")
    budget = data.get("budget") or {}
    if budget:
        console.print(
            f"гроші: ${budget.get('spent_usd', 0):.2f} з ${budget.get('cap_usd', 0):.2f} · "
            f"{budget.get('elapsed_h', 0):.1f} з {budget.get('max_hours', 0):.1f} год"
        )
    box = data.get("box") or {}
    if box:
        console.print(
            f"бокс: {box.get('gpu')} · machine {box.get('machine_id')} · "
            f"${box.get('dph_total', 0):.3f}/год · {box.get('geolocation')}"
        )
    for case in data.get("cases") or []:
        mark = {"done": "✓", "incomplete": "✗", "failed": "✗"}.get(case.get("status"), "·")
        line = (f"  {mark} {case.get('case')}: {case.get('pages_done')}/"
                f"{case.get('n_pages_expected')}")
        if case.get("missing_count"):
            line += f" · бракує {case['missing_count']}"
        if case.get("detail"):
            line += f" · {case['detail']}"
        console.print(line)
        # 🔴 Абсолютний шлях — приймач того, що результат ліг у ПРАВИЛЬНИЙ
        # простір. Без нього «прогін нічого не дав» і «прогін ліг у чужий
        # проєкт» читаються однаково (спрацювало пять разів за кампанію).
        if case.get("out_dir"):
            console.print(f"      [dim]→ {case['out_dir']}[/dim]")

    _warn_about_unfetched_work(data, plan_path)
    for inc in (data.get("incidents") or [])[-10:]:
        console.print(f"  [yellow]⚠ {inc.get('kind')}: {inc.get('detail')}[/yellow]"
                      + (f" → {inc['action']}" if inc.get("action") else ""))
    if data.get("human_action_required"):
        console.print(f"[bold yellow]потрібне рішення: {data.get('human_action')}[/bold yellow]")


def _warn_about_unfetched_work(data: dict, plan_path: Path | None) -> None:
    """Сказати, що робота ЖИВА і лежить у R2, коли на диску її немає.

    🔴 Наглядач помирає між кінцем роботи й фазою забору частіше, ніж хотілось
    би: харнес убив таск, сесія агента впала, машину погасив дедлайн-сторож.
    Робота при цьому ЖИВА — вона в `ckpt/<справа>/<модель>/`. Але жоден звичний
    приймач її не бачить: на диску декоду немає, стан показує обірваний захід,
    реєстр рахує справу непрочитаною. В одному з просторів так тихо
    загубились 16 справ: півтора тижня вони стояли у звітах як невиконана
    робота, а забір повернув 11 із них — 5538 сторінок за $0.00.
    """
    unfinished = [c for c in (data.get("cases") or [])
                  if c.get("status") not in ("done",)]
    if not unfinished:
        return
    if plan_path is None:
        console.print(
            "[yellow]⚠ є незавершені справи. Перш ніж перезапускати — перевірте, "
            "чи не лежить уже пораховане в R2:\n"
            "  gpurunner htr fetch-ckpt --plan <план> --dry-run[/yellow]")
        return

    from gpurunner.htr.fetch_ckpt import count_ckpt_via_urls, count_on_disk

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        err_console.print(f"[yellow]план не прочитати: {e}[/yellow]")
        return
    by_case = {str(c.get("case")): c for c in plan.get("cases") or []}
    for case in unfinished:
        entry = by_case.get(str(case.get("case")))
        if not entry:
            continue
        out_dir = Path(str(entry.get("out_dir") or ""))
        on_disk = max(count_on_disk(out_dir).values()) if str(out_dir) else 0
        in_r2 = count_ckpt_via_urls([str(u) for u in (entry.get("resume_urls") or [])])
        if in_r2 and in_r2 > 0 and on_disk < int(entry.get("n_pages") or 0):
            console.print(
                f"[bold yellow]⚠ {case.get('case')}: на диску {on_disk}, "
                f"а в R2 лежить {in_r2} чекпоінт(ів) — ЗАБЕРІТЬ, не переплачуйте:\n"
                f"  gpurunner htr fetch-ckpt --plan {plan_path}[/bold yellow]")


# ---- boxes registry --------------------------------------------------------

boxes_app = typer.Typer(
    help="Реєстр орендних боксів: що ця машина обіцяла і що вона зробила.")
app.add_typer(boxes_app, name="boxes")


@boxes_app.command("ls")
def boxes_ls_cmd(
    all_rows: bool = typer.Option(False, "--all", help="Показати й невідомі машини."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Вердикти по машинах: кого не брати, кого шукати адресно."""
    from gpurunner.core import boxes

    verdicts = boxes.verdicts()
    rows = [v for v in verdicts.values() if all_rows or v.state != "unknown"]
    if not rows:
        console.print("[dim]реєстр порожній — жодної оренди ще не записано[/dim]")
        return
    rows.sort(key=lambda v: ({"banned": 0, "warned": 1, "starred": 2, "unknown": 3}[v.state],
                             -(v.runs_ok or 0)))
    if as_json:
        console.print_json(json.dumps(
            [{"machine_id": v.machine_id, "state": v.state, "reason": v.reason,
              "runs_ok": v.runs_ok, "gpu": v.gpu_name, "geo": v.geolocation,
              "measured": v.best_measured} for v in rows], ensure_ascii=False))
        return
    t = Table(show_header=True, header_style="bold")
    for col in ("machine", "стан", "gpu", "гео", "успіхів", "чому"):
        t.add_column(col)
    mark = {"banned": "[red]✗ бан[/red]", "warned": "[yellow]⚠ підозра[/yellow]",
            "starred": "[green]★ зірка[/green]", "unknown": "—"}
    for v in rows:
        t.add_row(str(v.machine_id), mark[v.state], v.gpu_name or "—",
                  v.geolocation or "—", str(v.runs_ok), v.reason)
    console.print(t)


@boxes_app.command("explain")
def boxes_explain_cmd(machine_id: int = typer.Argument(...)) -> None:
    """Уся історія машини: коли, що заміряли, чим скінчилось."""
    from gpurunner.core import boxes

    console.print(boxes.explain(machine_id))


@boxes_app.command("ban")
def boxes_ban_cmd(
    machine_id: int = typer.Argument(...),
    reason: str = typer.Option(..., "--reason", help="Чому — це попадає в git."),
) -> None:
    """Не брати цю машину ніколи. Ручне рішення не протухає по TTL."""
    _boxes_override("never", machine_id, reason)
    console.print(f"[red]machine {machine_id} — у чорному списку[/red]: {reason}")


@boxes_app.command("star")
def boxes_star_cmd(
    machine_id: int = typer.Argument(...),
    reason: str = typer.Option(..., "--reason"),
) -> None:
    """Шукати цю машину адресно, навіть якщо на ринку є дешевші."""
    _boxes_override("star", machine_id, reason)
    console.print(f"[green]machine {machine_id} — зірка[/green]: {reason}")


@boxes_app.command("prune")
def boxes_prune_cmd(
    apply: bool = typer.Option(False, "--apply", help="Справді переписати журнал."),
) -> None:
    """Прибрати вироки, за якими не стоїть знаменника.

    🔴 «Флот обсипався» виносився за часткою збоїв БЕЗ мінімального знаменника,
    тож дві невдалі сторінки з двох давали 100% і забирали з ринку СПРАВНУ
    машину на 21 день. Саме правило вже полагоджене, але записи, зроблені до
    того, лишились у реєстрі й далі відсіюють хости.

    ⚠ Стан хоста (`gone`, `exited`, `offline`) не чіпається: він про машину, а
    не про нашу арифметику.
    """
    from gpurunner.core import boxes

    dropped = boxes.prune_thin(apply=apply)
    if not dropped:
        console.print("[green]вироків без знаменника немає[/green]")
        return
    for obs, why in dropped:
        console.print(f"  [yellow]{obs.machine_id}[/yellow]: {why}")
    if apply:
        console.print(f"[green]прибрано {len(dropped)} спостереж.[/green]")
    else:
        console.print(f"[dim]{len(dropped)} спостереж.; --apply щоб прибрати[/dim]")


@boxes_app.command("forget")
def boxes_forget_cmd(machine_id: int = typer.Argument(...)) -> None:
    """Зняти РУЧНИЙ вердикт. Виміряну історію це не чіпає."""
    from gpurunner.core import boxes

    with boxes.mutate_overrides(note=f"forget {machine_id}") as data:
        removed = [k for k in ("never", "star")
                   if data[k].pop(str(machine_id), None) is not None]
    if removed:
        console.print(f"знято ручний вердикт ({', '.join(removed)}) з machine {machine_id}")
    else:
        console.print(f"[dim]ручного вердикту про machine {machine_id} не було[/dim]")


@boxes_app.command("absolve")
def boxes_absolve_cmd(
    machine_id: int = typer.Argument(...),
    since: str = typer.Option(..., "--since",
                              help="скасувати вироки з цієї миті (ISO, UTC), напр. 2026-09-23T21:00"),
    why: str = typer.Option(..., "--why", help="чия вада: посилання на коміт чи опис"),
) -> None:
    """Скасувати вироки машині, які поставила вада НАШОГО коду.

    Журнал не змінюється: скасовані спостереження лише не рахуються ударами.
    """
    from gpurunner.core import boxes

    rows = [o for o in boxes.read_all()
            if int(o.machine_id) == machine_id and o.verdict == "bad" and o.ts >= since]
    if not rows:
        console.print(f"[dim]у machine {machine_id} немає поганих вироків від {since}[/dim]")
        return
    with boxes.mutate_overrides(note=f"absolve {machine_id}") as data:
        for o in rows:
            data.setdefault("absolve", {})[boxes.absolve_key(machine_id, o.ts)] = why
    for o in rows:
        console.print(f"скасовано: machine {machine_id} · {o.ts[:16]} · {o.outcome}")


def _boxes_override(kind: str, machine_id: int, reason: str) -> None:
    from gpurunner.core import boxes

    with boxes.mutate_overrides(note=f"{kind} {machine_id}") as data:
        data.setdefault(kind, {})[str(machine_id)] = reason
        other = "star" if kind == "never" else "never"
        data.get(other, {}).pop(str(machine_id), None)


# ---- saturn catalogue ------------------------------------------------------

saturn_app = typer.Typer(
    help="Saturn Cloud. Типи інстансів залежать від конкретного деплою, тому "
         "їх видно лише через API акаунта.")
app.add_typer(saturn_app, name="saturn")


@saturn_app.command("sizes")
def saturn_sizes_cmd(
    gpu_only: bool = typer.Option(False, "--gpu-only", help="Лише розміри з GPU."),
) -> None:
    """Каталог instance_type цього акаунта — назви для `-p instance_type=…`."""
    from gpurunner.backends.saturn import SaturnBackend

    try:
        sizes = SaturnBackend().list_sizes()
    except (AuthError, BackendError) as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3) from None

    t = Table(show_header=True, header_style="bold")
    for col in ("instance_type", "cores", "RAM", "GPU", "опис"):
        t.add_column(col)
    for s in sizes:
        gpus = s.get("gpu") or 0
        if gpu_only and not gpus:
            continue
        t.add_row(
            str(s.get("name")),
            str(s.get("cores") or "—"),
            str(s.get("memory") or "—"),
            str(gpus or "—"),
            str(s.get("display_name") or s.get("description") or "—"),
        )
    console.print(t)
    console.print(
        "[dim]Запуск: gpurunner run <job> -b saturn --gpu T4 -p image=<образ> "
        "[-p instance_type=<назва>] -p input_root=<тека>[/dim]"
    )


# ---- status ---------------------------------------------------------------


@app.command()
def status(
    handle_id: str | None = typer.Argument(None, help="Handle id або remote id. Без нього — усі."),
    ping: bool = typer.Option(True, "--ping/--no-ping", help="Оновити статус із бекенда."),
    json_out: bool = typer.Option(
        False, "--json",
        help="Машинний вивід: останній рядок stdout — JSON зі станом (див. docs/contract.md)."),
) -> None:
    """Показати статус одного handle або перелік усіх відомих."""
    with contract.human_to_stderr(console, json_out):
        _status_impl(handle_id, ping, json_out)


def _status_impl(handle_id: str | None, ping: bool, json_out: bool) -> None:
    if handle_id is None:
        handles = manifest.load()
        if json_out:
            contract.emit({"ok": True, "command": "status",
                           "handles": [contract.handle_payload(h) for h in handles]})
            return
        if not handles:
            console.print("[dim]No tracked jobs.[/dim]")
            return
        table = Table(title="Known jobs")
        table.add_column("handle", style="cyan")
        table.add_column("status", style="magenta")
        table.add_column("backend")
        table.add_column("job")
        table.add_column("remote_id", overflow="fold")
        # Причина падіння прямо в списку: без неї єдиний спосіб дізнатись, чому
        # впав прогін тижневої давнини, — шукати його .log на диску.
        table.add_column("error", style="red", overflow="ellipsis", max_width=60)
        for row in sorted(handles, key=lambda x: x.created_at, reverse=True):
            table.add_row(
                row.id[:8],
                str(row.status.value if hasattr(row.status, "value") else row.status),
                row.backend,
                row.job_name,
                row.remote_id,
                (row.error or "").replace("\n", " ")[:60],
            )
        console.print(table)
        return

    h = manifest.get(handle_id)
    if h is None:
        err_console.print(f"[red]Unknown handle: {handle_id}[/red]")
        if json_out:
            contract.emit({"ok": False, "command": "status", "handle": handle_id,
                           "error": "unknown handle", "exit_code": 2})
        raise typer.Exit(code=2)

    refreshed, message = False, ""
    if ping:
        try:
            bk = get_backend(h.backend)()
            rep = bk.status(h)
            refreshed, message = True, str(rep.message or "")
            h.status = rep.status
            # Причину падіння зберігаємо, а не лише друкуємо: без цього поле
            # error лишалося порожнім у 284 хендлах поспіль, і відновити історію
            # відмов можна було тільки з .log-файлів на диску.
            h.error = rep.error
            manifest.update(h)
            extra = f"\n[dim]message:[/dim] {rep.message}" if rep.message else ""
            if rep.error:
                extra += f"\n[red]error:[/red] {rep.error}"
        except (AuthError, BackendError) as e:
            err_console.print(f"[yellow]Could not refresh status: {e}[/yellow]")
            extra = ""
    else:
        extra = ""

    body = (
        f"[bold]id[/bold]        {h.id}\n"
        f"[bold]backend[/bold]   {h.backend}\n"
        f"[bold]remote[/bold]    {h.remote_id}\n"
        f"[bold]job[/bold]       {h.job_name}\n"
        f"[bold]gpu[/bold]       {h.gpu}\n"
        f"[bold]status[/bold]    {h.status}\n"
        f"[bold]created[/bold]   {h.created_at}\n"
        f"[bold]output[/bold]    {h.output_dir or '—'}{extra}"
    )
    if h.backend == "colab":
        from gpurunner.backends.colab import ColabBackend

        body += f"\n[bold]notebook[/bold]  {ColabBackend().notebook_url(h)}"

    # Журнал переходів. Раніше хендл ніс лише ОСТАННІЙ статус, тож питання
    # «коли він насправді почав рахувати і скільки йшов» не мало відповіді —
    # її доводилось шукати в .log-файлах прогону, якщо ті взагалі вціліли.
    history = manifest.events(h.id)
    if history:
        body += "\n\n[bold]історія[/bold]"
        for ev in history[-8:]:
            stamp = str(ev["ts"]).replace("T", " ")[:19]
            line = f"\n  {stamp}  {ev['status'] or ''}"
            if ev["note"]:
                line += f"  [dim]{ev['note']}[/dim]"
            if ev["error"]:
                line += f"  [red]{str(ev['error']).splitlines()[0][:60]}[/red]"
            body += line
        span = _elapsed(history[0]["ts"], history[-1]["ts"])
        if span:
            body += f"\n  [dim]від першої до останньої події: {span}[/dim]"
    console.print(Panel(body, title=f"job {h.id[:8]}", border_style="cyan"))
    if json_out:
        state = contract.state_of(h.status)
        contract.emit({"ok": True, "command": "status", **contract.handle_payload(h),
                       # `refreshed: false` — стан із маніфесту: бекенд не відповів або --no-ping
                       "refreshed": refreshed, "terminal": state in contract.TERMINAL,
                       "message": message, "error": str(h.error or "")})


def _elapsed(first: str, last: str) -> str:
    """Людський проміжок між двома ISO-мітками; порожньо, якщо не розбираються."""
    try:
        delta = datetime.fromisoformat(str(last)) - datetime.fromisoformat(str(first))
    except (TypeError, ValueError):
        return ""
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}хв {seconds % 60}s"
    return f"{seconds // 3600}год {seconds % 3600 // 60}хв"


# ---- fetch ----------------------------------------------------------------


def _fetch_one(
    h: JobHandle,
    *,
    out: Path | None,
    resume: bool,
) -> tuple[str, int]:
    """Fetch outputs for a single handle. Returns (status_label, file_count).

    status_label is one of: 'ok', 'skipped', 'error: ...'.
    """
    target = out or (Path(h.output_dir) if h.output_dir else Path("out") / h.id[:8])

    if resume:
        try:
            job_cls = get_job(h.job_name)
            if job_cls().is_output_complete(target):
                return "skipped", 0
        except KeyError:
            # Unknown job — fall back to "any file in dir" heuristic.
            if target.exists() and any(p.is_file() for p in target.rglob("*")):
                return "skipped", 0

    target.mkdir(parents=True, exist_ok=True)
    try:
        bk = get_backend(h.backend)()
        files = bk.fetch_outputs(h, target)
    except (AuthError, BackendError) as e:
        return f"error: {e}", 0

    h.output_dir = str(target.resolve())
    manifest.update(h)
    return "ok", len(files)


@app.command()
def fetch(
    handle_ids: list[str] = typer.Argument(
        None,
        help="Один чи кілька handle id (або remote id). З --all не вказувати.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Тека під результати ОДНОГО handle. З --all ігнорується (береться output_dir кожного).",
    ),
    all_handles: bool = typer.Option(
        False,
        "--all",
        help="Забрати всі handle зі статусом COMPLETED із маніфесту. Звужується через --backend.",
    ),
    backend_filter: str | None = typer.Option(
        None,
        "--backend",
        "-b",
        help="З --all: лише handle цього бекенда.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Пропускати handle, чия тека результатів уже повна (за Job.is_output_complete).",
    ),
    json_out: bool = typer.Option(
        False, "--json",
        help="Машинний вивід: останній рядок stdout — JSON із результатом по кожному handle."),
) -> None:
    """Забрати артефакти віддаленого прогону."""
    with contract.human_to_stderr(console, json_out):
        _fetch_impl(handle_ids, out, all_handles, backend_filter, resume, json_out)


def _fetch_impl(handle_ids: list[str] | None, out: Path | None, all_handles: bool,
                backend_filter: str | None, resume: bool, json_out: bool) -> None:
    fetched: list[dict[str, Any]] = []
    if all_handles:
        targets = manifest.load()
        if backend_filter:
            targets = [h for h in targets if h.backend == backend_filter]
        targets = [h for h in targets if h.status == JobStatus.COMPLETED]
        if not targets:
            console.print("[dim]No COMPLETED handles match filter.[/dim]")
            return
    else:
        if not handle_ids:
            err_console.print("[red]fetch needs at least one handle id, or --all[/red]")
            raise typer.Exit(code=2)
        targets = []
        for hid in handle_ids:
            try:
                h = manifest.get(hid)
            except manifest.AmbiguousHandle as e:
                err_console.print(f"[red]{e}[/red]")
                raise typer.Exit(code=2) from None
            if h is None:
                err_console.print(f"[red]Unknown handle: {hid}[/red]")
                raise typer.Exit(code=2)
            targets.append(h)

    if out is not None and len(targets) > 1:
        err_console.print(
            "[red]--out is only valid for a single handle; "
            "for batches each handle uses its own output_dir.[/red]"
        )
        raise typer.Exit(code=2)

    n_ok = n_skipped = n_err = total_files = 0
    for h in targets:
        per_out = out if (out is not None and len(targets) == 1) else None
        label, files = _fetch_one(h, out=per_out, resume=resume)
        target = per_out or (Path(h.output_dir) if h.output_dir else Path("out") / h.id[:8])
        fetched.append({"handle": h.id, "short": h.id[:8], "result": label.split(":")[0],
                        "files": files, "output_dir": str(target),
                        "error": label[7:] if label.startswith("error: ") else ""})
        if label == "ok":
            n_ok += 1
            total_files += files
            console.print(
                f"[green]✓[/green] {h.id[:8]} ({h.job_name}, {h.backend}): "
                f"{files} file(s) → [cyan]{target}[/cyan]"
            )
        elif label == "skipped":
            n_skipped += 1
            console.print(
                f"[dim]· {h.id[:8]} ({h.job_name}): skipped — already complete at {target}[/dim]"
            )
        else:
            n_err += 1
            err_console.print(f"[red]✗[/red] {h.id[:8]} ({h.job_name}): {label}")

    if all_handles or len(targets) > 1:
        console.print(
            f"\n[bold]fetched {n_ok}[/bold] · skipped {n_skipped} · errors {n_err} · "
            f"{total_files} files total"
        )
    if json_out:
        contract.emit({"ok": n_err == 0, "command": "fetch", "fetched": fetched,
                       **({"exit_code": 3} if n_err else {})})
    if n_err:
        raise typer.Exit(code=3)


# ---- sweep ----------------------------------------------------------------


def _read_params_file(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per line. Empty lines / # comments allowed."""
    if not path.exists():
        raise typer.BadParameter(f"params file not found: {path}")
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"{path}:{i}: invalid JSON: {e}") from None
        if not isinstance(obj, dict):
            raise typer.BadParameter(f"{path}:{i}: expected JSON object, got {type(obj).__name__}")
        out.append(obj)
    return out


@app.command()
def sweep(
    job: str = typer.Argument(..., help="Назва job. Див. `gpurunner ls`."),
    backend: str = typer.Option("modal", "--backend", "-b", help="Назва бекенда."),
    params_file: Path = typer.Option(
        ...,
        "--params-file",
        "-f",
        help="JSONL: один JSON-об'єкт параметрів на рядок. Поле 'label' (за бажанням) задає теку результатів.",
    ),
    out_root: Path = typer.Option(
        Path("out") / "sweep",
        "--out-root",
        help="Результати кожного запуску лягають у <out-root>/<label>/.",
    ),
    max_concurrent: int = typer.Option(
        10,
        "--max-concurrent",
        "-c",
        help=(
            "Скільки job може виконуватись одночасно. План Modal Starter обмежує "
            "одночасні GPU-виклики десятьма; понад ліміт запуски зависають "
            "у стані 'Inactive' і не стартують."
        ),
    ),
    gpu: str = typer.Option("", "--gpu", "-g", help="Прискорювач. Без нього — default_gpu бекенда."),
    poll: int = typer.Option(30, "--poll", "-i", help="Інтервал опитування статусу, с."),
    skip_complete: bool = typer.Option(
        True,
        "--skip-complete/--no-skip-complete",
        help="Пропускати рядки, чия тека результатів уже проходить Job.is_output_complete.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Показати план, нічого не запускаючи."),
) -> None:
    """Пакетний запуск: по одному job на рядок --params-file, з лімітом одночасних."""
    try:
        job_cls = get_job(job)
        backend_cls = get_backend(backend)
    except KeyError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None

    rows = _read_params_file(params_file)
    if not rows:
        err_console.print(f"[red]No params rows in {params_file}[/red]")
        raise typer.Exit(code=2)

    job_inst = job_cls()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Plan: derive (label, target_dir, job_params) for each row, optionally
    # filtering out already-complete targets.
    plan: list[tuple[str, Path, dict[str, Any]]] = []
    skipped_labels: list[str] = []
    for i, row in enumerate(rows):
        row = dict(row)
        label = str(row.pop("label", f"row_{i:04d}"))
        target = out_root / label
        if skip_complete and job_inst.is_output_complete(target):
            skipped_labels.append(label)
            continue
        plan.append((label, target, row))

    console.print(
        f"[bold]sweep plan[/bold]  job={job}  backend={backend}  "
        f"rows={len(rows)}  to-do={len(plan)}  already-complete={len(skipped_labels)}  "
        f"out={out_root}"
    )
    if skipped_labels:
        head = ", ".join(skipped_labels[:6])
        more = f" (+{len(skipped_labels) - 6} more)" if len(skipped_labels) > 6 else ""
        console.print(f"[dim]  skipping: {head}{more}[/dim]")

    if not plan:
        console.print("[green]nothing to do[/green]")
        return

    if dry_run:
        for label, target, p in plan:
            console.print(f"  [cyan]{label}[/cyan] → {target}  (params keys: {sorted(p.keys())})")
        return

    bk = backend_cls()
    try:
        bk.check_auth()
    except AuthError as e:
        err_console.print(f"[red]Auth failed: {e}[/red]")
        raise typer.Exit(code=2) from None

    effective_gpu = gpu or bk.default_gpu or "T4"
    n_bad = _run_sweep(
        job_inst,
        bk,
        plan,
        gpu=effective_gpu,
        max_concurrent=max_concurrent,
        poll=poll,
        out_root=out_root,
    )
    if n_bad:
        raise typer.Exit(code=2)


def _run_sweep(
    job_inst: Any,
    bk: Any,
    plan: list[tuple[str, Path, dict[str, Any]]],
    *,
    gpu: str,
    max_concurrent: int,
    poll: int,
    out_root: Path,
) -> int:
    """Core sweep loop. Returns the number of failed rows (>0 means partial failure)."""
    """Core sweep loop: submit up to ``max_concurrent``, drain on terminal, repeat."""
    pending: dict[str, tuple[str, Path, JobHandle]] = {}  # handle_id → (label, target, handle)
    results: dict[str, str] = {}
    started = time.monotonic()

    terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}

    def _drain() -> int:
        drained = 0
        for hid in list(pending.keys()):
            label, target, h = pending[hid]
            try:
                rep = bk.status(h)
            except (AuthError, BackendError) as e:
                stamp = int(time.monotonic() - started)
                console.print(f"[yellow][{stamp:>4}s] {label}: status error: {e}[/yellow]")
                continue
            if rep.status not in terminal:
                continue
            h.status = rep.status
            h.error = rep.error
            manifest.update(h)
            stamp = int(time.monotonic() - started)
            if rep.status == JobStatus.COMPLETED:
                try:
                    files = bk.fetch_outputs(h, target)
                except (AuthError, BackendError) as e:
                    results[label] = f"fetch-error: {e}"
                    err_console.print(f"[red][{stamp:>4}s] {label}: fetch error: {e}[/red]")
                else:
                    h.output_dir = str(target.resolve())
                    manifest.update(h)
                    results[label] = "ok"
                    console.print(
                        f"[green][{stamp:>4}s] {label}: COMPLETED — {len(files)} files → {target}[/green]"
                    )
            else:
                results[label] = f"{rep.status}: {rep.error or '—'}"
                err_console.print(
                    f"[yellow][{stamp:>4}s] {label}: {rep.status}"
                    + (f" — {rep.error}" if rep.error else "")
                    + "[/yellow]"
                )
            del pending[hid]
            drained += 1
        return drained

    for label, target, row_params in plan:
        while len(pending) >= max_concurrent:
            _drain()
            if len(pending) >= max_concurrent:
                time.sleep(poll)

        console.print(f"[dim]→ submitting {label}…[/dim]")
        try:
            handle = bk.submit(job_inst, row_params, gpu=gpu)
        except (AuthError, BackendError, ValueError, FileNotFoundError) as e:
            results[label] = f"submit-error: {e}"
            err_console.print(f"[red]submit error for {label}: {e}[/red]")
            continue
        handle.output_dir = str(target.resolve())
        manifest.add(handle)
        pending[handle.id] = (label, target, handle)
        console.print(
            f"  [dim]spawned {handle.id[:8]} remote={handle.remote_id}[/dim]"
        )

    while pending:
        _drain()
        if pending:
            time.sleep(poll)

    elapsed = time.monotonic() - started
    n_ok = sum(1 for v in results.values() if v == "ok")
    n_bad = len(results) - n_ok
    console.print(
        f"\n[bold]sweep done[/bold] in {elapsed / 60:.1f} min — "
        f"ok {n_ok} · failed {n_bad}"
    )

    summary_path = out_root / "_sweep_summary.json"
    summary_payload: dict[str, Any] = {
        "started_at": datetime.fromtimestamp(time.time() - elapsed, tz=UTC).isoformat(timespec="seconds"),
        "finished_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "elapsed_min": round(elapsed / 60, 1),
        "rows": len(plan),
        "ok": n_ok,
        "failed": n_bad,
        "results": results,
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"  summary: [cyan]{summary_path}[/cyan]")
    return n_bad


# ---- cancel ---------------------------------------------------------------


@app.command()
def cancel(
    handle_ids: list[str] = typer.Argument(
        None,
        help="Один чи кілька handle id (або remote id). З --all-running чи --from-file не вказувати.",
    ),
    all_running: bool = typer.Option(
        False,
        "--all-running",
        help="Погасити всі СВОЇ нетермінальні прогони (за GPURUNNER_OWNER).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="🔴 Дозволити гасити ЧУЖІ прогони, названі явно. Реєстр спільний на "
             "машину, тож без цього прапорця чужий інстанс не чіпається — саме "
             "так 2026-08-11 знищили роботу паралельної сесії.",
    ),
    any_owner: bool = typer.Option(
        False,
        "--any-owner",
        help="🔴 З --all-running: зачепити й ЧУЖІ прогони. Реєстр спільний на "
             "машині, тож це вб'є живі бокси паралельних сесій.",
    ),
    backend_filter: str | None = typer.Option(
        None,
        "--backend",
        "-b",
        help="З --all-running: лише цей бекенд.",
    ),
    from_file: Path | None = typer.Option(
        None,
        "--from-file",
        help="Читати handle id з файла, по одному на рядок (формат to_stop.txt).",
    ),
) -> None:
    """Погасити прогони — один, кілька або всі свої нетермінальні."""
    if all_running:
        targets = manifest.load()
        if backend_filter:
            targets = [h for h in targets if h.backend == backend_filter]
        non_terminal = {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.UNKNOWN}
        targets = [h for h in targets if h.status in non_terminal]
        # 🔴 Реєстр СПІЛЬНИЙ на всю машину. Без цього фільтра «погашу своє»
        # означає «погашу все, зокрема живі бокси паралельної сесії» — саме
        # так 2026-08-11 агенти п'ять разів гасили одне одному роботу.
        # Зараз у базі 49 нетермінальних хендлів від різних заходів.
        me = manifest.current_owner()
        if not any_owner:
            if not me:
                err_console.print(
                    "[red]--all-running без GPURUNNER_OWNER заборонено[/red]: реєстр "
                    "спільний для всіх сесій на машині, і масове гасіння вб'є чужі "
                    "живі бокси. Назви хендли явно або постав GPURUNNER_OWNER, "
                    "або --any-owner, якщо справді хочеш зачепити все."
                )
                raise typer.Exit(code=2)
            foreign = [h for h in targets if h.owner != me]
            targets = [h for h in targets if h.owner == me]
            if foreign:
                console.print(
                    f"[dim]пропущено {len(foreign)} чужих/безіменних прогонів "
                    f"(--any-owner щоб зачепити й їх)[/dim]"
                )
    elif from_file:
        if not from_file.exists():
            err_console.print(f"[red]File not found: {from_file}[/red]")
            raise typer.Exit(code=2)
        ids = [
            ln.strip()
            for ln in from_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        targets = []
        for hid in ids:
            try:
                h = manifest.get(hid)
            except manifest.AmbiguousHandle as e:
                err_console.print(f"[yellow]{e}[/yellow]")
                continue
            if h is None:
                err_console.print(f"[yellow]skip unknown handle: {hid}[/yellow]")
                continue
            targets.append(h)
    else:
        if not handle_ids:
            err_console.print(
                "[red]cancel needs handle ids, --all-running, or --from-file[/red]"
            )
            raise typer.Exit(code=2)
        targets = []
        for hid in handle_ids:
            try:
                h = manifest.get(hid)
            except manifest.AmbiguousHandle as e:
                err_console.print(f"[red]{e}[/red]")
                raise typer.Exit(code=2) from None
            if h is None:
                err_console.print(f"[red]Unknown handle: {hid}[/red]")
                raise typer.Exit(code=2)
            targets.append(h)

    # 🔴 Чужі прогони не гасяться без явного `--force`. Мітка інстансу для
    # цього не годиться: обидві сесії створюють `gpurunner-htr_case-…`, і саме
    # довіра до мітки коштувала чужої роботи.
    if not all_running and not force:
        me = manifest.current_owner()
        foreign = [h for h in targets if (h.owner or "") != me]
        if foreign:
            for h in foreign:
                case = (h.params or {}).get("case") or ""
                err_console.print(
                    f"[red]✗ {h.id[:8]} належить {h.owner or 'невідомо кому'}[/red]"
                    + (f" (справа {case})" if case else "")
                )
            err_console.print(
                f"[red]Це чужі прогони.[/red] Твій власник: {me or '(не заданий)'}. "
                f"Якщо ти справді хочеш їх знищити — `--force`; але спершу спитай "
                f"людину: на тому боксі може йти чужа робота."
            )
            raise typer.Exit(code=2)

    if not targets:
        console.print("[dim]nothing to cancel[/dim]")
        return

    # Показати, що саме зникне: інстанс коштує грошей і не воскресає.
    for h in targets:
        case = (h.params or {}).get("case") or ""
        console.print(f"[yellow]→ знищу[/yellow] {h.id[:8]} · {h.backend} · {h.job_name}"
                      + (f" · {case}" if case else "")
                      + (f" · власник {h.owner}" if h.owner else " · власник невідомий"))

    n_ok = n_err = 0
    for h in targets:
        try:
            bk = get_backend(h.backend)()
            bk.cancel(h)
        except (AuthError, BackendError) as e:
            n_err += 1
            err_console.print(f"[red]✗ {h.id[:8]} ({h.backend}): {e}[/red]")
            continue
        h.status = JobStatus.CANCELLED
        manifest.update(h)
        n_ok += 1
        console.print(f"[green]✓ cancelled[/green] {h.id[:8]} ({h.backend}, {h.job_name})")

    console.print(f"\n[bold]cancelled {n_ok}[/bold] · errors {n_err}")
    if n_err:
        raise typer.Exit(code=3)


# ---- watch ----------------------------------------------------------------


@app.command()
def watch(
    handle_id: str = typer.Argument(..., help="Handle id або remote id."),
    poll: int = typer.Option(30, "--poll", "-i", help="Інтервал опитування, с."),
    timeout: int | None = typer.Option(None, "--timeout", help="Припинити стеження через N секунд."),
) -> None:
    """Опитувати job, доки статус не стане термінальним."""
    h = manifest.get(handle_id)
    if h is None:
        err_console.print(f"[red]Unknown handle: {handle_id}[/red]")
        raise typer.Exit(code=2)

    terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    bk = get_backend(h.backend)()
    started = time.monotonic()
    last: str | None = None
    while True:
        try:
            rep = bk.status(h)
        except (AuthError, BackendError) as e:
            err_console.print(f"[yellow]Poll error: {e} — retrying[/yellow]")
            time.sleep(poll)
            continue
        h.status = rep.status
        h.error = rep.error
        manifest.update(h)
        line = f"[{time.strftime('%H:%M:%S')}] {rep.status}"
        if rep.message:
            line += f" — {rep.message}"
        if line != last:
            console.print(line)
            last = line
        if rep.status in terminal:
            if rep.status == JobStatus.COMPLETED:
                console.print(f"[green]done.[/green] Fetch with: [dim]gpurunner fetch {h.id[:8]}[/dim]")
            else:
                if rep.error:
                    err_console.print(f"[red]error:[/red] {rep.error}")
                _save_failure_log(bk, h)
            return
        if timeout is not None and time.monotonic() - started > timeout:
            console.print("[yellow]watch timeout[/yellow]")
            return
        time.sleep(poll)


def _save_failure_log(bk: Backend, h: JobHandle) -> None:
    """Best-effort dump of the backend-side log for a run that just failed.

    Kaggle drops a kernel's log when the session ends and Modal keeps outputs for
    ~24 h, so "I'll look at it tomorrow" has repeatedly meant "it's gone". The
    failures worth diagnosing are exactly the ones nobody fetches, since fetch is
    what people run after a *success*.
    """
    target = Path(h.output_dir) if h.output_dir else Path("out") / h.id[:8]
    try:
        lines = bk.logs(h)
    except (AuthError, BackendError, NotImplementedError):
        return
    if not lines:
        return
    try:
        target.mkdir(parents=True, exist_ok=True)
        dest = target / "_backend.log"
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8", errors="replace")
    except OSError as e:
        err_console.print(f"[yellow]could not save backend log: {e}[/yellow]")
        return
    console.print(f"[dim]backend log saved to {dest}[/dim]")


# ---- logs -----------------------------------------------------------------


@app.command()
def logs(
    handle_id: str = typer.Argument(..., help="Handle id або remote id."),
) -> None:
    """Надрукувати логи з віддаленого боку (якщо бекенд їх віддає)."""
    h = manifest.get(handle_id)
    if h is None:
        err_console.print(f"[red]Unknown handle: {handle_id}[/red]")
        raise typer.Exit(code=2)

    bk = get_backend(h.backend)()
    try:
        lines = bk.logs(h)
    except (AuthError, BackendError) as e:
        err_console.print(f"[red]Logs failed: {e}[/red]")
        raise typer.Exit(code=3) from None
    for ln in lines:
        console.print(ln)


# ---- dash -----------------------------------------------------------------


@app.command()
def dash(
    port: int = typer.Option(8765, "--port", "-p", help="Порт локального сервера."),
    host: str = typer.Option(
        "127.0.0.1", "--host",
        help="Адреса. Дашборд не має авторизації і показує баланси — "
             "виставляти його назовні не варто."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Відкрити браузер."),
) -> None:
    """Підняти локальний дашборд: баланси, квоти, активні прогони."""
    try:
        import uvicorn

        from gpurunner.web.app import create_app
    except ImportError as e:
        err_console.print(
            f"[red]Дашборду бракує залежностей ({e.name}).[/red]\n"
            "Встанови екстру: [cyan]uv sync --extra web[/cyan]"
        )
        raise typer.Exit(code=2) from None

    url = f"http://{host}:{port}/"
    console.print(f"[green]дашборд:[/green] [cyan]{url}[/cyan]  [dim](Ctrl+C щоб зупинити)[/dim]")
    if host not in ("127.0.0.1", "localhost", "::1"):
        err_console.print(
            "[yellow]⚠ слухаю не на localhost, а авторизації тут немає — "
            "баланси й імена прогонів побачить кожен, хто дотягнеться до порту[/yellow]"
        )
    if open_browser:
        import threading
        import webbrowser

        # Затримка, щоб браузер не постукав у порт раніше, ніж uvicorn його займе.
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
