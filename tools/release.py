#!/usr/bin/env python3
"""Реліз gpurunner: один шлях, який не можна пройти неправильно.

    python tools/release.py check 0.2.5      # лише перевірки, нічого не змінює
    python tools/release.py gates            # чи зелене дерево — вирок, якому можна вірити
    python tools/release.py gates --dirty    # те саме, але зі своїм незакоміченим
    python tools/release.py cut   0.2.5 --apply

Цей репозиторій **приватний**, а пакет публічний, і між ними стоїть
orphan-гілка `public`. Через це реліз має чотири правила, кожне з яких уже
порушувалось руками, і кожне коштувало окремо:

🔴 **Тег ставиться на `public`, НІКОЛИ на `main`.** `release.yml` тригериться
на `tags: ["v*"]`, а ворота приватних даних дивляться історію САМЕ ТЕГА. Тег
на `main` запускає реліз із приватного дерева: сторож там дає 130 збігів
(пошта автора, шляхи робочих просторів, прізвище роду). У найкращому разі це
червоний прогін, у гіршому — спроба обійти ворота `--force`.

🔴 **`public` збирається з ДЕРЕВА КОМІТА, а не з робочої теки.** Звичний
`git checkout --orphan` + `git add -A` забирає все, що лежить у теці, — а там
може писати інша сесія. 21.09.2026 така спроба двічі забрала чужу
незакомічену роботу в сусідньому репозиторії. Тут замість цього
`git commit-tree <ref>^{tree}`: робочої теки команда не торкається взагалі.

🔴 **Ворота ганяються на релізному дереві, а не на робочому.** Навіть у
worktree з чистого коміта `.venv` головного репо підсовує код через
editable-встановлення: `gpurunner.__file__` вказує в `<репо>/src`. 22.09.2026
це дало 5 «падінь релізу», яких у релізі не було. Тому скрипт ставить окреме
середовище ВСЕРЕДИНІ worktree.

🔴 **PyPI не дає правити метадані випущеної версії.** Сторінка проєкту бере
адреси з ОСТАННЬОГО релізу, а колесо не видаляється й номер не
перевикористовується. Тому всі перевірки стоять ДО пушу тега, а не після.

Що скрипт робить сам, а що лишає людині: коміт із версією та журналом змін
робить людина (це зміст, а не механіка). Скрипт перевіряє, що він є,
правильний і запушений, — і далі веде публікацію до кінця, включно з
дзеркалом.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Ім'я дистрибутива на PyPI. ⚠ НЕ дорівнює імені пакета: `gpurunner` там
#: зайняв схожий проєкт, і майданчик не дав його взяти.
DIST = "gpuhire"

#: Публічне дзеркало: те саме чисте дерево, куди ведуть адреси пакета.
#: Приватний репозиторій лишається домом розробки.
MIRROR = "SERGIUSH-UA/gpuhire"
MIRROR_URL = f"https://github.com/{MIRROR}.git"

#: Orphan-гілка публікації. Не merge-гілка: перезбирається щоразу.
PUBLIC = "public"

#: Звідки береться дерево релізу. Гілка розробки, не робоча тека.
DEFAULT_REF = "main"

VERSION_FILE = "src/gpurunner/__init__.py"
VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.M)
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


# ── дрібні чисті помічники (їх перевіряють тести) ───────────────────────────
def version_of(text: str) -> str:
    """`__version__` із тексту `__init__.py`. Немає — порожньо."""
    got = VERSION_RE.search(text)
    return got.group(1) if got else ""


def changelog_has(text: str, version: str) -> bool:
    """Чи є в журналі розділ саме цієї версії.

    🔴 Межа обов'язкова: без неї `0.2.1` знайшлося б усередині `0.2.10`, і
    реліз поїхав би з чужими нотатками.
    """
    head = re.escape(f"## [{version}]")
    return re.search(rf"^{head}(?=[\s—-])", text, re.M) is not None


def tag_of(version: str) -> str:
    return f"v{version}"


def valid_version(version: str) -> bool:
    return bool(SEMVER_RE.match(version))


# ── git ──────────────────────────────────────────────────────────────────────
def git(*args: str, check: bool = True, cwd: Path | None = None) -> str:
    done = subprocess.run(["git", *args], cwd=str(cwd or ROOT), text=True,
                          encoding="utf-8", errors="replace",
                          capture_output=True, check=False)
    if check and done.returncode:
        raise SystemExit(f"🔴 git {' '.join(args)} → код {done.returncode}\n"
                         f"{(done.stderr or done.stdout).strip()}")
    return (done.stdout or "").strip()


def file_at(ref: str, path: str) -> str:
    """Вміст файла В КОМІТІ, а не в робочій теці.

    🔴 Уся перевірка версії й журналу читає саме так. Робоча тека може містити
    чужі правки, півкоміту або нічого — а поїде рівно те, що в `ref`.
    """
    return git("show", f"{ref}:{path}")


def ref_exists(ref: str) -> bool:
    return subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref],
                          cwd=str(ROOT), capture_output=True).returncode == 0


# ── перевірки ────────────────────────────────────────────────────────────────
@dataclass
class Report:
    """Підсумок перевірок. Порожній `problems` — можна публікувати."""

    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, why: str) -> None:
        self.problems.append(why)
        print(f"  🔴 {why}")

    def ok(self, what: str) -> None:
        print(f"  ✅ {what}")

    def note(self, what: str) -> None:
        self.notes.append(what)
        print(f"  ⚠ {what}")


def on_pypi(version: str) -> bool | None:
    """Чи вже лежить ця версія на PyPI. `None` — спитати не вдалось.

    🔴 Три стани, і зводити їх до двох не можна: «не знаю» показане як «ні»
    коштувало б спаленого номера версії, бо колесо не видаляється.
    """
    url = f"https://pypi.org/simple/{DIST}/"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            body = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None
    return f"{DIST}-{version}" in body


def supervisor_alive() -> bool:
    """Чи живий відчеплений наглядач.

    🔴 Доки він живий, `uv sync` у цьому проєкті перезаписує `site-packages`
    під процесом, який звідти виконується (на Windows — `os error 32`, файл
    зайнятий). А скрипт саме `uv sync` і робить, щоб зібрати чисте середовище.
    """
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq gpurunner.exe"],
                             capture_output=True, text=True, check=False).stdout or ""
        return "gpurunner.exe" in out
    return subprocess.run(["pgrep", "-f", "gpurunner htr supervise"],
                          capture_output=True, check=False).returncode == 0


def check(version: str, ref: str, rep: Report) -> None:
    """Усе, що можна перевірити без жодного запису.

    🔴 Починається з `fetch`, і це не ввічливість. Дві перевірки нижче читають
    ГІЛКИ ВІДСТЕЖЕННЯ (`origin/<ref>`, теги origin), а `cut` пушить `public` із
    `--force-with-lease`, який звіряється з ними ж. Зі застарілим
    `origin/public` пуш відхиляється — безпечно, але посеред релізу це
    виглядає як незрозуміла відмова; а застарілий `origin/<ref>` ще й збреше
    про «не запушено». У репозиторії, де паралельно працює інша сесія, це не
    рідкість, а типовий стан.
    """
    print(f"\n▶ версія {version}, дерево {ref}")
    git("fetch", "--quiet", "origin", "--tags", check=False)

    if not valid_version(version):
        rep.fail(f"«{version}» не схоже на X.Y.Z")
        return
    if not ref_exists(ref):
        rep.fail(f"немає гілки {ref}")
        return

    got = version_of(file_at(ref, VERSION_FILE))
    if got == version:
        rep.ok(f"{VERSION_FILE} у {ref} каже {got}")
    else:
        rep.fail(f"{VERSION_FILE} у {ref} каже «{got}», а реліз — {version}. "
                 f"Версія й тег мусять збігатись: це перевіряє й сам workflow")

    if changelog_has(file_at(ref, "CHANGELOG.md"), version):
        rep.ok(f"CHANGELOG має розділ [{version}]")
    else:
        rep.fail(f"у CHANGELOG.md немає розділу «## [{version}] — …». "
                 f"Порожня сторінка релізу читається як «нічого не змінилось»")

    tag = tag_of(version)
    if ref_exists(tag):
        rep.fail(f"тег {tag} уже існує локально — номер не перевикористовується")
    elif git("ls-remote", "--tags", "origin", tag):
        rep.fail(f"тег {tag} уже є в origin")
    else:
        rep.ok(f"тег {tag} вільний")

    published = on_pypi(version)
    if published is True:
        rep.fail(f"{DIST} {version} уже на PyPI — колесо не видаляється")
    elif published is None:
        rep.note("спитати PyPI не вдалось; це НЕ означає «версії там немає»")
    else:
        rep.ok(f"на PyPI {DIST} {version} ще немає")

    unpushed = git("log", "--oneline", f"origin/{ref}..{ref}", check=False)
    if unpushed:
        rep.fail(f"{ref} не запушено ({len(unpushed.splitlines())} комітів) — "
                 f"CI зібрав би інше дерево")
    else:
        rep.ok(f"{ref} = origin/{ref}")

    dirty = git("status", "--porcelain")
    if dirty:
        rep.note(f"у робочій теці {len(dirty.splitlines())} змінених файлів — "
                 f"у реліз вони НЕ потраплять (дерево береться з {ref})")

    if supervisor_alive():
        rep.fail("живий наглядач (`gpurunner.exe`): `uv sync` для воріт "
                 "перезапише середовище під ним. Дочекайтесь кінця заходу")
    else:
        rep.ok("наглядача не запущено — середовище можна збирати")


# ── публікація ───────────────────────────────────────────────────────────────
def build_public(version: str, ref: str) -> str:
    """Перезібрати `public` як orphan-коміт із ДЕРЕВА `ref`.

    🔴 Ні `checkout`, ні `add`, ні `stash`: робоча тека не читається взагалі.
    `commit-tree` бере готовий об'єкт дерева й робить коміт без предків.
    """
    tree = git("rev-parse", f"{ref}^{{tree}}")
    template = (ROOT / "tools" / "public_commit.txt").read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(template.format(version=version))
        msg_path = fh.name
    try:
        commit = git("commit-tree", tree, "-F", msg_path)
    finally:
        os.unlink(msg_path)
    git("branch", "-f", PUBLIC, commit)

    # Приймач: дерево публікації тотожне релізному. Порожній діф — єдиний
    # доказ, що в `public` поїхало саме те, що перевіряли.
    diff = git("diff", "--stat", PUBLIC, ref)
    if diff:
        raise SystemExit(f"🔴 public розійшовся з {ref}:\n{diff}")
    return commit


def dirty_paths(porcelain_z: str) -> tuple[list[str], list[str]]:
    """Розбір `git status --porcelain -z` → (що скопіювати, що прибрати).

    🔴 Саме `-z`: у звичайному виводі шлях із не-ASCII береться в лапки й
    екранується (`"data/\\320\\260"`), і наївний розбір дає неіснуючий файл.
    Справи тут звуться `spr-47а`, `spr-84г`, тож це не екзотика.
    """
    parts = [p for p in porcelain_z.split("\0") if p]
    copy: list[str] = []
    drop: list[str] = []
    i = 0
    while i < len(parts):
        entry = parts[i]
        i += 1
        if len(entry) < 4:
            continue
        xy, path = entry[:2], entry[3:]
        # перейменування: наступний запис — стара назва, і її треба зняти
        if xy[0] in "RC" and i < len(parts):
            drop.append(parts[i])
            i += 1
        (drop if "D" in xy else copy).append(path)
    return copy, drop


def apply_dirty(tree: Path) -> int:
    """Накласти незакомічені зміни робочої теки на ізольований зліпок.

    Відповідь на питання «чи зелена робота, якої ще немає в комітах» — його
    `gates` із дерева коміта дати не може за побудовою.

    🔴 Жодного запису в репозиторій: файли ЧИТАЮТЬСЯ й копіюються в тимчасову
    теку. Ні `add`, ні `stash`, ні індексу — у дереві цієї миті може писати
    інша сесія, і будь-яка команда, що чіпає індекс, забрала б її роботу.
    """
    copy, drop = dirty_paths(git("status", "--porcelain", "-z"))
    for rel in drop:
        target = tree / rel
        if target.is_file():
            target.unlink()
    for rel in copy:
        src = ROOT / rel
        if not src.is_file():
            continue
        target = tree / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    return len(copy) + len(drop)


def gates(ref: str, *, run_tests: bool, dirty: bool = False) -> None:
    """Ворота на РЕЛІЗНОМУ дереві, в ізольованому середовищі.

    🔴 Окреме середовище всередині worktree обов'язкове. Без нього
    editable-встановлення головного репо підсовує його `src`, і ворота
    перевіряють код, якого в релізі немає.
    """
    work = Path(tempfile.mkdtemp(prefix="gpurunner-rel-"))
    tree = work / "tree"
    try:
        git("worktree", "add", "-q", "--detach", str(tree), ref)
        if dirty:
            n = apply_dirty(tree)
            print(f"\n▶ ворота на ЗЛІПКУ робочої теки поверх {ref} "
                  f"({n} файлів, {tree})")
            print("  ⚠ це НЕ те, що поїде в реліз: зліпок містить незакомічене")
        else:
            print(f"\n▶ ворота на чистому дереві {ref} ({tree})")
        if run_tests:
            run(["uv", "sync", "--locked", "--group", "dev", "--extra", "all"], tree)
        py = tree / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not py.is_file():
            py = Path(sys.executable)

        # ⚠ Звідки береться КОД, важить лише для pytest: він імпортує пакет.
        # `ruff`, `mypy` і сторож приватних даних читають ФАЙЛИ в цій теці, і
        # для них чужий інтерпретатор нешкідливий. Сторож, поставлений на всі
        # ворота одразу, забороняв би й нешкідливе — а такий швидко знімають.
        if run_tests:
            where = subprocess.run(
                [str(py), "-c", "import gpurunner;print(gpurunner.__file__)"],
                cwd=str(tree), capture_output=True, text=True,
                check=False).stdout.strip()
            if where and not where.lower().startswith(str(tree).lower()):
                raise SystemExit(
                    f"🔴 тести дивились би НЕ на релізне дерево: gpurunner узято з\n"
                    f"   {where}\n"
                    f"   Це editable-встановлення головного репо; у такому "
                    f"вигляді вони нічого не доводять.")
            print(f"  ✅ код береться з worktree: {where}")
        else:
            print("  ⚠ --skip-tests: ідуть лише перевірки ФАЙЛІВ (ruff, mypy, "
                  "приватні дані). Що код робить — не перевірено")
        run([str(py), "-m", "ruff", "check", "src", "tests"], tree)
        run([str(py), "-m", "mypy", "src"], tree)
        run([str(py), "tools/scan_private.py"], tree)
        # ⚠ Нотатки релізу тут НЕ перевіряються: вони прив'язані до номера
        # версії, а не до дерева, і `check` уже питає про них іменем версії.
        # Виклик із гілкою замість версії друкував «немає розділу [main]» —
        # тривога, яка нічого не означає.
        if run_tests:
            run([str(py), "-m", "pytest", "-q"], tree)
    finally:
        git("worktree", "remove", "--force", str(tree), check=False)
        git("worktree", "prune", check=False)
        shutil.rmtree(work, ignore_errors=True)


def run(cmd: list[str], cwd: Path, *, check: bool = True) -> None:
    print(f"  · {' '.join(cmd[:4])}…")
    done = subprocess.run(cmd, cwd=str(cwd), check=False)
    if check and done.returncode:
        raise SystemExit(f"🔴 впало: {' '.join(cmd)} (код {done.returncode})")


def mirror_ready() -> bool:
    """Чи вимкнено Actions у дзеркалі.

    🔴 У дзеркалі лежать ТІ САМІ workflow'и. Увімкнені Actions означають, що
    пуш тега туди запускає другий реліз: 22.09.2026 чотири такі прогони
    впали з `invalid-publisher` — PyPI відхилив токен чужого репозиторію.
    Шкоди не сталось лише тому, що Trusted Publishing звужений до одного репо.
    """
    if not shutil.which("gh"):
        print("  ⚠ немає `gh` — стан Actions дзеркала не перевірено")
        return True
    out = subprocess.run(["gh", "api", f"repos/{MIRROR}/actions/permissions"],
                         capture_output=True, text=True, check=False).stdout or ""
    if '"enabled":false' in out.replace(" ", ""):
        return True
    print(f"  🔴 у дзеркалі {MIRROR} УВІМКНЕНО Actions — пуш тега запустить "
          f"там другий реліз. Вимкнути:\n"
          f"     gh api -X PUT repos/{MIRROR}/actions/permissions -F enabled=false")
    return False


def cut(version: str, ref: str, *, run_tests: bool, mirror: bool) -> None:
    rep = Report()
    print("▶ перевірки перед публікацією")
    check(version, ref, rep)
    if rep.problems:
        raise SystemExit(f"\n🔴 не публікую: {len(rep.problems)} причин вище")

    gates(ref, run_tests=run_tests)

    print("\n▶ гілка публікації")
    commit = build_public(version, ref)
    print(f"  ✅ public = {commit[:9]} (orphan, дерево {ref})")

    env = {**os.environ, "GPURUNNER_PUBLIC_REF": PUBLIC}
    done = subprocess.run([sys.executable, "tools/scan_private.py", "--public"],
                          cwd=str(ROOT), env=env, check=False)
    if done.returncode:
        raise SystemExit("🔴 у гілці публікації знайдено приватні дані — тег не ставлю")

    if mirror and not mirror_ready():
        raise SystemExit("🔴 дзеркало не готове")

    tag = tag_of(version)
    print("\n▶ публікація")
    git("push", "--force-with-lease", "origin", PUBLIC)
    git("tag", tag, PUBLIC)
    git("push", "origin", tag)
    print(f"  ✅ {tag} на {PUBLIC} → release.yml запущено")

    if mirror:
        git("push", MIRROR_URL, tag)
        git("push", "--force", MIRROR_URL, f"{PUBLIC}:main")
        print(f"  ✅ дзеркало {MIRROR} на {tag}")

    print(f"\n✅ готово. Далі стежити: gh run list --limit 1\n"
          f"   Після успіху звірити адреси: "
          f"https://pypi.org/pypi/{DIST}/json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["check", "gates", "cut"])
    ap.add_argument("version", nargs="?", default="",
                    help="X.Y.Z (без «v»); для `gates` не потрібна")
    ap.add_argument("--ref", default=DEFAULT_REF, help=f"дерево релізу (типово {DEFAULT_REF})")
    ap.add_argument("--apply", action="store_true",
                    help="для `cut`: справді публікувати")
    ap.add_argument("--skip-tests", action="store_true",
                    help="ворота без pytest і без збирання середовища (швидко, "
                         "але доводить менше)")
    ap.add_argument("--no-mirror", action="store_true", help="не чіпати дзеркало")
    ap.add_argument("--dirty", action="store_true",
                    help="для `gates`: накласти НЕЗАКОМІЧЕНІ зміни робочої теки "
                         "на зліпок. Репозиторій не змінюється; у реліз таке "
                         "дерево не поїде")
    args = ap.parse_args()

    version = args.version.lstrip("v")

    if args.action == "gates":
        # 🔴 Відповідь на питання «чи зелене дерево?», якій можна вірити.
        # Повний `pytest` у спільній робочій теці НЕ відповідає на нього:
        # результат залежить від того, чиї півправки лежать у ній цієї
        # хвилини, і 22.09.2026 це двічі дало фальшивий вирок — в обидва
        # боки. Тут дерево береться з коміта, середовище збирається окремо.
        if supervisor_alive():
            print("🔴 живий наглядач: `uv sync` перезапише середовище під ним")
            return 1
        gates(args.ref, run_tests=not args.skip_tests, dirty=args.dirty)
        how = ("окреме середовище, тести" if not args.skip_tests
               else "лише перевірки файлів")
        what = f"зліпок робочої теки поверх {args.ref}" if args.dirty else args.ref
        print(f"\n✅ {what} зелений ({how})")
        return 0

    if not version:
        print("🔴 потрібна версія: X.Y.Z")
        return 2

    if args.action == "check":
        rep = Report()
        check(version, args.ref, rep)
        print()
        if rep.problems:
            print(f"🔴 {len(rep.problems)} причин не публікувати")
            return 1
        print("✅ перевірки пройдено; ворота на дереві ганяє `cut`")
        return 0

    if not args.apply:
        print("це змінить гілку публікації й поставить тег. "
              "Додайте --apply, коли впевнені.")
        return 2
    cut(version, args.ref, run_tests=not args.skip_tests, mirror=not args.no_mirror)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
