"""Службові файли їдуть архівом через сховище, а не пофайловим SFTP.

🔴🔴 Вада, яку тут закрито, коштувала цілого заходу 23.09.2026. Тіло результату
(5115 сторінок, 237 справ) приїхало зі сховища одним об'єктом на чекпоінт і
лежало вдома повним. Після цього лишався «дотяг лише службового» по SFTP — і на
черзі він вироджувався в `listdir` на КОЖНУ справу плюс `get` на КОЖЕН лог
шарда: ~2600 обертів через океан. Фаза забору висіла понад 30 хвилин, з'їла свою
стелю й лишила всі 237 справ без вердикту.

Заміри одного плеча на орендованому боксі (`htr/box_transport.py`): SFTP
paramiko 0.48 МБ/с проти 20.07 у HTTP — у 42 рази. За 99 заходів SFTP не дав
повноти НІ РАЗУ самостійно і п'ять разів зіпсував вердикт уже привезеному
текстові.
"""

from __future__ import annotations

import ast
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from gpurunner.supervise.htr import Supervisor
from gpurunner.supervise.plan import CasePlan, Plan

RUNNER = (Path(__file__).resolve().parents[1]
          / "src" / "gpurunner" / "_embedded" / "htr_case_runner.py")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))


# ---- бокс: раннер віддає архів сам ---------------------------------------------


def test_the_runner_packs_the_service_files_and_puts_them(tmp_path: Path,
                                                          monkeypatch) -> None:
    """Один tar, один `PUT` — тим самим шляхом, яким їдуть чекпоінти.

    У архіві мусить бути і корінь боксу (лог раннера, стан, прогрес), і логи
    шардів САМЕ ЦІЄЇ справи: 11.09.2026 (904-24-198) причина 34 збоїв згоріла
    разом із боксом, бо чекпоінт логів не везе.
    """
    from gpurunner._embedded import htr_case_runner as runner

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "_runner.log").write_text("лог боксу", encoding="utf-8")
    (root / "_status.json").write_text('{"phase": "done"}', encoding="utf-8")
    case_out = tmp_path / "working" / "spr-3644"
    (case_out / "logs").mkdir(parents=True)
    (case_out / "logs" / "shard_00.log").write_text("шард нуль", encoding="utf-8")
    (case_out / "logs" / "shard_01.log").write_text("шард один", encoding="utf-8")

    sent: list[list[str]] = []

    def _curl(cmd: list[str], *a: Any, **kw: Any) -> int:
        sent.append(list(cmd))
        # Копіюємо тіло: після виклику раннер прибирає тимчасовий файл.
        body = Path(cmd[cmd.index("--upload-file") + 1])
        (tmp_path / "uploaded.tgz").write_bytes(body.read_bytes())
        return 0

    monkeypatch.setattr(runner.subprocess, "call", _curl)
    added = runner._service_to_url("https://r2/service/spr-3644.tgz?sig=1",
                                   str(case_out), root=str(root))

    assert added == 4, "два файли кореня плюс два логи шардів"
    assert len(sent) == 1, "один PUT на справу, а не по файлу"
    assert sent[0][:2] == ["curl", "-fsS"] and "PUT" in sent[0]
    assert "--max-time" in sent[0], (
        "без стелі зависання на віддачі не обмежене нічим — це той самий "
        "мовчазний простій на оплачуваній машині")

    with tarfile.open(tmp_path / "uploaded.tgz") as tf:
        names = sorted(tf.getnames())
    assert names == ["_runner.log", "_status.json",
                     "logs/shard_00.log", "logs/shard_01.log"]


def test_the_runner_survives_a_storage_that_does_not_take_it(tmp_path: Path,
                                                             monkeypatch) -> None:
    """🔴 Діагностика не варта того, щоб через неї падала справа.

    Посилання могло протухнути, сховище — відмовити. Текст при цьому вже
    порахований і залитий, і втратити його через невдалий лог було б абсурдом.
    """
    from gpurunner._embedded import htr_case_runner as runner

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "_runner.log").write_text("лог", encoding="utf-8")
    monkeypatch.setattr(runner.subprocess, "call", lambda *a, **kw: 22)

    assert runner._service_to_url("https://r2/x.tgz", None, root=str(root)) == 0


def test_the_runner_sends_the_bundle_after_every_case_not_at_the_end() -> None:
    """🔴 Віддача стоїть у `finally` циклу черги, поруч зі скиданням кадрів.

    Інакше діагностика справи, після якої бокс помер, лишається на боксі — а
    саме такі справи й треба розбирати. Сторож дивиться СТРУКТУРУ, бо порядок
    тут і є вся суть: «у кінці заходу» виглядало б так само правильно.
    """
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "run_one"):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Try) and inner.finalbody:
                calls = [n.func.id for n in ast.walk(ast.Module(inner.finalbody, []))
                         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
                if "_service_to_url" in calls:
                    return
        raise AssertionError("`_service_to_url` немає у `finally` справи")
    raise AssertionError("не знайшов `run_one` у раннері")


def test_the_bundle_goes_to_the_box_path_not_the_home_one() -> None:
    """🪤 Логи шардів лежать у теці БОКСУ (`_out_root`).

    `out_dir` — адреса вдома (`E:\\Projects\\MeGen\\reports\\htr\\…`), і на
    боксі її не існує. Помилка тут не падає: архів просто поїхав би без логів,
    тобто перевірка виглядала б зробленою.
    """
    src = RUNNER.read_text(encoding="utf-8")
    i = src.index("_service_to_url(merged")
    window = src[i:i + 200]
    assert '"_out_root"' in window, "віддача мусить брати теку боксу"
    assert '"out_dir"' not in window


# ---- дім: забір одним GET ------------------------------------------------------


def _plan_with_service(tmp_path: Path) -> Plan:
    return Plan(
        assets_url="https://r2/a.tgz",
        cases=[CasePlan(case="spr-3644", pages_url="https://r2/p.tar", n_pages=2,
                        out_dir=str(tmp_path / "out" / "spr-3644"),
                        service_put_url="https://r2/service/spr-3644.tgz?put",
                        service_url="https://r2/service/spr-3644.tgz?get")],
        budget_usd=1.0, max_hours=1.0,
    )


def _bundle(path: Path) -> None:
    """Такий самий архів, який складає раннер."""
    tray = path.parent / "_tray"
    (tray / "logs").mkdir(parents=True, exist_ok=True)
    (tray / "_runner.log").write_text("лог боксу", encoding="utf-8")
    (tray / "_progress.json").write_text('{"pages_done": 2}', encoding="utf-8")
    (tray / "logs" / "shard_00.log").write_text("шард", encoding="utf-8")
    with tarfile.open(path, "w:gz") as tf:
        for item in ("_runner.log", "_progress.json", "logs/shard_00.log"):
            tf.add(tray / item, arcname=item)


def test_the_supervisor_takes_the_bundle_by_http_and_lays_it_out(
        tmp_path: Path, monkeypatch) -> None:
    """Корінь — у корінь стейджингу, логи — у теку справи.

    Розкладка забирає три файли кореня з КОРЕНЯ стейджингу (`_runner.log` і
    далі) і переносить у теку першої справи; логи шардів мусять лягти всередину
    своєї справи. Помилившись розкладкою, ми привезли б лог і стерли б його
    разом зі стейджингом — а бокс на той момент уже знищено.
    """
    from gpurunner.supervise import htr as htr_mod

    sup = Supervisor(_plan_with_service(tmp_path), backend=object(),  # type: ignore[arg-type]
                     session="S")
    staging = tmp_path / "staging"

    def _curl(cmd: list[str], *a: Any, **kw: Any) -> int:
        out = Path(cmd[cmd.index("-o") + 1])
        assert cmd[-1] == "https://r2/service/spr-3644.tgz?get"
        _bundle(out)
        return 0

    monkeypatch.setattr(htr_mod.subprocess, "call", _curl, raising=False)
    import subprocess as _sp

    monkeypatch.setattr(_sp, "call", _curl)
    got = sup._fetch_service_bundles(staging)

    assert (staging / "_runner.log").read_text(encoding="utf-8") == "лог боксу"
    assert (staging / "_progress.json").is_file()
    assert (staging / "spr-3644" / "logs" / "shard_00.log").is_file()
    assert len(got) == 3
    assert not list(staging.glob("*.tgz")), "тимчасовий архів мусить прибиратись"


def test_a_missing_bundle_is_not_an_incident(tmp_path: Path, monkeypatch) -> None:
    """404 = раннер до цієї справи не дійшов. Це стан, а не збій.

    🔴 Вирок справі ставить ЗВІРКА ПО ДИСКУ, і діагностика на нього не впливає
    ніяк: саме плутанина цих двох речей поставила `failed` 237 справам, у яких
    текст був повний.
    """
    import subprocess as _sp

    sup = Supervisor(_plan_with_service(tmp_path), backend=object(),  # type: ignore[arg-type]
                     session="S")
    monkeypatch.setattr(_sp, "call", lambda *a, **kw: 22)

    assert sup._fetch_service_bundles(tmp_path / "staging") == []
    assert not sup.state.incidents, "404 на діагностиці не інцидент"


def test_a_broken_bundle_is_named_but_does_not_stop_the_queue(
        tmp_path: Path, monkeypatch) -> None:
    """Битий архів — нотатка з іменем справи, і рух далі."""
    import subprocess as _sp

    sup = Supervisor(_plan_with_service(tmp_path), backend=object(),  # type: ignore[arg-type]
                     session="S")

    def _junk(cmd: list[str], *a: Any, **kw: Any) -> int:
        Path(cmd[cmd.index("-o") + 1]).write_text("це не архів", encoding="utf-8")
        return 0

    monkeypatch.setattr(_sp, "call", _junk)
    assert sup._fetch_service_bundles(tmp_path / "staging") == []
    assert any(i.kind == "service_bundle_bad" for i in sup.state.incidents)


# ---- схема плану ---------------------------------------------------------------


def test_the_plan_carries_both_directions_of_the_service_link(
        tmp_path: Path, monkeypatch) -> None:
    """PUT для раннера й GET для нас — presigned підписується окремо на кожен.

    🪤 Саме тому одного поля мало: на складі машини це був би один шлях (раннер
    розрізняє напрям методом запиту), а в бакеті — два різні підписи.
    """
    from gpurunner.htr import r2
    from gpurunner.htr.plan_build import BuildOptions, build_plan

    case = tmp_path / "frames" / "spr-3644"
    case.mkdir(parents=True)
    (case / "0001.jpg").write_bytes(b"\xff\xd8jpeg")

    class _S3:
        def __init__(self) -> None:
            self.uploaded: list[str] = []

        def upload_file(self, path: str, bucket: str, key: str, **kw: Any) -> None:
            self.uploaded.append(key)

        def generate_presigned_url(self, op: str, *, Params: dict, ExpiresIn: int) -> str:
            return f"https://r2/{Params['Key']}?op={op}"

        def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
            raise RuntimeError("404")

    s3 = _S3()
    monkeypatch.setattr(r2, "client", lambda env=None: s3)
    plan = build_plan([case], BuildOptions(out_root=(tmp_path / "out").resolve(),
                                           model="pysar_cyr_v17.pt"),
                      assets_key="assets/a.tgz", log=lambda _l: None)
    entry = plan["cases"][0]
    assert entry["service_put_url"] == "https://r2/service/spr-3644.tgz?op=put_object"
    assert entry["service_url"] == "https://r2/service/spr-3644.tgz?op=get_object"

    # І назад — план мусить це прочитати, інакше поле є лише в JSON.
    from gpurunner.supervise.plan import load_plan

    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded = load_plan(path)
    assert loaded.cases[0].service_put_url.endswith("op=put_object")
    assert loaded.cases[0].service_url.endswith("op=get_object")


def test_the_runner_gets_somewhere_to_put_the_bundle(tmp_path: Path) -> None:
    """🔴 Посилання мусить дійти до параметрів справи в черзі.

    Без цього віддача в раннері тихо повертає нуль — тобто вся ця робота
    виглядала б зробленою, а діагностика лишалась би на боксі.
    """
    from gpurunner.core.offer_score import Need

    plan = _plan_with_service(tmp_path)
    sup = Supervisor(plan, backend=object(), session="S")  # type: ignore[arg-type]
    params = sup._params_for(plan.cases[0], Need(pages=2, max_hours=1.0, budget_usd=1.0),
                            resume=False, queue=plan.cases)

    assert params["service_put_url"] == "https://r2/service/spr-3644.tgz?put"
    assert params["cases"][0]["service_put_url"] == "https://r2/service/spr-3644.tgz?put"


def test_the_box_store_serves_the_bundle_at_the_same_address() -> None:
    """Склад на машині: та сама тека, той самий шлях для PUT і GET."""
    from gpurunner.htr import box_transport as bt

    urls = bt.urls_for("spr-3644", "ckpt/spr-3644", 4)
    assert urls["service_put_url"].endswith("/service/spr-3644.tgz")
    assert urls["service_put_url"].startswith(bt.base_url())


# ---- стеля часу на віддачу ------------------------------------------------------


def test_every_upload_of_the_runner_has_a_ceiling() -> None:
    """🔴 `curl` без `--max-time` на завислому сокеті стоїть НАВІЧНО.

    У раннера це дорожче за будь-де: віддача стоїть у циклі роботи, тож
    завислий `curl` спиняє не забір, а ЧИТАННЯ — бокс тарифікується, прогрес не
    рухається, і зовні це виглядає як «працює». Саме так і було з чекпоінтами:
    стелі там не стояло взагалі.

    Сторож дивиться на КОЖНУ віддачу, а не на конкретний виклик: наступна
    поїде тим самим `curl`, і про неї ніхто не згадає.
    """
    import ast

    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    uploads = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        argv = node.args[0]
        if not isinstance(argv, ast.List):
            continue
        flat = [a.value for a in argv.elts if isinstance(a, ast.Constant)]
        if "--upload-file" not in flat:
            continue
        uploads += 1
        assert "--max-time" in flat, (
            "віддача без стелі часу: " + " ".join(str(x) for x in flat[:6]))
    assert uploads >= 2, f"знайдено {uploads} віддач — сторож дивиться не туди"


def test_the_ceiling_grows_with_the_file() -> None:
    """Стала стеля або вбиває великий чекпоінт, або не ловить малий.

    Тому вона рахується від розміру, з підлогою: стеля АВАРІЇ, а не очікуваний
    час. Темп навмисно втричі-вчотирнадцятеро нижчий за виміряний — вона мусить
    ловити мертвий сокет, а не карати повільний, але живий канал.
    """
    from gpurunner._embedded.htr_case_runner import UPLOAD_MIN_SEC, _upload_ceiling

    assert _upload_ceiling(0) == UPLOAD_MIN_SEC
    assert _upload_ceiling(1_000_000) == UPLOAD_MIN_SEC, "дрібне бере підлогу"
    assert _upload_ceiling(500_000_000) > UPLOAD_MIN_SEC * 5, "велике бере запас"
