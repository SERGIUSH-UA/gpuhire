"""Плече дім→бокс: дані їдуть HTTP у склад, `scp` лишається запасним.

🔴 Чому це взагалі переписано. Те саме залізо й той самий канал, заміряно на
орендованому боксі 20.09.2026 на 150 МБ:

| чим                       | МБ/с  |
|---------------------------|-------|
| SFTP paramiko             | 0.48  |
| зворотний тунель paramiko | 0.49  |
| системний `scp`           | 7.56  |
| HTTP одним потоком        | 20.07 |

Розкид у 42 рази вирішує протокол, не смуга. На архіві 2.25 ГБ різниця між
`scp` і HTTP — це п'ять хвилин проти чверті години, і та чверть години йде ДО
першої прочитаної сторінки, на вже оплаченій машині.

Тут перевіряється те, що не видно з коду: справжній склад приймає `PUT`,
обірвана заливка НЕ виглядає доїханою, а вибір плеча робиться ЗА ВИМІРОМ.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import pytest

from gpurunner.core.backend import BackendError
from gpurunner.htr import box_transport as bt
from gpurunner.htr import origin

TOKEN = secrets.token_urlsafe(18)


@pytest.fixture
def store(tmp_path: Path):
    """Справжній склад із секретом: (база url із секретом, тека)."""
    root = tmp_path / "origin"
    httpd = origin.serve(root, port=0, host="127.0.0.1", token=TOKEN)
    base = bt.base_url(httpd.server_address[1], host="127.0.0.1", token=TOKEN)
    yield base, root
    httpd.shutdown()


def test_a_file_travels_into_the_store_and_lands_whole(store, tmp_path: Path) -> None:
    """Той самий шлях, яким потім качає раннер: `PUT` у склад, `GET` зі складу."""
    base, root = store
    local = tmp_path / "assets.tgz"
    local.write_bytes(b"w" * 250_000)

    assert bt.push_http(base, local, "assets.tgz") == 250_000
    assert (root / "assets.tgz").read_bytes() == local.read_bytes()
    assert not list(root.rglob("*.part")), "часткових файлів не лишається"


def test_a_case_archive_goes_where_the_runner_will_look(store, tmp_path: Path) -> None:
    """Адреса заливки й адреса читання — один шлях, інакше 404 на оплаченій карті."""
    base, root = store
    local = tmp_path / "sprava.tar"
    local.write_bytes(b"k" * 4096)

    bt.push_http(base, local, "cases/sprava.tar")
    reads = bt.urls_for("sprava", "ckpt/sprava", 2, port=0, token=TOKEN)
    assert reads["pages_url"].endswith(f"/{TOKEN}/cases/sprava.tar")
    assert (root / "cases" / "sprava.tar").is_file()


def test_a_wrong_secret_does_not_land_anything(store, tmp_path: Path) -> None:
    """🔴 Саме `PUT` робить відкритий склад небезпечним.

    Бокс стоїть у чужому дата-центрі з публічною адресою: без секрету будь-хто
    писав би на нашу машину що завгодно.
    """
    base, root = store
    local = tmp_path / "x.bin"
    local.write_bytes(b"z" * 1000)
    wrong = base.rsplit("/", 1)[0] + "/ne-toi-sekret"

    with pytest.raises(BackendError):
        bt.push_http(wrong, local, "cases/x.bin")
    assert not (root / "cases" / "x.bin").exists()


def test_a_truncated_upload_is_not_called_delivered(store, tmp_path: Path) -> None:
    """🔴 Обірвана заливка лишає файл, який ВИГЛЯДАЄ покладеним.

    Виявлялось це вже на оплачуваній карті — тим, що раннер не може розпакувати
    архів. Тому розмір звіряється тим самим каналом: склад пише через `.part` і
    перейменовує лише ціле, тож `HEAD` на повний розмір — чесна відповідь
    «доїхало», а не «щось там лежить».
    """
    base, root = store
    local = tmp_path / "half.tar"
    local.write_bytes(b"q" * 8192)

    # Склад отримує тіло коротше за оголошену довжину — рівно так виглядає
    # обрив посеред заливки.
    import urllib.error
    import urllib.request

    req = urllib.request.Request(f"{base}/cases/half.tar", method="PUT",
                                 data=b"q" * 100)
    req.add_header("Content-Length", "8192")
    # Коротка стеля навмисно: склад чекає решти тіла, і саме клієнт має
    # здатись першим — інакше тест платить десятьма секундами за очікування,
    # яке нічого не доводить.
    with pytest.raises((urllib.error.HTTPError, OSError)):
        urllib.request.urlopen(req, timeout=2)

    assert not (root / "cases" / "half.tar").exists(), (
        "півархіву під справжнім іменем не буває")
    assert bt.head_size(f"{base}/cases/half.tar") == -1


def test_a_silent_curl_is_not_a_delivery(store, tmp_path: Path, monkeypatch) -> None:
    """🔴 Успішний код `curl` — це НЕ доказ, що файл на машині.

    Тут `curl` «спрацював», не передавши нічого (так виглядає й обірване
    з'єднання, і хибний шлях, і повний диск на боксі). Рішення ухвалює звірка
    розміру тим самим каналом, а не код виходу процесу: інакше захід їде далі й
    падає вже на оплачуваній карті, коли раннер не може розпакувати архів.
    """
    base, root = store
    local = tmp_path / "nothing.tar"
    local.write_bytes(b"n" * 4096)
    monkeypatch.setattr(bt, "_run", lambda *a, **kw: None)

    with pytest.raises(BackendError, match="обірвалась"):
        bt.push_http(base, local, "cases/nothing.tar")
    assert not (root / "cases" / "nothing.tar").exists()


def test_the_health_answer_needs_the_secret_too(store) -> None:
    """🔴 Інакше це маячок «тут склад» для будь-якого сканера портів.

    Рівно те, від чого на складі стоїть 404 замість 403: відповідь не має
    підтверджувати сканеру, що за цим портом є ціль.
    """
    base, _ = store
    without = base.rsplit("/", 1)[0]
    assert bt.reachable(base) is True
    assert bt.reachable(without) is False
    assert bt.reachable(f"{without}/ne-toi-sekret") is False


def test_the_store_reports_whether_it_answers_from_here(store) -> None:
    """Проба — це ВИМІР, а не здогад про хостера.

    Порт складу мусить бути прокинутий орендою назовні, і чи він досяжний
    звідси — не знає ніхто, крім самого запиту: між нами й боксом бувають
    фаєрвол хоста, NAT і просто інший дата-центр.
    """
    base, _ = store
    assert bt.reachable(base) is True
    assert bt.reachable("http://127.0.0.1:1/nope", timeout=2) is False


def test_the_secret_is_not_in_the_command_line_of_the_store() -> None:
    """🔴 Рядок запуску видно в `ps` сусідам по хосту, а бокс багатоквартирний.

    Префікс `VAR=... python3` теж видно: його бачить оболонка, і саме її
    командний рядок лишається в `ps`. Тому секрет їде ФАЙЛОМ із правами 600.
    """
    said: list[str] = []
    files: dict[str, str] = {}

    def run(cmd: str) -> str:
        said.append(cmd)
        return "200" if "http_code" in cmd else ""

    bt.start_origin(run, lambda path, text: files.__setitem__(path, text),
                    token="sekret-zakhodu")

    launcher = [p for p in files if p.endswith("_origin.sh")]
    assert launcher, "запускача немає — секрет нема де тримати"
    assert "sekret-zakhodu" in files[launcher[0]]
    assert any("chmod 600" in cmd for cmd in said), "запускач мусить бути закритий"
    started = [cmd for cmd in said if "setsid" in cmd]
    assert started and "sekret-zakhodu" not in started[0], (
        "секрет потрапив у рядок запуску — його видно в `ps`")


# ---- вибір плеча -----------------------------------------------------------


class _Box:
    """Машина, яка або дає прокинутий порт, або ні."""

    def __init__(self, endpoint: dict[str, Any] | None) -> None:
        self.endpoint = endpoint

    def http_endpoint(self, handle: Any, port: int = 0) -> dict[str, Any] | None:
        return self.endpoint

    def ssh_endpoint(self, handle: Any) -> dict[str, Any]:
        return {"host": "1.2.3.4", "port": 22, "key": "/tmp/k", "user": "root"}


def _sup(tmp_path: Path, endpoint: dict[str, Any] | None):
    from gpurunner.supervise.htr import Supervisor
    from gpurunner.supervise.plan import CasePlan, Plan

    plan = Plan(assets_url="", assets_path=str(tmp_path / "a.tgz"), transport="box",
                cases=[CasePlan(case="sprava", pages_url="", n_pages=2,
                                out_dir=str(tmp_path / "out"),
                                pages_path=str(tmp_path / "p.tar"))],
                budget_usd=1.0, max_hours=1.0)
    return Supervisor(plan, backend=_Box(endpoint), session="S")  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPURUNNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GPURUNNER_BOXES_FILE", str(tmp_path / "boxes.jsonl"))


def test_no_forwarded_port_means_scp_and_says_so(tmp_path: Path) -> None:
    """Відсутність порту — не аварія, але й не тиша.

    `scp` це 7.56 МБ/с проти 20.07 у HTTP: учетверо гірше, тобто на великому
    архіві — зайві хвилини оплаченого простою. Мовчазний перехід на вчетверо
    повільніше плече означав би, що ми не знаємо, чому захід став дорожчим.
    """
    sup = _sup(tmp_path, None)
    assert sup._origin_outside_url(object()) == ""  # type: ignore[arg-type]
    assert any(i.kind == "origin_port_missing" for i in sup.state.incidents)


def test_a_forwarded_but_silent_port_also_means_scp(tmp_path: Path) -> None:
    """🔴 Прокинутий ≠ досяжний, і різницю знає лише запит.

    Здогад тут коштував би найдорожчого: доставка «почалась» і зависла на
    оплачуваній машині, а стеля часу фази з'їлась би на мертвому сокеті.
    """
    sup = _sup(tmp_path, {"host": "127.0.0.1", "port": 1, "inside": 8752})
    assert sup._origin_outside_url(object()) == ""  # type: ignore[arg-type]
    assert any(i.kind == "origin_unreachable" for i in sup.state.incidents)


def test_a_live_store_is_used_for_the_upload(tmp_path: Path) -> None:
    """А коли склад відповідає — веземо в нього, і адреса несе секрет заходу."""
    root = tmp_path / "origin"
    sup = _sup(tmp_path, None)
    httpd = origin.serve(root, port=0, host="127.0.0.1", token=sup._origin_token)
    try:
        sup.backend.endpoint = {"host": "127.0.0.1",          # type: ignore[attr-defined]
                                "port": httpd.server_address[1], "inside": 8752}
        base = sup._origin_outside_url(object())              # type: ignore[arg-type]
        assert base and sup._origin_token in base
        assert any(i.kind == "origin_outside" for i in sup.state.incidents)

        local = tmp_path / "a.tgz"
        local.write_bytes(b"a" * 2048)
        deliver = sup._deliverer(base, None, tmp_path, None)
        assert deliver(local, "assets.tgz", "/remote/assets.tgz") == 2048
        assert (root / "assets.tgz").is_file()
    finally:
        httpd.shutdown()


def test_the_rented_box_is_asked_to_forward_the_store_port() -> None:
    """🔴 Порт просять ПРИ ОРЕНДІ, бо доданий пізніше вимагає нової машини.

    Проситься завжди, навіть коли транспорт заходу — бакет: зайвий прокинутий
    порт не коштує нічого, а його відсутність коштує оренди.
    """
    from gpurunner.backends.vast import DIRECT_PORTS, _instance_payload

    payload = _instance_payload({"num_gpus": 1}, image="i", disk_gb=10, label="l",
                                onstart="true", num_gpus=1, open_ports=DIRECT_PORTS)
    assert payload["env"] == {f"-p {bt.ORIGIN_PORT}:{bt.ORIGIN_PORT}": "1"}
    assert bt.ORIGIN_PORT in DIRECT_PORTS


def test_both_rent_paths_ask_for_the_port() -> None:
    """Оренда буває двох видів (job і гола) — порт потрібен обом."""
    src = (Path(__file__).resolve().parents[1] / "src" / "gpurunner"
           / "backends" / "vast.py").read_text(encoding="utf-8")
    assert src.count("open_ports=DIRECT_PORTS") == 2, (
        "один із двох шляхів оренди лишився без прокинутого порту — і на тій "
        "машині доставка тихо поїде вчетверо повільніше")


# ---- стелі часу від ВИМІРУ, а не від сталої --------------------------------


def test_the_transfer_ceiling_follows_the_measured_channel(tmp_path: Path) -> None:
    """🔴 Стала стеля хибна в обидві сторони.

    `timeout=1800` на швидкому каналі означав пів години на мертвому сокеті —
    пів години оплаченої оренди після того, як усе вже зрозуміло; на повільному
    вона різала живий перенос посередині. Канал — єдиний вхід, що гуляє на два
    з половиною порядки (282 машини реєстру: медіана факт/обіцянка 0.266,
    мінімум 0.004), тож одна стеля для 0.004 і 20 МБ/с не може бути правильною
    для обох.
    """
    from gpurunner.supervise.htr import (
        TRANSFER_CEILING_MAX_SEC,
        TRANSFER_CEILING_MIN_SEC,
    )

    sup = _sup(tmp_path, None)
    big = 600_000_000

    sup._uplink_mbs = 20.0
    fast = sup._transfer_ceiling(big)
    sup._uplink_mbs = 0.5
    slow = sup._transfer_ceiling(big)

    assert slow > fast, "повільний канал мусить діставати більше часу"
    assert TRANSFER_CEILING_MIN_SEC <= fast <= TRANSFER_CEILING_MAX_SEC
    assert slow <= TRANSFER_CEILING_MAX_SEC, "але не безмежно"
    assert sup._transfer_ceiling(1024) == TRANSFER_CEILING_MIN_SEC, (
        "дрібний перенос бере підлогу — моргання мережі не привід падати")


def test_without_a_measurement_the_ceiling_is_pessimistic(tmp_path: Path) -> None:
    """Перший перенос заходу — той самий, яким канал і міряють.

    Вимагати виміру від того, чим його знімають, означало б не мати стелі
    взагалі на найуразливішому кроці.
    """
    sup = _sup(tmp_path, None)
    assert sup._uplink_mbs == 0.0
    assert sup._transfer_ceiling(600_000_000) > sup._transfer_ceiling(1024)


def test_the_measured_uplink_reaches_the_registry() -> None:
    """🔴 Доти замір жив нотаткою сесії й помирав разом із нею.

    Наступний захід про аплінк до тієї самої машини не знав НІЧОГО — при тому,
    що канал гуляє сильніше за будь-який інший вхід. Це окреме число від
    `net_mbps` проби: та міряє шлях ХОСТА до бакета, а це наш канал до цієї
    машини, і їхня різниця й була причиною, з якої повільна доставка виглядала
    як швидка машина.
    """
    from tests.srcprobe import method_body

    src = (Path(__file__).resolve().parents[1] / "src" / "gpurunner"
           / "supervise" / "htr.py").read_text(encoding="utf-8")
    body = method_body(src, "_record_offer")
    assert '"uplink_mbs"' in body and '"uplink_via"' in body, (
        "канал не потрапляє в реєстр машин")
    gate = method_body(src, "_gate_delivery_speed")
    assert "self._uplink_mbs = rate" in gate, "замір нікуди не зберігається"


def test_the_store_gives_up_on_a_client_that_stopped_sending(tmp_path: Path,
                                                             monkeypatch) -> None:
    """🔴 Інакше один такий клієнт тримає наш потік, доки не здасться TCP.

    Клієнт оголосив тіло більше, ніж надіслав (обрив, вбитий процес, повний
    диск на його боці) — і склад чекав решти десятками хвилин на машині, за яку
    платять погодинно, тримаючи при цьому недописаний `.part`. Заміряно на
    власному тесті: очікування 10 с на порожньому місці, і це була найповільніша
    перевірка всього набору.

    Стеля стоїть на ОДНЕ читання, а не на файл, тож на цілий чекпоінт вона не
    впливає.
    """
    import socket

    # 🔴 Спершу — що стеля взагалі Є за замовчуванням. Нижче ми підмінюємо її
    # на секунду, щоб тест не платив хвилинами; без цього рядка підміна ховала
    # б саме ту ваду, від якої стоїть увесь тест (перевірено мутацією).
    default = origin.Handler.timeout
    assert isinstance(default, (int, float)) and 0 < default <= 600, (
        f"склад без стелі читання ({default!r}): клієнт, який перестав слати, "
        f"триматиме потік, доки не здасться TCP")

    monkeypatch.setattr(origin.Handler, "timeout", 1)
    root = tmp_path / "origin2"
    httpd = origin.serve(root, port=0, host="127.0.0.1", token=TOKEN)
    try:
        port = httpd.server_address[1]
        crlf = b"\r\n"
        with socket.create_connection(("127.0.0.1", port), timeout=15) as sock:
            sock.sendall(b"PUT /" + TOKEN.encode() + b"/cases/stuck.tar HTTP/1.1" + crlf
                         + b"Host: 127.0.0.1" + crlf
                         + b"Content-Length: 8192" + crlf + crlf)
            sock.sendall(b"q" * 100)
            # Клієнт БІЛЬШЕ НЕ ШЛЕ нічого й не закриває з'єднання — саме так
            # виглядає обрив, і саме тут склад раніше чекав вічно.
            answer = sock.recv(64)
        assert answer.startswith(b"HTTP/"), answer[:80]
        assert not (root / "cases" / "stuck.tar").exists()
        assert not list(root.rglob("*.part")), "недописане мусить прибиратись"
    finally:
        httpd.shutdown()


# ---- проба міряє так, як качає транспорт -----------------------------------


def test_the_probe_reports_both_one_stream_and_eight() -> None:
    """🔴 Без другого числа рішення про паралельність ухвалюється наосліп.

    Транспорт на боксі ескалює до восьми діапазонних з'єднань нижче 4 МБ/с —
    тобто однопотокова проба міряє НЕ ТЕ, чим потім качають, і занижує саме
    там, де транспорт виграє. Звідси й бралось «раз 10 МБ/с, раз 1.5»: часто це
    той самий канал, зміряний одним потоком і використаний вісьмома.
    """
    from gpurunner.backends.vast import VastBackend

    probe = VastBackend.parse_probe(
        "net_bps=2000000\nnet_http=206\nnet_rc=0\nnet_par_bps=16000000\n")

    assert probe["net_mbps"] == pytest.approx(16.0)
    assert probe["net_par_mbps"] == pytest.approx(128.0)


def test_an_old_probe_without_the_parallel_number_is_not_a_zero_channel() -> None:
    """Стара проба (або бокс без `date +%s%N`) мусить читатись як «не міряли».

    Нуль тут прочитався б як «паралельно канал мертвий» — і ворота відмовили б
    здоровій машині за вимір, якого не робили.
    """
    from gpurunner.supervise.gate import _measured

    view = _measured({"net_bps": 2_000_000, "net_http": "206"})
    assert view["net_par_mbps"] == 0
    assert view["net_measured"] is True


def test_the_download_gate_counts_with_the_channel_it_will_actually_use(
        ) -> None:
    """🔴 Міряти однопотоково, а платити багатопотоково — це відмова машині,
    яка встигає.

    Виміряно 22.09.2026 (spr-160, RTX A4000×4, Японія): офер обіцяв 1821
    Мбіт/с, проба дала 21.6. Ворота на 2.25 ГБ рахують від того числа, яким
    архів справді поїде.
    """
    from gpurunner.supervise.gate import _measured

    slow = _measured({"net_bps": 2_700_000, "net_http": "206"})
    fast = _measured({"net_bps": 2_700_000, "net_http": "206",
                           "net_par_bps": 21_600_000})
    assert fast["net_par_mbps"] > slow["net_mbps"] * 4

    src = (Path(__file__).resolve().parents[1] / "src" / "gpurunner"
           / "supervise" / "gate.py").read_text(encoding="utf-8")
    i = src.index("need.max_archive_mb > 0")
    window = src[i:i + 900]
    assert "net_par_mbps" in window, (
        "час качання архіву рахується однопотоковим числом, хоч качають вісьмома")


def test_the_parallel_probe_moves_the_same_volume_as_the_single_one() -> None:
    """Порівнюються два СПОСОБИ, а не два розміри.

    Інакше відношення чисел нічого не каже: більший обсяг сам собою дає інакшу
    швидкість, і «паралельність допомагає» стало б нефальсифіковним.
    """
    from gpurunner.backends.vast import _PROBE_SCRIPT

    assert "-r 0-33554432" in _PROBE_SCRIPT, "однопотокова проба — 32 МБ"
    assert "i -lt 8" in _PROBE_SCRIPT, "паралельна — вісім діапазонів"
    assert "b=$((i*4194304))" in _PROBE_SCRIPT, "по 4 МБ = ті самі 32 МБ"
    assert "33554432*1000000000/D" in _PROBE_SCRIPT, (
        "швидкість рахується з того самого обсягу")


# ---- рятунок у режимі box --------------------------------------------------


def test_the_rescue_can_reach_the_store_when_the_supervisor_died(
        tmp_path: Path) -> None:
    """🔴 Доти в режимі `box` рятунку не було ВЗАГАЛІ.

    У бакеті на цей випадок лежать присигновані посилання, записані в ПЛАН; у
    режимі `box` план про машину знати нема звідки — її на той момент ще не
    існувало, тож `resume_urls` там порожні за побудовою. Помер наглядач після
    читання й до забору — і прорахований текст діставати нічим, хоч він лежить
    на живій машині.

    Тепер адресу складу (разом із секретом заходу) пише в стан той, хто її
    дізнався, і забір будує з неї ті самі посилання.
    """
    import json

    from gpurunner.htr.fetch_ckpt import cases_from_plan

    plan = {"cases": [{"case": "sprava", "n_pages": 10, "out_dir": str(tmp_path),
                       "ckpt_prefix": "ckpt/sprava/model_v1.pt", "ckpt_slots": 3,
                       "resume_urls": []}]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    blind = cases_from_plan(path)
    assert blind[0]["resume_urls"] == [], "без адреси брати нема звідки — це чесно"

    base = "http://203.0.113.7:41000/sekret-zakhodu"
    seeing = cases_from_plan(path, origin_base=base)
    urls = seeing[0]["resume_urls"]
    assert len(urls) == 3
    assert urls[0] == f"{base}/ckpt/sprava/model_v1.pt/ckpt_0001.tgz"


def test_the_checkpoint_names_have_one_source_of_truth() -> None:
    """🔴 Два списки імен, складені окремо, розійшлись би ТИХО.

    Помітно це стало б рівно в той день, коли точка відновлення знадобилась —
    тобто коли перевіряти вже нічим.
    """
    names = bt.ckpt_names("ckpt/sprava/m.pt", 2)
    urls = bt.urls_for("sprava", "ckpt/sprava/m.pt", 2, token="tok")
    assert names == ["ckpt/sprava/m.pt/ckpt_0001.tgz", "ckpt/sprava/m.pt/ckpt_0002.tgz"]
    assert [u.split("/tok/", 1)[1] for u in urls["ckpt_urls"]] == names


def test_the_supervisor_writes_the_store_address_into_its_state(
        tmp_path: Path) -> None:
    """Нотатка — для людини, поле стану — для машини.

    Рятунок читає стан, а не журнал: розбирати текст нотатки означало б
    залежати від її формулювання.
    """
    from gpurunner.supervise.state import SupervisorState

    assert "origin_base" in {f.name for f in __import__("dataclasses").fields(
        SupervisorState)}

    from tests.srcprobe import method_body

    src = (Path(__file__).resolve().parents[1] / "src" / "gpurunner"
           / "supervise" / "htr.py").read_text(encoding="utf-8")
    body = method_body(src, "_origin_outside_url")
    assert "self.state.origin_base = base" in body
    assert "self.state.save()" in body, (
        "поле, яке не збережено, не переживе смерті наглядача — а саме для неї "
        "воно й потрібне")


# ---- вісім потоків замість одного ------------------------------------------


def test_a_big_file_travels_in_parallel_and_is_assembled_whole(
        store, tmp_path: Path) -> None:
    """🔴 Одне з'єднання не вибирає каналу на великому колі.

    Заміряно 23.09.2026: домашній канал одним потоком 3.83 МБ/с, вісьмома 8.94
    (×2.3). А перша жива доставка 135 МБ у Корею одним потоком дала 1.2 МБ/с
    при каналі 72 Мбіт/с — усемеро менше за можливе. Протокол тут ні до чого:
    поки летить підтвердження, єдине вікно TCP мовчить.

    Приймач — БАЙТИ: шматки мусять зійтись у той самий файл, а не «десь там
    лежати». Помилка складання виглядала б як битий архів уже на оплаченій
    машині.
    """
    import os

    base, root = store
    local = tmp_path / "assets.tgz"
    body = os.urandom(9 << 20)          # більший за поріг паралельності
    local.write_bytes(body)
    joined: list[tuple[str, int]] = []

    def assemble(rel: str, names: list[str]) -> None:
        # Те саме, що `cat` робить на машині, тільки тут — локально.
        joined.append((rel, len(names)))
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            for name in names:
                piece = root / name
                out.write(piece.read_bytes())
                piece.unlink()

    sent = bt.push_http_parallel(base, local, "assets.tgz", assemble=assemble)

    assert sent == len(body)
    assert joined == [("assets.tgz", bt.PUSH_PARTS)], "везли не вісьмома"
    assert (root / "assets.tgz").read_bytes() == body, "байти не зійшлись"
    assert not list(root.glob("*.p0*")), "шматки мусять прибиратись"


def test_a_small_file_is_not_split_into_eight(store, tmp_path: Path) -> None:
    """Дрібне не варте восьми рукостискань — на ньому виграшу немає."""
    base, root = store
    local = tmp_path / "small.tar"
    local.write_bytes(b"s" * 4096)
    called: list[str] = []

    bt.push_http_parallel(base, local, "cases/small.tar",
                          assemble=lambda rel, names: called.append(rel))

    assert called == [], "дрібний файл поїхав шматками — це зайві оберти"
    assert (root / "cases" / "small.tar").is_file()


def test_a_lost_piece_is_a_refusal_not_a_broken_archive(
        store, tmp_path: Path) -> None:
    """🔴 Склейка, у якій бракує шматка, дає файл, що ВИГЛЯДАЄ цілим.

    Виявилось би це вже на оплачуваній машині — тим, що раннер не може
    розпакувати архів. Тому після склейки розмір звіряється тим самим каналом.
    """
    import os

    base, root = store
    local = tmp_path / "assets.tgz"
    local.write_bytes(os.urandom(9 << 20))

    def half(rel: str, names: list[str]) -> None:
        target = root / rel
        with target.open("wb") as out:
            for name in names[:-1]:            # один шматок «загубився»
                out.write((root / name).read_bytes())

    with pytest.raises(BackendError, match="не зійшлись"):
        bt.push_http_parallel(base, local, "assets.tgz", assemble=half)


def test_the_supervisor_delivers_in_parallel_and_joins_on_the_box() -> None:
    """Склейка — ОДНА команда на файл, і шматки після неї прибираються.

    Команда на машині коштує рукостискання, тож вісім заливок мусять давати
    одну команду, а не вісім.
    """
    from tests.srcprobe import method_body

    src = (Path(__file__).resolve().parents[1] / "src" / "gpurunner"
           / "supervise" / "htr.py").read_text(encoding="utf-8")
    body = method_body(src, "_deliverer")
    assert "push_http_parallel" in body, "доставка знову одним потоком"
    assert "cat {pieces} > {target} && rm -f {pieces}" in body
    assert "shlex.quote" in body, "імена йдуть у шел — без лапок це чужа команда"
