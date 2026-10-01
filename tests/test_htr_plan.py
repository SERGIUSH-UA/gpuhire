"""Складач плану: тека виводу, імена справ, геометрія, інкрементальність.

🔴 Головне тут — `out_dir`. Складач жив скриптом у репозиторії одного
дослідження й рахував теку виводу від СВОГО кореня, тож справи з інших
просторів розкладались у той репозиторій. За одну
кампанію це спрацювало п'ять разів; двічі виглядало як «робота втрачена», хоч
на диску лежало 1101 і 441 готова сторінка — просто в чужому проєкті.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpurunner.htr.plan_build import (
    BuildOptions,
    assert_unique_slugs,
    build_plan,
    case_slug,
    ckpt_urls_for,
    frames_of,
)


class FakeS3:
    """Мінімальний R2: пам'ятає, що залито, і роздає передбачувані посилання."""

    def __init__(self, *, fail_on: str = "") -> None:
        self.uploaded: list[str] = []
        self.fail_on = fail_on

    def upload_file(self, path: str, bucket: str, key: str, **kw: Any) -> None:
        if self.fail_on and self.fail_on in key:
            raise RuntimeError("ServiceUnavailable: Reduce your concurrent request rate")
        self.uploaded.append(key)

    def generate_presigned_url(self, op: str, *, Params: dict, ExpiresIn: int) -> str:
        return f"https://r2/{Params['Key']}?op={op}&exp={ExpiresIn}"


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    """Тека справи з трьома кадрами."""
    directory = tmp_path / "frames" / "spr-1739"
    directory.mkdir(parents=True)
    for i in range(3):
        (directory / f"{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xe0jpeg")
    return directory


@pytest.fixture
def opts(tmp_path: Path) -> BuildOptions:
    return BuildOptions(
        out_root=(tmp_path / "prostir" / "reports" / "htr").resolve(),
        model="pysar_cyr_v17.pt", voices="diak_cyr_v4.mlmodel")


def _build(case_dirs, opts, monkeypatch, s3: FakeS3, **kw):
    from gpurunner.htr import plan_build, r2

    monkeypatch.setattr(r2, "client", lambda env=None: s3)
    monkeypatch.setattr(plan_build.r2, "client", lambda env=None: s3)
    return build_plan(case_dirs, opts, assets_key="assets/a.tgz",
                      log=lambda _line: None, **kw)


def test_out_dir_is_absolute_and_points_where_told(case_dir, opts, monkeypatch) -> None:
    """🔴🔴 Тека справи береться з `out_root`, а не від кореня складача."""
    plan = _build([case_dir], opts, monkeypatch, FakeS3())
    out_dir = Path(plan["cases"][0]["out_dir"])
    assert out_dir.is_absolute()
    assert out_dir == opts.out_root / "spr-1739"


def test_a_relative_out_root_is_refused(case_dir, tmp_path, monkeypatch) -> None:
    """Відносний шлях розкладеться від теки ЗАПУСКУ наглядача, а це майже
    завжди інший простір, ніж той, якому належить справа."""
    opts = BuildOptions(out_root=Path("reports/htr"), model="m.pt")
    with pytest.raises(ValueError, match="АБСОЛЮТНИМ"):
        _build([case_dir], opts, monkeypatch, FakeS3())


def test_cyrillic_volume_letters_survive_the_slug() -> None:
    """🔴 Доти кожна кирилична літера ставала дефісом, і `230-1-2`, `230-1-2а`,
    `230-1-2б` (три РІЗНІ томи іменних списків дворян 1844) давали ОДИН слуг:
    спільний префікс у R2 і спільну теку виводу. Три декоди злились би в один,
    і це не впало б — просто дало б чужий текст під правильним шифром."""
    slugs = {case_slug(Path(f"D:/raw/230-1-2{suffix}")) for suffix in ("", "а", "б")}
    assert len(slugs) == 3


def test_two_cases_with_the_same_name_are_refused_before_any_upload() -> None:
    with pytest.raises(ValueError, match="однакове ім'я"):
        assert_unique_slugs([Path("D:/a/spr-1"), Path("D:/b/spr-1")])


def test_frames_are_counted_flat_like_the_runner_sees_them(case_dir) -> None:
    """Знаменник мусить збігатися з тим, що прогін прочитає: раннер бере
    `iterdir()`, тож вкладена тека до рахунку не входить."""
    nested = case_dir / "thumbs"
    nested.mkdir()
    (nested / "x.jpg").write_bytes(b"\xff\xd8")
    assert len(frames_of(case_dir)) == 3


def test_checkpoint_urls_cover_the_whole_time_ceiling() -> None:
    """🔴 При чекпоінті раз на 120 с шістдесят посилань вичерпуються за 2 год —
    на 8-годинному заході останні шість годин не мали точки відновлення."""
    # Ніколи менше підлоги, і завжди з запасом на всю стелю часу.
    assert ckpt_urls_for(0.5) == 60
    assert ckpt_urls_for(2.0) >= 2 * 3600 / 120
    assert ckpt_urls_for(8.0) >= 8 * 3600 / 120


def test_the_plan_is_written_after_every_case_not_only_at_the_end(
        tmp_path, opts, monkeypatch) -> None:
    """🔴 На черзі з 22 справ R2 відмовив на останній, і план не створився
    ВЗАГАЛІ — файл лишився нульовим, тобто втратилась і вже залита частина."""
    dirs = []
    for name in ("spr-1", "spr-2", "spr-3"):
        directory = tmp_path / "frames" / name
        directory.mkdir(parents=True)
        (directory / "0001.jpg").write_bytes(b"\xff\xd8")
        dirs.append(directory)

    out = tmp_path / "plan.json"
    with pytest.raises(RuntimeError, match="ServiceUnavailable"):
        _build(dirs, opts, monkeypatch, FakeS3(fail_on="spr-3"), out_path=out)

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert [c["case"] for c in saved["cases"]] == ["spr-1", "spr-2"]


def test_case_keys_must_be_one_per_case_or_none(case_dir, opts, monkeypatch) -> None:
    """Порядок — єдине, що зв'язує ключ зі справою; часткового набору бути не
    може, бо «здогадаюсь, до якої з трьох це» кладе декод у чужу книгу."""
    with pytest.raises(ValueError, match="по одній"):
        _build([case_dir], opts, monkeypatch, FakeS3(),
               case_keys=["DAVO/337/4", "DAVO/337/5"])


def test_checkpoint_prefix_carries_the_model(case_dir, opts, monkeypatch) -> None:
    """🔴 Без моделі в ключі прогін новою моделлю підхоплював чужі тексти:
    раннер бачить сторінку в стані як зроблену й пропускає її. Платимо за v17,
    отримуємо v16 — і підсумок чесно каже «модель v17, complete: true»."""
    plan = _build([case_dir], opts, monkeypatch, FakeS3())
    assert "pysar_cyr_v17.pt" in plan["cases"][0]["ckpt_urls"][0]


def test_both_voices_reach_the_plan(case_dir, opts, monkeypatch) -> None:
    """🔴 `voices` дефолтиться в порожній рядок, тож гілка другого голосу просто
    не запускалась: 2578 сторінок пройшли одним голосом (11.08.2026)."""
    plan = _build([case_dir], opts, monkeypatch, FakeS3())
    assert plan["params"]["voices"] == "diak_cyr_v4.mlmodel"


def test_link_lifetime_is_raised_to_outlive_the_run(case_dir, opts, monkeypatch) -> None:
    """Протермінований GET віддає 403, а раннер читає це як «чекпоінта немає» і
    стартує з нуля — відновлення перетворюється на повний повторний прогін."""
    opts.max_hours = 12.0
    opts.url_hours = 3.0
    _build([case_dir], opts, monkeypatch, FakeS3())
    assert opts.url_hours >= 12.0 * 1.5


def test_frame_geometry_lands_in_the_plan(case_dir, opts, monkeypatch) -> None:
    """Площа кадру — вхід до розрахунку VRAM на шард, тож вона мусить доїхати
    до наглядача разом із планом."""
    from gpurunner.htr import plan_build
    from gpurunner.htr.plan_build import Geometry

    monkeypatch.setattr(plan_build, "measure_frames",
                        lambda frames, sample=60: Geometry(7.1, 8.0, 1.26, len(frames)))
    plan = _build([case_dir], opts, monkeypatch, FakeS3())
    assert plan["cases"][0]["frame_mpx_median"] == 7.1
    assert plan["cases"][0]["frame_aspect_median"] == 1.26


# ---- спіймано сухим прогоном на живих даних ЦДІАК ф.127 (04.09.2026) --------


def test_a_pages_subfolder_does_not_become_the_case_name() -> None:
    """🔴🔴 Кадри часто лежать у `<справа>/pages`, і саме цю теку подають на
    вхід — рахунок кадрів не рекурсивний за задумом. Без окремого правила дві
    РІЗНІ справи давали один слуг `pages`: спільний префікс у R2 і спільна тека
    виводу, тобто два декоди зливаються в один без жодної помилки."""
    assert case_slug(Path("D:/raw/cdiak_127/spr-1649/pages")) == "spr-1649"
    assert case_slug(Path("D:/raw/cdiak_127/spr-1655/frames")) == "spr-1655"


def test_service_names_are_only_stripped_when_they_are_the_leaf() -> None:
    """Справа, яка САМА зветься `pages`, лишається собою: правило про підтеку,
    а не про заборонене слово."""
    assert case_slug(Path("D:/raw/pages")) == "raw"
    assert case_slug(Path("D:/raw/spr-1/pages_dl_07")) == "pages_dl_07"


def test_two_pages_subfolders_are_caught_as_a_collision() -> None:
    with pytest.raises(ValueError, match="однакове ім'я"):
        assert_unique_slugs([Path("D:/raw/spr-1/pages"), Path("D:/raw/spr-1/frames")])


def test_unmeasured_geometry_says_so_out_loud(case_dir, monkeypatch, capsys) -> None:
    """🔴 Мовчазний нуль читався б як «кадри звичайні»: геометрія визначає VRAM
    на шард, і «не міряли» — це інший стан, ніж «сторінка»."""
    import builtins

    from gpurunner.htr.plan_build import measure_frames as measure

    real_import = builtins.__import__

    def no_pil(name, *args, **kw):
        if name == "PIL":
            raise ImportError("no PIL")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", no_pil)
    geometry = measure(frames_of(case_dir))
    monkeypatch.undo()

    assert geometry.mpx_median == 0.0
    assert "НЕ ЗМІРЯНО" in capsys.readouterr().err


def test_lines_from_a_previous_run_land_in_the_plan(case_dir, opts, monkeypatch) -> None:
    """spr-8248: 112 рядків/стор і 247 стор/год проти 4594 на 230-1-24 із 35
    рядками — той самий бокс, та сама модель; план про рядки не знав. Тепер
    складач читає мету ПОПЕРЕДНЬОГО прогону тієї самої справи, коли вона є."""
    plan = _build([case_dir], opts, monkeypatch, FakeS3())
    assert plan["cases"][0]["lines_per_page_median"] == 0.0

    out_dir = Path(plan["cases"][0]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_htr_meta.json").write_text(json.dumps({
        "version": 1, "pages": {f"{i:04d}.jpg": {"lines": n}
                                for i, n in enumerate([110, 112, 115], 1)}}),
        encoding="utf-8")
    plan = _build([case_dir], opts, monkeypatch, FakeS3())
    assert plan["cases"][0]["lines_per_page_median"] == 112


# ---- перечитування іншою моделлю -------------------------------------------------


class ListingS3(FakeS3):
    """R2 із переліком: засів кешу дивиться, чи прогін уже має чекпоінти."""

    def __init__(self, existing: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.existing = list(existing)
        self.members: dict[str, list[str]] = {}

    def upload_file(self, path: str, bucket: str, key: str, **kw: Any) -> None:
        import tarfile

        super().upload_file(path, bucket, key, **kw)
        if key.endswith(".tgz"):
            with tarfile.open(path) as tf:
                self.members[key] = tf.getnames()

    def get_paginator(self, op: str) -> Any:
        s3 = self

        class _Pages:
            def paginate(self, **kw: Any):
                prefix = kw.get("Prefix", "")
                keys = [k for k in s3.existing + s3.uploaded if k.startswith(prefix)]
                yield {"Contents": [{"Key": k, "Size": 1} for k in keys]}

        return _Pages()


SKRYBA = "skryba_f792_v6.mlmodel"


def _skryba(opts: BuildOptions, **kw) -> BuildOptions:
    import dataclasses

    return dataclasses.replace(opts, model=SKRYBA, voices="", **kw)


def _seg_dir(tmp_path: Path) -> Path:
    seg = tmp_path / "reports" / "spr-1739" / "data" / "derived" / "htr_seg" / "pages_dl_01__085f51d2"
    seg.mkdir(parents=True)
    for i in range(3):
        (seg / f"{i:04d}.o0.c400.seg.json.gz").write_bytes(b"gz")
    (seg / "note.txt").write_text("не кеш", encoding="utf-8")
    return seg


def test_a_rerun_with_another_model_gets_its_own_name(case_dir, opts, monkeypatch) -> None:
    """🔴 Тексти Скриби не мають лягти в теку Писаря: забір не перезаписує `.txt`."""
    plan = _build([case_dir], _skryba(opts, names=["spr-1739-skryba_v6"]), monkeypatch,
                  ListingS3())
    entry = plan["cases"][0]
    assert entry["case"] == "spr-1739-skryba_v6"
    assert entry["out_dir"].endswith("spr-1739-skryba_v6")
    assert "ckpt/spr-1739-skryba_v6/skryba_f792_v6.mlmodel/ckpt_0001.tgz" in entry["resume_urls"][0]


def test_a_renamed_run_uploads_the_frames_under_the_key_its_link_points_to(
        case_dir, opts, monkeypatch) -> None:
    """🔴 Архів кадрів заливався за ім'ям теки (`spr-1739.tar`), а `pages_url` вела на
    `spr-1739-skryba_v6.tar` — передполіт 404 на першому ж перечитуванні справи,
    якої під новим ім'ям у R2 ще не було (ДАЖО 178-53-36, 15.09.2026)."""
    s3 = ListingS3()
    plan = _build([case_dir], _skryba(opts, names=["spr-1739-skryba_v6"]), monkeypatch, s3)
    url_key = plan["cases"][0]["pages_url"].split("?")[0].split("/", 3)[-1]
    assert url_key == "cases/spr-1739-skryba_v6.tar"
    assert url_key in s3.uploaded


def test_the_segmentation_is_seeded_as_the_first_checkpoint(case_dir, opts, tmp_path,
                                                           monkeypatch) -> None:
    from gpurunner._embedded import htr_case_runner as runner
    from gpurunner.htr.plan_build import SEG_CACHE_ARC

    s3 = ListingS3()
    _build([case_dir], _skryba(opts, names=["spr-1739-skryba_v6"],
                               seed_seg=[str(_seg_dir(tmp_path))]), monkeypatch, s3)
    key = "ckpt/spr-1739-skryba_v6/skryba_f792_v6.mlmodel/ckpt_0001.tgz"
    assert key in s3.members
    assert sorted(s3.members[key]) == [f"{SEG_CACHE_ARC}/{i:04d}.o0.c400.seg.json.gz"
                                       for i in range(3)]
    # бокс шукає кеш рівно там, куди його засіяно
    assert runner.SEG_CACHE_ARC == SEG_CACHE_ARC


def test_seeding_never_overwrites_an_existing_checkpoint_series(case_dir, opts, tmp_path,
                                                               monkeypatch) -> None:
    """Префікс уже має чекпоінти — це відновлення; `ckpt_0001` поверх знищив би базу."""
    prefix = "ckpt/spr-1739-skryba_v6/skryba_f792_v6.mlmodel/"
    s3 = ListingS3(existing=(prefix + "ckpt_0001.tgz",))
    _build([case_dir], _skryba(opts, names=["spr-1739-skryba_v6"],
                               seed_seg=[str(_seg_dir(tmp_path))]), monkeypatch, s3)
    assert not [k for k in s3.uploaded if k.startswith(prefix)]


def test_names_and_seeds_must_be_one_per_case(case_dir, opts, monkeypatch) -> None:
    with pytest.raises(ValueError, match="по одному"):
        _build([case_dir], _skryba(opts, names=["а", "б"]), monkeypatch, ListingS3())


def test_the_plan_knows_how_much_the_box_will_download(case_dir, opts, monkeypatch) -> None:
    """Ворота звіряють канал з обсягом — без нього межа лишається сліпою сталою."""
    plan = _build([case_dir], opts, monkeypatch, FakeS3())
    assert plan["cases"][0]["pages_bytes"] == sum(f.stat().st_size for f in frames_of(case_dir))

# ---- заливка, якої не треба ------------------------------------------------------


class SeededS3(FakeS3):
    """R2, у якому ВЖЕ щось лежить: відповідає на `head_object`.

    Саме таким бакет і застає кожен другий захід — ключ архіву справи сталий
    (`cases/<слуг>.tar`), тож кадри першої заливки лежать там і далі.
    """

    def __init__(self, have: dict[str, tuple[int, float]]) -> None:
        super().__init__()
        #: {ключ: (розмір, unix-час об'єкта)}
        self.have = dict(have)
        self.asked: list[str] = []

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        from datetime import UTC, datetime

        self.asked.append(Key)
        if Key not in self.have:
            raise RuntimeError("404 NoSuchKey")
        size, when = self.have[Key]
        return {"ContentLength": size,
                "LastModified": datetime.fromtimestamp(when, tz=UTC)}


def _case_tar_size(case_dir: Path) -> int:
    from gpurunner.htr.plan_build import expected_tar_size

    return expected_tar_size(frames_of(case_dir))


def _later_than(case_dir: Path) -> float:
    return max(f.stat().st_mtime for f in frames_of(case_dir)) + 60


def test_the_predicted_archive_size_is_exact(tmp_path: Path) -> None:
    """🔴 Приймач усієї перевірки: передбачене й спаковане мусять зійтись ДО БАЙТА.

    На цьому числі стоїть рішення «заливати чи ні», тож «приблизно» тут гірше
    за ніщо: зайві байти дали б вічну перезаливку (і перевірка виглядала б
    зробленою, не працюючи), а бракуючі — захід на чужих кадрах.

    Імена навмисно різні: коротке ASCII йде звичайним заголовком, а довге
    кирилицею тягне за собою розширений заголовок PAX. Арифметикою «512 плюс
    тіло» друге не порахувати — тому розмір складає той самий `tarfile`.
    """
    from gpurunner.htr.plan_build import expected_tar_size, pack

    case = tmp_path / "spr-99"
    case.mkdir()
    (case / "0001.jpg").write_bytes(b"\xff\xd8" + b"a" * 1000)
    (case / "0002.jpg").write_bytes(b"\xff\xd8" + b"b" * 511)
    long_cyr = "справа-230-1-2а-розворот-" + "у" * 60 + ".jpg"
    (case / long_cyr).write_bytes(b"\xff\xd8" + b"c" * 4097)

    frames = frames_of(case)
    assert len(frames) == 3
    dest = tmp_path / "out"
    dest.mkdir()
    archive = pack(case, frames, dest, "spr-99")
    assert expected_tar_size(frames) == archive.stat().st_size


def test_frames_already_in_the_bucket_are_not_uploaded_again(
        case_dir, opts, monkeypatch) -> None:
    """🔴🔴 24 ГБ і 10-15 хвилин домашнього аплінка перед КОЖНИМ заходом.

    Ключ архіву сталий, тож кадри першої заливки нікуди не зникли — а складач
    пакував і заливав їх заново щоразу, і сам прогін при цьому тривав хвилини.
    Прапорець `--skip-upload` для цього був, але дію мусила ухвалити людина,
    яка не може знати, чи об'єкт на місці. Тепер відповідає бакет.
    """
    key = "cases/spr-1739.tar"
    s3 = SeededS3({key: (_case_tar_size(case_dir), _later_than(case_dir))})
    plan = _build([case_dir], opts, monkeypatch, s3)

    assert key not in s3.uploaded, "кадри залито вдруге, хоч вони вже в бакеті"
    assert key in s3.asked, "бакет навіть не спитали"
    # 🔴 Посилання мусить лишитись: не заливати — не означає не давати адреси.
    assert plan["cases"][0]["pages_url"].startswith("https://r2/" + key)


def test_a_different_size_in_the_bucket_means_a_fresh_upload(
        case_dir, opts, monkeypatch) -> None:
    """Кадри додали чи перезняли — значить у бакеті вже НЕ ця справа.

    Мовчки поїхати на старому архіві означало б платити за прогін, у якому
    частини сторінок на машині просто немає, — а знаменник повноти складач
    рахує з локальних кадрів, тобто захід виглядав би недочитаним без причини.
    """
    key = "cases/spr-1739.tar"
    s3 = SeededS3({key: (_case_tar_size(case_dir) - 512, _later_than(case_dir))})
    _build([case_dir], opts, monkeypatch, s3)
    assert key in s3.uploaded


def test_a_frame_newer_than_the_object_means_a_fresh_upload(
        case_dir, opts, monkeypatch) -> None:
    """Розмір зійшовся, але кадр правили ПІСЛЯ заливки — заливаємо.

    Один збіг розміру на перезнятій сторінці малоймовірний, та ціна його —
    захід із чужим кадром під правильним номером, і побачити це можна лише
    оком на декоді. Дата об'єкта закриває цей випадок безплатно.
    """
    key = "cases/spr-1739.tar"
    stale = min(f.stat().st_mtime for f in frames_of(case_dir)) - 3600
    s3 = SeededS3({key: (_case_tar_size(case_dir), stale)})
    _build([case_dir], opts, monkeypatch, s3)
    assert key in s3.uploaded


def test_the_assets_archive_is_not_re_uploaded_either(
        case_dir, opts, tmp_path, monkeypatch) -> None:
    """Ваги — другий за розміром шматок аплінка після кадрів.

    🔴 Але пропуск тут вирішує САМЕ БАКЕТ, а не `--skip-upload`: 06.09.2026
    план зі свіжими ассетами й тим прапорцем поїхав на бокс із посиланням у
    порожнечу — 404 на пробі каналу й змарнована оренда.
    """
    assets = tmp_path / "assets" / "pysar_cyr_v17.tgz"
    assets.parent.mkdir(parents=True)
    import tarfile as _tar

    weights = tmp_path / "pysar_cyr_v17.pt"
    weights.write_bytes(b"w" * 4096)
    with _tar.open(assets, "w:gz") as tf:
        tf.add(weights, arcname=weights.name)
    key = f"assets/{assets.name}"
    s3 = SeededS3({key: (assets.stat().st_size, assets.stat().st_mtime + 60)})

    plan = _build([case_dir], opts, monkeypatch, s3, assets=assets)
    assert key not in s3.uploaded
    assert plan["assets_url"].startswith("https://r2/" + key)


def test_an_unanswered_bucket_means_upload_not_hope(
        case_dir, opts, monkeypatch) -> None:
    """🔴 Невизначеність відповідає в бік ЗАЛИВКИ.

    Бакет може не відповісти з десятка причин (мережа, права, збій боку R2), і
    жодна з них не є доказом, що об'єкт на місці. Хибне «є» коштує цілого
    заходу на посиланні в порожнечу, хибне «немає» — однієї заливки.
    """
    class MuteS3(FakeS3):
        def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
            raise RuntimeError("AccessDenied")

    s3 = MuteS3()
    _build([case_dir], opts, monkeypatch, s3)
    assert "cases/spr-1739.tar" in s3.uploaded


def test_a_queue_of_small_cases_does_not_inflate_the_job_source(
        tmp_path: Path, opts, monkeypatch) -> None:
    """🔴🔴 25 хвилин оренди за один файл.

    Слоти чекпоінтів рахувались від стелі часу ЗАХОДУ (14 год ⇒ 548) і так — на
    КОЖНУ справу. На черзі з 194 справ це 121 832 presigned-посилання, вшитих у
    `job.py`: 48 МБ джерела, яке потім їде на машину paramiko-SFTP синхронними
    шматками по 32 КБ. Заміряно 23.09.2026: підйом боксу 1644 с, і в лозі це
    виглядало як тиша.

    Міра мусить бути власна — РОЗМІР СПРАВИ: 24 сторінки читаються хвилини й
    потребують двох-трьох точок, а не трьохсот.
    """
    from gpurunner.htr.plan_build import CKPT_URLS_MIN_PER_CASE, ckpt_urls_for

    # Одна справа на весь захід — стеля часу лишається мірою, як було.
    assert ckpt_urls_for(14.0) > 300

    small = ckpt_urls_for(14.0, pages=24)
    assert small == CKPT_URLS_MIN_PER_CASE, small
    assert ckpt_urls_for(14.0, pages=5000) > small, "велика справа дістає більше"
    assert ckpt_urls_for(14.0, pages=5000) <= ckpt_urls_for(14.0), (
        "але не більше за часову стелю — вона межа зверху")

    # І це число справді доходить до плану.
    case = tmp_path / "frames" / "spr-mini"
    case.mkdir(parents=True)
    for i in range(3):
        (case / f"{i:04d}.jpg").write_bytes(b"\xff\xd8jpeg")
    plan = _build([case], opts, monkeypatch, FakeS3())
    entry = plan["cases"][0]
    assert entry["ckpt_slots"] == CKPT_URLS_MIN_PER_CASE
    assert len(entry["ckpt_urls"]) == CKPT_URLS_MIN_PER_CASE


def test_frames_are_uploaded_under_the_key_the_plan_points_to(
        case_dir, opts, monkeypatch) -> None:
    """Ключ плану несе шифру (`cases/<шифра>/<справа>.tar`); заливка мусить
    класти кадри саме туди, інакше бокс дістане 404 на `pages_url`."""
    s3 = FakeS3()
    plan = _build([case_dir], opts, monkeypatch, s3, case_keys=["CDIAK/2/160"])
    url = plan["cases"][0]["pages_url"]
    uploaded = [k for k in s3.uploaded if k.startswith("cases/")]
    assert uploaded, "кадри не залито"
    assert all(f"/{k}?" in url for k in uploaded), (uploaded, url)
    assert "CDIAK-2-160" in uploaded[0]


def test_the_plan_has_a_heartbeat_the_box_writes_and_home_reads(
        case_dir, opts, monkeypatch) -> None:
    plan = _build([case_dir], opts, monkeypatch, FakeS3())
    assert plan["plan_id"]
    assert "heartbeat/" in plan["heartbeat_put_url"] and "put_object" in plan["heartbeat_put_url"]
    assert "heartbeat/" in plan["heartbeat_url"] and "get_object" in plan["heartbeat_url"]
