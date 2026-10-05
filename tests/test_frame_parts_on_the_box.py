"""🧩 Архів кадрів частинами на СПРАВЖНЬОМУ коді раннера з фейковим шардом.

30.09.2026, ф.315: справа їхала одним tar на 4–5 ГБ, і після пересадки нова
машина тягла його цілком — 14–18 хв простою, хоч би лишилось дочитати п'яту
частину. Тут перевіряється, що частину, прочитану попередньою орендою, бокс не
качає, а знаменник справи при цьому лишається повним.

Раннер живе на Linux-шляхах (`/tmp/htrcase`, групи процесів), тож тест — для
Linux (CI; локально — WSL).
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.skipif(sys.platform == "win32",
                                 reason="бокс-раннер живе на Linux-шляхах"),
              pytest.mark.usefixtures("one_box_at_a_time")]

EMB = Path(__file__).resolve().parents[1] / "src" / "gpurunner" / "_embedded"
FAKE = Path(__file__).resolve().parent / "data" / "fake_case_shard.py"

STEMS = [f"{i:04d}" for i in range(1, 13)]
PARTS = [STEMS[0:4], STEMS[4:8], STEMS[8:12]]


def _runner(root: Path) -> dict:
    src = (EMB / "_common.py").read_text(encoding="utf-8") + "\n" + (
        EMB / "htr_case_runner.py").read_text(encoding="utf-8").replace(
        'if __name__ == "__main__":\n    main({})', "")
    ns: dict = {"__name__": "runner_under_test"}
    exec(compile(src, "htr_case_runner.py", "exec"), ns)
    ns["KAGGLE_INPUT"] = root / "input"
    ns["KAGGLE_WORKING"] = root / "working"
    ns["PROGRESS_PATH"] = root / "_progress.json"
    ns["_ensure_deps"] = lambda *a, **k: None
    ns["_free_vram_per_card"] = lambda: [24.0]
    ns["_gpu_name"] = lambda: "fake"
    ns["_usable_cores"] = lambda: 16
    (root / "input" / "models").mkdir(parents=True)
    (root / "input" / "models" / "pysar_cyr_v17.pt").write_bytes(b"m")
    (root / "input" / "scripts").mkdir(parents=True)
    for name in ns["NEEDED_SCRIPTS"]:
        (root / "input" / "scripts" / name).write_text("# stub\n", encoding="utf-8")
    shutil.copy(FAKE, root / "input" / "scripts" / "htr_case_run.py")
    return ns


def _part(root: Path, n: int, stems: list[str]) -> str:
    tar = root / f"spr-1.part-{n:03d}-of-003.tar"
    with tarfile.open(tar, "w") as tf:
        for stem in stems:
            info = tarfile.TarInfo(f"{stem}.jpg")
            info.size = 4
            tf.addfile(info, io.BytesIO(b"\xff\xd8jp"))
    return f"file://{tar}"


def _case(root: Path, *, missing_part: int = 0) -> dict:
    """Справа з трьох частин; `missing_part` — частина, якої в сховищі НЕМАЄ."""
    parts = []
    for n, stems in enumerate(PARTS, 1):
        url = (f"file://{root}/немає-{n}.tar" if n == missing_part
               else _part(root, n, stems))
        parts.append({"url": url, "frames": [f"{s}.jpg" for s in stems]})
    return {"case": "spr-1", "pages_url": parts[0]["url"], "pages_parts": parts,
            "estimated_n_pages": 12, "ckpt_urls": [], "resume_urls": []}


def _already_read(root: Path, stems: list[str], *, in_meta: list[str] | None = None) -> Path:
    """Стан попередньої оренди: тексти й мета — те, що приїхало б чекпоінтами."""
    out = root / "working" / "spr-1" / "out"
    out.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (out / f"{stem}.txt").write_text("текст", encoding="utf-8")
    known = stems if in_meta is None else in_meta
    (out / "_htr_meta.part1.json").write_text(json.dumps(
        {"version": 1, "pages": {f"{s}.jpg": {"lines": 3} for s in known}}), encoding="utf-8")
    return out


def _run(root: Path, monkeypatch: pytest.MonkeyPatch, case: dict, **env: str):
    log = root / "read.log"
    monkeypatch.setenv("FAKE_READ_LOG", str(log))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ns = _runner(root)
    out = ns["_main_inner"]({"cases": [case], "shards": 2, "shards_max": 2,
                             "regulate": False, "seg_cache": False,
                             # як у наглядача: канал уже зміряли ворота
                             "min_net_mbps": 0,
                             "model": "pysar_cyr_v17.pt", "queue_retry_passes": 0})
    read = log.read_text(encoding="utf-8").split() if log.exists() else []
    summary = json.loads((root / "working" / "spr-1" / "htr_case_summary.json")
                         .read_text(encoding="utf-8"))
    return out, sorted(read), summary


def test_a_fresh_case_in_parts_is_read_whole(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    out, read, summary = _run(tmp_path, monkeypatch, _case(tmp_path))
    assert out["results"][0]["complete"]
    assert read == STEMS
    assert summary["n_pages_expected"] == 12 and summary["n_pages_txt"] == 12


def test_a_part_read_by_the_previous_rent_is_not_downloaded(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Перша частина прочитана вся, з другої — одна сторінка. Архіву першої в
    сховищі немає взагалі: спроба його качати впала б. Знаменник — усі 12."""
    _already_read(tmp_path, [*PARTS[0], "0005"])
    out, read, summary = _run(tmp_path, monkeypatch, _case(tmp_path, missing_part=1))
    assert out["results"][0]["complete"], out["results"][0]
    assert read == STEMS[5:], "прочитане попередньою орендою не перечитується"
    assert summary["n_pages_expected"] == 12, "знаменник справи — не лише привезені кадри"
    assert summary["n_pages_txt"] == 12 and not summary["missing_pages"]
    assert summary["resumed_pages"] == 5


def test_a_part_with_a_page_outside_the_meta_is_downloaded(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """🔴 Текст є, а запису в меті немає — шард таку сторінку ПЕРЕЧИТАЄ. Частину
    без кадрів лишати не можна: на місці кадру лежав би порожній файл."""
    _already_read(tmp_path, PARTS[0], in_meta=PARTS[0][:3])
    out, read, summary = _run(tmp_path, monkeypatch, _case(tmp_path))
    assert out["results"][0]["complete"]
    assert "0004" in read, "сторінка без запису в меті перечитана зі справжнього кадру"
    assert summary["n_pages_txt"] == 12


def test_a_page_that_goes_missing_in_a_skipped_part_brings_the_part_back(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Запобіжник перед догоном: частину визнано прочитаною, а тексту сторінки
    вже немає. Порожній файл догін не прочитав би ніколи — довозимо кадри."""
    _already_read(tmp_path, PARTS[0])
    out, read, summary = _run(tmp_path, monkeypatch, _case(tmp_path), FAKE_VANISH="0002")
    assert out["results"][0]["complete"], out["results"][0]
    assert "0002" in read
    assert summary["catchup_rounds"] >= 1 and summary["n_pages_txt"] == 12


def test_a_fully_read_case_downloads_nothing(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    _already_read(tmp_path, STEMS)
    case = _case(tmp_path)
    for part in case["pages_parts"]:
        part["url"] = f"file://{tmp_path}/немає.tar"
    case["pages_url"] = case["pages_parts"][0]["url"]
    out, read, summary = _run(tmp_path, monkeypatch, case)
    assert out["results"][0]["complete"]
    assert read == [] and summary["n_pages_expected"] == 12


def test_state_from_checkpoints_on_a_fresh_box_skips_the_read_part(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """🔴 30.09.2026, spr-8464: відновлення стало першим кроком справи, а тека
    `/tmp/htrcase` на свіжій машині ще не існувала — curl падав на записі
    (rc=23) на всіх 14 чекпоінтах, і пересадка перечитала справу з нуля.

    Тут стан приходить так, як на справжній машині: чекпоінтом за посиланням,
    на машину без `/tmp/htrcase`."""
    staged = tmp_path / "ckpt_src"
    _already_read(staged, PARTS[0])
    ball = tmp_path / "ckpt_0001.tgz"
    with tarfile.open(ball, "w:gz") as tf:
        tf.add(staged / "working" / "spr-1" / "out", arcname="out")
    shutil.rmtree("/tmp/htrcase", ignore_errors=True)
    case = _case(tmp_path, missing_part=1)
    case["resume_urls"] = [f"file://{ball}"]
    out, read, summary = _run(tmp_path, monkeypatch, case)
    assert out["results"][0]["complete"], out["results"][0]
    assert read == STEMS[4:], "прочитане в чекпоінті не перечитується"
    assert summary["resumed_pages"] == 4 and summary["n_pages_expected"] == 12


def test_a_whole_case_after_a_case_in_parts_reads_its_own_frames(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """🔴🔴 30.09.2026, черга [spr-7298 частинами, spr-6750 одним архівом]: друга
    справа успадкувала з верхнього рівня job'а частини першої, прочитала чужі
    кадри й залила 2357 чужих сторінок під своїми ключами."""
    first = _case(tmp_path)
    tar = tmp_path / "spr-2.tar"
    with tarfile.open(tar, "w") as tf:
        for stem in ("0101", "0102", "0103"):
            info = tarfile.TarInfo(f"{stem}.jpg")
            info.size = 4
            tf.addfile(info, io.BytesIO(b"\xff\xd8jp"))
    second = {"case": "spr-2", "pages_url": f"file://{tar}", "estimated_n_pages": 3,
              "ckpt_urls": [], "resume_urls": []}
    log = tmp_path / "read.log"
    monkeypatch.setenv("FAKE_READ_LOG", str(log))
    ns = _runner(tmp_path)
    # як у наглядача: верхній рівень job'а несе посилання ПЕРШОЇ справи
    out = ns["_main_inner"]({**first, "cases": [first, second], "shards": 2, "shards_max": 2,
                             "regulate": False, "seg_cache": False, "min_net_mbps": 0,
                             "model": "pysar_cyr_v17.pt", "queue_retry_passes": 0,
                             "pipeline": False})
    assert all(r.get("complete") for r in out["results"]), out["results"]
    second_out = tmp_path / "working" / "spr-2" / "out"
    assert sorted(p.stem for p in second_out.glob("*.txt")) == ["0101", "0102", "0103"], \
        "друга справа читає СВОЇ кадри, а не частини першої"
