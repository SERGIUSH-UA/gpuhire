"""🚂 Конвеєр черги на СПРАВЖНЬОМУ коді раннера з фейковим шардом.

23.09.2026 черга 194 справ по ~15 сторінок: справа займала ~41 с, з яких
читання ~12 — шарди піднімались заново на кожну справу (моделі 7 с), а
підготовка й закриття справ стояли між справами. Тут перевіряється, що шарди
живуть усю чергу, справи закриваються повними, а отруйна сторінка не валить
ні флот, ні чергу.

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

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="бокс-раннер живе на Linux-шляхах")

EMB = Path(__file__).resolve().parents[1] / "src" / "gpurunner" / "_embedded"
FAKE = Path(__file__).resolve().parent / "data" / "fake_queue_shard.py"


def _runner(root: Path) -> dict:
    src = (EMB / "_common.py").read_text(encoding="utf-8") + "\n" + (
        EMB / "htr_case_runner.py").read_text(encoding="utf-8").replace(
        'if __name__ == "__main__":\n    main({})', "")
    ns: dict = {"__name__": "runner_under_test"}
    exec(compile(src, "htr_case_runner.py", "exec"), ns)
    ns["KAGGLE_INPUT"] = root / "input"
    ns["KAGGLE_WORKING"] = root / "working"
    ns["QUEUE_DIR"] = root / "queue"
    ns["PROGRESS_PATH"] = root / "_progress.json"
    ns["_ensure_deps"] = lambda: None
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


def _cases(root: Path, n: int) -> list[dict]:
    out = []
    for c in range(1, n + 1):
        tar = root / f"case{c}.tar"
        pages = 4 + c % 7
        with tarfile.open(tar, "w") as tf:
            for p in range(1, pages + 1):
                info = tarfile.TarInfo(f"{p:04d}.jpg")
                info.size = 4
                tf.addfile(info, io.BytesIO(b"\xff\xd8jp"))
        out.append({"case": f"spr-{c}", "pages_url": f"file://{tar}",
                    "estimated_n_pages": pages, "ckpt_urls": [], "resume_urls": []})
    return out


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int, **env: str):
    monkeypatch.setenv("FAKE_STARTS", str(tmp_path / "starts.log"))
    monkeypatch.setenv("FAKE_PAGE_SEC", "0.02")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ns = _runner(tmp_path)
    cases = _cases(tmp_path, n)
    out = ns["_main_inner"]({"cases": cases, "shards": 4, "shards_max": 4,
                             "regulate": False, "seg_cache": False,
                             "model": "pysar_cyr_v17.pt", "queue_retry_passes": 1})
    starts = (tmp_path / "starts.log").read_text().splitlines()
    return ns, cases, out, starts


def test_shards_live_through_the_whole_queue(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    ns, cases, out, starts = _run(tmp_path, monkeypatch, 24)
    assert len(starts) == 4, f"шарди піднімались {len(starts)} разів на 24 справи"
    assert sum(1 for r in out["results"] if r.get("complete")) == 24
    for c in cases:
        d = tmp_path / "working" / ns["_case_slug"](c["case"])
        assert len(list((d / "out").glob("*.txt"))) == c["estimated_n_pages"], c["case"]
        summary = json.loads((d / "htr_case_summary.json").read_text(encoding="utf-8"))
        assert summary["complete"] and summary["pipeline"]
    assert not list(Path("/tmp/htrcase").glob("pq_*")), "кадри закритих справ прибрано"


def test_a_poison_page_neither_kills_the_fleet_nor_blocks_the_queue(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Шард помирає на сторінці — піднімається знову; двічі — сторінка в
    карантині, справа закривається, черга йде далі (без дедлоку вікна)."""
    _ns, _cases, out, starts = _run(tmp_path, monkeypatch, 12, FAKE_POISON="spr-3:0002")
    assert sum(1 for r in out["results"] if r.get("complete")) == 12
    q = tmp_path / "working" / "spr-3" / "out" / "_htr_quarantine.json"
    assert "0002.jpg" in json.loads(q.read_text(encoding="utf-8"))["pages"]
    assert len(starts) <= 4 + 3, "оживлень не більше за стелю перезапусків"


def test_a_slow_page_does_not_stall_the_rest_of_the_queue(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """🔴 Блокування голови черги. Справи закривались суворо по порядку, а
    підготовка тримала не більше 4 незакритих: одна повільна сторінка зупиняла
    весь флот (приймач 23.09.2026 — паузи 3 і 18 хв із 47). Поки сторінка
    spr-2 читається 4 с, решта шардів мусить дочитати інші справи."""
    ns, cases, out, _ = _run(tmp_path, monkeypatch, 24, FAKE_SLOW="spr-2:0001:4")
    assert sum(1 for r in out["results"] if r.get("complete")) == 24
    work = tmp_path / "working"
    slow_done = (work / "spr-2" / "out" / "0001.txt").stat().st_mtime
    finished_before = 0
    for c in cases[2:]:
        texts = list((work / ns["_case_slug"](c["case"]) / "out").glob("*.txt"))
        if max(p.stat().st_mtime for p in texts) < slow_done:
            finished_before += 1
    assert finished_before >= 12, (
        f"поки читалась повільна сторінка, дочитано лише {finished_before} справ")

