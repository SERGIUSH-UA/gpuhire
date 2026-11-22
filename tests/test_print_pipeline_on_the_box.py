"""🖨 Друк на СПРАВЖНЬОМУ коді раннера і справжньому робітнику друку.

Підмінено лише конвеєр Paddle (сторінка → блоки) і підйом середовища: черга,
командний рядок робітника, його argparse, повнота й повтор — бойові.

Перший живий захід (12.10.2026, 2 номери газети, RTX 3090) не прочитав жодної
сторінки, хоч середовище й vLLM піднялись:

1. раннер дав робітнику прапорці шардів рукописного рушія (`--shard 1/64
   --claim`), argparse робітника їх не знає — rc=2 тричі поспіль, справи
   закрились порожніми;
2. повтор неповних справ пішов шляхом однієї справи (шарди kraken) і впав на
   воротах друку.

Раннер живе на Linux-шляхах (`/tmp/htrcase`), тож тест — для Linux (CI;
локально — WSL).
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.skipif(sys.platform == "win32",
                                 reason="бокс-раннер живе на Linux-шляхах"),
              pytest.mark.usefixtures("one_box_at_a_time")]

EMB = Path(__file__).resolve().parents[1] / "src" / "gpurunner" / "_embedded"

# Конвеєр Paddle, що віддає по блоку тексту на сторінку. Маркер `FAKE_SHORT_ONCE`
# — один раз віддати на сторінку менше (рушій «не віддав сторінку»): справа
# лишається неповною, і її мусить дочитати повтор.
FAKE_PIPE = '''
class _FakeRes:
    def __init__(self, path):
        stem = Path(path).stem
        self.json = {"res": {"width": 1000, "parsing_res_list": [
            {"block_label": "text", "block_content": "Пирятинського району " + stem,
             "block_bbox": [100, 100, 900, 200], "block_order": 1}]}}
        self.markdown = {"markdown_texts": "Пирятинського району " + stem}


class _FakePipe:
    def predict(self, paths, use_queues=True):
        short = Path(os.environ.get("FAKE_SHORT_ONCE", "/nonexistent"))
        if short.exists():
            short.unlink()
            paths = paths[:-1]
        for p in paths:
            yield _FakeRes(p)


make_pipeline = lambda url: _FakePipe()  # noqa: E731
server_alive = lambda url, timeout=10: True  # noqa: E731

if __name__ == "__main__":
    sys.exit(main())
'''


def _worker_src() -> str:
    src = (EMB / "print_vl_worker.py").read_text(encoding="utf-8")
    tail = 'if __name__ == "__main__":\n    sys.exit(main())'
    assert tail in src
    return src.replace(tail, FAKE_PIPE)


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
    ns["_free_vram_per_card"] = lambda: [24.0]
    ns["_gpu_name"] = lambda: "fake"
    ns["_usable_cores"] = lambda: 8

    def env_up(params):
        ns["_PRINT_ENV"].update(python=sys.executable, server="http://fake/v1")
        return sys.executable, "http://fake/v1"

    ns["_ensure_print_env"] = env_up
    return ns


def _issues(root: Path, n: int, pages: int = 3) -> list[dict]:
    out = []
    for c in range(1, n + 1):
        tar = root / f"issue{c}.tar"
        with tarfile.open(tar, "w") as tf:
            for p in range(1, pages + 1):
                info = tarfile.TarInfo(f"{p:04d}.jpg")
                info.size = 4
                tf.addfile(info, io.BytesIO(b"\xff\xd8jp"))
        out.append({"case": f"{c}-{770 + c}-1936-01-0{c}", "pages_url": f"file://{tar}",
                    "estimated_n_pages": pages, "ckpt_urls": [], "resume_urls": []})
    return out


def _params(cases: list[dict]) -> dict:
    return {"cases": cases, "engine": "paddle_vl", "script": "print",
            "model": "PaddleOCR-VL-1.6-0.9B", "_print_worker_src": _worker_src(),
            "queue_retry_passes": 1}


def _texts(ns: dict, case: dict) -> list[str]:
    out_dir = ns["KAGGLE_WORKING"] / ns["_case_slug"](case["case"]) / "out"
    return sorted(p.name for p in out_dir.glob("*.txt"))


def test_print_queue_is_read_by_the_print_worker(tmp_path: Path) -> None:
    ns = _runner(tmp_path)
    cases = _issues(tmp_path, 2)
    out = ns["_main_inner"](_params(cases))
    assert [r.get("complete") for r in out["results"]] == [True, True]
    for case in cases:
        assert _texts(ns, case) == ["0001.txt", "0002.txt", "0003.txt"]
    log = (ns["QUEUE_DIR"] / "logs" / "shard01.log").read_text(encoding="utf-8")
    assert "unrecognized arguments" not in log
    meta = json.loads((ns["KAGGLE_WORKING"] / ns["_case_slug"](cases[0]["case"]) / "out"
                       / "_htr_meta.json").read_text(encoding="utf-8"))
    assert meta["engine"] == "print"


def test_incomplete_print_issue_is_retried_by_the_pipeline(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "short_once"
    marker.write_text("1")
    monkeypatch.setenv("FAKE_SHORT_ONCE", str(marker))
    ns = _runner(tmp_path)
    cases = _issues(tmp_path, 2)
    out = ns["_main_inner"](_params(cases))
    assert not marker.exists()          # перший прохід справді лишив дірку
    assert [r.get("complete") for r in out["results"]] == [True, True]
    for case in cases:
        assert _texts(ns, case) == ["0001.txt", "0002.txt", "0003.txt"]
