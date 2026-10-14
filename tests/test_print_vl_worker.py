"""Робітник друку (`_embedded/print_vl_worker.py`): контракт шарда рукописного рушія.

Справжній PaddleOCR-VL тут не потрібен: конвеєр підставний, бо перевіряється
наш код — чистий текст у порядку читання, мета, події прогресу з `qi`, карантин,
приймач повноти з диска і вихід шарда, коли впав сервер, а не сторінка.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

EMB = Path(__file__).resolve().parents[1] / "src" / "gpurunner" / "_embedded"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("print_vl_worker", EMB / "print_vl_worker.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load()


def _res(blocks: list[tuple[str, str, list[int]]], width: int = 1000) -> dict[str, Any]:
    return {"res": {"width": width, "height": 1400, "parsing_res_list": [
        {"block_label": lab, "block_content": txt, "block_bbox": bb, "block_order": k}
        for k, (lab, txt, bb) in enumerate(blocks, 1)]}}


class _Res:
    def __init__(self, data: dict[str, Any]) -> None:
        self.json = data
        self.markdown = {"markdown_texts": "# md"}


class _Pipe:
    """Підставний конвеєр: відповідь на кожну сторінку, на `bad` — падає."""

    def __init__(self, page: dict[str, Any], bad: set[str] | None = None) -> None:
        self.page, self.bad = page, bad or set()
        self.calls: list[list[str]] = []

    def predict(self, paths: list[str], use_queues: bool = True):
        self.calls.append([Path(p).name for p in paths])
        for p in paths:
            if Path(p).name in self.bad:
                raise RuntimeError("vLLM впав")
            yield _Res(self.page)


PAGE = _res([
    ("doc_title", "КОЛЕКТИВІЗОВАНЕ СЕЛО", [100, 10, 900, 80]),
    ("image", "лого", [10, 10, 90, 90]),
    ("text", "Бюро обкому КП(б)У та президія обласного виконавчого комітету "
             "постановляють:", [100, 100, 400, 200]),
    ("text", "Бюро обкому КП(б)У та президія обласного виконавчого комітету "
             "постановляють:", [110, 105, 400, 200]),          # рамки перекриваються
    ("text", "ухи ькі чу", [0, 300, 20, 900]),                 # сусідня сторінка
    ("text", "Гриль- النوالبب\nдабут Василь, Сянар Настя", [420, 100, 700, 300]),
    ("table", "<table><tr><td>Гною</td><td>7.700 тис. тонн</td></tr></table>",
     [100, 400, 700, 500]),
])


def _case(root: Path, name: str = "3 (770) 1936-01-06", pages: int = 3) -> Path:
    d = root / name
    d.mkdir(parents=True)
    for k in range(1, pages + 1):
        Image.new("L", (40, 60), 200).save(d / f"{k:04d}.jpg", "JPEG")
    return d


def _args(queue: str = "", **kw: Any) -> Any:
    class A:
        pass
    a = A()
    a.queue, a.progress_json, a.vl_server = queue, True, "http://127.0.0.1:1/v1"
    a.model, a.limit = W.MODEL_NAME, 0
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _events(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [json.loads(ln[len(W.PROGRESS_PREFIX):]) for ln in capsys.readouterr().out.splitlines()
            if ln.startswith(W.PROGRESS_PREFIX)]


def _box(pipe: _Pipe) -> dict[str, Any]:
    return {"pipe": pipe, "make": lambda: pytest.fail("конвеєр не мав створюватись")}


# ── текст сторінки ───────────────────────────────────────────────────────────
def test_page_text_keeps_reading_order_and_drops_noise() -> None:
    text, kept = W.page_text(*W.blocks_of(PAGE))

    assert text.split("\n\n")[0] == "КОЛЕКТИВІЗОВАНЕ СЕЛО"
    assert text.count("Бюро обкому") == 1, "повтор блоку з рамок, що перекриваються"
    assert "лого" not in text and "ухи ькі" not in text
    assert "النوالبب" not in text and "дабут Василь, Сянар Настя" in text
    assert "<td>" not in text and "Гною 7.700 тис. тонн" in text
    assert kept[0]["heading"] and not kept[1]["heading"]


def test_short_repeats_are_legitimate() -> None:
    page = _res([("text", "Пирятин", [100, 0, 300, 50]), ("text", "Пирятин", [100, 900, 300, 950])])
    assert W.page_text(*W.blocks_of(page))[0].count("Пирятин") == 2


# ── справа ───────────────────────────────────────────────────────────────────
def test_case_writes_the_shard_contract(tmp_path: Path, capsys) -> None:
    src, out = _case(tmp_path), tmp_path / "out"
    pipe = _Pipe(PAGE)

    assert W.run_case(_args(), _box(pipe), src, out, "VYD-test") == 0

    for stem in ("0001", "0002", "0003"):
        assert (out / f"{stem}.txt").read_text(encoding="utf-8").startswith("КОЛЕКТИВІЗОВАНЕ")
        assert (out / f"{stem}.md").exists() and (out / f"{stem}.blocks.json").exists()
    meta = json.loads((out / "_htr_meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == W.MODEL_NAME and meta["engine"] == "print"
    assert set(meta["pages"]) == {"0001.jpg", "0002.jpg", "0003.jpg"}
    assert meta["missing"] == [] and meta["done"] is True and meta["case_key"] == "VYD-test"
    phases = [e["phase"] for e in _events(capsys)]
    assert phases.count("page_start") == 3 and phases.count("htr") == 3
    assert phases[-1] == "done"
    assert pipe.calls == [["0001.jpg", "0002.jpg", "0003.jpg"]], "одним переліком"


def test_resume_reads_only_what_is_missing(tmp_path: Path) -> None:
    src, out = _case(tmp_path), tmp_path / "out"
    W.run_case(_args(), _box(_Pipe(PAGE)), src, out)
    (out / "0002.txt").unlink()
    meta = json.loads((out / "_htr_meta.json").read_text(encoding="utf-8"))
    meta["pages"].pop("0002.jpg")
    (out / "_htr_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    pipe = _Pipe(PAGE)

    assert W.run_case(_args(), _box(pipe), src, out) == 0
    assert pipe.calls == [["0002.jpg"]]


def test_failed_page_is_quarantined_and_the_rest_is_read(tmp_path: Path, monkeypatch) -> None:
    src, out = _case(tmp_path), tmp_path / "out"
    monkeypatch.setattr(W, "server_alive", lambda url, timeout=10: True)
    pipe = _Pipe(PAGE, bad={"0002.jpg"})

    assert W.run_case(_args(), _box(pipe), src, out) == 0, "карантин не робить справу неповною"
    assert (out / "0001.txt").exists() and (out / "0003.txt").exists()
    assert not (out / "0002.txt").exists()
    assert "0002.jpg" in W.load_quarantine(out)
    assert pipe.calls[-1] == ["0003.jpg"], "після падіння — строго решта черги"


def test_dead_server_stops_the_shard_instead_of_quarantining_everything(
        tmp_path: Path, monkeypatch) -> None:
    src, out = _case(tmp_path, pages=5), tmp_path / "out"
    monkeypatch.setattr(W, "server_alive", lambda url, timeout=10: False)
    pipe = _Pipe(PAGE, bad={"0001.jpg", "0002.jpg", "0003.jpg", "0004.jpg", "0005.jpg"})

    with pytest.raises(W.ServerDown):
        W.run_case(_args(), _box(pipe), src, out)
    assert W.load_quarantine(out) == {}, "смерть сервера — не вада сторінки"


# ── черга ────────────────────────────────────────────────────────────────────
def test_queue_runs_cases_and_tags_events_with_qi(tmp_path: Path, capsys) -> None:
    a, b = _case(tmp_path, "3 (770) 1936-01-06", 2), _case(tmp_path, "4 (771) 1936-01-08", 1)
    q = tmp_path / "queue.jsonl"
    q.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in [
        {"case_dir": str(a), "out_dir": str(tmp_path / "o1"), "index": 1},
        {"case_dir": str(b), "out_dir": str(tmp_path / "o2"), "index": 2},
        {"case_dir": str(tmp_path / "прибрано"), "out_dir": str(tmp_path / "o3"), "index": 3},
        {"end": True}]) + "\n", encoding="utf-8")

    assert W.run_queue(_args(str(q)), _box(_Pipe(PAGE))) == 0

    ev = _events(capsys)
    assert [e["qi"] for e in ev if e["phase"] == "case_done"] == [1, 2, 3]
    assert {e["qi"] for e in ev if e["phase"] == "htr"} == {1, 2}
    assert ev[-1]["phase"] == "queue_done" and "qi" not in ev[-1]
    assert (tmp_path / "o2" / "0001.txt").exists()
