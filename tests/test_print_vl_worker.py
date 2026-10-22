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


# ── пакет номерів одним переліком (16.10.2026) ───────────────────────────────
def _queue(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    q = tmp_path / "queue.jsonl"
    q.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in [*entries, {"end": True}])
                 + "\n", encoding="utf-8")
    return q


def test_waiting_issues_go_into_the_engine_as_one_list(tmp_path: Path, capsys) -> None:
    """Номер газети — 2 сторінки: окремий `predict` на кожен спорожнював
    конвеєр Paddle, і номер коштував 10–22 с замість ~8."""
    entries = [{"case_dir": str(_case(tmp_path, f"{k} 1937-01-{k:02d}", 2)),
                "out_dir": str(tmp_path / f"o{k}"), "index": k} for k in range(1, 6)]
    pipe = _Pipe(PAGE)
    assert W.run_queue(_args(str(_queue(tmp_path, entries))), _box(pipe)) == 0

    assert len(pipe.calls) == 1 and len(pipe.calls[0]) == 10
    ev = _events(capsys)
    assert [e["qi"] for e in ev if e["phase"] == "case_done"] == [1, 2, 3, 4, 5]
    for k in range(1, 6):
        assert (tmp_path / f"o{k}" / "0002.txt").exists()
        meta = json.loads((tmp_path / f"o{k}" / "_htr_meta.json").read_text(encoding="utf-8"))
        assert meta["done"] and set(meta["pages"]) == {"0001.jpg", "0002.jpg"}


def test_a_batch_never_outgrows_its_page_ceiling(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(W, "BATCH_PAGES", 4)
    entries = [{"case_dir": str(_case(tmp_path, f"{k} 1937-02-{k:02d}", 2)),
                "out_dir": str(tmp_path / f"o{k}"), "index": k} for k in range(1, 6)]
    pipe = _Pipe(PAGE)
    W.run_queue(_args(str(_queue(tmp_path, entries))), _box(pipe))
    assert [len(c) for c in pipe.calls] == [4, 4, 2]


def test_case_done_stays_in_queue_order_around_finished_and_removed_issues(
        tmp_path: Path, capsys) -> None:
    """🔴 Раннер бере `qi_done` як максимум: справа 3, закрита раніше за 1 і 2,
    оголосила б закритими справи, які ще читаються."""
    a = _case(tmp_path, "1 1937-03-01", 2)
    b = _case(tmp_path, "2 1937-03-02", 1)
    W.run_queue(_args(str(_queue(tmp_path, [
        {"case_dir": str(b), "out_dir": str(tmp_path / "ob"), "index": 1}]))), _box(_Pipe(PAGE)))
    capsys.readouterr()                                       # b уже прочитано — нічого читати
    (tmp_path / "queue.jsonl").unlink()
    entries = [{"case_dir": str(a), "out_dir": str(tmp_path / "oa"), "index": 1},
               {"case_dir": str(b), "out_dir": str(tmp_path / "ob"), "index": 2},
               {"case_dir": str(tmp_path / "прибрано"), "out_dir": str(tmp_path / "ox"), "index": 3},
               {"case_dir": str(_case(tmp_path, "4 1937-03-04", 1)),
                "out_dir": str(tmp_path / "od"), "index": 4}]
    W.run_queue(_args(str(_queue(tmp_path, entries))), _box(_Pipe(PAGE)))
    done = [e["qi"] for e in _events(capsys) if e["phase"] == "case_done"]
    assert done == [1, 2, 3, 4]


def test_a_failed_page_in_a_batch_is_quarantined_in_its_own_issue(
        tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(W, "server_alive", lambda url, timeout=10: True)
    a = _case(tmp_path, "1 1937-04-01", 2)
    b = _case(tmp_path, "2 1937-04-02", 2)
    (b / "0001.jpg").rename(b / "0101.jpg")                    # унікальне ім'я збійної сторінки
    pipe = _Pipe(PAGE, bad={"0101.jpg"})
    rc = W.run_queue(_args(str(_queue(tmp_path, [
        {"case_dir": str(a), "out_dir": str(tmp_path / "oa"), "index": 1},
        {"case_dir": str(b), "out_dir": str(tmp_path / "ob"), "index": 2}]))), _box(pipe))

    assert rc == 0
    assert (tmp_path / "oa" / "0001.txt").exists() and (tmp_path / "oa" / "0002.txt").exists()
    assert "0101.jpg" in W.load_quarantine(tmp_path / "ob")
    assert not W.load_quarantine(tmp_path / "oa")
    assert (tmp_path / "ob" / "0002.txt").exists(), "решту пакета дочитано після збою"
    ev = _events(capsys)
    bad = [e for e in ev if e["phase"] == "htr" and e.get("error")]
    assert [e["qi"] for e in bad] == [2]
