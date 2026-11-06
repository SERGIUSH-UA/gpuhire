"""Unit tests for PaddleOCRJob param validation + remote code rendering."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from gpurunner.jobs import PaddleOCRJob, get_job


def test_get_job_resolves() -> None:
    cls = get_job("paddleocr")
    assert cls is PaddleOCRJob


def test_validate_urls_list_strings() -> None:
    job = PaddleOCRJob()
    p = job.validate_params({"urls": ["https://example.com/a.pdf", "https://example.com/b.pdf"]})
    assert len(p["items"]) == 2
    assert p["items"][0]["url"] == "https://example.com/a.pdf"
    assert p["items"][0]["label"] == "pdf_0000"
    assert p["items"][1]["label"] == "pdf_0001"


def test_validate_urls_with_labels() -> None:
    job = PaddleOCRJob()
    p = job.validate_params({
        "urls": [
            {"url": "https://example.com/a.pdf", "label": "pev_1880_01"},
            {"url": "https://example.com/b.pdf", "label": "pev_1880_02"},
        ]
    })
    assert p["items"][0]["label"] == "pev_1880_01"
    assert p["items"][1]["label"] == "pev_1880_02"


def test_validate_urls_file(tmp_path: Path) -> None:
    f = tmp_path / "urls.txt"
    f.write_text(
        "# header comment\n"
        "https://example.com/a.pdf\n"
        "\n"
        "https://example.com/b.pdf\n"
        "  https://example.com/c.pdf  \n",
        encoding="utf-8",
    )
    p = PaddleOCRJob().validate_params({"urls_file": str(f)})
    assert len(p["items"]) == 3
    assert p["items"][2]["url"] == "https://example.com/c.pdf"


def test_validate_rejects_non_http() -> None:
    job = PaddleOCRJob()
    with pytest.raises(ValueError, match="not an http"):
        job.validate_params({"urls": ["ftp://example.com/a.pdf"]})


def test_validate_rejects_empty() -> None:
    with pytest.raises(ValueError, match="needs 'urls'"):
        PaddleOCRJob().validate_params({})


def test_validate_scale_bounds() -> None:
    job = PaddleOCRJob()
    with pytest.raises(ValueError, match="scale must be"):
        job.validate_params({"urls": ["https://x/a.pdf"], "scale": 10.0})


def test_render_remote_code_injects_params_and_runner() -> None:
    job = PaddleOCRJob()
    code = job.render_remote_code({
        "urls": ["https://example.com/a.pdf"],
        "lang": "uk",
        "scale": 2.5,
    })
    assert "PARAMS = " in code
    assert "https://example.com/a.pdf" in code
    assert '"lang": "uk"' in code or "'lang': 'uk'" in code
    # runner module body must be inlined
    assert "def main(params:" in code
    assert "PaddleOCR(" in code
    # entry point must call main
    assert "main(PARAMS)" in code
    # runner's own __main__ guard must be stripped
    assert 'if __name__ == "__main__":' not in code


def _injected_urls(code: str) -> list[str]:
    """URLs from the injected PARAMS block of rendered remote code.

    Substring checks over the whole rendered code are unsound: the inlined
    runner source carries its own docstrings with example filenames (e.g.
    ``…Д-174-1-т.1.pdf``), so a bare `"1.pdf" not in code` is always false.

    Parsed through the AST rather than by regex: the injected block's exact
    quoting is an implementation detail of ``Job._params_block`` (it changed once
    already, when ``r'''…'''`` turned out to be a syntax error for any value
    containing ``'''``), and a test that pins the spelling breaks on every such
    fix while proving nothing about the payload.
    """
    tree = ast.parse(code)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets)
            and isinstance(node.value, ast.Call)
            and node.value.args
        ):
            payload = ast.literal_eval(node.value.args[0])
            return [item["url"] for item in json.loads(payload)["items"]]
    raise AssertionError("rendered code has no PARAMS block")


def test_render_sharding_splits_items() -> None:
    job = PaddleOCRJob()
    urls = [f"https://example.com/{i}.pdf" for i in range(10)]
    code_shard0 = job.render_remote_code({"urls": urls}, shard_index=0, total_shards=3)
    code_shard1 = job.render_remote_code({"urls": urls}, shard_index=1, total_shards=3)

    assert _injected_urls(code_shard0) == [
        f"https://example.com/{i}.pdf" for i in (0, 3, 6, 9)
    ]
    assert _injected_urls(code_shard1) == [
        f"https://example.com/{i}.pdf" for i in (1, 4, 7)
    ]


def test_estimate_runtime_scales_with_items() -> None:
    job = PaddleOCRJob()
    short = job.estimate_runtime({"urls": ["https://x/1.pdf"]})
    long = job.estimate_runtime({"urls": [f"https://x/{i}.pdf" for i in range(20)]})
    assert long > short


def test_empty_big_page_is_retried_downscaled_with_coords_back_in_render_scale(tmp_path: Path) -> None:
    """Детектор на великому рендері машинопису дав 0 блоків (опис ANRM ф.205, 26.10.2026):
    порожню сторінку понад RETRY_MAX_SIDE раннер повторює зменшеною, а координати блоків
    повертає в масштаб основного рендера."""
    from PIL import Image

    from gpurunner._embedded import paddleocr_runner as R

    seen = []

    class FakeOcr:
        def predict(self, path):
            w, h = Image.open(path).size
            seen.append(max(w, h))
            if max(w, h) > R.RETRY_MAX_SIDE:
                return [{"rec_texts": ["."], "rec_scores": [0.1], "rec_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}]
            return [{"rec_texts": ["1931. 447. Дело"], "rec_scores": [0.9],
                     "rec_polys": [[[90, 90], [180, 90], [180, 99], [90, 99]]]}]

    big = Image.new("L", (2500, 3100), 255)
    texts, blocks = R._ocr_one_page(FakeOcr(), big, tmp_path / "p.png")
    assert texts == ["."]
    texts, blocks, k = R._retry_downscaled(FakeOcr(), big, tmp_path / "p.png")
    assert texts == ["1931. 447. Дело"] and k is not None
    assert seen[-1] <= R.RETRY_MAX_SIDE
    assert abs(blocks[0]["poly"][0][0] - 90 * 3100 / (R.RETRY_MAX_SIDE * 0.9)) < 0.05


def test_run_one_writes_retried_text_for_an_empty_big_page(tmp_path: Path, monkeypatch) -> None:
    """Повтор — частина проходу: сторінку, порожню на великому рендері, `run_one` пише з
    текстом зменшеного повтору й позначкою `retry_downscale`."""
    from PIL import Image

    from gpurunner._embedded import paddleocr_runner as R

    class FakeOcr:
        def predict(self, path):
            if max(Image.open(path).size) > R.RETRY_MAX_SIDE:
                return [{"rec_texts": ["."], "rec_scores": [0.1], "rec_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}]
            return [{"rec_texts": ["138. 271 Дело", "13.6. 1813 г."], "rec_scores": [0.9, 0.9],
                     "rec_polys": [[[1, 1], [2, 1], [2, 2], [1, 2]], [[5, 1], [6, 1], [6, 2], [5, 2]]]}]

    monkeypatch.setattr(R, "_download", lambda url, dst: dst.write_bytes(b"%PDF"))
    monkeypatch.setattr(R, "_iter_pages", lambda pdf, scale: iter([(0, Image.new("L", (2500, 3100), 255))]))
    R.run_one({"url": "file:///x.pdf", "label": "opys"}, tmp_path, lang="ru", scale=1.0,
              skip_existing=False, ocr=FakeOcr(), scratch_pdf=tmp_path / "c.pdf",
              scratch_png=tmp_path / "p.png")
    page = json.loads((tmp_path / "opys" / "page_0000.json").read_text(encoding="utf-8"))
    assert page["blocks"][0]["text"] == "138. 271 Дело" and page["retry_downscale"]
