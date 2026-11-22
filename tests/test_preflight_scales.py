"""Передполіт на плані з тисячами справ: ціна перевірки не росте швидше за роботу.

12.10.2026 пачка з 1 789 газетних номерів по 4 сторінки за кілька годин
перевірки дійшла до 137-ї справи: кожна тягнула 8 МБ на замір швидкості й до
восьми запитів по чекпоінти, послідовно. Тут перевіряється, що:

* швидкість міряється один раз на план, решта справ — один байт;
* чекпоінти питаються до першого відсутнього слота (раннер заповнює їх
  суцільно з першого), тож свіжа справа — один запит;
* паралельна перевірка не перемішує звіт: справи й проблеми — у порядку плану;
* тисяча свіжих справ дає одну нотатку, а не тисячу.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from gpurunner.htr import preflight


def _plan(tmp_path: Path, n: int, *, slots: int = 8) -> Path:
    cases = []
    for i in range(n):
        cases.append({
            "case": f"c{i:04d}",
            "n_pages": 4,
            "out_dir": str((tmp_path / "out" / f"c{i:04d}").resolve()),
            "pages_url": f"https://r2/pages/c{i:04d}.tar",
            "ckpt_urls": [f"https://r2/put/c{i:04d}/{k}" for k in range(1, slots + 1)],
            "resume_urls": [f"https://r2/ckpt/c{i:04d}/{k}" for k in range(1, slots + 1)],
        })
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"assets_url": "https://r2/assets.tgz", "cases": cases}),
                    encoding="utf-8")
    return path


class _Net:
    """Підставна мережа: рахує запити й відповідає за правилами тесту."""

    def __init__(self, *, ckpt: dict[str, int] | None = None,
                 dead: set[str] | None = None) -> None:
        self.ckpt = ckpt or {}       # справа → скільки слотів заповнено
        self.dead = dead or set()    # справи з мертвим посиланням на кадри
        self.big = 0                 # заміри швидкості (8 МБ)
        self.small = 0               # однобайтові запити
        self.lock = threading.Lock()

    def probe(self, url: str, *, want_bytes: int = preflight.PROBE_BYTES) -> dict[str, Any]:
        with self.lock:
            if want_bytes > 1:
                self.big += 1
            else:
                self.small += 1
        case = url.rsplit("/", 1)[-1].removesuffix(".tar")
        if case in self.dead:
            return {"ok": False, "http": "403", "mbps": 0.0, "why": "ПРОТЕРМІНОВАНО", "sec": 0.0}
        return {"ok": True, "http": "206", "mbps": 90.0, "why": "", "sec": 0.1}

    def exists(self, url: str) -> bool:
        with self.lock:
            self.small += 1
        _, case, slot = url.rsplit("/", 2)
        return int(slot) <= self.ckpt.get(case, 0)


def _patch(monkeypatch, net: _Net) -> None:
    monkeypatch.setattr(preflight, "probe", net.probe)
    monkeypatch.setattr(preflight, "exists", net.exists)


def test_speed_is_measured_once_per_plan(tmp_path: Path, monkeypatch) -> None:
    net = _Net()
    _patch(monkeypatch, net)

    report = preflight.check_plan(_plan(tmp_path, 300))

    # ассети + перша справа; решта 299 — лише «чи живе»
    assert net.big == 2
    assert report["cases"][0]["pages_url"]["mbps"] == 90.0
    assert all(c["pages_url"]["mbps"] is None for c in report["cases"][1:])
    assert report["ok"]


def test_fresh_case_costs_one_checkpoint_request(tmp_path: Path, monkeypatch) -> None:
    net = _Net()
    _patch(monkeypatch, net)

    report = preflight.check_plan(_plan(tmp_path, 50))

    # 50 справ: один байт на кадри (крім першої) + один слот чекпоінта
    assert net.small == 49 + 50
    assert all(c["ckpt_found"] == 0 and c["ckpt_checked"] == 1 for c in report["cases"])


def test_checkpoints_are_counted_up_to_the_first_gap(tmp_path: Path, monkeypatch) -> None:
    net = _Net(ckpt={"c0001": 3, "c0002": 8})
    _patch(monkeypatch, net)

    report = preflight.check_plan(_plan(tmp_path, 3), ckpt_probe=8)
    by = {c["case"]: c for c in report["cases"]}

    assert (by["c0000"]["ckpt_found"], by["c0000"]["ckpt_checked"]) == (0, 1)
    assert (by["c0001"]["ckpt_found"], by["c0001"]["ckpt_checked"]) == (3, 4)
    assert (by["c0002"]["ckpt_found"], by["c0002"]["ckpt_checked"]) == (8, 8)


def test_parallel_check_keeps_plan_order(tmp_path: Path, monkeypatch) -> None:
    net = _Net(dead={"c0007", "c0042"})
    _patch(monkeypatch, net)

    report = preflight.check_plan(_plan(tmp_path, 60))

    assert [c["case"] for c in report["cases"]] == [f"c{i:04d}" for i in range(60)]
    assert report["problems"] == ["c0007: кадри — ПРОТЕРМІНОВАНО",
                                  "c0042: кадри — ПРОТЕРМІНОВАНО"]
    assert not report["ok"]


def test_many_fresh_cases_make_one_note(tmp_path: Path, monkeypatch) -> None:
    net = _Net(ckpt={"c0003": 2})
    _patch(monkeypatch, net)

    report = preflight.check_plan(_plan(tmp_path, 40))
    fresh = [n for n in report["notes"] if "чекпоінтів ЩЕ НЕМАЄ" in n]

    assert len(fresh) == 1
    assert "39 з 40" in fresh[0]


def test_few_fresh_cases_are_named(tmp_path: Path, monkeypatch) -> None:
    net = _Net()
    _patch(monkeypatch, net)

    report = preflight.check_plan(_plan(tmp_path, 2))
    fresh = [n for n in report["notes"] if "чекпоінтів ЩЕ НЕМАЄ" in n]

    assert [n.split(":")[0] for n in fresh] == ["c0000", "c0001"]
