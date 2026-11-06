"""Чекпоінти раннера не вичерпуються: ковзний останній слот (`_CkptSink`).

23.09.2026 книга ЦДІАК spr-68 (1 308 сторінок): план дав 16 слотів, раннер
спалив їх за ~32 хв, і до кінця заходу чекпоінтів не було — 589 прочитаних
сторінок лишились лише на диску машини, яку знищили.

Раннер живе на Linux-шляхах (`/tmp/htrcase`), тож тест — для Linux (CI;
локально — WSL).
"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="бокс-раннер живе на Linux-шляхах")

EMB = Path(__file__).resolve().parents[1] / "src" / "gpurunner" / "_embedded"


def _runner() -> dict:
    src = (EMB / "_common.py").read_text(encoding="utf-8") + "\n" + (
        EMB / "htr_case_runner.py").read_text(encoding="utf-8").replace(
        'if __name__ == "__main__":\n    main({})', "")
    ns: dict = {"__name__": "runner_under_test"}
    exec(compile(src, "htr_case_runner.py", "exec"), ns)
    return ns


def _restore(slots: list[Path], dest: Path) -> None:
    """Як відновлення на новій машині: наявні архіви по порядку, з перезаписом."""
    for ball in slots:
        if ball.is_file():
            with tarfile.open(ball) as tf:
                tf.extractall(dest, filter="data")


def _texts(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted(root.rglob("*.txt"))}


@pytest.mark.parametrize("start", [0, 3])
def test_slots_run_out_but_checkpoints_do_not(tmp_path: Path, start: int) -> None:
    """Слотів 3, раундів 8: відновлення зі сховища дає ВСЕ прочитане.

    `start=3` — друга оренда після першої, що вже зайняла всі слоти.
    """
    ns = _runner()
    work, store = tmp_path / "work", tmp_path / "store"
    (work / "out").mkdir(parents=True)
    store.mkdir()
    slots = [store / f"ckpt_{i:04d}.tgz" for i in range(1, 4)]
    sink = ns["_CkptSink"](work, [s.as_uri() for s in slots], start)
    for rnd in range(1, 9):
        for k in range(5):
            (work / "out" / f"{rnd:02d}{k}.txt").write_text(f"р{rnd}", encoding="utf-8")
        assert sink.push() == 5, f"раунд {rnd}: нові сторінки не залиті"
    # переписана сторінка (дочитування) теж мусить доїхати
    (work / "out" / "010.txt").write_text("виправлено", encoding="utf-8")
    assert sink.push() == 1
    assert sink.push() == 0, "раунд без нового не пише нічого"
    assert not sink.stuck
    assert sink.tail_rewrites > 0

    back = tmp_path / "back"
    _restore(slots, back)
    assert _texts(back) == _texts(work)


def test_dead_links_are_visible_as_stuck(tmp_path: Path) -> None:
    ns = _runner()
    work = tmp_path / "work"
    (work / "out").mkdir(parents=True)
    dead = (tmp_path / "немає" / "теки" / "ckpt_0001.tgz").as_uri()
    sink = ns["_CkptSink"](work, [dead], 0)
    for rnd in range(ns["_CkptSink"].FAIL_STREAK_STUCK):
        (work / "out" / f"{rnd}.txt").write_text("т", encoding="utf-8")
        assert sink.push() == 0
    assert sink.stuck
    health: dict = {}
    ns["_ckpt_health"](health, [sink])
    assert health["exhausted"] is True, "наглядач мусить побачити, що точки відновлення немає"


def test_the_runner_answers_a_checkpoint_request_within_seconds(tmp_path: Path) -> None:
    """Наглядач перед забором просить чекпоінт ЗАРАЗ; раунд не чекає 120 с."""
    import threading
    import time

    ns = _runner()
    ns["CKPT_NOW"] = tmp_path / "ckpt_now"
    ns["CKPT_DONE"] = tmp_path / "ckpt_done"
    ns["CKPT_PID"] = tmp_path / "ckpt.pid"
    ns["CKPT_POLL_SEC"] = 0.1
    pushed = []
    stop = threading.Event()
    t = threading.Thread(target=ns["_checkpoint_loop"],
                         args=(stop, 3600, lambda: pushed.append(time.time())), daemon=True)
    t.start()
    time.sleep(0.3)
    assert pushed == [], "без прохання й до строку раунду немає"
    assert (tmp_path / "ckpt.pid").is_file()
    (tmp_path / "ckpt_now").touch()
    deadline = time.time() + 5
    while not (tmp_path / "ckpt_done").exists() and time.time() < deadline:
        time.sleep(0.05)
    stop.set()
    t.join(2)
    assert (tmp_path / "ckpt_done").exists()
    assert len(pushed) == 1
    assert not (tmp_path / "ckpt_now").exists()


def test_the_heartbeat_goes_to_the_store_with_the_log(tmp_path: Path) -> None:
    """Без наглядача з серцебиття видно, що бокс робив і коли востаннє жив."""
    import io
    import json

    ns = _runner()
    root = tmp_path / "gpurunner"
    root.mkdir()
    (root / "_runner.log").write_text("[htr-case] справа 1 закрита\n", encoding="utf-8")
    ns["PROGRESS_PATH"] = root / "_progress.json"
    target = tmp_path / "store" / "beat.tgz"
    target.parent.mkdir()
    ns["_BEAT_URL"][0] = target.as_uri()
    ns["_BOX_PAGES"][0] = 42
    assert ns["_beat_to_store"](final=True)
    with tarfile.open(target) as tf:
        beat = json.load(io.TextIOWrapper(tf.extractfile("beat.json"), encoding="utf-8"))
        log = tf.extractfile("_runner.log").read().decode("utf-8")
    assert beat["final"] is True and beat["box_pages_done"] == 42 and beat["t"] > 0
    assert "справа 1 закрита" in log
