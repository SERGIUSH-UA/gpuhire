"""Справи в черзі не мають бачити кадри одна одної.

🔴🔴 Найдорожча знахідка аудиту 2026-08-11, і не грошима. Тека завантаження
була посправною (`pages_dl_%02d`), а РОЗПАКУВАННЯ йшло в константу
`/tmp/htrcase/pages` без прибирання. `htr_cloud_plan.py` пакує кадри пласко
(`arcname=frame.name`), а імена кадрів у різних справах однакові
(`0001_L.jpg`) — тож справа 2 перезаписувала частину кадрів справи 1 і
успадковувала її хвіст як свій. Тексти чужої справи лягали в її теку під її ж
іменами; знаменник рахується з тієї самої теки, тож усі три ворота повноти
казали «повно». Ціна — хибна прив'язка аркуша, тобто рівень цитати.
"""

from __future__ import annotations

import ast
from pathlib import Path

RUNNER = (Path(__file__).resolve().parents[1]
          / "src" / "gpurunner" / "_embedded" / "htr_case_runner.py")
SRC = RUNNER.read_text(encoding="utf-8")


def test_unpack_dir_is_per_case() -> None:
    assert '"_pages_unpack"] = "/tmp/htrcase/pages_%02d"' in SRC, (
        "тека розпакування мусить нести номер справи"
    )


def test_runner_reads_the_per_case_unpack_dir() -> None:
    assert 'params.get("_pages_unpack")' in SRC


def test_unpack_dir_is_wiped_before_extract() -> None:
    """Резерв на випадок, коли тека лишилась від попереднього запуску."""
    i = SRC.index('params.get("_pages_unpack")')
    window = SRC[i:i + 600]
    assert "rmtree" in window, "перед розпакуванням тека мусить прибиратись"


def test_download_and_unpack_dirs_do_not_collide() -> None:
    """Обидві теки посправні й РІЗНІ — інакше tgz потрапив би у власний вихід."""
    assert '"/tmp/htrcase/pages_dl_%02d"' in SRC
    assert '"/tmp/htrcase/pages_%02d"' in SRC


def test_line_boxes_survive_a_re_rent() -> None:
    """`.lines.json` більше не викидається з чекпоінтів.

    Інакше після смерті боксу нова оренда бачить сторінку в меті як зроблену,
    пропускає її — і рамки для вже зробленої половини справи не породжуються
    ніколи, а ворота повноти (вони рахують лише `*.txt`) кажуть «повно».
    """
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "CKPT_SKIP_SUFFIXES" for t in node.targets
        ):
            assert ast.literal_eval(node.value) == ()
            return
    raise AssertionError("CKPT_SKIP_SUFFIXES не знайдено")


def test_queue_case_does_not_publish_a_terminal_phase() -> None:
    """Неповна справа №1 із трьох не має гасити бокс посеред черги."""
    assert 'fleet["phase"] = "failed" if final else "running"' in SRC


def test_downloads_have_a_time_ceiling() -> None:
    """Завислий curl тарифікується до стелі годин, а серцебиття це маскує."""
    assert SRC.count("--max-time") >= 2


# ---- рятування при обриві ----------------------------------------------------


def test_checkpoint_numbering_continues_after_resume() -> None:
    """🔴 Нова оренда починала нумерацію з `ckpt_0001.tgz` — поверх першого
    архіву попередньої серії.

    Поки resume спрацював, це безпечно (перший чекпоінт нового боксу містить
    усе відновлене, тобто є надмножиною). Але коли resume впав ЧАСТКОВО —
    протухле посилання, битий архів — новий `ckpt_0001` уже НЕ надмножина, і
    він знищує базу попереднього прогону остаточно. Тобто рятувальний
    механізм ламався рівно в тому випадку, заради якого існує.
    """
    assert 'ck_state = {"rounds": int(params.get("_ckpt_start") or 0)' in SRC
    assert 'params["_ckpt_start"] = last_present' in SRC


def test_last_present_counts_existing_archives_not_extracted_ones() -> None:
    """Лічильник рахує, що в хмарі ІСНУЄ, а не що вдалось розпакувати.

    Битий архів усе одно займає свій номер: писати поверх нього не можна.
    """
    i = SRC.index("last_present = i")
    # присвоєння стоїть ДО try/except розпакування
    assert SRC.index("tf.extractall(work)", i) > i


def test_empty_round_does_not_burn_a_presigned_slot() -> None:
    """Нема чого лити — нема чого й нумерувати: інакше тиша вичерпує посилання."""
    assert 'if n:\n            ck_state["rounds"] = nxt' in SRC


def test_line_boxes_are_checkpointed_so_a_re_rent_keeps_them() -> None:
    """Рамки рядків мусять їхати в чекпоінт — інакше після переоренди їх не
    буде для вже зробленої половини справи, а ворота повноти (лише `*.txt`)
    цього не помітять."""
    assert "CKPT_SKIP_SUFFIXES = ()" in SRC


# ---- самолікування при OOM ----------------------------------------------------


def test_oom_shrinks_the_fleet_instead_of_failing_pages() -> None:
    """🔴🔴 OOM лише РАХУВАВСЯ, і це робило лікування дорожчим за хворобу.

    Шард, що вперся у VRAM, падав на кожній наступній сторінці до кінця своєї
    частки, а «лікуванням» ставав догінний прохід, який переганяв усе
    пропущене ОДНИМ шардом. На справі 2026-08-12: 70% сторінок у збоях, потім
    ті самі 70% серіалізовано — найдорожчий можливий спосіб.

    Тепер флот звужується НА ХОДУ: зроблене лежить на диску й пропускається,
    тож ніщо не рахується двічі.
    """
    assert "OOM_SHRINK_TRIGGER" in SRC and "OOM_SHRINK_MAX" in SRC
    assert "звужую флот" in SRC
    i = SRC.index("oom_shrinks += 1")
    assert "max(1, shards - max(1, shards // 3))" in SRC[i:i + 400]


def test_shrink_is_bounded_and_never_goes_below_one() -> None:
    """Звуження не має крутитись вічно й не має дійти до нуля шардів."""
    i = SRC.index("if oom < OOM_SHRINK_TRIGGER")
    cond = SRC[i:i + 260]
    assert "shards <= 1" in cond and "oom_shrinks >= OOM_SHRINK_MAX" in cond


def test_no_shrink_when_there_is_nothing_left_to_do() -> None:
    """Якщо сторінок не лишилось, звужувати нема сенсу — це вже кінець роботи."""
    i = SRC.index("if oom < OOM_SHRINK_TRIGGER")
    assert "or not left" in SRC[i:i + 260]


def test_single_shard_oom_is_named_not_silently_retried() -> None:
    """OOM на одному шарді означає, що справі потрібна інша карта — і це
    мусить бути сказано, а не сховано в лічильнику."""
    assert "звужувати нікуди" in SRC
