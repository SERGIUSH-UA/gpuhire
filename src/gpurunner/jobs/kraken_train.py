"""KrakenTrainJob — ketos fine-tune розпізнавача (McCATMuS → CHURRO-дистилят).

Кейс: власна kraken-модель на псевдо-GT з CHURRO (ф.792, польський
скоропис). Вхід — прекомпільовані arrow-датасети (``ketos compile -f path
--force-type baseline`` робиться ЛОКАЛЬНО — у kraken 7 train приймає лише
binary) + базова модель *.mlmodel. Вихід — ``kraken_churro_best.mlmodel``
(load_any-сумісна) + ваги, лог і крива val_accuracy.

**Продовження перерваного трену — теплим стартом, не resume.** ``ketos train``
не має ``--resume``, і стан оптимізатора нікуди не зберігається, тож повного
продовження не існує в принципі. Робочий рецепт: забрати
``kraken_churro_best.mlmodel``, залити його датасетом і запустити знову з
``-p model_dataset=<slug>``. Раннер конвертує найкращу епоху в ``.mlmodel``
навіть коли трен обірвано по wall-limit, саме щоб цей шлях завжди був
доступний; рядок ``continue_with`` у ``ktrain_summary.json`` містить готову
команду. Для повноцінного resume (з моментами AdamW і позицією в lr-циклі)
див. ``parseq_train`` — там власний цикл, тому це можливо.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job


class KrakenTrainJob(Job):
    """Ketos recognition fine-tune on precompiled arrow datasets."""

    name: ClassVar[str] = "kraken_train"
    description: ClassVar[str] = (
        "Fine-tune a kraken recognition model (ketos train -f binary) from "
        "precompiled train/val arrow datasets, convert best epoch to .mlmodel."
    )
    supported_backends: ClassVar[tuple[str, ...]] = (
        "kaggle",
        "colab",
        "vast",
        "lightning",
        "beam",
        "saturn",
    )

    def requirements(self) -> list[str]:
        return ["kraken>=5.0"]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if "dataset" not in params:
            raise ValueError("kraken_train needs 'dataset=<owner>/<slug>' (arrow-база)")
        dataset = str(params["dataset"]).strip()
        model_dataset = str(params.get("model_dataset", "")).strip()
        out: dict[str, Any] = {
            "dataset": dataset,
            "model_dataset": model_dataset,
            "epochs": int(params.get("epochs", -1)),
            "min_epochs": int(params.get("min_epochs", 7)),
            # 🔴 lag 3, НЕ 10. З lag=10 трен `928fc985` намотав 27 епох замість
            # ~12 і згорів на wall-limit. Плато на нашому матеріалі видно за
            # 2-3 епохи — довше чекати означає палити квоту на перенавчання.
            "lag": int(params.get("lag", 3)),
            # 🔴🔴 `--min-delta` у ketos має дефолт 0.0 — і ми його НІКОЛИ не
            # передавали, тобто рання зупинка фактично НЕ ПРАЦЮВАЛА: покращення
            # на 0.0009 (≈4 символи на val із 4529) скидало лічильник lag назад
            # у 0. Через це трен `50adf3c6` на явному плато (val стоїть від 3-ї
            # епохи, train_loss падає — класичне перенавчання) не зупинявся й
            # ішов до wall-limit, спаливши 2.6 год квоти намарно.
            # 0.002 = «покращення менше за 0.2 пп не рахуємо». Перевірено на
            # реальній кривій 50adf3c6: з (lag=3, min_delta=0.002) зупинка
            # настала б на ep9 — 1.9 год замість 4.5.
            "min_delta": float(params.get("min_delta", 0.002)),
            # 🔴 самозупинка за N годин: `-N` епохи НЕ обмежує, а обрив по
            # 12-годинному ліміту Kaggle лишає кернел без конвертації.
            # 0 = вимкнено.
            # 🔴🔴 ДЕФОЛТ 3.0, а не 11.0. На 2×T4 епоха kraken на нашому корпусі
            # (105k рядків) — 11 хв, тобто 3 год = 16 епох, а плато настає на
            # 6-9. Стеля 11 год ЗАОХОЧУВАЛА палити квоту: трен їхав годинами
            # після того, як перестав учитись. Разом із (lag=3, min_delta=0.002)
            # це подвійна засувка: спершу зупиняє early stopping, wall-limit —
            # лише страховка. Піднімати свідомо і лише під більший корпус.
            "wall_limit_h": float(params.get("wall_limit_h", 3.0)),
            "lr": float(params.get("lr", 1e-4)),
            "batch": int(params.get("batch", 16)),
            "resize": str(params.get("resize", "union")),
            "precision": str(params.get("precision", "16-mixed")),
            "workers": int(params.get("workers", 2)),
            # `auto` = задіяти ВСІ видимі карти (`-d cuda:0,cuda:1` → Lightning
            # DDP). До 2026-08-01 раннер хардкодив cuda:0 з коментарем «ketos
            # уміє лише одну» — це неправда: `-d` без валідатора, а
            # `to_ptl_device` розбирає список через кому. На Kaggle T4×2 через
            # це половина заліза простоювала кожен трен. Явне «cuda:0» лишає
            # старий однокартковий режим (діагностика, відтворення старих числ).
            "devices": str(params.get("devices", "auto")).strip() or "auto",
            # 🔴 Віддавати ваги КОЖНОЇ епохи, а не лише найкращу за val_accuracy
            # (вимога дослідника 2026-08-08, за зразком `parseq_train`). Підстава
            # виміряна на Писарі двічі: локальний відбір епохи по holdout
            # розійшовся з вибором трену і дав +1.4 пп recall у v12 та +2.3 пп
            # у v14. Для Дяка це важить ще більше — його `val` це 143 рядки, у
            # яких 46 символів корпусу не трапляються ЖОДНОГО разу
            # («alphabet mismatch» у лозі), тобто «найкраща епоха» обирається
            # метрикою, яка частину алфавіту не бачить.
            # Ціна — ~16 МБ × число епох у виході (25 епох ≈ 400 МБ).
            "all_epochs": bool(params.get("all_epochs", True)),
            # VGSL-топологія для тренування З НУЛЯ (коли model_dataset не заданий).
            # Дефолт — стандартний рецепт kraken для рядкового розпізнавання;
            # висота 120 узгоджена з нарізкою наших кропів.
            "spec": str(params.get("spec", "")).strip(),
        }
        if not out["model_dataset"] and not out["spec"]:
            out["spec"] = (
                "[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 "
                "Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 "
                "Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]"
            )
        if out["resize"] not in ("add", "union", "both", "new", "fail"):
            raise ValueError(f"resize must be one of add/union/both/new/fail, got {out['resize']!r}")
        if out["batch"] < 1:
            raise ValueError("batch must be >= 1")
        # Нижче — перевірки того, що ketos приймає мовчки, а падає (або тихо
        # робить не те) вже на кернелі, за 3 хвилини після старту оплаченої сесії.
        if out["precision"] not in ("32", "32-true", "16-mixed", "bf16-mixed", "64"):
            raise ValueError(
                f"precision must be one of 32/32-true/16-mixed/bf16-mixed/64, "
                f"got {out['precision']!r}"
            )
        if out["workers"] < 0:
            raise ValueError("workers must be >= 0")
        if out["epochs"] < -1 or out["epochs"] == 0:
            raise ValueError("epochs must be -1 (без ліміту) or >= 1")
        if out["min_epochs"] < 0:
            raise ValueError("min_epochs must be >= 0")
        if out["lag"] < 0:
            raise ValueError("lag must be >= 0")
        # Від'ємний ліміт вмикав guard і зупиняв трен на першій же перевірці —
        # тобто «менше нуля» тихо означало «зупинись негайно». 0 = вимкнено.
        if out["wall_limit_h"] < 0:
            raise ValueError("wall_limit_h must be >= 0 (0 = вимкнено)")
        if out["lr"] <= 0:
            raise ValueError("lr must be > 0")
        if out["min_delta"] < 0:
            raise ValueError("min_delta must be >= 0")
        return out

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        return timedelta(hours=6)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return ["kraken_churro_best.mlmodel", "ktrain_summary.json", "ktrain.log"]

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        normalized = self.validate_params(params)
        out = [r for r in (normalized["dataset"], normalized["model_dataset"])
               if r and "/" in r]
        return out

    def colab_input_dirs(self, params: dict[str, Any]) -> dict[str, str]:
        """Приймає і '<owner>/<slug>' (Kaggle), і голу назву теки (lightning/vast)."""
        normalized = self.validate_params(params)
        out: dict[str, str] = {}
        for ref in (normalized["dataset"], normalized["model_dataset"]):
            if ref:
                slug = ref.split("/versions/", 1)[0].split("/")[-1]
                out[slug] = slug
        return out

    def render_remote_code(self, params: dict[str, Any], *, shard_index: int = 0,
                           total_shards: int = 1) -> str:
        normalized = self.validate_params(params)
        if total_shards > 1:
            raise ValueError("kraken_train doesn't support sharding")
        return self.render_kaggle_code("kraken_train_runner.py", normalized)
