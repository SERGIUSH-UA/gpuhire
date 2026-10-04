"""Job ABC: a unit of work that can be packaged and submitted to a Backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import Any, ClassVar


class Job(ABC):
    """Abstract job description.

    A concrete subclass defines:
      - what packages it needs at runtime (`requirements`)
      - which backends can run it (`supported_backends`)
      - how to validate user-supplied params
      - how to render the Python code that runs remotely

    The Backend turns this into something it can submit (Kaggle: a notebook +
    metadata; Modal: a deployed function). The Job itself stays backend-agnostic.
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""
    # "colab", "vast", "lightning", "beam" and "saturn" all run the same rendering
    # as "kaggle" (each emulates /kaggle/{input,working}), so anything
    # Kaggle-capable works there by construction. Only "modal" needs its own
    # runner module.
    supported_backends: ClassVar[tuple[str, ...]] = (
        "kaggle",
        "modal",
        "colab",
        "vast",
        "lightning",
        "beam",
        "saturn",
    )

    @abstractmethod
    def requirements(self) -> list[str]:
        """pip packages the remote runtime needs (`paddlepaddle-gpu`, etc)."""

    @abstractmethod
    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate + normalize user params. Raise ValueError on bad input."""

    @abstractmethod
    def render_remote_code(
        self,
        params: dict[str, Any],
        *,
        shard_index: int = 0,
        total_shards: int = 1,
    ) -> str:
        """Return Python source that will execute remotely.

        The code must:
          - import its deps (already pip-installed by the runtime setup)
          - read its inputs from `params` (passed as constants by the renderer)
          - write outputs into the runtime's working dir (Kaggle: /kaggle/working/,
            Modal: returned via function value)
        """

    # ---- rendering helpers ------------------------------------------------
    # Every Kaggle-shaped job builds the same three-part string: injected PARAMS,
    # the embedded runner's source, a trailing `main(PARAMS)`. That template was
    # copy-pasted into 13 jobs, each free to drift — and one had already dropped
    # the `__main__` strip. It lives here now; jobs pass their runner's filename.

    @staticmethod
    def _params_block(params: dict[str, Any]) -> str:
        """`PARAMS = ...` line that survives any value the user can type.

        🔴 The old form was ``json.loads(r'''<json>''')``, which is a syntax
        error the moment a value contains ``'''`` — reachable through
        ``-p charset_extra=…`` (parseq) and ``-p spec=…`` (kraken), both free
        text. ``repr`` of the JSON string escapes quotes and backslashes alike,
        so there is no input left that can break the generated module.
        """
        import json

        return f"PARAMS = __import__('json').loads({json.dumps(params, ensure_ascii=False)!r})"

    @staticmethod
    def _runner_source(filename: str, *, strip_main: bool = True) -> str:
        """Read an embedded runner's source, optionally dropping its __main__ block.

        The block must go: in a Kaggle cell ``__name__`` *is* ``"__main__"``, so a
        surviving guard would run the runner's hardcoded smoke params before the
        injected ``main(PARAMS)`` ever fires.
        """
        from importlib import resources

        src = resources.files("gpurunner._embedded").joinpath(filename).read_text(encoding="utf-8")
        if strip_main:
            src = src.split('\nif __name__ == "__main__":', 1)[0]
        return src

    def render_kaggle_code(
        self,
        runner_filename: str,
        params: dict[str, Any],
        *,
        prelude: str = "",
        entry: str = "main(PARAMS)",
        strip_main: bool = True,
        common: bool = True,
    ) -> str:
        """Assemble the standard injected-params + embedded-runner module.

        ``prelude`` lands between PARAMS and the runner (dino uses it to fake
        ``sys.argv`` for its argparse-style runner); ``entry`` is the trailing
        call, and jobs whose runner *is* an argparse CLI pass ``strip_main=False``
        plus an empty ``entry``.

        ``common`` prepends ``_embedded/_common.py``. It goes *before* the runner
        on purpose: a runner that still carries its own copy of a shared helper
        simply overrides it, so switching this on breaks nothing and the copies
        can be retired one runner at a time.
        """
        parts = [
            "# --- gpurunner injected params ---",
            self._params_block(params),
            "",
        ]
        if common:
            parts += [
                "# --- gpurunner common prelude ---",
                self._runner_source("_common.py", strip_main=False),
                "",
            ]
        if prelude:
            parts += [prelude.rstrip("\n"), ""]
        parts += [
            "# --- embedded runner ---",
            self._runner_source(runner_filename, strip_main=strip_main),
            "",
        ]
        if entry:
            parts += ["# --- entry point ---", entry, ""]
        return "\n".join(parts)

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        """Rough estimate so Backend can pick GPU/shard count. Override if useful."""
        return timedelta(hours=1)

    def recommended_shards(self, params: dict[str, Any], *, max_per_shard: timedelta) -> int:
        """How many shards to split this job into. Default = single shard."""
        est = self.estimate_runtime(params)
        if est <= max_per_shard:
            return 1
        # ceil division
        n = int(est.total_seconds() // max_per_shard.total_seconds())
        if est.total_seconds() % max_per_shard.total_seconds() > 0:
            n += 1
        return max(1, n)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        """Filename glob patterns the runtime is expected to produce. Optional hint."""
        return ["*"]

    def is_output_complete(self, out_dir: Path) -> bool:
        """Heuristic used by ``fetch --resume`` to skip already-fetched handles.

        Default: any regular file under ``out_dir`` counts as "complete enough".
        Override for jobs that have a sentinel file (PaddleOCR writes
        ``_summary.json``) or a known per-item artifact.
        """
        if not out_dir.exists():
            return False
        return any(p.is_file() for p in out_dir.rglob("*"))

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        """Kaggle Dataset slugs to attach to the runtime (mounted at /kaggle/input/<slug>/).

        Each entry is ``<owner>/<dataset-slug>`` (Kaggle Dataset reference). Pinning
        a version: ``<owner>/<slug>/versions/<n>``. Default: no datasets.

        Backend-specific (Kaggle); other backends may ignore.
        """
        return []

    def kernel_sources(self, params: dict[str, Any]) -> list[str]:
        """Kaggle Kernel slugs whose outputs to attach. Default: none. Kaggle-only."""
        return []

    def colab_input_dirs(self, params: dict[str, Any]) -> dict[str, str]:
        """Input data to stage from Drive. Colab-only.

        Returns ``{<slug>: <drive folder name>}``; the ColabBackend copies
        ``MyDrive/gpurunner/data/<drive folder name>/*`` into ``/kaggle/input/<slug>/``
        so that code rendered by ``render_remote_code()`` finds it exactly where the
        Kaggle backend would have mounted it.

        Default derives from ``dataset_sources()`` (``<owner>/<slug>`` → ``<slug>``),
        so a job that already declares Kaggle datasets needs no extra work. Override
        when the job accepts a bare dataset name (Modal-style).
        """
        out: dict[str, str] = {}
        for ref in self.dataset_sources(params):
            slug = ref.split("/versions/", 1)[0].split("/")[-1]
            if slug:
                out[slug] = slug
        return out

    def modal_input_volumes(self, params: dict[str, Any]) -> dict[str, str]:
        """Modal Volume names to mount read-only for input data. Modal-only.

        Returns a dict mapping mount paths (``/mnt/foo``) to existing volume names.
        Used when the job needs to read large input data that's been uploaded to
        a Volume out of band (``modal volume put``). Default: none.
        """
        return {}

    def modal_secrets(self, params: dict[str, Any]) -> list[str]:
        """Modal Secret names to attach as env vars in the runtime. Modal-only.

        Each secret is resolved via ``modal.Secret.from_name(<name>)``. Create
        ahead of time, e.g.:
            modal secret create kaggle-creds KAGGLE_KEY=KGAT_...
        """
        return []

    def modal_image_spec(self) -> dict[str, Any]:
        """Image specification for ModalBackend.

        Returns a dict with keys:
          - ``python_version``: str like ``"3.12"``
          - ``pip_packages``: list of pip package specs
          - ``extra_index_url``: optional extra PyPI index URL
          - ``apt_packages``: list of debian packages (optional)

        Default derives from ``requirements()`` by stripping pip flags.
        """
        reqs = self.requirements()
        pkgs: list[str] = []
        skip_next = False
        extra_index: str | None = None
        for r in reqs:
            if skip_next:
                # ``--extra-index-url`` and similar consume the next token
                if r.startswith("http"):
                    extra_index = r
                skip_next = False
                continue
            if r in ("--extra-index-url", "--index-url"):
                skip_next = True
                continue
            if r.startswith("--"):
                continue
            pkgs.append(r)
        return {
            "python_version": "3.12",
            "pip_packages": pkgs,
            "extra_index_url": extra_index,
            "apt_packages": [],
        }

    def render_runner_module(self) -> str:
        """Self-contained Python source defining ``main(params) -> dict``.

        Used by ModalBackend: this source is sent to the remote function and
        executed there. The remote wrapper invokes ``main(params)`` and returns
        whatever it produces.

        Default: raises. Subclasses that intend to run on Modal must override.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support Modal — "
            "override render_runner_module() to return a module source with main(params)."
        )
