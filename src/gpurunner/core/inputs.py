"""Resolution of input directories for the Kaggle-FS-emulating backends.

Colab, Vast, Lightning, Beam and Saturn all stage local directories into
``/kaggle/input/<slug>/`` so that a job's ``render_remote_code()`` finds its data
exactly where the Kaggle backend would have mounted it. The contract for telling
them *which* local directory maps to which slug is identical everywhere, so it
lives here instead of being copy-pasted per backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpurunner.core.backend import BackendError
from gpurunner.core.job import Job


def resolve_inputs(
    job: Job,
    params: dict[str, Any],
    *,
    inputs: Any = None,
    input_root: Any = None,
) -> dict[str, Path]:
    """``{slug: local dir}`` to upload into ``/kaggle/input/<slug>/``.

    Explicit ``-p inputs='{"slug": "E:/path"}'`` wins. Otherwise the job's declared
    inputs (``colab_input_dirs``, which most jobs get for free from
    ``dataset_sources``) are looked up under ``-p input_root=<dir>``.
    """
    if inputs:
        if not isinstance(inputs, dict):
            raise BackendError('-p inputs must be a JSON object: {"slug": "/local/dir"}')
        resolved = {slug: Path(str(p)) for slug, p in inputs.items()}
    else:
        declared = job.colab_input_dirs(params)
        if not declared:
            return {}
        if not input_root:
            raise BackendError(
                f"job needs inputs {sorted(declared)} — pass -p input_root=<dir> (each slug is a "
                'subdir) or -p inputs=\'{"slug": "/local/dir"}\''
            )
        resolved = {slug: Path(str(input_root)) / name for slug, name in declared.items()}

    for slug, path in resolved.items():
        if not path.is_dir():
            raise BackendError(f"input {slug!r}: {path} is not a directory")
    return resolved
