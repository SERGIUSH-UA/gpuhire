"""Concrete Backend implementations."""

from gpurunner.backends.beam import BeamBackend
from gpurunner.backends.colab import ColabBackend
from gpurunner.backends.kaggle import KaggleBackend
from gpurunner.backends.lightning import LightningBackend
from gpurunner.backends.modal import ModalBackend
from gpurunner.backends.saturn import SaturnBackend
from gpurunner.backends.vast import VastBackend
from gpurunner.core.backend import Backend

__all__ = [
    "BeamBackend",
    "ColabBackend",
    "KaggleBackend",
    "LightningBackend",
    "ModalBackend",
    "SaturnBackend",
    "VastBackend",
]

#: Every backend name the CLI knows, in the order they are listed to the user.
BACKEND_NAMES: tuple[str, ...] = (
    "kaggle",
    "modal",
    "colab",
    "vast",
    "lightning",
    "beam",
    "saturn",
)


def get_backend(name: str) -> type[Backend]:
    """Resolve a backend name to a class. Raises KeyError on unknown."""
    table: dict[str, type] = {
        "kaggle": KaggleBackend,
        "modal": ModalBackend,
        "colab": ColabBackend,
        "vast": VastBackend,
        "lightning": LightningBackend,
        "beam": BeamBackend,
        "saturn": SaturnBackend,
    }
    if name not in table:
        raise KeyError(
            f"Unknown backend {name!r}. Available: {sorted(table.keys())}"
        )
    return table[name]
