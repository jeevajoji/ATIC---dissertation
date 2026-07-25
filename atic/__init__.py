"""ATIC package with lazy model imports.

Keeping the top-level package lightweight allows bitstream inspection and
corruption checks on machines that do not have the GPU training stack.
"""

from typing import TYPE_CHECKING

__all__ = ["ArchitectureConfig", "ATICModel"]

if TYPE_CHECKING:
    from atic.config import ArchitectureConfig
    from atic.model import ATICModel


def __getattr__(name):
    if name == "ArchitectureConfig":
        from atic.config import ArchitectureConfig

        return ArchitectureConfig
    if name == "ATICModel":
        from atic.model import ATICModel

        return ATICModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
