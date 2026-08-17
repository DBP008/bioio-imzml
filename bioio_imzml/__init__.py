"""Top-level package for bioio_imzml."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bioio-imzml")
except PackageNotFoundError:
    __version__ = "uninstalled"

from .peak_picking import PeakPickingResult, auto_pick_peaks
from .reader import Reader
from .reader_metadata import ReaderMetadata

__all__ = ["PeakPickingResult", "Reader", "ReaderMetadata", "auto_pick_peaks"]
