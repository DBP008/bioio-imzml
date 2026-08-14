"""Top-level package for bioio_imzml."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bioio-imzml")
except PackageNotFoundError:
    __version__ = "uninstalled"

from .reader import Reader
from .reader_metadata import ReaderMetadata

__all__ = ["Reader", "ReaderMetadata"]
