from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pyimzml.ImzMLParser import SIZE_DICT

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem
    from pyimzml.ImzMLParser import PortableSpectrumReader

###############################################################################


def find_ibd_path(fs: "AbstractFileSystem", imzml_path: str) -> str:
    """Sibling `.ibd` file for an `.imzML` path (same base name, extension
    case-insensitive, as allowed by the imzML spec).
    """
    base_name = Path(imzml_path).stem
    parent = str(Path(imzml_path).parent)
    want = f"{base_name}.ibd".lower()
    for entry in fs.ls(parent, detail=False):
        if Path(entry).name.lower() == want:
            return entry
    raise FileNotFoundError(
        f"No sibling .ibd file found for '{imzml_path}' (expected '{base_name}.ibd')."
    )


def nearest_intensities(
    mzs: np.ndarray,
    intensities: np.ndarray,
    targets: np.ndarray,
    tolerance: float | None = None,
) -> np.ndarray:
    """Intensity at the nearest measured m/z to each value in `targets`.

    `mzs` must be sorted ascending, as the imzML spec requires. If
    `tolerance` is given (in the same units as `targets`, i.e. m/z), a target
    with no measured peak within that distance gets 0 instead of the (too
    distant) nearest peak's intensity.
    """
    if len(mzs) == 0:
        return np.zeros(len(targets), dtype=np.float32)
    idx = np.clip(np.searchsorted(mzs, targets), 0, len(mzs) - 1)
    idx_prev = np.clip(idx - 1, 0, len(mzs) - 1)
    use_prev = np.abs(mzs[idx_prev] - targets) < np.abs(mzs[idx] - targets)
    nearest = np.where(use_prev, idx_prev, idx)
    result = intensities[nearest]

    if tolerance is not None:
        diff = np.abs(mzs[nearest] - targets)
        result = np.where(diff <= tolerance, result, 0.0)

    return result


def scan_mz_bounds(portable: "PortableSpectrumReader", ibd_file) -> tuple[float, float]:
    """Global (min, max) m/z across every spectrum.

    Reads only the first and last value of each spectrum's m/z array, relying on
    the imzML spec's "increasing m/z scan" ordering, so this is cheap even for
    datasets with hundreds of thousands of spectra.
    """
    lo = np.inf
    hi = -np.inf
    step = SIZE_DICT[portable.mzPrecision]
    dtype = np.dtype(portable.mzPrecision)
    for offset, length in zip(portable.mzOffsets, portable.mzLengths):
        if length == 0:
            continue
        ibd_file.seek(offset)
        first = np.frombuffer(ibd_file.read(step), dtype=dtype)[0]
        ibd_file.seek(offset + (length - 1) * step)
        last = np.frombuffer(ibd_file.read(step), dtype=dtype)[0]
        lo = min(lo, first)
        hi = max(hi, last)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Could not determine an m/z range: file has no spectra.")
    return float(lo), float(hi)
