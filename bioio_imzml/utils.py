from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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


def local_window_intensities(
    mzs: np.ndarray,
    intensities: np.ndarray,
    targets: np.ndarray,
    tolerance: np.ndarray | float | None = None,
    agg: Literal["nearest", "sum"] = "sum",
) -> np.ndarray:
    """Intensity at each value in `targets`, matched against measured peaks
    within a local window around each target.

    `mzs` must be sorted ascending, as the imzML spec requires. If
    `tolerance` is given (in the same units as `targets`, i.e. m/z; scalar or
    one value per target), it bounds the matching window around each target.

    `agg="sum"` (default): the sum of every measured peak's intensity within
    `[target - tolerance, target + tolerance]`; `tolerance=None` means a
    zero-width window, i.e. only an exact m/z match counts.

    `agg="nearest"`: the nearest measured peak's intensity instead of a sum:
    a target with no measured peak within `tolerance` gets 0 instead of the
    (too distant) nearest peak's intensity.
    """
    if len(mzs) == 0:
        return np.zeros(len(targets), dtype=np.float32)

    if agg == "sum":
        tol = 0.0 if tolerance is None else tolerance
        lo = np.searchsorted(mzs, targets - tol, side="left")
        hi = np.searchsorted(mzs, targets + tol, side="right")
        cumsum = np.concatenate(([0.0], np.cumsum(intensities)))
        return (cumsum[hi] - cumsum[lo]).astype(np.float32)

    idx = np.clip(np.searchsorted(mzs, targets), 0, len(mzs) - 1)
    idx_prev = np.clip(idx - 1, 0, len(mzs) - 1)
    use_prev = np.abs(mzs[idx_prev] - targets) < np.abs(mzs[idx] - targets)
    nearest = np.where(use_prev, idx_prev, idx)
    result = intensities[nearest]

    if tolerance is not None:
        diff = np.abs(mzs[nearest] - targets)
        result = np.where(diff <= tolerance, result, 0.0)

    return result


def mz_tolerance_window(
    mz_axis: np.ndarray,
    absolute: float | None,
    relative: float | None,
) -> np.ndarray:
    """Per-channel tolerance combining an absolute and a relative component:
    `tolerance = absolute + m/z * relative`. Both are in the same units as
    `mz_axis` (m/z); e.g. for a 3 ppm relative component pass
    `relative=3e-6`. Either component may be None (treated as 0); with both
    None every value is 0.
    """
    mz_axis = np.asarray(mz_axis, dtype=np.float64)
    return (absolute or 0.0) + mz_axis * (relative or 0.0)


def estimate_mz_tolerance(mz_axis: np.ndarray) -> np.ndarray:
    """Auto-estimated per-channel tolerance when the caller sets neither
    tolerance component: half the distance to each channel's nearest
    neighboring target, so adjacent channels' windows never overlap. A
    channel with no neighbor (a single target) gets an unbounded tolerance
    (no filtering).

    `mz_axis` must be sorted ascending.
    """
    n = len(mz_axis)
    if n <= 1:
        return np.full(n, np.inf, dtype=np.float64)
    gaps = np.diff(mz_axis)
    gap_to_left = np.concatenate(([np.inf], gaps))
    gap_to_right = np.concatenate((gaps, [np.inf]))
    return np.minimum(gap_to_left, gap_to_right) / 2.0


def tolerance_decimal_places(tol: float, sig_figs: int = 3, default: int = 4) -> int:
    """Decimal places needed to show `tol` with at most `sig_figs`
    significant digits, e.g. 0.0055 -> 5 (for sig_figs=3). Falls back to
    `default` decimals when `tol` isn't a positive finite number (zero, inf,
    nan), where "significant digits" isn't a meaningful concept.
    """
    if not np.isfinite(tol) or tol <= 0:
        return default
    exponent = int(f"{tol:.{sig_figs - 1}e}".split("e")[1])
    return max(0, sig_figs - 1 - exponent)


def parse_creation_date(value: str | None) -> datetime | None:
    """Parse an mzML `<run startTimeStamp>` value into a `datetime`.

    The mzML spec types this as `xsd:dateTime` (ISO 8601), but some vendor
    converters (e.g. RAW2IMZML) write it as `MM/DD/YYYY HH:MM:SS AM/PM`
    instead; both are tried. Returns None if `value` is None or matches
    neither format.
    """
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass
    try:
        return datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p")
    except ValueError:
        return None


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
