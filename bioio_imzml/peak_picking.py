from typing import Any, NamedTuple

import numpy as np
import xarray as xr
from bioio_base import dimensions, types
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks, savgol_filter

from .reader import Reader

###############################################################################


def find_peaks_in_spectrum(
    mz_axis: np.ndarray,
    intensity: np.ndarray,
    *,
    smooth: bool = True,
    savgol_window: int = 7,
    savgol_polyorder: int = 2,
    snr_threshold: float = 3.0,
    min_relative_intensity: float = 0.005,
    min_separation_mz: float = 0.5,
) -> np.ndarray:
    """Candidate peak m/z values in a spectrum, sorted by descending intensity.

    Pure array function -- no file I/O, works on any `(mz_axis, intensity)`
    pair such as the one returned by `mean_spectrum`.

    `snr_threshold` is a prominence threshold as a multiple of the spectrum's
    MAD noise floor; `min_relative_intensity` is a height threshold as a
    fraction of the spectrum's max; `min_separation_mz` is the minimum
    distance between two accepted peaks, converted to samples using the
    median spacing of `mz_axis`.
    """
    if smooth and len(intensity) > savgol_window:
        signal = savgol_filter(intensity, savgol_window, savgol_polyorder)
    else:
        signal = intensity

    noise_level = np.median(np.abs(signal - np.median(signal)))
    step = np.median(np.diff(mz_axis)) if len(mz_axis) > 1 else 1.0
    distance = max(1, int(round(min_separation_mz / step)))

    peak_indices, _ = find_peaks(
        signal,
        height=min_relative_intensity * np.max(signal),
        prominence=snr_threshold * noise_level,
        distance=distance,
    )

    order = np.argsort(signal[peak_indices])[::-1]
    return mz_axis[peak_indices][order]


def mean_spectrum(
    image: types.PathLike,
    *,
    min_mz: float | None = None,
    max_mz: float | None = None,
    bin_width: float = 0.05,
    fs_kwargs: dict[str, Any] = {},
) -> tuple[np.ndarray, np.ndarray]:
    """Mean spectrum across every pixel of an imzML file, via `Reader`.

    For "continuous" mode files this is exact (every pixel already shares
    one m/z axis, and `Reader` reads it directly); `min_mz`/`max_mz` just
    restrict the returned range. For "processed" mode files, `Reader`'s own
    nearest-neighbor channel matching at `bin_width` spacing stands in for
    binning -- the same matching used when actually extracting channels, so
    the candidates this finds line up with what extraction will return.

    Returns
    -------
    mz_axis, mean_intensity : np.ndarray
    """
    reader = Reader(image, mz_step=bin_width, fs_kwargs=fs_kwargs)
    mz_axis = reader.mz_values
    non_channel_dims = [
        d for d in reader.xarray_data.dims if d != dimensions.DimensionNames.Channel
    ]
    mean_intensity = reader.xarray_data.mean(dim=non_channel_dims).values.astype(
        np.float64
    )

    if min_mz is not None or max_mz is not None:
        lo = mz_axis[0] if min_mz is None else min_mz
        hi = mz_axis[-1] if max_mz is None else max_mz
        keep = (mz_axis >= lo) & (mz_axis <= hi)
        mz_axis, mean_intensity = mz_axis[keep], mean_intensity[keep]

    return mz_axis, mean_intensity


def pixel_frequency_and_spatial_chaos(
    image: types.PathLike,
    candidate_mzs: np.ndarray,
    *,
    mz_tolerance_absolute: float | None = None,
    mz_tolerance_relative: float | None = None,
    presence_threshold: float = 1e-3,
    fs_kwargs: dict[str, Any] = {},
) -> tuple[np.ndarray, np.ndarray]:
    """Per-candidate `(pixel_frequency, spatial_chaos)`, one value per entry
    in `candidate_mzs` (no filtering applied here -- see `auto_pick_peaks`).

    Extracts ion images for every candidate at once via `Reader`'s existing
    lazy channel extraction (one pass over the file), instead of re-parsing
    per candidate. A pixel counts towards `pixel_frequency` when its
    intensity exceeds `presence_threshold` times that channel's own peak
    intensity -- profile-mode spectra rarely hit an exact zero, so a strict
    `> 0` check would count floating-point noise as signal everywhere.
    `spatial_chaos` is 0 (structured) to 1 (spatially random), estimated as
    the mean absolute difference between each channel's ion image and a
    3x3-smoothed version of itself, relative to its mean.

    Call this directly (instead of `auto_pick_peaks`) to inspect why a
    candidate would be kept or dropped before committing to thresholds.
    """
    reader = Reader(
        image,
        mz=candidate_mzs,
        mz_tolerance_absolute=mz_tolerance_absolute,
        mz_tolerance_relative=mz_tolerance_relative,
        fs_kwargs=fs_kwargs,
    )
    channel_dim = dimensions.DimensionNames.Channel
    xarr = reader.xarray_data
    if reader.is_continuous:
        # "continuous" mode Readers ignore `mz=` and expose their native
        # axis instead -- select the candidate channels out of it. Safe to
        # match exactly: `candidate_mzs` came from this same native axis via
        # `mean_spectrum`/`find_peaks_in_spectrum`, no resampling in between.
        idx = np.searchsorted(reader.mz_values, candidate_mzs)
        idx = np.clip(idx, 0, len(reader.mz_values) - 1)
        xarr = xarr.isel({channel_dim: idx})
    xarr = xarr.astype(np.float64)

    non_channel_dims = [d for d in xarr.dims if d != channel_dim]
    n_pixels = int(np.prod([xarr.sizes[d] for d in non_channel_dims]))

    peak = xarr.max(dim=non_channel_dims)  # (C,)
    has_signal = peak > 0
    pixel_frequency = (
        (xarr > peak * presence_threshold).sum(dim=non_channel_dims) / n_pixels
    ).where(has_signal, 0.0)

    # smooth only the spatial (Y, X) axes -- box size 1 elsewhere is a no-op
    norm = xarr / peak.where(has_signal, 1.0)
    smooth_size = tuple(
        3
        if d in (dimensions.DimensionNames.SpatialY, dimensions.DimensionNames.SpatialX)
        else 1
        for d in xarr.dims
    )
    smoothed = uniform_filter(norm.data, size=smooth_size)
    diff = xr.DataArray(
        np.abs(norm.data - smoothed), dims=xarr.dims, coords=xarr.coords
    )

    spatial_chaos = (
        diff.mean(dim=non_channel_dims) / (norm.mean(dim=non_channel_dims) + 1e-6)
    ).clip(0.0, 1.0)
    spatial_chaos = spatial_chaos.where(has_signal, 1.0)

    return pixel_frequency.values, spatial_chaos.values


class PeakPickingResult(NamedTuple):
    """Result of `auto_pick_peaks`.

    `mzs` is sorted by descending mean-spectrum intensity; `pixel_frequency`
    and `spatial_chaos` line up with it 1:1, so a dropped candidate's values
    are visible by comparing against the thresholds passed to
    `auto_pick_peaks`. `mean_spectrum_mz`/`mean_spectrum_intensity` are the
    full spectrum the candidates were detected from.
    """

    mzs: np.ndarray
    pixel_frequency: np.ndarray
    spatial_chaos: np.ndarray
    mean_spectrum_mz: np.ndarray
    mean_spectrum_intensity: np.ndarray


def auto_pick_peaks(
    image: types.PathLike,
    *,
    min_mz: float | None = None,
    max_mz: float | None = None,
    bin_width: float = 0.05,
    top_n_peaks: int | None = None,
    snr_threshold: float = 3.0,
    min_relative_intensity: float = 0.005,
    min_separation_mz: float = 0.5,
    min_pixel_frequency: float = 0.01,
    max_spatial_chaos: float = 0.7,
    mz_tolerance_absolute: float | None = None,
    mz_tolerance_relative: float | None = None,
    fs_kwargs: dict[str, Any] = {},
) -> PeakPickingResult:
    """Auto-detect m/z channels worth extracting from an imzML file.

    Finds candidate peaks on the file's mean spectrum (see
    `find_peaks_in_spectrum`), then drops candidates that are either too rare
    across pixels (`min_pixel_frequency`) or spatially unstructured
    (`max_spatial_chaos`) -- i.e. noise/matrix artifacts rather than real
    signal. Feed the result into `Reader(image, mz=result.mzs)` to extract
    the surviving channels lazily.

    Parameters mirror `mean_spectrum` (`min_mz`, `max_mz`, `bin_width`),
    `find_peaks_in_spectrum` (`snr_threshold`, `min_relative_intensity`,
    `min_separation_mz`), and `Reader` (`mz_tolerance_absolute`,
    `mz_tolerance_relative`, `fs_kwargs`). `top_n_peaks` caps the number of
    channels returned after filtering (default: all of them).
    """
    mean_mz, mean_intensity = mean_spectrum(
        image, min_mz=min_mz, max_mz=max_mz, bin_width=bin_width, fs_kwargs=fs_kwargs
    )
    candidates = find_peaks_in_spectrum(
        mean_mz,
        mean_intensity,
        snr_threshold=snr_threshold,
        min_relative_intensity=min_relative_intensity,
        min_separation_mz=min_separation_mz,
    )

    if len(candidates) == 0:
        empty = np.zeros(0, dtype=np.float64)
        return PeakPickingResult(candidates, empty, empty, mean_mz, mean_intensity)

    frequency, chaos = pixel_frequency_and_spatial_chaos(
        image,
        candidates,
        mz_tolerance_absolute=mz_tolerance_absolute,
        mz_tolerance_relative=mz_tolerance_relative,
        fs_kwargs=fs_kwargs,
    )

    # boolean masking preserves `candidates`' existing descending-intensity order
    keep = (frequency >= min_pixel_frequency) & (chaos <= max_spatial_chaos)
    mzs, frequency, chaos = candidates[keep], frequency[keep], chaos[keep]

    if top_n_peaks is not None:
        mzs, frequency, chaos = (
            mzs[:top_n_peaks],
            frequency[:top_n_peaks],
            chaos[:top_n_peaks],
        )

    return PeakPickingResult(mzs, frequency, chaos, mean_mz, mean_intensity)
