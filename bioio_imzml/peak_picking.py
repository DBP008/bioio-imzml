import warnings
from typing import Any, Literal, NamedTuple

import numpy as np
import xarray as xr
from bioio_base import dimensions, types
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks, savgol_filter

from .reader import Reader
from .utils import mz_tolerance_window

###############################################################################


def find_peaks_in_spectrum(
    mz_axis: np.ndarray,
    intensity: np.ndarray,
    *,
    smooth: bool = True,
    savgol_window: int = 7,
    savgol_polyorder: int = 2,
    snr_threshold: float | None = 3.0,
    min_relative_intensity: float = 0.005,
    min_separation_mz: float | None = 0.5,
    min_separation_relative: float | None = None,
) -> np.ndarray:
    """Candidate peak m/z values in a spectrum, sorted by descending intensity.

    Pure array function -- no file I/O, works on any `(mz_axis, intensity)`
    pair such as the one returned by `mean_spectrum`.

    `snr_threshold` is a prominence threshold as a multiple of the spectrum's
    MAD noise floor, or `None` to disable prominence filtering entirely;
    `min_relative_intensity` is a height threshold as a fraction of the
    spectrum's max.

    The minimum spacing between two accepted peaks is a per-peak window
    `min_separation_mz + m/z * min_separation_relative` (via
    `mz_tolerance_window`), so it can track instrument resolution rather than a
    fixed Da gap -- pass `min_separation_relative` (a fraction, e.g. `5e-6` for
    5 ppm) to make it scale with m/z. Enforced as a greedy pass: peaks are
    walked from most to least intense and a candidate is dropped if a
    stronger, already-kept peak lies within its window. With both components
    `None`/0 the window is 0, so every local maximum passing the
    height/prominence filters survives.
    """
    if smooth and len(intensity) > savgol_window:
        signal = savgol_filter(intensity, savgol_window, savgol_polyorder)
    else:
        signal = intensity

    prominence = None
    if snr_threshold is not None:
        noise_level = np.median(np.abs(signal - np.median(signal)))
        prominence = snr_threshold * noise_level

    peak_indices, _ = find_peaks(
        signal,
        height=min_relative_intensity * np.max(signal),
        prominence=prominence,
    )

    peak_mz = mz_axis[peak_indices]
    order = np.argsort(signal[peak_indices])[::-1]  # descending intensity
    gap = mz_tolerance_window(peak_mz, min_separation_mz, min_separation_relative)

    kept: list[float] = []
    # ponytail: O(n*k) greedy dedup; n is peak count (~1e3), swap for a sorted
    # structure + bisect only if a spectrum ever yields far more candidates.
    for i in order:
        m = peak_mz[i]
        if not kept or np.min(np.abs(np.asarray(kept) - m)) > gap[i]:
            kept.append(m)
    return np.asarray(kept, dtype=np.float64)


def mean_spectrum(
    image: types.PathLike,
    *,
    min_mz: float | None = None,
    max_mz: float | None = None,
    bin_width: float = 0.05,
    mz_agg: Literal["nearest", "sum"] = "sum",
    fs_kwargs: dict[str, Any] = {},
) -> tuple[np.ndarray, np.ndarray]:
    """Mean spectrum across every pixel of an imzML file, via `Reader`.

    For "continuous" mode files this is exact (every pixel already shares
    one m/z axis, and `Reader` reads it directly); `min_mz`/`max_mz` just
    restrict the returned range. For "processed" mode files, `Reader`'s own
    channel matching at `bin_width` spacing stands in for binning -- the same
    matching used when actually extracting channels, so the candidates this
    finds line up with what extraction will return. `mz_agg="sum"` (default)
    adds up every measured peak in the bin; `mz_agg="nearest"` instead takes
    each bin's single closest measured peak, which can look jagged at a fine
    `bin_width` (see `find_peaks_in_spectrum`/`auto_pick_peaks` caveats).

    Returns
    -------
    mz_axis, mean_intensity : np.ndarray
    """
    # With both bounds known, build the grid in-window so a "processed" mode
    # Reader only matches channels in [min_mz, max_mz], not over the file's
    # full native span. The post-slice below still covers the single-bound
    # case, continuous mode, and the arange overshoot past max_mz.
    if min_mz is not None and max_mz is not None:
        targets = np.arange(min_mz, max_mz + bin_width, bin_width)
        reader = Reader(image, mz=targets, mz_agg=mz_agg, fs_kwargs=fs_kwargs)
    else:
        reader = Reader(image, mz_step=bin_width, mz_agg=mz_agg, fs_kwargs=fs_kwargs)
    mz_axis = reader.mz_values
    # dask-backed: .mean() reduces one row chunk at a time rather than
    # materializing the full (C, Z, Y, X) cube (>1 TiB at fine bin_width).
    non_channel_dims = [
        d
        for d in reader.xarray_dask_data.dims
        if d != dimensions.DimensionNames.Channel
    ]
    mean_intensity = reader.xarray_dask_data.mean(dim=non_channel_dims).values.astype(
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
    block_bytes: float = 256e6,
    fs_kwargs: dict[str, Any] = {},
) -> tuple[np.ndarray, np.ndarray]:
    """Per-candidate `(pixel_frequency, spatial_chaos)`, one value per entry
    in `candidate_mzs` (no filtering applied here -- see `auto_pick_peaks`).

    `block_bytes` caps the float64 working set of the channel loop (~256MB by
    default); lower it only on a tight machine.

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
    spatial_dims = (
        dimensions.DimensionNames.SpatialY,
        dimensions.DimensionNames.SpatialX,
    )
    xarr = reader.xarray_dask_data
    # Realign the channel axis to `candidate_mzs`' order: a "continuous" Reader
    # exposes its own native axis and a "processed" one sorts `mz` ascending,
    # so neither keeps the descending-intensity order from
    # `find_peaks_in_spectrum`. Nearest-match is safe -- `candidate_mzs` came
    # from this same axis via `mean_spectrum`, no resampling in between.
    idx = np.searchsorted(reader.mz_values, candidate_mzs)
    idx = np.clip(idx, 0, len(reader.mz_values) - 1)
    xarr = xarr.isel({channel_dim: idx})

    non_channel_dims = [d for d in xarr.dims if d != channel_dim]
    n_pixels = int(np.prod([xarr.sizes[d] for d in non_channel_dims]))

    # Process channels in memory-bounded blocks: with the intensity/SNR filters
    # off by default candidates can run to tens of thousands, and the dense
    # float64 cube plus its working copies would OOM. Each candidate is
    # independent and uniform_filter is size-1 on the channel axis, so blocking
    # never changes the result.
    batch = max(1, int(block_bytes // (n_pixels * 8)))
    freqs, chaoses = [], []
    for start in range(0, xarr.sizes[channel_dim], batch):
        block = (
            xarr.isel({channel_dim: slice(start, start + batch)})
            .astype(np.float64)
            .compute()
        )
        peak = block.max(dim=non_channel_dims)  # (C,)
        has_signal = peak > 0
        freqs.append(
            ((block > peak * presence_threshold).sum(dim=non_channel_dims) / n_pixels)
            .where(has_signal, 0.0)
            .values
        )
        # smooth only the spatial axes -- box size 1 elsewhere is a no-op
        norm = block / peak.where(has_signal, 1.0)
        smooth_size = tuple(3 if d in spatial_dims else 1 for d in block.dims)
        diff = xr.DataArray(
            np.abs(norm.data - uniform_filter(norm.data, size=smooth_size)),
            dims=block.dims,
            coords=block.coords,
        )
        chaoses.append(
            (diff.mean(dim=non_channel_dims) / (norm.mean(dim=non_channel_dims) + 1e-6))
            .clip(0.0, 1.0)
            .where(has_signal, 1.0)
            .values
        )

    return np.concatenate(freqs), np.concatenate(chaoses)


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
    smooth: bool = True,
    savgol_window: int = 7,
    savgol_polyorder: int = 2,
    snr_threshold: float | None = None,
    min_relative_intensity: float = 0.0,
    min_pixel_frequency: float | None = 0.01,
    max_spatial_chaos: float | None = 0.4,
    top_n_peaks: int | None = None,
    mz_tolerance_absolute: float | None = None,
    mz_tolerance_relative: float | None = None,
    min_separation_absolute: float | None = None,
    min_separation_relative: float | None = None,
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
    `find_peaks_in_spectrum` (`smooth`, `savgol_window`, `savgol_polyorder`,
    `snr_threshold`, `min_relative_intensity`, `min_separation_absolute`/
    `min_separation_relative` -- forwarded as its `min_separation_mz`/
    `min_separation_relative`), and `Reader` (`mz_tolerance_absolute`,
    `mz_tolerance_relative`, `fs_kwargs`).
    `top_n_peaks` caps the number of channels returned after filtering
    (default: all of them).

    Three independent m/z windows govern picking, and physically they should
    satisfy `bin_width <= mz_tol <= 0.5 * min_peak_separation`:

    * `bin_width` -- the mean-spectrum grid step (detection resolution).
    * `mz_tolerance_absolute`/`mz_tolerance_relative` -- the extraction/scoring
      window half-width `+/-tol` (mass accuracy), used both to build each
      channel and to score per-pixel frequency/chaos.
    * `min_separation_absolute`/`min_separation_relative` -- the minimum spacing
      between two accepted candidates (instrument resolving power). Combined as
      `absolute + m/z * relative` (via `mz_tolerance_window`), so it can grow
      with m/z instead of being a fixed Da gap. When both are `None` it
      defaults to `2 * bin_width` (a couple of grid samples apart), so dedup is
      always enforced regardless of the extraction tolerance; pass `0` for both
      to disable it (distinctness then rests on the height/prominence filters
      alone, like LipostarMSI). This is deliberately decoupled from
      `mz_tolerance_*`: setting separation equal to the half-width `tol` would
      let two accepted peaks sit `tol` apart with 50%-overlapping extraction
      windows and double-count intensity under `mz_agg="sum"`.

    A `UserWarning` (never an error) is emitted when `min_peak_separation`
    comes out tighter than `2 * mz_tol`, the case that double-counts intensity
    under `mz_agg="sum"`.

    Defaults mirror LipostarMSI's out-of-the-box peak selection: the SNR and
    relative-intensity filters ship disabled (`snr_threshold=None`,
    `min_relative_intensity=0.0`), leaving the per-pixel frequency/chaos passes
    to do the culling. `max_spatial_chaos=0.4` corresponds to LipostarMSI's
    "min spatial chaos 0.6" under this module's opposite convention
    (0 = structured, 1 = random).

    The mean spectrum this detects candidates on uses `mean_spectrum`'s
    default `mz_agg="sum"` (every measured peak within each bin summed, not
    just the closest one). Call `mean_spectrum(image, bin_width=...,
    mz_agg="nearest")` directly and feed it to `find_peaks_in_spectrum`
    instead if you specifically want single-peak-per-bin values -- that mode
    can look jagged at a fine `bin_width`, since adjacent bins can jump
    around even where the real signal is smooth.
    `smooth`/`savgol_window`/`savgol_polyorder` control the Savitzky-Golay
    smoothing applied before detection (not to `result.mean_spectrum_intensity`
    itself, which stays raw); widen `savgol_window` if that jaggedness is
    producing spurious or duplicate-looking candidates.

    `snr_threshold`, `min_pixel_frequency`, and `max_spatial_chaos` each
    accept `None` to disable that filter -- e.g. data with sparse per-pixel
    peak-picking (single-cell-resolution processed-mode files) can score
    every candidate as spatially "chaotic" under the default threshold, so
    `max_spatial_chaos=None` keeps whatever passes intensity/frequency
    filtering instead of rejecting everything. When both
    `min_pixel_frequency` and `max_spatial_chaos` are `None`, the per-pixel
    pass over the file (the slow part) is skipped entirely and the result's
    `pixel_frequency`/`spatial_chaos` come back filled with `NaN`.
    """
    mean_mz, mean_intensity = mean_spectrum(
        image, min_mz=min_mz, max_mz=max_mz, bin_width=bin_width, fs_kwargs=fs_kwargs
    )
    # ponytail: default separation = 2*bin_width (a couple grid samples apart);
    # instrument-agnostic. Tune per instrument (~FWHM) if it merges real peaks
    # or lets duplicates through.
    sep_abs = min_separation_absolute
    if sep_abs is None and min_separation_relative is None:
        sep_abs = 2 * bin_width
    # Warn (never raise) when peak separation ends up tighter than the full
    # extraction window: accepted peaks then have overlapping ±tol windows and
    # mz_agg="sum" double-counts their shared intensity.
    if len(mean_mz) > 0 and (
        mz_tolerance_absolute is not None or mz_tolerance_relative is not None
    ):
        rep_mz = float(np.median(mean_mz))
        tol = (mz_tolerance_absolute or 0.0) + rep_mz * (mz_tolerance_relative or 0.0)
        sep = (sep_abs or 0.0) + rep_mz * (min_separation_relative or 0.0)
        if sep < 2 * tol:
            warnings.warn(
                f"min_peak_separation ({sep:.4g} m/z at m/z {rep_mz:.4g}) is "
                f"smaller than the full extraction window 2*mz_tol ({2 * tol:.4g}); "
                "accepted peaks can have overlapping ±tol windows and "
                "mz_agg='sum' will double-count intensity. Raise "
                "min_separation_* to >= 2x the tolerance.",
                UserWarning,
                stacklevel=2,
            )
    candidates = find_peaks_in_spectrum(
        mean_mz,
        mean_intensity,
        smooth=smooth,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        snr_threshold=snr_threshold,
        min_relative_intensity=min_relative_intensity,
        min_separation_mz=sep_abs,
        min_separation_relative=min_separation_relative,
    )

    if len(candidates) == 0:
        empty = np.zeros(0, dtype=np.float64)
        return PeakPickingResult(candidates, empty, empty, mean_mz, mean_intensity)

    if min_pixel_frequency is None and max_spatial_chaos is None:
        frequency = np.full(len(candidates), np.nan)
        chaos = np.full(len(candidates), np.nan)
    else:
        frequency, chaos = pixel_frequency_and_spatial_chaos(
            image,
            candidates,
            mz_tolerance_absolute=mz_tolerance_absolute,
            mz_tolerance_relative=mz_tolerance_relative,
            fs_kwargs=fs_kwargs,
        )

    # boolean masking preserves `candidates`' existing descending-intensity order
    keep = np.ones(len(candidates), dtype=bool)
    if min_pixel_frequency is not None:
        keep &= frequency >= min_pixel_frequency
    if max_spatial_chaos is not None:
        keep &= chaos <= max_spatial_chaos
    mzs, frequency, chaos = candidates[keep], frequency[keep], chaos[keep]

    if top_n_peaks is not None:
        mzs, frequency, chaos = (
            mzs[:top_n_peaks],
            frequency[:top_n_peaks],
            chaos[:top_n_peaks],
        )

    return PeakPickingResult(mzs, frequency, chaos, mean_mz, mean_intensity)
