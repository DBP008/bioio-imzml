from pathlib import Path

import numpy as np
import pytest

from bioio_imzml.peak_picking import (
    auto_pick_peaks,
    find_peaks_in_spectrum,
    mean_spectrum,
    pixel_frequency_and_spatial_chaos,
)
from bioio_imzml.tests.conftest import (
    CONTINUOUS_HEIGHT,
    CONTINUOUS_MZ_AXIS,
    CONTINUOUS_WIDTH,
    SPATIAL_NOISE_PEAK_MZ,
    SPATIAL_REAL_PEAK_MZ,
)

###############################################################################


def test_find_peaks_in_spectrum_synthetic() -> None:
    mz_axis = np.arange(0.0, 100.0, 0.2)
    peak_centers = [20.0, 50.0, 80.0]
    intensity = np.zeros_like(mz_axis)
    for center in peak_centers:
        intensity += 10.0 * np.exp(-0.5 * ((mz_axis - center) / 3.0) ** 2)

    detected = find_peaks_in_spectrum(mz_axis, intensity, min_separation_mz=1.0)

    assert sorted(detected.tolist()) == pytest.approx(sorted(peak_centers), abs=0.5)


def test_find_peaks_relative_separation_scales_with_mz() -> None:
    # two peaks 0.02 apart at m/z ~500; the taller is at 500.00
    mz_axis = np.arange(499.0, 501.0, 0.001)
    intensity = 10.0 * np.exp(-0.5 * ((mz_axis - 500.00) / 0.002) ** 2)
    intensity += 5.0 * np.exp(-0.5 * ((mz_axis - 500.02) / 0.002) ** 2)

    both = find_peaks_in_spectrum(
        mz_axis,
        intensity,
        snr_threshold=None,
        min_relative_intensity=0.0,
        min_separation_mz=0.001,
    )
    # 100 ppm at m/z 500 = 0.05 Da > 0.02 gap -> the weaker peak is merged away
    merged = find_peaks_in_spectrum(
        mz_axis,
        intensity,
        snr_threshold=None,
        min_relative_intensity=0.0,
        min_separation_mz=0.0,
        min_separation_relative=1e-4,
    )

    assert len(both) == 2
    assert len(merged) == 1
    assert np.isclose(merged[0], 500.00, atol=0.005)  # the stronger peak survives


def test_min_separation_merges_close_peaks() -> None:
    # two Gaussian peaks 1.0 m/z apart on a 0.05 grid; taller at 50.0
    mz_axis = np.arange(48.0, 53.0, 0.05)
    intensity = 10.0 * np.exp(-0.5 * ((mz_axis - 50.0) / 0.15) ** 2)
    intensity += 5.0 * np.exp(-0.5 * ((mz_axis - 51.0) / 0.15) ** 2)

    both = find_peaks_in_spectrum(
        mz_axis, intensity, snr_threshold=None, min_separation_mz=0.0
    )
    merged = find_peaks_in_spectrum(
        mz_axis, intensity, snr_threshold=None, min_separation_mz=2.0
    )

    assert len(both) == 2
    assert len(merged) == 1
    assert np.isclose(merged[0], 50.0, atol=0.05)  # stronger peak survives


def test_auto_pick_peaks_separation_decoupled_from_tolerance(
    spatial_imzml: Path,
) -> None:
    # spatial_imzml has two peaks (~200, ~250). Separation, not tolerance,
    # controls candidate distinctness: a large tolerance with zero separation
    # keeps both; a tiny tolerance with a wide separation merges to one.
    fine = auto_pick_peaks(
        spatial_imzml,
        min_pixel_frequency=None,
        max_spatial_chaos=None,
        mz_tolerance_absolute=0.5,
        min_separation_absolute=2.0,
        min_separation_relative=0.0,
    )
    coarse = auto_pick_peaks(
        spatial_imzml,
        min_pixel_frequency=None,
        max_spatial_chaos=None,
        mz_tolerance_absolute=0.5,
        min_separation_absolute=100.0,
    )
    assert len(fine.mzs) > len(coarse.mzs)


def test_auto_pick_peaks_warns_on_overlapping_windows(
    spatial_imzml: Path,
) -> None:
    # separation (1.0) < 2*tolerance (2*2.0=4.0) -> overlapping extraction
    # windows / double-counting warning.
    with pytest.warns(UserWarning, match="double-count"):
        auto_pick_peaks(
            spatial_imzml,
            min_pixel_frequency=None,
            max_spatial_chaos=None,
            mz_tolerance_absolute=2.0,
            min_separation_absolute=1.0,
        )


def test_mean_spectrum_continuous(continuous_imzml: Path) -> None:
    mz_axis, mean_intensity = mean_spectrum(continuous_imzml)

    np.testing.assert_allclose(mz_axis, CONTINUOUS_MZ_AXIS)
    n_pixels = CONTINUOUS_WIDTH * CONTINUOUS_HEIGHT
    expected = CONTINUOUS_MZ_AXIS / 100.0 + np.mean(np.arange(n_pixels)) * 10.0
    np.testing.assert_allclose(mean_intensity, expected)


def test_mean_spectrum_bounds_narrow_grid(processed_imzml: Path) -> None:
    # Both bounds given -> the grid is generated in-window (starts exactly at
    # min_mz, stepped by bin_width), not built over the file's native span and
    # sliced. Fixture peaks sit at ~150 and ~250; window excludes the 150 peak.
    mz_axis, _ = mean_spectrum(
        processed_imzml, min_mz=200.0, max_mz=250.0, bin_width=1.0
    )
    assert mz_axis[0] == pytest.approx(200.0)
    assert mz_axis[-1] <= 250.0
    np.testing.assert_allclose(np.diff(mz_axis), 1.0)


def test_pixel_frequency_and_spatial_chaos_distinguish_noise(
    spatial_imzml: Path,
) -> None:
    mz_axis, intensity = mean_spectrum(spatial_imzml)
    candidates = find_peaks_in_spectrum(mz_axis, intensity, min_separation_mz=1.0)
    frequency, chaos = pixel_frequency_and_spatial_chaos(spatial_imzml, candidates)

    real_idx = int(np.argmin(np.abs(candidates - SPATIAL_REAL_PEAK_MZ)))
    noise_idx = int(np.argmin(np.abs(candidates - SPATIAL_NOISE_PEAK_MZ)))

    # real peak: coherent across half the grid; noise peak: a single pixel.
    assert frequency[real_idx] == pytest.approx(0.5, abs=1e-6)
    assert frequency[noise_idx] == pytest.approx(1.0 / 64.0, abs=1e-6)
    assert chaos[noise_idx] > chaos[real_idx]


def test_pixel_frequency_and_spatial_chaos_batching_is_value_neutral(
    spatial_imzml: Path,
) -> None:
    # The channel axis is processed in memory-bounded blocks; forcing batch=1
    # via a tiny block_bytes must match the single-block default (candidates are
    # independent; uniform_filter is size-1 on the channel axis).
    mz_axis, intensity = mean_spectrum(spatial_imzml)
    candidates = find_peaks_in_spectrum(mz_axis, intensity, min_separation_mz=1.0)

    freq_full, chaos_full = pixel_frequency_and_spatial_chaos(spatial_imzml, candidates)
    freq_batched, chaos_batched = pixel_frequency_and_spatial_chaos(
        spatial_imzml, candidates, block_bytes=1.0
    )

    assert len(candidates) > 1  # otherwise batching isn't exercised
    np.testing.assert_allclose(freq_batched, freq_full)
    np.testing.assert_allclose(chaos_batched, chaos_full)


def test_pixel_frequency_and_spatial_chaos_processed_mode_order(
    spatial_processed_imzml: Path,
) -> None:
    # "processed" mode `Reader`s sort `mz=` ascending internally, so this
    # deliberately passes candidates in *descending* order (mirroring
    # `find_peaks_in_spectrum`'s descending-intensity output) to catch a
    # regression where the returned frequency/chaos arrays silently came
    # back in the reader's sorted order instead of `candidate_mzs`' order.
    candidates = np.array([SPATIAL_NOISE_PEAK_MZ, SPATIAL_REAL_PEAK_MZ])
    frequency, chaos = pixel_frequency_and_spatial_chaos(
        spatial_processed_imzml, candidates
    )

    noise_idx, real_idx = 0, 1
    assert frequency[real_idx] == pytest.approx(0.5, abs=1e-6)
    assert frequency[noise_idx] == pytest.approx(1.0 / 64.0, abs=1e-6)
    assert chaos[noise_idx] > chaos[real_idx]


def test_auto_pick_peaks_filters_spatial_noise(spatial_imzml: Path) -> None:
    result = auto_pick_peaks(
        spatial_imzml,
        min_pixel_frequency=0.05,
    )

    assert any(np.isclose(mz, SPATIAL_REAL_PEAK_MZ, atol=5.0) for mz in result.mzs)
    assert not any(np.isclose(mz, SPATIAL_NOISE_PEAK_MZ, atol=5.0) for mz in result.mzs)


def test_auto_pick_peaks_max_spatial_chaos_none_disables_filter(
    spatial_imzml: Path,
) -> None:
    result = auto_pick_peaks(
        spatial_imzml,
        min_pixel_frequency=0.01,  # below the noise peak's 1/64 frequency
        max_spatial_chaos=None,
    )

    # the noise peak passes the (loosened) frequency filter on its own; only
    # the chaos filter would drop it, so disabling it must let it through.
    assert any(np.isclose(mz, SPATIAL_NOISE_PEAK_MZ, atol=5.0) for mz in result.mzs)
    # only max_spatial_chaos is disabled -- frequency/chaos are still
    # computed and reported (just not used to filter).
    assert not np.isnan(result.pixel_frequency).any()
    assert not np.isnan(result.spatial_chaos).any()


def test_auto_pick_peaks_no_quality_filters_skips_pixel_pass(
    spatial_imzml: Path,
) -> None:
    result = auto_pick_peaks(
        spatial_imzml,
        min_pixel_frequency=None,
        max_spatial_chaos=None,
    )

    # both filters disabled -- every detected candidate survives, and the
    # (skipped) per-pixel metrics come back as NaN rather than 0.
    assert any(np.isclose(mz, SPATIAL_REAL_PEAK_MZ, atol=5.0) for mz in result.mzs)
    assert any(np.isclose(mz, SPATIAL_NOISE_PEAK_MZ, atol=5.0) for mz in result.mzs)
    assert np.isnan(result.pixel_frequency).all()
    assert np.isnan(result.spatial_chaos).all()


def test_find_peaks_in_spectrum_snr_threshold_none_disables_prominence() -> None:
    mz_axis = np.arange(0.0, 100.0, 0.2)
    rng = np.random.default_rng(0)
    intensity = rng.uniform(0.0, 1.0, size=mz_axis.shape)
    intensity += 10.0 * np.exp(-0.5 * ((mz_axis - 50.0) / 3.0) ** 2)

    strict = find_peaks_in_spectrum(mz_axis, intensity, min_relative_intensity=0.0)
    loose = find_peaks_in_spectrum(
        mz_axis, intensity, min_relative_intensity=0.0, snr_threshold=None
    )

    # disabling the prominence filter can only relax detection, never tighten it
    assert len(loose) >= len(strict)
