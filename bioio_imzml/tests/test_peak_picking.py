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


def test_mean_spectrum_continuous(continuous_imzml: Path) -> None:
    mz_axis, mean_intensity = mean_spectrum(continuous_imzml)

    np.testing.assert_allclose(mz_axis, CONTINUOUS_MZ_AXIS)
    n_pixels = CONTINUOUS_WIDTH * CONTINUOUS_HEIGHT
    expected = CONTINUOUS_MZ_AXIS / 100.0 + np.mean(np.arange(n_pixels)) * 10.0
    np.testing.assert_allclose(mean_intensity, expected)


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
        min_separation_mz=1.0,
        min_pixel_frequency=0.05,
    )

    assert any(np.isclose(mz, SPATIAL_REAL_PEAK_MZ, atol=5.0) for mz in result.mzs)
    assert not any(np.isclose(mz, SPATIAL_NOISE_PEAK_MZ, atol=5.0) for mz in result.mzs)


def test_auto_pick_peaks_max_spatial_chaos_none_disables_filter(
    spatial_imzml: Path,
) -> None:
    result = auto_pick_peaks(
        spatial_imzml,
        min_separation_mz=1.0,
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
        min_separation_mz=1.0,
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
