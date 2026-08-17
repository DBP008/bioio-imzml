from pathlib import Path

import numpy as np
import pytest
from pyimzml.ImzMLWriter import ImzMLWriter

###############################################################################
# Continuous-mode fixture: every pixel shares one m/z axis.

CONTINUOUS_MZ_AXIS = np.array([100.0, 200.0, 300.0, 400.0])
CONTINUOUS_WIDTH = 3
CONTINUOUS_HEIGHT = 2


@pytest.fixture
def continuous_imzml(tmp_path: Path) -> Path:
    path = tmp_path / "continuous.imzML"
    with ImzMLWriter(str(path), mode="continuous") as writer:
        for y in range(1, CONTINUOUS_HEIGHT + 1):
            for x in range(1, CONTINUOUS_WIDTH + 1):
                pixel_id = (y - 1) * CONTINUOUS_WIDTH + (x - 1)
                intensities = CONTINUOUS_MZ_AXIS / 100.0 + pixel_id * 10.0
                writer.addSpectrum(CONTINUOUS_MZ_AXIS, intensities, (x, y))
    return path


def continuous_expected(pixel_id: int) -> np.ndarray:
    return (CONTINUOUS_MZ_AXIS / 100.0 + pixel_id * 10.0).astype(np.float32)


###############################################################################
# Processed-mode fixture: every pixel has its own m/z axis (two peaks each,
# jittered around two shared centers so nearest-neighbor lookup is exercised).

PROCESSED_PEAK_CENTERS = np.array([150.0, 250.0])
PROCESSED_WIDTH = 2
PROCESSED_HEIGHT = 2


@pytest.fixture
def processed_imzml(tmp_path: Path) -> Path:
    path = tmp_path / "processed.imzML"
    with ImzMLWriter(str(path), mode="processed") as writer:
        for y in range(1, PROCESSED_HEIGHT + 1):
            for x in range(1, PROCESSED_WIDTH + 1):
                pixel_id = (y - 1) * PROCESSED_WIDTH + (x - 1)
                jitter = np.array([-0.01, 0.02]) * (pixel_id + 1)
                mzs = PROCESSED_PEAK_CENTERS + jitter
                intensities = np.array([pixel_id + 1.0, (pixel_id + 1.0) * 2.0])
                writer.addSpectrum(mzs, intensities, (x, y))
    return path


def processed_expected(pixel_id: int) -> np.ndarray:
    return np.array([pixel_id + 1.0, (pixel_id + 1.0) * 2.0], dtype=np.float32)


###############################################################################
# Spatial-structure fixture for peak_picking tests: a wider "continuous" mode
# grid with one peak spread coherently across half the pixels (real signal)
# and one peak present in a single pixel only (spatial noise).

SPATIAL_MZ_AXIS = np.arange(100.0, 305.0, 5.0)
SPATIAL_WIDTH = 8
SPATIAL_HEIGHT = 8
SPATIAL_REAL_PEAK_MZ = 200.0
SPATIAL_NOISE_PEAK_MZ = 250.0
SPATIAL_NOISE_PIXEL = (1, 1)  # (x, y), 1-indexed


def _gaussian_bump(
    mz_axis: np.ndarray, center: float, amplitude: float, sigma: float = 6.0
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((mz_axis - center) / sigma) ** 2)


@pytest.fixture
def spatial_imzml(tmp_path: Path) -> Path:
    path = tmp_path / "spatial.imzML"
    real_bump = _gaussian_bump(SPATIAL_MZ_AXIS, SPATIAL_REAL_PEAK_MZ, 50.0)
    noise_bump = _gaussian_bump(SPATIAL_MZ_AXIS, SPATIAL_NOISE_PEAK_MZ, 50.0)
    with ImzMLWriter(str(path), mode="continuous") as writer:
        for y in range(1, SPATIAL_HEIGHT + 1):
            for x in range(1, SPATIAL_WIDTH + 1):
                intensities = np.zeros_like(SPATIAL_MZ_AXIS)
                if x <= SPATIAL_WIDTH // 2:
                    intensities = intensities + real_bump
                if (x, y) == SPATIAL_NOISE_PIXEL:
                    intensities = intensities + noise_bump
                writer.addSpectrum(SPATIAL_MZ_AXIS, intensities, (x, y))
    return path


@pytest.fixture
def spatial_processed_imzml(tmp_path: Path) -> Path:
    """Same real-vs-noise layout as `spatial_imzml`, but written in
    "processed" mode (a distinct m/z array per spectrum, so `Reader`
    structurally detects `is_continuous=False`) to exercise the "processed"
    branch of `pixel_frequency_and_spatial_chaos`.
    """
    path = tmp_path / "spatial_processed.imzML"
    real_bump = _gaussian_bump(SPATIAL_MZ_AXIS, SPATIAL_REAL_PEAK_MZ, 50.0)
    noise_bump = _gaussian_bump(SPATIAL_MZ_AXIS, SPATIAL_NOISE_PEAK_MZ, 50.0)
    with ImzMLWriter(str(path), mode="processed") as writer:
        for y in range(1, SPATIAL_HEIGHT + 1):
            for x in range(1, SPATIAL_WIDTH + 1):
                intensities = np.zeros_like(SPATIAL_MZ_AXIS)
                if x <= SPATIAL_WIDTH // 2:
                    intensities = intensities + real_bump
                if (x, y) == SPATIAL_NOISE_PIXEL:
                    intensities = intensities + noise_bump
                writer.addSpectrum(SPATIAL_MZ_AXIS, intensities, (x, y))
    return path
