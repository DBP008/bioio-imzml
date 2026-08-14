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
