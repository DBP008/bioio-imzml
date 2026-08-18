from pathlib import Path

import numpy as np
import pytest
from bioio_base import exceptions

from bioio_imzml import Reader
from bioio_imzml.tests.conftest import (
    CONTINUOUS_HEIGHT,
    CONTINUOUS_MZ_AXIS,
    CONTINUOUS_WIDTH,
    PROCESSED_HEIGHT,
    PROCESSED_PEAK_CENTERS,
    PROCESSED_WIDTH,
    continuous_expected,
    processed_expected,
)
from bioio_imzml.utils import parse_creation_date, tolerance_decimal_places

###############################################################################


def test_continuous_shape_and_values(continuous_imzml: Path) -> None:
    reader = Reader(continuous_imzml)

    assert reader.dims.order == "TCZYX"
    assert reader.shape == (
        1,
        len(CONTINUOUS_MZ_AXIS),
        1,
        CONTINUOUS_HEIGHT,
        CONTINUOUS_WIDTH,
    )
    assert reader.is_continuous

    data = reader.data  # (T, C, Z, Y, X)
    for y in range(CONTINUOUS_HEIGHT):
        for x in range(CONTINUOUS_WIDTH):
            pixel_id = y * CONTINUOUS_WIDTH + x
            np.testing.assert_allclose(
                data[0, :, 0, y, x], continuous_expected(pixel_id)
            )


def test_continuous_channel_names(continuous_imzml: Path) -> None:
    reader = Reader(continuous_imzml)
    decimals = tolerance_decimal_places(0.0)
    assert reader.channel_names == [
        f"{mz:.{decimals}f}±{0.0:.{decimals}f}" for mz in CONTINUOUS_MZ_AXIS
    ]
    np.testing.assert_allclose(reader.mz_values, CONTINUOUS_MZ_AXIS)


def test_continuous_delayed_matches_immediate(continuous_imzml: Path) -> None:
    reader = Reader(continuous_imzml)
    np.testing.assert_allclose(reader.dask_data.compute(), reader.data)


def test_metadata_includes_imzml_metadata(continuous_imzml: Path) -> None:
    reader = Reader(continuous_imzml)
    metadata = reader.metadata

    imzml_metadata = metadata["imzml_metadata"]
    assert set(imzml_metadata.keys()) == {
        "file_description",
        "referenceable_param_groups",
        "samples",
        "softwares",
        "scan_settings",
        "instrument_configurations",
        "data_processings",
        "creation_date",
    }


def test_metadata_includes_reader_init_params(
    continuous_imzml: Path, processed_imzml: Path
) -> None:
    default_reader = Reader(continuous_imzml)
    assert default_reader.metadata["reader_init_params"] == {
        "mz": None,
        "mz_step": None,
        "n_bins": 512,
        "mz_tolerance_absolute": None,
        "mz_tolerance_relative": None,
        "is_continuous": True,
    }

    custom_reader = Reader(processed_imzml, mz_step=0.5, mz_tolerance_absolute=0.1)
    assert custom_reader.metadata["reader_init_params"] == {
        "mz": None,
        "mz_step": 0.5,
        "n_bins": 512,
        "mz_tolerance_absolute": 0.1,
        "mz_tolerance_relative": None,
        "is_continuous": False,
    }


def test_standard_metadata_imaging_datetime_absent(continuous_imzml: Path) -> None:
    # the writer used by the fixtures doesn't set <run startTimeStamp>
    reader = Reader(continuous_imzml)
    assert reader.standard_metadata.imaging_datetime is None


def test_parse_creation_date() -> None:
    from datetime import datetime

    expected = datetime(2026, 4, 23, 12, 10, 2)
    assert parse_creation_date(None) is None
    assert parse_creation_date("not a date") is None
    assert parse_creation_date("2026-04-23T12:10:02") == expected
    assert parse_creation_date("4/23/2026 12:10:02 PM") == expected


def test_processed_explicit_targets(processed_imzml: Path) -> None:
    reader = Reader(processed_imzml, mz=PROCESSED_PEAK_CENTERS)

    assert not reader.is_continuous
    assert reader.shape == (
        1,
        len(PROCESSED_PEAK_CENTERS),
        1,
        PROCESSED_HEIGHT,
        PROCESSED_WIDTH,
    )

    data = reader.data
    for y in range(PROCESSED_HEIGHT):
        for x in range(PROCESSED_WIDTH):
            pixel_id = y * PROCESSED_WIDTH + x
            np.testing.assert_allclose(
                data[0, :, 0, y, x], processed_expected(pixel_id)
            )


def test_processed_tolerance_zeroes_distant_targets(processed_imzml: Path) -> None:
    far_target = np.array([999.0])  # nowhere near the ~150/250 peaks

    no_tolerance = Reader(processed_imzml, mz=far_target)
    assert no_tolerance.data[0, 0, 0, 0, 0] != 0  # nearest peak returned regardless
    # a lone target has no neighbor to estimate a window from, so it's left
    # unbounded rather than filtered.
    assert np.isinf(no_tolerance.mz_tolerance[0])

    # same units as m/z: the jittered peaks are within ~0.1 of their center,
    # so a 1.0 Da tolerance easily rejects a target that's ~750 away.
    with_tolerance = Reader(processed_imzml, mz=far_target, mz_tolerance_absolute=1.0)
    assert with_tolerance.mz_tolerance_absolute == 1.0
    np.testing.assert_allclose(with_tolerance.mz_tolerance, [1.0])
    assert with_tolerance.data[0, 0, 0, 0, 0] == 0  # too far, zeroed out


def test_processed_tolerance_keeps_close_targets(processed_imzml: Path) -> None:
    # PROCESSED_PEAK_CENTERS targets sit within jitter distance (<=0.08) of a
    # real peak at every pixel, so a modest tolerance should keep every value.
    reader = Reader(
        processed_imzml, mz=PROCESSED_PEAK_CENTERS, mz_tolerance_absolute=0.1
    )
    data = reader.data
    for y in range(PROCESSED_HEIGHT):
        for x in range(PROCESSED_WIDTH):
            pixel_id = y * PROCESSED_WIDTH + x
            np.testing.assert_allclose(
                data[0, :, 0, y, x], processed_expected(pixel_id)
            )


def test_processed_tolerance_combines_absolute_and_relative(
    processed_imzml: Path,
) -> None:
    # tolerance = absolute + m/z * relative (relative is a plain fraction of
    # m/z, e.g. 3e-6 for 3 ppm)
    reader = Reader(
        processed_imzml,
        mz=PROCESSED_PEAK_CENTERS,
        mz_tolerance_absolute=0.005,
        mz_tolerance_relative=3e-6,
    )
    expected = 0.005 + np.asarray(PROCESSED_PEAK_CENTERS) * 3e-6
    np.testing.assert_allclose(reader.mz_tolerance, expected)
    assert reader.channel_names == [
        f"{mz:.{tolerance_decimal_places(tol)}f}±{tol:.{tolerance_decimal_places(tol)}f}"
        for mz, tol in zip(PROCESSED_PEAK_CENTERS, expected)
    ]


def test_processed_tolerance_auto_estimated(processed_imzml: Path) -> None:
    # neither tolerance component given: falls back to half the distance to
    # each target's nearest neighboring target (100 apart here, so 50 each).
    reader = Reader(processed_imzml, mz=PROCESSED_PEAK_CENTERS)
    np.testing.assert_allclose(reader.mz_tolerance, [50.0, 50.0])
    assert reader.mz_tolerance_absolute is None
    assert reader.mz_tolerance_relative is None


def test_continuous_tolerance_always_zero(continuous_imzml: Path) -> None:
    # "continuous" mode reads its native axis exactly; no auto-estimate
    # applies even though targets are far apart (100 m/z steps).
    reader = Reader(continuous_imzml)
    np.testing.assert_allclose(reader.mz_tolerance, np.zeros(len(CONTINUOUS_MZ_AXIS)))


def test_processed_auto_bins(processed_imzml: Path) -> None:
    reader = Reader(processed_imzml, n_bins=5)

    assert reader.shape[1] == 5
    lo, hi = reader.mz_values[0], reader.mz_values[-1]
    assert lo == pytest.approx(min(PROCESSED_PEAK_CENTERS) - 0.08, abs=0.5)
    assert hi == pytest.approx(max(PROCESSED_PEAK_CENTERS) + 0.08, abs=0.5)


def test_processed_auto_step(processed_imzml: Path) -> None:
    reader = Reader(processed_imzml, mz_step=0.1)

    diffs = np.diff(reader.mz_values)
    np.testing.assert_allclose(diffs, 0.1)
    lo, hi = reader.mz_values[0], reader.mz_values[-1]
    assert lo == pytest.approx(min(PROCESSED_PEAK_CENTERS) - 0.08, abs=0.5)
    assert hi == pytest.approx(max(PROCESSED_PEAK_CENTERS) + 0.08, abs=0.5)


def test_processed_delayed_matches_immediate(processed_imzml: Path) -> None:
    reader = Reader(processed_imzml, mz=PROCESSED_PEAK_CENTERS)
    np.testing.assert_allclose(reader.dask_data.compute(), reader.data)


def test_unsupported_extension(tmp_path: Path) -> None:
    bogus = tmp_path / "not_an_imzml.txt"
    bogus.write_text("hello")
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(bogus)


def test_missing_ibd(tmp_path: Path) -> None:
    orphan = tmp_path / "orphan.imzML"
    orphan.write_text("<mzML></mzML>")
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(orphan)
