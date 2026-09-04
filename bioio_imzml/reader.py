import warnings
from collections.abc import Sequence
from typing import Any, Literal

import dask.array as da
import numpy as np
import xarray as xr
from bioio_base import constants, dimensions, exceptions, io, reader, transforms, types
from bioio_base.standard_metadata import StandardMetadata
from dask.delayed import delayed
from fsspec.spec import AbstractFileSystem
from pyimzml.ImzMLParser import ImzMLParser, PortableSpectrumReader

from .utils import (
    estimate_mz_tolerance,
    find_ibd_path,
    local_window_intensities,
    mz_tolerance_window,
    parse_creation_date,
    scan_mz_bounds,
    tolerance_decimal_places,
)

###############################################################################


class Reader(reader.Reader):
    """
    Reads imzML mass spectrometry imaging (MSI) data as a `(T, C, Z, Y, X)` image,
    where the channel axis `C` is m/z.

    Parameters
    ----------
    image: types.PathLike
        Path to the `.imzML` file. A sibling `.ibd` file (same base name) must
        exist alongside it.
    mz: Optional[Sequence[float] | np.ndarray]
        Target m/z channel values to extract. Only used for "processed" mode
        files (see Notes); ignored for "continuous" mode files, which always
        expose their native shared m/z axis exactly.
        Default: None (fall back to `mz_step` or `n_bins` channels).
    mz_step: Optional[float]
        Fixed m/z spacing to generate channels for "processed" mode files
        when `mz` isn't given, e.g. `mz_step=0.1` for a channel every 0.1
        m/z. Takes precedence over `n_bins`. The range is the file's true
        global m/z span, found with a cheap scan of each spectrum's
        first/last value.
        Default: None (fall back to `n_bins` evenly spaced channels)
    n_bins: int
        Number of evenly spaced m/z channels to generate for "processed" mode
        files when neither `mz` nor `mz_step` is given. The range is the
        file's true global m/z span, found with a cheap scan of each
        spectrum's first/last value.
        Default: 512
    mz_tolerance_absolute: Optional[float]
        Only used for "processed" mode files. The absolute component of the
        per-channel matching window, in the same units as `mz` itself (e.g.
        `mz_tolerance_absolute=0.005` for 0.005 Da). Combined with
        `mz_tolerance_relative` as `tolerance = absolute + m/z * relative`. A
        pixel with no peak within `tolerance` of a target gets 0 for that
        channel (see `mz_agg` for how peaks within the window combine).
        Default: None (see Notes for the fallback when both components are
        unset)
    mz_tolerance_relative: Optional[float]
        Only used for "processed" mode files. The relative component of the
        matching window, as a fraction of m/z -- see `mz_tolerance_absolute`.
        Also in the same units as `mz` (a plain fraction, not a percentage or
        ppm count): convert yourself, e.g. `mz_tolerance_relative=3e-6` for a
        3 ppm window.
        Default: None (see Notes for the fallback when both components are
        unset)
    mz_agg: Literal["nearest", "sum"]
        Only used for "processed" mode files. How each channel's value is
        computed from the peaks within its tolerance window: "sum" adds up
        every measured peak's intensity in the window, matching how many
        external tools (e.g. Lipostar, MetaboScape) aggregate signal in a
        window; "nearest" takes the closest measured peak's intensity
        instead, dropping the rest.
        Default: "sum"
    add_tic: bool
        If True, append one extra channel named "TIC" giving each pixel's
        Total Ion Count -- the sum of every peak intensity in that pixel's
        full raw spectrum (computed before channel extraction, so it captures
        signal outside the m/z grid and signal dropped by tolerance windows,
        unlike summing the extracted channels). Works the same in both
        "continuous" and "processed" mode. Because TIC is a derived channel,
        `channel_names` then has one more entry ("TIC") than `mz_values` /
        `mz_tolerance`, which continue to describe only the m/z channels.
        Default: False
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created
        filesystem.
        Default: {}

    Notes
    -----
    For "processed" mode files, leaving both tolerance parameters unset
    doesn't disable filtering -- it auto-estimates one instead: each
    channel's tolerance becomes half the distance to its nearest neighboring
    target m/z, so adjacent channels' windows never overlap (a lone target
    with no neighbor is left unbounded, i.e. no filtering). Pass either
    tolerance parameter explicitly to opt out of the estimate and use a fixed
    window instead. "continuous" mode files always read their native axis
    exactly, so no window ever applies there (tolerance is 0). Either way,
    `channel_names` reports the resulting per-channel window as
    `"<m/z>±<tolerance>"`, with both sides shown to as many decimal places as
    the tolerance needs for 3 significant digits (e.g. `150.00000±0.00550`),
    falling back to 4 decimals when the tolerance is 0 or unbounded (inf).
    `mz_agg` is likewise ignored for "continuous" mode files.

    imzML stores spectra in one of two modes:

    * "continuous": every pixel shares one m/z axis, so the intensity array at
      each pixel already lines up with every other pixel's. This reader
      detects that case structurally (identical m/z byte offset and length for
      every spectrum) and reads it directly, with no resampling.
    * "processed": each pixel has its own m/z axis (typical for high
      resolution / profile data). There is no single true set of channels, so
      this reader resamples each spectrum onto shared target m/z values
      (given explicitly via `mz` or auto-generated with `mz_step` or
      `n_bins`), matching peaks within each channel's tolerance window per
      `mz_agg`.
    """

    NAME = "bioio-imzml"

    _fs: AbstractFileSystem
    _path: str

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        if not str(path).lower().endswith(".imzml"):
            raise exceptions.UnsupportedFileFormatError(
                "bioio-imzml", str(path), "File extension is not '.imzML'."
            )
        try:
            find_ibd_path(fs, path)
        except FileNotFoundError as e:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-imzml", str(path), str(e)
            )
        try:
            with fs.open(path, "rb") as f:
                head = f.read(4096)
        except Exception as e:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-imzml", str(path), str(e)
            )

        if b"mzML" not in head:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-imzml",
                str(path),
                "File does not look like an mzML/imzML document.",
            )
        return True

    def __init__(
        self,
        image: types.PathLike,
        mz: Sequence[float] | np.ndarray | None = None,
        mz_step: float | None = None,
        n_bins: int = 512,
        mz_tolerance_absolute: float | None = None,
        mz_tolerance_relative: float | None = None,
        mz_agg: Literal["nearest", "sum"] = "sum",
        add_tic: bool = False,
        fs_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ):
        self._fs, self._path = io.pathlike_to_fs(
            image, enforce_exists=True, fs_kwargs=fs_kwargs
        )
        self._is_supported_image(self._fs, self._path)
        self._ibd_path = find_ibd_path(self._fs, self._path)

        with (
            self._fs.open(self._path, "rb") as imzml_f,
            self._fs.open(self._ibd_path, "rb") as ibd_f,
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="pyimzml.ontology.ontology"
            )
            parser = ImzMLParser(imzml_f, ibd_file=ibd_f)

        self._imzmldict: dict[str, Any] = parser.imzmldict
        assert parser.root is not None
        assert parser.metadata is not None
        run_elem = parser.root.find(f"{parser.sl}run")
        self._imzml_metadata: dict[str, Any] = {
            **parser.metadata.pretty(),
            "creation_date": (
                run_elem.get("startTimeStamp") if run_elem is not None else None
            ),
        }
        self._width = int(self._imzmldict["max count of pixels x"])
        self._height = int(self._imzmldict["max count of pixels y"])
        self._depth = int(self._imzmldict.get("max count of pixels z", 1))
        self._pixel_size_x = self._imzmldict.get("pixel size x")
        self._pixel_size_y = self._imzmldict.get("pixel size y")

        self._portable: PortableSpectrumReader = parser.portable_spectrum_reader()
        self._continuous = (
            len(set(self._portable.mzOffsets)) <= 1
            and len(set(self._portable.mzLengths)) <= 1
        )

        # (z0, y0) -> [(x0, spectrum_index), ...], built once so dask tasks can
        # grab exactly the pixels for one row without rescanning coordinates.
        self._index_by_row: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for i, (x, y, z) in enumerate(self._portable.coordinates):
            self._index_by_row.setdefault((z - 1, y - 1), []).append((x - 1, i))

        if self._continuous:
            with self._fs.open(self._ibd_path, "rb") as f:
                mzs, _ = self._portable.read_spectrum_from_file(f, 0)
            self._mz_axis = np.asarray(mzs, dtype=np.float64)
        elif mz is not None:
            self._mz_axis = np.asarray(sorted(mz), dtype=np.float64)
        else:
            with self._fs.open(self._ibd_path, "rb") as f:
                lo, hi = scan_mz_bounds(self._portable, f)
            if mz_step is not None:
                self._mz_axis = np.arange(lo, hi + mz_step, mz_step)
            else:
                self._mz_axis = np.linspace(lo, hi, n_bins)

        self._mz_tolerance_absolute = mz_tolerance_absolute
        self._mz_tolerance_relative = mz_tolerance_relative
        self._mz_agg = mz_agg
        self._add_tic = add_tic
        # Per-channel window for channel_names and "processed" mode matching
        # (see _read_row); "continuous" mode reads its native axis exactly.
        if mz_tolerance_absolute is None and mz_tolerance_relative is None:
            self._mz_tolerance = (
                np.zeros(len(self._mz_axis), dtype=np.float64)
                if self._continuous
                else estimate_mz_tolerance(self._mz_axis)
            )
        else:
            self._mz_tolerance = mz_tolerance_window(
                self._mz_axis, mz_tolerance_absolute, mz_tolerance_relative
            )
        self._reader_init_params: dict[str, Any] = {
            "mz": None if mz is None else list(mz),
            "mz_step": mz_step,
            "n_bins": n_bins,
            "mz_tolerance_absolute": mz_tolerance_absolute,
            "mz_tolerance_relative": mz_tolerance_relative,
            "mz_agg": mz_agg,
            "add_tic": add_tic,
            "is_continuous": self._continuous,
        }
        self._scenes: tuple[str, ...] = ("Image:0",)

    @property
    def scenes(self) -> tuple[str, ...] | None:
        return self._scenes

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        # imzML has no field for a Z step; 3D IMS files leave it undefined.
        return types.PhysicalPixelSizes(
            Z=None, Y=self._pixel_size_y, X=self._pixel_size_x
        )

    @property
    def standard_metadata(self) -> StandardMetadata:
        metadata = super().standard_metadata
        metadata.imaging_datetime = parse_creation_date(
            self._imzml_metadata.get("creation_date")
        )
        return metadata

    @property
    def mz_values(self) -> np.ndarray:
        """The channel axis as numeric m/z values (`channel_names` gives the
        same values formatted as strings, per the base Reader contract).
        """
        return self._mz_axis

    @property
    def is_continuous(self) -> bool:
        """True if every spectrum shares one m/z axis (read directly, with no
        resampling); False if channels were built by nearest-neighbor lookup.
        """
        return self._continuous

    @property
    def mz_tolerance_absolute(self) -> float | None:
        """The Da (absolute) component passed for the matching window, as
        given to `__init__` (None means 0 Da).
        """
        return self._mz_tolerance_absolute

    @property
    def mz_tolerance_relative(self) -> float | None:
        """The relative component passed for the matching window, as a
        fraction of m/z, as given to `__init__` (None means 0).
        """
        return self._mz_tolerance_relative

    @property
    def mz_tolerance(self) -> np.ndarray:
        """The final per-channel tolerance (same units as m/z), one value per
        `mz_values` entry: `mz_tolerance_absolute + m/z * mz_tolerance_relative`
        if either was given to `__init__`; otherwise, for "processed" mode
        files, an auto-estimate (see class Notes); otherwise (i.e.
        "continuous" mode with neither given) all zeros. This is the same
        array used to filter nearest-neighbor matches in "processed" mode and
        to format `channel_names`.
        """
        return self._mz_tolerance

    @property
    def mz_agg(self) -> Literal["nearest", "sum"]:
        """How each "processed" mode channel's value is computed from the
        peaks within its tolerance window, as given to `__init__`.
        """
        return self._mz_agg

    @staticmethod
    def _read_row(
        fs: AbstractFileSystem,
        ibd_path: str,
        portable: PortableSpectrumReader,
        pixels: list[tuple[int, int]],
        width: int,
        mz_axis: np.ndarray,
        continuous: bool,
        mz_tolerance: np.ndarray | None,
        mz_agg: Literal["nearest", "sum"],
        add_tic: bool,
    ) -> np.ndarray:
        """One (channels, width) row: all pixels at a given (z, y)."""
        n_mz = len(mz_axis)
        out = np.zeros((n_mz + (1 if add_tic else 0), width), dtype=np.float32)
        if not pixels:
            return out

        with fs.open(ibd_path, "rb") as f:
            for x0, spectrum_index in pixels:
                mzs, intensities = portable.read_spectrum_from_file(f, spectrum_index)
                if continuous:
                    out[:n_mz, x0] = intensities
                else:
                    out[:n_mz, x0] = local_window_intensities(
                        mzs, intensities, mz_axis, mz_tolerance, mz_agg
                    )
                if add_tic:
                    out[n_mz, x0] = np.sum(intensities)

        return out

    def _create_dask_array(self) -> da.Array:
        n_channels = len(self._mz_axis) + (1 if self._add_tic else 0)
        z_planes = []
        for z0 in range(self._depth):
            rows = [
                da.from_delayed(
                    delayed(Reader._read_row)(
                        self._fs,
                        self._ibd_path,
                        self._portable,
                        self._index_by_row.get((z0, y0), []),
                        self._width,
                        self._mz_axis,
                        self._continuous,
                        self._mz_tolerance,
                        self._mz_agg,
                        self._add_tic,
                    ),
                    shape=(n_channels, self._width),
                    dtype=np.float32,
                )
                for y0 in range(self._height)
            ]
            z_planes.append(da.stack(rows, axis=1))  # (C, Y, X)

        return da.stack(z_planes, axis=1)  # (C, Z, Y, X)

    def _to_data_array(self, image_data: types.ArrayLike) -> xr.DataArray:
        image_data = transforms.reshape_data(
            image_data, "CZYX", dimensions.DEFAULT_DIMENSION_ORDER
        )
        names = []
        for mz, tol in zip(self._mz_axis, self._mz_tolerance):
            decimals = tolerance_decimal_places(tol)
            names.append(f"{mz:.{decimals}f}±{tol:.{decimals}f}")
        if self._add_tic:
            names.append("TIC")
        coords = {dimensions.DimensionNames.Channel: names}
        # self._imzmldict (pixel/dimension counts, sizes) is deliberately not
        # spread in here -- pyimzml's own docs call it deprecated, and every
        # field it has is already nested under imzml_metadata["scan_settings"].
        metadata = {
            "imzml_metadata": self._imzml_metadata,
            "reader_init_params": self._reader_init_params,
        }
        return xr.DataArray(
            image_data,
            dims=list(dimensions.DEFAULT_DIMENSION_ORDER),
            coords=coords,
            attrs={
                constants.METADATA_UNPROCESSED: metadata,
                constants.METADATA_PROCESSED: metadata,
            },
        )

    def _read_delayed(self) -> xr.DataArray:
        return self._to_data_array(self._create_dask_array())

    def _read_immediate(self) -> xr.DataArray:
        return self._to_data_array(self._create_dask_array().compute())
