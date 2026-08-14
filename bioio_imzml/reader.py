from collections.abc import Sequence
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
from bioio_base import constants, dimensions, exceptions, io, reader, transforms, types
from dask import delayed
from fsspec.spec import AbstractFileSystem
from pyimzml.ImzMLParser import ImzMLParser, PortableSpectrumReader

from .utils import find_ibd_path, nearest_intensities, scan_mz_bounds

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
    mz_tolerance: Optional[float]
        Only used for "processed" mode files. Maximum allowed distance
        between a target m/z and the nearest measured peak, in the same units
        as `mz` itself (e.g. `mz_tolerance=0.005` next to `mz=[798.54]`). A
        pixel with no peak within tolerance gets 0 for that channel instead of
        the (too distant) nearest peak's intensity.
        Default: None (always use the nearest peak, however far away)
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created
        filesystem.
        Default: {}

    Notes
    -----
    imzML stores spectra in one of two modes:

    * "continuous": every pixel shares one m/z axis, so the intensity array at
      each pixel already lines up with every other pixel's. This reader
      detects that case structurally (identical m/z byte offset and length for
      every spectrum) and reads it directly, with no resampling.
    * "processed": each pixel has its own m/z axis (typical for high
      resolution / profile data). There is no single true set of channels, so
      this reader resamples each spectrum onto shared target m/z values via
      nearest-neighbor lookup, given explicitly via `mz` or auto-generated
      with `mz_step` or `n_bins`.
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
        mz_tolerance: float | None = None,
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
        ):
            parser = ImzMLParser(imzml_f, ibd_file=ibd_f)

        self._imzmldict: dict[str, Any] = parser.imzmldict
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

        self._mz_tolerance = mz_tolerance
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
    def mz_tolerance(self) -> float | None:
        """The tolerance (same units as m/z) passed to nearest-neighbor lookup
        for "processed" mode files (None means no tolerance: always use the
        nearest peak).
        """
        return self._mz_tolerance

    @staticmethod
    def _read_row(
        fs: AbstractFileSystem,
        ibd_path: str,
        portable: PortableSpectrumReader,
        pixels: list[tuple[int, int]],
        width: int,
        mz_axis: np.ndarray,
        continuous: bool,
        mz_tolerance: float | None,
    ) -> np.ndarray:
        """One (channels, width) row: all pixels at a given (z, y)."""
        out = np.zeros((len(mz_axis), width), dtype=np.float32)
        if not pixels:
            return out

        with fs.open(ibd_path, "rb") as f:
            for x0, spectrum_index in pixels:
                mzs, intensities = portable.read_spectrum_from_file(f, spectrum_index)
                if continuous:
                    out[:, x0] = intensities
                else:
                    out[:, x0] = nearest_intensities(
                        mzs, intensities, mz_axis, mz_tolerance
                    )

        return out

    def _create_dask_array(self) -> da.Array:
        n_channels = len(self._mz_axis)
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
        coords = {
            dimensions.DimensionNames.Channel: [f"{mz:.4f}" for mz in self._mz_axis]
        }
        return xr.DataArray(
            image_data,
            dims=list(dimensions.DEFAULT_DIMENSION_ORDER),
            coords=coords,
            attrs={
                constants.METADATA_UNPROCESSED: self._imzmldict,
                constants.METADATA_PROCESSED: self._imzmldict,
            },
        )

    def _read_delayed(self) -> xr.DataArray:
        return self._to_data_array(self._create_dask_array())

    def _read_immediate(self) -> xr.DataArray:
        arr = np.zeros(
            (len(self._mz_axis), self._depth, self._height, self._width),
            dtype=np.float32,
        )
        with self._fs.open(self._ibd_path, "rb") as f:
            for i, (x, y, z) in enumerate(self._portable.coordinates):
                mzs, intensities = self._portable.read_spectrum_from_file(f, i)
                if self._continuous:
                    arr[:, z - 1, y - 1, x - 1] = intensities
                else:
                    arr[:, z - 1, y - 1, x - 1] = nearest_intensities(
                        mzs, intensities, self._mz_axis, self._mz_tolerance
                    )

        return self._to_data_array(arr)
