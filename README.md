# bioio-imzml

[![CI](https://github.com/DBP008/bioio-imzml/actions/workflows/ci.yml/badge.svg)](https://github.com/DBP008/bioio-imzml/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/bioio-imzml.svg)](https://pypi.org/project/bioio-imzml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

A *basic* [BioIO](https://github.com/bioio-devs/bioio) reader plugin for imzML mass
spectrometry imaging (MSI) data, read with
[pyimzML](https://github.com/alexandrovteam/pyimzML).

## Installation

```bash
pip install bioio-imzml
```

Requires a sibling `.imzML` + `.ibd` file pair (the standard imzML layout).

## Usage

```python
from bioio import BioImage

img = BioImage("sample.imzML")
img.dims.order  # "TCZYX" -- C is the m/z axis
img.channel_names  # "<m/z>±<tolerance>" strings, e.g. "798.5400±0.0000"
img.data  # (T, C, Z, Y, X) numpy array
```

imzML-specific options (`mz`, `mz_step`, `n_bins`, `mz_tolerance_absolute`,
`mz_tolerance_relative`) work the same way through `BioImage`, since it
forwards unrecognized keyword arguments straight to the reader. Pass
`reader=bioio_imzml.Reader` to skip plugin auto-detection (useful when more
than one installed plugin could claim the file):

```python
import bioio_imzml
from bioio import BioImage

# "processed" mode files (one m/z axis per pixel) need target channels:
img = BioImage("sample.imzML", reader=bioio_imzml.Reader, mz=[798.54, 826.57, 885.55])

# reject a target with no real peak nearby instead of returning whatever
# peak happens to be closest, however far away. mz_tolerance_absolute and
# mz_tolerance_relative are both in the same units as mz (m/z) -- relative
# is a plain fraction, not a ppm count, so convert yourself (3 ppm = 3e-6).
# They combine per channel as: tolerance = absolute + m/z * relative
img = BioImage(
    "sample.imzML",
    reader=bioio_imzml.Reader,
    mz=[798.54, 826.57],
    mz_tolerance_absolute=0.005,
    mz_tolerance_relative=3e-6,  # 3 ppm
)
img.reader.mz_tolerance  # the resulting per-channel tolerance, e.g. [0.0074, 0.0075]
img.channel_names  # ["798.5400±0.0074", "826.5700±0.0075"]

# leave both unset and "processed" mode files get a tolerance for free:
# half the distance to each target's nearest neighboring target, so windows
# never overlap (a lone target with no neighbor is left unbounded).
img = BioImage("sample.imzML", reader=bioio_imzml.Reader, mz=[798.54, 826.57, 885.55])
img.reader.mz_tolerance  # e.g. [14.015, 14.015, 29.49] (half the gaps above/below)

# or let the reader pick evenly spaced channels across the file's m/z range,
# either a fixed count (n_bins) or a fixed step (mz_step) in m/z units:
img = BioImage("sample.imzML", reader=bioio_imzml.Reader, n_bins=512)
img = BioImage("sample.imzML", reader=bioio_imzml.Reader, mz_step=0.1)
```

## Auto peak-picking

Don't know which m/z channels a file actually has signal at? `auto_pick_peaks`
finds candidate peaks on the file's mean spectrum, then drops candidates that
are too rare across pixels or spatially unstructured (noise/matrix artifacts
rather than real signal):

```python
import bioio_imzml
from bioio import BioImage

# pin a tolerance and reuse it for picking and extraction, so extraction
# matches what pixel_frequency/spatial_chaos actually scored. Size it to
# bin_width, not to ppm mass-accuracy precision: candidates come from a
# bin_width-binned mean spectrum, so a candidate's reported m/z can be off
# from the true peak by up to ~bin_width/2.
bin_width = 0.05
tol_abs = bin_width

result = bioio_imzml.auto_pick_peaks(
    "sample.imzML",
    min_mz=650,
    max_mz=850,
    bin_width=bin_width,
    mz_tolerance_absolute=tol_abs,
)
result.mzs  # candidate m/z values, sorted by descending intensity
result.pixel_frequency  # fraction of pixels with signal, one per mz
result.spatial_chaos  # 0 (structured) .. 1 (spatially random), one per mz

if len(result.mzs) == 0:
    # min_pixel_frequency/max_spatial_chaos defaults can reject every
    # candidate on data with sparse per-pixel peak-picking (e.g.
    # single-cell-resolution processed-mode files) -- loosen or disable a
    # filter rather than pass an empty mz list on to Reader/BioImage:
    result = bioio_imzml.auto_pick_peaks(
        "sample.imzML",
        min_mz=650,
        max_mz=850,
        bin_width=bin_width,
        mz_tolerance_absolute=tol_abs,
        max_spatial_chaos=None,
    )

img = BioImage(
    "sample.imzML",
    reader=bioio_imzml.Reader,
    mz=result.mzs,
    mz_tolerance_absolute=tol_abs,
)
```

Tune detection sensitivity (`snr_threshold`, `min_relative_intensity`,
`min_separation_mz`) and the quality filters (`min_pixel_frequency`,
`max_spatial_chaos`) as keyword arguments; see the docstring for defaults.
`snr_threshold`, `min_pixel_frequency`, and `max_spatial_chaos` each accept
`None` to disable that filter -- passing `None` for both quality filters
also skips the per-pixel pass over the file entirely (the slow part),
leaving `result.pixel_frequency`/`result.spatial_chaos` as `NaN`.
`bioio_imzml.peak_picking` also exposes the individual steps --
`mean_spectrum`, `find_peaks_in_spectrum`, and
`pixel_frequency_and_spatial_chaos` -- to inspect intermediate results or why
a candidate was dropped before committing to thresholds.

## Continuous vs. processed mode

imzML stores spectra in one of two ways:

- **continuous**: every pixel shares one m/z axis, so intensities already line
  up across pixels. Detected automatically (identical m/z byte offset and
  length for every spectrum) and read directly -- no resampling, no channel
  arguments needed.
- **processed**: each pixel has its own m/z axis (typical for high-resolution
  profile data). There's no single true channel set, so this reader resamples
  every spectrum onto shared target m/z values by nearest-neighbor lookup,
  given via `mz=` or auto-generated with `n_bins=`.

`reader.is_continuous` reports which case applies to a given file.

## Development

```bash
uv sync --extra test
uv run pytest
uv run ruff check .
uv run ruff format .
uv run ty check
```

Bump the version (updates `pyproject.toml`) and tag a release to publish to
PyPI via CI:

```bash
uv version --bump patch  # or minor / major
git commit -am "Bump version"
git tag "v$(uv version --short)"
git push --tags
```

## License

[BSD-3-Clause](LICENSE)
