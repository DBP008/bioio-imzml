# bioio-imzml

[![CI](https://github.com/DBP008/bioio-imzml/actions/workflows/ci.yml/badge.svg)](https://github.com/DBP008/bioio-imzml/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/bioio-imzml.svg)](https://pypi.org/project/bioio-imzml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

A [BioIO](https://github.com/bioio-devs/bioio) reader plugin for imzML mass
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
img.channel_names  # m/z values, formatted as strings
img.data  # (T, C, Z, Y, X) numpy array
```

imzML-specific options (`mz`, `mz_step`, `n_bins`, `mz_tolerance`) work the same way
through `BioImage(..., mz=[...])`, since `BioImage` forwards unrecognized
keyword arguments straight to the reader. Or use the reader directly:

```python
from bioio_imzml import Reader

# "processed" mode files (one m/z axis per pixel) need target channels:
reader = Reader("sample.imzML", mz=[798.54, 826.57, 885.55])

# reject a target with no real peak nearby instead of returning whatever
# peak happens to be closest, however far away. mz_tolerance is in the same
# units as mz itself (m/z, i.e. Da):
reader = Reader("sample.imzML", mz=[798.54, 826.57], mz_tolerance=0.005)

# or let the reader pick evenly spaced channels across the file's m/z range,
# either a fixed count (n_bins) or a fixed step (mz_step) in m/z units:
reader = Reader("sample.imzML", n_bins=512)
reader = Reader("sample.imzML", mz_step=0.1)
```

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
