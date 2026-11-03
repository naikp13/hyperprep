# hyperprep

**One-stop pre-processing for hyperspectral imagery from terrestrial scanners, drones and satellites.**

`hyperprep` takes you from raw sensor data to analysis-ready reflectance, whichever platform you use. It wraps and extends
[hylite](https://github.com/samthiele/hylite) for terrestrial and 3-D workflows, and adds pure-numpy implementations of the
standard drone and satellite steps. You can run every step from Python or chain the steps in a YAML pipeline
(`hyperprep run config.yaml`).

```
pip install "hyperprep[all] @ git+https://github.com/naikp13/hyperprep"   # everything
pip install "git+https://github.com/naikp13/hyperprep"                    # core only: numpy, scipy, pyyaml
```

Extras: `terrestrial` (hylite), `satellite` (rasterio, h5py), `geo` (rasterio), `plot` (matplotlib), `dev` (pytest).

## What's included

| Stage | Terrestrial / lab scanners | Drone / airborne pushbroom | Satellite |
|---|---|---|---|
| **Read** | ENVI (pure numpy), sensor capture folders (FENIX, FX10/17/50, OWL, Rikola, Telops via hylite) | ENVI, HySpex RAD, GNSS/IMU (HySpexNav / PosPac / CSV) | EnMAP L1B/L1C/L2A, PRISMA L1/L2, EMIT L1B/L2A (+GLT), DESIS, any GeoTIFF |
| **Radiometric** | dark / white reference reflectance, bad pixel repair, lens & VNIR-SWIR shift (hylite) | DN→radiance (gain/offset), HySpex scale factor, bad detector repair, destriping, spike removal | DN→radiance, VNIR/SWIR overlap merge, smile estimation & correction, destriping |
| **Reflectance** | panel ELC, illumination model with sky-view and Oren-Nayar (Thiele et al. 2021) | ELC (1-n panels), downwelling-irradiance normalisation, sun-angle normalisation | TOA reflectance (earth-sun distance, solar spectrum), dark object subtraction, ELC, IARR / flat field |
| **Geometry** | projection onto point clouds (frame & pushbroom), boresight optimisation, hyperclouds | direct georeferencing (ray casting on plane or DEM), lever arm & boresight, GLT orthorectification, mosaicking | GLT orthorectification, bbox subsetting |
| **Illumination / BRDF** | topographic correction (cosine, C, SCS+C, Minnaert, ambient), colour matching between scenes | cross-track BRDF / vignetting normalisation, flight-line matching | DEM-based topographic correction |
| **Clean-up** (all platforms) | bad-band (water vapour) removal, Savitzky-Golay, spatial smoothing, MNF denoising, continuum removal, normalisation, spectral resampling (Gaussian SRF), band binning, cloud / shadow / water / no-data masks, value-range filtering | | |

## Quick start

```python
import hyperprep as hp
from hyperprep import spectral, noise, masks
from hyperprep.satellite import read_product, toa_reflectance, merge_overlap

img = read_product('ENMAP01-____L1B-DT0000012345_...')           # folder, .zip, .he5 or .nc
img = merge_overlap(img)                                         # VNIR/SWIR detector overlap
img = toa_reflectance(img, esun_path='tsis1_hsrs.txt')           # sun angle and date from metadata
img = masks.apply_mask(img, masks.dilate(masks.cloud_mask(img), 3))
img = spectral.remove_bad_bands(img)
img = spectral.savgol(img, window=5)
hp.save('enmap_toa.hdr', img)
```

The same steps as a pipeline:

```yaml
# configs/my_enmap.yaml
input: /data/ENMAP01-____L1B-DT0000012345_...
output: /out/enmap_toa.hdr
steps:
  - read: {sensor: enmap}
  - merge_overlap: {}
  - toa_reflectance: {esun_path: tsis1_hsrs.txt}
  - mask_clouds: {buffer: 3}
  - remove_bad_bands: {}
  - savgol: {window: 5}
```

```bash
hyperprep run configs/my_enmap.yaml
hyperprep steps          # list the available pipeline steps
hyperprep info img.hdr   # summarise an image or product
```

## Workflows

The [workflow guide](docs/workflows.md) gives the recommended order of steps for each platform, explains which choices
matter, and maps each cell of the original notebooks to its replacement function.

* **Terrestrial**: [`examples/01_sensor_correction.py`](examples/01_sensor_correction.py) →
  [`02_projection_mapping.py`](examples/02_projection_mapping.py) →
  [`03_illumination_correction.py`](examples/03_illumination_correction.py) →
  [`04_build_hypercloud.py`](examples/04_build_hypercloud.py)
* **Drone**: [`examples/05_drone_pushbroom.py`](examples/05_drone_pushbroom.py) runs end to end on synthetic data;
  for real data, start from [`configs/drone_hyspex.yaml`](configs/drone_hyspex.yaml)
* **Satellite**: [`examples/06_satellite.py`](examples/06_satellite.py), plus configs for
  [EMIT](configs/satellite_emit_l1b.yaml), [EnMAP](configs/satellite_enmap_l2a.yaml) and
  [PRISMA](configs/satellite_prisma_l1.yaml)

## Package layout

```
hyperprep/
  image.py          HSImage container (rows, cols, bands) + hylite conversion
  io.py             ENVI / GeoTIFF / spectral text I/O
  spectral.py       bad bands, resampling, smoothing, continuum removal, normalisation
  noise.py          noise/SNR, bad detector pixels, spikes, destriping, MNF
  masks.py          no-data, saturation, cloud, cirrus, shadow, water, vegetation
  solar.py          solar position, sun vector, earth-sun distance, solar irradiance
  geo.py            WGS84 <-> UTM, attitude rotation matrices
  topo.py           normals from DEM, incidence, Oren-Nayar, topographic corrections
  ortho.py          GLT build/apply, mosaicking
  calibration/      radiometric (white/dark, gain/offset, HySpex), ELC & panels, hylite sensors
  terrestrial/      projection (HyScene, pushbroom, boresight), illumination, hypercloud
  drone/            navigation tracks, direct georeferencing, BRDF & flight-line corrections
  satellite/        product readers, TOA reflectance, DOS, smile, overlap, subsetting
  pipeline.py       step registry + YAML/JSON runner
  cli.py            `hyperprep` command
```

Image arrays are `(rows, cols, bands)`. hylite uses `(x, y, bands)`, so use `HSImage.to_hylite()` and
`HSImage.from_hylite()` to convert between the two.

## Testing

```bash
pip install -e ".[dev,terrestrial]"
pytest
```

The tests use synthetic data. They check that each correction recovers a known reflectance, that georeferencing
matches analytic geometry, and that each reader parses a mock product. The hylite integration tests are skipped
when hylite isn't installed.

## Notes and limitations

* **Atmospheric correction for satellites** is scene-based (TOA + DOS, ELC, IARR). For physically based surface
  reflectance, use the agency L2 products (EnMAP L2A, PRISMA L2D, EMIT L2A) or ISOFIT, and use `hyperprep` for the steps
  before and after.
* **Solar irradiance** defaults to a 5778 K black body scaled to 1361 W m⁻². For quantitative TOA reflectance, pass a
  measured solar spectrum (TSIS-1 HSRS or Thuillier) with `esun_path`.
* **Direct georeferencing** assumes a 2.5-D surface (plane or DEM). For cliffs and open pits, project onto a point cloud
  with `hyperprep.terrestrial.projection` instead.
* The product readers were written from the public format specifications and tested on synthetic files. Please open an
  issue if a product revision doesn't parse.

## Credits

The terrestrial workflow builds on [hylite](https://github.com/samthiele/hylite) (please cite it when you use the
terrestrial modules) and on the illumination model of Thiele et al. (2021), *IEEE TGRS*,
https://doi.org/10.1109/TGRS.2021.3098725.

MIT licence.
