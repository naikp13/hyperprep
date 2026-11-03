"""
Input / output helpers.

* Pure-numpy ENVI reader/writer (BSQ, BIL, BIP; all common data types), so the
  toolkit works without GDAL.
* GeoTIFF / generic raster support through ``rasterio`` when installed.
* Spectral library text files (e.g. panel radiance averages exported by
  Specim/LUMO, ASD exports, hylite ``loadLibraryTXT`` style files).
* Navigation / IMU text files are handled in :mod:`hyperprep.drone.navigation`.
"""
from __future__ import annotations

import os
import re
from typing import Dict, Optional, Union

import numpy as np

from .image import HSImage

ENVI_DTYPES = {
    1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64,
    6: np.complex64, 9: np.complex128, 12: np.uint16, 13: np.uint32,
    14: np.int64, 15: np.uint64,
}
ENVI_DTYPES_INV = {np.dtype(v): k for k, v in ENVI_DTYPES.items()}


# --------------------------------------------------------------------------- ENVI
def _header_path(path: str) -> str:
    if path.lower().endswith('.hdr'):
        return path
    base = os.path.splitext(path)[0]
    for cand in (path + '.hdr', base + '.hdr'):
        if os.path.exists(cand):
            return cand
    return base + '.hdr'


def _data_path(hdr: str) -> str:
    base = os.path.splitext(hdr)[0]
    for ext in ('', '.dat', '.img', '.raw', '.bsq', '.bil', '.bip', '.DAT', '.IMG', '.RAW'):
        if os.path.exists(base + ext) and not (base + ext).lower().endswith('.hdr'):
            return base + ext
    raise FileNotFoundError("Could not find binary data file for header %s" % hdr)


def read_envi_header(path: str) -> Dict[str, Union[str, list]]:
    """Parse an ENVI header into a dict (lower-case keys; ``{}`` blocks become lists of strings)."""
    with open(_header_path(path), 'r', errors='ignore') as f:
        txt = f.read()
    if not txt.lstrip().upper().startswith('ENVI'):
        raise ValueError("%s is not an ENVI header" % path)
    header: Dict[str, Union[str, list]] = {}
    # join multi-line {...} blocks
    for m in re.finditer(r'^\s*([^=\n]+?)\s*=\s*(\{.*?\}|[^\n]*)', txt, flags=re.M | re.S):
        key, val = m.group(1).strip().lower(), m.group(2).strip()
        if val.startswith('{'):
            inner = val[1:-1].strip()
            if key == 'description':
                header[key] = inner
            else:
                header[key] = [v.strip() for v in inner.split(',')] if inner else []
        else:
            header[key] = val
    return header


def _as_float_list(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip('{}').split(',')
    try:
        return np.array([float(x) for x in v if str(x).strip() != ''])
    except ValueError:
        return None


def read_envi(path: str, memmap: bool = False, mask_zero: bool = False, dtype=np.float32) -> HSImage:
    """Read an ENVI image into an :class:`HSImage`.

    Args:
        path: path to the ``.hdr`` or binary file.
        memmap: return a memory-mapped array instead of loading data into RAM (no dtype conversion).
        mask_zero: set pixels that are zero in every band to NaN (common no-data convention).
        dtype: output dtype when not memory mapping.
    """
    hdr_path = _header_path(path)
    h = read_envi_header(hdr_path)
    samples, lines, bands = int(h['samples']), int(h['lines']), int(h['bands'])
    dt = np.dtype(ENVI_DTYPES[int(h.get('data type', 4))])
    if int(h.get('byte order', 0)) == 1:
        dt = dt.newbyteorder('>')
    offset = int(h.get('header offset', 0))
    interleave = str(h.get('interleave', 'bsq')).lower()
    shapes = {'bsq': (bands, lines, samples), 'bil': (lines, bands, samples), 'bip': (lines, samples, bands)}
    axes = {'bsq': (1, 2, 0), 'bil': (0, 2, 1), 'bip': (0, 1, 2)}
    dpath = _data_path(hdr_path)
    raw = np.memmap(dpath, dtype=dt, mode='r', offset=offset, shape=shapes[interleave])
    arr = np.transpose(raw, axes[interleave])
    if not memmap:
        arr = np.array(arr, dtype=dtype)
        if mask_zero:
            arr[(arr == 0).all(axis=-1)] = np.nan
        if 'data ignore value' in h:
            try:
                arr[arr == float(h['data ignore value'])] = np.nan
            except (TypeError, ValueError):
                pass

    wav = _as_float_list(h.get('wavelength'))
    if wav is not None and len(wav) == bands:
        units = str(h.get('wavelength units', 'nm')).lower()
        if units.startswith('micro') or units in ('um', 'µm') or (np.nanmax(wav) < 30):
            wav = wav * 1000.0
    else:
        wav = None
    fwhm = _as_float_list(h.get('fwhm'))
    if fwhm is not None and (len(fwhm) != bands):
        fwhm = None
    elif fwhm is not None and wav is not None and np.nanmax(fwhm) < 1 and np.nanmax(wav) > 100:
        fwhm = fwhm * 1000.0
    bbl = _as_float_list(h.get('bbl'))
    transform, crs = _parse_map_info(h)
    meta = {k: v for k, v in h.items() if k not in ('wavelength', 'fwhm', 'bbl', 'samples', 'lines', 'bands')}
    meta['source'] = os.path.abspath(dpath)
    return HSImage(arr, wav, fwhm, transform, crs, meta, None if bbl is None or len(bbl) != bands else bbl)


def _parse_map_info(h):
    mi = h.get('map info')
    if not mi or len(mi) < 7:
        return None, h.get('coordinate system string')
    try:
        px, py = float(mi[1]) - 1.0, float(mi[2]) - 1.0  # ENVI uses 1-based reference pixels
        e, n = float(mi[3]), float(mi[4])
        dx, dy = float(mi[5]), float(mi[6])
        rot = 0.0
        for item in mi[7:]:
            if 'rotation' in item.lower():
                rot = float(item.split('=')[-1])
        x0 = e - px * dx
        y0 = n + py * dy
        if rot:
            r = np.deg2rad(rot)
            transform = (x0, dx * np.cos(r), dy * np.sin(r), y0, dx * np.sin(r), -dy * np.cos(r))
        else:
            transform = (x0, dx, 0.0, y0, 0.0, -dy)
        crs = h.get('coordinate system string')
        if crs is None and mi[0].strip().upper() == 'UTM' and len(mi) >= 9:
            crs = 'UTM zone %s %s' % (mi[7], mi[8])
        return transform, crs
    except (ValueError, IndexError):
        return None, None


def write_envi(path: str, image: HSImage, interleave: str = 'bsq', dtype=np.float32, extra: Optional[dict] = None):
    """Write an :class:`HSImage` to ENVI format (``path`` may end with .hdr, .dat or nothing)."""
    base = os.path.splitext(path)[0] if path.lower().endswith(('.hdr', '.dat', '.img')) else path
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    arr = np.asarray(image.data, dtype=dtype)
    interleave = interleave.lower()
    order = {'bsq': (2, 0, 1), 'bil': (0, 2, 1), 'bip': (0, 1, 2)}[interleave]
    np.ascontiguousarray(np.transpose(arr, order)).tofile(base + '.dat')

    lines = ['ENVI',
             'description = {written by hyperprep}',
             'samples = %d' % image.cols, 'lines = %d' % image.rows, 'bands = %d' % image.bands,
             'header offset = 0', 'file type = ENVI Standard',
             'data type = %d' % ENVI_DTYPES_INV[np.dtype(dtype)],
             'interleave = %s' % interleave, 'byte order = 0']
    if image.wavelengths is not None:
        lines.append('wavelength units = Nanometers')
        lines.append('wavelength = {%s}' % ', '.join('%.4f' % w for w in image.wavelengths))
    if image.fwhm is not None:
        lines.append('fwhm = {%s}' % ', '.join('%.4f' % w for w in image.fwhm))
    if image.bbl is not None:
        lines.append('bbl = {%s}' % ', '.join(str(int(b)) for b in image.bbl))
    if image.transform is not None:
        x0, dx, rx, y0, ry, dy = image.transform
        mi = ['Arbitrary', '1', '1', repr(float(x0)), repr(float(y0)), repr(float(abs(dx))), repr(float(abs(dy)))]
        if rx or ry:
            mi.append('rotation=%.6f' % np.rad2deg(np.arctan2(ry, dx)))
        lines.append('map info = {%s}' % ', '.join(mi))
    if isinstance(image.crs, str) and image.crs.strip().startswith(('PROJCS', 'GEOGCS', 'PROJCRS', 'GEOGCRS')):
        lines.append('coordinate system string = {%s}' % image.crs)
    lines.append('data ignore value = nan' if np.issubdtype(np.dtype(dtype), np.floating) else 'data ignore value = 0')
    skip = {'description', 'samples', 'lines', 'bands', 'header offset', 'file type', 'data type', 'interleave',
            'byte order', 'wavelength units', 'wavelength', 'fwhm', 'bbl', 'map info', 'coordinate system string',
            'data ignore value', 'source'}
    for k, v in {**image.meta, **(extra or {})}.items():
        if k in skip or not isinstance(v, (str, int, float, list, tuple)):
            continue
        if isinstance(v, (list, tuple)):
            v = '{%s}' % ', '.join(str(x) for x in v)
        lines.append('%s = %s' % (k, v))
    with open(base + '.hdr', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return base + '.hdr'


# ------------------------------------------------------------------- rasterio
def read_raster(path: str, wavelengths=None, band_dim_last: bool = True) -> HSImage:
    """Read any GDAL-supported raster (GeoTIFF, JP2, ...) using rasterio."""
    try:
        import rasterio
    except ImportError as e:  # pragma: no cover
        raise ImportError("read_raster requires rasterio: pip install hyperprep[geo]") from e
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        nodata = src.nodata
        transform = tuple(src.transform.to_gdal())
        crs = src.crs.to_wkt() if src.crs else None
        desc = src.descriptions
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr = np.transpose(arr, (1, 2, 0)) if band_dim_last else arr
    if wavelengths is None:
        try:
            wavelengths = [float(re.findall(r'[\d.]+', d)[0]) for d in desc]
        except Exception:
            wavelengths = None
    return HSImage(arr, wavelengths, transform=transform, crs=crs, meta={'source': os.path.abspath(path)})


def write_geotiff(path: str, image: HSImage, dtype='float32', nodata=np.nan):
    """Write an :class:`HSImage` to a (Cloud-Optimised-friendly, tiled) GeoTIFF using rasterio."""
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as e:  # pragma: no cover
        raise ImportError("write_geotiff requires rasterio: pip install hyperprep[geo]") from e
    transform = Affine.from_gdal(*image.transform) if image.transform is not None else Affine.identity()
    profile = dict(driver='GTiff', height=image.rows, width=image.cols, count=image.bands, dtype=dtype,
                   transform=transform, crs=image.crs, nodata=nodata, tiled=True, compress='deflate',
                   interleave='band')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(np.transpose(image.data, (2, 0, 1)).astype(dtype))
        for i, w in enumerate(image.wavelengths):
            dst.set_band_description(i + 1, '%.2f nm' % w)
    return path


# ------------------------------------------------------------------ generic
def load(path: str, **kwds) -> HSImage:
    """Load an image, picking the reader from the file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff', '.jp2', '.vrt'):
        return read_raster(path, **kwds)
    if ext in ('.nc', '.he5', '.h5', '.xml', '.zip'):
        from .satellite.readers import read_product
        return read_product(path, **kwds)
    return read_envi(path, **kwds)


def save(path: str, image: HSImage, **kwds) -> str:
    """Save an image, picking the writer from the file extension (default ENVI)."""
    if path.lower().endswith(('.tif', '.tiff')):
        return write_geotiff(path, image, **kwds)
    return write_envi(path, image, **kwds)


# ------------------------------------------------------------- spectral libs
def read_spectrum_txt(path: str, skip_non_numeric: bool = True):
    """Read a two-or-more column text spectrum (wavelength, value[, value ...]).

    Handles the ``*_RadAverage.txt`` panel exports used in the FENIX illumination
    notebook as well as ASD/SVC style ASCII exports. Wavelengths in micrometres
    are converted to nanometres.

    Returns:
        ``(wavelengths, values)`` where ``values`` has shape ``(n_wavelengths,)``
        for a single spectrum or ``(n_wavelengths, n_spectra)``.
    """
    rows = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            parts = re.split(r'[\s,;]+', line.strip())
            try:
                rows.append([float(p) for p in parts if p != ''])
            except ValueError:
                if not skip_non_numeric:
                    raise
    rows = [r for r in rows if len(r) >= 2]
    n = min(len(r) for r in rows)
    arr = np.array([r[:n] for r in rows])
    wav, vals = arr[:, 0], arr[:, 1:]
    if np.nanmax(wav) < 30:
        wav = wav * 1000.0
    return wav, vals[:, 0] if vals.shape[1] == 1 else vals
