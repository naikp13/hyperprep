"""
Readers for spaceborne imaging spectrometer products.

Supported (best effort, tag-name based so minor format revisions still parse):

* **EnMAP** L1B (VNIR + SWIR GeoTIFF), L1C / L2A (single GeoTIFF) with ``METADATA.XML``
* **PRISMA** L1 (radiance) and L2B/C/D (reflectance) HDF-EOS5 ``.he5``
* **EMIT** L1B RAD and L2A RFL NetCDF (with GLT for orthorectification)
* **DESIS** L1B/L1C/L2A GeoTIFF with ``METADATA.xml``

Optional dependencies: ``rasterio`` (GeoTIFF) and ``h5py`` (HDF5/NetCDF4);
install with ``pip install hyperprep[satellite]``.
"""
from __future__ import annotations

import glob
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, Optional

import numpy as np

from ..image import HSImage


# ------------------------------------------------------------------ helpers
def _local(tag: str) -> str:
    return tag.split('}')[-1]


def _xml_band_table(root, fields=('wavelengthCenterOfBand', 'FWHMOfBand', 'GainOfBand', 'OffsetOfBand')) -> Dict[str, np.ndarray]:
    """Collect per-band values from ``bandID``/``band`` elements in an EnMAP/DESIS metadata XML."""
    out = {f: [] for f in fields}
    for el in root.iter():
        if _local(el.tag) in ('bandID', 'band'):
            vals = {_local(c.tag): c.text for c in el}
            if fields[0] in vals:
                for f in fields:
                    try:
                        out[f].append(float(vals.get(f)))
                    except (TypeError, ValueError):
                        out[f].append(np.nan)
    return {k: np.array(v) for k, v in out.items() if v}


def _xml_find(root, *names) -> Optional[str]:
    """Text of the first element whose local tag matches the path ``names`` (suffix match)."""
    for el in root.iter():
        if _local(el.tag) == names[0]:
            node = el
            ok = True
            for n in names[1:]:
                nxt = next((c for c in node if _local(c.tag) == n), None)
                if nxt is None:
                    ok = False
                    break
                node = nxt
            if ok and node.text and node.text.strip():
                return node.text.strip()
    return None


def _find(folder: str, *patterns) -> Optional[str]:
    for p in patterns:
        hits = sorted(glob.glob(os.path.join(folder, p))) + sorted(glob.glob(os.path.join(folder, '**', p), recursive=True))
        if hits:
            return hits[0]
    return None


def _read_tif(path):
    try:
        import rasterio
    except ImportError as e:  # pragma: no cover
        raise ImportError("Reading GeoTIFF products requires rasterio: pip install hyperprep[satellite]") from e
    with rasterio.open(path) as src:
        arr = np.transpose(src.read(), (1, 2, 0)).astype(np.float32)
        nod = src.nodata
        transform = tuple(src.transform.to_gdal())
        crs = src.crs.to_wkt() if src.crs else None
    if nod is not None:
        arr[arr == nod] = np.nan
    return arr, transform, crs


# ------------------------------------------------------------------ EnMAP
def read_enmap(folder: str, scale_reflectance: bool = True) -> HSImage:
    """Read an (unzipped) EnMAP product folder.

    L1B: VNIR and SWIR cubes are converted to radiance with the per-band gain/offset and
    concatenated (overlap bands are kept; use :func:`hyperprep.satellite.merge_overlap`).
    L2A: reflectance is divided by 10000 when ``scale_reflectance`` and values exceed 1.5.
    """
    xml = _find(folder, '*METADATA.XML', '*metadata.xml')
    if xml is None:
        raise FileNotFoundError("No EnMAP METADATA.XML in %s" % folder)
    root = ET.parse(xml).getroot()
    tbl = _xml_band_table(root)
    meta = {'sensor': 'EnMAP', 'metadata': xml}
    for key, path in (('sun_zenith', ('sunElevationAngle', 'center')), ('sun_azimuth', ('sunAzimuthAngle', 'center')),
                      ('acquisition_time', ('startTime',)), ('level', ('processingLevel',))):
        v = _xml_find(root, *path)
        if v is not None:
            meta[key] = v
    if 'sun_zenith' in meta:  # stored as elevation
        meta['sun_elevation'] = float(meta.pop('sun_zenith'))
        meta['sun_zenith'] = 90.0 - meta['sun_elevation']
    single = _find(folder, '*SPECTRAL_IMAGE.TIF', '*SPECTRAL_IMAGE.tif')
    if single is not None:
        arr, tr, crs = _read_tif(single)
        if scale_reflectance and 'L2A' in os.path.basename(single).upper() and np.nanpercentile(arr, 99) > 1.5:
            arr = arr / 10000.0
        meta['units'] = 'reflectance' if 'L2A' in os.path.basename(single).upper() else 'radiance'
    else:
        vnir = _find(folder, '*SPECTRAL_IMAGE_VNIR.TIF', '*SPECTRAL_IMAGE_VNIR.tif')
        swir = _find(folder, '*SPECTRAL_IMAGE_SWIR.TIF', '*SPECTRAL_IMAGE_SWIR.tif')
        if vnir is None or swir is None:
            raise FileNotFoundError("No EnMAP spectral image in %s" % folder)
        a, tr, crs = _read_tif(vnir)
        b, _, _ = _read_tif(swir)
        arr = np.concatenate([a, b], axis=-1)
        meta['units'] = 'radiance'
        meta['n_vnir'] = a.shape[-1]
    nb = arr.shape[-1]
    if meta['units'] == 'radiance' and 'GainOfBand' in tbl and len(tbl['GainOfBand']) == nb:
        # L1B DN -> radiance with the per-band gain/offset (units as defined in the EnMAP
        # product specification; check before mixing with other sensors)
        arr = arr * tbl['GainOfBand'] + tbl['OffsetOfBand']
        meta['radiance_units'] = 'EnMAP L1B (gain * DN + offset)'
    wav = tbl.get('wavelengthCenterOfBand')
    fwhm = tbl.get('FWHMOfBand')
    if wav is None or len(wav) != nb:
        wav, fwhm = None, None
    return HSImage(arr, wav, fwhm, tr, crs, meta)


# ------------------------------------------------------------------ PRISMA
def read_prisma(path: str, scale: bool = True, drop_empty: bool = True) -> HSImage:
    """Read a PRISMA L1 (radiance) or L2 (reflectance) ``.he5`` file.

    VNIR and SWIR cubes are stacked in increasing wavelength order. Bands with a centre
    wavelength of 0 (unused detector rows) are dropped. No map transform is set; use the
    ``Latitude``/``Longitude`` layers returned in ``meta['lat']`` / ``meta['lon']`` with
    :func:`hyperprep.ortho.build_glt` (after converting to UTM) to orthorectify.
    """
    import h5py

    with h5py.File(path, 'r') as f:
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}
        swaths = f['HDFEOS/SWATHS']
        name = [k for k in swaths.keys() if 'HCO' in k] or list(swaths.keys())
        sw = swaths[name[0]]
        level = name[0]
        vnir = np.asarray(sw['Data Fields/VNIR_Cube'], np.float32)  # (lines, bands, samples)
        swir = np.asarray(sw['Data Fields/SWIR_Cube'], np.float32)
        geo = sw['Geolocation Fields'] if 'Geolocation Fields' in sw else None
        lat = lon = None
        if geo is not None:
            for k in geo.keys():
                if k.lower().startswith('latitude'):
                    lat = np.asarray(geo[k])
                if k.lower().startswith('longitude'):
                    lon = np.asarray(geo[k])
    vnir = np.transpose(vnir, (0, 2, 1))
    swir = np.transpose(swir, (0, 2, 1))
    is_l2 = 'L2' in level
    if scale:
        if is_l2:
            for cube, k in ((vnir, 'Vnir'), (swir, 'Swir')):
                mn, mx = float(attrs.get('L2Scale%sMin' % k, 0)), float(attrs.get('L2Scale%sMax' % k, 1))
                cube *= (mx - mn) / 65535.0
                cube += mn
        else:
            vnir = vnir / float(attrs.get('ScaleFactor_Vnir', 1)) - float(attrs.get('Offset_Vnir', 0))
            swir = swir / float(attrs.get('ScaleFactor_Swir', 1)) - float(attrs.get('Offset_Swir', 0))
    cw = np.concatenate([np.asarray(attrs.get('List_Cw_Vnir'), float), np.asarray(attrs.get('List_Cw_Swir'), float)])
    fw = np.concatenate([np.asarray(attrs.get('List_Fwhm_Vnir'), float), np.asarray(attrs.get('List_Fwhm_Swir'), float)])
    arr = np.concatenate([vnir, swir], axis=-1)
    keep = cw > 0 if drop_empty else np.ones_like(cw, bool)
    order = np.argsort(cw[keep])
    arr, cw, fw = arr[..., keep][..., order], cw[keep][order], fw[keep][order]
    meta = {'sensor': 'PRISMA', 'level': level, 'units': 'reflectance' if is_l2 else 'radiance',
            'source': os.path.abspath(path)}
    for k_in, k_out in (('Sun_zenith_angle', 'sun_zenith'), ('Sun_azimuth_angle', 'sun_azimuth'),
                        ('Product_StartTime', 'acquisition_time')):
        if k_in in attrs:
            v = attrs[k_in]
            meta[k_out] = v.decode() if isinstance(v, bytes) else (float(v) if np.ndim(v) == 0 and not isinstance(v, str) else v)
    if lat is not None:
        meta['lat'], meta['lon'] = lat, lon
    return HSImage(arr.astype(np.float32), cw, fw, meta=meta)


# ------------------------------------------------------------------ EMIT
def read_emit(path: str, ortho: bool = False, mask_bad_bands: bool = True) -> HSImage:
    """Read an EMIT L1B RAD or L2A RFL NetCDF file.

    Args:
        ortho: apply the product's GLT to return a map-geometry image with geotransform and CRS.
        mask_bad_bands: set ``bbl`` from ``good_wavelengths`` (L2A).
    """
    import h5py

    with h5py.File(path, 'r') as f:
        var = 'reflectance' if 'reflectance' in f else ('radiance' if 'radiance' in f else None)
        if var is None:
            raise ValueError("No radiance/reflectance variable in %s" % path)
        arr = np.asarray(f[var], np.float32)
        fill = f[var].attrs.get('_FillValue', -9999)
        sbp = f['sensor_band_parameters']
        wav = np.asarray(sbp['wavelengths'], float)
        fwhm = np.asarray(sbp['fwhm'], float) if 'fwhm' in sbp else None
        good = np.asarray(sbp['good_wavelengths']).astype(bool) if 'good_wavelengths' in sbp else None
        glt_x = np.asarray(f['location/glt_x']) if 'location' in f and 'glt_x' in f['location'] else None
        glt_y = np.asarray(f['location/glt_y']) if 'location' in f and 'glt_y' in f['location'] else None
        gt = tuple(np.asarray(f.attrs['geotransform'], float)) if 'geotransform' in f.attrs else None
        crs = f.attrs.get('spatial_ref')
        tstart = f.attrs.get('time_coverage_start')
    arr[arr == np.float32(np.ravel(fill)[0])] = np.nan
    meta = {'sensor': 'EMIT', 'units': var, 'source': os.path.abspath(path)}
    if tstart is not None:
        meta['acquisition_time'] = tstart.decode() if isinstance(tstart, bytes) else str(tstart)
    if isinstance(crs, bytes):
        crs = crs.decode()
    img = HSImage(arr, wav, fwhm, None, None, meta, good if (mask_bad_bands and good is not None) else None)
    if ortho:
        if glt_x is None or gt is None:
            raise ValueError("Product has no GLT / geotransform")
        img = emit_apply_glt(img, glt_x, glt_y, gt, crs)
    else:
        img.meta['glt_x'], img.meta['glt_y'], img.meta['geotransform'], img.meta['spatial_ref'] = glt_x, glt_y, gt, crs
    return img


def emit_apply_glt(image: HSImage, glt_x: np.ndarray, glt_y: np.ndarray, geotransform, crs=None) -> HSImage:
    """Orthorectify raw EMIT data with its (1-based, 0 = no data) GLT."""
    from ..ortho import GLT, apply_glt

    g = GLT(np.asarray(glt_y, np.int32) - 1, np.asarray(glt_x, np.int32) - 1, tuple(geotransform), crs)
    out = apply_glt(image, g)
    for k in ('glt_x', 'glt_y', 'geotransform', 'spatial_ref'):
        out.meta.pop(k, None)
    return out


# ------------------------------------------------------------------ DESIS
def read_desis(folder: str) -> HSImage:
    """Read a DESIS product folder (``*SPECTRAL_IMAGE.tif`` + ``*METADATA.xml``); L1B DN are converted
    to radiance with the per-band gain/offset when present."""
    tif = _find(folder, '*SPECTRAL_IMAGE.tif', '*SPECTRAL_IMAGE.TIF')
    xml = _find(folder, '*METADATA.xml', '*METADATA.XML')
    if tif is None or xml is None:
        raise FileNotFoundError("DESIS SPECTRAL_IMAGE.tif / METADATA.xml not found in %s" % folder)
    arr, tr, crs = _read_tif(tif)
    root = ET.parse(xml).getroot()
    tbl = _xml_band_table(root, ('wavelengthCenterOfBand', 'wavelengthWidthOfBand', 'gainOfBand', 'offsetOfBand'))
    meta = {'sensor': 'DESIS', 'metadata': xml}
    lvl = _xml_find(root, 'processingLevel') or ''
    meta['level'] = lvl
    if 'L2' in lvl.upper():
        arr = arr / 10000.0
        meta['units'] = 'reflectance'
    else:
        if 'gainOfBand' in tbl and len(tbl['gainOfBand']) == arr.shape[-1]:
            arr = arr * tbl['gainOfBand'] + np.nan_to_num(tbl.get('offsetOfBand', 0))
        meta['units'] = 'radiance'
    for key, path in (('sun_zenith', ('sunZenithAngle',)), ('sun_azimuth', ('sunAzimuthAngle',)),
                      ('acquisition_time', ('startTime',))):
        v = _xml_find(root, *path)
        if v is not None:
            meta[key] = v
    wav = tbl.get('wavelengthCenterOfBand')
    fwhm = tbl.get('wavelengthWidthOfBand')
    if wav is None or len(wav) != arr.shape[-1]:
        wav = fwhm = None
    return HSImage(arr, wav, fwhm, tr, crs, meta)


# ------------------------------------------------------------------ dispatch
def detect_sensor(path: str) -> str:
    name = os.path.basename(os.path.normpath(path)).upper()
    if name.startswith('ENMAP') or 'ENMAP' in name:
        return 'enmap'
    if name.startswith('PRS_') or name.endswith('.HE5'):
        return 'prisma'
    if name.startswith('EMIT') or name.endswith('.NC'):
        return 'emit'
    if name.startswith('DESIS') or 'DESIS' in name:
        return 'desis'
    if os.path.isdir(path):
        if _find(path, '*METADATA.XML') and _find(path, '*SPECTRAL_IMAGE*'):
            return 'enmap'
    raise ValueError("Cannot detect sensor for %s; call the reader directly" % path)


def read_product(path: str, sensor: Optional[str] = None, **kwds) -> HSImage:
    """Read a satellite product, detecting the sensor from the file / folder name.

    Zip archives (EnMAP / DESIS deliveries) are extracted next to the archive first.
    """
    if path.lower().endswith('.zip'):
        import zipfile
        dest = os.path.splitext(path)[0]
        if not os.path.isdir(dest):
            with zipfile.ZipFile(path) as z:
                z.extractall(dest)
        path = dest
    if path.lower().endswith('.xml'):
        path = os.path.dirname(path)
    sensor = (sensor or detect_sensor(path)).lower()
    reader = {'enmap': read_enmap, 'prisma': read_prisma, 'emit': read_emit, 'desis': read_desis}[sensor]
    return reader(path, **kwds)
