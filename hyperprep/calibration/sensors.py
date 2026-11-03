"""
Sensor-specific corrections for terrestrial / lab scanners via hylite.

Refactored from ``1_radiance_to_reflectance_Fenixsingle.ipynb``. The original
notebook kept three mutually exclusive cells (OWL / FX50 / FENIX), referenced an
un-imported ``Fenix`` class and mixed ``image``/``images``. Here one function
handles every supported sensor, sets the white panel spectrum consistently, and
optionally writes the result.

Requires ``hylite`` (``pip install hyperprep[terrestrial]``).
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Union

from ..image import HSImage

SENSORS = ('fenix', 'fx10', 'fx17', 'fx50', 'owl', 'rikola', 'rikola_hsc2', 'rikola_rsc1', 'telops')


def _sensor_class(name: str):
    from hylite import sensors

    table = {
        'fenix': sensors.Fenix, 'fx10': sensors.FX10, 'fx17': sensors.FX17, 'fx50': sensors.FX50,
        'owl': sensors.OWL, 'rikola': sensors.Rikola, 'rikola_hsc2': sensors.Rikola_HSC2,
        'rikola_rsc1': sensors.Rikola_RSC1, 'telops': sensors.TelopsNano,
    }
    try:
        return table[name.lower()]
    except KeyError:
        raise ValueError("Unknown sensor %r. Options are %s" % (name, ', '.join(SENSORS)))


def get_reference_target(name: str):
    """Return a hylite calibration target (e.g. ``'R90'``, ``'R55_SiSu'``, ``'PVC_White'``)."""
    import hylite.reference.spectra as spectra

    if hasattr(spectra, name):
        return getattr(spectra, name)
    for lib in (spectra.spectralon, spectra.PVC):
        if name in lib:
            return lib[name]
    raise ValueError("Unknown calibration target %r" % name)


def correct_sensor_folder(path: str, sensor: str = 'fenix', white_panel: Optional[str] = 'R90',
                          outdir: Optional[str] = None, as_hsimage: bool = False, verbose: bool = True, **kwds):
    """Apply sensor corrections (dark/white referencing, lens & smile correction, bad pixel
    replacement, VNIR-SWIR shift, ...) to a raw sensor capture folder.

    Args:
        path: capture folder as written by the sensor software (e.g. ``.../capture/`` parent).
        sensor: one of :data:`SENSORS`.
        white_panel: name of the white reference panel used during acquisition (hylite target
            name, e.g. ``'R90'`` or ``'R55_SiSu'``), or None to assume reflectance = 1.
        outdir: if given, the corrected image is saved as ``<outdir>/<folder>_refl.hdr``.
        as_hsimage: return an :class:`~hyperprep.image.HSImage` instead of a ``hylite.HyImage``.
        **kwds: forwarded to the hylite sensor, e.g. ``flip=True`` (camera mounted backwards),
            ``shift=True`` (FENIX VNIR/SWIR spatial shift, near-field only), ``bpr=True``,
            ``lens=True``, ``rad=True``.
    """
    cls = _sensor_class(sensor)
    if white_panel is not None and hasattr(cls, 'set_white_ref_spectra'):
        cls.set_white_ref_spectra(get_reference_target(white_panel))
    try:
        image = cls.correct_folder(path, verbose=verbose, **kwds)
    except TypeError:  # some hylite sensors (e.g. FX base) take no keywords in correct_folder
        image = cls.correct_folder(path)
    if outdir is not None:
        from hylite import io as hio

        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, os.path.basename(os.path.normpath(path)) + '_refl.hdr')
        hio.save(out, image)
        if verbose:
            print("Saved %s" % out)
    return HSImage.from_hylite(image) if as_hsimage else image


def batch_correct(paths: Union[str, List[str]], sensor: str = 'fenix', outdir: Optional[str] = None,
                  pattern: str = '*', **kwds) -> Dict[str, object]:
    """Run :func:`correct_sensor_folder` on many capture folders (a list or a glob of a parent dir).

    Failures are reported and skipped so a batch is not lost to a single bad folder.
    """
    if isinstance(paths, str):
        paths = sorted(p for p in glob.glob(os.path.join(paths, pattern)) if os.path.isdir(p))
    results = {}
    for p in paths:
        try:
            results[p] = correct_sensor_folder(p, sensor=sensor, outdir=outdir, **kwds)
        except Exception as e:  # pragma: no cover - depends on real data
            print("Failed to correct %s: %s" % (p, e))
            results[p] = e
    return results
