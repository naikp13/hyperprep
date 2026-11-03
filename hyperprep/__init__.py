"""
hyperprep - one-stop pre-processing for terrestrial, drone and satellite hyperspectral imagery.

Sub-packages:

* :mod:`hyperprep.calibration` - DN -> radiance -> reflectance, white/dark references, panels, ELC
* :mod:`hyperprep.spectral` / :mod:`hyperprep.noise` / :mod:`hyperprep.masks` - shared spectral,
  noise and masking steps
* :mod:`hyperprep.topo` / :mod:`hyperprep.solar` - illumination geometry and topographic correction
* :mod:`hyperprep.terrestrial` - sensor correction, projection mapping, illumination correction,
  hyperclouds (hylite based)
* :mod:`hyperprep.drone` - GNSS/IMU tracks, direct georeferencing, BRDF and flight-line corrections
* :mod:`hyperprep.satellite` - EnMAP / PRISMA / EMIT / DESIS readers, TOA reflectance, DOS, smile,
  detector overlap
* :mod:`hyperprep.pipeline` - YAML-driven pipelines (``hyperprep run config.yaml``)
"""
from .image import HSImage
from .io import load, read_envi, save, write_envi

__version__ = '0.1.0'
__all__ = ['HSImage', 'load', 'save', 'read_envi', 'write_envi', '__version__']
