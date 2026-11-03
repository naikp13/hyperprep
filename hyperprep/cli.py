"""Command line interface: ``hyperprep run|steps|info``."""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(prog='hyperprep', description='Hyperspectral pre-processing toolkit')
    sub = p.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('run', help='run a YAML/JSON pipeline config')
    r.add_argument('config')
    r.add_argument('-i', '--input', help='override the input path')
    r.add_argument('-o', '--output', help='override the output path')
    r.add_argument('-q', '--quiet', action='store_true')

    sub.add_parser('steps', help='list available pipeline steps')

    i = sub.add_parser('info', help='summarise an image or product')
    i.add_argument('path')

    a = p.parse_args(argv)
    if a.cmd == 'run':
        from .pipeline import run_config
        run_config(a.config, verbose=not a.quiet, input=a.input, output=a.output)
    elif a.cmd == 'steps':
        from .pipeline import list_steps
        for k, v in list_steps().items():
            print('%-26s %s' % (k, v))
    elif a.cmd == 'info':
        from .io import load
        img = load(a.path)
        print(img)
        print('  wavelengths: %.1f - %.1f nm (%d bands)' % (np.nanmin(img.wavelengths), np.nanmax(img.wavelengths), img.bands))
        print('  value range: %.4g - %.4g' % (np.nanmin(img.data), np.nanmax(img.data)))
        print('  geotransform: %s' % (img.transform,))
        for k in ('sensor', 'units', 'sun_zenith', 'sun_azimuth', 'acquisition_time'):
            if k in img.meta:
                print('  %s: %s' % (k, img.meta[k]))
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
