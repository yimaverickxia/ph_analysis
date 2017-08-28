#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import pandas as pd

__author__ = 'Yuji Ikeda'
__version__ = '0.1.0'


class Volume(object):
    def __init__(self, atoms):
        self._atoms = atoms
        self._volumes_atom = None
        self._volumes_symbol = None
        self._initialize_data()

    def _initialize_data(self):
        data = pd.DataFrame()
        data['symbol'] = self._atoms.get_chemical_symbols()
        data['atom'] = ''
        self._data = data

    def run(self):
        self.generate_atomic_volume()
        self.write_atomic_volume()

    def _get_distances1(self, rpos):
        cell = self._atoms.get_cell()
        rpos = np.dot(rpos, cell)
        distances = (rpos[:, 0] * rpos[:, 0] +
                     rpos[:, 1] * rpos[:, 1] +
                     rpos[:, 2] * rpos[:, 2])
        return np.sqrt(distances)

    def generate_atomic_volume(self, prec=1e-6):
        raise NotImplementedError

    def write_atomic_volume(self):
        filename = self._create_filename()
        data = self._data

        functions = [
            ('sum', np.sum),
            ('avg.', np.average),
            ('s.d.', lambda x: np.std(x, ddof=0)),
            ('abs._sum', lambda x: np.sum(np.abs(x))),
            ('abs._avg.', lambda x: np.average(np.abs(x))),
            ('abs._s.d.', lambda x: np.std(np.abs(x), ddof=0)),
        ]

        with open(filename, "w") as f:
            f.write(self._create_header())
            f.write('{:<22s}{:<18s}'.format('#', 'Voronoi_volume'))
            f.write('\n')
            for i, x in data.iterrows():
                f.write('atom ')
                f.write('{:11d}'.format(i))
                f.write(' {:5s}'.format(x['symbol']))
                f.write('{:18.12f}'.format(x['volume']))
                f.write('\n')

            f.write('\n')

            # Write statistics for all atoms
            data_stat = data.groupby('atom', sort=False).agg(functions)
            for k0, x in data_stat.iterrows():
                for k1, v in x.iteritems():
                    f.write('{:16}'.format(k1[1]))
                    f.write(' {:5s}'.format(k0))
                    f.write('{:18.12f}'.format(v))
                    f.write('\n')
                f.write('\n')

            # Write statistics for each symbol
            data_stat = data.groupby('symbol', sort=False).agg(functions)
            for k0, x in data_stat.iterrows():
                for k1, v in x.iteritems():
                    f.write('{:16s}'.format(k1[1]))
                    f.write(' {:5s}'.format(k0))
                    f.write('{:18.12f}'.format(v))
                    f.write('\n')
                f.write('\n')

    def _create_header(self):
        raise NotImplementedError

    def _create_filename(self):
        raise NotImplementedError

    def get_data(self):
        return self._data
