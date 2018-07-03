#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import pandas as pd

__author__ = 'Yuji Ikeda'
__version__ = '0.1.0'


def create_statistical_functions():
    return [
        ('sum', np.sum),
        ('avg.', np.average),
        ('s.d.', lambda x: np.std(x, ddof=0)),
        ('abs._sum', lambda x: np.sum(np.abs(x))),
        ('abs._avg.', lambda x: np.average(np.abs(x))),
        ('abs._s.d.', lambda x: np.std(np.abs(x), ddof=0)),
    ]


def create_data_stat(data, keys, properties, sort=False):
    """

    :param data: pandas.DataFrame
    :param keys: List of strings
    :param properties: List of strings
    :return:
    """
    functions = create_statistical_functions()
    return data.groupby(keys, sort=sort).agg(functions)[properties]


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
        data = data.reset_index()  # data['index'] is created
        self._data = data

    def run(self):
        self.generate_atomic_volume()
        self.write()

    def _get_distances1(self, rpos):
        cell = self._atoms.get_cell()
        rpos = np.dot(rpos, cell)
        distances = (rpos[:, 0] * rpos[:, 0] +
                     rpos[:, 1] * rpos[:, 1] +
                     rpos[:, 2] * rpos[:, 2])
        return np.sqrt(distances)

    def generate_atomic_volume(self, prec=1e-6):
        raise NotImplementedError

    def write(self):
        filename = self._create_filename()
        with open(filename, "w") as f:
            self._data.to_string(
                f,
                columns=['symbol', 'volume'],
                formatters=['{:6s}'.format, '{:18.12f}'.format],
                index=False,
            )

    def _create_header(self):
        raise NotImplementedError

    def _create_filename(self):
        raise NotImplementedError

    def get_data(self):
        return self._data
