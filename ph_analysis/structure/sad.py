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
        ('sqrt_avg.', lambda x: np.sqrt(np.average(x))),
        ('s.d.', lambda x: np.std(x, ddof=0)),
        ('abs._sum', lambda x: np.sum(np.abs(x))),
        ('abs._avg.', lambda x: np.average(np.abs(x))),
        ('abs._s.d.', lambda x: np.std(np.abs(x), ddof=0)),
    ]


def create_data_stat(data, keys, properties):
    """

    :param data: pandas.DataFrame
    :param keys: List of strings
    :param properties: List of strings
    :return:
    """
    functions = create_statistical_functions()
    return data.groupby(keys, sort=False).agg(functions)[properties]


class SAD(object):
    def __init__(self, atoms, atoms_ideal):
        self._atoms = atoms
        self._atoms_ideal = atoms_ideal
        self._initialize_data()

    def _initialize_data(self):
        data = pd.DataFrame()
        data['symbol'] = self._atoms.get_chemical_symbols()
        data['atom'] = ''
        data = data.reset_index()  # data['index'] is created
        self._data = data

    def run(self):
        self.calculate_sad()
        self.write()

    def _calculate_origin(self):
        positions = self._atoms.get_scaled_positions()
        positions_ideal = self._atoms_ideal.get_scaled_positions()
        diff = positions - positions_ideal
        diff -= np.rint(diff)
        return np.average(diff, axis=0)

    def calculate_sad(self):
        atoms = self._atoms
        atoms_ideal =self._atoms_ideal

        origin = self._calculate_origin()

        positions = atoms.get_scaled_positions()
        positions_ideal = atoms_ideal.get_scaled_positions()
        diff = positions - (positions_ideal + origin)
        diff -= np.rint(diff)
        diff = np.dot(diff, atoms.get_cell())  # frac -> A
        sad = np.sum(diff ** 2, axis=1)

        self._data['sad'] = sad

    def write(self):
        filename = self._create_filename()
        data = self._data

        properties = ['sad']

        with open(filename, 'w') as f:
            f.write(self._create_header())
            f.write('{:<22s}{:<18s}'.format('#', 'SAD_(A^2)'))
            f.write('\n')
            for i, x in data.iterrows():
                f.write('atom ')
                f.write('{:11d}'.format(x['index']))
                f.write(' {:5s}'.format(x['symbol']))
                f.write('{:18.12f}'.format(x['sad']))
                f.write('\n')

            f.write('\n')

            # Write statistics for all atoms
            data_stat = create_data_stat(data, 'atom', properties)
            for k0, x in data_stat.iterrows():
                for k1, v in x.iteritems():
                    f.write('{:16}'.format(k1[1]))
                    f.write(' {:5s}'.format(k0))
                    f.write('{:18.12f}'.format(v))
                    f.write('\n')
                f.write('\n')

            # Write statistics for each symbol
            data_stat = create_data_stat(data, 'symbol', properties)
            for k0, x in data_stat.iterrows():
                for k1, v in x.iteritems():
                    f.write('{:16s}'.format(k1[1]))
                    f.write(' {:5s}'.format(k0))
                    f.write('{:18.12f}'.format(v))
                    f.write('\n')
                f.write('\n')

    def _create_header(self):
        return ''

    def _create_filename(self):
        return 'sad.dat'

    def get_data(self):
        return self._data
