#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import pandas as pd
from .volume import create_data_stat
from .volume_factory import VolumeFactory

__author__ = 'Yuji Ikeda'
__version__ = '0.1.0'


class VolumeList(object):
    def __init__(self, atoms_list, method):
        self._atoms_list = atoms_list
        self._method = method
        self._factory = VolumeFactory()
        self._initialize_data()

    def _initialize_data(self):
        self._data = pd.DataFrame()

    def run(self, sort):
        self.generate_atomic_volume()
        self.write_atomic_volume(sort)

    def generate_atomic_volume(self):
        data_list = []
        for i, atoms in enumerate(self._atoms_list):
            # TODO(ikeda): additional arguments should be acceptable
            volume = self._factory.create(self._method, atoms)
            volume.generate_atomic_volume()
            data_tmp = volume.get_data()
            data_tmp['structure'] = i
            data_list.append(data_tmp)
        self._data = pd.concat(data_list, ignore_index=True)

    def write_atomic_volume(self, sort=False):
        filename = self._create_filename()
        data = self._data

        properties = ['volume']

        headers = ['', 'str.', 'index', 'symb.', 'volume']
        with open(filename, "w") as f:
            f.write(self._create_header())
            f.write('{:<10s} {:<5s} {:<5s} {:<5s} {:<18s}'.format(
                *headers))
            f.write('\n')
            for i, x in data.iterrows():
                f.write('atom      ')
                f.write(' {:5d}'.format(x['structure']))
                f.write(' {:5d}'.format(x['index']))
                f.write(' {:5s}'.format(x['symbol']))
                f.write(' {:18.12f}'.format(x['volume']))
                f.write('\n')

            f.write('\n')

            # Write statistics for all atoms
            data_stat = create_data_stat(data, 'atom', properties, sort)
            for k0, x in data_stat.iterrows():
                for k1, v in x.iteritems():
                    f.write('{:10s}'.format(k1[1]))
                    f.write(' {:5s}'.format(''))
                    f.write(' {:5s}'.format(''))
                    f.write(' {:5s}'.format(k0))
                    f.write(' {:18.12f}'.format(v))
                    f.write('\n')
                f.write('\n')

            # Write statistics for each symbol
            data_stat = create_data_stat(data, 'symbol', properties, sort)
            for k0, x in data_stat.iterrows():
                for k1, v in x.iteritems():
                    f.write('{:10s}'.format(k1[1]))
                    f.write(' {:5s}'.format(''))
                    f.write(' {:5s}'.format(''))
                    f.write(' {:5s}'.format(k0))
                    f.write(' {:18.12f}'.format(v))
                    f.write('\n')
                f.write('\n')

            # Write statistics for each structure
            keys = ['structure']
            data_stat = create_data_stat(data, keys, properties, sort)
            for k0, x in data_stat.iterrows():
                for k1, v in x.iteritems():
                    f.write('{:10s}'.format(k1[1]))
                    f.write(' {:5d}'.format(k0))
                    f.write(' {:5s}'.format(''))
                    f.write(' {:5s}'.format(''))
                    f.write(' {:18.12f}'.format(v))
                    f.write('\n')
                f.write('\n')

            # Write statistics for each symbol
            keys = ['structure', 'symbol']
            data_stat = create_data_stat(data, keys, properties, sort)
            for k0, x in data_stat.iterrows():
                for k1, v in x.iteritems():
                    f.write('{:10s}'.format(k1[1]))
                    f.write(' {:5d}'.format(k0[0]))
                    f.write(' {:5s}'.format(''))
                    f.write(' {:5s}'.format(k0[1]))
                    f.write(' {:18.12f}'.format(v))
                    f.write('\n')
                f.write('\n')

    def _create_header(self):
        return ''

    def _create_filename(self):
        return 'atomic_volume.dat'

    def get_data(self):
        return self._data
