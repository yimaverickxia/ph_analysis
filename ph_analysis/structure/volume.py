#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np

__author__ = 'Yuji Ikeda'
__version__ = '0.1.0'


def group_into_symbols(symbols, values_atom):
    values_symbol = {}
    for s in sorted(set(symbols), key=symbols.index):
        indices = [ia for ia in range(len(symbols)) if symbols[ia] == s]
        values_symbol[s] = values_atom[indices]
    return values_symbol


class Volume(object):
    def __init__(self, atoms):
        self._atoms = atoms
        self._volumes_atom = None
        self._volumes_symbol = None

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

    def generate_values_for_symbols(self):
        symbols = self._atoms.get_chemical_symbols()
        volumes_atom = self._volumes_atom

        volumes_symbol = group_into_symbols(symbols, volumes_atom)

        self._volumes_symbol = volumes_symbol

    def write_atomic_volume(self):
        raise NotImplementedError

    def _create_filename(self):
        raise NotImplementedError
