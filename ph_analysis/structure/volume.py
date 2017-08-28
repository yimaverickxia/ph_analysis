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
        symbols = self._atoms.get_chemical_symbols()
        volumes_atom = self._volumes_atom
        volumes_symbol = self._volumes_symbol
        filename = self._create_filename()

        with open(filename, "w") as f:
            f.write(self._create_header())
            f.write("#" + " " * 21)
            f.write("{:18s}".format("Voronoi_volume"))
            f.write("\n")
            for i, s in enumerate(symbols):
                f.write("atom")
                f.write(" {:11d} {:5s}".format(i, s))
                f.write("{:18.12f}".format(self._volumes_atom[i]))
                f.write("\n")
            f.write("\n")

            def write_statistics(values, symbol=""):
                f.write("{:16s} {:5s}".format("sum", symbol))
                for v in values:
                    f.write("%18.12f" % (np.sum(v)))
                f.write("\n")

                f.write("{:16s} {:5s}".format("average", symbol))
                for v in values:
                    f.write("%18.12f" % (np.average(v)))
                f.write("\n")

                f.write("{:16s} {:5s}".format("s.d.", symbol))
                for v in values:
                    f.write("%18.12f" % (np.std(v)))
                f.write("\n")

                f.write("{:16s} {:5s}".format("absolute_sum", symbol))
                for v in values:
                    f.write("%18.12f" % (np.sum(abs(v))))
                f.write("\n")

                f.write("{:16s} {:5s}".format("absolute_average", symbol))
                for v in values:
                    f.write("%18.12f" % (np.average(abs(v))))
                f.write("\n")

                f.write("{:16s} {:5s}".format("absolute_s.d.", symbol))
                for v in values:
                    f.write("%18.12f" % (np.std(abs(v))))
                f.write("\n")

                f.write("\n")

            properties = [
                volumes_atom,
            ]
            write_statistics(properties)
            for s in sorted(set(symbols), key=symbols.index):
                properties = [
                    volumes_symbol[s],
                ]
                write_statistics(properties, s)

    def _create_header(self):
        raise NotImplementedError

    def _create_filename(self):
        raise NotImplementedError

    def get_volumes_atom(self):
        return self._volumes_atom
