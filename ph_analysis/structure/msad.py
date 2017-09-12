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


class MSAD(object):
    def __init__(self, atoms, atoms_ideal):
        self._atoms = atoms
        self._atoms_ideal = atoms_ideal

    def run(self):
        self.calculate_msad()
        self.generate_values_for_symbols()
        self.write()

    def _calculate_origin(self):
        positions = self._atoms.get_scaled_positions()
        positions_ideal = self._atoms_ideal.get_scaled_positions()
        diff = positions - positions_ideal
        return np.average(diff, axis=0)

    def calculate_msad(self):
        atoms = self._atoms
        atoms_ideal =self._atoms_ideal

        origin = self._calculate_origin()

        positions = atoms.get_scaled_positions()
        positions_ideal = atoms_ideal.get_scaled_positions()
        diff = positions - (positions_ideal + origin)
        diff = np.dot(diff, atoms.get_cell())  # frac -> A
        msad = np.sum(diff ** 2, axis=1)

        self._msad = msad

        self.generate_values_for_symbols()

    def generate_values_for_symbols(self):
        symbols = self._atoms.get_chemical_symbols()
        msad = self._msad

        msad_symbol = group_into_symbols(symbols, msad)

        self._msad_symbol = msad_symbol

    def write(self):
        symbols = self._atoms.get_chemical_symbols()
        msad = self._msad

        filename = 'msad.dat'
        with open(filename, 'w') as f:
            f.write("#" + " " * 21)
            f.write("{:18s}".format('SAD_(A^2)'))
            f.write('\n')
            for i, s in enumerate(symbols):
                f.write("atom")
                f.write(" {:11d} {:5s}".format(i, s))
                f.write("{:18.12f}".format(msad[i]))
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

                f.write("{:16s} {:5s}".format("sqrt(average)", symbol))
                for v in values:
                    f.write("%18.12f" % (np.sqrt(np.average(v))))
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
                msad,
            ]
            write_statistics(properties)
            for s in sorted(set(symbols), key=symbols.index):
                properties = [
                    self._msad_symbol[s],
                ]
                write_statistics(properties, s)


