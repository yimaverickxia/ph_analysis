#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TODO(ikeda): The function "get_rotations_cart" is not appropriate for this
#     module.
from __future__ import absolute_import, division, print_function

import numpy as np

from ph_analysis.fc.fc_analyzer_base import FCAnalyzerBase
from .mappings_modifier import MappingsModifier
from ..structure.structure_analyzer import StructureAnalyzer
from ..structure.symtools import get_rotations_cart


class FCDistributionAnalyzer(FCAnalyzerBase):
    def __init__(self, force_constants, atoms, atoms_ideal, supercell_matrix, symprec=1e-5):
        super(FCDistributionAnalyzer, self).__init__(
            force_constants=force_constants,
            atoms=atoms,
            atoms_ideal=atoms_ideal,
            supercell_matrix=supercell_matrix,
        )

        self.set_symprec(symprec)

        self._create_distance_matrix()
        self._create_symbol_numbers()
        self._create_rotations_cart()
        self._create_mappings()

    def set_symprec(self, symprec):
        self._symprec = symprec

    def _create_distance_matrix(self):
        sa = StructureAnalyzer(self._atoms)
        sa.generate_distance_matrix()
        self._distance_matrix = sa.get_distance_matrix()

    def _create_symbol_numbers(self):
        symbols = self._atoms.get_chemical_symbols()
        self._symbol_types, self._symbol_numbers = (
            SymbolNumbersGenerator().generate_symbol_numbers(symbols)
        )

    def _create_rotations_cart(self):
        self._rotations_cart = get_rotations_cart(self._atoms_ideal)

    def _create_mappings(self):
        # mappings: each index is for the "after" symmetry operations, and
        #     each element is for the "original" positions. 
        #     mappings[k][i] = j means the atom j moves to the positions of
        #     the atom i for the k-th symmetry operations.
        sa = StructureAnalyzer(self._atoms_ideal)
        mappings = sa.get_mappings_for_symops(prec=self._symprec)
        print("mappings: Finished.")
        mappings_inverse = MappingsModifier(mappings).invert_mappings()

        self._mappings = mappings
        self._mappings_inverse = mappings_inverse

    def analyze_fc_distribution(self,
                                a1,
                                a2,
                                filename="fc_values.dat"):

        symbols = self._atoms.get_chemical_symbols()
        rotations = self._rotations_cart
        mappings_inverse = self._mappings_inverse
        fcs = self._force_constants

        # a1, a2: indices of atomic positions where i1 and i2 come
        # i1, i2: indices of atomic positions before symmetry operations
        i0s = mappings_inverse[:, a1]
        i1s = mappings_inverse[:, a2]
        s0s = np.array([symbols[x] for x in i0s])
        s1s = np.array([symbols[x] for x in i1s])
        fc_symbols = np.array([[s0, s1] for s0, s1 in zip(s0s, s1s)])
        distances = np.fromiter(
            [self._distance_matrix[i0, i1] for i0, i1 in zip(i0s, i1s)], float
        )
        rotate = lambda m, r: np.dot(np.dot(r, m), r.T)
        fc_values = np.array(
            [rotate(fcs[i0, i1], r) for i0, i1, r in zip(i0s, i1s, rotations)]
        )

        self.write(fc_symbols, fc_values, distances, filename)

    def write(self, fc_symbols, fc_values, distances, filename):
        with open(filename, "w") as f:
            f.write('{:4s}'.format('e0'))
            f.write('{:4s}'.format('e1'))
            f.write('{:22s}'.format('distance'))
            f.write(' ' * 4)
            f.write('{:22s}'.format('xx'))
            f.write('{:22s}'.format('xy'))
            f.write('{:22s}'.format('xz'))
            f.write('{:22s}'.format('yx'))
            f.write('{:22s}'.format('yy'))
            f.write('{:22s}'.format('yz'))
            f.write('{:22s}'.format('zx'))
            f.write('{:22s}'.format('zy'))
            f.write('{:22s}'.format('zz'))
            f.write('\n')
            for si, d, v in zip(fc_symbols, distances, fc_values):
                f.write("{:4s}".format(si[0]))
                f.write("{:4s}".format(si[1]))
                f.write("{:22.15f}".format(d))
                f.write(" " * 4)
                for i in range(3):
                    for j in range(3):
                        f.write("{:22.15f}".format(v[i, j]))
                f.write("\n")


class SymbolNumbersGenerator(object):
    def generate_symbol_numbers(self, symbols):
        symbol_types = sorted(set(symbols), key=symbols.index)
        symbol_numbers = [symbol_types.index(s) for s in symbols]
        return symbol_types, symbol_numbers
