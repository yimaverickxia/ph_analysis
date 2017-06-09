#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import argparse
import numpy as np
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS
from vasp.poscar import Poscar
from ph_analysis.analysis.fc_analyzer_base import FCAnalyzerBase

__author__ = 'Yuji Ikeda'


class FCReducer(object):
    def __init__(self,
                 poscar_filename="POSCAR",
                 fc_filename="FORCE_CONSTANTS"):
        self._poscar = Poscar(poscar_filename)
        self._force_constants = parse_FORCE_CONSTANTS(fc_filename)
        self._fc_reduced = None

    def find_indices_from_positions(self, positions):
        """Find atomic indices for removed FCs"""

        poscar = self._poscar

        indices_all = []
        for position_combination in positions:
            indices_combination = []
            for p in position_combination:
                index = poscar.get_index_from_position(p)
                indices_combination.append(index)
            indices_all.append(indices_combination)
        indices_all = np.array(indices_all)

        return indices_all

    def write(self, filename):
        write_FORCE_CONSTANTS(self._fc_reduced, filename=filename)

    def do_postprocess(self):
        poscar = self._poscar

        fc_analyzer = FCAnalyzerBase(
            atoms=poscar.get_atoms(),
            force_constants=self._fc_reduced,
            is_symmetrized=False,
        )
        fc_analyzer.force_translational_invariance()
        self._fc_reduced = fc_analyzer.get_force_constants()

    def keep_FCs_for_positions(self, positions):
        """

        Parameters
        ----------
        positions: n x 2 x 3 array
            positions = np.array([
                [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            ])
        """
        poscar = self._poscar
        force_constants_orig = self._force_constants
        force_constants = np.zeros_like(force_constants_orig)

        positions = np.array(positions)

        indices_all = self.find_indices_from_positions(positions)
        print('indices_all:', indices_all)

        mappings = poscar.get_mappings_for_symops()
        for mapping in mappings:
            for indices_combination in indices_all:
                indices_removed = mapping[indices_combination]
                i0 = indices_removed[0]
                i1 = indices_removed[1]
                force_constants[i0, i1] = force_constants_orig[i0, i1]
                force_constants[i1, i0] = force_constants_orig[i1, i0]

        self._fc_reduced = force_constants
        self.do_postprocess()

    def remove_FCs_for_positions(self, positions):
        """

        Parameters
        ----------
        positions: n x 2 x 3 array
            positions = np.array([
                [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            ])
        """
        poscar = self._poscar
        force_constants = np.copy(self._force_constants)

        positions = np.array(positions)

        indices_all = self.find_indices_from_positions(positions)
        print('indices_all:', indices_all)
    
        mappings = poscar.get_mappings_for_symops()
        for mapping in mappings:
            for indices_combination in indices_all:
                indices_removed = mapping[indices_combination]
                i0 = indices_removed[0]
                i1 = indices_removed[1]
                force_constants[i0, i1] = 0.0
                force_constants[i1, i0] = 0.0

        self._fc_reduced = force_constants
        self.do_postprocess()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--poscar',
                        default="POSCAR",
                        type=str,
                        help="Filename of POSCAR.")
    parser.add_argument("--fc",
                        default="FORCE_CONSTANTS",
                        type=str,
                        help="Filename of FORCE_CONSTANTS.")
    args = parser.parse_args()
    FCReducer(
        poscar_filename=args.poscar,
        fc_filename=args.fc,
    )

if __name__ == "__main__":
    main()
