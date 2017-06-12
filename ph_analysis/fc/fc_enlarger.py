#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import sys
from fractions import Fraction

import numpy as np
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS
from phonopy.harmonic.dynamical_matrix import get_smallest_vectors
from phonopy.harmonic.force_constants import set_permutation_symmetry
from phonopy.interface.vasp import read_vasp
from phonopy.structure.cells import Supercell, Primitive
from vasp.poscar import Poscar
from .fc_symmetrizer_spg import FCSymmetrizerSPG
from ..structure.configuration_randomizer import ConfigurationRandomizer

__author__ = 'Yuji Ikeda'


class FCEnlarger(object):
    def __init__(self, dict_input=None):
        dict_input = self._read_input(dict_input)
        self._configure(dict_input)
        self.print_info()

    def _read_input(self, tmp):
        dict_input = get_default_input()

        if tmp is not None:
            dict_input.update(tmp)

        return dict_input

    def _configure(self, dict_input):
        """

        supercell_disordered: The structure corresponding to the force
            constants.
        """
        self._atoms_disordered = read_vasp(dict_input["structure_disordered"])
        self._atoms_average = read_vasp(dict_input["structure_average"])
        self._symprec = dict_input["symprec"]
        self._map_s2s = dict_input["map_s2s"]

        self._random_seed = dict_input["random_seed"]
        self._num_configurations = dict_input['num_configurations']

        primitive_matrix = parse_3x3_matrix(dict_input["primitive_matrix"])
        self._supercell_matrix = dict_input["supercell_matrix"]
        self._enlargement_matrix = dict_input["enlargement_matrix"]

        self._fc_filename = dict_input["force_constants"]
        self._force_constants = parse_FORCE_CONSTANTS(self._fc_filename)

        self._supercell_disordered = Supercell(
            self._atoms_disordered,
            self._supercell_matrix,
            symprec=self._symprec)

        self._supercell_average = Supercell(
            self._atoms_average,
            self._supercell_matrix,
            symprec=self._symprec)

        # TODO(ikeda): The following part should be separated from this method.
        inv_supercell_matrix = np.linalg.inv(self._supercell_matrix)
        trans_mat = np.dot(inv_supercell_matrix, primitive_matrix)
        self._primitive_average = Primitive(
            self._supercell_average, trans_mat, symprec=self._symprec)

    def print_info(self):
        print("fc_filename:", self._fc_filename)

    def run(self):
        self.check_atoms_correspondence()
        self._generate_force_constants_pair()
        self._generate_force_constants_site()
        print("Creating an enlarged structure: ", end="")
        self._generate_enlarged_cell()
        print("Finished.")
        self._relative_positions_site = (
            _convert_relative_positions_for_enlarged_cell(
                self._relative_positions_site,
                self._enlargement_matrix))

        print("Creating an enlarged force constants: ", end="")
        self.generate_enlarged_force_constants()
        print("Finished.")

    def check_atoms_correspondence(self):
        """Check the correspondence of the atomic positions.

        Check whether the atomic positions of the disordered structure are
        equal to those of the average structure or not.
        """
        symprec = self._symprec
        positions_disordered = self._atoms_disordered.get_scaled_positions()
        positions_average = self._atoms_average.get_scaled_positions()
        if (np.abs(positions_disordered - positions_average) > symprec).any():
            print("ERROR: The atomic positions of the disordered structure "
                  "must be equal to those of the average structure.")
            raise ValueError

    def _generate_force_constants_pair(self):
        supercell_disordered = self._supercell_disordered

        force_constants_analyzer = FCSymmetrizerSPG(
            self._force_constants,
            atoms=supercell_disordered,
            atoms_ideal=self._supercell_average,
            is_symmetrized=True)
        force_constants_analyzer.average_force_constants_spg()

        self._force_constants_pair = (
            force_constants_analyzer.get_force_constants_pair())

        force_constants_analyzer.write_force_constants_symmetrized()
        force_constants_analyzer.write_force_constants_pair()

    def _generate_force_constants_site(self):
        """

        natoms_supercell: The number of atoms in the supercell.
        natoms_primitive: The number of atoms in the primitive cell.
        """
        supercell = self._supercell_average
        primitive = self._primitive_average

        natoms_supercell = supercell.get_number_of_atoms()
        p2s_map = primitive.get_primitive_to_supercell_map()
        smallest_vectors, multiplicity = get_smallest_vectors(
            supercell, primitive, symprec=self._symprec)
        natoms_primitive = primitive.get_number_of_atoms()
        relative_positions_site = [[] for _ in range(natoms_primitive)]
        force_constants_site = [[] for _ in range(natoms_primitive)]
        for index, i in enumerate(p2s_map):
            for j in range(natoms_supercell):
                if i == j:
                    continue
                nsites = multiplicity[j][i]
                # TODO(ikeda):
                # "get_fc_tmp" should be renamed.
                # Finally, force_constants_pair format will change,
                # and this part should also change.
                # Instead, new variable fora the correspondence between
                # the symbols and the order should be given.
                # TODO(ikeda):
                # relative_positions should be a "numpy.array".
                # TODO(ikeda):
                # Check what happen if we exchange "i" and "j".
                # Maybe we should take the correspondence to the enlarged
                # force constants.
                fc_tmp = get_fc_tmp(i, j, self._force_constants_pair, nsites)
                for k in range(nsites):
                    relative_positions_site[index].append(
                        smallest_vectors[j][i][k])
                    force_constants_site[index].append(fc_tmp)

        self._relative_positions_site = relative_positions_site
        self._force_constants_site = force_constants_site

    def _generate_enlarged_cell(self):
        """

        If The tag "replacements" is not given, we just copy the original
        structure as the enlarged cell.
        """
        if self._map_s2s is not None:
            self._enlarged_cell_average = Supercell(
                self._primitive_average,
                self._enlargement_matrix,
                symprec=self._symprec)
            self._enlarged_cell = self._disorder_enlarged_cell()
        else:
            enlargement_matrix = np.eye(3, dtype=int)
            self._enlarged_cell_average = self._atoms_average
            self._enlarged_cell = Supercell(
                self._atoms_disordered,
                enlargement_matrix,
                symprec=self._symprec)
        return self

    def _disorder_enlarged_cell(self):
        configuration_randomizer = ConfigurationRandomizer(
            atoms=self._enlarged_cell_average,
            map_s2s=self._map_s2s,
            random_seed=self._random_seed)

        for i in range(self._num_configurations):
            tmp = configuration_randomizer.create_randomized_configuration()
        enlarged_cell = tmp

        return enlarged_cell

    def generate_enlarged_force_constants(self):
        """

        natoms: The number of atoms in the enlarged cell.
        """
        enlarged_cell = self._enlarged_cell
        scaled_positions = enlarged_cell.get_scaled_positions()
        symbols = enlarged_cell.get_chemical_symbols()
        s2u_map = enlarged_cell.get_supercell_to_unitcell_map()
        natoms = enlarged_cell.get_number_of_atoms()
        enlarged_force_constants = np.zeros((natoms, natoms, 3, 3))
        for i in range(natoms):
            i_equivalent = s2u_map[i]
            relative_positions = self._relative_positions_site[i_equivalent]
            fc_tmp1 = self._force_constants_site[i_equivalent]
            for relative_position, fc_tmp2 in zip(relative_positions, fc_tmp1):
                j = get_index_at_relative_position(
                    i,
                    scaled_positions,
                    relative_position,
                    symprec=self._symprec)
                s_i = symbols[i]
                s_j = symbols[j]
                enlarged_force_constants[i, j] += fc_tmp2[(s_i, s_j)]

        set_permutation_symmetry(enlarged_force_constants)
        set_translational_invariance_for_diagonal(enlarged_force_constants)
        self._fc_enlarged = enlarged_force_constants

    def write_cell_enlarged(self, filename):
        poscar = Poscar()
        poscar.set_atoms(self._enlarged_cell)
        poscar.write(filename)

    def write_cell_enlarged_ideal(self, filename):
        poscar = Poscar()
        poscar.set_atoms(self._enlarged_cell_average)
        poscar.write(filename)

    def write_fc_enlarged(self, filename):
        fc = self._fc_enlarged
        write_FORCE_CONSTANTS(fc, filename)


def get_default_input():
    """

    supercell_matrix : (3 x 3) array
        The supercell matrix which corresponds to the size of
        the force constant matrix. This matrix should be the same as that for
        the phonopy input.
    num_configurations : Integer
        Number of considered configurations. Currently only the last one is
        remained.
    """
    default_input = {
        "force_constants": "FORCE_CONSTANTS_orig",
        "structure_disordered": "POSCAR_orig",
        "structure_average": "POSCAR_orig_ideal",
        "map_s2s": None,
        "random_seed": None,
        'num_configurations': 1,
        "primitive_matrix": np.eye(3),
        "supercell_matrix": np.eye(3, dtype=int),
        "enlargement_matrix": np.eye(3, dtype=int),
        "symprec": 1.e-5,
    }
    return default_input


def parse_3x3_matrix(matrix):
    """Fractional values for primitive_matrix are parsed.
    """
    matrix_converted = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            matrix_converted[i, j] = float(Fraction(matrix[i][j]))
    return matrix_converted


def get_fc_tmp(i, j, force_constants_pair, nsites):
    """

    Parameters
    ----------
        nsites: Multiplicity of the site j (i) w.r.t. the site i (j).
    """
    fc_tmp = {}
    for (pair_type, force_constants) in force_constants_pair.items():
        fc_tmp[pair_type] = force_constants[i, j] / nsites
    return fc_tmp


def get_index_at_relative_position(i,
                                   scaled_positions,
                                   relative_position,
                                   symprec=1e-5):
    for j, scaled_position in enumerate(scaled_positions):
        diff = (relative_position + scaled_positions[i]) - scaled_position
        diff -= np.rint(diff)
        if np.all(np.abs(diff) < symprec):
            return j


def _convert_relative_positions_for_enlarged_cell(relative_positions_site,
                                                  enlargement_matrix):
    conversion_matrix = np.linalg.inv(enlargement_matrix)
    relative_positions_site_for_enlarged_cell = []
    for relative_positions in relative_positions_site:
        relative_positions_site_for_enlarged_cell.append(
            np.dot(relative_positions, conversion_matrix.T))
    return relative_positions_site_for_enlarged_cell


def set_translational_invariance_for_diagonal(force_constants):
    for i in range(force_constants.shape[0]):
        force_constants[i, i] = -1.0 * np.sum(force_constants[i, :], axis=0)


def main():
    import yaml
    dict_input = yaml.load(sys.argv[1])
    fc_enlarger = FCEnlarger(dict_input)
    fc_enlarger.run()


if __name__ == "__main__":
    main()
