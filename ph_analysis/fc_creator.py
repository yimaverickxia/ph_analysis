#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import shutil
import subprocess
from vasp.poscar import Poscar
from autotools import symlink_force


class FCCreator(object):
    def __init__(
            self,
            directory_data="./",
            poscar_filename="POSCAR",
            positions_1nn=None):
        self._phonopy = subprocess.check_output(["which", "phonopy"]).strip()
        self._directory_data = directory_data
        self._poscar_filename = poscar_filename
        self._positions_1nn = positions_1nn

        print("phonopy_path:", self._phonopy)
        print("directory_data:", self._directory_data)
        print("poscar_filename:", self._poscar_filename)

    def run(self):
        self.create_poscar_ideal()
        self.create_force_constants()
        # self.reduce_force_constants()

    def create_poscar_ideal(self):
        """

        For disordered systems with relaxed atomic positions,
        it might be better to use initial atomic positions to use the
        symmetry of the structure (POSCAR_initial).
        """
        directory_data = self._directory_data
        symlink_force(directory_data + self._poscar_filename, "POSCAR")
        poscar = Poscar("POSCAR")
        number_of_atoms = poscar.get_atoms().get_number_of_atoms()
        dummy_symbols = ["X"] * number_of_atoms
        poscar.get_atoms().set_chemical_symbols(dummy_symbols)
        poscar.write_poscar("POSCAR_ideal")

    def create_force_constants(self):
        directory_data = self._directory_data
        phonopy = self._phonopy
        symlink_force(directory_data + "/FORCE_SETS", "FORCE_SETS")
        symlink_force(directory_data + "/writefc.conf", "writefc.conf")
        # Create FORCE_CONSTANTS
        with open('phonopy_writefc_conf.log', 'w') as f:
            subprocess.call([phonopy, 'writefc.conf', '-v'], stdout=f)

    def reduce_force_constants(self):
        from fc_reducer.fc_reducer import FCReducer
        positions_1nn = self._positions_1nn
        fc_reducer = FCReducer(
            poscar_filename="POSCAR_ideal",
            fc_filename="FORCE_CONSTANTS",
        )
        fc_reducer.keep_FCs_for_positions(positions_1nn)
        shutil.move("FORCE_CONSTANTS", "FORCE_CONSTANTS_unreduced")
        shutil.move("FORCE_CONSTANTS_reduced_TI", "FORCE_CONSTANTS")
