#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import shutil
import subprocess

__author__ = 'Yuji Ikeda'


class FCCreator(object):
    def __init__(
            self,
            directory_data="./",
            poscar_filename="POSCAR",
            positions_1nn=None):
        self._directory_data = directory_data
        self._poscar_filename = poscar_filename
        self._positions_1nn = positions_1nn

        print("directory_data:", self._directory_data)
        print("poscar_filename:", self._poscar_filename)

    def run(self):
        self.copy_files()
        self.create_poscar_ideal()
        self.create_force_constants()
        # self.reduce_force_constants()

    def copy_files(self):
        directory_data = self._directory_data
        shutil.copy2(directory_data + '/POSCAR', '.')
        shutil.copy2(directory_data + '/FORCE_SETS', '.')
        shutil.copy2(directory_data + '/writefc.conf', '.')

    def create_poscar_ideal(self):
        from .poscar_ideal_creator import PoscarIdealCreator
        PoscarIdealCreator().run()

    def create_force_constants(self):
        with open('writefc_conf.log', 'w') as f:
            subprocess.call(['phonopy', 'writefc.conf', '-v'], stdout=f)

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
