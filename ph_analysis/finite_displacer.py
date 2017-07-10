#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import os
import shutil
import subprocess
import numpy as np
from vasp.incar import Incar
from vasp.chgcar import Chgcar
from vasp.kpoints import Kpoints
from ph_analysis.conf_creation import ConfCreation


__author__ = "Yuji Ikeda"


class FiniteDisplacer(object):
    def __init__(self, directory_data, dim, distance, thrown_file):
        self._directory_data = directory_data
        self._dim = dim
        self._distance = distance
        self._thrown_file = thrown_file
        self._ispin = None
        self._magmom = None

    def run(self):
        dim = self._dim
        distance = self._distance

        root = os.getcwd()

        dirname = '_'.join([str(x) for x in dim]) + '/' + str(distance)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        os.chdir(dirname)

        self.copy_files()
        self.extract_ispin()
        if self._ispin == '2':
            self.analyze_chgcar()
        self.modify_files()
        self.create_poscars()
        self.create_directories()

        os.chdir(root)

    def copy_files(self):
        directory_data = self._directory_data
        thrown_file = self._thrown_file

        shutil.copy2(directory_data + '/POSCAR', 'POSCAR_initial')
        shutil.copy2(directory_data + '/CONTCAR', 'POSCAR')
        shutil.copy2(directory_data + '/KPOINTS', 'KPOINTS')
        shutil.copy2(directory_data + '/POTCAR', 'POTCAR')
        shutil.copy2(directory_data + '/INCAR', 'INCAR')
        shutil.copy2(directory_data + '/CHGCAR', 'CHGCAR')
        shutil.copy2(directory_data + '/' + thrown_file, thrown_file)

    def analyze_chgcar(self):
        """Analyze CHGCAR

        magmom is used in:
            1. INCAR: To give initial guess for dim != [1, 1, 1]
            2. disp.conf and write_fc.conf: To find symmetry operations
               with the consideration of magnetic moments
        """
        chgcar = Chgcar('CHGCAR')
        chgcar.generate_atomic_charge()
        chgcar.write_atomic_charge()
        scaling = 1.5
        magmom = chgcar.get_atomic_charge()[1] * scaling
        self._magmom = magmom

    def extract_ispin(self):
        incar = Incar('INCAR')
        ispin = incar.get_dictionary().get('ISPIN', '1')
        self._ispin = ispin

    def modify_files(self):
        dim = self._dim
        magmom = self._magmom

        # INCAR
        incar = Incar('INCAR')

        if dim == [1, 1, 1]:
            icharg = '1'
        else:
            icharg = None
            incar.generate_supercell(dim)  # MAGMOM

        if magmom is not None:
            n = np.product(dim)
            magmom_supercell = [x for x in magmom for _ in range(n)]
            magmom_str = ' '.join(
                ['{:.4f}'.format(x) for x in magmom_supercell]
            )
        else:
            magmom_str = None

        incar_overwritten = {
            'EDIFF': '1.0E-8',
            'NELM': '500',
            'NSW': '1',
            'ISIF': '2',
            'LWAVE': '.FALSE.',
            'LCHARG': '.FALSE.',
            'ICHARG': icharg,
            'MAGMOM': magmom_str,
        }
        incar.update_dictionary(incar_overwritten)
        incar.write('INCAR')

        # KPOINTS
        kpoints = Kpoints('KPOINTS')
        kpoints.generate_supercell(dim)
        kpoints.write('KPOINTS')

    def create_poscars(self):
        dim = self._dim
        distance = self._distance
        magmom = self._magmom

        conf_creation = ConfCreation(
            dim=dim,
            distance=distance,
            magmom=magmom
        )
        conf_creation.run()

        subprocess.call(
            'phonopy -v disp.conf > phonopy_disp.log',
            shell=True,
        )

    def create_directories(self):
        thrown_file = self._thrown_file

        root = os.getcwd()
        dir_list = sorted(os.listdir('.'))
        poscar_list = [p for p in dir_list if 'POSCAR-' in p]
        for poscar in poscar_list:
            disp_dir = poscar.replace('POSCAR-', 'disp')
            if os.path.exists(disp_dir):
                shutil.rmtree(disp_dir)
            os.mkdir(disp_dir)
            os.chdir(disp_dir)
            shutil.copy2('../' + poscar, 'POSCAR')
            shutil.copy2('../INCAR'         , '.')
            shutil.copy2('../POTCAR'        , '.')
            shutil.copy2('../KPOINTS'       , '.')
            shutil.copy2('../' + thrown_file, '.')
            if os.path.exists('../CHGCAR'):
                os.symlink('../CHGCAR', 'CHGCAR')
            os.chdir(root)
