#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import shutil
import time
import subprocess
import numpy as np
from .phonopy_conf_creator import PhonopyConfCreator
from vasp.poscar import Poscar
from autotools import symlink_force


class PhononCalculator(object):
    def __init__(self,
                 directory_data="./",
                 poscar_filename="POSCAR",
                 poscar_average_filename=None,
                 is_average_mass=False,
                 dim_sqs=None,
                 is_primitive=False,
                 is_band=True,
                 is_partial_dos=False,
                 is_tetrahedron=False,
                 is_tprop=False,
                 mesh=None):

        if dim_sqs is None:
            dim_sqs = np.array([1, 1, 1])

        if mesh is None:
            mesh = np.array([1, 1, 1])

        self._variables = None

        self._home = os.path.expanduser("~")
        self._phonopy = subprocess.check_output(["which", "phonopy"]).strip()
        print("phonopy_path:", self._phonopy)

        self._directory_data = directory_data
        self._poscar_filename = poscar_filename
        self._poscar_average_filename = poscar_average_filename

        self._is_average_mass = is_average_mass

        self.set_dim_sqs(dim_sqs)
        self._is_band = is_band
        self.set_is_tetrahedron(is_tetrahedron)
        self.set_is_partial_dos(is_partial_dos)
        self.set_is_tprop(is_tprop)
        self._is_primitive = is_primitive

        self._mesh = np.array(mesh)

    def set_dim_sqs(self, dim_sqs):
        self._dim_sqs = dim_sqs

    def set_is_tetrahedron(self, is_tetrahedron):
        self._is_tetrahedron = is_tetrahedron

    def set_is_partial_dos(self, is_partial_dos):
        self._is_partial_dos = is_partial_dos

    def set_is_tprop(self, is_tprop):
        self._is_tprop = is_tprop

    def set_mesh(self, mesh):
        self._mesh = mesh

    def set_variables(self, variables):
        self._variables = variables

    def run(self):
        self.copy_files()
        self.create_phonopy_conf()
        conf_files = self.gather_conf_files()
        for conf_file in conf_files:
            self.run_phonopy(conf_file)

    def copy_files(self):
        directory_data = self._directory_data
        symlink_force(directory_data + 'writefc.conf', 'writefc.conf')
        symlink_force(directory_data + 'POSCAR', 'POSCAR')
        symlink_force(directory_data + 'POSCAR_ideal', 'POSCAR_ideal')
        symlink_force(directory_data + 'FORCE_CONSTANTS', 'FORCE_CONSTANTS')

    def create_phonopy_conf(self):

        home = self._home
        directory_data = self._directory_data
        dim_sqs = self._dim_sqs
        variables = self._variables

        mesh = self._mesh.copy()

        print("directory_data:", directory_data)
        print("mesh:", mesh)

        spg_number = self.create_spg_number()

        # Get band path for the specific space group
        phonopy_conf_creator = PhonopyConfCreator(
            spg_number,
            mesh=mesh,
            tmax=3000,
            dim_sqs=dim_sqs,
            is_average_mass=self._is_average_mass,
            is_primitive=self._is_primitive,
            band_points=101,
            poscar_name="POSCAR",  # For getting the chemical symbols
            magmom_line=None,
            variables=variables,
        )
        phonopy_conf_creator.run()

    def create_spg_number(self):
        '''
        
        spg_number is used to determine the primitive axis and band paths.
        '''
        if self._poscar_average_filename is not None:
            poscar_filename = self._poscar_average_filename
        else:
            poscar_filename = self._poscar_filename
        print('SPG number is searched from {}'.format(poscar_filename))

        spg_number = Poscar(poscar_filename).get_symmetry_dataset()["number"]
        print("spg_number:", spg_number)
        return spg_number

    def gather_conf_files(self):
        conf_files = [
            "dos_smearing.conf",
        ]
        if self._is_band:
            conf_files.append("band.conf")
        if self._is_tetrahedron:
            conf_files.append("dos_tetrahedron.conf")
        if self._is_partial_dos:
            conf_files.append("partial_dos_smearing.conf")
        if self._is_tetrahedron and self._is_partial_dos:
            conf_files.append("partial_dos_tetrahedron.conf")
        if self._is_tprop:
            conf_files.append("tprop.conf")
        return conf_files

    def run_phonopy(self, conf_file):
        root = os.getcwd()
        home = self._home
        phonopy = self._phonopy
        print("=" * 80)
        print(conf_file)
        print("=" * 80)
        dir_name = conf_file.replace(".conf", "_calc")
        log_file = conf_file.replace(".conf", ".log")
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)
        os.chdir(dir_name)
        os.symlink("../" + conf_file, conf_file)
        os.symlink("../" + "POSCAR", "POSCAR")
        os.symlink("../" + "FORCE_CONSTANTS", "FORCE_CONSTANTS")
        if os.path.exists(log_file):
            os.remove(log_file)
        time1 = time.time()
        with open(log_file, "w") as f:
            subprocess.call(
                [phonopy, conf_file, "-v"],
                stdout=f,
            )
        time2 = time.time()
        dtime = time2 - time1
        print("Time for calc.: {:12.6f} s".format(dtime))

        if conf_file == "tprop.conf":
            subprocess.call(
                ["python", home + "/script/python/phonopy_tprop_arranger.py"]
            )

        os.chdir(root)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datadir",
                        default="..",
                        type=str,
                        help="Data directory")
    parser.add_argument("--tetrahedron",
                        action="store_true",
                        help="Calculate using tetrahedron method.")
    parser.add_argument("--partial_dos",
                        action="store_true",
                        help="Calculate partial DOS.")
    parser.add_argument("--tprop",
                        action="store_true",
                        help="Calculate thermal properties.")
    args = parser.parse_args()

    phonon_analyzer = PhononCalculator(
        directory_data=args.datadir,
        is_tetrahedron=args.tetrahedron,
        is_partial_dos=args.partial_dos,
        is_tprop=args.tprop,
    )
    phonon_analyzer.run()

if __name__ == "__main__":
    main()
