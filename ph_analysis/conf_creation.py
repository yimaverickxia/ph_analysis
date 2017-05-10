#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from vasp.incar import Incar
from vasp.outcar import Outcar


class ConfCreation(object):
    def __init__(self,
                 dim,
                 magmom,
                 incar=None,
                 outcar=None,
                 distance=None,
                 is_symmetry=True):
        self._dim = dim
        self._magmom = magmom
        self._incar = incar
        self._outcar = outcar
        self._distance = distance
        self._is_symmetry = is_symmetry

    def run(self):
        self.create_disp_conf()
        self.create_writefc_conf()

    def create_disp_conf(self):
        with open("disp.conf", "w") as f:
            f.write("CREATE_DISPLACEMENTS = .TRUE.\n")
            self.write_symmetry(f)
            self.write_dim(f)
            self.write_distance(f)
            self.write_magmom(f)

    def create_writefc_conf(self):
        with open("writefc.conf", "w") as f:
            f.write("FORCE_CONSTANTS = WRITE\n")
            self.write_symmetry(f)
            self.write_dim(f)
            self.write_distance(f)
            self.write_magmom(f)

    def write_magmom(self, f):
        if self._magmom is not None:
            magmom = self.read_magmom()
            f.write("MAGMOM =")
            f.write(" %.4f" * len(magmom) % tuple(magmom))
            f.write("\n")

    def read_magmom(self):
        if self._incar != None:
            print("MAGMOM is read from INCAR")
            INCAR_dictionary = Incar(self._incar).get_dictionary()
            magmom = INCAR_dictionary["MAGMOM"]
            magmom = [float(x) for x in magmom.split()]
        elif self._outcar != None:
            print("MAGMOM is read from OUTCAR")
            magmom = Outcar(self._outcar).get_dictionary()["MAGMOM"]
            magmom = [m * 1.5 for m in magmom]
        else:
            magmom = self._magmom
        return magmom

    def write_symmetry(self, f):
        if not self._is_symmetry:
            f.write("SYMMETRY = .FALSE.\n")

    def write_dim(self, f):
        f.write("DIM = %d %d %d\n" % tuple(self._dim))

    def write_distance(self, f):
        if self._distance is not None:
            f.write("DISPLACEMENT_DISTANCE = {:f}\n".format(self._distance))
