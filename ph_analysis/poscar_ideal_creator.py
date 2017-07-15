#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from vasp.poscar import Poscar

__author__ = 'Yuji Ikeda'


class PoscarIdealCreator(object):
    @staticmethod
    def run():
        """

        For disordered systems with relaxed atomic positions,
        it might be better to use initial atomic positions to use the
        symmetry of the structure (POSCAR_initial).
        """
        from phonopy.structure.atoms import symbol_map
        symbol_map['X'] = 0
        poscar = Poscar('POSCAR')
        number_of_atoms = poscar.get_atoms().get_number_of_atoms()
        dummy_symbols = ['X'] * number_of_atoms
        poscar.get_atoms().set_chemical_symbols(dummy_symbols)
        poscar.write_poscar('POSCAR_ideal')
