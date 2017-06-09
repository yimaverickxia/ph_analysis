#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from vasp.poscar_list import PoscarList
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS
from phonopy.harmonic.force_constants import symmetrize_force_constants
from .fc_reorderer import FCReorderer

__author__ = 'Yuji Ikeda'


class FCAverager(object):
    def __init__(self, poscar_filenames, fc_filenames, weights):
        self._poscar_filenames = poscar_filenames
        self._fc_filenames = fc_filenames
        self._weights = weights

    def average_fc(self):
        indices_list = self._extract_indices_list()
        fc_list_tmp = self._extract_fc_list()
        fc_list = []
        for fc, indices in zip(fc_list_tmp, indices_list):
            fc_reordered = FCReorderer().reorder_fc(fc, indices)
            fc_list.append(fc_reordered)

        fc_average = np.average(fc_list, axis=0, weights=self._weights)

        self._write_fcs(fc_list=fc_list, fc_average=fc_average)

    def _extract_indices_list(self):
        poscar_list = PoscarList(self._poscar_filenames)
        indices_list = poscar_list.get_indices_of_positions()
        return indices_list

    def _extract_fc_list(self):
        fc_list = []
        for fc_filename in self._fc_filenames:
            force_constants = parse_FORCE_CONSTANTS(fc_filename)
            symmetrize_force_constants(force_constants)
            fc_list.append(force_constants)
        return fc_list

    @staticmethod
    def _write_fcs(fc_list, fc_average):
        for i, fc in enumerate(fc_list):
            filename_write = "FORCE_CONSTANTS_{}".format(i)
            write_FORCE_CONSTANTS(fc, filename_write)

        filename_write = "FORCE_CONSTANTS_AVERAGE"
        write_FORCE_CONSTANTS(fc_average, filename_write)
