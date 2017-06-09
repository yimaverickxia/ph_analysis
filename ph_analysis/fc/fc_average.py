#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import argparse
import numpy as np
from vasp.poscar_list import PoscarList
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS
from phonopy.harmonic.force_constants import symmetrize_force_constants


__author__ = "Yuji Ikeda"


def write_fc_average(poscar_filename_list, fc_filename_list, weights):

    poscar_list = PoscarList(poscar_filename_list)
    indices_list = poscar_list.get_indices_of_positions()

    fc_list = []
    for fc_filename in fc_filename_list:
        force_constants = parse_FORCE_CONSTANTS(fc_filename)
        symmetrize_force_constants(force_constants)
        fc_list.append(force_constants)
    print(fc_list)

    fc_list_tmp = []
    for fc, indices in zip(fc_list, indices_list):
        fc_tmp = np.full_like(fc_list[0], np.inf)
        for i1, j1 in enumerate(indices):
            for i2, j2 in enumerate(indices):
                fc_tmp[j1, j2] = fc[i1, i2]
        fc_list_tmp.append(fc_tmp)
    fc_list = fc_list_tmp

    fc_average = np.average(fc_list, axis=0, weights=weights)

    for i, fc in enumerate(fc_list):
        filename_write = "FORCE_CONSTANTS_{}".format(i)
        write_FORCE_CONSTANTS(force_constants, filename_write)

    filename_write = "FORCE_CONSTANTS_AVERAGE"
    write_FORCE_CONSTANTS(fc_average, filename_write)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights",
                        nargs="+",
                        type=float,
                        help="Weights for the sets of the force constants.")
    parser.add_argument("-p", "--poscar",
                        nargs="+",
                        type=str,
                        help="Filenames of POSCAR.")
    parser.add_argument("-f", "--fc",
                        nargs="+",
                        type=str,
                        help="Filenames of FORCE_CONSTANTS.")
    args = parser.parse_args()

    if len(args.poscar) != len(args.fc):
        print("ERROR: {}".format(__name__))
        print("len(args.poscar) must be equal to len(args.fc).")
        raise ValueError

    write_fc_average(args.poscar, args.fc, args.weights)


if __name__ == "__main__":
    main()
