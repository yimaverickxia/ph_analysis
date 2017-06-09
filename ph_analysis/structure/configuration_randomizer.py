#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

__author__ = "Yuji Ikeda"

import numpy as np


class ConfigurationRandomizer(object):
    def __init__(self, atoms, map_s2s, random_seed=None):
        """

        Parameters
        ----------
        atoms : Atoms object.
        map_s2s : dictionary.
        """
        self._atoms = atoms
        self._map_s2s = map_s2s
        if random_seed is not None:
            self.set_random_seed(random_seed)

    def set_random_seed(self, random_seed):
        np.random.seed(random_seed)

    def create_randomized_configuration(self):
        symbols_orig = self._atoms.get_chemical_symbols()
        symbols_randomized = symbols_orig[:]
        for symbol_replaced, dict_symbols_new in self._map_s2s.items():
            indices_replaced = [
                i for i, _ in enumerate(symbols_orig) if _ == symbol_replaced
            ]
            natoms_replaced = len(indices_replaced)

            indices_replaced = np.random.permutation(indices_replaced)

            sum_ratio = float(sum([_ for _ in dict_symbols_new.values()]))
            i_s = 0
            for symbol_new, ratio in dict_symbols_new.items():
                i_f = i_s + int(round(natoms_replaced * ratio / sum_ratio))
                for index in indices_replaced[i_s:i_f]:
                    symbols_randomized[index] = symbol_new
                i_s += i_f

            if i_f != natoms_replaced:
                print("ERROR: The number of the replaced atoms is not equal "
                      "to that of the atoms which should be replaced.")
                print(i_f, natoms_replaced)
                print("One should change the values of the \"map_s2s\"")
                raise ValueError

        print("symbols_randomized:")
        print(symbols_randomized)

        import copy
        atoms_new = copy.deepcopy(self._atoms)
        atoms_new.set_chemical_symbols(symbols_randomized)
        return atoms_new


def create_randomized_configurations(mapfile,
                                     infile,
                                     random_seed,
                                     numconf,
                                     is_sorted):
    """

    Parameters
    ----------
    mapfile : String
        Input file including mapping information.
    infile : String
        POSCAR filename.
    """
    import json
    from vasp.poscar import Poscar
    from collections import OrderedDict

    atoms = Poscar(infile).get_atoms()

    with open(mapfile, "r") as f:
        map_s2s = json.load(f, object_pairs_hook=OrderedDict)

    order = create_order_of_symbols(map_s2s)

    configuration_randomizer = ConfigurationRandomizer(
        atoms=atoms,
        map_s2s=map_s2s,
        random_seed=random_seed,
    )

    for i in range(numconf):
        atoms = configuration_randomizer.create_randomized_configuration()
        filename = "RPOSCAR-{}".format(i + 1)
        poscar = Poscar().set_atoms(atoms)
        if is_sorted:
            poscar.sort_by_symbols(order=order)
        poscar.write(filename)


def create_order_of_symbols(map_s2s):
    order = []
    for s_replaced, map_tmp in map_s2s.items():
        for s_new in map_tmp:
            order.append(s_new)
    return order


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mapfile",
                        # default="map.json",
                        type=str,
                        help="File including map_s2s.")
    parser.add_argument("-i", "--infile",
                        default="POSCAR",
                        type=str,
                        help="Input POSCAR file.")
    parser.add_argument("-n", "--numconf",
                        default=1,
                        type=int,
                        help="Number of POSCAR files "
                             "with randomized atomic configurations.")
    parser.add_argument("-s", "--seed",
                        type=int,
                        help="Random seed.")
    parser.add_argument("--is_sorted",
                        action="store_true",
                        help="Output POSCAR files are sorted.")
    args = parser.parse_args()

    create_randomized_configurations(
        mapfile=args.mapfile,
        infile=args.infile,
        random_seed=args.seed,
        numconf=args.numconf,
        is_sorted=args.is_sorted,
    )


if __name__ == "__main__":
    main()
