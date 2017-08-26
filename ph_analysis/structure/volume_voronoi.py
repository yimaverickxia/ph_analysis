#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scipy.spatial import ConvexHull, Voronoi

__author__ = 'Yuji Ikeda'
__version__ = '0.1.0'


def group_into_symbols(symbols, values_atom):
    values_symbol = {}
    for s in sorted(set(symbols), key=symbols.index):
        indices = [ia for ia in range(len(symbols)) if symbols[ia] == s]
        values_symbol[s] = values_atom[indices]
    return values_symbol


class VolumeVoronoi(object):
    def __init__(self, atoms):
        self._atoms = atoms
        self.create_filename()

    def run(self):
        self.generate_atomic_volume()
        self.write_atomic_volume()

    def _get_distances1(self, rpos):
        cell = self._atoms.get_cell()
        rpos = np.dot(rpos, cell)
        distances = (rpos[:, 0] * rpos[:, 0] +
                     rpos[:, 1] * rpos[:, 1] +
                     rpos[:, 2] * rpos[:, 2])
        return np.sqrt(distances)

    def generate_atomic_volume(self, prec=1e-6):
        atoms = self._atoms
        cell = atoms.get_cell()
        natoms = atoms.get_number_of_atoms()
        scaled_positions = atoms.get_scaled_positions()

        expansion = [
            [ 0,  0,  0],
            [ 0,  0,  1],
            [ 0,  0, -1],
            [ 0,  1,  0],
            [ 0,  1,  1],
            [ 0,  1, -1],
            [ 0, -1,  0],
            [ 0, -1,  1],
            [ 0, -1, -1],
            [ 1,  0,  0],
            [ 1,  0,  1],
            [ 1,  0, -1],
            [ 1,  1,  0],
            [ 1,  1,  1],
            [ 1,  1, -1],
            [ 1, -1,  0],
            [ 1, -1,  1],
            [ 1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  1],
            [-1,  0, -1],
            [-1,  1,  0],
            [-1,  1,  1],
            [-1,  1, -1],
            [-1, -1,  0],
            [-1, -1,  1],
            [-1, -1, -1],
            ]
        expansion = np.array(expansion)

        scaled_positions_expanded = np.reshape(
            scaled_positions[None, None, :, :] + expansion[None, :, None, :],
            (-1, 3))

        positions_expanded = np.dot(scaled_positions_expanded, cell)

        voronoi = Voronoi(positions_expanded)

        volumes_atom = np.full(natoms, np.nan)
        for i in range(natoms):
            j = voronoi.point_region[i]
            region = voronoi.regions[j]
            if np.all(np.asarray(region) >= 0):
                vertices = voronoi.vertices[region]
                # Since a Voronoi cell is always a convex polyhedron,
                # we can obtain it's volume using ConvexHull.
                # https://stackoverflow.com/questions/17129115
                volume = ConvexHull(vertices).volume
                volumes_atom[i] = volume
            else:
                raise ValueError('Region includes infinite point')

        self._volumes_atom = volumes_atom

        self.generate_values_for_symbols()

    def generate_values_for_symbols(self):
        symbols = self._atoms.get_chemical_symbols()
        volumes_atom = self._volumes_atom

        volumes_symbol = group_into_symbols(symbols, volumes_atom)

        self._volumes_symbol = volumes_symbol

    def write_atomic_volume(self):
        symbols = self._atoms.get_chemical_symbols()
        volumes_atom = self._volumes_atom
        volumes_symbol = self._volumes_symbol
        filename = self._filename

        with open(filename, "w") as f:
            f.write("#" + " " * 21)
            f.write("{:18s}".format("Voronoi_volume"))
            f.write("\n")
            for i, s in enumerate(symbols):
                f.write("atom")
                f.write(" {:11d} {:5s}".format(i, s))
                f.write("{:18.12f}".format(self._volumes_atom[i]))
                f.write("\n")
            f.write("\n")

            def write_statistics(values, symbol=""):
                f.write("{:16s} {:5s}".format("sum", symbol))
                for v in values:
                    f.write("%18.12f" % (np.sum(v)))
                f.write("\n")

                f.write("{:16s} {:5s}".format("average", symbol))
                for v in values:
                    f.write("%18.12f" % (np.average(v)))
                f.write("\n")

                f.write("{:16s} {:5s}".format("s.d.", symbol))
                for v in values:
                    f.write("%18.12f" % (np.std(v)))
                f.write("\n")

                f.write("{:16s} {:5s}".format("absolute_sum", symbol))
                for v in values:
                    f.write("%18.12f" % (np.sum(abs(v))))
                f.write("\n")

                f.write("{:16s} {:5s}".format("absolute_average", symbol))
                for v in values:
                    f.write("%18.12f" % (np.average(abs(v))))
                f.write("\n")

                f.write("{:16s} {:5s}".format("absolute_s.d.", symbol))
                for v in values:
                    f.write("%18.12f" % (np.std(abs(v))))
                f.write("\n")

                f.write("\n")

            properties = [
                volumes_atom,
            ]
            write_statistics(properties)
            for s in sorted(set(symbols), key=symbols.index):
                properties = [
                    volumes_symbol[s],
                ]
                write_statistics(properties, s)

    def create_filename(self):
        self._filename = 'atomic_volume.dat'


def main():
    import argparse
    from phonopy.interface.vasp import read_vasp
    parser = argparse.ArgumentParser()
    parser.add_argument('atoms',
                        type=str,
                        help="POSCAR")
    args = parser.parse_args()
    atoms = read_vasp(args.atoms)
    VolumeVoronoi(atoms).run()

if __name__ == '__main__':
    main()

