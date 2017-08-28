#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scipy.spatial import ConvexHull, Voronoi
from .volume import Volume

__author__ = 'Yuji Ikeda'
__version__ = '0.1.0'


class VolumeVoronoi(Volume):
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

    def _create_header(self):
        return ''

    def _create_filename(self):
        return 'atomic_volume.dat'


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
