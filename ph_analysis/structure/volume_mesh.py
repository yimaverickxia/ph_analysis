#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from .volume import Volume

__author__ = 'Yuji Ikeda'
__version__ = '0.1.0'


class VolumeMesh(Volume):
    def __init__(self, atoms, mesh):
        super(VolumeMesh, self).__init__(atoms)
        self._mesh = mesh

    def generate_atomic_volume(self, prec=1e-6):
        atoms = self._atoms
        mesh = self._mesh
        cell = atoms.get_cell()
        natoms = atoms.get_number_of_atoms()
        atomic_positions = atoms.get_scaled_positions()
        volumes_atom = np.zeros(natoms)

        get_distances = self._get_distances1

        positions = []
        for x0 in np.linspace(0, 1, mesh[0], endpoint=False):
            for x1 in np.linspace(0, 1, mesh[1], endpoint=False):
                for x2 in np.linspace(0, 1, mesh[2], endpoint=False):
                    position = np.array([x0, x1, x2])
                    positions.append(position)
        positions = np.array(positions)

        for position in positions:
            # TODO(ikeda): This may fail for extremely strange cell shape
            rpos = position - atomic_positions  # broadcasting
            rpos -= np.rint(rpos)  # np.rint seems to be faster than np.around.
            distance = get_distances(rpos)

            distance_min = min(distance)
            atoms_close = []
            weight = 0
            for ia in range(natoms):
                if distance[ia] - distance_min < prec:
                    atoms_close.append(ia)
                    weight += 1

            for ia in atoms_close:
                volumes_atom[ia] += 1.0 / float(weight)

        volumes_atom /= np.product(mesh)

        self._volumes_atom = volumes_atom * np.linalg.det(cell)

        self.generate_values_for_symbols()

    def _create_header(self):
        return '# {} {} {}\n'.format(*self._mesh)

    def _create_filename(self):
        return 'atomic_volume_{}_{}_{}.dat'.format(*self._mesh)


def main():
    import argparse
    from phonopy.interface.vasp import read_vasp
    parser = argparse.ArgumentParser()
    parser.add_argument('atoms',
                        type=str,
                        help="POSCAR")
    parser.add_argument("-m", "--mesh",
                        nargs=3,
                        default=[1, 1, 1],
                        type=int,
                        # required=True,
                        help="mesh")
    args = parser.parse_args()
    atoms = read_vasp(args.atoms)
    VolumeMesh(atoms, args.mesh).run()

if __name__ == '__main__':
    main()
