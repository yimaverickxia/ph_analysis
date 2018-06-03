#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import numpy as np
from fractions import Fraction
from phonopy.structure.atoms import atom_data, symbol_map
from vasp.poscar import Poscar
from primitive_axis import PrimitiveAxis
from .band_path import BandPath


phonopy_conf_order = [
    "ATOM_NAME",
    "EIGENVECTORS",
    "MASS",
    "MAGMOM",
    "DIM",
    "PRIMITIVE_AXIS",
    # Displacement creation tags
    "CREATE_DISPLACEMENTS",
    "DISPLACEMENT_DISTANCE",
    #
    "BAND",
    "BAND_POINTS",
    "BAND_LABELS",
    "MP",
    "MP_SHIFT",
    "GAMMA_CENTER",
    "WRITE_MESH",
    "DOS",
    "DOS_RANGE",
    "PDOS",
    "SIGMA",
    "TETRAHEDRON",
    "TPROP",
    "TDISP",
    "TMAX",
    "TMIN",
    "TSTEP",
    "FORCE_CONSTANTS",
]


# Determine dim automatically from disp.conf.
def get_dim(conf_file):
    dim = None
    with open(conf_file, "r") as f:
        for line in f:
            if "DIM" not in line:
                continue
            dim = [int(x) for x in line.split("=")[-1].split()]
            break
    return dim


def get_average_masses(symbols, mode="a"):
    masses = [atom_data[symbol_map[x]][3] for x in symbols]
    if mode == "a":  # arithmetic mean
        return [sum(masses) / len(masses)] * len(masses)
    elif mode == "g":  # geometric mean
        return [np.prod(masses) ** (1.0 / len(masses))] * len(masses)
    else:
        print("ERROR: mode must be 'a' or 'g'")
        raise ValueError


class PhonopyConfCreator(object):
    def __init__(self,
                 spg_number,
                 mesh=None,
                 tmax=None,
                 is_sqs=False,
                 is_average_mass=False,
                 dim_sqs=None,
                 prior_primitive_axis=None,
                 band_points=None,
                 poscar_name="POSCAR",
                 magmom_line=None,
                 mode_mean_masses="a",
                 is_primitive=False,
                 is_fct=False,
                 is_bcm=False,
                 variables=None):
        if variables is None:
            self._dos_input = {
                "f_min" : -10.0 ,  # THz
                "f_max" :  15.0 ,  # THz
                "d_freq":   0.01,  # THz
                "sigma" :   0.1 ,  # THz
            }
        else:
            self._dos_input = variables

        if mesh is None:
            mesh = [1, 1, 1]
        if dim_sqs is None:
            dim_sqs = [1, 1, 1]

        self._dictionary = {}
        self._poscar = Poscar(poscar_name)
        self._mesh = mesh
        self._tmax = tmax
        self._band_points = band_points

        self._spg_number = spg_number
        self._is_sqs = is_sqs
        self._is_average_mass = is_average_mass
        self._dim_sqs = dim_sqs
        self._is_primitive = is_primitive
        self._is_fct = is_fct
        self._is_bcm = is_bcm
        self._prior_primitive_axis = prior_primitive_axis
        self._mode_mean_masses = mode_mean_masses
        self._magmom_line = magmom_line

        self._dim = get_dim('writefc.conf')

    def run(self):
        self.generate_primitive_axis()
        print("primitive_axis:")
        print(self._primitive_axis)

        if self._is_sqs:
            self._primitive_axis = convert_primitive_axis(
                self._primitive_axis,
                dim=self._dim_sqs)

        primitive_axis_string = write_primitive_axis(self._primitive_axis)

        self._dictionary.update({
            "DIM": (" {:d}" * len(self._dim)).format(*self._dim),
            "PRIMITIVE_AXIS": primitive_axis_string,
            "FORCE_CONSTANTS": "READ",
        })
        if self._magmom_line is not None:
            self._dictionary["MAGMOM"] = self._magmom_line.strip()

        if self._is_average_mass:
            self.create_average_masses(self._poscar)

        self.generate_dictionary_dictionary()

        self.write()

    def create_average_masses(self, poscar):
        ncell = 1.0 / np.linalg.det(self._primitive_axis)
        ncell = int(round(ncell))
        print("ncell:", ncell)

        natoms = poscar.get_atoms().get_number_of_atoms()
        natoms_in_primitive = natoms / ncell

        chemical_symbols = poscar.get_atoms().get_chemical_symbols()
        masses = get_average_masses(chemical_symbols,
                                    mode=self._mode_mean_masses)
        print("masses:")
        print(masses)
        self._dictionary.update({
            "MASS": (" {:.16f}" * natoms_in_primitive)
            .format(*masses[:natoms_in_primitive])
        })

    def generate_dictionary_dictionary(self):
        lattice_constants = self._poscar.get_dictionary()
        band = BandPath(self._spg_number, lattice_constants)
        band_path, band_labels = band.get_band_phonopy()
        variables = self._dos_input

        dos_range_string = "{} {} {}".format(
            variables["f_min"], variables["f_max"], variables["d_freq"]
        )

        self._dictionary_dictionary = {
            "band.conf": self._dictionary.copy(),
            "eigenvectors.conf": self._dictionary.copy(),
            "tprop.conf": self._dictionary.copy(),
            "dos_tetrahedron.conf": self._dictionary.copy(),
            "partial_dos_tetrahedron.conf": self._dictionary.copy(),
            "dos_smearing.conf": self._dictionary.copy(),
            "partial_dos_smearing.conf": self._dictionary.copy(),
        }

        self._dictionary_dictionary["band.conf"].update({
            "BAND": band_path,
            "BAND_LABELS": band_labels,
        })
        if self._band_points is not None:
            self._dictionary_dictionary["band.conf"].update({
                "BAND_POINTS": "{:d}".format(self._band_points)
            })

        self._dictionary_dictionary["eigenvectors.conf"].update({
            "EIGENVECTORS": ".TRUE.",
            "BAND": band_path,
            "BAND_LABELS": band_labels,
            "BAND_POINTS": "2",
        })

        self._dictionary_dictionary["tprop.conf"].update({
            "MP": " {:d} {:d} {:d}".format(*self._mesh),
            "GAMMA_CENTER": ".TRUE.",
            "TPROP": ".TRUE.",
            "TMAX": "{:d}".format(self._tmax),
        })

        self._dictionary_dictionary["dos_tetrahedron.conf"].update({
            "MP": " {:d} {:d} {:d}".format(*self._mesh),
            "GAMMA_CENTER": ".TRUE.",
            "DOS": ".TRUE.",
            "DOS_RANGE": " {}".format(dos_range_string),
            "TETRAHEDRON": ".TRUE.",
        })

        self._dictionary_dictionary["partial_dos_tetrahedron.conf"].update({
            "MP": " {:d} {:d} {:d}".format(*self._mesh),
            "GAMMA_CENTER": ".TRUE.",
            "WRITE_MESH": ".FALSE.",
            "DOS": ".TRUE.",
            "DOS_RANGE": " {}".format(dos_range_string),
            "TETRAHEDRON": ".TRUE.",
            "PDOS": "",
        })

        self._dictionary_dictionary["dos_smearing.conf"].update({
            "MP": " {:d} {:d} {:d}".format(*self._mesh),
            "GAMMA_CENTER": ".TRUE.",
            "DOS": ".TRUE.",
            "DOS_RANGE": " {}".format(dos_range_string),
            "SIGMA": " {}".format(variables["sigma"]),
        })

        self._dictionary_dictionary["partial_dos_smearing.conf"].update({
            "MP": " {:d} {:d} {:d}".format(*self._mesh),
            "GAMMA_CENTER": ".TRUE.",
            "WRITE_MESH": ".FALSE.",
            "DOS": ".TRUE.",
            "DOS_RANGE": " {}".format(dos_range_string),
            "SIGMA": " {}".format(variables["sigma"]),
            "PDOS": "",
        })

    def generate_primitive_axis(self):
        if self._is_primitive:
            primitive_axis = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
            primitive_axis = np.array(primitive_axis)
        else:
            primitive_axis = PrimitiveAxis(
                spg_number=self._spg_number,
                is_fct=self._is_fct,
                is_bcm=self._is_bcm).get_primitive_axis()
        self._primitive_axis = primitive_axis

    def update(self, dictionary):
        self._dictionary.update(dictionary)

    def write(self):
        for filename, dictionary in self._dictionary_dictionary.items():
            write_phonopy_conf(dictionary, filename)


def write_phonopy_conf(phonopy_dict, filename):
    for k in phonopy_dict.keys():
        if k not in phonopy_conf_order:
            print("ERROR {}:".format(__name__))
            print("{:s} is not found in phonopy_conf_order".format(k))
            raise ValueError

    with open(filename, "w") as f:
        for k in phonopy_conf_order:
            if k in phonopy_dict:
                f.write("{:s} = {:s}\n".format(k, phonopy_dict[k]))


def write_primitive_axis(primitive_axis):
    string = ""
    for p1 in primitive_axis:
        for p2 in p1:
            string += " %s" % Fraction(p2).limit_denominator(100)
        string += " "
    return string


def convert_primitive_axis(primitive_axis, dim):
    for i in range(3):
        primitive_axis[:, i] *= 1. / dim[i]
    return primitive_axis


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name",
                        nargs="+",
                        type=str,
                        help="Element name")
    parser.add_argument("-m", "--mesh",
                        nargs=3,
                        default=[80, 80, 80, ],
                        type=int,
                        help="meshes in reciprocal space")
    parser.add_argument("--dim_sqs",
                        nargs=3,
                        type=int,
                        required=True,
                        help="Dimension of SQS.")
    parser.add_argument("--spg_number",
                        type=int,
                        help="space group number")
    parser.add_argument("--band_points",
                        type=int,
                        help="band points")
    parser.add_argument("-l", "--large",
                        action="store_true",
                        help="dimension of supercell")
    parser.add_argument("--fct",
                        action="store_true",
                        help="body centered tetragonal cell is given as "
                               "face centered tetragonal cell")
    parser.add_argument("--bcm",
                        action="store_true",
                        help="")
    parser.add_argument("--primitive_axis",
                        nargs=9,
                        default=[1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0, ],
                        type=float,
                        help="primitive axis of POSCAR")
    parser.add_argument("-r", "--read",
                        default="POSCAR",
                        type=str,
                        help="read POSCAR type file")
    args = parser.parse_args()

    if args.spg_number is not None:
        spg_number = args.spg_number
    else:
        print("ERROR {}:".format(__name__))
        print("Either spg_number or structure must be given.")
        raise ValueError

    PhonopyConfCreator(
        spg_number,
        mesh=args.mesh,
        is_sqs=args.large,
        dim_sqs=args.dim_sqs,
        prior_primitive_axis=args.primitive_axis,
        band_points=args.band_points,
        poscar_name=args.read,
        is_fct=args.fct,
        is_bcm=args.bcm)


if __name__ == "__main__":
    main()
