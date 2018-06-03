#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import sys
import os
import fractions
import yaml
import numpy as np

__author__ = "Yuji Ikeda"

# TODO(ikeda): The Atoms object should be given instead of the spg_number.
# TODO(ikeda): Lists should be used instead of lt or gt signs.
# TODO(ikeda): Distances should be calculated from Atoms and not from yaml.
class BandPath(object):
    def __init__(self,
                 spg_number,
                 lattice_constants=None):
        self._spg_number = spg_number
        self.generate_band_data()
        self.lattice_constants = lattice_constants
        # if self.lattice_constants is not None:
        #     self.change_orientation()

    def generate_band_data(self):
        spg_number = self._spg_number

        str_x00  = "[ξ00]"
        str_x10  = "[ξ10]"
        str_x11  = "[ξ11]"
        str_0x0  = "[0ξ0]"
        str_00x  = "[00ξ]"
        str_xx0  = "[ξξ0]"
        str_xxx  = "[ξξξ]"
        str_xx2x = "[ξξ2ξ]"

        str_1x0  = "[1ξ0]"
        str_x01  = "[ξ01]"
        str_1x1  = "[1ξ1]"
        str_xx1  = "[ξξ1]"
        str_10x  = "[10ξ]"
        str_11x  = "[11ξ]"

        str_00z  = "[00ζ]"
        str_x0z  = "[ξ0ζ]"
        str_01z  = "[01ζ]"
        str_10z  = "[10ζ]"
        str_11z  = "[11ζ]"
        str_xxz  = "[ξξζ]"
        str_xx2z = "[ξξ2ζ]"
        str_x2xz = "[ξ2ξζ]"

        str_0y0  = "[0η0]"
        str_0y1  = "[0η1]"
        str_1y0  = "[1η0]"
        str_1y1  = "[1η1]"
        str_xy0  = "[ξη0]"
        str_0yz  = "[0ηζ]"
        str_xyz  = "[ξηζ]"

        if spg_number in [1, 2]:
            points = {
                u"Γ": [ 0.0,  0.0,  0.0],
                u"X": [ 0.5,  0.0,  0.0],
                u"Y": [ 0.0,  0.5,  0.0],
                u"Z": [ 0.0,  0.0,  0.5],
                u"L": [ 0.5,  0.5,  0.0],
                u"M": [ 0.0,  0.5,  0.5],
                u"N": [ 0.5,  0.0,  0.5],
                u"R": [ 0.5,  0.5,  0.5],
            }
            band_path = [
                [u"X", u"Γ", u"[x00]"],
                [u"Γ", u"M", u"[0yz]"],
                [u"Y", u"Γ", u"[0y0]"],
                [u"Γ", u"N", u"[x0z]"],
                [u"Z", u"Γ", u"[00z]"],
                [u"Γ", u"L", u"[xy0]"],
                [u"Γ", u"R", u"[xyz]"],
            ]

        # simple (or primitive) monoclinic and its family
        # The names of the special points are defined by myself.
        elif spg_number in [3, 4, 6, 7, 10, 11, 13, 14]:
            points = {
                u"Γ": [ 0.0,  0.0,  0.0],
                u"X": [ 0.5,  0.0,  0.0],
                u"Y": [ 0.0,  0.5,  0.0],
                u"Z": [ 0.0,  0.0,  0.5],
                u"L": [ 0.5,  0.5,  0.0],
                u"M": [ 0.0,  0.5,  0.5],
                u"N": [-0.5,  0.0,  0.5],
                u"R": [-0.5,  0.5,  0.5],
            }
            band_path = [
                [u"Γ", u"Y", u"[0y0]" ],
                [u"Y", u"L", None     ],
                [u"L", u"X", None     ],
                [u"X", u"Γ", u"[x00]" ],

                [u"Y", u"M", None     ],
                [u"M", u"Z", None     ],
                [u"Z", u"Γ", u"[00z]" ],

                [u"Y", u"R", None     ],
                [u"R", u"N", None     ],
                [u"N", u"Γ", u"[-x0z]"],
            ]
            

        # C-centered monoclinic and its family
        # The names of the special points are defined by myself.
        elif (spg_number == 5 or
              spg_number == 8 or
              spg_number == 9 or
              spg_number == 12 or
              spg_number == 15):
            band_data = [
                [[ 1./2., 0.   , 0.   ,], u"X"     , "0.5", str_x00, None],
                [[ 0.   , 0.   , 0.   ,], u"Γ     ", "0"  , str_00z, None],
                [[ 0.   , 0.   , 1./2.,], u"Z"     , "0.5", None   , None],
            ]

        # simple orthorhombic and its family
        # all commensurate points for 2x2x2 supercell is included
        elif (16 <= spg_number <= 19 or  # 222
              25 <= spg_number <= 34 or  # mm2
              47 <= spg_number <= 62):   # mmm
            band_data = [
                [[ 0.   , 0.   , 0.   ,], u"Γ     ", "0  ", str_x00, None],
                [[ 1./2., 0.   , 0.   ,], u"X     ", "0.5", None   , None],
                [[ 1./2., 1./2., 0.   ,], u"S     ", ""   , None   , None],
                [[ 0.   , 1./2., 0.   ,], u"Y     ", "0.5", str_0y0, None],
                [[ 0.   , 0.   , 0.   ,], u"Γ     ", "0  ", str_00z, None],
                [[ 0.   , 0.   , 1./2.,], u"Z     ", "0.5", None   , None],
                [[ 1./2., 0.   , 1./2.,], u"U     ", ""   , None   , None],
                [[ 1./2., 1./2., 1./2.,], u"R     ", ""   , None   , None],
                [[ 0.   , 1./2., 1./2.,], u"T     ", ""   , None   , None],
                [[ 0.   , 0.   , 1./2.,], u"Z     ", ""   , None   , None],
            ]
            points = {
                u"Γ": [ 0.0 ,  0.0 ,  0.0 ],
                u"X": [ 0.5 ,  0.0 ,  0.0 ],
                u"Y": [ 0.0 ,  0.5 ,  0.0 ],
                u"Z": [ 0.0 ,  0.0 ,  0.5 ],
                u"S": [ 0.5 ,  0.5 ,  0.0 ],
                u"T": [ 0.0 ,  0.5 ,  0.5 ],
                u"U": [ 0.5 ,  0.0 ,  0.5 ],
                u"R": [ 0.5 ,  0.5 ,  0.5 ],
            }
            band_path = [
                [u"Γ", u"X", str_x00],
                [u"X", u"S", str_1y0],
                [u"S", u"Y", str_x10],
                [u"Y", u"Γ", str_0y0],
                [u"Γ", u"Z", str_00z],
                [u"Z", u"U", str_x01],
                [u"U", u"R", str_1y1],
                [u"R", u"T", str_x11],
                [u"T", u"Z", str_0y1],
                [u"X", u"U", str_10z],
                [u"Y", u"T", str_01z],
                [u"S", u"R", str_11z],
            ]

        # C-centered orthorhombic and its family (a < b < c)
        # all commensurate points for 2x2x2 supercell is included
        elif (20 <= spg_number <= 21 or  # 222
              35 <= spg_number <= 37 or  # mm2, C
              38 <= spg_number <= 41 or  # mm2, A
              63 <= spg_number <= 68):   # mmm
            band_data = [
                [[ 1./2., 0.   , 0.   ,], u"S     ", "0.5", str_xy0, None],
                [[ 1.   , 0.   , 0.   ,], u"Γ     ", "0  ", str_x00, None],
                [[ 1./2., 1./2., 0.   ,], u"Y     ", "1  ", str_0y0, None],
                [[ 0.   , 0.   , 0.   ,], u"Γ     ", "0  ", str_00z, None],
                [[ 0.   , 0.   , 1./2.,], u"Z     ", "0.5", None   , None],
                [[ 1./2., 1./2., 1./2.,], u"T     ", ""   , None   , None],
                [[ 1.   , 0.   , 1./2.,], u"Z     ", ""   , None   , None],
                [[ 1./2., 0.   , 1./2.,], u"R     ", ""   , None   , None],
            ]

        # face-centered orthorhombic and its family (a < b < c)
        # all commensurate points for 2x2x2 supercell is included
        elif (spg_number == 22 or        # 222
              42 <= spg_number <= 43 or  # mm2
              69 <= spg_number <= 70):   # mmm
            band_data = [
                [[ 0.   , 0.   , 0.   ,], u"Γ     ", "0  ", str_x00, None], # via X
                [[ 0.   , 1./2., 1./2.,], u"T     ", "1  ", None   , None],
                [[ 1./2., 0.   , 0.   ,], u"Z     ", ""   , None   , None], # via A
                [[ 1./2., 0.   , 1./2.,], u"Y     ", "1  ", str_0y0, None],
                [[ 0.   , 0.   , 0.   ,], u"Γ     ", "0  ", str_x0z, None],
                [[ 1./2., 1.   , 1./2.,], u"Y     ", "1  ", None   , None],
                [[ 1./2., 1./2 , 0.   ,], u"T     ", "1  ", str_0yz, None],
                [[ 1.   , 1.   , 1.   ,], u"Γ     ", "0  ", str_00z, None],
                [[ 1./2 , 1./2., 1.   ,], u"Z     ", "1  ", str_xy0, None],
                [[ 0.   , 0.   , 0.   ,], u"Γ     ", "0  ", str_xyz, None],
                [[ 1./2., 1./2., 1./2.,], u"L     ", ""   , None   , None],
            ]

        # body-centered orthorhombic and its family (a < b < c)
        elif (23 <= spg_number <= 24 or  # 222
              44 <= spg_number <= 46 or  # 222
              71 <= spg_number <= 74):   # mmm
            band_data = [
                [[ 1./2., 0.   , 0.   ,], u"R     ", ""   , None   , None],
                [[ 1./4., 1./4., 1./4.,], u"W     ", ""   , None   , None],
                [[ 1./2., 1./2., 0.   ,], u"T     ", ""   , None   , None],
                [[ 1./2., 1./2., 1./2.,], u"Z     ", "1  ", str_00z, None], # via X
                [[ 0.   , 0.   , 1.   ,], u"Γ     ", "0  ", str_x00, None], # via Y
                [[ 1./2.,-1./2., 1./2.,], u"Z     ", "1  ", str_0y0, None],
                [[ 0.   , 0.   , 0.   ,], u"Γ     ", "0  ", str_xyz, None],
                [[ 1./4., 1./4., 1./4.,], u"W     ", "0.5", None   , None],
                [[ 0.   , 1./2., 0.   ,], u"S     ", None , None   , None],
            ]

        # simple tetragonal and its family
        # all commensurate points for 2x2x2 supercell is included
        elif (75 <= spg_number <= 78 or    #  4   , C_4
              spg_number == 81 or          # -4   , S_4
              83 <= spg_number <= 86 or    #  4/m , C_4h
              89 <= spg_number <= 96 or    #  422 , D_4
              99 <= spg_number <= 106 or   #  4mm , C_4v
              111 <= spg_number <= 118 or  # -42m , D_2d
              123 <= spg_number <= 138):   # 4/mmm, D_4h
            points = {
                u"Γ": [ 0.0 ,  0.0 ,  0.0 ],
                u"X": [ 0.5 ,  0.0 ,  0.0 ],
                u"M": [ 0.5 ,  0.5 ,  0.0 ],
                u"Z": [ 0.0 ,  0.0 ,  0.5 ],
                u"R": [ 0.5 ,  0.0 ,  0.5 ],
                u"A": [ 0.5 ,  0.5 ,  0.5 ],
            }
            band_path = [
                [u"Γ", u"X", str_x00],
                [u"X", u"M", str_1x0],
                [u"M", u"Γ", str_xx0],
                [u"Γ", u"Z", str_00x],
                [u"Z", u"R", str_x01],
                [u"R", u"A", str_1x1],
                [u"A", u"Z", str_xx1],
                [u"X", u"R", str_10x],
                [u"M", u"A", str_11x],
            ]

        # body centered tetragonal and its family
        # it depends on whether c/a > 1 or c/a < 1
        elif (79 <= spg_number <= 80 or    #  4   , C_4
              spg_number <= 82 or          # -4   , S_4
              87 <= spg_number <= 88 or    #  4/m , C_4h
              97 <= spg_number <= 98 or    #  422 , D_4
              107 <= spg_number <= 110 or  #  4mm , C_4v
              119 <= spg_number <= 122 or  # -42m , D_2d
              139 <= spg_number <= 142):   # 4/mmm, D_4h
            band_data = [
                [[ 1.0 , 0.0 , 0.0 ,], u"Γ     ", "0  ", str_xx2z, None],
                [[ 0.0 , 0.0 , 0.5 ,], u"X     ", "0.5", None    , None],
                [[-0.5 , 0.5 , 0.5 ,], u"M     ", "0.5", str_x00 , None],
                [[ 0.0 , 1.0 , 0.0 ,], u"Γ     ", "0  ", str_x2xz, None],
                [[ 0.5 , 0.0 , 0.0 ,], u"N     ", "0.5", str_x0z , None],
                [[ 0.0 , 0.0 , 0.0 ,], u"Γ     ", "0  ", str_00z , None],
                [[ 0.5 , 0.5 ,-0.5 ,], u"M     ", "1  ", None    , None],
                [[ 0.25, 0.25, 0.25,], u"P     ", "0.5", str_xxz , True],
                [[ 0.0 , 0.0 , 0.0 ,], u"Γ     ", "0  ", str_xx0 , None],
                [[ 0.0 , 0.0 , 0.5 ,], u"X     ", "0.5", None    , None],
                # [[ 0.25, 0.25, 0.25,], u"P     ", "0.5", None    , None],
                # [[ 0.5 , 0.0 , 0.0 ,], u"N     ", "0.5", None    , None],
            ]

        elif (143 <= spg_number <= 167 or  # trigonal bravais lattice
              168 <= spg_number <= 194):   # hexagonal bravais lattice
            band_data = [
                [[ 0.   , 0.   , 1./2.,], u"A     ", ""   , None   , None],
                [[ 1./2., 0.   , 1./2.,], u"L     ", ""   , None   , None],
                [[ 1./2., 0.   , 0.   ,], u"M     ", "1/2", str_x00, None],
                [[ 0.   , 0.   , 0.   ,], u"Γ     ", "0"  , str_00z, None],
                [[ 0.   , 0.   , 1./2.,], u"A     ", "1/2", None   , None],
                [[ 1./3., 1./3., 1./2.,], u"H     ", ""   , None   , None],
                [[ 1./3., 1./3., 0.   ,], u"K     ", "1/3", str_xx0, None],
                [[ 0.   , 0.   , 0.   ,], u"Γ     ", "0"  , None   , None],
            ]
            points = {
                u"Γ": [ 0.0  ,  0.0  ,  0.0  ],
                u"K": [ 1./3.,  1./3.,  0.0  ],
                u"M": [ 1./2.,  0.0  ,  0.0  ],
                u"A": [ 0.0  ,  0.0  ,  1./2.],
                u"H": [ 1./3.,  1./3.,  1./2.],
                u"L": [ 1./2.,  0.0  ,  1./2.],
            }
            band_path = [
                [u"Γ", u"K", u"[xx0]"],
                [u"K", u"M", None    ],
                [u"M", u"Γ", u"[x00]"],
                [u"Γ", u"A", u"[00z]"],
                [u"A", u"H", u"[xx1]"],
                [u"H", u"L", None    ],
                [u"L", u"A", u"[x01]"],
                [u"K", u"H", None    ],
                [u"M", u"L", None    ],
            ]

        # simple cubic and its family
        elif (spg_number in [195, 198] or            # 23      , T
              spg_number in [200, 201, 205] or       # 2/m-3   , T_h
              spg_number in [207, 208, 212, 213] or  # 432     , O
              spg_number in [215, 218] or            # -43m    , T_d
              spg_number in [221, 222, 223, 224]):   # 4/m-32/m, O_h
            band_data = [
                [[ 0.0, 0.0, 0.0,], u"Γ     ", "0"  , str_00x, None],
                [[ 0.0, 0.5, 0.0,], u"X     ", "0.5", None   , None],
                [[ 0.5, 0.5, 0.0,], u"M     ", "0.5", str_xx0, None],
                [[ 0.0, 0.0, 0.0,], u"Γ     ", "0"  , str_xxx, None],
                [[ 0.5, 0.5, 0.5,], u"R     ", "0.5", None   , None],
                [[ 0.5, 0.5, 0.0,], u"M     ", "0.5", None   , None],
            ]
            points = {
                u"Γ": [0.0, 0.0, 0.0],
                u"X": [0.5, 0.0, 0.0],
                u"M": [0.5, 0.5, 0.0],
                u"R": [0.5, 0.5, 0.5],
            }
            band_path = [
                [u"Γ", u"X", u"[x00]"],
                [u"X", u"M", u"[1x0]"],
                [u"M", u"Γ", u"[xx0]"],
                [u"Γ", u"R", u"[xxx]"],
                [u"R", u"M", u"[11x]"],
                [u"R", u"X", u"[1xx]"],
            ]

        # face centered cubic and its family
        elif spg_number in [
                196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228]:
            points = {
                u"Γ"  : [ 0.0 ,  0.0 ,  0.0 ],
                u"X"  : [ 0.0 ,  0.5 ,  0.5 ],
                u"W"  : [ 0.25,  0.5 ,  0.75],
                u"X_2": [ 0.5 ,  0.5 ,  1.0 ],
                u"L"  : [ 0.5 ,  0.5 ,  0.5 ],
            }
            # band_path = [
            #     [u"Γ"  , u"X"  , str_x00],
            #     [u"X"  , u"W"  , None   ],
            #     [u"W"  , u"X_2", None   ],
            #     [u"X_2", u"Γ"  , str_xx0],
            #     [u"Γ"  , u"L"  , str_xxx],
            # ]
            band_path = [
                [u"Γ"  , u"X"  , str_x00],
                [u"X_2", u"Γ"  , str_xx0],
                [u"Γ"  , u"L"  , str_xxx],
            ]

        # body centered cubic and its family
        elif spg_number in [197, 199, 204, 206, 211, 214, 217, 220, 229, 230]:
            band_data = [
                [[ 0.0 , 1.0 , 0.0 ,], u"Γ     ", "0"  , str_xx2x, None],
                [[ 0.5 , 0.0 , 0.0 ,], u"N     ", "0.5", str_xx0 , None],
                [[ 0.0 , 0.0 , 0.0 ,], u"Γ     ", "0"  , str_x00 , None],
                [[-0.5 , 0.5 , 0.5 ,], u"H     ", "1"  , None    , None],
                [[ 0.25, 0.25, 0.25,], u"P     ", "0.5", str_xxx , True],
                [[ 0.0 , 0.0 , 0.0 ,], u"Γ     ", "0  ", None    , None],
            ]
            points = {
                u"Γ": [ 0.0 ,  0.0 ,  0.0 ],
                u"H": [-0.5 ,  0.5 ,  0.5 ],
                u"N": [ 0.0 ,  0.0 ,  0.5 ],
                u"P": [ 0.25,  0.25,  0.25],
            }
            band_path = [
                [u"Γ", u"H", u"[x00]"],
                [u"H", u"N", u"[1x0]"],
                [u"N", u"Γ", u"[xx0]"],
                [u"Γ", u"P", u"[xxx]"],
                [u"P", u"H", u"[x11]"],
                [u"P", u"N", u"[xx1]"],
            ]

        else:
            print("ERROR: {}".format(__name__))
            print("Invarid spg_number {}".format(spg_number))
            raise ValueError

        self._band_data = {
            "points": points,
            "band_path": band_path,
        }

    def change_orientation(self):
        print("Orientation for q points are changed.")
        spg_number = self._spg_number
        a = self.lattice_constants["a"]
        b = self.lattice_constants["b"]
        c = self.lattice_constants["c"]

        # default: no r_matrix
        r_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        if (16 <= spg_number <= 74):  # orthorhombic
            if (a < b < c):
                r_matrix = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ])
            elif (a < c < b):
                r_matrix = np.array([
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                ])
            elif (b < a < c):
                r_matrix = np.array([
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                ])
            elif (b < c < a):
                r_matrix = np.array([
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0],
                ])
            elif (c < a < b):
                r_matrix = np.array([
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                ])
            elif (c < b < a):
                r_matrix = np.array([
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                ])
        elif (75 <= spg_number <= 142):  # tetragonal
            if (a < c and b < c):
                r_matrix = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ])
            elif (a < b and c < b):
                r_matrix = np.array([
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                ])
            elif (b < a and c < a):
                r_matrix = np.array([
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0],
                ])

        band_path = self._band_data["band_path"]
        self._band_data["band_path"] = np.dot(band_path, np.linalg.inv(r_matrix))

    def get_band_phonopy(self):

        str_band_path = " "
        str_band_labels = " "

        points = self._band_data["points"]
        band_path = self._band_data["band_path"]

        get_string_from_point = self.get_string_from_point
        get_string_from_label = self.get_string_from_label

        for i, path_segment in enumerate(band_path):
            label_0 = path_segment[0]
            label_1 = path_segment[1]
            if i == 0:
                str_band_path += get_string_from_point(points[label_0])
            elif label_0 is not label_old:
                str_band_path += ","
                str_band_path += get_string_from_point(points[label_0])
            str_band_path += " "
            str_band_path += get_string_from_point(points[label_1])

            if i == 0:
                str_band_labels += get_string_from_label(label_0)
            elif not label_0.split("_")[0] == label_old.split("_")[0]:
                str_band_labels += "|"
                str_band_labels += get_string_from_label(label_0)
            str_band_labels += " "
            str_band_labels += get_string_from_label(label_1)

            label_old = label_1

        return str_band_path, str_band_labels

    def get_string_from_point(self, point):
        string = ""
        for p in point:
            tmp = fractions.Fraction(p).limit_denominator(100)
            string += " {}".format(tmp)
        return string

    def get_string_from_label(self, label):
        if label.split("_")[0] == u"Γ":
            return r"\Gamma"
        else:
            return label.split("_")[0]

    def get_band_gnuplot_utf8(
            self,
            interval,
            distance,
            has_x2=False,
            filename="plot_label.plt"):

        fg = open(filename, "w")

        band_path = self._band_data["band_path"]
        band_labels = self._band_data["band_labels"]
        reduced_wave_vectors = self._band_data["reduced_wave_vectors"]
        directions = self._band_data["directions"]
        direction_flags = self._band_data["direction_flags"]

        band_path = list(band_path)
        band_labels = list(band_labels)

        fg.write("#!/usr/bin/env gnuplot\n")

        fg.write("if (f_min != NaN) {\n")
        fg.write("    f_pos = f_min * f_scale * 0.5\n")
        fg.write("} else {f_pos = -5}\n")

        def plot_labels(band_labels):
            for i, l in enumerate(band_labels):

                fg.write("  '%s'" % l.strip().encode("utf-8"))

                if i == 0:
                    n = 0
                else:
                    n = interval * i - 1
                fg.write("  {:16.8E}".format(distance[n]))

                if i == len(band_labels) - 1:
                    fg.write("  \\\n")
                    fg.write(")\n")
                else:
                    fg.write(", \\\n")

        fg.write("if (int_has_x2 == 1) {\n")
        fg.write("set x2label 'Wave vector'\n")
        fg.write("set x2tics ( \\\n")
        plot_labels(band_labels)

        fg.write("set xlabel 'Reduced wave vector'\n")
        fg.write("set xtics ( \\\n")
        for i, r in enumerate(reduced_wave_vectors):
            if r is not None:
                fg.write(
                    "  {:10s}".format("'" + r.strip().encode("utf-8") + "'"))

                if i == 0:
                    n = 0
                else:
                    n = interval * i - 1
                fg.write("  %16.8E" % distance[n])

            if i == len(band_labels) - 1:
                fg.write("  \\\n")
            else:
                fg.write(", \\\n")
        fg.write(")\n")
        fg.write("\n")

        # If reduced vectors are shown, we also show the directions.
        if any([x is not None for x in directions]):
            for i, (d, f) in enumerate(zip(directions, direction_flags)):
                if d is not None:
                    fg.write("set label '{:s}' \\\n".format(d))

                    if i == 0:
                        n = 0
                    else:
                        n = interval * i - 1
                    n_next = interval * (i + 1) - 1
                    fg.write("at ")
                    if f is True:
                        fg.write("{:16.8E}".format(distance[n]))
                    else:
                        fg.write("(")
                        fg.write("{:16.8E}".format(distance[n]))
                        fg.write(" + ")
                        fg.write("{:16.8E}".format(distance[n_next]))
                        fg.write(") / 2.0")
                    fg.write(", f_pos \\\n")
                    fg.write("front \\\n")
                    fg.write("center\n")
                    fg.write("\n")

        fg.write("} else {\n")
        fg.write("    set xlabel 'Wave vector'\n")
        fg.write("    set xtics ( \\\n")
        plot_labels(band_labels)
        fg.write("}\n")

        fg.write("\n")
        fg.write("set xrange[%16.8E:%16.8E]\n" % (distance[0], distance[-1]))

        fg.close()
    # set object rectangle front \
    # at (3.04855853E-01 + 0.00000000E+00) / 2.0, f_pos \
    # size 0.11, 7 fs noborder

    def get_band_kpoints(self):
        pass

    def get_band_data(self):
        return self._band_data


def read_band_yaml(filename):
    band_yaml = yaml.load(open(filename, "r"))
    interval = int(band_yaml["nqpoint"]) / int(band_yaml["npath"])
    distance = [x["distance"] for x in band_yaml["phonon"]]
    distance = np.array(distance)
    distance /= distance[-1]
    return interval, distance


def read_gruneisen_yaml(filename):
    gruneisen_dict = gruneisen_to_dat.read_gruneisen_yaml("gruneisen.yaml")
    interval = gruneisen_dict["nqpoint"][0]
    distance = gruneisen_dict["distance"].flatten()
    return interval, distance


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--poscar",
                        type=str,
                        help="space group number")
    parser.add_argument("--spg_number",
                        type=int,
                        required=True,
                        help="space group number")
    parser.add_argument("-o", "--out",
                        type=str,
                        required=True,
                        help="type of output file")
    parser.add_argument("--has_x2",
                        action="store_true",
                        help="reduced wave vector is written")
    args = parser.parse_args()

    lattice_constants = None
    if args.poscar is not None:
        lattice_constants = Poscar(args.poscar).get_dictionary()
        # In future, spg_number can be taken from poscar_data.
        if args.spg_number is None:
            pass

    band = BandPath(args.spg_number, lattice_constants)
    band_data = band.get_band_data()

    if args.out == "phonopy":
        band_path, band_labels = band.get_band_phonopy()
        print(band_path)
        print(band_labels)

    elif args.out == "gnuplot":
        if os.path.exists("band.yaml"):
            interval, distance = read_band_yaml("band.yaml")
        elif os.path.exists("gruneisen.yaml"):
            interval, distance = read_gruneisen_yaml("gruneisen.yaml")
        else:
            print("ERROR: either band.yaml or gruneisen.yaml is needed")
            sys.exit(1)
        band.get_band_gnuplot_utf8(
            interval,
            distance)

    elif args.out == "kpoints":
        sys.exit("ERROR: An output for KPOINTS has not implemented yet.")

    else:
        sys.exit("ERROR: Invalid args.out.")

if __name__ == "__main__":
    main()
