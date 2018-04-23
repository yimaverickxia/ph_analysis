#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from functools import reduce
import numpy as np
import pandas as pd


def get_sfe1(fcc, hcp, dhcp, area):
    sfe = 2.0 * (hcp - fcc) / area
    return sfe


def get_sfe2(fcc, hcp, dhcp, area):
    sfe = (hcp + 2.0 * dhcp - 3.0 * fcc) / area
    return sfe


def get_tbe2(fcc, hcp, dhcp, area):
    return 2.0 * (dhcp - fcc) / area


def get_area_from_volume_per_atom(volume_per_atom):
    a = (volume_per_atom * 4) ** (1.0 / 3.0)
    bond_length = a * np.sqrt(2.0) * 0.5
    area = (np.sqrt(3.0) * 0.5) * bond_length ** 2
    return area


def convert_eV_per_Ang2_to_mJ_per_m2(energy_old):
    from scipy.constants import electron_volt
    energy_new = energy_old * electron_volt * 1000.0 * 1e+20
    return energy_new


def merge(df, columns):
    structures = ['fcc', 'hcp', 'dhcp']
    df_dict = {
            s: df[df['structure'] == s].drop('structure', 1) for s in structures}
    for k, v in df_dict.items():
        cols = {c: c + '_' + k for c in v.columns if c not in columns}
        v.rename(columns=cols, inplace=True)
    df_merged = reduce(
            lambda left, right: pd.merge(left, right, how='inner', on=columns),
            df_dict.values())
    return df_merged


def calculate_sfes(df, columns, cols_free_energy):
    df_final = merge(df, columns)
    df_final = add_sfes(df_final, cols_free_energy)
    return df_final


def add_sfes(df_final, col_free_energy):
    volume_per_atom = df_final['volume_per_atom']
    area = get_area_from_volume_per_atom(volume_per_atom)
    sfe_functions = [get_sfe1, get_sfe2]
    convert = convert_eV_per_Ang2_to_mJ_per_m2
    for i, sfe_function in enumerate(sfe_functions):
        key = 'ISFE{}'.format(i + 1)
        df_final[key] = convert(sfe_function(
            df_final[col_free_energy + '_fcc' ],
            df_final[col_free_energy + '_hcp' ],
            df_final[col_free_energy + '_dhcp'],
            area,
        ))
    return df_final
