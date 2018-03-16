#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scipy.constants import physical_constants

k_b = physical_constants['Boltzmann constant in eV/K'][0]


def fermi_dirac_distribution(temperature, efermi, energy):
    x = energy - efermi
    if temperature == 0.0:
        return np.heaviside(-x, 0.5)
    else:
        return 1.0 / (np.exp(x / (k_b * temperature)) + 1.0)


def calculate_helmholtz_energy(temperature, efermi, energies, dos):
    internal_energy = calculate_internal_energy(temperature, efermi, energies, dos)
    entropy = calculate_entropy(temperature, efermi, energies, dos)
    return internal_energy - temperature * entropy


def calculate_internal_energy(temperature, efermi, energies, dos):
    """Difference of the internal energy from that at the zero temperature"""
    f0 = fermi_dirac_distribution(0.0, efermi, energies)
    f1 = fermi_dirac_distribution(temperature, efermi, energies)
    return np.trapz(dos * (f1 - f0) * energies, energies)


def calculate_entropy(temperature, efermi, energies, dos):
    if temperature == 0.0:
        return 0.0
    else:
        f = fermi_dirac_distribution(temperature, efermi, energies)
        return k_b * np.trapz(dos * s(f), energies)


@np.vectorize
def s(f):
    if f == 0.0 or f == 1.0:
        return 0.0
    else:
        return -(f * np.log(f) + (1.0 - f) * np.log(1.0 - f))
