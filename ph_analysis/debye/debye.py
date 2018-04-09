#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scipy.constants import Boltzmann, atomic_mass, hbar, electron_volt


def f_nu(nu):
    tmp0 = (2.0 / 3.0) * ((1.0 + nu) / (1.0 - 2.0 * nu))
    tmp1 = (1.0 / 3.0) * ((1.0 + nu) / (1.0 - nu))
    tmp2 = (1.0 / 3.0 * (2.0 * tmp0 ** (3.0 / 2.0) + tmp1 ** (3.0 / 2.0)))

    return tmp2 ** (-1.0 / 3.0)


def inv_f_nu(f):
    from scipy.optimize import brentq
    return brentq(
        lambda x: f_nu(x) - f,
        a=-0.999, b=0.499,
        xtol=np.finfo(float).eps,
    )


class Debye(object):
    def __init__(self, poisson=0.25):
        self._poisson = poisson
        self._f_nu = f_nu(self._poisson)

    def run(self, volume, bulk_modulus, mass_in_da):
        """Calculate Debye temperature

        volume: Must be in A^{3}
        bulk_modulus: Must be in eV/A^{3}
        mass: Must be in Da
        """
        tmp0 = (6.0 * np.pi ** 2) ** (1.0 / 3.0)

        v_in_m3 = (volume * 1e-10 ** 3)  # m^{-3}
        tmp1 = v_in_m3 ** (1.0 / 6.0)

        b_in_pa = bulk_modulus * electron_volt / (1e-10 ** 3)  # Pa
        mass = mass_in_da * atomic_mass  # in kg
        tmp2 = np.sqrt(b_in_pa / mass)

        tmp3 = tmp0 * tmp1 * tmp2

        debye_temperature = hbar / Boltzmann * self._f_nu * tmp3

        return debye_temperature

    def get_poisson(self):
        return self._poisson


@np.vectorize
def calculate_debye_from_gruneisen(volume, v0, d0, gruneisen):
    debye = d0 * (v0 / volume) ** gruneisen
    return debye
