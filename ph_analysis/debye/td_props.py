#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from functools import partial
import numpy as np
from scipy.constants import physical_constants
from scipy.integrate import quad

k_b = physical_constants['Boltzmann constant in eV/K'][0]


@np.vectorize
def function_debye(x, n):
    """Debye function https://en.wikipedia.org/wiki/Debye_function"""
    if x == 0.0:
        return 1.0
    else:
        return n / (x ** n) * quad(lambda t: t ** n / np.expm1(t), 0.0, x)[0]

function_debye3 = partial(function_debye, n=3)


@np.vectorize
def calculate_helmholtz_energy(temperature, debye):
    tmp0 = 9.0 * k_b * debye / 8.0
    if temperature == 0.0:
        tmp1 = 0.0
    else:
        x = debye / temperature
        tmp1 = k_b * temperature * (
            3.0 * np.log(-np.expm1(-x)) - function_debye3(x))
    free_energy = tmp0 + tmp1
    return free_energy


@np.vectorize
def calculate_internal_energy(temperature, debye):
    tmp0 = 9.0 * k_b * debye / 8.0
    if temperature == 0.0:
        tmp1 = 0.0
    else:
        x = debye / temperature
        tmp1 = 3.0 * k_b * temperature * function_debye3(x)
    internal_energy = tmp0 + tmp1
    return internal_energy


@np.vectorize
def calculate_entropy(temperature, debye):
    if temperature == 0.0:
        entropy = 0.0
    else:
        x = debye / temperature
        entropy = k_b * (4.0 * function_debye3(x) - 3.0 * np.log(-np.expm1(-x)))
    return entropy
