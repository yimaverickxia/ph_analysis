#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from collections import OrderedDict
import numpy as np
from .eos import EOSFactory
from phonopy.qha.eos import EOSFit

__author__ = 'Yuji Ikeda'


def calculate_rmse(f, xdata, ydata, p):
    return np.sqrt(np.average((ydata - f(xdata, *p)) ** 2))


class EOSFitting(object):
    """

    At this moment, this is wrapper of EOSFit in PHONOPY
    to give a different initial guess of fitting parameters.
    """
    def __init__(self, volumes, energies, eos_name):
        self._volumes = np.array(volumes)
        self._energies = np.array(energies)
        self._eos_name = eos_name

    def fit(self) -> OrderedDict:
        volumes = self._volumes
        energies = self._energies
        eos = EOSFactory(self._eos_name).create().ev
        eos_fit = EOSFit(volumes, energies, eos=eos)
        imin = np.argmin(energies)
        # iloc assumes they are pandas.DataFrame.
        parameters_initial = [energies[imin], 1.0, 4.0, volumes[imin]]
        eos_fit.run(initial_parameter=parameters_initial)
        popt = eos_fit.get_parameters()

        fitting_error = calculate_rmse(eos, volumes, energies, popt)

        d = OrderedDict()
        d['F0'] = eos_fit.get_energy()
        d['V0'] = eos_fit.get_volume()
        d['B0'] = eos_fit.get_bulk_modulus()
        d['Bp0'] = eos_fit.get_b_prime()
        d['F_RMSE'] = fitting_error
        d['NV'] = len(volumes)

        return d

