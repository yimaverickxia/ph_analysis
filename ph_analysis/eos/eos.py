#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from phonopy.qha.eos import get_eos

__author__ = 'Yuji Ikeda'


class EOS(object):
    """

    p[0] = E_0
    p[1] = B_0
    p[2] = B'_0
    p[3] = V_0
    """
    @staticmethod
    def ev(volume, *p):
        raise NotImplementedError

    @staticmethod
    def pv(volume, *p):
        raise NotImplementedError

    @staticmethod
    def bv(volume, *p):
        raise NotImplementedError


class EOSVinet(EOS):
    @staticmethod
    def ev(volume, *p):
        return get_eos('vinet')(volume, *p)

    @staticmethod
    def pv(volume, *p):
        x = (volume / p[3]) ** (1.0 / 3)
        xi = 3.0 / 2 * (p[2] - 1)
        return 3 * p[1] / (x ** 2) * (1 - x) * np.exp(xi * (1 - x))

    @staticmethod
    def bv(volume, *p):
        x = (volume / p[3]) ** (1.0 / 3)
        xi = 3.0 / 2 * (p[2] - 1)
        return p[1] * ((2 - x) / (x ** 2) + xi * (1 - x) / x) * np.exp(xi * (1 - x))


class EOSBM2(EOS):
    @staticmethod
    def ev(volume, *p):
        x = (p[3] / volume)
        return p[0] + (9.0 / 8.0) * p[1] * p[3] * (x ** (2.0 / 3.0) - 1.0) ** 2

    @staticmethod
    def pv(volume, *p):
        x = (p[3] / volume)
        return (3.0 / 2.0) * p[1] * (x ** (7.0 / 3.0) - x ** (5.0 / 3.0))

    @staticmethod
    def bv(volume, *p):
        x = (p[3] / volume)
        return p[1] * (7.0 / 2.0 * x ** (7.0 / 3.0) - 5.0 / 2.0 * x ** (5.0 / 3.0))


class EOSBM3(EOS):
    @staticmethod
    def ev(volume, *p):
        return get_eos('birch_murnaghan')(volume, *p)

    @staticmethod
    def pv(volume, *p):
        x = (p[3] / volume)
        f = (x ** (2. / 3.) - 1.) * 0.5
        c3 = 9. / 2. * p[1] * (p[2] - 4.)
        return EOSBM2.pv(volume, *p) + c3 * f ** 2 * x ** (5. / 3.)

    @staticmethod
    def bv(volume, *p):
        x = (p[3] / volume)
        f = (x ** (2. / 3.) - 1.) * 0.5
        c3 = 9. / 2. * p[1] * (p[2] - 4.)
        return EOSBM2.bv(volume, *p) + c3 / 3. * f * (9. * f + 2.) * x ** (5. / 3.)


class EOSMurnaghan(EOS):
    @staticmethod
    def ev(volume, *p):
        return get_eos('murnaghan')(volume, *p)


class EOSFactory(object):
    def __init__(self, name: str):
        self._name = name

    def create(self) -> EOS:
        name = self._name
        if name == 'Vinet':
            return EOSVinet()
        elif name == 'BM2':
            return EOSBM2()
        elif name == 'BM3':
            return EOSBM3()
        elif name == 'Murnaghan':
            return EOSMurnaghan()
        else:
            raise ValueError(name)
