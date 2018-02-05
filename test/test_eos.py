#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import unittest
import numpy as np
from ph_analysis.eos.eos import EOSFactory

eos_names = [
    'Vinet',
    'BM2',
    # 'BM3',
    # 'Murnaghan',
]


class TestEOS(unittest.TestCase):
    def test(self):
        for name in eos_names:
            print(name)
            eos = EOSFactory(name).create()
            volumes = np.linspace(8.0, 12.0, 10001)
            d = volumes[1] - volumes[0]
            p = [0.0, 1.0, 4.0, 10.0]
            energies = eos.ev(volumes, *p)

            pressures_n = -np.gradient(energies, d)
            pressures_a = eos.pv(volumes, *p)
            np.testing.assert_almost_equal(pressures_n[1:-1], pressures_a[1:-1])

            bulk_moduli_n = -np.gradient(pressures_a, d) * volumes
            bulk_moduli_a = eos.bv(volumes, *p)
            np.testing.assert_almost_equal(bulk_moduli_a[1:-1], bulk_moduli_n[1:-1])


if __name__ == '__main__':
    unittest.main()
