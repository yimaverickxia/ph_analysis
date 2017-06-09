#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np

__author__ = 'Yuji Ikeda'


class FCReorderer(object):
    @staticmethod
    def reorder_fcs(fc, indices):
        fc_reordered = np.full_like(fc, np.nan)  # Initialized by np.nan to detect possible errors
        for i1, j1 in enumerate(indices):
            for i2, j2 in enumerate(indices):
                fc_reordered[j1, j2] = fc[i1, i2]
        return fc_reordered
