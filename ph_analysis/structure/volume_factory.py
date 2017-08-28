#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from .volume_mesh import VolumeMesh
from .volume_voronoi import VolumeVoronoi

__author__ = 'Yuji Ikeda'
__version__ = '0.1.0'


class VolumeFactory(object):
    def create(self, method, *args, **kwargs):
        if method == 'mesh':
            return VolumeMesh(*args, **kwargs)
        elif method == 'voronoi':
            return VolumeVoronoi(*args, **kwargs)
        else:
            raise ValueError('Unknown method', method)
