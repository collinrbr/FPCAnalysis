#!/usr/bin/env python

import sys
sys.path.append(".")
sys.path.append('..')

import lib.analysis as anl
import lib.array_ops as ao
import lib.data_h5 as dh5
import lib.data_netcdf4 as dnc
import lib.fpc as fpc
import lib.frametransform as ft
import lib.metadata as md

import lib.plot.oned as plt1d
import lib.plot.twod as plt2d
import lib.plot.debug as pltdebug
import lib.plot.fourier as pltfr
import lib.plot.resultsmanager as rsltmng
import lib.plot.velspace as pltvv

import os
import math
import numpy as np
try:
    filename = sys.argv[1]
    startval = float(sys.argv[2])
    endval = float(sys.argv[3])

except:
    print("This script assigns metadata of 1 inbetween the bounds and 0 elsewhere")
    print("usage: " + sys.argv[0] + " netcdf4flnm startval endval")
    sys.exit()

#build metadata
metadata = md.build_metadata(xlim, dx, startval, endval)

#append metadata
from netCDF4 import Dataset
ncout = Dataset(filename, 'r+', format='NETCDF4')
try:
    sda = ncout.createVariable('sda','f4',('nx',))
    sda.description = '1 = signature, 0 = no signature'
    sda[:] = metadata[:]
except:
    sda = ncout.variables['npar']
    sda[:] = metadata[:]

ncout.close()
