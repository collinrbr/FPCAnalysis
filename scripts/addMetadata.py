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
    analysisinputflnm = sys.argv[1]
    filename = sys.argv[2]
    startval = float(sys.argv[3])
    endval = float(sys.argv[4])

except:
    print("This script assigns metadata of 1 inbetween the bounds and 0 elsewhere")
    print("usage: " + sys.argv[0] + " analysisinputflnm netcdf4flnm startval endval")
    sys.exit()

#load relevant time slice fields
path,resultsdir,vmax,dv,numframe,dx,xlim,ylim,zlim = anl.analysis_input(flnm = analysisinputflnm)
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"
dfields = dh5.field_loader(path=path_fields,num=numframe)

if xlim is None:
    xlim = [dfields['ex_xx'][0],dfields['ex_xx'][-1]]

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
