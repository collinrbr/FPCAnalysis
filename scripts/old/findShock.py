#!/usr/bin/env python

import sys
sys.path.append(".")

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
    startval = float(sys.argv[2])
    endval = float(sys.argv[3])

except:
    print("This generates a plot of the Ex(x,y=0,z=0) field with lines specified at the provided shock bounds (in units of di) and returns the xx index associated with the nearest element in the array to each bound.")
    print("Use these bounds with addMetadata to assign meta data")
    print("usage: " + sys.argv[0] + " analysisinputflnm startindex endindex")
    sys.exit()

#load path
path,resultsdir,vmax,dv,numframe,dx,xlim,ylim,zlim = anl.analysis_input(flnm = analysisinputflnm)
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"

#load relevant time slice fields
dfields = dh5.field_loader(path=path_fields,num=numframe)

#plots Ex field with lines at specified x pos prints indexes of shock bounds
yyindex = 0
zzindex = 0
plt1d.plot_field(dfields, 'ex', axis='_xx', yyindex = yyindex, zzindex = zzindex, axvx1 = startval, axvx2 = endval,flnm=resultsdir+'Ex(x)_shockbounds.png')
