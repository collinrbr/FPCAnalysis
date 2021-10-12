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
except:
    print("This prints max speed in provided limits in analysis input")
    print("usage: " + sys.argv[0] + " analysisinputflnm")
    sys.exit()

#-------------------------------------------------------------------------------
# load data
#-------------------------------------------------------------------------------
#load path
path,resultsdir,vmax,dv,numframe,dx,xlim,ylim,zlim = anl.analysis_input(flnm = analysisinputflnm)
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"

#Load particle data
dparticles = dh5.readParticles(path_particles, numframe)

#-------------------------------------------------------------------------------
# check max speed
#-------------------------------------------------------------------------------
maxsx, maxsy, maxsz = anl.get_abs_max_velocity(dparticles)

print("Maxsx : " + str(maxsx))
print("Maxsy : " + str(maxsy))
print("Maxsz : " + str(maxsz))
