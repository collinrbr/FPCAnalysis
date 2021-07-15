#!/usr/bin/env python
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

#-------------------------------------------------------------------------------
# load data
#-------------------------------------------------------------------------------
#load path
path,vmax,dv,numframe,dx,xlim,ylim,zlim = anl.analysis_input()
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"

#Load particle data
dparticles = dh5.readParticles(path_particles, numframe)

#-------------------------------------------------------------------------------
# check max speed
#-------------------------------------------------------------------------------
maxsx, maxsy, maxsz = ao.get_abs_max_velocity(dparticles)

print("Maxsx : " + str(maxsx))
print("Maxsy : " + str(maxsy))
print("Maxsz : " + str(maxsz))
