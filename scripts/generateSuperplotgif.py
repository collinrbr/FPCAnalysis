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
    filename = sys.argv[2]

except:
    print("This script makes superplot from netcdf4 file.")
    print("usage: " + sys.argv[0] + " analysisinputflnm netcdf4flnm")

#-------------------------------------------------------------------------------
# load data
#-------------------------------------------------------------------------------
#load path
path,resultsdir,vmax,dv,numframe,dx,xlim,ylim,zlim = anl.analysis_input(flnm = analysisinputflnm)
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"

#load original netcdf4 file
Hist, CEx, CEy, CEz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, Vframe_relative_to_sim, _, params_in = dnc.load3Vnetcdf4(filename)

#-------------------------------------------------------------------------------
# make superplot pngs
#-------------------------------------------------------------------------------
directory = resultsdir+"superplotGraphs"
flnm = resultsdir+"superplottest.gif"
pltvv.make_superplot_gif(vx, vy, vz, vmax, Hist, CEx, CEy, CEz, x, directory, flnm)

#-------------------------------------------------------------------------------
# make pngs into gif
#-------------------------------------------------------------------------------
rsltmng.make_gif_from_folder(directory,flnm)
