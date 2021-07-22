#!/usr/bin/env python

import sys
sys.path.append(".")

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
    filename = sys.argv[1]

except:
    print("This script generates correlation sweep gifs as along xx")
    print("usage: " + sys.argv[0] + " netcdf4flnm")
    sys.exit()

#load relevant time slice fields
path,vmax,dv,numframe,dx,xlim,ylim,zlim = anl.analysis_input()

#load original netcdf4 file
CEx_in, CEy_in, vx_in, vy_in, x_in, enerCEx_in, enerCEy_in, Vframe_relative_to_sim, _, params_in = dnc.load_netcdf4(filename)

#TODO: make this show ExB
pltvv.make_velsig_gif(vx_in, vy_in, vmax, CEx_in, 'ex', x_in, 'CExFrame'+str(numframe), 'CExFrame'+str(numframe)+'.gif')
pltvv.make_velsig_gif(vx_in, vy_in, vmax, CEy_in, 'ey', x_in, 'CEyFrame'+str(numframe), 'CEyFrame'+str(numframe)+'.gif')
