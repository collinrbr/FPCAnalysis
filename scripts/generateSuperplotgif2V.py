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
    print("This script makes superplot from 2V netcdf4 file.")
    print("usage: " + sys.argv[0] + " analysisinputflnm netcdf4flnm")

#-------------------------------------------------------------------------------
# load data
#-------------------------------------------------------------------------------
#load path
path,resultsdir,vmax,dv,numframe,dx,xlim,ylim,zlim = anl.analysis_input(flnm = analysisinputflnm)
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"

#load original netcdf4 file
try:
    (Hist_vxvy, Hist_vxvz, Hist_vyvz,
       C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
       C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
       C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
       vx, vy, vz, x_in,
       enerCEx_in, enerCEy_in, enerCEz_in,
       Vframe_relative_to_sim_in, metadata_in, params_in) = dnc.load2vdata(filename)
except:
    #load newer netcdf4 file
    (Hist_vxvy, Hist_vxvz, Hist_vyvz,
        C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
        C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
        C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
        vx, vy, vz, x_in,
        enerCEx_in, enerCEy_in, enerCEz_in,
        npar_in, Vframe_relative_to_sim_in, metadata_in, params_in) = dnc.load2vdata(filename)

#-------------------------------------------------------------------------------
# make superplot pngs
#-------------------------------------------------------------------------------
directory = resultsdir+"superplotGraphs"
flnm = resultsdir+"superplottest.gif"
pltvv.make_9panel_sweep_from_2v(Hist_vxvy, Hist_vxvz, Hist_vyvz,
                              C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
                              C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
                              C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
                              vx, vy,vz,params_in,x_in,metadata_in,
                              directory,plotLog=False)

#-------------------------------------------------------------------------------
# make pngs into gif
#-------------------------------------------------------------------------------
rsltmng.make_gif_from_folder(directory,flnm)
