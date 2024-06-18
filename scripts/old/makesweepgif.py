#!/usr/bin/env python

import sys
sys.path.append(".")
sys.path.append("..")

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
    path = sys.argv[1]

except:
    print("This script plots CEi and dist func for each slice of x for a given netcdf4 file")
    print("usage: " + sys.argv[0] + " path plotLog(default F)")
    sys.exit()

try:
    plotLog = sys.argv[2]
    if(plotLog == 'T'):
        plotLog = True
        print("Plotting with log scale!")
    else:
        plotLog = False
        print("Plotting with normal scale!")
except:
    plotLog = False #TODO: more input parsing
    print("Plotting with normal scale!")

#load data
try:
    (Hist_vxvy, Hist_vxvz, Hist_vyvz,
    C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
    C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
    C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
    vx, vy, vz, x_in,
    enerCEx_in, enerCEy_in, enerCEz_in,
    npar_in, Vframe_relative_to_sim_in, metadata_in, params_in) = dnc.load2vdata(path)
except:
    (Hist_vxvy, Hist_vxvz, Hist_vyvz,
    C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
    C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
    C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
    vx, vy, vz, x_in,
    enerCEx_in, enerCEy_in, enerCEz_in,
    Vframe_relative_to_sim_in, metadata_in, params_in) = dnc.load2vdata(path)

if(plotLog):
    directory = path+'.log9panelplot/'
else:
    directory = path+'.9panelplot/'

try:
    os.system('mkdir ' + directory)
except:
    pass
pltvv.make_9panel_sweep_from_2v(Hist_vxvy, Hist_vxvz, Hist_vyvz,
                                C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
                                C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
                                C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
                                vx, vy,vz,params_in,x_in,metadata_in,
                                directory,plotLog=plotLog)

#make gif from png
if(plotLog):
    outname = path+'.log9panelplot'
else:
    outname = path+'.9panelplot'
rsltmng.make_gif_from_folder(directory,outname+'.gif')
