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
    analysisinputflnm = sys.argv[1]
    filename = sys.argv[2]

except:
    print("This script generates correlation sweep gifs as along xx")
    print("usage: " + sys.argv[0] + " analysisinputflnm netcdf4flnm")
    sys.exit()

#load relevant time slice fields
path,resultsdir,vmax,dv,numframe,dx,xlim,ylim,zlim = anl.analysis_input(flnm = analysisinputflnm)
dfields = dh5.field_loader(path=path,num=numframe)

#load original netcdf4 file
Hist, CEx, CEy, CEz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, Vframe_relative_to_sim, _, params_in = dnc.load3Vnetcdf4(filename)

# #TODO: make this show ExB
# pltvv.make_velsig_gif(vx_in, vy_in, vmax, CEx_in, 'ex', x_in, 'CExFrame'+str(numframe), 'CExFrame'+str(numframe)+'.gif')
# pltvv.make_velsig_gif(vx_in, vy_in, vmax, CEy_in, 'ey', x_in, 'CEyFrame'+str(numframe), 'CEyFrame'+str(numframe)+'.gif')

CEx_xy = [ao.array_3d_to_2d(CEx[i],'xy') for i in range(0,len(CEx))]
vx_xy, vy_xy = ao.mesh_3d_to_2d(vx,vy,vz,'xy')
pltvv.make_velsig_gif_with_EcrossB(vx_xy, vy_xy, vmax, CEx_xy, 'ex', x, dx, dfields, resultsdir+'CExExB', resultsdir+'CExExB.gif', xlim = xlim, ylim = ylim, zlim = zlim)
rsltmng.make_gif_from_folder(resultsdir+'CExExB', resultsdir+'CExExB.gif')

CEy_xy = [ao.array_3d_to_2d(CEy[i],'xy') for i in range(0,len(CEy))]
vx_xy, vy_xy = ao.mesh_3d_to_2d(vx,vy,vz,'xy')
pltvv.make_velsig_gif_with_EcrossB(vx_xy, vy_xy, vmax, CEy_xy, 'ey', x, dx, dfields, resultsdir+'CEyExB', resultsdir+'CEyExB.gif', xlim = xlim, ylim = ylim, zlim = zlim)
rsltmng.make_gif_from_folder(resultsdir+'CEyExB', resultsdir+'CEyExB.gif')
