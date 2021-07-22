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
    print("This generates FPC netcdf4 file")
    print("usage: " + sys.argv[0] + " analysisinputflnm")
    sys.exit()


#-------------------------------------------------------------------------------
# load data
#-------------------------------------------------------------------------------
#load path
path,resultsdir,vmax,dv,numframe,dx,xlim,ylim,zlim = anl.analysis_input(flnm = analysisinputflnm)
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"

print("Loading data...")
#load relevant time slice fields
dfields = dh5.field_loader(path=path_fields,num=numframe)

#Load all fields along all time slices
all_dfields = dh5.all_dfield_loader(path=path_fields, verbose=False)

#Load slice of particle data
if xlim is not None and ylim is not None and zlim is not None:
    dparticles = dh5.read_box_of_particles(path_particles, numframe, xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1])
#Load only a slice in x but all of y and z
elif xlim is not None and ylim is None and zlim is None:
    dparticles = dh5.read_box_of_particles(path_particles, numframe, xlim[0], xlim[1], dfields['ex_yy'][0], dfields['ex_yy'][-1], dfields['ex_zz'][0], dfields['ex_zz'][-1])
#Load all the particles
else:
    dparticles = dh5.read_particles(path_particles, numframe)
#-------------------------------------------------------------------------------
# estimate shock vel and lorentz transform
#-------------------------------------------------------------------------------
print("Lorentz transforming fields...")
vshock,_ = ft.shock_from_ex_cross(all_dfields,dt=0.01)

#Lorentz transform fields
dfields = ft.lorentz_transform_vx(dfields,vshock)
_fields = []
for k in range(0,len(all_dfields['dfields'])):
    _fields.append(ft.lorentz_transform_vx(all_dfields['dfields'][k],vshock))
all_dfields['dfields'] = _fields

#-------------------------------------------------------------------------------
# do FPC analysis
#-------------------------------------------------------------------------------
print("Doing FPC analysis for each slice of x...")
if dx is None:
    #Assumes rectangular grid that is uniform for all fields
    #If dx not specified, just use the grid cell spacing for the EM fields
    dx = dfields['ex_xx'][1]-dfields['ex_xx'][0]
CEx, CEy, CEz, x, Hist, vx, vy, vz = fpc.compute_correlation_over_x(dfields, dparticles, vmax, dv, dx, vshock, xlim, ylim, zlim)

#-------------------------------------------------------------------------------
# compute energization
#-------------------------------------------------------------------------------
#for now, we project onto vx vy plane until integrating.
CEx_xy = []
CEy_xy = []
CEz_xy = []
for i in range(0,len(CEx)):
    CEx_xy = ao.array_3d_to_2d(CEx[i],'xy')
    CEy_xy = ao.array_3d_to_2d(CEy[i],'xy')
    CEz_xy = ao.array_3d_to_2d(CEz[i],'xy')
    CEx_xy.append(CEx_xy)
    CEy_xy.append(CEy_xy)
    CEz_xy.append(CEz_xy)

#compute energization from correlations
enerCEx = anl.compute_energization_over_x(CEx_xy,dv)
enerCEy = anl.compute_energization_over_x(CEy_xy,dv)
enerCEz = anl.compute_energization_over_x(CEz_xy,dv)

#-------------------------------------------------------------------------------
# Save data with relevant input parameters
#-------------------------------------------------------------------------------
print("Saving results in netcdf4 file...")
inputdict = dnc.parse_input_file(path)
params = dnc.build_params(inputdict,numframe)

flnm = 'FPCnometadata.nc'
dnc.save3vdata(Hist, CEx, CEy, CEz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, dfields['Vframe_relative_to_sim'], metadata_out = [], params = params, filename = resultsdir+flnm)
print("Done! Please use findShock.py and addMetadata to assign metadata...")
