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

#-------------------------------------------------------------------------------
# load data
#-------------------------------------------------------------------------------
#load path
path,vmax,dv,numframe,dx,xlim,ylim,zlim = ao.analysis_input()
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"

print("Loading data...")
#load relevant time slice fields
dfields = dh5.field_loader(path=path_fields,num=numframe)

#Load all fields along all time slices
all_dfields = dh5.all_dfield_loader(path=path_fields, verbose=False)

#Load slice of particle data
dparticles = dh5.read_box_of_particles(path_particles, numframe, xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1])

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
#Define parameters related to analysis
dx = dfields['ex_xx'][1]-dfields['ex_xx'][0] #assumes rectangular grid thats uniform for all fields
CEx, CEy, CEz, x, Hist, vx, vy, vz = fpc.compute_correlation_over_x(dfields, dparticles, vmax, dv, dx, vshock)

#-------------------------------------------------------------------------------
# make superplot pngs
#-------------------------------------------------------------------------------
directory = "superplotGraphs"
flnm = "superplottest.gif"
pltvv.make_superplot_gif(vx, vy, vz, vmax, Hist, CEx, CEy, CEz, x, directory, flnm)

#-------------------------------------------------------------------------------
# make pngs into gif
#-------------------------------------------------------------------------------
rsltmng.make_gif_from_folder(directory,flnm)
