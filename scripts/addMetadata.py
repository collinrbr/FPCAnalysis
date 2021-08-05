#!/usr/bin/env python

import sys
sys.path.append(".")
sys.path.append('..')

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
    startval = float(sys.argv[3])
    endval = float(sys.argv[4])

except:
    print("This script assigns metadata of 1 inbetween the bounds and 0 elsewhere")
    print("usage: " + sys.argv[0] + " analysisinputflnm netcdf4flnm startval endval")
    sys.exit()

#load relevant time slice fields
path,resultsdir,vmax,dv,numframe,dx,xlim,ylim,zlim = anl.analysis_input(flnm = analysisinputflnm)
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"
dfields = dh5.field_loader(path=path_fields,num=numframe)

if xlim is None:
    xlim = [dfields['ex_xx'][0],dfields['ex_xx'][-1]]

#build metadata
metadata = md.build_metadata(xlim, dx, startval, endval)

#load original netcdf4 file
Hist, CEx, CEy, CEz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, Vframe_relative_to_sim, _, params_in = dnc.load3Vnetcdf4(filename)

#make new file with updated metadata
dnc.save3Vdata(Hist, CEx, CEy, CEz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, Vframe_relative_to_sim, metadata_out = metadata, params = params_in, filename = filename+'.withmetadata')

#replace old file
os.system('rm '+filename)
os.system('mv '+filename+'.withmetadata '+filename)
