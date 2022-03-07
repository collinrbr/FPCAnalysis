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
    analysisinputdir = sys.argv[1]

except:
    print("This script queues up projectncdata.py using all analysis inputs in specified folder to find output")
    print("usage: " + sys.argv[0] + " analysisinputdir")
    sys.exit()

filenames = os.listdir(analysisinputdir)
filenames = sorted(filenames)
try:
    filenames.remove('.DS_store')
except:
    pass

print("Input files that are queued up:")
print(filenames)

for flnm in filenames:
    analysisinputflnm = analysisinputdir+'/'+flnm
    path,resultsdir,vmax,dv,numframe,dx,xlim,ylim,zlim = anl.analysis_input(flnm = analysisinputflnm)

    projfile = resultsdir+'/FPCnometadata.nc'
    cmd = 'python3 scripts/projectncdata.py '+projfile
    print(cmd)
    #os.system(cmd)
