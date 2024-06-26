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
    print("This script queues up preslicedata.py on all analysis inputs in specified folder")
    print("usage: " + sys.argv[0] + " analysisinputdir outmasterdir")
    sys.exit()

outmasterdir = sys.argv[2]

filenames = os.listdir(analysisinputdir)
filenames = sorted(filenames)
try:
    filenames.remove('.DS_store')
except:
    pass

print("Files that are queued up:")
print(filenames)

for flnm in filenames:
    temp =flnm.split('.')
    cmd = 'mkdir '+outmasterdir+'/'+temp[0]
    print(cmd)
    exitval = os.system(cmd)
    cmd = 'touch '+outmasterdir+'/'+temp[0]+'.output'
    print(cmd)
    os.system(cmd)
    cmd = 'python3 scripts/preslicedata.py ' +analysisinputdir+'/'+flnm+' '+outmasterdir+'/'+temp[0]+' T >> '+outmasterdir+'/'+temp[0]+'.output'
    print(cmd)
    exitval = os.system(cmd)

    if(exitval != 0):
        print("Couldn't find restart files, trying standard files")
        cmd = 'python3 scripts/preslicedata.py ' +analysisinputdir+'/'+flnm+' '+outmasterdir+'/'+temp[0]+' F >> '+outmasterdir+'/'+temp[0]+'.output'
        print(cmd)
        exitval = os.system(cmd)
