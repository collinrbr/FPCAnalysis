#!/usr/bin/env python
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
    startval = float(sys.argv[2])
    endval = float(sys.argv[3])

except:
    print("This script assigns metadata of 1 inbetween the bounds and 0 elsewhere")
    print("usage: " + sys.argv[0] + " netcdf4flnm startindex endindex")
    sys.exit()

#load relevant time slice fields
path,vmax,dv,numframe,dx,xlim,ylim,zlim = anl.analysis_input()
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"
dfields = dh5.field_loader(path=path_fields,num=numframe)

#build metadata
metadata = md.build_metadata(dfields, startval, endval)

#load original netcdf4 file
CEx_in, CEy_in, vx_in, vy_in, x_in, enerCEx_in, enerCEy_in, _, params_in = dnc.load_netcdf4(filename)

#make new file with updated metadata
dnc.savedata(CEx_in, CEy_in, vx_in, vy_in, x_in, enerCEx_in, enerCEy_in, metadata_out = metadata, params = params_in, filename = filename+'.withmetadata')

#replace old file
os.system('rm '+filename)
os.system('mv '+filename+'.withmetadata '+filename)
