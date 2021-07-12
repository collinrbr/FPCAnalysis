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
    fieldkey = str(sys.argv[1])
    planename = str(sys.argv[2])

except:
    print("This script pmesh field sweep gifs along axis normal to given plane")
    print("usage: " + sys.argv[0] + " fieldkey planename (xplotlimmin xplotlimmax)")
    sys.exit()

#load relevant time slice fields
path,vmax,dv,numframe,dx,_,_,_ = ao.analysis_input()
path_fields = path

#load fields
dfields = dh5.field_loader(path=path_fields,num=numframe)

#make gif from pngs
try:
    xplotlimmin = float(sys.argv[3])
    xplotlimmax = float(sys.argv[4])
    directory = fieldkey+planename+'zoomedin_frame'+str(numframe)
    plt2d.make_fieldpmesh_sweep(dfields,fieldkey,planename,directory,xlimmin=xplotlimmin,xlimmax=xplotlimmax)

except:
    directory = fieldkey+planename+'frame'+str(numframe)
    plt2d.make_fieldpmesh_sweep(dfields,fieldkey,planename,directory)

rsltmng.make_gif_from_folder(directory,directory+'.gif')
