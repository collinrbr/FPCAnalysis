#!/usr/bin/env python
import lib.loadfunctions as lf
import lib.analysisfunctions as af
import lib.plotfunctions as pf
import lib.savefunctions as svf
import lib.sanityfunctions as sanf
import lib.fieldtransformfunctions as ftf
import sys
import os

try:
    filename = sys.argv[1]

except:
    print("This script generates correlation sweep gifs as along xx")
    print("usage: " + sys.argv[0] + " netcdf4flnm")
    sys.exit()

#load relevant time slice fields
path,vmax,dv,numframe = lf.analysis_input()

#load original netcdf4 file
CEx_in, CEy_in, vx_in, vy_in, x_in, enerCEx_in, enerCEy_in, _, params_in = svf.load_netcdf4(filename)

#TODO: make this show ExB
pf.make_velsig_gif(vx_in, vy_in, vmax, CEx_in, 'ex', x_in, 'CExFrame'+str(numframe), 'CExFrame'+str(numframe)+'.gif')
pf.make_velsig_gif(vx_in, vy_in, vmax, CEy_in, 'ey', x_in, 'CEyFrame'+str(numframe), 'CEyFrame'+str(numframe)+'.gif')
