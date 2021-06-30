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
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"
dfields = lf.field_loader(path=path_fields,num=numframe)

#load original netcdf4 file
CEx_in, CEy_in, vx_in, vy_in, x_in, enerCEx_in, enerCEy_in, _, params_in = svf.load_netcdf4(filename)

pf.make_velsig_gif_with_EcrossB(vx_in, vy_in, vmax, CEx_out, 'ex', x_out, dx, dfields, 'CExFrame'+numframe+'ExB', 'CExFrame'+numframe+'ExB.gif')
pf.make_velsig_gif_with_EcrossB(vx_in, vy_in, vmax, CEy_out, 'ey', x_out, dx, dfields, 'CEyFrame'+numframe+'ExB', 'CEyFrame'+numframe+'ExB.gif')
