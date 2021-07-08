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

#load fields
dfields = lf.field_loader(path=path_fields,num=numframe)
dx = dfields['ex_xx'][1]-dfields['ex_xx'][0] #assumes rectangular grid thats uniform for all fields

#load original netcdf4 file
CEx_in, CEy_in, vx_in, vy_in, x_in, enerCEx_in, enerCEy_in, _, params_in = svf.load_netcdf4(filename)

# #make gif with ExB for given time frame
pf.make_velsig_gif_with_EcrossB(vx_in, vy_in, vmax, CEx_in, 'ex', x_in, dx, dfields,  'CExFrame'+str(numframe)+'ExB', 'CExFrame'+str(numframe)+'ExB.gif')
pf.make_velsig_gif_with_EcrossB(vx_in, vy_in, vmax, CEy_in, 'ey', x_in, dx, dfields,  'CExFrame'+str(numframe)+'ExB', 'CExFrame'+str(numframe)+'ExB.gif')

#make gif from pngs
pf.make_gif_from_folder('CExFrame'+str(numframe)+'ExB', 'CExFrame'+str(numframe)+'ExB.gif')
pf.make_gif_from_folder('CEyFrame'+str(numframe)+'ExB', 'CEyFrame'+str(numframe)+'ExB.gif')
