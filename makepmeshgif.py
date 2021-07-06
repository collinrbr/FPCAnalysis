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
    fieldkey = str(sys.argv[1])
    planename = str(sys.argv[2])

except:
    print("This script pmesh field sweep gifs along axis normal to given plane")
    print("usage: " + sys.argv[0] + " fieldkey planename")
    sys.exit()

#load relevant time slice fields
path,vmax,dv,numframe = lf.analysis_input()
path_fields = path

#load fields
dfields = lf.field_loader(path=path_fields,num=numframe)

# #make gif with ExB for given time frame
# pf.make_velsig_gif_with_EcrossB(vx_in, vy_in, vmax, CEx_in, 'ex', x_in, dx, dfields,  'CExFrame'+str(numframe)+'ExB', 'CExFrame'+str(numframe)+'ExB.gif')
# pf.make_velsig_gif_with_EcrossB(vx_in, vy_in, vmax, CEy_in, 'ey', x_in, dx, dfields,  'CExFrame'+str(numframe)+'ExB', 'CExFrame'+str(numframe)+'ExB.gif')

#make gif from pngs
directory = fieldkey+planename
pf.make_fieldpmesh_sweep(dfields,fieldkey,planename,directory)
pf.make_gif_from_folder(directory,directory+'.gif')
