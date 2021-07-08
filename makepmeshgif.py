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
    print("usage: " + sys.argv[0] + " fieldkey planename (xplotlimmin xplotlimmax)")
    sys.exit()

#load relevant time slice fields
path,vmax,dv,numframe,dx,_,_,_ = lf.analysis_input()
path_fields = path

#load fields
dfields = lf.field_loader(path=path_fields,num=numframe)

#make gif from pngs
try:
    xplotlimmin = float(sys.argv[3])
    xplotlimmax = float(sys.argv[4])
    directory = fieldkey+planename+'zoomedin_frame'+str(numframe)
    pf.make_fieldpmesh_sweep(dfields,fieldkey,planename,directory,xlimmin=xplotlimmin,xlimmax=xplotlimmax)

except:
    directory = fieldkey+planename+'frame'+str(numframe)
    pf.make_fieldpmesh_sweep(dfields,fieldkey,planename,directory)

pf.make_gif_from_folder(directory,directory+'.gif')
