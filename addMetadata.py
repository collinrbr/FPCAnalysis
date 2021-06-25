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
    startval = float(sys.argv[2])
    endval = float(sys.argv[3])

except:
    print("This script assigns metadata of 1 inbetween the bounds and 0 elsewhere")
    print("usage: " + sys.argv[0] + " netcdf4flnm startindex endindex")
    sys.exit()

#load relevant time slice fields
path,vmax,dv,numframe = lf.analysis_input()
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"
dfields = lf.field_loader(path=path_fields,num=numframe)

#build metadata
metadata = svf.build_metadata(dfields, startval, endval)

#load original netcdf4 file
CEx_in, CEy_in, vx_in, vy_in, x_in, _, params_in = svf.load_netcdf4(filename)

#make new file with updated metadata
svf.savedata(CEx_in, CEy_in, vx_in, vy_in, x_in, metadata_out = metadata, params = params_in, filename = filename+'.withmetadata')

#replace old file
os.system('rm '+filename)
os.system('mv '+filename+'.withmetadata '+filename)
