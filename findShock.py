#!/usr/bin/env python
import lib.loadfunctions as lf
import lib.analysisfunctions as af
import lib.plotfunctions as pf
import lib.savefunctions as svf
import lib.sanityfunctions as sanf
import lib.fieldtransformfunctions as ftf
import sys

try:
    startval = float(sys.argv[1])
    endval = float(sys.argv[2])

except:
    print("This generates a plot of the Ex(x,y=0,z=0) field with lines specified at the provided shock bounds (in units of di) and returns the xx index associated with the nearest element in the array to each bound.")
    print("Use these bounds with addMetadata to assign meta data")
    print("usage: " + sys.argv[0] + " startindex endindex")
    sys.exit()

#load path
path,vmax,dv,numframe = lf.analysis_input()
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"

#load relevant time slice fields
dfields = lf.field_loader(path=path_fields,num=numframe)

#plots Ex field with lines at specified x pos prints indexes of shock bounds
yyindex = 0
zzindex = 0
pf.plot_field(dfields, 'ex', axis='_xx', yyindex = yyindex, zzindex = zzindex, axvx1 = startval, axvx2 = endval,flnm=path+'Ex(x)_shockbounds.png')
