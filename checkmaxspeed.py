#!/usr/bin/env python
import lib.loadfunctions as lf
import lib.analysisfunctions as af
import lib.plotfunctions as pf
import lib.savefunctions as svf
import lib.sanityfunctions as sanf
import lib.fieldtransformfunctions as ftf

#-------------------------------------------------------------------------------
# load data
#-------------------------------------------------------------------------------
#load path
path,vmax,dv,numframe = lf.analysis_input()
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"

#Load slice of particle data
dparticles = lf.readParticles(path_particles, numframe)

#-------------------------------------------------------------------------------
# check max speed
#-------------------------------------------------------------------------------
maxsx, maxsy, maxsz = sanf.get_abs_max_velocity(dparticles)

print("Maxsx : " + str(maxsx))
print("Maxsy : " + str(maxsy))
print("Maxsz : " + str(maxsz))
