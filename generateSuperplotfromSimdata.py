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

print("Loading data...")
#load relevant time slice fields
dfields = lf.field_loader(path=path_fields,num=numframe)

#Load all fields along all time slices
all_dfields = lf.all_dfield_loader(path=path_fields, verbose=False)

#Load slice of particle data
dparticles = lf.readSliceOfParticles(path_particles, numframe, dfields['ex_xx'][0], dfields['ex_xx'][-1], dfields['ex_yy'][0], dfields['ex_yy'][1], dfields['ex_zz'][0], dfields['ex_zz'][1])

#-------------------------------------------------------------------------------
# estimate shock vel and lorentz transform
#-------------------------------------------------------------------------------
print("Lorentz transforming fields...")
vshock,_ = af.shock_from_ex_cross(all_dfields,dt=0.01)

#Lorentz transform fields
dfields = ftf.lorentz_transform_vx(dfields,vshock)
_fields = []
for k in range(0,len(all_dfields['dfields'])):
    _fields.append(ftf.lorentz_transform_vx(all_dfields['dfields'][k],vshock))
all_dfields['dfields'] = _fields

#-------------------------------------------------------------------------------
# do FPC analysis
#-------------------------------------------------------------------------------
#Define parameters related to analysis
dx = dfields['ex_xx'][1]-dfields['ex_xx'][0] #assumes rectangular grid thats uniform for all fields
CEx, CEy, CEz, x, Hist, vx, vy, vz = af.compute_correlation_over_x(dfields, dparticles, vmax, dv, dx, vshock)

#-------------------------------------------------------------------------------
# make superplot
#-------------------------------------------------------------------------------
pf.make_superplot_gif(vx, vy, vz, vmax, Hist, CEx, CEy, CEz, x, 'superplotGraphs', 'superplottest.gif')
