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

#load particle data
dparticles = lf.readParticlesPosandVelocityOnly(path_particles, numframe)

#Load all fields along all time slices
all_dfields = lf.all_dfield_loader(path=path_fields, verbose=False)

#-------------------------------------------------------------------------------
# estimate shock vel and lorentz transform
#-------------------------------------------------------------------------------
print("Lorentz transforming fields...")
vshock,_ = af.shock_from_ex_cross(all_fields,dt=0.01)

#Lorentz transform fields
dfields = ftf.lorentz_transform_vx(dfields,vx)
_fields = []
for k in range(0,len(all_dfields['dfields'])):
    _fields.append(ftf.lorentz_transform_vx(all_dfields['dfields'][k],vx))
all_dfields['dfields'] = _fields

#-------------------------------------------------------------------------------
# do FPC analysis
#-------------------------------------------------------------------------------
print("Doing FPC analysis for each slice of x...")
dx = dfields['ex_xx'][1]-dfields['ex_xx'][0] #assumes rectangular grid thats uniform for all fields
CEx_out, CEy_out, x_out, Hxy_out, vx_out, vy_out = af.compute_correlation_over_x(dfields, dparticles, vmax, dv, dx, vshock)

#-------------------------------------------------------------------------------
# Save data with relevant input parameters
#-------------------------------------------------------------------------------
print("Saving results in netcdf4 file...")
inputdict = svf.parse_input_file(path)
params = svf.build_params(inputdict,numframe)

flnm = 'FPCnometadata.nc'
svf.savedata(CEx_out, CEy_out, vx_out, vy_out, x_out, metadata_out = [], params = params, filename = flnm)
print("Done! Please use findShock.py and addMetadata to assign metadata.")
