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
path,vmax,dv,numframe,dx,xlim,ylim,zlim = lf.analysis_input()
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"

print("Loading data...")
#load relevant time slice fields
dfields = lf.field_loader(path=path_fields,num=numframe)

#Load all fields along all time slices
all_dfields = lf.all_dfield_loader(path=path_fields, verbose=False)

#Load slice of particle data
if xlim is not None and ylim is not None and zlim is not None:
    dparticles = lf.readSliceOfParticles(path_particles, numframe, xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1])
#Load only a slice in x but all of y and z
elif xlim is not None and ylim is None and zlim is None:
    dparticles = lf.readSliceOfParticles(path_particles, numframe, xlim[0], xlim[1], dfields['ex_yy'][0], dfields['ex_yy'][-1], dfields['ex_zz'][0], dfields['ex_zz'][-1])
#Load all the particles
else:
    dparticles = lf.readParticles(path_particles, numframe)
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
print("Doing FPC analysis for each slice of x...")
if dx is None:
    #Assumes rectangular grid that is uniform for all fields
    #If dx not specified, just use the grid cell spacing for the EM fields
    dx = dfields['ex_xx'][1]-dfields['ex_xx'][0]
CEx, CEy, CEz, x, Hist, vx, vy, vz = af.compute_correlation_over_x(dfields, dparticles, vmax, dv, dx, vshock, xlim, ylim, zlim)

#-------------------------------------------------------------------------------
# Convert to old format
#-------------------------------------------------------------------------------
#for now, we just do CEx_xy CEy_xy
#Here we convert to the previous 2d format
#TODO: this takes a minute, probably only want to project once
CEx_out = []
CEy_out = []
for i in range(0,len(CEx)):
    CEx_xy = af.threeCorToTwoCor(CEx[i],'xy')
    CEy_xy = af.threeCorToTwoCor(CEy[i],'xy')
    CEx_out.append(CEx_xy)
    CEy_out.append(CEy_xy)
vx_xy, vy_xy = af.threeVelToTwoVel(vx,vy,vz,'xy')
vx_out = vx_xy
vy_out = vy_xy
x_out = x


#compute energization from correlations
enerCEx_out = af.compute_energization_over_x(CEx_out,dv)
enerCEy_out = af.compute_energization_over_x(CEy_out,dv)

#-------------------------------------------------------------------------------
# Save data with relevant input parameters
#-------------------------------------------------------------------------------
print("Saving results in netcdf4 file...")
inputdict = svf.parse_input_file(path)
params = svf.build_params(inputdict,numframe)

flnm = 'FPCnometadata.nc'
svf.savedata(CEx_out, CEy_out, vx_out, vy_out, x_out, enerCEx_out, enerCEy_out, metadata_out = [], params = params, filename = flnm)
print("Done! Please use findShock.py and addMetadata to assign metadata.")
