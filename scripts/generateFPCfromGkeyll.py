import sys
import os
import math
import numpy as np

from FPCAnalysis import *

try:
    flnm_prefix = str(sys.argv[1])
except:
    print("This generates FPC netcdf4 file using Gkeyll data...")
    print("usage: " + sys.argv[0] + " flnm_prefix framenum species(ion by default)")
    sys.exit()

try:
    num = int(sys.argv[2])
except:
    print("Please enter frame number...")
    sys.exit()

try:
    species = sys.argv[3]
except:
    species = 'ion'

#TODO: make work for all (1D/2D/3D)(2V/3V) data

#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------
params = dgkl.get_input_params(flnm_prefix,num,species=species)
ddist = dgkl.load_dist(flnm_prefix,num,species=species)
dfields = dgkl.load_fields(flnm_prefix,num,species=species)
dflow = dgkl.load_flow(flnm_prefix,num,species=species)

#assign metadata that might not be there
if(not('thetaBn' in params.keys())):
    print("Error, thetaBn was not computed for this shock...")
if(not('MachAlfven' in params.keys())):
    print("Error, MachAlfven (of inflow) was not computed for this shock...")

#-------------------------------------------------------------------------------
# Compute Shock Velocity and Transform Frame
#-------------------------------------------------------------------------------
ratio = dfields['bz'][0,0,0]/dfields['bz'][0,0,-1] #dirty approx
print("Computed shock ratio (typically about 2.5):")
print(ratio)
vshock = -dflow['ux'][0,0,-1] #-1 * (Inflow velocity) /(ratio-1) (from Juno et al 2021)


print();print();print();print();
print("WARNING: computing in the frame of rest of the output (most likely the simulation rest frame) until frame transform is implemented for gkeyll data...")
print();print();print();print();

#TODO: shift particles and fields to be in same frame here!!!!
#Note: while these functions are written, we need to make they are self consistent units wise
#dfields = ft.lorentz_transform_vx_gkeyll(dfields,vshock)
#ddist = ft.shift_dist_gkeyll()

#-------------------------------------------------------------------------------
# Compute FPC
#-------------------------------------------------------------------------------
dx = ddist['hist_xx'][1]-ddist['hist_xx'][0]
vmax = np.max(ddist['vx'])
dfpc = fpc.compute_correlation_over_x_from_dist(ddist,dfields, vmax, dx, vshock, xlim=None, ylim=[0,1], zlim=[0,1],project=True)

#compute energization from correlations
dv = dfpc['vz'][1,0,0]-dfpc['vz'][0,0,0]
enerCEx = anl.compute_energization_over_x(dfpc['CExvxvy'],dv)
enerCEy = anl.compute_energization_over_x(dfpc['CExvxvy'],dv)
enerCEz = anl.compute_energization_over_x(dfpc['CExvxvy'],dv)

dnc.save2Vdata(dfpc['Histvxvy'],dfpc['Histvxvz'],dfpc['Histvyvz'],dfpc['CExvxvy'],dfpc['CExvxvz'],dfpc['CExvyvz'],dfpc['CEyvxvy'],dfpc['CEyvxvz'],dfpc['CEyvyvz'],dfpc['CEzvxvy'],dfpc['CEzvxvz'],dfpc['CEzvyvz'], dfpc['vx'], dfpc['vy'], dfpc['vz'], dfpc['xx'], enerCEx, enerCEy, enerCEz, dfields['Vframe_relative_to_sim'], metadata = [], params = params, filename = flnm_prefix+'_'+species+'_'+str(num)+'_nometadata.nc' )
print("Done! Please use findShock.py and addMetadata to assign metadata...")
