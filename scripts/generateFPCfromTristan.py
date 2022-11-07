#!/usr/bin/env python

import sys
sys.path.append(".")

import lib.analysis as anl
import lib.array_ops as ao
import lib.data_tristan as dtr
import lib.data_netcdf4 as dnc
import lib.fpc as fpc
import lib.frametransform as ft
import lib.metadata as md

import os
import math
import numpy as np
try:
    path = sys.argv[1]
    num = sys.argv[2]
    vshock = float(sys.argv[3])
    vmax = float(sys.argv[4])
    dv = float(sys.argv[5])
except:
    print("This generates FPC netcdf4 file from Tristan data")
    print("usage: " + sys.argv[0] + " path framenum vshock vmax dv useflucfields(default F)")
    print("Here, vshock is the velocity of the shock relative to the frame of the data in units of v/vti")
    print("Warning: Frame num expects leading zeros.")
    sys.exit()

try:
    usedfluc = sys.argv[6]
    if(usedfluc == 'T'):
        usedfluc = True
        print("Using fluctuating fields!")
    else:
        usedfluc = False
        print("Using total fields!")
except:
    usedfluc = False
    print("Using total fields!")


print('path: ', path)
print('framenum: ', num)
print('vshock: ', vshock)
print('vmax: ', vmax)
print('dv: ', dv)

#-------------------------------------------------------------------------------
# load data
#-------------------------------------------------------------------------------
#load path
print("Loading data...")
params = dtr.load_params(path,num)
dfields = dtr.load_fields(path,num,normalizeFields=True)

for key in 'ex ey ez bx by bz'.split(): #our functions assume 3D fields. When the fields are 2D, we load the 2D data into two identitical sheets to mimic the 3D data structure
    if(not(key+'_zz' in dfields.keys())):
        dfields[key+'_zz'] = np.asarray([0,1])
    if(not(key+'_yy' in dfields.keys())):
        print(key)
        dfields[key+'_yy'] = np.asarray([0,1])
dpar_elec, dpar_ion = dtr.load_particles(path,num,normalizeVelocity=True) #we like to normalize to each species thermal velocity in the upstream. This function will attempt to calculate this use moments of the dist. It's not perfect, but it works for now
dpar_ion = dtr.format_par_like_dHybridR(dpar_ion) #For now, we rename the particle data keys too look like the keys we used when processing dHybridR data so this data is compatible with our old routines
dpar_elec = dtr.format_par_like_dHybridR(dpar_elec)

print("Computing relevant parameters...")
_dfields = dtr.load_fields(path, num)
_dfields = anl.get_average_fields_over_yz(dfields)
params['thetaBn'] = np.arctan(_dfields['by'][0,0,-1]/_dfields['bx'][0,0,-1])* 180.0/np.pi

upperxbound = np.max(dpar_ion['x1'])
lowerxbound = upperxbound*.95 #WARNING: assumes that the beam is undisturbed in this region
gdpts = (dpar_ion['x1'] > lowerxbound)
params['MachAlfven'] = np.mean(dpar_ion['p1'][gdpts][:])-vshock #inflow mach alfven (assumes beta = 1) #TODO: stop using this assumption everywhere in this library

#-------------------------------------------------------------------------------
# estimate shock vel and lorentz transform
#-------------------------------------------------------------------------------
print("Lorentz transforming fields...")
#Lorentz transform fields
dfields = ft.lorentz_transform_vx(dfields,vshock)

if(usedfluc):
    dfields = anl.remove_average_fields_over_yz(dfields)

#-------------------------------------------------------------------------------
# do FPC analysis for ions and project output
#-------------------------------------------------------------------------------
print("Doing FPC analysis for each slice of x for ions...")
dx = dfields['ex_xx'][150]-dfields['ex_xx'][0]
xlim = [np.min(dfields['ex_xx']),np.max(dfields['ex_xx'])]
ylim = [np.min(dfields['ex_yy']),np.max(dfields['ex_yy'])]
zlim = [np.min(dfields['ex_zz']),np.max(dfields['ex_zz'])]
CEx, CEy, CEz, x, Hist, vx, vy, vz, num_par = fpc.compute_correlation_over_x(dfields, dpar_ion, vmax, dv, dx, vshock, xlim, ylim, zlim)

Histxy = []
Histxz = []
Histyz = []
for i in range(0,len(Hist)):
    tempHistxy = ao.array_3d_to_2d(Hist[i],'xy')
    tempHistxz = ao.array_3d_to_2d(Hist[i],'xz')
    tempHistyz = ao.array_3d_to_2d(Hist[i],'yz')
    Histxy.append(tempHistxy)
    Histxz.append(tempHistxz)
    Histyz.append(tempHistyz)
Histxy = np.asarray(Histxy)
Histxz = np.asarray(Histxz)
Histyz = np.asarray(Histyz)

CExxy = []
CExxz = []
CExyz = []
for i in range(0,len(Hist)):
    tempCExxy = ao.array_3d_to_2d(CEx[i],'xy')
    tempCExxz = ao.array_3d_to_2d(CEx[i],'xz')
    tempCExyz = ao.array_3d_to_2d(CEx[i],'yz')
    CExxy.append(tempCExxy)
    CExxz.append(tempCExxz)
    CExyz.append(tempCExyz)
CExxy = np.asarray(CExxy)
CExxz = np.asarray(CExxz)
CExyz = np.asarray(CExyz)

CEyxy = []
CEyxz = []
CEyyz = []
for i in range(0,len(Hist)):
    tempCEyxy = ao.array_3d_to_2d(CEy[i],'xy')
    tempCEyxz = ao.array_3d_to_2d(CEy[i],'xz')
    tempCEyyz = ao.array_3d_to_2d(CEy[i],'yz')
    CEyxy.append(tempCEyxy)
    CEyxz.append(tempCEyxz)
    CEyyz.append(tempCEyyz)
CEyxy = np.asarray(CEyxy)
CEyxz = np.asarray(CEyxz)
CEyyz = np.asarray(CEyyz)

CEzxy = []
CEzxz = []
CEzyz = []
for i in range(0,len(Hist)):
    tempCEzxy = ao.array_3d_to_2d(CEz[i],'xy')
    tempCEzxz = ao.array_3d_to_2d(CEz[i],'xz')
    tempCEzyz = ao.array_3d_to_2d(CEz[i],'yz')
    CEzxy.append(tempCEzxy)
    CEzxz.append(tempCEzxz)
    CEzyz.append(tempCEzyz)
CEzxy = np.asarray(CEzxy)
CEzxz = np.asarray(CEzxz)
CEzyz = np.asarray(CEzyz)



#-------------------------------------------------------------------------------
# compute energization
#-------------------------------------------------------------------------------
#for now, we project onto vx vy plane until integrating.
try: #if already 2V, don't project, TODO: using try except is bad coding, do something else
    CEx_xy = CExxy
    CEy_xy = CEyxy
    CEz_xy = CEzxy
except:
    CEx_xy = []
    CEy_xy = []
    CEz_xy = []
    for i in range(0,len(CEx)):
        CEx_xy2d = ao.array_3d_to_2d(CEx[i],'xy')
        CEy_xy2d = ao.array_3d_to_2d(CEy[i],'xy')
        CEz_xy2d = ao.array_3d_to_2d(CEz[i],'xy')
        CEx_xy.append(CEx_xy2d)
        CEy_xy.append(CEy_xy2d)
        CEz_xy.append(CEz_xy2d)

#compute energization from correlations
enerCEx = anl.compute_energization_over_x(CEx_xy,dv)
enerCEy = anl.compute_energization_over_x(CEy_xy,dv)
enerCEz = anl.compute_energization_over_x(CEz_xy,dv)

#-------------------------------------------------------------------------------
# Save data with relevant input parameters
#-------------------------------------------------------------------------------
print("Saving ion results in netcdf4 file...")
flnm = path.replace("/", "_")+'ion_FPCnometadata'
if(usedfluc):
    flnm += 'dfluc'
flnm = flnm.replace('~','-')
dnc.save2Vdata(Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, dfields['Vframe_relative_to_sim'], params = params, num_par = num_par, filename = flnm+'_2v.nc')
print('Done with ion results!')

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#REPEAT FOR ELECTRONS
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("Doing FPC analysis for each slice of x for electrons...")
CEx, CEy, CEz, x, Hist, vx, vy, vz, num_par = fpc.compute_correlation_over_x(dfields, dpar_elec, vmax, dv, dx, vshock, xlim, ylim, zlim)
Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz = fpc.project_CEi_hist(Hist, CEx, CEy, CEz)

Histxy = []
Histxz = []
Histyz = []
for i in range(0,len(Hist)):
    tempHistxy = ao.array_3d_to_2d(Hist[i],'xy')
    tempHistxz = ao.array_3d_to_2d(Hist[i],'xz')
    tempHistyz = ao.array_3d_to_2d(Hist[i],'yz')
    Histxy.append(tempHistxy)
    Histxz.append(tempHistxz)
    Histyz.append(tempHistyz)
Histxy = np.asarray(Histxy)
Histxz = np.asarray(Histxz)
Histyz = np.asarray(Histyz)

CExxy = []
CExxz = []
CExyz = []
for i in range(0,len(Hist)):
    tempCExxy = ao.array_3d_to_2d(CEx[i],'xy')
    tempCExxz = ao.array_3d_to_2d(CEx[i],'xz')
    tempCExyz = ao.array_3d_to_2d(CEx[i],'yz')
    CExxy.append(tempCExxy)
    CExxz.append(tempCExxz)
    CExyz.append(tempCExyz)
CExxy = np.asarray(CExxy)
CExxz = np.asarray(CExxz)
CExyz = np.asarray(CExyz)

CEyxy = []
CEyxz = []
CEyyz = []
for i in range(0,len(Hist)):
    tempCEyxy = ao.array_3d_to_2d(CEy[i],'xy')
    tempCEyxz = ao.array_3d_to_2d(CEy[i],'xz')
    tempCEyyz = ao.array_3d_to_2d(CEy[i],'yz')
    CEyxy.append(tempCEyxy)
    CEyxz.append(tempCEyxz)
    CEyyz.append(tempCEyyz)
CEyxy = np.asarray(CEyxy)
CEyxz = np.asarray(CEyxz)
CEyyz = np.asarray(CEyyz)

CEzxy = []
CEzxz = []
CEzyz = []
for i in range(0,len(Hist)):
    tempCEzxy = ao.array_3d_to_2d(CEz[i],'xy')
    tempCEzxz = ao.array_3d_to_2d(CEz[i],'xz')
    tempCEzyz = ao.array_3d_to_2d(CEz[i],'yz')
    CEzxy.append(tempCEzxy)
    CEzxz.append(tempCEzxz)
    CEzyz.append(tempCEzyz)
CEzxy = np.asarray(CEzxy)
CEzxz = np.asarray(CEzxz)
CEzyz = np.asarray(CEzyz)

#-------------------------------------------------------------------------------
# compute energization
#-------------------------------------------------------------------------------
#for now, we project onto vx vy plane until integrating.
try: #if already 2V, don't project, TODO: using try except is bad coding, do something else
    CEx_xy = CExxy
    CEy_xy = CEyxy
    CEz_xy = CEzxy
except:
    CEx_xy = []
    CEy_xy = []
    CEz_xy = []
    for i in range(0,len(CEx)):
        CEx_xy2d = ao.array_3d_to_2d(CEx[i],'xy')
        CEy_xy2d = ao.array_3d_to_2d(CEy[i],'xy')
        CEz_xy2d = ao.array_3d_to_2d(CEz[i],'xy')
        CEx_xy.append(CEx_xy2d)
        CEy_xy.append(CEy_xy2d)
        CEz_xy.append(CEz_xy2d)

#compute energization from correlations
enerCEx = anl.compute_energization_over_x(CEx_xy,dv)
enerCEy = anl.compute_energization_over_x(CEy_xy,dv)
enerCEz = anl.compute_energization_over_x(CEz_xy,dv)

#-------------------------------------------------------------------------------
# Save data with relevant input parameters
#-------------------------------------------------------------------------------
print("Saving elec results in netcdf4 file...")

flnm = path.replace("/", "_")+'elec_FPCnometadata'
if(usedfluc):
    flnm += 'dfluc'
flnm = flnm.replace('~','-')
dnc.save2Vdata(Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, dfields['Vframe_relative_to_sim'], params = params, num_par = num_par, filename = flnm+'_2v.nc')
print('Done!!!')
