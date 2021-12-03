#!/usr/bin/env python

#quick script to project hist/CEi(vx,vy,vz) onto hist/CEi(vx,vy) hist/CEi(vx,vz) hist/CEi(vy,vz)

import sys
sys.path.append(".")

import lib.analysis as anl
import lib.array_ops as ao
import lib.data_h5 as dh5
import lib.data_netcdf4 as dnc
import lib.fpc as fpc
import lib.frametransform as ft
import lib.metadata as md

import lib.plot.oned as plt1d
import lib.plot.twod as plt2d
import lib.plot.debug as pltdebug
import lib.plot.fourier as pltfr
import lib.plot.resultsmanager as rsltmng
import lib.plot.velspace as pltvv

import os
import math
import numpy as np

from netCDF4 import Dataset
from datetime import datetime

try:
    flnmin = sys.argv[1]
except:
    print("This converts nc data of the form CE_i/Hist(x;vx,vy,vz) to CE_i/Hist(x;vx,vy),CE_i/Hist(x;vx,vz),CE_i/Hist(x;vy,vz)")
    print("usage: " + sys.argv[0] + " filename")
    sys.exit()


#-------------------------------------------------------------------------------
# load data
#-------------------------------------------------------------------------------
#load original netcdf4 file
print("Loading data...")
Hist, CEx, CEy, CEz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, Vframe_relative_to_sim, metadata, params = dnc.load3Vnetcdf4(flnmin)

#-------------------------------------------------------------------------------
# Project data
#-------------------------------------------------------------------------------
print("Projecting data...")
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
# savedata
#-------------------------------------------------------------------------------
#Save data into netcdf4 file-----------------------------------------------------
print("Saving data into netcdf4 file...")

filename = flnmin[:flnmin.index('.')]+'_2v.nc'

# open a netCDF file to write
ncout = Dataset(filename, 'w', format='NETCDF4')

#save data in netcdf file-------------------------------------------------------
#define simulation parameters
for key in params:
    #setattr(ncout,key,params[key])
    if(not(isinstance(params[key],str))):
        _ = ncout.createVariable(key,None)
        _[:] = params[key]

ncout.description = 'dHybridR MLA data 2V format'
ncout.generationtime = str(datetime.now())
ncout.version = dnc.get_git_head()

#make dimensions that dependent data must 'match'
ncout.createDimension('nx', None)  # NONE <-> unlimited TODO: make limited if it saves memory or improves compression?
ncout.createDimension('nvx', None)
ncout.createDimension('nvy', None)
ncout.createDimension('nvz', None)

vx = vx[0][0][:]
vx_out = ncout.createVariable('vx','f4', ('nvx',))
vx_out.nvx = len(vx)
vx_out.longname = 'v_x/v_ti'
vx_out[:] = vx[:]

vy = np.asarray([vy[0][i][0] for i in range(0,len(vy))])
vy_out = ncout.createVariable('vy','f4', ('nvy',))
vy_out.nvy = len(vy)
vy_out.longname = 'v_y/v_ti'
vy_out[:] = vy[:]

vz = np.asarray([vz[i][0][0] for i in range(0,len(vz))]) #assumes same number of data points along all axis in vz_out mesh var
vz_out = ncout.createVariable('vz','f4', ('nvz',))
vz_out.nvz = len(vz)
vz_out.longname = 'v_z/v_ti'
vz_out[:] = vz[:]

x_out = ncout.createVariable('x','f4',('nx',))
x_out.nx = len(x)
x_out[:] = x[:]

C_ex_vxvy = ncout.createVariable('C_Ex_vxvy','f4',('nx','nvx','nvy'))
C_ex_vxvy.longname = 'C_{Ex}(x;vx,vy)'
C_ex_vxvy[:] = CExxy[:]
C_ex_vxvz = ncout.createVariable('C_Ex_vxvz','f4',('nx','nvx','nvz'))
C_ex_vxvz.longname = 'C_{Ex}(x;vx,vz)'
C_ex_vxvz[:] = CExxz[:]
C_ex_vyvz = ncout.createVariable('C_Ex_vyvz','f4',('nx','nvy','nvz'))
C_ex_vyvz.longname = 'C_{Ex}(x;vy,vz)'
C_ex_vyvz[:] = CExyz[:]

C_ey_vxvy = ncout.createVariable('C_Ey_vxvy','f4',('nx','nvx','nvy'))
C_ey_vxvy.longname = 'C_{Ey}(x;vx,vy)'
C_ey_vxvy[:] = CEyxy[:]
C_ey_vxvz = ncout.createVariable('C_Ey_vxvz','f4',('nx','nvx','nvz'))
C_ey_vxvz.longname = 'C_{Ey}(x;vx,vz)'
C_ey_vxvz[:] = CEyxz[:]
C_ey_vyvz = ncout.createVariable('C_Ey_vyvz','f4',('nx','nvy','nvz'))
C_ey_vyvz.longname = 'C_{Ey}(x;vy,vz)'
C_ey_vyvz[:] = CEyyz[:]

C_ez_vxvy = ncout.createVariable('C_Ez_vxvy','f4',('nx','nvx','nvy'))
C_ez_vxvy.longname = 'C_{Ez}(x;vx,vy)'
C_ez_vxvy[:] = CEzxy[:]
C_ez_vxvz = ncout.createVariable('C_Ez_vxvz','f4',('nx','nvx','nvz'))
C_ez_vxvz.longname = 'C_{Ez}(x;vx,vz)'
C_ez_vxvz[:] = CEzxz[:]
C_ez_vyvz = ncout.createVariable('C_Ez_vyvz','f4',('nx','nvy','nvz'))
C_ez_vyvz.longname = 'C_{Ez}(x;vy,vz)'
C_ez_vyvz[:] = CEzyz[:]

Hist_vxvy = ncout.createVariable('Hist_vxvy','f4',('nx','nvx','nvy'))
Hist_vxvy.longname = 'Hist(x;vx,vy)'
Hist_vxvy[:] = Histxy[:]
Hist_vxvz = ncout.createVariable('Hist_vxvz','f4',('nx','nvx','nvz'))
Hist_vxvz.longname = 'Hist(x;vx,vz)'
Hist_vxvz[:] = Histxz[:]
Hist_vyvz = ncout.createVariable('Hist_vyvz','f4',('nx','nvy','nvz'))
Hist_vyvz.longname = 'Hist(x;vy,vz)'
Hist_vyvz[:] = Histyz[:]

sda = ncout.createVariable('sda','f4',('nx',))
sda.description = '1 = signature, 0 = no signature'
sda[:] = metadata[:]

enerCEx_out = ncout.createVariable('E_CEx','f4',('nx',))
enerCEx_out.description = 'Energization computed by integrating over CEx in velocity space'
enerCEx_out[:] = enerCEx[:]

enerCEy_out = ncout.createVariable('E_CEy','f4',('nx',))
enerCEy_out.description = 'Energization computed by integrating over CEy in velocity space'
enerCEy_out[:] = enerCEy[:]

enerCEz_out = ncout.createVariable('E_CEz','f4',('nx',))
enerCEz_out.description = 'Energization computed by integrating over CEy in velocity space'
enerCEz_out[:] = enerCEz[:]

n_par_out = ncout.createVariable('n_par','f4',('nx',))
n_par_out.description = 'number of particles in each integration box. Note: this number might not match zeroth moment of the distribution function (i.e. integration over velocity space of hist) as particles outside the range of vmax are not counted in hist'
n_par_out[:] = npar[:]

Vframe_relative_to_sim_out = ncout.createVariable('Vframe_relative_to_sim', 'f4')
Vframe_relative_to_sim_out[:] = Vframe_relative_to_sim

#save file
ncout.close()

print("Done!")
