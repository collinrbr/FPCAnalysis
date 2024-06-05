#!/usr/bin/env python

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
try:
    analysisinputflnm = sys.argv[1]
except:
    print("This generates FPC netcdf4 file from dHybridR data. Use_restart is false by default.")
    print("usage: " + sys.argv[0] + " analysisinputflnm use_restart(T/F) is_2D3V(T/F) num_threads(default 1) dpar_folder use_dfluc(default F)")
    sys.exit()

try:
    use_restart = str(sys.argv[2].upper())
    if(use_restart != 'T' and use_restart != 'F'):
        print("Error, use_restart should be T or F...")
        sys.exit()
except:
    use_restart = 'F'

try:
    is_2D3V = str(sys.argv[3].upper())
    if(use_restart != 'T' and use_restart != 'F'):
        print("Error, use_restart should be T or F...")
        sys.exit()
except:
    is_2D3V = 'F'

try:
    num_threads = int(sys.argv[4])
    if(num_threads <= 0):
        print("Error, please request at least 1 thread...")
except:
    num_threads = 1

try:
    dpar_folder = sys.argv[5]+'/'
except:
    dpar_folder = None

if(is_2D3V == 'T'):
    is2d3v = True
else:
    is2d3v = False

if(use_restart == 'T'):
    use_restart = True
else:
    use_restart = False

try:
    use_dfluc = sys.argv[6]
    if(use_dfluc == 'T'):
        use_dfluc = True
    else:
        use_dfluc = False
except:
    use_dfluc = False

if(num_threads != 1 and dpar_folder == None): #TODO: clean up code that assumes multithreading without preslicing data
    print("Error, multithreading now expects pre-slicing of the data.")
    print("Please use preslicedata.py, and pass output folder to dpar_folder")

#-------------------------------------------------------------------------------
# load data
#-------------------------------------------------------------------------------
#load path
path,resultsdir,vmax,dv,numframe,dx,xlim,ylim,zlim = anl.analysis_input(flnm = analysisinputflnm)
path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"

#load relevant time slice fields
print("Loading field data...")
dfields = dh5.field_loader(path=path_fields,num=numframe,is2d3v=is2d3v)

#Load all fields along all time slices
all_dfields = dh5.all_dfield_loader(path=path_fields, is2d3v=is2d3v)

#check input to make sure box makes sense
if(not(is2d3v)): #TODO: add check_input for 2d3v
    anl.check_input(analysisinputflnm,dfields)

#Load data using normal output files
if(not(use_restart) and dpar_folder == None):
    print("Loading particle data...")
    #Load slice of particle data
    if xlim is not None and ylim is not None and zlim is not None:
        dparticles = dh5.read_box_of_particles(path_particles, numframe, xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1], is2d3v=is2d3v)
    #Load all data in unspecified limits and only data in bounds in specified limits
    elif xlim is not None or ylim is not None or zlim is not None:
        if xlim is None:
            xlim = [dfields['ex_xx'][0],dfields['ex_xx'][-1]]
        if ylim is None:
            ylim = [dfields['ex_yy'][0],dfields['ex_yy'][-1]]
        if zlim is None:
            zlim = [dfields['ex_zz'][0],dfields['ex_zz'][-1]]
        dparticles = dh5.read_box_of_particles(path_particles, numframe, xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1], is2d3v=is2d3v)
    #Load all the particles
    else:
        dparticles = dh5.read_particles(path_particles, numframe, is2d3v=is2d3v)

#Load data using restart files
if(use_restart and dpar_folder == None):
    print("Loading particle data using restart files...")
    #Load slice of particle data
    if xlim is not None:
        dparticles = dh5.read_restart(path, xlim=xlim,nthreads=num_threads)
    #Load all data in unspecified limits and only data in bounds in specified limits
    else:
        xlim = [dfields['ex_xx'][0],dfields['ex_xx'][-1]]
        dparticles = dh5.read_restart(path,nthreads=num_threads)

    #set up other bounds (TODO: clean this up (redundant code in above if block; code this only once))
    if ylim is None:
        ylim = [dfields['ex_yy'][0],dfields['ex_yy'][-1]]
    if zlim is None:
        zlim = [dfields['ex_zz'][0],dfields['ex_zz'][-1]]

#setup bounds if dpar_folder is used
if(dpar_folder != None): #TODO: clean up how xlim, ylim, zlim is assigned everywhere in code!!!
        if xlim is None:
            xlim = [dfields['ex_xx'][0],dfields['ex_xx'][-1]]
        if ylim is None:
            ylim = [dfields['ex_yy'][0],dfields['ex_yy'][-1]]
        if zlim is None:
            zlim = [dfields['ex_zz'][0],dfields['ex_zz'][-1]]

#-------------------------------------------------------------------------------
# If vmax is not specified, use maximum particle velocity as vmax
#-------------------------------------------------------------------------------
if(vmax == None):
    print("Vmax was not specified. Using maximum...")
    print("Computing maximum...")
    vmax = anl.get_max_speed(dparticles)
    print("Done!")


#-------------------------------------------------------------------------------
# Check if data is 2D. Pad to make pseudo 3d if data is 2d
#-------------------------------------------------------------------------------
#print(is_2D3V)
if(is_2D3V == 'T'): #TODO: fix inconsistency (is_2D3V is either 'T'/'F' while is2D3V is either True or False) have one var and use only boolean
    if('x3' not in dparticles.keys()): #all simulations that are '2d' should be 2d 3v with coordinates (xx,yy;vx,vy,vz)
        dparticles = dh5.par_2d_to_3d(dparticles)
        dfields = dh5.dict_2d_to_3d(dfields,0)
        _fields = []
        for k in range(0,len(all_dfields['dfields'])):
            _fields.append(dh5.dict_2d_to_3d(all_dfields['dfields'][k],0))
        all_dfields['dfields'] = _fields

#-------------------------------------------------------------------------------
# estimate shock vel and lorentz transform
#-------------------------------------------------------------------------------
print("Lorentz transforming fields...")
vshock,_ = ft.shock_from_ex_cross(all_dfields,dt=0.01)

#Lorentz transform fields
dfields = ft.lorentz_transform_vx(dfields,vshock)
_fields = []
for k in range(0,len(all_dfields['dfields'])):
    _fields.append(ft.lorentz_transform_vx(all_dfields['dfields'][k],vshock))
all_dfields['dfields'] = _fields

#-------------------------------------------------------------------------------
# use fluc if requested
#-------------------------------------------------------------------------------
if(use_dfluc):
    dfields = anl.remove_average_fields_over_yz(dfields)

#-------------------------------------------------------------------------------
# do FPC analysis
#-------------------------------------------------------------------------------
print("Doing FPC analysis for each slice of x...")
if dx is None:
    #Assumes rectangular grid that is uniform for all fields
    #If dx not specified, just use the grid cell spacing for the EM fields
    dx = dfields['ex_xx'][1]-dfields['ex_xx'][0]
if(num_threads == 1):
    CEx, CEy, CEz, x, Hist, vx, vy, vz, num_par = fpc.compute_correlation_over_x(dfields, dparticles, vmax, dv, dx, vshock, xlim, ylim, zlim)
else:
    CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz,x, Histxy,Histxz,Histyz, vx, vy, vz, num_par = fpc.comp_cor_over_x_multithread(dfields, dpar_folder, vmax, dv, dx, vshock, xlim=xlim, ylim=ylim, zlim=zlim, max_workers=num_threads)

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
print("Saving results in netcdf4 file...")
inputdict = dnc.parse_input_file(path)
params = dnc.build_params(inputdict,numframe)

flnm = 'FPCnometadata'
if(num_threads == 1):#For now, we pre project if multithreading
    dnc.save3Vdata(Hist, CEx, CEy, CEz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, dfields['Vframe_relative_to_sim'], params = params, num_par = num_par, filename = resultsdir+flnm+'.nc')
else:
    dnc.save2Vdata(Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, dfields['Vframe_relative_to_sim'], params = params, num_par = num_par, filename = resultsdir+flnm+'_2v.nc')
print("Done! Please use findShock.py and addMetadata to assign metadata...")