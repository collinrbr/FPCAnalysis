#!/usr/bin/env python

import sys
sys.path.append(".")

from FPCAnalysis import *

import os
import math
import numpy as np
try:
    analysisinputflnm = sys.argv[1]
    outdirname = sys.argv[2]
    use_restart = sys.argv[3]
except:
    print("This makes hdf5 files of presliced along x data. Uses xlim and dx specified by analysis input folder.")
    print("usage: " + sys.argv[0] + " analysisinputflnm outdirname userestart(T/F)")
    sys.exit()

is2d3v = False #TODO: make compatiable with 2d3v data

if(use_restart == 'T'): #Note, we preslice restart files even tho they are already divided into blocks as this makes the pipeline more streamlined and allows us to create smaller slices in x
    use_restart = True
elif(use_restart == 'F'):
    use_restart = False
else:
    print('Please pass T or F for userestart...')

num_threads = 1 #TODO: check that this is not needed and remove it

try:
    cmd = 'mkdir ' + outdirname
    print(cmd)
    os.system(cmd)
except:
    pass

#-------------------------------------------------------------------------------
# load data
#-------------------------------------------------------------------------------
#load path
anldict = anl.analysis_input(flnm = analysisinputflnm, make_resultsdir=False)
path = anldict['path']
resultsdir = anldict['resultsdir']
vmax = anldict['vmax']
dv = anldict['dv']
numframe = anldict['numframe']
dx = anldict['dx']
xlim = anldict['xlim']
ylim = anldict['ylim']
zlim = anldict['zlim']

path_fields = path
path_particles = path+"Output/Raw/Sp01/raw_sp01_{:08d}.h5"

is2d3v = False

#load relevant time slice fields
print("Loading field data (for simulation box dimensions)...") #must load field data to get simulation box dimensions
dfields = ddhr.field_loader(path=path_fields,num=numframe,is2d3v=is2d3v)

# #Load all fields along all time slices
# all_dfields = ddhr.all_dfield_loader(path=path_fields, is2d3v=is2d3v)

#check input to make sure box makes sense
if(not(is2d3v)): #TODO: add check_input for 2d3v
    anl.check_input(analysisinputflnm,dfields)

#Load data using normal output files
if(not(use_restart)):
    print("Loading particle data...")
    #Load slice of particle data
    if xlim is not None and ylim is not None and zlim is not None:
        dparticles = ddhr.read_box_of_particles(path_particles, numframe, xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1], is2d3v=is2d3v)
    #Load all data in unspecified limits and only data in bounds in specified limits
    elif xlim is not None or ylim is not None or zlim is not None:
        if xlim is None:
            xlim = [dfields['ex_xx'][0],dfields['ex_xx'][-1]]
        if ylim is None:
            ylim = [dfields['ex_yy'][0],dfields['ex_yy'][-1]]
        if zlim is None:
            zlim = [dfields['ex_zz'][0],dfields['ex_zz'][-1]]
        dparticles = ddhr.read_box_of_particles(path_particles, numframe, xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1], is2d3v=is2d3v)
    #Load all the particles
    else:
        xlim = [dfields['ex_xx'][0],dfields['ex_xx'][-1]]
        ylim = [dfields['ex_yy'][0],dfields['ex_yy'][-1]]
        zlim = [dfields['ex_zz'][0],dfields['ex_zz'][-1]]
        dparticles = ddhr.read_particles(path_particles, numframe, is2d3v=is2d3v)

#-------------------------------------------------------------------------------
# slice data
#-------------------------------------------------------------------------------
#setup sweeping box
if(xlim == None): #TODO: clean up doing this (same logic is done in multiple places in this script)
     xlim = [dfields['ex_xx'][0],dfields['ex_xx'][-1]]
if(ylim == None):
     ylim = [dfields['ex_yy'][0],dfields['ex_yy'][-1]]
if(zlim == None):
     zlim = [dfields['ex_zz'][0],dfields['ex_zz'][-1]]


x1 = xlim[0]
x2 = x1+dx
xEnd = xlim[1]
y1 = ylim[0]
y2 = ylim[1]
z1 = zlim[0]
z2 = zlim[1]
dparkeys='p1 p2 p3 x1 x2 x3'.split()
while(x2 <= xEnd):
    print("x1: ", x1, "x2: ", x2)
    if(use_restart): #if we are using restart files, must load relevant files
        dparticles = ddhr.read_restart(path,verbose=True,xlim=[x1,x2],nthreads=1)
    gptsparticle = (x1 <= dparticles['x1']) & (dparticles['x1'] <= x2) & (y1 <= dparticles['x2']) & (dparticles['x2'] <= y2) & (z1 <= dparticles['x3']) & (dparticles['x3'] <= z2)
    _tempdpar = {}
    for key in dparkeys:
            _tempdpar[key] = dparticles[key][gptsparticle][:]

    outflnm = outdirname + '/' + '{:012.6f}'.format(x1) + '_' + '{:012.6f}'.format(x2)
    ddhr.write_particles_to_hdf5(_tempdpar,outflnm)
    x1 += dx
    x2 += dx
cmd = 'touch origin.txt'
os.system(cmd)
cmd = "echo '"+path +"'  >>  origin.txt"
print("Done!")
