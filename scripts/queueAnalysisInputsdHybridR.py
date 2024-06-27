#!/usr/bin/env python

import sys
sys.path.append(".")

import FPCAnalysis

import os
import math
import numpy as np

try:
    analysisinputdir = sys.argv[1]

except:
    print("This script queues up generateFPC.py on all analysis inputs in specified folder")
    print("Warning: this script assumes the user has presliced the data...")
    print("usage: " + sys.argv[0] + " analysisinputdir numcores(opt) logdir(opt)")
    sys.exit()

try:
    numcores = sys.argv[2]
except:
    numcores = 1

try:
    logdir = sys.argv[3]+'/'
    try:
        cmd = 'mkdir '+logdir
        print(cmd)
        os.system(cmd)
    except:
        pass
except:
    logdir = ''

filenames = os.listdir(analysisinputdir)
filenames = sorted(filenames)
try:
    filenames.remove('.DS_store')
except:
    pass

print("Files that are queued up:")
print(filenames)

for flnm in filenames:
    #read input for directions
    f = open(analysisinputdir+'/'+flnm, "r")
    use_restart = 'F'
    is_2D3V = 'F'
    preslice_dir = None
    while(True):
        #read next line
        line = f.readline()
        line = line.strip()
        line = line.split('=')
        if(len(line)==1):
        	break
        if(line[0]=='preslice_dir'):
            preslice_dir = str(line[1].split("'")[1])
    f.close()

    cmd = 'touch '  + logdir + flnm+'.output'
    print(cmd)
    os.system(cmd)
    if(preslice_dir==None):
        print("Warning: multiprocessing requires preslicing...")
        cmd = 'python scripts/generateFPCfromdHybridR.py'+analysisinputdir+'/'+flnm+' T F  >> '+logdir+flnm+'.output'
    else:
        cmd = 'python scripts/generateFPCfromdHybridR.py '+analysisinputdir+'/'+flnm+' T F '+str(numcores)+' '+preslice_dir + ' >> '+logdir+flnm+'.output'

    print(cmd)
    os.system(cmd)
