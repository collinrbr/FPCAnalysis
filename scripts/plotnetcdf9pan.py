from FPCAnalysis import *

import os
import sys

try:
    path = sys.argv[1]

except:
    print("This script plots CEi and dist func for each slice of x for a given netcdf4 file")
    print("usage: " + sys.argv[0] + " path/to/FPCdata/within/*.nc plotLog(default F)")
    sys.exit()

try:
    plotLog = sys.argv[2]
    if(plotLog == 'T'):
        plotLog = True
        print("Plotting with log scale! (log-lin-log for FPC data)")
    else:
        plotLog = False
        print("Plotting with normal scale!")
except:
    plotLog = False #TODO: more input parsing
    print("Plotting with normal scale!")

#load data
print("Loading data...")
try:
    (Hist_vxvy, Hist_vxvz, Hist_vyvz,
    C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
    C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
    C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
    vx, vy, vz, x_in,
    enerCEx_in, enerCEy_in, enerCEz_in,
    npar_in, Vframe_relative_to_sim_in, metadata_in, params_in) = dnc.load2vdata(path)
except:
    (Hist_vxvy, Hist_vxvz, Hist_vyvz,
    C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
    C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
    C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
    vx, vy, vz, x_in,
    enerCEx_in, enerCEy_in, enerCEz_in,
    Vframe_relative_to_sim_in, metadata_in, params_in) = dnc.load2vdata(path)
print("Done!")

if(plotLog):
    directory = path+'.log9panelplot/'
else:
    directory = path+'.9panelplot/'

try:
    os.makedirs(directory)
except:
    pass

pltvv.make_9panel_sweep_from_2v(Hist_vxvy, Hist_vxvz, Hist_vyvz,
                                C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
                                C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
                                C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
                                vx, vy,vz,params_in,x_in,metadata_in,
                                directory,plotLog=plotLog)

print("Done making plots! Plots are saved to the same directory as the supplied netcdf4 file...")
