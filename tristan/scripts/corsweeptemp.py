import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import os

import matplotlib.pyplot as plt
import numpy as np
import copy
import pickle

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa
import lib.arrayaux as ao #array operations
import lib.fpcaux as fpc
import lib.plotcoraux as pfpc

pathfpcdata = ''
electotflnm = '/data/backed_up/analysis/collbrown/nonadiaperp/analysisfiles/ncsweeps/elecfacavglocframe.nc'

#load elec tot
(Hist_vxvyelec, Hist_vxvzelec, Hist_vyvzelec,
C_Ex_vxvytotelec, C_Ex_vxvztotelec, C_Ex_vyvztotelec,
C_Ey_vxvytotelec, C_Ey_vxvztotelec, C_Ey_vyvztotelec,
C_Ez_vxvytotelec, C_Ez_vxvztotelec, C_Ez_vyvztotelec,
vxelec, vyelec, vzelec, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+electotflnm)

dvelec = np.abs(vxelec[1,1,1]-vxelec[0,0,0])
vmaxelec = np.max(vxelec)

os.system('mkdir figures')
os.system('mkdir figures/sweeps')
os.system('mkdir figures/sweeps/elecfacavglocframe')

_tempint = int(0.55*len(x_in))

for _xidx in range(_tempint,len(x_in)):
    print("Making plot of CEi and dist for x: ", x_in[_xidx], " of ", x_in[-1])

    flnm = 'figures/sweeps/elecfacavglocframe/elecfacavglocframe_x_'+str("{:07d}".format(_xidx))
    pfpc.plot_cor_and_dist_supergrid(vxelec, vyelec, vzelec, vmaxelec,
                                Hist_vxvyelec[_xidx], Hist_vxvzelec[_xidx], Hist_vxvzelec[_xidx],
                                C_Ex_vxvytotelec[_xidx], C_Ex_vxvztotelec[_xidx], C_Ex_vyvztotelec[_xidx],
                                C_Ey_vxvytotelec[_xidx], C_Ey_vxvztotelec[_xidx], C_Ey_vyvztotelec[_xidx],
                                C_Ez_vxvytotelec[_xidx], C_Ez_vxvztotelec[_xidx], C_Ez_vyvztotelec[_xidx],
                                flnm = flnm, computeJdotE = True, plotFAC = False, plotAvg = False, plotFluc = False, isIon = False, listpos=True, xposval=x_in[_xidx])
