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
ionflucflnm = '/data/backed_up/analysis/collbrown/nonadiaperp/analysisfiles/ncsweeps/ionfluc.nc'
iontotflnm = '/data/backed_up/analysis/collbrown/nonadiaperp/analysisfiles/ncsweeps/iontot.nc'
elecflucflnm = '/data/backed_up/analysis/collbrown/nonadiaperp/analysisfiles/ncsweeps/elecfluc.nc'
electotflnm = '/data/backed_up/analysis/collbrown/nonadiaperp/analysisfiles/ncsweeps/electot.nc'

#load ion fluc
(Hist_vxvyion, Hist_vxvzion, Hist_vyvzion,
C_Ex_vxvyflucion, C_Ex_vxvzflucion, C_Ex_vyvzflucion,
C_Ey_vxvyflucion, C_Ey_vxvzflucion, C_Ey_vyvzflucion,
C_Ez_vxvyflucion, C_Ez_vxvzflucion, C_Ez_vyvzflucion,
vxion, vyion, vzion, x_in,
_, _, _, #TODO: remove unused inputs
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+ionflucflnm)
dvion = np.abs(vxion[1,1,1]-vxion[0,0,0])
vmaxion = np.max(vxion)

#load ion tot
(Hist_vxvyion, Hist_vxvzion, Hist_vyvzion,
C_Ex_vxvytotion, C_Ex_vxvztotion, C_Ex_vyvztotion,
C_Ey_vxvytotion, C_Ey_vxvztotion, C_Ey_vyvztotion,
C_Ez_vxvytotion, C_Ez_vxvztotion, C_Ez_vyvztotion,
vxion, vyion, vzion, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+iontotflnm)

#load elec fluc
(Hist_vxvyelec, Hist_vxvzelec, Hist_vyvzelec,
C_Ex_vxvyflucelec, C_Ex_vxvzflucelec, C_Ex_vyvzflucelec,
C_Ey_vxvyflucelec, C_Ey_vxvzflucelec, C_Ey_vyvzflucelec,
C_Ez_vxvyflucelec, C_Ez_vxvzflucelec, C_Ez_vyvzflucelec,
vxelec, vyelec, vzelec, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+elecflucflnm)
dvelec = np.abs(vxelec[1,1,1]-vxelec[0,0,0])
vmaxelec = np.max(vxelec)

#load elec tot
(Hist_vxvyelec, Hist_vxvzelec, Hist_vyvzelec,
C_Ex_vxvytotelec, C_Ex_vxvztotelec, C_Ex_vyvztotelec,
C_Ey_vxvytotelec, C_Ey_vxvztotelec, C_Ey_vyvztotelec,
C_Ez_vxvytotelec, C_Ez_vxvztotelec, C_Ez_vyvztotelec,
vxelec, vyelec, vzelec, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+electotflnm)

os.system('mkdir figures')
os.system('mkdir figures/sweeps')
os.system('mkdir figures/sweeps/ionfluc')
os.system('mkdir figures/sweeps/iontot')
os.system('mkdir figures/sweeps/elecfluc')
os.system('mkdir figures/sweeps/electot')

for _xidx in range(0,len(x_in)):
    print("Making plot of CEi and dist for x: ", x_in[_xidx], " of ", x_in[-1])

    vmaxion = np.max(np.abs(vxion))
    vmaxelec = np.abs(np.max(vxelec))
    
    flnm = 'figures/sweeps/ionfluc/ionfluc_x_'+str("{:07d}".format(_xidx))
    pfpc.plot_cor_and_dist_supergrid(vxion, vyion, vzion, vmaxion,
                                Hist_vxvyion[_xidx], Hist_vxvzion[_xidx], Hist_vxvzion[_xidx],
                                C_Ex_vxvyflucion[_xidx], C_Ex_vxvzflucion[_xidx], C_Ex_vyvzflucion[_xidx],
                                C_Ey_vxvyflucion[_xidx], C_Ey_vxvzflucion[_xidx], C_Ey_vyvzflucion[_xidx],
                                C_Ez_vxvyflucion[_xidx], C_Ez_vxvzflucion[_xidx], C_Ez_vyvzflucion[_xidx],
                                flnm = flnm, computeJdotE = True, plotFAC = False, plotAvg = False, plotFluc = True, isIon = True, listpos=True, xposval=x_in[_xidx])

    flnm = 'figures/sweeps/iontot/iontot_x_'+str("{:07d}".format(_xidx))
    pfpc.plot_cor_and_dist_supergrid(vxion, vyion, vzion, vmaxion,
                                Hist_vxvyion[_xidx], Hist_vxvzion[_xidx], Hist_vxvzion[_xidx],
                                C_Ex_vxvytotion[_xidx], C_Ex_vxvztotion[_xidx], C_Ex_vyvztotion[_xidx],
                                C_Ey_vxvytotion[_xidx], C_Ey_vxvztotion[_xidx], C_Ey_vyvztotion[_xidx],
                                C_Ez_vxvytotion[_xidx], C_Ez_vxvztotion[_xidx], C_Ez_vyvztotion[_xidx],
                                flnm = flnm, computeJdotE = True, plotFAC = False, plotAvg = False, plotFluc = False, isIon = True, listpos=True, xposval=x_in[_xidx])

    flnm = 'figures/sweeps/elecfluc/elecfluc_x_'+str("{:07d}".format(_xidx))
    pfpc.plot_cor_and_dist_supergrid(vxelec, vyelec, vzelec, vmaxelec,
                                Hist_vxvyelec[_xidx], Hist_vxvzelec[_xidx], Hist_vxvzelec[_xidx],
                                C_Ex_vxvyflucelec[_xidx], C_Ex_vxvzflucelec[_xidx], C_Ex_vyvzflucelec[_xidx],
                                C_Ey_vxvyflucelec[_xidx], C_Ey_vxvzflucelec[_xidx], C_Ey_vyvzflucelec[_xidx],
                                C_Ez_vxvyflucelec[_xidx], C_Ez_vxvzflucelec[_xidx], C_Ez_vyvzflucelec[_xidx],
                                flnm = flnm, computeJdotE = True, plotFAC = False, plotAvg = False, plotFluc = True, isIon = False, listpos=True, xposval=x_in[_xidx])

    flnm = 'figures/sweeps/electot/electot_x_'+str("{:07d}".format(_xidx))
    pfpc.plot_cor_and_dist_supergrid(vxelec, vyelec, vzelec, vmaxelec,
                                Hist_vxvyelec[_xidx], Hist_vxvzelec[_xidx], Hist_vxvzelec[_xidx],
                                C_Ex_vxvytotelec[_xidx], C_Ex_vxvztotelec[_xidx], C_Ex_vyvztotelec[_xidx],
                                C_Ey_vxvytotelec[_xidx], C_Ey_vxvztotelec[_xidx], C_Ey_vyvztotelec[_xidx],
                                C_Ez_vxvytotelec[_xidx], C_Ez_vxvztotelec[_xidx], C_Ez_vyvztotelec[_xidx],
                                flnm = flnm, computeJdotE = True, plotFAC = False, plotAvg = False, plotFluc = False, isIon = False, listpos=True, xposval=x_in[_xidx])
