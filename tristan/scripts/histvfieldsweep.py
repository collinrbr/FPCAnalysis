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

#user params
framenum = '700' #frame to make figure of (should be a string)
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]

loadflow = True
dflowflnm = 'dflow.pickle'

normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)

params = ld.load_params(flpath,framenum)
dt = params['c']/params['comp'] #in units of wpe
c = params['c']
stride = 100
dt,c = aa.norm_constants(params,dt,c,stride)

#compute shock velocity and boost to shock rest frame
vshock = 1.5
dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

pathfpcdata = ''
ionflucflnm = '/data/backed_up/analysis/collbrown/nonadiaperp/analysisfiles/ncsweeps/ionfluc.nc'
iontotflnm = '/data/backed_up/analysis/collbrown/nonadiaperp/analysisfiles/ncsweeps/iontot.nc'
elecflucflnm = '/data/backed_up/analysis/collbrown/nonadiaperp/analysisfiles/ncsweeps/elecfluc.nc'
electotflnm = '/data/backed_up/analysis/collbrown/nonadiaperp/analysisfiles/ncsweeps/electot.nc'

#load ion tot
(Hist_vxvyion, Hist_vxvzion, Hist_vyvzion,
C_Ex_vxvytotion, C_Ex_vxvztotion, C_Ex_vyvztotion,
C_Ey_vxvytotion, C_Ey_vxvztotion, C_Ey_vyvztotion,
C_Ez_vxvytotion, C_Ez_vxvztotion, C_Ez_vyvztotion,
vxion, vyion, vzion, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+iontotflnm)

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
os.system('mkdir figures/sweeps/histion')
os.system('mkdir figures/sweeps/histelec')


dfavg = aa.get_average_fields_over_yz(dfields)


offsetidx = 100
for _xidx in range(offsetidx,len(x_in)-50):
    print("Making plot of fields and dist for x: ", x_in[_xidx], " of ", x_in[-45])

    vmaxion = np.max(np.abs(vxion))
    vmaxelec = np.abs(np.max(vxelec))
    
    dx = x_in[1]-x_in[0]
    xval1 = x_in[_xidx]-dx/2
    xval2 = x_in[_xidx]+dx/2

    flnm = 'figures/sweeps/histion/iontot_x_'+str("{:07d}".format(_xidx-offsetidx))
    pfpc.plot_dist_v_fields_supergrid(vxion, vyion, vzion, vmaxion,
                                Hist_vxvyion[_xidx], Hist_vxvzion[_xidx], Hist_vxvzion[_xidx],
                                C_Ex_vxvytotion[_xidx], C_Ex_vxvztotion[_xidx], C_Ex_vyvztotion[_xidx],
                                C_Ey_vxvytotion[_xidx], C_Ey_vxvztotion[_xidx], C_Ey_vyvztotion[_xidx],
                                C_Ez_vxvytotion[_xidx], C_Ez_vxvztotion[_xidx], C_Ez_vyvztotion[_xidx],
                                dfavg,xval1,xval2,
                                flnm = flnm, computeJdotE = True, plotFAC = False, plotAvg = False, plotFluc = False, isIon = True, listpos=True, xposval=x_in[_xidx])

    flnm = 'figures/sweeps/histelec/electot_x_'+str("{:07d}".format(_xidx-offsetidx))
    pfpc.plot_dist_v_fields_supergrid(vxelec, vyelec, vzelec, vmaxelec,
                                Hist_vxvyelec[_xidx], Hist_vxvzelec[_xidx], Hist_vxvzelec[_xidx],
                                C_Ex_vxvytotelec[_xidx], C_Ex_vxvztotelec[_xidx], C_Ex_vyvztotelec[_xidx],
                                C_Ey_vxvytotelec[_xidx], C_Ey_vxvztotelec[_xidx], C_Ey_vyvztotelec[_xidx],
                                C_Ez_vxvytotelec[_xidx], C_Ez_vxvztotelec[_xidx], C_Ez_vyvztotelec[_xidx],
                                dfavg,xval1,xval2,
                                flnm = flnm, computeJdotE = True, plotFAC = False, plotAvg = False, plotFluc = False, isIon = False, listpos=True, xposval=x_in[_xidx])
