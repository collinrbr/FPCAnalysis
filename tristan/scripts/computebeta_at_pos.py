import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa

import pickle

#------------------------------------------------------------------------------------------------------------------------------------
# Begin script
#------------------------------------------------------------------------------------------------------------------------------------
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
#dfields_many_frames = {'frame':[],'dfields':[]}
#for _num in frames:
#    num = int(_num)
#    d = ld.load_fields(flpath,_num,normalizeFields=normalize)
#    dfields_many_frames['dfields'].append(d)
#    dfields_many_frames['frame'].append(num)
#vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)
vshock = 1.5

dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
inputs = ld.load_input(inputpath)
beta0 = aa.compute_beta0(params,inputs)

loadflow=True
if(loadflow):
    filein = open(dflowflnm, 'rb')
    dflow = pickle.load(filein)
    filein.close()
else:
    dpar_ion = ft.shift_particles(dpar_ion, vshock, beta0, params['mi']/params['me'], isIon=True)
    dpar_elec = ft.shift_particles(dpar_elec, vshock, beta0, params['mi']/params['me'], isIon=False)
    dflow = aa.compute_dflow(dfields, dpar_ion, dpar_elec)

dfavg = aa.get_average_fields_over_yz(dfields) 
dflowavg = aa.get_average_flow_over_yz(dflow)

bx = dfavg['bx'][0,0,:]
by = dfavg['by'][0,0,:]
bz = dfavg['bz'][0,0,:]
den = dfavg['dens'][0,0,:]
vx = dflowavg['ue'][0,0,:]
vy = dflowavg['ve'][0,0,:]
vz = dflowavg['we'][0,0,:]

upstreamidx = int(3/4*len(dfavg['dens'][0,0,:])) #note, if we go too far upstream, the system has no particles in it- so upstream is not the edge of the simulation but slightly away from it

bx0 = dfavg['bx'][0,0,upstreamidx]
by0 = dfavg['by'][0,0,upstreamidx]
bz0 = dfavg['bz'][0,0,upstreamidx]
den0 = dfavg['dens'][0,0,upstreamidx]
vx0 = dflowavg['ue'][0,0,upstreamidx]
vy0 = dflowavg['ve'][0,0,upstreamidx]
vz0 = dflowavg['we'][0,0,upstreamidx]

#TODO: it seems that i was using vth^2 as a proxy for vth, but we would need to be a in a differetnf frame to do this!---- fix this
betas = beta0*den/den0*((vx**2+vy**2+vz**2)/(vx0**2+vy0**2+vz0**2))*((bx0**2+by0**2+bz0**2)/(bx**2+by**2+bz**2))
TODO: fix beta!!!


import os
os.system('mkdir figures')
flnmout = 'figures/betavspos.png'

plt.figure()
plt.plot(dfavg['bx_xx'],betas)
plt.xlabel(r'$x/d_i$')
plt.ylabel(r'$\beta$')
plt.savefig(flnmout,format='png',dpi=300,bbox_inches='tight')
plt.close()
