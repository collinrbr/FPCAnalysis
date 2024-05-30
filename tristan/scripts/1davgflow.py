import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np
import pickle

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa

import os
os.system('mkdir figures')
os.system('mkdir figures/1dplots')

#user params
flowflnm = 'dflow.pickle'
framenum = '700' #frame to make figure of (should be a string)
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]
inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'

params = ld.load_params(flpath,framenum)

#compute dflow
normalize = True
dpar_elec, dpar_ion = ld.load_particles(flpath,framenum,normalizeVelocity=normalize)
uinj = 0.015062 #from input file #TODO: load automatically 
inputs = ld.load_input(inputpath)

normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)
dden = ld.load_den(flpath,framenum)
for _key in dden.keys():
    dfields[_key] = dden[_key]

inputs = ld.load_input(inputpath)
beta0 = aa.compute_beta0(params,inputs)
try:
    #load dflow
    filein = open(flowflnm, 'rb')
    dflow = pickle.load(filein)
    filein.close()
except:
    print("Couldn't load flow, computing it from particle data")
    print("Warning using hard coded value of shock.")
    print("TODO: load from file")
    vshock = 1.53
    dpar_ion = ft.shift_particles(dpar_ion, vshock, beta0, params['mi']/params['me'], isIon=True) #converts from normalization by va to vth
    dpar_elec = ft.shift_particles(dpar_elec, vshock, beta0, params['mi']/params['me'], isIon=False)
    dflow = aa.compute_dflow(dfields, dpar_ion, dpar_elec)
    dflowavg = aa.get_average_flow_over_yz(dflow)

params = ld.load_params(flpath,framenum)
dt = params['c']/params['comp'] #in units of wpe
c = params['c']
stride = 100
dt,c = aa.norm_constants(params,dt,c,stride)

#compute shock velocity and boost to shock rest frame
dfields_many_frames = {'frame':[],'dfields':[]}
for _num in frames:
    num = int(_num)
    d = ld.load_fields(flpath,_num,normalizeFields=normalize)
    dfields_many_frames['dfields'].append(d)
    dfields_many_frames['frame'].append(num)
vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)
dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

dfavg = aa.get_average_fields_over_yz(dfields)

dflowavg = aa.get_average_flow_over_yz(dflow)

#Plot quanitiies
fig, axs = plt.subplots(5, 1, figsize=(25,12), sharex=True)
plt.style.use('cb.mplstyle')
axs[0].plot(dfavg['by_xx'],dfavg['by'][0,0,:])
axs[0].set_ylabel(r'$<B_y>_{y,z}$')
axs[1].plot(dflowavg['vi_xx'],dflowavg['vi'][0,0,:])
axs[1].set_ylabel(r'$<v_{y,i}/v_{th,i}>$')
axs[2].plot(dflowavg['ve_xx'],dflowavg['ve'][0,0,:])
axs[2].set_ylabel(r'$<v_{y,e}/v_{th,e}>$')
axs[3].plot(dflowavg['ui_xx'],dflowavg['ui'][0,0,:])
axs[3].set_ylabel(r'$<v_{x,i}/v_{th,i}>$')
axs[4].plot(dflowavg['ue_xx'],dflowavg['ue'][0,0,:])
axs[4].set_ylabel(r'$<v_{x,e}/v_{th,e}>$')
axs[-1].set_xlabel('x/d_i')
for ax in axs:
    ax.grid()
plt.savefig('figures/1dplots/1dflowfig_xdriftandydriftvsx_shockframe.png',format='png',dpi=300)
plt.close()

fig, axs = plt.subplots(2, 1, figsize=(20,8), sharex=True)
plt.style.use('cb.mplstyle')
axs[0].plot(dflowavg['vi_xx'],dflowavg['ui'][0,0,:],color='r',ls='-',label=r"$<v_{x,i}/v_{th,i}>$")
axs[0].plot(dflowavg['vi_xx'],dflowavg['vi'][0,0,:],color='g',ls='-.',label=r"$<v_{y,i}/v_{th,i}>$")
axs[0].plot(dflowavg['vi_xx'],dflowavg['wi'][0,0,:],color='b',ls=':',label=r"$<v_{z,i}/v_{th,i}>$")
axs[1].plot(dflowavg['ve_xx'],dflowavg['ue'][0,0,:],color='r',ls='-',label=r"$<v_{x,e}/v_{th,e}>$")
axs[1].plot(dflowavg['ve_xx'],dflowavg['ve'][0,0,:],color='g',ls='-.',label=r"$<v_{y,e}/v_{th,e}>$")
axs[1].plot(dflowavg['ve_xx'],dflowavg['we'][0,0,:],color='b',ls=':',label=r"$<v_{z,e}/v_{th,e}>$")
axs[0].legend()
axs[1].legend()
axs[0].grid()
axs[1].grid()
axs[-1].set_xlabel('x/d_i')
plt.savefig('figures/1dplots/1dflowfig_shockframe.png',format='png',dpi=300)
plt.close()
