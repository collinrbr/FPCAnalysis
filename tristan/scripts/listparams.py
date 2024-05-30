import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa
import lib.plotcoraux as pfpc
import lib.fpcaux as fpc

#user params
framenum = '700' #frame to make figure of (should be a string)
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]
vmaxion = 30
dvion = 1.
vmaxelec = 15
dvelec = 1.


inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
inputs = ld.load_input(inputpath,verbose=True)
print(inputs)

#### WORK IN PROGRESSS

normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)
dden = ld.load_den(flpath,framenum)
for _key in dden.keys():
    dfields[_key] = dden[_key]

params = ld.load_params(flpath,framenum)
dt = params['c']/params['comp'] #in units of wpe
c = params['c']
stride = 100
dt,c = aa.norm_constants(params,dt,c,stride)

print('params comp sigma: ', params['comp'], params['sigma'])

print('Time to frame 0: ', 360000/650800*6.66) #360000 is the number of time steps until frame 0, 650800 is the total time steps (time steps != frames as we only output every 100th step), 6.66 is in units inv omega ci and is the approximate total runtime is from Aaron's message (see direct message thread with jimmy and aaron)
print('Frame: ', framenum, ' dt: ', dt, ' dt*framenum: ', dt*float(framenum))
print('Time since init: ', 360000/650800*6.66 + dt*float(framenum))

print(params.keys())
inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
inputs = ld.load_input(inputpath)
beta0 = aa.compute_beta0(params,inputs)

print("beta0,",beta0)

vti0 = np.sqrt(params['delgam'])#Note: velocity is in units γV_i/c so we do not include '*params['c']'
vte0 = np.sqrt(params['mi']/params['me'])*vti0 #WARNING: THIS ASSUME Ti/Te = 1, TODO: don't assume Ti/Te = 1

print("vte0: ", vte0)

print("Te: ", vte0**2*params['me']/2.)
print("Ti: ", vti0**2*params['mi']/2.)

print('vae: ',vte0/(beta0/2.)) #assumes Te=Ti
            #filter xx
print('va: ', vti0/(beta0/2.)) #assumes Te=Ti 
massratio = ld.load_params(flpath,framenum)['mi']/ld.load_params(flpath,framenum)['me']
print('mass ratio: ', massratio)

#TODO: shock velocity (load gamma0, convert to vti like in load aux, then to va using beta)
uinj_voverc = inputs['gamma0']
uinj_vth = uinj_voverc/np.sqrt(params['delgam'])
uinj_va = uinj_vth*np.sqrt(beta0)*params['c'] #assumes me << mi
print('uinj_vth (units of v/vthi): ', uinj_vth)
print('u injection (units of v/va): ', uinj_va)

print('shock normal angle: ',inputs['btheta'])                   

print("computing shock velocity in frame...")
normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)
dden = ld.load_den(flpath,framenum)
for _key in dden.keys():
    dfields[_key] = dden[_key]

print("dfields yy length: ", dfields['ex_yy'][-1])

params = ld.load_params(flpath,framenum)
dt = params['c']/params['comp'] #in units of wpe
c = params['c']

print('c from input', c)
stride = 100
dt,c = aa.norm_constants(params,dt,c,stride)

print('c',  c)

sigma_ion = params['sigma']*params['me']/params['mi']
print("sigma_ion; np.sqrt(sigma_ion) ; (va_c)", sigma_ion, np.sqrt(sigma_ion),  np.sqrt(sigma_ion)*params['c'])

#TODO: double check if I should divide here!!! print('vth_c: ', np.sqrt(sigma_ion)/np.sqrt(beta0)) 

print('va_over_c: ',np.sqrt(sigma_ion))

#TODO: omega ce over omege pe
print("wpe_over_wce = 1./(np.sqrt(sigma_ion)*np.sqrt(params['mi']/params['me']))", 1./(np.sqrt(sigma_ion)*np.sqrt(params['mi']/params['me'])))

#compute shock velocity and boost to shock rest frame
dfields_many_frames = {'frame':[],'dfields':[]}
for _num in frames:
    num = int(_num)
    d = ld.load_fields(flpath,_num,normalizeFields=normalize)
    dfields_many_frames['dfields'].append(d)
    dfields_many_frames['frame'].append(num)
vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)

print("shock vel (simulation frame): ", vshock)
