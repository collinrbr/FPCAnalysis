import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa

def plot_1d_field(dfields,key,yindex,flnm):
    fieldcoord = np.asarray(dfields[fieldkey+'_xx'][:])
    fieldval = np.asarray(dfields[fieldkey][0,yindex,:])

    plt.figure(figsize=(20,10))
    plt.style.use('cb.mplstyle')
    if(key == 'ey'):
        plt.ylabel(r'$<E_y/E_0>$')
    else:
        plt.ylabel(key)
    plt.plot(fieldcoord,fieldval)
    plt.grid()
    plt.savefig(flnm+'.png',format='png')
    plt.close()    

def plot_1d_fields(dfields,yindex,flnm,isaveraged = False):    
    fieldcoord = np.asarray(dfields['ex_xx'][:])
    fieldvals = []
    for fieldkey in ['ex','ey','ez','bx','by','bz']:
        fieldvals.append(np.asarray(dfields[fieldkey][0,yindex,:]))

    if(isaveraged):
        legendlbls = [r'$<E_x>$',r'$<E_y>$',r'$<E_z>$',r'$<B_x>$',r'$<B_y>$',r'$<B_z>$']
    else:
        legendlbls = [r'$E_x$',r'$E_y$',r'$E_z$',r'$B_x$',r'$B_y$',r'$B_z$']
    colors = ['r','g','b','r','g','b']
    linestyles = ['-','-.',':','-','-.',':']

    fig, axs = plt.subplots(2,1,figsize=(20,8))
    plt.style.use('cb.mplstyle')

    for _i in range(0,3):
        axs[0].plot(fieldcoord,fieldvals[_i],label=legendlbls[_i],ls=linestyles[_i])

    for _i in range(3,6):
        axs[1].plot(fieldcoord,fieldvals[_i],label=legendlbls[_i],ls=linestyles[_i])

    axs[0].legend()
    axs[1].legend()
    plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
    plt.close()

#------------------------------------------------------------------------------------------------------------------------------------
# Begin script
#------------------------------------------------------------------------------------------------------------------------------------
#user params
framenum = '700' #frame to make figure of (should be a string)
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]

normalize = False
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)
dden = ld.load_den(flpath,framenum)
for _key in dden.keys():
    dfields[_key] = dden[_key]

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

dfavg_simframe = aa.get_average_fields_over_yz(dfields)
#dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame
dfavg = aa.get_average_fields_over_yz(dfields)

import os
os.system('mkdir figures')
os.system('mkdir figures/1dplots')

yindex = 0

print("Making 1d transverse averaged fields plot...")
plot_1d_fields(dfavg,yindex,'figures/1dplots/1davgfields_shockrestframe_nonorm.png',isaveraged=True)
plot_1d_fields(dfavg_simframe,yindex,'figures/1dplots/1davgfields_simframe_nonorm.png',isaveraged=True)

