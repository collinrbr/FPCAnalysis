import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa

def plot_1d_den(dden,yindex,flnm,isaveraged = False):    
    fieldcoord = np.asarray(dden['dens_xx'][:])
    fieldvals = [(np.asarray(dden['dens'][0,yindex,:]))]

    if(isaveraged):
        legendlbls = [r'$<n>$']
    else:
        legendlbls = [r'$n$']
    colors = ['r','g','b']
    linestyles = ['-','-.',':']

    plt.figure(figsize=(20,4))
    plt.style.use('cb.mplstyle')

    for _i in range(0,1):
        plt.plot(fieldcoord,fieldvals[_i],label=legendlbls[_i],ls=linestyles[_i])

    plt.xlabel(r'$x/d_i$')
    plt.legend()
    plt.savefig(flnm+'.png',format='png',dpi=300,bbox_inches='tight')
    plt.close()

#------------------------------------------------------------------------------------------------------------------------------------
# Begin script
#------------------------------------------------------------------------------------------------------------------------------------
#user params
framenum = '700' #frame to make figure of (should be a string)
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]
inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'

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

inputs = ld.load_input(inputpath)
beta0 = aa.compute_beta0(params,inputs)

##compute shock velocity and boost to shock rest frame
#dfields_many_frames = {'frame':[],'dfields':[]}
#for _num in frames:
#    num = int(_num)
#    d = ld.load_fields(flpath,_num,normalizeFields=normalize)
#    dfields_many_frames['dfields'].append(d)
#    dfields_many_frames['frame'].append(num)
#vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)
#dcurr = ld.load_current(flpath, framenum,normalizeFields=normalize)
#import copy
#dcurr_simframe = copy.deepcopy(dcurr)
#dcurr = ft.shift_curr(dcurr, vshock, beta0)

import os #TODO: move to top (do the same for all similar blocks)
os.system('mkdir figures')
os.system('mkdir figures/1dplots')

yindex = 0

print("Making 1d transverse averaged density plot...")
plot_1d_den(dden,yindex,'figures/1dplots/1davgden',isaveraged=True)

