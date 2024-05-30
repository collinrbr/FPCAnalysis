

#TODO:

#TODO: test by plottinng inverse of unfiltered wave andn comparing to plot ofjust fluc fields

#TODO: test by plotting sinngle filtered mode

#TODO: compute FPC using inverse fields

def pmeshsuperplot(dfields,dfluclowpass,dfluchighpass,flnmspmesh,btot0,etot0):
    zzindex = 0

    #make plots of fields
    fig, axs = plt.subplots(2,figsize=(10,2*2),sharex=True)

    fig.subplots_adjust(hspace=.1)

    plt.style.use('cb.mplstyle')

    etotlp = np.zeros(dfields['ex'].shape)
    for _i in range(0,len(etotlp)):
        for _j in range(0,len(etotlp[_i])):
            for _k in range(0,len(etotlp[_i][_j])):
                etotlp[_i,_j,_k] = np.linalg.norm([dfluclowpass['ex'][_i,_j,_k],dfluclowpass['ey'][_i,_j,_k],dfluclowpass['ez'][_i,_j,_k]])
    
    etothp = np.zeros(dfields['ex'].shape)
    for _i in range(0,len(etothp)):
        for _j in range(0,len(etothp[_i])):
            for _k in range(0,len(etothp[_i][_j])):
                etothp[_i,_j,_k] = np.linalg.norm([dfluchighpass['ex'][_i,_j,_k],dfluchighpass['ey'][_i,_j,_k],dfluchighpass['ez'][_i,_j,_k]])

    #Etot low pass
    etotlp_im = axs[0].pcolormesh(dfields['bx_xx'], dfields['bx_yy'], etotlp[zzindex,:,:]/etot0, cmap="magma", shading="gouraud")
    fig.colorbar(etotlp_im, ax=axs[0])

    #Etot highpass
    etothp_im = axs[1].pcolormesh(dfields['bx_xx'], dfields['bx_yy'], etothp[zzindex,:,:]/etot0, cmap="magma", shading="gouraud")
    fig.colorbar(etothp_im, ax=axs[1])

    #print axes labels
    axs[-1].set_xlabel('$x/d_i$')
    for _i in range(0,len(axs)):
        axs[_i].set_ylabel('$y/d_i$')
        axs[_i].grid()
        axs[_i].set_xlim(5,10)

    #print labels of each plot
    import matplotlib.patheffects as PathEffects
    pltlabels = ['$B_x/B_{0}$','$B_y/B_{0}$','$B_z/B_{0}$','$|B|/B_{0}$','$E_x/E_{0}$','$E_y/E_{0}$','$E_z/E_{0}$','$|E|/E_{0}$','$n/n_{0}$','$v_{x,i}/v_{ti}$','$v_{y,i}/v_{ti}$','$v_{z,i}/v_{ti}$','$v_{x,e}/v_{te}$','$v_{y,e}/v_{te}$','$v_{z,e}/v_{te}$']
    pltlabels = ['$|\mathbf{E}^{k_{||} d_i < 15}|/E_0$','$|\mathbf{E}^{k_{||} d_i > 15}|/E_0$']
    _xtext = dfields['bx_xx'][int(len(dfields['bx_xx'])*.1)]
    _ytext = dfields['bx_yy'][int(len(dfields['bx_yy'])*.6)]
    for _i in range(0,len(axs)):
        _txt = axs[_i].text(_xtext,_ytext,pltlabels[_i],color='white',fontsize=24)
        _txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    plt.savefig(flnmspmesh+'.png',format='png',dpi=1200,bbox_inches='tight')
    plt.close()


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
import lib.arrayaux as ao #array operations

import os
os.system('mkdir figures')
os.system('mkdir figures/pmeshes')

flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
framenum = '700' #frame to make figure of (should be a string)

params = ld.load_params(flpath,framenum)
dt = params['c']/params['comp'] #in units of wpe
c = params['c']
stride = 100
dt,c = aa.norm_constants(params,dt,c,stride)

#Load original fields- and truncate/avg like we did before WFT transforming
normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)
vshock = 1.5625375379202415
print("Warning: using hard coded value of vshock to save computational time...")
print("vshock = ",vshock)
dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame
dfluc = aa.remove_average_fields_over_yz(dfields)

btot = np.zeros(dfields['bx'].shape)
for _i in range(0,len(btot)):
    for _j in range(0,len(btot[_i])):
        for _k in range(0,len(btot[_i][_j])):
                btot[_i,_j,_k] = np.linalg.norm([dfields['bx'][_i,_j,_k],dfields['by'][_i,_j,_k],dfields['bz'][_i,_j,_k]])
btot0 = np.mean(btot[:,:,-1])

etot = np.zeros(dfields['ex'].shape)
for _i in range(0,len(etot)):
    for _j in range(0,len(etot[_i])):
        for _k in range(0,len(etot[_i][_j])):
            etot[_i,_j,_k] = np.linalg.norm([dfields['ex'][_i,_j,_k],dfields['ey'][_i,_j,_k],dfields['ez'][_i,_j,_k]])
etot0 = np.mean(etot[:,:,-1])

startidx_xx = ao.find_nearest(dfields['ex_xx'],5.)
startidx_yy = 0
startidx_zz = 0
endidx_xx = ao.find_nearest(dfields['ex_xx'],10.)
endidx_yy = len(dfields['ex_yy'])
endidx_zz = len(dfields['ex_zz'])
startidxs = [startidx_zz,startidx_yy,startidx_xx]
endidxs = [endidx_zz,endidx_yy,endidx_xx]
dfields = ao.subset_dict(dfields,startidxs,endidxs,planes=['z','y','x'])
dfluc = ao.subset_dict(dfluc,startidxs,endidxs,planes=['z','y','x'])
dfields = ao.avg_dict(dfields,binidxsz=[1,1,10],planes=['z','y','x'])
dfluc = ao.avg_dict(dfluc,binidxsz=[1,1,10],planes=['z','y','x'])

print("keys: ", dfields.keys())

#picklename = ['fftnofiltfilteredfields.pickle','fftfilterabovefilteredfields.pickle','fftfilterbelowfilteredfields.pickle']

#dummy figure to load font (needed bc of matplot lib bug- mpl style doesnt always work the first time)
#dummmy plot to load font (fixes weird mpl bug)
plt.figure()
plt.style.use("cb.mplstyle")
plt.plot([0,1],[0,1])
plt.close()

filteredflnm = 'analysisfiles/fftfilterabovefilteredfields.pickle'
filein = open(filteredflnm, 'rb')
dfluchighpass = pickle.load(filein)
dfluchighpass['Vframe_relative_to_sim']=0.
filein.close()

filteredflnm = 'analysisfiles/fftfilterbelowfilteredfields.pickle'
filein = open(filteredflnm, 'rb')
dfluclowpass = pickle.load(filein)
dfluclowpass['Vframe_relative_to_sim']=0.
filein.close()

flnmspmesh = 'figures/filteredefieldsfig.png'
pmeshsuperplot(dfields,dfluclowpass,dfluchighpass,flnmspmesh,btot0,etot0)
