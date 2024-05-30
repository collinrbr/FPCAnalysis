

#TODO:

#TODO: test by plottinng inverse of unfiltered wave andn comparing to plot ofjust fluc fields

#TODO: test by plotting sinngle filtered mode

#TODO: compute FPC using inverse fields

def pmeshsuperplot(dfields,flnmspmesh,btot0,etot0):
    zzindex = 0

    #make plots of fields
    fig, axs = plt.subplots(8,figsize=(10,15*2),sharex=True)

    fig.subplots_adjust(hspace=.1)

    plt.style.use('cb.mplstyle')

    #compute Btot
    btot = np.zeros(dfields['bx'].shape)
    for _i in range(0,len(btot)):
        for _j in range(0,len(btot[_i])):
            for _k in range(0,len(btot[_i][_j])):
                btot[_i,_j,_k] = np.linalg.norm([dfields['bx'][_i,_j,_k],dfields['by'][_i,_j,_k],dfields['bz'][_i,_j,_k]])

    etot = np.zeros(dfields['ex'].shape)
    for _i in range(0,len(etot)):
        for _j in range(0,len(etot[_i])):
            for _k in range(0,len(etot[_i][_j])):
                etot[_i,_j,_k] = np.linalg.norm([dfields['ex'][_i,_j,_k],dfields['ey'][_i,_j,_k],dfields['ez'][_i,_j,_k]])

    #Bx
    bx_im = axs[0].pcolormesh(dfields['bx_xx'], dfields['bx_yy'], dfields['bx'][zzindex,:,:]/btot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(bx_im, ax=axs[0])

    #By
    by_im = axs[1].pcolormesh(dfields['by_xx'], dfields['by_yy'], dfields['by'][zzindex,:,:]/btot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(by_im, ax=axs[1])

    #Bz
    bz_im = axs[2].pcolormesh(dfields['bz_xx'], dfields['bz_yy'], dfields['bz'][zzindex,:,:]/btot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(bz_im, ax=axs[2])

    #Btot
    btot_im = axs[3].pcolormesh(dfields['bx_xx'], dfields['bx_yy'], btot[zzindex,:,:]/btot0, cmap="magma", shading="gouraud")
    fig.colorbar(btot_im, ax=axs[3])

    #Ex
    ex_im = axs[4].pcolormesh(dfields['ex_xx'], dfields['ex_yy'], dfields['ex'][zzindex,:,:]/etot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(ex_im, ax=axs[4])

    #Ey
    ey_im = axs[5].pcolormesh(dfields['ey_xx'], dfields['ey_yy'], dfields['ey'][zzindex,:,:]/etot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(bx_im, ax=axs[5])

    #Ez
    ez_im = axs[6].pcolormesh(dfields['ez_xx'], dfields['ez_yy'], dfields['ez'][zzindex,:,:]/btot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(bx_im, ax=axs[6])

    #Etot
    etot_im = axs[7].pcolormesh(dfields['bx_xx'], dfields['bx_yy'], etot[zzindex,:,:]/etot0, cmap="magma", shading="gouraud")
    fig.colorbar(etot_im, ax=axs[7])

    #print axes labels
    axs[-1].set_xlabel('$x$ ($d_i$)')
    for _i in range(0,len(axs)):
        axs[_i].set_ylabel('$y$ ($d_i$)')

    #print labels of each plot
    import matplotlib.patheffects as PathEffects
    pltlabels = ['$B_x/B_{0}$','$B_y/B_{0}$','$B_z/B_{0}$','$|B|/B_{0}$','$E_x/E_{0}$','$E_y/E_{0}$','$E_z/E_{0}$','$|E|/E_{0}$','$n/n_{0}$','$v_{x,i}/v_{ti}$','$v_{y,i}/v_{ti}$','$v_{z,i}/v_{ti}$','$v_{x,e}/v_{te}$','$v_{y,e}/v_{te}$','$v_{z,e}/v_{te}$']
    _xtext = dfields['bx_xx'][int(len(dfields['bx_xx'])*.75)]
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
os.system('mkdir analysisfiles')

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

print("starting inverse...")
flnmprefix = 'fftnofilt'
filterabove = False
kycutoff = 15.
dfields_inv = aa.yz_fft_filter(dfluc,kycutoff,filterabove,dontfilter=True,verbose=True)
with open('analysisfiles/'+flnmprefix+'filteredfields.pickle', 'wb') as handle:
    pickle.dump(dfields_inv, handle, protocol=pickle.HIGHEST_PROTOCOL)
pmeshsuperplot(dfields_inv,'figures/pmeshes/'+flnmprefix+'afterinvfields',btot0,etot0)

flnmprefix = 'fftfilterbelow'
filterabove = False
kycutoff = 15.
dfields_inv = aa.yz_fft_filter(dfluc,kycutoff,filterabove,dontfilter=False,verbose=True)
with open('analysisfiles/'+flnmprefix+'filteredfields.pickle', 'wb') as handle:
    pickle.dump(dfields_inv, handle, protocol=pickle.HIGHEST_PROTOCOL)
pmeshsuperplot(dfields_inv,'figures/pmeshes/'+flnmprefix+'afterinvfields',btot0,etot0)

flnmprefix = 'fftfilterabove'
filterabove = True
kycutoff = 15.
dfields_inv = aa.yz_fft_filter(dfluc,kycutoff,filterabove,dontfilter=False,verbose=True)
with open('analysisfiles/'+flnmprefix+'filteredfields.pickle', 'wb') as handle:
    pickle.dump(dfields_inv, handle, protocol=pickle.HIGHEST_PROTOCOL)
pmeshsuperplot(dfields_inv,'figures/pmeshes/'+flnmprefix+'afterinvfields',btot0,etot0)

#TODO: rename files as we now filter total fields
flnmprefix = 'fftnofilttot'
filterabove = False
kycutoff = 15.
dfields_inv = aa.yz_fft_filter(dfields,kycutoff,filterabove,dontfilter=True,verbose=True)
with open('analysisfiles/'+flnmprefix+'filteredfields.pickle', 'wb') as handle:
    pickle.dump(dfields_inv, handle, protocol=pickle.HIGHEST_PROTOCOL)
pmeshsuperplot(dfields_inv,'figures/pmeshes/'+flnmprefix+'afterinvfields',btot0,etot0)

flnmprefix = 'fftfilterbelowtot'
filterabove = False
kycutoff = 15.
dfields_inv = aa.yz_fft_filter(dfields,kycutoff,filterabove,dontfilter=False,verbose=True)
with open('analysisfiles/'+flnmprefix+'filteredfields.pickle', 'wb') as handle:
    pickle.dump(dfields_inv, handle, protocol=pickle.HIGHEST_PROTOCOL)
pmeshsuperplot(dfields_inv,'figures/pmeshes/'+flnmprefix+'afterinvfields',btot0,etot0)

flnmprefix = 'fftfilterabovetot'
filterabove = True
kycutoff = 15.
dfields_inv = aa.yz_fft_filter(dfields,kycutoff,filterabove,dontfilter=False,verbose=True)
with open('analysisfiles/'+flnmprefix+'filteredfields.pickle', 'wb') as handle:
    pickle.dump(dfields_inv, handle, protocol=pickle.HIGHEST_PROTOCOL)
pmeshsuperplot(dfields_inv,'figures/pmeshes/'+flnmprefix+'afterinvfields',btot0,etot0)
