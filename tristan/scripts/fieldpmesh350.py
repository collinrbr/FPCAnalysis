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

def pmeshsuperplot_old(dfields,dflow,flnmspmesh,btot0,etot0,xlim0=None,xlim1=None,usetotallabels=True):
    zzindex = 0

    #make plots of fields
    fig, axs = plt.subplots(15,figsize=(10*4,15*2),sharex=True)

    fig.subplots_adjust(hspace=.1)

    plt.style.use('cb.mplstyle')

    #compute Btot
    btot = np.zeros(dfields['bx'].shape)
    for _i in range(0,len(btot)):
        for _j in range(0,len(btot[_i])):
            for _k in range(0,len(btot[_i][_j])):
                btot[_i,_j,_k] = np.linalg.norm([dfields['bx'][_i,_j,_k],dfields['by'][_i,_j,_k],dfields['bz'][_i,_j,_k]])
    
    #compute Etot TODO: speed this block up
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
    fig.colorbar(ey_im, ax=axs[5])

    #Ez
    ez_im = axs[6].pcolormesh(dfields['ez_xx'], dfields['ez_yy'], dfields['ez'][zzindex,:,:]/btot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(ez_im, ax=axs[6])

    #Etot
    etot_im = axs[7].pcolormesh(dfields['ex_xx'], dfields['ex_yy'], etot[zzindex,:,:]/etot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(etot_im, ax=axs[7])

    #den
    den_im = axs[8].pcolormesh(dfields['dens_xx'],dfields['dens_yy'],dfields['dens'][zzindex,:,:], cmap="Greens", shading="gouraud")
    fig.colorbar(den_im, ax=axs[8])

    #uxi
    vscale = np.abs(np.mean(dflow['ui'])*3)
    ui_im = axs[9].pcolormesh(dflow['ui_xx'],dflow['ui_yy'],dflow['ui'][zzindex,:,:], vmin=-vscale, vmax=vscale, cmap="PuOr", shading="gouraud")
    fig.colorbar(ui_im, ax=axs[9])

    #uyi
    vscale = np.abs(np.mean(dflow['vi'])*3)
    vi_im = axs[10].pcolormesh(dflow['vi_xx'],dflow['vi_yy'],dflow['vi'][zzindex,:,:], vmin=-vscale, vmax=vscale, cmap="PuOr", shading="gouraud")
    fig.colorbar(vi_im, ax=axs[10])

    #uzi
    vscale = np.abs(np.mean(dflow['wi'])*3)
    wi_im = axs[11].pcolormesh(dflow['wi_xx'],dflow['wi_yy'],dflow['wi'][zzindex,:,:], vmin=-vscale, vmax=vscale, cmap="PuOr", shading="gouraud")
    fig.colorbar(wi_im, ax=axs[11])

    #uxe
    vscale = np.abs(np.mean(dflow['ue'])*3)
    ue_im = axs[12].pcolormesh(dflow['ue_xx'],dflow['ue_yy'],dflow['ue'][zzindex,:,:], vmin=-vscale, vmax=vscale, cmap="PuOr", shading="gouraud")
    fig.colorbar(ue_im, ax=axs[12])

    #uye
    vscale = np.abs(np.mean(dflow['ve'])*3)
    ve_im = axs[13].pcolormesh(dflow['ve_xx'],dflow['ve_yy'],dflow['ve'][zzindex,:,:], vmin=-vscale, vmax=vscale, cmap="PuOr", shading="gouraud")
    fig.colorbar(ve_im, ax=axs[13])

    #uze
    vscale = np.abs(np.mean(dflow['we'])*3)
    we_im = axs[14].pcolormesh(dflow['we_xx'],dflow['we_yy'],dflow['we'][zzindex,:,:], vmin=-vscale, vmax=vscale, cmap="PuOr", shading="gouraud")
    fig.colorbar(we_im, ax=axs[14])

    #print axes labels
    axs[-1].set_xlabel('$x$ ($d_i$)')
    for _i in range(0,len(axs)):
        axs[_i].set_ylabel('$y$ ($d_i$)')

    #set xlim
    if(xlim0 != None and xlim1 != None):
        axs[-1].set_xlim(xlim0,xlim1)

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


def pmeshsuperplot(dfields,dflow,flnmspmesh,btot0,etot0,xlim0=None,xlim1=None,usetotallabels=True):
    zzindex = 0

    #make plots of fields
    fig, axs = plt.subplots(4,4,figsize=(50,16),sharex=True)

    fig.subplots_adjust(hspace=.1)

    plt.style.use('cb.mplstyle')

    #compute Btot
    btot = np.zeros(dfields['bx'].shape)
    for _i in range(0,len(btot)):
        for _j in range(0,len(btot[_i])):
            for _k in range(0,len(btot[_i][_j])):
                btot[_i,_j,_k] = np.linalg.norm([dfields['bx'][_i,_j,_k],dfields['by'][_i,_j,_k],dfields['bz'][_i,_j,_k]])

    #compute Etot TODO: speed this block up
    etot = np.zeros(dfields['ex'].shape)
    for _i in range(0,len(etot)):
        for _j in range(0,len(etot[_i])):
            for _k in range(0,len(etot[_i][_j])):
                etot[_i,_j,_k] = np.linalg.norm([dfields['ex'][_i,_j,_k],dfields['ey'][_i,_j,_k],dfields['ez'][_i,_j,_k]])

    #Bx
    bx_im = axs[0,0].pcolormesh(dfields['bx_xx'], dfields['bx_yy'], dfields['bx'][zzindex,:,:]/btot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(bx_im, ax=axs[0,0])

    #By
    by_im = axs[0,1].pcolormesh(dfields['by_xx'], dfields['by_yy'], dfields['by'][zzindex,:,:]/btot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(by_im, ax=axs[0,1])

    #Bz
    bz_im = axs[0,2].pcolormesh(dfields['bz_xx'], dfields['bz_yy'], dfields['bz'][zzindex,:,:]/btot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(bz_im, ax=axs[0,2])

    #Btot
    btot_im = axs[0,3].pcolormesh(dfields['bx_xx'], dfields['bx_yy'], btot[zzindex,:,:]/btot0, cmap="magma", shading="gouraud")
    fig.colorbar(btot_im, ax=axs[0,3])

    #Ex
    ex_im = axs[1,0].pcolormesh(dfields['ex_xx'], dfields['ex_yy'], dfields['ex'][zzindex,:,:]/etot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(ex_im, ax=axs[1,0])

    #Ey
    ey_im = axs[1,1].pcolormesh(dfields['ey_xx'], dfields['ey_yy'], dfields['ey'][zzindex,:,:]/etot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(ey_im, ax=axs[1,1])

    #Ez
    ez_im = axs[1,2].pcolormesh(dfields['ez_xx'], dfields['ez_yy'], dfields['ez'][zzindex,:,:]/btot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(ez_im, ax=axs[1,2])

    #Etot
    etot_im = axs[1,3].pcolormesh(dfields['ex_xx'], dfields['ex_yy'], etot[zzindex,:,:]/etot0, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(etot_im, ax=axs[1,3])

    #den
    den_im = axs[2,0].pcolormesh(dfields['dens_xx'],dfields['dens_yy'],dfields['dens'][zzindex,:,:], cmap="Greens", shading="gouraud")
    fig.colorbar(den_im, ax=axs[2,0])

    #uxi
    vscale = np.abs(np.mean(dflow['ui'])*3)
    ui_im = axs[2,1].pcolormesh(dflow['ui_xx'],dflow['ui_yy'],dflow['ui'][zzindex,:,:], vmin=-vscale, vmax=vscale, cmap="PuOr", shading="gouraud")
    fig.colorbar(ui_im, ax=axs[2,1])

    #uyi
    vscale = np.abs(np.mean(dflow['vi'])*3)
    vi_im = axs[2,2].pcolormesh(dflow['vi_xx'],dflow['vi_yy'],dflow['vi'][zzindex,:,:], vmin=-vscale, vmax=vscale, cmap="PuOr", shading="gouraud")
    fig.colorbar(vi_im, ax=axs[2,2])

    #uzi
    vscale = np.abs(np.mean(dflow['wi'])*3)
    wi_im = axs[2,3].pcolormesh(dflow['wi_xx'],dflow['wi_yy'],dflow['wi'][zzindex,:,:], vmin=-vscale, vmax=vscale, cmap="PuOr", shading="gouraud")
    fig.colorbar(wi_im, ax=axs[2,3])

    #uxe
    vscale = np.abs(np.mean(dflow['ue'])*3)
    ue_im = axs[3,1].pcolormesh(dflow['ue_xx'],dflow['ue_yy'],dflow['ue'][zzindex,:,:], vmin=-vscale, vmax=vscale, cmap="PuOr", shading="gouraud")
    fig.colorbar(ue_im, ax=axs[3,1])

    #uye
    vscale = np.abs(np.mean(dflow['ve'])*3)
    ve_im = axs[3,2].pcolormesh(dflow['ve_xx'],dflow['ve_yy'],dflow['ve'][zzindex,:,:], vmin=-vscale, vmax=vscale, cmap="PuOr", shading="gouraud")
    fig.colorbar(ve_im, ax=axs[3,2])

    #uze
    vscale = np.abs(np.mean(dflow['we'])*3)
    we_im = axs[3,3].pcolormesh(dflow['we_xx'],dflow['we_yy'],dflow['we'][zzindex,:,:], vmin=-vscale, vmax=vscale, cmap="PuOr", shading="gouraud")
    fig.colorbar(we_im, ax=axs[3,3])

    #print axes labels
    axs[3,3].set_xlabel('$x/d_i$')
    axs[3,2].set_xlabel('$x/d_i$')
    axs[3,1].set_xlabel('$x/d_i$')
    axs[3,0].set_xlabel('$x/d_i$')
    for _i in range(0,len(axs)):
        for _j in range(0,len(axs[_i])):
            axs[_i,_j].set_ylabel('$y/d_i$')
            axs[_i,_j].set_xlim(xlim0,xlim1)

    #print labels of each plot
    import matplotlib.patheffects as PathEffects
    if(usetotallabels):
        pltlabels = ['$B_x/B_{0}$','$B_y/B_{0}$','$B_z/B_{0}$','$|B|/B_{0}$','$E_x/E_{0}$','$E_y/E_{0}$','$E_z/E_{0}$','$|E|/E_{0}$','$n/n_{0}$','$v_{x,i}/v_{ti}$','$v_{y,i}/v_{ti}$','$v_{z,i}/v_{ti}$','','$v_{x,e}/v_{te}$','$v_{y,e}/v_{te}$','$v_{z,e}/v_{te}$']
    else:
        pltlabels = ['$\delta B_x/B_{0}$','$\delta B_y/B_{0}$','$\delta B_z/B_{0}$','$|\delta B|/B_{0}$','$\delta E_x/E_{0}$','$\delta E_y/E_{0}$','$\delta E_z/E_{0}$','$|\delta E|/E_{0}$','$n/n_{0}$','$v_{x,i}/v_{ti}$','$v_{y,i}/v_{ti}$','$v_{z,i}/v_{ti}$','','$v_{x,e}/v_{te}$','$v_{y,e}/v_{te}$','$v_{z,e}/v_{te}$']
    _xtext = 9.5#dfields['bx_xx'][int(len(dfields['bx_xx'])*.1)]
    _ytext = dfields['bx_yy'][int(len(dfields['bx_yy'])*.6)]
    for _i in range(0,len(axs)):
        for _j in range(0,len(axs[_i])):
            _ival = _i*4+_j
            _txt = axs[_i,_j].text(_xtext,_ytext,pltlabels[_ival],color='white',fontsize=24)
            _txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    plt.subplots_adjust(hspace=.25,wspace=.075)
    plt.savefig(flnmspmesh+'.png',format='png',dpi=400,bbox_inches='tight')
    plt.close()

#------------------------------------------------------------------------------------------------------------------------------------
# Begin script
#------------------------------------------------------------------------------------------------------------------------------------
#user params
framenum = '350' #frame to make figure of (should be a string)
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
dfields_many_frames = {'frame':[],'dfields':[]}
for _num in frames:
    num = int(_num)
    d = ld.load_fields(flpath,_num,normalizeFields=normalize)
    dfields_many_frames['dfields'].append(d)
    dfields_many_frames['frame'].append(num)
vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)
vshock = 1.5

dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

dpar_elec, dpar_ion = ld.load_particles(flpath,framenum,normalizeVelocity=normalize)
inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
inputs = ld.load_input(inputpath)
beta0 = aa.compute_beta0(params,inputs)

if(loadflow):
    filein = open(dflowflnm, 'rb')
    dflow = pickle.load(filein)
    filein.close()
else:
    dpar_ion = ft.shift_particles(dpar_ion, vshock, beta0, params['mi']/params['me'], isIon=True)
    dpar_elec = ft.shift_particles(dpar_elec, vshock, beta0, params['mi']/params['me'], isIon=False)
    dflow = aa.compute_dflow(dfields, dpar_ion, dpar_elec)

#compute Btot
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

#dummy figure to load font (needed bc of matplot lib bug- mpl style doesnt always work the first time)
#dummmy plot to load font (fixes weird mpl bug)
plt.figure()
plt.style.use("cb.mplstyle")
plt.plot([0,1],[0,1])
plt.close()

import os
os.system('mkdir figures')
os.system('mkdir figures/pmeshes350')

flnmpmesh = 'figures/pmeshes350/nonadiaperppmesh'
print("Making pmesh of total fields...",flpath)
pmeshsuperplot(dfields,dflow,flnmpmesh,btot0,etot0,xlim0=0,xlim1=12,usetotallabels=True)

dfluc = aa.remove_average_fields_over_yz(dfields)
flnmpmesh = 'figures/pmeshes350/nonadiaperppmeshdfluc'
print("Making pmesh of fluc fields (shock rest frame)...",flpath)
pmeshsuperplot(dfluc,dflow,flnmpmesh,btot0,etot0,xlim0=0,xlim1=12,usetotallabels=False)