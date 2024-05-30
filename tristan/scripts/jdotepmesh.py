import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa
import lib.arrayaux as ao #array operations

import pickle
import os

def pmeshsuperplot(dfields,dflow,flnmspmesh,btot0,etot0,pltlabels,isIon,xlim0=None,xlim1=None):
    zzindex = 0

    #make plots of fields
    fig, axs = plt.subplots(3,figsize=(10,3*2),sharex=True)

    fig.subplots_adjust(hspace=.1)

    plt.style.use('cb.mplstyle')

    if(isIon):
        xflowkey = 'ui' 
        yflowkey = 'vi'
        zflowkey = 'wi'
    else:
        xflowkey = 'ue'
        yflowkey = 've'
        zflowkey = 'we'
    jdotexx = dfields['ex']*dflow[xflowkey]
    jdoteyy = dfields['ey']*dflow[yflowkey]
    jdotezz = dfields['ez']*dflow[zflowkey]

    absvval = np.abs(np.mean(jdotexx[0])*3)
    x_im = axs[0].pcolormesh(dfields['bx_xx'], dfields['bx_yy'], jdotexx[0], vmin=-absvval, vmax=absvval, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(x_im, ax=axs[0])

    absvval = np.abs(np.mean(jdoteyy[0])*3)
    y_im = axs[1].pcolormesh(dfields['by_xx'], dfields['by_yy'], jdoteyy[0], vmin=-absvval, vmax=absvval, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(y_im, ax=axs[1])

    absvval = np.abs(np.mean(jdoteyy[0])*3)
    z_im = axs[2].pcolormesh(dfields['bz_xx'], dfields['bz_yy'], jdotezz[0], vmin=-absvval, vmax=absvval, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(z_im, ax=axs[2])

    #print axes labels
    axs[-1].set_xlabel('$x$ ($d_i$)')
    for _i in range(0,len(axs)):
        axs[_i].set_ylabel('$y$ ($d_i$)')

    #set xlim
    if(xlim0 != None and xlim1 != None):
        axs[-1].set_xlim(xlim0,xlim1)

    #print labels of each plot
    import matplotlib.patheffects as PathEffects
    _xtext = dfields['bx_xx'][int(len(dfields['bx_xx'])*.75)]
    _ytext = dfields['bx_yy'][int(len(dfields['bx_yy'])*.6)]
    for _i in range(0,len(axs)):
        _txt = axs[_i].text(_xtext,_ytext,pltlabels[_i],color='white',fontsize=24)
        _txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    plt.savefig(flnmspmesh+'.png',format='png',dpi=1200,bbox_inches='tight')
    plt.close()

def pmeshsuperplot_indiv(dfields,dflow,plotkey,flnmspmesh,btot0,etot0,pltlabel,xlim0=None,xlim1=None):
    zzindex = 0

    #make plots of fields
    fig, axs = plt.subplots(1,figsize=(10,3*2),sharex=True)

    fig.subplots_adjust(hspace=.1)

    plt.style.use('cb.mplstyle')

    if(plotkey == 'jidotexx'):
        jdotexx = dfields['ex']*dflow['ui']
        plotvar = jdotexx
    if(plotkey == 'jidoteyy'):
        jdoteyy = dfields['ey']*dflow['vi']
        plotvar = jdoteyy
    if(plotkey == 'jidotezz'):
        jdotezz = dfields['ez']*dflow['wi']
        plotvar = jdotezz
    if(plotkey == 'jedotexx'):
        jdotexx = -dfields['ex']*dflow['ue']
        plotvar = jdotexx
    if(plotkey == 'jedoteyy'):
        jdoteyy = -dfields['ey']*dflow['ve']
        plotvar = jdoteyy
    if(plotkey == 'jedotezz'):
        jdotezz = -dfields['ez']*dflow['we']
        plotvar = jdotezz
    if(plotkey == 'jidotepar'):
        jdotexx = dfields['epar']*dflow['jpari']
        plotvar = jdotexx
    if(plotkey == 'jidoteperp1'):
        jdoteyy = dfields['eperp1']*dflow['jperp1i']
        plotvar = jdoteyy
    if(plotkey == 'jidoteperp2'):
        jdotezz = dfields['eperp2']*dflow['jperp2i']
        plotvar = jdotezz
    if(plotkey == 'jedotepar'):
        jdotexx = -dfields['epar']*dflow['jpare']
        plotvar = jdotexx
    if(plotkey == 'jedoteperp1'):
        jdoteyy = -dfields['eperp1']*dflow['jperp1e']
        plotvar = jdoteyy
    if(plotkey == 'jedoteperp2'):
        jdotezz = -dfields['eperp2']*dflow['jperp2e']
        plotvar = jdotezz
    if(plotkey == 'ui'):
        plotvar = dflow['ui']
    if(plotkey == 'vi'):
        plotvar = dflow['vi']
    if(plotkey == 'wi'):
        plotvar = dflow['wi']
    if(plotkey == 'ue'):
        plotvar = -dflow['ue']
    if(plotkey == 've'):
        plotvar = -dflow['ve']
    if(plotkey == 'we'):
        plotvar = -dflow['we']
    if(plotkey == 'jpari'):
        plotvar = dflow['jpari']
    if(plotkey == 'jperp1i'):
        plotvar = dflow['jperp1i']
    if(plotkey == 'jperp2i'):
        plotvar = dflow['jperp2i']
    if(plotkey == 'jpare'):
        plotvar = -dflow['jpare']
    if(plotkey == 'jperp1e'):
        plotvar = -dflow['jperp1e']
    if(plotkey == 'jperp2e'):
        plotvar = -dflow['jperp2e']
    if(plotkey == 'epar'):
        plotvar = dfields['epar']
    if(plotkey == 'eperp1'):
        plotvar = dfields['eperp1']
    if(plotkey == 'eperp2'):
        plotvar = dfields['eperp2']
    if(plotkey == 'ex'):
        plotvar = dfields['ex']
    if(plotkey == 'ey'):
        plotvar = dfields['ey']
    if(plotkey == 'ez'):
        plotvar = dfields['ez']
    if(plotkey == 'poyntxx'):
        poynt = np.zeros(dfields['ex'].shape)
        print(dfields['ex'].shape)
        #TODO: vectorize
        for _idxz in range(poynt.shape[0]):
            for _idxy in range(poynt.shape[1]):
                for _idxx in range(poynt.shape[2]):
                    poynt[_idxz][_idxy][_idxx] = np.cross([dfields['ex'][_idxz][_idxy][_idxx],dfields['ey'][_idxz][_idxy][_idxx],dfields['ez'][_idxz][_idxy][_idxx]],[dfields['bx'][_idxz][_idxy][_idxx],dfields['by'][_idxz][_idxy][_idxx],dfields['bz'][_idxz][_idxy][_idxx]])[0]
        plotvar = poynt
    if(plotkey == 'poyntyy'):
        poynt = np.zeros(dfields['ex'].shape)
        #TODO: vectorize
        for _idxz in range(poynt.shape[0]):
            for _idxy in range(poynt.shape[1]):
                for _idxx in range(poynt.shape[2]):
                    poynt[_idxz][_idxy][_idxx] = np.cross([dfields['ex'][_idxz][_idxy][_idxx],dfields['ey'][_idxz][_idxy][_idxx],dfields['ez'][_idxz][_idxy][_idxx]],[dfields['bx'][_idxz][_idxy][_idxx],dfields['by'][_idxz][_idxy][_idxx],dfields['bz'][_idxz][_idxy][_idxx]])[1]
        plotvar = poynt
    if(plotkey == 'poyntzz'):
        poynt = np.zeros(dfields['ex'].shape)
        #TODO: vectorize
        for _idxz in range(poynt.shape[0]):
            for _idxy in range(poynt.shape[1]):
                for _idxx in range(poynt.shape[2]):
                    poynt[_idxz][_idxy][_idxx] = np.cross([dfields['ex'][_idxz][_idxy][_idxx],dfields['ey'][_idxz][_idxy][_idxx],dfields['ez'][_idxz][_idxy][_idxx]],[dfields['bx'][_idxz][_idxy][_idxx],dfields['by'][_idxz][_idxy][_idxx],dfields['bz'][_idxz][_idxy][_idxx]])[2]
        plotvar = poynt

    #absvval = np.max(np.abs(plotvar))#(np.abs(np.mean(np.abs(plotvar))*3)
    absvval = np.abs(np.mean(np.abs(plotvar))*3)
    x_im = axs.pcolormesh(dfields['bx_xx'], dfields['bx_yy'], plotvar[0], vmin=-absvval, vmax=absvval, cmap="RdYlBu", shading="gouraud")
    fig.colorbar(x_im, ax=axs)

    #print axes labels
    axs.set_xlabel('$x / d_i$')
    axs.set_ylabel('$y / d_i$')

    #set xlim
    if(xlim0 != None and xlim1 != None):
        axs.set_xlim(xlim0,xlim1)

    #print labels of each plot
    import matplotlib.patheffects as PathEffects
    _xtext = dfields['bx_xx'][int(len(dfields['bx_xx'])*.75)]
    _ytext = dfields['bx_yy'][int(len(dfields['bx_yy'])*.6)]
    _txt = axs.text(_xtext,_ytext,pltlabel,color='white',fontsize=24)
    _txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    plt.savefig(flnmspmesh+'.png',format='png',dpi=1200,bbox_inches='tight')
    plt.close()

#------------------------------------------------------------------------------------------------------------------------------------
# Begin script
#------------------------------------------------------------------------------------------------------------------------------------
#user params
framenum = '700' #frame to make figure of (should be a string)
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]

loadflow = True
dflowflnm = 'pickles/dflow.pickle'

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

    os.system('mkdir pickles')
    picklefile = 'pickles/dflow.pickle'
    with open(picklefile, 'wb') as handle:
        pickle.dump(dflow, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved pickle to ',picklefile)
    precomputedflnm = picklefile 

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

import os
os.system('mkdir figures')
os.system('mkdir figures/jdotepmeshes')


dfluc = aa.remove_average_fields_over_yz(dfields)

startidx_xx = ao.find_nearest(dfields['ex_xx'],5.)
startidx_yy = 0
startidx_zz = 0
endidx_xx = ao.find_nearest(dfields['ex_xx'],12.)+1
endidx_yy = len(dfields['ex_yy'])
endidx_zz = len(dfields['ex_zz'])
startidxs = [startidx_zz,startidx_yy,startidx_xx]
endidxs = [endidx_zz,endidx_yy,endidx_xx]
dfields = ao.subset_dict(dfields,startidxs,endidxs,planes=['z','y','x'])
dfluc = ao.subset_dict(dfluc,startidxs,endidxs,planes=['z','y','x'])
dflow = ao.subset_dict(dflow,startidxs,endidxs,planes=['z','y','x'])
dfields = ao.avg_dict(dfields,binidxsz=[1,2,2],planes=['z','y','x'])
dflow = ao.avg_dict(dflow,binidxsz=[1,2,2],planes=['z','y','x'])
dfluc = ao.avg_dict(dfluc,binidxsz=[1,2,2],planes=['z','y','x'])


dflucbox = aa.convert_fluc_to_par(dfields,dfluc)
dfluclocal = aa.convert_fluc_to_local_par(dfields,dfluc)


plt.figure()
plt.style.use("cb.mplstyle")
plt.plot([0,1],[0,1])
plt.close()


#Make pub fig---------------------------------
zzindex = 0

fig, axs = plt.subplots(3,figsize=(10,3*2),sharex=True)

fig.subplots_adjust(hspace=.1)

plt.style.use('cb.mplstyle')

plotval1 = dfluc['ey']
plotval2 = dflucbox['epar']
plotval3 = dfluclocal['epar']


absvval = np.max(np.abs(plotval1[0]/etot0))
x_im = axs[0].pcolormesh(dfields['bx_xx'], dfields['bx_yy'], plotval1[0]/etot0, vmin=-absvval, vmax=absvval, cmap="RdYlBu", shading="gouraud")
fig.colorbar(x_im, ax=axs[0])

absvval = np.max(np.abs(plotval2[0]/etot0))
y_im = axs[1].pcolormesh(dfields['by_xx'], dfields['by_yy'], plotval2[0]/etot0, vmin=-absvval, vmax=absvval, cmap="RdYlBu", shading="gouraud")
fig.colorbar(y_im, ax=axs[1])

absvval = np.max(np.abs(plotval3[0]/etot0))
z_im = axs[2].pcolormesh(dfields['bz_xx'], dfields['bz_yy'], plotval3[0]/etot0, vmin=-absvval, vmax=absvval, cmap="RdYlBu", shading="gouraud")
fig.colorbar(z_im, ax=axs[2])

#print axes labels
axs[-1].set_xlabel('$x / d_i$')
for _i in range(0,len(axs)):
    axs[_i].set_ylabel('$y / d_i$')

#set xlim
axs[-1].set_xlim(5,12)

#print labels of each plot
import matplotlib.patheffects as PathEffects
pltlabels = [r'$E_y/E_0$',r'$E_{||}^{box}/E_0$',r'$E_{||}^{local}/E_0$']
_xtext = 8.5#dfields['bx_xx'][int(len(dfields['bx_xx'])*.75)]
_ytext = dfields['bx_yy'][int(len(dfields['bx_yy'])*.6)]
for _i in range(0,len(axs)):
    _txt = axs[_i].text(_xtext,_ytext,pltlabels[_i],color='white',fontsize=24)
    _txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

flnmspmesh = 'figures/eyvseparlocalvseparbox'
plt.savefig(flnmspmesh+'.png',format='png',dpi=400,bbox_inches='tight')
plt.close()


#end make pub fig ---------------------------


flnmpmesh = 'figures/jdotepmeshes/totjotdeion_limitvmax'
print("Making pmesh of total fields...",flpath)
isIon = True
pltlabels = [r'$j_i \cdot E_x$',r'$j_i \cdot E_y$',r'$j_i \cdot E_z$']
#pmeshsuperplot(dfields,dflow,flnmpmesh,btot0,etot0,pltlabels,isIon,xlim0=0,xlim1=12)

flnmpmesh = 'figures/jdotepmeshes/totjotdeelec_limitvmax'
print("Making pmesh of total fields...",flpath)
isIon = False
pltlabels = [r'$j_e \cdot E_x$',r'$j_e \cdot E_y$',r'$j_e \cdot E_z$']
#pmeshsuperplot(dfields,dflow,flnmpmesh,btot0,etot0,pltlabels,isIon,xlim0=0,xlim1=12)

dfluc = aa.remove_average_fields_over_yz(dfields)
flnmpmesh = 'figures/jdotepmeshes/flucjotdeion_limitvmax'
print("Making pmesh of fluc fields (shock rest frame)...",flpath)
isIon = True
pltlabels = [r'$j_i \cdot \widetilde{E_x}$',r'$j_i \cdot \widetilde{E_y}$',r'$j_i \cdot \widetilde{E_z}$']
#pmeshsuperplot(dfluc,dflow,flnmpmesh,btot0,etot0,pltlabels,isIon,xlim0=0,xlim1=12)

flnmpmesh = 'figures/jdotepmeshes/flucjotdeelec_limitvmax'
print("Making pmesh of fluc fields (shock rest frame)...",flpath)
isIon = False
pltlabels = [r'$j_e \cdot \widetilde{E_x}$',r'$j_e \cdot \widetilde{E_y}$',r'$j_e \cdot \widetilde{E_z}$']
#pmeshsuperplot(dfluc,dflow,flnmpmesh,btot0,etot0,pltlabels,isIon,xlim0=0,xlim1=12)

#TODO FAC J dot E...
dflow = aa.convert_flow_to_par(dfields,dflow)#aa.convert_flow_to_local_par(dfields,dflow)
dfields = aa.convert_to_par(dfields)#aa.convert_to_local_par(dfields)


TODO: implement this comment below!!!
#TODO: dflow should have fluid moments but not current! TODO: compute current by taking q_s n_s u_s!!!

plotkeys = ['jidotexx','jidoteyy','jidotezz','jedotexx','jedoteyy','jedotezz','jidotepar','jidoteperp1','jidoteperp2','jedotepar','jedoteperp1','jedoteperp2','ui','vi','wi','ue','ve','we','jpari','jperp1i','jperp2i','jpare','jperp1e','jperp2e','epar','eperp1','eperp2','ex','ey','ez','poyntxx','poyntyy','poyntzz']

plotlabels = [r'$j_{x,i} E_{x}$',r'$j_{y,i} E_{y}$',r'$j_{z,i} E_{z}$',
              r'$j_{x,e} E_{x}$',r'$j_{y,e} E_{y}$',r'$j_{z,e} E_{z}$',
              r'$j_{||,i} E_{||}$',r'$j_{\perp,1,i} E_{\perp,1}$',r'$j_{\perp,2,i} E_{\perp,2}$',
              r'$j_{||,e} E_{||}$',r'$j_{\perp,1,e} E_{\perp,1}$',r'$j_{\perp,2,e} E_{\perp,2}$',
              r'$j_{x,i}$',r'$j_{y,i}$',r'$j_{z,i}$',
              r'$j_{x,e}$',r'$j_{y,e}$',r'$j_{z,e}$',
              r'$j_{||,i}$',r'$j_{\perp,1,i}$',r'$j_{\perp,2,i}$',
              r'$j_{||,e}$',r'$j_{\perp,1,e}$',r'$j_{\perp,2,e}$',
              r'$E_{||}$',r'$E_{\perp,1}$',r'$E_{\perp,2}$',
              r'$E_x$',r'$E_y$',r'$E_z$',
              r'$(\mathbf{E} \times \mathbf{B})_x$',r'$(\mathbf{E} \times \mathbf{B})_y$',r'$(\mathbf{E} \times \mathbf{B})_z$']

print("TODO: fix colorbar bounds and add latexed labels")
os.system('mkdir figures')
os.system('mkdir figures/indivpmesh')
_i = 0
for plky in plotkeys:
    flnmpmesh = 'figures/indivpmesh/'+plky+'.png'
    _pltlabel = plotlabels[_i]
    pmeshsuperplot_indiv(dfields,dflow,plky,flnmpmesh,btot0,etot0,_pltlabel,xlim0=None,xlim1=None)
    _i += 1
