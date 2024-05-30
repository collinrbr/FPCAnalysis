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
import lib.wavemodeaux as wa
import lib.plotwftaux as pw

import os

os.system('mkdir figures')
os.system('mkdir analysisfiles')
os.system('mkdir figures/spectra')
os.system('mkdir figures/spectra/debug')

#TODO: add flow for ions and electrons separately 
def generate_fftwlt(dfieldsdict,dfieldstot,retstep,flnm):
    pckldict = {}

    if(retstep != 1):print("Warning: retstep != 1. This may drop crucial data to save memory. It is preferred to truncate fields or to lower fields resolution typically. Are you sure?")

    from sys import getsizeof
    print("Doing epar WFT")
    dfields = aa.convert_to_local_par(dfieldsdict)
    kz, ky, kx, eparkzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'epar',retstep=retstep)
    #print("Doing detrend epar wft")
    #filterabove = True
    #kycutoff = 15.
    #dfields_inv = aa.yz_fft_filter(dfieldstot,kycutoff,filterabove,dontfilter=True,verbose=True)
    #dfieldsdict = aa.convert_to_local_par(dfieldsdict,detrendfields=dfields_inv)
    #kz, ky, kx, epardetrendkzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'epar_detrend',retstep=retstep)
    #print("Doing detrend eperp1 wft")
    #kz, ky, kx, eperp1detrendkzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'eperp1_detrend',retstep=retstep)
    #print("Doing detrend eperp2 wft")
    #kz, ky, kx, eperp2detrendkzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'eperp2_detrend',retstep=retstep)
    
    #print("Doing detrend b epar wft")
    #kz, ky, kx, bpardetrendkzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'bpar_detrend',retstep=retstep)

    #print("Doing detrend b eperp1 wft")
    #kz, ky, kx, bperp1detrendkzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'bperp1_detrend',retstep=retstep)

    #print("Doing detrend b eperp2 wft")
    #kz, ky, kx, bperp2detrendkzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'bperp2_detrend',retstep=retstep)

    print("Starting bx WFT")
    kz, ky, kx, bxkzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'bx',retstep=retstep)
    print('kx',kx,'ky',ky,'kz',kz)
    print('sizeof bxkzkykxxx',getsizeof(bxkzkykxxx))
    #kz,ky,kx,bxkzkykxxx = reduce_kzkykxxx(kz,ky,kx,bxkzkykxxx)
    #print('kx',kx,'ky',ky,'kz',kz)
    #print('size of reduced bxkzkykxxx', getsizeof(bxkzkykxxx))
    print("Done! Doing bx WFT")
    kz, ky, kx, bykzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'by',retstep=retstep)
    #kz,ky,kx,bykzkykxxx = reduce_kzkykxxx(kz,ky,kx,bykzkykxxx)
    print("Done! Doing by WFT")
    kz, ky, kx, bzkzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'bz',retstep=retstep)
    #kz,ky,kx,bzkzkykxxx = reduce_kzkykxxx(kz,ky,kx,bzkzkykxxx)
    print("Done! Doing bz WFT")
    kz, ky, kx, exkzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'ex',retstep=retstep)
    #kz,ky,kx,exkzkykxxx = reduce_kzkykxxx(kz,ky,kx,exkzkykxxx)
    print("Done! Doing ex WFT")
    kz, ky, kx, eykzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'ey',retstep=retstep)
    #kz,ky,kx,eykzkykxxx = reduce_kzkykxxx(kz,ky,kx,eykzkykxxx)
    print("Done! Doing ey WFT")
    kz, ky, kx, ezkzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'ez',retstep=retstep)
    #kz,ky,kx,ezkzkykxxx = reduce_kzkykxxx(kz,ky,kx,ezkzkykxxx)
    print("Done! Doing ez WFT")
    kz, ky, kx, uxkzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'ux',retstep=retstep)
    #kz,ky,kx,uxkzkykxxx = reduce_kzkykxxx(kz,ky,kx,uxkzkykxxx)
    print("Done! Doing ux WFT")
    kz, ky, kx, uykzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'uy',retstep=retstep)
    #kz,ky,kx,uykzkykxxx = reduce_kzkykxxx(kz,ky,kx,uykzkykxxx)
    print("Done! Doing uy WFT")
    kz, ky, kx, uzkzkykxxx = aa.transform_field_to_kzkykxxx(dfieldsdict,'uz',retstep=retstep)
    #kz,ky,kx,uzkzkykxxx = reduce_kzkykxxx(kz,ky,kx,uzkzkykxxx)
    print("Done! Making pckl")

    pckldict = {'xx':dfieldsdict['ex_xx'],
                'kz':kz,
                'ky':ky,
                'kx':kx,
                'bxkzkykxxx':bxkzkykxxx,
                'bykzkykxxx':bykzkykxxx,
                'bzkzkykxxx':bzkzkykxxx,
                'exkzkykxxx':exkzkykxxx,
                'eykzkykxxx':eykzkykxxx,
                'ezkzkykxxx':ezkzkykxxx,
                'uxkzkykxxx':uxkzkykxxx,
                'uykzkykxxx':uykzkykxxx,
                'uzkzkykxxx':uzkzkykxxx,
                'eparkzkykxxx':eparkzkykxxx,
                }
                #'epardetrendkzkykxxx':epardetrendkzkykxxx,
                #'eperp1detrendkzkykxxx':eperp1detrendkzkykxxx,
                #'eperp2detrendkzkykxxx':eperp2detrendkzkykxxx,
                #'bpardetrendkzkykxxx':bpardetrendkzkykxxx,
                #'bperp1detrendkzkykxxx':bperp1detrendkzkykxxx,
                #'bperp2detrendkzkykxxx':bperp2detrendkzkykxxx 
                #}


    picklefile = flnm+'.retstep'+str(retstep)+'.pickle'

    with open(picklefile, 'wb') as handle:
        pickle.dump(pckldict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Saved FFTWLT pickle to ',picklefile)

    return pckldict

#------------------------------------------------------------------------------------------------------------------------------------
# Begin script
#------------------------------------------------------------------------------------------------------------------------------------
#user params
framenum = '700' #frame to make figure of (should be a string)
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]
xxpos = 8.5

wftoutflnm = 'analysisfiles/wft' #Warning: if filename includes a directory, please ensure that directory exists, otherwise, it won't save

ispregenerated = True
pregeneratedflnm = 'analysisfiles/wft.retstep2.pickle' #'/data/not_backed_up/intermediate_analysis/collbrown/shocks/wlt_nonadiaperp/testwft.retstep1.pickle'

printmodes = True #prints sorted list of modes at the end

normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)

params = ld.load_params(flpath,framenum)
dt = params['c']/params['comp'] #in units of wpe
c = params['c']
stride = 100
dt,c = aa.norm_constants(params,dt,c,stride)

#compute shock velocity and boost to shock rest frame
if(False):#not(ispregenerated)): #save time if not using fields data 
    pass #TODO: revert
    #dfields_many_frames = {'frame':[],'dfields':[]}
    #for _num in frames:
    #    num = int(_num)
    #    d = ld.load_fields(flpath,_num,normalizeFields=normalize)
    #    dfields_many_frames['dfields'].append(d)
    #    dfields_many_frames['frame'].append(num)
    #vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)
else:
    vshock = 1.5625375379202415
    print("Warning: using hard coded value of vshock to save computational time...")
    print("vshock = ",vshock)

dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

dfluc = aa.remove_average_fields_over_yz(dfields)

dpar_elec, dpar_ion = ld.load_particles(flpath,framenum,normalizeVelocity=normalize)
inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
inputs = ld.load_input(inputpath)
beta0 = aa.compute_beta0(params,inputs)
dpar_ion = ft.shift_particles(dpar_ion, vshock, beta0, params['mi']/params['me'], isIon=True)
dpar_elec = ft.shift_particles(dpar_elec, vshock, beta0, params['mi']/params['me'], isIon=False)
if(not(ispregenerated)):
    print("WARNING: returning zero array for dflow for now...")
    dflow = aa.compute_dflow(dfields, dpar_ion, dpar_elec, return_empty = True)
else:
    dflow = aa.compute_dflow(dfields, dpar_ion, dpar_elec, return_empty = True)

for _key in dflow.keys():
    dfields[_key] = dflow[_key]

#TODO: update to do wft of ions and elecs and change key names
dfluc['ux'] = dflow['ui']
dfluc['uy'] = dflow['vi']
dfluc['uz'] = dflow['wi']
dfluc['ux_xx'] = dflow['ui_xx']
dfluc['ux_yy'] = dflow['ui_yy']
dfluc['ux_zz'] = dflow['ui_zz']
dfluc['uy_xx'] = dflow['vi_xx']
dfluc['uy_yy'] = dflow['vi_yy']
dfluc['uy_zz'] = dflow['vi_zz']
dfluc['uz_xx'] = dflow['wi_xx']
dfluc['uz_yy'] = dflow['wi_yy']
dfluc['uz_zz'] = dflow['wi_zz']

#Reduce size of sim and reduce detail by taking averages
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

#dfields = ao.truncate_dict(dfields,reducfrac=[1,1,2],planes=['z','y','x'])
#dflow = ao.truncate_dict(dflow,reducfrac=[1,1,2],planes=['z','y','x'])
#dfluc = ao.truncate_dict(dfluc,reducfrac=[1,1,2],planes=['z','y','x'])
dfields = ao.avg_dict(dfields,binidxsz=[1,1,7],planes=['z','y','x'])
dflow = ao.avg_dict(dflow,binidxsz=[1,1,7],planes=['z','y','x'])
dfluc = ao.avg_dict(dfluc,binidxsz=[1,1,7],planes=['z','y','x'])

dfluc = aa.convert_fluc_to_par(dfields,dfluc)
dfields = aa.convert_to_local_par(dfields) 

if(not(ispregenerated)):
    WFTdata = generate_fftwlt(dfluc,dfields,2,wftoutflnm)
else:
    print("Loading from file: ", pregeneratedflnm)
    filein = open(pregeneratedflnm, 'rb')
    WFTdata = pickle.load(filein)
    filein.close()

#dummmy plot to load font (fixes weird mpl bug)
plt.figure()
plt.style.use("cb.mplstyle")
plt.plot([0,1],[0,1])
plt.close()

#TODO: remove!!! below block!!
printmodes = True
if(printmodes):

    xidx = ao.find_nearest(WFTdata['xx'],xxpos)

    dx = dfields['ex_xx'][1]-dfields['ex_xx'][0]
    xlim = [dfields['ex_xx'][xidx]-dx/2.,dfields['ex_xx'][xidx]+dx/2.]
    ylim = [dfields['ex_yy'][0]-dx/2.,dfields['ex_yy'][-1]+dx/2.]
    zlim = [dfields['ex_zz'][0]-dx/2.,dfields['ex_zz'][-1]+dx/2.]

    #computes properties of all the wavemodes found at xpos
    print("Computing wavemodes...")
    dwavemodes = wa.compute_wavemodes(None,dfields,xlim,ylim,zlim,
                     WFTdata['kx'],WFTdata['ky'],WFTdata['kz'],
                     WFTdata['bxkzkykxxx'],WFTdata['bykzkykxxx'],WFTdata['bzkzkykxxx'],
                     WFTdata['exkzkykxxx'],WFTdata['eykzkykxxx'],WFTdata['ezkzkykxxx'],
                     WFTdata['uxkzkykxxx'],WFTdata['uykzkykxxx'],WFTdata['uzkzkykxxx'],
                     specifyxxidx=xidx)

    from uncertainties import ufloat
    sortkey = 'normE'#'Epar' 
    dwavemodes = wa.sort_wavemodes(dwavemodes,sortkey)
    
    for _i in range(0,10000):
        

        delta_kx = ufloat(dwavemodes['wavemodes'][_i]['kx'],.2) #.2 comes from evaluating the integral in equation 3 of Najimi et al 1997 (is approximate and used as upper bound for error in kpar and perp)
        dwavemodes['wavemodes'][_i]['delta_kperp1'] = .2
        dwavemodes['wavemodes'][_i]['delta_kperp2'] = .2
        dwavemodes['wavemodes'][_i]['delta_kpar'] = .2
        om1r,om1i,om2r,om2i,om3r,om3i = wa.get_freq_from_wvmd(dwavemodes['wavemodes'][_i],debug = False,comp_error_prop=True)

        #om1detrend,om2detrend,om3detrend = wa.get_freq_from_wvmd(dwavemodes['wavemodes'][_i], debug = False, usedetrend = True,comp_error_prop=True)

        kx = dwavemodes['wavemodes'][_i]['kx']
        ky = dwavemodes['wavemodes'][_i]['ky']
        kz = dwavemodes['wavemodes'][_i]['kz']
        kpar = dwavemodes['wavemodes'][_i]['kpar']
        kperp1 = dwavemodes['wavemodes'][_i]['kperp1']
        kperp2 = dwavemodes['wavemodes'][_i]['kperp2']
        sortkeyval = dwavemodes['wavemodes'][_i][sortkey]

        if(_i < 3000000 and np.abs(kpar) < 3 and np.abs(kperp1) < 3 and np.abs(kperp2) < 3):
            print(_i)
            print('kx: ', kx, ' ky: ', ky, ' kz: ', kz, ' kpar: ', kpar, ' kperp1: ', kperp1, ' kperp2: ', kperp2,' sortkeyval: ', np.abs(sortkeyval))
            print('real: ',om1r,om2r,om3r)
            print('imag: ', om1i, om2i, om3i)
         #   print(om1detrend,om2detrend,om3detrend)



#make pub figure---------------------
import os
os.system("mkdir figures")
os.system("mkdir figures/spectra")

fig = plt.figure(figsize=(10,17))
ax1 = plt.subplot(411)
ax2 = plt.subplot(412)
ax3 = plt.subplot(413)
ax4 = plt.subplot(414)

plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

#tot
kx = WFTdata['kx']
xx = WFTdata['xx']
wlt = np.abs(np.sqrt(np.abs(WFTdata['exkzkykxxx'])**2+np.abs(WFTdata['eykzkykxxx'])**2+np.abs(WFTdata['ezkzkykxxx'])**2))
wlt = np.sum(wlt,axis=(0,1))
img1 = ax1.pcolormesh(xx,kx,np.abs(wlt),cmap='Spectral', shading='gouraud')
ax1.set_xlim(5,12)
ax1.set_ylim(0,100)

#ex
kx = WFTdata['kx']
xx = WFTdata['xx']
wlt = np.abs(WFTdata['exkzkykxxx'])
wlt = np.sum(wlt,axis=(0,1))
img2 = ax2.pcolormesh(xx,kx,np.abs(wlt),cmap='Spectral', shading='gouraud')
ax2.set_xlim(5,12)
ax2.set_ylim(0,100)

#ey
kx = WFTdata['kx']
xx = WFTdata['xx']
wlt = np.abs(WFTdata['eykzkykxxx'])
wlt = np.sum(wlt,axis=(0,1))
img3 = ax3.pcolormesh(xx,kx,np.abs(wlt),cmap='Spectral', shading='gouraud')
ax3.set_xlim(5,12)
ax3.set_ylim(0,100)

#ez
kx = WFTdata['kx']
xx = WFTdata['xx']
wlt = np.abs(WFTdata['ezkzkykxxx'])
wlt = np.sum(wlt,axis=(0,1))
img4 = ax4.pcolormesh(xx,kx,np.abs(wlt),cmap='Spectral', shading='gouraud')
ax4.set_xlim(5,12)
ax4.set_ylim(0,100)

ax1.set_ylabel('$k_x \, \, d_{i,0}$')
ax1.grid()
ax2.set_ylabel('$k_x \, \, d_{i,0}$')
ax2.grid()
ax3.set_ylabel('$k_x \, \, d_{i,0}$')
ax3.grid()
ax4.set_ylabel('$k_x \, \, d_{i,0}$')
ax4.grid()

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider1 = make_axes_locatable(ax1)
divider2 = make_axes_locatable(ax2)
divider3 = make_axes_locatable(ax3)
divider4 = make_axes_locatable(ax4)
ax_1 = divider1.append_axes("bottom", size="27%", pad=0.12)
ax_2 = divider2.append_axes("bottom", size="27%", pad=0.12)
ax_3 = divider3.append_axes("bottom", size="27%", pad=0.12)
ax_1.remove()
ax_2.remove()
ax_3.remove()
plt.subplots_adjust(hspace=-0.1)

ax5 = divider4.append_axes("bottom", size="27%", pad=0.12)
cax1 = divider1.append_axes("right", size="2%", pad=0.1)
cax2 = divider2.append_axes("right", size="2%", pad=0.1)
cax3 = divider3.append_axes("right", size="2%", pad=0.1)
cax4 = divider4.append_axes("right", size="2%", pad=0.15)

clrbarlbl1 = r'$\big<|\delta \mathbf{\hat{E}}(k_x;x)|\big>_{k_y}$'
clrbarlbl2 = r'$\big<|\delta \hat{E}_x(k_x;x)|\big>_{k_y}$'
clrbarlbl3 = r'$\big<|\delta \hat{E}_y(k_x;x)|\big>_{k_y}$'
clrbarlbl4 = r'$\big<|\delta \hat{E}_z(k_x;x)|\big>_{k_y}$'
cbar1 = plt.colorbar(img1, ax=ax1, cax=cax1)
cbar1.set_label(clrbarlbl1,labelpad=35, rotation=270)
cbar2 = plt.colorbar(img2, ax=ax2, cax=cax2)
cbar2.set_label(clrbarlbl2,labelpad=35, rotation=270)
cbar3 = plt.colorbar(img3, ax=ax3, cax=cax3)
cbar3.set_label(clrbarlbl3,labelpad=35, rotation=270)
cbar4 = plt.colorbar(img4, ax=ax4, cax=cax4)
cbar4.set_label(clrbarlbl4,labelpad=35, rotation=270)

dfavg = aa.get_average_fields_over_yz(dfields)
xx = dfavg['by_xx']
fieldvals = dfavg['by'][0,0,:]
ax5.plot(xx,fieldvals,color='black',linewidth=1)
ax5.grid()
ax5.set_xlabel(r"$x/d_{i,0}$")
ax5.set_ylabel(r"$\overline{B_y}(x)$")
ax5.set_xlim(5,12)

ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
ax4.set_xticklabels([])

flnm = 'figures/wftefieldsuperplt.png'
plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
plt.close()

import os
os.system("mkdir figures")
os.system("mkdir figures/spectra")

fig = plt.figure(figsize=(10,17))
ax1 = plt.subplot(411)
ax2 = plt.subplot(412)
ax3 = plt.subplot(413)
ax4 = plt.subplot(414)

plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

#tot
kx = WFTdata['kx']
xx = WFTdata['xx']
wlt = np.abs(np.sqrt(np.abs(WFTdata['bxkzkykxxx'])**2+np.abs(WFTdata['bykzkykxxx'])**2+np.abs(WFTdata['bzkzkykxxx'])**2))
wlt = np.sum(wlt,axis=(0,1))
img1 = ax1.pcolormesh(xx,kx,np.abs(wlt),cmap='Spectral', shading='gouraud')
ax1.set_xlim(5,12)
ax1.set_ylim(0,100)

#ex
kx = WFTdata['kx']
xx = WFTdata['xx']
wlt = np.abs(WFTdata['bxkzkykxxx'])
wlt = np.sum(wlt,axis=(0,1))
img2 = ax2.pcolormesh(xx,kx,np.abs(wlt),cmap='Spectral', shading='gouraud')
ax2.set_xlim(5,12)
ax2.set_ylim(0,100)

#ey
kx = WFTdata['kx']
xx = WFTdata['xx']
wlt = np.abs(WFTdata['bykzkykxxx'])
wlt = np.sum(wlt,axis=(0,1))
img3 = ax3.pcolormesh(xx,kx,np.abs(wlt),cmap='Spectral', shading='gouraud')
ax3.set_xlim(5,12)
ax3.set_ylim(0,100)

#ez
kx = WFTdata['kx']
xx = WFTdata['xx']
wlt = np.abs(WFTdata['bzkzkykxxx'])
wlt = np.sum(wlt,axis=(0,1))
img4 = ax4.pcolormesh(xx,kx,np.abs(wlt),cmap='Spectral', shading='gouraud')
ax4.set_xlim(5,12)
ax4.set_ylim(0,100)

ax1.set_ylabel('$k_x \, \, d_{i,0}$')
ax1.grid()
ax2.set_ylabel('$k_x \, \, d_{i,0}$')
ax2.grid()
ax3.set_ylabel('$k_x \, \, d_{i,0}$')
ax3.grid()
ax4.set_ylabel('$k_x \, \, d_{i,0}$')
ax4.grid()

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider1 = make_axes_locatable(ax1)
divider2 = make_axes_locatable(ax2)
divider3 = make_axes_locatable(ax3)
divider4 = make_axes_locatable(ax4)
ax_1 = divider1.append_axes("bottom", size="27%", pad=0.12)
ax_2 = divider2.append_axes("bottom", size="27%", pad=0.12)
ax_3 = divider3.append_axes("bottom", size="27%", pad=0.12)
ax_1.remove()
ax_2.remove()
ax_3.remove()
plt.subplots_adjust(hspace=-0.1)

ax5 = divider4.append_axes("bottom", size="27%", pad=0.12)
cax1 = divider1.append_axes("right", size="2%", pad=0.1)
cax2 = divider2.append_axes("right", size="2%", pad=0.1)
cax3 = divider3.append_axes("right", size="2%", pad=0.1)
cax4 = divider4.append_axes("right", size="2%", pad=0.15)

clrbarlbl1 = r'$\big<|\delta \mathbf{\hat{B}}(k_x;x)|\big>_{k_y}$'
clrbarlbl2 = r'$\big<|\delta \hat{B}_x(k_x;x)|\big>_{k_y}$'
clrbarlbl3 = r'$\big<|\delta \hat{B}_y(k_x;x)|\big>_{k_y}$'
clrbarlbl4 = r'$\big<|\delta \hat{B}_z(k_x;x)|\big>_{k_y}$'
cbar1 = plt.colorbar(img1, ax=ax1, cax=cax1)
cbar1.set_label(clrbarlbl1,labelpad=35, rotation=270)
cbar2 = plt.colorbar(img2, ax=ax2, cax=cax2)
cbar2.set_label(clrbarlbl2,labelpad=35, rotation=270)
cbar3 = plt.colorbar(img3, ax=ax3, cax=cax3)
cbar3.set_label(clrbarlbl3,labelpad=35, rotation=270)
cbar4 = plt.colorbar(img4, ax=ax4, cax=cax4)
cbar4.set_label(clrbarlbl4,labelpad=35, rotation=270)

dfavg = aa.get_average_fields_over_yz(dfields)
xx = dfavg['by_xx']
fieldvals = dfavg['by'][0,0,:]
ax5.plot(xx,fieldvals,color='black',linewidth=1)
ax5.grid()
ax5.set_xlabel(r"$x/d_{i,0}$")
ax5.set_ylabel(r"$\overline{B_y}(x)$")
ax5.set_xlim(5,12)

ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
ax4.set_xticklabels([])

flnm = 'figures/wftbfieldsuperplt.png'
plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
plt.close()
#end make pub figure---------------------






flnmprefix = 'figures/spectra/'

picklefile = 'dwavemodes8.5.pickle'
with open(picklefile, 'wb') as handle:
    pickle.dump(dwavemodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

#2D (k_component1 vs k_component2) power spectrum plot
#speckey = 'Epar' #'Epar' or 'normE' typically
#klim = 50
#hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
#hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)
#speckey = 'Eperp1' #'Epar' or 'normE' typically
#klim = 50
#hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
#hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)
#speckey = 'Eperp2' #'Epar' or 'normE' typically
#klim = 50
#hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
#hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)
#speckey = 'normE' #'Epar' or 'normE' typically
#klim = 50
#hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
#hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)

#Plot wft...
import os
os.system("mkdir figures")
os.system("mkdir figures/spectra")

flnmprefix = 'figures/spectra/'
kx = WFTdata['kx']
wlt = np.abs(np.sqrt(np.abs(WFTdata['exkzkykxxx'])**2+np.abs(WFTdata['eykzkykxxx'])**2+np.abs(WFTdata['ezkzkykxxx'])**2))
wlt = np.sum(wlt,axis=(0,1))
xx = WFTdata['xx']
pw.plot_wlt(xx, kx, wlt, flnm = flnmprefix+'enormprojectedwltkxxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

ky = WFTdata['ky']
wlt = np.abs(np.sqrt(np.abs(WFTdata['exkzkykxxx'])**2+np.abs(WFTdata['eykzkykxxx'])**2+np.abs(WFTdata['ezkzkykxxx'])**2))
wlt = np.sum(wlt,axis=(0,2))
xx = WFTdata['xx']
pw.plot_wlt_ky(xx, ky, wlt, flnm = flnmprefix+'enormprojectedwltkyxx.png', xlim = None, ylim = [-100,100], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

kx = WFTdata['kx']
wlt = np.abs(np.sqrt(np.abs(WFTdata['bxkzkykxxx'])**2+np.abs(WFTdata['bykzkykxxx'])**2+np.abs(WFTdata['bzkzkykxxx'])**2))
wlt = np.sum(wlt,axis=(0,1))
xx = WFTdata['xx']
pw.plot_wlt(xx, kx, wlt, flnm = flnmprefix+'bnormprojectedwltkxxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

ky = WFTdata['ky']
wlt = np.sqrt(np.abs(WFTdata['bxkzkykxxx'])**2+np.abs(WFTdata['bykzkykxxx'])**2+np.abs(WFTdata['bzkzkykxxx'])**2)
wlt = np.sum(wlt,axis=(0,2))
xx = WFTdata['xx']
pw.plot_wlt_ky(xx, ky, wlt, flnm = flnmprefix+'bnormprojectedwltkyxx.png', xlim = None, ylim = [-100,100], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

kx = WFTdata['kx']
wlt = np.abs(WFTdata['exkzkykxxx'])
wlt = np.sum(wlt,axis=(0,1))
xx = WFTdata['xx']
pw.plot_wlt(xx, kx, wlt, flnm = flnmprefix+'exprojectedwltkxxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

ky = WFTdata['ky']
wlt = np.abs(WFTdata['exkzkykxxx'])
wlt = np.sum(wlt,axis=(0,2))
xx = WFTdata['xx']
pw.plot_wlt_ky(xx, ky, wlt, flnm = flnmprefix+'exprojectedwltkyxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

kx = WFTdata['kx']
wlt = np.abs(WFTdata['eykzkykxxx'])
wlt = np.sum(wlt,axis=(0,1))
xx = WFTdata['xx']
pw.plot_wlt(xx, kx, wlt, flnm = flnmprefix+'eyprojectedwltkxxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

ky = WFTdata['ky']
wlt = np.abs(WFTdata['eykzkykxxx'])
wlt = np.sum(wlt,axis=(0,2))
xx = WFTdata['xx']
pw.plot_wlt_ky(xx, ky, wlt, flnm = flnmprefix+'eyprojectedwltkyxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

kx = WFTdata['kx']
wlt = np.abs(WFTdata['ezkzkykxxx'])
wlt = np.sum(wlt,axis=(0,1))
xx = WFTdata['xx']
pw.plot_wlt(xx, kx, wlt, flnm = flnmprefix+'ezprojectedwltkxxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

ky = WFTdata['ky']
wlt = np.abs(WFTdata['ezkzkykxxx'])
wlt = np.sum(wlt,axis=(0,2))
xx = WFTdata['xx']
pw.plot_wlt_ky(xx, ky, wlt, flnm = flnmprefix+'ezprojectedwltkyxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

kx = WFTdata['kx']
wlt = np.abs(WFTdata['exkzkykxxx'])
wlt = np.sum(wlt,axis=(0,1))
xx = WFTdata['xx']
pw.plot_wlt(xx, kx, wlt, flnm = flnmprefix+'exprojectedwltkxxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

ky = WFTdata['ky']
wlt = np.abs(WFTdata['bxkzkykxxx'])
wlt = np.sum(wlt,axis=(0,2))
xx = WFTdata['xx']
pw.plot_wlt_ky(xx, ky, wlt, flnm = flnmprefix+'bxprojectedwltkyxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

kx = WFTdata['kx']
wlt = np.abs(WFTdata['bykzkykxxx'])
wlt = np.sum(wlt,axis=(0,1))
xx = WFTdata['xx']
pw.plot_wlt(xx, kx, wlt, flnm = flnmprefix+'byprojectedwltkxxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

ky = WFTdata['ky']
wlt = np.abs(WFTdata['bykzkykxxx'])
wlt = np.sum(wlt,axis=(0,2))
xx = WFTdata['xx']
pw.plot_wlt_ky(xx, ky, wlt, flnm = flnmprefix+'byprojectedwltkyxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

kx = WFTdata['kx']
wlt = np.abs(WFTdata['bzkzkykxxx'])
wlt = np.sum(wlt,axis=(0,1))
xx = WFTdata['xx']
pw.plot_wlt(xx, kx, wlt, flnm = flnmprefix+'bzprojectedwltkxxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

ky = WFTdata['ky']
wlt = np.abs(WFTdata['bzkzkykxxx'])
wlt = np.sum(wlt,axis=(0,2))
xx = WFTdata['xx']
pw.plot_wlt_ky(xx, ky, wlt, flnm = flnmprefix+'bzprojectedwltkyxx.png', xlim = None, ylim = [-75,75], xxline = None, yyline = None, clrbarlbl = None,axhline=None)

flnm = 'figures/spectra/bykyxx.png'
pw.plot_kyxx_box_aligned(WFTdata,'bykzkykxxx',flnm)

flnm = 'figures/spectra/bxkyxx.png'
pw.plot_kyxx_box_aligned(WFTdata,'bxkzkykxxx',flnm)

flnm = 'figures/spectra/bzkyxx.png'
pw.plot_kyxx_box_aligned(WFTdata,'bzkzkykxxx',flnm)

flnm = 'figures/spectra/eykyxx.png'
pw.plot_kyxx_box_aligned(WFTdata,'eykzkykxxx',flnm)

flnm = 'figures/spectra/exkyxx.png'
pw.plot_kyxx_box_aligned(WFTdata,'exkzkykxxx',flnm)

flnm = 'figures/spectra/ezkyxx.png'
pw.plot_kyxx_box_aligned(WFTdata,'ezkzkykxxx',flnm)

xidx = ao.find_nearest(WFTdata['xx'],xxpos)
#array is ordered FIELDCOMPkzkykxxx

#define box that will be used to define the 'field aligned' direction
dx = dfields['ex_xx'][1]-dfields['ex_xx'][0]
xlim = [dfields['ex_xx'][xidx]-dx/2.,dfields['ex_xx'][xidx]+dx/2.]
ylim = [dfields['ex_yy'][0]-dx/2.,dfields['ex_yy'][-1]+dx/2.]
zlim = [dfields['ex_zz'][0]-dx/2.,dfields['ex_zz'][-1]+dx/2.]

#computes properties of all the wavemodes found at xpos
print("Computing wavemodes...")
dwavemodes = wa.compute_wavemodes(None,dfields,xlim,ylim,zlim,
                     WFTdata['kx'],WFTdata['ky'],WFTdata['kz'],
                     WFTdata['bxkzkykxxx'],WFTdata['bykzkykxxx'],WFTdata['bzkzkykxxx'],
                     WFTdata['exkzkykxxx'],WFTdata['eykzkykxxx'],WFTdata['ezkzkykxxx'],
                     WFTdata['uxkzkykxxx'],WFTdata['uykzkykxxx'],WFTdata['uzkzkykxxx'],
                     eparlocalkzkykxxx = WFTdata['eparkzkykxxx'],epardetrendkzkykxxx = WFTdata['epardetrendkzkykxxx'],
                     eperp1detrendkzkykxxx = WFTdata['eperp1detrendkzkykxxx'], eperp2detrendkzkykxxx = WFTdata['eperp2detrendkzkykxxx'],
                     bpardetrendkzkykxxx = WFTdata['bpardetrendkzkykxxx'], bperp1detrendkzkykxxx = WFTdata['bperp1detrendkzkykxxx'], bperp2detrendkzkykxxx = WFTdata['bperp2detrendkzkykxxx'],
                     specifyxxidx=xidx)

if(printmodes):
    sortkey = 'normE'#'Epar' 
    dwavemodes = wa.sort_wavemodes(dwavemodes,sortkey)
    for _i in range(0,10000):

        om1,om2,om3 = wa.get_freq_from_wvmd(dwavemodes['wavemodes'][_i],debug = False)

        om1detrend,om2detrend,om3detrend = wa.get_freq_from_wvmd(dwavemodes['wavemodes'][_i], debug = False, usedetrend = True)

        print(om1detrend)

        kx = dwavemodes['wavemodes'][_i]['kx']
        ky = dwavemodes['wavemodes'][_i]['ky']
        kz = dwavemodes['wavemodes'][_i]['kz']
        kpar = dwavemodes['wavemodes'][_i]['kpar']
        kperp1 = dwavemodes['wavemodes'][_i]['kperp1']
        kperp2 = dwavemodes['wavemodes'][_i]['kperp2']
        sortkeyval = dwavemodes['wavemodes'][_i][sortkey]

        if(_i < 10 or kpar > 14.):
            print(_i)
            print('kx: ', kx, ' ky: ', ky, ' kz: ', kz, ' kpar: ', kpar, ' kperp1: ', kperp1, ' kperp2: ', kperp2,' sortkeyval: ', np.abs(sortkeyval))
            print(om1,om2,om3)
            print(om1detrend,om2detrend,om3detrend)


#TODO: save files to figures/spectra folder

flnmprefix = 'figures/spectra/'

picklefile = 'dwavemodes8.5.pickle'
with open(picklefile, 'wb') as handle:
    pickle.dump(dwavemodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

#2D (k_component1 vs k_component2) power spectrum plot
speckey = 'Epar' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)
speckey = 'Eperp1' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)
speckey = 'Eperp2' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)
speckey = 'normE' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)


#1D power spectrum plot
binsize = WFTdata['ky'][1]-WFTdata['ky'][0] #normally, we would hand select this value, but as kpar ~= ky, this works well

#TODO: small binning bug (more related to how the data is presented than any key results)
print("**** FIX SPEC BEING ZERO AT K = 0 in the final binned output; it should be small, not zero")

binkey = 'kpar'
binlowerbound = 0
binupperbound = 100

key = 'Epar'
pw.plot_spec_1d(dwavemodes,flnm=flnmprefix+'1dspec'+binkey+key,key=key,binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound)
key = 'normE'
pw.plot_spec_1d(dwavemodes,flnm=flnmprefix+'1dspec'+binkey+key,key=key,binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound)
key = 'Bpar'
pw.plot_spec_1d(dwavemodes,flnm=flnmprefix+'1dspec'+binkey+key,key=key,binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound)
key = 'normB'
pw.plot_spec_1d(dwavemodes,flnm=flnmprefix+'1dspec'+binkey+key,key=key,binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound)
key = 'Epar_local'
pw.plot_spec_1d(dwavemodes,flnm=flnmprefix+'1dspec'+binkey+key,key=key,binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound)
key = 'Epar_detrend'
pw.plot_spec_1d(dwavemodes,flnm=flnmprefix+'1dspec'+binkey+key,key=key,binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound)

pw.plot_spec_1d(dwavemodes,flnm=flnmprefix+'1dspec'+binkey+'Epar_Bpar',key=['Epar','Bpar'],binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound)#plot Epar Bpar on same plot
pw.plot_spec_1d(dwavemodes,flnm=flnmprefix+'1dspec'+binkey+'normE_normB',key=['normE','normB'],binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound)#plot normE normB on same plot
pw.plot_spec_1d(dwavemodes,flnm=flnmprefix+'1dspec'+binkey+'Eperp1_Bperp1',key=['Eperp1','Bperp1'],binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound)#plot Epar Bpar on same plot
pw.plot_spec_1d(dwavemodes,flnm=flnmprefix+'1dspec'+binkey+'Eperp2_Bperp2',key=['Eperp2','Bperp2'],binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound)#plot Epar Bpar on same plot

#2D (k_component vs x) power spectrum plot
#key = 'Epar'
#pw.plot_spec_2d(WFTdata,dfields,xxrange=[7,11],flnm='2dspec'+binkey+key,key=key,binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound,verbose=True)
#key = 'normE'
#pw.plot_spec_2d(WFTdata,dfields,xxrange=[7,11],flnm='2dspec'+binkey+key,key=key,binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound,verbose=True)
#key = 'Bpar'
#pw.plot_spec_2d(WFTdata,dfields,xxrange=[7,11],flnm='2dspec'+binkey+key,key=key,binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound,verbose=True)
#key = 'normB'
#pw.plot_spec_2d(WFTdata,dfields,xxrange=[7,11],flnm='2dspec'+binkey+key,key=key,binkey=binkey,binsize=binsize,binlowerbound=binlowerbound,binupperbound=binupperbound,verbose=True)

#debug plots
key = 'Epar'
pw.plot_spec_1dnobin(dwavemodes,flnm=flnmprefix+'1dspecnobin'+binkey+key,key=key,binkey=binkey,binlowerbound=binlowerbound,binupperbound=binupperbound)
key = 'normE'
pw.plot_spec_1dnobin(dwavemodes,flnm=flnmprefix+'1dspecnobin'+binkey+key,key=key,binkey=binkey,binlowerbound=binlowerbound,binupperbound=binupperbound)
key = 'Bpar'
pw.plot_spec_1dnobin(dwavemodes,flnm=flnmprefix+'1dspecnobin'+binkey+key,key=key,binkey=binkey,binlowerbound=binlowerbound,binupperbound=binupperbound)
key = 'normB'
