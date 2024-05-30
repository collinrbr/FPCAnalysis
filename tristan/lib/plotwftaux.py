import numpy as np
import matplotlib.pyplot as plt

#TODO: use or remove gridsize
def plot_power_spectrum(dwavemodes,flnm='',key='normB',gridsize1=150,gridsize2=150,gridsize3=150,kperp1lim=None,kperp2lim=None,kparlim=None):
    _x1temp = []
    _x2temp = []
    _ytemp = []
    _ztemp = []
    for wvmd in dwavemodes['wavemodes']:
        _x1temp.append(wvmd['kperp1'])
        _x2temp.append(wvmd['kperp2'])
        _ytemp.append(wvmd['kpar'])
        _ztemp.append(np.abs(wvmd[key]))

    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    hxbin0 = axs[0].hexbin(_x1temp, _x2temp, cmap='Spectral', C=_ztemp,gridsize=(gridsize1,gridsize2),reduce_C_function=np.mean,linewidths=0.1)
    axs[0].set_xlabel('$k_{\perp,1} \, \, d_{i,0}$',fontsize=18)
    axs[0].set_ylabel('$k_{\perp_2} \, \, d_{i,0}$',fontsize=16)
    if(kperp1lim != None):
        axs[0].set_xlim(-kperp1lim,kperp1lim)
        axs[0].set_ylim(-kperp2lim,kperp2lim)
    axs[0].set_aspect('equal')
    plt.colorbar(hxbin0,ax=axs[0],fraction=.05)

    hxbin1 = axs[1].hexbin(_x1temp, _ytemp, cmap='Spectral', C=_ztemp,gridsize=(gridsize1,gridsize3),reduce_C_function=np.mean,linewidths=0.1)
    axs[1].set_xlabel('$k_{\perp,1} \, \, d_{i,0}$',fontsize=16)
    axs[1].set_ylabel('$k_{||} \, \, d_{i,0}$',fontsize=16)
    if(kperp1lim != None):
        axs[1].set_xlim(-kperp1lim,kperp1lim)
        axs[1].set_ylim(-kparlim,kparlim)
    axs[1].set_aspect('equal')
    plt.colorbar(hxbin1,ax=axs[1],fraction=.05)

    hxbin2 = axs[2].hexbin(_x2temp, _ytemp, cmap='Spectral', C=_ztemp,gridsize=(gridsize2,gridsize3),reduce_C_function=np.mean,linewidths=0.1)
    axs[2].set_xlabel('$k_{\perp,2} \, \, d_{i,0}$',fontsize=16)
    axs[2].set_ylabel('$k_{||} \, \, d_{i,0}$',fontsize=16)
    if(kperp1lim != None):
        axs[2].set_xlim(-kperp2lim,kperp2lim)
        axs[2].set_ylim(-kparlim,kparlim)
    axs[2].set_aspect('equal')
    plt.colorbar(hxbin2,ax=axs[2],fraction=.05)

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    plt.subplots_adjust(wspace=.5)
    
    tpad = 8
    if(key=='normB'):
        axs[0].set_title(r'$<\hat{|\delta \mathbf{B}|}(k_{\perp,1},k_{\perp,2})>_{k_{||}}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<\hat{|\delta \mathbf{B}|}(k_{||},k_{\perp,1})>_{k_{\perp,2}}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<\hat{|\delta \mathbf{B}|}(k_{||},k_{\perp,2})>_{k_{\perp,1}}$',loc='right',pad=tpad)

    if(key=='Bpar'):
        axs[0].set_title(r'$<|\delta \hat{B}_{||}|(k_{\perp,1},k_{\perp,2})>_{k_{||}}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{B}_{||}|(k_{||},k_{\perp,1})>_{k_{\perp,2}}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{B}_{||}|(k_{||},k_{\perp,2})>_{k_{\perp,1}}$',loc='right',pad=tpad)

    if(key=='Bperp1'):
        axs[0].set_title(r'$<|\delta \hat{B}_{\perp,1}|(k_{\perp,1},k_{\perp,2})>_{k_{||}}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{B}_{\perp,1}|(k_{||},k_{\perp,1})>_{k_{\perp,2}}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{B}_{\perp,1}|(k_{||},k_{\perp,2})>_{k_{\perp,1}}$',loc='right',pad=tpad)

    if(key=='Bperp2'):
        axs[0].set_title(r'$<|\delta \hat{B}_{\perp,2}|(k_{\perp,1},k_{\perp,2})>_{k_{||}}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{B}_{\perp,2}|(k_{||},k_{\perp,1})>_{k_{\perp,2}}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{B}_{\perp,2}|(k_{||},k_{\perp,2})>_{k_{\perp,1}}$',loc='right',pad=tpad)

    if(key=='normE'):
        axs[0].set_title(r'$<\hat{|\delta \mathbf{E}|}(k_{\perp,1},k_{\perp,2})>_{k_{||}}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<\hat{|\delta \mathbf{E}|}(k_{||},k_{\perp,1})>_{k_{\perp,2}}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<\hat{|\delta \mathbf{E}|}(k_{||},k_{\perp,2})>_{k_{\perp,1}}$',loc='right',pad=tpad)

    if(key=='Epar'):
        axs[0].set_title(r'$<|\delta \hat{E}_{||}|(k_{\perp,1},k_{\perp,2})>_{k_{||}}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{E}_{||}|(k_{||},k_{\perp,1})>_{k_{\perp,2}}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{E}_{||}|(k_{||},k_{\perp,2})>_{k_{\perp,1}}$',loc='right',pad=tpad)

    if(key=='Eperp1'):
        axs[0].set_title(r'$<|\delta \hat{E}_{\perp,1}|(k_{\perp,1},k_{\perp,2})>_{k_{||}}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{E}_{\perp,1}|(k_{||},k_{\perp,1})>_{k_{\perp,2}}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{E}_{\perp,1}|(k_{||},k_{\perp,2})>_{k_{\perp,1}}$',loc='right',pad=tpad)

    if(key=='Eperp2'):
        axs[0].set_title(r'$<|\delta \hat{E}_{\perp,2}|(k_{\perp,1},k_{\perp,2})>_{k_{||}}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{E}_{\perp,2}|(k_{||},k_{\perp,1})>_{k_{\perp,2}}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{E}_{\perp,2}|(k_{||},k_{\perp,2})>_{k_{\perp,1}}$',loc='right',pad=tpad)

    if(flnm != ''):
        plt.savefig(flnm,dpi=300,format='png',bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return hxbin0, hxbin1, hxbin2

#TODO: remove or used gridsize2,3
def plot_power_spectrum_cart(dwavemodes,flnm='',key='normB',gridsize1=150,gridsize2=150,gridsize3=150,kxlim=None,kylim=None,kzlim=None):
    _x1temp = []
    _x2temp = []
    _ytemp = []
    _ztemp = []
    for wvmd in dwavemodes['wavemodes']:
        _x1temp.append(wvmd['kx'])
        _x2temp.append(wvmd['ky'])
        _ytemp.append(wvmd['kz'])
        _ztemp.append(np.abs(wvmd[key]))

    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    hxbin0 = axs[0].hexbin(_x1temp, _x2temp, cmap='Spectral', C=_ztemp,gridsize=(gridsize1,gridsize2),reduce_C_function=np.mean,linewidths=0.1)
    axs[0].set_xlabel('$k_{x} \, \, d_i$',fontsize=16)
    axs[0].set_ylabel('$k_{y} \, \, d_i$',fontsize=16)
    if(kxlim != None):
        axs[0].set_xlim(-kxlim,kxlim)
        axs[0].set_ylim(-kylim,kylim)
    axs[0].set_aspect('equal')
    plt.colorbar(hxbin0,ax=axs[0],fraction=.05)

    hxbin1 = axs[1].hexbin(_x1temp, _ytemp, cmap='Spectral', C=_ztemp,gridsize=(gridsize1,gridsize3),reduce_C_function=np.mean,linewidths=0.1)
    axs[1].set_xlabel('$k_{x} \, \, d_i$',fontsize=16)
    axs[1].set_ylabel('$k_{z} \, \, d_i$',fontsize=16)
    if(kxlim != None):
        axs[1].set_xlim(-kxlim,kxlim)
        axs[1].set_ylim(-kzlim,kzlim)
    axs[1].set_aspect('equal')
    plt.colorbar(hxbin1,ax=axs[1],fraction=.05)

    hxbin2 = axs[2].hexbin(_x2temp, _ytemp, cmap='Spectral', C=_ztemp,gridsize=(gridsize2,gridsize3),reduce_C_function=np.mean,linewidths=0.1)
    axs[2].set_xlabel('$k_{y} \, \, d_i$',fontsize=16)
    axs[2].set_ylabel('$k_{z} \, \, d_i$',fontsize=16)
    if(kxlim != None):
        axs[2].set_xlim(-kylim,kylim)
        axs[2].set_ylim(-kzlim,kzlim)
    axs[2].set_aspect('equal')
    plt.colorbar(hxbin2,ax=axs[2],fraction=.05)

    plt.subplots_adjust(wspace=.5)

    tpad = 8
    if(key=='normB'):
        axs[0].set_title(r'$<\hat{|\delta \mathbf{B}|}(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<\hat{|\delta \mathbf{B}|}(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<\hat{|\delta \mathbf{B}|}(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(key=='Bpar'):
        axs[0].set_title(r'$<|\delta \hat{B}_{||}|(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{B}_{||}|(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{B}_{||}|(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(key=='Bperp1'):
        axs[0].set_title(r'$<|\delta \hat{B}_{\perp,1}|(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{B}_{\perp,1}|(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{B}_{\perp,1}|(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(key=='Bperp2'):
        axs[0].set_title(r'$<|\delta \hat{B}_{\perp,2}|(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{B}_{\perp,2}|(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{B}_{\perp,2}|(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(key=='normE'):
        axs[0].set_title(r'$<|\delta \hat{\mathbf{E}|}(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{\mathbf{E}|}(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{\mathbf{E}|}(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(key=='Epar'):
        axs[0].set_title(r'$<|\delta \hat{E}_{||}|(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{E}_{||}|(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{E}_{||}|(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(key=='Eperp1'):
        axs[0].set_title(r'$<|\delta \hat{E}_{\perp,1}|(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{E}_{\perp,1}|(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{E}_{\perp,1}|(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(key=='Eperp2'):
        axs[0].set_title(r'$<|\delta \hat{E}_{\perp,2}|(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{E}_{\perp,2}|(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{E}_{\perp,2}|(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(key=='Ex'):
        axs[0].set_title(r'$<|\delta \hat{E}_{x}|(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{E}_{x}|(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{E}_{x}|(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(key=='Ey'):
        axs[0].set_title(r'$<|\delta \hat{E}_{y}|(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{E}_{y}|(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{E}_{y}|(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(key=='Ez'):
        axs[0].set_title(r'$<|\delta \hat{E}_{z}|(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{E}_{z}|(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{E}_{z}|(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(key=='Bx'):
        axs[0].set_title(r'$<|\delta \hat{B}_{x}|(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{B}_{x}|(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{B}_{x}|(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(key=='By'):
        axs[0].set_title(r'$<|\delta \hat{B}_{y}|(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{B}_{y}|(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{B}_{y}|(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(key=='Bz'):
        axs[0].set_title(r'$<|\delta \hat{B}_{z}|(k_{x},k_{y})>_{k_z}$',loc='right',pad=tpad)
        axs[1].set_title(r'$<|\delta \hat{B}_{z}|(k_{x},k_{z})>_{k_y}$',loc='right',pad=tpad)
        axs[2].set_title(r'$<|\delta \hat{B}_{z}|(k_{y},k_{z})>_{k_x}$',loc='right',pad=tpad)

    if(flnm != ''):
        plt.savefig(flnm,dpi=300,format='png',bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return hxbin0, hxbin1, hxbin2

def plot_spec_1d(dwavemodes,flnm='1dspec',key='normE',binkey='kpar',binsize=.2,binlowerbound=0,binupperbound=10):
    #note: compute dwavemodes is for 1 position only
    if(not(isinstance(key, list))):
        key = [key]

    indepvar = [dwavemodes['wavemodes'][_i][binkey] for _i in range(0,len(dwavemodes['wavemodes']))]

    depvarlist = []
    for _key in key:
        ind_bins = np.arange(binlowerbound,binupperbound,binsize)
        bin_indicies = np.digitize(indepvar, ind_bins) #note: this returns 0 if it is left of range, 1 if it is in first bin, 2 in second bin, ..., len(ind_bins) if it is right of range; we account for this by ignoring results out of range, shifting by one, and plotting vs bincenters; otherwise- values are plotted at right bin edge location and leftmost bin will be falsely empty

        num_in_bin = np.zeros(len(ind_bins))
        depvar_binned = np.zeros(len(ind_bins)-1)
        for _i in range(0,len(dwavemodes['wavemodes'])):
            if(bin_indicies[_i] < len(depvar_binned)): #don't want out of bounds data (above range)
                if(dwavemodes['wavemodes'][_i][binkey] >= binlowerbound): #don't want out of bounds data (below range)
                    depvar_binned[bin_indicies[_i]-1] += np.abs(dwavemodes['wavemodes'][_i][_key]) #Want amplitude
                    num_in_bin[bin_indicies[_i]-1] += 1.

        #normalize by number in each bin (might not be the same if the parallel direction is at an odd angle)
        for _i in range(0,len(depvar_binned)):
            if(num_in_bin[_i] > 0): 
                depvar_binned[_i] /= num_in_bin[_i]

        depvarlist.append(depvar_binned)

    bincenters = np.asarray([(ind_bins[_idx]+ind_bins[_idx+1])/2. for _idx in range(0,len(ind_bins)-1)])

    plt.figure(figsize=(6,3))
    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
    
    colors = ['black','red','green','blue']
    lss = ['-','-.','--',':']
    for _i in range(0,len(key)):
        if(key[_i] == 'normE'):
            lgkey = '$|\hat{\mathbf{E}}|$'
        elif(key[_i] == 'normB'):
            lgkey = '$|\hat{\mathbf{B}}|$'
        elif(key[_i] == 'Epar'):
            lgkey = '$|\hat{E}_{||}|$'
        elif(key[_i] == 'Bpar'):
            lgkey = '$|\hat{B}_{||}|$'
        elif(key[_i] == 'Eperp1'):
            lgkey = '$|\hat{E}_{\perp,1}|$'
        elif(key[_i] == 'Bperp1'):
            lgkey = '$|\hat{B}_{\perp,1}|$'
        elif(key[_i] == 'Eperp2'):
            lgkey = '$|\hat{E}_{\perp,2}|$'
        elif(key[_i] == 'Bperp2'):
            lgkey = '$|\hat{B}_{\perp,2}|$'
        elif(key[_i] == 'Epar_local'):
            lgkey = '$|\hat{E}^{local}_{||}|$'
        elif(key[_i] == 'Epar_detrend'):
            lgkey = '$|\hat{E}^{detrend}_{||}|$'
        else:
            lgkey = str(key[_i])

        plt.plot(bincenters,depvarlist[_i],color=colors[_i],lw=1.5,ls=lss[_i],label=lgkey)
   
    if(len(key) == 1):
        if(key[0] == 'normE'):
            plt.ylabel("$|\hat{\mathbf{E}}|$")
        elif(key[0] == 'Epar'):
            plt.ylabel("$|\hat{E}_{||}|$")
        elif(key[0] == 'normB'):
            plt.ylabel("$|\hat{\mathbf{B}}|$")
        elif(key[0] == 'Bpar'):
            plt.ylabel("$|\hat{B}_{||}|$")
        elif(key[0] == 'Eperp1'):
            plt.ylabel("$|\hat{E}_{\perp,1}|$")
        elif(key[0] == 'Bperp1'):
            plt.ylabel("$|\hat{B}_{\perp,1}|$")
        elif(key[0] == 'Eperp2'):
            plt.ylabel("$|\hat{E}_{\perp,2}|$")
        elif(key[0] == 'Bperp2'):
            plt.ylabel("$|\hat{B}_{\perp,2}|$")
        elif(key[0] == 'Epar_local'):
            plt.ylabel('$|\hat{E}^{local}_{||}|$')
        elif(key[0] == 'Epar_detrend'):
            plt.ylabel('$|\hat{E}^{detrend}_{||}|$')
        else:
            plt.ylabel(key)
    else:
        plt.legend()

    if(binkey == 'kpar'):
        plt.xlabel("$k_{||} d_i$")
    elif(binkey == 'kperp1'):
        plt.xlabel("$k_{\perp,1} d_i$")
    else:
        plt.xlabel(binkey)
    
    plt.grid()
    plt.savefig(flnm+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()

def plot_spec_1dnobin(dwavemodes,flnm='1dspecnobin',key='normE',binkey='kpar',binsize=.2,binlowerbound=-10,binupperbound=10):
    #note: compute dwavemodes is for 1 position only

    indepvar = [dwavemodes['wavemodes'][_i][binkey] for _i in range(0,len(dwavemodes['wavemodes']))]
    depvar = [np.abs(dwavemodes['wavemodes'][_i][key]) for _i in range(0,len(dwavemodes['wavemodes']))]

    plt.figure(figsize=(5,3))
    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
    plt.scatter(indepvar,depvar,s=.5)
    if(binkey == 'kpar'):
        plt.xlabel("$k_{||} d_i$")
    else:
        plt.xlabel(binkey)
    if(key == 'normE'):
        plt.ylabel("$|\hat{\mathbf{E}}|$")
    elif(key == 'Epar'):
        plt.ylabel("$\hat{E}_{||}$")
    else:
        plt.ylabel(key)
    plt.xlim(binlowerbound,binupperbound)
    plt.savefig(flnm+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()

def plot_kyxx_box_aligned(WFTdata,plotkey,flnm,kmax = 20):
    """
    """

    plotzz = np.abs(np.sum(WFTdata[plotkey],axis=(0,2))) #reduce from kzkykxxx to ky xx and take norm of complex value

    plt.figure(figsize=(20,15))
    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
    plt.pcolormesh(WFTdata['xx'],WFTdata['ky'],plotzz,shading='nearest')
    cbr = plt.colorbar()
    if(plotkey == 'bxkzkykxxx'):
        cbr.set_label("$|\hat{B}_{x}|$")
    elif(plotkey == 'bykzkykxxx'):
        cbr.set_label("$|\hat{B}_{y}|$")
    elif(plotkey == 'bzkzkykxxx'):
        cbr.set_label("$|\hat{B}_{z}|$")
    elif(plotkey == 'exkzkykxxx'):
        cbr.set_label("$|\hat{E}_{x}|$")
    elif(plotkey == 'eykzkykxxx'):
        cbr.set_label("$|\hat{E}_{y}|$")
    elif(plotkey == 'ezkzkykxxx'):
        cbr.set_label("$|\hat{E}_{z}|$")
    else:
        cbr.set_label(key)

    plt.ylim(-kmax,kmax)

    plt.savefig(flnm,format='png',bbox_inches='tight',dpi=300)
    plt.close()

def plot_spec_2d(WFTdata,dfields,loadflnm='',xxrange=None,flnm='2dspec',key='normE',binkey='kpar',binsize=.2,binlowerbound=-10,binupperbound=10,verbose=False):
    
    from lib.wavemodeaux import compute_wavemodes
    from lib.arrayaux import find_nearest

    xplot = []
    yplot = []
    zplot = []

    if(loadflnm == ''):
        ind_bins = np.arange(binlowerbound,binupperbound,binsize)

        for xidx in range(0,len(WFTdata['xx'])):
            if(xxrange == None or (WFTdata['xx'][xidx] >= xxrange[0] and WFTdata['xx'][xidx] <= xxrange[1])):
                if(verbose):print("In plot_spec_2d, computing ",WFTdata['xx'][xidx])
        
                xidxfields = find_nearest(dfields['ex_xx'],WFTdata['xx'][xidx])
                dx = dfields['ex_xx'][1]-dfields['ex_xx'][0]
                xlim = [dfields['ex_xx'][xidxfields]-dx/2.,dfields['ex_xx'][xidxfields]+dx/2.]
                ylim = [dfields['ex_yy'][0]-dx/2.,dfields['ex_yy'][-1]+dx/2.]
                zlim = [dfields['ex_zz'][0]-dx/2.,dfields['ex_zz'][-1]+dx/2.]

                dwavemodes = compute_wavemodes(None,dfields,xlim,ylim,zlim,
                     WFTdata['kx'],WFTdata['ky'],WFTdata['kz'],
                     WFTdata['bxkzkykxxx'],WFTdata['bykzkykxxx'],WFTdata['bzkzkykxxx'],
                     WFTdata['exkzkykxxx'],WFTdata['eykzkykxxx'],WFTdata['ezkzkykxxx'],
                     WFTdata['uxkzkykxxx'],WFTdata['uykzkykxxx'],WFTdata['uzkzkykxxx'],
                     specifyxxidx=xidx)
                indepvar = [dwavemodes['wavemodes'][_i][binkey] for _i in range(0,len(dwavemodes['wavemodes']))]
                bin_indicies = np.digitize(indepvar, ind_bins)#note: this returns 0 if it is left of range, 1 if it is in first bin, 2 in second bin, ..., len(ind_bins) if it is right of range; we account for this by ignoring results out of range, shifting by one, and plotting vs bincenters; otherwise- values are plotted at right bin edge location and leftmost bin will be falsely empty


                num_in_bin = np.zeros(len(ind_bins))
                depvar_binned = np.zeros(len(ind_bins)-1)
                for _i in range(0,len(dwavemodes['wavemodes'])):
                    if(bin_indicies[_i] < len(depvar_binned)): #don't want out of bounds data (above range)
                        if(dwavemodes['wavemodes'][_i][binkey] >= binlowerbound): #don't want out of bounds data (below range)
                            depvar_binned[bin_indicies[_i]-1] += np.abs(dwavemodes['wavemodes'][_i][key]) #Want amplitude
                num_in_bin[bin_indicies[_i]-1] += 1.

                #normalize by number in each bin (might not be the same if the parallel direction is at an odd angle)
                for _i in range(0,len(depvar_binned)):
                    if(num_in_bin[_i] > 0):
                        depvar_binned[_i] /= num_in_bin[_i]

                xplot.append(WFTdata['xx'][xidx])
                zplot.append(depvar_binned)
        bincenters = np.asarray([(ind_bins[_idx]+ind_bins[_idx+1])/2. for _idx in range(0,len(ind_bins)-1)])
        yplot = bincenters

        #save to pickle
        print("Saving plot data to pickle called " , flnm+'.pickle')

        import pickle
        data = {'xplot':xplot,'yplot':yplot,'zplot':zplot}
        with open(flnm+'.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        import pickle
        with open(loadflnm, 'rb') as handle:
            data = pickle.load(handle)
        
        xplot = data['xplot']
        yplot = data['yplot']
        zplot = data['zplot']

    plt.figure(figsize=(15,5))
    plt.pcolormesh(xplot,yplot,np.asarray(zplot).T,shading='nearest')
    cbr = plt.colorbar()
    if(key == 'normE'):
        cbr.set_label("$|\hat{\mathbf{E}}|$")
    elif(key == 'Epar'):
        cbr.set_label("$|\hat{E}_{||}|$")
    elif(key == 'normB'):
        cbr.set_label("$|\hat{\mathbf{B}}|$")
    elif(key == 'Bpar'):
        cbr.set_label("$|\hat{B}_{||}|$")
    elif(key == 'Eperp1'):
        cbr.set_label("$|\hat{E}_{\perp,1}|$")
    elif(key == 'Bperp1'):
        cbr.set_label("$|\hat{B}_{\perp,1}|$")
    elif(key == 'Eperp2'):
        cbr.set_label("$|\hat{E}_{\perp,2}|$")
    elif(key == 'Bperp2'):
        cbr.set_label("$|\hat{B}_{\perp,2}|$")
    else:
        cbr.set_label(key)

    if(binkey == 'kpar'):
        plt.ylabel("$k_{||} d_i$")
    elif(binkey == 'kperp1'):
        plt.xlabel("$k_{\perp,1} d_i$")
    else:
        plt.ylabel(binkey)

    plt.xlabel("$x / d_i$")

    plt.savefig(flnm+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()

def plot_wlt(xx, kx, wlt, ky0 = None, kz0 = None, flnm = '', xlim = None, ylim = None, xxline = None, yyline = None, clrbarlbl = None,axhline=None):
    """
    """


    plt.figure(figsize=(10,5))
    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
    plt.pcolormesh(xx,kx,np.abs(wlt),cmap='Spectral', shading='gouraud')
    cbar = plt.colorbar()
    if(clrbarlbl != None):
        cbar.set_label(clrbarlbl,labelpad=25, rotation=270)
    plt.xlabel('$x$')
    plt.ylabel('$k_x$')
    plt.grid()
    if(ky0 != None and kz0 != None):
        plt.title('ky='+str(ky0)[0:6]+' kz='+str(kz0)[0:6])
    if(xxline != None and yyline != None):
        plt.plot(xxline,yyline)
    if(xlim != None):
        plt.xlim(xlim[0],xlim[1])
    if(ylim != None):
        plt.ylim(ylim[0],ylim[1])
    if(axhline != None):
        plt.axhline(axhline)

    if(flnm == ''):
        plt.show()
    else:
        #flnm='ky='+str(ky0)[0:6]+'kz='+str(kz0)[0:6]+'wlt'
        plt.savefig(flnm,format='png',dpi=250)
        plt.close('all')#saves RAM
    plt.close()


def plot_wlt_ky(xx, ky, wlt, flnm = '', xlim = None, ylim = None, xxline = None, yyline = None, clrbarlbl = None,axhline=None):
    """
    """


    plt.figure(figsize=(10,5))
    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
    plt.pcolormesh(xx,ky,np.abs(wlt),cmap='Spectral', shading='gouraud')
    cbar = plt.colorbar()
    if(clrbarlbl != None):
        cbar.set_label(clrbarlbl,labelpad=25, rotation=270)
    plt.xlabel('$x$')
    plt.ylabel('$k_y$')
    plt.grid()
    if(xxline != None and yyline != None):
        plt.plot(xxline,yyline)
    if(xlim != None):
        plt.xlim(xlim[0],xlim[1])
    if(ylim != None):
        plt.ylim(ylim[0],ylim[1])
    if(axhline != None):
        plt.axhline(axhline)

    if(flnm == ''):
        plt.show()
    else:
        #flnm='ky='+str(ky0)[0:6]+'kz='+str(kz0)[0:6]+'wlt'
        plt.savefig(flnm,format='png',dpi=250)
        plt.close('all')#saves RAM
    plt.close()
