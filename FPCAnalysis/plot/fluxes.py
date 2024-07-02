# fluxes.py>

# functions related to plotting shock fluxes

import numpy as np
import matplotlib.pyplot as plt
import os

def individual_flux_percents(dfluxes,flnm=''):
    
    #make indiv plots for fluxes
    ionframx = dfluxes['ionframx']
    ionqxs = dfluxes['ionqxs']
    elecqxs = dfluxes['elecqxs']
    ionethxs = dfluxes['ionethxs']
    elecethxs = dfluxes['elecethxs']
    ionpdotusxs = dfluxes['ionpdotusxs']
    elecpdotusxs = dfluxes['elecpdotusxs']
    poyntxxs = dfluxes['poyntxxs']

    interpolxxs = dfluxes['interpolxxs']
    
    plabels = [r'$S_x$',r'$\mathcal{P}_e \cdot \mathbf{U}_e$',r'$3/2 p_e  U_{x,e}$',r'$q_{x,e}$',r'$\mathcal{P}_i \cdot \mathbf{U}_i$',r'$3/2 p_i  U_{x,i}$',r'$q_{x,i}$',r'$F_{x,ram,i}$']
    plabels.reverse()
    fluxes = [ionframx,ionqxs,ionethxs,ionpdotusxs,elecqxs,elecethxs,elecpdotusxs,poyntxxs]
    fluxesvarname = ['ionframx','ionqxs','ionethxs','ionpdotusxs','elecqxs','elecethxs','elecpdotusxs','poyntxxs']
    normfac = np.abs(ionframx[-2]+ionqxs[-2]+ionethxs[-2]+ionpdotusxs[-2]+elecqxs[-2]+elecethxs[-2]+elecpdotusxs[-2]+poyntxxs[-2])
    import os
    for _i in range(0,len(plabels)):
        plt.figure(figsize=(16, 6))
        plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
        xx=interpolxxs[0:-1]
        plt.plot(xx,fluxes[_i]/normfac)
        plt.grid()
        plt.xlim(0,12)
        plt.xlabel(r'$x/d_i$')
        plt.ylabel(plabels[_i]+r'$/|(\sum_s Q_s +S)_{x,up}|$')
        plt.show()
        
        plt.figure(figsize=(16, 6))
        plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
        xx=interpolxxs[0:-1]
        plt.plot(xx,fluxes[_i])
        plt.grid()
        plt.xlim(0,12)
        plt.xlabel(r'$x/d_i$')
        plt.ylabel(plabels[_i])
        if(flnm == ''):
            plt.show()
        else:
            plt.savefig(flnm+fluxesvarname[_i]+'.png')
        plt.close()
    
    plt.figure(figsize=(16, 6))
    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
    xx=interpolxxs[0:-1]
    plt.plot(xx,(ionframx+ionqxs+ionethxs+ionpdotusxs+elecqxs+elecethxs+elecpdotusxs+poyntxxs)/normfac)
    plt.grid()
    plt.xlim(0,12)
    plt.xlabel(r'$x/d_i$')
    plt.ylabel(r'$(\sum_s Q_s +S)/(|\sum_s Q_s +S)_{x,up}|$')
    plt.show()
    if(flnm == ''):
        plt.show()
    else:
        plt.savefig(flnm+'tot'+'.png')
    plt.close()

def stack_plot_pos_neg_flux(dfluxes,flnm = '',xlim=[]):

    from FPCAnalysis.array_ops import split_positive_negative

    ionframx = dfluxes['ionframx']
    ionqxs = dfluxes['ionqxs']
    elecqxs = dfluxes['elecqxs']
    ionethxs = dfluxes['ionethxs']
    elecethxs = dfluxes['elecethxs']
    ionpdotusxs = dfluxes['ionpdotusxs']
    elecpdotusxs = dfluxes['elecpdotusxs']
    poyntxxs = dfluxes['poyntxxs']

    interpolxxs = dfluxes['interpolxxs']
  
    ionframxpos,ionframxneg = split_positive_negative(ionframx)
    ionqxspos, ionqxsneg = split_positive_negative(ionqxs)
    elecqxspos, elecqxsneg = split_positive_negative(elecqxs)
    ionethxspos, ionethxsneg = split_positive_negative(ionethxs)
    elecethxspos, elecethxsneg = split_positive_negative(elecethxs)
    ionpdotusxspos, ionpdotusxsneg = split_positive_negative(ionpdotusxs)
    elecpdotusxspos, elecpdotusxsneg = split_positive_negative(elecpdotusxs)
    poyntxxspos, poyntxxsneg = split_positive_negative(poyntxxs)
    
    import matplotlib.patches as mpatches
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), gridspec_kw={'width_ratios': [5, 1]})
    
    xx=interpolxxs[0:-1]
    runtotpos = np.zeros(len(xx))
    runtotneg = np.zeros(len(xx))
    
    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
    
    normtot = np.abs(ionframx[-2]+ionqxs[-2]+elecqxs[-2]+ionethxs[-2]+elecethxs[-2]+ionpdotusxs[-2]+elecpdotusxs[-2]+poyntxxs[-2])
    
    alpha = 0.8
    tolfrac = 0.001
    
    #we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
    if(np.max(np.abs(ionframxpos)) < np.abs(normtot)*tolfrac):
        alpha = 0.
    else:
        alpha = 0.8
    fill_patchionframxpos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+ionframxpos)/normtot,hatch='++', color='blue', alpha=alpha)
    runtotpos += ionframxpos
    
    #we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
    if(np.max(np.abs(ionqxspos)) < np.abs(normtot)*tolfrac):
        alpha = 0.
    else:
        alpha = 0.8
    fill_patchionqxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+ionqxspos)/normtot,hatch='\\', color='gray', alpha=alpha)
    runtotpos += ionqxspos
    
    #we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
    if(np.max(np.abs(ionethxspos)) < np.abs(normtot)*tolfrac):
        alpha = 0.
    else:
        alpha = 0.8
    fill_patchionethxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+ionethxspos)/normtot,hatch='/', color='green', alpha=alpha)
    runtotpos += ionethxspos
    
    #we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
    if(np.max(np.abs(ionpdotusxspos)) < np.abs(normtot)*tolfrac):
        alpha = 0.
    else:
        alpha = 0.8
    fill_patchionpdotusxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+ionpdotusxspos)/normtot, hatch='x', color='purple', alpha=alpha)
    runtotpos += ionpdotusxspos
    
    #we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
    if(np.max(np.abs(elecqxspos)) < np.abs(normtot)*tolfrac):
        alpha = 0.
    else:
        alpha = 0.8
        # Create mask for where the difference exceeds the threshold
    mask = np.abs(elecqxspos) > np.abs(normtot)*.01 #We only implent it here as for this specific case, it is only needed here! It probably would be better to just use the mask everywhere instead of changing alpha!
    fill_patchelecqxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+elecqxspos)/normtot,where=mask,hatch='+', color='red', alpha=alpha)
    runtotpos += elecqxspos
    
    #we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
    if(np.max(np.abs(elecethxspos)) < np.abs(normtot)*tolfrac):
        alpha = 0.
    else:
        alpha = 0.8
    fill_patchelecethxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+elecethxspos)/normtot,hatch='-', color='orange', alpha=alpha)
    runtotpos += elecethxspos
    
    #we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
    if(np.max(np.abs(elecpdotusxspos)) < np.abs(normtot)*tolfrac):
        alpha = 0.
    else:
        alpha = 0.8
    fill_patchelecpdotusxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+elecpdotusxspos)/normtot, hatch='|', color='pink', alpha=alpha)
    runtotpos += elecpdotusxspos
    
    #we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
    if(np.max(np.abs(poyntxxspos)) < np.abs(normtot)*tolfrac):
        alpha = 0.
    else:
        alpha = 0.8
    fill_patchpoyntxxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+poyntxxspos)/normtot, hatch='///', color='black', alpha=alpha)
    runtotpos += poyntxxspos
    
    #note: we only hide small positive areas since in this specific case, it only creates the illusion of positive contributions for the positive areas
    alpha = 0.8
    
    fill_patchionframxneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+ionframxneg)/normtot,hatch='++', color='blue', alpha=alpha)
    runtotneg += ionframxneg
    
    fill_patchionqxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+ionqxsneg)/normtot,hatch='\\', color='gray', alpha=alpha)
    runtotneg += ionqxsneg
    
    fill_patchionethxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+ionethxsneg)/normtot,hatch='/', color='green', alpha=alpha)
    runtotneg += ionethxsneg
    
    fill_patchionpdotusxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+ionpdotusxsneg)/normtot, hatch='x', color='purple', alpha=alpha)
    runtotneg += ionpdotusxsneg
    
    fill_patchelecqxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+elecqxsneg)/normtot,hatch='+', color='red', alpha=alpha)
    runtotneg += elecqxsneg
    
    fill_patchelecethxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+elecethxsneg)/normtot,hatch='-', color='orange', alpha=alpha)
    runtotneg += elecethxsneg
    
    fill_patchelecpdotusxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+elecpdotusxsneg)/normtot, hatch='|', color='pink', alpha=alpha)
    runtotneg += elecpdotusxsneg
    
    fill_patchpoyntxxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+poyntxxsneg)/normtot, hatch='///', color='black', alpha=alpha)
    runtotneg += poyntxxsneg
    
    # Create a legend for the fill_between plot on the second subplot
    legend_elements = []
    
    _idx = 0
    plabels = [r'$S_x$',r'$(\mathcal{P}_e \cdot \mathbf{U}_e)_x$',r'$3/2 p_e  U_{x,e}$',r'$q_{x,e}$',r'$(\mathcal{P}_i \cdot \mathbf{U}_i)_x$',r'$3/2 p_i  U_{x,i}$',r'$q_{x,i}$',r'$F_{x,ram,i}$']
    plabels.reverse()
    for fp in [fill_patchionframxneg,fill_patchionqxsneg,fill_patchionethxsneg,fill_patchionpdotusxsneg,fill_patchelecqxsneg,fill_patchelecethxsneg,fill_patchelecpdotusxsneg,fill_patchpoyntxxsneg]:
        facecolor = fp.get_facecolor()
        alpha = fp.get_alpha()
        hatch = fp.get_hatch()
        edgecolor = fp.get_edgecolor()
        legend_elements.append(mpatches.Patch(facecolor=facecolor, alpha=alpha, hatch=hatch, edgecolor=edgecolor, label=plabels[_idx]))
        _idx += 1
    
    legend_elements.reverse()
    
    ax2.legend(handles=legend_elements, loc='center',bbox_to_anchor=(0, .4, 0.25, 0.25),fontsize=12)
    
    ax1.set_xlabel(r'$x/d_i$')
    ax1.grid()
    if(xlim != []):
        ax1.set_xlim(xlim[0],xlim[1])
    #ax1.set_ylim(0,2.0)
    ax2.set_axis_off()
    if(flnm == ''):
        plt.show()
    else:
        plt.show()
    plt.close()
    
