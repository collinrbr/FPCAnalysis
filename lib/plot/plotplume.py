# plume.py>

# functions to plotting plume results

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_sweep(plume_sweeps,xaxiskey,yaxiskey,wavemodes=[''],xlbl='',ylbl='',lbls=[''],xlim=None,ylim=None):
    """
    WARNING: does NOT check if plume sweeps are different solutions to the same dispersion relation. Assumes they all are.
    """
    plt.figure()
    for idx, p_swp in enumerate(plume_sweeps):
        if(lbls != ['']):
            plt.semilogx(p_swp[xaxiskey],p_swp[yaxiskey],label=lbls[idx])
        else:
            plt.semilogx(p_swp[xaxiskey],p_swp[yaxiskey])
        if(xlbl == ''):
            plt.xlabel(xaxiskey) #TODO: write key to string function
        if(ylbl == ''):
            plt.ylabel(yaxiskey)
        if(lbls != ['']):
            plt.legend()
        if(xlim != None):
            plt.xlim(xlim[0], xlim[1])
        if(ylim != None):
            plt.ylim(ylim[0], ylim[1])
    if(wavemodes !=['']):
        for wm in wavemodes:
            strict_tol = .001
            loose_tol = .1
            if (np.abs(np.linalg.norm(wm['Eperp1'])-1) > strict_tol):
                print('WARNING: wavemodes was not normalized...')
            if (np.abs(wm['kperp2']) > strict_tol):
                print('WARNING: wavemode is not in the correct coordinate system...')
            if (xaxiskey=='kpar' and np.abs(wm['kperp']-p_swp['kperp'][0]) > loose_tol): #we use a loose tolerance as sweeps are often run at kperps rounded to nearest tenth
                print('WARNING: wavemode has a different kperp value than sweep..')

            if(yaxiskey == 'ezr'):
                plt.scatter([wm['kpar']],[wm['Epar'].real])
            if(yaxiskey == 'ezi'):
                plt.scatter([wm['kpar']],[wm['Epar'].imag])
    plt.show()
