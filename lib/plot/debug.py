# debug.py>

# random and less useful plots related to previous debugging efforts

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_1d_dist(dparticles, parkey, vmax, x1, x2, y1, y2, flnm = ''):
    """
    Makes 1d distribution function of particle data within given spatial domain

    Parameters
    ----------
    dparticles : dict
        particle data dictionary
    parkey : str
        key of particle dictionary you want to plot (p1,p2,p3,x1,x2 or x3)
    x1 : float
        lower bound in xx space that you want to count
    x2 : float
        upper bound in xx space that you want to count
    y1 : float
        lower bound in xx space that you want to count
    y2 : float
        upper bound in xx space that you want to count
    flnm : str, optional
        specifies filename if plot is to be saved as png.
        if set to default, plt.show() will be called instead
    """

    #TODO: make gpts particles work for spatial domain
    gptsparticle = (x1 < dparticles['x1'] ) & (dparticles['x1'] < x2) & (y1 < dparticles['x2'] ) & (dparticles['x2'] < y2)
    histdata = dparticles[parkey][gptsparticle]
    binsplt = np.linspace(-vmax,vmax,1000)

    plt.figure()
    plt.hist(histdata, bins = binsplt)
    plt.xlabel(parkey)
    plt.ylabel('n')

    if(flnm != ''):
        plt.savefig(flnm,format='png')
    else:
        plt.show()

    plt.close()

#TODO: figure out enerCEx and normalize it in compute energization
def check_JiEi_vs_CEi(dfields,dflow,dparticles,x,enerCEx,enerCEy,dx,xlim=None,ylim=None,zlim=None):
    """

    """
    from lib.analysis import calc_Ji_Ei
    from lib.analysis import get_num_par_in_box

    #normalize by number of particles in box
    if xlim is not None:
        x1 = xlim[0]
        x2 = x1+dx
        xEnd = xlim[1]
    # If xlim is None, use lower x edge to upper x edge extents
    else:
        x1 = dfields['ex_xx'][0]
        x2 = x1 + dx
        xEnd = dfields['ex_xx'][-1]
    if ylim is not None:
        y1 = ylim[0]
        y2 = ylim[1]
    # If ylim is None, use lower y edge to lower y edge + dx extents
    else:
        y1 = dfields['ex_yy'][0]
        y2 = y1 + dx
    if zlim is not None:
        z1 = zlim[0]
        z2 = zlim[1]
    # If zlim is None, use lower z edge to lower z edge + dx extents
    else:
        z1 = dfields['ex_zz'][0]
        z2 = z1 + dx

    JxExplot = []
    JyEyplot = []
    JzEzplot = []
    enerCExplot = []
    enerCEyplot = []
    i = 0
    while(x2 <= xEnd):
        JxEx,JyEy,JzEz = calc_Ji_Ei(dfields,dflow,x1,x2,y1,y2,z1,z2)
        JxExplot.append(JxEx)
        JyEyplot.append(JyEy)
        JzEzplot.append(JzEz)

        numparinbox = get_num_par_in_box(dparticles,x1,x2,y1,y2,z1,z2)
        enerCExplot.append(enerCEx[i]/numparinbox)
        enerCEyplot.append(enerCEy[i]/numparinbox)
        x1 += dx
        x2 += dx
        i += 1

    plt.figure()
    plt.plot(x,JxExplot,label='JxEx')
    plt.plot(x,JyEyplot,label='JyEy')
    #plt.plot(x,JzEzplot,label='JzEz')
    plt.plot(x,enerCExplot)
    plt.plot(x,enerCEyplot)
    plt.xlabel('x (di)')
    plt.ylabel('Energy(x)')
    plt.legend()
    plt.show()
