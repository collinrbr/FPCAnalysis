# 2dfields.py>

# functions related to plotting 1d field data

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_fft_norm(dfields,fieldkey,planename,flnm = '',takeaxisaverage=True, xxindex=float('nan'), yyindex=float('nan'), zzindex=float('nan'), plotlog = True):
    """
    WIP

    """
    from lib.analysisfunctions import take_fft2

    if(planename=='xy'):
        ttl = fieldkey+'(x,y)'
        xlbl = 'kx (di)'
        ylbl = 'ky (di)'
        axisidx = 0 #used to take average along z if no index is specified
        axis = '_zz'
        daxis0 = dfields[fieldkey+'_yy'][1]-dfields[fieldkey+'_yy'][0]
        daxis1 = dfields[fieldkey+'_xx'][1]-dfields[fieldkey+'_xx'][0]

    elif(planename=='xz'):
        ttl = fieldkey+'(x,z)'
        xlbl = 'kx (di)'
        ylbl = 'kz (di)'
        axisidx = 1 #used to take average along y if no index is specified
        axis = '_yy'
        daxis0 = dfields[fieldkey+'_zz'][1]-dfields[fieldkey+'_zz'][0]
        daxis1 = dfields[fieldkey+'_xx'][1]-dfields[fieldkey+'_xx'][0]

    elif(planename=='yz'):
        ttl = fieldkey+'(y,z)'
        xlbl = 'ky (di)'
        ylbl = 'kz (di)'
        axisidx = 2 #used to take average along x if no index is specified
        axis = '_xx'
        daxis0 = dfields[fieldkey+'_zz'][1]-dfields[fieldkey+'_zz'][0]
        daxis1 = dfields[fieldkey+'_yy'][1]-dfields[fieldkey+'_yy'][0]

    if(takeaxisaverage):
        fieldpmesh = np.mean(dfields[fieldkey],axis=axisidx)
    elif(planename == 'xy'):
        fieldpmesh = np.asarray(dfields[fieldkey])[zzindex,:,:]
    elif(planename == 'xz'):
        fieldpmesh = np.asarray(dfields[fieldkey])[:,yyindex,:]
    elif(planename == 'yz'):
        fieldpmesh = np.asarray(dfields[fieldkey])[:,:,xxindex]

    #take fft of data and compute power
    k0, k1, fieldpmesh = take_fft2(fieldpmesh,daxis0,daxis1)
    fieldpmesh = np.real(fieldpmesh*np.conj(fieldpmesh))/(float(len(k0)*len(k1))) #convert to power

    #plot wavelength (with infinity at 0) for debug
    # k0 = 1./k0
    # k1 = 1./k1
    #
    # for k in range(0,len(k0)):
    #     if(np.isinf(k0[k])):
    #         k0[k] = 0.
    #
    # for k in range(0,len(k1)):
    #     if(np.isinf(k1[k])):
    #         k1[k] = 0.
    #
    # print(k0)

    #make 2d arrays for more explicit plotting
    xplot = np.zeros((len(k0),len(k1)))
    yplot = np.zeros((len(k0),len(k1)))
    for i in range(0,len(k1)):
        for j in range(0,len(k0)):
            xplot[j][i] = k1[i]

    for i in range(0,len(k1)):
        for j in range(0,len(k0)):
            yplot[j][i] = k0[j]

    #sort data so we can plot it
    xplot, yplot, fieldpmesh = _sort_for_contour(xplot, yplot, fieldpmesh)

    if(plotlog):
        #get x index where data is zero
        #get y index where data is zero
        #get subset based on this

        xzeroidx = np.where(xplot[0] == 0.)[0][0]
        yzeroidx = np.where(yplot[:,0] == 0.)[0][0]
        fieldpmesh = fieldpmesh[xzeroidx+1:,yzeroidx+1:]
        xplot = xplot[xzeroidx+1:,yzeroidx+1:]
        yplot = yplot[xzeroidx+1:,yzeroidx+1:]

    # xzeroidx = np.where(xplot[0] == 0.)[0][0]
    # yzeroidx = np.where(yplot[:,0] == 0.)[0][0]
    # fieldpmesh[xzeroidx,yzeroidx] = 0.

    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots
    plt.figure(figsize=(6.5,6))
    plt.figure(figsize=(6.5,6))
    plt.pcolormesh(xplot, yplot, fieldpmesh, cmap="Spectral", shading="gouraud")
    if(takeaxisaverage):
        plt.title(ttl,loc="right")
    elif(planename == 'xy'):
        plt.title(ttl+' z (di): '+str(dfields[fieldkey+axis][zzindex]),loc="right")
    elif(planename == 'xz'):
        plt.title(ttl+' y (di): '+str(dfields[fieldkey+axis][yyindex]),loc="right")
    elif(planename == 'yz'):
        plt.title(ttl+' x (di): '+str(dfields[fieldkey+axis][xxindex]),loc="right")
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    if(plotlog):
        plt.xscale('log')
        plt.yscale('log')
    plt.grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    #clb = plt.colorbar(format="%.1f", ticks=np.linspace(-maxCe, maxCe, 8), fraction=0.046, pad=0.04) #TODO: make static colorbar based on max range of C
    plt.colorbar()
    # plt.xlim(0,.5)
    # plt.ylim(0,.5)
    #plt.setp(plt.gca(), aspect=1.0)
    plt.gcf().subplots_adjust(bottom=0.15)
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png')
        plt.close('all')#saves RAM
    else:
        plt.show()
        plt.close()

    return k0, k1, fieldpmesh, xplot, yplot #debug. TODO: remove

def _sort_for_contour(xcoord,ycoord,dheight):
    """
    Sorts data for use in matplotlibs' countourf/ pmeshgrid plotting functions.
    Sorts xcoord by column (rows are identical), ycoord by row (columns are identical)
    and maintians parallelization with dheight.

    Countourf and pmesh grid are most explicitly plotted when 3 2d arrays are
    passed to it, xcoords ycoords dheight. The 3 2d arrays xxcoords
    are parrallel such that dheight(xcoord[i][j],ycoord[i][j]) = dheight[i][j].
    In some routines (particularly in our fft routine), we build these three arrays
    such that the coordinate arrays are out of order (but all 3 are parallel).
    Thus, we must sort these arrays while maintaining their parallelization

    Parameters
    ----------
    """

    temprowx = xcoord[0]
    xsort = np.argsort(temprowx)
    tempcoly = ycoord[:,0]
    ysort = np.argsort(tempcoly)
    for i in range(0,len(dheight)): #sort by col
        dheight[i] = dheight[i][xsort]
    dheight = dheight[ysort] #sort by row
    xcoord = np.sort(xcoord) #sort x data
    ycoord = np.sort(ycoord,axis=0) #sort y data

    return xcoord, ycoord, dheight
