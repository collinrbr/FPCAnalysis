# 2dfields.py>

# functions related to plotting 1d field data

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_fft_norm(dfields,fieldkey,planename,flnm = '',takeaxisaverage=True, xxindex=float('nan'), yyindex=float('nan'), zzindex=float('nan'), plotlog = True, xaxislim=None, yaxislim=None):
    """
    WIP

    """
    from lib.analysis import take_fft2


    fieldttl = ''
    if(fieldkey == 'ex'):
        fieldttl = '$F\{E_x'
    elif(fieldkey == 'ey'):
        fieldttl = '$F\{E_y'
    elif(fieldkey == 'ez'):
        fieldttl = '$F\{E_z'
    elif(fieldkey == 'bx'):
        fieldttl = '$F\{B_x'
    elif(fieldkey == 'by'):
        fieldttl = '$F\{B_y'
    elif(fieldkey == 'bz'):
        fieldttl = '$F\{B_z'

    if(planename=='xy'):
        ttl = fieldttl+'(x,y)\}$ at '
        xlbl = '$k_x$ (di)$^{-1}$'
        ylbl = '$k_y$ (di)$^{-1}$'
        axisidx = 0 #used to take average along z if no index is specified
        axis = '_zz'
        daxis0 = dfields[fieldkey+'_yy'][1]-dfields[fieldkey+'_yy'][0]
        daxis1 = dfields[fieldkey+'_xx'][1]-dfields[fieldkey+'_xx'][0]

    elif(planename=='xz'):
        ttl = fieldttl+'(x,z)\}$ at '
        xlbl = '$k_x$ (di)$^{-1}$'
        ylbl = '$k_z$ (di)$^{-1}$'
        axisidx = 1 #used to take average along y if no index is specified
        axis = '_yy'
        daxis0 = dfields[fieldkey+'_zz'][1]-dfields[fieldkey+'_zz'][0]
        daxis1 = dfields[fieldkey+'_xx'][1]-dfields[fieldkey+'_xx'][0]

    elif(planename=='yz'):
        ttl = fieldttl+'(y,z)\}$ at '
        xlbl = '$k_y$ (di)$^{-1}$'
        ylbl = '$k_z$ (di)$^{-1}$'
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

    #get x index where data is zero
    #get y index where data is zero
    #get subset based on this
    xzeroidx = np.where(xplot[0] == 0.)[0][0]
    yzeroidx = np.where(yplot[:,0] == 0.)[0][0]
    fieldpmesh[xzeroidx,yzeroidx] = 0.
    if(plotlog):
        fieldpmesh = fieldpmesh[xzeroidx+1:,yzeroidx+1:]
        xplot = xplot[xzeroidx+1:,yzeroidx+1:]
        yplot = yplot[xzeroidx+1:,yzeroidx+1:]
    else:
        fieldpmesh = fieldpmesh[xzeroidx:,yzeroidx:]
        xplot = xplot[xzeroidx:,yzeroidx:]
        yplot = yplot[xzeroidx:,yzeroidx:]


    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots
    plt.figure(figsize=(6.5,6))
    plt.pcolormesh(xplot, yplot, fieldpmesh, cmap="Spectral", shading="gouraud")
    if(xaxislim is not None):
        plt.xlim(0,xaxislim)
    if(yaxislim is not None):
        plt.ylim(0,yaxislim)
    if(takeaxisaverage):
        plt.title(ttl,loc="right")
    elif(planename == 'xy'):
        plt.title(ttl+' z = '+str(dfields[fieldkey+axis][zzindex])+' (di)',loc="right")
    elif(planename == 'xz'):
        plt.title(ttl+' y = '+str(dfields[fieldkey+axis][yyindex])+' (di)',loc="right")
    elif(planename == 'yz'):
        plt.title(ttl+' x = '+str(dfields[fieldkey+axis][xxindex])+' (di)',loc="right")
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

    return #k0, k1, fieldpmesh, xplot, yplot #debug. TODO: remove

def make_2dfourier_sweep(dfields,fieldkey,planename,directory,plotlog=True,xaxislim=None,yaxislim=None):
    """

    """

    try:
        os.mkdir(directory)
    except:
        pass

    #sweep along 'third' axis
    if(planename=='yz'):
        sweepvar = dfields[fieldkey+'_xx'][:]
    elif(planename=='xz'):
        sweepvar = dfields[fieldkey+'_yy'][:]
    elif(planename=='xy'):
        sweepvar = dfields[fieldkey+'_zz'][:]

    for i in range(0,len(sweepvar)):
        print('Making plot '+str(i)+' of '+str(len(sweepvar)))
        flnm = directory+'/'+str(i).zfill(6)
        if(planename=='yz'):
            plot_fft_norm(dfields,fieldkey,planename,flnm = flnm,plotlog=plotlog,takeaxisaverage=False, xxindex=i, yyindex=float('nan'), zzindex=float('nan'),xaxislim=xaxislim,yaxislim=yaxislim)
        elif(planename=='xz'):
            plot_fft_norm(dfields,fieldkey,planename,flnm = flnm,plotlog=plotlog,takeaxisaverage=False, xxindex=float('nan'), yyindex=i, zzindex=float('nan'),xaxislim=xaxislim,yaxislim=yaxislim)
        elif(planename=='xy'):
            plot_fft_norm(dfields,fieldkey,planename,flnm = flnm,plotlog=plotlog,takeaxisaverage=False, xxindex=float('nan'), yyindex=float('nan'), zzindex=i, xaxislim=xaxislim,yaxislim=yaxislim)
        else:
            print("Please enter a valid planename...")
            break

def plot1d_fft(dfields,fieldkey):
    axis = '_xx'
    zzindex = 0
    yyindex = 0
    data = np.asarray([dfields[fieldkey][zzindex][yyindex][i] for i in range(0,len(dfields[fieldkey+axis]))])
    dx = dfields[fieldkey+axis][1]-dfields['ex_xx'][0]


    k0 = 2.*np.pi*np.fft.fftfreq(len(data),dx)
    k0 = k0[0:int(len(k0)/2)]

    fftdata = np.fft.fft(data)
    fftdata = np.real(fftdata*np.conj(fftdata))
    fftdata = fftdata[0:int(len(fftdata)/2)]

    plt.figure()
    plt.xlabel('k'+axis[1:2])
    plt.ylabel(fieldkey+' Power')
    plt.plot(k0,fftdata)
    plt.show()


    return k0,fftdata

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
