# fourier.py>

# functions related to plotting 1d field data

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_fft_norm(dfields,fieldkey,planename,flnm = '',takeaxisaverage=True, xxindex=float('nan'), yyindex=float('nan'), zzindex=float('nan'), plotlog = True, xaxislim=None, yaxislim=None):
    """
    Plot norm of FFT along given plane

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    planename : str
        name of plane you want to get 2d grid of
    flnm : str, optional
        specifies filename if plot is to be saved as png.
        if set to default, plt.show() will be called instead
    takeaxisaverage : bool, optional
        if true, averages data over entire 3rd axis (one not in plane specified by planename)
    xxindex : int
        index of data along xx axis
    yyindex : int
        index of data along yy axis
    zzindex : int
        index of data along zz axis
    xaxislim : float
        upper and lower bound of plot [-xaxislim to xaxislim]
    yaxislim : float
        upper and lower bound of plot [-yaxislim to yaxislim]
    """
    from FPCAnalysis.analysis import take_fft2


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
    # else:
    #     fieldpmesh = fieldpmesh[xzeroidx:,yzeroidx:]
    #     xplot = xplot[xzeroidx:,yzeroidx:]
    #     yplot = yplot[xzeroidx:,yzeroidx:]


    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots
    plt.figure(figsize=(6.5,6))
    plt.pcolormesh(xplot, yplot, fieldpmesh, cmap="Spectral", shading="gouraud")
    if(xaxislim is not None):
        plt.xlim(-xaxislim,xaxislim)
    if(yaxislim is not None):
        plt.ylim(-yaxislim,yaxislim)
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

def make_2dfourier_sweep(dfields,fieldkey,planename,directory,plotlog=False,xaxislim=None,yaxislim=None):
    """
    Makes sweep gif of plots produced by plot_fft_norm
    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    planename : str
        name of plane you want to get 2d grid of
    directory : str
        name of the output directory of each swing png
    plotlog : bool, opt
        if true, makes plot log scale
    xaxislim : float
        upper and lower bound of plot [-xaxislim to xaxislim]
    yaxislim : float
        upper and lower bound of plot [-yaxislim to yaxislim]
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
    """
    Plots fft of 1d line slice along x axis at gridpoint closest to origin (y~=0, z~=0)

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    """
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
    xcoord : 2d array
        xx coordinates of data (independent parameter)
    ycoord : 2d array
        yy coordinates of data (independent parameter)
    dheight : 2d array
        zz value of data (dependent parameter)
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

def plot_wlt(xx, kx, wlt, ky0 = None, kz0 = None, flnm = '', plotstrongestkx = False, xlim = None, ylim = None, xxline = None, yyline = None, clrbarlbl = None,axhline=None):
    """
    (ky kz should be floats if passed (because we commonly take WLT of f(x,ky0,kz0)))
    """

    from FPCAnalysis.array_ops import find_nearest

    plt.figure(figsize=(10,5))
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
    if(plotstrongestkx):
        kxline = []
        for i in range(0,len(xx)):
            kxline.append(kx[find_nearest(wlt[:,i],np.max(wlt[:,i]))])
        plt.plot(xx,kxline)
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
    
    from FPCAnalysis.wavemodeaux import compute_wavemodes
    from FPCAnalysis.arrayaux import find_nearest

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


def plot_wlt_over_field(xx, kx, wlt, fieldvals, ky0 = None, kz0 = None, flnm = '', plotstrongestkx = False, xlim = None, 
               ylim = None, xxline = None, yyline = None, clrbarlbl = None,axhline=None,
               Bz2_over_Bz1 = None, xpos_shock = None, xpos_line = None):
    
    """
    (ky kz should be floats if passed (because we commonly take WLT of f(x,ky0,kz0)))

    same as plot_wlt, but shows bz field for reference
    """

    from FPCAnalysis.array_ops import find_nearest
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    #fig, ax = plt.subplots(2,1,figsize=(10,6),sharex=True, gridspec_kw={'height_ratios': [10, 3]})
    fig = plt.figure(figsize=(10,4.5))
    ax1 = plt.subplot(111)
    #ax2 = ax[1]

    img1 = ax1.pcolormesh(xx,kx,np.abs(wlt),cmap='Spectral', shading='gouraud')
    #cbar = plt.colorbar(img1, ax=ax1)
    #if(clrbarlbl != None):
    #    cbar.set_label(clrbarlbl,labelpad=25, rotation=270, panchor=False)

    ax1.set_ylabel('$k_x \, \, d_{i,0}$')
    ax1.grid()

    if(ylim != None):
        ax1.set_ylim(ylim[0],ylim[1])

    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes("bottom", size="27%", pad=0.08)
    cax = divider.append_axes("right", size="2%", pad=0.08)
    cbar = plt.colorbar( img1, ax=ax1, cax=cax )
    if(clrbarlbl != None):
        cbar.set_label(clrbarlbl,labelpad=25, rotation=270)

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    ax2.plot(xx,fieldvals,color='black',linewidth=1)

    print("plotting hand assigned MHD comparison.....")
    bzmhdrh = np.zeros((len(xx)))
    bzmhdrh_xx = np.zeros((len(xx)))

    bzupstream = fieldvals[-1]

    ax1.axvline(x=xpos_line,color='black')

    make_instant_jump = True #adds 'extra point' to make jump instananeous
    if(make_instant_jump):
        bzmhdrh = np.zeros((len(xx)+1))
        bzmhdrh_xx = np.zeros((len(xx)+1))

    ij_idx_correction = 0 #used to to 'correct' idx
    for _mhdrhidx in range(0,len(bzmhdrh[:])):
        if(xx[_mhdrhidx-ij_idx_correction] <= xpos_shock):
            bzmhdrh[_mhdrhidx] = bzupstream*Bz2_over_Bz1
            bzmhdrh_xx[_mhdrhidx] = xx[_mhdrhidx]
        else:
            if(make_instant_jump):
                bzmhdrh[_mhdrhidx] = bzupstream*Bz2_over_Bz1
                bzmhdrh_xx[_mhdrhidx] = xx[_mhdrhidx-ij_idx_correction]
                _mhdrhidx+=1
                make_instant_jump = False
                ij_idx_correction = 1

            bzmhdrh[_mhdrhidx] = bzupstream
            bzmhdrh_xx[_mhdrhidx] = xx[_mhdrhidx-ij_idx_correction]

    ax2.plot(bzmhdrh_xx,bzmhdrh,ls=':',color='grey')

    ax2.set_xlabel('$x / d_{i,0}$')
    ax2.set_ylabel(r'$\overline{B_z}(x)$')
    ax2.grid()

    ax1.set_xlim(xx[0],xx[-1])
    ax2.set_xlim(xx[0],xx[-1])

    plt.subplots_adjust(hspace=.05)

    #BELOW IS NOT TESTED and most of it needs to be removed
    if(ky0 != None and kz0 != None):
        plt.title('ky='+str(ky0)[0:6]+' kz='+str(kz0)[0:6])
    if(xxline != None and yyline != None):
        plt.plot(xxline,yyline)
    #if(xlim != None):
        #plt.xlim(xlim[0],xlim[1])
    #if(ylim != None):
    #    ax1.set_ylim(ylim[0],ylim[1])
    if(plotstrongestkx):
        kxline = []
        for i in range(0,len(xx)):
            kxline.append(kx[find_nearest(wlt[:,i],np.max(wlt[:,i]))])
        plt.plot(xx,kxline)
    if(axhline != None):
        plt.axhline(axhline)

    if(flnm == ''):
        plt.show()
    else:
        #flnm='ky='+str(ky0)[0:6]+'kz='+str(kz0)[0:6]+'wlt'
        plt.savefig(flnm,format='png',dpi=1200,bbox_inches='tight')
        plt.close('all')#saves RAM
    plt.close()

