# velsig.py>

# functions related to velocity signatures and 2d distribution functions

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_velsig(vx,vy,vz,dv,vmax,CEiproj,fieldkey,planename,ttl=r'$C_{E_i}(v_i,v_j)$; ',flnm='',xlabel=r"$v_i/v_{ti}$",ylabel=r"$v_j/v_{ti}$",plotLog=False,computeJdotE=True,axvlinex = None, maxCe = None):

    from FPCAnalysis.array_ops import mesh_3d_to_2d
    import matplotlib
    from matplotlib.colors import LogNorm
    from FPCAnalysis.array_ops import array_3d_to_2d
    import matplotlib.colors as colors
    from FPCAnalysis.analysis import compute_energization


    CEiplot = CEiproj
    yplot, xplot = mesh_3d_to_2d(vx,vy,vz,planename)

    plt.style.use('cb.mplstyle')

    if(maxCe == None):
        maxCe = max(np.max(CEiplot),abs(np.max(CEiplot)))

    plt.figure(figsize=(6.5,6))
    if(plotLog):
        im = plt.pcolormesh(xplot,yplot,CEiplot,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im = plt.pcolormesh(xplot,yplot,CEiplot,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().set_aspect('equal', 'box')
    plt.grid()
    if(computeJdotE):
        JdotE = compute_energization(CEiplot,dv)
        if(fieldkey == 'ex'):
            plt.gca().set_title(ttl+'$J  E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
        elif(fieldkey == 'ey'):
            plt.gca().set_title(ttl+'$J  E_y$ = ' + "{:.2e}".format(JdotE),loc='left')
        elif(fieldkey == 'ez'):
            plt.gca().set_title(ttl+'$J  E_z$ = ' + "{:.2e}".format(JdotE),loc='left')
        else:
            plt.gca().set_title(ttl+'$J  E_i$ = ' + "{:.2e}".format(JdotE),loc='left')
    else:
        plt.gca().set_title(ttl,loc='left')

    clrbar = plt.colorbar(im, ax=plt.gca())#,format='%.1e')
    if(not(plotLog)):
        clrbar.formatter.set_powerlimits((0, 0))

    plt.xlim(-vmax,vmax)
    plt.ylim(-vmax,vmax)

    if(axvlinex != None):
        plt.axvline(axvlinex)

    if(flnm != ''):
        print("Saving fig as",flnm)
        plt.savefig(flnm+'.png',format='png',dpi=300,bbox_inches='tight')
        plt.close('all')#saves RAM
    else:
        plt.show()
    plt.close()
    # fig = plt.gcf()
    # ax = plt.gca()

    #return ax, fig


#TODO: update/ remove this
def plot_velsig_old(vx,vy,vmax,Ce,fieldkey,flnm = '',ttl=''):
    """
    # Plots correlation data from make2dHistandCex,make2dHistandCey,etc

    Parameters
    ----------
    vx : 2d array
        vx velocity grid
    vy : 2d array
        vy velocity grid
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    Ce : 2d array
        velocity space sigature data
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    flnm : str, optional
        specifies filename if plot is to be saved as png.
        if set to default, plt.show() will be called instead
    ttl : str, optional
        title of plot
    """
    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    maxCe = max(np.max(Ce),abs(np.max(Ce)))

    #ordering when plotting is flipped
    #see https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.pcolormesh.html
    plotv1 = vy
    plotv2 = vx

    plt.figure(figsize=(6.5,6))
    plt.pcolormesh(plotv1, plotv2, Ce, vmax=maxCe, vmin=-maxCe, cmap="seismic", shading="gouraud")
    plt.xlim(-vmax, vmax)
    plt.ylim(-vmax, vmax)
    plt.xticks(np.linspace(-vmax, vmax, 9))
    plt.yticks(np.linspace(-vmax, vmax, 9))
    plt.title(ttl+" $C($"+fieldkey+"$)(v_x, v_y)$",loc="right")
    plt.xlabel(r"$v_x/v_{ti}$")
    plt.ylabel(r"$v_y/v_{ti}$")
    plt.grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    clb = plt.colorbar(format="%.1f", ticks=np.linspace(-maxCe, maxCe, 8), fraction=0.046, pad=0.04) #TODO: make static colorbar based on max range of C
    plt.setp(plt.gca(), aspect=1.0)
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.savefig("CExxposindex"+str(xposindex)+".png", dpi=300) #TODO: rename
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png')
        plt.close('all')#saves RAM
    else:
        plt.show()
    plt.close()

def make_velsig_gif(vx, vy, vmax, C, fieldkey, x_out, directory, flnm):
    """
    Makes gif of velocity space signatures that sweeps over x for C(x; vy, vx)

    Parameters
    ----------
    vx : 2d array
        vx velocity grid
    vy : 2d array
        vy velocity grid
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    C : 3d array
        correlation data over x slices (C(x; vy, vx))
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    x_out : 1d array
        array of x position of C data
    directory : str
        name of directory you want to create and put plots into
        (omit final '/')
    flnm : str
        filename of the final gif
    """

    #make plots of data and put into directory
    try:
        os.mkdir(directory)
    except:
        pass
    for i in range(0,len(C)):
        print('Making plot ' + str(i)+' of '+str(len(C)))
        ttl = directory+'/'+str(i).zfill(6)
        plot_velsig(vx,vy,vmax,C[i],fieldkey,flnm = ttl,ttl='x(di): '+str(x_out[i]))
        plt.close('all')

    #make gif
    import imageio #should import here as it might not be installed on every machine
    images = []
    filenames =  os.listdir(directory)
    filenames = sorted(filenames)
    try:
        filenames.remove('.DS_store')
    except:
        pass

    for filename in filenames:
        images.append(imageio.imread(directory+'/'+filename))
    imageio.mimsave(flnm, images)

def plot_velsig_wEcrossB(vx,vy,vmax,Ce,ExBvx,ExBvy,fieldkey,flnm = '',ttl=''):
    """
    Plots correlation data from make2dHistandCex,make2dHistandCey,etc with
    point at ExBvx, ExBvy.

    Parameters
    ----------
    vx : 2d array
        vx velocity grid
    vy : 2d array
        vy velocity grid
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    Ce : 2d array
        velocity space sigature data
    ExBvx : float
        x coordinate of point
    ExBvy : float
        y coordinate of point
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    flnm : str, optional
        specifies filename if plot is to be saved as png.
        if set to default, plt.show() will be called instead
    ttl : str, optional
        title of plot
    """

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    maxCe = max(np.max(Ce),abs(np.min(Ce)))

    #ordering when plotting is flipped
    #see https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.pcolormesh.html
    plotv1 = vy
    plotv2 = vx

    plt.figure(figsize=(6.5,6))
    plt.figure(figsize=(6.5,6))
    plt.pcolormesh(plotv1, plotv2, Ce, vmax=maxCe, vmin=-maxCe, cmap="seismic", shading="gouraud")
    plt.xlim(-vmax, vmax)
    plt.ylim(-vmax, vmax)
    plt.xticks(np.linspace(-vmax, vmax, 9))
    plt.yticks(np.linspace(-vmax, vmax, 9))
    plt.title(ttl+" $C($"+fieldkey+"$)(v_x, v_y)$",loc="right")
    plt.xlabel(r"$v_x/v_{ti}$")
    plt.ylabel(r"$v_y/v_{ti}$")
    plt.grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    clb = plt.colorbar(format="%.1f", ticks=np.linspace(-maxCe, maxCe, 8), fraction=0.046, pad=0.04) #TODO: make static colorbar based on max range of C
    plt.setp(plt.gca(), aspect=1.0)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.scatter([ExBvx],[ExBvy])
    #plt.savefig("CExxposindex"+str(xposindex)+".png", dpi=300) #TODO: rename
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png')
        plt.close('all')#saves RAM
    else:
        plt.show()
    plt.close()


def make_velsig_gif_with_EcrossB(vx, vy, vmax, C, fieldkey, x_out, dx, dfields, directory, flnm, xlim = None, ylim = None, zlim = None):
    """
    Makes gif of velocity space signatures that sweeps over x for C(x; vy, vx)

    Parameters
    ----------
    vx : 2d array
        vx velocity grid
    vy : 2d array
        vy velocity grid
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    C : 3d array
        correlation data over x slices (C(x; vy, vx))
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    x_out : 1d array
        array of x position of C data
    dx : float
        width of x slice
    dfields : dict
        field data dictionary from field_loader
    directory : str
        name of directory you want to create and put plots into
        (omit final '/')
    flnm : str
        filename of the final gif
    xlim : array
        array of limits in x, defaults to None
    ylim : array
        array of limits in y, defaults to None
    zlim : array
        array of limits in z, defaults to None
    """

    from lib.analysis import calc_E_crossB

    #set up sweeping box
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

    #make plots of data and put into directory
    try:
        os.mkdir(directory)
    except:
        pass

    i = 0
    while(x2 <= xEnd):
        print('Making plot ' + str(i)+' of '+str(len(C)))
        flnm = directory+'/'+str(i).zfill(6)
        ExBvx, ExBvy, _ = calc_E_crossB(dfields,x1,x2,y1,y2,z1,z2)
        plot_velsig_wEcrossB(vx,vy,vmax,C[i],ExBvx,ExBvy,fieldkey,flnm = flnm,ttl='x(di): '+str(x_out[i]))
        x1 += dx
        x2 += dx
        i += 1
        plt.close('all')

    #make gif
    import imageio #should import here as it might not be installed on every machine
    images = []
    filenames =  os.listdir(directory)
    filenames = sorted(filenames)
    try:
        filenames.remove('.DS_store')
    except:
        pass

    # for filename in filenames:
    #     images.append(imageio.imread(directory+'/'+filename))
    # imageio.mimsave(flnm, images)

def plot_dist(vx, vy, vmax, H,flnm = '',ttl=''):
    """
    Plots distribution

    Parameters
    ----------
    vx : 2d array
        vx velocity grid
    vy : 2d array
        vy velocity grid
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    H : 2d array
        distribution data
    """
    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    plt.figure(figsize=(6.5,6))
    plt.figure(figsize=(6.5,6))
    plt.pcolormesh(vy, vx, H, cmap="plasma", shading="gouraud")
    plt.xlim(-vmax, vmax)
    plt.ylim(-vmax, vmax)
    plt.xticks(np.linspace(-vmax, vmax, 9))
    plt.yticks(np.linspace(-vmax, vmax, 9))
    if(ttl == ''):
        plt.title(r"$f(v_x, v_y)$",loc="right")
    else:
        plt.title(ttl)
    plt.xlabel(r"$v_x/v_{ti}$")
    plt.ylabel(r"$v_y/v_{ti}$")
    plt.grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    plt.colorbar()
    plt.gcf().subplots_adjust(bottom=0.15)
    if(flnm != ''):
        plt.savefig(flnm,format='png')
    else:
        plt.show()
    plt.close()

def dist_log_plot_3dir(vx, vy, vz, vmax, H_in, flnm = '',ttl='',xlbl=r"$v_x/v_{ti}$",ylbl=r"$v_y/v_{ti}$",zlbl=r"$v_z/v_{ti}$",plotSymLog=False):
    """
    Makes 3 panel plot of the distribution function in log space

    WARNING: gouraud shading seems to only work on the first figure (i.e. the colormesh for only axs[0] is smoothed)
    This seems to possibly be a larger bug in matplotlib

    Paramters
    ---------
    vx : 2d array
        vx velocity grid
    vy : 2d array
        vy velocity grid
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    H_in : 2d array
        distribution data
    flnm : str, optional
        specifies filename if plot is to be saved as png.
        if set to default, plt.show() will be called instead
    ttl : str, optional
        title of plot
    plotSymLog : bool, optional
        if true, will plot 'negative and positive logarithmic' with linear scale near zero
    """

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
    from FPCAnalysis.array_ops import mesh_3d_to_2d
    import matplotlib
    from matplotlib.colors import LogNorm
    from FPCAnalysis.array_ops import array_3d_to_2d
    import matplotlib.colors as colors

    from copy import copy
    H = copy(H_in) #deep copy

    #get lowest nonzero number
    minval = np.min(H[np.nonzero(H)])

    # #set all zeros to small value
    # H[np.where(H == 0)] = 10**-100

    H_xy = array_3d_to_2d(H,'xy')
    H_xz = array_3d_to_2d(H,'xz')
    H_yz = array_3d_to_2d(H,'yz')

    return dist_log_plot_3dir_2v(vx, vy, vz, vmax, H_xy, H_xz, H_yz, flnm = flnm,ttl=ttl,xlbl=xlbl,ylbl=ylbl,zlbl=zlbl,plotSymLog=plotSymLog)

def dist_log_plot_3dir_2v(vx, vy, vz, vmax, H_xy, H_xz, H_yz, flnm = '',ttl='',xlbl=r"$v_x/v_{ti}$",ylbl=r"$v_y/v_{ti}$",zlbl=r"$v_z/v_{ti}$",plotSymLog=False):
    """
    Makes 3 panel plot of the distribution function in log space if dist function has already been projected.

    WARNING: gouraud shading seems to only work on the first figure (i.e. the colormesh for only axs[0] is smoothed)
    This seems to possibly be a larger bug in matplotlib

    Paramters
    ---------
    vx : 2d array
        vx velocity grid
    vy : 2d array
        vy velocity grid
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    H_in : 2d array
        distribution data
    flnm : str, optional
        specifies filename if plot is to be saved as png.
        if set to default, plt.show() will be called instead
    ttl : str, optional
        title of plot
    plotSymLog : bool, optional
        if true, will plot 'negative and positive logarithmic' with linear scale near zero
    """

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
    from FPCAnalysis.array_ops import mesh_3d_to_2d
    import matplotlib
    from matplotlib.colors import LogNorm
    from FPCAnalysis.array_ops import array_3d_to_2d
    import matplotlib.colors as colors

    #from copy import copy
    #H = copy(H_in) #deep copy

    #get lowest nonzero number
    minval = np.min([np.min(H_xy[np.nonzero(H_xy)]),np.min(H_xz[np.nonzero(H_xz)]),np.min(H_yz[np.nonzero(H_yz)])])
    maxval = np.max([H_xy, H_xz, H_yz])

    # #set all zeros to small value
    # H[np.where(H == 0)] = 10**-100

    vx_xy, vy_xy = mesh_3d_to_2d(vx,vy,vz,'xy')
    vx_xz, vz_xz = mesh_3d_to_2d(vx,vy,vz,'xz')
    vy_yz, vz_yz = mesh_3d_to_2d(vx,vy,vz,'yz')

    fntsize = 10
    plt.rcParams.update({'font.size': fntsize})

    fig, axs = plt.subplots(1,3,figsize=(3*3.25,1*3.25))
    cmap = matplotlib.cm.get_cmap('plasma')
    bkgcolor = 'black'
    numtks = 5
    if(not(plotSymLog)):
        cmap.set_under(bkgcolor) #this doesn't really work like it's supposed to, so we just change the background color to black

    if(plotSymLog):
        _vmax = np.max([-1*np.min(H_xy),np.max(H_xy)])
        pcm0 = axs[0].pcolormesh(vy_xy, vx_xy, H_xy, cmap='PiYG', shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-1*_vmax, vmax=_vmax))
    else:
        axs[0].set_facecolor(bkgcolor)
        pcm0 = axs[0].pcolormesh(vy_xy, vx_xy, H_xy, cmap=cmap, shading="gouraud",norm=LogNorm(vmin=minval, vmax=maxval))
    axs[0].set_xlim(-vmax, vmax)
    axs[0].set_ylim(-vmax, vmax)
    axs[0].set_xticks(np.linspace(-vmax, vmax, numtks))
    axs[0].set_yticks(np.linspace(-vmax, vmax, numtks))
    # if(ttl == ''):
    #     plt.title(r"$f(v_x, v_y)$",loc="right")
    # else:
    #     plt.title(ttl)
    axs[0].set_xlabel(xlbl,fontsize = fntsize)
    axs[0].set_ylabel(ylbl,fontsize = fntsize)
    axs[0].grid(color="grey", linestyle="--", linewidth=1.0, alpha=0.6)
    axs[0].set_aspect('equal', 'box')
    #axs[0].colorbar(cmap = cmap, extend='min')
    #axs[0].gcf().subplots_adjust(bottom=0.15)

    #if data is 2V only, then there is only 1 point in vz, so it can't be plotted by pcolormesh
    #TODO: clean this up
    if(plotSymLog):
        _vmax = np.max([-1*np.min(H_xz),np.max(H_xz)])
        pcm1 = axs[1].pcolormesh(vz_xz, vx_xz, H_xz, cmap='PiYG', shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-1*_vmax, vmax=_vmax))
    else:
        axs[1].set_facecolor(bkgcolor)
        pcm1 = axs[1].pcolormesh(vz_xz,vx_xz,H_xz, cmap=cmap, shading="gouraud",norm=LogNorm(vmin=minval, vmax=maxval))
    axs[1].set_xlim(-vmax, vmax)
    axs[1].set_ylim(-vmax, vmax)
    axs[1].set_xticks(np.linspace(-vmax, vmax, numtks))
    axs[1].set_yticks(np.linspace(-vmax, vmax, numtks))
    axs[1].set_xlabel(xlbl,fontsize = fntsize)
    axs[1].set_ylabel(zlbl,fontsize = fntsize)
    axs[1].grid(color="grey", linestyle="--", linewidth=1.0, alpha=0.6)
    axs[1].set_aspect('equal', 'box')
    #axs[1].colorbar(cmap = cmap, extend='min')

    if(plotSymLog):
        _vmax = np.max([-1*np.min(H_yz),np.max(H_yz)])
        pcm2 = axs[2].pcolormesh(vz_yz, vy_yz, H_yz.T, cmap='PiYG', shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-1*_vmax, vmax=_vmax))
    else:
        axs[2].set_facecolor(bkgcolor)
        pcm2 = axs[2].pcolormesh(vz_yz,vy_yz,H_yz.T, cmap=cmap, shading="gouraud",norm=LogNorm(vmin=minval, vmax=maxval))
    axs[2].set_xlim(-vmax, vmax)
    axs[2].set_ylim(-vmax, vmax)
    axs[2].set_xticks(np.linspace(-vmax, vmax, numtks))
    axs[2].set_yticks(np.linspace(-vmax, vmax, numtks))
    axs[2].set_xlabel(ylbl,fontsize = fntsize)
    axs[2].set_ylabel(zlbl,fontsize = fntsize)
    axs[2].grid(color="grey", linestyle="--", linewidth=1.0, alpha=0.6)
    axs[2].set_aspect('equal', 'box')
    #axs[2].colorbar(cmap = cmap, extend='min')

    fig.colorbar(pcm0, ax=axs[0])
    fig.colorbar(pcm1, ax=axs[1])
    fig.colorbar(pcm2, ax=axs[2])

    plt.subplots_adjust(hspace=.5,wspace=.5)

    if(flnm != ''):
        plt.savefig(flnm,format='png')
    else:
        plt.show()

    return axs, fig

def plot_cor_and_dist_supergrid(vx, vy, vz, vmax,
                                H_xy, H_xz, H_yz,
                                CEx_xy,CEx_xz, CEx_yz,
                                CEy_xy,CEy_xz, CEy_yz,
                                CEz_xy,CEz_xz, CEz_yz,
                                flnm = '', ttl = '', computeJdotE = True, params = None, metadata = None, xpos = None, plotLog = False, plotLogHist = True,
                                plotFAC = False, plotFluc = False, plotAvg = False, isIon = True, listpos=False,xposval=None,normtoN = False,Nval = None, isLowPass=False,isHighPass=False,plotDiagJEOnly=True,
				vxmin=None, vxmax=None, vymin=None, vymax=None, vzmin=None, vzmax=None):
    """
    Makes super figure of distribution and velocity sigantures from all different projections
    i.e. different viewing angles

    Parameters
    ----------
    vx : 2d array
        vx velocity grid
    vy : 2d array
        vy velocity grid
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    H_** : 2d array
        projection onto ** axis of distribution function
    CE*_* : 2d array
        projection onto ** axis of CE*
    flnm : str, optional
        specifies filename if plot is to be saved as png.
        if set to default, plt.show() will be called instead
    ttl : str, optional
        title of plot
    computeJdotE : bool, optional
        compute and write JdotE for each panel as title of each sub plot
    params : dict, optional
        dictionary with simulation parameters in it
    metadata : string, optional
        string describing metadata to be shown on plot
    """
    from matplotlib.colors import LogNorm
    from FPCAnalysis.array_ops import mesh_3d_to_2d
    from FPCAnalysis.analysis import compute_energization
    import matplotlib.colors as colors

    if(normtoN):
        if(Nval == None):
            Nval = np.sum(H_xy)
        CEx_xy/=Nval 
        CEx_xz/=Nval
        CEx_yz/=Nval
        CEy_xy/=Nval
        CEy_xz/=Nval
        CEy_yz/=Nval
        CEz_xy/=Nval
        CEz_xz/=Nval
        CEz_yz/=Nval

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    fig, axs = plt.subplots(4,3,figsize=(4*5,3*5),sharex=False)

    plt.rcParams['axes.titlepad'] = 8  # pad is in points...

    _hspace = .15
    _wspace = -.15
    if(computeJdotE):
        _hspace+=.275
    if(plotLog):
        _wspace+=.175
    fig.subplots_adjust(hspace=_hspace,wspace=_wspace)

    vx_xy, vy_xy = mesh_3d_to_2d(vx,vy,vz,'xy')
    vx_xz, vz_xz = mesh_3d_to_2d(vx,vy,vz,'xz')
    vy_yz, vz_yz = mesh_3d_to_2d(vx,vy,vz,'yz')

    dv = vy_yz[1][1]-vy_yz[0][0] #assumes square velocity grid

    if(isIon):
        vnormstr = 'v_{ti}'
    else:
        vnormstr = 'v_{te}'

    fig.suptitle(ttl)

    # fig, axes = plt.subplots(nrows=2)
    # ax0label = axes[0].set_ylabel('Axes 0')
    # ax1label = axes[1].set_ylabel('Axes 1')
    #
    # title = axes[0].set_title('Title')
    #
    clboffset = np.array([vmax*.15,-vmax*.15])
    # title.set_position(ax0label.get_position() + offset)
    # title.set_rotation(90)
    try:
        minHxyval = np.min(H_xy[np.nonzero(H_xy)])
        minHxzval = np.min(H_xz[np.nonzero(H_xz)])
        minHyzval = np.min(H_yz[np.nonzero(H_yz)])
        maxH_xy = H_xy.max()
        maxH_xz = H_xz.max()
        maxH_yz = H_yz.max()
    except:
        minHxyval = 0.00000000001
        minHxzval = 0.00000000001
        minHyzval = 0.00000000001
        maxH_xy = 1.
        maxH_xz = 1.
        maxH_yz = 1.

    #H_xy
    if(plotLogHist):
        im00= axs[0,0].pcolormesh(vy_xy, vx_xy, H_xy, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHxyval, vmax=maxH_xy))
    else:
        im00= axs[0,0].pcolormesh(vy_xy, vx_xy, H_xy, cmap="plasma", shading="gouraud")
    #axs[0,0].set_title(r"$f(v_x, v_y)$")
    if(plotFAC):
        axs[0,0].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
    else:
        axs[0,0].set_ylabel(r"$v_y/"+vnormstr+"$")
    axs[0,0].grid()
    clrbar00 = plt.colorbar(im00, ax=axs[0,0])#,format='%.1e')
    if(not(plotLogHist)):
        clrbar00.formatter.set_powerlimits((0, 0))
    if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
        axs[0,0].text(-vmax*2.2,0, r"$f$", ha='center', rotation=90, wrap=False)
    else:
    	axs[0,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$f$", ha='center', rotation=90, wrap=False)
    
    if(params != None):
        if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None): 
            axs[0,0].text(-vmax*2.6,0, '$M_A = $ ' + str(abs(params['MachAlfven'])), ha='center', rotation=90, wrap=False)
        else:
            axs[0,0].text(-((np.abs(vxmax-vxmin)/2.))*2.6,0, '$M_A = $ ' + str(abs(params['MachAlfven'])), ha='center', rotation=90, wrap=False)
	
    if(listpos):
        if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
            axs[0,0].text(-vmax*2.6,0, '$x / d_i = $ ' + str("{:.4f}".format(xposval)), ha='center', rotation=90, wrap=False)
        else:
            axs[0,0].text(-((np.abs(vxmax-vxmin)/2.))*2.6,0, '$x / d_i = $ ' + str("{:.4f}".format(xposval)), ha='center', rotation=90, wrap=False)


    #H_xz
    if(plotLogHist):
        im01 = axs[0,1].pcolormesh(vz_xz, vx_xz, H_xz, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHxzval, vmax=maxH_xz))
    else:
        im01 = axs[0,1].pcolormesh(vz_xz, vx_xz, H_xz, cmap="plasma", shading="gouraud")
    #axs[0,1].set_title(r"$f(v_x, v_z)$")
    if(plotFAC):
        axs[0,1].set_ylabel(r"$v_{\perp,2}/"+vnormstr+"$")
    else:
        axs[0,1].set_ylabel(r"$v_z/"+vnormstr+"$")
    axs[0,1].grid()
    clrbar01 = plt.colorbar(im01, ax=axs[0,1])#,format='%.1e')
    if(not(plotLogHist)):
        clrbar01.formatter.set_powerlimits((0, 0))

    #H_yz
    if(plotLogHist):
        im02 = axs[0,2].pcolormesh(vz_yz, vy_yz, H_yz.T, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHyzval, vmax=maxH_yz))
    else:
        im02 = axs[0,2].pcolormesh(vz_yz, vy_yz, H_yz.T, cmap="plasma", shading="gouraud")
    #axs[0,2].set_title(r"$f(v_y, v_z)$")
    if(plotFAC):
        axs[0,2].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
    else:
        axs[0,2].set_ylabel(r"$v_y/"+vnormstr+"$")
    axs[0,2].grid()
    clrbar02 = plt.colorbar(im02, ax=axs[0,2])#,format='%.1e')
    if(not(plotLogHist)):
        clrbar02.formatter.set_powerlimits((0, 0))

    #CEx_xy
    maxCe = max(np.max(CEx_xy),abs(np.min(CEx_xy)))
    if(plotLog):
        im10 = axs[1,0].pcolormesh(vy_xy,vx_xy,CEx_xy,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im10 = axs[1,0].pcolormesh(vy_xy,vx_xy,CEx_xy,vmax=maxCe,vmin=-maxCe,cmap="seismic", shading="gouraud")
    #axs[1,0].set_title('$C_{Ex}(v_x,v_y)$')
    if(plotFAC):
        axs[1,0].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
    else:
        axs[1,0].set_ylabel(r"$v_y/"+vnormstr+"$")
    axs[1,0].grid()
    if(plotFAC):
        if(plotAvg):
            if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                axs[1,0].text(-vmax*2.2,0, r"$\overline{C_{E_{||}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[1,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\overline{C_{E_{||}}}$", ha='center', rotation=90, wrap=False)
        elif(plotFluc):
            if(isLowPass):
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[1,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{||}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[1,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{||}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
            
            elif(isHighPass):
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[1,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{||}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[1,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{||}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
            
            else:
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[1,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{||}}}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[1,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{||}}}$", ha='center', rotation=90, wrap=False)

        else:
            if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                axs[1,0].text(-vmax*2.2,0, r"$C_{E_{||}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[1,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$C_{E_{||}}$", ha='center', rotation=90, wrap=False)
    else:
        if(plotAvg):
            if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                axs[1,0].text(-vmax*2.2,0, r"$\overline{C_{E_{x}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[1,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\overline{C_{E_{x}}}$", ha='center', rotation=90, wrap=False)
        elif(plotFluc):
            if(isLowPass):
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[1,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{x}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[1,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{x}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
            elif(isHighPass):
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[1,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{x}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[1,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{x}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
            else:
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[1,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{x}}}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[1,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{x}}}$", ha='center', rotation=90, wrap=False)
        else:
            if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                axs[1,0].text(-vmax*2.2,0, r"$C_{E_{x}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[1,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$C_{E_{x}}$", ha='center', rotation=90, wrap=False)

    if(params != None):
        axs[1,0].text(-vmax*2.6,0, '$\Theta_{Bn} = $ ' + str(params['thetaBn']), ha='center', rotation=90, wrap=False)
    if(computeJdotE):
        if(not(plotDiagJEOnly)):
            JdotE = compute_energization(CEx_xy,dv)
            if(plotFAC):
                if(plotAvg):
                    axs[1,0].set_title('$\overline{j_{||}}  \overline{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[1,0].set_title('$\widetilde{j_{||}}^{k_{||} d_i < 15}  \widetilde{E_{||}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[1,0].set_title('$\widetilde{j_{||}}^{k_{||} d_i > 15}  \widetilde{E_{||}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[1,0].set_title('$\widetilde{j_{||}}  \widetilde{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[1,0].set_title('$j_{||}  E_{||}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[1,0].set_title('$\overline{j_x}  \overline{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[1,0].set_title('$\widetilde{j_x}^{k_{||} d_i > 15}  \widetilde{E_x}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[1,0].set_title('$\widetilde{j_x}^{k_{||} d_i > 15}  \widetilde{E_x}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[1,0].set_title('$\widetilde{j_x}  \widetilde{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[1,0].set_title('$j_x  E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
    clrbar10 = plt.colorbar(im10, ax=axs[1,0])#,format='%.1e')
    if(not(plotLog)):
        clrbar10.formatter.set_powerlimits((0, 0))

    #CEx_xz
    maxCe = max(np.max(CEx_xz),abs(np.min(CEx_xz)))
    if(plotLog):
        im11 = axs[1,1].pcolormesh(vz_xz,vx_xz,CEx_xz, cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im11 = axs[1,1].pcolormesh(vz_xz,vx_xz,CEx_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[1,1].set_title('$C_{Ex}(v_x,v_z)$')
    if(plotFAC):
        axs[1,1].set_ylabel(r"$v_{\perp,2}/"+vnormstr+"$")
    else:
        axs[1,1].set_ylabel(r"$v_z/"+vnormstr+"$")
    axs[1,1].grid()
    if(computeJdotE):
        if(not(plotDiagJEOnly)):
            JdotE = compute_energization(CEx_xz,dv)
            if(plotFAC):
                if(plotAvg):
                    axs[1,1].set_title('$\overline{j_{||}}  \overline{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[1,1].set_title('$\widetilde{j_{||}}^{k_{||} d_i < 15}  \widetilde{E_{||}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[1,1].set_title('$\widetilde{j_{||}}^{k_{||} d_i > 15}  \widetilde{E_{||}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[1,1].set_title('$\widetilde{j_{||}}  \widetilde{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[1,1].set_title('$j_{||}  E_{||}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[1,1].set_title('$\overline{j_x}  \overline{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[1,1].set_title('$\widetilde{j_x}^{k_{||} d_i > 15}  \widetilde{E_x}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[1,1].set_title('$\widetilde{j_x}^{k_{||} d_i > 15}  \widetilde{E_x}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[1,1].set_title('$\widetilde{j_x}  \widetilde{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[1,1].set_title('$j_x  E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
    clrbar11 = plt.colorbar(im11, ax=axs[1,1])#,format='%.1e')
    if(not(plotLog)):
        clrbar11.formatter.set_powerlimits((0, 0))

    #CEx_yz
    maxCe = max(np.max(CEx_yz),abs(np.min(CEx_yz)))
    if(plotLog):
        im12 = axs[1,2].pcolormesh(vz_yz,vy_yz,CEx_yz.T,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im12 = axs[1,2].pcolormesh(vz_yz,vy_yz,CEx_yz.T,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[1,2].set_title('$C_{Ex}(v_y,v_z)$')
    if(plotFAC):
        axs[1,2].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
    else:
        axs[1,2].set_ylabel(r"$v_y/"+vnormstr+"$")
    axs[1,2].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEx_yz.T,dv)
        if(True):
            if(plotFAC):
                if(plotAvg):
                    axs[1,2].set_title('$\overline{j_{||}}  \overline{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[1,2].set_title('$\widetilde{j_{||}}^{k_{||} d_i < 15}  \widetilde{E_{||}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[1,2].set_title('$\widetilde{j_{||}}^{k_{||} d_i > 15}  \widetilde{E_{||}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[1,2].set_title('$\widetilde{j_{||}}  \widetilde{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[1,2].set_title('$j_{||}  E_{||}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[1,2].set_title('$\overline{j_x}  \overline{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[1,2].set_title('$\widetilde{j_x}^{k_{||} d_i > 15}  \widetilde{E_x}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[1,2].set_title('$\widetilde{j_x}^{k_{||} d_i > 15}  \widetilde{E_x}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[1,2].set_title('$\widetilde{j_x}  \widetilde{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[1,2].set_title('$j_x  E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
    clrbar12 = plt.colorbar(im12, ax=axs[1,2])#,format='%.1e')
    if(not(plotLog)):
        clrbar12.formatter.set_powerlimits((0, 0))

    #CEy_xy
    maxCe = max(np.max(CEy_xy),abs(np.min(CEy_xy)))
    if(plotLog):
        im20 = axs[2,0].pcolormesh(vy_xy,vx_xy,CEy_xy,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im20 = axs[2,0].pcolormesh(vy_xy,vx_xy,CEy_xy,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[2,0].set_title('$C_{Ey}(v_x,v_y)$')
    if(plotFAC):
        axs[2,0].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
    else:
        axs[2,0].set_ylabel(r"$v_y/"+vnormstr+"$")
    if(plotFAC):
        if(plotAvg):          
            if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                axs[2,0].text(-vmax*2.2,0, r"$\overline{C_{E_{\perp,1}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[2,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\overline{C_{E_{\perp,1}}}$", ha='center', rotation=90, wrap=False)
        elif(plotFluc):
            if(isLowPass):
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[2,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,1}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[2,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{\perp,1}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
            elif(isHighPass):
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[2,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,1}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[2,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{\perp,1}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
            else:
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[2,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,1}}}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[2,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{\perp,1}}}$", ha='center', rotation=90, wrap=False)
        else:
            if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                axs[2,0].text(-vmax*2.2,0, r"$C_{E_{\perp,1}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[2,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$C_{E_{\perp,1}}$", ha='center', rotation=90, wrap=False)
    else:
        if(plotAvg):
            if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                axs[2,0].text(-vmax*2.2,0, r"$\overline{C_{E_{y}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[2,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\overline{C_{E_{y}}}$", ha='center', rotation=90, wrap=False)
        elif(plotFluc):
            if(isLowPass):
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[2,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{y}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[2,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{y}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
            elif(isHighPass):
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[2,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{y}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[2,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{y}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
            else:
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[2,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{y}}}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[2,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{y}}}$", ha='center', rotation=90, wrap=False)
        else:
            if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                axs[2,0].text(-vmax*2.2,0, r"$C_{E_{y}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[2,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$C_{E_{y}}$", ha='center', rotation=90, wrap=False)
    axs[2,0].grid()
    if(xpos != None):
        axs[2,0].text(-vmax*2.6,0,'$x/d_i = $' + str(xpos), ha='center', rotation=90, wrap=False)
    if(computeJdotE):
        JdotE = compute_energization(CEy_xy,dv)
        if(not(plotDiagJEOnly)):
            JdotE = compute_energization(CEx_xz,dv)
            if(plotFAC):
                if(plotAvg):
                    axs[2,0].set_title('$\overline{j_{\perp,1}}  \overline{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[2,0].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i < 15}  \widetilde{E_{\perp,1}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[2,0].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i > 15}  \widetilde{E_{\perp,1}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[2,0].set_title('$\widetilde{j_{\perp,1}}  \widetilde{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left') 
                else:
                    axs[2,0].set_title('$j_{\perp,1}  E_{\perp,1}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[2,0].set_title('$\overline{j_y}  \overline{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')             
                elif(plotFluc):
                    if(isLowPass):
                        axs[2,0].set_title('$\widetilde{j_y}^{k_{||} d_i < 15}  \widetilde{E_y}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[2,0].set_title('$\widetilde{j_y}^{k_{||} d_i > 15}  \widetilde{E_y}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[2,0].set_title('$\widetilde{j_y}  \widetilde{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')             
                else:
                    axs[2,0].set_title('$j_y  E_y$ = ' + "{:.2e}".format(JdotE),loc='left')
    
    clrbar20 = plt.colorbar(im20, ax=axs[2,0])#,format='%.1e')
    if(not(plotLog)):
        clrbar20.formatter.set_powerlimits((0, 0))

    #CEy_xz
    maxCe = max(np.max(CEy_xz),abs(np.min(CEy_xz)))
    if(plotLog):
        im21 = axs[2,1].pcolormesh(vz_xz,vx_xz,CEy_xz,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im21 = axs[2,1].pcolormesh(vz_xz,vx_xz,CEy_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[2,1].set_title('$C_{Ey}(v_x,v_z)$')
    if(plotFAC):
        axs[2,1].set_ylabel(r"$v_{\perp,2}/"+vnormstr+"$")
    else:
        axs[2,1].set_ylabel(r"$v_z/"+vnormstr+"$")
    axs[2,1].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEy_xz,dv)
        if(True):
            JdotE = compute_energization(CEy_xz,dv)
            if(plotFAC):
                if(plotAvg):
                    axs[2,1].set_title('$\overline{j_{\perp,1}}  \overline{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[2,1].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i < 15}  \widetilde{E_{\perp,1}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[2,1].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i > 15}  \widetilde{E_{\perp,1}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[2,1].set_title('$\widetilde{j_{\perp,1}}  \widetilde{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[2,1].set_title('$j_{\perp,1}  E_{\perp,1}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[2,1].set_title('$\overline{j_y}  \overline{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')  
                elif(plotFluc):
                    if(isLowPass):
                        axs[2,1].set_title('$\widetilde{j_y}^{k_{||} d_i < 15}  \widetilde{E_y}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[2,1].set_title('$\widetilde{j_y}^{k_{||} d_i > 15}  \widetilde{E_y}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[2,1].set_title('$\widetilde{j_y}  \widetilde{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')             
                else:
                    axs[2,1].set_title('$j_y  E_y$ = ' + "{:.2e}".format(JdotE),loc='left')

    clrbar21 = plt.colorbar(im21, ax=axs[2,1])#,format='%.1e')
    if(not(plotLog)):
        clrbar21.formatter.set_powerlimits((0, 0))

    #CEy_yz
    maxCe = max(np.max(CEy_yz),abs(np.min(CEy_yz)))
    if(plotLog):
        im22 = axs[2,2].pcolormesh(vz_yz,vy_yz,CEy_yz.T, cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im22 = axs[2,2].pcolormesh(vz_yz,vy_yz,CEy_yz.T,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[2,2].set_title('$C_{Ey}(v_y,v_z)$')
    if(plotFAC):
        axs[2,2].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
    else:
        axs[2,2].set_ylabel(r"$v_y/"+vnormstr+"$")
    axs[2,2].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEy_yz.T,dv)
        if(not(plotDiagJEOnly)):
            if(plotFAC):
                if(plotAvg):
                    axs[2,2].set_title('$\overline{j_{\perp,1}}  \overline{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[2,2].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i < 15}  \widetilde{E_{\perp,1}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[2,2].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i > 15}  \widetilde{E_{\perp,1}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[2,2].set_title('$\widetilde{j_{\perp,1}}  \widetilde{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[2,2].set_title('$j_{\perp,1}  E_{\perp,1}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[2,2].set_title('$\overline{j_y}  \overline{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[2,2].set_title('$\widetilde{j_y}^{k_{||} d_i < 15}  \widetilde{E_y}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[2,2].set_title('$\widetilde{j_y}^{k_{||} d_i > 15}  \widetilde{E_y}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[2,2].set_title('$\widetilde{j_y}  \widetilde{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[2,2].set_title('$j_y  E_y$ = ' + "{:.2e}".format(JdotE),loc='left')

    clrbar22 = plt.colorbar(im22, ax=axs[2,2])#,format='%.1e')
    if(not(plotLog)):
        clrbar22.formatter.set_powerlimits((0, 0))

    #CEz_xy
    maxCe = max(np.max(CEz_xy),abs(np.min(CEz_xy)))
    if(plotLog):
        im30 = axs[3,0].pcolormesh(vy_xy,vx_xy,CEz_xy,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im30 = axs[3,0].pcolormesh(vy_xy,vx_xy,CEz_xy,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[3,0].set_title('$C_{Ez}(v_x,v_y)$')
    if(plotFAC):
        axs[3,0].set_xlabel(r"$v_{||}/"+vnormstr+"$")
        axs[3,0].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
    else:
        axs[3,0].set_xlabel(r"$v_x/"+vnormstr+"$")
        axs[3,0].set_ylabel(r"$v_y/"+vnormstr+"$")
    if(plotFAC):
        if(plotAvg):
            if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                axs[3,0].text(-vmax*2.2,0, r"$\overline{C_{E_{\perp,2}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[3,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\overline{C_{E_{\perp,2}}}$", ha='center', rotation=90, wrap=False)
        elif(plotFluc):
            if(isLowPass):
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[3,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,2}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[3,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{\perp,2}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
            elif(isHighPass):
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[3,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,2}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[3,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{\perp,2}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
            else:
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[3,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,2}}}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[3,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{\perp,2}}}$", ha='center', rotation=90, wrap=False)
        else:
            if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                axs[3,0].text(-vmax*2.2,0, r"$C_{E_{\perp,2}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[3,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$C_{E_{\perp,2}}$", ha='center', rotation=90, wrap=False)
    else:
        if(plotAvg):
            if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                axs[3,0].text(-vmax*2.2,0, r"$\overline{C_{E_{z}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[3,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\overline{C_{E_{z}}}$", ha='center', rotation=90, wrap=False)
        elif(plotFluc):
            if(isLowPass):
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[3,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{z}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[3,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{z}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
            elif(isHighPass):
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[3,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{z}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[3,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{z}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
            else:
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[3,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{z}}}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[3,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$\widetilde{C_{E_{z}}}$", ha='center', rotation=90, wrap=False)
        else:
                if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                    axs[3,0].text(-vmax*2.2,0, r"$C_{E_{z}}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[3,0].text(-np.abs((vxmax+vxmin)/2.-(vxmin))*2.2+(vxmax+vxmin)/2.,(vymax+vymin)/2.-.05*np.abs(vymax-vymin), r"$C_{E_{z}}$", ha='center', rotation=90, wrap=False)
    axs[3,0].grid()
    if(metadata != None):
        axs[3,0].text(-vmax*2.6,0, metadata, ha='center', rotation=90, wrap=False)
    if(computeJdotE):
        JdotE = compute_energization(CEz_xy,dv)
        if(True):
            if(plotFAC):
                if(plotAvg):
                    axs[3,0].set_title('$\overline{j_{\perp,2}}  \overline{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[3,0].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i < 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[3,0].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i > 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[3,0].set_title('$\widetilde{j_{\perp,2}}  \widetilde{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[3,0].set_title('$j_{\perp,2}  E_{\perp,2}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[3,0].set_title('$\overline{j_z}  \overline{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[3,0].set_title('$\widetilde{j_z}^{k_{||} d_i < 15}  \widetilde{E_z}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[3,0].set_title('$\widetilde{j_z}^{k_{||} d_i > 15}  \widetilde{E_z}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[3,0].set_title('$\widetilde{j_z}  \widetilde{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[3,0].set_title('$j_z  E_z$ = ' + "{:.2e}".format(JdotE),loc='left')

    clrbar30 = plt.colorbar(im30, ax=axs[3,0])#,format='%.1e')
    if(not(plotLog)):
        clrbar30.formatter.set_powerlimits((0, 0))

    #CEz_xz
    maxCe = max(np.max(CEz_xz),abs(np.min(CEz_xz)))
    if(plotLog):
        im31 = axs[3,1].pcolormesh(vz_xz,vx_xz,CEz_xz,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im31 = axs[3,1].pcolormesh(vz_xz,vx_xz,CEz_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[3,1].set_title('$C_{Ez}(v_x,v_z)$')
    if(plotFAC):
        axs[3,1].set_xlabel(r"$v_{||}/"+vnormstr+"$")
        axs[3,1].set_ylabel(r"$v_{\perp,2}/"+vnormstr+"$")
    else:
        axs[3,1].set_xlabel(r"$v_x/"+vnormstr+"$")
        axs[3,1].set_ylabel(r"$v_z/"+vnormstr+"$")
    axs[3,1].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEz_xz,dv)
        if(not(plotDiagJEOnly)):
            if(plotFAC):
                if(plotAvg):
                    axs[3,1].set_title('$\overline{j_{\perp,2}}  \overline{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[3,1].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i < 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[3,1].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i > 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[3,1].set_title('$\widetilde{j_{\perp,2}}  \widetilde{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[3,1].set_title('$j_{\perp,2}  E_{\perp,2}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[3,1].set_title('$\overline{j_z}  \overline{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[3,1].set_title('$\widetilde{j_z}^{k_{||} d_i < 15}  \widetilde{E_z}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[3,1].set_title('$\widetilde{j_z}^{k_{||} d_i > 15}  \widetilde{E_z}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[3,1].set_title('$\widetilde{j_z}  \widetilde{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[3,1].set_title('$j_z  E_z$ = ' + "{:.2e}".format(JdotE),loc='left')

    clrbar31 = plt.colorbar(im31, ax=axs[3,1])#,format='%.1e')
    if(not(plotLog)):
        clrbar31.formatter.set_powerlimits((0, 0))

    #CEz_yz
    maxCe = max(np.max(CEz_yz),abs(np.min(CEz_yz)))
    if(plotLog):
        im32 = axs[3,2].pcolormesh(vz_yz,vy_yz,CEz_yz.T,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im32 = axs[3,2].pcolormesh(vz_yz,vy_yz,CEz_yz.T,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[3,2].set_title('$C_{Ez}(v_y,v_z)$')
    if(plotFAC):
        axs[3,2].set_xlabel(r"$v_{\perp,2}/"+vnormstr+"$")
        axs[3,2].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
    else:
        axs[3,2].set_xlabel(r"$v_z/"+vnormstr+"$")
        axs[3,2].set_ylabel(r"$v_y/"+vnormstr+"$")
    axs[3,2].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEz_yz.T,dv)
        if(not(plotDiagJEOnly)):
            if(plotFAC):
                if(plotAvg):
                    axs[3,2].set_title('$\overline{j_{\perp,2}}  \overline{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[3,2].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i < 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[3,2].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i > 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[3,2].set_title('$\widetilde{j_{\perp,2}}  \widetilde{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[3,2].set_title('$j_{\perp,2}  E_{\perp,2}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[3,2].set_title('$\overline{j_z}  \overline{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[3,2].set_title('$\widetilde{j_z}^{k_{||} d_i < 15}  \widetilde{E_z}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[3,2].set_title('$\widetilde{j_z}^{k_{||} d_i > 15}  \widetilde{E_z}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[3,2].set_title('$\widetilde{j_z}  \widetilde{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[3,2].set_title('$j_z  E_z$ = ' + "{:.2e}".format(JdotE),loc='left')

    clrbar32 = plt.colorbar(im32, ax=axs[3,2])#,format='%.1e')
    if(not(plotLog)):
        clrbar32.formatter.set_powerlimits((0, 0))

    for _i in range(0,4):
        for _j in range(0,3):
            if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                axs[_i,_j].set_aspect('equal', 'box')
            else:
                axs[_i,_j].set_box_aspect(1) 
                axs[_i,_j].set_aspect('auto')

    for _i in range(0,4):
        for _j in range(0,3):
            if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
                axs[_i,_j].set_xlim(-vmax,vmax)
                axs[_i,_j].set_ylim(-vmax,vmax)
            else:
                if(_j == 0):
                    axs[_i,_j].set_xlim(vxmin,vxmax)
                    axs[_i,_j].set_ylim(vymin,vymax)
                elif(_j == 1):
                    axs[_i,_j].set_xlim(vxmin,vxmax)
                    axs[_i,_j].set_ylim(vzmin,vzmax)
                elif(_j == 2):
                    axs[_i,_j].set_xlim(vzmin,vzmax)
                    axs[_i,_j].set_ylim(vymin,vymax)

    #set ticks
    intvl = 1.
    if(vmax > 5):
        intvl = 5.
    if(vmax > 10):
        intvl = 10.
    if(vmax > 20):
        intvl = 20.

    if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None):
        tcks = np.arange(0,vmax,intvl)
        tcks = np.concatenate((-1*np.flip(tcks),tcks))
        for _i in range(0,4):
            for _j in range(0,3):
                axs[_i,_j].set_xticks(tcks)
                axs[_i,_j].set_yticks(tcks)
    else:
        intvlxx = 1.
        if(np.abs(vxmax-vxmin)>5):intvlxx=5.
        elif(np.abs(vxmax-vxmin)>10):intvlxx=10.
        elif(np.abs(vxmax-vxmin)>20):intvlxx=15.
        intvlyy = 1.
        if(np.abs(vymax-vymin)>5):intvlyy=5.
        elif(np.abs(vymax-vymin)>10):intvlyy=10.
        elif(np.abs(vymax-vymin)>20):intvlxx=15.
        intvlzz = 1.
        if(np.abs(vzmax-vzmin)>5):intvlzz=5.
        elif(np.abs(vzmax-vzmin)>10):intvlzz=10.
        elif(np.abs(vzmax-vzmin)>20):intvlxx=15.

        tcksxx = np.arange(vxmin,vxmax+intvlxx,intvlxx)
        tcksyy = np.arange(vymin,vymax+intvlyy,intvlyy)
        tckszz = np.arange(vzmin,vzmax+intvlzz,intvlzz)

        for _i in range(0,4):
             for _j in range(0,3):
                 if(_j == 0):
                     axs[_i,_j].set_xticks(tcksxx)
                     axs[_i,_j].set_yticks(tcksyy)
                 if(_j == 1):
                     axs[_i,_j].set_xticks(tcksxx)
                     axs[_i,_j].set_yticks(tckszz)
                 if(_j == 2):
                     axs[_i,_j].set_xticks(tckszz)
                     axs[_i,_j].set_yticks(tcksyy)
    
    #plt.subplots_adjust(hspace=.5,wspace=-.3)


    if(vxmin == None and vxmax == None and vymin == None and vymax == None and vzmin == None and vzmax == None): 
        maxplotvvalxx = np.max(vz_yz) #Assumes square grid of even size
        maxplotvvalyy = np.max(vz_yz) #Assumes square grid of even size
        maxplotvvalzz = np.max(vz_yz) #Assumes square grid of even size
        centerplotxx = 0.
        centerplotyy = 0.
        centerplotzz = 0.
    else:
        #TODO: rename this variable
        maxplotvvalxx = vxmax-((vxmax+vxmin)/2.)
        maxplotvvalyy = vymax-((vymax+vymin)/2.)
        maxplotvvalzz = vzmax-((vzmax+vzmin)/2.)
        centerplotxx = (vxmax+vxmin)/2.
        centerplotyy = (vymax+vymin)/2.
        centerplotzz = (vzmax+vzmin)/2.
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight')
        #must make figure first to grab x10^val on top of color bar- after grabbing it, we can move it- a little wasteful but it was quick solution
        clrbar10text = str(clrbar10.ax.yaxis.get_offset_text().get_text())
        clrbar10.ax.yaxis.get_offset_text().set_visible(False)
        axs[1,0].text(1.63*maxplotvvalxx+centerplotxx,-1.27*maxplotvvalyy+centerplotyy,clrbar10text, va='bottom', ha='center')
        clrbar11text = str(clrbar11.ax.yaxis.get_offset_text().get_text())
        clrbar11.ax.yaxis.get_offset_text().set_visible(False) 
        axs[1,1].text(1.63*maxplotvvalxx+centerplotxx,-1.27*maxplotvvalzz+centerplotzz,clrbar11text, va='bottom', ha='center')
        clrbar12text = str(clrbar12.ax.yaxis.get_offset_text().get_text())
        clrbar12.ax.yaxis.get_offset_text().set_visible(False)
        axs[1,2].text(1.63*maxplotvvalzz+centerplotzz,-1.27*maxplotvvalyy+centerplotyy,clrbar12text, va='bottom', ha='center')
        clrbar20text = str(clrbar20.ax.yaxis.get_offset_text().get_text())
        clrbar20.ax.yaxis.get_offset_text().set_visible(False)
        axs[2,0].text(1.63*maxplotvvalxx+centerplotxx,-1.27*maxplotvvalyy+centerplotyy,clrbar20text, va='bottom', ha='center')
        clrbar21text = str(clrbar21.ax.yaxis.get_offset_text().get_text())
        clrbar21.ax.yaxis.get_offset_text().set_visible(False)
        axs[2,1].text(1.63*maxplotvvalxx+centerplotxx,-1.27*maxplotvvalzz+centerplotzz,clrbar21text, va='bottom', ha='center')
        clrbar22text = str(clrbar22.ax.yaxis.get_offset_text().get_text())
        clrbar22.ax.yaxis.get_offset_text().set_visible(False)
        axs[2,2].text(1.63*maxplotvvalzz+centerplotzz,-1.27*maxplotvvalyy+centerplotyy,clrbar22text, va='bottom', ha='center')
        clrbar30text = str(clrbar30.ax.yaxis.get_offset_text().get_text())
        clrbar30.ax.yaxis.get_offset_text().set_visible(False)
        axs[3,0].text(1.63*maxplotvvalxx+centerplotxx,-1.27*maxplotvvalyy+centerplotyy,clrbar30text, va='bottom', ha='center')
        clrbar31text = str(clrbar31.ax.yaxis.get_offset_text().get_text())
        clrbar31.ax.yaxis.get_offset_text().set_visible(False)
        axs[3,1].text(1.63*maxplotvvalxx+centerplotxx,-1.27*maxplotvvalzz+centerplotzz,clrbar31text, va='bottom', ha='center')
        clrbar32text = str(clrbar32.ax.yaxis.get_offset_text().get_text())
        clrbar32.ax.yaxis.get_offset_text().set_visible(False)
        axs[3,2].text(1.63*maxplotvvalzz+centerplotzz,-1.27*maxplotvvalyy+centerplotyy,clrbar32text, va='bottom', ha='center')

        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight')


        plt.close('all') #saves RAM
    else:
        import os

        #for some reason, we have to generate the plot to move the colorpower power elsewhere for some unknown reason
        plt.savefig('_tempdelete.png',format='png',dpi=250,bbox_inches='tight')
        os.remove('_tempdelete.png')

        #must make figure first to grab x10^val on top of color bar- after grabbing it, we can move it- a little wasteful but it was quick solution
        clrbar10text = str(clrbar10.ax.yaxis.get_offset_text().get_text())
        clrbar10.ax.yaxis.get_offset_text().set_visible(False)
        axs[1,0].text(1.63*maxplotvvalxx,-1.27*maxplotvvalyy,clrbar10text, va='bottom', ha='center')
        clrbar11text = str(clrbar11.ax.yaxis.get_offset_text().get_text())
        clrbar11.ax.yaxis.get_offset_text().set_visible(False) 
        axs[1,1].text(1.63*maxplotvvalxx,-1.27*maxplotvvalzz,clrbar11text, va='bottom', ha='center')
        clrbar12text = str(clrbar12.ax.yaxis.get_offset_text().get_text())
        clrbar12.ax.yaxis.get_offset_text().set_visible(False)
        axs[1,2].text(1.63*maxplotvvalzz,-1.27*maxplotvvalyy,clrbar12text, va='bottom', ha='center')
        clrbar20text = str(clrbar20.ax.yaxis.get_offset_text().get_text())
        clrbar20.ax.yaxis.get_offset_text().set_visible(False)
        axs[2,0].text(1.63*maxplotvvalxx,-1.27*maxplotvvalyy,clrbar20text, va='bottom', ha='center')
        clrbar21text = str(clrbar21.ax.yaxis.get_offset_text().get_text())
        clrbar21.ax.yaxis.get_offset_text().set_visible(False)
        axs[2,1].text(1.63*maxplotvvalxx,-1.27*maxplotvvalzz,clrbar21text, va='bottom', ha='center')
        clrbar22text = str(clrbar22.ax.yaxis.get_offset_text().get_text())
        clrbar22.ax.yaxis.get_offset_text().set_visible(False)
        axs[2,2].text(1.63*maxplotvvalzz,-1.27*maxplotvvalyy,clrbar22text, va='bottom', ha='center')
        clrbar30text = str(clrbar30.ax.yaxis.get_offset_text().get_text())
        clrbar30.ax.yaxis.get_offset_text().set_visible(False)
        axs[3,0].text(1.63*maxplotvvalxx,-1.27*maxplotvvalyy,clrbar30text, va='bottom', ha='center')
        clrbar31text = str(clrbar31.ax.yaxis.get_offset_text().get_text())
        clrbar31.ax.yaxis.get_offset_text().set_visible(False)
        axs[3,1].text(1.63*maxplotvvalxx,-1.27*maxplotvvalzz,clrbar31text, va='bottom', ha='center')
        clrbar32text = str(clrbar32.ax.yaxis.get_offset_text().get_text())
        clrbar32.ax.yaxis.get_offset_text().set_visible(False)
        axs[3,2].text(1.63*maxplotvvalzz,-1.27*maxplotvvalyy,clrbar32text, va='bottom', ha='center')

        plt.show()
    
    plt.close()

def plot_cor_and_dist_supergrid_row(vx, vy, vz, vmax,
                                arr_xy,arr_xz, arr_yz,
                                arrtype,
                                flnm = '', ttl = '', computeJdotE = True, params = None, metadata = None, xpos = None, plotLog = False, plotLogHist = True,
                                plotFAC = False, plotFluc = False, plotAvg = False, isIon = True, listpos=False,xposval=None,normtoN = False,Nval = None, isLowPass=False,isHighPass=False,plotDiagJEOnly=True):
    """
    Parameters
    ----------
    arrtype : string
        type of array to be plotted ('CEx','CEy','CEz','Ctot','CEperp1','CEperp2','CEperp2','H','H_fac'); no need to specify here if avg or fluc

    Otherwise, same as plot_cor_and_dist_supergrid, except it plots just one row
    """
    from matplotlib.colors import LogNorm
    from FPCAnalysis.array_ops import mesh_3d_to_2d
    from FPCAnalysis.analysis import compute_energization
    import matplotlib.colors as colors

    if(not(arrtype in ['CEx','CEy','CEz','Ctot','CEperp1','CEperp2','CEperp2','H'])):
        print("Error! arrtype must be one of ('CEx','CEy','CEz','Ctot','CEperp1','CEperp2','CEperp2','H','H_fac').")
        return

    if(normtoN):
        if(Nval == None):
            Nval = np.sum(H_xy)
        arr_xy/=Nval 
        arr_xz/=Nval
        arr_yz/=Nval

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    fig, axs = plt.subplots(1,3,figsize=(1*15,3*15))
    axs = np.reshape(axs, (1, 3)) #Reshape so we can more easily reuse old code

    cbarshrink = .075 #controls size of colorbar

    plt.rcParams['axes.titlepad'] = 8  # pad is in points

    _hspace = 0
    _wspace = .4
    if(computeJdotE):
        _hspace+=0
    if(plotLog):
        _wspace+=.175
    fig.subplots_adjust(hspace=_hspace,wspace=_wspace)

    vx_xy, vy_xy = mesh_3d_to_2d(vx,vy,vz,'xy')
    vx_xz, vz_xz = mesh_3d_to_2d(vx,vy,vz,'xz')
    vy_yz, vz_yz = mesh_3d_to_2d(vx,vy,vz,'yz')

    maxplotvval = np.max(vz_yz) #Assumes square grid of even size

    dv = vy_yz[1][1]-vy_yz[0][0] #assumes square velocity grid

    if(isIon):
        vnormstr = 'v_{ti}'
    else:
        vnormstr = 'v_{te}'

    fig.suptitle(ttl)

    if(arrtype in ['H']):
        try:
            minHxyval = np.min(arr_xy[np.nonzero(arr_xy)])
            minHxzval = np.min(arr_xz[np.nonzero(arr_xz)])
            minHyzval = np.min(arr_yz[np.nonzero(arr_yz)])
            maxH_xy = arr_xy.max()
            maxH_xz = arr_xz.max()
            maxH_yz = arr_yz.max()
        except:
            minHxyval = 0.00000000001
            minHxzval = 0.00000000001
            minHyzval = 0.00000000001
            maxH_xy = 1.
            maxH_xz = 1.
            maxH_yz = 1.

    if(arrtype == 'H'):
        H_xy = arr_xy
        H_xz = arr_xz
        H_yz = arr_yz
        #H_xy
        if(plotLogHist):
            im00= axs[0,0].pcolormesh(vy_xy, vx_xy, H_xy, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHxyval, vmax=maxH_xy))
        else:
            im00= axs[0,0].pcolormesh(vy_xy, vx_xy, H_xy, cmap="plasma", shading="gouraud")

        if(plotFAC):
            axs[0,0].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
        else:
            axs[0,0].set_ylabel(r"$v_y/"+vnormstr+"$")
        axs[0,0].set_aspect('equal', 'box')
        axs[0,0].grid()
        clrbar10 = plt.colorbar(im00, ax=axs[0,0],shrink = cbarshrink)#,format='%.1e')
        if(not(plotLogHist)):
            clrbar10.formatter.set_powerlimits((0, 0))
        axs[0,0].text(-vmax*2.2,0, r"$f$", ha='center', rotation=90, wrap=False)
        if(params != None):
            axs[0,0].text(-vmax*2.9,0, '$M_A = $ ' + str(abs(params['MachAlfven'])), ha='center', rotation=90, wrap=False)

        if(listpos):
            axs[0,0].text(-vmax*2.9,0, '$x / d_i = $ ' + str("{:.4f}".format(xposval)), ha='center', rotation=90, wrap=False)


        #H_xz
        if(plotLogHist):
            im01 = axs[0,1].pcolormesh(vz_xz, vx_xz, H_xz, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHxzval, vmax=maxH_xz))
        else:
            im01 = axs[0,1].pcolormesh(vz_xz, vx_xz, H_xz, cmap="plasma", shading="gouraud")
        #axs[0,1].set_title(r"$f(v_x, v_z)$")
        if(plotFAC):
            axs[0,1].set_ylabel(r"$v_{\perp,2}/"+vnormstr+"$")
        else:
            axs[0,1].set_ylabel(r"$v_z/"+vnormstr+"$")
        axs[0,1].set_aspect('equal', 'box')
        axs[0,1].grid()
        clrbar11 = plt.colorbar(im01, ax=axs[0,1],shrink = cbarshrink)#,format='%.1e')
        if(not(plotLogHist)):
            clrbar11.formatter.set_powerlimits((0, 0))

        #H_yz
        if(plotLogHist):
            im02 = axs[0,2].pcolormesh(vz_yz, vy_yz, H_yz.T, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHyzval, vmax=maxH_yz))
        else:
            im02 = axs[0,2].pcolormesh(vz_yz, vy_yz, H_yz.T, cmap="plasma", shading="gouraud")
        #axs[0,2].set_title(r"$f(v_y, v_z)$")
        if(plotFAC):
            axs[0,2].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
        else:
            axs[0,2].set_ylabel(r"$v_y/"+vnormstr+"$")
        axs[0,2].set_aspect('equal', 'box')
        axs[0,2].grid()
        clrbar12 = plt.colorbar(im02, ax=axs[0,2], shrink = cbarshrink)#,format='%.1e')
        if(not(plotLogHist)):
            clrbar12.formatter.set_powerlimits((0, 0))

    if(arrtype in ['CEx','CEpar']):
        CEx_xy = arr_xy
        CEx_xz = arr_xz
        CEx_yz = arr_yz
        maxCe = max(np.max(CEx_xy),abs(np.min(CEx_xy)))
        if(plotLog):
            im10 = axs[0,0].pcolormesh(vy_xy,vx_xy,CEx_xy,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
        else:
            im10 = axs[0,0].pcolormesh(vy_xy,vx_xy,CEx_xy,vmax=maxCe,vmin=-maxCe,cmap="seismic", shading="gouraud")
        if(plotFAC):
            axs[0,0].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
        else:
            axs[0,0].set_ylabel(r"$v_y/"+vnormstr+"$")
        axs[0,0].set_aspect('equal', 'box')
        axs[0,0].grid()
        if(plotFAC):
            if(plotAvg):
                axs[0,0].text(-vmax*2.2,0, r"$\overline{C_{E_{||}}}$", ha='center', rotation=90, wrap=False)
            elif(plotFluc):
                if(isLowPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{||}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                elif(isHighPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{||}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{||}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[0,0].text(-vmax*2.2,0, r"$C_{E_{||}}$", ha='center', rotation=90, wrap=False)
        else:
            if(plotAvg):
                axs[0,0].text(-vmax*2.2,0, r"$\overline{C_{E_{x}}}$", ha='center', rotation=90, wrap=False)
            elif(plotFluc):
                if(isLowPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{x}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                elif(isHighPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{x}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{x}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[0,0].text(-vmax*2.2,0, r"$C_{E_{x}}$", ha='center', rotation=90, wrap=False)
        if(params != None):
            axs[0,0].text(-vmax*2.9,0, '$\Theta_{Bn} = $ ' + str(params['thetaBn']), ha='center', rotation=90, wrap=False)
        if(computeJdotE):
            if(not(plotDiagJEOnly)):
                JdotE = compute_energization(CEx_xy,dv)
                if(plotFAC):
                    if(plotAvg):
                        axs[0,0].set_title('$\overline{j_{||}}  \overline{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,0].set_title('$\widetilde{j_{||}}^{k_{||} d_i < 15}  \widetilde{E_{||}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,0].set_title('$\widetilde{j_{||}}^{k_{||} d_i > 15}  \widetilde{E_{||}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,0].set_title('$\widetilde{j_{||}}  \widetilde{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,0].set_title('$j_{||}  E_{||}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    if(plotAvg):
                        axs[0,0].set_title('$\overline{j_x}  \overline{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,0].set_title('$\widetilde{j_x}^{k_{||} d_i > 15}  \widetilde{E_x}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,0].set_title('$\widetilde{j_x}^{k_{||} d_i > 15}  \widetilde{E_x}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,0].set_title('$\widetilde{j_x}  \widetilde{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,0].set_title('$j_x  E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
        clrbar10 = plt.colorbar(im10, ax=axs[1,0], shrink = cbarshrink)#,format='%.1e')
        if(not(plotLog)):
            clrbar10.formatter.set_powerlimits((0, 0))

        #CEx_xz
        maxCe = max(np.max(CEx_xz),abs(np.min(CEx_xz)))
        if(plotLog):
            im11 = axs[0,1].pcolormesh(vz_xz,vx_xz,CEx_xz, cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
        else:
            im11 = axs[0,1].pcolormesh(vz_xz,vx_xz,CEx_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")

        if(plotFAC):
            axs[0,1].set_ylabel(r"$v_{\perp,2}/"+vnormstr+"$")
        else:
            axs[0,1].set_ylabel(r"$v_z/"+vnormstr+"$")
        axs[0,1].set_aspect('equal', 'box')
        axs[0,1].grid()
        if(computeJdotE):
            if(not(plotDiagJEOnly)):
                JdotE = compute_energization(CEx_xz,dv)
                if(plotFAC):
                    if(plotAvg):
                        axs[0,1].set_title('$\overline{j_{||}}  \overline{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,1].set_title('$\widetilde{j_{||}}^{k_{||} d_i < 15}  \widetilde{E_{||}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,1].set_title('$\widetilde{j_{||}}^{k_{||} d_i > 15}  \widetilde{E_{||}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,1].set_title('$\widetilde{j_{||}}  \widetilde{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,1].set_title('$j_{||}  E_{||}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    if(plotAvg):
                        axs[0,1].set_title('$\overline{j_x}  \overline{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,1].set_title('$\widetilde{j_x}^{k_{||} d_i > 15}  \widetilde{E_x}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,1].set_title('$\widetilde{j_x}^{k_{||} d_i > 15}  \widetilde{E_x}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,1].set_title('$\widetilde{j_x}  \widetilde{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,1].set_title('$j_x  E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
        clrbar11 = plt.colorbar(im11, ax=axs[1,1], shrink = cbarshrink)#,format='%.1e')
        if(not(plotLog)):
            clrbar11.formatter.set_powerlimits((0, 0))

        #CEx_yz
        maxCe = max(np.max(CEx_yz),abs(np.min(CEx_yz)))
        if(plotLog):
            im12 = axs[0,2].pcolormesh(vz_yz,vy_yz,CEx_yz.T,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
        else:
            im12 = axs[0,2].pcolormesh(vz_yz,vy_yz,CEx_yz.T,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")

        if(plotFAC):
            axs[0,2].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
        else:
            axs[0,2].set_ylabel(r"$v_y/"+vnormstr+"$")
        axs[0,2].set_aspect('equal', 'box')
        axs[0,2].grid()
        if(computeJdotE):
            JdotE = compute_energization(CEx_yz.T,dv)
            if(True):
                if(plotFAC):
                    if(plotAvg):
                        axs[0,2].set_title('$\overline{j_{||}}  \overline{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,2].set_title('$\widetilde{j_{||}}^{k_{||} d_i < 15}  \widetilde{E_{||}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,2].set_title('$\widetilde{j_{||}}^{k_{||} d_i > 15}  \widetilde{E_{||}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,2].set_title('$\widetilde{j_{||}}  \widetilde{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,2].set_title('$j_{||}  E_{||}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    if(plotAvg):
                        axs[0,2].set_title('$\overline{j_x}  \overline{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,2].set_title('$\widetilde{j_x}^{k_{||} d_i > 15}  \widetilde{E_x}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,2].set_title('$\widetilde{j_x}^{k_{||} d_i > 15}  \widetilde{E_x}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,2].set_title('$\widetilde{j_x}  \widetilde{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,2].set_title('$j_x  E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
        clrbar12 = plt.colorbar(im12, ax=axs[1,2], shrink = cbarshrink)#,format='%.1e')
        if(not(plotLog)):
            clrbar12.formatter.set_powerlimits((0, 0))
    if(arrtype in ['CEy','CEperp1']):
        CEy_xy = arr_xy
        CEy_xz = arr_xz
        CEy_yz = arr_yz

        #CEy_xy
        maxCe = max(np.max(CEy_xy),abs(np.min(CEy_xy)))
        if(plotLog):
            im20 = axs[0,0].pcolormesh(vy_xy,vx_xy,CEy_xy,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
        else:
            im20 = axs[0,0].pcolormesh(vy_xy,vx_xy,CEy_xy,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")

        if(plotFAC):
            axs[0,0].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
        else:
            axs[0,0].set_ylabel(r"$v_y/"+vnormstr+"$")
        axs[0,0].set_aspect('equal', 'box')
        if(plotFAC):
            if(plotAvg):
                axs[0,0].text(-vmax*2.2,0, r"$\overline{C_{E_{\perp,1}}}$", ha='center', rotation=90, wrap=False)
            elif(plotFluc):
                if(isLowPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,1}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                elif(isHighPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,1}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,1}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[0,0].text(-vmax*2.2,0, r"$C_{E_{\perp,1}}$", ha='center', rotation=90, wrap=False)
        else:
            if(plotAvg):
                axs[0,0].text(-vmax*2.2,0, r"$\overline{C_{E_{y}}}$", ha='center', rotation=90, wrap=False)
            elif(plotFluc):
                if(isLowPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{y}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                elif(isHighPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{y}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{y}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[0,0].text(-vmax*2.2,0, r"$C_{E_{y}}$", ha='center', rotation=90, wrap=False)
        axs[0,0].grid()
        if(xpos != None):
            axs[0,0].text(-vmax*2.2,0,'$x/d_i = $' + str(xpos), ha='center', rotation=90, wrap=False)
        if(computeJdotE):
            JdotE = compute_energization(CEy_xy,dv)
            if(not(plotDiagJEOnly)):
                JdotE = compute_energization(CEx_xz,dv)
                if(plotFAC):
                    if(plotAvg):
                        axs[0,0].set_title('$\overline{j_{\perp,1}}  \overline{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,0].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i < 15}  \widetilde{E_{\perp,1}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,0].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i > 15}  \widetilde{E_{\perp,1}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,0].set_title('$\widetilde{j_{\perp,1}}  \widetilde{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left') 
                    else:
                        axs[0,0].set_title('$j_{\perp,1}  E_{\perp,1}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    if(plotAvg):
                        axs[0,0].set_title('$\overline{j_y}  \overline{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')             
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,0].set_title('$\widetilde{j_y}^{k_{||} d_i < 15}  \widetilde{E_y}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,0].set_title('$\widetilde{j_y}^{k_{||} d_i > 15}  \widetilde{E_y}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,0].set_title('$\widetilde{j_y}  \widetilde{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')             
                    else:
                        axs[0,0].set_title('$j_y  E_y$ = ' + "{:.2e}".format(JdotE),loc='left')
        
        clrbar10 = plt.colorbar(im20, ax=axs[0,0], shrink = cbarshrink)#,format='%.1e')
        if(not(plotLog)):
            clrbar10.formatter.set_powerlimits((0, 0))

        #CEy_xz
        maxCe = max(np.max(CEy_xz),abs(np.min(CEy_xz)))
        if(plotLog):
            im21 = axs[0,1].pcolormesh(vz_xz,vx_xz,CEy_xz,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
        else:
            im21 = axs[0,1].pcolormesh(vz_xz,vx_xz,CEy_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")

        if(plotFAC):
            axs[0,1].set_ylabel(r"$v_{\perp,2}/"+vnormstr+"$")
        else:
            axs[0,1].set_ylabel(r"$v_z/"+vnormstr+"$")
        axs[0,1].set_aspect('equal', 'box')
        axs[0,1].grid()
        if(computeJdotE):
            JdotE = compute_energization(CEy_xz,dv)
            if(True):
                JdotE = compute_energization(CEy_xz,dv)
                if(plotFAC):
                    if(plotAvg):
                        axs[0,1].set_title('$\overline{j_{\perp,1}}  \overline{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,1].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i < 15}  \widetilde{E_{\perp,1}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,1].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i > 15}  \widetilde{E_{\perp,1}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,1].set_title('$\widetilde{j_{\perp,1}}  \widetilde{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,1].set_title('$j_{\perp,1}  E_{\perp,1}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    if(plotAvg):
                        axs[0,1].set_title('$\overline{j_y}  \overline{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')  
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,1].set_title('$\widetilde{j_y}^{k_{||} d_i < 15}  \widetilde{E_y}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,1].set_title('$\widetilde{j_y}^{k_{||} d_i > 15}  \widetilde{E_y}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,1].set_title('$\widetilde{j_y}  \widetilde{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')             
                    else:
                        axs[0,1].set_title('$j_y  E_y$ = ' + "{:.2e}".format(JdotE),loc='left')

        clrbar11 = plt.colorbar(im21, ax=axs[0,1], shrink = cbarshrink)#,format='%.1e')
        if(not(plotLog)):
            clrbar11.formatter.set_powerlimits((0, 0))

        #CEy_yz
        maxCe = max(np.max(CEy_yz),abs(np.min(CEy_yz)))
        if(plotLog):
            im22 = axs[0,2].pcolormesh(vz_yz,vy_yz,CEy_yz.T, cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
        else:
            im22 = axs[0,2].pcolormesh(vz_yz,vy_yz,CEy_yz.T,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")

        if(plotFAC):
            axs[0,2].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
        else:
            axs[0,2].set_ylabel(r"$v_y/"+vnormstr+"$")
        axs[0,2].set_aspect('equal', 'box')
        axs[0,2].grid()
        if(computeJdotE):
            JdotE = compute_energization(CEy_yz.T,dv)
            if(not(plotDiagJEOnly)):
                if(plotFAC):
                    if(plotAvg):
                        axs[0,2].set_title('$\overline{j_{\perp,1}}  \overline{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,2].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i < 15}  \widetilde{E_{\perp,1}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,2].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i > 15}  \widetilde{E_{\perp,1}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,2].set_title('$\widetilde{j_{\perp,1}}  \widetilde{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,2].set_title('$j_{\perp,1}  E_{\perp,1}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    if(plotAvg):
                        axs[0,2].set_title('$\overline{j_y}  \overline{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,2].set_title('$\widetilde{j_y}^{k_{||} d_i < 15}  \widetilde{E_y}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,2].set_title('$\widetilde{j_y}^{k_{||} d_i > 15}  \widetilde{E_y}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,2].set_title('$\widetilde{j_y}  \widetilde{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,2].set_title('$j_y  E_y$ = ' + "{:.2e}".format(JdotE),loc='left')

        clrbar12 = plt.colorbar(im22, ax=axs[2,2], shrink = cbarshrink)#,format='%.1e')
        if(not(plotLog)):
            clrbar12.formatter.set_powerlimits((0, 0))
    
    if(arrtype in ['CEz','CEperp2']):
        CEz_xy = arr_xy
        CEz_xz = arr_xz
        CEz_yz = arr_yz

        #CEz_xy
        maxCe = max(np.max(CEz_xy),abs(np.min(CEz_xy)))
        if(plotLog):
            im30 = axs[0,0].pcolormesh(vy_xy,vx_xy,CEz_xy,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
        else:
            im30 = axs[0,0].pcolormesh(vy_xy,vx_xy,CEz_xy,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")

        axs[0,0].set_aspect('equal', 'box')
        if(plotFAC):
            if(plotAvg):
                axs[0,0].text(-vmax*2.2,0, r"$\overline{C_{E_{\perp,2}}}$", ha='center', rotation=90, wrap=False)
            elif(plotFluc):
                if(isLowPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,2}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                elif(isHighPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,2}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,2}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[0,0].text(-vmax*2.2,0, r"$C_{E_{\perp,2}}$", ha='center', rotation=90, wrap=False)
        else:
            if(plotAvg):
                axs[0,0].text(-vmax*2.2,0, r"$\overline{C_{E_{z}}}$", ha='center', rotation=90, wrap=False)
            elif(plotFluc):
                if(isLowPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{z}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                elif(isHighPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{z}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{z}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[0,0].text(-vmax*2.2,0, r"$C_{E_{z}}$", ha='center', rotation=90, wrap=False)
        axs[0,0].grid()
        if(metadata != None):
            axs[0,0].text(-vmax*2.2,0, metadata, ha='center', rotation=90, wrap=False)
        if(computeJdotE):
            JdotE = compute_energization(CEz_xy,dv)
            if(True):
                if(plotFAC):
                    if(plotAvg):
                        axs[0,0].set_title('$\overline{j_{\perp,2}}  \overline{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,0].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i < 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,0].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i > 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,0].set_title('$\widetilde{j_{\perp,2}}  \widetilde{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,0].set_title('$j_{\perp,2}  E_{\perp,2}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    if(plotAvg):
                        axs[0,0].set_title('$\overline{j_z}  \overline{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,0].set_title('$\widetilde{j_z}^{k_{||} d_i < 15}  \widetilde{E_z}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,0].set_title('$\widetilde{j_z}^{k_{||} d_i > 15}  \widetilde{E_z}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,0].set_title('$\widetilde{j_z}  \widetilde{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,0].set_title('$j_z  E_z$ = ' + "{:.2e}".format(JdotE),loc='left')

        clrbar10 = plt.colorbar(im30, ax=axs[0,0], shrink = cbarshrink)#,format='%.1e')
        if(not(plotLog)):
            clrbar10.formatter.set_powerlimits((0, 0))

        #CEz_xz
        maxCe = max(np.max(CEz_xz),abs(np.min(CEz_xz)))
        if(plotLog):
            im31 = axs[0,1].pcolormesh(vz_xz,vx_xz,CEz_xz,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
        else:
            im31 = axs[0,1].pcolormesh(vz_xz,vx_xz,CEz_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
        #axs[3,1].set_title('$C_{Ez}(v_x,v_z)$')

        axs[0,1].set_aspect('equal', 'box')
        axs[0,1].grid()
        if(computeJdotE):
            JdotE = compute_energization(CEz_xz,dv)
            if(not(plotDiagJEOnly)):
                if(plotFAC):
                    if(plotAvg):
                        axs[0,1].set_title('$\overline{j_{\perp,2}}  \overline{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,1].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i < 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,1].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i > 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,1].set_title('$\widetilde{j_{\perp,2}}  \widetilde{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,1].set_title('$j_{\perp,2}  E_{\perp,2}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    if(plotAvg):
                        axs[0,1].set_title('$\overline{j_z}  \overline{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,1].set_title('$\widetilde{j_z}^{k_{||} d_i < 15}  \widetilde{E_z}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,1].set_title('$\widetilde{j_z}^{k_{||} d_i > 15}  \widetilde{E_z}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,1].set_title('$\widetilde{j_z}  \widetilde{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,1].set_title('$j_z  E_z$ = ' + "{:.2e}".format(JdotE),loc='left')

        clrbar11 = plt.colorbar(im31, ax=axs[0,1], shrink = cbarshrink)#,format='%.1e')
        if(not(plotLog)):
            clrbar11.formatter.set_powerlimits((0, 0))

        #CEz_yz
        maxCe = max(np.max(CEz_yz),abs(np.min(CEz_yz)))
        if(plotLog):
            im32 = axs[0,2].pcolormesh(vz_yz,vy_yz,CEz_yz.T,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
        else:
            im32 = axs[0,2].pcolormesh(vz_yz,vy_yz,CEz_yz.T,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")

        axs[0,2].set_aspect('equal', 'box')
        axs[0,2].grid()
        if(computeJdotE):
            JdotE = compute_energization(CEz_yz.T,dv)
            if(not(plotDiagJEOnly)):
                if(plotFAC):
                    if(plotAvg):
                        axs[0,2].set_title('$\overline{j_{\perp,2}}  \overline{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,2].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i < 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,2].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i > 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,2].set_title('$\widetilde{j_{\perp,2}}  \widetilde{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,2].set_title('$j_{\perp,2}  E_{\perp,2}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    if(plotAvg):
                        axs[0,2].set_title('$\overline{j_z}  \overline{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,2].set_title('$\widetilde{j_z}^{k_{||} d_i < 15}  \widetilde{E_z}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,2].set_title('$\widetilde{j_z}^{k_{||} d_i > 15}  \widetilde{E_z}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,2].set_title('$\widetilde{j_z}  \widetilde{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,2].set_title('$j_z  E_z$ = ' + "{:.2e}".format(JdotE),loc='left')

        clrbar12 = plt.colorbar(im32, ax=axs[0,2], shrink = cbarshrink)#,format='%.1e')
        if(not(plotLog)):
            clrbar12.formatter.set_powerlimits((0, 0))

    if(arrtype in ['Ctot']):
        CEtot_xy = arr_xy
        CEtot_xz = arr_xz
        CEtot_yz = arr_yz

        #CEtot_xy
        maxCe = max(np.max(CEtot_xy),abs(np.min(CEtot_xy)))
        if(plotLog):
            im30 = axs[0,0].pcolormesh(vy_xy,vx_xy,CEtot_xy,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
        else:
            im30 = axs[0,0].pcolormesh(vy_xy,vx_xy,CEtot_xy,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")

        axs[0,0].set_aspect('equal', 'box')
        if(plotFAC):
            if(plotAvg):
                axs[0,0].text(-vmax*2.2,0, r"$\overline{C_{E_{tot}}}$", ha='center', rotation=90, wrap=False)
            elif(plotFluc):
                if(isLowPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{tot}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                elif(isHighPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{tot}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{tot}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[0,0].text(-vmax*2.2,0, r"$C_{E_{tot}}$", ha='center', rotation=90, wrap=False)
        else:
            if(plotAvg):
                axs[0,0].text(-vmax*2.2,0, r"$\overline{C_{E_{z}}}$", ha='center', rotation=90, wrap=False)
            elif(plotFluc):
                if(isLowPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{tot}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
                elif(isHighPass):
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{tot}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
                else:
                    axs[0,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{tot}}}$", ha='center', rotation=90, wrap=False)
            else:
                axs[0,0].text(-vmax*2.2,0, r"$C_{E_{tot}}$", ha='center', rotation=90, wrap=False)
        axs[0,0].grid()
        if(metadata != None):
            axs[0,0].text(-vmax*2.2,0, metadata, ha='center', rotation=90, wrap=False)
        if(computeJdotE):
            JdotE = compute_energization(CEtot_xy,dv)
            if(True):
                if(plotFAC):
                    if(plotAvg):
                        axs[0,0].set_title('$\overline{\mathbf{j}} \cdot \overline{\mathbf{E}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,0].set_title('$\widetilde{\mathbf{j}}^{k_{||} d_i < 15}  \cdot \widetilde{\mathbf{E}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,0].set_title('$\widetilde{\mathbf{j}}^{k_{||} d_i > 15}  \cdot \widetilde{\mathbf{E}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,0].set_title('$\widetilde{\mathbf{j}} \cdot \widetilde{\mathbf{E}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,0].set_title('$\mathbf{j} \cdot  \mathbf{E}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    if(plotAvg):
                        axs[0,0].set_title('$\overline{\mathbf{j}}  \cdot \overline{\mathbf{E}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,0].set_title('$\widetilde{\mathbf{j}}^{k_{||} d_i < 15} \cdot \widetilde{\mathbf{E}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,0].set_title('$\widetilde{\mathbf{j}}^{k_{||} d_i > 15} \cdot \widetilde{\mathbf{E}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,0].set_title('$\widetilde{\mathbf{j}} \cdot \widetilde{\mathbf{E}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,0].set_title('$\mathbf{j} \cdot  \mathbf{E}$ = ' + "{:.2e}".format(JdotE),loc='left')

        clrbar10 = plt.colorbar(im30, ax=axs[0,0], shrink = cbarshrink)#,format='%.1e')
        if(not(plotLog)):
            clrbar10.formatter.set_powerlimits((0, 0))

        #CEtot_xz
        maxCe = max(np.max(CEtot_xz),abs(np.min(CEtot_xz)))
        if(plotLog):
            im31 = axs[0,1].pcolormesh(vz_xz,vx_xz,CEtot_xz,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
        else:
            im31 = axs[0,1].pcolormesh(vz_xz,vx_xz,CEtot_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
        #axs[3,1].set_title('$C_{Ez}(v_x,v_z)$')

        axs[0,1].set_aspect('equal', 'box')
        axs[0,1].grid()
        if(computeJdotE):
            JdotE = compute_energization(CEtot_xz,dv)
            if(not(plotDiagJEOnly)):
                if(plotFAC):
                    if(plotAvg):
                        axs[0,1].set_title('$\overline{j_{\perp,2}}  \overline{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,1].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i < 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,1].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i > 15}  \widetilde{E_{\perp,2}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,1].set_title('$\widetilde{j_{\perp,2}}  \widetilde{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,1].set_title('$j_{\perp,2}  E_{\perp,2}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    if(plotAvg):
                        axs[0,1].set_title('$\overline{j_z}  \overline{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,1].set_title('$\widetilde{j_z}^{k_{||} d_i < 15}  \widetilde{E_z}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,1].set_title('$\widetilde{j_z}^{k_{||} d_i > 15}  \widetilde{E_z}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,1].set_title('$\widetilde{j_z}  \widetilde{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,1].set_title('$j_z  E_z$ = ' + "{:.2e}".format(JdotE),loc='left')

        clrbar11 = plt.colorbar(im31, ax=axs[0,1], shrink = cbarshrink)#,format='%.1e')
        if(not(plotLog)):
            clrbar11.formatter.set_powerlimits((0, 0))

        #CEtot_yz
        maxCe = max(np.max(CEtot_yz),abs(np.min(CEtot_yz)))
        if(plotLog):
            im32 = axs[0,2].pcolormesh(vz_yz,vy_yz,CEtot_yz.T,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
        else:
            im32 = axs[0,2].pcolormesh(vz_yz,vy_yz,CEtot_yz.T,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")

        axs[0,2].set_aspect('equal', 'box')
        axs[0,2].grid()
        if(computeJdotE):
            JdotE = compute_energization(CEtot_yz.T,dv)
            if(not(plotDiagJEOnly)):
                if(plotFAC):
                    if(plotAvg):
                        axs[0,2].set_title('$\overline{\mathbf{j}}  \cdot \overline{\mathbf{E}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,2].set_title('$\widetilde{\mathbf{j}}^{k_{||} d_i < 15} \cdot \widetilde{\mathbf{E}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,2].set_title('$\widetilde{\mathbf{j}}^{k_{||} d_i > 15} \cdot  \widetilde{\mathbf{E}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,2].set_title('$\widetilde{\mathbf{j}}  \cdot \widetilde{\mathbf{E}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,2].set_title('$\mathbf{j} \cdot  \cdot \mathbf{E}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    if(plotAvg):
                        axs[0,2].set_title('$\overline{\mathbf{j}} \cdot  \overline{\mahtbf{E}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(plotFluc):
                        if(isLowPass):
                            axs[0,2].set_title('$\widetilde{\mathbf{j}}^{k_{||} d_i < 15}  \cdot \widetilde{\mathbf{E}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        elif(isHighPass):
                            axs[0,2].set_title('$\widetilde{\mathbf{j}}^{k_{||} d_i > 15}  \cdot \widetilde{\mathbf{E}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                        else:
                            axs[0,2].set_title('$\widetilde{\mathbf{j}} \cdot  \widetilde{\mathbf{E}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[0,2].set_title('$\mathbf{j} \cdot  \mathbf{E}$ = ' + "{:.2e}".format(JdotE),loc='left')

        clrbar12 = plt.colorbar(im32, ax=axs[0,2], shrink = cbarshrink)#,format='%.1e')
        if(not(plotLog)):
            clrbar12.formatter.set_powerlimits((0, 0))


    for _j in range(0,3):
        axs[0,_j].set_xlim(-vmax,vmax)
        axs[0,_j].set_ylim(-vmax,vmax)


    if(plotFAC):
        axs[0,0].set_xlabel(r"$v_{||}/"+vnormstr+"$")
        axs[0,0].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
    else:
        axs[0,0].set_xlabel(r"$v_x/"+vnormstr+"$")
        axs[0,0].set_ylabel(r"$v_y/"+vnormstr+"$")

    if(plotFAC):
        axs[0,1].set_xlabel(r"$v_{||}/"+vnormstr+"$")
        axs[0,1].set_ylabel(r"$v_{\perp,2}/"+vnormstr+"$")
    else:
        axs[0,1].set_xlabel(r"$v_x/"+vnormstr+"$")
        axs[0,1].set_ylabel(r"$v_z/"+vnormstr+"$")

    if(plotFAC):
        axs[0,2].set_xlabel(r"$v_{\perp,2}/"+vnormstr+"$")
        axs[0,2].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
    else:
        axs[0,2].set_xlabel(r"$v_z/"+vnormstr+"$")
        axs[0,2].set_ylabel(r"$v_y/"+vnormstr+"$")

    #set ticks
    intvl = 1.
    if(vmax > 5):
        intvl = 5.
    if(vmax > 10):
        intvl = 10.
    tcks = np.arange(0,vmax,intvl)
    tcks = np.concatenate((-1*np.flip(tcks),tcks))

    for _j in range(0,3):
        axs[0,_j].set_xticks(tcks)
        axs[0,_j].set_yticks(tcks)


    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight')
            
        #must make figure first to grab x10^val on top of color bar- after grabbing it, we can move it- a little wasteful but it was quick solution
        clrbar10text = str(clrbar10.ax.yaxis.get_offset_text().get_text())
        clrbar10.ax.yaxis.get_offset_text().set_visible(False)
        axs[0,0].text(1.63*maxplotvval,-1.27*maxplotvval,clrbar10text, va='bottom', ha='center')
        clrbar11text = str(clrbar11.ax.yaxis.get_offset_text().get_text())
        clrbar11.ax.yaxis.get_offset_text().set_visible(False) 
        axs[0,1].text(1.63*maxplotvval,-1.27*maxplotvval,clrbar11text, va='bottom', ha='center')
        clrbar12text = str(clrbar12.ax.yaxis.get_offset_text().get_text())
        clrbar12.ax.yaxis.get_offset_text().set_visible(False)
        axs[0,2].text(1.63*maxplotvval,-1.27*maxplotvval,clrbar12text, va='bottom', ha='center')
 
        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight')
    else:
        plt.show()
    plt.close('all') #saves RAM

def make_9panel_sweep_from_2v(Hist_vxvy, Hist_vxvz, Hist_vyvz,
                              C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
                              C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
                              C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
                              vx, vy,vz,params_in,x,metadata,
                              directory,plotLog=False):
    """
    Makes sweep of super figure of distribution and velocity sigantures from all different projections
    i.e. different viewing angles

    Parameters
    ----------
    vx : 2d array
        vx velocity grid
    vy : 2d array
        vy velocity grid
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    H_** : 2d array
        projection onto ** axis of distribution function
    CE*_* : 2d array
        projection onto ** axis of CE*
    flnm : str, optional
        specifies filename if plot is to be saved as png.
        if set to default, plt.show() will be called instead
    params : dict
        dictionary with simulation parameters in it
    metadata : array of strings
        array where each element is a string describing metadata to be shown on plot
    directory : string
        name of output directory for pngs of each frame of sweep
    plotLog : bool
        if true will plot using colors.SymLogNorm()
        that is will plot both negative and positive values using a log color scale when possible and a linear scale from -1 to 1
    """

    try:
        os.mkdir(directory)
    except:
        pass

    vmax = np.max(vz)
    print(vmax)

    for i in range(0,len(Hist_vxvy)):
        print('Making plot ' + str(i) + ' of ' + str(len(Hist_vxvy)))
        mdt = str('Metadata = ' + str(metadata[i]))
        plot_cor_and_dist_supergrid(vx, vy, vz, vmax,
                                    Hist_vxvy[i], Hist_vxvz[i], Hist_vyvz[i],
                                    C_Ex_vxvy[i], C_Ex_vxvz[i], C_Ex_vyvz[i],
                                    C_Ey_vxvy[i], C_Ey_vxvz[i], C_Ey_vyvz[i],
                                    C_Ez_vxvy[i], C_Ez_vxvz[i], C_Ez_vyvz[i],
                                    flnm = directory+str(i).zfill(7), computeJdotE = False, params = params_in, metadata = mdt, xpos = x[i], plotLog=plotLog)

def make_superplot_gif(vx, vy, vz, vmax, Hist, CEx, CEy, CEz, x, directory):
    """
    Make superplots of data and put into directory

    Parameters
    ----------
    vx : 3d array
        vx velocity grid
    vy : 3d array
        vy velocity grid
    vz : 3d array
        vz velocity grid
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    Hist : 4d array
        distribution function data f(x;vx,vy,vz)
    CEx : 4d array
        distribution function data CEx(x;vx,vy,vz)
    CEy : 4d array
        distribution function data CEy(x;vx,vy,vz)
    CEz : 4d array
        distribution function data CEz(x;vx,vy,vz)
    x : 1d array
        x coordinate data
    directory : str
        name of directory you want to create and put plots into
        (omit final '/')
    """

    from FPCAnalysis.array_ops import array_3d_to_2d

    try:
        os.mkdir(directory)
    except:
        pass

    for i in range(0,len(x)):
        print('Making plot ' + str(i)+' of '+str(len(x)))
        flnm = directory+'/'+str(i).zfill(6)

        #Project onto 2d axis
        H_xy = array_3d_to_2d(Hist[i],'xy')
        H_xz = array_3d_to_2d(Hist[i],'xz')
        H_yz = array_3d_to_2d(Hist[i],'yz')
        CEx_xy = array_3d_to_2d(CEx[i],'xy')
        CEx_xz = array_3d_to_2d(CEx[i],'xz')
        CEx_yz = array_3d_to_2d(CEx[i],'yz')
        CEy_xy = array_3d_to_2d(CEy[i],'xy')
        CEy_xz = array_3d_to_2d(CEy[i],'xz')
        CEy_yz = array_3d_to_2d(CEy[i],'yz')
        CEz_xy = array_3d_to_2d(CEz[i],'xy')
        CEz_xz = array_3d_to_2d(CEz[i],'xz')
        CEz_yz = array_3d_to_2d(CEz[i],'yz')

        plot_cor_and_dist_supergrid(vx, vy, vz, vmax,
                                        H_xy, H_xz, H_yz,
                                        CEx_xy,CEx_xz, CEx_yz,
                                        CEy_xy,CEy_xz, CEy_yz,
                                        CEz_xy,CEz_xz, CEz_yz,
                                        flnm = flnm, ttl = 'x(di): ' + str(x[i]))
        plt.close('all') #saves RAM

def project_dist_1d(vx,vy,vz,hist,axis):
    """

    """
    import matplotlib.ticker as mtick

    if(axis == 'vx'):
        plotx = vx[0,0,:]
        hist_vyvx = np.sum(hist,axis=0)
        hist_vx = np.sum(hist_vyvx,axis=0)
        ploty = hist_vx

    elif(axis == 'vy'):
        plotx = vy[0,:,0]
        hist_vyvx = np.sum(hist,axis=0)
        hist_vy = np.sum(hist_vyvx,axis=1)
        ploty = hist_vy

    elif(axis == 'vz'):
        plotx = vz[:,0,0]
        hist_vzvy = np.sum(hist,axis=2)
        hist_vz = np.sum(hist_vzvy,axis=1)
        ploty = hist_vz

    else:
        print("Please uses axis = vx, vy, or vz...")
        return

    plotymax = 1.1*np.max([np.max(ploty),-1*np.min(ploty)])

    plt.figure()
    plt.ylim(-plotymax,plotymax)
    plt.plot(plotx,ploty,color='black',linewidth=1.5)
    plt.gca().set_aspect(1.0/plt.gca().get_data_ratio(), adjustable='box')
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.xlabel(axis+'/vti')
    plt.grid()
    plt.ylabel('f('+axis+'/vti)')
    plt.show()

def make_FPC_plot(dfields,dpar,dv,vmax,vshock,x1,x2,fieldkey,planename,flnm='',usefluc=False,usebar=False):
    """
    Wrapper for plot_velsig
    """

    ttl = ''
    directionkey = fieldkey[-1]
    #sets up labels for plot
    if(usefluc):
        ttl = '$\widetilde{C}'
    elif(usebar):
        ttl = '$\overline{C}'
    else:
        ttl = '$C'

    if(fieldkey == 'epar'):
        directionkey = 'epar'
        ttl += '_{E_{||}}'
    elif(fieldkey == 'eperp1'):
        directionkey = 'eperp1'
        ttl += '_{E_{\perp,1}}'
    elif(fieldkey == 'eperp2'):
        directionkey = 'eperp2'
        ttl += '_{E_{\perp,2}}'
    elif(fieldkey == 'ex'):
        ttl += '_{E_{x}}'
    elif(fieldkey == 'ey'):
        ttl += '_{E_{y}}'
    elif(fieldkey == 'ez'):
        ttl += '_{E_{z}}'

    if(planename == 'parperp1'):
        ttl += '(v_{||},v_{\perp,1})$'
        xlbl = "$v_{||}/v_{ts}$"
        ylbl =  "$v_{\perp,1}/v_{ts}$"
    elif(planename == 'parperp2'):
        ttl += '(v_{||},v_{\perp,2})$'
        xlbl = "$v_{||}/v_{ts}$"
        ylbl =  "$v_{\perp,2}/v_{ts}$"
    elif(planename == 'perp1perp2'):
        ttl += '(v_{\perp,1},v_{\perp,2})$'
        xlbl = "$v_{\perp,1}/v_{ts}$"
        ylbl =  "$v_{\perp,2}/v_{ts}$"
    elif(planename == 'xy'):
        ttl += '(v_x,v_y)$'
        xlbl = "$v_{x}/v_{ts}$"
        ylbl =  "$v_{y}/v_{ts}$"
    elif(planename == 'xz'):
        ttl += '(v_x,v_z)$'
        xlbl = "$v_{x}/v_{ts}$"
        ylbl =  "$v_{z}/v_{ts}$"
    elif(planename == 'yz'):
        ttl += '(v_y,v_z)$'
        xlbl = "$v_{y}/v_{ts}$"
        ylbl =  "$v_{z}/v_{ts}$"
    ttl+='; '

    #for now, we use the full yz domain to capture as many particles as possible
    y1 = dfields['ex_yy'][0]
    y2 = dfields['ex_yy'][-1]
    z1 = dfields['ex_zz'][0]
    z2 = dfields['ex_zz'][-1]

    #computes FPC
    #note: v1[k,j,i],v2[k,j,i],v3[k,j,i] corresponds to cor[k,j,i]
    #where v1 corresponds to vx/vpar (in stand/fieldaligned)
    #where v2 corresponds to vy/vperp1 (in stand/fieldaligned)
    #where v3 corresponds to vz/vperp2 (in stand/fieldaligned)
    from FPCAnalysis.fpc import compute_hist_and_cor

    v1, v2, v3, totalPtcl, totalFieldpts, hist, cor = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2,
                                                                            dpar, dfields, vshock, fieldkey, directionkey)
    del totalFieldpts #this needs to be removed from the code as it is not used anymore


    print("npar in box: ",str(np.sum(hist)))

    #makes plot
    from FPCAnalysis.array_ops import array_3d_to_2d
    CEiproj = array_3d_to_2d(cor, planename)

    if(flnm != ''):
        print("Requesting that file is saved as",flnm,"!")

    plot_velsig(v1,v2,v3,dv,vmax,CEiproj,fieldkey,planename,ttl=ttl,xlabel=xlbl,ylabel=ylbl,flnm=flnm)

def plot_dist_v_fields_supergrid(vx, vy, vz, vmax,
                                H_xy, H_xz, H_yz,
                                CEx_xy,CEx_xz, CEx_yz,
                                CEy_xy,CEy_xz, CEy_yz,
                                CEz_xy,CEz_xz, CEz_yz,
                                dfavg,xval1,xval2,
                                flnm = '', ttl = '', computeJdotE = True, params = None, metadata = None, xpos = None, plotLog = False, plotLogHist = True,
                                plotFAC = False, plotFluc = False, plotAvg = False, isIon = True, listpos=False,xposval=None,normtoN = False,Nval=None,isLowPass=False,isHighPass=False,plotDiagJEOnly=True):
    """
    Makes super figure of distribution and velocity sigantures from all different projections
    i.e. different viewing angles

    Parameters
    ----------
    vx : 2d array
        vx velocity grid
    vy : 2d array
        vy velocity grid
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    H_** : 2d array
        projection onto ** axis of distribution function
    CE*_* : 2d array
        projection onto ** axis of CE*
    flnm : str, optional
        specifies filename if plot is to be saved as png.
        if set to default, plt.show() will be called instead
    ttl : str, optional
        title of plot
    computeJdotE : bool, optional
        compute and write JdotE for each panel as title of each sub plot
    params : dict, optional
        dictionary with simulation parameters in it
    metadata : string, optional
        string describing metadata to be shown on plot
    """
    from matplotlib.colors import LogNorm
    from FPCAnalysis.array_ops import mesh_3d_to_2d
    from FPCAnalysis.analysis import compute_energization
    import matplotlib.colors as colors

    if(normtoN):
        if(Nval == None):
            Nval = np.sum(H_xy)
        CEx_xy/=Nval
        CEx_xz/=Nval
        CEx_yz/=Nval
        CEy_xy/=Nval
        CEy_xz/=Nval
        CEy_yz/=Nval
        CEz_xy/=Nval
        CEz_xz/=Nval
        CEz_yz/=Nval

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    #fig, axs = plt.subplots(4,3,figsize=(4*5,3*5),sharex=True)

    # Create figure and subplots
    fig = plt.figure(figsize=(4*5, 3*5*.5))
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax3 = plt.subplot2grid((2, 3), (0, 2))

    ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    #col_start = 1  # Start column index
    #col_width = 0.6  # Fraction of the total width
    #ax4 = plt.subplot2grid((2, 3), (1, col_start), colspan=int(col_width))

    axs = []
    axs.append([ax1,ax2,ax3])
    axs = np.asarray(axs)

    plt.rcParams['axes.titlepad'] = 8  # pad is in points...

    _hspace = .15
    _wspace = -.15
    if(computeJdotE):
        _hspace+=.275
    if(plotLog):
        _wspace+=.175
    fig.subplots_adjust(hspace=_hspace,wspace=_wspace)

    vx_xy, vy_xy = mesh_3d_to_2d(vx,vy,vz,'xy')
    vx_xz, vz_xz = mesh_3d_to_2d(vx,vy,vz,'xz')
    vy_yz, vz_yz = mesh_3d_to_2d(vx,vy,vz,'yz')

    dv = vy_yz[1][1]-vy_yz[0][0] #assumes square velocity grid

    if(isIon):
        vnormstr = 'v_{ti}'
    else:
        vnormstr = 'v_{te}'

    fig.suptitle(ttl)

    # fig, axes = plt.subplots(nrows=2)
    # ax0label = axes[0].set_ylabel('Axes 0')
    # ax1label = axes[1].set_ylabel('Axes 1')
    #
    # title = axes[0].set_title('Title')
    #
    clboffset = np.array([vmax*.15,-vmax*.15])
    # title.set_position(ax0label.get_position() + offset)
    # title.set_rotation(90)

    try:
        minHxyval = np.min(H_xy[np.nonzero(H_xy)])
        minHxzval = np.min(H_xz[np.nonzero(H_xz)])
        minHyzval = np.min(H_yz[np.nonzero(H_yz)])
        maxH_xy = H_xy.max()
        maxH_xz = H_xz.max()
        maxH_yz = H_yz.max()
    except:
        minHxyval = 0.00000000001
        minHxzval = 0.00000000001
        minHyzval = 0.00000000001
        maxH_xy = 1.
        maxH_xz = 1.
        maxH_yz = 1.

    #H_xy
    if(plotLogHist):
        im00= axs[0,0].pcolormesh(vy_xy, vx_xy, H_xy, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHxyval, vmax=maxH_xy))
    else:
        im00= axs[0,0].pcolormesh(vy_xy, vx_xy, H_xy, cmap="plasma", shading="gouraud",vmin=1,vmax=600)
    #axs[0,0].set_title(r"$f(v_x, v_y)$")
    if(plotFAC):
        axs[0,0].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
    else:
        axs[0,0].set_ylabel(r"$v_y/"+vnormstr+"$")
    axs[0,0].set_aspect('equal', 'box')
    axs[0,0].grid()
    clrbar00 = plt.colorbar(im00, ax=axs[0,0])#,format='%.1e')
    if(not(plotLogHist)):
        clrbar00.formatter.set_powerlimits((0, 0))
    #axs[0,0].text(-vmax*2.2,0, r"$f$", ha='center', rotation=90, wrap=False)
    if(params != None):
        axs[0,0].text(-vmax*2.6,0, '$M_A = $ ' + str(abs(params['MachAlfven'])), ha='center', rotation=90, wrap=False)

    #if(listpos):
    #    axs[0,0].text(-vmax*2.6,0, '$x / d_i = $ ' + str("{:.4f}".format(xposval)), ha='center', rotation=90, wrap=False)


    #H_xz
    if(plotLogHist):
        im01 = axs[0,1].pcolormesh(vz_xz, vx_xz, H_xz, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHxzval, vmax=maxH_xz))
    else:
        im01 = axs[0,1].pcolormesh(vz_xz, vx_xz, H_xz, cmap="plasma", shading="gouraud",vmin=1,vmax=600)
    #axs[0,1].set_title(r"$f(v_x, v_z)$")
    if(plotFAC):
        axs[0,1].set_ylabel(r"$v_{\perp,2}/"+vnormstr+"$")
    else:
        axs[0,1].set_ylabel(r"$v_z/"+vnormstr+"$")
    axs[0,1].set_aspect('equal', 'box')
    axs[0,1].grid()
    clrbar01 = plt.colorbar(im01, ax=axs[0,1])#,format='%.1e')
    if(not(plotLogHist)):
        clrbar01.formatter.set_powerlimits((0, 0))

    #H_yz
    if(plotLogHist):
        im02 = axs[0,2].pcolormesh(vz_yz, vy_yz, H_yz.T, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHyzval, vmax=maxH_yz))
    else:
        im02 = axs[0,2].pcolormesh(vz_yz, vy_yz, H_yz.T, cmap="plasma", shading="gouraud",vmin=1,vmax=600)
    #axs[0,2].set_title(r"$f(v_y, v_z)$")
    if(plotFAC):
        axs[0,2].set_ylabel(r"$v_{\perp,1}/"+vnormstr+"$")
    else:
        axs[0,2].set_ylabel(r"$v_y/"+vnormstr+"$")
    axs[0,2].set_aspect('equal', 'box')
    axs[0,2].grid()
    clrbar02 = plt.colorbar(im02, ax=axs[0,2])#,format='%.1e')
    if(not(plotLogHist)):
        clrbar02.formatter.set_powerlimits((0, 0))

    im00.set_clim(1, 600)
    im01.set_clim(1, 600)
    im02.set_clim(1, 600)

    axs[0,0].set_xlabel(r"$v_{x}/"+vnormstr+"$")
    axs[0,1].set_xlabel(r"$v_{x}/"+vnormstr+"$")
    axs[0,2].set_xlabel(r"$v_{z}/"+vnormstr+"$")

    axs[0,0].set_title(r"$f(v_x,v_y)$", loc='right')
    axs[0,1].set_title(r"$f(v_x,v_z)$", loc='right')
    axs[0,2].set_title(r"$f(v_y,v_z)$", loc='right')
    

    btot = np.sqrt(dfavg['bx'][0,0,:]**2+dfavg['by'][0,0,:]**2+dfavg['bz'][0,0,:]**2)
    btot0 = btot[int(len(dfavg['bx_xx'])*.8)]
    ax4.plot(dfavg['bx_xx'],btot/btot0,color='black')
    ax4.axvspan(xval1, xval2, facecolor='gray', alpha=0.420)
    ax4.set_ylabel(r'$|\mathbf{B}|/B_0$')
    ax4.grid()


    for _i in range(0,1):
        for _j in range(0,3):
            axs[_i,_j].set_xlim(-vmax,vmax)
            axs[_i,_j].set_ylim(-vmax,vmax)
            #axs[_i,_j].set_title(r"$f$", loc='right')
    ax4.set_xlim(5,12)
    ax4.set_xlabel(r'$x/d_i$')

    #set ticks
    intvl = 1.
    if(vmax > 5):
        intvl = 5.
    if(vmax > 10):
        intvl = 10.
    tcks = np.arange(0,vmax,intvl)
    tcks = np.concatenate((-1*np.flip(tcks),tcks))
    for _i in range(0,1):
        for _j in range(0,3):
            axs[_i,_j].set_xticks(tcks)
            axs[_i,_j].set_yticks(tcks)

    #plt.subplots_adjust(hspace=.5,wspace=-.3)


    maxplotvval = np.max(vz_yz) #Assumes square grid of even size
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight')

        #must make figure first to grab x10^val on top of color bar- after grabbing it, we can move it- a little wasteful but it was quick solution
        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight')

        plt.close('all') #saves RAM
    else:
        plt.show()
    plt.close()




def plot_gyro(vpar,vperp,corepargyro,coreperpgyro,flnm='',isIon=True,plotLog=False,computeJdotE=True,npar=None,plotAvg = False, plotFluc = False,isLowPass=False,isHighPass=False):
    """

    """
    if(npar != None):
        corepargyro /= npar
        coreperpgyro /= npar

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    fig, axs = plt.subplots(1,2,figsize=(2*5,4*5),sharey=True)

    vmax = np.max(vpar)

    _hspace = .15
    _wspace = .3
    if(plotLog):
        _wspace+=.175
    fig.subplots_adjust(hspace=_hspace,wspace=_wspace)

    if(isIon):
        vnormstr = 'v_{ti}'
    else:
        vnormstr = 'v_{te}'

    clboffset = np.array([vmax*.15,-vmax*.15])

    maxCe = max(np.max(corepargyro),abs(np.max(corepargyro)))
    if(plotLog):
        im11 = axs[0].pcolormesh(vpar,vperp,corepargyro, cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im11 = axs[0].pcolormesh(vpar,vperp,corepargyro,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[0].set_xlabel(r"$v_{||}/"+vnormstr+"$")
    axs[0].set_ylabel(r"$v_{\perp}/"+vnormstr+"$")
    axs[0].set_aspect('equal', 'box')
    axs[0].grid()
    if(plotFluc):
        if(isLowPass):
            axs[0].set_title(r'$\widetilde{C_{E_{||}}}^{k_{||} d_i < 15}(v_{||},v_{\perp})$', loc='right')
        elif(isHighPass):
            axs[0].set_title(r'$\widetilde{C_{E_{||}}}^{k_{||} d_i > 15}(v_{||},v_{\perp})$', loc='right')
        else:
            axs[0].set_title(r'$\widetilde{C_{E_{||}}}(v_{||},v_{\perp})$', loc='right')
    elif(plotAvg):
        axs[0].set_title(r'$\overline{C_{E_{||}}}(v_{||},v_{\perp})$', loc='right')
    else:
        axs[0].set_title(r'$C_{E_{||}}(v_{||},v_{\perp})$', loc='right')

    maxCe = max(np.max(coreperpgyro),abs(np.max(coreperpgyro)))
    if(plotLog):
        im12 = axs[1].pcolormesh(vpar,vperp,coreperpgyro,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im12 = axs[1].pcolormesh(vpar,vperp,coreperpgyro,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[1].set_xlabel(r"$v_{||}/"+vnormstr+"$")
    axs[1].set_aspect('equal', 'box')
    axs[1].grid()
    if(plotFluc):
        if(isLowPass):
            axs[1].set_title(r'$\widetilde{C_{E_{\perp}}}^{k_{||} d_i < 15}(v_{||},v_{\perp})$', loc='right')
        elif(isHighPass):
            axs[1].set_title(r'$\widetilde{C_{E_{\perp}}}^{k_{||} d_i > 15}(v_{||},v_{\perp})$', loc='right')
        else:
            axs[1].set_title(r'$\widetilde{C_{E_{\perp}}}(v_{||},v_{\perp})$', loc='right')
    elif(plotAvg):
        axs[1].set_title(r'$\overline{C_{E_{\perp}}}(v_{||},v_{\perp})$', loc='right')
    else:
        axs[1].set_title(r'$C_{E_{\perp}}(v_{||},v_{\perp})$', loc='right')

    axs[0].set_xlim(-vmax,vmax)
    axs[0].set_ylim(0,vmax)
    axs[1].set_xlim(-vmax,vmax)
    axs[1].set_ylim(0,vmax)

    if(computeJdotE):
        JdotEpar = np.sum(corepargyro)
        JdotEperp = np.sum(coreperpgyro)

        sval = 'e'
        if(isIon):
            sval = 'i'

        if(plotFluc):
            if(isLowPass):
                axs[0].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{||,'+sval+'}}^{k_{||} d_i < 15}  \widetilde{E}_{||}^{k_{||} d_i < 15} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
            elif(isHighPass):
                axs[0].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{||,'+sval+'}}^{k_{||} d_i < 15}  \widetilde{E}_{||}^{k_{||} d_i > 15} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
            else:
                axs[0].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{||,'+sval+'}}  \widetilde{E}_{||} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
        elif(plotAvg):
            axs[0].text(-vmax*0.85,vmax*0.75,r'$\overline{j_{||,'+sval+'}}  \overline{E}_{||} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
        else:
            axs[0].text(-vmax*0.85,vmax*0.75,r'$j_{||,'+sval+'}  E_{||} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
       
        if(plotFluc):
            if(isLowPass):
                axs[1].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,'+sval+'}}^{k_{||} d_i < 15}  \widetilde{E}_{\perp}^{k_{||} d_i < 15} = $'+ "{:.2e}".format(JdotEperp),fontsize=12)
            elif(isHighPass):
                axs[1].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,'+sval+'}}^{k_{||} d_i < 15}  \widetilde{E}_{\perp}^{k_{||} d_i > 15} = $'+ "{:.2e}".format(JdotEperp),fontsize=12)
            else:
                axs[1].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,'+sval+'}}  \widetilde{E}_{\perp} = $'+ "{:.2e}".format(JdotEperp),fontsize=12)
        elif(plotAvg):
            axs[1].text(-vmax*0.85,vmax*0.75,r'$\overline{j_{\perp,'+sval+'}}  \overline{E}_{\perp} = $'+ "{:.2e}".format(JdotEperp),fontsize=12)
        else:
            axs[1].text(-vmax*0.85,vmax*0.75,r'$j_{\perp,'+sval+'}  E_{\perp} = $'+ "{:.2e}".format(JdotEperp),fontsize=12)

#     #set ticks
#     intvl = 1.
#     tcks = np.arange(0,vmax,intvl)
#     tcks = np.concatenate((-1*np.flip(tcks),tcks))
#     for _i in range(0,1):
#             axs[_i].set_xticks(tcks)
#             axs[_i].set_yticks(tcks)

    clrbar11 = plt.colorbar(im11, ax=axs[0],fraction=0.024, pad=0.04)
    if(not(plotLog)):
        clrbar11.formatter.set_powerlimits((0, 0))
    clrbar12 = plt.colorbar(im12, ax=axs[1],fraction=0.024, pad=0.04)#,format='%.1e')

    #clrbar11.ax.yaxis.set_label_coords(1.1, -0.1)
    #clrbar12.ax.yaxis.set_label_coords(1.1, -0.1)

    maxplotvval = np.max(vpar) #Assumes square grid of even size
    if(not(plotLog)):
        clrbar12.formatter.set_powerlimits((0, 0))
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight')
   
        clrbar11text = str(clrbar11.ax.yaxis.get_offset_text().get_text())
        clrbar11.ax.yaxis.get_offset_text().set_visible(False)
        axs[0].text(1.25*maxplotvval,-0.25*maxplotvval,clrbar11text, va='bottom', ha='center')
        clrbar12text = str(clrbar12.ax.yaxis.get_offset_text().get_text())
        clrbar12.ax.yaxis.get_offset_text().set_visible(False)
        axs[1].text(1.25*maxplotvval,-0.25*maxplotvval,clrbar12text, va='bottom', ha='center')

        #must make figure first to grab x10^val on top of color bar- after grabbing it, we can move it- a little wasteful but it was quick solution
        #clrbar11text = str(clrbar11.ax.yaxis.get_offset_text().get_text())
        #clrbar11.ax.yaxis.get_offset_text().set_visible(False)
        #clrbar11.ax.text(1.2, -1.25, clrbar11text, va='bottom', ha='center')
        #clrbar12text = str(clrbar12.ax.yaxis.get_offset_text().get_text())
        #clrbar12.ax.yaxis.get_offset_text().set_visible(False)
        #clrbar12.ax.text(1.2, -1.25, clrbar12text, va='bottom', ha='center')
        
        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight')
        plt.close('all') #saves RAM
    else:
        plt.show()
    plt.close()

def plot_gyro_single(vpar,vperp,coregyro,flnm='',coresubscript='tot',isIon=True,plotLog=False,computeJdotE=True,npar=None,plotAvg = False, plotFluc = False,isLowPass=False,isHighPass=False):
    """

    """
    if(npar != None):
        corepargyro /= npar
        coreperpgyro /= npar

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    fig, axs = plt.subplots(1,1,figsize=(2*5,4*5),sharey=True)

    vmax = np.max(vpar)

    if(isIon):
        vnormstr = 'v_{ti}'
    else:
        vnormstr = 'v_{te}'

    clboffset = np.array([vmax*.15,-vmax*.15])

    maxCe = np.max(coregyro)
    if(plotLog):
        im11 = axs.pcolormesh(vpar,vperp,coregyro, cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im11 = axs.pcolormesh(vpar,vperp,coregyro,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs.set_xlabel(r"$v_{||}/"+vnormstr+"$")
    axs.set_ylabel(r"$v_{\perp}/"+vnormstr+"$")
    axs.set_aspect('equal', 'box')
    axs.grid()
    if(plotFluc):
        if(isLowPass):
            axs.set_title(r'$\widetilde{C_{E_{'+coresubscript+'}}}^{k_{||} d_i < 15}(v_{||},v_{\perp})$', loc='right')
        elif(isHighPass):
            axs.set_title(r'$\widetilde{C_{E_{'+coresubscript+'}}}^{k_{||} d_i > 15}(v_{||},v_{\perp})$', loc='right')
        else:
            axs.set_title(r'$\widetilde{C_{E_{'+coresubscript+'}}}(v_{||},v_{\perp})$', loc='right')
    elif(plotAvg):
        axs.set_title(r'$\overline{C_{E_{'+coresubscript+'}}}(v_{||},v_{\perp})$', loc='right')
    else:
        axs.set_title(r'$C_{E_{'+coresubscript+'}}(v_{||},v_{\perp})$', loc='right')

    if(computeJdotE):
        JdotE = np.sum(coregyro)

        sval = 'e'
        if(isIon):
            sval = 'i'

            if(coresubscript=='tot'):
                if(plotFluc):
                    if(isLowPass):
                        axs.text(-vmax*0.85,vmax*0.75,r'$\widetilde{\mathbf{j}_{'+sval+'}}^{k_{||} d_i < 15}  \cdot \widetilde{\mathbf{E}}^{k_{||} d_i < 15} = $'+ "{:.2e}".format(JdotE),fontsize=24)
                    elif(isHighPass):
                        axs.text(-vmax*0.85,vmax*0.75,r'$\widetilde{\mathbf{j}_{'+sval+'}}^{k_{||} d_i < 15} \cdot \widetilde{\mathbf{E}}^{k_{||} d_i > 15} = $'+ "{:.2e}".format(JdotE),fontsize=24)
                    else:
                        axs.text(-vmax*0.85,vmax*0.75,r'$\widetilde{\mathbf{j}_{'+sval+'}} \cdot \widetilde{\mathbf{E}} = $'+ "{:.2e}".format(JdotE),fontsize=24)
                elif(plotAvg):
                    axs.text(-vmax*0.85,vmax*0.75,r'$\overline{\mathbf{j}_{'+sval+'}} \cdot \overline{\mathbf{E}} = $'+ "{:.2e}".format(JdotE),fontsize=24)
                else:
                    axs.text(-vmax*0.85,vmax*0.75,r'$\mathbf{j}_{'+sval+'} \cdot \mathbf{E} = $'+ "{:.2e}".format(JdotE),fontsize=12)
            else:
                if(plotFluc):
                    if(isLowPass):
                        axs.text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{'+coresubscript+','+sval+'}}^{k_{||} d_i < 15}  \widetilde{E}_{'+coresubscript+'}^{k_{||} d_i < 15} = $'+ "{:.2e}".format(JdotE),fontsize=24)
                    elif(isHighPass):
                        axs.text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{'+coresubscript+','+sval+'}}^{k_{||} d_i < 15}  \widetilde{E}_{'+coresubscript+'}^{k_{||} d_i > 15} = $'+ "{:.2e}".format(JdotE),fontsize=24)
                    else:
                        axs.text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{'+coresubscript+','+sval+'}}  \widetilde{E}_{'+coresubscript+'} = $'+ "{:.2e}".format(JdotE),fontsize=24)
                elif(plotAvg):
                    axs.text(-vmax*0.85,vmax*0.75,r'$\overline{j_{'+coresubscript+','+sval+'}}  \overline{E}_{'+coresubscript+'} = $'+ "{:.2e}".format(JdotE),fontsize=24)
                else:
                    axs.text(-vmax*0.85,vmax*0.75,r'$j_{'+coresubscript+','+sval+'}  E_{'+coresubscript+'} = $'+ "{:.2e}".format(JdotE),fontsize=24)

    # #set ticks
    # intvl = 1.
    # if(vmax > 5):
    #     intvl = 5.
    # if(vmax > 10):
    #     intvl = 10.
    # tcks = np.arange(0,vmax,intvl)
    # tcks = np.concatenate((-1*np.flip(tcks),tcks))
    # axs[0].set_xticks(tcks)
    # axs[0].set_yticks(tcks)

    clrbar11 = plt.colorbar(im11, ax=axs,fraction=0.024, pad=0.04)
    if(not(plotLog)):
        clrbar11.formatter.set_powerlimits((0, 0))

    maxplotvval = np.max(vpar) #Assumes square grid of even size

    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight') #for some reason this needs to be done to get the following lines to work...
   
        clrbar11text = str(clrbar11.ax.yaxis.get_offset_text().get_text())
        clrbar11.ax.yaxis.get_offset_text().set_visible(False)
        axs.text(1.25*maxplotvval,-0.175*maxplotvval,clrbar11text, va='bottom', ha='center')
        
        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight')
        plt.close('all') #saves RAM
    else:
        plt.show()
    plt.close()



def plot_gyro_3comp(vpar,vperp,corepargyro,coreperp1gyro,coreperp2gyro,flnm='',isIon=True,plotLog=False,computeJdotE=True,npar=None,plotAvg = False, plotFluc = False,isLowPass=False,isHighPass=False):
    """

    """
    if(npar != None):
        corepargyro /= npar
        coreperp1gyro /= npar
        coreperp2gyro /= npar

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    fig, axs = plt.subplots(1,3,figsize=(3*5,4*5),sharey=True)

    vmax = np.max(vpar)

    _hspace = .15
    _wspace = .3
    if(plotLog):
        _wspace+=.175
    fig.subplots_adjust(hspace=_hspace,wspace=_wspace)

    if(isIon):
        vnormstr = 'v_{ti}'
    else:
        vnormstr = 'v_{te}'

    clboffset = np.array([vmax*.15,-vmax*.15])

    maxCe = max(np.max(corepargyro),abs(np.max(corepargyro)))
    if(plotLog):
        im11 = axs[0].pcolormesh(vpar,vperp,corepargyro, cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im11 = axs[0].pcolormesh(vpar,vperp,corepargyro,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[0].set_xlabel(r"$v_{||}/"+vnormstr+"$")
    axs[0].set_ylabel(r"$v_{\perp}/"+vnormstr+"$")
    axs[0].set_aspect('equal', 'box')
    axs[0].grid()
    if(plotFluc):
        if(isLowPass):
            axs[0].set_title(r'$\widetilde{C_{E_{||}}}^{k_{||} d_i < 15}(v_{||},v_{\perp})$', loc='right')
        elif(isHighPass):
            axs[0].set_title(r'$\widetilde{C_{E_{||}}}^{k_{||} d_i > 15}(v_{||},v_{\perp})$', loc='right')
        else:
            axs[0].set_title(r'$\widetilde{C_{E_{||}}}(v_{||},v_{\perp})$', loc='right')
    elif(plotAvg):
        axs[0].set_title(r'$\overline{C_{E_{||}}}(v_{||},v_{\perp})$', loc='right')
    else:
        axs[0].set_title(r'$C_{E_{||}}(v_{||},v_{\perp})$', loc='right')

    maxCe = max(np.max(coreperp1gyro),abs(np.max(coreperp1gyro)))
    if(plotLog):
        im12 = axs[1].pcolormesh(vpar,vperp,coreperp1gyro,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im12 = axs[1].pcolormesh(vpar,vperp,coreperp1gyro,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[1].set_xlabel(r"$v_{||}/"+vnormstr+"$")
    axs[1].set_aspect('equal', 'box')
    axs[1].grid()
    if(plotFluc):
        if(isLowPass):
            axs[1].set_title(r'$\widetilde{C_{E_{\perp,1}}}^{k_{||} d_i < 15}(v_{||},v_{\perp})$', loc='right')
        elif(isHighPass):
            axs[1].set_title(r'$\widetilde{C_{E_{\perp,1}}}^{k_{||} d_i > 15}(v_{||},v_{\perp})$', loc='right')
        else:
            axs[1].set_title(r'$\widetilde{C_{E_{\perp,1}}}(v_{||},v_{\perp})$', loc='right')
    elif(plotAvg):
        axs[1].set_title(r'$\overline{C_{E_{\perp,1}}}(v_{||},v_{\perp})$', loc='right')
    else:
        axs[1].set_title(r'$C_{E_{\perp,1}}(v_{||},v_{\perp})$', loc='right')

    maxCe = max(np.max(coreperp2gyro),abs(np.max(coreperp2gyro)))
    if(plotLog):
        im13 = axs[2].pcolormesh(vpar,vperp,coreperp2gyro,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im13 = axs[2].pcolormesh(vpar,vperp,coreperp2gyro,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[2].set_xlabel(r"$v_{||}/"+vnormstr+"$")
    axs[2].set_aspect('equal', 'box')
    axs[2].grid()
    if(plotFluc):
        if(isLowPass):
            axs[2].set_title(r'$\widetilde{C_{E_{\perp,2}}}^{k_{||} d_i < 15}(v_{||},v_{\perp})$', loc='right')
        elif(isHighPass):
            axs[2].set_title(r'$\widetilde{C_{E_{\perp,2}}}^{k_{||} d_i > 15}(v_{||},v_{\perp})$', loc='right')
        else:
            axs[2].set_title(r'$\widetilde{C_{E_{\perp,2}}}(v_{||},v_{\perp})$', loc='right')
    elif(plotAvg):
        axs[2].set_title(r'$\overline{C_{E_{\perp,2}}}(v_{||},v_{\perp})$', loc='right')
    else:
        axs[2].set_title(r'$C_{E_{\perp,2}}(v_{||},v_{\perp})$', loc='right')

    axs[0].set_xlim(-vmax,vmax)
    axs[0].set_ylim(0,vmax)
    axs[1].set_xlim(-vmax,vmax)
    axs[1].set_ylim(0,vmax)
    axs[2].set_xlim(-vmax,vmax)
    axs[2].set_ylim(0,vmax)

    if(computeJdotE):
        JdotEpar = np.sum(corepargyro)
        JdotEperp1 = np.sum(coreperp1gyro)
        JdotEperp2 = np.sum(coreperp2gyro)

        sval = 'e'
        if(isIon):
            sval = 'i'

        if(plotFluc):
            if(isLowPass):
                axs[0].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{||,'+sval+'}}^{k_{||} d_i < 15}  \widetilde{E}_{||}^{k_{||} d_i < 15} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
            elif(isHighPass):
                axs[0].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{||,'+sval+'}}^{k_{||} d_i < 15}  \widetilde{E}_{||}^{k_{||} d_i > 15} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
            else:
                axs[0].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{||,'+sval+'}}  \widetilde{E}_{||} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
        elif(plotAvg):
            axs[0].text(-vmax*0.85,vmax*0.75,r'$\overline{j_{||,'+sval+'}}  \overline{E}_{||} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
        else:
            axs[0].text(-vmax*0.85,vmax*0.75,r'$j_{||,'+sval+'}  E_{||} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)

        if(plotFluc):
            if(isLowPass):
                axs[1].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,1,'+sval+'}}^{k_{||} d_i < 15}  \widetilde{E}_{\perp,1}^{k_{||} d_i < 15} = $'+ "{:.2e}".format(JdotEperp1),fontsize=12)
            elif(isHighPass):
                axs[1].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,1,'+sval+'}}^{k_{||} d_i < 15}  \widetilde{E}_{\perp,1}^{k_{||} d_i > 15} = $'+ "{:.2e}".format(JdotEperp1),fontsize=12)
            else:
                axs[1].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,1,'+sval+'}}  \widetilde{E}_{\perp,1} = $'+ "{:.2e}".format(JdotEperp1),fontsize=12)
        elif(plotAvg):
            axs[1].text(-vmax*0.85,vmax*0.75,r'$\overline{j_{\perp,1,'+sval+'}}  \overline{E}_{\perp,1} = $'+ "{:.2e}".format(JdotEperp1),fontsize=12)
        else:
            axs[1].text(-vmax*0.85,vmax*0.75,r'$j_{\perp,1,'+sval+'}  E_{\perp,1} = $'+ "{:.2e}".format(JdotEperp1),fontsize=12)

        if(plotFluc):
            if(isLowPass):
                axs[2].text(-vmax*0.85,vmax*0.7,r'$\widetilde{j_{\perp,2,'+sval+'}}^{k_{||} d_i < 15}  \widetilde{E}_{\perp,2}^{k_{||} d_i < 15} = $'+ "{:.2e}".format(JdotEperp2),fontsize=12)
            elif(isHighPass):
                axs[2].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,2,'+sval+'}}^{k_{||} d_i < 15}  \widetilde{E}_{\perp,2}^{k_{||} d_i > 15} = $'+ "{:.2e}".format(JdotEperp2),fontsize=12)
            else:
                axs[2].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,2,'+sval+'}}  \widetilde{E}_{\perp,2} = $'+ "{:.2e}".format(JdotEperp2),fontsize=12)
        elif(plotAvg):
            axs[2].text(-vmax*0.85,vmax*0.75,r'$\overline{j_{\perp,2,'+sval+'}}  \overline{E}_{\perp,2} = $'+ "{:.2e}".format(JdotEperp2),fontsize=12)
        else:
            axs[2].text(-vmax*0.85,vmax*0.75,r'$j_{\perp,2,'+sval+'}  E_{\perp,2} = $'+ "{:.2e}".format(JdotEperp2),fontsize=12)

#     #set ticks
#     intvl = 1.
#     tcks = np.arange(0,vmax,intvl)
#     tcks = np.concatenate((-1*np.flip(tcks),tcks))
#     for _i in range(0,1):
#             axs[_i].set_xticks(tcks)
#             axs[_i].set_yticks(tcks)


    clrbar11 = plt.colorbar(im11, ax=axs[0],fraction=0.024, pad=0.04)
    if(not(plotLog)):
        clrbar11.formatter.set_powerlimits((0, 0))
    clrbar12 = plt.colorbar(im12, ax=axs[1],fraction=0.024, pad=0.04)#,format='%.1e')
    clrbar13 = plt.colorbar(im13, ax=axs[2],fraction=0.024, pad=0.04)#,format='%.1e')
    #clrbar11.ax.yaxis.set_label_coords(1.1, -0.1)
    #clrbar12.ax.yaxis.set_label_coords(1.1, -0.1)

    maxplotvval = np.max(vpar) #Assumes square grid of even size
    if(not(plotLog)):
        clrbar12.formatter.set_powerlimits((0, 0))
    if(not(plotLog)):
        clrbar13.formatter.set_powerlimits((0, 0))
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight')

        clrbar11text = str(clrbar11.ax.yaxis.get_offset_text().get_text())
        clrbar11.ax.yaxis.get_offset_text().set_visible(False)
        axs[0].text(1.25*maxplotvval,-0.25*maxplotvval,clrbar11text, va='bottom', ha='center')
        clrbar12text = str(clrbar12.ax.yaxis.get_offset_text().get_text())
        clrbar12.ax.yaxis.get_offset_text().set_visible(False)
        axs[1].text(1.25*maxplotvval,-0.25*maxplotvval,clrbar12text, va='bottom', ha='center')
        clrbar13text = str(clrbar13.ax.yaxis.get_offset_text().get_text())
        clrbar13.ax.yaxis.get_offset_text().set_visible(False)
        axs[2].text(1.25*maxplotvval,-0.25*maxplotvval,clrbar13text, va='bottom', ha='center')

        #must make figure first to grab x10^val on top of color bar- after grabbing it, we can move it- a little wasteful but it was quick solution
        #clrbar11text = str(clrbar11.ax.yaxis.get_offset_text().get_text())
        #clrbar11.ax.yaxis.get_offset_text().set_visible(False)
        #clrbar11.ax.text(1.2, -1.25, clrbar11text, va='bottom', ha='center')
        #clrbar12text = str(clrbar12.ax.yaxis.get_offset_text().get_text())
        #clrbar12.ax.yaxis.get_offset_text().set_visible(False)
        #clrbar12.ax.text(1.2, -1.25, clrbar12text, va='bottom', ha='center')

        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight')
        plt.close('all') #saves RAM
    else:
        plt.show()
    plt.close()

def project_and_plot_supergrid(vx,vy,vz,vmax,hist,corex,corey,corez,flnm,title=None,plotFAC=False,plotAvg=False,plotFluc=False,isIon=True,isLowPass=False,isHighPass=False,vxmin=None,vxmax=None,vymin=None,vymax=None,vzmin=None,vzmax=None):
    from FPCAnalysis.array_ops import array_3d_to_2d

    H_xy = array_3d_to_2d(hist, 'xy')
    H_xz = array_3d_to_2d(hist, 'xz')
    H_yz = array_3d_to_2d(hist, 'yz')

    CEx_xy = array_3d_to_2d(corex, 'xy')
    CEx_xz = array_3d_to_2d(corex, 'xz')
    CEx_yz = array_3d_to_2d(corex, 'yz')

    CEy_xy = array_3d_to_2d(corey, 'xy')
    CEy_xz = array_3d_to_2d(corey, 'xz')
    CEy_yz = array_3d_to_2d(corey, 'yz')

    CEz_xy = array_3d_to_2d(corez, 'xy')
    CEz_xz = array_3d_to_2d(corez, 'xz')
    CEz_yz = array_3d_to_2d(corez, 'yz')

    plot_cor_and_dist_supergrid(vx, vy, vz, vmax,
                                H_xy, H_xz, H_yz,
                                CEx_xy,CEx_xz, CEx_yz,
                                CEy_xy,CEy_xz, CEy_yz,
                                CEz_xy,CEz_xz, CEz_yz,
                                flnm = flnm, ttl=title, computeJdotE = True, plotFAC = plotFAC, plotAvg = plotAvg, plotFluc = plotFluc, isIon = isIon, isLowPass=isLowPass,isHighPass=isHighPass,vxmin=vxmin,vxmax=vxmax,vymin=vymin,vymax=vymax,vzmin=vzmin,vzmax=vzmax)

def project_and_plot_supergrid_row(vx,vy,vz,vmax,arr,arrtype,flnm,plotFAC=False,plotAvg=False,plotFluc=False,isIon=True,isLowPass=False,isHighPass=False):
    
    from FPCAnalysis.array_ops import array_3d_to_2d

    arr_xy = array_3d_to_2d(arr, 'xy')
    arr_xz = array_3d_to_2d(arr, 'xz')
    arr_yz = array_3d_to_2d(arr, 'yz')

    plot_cor_and_dist_supergrid_row(vx, vy, vz, vmax,
                                arr_xy,arr_xz, arr_yz,
                                arrtype,
                                flnm = flnm, ttl = '', computeJdotE = True, params = None, metadata = None, xpos = None, plotLog = False, plotLogHist = True,
                                plotFAC = plotFAC, plotFluc = plotFluc, plotAvg = plotAvg, isIon = isIon, listpos=False,xposval=None,normtoN = False,Nval = None, isLowPass=isLowPass,isHighPass=isHighPass,plotDiagJEOnly=True)
    
def plot_phaseposvsvx(dparticles,poskey,velkey,xmin,xmax,dx,vmax,dv,cbarmax=None,flnm=''):
    velbins = np.arange(-vmax, vmax+dv, dv)
    velbins = (velbins[1:] + velbins[:-1])/2.
    posbins = np.arange(xmin, xmax+dx, dx)
    posbins = (posbins[1:] + posbins[:-1])/2.

    phaseplothist,_ = np.histogramdd((dparticles[poskey], dparticles[velkey]), bins=[posbins, velbins])

    velbincenters = np.asarray([(velbins[_idx]+velbins[_idx+1])/2. for _idx in range(len(velbins)-1)])
    posbincenters = np.asarray([(posbins[_idx]+posbins[_idx+1])/2. for _idx in range(len(posbins)-1)])
    
    fig = plt.figure(figsize=(12,3.420))
    if(cbarmax != None):
        im = plt.pcolormesh(posbincenters,velbincenters,phaseplothist.T,shading='gouraud',cmap="plasma",vmax=cbarmax)
        fig.colorbar(im,extend='max')
    else:
        im = plt.pcolormesh(posbincenters,velbincenters,phaseplothist.T,shading='gouraud',cmap="plasma")
        fig.colorbar(im)

    plt.grid()
    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
    if(poskey == 'x1'):
        plt.xlabel(r"$x / d_{i,0}$", size=24)
    elif(poskey == 'x2'):
        plt.xlabel(r"$y / d_{i,0}$", size=24)
    elif(poskey == 'x2'):
        plt.xlabel(r"$z / d_{i,0}$", size=24)
    else:
        plt.xlabel(poskey, size=24)

    if(velkey == 'p1'):
        plt.ylabel(r"$v_x/v_{ti}$", size=24)
    elif(velkey == 'p2'):
        plt.ylabel(r"$v_y/v_{ti}$", size=24)
    elif(velkey == 'p3'):
        plt.ylabel(r"$v_z/v_{ti}$", size=24)
        
    if(flnm == ''):
        plt.show()
    else:
        plt.savefig('histxxvx.png',format='png',dpi=300,facecolor='white', transparent=False,bbox_inches='tight')
    plt.close()
