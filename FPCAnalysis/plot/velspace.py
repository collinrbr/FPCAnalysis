# velsig.py>

# functions related to velocity signatures and 2d distribution functions

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_velsig(vx,vy,vz,dv,vmax,CEiproj,fieldkey,planename,ttl=r'$C_{E_i}(v_i,v_j)$; ',flnm='',xlabel=r"$v_i/v_{ti}$",ylabel=r"$v_j/v_{ti}$",plotLog=False,computeJdotE=True,axvlinex = None, maxCe = None):

    from lib.array_ops import mesh_3d_to_2d
    import matplotlib
    from matplotlib.colors import LogNorm
    from lib.array_ops import array_3d_to_2d
    import matplotlib.colors as colors
    from lib.analysis import compute_energization


    CEiplot = CEiproj
    yplot, xplot = mesh_3d_to_2d(vx,vy,vz,planename)

    plt.style.use('postgkyl.mplstyle')

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
            plt.gca().set_title(ttl+'$J \cdot E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
        elif(fieldkey == 'ey'):
            plt.gca().set_title(ttl+'$J \cdot E_y$ = ' + "{:.2e}".format(JdotE),loc='left')
        elif(fieldkey == 'ez'):
            plt.gca().set_title(ttl+'$J \cdot E_z$ = ' + "{:.2e}".format(JdotE),loc='left')
        else:
            plt.gca().set_title(ttl+'$J \cdot E_i$ = ' + "{:.2e}".format(JdotE),loc='left')
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
    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots

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

    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots

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
    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots

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

    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots
    from lib.array_ops import mesh_3d_to_2d
    import matplotlib
    from matplotlib.colors import LogNorm
    from lib.array_ops import array_3d_to_2d
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

    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots
    from lib.array_ops import mesh_3d_to_2d
    import matplotlib
    from matplotlib.colors import LogNorm
    from lib.array_ops import array_3d_to_2d
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
        pcm2 = axs[2].pcolormesh(vz_yz, vy_yz, H_yz, cmap='PiYG', shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-1*_vmax, vmax=_vmax))
    else:
        axs[2].set_facecolor(bkgcolor)
        pcm2 = axs[2].pcolormesh(vz_yz,vy_yz,H_yz, cmap=cmap, shading="gouraud",norm=LogNorm(vmin=minval, vmax=maxval))
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
                                flnm = '', ttl = '', computeJdotE = False, params = None, metadata = None, xpos = None, plotLog = False):
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
    from lib.array_ops import mesh_3d_to_2d
    from lib.analysis import compute_energization
    from matplotlib.colors import LogNorm
    import matplotlib.colors as colors

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    fig, axs = plt.subplots(4,3,figsize=(4*5,3*5),sharex=True)

    _hspace = .1
    _wspace = -.35
    if(computeJdotE):
        _hspace+=.1
    if(plotLog):
        _wspace+=.175
    fig.subplots_adjust(hspace=_hspace,wspace=_wspace)

    vx_xy, vy_xy = mesh_3d_to_2d(vx,vy,vz,'xy')
    vx_xz, vz_xz = mesh_3d_to_2d(vx,vy,vz,'xz')
    vy_yz, vz_yz = mesh_3d_to_2d(vx,vy,vz,'yz')

    dv = vy_yz[1][1]-vy_yz[0][0] #assumes square velocity grid

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

    minHxyval = np.min(H_xy[np.nonzero(H_xy)])
    minHxzval = np.min(H_xz[np.nonzero(H_xz)])
    minHyzval = np.min(H_yz[np.nonzero(H_yz)])

    #H_xy
    if(plotLog):
        im00= axs[0,0].pcolormesh(vy_xy, vx_xy, H_xy, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHxyval, vmax=H_xy.max()))
    else:
        im00= axs[0,0].pcolormesh(vy_xy, vx_xy, H_xy, cmap="plasma", shading="gouraud")
    #axs[0,0].set_title(r"$f(v_x, v_y)$")
    axs[0,0].set_ylabel(r"$v_y/v_{ti}$")
    axs[0,0].set_aspect('equal', 'box')
    axs[0,0].grid()
    clrbar00 = plt.colorbar(im00, ax=axs[0,0])#,format='%.1e')
    if(not(plotLog)):
        clrbar00.formatter.set_powerlimits((0, 0))
    axs[0,0].text(-vmax*2.0,0, r"$f$", ha='center', rotation=90, wrap=False)
    if(params != None):
        axs[0,0].text(-vmax*2.6,0, '$M_A = $ ' + str(abs(params['MachAlfven'])), ha='center', rotation=90, wrap=False)

    #H_xz
    if(plotLog):
        im01 = axs[0,1].pcolormesh(vz_xz, vx_xz, H_xz, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHxzval, vmax=H_xz.max()))
    else:
        im01 = axs[0,1].pcolormesh(vz_xz, vx_xz, H_xz, cmap="plasma", shading="gouraud")
    #axs[0,1].set_title(r"$f(v_x, v_z)$")
    axs[0,1].set_ylabel(r"$v_z/v_{ti}$")
    axs[0,1].set_aspect('equal', 'box')
    axs[0,1].grid()
    clrbar01 = plt.colorbar(im01, ax=axs[0,1])#,format='%.1e')
    if(not(plotLog)):
        clrbar01.formatter.set_powerlimits((0, 0))

    #H_yz
    if(plotLog):
        im02 = axs[0,2].pcolormesh(vz_yz, vy_yz, H_yz.T, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHyzval, vmax=H_yz.max()))
    else:
        im02 = axs[0,2].pcolormesh(vz_yz, vy_yz, H_yz.T, cmap="plasma", shading="gouraud")
    #axs[0,2].set_title(r"$f(v_y, v_z)$")
    axs[0,2].set_ylabel(r"$v_y/v_{ti}$")
    axs[0,2].set_aspect('equal', 'box')
    axs[0,2].grid()
    clrbar02 = plt.colorbar(im02, ax=axs[0,2])#,format='%.1e')
    if(not(plotLog)):
        clrbar02.formatter.set_powerlimits((0, 0))

    #CEx_xy
    maxCe = max(np.max(CEx_xy),abs(np.max(CEx_xy)))
    if(plotLog):
        im10 = axs[1,0].pcolormesh(vy_xy,vx_xy,CEx_xy,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im10 = axs[1,0].pcolormesh(vy_xy,vx_xy,CEx_xy,vmax=maxCe,vmin=-maxCe,cmap="seismic", shading="gouraud")
    #axs[1,0].set_title('$C_{Ex}(v_x,v_y)$')
    axs[1,0].set_ylabel(r"$v_y/v_{ti}$")
    axs[1,0].set_aspect('equal', 'box')
    axs[1,0].grid()
    axs[1,0].text(-vmax*2.0,0, r"$C_{Ex}$", ha='center', rotation=90, wrap=False)
    if(params != None):
        axs[1,0].text(-vmax*2.6,0, '$\Theta_{Bn} = $ ' + str(params['thetaBn']), ha='center', rotation=90, wrap=False)
    if(computeJdotE):
        JdotE = compute_energization(CEx_xy,dv)
        axs[1,0].set_title('$J \cdot E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
    clrbar10 = plt.colorbar(im10, ax=axs[1,0])#,format='%.1e')
    if(not(plotLog)):
        clrbar10.formatter.set_powerlimits((0, 0))

    #CEx_xz
    maxCe = max(np.max(CEx_xz),abs(np.max(CEx_xz)))
    if(plotLog):
        im11 = axs[1,1].pcolormesh(vz_xz,vx_xz,CEx_xz, cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im11 = axs[1,1].pcolormesh(vz_xz,vx_xz,CEx_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[1,1].set_title('$C_{Ex}(v_x,v_z)$')
    axs[1,1].set_ylabel(r"$v_z/v_{ti}$")
    axs[1,1].set_aspect('equal', 'box')
    axs[1,1].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEx_xz,dv)
        axs[1,1].set_title('$J \cdot E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
    clrbar11 = plt.colorbar(im11, ax=axs[1,1])#,format='%.1e')
    if(not(plotLog)):
        clrbar11.formatter.set_powerlimits((0, 0))

    #CEx_yz
    maxCe = max(np.max(CEx_yz),abs(np.max(CEx_yz)))
    if(plotLog):
        im12 = axs[1,2].pcolormesh(vz_yz,vy_yz,CEx_yz.T,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im12 = axs[1,2].pcolormesh(vz_yz,vy_yz,CEx_yz.T,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[1,2].set_title('$C_{Ex}(v_y,v_z)$')
    axs[1,2].set_ylabel(r"$v_y/v_{ti}$")
    axs[1,2].set_aspect('equal', 'box')
    axs[1,2].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEx_yz.T,dv)
        axs[1,2].set_title('$J \cdot E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
    clrbar12 = plt.colorbar(im12, ax=axs[1,2])#,format='%.1e')
    if(not(plotLog)):
        clrbar12.formatter.set_powerlimits((0, 0))

    #CEy_xy
    maxCe = max(np.max(CEy_xy),abs(np.max(CEy_xy)))
    if(plotLog):
        im20 = axs[2,0].pcolormesh(vy_xy,vx_xy,CEy_xy,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im20 = axs[2,0].pcolormesh(vy_xy,vx_xy,CEy_xy,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[2,0].set_title('$C_{Ey}(v_x,v_y)$')
    axs[2,0].set_ylabel(r"$v_y/v_{ti}$")
    axs[2,0].set_aspect('equal', 'box')
    axs[2,0].text(-vmax*2.0,0, r"$C_{Ey}$", ha='center', rotation=90, wrap=False)
    axs[2,0].grid()
    if(xpos != None):
        axs[2,0].text(-vmax*2.6,0,'$x/d_i = $' + str(xpos), ha='center', rotation=90, wrap=False)
    if(computeJdotE):
        JdotE = compute_energization(CEy_xy,dv)
        axs[2,0].set_title('$J \cdot E_y$ = ' + "{:.2e}".format(JdotE),loc='left')
    clrbar20 = plt.colorbar(im20, ax=axs[2,0])#,format='%.1e')
    if(not(plotLog)):
        clrbar20.formatter.set_powerlimits((0, 0))

    #CEy_xz
    maxCe = max(np.max(CEy_xz),abs(np.max(CEy_xz)))
    if(plotLog):
        im21 = axs[2,1].pcolormesh(vz_xz,vx_xz,CEy_xz,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im21 = axs[2,1].pcolormesh(vz_xz,vx_xz,CEy_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[2,1].set_title('$C_{Ey}(v_x,v_z)$')
    axs[2,1].set_ylabel(r"$v_z/v_{ti}$")
    axs[2,1].set_aspect('equal', 'box')
    axs[2,1].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEy_xz,dv)
        axs[2,1].set_title('$J \cdot E_y$ = ' + "{:.2e}".format(JdotE),loc='left')
    clrbar21 = plt.colorbar(im21, ax=axs[2,1])#,format='%.1e')
    if(not(plotLog)):
        clrbar21.formatter.set_powerlimits((0, 0))

    #CEy_yz
    maxCe = max(np.max(CEy_yz),abs(np.max(CEy_yz)))
    if(plotLog):
        im22 = axs[2,2].pcolormesh(vz_yz,vy_yz,CEy_yz.T, cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im22 = axs[2,2].pcolormesh(vz_yz,vy_yz,CEy_yz.T,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[2,2].set_title('$C_{Ey}(v_y,v_z)$')
    axs[2,2].set_ylabel(r"$v_y/v_{ti}$")
    axs[2,2].set_aspect('equal', 'box')
    axs[2,2].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEy_yz.T,dv)
        axs[2,2].set_title('$J \cdot E_y$ = ' + "{:.2e}".format(JdotE),loc='left')
    clrbar22 = plt.colorbar(im22, ax=axs[2,2])#,format='%.1e')
    if(not(plotLog)):
        clrbar22.formatter.set_powerlimits((0, 0))

    #CEz_xy
    maxCe = max(np.max(CEz_xy),abs(np.max(CEz_xy)))
    if(plotLog):
        im30 = axs[3,0].pcolormesh(vy_xy,vx_xy,CEz_xy,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im30 = axs[3,0].pcolormesh(vy_xy,vx_xy,CEz_xy,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[3,0].set_title('$C_{Ez}(v_x,v_y)$')
    axs[3,0].set_xlabel(r"$v_x/v_{ti}$")
    axs[3,0].set_ylabel(r"$v_y/v_{ti}$")
    axs[3,0].set_aspect('equal', 'box')
    axs[3,0].text(-vmax*2.0,0, r"$C_{Ez}$", ha='center', rotation=90, wrap=False)
    axs[3,0].grid()
    if(metadata != None):
        axs[3,0].text(-vmax*2.6,0, metadata, ha='center', rotation=90, wrap=False)
    if(computeJdotE):
        JdotE = compute_energization(CEz_xy,dv)
        axs[3,0].set_title('$J \cdot E_z$ = ' + "{:.2e}".format(JdotE),loc='left')
    clrbar30 = plt.colorbar(im30, ax=axs[3,0])#,format='%.1e')
    if(not(plotLog)):
        clrbar30.formatter.set_powerlimits((0, 0))

    #CEz_xz
    maxCe = max(np.max(CEz_xz),abs(np.max(CEz_xz)))
    if(plotLog):
        im31 = axs[3,1].pcolormesh(vz_xz,vx_xz,CEz_xz,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im31 = axs[3,1].pcolormesh(vz_xz,vx_xz,CEz_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[3,1].set_title('$C_{Ez}(v_x,v_z)$')
    axs[3,1].set_xlabel(r"$v_x/v_{ti}$")
    axs[3,1].set_ylabel(r"$v_z/v_{ti}$")
    axs[3,1].set_aspect('equal', 'box')
    axs[3,1].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEz_xz,dv)
        axs[3,1].set_title('$J \cdot E_z$ = ' + "{:.2e}".format(JdotE),loc='left')
    clrbar31 = plt.colorbar(im31, ax=axs[3,1])#,format='%.1e')
    if(not(plotLog)):
        clrbar31.formatter.set_powerlimits((0, 0))

    #CEz_yz
    maxCe = max(np.max(CEz_yz),abs(np.max(CEz_yz)))
    if(plotLog):
        im32 = axs[3,2].pcolormesh(vz_yz,vy_yz,CEz_yz.T,cmap="seismic", shading="gouraud",norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=-maxCe, vmax=maxCe))
    else:
        im32 = axs[3,2].pcolormesh(vz_yz,vy_yz,CEz_yz.T,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    #axs[3,2].set_title('$C_{Ez}(v_y,v_z)$')
    axs[3,2].set_ylabel(r"$v_y/v_{ti}$")
    axs[3,2].set_xlabel(r"$v_z/v_{ti}$")
    axs[3,2].set_aspect('equal', 'box')
    axs[3,2].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEz_yz.T,dv)
        axs[3,2].set_title('$J \cdot E_z$ = ' + "{:.2e}".format(JdotE),loc='left')
    clrbar32 = plt.colorbar(im32, ax=axs[3,2])#,format='%.1e')
    if(not(plotLog)):
        clrbar32.formatter.set_powerlimits((0, 0))

    for _i in range(0,4):
        for _j in range(0,3):
            axs[_i,_j].set_xlim(-vmax,vmax)
            axs[_i,_j].set_ylim(-vmax,vmax)

    #set ticks
    intvl = 5.
    tcks = np.arange(0,vmax,intvl)
    tcks = np.concatenate((-1*np.flip(tcks),tcks))
    for _i in range(0,4):
        for _j in range(0,3):
            axs[_i,_j].set_xticks(tcks)
            axs[_i,_j].set_yticks(tcks)

    #plt.subplots_adjust(hspace=.5,wspace=-.3)
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight')
        plt.close('all') #saves RAM
    else:
        plt.show()
    plt.close()

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
        #try: #sometimes netcdf4 will contain bin with no particles, which messes up plotting routine. For now we skip those bins
        plot_cor_and_dist_supergrid(vx, vy, vz, vmax,
                                    Hist_vxvy[i], Hist_vxvz[i], Hist_vyvz[i],
                                    C_Ex_vxvy[i], C_Ex_vxvz[i], C_Ex_vyvz[i],
                                    C_Ey_vxvy[i], C_Ey_vxvz[i], C_Ey_vyvz[i],
                                    C_Ez_vxvy[i], C_Ez_vxvz[i], C_Ez_vyvz[i],
                                    flnm = directory+str(i).zfill(6), computeJdotE = False, params = params_in, metadata = mdt, xpos = x[i], plotLog=plotLog)
        # except:
        #     print("Failed to make plot for this slice!!")
        #     print("npar:", np.sum(Hist_vxvy))

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

    from lib.array_ops import array_3d_to_2d

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
    from lib.fpc import compute_hist_and_cor

    v1, v2, v3, totalPtcl, totalFieldpts, hist, cor = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2,
                                                                            dpar, dfields, vshock, fieldkey, directionkey)
    del totalFieldpts #this needs to be removed from the code as it is not used anymore


    print("npar in box: ",str(np.sum(hist)))

    #makes plot
    from lib.array_ops import array_3d_to_2d
    CEiproj = array_3d_to_2d(cor, planename)

    if(flnm != ''):
        print("Requesting that file is saved as",flnm,"!")

    plot_velsig(v1,v2,v3,dv,vmax,CEiproj,fieldkey,planename,ttl=ttl,xlabel=xlbl,ylabel=ylbl,flnm=flnm)