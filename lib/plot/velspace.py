# velsig.py>

# functions related to velocity signatures and 2d distribution functions

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_velsig(vx,vy,vmax,Ce,fieldkey,flnm = '',ttl=''):
    """
    Plots correlation data from make2dHistandCex,make2dHistandCey,etc

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

def plot_cor_and_dist_supergrid(vx, vy, vz, vmax,
                                H_xy, H_xz, H_yz,
                                CEx_xy,CEx_xz, CEx_yz,
                                CEy_xy,CEy_xz, CEy_yz,
                                CEz_xy,CEz_xz, CEz_yz,
                                flnm = '', ttl = ''):
    """

    """
    from lib.array_ops import mesh_3d_to_2d

    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots

    fig, axs = plt.subplots(4,3,figsize=(4*5,3*5))

    vx_xy, vy_xy = mesh_3d_to_2d(vx,vy,vz,'xy')
    vx_xz, vz_xz = mesh_3d_to_2d(vx,vy,vz,'xz')
    vy_yz, vz_yz = mesh_3d_to_2d(vx,vy,vz,'yz')

    fig.suptitle(ttl)

    #H_xy
    im00= axs[0,0].pcolormesh(vy_xy, vx_xy, H_xy, cmap="plasma", shading="gouraud")
    axs[0,0].set_title(r"$f(v_x, v_y)$")
    axs[0,0].set_xlabel(r"$v_x/v_{ti}$")
    axs[0,0].set_ylabel(r"$v_y/v_{ti}$")
    axs[0,0].set_aspect('equal', 'box')
    plt.colorbar(im00, ax=axs[0,0])
    #H_xz
    im01 = axs[0,1].pcolormesh(vz_xz, vx_xz, H_xz, cmap="plasma", shading="gouraud")
    axs[0,1].set_title(r"$f(v_x, v_z)$")
    axs[0,1].set_xlabel(r"$v_x/v_{ti}$")
    axs[0,1].set_ylabel(r"$v_z/v_{ti}$")
    axs[0,1].set_aspect('equal', 'box')
    plt.colorbar(im01, ax=axs[0,1])
    #H_yz
    im02 = axs[0,2].pcolormesh(vz_yz, vy_yz, H_yz.T, cmap="plasma", shading="gouraud")
    axs[0,2].set_title(r"$f(v_y, v_z)$")
    axs[0,2].set_ylabel(r"$v_y/v_{ti}$")
    axs[0,2].set_xlabel(r"$v_z/v_{ti}$")
    axs[0,2].set_aspect('equal', 'box')
    plt.colorbar(im02, ax=axs[0,2])
    #CEx_xy
    maxCe = max(np.max(CEx_xy),abs(np.max(CEx_xy)))
    im10 = axs[1,0].pcolormesh(vy_xy,vx_xy,CEx_xy,vmax=maxCe,vmin=-maxCe,cmap="seismic", shading="gouraud")
    axs[1,0].set_title('$C_{Ex}(v_x,v_y)$')
    axs[1,0].set_xlabel(r"$v_x/v_{ti}$")
    axs[1,0].set_ylabel(r"$v_y/v_{ti}$")
    axs[1,0].set_aspect('equal', 'box')
    plt.colorbar(im10, ax=axs[1,0])
    #CEx_xz
    maxCe = max(np.max(CEx_xz),abs(np.max(CEx_xz)))
    im11 = axs[1,1].pcolormesh(vz_xz,vx_xz,CEx_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[1,1].set_title('$C_{Ex}(v_x,v_z)$')
    axs[1,1].set_xlabel(r"$v_x/v_{ti}$")
    axs[1,1].set_ylabel(r"$v_z/v_{ti}$")
    axs[1,1].set_aspect('equal', 'box')
    plt.colorbar(im11, ax=axs[1,1])
    #CEx_yz
    maxCe = max(np.max(CEx_yz),abs(np.max(CEx_yz)))
    im12 = axs[1,2].pcolormesh(vz_yz,vy_yz,CEx_yz.T,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[1,2].set_title('$C_{Ex}(v_y,v_z)$')
    axs[1,2].set_ylabel(r"$v_y/v_{ti}$")
    axs[1,2].set_xlabel(r"$v_z/v_{ti}$")
    axs[1,2].set_aspect('equal', 'box')
    plt.colorbar(im12, ax=axs[1,2])
    #CEy_xy
    maxCe = max(np.max(CEy_xy),abs(np.max(CEy_xy)))
    im20 = axs[2,0].pcolormesh(vy_xy,vx_xy,CEy_xy,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[2,0].set_title('$C_{Ey}(v_x,v_y)$')
    axs[2,0].set_xlabel(r"$v_x/v_{ti}$")
    axs[2,0].set_ylabel(r"$v_y/v_{ti}$")
    axs[2,0].set_aspect('equal', 'box')
    plt.colorbar(im20, ax=axs[2,0])
    #CEy_xz
    maxCe = max(np.max(CEy_xz),abs(np.max(CEy_xz)))
    im21 = axs[2,1].pcolormesh(vz_xz,vx_xz,CEy_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[2,1].set_title('$C_{Ey}(v_x,v_z)$')
    axs[2,1].set_xlabel(r"$v_x/v_{ti}$")
    axs[2,1].set_ylabel(r"$v_z/v_{ti}$")
    axs[2,1].set_aspect('equal', 'box')
    plt.colorbar(im21, ax=axs[2,1])
    #CEy_yz
    maxCe = max(np.max(CEy_yz),abs(np.max(CEy_yz)))
    im22 = axs[2,2].pcolormesh(vz_yz,vy_yz,CEy_yz.T,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[2,2].set_title('$C_{Ey}(v_y,v_z)$')
    axs[2,2].set_ylabel(r"$v_y/v_{ti}$")
    axs[2,2].set_xlabel(r"$v_z/v_{ti}$")
    axs[2,2].set_aspect('equal', 'box')
    plt.colorbar(im22, ax=axs[2,2])
    #CEz_xy
    maxCe = max(np.max(CEz_xy),abs(np.max(CEz_xy)))
    im30 = axs[3,0].pcolormesh(vy_xy,vx_xy,CEz_xy,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[3,0].set_title('$C_{Ez}(v_x,v_y)$')
    axs[3,0].set_xlabel(r"$v_x/v_{ti}$")
    axs[3,0].set_ylabel(r"$v_y/v_{ti}$")
    axs[3,0].set_aspect('equal', 'box')
    plt.colorbar(im30, ax=axs[3,0])
    #CEz_xz
    maxCe = max(np.max(CEz_xz),abs(np.max(CEz_xz)))
    im31 = axs[3,1].pcolormesh(vz_xz,vx_xz,CEz_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[3,1].set_title('$C_{Ez}(v_x,v_z)$')
    axs[3,1].set_xlabel(r"$v_x/v_{ti}$")
    axs[3,1].set_ylabel(r"$v_z/v_{ti}$")
    axs[3,1].set_aspect('equal', 'box')
    plt.colorbar(im31, ax=axs[3,1])
    #CEz_yz
    maxCe = max(np.max(CEz_yz),abs(np.max(CEz_yz)))
    im32 = axs[3,2].pcolormesh(vz_yz,vy_yz,CEz_yz.T,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[3,2].set_title('$C_{Ez}(v_y,v_z)$')
    axs[3,2].set_ylabel(r"$v_y/v_{ti}$")
    axs[3,2].set_xlabel(r"$v_z/v_{ti}$")
    axs[3,2].set_aspect('equal', 'box')
    plt.colorbar(im32, ax=axs[3,2])

    plt.subplots_adjust(wspace=-.3,hspace=1.25)
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png')
        plt.close('all') #saves RAM
    else:
        plt.show()
    plt.close()

def make_superplot_gif(vx, vy, vz, vmax, Hist, CEx, CEy, CEz, x, directory, flnm):
    #make plots of data and put into directory

    from lib.array_ops import array_3d_to_2d
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
