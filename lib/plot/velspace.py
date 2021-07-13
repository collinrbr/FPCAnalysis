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
    plotv1 = vx
    plotv2 = vy

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
    plotv1 = vx
    plotv2 = vy

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


def make_velsig_gif_with_EcrossB(vx, vy, vmax, C, fieldkey, x_out, dx, dfields, directory, flnm):
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
    """

    from lib.sanityfunctions import calc_E_crossB

    #make plots of data and put into directory
    try:
        os.mkdir(directory)
    except:
        pass
    print("Warning: This function uses fields near the x axis. This should suffice for a quick sanity check, but please check averaging box bounds in this function if needed.")
    xsweep = 0.
    for i in range(0,len(C)):
        print('Making plot ' + str(i)+' of '+str(len(C)))
        flnm = directory+'/'+str(i).zfill(6)
        ExBvx, ExBvy, _ = calc_E_crossB(dfields,xsweep,xsweep+dx,dfields[fieldkey+'_yy'][0],dfields[fieldkey+'_yy'][1],dfields[fieldkey+'_zz'][0],dfields[fieldkey+'_zz'][1])
        plot_velsig_wEcrossB(vx,vy,vmax,C[i],ExBvx,ExBvy,fieldkey,flnm = flnm,ttl='x(di): '+str(x_out[i]))
        xsweep += dx
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
    axs[0,0].pcolormesh(vy_xy, vx_xy, H_xy, cmap="plasma", shading="gouraud")
    axs[0,0].set_title(r"$f(v_x, v_y)$ ")
    axs[0,0].set_xlabel(r"$v_x/v_{ti}$")
    axs[0,0].set_ylabel(r"$v_y/v_{ti}$")
    #H_xz
    axs[0,1].pcolormesh(vz_xz, vx_xz, H_xz, cmap="plasma", shading="gouraud")
    axs[0,1].set_title(r"$f(v_x, v_z)$")
    axs[0,1].set_xlabel(r"$v_x/v_{ti}$")
    axs[0,1].set_ylabel(r"$v_z/v_{ti}$")
    #H_yz
    axs[0,2].pcolormesh(vz_yz, vy_yz, H_yz, cmap="plasma", shading="gouraud")
    axs[0,2].set_title(r"$f(v_y, v_z)$")
    axs[0,2].set_xlabel(r"$v_y/v_{ti}$")
    axs[0,2].set_ylabel(r"$v_z/v_{ti}$")
    #CEx_xy
    maxCe = max(np.max(CEx_xy),abs(np.max(CEx_xy)))
    axs[1,0].pcolormesh(vy_xy,vx_xy,CEx_xy,vmax=maxCe,vmin=-maxCe,cmap="seismic", shading="gouraud")
    axs[1,0].set_title('CEx')
    axs[1,0].set_xlabel(r"$v_x/v_{ti}$")
    axs[1,0].set_ylabel(r"$v_y/v_{ti}$")
    #CEx_xz
    maxCe = max(np.max(CEx_xz),abs(np.max(CEx_xz)))
    axs[1,1].pcolormesh(vz_xz,vx_xz,CEx_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[1,1].set_title('CEx')
    axs[1,1].set_xlabel(r"$v_x/v_{ti}$")
    axs[1,1].set_ylabel(r"$v_z/v_{ti}$")
    #CEx_yz
    maxCe = max(np.max(CEx_yz),abs(np.max(CEx_yz)))
    axs[1,2].pcolormesh(vz_yz,vy_yz,CEx_yz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[1,2].set_title('CEx')
    axs[1,2].set_xlabel(r"$v_y/v_{ti}$")
    axs[1,2].set_ylabel(r"$v_z/v_{ti}$")
    #CEy_xy
    maxCe = max(np.max(CEy_xy),abs(np.max(CEy_xy)))
    axs[2,0].pcolormesh(vy_xy,vx_xy,CEy_xy,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[2,0].set_title('CEy')
    axs[2,0].set_xlabel(r"$v_x/v_{ti}$")
    axs[2,0].set_ylabel(r"$v_y/v_{ti}$")
    #CEy_xz
    maxCe = max(np.max(CEy_xz),abs(np.max(CEy_xz)))
    axs[2,1].pcolormesh(vz_xz,vx_xz,CEy_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[2,1].set_title('CEy')
    axs[2,1].set_xlabel(r"$v_x/v_{ti}$")
    axs[2,1].set_ylabel(r"$v_z/v_{ti}$")
    #CEy_yz
    maxCe = max(np.max(CEy_yz),abs(np.max(CEy_yz)))
    axs[2,2].pcolormesh(vz_yz,vy_yz,CEy_yz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[2,2].set_title('CEy')
    axs[2,2].set_xlabel(r"$v_y/v_{ti}$")
    axs[2,2].set_ylabel(r"$v_z/v_{ti}$")
    #CEz_xy
    maxCe = max(np.max(CEz_xy),abs(np.max(CEz_xy)))
    axs[3,0].pcolormesh(vy_xy,vx_xy,CEz_xy,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[3,0].set_title('CEz')
    axs[3,0].set_xlabel(r"$v_x/v_{ti}$")
    axs[3,0].set_ylabel(r"$v_y/v_{ti}$")
    #CEz_xz
    maxCe = max(np.max(CEz_xz),abs(np.max(CEz_xz)))
    axs[3,1].pcolormesh(vz_xz,vx_xz,CEz_xz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[3,1].set_title('CEz')
    axs[3,1].set_xlabel(r"$v_x/v_{ti}$")
    axs[3,1].set_ylabel(r"$v_z/v_{ti}$")
    #CEz_yz
    maxCe = max(np.max(CEz_yz),abs(np.max(CEz_yz)))
    axs[3,2].pcolormesh(vz_yz,vy_yz,CEz_yz,vmax=maxCe,vmin=-maxCe, cmap="seismic", shading="gouraud")
    axs[3,2].set_title('CEz')
    axs[3,2].set_xlabel(r"$v_y/v_{ti}$")
    axs[3,2].set_ylabel(r"$v_z/v_{ti}$")

    plt.subplots_adjust(wspace=.5,hspace=.5)
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