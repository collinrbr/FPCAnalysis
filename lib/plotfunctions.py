# plotfunctions.py>

# Here we have functions related to plotting dHybridR data

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_field(dfields, fieldkey, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0, axvx1 = float('nan'), axvx2 = float('nan'), flnm = ''):
    """
    Plots field data at some static frame down a line along x,y,z for some
    selected field.

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    axis : str, optional
        name of axis you want to plot along (_xx, _yy, _zz)
    xxindex : int, optional
        index of data along xx axis (ignored if axis = '_xx')
    yyindex : int, optional
        index of data along yy axis (ignored if axis = '_yy')
    zzindex : int, optional
        index of data along zz axis (ignored if axis = '_zz')
    axvx1 : float, optional
        x position of vertical line on plot
    axvx2 : float, optional
        x position of vertical line on plot
    """


    if(axis == '_zz'):
        fieldval = np.asarray([dfields[fieldkey][i][yyindex][xxindex] for i in range(0,len(dfields[fieldkey+axis]))])
        xlbl = 'z'
    elif(axis == '_yy'):
        fieldval = np.asarray([dfields[fieldkey][zzindex][i][xxindex] for i in range(0,len(dfields[fieldkey+axis]))])
        xlbl = 'y'
    elif(axis == '_xx'):
        fieldval = np.asarray([dfields[fieldkey][zzindex][yyindex][i] for i in range(0,len(dfields[fieldkey+axis]))])
        xlbl = 'x'

    fieldcoord = np.asarray(dfields[fieldkey+axis])

    plt.figure(figsize=(20,10))
    plt.xlabel(xlbl)
    plt.ylabel(fieldkey)
    plt.plot(fieldcoord,fieldval)
    if(not(axvx1 != axvx1)): #if not nan
        plt.axvline(x=axvx1)
    if(not(axvx2 != axvx2)): #if not nan
        plt.axvline(x=axvx2)
    if(flnm == ''):
        plt.show()
    else:
        plt.savefig(flnm,format='png')
    plt.close()

def plot_all_fields(dfields, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0, flnm = ''):
    """
    Plots all field data at some static frame down a line along x,y,z.

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    axis : str, optional
        name of axis you want to plot along (_xx, _yy, _zz)
    xxindex : int, optional
        index of data along xx axis (ignored if axis = '_xx')
    yyindex : int, optional
        index of data along yy axis (ignored if axis = '_yy')
    zzindex : int, optional
        index of data along zz axis (ignored if axis = '_zz')
    """
    if(axis == '_zz'):
        ex = np.asarray([dfields['ex'][i][yyindex][xxindex] for i in range(0,len(dfields['ex'+axis]))])
        ey = np.asarray([dfields['ey'][i][yyindex][xxindex] for i in range(0,len(dfields['ey'+axis]))])
        ez = np.asarray([dfields['ez'][i][yyindex][xxindex] for i in range(0,len(dfields['ez'+axis]))])
        bx = np.asarray([dfields['bx'][i][yyindex][xxindex] for i in range(0,len(dfields['bx'+axis]))])
        by = np.asarray([dfields['by'][i][yyindex][xxindex] for i in range(0,len(dfields['by'+axis]))])
        bz = np.asarray([dfields['bz'][i][yyindex][xxindex] for i in range(0,len(dfields['bz'+axis]))])
    elif(axis == '_yy'):
        ex = np.asarray([dfields['ex'][zzindex][i][xxindex] for i in range(0,len(dfields['ex'+axis]))])
        ey = np.asarray([dfields['ey'][zzindex][i][xxindex] for i in range(0,len(dfields['ex'+axis]))])
        ez = np.asarray([dfields['ez'][zzindex][i][xxindex] for i in range(0,len(dfields['ex'+axis]))])
        bx = np.asarray([dfields['bx'][zzindex][i][xxindex] for i in range(0,len(dfields['bx'+axis]))])
        by = np.asarray([dfields['by'][zzindex][i][xxindex] for i in range(0,len(dfields['by'+axis]))])
        bz = np.asarray([dfields['bz'][zzindex][i][xxindex] for i in range(0,len(dfields['bz'+axis]))])
    elif(axis == '_xx'):
        ex = np.asarray([dfields['ex'][zzindex][yyindex][i] for i in range(0,len(dfields['ex'+axis]))])
        ey = np.asarray([dfields['ey'][zzindex][yyindex][i] for i in range(0,len(dfields['ex'+axis]))])
        ez = np.asarray([dfields['ez'][zzindex][yyindex][i] for i in range(0,len(dfields['ex'+axis]))])
        bx = np.asarray([dfields['bx'][zzindex][yyindex][i] for i in range(0,len(dfields['bx'+axis]))])
        by = np.asarray([dfields['by'][zzindex][yyindex][i] for i in range(0,len(dfields['by'+axis]))])
        bz = np.asarray([dfields['bz'][zzindex][yyindex][i] for i in range(0,len(dfields['bz'+axis]))])

    fieldcoord = np.asarray(dfields['ex'+axis]) #assumes all fields have same coordinates

    fig, axs = plt.subplots(6,figsize=(20,10))
    axs[0].plot(fieldcoord,ex,label="ex")
    axs[0].set_ylabel("$ex$")
    axs[1].plot(fieldcoord,ey,label='ey')
    axs[1].set_ylabel("$ey$")
    axs[2].plot(fieldcoord,ez,label='ez')
    axs[2].set_ylabel("$ez$")
    axs[3].plot(fieldcoord,bx,label='bx')
    axs[3].set_ylabel("$bx$")
    axs[4].plot(fieldcoord,by,label='by')
    axs[4].set_ylabel("$by$")
    axs[5].plot(fieldcoord,bz,label='bz')
    axs[5].set_ylabel("$bz$")
    if(axis == '_xx'):
        axs[5].set_xlabel("$x$")
    elif(axis == '_yy'):
        axs[5].set_xlabel("$y$")
    elif(axis == '_yy'):
        axs[5].set_xlabel("$z$")
    plt.subplots_adjust(hspace=0.5)
    if(flnm == ''):
        plt.show()
    else:
        plt.savefig(flnm,format='png')
    plt.close()

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

def makefieldpmesh(dfields,fieldkey,planename,flnm = '',takeaxisaverage=True, xxindex=float('nan'), yyindex=float('nan'), zzindex=float('nan'), xlimmin=None,xlimmax=None):
    """
    Makes pmesh of given field

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    planename : str
        name of plane you want to plot (xy, xz, yz)
    flnm : str
        filename of output file
    takeaxisaverage : bool, optional
        if true, take average along axis not included in planename
    xxindex : int, optional
        index of data along xx axis (ignored if axis = '_xx')
    yyindex : int, optional
        index of data along yy axis (ignored if axis = '_yy')
    zzindex : int, optional
        index of data along zz axis (ignored if axis = '_zz')
    """

    if(planename=='xy'):
        ttl = fieldkey+'(x,y)'
        xlbl = 'x (di)'
        ylbl = 'y (di)'
        xplot1d = dfields[fieldkey+'_xx'][:]
        yplot1d = dfields[fieldkey+'_yy'][:]
        axisidx = 0 #used to take average along z if no index is specified
        axis = '_zz'

    elif(planename=='xz'):
        ttl = fieldkey+'(x,z)'
        xlbl = 'x (di)'
        ylbl = 'z (di)'
        xplot1d = dfields[fieldkey+'_xx'][:]
        yplot1d = dfields[fieldkey+'_zz'][:]
        axisidx = 1 #used to take average along y if no index is specified
        axis = '_yy'

    elif(planename=='yz'):
        ttl = fieldkey+'(y,z)'
        xlbl = 'y (di)'
        ylbl = 'z (di)'
        xplot1d = dfields[fieldkey+'_yy'][:]
        yplot1d = dfields[fieldkey+'_zz'][:]
        axisidx = 2 #used to take average along x if no index is specified
        axis = '_xx'

    if(takeaxisaverage):
        fieldpmesh = np.mean(dfields[fieldkey],axis=axisidx)
    elif(planename == 'xy'):
        fieldpmesh = np.asarray(dfields[fieldkey])[zzindex,:,:]
    elif(planename == 'xz'):
        fieldpmesh = np.asarray(dfields[fieldkey])[:,yyindex,:]
    elif(planename == 'yz'):
        fieldpmesh = np.asarray(dfields[fieldkey])[:,:,xxindex]

    #make 2d arrays for more explicit plotting
    xplot = np.zeros((len(yplot1d),len(xplot1d)))
    yplot = np.zeros((len(yplot1d),len(xplot1d)))
    for i in range(0,len(yplot1d)):
        for j in range(0,len(xplot1d)):
            xplot[i][j] = xplot1d[j]

    for i in range(0,len(yplot1d)):
        for j in range(0,len(xplot1d)):
            yplot[i][j] = yplot1d[i]

    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots
    plt.figure(figsize=(6.5,6))
    plt.figure(figsize=(6.5,6))
    plt.pcolormesh(xplot, yplot, fieldpmesh, cmap="inferno", shading="gouraud")
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
    plt.grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    #clb = plt.colorbar(format="%.1f", ticks=np.linspace(-maxCe, maxCe, 8), fraction=0.046, pad=0.04) #TODO: make static colorbar based on max range of C
    plt.colorbar()
    #plt.setp(plt.gca(), aspect=1.0)
    plt.gcf().subplots_adjust(bottom=0.15)
    if(xlimmin != None and xlimmax != None):
        plt.xlim(xlimmin, xlimmax)
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png')
        plt.close('all')#saves RAM
    else:
        plt.show()
        plt.close()

def plot_flow(dflow, flowkey, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0, axvx1 = float('nan'), axvx2 = float('nan'), flnm = ''):
    """
    Plots flow data

    Parameters
    ----------
    dflow : dict
        flow data dictionary from flow_loader
    flowkey : str
        name of flow you want to plot (ux, uy, uz)
    xxindex : int, optional
        index of data along xx axis (ignored if axis = '_xx')
    yyindex : int, optional
        index of data along yy axis (ignored if axis = '_yy')
    zzindex : int, optional
        index of data along zz axis (ignored if axis = '_zz')
    axvx1 : float, optional
        x position of vertical line on plot
    axvx2 : float, optional
        x position of vertical line on plot
    """
    if(axis == '_zz'):
        flowval = np.asarray([dflow[flowkey][i][yyindex][xxindex] for i in range(0,len(dflow[flowkey+axis]))])
        xlbl = 'z'
    elif(axis == '_yy'):
        flowval = np.asarray([dflow[flowkey][zzindex][i][xxindex] for i in range(0,len(dflow[flowkey+axis]))])
        xlbl = 'y'
    elif(axis == '_xx'):
        flowval = np.asarray([dflow[flowkey][zzindex][yyindex][i] for i in range(0,len(dflow[flowkey+axis]))])
        xlbl = 'x'

    flowcoord = np.asarray(dflow[flowkey+axis])

    plt.figure(figsize=(20,10))
    plt.xlabel(xlbl)
    plt.ylabel(flowkey)
    if(not(axvx1 != axvx1)): #if not nan
        plt.axvline(x=axvx1)
    if(not(axvx2 != axvx2)): #if not nan
        plt.axvline(x=axvx2)
    plt.plot(flowcoord,flowval)
    if(flnm == ''):
        plt.show()
    else:
        plt.savefig(flnm,format='png')
    plt.close()

def plot_all_flow(dflow, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0, flnm = ''):
    """
    Plots all flow data at some static frame down a line along x,y,z.

    Parameters
    ----------
    dflow : dict
        flow data dictionary from flow_loader
    axis : str, optional
        name of axis you want to plot along (_xx, _yy, _zz)
    xxindex : int, optional
        index of data along xx axis (ignored if axis = '_xx')
    yyindex : int, optional
        index of data along yy axis (ignored if axis = '_yy')
    zzindex : int, optional
        index of data along zz axis (ignored if axis = '_zz')
    """

    if(axis == '_zz'):
        ux = np.asarray([dflow['ux'][i][yyindex][xxindex] for i in range(0,len(dflow['ux'+axis]))])
        uy = np.asarray([dflow['uy'][i][yyindex][xxindex] for i in range(0,len(dflow['uy'+axis]))])
        uz = np.asarray([dflow['uz'][i][yyindex][xxindex] for i in range(0,len(dflow['uz'+axis]))])
    elif(axis == '_yy'):
        ux = np.asarray([dflow['ux'][zzindex][i][xxindex] for i in range(0,len(dflow['ux'+axis]))])
        uy = np.asarray([dflow['uy'][zzindex][i][xxindex] for i in range(0,len(dflow['uy'+axis]))])
        uz = np.asarray([dflow['uz'][zzindex][i][xxindex] for i in range(0,len(dflow['uz'+axis]))])
    elif(axis == '_xx'):
        ux = np.asarray([dflow['ux'][zzindex][yyindex][i] for i in range(0,len(dflow['ux'+axis]))])
        uy = np.asarray([dflow['uy'][zzindex][yyindex][i] for i in range(0,len(dflow['uy'+axis]))])
        uz = np.asarray([dflow['uz'][zzindex][yyindex][i] for i in range(0,len(dflow['uz'+axis]))])

    fieldcoord = np.asarray(dflow['ux'+axis]) #assumes all fields have same coordinates

    fig, axs = plt.subplots(3,figsize=(20,10))
    axs[0].plot(fieldcoord,ux,label="vx")
    axs[0].set_ylabel("$ux$")
    axs[1].plot(fieldcoord,uy,label='vy')
    axs[1].set_ylabel("$uy$")
    axs[2].plot(fieldcoord,uz,label='vz')
    axs[2].set_ylabel("$uz$")
    if(axis == '_xx'):
        axs[2].set_xlabel("$x$")
    elif(axis == '_yy'):
        axs[2].set_xlabel("$y$")
    elif(axis == '_yy'):
        axs[2].set_xlabel("$z$")
    plt.subplots_adjust(hspace=0.5)
    if(flnm != ''):
        plt.savefig(flnm,format='png')
    else:
        plt.show()

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

    plt.figure()

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

def plot_field_time(dfieldsdict, fieldkey, xxindex = 0, yyindex = 0, zzindex = 0):
    """
    Plots field at static location as a function of time.

    Parameters
    ----------
    dfieldsdict : dict
        dictonary of dfields and corresponding frame number from all_field_loader
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    xxindex : int
        index of data along xx axis
    yyindex : int
        index of data along yy axis
    zzindex : int
        index of data along zz axis
    """

    fieldval = np.asarray([dfields[fieldkey][zzindex][yyindex][xxindex] for dfields in dfieldsdict['dfields']])
    xlbl = 't'

    fieldcoord = np.asarray(dfieldsdict['frame'])

    plt.figure(figsize=(20,10))
    plt.xlabel(xlbl)
    plt.ylabel(fieldkey)
    plt.plot(fieldcoord,fieldval)
    plt.show()

def stack_line_plot(dfieldsdict, fieldkey, xshockvals = [], axis = '_xx', xxindex = 0, yyindex = 0, zzindex = 0):
    """

    """

    fig, axs = plt.subplots(len(dfieldsdict['frame']), sharex=True, sharey=True)
    fieldcoord = np.asarray(dfieldsdict['dfields'][0][fieldkey+axis])
    fig.set_size_inches(18.5, 30.)

    #sbpltlocation = len(dfielddict['frame'])+10+1
    for k in range(0,len(dfieldsdict['frame'])):

        #_ax = plt.subplots(len(dfielddict['frame']),k,sharex=True)
        if(axis == '_zz'):
            fieldval = np.asarray([dfieldsdict['dfields'][k][fieldkey][i][yyindex][xxindex] for i in range(0,len(dfieldsdict['dfields'][k][fieldkey+axis]))])
            xlbl = 'z'
        elif(axis == '_yy'):
            fieldval = np.asarray([dfieldsdict['dfields'][k][fieldkey][zzindex][i][xxindex] for i in range(0,len(dfieldsdict['dfields'][k][fieldkey+axis]))])
            xlbl = 'y'
        elif(axis == '_xx'):
            fieldval = np.asarray([dfieldsdict['dfields'][k][fieldkey][xxindex][yyindex][i] for i in range(0,len(dfieldsdict['dfields'][k][fieldkey+axis]))])
            xlbl = 'x'

        axs[k].plot(fieldcoord,fieldval)
        if(len(xshockvals) > 0):
            axs[k].scatter([xshockvals[k]],[0.])
        #axs[k].ylabel(fieldkey+'(frame = '+str(dfielddict['frame'][k])+')')


    #plt.figure(figsize=(20,10))
    # plt.xlabel(xlbl)
    # plt.ylabel(fieldkey)
    # plt.plot(fieldcoord,fieldval)
    plt.show()

#(vx,vy,vmax,Ce,fieldkey,flnm = '',ttl='')
def plot_cor_and_dist_supergrid(vx, vy, vz, vmax,
                                H_xy, H_xz, H_yz,
                                CEx_xy,CEx_xz, CEx_yz,
                                CEy_xy,CEy_xz, CEy_yz,
                                CEz_xy,CEz_xz, CEz_yz,
                                flnm = '', ttl = ''):
    """

    """
    from lib.analysisfunctions import threeVelToTwoVel

    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots

    fig, axs = plt.subplots(4,3,figsize=(4*5,3*5))

    vx_xy, vy_xy = threeVelToTwoVel(vx,vy,vz,'xy')
    vx_xz, vz_xz = threeVelToTwoVel(vx,vy,vz,'xz')
    vy_yz, vz_yz = threeVelToTwoVel(vx,vy,vz,'yz')

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

#TODO: remove flnm parameter, the passed value is not used
def make_superplot_gif(vx, vy, vz, vmax, Hist, CEx, CEy, CEz, x, directory, flnm):
    #make plots of data and put into directory

    from lib.analysisfunctions import threeHistToTwoHist
    from lib.analysisfunctions import threeCorToTwoCor

    try:
        os.mkdir(directory)
    except:
        pass

    for i in range(0,len(x)):
        print('Making plot ' + str(i)+' of '+str(len(x)))
        flnm = directory+'/'+str(i).zfill(6)

        #Project onto 2d axis
        H_xy = threeHistToTwoHist(Hist[i],'xy')
        H_xz = threeHistToTwoHist(Hist[i],'xz')
        H_yz = threeHistToTwoHist(Hist[i],'yz')
        CEx_xy = threeCorToTwoCor(CEx[i],'xy')
        CEx_xz = threeCorToTwoCor(CEx[i],'xz')
        CEx_yz = threeCorToTwoCor(CEx[i],'yz')
        CEy_xy = threeCorToTwoCor(CEy[i],'xy')
        CEy_xz = threeCorToTwoCor(CEy[i],'xz')
        CEy_yz = threeCorToTwoCor(CEy[i],'yz')
        CEz_xy = threeCorToTwoCor(CEz[i],'xy')
        CEz_xz = threeCorToTwoCor(CEz[i],'xz')
        CEz_yz = threeCorToTwoCor(CEz[i],'yz')

        plot_cor_and_dist_supergrid(vx, vy, vz, vmax,
                                        H_xy, H_xz, H_yz,
                                        CEx_xy,CEx_xz, CEx_yz,
                                        CEy_xy,CEy_xz, CEy_yz,
                                        CEz_xy,CEz_xz, CEz_yz,
                                        flnm = flnm, ttl = 'x(di): ' + str(x[i]))
        plt.close('all') #saves RAM

def make_fieldpmesh_sweep(dfields,fieldkey,planename,directory,xlimmin=None,xlimmax=None):
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
            makefieldpmesh(dfields,fieldkey,planename,flnm = flnm,takeaxisaverage=False,xxindex=i,xlimmin=xlimmin,xlimmax=xlimmax)
        elif(planename=='xz'):
            makefieldpmesh(dfields,fieldkey,planename,flnm = flnm,takeaxisaverage=False,yyindex=i,xlimmin=xlimmin,xlimmax=xlimmax)
        elif(planename=='xy'):
            makefieldpmesh(dfields,fieldkey,planename,flnm = flnm,takeaxisaverage=False,zzindex=i,xlimmin=xlimmin,xlimmax=xlimmax)
        else:
            print("Please enter a valid planename...")
            break

def make_gif_from_folder(directory,flnm):
    #Not sure why this is necessary to break this up into a seperate function rather than including in make_superplot_gif
    #make gif
    import imageio #should import here as it might not be installed on every machine
    images = []
    filenames = os.listdir(directory)
    filenames = sorted(filenames)
    try:
        filenames.remove('.DS_store')
    except:
        pass

    print(filenames)

    for filename in filenames:
        images.append(imageio.imread(directory+'/'+filename))
    imageio.mimsave(flnm, images)


def plot_fft_norm(dfields,fieldkey,planename,flnm = '',takeaxisaverage=True, xxindex=float('nan'), yyindex=float('nan'), zzindex=float('nan'), plotlog = True):
    """


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

def plot_stack_field_along_x(dfields,fieldkey,stackaxis,yyindex=0,zzindex=0,xlow=None,xhigh=None):
    """

    """
    if(stackaxis != '_yy' and stackaxis != '_zz'):
        print("Please stack along _yy or _zz")

    plt.figure()
    fieldcoord = np.asarray(dfields[fieldkey+'_xx'])
    for k in range(0,len(dfields[fieldkey+stackaxis])):
        fieldval = np.asarray([dfields[fieldkey][zzindex][yyindex][i] for i in range(0,len(dfields[fieldkey+'_xx']))])
        if(stackaxis == '_yy'):
            yyindex += 1
        elif(stackaxis == '_zz'):
            zzindex += 1
        plt.xlabel('x')
        plt.ylabel(fieldkey)
        if(xlow != None and xhigh != None):
            plt.xlim(xlow,xhigh)
        plt.plot(fieldcoord,fieldval)

    plt.show()
    plt.close()
