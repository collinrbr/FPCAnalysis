# plotfunctions.py>

# Here we have functions related to plotting dHybridR data

def plot_field(dfields, fieldkey, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0):
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
    """


    if(axis == '_zz'):
        fieldval = np.asarray([dfields[fieldkey][i][yyindex][zzindex] for i in range(0,len(dfields[fieldkey+axis]))])
        xlbl = 'z'
    elif(axis == '_yy'):
        fieldval = np.asarray([dfields[fieldkey][xxindex][i][zzindex] for i in range(0,len(dfields[fieldkey+axis]))])
        xlbl = 'y'
    elif(axis == '_xx'):
        fieldval = np.asarray([dfields[fieldkey][xxindex][yyindex][i] for i in range(0,len(dfields[fieldkey+axis]))])
        xlbl = 'x'

    fieldcoord = np.asarray(dfields[fieldkey+axis])

    plt.figure(figsize=(20,10))
    plt.xlabel(xlbl)
    plt.ylabel(fieldkey)
    plt.plot(fieldcoord,fieldval)
    plt.show()

def plot_all_fields(dfields, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0):
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
        ex = np.asarray([dfields['ex'][i][yyindex][zzindex] for i in range(0,len(dfields['ex'+axis]))])
        ey = np.asarray([dfields['ey'][i][yyindex][zzindex] for i in range(0,len(dfields['ey'+axis]))])
        ez = np.asarray([dfields['ez'][i][yyindex][zzindex] for i in range(0,len(dfields['ez'+axis]))])
        bx = np.asarray([dfields['bx'][i][yyindex][zzindex] for i in range(0,len(dfields['bx'+axis]))])
        by = np.asarray([dfields['by'][i][yyindex][zzindex] for i in range(0,len(dfields['by'+axis]))])
        bz = np.asarray([dfields['bz'][i][yyindex][zzindex] for i in range(0,len(dfields['bz'+axis]))])
    elif(axis == '_yy'):
        ex = np.asarray([dfields['ex'][xxindex][i][zzindex] for i in range(0,len(dfields['ex'+axis]))])
        ey = np.asarray([dfields['ey'][xxindex][i][zzindex] for i in range(0,len(dfields['ex'+axis]))])
        ez = np.asarray([dfields['ez'][xxindex][i][zzindex] for i in range(0,len(dfields['ex'+axis]))])
        bx = np.asarray([dfields['bx'][xxindex][i][zzindex] for i in range(0,len(dfields['bx'+axis]))])
        by = np.asarray([dfields['by'][xxindex][i][zzindex] for i in range(0,len(dfields['by'+axis]))])
        bz = np.asarray([dfields['bz'][xxindex][i][zzindex] for i in range(0,len(dfields['bz'+axis]))])
    elif(axis == '_xx'):
        ex = np.asarray([dfields['ex'][xxindex][yyindex][i] for i in range(0,len(dfields['ex'+axis]))])
        ey = np.asarray([dfields['ey'][xxindex][yyindex][i] for i in range(0,len(dfields['ex'+axis]))])
        ez = np.asarray([dfields['ez'][xxindex][yyindex][i] for i in range(0,len(dfields['ex'+axis]))])
        bx = np.asarray([dfields['bx'][xxindex][yyindex][i] for i in range(0,len(dfields['bx'+axis]))])
        by = np.asarray([dfields['by'][xxindex][yyindex][i] for i in range(0,len(dfields['by'+axis]))])
        bz = np.asarray([dfields['bz'][xxindex][yyindex][i] for i in range(0,len(dfields['bz'+axis]))])

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
    plt.show()

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

    maxCe = max(np.max(Ce),np.max(abs(Ce)))

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

    maxCe = max(np.max(Ce),np.max(abs(Ce)))

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

def makefieldpmesh(dfields,fieldkey,planename):
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

    """
    if(planename=='xy'):
        ttl = fieldkey+'(x,y)'
        xlbl = 'x (di)'
        ylbl = 'y (di)'
        fieldpmesh = np.mean(dfields[fieldkey],axis=0)
        xplot1d = dfields[fieldkey+'_xx'][:]
        yplot1d = dfields[fieldkey+'_yy'][:]

    elif(planename=='xz'):
        ttl = fieldkey+'(x,z)'
        xlbl = 'x (di)'
        ylbl = 'z (di)'
        fieldpmesh = np.mean(dfields[fieldkey],axis=1)
        xplot1d = dfields[fieldkey+'_xx'][:]
        yplot1d = dfields[fieldkey+'_zz'][:]



    elif(planename=='yz'):
        ttl = fieldkey+'(y,z)'
        xlbl = 'y (di)'
        ylbl = 'z (di)'
        fieldpmesh = np.mean(dfields[fieldkey],axis=2)
        xplot1d = dfields[fieldkey+'_yy'][:]
        yplot1d = dfields[fieldkey+'_zz'][:]

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
    plt.title(ttl,loc="right")
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    #clb = plt.colorbar(format="%.1f", ticks=np.linspace(-maxCe, maxCe, 8), fraction=0.046, pad=0.04) #TODO: make static colorbar based on max range of C
    plt.colorbar()
    #plt.setp(plt.gca(), aspect=1.0)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()
    plt.close()
