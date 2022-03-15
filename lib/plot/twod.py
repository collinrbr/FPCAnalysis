# 2dfields.py>

# functions related to plotting 1d field data

import numpy as np
import matplotlib.pyplot as plt
import os

def make_field_pmesh(ddict,fieldkey,planename,flnm = '',takeaxisaverage=False, xxindex=float('nan'), yyindex=float('nan'), zzindex=float('nan'), xlimmin=None,xlimmax=None):
    """
    Makes pmesh of given field

    Parameters
    ----------
    ddict : dict
        field or flow data dictionary
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz, ux, uy, uz)
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
    xlimmin : float
        minimum plotted x value (ignored if xx is not in plane)
    xlimmax : float
        maximum plotted x value (ignored if xx is not in plane)
    """

    fieldttl = ''
    if(fieldkey == 'ex'):
        fieldttl = '$E_x'
    elif(fieldkey == 'ey'):
        fieldttl = '$E_y'
    elif(fieldkey == 'ez'):
        fieldttl = '$E_z'
    elif(fieldkey == 'bx'):
        fieldttl = '$B_x'
    elif(fieldkey == 'by'):
        fieldttl = '$B_y'
    elif(fieldkey == 'bz'):
        fieldttl = '$B_z'
    elif(fieldkey == 'ux'):
        fieldttl = '$U_x'
    elif(fieldkey == 'uy'):
        fieldttl = '$U_y'
    elif(fieldkey == 'uz'):
        fieldttl = '$U_z'

    if(planename=='xy'):
        ttl = fieldttl+'(x,y)$ at '
        xlbl = '$x$ (di)'
        ylbl = '$y$ (di)'
        xplot1d = ddict[fieldkey+'_xx'][:]
        yplot1d = ddict[fieldkey+'_yy'][:]
        axisidx = 0 #used to take average along z if no index is specified
        axis = '_zz'

    elif(planename=='xz'):
        ttl = fieldttl+'(x,z)$ at '
        xlbl = '$x$ (di)'
        ylbl = '$z$ (di)'
        xplot1d = ddict[fieldkey+'_xx'][:]
        yplot1d = ddict[fieldkey+'_zz'][:]
        axisidx = 1 #used to take average along y if no index is specified
        axis = '_yy'

    elif(planename=='yz'):
        ttl = fieldttl+'(y,z)$ at '
        xlbl = '$y$ (di)'
        ylbl = '$z$ (di)'
        xplot1d = ddict[fieldkey+'_yy'][:]
        yplot1d = ddict[fieldkey+'_zz'][:]
        axisidx = 2 #used to take average along x if no index is specified
        axis = '_xx'

    if(takeaxisaverage):
        fieldpmesh = np.mean(ddict[fieldkey],axis=axisidx)
    elif(planename == 'xy'):
        fieldpmesh = np.asarray(ddict[fieldkey])[zzindex,:,:]
    elif(planename == 'xz'):
        fieldpmesh = np.asarray(ddict[fieldkey])[:,yyindex,:]
    elif(planename == 'yz'):
        fieldpmesh = np.asarray(ddict[fieldkey])[:,:,xxindex]

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
    if((planename == 'xy' or planename == 'xz') and (xlimmin == None and xlimmax == None)):
        plt.figure(figsize=(2*6.5,6))
    else:
        plt.figure(figsize=(6.5,6))
    plt.pcolormesh(xplot, yplot, fieldpmesh, cmap="inferno", shading="gouraud")
    if(takeaxisaverage):
        plt.title(ttl,loc="right")
    elif(planename == 'xy'):
        plt.title(ttl+' z = '+str(ddict[fieldkey+axis][zzindex])+' (di)',loc="right")
    elif(planename == 'xz'):
        plt.title(ttl+' y = '+str(ddict[fieldkey+axis][yyindex])+' (di)',loc="right")
    elif(planename == 'yz'):
        plt.title(ttl+' x = '+str(ddict[fieldkey+axis][xxindex])+' (di)',loc="right")
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
        plt.savefig(flnm+'.png',format='png',dpi=300)
        plt.close('all')#saves RAM
    else:
        plt.show()
        plt.close()

def make_super_pmeshplot(dfields,dflow,dden,zzindex = 0,flnm=''):

    fig, axs = plt.subplots(8,figsize=(15,8))

    #Bx
    axs[0].pcolormesh(dfields['bx_xx'], dfields['bx_yy'], dfields['bx'][zzindex,:,:], cmap="plasma", shading="gouraud")

    #By
    axs[1].pcolormesh(dfields['by_xx'], dfields['by_yy'], dfields['by'][zzindex,:,:], cmap="plasma", shading="gouraud")

    #Bz
    axs[2].pcolormesh(dfields['bz_xx'], dfields['bz_yy'], dfields['bz'][zzindex,:,:], cmap="plasma", shading="gouraud")

    #Btot
    btot = np.zeros(dfields['bx'].shape)
    for _i in range(0,len(btot)):
        for _j in range(0,len(btot[_i])):
            for _k in range(0,len(btot[_i][_j])):
                btot[_i,_j,_k] = np.linalg.norm([dfields['bx'][_i,_j,_k],dfields['by'][_i,_j,_k],dfields['bz'][_i,_j,_k]])
    axs[3].pcolormesh(dfields['bx_xx'], dfields['bx_yy'], btot[zzindex,:,:], cmap="magma", shading="gouraud")

    #den
    axs[4].pcolormesh(dden['den_xx'],dden['den_yy'],dden['den'][zzindex,:,:], cmap="Spectral", shading="gouraud")

    #vx
    axs[5].pcolormesh(dflow['ux_xx'],dden['ux_yy'],dden['ux'][zzindex,:,:], cmap="bwr", shading="gouraud")

    #vy
    axs[6].pcolormesh(dflow['uy_xx'],dden['uy_yy'],dden['uy'][zzindex,:,:], cmap="bwr", shading="gouraud")

    #vz
    axs[7].pcolormesh(dflow['uz_xx'],dden['uz_yy'],dden['uz'][zzindex,:,:], cmap="bwr", shading="gouraud")

    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=300)
    else:
        plt.show()
        plt.close()

def make_fieldpmesh_sweep(dfields,fieldkey,planename,directory,xlimmin=None,xlimmax=None):
    """
    Makes sweep gif of field pmesh

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    planename : str
        name of plane you want to plot (xy, xz, yz)
    directory : str
        name of directory you want to create and put plots into
        (omit final '/')
    xlimmin : float
        minimum plotted x value (ignored if xx is not in plane)
    xlimmax : float
        maximum plotted x value (ignored if xx is not in plane)
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

def compare_pmesh_fields_yz(dfields, flnm = '', ttl ='', takeaxisaverage=False, xxindex=float('nan')):
    """
    Plots pmesh of all fields on same figure for easier comparison along the yz projection

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    flnm : str
        filename of output file
    ttl : str, optional
        title of plot
    takeaxisaverage : bool, optional
        if true, take average along axis not included in planename
    xxindex : int
        xx index of slice that is to be plotted
    """

    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots
    fig, axs = plt.subplots(2,3,figsize=(4*5,3*5))

    #plot x,y-------------------------------------------------------------------
    xlbl = 'y (di)'
    ylbl = 'z (di)'
    xplot1d = dfields['ex_yy'][:]
    yplot1d = dfields['ex_zz'][:]
    axisidx = 2 #used to take average along z if no index is specified
    axis = '_zz'
    if(takeaxisaverage):
        fieldpmeshex = np.mean(dfields['ex'],axis=axisidx) #TODO: convert averaging to box average
        fieldpmeshey = np.mean(dfields['ey'],axis=axisidx)
        fieldpmeshez = np.mean(dfields['ez'],axis=axisidx)
        fieldpmeshbx = np.mean(dfields['bx'],axis=axisidx)
        fieldpmeshby = np.mean(dfields['by'],axis=axisidx)
        fieldpmeshbz = np.mean(dfields['bz'],axis=axisidx)
    else:
        fieldpmeshex = np.asarray(dfields['ex'])[:,:,xxindex]
        fieldpmeshey = np.asarray(dfields['ey'])[:,:,xxindex]
        fieldpmeshez = np.asarray(dfields['ez'])[:,:,xxindex]
        fieldpmeshbx = np.asarray(dfields['bx'])[:,:,xxindex]
        fieldpmeshby = np.asarray(dfields['by'])[:,:,xxindex]
        fieldpmeshbz = np.asarray(dfields['bz'])[:,:,xxindex]

    #make 2d arrays for more explicit plotting #TODO: use mesh_3d_to_2d here
    xplot = np.zeros((len(yplot1d),len(xplot1d)))
    yplot = np.zeros((len(yplot1d),len(xplot1d)))
    for i in range(0,len(yplot1d)):
        for j in range(0,len(xplot1d)):
            xplot[i][j] = xplot1d[j]
    for i in range(0,len(yplot1d)):
        for j in range(0,len(xplot1d)):
            yplot[i][j] = yplot1d[i]

    axs[0,0].pcolormesh(xplot, yplot, fieldpmeshex, cmap="inferno", shading="gouraud")
    axs[0,0].grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    axs[0,0].set_title('Ex(y,z)')
    axs[0,0].set_xlabel(xlbl)
    axs[0,0].set_ylabel(ylbl)
    axs[0,1].pcolormesh(xplot, yplot, fieldpmeshey, cmap="inferno", shading="gouraud")
    axs[0,1].grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    axs[0,1].set_title('Ey(y,z)')
    axs[0,1].set_xlabel(xlbl)
    axs[0,1].set_ylabel(ylbl)
    axs[0,2].pcolormesh(xplot, yplot, fieldpmeshez, cmap="inferno", shading="gouraud")
    axs[0,2].grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    axs[0,2].set_title('Ez(y,z)')
    axs[0,2].set_xlabel(xlbl)
    axs[0,2].set_ylabel(ylbl)
    axs[1,0].pcolormesh(xplot, yplot, fieldpmeshbx, cmap="inferno", shading="gouraud")
    axs[1,0].grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    axs[1,0].set_title('Bx(y,z)')
    axs[1,0].set_xlabel(xlbl)
    axs[1,0].set_ylabel(ylbl)
    axs[1,1].pcolormesh(xplot, yplot, fieldpmeshby, cmap="inferno", shading="gouraud")
    axs[1,1].grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    axs[1,1].set_title('By(y,z)')
    axs[1,1].set_xlabel(xlbl)
    axs[1,1].set_ylabel(ylbl)
    axs[1,2].pcolormesh(xplot, yplot, fieldpmeshbz, cmap="inferno", shading="gouraud")
    axs[1,2].grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    axs[1,2].set_title('Bz(y,z)')
    axs[1,2].set_xlabel(xlbl)
    axs[1,2].set_ylabel(ylbl)

    fig.suptitle(ttl)
    #TODO: add position in title
    # if(takeaxisaverage):
    #     plt.title(ttl,loc="right")
    # elif(planename == 'xy'):
    #     plt.title(ttl+' z (di): '+str(dfields[fieldkey+axis][zzindex]),loc="right")
    # elif(planename == 'xz'):
    #     plt.title(ttl+' y (di): '+str(dfields[fieldkey+axis][yyindex]),loc="right")
    # elif(planename == 'yz'):
    #     plt.title(ttl+' x (di): '+str(dfields[fieldkey+axis][xxindex]),loc="right")

    #TODO: add colorbars
    #clb = plt.colorbar(format="%.1f", ticks=np.linspace(-maxCe, maxCe, 8), fraction=0.046, pad=0.04) #TODO: make static colorbar based on max range of C
    #plt.colorbar()
    #plt.setp(plt.gca(), aspect=1.0)

    #TODO: add limits to plots
    # if(xlimmin != None and xlimmax != None):
    #     plt.xlim(xlimmin, xlimmax)

    plt.gcf().subplots_adjust(bottom=0.15)
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png')
        plt.close('all')#saves RAM
    else:
        plt.show()
        plt.close()

def compare_pmesh_fields_yz_sweep(dfields,directory):
    """
    Makes sweep of pmesh of all fields along the yz projection

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    directory : str
        name of directory you want to create and put plots into
        (omit final '/')
    """
    try:
        os.mkdir(directory)
    except:
        pass

    for i in range(0,len(dfields['ex_xx'])):
        print('Making plot '+str(i)+' of '+str(len(dfields['ex_xx'])))
        flnm = directory+'/'+str(i).zfill(6)
        compare_pmesh_fields_yz(dfields, flnm = flnm, ttl ='x (di): ' + str(dfields['ex_xx'][i]), takeaxisaverage=False, xxindex=i)
