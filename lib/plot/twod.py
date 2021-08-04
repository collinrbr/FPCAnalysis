# 2dfields.py>

# functions related to plotting 1d field data

import numpy as np
import matplotlib.pyplot as plt
import os

def make_field_pmesh(dfields,fieldkey,planename,flnm = '',takeaxisaverage=True, xxindex=float('nan'), yyindex=float('nan'), zzindex=float('nan'), xlimmin=None,xlimmax=None):
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

    if(planename=='xy'):
        ttl = fieldttl+'(x,y)$ at '
        xlbl = '$x$ (di)'
        ylbl = '$y$ (di)'
        xplot1d = dfields[fieldkey+'_xx'][:]
        yplot1d = dfields[fieldkey+'_yy'][:]
        axisidx = 0 #used to take average along z if no index is specified
        axis = '_zz'

    elif(planename=='xz'):
        ttl = fieldttl+'(x,z)$ at '
        xlbl = '$x$ (di)'
        ylbl = '$z$ (di)'
        xplot1d = dfields[fieldkey+'_xx'][:]
        yplot1d = dfields[fieldkey+'_zz'][:]
        axisidx = 1 #used to take average along y if no index is specified
        axis = '_yy'

    elif(planename=='yz'):
        ttl = fieldttl+'(y,z)$ at '
        xlbl = '$y$ (di)'
        ylbl = '$z$ (di)'
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
    if((planename == 'xy' or planename == 'xz') and (xlimmin == None and xlimmax == None)):
        plt.figure(figsize=(2*6.5,6))
    else:
        plt.figure(figsize=(6.5,6))
    plt.pcolormesh(xplot, yplot, fieldpmesh, cmap="inferno", shading="gouraud")
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

def compare_pmesh_fields_yz(dfields, flnm = '', ttl ='', takeaxisaverage=False, xxindex=float('nan')):
    """

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

    """
    try:
        os.mkdir(directory)
    except:
        pass

    for i in range(0,len(dfields['ex_xx'])):
        print('Making plot '+str(i)+' of '+str(len(dfields['ex_xx'])))
        flnm = directory+'/'+str(i).zfill(6)
        compare_pmesh_fields_yz(dfields, flnm = flnm, ttl ='x (di): ' + str(dfields['ex_xx'][i]), takeaxisaverage=False, xxindex=i)
