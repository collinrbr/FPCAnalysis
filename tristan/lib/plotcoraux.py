import matplotlib.pyplot as plt
import numpy as np

def plot_dist_v_fields_supergrid(vx, vy, vz, vmax,
                                H_xy, H_xz, H_yz,
                                CEx_xy,CEx_xz, CEx_yz,
                                CEy_xy,CEy_xz, CEy_yz,
                                CEz_xy,CEz_xz, CEz_yz,
                                dfavg,xval1,xval2,
                                flnm = '', ttl = '', computeJdotE = True, params = None, metadata = None, xpos = None, plotLog = False, plotLogHist = True,
                                plotFAC = False, plotFluc = False, plotAvg = False, isIon = True, listpos=False,xposval=None,normtoN = True,isLowPass=False,isHighPass=False,plotDiagJEOnly=True):
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
    from lib.arrayaux import mesh_3d_to_2d
    from lib.analysisaux import compute_energization
    import matplotlib.colors as colors

    if(normtoN):
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

    minHxyval = np.min(H_xy[np.nonzero(H_xy)])
    minHxzval = np.min(H_xz[np.nonzero(H_xz)])
    minHyzval = np.min(H_yz[np.nonzero(H_yz)])

    #H_xy
    if(plotLogHist):
        im00= axs[0,0].pcolormesh(vy_xy, vx_xy, H_xy, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHxyval, vmax=H_xy.max()))
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
        im01 = axs[0,1].pcolormesh(vz_xz, vx_xz, H_xz, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHxzval, vmax=H_xz.max()))
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
        im02 = axs[0,2].pcolormesh(vz_yz, vy_yz, H_yz.T, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHyzval, vmax=H_yz.max()))
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

 

def plot_cor_and_dist_supergrid(vx, vy, vz, vmax,
                                H_xy, H_xz, H_yz,
                                CEx_xy,CEx_xz, CEx_yz,
                                CEy_xy,CEy_xz, CEy_yz,
                                CEz_xy,CEz_xz, CEz_yz,
                                flnm = '', ttl = '', computeJdotE = True, params = None, metadata = None, xpos = None, plotLog = False, plotLogHist = True,
                                plotFAC = False, plotFluc = False, plotAvg = False, isIon = True, listpos=False,xposval=None,normtoN = True,isLowPass=False,isHighPass=False,plotDiagJEOnly=True):
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
    from lib.arrayaux import mesh_3d_to_2d
    from lib.analysisaux import compute_energization
    import matplotlib.colors as colors

    if(normtoN):
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

    fig, axs = plt.subplots(4,3,figsize=(4*5,3*5),sharex=True)

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

    minHxyval = np.min(H_xy[np.nonzero(H_xy)])
    minHxzval = np.min(H_xz[np.nonzero(H_xz)])
    minHyzval = np.min(H_yz[np.nonzero(H_yz)])

    #H_xy
    if(plotLogHist):
        im00= axs[0,0].pcolormesh(vy_xy, vx_xy, H_xy, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHxyval, vmax=H_xy.max()))
    else:
        im00= axs[0,0].pcolormesh(vy_xy, vx_xy, H_xy, cmap="plasma", shading="gouraud")
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
    axs[0,0].text(-vmax*2.2,0, r"$f$", ha='center', rotation=90, wrap=False)
    if(params != None):
        axs[0,0].text(-vmax*2.6,0, '$M_A = $ ' + str(abs(params['MachAlfven'])), ha='center', rotation=90, wrap=False)

    if(listpos):
        axs[0,0].text(-vmax*2.6,0, '$x / d_i = $ ' + str("{:.4f}".format(xposval)), ha='center', rotation=90, wrap=False)


    #H_xz
    if(plotLogHist):
        im01 = axs[0,1].pcolormesh(vz_xz, vx_xz, H_xz, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHxzval, vmax=H_xz.max()))
    else:
        im01 = axs[0,1].pcolormesh(vz_xz, vx_xz, H_xz, cmap="plasma", shading="gouraud")
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
        im02 = axs[0,2].pcolormesh(vz_yz, vy_yz, H_yz.T, cmap="plasma", shading="gouraud",norm=LogNorm(vmin=minHyzval, vmax=H_yz.max()))
    else:
        im02 = axs[0,2].pcolormesh(vz_yz, vy_yz, H_yz.T, cmap="plasma", shading="gouraud")
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
    axs[1,0].set_aspect('equal', 'box')
    axs[1,0].grid()
    if(plotFAC):
        if(plotAvg):
            axs[1,0].text(-vmax*2.2,0, r"$\overline{C_{E_{||}}}$", ha='center', rotation=90, wrap=False)
        elif(plotFluc):
            if(isLowPass):
                axs[1,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{||}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
            elif(isHighPass):
                axs[1,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{||}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
            else:
                axs[1,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{||}}}$", ha='center', rotation=90, wrap=False)
        else:
            axs[1,0].text(-vmax*2.2,0, r"$C_{E_{||}}$", ha='center', rotation=90, wrap=False)
    else:
        if(plotAvg):
            axs[1,0].text(-vmax*2.2,0, r"$\overline{C_{E_{x}}}$", ha='center', rotation=90, wrap=False)
        elif(plotFluc):
            if(isLowPass):
                axs[1,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{x}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
            elif(isHighPass):
                axs[1,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{x}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
            else:
                axs[1,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{x}}}$", ha='center', rotation=90, wrap=False)
        else:
            axs[1,0].text(-vmax*2.2,0, r"$C_{E_{x}}$", ha='center', rotation=90, wrap=False)
    if(params != None):
        axs[1,0].text(-vmax*2.6,0, '$\Theta_{Bn} = $ ' + str(params['thetaBn']), ha='center', rotation=90, wrap=False)
    if(computeJdotE):
        if(not(plotDiagJEOnly)):
            JdotE = compute_energization(CEx_xy,dv)
            if(plotFAC):
                if(plotAvg):
                    axs[1,0].set_title('$\overline{j_{||}} \cdot \overline{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[1,0].set_title('$\widetilde{j_{||}}^{k_{||} d_i < 15} \cdot \widetilde{E_{||}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[1,0].set_title('$\widetilde{j_{||}}^{k_{||} d_i > 15} \cdot \widetilde{E_{||}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[1,0].set_title('$\widetilde{j_{||}} \cdot \widetilde{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[1,0].set_title('$j_{||} \cdot E_{||}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[1,0].set_title('$\overline{j_x} \cdot \overline{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[1,0].set_title('$\widetilde{j_x}^{k_{||} d_i > 15} \cdot \widetilde{E_x}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[1,0].set_title('$\widetilde{j_x}^{k_{||} d_i > 15} \cdot \widetilde{E_x}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[1,0].set_title('$\widetilde{j_x} \cdot \widetilde{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[1,0].set_title('$j_x \cdot E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
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
    axs[1,1].set_aspect('equal', 'box')
    axs[1,1].grid()
    if(computeJdotE):
        if(not(plotDiagJEOnly)):
            JdotE = compute_energization(CEx_xz,dv)
            if(plotFAC):
                if(plotAvg):
                    axs[1,1].set_title('$\overline{j_{||}} \cdot \overline{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[1,1].set_title('$\widetilde{j_{||}}^{k_{||} d_i < 15} \cdot \widetilde{E_{||}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[1,1].set_title('$\widetilde{j_{||}}^{k_{||} d_i > 15} \cdot \widetilde{E_{||}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[1,1].set_title('$\widetilde{j_{||}} \cdot \widetilde{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[1,1].set_title('$j_{||} \cdot E_{||}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[1,1].set_title('$\overline{j_x} \cdot \overline{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[1,1].set_title('$\widetilde{j_x}^{k_{||} d_i > 15} \cdot \widetilde{E_x}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[1,1].set_title('$\widetilde{j_x}^{k_{||} d_i > 15} \cdot \widetilde{E_x}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[1,1].set_title('$\widetilde{j_x} \cdot \widetilde{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[1,1].set_title('$j_x \cdot E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
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
    axs[1,2].set_aspect('equal', 'box')
    axs[1,2].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEx_yz.T,dv)
        if(True):
            if(plotFAC):
                if(plotAvg):
                    axs[1,2].set_title('$\overline{j_{||}} \cdot \overline{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[1,2].set_title('$\widetilde{j_{||}}^{k_{||} d_i < 15} \cdot \widetilde{E_{||}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[1,2].set_title('$\widetilde{j_{||}}^{k_{||} d_i > 15} \cdot \widetilde{E_{||}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[1,2].set_title('$\widetilde{j_{||}} \cdot \widetilde{E_{||}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[1,2].set_title('$j_{||} \cdot E_{||}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[1,2].set_title('$\overline{j_x} \cdot \overline{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[1,2].set_title('$\widetilde{j_x}^{k_{||} d_i > 15} \cdot \widetilde{E_x}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[1,2].set_title('$\widetilde{j_x}^{k_{||} d_i > 15} \cdot \widetilde{E_x}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[1,2].set_title('$\widetilde{j_x} \cdot \widetilde{E_x}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[1,2].set_title('$j_x \cdot E_x$ = ' + "{:.2e}".format(JdotE),loc='left')
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
    axs[2,0].set_aspect('equal', 'box')
    if(plotFAC):
        if(plotAvg):
            axs[2,0].text(-vmax*2.2,0, r"$\overline{C_{E_{\perp,1}}}$", ha='center', rotation=90, wrap=False)
        elif(plotFluc):
            if(isLowPass):
                axs[2,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,1}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
            elif(isHighPass):
                axs[2,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,1}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
            else:
                axs[2,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,1}}}$", ha='center', rotation=90, wrap=False)
        else:
            axs[2,0].text(-vmax*2.2,0, r"$C_{E_{\perp,1}}$", ha='center', rotation=90, wrap=False)
    else:
        if(plotAvg):
            axs[2,0].text(-vmax*2.2,0, r"$\overline{C_{E_{y}}}$", ha='center', rotation=90, wrap=False)
        elif(plotFluc):
            if(isLowPass):
                 axs[2,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{y}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
            elif(isHighPass):
                axs[2,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{y}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
            else:
                axs[2,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{y}}}$", ha='center', rotation=90, wrap=False)
        else:
            axs[2,0].text(-vmax*2.2,0, r"$C_{E_{y}}$", ha='center', rotation=90, wrap=False)
    axs[2,0].grid()
    if(xpos != None):
        axs[2,0].text(-vmax*2.6,0,'$x/d_i = $' + str(xpos), ha='center', rotation=90, wrap=False)
    if(computeJdotE):
        JdotE = compute_energization(CEy_xy,dv)
        if(not(plotDiagJEOnly)):
            JdotE = compute_energization(CEx_xz,dv)
            if(plotFAC):
                if(plotAvg):
                    axs[2,0].set_title('$\overline{j_{\perp,1}} \cdot \overline{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[2,0].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i < 15} \cdot \widetilde{E_{\perp,1}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[2,0].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i > 15} \cdot \widetilde{E_{\perp,1}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[2,0].set_title('$\widetilde{j_{\perp,1}} \cdot \widetilde{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left') 
                else:
                    axs[2,0].set_title('$j_{\perp,1} \cdot E_{\perp,1}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[2,0].set_title('$\overline{j_y} \cdot \overline{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')             
                elif(plotFluc):
                    if(isLowPass):
                        axs[2,0].set_title('$\widetilde{j_y}^{k_{||} d_i < 15} \cdot \widetilde{E_y}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[2,0].set_title('$\widetilde{j_y}^{k_{||} d_i > 15} \cdot \widetilde{E_y}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[2,0].set_title('$\widetilde{j_y} \cdot \widetilde{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')             
                else:
                    axs[2,0].set_title('$j_y \cdot E_y$ = ' + "{:.2e}".format(JdotE),loc='left')
    
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
    axs[2,1].set_aspect('equal', 'box')
    axs[2,1].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEy_xz,dv)
        if(True):
            JdotE = compute_energization(CEy_xz,dv)
            if(plotFAC):
                if(plotAvg):
                    axs[2,1].set_title('$\overline{j_{\perp,1}} \cdot \overline{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[2,1].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i < 15} \cdot \widetilde{E_{\perp,1}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[2,1].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i > 15} \cdot \widetilde{E_{\perp,1}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[2,1].set_title('$\widetilde{j_{\perp,1}} \cdot \widetilde{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[2,1].set_title('$j_{\perp,1} \cdot E_{\perp,1}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[2,1].set_title('$\overline{j_y} \cdot \overline{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')  
                elif(plotFluc):
                    if(isLowPass):
                        axs[2,1].set_title('$\widetilde{j_y}^{k_{||} d_i < 15} \cdot \widetilde{E_y}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[2,1].set_title('$\widetilde{j_y}^{k_{||} d_i > 15} \cdot \widetilde{E_y}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[2,1].set_title('$\widetilde{j_y} \cdot \widetilde{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')             
                else:
                    axs[2,1].set_title('$j_y \cdot E_y$ = ' + "{:.2e}".format(JdotE),loc='left')

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
    axs[2,2].set_aspect('equal', 'box')
    axs[2,2].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEy_yz.T,dv)
        if(not(plotDiagJEOnly)):
            if(plotFAC):
                if(plotAvg):
                    axs[2,2].set_title('$\overline{j_{\perp,1}} \cdot \overline{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[2,2].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i < 15} \cdot \widetilde{E_{\perp,1}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[2,2].set_title('$\widetilde{j_{\perp,1}}^{k_{||} d_i > 15} \cdot \widetilde{E_{\perp,1}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[2,2].set_title('$\widetilde{j_{\perp,1}} \cdot \widetilde{E_{\perp,1}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[2,2].set_title('$j_{\perp,1} \cdot E_{\perp,1}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[2,2].set_title('$\overline{j_y} \cdot \overline{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[2,2].set_title('$\widetilde{j_y}^{k_{||} d_i < 15} \cdot \widetilde{E_y}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[2,2].set_title('$\widetilde{j_y}^{k_{||} d_i > 15} \cdot \widetilde{E_y}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[2,2].set_title('$\widetilde{j_y} \cdot \widetilde{E_y}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[2,2].set_title('$j_y \cdot E_y$ = ' + "{:.2e}".format(JdotE),loc='left')

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
    axs[3,0].set_aspect('equal', 'box')
    if(plotFAC):
        if(plotAvg):
            axs[3,0].text(-vmax*2.2,0, r"$\overline{C_{E_{\perp,2}}}$", ha='center', rotation=90, wrap=False)
        elif(plotFluc):
            if(isLowPass):
                axs[3,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,2}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
            elif(isHighPass):
                axs[3,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,2}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
            else:
                axs[3,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{\perp,2}}}$", ha='center', rotation=90, wrap=False)
        else:
            axs[3,0].text(-vmax*2.2,0, r"$C_{E_{\perp,2}}$", ha='center', rotation=90, wrap=False)
    else:
        if(plotAvg):
            axs[3,0].text(-vmax*2.2,0, r"$\overline{C_{E_{z}}}$", ha='center', rotation=90, wrap=False)
        elif(plotFluc):
            if(isLowPass):
                axs[3,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{z}}}^{k_{||} d_i < 15}$", ha='center', rotation=90, wrap=False)
            elif(isHighPass):
                axs[3,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{z}}}^{k_{||} d_i > 15}$", ha='center', rotation=90, wrap=False)
            else:
                axs[3,0].text(-vmax*2.2,0, r"$\widetilde{C_{E_{z}}}$", ha='center', rotation=90, wrap=False)
        else:
            axs[3,0].text(-vmax*2.2,0, r"$C_{E_{z}}$", ha='center', rotation=90, wrap=False)
    axs[3,0].grid()
    if(metadata != None):
        axs[3,0].text(-vmax*2.6,0, metadata, ha='center', rotation=90, wrap=False)
    if(computeJdotE):
        JdotE = compute_energization(CEz_xy,dv)
        if(True):
            if(plotFAC):
                if(plotAvg):
                    axs[3,0].set_title('$\overline{j_{\perp,2}} \cdot \overline{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[3,0].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i < 15} \cdot \widetilde{E_{\perp,2}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[3,0].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i > 15} \cdot \widetilde{E_{\perp,2}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[3,0].set_title('$\widetilde{j_{\perp,2}} \cdot \widetilde{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[3,0].set_title('$j_{\perp,2} \cdot E_{\perp,2}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[3,0].set_title('$\overline{j_z} \cdot \overline{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[3,0].set_title('$\widetilde{j_z}^{k_{||} d_i < 15} \cdot \widetilde{E_z}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[3,0].set_title('$\widetilde{j_z}^{k_{||} d_i > 15} \cdot \widetilde{E_z}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[3,0].set_title('$\widetilde{j_z} \cdot \widetilde{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[3,0].set_title('$j_z \cdot E_z$ = ' + "{:.2e}".format(JdotE),loc='left')

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
    axs[3,1].set_aspect('equal', 'box')
    axs[3,1].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEz_xz,dv)
        if(not(plotDiagJEOnly)):
            if(plotFAC):
                if(plotAvg):
                    axs[3,1].set_title('$\overline{j_{\perp,2}} \cdot \overline{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[3,1].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i < 15} \cdot \widetilde{E_{\perp,2}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[3,1].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i > 15} \cdot \widetilde{E_{\perp,2}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[3,1].set_title('$\widetilde{j_{\perp,2}} \cdot \widetilde{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[3,1].set_title('$j_{\perp,2} \cdot E_{\perp,2}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[3,1].set_title('$\overline{j_z} \cdot \overline{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[3,1].set_title('$\widetilde{j_z}^{k_{||} d_i < 15} \cdot \widetilde{E_z}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[3,1].set_title('$\widetilde{j_z}^{k_{||} d_i > 15} \cdot \widetilde{E_z}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[3,1].set_title('$\widetilde{j_z} \cdot \widetilde{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[3,1].set_title('$j_z \cdot E_z$ = ' + "{:.2e}".format(JdotE),loc='left')

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
    axs[3,2].set_aspect('equal', 'box')
    axs[3,2].grid()
    if(computeJdotE):
        JdotE = compute_energization(CEz_yz.T,dv)
        if(not(plotDiagJEOnly)):
            if(plotFAC):
                if(plotAvg):
                    axs[3,2].set_title('$\overline{j_{\perp,2}} \cdot \overline{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[3,2].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i < 15} \cdot \widetilde{E_{\perp,2}}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[3,2].set_title('$\widetilde{j_{\perp,2}}^{k_{||} d_i > 15} \cdot \widetilde{E_{\perp,2}}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[3,2].set_title('$\widetilde{j_{\perp,2}} \cdot \widetilde{E_{\perp,2}}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[3,2].set_title('$j_{\perp,2} \cdot E_{\perp,2}$ = ' + "{:.2e}".format(JdotE),loc='left')
            else:
                if(plotAvg):
                    axs[3,2].set_title('$\overline{j_z} \cdot \overline{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                elif(plotFluc):
                    if(isLowPass):
                        axs[3,2].set_title('$\widetilde{j_z}^{k_{||} d_i < 15} \cdot \widetilde{E_z}^{k_{||} d_i < 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    elif(isHighPass):
                        axs[3,2].set_title('$\widetilde{j_z}^{k_{||} d_i > 15} \cdot \widetilde{E_z}^{k_{||} d_i > 15}$ = ' + "{:.2e}".format(JdotE),loc='left')
                    else:
                        axs[3,2].set_title('$\widetilde{j_z} \cdot \widetilde{E_z}$ = ' + "{:.2e}".format(JdotE),loc='left')
                else:
                    axs[3,2].set_title('$j_z \cdot E_z$ = ' + "{:.2e}".format(JdotE),loc='left')

    clrbar32 = plt.colorbar(im32, ax=axs[3,2])#,format='%.1e')
    if(not(plotLog)):
        clrbar32.formatter.set_powerlimits((0, 0))

    for _i in range(0,4):
        for _j in range(0,3):
            axs[_i,_j].set_xlim(-vmax,vmax)
            axs[_i,_j].set_ylim(-vmax,vmax)

    #set ticks
    intvl = 1.
    if(vmax > 5):
        intvl = 5.
    if(vmax > 10):
        intvl = 10.
    tcks = np.arange(0,vmax,intvl)
    tcks = np.concatenate((-1*np.flip(tcks),tcks))
    for _i in range(0,4):
        for _j in range(0,3):
            axs[_i,_j].set_xticks(tcks)
            axs[_i,_j].set_yticks(tcks)

    #plt.subplots_adjust(hspace=.5,wspace=-.3)


    maxplotvval = np.max(vz_yz) #Assumes square grid of even size
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=250,bbox_inches='tight')
        
        #must make figure first to grab x10^val on top of color bar- after grabbing it, we can move it- a little wasteful but it was quick solution
        clrbar10text = str(clrbar10.ax.yaxis.get_offset_text().get_text())
        clrbar10.ax.yaxis.get_offset_text().set_visible(False)
        axs[1,0].text(1.5*maxplotvval,-1.3*maxplotvval,clrbar10text, va='bottom', ha='center')
        clrbar11text = str(clrbar11.ax.yaxis.get_offset_text().get_text())
        clrbar11.ax.yaxis.get_offset_text().set_visible(False) 
        axs[1,1].text(1.5*maxplotvval,-1.3*maxplotvval,clrbar11text, va='bottom', ha='center')
        clrbar12text = str(clrbar12.ax.yaxis.get_offset_text().get_text())
        clrbar12.ax.yaxis.get_offset_text().set_visible(False)
        axs[1,2].text(1.5*maxplotvval,-1.3*maxplotvval,clrbar12text, va='bottom', ha='center')
        clrbar20text = str(clrbar20.ax.yaxis.get_offset_text().get_text())
        clrbar20.ax.yaxis.get_offset_text().set_visible(False)
        axs[2,0].text(1.5*maxplotvval,-1.3*maxplotvval,clrbar20text, va='bottom', ha='center')
        clrbar21text = str(clrbar21.ax.yaxis.get_offset_text().get_text())
        clrbar21.ax.yaxis.get_offset_text().set_visible(False)
        axs[2,1].text(1.5*maxplotvval,-1.3*maxplotvval,clrbar21text, va='bottom', ha='center')
        clrbar22text = str(clrbar22.ax.yaxis.get_offset_text().get_text())
        clrbar22.ax.yaxis.get_offset_text().set_visible(False)
        axs[2,2].text(1.5*maxplotvval,-1.3*maxplotvval,clrbar22text, va='bottom', ha='center')
        clrbar30text = str(clrbar30.ax.yaxis.get_offset_text().get_text())
        clrbar30.ax.yaxis.get_offset_text().set_visible(False)
        axs[3,0].text(1.5*maxplotvval,-1.3*maxplotvval,clrbar30text, va='bottom', ha='center')
        clrbar31text = str(clrbar31.ax.yaxis.get_offset_text().get_text())
        clrbar31.ax.yaxis.get_offset_text().set_visible(False)
        axs[3,1].text(1.5*maxplotvval,-1.3*maxplotvval,clrbar31text, va='bottom', ha='center')
        clrbar32text = str(clrbar32.ax.yaxis.get_offset_text().get_text())
        clrbar32.ax.yaxis.get_offset_text().set_visible(False)
        axs[3,2].text(1.5*maxplotvval,-1.3*maxplotvval,clrbar32text, va='bottom', ha='center')


        #clrbar10text = str(clrbar10.ax.yaxis.get_offset_text().get_text())
        #clrbar10.ax.yaxis.get_offset_text().set_visible(False)
        #clrbar10.ax.text(1.02, -1.02, clrbar10text, va='bottom', ha='center')
        #clrbar11text = str(clrbar11.ax.yaxis.get_offset_text().get_text())
        #clrbar11.ax.yaxis.get_offset_text().set_visible(False)
        #clrbar11.ax.text(1.02, -1.02, clrbar11text, va='bottom', ha='center')
        #clrbar12text = str(clrbar12.ax.yaxis.get_offset_text().get_text())
        #clrbar12.ax.yaxis.get_offset_text().set_visible(False)
        #clrbar12.ax.text(1.02, -1.02, clrbar12text, va='bottom', ha='center')
        #clrbar20text = str(clrbar20.ax.yaxis.get_offset_text().get_text())
        #clrbar20.ax.yaxis.get_offset_text().set_visible(False)
        #clrbar20.ax.text(1.02, -1.02, clrbar20text, va='bottom', ha='center')
        #clrbar21text = str(clrbar21.ax.yaxis.get_offset_text().get_text())
        #clrbar21.ax.yaxis.get_offset_text().set_visible(False)
        #clrbar21.ax.text(1.02, -1.02, clrbar21text, va='bottom', ha='center')
        #clrbar22text = str(clrbar22.ax.yaxis.get_offset_text().get_text())
        #clrbar22.ax.yaxis.get_offset_text().set_visible(False)
        #clrbar22.ax.text(1.02, -1.02, clrbar22text, va='bottom', ha='center')
        #clrbar30text = str(clrbar30.ax.yaxis.get_offset_text().get_text())
        #clrbar30.ax.yaxis.get_offset_text().set_visible(False)
        #clrbar30.ax.text(1.02, -1.02, clrbar30text, va='bottom', ha='center')
        #clrbar31text = str(clrbar31.ax.yaxis.get_offset_text().get_text())
        #clrbar31.ax.yaxis.get_offset_text().set_visible(False)
        #clrbar31.ax.text(1.02, -1.02, clrbar31text, va='bottom', ha='center')
        #clrbar32text = str(clrbar32.ax.yaxis.get_offset_text().get_text())
        #clrbar32.ax.yaxis.get_offset_text().set_visible(False)
        #clrbar32.ax.text(1.02, -1.02, clrbar32text, va='bottom', ha='center')
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
                axs[0].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{||,'+sval+'}}^{k_{||} d_i < 15} \cdot \widetilde{E}_{||}^{k_{||} d_i < 15} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
            elif(isHighPass):
                axs[0].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{||,'+sval+'}}^{k_{||} d_i < 15} \cdot \widetilde{E}_{||}^{k_{||} d_i > 15} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
            else:
                axs[0].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{||,'+sval+'}} \cdot \widetilde{E}_{||} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
        elif(plotAvg):
            axs[0].text(-vmax*0.85,vmax*0.75,r'$\overline{j_{||,'+sval+'}} \cdot \overline{E}_{||} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
        else:
            axs[0].text(-vmax*0.85,vmax*0.75,r'$j_{||,'+sval+'} \cdot E_{||} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
       
        if(plotFluc):
            if(isLowPass):
                axs[1].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,'+sval+'}}^{k_{||} d_i < 15} \cdot \widetilde{E}_{\perp}^{k_{||} d_i < 15} = $'+ "{:.2e}".format(JdotEperp),fontsize=12)
            elif(isHighPass):
                axs[1].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,'+sval+'}}^{k_{||} d_i < 15} \cdot \widetilde{E}_{\perp}^{k_{||} d_i > 15} = $'+ "{:.2e}".format(JdotEperp),fontsize=12)
            else:
                axs[1].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,'+sval+'}} \cdot \widetilde{E}_{\perp} = $'+ "{:.2e}".format(JdotEperp),fontsize=12)
        elif(plotAvg):
            axs[1].text(-vmax*0.85,vmax*0.75,r'$\overline{j_{\perp,'+sval+'}} \cdot \overline{E}_{\perp} = $'+ "{:.2e}".format(JdotEperp),fontsize=12)
        else:
            axs[1].text(-vmax*0.85,vmax*0.75,r'$j_{\perp,'+sval+'} \cdot E_{\perp} = $'+ "{:.2e}".format(JdotEperp),fontsize=12)

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
                axs[0].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{||,'+sval+'}}^{k_{||} d_i < 15} \cdot \widetilde{E}_{||}^{k_{||} d_i < 15} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
            elif(isHighPass):
                axs[0].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{||,'+sval+'}}^{k_{||} d_i < 15} \cdot \widetilde{E}_{||}^{k_{||} d_i > 15} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
            else:
                axs[0].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{||,'+sval+'}} \cdot \widetilde{E}_{||} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
        elif(plotAvg):
            axs[0].text(-vmax*0.85,vmax*0.75,r'$\overline{j_{||,'+sval+'}} \cdot \overline{E}_{||} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)
        else:
            axs[0].text(-vmax*0.85,vmax*0.75,r'$j_{||,'+sval+'} \cdot E_{||} = $'+ "{:.2e}".format(JdotEpar),fontsize=12)

        if(plotFluc):
            if(isLowPass):
                axs[1].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,1,'+sval+'}}^{k_{||} d_i < 15} \cdot \widetilde{E}_{\perp,1}^{k_{||} d_i < 15} = $'+ "{:.2e}".format(JdotEperp1),fontsize=12)
            elif(isHighPass):
                axs[1].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,1,'+sval+'}}^{k_{||} d_i < 15} \cdot \widetilde{E}_{\perp,1}^{k_{||} d_i > 15} = $'+ "{:.2e}".format(JdotEperp1),fontsize=12)
            else:
                axs[1].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,1,'+sval+'}} \cdot \widetilde{E}_{\perp,1} = $'+ "{:.2e}".format(JdotEperp1),fontsize=12)
        elif(plotAvg):
            axs[1].text(-vmax*0.85,vmax*0.75,r'$\overline{j_{\perp,1,'+sval+'}} \cdot \overline{E}_{\perp,1} = $'+ "{:.2e}".format(JdotEperp1),fontsize=12)
        else:
            axs[1].text(-vmax*0.85,vmax*0.75,r'$j_{\perp,1,'+sval+'} \cdot E_{\perp,1} = $'+ "{:.2e}".format(JdotEperp1),fontsize=12)

        if(plotFluc):
            if(isLowPass):
                axs[2].text(-vmax*0.85,vmax*0.7,r'$\widetilde{j_{\perp,2,'+sval+'}}^{k_{||} d_i < 15} \cdot \widetilde{E}_{\perp,2}^{k_{||} d_i < 15} = $'+ "{:.2e}".format(JdotEperp2),fontsize=12)
            elif(isHighPass):
                axs[2].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,2,'+sval+'}}^{k_{||} d_i < 15} \cdot \widetilde{E}_{\perp,2}^{k_{||} d_i > 15} = $'+ "{:.2e}".format(JdotEperp2),fontsize=12)
            else:
                axs[2].text(-vmax*0.85,vmax*0.75,r'$\widetilde{j_{\perp,2,'+sval+'}} \cdot \widetilde{E}_{\perp,2} = $'+ "{:.2e}".format(JdotEperp2),fontsize=12)
        elif(plotAvg):
            axs[2].text(-vmax*0.85,vmax*0.75,r'$\overline{j_{\perp,2,'+sval+'}} \cdot \overline{E}_{\perp,2} = $'+ "{:.2e}".format(JdotEperp2),fontsize=12)
        else:
            axs[2].text(-vmax*0.85,vmax*0.75,r'$j_{\perp,2,'+sval+'} \cdot E_{\perp,2} = $'+ "{:.2e}".format(JdotEperp2),fontsize=12)

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

