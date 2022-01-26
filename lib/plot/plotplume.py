# plume.py>

# functions to plotting things related to the PLUME solve and other theory related plots

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_sweep(plume_sweeps,xaxiskey,yaxiskey,wavemodes=[''],xlbl='',ylbl='',lbls=[''],xlim=None,ylim=None,flnm=''):
    """
    WARNING: does NOT check if plume sweeps are different solutions to the same dispersion relation. Assumes they all are.
    """

    from lib.plot.resultsmanager import plume_keyname_to_plotname

    plt.figure()
    for idx, p_swp in enumerate(plume_sweeps):
        if(lbls != ['']):
            plt.semilogx(p_swp[xaxiskey],p_swp[yaxiskey],label=lbls[idx])
        else:
            plt.semilogx(p_swp[xaxiskey],p_swp[yaxiskey])

        if(xlbl == ''):
            xlbl = plume_keyname_to_plotname(xaxiskey)
            if(xlbl != ''):
                plt.xlabel(xlbl)
            else:
                plt.xlabel(xaxiskey)
        else:
            plt.xlabel(xlbl)

        if(ylbl == ''):
            ylbl = plume_keyname_to_plotname(yaxiskey)
            if(ylbl != ''):
                plt.ylabel(ylbl)
            else:
                plt.ylabel(yaxiskey)
        else:
            plt.ylabel(ylbl)

        if(lbls != ['']):
            plt.legend(prop={'size': 9})
        if(xlim != None):
            plt.xlim(xlim[0], xlim[1])
        if(ylim != None):
            plt.ylim(ylim[0], ylim[1])
    if(wavemodes !=['']):
        for wm in wavemodes:
            strict_tol = .001
            loose_tol = .1
            if(np.abs(np.linalg.norm(wm['Eperp1'])-1.) > strict_tol):
                print('WARNING: wavemodes was not normalized...')
            if(np.abs(wm['kperp2']) > strict_tol):
                print('WARNING: wavemode is not in the correct coordinate system...')
                print('kperp2 = ' + str(wm['kperp2']) + ' which is above the tolerance for approximately zero...')
            if(xaxiskey=='kpar' and np.abs(wm['kperp']-p_swp['kperp'][0]) > loose_tol): #we use a loose tolerance as sweeps are often run at kperps rounded to nearest tenth
                print('WARNING: wavemode has a different kperp value than sweep..')

            if(yaxiskey == 'exr'):
                plt.scatter([wm['kpar']],[wm['Eperp1'].real])
            if(yaxiskey == 'exi'):
                plt.scatter([wm['kpar']],[wm['Eperp1'].imag])
            if(yaxiskey == 'eyr'):
                plt.scatter([wm['kpar']],[wm['Eperp2'].real])
            if(yaxiskey == 'eyi'):
                plt.scatter([wm['kpar']],[wm['Eperp2'].imag])
            if(yaxiskey == 'ezr'):
                plt.scatter([wm['kpar']],[wm['Epar'].real])
            if(yaxiskey == 'ezi'):
                plt.scatter([wm['kpar']],[wm['Epar'].imag])

            if(yaxiskey == 'bxr'):
                bxr = wm['Bperp1'].real/wm['vth']
                plt.scatter([wm['kpar']],[bxr])
            if(yaxiskey == 'bxi'):
                bxi = wm['Bperp1'].imag/wm['vth']
                plt.scatter([wm['kpar']],[bxi])
            if(yaxiskey == 'byr'):
                byr = wm['Bperp2'].real/wm['vth']
                plt.scatter([wm['kpar']],[byr])
            if(yaxiskey == 'byi'):
                byi = wm['Bperp2'].imag/wm['vth']
                plt.scatter([wm['kpar']],[byi])
            if(yaxiskey == 'bzr'):
                bzr = wm['Bpar'].real/wm['vth']
                plt.scatter([wm['kpar']],[bzr])
            if(yaxiskey == 'bzi'):
                bzi = wm['Bpar'].imag/wm['vth']
                plt.scatter([wm['kpar']],[bzi])

            if(yaxiskey == 'ux1r'):
                plt.scatter([wm['kpar']],[wm['Ux'].real])
            if(yaxiskey == 'ux1i'):
                plt.scatter([wm['kpar']],[wm['Ux'].imag])
            if(yaxiskey == 'uy1r'):
                plt.scatter([wm['kpar']],[wm['Uy'].real])
            if(yaxiskey == 'uy1i'):
                plt.scatter([wm['kpar']],[wm['Uy'].imag])
            if(yaxiskey == 'uz1r'):
                plt.scatter([wm['kpar']],[wm['Uz'].real])
            if(yaxiskey == 'uz1i'):
                plt.scatter([wm['kpar']],[wm['Uz'].imag])
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=600,bbox_inches="tight")
    else:
        plt.show()

#todo: make sure given wavemodes are close to give kpars
def plot_wavemodes_and_compare_to_sweeps_kperp(kpars,beta_i,tau,wavemodes_matching_kpar,kperplim = [.1,10], flnm = '',delta_beta_i = 0, delta_tau = 0):
    from lib.plume import get_freq_from_wvmd
    from lib.plume import kaw_curve
    from lib.plume import fastmagson_curve
    from lib.plume import slowmagson_curve
    from lib.plume import whistler_curve

    kperps = np.linspace(kperplim[0],kperplim[1],1000)
    kawcrvs = []
    fastcrvs = []
    slowcrvs = []
    whicrvs = []
    #plot theoretical curves
    for kpar in kpars:
        kawcrv = []
        fastcrv = []
        slowcrv = []
        whicrv = []
        for kperp in kperps:
            kawcrv.append(kaw_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            fastcrv.append(fastmagson_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            slowcrv.append(slowmagson_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            whicrv.append(whistler_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
        kawcrvs.append(kawcrv)
        fastcrvs.append(fastcrv)
        slowcrvs.append(slowcrv)
        whicrvs.append(whicrv)

    plotkperps = []
    plotkperp_errors = []
    omegas = []
    omega_errors = []
    #grab points and compute error for each wavemode
    for match_list in wavemodes_matching_kpar:
        plotkperp = []
        plotkperp_error = []
        omega = []
        omega_error = []
        for wvmd in match_list['wavemodes']:
            _,_,omega_faradayreal,_,_,_ = get_freq_from_wvmd(wvmd,comp_error_prop=True)
            plotkperp.append(wvmd['kperp'])
            plotkperp_error.append(wvmd['delta_kperp1'])
            omega.append(omega_faradayreal.n)
            omega_error.append(omega_faradayreal.s)

        plotkperps.append(plotkperp)
        plotkperp_errors.append(plotkperp_error)
        omegas.append(omega)
        omega_errors.append(omega_error)

    if(len(kpars) != 3):
        print('Error, this function is set up to plot 3 curves (per wavemode) only... TODO: generalize this')
        return

    linestyle = ['--',':','-']
    lnwidth = 1.75

    plt.figure(figsize=(10,10))
    for i in range(0,len(kawcrvs)):
        plt.errorbar(plotkperps[i],np.real(omegas[i]), xerr = plotkperp_errors[i], yerr=omega_errors[i], fmt="o",color='C0')
        plt.plot(kperps,kawcrvs[i],linestyle[i],color='black',linewidth=lnwidth)
        plt.plot(kperps,fastcrvs[i],linestyle[i],color='blue',linewidth=lnwidth)
        plt.plot(kperps,slowcrvs[i],linestyle[i],color='green',linewidth=lnwidth)
        plt.plot(kperps,whicrvs[i],linestyle[i],color='red',linewidth=lnwidth)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$k_{\perp} d_i$')
    plt.ylabel('$\omega / \Omega_i$')
    plt.grid(True, which="both", ls="-")
    plt.axis('scaled')
    plt.ylim(.1,10)
    plt.xlim(.1,10)
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=600,bbox_inches="tight")
    else:
        plt.show()

#TODO: make sure given wavemodes are close to given kperps
def plot_wavemodes_and_compare_to_sweeps_kpar(kperps,beta_i,tau,wavemodes_matching_kpar,kparlim = [.1,10], flnm = '',delta_beta_i = 0, delta_tau = 0):
    from lib.plume import get_freq_from_wvmd
    from lib.plume import kaw_curve
    from lib.plume import fastmagson_curve
    from lib.plume import slowmagson_curve
    from lib.plume import whistler_curve

    kpars = np.linspace(kparlim[0],kparlim[1],1000)
    kawcrvs = []
    fastcrvs = []
    slowcrvs = []
    whicrvs = []
    #plot theoretical curves
    for kperp in kperps:
        kawcrv = []
        fastcrv = []
        slowcrv = []
        whicrv = []
        for kpar in kpars:
            kawcrv.append(kaw_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            fastcrv.append(fastmagson_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            slowcrv.append(slowmagson_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            whicrv.append(whistler_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
        kawcrvs.append(kawcrv)
        fastcrvs.append(fastcrv)
        slowcrvs.append(slowcrv)
        whicrvs.append(whicrv)

    plotkpars = []
    plotkpar_errors = []
    omegas = []
    omega_errors = []
    #grab points and compute error for each wavemode
    for match_list in wavemodes_matching_kpar:
        plotkpar = []
        plotkpar_error = []
        omega = []
        omega_error = []
        for wvmd in match_list['wavemodes']:
            _,_,omega_faradayreal,_,_,_ = get_freq_from_wvmd(wvmd,comp_error_prop=True)
            plotkpar.append(wvmd['kpar'])
            plotkpar_error.append(wvmd['delta_kpar'])
            omega.append(omega_faradayreal.n)
            omega_error.append(omega_faradayreal.s)

        plotkpars.append(plotkpar)
        plotkpar_errors.append(plotkpar_error)
        omegas.append(omega)
        omega_errors.append(omega_error)

    if(len(kperps) != 3):
        print('Error, this function is set up to plot 3 curves (per wavemode) only... TODO: generalize this')
        return

    linestyle = ['--',':','-']
    lnwidth = 1.75

    plt.figure(figsize=(10,10))
    for i in range(0,len(kawcrvs)):
        plt.errorbar(plotkpars[i],np.real(omegas[i]), xerr = plotkpar_errors[i], yerr=omega_errors[i], fmt="o",color='C0')
        plt.plot(kpars,kawcrvs[i],linestyle[i],color='black',linewidth=lnwidth)
        plt.plot(kpars,fastcrvs[i],linestyle[i],color='blue',linewidth=lnwidth)
        plt.plot(kpars,slowcrvs[i],linestyle[i],color='green',linewidth=lnwidth)
        plt.plot(kpars,whicrvs[i],linestyle[i],color='red',linewidth=lnwidth)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$k_{||} d_i$')
    plt.ylabel('$\omega / \Omega_i$')
    plt.grid(True, which="both", ls="-")
    plt.axis('scaled')
    plt.ylim(.1,10)
    plt.xlim(.1,10)
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=600,bbox_inches="tight")
    else:
        plt.show()
