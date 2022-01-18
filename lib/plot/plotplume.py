# plume.py>

# functions to plotting plume results

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

def plot_kperp_disp_sweeps(kperpsweep,wavemodes_matching_kpar,kaw_curves_matching_kpar,fm_curves_matching_kpar,slow_curves_matching_kpar,whi_curves_matching_kpar,uncertainty=.5,flnm='',beta_i = 1. , tau = 1.):
    """
    WARNING: beta_i and tau should match beta_i and tau use for select_wavemodes_and_compute_curves
    """
    from lib.plume import get_freq_from_wvmd
    from lib.plume import kaw_curve


    #compute kpar and omega for each wavemodes
    omegas = []
    omega_errors = []
    pltkperps = []
    pltkperp_errors = []
    for k in range(0,len(wavemodes_matching_kpar)):
        _omegarow = []
        _omega_errorrow = []
        _kperprow = []
        _kperp_errorrow = []
        for wvmd in wavemodes_matching_kpar[k]['wavemodes']:
            _,omega2,_ = get_freq_from_wvmd(wvmd)
            _omegarow.append(omega2)
            _omega_errorrow.append(kaw_curve(wvmd['kperp'],wvmd['kpar'],comp_error_prop=True,uncertainty = uncertainty, beta_i = beta_i, tau = tau).s)     #WARNING: we use KAW disp relation to compute error propogation
            _kperprow.append(wvmd['kperp'])
            _kperp_errorrow.append(wvmd['kperp']*uncertainty)
        omegas.append(_omegarow)
        omega_errors.append(_omega_errorrow)
        pltkperps.append(_kperprow)
        pltkperp_errors.append(_kperp_errorrow)

    if(len(kaw_curves_matching_kpar) != 3):
        print('Error, this function is set up to plot 3 curves (per wavemode) only... TODO: generalize this')
        return

    linestyle = ['--',':','-']
    lnwidth = 1.75

    plt.figure(figsize=(10,10))
    for i in range(0,len(kaw_curves_matching_kpar)):
        plt.errorbar(pltkperps[i],np.real(omegas[i]), xerr = pltkperp_errors[i], yerr=omega_errors[i], fmt="o",color='C0')
        plt.plot(kperpsweep,kaw_curves_matching_kpar[i],linestyle[i],color='black',linewidth=lnwidth)
        plt.plot(kperpsweep,fm_curves_matching_kpar[i],linestyle[i],color='blue',linewidth=lnwidth)
        plt.plot(kperpsweep,slow_curves_matching_kpar[i],linestyle[i],color='green',linewidth=lnwidth)
        plt.plot(kperpsweep,whi_curves_matching_kpar[i],linestyle[i],color='red',linewidth=lnwidth)

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

def plot_kpar_disp_sweeps(kparsweep,wavemodes_matching_kperp,kaw_curves_matching_kperp,fm_curves_matching_kperp,slow_curves_matching_kperp,whi_curves_matching_kperp,uncertainty=.5,flnm='',beta_i=1.,tau=1.):
    """
    WARNING: beta_i and tau should match beta_i and tau use for select_wavemodes_and_compute_curves
    """
    from lib.plume import get_freq_from_wvmd
    from lib.plume import kaw_curve


    #compute kpar and omega for each wavemodes
    omegas = []
    omega_errors = []
    pltkpars = []
    pltkpar_errors = []
    for k in range(0,len(wavemodes_matching_kperp)):
        _omegarow = []
        _omega_errorrow = []
        _kparrow = []
        _kpar_errorrow = []
        for wvmd in wavemodes_matching_kperp[k]['wavemodes']:
            _,omega2,_ = get_freq_from_wvmd(wvmd)
            _omegarow.append(omega2)
            _omega_errorrow.append(kaw_curve(wvmd['kperp'],wvmd['kpar'],comp_error_prop=True,uncertainty = uncertainty, beta_i = beta_i, tau = tau).s,)     #WARNING: we use KAW disp relation to compute error propogation
            _kparrow.append(wvmd['kpar'])
            _kpar_errorrow.append(wvmd['kpar']*uncertainty)
        omegas.append(_omegarow)
        omega_errors.append(_omega_errorrow)
        pltkpars.append(_kparrow)
        pltkpar_errors.append(_kpar_errorrow)

    if(len(kaw_curves_matching_kperp) != 3):
        print('Error, this function is set up to plot 3 curves (per wavemode) only... TODO: generalize this')
        return

    linestyle = ['-',':','--']
    lnwidth = 1.75

    plt.figure(figsize=(10,10))
    for i in range(0,len(kaw_curves_matching_kperp)):
        plt.errorbar(pltkpars[i],np.real(omegas[i]), xerr = pltkpar_errors[i], yerr=omega_errors[i], fmt="o",color='C0')
        plt.plot(kparsweep,kaw_curves_matching_kperp[i],linestyle[i],color='black',linewidth=lnwidth)
        plt.plot(kparsweep,fm_curves_matching_kperp[i],linestyle[i],color='blue',linewidth=lnwidth)
        plt.plot(kparsweep,slow_curves_matching_kperp[i],linestyle[i],color='green',linewidth=lnwidth)
        plt.plot(kparsweep,whi_curves_matching_kperp[i],linestyle[i],color='red',linewidth=lnwidth)

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
