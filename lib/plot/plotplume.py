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

            if(yaxiskey == 'bxr'): #TODO: check normalization. Migth be b_i is normalizated to e_i. that is we should plot b_i/e_i
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
    plt.close()

def plot_sweep_norms(plume_sweep,xaxiskey,xlbl='',xlim=None,ylim=None,flnm='',axvlinex=None):
    """
    WARNING: does NOT check if plume sweeps are different solutions to the same dispersion relation. Assumes they all are.
    """

    from lib.plot.resultsmanager import plume_keyname_to_plotname

    fig,ax = plt.subplots(2,1,figsize=(16,8),sharex=True)
    field_keys = ['bxr','bxi','byr','byi','bzr','bzi','exr','exi','eyr','eyi','ezr','ezi']
    lbls = [r'$\hat{B}_{\perp,1}$',r'$\hat{B}_{\perp,1}$',r'$\hat{B}_{\perp,2}$',r'$\hat{B}_{\perp,2}$',r'$\hat{B}_{||}$',r'$\hat{B}_{||}$',r'$\hat{E}_{\perp,1}$',r'$\hat{E}_{\perp,1}$',r'$\hat{E}_{\perp,2}$',r'$\hat{E}_{\perp,2}$',r'$\hat{E}_{||}$',r'$\hat{E}_{||}$']
   
    for _idx in range(0,len(lbls)):
        lbls[_idx] += r'$/\hat{E}_{\perp,1}$'
        lbls[_idx] = r'$|$'+lbls[_idx] + r'$|$'

    colors = ['black','red','blue','yellow','gray','orange']
    lstyles = ['-','--','-.','..','--','-.']
    
    plot_arrs = []
    plot_lbls = []
    for _idx in range(0,len(field_keys),2):
        plot_arrs.append([np.linalg.norm([plume_sweep[field_keys[_idx]][_j],plume_sweep[field_keys[_idx+1]][_j]]) for _j in range(0,len(plume_sweep[field_keys[_idx]]))])
        plot_lbls.append(lbls[_idx])

    for _idx in range(0,3):
        ax[0].semilogx(plume_sweep[xaxiskey],plot_arrs[_idx],label=plot_lbls[_idx],linewidth=2.5,linestyle=lstyles[_idx],color=colors[_idx])
    for _idx in range(4,6):
        ax[1].semilogx(plume_sweep[xaxiskey],plot_arrs[_idx],label=plot_lbls[_idx],linewidth=2.5,linestyle=lstyles[_idx],color=colors[_idx])


#    for idx,yaxiskey in enumerate(field_keys):
#        if(yaxiskey[2] == 'r'):
#            ax[0].semilogx(plume_sweep[xaxiskey],plume_sweep[yaxiskey],label=lbls[idx],linewidth=2.5,linestyle=lstyles[idx],color=colors[idx])
#        else:
#            ax[1].semilogx(plume_sweep[xaxiskey],plume_sweep[yaxiskey],label=lbls[idx],linewidth=2.5,linestyle=lstyles[idx],color=colors[idx])

    if(xlbl == ''):
        xlbl = plume_keyname_to_plotname(xaxiskey)
        if(xlbl != ''):
            ax[1].set_xlabel(xlbl)
        else:
            ax[1].set_xlabel(xaxiskey)
    else:
        ax[1].set_xlabel(xlbl)

    #if(ylbl == ''):
    #    ylbl = plume_keyname_to_plotname(yaxiskey)
    #    if(ylbl != ''):
    #        plt.ylabel(ylbl)
    #    else:
    #        plt.ylabel(yaxiskey)
    #else:
    #    plt.ylabel(ylbl)

    if(axvlinex != None):
        ax[0].axvline(axvlinex,linewidth=.5,color='black')
        ax[1].axvline(axvlinex,linewidth=.5,color='black')

    ax[0].legend(prop={'size': 12})
    ax[1].legend(prop={'size': 12})
    if(xlim != None):
        ax[0].set_xlim(xlim[0], xlim[1])
        ax[1].set_xlim(xlim[0], xlim[1])
    if(ylim != None):
        ax[0].set_ylim(ylim[0], ylim[1])
        ax[1].set_ylim(ylim[0], ylim[1])
    fig.subplots_adjust(hspace=0)
    plt.setp(ax[0].get_xticklabels(), visible=True)
    ax[1].tick_params(axis='x',top=True,direction='inout',length=10,which='both')
    ax[0].grid()
    ax[1].grid()
    #ax[0].set_ylabel(r'$|\{\hat{B}_i\}|$')
    #ax[1].set_ylabel(r'$|\{\hat{E}_i\}|$')
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=600,bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def plot_sweep_field(plume_sweep,xaxiskey,xlbl='',xlim=None,ylim=None,flnm='',plotE=False,axvlinex=None):
    """
    WARNING: does NOT check if plume sweeps are different solutions to the same dispersion relation. Assumes they all are.
    """

    from lib.plot.resultsmanager import plume_keyname_to_plotname

    fig,ax = plt.subplots(2,1,figsize=(16,8),sharex=True)
    if(plotE == False):
        field_keys = ['bxr','bxi','byr','byi','bzr','bzi']
        lbls = [r'$\hat{B}_{x}$',r'$\hat{B}_{x}$',r'$\hat{B}_{y}$',r'$\hat{B}_{y}$',r'$\hat{B}_{z}$',r'$\hat{B}_{z}$']
    else:
        field_keys = ['exr','exi','eyr','eyi','ezr','ezi']
        lbls = [r'$\hat{E}_{x}$',r'$\hat{E}_{x}$',r'$\hat{E}_{y}$',r'$\hat{E}_{y}$',r'$\hat{E}_{z}$',r'$\hat{E}_{z}$']
    colors = ['black','black','red','red','blue','blue']
    lstyles = ['-','-','--','--','-.','-.']
    for idx,yaxiskey in enumerate(field_keys):
        if(yaxiskey[2] == 'r'):
            ax[0].semilogx(plume_sweep[xaxiskey],plume_sweep[yaxiskey],label=lbls[idx],linewidth=2.5,linestyle=lstyles[idx],color=colors[idx])
        else:
            ax[1].semilogx(plume_sweep[xaxiskey],plume_sweep[yaxiskey],label=lbls[idx],linewidth=2.5,linestyle=lstyles[idx],color=colors[idx])

    if(xlbl == ''):
        xlbl = plume_keyname_to_plotname(xaxiskey)
        if(xlbl != ''):
            ax[1].set_xlabel(xlbl)
        else:
            ax[1].set_xlabel(xaxiskey)
    else:
        ax[1].set_xlabel(xlbl)

    #if(ylbl == ''):
    #    ylbl = plume_keyname_to_plotname(yaxiskey)
    #    if(ylbl != ''):
    #        plt.ylabel(ylbl)
    #    else:
    #        plt.ylabel(yaxiskey)
    #else:
    #    plt.ylabel(ylbl)

    if(axvlinex != None):
        ax[0].axvline(axvlinex,linewidth=.5,color='black')
        ax[1].axvline(axvlinex,linewidth=.5,color='black')

    ax[0].legend(prop={'size': 9})
    if(xlim != None):
        ax[0].set_xlim(xlim[0], xlim[1])
        ax[1].set_xlim(xlim[0], xlim[1])
    if(ylim != None):
        ax[0].set_ylim(ylim[0], ylim[1])
    fig.subplots_adjust(hspace=0)
    plt.setp(ax[0].get_xticklabels(), visible=True)
    ax[1].tick_params(axis='x',top=True,direction='inout',length=10,which='both')
    ax[0].grid()
    ax[1].grid()
    if(plotE == False):
        ax[0].set_ylabel(r'Re$\{\hat{B}_i\}$')
        ax[1].set_ylabel(r'Im$\{\hat{B}_i\}$')
    else:
        ax[0].set_ylabel(r'Re$\{\hat{E}_i\}$')
        ax[1].set_ylabel(r'Im$\{\hat{E}_i\}$')
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=600,bbox_inches="tight")
    else:
        plt.show()
    plt.close()

#todo: make sure given wavemodes are close to give kpars
def plot_wavemodes_and_compare_to_sweeps_kperp(kpars,beta_i,tau,wavemodes_matching_kpar,kperplim = [.1,10], flnm = '',delta_beta_i = 0, delta_tau = 0,xlim=[],ylim=[],plot_testcrv=False,flip_omega_sign=False):
    from lib.plume import get_freq_from_wvmd
    from lib.plume import kaw_curve
    from lib.plume import fastmagson_curve
    from lib.plume import slowmagson_curve
    from lib.plume import whistler_curve
    from lib.plume import test_curve


    kperps = np.linspace(kperplim[0],kperplim[1],1000)
    kawcrvs = []
    kawcrv_errors = []
    fastcrvs = []
    fastcrv_errors = []
    slowcrvs = []
    slowcrv_errors = []
    whicrvs = []
    whicrv_errors = []
    testcrvs = []
    testcrv_errors  = []
    #plot theoretical curves
    for kpar in kpars:
        kawcrv = []
        kawcrv_error = []
        fastcrv = []
        fastcrv_error = []
        slowcrv = []
        slowcrv_error = []
        whicrv = []
        whicrv_error = []
        testcrv = []
        testcrv_error = []
        for kperp in kperps:
            kawcrv.append(kaw_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            kawcrv_error.append(kaw_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)
            fastcrv.append(fastmagson_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            fastcrv_error.append(fastmagson_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)
            slowcrv.append(slowmagson_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            slowcrv_error.append(slowmagson_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)
            whicrv.append(whistler_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            whicrv_error.append(whistler_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)
        
            testcrv.append(test_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            testcrv_error.append(test_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)

        kawcrvs.append(np.asarray(kawcrv))
        kawcrv_errors.append(np.asarray(kawcrv_error))
        fastcrvs.append(np.asarray(fastcrv))
        fastcrv_errors.append(fastcrv_error)
        slowcrvs.append(np.asarray(slowcrv))
        slowcrv_errors.append(slowcrv_error)
        whicrvs.append(np.asarray(whicrv))
        whicrv_errors.append(np.asarray(whicrv_error))
        testcrvs.append(np.asarray(testcrv))
        testcrv_errors.append(np.asarray(testcrv_error))

    plotkperps = []
    plotkperp_errors = []
    omegas = []
    omega_errors = []
    omegas0 = []
    omega0_errors = []
    omegas2 = []
    omega2_errors = []
    #grab points and compute error for each wavemode
    for match_list in wavemodes_matching_kpar:
        plotkperp = []
        plotkperp_error = []
        omega = []
        omega_error = []
        omega0 = []
        omega0_error = []
        omega2 = []
        omega2_error = []
        for wvmd in match_list['wavemodes']:
            omega_faradayreal0,_,omega_faradayreal,_,omega_faradayreal2,_ = get_freq_from_wvmd(wvmd,comp_error_prop=True)
            plotkperp.append(wvmd['kperp'])
            plotkperp_error.append(wvmd['delta_kperp1'])
            omega.append(omega_faradayreal.n)
            omega_error.append(omega_faradayreal.s)
            omega0.append(omega_faradayreal0.n)
            omega0_error.append(omega_faradayreal0.s)
            omega2.append(omega_faradayreal2.n)
            omega2_error.append(omega_faradayreal2.s)

        if(flip_omega_sign): #needed b/c we might have a sign error
            omega = -1.*np.asarray(omega)
            omega0 = -1.*np.asarray(omega0)
            omega2 = -1.*np.asarray(omega2)

        plotkperps.append(plotkperp)
        plotkperp_errors.append(plotkperp_error)
        omegas.append(omega)
        omega_errors.append(omega_error)
        omegas0.append(omega0)
        omega0_errors.append(omega0_error)
        omegas2.append(omega2)
        omega2_errors.append(omega2_error)

    #if(len(kpars) != 3):
        #print('Error, this function is set up to plot 3 curves (per wavemode) only... TODO: generalize this')
        #return

    linestyle = ['-',':','--','-.','.',',','-',':','--','-.','.',',','-',':','--','-.','.',',','-',':','--','-.','.',',']
    lnwidth = 1.75

    plt.figure(figsize=(8,8))
    for i in range(0,len(kawcrvs)):
        plt.errorbar(plotkperps[i],np.real(omegas[i]), xerr = plotkperp_errors[i], yerr=omega_errors[i], fmt="o",color='C0')
        plt.errorbar(plotkperps[i],np.real(omegas0[i]), xerr = plotkperp_errors[i], yerr=omega0_errors[i], fmt="s",color='C1')
        plt.errorbar(plotkperps[i],np.real(omegas2[i]), xerr = plotkperp_errors[i], yerr=omega2_errors[i], fmt='*',color='C3')
        plt.plot(kperps,kawcrvs[i],linestyle[i],color='black',linewidth=lnwidth,label='$k_{||}$='+str(format(kpars[i],'.2f')))
        plt.fill_between(kperps,kawcrvs[i]-kawcrv_errors[i],kawcrvs[i]+kawcrv_errors[i],alpha=.2,color='black')
        plt.plot(kperps,fastcrvs[i],linestyle[i],color='blue',linewidth=lnwidth)
        plt.fill_between(kperps,fastcrvs[i]-fastcrv_errors[i],fastcrvs[i]+fastcrv_errors[i],alpha=.2,color='blue')
        plt.plot(kperps,slowcrvs[i],linestyle[i],color='green',linewidth=lnwidth)
        plt.fill_between(kperps,slowcrvs[i]-slowcrv_errors[i],slowcrvs[i]+slowcrv_errors[i],alpha=.2,color='green')
        plt.plot(kperps,whicrvs[i],linestyle[i],color='red',linewidth=lnwidth)
        plt.fill_between(kperps,whicrvs[i]-whicrv_errors[i],whicrvs[i]+whicrv_errors[i],alpha=.2,color='red')
        if(plot_testcrv):
            plt.plot(kperps,testcrvs[i],linestyle[i],color='gray',linewidth=lnwidth)
            plt.fill_between(kperps,testcrvs[i]-testcrv_errors[i],testcrvs[i]+testcrv_errors[i],alpha=.2,color='gray')

    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.xlabel('$k_{\perp} d_i$')
    plt.ylabel('$\omega / \Omega_i$')
    plt.grid(True, which="both", ls="-")
    plt.axis('scaled')
    if(ylim == []):
        plt.ylim(.1,10)
    else:
        plt.ylim(ylim[0],ylim[1])
    if(xlim == []):
        plt.xlim(.1,10)
    else:
        plt.xlim(xlim[0],xlim[1])
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=600,bbox_inches="tight")
        plt.close()
    else:
        plt.show()

#TODO: make sure given wavemodes are close to given kperps
def plot_wavemodes_and_compare_to_sweeps_kperp(kpars,beta_i,tau,wavemodes_matching_kpar,kperplim = [.1,10], flnm = '',delta_beta_i = 0, delta_tau = 0,xlim=[],ylim=[],plot_testcrv=False,flip_omega_sign=False,extralines=[],extracolors=[],scalarfac = 1.):
    from lib.plume import get_freq_from_wvmd
    from lib.plume import kaw_curve
    from lib.plume import fastmagson_curve
    from lib.plume import slowmagson_curve
    from lib.plume import whistler_curve
    from lib.plume import test_curve

    if(scalarfac == 1.):
        print("WARNING: empirical curves plotted may be in a different normalization, as the frequency/ wavenumber computed from wavemode is normalized to the upstream parameters but empircal curves need local normalization")
        print("Please account scalarfact = btotlocaloverbtotup/(np.sqrt(ntotlocalovernup)) by passing optional parameter scalarfac = scalarfact")
        print("However, this factor is often approx 1")


    kperps = np.linspace(kperplim[0],kperplim[1],1000)
    kawcrvs = []
    kawcrv_errors = []
    fastcrvs = []
    fastcrv_errors = []
    slowcrvs = []
    slowcrv_errors = []
    whicrvs = []
    whicrv_errors = []
    testcrvs = []
    testcrv_errors  = []
    #plot theoretical curves
    for kpar in kpars:
        kawcrv = []
        kawcrv_error = []
        fastcrv = []
        fastcrv_error = []
        slowcrv = []
        slowcrv_error = []
        whicrv = []
        whicrv_error = []
        testcrv = []
        testcrv_error = []
        for kperp in kperps:
            kawcrv.append(kaw_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            kawcrv_error.append(kaw_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)
            fastcrv.append(fastmagson_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            fastcrv_error.append(fastmagson_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)
            slowcrv.append(slowmagson_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            slowcrv_error.append(slowmagson_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)
            whicrv.append(whistler_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            whicrv_error.append(whistler_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)

            testcrv.append(test_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            testcrv_error.append(test_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)

        kawcrvs.append(np.asarray(kawcrv))
        kawcrv_errors.append(np.asarray(kawcrv_error))
        fastcrvs.append(np.asarray(fastcrv))
        fastcrv_errors.append(fastcrv_error)
        slowcrvs.append(np.asarray(slowcrv))
        slowcrv_errors.append(slowcrv_error)
        whicrvs.append(np.asarray(whicrv))
        whicrv_errors.append(np.asarray(whicrv_error))
        testcrvs.append(np.asarray(testcrv))
        testcrv_errors.append(np.asarray(testcrv_error))

    plotkperps = []
    plotkperp_errors = []
    omegas = []
    omega_errors = []
    omegas0 = []
    omega0_errors = []
    omegas2 = []
    omega2_errors = []
    #grab points and compute error for each wavemode
    for match_list in wavemodes_matching_kpar:
        plotkperp = []
        plotkperp_error = []
        omega = []
        omega_error = []
        omega0 = []
        omega0_error = []
        omega2 = []
        omega2_error = []
        for wvmd in match_list['wavemodes']:
            omega_faradayreal0,tempgamma0,omega_faradayreal,tempgamma1,omega_faradayreal2,tempgamma2 = get_freq_from_wvmd(wvmd,comp_error_prop=True)

            print(wvmd)

            print("om0 gam0 om1 gam1 om2 gam2")
            print(omega_faradayreal0,tempgamma0,omega_faradayreal,tempgamma1,omega_faradayreal2,tempgamma2)

            plotkperp.append(wvmd['kperp'])
            plotkperp_error.append(wvmd['delta_kperp1'])
            omega.append(omega_faradayreal.n)
            omega_error.append(omega_faradayreal.s)
            omega0.append(omega_faradayreal0.n)
            omega0_error.append(omega_faradayreal0.s)
            omega2.append(omega_faradayreal2.n)
            omega2_error.append(omega_faradayreal2.s)

        if(flip_omega_sign): #needed b/c we might have a sign error
            print("did flip omega sign")
            omega = -1.*np.asarray(omega)
            omega0 = -1.*np.asarray(omega0)
            omega2 = -1.*np.asarray(omega2)
        else:
            print("did NOT flip omega sign")

        plotkperps.append(plotkperp)
        plotkperp_errors.append(plotkperp_error)
        omegas.append(omega)
        omega_errors.append(omega_error)
        omegas0.append(omega0)
        omega0_errors.append(omega0_error)
        omegas2.append(omega2)
        omega2_errors.append(omega2_error)

    #if(len(kpars) != 3):
        #print('Error, this function is set up to plot 3 curves (per wavemode) only... TODO: generalize this')
        #return

    linestyle = ['-',':','--','-.','.',',','-',':','--','-.','.',',','-',':','--','-.','.',',','-',':','--','-.','.',',']
    lnwidth = 1.75

    kawcrvs = np.asarray(kawcrvs)
    kawcrv_errors = np.asarray(kawcrv_errors)
    fastcrvs = np.asarray(fastcrvs)
    fastcrv_errors = np.asarray(fastcrv_errors)
    slowcrvs = np.asarray(slowcrvs)
    slowcrv_errors = np.asarray(slowcrv_errors)
    whicrvs = np.asarray(whicrvs)
    whicrv_errors = np.asarray(whicrv_errors)

    plt.figure(figsize=(6,6))
    for i in range(0,len(kawcrvs)):
        plt.errorbar(plotkperps[i],np.real(omegas[i]), xerr = plotkperp_errors[i], yerr=omega_errors[i], fmt="o",color='C0')
        plt.errorbar(plotkperps[i],np.real(omegas0[i]), xerr = plotkperp_errors[i], yerr=omega0_errors[i], fmt="s",color='C1')
        plt.errorbar(plotkperps[i],np.real(omegas2[i]), xerr = plotkperp_errors[i], yerr=omega2_errors[i], fmt='*',color='C3')
        plt.plot(kperps,scalarfac*kawcrvs[i],linestyle[i],color='black',linewidth=lnwidth,label='$k_{||}$='+str(format(kpars[i],'.2f')))
        plt.fill_between(kperps,scalarfac*kawcrvs[i]-scalarfac*kawcrv_errors[i],scalarfac*kawcrvs[i]+scalarfac*kawcrv_errors[i],alpha=.2,color='black')
        plt.plot(kperps,scalarfac*fastcrvs[i],linestyle[i],color='blue',linewidth=lnwidth)
        plt.fill_between(kperps,scalarfac*fastcrvs[i]-scalarfac*fastcrv_errors[i],scalarfac*fastcrvs[i]+scalarfac*fastcrv_errors[i],alpha=.2,color='blue')
        plt.plot(kperps,scalarfac*slowcrvs[i],linestyle[i],color='green',linewidth=lnwidth)
        plt.fill_between(kperps,scalarfac*slowcrvs[i]-scalarfac*slowcrv_errors[i],scalarfac*slowcrvs[i]+scalarfac*slowcrv_errors[i],alpha=.2,color='green')
        plt.plot(kperps,scalarfac*whicrvs[i],linestyle[i],color='red',linewidth=lnwidth)
        plt.fill_between(kperps,scalarfac*whicrvs[i]-scalarfac*whicrv_errors[i],scalarfac*whicrvs[i]+scalarfac*whicrv_errors[i],alpha=.2,color='red')
        if(plot_testcrv):
            plt.plot(kperps,testcrvs[i],linestyle[i],color='gray',linewidth=lnwidth)
            plt.fill_between(kperps,testcrvs[i]-testcrv_errors[i],testcrvs[i]+testcrv_errors[i],alpha=.2,color='gray')

    if(extralines != []):
        for _i in range(0,len(extralines[0])):
            print(_i)
            plt.plot(extralines[0][_i],extralines[1][_i],color=extracolors[_i],ls=':')

    plt.yscale('log')
    plt.xscale('log')
    plt.legend(prop={'size': 20})
    plt.xlabel('$k_{\perp} d_i$',fontsize=22)
    plt.ylabel('$\omega / \Omega_i$',fontsize=22)
    plt.gca().yaxis.set_tick_params(labelsize=18)
    plt.gca().xaxis.set_tick_params(labelsize=18)
    plt.grid(True, which="both", ls="-")
    plt.axis('scaled')
    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots
    if(ylim == []):
        plt.ylim(.1,10)
    else:
        plt.ylim(ylim[0],ylim[1])
    if(xlim == []):
        plt.xlim(.1,10)
    else:
        plt.xlim(xlim[0],xlim[1])
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=600,bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return omegas, omegas0, omegas2, plotkperps, omega_errors

#TODO: make sure given wavemodes are close to given kperps
def plot_wavemodes_and_compare_to_sweeps_kpar(kperps,beta_i,tau,wavemodes_matching_kpar,kparlim = [.1,10], flnm = '',delta_beta_i = 0, delta_tau = 0,xlim=[],ylim=[],plot_testcrv=False,flip_omega_sign=False,extralines=[],extracolors=[],scalarfac = 1.):
    from lib.plume import get_freq_from_wvmd
    from lib.plume import kaw_curve
    from lib.plume import fastmagson_curve
    from lib.plume import slowmagson_curve
    from lib.plume import whistler_curve
    from lib.plume import test_curve

    if(scalarfac == 1.):
        print("WARNING: empirical curves plotted may be in a different normalization, as the frequency/ wavenumber computed from wavemode is normalized to the upstream parameters but empircal curves need local normalization")
        print("Please account scalarfact = btotlocaloverbtotup/(np.sqrt(ntotlocalovernup)) by passing optional parameter scalarfac = scalarfact")
        print("However, this factor is often approx 1")

    kpars = np.linspace(kparlim[0],kparlim[1],1000)
    kawcrvs = []
    kawcrv_errors = []
    fastcrvs = []
    fastcrv_errors = []
    slowcrvs = []
    slowcrv_errors = []
    whicrvs = []
    whicrv_errors = []
    testcrvs = []
    testcrv_errors =  []
    #plot theoretical curves
    for kperp in kperps:
        kawcrv = []
        kawcrv_error = []
        fastcrv = []
        fastcrv_error = []
        slowcrv = []
        slowcrv_error = []
        whicrv = []
        whicrv_error = []
        testcrv = []
        testcrv_error = []
        for kpar in kpars:
            kawcrv.append(kaw_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            kawcrv_error.append(kaw_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)
            fastcrv.append(fastmagson_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            fastcrv_error.append(fastmagson_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)
            slowcrv.append(slowmagson_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            slowcrv_error.append(slowmagson_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)
            whicrv.append(whistler_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            whicrv_error.append(whistler_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)

            testcrv.append(test_curve(kperp,kpar,beta_i,tau,comp_error_prop=False))
            testcrv_error.append(test_curve(kperp,kpar,beta_i,tau,delta_beta_i=delta_beta_i,delta_tau=delta_tau,comp_error_prop=True).s)
        kawcrvs.append(np.asarray(kawcrv))
        kawcrv_errors.append(np.asarray(kawcrv_error))
        fastcrvs.append(np.asarray(fastcrv))
        fastcrv_errors.append(fastcrv_error)
        slowcrvs.append(np.asarray(slowcrv))
        slowcrv_errors.append(slowcrv_error)
        whicrvs.append(np.asarray(whicrv))
        whicrv_errors.append(np.asarray(whicrv_error))
        testcrvs.append(np.asarray(testcrv))
        testcrv_errors.append(np.asarray(testcrv_error))


    plotkpars = []
    plotkpar_errors = []
    omegas = []
    omega_errors = []
    omegas0 = []
    omega0_errors = []
    omegas2 = []
    omega2_errors = []
    #grab points and compute error for each wavemode
    for match_list in wavemodes_matching_kpar:
        plotkpar = []
        plotkpar_error = []
        omega = []
        omega_error = []
        omega0 = []
        omega0_error = []
        omega2 = []
        omega2_error = []
        for wvmd in match_list['wavemodes']:
            omega_faradayreal0,_,omega_faradayreal,_,omega_faradayreal2,_ = get_freq_from_wvmd(wvmd,comp_error_prop=True)
            plotkpar.append(wvmd['kpar'])
            plotkpar_error.append(wvmd['delta_kpar'])
            omega.append(omega_faradayreal.n)
            omega_error.append(omega_faradayreal.s)
            omega0.append(omega_faradayreal0.n)
            omega0_error.append(omega_faradayreal0.s)
            omega2.append(omega_faradayreal2.n)
            omega2_error.append(omega_faradayreal2.s)

        if(flip_omega_sign): #needed b/c we might have a sign error
            omega = -1.*np.asarray(omega)
            omega0 = -1.*np.asarray(omega0)
            omega2 = -1.*np.asarray(omega2)

        plotkpars.append(plotkpar)
        plotkpar_errors.append(plotkpar_error)
        omegas.append(omega)
        omega_errors.append(omega_error)
        omegas0.append(omega0)
        omega0_errors.append(omega0_error)
        omegas2.append(omega2)
        omega2_errors.append(omega2_error)
    #linestyle = ['--',':','-']
    linestyle = ['-',':','--','-.','.',',','-',':','--','-.','.',',','-',':','--','-.','.',',','-',':','--','-.','.',',']
    lnwidth = 1.75

    kawcrvs = np.asarray(kawcrvs)
    kawcrv_errors = np.asarray(kawcrv_errors)
    fastcrvs = np.asarray(fastcrvs)
    fastcrv_errors = np.asarray(fastcrv_errors)
    slowcrvs = np.asarray(slowcrvs)
    slowcrv_errors = np.asarray(slowcrv_errors)
    whicrvs = np.asarray(whicrvs)
    whicrv_errors = np.asarray(whicrv_errors)

    plt.figure(figsize=(6,6))
    for i in range(0,len(kawcrvs)):
        plt.errorbar(plotkpars[i],np.real(omegas[i]), xerr = plotkpar_errors[i], yerr=omega_errors[i], fmt="o",color='C0')
        plt.errorbar(plotkpars[i],np.real(omegas0[i]), xerr = plotkpar_errors[i], yerr=omega0_errors[i], fmt="s",color='C1')
        plt.errorbar(plotkpars[i],np.real(omegas2[i]), xerr = plotkpar_errors[i], yerr=omega2_errors[i], fmt="*",color='C3')
        plt.plot(kpars,scalarfac*kawcrvs[i],linestyle[i],color='black',linewidth=lnwidth,label='$k_\perp$='+str(format(kperps[i],'.2f')))
        plt.fill_between(kpars,scalarfac*kawcrvs[i]-scalarfac*kawcrv_errors[i],scalarfac*kawcrvs[i]+scalarfac*kawcrv_errors[i],alpha=.2,color='black')
        plt.plot(kpars,scalarfac*fastcrvs[i],linestyle[i],color='blue',linewidth=lnwidth)
        plt.fill_between(kpars,scalarfac*fastcrvs[i]-scalarfac*fastcrv_errors[i],scalarfac*fastcrvs[i]+scalarfac*fastcrv_errors[i],alpha=.2,color='blue')
        plt.plot(kpars,scalarfac*slowcrvs[i],linestyle[i],color='green',linewidth=lnwidth)
        plt.fill_between(kpars,scalarfac*slowcrvs[i]-scalarfac*slowcrv_errors[i],scalarfac*slowcrvs[i]+scalarfac*slowcrv_errors[i],alpha=.2,color='green')
        plt.plot(kpars,scalarfac*whicrvs[i],linestyle[i],color='red',linewidth=lnwidth)
        plt.fill_between(kpars,scalarfac*whicrvs[i]-scalarfac*whicrv_errors[i],scalarfac*whicrvs[i]+scalarfac*whicrv_errors[i],alpha=.2,color='red')
        if(plot_testcrv):
            plt.plot(kpars,testcrvs[i],linestyle[i],color='gray',linewidth=lnwidth)
            plt.fill_between(kpars,testcrvs[i]-testcrv_errors[i],testcrvs[i]+testcrv_errors[i],alpha=.2,color='gray')

    if(extralines != []):
        for _i in range(0,len(extralines[0])):
            print(_i)
            plt.plot(extralines[0][_i],extralines[1][_i],color=extracolors[_i],ls=':')

    plt.yscale('log')
    plt.xscale('log')
    plt.legend(prop={'size': 20})
    plt.xlabel('$k_{||} d_i$', fontsize = 22)
    plt.ylabel('$\omega / \Omega_i$', fontsize = 22)
    plt.gca().yaxis.set_tick_params(labelsize=18)
    plt.gca().xaxis.set_tick_params(labelsize=18)
    plt.grid(True, which="both", ls="-")
    plt.axis('scaled')
    plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots
    if(ylim == []):
        plt.ylim(.1,10)
    else:
        plt.ylim(ylim[0],ylim[1])
    if(xlim == []):
        plt.xlim(.1,10)
    else:
        plt.xlim(xlim[0],xlim[1])
    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=600,bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return omegas, omegas0, omegas2, plotkpars, omega_errors
