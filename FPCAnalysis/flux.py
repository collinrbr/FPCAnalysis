# analysis.py>

#plasma analysis functions

import numpy as np
import math


def compute_flux_tristanmp1(interpolxxs,dfields,dpar_ion,dpar_elec,params,inputs,verbose=False,justxcomps=True):
    """
    computes fluxes and normalizes

    Assumes particle data is normalized the vts!

    Assumes data is in same frame!

    Assumes dfields and dpar is normalized!

    Note, here we technically compute total flux/ energy (rather than densities) for all terms!

    Note, our algorithm assumes that the interpolxxs is aligned with the fields grid! That is each element
    in interpolxxs should land on some _xx grid point. Otherwise, the computation of the field energy density can be off by up to the 
    amount containing in one delta x grid spacing.
    """

    if(not(interpolxxs[1]) in dfields['ex_xx']):
        print("Warning!!!! This function assumes the integration bins are aligned with the fields grid. There will be some potentially significantly impactful `rounding` errors that impact normalization if this is not true. Please align bins with fields grid....")

    #get constants
    if(verbose):print('Computing parameters...')
    from FPCAnalysis.analysis import compute_beta0_tristanmp1, get_betai_betae_from_tot_and_ratio, norm_constants_tristanmp1
    beta0 = compute_beta0_tristanmp1(params,inputs)
    beta_ion,beta_elc= get_betai_betae_from_tot_and_ratio(beta0,inputs['temperature_ratio'])
    dt = params['c']/params['comp'] #in units of wpe
    dt, c_alf = norm_constants_tristanmp1(params,dt,inputs)
    mi = params['mi']
    me = params['me']
    interdx = interpolxxs[1]-interpolxxs[0] #assumes uniform spacing!

    #compute fluc and steady state fields
    if(verbose):print('Computing dfavg and dfluc...')
    from FPCAnalysis.analysis import get_average_fields_over_yz, remove_average_fields_over_yz
    dfavg = get_average_fields_over_yz(dfields)
    dfluc = remove_average_fields_over_yz(dfields)

    positions = np.asarray([(interpolxxs[_i]+interpolxxs[_i+1])/2. for _i in range(len(interpolxxs)-1)])
    
    #revert normalization of fields
    if(verbose):print('Reverting fields normalization...')
    bnorm = params['c']**2*params['sigma']/params['comp']
    sigma_ion = params['sigma']*params['me']/params['mi'] #NOTE: this is subtely differetn than what aaron's normalization is- fix it (missingn factor of gamma0 and mi+me)
    enorm = bnorm*np.sqrt(sigma_ion)*params['c'] #note, there is an extra factor of 'c hat' (c in code units, which is .45 for the main run being analyzed) that we take out
    fieldkeys = ['ex','ey','ez','bx','by','bz']
    for fk in fieldkeys:
        if(fk[0] == 'e'):
            dfields[fk] *= enorm
            dfluc[fk] *= enorm
            dfavg[fk] *= enorm
        else:
            dfields[fk] *= bnorm
            dfluc[fk] *= bnorm
            dfavg[fk] *= bnorm

     
    #bin particles
    if(verbose):print("Binning particles...")
    ionvxbins = []
    ionvybins = []
    ionvzbins = []
    elecvxbins = []
    elecvybins = []
    elecvzbins = []

    massratio = params['mi']/params['me']
    vti0 = np.sqrt(params['delgam'])
    vte0 = np.sqrt(params['mi']/params['me'])*vti0 #WARNING: THIS ASSUME Ti/Te = 1, TODO: don't assume Ti/Te = 1

    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        gptsparticleion = (x1 < dpar_ion['xi']) & (dpar_ion['xi'] <= x2)
        ionvxs = dpar_ion['ui'][gptsparticleion][:]
        ionvys = dpar_ion['vi'][gptsparticleion][:]
        ionvzs = dpar_ion['wi'][gptsparticleion][:]

        gptsparticleelec = (x1 < dpar_elec['xe']) & (dpar_elec['xe'] <= x2)
        elecvxs = dpar_elec['ue'][gptsparticleelec][:]*vte0/vti0
        elecvys = dpar_elec['ve'][gptsparticleelec][:]*vte0/vti0
        elecvzs = dpar_elec['we'][gptsparticleelec][:]*vte0/vti0

        ionvxbins.append(ionvxs)
        ionvybins.append(ionvys)
        ionvzbins.append(ionvzs)

        elecvxbins.append(elecvxs)
        elecvybins.append(elecvys)
        elecvzbins.append(elecvzs)
    if(verbose):print("done binning particles!")

    #compute dens
    if(verbose):print("computing density...")
    iondens = []
    elecdens = []
    _c = 0
    for ivxs in ionvxbins:
        iondens.append(float(len(ivxs)))
        # if(verbose):print('ion dens',interpolxxs[_c],float(len(ivxs)))
        _c += 1
    _c = 0
    for evxs in elecvxbins:
        elecdens.append(float(len(evxs)))
        # if(verbose):print('elec dens',interpolxxs[_c],float(len(evxs)))
        _c += 1

    #compute bulk v
    if(verbose):print("computing bulk vel...")
    ionvx = []
    ionvy = []
    ionvz = []
    elecvx = []
    elecvy = []
    elecvz = []
    for ivxs in ionvxbins:
        if(len(ivxs) > 0):
            ionvx.append(np.mean(ivxs))
        else:
            ionvx.append(0.)
    for ivys in ionvybins:
        if(len(ivys) > 0):
            ionvy.append(np.mean(ivys))
        else:
            ionvy.append(0.)
    for ivzs in ionvzbins:
        if(len(ivzs) > 0):
            ionvz.append(np.mean(ivzs))
        else:
            ionvz.append(0.)
    for evxs in elecvxbins:
        if(len(evxs) > 0):
            elecvx.append(np.mean(evxs))
        else:
            elecvx.append(0.)
    for evys in elecvybins:
        if(len(evys) > 0):
            elecvy.append(np.mean(evys))
        else:
            elecvy.append(0.)
    for evzs in elecvzbins:
        if(len(evzs) > 0):
            elecvz.append(np.mean(evzs))
        else:
            elecvz.append(0.)

    #compute qxs (heat flux)
    if(verbose):print("computing heat flux (qx)...")
    ionqxs = []
    for _i in range(0,len(ionvxbins)):
        ivx = ionvxbins[_i]
        ivy = ionvybins[_i]
        ivz = ionvzbins[_i]
        ionqx = 0.5*mi*np.sum((valx-ionvx[_i])*((valx-ionvx[_i])**2+(valy-ionvy[_i])**2+(valz-ionvz[_i])**2) for valx,valy,valz in zip(ivx,ivy,ivz))
        ionqxs.append(ionqx)
    if(not(justxcomps)):
        ionqys = []
        for _i in range(0,len(ionvxbins)):
            ivx = ionvxbins[_i]
            ivy = ionvybins[_i]
            ivz = ionvzbins[_i]
            ionqy = 0.5*mi*np.sum((valy-ionvy[_i])*((valx-ionvx[_i])**2+(valy-ionvy[_i])**2+(valz-ionvz[_i])**2) for valx,valy,valz in zip(ivx,ivy,ivz))
            ionqys.append(ionqy)
        ionqzs = []
        for _i in range(0,len(ionvxbins)):
            ivx = ionvxbins[_i]
            ivy = ionvybins[_i]
            ivz = ionvzbins[_i]
            ionqz = 0.5*mi*np.sum((valz-ionvz[_i])*((valx-ionvx[_i])**2+(valy-ionvy[_i])**2+(valz-ionvz[_i])**2) for valx,valy,valz in zip(ivx,ivy,ivz))
            ionqzs.append(ionqz)
    elecqxs = []
    for _i in range(0,len(elecvxbins)):
        evx = elecvxbins[_i]
        evy = elecvybins[_i]
        evz = elecvzbins[_i]
        elecqx = 0.5*me*np.sum((valx-elecvx[_i])*((valx-elecvx[_i])**2+(valy-elecvy[_i])**2+(valz-elecvz[_i])**2) for valx,valy,valz in zip(evx,evy,evz))
        elecqxs.append(elecqx)
    if(not(justxcomps)):
        elecqys = []
        for _i in range(0,len(elecvxbins)):
            evx = elecvxbins[_i]
            evy = elecvybins[_i]
            evz = elecvzbins[_i]
            elecqy = 0.5*me*np.sum((valy-elecvy[_i])*((valx-elecvx[_i])**2+(valy-elecvy[_i])**2+(valz-elecvz[_i])**2) for valx,valy,valz in zip(evx,evy,evz))
            elecqys.append(elecqy)
        elecqzs = []
        for _i in range(0,len(elecvxbins)):
            evx = elecvxbins[_i]
            evy = elecvybins[_i]
            evz = elecvzbins[_i]
            elecqz = 0.5*me*np.sum((valz-elecvz[_i])*((valx-elecvx[_i])**2+(valy-elecvy[_i])**2+(valz-elecvz[_i])**2) for valx,valy,valz in zip(evx,evy,evz))
            elecqzs.append(elecqz)

    #compute ram kinetic energy flux (that is this block computes the bulk flow energy, but we call it ram pressure)
    if(verbose):print('compute kinetic energy flux....')
    ionframx = []
    for _i, ivxs in enumerate(ionvxbins):
        fx = ionvx[_i]*0.5*(iondens[_i])*mi*(ionvx[_i]**2+ionvy[_i]**2+ionvz[_i]**2)
        ionframx.append(fx)
    if(not(justxcomps)):
        ionframy = []
        for _i, ivys in enumerate(ionvybins):
            fy = ionvy[_i]*0.5*(iondens[_i])*mi*(ionvx[_i]**2+ionvy[_i]**2+ionvz[_i]**2)
            ionframy.append(fy)
        ionframz = []
        for _i, ivzs in enumerate(ionvzbins):
            fz = ionvz[_i]*0.5*(iondens[_i])*mi*(ionvx[_i]**2+ionvy[_i]**2+ionvz[_i]**2)
            ionframz.append(fz)
    elecframx = []
    for _i, ivxs in enumerate(elecvxbins):
        fx = elecvx[_i]*0.5*(iondens[_i])*me*(elecvx[_i]**2+elecvy[_i]**2+elecvz[_i]**2)
        elecframx.append(fx)
    if(not(justxcomps)):
        elecframy = []
        for _i, ivxs in enumerate(elecvybins):
            fy = elecvy[_i]*0.5*(iondens[_i])*me*(elecvx[_i]**2+elecvy[_i]**2+elecvz[_i]**2)
            elecframy.append(fy)
        elecframz = []
        for _i, ivxs in enumerate(elecvzbins):
            fz = elecvz[_i]*0.5*(iondens[_i])*me*(elecvx[_i]**2+elecvy[_i]**2+elecvz[_i]**2)
            elecframz.append(fz)

    #compute enthalpy flux first term
    if(verbose):print("compute enthalpy flux....")
    ionethxs = []
    for _i in range(0,len(ionvxbins)):
        ivx = ionvxbins[_i]
        ivy = ionvybins[_i]
        ivz = ionvzbins[_i]
        ionethx = ionvx[_i]*0.5*mi*np.sum((valx-ionvx[_i])**2+(valy-ionvy[_i])**2+(valz-ionvz[_i])**2 for valx,valy,valz in zip(ivx,ivy,ivz))
        ionethxs.append(ionethx)
    if(not(justxcomps)):
        ionethys = []
        for _i in range(0,len(ionvxbins)):
            ivx = ionvxbins[_i]
            ivy = ionvybins[_i]
            ivz = ionvzbins[_i]
            ionethy = ionvy[_i]*0.5*mi*np.sum((valx-ionvx[_i])**2+(valy-ionvy[_i])**2+(valz-ionvz[_i])**2 for valx,valy,valz in zip(ivx,ivy,ivz))
            ionethys.append(ionethy)
        ionethzs = []
        for _i in range(0,len(ionvxbins)):
            ivx = ionvxbins[_i]
            ivy = ionvybins[_i]
            ivz = ionvzbins[_i]
            ionethz = ionvz[_i]*0.5*mi*np.sum((valx-ionvx[_i])**2+(valy-ionvy[_i])**2+(valz-ionvz[_i])**2 for valx,valy,valz in zip(ivx,ivy,ivz))
            ionethzs.append(ionethz)
    elecethxs = []
    for _i in range(0,len(elecvxbins)):
        evx = elecvxbins[_i]
        evy = elecvybins[_i]
        evz = elecvzbins[_i]
        elecethx = elecvx[_i]*0.5*me*np.sum((valx-elecvx[_i])**2+(valy-elecvy[_i])**2+(valz-elecvz[_i])**2 for valx,valy,valz in zip(evx,evy,evz))
        elecethxs.append(elecethx)
    if(not(justxcomps)):
        elecethys = []
        for _i in range(0,len(elecvxbins)):
            evx = elecvxbins[_i]
            evy = elecvybins[_i]
            evz = elecvzbins[_i]
            elecethy = elecvy[_i]*0.5*me*np.sum((valx-elecvx[_i])**2+(valy-elecvy[_i])**2+(valz-elecvz[_i])**2 for valx,valy,valz in zip(evx,evy,evz))
            elecethys.append(elecethy)
        elecethzs = []
        for _i in range(0,len(elecvxbins)):
            evx = elecvxbins[_i]
            evy = elecvybins[_i]
            evz = elecvzbins[_i]
            elecethz = elecvz[_i]*0.5*me*np.sum((valx-elecvx[_i])**2+(valy-elecvy[_i])**2+(valz-elecvz[_i])**2 for valx,valy,valz in zip(evx,evy,evz))
            elecethzs.append(elecethz)

    #compute enthalpy flux second term
    ionpdotusxs = []
    for _i in range(0,len(ionvxbins)):
        ibvx = ionvx[_i]  #bulk ion velocity
        ibvy = ionvy[_i]
        ibvz = ionvz[_i]
        ipdu = mi*np.sum((ibvx*(ionvxbins[_i][idx]-ibvx)**2+ibvy*(ionvxbins[_i][idx]-ibvx)*(ionvybins[_i][idx]-ibvy)+ibvz*(ionvxbins[_i][idx]-ibvx)*(ionvzbins[_i][idx]-ibvz) for idx in range(0,len(ionvxbins[_i]))))
        ionpdotusxs.append(ipdu)
    if(not(justxcomps)):
        ionpdotusys = []
        for _i in range(0,len(ionvxbins)):
            ibvx = ionvx[_i]  #bulk ion velocity
            ibvy = ionvy[_i]
            ibvz = ionvz[_i]
            ipdu = mi*np.sum((ibvx*(ionvybins[_i][idx]-ibvy)*(ionvxbins[_i][idx]-ibvx)+ibvy*(ionvybins[_i][idx]-ibvy)**2+ibvz*(ionvybins[_i][idx]-ibvy)*(ionvzbins[_i][idx]-ibvz) for idx in range(0,len(ionvxbins[_i]))))
            ionpdotusys.append(ipdu)
        ionpdotuszs = []
        for _i in range(0,len(ionvxbins)):
            ibvx = ionvx[_i]  #bulk ion velocity
            ibvy = ionvy[_i]
            ibvz = ionvz[_i]
            ipdu = mi*np.sum((ibvx*(ionvzbins[_i][idx]-ibvz)*(ionvxbins[_i][idx]-ibvx)+ibvy*(ionvzbins[_i][idx]-ibvz)*(ionvybins[_i][idx]-ibvy)+ibvz*(ionvzbins[_i][idx]-ibvz)**2 for idx in range(0,len(ionvxbins[_i]))))
            ionpdotuszs.append(ipdu)
    elecpdotusxs = []
    for _i in range(0,len(elecvxbins)):
        ebvx = elecvx[_i]  #bulk elec velocity
        ebvy = elecvy[_i]
        ebvz = elecvz[_i]
        epdu = me*np.sum((ebvx*(elecvxbins[_i][idx]-ebvx)**2+ebvy*(elecvxbins[_i][idx]-ebvx)*(elecvybins[_i][idx]-ebvy)+ebvz*(elecvxbins[_i][idx]-ebvx)*(elecvzbins[_i][idx]-ebvz) for idx in range(0,len(elecvxbins[_i]))))
        elecpdotusxs.append(epdu)

    #TODO: remove this if this metric doesn't isolate compressive heating in this system like we hoped
    elecpdotusxsxx = []
    for _i in range(0,len(elecvxbins)):
        ebvx = elecvx[_i]  #bulk elec velocity
        ebvy = elecvy[_i]
        ebvz = elecvz[_i]
        epdu = me*np.sum((ebvx*(elecvxbins[_i][idx]-ebvx)**2 for idx in range(0,len(elecvxbins[_i]))))
        elecpdotusxsxx.append(epdu)

    elecpdotusxsxy = []
    for _i in range(0,len(elecvxbins)):
        ebvx = elecvx[_i]  #bulk elec velocity
        ebvy = elecvy[_i]
        ebvz = elecvz[_i]
        epdu = me*np.sum((ebvy*(elecvxbins[_i][idx]-ebvx)*(elecvybins[_i][idx]-ebvy) for idx in range(0,len(elecvxbins[_i]))))
        elecpdotusxsxy.append(epdu)
    
    elecpdotusxsxz = []
    for _i in range(0,len(elecvxbins)):
        ebvx = elecvx[_i]  #bulk elec velocity
        ebvy = elecvy[_i]
        ebvz = elecvz[_i]
        epdu = me*np.sum((ebvz*(elecvxbins[_i][idx]-ebvx)*(elecvzbins[_i][idx]-ebvz) for idx in range(0,len(elecvxbins[_i]))))
        elecpdotusxsxz.append(epdu)


    if(not(justxcomps)):
        elecpdotusys = []
        for _i in range(0,len(elecvxbins)):
            ebvx = elecvx[_i]  #bulk elec velocity
            ebvy = elecvy[_i]
            ebvz = elecvz[_i]
            epdu = me*np.sum((ebvx*(elecvybins[_i][idx]-ebvy)*(elecvxbins[_i][idx]-ebvx)+ebvy*(elecvybins[_i][idx]-ebvy)**2+ebvz*(elecvybins[_i][idx]-ebvy)*(elecvzbins[_i][idx]-ebvz) for idx in range(0,len(elecvxbins[_i]))))
            elecpdotusys.append(epdu)
        elecpdotuszs = []
        for _i in range(0,len(elecvxbins)):
            ebvx = elecvx[_i]  #bulk elec velocity
            ebvy = elecvy[_i]
            ebvz = elecvz[_i]
            epdu = me*np.sum((ebvx*(elecvzbins[_i][idx]-ebvz)*(elecvxbins[_i][idx]-ebvx)+ebvy*(elecvzbins[_i][idx]-ebvz)*(elecvybins[_i][idx]-ebvy)+ebvz*(elecvzbins[_i][idx]-ebvz)**2 for idx in range(0,len(elecvxbins[_i]))))
            elecpdotuszs.append(epdu)
    

    #compute total energy flux (TODO: Speed up! As there are redundant calculations in this block, but thats fine for now)
    if(verbose):print("compute total energy flux....")
    iontotefluxxs = []
    electotefluxxs = []
    for _i in range(0,len(ionvxbins)):
        ibvx = ionvx[_i]  #bulk ion velocity
        ibvy = ionvy[_i]
        ibvz = ionvz[_i]
        itfx = mi/2.*np.sum((ionvxbins[_i][idx]*(ionvxbins[_i][idx]**2+ionvybins[_i][idx]**2+ionvzbins[_i][idx]**2) for idx in range(0,len(ionvxbins[_i]))))
        iontotefluxxs.append(itfx)
    for _i in range(0,len(elecvxbins)):
        ebvx = elecvx[_i]  #bulk elec velocity
        ebvy = elecvy[_i]
        ebvz = elecvz[_i]
        etfx = me/2.*np.sum((elecvxbins[_i][idx]*(elecvxbins[_i][idx]**2+elecvybins[_i][idx]**2+elecvzbins[_i][idx]**2) for idx in range(0,len(elecvxbins[_i]))))
        electotefluxxs.append(etfx)

    if(not(justxcomps)):
        iontotefluyys = []
        electotefluyys = []
        for _i in range(0,len(ionvybins)):
            ibvx = ionvx[_i]  #bulk ion velocity
            ibvy = ionvy[_i]
            ibvz = ionvz[_i]
            itfy = mi/2.*np.sum((ionvybins[_i][idx]*(ionvxbins[_i][idx]**2+ionvybins[_i][idx]**2+ionvzbins[_i][idx]**2) for idx in range(0,len(ionvybins[_i]))))
            iontotefluyys.append(itfy)
        for _i in range(0,len(elecvybins)):
            ebvx = elecvx[_i]  #bulk elec velocity
            ebvy = elecvy[_i]
            ebvz = elecvz[_i]
            etfy = me/2.*np.sum((elecvybins[_i][idx]*(elecvxbins[_i][idx]**2+elecvybins[_i][idx]**2+elecvzbins[_i][idx]**2) for idx in range(0,len(elecvybins[_i]))))
            electotefluyys.append(etfy)

        iontotefluzzs = []
        electotefluzzs = []
        for _i in range(0,len(ionvybins)):
            ibvx = ionvx[_i]  #bulk ion velocity
            ibvy = ionvy[_i]
            ibvz = ionvz[_i]
            itfz = mi/2.*np.sum((ionvzbins[_i][idx]*(ionvxbins[_i][idx]**2+ionvybins[_i][idx]**2+ionvzbins[_i][idx]**2) for idx in range(0,len(ionvzbins[_i]))))
            iontotefluzzs.append(itfz)
        for _i in range(0,len(elecvzbins)):
            ebvx = elecvx[_i]  #bulk elec velocity
            ebvy = elecvy[_i]
            ebvz = elecvz[_i]
            etfz = me/2.*np.sum((elecvzbins[_i][idx]*(elecvxbins[_i][idx]**2+elecvybins[_i][idx]**2+elecvzbins[_i][idx]**2) for idx in range(0,len(elecvzbins[_i]))))
            electotefluzzs.append(etfz)

    #normalization factors--------------------------------------------------------------
    #compute upstream thermal energy for normalization factor
    Evthiup = 0
    Ethiupindexes = [-_idx for _idx in range(1,int(0.1*len(positions)))]
    if(Ethiupindexes == []):
        Ethiupindexes = [-1]
        print("Warning, found no bins in the upstream. Consider reducing dx of bins...")
        print("Using rightmost bin as upstream...")
    if(verbose):print("using ",positions[Ethiupindexes[-1]]-interdx/2,' to ',positions[Ethiupindexes[0]]+interdx/2,' for normalization...')
    for _i in Ethiupindexes:
        bvxi = np.mean(ionvxbins[_i])
        bvyi = np.mean(ionvybins[_i])
        bvzi = np.mean(ionvzbins[_i])
        Evthiup += 0.5*mi*np.sum((ionvxbins[_i]-bvxi)**2+(ionvybins[_i]-bvyi)**2+(ionvzbins[_i]-bvzi)**2)
    Evthiup = Evthiup / float(len(Ethiupindexes))

    Evtheup = 0
    Etheupindexes = [-_idx for _idx in range(1,int(0.1*len(positions)))]
    if(Etheupindexes == []):
        Ethiupindexes = [-1,-2]
        print("Warning, found no bins in the upstream. Consider reducing dx of bins...")
        print("Using rightmost bin as upstream...")
    for _i in Etheupindexes:
        bvxe = np.mean(elecvxbins[_i])
        bvye = np.mean(elecvybins[_i])
        bvze = np.mean(elecvzbins[_i])
        Evtheup += 0.5*me*np.sum((elecvxbins[_i]-bvxe)**2+(elecvybins[_i]-bvye)**2+(elecvzbins[_i]-bvze)**2)
    Evtheup = Evtheup / float(len(Etheupindexes))

    #compute upstream B field ener for normalization factor
    x1 = positions[-(int(0.1*len(positions)))]-interdx/2.
    x2 = positions[-1]+interdx/2.
    Bxs = dfields['bx']
    Bys = dfields['by']
    Bzs = dfields['bz']
    goodfieldpts = (x1 < dfields['by_xx']) & (dfields['by_xx'] <= x2)
    Bfieldenerup = np.sum(Bxs[:,:,goodfieldpts]**2+Bys[:,:,goodfieldpts]**2+Bzs[:,:,goodfieldpts]**2)
    Bfieldenerup = Bfieldenerup/((x2-x1)/interdx)

    #normalize using upstream beta_ion
    #Ethviup/a*Bfieldenerup = betaion where a is factor we want to find
    # a = Ethviup/betaion*Bfieldeneruup
    valfac = Evthiup/(beta_ion*(1./(8.*np.pi))*Bfieldenerup) #computes factor we need to scale fields^2 by to get in same units as particles
    fieldEfac = (1./(8.*np.pi))*valfac #multiple unnormalized E^2 or B^2 to get total energy in correct units
    c_thi = c_alf/np.sqrt(beta_ion)
    poyntFluxEfac = valfac*c_thi/params['c']/(4.*np.pi) #a normalizes field value squared

    # if(verbose):print("computing poynting flux....")
    poyntxxs = []
    poyntyys = []
    poyntzzs = []
    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        Exs = dfields['ex']
        Eys = dfields['ey']
        Ezs = dfields['ez']
        Bxs = dfields['bx']
        Bys = dfields['by']
        Bzs = dfields['bz']
        goodfieldpts = (x1 < dfields['ey_xx']) & (dfields['ey_xx'] <= x2)
        pxx = np.sum(Eys[:,:,goodfieldpts]*Bzs[:,:,goodfieldpts]-Ezs[:,:,goodfieldpts]*Bys[:,:,goodfieldpts])
        pyy = np.sum(-Exs[:,:,goodfieldpts]*Bzs[:,:,goodfieldpts]+Ezs[:,:,goodfieldpts]*Bxs[:,:,goodfieldpts])
        pzz = np.sum(Exs[:,:,goodfieldpts]*Bys[:,:,goodfieldpts]-Eys[:,:,goodfieldpts]*Bxs[:,:,goodfieldpts])
        poyntxxs.append(pxx)
        poyntyys.append(pyy)
        poyntzzs.append(pzz)
    poyntxxs = poyntFluxEfac*np.asarray(poyntxxs) 
    poyntyys = poyntFluxEfac*np.asarray(poyntyys) 
    poyntzzs = poyntFluxEfac*np.asarray(poyntzzs) 

    #compute W fields
    if(verbose):print("computing W fields....")
    WEfields = []
    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        Exs = dfields['ex']
        Eys = dfields['ey']
        Ezs = dfields['ez']
        goodfieldpts = (x1 < dfields['ey_xx']) & (dfields['ey_xx'] <= x2)
        ef = np.sum(Exs[:,:,goodfieldpts]**2+Eys[:,:,goodfieldpts]**2+Ezs[:,:,goodfieldpts]**2)
        WEfields.append(ef)
    WEfields = fieldEfac*np.asarray(WEfields) 

    WBfields = []
    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        Bxs = dfields['bx']
        Bys = dfields['by']
        Bzs = dfields['bz']
        goodfieldpts = (x1 < dfields['ey_xx']) & (dfields['ey_xx'] <= x2)
        bf = np.sum(Bxs[:,:,goodfieldpts]**2+Bys[:,:,goodfieldpts]**2+Bzs[:,:,goodfieldpts]**2)
        WBfields.append(bf)
    WBfields = fieldEfac*np.asarray(WBfields) 

    Wfields = WEfields+WBfields

    #compute total particle energy
    if(verbose):print("Computing total particle energy...")
    Wion = np.asarray([0.5*mi*(np.sum((valx)**2+(valy)**2+(valz)**2 for valx,valy,valz in zip(vx,vy,vz))) for vx,vy,vz in zip(ionvxbins,ionvybins,ionvzbins)])
    Welec = np.asarray([0.5*me*(np.sum((valx)**2+(valy)**2+(valz)**2 for valx,valy,valz in zip(vx,vy,vz))) for vx,vy,vz in zip(elecvxbins,elecvybins,elecvzbins)])

    #compute steady and fluc fluxes
    if(verbose):print("computing steady and fluc fluxes...")
    Ebar = np.stack((dfavg['ex'], dfavg['ey'], dfavg['ez']), axis=-1)
    Bbar = np.stack((dfavg['bx'], dfavg['by'], dfavg['bz']), axis=-1)
    Ebar_cross_Bbar = np.cross(Ebar, Bbar)
    EbarxBbar_x = Ebar_cross_Bbar[..., 0]
    EbarxBbar_y = Ebar_cross_Bbar[..., 1]
    EbarxBbar_z = Ebar_cross_Bbar[..., 2]

    Efluc = np.stack((dfluc['ex'], dfluc['ey'], dfluc['ez']), axis=-1)
    Bfluc = np.stack((dfluc['bx'], dfluc['by'], dfluc['bz']), axis=-1)
    Efluc_cross_Bfluc = np.cross(Efluc, Bfluc)
    EflucxBfluc_x = Efluc_cross_Bfluc[..., 0]
    EflucxBfluc_y = Efluc_cross_Bfluc[..., 1]
    EflucxBfluc_z = Efluc_cross_Bfluc[..., 2]
    Bbar_dot_bbar = np.sum(Bbar * Bbar, axis=-1) #takes dot product at each location on the 2d grid
    Bbar_dot_bbar_out = []

    #reduce to 1D (as these are functions of x only!!!) with correct normalization (the norm fac used later scales 1 point of fields^2 units to total Evthi according to ion plasma beta)
    EflucxBfluc_x_out = []
    EflucxBfluc_y_out = []
    EflucxBfluc_z_out = []
    EbarxBbar_x_out = []
    EbarxBbar_y_out = []
    EbarxBbar_z_out = []
    for xx in interpolxxs:
        x1 = xx
        x2 = xx+interdx
        goodfieldpts = (x1 < dfavg['ey_xx']) & (dfavg['ey_xx'] <= x2)
        
        EflucxBflucxx = np.sum(EflucxBfluc_x[:,:,goodfieldpts]) 
        EflucxBflucxx = EflucxBflucxx 
        EflucxBfluc_x_out.append(EflucxBflucxx)
        EflucxBflucyy = np.sum(EflucxBfluc_y[:,:,goodfieldpts]) 
        EflucxBflucyy = EflucxBflucyy
        EflucxBfluc_y_out.append(EflucxBflucyy)
        EflucxBfluczz = np.sum(EflucxBfluc_z[:,:,goodfieldpts]) 
        EflucxBfluczz = EflucxBfluczz
        EflucxBfluc_z_out.append(EflucxBfluczz)

        EbarxBbarxx = np.sum(EbarxBbar_x[:,:,goodfieldpts]) 
        EbarxBbarxx = EbarxBbarxx
        EbarxBbar_x_out.append(EbarxBbarxx)
        EbarxBbaryy = np.sum(EbarxBbar_y[:,:,goodfieldpts]) 
        EbarxBbaryy = EbarxBbaryy
        EbarxBbar_y_out.append(EbarxBbaryy)
        EbarxBbarzz = np.sum(EbarxBbar_z[:,:,goodfieldpts]) 
        EbarxBbarzz = EbarxBbarzz
        EbarxBbar_z_out.append(EbarxBbarzz)
        
        Bbar_dot_bbar_temp =  np.sum(Bbar_dot_bbar[:,:,goodfieldpts])
        Bbar_dot_bbar_temp = Bbar_dot_bbar_temp
        Bbar_dot_bbar_out.append(Bbar_dot_bbar_temp)
        
    EflucxBfluc_x_avg = poyntFluxEfac*np.asarray(EflucxBfluc_x_out[0:-1])
    EflucxBfluc_y_avg = poyntFluxEfac*np.asarray(EflucxBfluc_y_out[0:-1])
    EflucxBfluc_z_avg = poyntFluxEfac*np.asarray(EflucxBfluc_z_out[0:-1])
    EbarxBbar_x = poyntFluxEfac*np.asarray(EbarxBbar_x_out[0:-1])
    EbarxBbar_y = poyntFluxEfac*np.asarray(EbarxBbar_y_out[0:-1])
    EbarxBbar_z = poyntFluxEfac*np.asarray(EbarxBbar_z_out[0:-1])
    Bbar_dot_bbar = fieldEfac*np.asarray(Bbar_dot_bbar_out[0:-1])

    #return everything
    fluxes = {}
    fluxes['iondens'] = np.asarray(iondens)
    fluxes['elecdens'] = np.asarray(elecdens)
    fluxes['ionvx'] = np.asarray(ionvx)
    if(not(justxcomps)):
        fluxes['ionvy'] = np.asarray(ionvy)
        fluxes['ionvz'] = np.asarray(ionvz)
    fluxes['elecvx'] = np.asarray(elecvx)
    if(not(justxcomps)):
        fluxes['elecvy'] = np.asarray(elecvy)
        fluxes['elecvz'] = np.asarray(elecvz)

    fluxes['ionqxs'] = np.asarray(ionqxs)
    fluxes['elecqxs'] = np.asarray(elecqxs)
    if(not(justxcomps)):
        fluxes['ionqys'] = np.asarray(ionqys)
        fluxes['elecqys'] = np.asarray(elecqys)
        fluxes['ionqzs'] = np.asarray(ionqzs)
        fluxes['elecqzs'] = np.asarray(elecqzs)

    fluxes['ionframx'] = np.asarray(ionframx)
    fluxes['elecframx'] = np.asarray(elecframx)
    if(not(justxcomps)):
        fluxes['ionframy'] = np.asarray(ionframy)
        fluxes['elecframy'] = np.asarray(elecframy)
        fluxes['ionframz'] = np.asarray(ionframz)
        fluxes['elecframz'] = np.asarray(elecframz)

    fluxes['ionethxs'] = np.asarray(ionethxs)
    fluxes['elecethxs'] = np.asarray(elecethxs)
    if(not(justxcomps)):
        fluxes['ionethys'] = np.asarray(ionethys)
        fluxes['elecethys'] = np.asarray(elecethys)
        fluxes['ionethzs'] = np.asarray(ionethzs)
        fluxes['elecethzs'] = np.asarray(elecethzs)

    fluxes['ionpdotusxs'] = np.asarray(ionpdotusxs)
    if(not(justxcomps)):
        fluxes['ionpdotusys'] = np.asarray(ionpdotusys)
        fluxes['ionpdotuszs'] = np.asarray(ionpdotuszs)
    fluxes['elecpdotusxs'] = np.asarray(elecpdotusxs)
    fluxes['elecpdotusxsxy'] = np.asarray(elecpdotusxsxy) #TODO: remove this if this doesnt isolate heating like desired
    fluxes['elecpdotusxsxz'] = np.asarray(elecpdotusxsxz) #TODO: remove this if this doesnt isolate heating like desired
    if(not(justxcomps)):
        fluxes['elecpdotusys'] = np.asarray(elecpdotusys)
        fluxes['elecpdotuszs'] = np.asarray(elecpdotuszs)

    fluxes['iontotefluxxs'] = np.asarray(iontotefluxxs)
    fluxes['electotefluxxs'] = np.asarray(electotefluxxs)
    if(not(justxcomps)):
        fluxes['iontotefluyys'] = np.asarray(iontotefluyys)
        fluxes['electotefluyys'] = np.asarray(electotefluyys)
        fluxes['iontotefluzzs'] = np.asarray(iontotefluzzs)
        fluxes['electotefluzzs'] = np.asarray(electotefluzzs)
        
    fluxes['poyntxxs'] = np.asarray(poyntxxs)
    fluxes['poyntyys'] = np.asarray(poyntyys)
    fluxes['poyntzzs'] = np.asarray(poyntzzs)

    fluxes['Wfields'] = np.asarray(Wfields)
    fluxes['WEfields'] = np.asarray(WEfields)
    fluxes['WBfields'] = np.asarray(WBfields)
    fluxes['Wion'] = np.asarray(Wion)
    fluxes['Welec'] = np.asarray(Welec)
    fluxes['EflucxBfluc_x_avg'] = np.asarray(EflucxBfluc_x_avg)
    fluxes['EflucxBfluc_y_avg'] = np.asarray(EflucxBfluc_y_avg)
    fluxes['EflucxBfluc_z_avg'] = np.asarray(EflucxBfluc_z_avg)
    fluxes['EbarxBbar_x'] = np.asarray(EbarxBbar_x)
    fluxes['EbarxBbar_y'] = np.asarray(EbarxBbar_y)
    fluxes['EbarxBbar_z'] = np.asarray(EbarxBbar_z)
    fluxes['Bbar_dot_bbar'] = np.asarray(Bbar_dot_bbar)

    fluxes['interpolxxs'] = interpolxxs
    fluxes['interdx'] = interdx

    fluxes['positions'] = positions

    fluxes['fieldEfac'] = fieldEfac
    fluxes['poyntFluxEfac'] = poyntFluxEfac
    fluxes['valfac'] = valfac

    #re normalize fields
    fieldkeys = ['ex','ey','ez','bx','by','bz']
    for fk in fieldkeys:
        if(fk[0] == 'e'):
            dfields[fk] /= enorm
            dfluc[fk] /= enorm
            dfavg[fk] /= enorm
        else:
            dfields[fk] /= bnorm
            dfluc[fk] /= bnorm
            dfavg[fk] /= bnorm

    return fluxes


def compute_diamag_drift(dfields,dpar_elec,params,interpolxxs,verbose=False):
    """
    u_dia = -1/(q_e n_e)\frac{nabla p_{\perp,e} \times \mathbf{B}}{|\mathbf{B}|^2}

    Claim: u_dia is the thing causing the adiabatic heating. That is u_dia is caused
    by the thing conserving the adiabatic invariant, which we will call adiabatic heating.

    The 'equivalency' of u_dia and adiabatic heating was shown by Juno et al 2021 (assumes no scattering tho!)

    Note, the above claim assumes a sufficiently high mass ratio    
    """

    #get constants
    if(verbose):print('Computing parameters...')

    me = params['me']

    qe = dpar_elec['q']

    positions = np.asarray([(interpolxxs[_i]+interpolxxs[_i+1])/2. for _i in range(len(interpolxxs)-1)])
    
    #revert normalization of fields
    if(verbose):print('Reverting fields normalization...')
    bnorm = params['c']**2*params['sigma']/params['comp']
    sigma_ion = params['sigma']*params['me']/params['mi'] #NOTE: this is subtely differetn than what aaron's normalization is- fix it (missingn factor of gamma0 and mi+me)
    enorm = bnorm*np.sqrt(sigma_ion)*params['c'] #note, there is an extra factor of 'c hat' (c in code units, which is .45 for the main run being analyzed) that we take out
    fieldkeys = ['ex','ey','ez','bx','by','bz']
    for fk in fieldkeys:
        if(fk[0] == 'e'):
            dfields[fk] *= enorm
        else:
            dfields[fk] *= bnorm

    #bin particles
    if(verbose):print("Binning particles...")
    elecvxbins = []
    elecvybins = []
    elecvzbins = []

    vti0 = np.sqrt(params['delgam'])
    vte0 = np.sqrt(params['mi']/params['me'])*vti0 #WARNING: THIS ASSUME Ti/Te = 1, TODO: don't assume Ti/Te = 1

    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]

        gptsparticleelec = (x1 < dpar_elec['xe']) & (dpar_elec['xe'] <= x2)
        elecvxs = dpar_elec['ue'][gptsparticleelec][:]*vte0
        elecvys = dpar_elec['ve'][gptsparticleelec][:]*vte0
        elecvzs = dpar_elec['we'][gptsparticleelec][:]*vte0

        elecvxbins.append(elecvxs)
        elecvybins.append(elecvys)
        elecvzbins.append(elecvzs)
    if(verbose):print("done binning particles!")

    #compute bulk v
    if(verbose):print("computing bulk vel...")
    elecvx = []
    elecvy = []
    elecvz = []
    for evxs in elecvxbins:
        if(len(evxs) > 0):
            elecvx.append(np.mean(evxs))
        else:
            elecvx.append(0.)
    for evys in elecvybins:
        if(len(evys) > 0):
            elecvy.append(np.mean(evys))
        else:
            elecvy.append(0.)
    for evzs in elecvzbins:
        if(len(evzs) > 0):
            elecvz.append(np.mean(evzs))
        else:
            elecvz.append(0.)

    #compute perp pressure
    pperp = []
    nspec = []
    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        bvxe = np.mean(elecvxbins[_i])
        bvye = np.mean(elecvybins[_i])
        bvze = np.mean(elecvzbins[_i])

        px = me*np.sum([(evxval-bvxe)**2 for evxval in elecvxbins[_i]]) 
        # py = me*np.sum([(evyval-bvye)**2 for evyval in elecvybins[_i]]) #not used here so commented out
        pz = me*np.sum([(evzval-bvze)**2 for evzval in elecvzbins[_i]])

        nspec.append(float(len(elecvzbins[_i])))

        #Assume that \mathbf{B} = B(x) \hat{y} (or well approximated as such)
        ppr = (px+pz)/3.
        pperp.append(ppr)
    pperp = np.asarray(pperp)

    #compute diamagnetic drift
    udiax = []
    udiay = []
    udiaz = []

    #TODO convert above to FAC? (if so, re do comments above) <doesnt matter for our perp simulation that much>
    interdx = interpolxxs[1]-interpolxxs[0]
    gradnormfac = params['comp']*np.sqrt(params['mi']/params['me'])/interdx
    gradpperp = np.gradient(pperp)*(gradnormfac)
    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        Bxs = dfields['bx']
        Bys = dfields['by']
        Bzs = dfields['bz']

        goodfieldpts = (x1 < dfields['ey_xx']) & (dfields['ey_xx'] <= x2)
        Bx = np.mean(Bxs[:,:,goodfieldpts])
        By = np.mean(Bys[:,:,goodfieldpts])
        Bz = np.mean(Bzs[:,:,goodfieldpts])

        ne = nspec[_i]

        ud = (-params['c']/(qe*ne))*np.cross([gradpperp[_i],0,0],[Bx,By,Bz])/np.linalg.norm([Bx,By,Bz])**2
        udiax.append(ud[0])
        udiay.append(ud[1])
        udiaz.append(ud[2])
    udiax = np.asarray(udiax)
    udiay = np.asarray(udiay)
    udiaz = np.asarray(udiaz)

    #re normalize fields 
    fieldkeys = ['ex','ey','ez','bx','by','bz']
    for fk in fieldkeys:
        if(fk[0] == 'e'):
            dfields[fk] /= enorm
        else:
            dfields[fk] /= bnorm

    return positions, udiax/vti0, udiay/vti0, udiaz/vti0, nspec, elecvx, elecvy, elecvz #TODO: remove the extra elecvx outputs at end

def compute_gradb_drift(dfields,dpar_elec,params,interpolxxs,verbose=False):
    """

    """

    #get constants
    if(verbose):print('Computing parameters...')

    me = params['me']

    qe = dpar_elec['q']

    #compute fluc and steady state fields
    positions = np.asarray([(interpolxxs[_i]+interpolxxs[_i+1])/2. for _i in range(len(interpolxxs)-1)])
    
    #revert normalization of fields
    if(verbose):print('Reverting fields normalization...')
    bnorm = params['c']**2*params['sigma']/params['comp']
    sigma_ion = params['sigma']*params['me']/params['mi'] #NOTE: this is subtely differetn than what aaron's normalization is- fix it (missingn factor of gamma0 and mi+me)
    enorm = bnorm*np.sqrt(sigma_ion)*params['c'] #note, there is an extra factor of 'c hat' (c in code units, which is .45 for the main run being analyzed) that we take out
    fieldkeys = ['ex','ey','ez','bx','by','bz']
    for fk in fieldkeys:
        if(fk[0] == 'e'):
            dfields[fk] *= enorm
        else:
            dfields[fk] *= bnorm
     
    #bin particles
    if(verbose):print("Binning particles...")
    elecvxbins = []
    elecvybins = []
    elecvzbins = []

    vti0 = np.sqrt(params['delgam'])
    vte0 = np.sqrt(params['mi']/params['me'])*vti0 #WARNING: THIS ASSUME Ti/Te = 1, TODO: don't assume Ti/Te = 1

    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]

        gptsparticleelec = (x1 < dpar_elec['xe']) & (dpar_elec['xe'] <= x2)
        elecvxs = dpar_elec['ue'][gptsparticleelec][:]*vte0
        elecvys = dpar_elec['ve'][gptsparticleelec][:]*vte0
        elecvzs = dpar_elec['we'][gptsparticleelec][:]*vte0

        elecvxbins.append(elecvxs)
        elecvybins.append(elecvys)
        elecvzbins.append(elecvzs)
    if(verbose):print("done binning particles!")

    #compute bulk v
    if(verbose):print("computing bulk vel...")
    elecvx = []
    elecvy = []
    elecvz = []
    for evxs in elecvxbins:
        if(len(evxs) > 0):
            elecvx.append(np.mean(evxs))
        else:
            elecvx.append(0.)
    for evys in elecvybins:
        if(len(evys) > 0):
            elecvy.append(np.mean(evys))
        else:
            elecvy.append(0.)
    for evzs in elecvzbins:
        if(len(evzs) > 0):
            elecvz.append(np.mean(evzs))
        else:
            elecvz.append(0.)

    #compute perp pressure
    pperp = []
    nspec = []
    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        bvxe = np.mean(elecvxbins[_i])
        bvye = np.mean(elecvybins[_i])
        bvze = np.mean(elecvzbins[_i])

        px = me*np.sum([(evxval-bvxe)**2 for evxval in elecvxbins[_i]]) 
        py = me*np.sum([(evyval-bvye)**2 for evyval in elecvybins[_i]])
        pz = me*np.sum([(evzval-bvze)**2 for evzval in elecvzbins[_i]])

        nspec.append(float(len(elecvzbins[_i])))

        #Assume that \mathbf{B} = B(x) \hat{y} (or well approximated as such)
        ppr = (px+pz)/3.
        pperp.append(ppr)
    pperp = np.asarray(pperp)

    bxlist = []
    bylist = []
    bzlist = []
    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        goodfieldpts = (x1 < dfields['ey_xx']) & (dfields['ey_xx'] <= x2)
        Bxs = dfields['bx']
        Bys = dfields['by']
        Bzs = dfields['bz']

        Bx = np.mean(Bxs[:,:,goodfieldpts])
        By = np.mean(Bys[:,:,goodfieldpts])
        Bz = np.mean(Bzs[:,:,goodfieldpts])

        bxlist.append(Bx)
        bylist.append(By)
        bzlist.append(Bz)
    bxlist = np.asarray(bxlist)
    bylist = np.asarray(bylist)
    bzlist = np.asarray(bzlist)
    
    #compute gradb drift
    ugradx = []
    ugrady = []
    ugradz = []

    #TODO convert above to FAC? (if so, re do comments above) <doesnt matter for our perp simulation that much>
    interdx = interpolxxs[1]-interpolxxs[0]
    gradnormfac = params['comp']*np.sqrt(params['mi']/params['me'])/interdx
    grad_by_wrt_x = np.gradient(bylist)*(gradnormfac)
    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        Bxs = dfields['bx']
        Bys = dfields['by']
        Bzs = dfields['bz']

        goodfieldpts = (x1 < dfields['ey_xx']) & (dfields['ey_xx'] <= x2)
        Bx = np.mean(Bxs[:,:,goodfieldpts])
        By = np.mean(Bys[:,:,goodfieldpts])
        Bz = np.mean(Bzs[:,:,goodfieldpts])

        ne = nspec[_i]

        ugb = (params['c']/(qe*ne))*pperp[_i]/np.linalg.norm([Bx,By,Bz])**3*np.cross([Bx,By,Bz],[grad_by_wrt_x[_i],0,0])


        ugradx.append(ugb[0])
        ugrady.append(ugb[1])
        ugradz.append(ugb[2])
    ugradx = np.asarray(ugradx)
    ugrady = np.asarray(ugrady)
    ugradz = np.asarray(ugradz)

    #re normalize fields 
    fieldkeys = ['ex','ey','ez','bx','by','bz']
    for fk in fieldkeys:
        if(fk[0] == 'e'):
            dfields[fk] /= enorm
        else:
            dfields[fk] /= bnorm

    return positions, ugradx/vti0, ugrady/vti0, ugradz/vti0, nspec, elecvx, elecvy, elecvz

def compute_mag_drift(dfields,dpar_elec,params,interpolxxs,verbose=False):
    """

    """

    #get constants
    if(verbose):print('Computing parameters...')

    me = params['me']

    qe = dpar_elec['q']

    positions = np.asarray([(interpolxxs[_i]+interpolxxs[_i+1])/2. for _i in range(len(interpolxxs)-1)])
    
    #revert normalization of fields
    if(verbose):print('Reverting fields normalization...')
    bnorm = params['c']**2*params['sigma']/params['comp']
    sigma_ion = params['sigma']*params['me']/params['mi'] #NOTE: this is subtely differetn than what aaron's normalization is- fix it (missingn factor of gamma0 and mi+me)
    enorm = bnorm*np.sqrt(sigma_ion)*params['c'] #note, there is an extra factor of 'c hat' (c in code units, which is .45 for the main run being analyzed) that we take out
    fieldkeys = ['ex','ey','ez','bx','by','bz']
    for fk in fieldkeys:
        if(fk[0] == 'e'):
            dfields[fk] *= enorm
        else:
            dfields[fk] *= bnorm
     
    #bin particles
    if(verbose):print("Binning particles...")
    elecvxbins = []
    elecvybins = []
    elecvzbins = []

    vti0 = np.sqrt(params['delgam'])
    vte0 = np.sqrt(params['mi']/params['me'])*vti0 #WARNING: THIS ASSUME Ti/Te = 1, TODO: don't assume Ti/Te = 1

    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]

        gptsparticleelec = (x1 < dpar_elec['xe']) & (dpar_elec['xe'] <= x2)
        elecvxs = dpar_elec['ue'][gptsparticleelec][:]*vte0
        elecvys = dpar_elec['ve'][gptsparticleelec][:]*vte0
        elecvzs = dpar_elec['we'][gptsparticleelec][:]*vte0

        elecvxbins.append(elecvxs)
        elecvybins.append(elecvys)
        elecvzbins.append(elecvzs)
    if(verbose):print("done binning particles!")

    #compute bulk v
    if(verbose):print("computing bulk vel...")
    elecvx = []
    elecvy = []
    elecvz = []
    for evxs in elecvxbins:
        if(len(evxs) > 0):
            elecvx.append(np.mean(evxs))
        else:
            elecvx.append(0.)
    for evys in elecvybins:
        if(len(evys) > 0):
            elecvy.append(np.mean(evys))
        else:
            elecvy.append(0.)
    for evzs in elecvzbins:
        if(len(evzs) > 0):
            elecvz.append(np.mean(evzs))
        else:
            elecvz.append(0.)

    #compute perp pressure
    pperp = []
    nspec = []
    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        bvxe = np.mean(elecvxbins[_i])
        bvye = np.mean(elecvybins[_i])
        bvze = np.mean(elecvzbins[_i])

        px = me*np.sum([(evxval-bvxe)**2 for evxval in elecvxbins[_i]]) 
        py = me*np.sum([(evyval-bvye)**2 for evyval in elecvybins[_i]])
        pz = me*np.sum([(evzval-bvze)**2 for evzval in elecvzbins[_i]])

        nspec.append(float(len(elecvzbins[_i])))

        #Assume that \mathbf{B} = B(x) \hat{y} (or well approximated as such)
        ppr = (px+pz)/3.
        pperp.append(ppr)
    pperp = np.asarray(pperp)

    bxlist = []
    bylist = []
    bzlist = []
    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        goodfieldpts = (x1 < dfields['ey_xx']) & (dfields['ey_xx'] <= x2)
        Bxs = dfields['bx']
        Bys = dfields['by']
        Bzs = dfields['bz']

        Bx = np.mean(Bxs[:,:,goodfieldpts])
        By = np.mean(Bys[:,:,goodfieldpts])
        Bz = np.mean(Bzs[:,:,goodfieldpts])

        bxlist.append(-Bx/np.linalg.norm([Bx,By,Bz])**2)
        bylist.append(-By/np.linalg.norm([Bx,By,Bz])**2)
        bzlist.append(-Bz/np.linalg.norm([Bx,By,Bz])**2)
    bxlist = np.asarray(bxlist)
    bylist = np.asarray(bylist)
    bzlist = np.asarray(bzlist)
    
    #compute mag drift
    umagx = []
    umagy = []
    umagz = []

    #TODO convert above to FAC? (if so, re do comments above) <doesnt matter for our perp simulation that much>
    interdx = interpolxxs[1]-interpolxxs[0]
    gradnormfac = params['comp']*np.sqrt(params['mi']/params['me'])/interdx
    grad_pperp_over_by_wrt_x = -np.gradient(pperp*bylist)*(gradnormfac)
    grad_pperp_over_bz_wrt_x = np.gradient(pperp*bzlist)*(gradnormfac)
    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        Bxs = dfields['bx']
        Bys = dfields['by']
        Bzs = dfields['bz']

        goodfieldpts = (x1 < dfields['ey_xx']) & (dfields['ey_xx'] <= x2)
        Bx = np.mean(Bxs[:,:,goodfieldpts])
        By = np.mean(Bys[:,:,goodfieldpts])
        Bz = np.mean(Bzs[:,:,goodfieldpts])

        ne = nspec[_i]

        umg = (-params['c']/(qe*ne))*np.array([0,grad_pperp_over_bz_wrt_x[_i],grad_pperp_over_by_wrt_x[_i]])
        
        umagx.append(umg[0])
        umagy.append(umg[1])
        umagz.append(umg[2])
    umagx = np.asarray(umagx)
    umagy = np.asarray(umagy)
    umagz = np.asarray(umagz)

    #re normalize fields 
    fieldkeys = ['ex','ey','ez','bx','by','bz']
    for fk in fieldkeys:
        if(fk[0] == 'e'):
            dfields[fk] /= enorm
        else:
            dfields[fk] /= bnorm

    return positions, umagx/vti0, umagy/vti0, umagz/vti0, nspec, elecvx, elecvy, elecvz

def compute_diamag_drift2D(dfields,temperaturedata2D,params,computefluc=False,computebar=False,verbose=False):
    """
    u_dia = -1/(q_e n_e)\frac{nabla p_{\perp,e} \times \mathbf{B}}{|\mathbf{B}|^2}

    Claim: u_dia is the thing causing the adiabatic heating. That is u_dia is caused
    by the thing conserving the adiabatic invariant, which we will call adiabatic heating.

    The 'equivalency' of u_dia and adiabatic heating was shown by Juno et al 2021 (assumes no scattering tho!)

    Note, the above claim assumes a sufficiently high mass ratio   

    This function has been tested and agrees with the 1d version
    """

    #get constants
    if(verbose):print('Computing parameters...')

    vti0 = 1.

    qe = -1

    ny = len(temperaturedata2D['elecyys'])
    nx = len(temperaturedata2D['elecxxs'])

    positionsxx = temperaturedata2D['elecxxs']
    positionsyy = temperaturedata2D['elecyys']
    
    udiax = np.zeros((nx,ny))
    udiay = np.zeros((nx,ny))
    udiaz = np.zeros((nx,ny))

    elecperpboxfac = np.asarray(temperaturedata2D['elecperpboxfac'])
    import copy 
    bxvals = copy.deepcopy(dfields['bx'])
    byvals = copy.deepcopy(dfields['by'])
    bzvals = copy.deepcopy(dfields['bz'])
    
    if(computefluc):
        elecperpboxfac = elecperpboxfac - elecperpboxfac.mean(axis=(1), keepdims=True)
        dfluc = FPCAnalysis.anl.remove_average_fields_over_yz(dfields)
        bxvals = copy.deepcopy(dfluc['bx'])
        byvals = copy.deepcopy(dfluc['by'])
        bzvals = copy.deepcopy(dfluc['bz'])
    elif(computebar):
        elecperpboxfac = elecperpboxfac.mean(axis=(1), keepdims=True)
        elecperpboxfac = np.repeat(elecperpboxfac, len(temperaturedata2D['elecxxs']), axis=1)
        dfavg = FPCAnalysis.anl.get_average_fields_over_yz(dfields)
        bxvals = copy.deepcopy(dfavg['bx'])
        byvals = copy.deepcopy(dfavg['by'])
        bzvals = copy.deepcopy(dfavg['bz'])
        
    gradelecperpboxfacx = np.gradient(elecperpboxfac,axis=1)
    gradelecperpboxfacy = np.gradient(elecperpboxfac,axis=0)
    edens = temperaturedata2D['elecdens']
    elecxxs = temperaturedata2D['elecxxs']
    elecyys = temperaturedata2D['elecyys']
    dx = elecxxs[1]-elecxxs[0]
    dy = elecyys[1]-elecyys[0]

    elecxxs = temperaturedata2D['elecxxs']
    elecyys = temperaturedata2D['elecyys']
    facdx = elecxxs[1]-elecxxs[0]
    facdy = elecyys[1]-elecyys[0]
    gradnormfacxx = params['comp']*np.sqrt(params['mi']/params['me'])/facdx
    gradnormfacyy = params['comp']*np.sqrt(params['mi']/params['me'])/facdy
    gradelecperpboxfacx *= gradnormfacxx
    gradelecperpboxfacy *= gradnormfacyy

    for _i in range(0,len(elecxxs)):
        if(verbose):print(_i, " of ", len(elecxxs))
        for _j in range(0,len(elecyys)):
            x1 = elecxxs[_i] - dx/2.
            x2 = elecxxs[_i] + dx/2.
            y1 = elecyys[_j] - dy/2.
            y2 = elecyys[_j] + dy/2.

            goodfieldptsxx = (x1 < dfields['ey_xx']) & (dfields['ey_xx'] <= x2) 
            goodfieldptsyy = (y1 < dfields['ey_yy']) & (dfields['ey_yy'] <= y2)
            Bx = np.mean(bxvals[:,goodfieldptsyy,:][:, :,goodfieldptsxx], axis=(1,2))[0]
            By = np.mean(byvals[:,goodfieldptsyy,:][:, :,goodfieldptsxx], axis=(1,2))[0]
            Bz = np.mean(bzvals[:,goodfieldptsyy,:][:, :,goodfieldptsxx], axis=(1,2))[0]

            ne = edens[_j,_i]
            qe = -1
            udiaxtemp = (-params['c']/(qe*ne))*np.cross(np.array([gradelecperpboxfacx[_j,_i]*edens[_j,_i],gradelecperpboxfacy[_j,_i]*edens[_j,_i],0]),np.array([Bx,By,Bz]))/np.linalg.norm(np.array([Bx,By,Bz]))**2
            udiax[_i,_j] = udiaxtemp[0]
            udiay[_i,_j] = udiaxtemp[1]
            udiaz[_i,_j] = udiaxtemp[2]

    
    return positionsxx, positionsyy, udiax/vti0, udiay/vti0, udiaz/vti0