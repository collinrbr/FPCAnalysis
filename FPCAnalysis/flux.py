# analysis.py>

#plasma analysis functions

import numpy as np
import math


def compute_flux_tristanmp1(interpolxxs,dfields,dpar_ion,dpar_elec,params,inputs,verbose=False):
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
    elecqxs = []
    for _i in range(0,len(ionvxbins)):
        ivx = ionvxbins[_i]
        ivy = ionvybins[_i]
        ivz = ionvzbins[_i]
        ionqx = 0.5*mi*np.sum((valx-ionvx[_i])*((valx-ionvx[_i])**2+(valy-ionvy[_i])**2+(valz-ionvz[_i])**2) for valx,valy,valz in zip(ivx,ivy,ivz))
        ionqxs.append(ionqx)
    for _i in range(0,len(elecvxbins)):
        evx = elecvxbins[_i]
        evy = elecvybins[_i]
        evz = elecvzbins[_i]
        elecqx = 0.5*me*np.sum((valx-elecvx[_i])*((valx-elecvx[_i])**2+(valy-elecvy[_i])**2+(valz-elecvz[_i])**2) for valx,valy,valz in zip(evx,evy,evz))
        elecqxs.append(elecqx)

    #compute ram kinetic energy flux (that is this block computes the bulk flow energy, but we call it ram pressure)
    if(verbose):print('compute kinetic energy flux....')
    ionframx = []
    for _i, ivxs in enumerate(ionvxbins):
        fx = ionvx[_i]*0.5*(iondens[_i])*mi*(ionvx[_i]**2+ionvy[_i]**2+ionvz[_i]**2)
        ionframx.append(fx)

    elecframx = []
    for _i, ivxs in enumerate(ionvxbins):
        fx = elecvx[_i]*0.5*(iondens[_i])*me*(elecvx[_i]**2+elecvy[_i]**2+elecvz[_i]**2)
        elecframx.append(fx)

    #compute enthalpy flux first term
    if(verbose):print("compute enthalpy flux....")
    ionethxs = []
    elecethxs = []
    for _i in range(0,len(ionvxbins)):
        ivx = ionvxbins[_i]
        ivy = ionvybins[_i]
        ivz = ionvzbins[_i]
        ionethx = ionvx[_i]*0.5*mi*np.sum((valx-ionvx[_i])**2+(valy-ionvy[_i])**2+(valz-ionvz[_i])**2 for valx,valy,valz in zip(ivx,ivy,ivz))
        ionethxs.append(ionethx)
    for _i in range(0,len(elecvxbins)):
        evx = elecvxbins[_i]
        evy = elecvybins[_i]
        evz = elecvzbins[_i]
        elecethx = elecvx[_i]*0.5*me*np.sum((valx-elecvx[_i])**2+(valy-elecvy[_i])**2+(valz-elecvz[_i])**2 for valx,valy,valz in zip(evx,evy,evz))
        elecethxs.append(elecethx)

    #compute enthalpy flux second term
    ionpdotusxs = []
    elecpdotusxs = []
    for _i in range(0,len(ionvxbins)):
        ibvx = ionvx[_i]  #bulk ion velocity
        ibvy = ionvy[_i]
        ibvz = ionvz[_i]
        ipdu = mi*np.sum((ibvx*(ionvxbins[_i][idx]-ibvx)**2+ibvy*(ionvxbins[_i][idx]-ibvx)*(ionvybins[_i][idx]-ibvy)+ibvz*(ionvxbins[_i][idx]-ibvx)*(ionvzbins[_i][idx]-ibvz) for idx in range(0,len(ionvxbins[_i]))))
        ionpdotusxs.append(ipdu)
    for _i in range(0,len(elecvxbins)):
        ebvx = elecvx[_i]  #bulk elec velocity
        ebvy = elecvy[_i]
        ebvz = elecvz[_i]
        epdu = me*np.sum((ebvx*(elecvxbins[_i][idx]-ebvx)**2+ebvy*(elecvxbins[_i][idx]-ebvx)*(elecvybins[_i][idx]-ebvy)+ebvz*(elecvxbins[_i][idx]-ebvx)*(elecvzbins[_i][idx]-ebvz) for idx in range(0,len(elecvxbins[_i]))))
        elecpdotusxs.append(epdu)

    #compute total energy flux
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
    poyntFluxEfac = valfac*c_thi/(4.*np.pi) #a normalizes field value squared

    #compute poynt flux
    if(verbose):print("computing poynting flux....")
    poyntxxs = []
    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        Eys = dfields['ey']
        Ezs = dfields['ez']
        Bys = dfields['by']
        Bzs = dfields['bz']
        goodfieldpts = (x1 < dfields['ey_xx']) & (dfields['ey_xx'] <= x2)
        pxx = np.sum(Eys[:,:,goodfieldpts]*Bzs[:,:,goodfieldpts]-Ezs[:,:,goodfieldpts]*Bys[:,:,goodfieldpts])
        poyntxxs.append(pxx)
    poyntxxs = poyntFluxEfac*np.asarray(poyntxxs) 

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
        EflucxBflucxx = EflucxBflucxx/float(len(goodfieldpts)) 
        EflucxBfluc_x_out.append(EflucxBflucxx)
        EflucxBflucyy = np.sum(EflucxBfluc_y[:,:,goodfieldpts]) 
        EflucxBflucyy = EflucxBflucyy/float(len(goodfieldpts)) 
        EflucxBfluc_y_out.append(EflucxBflucyy)
        EflucxBfluczz = np.sum(EflucxBfluc_z[:,:,goodfieldpts]) 
        EflucxBfluczz = EflucxBfluczz/float(len(goodfieldpts)) 
        EflucxBfluc_z_out.append(EflucxBfluczz)

        EbarxBbarxx = np.sum(EbarxBbar_x[:,:,goodfieldpts]) 
        EbarxBbarxx = EbarxBbarxx/float(len(goodfieldpts)) 
        EbarxBbar_x_out.append(EbarxBbarxx)
        EbarxBbaryy = np.sum(EbarxBbar_y[:,:,goodfieldpts]) 
        EbarxBbaryy = EbarxBbaryy/float(len(goodfieldpts)) 
        EbarxBbar_y_out.append(EbarxBbaryy)
        EbarxBbarzz = np.sum(EbarxBbar_z[:,:,goodfieldpts]) 
        EbarxBbarzz = EbarxBbarzz/float(len(goodfieldpts))
        EbarxBbar_z_out.append(EbarxBbarzz)
        
        Bbar_dot_bbar_temp =  np.sum(Bbar_dot_bbar[:,:,goodfieldpts])
        Bbar_dot_bbar_temp = Bbar_dot_bbar_temp/float(len(goodfieldpts))
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
    fluxes['ionvy'] = np.asarray(ionvy)
    fluxes['ionvz'] = np.asarray(ionvz)
    fluxes['elecvx'] = np.asarray(elecvx)
    fluxes['elecvy'] = np.asarray(elecvy)
    fluxes['elecvz'] = np.asarray(elecvz)

    fluxes['ionqxs'] = np.asarray(ionqxs)
    fluxes['elecqxs'] = np.asarray(elecqxs)
    fluxes['ionframx'] = np.asarray(ionframx)
    fluxes['elecframx'] = np.asarray(elecframx)
    fluxes['ionethxs'] = np.asarray(ionethxs)
    fluxes['elecethxs'] = np.asarray(elecethxs)
    fluxes['ionpdotusxs'] = np.asarray(ionpdotusxs)
    fluxes['elecpdotusxs'] = np.asarray(elecpdotusxs)
    fluxes['iontotefluxxs'] = np.asarray(iontotefluxxs)
    fluxes['electotefluxxs'] = np.asarray(electotefluxxs)
    
    fluxes['poyntxxs'] = np.asarray(poyntxxs)
    fluxes['Wfields'] = np.asarray(Wfields)
    fluxes['WEfields'] = np.asarray(WEfields)
    fluxes['WBfields'] = np.asarray(WBfields)
    fluxes['Wion'] = np.asarray(Wion)
    fluxes['Welec'] = np.asarray(Welec)
    fluxes['EflucxBfluc_x_avg'] = np.asarray(EflucxBfluc_x_avg)
    fluxes['EflucxBfluc_y_avg'] = np.asarray(EflucxBfluc_y_avg)
    fluxes['EflucxBfluc_z_avg'] = np.asarray(EflucxBfluc_z_avg)
    fluxes['EbarxBbar_x'] = np.asarray(EflucxBfluc_x_avg)
    fluxes['EbarxBbar_y'] = np.asarray(EflucxBfluc_x_avg)
    fluxes['EbarxBbar_z'] = np.asarray(EflucxBfluc_x_avg)
    fluxes['Bbar_dot_bbar'] = np.asarray(EflucxBfluc_x_avg)

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


def compute_diamag_drift(dfields,dpar_elec,dfluxes,interpolxxs,verbose=False):
    """
    u_dia = -1/(q_e n_e)\frac{nabla p_{\perp,e} \times \mathbf{B}}{|\mathbf{B}|^2}

    Claim: u_dia is the thing causing the adiabatic heating. That is u_dia is caused
    by the thing conserving the adiabatic invariant, which we will call adiabatic heating.

    The 'equivalency' of u_dia and adiabatic heating was shown by Juno et al 2021

    Note, the above claim assumes a sufficiently high mass ratio    
    """

    print("NOTE, there might be a factor of c in the grad b drift part of the diamagnetic drift... TODO: figure out and remove this statement")


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

    qe = dpar_elec['q']

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

    #compute perp pressure
    pperp = []
    nspec = []
    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        bvxe = np.mean(elecvxbins[_i])
        bvye = np.mean(elecvybins[_i])
        bvze = np.mean(elecvzbins[_i])

        px = np.sum([(evxval-bvxe)**2 for evxval in elecvxbins[_i]])
        py = np.sum([(evyval-bvxe)**2 for evyval in elecvybins[_i]])
        pz = np.sum([(evzval-bvxe)**2 for evzval in elecvzbins[_i]])

        #we include the 1/ne factor here!
        px /= float(len(elecvxbins[_i]))
        py /= float(len(elecvybins[_i]))
        pz /= float(len(elecvzbins[_i]))

        nspec.append(float(len(elecvzbins[_i])))

        #Assume that \mathbf{B} = B(x) \hat{y} (or well approximated as such)
        ppr = py+pz 
        pperp.append(ppr)
    pperp = np.asarray(pperp)

    #compute diamagnetic drift
    udiax = []
    udiay = []
    udiaz = []
    #TODO convert above to FAC? <doesnt matter for our perp simulation that much>
    gradpperp = np.gradient(pperp)
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

        #TODO: use units here!!!
        ud = -1/qe*np.cross([0,np.gradpperp[_i],0],[Bx,By,Bz])/np.linalg.norm([Bx,By,Bz])**2
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
            dfluc[fk] /= enorm
            dfavg[fk] /= enorm
        else:
            dfields[fk] /= bnorm
            dfluc[fk] /= bnorm
            dfavg[fk] /= bnorm

    return positions, udiax, udiay, udiaz, nspec

