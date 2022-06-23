# plume.py>

# loading and analysis functions related to plume and comparisons with plume and other misc. theory

import numpy as np
import math

def load_plume_sweep(flnm):
    """
    Load data from plume sweep

    Assumes 2 species

    Parameters
    ----------
    flnm : str
        path to sweep to be loaded

    Returns
    -------
    plume_sweep : dict
        dictionary of data related to plume
    """

    f = open(flnm)

    plume_sweep = {
        "kperp": [],
        "kpar": [],
        "betap": [],
        "vtp": [],
        "w": [],
        "g": [],
        "bxr": [],
        "bxi": [],
        "byr": [],
        "byi": [],
        "bzr": [],
        "bzi": [],
        "exr": [],
        "exi": [],
        "eyr": [],
        "eyi": [],
        "ezr": [],
        "ezi": [],
        "ux1r": [],
        "ux1i": [],
        "uy1r": [],
        "uy1i": [],
        "uz1r": [],
        "uz1i": [],
        "ux2r": [],
        "ux2i": [],
        "uy2r": [],
        "uy2i": [],
        "uz2r": [],
        "uz2i": [],
    }

    line = f.readline()
    while (line != ''):
        line = line.split()
        plume_sweep['kperp'].append(float(line[0]))
        plume_sweep['kpar'].append(float(line[1]))
        plume_sweep['betap'].append(float(line[2]))
        plume_sweep['vtp'].append(float(line[3]))
        plume_sweep['w'].append(float(line[4]))
        plume_sweep['g'].append(float(line[5]))
        plume_sweep['bxr'].append(float(line[6]))
        plume_sweep['bxi'].append(float(line[7]))
        plume_sweep['byr'].append(float(line[8]))
        plume_sweep['byi'].append(float(line[9]))
        plume_sweep['bzr'].append(float(line[10]))
        plume_sweep['bzi'].append(float(line[11]))
        plume_sweep['exr'].append(float(line[12]))
        plume_sweep['exi'].append(float(line[13]))
        plume_sweep['eyr'].append(float(line[14]))
        plume_sweep['eyi'].append(float(line[15]))
        plume_sweep['ezr'].append(float(line[16]))
        plume_sweep['ezi'].append(float(line[17]))
        plume_sweep['ux1r'].append(float(line[18]))
        plume_sweep['ux1i'].append(float(line[19]))
        plume_sweep['uy1r'].append(float(line[20]))
        plume_sweep['uy1i'].append(float(line[21]))
        plume_sweep['uz1r'].append(float(line[22]))
        plume_sweep['uz1i'].append(float(line[23]))
        plume_sweep['ux2r'].append(float(line[24]))
        plume_sweep['ux2i'].append(float(line[25]))
        plume_sweep['uy2r'].append(float(line[26]))
        plume_sweep['uy2i'].append(float(line[27]))
        plume_sweep['uz2r'].append(float(line[28]))
        plume_sweep['uz2i'].append(float(line[29]))

        line = f.readline()

    for key in plume_sweep.keys():
        plume_sweep[key] = np.asarray(plume_sweep[key])

    #normalize B
    plume_sweep['bxr'] = plume_sweep['bxr']*plume_sweep['vtp']
    plume_sweep['byr'] = plume_sweep['byr']*plume_sweep['vtp']
    plume_sweep['bzr'] = plume_sweep['bzr']*plume_sweep['vtp']
    plume_sweep['bxi'] = plume_sweep['bxi']*plume_sweep['vtp']
    plume_sweep['byi'] = plume_sweep['byi']*plume_sweep['vtp']
    plume_sweep['bzi'] = plume_sweep['bzi']*plume_sweep['vtp']

    return plume_sweep

def rotate_and_norm_to_plume_basis(wavemode,epar,eperp1,eperp2,comp_error_prop=False):
    """
    Rotates wavemode around epar axis so that eperp2 = 0

    WARNING: plume's basis of x,y,z is not the same as our simulations basis of x,y,z
    This notation is overloaded

    Parameters
    ----------
    wavemode : dict
        subdictionary of dwavemodes from compute_wavemodes
    epar,eperp1,eperp2 : [float,float,float]
        vectors of field aligned coordinate basis
    comp_error_prop : bool, opt
        when true, propogates error from delta_kx everywhere
        WARNING: must manually assign delta_kx for wavemode before setting equal to True
    """
    from copy import deepcopy
    plume_basis_wavemode = deepcopy(wavemode)

    #by convention we flip coordinate systems if kpar is negative
    if(plume_basis_wavemode['kpar'] < 0):
        epar = _rotate(math.pi,eperp1,epar)
        eperp2 = _rotate(math.pi,eperp1,eperp2)

    #by convention we rotate about epar until kperp2 is zero
    #i.e. we change our basis vectors so that our wavemode is in the span of two of the field align basis vectors, epar and eperp1
    #note:we assume epar, eperp1, and eperp2 are orthonormal #TODO: check for this

    if(comp_error_prop):
        from uncertainties import ufloat
        try:
            _kx = ufloat(plume_basis_wavemode['kx'],plume_basis_wavemode['delta_kx'])
        except:
            print("Error, please assign key delta_kx to wavemode before passing")
            _kx = ufloat(plume_basis_wavemode['kx'],plume_basis_wavemode['delta_kx']) #purposefully break it anyways after explaining error
    else:
        _kx = plume_basis_wavemode['kx']

    proj = _project_onto_plane(epar,[_kx,plume_basis_wavemode['ky'],plume_basis_wavemode['kz']])
    angl = _angle_between_vecs(proj,eperp1) #note this does not tell us the direction we need to rotate, just the amount
    #angl += math.pi #TODO: check if this is the correct basis #NOTE: unless we rotate by this additional pi, our normfactor is off by a sign factor

    #try first direction
    eperp1 = _rotate(angl,epar,eperp1)
    eperp2 = _rotate(angl,epar,eperp2)

    #if failed, try second direction
    if(np.abs(np.dot(eperp2,[wavemode['kx'],wavemode['ky'],wavemode['kz']])) > 0.01):
        eperp1 = _rotate(-2.*angl,epar,eperp1) #times 2 to make up for first rotation
        eperp2 = _rotate(-2.*angl,epar,eperp2)

    _k = [wavemode['kx'],wavemode['ky'],wavemode['kz']]
    if(comp_error_prop):
        _k[0] = ufloat(wavemode['kx'],wavemode['delta_kx'])
    _E = [plume_basis_wavemode['Ex'],plume_basis_wavemode['Ey'],plume_basis_wavemode['Ez']]
    _B = [plume_basis_wavemode['Bx'],plume_basis_wavemode['By'],plume_basis_wavemode['Bz']]

    #we compute these here, as if epar/eperp1/eperp2 has uncertainties, we either need to define complex operations with uncertainty,
    #or we can save the uncertainty and convert these vars back to simple scalars
    plume_basis_wavemode['kpar'] = np.dot(epar,_k)
    plume_basis_wavemode['kperp1'] = np.dot(eperp1,_k)
    plume_basis_wavemode['kperp2'] = np.dot(eperp2,_k)
    if(comp_error_prop):#only care to track error for kperp, also want to store under seperate key
        plume_basis_wavemode['delta_eperp1'] = [_val.s for _val in eperp1]
        plume_basis_wavemode['delta_eperp2'] = [_val.s for _val in eperp2]
        eperp1 = [_val.n for _val in eperp1]
        eperp2 = [_val.n for _val in eperp2]
        plume_basis_wavemode['delta_kperp1'] = plume_basis_wavemode['kperp1'].s
        plume_basis_wavemode['delta_kperp2'] = plume_basis_wavemode['kperp2'].s
        plume_basis_wavemode['delta_kpar'] = plume_basis_wavemode['kpar'].s
        plume_basis_wavemode['kperp1'] = plume_basis_wavemode['kperp1'].n
        plume_basis_wavemode['kperp2'] = plume_basis_wavemode['kperp2'].n
        plume_basis_wavemode['kpar'] = plume_basis_wavemode['kpar'].n

    #double check rotations
    if(np.abs(np.dot(eperp2,[wavemode['kx'],wavemode['ky'],wavemode['kz']])) > 0.01):
        print("Error, rotation did not result in kperp2 ~= 0")
    if(np.abs(np.dot(epar,eperp1)) > .01 or np.abs(np.dot(eperp1,eperp2)) > .01 or np.abs(np.dot(epar,eperp2)) > .01):
        print("Error, basis is no longer orthogonal...")
    if(np.abs(np.linalg.norm(epar)-1.) > .01 or np.abs(np.linalg.norm(eperp1)-1.) > .01 or np.abs(np.linalg.norm(eperp2)-1.) > .01):
        print("Error, basis is no longer normal...")

    #by convention we normalize so that Eperp1 = 1+0i
    normfactor = np.dot(eperp1,[plume_basis_wavemode['Ex'],plume_basis_wavemode['Ey'],plume_basis_wavemode['Ez']])
    plume_basis_wavemode['Ex'] /= normfactor
    plume_basis_wavemode['Ey'] /= normfactor
    plume_basis_wavemode['Ez'] /= normfactor
    plume_basis_wavemode['Bx'] /= normfactor
    plume_basis_wavemode['By'] /= normfactor
    plume_basis_wavemode['Bz'] /= normfactor

    #recomputed all quantities that are impacted by rotation and normalization
    plume_basis_wavemode['normB'] = np.linalg.norm([plume_basis_wavemode['Bx'],plume_basis_wavemode['By'],plume_basis_wavemode['Bz']])
    plume_basis_wavemode['normE'] = np.linalg.norm([plume_basis_wavemode['Ex'],plume_basis_wavemode['Ey'],plume_basis_wavemode['Ez']])

    #if(comp_error_prop):
    #    print("Warning, we should consider tracking error of fields...")
    #    throw = error#TODO: grab errors for fields?

    plume_basis_wavemode['eperp1'] = eperp1
    plume_basis_wavemode['eperp2'] = eperp2
    plume_basis_wavemode['epar'] = epar

    plume_basis_wavemode['kperp'] = math.sqrt(plume_basis_wavemode['kperp1']**2+plume_basis_wavemode['kperp2']**2)

    plume_basis_wavemode['Epar'] = np.dot(epar,_E)
    plume_basis_wavemode['Eperp1'] = np.dot(eperp1,_E)
    plume_basis_wavemode['Eperp2'] = np.dot(eperp2,_E)
    plume_basis_wavemode['Bpar'] = np.dot(epar,_B)
    plume_basis_wavemode['Bperp1'] = np.dot(eperp1,_B)
    plume_basis_wavemode['Bperp2'] = np.dot(eperp2,_B)

    _EcrossB = np.cross(_E,_B)
    plume_basis_wavemode['EcrossBx'] = _EcrossB[0]
    plume_basis_wavemode['EcrossBy'] = _EcrossB[1]
    plume_basis_wavemode['EcrossBz'] = _EcrossB[2]
    plume_basis_wavemode['normEcrossB'] = np.linalg.norm(_EcrossB)
    plume_basis_wavemode['EcrossBpar'] = np.dot(epar,_EcrossB)
    plume_basis_wavemode['EcrossBperp1'] = np.dot(eperp1,_EcrossB)
    plume_basis_wavemode['EcrossBperp2'] = np.dot(eperp2,_EcrossB)

    return plume_basis_wavemode

def kaw_curve(kperp,kpar,beta_i,tau,
              comp_error_prop=False,delta_kperp = 0., delta_kpar = 0., delta_beta_i = 0., delta_tau = 0.):
    """
    Empirical dispersion relation for a kinetic alfven wave

    From Howes et al. 2014

    Parameters
    ----------
    kperp : float
        kperp value
    kpar : float
        kpar value
    beta_i : float
        ion plasma beta
    tau : float
        Ti/Te temperature ratio
    comp_error_prop : bool, opt
        if true, propogates error
    delta_kperp,delta_kpar,delta_beta_i,delta_tau : float
        source error

    Returns
    -------
    omega_over_Omega_i : float
        ratio of freq to gyrofreq
    """

    if(comp_error_prop):
        from uncertainties import ufloat
        from uncertainties.umath import sqrt

        kperp = ufloat(kperp,delta_kperp)
        kpar = ufloat(kpar,delta_kpar)
        beta_i = ufloat(beta_i,delta_beta_i)
        tau = ufloat(tau,delta_tau)

        omega_over_Omega_i=kpar*sqrt(1.+ kperp**2./(1.+ 2./(beta_i*(1.+1./tau))))

    else:
        omega_over_Omega_i=kpar*math.sqrt(1.+ kperp**2./(1.+ 2./(beta_i*(1.+1./tau))))

    return omega_over_Omega_i

def fastmagson_curve(kperp,kpar,beta_i,tau,
              comp_error_prop=False,delta_kperp = 0., delta_kpar = 0., delta_beta_i = 0., delta_tau = 0.):
    """
    Dispersion for fast magnetosonic wave

    From Klein et al. 2012

    Parameters
    ----------
    kperp : float
        kperp value
    kpar : float
        kpar value
    beta_i : float
        ion plasma beta
    tau : float
        Ti/Te temperature ratio
    comp_error_prop : bool, opt
        if true, propogates error
    delta_kperp,delta_kpar,delta_beta_i,delta_tau : float
        source error

    Returns
    -------
    omega_over_Omega_i : float
        ratio of freq to gyrofreq
    """

    if(comp_error_prop):
        from uncertainties import ufloat
        from uncertainties.umath import sqrt

        kperp = ufloat(kperp,delta_kperp)
        kpar = ufloat(kpar,delta_kpar)
        beta_i = ufloat(beta_i,delta_beta_i)
        tau = ufloat(tau,delta_tau)

        bt = beta_i*(1+1./tau)
        k_tot = sqrt(kperp**2.+kpar**2.)

        omega_over_Omega_i=k_tot*sqrt( (1.+bt + sqrt( (1.+bt)**2. -4.*bt*(kpar/k_tot)**2.))/2.)

    else:
        bt = beta_i*(1+1./tau)
        k_tot = math.sqrt(kperp**2.+kpar**2.)

        omega_over_Omega_i=k_tot*math.sqrt( (1.+bt + math.sqrt( (1.+bt)**2. -4.*bt*(kpar/k_tot)**2.))/2.)

    return omega_over_Omega_i

def slowmagson_curve(kperp,kpar,beta_i,tau,
              comp_error_prop=False,delta_kperp = 0., delta_kpar = 0., delta_beta_i = 0., delta_tau = 0.):
    """
    Dispersion for slow magnetosonic wave

    From Klein et al. 2012

    Parameters
    ----------
    kperp : float
        kperp value
    kpar : float
        kpar value
    beta_i : float
        ion plasma beta
    tau : float
        Ti/Te temperature ratio
    comp_error_prop : bool, opt
        if true, propogates error
    delta_kperp,delta_kpar,delta_beta_i,delta_tau : float
        source error

    Returns
    -------
    omega_over_Omega_i : float
        ratio of freq to gyrofreq
    """
    if(comp_error_prop):
        from uncertainties import ufloat
        from uncertainties.umath import sqrt

        kperp = ufloat(kperp,delta_kperp)
        kpar = ufloat(kpar,delta_kpar)
        beta_i = ufloat(beta_i,delta_beta_i)
        tau = ufloat(tau,delta_tau)

        bt = beta_i*(1+1./tau)
        k_tot = sqrt(kperp**2.+kpar**2.)

        omega_over_Omega_i=k_tot*sqrt( (1.+bt - sqrt( (1.+bt)**2. -4.*bt*(kpar/k_tot)**2.))/2.)

    else:
        bt = beta_i*(1+1./tau)
        k_tot = math.sqrt(kperp**2.+kpar**2.)

        omega_over_Omega_i=k_tot*math.sqrt( (1.+bt - math.sqrt( (1.+bt)**2. -4.*bt*(kpar/k_tot)**2.))/2.)

    return omega_over_Omega_i

def whistler_curve(kperp,kpar,beta_i,tau,
              comp_error_prop=False,delta_kperp = 0., delta_kpar = 0., delta_beta_i = 0., delta_tau = 0.):
    """
    Analytical limit of fastmagson_curve(kperp,kpar) when kpar << kperp

    Parameters
    ----------
    kperp : float
        kperp value
    kpar : float
        kpar value
    beta_i : float
        ion plasma beta
    tau : float
        Ti/Te temperature ratio
    comp_error_prop : bool, opt
        if true, propogates error
    delta_kperp,delta_kpar,delta_beta_i,delta_tau : float
        source error

    Returns
    -------
    omega_over_Omega_i : float
        ratio of freq to gyrofreq
    """
    if(comp_error_prop):
        from uncertainties import ufloat
        from uncertainties.umath import sqrt

        kperp = ufloat(kperp,delta_kperp)
        kpar = ufloat(kpar,delta_kpar)
        beta_i = ufloat(beta_i,delta_beta_i)
        tau = ufloat(tau,delta_tau)

        k_tot = sqrt(kperp**2.+kpar**2.)
        omega_over_Omega_i=k_tot*sqrt(1.+beta_i*(1.+1./tau) + kpar**2.)
    else:
        k_tot = math.sqrt(kperp**2.+kpar**2.)
        omega_over_Omega_i=k_tot*math.sqrt(1.+beta_i*(1.+1./tau) + kpar**2.)

    return omega_over_Omega_i

def test_curve(kperp,kpar,beta_i,tau,
              comp_error_prop=False,delta_kperp = 0., delta_kpar = 0., delta_beta_i = 0., delta_tau = 0.):
    """
    Analytical limit of fastmagson_curve(kperp,kpar) when kpar << kperp

    Parameters
    ----------
    kperp : float
        kperp value
    kpar : float
        kpar value
    beta_i : float
        ion plasma beta
    tau : float
        Ti/Te temperature ratio
    comp_error_prop : bool, opt
        if true, propogates error
    delta_kperp,delta_kpar,delta_beta_i,delta_tau : float
        source error

    Returns
    -------
    omega_over_Omega_i : float
        ratio of freq to gyrofreq
    """
    
    #MTSI
    U = 6
    massratio = 1836

    if(comp_error_prop):
        from uncertainties import ufloat
        from uncertainties.umath import sqrt

        kperp = ufloat(kperp,delta_kperp)
        kpar = ufloat(kpar,delta_kpar)
        beta_i = ufloat(beta_i,delta_beta_i)
        tau = ufloat(tau,delta_tau)

        k_tot = sqrt(kperp**2.+kpar**2.)
        omega_over_Omega_i=k_tot
    else:
        k_tot = math.sqrt(kperp**2.+kpar**2.)
        omega_over_Omega_i=k_tot

    return omega_over_Omega_i

def select_wavemodes(dwavemodes, depth, kpars, kperps, tol=0.05):
    """
    Grabs wavemodes with given kpar and kperp from kpar/kperp list

    Searches only the first 0 to depth in dwavemodess

    Parameters
    ----------
    dwavemodes : dict
        dict returned by compute wavemodes
    depth : int
        depth to search dictionary
    kpars/kperps : array
        parallel arrays that specify kpar and kperp of desired wavemode
    tol : float, opt
        tolerance of agreement between wavemode and given kpar/kperp
    """
    from lib.plume import rotate_and_norm_to_plume_basis
    #iterate over kpars
    wavemodes_matching_kpar = []

    for kpar,kperp in zip(kpars,kperps):
        _wvmds = {'wavemodes':[]}

        #pick out wavemodes
        for k in range(0,depth):
            wvmd = rotate_and_norm_to_plume_basis(dwavemodes['wavemodes'][k],dwavemodes['epar'],dwavemodes['eperp1'],dwavemodes['eperp2']) #need to be in plume basis so that kperp2 = 0
            if(np.abs(wvmd['kpar']-kpar)<tol and np.abs(wvmd['kperp']-kperp)<tol):
                _wvmds['wavemodes'].append(wvmd)

        #save to output arrays
        wavemodes_matching_kpar.append(_wvmds)

    #iterate over wavemodes
    wavemodes_matching_kperp = []
    for kpar,kperp in zip(kpars,kperps):
        _wvmds = {'wavemodes':[]}

        #pick out wavemodes
        for k in range(0,depth):
            wvmd = rotate_and_norm_to_plume_basis(dwavemodes['wavemodes'][k],dwavemodes['epar'],dwavemodes['eperp1'],dwavemodes['eperp2']) #need to be in plume basis so that kperp2 = 0
            if(np.abs(wvmd['kpar']-kpar)<tol and np.abs(wvmd['kperp']-kperp)<tol):
                _wvmds['wavemodes'].append(wvmd)

        #save to output arrays
        wavemodes_matching_kperp.append(_wvmds)

    #return
    return wavemodes_matching_kpar, wavemodes_matching_kperp #TODO: these are the same thing, only return 1 thing and fix elsewhere

def get_freq_from_wvmd(wm,tol=0.01, comp_error_prop=False,debug=False):
    """
    Computes frequency using faradays law of given wavemode

    There are different returns depending on comp_error_prop

    Note: there are 3 equations that we can derive using faraday's law, assuming plane wave solutions, and k = kperp + kpar

    TODO: there might be two different 'tol' vars in this function. remove/rename one

    Parameters
    ----------
    wm : dict
        wavemode from rotate_and_norm_to_plume_basis
    tol : float
        tolerance of agreement between omega1 and omega3
    comp_error_prop : bool, optional
        if true, propogates error
        note: must propgate error in rotate_and_norm_to_plume_basis first
    debug : bool, optional
        if true, will print debug statements

    Returns (comp_eror_prop == True)
    --------------------------------
    omega1real, omega1imag, omega2real, omega2imag, omega3real, omega3imag : ufloat
        omega/Omega_i using different constraints
        must break up like this as ufloat does not handle complex numbers yet

    Returns (comp_eror_prop == False)
    --------------------------------
    omega1, omega2, omega3 : float
        omega/Omega_i using different constraints
    """
    if(np.abs(wm['kperp2'])>tol):
        print("WARNING: not in correct basiss... Please normalize such that kperp2 = 0 (see rotate_and_norm_to_plume_basis())...")

    #consistency check using div B = 0
    kdotb=(wm['Bperp1']*wm['kperp1']+wm['Bpar']*wm['kpar'])/(np.linalg.norm([wm['kperp1'],wm['kperp2'],wm['kpar']])*np.linalg.norm([wm['Bperp1'],wm['Bperp2'],wm['Bpar']]))
    if(np.abs(kdotb) > 0.1):
        print("Warning, div B != 0: div B = " + str(kdotb))

    if(comp_error_prop):
        from uncertainties import ufloat
        from uncertainties.umath import sqrt

        #should have delta_kpar and delta_kperp at this stage
        kperp1 = ufloat(wm['kperp1'],wm['delta_kperp1'])
        kperp2 = ufloat(wm['kperp2'],wm['delta_kperp2'])
        kpar = ufloat(wm['kpar'],wm['delta_kpar'])

        omega1 = (-kpar.n/wm['Bperp1'])*wm['Eperp2'] #kpar is assumed to have no error in it
        omega1_error = (-kpar.s/wm['Bperp1'])*wm['Eperp2']
        omega1real = ufloat(omega1.real,np.abs(omega1_error.real))
        omega1imag = ufloat(omega1.imag,np.abs(omega1_error.imag))

        omega2 = -(1./wm['Bperp2'])*(wm['kpar']*wm['Eperp1']-kperp1.n*wm['Epar'])
        omega2_error = -(1./wm['Bperp2'])*(kperp1.s*wm['Epar']) #uncertainties does not handle complex numbers yet, but fornuately, as these complex numbers are scalars w/o error, it's trivial to do by hand
        omega2real = ufloat(omega2.real,np.abs(omega2_error.real))
        omega2imag = ufloat(omega2.imag,np.abs(omega2_error.imag))

        omega3 = kperp1.n/wm['Bpar']*wm['Eperp2']
        omega3_error = kperp1.s/wm['Bpar']*wm['Eperp2'] #uncertainties does not handle complex numbers yet, but fornuately, as these complex numbers are scalars w/o error, it's trivial to do by hand
        omega3real = ufloat(omega3.real,np.abs(omega3_error.real))
        omega3imag = ufloat(omega3.imag,np.abs(omega3_error.imag))

        tol = 0.5
        if(np.abs(np.abs(omega1)-np.abs(omega3))>tol and debug):
            print("WARNING!!! Omega1 != omega3 using our faradays law!!! Previously, we have assumed these two equations were equivalent")

            print('omega1',omega1)
            print('omega3',omega3)

        return omega1real, omega1imag, omega2real, omega2imag, omega3real, omega3imag

    else:
        #get omega using first constraint
        omega1 = (-wm['kpar']/wm['Bperp1'])*wm['Eperp2']

        #get omega using first constraint
        omega2 = -(1./wm['Bperp2'])*(wm['kpar']*wm['Eperp1']-wm['kperp1']*wm['Epar'])

        #get omega using second constraint
        omega3 = wm['kperp1']/wm['Bpar']*wm['Eperp2']

        tol = 0.5
        if(np.abs(np.abs(omega1)-np.abs(omega3))>tol and debug):
            print("WARNING!!! Omega1 != omega3 using our faradays law!!! Previously, we have assumed these two equations were equivalent")

            print('omega1',omega1)
            print('omega3',omega3)
        return omega1, omega2, omega3

def _project_onto_plane(norm,vec):
    """
    Projects vector onto plane

    Parameters
    ----------
    norm : [float,float,float]
        normal of plane
    vec : [float,float,float]
        vector to be projected

    Returns
    -------
    projection : [float,float,float]
        projection onto plane
    """
    norm = np.asarray(norm)
    vec = np.asarray(vec)
    projection = vec-np.dot(vec,norm)*norm/(np.linalg.norm(norm)**2.)

    return projection

def _angle_between_vecs(vec1,vec2):
    """
    Computes angle between two vectors

    Parameters
    ----------
    vec1,vec2 : [float,float,float]
        vectors

    Returns
    -------
    tht : float
        angle between vectors
    """
    _vec1 = np.asarray(vec1)
    _vec2 = np.asarray(vec2)

    try: #TODO: use keyword parameter if trying to propogate error instead of a try except block
        tht = np.arccos(np.dot(_vec1,_vec2)/(np.linalg.norm(_vec1)*np.linalg.norm(_vec2)))
    except:
        from uncertainties import unumpy
        from uncertainties.umath import sqrt
        from uncertainties import ufloat

        len1 = sqrt(vec1[0]**2.+vec1[1]**2.+vec1[2]**2.) #this function uses sqrt, which requires it's own uncertainties function when working with ufloat
        len2 = sqrt(vec2[0]**2.+vec2[1]**2.+vec2[2]**2.)

        tht = unumpy.arccos(np.dot(_vec1,_vec2)/(len1*len2))
        tht = tht.ravel()[0] #above function returns ndarray that does have the wanted attributes used by later functions, must convert back to ufloat

    return tht

def _rotate(tht,rotationaxis,vect):
    """
    Rotates vect about rotationaxis by angle

    Parameters
    ----------
    tht : float
        angle to rotate by
    rotationaxis : [float,float,float]
        axis to rotate around

    Returns
    -------
    rotatedvec : [float,float,float]
        rotated vector
    """
    rotationaxis = np.asarray(rotationaxis)
    vect = np.asarray(vect)

    #normalize rotationaxis
    ux = rotationaxis[0] / np.linalg.norm(rotationaxis)
    uy = rotationaxis[1] / np.linalg.norm(rotationaxis)
    uz = rotationaxis[2] / np.linalg.norm(rotationaxis)

    #Rotation matrix
    try: #TODO: use keyword parameter if trying to propogate error instead of a try except block
        r11 = math.cos(tht)+ux**2.*(1.-math.cos(tht))
        r21 = uy*ux*(1.-math.cos(tht))+uz*math.sin(tht)
        r31 = uz*ux*(1.-math.cos(tht))-uy*math.sin(tht)
        r12 = ux*uy*(1.-math.cos(tht))-uz*math.sin(tht)
        r22 = math.cos(tht)+uy**2.*(1.-math.cos(tht))
        r32 = uz*uy*(1.-math.cos(tht))+ux*math.sin(tht)
        r13 = ux*uz*(1.-math.cos(tht))+uy*math.sin(tht)
        r23 = uy*uz*(1-math.cos(tht))-ux*math.sin(tht)
        r33 = math.cos(tht)+uz**2.*(1.-math.cos(tht))
    except:
        from uncertainties.umath import cos, sin
        from uncertainties import ufloat

        r11 = cos(tht)+ux**2.*(1.-cos(tht))
        r21 = uy*ux*(1.-cos(tht))+uz*sin(tht)
        r31 = uz*ux*(1.-cos(tht))-uy*sin(tht)
        r12 = ux*uy*(1.-cos(tht))-uz*sin(tht)
        r22 = cos(tht)+uy**2.*(1.-cos(tht))
        r32 = uz*uy*(1.-cos(tht))+ux*sin(tht)
        r13 = ux*uz*(1.-cos(tht))+uy*sin(tht)
        r23 = uy*uz*(1-cos(tht))-ux*sin(tht)
        r33 = cos(tht)+uz**2.*(1.-cos(tht))

    R = [[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]]
    #R = [[r11,r21,r31],[r12,r22,r32],[r13,r23,r33]] #TODO: double check if matrix should be inverted

    rotatedvec = np.matmul(R,vect)
    return rotatedvec
