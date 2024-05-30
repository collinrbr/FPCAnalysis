import numpy as np
import math

def compute_wavemodes(xx,dfields,xlim,ylim,zlim,
                     kx,ky,kz,
                     bxkzkykxxx,bykzkykxxx,bzkzkykxxx,
                     exkzkykxxx,eykzkykxxx,ezkzkykxxx,
                     uxkzkykxxx,uykzkykxxx,uzkzkykxxx,
                     eparlocalkzkykxxx=None,epardetrendkzkykxxx=None,
                     eperp1detrendkzkykxxx=None,eperp2detrendkzkykxxx=None,
                     bpardetrendkzkykxxx=None,bperp1detrendkzkykxxx=None,bperp2detrendkzkykxxx=None,
                     vth=None,specifyxxidx=-1,verbose=False,is2d=True):

    """
    #TODO: document new variables

    Creates dwavemodes dictionary, that contains frequency/wavenumber space data of our fields
    in cartesian coordinates and field aligned coordinates for every discrete computed point in
    frequency/wavenumber space

    As we use the wavelet transform, this frequency space data can be localized in xx,
    which we select when computing wavemodes

    Parameters
    ----------
    xx : float
        xx position of the analysis
    dfields : dict
        field data dictionary from field_loader
    xlim : [float,float]
        bounds of box we want to compute field aligned coordinates in xx
    ylim : [float,float]
        bounds of box we want to compute field aligned coordinates in yy
    zlim : [float,float]
        bounds of box we want to compute field aligned coordinates in zz
    kx,ky,kz : 1d array
        arrays containing wavenumbers grid of analysis
    **kzkykxxx : 4d array
        fields/flow that have been transformed using fft in yy,zz and wlt in xx
        e.g. bxkzkykxxx <=> B_x(kz,ky,kx;xx)

    Returns
    -------
    dwavemodes : dict
        wavemode data in frequency/wavenumber space for every
    """

    from lib.analysisaux import compute_field_aligned_coord
    from lib.arrayaux import find_nearest

    dwavemodes = {'wavemodes':[],'sortkey':None}

    #note: xx should be in the middle of xlim TODO: check for this
    epar,eperp1,eperp2 = compute_field_aligned_coord(dfields,xlim,ylim,zlim) #WARNING: dfields should be total fields here

    if(specifyxxidx == -1):
        xxidx = find_nearest(dfields['bz_xx'],xx)
    else:
        xxidx = specifyxxidx

    nkz, nky, nkx, nxx = bxkzkykxxx.shape

    #Build 1d arrays that we can easily sort by desired metric
    for i in range(0,nkx):
        if(verbose):print(str(i)+' of ' + str(nkx))
        for j in range(0,nky):
            for k in range(0,nkz):
                if(is2d):
                    if(kz[k] != 0):
                        continue

                wavemode = {}
                wavemode['kx'] = kx[i]
                wavemode['ky'] = ky[j]
                wavemode['kz'] = kz[k]

                wavemode['Bx'] = bxkzkykxxx[k,j,i,xxidx]
                wavemode['By'] = bykzkykxxx[k,j,i,xxidx]
                wavemode['Bz'] = bzkzkykxxx[k,j,i,xxidx]
                wavemode['Ex'] = exkzkykxxx[k,j,i,xxidx]
                wavemode['Ey'] = eykzkykxxx[k,j,i,xxidx]
                wavemode['Ez'] = ezkzkykxxx[k,j,i,xxidx]
                wavemode['Ux'] = uxkzkykxxx[k,j,i,xxidx]
                wavemode['Uy'] = uykzkykxxx[k,j,i,xxidx]
                wavemode['Uz'] = uzkzkykxxx[k,j,i,xxidx]

                if(eparlocalkzkykxxx is not None):
                    wavemode['Epar_local'] = eparlocalkzkykxxx[k,j,i,xxidx]
                if(epardetrendkzkykxxx is not None):
                    wavemode['Epar_detrend'] = epardetrendkzkykxxx[k,j,i,xxidx]
                if(eperp1detrendkzkykxxx is not None):
                    wavemode['Eperp1_detrend'] = eperp1detrendkzkykxxx[k,j,i,xxidx]
                if(eperp2detrendkzkykxxx is not None):
                    wavemode['Eperp2_detrend'] = eperp2detrendkzkykxxx[k,j,i,xxidx]
                if(eparlocalkzkykxxx is not None):
                    wavemode['Epar_local'] = eparlocalkzkykxxx[k,j,i,xxidx]
                if(bpardetrendkzkykxxx is not None):
                    wavemode['Bpar_detrend'] = bpardetrendkzkykxxx[k,j,i,xxidx]
                if(bperp1detrendkzkykxxx is not None):
                    wavemode['Bperp1_detrend'] = bperp1detrendkzkykxxx[k,j,i,xxidx]
                if(bperp2detrendkzkykxxx is not None):
                    wavemode['Bperp2_detrend'] = bperp2detrendkzkykxxx[k,j,i,xxidx]


                wavemode['normB'] = np.linalg.norm([wavemode['Bx'],wavemode['By'],wavemode['Bz']])
                wavemode['normE'] = np.linalg.norm([wavemode['Ex'],wavemode['Ey'],wavemode['Ez']])

                _k = [wavemode['kx'],wavemode['ky'],wavemode['kz']]
                _E = [wavemode['Ex'],wavemode['Ey'],wavemode['Ez']]
                _B = [wavemode['Bx'],wavemode['By'],wavemode['Bz']]

                wavemode['kpar'] = np.dot(epar,_k)
                wavemode['kperp1'] = np.dot(eperp1,_k)
                wavemode['kperp2'] = np.dot(eperp2,_k)
                wavemode['kperp'] = math.sqrt(wavemode['kperp1']**2+wavemode['kperp2']**2)

                wavemode['Epar'] = np.dot(epar,_E)
                wavemode['Eperp1'] = np.dot(eperp1,_E)
                wavemode['Eperp2'] = np.dot(eperp2,_E)
                wavemode['Bpar'] = np.dot(epar,_B)
                wavemode['Bperp1'] = np.dot(eperp1,_B)
                wavemode['Bperp2'] = np.dot(eperp2,_B)

                _EcrossB = np.cross(_E,_B)
                wavemode['EcrossBx'] = _EcrossB[0]
                wavemode['EcrossBy'] = _EcrossB[1]
                wavemode['EcrossBz'] = _EcrossB[2]
                wavemode['normEcrossB'] = np.linalg.norm(_EcrossB)
                wavemode['EcrossBpar'] = np.dot(epar,_EcrossB)
                wavemode['EcrossBperp1'] = np.dot(eperp1,_EcrossB)
                wavemode['EcrossBperp2'] = np.dot(eperp2,_EcrossB)

                #vector wave coordinate system
                e2 = _k/np.linalg.norm(_k)
                e3 = epar
                e1 = np.cross(e2,e3)
                e1 = e1/np.linalg.norm(e1)
                wavemode['Ee1'] = np.dot(e1,_E)
                wavemode['Ee2'] = np.dot(e2,_E)
                wavemode['Ee3'] = np.dot(e3,_E)
                wavemode['Be1'] = np.dot(e1,_B)
                wavemode['Be2'] = np.dot(e2,_B)
                wavemode['Be3'] = np.dot(e3,_B)
                wavemode['EcrossBe1'] = np.dot(e1,_EcrossB)
                wavemode['EcrossBe2'] = np.dot(e2,_EcrossB)
                wavemode['EcrossBe3'] = np.dot(e3,_EcrossB)

                #consistency check
                wavemode['kdotB'] = np.dot(_k,_B)/(np.linalg.norm(_k)*np.linalg.norm(_B)) #should be approx zero

                if(vth != None):
                    wavemode['vth']=vth

                dwavemodes['wavemodes'].append(wavemode)

    dwavemodes['wavemodes'] = np.asarray(dwavemodes['wavemodes'])
    dwavemodes['epar'] = epar
    dwavemodes['eperp1'] = eperp1
    dwavemodes['eperp2'] = eperp2

    return dwavemodes

def sort_wavemodes(dwavemodes,key):
    """
    Sorts wavemode dictionary by provided key

    Paramters
    ---------
    dwavemodes : dict
        dictionary returned by compute_wavemodes

    Returns
    -------
    dwavemodes : dict
        sorted dictionary returned by compute_wavemodes
    """
    #sorts by key

    #get sort index structure
    _temparray = np.asarray([wmd[key] for wmd in dwavemodes['wavemodes']])
    _sortidx = np.flip(_temparray.argsort())

    #sort
    dwavemodes['wavemodes'] = np.asarray(dwavemodes['wavemodes'])[_sortidx]
    dwavemodes['sortkey'] = key

    return dwavemodes

def get_freq_from_wvmd(wm,tol=0.01, comp_error_prop=False,debug=True,usedetrend=False):
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
    if(np.abs(wm['kperp2'])>tol and debug):
        print("WARNING: not in correct basiss... Please normalize such that kperp2 = 0 (see rotate_and_norm_to_plume_basis())...")

    #consistency check using div B = 0
    kdotb=(wm['Bperp1']*wm['kperp1']+wm['Bpar']*wm['kpar'])/(np.linalg.norm([wm['kperp1'],wm['kperp2'],wm['kpar']])*np.linalg.norm([wm['Bperp1'],wm['Bperp2'],wm['Bpar']]))
    if(np.abs(kdotb) > 0.1):
        print("Warning, div B != 0: div B = " + str(kdotb))

    #TODO: comp error prop for detrend
    if(usedetrend):
         #get omega using first constraint
        omega1 = -(wm['kpar']/wm['Bperp1_detrend'])*wm['Eperp2_detrend']

        #get omega using first constraint
        omega2 = -(1./wm['Bperp2_detrend'])*(wm['kpar']*wm['Eperp1_detrend']-wm['kperp1']*wm['Epar_detrend'])

        #get omega using second constraint
        omega3 = wm['kperp1']/wm['Bpar_detrend']*wm['Eperp2_detrend']

        return omega1,omega2,omega3

    if(comp_error_prop):
        from uncertainties import ufloat
        from uncertainties.umath import sqrt

        #should have delta_kpar and delta_kperp at this stage
        kperp1 = ufloat(wm['kperp1'],wm['delta_kperp1'])
        kperp2 = ufloat(wm['kperp2'],wm['delta_kperp2'])
        kpar = ufloat(wm['kpar'],wm['delta_kpar'])

        omega1 = -(kpar.n/wm['Bperp1'])*wm['Eperp2'] #kpar is assumed to have no error in it
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
        omega1 = -(wm['kpar']/wm['Bperp1'])*wm['Eperp2']

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

def _wavemodes_key_to_label(key):
    """
    Returns plot label provided key from dwavemodes dict

    Parameters
    ----------
    key : string
        key in dwavemodes dict

    Returns
    -------
    lbl : string
        label that can be used with matplotlib to label axis/ tables
    """

    if(key == 'kpar'):
        lbl = r'$k_{||}$'
    elif(key == 'kperp1'):
        lbl = r'$k_{\perp 1}$'
    elif(key == 'kperp2'):
        lbl = r'$k_{\perp 2}$'
    elif(key == 'kperp'):
        lbl = r'$k_\perp$'
    elif(key == 'Epar'):
        lbl = r'$\delta E_{||}$'
    elif(key == 'Eperp1'):
        lbl = r'$\delta E_{\perp 1}$'
    elif(key == 'Eperp2'):
        lbl = r'$\delta E_{\perp 2}$'
    elif(key == 'normE'):
        lbl = r'$||\delta \mathbf{E}||$'
    elif(key == 'normEcrossB'):
        lbl = r'$||\delta \mathbf{E} \times \delta \mathbf{B}||$'
    elif(key == 'Bpar'):
        lbl = r'$\delta B_{||}$'
    elif(key == 'Bperp1'):
        lbl = r'$\delta B_{\perp 1}$'
    elif(key == 'Bperp2'):
        lbl = r'$\delta B_{\perp 2}$'
    elif(key == 'normB'):
        lbl = r'$||\delta \mathbf{B}||$'
    elif(key == 'kx'):
        lbl = r'$k_{x}$'
    elif(key == 'ky'):
        lbl = r'$k_{y}$'
    elif(key == 'kz'):
        lbl = r'$k_{z}$'
    elif(key == 'Ex'):
        lbl = r'$E_{x}$'
    elif(key == 'Ey'):
        lbl = r'$E_{y}$'
    elif(key == 'Ez'):
        lbl = r'$E_{z}$'
    elif(key == 'Bx'):
        lbl = r'$B_{x}$'
    elif(key == 'By'):
        lbl = r'$B_{y}$'
    elif(key == 'Bz'):
        lbl = r'$B_{z}$'
    elif(key == 'EcrossBpar'):
        lbl = r'$(\delta \mathbf{E} \times \delta \mathbf{B})_{||}$'
    elif(key == 'EcrossBperp1'):
        lbl = r'$(\delta \mathbf{E} \times \delta \mathbf{B})_{\perp 1}$'
    elif(key == 'EcrossBperp2'):
        lbl = r'$(\delta \mathbf{E} \times \delta \mathbf{B})_{\perp 2}$'
    elif(key == 'Ee1'):
        lbl = r'$E_{e1}$'
    elif(key == 'Ee2'):
        lbl = r'$E_{e2}$'
    elif(key == 'Ee3'):
        lbl = r'$E_{e3}$'
    elif(key == 'Be1'):
        lbl = r'$B_{e1}$'
    elif(key == 'Be2'):
        lbl = r'$B_{e2}$'
    elif(key == 'Be3'):
        lbl = r'$B_{e3}$'
    elif(key == 'EcrossBe1'):
        lbl = r'$(\delta \mathbf{E} \times \delta \mathbf{B})_{e1}$'
    elif(key == 'EcrossBe2'):
        lbl = r'$(\delta \mathbf{E} \times \delta \mathbf{B})_{e2}$'
    elif(key == 'EcrossBe3'):
        lbl = r'$(\delta \mathbf{E} \times \delta \mathbf{B})_{e3}$'
    elif(key == 'kdotB'):
        lbl = r'$\frac{\mathbf{k} \cdot \mathbf{B}}{||\mathbf{k}|| ||\mathbf{B}||}$'
    else:
        print('Did not find label for ' + key + ' ...')
        lbl = key

    return lbl
