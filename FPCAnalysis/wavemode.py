# analysis.py>

#wavemode analysis functions

import numpy as np
import math

def compute_wavemodes(xx,dfields,xlim,ylim,zlim,
                     kx,ky,kz,
                     bxkzkykxxx,bykzkykxxx,bzkzkykxxx,
                     exkzkykxxx,eykzkykxxx,ezkzkykxxx,
                     uxkzkykxxx=None,uykzkykxxx=None,uzkzkykxxx=None,
                     eparlocalkzkykxxx=None,epardetrendkzkykxxx=None,
                     eperp1detrendkzkykxxx=None,eperp2detrendkzkykxxx=None,
                     bpardetrendkzkykxxx=None,bperp1detrendkzkykxxx=None,bperp2detrendkzkykxxx=None,
                     vth=None,specifyxxidx=-1,verbose=False,is2d=False,morletw=6):

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
    morletw : float
        value of the free tuning parameter in the morlet wavelet used to compute the wlt

    Returns
    -------
    dwavemodes : dict
        wavemode data in frequency/wavenumber space for every
    """

    from FPCAnalysis.analysis import compute_field_aligned_coord
    from FPCAnalysis.array_ops import find_nearest
    from FPCAnalysis.analysis import compute_morletwlt_error

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
                
                if(uxkzkykxxx != None):
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

                wavemode['delta_kx'] = compute_morletwlt_error(wavemode['kx'],morletw)

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

def wavemodes_table(dwavemodes,keys,depth,flnm='',writeToText=False):
    """
    Makes table of properties of wavemodes

    TODO: move to plot lib folder

    Parameters
    ----------
    dwavemodes : dict
        dictionary returned by compute_wavemodes
    keys : array of strings
        keys associated with properties of wavemodes
        call dwavemodes['wavemodes'][0].keys() to see list of possible keys
    depth : int
        number of wavemodes plotted on table in order starting with first wavemode in dict
    flnm : string (opt)
        name of file to save table to
    writeToText : bool (opt)
        if true, will also print out text file containing same data as table
    """
    from FPCAnalysis.plot.table import make_table

    collbls = [_wavemodes_key_to_label(key) for key in keys]
    rowlbls = range(0,depth)

    data = []
    for i in range(0,depth):
        row = [dwavemodes['wavemodes'][i][key] for key in keys]
        for j in range(0,len(row)):
            row[j] = np.round(row[j],3)
        data.append(row)

    make_table(rowlbls,collbls,data,flnm=flnm)

    if(writeToText):
        if(flnm == ''):
            flnm = 'tabledata'
        with open(flnm+'.txt', 'w') as f:
            for row in data:
                for val in row:
                    f.write(str(val)+' ')
                f.write('\n')
