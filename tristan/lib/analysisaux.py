import numpy as np
import math

def norm_constants(params,dt,inputs):
    """
    Normalizes time to inverse ion cyclotron and length to d_i

    Warning: assumes mi >> me

    Parameters
    ----------
    params : dict
        params dict from load_params in loadaux 
    dt : float
        time step in code units (normally integer fraction of elecrron plasma freq(for TRISTAN 1 this is params['c']/params['comp'])
    inputs : dict
        input dict from load_input in loadaux


    returns
    -------
    dt : float
        dt in units of inverse ion cyclotron freq
    c : float
        c in units of va (i.e. upstream alfven velocity)
    """

    stride = inputs['stride']
    dt = params['c']/params['comp'] #in units of wpe
    sigma_ion = params['sigma']*params['me']/params['mi'] #note: sigma includes an extra gamma0-1 factor where gamma0 is from input
    wpe_over_wce = 1./(np.sqrt(sigma_ion)*np.sqrt(params['mi']/params['me']))
    wce_over_wci = params['mi']/params['me']
    c = 1./sigma_ion #Assumes me << mi; returns c in unit of va #Assumes (gamma0-1) factor is neglible
    dt = stride*dt/(wpe_over_wce*wce_over_wci) #originally in wpe, now in units of wci

    return dt 

def compute_beta0(params,inputs):
    """
    Computes beta0, the upstream total plasma beta

    Parameters
    ----------
    params : dict
        params dict from load_params in loadaux 
    inputs : dict
        input dict from load_input in loadaux 

    Returns
    -------
    beta0 : float
        total plasma beta in far upstream region
    """

    uinj = inputs['gamma0'] #injection velocity in units of v/c

    gam0 = 1./np.sqrt(1.-(uinj)**2)
    beta0 = 4*gam0*params['delgam']/(params['sigma']*(gam0-1.)*(1.+params['me']/params['mi']))

    return beta0

def get_average_fields_over_yz(dfields, Efield_only = False):
    """
    Returns yz average of field i.e. dfield_avg(x,y,z) = <field(x,y,z)>_(y,z)

    Parameters
    ----------
    dfields : dict
        field data dictionary from load_fields in loadaux
    Efield_only : bool, opt
        if true, returns input bfield values

    Returns
    -------
    dfieldsavg : dict
        avg field data dictionary
    """

    from copy import deepcopy

    dfieldavg = deepcopy(dfields)

    dfieldavg['ex'][:] = dfieldavg['ex'].mean(axis=(0,1))
    dfieldavg['ey'][:] = dfieldavg['ey'].mean(axis=(0,1))
    dfieldavg['ez'][:] = dfieldavg['ez'].mean(axis=(0,1))

    if(not(Efield_only)):
        dfieldavg['bx'][:] = dfieldavg['bx'].mean(axis=(0,1))
        dfieldavg['by'][:] = dfieldavg['by'].mean(axis=(0,1))
        dfieldavg['bz'][:] = dfieldavg['bz'].mean(axis=(0,1))

    return dfieldavg

def compute_dflow(dfields, dpar_ion, dpar_elec, is2D=True, debug=False, return_empty=False, return_bins=True):
    """
    Compues velocity fluid moment for ions and electrons

    Bins and then takes average velocity in each bin. Grid will match dfields
    
    Parameters
    ----------
    dfields : dict
        field data dictionary from load_fields in loadaux
    dpar_ion : dict
        ion data from load_particles in loadaux
    dpar_elec : dict
        elec data from load_particles in loadaux
    is2D : bool (optional)
        specifies if data is 2d- needed as we spoof 2d data into 3d structure for backwards compatability of FPC routines
    debug : bool (optional)
        if true, prints statements to help with debugging
    return_empty : bool (optional)
        if true, returns empty array and skips computing flow data to save computational time- used for debugging other routines
    return_bins : bool (optional)
        if true, returns binned particles (binned by position), instead of dflow dictionary

    Returns
    -------
    dflow : dict
        ion and electron velocity moments
    """

    from lib.arrayaux import find_nearest

    if(debug): print("Entering compute dflow and initializing arrays...")

    if(return_empty):
        dflow = {}
        outkeys = 'ui vi wi ue ve we'.split()
        for _keyidx in range(0,len(outkeys)):
            dflow[outkeys[_keyidx]] = np.zeros(dfields['ex'].shape)
            dflow[outkeys[_keyidx]+'_xx'] = dfields['ex_xx'][:]
            dflow[outkeys[_keyidx]+'_yy'] = dfields['ex_yy'][:]
            dflow[outkeys[_keyidx]+'_zz'] = dfields['ex_zz'][:]
        return dflow

    #bin particles
    nx = len(dfields['ex_xx'])
    ny = len(dfields['ex_yy'])
    nz = len(dfields['ex_zz'])
    ion_bins = [[[ [] for _ in range(nx)] for _ in range(ny)] for _ in range(nz)]
    elec_bins = [[[ [] for _ in range(nx)] for _ in range(ny)] for _ in range(nz)]

    for _i in range(0,len(dpar_ion['xi'])):
        if(debug and _i % 100000 == 0): print("Binned: ", _i," ions of ", len(dpar_ion['xi']))
        xx = dpar_ion['xi'][_i]
        yy = dpar_ion['yi'][_i]
        zz = dpar_ion['zi'][_i]

        xidx = find_nearest(dfields['ex_xx'], xx)
        yidx = find_nearest(dfields['ex_yy'], yy)
        zidx = find_nearest(dfields['ex_zz'], zz)
        if(is2D):zidx = 0

        ion_bins[zidx][yidx][xidx].append({'ui':dpar_ion['ui'][_i] ,'vi':dpar_ion['vi'][_i] ,'wi':dpar_ion['wi'][_i]})

    for _i in range(0,len(dpar_elec['xe'])):
        if(debug and _i % 100000 == 0): print("Binned: ", _i," elecs of ", len(dpar_elec['xe']))
        xx = dpar_elec['xe'][_i]
        yy = dpar_elec['ye'][_i]
        zz = dpar_elec['ze'][_i]

        xidx = find_nearest(dfields['ex_xx'], xx)
        yidx = find_nearest(dfields['ex_yy'], yy)
        zidx = find_nearest(dfields['ex_zz'], zz)
        if(is2D):zidx = 0

        elec_bins[zidx][yidx][xidx].append({'ue':dpar_elec['ue'][_i] ,'ve':dpar_elec['ve'][_i] ,'we':dpar_elec['we'][_i]})

    #find average in each bin
    dflow = {}
    outkeys = 'ui vi wi ue ve we'.split()
    for _keyidx in range(0,len(outkeys)):
        dflow[outkeys[_keyidx]] = np.zeros(dfields['ex'].shape)
        dflow[outkeys[_keyidx]+'_xx'] = dfields['ex_xx'][:]
        dflow[outkeys[_keyidx]+'_yy'] = dfields['ex_yy'][:]
        dflow[outkeys[_keyidx]+'_zz'] = dfields['ex_zz'][:]
          
        if(debug): print("Computing moment for key: ", outkeys[_keyidx])

        for _i in range(0, nx):
            for _j in range(0, ny):
                for _k in range(0, nz):
                    if((len(ion_bins[_k][_j][_i]) > 0 and outkeys[_keyidx][-1]=='i') or (len(elec_bins[_k][_j][_i]) > 0 and outkeys[_keyidx][-1]=='e') ):
                        if(outkeys[_keyidx][-1] == 'i'):
                            #if(debug):print([ion_bins[_k][_j][_i][_idx][outkeys[_keyidx]] for _idx in range(0,len(ion_bins[_k][_j][_i]))])
                            dflow[outkeys[_keyidx]][_k,_j,_i] = np.mean([ion_bins[_k][_j][_i][_idx][outkeys[_keyidx]] for _idx in range(0,len(ion_bins[_k][_j][_i]))])
                        elif(outkeys[_keyidx][-1] == 'e'):
                            dflow[outkeys[_keyidx]][_k,_j,_i] = np.mean([elec_bins[_k][_j][_i][_idx][outkeys[_keyidx]] for _idx in range(0,len(elec_bins[_k][_j][_i]))])
                    else:
                        if(debug):print("Warning: no particles found in bin...")
                        dflow[outkeys[_keyidx]][_k,_j,_i] = 0.

        if(is2D):dflow[outkeys[_keyidx]][1,:,:]=dflow[outkeys[_keyidx]][0,:,:]
    return dflow

def remove_average_fields_over_yz(dfields, Efield_only = False):
    """
    Removes yz average from field data i.e. delta_field(x,y,z) = field(x,y,z)-<field(x,y,z)>_(y,z)

    Parameters
    ----------
    dfields : dict
        field data dictionary from flow_loader
    Efield_only : bool, opt
        if true, returns total bfield

    Returns
    -------
    dfieldsfluc : dict
        delta field data dictionary
    """
    from copy import deepcopy

    dfieldfluc = deepcopy(dfields) #deep copy
    dfieldfluc['ex'] = dfieldfluc['ex']-dfieldfluc['ex'].mean(axis=(0,1))
    dfieldfluc['ey'] = dfieldfluc['ey']-dfieldfluc['ey'].mean(axis=(0,1))
    dfieldfluc['ez'] = dfieldfluc['ez']-dfieldfluc['ez'].mean(axis=(0,1))

    if(not(Efield_only)):
        dfieldfluc['bx'] = dfieldfluc['bx']-dfieldfluc['bx'].mean(axis=(0,1))
        dfieldfluc['by'] = dfieldfluc['by']-dfieldfluc['by'].mean(axis=(0,1))
        dfieldfluc['bz'] = dfieldfluc['bz']-dfieldfluc['bz'].mean(axis=(0,1))

    return dfieldfluc

def remove_average_cur_over_yz(dflow):
    """
    Computes u_tilde,s = u_s - <u_s>_yz

    Parameters
    ----------
    dflow : dict
        field data dictionary from compute_dflow

    Returns
    -------
    dflowfluc : dict
        field fluc data dictionary
    """
    from copy import deepcopy

    dflowfluc = deepcopy(dflow) #deep copy
    dflowfluc['ui'] = dflowfluc['ui']-dflowfluc['ui'].mean(axis=(0,1))
    dflowfluc['vi'] = dflowfluc['vi']-dflowfluc['vi'].mean(axis=(0,1))
    dflowfluc['wi'] = dflowfluc['wi']-dflowfluc['wi'].mean(axis=(0,1))

    dflowfluc['ue'] = dflowfluc['ue']-dflowfluc['ue'].mean(axis=(0,1))
    dflowfluc['ve'] = dflowfluc['ve']-dflowfluc['ve'].mean(axis=(0,1))
    dflowfluc['we'] = dflowfluc['we']-dflowfluc['we'].mean(axis=(0,1))

    return dflowfluc

def get_average_fields_over_yz(dfields):
    """
    Returns yz average of quantity i.e. dflow_avg(x,y,z) = <field(x,y,z)>_(y,z)

    Computes u_bar,s = <u_s>_yz

    Parameters
    ----------
    dflow : dict
        field data dictionary from compute_dflow

    Returns
    -------
    dfieldsavg : dict
        avg field data dictionary
    """

    from copy import deepcopy

    dfieldavg = deepcopy(dfields)

    dfieldavg['ex'][:] = dfieldavg['ex'].mean(axis=(0,1))
    dfieldavg['ey'][:] = dfieldavg['ey'].mean(axis=(0,1))
    dfieldavg['ez'][:] = dfieldavg['ez'].mean(axis=(0,1))
    dfieldavg['bx'][:] = dfieldavg['bx'].mean(axis=(0,1))
    dfieldavg['by'][:] = dfieldavg['by'].mean(axis=(0,1))
    dfieldavg['bz'][:] = dfieldavg['bz'].mean(axis=(0,1))

    return dfieldavg

def get_average_flow_over_yz(dflow):
    """
    Gets yz average from flow data i.e. flow_avg(x,y,z) = <flow(x,y,z)>_(y,z)

    Parameters
    ----------
    dflow : dict
        flow data dictionary from flow_loader

    Returns
    -------
    dflow : dict
        delta flow data dictionary
    """
    from copy import deepcopy
    dflowavg = deepcopy(dflow)

    for key in dflowavg.keys():
        if(not('_xx' in key) and not('_yy' in key) and not('_zz' in key)):
            dflowavg[key][:] = dflowavg[key].mean(axis=(0,1))

    return dflowavg

def get_average_den_over_yz(dden):
    """
    Gets yz average from den data i.e. den_avg(x,y,z) = <den(x,y,z)>_(y,z)

    Parameters
    ----------
    dden : dict
        den data dictionary

    Returns
    -------
    dden : dict
        avg den data dictionary
    """
    from copy import deepcopy
    ddenavg = deepcopy(dden)

    for key in ddenavg.keys():
        if(not('Vframe' in key) and not('_xx' in key) and not('_yy' in key) and not('_zz' in key)):
            ddenavg[key][:] = ddenavg[key].mean(axis=(0,1))

    return ddenavg

def get_B_avg(dfields,xlim,ylim,zlim):
    """
    Gets average B in box

    Parameters
    ----------
    dfields : dict
        dict from field_loader
    *lim : [float,float]
        upper and lower bounds of box

    Returns
    -------
    [B0x, B0y, B0z] : [float,float,float]
        B0 at specified xx
    """

    from lib.arrayaux import get_average_in_box

    x1 = xlim[0]
    x2 = xlim[1]
    y1 = ylim[0]
    y2 = ylim[1]
    z1 = zlim[0]
    z2 = zlim[1]

    B0x = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'bx')
    B0y = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'by')
    B0z = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'bz')

    return [B0x, B0y, B0z]

def compute_field_aligned_coord(dfields,xlim,ylim,zlim):
    """
    Computes field aligned coordinate basis using average B0 in provided box

    vpar in parallel to B0
    vperp2 is in direction of [xhat] cross vpar
    vperp is in direction of vpar cross vperp2

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader in loadaux
    xlim : array
        xx bounds of analysis (i.e. where the sweep starts and stops)
    ylim : array
        yy bounds of each integration box
    zlim : array
        zz bounds of each integration box

    Returns
    -------
    vparbasis/vperp1basis/vperp2basis : [float,float,float]
        field aligned basis (ordered [vx,vy,vz])
    """
    from lib.arrayaux import find_nearest
    from copy import deepcopy

    if(np.abs(xlim[1]-xlim[0]) > 4.):
        print("Warning, when computing field aligned coordinates, we found that xlim[1]-xlim[0] is large. Consider reducing size...")

    xavg = (xlim[1]+xlim[0])/2.
    xxidx = find_nearest(dfields['bz_xx'],xavg)
    B0 = get_B_avg(dfields,xlim,ylim,zlim) #***Assumes xlim is sufficiently thin*** as get_B0 uses <B(x0,y,z)>_(yz)=B0

    #get normalized basis vectors
    eparbasis = deepcopy(B0)
    eparbasis /= np.linalg.norm(eparbasis)
    eperp2basis = np.cross([1.,0,0],B0) #x hat cross B0
    tol = 0.005
    _B0 = B0 / np.linalg.norm(B0)
    if(np.abs(np.linalg.norm(np.cross([_B0[0],_B0[1],_B0[2]],[1.,0.,0.]))) < tol):
        print("Warning, it seems B0 is parallel to xhat (typically the shock normal)...")
        print("(Bx,By,Bz): ", _B0[0],_B0[1],_B0[2])
        print("xhat: 1,0,0")
        return np.asarray([1.,0,0]),np.asarray([0,1.,0]),np.asarray([0,0,1.])
    eperp2basis /= np.linalg.norm(eperp2basis)
    eperp1basis = np.cross(eparbasis,eperp2basis)
    eperp1basis /= np.linalg.norm(eperp1basis)

    return eparbasis, eperp1basis, eperp2basis

def convert_flow_to_local_par(dfields,dflow):
    """
    Converts flow (fluid moment!) in field aligned coordinates using a local definition of parallel. That is it finds local value of B and then takes the local projection of the current with it

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader in loadaux
    dflow : dict
        field data dictionary from compute_dflow

    Returns
    -------
    dflow : dict
        field data dictionary with FAC data
    """

    import copy

    dflow = copy.deepcopy(dflow)

    jxi = dflow['ui'][:,:,:]
    jyi = dflow['vi'][:,:,:]
    jzi = dflow['wi'][:,:,:]
    jxe = dflow['ue'][:,:,:]
    jye = dflow['ve'][:,:,:]
    jze = dflow['we'][:,:,:]
    bx = dfields['bx'][:,:,:]
    by = dfields['by'][:,:,:]
    bz = dfields['bz'][:,:,:]

    dflow['upari'] = (jxi*bx+jyi*by+jzi*bz)/np.sqrt(bx*bx+by*by+bz*bz)
    dflow['upari_xx'] = dfields['ex_xx'][:]
    dflow['upari_yy'] = dfields['ex_yy'][:]
    dflow['upari_zz'] = dfields['ex_zz'][:]

    dflow['uperp2i'] = (-bz*jyi+by*jzi)/np.sqrt(bx*bx+by*by+bz*bz) #dot(E0,cross([1,0,0],B0))/sqrt(\mathbf{B0})
    dflow['uperp2i_xx'] = dfields['ex_xx'][:]
    dflow['uperp2i_yy'] = dfields['ex_yy'][:]
    dflow['uperp2i_zz'] = dfields['ex_zz'][:]

    dflow['uperp1i'] = ((-by*by-bz*bz)*jxi+bx*by*jyi+bx*bz*jzi)/np.sqrt(bx*bx+by*by+bz*bz) #Dot[e0, Cross[Cross[x, b0], b0]]
    dflow['uperp1i_xx'] = dfields['ex_xx'][:]
    dflow['uperp1i_yy'] = dfields['ex_yy'][:]
    dflow['uperp1i_zz'] = dfields['ex_zz'][:]
    
    dflow['upare'] = (jxe*bx+jye*by+jze*bz)/np.sqrt(bx*bx+by*by+bz*bz)
    dflow['upare_xx'] = dfields['ex_xx'][:]
    dflow['upare_yy'] = dfields['ex_yy'][:]
    dflow['upare_zz'] = dfields['ex_zz'][:]

    dflow['uperp2e'] = (-bz*jye+by*jze)/np.sqrt(bx*bx+by*by+bz*bz) #dot(E0,cross([1,0,0],B0))/sqrt(\mathbf{B0})
    dflow['uperp2e_xx'] = dfields['ex_xx'][:]
    dflow['uperp2e_yy'] = dfields['ex_yy'][:]
    dflow['uperp2e_zz'] = dfields['ex_zz'][:]

    dflow['uperp1e'] = ((-by*by-bz*bz)*jxe+bx*by*jye+bx*bz*jze)/np.sqrt(bx*bx+by*by+bz*bz) #Dot[e0, Cross[Cross[x, b0], b0]]
    dflow['uperp1e_xx'] = dfields['ex_xx'][:]
    dflow['uperp1e_yy'] = dfields['ex_yy'][:]
    dflow['uperp1e_zz'] = dfields['ex_zz'][:]

    return dflow

def convert_flowfluc_to_local_par(dfields,dflow,dflowfluc):
    """
    Converts fluc flow (fluid moment!) in field aligned coordinates using a local definition of parallel. That is it finds local value of B and then takes the local projection of the current with it

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader in loadaux
    dflow : dict
        field data dictionary from compute_dflow
    dflowfluc : dict
        fluc field data dictionary

    Returns
    -------
    dflow : dict
        fluc field data dictionary with FAC data
    """
    return convert_flow_to_local_par(dfields,dflowfluc)

def convert_flow_to_par(dfields,dflow):
    """
    Converts flow (fluid moment!) in field aligned coordinates using a plane averaged definition of parallel. That is it finds yz average value of B and then takes the local projection of the current with it at every location

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader in loadaux
    dflow : dict
        field data dictionary from compute_dflow

    Returns
    -------
    dflow : dict
        field data dictionary with FAC data
    """

    import copy

    dflow = copy.deepcopy(dflow)

    jxi = dflow['ui'][:,:,:]
    jyi = dflow['vi'][:,:,:]
    jzi = dflow['wi'][:,:,:]
    jxe = dflow['ue'][:,:,:]
    jye = dflow['ve'][:,:,:]
    jze = dflow['we'][:,:,:]
    bx = dfields['bx'][:,:,:]
    by = dfields['by'][:,:,:]
    bz = dfields['bz'][:,:,:]

    #take average along transverse direction
    average_values1 = np.mean(bx, axis=1)
    bx = np.tile(average_values1[:, np.newaxis, :], (1, bx.shape[1], 1))
    average_values2 = np.mean(by, axis=1)
    by = np.tile(average_values2[:, np.newaxis, :], (1, by.shape[1], 1))
    average_values3 = np.mean(bz, axis=1)
    bz = np.tile(average_values3[:, np.newaxis, :], (1, bz.shape[1], 1))

    dflow['upari'] = (jxi*bx+jyi*by+jzi*bz)/np.sqrt(bx*bx+by*by+bz*bz)
    dflow['upari_xx'] = dfields['ex_xx'][:]
    dflow['upari_yy'] = dfields['ex_yy'][:]
    dflow['upari_zz'] = dfields['ex_zz'][:]

    dflow['uperp2i'] = (-bz*jyi+by*jzi)/np.sqrt(bx*bx+by*by+bz*bz) #dot(E0,cross([1,0,0],B0))/sqrt(\mathbf{B0})
    dflow['uperp2i_xx'] = dfields['ex_xx'][:]
    dflow['uperp2i_yy'] = dfields['ex_yy'][:]
    dflow['uperp2i_zz'] = dfields['ex_zz'][:]

    dflow['uperp1i'] = ((-by*by-bz*bz)*jxi+bx*by*jyi+bx*bz*jzi)/np.sqrt(bx*bx+by*by+bz*bz) #Dot[e0, Cross[Cross[x, b0], b0]]
    dflow['uperp1i_xx'] = dfields['ex_xx'][:]
    dflow['uperp1i_yy'] = dfields['ex_yy'][:]
    dflow['uperp1i_zz'] = dfields['ex_zz'][:]

    dflow['upare'] = (jxe*bx+jye*by+jze*bz)/np.sqrt(bx*bx+by*by+bz*bz)
    dflow['upare_xx'] = dfields['ex_xx'][:]
    dflow['upare_yy'] = dfields['ex_yy'][:]
    dflow['upare_zz'] = dfields['ex_zz'][:]

    dflow['uperp2e'] = (-bz*jye+by*jze)/np.sqrt(bx*bx+by*by+bz*bz) #dot(E0,cross([1,0,0],B0))/sqrt(\mathbf{B0})
    dflow['uperp2e_xx'] = dfields['ex_xx'][:]
    dflow['uperp2e_yy'] = dfields['ex_yy'][:]
    dflow['uperp2e_zz'] = dfields['ex_zz'][:]

    dflow['uperp1e'] = ((-by*by-bz*bz)*jxe+bx*by*jye+bx*bz*jze)/np.sqrt(bx*bx+by*by+bz*bz) #Dot[e0, Cross[Cross[x, b0], b0]]
    dflow['uperp1e_xx'] = dfields['ex_xx'][:]
    dflow['uperp1e_yy'] = dfields['ex_yy'][:]
    dflow['uperp1e_zz'] = dfields['ex_zz'][:]

    return dflow

def convert_flowfluc_to_par(dfields,dflow,dflowfluc):
    """
    Converts fluc flow (fluid moment!) in field aligned coordinates using a plane averaged definition of parallel. That is it finds yz average value of B and then takes the local projection of the current with it at every location

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader in loadaux
    dflow : dict
        field data dictionary from compute_dflow
    dflowfluc : dict
        fluc field data dictionary

    Returns
    -------
    dflow : dict
        fluc field data dictionary with FAC data
    """
    return convert_flow_to_par(dfields,dflowfluc)

def convert_to_par(dfields,detrendfields=None):
    """
    Converts field data in field aligned coordinates using a plane averaged definition of parallel. That is it finds yz average value of B and then takes the local projection of the current with it at every location

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader in loadaux
    detrendfields : dict
        if provided, fac will be computed using this dictionary instead

    Returns
    -------
    dfields : dict
        field data dictionary with FAC data
    """

    import copy

    dfields = copy.deepcopy(dfields)

    ex = dfields['ex'][:,:,:]
    ey = dfields['ey'][:,:,:]
    ez = dfields['ez'][:,:,:]
    bx = dfields['bx'][:,:,:]
    by = dfields['by'][:,:,:]
    bz = dfields['bz'][:,:,:]

    #take average along transverse direction
    average_values1 = np.mean(bx, axis=1)
    bx = np.tile(average_values1[:, np.newaxis, :], (1, bx.shape[1], 1))
    average_values2 = np.mean(by, axis=1)
    by = np.tile(average_values2[:, np.newaxis, :], (1, by.shape[1], 1))
    average_values3 = np.mean(bz, axis=1)
    bz = np.tile(average_values3[:, np.newaxis, :], (1, bz.shape[1], 1))

    dfields['epar'] = np.zeros(dfields['ex'].shape)
    dfields['epar'] = (ex*bx+ey*by+ez*bz)/np.sqrt(bx*bx+by*by+bz*bz)
    dfields['epar_xx'] = dfields['ex_xx'][:]
    dfields['epar_yy'] = dfields['ex_yy'][:]
    dfields['epar_zz'] = dfields['ex_zz'][:]

    dfields['eperp2'] = (-bz*ey+by*ez)/np.sqrt(bx*bx+by*by+bz*bz) #dot(E0,cross([1,0,0],B0))/sqrt(\mathbf{B0})
    dfields['eperp2_xx'] = dfields['ex_xx'][:]
    dfields['eperp2_yy'] = dfields['ex_yy'][:]
    dfields['eperp2_zz'] = dfields['ex_zz'][:]

    dfields['eperp1'] = ((-by*by-bz*bz)*ex+bx*by*ey+bx*bz*ez)/np.sqrt(bx*bx+by*by+bz*bz) #Dot[e0, Cross[Cross[x, b0], b0]]
    dfields['eperp1_xx'] = dfields['ex_xx'][:]
    dfields['eperp1_yy'] = dfields['ex_yy'][:]
    dfields['eperp1_zz'] = dfields['ex_zz'][:]

    return dfields

def convert_fluc_to_par(dfields,dfluc):
    """
    Converts fluc field data in field aligned coordinates using a plane averaged definition of parallel. That is it finds yz average value of B and then takes the local projection of the current with it at every location

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader in loadaux
    dfluc : dict
        fluc fields dict

    Returns
    -------
    dfields : dict
        fluc field data dictionary with FAC data
    """
    import copy

    dfluc = copy.deepcopy(dfluc)

    ex = dfluc['ex'][:,:,:]
    ey = dfluc['ey'][:,:,:]
    ez = dfluc['ez'][:,:,:]
    bx = dfields['bx'][:,:,:]
    by = dfields['by'][:,:,:]
    bz = dfields['bz'][:,:,:]

    #take average along transverse direction
    average_values1 = np.mean(bx, axis=1)
    bx = np.tile(average_values1[:, np.newaxis, :], (1, bx.shape[1], 1))
    average_values2 = np.mean(by, axis=1)
    by = np.tile(average_values2[:, np.newaxis, :], (1, by.shape[1], 1))
    average_values3 = np.mean(bz, axis=1)
    bz = np.tile(average_values3[:, np.newaxis, :], (1, bz.shape[1], 1))

    dfluc['epar'] = np.zeros(dfields['ex'].shape)
    dfluc['epar'] = (ex*bx+ey*by+ez*bz)/np.sqrt(bx*bx+by*by+bz*bz)
    dfluc['epar_xx'] = dfields['ex_xx'][:]
    dfluc['epar_yy'] = dfields['ex_yy'][:]
    dfluc['epar_zz'] = dfields['ex_zz'][:]

    dfluc['eperp2'] = (-bz*ey+by*ez)/np.sqrt(bx*bx+by*by+bz*bz) #dot(E0,cross([1,0,0],B0))/sqrt(\mathbf{B0})
    dfluc['eperp2_xx'] = dfields['ex_xx'][:]
    dfluc['eperp2_yy'] = dfields['ex_yy'][:]
    dfluc['eperp2_zz'] = dfields['ex_zz'][:]

    dfluc['eperp1'] = ((-by*by-bz*bz)*ex+bx*by*ey+bx*bz*ez)/np.sqrt(bx*bx+by*by+bz*bz) #Dot[e0, Cross[Cross[x, b0], b0]]
    dfluc['eperp1_xx'] = dfields['ex_xx'][:]
    dfluc['eperp1_yy'] = dfields['ex_yy'][:]
    dfluc['eperp1_zz'] = dfields['ex_zz'][:]

    return dfluc

def convert_to_local_par(dfields,detrendfields=None):
    """
    Converts fluc field data in field aligned coordinates using a local definition of parallel. That is it finds local of B and then takes the local projection of the current with it at every location

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader in loadaux
    detrendfields : dict
        if provided, fac will be computed using this dictionary instead

    Returns
    -------
    dfields : dict
        fluc field data dictionary with FAC data
    """
    
    import copy

    dfields = copy.deepcopy(dfields)

    ex = dfields['ex'][:,:,:]
    ey = dfields['ey'][:,:,:]
    ez = dfields['ez'][:,:,:]
    bx = dfields['bx'][:,:,:]
    by = dfields['by'][:,:,:]
    bz = dfields['bz'][:,:,:]
    
    dfields['epar'] = np.zeros(dfields['ex'].shape)
    dfields['epar'] = (ex*bx+ey*by+ez*bz)/np.sqrt(bx*bx+by*by+bz*bz)
    dfields['epar_xx'] = dfields['ex_xx'][:]
    dfields['epar_yy'] = dfields['ex_yy'][:]
    dfields['epar_zz'] = dfields['ex_zz'][:]

    dfields['eperp2'] = (-bz*ey+by*ez)/np.sqrt(bx*bx+by*by+bz*bz) #dot(E0,cross([1,0,0],B0))/sqrt(\mathbf{B0})
    dfields['eperp2_xx'] = dfields['ex_xx'][:]
    dfields['eperp2_yy'] = dfields['ex_yy'][:]
    dfields['eperp2_zz'] = dfields['ex_zz'][:]

    dfields['eperp1'] = ((-by*by-bz*bz)*ex+bx*by*ey+bx*bz*ez)/np.sqrt(bx*bx+by*by+bz*bz) #Dot[e0, Cross[Cross[x, b0], b0]]
    dfields['eperp1_xx'] = dfields['ex_xx'][:]
    dfields['eperp1_yy'] = dfields['ex_yy'][:]
    dfields['eperp1_zz'] = dfields['ex_zz'][:]

    if(detrendfields != None):
        bx_0 = dfields['bx'] #bx_0riginal
        by_0 = dfields['by']
        bz_0 = dfields['bz']

        bx = detrendfields['bx'][:,:,:]
        by = detrendfields['by'][:,:,:]
        bz = detrendfields['bz'][:,:,:]
        
        dfields['epar_detrend'] = (ex*bx+ey*by+ez*bz)/np.sqrt(bx*bx+by*by+bz*bz)
        dfields['epar_detrend_xx'] = dfields['ex_xx'][:]
        dfields['epar_detrend_yy'] = dfields['ex_yy'][:]
        dfields['epar_detrend_zz'] = dfields['ex_zz'][:]

        dfields['eperp2_detrend'] = (-bz*ey+by*ez)/np.sqrt(bx*bx+by*by+bz*bz) #dot(E0,cross([1,0,0],B0))/sqrt(\mathbf{B0})
        dfields['eperp2_detrend_xx'] = dfields['ex_xx'][:]
        dfields['eperp2_detrend_yy'] = dfields['ex_yy'][:]
        dfields['eperp2_detrend_zz'] = dfields['ex_zz'][:]
        
        dfields['eperp1_detrend'] = ((-by*by-bz*bz)*ex+bx*by*ey+bx*bz*ez)/np.sqrt(bx*bx+by*by+bz*bz) #Dot[e0, Cross[Cross[x, b0], b0]]
        dfields['eperp1_detrend_xx'] = dfields['ex_xx'][:]
        dfields['eperp1_detrend_yy'] = dfields['ex_yy'][:]
        dfields['eperp1_detrend_zz'] = dfields['ex_zz'][:]

        dfields['bpar_detrend'] = (bx_0*bx+by_0*by+bz_0*bz)/np.sqrt(bx*bx+by*by+bz*bz)
        dfields['bpar_detrend_xx'] = dfields['bx_xx'][:]
        dfields['bpar_detrend_yy'] = dfields['bx_yy'][:]
        dfields['bpar_detrend_zz'] = dfields['bx_zz'][:]

        dfields['bperp2_detrend'] = (-bz*by_0+by*bz_0)/np.sqrt(bx*bx+by*by+bz*bz) #dot(E0,cross([1,0,0],B0))/sqrt(\mathbf{B0})
        dfields['bperp2_detrend_xx'] = dfields['bx_xx'][:]
        dfields['bperp2_detrend_yy'] = dfields['bx_yy'][:]
        dfields['bperp2_detrend_zz'] = dfields['bx_zz'][:]
        
        dfields['bperp1_detrend'] = ((-by*by-bz*bz)*bx_0+bx*by*by_0+bx*bz*bz_0)/np.sqrt(bx*bx+by*by+bz*bz) #Dot[e0, Cross[Cross[x, b0], b0]]
        dfields['bperp1_detrend_xx'] = dfields['bx_xx'][:]
        dfields['bperp1_detrend_yy'] = dfields['bx_yy'][:]
        dfields['bperp1_detrend_zz'] = dfields['bx_zz'][:]

    return dfields

def convert_fluc_to_local_par(dfields,dfluc):
    """
    Converts fluc field data in field aligned coordinates using a local definition of parallel. That is it finds local value of B and then takes the local projection of the current with it at every location

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader in loadaux
    dfluc : dict
        fluc fields dict

    Returns
    -------
    dfields : dict
        fluc field data dictionary with FAC data
    """
    import copy

    dfluc = copy.deepcopy(dfluc)

    ex = dfluc['ex'][:,:,:]
    ey = dfluc['ey'][:,:,:]
    ez = dfluc['ez'][:,:,:]
    bx = dfields['bx'][:,:,:]
    by = dfields['by'][:,:,:]
    bz = dfields['bz'][:,:,:]

    dfluc['epar'] = np.zeros(dfields['ex'].shape)
    dfluc['epar'] = (ex*bx+ey*by+ez*bz)/np.sqrt(bx*bx+by*by+bz*bz)
    dfluc['epar_xx'] = dfields['ex_xx'][:]
    dfluc['epar_yy'] = dfields['ex_yy'][:]
    dfluc['epar_zz'] = dfields['ex_zz'][:]

    dfluc['eperp2'] = (-bz*ey+by*ez)/np.sqrt(bx*bx+by*by+bz*bz) #dot(E0,cross([1,0,0],B0))/sqrt(\mathbf{B0})
    dfluc['eperp2_xx'] = dfields['ex_xx'][:]
    dfluc['eperp2_yy'] = dfields['ex_yy'][:]
    dfluc['eperp2_zz'] = dfields['ex_zz'][:]

    dfluc['eperp1'] = ((-by*by-bz*bz)*ex+bx*by*ey+bx*bz*ez)/np.sqrt(bx*bx+by*by+bz*bz) #Dot[e0, Cross[Cross[x, b0], b0]]
    dfluc['eperp1_xx'] = dfields['ex_xx'][:]
    dfluc['eperp1_yy'] = dfields['ex_yy'][:]
    dfluc['eperp1_zz'] = dfields['ex_zz'][:]

    return dfluc

def change_velocity_basis(dfields,dpar,xlim,ylim,zlim,debug=False):
    """
    Converts to field aligned coordinate system
    Parallel direction is along average magnetic field direction at average in limits

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader
    dpar : dict
        dict returned by read_particles
    xlim : array
        xx bounds of analysis (i.e. where the sweep starts and stops)
    ylim : array
        yy bounds of each integration box
    zlim : array
        zz bounds of each integration box
    debug : bool, opt
        print debug statements if energy is not conserved

    Returns
    -------
    dparnewbasis : dict
        particle dictionary in new basis
    """
    from copy import deepcopy

    if(dfields['Vframe_relative_to_sim'] != dpar['Vframe_relative_to_sim']):
        print("Warning, field data is not in the same frame as particle data...")

    vparbasis, vperp1basis, vperp2basis = compute_field_aligned_coord(dfields,xlim,ylim,zlim)
    #check orthogonality of these vectors
    if(debug):
        tol = 0.01
        if(np.abs(np.dot(vparbasis,vperp1basis)) > tol or np.abs(np.dot(vparbasis,vperp2basis)) > tol or np.abs(np.dot(vperp1basis,vperp2basis) > tol)):
            print("Warning: orthogonality was not kept...")

    #make change of basis matrix
    _ = np.asarray([vparbasis,vperp1basis,vperp2basis]).T
    changebasismatrix = np.linalg.inv(_)

    #change basis
    dparnewbasis = {}
    dparnewbasis['ppar'],dparnewbasis['pperp1'],dparnewbasis['pperp2'] = np.matmul(changebasismatrix,[dpar['p1'][:],dpar['p2'][:],dpar['p3'][:]])
    dparnewbasis['x1'] = deepcopy(dpar['x1'][:])
    dparnewbasis['x2'] = deepcopy(dpar['x2'][:])
    dparnewbasis['x3'] = deepcopy(dpar['x3'][:])
    dparnewbasis['q'] = dpar['q']

    #check v^2 for both basis to make sure everything matches
    if(debug):
        for i in range(0,20):
            normnewbasis = np.linalg.norm([dparnewbasis['ppar'][i],dparnewbasis['pperp1'][i],dparnewbasis['pperp2'][i]])
            normoldbasis = np.linalg.norm([dpar['p1'][i],dpar['p2'][i],dpar['p3'][i]])
            if(np.abs(normnewbasis-normoldbasis) > 0.01):
                print('Warning. Change of basis did not converse total energy...')

    return dparnewbasis

def change_velocity_basis_local(dfields,dpar,loadfrac=1,debug=False):
    """
    Converts to field aligned coordinate system

    **differs from change_velocity_basis in that the local FAC at the location of each particle is used**

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader
    dpar : dict
        dict returned by read_particles
    loadfrac : int
        loads every *loadfrac*th particle for debugging (=1 loads all)
    debug : bool, opt
        print debug statements if energy is not conserved

    Returns
    -------
    dparnewbasis : dict
        particle dictionary in new basis
    """
    from copy import deepcopy

    if(dfields['Vframe_relative_to_sim'] != dpar['Vframe_relative_to_sim']):
        print("Warning, field data is not in the same frame as particle data...")

    dparnewbasis = {}
    dparnewbasis['x1'] = deepcopy(dpar['x1'][::loadfrac])
    dparnewbasis['x2'] = deepcopy(dpar['x2'][::loadfrac])
    dparnewbasis['x3'] = deepcopy(dpar['x3'][::loadfrac])
    dparnewbasis['ppar'] = np.zeros((len(dpar['x1'][::loadfrac])))
    dparnewbasis['pperp1'] = np.zeros((len(dpar['x1'][::loadfrac])))
    dparnewbasis['pperp2'] = np.zeros((len(dpar['x1'][::loadfrac])))
    dparnewbasis['q'] = dpar['q']

    for _ky in dpar.keys():
        try:
            dparnewbasis[_ky] = deepcopy(dpar[_ky][::loadfrac])
        except:
            pass

    changebasismatrixes = []

    for _idx in range(0,len(dparnewbasis['x1'])):
        from lib.fpcaux import weighted_field_average

        bx = weighted_field_average(dpar['x1'][_idx], dpar['x2'][_idx], dpar['x3'][_idx], dfields, 'bx')
        by = weighted_field_average(dpar['x1'][_idx], dpar['x2'][_idx], dpar['x3'][_idx], dfields, 'by')
        bz = weighted_field_average(dpar['x1'][_idx], dpar['x2'][_idx], dpar['x3'][_idx], dfields, 'bz')

        #FROM COMPUTE FIELD ALIGNED
        B0 = [bx,by,bz]

        #get normalized basis vectors
        vparbasis = deepcopy(B0)
        vparbasis /= np.linalg.norm(vparbasis)
        vperp2basis = np.cross([1.,0,0],B0) #x hat cross B0
        tol = 0.005
        _B0 = B0 / np.linalg.norm(B0)
        
        vperp2basis /= np.linalg.norm(vperp2basis)
        vperp1basis = np.cross(vparbasis,vperp2basis)
        vperp1basis /= np.linalg.norm(vperp1basis)

        _ = np.asarray([vparbasis,vperp1basis,vperp2basis]).T
        changebasismatrix = np.linalg.inv(_)

        _ppar,_pperp1,_pperp2 = np.matmul(changebasismatrix,[dpar['p1'][_idx],dpar['p2'][_idx],dpar['p3'][_idx]])

        dparnewbasis['ppar'][_idx] = _ppar
        dparnewbasis['pperp1'][_idx] = _pperp1
        dparnewbasis['pperp2'][_idx] =_pperp2

        changebasismatrixes.append(changebasismatrix)

    #check v^2 for both basis to make sure everything matches
    if(debug):
        for i in range(0,20):
            normnewbasis = np.linalg.norm([dparnewbasis['ppar'][i],dparnewbasis['pperp1'][i],dparnewbasis['pperp2'][i]])
            normoldbasis = np.linalg.norm([dpar['p1'][i],dpar['p2'][i],dpar['p3'][i]])
            if(np.abs(normnewbasis-normoldbasis) > 0.001):
                print('Warning. Change of basis did not converse total energy...')
                print(np.abs(normnewbasis-normoldbasis))

    return dparnewbasis, changebasismatrixes

def wlt(t,data,w=6,klim=None,retstep=1,powerTwoSpace=False):
    """
    Peforms wavelet transform using morlet wavelet on data that is a function of t i.e. data(t)

    Paramters
    ---------
    t : 1d array
        independent data array
    data : 1d array
        dependent data array
    w : float, opt
        omega term in morlet wavelet function (relates to the number of humps)
    retstep : int, opt
        spacing between samples of k in returned by wavelet transform
        used mostly to save memory as wavelet transform returns dense sampling of k
    powerTwoSpace : bool, optimize
        if true, will space widths using powers of two (not well tested, avoid use)

    Returns
    -------
    k : array, float
        wavenumbers associated with the output
    cwtm : 2d array, float
        wavelet transform data
    """
    from scipy import signal
    from lib.arrayaux import find_nearest

    dt = t[1]-t[0]

    if(powerTwoSpace): #from Torrence et al 1997 (practical guide to wavelet analysis) (suggested to use different spacing)
        s0 = 1.*dt
        J = len(data)
        delta_j = np.log2(len(data)*dt/s0)/(J) #guess for now
        print('delta_j, ', delta_j)
        widths = []
        for _j in range(J-1,-1,-1):
            widths.append(s0*2.**(_j*delta_j))
        widths = np.asarray(widths)
        freq = w/(2*widths*np.pi*s0)
    else: 
        fs = 1./dt
        freq = np.linspace(dt/10,fs/4.,int(len(data)/retstep))
        widths = w*fs / (2*freq*np.pi)
    
    cwtm = signal.cwt(data, signal.morlet2, widths, w=w)

    k = 2.0*math.pi*freq
    if(klim != None):
        lowerkidx = find_nearest(k,klim[0])
        upperkidx = find_nearest(k,klim[1])
        k = k[loweridx:upperkidx+1]
        cwtm = cwtm[loweridx:upperkidx+1,:]

    #normalize
    for _idx in range(0,len(cwtm[:,0])):
        cwtm[_idx,:] *= (np.abs(k[_idx]))**0.5


    return k, cwtm

def take_fft2(data,daxisx0,daxis1):
    """
    Computes 2d fft on given data

    Parameters
    ----------
    data : 2d array
        2d data to be transformed
    daxisx0 : float
        cartesian spatial spacing between points along 0th axis of data
    daxisx1 : float
        cartesian spatial spacing between points along 1st axis of data

    Returns
    -------
    k0 : 1d array
        wavenumber coordinates corresponding to 0th axis
    k1 : 1d array
        wavenumber coordinates corresponding to 1st axis
    """

    k0 = 2.*np.pi*np.fft.fftfreq(len(data),daxisx0)
    k1 = 2.*np.pi*np.fft.fftfreq(len(data[1]),daxis1)

    fftdata = np.fft.fft2(data)/(float(len(data)*len(data[1])))

    return k0, k1, fftdata

def _ffttransform_in_yz(dfields,fieldkey):
    """
    Takes f(z,y,x) and computes f(x,kz,ky) using a 2d fft for some given field

    Parameters
    ----------
    dfields : dict
        dict from field_loader
    fieldkey : str
        name of field you want to transform (ex, ey, ez, bx, by, bz, ux, uy, uz)

    Returns
    -------
    ky/kz : 1d array
        coordinates in wavenumber space
    fieldfftsweepoverx : 3d array
        f(x,kz,ky) for specified field f
    """

    fieldfftsweepoverx = []
    for xxindex in range(0,len(dfields[fieldkey][0][0])):
        fieldslice = np.asarray(dfields[fieldkey])[:,:,xxindex]
        daxis0 = dfields[fieldkey+'_zz'][1]-dfields[fieldkey+'_zz'][0]
        daxis1 = dfields[fieldkey+'_yy'][1]-dfields[fieldkey+'_zz'][0]
        kz, ky, fieldslicefft = take_fft2(fieldslice,daxis0,daxis1)
        fieldfftsweepoverx.append(fieldslicefft)
    fieldfftsweepoverx = np.asarray(fieldfftsweepoverx)

    return kz, ky, fieldfftsweepoverx

def transform_field_to_kzkykxxx(ddict,fieldkey,retstep=12):
    """
    Takes fft in y and z and wavelet transform in x of given field/ flow.

    E.g. takes B(z,y,x) and computes B(kz,ky,kx;x)

    Parameters
    ----------
    ddict : dict
        field or flow data dictionary
    fieldkey : str
        name of field you want to transform (ex, ey, ez, bx, by, bz, ux, uy, uz)
    retstep : int, opt
        spacing between samples of k in returned by wavelet transform
        used mostly to save memory as wavelet transform returns dense sampling of k

    Returns
    -------
    kz,ky,kx : 1d array
        coordinates
    fieldkzkykxxx : 4d array
        transformed fields
    """

    kz, ky, fieldxkzky = _ffttransform_in_yz(ddict,fieldkey)

    nxx = len(ddict[fieldkey+'_xx'])
    nkx = int(len(ddict[fieldkey+'_xx'])/retstep) #warning: this is hard coded to match wlt function output size
    nky = len(ky)
    nkz = len(kz)
    fieldkzkykxxx = np.zeros((nkz,nky,2*nkx,nxx),dtype=np.complex_)

    for kyidx in range(0,len(ky)):
        for kzidx in range(0,len(kz)):
            positivekx, rightfieldkz0ky0kxxx = wlt(ddict[fieldkey+'_xx'],fieldxkzky[:,kzidx,kyidx],retstep=retstep)
            negativekx, leftfieldkz0ky0kxxx = wlt(ddict[fieldkey+'_xx'],np.conj(fieldxkzky[:,kzidx,kyidx]),retstep=retstep)
            leftfieldkz0ky0kxxx = np.conj(leftfieldkz0ky0kxxx) #use reality condition to compute negative kxs
            fieldkzkykxxx[kzidx,kyidx,nkx:,:] = rightfieldkz0ky0kxxx[:,:]
            fieldkzkykxxx[kzidx,kyidx,0:nkx,:] = np.flip(leftfieldkz0ky0kxxx[:,:], axis=0)

    negativekx *= -1
    negativekx = np.flip(negativekx)
    kx = np.concatenate([negativekx,positivekx])

    return kz, ky, kx, fieldkzkykxxx

def take_ifft2(data):
    """
    Computes 2d ifft on given data

    Parameters
    ----------
    data : 2d array
        data in freq space

    Returns
    -------
    ifftdata : 2d array
        data in cartesian space
    """

    ifftdata = np.fft.ifft2(data)*(float(len(data)*len(data[1])))

    return ifftdata

def _iffttransform_in_yz(fftdfields,fieldkey):
    """
    Takes f(x,kz,ky) and computes f(x,z,y) using a 2d fft for some given field

    Parameters
    ----------
    fftdfields : dict
        dict of fields that have been fft transformed in yz
    fieldkey : str
        name of field you want to inverse transform (ex, ey, ez, bx, by, bz, ux, uy, uz)
    """

    fieldifftsweepoverx = []
    for xxindex in range(0,len(fftdfields[fieldkey])):
        fieldslicefft = np.asarray(fftdfields[fieldkey])[xxindex,:,:]
        fieldslice = take_ifft2(fieldslicefft)
        fieldifftsweepoverx.append(fieldslice)
    fieldifftsweepoverx = np.asarray(fieldifftsweepoverx)

    return fieldifftsweepoverx

def iwlt_noscale(t,k,cwtdata):
    """
    Computes inverse wavelet transform, without preserving scale
    i.e given f(t) with w.l.t. W{f(t)}, this function will return A*f(t) = W^(-1){W{f(t)}} where A is some unknown constant

    This function is meant to only be used until we learn how to implement a WLT that preserves this scale.

    Parameters
    ----------
    t : array
        time/position axis of wavelet transform
    k : array
        freq/wavenumber axis of wavelet transform
    cwtdata : 2d array
        wavelet transform data from wlt() function

    Returns
    -------
    f_t : 1d array
        reconstructed original signal computed using inverse wavelet transform
        note this signal will almost always be off by some constant factor
        WARNING: some signals can not be reconstructed well
    """

    N = len(t)
    J = len(k)

    f_t = []
    for _n in range(0,N):
        f_ti = 0.
        for _kidx in range(0,J):
            f_ti += np.real(cwtdata[_kidx,_n])/k[_kidx]**1.
        f_t.append(f_ti)
    f_t = np.asarray(f_t)

    return f_t

def yz_fft_filter(dfields,kycutoff,filterabove,dontfilter=False,verbose=False,keys=['ex','ey','ez','bx','by','bz']):

    import copy
    filteredfields = copy.deepcopy(dfields)

    for _key in keys:
        if(verbose):print("yz_fft_filter is on key: ", _key)
        kz, ky, filteredfields[_key] = _ffttransform_in_yz(filteredfields,_key) #compute A(x,kz,ky) 
        
        if(not(dontfilter)):
            for _i in range(0,len(ky)):
                if(filterabove):
                    if(np.abs(ky[_i]) > kycutoff):
                        filteredfields[_key][:,:,_i] = 0.
                else:   
                    if(np.abs(ky[_i]) <= kycutoff):
                        filteredfields[_key][:,:,_i] = 0.

        filteredfields[_key] = _iffttransform_in_yz(filteredfields,_key) #returns as A(x,z,y)
        filteredfields[_key] = np.swapaxes(filteredfields[_key], 0, 1) #returns as A(z,x,y)
        filteredfields[_key] = np.swapaxes(filteredfields[_key], 1, 2) #returns as A(z,y,x)
        filteredfields[_key] = np.real(filteredfields[_key])

    return filteredfields

def xyz_wlt_fft_filter(kz,ky,kx,dx,,bxkzkykxxx,bykzkykxxx,bzkzkykxxx,
                exkzkykxxx,eykzkykxxx,ezkzkykxxx,
                kycutoff,filterabove,dontfilter=False):
    """
    Mid pass filter in x y and z
    Uses a single wavenumber in y and z and a small range in x to filter

    We assume the user already has axis to the fields in freq space as it takes a long time to compute

    Note: some signals are difficult to filter as the inverse wavelet transform can not reconstruct the original singal well

    Parameters
    ----------
    kz : array
        wavenunmber values
    ky : array
        wavenunmber values
    kx : array
        wavenunmber values
    dx : scalar
        dx spacing
    xx : array
        xx data
    *ai*kzkykxxx : 4D array
        wft data for each field component
    kycutoff : scalar
        ky value to filter above/below
    filterabove : bool
        if truue, filters above; else filters below
    dontfilter : bool (opt)
        debug var- turns off filter

    Returns
    -------
    filteredfields : dict
        filtered fields dictionary

    """

    from lib.arrayaux import find_nearest

    keys = ['ex','ey','ez','bx','by','bz']
    freq_space = {'ex':exkzkykxxx,'ey':eykzkykxxx,'ez':ezkzkykxxx,'bx':bxkzkykxxx,'by':bykzkykxxx,'bz':bzkzkykxxx}
    
    #make dictionary
    filteredfields = {}
    for key in keys:
        filteredfields[key] = np.zeros((len(freq_space[key][:,0,0,0]),len(freq_space[key][0,:,0,0]),len(freq_space[key][0,0,0,:]))) #makes empty arrays of length of zz by yy by xx (warning, length of kx is technically arbitrary as it is the product of the wavelet transform)

    #to test/debug inverse transform, we inverse transform without filterings
    if(not(dontfilter)):
        for key in keys:
            for _kzidx in range(0,len(freq_space[key][:,0,0,0])):
                for _kyidx in range(0,len(freq_space[key][_kzidx,:,0,0])):
                        for _kxidx in range(0,len(freq_space[key][_kzidx,_kyidx,:,0])):
                            for _xxidx in range(0,len(freq_space[key][_kzidx,_kyidx,_kxidx,:])):
                                if(filterabove):
                                    if(np.abs(ky[_kyidx])>kycutoff):
                                        freq_space[key][_kzidx,_kyidx,_kxidx,_xxidx]  = 0.
                                else:
                                    if(np.abs(ky[_kyidx])<=kycutoff):
                                        freq_space[key][_kzidx,_kyidx,_kxidx,_xxidx]  = 0.

    #inverse transform
    for key in keys:
        print("Inverting ", key)
        #take iwlt (inverse transform in xx direction)
        nkx = int(len(freq_space[key][0,0,:,0])/2) #need to rebuild signal from only positive kxs 
        for _kzidx in range(0,len(freq_space[key][:,0,0,0])):
            for _kyidx in range(0,len(freq_space[key][_kzidx,:,0,0])):
                filteredfields[key][_kzidx,_kyidx,:]  = iwlt(xx,kx[nkx:],freq_space[key][_kzidx,_kyidx,nkx:,:])

        #take ifft2 (inverse transform in yy/zz direction)
        filteredfields[key] = np.swapaxes(filteredfields[key], 0, 2) #change index order from (kz,ky,x) to (x,ky,kz)
        filteredfields[key] = np.swapaxes(filteredfields[key], 1, 2) #change index order from  (x,ky,kz) to (x,kz,ky)
        filteredfields[key] = _iffttransform_in_yz(filteredfields,key) #note: input index order is (x,kz,ky) and output is (x,z,y)
        filteredfields[key] = np.swapaxes(filteredfields[key], 0, 2) #change index order from (x,z,y) to (y,z,x)
        filteredfields[key] = np.swapaxes(filteredfields[key], 0, 1) #change index order from (y,z,x) to (z,y,x)
        filteredfields[key] = np.real(filteredfields[key])

    #reconstruct coordinates of data
    for key in keys:
        filteredfields[key+'_xx'] = xx
        filteredfields[key+'_yy'] = [dx*_i+dx/2. for _i in range(0,len(freq_space[key][0,:,0,0]))]
        filteredfields[key+'_zz'] = [dx*_i+dx/2. for _i in range(0,len(freq_space[key][:,0,0,0]))]

    return filteredfields

def compute_energization(Cor,dv):
    """
    Computes energization of velocity signature by integrating over velocity space
    This function assumes a square grid

    Parameters
    ---------- 
    Cor : 2d array
        x slice of velocity signature
    dv : float
        spacing between velocity grid points

    Returns
    -------  
    netE : float
        net energization/ integral of C(x0; vy, vx)
    """

    netE = 0.
    for i in range(0,len(Cor)):
        for j in range(0,len(Cor[i])):
            netE += Cor[i][j]*dv*dv #assumes square grid

    return netE

def compute_gain_due_to_jdotE(dflow,xvals,jdotE,isIon,verbose=False):
    """
    xvals and jdotE are parallel 1D arrays
    
    Accumulates energy gain due to traversing each box from xvals[0] to xvals[-1]

    Assumes dx of xvals is small, and that that flow is approx constant in dx
    
    Assumes xvals and jdotE are parallel, with xvals starting at the lowest val

    Integrates 'backwards' as particles start at the far end of the box and then gain energy

    Parameters
    ----------
    dflow : dict
        flow dic
    xvals : array
        position data parallel to jdotE
    jdotE : array
        energization rate array (can be total j dot E, species j dot E, component jiEi, etc..)
    verbose : bool (opt)
        if true, prints debug statements

    Returns
    -------
    xcoord_Ener_due_to_jdotE : array
        position data parallel to Ener_due_to_jdotE
    Ener_due_to_jdotE : array
        total energy acculmulated due to 
    """

    from lib.arrayaux import find_nearest

    dflowavg = get_average_flow_over_yz(dflow)

    if(not(isIon)):
        flowcoordkey = 'ue_xx'
        flowvalkeyx = 'ue'
        flowvalkeyy = 've'
        flowvalkeyz = 'we'
    else:
        flowcoordkey = 'ui_xx'
        flowvalkeyx = 'ui'
        flowvalkeyy = 'vi'
        flowvalkeyz = 'wi'

    xcoord_Ener_due_to_jdotE = np.zeros(len(xvals)-1)
    Ener_due_to_jdotE = np.zeros(len(xvals)-1)

    xvals = np.flip(xvals)
    jdotE = np.flip(jdotE)

    for _i in range(0,len(Ener_due_to_jdotE)-1):
        delta_x = xvals[_i+1]-xvals[_i]
        xcoord_Ener_due_to_jdotE[_i] = xvals[_i+1]
    
        leftidx = find_nearest(xvals[_i+1],dflowavg[flowcoordkey])
        rightidx = find_nearest(xvals[_i],dflowavg[flowcoordkey])

        xvelocity = np.mean(dflowavg[flowvalkeyx][0,0,leftidx:rightidx+1])  #NOTE: if v varies a lot over the full box, this will be incorrect if the edges of this box don't fall on the edges of cells, as the edge cells will contribute more than they should. We assume this is small for simplicity 

        if(delta_x < 0):
            if(xvelocity >= 0):
                #Warning! we assume the parcel is always flowing in one direction. Sometimes, it flows approx zero but technically the 'wrong way'. In that case we just skip it. The user should be careful of this when interpretting results
                pass
        if(delta_x > 0):
            if(xvelocity <= 0):
                pass

        egain_across_box = np.abs(delta_x/xvelocity) * jdotE[_i]
        
        if(verbose):print('xvelocity: ',xvelocity)
        if(delta_x < 0):
            if(xvelocity >= 0):
                if(xvelocity > -.01):
                    #see above warning
                    pass

        if(delta_x > 0):
            if(xvelocity <= 0):
                if(xvelocity < .01):
                    #see above warning
                    pass

        Ener_due_to_jdotE[_i] = egain_across_box
        if(_i > 0):
            Ener_due_to_jdotE[_i] += Ener_due_to_jdotE[_i-1]

    return xcoord_Ener_due_to_jdotE, Ener_due_to_jdotE

def bin_integrate_gyro(x_grid, y_grid, C_vals, rmax, nrbins):
    """
    Integrates along curves of constant radius along 360 degrees of angle.

    For simplicity, we place value into bins based on bin center. If bin center is within r_n and r_n+dr, it goes into the nth bin. 
    This most easily conserves total energization rate. (Otherwise, we would need to carefully interpolate and divide each our square grid!)

    Bins based on location of bin center

    Parameters
    ----------
    #TODO: finish!

    """
    from lib.arrayaux import find_nearest

    drval = float(rmax)/float(nrbins)
    r_bins = np.asarray([itemp*drval for itemp in range(nrbins)])
    
    dr = r_bins[1]-r_bins[0]
    dx = x_grid[1][1]-x_grid[0][0]
    
    if(dr < dx):
        print("Warning, probably should reduce nrbins")
    
    C_binned_out = np.zeros(r_bins.shape)
    
    for _idx in range(0,len(x_grid)):
        for _jdx in range(0,len(x_grid[_idx])):
            rpos =  np.sqrt(x_grid[_idx][_jdx]**2+y_grid[_idx][_jdx]**2)   
            ridx = find_nearest(r_bins,rpos)  
            C_binned_out[ridx] += C_vals[_idx][_jdx]
    r_grid_out = np.zeros
    
    return r_bins, C_binned_out

def compute_gyro_fpc_from_cart_fpc(vx,vy,vz,corez,corey,corex,vmax,nrbins):
    """
    Note: x<-> perp1, y<->perp2, z<->par

    However! the index structure of corei[perp1,perp2,par]
    """
    coreperp = corey+corez

    vpargyro = np.zeros((len(corex),nrbins)) #assumes symmetry in shape of corex
    vperpgyro = np.zeros((len(corex),nrbins))
    corepargyro = np.zeros((len(corex),nrbins))
    coreperpgyro = np.zeros((len(corex),nrbins))
    for _vparidx in range(len(corex)):
        vperpgyro[_vparidx], corepargyro[_vparidx] = bin_integrate_gyro(vx[_vparidx], vy[_vparidx], corex[:,:,_vparidx], vmax, nrbins)
        vperpgyro[_vparidx], coreperpgyro[_vparidx] = bin_integrate_gyro(vx[_vparidx], vy[_vparidx], coreperp[:,:,_vparidx], vmax, nrbins)
        vpargyro[_vparidx][:]=vz[_vparidx,0,0]

    return vpargyro,vperpgyro,corepargyro,coreperpgyro

def compute_compgyro_fpc_from_cart_fpc(vx,vy,vz,corez,corey,corex,vmax,nrbins):
    """
    Note: x<-> perp1, y<->perp2, z<->par

    However! the index structure of corei[perp1,perp2,par]
    """
    coreperp1 = corey
    coreperp2 = corez

    vpargyro = np.zeros((len(corex),nrbins)) #assumes symmetry in shape of corex
    vperpgyro = np.zeros((len(corex),nrbins))
    corepargyro = np.zeros((len(corex),nrbins))
    coreperp1gyro = np.zeros((len(corex),nrbins))
    coreperp2gyro = np.zeros((len(corex),nrbins))
    for _vparidx in range(len(corex)):
        vperpgyro[_vparidx], corepargyro[_vparidx] = bin_integrate_gyro(vx[_vparidx], vy[_vparidx], corex[:,:,_vparidx], vmax, nrbins)
        vperpgyro[_vparidx], coreperp1gyro[_vparidx] = bin_integrate_gyro(vx[_vparidx], vy[_vparidx], coreperp1[:,:,_vparidx], vmax, nrbins)
        vperpgyro[_vparidx], coreperp2gyro[_vparidx] = bin_integrate_gyro(vx[_vparidx], vy[_vparidx], coreperp2[:,:,_vparidx], vmax, nrbins)
        vpargyro[_vparidx][:]=vz[_vparidx,0,0]

    return vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro
