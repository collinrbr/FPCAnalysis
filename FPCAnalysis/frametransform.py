# fieldtransformfunctions.py>

# functions related to lorentz transforming field and computing shock veloicty

import numpy as np


def lorentz_transform_vx(dfields, vx):
    """
    Takes lorentz transform where V=(vx,0,0)

    Assumes gamma is small

    Units are meant to work with dHybridR data, (default normalization)

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    vx : float
        boost velocity along x
    """

    from copy import copy

    dfieldslor = copy(dfields)  # deep copy

    dfieldslor['ex'] = dfields['ex']
    dfieldslor['ey'] = dfields['ey']-vx*dfields['bz']
    dfieldslor['ez'] = dfields['ez']+vx*dfields['by']
    dfieldslor['bx'] = dfields['bx']
    dfieldslor['by'] = dfields['by']  # assume v/c^2 is small
    dfieldslor['bz'] = dfields['bz']  # assume v/c^2 is small
    dfieldslor['Vframe_relative_to_sim'] = dfields['Vframe_relative_to_sim'] + vx

    return dfieldslor

def lorentz_transform_vx_c(dfields, vx, c):
    """
    Takes lorentz transform where V=(vx,0,0)

    Units are meant to work with Tristan mp2 data, using normalized=True when loading

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    vx : float
        boost velocity along x in Ma (aka v / va)
    """

    from copy import deepcopy

    dfieldslor = deepcopy(dfields)  # deep copy

    gamma = 1./np.sqrt(1.-(vx/c)**2)

    dfieldslor['ex'] = dfieldslor['ex']
    dfieldslor['ey'] = gamma*(dfields['ey']-vx*dfields['bz'])
    dfieldslor['ez'] = gamma*(dfields['ez']+vx*dfields['by'])

    dfieldslor['bx'] = dfields['bx']
    dfieldslor['by'] = gamma*(dfields['by']+vx/c**2*dfields['ez']) 
    dfieldslor['bz'] = gamma*(dfields['bz']-vx/c**2*dfields['ey'])

    dfieldslor['Vframe_relative_to_sim'] = (dfields['Vframe_relative_to_sim']+vx)/(1.+vx*dfields['Vframe_relative_to_sim']/c**2)

    return dfieldslor

def lorentz_transform_v(dfields, vx, vy, vz, c):
    """
    Takes lorentz transform where V=(vx,0,0)

    Units are meant to work with Tristan mp2 data, using normalized=True when loading

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    vx : float
        boost velocity along x in Ma (aka v / va)
    """

    from copy import deepcopy

    dfieldslor = deepcopy(dfields)  # deep copy

    vtot = np.sqrt(vx**2+vy**2+vz**2)
    if(vtot/c >= 1): print("ERROR vtot/c was greater than 1 vtot/c vtot c", vtot/c, vtot, c)
    if(vtot/c >= 1): exit()

    gamma = 1./np.sqrt(1-(vtot/c)**2)

    dfieldslor['ex']= gamma*(dfields['ex']+vy*dfields['bz']-vz*dfields['by'])-(gamma-1.0)*(dfields['ex']*vx/vtot+dfields['ey']*vy/vtot+dfields['ez']*vz/vtot)
    dfieldslor['ey']= gamma*(dfields['ey']-vx*dfields['bz']+vz*dfields['bx'])-(gamma-1.0)*(dfields['ex']*vx/vtot+dfields['ey']*vy/vtot+dfields['ez']*vz/vtot)
    dfieldslor['ez']= gamma*(dfields['ez']+vx*dfields['by']-vy*dfields['bx'])-(gamma-1.0)*(dfields['ex']*vx/vtot+dfields['ey']*vy/vtot+dfields['ez']*vz/vtot)

    #!!!!!Pressed for time and the magnetic fields aren't used whenever this function is used, so I will implement it later TODO: implement
    dfieldslor['bx'] = dfields['bx']
    dfieldslor['by'] = gamma*(dfields['by']+vx/c**2*dfields['ez']) #TODO: check these units (i.e. should there be an extra 1/c factor?)
    dfieldslor['bz'] = gamma*(dfields['bz']-vx/c**2*dfields['ey'])

    return dfieldslor


# TODO: check if linear frame shift is appropriate here and in lorentz_transform_vx
def transform_flow(dflow, vx):
    """
    Transforms frame of flow data

    Parameters
    ----------
    dflow : dict
        flow data dictionary from flow_loader
    vx : float
        boost velocity along x
    """
    from copy import copy

    dflowtransform = copy(dflow)  # deep copy
    dflowtransform['ux'] = dflow['ux']-vx
    dflowtransform['Vframe_relative_to_sim'] = dflow['Vframe_relative_to_sim'] + vx

    return dflowtransform


def shift_particles(dparticles, vx):
    """
    Transforms velocity frame of particles by vx

    Warning: Make sure vx and dparticles['p1'] have same units!

    We assume a linear boost is close enough

    Parameters
    ----------
    dparticles : dict
        particle data dictionary
    vx : float
        boost velocity along x
    """
    from copy import deepcopy

    dparticlestransform = deepcopy(dparticles)  # deep copy
    dparticlestransform['p1'] = dparticles['p1'] - vx
    dparticlestransform['Vframe_relative_to_sim'] = dparticles['Vframe_relative_to_sim'] + vx

    return dparticlestransform

def shift_particles_tristan(dparticles, vx, beta, mi_me, isIon, Ti_Te = 1., galileanboost = True, c = None):
    """
    Transforms velocity frame of particles

    Works with Tristan data

    TODO: rewrite to be relativistic (should do the same for shift current and for calculating the velocity of the shock)

    TODO: rewrite parameter inputs!

    c is in units of va (c_input = c/va (val of 3 means c is 3 times va)

    Parameters
    ----------
    dparticles : dict
        particle data dictionary
    vx : float
        boost velocity along x in Ma (aka v / va)
    """
    import copy

    if(galileanboost): #Useful to save comp time
        dparticlestransform = copy.deepcopy(dparticles)  # deep copy
        vxaliases = ['p1','ui','ue','vx'] #names of keys that correspond to 'vx'
        for key in vxaliases:
            if(key in dparticles.keys()):
                if(isIon):
                    dparticlestransform[key] = dparticles[key] - vx / np.sqrt(beta)
                else:
                    dparticlestransform[key] = dparticles[key] - vx * (1./(np.sqrt(beta)*(mi_me**0.5)))*(Ti_Te)
        dparticlestransform['Vframe_relative_to_sim'] = dparticles['Vframe_relative_to_sim'] + vx

    else:
        pass

    return dparticlestransform

def shift_curr(dcurr, vx, beta, Ti_Te = 1., q=1.):
    """
    Works with Tristan data (normalized = True)
    """

    #TODO: IF VX IS LARGE ENOUGH, WE SHOULD CONSIDER LORENTZ BOOST-> should do this for all boosts and for shock speed determination

    import copy

    dcurrtransform = copy.deepcopy(dcurr)
    keyalias = ['jx','ux']
    for key in dcurrtransform.keys():
        if(key in dcurr.keys()):
            dcurrtransform[key] = dcurr[key] - vx / np.sqrt(beta)
    dcurrtransform['Vframe_relative_to_sim'] = dcurr['Vframe_relative_to_sim'] + vx

    return dcurrtransform


def shift_hist():
    """
    Transforms velocity frame of distribution function
    """
    pass


def estimate_shock_pos(dfields, yyindex=0, zzindex=0, sigfrac = .1):
    """
    Estimates the position of the shock using first Ex, x axis crossing after the shock forms

    Takes 'rightmost' (i.e. most upstream) significant (i.e. not just small fluctuations)
    zerocrossing as the location of the shock

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    yyindex : int, opt
        index of data along yy axis
    zzindex : int, opt
        index of data along yy axis
    sigfrac : float, opt
        How large a fluctuation needs to be (relative to max) to be 'signficant'. For the shock ripple dHybridR m06th4 sim we used 0.05 and for the nonadiabatic tristan mp1 sim we used .175
    """

    xposshock = 0.

    # chop off upstream data by eliminating data to the right of the first
    #  significant (i.e >10% of the maximum) local maximimum
    import numpy as np
    from scipy.signal import argrelextrema

    Exvals = np.asarray([dfields['ex'][zzindex][yyindex][i]
                         for i in range(0, len(dfields['ex_xx']))])
    xvals = np.asarray(dfields['ex_xx'])

    if(len(xvals) != len(Exvals)):  # debug
        print("Error: shape of xvals does not match Exvals...")
        return 0.

    # find index of local maxima
    xposcandidatesidx = argrelextrema(Exvals, np.greater)

    # get absolute maxima and find shock front by assuming upstream local maxima
    #  are all small compared to this.
    #  I.e. locate first 'significantly' large local maxima
    #  TODO: instead of using maximum, to drop upstream,
    #  use physics relationship (from June 17 discussion with Dr. Howes) to find
    Exmax = np.amax(Exvals)

    shockfrontidx = 0
    for k in range(0, len(xposcandidatesidx[0])):
        if(Exvals[xposcandidatesidx[0][k]] >= sigfrac*Exmax):
            shockfrontidx = xposcandidatesidx[0][k]

    # sweep back from shock front until we cross zero
    zerocrossidx = shockfrontidx
    while(not(Exvals[zerocrossidx] > 0 and Exvals[zerocrossidx-1] <= 0)):
        zerocrossidx -= 1

    # take idx closest to zero
    if(Exvals[zerocrossidx] > abs(Exvals[zerocrossidx-1])):
        zerocrossidx -= 1

    xposshock = xvals[zerocrossidx]
    return xposshock


def shock_from_ex_cross(all_fields, dt):
    """
    Estimates shock velocity by tracking the first 'signficant' Ex zero crossing

    Parameters
    ----------
    all_fields : dict
        dictionary containing all field information and location (for each time slice)
        Fields are Ordered (z,y,x)
        Contains key with frame number

    dt : float
        size of time step in inverse Omega_ci0

    Returns
    -------
    vshock : float
        shock velocity
    xshockvals : array
        x position of Ex crossing at each frame. Parallel to all_fields['frame']
    """

    # get all shock crossings
    # note: doesn't work until shock forms
    xshockvals = []
    for k in range(0, len(all_fields['dfields'])):
        xshockvals.append(estimate_shock_pos(all_fields['dfields'][k]))

    # get shock velocity
    # assume that the shock forms after half the simulation TODO: do something better
    startidx = int(len(all_fields['frame'])/2)

    xvals = xshockvals[startidx:]
    framevals = all_fields['frame'][startidx:]

    # convert frame to time
    tvals = []
    for k in range(0, len(framevals)):
        tvals.append(framevals[k]*dt)

    # fit to line
    vshock, v0 = np.polyfit(tvals, xvals, 1)

    return vshock, xshockvals


def shockvel_from_compression_ratio(M):
    """
    Estimates shock velocity

    Parameters
    ----------
    M : float
        mach speed of inflow

    Returns
    -------
    vshock : float
        estimated mach speed of shock
    """

    def shock(M):
        gamma = 5./3.
        return lambda v: M/v-(gamma+1.)/((2./(M-v)**2)+gamma-1)

    from scipy.optimize import fsolve
    vshock = fsolve(shock(M), 1.)  # start search at vshock=2.

    return vshock


def get_comp_ratio(dfields, upstreambound, downstreambound):
    """
    Find ratio of downstream bz and upstream bz

    Note, typically upstreambound != downstream bound. That is we often exclude the
    the fields within the shock.

    Assumes downstream is towards the origin and upstream is away from the origin

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    downstreambound : float
        x position of the end of the upstream position
    upstreambound : float
        x position of the end of the downstream position

    Returns
    -------
    ratio : float
        compression ratio computed from compression of bz field
    bzdownstrm : float
        average bz downstream
    bzupstrm : float
        average bz upstream
    """

    if(upstreambound < downstreambound):
        print('Error, upstream bound should not be less than downstream bound...')
        return

    bzsumdownstrm = 0.
    numupstreampoints = 0.
    bzsumupstrm = 0.
    numdownstreampoints = 0.

    for i in range(0, len(dfields['bz'])):
        for j in range(0, len(dfields['bz'][i])):
            for k in range(0, len(dfields['bz'][i][j])):
                if(dfields['bz_xx'][k] >= upstreambound):
                    bzsumupstrm += dfields['bz'][i][j][k]
                    numupstreampoints += 1.
                elif(dfields['bz_xx'][k] <= downstreambound):
                    bzsumdownstrm += dfields['bz'][i][j][k]
                    numdownstreampoints += 1.

    bzdownstrm = bzsumdownstrm/numdownstreampoints
    bzupstrm = bzsumupstrm/numupstreampoints
    ratio = bzdownstrm/bzupstrm

    return ratio, bzdownstrm, bzupstrm
