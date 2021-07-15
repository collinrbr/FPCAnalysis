# fieldtransformfunctions.py>

#functions related to lorentz transforming field and computing shock veloicty

import numpy as np

def lorentz_transform_vx(dfields,vx):
    """
    Takes lorentz transform where V=(vx,0,0)
    TODO: check if units work (in particular where did gamma go)

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    vx : float
        boost velocity along x
    """

    from copy import copy

    dfieldslor = copy(dfields) #deep copy

    dfieldslor['ex'] = dfields['ex']
    dfieldslor['ey'] = dfields['ey']-vx*dfields['bz']
    dfieldslor['ez'] = dfields['ez']+vx*dfields['by']
    dfieldslor['bx'] = dfields['bx']
    dfieldslor['by'] = dfields['by']#assume v/c^2 is small
    dfieldslor['bz'] = dfields['bz']#assume v/c^2 is small
    dfieldslor['Vframe_relative_to_sim'] = dfields['Vframe_relative_to_sim'] - vx

    return dfieldslor

#TODO: check if linear frame shift if appropriate here and in lorentz_transform_vx
def transform_flow(dflow,vx):
    """
    Transforms frame of flow data
    """
    dflowtransform = copy(dflow) #deep copy
    dflowtransform['ux'] = dflow['ux']-vx
    dflowtransform['Vframe_relative_to_sim'] = dflow['Vframe_relative_to_sim'] - vx

    return dflowtransform

def shift_particles(dparticles,vx):
    """
    Transforms velocity frame of particles
    """
    dparticlestransform = copy(dparticles) #deep copy
    dparticlestransform['p1'] = dparticles['p1'] - vx #TODO: check units here
    dparticlestransform['Vframe_relative_to_sim'] = dparticles['Vframe_relative_to_sim'] - vx

    return dparticlestransform

def shift_hist():
    """
    Transforms velocity frame of distribution function
    """
    pass

def estimate_shock_pos(dfields, yyindex = 0, zzindex = 0):
    """
    Estimates the position of the shock using first Ex, x axis crossing after the shock forms

    Takes 'rightmost' (i.e. most upstream) significant (i.e. not just small fluctuations)
    zerocrossing as the location of the shock

    """

    xposshock = 0.

    #chop off upstream data by eliminating data to the right of the first
    #significant (i.e >10% of the maximum) local maximimum
    import numpy as np
    from scipy.signal import argrelextrema

    Exvals = np.asarray([dfields['ex'][zzindex][yyindex][i] for i in range(0,len(dfields['ex_xx']))])
    xvals = np.asarray(dfields['ex_xx'])

    if(len(xvals) != len(Exvals)): #debug
        print("Error: shape of xvals does not match Exvals...")
        return 0.

    #find index of local maxima
    xposcandidatesidx = argrelextrema(Exvals, np.greater)

    #get absolute maxima and find shock front by assuming upstream local maxima
    #are all small compared to this.
    #I.e. locate first 'significantly' large local maxima
    #TODO: instead of using maximum, to drop upstream,
    #use physics relationship (from June 17 discussion with Dr. Howes) to find
    Exmax = np.amax(Exvals)

    sigfrac = .05 #minimum fraction relative to Exmax that the local max must be greater than to be considered significant
    shockfrontidx = 0
    for k in range(0,len(xposcandidatesidx[0])):
        if(Exvals[xposcandidatesidx[0][k]] >= sigfrac*Exmax):
            shockfrontidx = xposcandidatesidx[0][k]

    #sweep back from shock front until we cross zero
    zerocrossidx = shockfrontidx
    while(not(Exvals[zerocrossidx] > 0 and Exvals[zerocrossidx-1] <= 0)):
        zerocrossidx -= 1

    #take idx closest to zero
    if(Exvals[zerocrossidx] > abs(Exvals[zerocrossidx-1])):
        zerocrossidx -= 1

    xposshock = xvals[zerocrossidx]
    return xposshock

def shock_from_ex_cross(all_fields,dt=0.01):
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

    #get all shock crossings
    xshockvals = []
    for k in range(0,len(all_fields['dfields'])):
        xshockvals.append(estimate_shock_pos(all_fields['dfields'][k])) #note: doesn't work until shock forms

    #get shock velocity
    #assume that the shock forms after half the simulation TODO: do something better
    startidx = int(len(all_fields['frame'])/2)

    xvals = xshockvals[startidx:]
    framevals = all_fields['frame'][startidx:]

    #convert frame to time
    print("Warning, using dt = 0.01 Omega^-1... TODO: automate loading this...")
    tvals = []
    for k in range(0, len(framevals)):
        tvals.append(framevals[k]*dt)

    #fit to line
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
        return lambda v:M/v-(gamma+1.)/((2./(M-v)**2)+gamma-1)

    from scipy.optimize import fsolve
    vshock = fsolve(shock(M),1.) #start search at vshock=2.

    return vshock

def get_comp_ratio(dfields,upstreambound,downstreambound):
    """
    Find ratio of downstream bz and upstream bz

    Note, typically upstreambound != downstream bound. We should exclude the
    the fields within the shock.

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

    for i in range(0,len(dfields['bz'])):
        for j in range(0,len(dfields['bz'][i])):
            for k in range(0,len(dfields['bz'][i][j])):
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
