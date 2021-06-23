# analysisfunctions.py>

#functions related to computing correlation and other core analysis functions

import numpy as np

def make2dHistandCey(vmax, dv, x1, x2, y1, y2, dpar, dfields):
    """
    Makes distribution and takes correlation wrt Ey in a given box.

    This function is '2d' as it projects out all z information. I.E. the box is
    always the maximum size in zz and vz.

    #WARNING: this will linearly average the fields within the specified bounds.
    However, if there are no gridpoints within the specified bounds
    it currently will *not* grab any field values and break.
    TODO: use appropriate weighting to nearest field when range is not exactly
    on the edges of the grid

    Parameters
    ----------
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    dv : float
        velocity space grid spacing
        (assumes square)
    x1 : float
        lower x bound
    x2 : float
        upper x bound
    y1 : float
        lower y bound
    y2 : float
        upper y bound
    dpar : dict
        xx vx yy vy data dictionary from readParticlesPosandVelocityOnly
    dfields : dict
        field data dictionary from field_loader
    Returns
    -------
    vx : 2d array
        vx velocity grid
    vy : 2d array
        vy velocity grid
    totalPtcl : float
        total number of particles in the correlation box
    totalFieldpts : float
        total number of field gridpoitns in the correlation box
    Hxy : 2d array
        distribution function in box
    Cey : 2d array
        velocity space sigature data
    """

    fieldkey = 'ey'

    #find average E field based on provided bounds
    gfieldptsx = (x1 <= dfields[fieldkey+'_xx'])  & (dfields[fieldkey+'_xx'] <= x2)
    gfieldptsy = (y1 <= dfields[fieldkey+'_yy']) & (dfields[fieldkey+'_yy'] <= y2)

    goodfieldpts = []
    for i in range(0,len(dfields['ex_xx'])):
        for j in range(0,len(dfields['ex_yy'])):
            for k in range(0,len(dfields['ex_zz'])):
                if(gfieldptsx[i] == True and gfieldptsy[j] == True):
                    goodfieldpts.append(dfields[fieldkey][k][j][i])

    #TODO?: consider forcing user to take correlation over only 1 cell
    if(len(goodfieldpts)==0):
        print("Using weighted_field_average...") #Debug
        avgfield = weighted_field_average((x1+x2)/2.,(y1+y2)/2.,0,dfields,fieldkey) #TODO: make 3d i.e. *don't* just 'project' all z information out and take fields at z = 0
    else:
        avgfield = np.average(goodfieldpts) #TODO: call getfieldaverageinbox here instead
    totalFieldpts = np.sum(goodfieldpts)

    #define mask that includes particles within range
    gptsparticle = (x1 < dpar['x1'] ) & (dpar['x1'] < x2) & (y1 < dpar['x2']) & (dpar['x2'] < y2)
    totalPtcl = np.sum(gptsparticle)

    #make bins
    vxbins = np.arange(-vmax, vmax, dv)
    vx = (vxbins[1:] + vxbins[:-1])/2.
    vybins = np.arange(-vmax, vmax, dv)
    vy = (vybins[1:] + vybins[:-1])/2.

    #make the bins 2d arrays
    _vx = np.zeros((len(vy),len(vx)))
    _vy = np.zeros((len(vy),len(vx)))
    for i in range(0,len(vy)):
        for j in range(0,len(vx)):
            _vx[i][j] = vx[j]

    for i in range(0,len(vy)):
        for j in range(0,len(vx)):
            _vy[i][j] = vy[i]

    vx = _vx
    vy = _vy

    #find distribution
    Hxy,_,_ = np.histogram2d(dpar['p2'][gptsparticle],dpar['p1'][gptsparticle],
                         bins=[vybins, vxbins])

    #calculate correlation
    Cey = -0.5*vy**2*np.gradient(Hxy, dv, edge_order=2, axis=0)*avgfield
    return vx, vy, totalPtcl, totalFieldpts, Hxy, Cey

def make2dHistandCex(vmax, dv, x1, x2, y1, y2, dpar, dfields):
    """
    Same as make2dHistandCey but takes correlation wrt Ex
    """

    fieldkey = 'ex'

    #find average E field based on provided bounds
    gfieldptsx = (x1 <= dfields[fieldkey+'_xx'])  & (dfields[fieldkey+'_xx'] <= x2)
    gfieldptsy = (y1 <= dfields[fieldkey+'_yy']) & (dfields[fieldkey+'_yy'] <= y2)

    goodfieldpts = []
    for i in range(0,len(dfields['ex_xx'])):
        for j in range(0,len(dfields['ex_yy'])):
            for k in range(0,len(dfields['ex_zz'])):
                if(gfieldptsx[i] == True and gfieldptsy[j] == True):
                    goodfieldpts.append(dfields[fieldkey][k][j][i])

    if(len(goodfieldpts)==0):
        print("Warning, no field grid points in given box. Please increase box size or center around grid point.")

    avgfield = np.average(goodfieldpts) #TODO: call getfieldaverageinbox here instead
    totalFieldpts = np.sum(goodfieldpts)

    #define mask that includes particles within range
    gptsparticle = (x1 < dpar['x1'] ) & (dpar['x1'] < x2) & (y1 < dpar['x2']) & (dpar['x2'] < y2)
    totalPtcl = np.sum(gptsparticle)

    #make bins
    vxbins = np.arange(-vmax, vmax, dv)
    vx = (vxbins[1:] + vxbins[:-1])/2.
    vybins = np.arange(-vmax, vmax, dv)
    vy = (vybins[1:] + vybins[:-1])/2.

    #make the bins 2d arrays
    _vx = np.zeros((len(vy),len(vx)))
    _vy = np.zeros((len(vy),len(vx)))
    for i in range(0,len(vy)):
        for j in range(0,len(vx)):
            _vx[i][j] = vx[j]

    for i in range(0,len(vy)):
        for j in range(0,len(vx)):
            _vy[i][j] = vy[i]

    vx = _vx
    vy = _vy

    #find distribution
    Hxy,_,_ = np.histogram2d(dpar['p2'][gptsparticle],dpar['p1'][gptsparticle],
                         bins=[vybins, vxbins])

    #calculate correlation
    Cex = -0.5*vx**2*np.gradient(Hxy, dv, edge_order=2, axis=1)*avgfield
    return vx, vy, totalPtcl, totalFieldpts, Hxy, Cex

def getfieldaverageinbox(x1, x2, y1, y2, dfields, fieldkey):
    """
    Get linear average of fields in box from grid points within box.

    Parameters
    ----------
    x1 : float
        lower x bound
    x2 : float
        upper x bound
    y1 : float
        lower y bound
    y2 : float
        upper y bound
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of field you are averaging (ex, ey, ez, bx, by, bz)

    Returns
    -------
    avgfield : float
        average field in box
    """

    #find average field based on provided bounds
    gfieldptsx = (x1 <= dfields[fieldkey+'_xx'])  & (dfields[fieldkey+'_xx'] <= x2)
    gfieldptsy = (y1 <= dfields[fieldkey+'_yy']) & (dfields[fieldkey+'_yy'] <= y2)

    goodfieldpts = []
    for i in range(0,len(dfields['ex_xx'])):
        for j in range(0,len(dfields['ex_yy'])):
            for k in range(0,len(dfields['ex_zz'])):
                if(gfieldptsx[i] == True and gfieldptsy[j] == True):
                    goodfieldpts.append(dfields[fieldkey][k][j][i])


    # #debug
    # print("numgridpts sampled: " + str(len(goodfieldpts)))


    avgfield = np.average(goodfieldpts)
    return avgfield


def compute_correlation_over_x(dfields, dparticles, vmax, dv, dx):
    """
    Computes f(x; vy, vx), CEx(x; vy, vx), and CEx(x; vy, vx) along different slices of x

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    dpar : dict
        xx vx yy vy data dictionary from readParticlesPosandVelocityOnly
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    dv : float
        spacing between points we sample in velocity space. (Square in vx, vy)
    dx : float
        width of x slice

    Returns
    -------
    CEx_out : 3d array
        CEx(x; vy, vx) data
    CEy_out : 3d array
        CEy(x; vy, vx) data
    x_out : 1d array
        average x position of each slice
    Hxy_out : 3d array
        f(x; vy, vx) data
    vx : 2d array
        vx velocity grid
    vy : 2d array
        vy velocity grid
    """

    CEx_out = []
    CEy_out = []
    x_out = []
    Hxy_out = []

    xsweep = 0.0
    for i in range(0,len(dfields['ex_xx'])):
        print(str(dfields['ex_xx'][i]) +' of ' + str(dfields['ex_xx'][len(dfields['ex_xx'])-1]))
        vx, vy, totalPtcl, totalFieldpts, Hxy, Cey = make2dHistandCey(vmax, dv, xsweep, xsweep+dx, dfields['ey_yy'][0], dfields['ey_yy'][1], dparticles, dfields)
        vx, vy, totalPtcl, totalFieldpts, Hxy, Cex = make2dHistandCex(vmax, dv, xsweep, xsweep+dx, dfields['ey_yy'][0], dfields['ey_yy'][1], dparticles, dfields)
        x_out.append(np.mean([xsweep,xsweep+dx]))
        CEy_out.append(Cey)
        CEx_out.append(Cex)
        Hxy_out.append(Hxy)
        xsweep+=dx

    return CEx_out, CEy_out, x_out, Hxy_out, vx, vy

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
        for j in range(0,len(Cor)):
            netE += Cor[i][j]*dv*dv #assumes square grid

    return netE

def find_nearest(array, value):
    """
    Finds index of element in array with value closest to given value

    Paramters
    ---------
    array : 1d array
        ordered array
    value : float
        value you want to approximately find in array

    Returns
    -------
    idx : int
        index of nearest element
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

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

def shockvel_from_compression_ratio(M):
    """
    Estimates shock velocity using compression ratio

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
