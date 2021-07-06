# analysisfunctions.py>

#functions related to computing correlation and other core analysis functions

import numpy as np

def compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dpar, dfields, vshock, fieldkey, directionkey):
    """
    Computes distribution function and correlation wrt to given field

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
    z1 : float
        lower y bound
    z2 : float
        upper y bound
    dpar : dict
        xx vx yy vy zz vz data dictionary from readParticles or readSliceOfParticles
    dfields : dict
        field data dictionary from field_loader
    vshock : float
        velocity of shock in x direction
    fieldkey : str
        name of the field you want to correlate with
        ex,ey,ez,bx,by, or bz
    directionkey : str
        name of direction you want to take the gradient with respect to
        x,y,or z
        *should match the direction of the fieldkey* TODO: check for this automatically

    Returns
    -------
    vx : 3d array
        vx velocity grid
    vy : 3d array
        vy velocity grid
    vz : 3d array
        vz velocity grid
    totalPtcl : float
        total number of particles in the correlation box
    totalFieldpts : float
        total number of field gridpoitns in the correlation box
    Hist : 3d array
        distribution function in box
    Cor : 3d array
        velocity space sigature data in box
    """

    #find average E field based on provided bounds
    gfieldptsx = (x1 <= dfields[fieldkey+'_xx']) & (dfields[fieldkey+'_xx'] <= x2)
    gfieldptsy = (y1 <= dfields[fieldkey+'_yy']) & (dfields[fieldkey+'_yy'] <= y2)
    gfieldptsz = (z1 <= dfields[fieldkey+'_zz']) & (dfields[fieldkey+'_zz'] <= z2)

    goodfieldpts = []
    for i in range(0,len(dfields['ex_xx'])):
        for j in range(0,len(dfields['ex_yy'])):
            for k in range(0,len(dfields['ex_zz'])):
                if(gfieldptsx[i] == True and gfieldptsy[j] == True and gfieldptsz[k]):
                    goodfieldpts.append(dfields[fieldkey][k][j][i])

    #define mask that includes particles within range
    gptsparticle = (x1 < dpar['x1'] ) & (dpar['x1'] < x2) & (y1 < dpar['x2']) & (dpar['x2'] < y2) & (z1 < dpar['x3']) & (dpar['x3'] < z2)
    totalPtcl = np.sum(gptsparticle)

    # #TODO?: consider forcing user to take correlation over only 1 cell
    # if(len(goodfieldpts)==0):
    #     print("Using weighted_field_average...") #Debug
    #     avgfield = weighted_field_average((x1+x2)/2.,(y1+y2)/2.,0,dfields,fieldkey) #TODO: make 3d i.e. *don't* just 'project' all z information out and take fields at z = 0
    # else:
    avgfield = np.average(goodfieldpts) #TODO: call getfieldaverageinbox here instead
    totalFieldpts = np.sum(goodfieldpts)

    #make bins
    vxbins = np.arange(-vmax-dv, vmax+dv, dv)
    vx = (vxbins[1:] + vxbins[:-1])/2.
    vybins = np.arange(-vmax-dv, vmax+dv, dv)
    vy = (vybins[1:] + vybins[:-1])/2.
    vzbins = np.arange(-vmax-dv, vmax+dv, dv)
    vz = (vzbins[1:] + vzbins[:-1])/2.

    #make the bins 3d arrays
    _vx = np.zeros((len(vz),len(vy),len(vx)))
    _vy = np.zeros((len(vz),len(vy),len(vx)))
    _vz = np.zeros((len(vz),len(vy),len(vx)))
    for i in range(0,len(vx)):
        for j in range(0,len(vy)):
            for k in range(0,len(vz)):
                _vx[k][j][i] = vx[i]

    for i in range(0,len(vx)):
        for j in range(0,len(vy)):
            for k in range(0,len(vz)):
                _vy[k][j][i] = vy[j]

    for i in range(0,len(vx)):
        for j in range(0,len(vy)):
            for k in range(0,len(vz)):
                _vz[k][j][i] = vz[k]

    vx = _vx
    vy = _vy
    vz = _vz

    #shift particle data to shock frame
    dpar_p1 = np.asarray(dpar['p1'][gptsparticle][:])
    dpar_p1 -= vshock
    dpar_p2 = np.asarray(dpar['p2'][gptsparticle][:])
    dpar_p3 = np.asarray(dpar['p3'][gptsparticle][:])

    #find distribution
    Hist,_ = np.histogramdd((dpar_p3,dpar_p2,dpar_p1),
                         bins=[vzbins,vybins,vxbins])

    if(directionkey == 'x'):
        axis = 2
        vv = vx
    elif(directionkey == 'y'):
        axis = 1
        vv = vy
    elif(directionkey == 'z'):
        axis = 0
        vv = vz

    #calculate correlation
    Cor = -0.5*vv**2.*np.gradient(Hist, dv, edge_order=2, axis=axis)*avgfield
    return vx, vy, vz, totalPtcl, totalFieldpts, Hist, Cor

def threeVelToTwoVel(vx,vy,vz,planename):
    """
    Converts 3d velocity space arrays to 2d
    Used for plotting

    Parameters
    ----------
    vx : 3d array
        3d vx velocity grid
    vy : 3d array
        3d vy velocity grid
    vz : 3d array
        3d vz velocity grid
    planename : str
        name of plane you want to get 2d grid of

    Returns
    -------
    *Returns 2 of 3 of the following based on planename*
    vx2d : 2d array
        2d vx velocity grid
    vy2d : 2d array
        2d vy velocity grid
    vz2d : 2d array
        2d vz velocity grid
    """

    if(planename == 'xy'):
        vx2d = np.zeros((len(vy),len(vx)))
        vy2d = np.zeros((len(vy),len(vx)))
        for i in range(0,len(vy)):
            for j in range(0,len(vx)):
                vx2d[i][j] = vx[0][i][j]
        for i in range(0,len(vy)):
            for j in range(0,len(vx)):
                vy2d[i][j] = vy[0][i][j]

        return vx2d, vy2d

    elif(planename == 'xz'):
        vx2d = np.zeros((len(vz),len(vx)))
        vz2d = np.zeros((len(vz),len(vx)))
        for i in range(0,len(vz)):
            for j in range(0,len(vx)):
                vx2d[i][j] = vx[i][0][j]
        for i in range(0,len(vz)):
            for j in range(0,len(vx)):
                vz2d[i][j] = vz[i][0][j]

        return vx2d, vz2d

    elif(planename == 'yz'):
        vy2d = np.zeros((len(vz),len(vy)))
        vz2d = np.zeros((len(vz),len(vy)))
        for i in range(0,len(vz)):
            for j in range(0,len(vy)):
                vy2d[i][j] = vy[i][j][0]
        for i in range(0,len(vz)):
            for j in range(0,len(vy)):
                vz2d[i][j] = vz[i][j][0]

        return vy2d, vz2d

def threeHistToTwoHist(Hist,planename):
    """
    Converts 3d Histogram to 2d Histogram by projecting additional axis information onto plane
    Probably should using for plotting only

    Parameters
    ----------
    Cor : 3d array
        3d correlation data
    planename : str
        name of plane you want to project onto

    Returns
    -------
    Hist2d : 2d array
        2d projection of the distribution
    """
    Hist2d = np.zeros((len(Hist),len(Hist[0])))
    if(planename == 'xy'):
        for i in range(0,len(Hist)):
            for j in range(0,len(Hist[i])):
                for k in range(0,len(Hist[i][j])):
                    Hist2d[k][j] += Hist[i][j][k]

        return Hist2d

    elif(planename == 'xz'):
        for i in range(0,len(Hist)):
            for j in range(0,len(Hist[i])):
                for k in range(0,len(Hist[i][j])):
                    Hist2d[k][i] += Hist[i][j][k] #TODO: check this

        return Hist2d

    elif(planename == 'yz'):
        for i in range(0,len(Hist)):
            for j in range(0,len(Hist[i])):
                for k in range(0,len(Hist[i][j])):
                    Hist2d[j][i] += Hist[i][j][k] #TODO: check this

        return Hist2d
    else:
        print("Please enter xy, xz, or yz for planename...")


def threeCorToTwoCor(Cor,planename):
    """
    Converts 3d correlation to 2d correlation

    Parameters
    ----------
    Cor : 3d array
        3d correlation data
    planename : str
        name of plane you want to project onto

    Returns
    -------
    2d array
        2d projection of the correlation
    """
    return threeHistToTwoHist(Cor,planename)

def getfieldaverageinbox(x1, x2, y1, y2, z1, z2, dfields, fieldkey):
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
    z1 : float
        lower z bound
    z2 : float
        upper z bound
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
    gfieldptsz = (z1 <= dfields[fieldkey+'_zz']) & (dfields[fieldkey+'_zz'] <= z2)

    goodfieldpts = []
    for i in range(0,len(dfields['ex_xx'])):
        for j in range(0,len(dfields['ex_yy'])):
            for k in range(0,len(dfields['ex_zz'])):
                if(gfieldptsx[i] == True and gfieldptsy[j] == True and gfieldptsz[k] == True):
                    goodfieldpts.append(dfields[fieldkey][k][j][i])

    avgfield = np.average(goodfieldpts)
    return avgfield


def compute_correlation_over_x(dfields, dparticles, vmax, dv, dx, vshock):
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
    vshock : float
        velocity of shock in x direction

    Returns
    -------
    CEx_out : 4d array
        CEx(x; vz, vy, vx) data
    CEy_out : 4d array
        CEy(x; vz, vy, vx) data
    CEz_out : 4d array
        CEz(x; vz, vy, vx) data
    x_out : 1d array
        average x position of each slice
    Hist_out : 4d array
        f(x; vz, vy, vx) data
    vx : 3d array
        vx velocity grid
    vy : 3d array
        vy velocity grid
    vz : 3d array
        vz velocity grid
    """

    CEx_out = []
    CEy_out = []
    CEz_out = []
    x_out = []
    Hist_out = []

    #TODO: make these an input parameters
    x1 = dfields['ex_xx'][0]
    x2 = dfields['ex_xx'][1]
    y1 = dfields['ex_yy'][0]
    y2 = dfields['ex_yy'][1]
    z1 = dfields['ex_zz'][0]
    z2 = dfields['ex_zz'][1]

    i = 0
    while(x2 <= dfields['ex_xx'][-1]):
        print(str(dfields['ex_xx'][i]) +' of ' + str(dfields['ex_xx'][len(dfields['ex_xx'])-1]))
        vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEx = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'ex', 'x')
        vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEy = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'ey', 'y')
        vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEz = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'ez', 'z')
        x_out.append(np.mean([x1,x2]))
        CEx_out.append(CEx)
        CEy_out.append(CEy)
        CEz_out.append(CEz)
        Hist_out.append(Hist)
        x1+=dx
        x2+=dx
        i+=1

    return CEx_out, CEy_out, CEz_out, x_out, Hist_out, vx, vy, vz

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

def compute_energization_over_x(Cor_array,dv):
    """
    Runs compute_energization for each x slice of data

    Parameters
    ----------
    Cor_array : 3d array
        array of x slices of velocity signatures
    dv : float
        spacing between velocity grid points

    Returns
    -------
    C_E_out : 1d array
        array of net energization/ integral of C(x0; vy, vx) for each slice of x
    """

    C_E_out = []
    for k in range(0,len(Cor_array)):
        C_E_out.append(compute_energization(Cor_array[k],dv))

    return np.asarray(C_E_out)

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

def take_fft2(data,daxisx0,daxis1):
    """
    Takes fft2 and returns wavenumber coordiantes
    """

    k0 = 2.*np.pi*np.fft.fftfreq(len(data),daxisx0)
    k1 = 2.*np.pi*np.fft.fftfreq(len(data[1]),daxis1)

    fftdata = np.fft.fft2(data)

    return k0, k1, fftdata
