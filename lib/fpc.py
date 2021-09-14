# fpc.py>

#functions related to computing FPC

import numpy as np

#TODO: this function is very non optimized, mainly due to repeated searching of dparticles array. Should optimize this by passing particle subsets (ie boxes of particles)
#to do FPC of
def compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dpar, dfields, vshock, fieldkey, directionkey):
    """
    Computes distribution function and correlation wrt to given field

    Function will automatically shift frame of particles if particles are in simulation frame.
    However, it is more efficient to shift particles before calling this function.

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
                if(gfieldptsx[i] == True and gfieldptsy[j] == True and gfieldptsz[k] == True):
                    goodfieldpts.append(dfields[fieldkey][k][j][i])

    #define mask that includes particles within range
    gptsparticle = (x1 < dpar['x1'] ) & (dpar['x1'] < x2) & (y1 < dpar['x2']) & (dpar['x2'] < y2) & (z1 < dpar['x3']) & (dpar['x3'] < z2)
    totalPtcl = np.sum(gptsparticle)

    avgfield = np.average(goodfieldpts) #TODO: call getfieldaverageinbox here instead
    totalFieldpts = np.sum(goodfieldpts)

    if(dfields['Vframe_relative_to_sim'] != vshock):
        "WARNING: dfields is not in the same frame as the provided vshock"

    #shift particle data to shock frame if needed
    if(dfields['Vframe_relative_to_sim'] == vshock and dpar['Vframe_relative_to_sim'] == 0.): #TODO: use shift particles function
        dpar_p1 = np.asarray(dpar['p1'][gptsparticle][:])
        dpar_p1 -= vshock
        dpar_p2 = np.asarray(dpar['p2'][gptsparticle][:])
        dpar_p3 = np.asarray(dpar['p3'][gptsparticle][:])
    elif(dpar['Vframe_relative_to_sim'] != vshock):
        "WARNING: particles were not in simulation frame or provided vshock frame. This FPC is probably incorrect..."
    else:
        dpar_p1 = np.asarray(dpar['p1'][gptsparticle][:])
        dpar_p2 = np.asarray(dpar['p2'][gptsparticle][:])
        dpar_p3 = np.asarray(dpar['p3'][gptsparticle][:])

    #build dparticles subset using shifted particle data
    #TODO: this isnt clean code (using dpar_p1/2/3 'multiple times' in histogram and in compute_cprime)
    dparsubset = {
      'p1':dpar_p1,
      'p2':dpar_p2,
      'p3':dpar_p3,
      'x1':dpar['x1'][gptsparticle][:],
      'x2':dpar['x2'][gptsparticle][:],
      'x3':dpar['x3'][gptsparticle][:],
      'Vframe_relative_to_sim':dpar['Vframe_relative_to_sim']
    }

    cprimebinned,hist,vx,vy,vz = compute_cprime_hist(dparsubset,dfields,fieldkey,vmax,dv)
    cor = compute_cor_from_cprime(cprimebinned,vx,vy,vz,dv,directionkey)

    return vx, vy, vz, totalPtcl, totalFieldpts, hist, cor

def compute_all_hist_and_cor():
    pass

def compute_correlation_over_x(dfields, dparticles, vmax, dv, dx, vshock, xlim=None, ylim=None, zlim=None):
    """
    Computes f(x; vy, vx), CEx(x; vy, vx), and CEx(x; vy, vx) along different slices of x

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    dparticles : dict
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
    xlim : array
        array of limits in x, defaults to None
    ylim : array
        array of limits in y, defaults to None
    zlim : array
        array of limits in z, defaults to None

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

    if xlim is not None:
        x1 = xlim[0]
        x2 = x1+dx
        xEnd = xlim[1]
    # If xlim is None, use lower x edge to upper x edge extents
    else:
        x1 = dfields['ex_xx'][0]
        x2 = x1 + dx
        xEnd = dfields['ex_xx'][-1]
    if ylim is not None:
        y1 = ylim[0]
        y2 = ylim[1]
    # If ylim is None, use lower y edge to lower y edge + dx extents
    else:
        y1 = dfields['ex_yy'][0]
        y2 = y1 + dx
    if zlim is not None:
        z1 = zlim[0]
        z2 = zlim[1]
    # If zlim is None, use lower z edge to lower z edge + dx extents
    else:
        z1 = dfields['ex_zz'][0]
        z2 = z1 + dx

    while(x2 <= xEnd):
        # This print statement is no longer correct now that we are taking the start points as inputs
        #print(str(dfields['ex_xx'][i]) +' of ' + str(dfields['ex_xx'][len(dfields['ex_xx'])-1]))
        vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEx = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'ex', 'x')
        vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEy = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'ey', 'y')
        vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEz = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'ez', 'z')
        print(x1,x2,y1,y2,z1,z2)
        print(totalPtcl)
        x_out.append(np.mean([x1,x2]))
        CEx_out.append(CEx)
        CEy_out.append(CEy)
        CEz_out.append(CEz)
        Hist_out.append(Hist)
        x1+=dx
        x2+=dx

    return CEx_out, CEy_out, CEz_out, x_out, Hist_out, vx, vy, vz

def compute_all_correlation_over_x():
    pass

def get_3d_weights(xx,yy,zz,idxxx1,idxxx2,idxyy1,idxyy2,idxzz1,idxzz2,dfields,fieldkey):
    """
    Calculates the weight associated with trilinear interpolation

    Parameters
    ----------
    xx : float
        test xx position
    yy : float
        test yy position
    zz : float
        test zz position
    idx**(1/2) : int
        index of positional value of box corner (lower then upper value)
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of the field you want to correlate with
        ex,ey,ez,bx,by, or bz

    Returns
    -------
    w* : float
        weight associated with each corner of box
    """

    #get weights by 'volume fraction' of cell
    w1 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy)*(dfields[fieldkey+'_zz'][idxzz1]-zz))
    w2 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy)*(dfields[fieldkey+'_zz'][idxzz1]-zz))
    w3 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy)*(dfields[fieldkey+'_zz'][idxzz1]-zz))
    w4 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy)*(dfields[fieldkey+'_zz'][idxzz2]-zz))
    w5 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy)*(dfields[fieldkey+'_zz'][idxzz1]-zz))
    w6 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy)*(dfields[fieldkey+'_zz'][idxzz2]-zz))
    w7 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy)*(dfields[fieldkey+'_zz'][idxzz2]-zz))
    w8 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy)*(dfields[fieldkey+'_zz'][idxzz2]-zz))

    vol = w1+w2+w3+w4+w5+w6+w7+w8

    if(vol == 0.):
        print("Error in getting weights! Found a zero volume.")

    #normalize to one
    w1 /= vol
    w2 /= vol
    w3 /= vol
    w4 /= vol
    w5 /= vol
    w6 /= vol
    w7 /= vol
    w8 /= vol

    #debug (should sum to 1)
    if(False):
        print('sum of weights: ' + str(w1+w2+w3+w4+w5+w6+w7+w8))

    return w1,w2,w3,w4,w5,w6,w7,w8

#estimates the field at some point within a cell by taking a weighted average of the surronding grid points
#NOTE: this assumes the sides of the box are all in either the xy,xz, or yz plane
#TODO:FIX (weight is no longer 1?)
def weighted_field_average(xx, yy, zz, dfields, fieldkey):
    """
    Uses trilinear interpolation to estimate field value at given test location

    Parameters
    ----------
    xx : float
        test xx position
    yy : float
        test yy position
    zz : float
        test zz position
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of the field you want to correlate with
        ex,ey,ez,bx,by, or bz

    Returns
    -------
    fieldaverage : float
        field value at given test location found using trilinear interpolation
    """

    from lib.array_ops import find_two_nearest

    idxxx1, idxxx2 = find_two_nearest(dfields[fieldkey+'_xx'],xx)
    idxyy1, idxyy2 = find_two_nearest(dfields[fieldkey+'_yy'],yy)
    idxzz1, idxzz2 = find_two_nearest(dfields[fieldkey+'_zz'],zz)

    #find weights
    w1,w2,w3,w4,w5,w6,w7,w8 = get_3d_weights(xx,yy,zz,idxxx1,idxxx2,idxyy1,idxyy2,idxzz1,idxzz2,dfields,fieldkey)


    #TODO: fix indexing here
    #take average of field
    tolerance = 0.001
    if(abs(w1+w2+w3+w4+w5+w6+w7+w8-1.0) >= tolerance):
        print("Warning: sum of weights in trilinear interpolation was not close enought to 1. Value was: " + str(w1+w2+w3+w4+w5+w6+w7+w8))
    fieldaverage = w1*dfields[fieldkey][idxzz1][idxyy1][idxxx1]
    fieldaverage +=w2*dfields[fieldkey][idxzz1][idxyy1][idxxx2]
    fieldaverage +=w3*dfields[fieldkey][idxzz1][idxyy2][idxxx1]
    fieldaverage +=w4*dfields[fieldkey][idxzz2][idxyy1][idxxx1]
    fieldaverage +=w5*dfields[fieldkey][idxzz1][idxyy2][idxxx2]
    fieldaverage +=w6*dfields[fieldkey][idxzz2][idxyy2][idxxx2]
    fieldaverage +=w7*dfields[fieldkey][idxzz2][idxyy2][idxxx1]
    fieldaverage +=w8*dfields[fieldkey][idxzz2][idxyy1][idxxx2]

    # #debug
    # if(True):
    #     print('fields:')
    #     print(dfields[fieldkey][idxzz1][idxyy1][idxxx1])
    #     print(dfields[fieldkey][idxzz1][idxyy1][idxxx2])
    #     print(dfields[fieldkey][idxzz1][idxyy2][idxxx1])
    #     print(dfields[fieldkey][idxzz2][idxyy1][idxxx1])
    #     print(dfields[fieldkey][idxzz1][idxyy2][idxxx2])
    #     print(dfields[fieldkey][idxzz2][idxyy2][idxxx2])
    #     print(dfields[fieldkey][idxzz2][idxyy2][idxxx1])
    #     print(dfields[fieldkey][idxzz2][idxyy1][idxxx2])
    #     print(fieldaverage)
    #     print('weights')
    #     print(w1)
    #     print(w2)
    #     print(w3)
    #     print(w4)
    #     print(w5)
    #     print(w6)
    #     print(w7)
    #     print(w8)

    return fieldaverage

def compute_cprime_hist(dparticles,dfields,fieldkey,vmax,dv):
    """
    Computes cprime for all particles passed to it

    Parameters
    ----------
    dparticles : dict
        particle data dictionary
    dfields : dict
        field data dictonary
    fieldkey : str
        name of the field you want to correlate with
        ex,ey,ez,bx,by, or bz
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    dv : float
        velocity space grid spacing
        (assumes square)

    Returns
    -------
    cprimebinned : 3d array
        cprime binned into appropriate distribution function using appropriate weights (TODO: document this var better)
    Hist : 3d array
        distribution function in box
    vx : 3d array
        vx velocity grid
    vy : 3d array
        vy velocity grid
    vz : 3d array
        vz velocity grid
    """

    #-TODO: shift particles to correctframe-----------

    from scipy.stats import binned_statistic_dd

    if(fieldkey == 'ex' or fieldkey == 'bx'):
        vvkey = 'p1'
    elif(fieldkey == 'ey' or fieldkey == 'by'):
        vvkey = 'p2'
    elif(fieldkey == 'ez' or fieldkey == 'bz'):
        vvkey = 'p3'


    #compute cprime for each particle
    cprimew = [] #TODO: not sure what to call this (this is technically not cprime until we bin)
    for i in range(0, len(dparticles['x1'])):
        fieldval = weighted_field_average(dparticles['x1'][i], dparticles['x2'][i], dparticles['x3'][i], dfields, fieldkey)
        q = 1. #WARNING: might not always be true TODO: automate grabbing q and fix this
        cprimew.append(q*dparticles[vvkey][i]*fieldval)
    cprimew = np.asarray(cprimew)

    #bin into cprime(vx,vy,vz)
    vxbins = np.arange(-vmax, vmax+dv, dv)
    vx = (vxbins[1:] + vxbins[:-1])/2.
    vybins = np.arange(-vmax, vmax+dv, dv)
    vy = (vybins[1:] + vybins[:-1])/2.
    vzbins = np.arange(-vmax, vmax+dv, dv)
    vz = (vzbins[1:] + vzbins[:-1])/2.

    hist,_ = np.histogramdd((dparticles['p3'],dparticles['p2'],dparticles['p1']),bins=[vzbins,vybins,vxbins])
    cprimebinned,_ = np.histogramdd((dparticles['p3'],dparticles['p2'],dparticles['p1']),bins=[vzbins,vybins,vxbins],weights=cprimew)

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

    return cprimebinned,hist,vx,vy,vz

def compute_cor_from_cprime(cprimebinned,vx,vy,vz,dv,directionkey):
    """
    Computes correlation from cprime

    Parameters
    ----------
    cprimebinned : 3d array
        cprime binned into appropriate distribution function using appropriate weights (TODO: document this var better)
    vx : 3d array
        vx velocity grid
    vy : 3d array
        vy velocity grid
    vz : 3d array
        vz velocity grid
    dv : float
        velocity space grid spacing
        (assumes square)
    directionkey : str
        direction we are taking the derivative w.r.t. (x,y,z)
    """
    #TODO: figure way to automatically handle direction of taking derivative
    if(directionkey == 'x'):
        axis = 2
        vv = vx
    elif(directionkey == 'y'):
        axis = 1
        vv = vy
    elif(directionkey == 'z'):
        axis = 0
        vv = vz

    cor = -vv/2.*np.gradient(cprimebinned, dv, edge_order=2, axis=axis)+cprimebinned/2
    return cor
