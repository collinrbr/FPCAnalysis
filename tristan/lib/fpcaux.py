import numpy as np

def compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2,
                            dpar, dfields, fieldkey, directionkey, useBoxFAC = True, altcorfields = None, computeinlocalrest = False, beta = None, massratio = None, c = None):
    """
    Computes distribution function and correlation wrt to given field

    Warning: assumes fields and particles are in the same frame

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
        xx vx yy vy zz vz data dictionary from read_particles or read_box_of_particles
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of the field you want to correlate with
        ex,ey,ez,bx,by, or bz
    directionkey : str
        name of direction you want to take the derivative with respect to
        x,y,or z
        *should match the direction of the fieldkey*
    useBoxFAC : bool, opt
        if TRUE, compute field aligned coordinate system using the average value of the magnetic field across the whole 
        transverse domain. Otherwise, will use local value of each particle to compute field aligned coordinate system
    altcorfields : dict, opt
        if not None, then all calculations will be performed using dfields except for the correlation. Used when computing
        fluct correlation in field aligned coordinates

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
    
    null_for_debug = False #returns zero array, which saves a lot of time, for debug purposes
    if(null_for_debug):
        # bin into cprime(vx,vy,vz)
        vxbins = np.arange(-vmax, vmax+dv, dv)
        vx = (vxbins[1:] + vxbins[:-1])/2.
        vybins = np.arange(-vmax, vmax+dv, dv)
        vy = (vybins[1:] + vybins[:-1])/2.
        vzbins = np.arange(-vmax, vmax+dv, dv)
        vz = (vzbins[1:] + vzbins[:-1])/2.

        # make the bins 3d arrays
        _vx = np.zeros((len(vz), len(vy), len(vx)))
        _vy = np.zeros((len(vz), len(vy), len(vx)))
        _vz = np.zeros((len(vz), len(vy), len(vx)))
        for i in range(0, len(vx)):
            for j in range(0, len(vy)):
                for k in range(0, len(vz)):
                    _vx[k][j][i] = vx[i]

        for i in range(0, len(vx)):
            for j in range(0, len(vy)):
                for k in range(0, len(vz)):
                    _vy[k][j][i] = vy[j]

        for i in range(0, len(vx)):
            for j in range(0, len(vy)):
                for k in range(0, len(vz)):
                    _vz[k][j][i] = vz[k]

        vx = _vx
        vy = _vy
        vz = _vz
    
        totalPtcl = 0
        hist = np.zeros(vx.shape)
        cor = np.zeros(vx.shape)
       
        print("Warning: returning zero array for hist and cor...")
        return vx, vy, vz, totalPtcl, hist, cor

    # check input
    if(fieldkey == 'ex' or fieldkey == 'bx'):
        if(directionkey != 'x'):
            print("Warning, direction of derivative does not match field direction")
    if(fieldkey == 'ey' or fieldkey == 'by'):
        if(directionkey != 'y'):
            print("Warning, direction of derivative does not match field direction")
    if(fieldkey == 'ez' or fieldkey == 'bz'):
        if(directionkey != 'z'):
            print("Warning, direction of derivative does not match field direction")

    #change keys for backwards compat
    if('xi' in dpar.keys()):
        dpar['x1'] = dpar['xi']
        dpar['x2'] = dpar['yi']
        dpar['x3'] = dpar['zi']
        dpar['p1'] = dpar['ui']
        dpar['p2'] = dpar['vi']
        dpar['p3'] = dpar['wi']
    if('xe' in dpar.keys()):
        dpar['x1'] = dpar['xe']
        dpar['x2'] = dpar['ye']
        dpar['x3'] = dpar['ze']
        dpar['p1'] = dpar['ue']
        dpar['p2'] = dpar['ve']
        dpar['p3'] = dpar['we']

    if(z1 != None and z2 != None):
        gptsparticle = (x1 <= dpar['x1']) & (dpar['x1'] <= x2) & (y1 <= dpar['x2']) & (dpar['x2'] <= y2) & (z1 <= dpar['x3']) & (dpar['x3'] <= z2)
    else:
       gptsparticle = (x1 <= dpar['x1']) & (dpar['x1'] <= x2) & (y1 <= dpar['x2']) & (dpar['x2'] <= y2)

    print("TODO: PLEASE RENAME THE COMPUTE IN LOC FRAME ROUTINE AND ALL NC AND EVERYTHING (i made a quick change duirng a time crunch) TO COMPUTE IN DOWN STREA FRAME and don't hardcode downsream location!")

    _vxdown = None
    _vydown = None
    _vzdown = None

    if(computeinlocalrest):
        _x1 = 6.5
        _x2 = 7.5
        gptsparticle_down = (_x1 <= dpar['x1']) & (dpar['x1'] <= _x2)
        _vxdown = np.mean(dpar['p1'][gptsparticle_down][:])
        _vydown = np.mean(dpar['p2'][gptsparticle_down][:])
        _vzdown = np.mean(dpar['p3'][gptsparticle_down][:])


    if(False): #TODO: reimplement this check by fixing boosts for particles... => dfields['Vframe_relative_to_sim'] != dpar['Vframe_relative_to_sim']):
        print("ERROR: particles and fields are not in the same frame...")
        return
    else:
        dpar_p1 = np.asarray(dpar['p1'][gptsparticle][:])
        dpar_p2 = np.asarray(dpar['p2'][gptsparticle][:])
        dpar_p3 = np.asarray(dpar['p3'][gptsparticle][:])

    totalPtcl = np.sum(gptsparticle)

    # build dparticles subset using shifted particle data
    dparsubset = {
        'q': dpar['q'],
        'p1': dpar_p1,
        'p2': dpar_p2,
        'p3': dpar_p3,
        'x1': dpar['x1'][gptsparticle][:],
        'x2': dpar['x2'][gptsparticle][:],
        'x3': dpar['x3'][gptsparticle][:],
        'Vframe_relative_to_sim': dpar['Vframe_relative_to_sim']
    }

    if('q' in dpar.keys()):
        dparsubset['q'] = dpar['q']

    cprimebinned, hist, vx, vy, vz = compute_cprime_hist(dparsubset, dfields, fieldkey, vmax, dv, useBoxFAC=useBoxFAC, altcorfields=altcorfields, computeinlocalrest=computeinlocalrest, beta = beta, massratio = massratio, c = c, vxdown=_vxdown,vydown=_vydown,vzdown=_vzdown)
    del dparsubset

    cor = compute_cor_from_cprime(cprimebinned, vx, vy, vz, dv, directionkey)
    del cprimebinned

    return vx, vy, vz, totalPtcl, hist, cor

def get_3d_weights(xx, yy, zz, idxxx1, idxxx2, idxyy1, idxyy2, idxzz1, idxzz2, dfields, fieldkey):
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

    # get weights by 'volume fraction' of cell
    w1 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy)*(dfields[fieldkey+'_zz'][idxzz1]-zz))
    w2 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy)*(dfields[fieldkey+'_zz'][idxzz1]-zz))
    w3 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy)*(dfields[fieldkey+'_zz'][idxzz1]-zz))
    w4 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy)*(dfields[fieldkey+'_zz'][idxzz2]-zz))
    w5 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy)*(dfields[fieldkey+'_zz'][idxzz1]-zz))
    w6 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy)*(dfields[fieldkey+'_zz'][idxzz2]-zz))
    w7 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy)*(dfields[fieldkey+'_zz'][idxzz2]-zz))
    w8 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy)*(dfields[fieldkey+'_zz'][idxzz2]-zz))

    vol = w1+w2+w3+w4+w5+w6+w7+w8

    # if vol is still zero, try computing 2d weights. For now, we assume 2d in xx and yy.
    if(vol == 0 and dfields[fieldkey+'_zz'][idxzz1]-zz == 0 and dfields[fieldkey+'_zz'][idxzz2]-zz == 0):
        w1 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy))
        w2 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy))
        w3 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy))
        w5 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy))

        # these correspond to idxzz2 and thus are zero
        w4 = 0.
        w6 = 0.
        w7 = 0.
        w8 = 0.

        vol = w1+w2+w3+w4+w5+w6+w7+w8

    if(vol == 0.):
        print("Error in getting weights! Found a zero volume.")

    # normalize to one
    w1 /= vol
    w2 /= vol
    w3 /= vol
    w4 /= vol
    w5 /= vol
    w6 /= vol
    w7 /= vol
    w8 /= vol

    # debug (should sum to 1)
    if(False):
        print('sum of weights: ' + str(w1+w2+w3+w4+w5+w6+w7+w8))

    return w1, w2, w3, w4, w5, w6, w7, w8

def weighted_field_average(xx, yy, zz, dfields, fieldkey, changebasismatrix = None):
    """
    Wrapper function for _weighted_field_average.

    Used to correlate to fields in field aligned coordinates when relevant

    See _weighted_field_average documentation
    """

    fieldaligned_keys = ['epar','eperp1','eperp2','bpar','bperp1','bperp2']
    if(fieldkey in fieldaligned_keys):
        if(fieldkey[0] == 'e'):
            #grab vals in standard coordinates
            exval = _weighted_field_average(xx, yy, zz, dfields, 'ex')
            eyval = _weighted_field_average(xx, yy, zz, dfields, 'ey')
            ezval = _weighted_field_average(xx, yy, zz, dfields, 'ez')

            #convert to field aligned
            epar,eperp1,eperp2 = np.matmul(changebasismatrix,[exval,eyval,ezval])

            #return correct key
            if(fieldkey == 'epar'):
                return epar
            elif(fieldkey == 'eperp1'):
                return eperp1
            if(fieldkey == 'eperp2'):
                return eperp2

        elif(fieldkey[0] == 'b'):
            #grab vals in standard coordinates
            bxval = _weighted_field_average(xx, yy, zz, dfields, 'bx')
            byval = _weighted_field_average(xx, yy, zz, dfields, 'by')
            bzval = _weighted_field_average(xx, yy, zz, dfields, 'bz')

            #convert to field aligned
            bpar,bperp1,bperp2 = np.matmul(changebasismatrix,[bxval,byval,bzval])

            #return correct key
            if(fieldkey == 'bpar'):
                return bpar
            elif(fieldkey == 'bperp1'):
                return bperp1
            if(fieldkey == 'bperp2'):
                return bperp2

    else:
        return _weighted_field_average(xx, yy, zz, dfields, fieldkey)


def _weighted_field_average(xx, yy, zz, dfields, fieldkey):
    """
    Uses trilinear interpolation to estimate field value at given test location

    Assumes the sides of the box are all in either the xy, xz, or yz plane

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

    from lib.arrayaux import find_two_nearest

    idxxx1, idxxx2 = find_two_nearest(dfields[fieldkey+'_xx'],xx)
    idxyy1, idxyy2 = find_two_nearest(dfields[fieldkey+'_yy'],yy)
    idxzz1, idxzz2 = find_two_nearest(dfields[fieldkey+'_zz'],zz)

    # find weights
    w1, w2, w3, w4, w5, w6, w7, w8 = get_3d_weights(xx, yy, zz, idxxx1, idxxx2,
                                    idxyy1, idxyy2, idxzz1, idxzz2, dfields, fieldkey)

    # take average of field
    tolerance = 0.001
    if(abs(w1+w2+w3+w4+w5+w6+w7+w8-1.0) >= tolerance):
        print("Warning: sum of weights in trilinear interpolation was not close enought to 1. Value was: " + str(w1+w2+w3+w4+w5+w6+w7+w8))
    fieldaverage = w1 * dfields[fieldkey][idxzz1][idxyy1][idxxx1]
    fieldaverage += w2 * dfields[fieldkey][idxzz1][idxyy1][idxxx2]
    fieldaverage += w3 * dfields[fieldkey][idxzz1][idxyy2][idxxx1]
    fieldaverage += w4 * dfields[fieldkey][idxzz2][idxyy1][idxxx1]
    fieldaverage += w5 * dfields[fieldkey][idxzz1][idxyy2][idxxx2]
    fieldaverage += w6 * dfields[fieldkey][idxzz2][idxyy2][idxxx2]
    fieldaverage += w7 * dfields[fieldkey][idxzz2][idxyy2][idxxx1]
    fieldaverage += w8 * dfields[fieldkey][idxzz2][idxyy1][idxxx2]

    return fieldaverage


def compute_cprime_hist(dparticles, dfields, fieldkey, vmax, dv, useBoxFAC=True, altcorfields=None, computeinlocalrest=False, beta=None, massratio=None, c = None, vxdown=None,vydown=None,vzdown=None):
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
    useBoxFAC : bool, opt
        if TRUE, compute field aligned coordinate system using the average value of the magnetic field across the whole 
        transverse domain. Otherwise, will use local value of each particle to compute field aligned coordinate system
    altcorfields : dict, opt
        if not None, then all calculations will be performed using dfields except for the correlation. Used when computing
        fluct correlation in field aligned coordinates

    Returns
    -------
    cprimebinned : 3d array
        distribution function weighted by charge, particles velocity,
        and field value in integration box
    Hist : 3d array
        distribution function in box
    vx : 3d array
        vx velocity grid
    vy : 3d array
        vy velocity grid
    vz : 3d array
        vz velocity grid
    """
    from scipy.stats import binned_statistic_dd
    from lib.ftransfromaux import lorentz_transform_v
    from lib.analysisaux import convert_fluc_to_par
    from lib.analysisaux import convert_fluc_to_local_par
    import copy

    if(fieldkey == 'ex' or fieldkey == 'bx'):
        vvkey = 'p1'
    elif(fieldkey == 'ey' or fieldkey == 'by'):
        vvkey = 'p2'
    elif(fieldkey == 'ez' or fieldkey == 'bz'):
        vvkey = 'p3'
    elif(fieldkey == 'epar' or fieldkey == 'bpar'):
        vvkey = 'ppar'
    elif(fieldkey == 'eperp1' or fieldkey == 'eperp1'):
        vvkey = 'pperp1'
    elif(fieldkey == 'eperp2' or fieldkey == 'eperp2'):
        vvkey = 'pperp2'

    if(computeinlocalrest):
        #Boost to local rest if requested
        #boost particles

        dparticles['p1'][:] = np.asarray(dparticles['p1'][:]-vxdown)
        dparticles['p2'][:] = np.asarray(dparticles['p2'][:]-vydown)
        dparticles['p3'][:] = np.asarray(dparticles['p3'][:]-vzdown)


        dpkeys = copy.deepcopy(list(dparticles.keys()))
        for _dpkey in dpkeys:
            if(not(_dpkey in ['Vframe_relative_to_sim','q','p1','p2','p3','x1','x2','x3'])):
                del dparticles[_dpkey]

        #print("debug new mean 2: ", np.mean(dparticles['p1']),np.mean(dparticles['p2']),np.mean(dparticles['p3']))

        #boost fields
        #norm velocity to correct units to boost
        # computes va from vth=1, given beta, mass ratio
        # v/vths vths/vthi vthi/va = v/va
        _vths = 1. #1 bc our particles are normalized to this value
        _vthi = np.sqrt(massratio)*_vths
        _vatot = (1./np.sqrt(beta/2))*_vthi 
        vxdown = vxdown/_vatot
        vydown = vydown/_vatot
        vzdown = vzdown/_vatot

        dfkeys = copy.deepcopy(list(dfields.keys()))
        subsets = ['ex','ey','ez','bx','by','bz']
        for _dfkey in dfkeys:    
            isboxalignedkey = any(sub in _dfkey for sub in subsets)
            if(not(isboxalignedkey)):
                del dfields[_dfkey] 
            
            dfields = lorentz_transform_v(dfields, vxdown, vydown, vzdown, c)
            if(useBoxFAC):
                dfields = convert_fluc_to_par(dfields,dfields) 
            else:
                dfields =  convert_fluc_to_local_par(dfields,dfields)
        #TODO: force user to use alt fields if using local frame routine


        if(altcorfields != None):
            dfkeys = copy.deepcopy(list(altcorfields.keys()))
            for _dfkey in dfkeys:
                isboxalignedkey = any(sub in _dfkey for sub in subsets)
                if(not(isboxalignedkey)):
                    del altcorfields[_dfkey]
            altcorfields = lorentz_transform_v(altcorfields, vxdown, vydown, vzdown, c)
            if(useBoxFAC):
                altcorfields = convert_fluc_to_par(dfields,altcorfields)
            else:
                altcorfields = convert_fluc_to_local_par(dfields,altcorfields)

    altcorfields['Vframe_relative_to_sim'] = 0
    dfields['Vframe_relative_to_sim'] = 0 #TODO: reomve this key- we don't use it anymore!


    #change to field aligned basis if needed
    fieldaligned_keys = ['epar','eperp1','eperp2','bpar','bperp1','bperp2']
    if(fieldkey in fieldaligned_keys):
        #we assume particle data is passed in standard basis and would need to be converted to field aligned
        from lib.analysisaux import change_velocity_basis
        from lib.analysisaux import change_velocity_basis_local
        from lib.arrayaux import find_nearest
        from lib.analysisaux import compute_field_aligned_coord

        #get smallest box that contains all particles (assumes using full yz domain)
        xx = np.min(dparticles['x1'][:])
        _xxidx = find_nearest(dfields['ex_xx'],xx)
        _x1 = dfields['ex_xx'][_xxidx]#get xx cell edge 1
        if(dfields['ex_xx'][_xxidx] > xx):
            _x1 = dfields['ex_xx'][_xxidx-1]
        xx = np.max(dparticles['x1'][:])
        _xxidx = find_nearest(dfields['ex_xx'],xx)
        _x2 = dfields['ex_xx'][_xxidx]#get xx cell edge 2
        if(dfields['ex_xx'][_xxidx] < xx):
            _x2 = dfields['ex_xx'][_xxidx+1]

        #use whole transverse box when computing field aligned
        if(useBoxFAC):
            dparticles = change_velocity_basis(dfields,dparticles,[_x1,_x2],[dfields['ex_yy'][0],dfields['ex_yy'][-1]],[dfields['ex_zz'][0],dfields['ex_zz'][-1]]) #WARNING: we also assume field aligned coordinates uses full yz domain in weighted field average!!!

            vparbasis, vperp1basis, vperp2basis = compute_field_aligned_coord(dfields,[_x1,_x2],[dfields['ex_yy'][0],dfields['ex_yy'][-1]],[dfields['ex_zz'][0],dfields['ex_zz'][-1]])
            _ = np.asarray([vparbasis,vperp1basis,vperp2basis]).T
            changebasismatrix = np.linalg.inv(_)

        #use local coordinates for field aligned
        else:
            dparticles, changebasismatrixes = change_velocity_basis_local(dfields,dparticles)
            pass

    else:
        changebasismatrix = None

    # compute cprime for each particle (often slowest part of code)
    cprimew = np.zeros(len(dparticles['x1']))
    if('q' in dparticles.keys()):
        q = dparticles['q'] 
    else:
        q = 1.
    for i in range(0, len(dparticles['x1'])):
        if(not(useBoxFAC)):
            changebasismatrix = changebasismatrixes[i]
        if(altcorfields == None):
            fieldval = weighted_field_average(dparticles['x1'][i], dparticles['x2'][i], dparticles['x3'][i], dfields, fieldkey, changebasismatrix = changebasismatrix)
        else:
            fieldval = weighted_field_average(dparticles['x1'][i], dparticles['x2'][i], dparticles['x3'][i], altcorfields, fieldkey, changebasismatrix = changebasismatrix)
        cprimew[i] = q*dparticles[vvkey][i]*fieldval
    cprimew = np.asarray(cprimew)

    # bin into cprime(vx,vy,vz)
    vxbins = np.arange(-vmax, vmax+dv, dv)
    vx = (vxbins[1:] + vxbins[:-1])/2.
    vybins = np.arange(-vmax, vmax+dv, dv)
    vy = (vybins[1:] + vybins[:-1])/2.
    vzbins = np.arange(-vmax, vmax+dv, dv)
    vz = (vzbins[1:] + vzbins[:-1])/2.

    if(vvkey in ['p1','p2','p3']):
        hist,_ = np.histogramdd((dparticles['p3'], dparticles['p2'], dparticles['p1']), bins=[vzbins, vybins, vxbins])
        cprimebinned,_ = np.histogramdd((dparticles['p3'], dparticles['p2'], dparticles['p1']), bins=[vzbins, vybins, vxbins], weights=cprimew)
    else:
        hist,_ = np.histogramdd((dparticles['pperp2'], dparticles['pperp1'], dparticles['ppar']), bins=[vzbins, vybins, vxbins])
        cprimebinned,_ = np.histogramdd((dparticles['pperp2'], dparticles['pperp1'], dparticles['ppar']), bins=[vzbins, vybins, vxbins], weights=cprimew)
    del cprimew

    # make the bins 3d arrays
    _vx = np.zeros((len(vz), len(vy), len(vx)))
    _vy = np.zeros((len(vz), len(vy), len(vx)))
    _vz = np.zeros((len(vz), len(vy), len(vx)))
    for i in range(0, len(vx)):
        for j in range(0, len(vy)):
            for k in range(0, len(vz)):
                _vx[k][j][i] = vx[i]

    for i in range(0, len(vx)):
        for j in range(0, len(vy)):
            for k in range(0, len(vz)):
                _vy[k][j][i] = vy[j]

    for i in range(0, len(vx)):
        for j in range(0, len(vy)):
            for k in range(0, len(vz)):
                _vz[k][j][i] = vz[k]

    vx = _vx
    vy = _vy
    vz = _vz

    return cprimebinned, hist, vx, vy, vz

def compute_cor_from_cprime(cprimebinned, vx, vy, vz, dv, directionkey):
    """
    Computes correlation from cprime

    Parameters
    ----------
    cprimebinned : 3d array
        distribution function weighted by charge, particles velocity,
        and field value in integration box
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

    Returns
    -------
    cor : 3d array
        FPC data
    """
    if(directionkey == 'x'):
        axis = 2
        vv = vx
    elif(directionkey == 'y'):
        axis = 1
        vv = vy
    elif(directionkey == 'z'):
        axis = 0
        vv = vz
    elif(directionkey == 'epar'):
        axis = 2
        vv = vx
    elif(directionkey == 'eperp1'):
        axis = 1
        vv = vy
    elif(directionkey == 'eperp2'):
        axis = 0
        vv = vz

    cor = -vv/2.*np.gradient(cprimebinned, dv, edge_order=2, axis=axis) + cprimebinned / 2.
    return cor
