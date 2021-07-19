# fpc.py>

#functions related to computing FPC

import numpy as np

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

    if(dfields['Vframe_relative_to_sim'] != vshock):
        "WARNING: dfields is not in the same frame as the provided vshock"

    #shift particle data to shock frame if needed
    if(dfields['Vframe_relative_to_sim' != vshock] and dpar['Vframe_relative_to_sim'] == 0.): #TODO: use shift particles function
        dpar_p1 = np.asarray(dpar['p1'][gptsparticle][:])
        dpar_p1 -= vshock
        dpar_p2 = np.asarray(dpar['p2'][gptsparticle][:])
        dpar_p3 = np.asarray(dpar['p3'][gptsparticle][:])
    elif(dpar['Vframe_relative_to_sim'] != vshock):
        "WARNING: particles were not in simulation frame or provided vshock frame. This FPC is probably incorrect..."

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
