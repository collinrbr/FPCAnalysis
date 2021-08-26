# data_h5.py>

# Here we have functions related to loading dHybridR data

import glob
import numpy as np
import h5py
import os

#TODO: rename to read_dhybridr_particles
def read_particles(path, num):
    """
    Loads dHybridR particle data

    Parameters
    ----------
    path : string
        path to data folder
    num : int
        frame of data this function will load
    Returns
    -------
    pts : dict
        dictionary containing particle data
    """

    #dens_vars = 'p1 p2 p3 q tag x1 x2'.split()
    dens_vars = 'p1 p2 p3 x1 x2 x3'.split()
    pts = {}
    with h5py.File(path.format(num),'r') as f:
        for k in dens_vars:
            pts[k] = f[k][:]
    pts['Vframe_relative_to_sim'] = 0.
    return pts

#TODO: rename to read_dhybridr_box_of_par
def read_box_of_particles(path, num, x1, x2, y1, y2, z1, z2):
    """
    Loads subset of dHybridR particle data

    Reads particles within a certain spatial subset only
    Warning: this method is slow for computers with insufficient RAM

    Parameters
    ----------
    path : string
        path to data folder
    num : int
        frame of data this function will load
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

    Returns
    -------
    pts : dict
        dictionary containing particle data
    """

    #dens_vars = 'p1 p2 p3 q tag x1 x2'.split()
    dens_vars = 'p1 p2 p3 x1 x2 x3'.split()

    pts = {}
    with h5py.File(path.format(num),'r') as f:
        gptsx = (x1 < f['x1'][:] ) & (f['x1'][:] < x2)
        gptsy = (y1 < f['x2'][:] ) & (f['x2'][:] < y2)
        gptsz = (z1 < f['x3'][:] ) & (f['x3'][:] < z2)
        for k in dens_vars:
                pts[k] = f[k][gptsx & gptsy & gptsz][:]
    pts['Vframe_relative_to_sim'] = 0.
    return pts

def readTristanParticles(path, num):
    """
    Loads TRISTAN particle data
    Parameters
    ----------
    path : string
        path to data folder
    num : int
        frame of data this function will load
    Returns
    -------
    pts_elc : dict
        dictionary containing electron particle data
    pts_ion : dict
        dictionary containing ion particle data
    """

    dens_vars_elc = 'ue ve we xe ye ze'.split()
    dens_vars_ion = 'ui vi wi xi yi zi'.split()
    pts_elc = {}
    pts_ion = {}
    with h5py.File(path.format(num),'r') as f:
        for k in dens_vars_elc:
            pts_elc[k] = f[k][:]
        for l in dens_vars_ion:
            pts_ion[l] = f[l][:]
    return pts_elc, pts_ion

def makeHistFromTristanData(vmax, dv, x1, x2, y1, y2, z1, z2, dpar, species=None):
    """
    Computes distribution function from Tristan particle data
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
    species : string
        'e' or 'i' depending on whether computing distribution function from electrons or ions
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
    Hist : 3d array
        distribution function in box
    """

    #define mask that includes particles within range
    gptsparticle = (x1 < dpar['x'+species] ) & (dpar['x'+species] < x2) & (y1 < dpar['y'+species]) & (dpar['y'+species] < y2) & (z1 < dpar['z'+species]) & (dpar['z'+species] < z2)
    totalPtcl = np.sum(gptsparticle)

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
    dpar_p1 = np.asarray(dpar['u'+species][gptsparticle][:])
    dpar_p2 = np.asarray(dpar['v'+species][gptsparticle][:])
    dpar_p3 = np.asarray(dpar['w'+species][gptsparticle][:])

    #find distribution
    Hist,_ = np.histogramdd((dpar_p3,dpar_p2,dpar_p1),
                         bins=[vzbins,vybins,vxbins])

    return vx, vy, vz, totalPtcl, Hist


def build_slice(x1,x2,y1,y2,z1,z2):
    """
    Builds slice to pass to field_loader or flow_loader
    """
    pass

def field_loader(field_vars='all', components='all', num=None,
                 path='./', slc=None, verbose=False):
    """
    Loads dHybridR field data
    Parameters
    ----------
    field_vars : string
        used to specify loading only subset of fields
    components : string
        used to specify loading only subset of field components
    num : int
        used to specify which frame (i.e. time slice) is loaded
    path : string
        path to data folder
    slc : 1d array
        index tuple that specifies loading subset of fields spatially
        form of (slice(z0idx,z1idx, 1),slice(y0idx,y1idx, 1),slice(x0idx,x1idx, 1))
        where idx is an integer index
    verbose : boolean
        if true, prints debug information
    Returns
    -------
    d : dict
        dictionary containing field information and location. Ordered (z,y,x)
    """

    _field_choices_ = {'B':'Magnetic',
                       'E':'Electric',
                       'J':'CurrentDens'}
    _ivc_ = {v: k for k, v in iter(_field_choices_.items())}
    if components == 'all':
        components = 'xyz'
    if(len(path) > 0):
        if path[-1] is not '/': path = path + '/'
    fpath = path+"Output/Fields/*"
    if field_vars == 'all':
        field_vars = [c[len(fpath)-1:] for c in glob.glob(fpath)]
        field_vars = [_ivc_[k] for k in field_vars]
    else:
        if isinstance(field_vars, basestring):
            field_vars = field_vars.upper().split()
        elif not type(field_vars) in (list, tuple):
            field_vars = [field_vars]
    if slc is None:
        slc = (np.s_[:],np.s_[:],np.s_[:])
    fpath = path+"Output/Fields/{f}/{T}{c}/{v}fld_{t}.h5"
    T = '' if field_vars[0] == 'J' else 'Total/'
    test_path = fpath.format(f = _field_choices_[field_vars[0]],
                             T = T,
                             c = 'x',
                             v = field_vars[0],
                             t = '*')
    if verbose: print(test_path)
    choices = glob.glob(test_path)
    #num_of_zeros = len()
    choices = [int(c[-11:-3]) for c in choices]
    choices.sort()
    fpath = fpath.format(f='{f}', T='{T}', c='{c}', v='{v}', t='{t:08d}')
    d = {}
    # while num not in choices:
    #     _ =  'Select from the following possible movie numbers: '\
    #          '\n{0} '.format(choices)
    #     num = int(input(_))
    for k in field_vars:
        T = '' if k == 'J' else 'Total/'
        for c in components:
            ffn = fpath.format(f = _field_choices_[k],
                               T = T,
                               c = c,
                               v = k,
                               t = num)
            kc = k.lower()+c
            if verbose: print(ffn)
            with h5py.File(ffn,'r') as f:
                d[kc] = np.asarray(f['DATA'],order='F')
                d[kc] = np.ascontiguousarray(d[kc])
                _N3,_N2,_N1 = f['DATA'].shape #python is fliped.
                x1,x2,x3 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:], f['AXIS']['X3 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                dx3 = (x3[1]-x3[0])/_N3
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
                d[kc+'_zz'] = dx3*np.arange(_N3) + dx3/2. + x3[0]
                d[kc+'_xx'] = d[kc+'_xx'][slc[2]]
                d[kc+'_yy'] = d[kc+'_yy'][slc[1]]
                d[kc+'_zz'] = d[kc+'_zz'][slc[0]]
                d[kc] = d[kc][slc]
    d['Vframe_relative_to_sim'] = 0.
    return d

def all_dfield_loader(field_vars='all', components='all', num=None,
                 path='./', slc=None, verbose=False):

    """
    Function to load all fields for all available frames.

    Parameters
    ----------
    field_vars : string
        used to specify loading only subset of fields
    components : string
        used to specify loading only subset of field components
    num : int
        used to specify which frame (i.e. time slice) is loaded
    path : string
        path to data folder
    slc : 1d array
        index tuple that specifies loading subset of fields spatially
        form of [np.s_[z0idx,z1idx],np.s_[y0idx,y1idx],np.s_[x0idx,x1idx]]
    verbose : boolean
        if true, prints debug information

    Returns
    -------
    alld : dict
        dictionary containing all field information and location (for each time slice)
        Fields are Ordered (z,y,x)
        Contains key with frame number
    """

    _field_choices_ = {'B':'Magnetic',
                       'E':'Electric',
                       'J':'CurrentDens'}
    _ivc_ = {v: k for k, v in iter(_field_choices_.items())}
    if components == 'all':
        components = 'xyz'
    if path[-1] is not '/': path = path + '/'
    fpath = path+"Output/Fields/*"
    if field_vars == 'all':
        field_vars = [c[len(fpath)-1:] for c in glob.glob(fpath)]
        field_vars = [_ivc_[k] for k in field_vars]
    else:
        if isinstance(field_vars, basestring):
            field_vars = field_vars.upper().split()
        elif not type(field_vars) in (list, tuple):
            field_vars = [field_vars]
    if slc is None:
        slc = (np.s_[:],np.s_[:],np.s_[:])
    fpath = path+"Output/Fields/{f}/{T}{c}/{v}fld_{t}.h5"
    T = '' if field_vars[0] == 'J' else 'Total/'
    test_path = fpath.format(f = _field_choices_[field_vars[0]],
                             T = T,
                             c = 'x',
                             v = field_vars[0],
                             t = '*')
    if verbose: print(test_path)
    choices = glob.glob(test_path)
    #num_of_zeros = len()
    choices = [int(c[-11:-3]) for c in choices]
    choices.sort()
    fpath = fpath.format(f='{f}', T='{T}', c='{c}', v='{v}', t='{t:08d}')

#     while num not in choices:
#         _ =  'Select from the following possible movie numbers: '\
#              '\n{0} '.format(choices)
#         num = int(input(_))

    alld= {'frame':[],'dfields':[]}
    for _num in choices:
        num = int(_num)
        d = {}
        for k in field_vars:
            T = '' if k == 'J' else 'Total/'
            for c in components:
                ffn = fpath.format(f = _field_choices_[k],
                                   T = T,
                                   c = c,
                                   v = k,
                                   t = num)
                kc = k.lower()+c
                if verbose: print(ffn)
                with h5py.File(ffn,'r') as f:
                    d[kc] = np.asarray(f['DATA'],order='F')
                    d[kc] = np.ascontiguousarray(d[kc])
                    _N3,_N2,_N1 = f['DATA'].shape #python is fliped.
                    x1,x2,x3 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:], f['AXIS']['X3 AXIS'][:] #TODO: double check that x1->xx x2->yy x3->zz
                    dx1 = (x1[1]-x1[0])/_N1
                    dx2 = (x2[1]-x2[0])/_N2
                    dx3 = (x3[1]-x3[0])/_N3
                    d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                    d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
                    d[kc+'_zz'] = dx3*np.arange(_N3) + dx3/2. + x3[0]
                    d[kc+'_xx'] = d[kc+'_xx'][slc[2]]
                    d[kc+'_yy'] = d[kc+'_yy'][slc[1]]
                    d[kc+'_zz'] = d[kc+'_zz'][slc[0]]
                    d[kc] = d[kc][slc]
                d['Vframe_relative_to_sim'] = 0.
        alld['dfields'].append(d)
        alld['frame'].append(num)

    return alld

def flow_loader(flow_vars=None, num=None, path='./', sp=1, verbose=False):
    """
    Loads dHybridR flow data

    Parameters
    ----------
    field_vars : string
        used to specify loading only subset of fields
    num : int
        used to specify which frame (i.e. time slice) is loaded
    path : string
        path to data folder
    sp : int
        species number. Used to load different species
    verbose : boolean
        if true, prints debug information

    Returns
    -------
    d : dict
        dictionary containing flow information and location. Ordered (z,y,x)
    """

    import glob
    if path[-1] is not '/': path = path + '/'
    #choices = num#get_output_times(path=path, sp=sp, output_type='flow')
    dpath = path+"Output/Phase/FluidVel/Sp{sp:02d}/{dv}/Vfld_{tm:08}.h5"
    d = {}
    # while num not in choices:
    #     _ =  'Select from the following possible movie numbers: '\
    #          '\n{0} '.format(choices)
    #     num = int(input(_))
    if type(flow_vars) is str:
        flow_vars = flow_vars.split()
    elif flow_vars is None:
        flow_vars = 'x y z'.split()
    #print(dpath.format(sp=sp, tm=num))
    for k in flow_vars:
        if verbose: print(dpath.format(sp=sp, dv=k, tm=num))
        with h5py.File(dpath.format(sp=sp, dv=k, tm=num),'r') as f:
            kc = 'u'+k
            _ = f['DATA'].shape #python is fliped
            dim = len(_)
            #print(kc,_)
            d[kc] = f['DATA'][:]
            if dim < 3:
                _N2,_N1 = _
                x1,x2 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
            else:
                _N3,_N2,_N1 = _
                x1 = f['AXIS']['X1 AXIS'][:]
                x2 = f['AXIS']['X2 AXIS'][:]
                x3 = f['AXIS']['X3 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                dx3 = (x3[1]-x3[0])/_N3
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
                d[kc+'_zz'] = dx3*np.arange(_N3) + dx3/2. + x3[0]
    _id = "{}:{}:{}".format(os.path.abspath(path), num, "".join(flow_vars))
    d['id'] = _id
    d['Vframe_relative_to_sim'] = 0.
    return d

def read_restart(path):
    """
    Loads restart file
    """

    pass
