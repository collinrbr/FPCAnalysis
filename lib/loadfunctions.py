# loadfunctions.py>

# Here we have functions related to loading dHybridR data

import glob
import numpy as np
import h5py
import os

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
    slc : 2d array
        index tuple that specifies loading subset of fields spatially
    verbose : boolean
        if true, prints debug information
    Returns
    -------
    d : dict
        dictionary containing field information and location. Ordered (z,y,x)
    """

    if(slc != None):
        print("Warning: taking slices of field data is currently unavailable. TODO: fix")
        return {}


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
        slc = np.s_[:,:]
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
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(input(_))
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
                d[kc] = np.asarray(f['DATA'][slc],order='F')
                d[kc] = np.ascontiguousarray(d[kc])
                _N3,_N2,_N1 = f['DATA'].shape #python is fliped.
                x1,x2,x3 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:], f['AXIS']['X3 AXIS'][:] #TODO: double check that x1->xx x2->yy x3->zz
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                dx3 = (x3[1]-x3[0])/_N3
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
                d[kc+'_zz'] = dx3*np.arange(_N3) + dx3/2. + x3[0]
                d[kc+'_xx'] = d[kc+'_xx']#[slc[1]]
                d[kc+'_yy'] = d[kc+'_yy']#[slc[0]]
                d[kc+'_zz'] = d[kc+'_zz']#[slc[0]]  #TODO: check if this is correct. Dont understand the variable slc or how it's used


    return d

def all_field_loader(field_vars='all', components='all', num=None,
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
    slc : 2d array
        index tuple that specifies loading subset of fields spatially
    verbose : boolean
        if true, prints debug information
    Returns
    -------
    alld : dict
        dictionary containing all field information and location (for each time slice)
        Fields are Ordered (z,y,x)
        Contains key with frame number
    """

    if(slc != None):
        print("Warning: taking slices of field data is currently unavailable. TODO: fix")
        return {}


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
        slc = np.s_[:,:]
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
                    d[kc] = np.asarray(f['DATA'][slc],order='F')
                    d[kc] = np.ascontiguousarray(d[kc])
                    _N3,_N2,_N1 = f['DATA'].shape #python is fliped.
                    x1,x2,x3 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:], f['AXIS']['X3 AXIS'][:] #TODO: double check that x1->xx x2->yy x3->zz
                    dx1 = (x1[1]-x1[0])/_N1
                    dx2 = (x2[1]-x2[0])/_N2
                    dx3 = (x3[1]-x3[0])/_N3
                    d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                    d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
                    d[kc+'_zz'] = dx3*np.arange(_N3) + dx3/2. + x3[0]
                    d[kc+'_xx'] = d[kc+'_xx']#[slc[1]]
                    d[kc+'_yy'] = d[kc+'_yy']#[slc[0]]
                    d[kc+'_zz'] = d[kc+'_zz']#[slc[0]]  #TODO: check if this is correct. Dont understand the variable slc or how it's used
        alld['dfields'].append(d)
        alld['frame'].append(num)

    return alld

def readParticlesPosandVelocityOnly(path, num):
    """
    Loads vx, vy, xx, yy particle data only. Sometimes this is necessary to save RAM.

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
    pts = {}
    twoDdist_vars = 'p1 p2 x1 x2'.split() #only need to load velocity and space information for dist func
    with h5py.File(path.format(num),'r') as f:
        for k in twoDdist_vars:
            pts[k] = f[k][:]
    return pts

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
    return d
