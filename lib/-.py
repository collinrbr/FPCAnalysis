# loadfunctions.py>

# Here we have functions related to loading dHybridR data

#Function to load fields
def field_loader(field_vars='all', components='all', num=None,
                 path='./', slc=None, verbose=False):
    if(slc != None):
        print("Warning: taking slices of field data is currently unavailable. TODO: fix")
        return {}


    _field_choices_ = {'B':'Magnetic',
                       'E':'Electric',
                       'J':'CurrentDens'}
    _ivc_ = {v: k for k, v in iter(_field_choices_.items())}
    if components == 'all':
        components = 'xyz'
    if(len(path) > 0)
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
