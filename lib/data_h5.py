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
    Loads all restart files. Use's modified code from Dr. Colby Haggerty.

    TODO?: maybe add feature that limits domain (use xrange to nums feature)

    (Not sure how to implement this cleanly)

    Parameters
    ---------
    path : string
        path to simulation data folder. I.e. one above the restart file folder

    Returns
    -------
        pts : dict
            dictionary containing particle data
    """

    class PartMapper3D(object):
        def __init__(self, path):
            self.path=path

            self.p = self.read_input(path)

            self.px,self.py,self.pz = self.p['node_number']
            self.nx,self.ny,self.nz = self.p['ncells']
            self.rx,self.ry,self.rz = self.p['boxsize']

            self.dx = self.rx/1./self.nx
            self.dy = self.ry/1./self.ny
            self.dz = self.rz/1./self.nz

        def read_input(self,path='./'):
            """Parse dHybrid input file for simulation information

            Args:
                path (str): path of input file
            """
            import os

            path = os.path.join(path, "input/input")
            inputs = {}
            repeated_sections = {}
            # Load in all of the input stuff
            with open(path) as f:
                in_bracs = False
                for line in f:
                    # Clean up string
                    line = line.strip()

                    # Remove comment '!'
                    trim_bang = line.find('!')
                    if trim_bang > -1:
                        line = line[:trim_bang].strip()

                    # Is the line not empty?
                    if line:
                        if not in_bracs:
                            in_bracs = True
                            current_key = line

                            # The input has repeated section and keys for differnt species
                            # This section tries to deal with that
                            sp_counter = 1
                            while current_key in inputs:
                                inputs[current_key+"_01"] = inputs[current_key]
                                sp_counter += 1
                                current_key = "{}_{:02d}".format(line, sp_counter)
                                repeated_sections[current_key] = sp_counter

                            inputs[current_key] = []

                        else:
                            if line == '{':
                                continue
                            elif line == '}':
                                in_bracs = False
                            else:
                                inputs[current_key].append(line)

            # Parse the input and cast it into usefull types
            param = {}
            repeated_keys = {}
            for key,inp in inputs.items():
                for sp in inp:
                    k = sp.split('=')
                    k,v = [v.strip(' , ') for v in k]

                    _fk = k.find('(')
                    if _fk > 0:
                        k = k[:_fk]

                    if k in param:
                        param["{}_{}".format(k, key)] = param[k]
                        k = "{}_{}".format(k, key)

                    param[k] = [self._auto_cast(c.strip()) for c in v.split(',')]

                    if len(param[k]) == 1:
                        param[k] = param[k][0]

            return param

        def _box_center(self, ip, jp, kp):
            dx = self.dx
            dy = self.dy
            dz = self.dz

            npx = self.nx//self.px
            Mx = (self.nx/1./self.px - npx)*self.px

            npy = self.ny//self.py
            My = (self.ny/1./self.py - npy)*self.py

            npz = self.nz//self.pz
            Mz = (self.nz/1./self.pz - npz)*self.pz

            if ip < Mx:
                xr = dx*(npx + 1)*ip + dx/2.
            else:
                xr = dx*(Mx + npx*ip) + dx/2.

            if jp < My:
                yr = dy*(npy + 1)*jp + dy/2.
            else:
                yr = dy*(My + npy*jp) + dy/2.

            if kp < Mz:
                zr = dz*(npz + 1)*kp + dz/2.
            else:
                zr = dz*(Mz + npz*kp) + dz/2.

            return xr,yr,zr

        def xrange_to_nums(self, x0, x1):
    #         i0 = np.int(np.floor(x0/self.rx*self.px))
    #         i1 = np.int(np.min([np.ceil(x1/self.rx*self.px), self.px - 1]))

    #         nums = np.arange(i0, i1)
    #         for _ny in np.arange(1, self.py):
    #             nums += np.arange(i0 + _ny*self.px, i1 + _ny*self.px)

    #         return nums

            #hacky way to do it (would be a bit more efficient to fix the math in the above commented out block)
            maxnum = self.px*self.py*self.pz
            nums = []
            for n in range(0,maxnum):
                ip,jp,kp = self._num_to_index(n)
                bcx,bcy,bcz = self._box_center(ip, jp, kp)
                xxboxwidth = PM.rx/PM.px #each restart file contains a subbox that normally contains many cells

                if(bcx-xxboxwidth/2. >= x0 and bcx+xxboxwidth/2. <= x1):
                    nums.append(n)

            return nums


        def _num_to_index(self, num):
            ip = num%self.px
            jp = (num//self.px)%self.py
            kp = num//(self.px*self.py)
            return ip,jp,kp

        def _index_to_num(self, ip, jp, kp):
            num = self.px*self.py*kp + self.px*jp + ip
            return num

        def parts_from_index(self, ip, jp, kp, sp='SP01'):
            fname = self.path+'/Restart/Rest_proc{:05d}.h5'
            num = self._index_to_num(ip, jp, kp)
            bcx,bcy,bcz = self._box_center(ip, jp, kp)
            dx,dy,dz = self.dx,self.dy,self.dz

            with h5py.File(fname.format(num),'r') as f:

                pts = f[sp][:]
                ind = f[sp+'INDEX'][:]
                pts[:, 0] = pts[:,0] + bcx + dx*(ind[:,0] - 4)
                pts[:, 1] = pts[:,1] + bcy + dy*(ind[:,1] - 4)
                pts[:, 2] = pts[:,2] + bcz + dz*(ind[:,2] - 4)

            return pts

        def parts_from_num(self, num, sp='SP01'):
            ip, jp, kp = self._num_to_index(num)
            return self.parts_from_index(ip, jp, kp, sp=sp)

        def _auto_cast(self,k):
            """Takes an input string and tries to cast it to a real type

            Args:
                k (str): A string that might be a int, float or bool
            """

            k = k.replace('"','').replace("'",'')

            for try_type in [int, float]:
                try:
                    return try_type(k)
                except:
                    continue

            if k == '.true.':
                return True
            if k == '.false.':
                return False

            return str(k)

    PM = PartMapper3D(path)
    numfiles = PM.px*PM.py*PM.pz

    procs = np.arange(0,numfiles) #loads entire simulation box. Probably more intuitive to load entire box then take subset if desired as restart files contain subboxes that contain multiple cells but are smaller than the simulation box

    pts = PM.parts_from_num(procs[-1])
    procs = procs[:-1]
    for _c,_p in enumerate(procs):
        _pts = PM.parts_from_num(_p)
        pts = np.concatenate([pts,_pts],axis=0)

    return pts
