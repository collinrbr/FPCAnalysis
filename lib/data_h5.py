# data_h5.py>

# Here we have functions related to loading dHybridR data

import glob
import numpy as np
import h5py
import os


def read_particles(path, numframe, is2d3v = False):
    """
    Loads dHybridR particle data
    TODO: rename to read_dhybridr_particles

    Parameters
    ----------
    path : string
        path to data folder
    numframe : int
        frame of data this function will load
    is2d3v : bool, opt
        set true is simualation is 2D 3V

    Returns
    -------
    pts : dict
        dictionary containing particle data
    """

    if(is2d3v):
        dens_vars = 'p1 p2 p3 x1 x2'.split()
    else:
        dens_vars = 'p1 p2 p3 x1 x2 x3'.split()

    pts = {}
    with h5py.File(path.format(numframe),'r') as f:
        for k in dens_vars:
            pts[k] = f[k][:]

    pts['Vframe_relative_to_sim'] = 0.

    return pts


def read_box_of_particles(path, numframe, x1, x2, y1, y2, z1, z2, is2d3v = False):
    """
    Loads subset of dHybridR particle data

    Reads particles within a certain spatial subset only
    Warning: this method is slow for computers with insufficient RAM

    #TODO: rename to read_dhybridr_box_of_par

    Parameters
    ----------
    path : string
        path to data folder
    numframe : int
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
    is2d3v : bool, opt
        set true is simualation is 2D 3V

    Returns
    -------
    pts : dict
        dictionary containing particle data
    """

    #dens_vars = 'p1 p2 p3 q tag x1 x2'.split()
    dens_vars = 'p1 p2 p3 x1 x2 x3'.split()

    pts = {}
    with h5py.File(path.format(numframe),'r') as f:
        gptsx = (x1 < f['x1'][:] ) & (f['x1'][:] < x2)
        gptsy = (y1 < f['x2'][:] ) & (f['x2'][:] < y2)
        if(not(is2d3v)):
            gptsz = (z1 < f['x3'][:] ) & (f['x3'][:] < z2)
        for k in dens_vars:
                if(not(is2d3v)):
                    pts[k] = f[k][gptsx & gptsy & gptsz][:]
                else:
                    if(k != 'x3'):
                        pts[k] = f[k][gptsx & gptsy][:]
    pts['Vframe_relative_to_sim'] = 0.

    return pts

def build_slice(x1, x2, y1, y2, z1, z2):
    """
    Builds slice to pass to field_loader or flow_loader

    Returns
    -------
    slc : 1d array
        index tuple that specifies loading subset of fields spatially
        form of (slice(z0idx,z1idx, 1),slice(y0idx,y1idx, 1),slice(x0idx,x1idx, 1))
        where idx is an integer index
    """
    pass

def field_loader(field_vars='all', components='all', num=None,
                 path='./', slc=None, verbose=False, is2d3v = False):
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
    is2d3v : bool, opt
        set true is simualation is 2D 3V

    Returns
    -------
    d : dict
        dictionary containing field information and location. Ordered (z,y,x)
    """

    _field_choices_ = {'B':'Magnetic',
                       'E':'Electric',
                       'J':'CurrentDens'}
    _ivc_ = {v : k for k, v in iter(_field_choices_.items())}
    if components == 'all':
        components = 'xyz'
    if(len(path) > 0):
        if path[-1] != '/': path = path + '/'
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
    if verbose:
        print(test_path)
    choices = glob.glob(test_path)

    choices = [int(c[-11:-3]) for c in choices]
    choices.sort()
    fpath = fpath.format(f='{f}', T='{T}', c='{c}', v='{v}', t='{t:08d}')
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
                if(is2d3v):
                    _N2,_N1 = f['DATA'].shape
                    x1,x2 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:]
                else:
                    _N3,_N2,_N1 = f['DATA'].shape
                    x1,x2,x3 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:], f['AXIS']['X3 AXIS'][:]

                dx1 = (x1[1]-x1[0])/_N1
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_xx'] = d[kc+'_xx'][slc[2]]

                dx2 = (x2[1]-x2[0])/_N2
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
                d[kc+'_yy'] = d[kc+'_yy'][slc[1]]

                if(not(is2d3v)):
                    dx3 = (x3[1]-x3[0])/_N3
                    d[kc+'_zz'] = dx3*np.arange(_N3) + dx3/2. + x3[0]
                    d[kc+'_zz'] = d[kc+'_zz'][slc[0]]

                if(not(is2d3v)):
                    d[kc] = d[kc][slc]

    d['Vframe_relative_to_sim'] = 0.

    return d

def all_dfield_loader(field_vars='all', components='all', num=None,
                 path='./', slc=None, verbose=False, is2d3v=False):

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
    is2d3v : bool, opt
        set true is simualation is 2D 3V

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
    if path[-1] != '/': path = path + '/'
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
    choices = [int(c[-11:-3]) for c in choices]
    choices.sort()

    alld= {'frame':[],'dfields':[]}
    for _num in choices:
        num = int(_num)
        d = field_loader(path=path,num=num,is2d3v=is2d3v)
        alld['dfields'].append(d)
        alld['frame'].append(num)

    return alld


def flow_loader(flow_vars=None, num=None, path='./', sp=1, verbose=False, is2d3v=False):
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
    is2d3v : bool, opt
        set true is simualation is 2D 3V

    Returns
    -------
    d : dict
        dictionary containing flow information and location. Ordered (z,y,x)
    """

    import glob
    if path[-1] != '/':
        path = path + '/'
    dpath = path+"Output/Phase/FluidVel/Sp{sp:02d}/{dv}/Vfld_{tm:08}.h5"
    d = {}
    if type(flow_vars) is str:
        flow_vars = flow_vars.split()
    elif flow_vars is None:
        flow_vars = 'x y z'.split()
    for k in flow_vars:
        if verbose: print(dpath.format(sp=sp, dv=k, tm=num))
        with h5py.File(dpath.format(sp=sp, dv=k, tm=num),'r') as f:
            kc = 'u'+k
            _ = f['DATA'].shape
            dim = len(_)
            d[kc] = f['DATA'][:]
            if is2d3v:
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


def dict_2d_to_3d(dict, axis):
    """
    Pads 2D3V data to look like 3D3V data that the rest of the pipeline can process

    Parameters
    ----------
    dict : dict
        2D3V field or flow dictionary
    axis : int
        axis to pad along

    Returns
    -------
    dict : dict
        Pseudo 3D3V data
    """
    datakeys = ['ex','ey','ez','bx','by','bz','ux','uy','uz'] #keys that might need to be padded
    dictkeys = list(dict.keys())

    for key in dictkeys:
        if key in datakeys:
            ny,nx = dict[key].shape
            if(axis == 0):
                temp = np.zeros((1,ny,nx))
                temp[0,:,:] = dict[key][:,:]
                dict[key+'_zz'] = np.asarray([0])
            if(axis == 1):
                temp = np.zeros((ny,1,nx))
                temp[:,0,:] = dict[key][:,:]
                dict[key+'_yy'] = np.asarray([0])
            if(axis == 2):
                temp = np.zeros((ny,nx,1))
                temp[:,:,0] = dict[key][:,:]
                dict[key+'_xx'] = np.asarray([0])
            dict[key] = temp

    return dict

def par_2d_to_3d(par):
    """
    Pads 2D3V data to look like 3D3V data that the rest of the pipeline can process

    Parameters
    ----------
    par : dict
        2D3V particle dictionary

    Returns
    -------
    par : dict
        Pseudo 3D3V particle dictionary
    """
    datakeys = ['x1','x2','x3']

    for key in datakeys:
        if key not in par.keys():
            num_par = len(par[list(par.keys())[0]])
            par[key] = np.zeros()

    return par


def _pts_to_par_dict(pts):
    """
    Takes pts data returned by PartMapper3D functions and makes it into particle
    dictionary used by the rest of this pipeline

    Parameters
    ----------
    pts : 2d array
        particle data from PartMapper3D

    Returns
    -------
    dpar : dict
        particle data dictionary
    """

    from copy import deepcopy

    dpar = {}

    dpar['Vframe_relative_to_sim'] = 0

    dpar['x1'] = deepcopy(pts[:,0])
    dpar['x2'] = deepcopy(pts[:,1])
    dpar['x3'] = deepcopy(pts[:,2])
    dpar['p1'] = deepcopy(pts[:,3])
    dpar['p2'] = deepcopy(pts[:,4])
    dpar['p3'] = deepcopy(pts[:,5])

    return dpar


def read_restart(path,verbose=True,xlim=None,nthreads=1):
    """
    Loads all restart files. Use's modified code from Dr. Colby Haggerty.

    (Not sure how to implement this cleanly)

    Parameters
    ---------
    path : string
        path to simulation data folder. I.e. one above the restart file folder
    verbose : bool, opt
        prints debug statements if true

    Returns
    -------
    pts : dict
        dictionary containing particle data
    """

    PM = PartMapper3D(path)
    numfiles = PM.px*PM.py*PM.pz

    if(xlim==None):
        procs = np.arange(0,numfiles) #loads entire simulation box.
                                      #Probably more intuitive to load entire
                                      #box then take subset if desired as
                                      #restart files contain subboxes that
                                      #contain multiple cells but are smaller
                                      #than the simulation box
    else:
        procs = PM.xrange_to_nums(xlim[0], xlim[1]) #it is wasteful of RAM to only
                                                    #restrict particle loading to
                                                    #the xx dimensions, however
                                                    #the results will be the same
                                                    #TODO: optimize by restricting in
                                                    #yy and zz too
    if(nthreads == 1):
        pts = PM.parts_from_num(procs[-1])
        procs = procs[:-1]
        for _c,_p in enumerate(procs):
            if(verbose):
                print(str(_p) + ' of ' + str(procs[-1]))
            _pts = PM.parts_from_num(_p)
            pts = np.concatenate([pts,_pts],axis=0)

        dpar = _pts_to_par_dict(pts)
        del pts

    else:
        from concurrent.futures import ProcessPoolExecutor

        #empty results array
        pts = [[],[],[],[],[],[]]

        tasks = procs

        #do multithreading
        with ProcessPoolExecutor(max_workers = nthreads) as executor:
            futures = []
            jobids = [] #array to track where in results array result returned by thread should go
            num_working = 0
            tasks_completed = 0
            taskidx = 0

            while(tasks_completed < len(x1task)): #while there are jobs to do
                if(num_working < max_workers and taskidx < len(x1task)): #if there is a free worker and job to do, give job
                    if(verbose):
                        print('Loading '+str(taskidx) + ' of ' + str(procs[-1]))
                    futures.append(executor.submit(PM._multi_process_part_mapper,tasks[taskidx],path))
                    jobids.append(taskidx)
                    taskidx += 1
                    num_working += 1
                else: #otherwise
                    exists_idle = False
                    nft = len(futures)
                    _i = 0
                    while(_i < nft):
                        if(futures[_i].done()): #if done get result
                            #get results and place in return vars
                            resultidx = jobids[_i]
                            _output = futures[_i].result() #return vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEx, CEy, CEz
                            pts = np.concatenate([pts,_output],axis=0)

                            if(verbose):
                                print('Loaded '+str(taskidx) + ' of ' + str(procs[-1]))

                            #update multithreading state vars
                            num_working -= 1
                            tasks_completed += 1
                            exists_idle = True
                            futures.pop(_i)
                            jobids.pop(_i)
                            nft -= 1
                            _i += 1

                    if(not(exists_idle)):
                        time.sleep(1)

    return dpar

def _multi_process_part_mapper(filenum,path):
    """

    """
    PM = PartMapper3D(path)
    _pts = PM.parts_from_num(_p)

    return _pts



class PartMapper3D(object):
    """
    Class that contains functions to loads positional and velocity data from
    restart files from a dHybridR simulation.

    Each restart file contains particle data about some sub region (subbox)
    of the simulation. Each subbox may contain many PIC cells
    """
    def __init__(self, path):
        self.path=path

        self.p = self.read_input(path)

        self.px, self.py, self.pz = self.p['node_number']
        self.nx, self.ny, self.nz = self.p['ncells']
        self.rx, self.ry, self.rz = self.p['boxsize']

        self.dx = self.rx/1./self.nx
        self.dy = self.ry/1./self.ny
        self.dz = self.rz/1./self.nz

    def read_input(self,path='./'):
        """
        Parse dHybrid input file for simulation information

        Parameters
        ----------
        path : string
            path of input file

        Returns
        -------
        param : dict
            simulation parameter dictionary
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

                        # The input has repeated section and keys for different species
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
        """
        Takes indicies associated with subbox and returns position in
        continuous simulation space

        Parameters
        ----------
        ip,jp,kp : int
            subbox indicies

        Returns
        -------
        xr,yr,zr : float
            position in continous simulation space
        """
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
            xr = dx * (npx + 1)*ip + dx/2.
        else:
            xr = dx * (Mx + npx*ip) + dx/2.

        if jp < My:
            yr = dy * (npy + 1) * jp + dy/2.
        else:
            yr = dy * (My + npy*jp) + dy/2.

        if kp < Mz:
            zr = dz * (npz + 1)*kp + dz/2.
        else:
            zr = dz * (Mz + npz*kp) + dz/2.

        return xr, yr, zr

    def xrange_to_nums(self, x0, x1):
        """
        Returns filename hash nums assosiated with subboxes in a given
        x range. Will include subboxes that are only parially in range

        Parameters
        ----------
        float : x0
            lower x bound
        float : x1
            upper x bound

        Returns
        -------
        nums : array
            has nums associated with pariticles within specified range
        """
#         i0 = np.int(np.floor(x0/self.rx*self.px))
#         i1 = np.int(np.min([np.ceil(x1/self.rx*self.px), self.px - 1]))

#         nums = np.arange(i0, i1)
#         for _ny in np.arange(1, self.py):
#             nums += np.arange(i0 + _ny*self.px, i1 + _ny*self.px)

#         return nums

        #hacky way to do it (would be a bit more efficient to fix the math
        #  in the above commented out block to work for 3d rather than 2d)
        maxnum = self.px*self.py*self.pz
        nums = []
        for n in range(0, maxnum):
            ip, jp, kp = self._num_to_index(n)
            bcx,bcy,bcz = self._box_center(ip, jp, kp)
            xxboxwidth = PM.rx / PM.px #each restart file contains a subbox
                                     #that normally contains many cells

            #This block will load restart file if it is even partially in the xrange
            if(bcx+xxboxwidth/2. >= x0 and bcx-xxboxwidth/2. <= x1):
                nums.append(n)

        return nums


    def _num_to_index(self, num):
        """
        Takes hash number related to filename and returns 3 indicies related
        to the indicies row, column, and aisle of each subbox

        Parameters
        ----------
        num : int
            hash number of filename containing subbox data

        Returns
        -------
        ip, jp, kp : int
            subbox indicies
        """

        ip = num%self.px
        jp = (num//self.px)%self.py
        kp = num//(self.px*self.py)
        return ip,jp,kp

    def _index_to_num(self, ip, jp, kp):
        """
        Takes  3 indicies related to the indicies row, column, and aisle of
        each subbox  and returns hash number related to filename

        Parameters
        ----------
        ip,jp,kp : int
            subbox indicies

        Returns
        -------
        num : int
            hash number of filename containing subbox data
        """
        num = self.px*self.py*kp + self.px*jp + ip
        return num

    def parts_from_index(self, ip, jp, kp, sp='SP01'):
        """
        Loads x1,x2,x3,p1,p2,p3 (position;velocity) data for given subbox

        Parameters
        ----------
        ip,jp,kp : int
            subbox indicies
        sp : string, opt
            species name

        Returns
        -------
        pts : 2d array
            x1,x2,x3,p1,p2,p3 data for each particle in subbox
        """
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
        """
        Loads x1,x2,x3,p1,p2,p3 (position;velocity) data for given subbox

        Parameters
        ----------
        num : int
            hash number of filename containing subbox data
        sp : string, opt
            species name

        Returns
        -------
        pts : 2d array
            x1,x2,x3,p1,p2,p3 data for each particle in subbox
        """

        ip, jp, kp = self._num_to_index(num)
        return self.parts_from_index(ip, jp, kp, sp=sp)

    def _auto_cast(self,k):
        """
        Takes an input string and tries to cast it to a real type

        Parameters
        ----------
            k : string
                A string that might be a int, float or bool
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
