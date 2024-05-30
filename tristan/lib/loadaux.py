import glob
import numpy as np
import h5py
import os
import math
from netCDF4 import Dataset
from datetime import datetime

def read_particles_dhybridr(path, numframe=None, is2d3v = False, loaddebugsubset=False):
    """
    Loads dHybridR particle data

    Parameters
    ----------
    path : string
        when numframe = none, this is the path and filename (e.g. foo/bar/data.hdf5)
        when numframe != none, this is the path to data with place to  numframe != none (e.g. Output/Raw/Sp01/raw_sp01_{:08d}.h5)
    numframe : int, opt
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
    if(numframe != None):
        loadfilename = path.format(numframe)
    else:
        loadfilename = path
    with h5py.File(loadfilename,'r') as f:
        for k in dens_vars:
            if(loaddebugsubset):
                pts[k]=f[k][::100]
            else:
                pts[k] = f[k][:]

    pts['Vframe_relative_to_sim'] = 0.

    return pts

def field_loader_dhybridr(field_vars='all', components='all', num=None,
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

def load_params(path,num,debug=False):
    """
    WARNING: num should be a string. TODO: rename to something else
    """

    params = {}

    with h5py.File(path + 'param.' + num, 'r') as paramfl:
        if(debug): print(list(paramfl.keys()))

        params['comp'] = paramfl['c_omp'][0]
        params['c'] = paramfl['c'][0]
        params['sigma'] = paramfl['sigma'][0]
        params['istep'] = paramfl['istep'][0]
        params['massratio'] = paramfl['mi'][0]/paramfl['me'][0]
        params['mi'] = paramfl['mi'][0]
        params['me'] = paramfl['me'][0]
        params['ppc'] = paramfl['ppc0'][0]
        try:
            params['sizex'] = paramfl['sizex'][0]
        except:
            params['sizex'] = paramfl['sizey'][0]
        params['delgam'] = paramfl['delgam'][0]

    return params

def load_input(path,verbose=False):
    inputs = {}
    with open(path, 'r') as file:
        for line in file:
            if(verbose):print(line)
            line = line.strip()
            if line and '=' in line:
                key, value = line.split('=')[0],line.split('=')[1]
                key = key.strip()
                value = value.split(' ')[1]
                value = value.strip()
                try:
                    value = float(value)
                except:
                    pass
                inputs[key] = value

    return inputs

def load_fields(path_fields, num, field_vars = 'ex ey ez bx by bz', normalizeFields=False, normalizeGrid=True):
    """
    This assumes 1D implies data in the 3rd axis only, 2D implies data in the 2nd and 3rd axis only.

    """
    
    if(normalizeFields):
        field_vars += ' dens'
    field_vars = field_vars.split()
    field = {}
    field['Vframe_relative_to_sim_out'] = 0.
    
    is1D = False
    is2D = False
    is3D = False
    with h5py.File(path_fields + 'flds.tot.' + num, 'r') as fieldfl:
        for k in field_vars:
            if(fieldfl[k][:].shape[0] > 1 and fieldfl[k][:].shape[1]>1): #3D grid
                is3D = True
                field[k] = fieldfl[k][:]
            elif(fieldfl[k][:].shape[1] > 1): #2D grid
                is2D = True
                _temp = fieldfl[k][0,:,:]
                field[k] = np.zeros((2,fieldfl[k][:].shape[1],fieldfl[k][:].shape[2]))
                field[k][0,:,:] = _temp
                field[k][1,:,:] = _temp
            else: #1D grid
                is1D = True
                _temp = fieldfl[k][0,0,:]
                field[k][0,0,:] = _temp
                field[k][1,0,:] = _temp
                field[k][0,1,:] = _temp
                field[k][1,1,:] = _temp

    #Reconstruct grid
    params = load_params(path_fields,num)

    if(is1D):
        dx = params['istep']
        for key in field_vars:
            field[key+'_xx'] = np.linspace(0., field[key].shape[2]*dx, field[key].shape[2])
            field[key+'_yy'] = np.asarray([0.,1.])
            field[key+'_zz'] = np.asarray([0.,1.])

    elif(is2D):
        dx = params['istep']
        dy = dx
        for key in field_vars:
            field[key+'_xx'] = np.linspace(0., field[key].shape[2]*dx, field[key].shape[2])
            field[key+'_yy'] = np.linspace(0., field[key].shape[1]*dy, field[key].shape[1])
            field[key+'_zz'] = np.asarray([0.,1.])

    elif(is3D):
        dx = params['istep']
        dy = dx
        for key in field_vars:
            field[key+'_xx'] = np.linspace(0., field[key].shape[2]*dx, field[key].shape[2])
            field[key+'_yy'] = np.linspace(0., field[key].shape[1]*dy, field[key].shape[1])
            field[key+'_zz'] = np.linspace(0., field[key].shape[0]*dz, field[key].shape[0])

    if(normalizeGrid):
        #normalize to d_i
        comp = params['comp']
        massratio = params['mi']/params['me']
        for key in field.keys():
            if(key+'_xx' in field.keys()):
                field[key+'_xx'] /= (comp*np.sqrt(massratio))
            if(key+'_yy' in field.keys()):
                field[key+'_yy'] /= (comp*np.sqrt(massratio))
            if(key+'_zz' in field.keys()):
                field[key+'_zz'] /= (comp*np.sqrt(massratio))

    if(normalizeFields):
        if('ex' in field.keys()):
            bnorm = params['c']**2*params['sigma']/params['comp']
            sigma_ion = params['sigma']*params['me']/params['mi'] #NOTE: this is subtely differetn than what aaron's normalization is- fix it (missingn factor of gamma0 and mi+me)
            enorm = bnorm*np.sqrt(sigma_ion)*params['c'] #note, there is an extra factor of 'c hat' (c in code units, which is .45 for the main run being analyzed) that we take out
    
        if('jx' in field.keys()):
            vti0 = np.sqrt(params['delgam'])#Note: velocity is in units γV_i/c so we do not include '*params['c']'
            jnorm = vti0 #normalize to vti

        #normalize to correct units
        for key in field_vars:
            if(key[0] == 'b'):
                field[key] /= bnorm
            elif(key[0] == 'e'):
                field[key] /= enorm 
            elif(key[0] == 'j'):
                field[key] /= jnorm

    field['Vframe_relative_to_sim'] = 0. 

    return field

def load_current(path, num,normalizeFields=False):

    flow_vars = 'jx jy jz'

    return load_fields(path,num,field_vars=flow_vars,normalizeFields=normalizeFields)

def load_den(path,num,normalize=False):

    den_vars = 'dens densi'

    return load_fields(path,num,field_vars=den_vars,normalizeFields=normalize)

def load_particles(path, num, normalizeVelocity=False,loaddebugsubset=False):
    """
    Loads TRISTAN particle data

    Parameters
    ----------
    path : string
        path to data folder
    num : int
        frame of data this function will load
    normalizeVelocity : bool (opt)#TODO: rename
        normalizes velocity to v_thermal,species and position to d_i

    Returns
    -------
    pts_elc : dict
        dictionary containing electron particle data
    pts_ion : dict
        dictionary containing ion particle data
    """

    dens_vars_elc = 'ue ve we xe ye ze gammae'.split()
    dens_vars_ion = 'ui vi wi xi yi zi gammai'.split()

    pts_elc = {}
    pts_ion = {}
    with h5py.File(path + 'prtl.tot.' + num, 'r') as f:

        for k in dens_vars_elc:
            if(loaddebugsubset):
                if(len(f[k]) > 1):
                    pts_elc[k] = f[k][::25]
                else:
                    pts_elc[k] = f[k][:]
            else:
                pts_elc[k] = f[k][:] #note: velocity is in units γV_i/c
        for l in dens_vars_ion:
            if(loaddebugsubset):
                if(len(f[k]) > 1):
                    pts_ion[l] = f[l][::25]
                else:
                    pts_ion[l] = f[l][:]
            else:
                pts_ion[l] = f[l][:]

        pts_elc['inde'] = f['inde'][:]
        pts_ion['indi'] = f['indi'][:]

        pts_elc['proce'] = f['proce'][:]
        pts_ion['proci'] = f['proci'][:]

    pts_elc['Vframe_relative_to_sim'] = 0. #tracks frame (along vx) relative to sim
    pts_ion['Vframe_relative_to_sim'] = 0. #tracks frame (along vx) relative to sim

    pts_elc['q'] = -1. #tracks frame (along vx) relative to sim
    pts_ion['q'] = 1. #tracks frame (along vx) relative to sim

    if(normalizeVelocity):

        params = load_params(path,num)
        massratio = load_params(path,num)['mi']/load_params(path,num)['me']
        vti0 = np.sqrt(params['delgam'])#Note: velocity is in units γV_i/c so we do not include '*params['c']'
        vte0 = np.sqrt(params['mi']/params['me'])*vti0 #WARNING: THIS ASSUME Ti/Te = 1, TODO: don't assume Ti/Te = 1
        comp = params['comp']

        #normalize
        elc_vkeys = 'ue ve we'.split()
        ion_vkeys = 'ui vi wi'.split()
        elc_poskeys = 'xe ye ze'.split()
        ion_poskeys = 'xi yi zi'.split()
        for k in elc_vkeys:
            pts_elc[k] /= vte0
        for k in ion_vkeys:
            pts_ion[k] /= vti0
        for k in elc_poskeys:
            pts_elc[k] /= (comp*np.sqrt(massratio))
        for k in ion_poskeys:
            pts_ion[k] /= (comp*np.sqrt(massratio))

    return pts_elc, pts_ion

def project_and_save(dfields, x, vx, vy, vz, hist, corex, corey, corez, flnm):
    """
    Wrapper to project and save
    """
    
    from lib.arrayaux import array_3d_to_2d

    Histxy = []
    Histxz = []
    Histyz = []
    CExxy = []
    CExxz = []
    CExyz = []
    CEyxy = []
    CEyxz = []
    CEyyz = []
    CEzxy = []
    CEzxz = []
    CEzyz = []
    for _i in range(0,len(hist)):
        Histxy.append(array_3d_to_2d(hist[_i],'xy'))
        Histxz.append(array_3d_to_2d(hist[_i],'xz'))
        Histyz.append(array_3d_to_2d(hist[_i],'yz'))

        CExxy.append(array_3d_to_2d(corex[_i],'xy'))
        CExxz.append(array_3d_to_2d(corex[_i],'xz'))
        CExyz.append(array_3d_to_2d(corex[_i],'yz'))

        CEyxy.append(array_3d_to_2d(corey[_i],'xy'))
        CEyxz.append(array_3d_to_2d(corey[_i],'xz'))
        CEyyz.append(array_3d_to_2d(corey[_i],'yz'))

        CEzxy.append(array_3d_to_2d(corez[_i],'xy'))
        CEzxz.append(array_3d_to_2d(corez[_i],'xz'))
        CEzyz.append(array_3d_to_2d(corez[_i],'yz'))

    Histxy = np.asarray(Histxy)
    Histxz = np.asarray(Histxz)
    Histyz = np.asarray(Histyz)
    CExxy = np.asarray(CExxy)
    CExxz = np.asarray(CExxz)
    CExyz = np.asarray(CExyz)
    CEyxy = np.asarray(CEyxy)
    CEyxz = np.asarray(CEyxz)
    CEyyz = np.asarray(CEyyz)
    CEzxy = np.asarray(CEzxy)
    CEzxz = np.asarray(CEzxz)
    CEzyz = np.asarray(CEzyz)

    Vframe_relative_to_sim = dfields['Vframe_relative_to_sim']

    enerCEx = np.zeros(len(x)) #TODO: remove
    enerCEy = np.zeros(len(x))
    enerCEz = np.zeros(len(x))
    
    _save2Vdata(Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, Vframe_relative_to_sim, filename = flnm)

def _save2Vdata(Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, Vframe_relative_to_sim, num_par = [], metadata = [], params = {}, filename = 'full2Vdata.nc' ):
    """
    Saves projected data into netcdf4 file

    WARNING: DOES NOT NORMALIZE CEi

    Parameters
    ----------
    (Hist/CEi)** : 3d array
        FPC data along xx axis
        note, first axis is xx (Hist/CEi)**[xx,v*,v*]
    vx,vy,vz,x : array
        coordinate data
    enerCEi : array
        energization computed by integrating CEi
    Vframe_relative_to_sim : float
        velocity of analysis frame relative to simulation frame
    num_par : array, opt
        number of particles in each integration box
    metadata : array, opt
        sda metadata associated with each integration box
    params : dict, opt
        input parameter dictionary
    filename : str, opt
        outputfilename
    """

    from netCDF4 import Dataset
    from datetime import datetime

    # open a netCDF file to write
    ncout = Dataset(filename, 'w', format='NETCDF4')

    #save data in netcdf file-------------------------------------------------------
    #define simulation parameters
    for key in params:
        #setattr(ncout,key,params[key])
        if(not(isinstance(params[key],str))):
            _ = ncout.createVariable(key,None)
            _[:] = params[key]

    ncout.description = 'dHybridR MLA data 2V format'
    ncout.generationtime = str(datetime.now())
    ncout.version = '0'#get_git_head() #TODO: after pushing to github, reimplement

    #make dimensions that dependent data must 'match'
    ncout.createDimension('nx', None)  # NONE <-> unlimited
    ncout.createDimension('nvx', None)
    ncout.createDimension('nvy', None)
    ncout.createDimension('nvz', None)

    vx = vx[0][0][:]
    vx_out = ncout.createVariable('vx','f4', ('nvx',))
    vx_out.nvx = len(vx)
    vx_out.longname = 'v_x/v_ti'
    vx_out[:] = vx[:]

    vy = np.asarray([vy[0][i][0] for i in range(0,len(vy))])
    vy_out = ncout.createVariable('vy','f4', ('nvy',))
    vy_out.nvy = len(vy)
    vy_out.longname = 'v_y/v_ti'
    vy_out[:] = vy[:]

    vz = np.asarray([vz[i][0][0] for i in range(0,len(vz))]) #assumes same number of data points along all axis in vz_out mesh var
    vz_out = ncout.createVariable('vz','f4', ('nvz',))
    vz_out.nvz = len(vz)
    vz_out.longname = 'v_z/v_ti'
    vz_out[:] = vz[:]

    x_out = ncout.createVariable('x','f4',('nx',))
    x_out.nx = len(x)
    x_out[:] = x[:]

    C_ex_vxvy = ncout.createVariable('C_Ex_vxvy','f4',('nx','nvx','nvy'))
    C_ex_vxvy.longname = 'C_{Ex}(x;vx,vy)'
    C_ex_vxvy[:] = CExxy[:]
    C_ex_vxvz = ncout.createVariable('C_Ex_vxvz','f4',('nx','nvx','nvz'))
    C_ex_vxvz.longname = 'C_{Ex}(x;vx,vz)'
    C_ex_vxvz[:] = CExxz[:]
    C_ex_vyvz = ncout.createVariable('C_Ex_vyvz','f4',('nx','nvy','nvz'))
    C_ex_vyvz.longname = 'C_{Ex}(x;vy,vz)'
    C_ex_vyvz[:] = CExyz[:]

    C_ey_vxvy = ncout.createVariable('C_Ey_vxvy','f4',('nx','nvx','nvy'))
    C_ey_vxvy.longname = 'C_{Ey}(x;vx,vy)'
    C_ey_vxvy[:] = CEyxy[:]
    C_ey_vxvz = ncout.createVariable('C_Ey_vxvz','f4',('nx','nvx','nvz'))
    C_ey_vxvz.longname = 'C_{Ey}(x;vx,vz)'
    C_ey_vxvz[:] = CEyxz[:]
    C_ey_vyvz = ncout.createVariable('C_Ey_vyvz','f4',('nx','nvy','nvz'))
    C_ey_vyvz.longname = 'C_{Ey}(x;vy,vz)'
    C_ey_vyvz[:] = CEyyz[:]

    C_ez_vxvy = ncout.createVariable('C_Ez_vxvy','f4',('nx','nvx','nvy'))
    C_ez_vxvy.longname = 'C_{Ez}(x;vx,vy)'
    C_ez_vxvy[:] = CEzxy[:]
    C_ez_vxvz = ncout.createVariable('C_Ez_vxvz','f4',('nx','nvx','nvz'))
    C_ez_vxvz.longname = 'C_{Ez}(x;vx,vz)'
    C_ez_vxvz[:] = CEzxz[:]
    C_ez_vyvz = ncout.createVariable('C_Ez_vyvz','f4',('nx','nvy','nvz'))
    C_ez_vyvz.longname = 'C_{Ez}(x;vy,vz)'
    C_ez_vyvz[:] = CEzyz[:]

    Hist_vxvy = ncout.createVariable('Hist_vxvy','f4',('nx','nvx','nvy'))
    Hist_vxvy.longname = 'Hist(x;vx,vy)'
    Hist_vxvy[:] = Histxy[:]
    Hist_vxvz = ncout.createVariable('Hist_vxvz','f4',('nx','nvx','nvz'))
    Hist_vxvz.longname = 'Hist(x;vx,vz)'
    Hist_vxvz[:] = Histxz[:]
    Hist_vyvz = ncout.createVariable('Hist_vyvz','f4',('nx','nvy','nvz'))
    Hist_vyvz.longname = 'Hist(x;vy,vz)'
    Hist_vyvz[:] = Histyz[:]

    sda = ncout.createVariable('sda','f4',('nx',))
    sda.description = '1 = signature, 0 = no signature'
    sda[:] = metadata[:]

    enerCEx_out = ncout.createVariable('E_CEx','f4',('nx',))
    enerCEx_out.description = 'Energization computed by integrating over CEx in velocity space'
    enerCEx_out[:] = enerCEx[:]

    enerCEy_out = ncout.createVariable('E_CEy','f4',('nx',))
    enerCEy_out.description = 'Energization computed by integrating over CEy in velocity space'
    enerCEy_out[:] = enerCEy[:]

    enerCEz_out = ncout.createVariable('E_CEz','f4',('nx',))
    enerCEz_out.description = 'Energization computed by integrating over CEy in velocity space'
    enerCEz_out[:] = enerCEz[:]

    try: #TODO: cleanup
        n_par_out = ncout.createVariable('n_par','f4',('nx',))
        n_par_out.description = 'number of particles in each integration box. Note: this number might not match zeroth moment of the distribution function (i.e. integration over velocity space of hist) as particles outside the range of vmax are not counted in hist'
        n_par_out[:] = npar[:]
    except:
        pass

    Vframe_relative_to_sim_out = ncout.createVariable('Vframe_relative_to_sim', 'f4')
    Vframe_relative_to_sim_out[:] = Vframe_relative_to_sim

    #save file
    ncout.close()

def load2vdata(filename):
    """
    Loads 2v netcdf4 data created by projection script

    Parameters
    ----------
    filename : str
        filename/path of netcdf4 file

    Returns
    -------
    (Hist_vxvy, Hist_vxvz, Hist_vyvz,
       C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
       C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
       C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
       vx, vy, vz, x_in,
       enerCEx_in, enerCEy_in, enerCEz_in,
       Vframe_relative_to_sim_in, metadata_in, params_in) : 3d/1d and floats
          data from save2Vdata(). See save2Vdata
    """
    from netCDF4 import Dataset
    from datetime import datetime

    ncin = Dataset(filename, 'r', format='NETCDF4')
    ncin.set_auto_mask(False)
    npar_in = None #quick fix to fact that not all netcdf4 files have this parameter

    params_in = {}
    for key in ncin.variables.keys():
        if(key == 'x'):
            x_in = ncin.variables['x'][:]
        elif(key == 'vx'):
            vx_in = ncin.variables['vx'][:]
        elif(key == 'vy'):
            vy_in = ncin.variables['vy'][:]
        elif(key == 'vz'):
            vz_in = ncin.variables['vz'][:]
        elif(key == 'C_Ex_vxvy'):
            C_Ex_vxvy = ncin.variables['C_Ex_vxvy'][:]
        elif(key == 'C_Ex_vxvz'):
            C_Ex_vxvz = ncin.variables['C_Ex_vxvz'][:]
        elif(key == 'C_Ex_vyvz'):
            C_Ex_vyvz = ncin.variables['C_Ex_vyvz'][:]
        elif(key == 'C_Ey_vxvy'):
            C_Ey_vxvy = ncin.variables['C_Ey_vxvy'][:]
        elif(key == 'C_Ey_vxvz'):
            C_Ey_vxvz = ncin.variables['C_Ey_vxvz'][:]
        elif(key == 'C_Ey_vyvz'):
            C_Ey_vyvz = ncin.variables['C_Ey_vyvz'][:]
        elif(key == 'C_Ez_vxvy'):
            C_Ez_vxvy = ncin.variables['C_Ez_vxvy'][:]
        elif(key == 'C_Ez_vxvz'):
            C_Ez_vxvz = ncin.variables['C_Ez_vxvz'][:]
        elif(key == 'C_Ez_vyvz'):
            C_Ez_vyvz = ncin.variables['C_Ez_vyvz'][:]
        elif(key == 'Hist_vxvy'):
            Hist_vxvy = ncin.variables['Hist_vxvy'][:]
        elif(key == 'Hist_vxvz'):
            Hist_vxvz = ncin.variables['Hist_vxvz'][:]
        elif(key == 'Hist_vyvz'):
            Hist_vyvz = ncin.variables['Hist_vyvz'][:]
        elif(key == 'sda'):
            metadata_in = ncin.variables['sda'][:] #unused metadata about the local slice
        elif(key == 'E_CEx'):
            enerCEx_in = ncin.variables['E_CEx'][:]
        elif(key == 'E_CEy'):
            enerCEy_in = ncin.variables['E_CEy'][:]
        elif(key == 'E_CEz'):
            enerCEz_in = ncin.variables['E_CEz'][:]
        elif(key == 'n_par'):
            npar_in = ncin.variables['n_par'][:]
        elif(key == 'Vframe_relative_to_sim'):
            Vframe_relative_to_sim_in = ncin.variables['Vframe_relative_to_sim'][:]
        else:
            if(not(isinstance(ncin.variables[key][:], str))):
                params_in[key] = ncin.variables[key][:]

    #add global attributes
    params_in.update(ncin.__dict__)

    #reconstruct vx, vy, vz 3d arrays
    _vx = np.zeros((len(vz_in),len(vy_in),len(vx_in)))
    _vy = np.zeros((len(vz_in),len(vy_in),len(vx_in)))
    _vz = np.zeros((len(vz_in),len(vy_in),len(vx_in)))
    for i in range(0,len(vx_in)):
        for j in range(0,len(vy_in)):
            for k in range(0,len(vz_in)):
                _vx[k][j][i] = vx_in[i]

    for i in range(0,len(vx_in)):
        for j in range(0,len(vy_in)):
            for k in range(0,len(vz_in)):
                _vy[k][j][i] = vy_in[j]

    for i in range(0,len(vx_in)):
        for j in range(0,len(vy_in)):
            for k in range(0,len(vz_in)):
                _vz[k][j][i] = vz_in[k]

    vx = _vx
    vy = _vy
    vz = _vz
    try:
        if(npar_in != None):#TODO: clean this up, this does not work when npar_in has values
            return (Hist_vxvy, Hist_vxvz, Hist_vyvz,
               C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
               C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
               C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
               vx, vy, vz, x_in,
               enerCEx_in, enerCEy_in, enerCEz_in,
               npar_in, Vframe_relative_to_sim_in, metadata_in, params_in)
        else:
            return (Hist_vxvy, Hist_vxvz, Hist_vyvz,
               C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
               C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
               C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
               vx, vy, vz, x_in,
               enerCEx_in, enerCEy_in, enerCEz_in,
               Vframe_relative_to_sim_in, metadata_in, params_in)
    except:
            return (Hist_vxvy, Hist_vxvz, Hist_vyvz,
               C_Ex_vxvy, C_Ex_vxvz, C_Ex_vyvz,
               C_Ey_vxvy, C_Ey_vxvz, C_Ey_vyvz,
               C_Ez_vxvy, C_Ez_vxvz, C_Ez_vyvz,
               vx, vy, vz, x_in,
               enerCEx_in, enerCEy_in, enerCEz_in,
               npar_in, Vframe_relative_to_sim_in, metadata_in, params_in)
