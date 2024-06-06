# data_tristan.py>

# Here we have functions related to loading tristan data

import glob
import numpy as np
import h5py
import os
import math
from netCDF4 import Dataset
from datetime import datetime

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

def format_par_like_dHybridR(dpar):
    """
    Adds keys (more specifically 'pointers' so no memory is wasted) that makes the data indexable
    with the same key names as dHybridR data

    """

    keys = dpar.keys()

    if('xi' in keys):
        dpar['x1'] = dpar['xi']
    if('yi' in keys):
        dpar['x2'] = dpar['yi']
    if('zi' in keys):
        dpar['x3'] = dpar['zi']
    if('ui' in keys):
        dpar['p1'] = dpar['ui']
    if('vi' in keys):
        dpar['p2'] = dpar['vi']
    if('wi' in keys):
        dpar['p3'] = dpar['wi']

    if('xe' in keys):
        dpar['x1'] = dpar['xe']
    if('ye' in keys):
        dpar['x2'] = dpar['ye']
    if('ze' in keys):
        dpar['x3'] = dpar['ze']
    if('ue' in keys):
        dpar['p1'] = dpar['ue']
    if('ve' in keys):
        dpar['p2'] = dpar['ve']
    if('we' in keys):
        dpar['p3'] = dpar['we']

    dpar['q'] = dpar['q']

    return dpar
