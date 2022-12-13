# data_tristan.py>

# Here we have functions related to loading tristan data

import glob
import numpy as np
import h5py
import os
import math

#See https://github.com/PrincetonUniversity/tristan-mp-v2/blob/master/inputs/input.full for details about input and output

#TODO: normalize grid and particle position data to c_omp (from params)

def load_params(path,num):
    """
    WARNING: num should be a string. TODO: rename to something else
    """

    params = {}

    with h5py.File(path + 'param.' + num, 'r') as paramfl:

        #print(list(paramfl.keys())) #for some reason, have to print keys this way?

        #TODO change keynames to match previously used keynames
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
            #print("Warning: couldn't find sizex key, trying to load sizey as sizex instead...")
            params['sizex'] = paramfl['sizey'][0]
        params['delgam'] = paramfl['delgam'][0]

    return params

def load_fields(path_fields, num, field_vars = 'ex ey ez bx by bz', normalizeFields=False):
    """
    This assumes 1D implies data in the 3rd axis only, 2D implies data in the 2nd and 3rd axis only.

    """
    if(normalizeFields):
        field_vars += ' dens'
    field_vars = field_vars.split()
    field = {}
    field['Vframe_relative_to_sim_out'] = 0.
    #with h5py.File(path_fields.format(num),'r') as f:
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

    elif(is2D):
        dx = params['istep']
        dy = dx
        for key in field_vars:
            field[key+'_xx'] = np.linspace(0., field[key].shape[2]*dx, field[key].shape[2])
            field[key+'_yy'] = np.linspace(0., field[key].shape[1]*dy, field[key].shape[1])

    elif(is3D):
        dx = params['istep']
        dy = dx
        for key in field_vars:
            field[key+'_xx'] = np.linspace(0., field[key].shape[2]*dx, field[key].shape[2])
            field[key+'_yy'] = np.linspace(0., field[key].shape[1]*dy, field[key].shape[1])
            field[key+'_zz'] = np.linspace(0., field[key].shape[0]*dz, field[key].shape[0])

    #print("Debug")
    if(normalizeFields):
        #normalize to d_i
        comp = load_params(path_fields,num)['comp']
        for key in field_vars:
            if(key+'_xx' in field.keys()):
                field[key+'_xx'] /= comp
            if(key+'_yy' in field.keys()):
                field[key+'_yy'] /= comp
            if(key+'_zz' in field.keys()):
                field[key+'_zz'] /= comp

        if('ex' in field.keys()):
            bnorm = np.mean((field['bx'][:,:,-10:]**2+field['by'][:,:,-10:]**2+field['bz'][:,:,-10:]**2)**0.5)
            enorm = np.mean((field['ex'][:,:,-10:]**2+field['ey'][:,:,-10:]**2+field['ez'][:,:,-10:]**2)**0.5)

        #normalize to correct units
        for key in field_vars:
            #print("Normalizing key:",key)
            if(key[0] == 'b'):
                field[key] /= bnorm
            elif(key[0] == 'e'):
                #TODO: get this in the correct normalization!!!!
                field[key] /= enorm

                # #should normalize to v_{a,0} B0/ c-----------------------------------
                # #see Haggerty 2019
                # #we either assume vti = v_a (i.e. beta of 1) and compute vti
                # #or attempt to compute vti
                #
                # bnorm = np.mean((field['bx'][:,:,-1]**2+field['by'][:,:,-1]**2+field['bz'][:,:,-1]**2)**0.5)
                # #vti0, vte0 = load_particles(path_fields, num, normalizeVelocity=False, _getvti=True)
                # #v0norm = vti0 #assumes plasma beta of 1 !!!!
                #
                # dennorm = field['dens'][0,0,:][np.nonzero(field['dens'][0,0,:])][-10]#den array is weird. Its zero towards the end and its first couple of nonzero vals towards the end is small
                # v0norm = bnorm/np.sqrt(4.*np.pi*dennorm*params['mi'])
                # cnorm = params['c']
                # print('norm')
                # print(bnorm,v0norm,cnorm,bnorm*v0norm/cnorm)
                # field[key] /= bnorm*v0norm/cnorm

    field['Vframe_relative_to_sim'] = 0. #TODO: fix this, it is incorrect at least some of the time as data is reported in the upstream frame (frame where v_x,up = 0) at least some of the time

    return field

def load_current(path, num,normalizeFields=False):

    flow_vars = 'jx jy jz'

    return load_fields(path,num,field_vars=flow_vars,normalizeFields=normalizeFields)

def load_den(path,num):

    den_vars = 'dens densi'

    return load_fields(path,num,field_vars=den_vars)

def load_particles(path, num, normalizeVelocity=False, _getvti=False):
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

    dens_vars_elc = 'ue ve we xe ye ze gammae'.split()
    dens_vars_ion = 'ui vi wi xi yi zi gammai'.split()

    pts_elc = {}
    pts_ion = {}
    with h5py.File(path + 'prtl.tot.' + num, 'r') as f:
        for k in dens_vars_elc:
            pts_elc[k] = f[k][:] #note: velocity is in units γV_i/c
        for l in dens_vars_ion:
            pts_ion[l] = f[l][:]
    pts_elc['Vframe_relative_to_sim'] = 0. #tracks frame (along vx) relative to sim
    pts_ion['Vframe_relative_to_sim'] = 0. #tracks frame (along vx) relative to sim

    pts_elc['q'] = -1. #tracks frame (along vx) relative to sim
    pts_ion['q'] = 1. #tracks frame (along vx) relative to sim

    if(normalizeVelocity or _getvti):

        params = load_params(path,num)

        vti0 = math.sqrt(params['delgam'])#Note: velocity is in units γV_i/c so we do not include '*params['c']'
        vte0 = math.sqrt(params['mi']/params['me'])*vti0

        comp = params['comp']

        if(_getvti):#TODO: this isnt the cleanest way of doing this, clean this up and find a differnt way to do this
            return vti0, vte0

        #print("Computed vti0 of ", vti0, " and vte0 of ", vte0)

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
            pts_elc[k] /= comp
        for k in ion_poskeys:
            pts_ion[k] /= comp

    return pts_elc, pts_ion

def format_dict_like_dHybridR(ddict):
    """

    """
    keys = ddict.keys()

    if('jx' in keys):
        ddict['ux'] = ddict['jx']
    if('jy' in keys):
        ddict['uy'] = ddict['jy']
    if('jz' in keys):
        ddict['uz'] = ddict['jz']
    if('jx_xx' in keys):
        ddict['ux_xx'] = ddict['jx_xx']
    if('jx_yy' in keys):
        ddict['ux_yy'] = ddict['jx_yy']
    if('jx_zz' in keys):
        ddict['ux_zz'] = ddict['jx_zz']
    if('jy_xx' in keys):
        ddict['uy_xx'] = ddict['jy_xx']
    if('jy_yy' in keys):
        ddict['uy_yy'] = ddict['jy_yy']
    if('jy_zz' in keys):
        ddict['uy_zz'] = ddict['jy_zz']
    if('jz_xx' in keys):
        ddict['uz_xx'] = ddict['jz_xx']
    if('jz_yy' in keys):
        ddict['uz_yy'] = ddict['jz_yy']
    if('jz_zz' in keys):
        ddict['uz_zz'] = ddict['jz_zz']

    if('dens' in keys):
        ddict['den'] = ddict['dens']
    if('dens_xx' in keys):
        ddict['den_xx'] = ddict['dens_xx']
    if('dens_yy' in keys):
        ddict['den_yy'] = ddict['dens_yy']
    if('dens_zz' in keys):
        ddict['den_zz'] = ddict['dens_zz']

    return ddict

def format_par_like_dHybridR(dpar):
    """
    Adds keys (more specifically 'pointers' so no memory is wasted) that makes the data indexable
    with the same key names as dHybridR data

    #TODO: double check that these are 'pointers'
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
