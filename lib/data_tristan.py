# data_tristan.py>

# Here we have functions related to loading tristan data

import glob
import numpy as np
import h5py
import os

#See https://github.com/PrincetonUniversity/tristan-mp-v2/blob/master/inputs/input.full for details about input and output

def load_params(path,num):
    """
    WARNING: num should be a string. TODO: rename to something else
    """

    params = {}

    with h5py.File(path + 'param.' + num, 'r') as paramfl:

        #TODO change keynames to match previously used keynames
        params['comp'] = paramfl['c_omp'][0]
        params['c'] = paramfl['c'][0]
        params['sigma'] = paramfl['sigma'][0]
        params['istep'] = paramfl['istep'][0]
        params['massratio'] = paramfl['mi'][0]/paramfl['me'][0]
        params['ppc'] = paramfl['ppc0'][0]
        params['sizex'] = paramfl['sizex'][0]

    return params

def load_fields(path_fields, num, field_vars = 'ex ey ez bx by bz'):
    """
    This assumes 1D implies data in the 3rd axis only, 2D implies data in the 2nd and 3rd axis only.
    """
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
            field[key+'_xx'] = np.linspace(0., field[key].shape[0]*dx, field[key].shape[0])

    elif(is2D):
        dx = params['istep']
        dy = dx
        for key in field_vars:
            field[key+'_xx'] = np.linspace(0., field[key].shape[2]*dx, field[key].shape[2])
            field[key+'_yy'] = np.linspace(0., field[key].shape[1]*dy, field[key].shape[1])

    field['Vframe_relative_to_sim'] = 0.

    return field

def load_flow(path, num):

    flow_vars = 'jx jy jz'

    return load_fields(path,num,field_vars=flow_vars)

def load_den(path,num):

    den_vars = 'dens densi'

    return load_fields(path,num,field_vars=den_vars)

def load_particles(path, num,  normalizeVelocity=True):
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

    if(normalizeVelocity):
        from lib.analysis import compute_vrms

        print("Attempting to compute thermal velocity in the far upstream region to normalize velocity...")
        print("Please see load_particles in data_tristan.py for assumptions...")

        pts_elc = format_par_like_dHybridR(pts_elc)
        pts_ion = format_par_like_dHybridR(pts_ion)

        #comupte vti and vte in the far upstream region
        vmax = 20.*np.mean(np.abs(pts_ion['p1'])) #note: we only consider particles with v_i <= vmax when computing v_rms
        dv = vmax/50.
        upperxbound = np.max(pts_ion['x1'])
        lowerxbound = upperxbound*.95 #WARNING: assumes that the beam is undisturbed in this region
        lowerybound = np.min(pts_ion['x2'])
        upperybound = np.max(pts_ion['x2'])
        lowerzbound = np.min(pts_ion['x3'])
        upperzbound = np.max(pts_ion['x3'])
        vti0 = compute_vrms(pts_ion,vmax,dv,lowerxbound,upperxbound,lowerybound,upperybound,lowerzbound,upperzbound)
        vte0 = compute_vrms(pts_elc,vmax,dv,lowerxbound,upperxbound,lowerybound,upperybound,lowerzbound,upperzbound)
        vti0 = np.sqrt(vti0)
        vte0 = np.sqrt(vte0)
        print("Computed vti0 of ", vti0, " and vte0 of ", vte0)

        #normalize
        elc_vkeys = 'ue ve we'.split()
        ion_vkeys = 'ui vi wi'.split()
        c = load_params(path,num)['c']
        print('')
        for k in elc_vkeys:
            pts_elc[k] = pts_elc[k]*c/(vte0*pts_elc['gammae']) #note: velocity is in units γV_i/c in original output
        for k in ion_vkeys:
            pts_ion[k] = pts_ion[k]*c/(vti0*pts_ion['gammai']) #note: velocity is in units γV_i/c in original output

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
        dpar['x3'] = dpar['xi']
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
        dpar['x3'] = dpar['xe']
    if('ue' in keys):
        dpar['p1'] = dpar['ue']
    if('ve' in keys):
        dpar['p2'] = dpar['ve']
    if('we' in keys):
        dpar['p3'] = dpar['we']

    return dpar

# def estimate_grid_setup(dfields, dparticles_ion):
#     """
#     Estimates the setup of the box using particle data.
#     Assumes there exists one particle near each boundary of the box and that the box is of integer size
#
#     (Original toy data was missing simulation box setup so we use this function to estimate it for now)
#     """
#
#     from copy import copy
#
#     x1, x2, y1, y2, z1, z2 = 0., 0., 0., 0., 0., 0.
#
#     x1 = round(min(dparticles_ion['xi']))
#     x2 = round(max(dparticles_ion['xi']))
#     y1 = round(min(dparticles_ion['yi']))
#     y2 = round(max(dparticles_ion['yi']))
#     z1 = round(min(dparticles_ion['zi']))
#     z2 = round(max(dparticles_ion['zi']))
#
#     nz,ny,nx = dfields['bz'].shape
#
#     dx = (x2-x1)/(nx)
#     dy = (y2-y1)/(ny)
#     dz = (z2-z1)/(nz)
#
#     _xx = [dx*float(i)+dx/2. for i in range(0,nx)]
#     _yy = [dy*float(i)+dy/2. for i in range(0,ny)]
#     _zz = [dz*float(i)+dz/2. for i in range(0,nz)]
#
#     keys = ['ex','ey','ez','bx','by','bz']
#     for key in keys:
#         dfields[key+'_xx'] = copy(_xx)
#         dfields[key+'_xx'] = np.asarray(dfields[key+'_xx'])
#         dfields[key+'_yy'] = copy(_yy)
#         dfields[key+'_yy'] = np.asarray(dfields[key+'_yy'])
#         dfields[key+'_zz'] = copy(_zz)
#         dfields[key+'_zz'] = np.asarray(dfields[key+'_zz'])
#
#     return dfields


#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------


#if we need to save ram, should probably make these functions 3
def take_subset_particles(x1, x2, y1, y2, z1, z2, dpar, species=None):
    pass
def save_subset(Vframe_relative_to_sim_out, x1_out, x2_out, y1_out, y2_out, z1_out, z2_out, params = {}, filename = 'tristanHist.nc'):
    pass
def load_subset():
    pass

# Note, it is typically a bad idea to 'prebin' data before computing correlation as we lose spatial information of each particle
# Instead, we must assume each particle is located at the average position of the spatial bin and use the average field value in that bin
# Overall, we lose the smaller scale field fluctuations in our velocity signature

# def makeHistFromTristanData(vmax, dv, x1, x2, y1, y2, z1, z2, dpar, species=None):
#     """
#     Computes distribution function from Tristan particle data
#     Parameters
#     ----------
#     vmax : float
#         specifies signature domain in velocity space
#         (assumes square and centered about zero)
#     dv : float
#         velocity space grid spacing
#         (assumes square)
#     x1 : float
#         lower x bound
#     x2 : float
#         upper x bound
#     y1 : float
#         lower y bound
#     y2 : float
#         upper y bound
#     z1 : float
#         lower y bound
#     z2 : float
#         upper y bound
#     dpar : dict
#         xx vx yy vy zz vz data dictionary from read_particles or read_box_of_particles
#     species : string
#         'e' or 'i' depending on whether computing distribution function from electrons or ions
#
#     Returns
#     -------
#     vx : 3d array
#         vx velocity grid
#     vy : 3d array
#         vy velocity grid
#     vz : 3d array
#         vz velocity grid
#     totalPtcl : float
#         total number of particles in the correlation box
#     Hist : 3d array
#         distribution function in box
#     """
#
#     #define mask that includes particles within range
#     gptsparticle = (x1 < dpar['x'+species] ) & (dpar['x'+species] < x2) & (y1 < dpar['y'+species]) & (dpar['y'+species] < y2) & (z1 < dpar['z'+species]) & (dpar['z'+species] < z2)
#     totalPtcl = np.sum(gptsparticle)
#
#     #make bins
#     vxbins = np.arange(-vmax-dv, vmax+dv, dv)
#     vx = (vxbins[1:] + vxbins[:-1])/2.
#     vybins = np.arange(-vmax-dv, vmax+dv, dv)
#     vy = (vybins[1:] + vybins[:-1])/2.
#     vzbins = np.arange(-vmax-dv, vmax+dv, dv)
#     vz = (vzbins[1:] + vzbins[:-1])/2.
#
#     #make the bins 3d arrays
#     _vx = np.zeros((len(vz),len(vy),len(vx)))
#     _vy = np.zeros((len(vz),len(vy),len(vx)))
#     _vz = np.zeros((len(vz),len(vy),len(vx)))
#     for i in range(0,len(vx)):
#         for j in range(0,len(vy)):
#             for k in range(0,len(vz)):
#                 _vx[k][j][i] = vx[i]
#
#     for i in range(0,len(vx)):
#         for j in range(0,len(vy)):
#             for k in range(0,len(vz)):
#                 _vy[k][j][i] = vy[j]
#
#     for i in range(0,len(vx)):
#         for j in range(0,len(vy)):
#             for k in range(0,len(vz)):
#                 _vz[k][j][i] = vz[k]
#
#     vx = _vx
#     vy = _vy
#     vz = _vz
#
#     #shift particle data to shock frame
#     dpar_p1 = np.asarray(dpar['u'+species][gptsparticle][:])
#     dpar_p2 = np.asarray(dpar['v'+species][gptsparticle][:])
#     dpar_p3 = np.asarray(dpar['w'+species][gptsparticle][:])
#
#     #find distribution
#     Hist,_ = np.histogramdd((dpar_p3,dpar_p2,dpar_p1),
#                          bins=[vzbins,vybins,vxbins])
#
#     return vx, vy, vz, totalPtcl, Hist
#
# def save_hist(Hist_out, Vframe_relative_to_sim_out, x1_out, x2_out, y1_out, y2_out, z1_out, z2_out, params = {}, filename = 'tristanHist.nc'):
#     """
#     Saves histogram in netcdf4 file for reading later
#     """
#     #TODO: save params in this file
#
#     from netCDF4 import Dataset
#     from datetime import datetime
#     from lib.net_cdf4 import get_git_head
#
#     # open a netCDF file to write
#     ncout = Dataset(filename, 'w', format='NETCDF4')
#
#     ncout.description = 'Tristan distribution data in some specified domain'
#     ncout.generationtime = str(datetime.now())
#     ncout.version = get_git_head()
#
#     ncout.createDimension('nvx', None)
#     ncout.createDimension('nvy', None)
#     ncout.createDimension('nvz', None)
#
#     vx_out = vx_out[0][0][:]
#     vx = ncout.createVariable('vx','f4', ('nvx',))
#     vx.nvx = len(vx_out)
#     vx.longname = 'v_x/v_ti'
#     vx[:] = vx_out[:]
#
#     vy_out = np.asarray([vy_out[0][i][0] for i in range(0,len(vy_out))])
#     vy = ncout.createVariable('vy','f4', ('nvy',))
#     vy.nvy = len(vy_out)
#     vy.longname = 'v_y/v_ti'
#     vy[:] = vy_out[:]
#
#     vz_out = np.asarray([vz_out[i][0][0] for i in range(0,len(vz_out))]) #assumes same number of data points along all axis in vz_out mesh var
#     vz = ncout.createVariable('vz','f4', ('nvz',))
#     vz.nvz = len(vz_out)
#     vz.longname = 'v_z/v_ti'
#     vz[:] = vz_out[:]
#
#     Hist = ncout.createVariable('Hist','f4', ('nvz', 'nvy', 'nvx'))
#     Hist.longname = 'Hist'
#     Hist[:] = Hist_out[:]
#
#     Vframe_relative_to_sim = ncout.createVariable('Vframe_relative_to_sim', 'f4')
#     Vframe_relative_to_sim[:] = Vframe_relative_to_sim_out
#
#     x1 = ncout.createVariable('x1', 'f4')
#     x1[:] = x1_out
#
#     x2 = ncout.createVariable('x2', 'f4')
#     x2[:] = x2_out
#
#     y1 = ncout.createVariable('y1', 'f4')
#     y1[:] = y1_out
#
#     y2 = ncout.createVariable('y2', 'f4')
#     y2[:] = y2_out
#
#     z1 = ncout.createVariable('y1', 'f4')
#     z1[:] = z1_out
#
#     z2 = ncout.createVariable('z2', 'f4')
#     z2[:] = z2_out
#
#     #save file
#     ncout.close()
#
# def load_hist(filename):
#     pass
