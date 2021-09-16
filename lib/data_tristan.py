# data_tristan.py>

# Here we have functions related to loading tristan data

import glob
import numpy as np
import h5py
import os

#See https://github.com/PrincetonUniversity/tristan-mp-v2/blob/master/inputs/input.full for details about input and output

def load_fields(path_fields):
    """

    """
    field_vars = 'ex ey ez bx by bz'.split()
    field = {}
    with h5py.File(path_fields.format(7),'r') as f:
        for k in field_vars:
            field[k] = f[k][:]

    # mx probably relates to size of box. Can use to reproduce grid
    #**** TODO: move tristan functions from data_h5 to here****

    return field

def estimate_grid_setup(dfields, dparticles_ion):
    """
    Estimates the setup of the box using particle data.
    Assumes there exists one particle near each boundary of the box and that the box is of integer size

    (Original toy data was missing simulation box setup so we use this function to estimate it for now)
    """

    from copy import copy

    x1, x2, y1, y2, z1, z2 = 0., 0., 0., 0., 0., 0.

    x1 = round(max(dparticles_ion['xi']))
    x2 = round(min(dparticles_ion['xi']))
    y1 = round(max(dparticles_ion['yi']))
    y2 = round(min(dparticles_ion['yi']))
    z1 = round(max(dparticles_ion['zi']))
    z2 = round(min(dparticles_ion['zi']))

    nz,ny,nx = dfields['bz'].shape

    dx = (x2-x1)/(nx)
    dy = (y2-y1)/(ny)
    dz = (z2-z1)/(nz)

    _xx = [dx*float(i)+dx/2. for i in range(0,nx)]
    _yy = [dy*float(i)+dy/2. for i in range(0,ny)]
    _zz = [dz*float(i)+dz/2. for i in range(0,nz)]

    for key in dfields.keys:
        dfields[key+'_xx'] = copy(_xx)
        dfields[key+'_yy'] = copy(_yy)
        dfields[key+'_zz'] = copy(_zz)

    return dfields

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
    pts_elc['Vframe_relative_to_sim_out'] = 0. #tracks frame (along vx) relative to sim
    pts_ion['Vframe_relative_to_sim_out'] = 0. #tracks frame (along vx) relative to sim
    with h5py.File(path.format(num),'r') as f:
        for k in dens_vars_elc:
            pts_elc[k] = f[k][:]
        for l in dens_vars_ion:
            pts_ion[l] = f[l][:]
    return pts_elc, pts_ion






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
