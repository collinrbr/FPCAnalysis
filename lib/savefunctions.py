# savefunctions.py>

#functions related making netcdf4 file to pass to MLA algo and loading netcdf4 files

import numpy as np

def savedata(CEx_out, CEy_out, vx_out, vy_out, x_out, metadata_out = [], params = {}, filename = 'dHybridRSDAtest.nc' ):
    """
    Creates netcdf4 data of normalized correlation data to send to MLA algo.

    Parameters
    ----------
    CEx_out : 3d array
        Ex correlation data created by analysis functions
    CEy_out : 3d array
        Ey correlation data created by analysis functions
    vx_out : 2d array
        velocity space grid created by analysis functions
    vy_out : 2d array
        velocity space grid created by analysis functions
    x_out : 1d array
        x slice position data created by analysis functions
    metadata_out : 1d array, optional
        meta data array with length equal to x_out and axis 3 of correlation data
        normally needs to be made by hand
    params : dict, optional
        dictionary containing global attributes relating to data.
        contains mostly physical input parameters from original simulation
    filename : str, optional
        filename of netcdf4. Should be formatted like *.nc
    """

    from netCDF4 import Dataset
    from datetime import datetime

    #normalize CEx, CEy to 1-------------------------------------------------------
    #Here we normalize to the maximum value in either CEx, CEy
    maxCval = max(np.amax(np.abs(CEx_out)),np.amax(np.abs(CEy_out)))
    CEx_out /= maxCval
    CEy_out /= maxCval

    # open a netCDF file to write
    ncout = Dataset(filename, 'w', format='NETCDF4')


    #save data in netcdf file-------------------------------------------------------
    #define attributes
    for key in params:
        setattr(ncout,key,params[key])
    ncout.description = 'dHybridR MLA data test 1'
    ncout.generationtime = str(datetime.now())

    #make dimensions that dependent data must 'match'
    ncout.createDimension('x', None)  # NONE <-> unlimited TODO: make limited if it saves memory or improves compression?
    ncout.createDimension('vx', None)
    ncout.createDimension('vy', None)

    vx_out = vx_out[0][:]
    vx = ncout.createVariable('vx','f4', ('vx',))
    vx.nvx = len(vx_out)
    vx.longname = 'v_x/v_ti'
    vx[:] = vx_out[:]

    vy_out = np.asarray([vy_out[i][0] for i in range(0,len(vy_out))])
    vy = ncout.createVariable('vy','f4', ('vy',))
    vy.nvy = len(vy_out)
    vy.longname = 'v_y/v_ti'
    vy[:] = vy_out[:]

    x = ncout.createVariable('x','f4',('x',))
    x.nx = len(x_out)
    x[:] = x_out[:]

    #tranpose data to match previous netcdf4 formatting
    for i in range(0,len(CEx_out)):
        tempCex = CEx_out[i].T
        CEx_out[i] = tempCex
        tempCey = CEy_out[i].T
        CEy_out[i] = tempCey

    C_ex = ncout.createVariable('C_Ex','f4', ('x', 'vx', 'vy'))
    C_ex.longname = 'C_{Ex}'
    C_ex[:] = CEx_out[:]

    C_ey = ncout.createVariable('C_Ey','f4', ('x', 'vx', 'vy'))
    C_ey.longname = 'C_{Ey}'
    C_ey[:] = CEy_out[:]

    metadata = ncout.createVariable('metadata','f4',('x',))
    metadata.description = '1 = signature, 0 = no signature'
    metadata[:] = metadata_out[:]

    #Save data into netcdf4 file-----------------------------------------------------
    print("Saving data into netcdf4 file")

    #save file
    ncout.close()

def load_netcdf4(filename):
    """
    Loads netcdf4 data created by savedata function

    Parameters
    ----------
    filename : str, optional
        filename of netcdf4. Should be formatted like *.nc

    Returns
    -------
    CEx_in  : 3d array
        Ex correlation data created by analysis functions
    CEy_in : 3d array
        Ey correlation data created by analysis functions
    vx_in : 2d array
        velocity space grid created by analysis functions
    vy_in : 2d array
        velocity space grid created by analysis functions
    x_in : 1d array
        x slice position data created by analysis functions
    metadata_in : 1d array, optional
        meta data array with length equal to x_out and axis 3 of correlation data
        normally needs to be made by hand
    params_in : dict, optional
        dictionary containing global attributes relating to data.
        contains mostly physical input parameters from original simulation
    """
    from netCDF4 import Dataset
    from datetime import datetime

    ncin = Dataset(flnm, 'r', format='NETCDF4')

    x_in = ncin.variables['x'][:]
    vx = ncin.variables['vx'][:]
    vy = ncin.variables['vy'][:]
    CEx_in[:] = ncin.variables['C_Ex'][:]
    CEy_in[:] = ncin.variables['C_Ey'][:]
    metadata_in[:] = ncin.variables['metadata'][:]

    #reconstruct vx, vy 2d arrays
    vy_in = []
    vx_in = []

    #load parameters in
    params_in = {}

    #test printing attributes
    print(ncin.keys())

    return CEx_in, CEy_in, vx_in, vy_in, x_in, metadata_in, params_in
