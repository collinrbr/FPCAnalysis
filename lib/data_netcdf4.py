# data_netcdf4.py>

#functions related making and loading netcdf4 file to pass to MLA algo and loading netcdf4 files

import numpy as np
import math

#TODO: make num_par not optional
def save3Vdata(Hist_out, CEx_out, CEy_out, CEz_out, vx_out, vy_out, vz_out, x_out, enerCEx_out, enerCEy_out, enerCEz_out, Vframe_relative_to_sim_out, num_par = [], metadata_out = [], params = {}, filename = 'full3Vdata.nc' ):
    """
    Creates netcdf4 data of normalized correlation data to send to MLA algo.

    Parameters
    ----------
    Hist_out : 4d array
        Hist created by fpc correlation functions
    CEx_out : 4d array
        Ex correlation data created by fpc correlation functions
    CEy_out : 4d array
        Ey correlation data created by fpc correlation functions
    CEz_out : 4d array
        Ez correlation data created by fpc correlation functions
    vx_out : 3d array
        velocity space grid created by fpc correlation functions
    vy_out : 3d array
        velocity space grid created by fpc correlation functions
    vz_out : 3d array
        velocity space grid created by fpc correlation functions
    x_out : 1d array
        x slice position data created by fpc correlation functions
    enerCEx_out : 1d array
        integral over velocity space of CEx
    enerCEy_out : 1d array
        integral over velocity space of CEy
    enerCEz_out : 1d array
        integral over velocity space of CEy
    num_par_out : 1d array
        number of particles in box
    metadata_out : 1d array, optional
        meta data array with length equal to x_out and axis 3 of correlation data
        normally needs to be made by hand
    params : dict, optional
        dictionary containing parameters relating to data/ simulation input.
        contains mostly physical input parameters from original simulation
    filename : str, optional
        filename of netcdf4. Should be formatted like *.nc
    """

    from netCDF4 import Dataset
    from datetime import datetime

    #normalize CEx, CEy to 1-------------------------------------------------------
    #Here we normalize to the maximum value in either CEx, CEy, CEz
    maxCval = max(np.amax(np.abs(CEx_out)),np.amax(np.abs(CEy_out)))
    maxCval = max(maxCval,np.amax(np.abs(CEz_out)))
    CEx_out /= maxCval
    CEy_out /= maxCval
    CEz_out /= maxCval

    #save data in netcdf file-------------------------------------------------------
    # open a netCDF file to write
    ncout = Dataset(filename, 'w', format='NETCDF4')

    #define simulation parameters
    for key in params:
        #setattr(ncout,key,params[key])
        if(not(isinstance(params[key],str))):
            _ = ncout.createVariable(key,None)
            _[:] = params[key]

    ncout.description = 'dHybridR MLA data'
    ncout.generationtime = str(datetime.now())
    ncout.version = get_git_head()

    #make dimensions that dependent data must 'match'
    ncout.createDimension('nx', None)  # NONE <-> unlimited TODO: make limited if it saves memory or improves compression?
    ncout.createDimension('nvx', None)
    ncout.createDimension('nvy', None)
    ncout.createDimension('nvz', None)

    vx_out = vx_out[0][0][:]
    vx = ncout.createVariable('vx','f4', ('nvx',))
    vx.nvx = len(vx_out)
    vx.longname = 'v_x/v_ti'
    vx[:] = vx_out[:]

    vy_out = np.asarray([vy_out[0][i][0] for i in range(0,len(vy_out))])
    vy = ncout.createVariable('vy','f4', ('nvy',))
    vy.nvy = len(vy_out)
    vy.longname = 'v_y/v_ti'
    vy[:] = vy_out[:]

    vz_out = np.asarray([vz_out[i][0][0] for i in range(0,len(vz_out))]) #assumes same number of data points along all axis in vz_out mesh var
    vz = ncout.createVariable('vz','f4', ('nvz',))
    vz.nvz = len(vz_out)
    vz.longname = 'v_z/v_ti'
    vz[:] = vz_out[:]

    x = ncout.createVariable('x','f4',('nx',))
    x.nx = len(x_out)
    x[:] = x_out[:]

    C_ex = ncout.createVariable('C_Ex','f4', ('nx', 'nvz', 'nvy', 'nvx'))
    C_ex.longname = 'C_{Ex}'
    C_ex[:] = CEx_out[:]

    C_ey = ncout.createVariable('C_Ey','f4', ('nx', 'nvz', 'nvy', 'nvx'))
    C_ey.longname = 'C_{Ey}'
    C_ey[:] = CEy_out[:]

    C_ez = ncout.createVariable('C_Ez','f4', ('nx', 'nvz', 'nvy', 'nvx'))
    C_ez.longname = 'C_{Ez}'
    C_ez[:] = CEz_out[:]

    Hist = ncout.createVariable('Hist','f4', ('nx', 'nvz', 'nvy', 'nvx'))
    Hist.longname = 'Hist'
    Hist[:] = Hist_out[:]

    n_par = ncout.createVariable('n_par','f4',('nx',))
    n_par.description = 'number of particles in each integration box. Note: this number might not match zeroth moment of the distribution function (i.e. integration over velocity space of hist) as particles outside the range of vmax are not counted in hist'
    n_par[:] = num_par[:]

    sda = ncout.createVariable('sda','f4',('nx',))
    sda.description = '1 = signature, 0 = no signature'
    sda[:] = metadata_out[:]

    enerCEx = ncout.createVariable('E_CEx','f4',('nx',))
    enerCEx.description = 'Energization computed by integrating over CEx in velocity space'
    enerCEx[:] = enerCEx_out[:]

    enerCEy = ncout.createVariable('E_CEy','f4',('nx',))
    enerCEy.description = 'Energization computed by integrating over CEy in velocity space'
    enerCEy[:] = enerCEy_out[:]

    enerCEz = ncout.createVariable('E_CEz','f4',('nx',))
    enerCEz.description = 'Energization computed by integrating over CEy in velocity space'
    enerCEz[:] = enerCEz_out[:]

    Vframe_relative_to_sim = ncout.createVariable('Vframe_relative_to_sim', 'f4')
    Vframe_relative_to_sim[:] = Vframe_relative_to_sim_out

    #Save data into netcdf4 file-----------------------------------------------------
    print("Saving data into netcdf4 file")

    #save file
    ncout.close()

def save2Vdata(Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, Vframe_relative_to_sim, num_par = [], metadata = [], params = {}, filename = 'full2Vdata.nc' ):
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
    ncout.version = get_git_head()

    #make dimensions that dependent data must 'match'
    ncout.createDimension('nx', None)  # NONE <-> unlimited TODO: make limited if it saves memory or improves compression?
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

    try:
        n_par_out = ncout.createVariable('n_par','f4',('nx',))
        n_par_out.description = 'number of particles in each integration box. Note: this number might not match zeroth moment of the distribution function (i.e. integration over velocity space of hist) as particles outside the range of vmax are not counted in hist'
        n_par_out[:] = npar[:]
    except:
        pass

    Vframe_relative_to_sim_out = ncout.createVariable('Vframe_relative_to_sim', 'f4')
    Vframe_relative_to_sim_out[:] = Vframe_relative_to_sim

    #save file
    ncout.close()


def load3Vnetcdf4(filename):
    """
    Loads 3v netcdf4 data created by save3Vdata function

    Parameters
    ----------
    filename : str
        filename of netcdf4. Should be formatted like *.nc

    Returns
    -------
    Hist_in : 4d array
        Distribution data created by analysis functions
        f(x;vx,vy,vz)
    CEx_in  : 4d array
        CEx data created by analysis functions
        CE_x(x;vx,vy,vz)
    CEy_in : 4d array
        CEy data created by analysis functions
        CE_x(x;vx,vy,vz)
    CEz_in : 4d array
        CEy data created by analysis functions
        CE_x(x;vx,vy,vz)
    vx_in : 3d array
        velocity space grid created by analysis functions
    vy_in : 3d array
        velocity space grid created by analysis functions
    vz_in : 3d array
        velocity space grid created by analysis functions
    x_in : 1d array
        x slice position data created by analysis functions
    enerCEx_in : 1d array
        integral over velocity space of CEx
    enerCEy_in : 1d array
        integral over velocity space of CEy
    enerCEz_in : 1d array
        integral over velocity space of CEy
    num_par_out : 1d array
        number of particles in box
    Vframe_relative_to_sim_in : float
        velocity of frame analysis was done in relative to the static simulation box frame
    metadata_in : 1d array, optional
        meta data array with length equal to x_out and axis 3 of correlation data
        normally needs to be made by hand
    params_in : dict, optional
        dictionary containing global attributes relating to data.
        contains mostly physical input parameters from original simulation
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
        elif(key == 'C_Ex'):
            CEx_in = ncin.variables['C_Ex'][:]
        elif(key == 'C_Ey'):
            CEy_in = ncin.variables['C_Ey'][:]
        elif(key == 'C_Ez'):
            CEz_in = ncin.variables['C_Ez'][:]
        elif(key == 'Hist'):
            Hist_in = ncin.variables['Hist'][:]
        elif(key == 'sda'):
            metadata_in = ncin.variables['sda'][:] #TODO: add ability to handle multiple types of metadata
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
        if(len(npar_in) > 0):
            return Hist_in, CEx_in, CEy_in, CEz_in, vx, vy, vz, x_in, enerCEx_in, enerCEy_in, enerCEz_in, npar_in, Vframe_relative_to_sim_in, metadata_in, params_in
    except:
        return Hist_in, CEx_in, CEy_in, CEz_in, vx, vy, vz, x_in, enerCEx_in, enerCEy_in, enerCEz_in, Vframe_relative_to_sim_in, metadata_in, params_in

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
            metadata_in = ncin.variables['sda'][:] #TODO: add ability to handle multiple types of metadata
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


#TODO: parse_input_file and read_input tries to do the same thing. However,
# both functions have different potential flaws. Need to make parse function
# that does not have flaws
def parse_input_file(path):
    """
    Puts dHybridR input file into dictionary.

    Warning: only works if the input file is formatted like 'var=val0,val1,val2,...'
    or 'var=val0'. I.e. input file must not have spaces between var val pairs

    Warning: this function is ad hoc and doesn't work for all input parameters,
    but it works for the ones we need it to

    Warning: this does not properly label injectors if there are multiple
    injectors for multiple species

    Warning: this function does not handle commenting out variable assignments well
    (i.e. avoiding leaving '!var=val0,val1,...' or '!C!var=val0,val1,...' in input
    file)

    Parameters
    ----------
    path : string
        path to directory containing simulation run

    Returns
    -------
    dict : dict
        dictionary containing input parameters
    """
    d = {}
    blockname = ''

    with open(path+'input/input') as f:
        for line in f:
            #get name of each block of input parameters
            contents = line.split(' ')
            speciescounter = 1 #used to handle having multiple species
            injectorcounter = 1 #used to handle having multiple injectors
            if(len(contents) == 1 and contents[0] != '\n' and contents[0] != '\t\n' and not('{' in contents[0]) and not('}' in contents[0]) and not('!' in contents[0])):
                blockname = contents[0].split('\n')[0]+'_'

            #check if line contains equal sign
            if '=' in line:
                for cnt in contents:
                    if '=' in cnt:
                        varvalpair = cnt.split('=')
                        if ',' in varvalpair[1]: #if multiple values, seperate and save
                            vals = varvalpair[1].split(',')
                            vals_out = []
                            for k in range(0,len(vals)):
                                vals[k] = vals[k].replace('"','')
                                if(vals[k].isnumeric() or _isfloat(vals[k])):
                                    vals_out.append(float(vals[k]))
                                elif(not(vals[k]=='' or '\n' in vals[k] or '\t' in vals[k])):
                                    vals_out.append(vals[k])

                            if(blockname == 'species_'):
                                if(blockname+str(speciescounter)+'_'+varvalpair[0] in d):
                                    speciescounter += 1
                                d[blockname+str(speciescounter)+'_'+varvalpair[0]] = vals_out
                            elif(blockname == 'plasma_injector_'):
                                if(blockname+str(injectorcounter)+'_'+varvalpair[0] in d):
                                    injectorcounter += 1
                                d[blockname+str(injectorcounter)+'_'+varvalpair[0]] = vals_out
                            else:
                                d[blockname+varvalpair[0]] = vals_out
                        else:
                            val = varvalpair[1]
                            if(val.isnumeric()):
                                val = float(val)
                            d[blockname+varvalpair[0]] = [val]

    return d

def read_input(path='./'):
    """
    Alternative parse dHybrid input file for simulation information (by Dr. Colby Haggerty)

    Parameters
    ----------
    path : str
        path of input file

    Returns
    -------
    dict : dict
        dictionary containing input parameters
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
            param[k] = [_auto_cast(c.strip()) for c in v.split(',')]
            if len(param[k]) == 1:
                param[k] = param[k][0]
    return param

def build_params(inputdict, numframe):
    """
    Computes relevant inputs parameters from simulation

    Parameters
    ----------
    inputdict : dict
        dictionary containing input parameters from parse_input_file

    numframe : int
        frame number (i.e. what time slice) with are analyzing

    Returns
    -------
    params : dict
        dictionary containing global attributes relating to data.
        contains mostly physical input parameters from original simulation
    """

    #relevant input data
    dt = float(inputdict['time_dt'][0]) #inverse Omega_ci0
    rqm = inputdict['species_1_rqm'][0] #charge to mass ratio (inverse)
    qi = 1./rqm

    #TODO check that species vth matches injector vth
    #assumes species 1 is ions. Not sure how to check this
    if(inputdict['species_1_vth'][0] == inputdict['plasma_injector_1_vth'][0]):
        vti = inputdict['species_1_vth'][0]
    else:
        print("Warning: injector vth does not match species vti")

    betaion = vti**2. #vti is normalized to v_alfven
    betaelec = 1. #TODO: figure out this

    Bx = inputdict['ext_emf_Bx'][0]
    By = inputdict['ext_emf_By'][0]
    Bz = inputdict['ext_emf_Bz'][0]
    if(Bx != 0.):
        shocknormalangle = abs(math.atan((By**2.+Bz**2.)**0.5/Bx))*360./(2.0*math.pi)
    else:
        shocknormalangle = 90.

    #define attributes/ simulation parameters
    params = {}
    params["MachAlfven"] = inputdict['plasma_injector_1_vdrift(1:3)'][0]
    params["MachAlfvenNote"] = 'TODO: compute mach alfven for this run'
    params["thetaBn"] = shocknormalangle
    params["thetaBndesc"] = 'units of degrees'
    params["betaelec"] = betaelec
    params["betaion"] = betaion
    params["simtime"] = numframe*dt
    params["simtimedesc"] = 'units of inverse Omega_{c,i,0}'
    params["qi"] = qi
    params["qidesc"] = 'charge to mass ratio'
    params["di"] = 0.0
    params["didesc"] = 'TODO: compute ion inertial length'
    params["vti"] = vti

    return params

def _auto_cast(k):
    """
    Takes an input string and tries to cast it to a real type (by Dr. Haggerty)

    Helper function for read_input

    Parameters
    ----------
    k : str
        A string that might be a int, float or bool

    Returns
    -------

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

def _isfloat(value):
    """
    Checks if string is float

    Parameters
    ----------
    value : str
        string you want to check if its a float

    Returns
    -------
    bool
        true if float
    """
    try:
        float(value)
        return True
    except ValueError:
        return False

def get_git_head():
    """
    Gets the hash string of the current head of the git repo for version management

    Returns
    -------
    temp : str
        hash string of current git head
    """

    import subprocess

    ["wc", "-l", "sorted_list.dat"]
    proc = subprocess.Popen(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tmp = proc.stdout.read()
    return str(tmp)[2:-3]
