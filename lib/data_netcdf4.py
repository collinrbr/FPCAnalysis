# data_netcdf4.py>

#functions related making and loading netcdf4 file to pass to MLA algo and loading netcdf4 files

import numpy as np

def savedata(CEx_out, CEy_out, vx_out, vy_out, x_out, enerCEx_out, enerCEy_out, Vframe_relative_to_sim_out, metadata_out = [], params = {}, filename = 'dHybridRSDAtest.nc' ):
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
    enerCEx_out : 1d array
        integral over velocity space of CEx
    enerCEy_out : 1d array
        integral over velocity space of CEy
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
    #Here we normalize to the maximum value in either CEx, CEy
    maxCval = max(np.amax(np.abs(CEx_out)),np.amax(np.abs(CEy_out)))
    CEx_out /= maxCval
    CEy_out /= maxCval

    # open a netCDF file to write
    ncout = Dataset(filename, 'w', format='NETCDF4')


    #save data in netcdf file-------------------------------------------------------
    #define simulation parameters
    for key in params:
        #setattr(ncout,key,params[key])
        if(not(isinstance(params[key],str))):
            _ = ncout.createVariable(key,None)
            _[:] = params[key]

    ncout.description = 'dHybridR MLA data test 1' #TODO: report dHybridR pipeline version here
    ncout.generationtime = str(datetime.now())
    ncout.version = get_git_head()

    #make dimensions that dependent data must 'match'
    ncout.createDimension('nx', None)  # NONE <-> unlimited TODO: make limited if it saves memory or improves compression?
    ncout.createDimension('nvx', None)
    ncout.createDimension('nvy', None)

    vx_out = vx_out[0][:]
    vx = ncout.createVariable('vx','f4', ('nvx',))
    vx.nvx = len(vx_out)
    vx.longname = 'v_x/v_ti'
    vx[:] = vx_out[:]

    vy_out = np.asarray([vy_out[i][0] for i in range(0,len(vy_out))])
    vy = ncout.createVariable('vy','f4', ('nvy',))
    vy.nvy = len(vy_out)
    vy.longname = 'v_y/v_ti'
    vy[:] = vy_out[:]

    x = ncout.createVariable('x','f4',('nx',))
    x.nx = len(x_out)
    x[:] = x_out[:]

    #tranpose data to match previous netcdf4 formatting
    for i in range(0,len(CEx_out)):
        tempCex = CEx_out[i].T
        CEx_out[i] = tempCex
        tempCey = CEy_out[i].T
        CEy_out[i] = tempCey

    C_ex = ncout.createVariable('C_Ex','f4', ('nx', 'nvx', 'nvy'))
    C_ex.longname = 'C_{Ex}'
    C_ex[:] = CEx_out[:]

    C_ey = ncout.createVariable('C_Ey','f4', ('nx', 'nvx', 'nvy'))
    C_ey.longname = 'C_{Ey}'
    C_ey[:] = CEy_out[:]

    metadata = ncout.createVariable('sda','f4',('nx',))
    metadata.description = '1 = signature, 0 = no signature'
    metadata[:] = metadata_out[:]

    enerCEx = ncout.createVariable('E_CEx','f4',('nx',))
    enerCEx.description = 'Energization computed by integrating over CEx in velocity space'
    enerCEx[:] = enerCEx_out[:]

    enerCEy = ncout.createVariable('E_CEy','f4',('nx',))
    enerCEy.description = 'Energization computed by integrating over CEy in velocity space'
    enerCEy[:] = enerCEy_out[:]

    Vframe_relative_to_sim = ncout.createVariable('Vframe_relative_to_sim', 'f4')
    Vframe_relative_to_sim[:] = Vframe_relative_to_sim_out

    #Save data into netcdf4 file-----------------------------------------------------
    print("Saving data into netcdf4 file")

    #save file
    ncout.close()

def savefulldata(CEx_out, CEy_out, CEz_out, vx_out, vy_out, vz_out, x_out, enerCEx_out, enerCEy_out, energCEz_out, Vframe_relative_to_sim_out, metadata_out = [], params = {}, filename = 'dHybridRSDAtest.nc' ):
    """
    Creates netcdf4 data of normalized correlation data to send to MLA algo.

    Parameters
    ----------
    CEx_out : 4d array
        Ex correlation data created by fpc correlation functions
    CEy_out : 4d array
        Ey correlation data created by fpc correlation functions
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

    # open a netCDF file to write
    ncout = Dataset(filename, 'w', format='NETCDF4')

    #save data in netcdf file-------------------------------------------------------
    #define simulation parameters
    for key in params:
        #setattr(ncout,key,params[key])
        if(not(isinstance(params[key],str))):
            _ = ncout.createVariable(key,None)
            _[:] = params[key]

    ncout.description = 'dHybridR MLA data test 2'
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

    x = ncout.createVariable('x','f4',('nx',))
    x.nx = len(x_out)
    x[:] = x_out[:]

    #tranpose data to match previous netcdf4 formatting
    for i in range(0,len(CEx_out)):
        tempCex = CEx_out[i].T
        CEx_out[i] = tempCex
        tempCey = CEy_out[i].T
        CEy_out[i] = tempCey

    C_ex = ncout.createVariable('C_Ex','f4', ('nx', 'nvx', 'nvy'))
    C_ex.longname = 'C_{Ex}'
    C_ex[:] = CEx_out[:]

    C_ey = ncout.createVariable('C_Ey','f4', ('nx', 'nvx', 'nvy'))
    C_ey.longname = 'C_{Ey}'
    C_ey[:] = CEy_out[:]

    metadata = ncout.createVariable('sda','f4',('nx',))
    metadata.description = '1 = signature, 0 = no signature'
    metadata[:] = metadata_out[:]

    enerCEx = ncout.createVariable('E_CEx','f4',('nx',))
    enerCEx.description = 'Energization computed by integrating over CEx in velocity space'
    enerCEx[:] = enerCEx_out[:]

    enerCEy = ncout.createVariable('E_CEy','f4',('nx',))
    enerCEy.description = 'Energization computed by integrating over CEy in velocity space'
    enerCEy[:] = enerCEy_out[:]

    Vframe_relative_to_sim = ncout.createVariable('Vframe_relative_to_sim', 'f4')
    Vframe_relative_to_sim[:] = Vframe_relative_to_sim_out

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
    enerCEx_out : 1d array
        integral over velocity space of CEx
    enerCEy_out : 1d array
        integral over velocity space of CEy
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

    params_in = {}
    for key in ncin.variables.keys():
        if(key == 'x'):
            x_in = ncin.variables['x'][:]
        elif(key == 'vx'):
            vx_in = ncin.variables['vx'][:]
        elif(key == 'vy'):
            vy_in = ncin.variables['vy'][:]
        elif(key == 'C_Ex'):
            CEx_in = ncin.variables['C_Ex'][:]
        elif(key == 'C_Ey'):
            CEy_in = ncin.variables['C_Ey'][:]
        elif(key == 'sda'):
            metadata_in = ncin.variables['sda'][:] #TODO: add ability to handle multiple types of metadata
        elif(key == 'E_CEx'):
            enerCEx_in = ncin.variables['E_CEx'][:]
        elif(key == 'E_CEy'):
            enerCEy_in = ncin.variables['E_CEy'][:]
        elif(key == 'Vframe_relative_to_sim'):
            Vframe_relative_to_sim_in = ncin.variables['Vframe_relative_to_sim'][:]
        else:
            if(not(isinstance(ncin.variables[key][:], str))):
                params_in[key] = ncin.variables[key][:]

    #add global attributes
    params_in.update(ncin.__dict__)

    #tranpose data back to match dHybridR pipeline ordering #TODO: remove this if no bugs show up. Unsure if ordering is consistent everywhere
    for i in range(0,len(CEx_in)):
        tempCex = CEx_in[i].T
        CEx_in[i] = tempCex
        tempCey = CEy_in[i].T
        CEy_in[i] = tempCey

    #reconstruct vx, vy 2d arrays
    _vx = np.zeros((len(vy_in),len(vx_in)))
    _vy = np.zeros((len(vy_in),len(vx_in)))
    for i in range(0,len(vy_in)):
        for j in range(0,len(vx_in)):
            _vx[i][j] = vx_in[j]

    for i in range(0,len(vy_in)):
        for j in range(0,len(vx_in)):
            _vy[i][j] = vy_in[i]

    vx_in = _vx
    vy_in = _vy

    return CEx_in, CEy_in, vx_in, vy_in, x_in, enerCEx_in, enerCEy_in, Vframe_relative_to_sim_in, metadata_in, params_in

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
