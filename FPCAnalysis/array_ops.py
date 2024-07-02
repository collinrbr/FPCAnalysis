# array_ops.py>

# functions that manipulate or operate arrays

import numpy as np


def find_nearest(array, value):
    """
    Finds index of element in array with value closest to given value

    Paramters
    ---------
    array : 1d array
        array
    value : float
        value you want to approximately find in array

    Returns
    -------
    idx : int
        index of nearest element
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_two_nearest(array, value):
    """
    Finds the index of the two elements in an array closest to some given value.
    Assumes given array is ordered.

    Parameters
    ----------
    array : array
        ordered data array
    value : float
        search value

    Returns
    -------
    idx1, idx2 : ints
        indicies closest to given value
    """
    if(len(array) == 1):
        return 0, 0

    array = np.asarray(array)
    idx1 = (np.abs(array - value)).argmin()
    if(idx1 == 0):  # if on left boundary
        idx2 = 1
    elif(idx1 == len(array)-1):  # if on right boundary
        idx2 = len(array)-2
    elif(np.abs(array[idx1+1]-value) < np.abs(array[idx1-1]-value)):
        idx2 = idx1+1
    else:
        idx2 = idx1-1

    # error checking (needed for 1d/2d simulations)
    if(idx1 >= len(array) and len(array) == 1):
        idx1 = 0
    if(idx2 >= len(array) and len(array) == 1):
        idx2 = 0

    return idx1, idx2


def mesh_3d_to_2d(meshx, meshy, meshz, planename):
    """
    Converts 3d velocity space arrays to 2d
    Used for plotting

    Parameters
    ----------
    meshx : 3d array
        3d meshx grid
    meshy : 3d array
        3d meshy grid
    meshz : 3d array
        3d meshz grid
    planename : str
        name of plane you want to get 2d grid of

    Returns
    -------
    *Returns 2 of 3 of the following based on planename*
    meshx2d : 2d array
        2d meshx grid
    meshy2d : 2d array
        2d meshy grid
    meshz2d : 2d array
        2d meshz grid
    """
    nz,ny,nx = meshx.shape
    if(planename == 'xy' or planename == 'parperp1'):
        meshx2d = np.zeros((ny, nx))
        meshy2d = np.zeros((ny, nx))

        meshx2d[:, :] = meshx[0, :, :]
        meshy2d[:, :] = meshy[0, :, :]

        return meshx2d, meshy2d

    elif(planename == 'xz' or planename == 'parperp2'):
        meshx2d = np.zeros((nz, nx))
        meshz2d = np.zeros((nz, nx))

        meshx2d[:, :] = meshx[:, 0, :]
        meshz2d[:, :] = meshz[:, 0, :]

        return meshx2d, meshz2d

    elif(planename == 'yz' or planename == 'perp1perp2'):
        meshy2d = np.zeros((nz, ny))
        meshz2d = np.zeros((nz, ny))

        meshy2d[:, :] = meshy[:, :, 0]
        meshz2d[:, :] = meshz[:, :, 0]

        return meshy2d, meshz2d

def array_3d_to_2d(arr3d, planename):
    """
    Projects data in 3d array to 2d array

    Parameters
    ----------
    arr3d : 3d array
        3d data
    planename : str
        name of plane you want to project onto

    Returns
    -------
    arr2d : 2d array
        2d projection of the data
    """
    nz = len(arr3d)
    ny = len(arr3d[0])
    nx = len(arr3d[0][0])
    if(planename == 'xy' or planename == 'parperp1'):
        arr2d = np.apply_along_axis(np.sum, 0, arr3d)
        arr2d = np.swapaxes(arr2d, 0, 1) #rest of the code assumes this ordering
        return arr2d

    elif(planename == 'xz' or planename ==  'parperp2'):
        arr2d = np.apply_along_axis(np.sum, 1, arr3d)
        arr2d = np.swapaxes(arr2d, 0, 1) #rest of the code assumes this ordering
        return arr2d

    elif(planename == 'yz' or planename == 'perp1perp2'):
        arr2d = np.apply_along_axis(np.sum, 2, arr3d)
        arr2d = np.swapaxes(arr2d, 0, 1) #rest of the code assumes this ordering
        return arr2d
    else:
        print("Please enter xy, xz, yz, parperp1, parperp2, or perp1perp2 for planename...")


def get_average_in_box(x1, x2, y1, y2, z1, z2, datadict, dictkey):
    """
    Get flat average of fields in box from grid points within box.

    Parameters
    ----------
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
    datadict : dict
        data dictionary (usually from field_loader or flow_loader)
    dictkey : str
        name of key you are averaging

    Returns
    -------
    avg : float
        average value in box
    """

    #get bounds of data
    xidxmin = find_nearest(datadict[dictkey+'_xx'], x1)
    xidxmax = find_nearest(datadict[dictkey+'_xx'], x2)
    yidxmin = find_nearest(datadict[dictkey+'_yy'], y1)
    yidxmax = find_nearest(datadict[dictkey+'_yy'], y2)
    zidxmin = find_nearest(datadict[dictkey+'_zz'], z1)
    zidxmax = find_nearest(datadict[dictkey+'_zz'], z2)

    #adjust if needed and possible
    if(datadict[dictkey+'_xx'][xidxmin] > x1 and xidxmin != 0): xidxmin = xidxmin - 1
    if(datadict[dictkey+'_xx'][xidxmax] < x2 and xidxmax != len(datadict[dictkey+'_xx'])-1): xidxmax = xidxmax + 1
    if(datadict[dictkey+'_yy'][yidxmin] > y1 and yidxmin != 0): yidxmin = yidxmin - 1
    if(datadict[dictkey+'_yy'][yidxmax] < y2 and yidxmax != len(datadict[dictkey+'_yy'])-1): yidxmax = yidxmax + 1
    if(datadict[dictkey+'_zz'][zidxmin] > z1 and zidxmin != 0): zidxmin = zidxmin - 1
    if(datadict[dictkey+'_zz'][zidxmax] < z2 and zidxmax != len(datadict[dictkey+'_zz'])-1): zidxmax = zidxmax + 1

    goodpts=datadict[dictkey][zidxmin:zidxmax,yidxmin:yidxmax,xidxmin:xidxmax]

    avg = np.average(goodpts)
    return avg

def subset_dict(ddict,startidxs,endidxs,planes=['z','y','x']):
    """
    Takes connected subset of provided dict

    Parameters
    ----------
    ddict : dict
        field, fluid, dens dict- (e.g. dfields from load_fields in loadaux)
    startidxs/endidxs : array of size 2/3
        start/end indexes arrays ([zidx,yidx,xidx] e.g. [0,56,100]) 
    planes : array of size 3
        planes to modify (['z','y','x'] or ['y','x'])

    Returns
    -------
    ddictout : dict
        subseet of provided dict
    """

    #get keys that need to be reduced:
    dkeys = list(ddict.keys())
    keys = [dkeys[_i] for _i in range(0,len(dkeys)) if ('_' in dkeys[_i] and dkeys[_i][-1] in planes)]

    import copy
    ddictout = copy.deepcopy(ddict)

    for kyidx in range(0,len(keys)):
        if(not(keys[kyidx].split('_')[0] in keys)):
            keys.append(keys[kyidx].split('_')[0])

    for ky in keys:
        if('_' in ky):
            if('x' in planes):
                if(ky[-1] == 'x'):
                    if(len(startidxs)==2):
                        ddictout[ky] = ddictout[ky][startidxs[1]:endidxs[1]]
                    if(len(startidxs)==3):
                        ddictout[ky] = ddictout[ky][startidxs[2]:endidxs[2]]
            if('y'in planes):
                if(ky[-1] == 'y'):
                    if(len(startidxs)==2):
                        ddictout[ky] = ddictout[ky][startidxs[0]:endidxs[0]]
                    if(len(startidxs)==3):
                        ddictout[ky] = ddictout[ky][startidxs[1]:endidxs[1]]
            if('z' in planes):
                if(ky[-1] == 'z'):
                    ddictout[ky] = ddictout[ky][startidxs[0]:endidxs[0]]
        else:
            if('x' in planes and 'y' in planes and not('z' in planes)):
                ddictout[ky] = ddictout[ky][startidxs[0]:endidxs[0],startidxs[1]:endidxs[1]]
            elif('x' in planes and 'y' in planes and 'z' in planes):
                ddictout[ky] = ddictout[ky][startidxs[0]:endidxs[0],startidxs[1]:endidxs[1],startidxs[2]:endidxs[2]]

    return ddictout

def truncate_dict(ddict,reducfrac=[1,1,2],planes=['z','y','x']):
    """
    Truncates fraction of array greater tha fraction size. 

    E.g. reducfrac=[3] for array [1,2,3,4,5,6] will return [1,2]

    Parameters
    ----------
    ddict : dict
        field, fluid, dens dict- (e.g. dfields from load_fields in loadaux)
    reducfrac : array of size 2/3
        reducfrac in each direction
    planes : array of size 3
        planes to modify (['z','y','x'] or ['y','x'])

    Returns
    -------
    ddictout : dict
        subseet of provided dict
    """

    #get keys that need to be reduced:
    dkeys = list(ddict.keys())
    keys = [dkeys[_i] for _i in range(0,len(dkeys)) if ('_' in dkeys[_i] and dkeys[_i][-1] in planes)]

    import copy
    ddictout = copy.deepcopy(ddict)

    for kyidx in range(0,len(keys)):
        if(not(keys[kyidx].split('_')[0] in keys)):
            keys.append(keys[kyidx].split('_')[0])

    for ky in keys:
        if('_' in ky):
            if('x' in planes):
                if(ky[-1] == 'x'):
                    if(len(reducfrac)==2):
                        ddictout[ky] = ddictout[ky][0:int(round(len(ddictout[ky])/reducfrac[1]))]
                    if(len(reducfrac)==3):
                        ddictout[ky] = ddictout[ky][0:int(round(len(ddictout[ky])/reducfrac[2]))]
            if('y'in planes):
                if(ky[-1] == 'y'):
                    if(len(reducfrac)==2):
                        ddictout[ky] = ddictout[ky][0:int(round(len(ddictout[ky])/reducfrac[0]))]
                    if(len(reducfrac)==3):
                        ddictout[ky] = ddictout[ky][0:int(round(len(ddictout[ky])/reducfrac[1]))]
            if('z' in planes):
                if(ky[-1] == 'z'):
                    ddictout[ky] = ddictout[ky][0:int(round(len(ddictout[ky])/reducfrac[0]))]
        else:
            if('x' in planes and 'y' in planes and not('z' in planes)):
                ddictout[ky] = ddictout[ky][0:int(round(len(ddictout[ky][0,:,0])/reducfrac[0])),0:int(round(len(ddictout[ky][0,0,:]/reducfrac[1])))]
            elif('x' in planes and 'y' in planes and 'z' in planes):
                ddictout[ky] = ddictout[ky][0:int(round(len(ddictout[ky][:,0,0])/reducfrac[0])),0:int(round(len(ddictout[ky][0,:,0])/reducfrac[1])),0:int(round(len(ddictout[ky][0,0,:])/reducfrac[2]))]

    return ddictout

def avg_dict(ddict,binidxsz=[2,2,2],planes=['z','y','x']):
    """
    Averages array into bins equal to integer multiples of original size

    Note: grid must be divisible without truncation into new grid

    Parameters
    ----------
    ddict : dict
        field, fluid, dens dict- (e.g. dfields from load_fields in loadaux)
    binidxsz : array of size 2/3 (of integers)
        new, larger, integer multiple size of binn
    planes : array of size 3
        planes to modify (['z','y','x'] or ['y','x'])

    Returns
    -------
    ddictout : dict
        subseet of provided dict
    """

    #get keys that need to be reduced:
    dkeys = list(ddict.keys())
    keys = [dkeys[_i] for _i in range(0,len(dkeys)) if ('_' in dkeys[_i] and dkeys[_i][-1] in planes)]

    import copy
    ddictout = copy.deepcopy(ddict)

    for kyidx in range(0,len(keys)):
        if(not(keys[kyidx].split('_')[0] in keys)):
            keys.append(keys[kyidx].split('_')[0])

    #TODO: test and use something like numpy.mean(x.reshape(-1, 2), 1) 
    for ky in keys:
        if('_' in ky):
            if('x' in planes):
                if(ky[-1] == 'x'):
                    if(len(binidxsz)==2):
                        print("Error! Must be used with a 3d sim") #TODO: implement for 2d arrays
                        return
                    if(len(binidxsz)==3):
                        ddictout[ky] = avg_bin_1darr(ddictout[ky],binidxsz[2])
            if('y'in planes):
                if(ky[-1] == 'y'):
                    if(len(binidxsz)==2):
                        print("Error! Must be a 3d sim") #TODO: implement for 2d arrays
                        return
                    if(len(binidxsz)==3):
                        ddictout[ky] = avg_bin_1darr(ddictout[ky],binidxsz[1])
            if('z' in planes):
                if(ky[-1] == 'z'):
                    ddictout[ky] = avg_bin_1darr(ddictout[ky],binidxsz[0])
        else:
            if('x' in planes and 'y' in planes and not('z' in planes)):
                print("Error! Must be used with a 3d sim")
                return
            elif('x' in planes and 'y' in planes and 'z' in planes):
                ddictout[ky] = avg_bin_3darr(ddictout[ky],binidxsz[0],binidxsz[1],binidxsz[2])

    return ddictout

#TODO: rename downsample_factor1
def avg_bin_3darr(data_array,downsample_factor1,downsample_factor2,downsample_factor3):
    """
    Averages 3D array of shape (factor1,factor2,factor3) into 3D array of shape (factor1/downsample_factor1,factor2/downsample_factor2,factor3/downsample_factor3)

    Parameters
    ----------
    data_array : 3D array
        data to be averaged
    downsample_factor(1/2/3) : int
        factor to downsample array by in each dim

    Returns
    -------
    downsampled_array : 3D array
        averaged data
    """

    nz,ny,nx = data_array.shape
    if(not(nz % downsample_factor1 == 0)):
        print("Error! nz must be divisible by downsample_factor1!")
        return

    if(not(ny % downsample_factor2 == 0)):
        print("Error! ny must be divisible by downsample_factor2!")
        return

    if(not(nx % downsample_factor3 == 0)):
        print("Error! nx must be divisible by downsample_factor3!")
        return

    new_height = data_array.shape[0] // downsample_factor1
    new_width = data_array.shape[1] // downsample_factor2
    new_depth = data_array.shape[2] // downsample_factor3
    reshaped_array = data_array[:new_height * downsample_factor1,
                            :new_width * downsample_factor2,
                            :new_depth * downsample_factor3]
    reshaped_array = reshaped_array.reshape(new_height, downsample_factor1,
                                        new_width, downsample_factor2,
                                        new_depth, downsample_factor3)
    downsampled_array = np.mean(reshaped_array, axis=(1, 3, 5))

    return np.asarray(downsampled_array)

def avg_bin_1darr(data_array,downsample_factor):
    """
    Averages 1D array of shape (factor1) into 1D array of shape (factor1/downsample_factor1)

    Parameters
    ----------
    data_array : 1D array
        data to be averaged
    downsample_factor : int
        factor to downsample array

    Returns
    -------
    downsampled_array : 1D array
        averaged data
    """

    narr = len(data_array)
    if(not(narr % downsample_factor == 0)):
        print("Error! narr must be divisible by downsample_factor")
        print('narr: ', narr, 'downsample_factor: ', downsample_factor, 'narr%downsample_factor', narr%downsample_factor)
        return

    new_length = len(data_array) // downsample_factor
    reshaped_array = data_array[:new_length * downsample_factor]
    reshaped_array = reshaped_array.reshape(new_length, downsample_factor)
    downsampled_array = np.mean(reshaped_array, axis=1)

    return np.asarray(downsampled_array)

#TODO: there is one function that is redundant (see subset_dict)- pick and remove one maybe?
def get_field_subset(dfields, startx, endx, starty, endy, startz, endz):
    """
    Grabs subset box of field data

    TODO: move this to another library file

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    startx : float
        lower x bound
    endx : float
        upper x bound
    starty : float
        lower y bound
    endy : float
        lower y bound
    startz : float
        lower z bound
    endz : float
        lower z bound

    Returns
    -------
    dfieldssubset : dict
        subset of field data
    """

    from copy import copy

    startx = find_nearest(dfields['bz_xx'], startx)
    endx = find_nearest(dfields['bz_xx'], endx)
    starty = find_nearest(dfields['bz_yy'], starty)
    endy = find_nearest(dfields['bz_yy'], endy)
    startz = find_nearest(dfields['bz_zz'], startz)
    endz = find_nearest(dfields['bz_zz'], endz)

    dfieldssubset = copy(dfields)

    dfieldssubset['ex_xx'] = dfieldssubset['ex_xx'][startx:endx]
    dfieldssubset['ex_yy'] = dfieldssubset['ex_yy'][starty:endy]
    dfieldssubset['ex_zz'] = dfieldssubset['ex_zz'][startz:endz]
    dfieldssubset['ey_xx'] = dfieldssubset['ey_xx'][startx:endx]
    dfieldssubset['ey_yy'] = dfieldssubset['ey_yy'][starty:endy]
    dfieldssubset['ey_zz'] = dfieldssubset['ey_zz'][startz:endz]
    dfieldssubset['ez_xx'] = dfieldssubset['ez_xx'][startx:endx]
    dfieldssubset['ez_yy'] = dfieldssubset['ez_yy'][starty:endy]
    dfieldssubset['ez_zz'] = dfieldssubset['ez_zz'][startz:endz]

    dfieldssubset['bx_xx'] = dfieldssubset['bx_xx'][startx:endx]
    dfieldssubset['bx_yy'] = dfieldssubset['bx_yy'][starty:endy]
    dfieldssubset['bx_zz'] = dfieldssubset['bx_zz'][startz:endz]
    dfieldssubset['by_xx'] = dfieldssubset['by_xx'][startx:endx]
    dfieldssubset['by_yy'] = dfieldssubset['by_yy'][starty:endy]
    dfieldssubset['by_zz'] = dfieldssubset['by_zz'][startz:endz]
    dfieldssubset['bz_xx'] = dfieldssubset['bz_xx'][startx:endx]
    dfieldssubset['bz_yy'] = dfieldssubset['bz_yy'][starty:endy]
    dfieldssubset['bz_zz'] = dfieldssubset['bz_zz'][startz:endz]

    dfieldssubset['ex'] = dfieldssubset['ex'][startz:endz, starty:endy, startx:endx]
    dfieldssubset['ey'] = dfieldssubset['ey'][startz:endz, starty:endy, startx:endx]
    dfieldssubset['ez'] = dfieldssubset['ez'][startz:endz, starty:endy, startx:endx]
    dfieldssubset['bx'] = dfieldssubset['bx'][startz:endz, starty:endy, startx:endx]
    dfieldssubset['by'] = dfieldssubset['by'][startz:endz, starty:endy, startx:endx]
    dfieldssubset['bz'] = dfieldssubset['bz'][startz:endz, starty:endy, startx:endx]

    return dfieldssubset


def get_flow_subset(dflow, startx, endx, starty, endy, startz, endz):
    """
    Grabs subset box of flow data

    TODO: move this to another library file

    Parameters
    ----------
    dfields : dict
        field data dictionary from flow_loader
    startx : float
        lower x bound
    endx : float
        upper x bound
    starty : float
        lower y bound
    endy : float
        lower y bound
    startz : float
        lower z bound
    endz : float
        lower z bound

    Returns
    -------
    dfieldssubset : dict
        subset of flow data
    """
    from copy import copy

    startx = find_nearest(dflow['ux_xx'], startx)
    endx = find_nearest(dflow['ux_xx'], endx)
    starty = find_nearest(dflow['ux_yy'], starty)
    endy = find_nearest(dflow['ux_yy'], endy)
    startz = find_nearest(dflow['ux_zz'], startz)
    endz = find_nearest(dflow['ux_zz'], endz)

    dflowsubset = copy(dflow)
    dflowsubset['ux_xx'] = dflowsubset['ux_xx'][startx:endx]
    dflowsubset['ux_yy'] = dflowsubset['ux_yy'][starty:endy]
    dflowsubset['ux_zz'] = dflowsubset['ux_zz'][startz:endz]
    dflowsubset['uy_xx'] = dflowsubset['uy_xx'][startx:endx]
    dflowsubset['uy_yy'] = dflowsubset['uy_yy'][starty:endy]
    dflowsubset['uy_zz'] = dflowsubset['uy_zz'][startz:endz]
    dflowsubset['uz_xx'] = dflowsubset['uz_xx'][startx:endx]
    dflowsubset['uz_yy'] = dflowsubset['uz_yy'][starty:endy]
    dflowsubset['uz_zz'] = dflowsubset['uz_zz'][startz:endz]

    dflowsubset['ux'] = dflowsubset['ux'][startz:endz, starty:endy, startx:endx]
    dflowsubset['uy'] = dflowsubset['uy'][startz:endz, starty:endy, startx:endx]
    dflowsubset['uz'] = dflowsubset['uz'][startz:endz, starty:endy, startx:endx]

    return dflowsubset


def find_local_maxima(data, threshold=.05, pltdebug=False):
    """
    Finds indicies of the local maxima of given data that are greater than some fraction of the
    max value in the data

    Assumes data is all positive

    Parameters
    ----------
    data : array
        array of data
    threshold : float, opt
        cutoff fraction
    pltdebug : bool
        shows debug plot if requested

    Return
    ------
    peaks : array
        list of indexes of peaks in data
    """

    # from scipy.signal import find_peaks
    from scipy.signal import argrelextrema
    from scipy.signal import savgol_filter

    data = savgol_filter(data, 11, 5)
    peaks = argrelextrema(data, np.greater)[0]

    # remove points below some fraction of the max peak
    _peaks = []
    maxdata = np.max(np.abs(data[peaks]))
    for i in range(0, len(peaks)):
        if(np.abs(data[peaks[i]]) > threshold*maxdata):
            _peaks.append(peaks[i])
    peaks = _peaks

    if(pltdebug):
        import matplotFPCAnalysis.pyplot as plt

        plt.plot(data)
        plt.plot(peaks, data[peaks], "x")
        plt.show()

    return peaks

def interpolate(independent_vars, dependent_vars, locations):
    independent_vars = np.array(independent_vars)
    dependent_vars = np.array(dependent_vars)
    locations = np.array(locations)
    interpolated_values = np.interp(locations, independent_vars, dependent_vars)
    return locations, interpolated_values

def split_positive_negative(arr):
    arr = np.array(arr)
            
    positive_array = np.where(arr > 0, arr, 0)
    negative_array = np.where(arr < 0, arr, 0)

    return positive_array, negative_array
    
