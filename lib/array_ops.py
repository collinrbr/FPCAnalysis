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
    if(planename == 'xy'):
        meshx2d = np.zeros((ny, nx))
        meshy2d = np.zeros((ny, nx))

        meshx2d[:, :] = meshx[0, :, :]
        meshy2d[:, :] = meshy[0, :, :]

        return meshx2d, meshy2d

    elif(planename == 'xz'):
        meshx2d = np.zeros((nz, nx))
        meshz2d = np.zeros((nz, nx))

        meshx2d[:, :] = meshx[:, 0, :]
        meshz2d[:, :] = meshz[:, 0, :]

        return meshx2d, meshz2d

    elif(planename == 'yz'):
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
    if(planename == 'xy'):
        arr2d = np.apply_along_axis(np.sum, 0, arr3d)
        arr2d = np.swapaxes(arr2d, 0, 1) #rest of the code assumes this ordering
        return arr2d

    elif(planename == 'xz'):
        arr2d = np.apply_along_axis(np.sum, 1, arr3d)
        arr2d = np.swapaxes(arr2d, 0, 1) #rest of the code assumes this ordering
        return arr2d

    elif(planename == 'yz'):
        arr2d = np.apply_along_axis(np.sum, 2, arr3d)
        arr2d = np.swapaxes(arr2d, 0, 1) #rest of the code assumes this ordering
        return arr2d
    else:
        print("Please enter xy, xz, or yz for planename...")


def get_average_in_box(x1, x2, y1, y2, z1, z2, datadict, dictkey):
    """
    Get linear average of fields in box from grid points within box.

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

    # mask data outside of given bounds
    gptsx = (x1 <= datadict[dictkey+'_xx']) & (datadict[dictkey+'_xx'] <= x2)
    gptsy = (y1 <= datadict[dictkey+'_yy']) & (datadict[dictkey+'_yy'] <= y2)
    gptsz = (z1 <= datadict[dictkey+'_zz']) & (datadict[dictkey+'_zz'] <= z2)

    goodpts = []
    for i in range(0, len(gptsx)):
        for j in range(0, len(gptsy)):
            for k in range(0, len(gptsz)):
                if(gptsx[i] and gptsy[j] and gptsz[k]):
                    goodpts.append(datadict[dictkey][k][j][i])

    avg = np.average(goodpts)
    return avg


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
        import matplotlib.pyplot as plt

        plt.plot(data)
        plt.plot(peaks, data[peaks], "x")
        plt.show()

    return peaks
