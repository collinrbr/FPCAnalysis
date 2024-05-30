import numpy as np

def find_nearest(array, val):
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
    idx = (np.abs(array-val)).argmin()
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
    Get linear average of fields in box from grid points within box.

    Assumes arrays are ordered.

    Assumes range spans at least two boxes in all directions.

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
    xidxmin = find_nearest(datadict['ex_xx'], x1)
    xidxmax = find_nearest(datadict['ex_xx'], x2)
    yidxmin = find_nearest(datadict['ex_yy'], y1)
    yidxmax = find_nearest(datadict['ex_yy'], y2)
    zidxmin = find_nearest(datadict['ex_zz'], z1)
    zidxmax = find_nearest(datadict['ex_zz'], z2)

    #adjust if needed and possible
    if(datadict[dictkey+'_xx'][xidxmin] > x1 and xidxmin != 0): xidxmin = xidxmin - 1
    if(datadict[dictkey+'_xx'][xidxmax] < x2 and xidxmax != len(datadict[dictkey+'_xx'])-1): xidxmax = xidxmax + 1
    if(datadict[dictkey+'_yy'][yidxmin] > y1 and yidxmin != 0): yidxmin = yidxmin - 1
    if(datadict[dictkey+'_yy'][yidxmax] < y2 and yidxmax != len(datadict[dictkey+'_yy'])-1): yidxmax = yidxmax + 1
    if(datadict[dictkey+'_zz'][zidxmin] > z1 and zidxmin != 0): zidxmin = zidxmin - 1
    if(datadict[dictkey+'_zz'][zidxmax] < z2 and zidxmax != len(datadict[dictkey+'_zz'])-1): zidxmax = zidxmax + 1

    numpoints = 0.
    totval = 0.
    for i in range(xidxmin,xidxmax+1):
        for j in range(yidxmin,yidxmax+1):
            for k in range(zidxmin,zidxmax+1):
                numpoints = numpoints + 1.
                totval += datadict[dictkey][k][j][i] 

    if(numpoints == 0.):print("Warning! get_average_in_box found no points in range!")
    avg = totval / numpoints
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
