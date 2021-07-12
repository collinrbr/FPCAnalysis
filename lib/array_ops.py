# array_ops.py>

#functions that manipulate or operate arrays

import numpy as np

def find_nearest(array, value):
    """
    Finds index of element in array with value closest to given value

    Paramters
    ---------
    array : 1d array
        ordered array
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

def mesh_3d_to_2d(meshx,meshy,meshz,planename):
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

    if(planename == 'xy'):
        meshx2d = np.zeros((len(meshy),len(meshx)))
        meshy2d = np.zeros((len(meshy),len(meshx)))
        for i in range(0,len(meshy)):
            for j in range(0,len(meshx)):
                meshx2d[i][j] = meshx[0][i][j]
        for i in range(0,len(meshy)):
            for j in range(0,len(meshx)):
                meshy2d[i][j] = meshy[0][i][j]

        return meshx2d, meshy2d

    elif(planename == 'xz'):
        meshx2d = np.zeros((len(meshz),len(meshx)))
        meshz2d = np.zeros((len(meshz),len(meshx)))
        for i in range(0,len(meshz)):
            for j in range(0,len(meshx)):
                meshx2d[i][j] = meshx[i][0][j]
        for i in range(0,len(meshz)):
            for j in range(0,len(meshx)):
                meshz2d[i][j] = meshz[i][0][j]

        return meshx2d, meshz2d

    elif(planename == 'yz'):
        meshy2d = np.zeros((len(meshz),len(meshy)))
        meshz2d = np.zeros((len(meshz),len(meshy)))
        for i in range(0,len(meshz)):
            for j in range(0,len(meshy)):
                meshy2d[i][j] = meshy[i][j][0]
        for i in range(0,len(meshz)):
            for j in range(0,len(meshy)):
                meshz2d[i][j] = meshz[i][j][0]

        return meshy2d, meshz2d

def array_3d_to_2d(arr3d,planename):
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
    arr2d = np.zeros((len(arr3d),len(arr3d[0])))
    if(planename == 'xy'):
        for i in range(0,len(arr3d)):
            for j in range(0,len(arr3d[i])):
                for k in range(0,len(arr3d[i][j])):
                    arr2d[k][j] += arr3d[i][j][k]
        return arr2d

    elif(planename == 'xz'):
        for i in range(0,len(arr3d)):
            for j in range(0,len(arr3d[i])):
                for k in range(0,len(arr3d[i][j])):
                    arr2d[k][i] += arr3d[i][j][k] #TODO: check this
        return arr2d

    elif(planename == 'yz'):
        for i in range(0,len(arr3d)):
            for j in range(0,len(arr3d[i])):
                for k in range(0,len(arr3d[i][j])):
                    arr2d[j][i] += arr3d[i][j][k] #TODO: check this
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

    #mask data outside of given bounds
    gptsx = (x1 <= datadict[dictkey+'_xx']) & (datadict[dictkey+'_xx'] <= x2)
    gptsy = (y1 <= datadict[dictkey+'_yy']) & (datadict[dictkey+'_yy'] <= y2)
    gptsz = (z1 <= datadict[dictkey+'_zz']) & (datadict[dictkey+'_zz'] <= z2)

    goodfieldpts = []
    for i in range(0,len(datadict[dictkey])):
        for j in range(0,len(datadict[dictkey][i])):
            for k in range(0,len(datadict[dictkey][i][j])):
                if(gptsx[i] == True and gptsy[j] == True and gptsz[k] == True):
                    goodpts.append(datadict[dictkey][k][j][i])

    avg = np.average(goodpts)
    return avg
