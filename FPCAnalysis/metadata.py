# metadata.py>

#functions related to creating metadata for MLA algo

import numpy as np

def build_metadata(xlim,dx,startval,endval):
    """
    Builds binary metadata for SDA

    Parameters
    ----------
    xlim : 2d array
        [lowerxbound, upperxbound]
    startval : float
        lower bound for metadata = 1 (units of di)
    endval : float
        upper bound for metadata = 1 (units of di)

    Returns
    -------
    metadata : array
        binary metadata
    """

    from FPCAnalysis.array_ops import find_nearest

    if(startval > endval):
        print("Error, startval should be less than end val...")
        return []

    array = np.arange(xlim[0],xlim[1],dx)

    startidx = find_nearest(array, startval)
    endidx = find_nearest(array, endval)

    metadata = np.zeros(len(array))
    metadata = metadata.astype(int)
    
    if(startval >= xlim[0] and startval <= xlim[1] or endval >= xlim[0] and endval <= xlim[1]):
        for i in range(0,len(metadata)):
            if(i >= startidx and i <= endidx):
                metadata[i] = 1
    return metadata
