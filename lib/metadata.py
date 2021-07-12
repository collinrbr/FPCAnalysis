# metadata.py>

#functions related to creating metadata for MLA algo

import numpy as np

def build_metadata(dfields,startval,endval):
    """
    Builds binary metadata for SDA

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    startval : float
        lower bound for metadata = 1 (units of di)
    endval : float
        upper bound for metadata = 1 (units of di)

    Returns
    -------
    metadata : array
        binary metadata
    """

    from lib.analysisfunctions import find_nearest

    startidx = find_nearest(dfields['ex_xx'], startval)
    endidx = find_nearest(dfields['ex_xx'], endval)

    metadata = np.zeros(len(dfields['ex_xx']))
    metadata = metadata.astype(int)

    for i in range(0,len(metadata)):
        if(i >= startidx and i <= endidx):
            metadata[i] = 1
    return metadata
