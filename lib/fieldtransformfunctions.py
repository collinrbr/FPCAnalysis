# fieldtransformfunctions.py>

#functions related to lorentz transforming field and computing shock veloicty


def lorentz_transform_vx(dfields,vx):
    """
    Takes lorentz transform where V=(vx,0,0)
    TODO: check if units work (in particular where did gamma go)

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    vx : float
        boost velocity along x
    """
    from copy import copy

    dfieldslor = copy(dfields) #deep copy

    dfieldslor['ex'] = dfields['ex']
    dfieldslor['ey'] = dfields['ey']-vx*dfields['bz']
    dfieldslor['ez'] = dfields['ez']+vx*dfields['by']
    dfieldslor['bx'] = dfields['bx']
    dfieldslor['by'] = dfields['by']#assume v/c^2 is small
    dfieldslor['bz'] = dfields['bz']#assume v/c^2 is small

    return dfieldslor

def getcompressionratio(dfields,upstreambound,downstreambound):
    """
    Find ratio of downstream bz and upstream bz

    Note, typically upstreambound != downstream bound. We should exclude the
    the fields within the shock.

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    downstreambound : float
        x position of the end of the upstream position
    upstreambound : float
        x position of the end of the downstream position

    Returns
    -------
    ratio : float
        compression ratio computed from compression of bz field
    bzdownstrm : float
        average bz downstream
    bzupstrm : float
        average bz upstream
    """

    if(upstreambound < downstreambound):
        print('Error, upstream bound should not be less than downstream bound...')
        return

    bzsumdownstrm = 0.
    numupstreampoints = 0.
    bzsumupstrm = 0.
    numdownstreampoints = 0.

    for i in range(0,len(dfields['bz'])):
        for j in range(0,len(dfields['bz'][i])):
            for k in range(0,len(dfields['bz'][i][j])):
                if(dfields['bz_xx'][k] >= upstreambound):
                    bzsumupstrm += dfields['bz'][i][j][k]
                    numupstreampoints += 1.
                elif(dfields['bz_xx'][k] <= downstreambound):
                    bzsumdownstrm += dfields['bz'][i][j][k]
                    numdownstreampoints += 1.

    bzdownstrm = bzsumdownstrm/numdownstreampoints
    bzupstrm = bzsumupstrm/numupstreampoints
    ratio = bzdownstrm/bzupstrm

    return ratio, bzdownstrm, bzupstrm
