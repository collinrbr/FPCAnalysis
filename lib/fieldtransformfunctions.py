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
    dfieldslor = copy(dfields) #deep copy

    dfieldslor['ex'] = dfields['ex']
    dfieldslor['ey'] = dfields['ey']-vx*dfields['bz']
    dfieldslor['ez'] = dfields['ez']+vx*dfields['by']
    dfieldslor['bx'] = dfields['bx']
    dfieldslor['by'] = dfiedls['by']#assume v/c^2 is small
    dfieldslor['bz'] = dfiedls['bz']#assume v/c^2 is small

    return dfieldslor

def getcompressionration(dfields,xShock):
    """
    Find ratio of downstream bz and upstream bz

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    xShock : float
        x position of shock
    """
    bzsumdownstrm = 0.
    bzsumupstrm = 0.

    for i in range(0,len(dfields['bz'])):
        for j in range(0,len(dfields['bz'][i])):
            for k in range(0,len(dfields['bz'][i][j])):
                if(dfields['bz_xx'][k] > xShock):
                    bzsumupstrm += dfields['bz'][i][j][k]
                else:
                    bzsumdownstrm += dfields['bz'][i][j][k]

    return bzsumdownstrm/bzsumupstrm
