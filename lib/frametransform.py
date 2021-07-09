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
