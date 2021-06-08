# sanityfunctions.py>

#functions related to 'sanity checks' of simulation

def calc_E_crossB(dfields,x1,x2,y1,y2):
    """
    Computes E cross B in some region.

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    x1 : float
        lower x bound
    x2 : float
        upper x bound
    y1 : float
        lower y bound
    y2 : float
        upper y bound

    Returns
    -------
    ExBvx : float
        x component of E cross B drift
    ExBvy : float
        y component of E cross B drift
    ExBvz : float
        z component of E cross B drift
    """
    exf = getfieldaverageinbox(0., 0., x1, x2, y1, y2, {}, dfields, 'ex')
    eyf = getfieldaverageinbox(0., 0., x1, x2, y1, y2, {}, dfields, 'ey')
    ezf = getfieldaverageinbox(0., 0., x1, x2, y1, y2, {}, dfields, 'ez')
    bxf = getfieldaverageinbox(0., 0., x1, x2, y1, y2, {}, dfields, 'bx')
    byf = getfieldaverageinbox(0., 0., x1, x2, y1, y2, {}, dfields, 'by')
    bzf = getfieldaverageinbox(0., 0., x1, x2, y1, y2, {}, dfields, 'bz')

    #E cross B / B^2
    magB = bxf**2.+byf**2.+bzf**2.
    ExBvx = (eyf*bzf-ezf*byf)/magB
    ExBvy = -1.*(exf*bzf-ezf*bxf)/magB
    ExBvz = (exf*bzf-ezf*bxf)/magB

    return ExBvx,ExBvy,ExBvz
