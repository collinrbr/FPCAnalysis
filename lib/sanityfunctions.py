# sanityfunctions.py>

#functions related to 'sanity checks' of simulation

import numpy as np

def getflowaverageinbox(x1,x2,y1,y2,dflow,flowkey):
    """
    Find average flow in provided provided bounds

    Parameters
    ----------
    x1 : float
        lower bound in xx space that you want to count
    x2 : float
        upper bound in xx space that you want to count
    y1 : float
        lower bound in xx space that you want to count
    y2 : float
        upper bound in xx space that you want to count
    dflow : dict
        flow data dictionary from flow_loader
    flowkey : str
        name of flow you want to plot (ux, uy, uz)

    Returns
    -------
    avgflow : float
        average flow in box

    """

    gflowptsx = (x1 <= dflow[flowkey+'_xx'])  & (dflow[flowkey+'_xx'] <= x2)
    gflowptsy = (y1 <= dflow[flowkey+'_yy']) & (dflow[flowkey+'_yy'] <= y2)

    goodflowpts = []
    for i in range(0,len(dflow[flowkey+'_xx'])):
        for j in range(0,len(dflow[flowkey+'_yy'])):
            for k in range(0,len(dflow[flowkey+'_zz'])):
                if(gflowptsx[i] == True and gflowptsy[j] == True):
                    goodflowpts.append(dflow[flowkey][k][j][i])

    avgflow = np.average(goodflowpts)
    return avgflow

def getfieldaverageinbox(x1, x2, y1, y2, dfields, fieldkey):
    """
    Find average field based on provided bounds

    Parameters
    ----------
    x1 : float
        lower bound in xx space that you want to count
    x2 : float
        upper bound in xx space that you want to count
    y1 : float
        lower bound in xx space that you want to count
    y2 : float
        upper bound in xx space that you want to count
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)

    Returns
    -------
    avgfield : float
        average field in box

    """
    gfieldptsx = (x1 <= dfields[fieldkey+'_xx'])  & (dfields[fieldkey+'_xx'] <= x2)
    gfieldptsy = (y1 <= dfields[fieldkey+'_yy']) & (dfields[fieldkey+'_yy'] <= y2)

    goodfieldpts = []
    for i in range(0,len(dfields['ex_xx'])):
        for j in range(0,len(dfields['ex_yy'])):
            for k in range(0,len(dfields['ex_zz'])):
                if(gfieldptsx[i] == True and gfieldptsy[j] == True):
                    goodfieldpts.append(dfields[fieldkey][k][j][i])


    #debug
    #print("numgridpts sampled: " + str(len(goodfieldpts)))


    avgfield = np.average(goodfieldpts)
    return avgfield

def getnumparticlesinbox(dparticles,x1,x2,y1,y2):
    """
    Counts the number of particles in a box

    Parameters
    ----------
    dparticles : dict
        particle data dictionary
    x1 : float
        lower x bound
    x2 : float
        upper x bound
    y1 : float
        lower y bound
    y2 : float
        upper y bound

    """
    gptsparticle = (x1 < dparticles['x1'] ) & (dparticles['x1'] < x2) & (y1 < dparticles['x2']) & (dparticles['x2'] < y2)
    totalPtcl = np.sum(gptsparticle)

    return totalPtcl

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
    exf = getfieldaverageinbox(x1, x2, y1, y2, dfields, 'ex')
    eyf = getfieldaverageinbox(x1, x2, y1, y2, dfields, 'ey')
    ezf = getfieldaverageinbox(x1, x2, y1, y2, dfields, 'ez')
    bxf = getfieldaverageinbox(x1, x2, y1, y2, dfields, 'bx')
    byf = getfieldaverageinbox(x1, x2, y1, y2, dfields, 'by')
    bzf = getfieldaverageinbox(x1, x2, y1, y2, dfields, 'bz')

    #E cross B / B^2
    magB = bxf**2.+byf**2.+bzf**2.
    ExBvx = (eyf*bzf-ezf*byf)/magB
    ExBvy = -1.*(exf*bzf-ezf*bxf)/magB
    ExBvz = (exf*bzf-ezf*bxf)/magB

    return ExBvx,ExBvy,ExBvz

def calc_JdotE(dfields, dflow, x1, x2, y1, y2):
    """
    Calculated JdotE in given box

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    dflow : dict
        flow data dictionary from flow_loader
    x1 : float
        lower x bound
    x2 : float
        upper x bound
    y1 : float
        lower y bound
    y2 : float
        upper y bound
    """

    ux = getflowaverageinbox(x1, x2, y1, y2,dflow,'ux')
    uy = getflowaverageinbox(x1, x2, y1, y2,dflow,'uy')
    uz = getflowaverageinbox(x1, x2, y1, y2,dflow,'uz')
    exf = getfieldaverageinbox(x1, x2, y1, y2, dfields, 'ex')
    eyf = getfieldaverageinbox(x1, x2, y1, y2, dfields, 'ey')
    ezf = getfieldaverageinbox(x1, x2, y1, y2, dfields, 'ez')

    JdE = ux*exf+uy*eyf+uz*ezf #TODO: check units (have definitely omitted q here)
    return JdE
