# analysis.py>

#plasma analysis functions

import numpy as np
import math

def analysis_input(flnm = 'analysisinput.txt'):
    """
    Loads text file that contains relevant FPC analysis parameters

    Parameters
    ----------
    flnm : str, optional
        flnm of analysis input

    Returns
    -------
    path : str
        path to data
    vmax : float
        bounds of FPC analysis in velocity space (assumes square)
    dv : float
        bounds of FPC
    """

    # Get file object
    f = open(flnm, "r")
    # Initialize optional input arguments to None
    dx = None
    xlim = None
    ylim = None
    zlim = None
    resultsdir = 'results/'

    while(True):
        #read next line
        line = f.readline()

        if not line:
        	break

        line = line.strip()
        line = line.split('=')

        if(line[0]=='path'):
            path = str(line[1].split("'")[1])
        elif(line[0]=='vmax'):
            vmax = float(line[1])
        elif(line[0]=='dv'):
            dv = float(line[1])
        elif(line[0]=='numframe'):
            numframe = int(line[1])
        elif(line[0]=='dx'):
            dx = float(line[1])
        elif(line[0]=='xlim'):
            xlim = [float(line[1].split(",")[0]), float(line[1].split(",")[1])]
        elif(line[0]=='ylim'):
            ylim = [float(line[1].split(",")[0]), float(line[1].split(",")[1])]
        elif(line[0]=='zlim'):
            zlim = [float(line[1].split(",")[0]), float(line[1].split(",")[1])]
        elif(line[0]=='resultsdir'):
            resultsdir = str(line[1].split("'")[1])
    f.close()

    #copy this textfile into results directory
    import os

    try:
        isdiff = not(filecmp.cmp(flnm, flnm+resultsdir))
    except:
        isdiff = False #file not found, so can copy it

    if(isdiff):
        print("WANRING: the resultsdir is already used by another analysis input!!!")
        print("Please make a new resultsdir or risk overwriting/ mixing up results")
        return
    else:
        os.system('mkdir '+str(resultsdir))
        os.system('cp '+str(flnm)+' '+str(resultsdir))

    return path,resultsdir,vmax,dv,numframe,dx,xlim,ylim,zlim

#TODO: check shape of Cor (make sure this is a list of 2d projections rather than 3d.)
def compute_energization(Cor,dv):
    """
    Computes energization of velocity signature by integrating over velocity space
    This function assumes a square grid

    Parameters
    ----------
    Cor : 2d array
        x slice of velocity signature
    dv : float
        spacing between velocity grid points

    Returns
    -------
    netE : float
        net energization/ integral of C(x0; vy, vx)
    """

    netE = 0.
    for i in range(0,len(Cor)):
        for j in range(0,len(Cor[i])):
            netE += Cor[i][j]*dv*dv #assumes square grid

    return netE

def compute_energization_over_x(Cor_array,dv):
    """
    Runs compute_energization for each x slice of data

    Parameters
    ----------
    Cor_array : 3d array
        array of x slices of velocity signatures
    dv : float
        spacing between velocity grid points

    Returns
    -------
    C_E_out : 1d array
        array of net energization/ integral of C(x0; vy, vx) for each slice of x
    """

    C_E_out = []
    for k in range(0,len(Cor_array)):
        C_E_out.append(compute_energization(Cor_array[k],dv))

    return np.asarray(C_E_out)

def get_compression_ratio(dfields,upstreambound,downstreambound):
    """
    Find ratio of downstream bz and upstream bz

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    xShock : float
        x position of shock
    """
    numupstream = 0.
    bzsumupstrm = 0.
    numdownstream = 0.
    bzsumdownstrm = 0.

    for i in range(0,len(dfields['bz'])):
        for j in range(0,len(dfields['bz'][i])):
            for k in range(0,len(dfields['bz'][i][j])):
                if(dfields['bz_xx'][k] >= upstreambound):
                    bzsumupstrm += dfields['bz'][i][j][k]
                    numupstream += 1.
                elif(dfields['bz_xx'][k] <= downstreambound):
                    bzsumdownstrm += dfields['bz'][i][j][k]
                    numdownstream += 1.

    bzupstrm = bzsumdownstrm/numupstream
    bzdownstrm = bzsumupstrm/numdownstream

    ratio = bzdownstrm/bzupstrm

    return ratio,bzupstrm,bzdownstrm

def get_num_par_in_box(dparticles,x1,x2,y1,y2,z1,z2):
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
    z1 : float
        lower z bound
    z2 : float
        upper z bound

    Returns
    -------
    totalPtcl : float
        number of particles in box
    """
    gptsparticle = (x1 < dparticles['x1'] ) & (dparticles['x1'] < x2) & (y1 < dparticles['x2']) & (dparticles['x2'] < y2) & (z1 < dparticles['x3']) & (dparticles['x3'] < z2)
    totalPtcl = np.sum(gptsparticle)

    return float(totalPtcl)


def calc_E_crossB(dfields,x1,x2,y1,y2,z1,z2):
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
    z1 : float
        lower z bound
    z2 : float
        lower z bound

    Returns
    -------
    ExBvx : float
        x component of E cross B drift
    ExBvy : float
        y component of E cross B drift
    ExBvz : float
        z component of E cross B drift
    """
    from lib.array_ops import get_average_in_box

    exf = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'ex')
    eyf = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'ey')
    ezf = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'ez')
    bxf = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'bx')
    byf = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'by')
    bzf = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'bz')

    #E cross B / B^2
    magB = bxf**2.+byf**2.+bzf**2.
    ExBvx = (eyf*bzf-ezf*byf)/magB
    ExBvy = -1.*(exf*bzf-ezf*bxf)/magB
    ExBvz = (exf*bzf-ezf*bxf)/magB

    return ExBvx,ExBvy,ExBvz

def calc_Ji_Ei(dfields, dflow, x1, x2, y1, y2, z1, z2):
    """
    Calculates JdotE in given box

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

    from lib.array_ops import get_average_in_box

    if(dfields['Vframe_relative_to_sim'] != dflow['Vframe_relative_to_sim']):
        print("Error, fields and flow are not in the same frame...")
        return

    ux = get_average_in_box(x1, x2, y1, y2, z1, z2, dflow,'ux')
    uy = get_average_in_box(x1, x2, y1, y2, z1, z2, dflow,'uy')
    uz = get_average_in_box(x1, x2, y1, y2, z1, z2, dflow,'uz')
    exf = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'ex')
    eyf = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'ey')
    ezf = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'ez')

    JxEx = ux*exf #TODO: check units (could have definitely omitted q here)
    JyEy = uy*eyf
    JzEz = uz*ezf
    return JxEx, JyEy, JzEz

def get_abs_max_velocity(dparticles):
    """
    Returns the max of the absolute value of each velocity component array

    Parameters
    ----------
    dparticles : dict
        particle data dictionary

    Returns
    -------
    maxspeedx : float
        abs max of vx array
    maxspeedy : float
        abs max of vy array
    maxspeedz : float
        abs max of vz array
    """

    maxspeedx = np.max(np.abs(dparticles['p1']))
    maxspeedy = np.max(np.abs(dparticles['p2']))
    maxspeedz = np.max(np.abs(dparticles['p3']))

    return maxspeedx, maxspeedy, maxspeedz


def check_input(analysisinputflnm,dfields):
    """
    Prints warnings if analysis is set up in an unexpected way

    Parameters
    ----------
    analysisinputflnm : str
        flnm of analysis input
    dfields : dict
        field data dictionary from field_loader
    """

    import sys
    path,resultsdir,vmax,dv,numframe,dx,xlim,ylim,zlim = analysis_input(flnm = analysisinputflnm)

    cellsizexx = dfields['ex_xx'][1]-dfields['ex_xx'][0]
    cellsizeyy = dfields['ex_yy'][1]-dfields['ex_yy'][0]
    cellsizezz = dfields['ex_zz'][1]-dfields['ex_zz'][0]

    #check bounds
    if(xlim is not None):
        if(xlim[0] > xlim[1]):
            print("Error: xlim is set up backwards. xlim[0] should be stricly less than xlim[1]")
            sys.exit()
        if(xlim[0] > xlim[1]-dx):
            print("Error: dx is too small. xlim[0] should not be greater than xlim[0] > xlim[1]-dx")
            sys.exit()
        tolerance = 0.0001
        if((xlim[1]-xlim[0])/dx % 1. > tolerance):
            print("Error: can't divide xlimits into uniformly sized boxes. (xlim[1]-xlim[0])/dx is not a whole number")
            sys.exit()
        if(xlim[0] < (dfields['ex_xx'][0])-cellsizexx/2.):
            print("Error: xlim[0] is outside of simulation box.")
            sys.exit()
        if(xlim[1] > (dfields['ex_xx'][-1])+cellsizexx/2.):
            print("Error: xlim[1] is outside of simulation box.")
            sys.exit()
    if(ylim is not None):
        if(ylim[0] < (dfields['ex_yy'][0])-cellsizeyy/2.):
            print("Error: ylim[0] is outside of simulation box.")
            sys.exit()
        if(ylim[1] > (dfields['ex_yy'][-1])+cellsizeyy/2.):
            print("Error: ylim[1] is outside of simulation box.")
    if(zlim is not None):
        if(zlim[0] < (dfields['ex_zz'][0])-cellsizezz/2.):
            print("Error: zlim[0] is outside of simulation box.")
            sys.exit()
        if(zlim[1] > (dfields['ex_zz'][-1])+cellsizezz/2.):
            print("Error: zlim[1] is outside of simulation box.")



def check_sim_stability(analysisinputflnm,dfields,dparticles,dt):
    """
    Checks max velocity to make sure sim is numerically stable. Prints warnings if it is not

    Parameters
    ----------
    analysisinputflnm : str
        flnm of analysis input
    dfields : dict
        field data dictionary from field_loader
    dparticles : dict
        particle data dictionary
    dt : float
        size of each time step in code units
    """
    path,vmax,dv,numframe,dx,xlim,ylim,zlim = analysis_input(flnm = analysisinputflnm)

    maxsx, maxsy, maxsz = get_abs_max_velocity(dparticles) #Max speed (i.e. max of absolute value of velocity)

    #check if max velocity is numerical stable (make optional to save time)
    #i.e. no particle should move more than 1 cell size in a step
    cellsizexx = dfields['ex_xx'][1]-dfields['ex_xx'][0]
    cellsizeyy = dfields['ex_yy'][1]-dfields['ex_yy'][0]
    cellsizezz = dfields['ex_zz'][1]-dfields['ex_zz'][0]
    if(dt*maxsx > cellsizexx):
        print("Warning: Courant-Friedrich-Lewy condition has been violated in this simulation. (dt*maxsx > cellsizexx)")
    if(dt*maxsy > cellsizeyy):
        print("Warning: Courant-Friedrich-Lewy condition has been violated in this simulation. (dt*maxsy > cellsizeyy)")
    if(dt*maxsz > cellsizezz):
        print("Warning: Courant-Friedrich-Lewy condition has been violated in this simulation. (dt*maxsz > cellsizezz)")

    #check if vmax is reasonable
    if(vmax >= 3.*maxsx or vmax >= 3.*maxsy or vmax >= 3.*maxsz):
        print("Warning: vmax is 3 times larger than the max velocity of any particle. It is computationally wasteful to run FPC analysis in the upper domain of velocity where there are no particles...")

#TODO: check if/force startval/endval to be at discrete location that matches the field positions we have
def deconvolve_for_fft(dfields,fieldkey,startval,endval):
    """
    Fits ramp to line and subtracts line to deconvolve

    Parameters
    ---------
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    startval : float
        start xx position of ramp
    endval : float
        end xx position of ramp

    Returns
    -------
    fieldvalsdeconvolved : 1d array
        value of deconvolved field
    """
    from lib.array_ops import find_nearest

    #grab field in ramp
    startvalidx = ao.find_nearest(startval,dfields[fieldkey])
    endvalidx = ao.find_nearest(endval,dfields[fieldkey])
    fieldinramp = dfields[fieldkey][:,:,startvalidx:endvalidx]
    fieldposinramp = dfields[fieldkey+'_xx'][startvalidx:endvalidx]

    #fit to line (y = mx+b)
    m, b = np.polyfit(tvals, xvals, 1)
    #TODO: this needs to fit to a plane... not a line
    #or maybe we should fit slices to a line...

    fieldvalsdeconvolved = []
    for i in range(0,len(fieldposinramp)):
        decon_field = fieldinramp[i]-m*fieldposinramp[i]-b
        fieldvalsdeconvolved.append(decon_field)


    fieldvalsdeconvolved = np.asarray(fieldvalsdeconvolved)
    print(fieldposinramp)
    print(fieldvalsdeconvolved)

    return fieldvalsdeconvolved

def take_fft1(data,daxis,axis=-1):
    """
    Computes 1d fft on given data

    Parameters
    ----------
    data : array
        data to be transformed
    daxisx0 : float
        cartesian spatial spacing between points
    """

    k = 2.*np.pi*np.fft.fftfreq(len(data),daxis)

    fftdata = np.fft.fft(data,axis=axis)/float(len(data))

    return k, fftdata


def take_fft2(data,daxisx0,daxis1):
    """
    Computes 2d fft on given data

    Parameters
    ----------
    data : 2d array
        2d data to be transformed
    daxisx0 : float
        cartesian spatial spacing between points along 0th axis of data
    daxisx1 : float
        cartesian spatial spacing between points along 1st axis of data

    Returns
    -------
    k0 : 1d array
        wavenumber coordinates corresponding to 0th axis
    k1 : 1d array
        wavenumber coordinates corresponding to 1st axis
    """

    k0 = 2.*np.pi*np.fft.fftfreq(len(data),daxisx0)
    k1 = 2.*np.pi*np.fft.fftfreq(len(data[1]),daxis1)

    fftdata = np.fft.fft2(data)/(float(len(data)*len(data[1])))

    return k0, k1, fftdata

def remove_average_fields_over_yz(dfields):
    """
    Removes yz average from field data i.e. delta_field(x,y,z) = field(x,y,z)-<field(x,y,z)>_(y,z)

    Parameters
    ----------
    dfields : dict
        field data dictionary from flow_loader

    Returns
    -------
    dfieldsfluc : dict
        delta field data dictionary
    """
    from copy import deepcopy

    dfieldfluc = deepcopy(dfields) #deep copy
    dfieldfluc['ex'] = dfieldfluc['ex']-dfieldfluc['ex'].mean(axis=(0,1))
    dfieldfluc['ey'] = dfieldfluc['ey']-dfieldfluc['ey'].mean(axis=(0,1))
    dfieldfluc['ez'] = dfieldfluc['ez']-dfieldfluc['ez'].mean(axis=(0,1))
    dfieldfluc['bx'] = dfieldfluc['bx']-dfieldfluc['bx'].mean(axis=(0,1))
    dfieldfluc['by'] = dfieldfluc['by']-dfieldfluc['by'].mean(axis=(0,1))
    dfieldfluc['bz'] = dfieldfluc['bz']-dfieldfluc['bz'].mean(axis=(0,1))

    return dfieldfluc

def remove_average_flow_over_yz(dflow):
    """
    Removes yz average from field data i.e. delta_field(x,y,z) = field(x,y,z)-<field(x,y,z)>_(y,z)

    Parameters
    ----------
    dfluc : dict
        flow data dictionary from flow_loader

    Returns
    -------
    dflowfluc : dict
        delta field data dictionary
    """
    from copy import deepcopy

    dflowfluc = deepcopy(dflow) #deep copy
    dflowfluc['ux'] = dfieldfluc['ux']-dfieldfluc['ux'].mean(axis=(0,1))
    dflowfluc['uy'] = dfieldfluc['uy']-dfieldfluc['uy'].mean(axis=(0,1))
    dflowfluc['uz'] = dfieldfluc['uz']-dfieldfluc['uz'].mean(axis=(0,1))

    return dflowfluc

def get_average_fields_over_yz(dfields):
    """
    Returns yz average of field i.e. dfield_avg(x,y,z) = <field(x,y,z)>_(y,z)

    TODO: this function doesn't seem to use a deep copy for dfields, i.e. it changes
    dfields. Need to fix this

    Parameters
    ----------
    dfields : dict
        field data dictionary from flow_loader

    Returns
    -------
    dfieldsavg : dict
        avg field data dictionary
    """

    from copy import deepcopy

    dfieldavg = deepcopy(dfields)

    dfieldavg['ex'][:] = dfieldavg['ex'].mean(axis=(0,1))
    dfieldavg['ey'][:] = dfieldavg['ey'].mean(axis=(0,1))
    dfieldavg['ez'][:] = dfieldavg['ez'].mean(axis=(0,1))
    dfieldavg['bx'][:] = dfieldavg['bx'].mean(axis=(0,1))
    dfieldavg['by'][:] = dfieldavg['by'].mean(axis=(0,1))
    dfieldavg['bz'][:] = dfieldavg['bz'].mean(axis=(0,1))

    return dfieldavg

def remove_average_flow_over_yz(dflow):
    """
    Removes yz average from flow data i.e. delta_flow(x,y,z) = flow(x,y,z)-<flow(x,y,z)>_(y,z)

    Parameters
    ----------
    dflow : dict
        flow data dictionary from flow_loader

    Returns
    -------
    dflowfluc : dict
        delta flow data dictionary
    """
    from copy import deepcopy
    dflowfluc = deepcopy(dflow)
    dflowfluc['ux'] = dflowfluc['ux']-dflowfluc['ux'].mean(axis=(0,1))
    dflowfluc['uy'] = dflowfluc['uy']-dflowfluc['uy'].mean(axis=(0,1))
    dflowfluc['uz'] = dflowfluc['uz']-dflowfluc['uz'].mean(axis=(0,1))

    return dflowfluc

def get_delta_fields(dfields,B0):
    """
    Computes the delta between the local B field and the external B field
    """

    from copy import deepcopy

    dfluc = remove_average_fields_over_yz(dfields)
    ddeltafields = deepcopy(dfluc)

    ddeltafields['ex'] = None
    ddeltafields['ey'] = None
    ddeltafields['ez'] = None
    ddeltafields['bx'] = ddeltafields['bx'] - B0[0]
    ddeltafields['by'] = ddeltafields['by'] - B0[1]
    ddeltafields['bz'] = ddeltafields['bz'] - B0[2]

    return ddeltafields

## This is unused and misguided, should remove
# def get_delta_perp_fields(dfields,B0):
#     """
#     Computes the perpendicular component of the delta fields wrt the total magnetic field at each point
#     """
#
#     from copy import deepcopy
#
#     ddeltaperpfields = get_delta_fields(dfields,B0)
#
#     ddeltaperpfields['bx'] = ddeltaperpfields['bx'] - (ddeltaperpfields['bx']*B0[0]+ddeltaperpfields['by']*B0[1]+ddeltaperpfields['bz']*B0[2])/(B0[0]**2+B0[1]**2+B0[2]**2)*B0[0]
#     ddeltaperpfields['by'] = ddeltaperpfields['by'] - (ddeltaperpfields['bx']*B0[0]+ddeltaperpfields['by']*B0[1]+ddeltaperpfields['bz']*B0[2])/(B0[0]**2+B0[1]**2+B0[2]**2)*B0[1]
#     ddeltaperpfields['bz'] = ddeltaperpfields['bz'] - (ddeltaperpfields['bx']*B0[0]+ddeltaperpfields['by']*B0[1]+ddeltaperpfields['bz']*B0[2])/(B0[0]**2+B0[1]**2+B0[2]**2)*B0[2]
#
#     return ddeltaperpfields

def wlt(t,data,w=6,klim=None,retstep=1):
    """
    Peforms wavelet transform using morlet wavelet on data that is a function of t i.e. data(t)

    Paramters
    ---------
    t : 1d array
        independent data array
    data : 1d array
        dependent data array
    w : float, opt
        omega term in morlet wavelet function (relates to the number of humps)
    retstep : int, opt
        spacing between samples of k in returned by wavelet transform
        used mostly to save memory as wavelet transform returns dense sampling of k
    """
    from scipy import signal
    from lib.array_ops import find_nearest

    dt = t[1]-t[0]
    fs = 1./dt

    freq = np.linspace(.01,fs/2.,len(data))
    widths = w*fs / (2*freq*np.pi)

    cwtm = signal.cwt(data, signal.morlet2, widths, w=w)

    k = 2.0*math.pi*freq
    if(klim != None):
        lowerkidx = find_nearest(k,klim[0])
        upperkidx = find_nearest(k,klim[1])
        k = k[loweridx:upperkidx+1]
        cwtm = cwtm[loweridx:upperkidx+1,:]

    if(retstep != 1):
        k=k[::retstep]
        cwtm=cwtm[::retstep]

    return k, cwtm

def _ffttransform_in_yz(dfields,fieldkey):
    """
    Takes f(z,y,x) and computes f(x,kz,ky) using a 2d fft for some given field
    """

    fieldfftsweepoverx = []
    for xxindex in range(0,len(dfields[fieldkey][0][0])):
        fieldslice = np.asarray(dfields[fieldkey])[:,:,xxindex]
        daxis0 = dfields[fieldkey+'_zz'][1]-dfields[fieldkey+'_zz'][0]
        daxis1 = dfields[fieldkey+'_yy'][1]-dfields[fieldkey+'_zz'][0]
        kz, ky, fieldslicefft = take_fft2(fieldslice,daxis0,daxis1)
        fieldfftsweepoverx.append(fieldslicefft)
    fieldfftsweepoverx = np.asarray(fieldfftsweepoverx)

    return kz, ky, fieldfftsweepoverx

def take_ifft2(data):
    """
    Computes 2d ifft on given data


    """

    ifftdata = np.fft.ifft2(data)*(float(len(data)*len(data[1])))

    return ifftdata

def _iffttransform_in_yz(fftdfields,fieldkey):
    """
    Takes f(x,kz,ky) and computes f(x,z,y) using a 2d fft for some given field
    """

    fieldifftsweepoverx = []
    for xxindex in range(0,len(fftdfields[fieldkey])):
        fieldslicefft = np.asarray(fftdfields[fieldkey])[xxindex,:,:]
        fieldslice = take_ifft2(fieldslicefft)
        fieldifftsweepoverx.append(fieldslice)
    fieldifftsweepoverx = np.asarray(fieldifftsweepoverx)

    return fieldifftsweepoverx

#filter fields to specified k
def yz_fft_filter(dfields,ky0,kz0):
    """
    """

    from lib.array_ops import find_nearest
    from copy import deepcopy

    dfieldsfiltered = deepcopy(dfields)

    keys = {'ex','ey','ez','bx','by','bz'}

    #take fft
    for key in keys:
        kz,ky,dfieldsfiltered[key] = _ffttransform_in_yz(dfieldsfiltered,key)

    #filter
    ky0idx = find_nearest(ky, ky0)
    kz0idx = find_nearest(kz, kz0)

    for key in keys:
        for _xxidx in range(0,len(dfieldsfiltered[key])):
            for _kzidx in range(0,len(dfieldsfiltered[key][_xxidx])):
                for _kyidx in range(0,len(dfieldsfiltered[key][_xxidx][_kzidx])):
                    if(not(_kyidx == ky0idx and _kzidx == kz0idx)):
                        dfieldsfiltered[key][_xxidx,_kzidx,_kyidx] = 0

    #take ifft
    for key in keys:
        dfieldsfiltered[key] = _iffttransform_in_yz(dfieldsfiltered,key) #note: input index order is (x,kz,ky) and output is (x,z,y)
        dfieldsfiltered[key] = np.swapaxes(dfieldsfiltered[key], 0, 2) #change index order from (x,z,y) to (z,y,x)
        dfieldsfiltered[key] = np.swapaxes(dfieldsfiltered[key], 0, 1)
        dfieldsfiltered[key] = np.real(dfieldsfiltered[key])

    return dfieldsfiltered



def find_potential_wavemodes(dfields,fieldkey,xpos,cutoffconst=.1):
    """

    """

    from lib.array_ops import find_nearest

    #compute delta fields
    dfieldsfluc = remove_average_fields_over_yz(dfields)

    #spacing in grids, needed to get wavenumber from fft
    daxis0 = dfieldsfluc[fieldkey+'_zz'][1]-dfieldsfluc[fieldkey+'_zz'][0]
    daxis1 = dfieldsfluc[fieldkey+'_yy'][1]-dfieldsfluc[fieldkey+'_yy'][0]

    kz, ky, fieldfftsweepoverx = _ffttransform_in_yz(dfieldsfluc,fieldkey)

    #pick slice nearest to given xpos
    xxidx = find_nearest(dfieldsfluc[fieldkey+'_xx'],xpos)
    fftslice = fieldfftsweepoverx[xxidx,:,:]

    #find field(xpos,ky0,kz0) with norm greater than cutoffconst*max(norm(fftslice))
    fftslice = np.real(fftslice*np.conj(fftslice))/(float(len(kz)*len(ky)))  #convert to norm
    maxnorm = np.max(fftslice)
    kylist = []
    kzlist = []
    prcntmaxlist = []
    for i in range(0,len(kz)):
        for j in range(0,len(ky)):
            if(fftslice[i][j] >= cutoffconst*maxnorm):
                kzlist.append(kz[i])
                kylist.append(ky[j])
                prcntmaxlist.append(fftslice[i][j]/maxnorm)

    #do wavelet transform for each ky, kz
    kxlist = []
    kxplotlist = []
    wltlist = []
    for i in range(0,len(kylist)):
        ky0 = kylist[i]
        ky0idx = find_nearest(ky,ky0)
        kz0 = kzlist[i]
        kz0idx = find_nearest(kz,kz0)

        xkykzdata = fieldfftsweepoverx[:,kz0idx,ky0idx]

        kx, wltdata = wlt(dfieldsfluc[fieldkey+'_xx'],xkykzdata)
        kxplotlist.append(kx)
        wltlist.append(wltdata)

        kxidx = find_nearest(wltdata[:,xxidx],np.max(wltdata[:,xxidx]))
        kxlist.append(kx[kxidx])

    #add negative values for kx
    nkx = len(kxlist)
    for i in range(0,nkx):
        kxlist.append(-1*kxlist[i])
        kylist.append(kylist[i])
        kzlist.append(kzlist[i])

    return kxlist, kylist, kzlist, kxplotlist, wltlist, prcntmaxlist

def is_perp(vec1,vec2,tol=0.001):
    """

    """

    #normalize vector
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)

    dotprod = np.vdot(vec1,vec2) #vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]

    if (abs(dotprod) <= tol):
        return True, dotprod
    else:
        return False, dotprod

def is_parallel(vec1,vec2,tol=0.001):
    """

    """

    #normalize vector
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)

    dotprod = np.vdot(vec1,vec2) #vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]

    if (abs(abs(dotprod)-1.0) <= tol):
        return True, dotprod
    else:
        return False, dotprod


def get_B_yzavg(dfields,xxidx):
    """
    Returns <B(x0,y,z)>_(y,z)
    """

    dfavg = get_average_fields_over_yz(dfields)

    B0x = dfavg['bx'][0,0,xxidx]
    B0y = dfavg['by'][0,0,xxidx]
    B0z = dfavg['bz'][0,0,xxidx]

    return [B0x, B0y, B0z]

def get_B_avg(dfields,xlim,ylim,zlim):

    from lib.array_ops import get_average_in_box

    x1 = xlim[0]
    x2 = xlim[1]
    y1 = ylim[0]
    y2 = ylim[1]
    z1 = zlim[0]
    z2 = zlim[1]

    B0x = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'bx')
    B0y = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'by')
    B0z = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'bz')

    return [B0x, B0y, B0z]

def predict_kx_alfven(ky,kz,B0,delBperp):
    """
    routine that computes what kx would need to be given ky kz for the fluctuation to be alfvenic
    """

    Bx = B0[0]
    By = B0[1]
    Bz = B0[2]
    dBx = delBperp[0]
    dBy = delBperp[1]
    dBz = delBperp[2]
    kx = (-1.+Bz*dBx*ky-Bx*dBz*ky-By*dBx*kz+Bx*dBy*kz)/(Bz*dBy-By*dBz)

    return kx

def _get_perp_component(x1,y1):
    """
    Computes x1perp wrt y1
    """
    x1perpx = x1[0]-(x1[0]*y1[0]+x1[1]*y1[1]+x1[2]*y1[2])/(y1[0]*y1[0]+y1[1]*y1[1]+y1[2]*y1[2])*y1[0]
    x1perpy = x1[1]-(x1[0]*y1[0]+x1[1]*y1[1]+x1[2]*y1[2])/(y1[0]*y1[0]+y1[1]*y1[1]+y1[2]*y1[2])*y1[1]
    x1perpz = x1[2]-(x1[0]*y1[0]+x1[1]*y1[1]+x1[2]*y1[2])/(y1[0]*y1[0]+y1[1]*y1[1]+y1[2]*y1[2])*y1[2]

    return [x1perpx,x1perpy,x1perpz]

def alfven_wave_check(dfields,dfieldfluc,klist,xx,tol=.05):
    """
    Checks if basic properties of an alfven wave are seen at some location in the simulation

    Note: dfields is normally the yz averaged removed fields (e.g. B_fluc(x,y,z) = B(x,y,z)-<B(x,y,z)>_(y,z))
    """

    from lib.array_ops import find_nearest

    xxidx = find_nearest(dfields['bz_xx'],xx)

    #TODO: rename these variables. Data is ordered B(x/kx,kz,ky)
    #TODO: make function that can compute del B/E (kx,ky,kz;xx)
    kz, ky, bxkzkyx = _ffttransform_in_yz(dfieldfluc,'bx')
    kz, ky, bykzkyx = _ffttransform_in_yz(dfieldfluc,'by')
    kz, ky, bzkzkyx = _ffttransform_in_yz(dfieldfluc,'bz')

    kz, ky, exkzkyx = _ffttransform_in_yz(dfieldfluc,'ex')
    kz, ky, eykzkyx = _ffttransform_in_yz(dfieldfluc,'ey')
    kz, ky, ezkzkyx = _ffttransform_in_yz(dfieldfluc,'ez')

    B0 = get_B_yzavg(dfields,xxidx)

    # #get delta perp fields
    # dperpf = get_delta_perp_fields(dfields,B0)

    # check if any of the given k's have the expected properties of an alfven wave
    # i.e. is deltaBperp parallel to kcrossB0, deltaB is perpendicular to B0, and delB is perpendicular to k
    # where deltaB is from
    results = []
    kxexpected = [] #what kx needs to be for the wave to be alfvenic for each k in klist
    delBlist = []
    delElist = []
    for i in range(0,len(klist)):
        #pick a k and compute kperp
        k = klist[i]
        kperp = _get_perp_component(k,B0) #

        #find nearest discrete point in (x,ky,kz) space we have data for
        kyidx = find_nearest(ky,k[1])
        kzidx = find_nearest(kz,k[2])
        kyperpidx = find_nearest(ky,kperp[1])
        kzperpidx = find_nearest(kz,kperp[2])

        if(k[0] < 0):
            _bxkzkyx = np.conj(bxkzkyx)
            _bykzkyx = np.conj(bykzkyx)
            _bzkzkyx = np.conj(bzkzkyx)
            _exkzkyx = np.conj(exkzkyx)
            _eykzkyx = np.conj(eykzkyx)
            _ezkzkyx = np.conj(ezkzkyx)
        else:
            _bxkzkyx = bxkzkyx
            _bykzkyx = bykzkyx
            _bzkzkyx = bzkzkyx
            _exkzkyx = exkzkyx
            _eykzkyx = eykzkyx
            _ezkzkyx = ezkzkyx

        #finalize transform into k space i.e. compute B(kx0,kz0,ky0) from B(x,kz,ky) for k and k perp
        #note: we never have an array B(kx,ky,kz), just that scalar quantities at k0 and kperp0, which we get from
        # the just for B(x,kz,ky) as computing the entire B(kx,ky,kz) array would be computationally expensive.
        # would have to perform wavelet transform for each (ky0,kz0)
        kx, bxkz0ky0kxxx = wlt(dfieldfluc['bx_xx'],_bxkzkyx[:,kzidx,kyidx]) #note kx is that same for all 6 returns here
        kx, bykz0ky0kxxx = wlt(dfieldfluc['by_xx'],_bykzkyx[:,kzidx,kyidx])
        kx, bzkz0ky0kxxx = wlt(dfieldfluc['bz_xx'],_bzkzkyx[:,kzidx,kyidx])
        kx, exkz0ky0kxxx = wlt(dfieldfluc['ex_xx'],_exkzkyx[:,kzidx,kyidx])
        kx, eykz0ky0kxxx = wlt(dfieldfluc['ey_xx'],_eykzkyx[:,kzidx,kyidx])
        kx, ezkz0ky0kxxx = wlt(dfieldfluc['ez_xx'],_ezkzkyx[:,kzidx,kyidx])

        kx, bxperpkz0ky0kxxx = wlt(dfieldfluc['bx_xx'],_bxkzkyx[:,kzperpidx,kyperpidx])
        kx, byperpkz0ky0kxxx = wlt(dfieldfluc['by_xx'],_bykzkyx[:,kzperpidx,kyperpidx])
        kx, bzperpkz0ky0kxxx = wlt(dfieldfluc['bz_xx'],_bzkzkyx[:,kzperpidx,kyperpidx])

        kxidx = find_nearest(kx,np.abs(k[0])) #WLT can not find negative kx. Instead we assume symmetry by taking np.abs
        kxperpidx = find_nearest(kx,np.abs(kperp[0]))

        # if(k[0] < 0): #use reality condition to correct for the fact that we cant compute negative kx using the wlt
        #     bxkz0ky0kxxx = np.conj(bxkz0ky0kxxx)
        #     bykz0ky0kxxx = np.conj(bykz0ky0kxxx)
        #     bzkz0ky0kxxx = np.conj(bzkz0ky0kxxx)
        #     bxperpkz0ky0kxxx = np.conj(bxperpkz0ky0kxxx)
        #     byperpkz0ky0kxxx = np.conj(byperpkz0ky0kxxx)
        #     bzperpkz0ky0kxxx = np.conj(bzperpkz0ky0kxxx)

        kcrossB0 = np.cross(k,B0)
        delB = [bxkz0ky0kxxx[kxidx,xxidx],bykz0ky0kxxx[kxidx,xxidx],bzkz0ky0kxxx[kxidx,xxidx]]
        delE = [exkz0ky0kxxx[kxidx,xxidx],eykz0ky0kxxx[kxidx,xxidx],ezkz0ky0kxxx[kxidx,xxidx]]
        delBlist.append(delB)
        delElist.append(delE)
        delBperp = [bxperpkz0ky0kxxx[kxperpidx,xxidx],byperpkz0ky0kxxx[kxperpidx,xxidx],bzperpkz0ky0kxxx[kxperpidx,xxidx]]

        kxexpected.append(predict_kx_alfven(k[1],k[2],B0,delBperp))

        #results.append([is_parallel(delBperp,kcrossB0,tol=0.1),is_perp(delB,B0,tol=0.1),is_perp(k,delB,tol=.1)])
        testAlfvenval = np.cross(delB,np.cross(k,B0))
        testAlfvenval /= (np.linalg.norm(delB)*np.linalg.norm(np.cross(k,B0)))
        if(np.linalg.norm(testAlfvenval) <= tol):
            belowtol = True
        else:
            belowtol = False
        #belowtol = (testAlfvenval <= tol)

        results.append([(belowtol,np.linalg.norm(testAlfvenval)),is_perp(k,delB,tol=tol)])

    #TODO: consider cleaning up computing delB (maybe move to own function)
    return results, kxexpected, delBlist, delElist

def compute_field_aligned_coord(dfields,xlim,ylim,zlim):
    """
    Computes field aligned coordinate basis using average B0 in provided box
    """
    from lib.array_ops import find_nearest
    from copy import deepcopy

    #TODO: rename vpar,vperp to epar, eperp...
    xavg = (xlim[1]+xlim[0])/2.
    xxidx = find_nearest(dfields['bz_xx'],xavg)
    B0 = get_B_avg(dfields,xlim,ylim,zlim) #***Assumes xlim is sufficiently thin*** as get_B0 uses <B(x0,y,z)>_(yz)=B0

    #get normalized basis vectors
    vparbasis = deepcopy(B0)
    vparbasis /= np.linalg.norm(vparbasis)
    #vperp1basis = _get_perp_component([0,1,0],vparbasis) #TODO: check that this returns something close to 0,1,0 as B0 is approximately in the xz plane (with some fluctuations)
    vperp2basis = np.cross([1,0,0],B0) #x hat cross B0
    tol = 0.005
    _B0 = B0 / np.linalg.norm(B0)
    if(np.linalg.norm([_B0[0]-1.,_B0[1]-0.,_B0[2]-0.]) < tol): #assumes B0 != [-1,0,0]
        print("Warning, B0 is perpendicular to x (typically the shock normal)...")
        print("Already in field aligned coordinates. Returning standard basis...")
        return np.asarray([1,0,0]),np.asarray([0,1,0]),np.asarray([0,0,1])
    vperp2basis /= np.linalg.norm(vperp2basis)
    vperp1basis = np.cross(vparbasis,vperp2basis)
    vperp1basis /= np.linalg.norm(vperp1basis)

    return vparbasis, vperp1basis, vperp2basis

def change_velocity_basis(dfields,dpar,xlim,ylim,zlim,debug=False):
    """
    Converts to field aligned coordinate system
    Parallel direction is along average magnetic field direction at average in limits
    """
    from copy import deepcopy

    if(dfields['Vframe_relative_to_sim'] != dpar['Vframe_relative_to_sim']):
        print("Warning, field data is not in the same frame as particle data...")

    gptsparticle = (xlim[0] <= dpar['x1']) & (dpar['x1'] <= xlim[1]) & (ylim[0] <= dpar['x2']) & (dpar['x2'] <= ylim[1]) & (zlim[0] <= dpar['x3']) & (dpar['x3'] <= zlim[1])

    vparbasis, vperp1basis, vperp2basis = compute_field_aligned_coord(dfields,xlim,ylim,zlim)
    #check orthogonality of these vectors
    if(debug):
        tol = 0.01
        if(np.abs(np.dot(vparbasis,vperp1basis)) > tol or np.abs(np.dot(vparbasis,vperp2basis)) > tol or np.abs(np.dot(vperp1basis,vperp2basis) > tol)):
            print("Warning: orthogonality was not kept...")

    #make change of basis matrix
    _ = np.asarray([vparbasis,vperp1basis,vperp2basis]).T
    changebasismatrix = np.linalg.inv(_)

    #change basis
    dparnewbasis = {}
    dparnewbasis['ppar'],dparnewbasis['pperp1'],dparnewbasis['pperp2'] = np.matmul(changebasismatrix,[dpar['p1'][gptsparticle][:],dpar['p2'][gptsparticle][:],dpar['p3'][gptsparticle][:]])
    # dparnewbasis['x1'] = deepcopy(dpar['x1'][:])
    # dparnewbasis['x2'] = deepcopy(dpar['x2'][:])
    # dparnewbasis['x3'] = deepcopy(dpar['x3'][:])

    #check v^2 for both basis to make sure everything matches
    if(debug):
        for i in range(0,20):
            normnewbasis = np.linalg.norm([dparnewbasis['ppar'][i],dparnewbasis['pperp1'][i],dparnewbasis['pperp2'][i]])
            normoldbasis = np.linalg.norm([dpar['p1'][gptsparticle][i],dpar['p2'][gptsparticle][i],dpar['p3'][gptsparticle][i]])
            if(np.abs(normnewbasis-normoldbasis) > 0.01):
                print('Warning. Change of basis did not converse total energy...')

    return dparnewbasis

def compute_temp_aniso(dparfieldaligned,vmax,dv,V=[0.,0.,0.]):
    """
    Uses 2nd moment of the distribtuion function to compute temp anisotropy

    2nd moment: P = NkT = m integral f(v) (v-V) (v-V) d3V

    Assumes particles with velocity greater than vmax (along any direction) are negligible


    """
    from copy import deepcopy


    if(V == [0.,0.,0.]):
        print("Warning, recieved a drift velocity of V = [0,0,0] when computing temperature anisotropy...")

    dpar = deepcopy(dparfieldaligned)
    dpar['ppar'] = dpar['ppar'] - V[0] #TODO: consider checking frame of particles relative to sim
    dpar['pperp1'] = dpar['pperp1'] - V[1]
    dpar['pperp2'] = dpar['pperp2'] - V[2]

    # bin into f(vpar,vperp)
    vparbins = np.arange(-vmax, vmax+dv, dv)
    vpar = (vparbins[1:] + vparbins[:-1])/2.
    vperpbins = np.arange(-vmax, vmax+dv, dv)
    vperp = (vperpbins[1:] + vperpbins[:-1])/2.

    hist,_ = np.histogramdd((np.sqrt(dpar['pperp2']*dpar['pperp2']+dpar['pperp1']*dpar['pperp1']), dpar['ppar']), bins=[vperpbins, vparbins])

    #integrate by hand
    Pperp = 0.
    for i in range(0, len(vpar)):
        for j in range(0, len(vperp)):
            Pperp += vperp[j]*vperp[j]*hist[j,i]

    Ppar = 0.
    for i in range(0, len(vpar)):
        for j in range(0, len(vperp)):
            Ppar += vpar[i]*vpar[i]*hist[j,i]

    Tperp_over_Tpar = Pperp/Ppar

    return Tperp_over_Tpar

def take_ifft2(data):
    """
    Computes 2d fft on given data

    Parameters
    ----------
    data : 2d array
        2d data to be transformed
    daxisx0 : float
        cartesian spatial spacing between points along 0th axis of data
    daxisx1 : float
        cartesian spatial spacing between points along 1st axis of data

    Returns
    -------

    """

    ifftdata = np.fft.ifft2(data)*(float(len(data)*len(data[1])))

    return ifftdata

def _iffttransform_in_yz(fftdfields,fieldkey):
    """
    Takes f(x,kz,ky) and computes f(x,z,y) using a 2d fft for some given field
    """

    fieldifftsweepoverx = []
    for xxindex in range(0,len(fftdfields[fieldkey])):
        fieldslicefft = np.asarray(fftdfields[fieldkey])[xxindex,:,:]
        fieldslice = take_ifft2(fieldslicefft)
        fieldifftsweepoverx.append(fieldslice)
    fieldifftsweepoverx = np.asarray(fieldifftsweepoverx)

    return fieldifftsweepoverx

def yz_fft_filter(dfields,ky0,kz0):
    """
    Filter fields to specified k
    """
    from copy import deepcopy
    from lib.array_ops import find_nearest

    dfieldsfiltered = deepcopy(dfields)

    keys = {'ex','ey','ez','bx','by','bz'}

    #take fft
    for key in keys:
        kz,ky,dfieldsfiltered[key] = _ffttransform_in_yz(dfieldsfiltered,key)

    #filter
    ky0idx = find_nearest(ky, ky0)
    kz0idx = find_nearest(kz, kz0)
    for key in keys:
        for _xxidx in range(0,len(dfieldsfiltered[key])):
            for _kzidx in range(0,len(dfieldsfiltered[key][_xxidx])):
                for _kyidx in range(0,len(dfieldsfiltered[key][_xxidx][_kzidx])):
                    if(not(_kyidx == ky0idx and _kzidx == kz0idx)):
                        dfieldsfiltered[key][_xxidx,_kzidx,_kyidx] = 0

    #take ifft
    for key in keys:
        dfieldsfiltered[key] = _iffttransform_in_yz(dfieldsfiltered,key) #note: input index order is (x,kz,ky) and output is (x,z,y)

        dfieldsfiltered[key] = np.swapaxes(dfieldsfiltered[key], 0, 2) #change index order from (x,z,y) to (z,y,x)
        dfieldsfiltered[key] = np.swapaxes(dfieldsfiltered[key], 0, 1)

        dfieldsfiltered[key] = np.real(dfieldsfiltered[key])


    return dfieldsfiltered

def transform_field_to_kzkykxxx(ddict,fieldkey,retstep=12):
    """
    Takes fft in y and z and wavelet transform in x of given field/ flow.

    E.g. takes B(z,y,x) and computes B(kz,ky,kx;x)

    Parameters
    ----------
    ddict : dict
        field or flow data dictionary
    fieldkey : str
        name of field you want to transform (ex, ey, ez, bx, by, bz, ux, uy, uz)
    retstep : int, opt
        spacing between samples of k in returned by wavelet transform
        used mostly to save memory as wavelet transform returns dense sampling of k

    Returns
    -------
    """

    kz, ky, fieldkzkyx = _ffttransform_in_yz(ddict,fieldkey)

    nxx = len(ddict[fieldkey+'_xx'])
    nkx = int(len(ddict[fieldkey+'_xx'])/retstep) #warning: this is hard coded to match wlt function output size
    nky = len(ky)
    nkz = len(kz)
    fieldkzkykxxx = np.zeros((nkz,nky,2*nkx,nxx),dtype=np.complex_)

    for kyidx in range(0,len(ky)):
        for kzidx in range(0,len(kz)):
            positivekx, rightfieldkz0ky0kxxx = wlt(ddict[fieldkey+'_xx'],fieldkzkyx[:,kzidx,kyidx],retstep=retstep)
            negativekx, leftfieldkz0ky0kxxx = wlt(ddict[fieldkey+'_xx'],np.conj(fieldkzkyx[:,kzidx,kyidx]),retstep=retstep)
            fieldkzkykxxx[kzidx,kyidx,nkx:,:] = rightfieldkz0ky0kxxx[:,:]
            fieldkzkykxxx[kzidx,kyidx,0:nkx,:] = np.flip(leftfieldkz0ky0kxxx[:,:], axis=0)

    negativekx *= -1
    negativekx = np.sort(negativekx)
    kx = np.concatenate([negativekx,positivekx])

    return kz, ky, kx, fieldkzkykxxx
