# analysis.py>

#plasma analysis functions

import numpy as np
import math

def analysis_input(flnm = 'analysisinput.txt',make_resultsdir=True):
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
    resultsdir : str
        path to directory where output is saved
    vmax : float
        bounds of FPC analysis in velocity space (assumes square)
    dv : float
        bounds of FPC
    numframe : int
        frame of the simulation to be analyzed
    dx : float
        width of integration box
    xlim : array
        xx bounds of analysis (i.e. where the sweep starts and stops)
    ylim : array
        yy bounds of each integration box
    zlim : array
        zz bounds of each integration box
    """

    # Get file object
    f = open(flnm, "r")
    # Initialize optional input arguments to None
    dx = None
    xlim = None
    ylim = None
    zlim = None
    vmax = None
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

    if(make_resultsdir):
        #copy this textfile into results directory
        import os

        try:
            isdiff = not(filecmp.cmp(flnm, flnm+resultsdir))
        except:
            isdiff = False #file not found, so can copy it

        if(isdiff):
            print("WARNING: the resultsdir is already used by another analysis input!!!")
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
    upstreambound : float
        x position of shock end of upstream
    downstreambound : float
        x position of shock end of upstream

    Returns
    -------
    ratio : float
        compression ratio
    bzupstrm : float
        avg bz upstream
    bzdownstrm : float
        avg bz downstream
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

    bzdownstrm = bzsumdownstrm/numupstream
    bzupstrm = bzsumupstrm/numdownstream

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
    Calculates JdotE components in given box

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

    JxEx = ux*exf #TODO: check units (could have omitted q here)
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

def get_max_speed(dparticles):
    """
    Returns the max of the absolute value of each velocity component array

    Parameters
    ----------
    dparticles : dict
        particle data dictionary

    Returns
    -------
    maxspeed : float
        maximum speed of any particles
    """

    maxspeed = np.sqrt(np.max(dparticles['p1']**2.+dparticles['p2']**2.+dparticles['p3']**2.))

    return maxspeed

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
    startvalidx = find_nearest(startval,dfields[fieldkey])
    endvalidx = find_nearest(endval,dfields[fieldkey])
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

def remove_average_fields_over_yz(dfields, Efield_only = False):
    """
    Removes yz average from field data i.e. delta_field(x,y,z) = field(x,y,z)-<field(x,y,z)>_(y,z)

    Parameters
    ----------
    dfields : dict
        field data dictionary from flow_loader
    Efield_only : bool, opt
        if true, returns total bfield

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

    if(not(Efield_only)):
        dfieldfluc['bx'] = dfieldfluc['bx']-dfieldfluc['bx'].mean(axis=(0,1))
        dfieldfluc['by'] = dfieldfluc['by']-dfieldfluc['by'].mean(axis=(0,1))
        dfieldfluc['bz'] = dfieldfluc['bz']-dfieldfluc['bz'].mean(axis=(0,1))

    return dfieldfluc

def remove_flow_over_yz(dflow):
    """
    Removes yz average from field data i.e. delta_field(x,y,z) = field(x,y,z)-<field(x,y,z)>_(y,z)

    Note; this was named remove__flow_over_yz (note the extra '_') at some point, not sure if it was called by this name anywhere

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

def get_average_fields_over_yz(dfields, Efield_only = False):
    """
    Returns yz average of field i.e. dfield_avg(x,y,z) = <field(x,y,z)>_(y,z)

    TODO: this function doesn't seem to use a deep copy for dfields, i.e. it changes
    dfields. Need to fix this

    Parameters
    ----------
    dfields : dict
        field data dictionary from flow_loader
    Efield_only : bool, opt
        if true, returns total bfield

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

    if(not(Efield_only)):
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
    Computes the delta between the local average B field and the external B field

    WARNING: this function is only valid for small ranges in position space as B0 typically changes rapidly throughout a simulation

    TODO: make get_delta_fields and related functions compatable with 1d 2d or 3d sims

    Parameters
    ----------
    dfields : dict
        flow data dictionary from field
    B0 : array
        [Bx,By,Bz] average field
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

    print('WARNING: this function is only valid for small ranges in position space as B0 typically changes rapidly throughout a simulation...')
    print("WARNING: this function is not the same remove_average_flow_over_yz() and is typically not as useful. Make sure this operation is correct for your analysis...")

    return ddeltafields

def wlt(t,data,w=6,klim=None,retstep=1,powerTwoSpace=False):
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
    powerTwoSpace : bool, optimize
        if true, will space widths using powers of two (not well tested, avoid use)

    #TODO: add returns
    """
    from scipy import signal
    from lib.array_ops import find_nearest

    dt = t[1]-t[0]

    if(powerTwoSpace): #from Torrence et al 1997 (practical guide to wavelet analysis) (suggested to use different spacing)
                       #still WIP, use caution when using
        s0 = 1.*dt
        J = len(data)
        delta_j = np.log2(len(data)*dt/s0)/(J) #guess for now
        print('delta_j, ', delta_j)
        widths = []
        for _j in range(J-1,-1,-1):
            widths.append(s0*2.**(_j*delta_j))
        widths = np.asarray(widths)
        freq = w/(2*widths*np.pi*s0)

    else: #default from scipy's example
        #TOOD: 1/.01 should stricly be larger than fs/2
        fs = 1./dt
        freq = np.linspace(dt/10,fs/4.,int(len(data)/retstep))
        widths = w*fs / (2*freq*np.pi)

    #print(widths)

    try:
        cwtm = signal.cwt(data, signal.morlet2, widths, w=w)
    except:
        print("Warning, it seems we will have to attempt to manually import needed functions from signal from scipy...")
        print("Attempting to do so...")
        from lib.supp import cwt #manually rename scipy signal library and put in folder named supplementary in libraries folder to attempt to load needed library
        from lib.supp import morlet2

        cwtm = cwt(data, morlet2, widths, w=w)

    k = 2.0*math.pi*freq
    if(klim != None):
        lowerkidx = find_nearest(k,klim[0])
        upperkidx = find_nearest(k,klim[1])
        k = k[loweridx:upperkidx+1]
        cwtm = cwtm[loweridx:upperkidx+1,:]

    # if(retstep != 1):
    #     k=k[::retstep]
    #     cwtm=cwtm[::retstep]

    #normalize
    for _idx in range(0,len(cwtm[:,0])):
        cwtm[_idx,:] *= (np.abs(k[_idx]))**0.5 #TODO: double check if this should be abs value


    return k, cwtm

def iwlt_noscale(t,k,cwtdata):
    """
    Computes inverse wavelet transform, without preserving scale
    i.e given f(t) with w.l.t. W{f(t)}, this function will return A*f(t) = W^(-1){W{f(t)}} where A is some unknown constant

    This function is meant to only be used until we learn how to implement a WLT that preserves this scale.

    Parameters
    ----------
    t : array
        time/position axis of wavelet transform
    k : array
        freq/wavenumber axis of wavelet transform
    cwtdata : 2d array
        wavelet transform data from wlt() function

    Returns
    -------
    f_t : 1d array
        reconstructed original signal computed using inverse wavelet transform
        note this signal will almost always be off by some constant factor
        WARNING: some signals can not be reconstructed well
    """

    N = len(t)
    J = len(k)

    f_t = []
    for _n in range(0,N):
        f_ti = 0.
        for _kidx in range(0,J):
            f_ti += np.real(cwtdata[_kidx,_n])/k[_kidx]**1.
        f_t.append(f_ti)
    f_t = np.asarray(f_t)

    return f_t

def force_find_iwlt_scale(t,w=6,retstep=1):
    """
    Finds the inverse wlt scale empirically for a morlet wave

    Out inverse wlt function is off by some reconstruction constant, this attempts to compute that constant

    TODO: compute this value analytically

    Parameters
    ----------
    t : array
        time/position axis of wavelet transform
    w : float, optional
        width parameter of morlet wave
    retstep : float, optional
        spacing using when computing wlt transform
        see wlt() documentation

    Returns
    -------
    ratio : float
        constant ratio between original and reconstructed signal
    """
    t = np.asarray(t)
    dt = t[1]-t[0]
    _yy = np.cos(10.*dt*t)


    k, cwt = wlt(t,_yy,retstep=retstep)
    _yyreconstructed = iwlt_noscale(t,k,cwt)

    ratio = np.sum(np.abs(_yy))/np.sum(np.abs(_yyreconstructed)) #take ratio of integrals of abs(data)

    return ratio

def iwlt(t,k,cwtdata,w=6):
    """
    Parameters
    ----------
    t : array
        time/position axis of wavelet transform
    k : array
        freq/wavenumber axis of wavelet transform
    cwtdata : 2d array
        wavelet transform data from wlt() function
    w : float, optional
        width parameter of morlet wave

    Returns
    -------
    f_t : 1d array
        reconstructed original signal computed using inverse wavelet transform
        WARNING: some signals can not be reconstructed well
    """
    #TODO: stop using force_find_iwlt scale

    f_t = iwlt_noscale(t,k,cwtdata)
    retstep = int(len(t)/len(k))
    ratio = force_find_iwlt_scale(t,w=6,retstep=retstep)
    f_t = ratio*f_t

    return f_t

def midpass_wlt_filter(t,data,k_filter_center,k_filter_width):
    """
    Midpass filter using wlt

    Paramters
    --------
    t : array
        time/position axis of wavelet transform
    data : 1d array
        data to be filtered
    k_filter_center : float
        midpass center in wavenumber/freq space
    k_filter_width : float
        midpass width in wavenumber/freq space

    Returns
    -------
    data : 1d array
        filtered data
    """
    from lib.array_ops import find_nearest
    k, cwt = wlt(t,data)

    kidx_upper = find_nearest(k,k_filter_center+k_filter_width/2.)
    kidx_lower = find_nearest(k,k_filter_center-k_filter_width/2.)

    for _tidx in range(0,len(cwt[0,:])):
        for _kidx in range(0,len(cwt[:,0])):
            if(not(_kidx <= kidx_upper and _kidx >=kidx_lower)):
                 cwt[_kidx,_tidx] = 0.

    data = iwlt(t,k,cwt)

    return data

def _ffttransform_in_yz(dfields,fieldkey):
    """
    Takes f(z,y,x) and computes f(x,kz,ky) using a 2d fft for some given field

    Parameters
    ----------
    dfields : dict
        dict from field_loader
    fieldkey : str
        name of field you want to transform (ex, ey, ez, bx, by, bz, ux, uy, uz)

    Returns
    -------
    ky/kz : 1d array
        coordinates in wavenumber space
    fieldfftsweepoverx : 3d array
        f(x,kz,ky) for specified field f
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

    Parameters
    ----------
    data : 2d array
        data in freq space

    Returns
    -------
    ifftdata : 2d array
        data in cartesian space
    """

    ifftdata = np.fft.ifft2(data)*(float(len(data)*len(data[1])))

    return ifftdata

def _iffttransform_in_yz(fftdfields,fieldkey):
    """
    Takes f(x,kz,ky) and computes f(x,z,y) using a 2d fft for some given field

    Parameters
    ----------
    fftdfields : dict
        dict of fields that have been fft transformed in yz
    fieldkey : str
        name of field you want to inverse transform (ex, ey, ez, bx, by, bz, ux, uy, uz)
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
    Filter fields at exactly specified k

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader
    ky0/kz0 : float
        value of midpass location

    Returns
    -------
    dfieldsfiltered : dict
        filted field dict
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
        dfieldsfiltered[key] = np.swapaxes(dfieldsfiltered[key], 0, 2) #change index order from (x,z,y) to (y,z,x)
        dfieldsfiltered[key] = np.swapaxes(dfieldsfiltered[key], 0, 1) #change index order from (y,z,x) to (z,y,x)
        dfieldsfiltered[key] = np.real(dfieldsfiltered[key])

    return dfieldsfiltered

def xyz_wlt_fft_filter(kz,ky,kx,xx,bxkzkykxxx,bykzkykxxx,bzkzkykxxx,
                exkzkykxxx,eykzkykxxx,ezkzkykxxx,
                kx_center0,kx_width0,ky0,kz0,dontfilter=False):
    """
    Mid pass filter in x y and z
    Uses a single wavenumber in y and z and a small range in x to filter

    We assume the user already has axis to the fields in freq space as it takes a long time to compute

    Note: some signals are difficult to filter as the inverse wavelet transform can not reconstruct the original singal well

    Parameters
    ----------
    kz/ky/kx/xx : 1d array
        coordinate arrays
    **kzkykxxx : 4d array
        fields transformed by fft in yy/zz and wlt in xx
    k_filter_center : float
        midpass center in wavenumber/freq space (related to xx direction)
    k_filter_width : float
        midpass width in wavenumber/freq space (related to xx direction)
    ky0/kz0 : float
        value of midpass location
    dontfilter : bool, optional
        when true, will skip filter to see how well the original signal can be rebuilt
        used to debug
    """
    # from copy import deepcopy
    #
    # dfieldsfiltered = deepcopy(dfields)

    from lib.array_ops import find_nearest

    if(kx_center0 <= 0 or kx_center0-kx_width0/2. <=0):
        print('Warning, at least part of the mid pass filter is negative (i.e. kx_center <= 0 or kx_center0-kx_width0/2. <=0).')#TODO: implement
        print('This function does not yet have the ability to filter negative kx values.')
        print('Breaking call...')
        return

    keys = {'ex','ey','ez','bx','by','bz'}
    freq_space = {'ex':exkzkykxxx,'ey':eykzkykxxx,'ez':ezkzkykxxx,'bx':bxkzkykxxx,'by':bykzkykxxx,'bz':bzkzkykxxx}
    ky0idx = find_nearest(ky, ky0)
    kz0idx = find_nearest(kz, kz0)

    #make dictionary
    filteredfields = {} #TODO: use consistent naming between similar functions
    for key in keys:
        filteredfields[key] = np.zeros((len(freq_space[key][:,0,0,0]),len(freq_space[key][0,:,0,0]),len(freq_space[key][0,0,0,:]))) #makes empty arrays of length of zz by yy by xx (warning, length of kx is technically arbitrary as it is the product of the wavelet transform)

    #to test/debug inverse transform, we inverse transform without filterings
    if(not(dontfilter)):
        for key in keys:
            #filter xx
            for _kzidx in range(0,len(freq_space[key][:,0,0,0])):
                for _kyidx in range(0,len(freq_space[key][_kzidx,:,0,0])):
                        for _kxidx in range(0,len(freq_space[key][_kzidx,_kyidx,:,0])):
                            for _xxidx in range(0,len(freq_space[key][_kzidx,_kyidx,_kxidx,:])):
                                if(not(_kidx <= kidx_upper and _kidx >=kidx_lower)):
                                     freq_space[key][_kzidx,_kyidx,_kxidx,_xxidx] = 0.
                                if(not(_kyidx == ky0idx and _kzidx == kz0idx)):
                                    freq_space[key][_kzidx,_kyidx,_kxidx,_xxidx]  = 0.

    #inverse transform
    for key in keys:
        #take iwlt (inverse transform in xx direction)
        nkx = int(len(freq_space[key][0,0,:,0])/2) #need to rebuild signal from only positive kxs
        for _kzidx in range(0,len(freq_space[key][:,0,0,0])):
            for _kyidx in range(0,len(freq_space[key][_kzidx,:,0,0])):
                filteredfields[key][_kzidx,_kyidx,:]  = iwlt(xx,kx[nkx:],freq_space[key][_kzidx,_kyidx,nkx:,:])

        #take ifft2 (inverse transform in yy/zz direction)
        filteredfields[key] = np.swapaxes(filteredfields[key], 0, 2) #change index order from (kz,ky,x) to (x,ky,kz)
        filteredfields[key] = np.swapaxes(filteredfields[key], 1, 2) #change index order from  (x,ky,kz) to (x,kz,ky)
        filteredfields[key] = _iffttransform_in_yz(filteredfields,key) #note: input index order is (x,kz,ky) and output is (x,z,y)
        filteredfields[key] = np.swapaxes(filteredfields[key], 0, 2) #change index order from (x,z,y) to (y,z,x)
        filteredfields[key] = np.swapaxes(filteredfields[key], 0, 1) #change index order from (y,z,x) to (z,y,x)
        filteredfields[key] = np.real(filteredfields[key])

    return filteredfields

# def find_potential_wavemodes(dfields,fieldkey,xpos,cutoffconst=.1):
#     """
#     This function didnt lead to useful results, and is no longer used...
#     """
#
#     from lib.array_ops import find_nearest
#
#     #compute delta fields
#     dfieldsfluc = remove_average_fields_over_yz(dfields)
#
#     #spacing in grids, needed to get wavenumber from fft
#     daxis0 = dfieldsfluc[fieldkey+'_zz'][1]-dfieldsfluc[fieldkey+'_zz'][0]
#     daxis1 = dfieldsfluc[fieldkey+'_yy'][1]-dfieldsfluc[fieldkey+'_yy'][0]
#
#     kz, ky, fieldfftsweepoverx = _ffttransform_in_yz(dfieldsfluc,fieldkey)
#
#     #pick slice nearest to given xpos
#     xxidx = find_nearest(dfieldsfluc[fieldkey+'_xx'],xpos)
#     fftslice = fieldfftsweepoverx[xxidx,:,:]
#
#     #find field(xpos,ky0,kz0) with norm greater than cutoffconst*max(norm(fftslice))
#     fftslice = np.real(fftslice*np.conj(fftslice))/(float(len(kz)*len(ky)))  #convert to norm
#     maxnorm = np.max(fftslice)
#     kylist = []
#     kzlist = []
#     prcntmaxlist = []
#     for i in range(0,len(kz)):
#         for j in range(0,len(ky)):
#             if(fftslice[i][j] >= cutoffconst*maxnorm):
#                 kzlist.append(kz[i])
#                 kylist.append(ky[j])
#                 prcntmaxlist.append(fftslice[i][j]/maxnorm)
#
#     #do wavelet transform for each ky, kz
#     kxlist = []
#     kxplotlist = []
#     wltlist = []
#     for i in range(0,len(kylist)):
#         ky0 = kylist[i]
#         ky0idx = find_nearest(ky,ky0)
#         kz0 = kzlist[i]
#         kz0idx = find_nearest(kz,kz0)
#
#         xkykzdata = fieldfftsweepoverx[:,kz0idx,ky0idx]
#
#         kx, wltdata = wlt(dfieldsfluc[fieldkey+'_xx'],xkykzdata)
#         kxplotlist.append(kx)
#         wltlist.append(wltdata)
#
#         kxidx = find_nearest(wltdata[:,xxidx],np.max(wltdata[:,xxidx]))
#         kxlist.append(kx[kxidx])
#
#     #add negative values for kx
#     nkx = len(kxlist)
#     for i in range(0,nkx):
#         kxlist.append(-1*kxlist[i])
#         kylist.append(kylist[i])
#         kzlist.append(kzlist[i])
#
#     return kxlist, kylist, kzlist, kxplotlist, wltlist, prcntmaxlist

def is_perp(vec1,vec2,tol=0.001):
    """
    Returns true if perpendicular

    Parameters
    ----------
    vec1 : array
        a vector
    vec2 : array
        a vector
    tol : float
        tolerance to closeness to zero

    Returns
    -------
    : bool
        True if perpendicular
    dotprod : float
        dotprod of vectors
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
    Returns true if parallel

    Parameters
    ----------
    vec1 : array
        a vector
    vec2 : array
        a vector
    tol : float
        tolerance to closeness to zero

    Returns
    -------
    : bool
        True if perpendicular
    dotprod : float
        dotprod of vectors
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

    Parameters
    ----------
    dfields : dict
        dict from field loader
    xxidx : int
        xx index to computer average at

    Returns
    -------
    [B0x, B0y, B0z] : [float,float,float]
        <B(x0,y,z)>_(y,z) at specified xx
    """

    dfavg = get_average_fields_over_yz(dfields)

    B0x = dfavg['bx'][0,0,xxidx]
    B0y = dfavg['by'][0,0,xxidx]
    B0z = dfavg['bz'][0,0,xxidx]

    return [B0x, B0y, B0z]

def get_B_avg(dfields,xlim,ylim,zlim):
    """
    Gets average B in box

    Parameters
    ----------
    dfields : dict
        dict from field_loader
    *lim : [float,float]
        upper and lower bounds of box

    Returns
    -------
    [B0x, B0y, B0z] : [float,float,float]
        B0 at specified xx
    """

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
    Routine that computes what kx would need to be given ky kz for the fluctuation to be alfvenic (in mhd limit)

    Parameters
    ----------
    ky/kz : float
        value in wavenumber space
    B0 : float
        average B0 (formatted [B0x, B0y, B0z]) in region
    delBperp : float
        fourier coefficient
        e.g. delBperp*e^{i k dot x}

    Returns
    -------
    kx : float
        kx needed to be alfvenic
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

    Paramters
    ---------
    x1/y1 : [float,float,float]
        vectors
    """
    x1perpx = x1[0]-(x1[0]*y1[0]+x1[1]*y1[1]+x1[2]*y1[2])/(y1[0]*y1[0]+y1[1]*y1[1]+y1[2]*y1[2])*y1[0]
    x1perpy = x1[1]-(x1[0]*y1[0]+x1[1]*y1[1]+x1[2]*y1[2])/(y1[0]*y1[0]+y1[1]*y1[1]+y1[2]*y1[2])*y1[1]
    x1perpz = x1[2]-(x1[0]*y1[0]+x1[1]*y1[1]+x1[2]*y1[2])/(y1[0]*y1[0]+y1[1]*y1[1]+y1[2]*y1[2])*y1[2]

    return [x1perpx,x1perpy,x1perpz]

#OLD AND NOT USED AND PROBABLY NOT THE BEST SCIENCE
# def alfven_wave_check(dfields,dfieldfluc,klist,xx,tol=.05):
#     """
#     Checks if basic properties of an alfven wave are seen at some location in the simulation
#
#     Note: dfields is normally the yz averaged removed fields (e.g. B_fluc(x,y,z) = B(x,y,z)-<B(x,y,z)>_(y,z))
#     """
#
#     from lib.array_ops import find_nearest
#
#     xxidx = find_nearest(dfields['bz_xx'],xx)
#
#     #TODO: rename these variables. Data is ordered B(x/kx,kz,ky)
#     #TODO: make function that can compute del B/E (kx,ky,kz;xx)
#     kz, ky, bxkzkyx = _ffttransform_in_yz(dfieldfluc,'bx')
#     kz, ky, bykzkyx = _ffttransform_in_yz(dfieldfluc,'by')
#     kz, ky, bzkzkyx = _ffttransform_in_yz(dfieldfluc,'bz')
#
#     kz, ky, exkzkyx = _ffttransform_in_yz(dfieldfluc,'ex')
#     kz, ky, eykzkyx = _ffttransform_in_yz(dfieldfluc,'ey')
#     kz, ky, ezkzkyx = _ffttransform_in_yz(dfieldfluc,'ez')
#
#     B0 = get_B_yzavg(dfields,xxidx)
#
#     # #get delta perp fields
#     # dperpf = get_delta_perp_fields(dfields,B0)
#
#     # check if any of the given k's have the expected properties of an alfven wave
#     # i.e. is deltaBperp parallel to kcrossB0, deltaB is perpendicular to B0, and delB is perpendicular to k
#     # where deltaB is from
#     results = []
#     kxexpected = [] #what kx needs to be for the wave to be alfvenic for each k in klist
#     delBlist = []
#     delElist = []
#     for i in range(0,len(klist)):
#         #pick a k and compute kperp
#         k = klist[i]
#         kperp = _get_perp_component(k,B0) #
#
#         #find nearest discrete point in (x,ky,kz) space we have data for
#         kyidx = find_nearest(ky,k[1])
#         kzidx = find_nearest(kz,k[2])
#         kyperpidx = find_nearest(ky,kperp[1])
#         kzperpidx = find_nearest(kz,kperp[2])
#
#         if(k[0] < 0):
#             _bxkzkyx = np.conj(bxkzkyx)
#             _bykzkyx = np.conj(bykzkyx)
#             _bzkzkyx = np.conj(bzkzkyx)
#             _exkzkyx = np.conj(exkzkyx)
#             _eykzkyx = np.conj(eykzkyx)
#             _ezkzkyx = np.conj(ezkzkyx)
#         else:
#             _bxkzkyx = bxkzkyx
#             _bykzkyx = bykzkyx
#             _bzkzkyx = bzkzkyx
#             _exkzkyx = exkzkyx
#             _eykzkyx = eykzkyx
#             _ezkzkyx = ezkzkyx
#
#         #finalize transform into k space i.e. compute B(kx0,kz0,ky0) from B(x,kz,ky) for k and k perp
#         #note: we never have an array B(kx,ky,kz), just that scalar quantities at k0 and kperp0, which we get from
#         # the just for B(x,kz,ky) as computing the entire B(kx,ky,kz) array would be computationally expensive.
#         # would have to perform wavelet transform for each (ky0,kz0)
#         kx, bxkz0ky0kxxx = wlt(dfieldfluc['bx_xx'],_bxkzkyx[:,kzidx,kyidx]) #note kx is that same for all 6 returns here
#         kx, bykz0ky0kxxx = wlt(dfieldfluc['by_xx'],_bykzkyx[:,kzidx,kyidx])
#         kx, bzkz0ky0kxxx = wlt(dfieldfluc['bz_xx'],_bzkzkyx[:,kzidx,kyidx])
#         kx, exkz0ky0kxxx = wlt(dfieldfluc['ex_xx'],_exkzkyx[:,kzidx,kyidx])
#         kx, eykz0ky0kxxx = wlt(dfieldfluc['ey_xx'],_eykzkyx[:,kzidx,kyidx])
#         kx, ezkz0ky0kxxx = wlt(dfieldfluc['ez_xx'],_ezkzkyx[:,kzidx,kyidx])
#
#         kx, bxperpkz0ky0kxxx = wlt(dfieldfluc['bx_xx'],_bxkzkyx[:,kzperpidx,kyperpidx])
#         kx, byperpkz0ky0kxxx = wlt(dfieldfluc['by_xx'],_bykzkyx[:,kzperpidx,kyperpidx])
#         kx, bzperpkz0ky0kxxx = wlt(dfieldfluc['bz_xx'],_bzkzkyx[:,kzperpidx,kyperpidx])
#
#         kxidx = find_nearest(kx,np.abs(k[0])) #WLT can not find negative kx. Instead we assume symmetry by taking np.abs
#         kxperpidx = find_nearest(kx,np.abs(kperp[0]))
#
#         # if(k[0] < 0): #use reality condition to correct for the fact that we cant compute negative kx using the wlt
#         #     bxkz0ky0kxxx = np.conj(bxkz0ky0kxxx)
#         #     bykz0ky0kxxx = np.conj(bykz0ky0kxxx)
#         #     bzkz0ky0kxxx = np.conj(bzkz0ky0kxxx)
#         #     bxperpkz0ky0kxxx = np.conj(bxperpkz0ky0kxxx)
#         #     byperpkz0ky0kxxx = np.conj(byperpkz0ky0kxxx)
#         #     bzperpkz0ky0kxxx = np.conj(bzperpkz0ky0kxxx)
#
#         kcrossB0 = np.cross(k,B0)
#         delB = [bxkz0ky0kxxx[kxidx,xxidx],bykz0ky0kxxx[kxidx,xxidx],bzkz0ky0kxxx[kxidx,xxidx]]
#         delE = [exkz0ky0kxxx[kxidx,xxidx],eykz0ky0kxxx[kxidx,xxidx],ezkz0ky0kxxx[kxidx,xxidx]]
#         delBlist.append(delB)
#         delElist.append(delE)
#         delBperp = [bxperpkz0ky0kxxx[kxperpidx,xxidx],byperpkz0ky0kxxx[kxperpidx,xxidx],bzperpkz0ky0kxxx[kxperpidx,xxidx]]
#
#         kxexpected.append(predict_kx_alfven(k[1],k[2],B0,delBperp))
#
#         #results.append([is_parallel(delBperp,kcrossB0,tol=0.1),is_perp(delB,B0,tol=0.1),is_perp(k,delB,tol=.1)])
#         testAlfvenval = np.cross(delB,np.cross(k,B0))
#         testAlfvenval /= (np.linalg.norm(delB)*np.linalg.norm(np.cross(k,B0)))
#         if(np.linalg.norm(testAlfvenval) <= tol):
#             belowtol = True
#         else:
#             belowtol = False
#         #belowtol = (testAlfvenval <= tol)
#
#         results.append([(belowtol,np.linalg.norm(testAlfvenval)),is_perp(k,delB,tol=tol)])
#
#     #TODO: consider cleaning up computing delB (maybe move to own function)
#     return results, kxexpected, delBlist, delElist

def compute_field_aligned_coord(dfields,xlim,ylim,zlim):
    """
    Computes field aligned coordinate basis using average B0 in provided box

    vpar in parallel to B0
    vperp2 is in direction of [xhat] cross vpar
    vperp is in direction of vpar cross vperp2

    TODO: rename vars (vperp->eperp, etc...)

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader
    xlim : array
        xx bounds of analysis (i.e. where the sweep starts and stops)
    ylim : array
        yy bounds of each integration box
    zlim : array
        zz bounds of each integration box

    Returns
    -------
    vparbasis/vperp1basis/vperp2basis : [float,float,float]
        field aligned basis (ordered [vx,vy,vz])
    """
    from lib.array_ops import find_nearest
    from copy import deepcopy

    if(np.abs(xlim[1]-xlim[0]) > 4.):
        print("Warning, when computing field aligned coordinates, we found that xlim[1]-xlim[0] is large. Consider reducing size...")

    #TODO: rename vpar,vperp to epar, eperp...
    xavg = (xlim[1]+xlim[0])/2.
    xxidx = find_nearest(dfields['bz_xx'],xavg)
    B0 = get_B_avg(dfields,xlim,ylim,zlim) #***Assumes xlim is sufficiently thin*** as get_B0 uses <B(x0,y,z)>_(yz)=B0

    #get normalized basis vectors
    vparbasis = deepcopy(B0)
    vparbasis /= np.linalg.norm(vparbasis)
    #vperp1basis = _get_perp_component([0,1,0],vparbasis) #TODO: check that this returns something close to 0,1,0 as B0 is approximately in the xz plane (with some fluctuations)
    vperp2basis = np.cross([1.,0,0],B0) #x hat cross B0
    tol = 0.005
    _B0 = B0 / np.linalg.norm(B0)
    if(np.abs(np.linalg.norm(np.cross([_B0[0],_B0[1],_B0[2]],[1.,0.,0.]))) < tol):
        print("Warning, it seems B0 is parallel to xhat (typically the shock normal)...")
        print("(Bx,By,Bz): ", _B0[0],_B0[1],_B0[2])
        print("xhat: 1,0,0")
        print("Already in field aligned coordinates. Returning standard basis...")
        return np.asarray([1.,0,0]),np.asarray([0,1.,0]),np.asarray([0,0,1.])
    vperp2basis /= np.linalg.norm(vperp2basis)
    vperp1basis = np.cross(vparbasis,vperp2basis)
    vperp1basis /= np.linalg.norm(vperp1basis)

    return vparbasis, vperp1basis, vperp2basis

def change_velocity_basis(dfields,dpar,xlim,ylim,zlim,debug=False):
    """
    Converts to field aligned coordinate system
    Parallel direction is along average magnetic field direction at average in limits

    Parameters
    ----------
    dfields : dict
        dict returned by field_loader
    dpar : dict
        dict returned by read_particles
    xlim : array
        xx bounds of analysis (i.e. where the sweep starts and stops)
    ylim : array
        yy bounds of each integration box
    zlim : array
        zz bounds of each integration box
    debug : bool, opt
        print debug statements if energy is not conserved

    Returns
    -------
    dparnewbasis : dict
        particle dictionary in new basis
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
    dparnewbasis['x1'] = deepcopy(dpar['x1'][:])
    dparnewbasis['x2'] = deepcopy(dpar['x2'][:])
    dparnewbasis['x3'] = deepcopy(dpar['x3'][:])

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

    Parameters
    ----------
    dparfieldaligned : dict
        dict returned by change_velocity_basis
    vmax : float
        max velocity to build distribution functions up to
    dv : float
        space in velocity space
    V : [float,float,float]
        drift velocity formatted Vx Vy Vz

    Returns
    -------
    Tperp_over_Tpar : float
        estimated temp anisotropy
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

    #integrate by riemann sum, note: the factor of delta v cancels
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
    kz,ky,kx : 1d array
        coordinates
    fieldkzkykxxx : 4d array
        transformed fields
    """

    kz, ky, fieldxkzky = _ffttransform_in_yz(ddict,fieldkey)

    nxx = len(ddict[fieldkey+'_xx'])
    nkx = int(len(ddict[fieldkey+'_xx'])/retstep) #warning: this is hard coded to match wlt function output size
    nky = len(ky)
    nkz = len(kz)
    fieldkzkykxxx = np.zeros((nkz,nky,2*nkx,nxx),dtype=np.complex_)

    for kyidx in range(0,len(ky)):
        for kzidx in range(0,len(kz)):
            positivekx, rightfieldkz0ky0kxxx = wlt(ddict[fieldkey+'_xx'],fieldxkzky[:,kzidx,kyidx],retstep=retstep)
            negativekx, leftfieldkz0ky0kxxx = wlt(ddict[fieldkey+'_xx'],np.conj(fieldxkzky[:,kzidx,kyidx]),retstep=retstep)
            leftfieldkz0ky0kxxx = np.conj(leftfieldkz0ky0kxxx) #use reality condition to compute negative kxs
            fieldkzkykxxx[kzidx,kyidx,nkx:,:] = rightfieldkz0ky0kxxx[:,:]
            fieldkzkykxxx[kzidx,kyidx,0:nkx,:] = np.flip(leftfieldkz0ky0kxxx[:,:], axis=0)

    negativekx *= -1
    negativekx = np.flip(negativekx)
    kx = np.concatenate([negativekx,positivekx])

    return kz, ky, kx, fieldkzkykxxx

def compute_vrms(dpar,vmax,dv,x1,x2,y1,y2,z1,z2):
    """
    https://farside.ph.utexas.edu/teaching/plasma/lectures/node29.html

    assumes the max speed of any of the particles is less than vmax

    Parameters
    ----------
    dpar : dict
        particle dictionary returned by read_particles
    dv : float
        spacing in velocity space
    x1,x2,y1,y2,z1,z2 : float
        bounds of box used to compute vrms

    Returns
    -------
    vrms_squared : float
        velocity using root mean squared
        note that the value is squared
    """
    #TODO: compute vdrift
    vzdrift = 0.
    vydrift = 0.
    vxdrift = 0.

    # bin into cprime(vx,vy,vz)
    vxbins = np.arange(-vmax, vmax+dv, dv)
    vx = (vxbins[1:] + vxbins[:-1])/2.
    vybins = np.arange(-vmax, vmax+dv, dv)
    vy = (vybins[1:] + vybins[:-1])/2.
    vzbins = np.arange(-vmax, vmax+dv, dv)
    vz = (vzbins[1:] + vzbins[:-1])/2.

    # define mask that includes particles within range and make dist
    gptsparticle = (x1 <= dpar['x1']) & (dpar['x1'] <= x2) & (y1 <= dpar['x2']) & (dpar['x2'] <= y2) & (z1 <= dpar['x3']) & (dpar['x3'] <= z2)
    hist,_ = np.histogramdd((dpar['p3'][gptsparticle][:]-vzdrift, dpar['p2'][gptsparticle][:]-vydrift, dpar['p1'][gptsparticle][:]-vxdrift), bins=[vzbins, vybins, vxbins])

    # computure pressure
    pressure = 0.
    for i in range(0,len(vx)):
        for j in range(0,len(vy)):
            for k in range(0,len(vz)):
                vel = math.sqrt(vx[i]**2.+vy[j]**2.+vz[k]**2.)
                pressure += hist[k,j,i]*vel**2.*dv**3.
    pressure = pressure / 3.

    #compute number density
    num_den = get_num_par_in_box(dpar,x1,x2,y1,y2,z1,z2)

    vrms_squared = pressure/num_den #TODO: check if im missing a factor here

    return vrms_squared

def compute_vrms_par(dpar,dfields,vmax,dv,x1,x2,y1,y2,z1,z2,vparbasis):
    """
    Same as compute_vrms, but finds the parallel vrms...

    vparbasis : array
        is the vpar basis vector (normalized) in cartesian coordinates
    """
    #TODO: compute vdrift, convert to field aligned coordinates, and shift
    vzdrift = 0.
    vydrift = 0.
    vxdrift = 0.

    dparnewbasis = change_velocity_basis(dfields,dpar,xlimsubset,ylim,zlim,debug=True)

    #bin into hist with field aligned coordinates
    vperp2bins = np.arange(-vmax, vmax+dv, dv)
    vperp2 = (vperp2bins[1:] + vperp2bins[:-1])/2.
    vperp1bins = np.arange(-vmax, vmax+dv, dv)
    vperp1 = (vperp1bins[1:] + vperp1bins[:-1])/2.
    vparbins = np.arange(-vmax, vmax+dv, dv)
    vpar = (vparbins[1:] + vparbins[:-1])/2.

    hist,_ = np.histogramdd((dparnewbasis['pperp2']-Vdrift[2], dparnewbasis['pperp1']-Vdrift[1],  dparnewbasis['ppar']-Vdrift[0],), bins=[vperp2bins, vperp1bins, vparbins])

    #compute number density
    num_den = get_num_par_in_box(dpar,x1,x2,y1,y2,z1,z2)

    # computure parallel pressure
    pressure = 0.
    for i in range(0,len(vperp2)):
        for j in range(0,len(vperp1)):
            for k in range(0,len(vpar)):
                vel = math.sqrt(vpar[i]**2.)#+vy[j]**2.+vz[k]**2.)
                pressure += hist[k,j,i]*vel**2.*dv**3.
    pressure = pressure / 3.

    vrms_squared_par = pressure/num_den #TODO: check if im missing a factor here

    vrms_par = np.sqrt(vrms_squared_par)

    return vrms_par

def compute_alfven_vel(dfields,dden,x1,x2,y1,y2,z1,z2):
    """
    Computes the average alfven veloicty normalized to dHybridR units, v_a/v_{a,ref}
    in the given box.

    Note, v_{a,ref} is defined in the simulation input

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    dden : dict
        fluid density data dictionary from den_loader
    x1 : float
        lower x bound
    x2 : float
        upper x bound
    y1 : float
        lower y bound
    y2 : float
        upper y bound
    z1 : float
        lower y bound
    z2 : float
        upper y bound

    Returns
    -------
    v_a : float
        alfven velocity normalized to reference alfven velocity

    """

    from lib.array_ops import get_average_in_box

    rho = get_average_in_box(x1,x2,y1,y2,z1,z2, dden, 'den')

    bx = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'bx')
    by = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'by')
    bz = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'bz')
    btot = math.sqrt(bx**2.+by**2.+bz**2.)

    v_a = btot/rho

    return v_a

def compute_alfven_vel_par(dfields,dden,x1,x2,y1,y2,z1,z2,vparbasis):
    """
    Computes the average alfven veloicty normalized to dHybridR units, v_a/v_{a,ref}
    in the given box.

    Note, v_{a,ref} is defined in the simulation input

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    dden : dict
        fluid density data dictionary from den_loader
    x1 : float
        lower x bound
    x2 : float
        upper x bound
    y1 : float
        lower y bound
    y2 : float
        upper y bound
    z1 : float
        lower y bound
    z2 : float
        upper y bound

    Returns
    -------
    v_a : float
        alfven velocity normalized to reference alfven velocity

    """

    from lib.array_ops import get_average_in_box

    rho = get_average_in_box(x1,x2,y1,y2,z1,z2, dden, 'den')

    bx = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'bx')
    by = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'by')
    bz = get_average_in_box(x1, x2, y1, y2, z1, z2, dfields, 'bz')

    bpar = np.abs(np.dot([bx,by,bz],vparbasis))


    v_a = bpar/rho

    return v_a

def compute_beta_i(dpar,dfields,dden,vmax,dv,x1,x2,y1,y2,z1,z2):
    """
    Computes plasma beta for ions using, beta_i = v_ion_th**2./v_ion_a**2.

    Parameters
    ----------
    dpar : dict
        xx vx yy vy zz vz data dictionary from read_particles or read_box_of_particles
    dfields : dict
        field data dictionary from field_loader
    dden : dict
        fluid density data dictionary from den_loader
    vmax : float
        max limit in velocity space used when estimating thermal velocity using moments of the distribution
    dv : float
        velocity space grid spacing
        (assumes square)
    x1 : float
        lower x bound
    x2 : float
        upper x bound
    y1 : float
        lower y bound
    y2 : float
        upper y bound
    z1 : float
        lower y bound
    z2 : float
        upper y bound

    Returns
    -------
    beta_i : float
        average ion plasma beta in box
    """

    #compute v_th
    v_ion_th = np.sqrt(compute_vrms(dpar,vmax,dv,x1,x2,y1,y2,z1,z2))

    #compute v_alfven_ion
    v_ion_a = compute_alfven_vel(dfields,dden,x1,x2,y1,y2,z1,z2)

    beta_i = v_ion_th**2./v_ion_a**2.

    return beta_i, v_ion_th, v_ion_a

def compute_beta_i_par(dpar,dfields,dden,vmax,dv,x1,x2,y1,y2,z1,z2):
    """
    Computes plasma beta for ions using, beta_i = v_ion_th**2./v_ion_a**2.

    Parameters
    ----------
    dpar : dict
        xx vx yy vy zz vz data dictionary from read_particles or read_box_of_particles
    dfields : dict
        field data dictionary from field_loader
    dden : dict
        fluid density data dictionary from den_loader
    vmax : float
        max limit in velocity space used when estimating thermal velocity using moments of the distribution
    dv : float
        velocity space grid spacing
        (assumes square)
    x1 : float
        lower x bound
    x2 : float
        upper x bound
    y1 : float
        lower y bound
    y2 : float
        upper y bound
    z1 : float
        lower y bound
    z2 : float
        upper y bound

    Returns
    -------
    beta_i_par : float
        average parallel ion plasma beta in box
    """

    #get parallel vector
    xlim = [x1,x2]
    ylim = [y1,y2]
    zlim = [z1,z2]
    vparbasis, vperp1basis, vperp2basis = compute_field_aligned_coord(dfields,xlim,ylim,zlim)

    #compute v_th
    v_ion_th_par = np.sqrt(compute_vrms_par(dpar,dfields,vmax,dv,x1,x2,y1,y2,z1,z2,vparbasis))

    #compute v_alfven_ion
    v_ion_a_par = compute_alfven_vel_par(dfields,dden,x1,x2,y1,y2,z1,z2,vparbasis)

    beta_i_par = v_ion_th_par**2./v_ion_a_par**2.

    return beta_i_par, v_ion_th_par, v_ion_a_par

def compute_electron_temp(dden,x1,x2,y1,y2,z1,z2,Te0=1.,gamma=1.66667,num_den_elec0=1.):
    """
    Parameters
    ----------
    dden : dict
        fluid density data dictionary from den_loader
    x1 : float
        lower x bound
    x2 : float
        upper x bound
    y1 : float
        lower y bound
    y2 : float
        upper y bound
    z1 : float
        lower y bound
    z2 : float
        upper y bound
    Te0 : float
        upstream (as in newly injected inflow) electron temperature (in code units)
    gamma : float
        adiabatic index
    Te0 : float
        upstream (as in newly injected inflow) electron density (in code units)

    Returns
    -------
    Te : float
        average electron temperature in given box
    """

    print("WARNING THIS DOES NOT INCLUDE DRIFT VELOCITY YET")

    from array_ops import get_average_in_box


    num_den_elec = get_average_in_box(x1,x2,y1,y2,z1,z2,dden, 'den')

    Te = Te0*(num_den_elec/num_den_elec0)**(gamma-1)

    return Te

def compute_tau(dpar,dden,vmax,dv,x1,x2,y1,y2,z1,z2):
    """
    Computes temperature ratio Te/Ti in given box

    Parameters
    ----------
    dpar : dict
        xx vx yy vy zz vz data dictionary from read_particles or read_box_of_particles
    dden : dict
        fluid density data dictionary from den_loader
    vmax : float
        max limit in velocity space used when estimating thermal velocity using moments of the distribution
    dv : float
        velocity space grid spacing
        (assumes square)
    x1 : float
        lower x bound
    x2 : float
        upper x bound
    y1 : float
        lower y bound
    y2 : float
        upper y bound
    z1 : float
        lower y bound
    z2 : float
        upper y bound

    Returns
    -------
    tau : float
        temperature ration Te/Ti in box
    """

    print("WARNING THIS DOES NOT INCLUDE DRIFT VELOCITY YET")

    Te = compute_electron_temp(dden,x1,x2,y1,y2,z1,z2)

    v_ion_th = compute_vrms(dpar,vmax,dv,x1,x2,y1,y2,z1,z2)
    Ti = v_ion_th**2.

    tau = Te/Ti

    return tau

def va_norm_to_vi_norm(dpar, v_w_anorm, vmax, x1, x2, y1, y2, z1, z2, vti = None):
    """
    Given some velocity normalized to v_{a,ref} (defined in dHybridR input), this
    function converts that velocity to instead normalized to the estimated thermal
    velocity in the given box

    It should be noted that v_ti,ref = v_alfven,ref in most simulations. That is
    the inflowing plasma beta = 1

    Parameters
    ----------
    dpar : dict
        xx vx yy vy zz vz data dictionary from read_particles or read_box_of_particles
    v_w_anorm : float
        velocity with v_alfven,ref normalization
    vmax : float
        max limit in velocity space used when estimating thermal velocity using moments of the distribution
    dv : float
        velocity space grid spacing
        (assumes square)
    x1 : float
        lower x bound
    x2 : float
        upper x bound
    y1 : float
        lower y bound
    y2 : float
        upper y bound
    z1 : float
        lower y bound
    z2 : float
        upper y bound

    Returns
    -------
    v_w_tinorm : float
        velocity with v_ti normalization
    """
    if(vti == None):
        vti = compute_vrms(dpar,vmax,dv,x1,x2,y1,y2,z1,z2)

    v_w_tinorm = v_w_anorm / vti

    return v_w_tinorm

def build_dist(dpar,vmax,dv,x1,x2,y1,y2,z1,z2):
    """
    """
    gptsparticle = (x1 <= dpar['x1']) & (dpar['x1'] <= x2) & (y1 <= dpar['x2']) & (dpar['x2'] <= y2) & (z1 <= dpar['x3']) & (dpar['x3'] <= z2)

    # bin into cprime(vx,vy,vz) #TODO: use function for this block (it's useful elsewhere to build distribution functions)
    vxbins = np.arange(-vmax, vmax+dv, dv)
    vx = (vxbins[1:] + vxbins[:-1])/2.
    vybins = np.arange(-vmax, vmax+dv, dv)
    vy = (vybins[1:] + vybins[:-1])/2.
    vzbins = np.arange(-vmax, vmax+dv, dv)
    vz = (vzbins[1:] + vzbins[:-1])/2.

    hist,_ = np.histogramdd((dpar['p3'][gptsparticle][:], dpar['p2'][gptsparticle][:], dpar['p1'][gptsparticle][:]), bins=[vzbins, vybins, vxbins])

    # make the bins 3d arrays TODO: use function (replace all instances of this with function)
    _vx = np.zeros((len(vz), len(vy), len(vx)))
    _vy = np.zeros((len(vz), len(vy), len(vx)))
    _vz = np.zeros((len(vz), len(vy), len(vx)))
    for i in range(0, len(vx)):
        for j in range(0, len(vy)):
            for k in range(0, len(vz)):
                _vx[k][j][i] = vx[i]

    for i in range(0, len(vx)):
        for j in range(0, len(vy)):
            for k in range(0, len(vz)):
                _vy[k][j][i] = vy[j]

    for i in range(0, len(vx)):
        for j in range(0, len(vy)):
            for k in range(0, len(vz)):
                _vz[k][j][i] = vz[k]
    vx = _vx
    vy = _vy
    vz = _vz

    return vx,vy,vz,hist

def build_dist_and_remove_average_par_over_yz(dpar,vmax,dv,dx,x1,x2,y1,y2,z1,z2,ymax,zmax):
    """

    """
    gptsparticle = (x1 <= dpar['x1']) & (dpar['x1'] <= x2) & (y1 <= dpar['x2']) & (dpar['x2'] <= y2) & (z1 <= dpar['x3']) & (dpar['x3'] <= z2)

    vx,vy,vz,full_hist = build_dist(dpar,vmax,dv,x1,x2,0,ymax,0,zmax)
    vx,vy,vz,sub_hist = build_dist(dpar,vmax,dv,x1,x2,y1,y2,z1,z2)

    #normalize sub_hist
    npar_sub = np.sum(sub_hist)
    sub_hist = sub_hist*np.sum(full_hist)/npar_sub

    delta_hist = sub_hist - full_hist

    return vx,vy,vz,delta_hist,full_hist

#TODO: not used much, consider removing
def project_dist_to_vx(vx,vy,vz,hist):

    hist_vyvx = np.sum(hist,axis=0)
    hist_vx = np.sum(hist_vyvx,axis=0)

    return vx[0,0,:], hist_vx
