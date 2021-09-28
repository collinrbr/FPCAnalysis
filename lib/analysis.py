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

def take_fft2(data,daxisx0,daxis1):
    """
    Computes 2d fft on given data

    Parameters
    ----------
    data : 2d array
        data to be transformed
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

    fftdata = np.fft.fft2(data)

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

def get_delta_perp_fields(dfields,B0):
    """
    Computes the perpendicular component of the delta fields wrt the total magnetic field at each point
    """

    from copy import deepcopy

    ddeltaperpfields = get_delta_fields(dfields,B0)

    ddeltaperpfields['bx'] = ddeltaperpfields['bx'] - (ddeltaperpfields['bx']*B0[0]+ddeltaperpfields['by']*B0[1]+ddeltaperpfields['bz']*B0[2])/(B0[0]**2+B0[1]**2+B0[2]**2)*B0[0]
    ddeltaperpfields['by'] = ddeltaperpfields['by'] - (ddeltaperpfields['bx']*B0[0]+ddeltaperpfields['by']*B0[1]+ddeltaperpfields['bz']*B0[2])/(B0[0]**2+B0[1]**2+B0[2]**2)*B0[1]
    ddeltaperpfields['bz'] = ddeltaperpfields['bz'] - (ddeltaperpfields['bx']*B0[0]+ddeltaperpfields['by']*B0[1]+ddeltaperpfields['bz']*B0[2])/(B0[0]**2+B0[1]**2+B0[2]**2)*B0[2]

    return ddeltaperpfields

def wlt(t,data,w=6):
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
    """
    from scipy import signal

    dt = t[1]-t[0]
    fs = 1./dt

    freq = np.linspace(.01,fs/2.,len(data))
    widths = w*fs / (2*freq*np.pi)

    cwtm = signal.cwt(data, signal.morlet2, widths, w=w)

    return 2.0*math.pi*freq, cwtm

def find_potential_wavemodes(dfields,fieldkey,xpos,cutoffconst=.1):
    """

    """

    from lib.array_ops import find_nearest

    #compute delta fields
    dfieldsfluc = remove_average_fields_over_yz(dfields)

    #spacing in grids, needed to get wavenumber from fft
    daxis0 = dfieldsfluc[fieldkey+'_zz'][1]-dfieldsfluc[fieldkey+'_zz'][0]
    daxis1 = dfieldsfluc[fieldkey+'_yy'][1]-dfieldsfluc[fieldkey+'_yy'][0]

    fieldfftsweepoverx = []
    for xxindex in range(0,len(dfieldsfluc[fieldkey][0][0])):
        fieldslice = np.asarray(dfieldsfluc[fieldkey])[:,:,xxindex]
        kz, ky, fieldslicefft = take_fft2(fieldslice,daxis0,daxis1)
        fieldfftsweepoverx.append(fieldslicefft)
    fieldfftsweepoverx = np.asarray(fieldfftsweepoverx)

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
    wltplotlist = []
    for i in range(0,len(kylist)):
        ky0 = kylist[i]
        ky0idx = find_nearest(ky,ky0)
        kz0 = kzlist[i]
        kz0idx = find_nearest(kz,kz0)

        xkykzdata = fieldfftsweepoverx[:,kz0idx,ky0idx]

        kx, wltdata = wlt(dfieldsfluc[fieldkey+'_xx'],xkykzdata)
        kxplotlist.append(kx)
        wltplotlist.append(wltdata)

        kxidx = find_nearest(wltdata[:,xxidx],np.max(wltdata[:,xxidx]))
        kxlist.append(kx[kxidx])

    return kxlist, kylist, kzlist, kxplotlist, wltplotlist, prcntmaxlist

def is_perp(vec1,vec2,tol=0.001):
    """

    """

    #normalize vector
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)

    dotprod = vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]

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

    dotprod = vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]

    if (abs(abs(dotprod)-1.0) <= tol):
        return True, dotprod
    else:
        return False, dotprod


#Note, B0 is from arctan(Bz/Bx) in the upstream region
def get_B0(dfields):
    """

    """

    dfavg = get_average_fields_over_yz(dfields)

    B0x = dfavg['bx'][0,0,-1]
    B0y = dfavg['by'][0,0,-1]
    B0z = dfavg['bz'][0,0,-1]

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
    kx = (Bz*dBx*ky-Bx*dBz*ky-By*dBx*kz+Bx*dBy*kz)/(Bz*dBy-By*dBz)

    return kx

def alfven_wave_check(dfields,klist,xx,yy,zz):
    """
    Checks if basic properties of an alfven wave are seen at some location in the simulation
    """

    from lib.array_ops import find_nearest

    xxidx = find_nearest(dfields['bz_xx'],xx)
    yyidx = find_nearest(dfields['bz_yy'],yy)
    zzidx = find_nearest(dfields['bz_zz'],zz)

    #get external field
    B0 = get_B0(dfields)

    #get delta fields (different from removing yz average)
    ddeltaf = get_delta_fields(dfields,B0)

    #get delta perp fields
    dperpf = get_delta_perp_fields(dfields,B0)

    # check if any of the predicted k's work for this
    results = []
    for i in range(0,len(klist)):
        k = klist[i]
        kcrossB0 = np.cross(k,B0)
        delB = [ddeltaf['bx'][zzidx,yyidx,xxidx],ddeltaf['by'][zzidx,yyidx,xxidx],ddeltaf['bz'][zzidx,yyidx,xxidx]]
        delBperp = [dperpf['bx'][zzidx,yyidx,xxidx],dperpf['by'][zzidx,yyidx,xxidx],dperpf['bz'][zzidx,yyidx,xxidx]]
        results.append([is_parallel(delBperp,kcrossB0,tol=0.1),is_perp(delB,B0,tol=0.1),is_perp(delB,k,tol=.1)])

    return results
