# fpc.py>

# functions related to computing FPC

import numpy as np


def compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2,
                            dpar, dfields, fieldkey, directionkey, useBoxFAC = True, altcorfields = None, beta = None, massratio = None, c = None):
    """
    Computes distribution function and correlation wrt to given field

    Warning: assumes fields and particles are in the same frame

    Parameters
    ----------
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
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
    dpar : dict
        xx vx yy vy zz vz data dictionary from read_particles or read_box_of_particles
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of the field you want to correlate with
        ex,ey,ez,bx,by, or bz
    directionkey : str
        name of direction you want to take the derivative with respect to
        x,y,or z
        *should match the direction of the fieldkey*
    useBoxFAC : bool, opt
        if TRUE, compute field aligned coordinate system using the average value of the magnetic field across the whole 
        transverse domain. Otherwise, will use local value of each particle to compute field aligned coordinate system
    altcorfields : dict, opt
        if not None, then all calculations will be performed using dfields except for the correlation. Used when computing
        fluct correlation in field aligned coordinates

    Returns
    -------
    vx : 3d array
        vx velocity grid
    vy : 3d array
        vy velocity grid
    vz : 3d array
        vz velocity grid
    totalPtcl : float
        total number of particles in the correlation box
    Hist : 3d array
        distribution function in box
    Cor : 3d array
        velocity space sigature data in box
    """
    
    null_for_debug = False #returns zero array, which saves a lot of time, for debug purposes
    if(null_for_debug):
        # bin into cprime(vx,vy,vz)
        vxbins = np.arange(-vmax, vmax+dv, dv)
        vx = (vxbins[1:] + vxbins[:-1])/2.
        vybins = np.arange(-vmax, vmax+dv, dv)
        vy = (vybins[1:] + vybins[:-1])/2.
        vzbins = np.arange(-vmax, vmax+dv, dv)
        vz = (vzbins[1:] + vzbins[:-1])/2.

        # make the bins 3d arrays
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
    
        totalPtcl = 0
        hist = np.zeros(vx.shape)
        cor = np.zeros(vx.shape)
       
        print("Warning: returning zero array for hist and cor...")
        return vx, vy, vz, totalPtcl, hist, cor

    # check input
    if(fieldkey == 'ex' or fieldkey == 'bx'):
        if(directionkey != 'x'):
            print("Warning, direction of derivative does not match field direction")
    if(fieldkey == 'ey' or fieldkey == 'by'):
        if(directionkey != 'y'):
            print("Warning, direction of derivative does not match field direction")
    if(fieldkey == 'ez' or fieldkey == 'bz'):
        if(directionkey != 'z'):
            print("Warning, direction of derivative does not match field direction")

    #change keys for backwards compat
    if('xi' in dpar.keys()):
        dpar['x1'] = dpar['xi']
        dpar['x2'] = dpar['yi']
        dpar['x3'] = dpar['zi']
        dpar['p1'] = dpar['ui']
        dpar['p2'] = dpar['vi']
        dpar['p3'] = dpar['wi']
    if('xe' in dpar.keys()):
        dpar['x1'] = dpar['xe']
        dpar['x2'] = dpar['ye']
        dpar['x3'] = dpar['ze']
        dpar['p1'] = dpar['ue']
        dpar['p2'] = dpar['ve']
        dpar['p3'] = dpar['we']

    if(z1 != None and z2 != None):
        gptsparticle = (x1 <= dpar['x1']) & (dpar['x1'] <= x2) & (y1 <= dpar['x2']) & (dpar['x2'] <= y2) & (z1 <= dpar['x3']) & (dpar['x3'] <= z2)
    else:
        gptsparticle = (x1 <= dpar['x1']) & (dpar['x1'] <= x2) & (y1 <= dpar['x2']) & (dpar['x2'] <= y2)

    _vxdown = None
    _vydown = None
    _vzdown = None


    if(False): #TODO: reimplement this check by fixing boosts for particles... => dfields['Vframe_relative_to_sim'] != dpar['Vframe_relative_to_sim']):
        print("ERROR: particles and fields are not in the same frame...")
        return
    else:
        dpar_p1 = np.asarray(dpar['p1'][gptsparticle][:])
        dpar_p2 = np.asarray(dpar['p2'][gptsparticle][:])
        dpar_p3 = np.asarray(dpar['p3'][gptsparticle][:])

    totalPtcl = np.sum(gptsparticle)

    # build dparticles subset using shifted particle data
    dparsubset = {
        'q': dpar['q'],
        'p1': dpar_p1,
        'p2': dpar_p2,
        'p3': dpar_p3,
        'x1': dpar['x1'][gptsparticle][:],
        'x2': dpar['x2'][gptsparticle][:],
        'x3': dpar['x3'][gptsparticle][:],
        'Vframe_relative_to_sim': dpar['Vframe_relative_to_sim']
    }

    if('q' in dpar.keys()):
        dparsubset['q'] = dpar['q']

    cprimebinned, hist, vx, vy, vz = compute_cprime_hist(dparsubset, dfields, fieldkey, vmax, dv, useBoxFAC=useBoxFAC, altcorfields=altcorfields, beta = beta, massratio = massratio, c = c, vxdown=_vxdown,vydown=_vydown,vzdown=_vzdown)
    del dparsubset

    cor = compute_cor_from_cprime(cprimebinned, vx, vy, vz, dv, directionkey)
    del cprimebinned

    return vx, vy, vz, totalPtcl, hist, cor

#TODO: lot of redundancy is this library. FIX THIS
#1. compute vx, vy, vz redundantly
#2. compute Hist redundantly
#2. a can improve CEx, CEy, CEz calc by not computing hist redundantly
#3. dont compute subset each time for CEx, CEy, CEz

#TODO: clean up sub routine of checkFrameandGrabSubset

def _comp_all_CEi(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, checkFrameandGrabSubset=True):
    """
    Wrapper function that computes FPC wrt xx, yy, zz and returns all three of them

    See documentation for compute_hist_and_cor
    """
    from FPCAnalysis.frametransform import shift_particles

    dparticles = shift_particles(dparticles, vshock) #TODO: our shift function

    vx, vy, vz, totalPtcl, Hist, CEx = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, 'ex', 'x')
    vx, vy, vz, totalPtcl, Hist, CEy = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, 'ey', 'y')
    vx, vy, vz, totalPtcl, Hist, CEz = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, 'ez', 'z')
        
    return vx, vy, vz, totalPtcl, Hist, CEx, CEy, CEz

def project_CEi_hist(Hist, CEx, CEy, CEz):
    """
    Project to 2V

    Parameters
    ----------
    Hist : 3D array
        distrubution function
    CEx : 3D array
        FPC wrt Ex fields
    CEy : 3D array
        FPC wrt Ey fields
    CEz : 3D array
        FPC wrt Ez fields

    Returns
    -------
    (Hist/CEi)** : 2D array
        2D projection onto ** axis
    """
    from FPCAnalysis.array_ops import array_3d_to_2d

    Histxy = array_3d_to_2d(Hist,'xy')
    Histxz = array_3d_to_2d(Hist,'xz')
    Histyz = array_3d_to_2d(Hist,'yz')

    CExxy = array_3d_to_2d(CEx,'xy')
    CExxz = array_3d_to_2d(CEx,'xz')
    CExyz = array_3d_to_2d(CEx,'yz')

    CEyxy = array_3d_to_2d(CEy,'xy')
    CEyxz = array_3d_to_2d(CEy,'xz')
    CEyyz = array_3d_to_2d(CEy,'yz')

    CEzxy = array_3d_to_2d(CEz,'xy')
    CEzxz = array_3d_to_2d(CEz,'xz')
    CEzyz = array_3d_to_2d(CEz,'yz')

    return Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz


def _grab_dpar_and_comp_all_CEi(vmax, dv, x1, x2, y1, y2, z1, z2, dpar_folder, dfields, vshock, project=False):
    """
    Wrapper function that loads correct particle data from presliced data and computes FPC

    See documentation for compute_hist_and_cor and comp_cor_over_x_multithread
    """

    from FPCAnalysis.data_dhybridr import get_dpar_from_bounds
    from FPCAnalysis.frametransform import shift_particles
    import gc

    dpar = get_dpar_from_bounds(dpar_folder,x1,x2)

    dpar = shift_particles(dpar, vshock)

    print("debug casting as np...",x1,x2)

    print("This worker is starting with x1: ",x1,' x2: ',x2,' y1: ',y1,' y2: ',y2,' z1: ', z1,' z2: ',z2)

    vx, vy, vz, totalPtcl, Hist, CEx, CEy, CEz = _comp_all_CEi(vmax, dv, x1, x2, y1, y2, z1, z2, dpar, dfields, vshock,checkFrameandGrabSubset=False)

    del dpar

    print("This worker is done with x1: ",x1,' x2: ',x2,' y1: ',y1,' y2: ',y2,' z1: ', z1,' z2: ',z2)
    if(project):
        import sys
        print("starting projection for ",x1,' x2: ',x2,' y1: ',y1,' y2: ',y2,' z1: ', z1,' z2: ',z2)
        Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz = project_CEi_hist(Hist, CEx, CEy, CEz)
        del CEx
        del CEy
        del CEz
        del Hist
        gc.collect()
        outputsize = sys.getsizeof([vx, vy, vz, totalPtcl, Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz])
        print("done with projection for ",x1,' x2: ',x2,' y1: ',y1,' y2: ',y2,' z1: ', z1,' z2: ',z2,' sizeofoutput: ', outputsize)
        return vx, vy, vz, totalPtcl, Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz
    else:
        return vx, vy, vz, totalPtcl, Hist, CEx, CEy, CEz

#TODO: update return documentation
def comp_cor_over_x_multithread(dfields, dpar_folder, vmax, dv, dx, vshock, xlim=None, ylim=None, zlim=None, max_workers = 8):
    """
    Computes distribution function and correlation wrt to given field for every slice in xx using multiprocessing

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    dpar_folder : string
        path to folder containing data from preslicedata.py
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    dv : float
        velocity space grid spacing
        (assumes square)
    dx : float
        integration box size (i.e. the slice size)
    vshock : float
        velocity of shock in x direction
    xlim : [float,float], opt
        upper and lower bounds of sweep
    ylim : [float,float], opt
        upper and lower bounds of integration box
    zlim : [float,float], opt
        upper and lower bounds of integration box

    Returns
    ------- 
    CEx_out : 4d array
        CEx(x; vz, vy, vx) data
    CEy_out : 4d array
        CEy(x; vz, vy, vx) data
    CEz_out : 4d array
        CEz(x; vz, vy, vx) data
    x_out : 1d array
        average x position of each slice
    Hist_out : 4d array
        f(x; vz, vy, vx) data
    vx : 3d array
        vx velocity grid
    vy : 3d array
        vy velocity grid
    vz : 3d array
        vz velocity grid
    num_par_out : 1d array
        number of particles in box
    """
    from concurrent.futures import ProcessPoolExecutor
    import time
    import gc

    #set up box bounds
    if xlim is not None:
        x1 = xlim[0]
        x2 = x1+dx
        xEnd = xlim[1]
    # If xlim is None, use lower x edge to upper x edge extents
    else:
        x1 = dfields['ex_xx'][0]
        x2 = x1 + dx
        xEnd = dfields['ex_xx'][-1]
    if ylim is not None:
        y1 = ylim[0]
        y2 = ylim[1]
    # If ylim is None, use lower y edge to lower y edge + dx extents
    else:
        y1 = dfields['ex_yy'][0]
        y2 = y1 + dx
    if zlim is not None:
        z1 = zlim[0]
        z2 = zlim[1]
    # If zlim is None, use lower z edge to lower z edge + dx extents
    else:
        z1 = dfields['ex_zz'][0]
        z2 = z1 + dx

    #build task array
    x1task = []
    x2task = []
    while(x2 <= xEnd):
        x1task.append(x1)
        x2task.append(x2)
        x1 += dx
        x2 += dx

    #make empty results array
    Histxy = [None for _tmp in x1task]
    Histxz = [None for _tmp in x1task]
    Histyz = [None for _tmp in x1task]
    CExxy = [None for _tmp in x1task]
    CExxz = [None for _tmp in x1task]
    CExyz = [None for _tmp in x1task]
    CEyxy = [None for _tmp in x1task]
    CEyxz = [None for _tmp in x1task]
    CEyyz = [None for _tmp in x1task]
    CEzxy = [None for _tmp in x1task]
    CEzxz = [None for _tmp in x1task]
    CEzyz = [None for _tmp in x1task]
    x_out = [None for _tmp in x1task]

    num_par_out = [None for _tmp in x1task]

    #do multithreading
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        futures = []
        jobidxs = []

        #queue up jobs
        for tskidx in range(0,len(x1task)): #if there is a free worker and job to do, give job
            print('queued scan pos-> x1: ',x1task[tskidx],' x2: ',x2task[tskidx],' y1: ',y1,' y2: ',y2,' z1: ', z1,' z2: ',z2)
            futures.append(executor.submit(_grab_dpar_and_comp_all_CEi, vmax, dv, x1task[tskidx], x2task[tskidx], y1, y2, z1, z2, dpar_folder, dfields, vshock, project=True))
            jobidxs.append(tskidx)

        #wait until finished
        print("Done queueing up processes, waiting until done...")
        not_finished = True
        while(not_finished):
            not_finished = False
            if(len(futures) >= 0):
                _i = 0
                while(_i < len(futures)):
                    if(not(futures[_i].done())):
                        not_finished = True
                        _i += 1
                    else:
                        tskidx = jobidxs[_i]
                        _output = futures[_i].result() #return vx, vy, vz, totalPtcl, Hist, CEx, CEy, CEz
                        print("Got result for x1: ",x1task[tskidx]," x2: ",x2task[tskidx],' npar:', _output[3])
                        vx = _output[0]
                        vy = _output[1]
                        vz = _output[2]
                        Histxy[tskidx] = _output[4]
                        Histxz[tskidx] = _output[5]
                        Histyz[tskidx] = _output[6]
                        CExxy[tskidx] = _output[7]
                        CExxz[tskidx] = _output[8]
                        CExyz[tskidx] = _output[9]
                        CEyxy[tskidx] = _output[10]
                        CEyxz[tskidx] = _output[11]
                        CEyyz[tskidx] = _output[12]
                        CEzxy[tskidx] = _output[13]
                        CEzxz[tskidx] = _output[14]
                        CEzyz[tskidx] = _output[15]
                        num_par_out[tskidx] = _output[3] 
                        x_out[tskidx] = (x2task[tskidx]+x1task[tskidx])/2.

                        #saves ram
                        print("Deleting future for x1: ",x1task[tskidx]," x2: ",x2task[tskidx])
                        del futures[_i]
                        del jobidxs[_i]

                        gc.collect()
                        print("Done deleting (and garbage collecting) future for x1: ",x1task[tskidx]," x2: ",x2task[tskidx])
                time.sleep(10.)

        print("Done with processes!")
        executor.shutdown() #will start to shut things down as resouces become free

        return CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz,x_out, Histxy,Histxz,Histyz, vx, vy, vz, num_par_out

def compute_correlation_over_x(dfields, dparticles, vmax, dv, dx, vshock, xlim=None, ylim=None, zlim=None):
    """
    Computes f(x; vy, vx), CEx(x; vy, vx), and CEx(x; vy, vx) along different slices (i.e. thin analysis boxes) of x

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    dparticles : dict
        xx vx yy vy data dictionary from readParticlesPosandVelocityOnly
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    dv : float
        spacing between points we sample in velocity space. (Square in vx, vy)
    dx : float
        width of x slice
    vshock : float
        velocity of shock in x direction
    xlim : array
        array of limits in x, defaults to None
    ylim : array
        array of limits in y, defaults to None
    zlim : array
        array of limits in z, defaults to None

    Returns
    -------
    CEx_out : 4d array
        CEx(x; vz, vy, vx) data
    CEy_out : 4d array
        CEy(x; vz, vy, vx) data
    CEz_out : 4d array
        CEz(x; vz, vy, vx) data
    x_out : 1d array
        average x position of each slice
    Hist_out : 4d array
        f(x; vz, vy, vx) data
    vx : 3d array
        vx velocity grid
    vy : 3d array
        vy velocity grid
    vz : 3d array
        vz velocity grid
    num_par_out : 1d array
        number of particles in box
    """

    CEx_out = []
    CEy_out = []
    CEz_out = []
    x_out = []
    Hist_out = []
    num_par_out = []

    if xlim is not None:
        x1 = xlim[0]
        x2 = x1+dx
        xEnd = xlim[1]
    # If xlim is None, use lower x edge to upper x edge extents
    else:
        x1 = dfields['ex_xx'][0]
        x2 = x1 + dx
        xEnd = dfields['ex_xx'][-1]
    if ylim is not None:
        y1 = ylim[0]
        y2 = ylim[1]
    # If ylim is None, use lower y edge to lower y edge + dx extents
    else:
        y1 = dfields['ex_yy'][0]
        y2 = y1 + dx
    if zlim is not None:
        z1 = zlim[0]
        z2 = zlim[1]
    # If zlim is None, use lower z edge to lower z edge + dx extents
    else:
        z1 = dfields['ex_zz'][0]
        z2 = z1 + dx

    while(x2 <= xEnd):
        print('scan pos-> x1: ',x1,' x2: ',x2,' y1: ',y1,' y2: ',y2,' z1: ', z1,' z2: ',z2)
        vx, vy, vz, totalPtcl, Hist, CEx = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, 'ex', 'x')
        vx, vy, vz, totalPtcl, Hist, CEy = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, 'ey', 'y')
        vx, vy, vz, totalPtcl, Hist, CEz = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, 'ez', 'z')
        print('number of particles found in box: ', totalPtcl)
        x_out.append(np.mean([x1,x2]))
        CEx_out.append(CEx)
        CEy_out.append(CEy)
        CEz_out.append(CEz)
        Hist_out.append(Hist)
        num_par_out.append(totalPtcl)
        x1 += dx
        x2 += dx

    return CEx_out, CEy_out, CEz_out, x_out, Hist_out, vx, vy, vz, num_par_out

def compute_correlation_over_x_field_aligned(dfields, dparticles, vmax, dv, dx, vshock, xlim=None, ylim=None, zlim=None):
    """
    Computes f(x; vy, vx), CEx(x; vy, vx), and CEx(x; vy, vx) along different slices (i.e. thin analysis boxes) of x

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    dparticles : dict
        xx vx yy vy data dictionary from readParticlesPosandVelocityOnly
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    dv : float
        spacing between points we sample in velocity space. (Square in vx, vy)
    dx : float
        width of x slice
    vshock : float
        velocity of shock in x direction
    xlim : array
        array of limits in x, defaults to None
    ylim : array
        array of limits in y, defaults to None
    zlim : array
        array of limits in z, defaults to None

    Returns
    -------
    CEperp2_out : 4d array
        CEperp2(x; vz, vy, vx) data
    CEperp1_out : 4d array
        CEperp1(x; vz, vy, vx) data
    CEpar_out : 4d array
        CEpar(x; vz, vy, vx) data
    x_out : 1d array
        average x position of each slice
    Hist_out : 4d array
        f(x; vz, vy, vx) data
    vperp2 : 3d array
        vx velocity grid
    vperp1 : 3d array
        vy velocity grid
    vpar : 3d array
        vz velocity grid
    num_par_out : 1d array
        number of particles in box
    """

    CEperp2_out = []
    CEperp1_out = []
    CEpar_out = []
    x_out = []
    Hist_out = []
    num_par_out = []

    if xlim is not None:
        x1 = xlim[0]
        x2 = x1+dx
        xEnd = xlim[1]
    # If xlim is None, use lower x edge to upper x edge extents
    else:
        x1 = dfields['ex_xx'][0]
        x2 = x1 + dx
        xEnd = dfields['ex_xx'][-1]
    if ylim is not None:
        y1 = ylim[0]
        y2 = ylim[1]
    # If ylim is None, use lower y edge to lower y edge + dx extents
    else:
        y1 = dfields['ex_yy'][0]
        y2 = y1 + dx
    if zlim is not None:
        z1 = zlim[0]
        z2 = zlim[1]
    # If zlim is None, use lower z edge to lower z edge + dx extents
    else:
        z1 = dfields['ex_zz'][0]
        z2 = z1 + dx

    while(x2 <= xEnd):
        print('scan pos-> x1: ',x1,' x2: ',x2,' y1: ',y1,' y2: ',y2,' z1: ', z1,' z2: ',z2)
        vperp2, vperp1, vpar, totalPtcl, Hist, CEperp2 = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'eperp2', 'eperp2') #TODO: distinguish eperp2 the field and eperp2 the basis key using different names here(repeat for eperp1 and epar)
        vperp2, vperp1, vpar, totalPtcl, Hist, CEperp1 = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'eperp1', 'eperp1')
        vperp2, vperp1, vpar, totalPtcl, Hist, CEpar = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'epar', 'epar')
        print('num particles in box: ', totalPtcl)
        x_out.append(np.mean([x1,x2]))
        CEperp2_out.append(CEperp2)
        CEperp1_out.append(CEperp1)
        CEpar_out.append(CEpar)
        Hist_out.append(Hist)
        num_par_out.append(totalPtcl)
        x1 += dx
        x2 += dx

    return CEperp2_out, CEperp1_out, CEpar_out, x_out, Hist_out, vperp2, vperp1, vpar, num_par_out


def get_3d_weights(xx, yy, zz, idxxx1, idxxx2, idxyy1, idxyy2, idxzz1, idxzz2, dfields, fieldkey):
    """
    Calculates the weight associated with trilinear interpolation

    Parameters
    ----------
    xx : float
        test xx position
    yy : float
        test yy position
    zz : float
        test zz position
    idx**(1/2) : int
        index of positional value of box corner (lower then upper value)
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of the field you want to correlate with
        ex,ey,ez,bx,by, or bz

    Returns
    -------
    w* : float
        weight associated with each corner of box
    """

    # get weights by 'volume fraction' of cell
    w1 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy)*(dfields[fieldkey+'_zz'][idxzz1]-zz))
    w2 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy)*(dfields[fieldkey+'_zz'][idxzz1]-zz))
    w3 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy)*(dfields[fieldkey+'_zz'][idxzz1]-zz))
    w4 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy)*(dfields[fieldkey+'_zz'][idxzz2]-zz))
    w5 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy)*(dfields[fieldkey+'_zz'][idxzz1]-zz))
    w6 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy)*(dfields[fieldkey+'_zz'][idxzz2]-zz))
    w7 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy)*(dfields[fieldkey+'_zz'][idxzz2]-zz))
    w8 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy)*(dfields[fieldkey+'_zz'][idxzz2]-zz))

    vol = w1+w2+w3+w4+w5+w6+w7+w8

    # if vol is still zero, try computing 2d weights. For now, we assume 2d in xx and yy. TODO: program ability to be 2d in xx/zz or yy/zz
    if(vol == 0 and dfields[fieldkey+'_zz'][idxzz1]-zz == 0 and dfields[fieldkey+'_zz'][idxzz2]-zz == 0):
        w1 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy))
        w2 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy1]-yy))
        w3 = abs((dfields[fieldkey+'_xx'][idxxx1]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy))
        w5 = abs((dfields[fieldkey+'_xx'][idxxx2]-xx)*(dfields[fieldkey+'_yy'][idxyy2]-yy))

        # these correspond to idxzz2 and thus are zero
        w4 = 0.
        w6 = 0.
        w7 = 0.
        w8 = 0.

        vol = w1+w2+w3+w4+w5+w6+w7+w8

    if(vol == 0.):
        print("Error in getting weights! Found a zero volume.")

    # normalize to one
    w1 /= vol
    w2 /= vol
    w3 /= vol
    w4 /= vol
    w5 /= vol
    w6 /= vol
    w7 /= vol
    w8 /= vol

    # debug (should sum to 1)
    if(False):
        print('sum of weights: ' + str(w1+w2+w3+w4+w5+w6+w7+w8))

    return w1, w2, w3, w4, w5, w6, w7, w8

def weighted_field_average(xx, yy, zz, dfields, fieldkey, changebasismatrix = None):
    """
    Wrapper function for _weighted_field_average.

    Used to correlate to fields in field aligned coordinates when relevant

    See _weighted_field_average documentation
    """

    fieldaligned_keys = ['epar','eperp1','eperp2','bpar','bperp1','bperp2']
    if(fieldkey in fieldaligned_keys):

        if(fieldkey[0] == 'e'):
            #grab vals in standard coordinates
            exval = _weighted_field_average(xx, yy, zz, dfields, 'ex')
            eyval = _weighted_field_average(xx, yy, zz, dfields, 'ey')
            ezval = _weighted_field_average(xx, yy, zz, dfields, 'ez')

            #convert to field aligned
            epar,eperp1,eperp2 = np.matmul(changebasismatrix,[exval,eyval,ezval])

            #return correct key
            if(fieldkey == 'epar'):
                return epar
            elif(fieldkey == 'eperp1'):
                return eperp1
            if(fieldkey == 'eperp2'):
                return eperp2

        elif(fieldkey[0] == 'b'):
            #grab vals in standard coordinates
            bxval = _weighted_field_average(xx, yy, zz, dfields, 'bx')
            byval = _weighted_field_average(xx, yy, zz, dfields, 'by')
            bzval = _weighted_field_average(xx, yy, zz, dfields, 'bz')

            #convert to field aligned
            bpar,bperp1,bperp2 = np.matmul(changebasismatrix,[bxval,byval,bzval])

            #return correct key
            if(fieldkey == 'bpar'):
                return bpar
            elif(fieldkey == 'bperp1'):
                return bperp1
            if(fieldkey == 'bperp2'):
                return bperp2

    else:
        return _weighted_field_average(xx, yy, zz, dfields, fieldkey)


def _weighted_field_average(xx, yy, zz, dfields, fieldkey):
    """
    Uses trilinear interpolation to estimate field value at given test location

    Assumes the sides of the box are all in either the xy, xz, or yz plane

    Parameters
    ----------
    xx : float
        test xx position
    yy : float
        test yy position
    zz : float
        test zz position
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of the field you want to correlate with
        ex,ey,ez,bx,by, or bz

    Returns
    -------
    fieldaverage : float
        field value at given test location found using trilinear interpolation
    """

    from FPCAnalysis.array_ops import find_two_nearest

    idxxx1, idxxx2 = find_two_nearest(dfields[fieldkey+'_xx'],xx)
    idxyy1, idxyy2 = find_two_nearest(dfields[fieldkey+'_yy'],yy)
    idxzz1, idxzz2 = find_two_nearest(dfields[fieldkey+'_zz'],zz)

    # find weights
    w1, w2, w3, w4, w5, w6, w7, w8 = get_3d_weights(xx, yy, zz, idxxx1, idxxx2,
                                    idxyy1, idxyy2, idxzz1, idxzz2, dfields, fieldkey)

    # take average of field
    tolerance = 0.001
    if(abs(w1+w2+w3+w4+w5+w6+w7+w8-1.0) >= tolerance):
        print("Warning: sum of weights in trilinear interpolation was not close enought to 1. Value was: " + str(w1+w2+w3+w4+w5+w6+w7+w8))
    fieldaverage = w1 * dfields[fieldkey][idxzz1][idxyy1][idxxx1]
    fieldaverage += w2 * dfields[fieldkey][idxzz1][idxyy1][idxxx2]
    fieldaverage += w3 * dfields[fieldkey][idxzz1][idxyy2][idxxx1]
    fieldaverage += w4 * dfields[fieldkey][idxzz2][idxyy1][idxxx1]
    fieldaverage += w5 * dfields[fieldkey][idxzz1][idxyy2][idxxx2]
    fieldaverage += w6 * dfields[fieldkey][idxzz2][idxyy2][idxxx2]
    fieldaverage += w7 * dfields[fieldkey][idxzz2][idxyy2][idxxx1]
    fieldaverage += w8 * dfields[fieldkey][idxzz2][idxyy1][idxxx2]

    return fieldaverage

#TODO: finish documentation
def compute_cprime_hist(dparticles, dfields, fieldkey, vmax, dv, useBoxFAC=True, altcorfields=None, beta=None, massratio=None, c = None, vxdown=None,vydown=None,vzdown=None):
    """
    Computes cprime for all particles passed to it

    Parameters
    ----------
    dparticles : dict
        particle data dictionary
    dfields : dict
        field data dictonary
    fieldkey : str
        name of the field you want to correlate with
        ex,ey,ez,bx,by, or bz
    vmax : float
        specifies signature domain in velocity space
        (assumes square and centered about zero)
    dv : float
        velocity space grid spacing
        (assumes square)
    useBoxFAC : bool, opt
        if TRUE, compute field aligned coordinate system using the average value of the magnetic field across the whole 
        transverse domain. Otherwise, will use local value of each particle to compute field aligned coordinate system
    altcorfields : dict, opt
        if not None, then all calculations will be performed using dfields except for the correlation. Used when computing
        fluct correlation in field aligned coordinates

    Returns
    -------
    cprimebinned : 3d array
        distribution function weighted by charge, particles velocity,
        and field value in integration box
    Hist : 3d array
        distribution function in box
    vx : 3d array
        vx velocity grid
    vy : 3d array
        vy velocity grid
    vz : 3d array
        vz velocity grid
    """
    from scipy.stats import binned_statistic_dd
    from FPCAnalysis.frametransform import lorentz_transform_v
    from FPCAnalysis.analysis import convert_fluc_to_par
    from FPCAnalysis.analysis import convert_fluc_to_local_par
    import copy

    if(fieldkey == 'ex' or fieldkey == 'bx'):
        vvkey = 'p1'
    elif(fieldkey == 'ey' or fieldkey == 'by'):
        vvkey = 'p2'
    elif(fieldkey == 'ez' or fieldkey == 'bz'):
        vvkey = 'p3'
    elif(fieldkey == 'epar' or fieldkey == 'bpar'):
        vvkey = 'ppar'
    elif(fieldkey == 'eperp1' or fieldkey == 'eperp1'):
        vvkey = 'pperp1'
    elif(fieldkey == 'eperp2' or fieldkey == 'eperp2'):
        vvkey = 'pperp2'

    if(altcorfields != None): altcorfields['Vframe_relative_to_sim'] = 0
    dfields['Vframe_relative_to_sim'] = 0 #TODO: reomve this key- we don't use it anymore!


    #change to field aligned basis if needed
    fieldaligned_keys = ['epar','eperp1','eperp2','bpar','bperp1','bperp2']
    if(fieldkey in fieldaligned_keys):
        #we assume particle data is passed in standard basis and would need to be converted to field aligned
        from FPCAnalysis.analysis import change_velocity_basis
        from FPCAnalysis.analysis import change_velocity_basis_local
        from FPCAnalysis.array_ops import find_nearest
        from FPCAnalysis.analysis import compute_field_aligned_coord

        #get smallest box that contains all particles (assumes using full yz domain)
        xx = np.min(dparticles['x1'][:])
        _xxidx = find_nearest(dfields['ex_xx'],xx)
        _x1 = dfields['ex_xx'][_xxidx]#get xx cell edge 1
        if(dfields['ex_xx'][_xxidx] > xx):
            _x1 = dfields['ex_xx'][_xxidx-1]
        xx = np.max(dparticles['x1'][:])
        _xxidx = find_nearest(dfields['ex_xx'],xx)
        _x2 = dfields['ex_xx'][_xxidx]#get xx cell edge 2
        if(dfields['ex_xx'][_xxidx] < xx):
            _x2 = dfields['ex_xx'][_xxidx+1]

        #use whole transverse box when computing field aligned
        if(useBoxFAC):
            dparticles = change_velocity_basis(dfields,dparticles,[_x1,_x2],[dfields['ex_yy'][0],dfields['ex_yy'][-1]],[dfields['ex_zz'][0],dfields['ex_zz'][-1]]) #WARNING: we also assume field aligned coordinates uses full yz domain in weighted field average!!!

            vparbasis, vperp1basis, vperp2basis = compute_field_aligned_coord(dfields,[_x1,_x2],[dfields['ex_yy'][0],dfields['ex_yy'][-1]],[dfields['ex_zz'][0],dfields['ex_zz'][-1]])
            _ = np.asarray([vparbasis,vperp1basis,vperp2basis]).T
            changebasismatrix = np.linalg.inv(_)

        #use local coordinates for field aligned
        else:
            dparticles, changebasismatrixes = change_velocity_basis_local(dfields,dparticles)
            pass

    else:
        changebasismatrix = None

    # compute cprime for each particle (often slowest part of code)
    cprimew = np.zeros(len(dparticles['x1']))
    if('q' in dparticles.keys()):
        q = dparticles['q'] 
    else:
        q = 1.
    for i in range(0, len(dparticles['x1'])):
        if(not(useBoxFAC)):
            changebasismatrix = changebasismatrixes[i]
        if(altcorfields == None):
            fieldval = weighted_field_average(dparticles['x1'][i], dparticles['x2'][i], dparticles['x3'][i], dfields, fieldkey, changebasismatrix = changebasismatrix)
        else:
            fieldval = weighted_field_average(dparticles['x1'][i], dparticles['x2'][i], dparticles['x3'][i], altcorfields, fieldkey, changebasismatrix = changebasismatrix)
        cprimew[i] = q*dparticles[vvkey][i]*fieldval
    cprimew = np.asarray(cprimew)

    # bin into cprime(vx,vy,vz)
    vxbins = np.arange(-vmax, vmax+dv, dv)
    vx = (vxbins[1:] + vxbins[:-1])/2.
    vybins = np.arange(-vmax, vmax+dv, dv)
    vy = (vybins[1:] + vybins[:-1])/2.
    vzbins = np.arange(-vmax, vmax+dv, dv)
    vz = (vzbins[1:] + vzbins[:-1])/2.

    if(vvkey in ['p1','p2','p3']):
        hist,_ = np.histogramdd((dparticles['p3'], dparticles['p2'], dparticles['p1']), bins=[vzbins, vybins, vxbins])
        cprimebinned,_ = np.histogramdd((dparticles['p3'], dparticles['p2'], dparticles['p1']), bins=[vzbins, vybins, vxbins], weights=cprimew)
    else:
        hist,_ = np.histogramdd((dparticles['pperp2'], dparticles['pperp1'], dparticles['ppar']), bins=[vzbins, vybins, vxbins])
        cprimebinned,_ = np.histogramdd((dparticles['pperp2'], dparticles['pperp1'], dparticles['ppar']), bins=[vzbins, vybins, vxbins], weights=cprimew)
    del cprimew

    # make the bins 3d arrays
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

    return cprimebinned, hist, vx, vy, vz

#TODO: add and track charge!!!
#TODO: rename vx, vy, vz to make sense irregardless of if data is in standard basis or field aligned basis
def compute_cor_from_cprime(cprimebinned, vx, vy, vz, dv, directionkey):
    """
    Computes correlation from cprime

    Parameters
    ----------
    cprimebinned : 3d array
        distribution function weighted by charge, particles velocity,
        and field value in integration box
    vx : 3d array
        vx velocity grid
    vy : 3d array
        vy velocity grid
    vz : 3d array
        vz velocity grid
    dv : float
        velocity space grid spacing
        (assumes square)
    directionkey : str
        direction we are taking the derivative w.r.t. (x,y,z)

    Returns
    -------
    cor : 3d array
        FPC data
    """
    if(directionkey == 'x'):
        axis = 2
        vv = vx
    elif(directionkey == 'y'):
        axis = 1
        vv = vy
    elif(directionkey == 'z'):
        axis = 0
        vv = vz
    elif(directionkey == 'epar'):
        axis = 2
        vv = vx
    elif(directionkey == 'eperp1'):
        axis = 1
        vv = vy
    elif(directionkey == 'eperp2'):
        axis = 0
        vv = vz

    cor = -vv/2.*np.gradient(cprimebinned, dv, edge_order=2, axis=axis) + cprimebinned / 2.
    return cor

#-----------------------------------------------------------------------------------------------------------------------
# Functions related to computing FPC if hist function is already computed
#-----------------------------------------------------------------------------------------------------------------------
def compute_fpc_from_dist(fieldval,hist,vx,vy,vz,vshock,directionkey,q=1.):
    """
    hist is 3d array (vz,vy,vx as axes)
    fieldval is avg field value
    """
    if(directionkey == 'x'):
        axis = 2
        vv = vx-vshock
        dv = vx[0,0,1]-vx[0,0,0]
    elif(directionkey == 'y'):
        axis = 1
        vv = vy
        dv = vy[0,1,0]-vy[0,0,0]
    elif(directionkey == 'z'):
        axis = 0
        vv = vz
        dv = vz[1,0,0]-vz[0,0,0]

    cor = -q*(vv**2./2)*np.gradient(hist, dv, edge_order=2, axis=axis)*fieldval

    totalPtcl = np.sum(hist)

    return vx, vy, vz, totalPtcl, hist, cor

def _comp_all_CEi_from_dist(x1, x2, y1, y2, z1, z2, ddist, dfields, vshock):
    #TODO: make 2D and 3D compatable

    #bin histograms and take field average
    _hist = []
    fieldkeys = ['ex','ey','ez','bx','by','bz']
    for fkey in fieldkeys:
        locals()[fkey+'avg'] = 0.
    num_field_points = 0
    for xxidx, x0 in enumerate(ddist['hist_xx']):
        if(x0 >= x1 and x0 <= x2):
            if(_hist == []):
                _hist = ddist['hist'][xxidx]
            else:
                _hist += ddist['hist'][xxidx]

            for fkey in fieldkeys:
                locals()[fkey+'avg'] += dfields[fkey][0,0,xxidx] #not 3d or 2d compatable

            num_field_points += 1

    for key in fieldkeys:
        locals()[key+'avg'] /= num_field_points #not 3d or 2d compatable

    vx = ddist['vx']
    vy = ddist['vy']
    vz = ddist['vz']
    for fkey in fieldkeys:
        dkey = fkey[-1]
        vx,vy,vz,totalPtcl,hist,locals()['C'+fkey] = compute_fpc_from_dist(locals()[fkey+'avg'],_hist,vx,vy,vz,vshock,dkey)

    return totalPtcl, _hist, locals()['Cex'], locals()['Cey'], locals()['Cez']

def compute_correlation_over_x_from_dist(ddist,dfields, vmax, dx, vshock, xlim=None, ylim=None, zlim=None, project=False):


    x_out = []
    num_par_out = []
    if(project == False):
        CEx_out = []
        CEy_out = []
        CEz_out = []
        Hist_out = []
    else:
        dfpckeys = ['Histvxvy','Histvxvz','Histvyvz','CExvxvy','CExvxvz','CExvyvz','CEyvxvy','CEyvxvz','CEyvyvz','CEzvxvy','CEzvxvz','CEzvyvz']
        dfpc = {}
    for key in dfpckeys:
        dfpc[key] = []
        CExvxvy_out = []
        CExvxvz_out = []
        CExvyvz_out = []
        CEyvxvy_out = []
        CEyvxvz_out = []
        CEyvyvz_out = []
        CEzvxvy_out = []
        CEzvxvz_out = []
        CEzvyvz_out = []
        Histvxvy_out = []
        Histvxvz_out = []
        Histvyvz_out = []



    if(dx < ddist['hist_xx'][1]-ddist['hist_xx'][0]):
        print("ERROR: dx is smaller than spacing between distribution functions")
        return #TODO raise error here

    if xlim is not None:
        x1 = xlim[0]
        x2 = x1+dx
        xEnd = xlim[1]
    # If xlim is None, use lower x edge to upper x edge extents
    else:
        x1 = ddist['hist_xx'][0]
        x2 = x1 + dx
        xEnd = ddist['hist_xx'][-1]
    if ylim is not None:
        y1 = ylim[0]
        y2 = ylim[1]
    # If ylim is None, use lower y edge to lower y edge + dx extents
    else:
        y1 = dfields['ex_yy'][0]
        y2 = y1 + dx
    if zlim is not None:
        z1 = zlim[0]
        z2 = zlim[1]
    # If zlim is None, use lower z edge to lower z edge + dx extents
    else:
        z1 = dfields['ex_zz'][0]
        z2 = z1 + dx

    while(x2 <= xEnd):
        print('scan pos-> x1: ',x1,' x2: ',x2,' y1: ',y1,' y2: ',y2,' z1: ', z1,' z2: ',z2)
        totalPtcl, hist, CEx, CEy, CEz = _comp_all_CEi_from_dist(x1, x2, y1, y2, z1, z2, ddist, dfields, vshock)
        print('num particles in box: ', totalPtcl)
        x_out.append(np.mean([x1,x2]))
        if(project == False):
            CEx_out.append(CEx)
            CEy_out.append(CEy)
            CEz_out.append(CEz)
            Hist_out.append(hist)
        else:
            Histvxvy,Histvxvz,Histvyvz,CExvxvy,CExvxvz,CExvyvz,CEyvxvy,CEyvxvz,CEyvyvz,CEzvxvy,CEzvxvz,CEzvyvz = project_CEi_hist(hist, CEx, CEy, CEz)
            for key in dfpckeys:
                dfpc[key].append(locals()[key])
        num_par_out.append(totalPtcl)
        x1 += dx
        x2 += dx

    vx = ddist['vx']
    vy = ddist['vy']
    vz = ddist['vz']

    if(project == False):
        return CEx_out, CEy_out, CEz_out, x_out, Hist_out, vx, vy, vz, num_par_out
    else:
        dfpc['num_par'] = num_par_out
        dfpc['xx'] = x_out
        dfpc['vx'] = ddist['vx']
        dfpc['vy'] = ddist['vy']
        dfpc['vz'] = ddist['vz']

        return dfpc

def project_and_store(vx,vy,vz,xx,CEx,CEy,CEz,Hist):
    """
    projects 4d data
    """

    dfpckeys = ['Histvxvy','Histvxvz','Histvyvz','CExvxvy','CExvxvz','CExvyvz','CEyvxvy','CEyvxvz','CEyvyvz','CEzvxvy','CEzvxvz','CEzvyvz']
    dfpc = {}
    for key in dfpckeys:
        dfpc[key] = []
    dfpc['xx'] = xx
    dfpc['vx'] = vx
    dfpc['vy'] = vy
    dfpc['vz'] = vz

    for xxidx, x0 in enumerate(xx):
        print("Projecting ",x0,' of ',xx[-1])
        Histvxvy,Histvxvz,Histvyvz,CExvxvy,CExvxvz,CExvyvz,CEyvxvy,CEyvxvz,CEyvyvz,CEzvxvy,CEzvxvz,CEzvyvz = project_CEi_hist(Hist[xxidx], CEx[xxidx], CEy[xxidx], CEz[xxidx])
        for key in dfpckeys:
            dfpc[key].append(locals()[key])

    return dfpc
