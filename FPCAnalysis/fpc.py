# fpc.py>

# functions related to computing FPC

import numpy as np


def compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2,
                            dpar, dfields, vshock, fieldkey, directionkey,checkFrameandGrabSubset=True):
    """
    Computes distribution function and correlation wrt to given field

    Function will automatically shift frame of particles if particles are in simulation frame.
    However, it is more efficient to shift particles before calling this function.

    TODO: this function is very non optimized, mainly due to repeated searching of dparticles array.
    Should optimize this by passing particle subsets (ie boxes of particles)
    to do FPC of

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
    vshock : float
        velocity of shock in x direction
    fieldkey : str
        name of the field you want to correlate with
        ex,ey,ez,bx,by, or bz
    directionkey : str
        name of direction you want to take the derivative with respect to
        x,y,or z
        *should match the direction of the fieldkey*
    checkFrameandGrabSubset : bool(opt) TODO: RENAME THIS VAR
        check if all given particles are in box and in correct frame
        should typically be true unless trying to save RAM

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
    totalFieldpts : float
        total number of field gridpoitns in the correlation box
    Hist : 3d array
        distribution function in box
    Cor : 3d array
        velocity space sigature data in box
    """
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


    #print(dpar.keys())
    #print(('q' in dpar.keys()))
    # # find average E field based on provided bounds #TODO: remove this
    # gfieldptsx = (x1 <= dfields[fieldkey+'_xx']) & (dfields[fieldkey+'_xx'] <= x2)
    # gfieldptsy = (y1 <= dfields[fieldkey+'_yy']) & (dfields[fieldkey+'_yy'] <= y2)
    # gfieldptsz = (z1 <= dfields[fieldkey+'_zz']) & (dfields[fieldkey+'_zz'] <= z2)
    #
    # goodfieldpts = []
    # for i in range(0, len(dfields['ex_xx'])):
    #     for j in range(0, len(dfields['ex_yy'])):
    #         for k in range(0, len(dfields['ex_zz'])):
    #             if(gfieldptsx[i] and gfieldptsy[j] and gfieldptsz[k]):
    #                 goodfieldpts.append(dfields[fieldkey][k][j][i])

    # define mask that includes particles within range
    #print('debug: ', x1,x2,y1,y2,z1,z2,'more debug: ',type(dpar['x1']),len(dpar['x1']),type(dpar['x2']),len(dpar['x2']),type(dpar['x3']),len(dpar['x3']))
    #gptsparticle = (x1 <= dpar['x1']) & (dpar['x1'] <= x2) & (y1 <= dpar['x2']) & (dpar['x2'] <= y2) & (z1 <= dpar['x3']) & (dpar['x3'] <= z2)

    if(checkFrameandGrabSubset):
        gptsparticle = (x1 <= dpar['x1']) & (dpar['x1'] <= x2) & (y1 <= dpar['x2']) & (dpar['x2'] <= y2) & (z1 <= dpar['x3']) & (dpar['x3'] <= z2)

        # shift particle data to shock frame if needed TODO:  clean this up
        #TODO: avoid doing this, it is very inefficient with RAM
        if(dfields['Vframe_relative_to_sim'] == vshock and dpar['Vframe_relative_to_sim'] == 0.): #TODO: use shift particles function
            dpar_p1 = np.asarray(dpar['p1'][gptsparticle][:])
            dpar_p1 -= vshock
            dpar_p2 = np.asarray(dpar['p2'][gptsparticle][:])
            dpar_p3 = np.asarray(dpar['p3'][gptsparticle][:])
        elif(dpar['Vframe_relative_to_sim'] != vshock):
            "WARNING: particles were not in simulation frame or provided vshock frame. This FPC is probably incorrect..."
        else:
            dpar_p1 = np.asarray(dpar['p1'][gptsparticle][:])
            dpar_p2 = np.asarray(dpar['p2'][gptsparticle][:])
            dpar_p3 = np.asarray(dpar['p3'][gptsparticle][:])

        totalPtcl = np.sum(gptsparticle)

        # # avgfield = np.average(goodfieldpts)
        # totalFieldpts = np.sum(goodfieldpts)

        if(dfields['Vframe_relative_to_sim'] != vshock):
            "WARNING: dfields is not in the same frame as the provided vshock"

        # build dparticles subset using shifted particle data
        # TODO: this isnt clean code (using dpar_p1/2/3 'multiple times' in histogram and in compute_cprime)
        dparsubset = {
          'p1': dpar_p1,
          'p2': dpar_p2,
          'p3': dpar_p3,
          'x1': dpar['x1'][gptsparticle][:],
          'x2': dpar['x2'][gptsparticle][:],
          'x3': dpar['x3'][gptsparticle][:],
          'Vframe_relative_to_sim': dpar['Vframe_relative_to_sim']
        }

        if('q' in dpar.keys()):
            dparsubset['q'] =dpar['q']

        cprimebinned, hist, vx, vy, vz = compute_cprime_hist(dparsubset, dfields, fieldkey, vmax, dv)
        del dparsubset

    else:
        gptsparticle = (x1 <= dpar['x1']) & (dpar['x1'] <= x2) & (y1 <= dpar['x2']) & (dpar['x2'] <= y2) & (z1 <= dpar['x3']) & (dpar['x3'] <= z2)
        try: #This is hacky TODO: clean this up by simply returning hist and cor arrays full of zeros
            dparsubset = {
              'p1': dpar['p1'][gptsparticle][:],
              'p2': dpar['p2'][gptsparticle][:],
              'p3': dpar['p3'][gptsparticle][:],
              'x1': dpar['x1'][gptsparticle][:],
              'x2': dpar['x2'][gptsparticle][:],
              'x3': dpar['x3'][gptsparticle][:],
              'Vframe_relative_to_sim': dpar['Vframe_relative_to_sim']
            }

            if('q' in dpar.keys()): #TODO: fix redundancy with this
                dparsubset['q'] = dpar['q']
        except:
            dparsubset = {
              'p1': np.asarray([0.]),
              'p2': np.asarray([0.]),
              'p3': np.asarray([0.]),
              'x1': np.asarray([0.]),
              'x2': np.asarray([0.]),
              'x3': np.asarray([0.]),
              'Vframe_relative_to_sim': dpar['Vframe_relative_to_sim']
            }

            if('q' in dpar.keys()):
                dparsubset['q'] = dpar['q']

            totalPtcl = len(dpar['p1'][:])
            totalFieldpts = -1 # TODO just remove this varaible, doesn't make sense anymore
            cprimebinned, hist, vx, vy, vz = compute_cprime_hist(dparsubset, dfields, fieldkey, vmax, dv)

            cor = compute_cor_from_cprime(cprimebinned, vx, vy, vz, dv, directionkey)
            del cprimebinned

            #make data empty
            totalPtcl = 0.
            totalFieldpts = -1
            hist = np.zeros(hist.shape)
            cor = np.zeros(hist.shape)

            return vx, vy, vz, totalPtcl, totalFieldpts, hist, cor

        totalPtcl = len(dpar['p1'][:])
        totalFieldpts = -1 # TODO just remove this varaible, doesn't make sense anymore
        #TODO: check frame!!!!!!
        dpar['p1'] -= vshock #TODO: clean this up
        cprimebinned, hist, vx, vy, vz = compute_cprime_hist(dparsubset, dfields, fieldkey, vmax, dv)

    cor = compute_cor_from_cprime(cprimebinned, vx, vy, vz, dv, directionkey)
    del cprimebinned

    totalFieldpts = -1
    return vx, vy, vz, totalPtcl, totalFieldpts, hist, cor

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
    vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEx = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'ex', 'x',checkFrameandGrabSubset=checkFrameandGrabSubset)
    vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEy = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'ey', 'y',checkFrameandGrabSubset=checkFrameandGrabSubset)
    vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEz = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'ez', 'z',checkFrameandGrabSubset=checkFrameandGrabSubset)

    return vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEx, CEy, CEz

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
    from lib.array_ops import array_3d_to_2d

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

    from lib.data_h5 import get_dpar_from_bounds
    import gc

    dpar = get_dpar_from_bounds(dpar_folder,x1,x2)

    print("This worker is starting with x1: ",x1,' x2: ',x2,' y1: ',y1,' y2: ',y2,' z1: ', z1,' z2: ',z2)

    vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEx, CEy, CEz = _comp_all_CEi(vmax, dv, x1, x2, y1, y2, z1, z2, dpar, dfields, vshock,checkFrameandGrabSubset=False)

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
        outputsize = sys.getsizeof([vx, vy, vz, totalPtcl, totalFieldpts, Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz])
        print("done with projection for ",x1,' x2: ',x2,' y1: ',y1,' y2: ',y2,' z1: ', z1,' z2: ',z2,' sizeofoutput: ', outputsize)
        return vx, vy, vz, totalPtcl, totalFieldpts, Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz
    else:
        return vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEx, CEy, CEz

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
    ------- #TODO: update return documentation
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
                        _output = futures[_i].result() #return vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEx, CEy, CEz
                        print("Got result for x1: ",x1task[tskidx]," x2: ",x2task[tskidx],' npar:', _output[3])
                        vx = _output[0]
                        vy = _output[1]
                        vz = _output[2]
                        Histxy[tskidx] = _output[5]
                        Histxz[tskidx] = _output[6]
                        Histyz[tskidx] = _output[7]
                        CExxy[tskidx] = _output[8]
                        CExxz[tskidx] = _output[9]
                        CExyz[tskidx] = _output[10]
                        CEyxy[tskidx] = _output[11]
                        CEyxz[tskidx] = _output[12]
                        CEyyz[tskidx] = _output[13]
                        CEzxy[tskidx] = _output[14]
                        CEzxz[tskidx] = _output[15]
                        CEzyz[tskidx] = _output[16]
                        num_par_out[tskidx] = _output[3] #TODO: use consistent ordering of variables
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




def comp_cor_over_x_multithreadv0(dfields, dpar_folder, vmax, dv, dx, vshock, xlim=None, ylim=None, zlim=None, max_workers = 8):
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

    #empty results array
    CEx_out = [None for _tmp in x1task]
    CEy_out = [None for _tmp in x1task]
    CEz_out = [None for _tmp in x1task]
    x_out = [None for _tmp in x1task]
    Hist_out = [None for _tmp in x1task]
    num_par_out = [None for _tmp in x1task]

    #do multithreading
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        futures = []
        jobids = [] #array to track where in results array result returned by thread should go
        num_working = 0
        tasks_completed = 0
        taskidx = 0

        while(tasks_completed < len(x1task)): #while there are jobs to do
            if(num_working < max_workers and taskidx < len(x1task)): #if there is a free worker and job to do, give job
                print('started scan pos-> x1: ',x1task[taskidx],' x2: ',x2task[taskidx],' y1: ',y1,' y2: ',y2,' z1: ', z1,' z2: ',z2)
                futures.append(executor.submit(_grab_dpar_and_comp_all_CEi, vmax, dv, x1task[taskidx], x2task[taskidx], y1, y2, z1, z2, dpar_folder, dfields, vshock))
                jobids.append(taskidx)
                taskidx += 1
                num_working += 1
            else: #otherwise
                exists_idle = False
                nft = len(futures)
                _i = 0
                while(_i < nft):
                    popped_element = False
                    if(futures[_i].done()): #if done get result
                        print("Found done process,", _i, ", grabbing results...")

                        #get results and place in return vars
                        resultidx = jobids[_i]
                        _output = futures[_i].result() #return vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEx, CEy, CEz
                        vx = _output[0]
                        vy = _output[1]
                        vz = _output[2]
                        num_par_out[resultidx] = _output[3] #TODO: use consistent ordering of variables
                        Hist_out[resultidx] = _output[5]
                        CEx_out[resultidx] = _output[6]
                        CEy_out[resultidx] = _output[7]
                        CEz_out[resultidx] = _output[8]
                        x_out[resultidx] = (x2task[resultidx]+x1task[resultidx])/2.

                        #update multithreading state vars
                        num_working -= 1
                        tasks_completed += 1
                        exists_idle = True
                        futures.pop(_i)
                        jobids.pop(_i)
                        popped_element = True #want to carefully iterate
                        nft -= 1
                        print('done with process,',_i,'ended scan pos-> x1: ',x1task[resultidx],' x2: ',x2task[resultidx],' y1: ',y1,' y2: ',y2,' z1: ', z1,' z2: ',z2,'num particles in box: ', _output[3])

                    if(not(popped_element)):
                        _i += 1


                if(not(exists_idle)):
                    time.sleep(0.001)

    return CEx_out, CEy_out, CEz_out, x_out, Hist_out, vx, vy, vz, num_par_out


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
        vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEx = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'ex', 'x')
        vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEy = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'ey', 'y')
        vx, vy, vz, totalPtcl, totalFieldpts, Hist, CEz = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'ez', 'z')
        print('num particles in box: ', totalPtcl)
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
        vperp2, vperp1, vpar, totalPtcl, totalFieldpts, Hist, CEperp2 = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'eperp2', 'eperp2') #TODO: distinguish eperp2 the field and eperp2 the basis key using different names here(repeat for eperp1 and epar)
        vperp2, vperp1, vpar, totalFieldpts, Hist, CEperp1 = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'eperp1', 'eperp1')
        vperp2, vperp1, vpar, totalFieldpts, Hist, CEpar = compute_hist_and_cor(vmax, dv, x1, x2, y1, y2, z1, z2, dparticles, dfields, vshock, 'epar', 'epar')
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
        #from lib.analysis import compute_field_aligned_coord
        #from lib.array_ops import find_nearest
        # _xxidx = find_nearest(dfields['ex_xx'],xx) #TODO: make a function that gets cell edge and call it here
        # _x1 = dfields['ex_xx'][_xxidx]#get xx cell edge 1
        # if(dfields['ex_xx'][_xxidx] - _x1 < 0):#get xx cell edge 2
        #     _x2 = _x1
        #     _x1 = dfields['ex_xx'][_xxidx-1]
        # else:
        #     _x2 = dfields['ex_xx'][_xxidx+1]
        # vparbasis, vperp1basis, vperp2basis = compute_field_aligned_coord(dfields,[_x1,_x2],[dfields['ex_yy'][0],dfields['ex_yy'][-1]],[dfields['ex_zz'][0],dfields['ex_zz'][-1]]) #WARNING: we also assume that field aligned is defined using the whole yz domain in compute cprime function as well!!!
        # _ = np.asarray([vparbasis,vperp1basis,vperp2basis]).T
        # changebasismatrix = np.linalg.inv(_)

        if(fieldkey[0] == 'e'):
            #grab vals in standard coordinates
            exval = _weighted_field_average(xx, yy, zz, dfields, 'ex')
            eyval = _weighted_field_average(xx, yy, zz, dfields, 'ey')
            ezval = _weighted_field_average(xx, yy, zz, dfields, 'ez')

            #convert to field aligned
            epar,eperp1,eperp2 = np.matmul(changebasismatrix,[exval,eyval,ezval])

            #return correct key
            #TODO: this code is kinda redundant, as we compute field aligned in all directions, we should consider optimizing this, but it would be difficult
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
            #TODO: this code is kinda redundant, as we compute field aligned in all directions, we should consider optimizing this, but it would be difficult
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

    from lib.array_ops import find_two_nearest

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


def compute_cprime_hist(dparticles, dfields, fieldkey, vmax, dv):
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

    #check if particle data is in correct coordinates
    if(not(vvkey in dparticles.keys())):
        #we assume particle data is passed in standard basis and would need to be converted to field aligned
        from lib.analysis import change_velocity_basis
        from lib.analysis import change_velocity_basis_local
        from lib.array_ops import find_nearest
        from lib.analysis import compute_field_aligned_coord

        xx = np.min(dparticles['x1'][:])
        _xxidx = find_nearest(dfields['ex_xx'],xx) #TODO: make a function that gets cell edge and call it here
        _x1 = dfields['ex_xx'][_xxidx]#get xx cell edge 1
        if(dfields['ex_xx'][_xxidx] > xx):
            _x1 = dfields['ex_xx'][_xxidx-1]
        xx = np.max(dparticles['x1'][:])
        _xxidx = find_nearest(dfields['ex_xx'],xx)
        _x2 = dfields['ex_xx'][_xxidx]#get xx cell edge 2
        if(dfields['ex_xx'][_xxidx] < xx):
            _x2 = dfields['ex_xx'][_xxidx+1]

        _q_in_keys = False
        if('q' in dparticles.keys()): #quick fix. TODO: implement this better in such a way that no keys are dropped...
            _qtemp = dparticles['q']
            _q_in_keys = True
        dparticles = change_velocity_basis(dfields,dparticles,[_x1,_x2],[dfields['ex_yy'][0],dfields['ex_yy'][-1]],[dfields['ex_zz'][0],dfields['ex_zz'][-1]]) #WARNING: we also assume field aligned coordinates uses full yz domain in weighted field average!!!
        if(_q_in_keys):
            dparticles['q'] = _qtemp

        vparbasis, vperp1basis, vperp2basis = compute_field_aligned_coord(dfields,[_x1,_x2],[dfields['ex_yy'][0],dfields['ex_yy'][-1]],[dfields['ex_zz'][0],dfields['ex_zz'][-1]])
        _ = np.asarray([vparbasis,vperp1basis,vperp2basis]).T
        changebasismatrix = np.linalg.inv(_)

    else:
        changebasismatrix = None


    #print('Debug: x1,',np.min(dparticles['x1']),' x2,',np.max(dparticles['x1']),' y1,',np.min(dparticles['x2']),' y2,',np.max(dparticles['x2']),' z1,',np.min(dparticles['x3']),' z2,',np.max(dparticles['x3']))

    # compute cprime for each particle #TODO: make this block more efficient, it is the slowest part of the code
    # import time
    # start = time.time()
    #TODO: improve performance of this block vvvv---------------------------------------------------------------------------
    cprimew = np.zeros(len(dparticles['x1']))
    #print('just b4', dparticles.keys())
    if('q' in dparticles.keys()):
        #print("Charge was found!")
        q = dparticles['q']  #TODO: assign q to gkeyll and dHybridR data
    else:
        #print("Warning! Charge was not specified for dpar dict. Defaulting to q = 1.")
        q = 1.
    for i in range(0, len(dparticles['x1'])):
        fieldval = weighted_field_average(dparticles['x1'][i], dparticles['x2'][i], dparticles['x3'][i], dfields, fieldkey,changebasismatrix = changebasismatrix)
        cprimew[i] = q*dparticles[vvkey][i]*fieldval
    cprimew = np.asarray(cprimew)
    #end = time.time()
    # print(end - start)
    #TODO: improve performance of this block ^^^^---------------------------------------------------------------------------

    #TODO: rename a lot of this, as it doesn't make sense in field aligned coordinates
    # bin into cprime(vx,vy,vz) #TODO: use function for this block (it's useful elsewhere to build distribution functions)
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