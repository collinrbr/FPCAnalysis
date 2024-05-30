import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np
import copy
import pickle

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa
import lib.arrayaux as ao #array operations
import lib.fpcaux as fpc

#Attempt to stop font not found errors; TODO: try something else to remove these errors
#import matplotlib.font_manager as fm
#fm._rebuild()

#--------------------------------------------------------------------------------------------------------------------------------
# Data path and user params
#--------------------------------------------------------------------------------------------------------------------------------
framenum = '700'
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]
verbose = True
printprogress = True

#next two should be set to false typically!
loaddebugsubset = False #load less particles for debug purposes!!! (load about 1/25 of  particles)
usedebugfields = True #load less detailed fields to save on computational resources (only for locframe routines)

loadfromfile = True

#parameters below are ignored if loadfromfile is True
vmaxion = 25.
dvion = 1.
vmaxelec = 15.
dvelec = 1.

#flnms below are ignored if loadfpcnc is False
pathfpcdata = ''
ionflucflnm = 'analysisfiles/ncsweeps/ionfluc.nc'
ionfacflnm = 'analysisfiles/ncsweeps/ionfac.nc'
iontotflnm = 'analysisfiles/ncsweeps/iontot.nc'
ionfacflucflnm = 'analysisfiles/ncsweeps/ionfacfluc.nc'
ionfacfluclocalflnm = 'analysisfiles/ncsweeps/ionfacfluclocal.nc'
ionfacavglocframeflnm = 'analysisfiles/ncsweeps/ionfacavglocframe.nc'
ionfacfluclocframeflnm = 'analysisfiles/ncsweeps/ionfacfluclocframe.nc'
elecflucflnm = 'analysisfiles/ncsweeps/elecfluc.nc'
electotflnm = 'analysisfiles/ncsweeps/electot.nc'
elecfacflnm =  'analysisfiles/ncsweeps/elecfac.nc'
elecfacflucflnm = 'analysisfiles/ncsweeps/elecfacfluc.nc'
elecfacfluclocalflnm = 'analysisfiles/ncsweeps/elecfacfluclocal.nc'
ionfacfluclowpassflnm = 'analysisfiles/ncsweeps/ionfacfluclowpassflnm.nc'
ionfacfluchighdetrendflnm = 'analysisfiles/ncsweeps/ionfacfluchighdetrend.nc'
elecfacfluclowpassflnm = 'analysisfiles/ncsweeps/elecfacfluclowpass.nc'
elecfacfluchighdetrendflnm  = 'analysisfiles/ncsweeps/elecfacfluchighdetrend.nc'
elecfacavglocframeflnm = 'analysisfiles/ncsweeps/elecfacavglocframe.nc'
elecfacfluclocframeflnm = 'analysisfiles/ncsweeps/elecfacfluclocframe.nc'
ion3vhistflnm = 'analysisfiles/ncsweeps/ionhistsweep.pickle'
elec3vhistflnm = 'analysisfiles/ncsweeps/elechistsweep.pickle'
dflowflnm = 'dflow.pickle' #TODO: move to subfolder in analysisfiles folder  

#--------------------------------------------------------------------------------------------------------------------------------
# End data path and user params
#--------------------------------------------------------------------------------------------------------------------------------
#load particle, fields, current and transform data, and compute 'flow' ie first moment in of dist in each cell----------------------------
normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)
dden = ld.load_den(flpath,framenum,normalize=normalize)

if(not(loadfromfile)):
    dpar_elec, dpar_ion = ld.load_particles(flpath,framenum,normalizeVelocity=normalize,loaddebugsubset=loaddebugsubset)

#compute timestep
params = ld.load_params(flpath,framenum)
bnorm = np.mean((dfields['bx'][:,:,-10:]**2+dfields['by'][:,:,-10:]**2+dfields['bz'][:,:,-10:]**2)**0.5)
params = ld.load_params(flpath,framenum)
dt = params['c']/params['comp'] #in units of wpe
sigma_ion = params['sigma']*params['me']/params['mi']
wpe_over_wce = 1./(np.sqrt(sigma_ion)*np.sqrt(params['mi']/params['me']))
wce_over_wci = params['mi']/params['me']
c = params['c']/(params['c']**2*params['sigma']) #c in unit of va
stride = 100 #TODO: automate loading
dt = stride*dt/(wpe_over_wce*wce_over_wci) #originally in wpe, now in units of wci

massrat = 625

#compute shock velocity and boost to shock rest frame
dfields_many_frames = {'frame':[],'dfields':[]}
for _num in frames:
    num = int(_num)
    d = ld.load_fields(flpath,_num,normalizeFields=normalize)
    dfields_many_frames['dfields'].append(d)
    dfields_many_frames['frame'].append(num)
vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)
dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame
inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input' #TODO: make all these paths load from a separate text file
inputs = ld.load_input(inputpath)
beta0 = aa.compute_beta0(params,inputs)

if(not(loadfromfile)):
    dpar_ion = ft.shift_particles(dpar_ion, vshock, beta0, params['mi']/params['me'], isIon=True)
    dpar_elec = ft.shift_particles(dpar_elec, vshock, beta0, params['mi']/params['me'], isIon=False)

#get average field value for each yz slice
dfavg = aa.get_average_fields_over_yz(dfields)
dfluc = aa.remove_average_fields_over_yz(dfields)

if(usedebugfields and not(loadfromfile)):
    startidx_xx = 0#ao.find_nearest(dfields['ex_xx'],5.)
    startidx_yy = 0
    startidx_zz = 0
    endidx_xx = ao.find_nearest(dfields['ex_xx'],12.)+1
    endidx_yy = len(dfields['ex_yy'])
    endidx_zz = len(dfields['ex_zz'])
    startidxs = [startidx_zz,startidx_yy,startidx_xx]
    endidxs = [endidx_zz,endidx_yy,endidx_xx]
    
    import copy
    _dfields = copy.deepcopy(dfields)
    _dfluc = copy.deepcopy(dfluc)
    _dfavg = copy.deepcopy(dfavg)
    _dfields = ao.subset_dict(_dfields,startidxs,endidxs,planes=['z','y','x'])
    _dfluc = ao.subset_dict(_dfluc,startidxs,endidxs,planes=['z','y','x'])
    _dfavg = ao.subset_dict(_dfavg,startidxs,endidxs,planes=['z','y','x'])

    _dfields = ao.avg_dict(_dfields,binidxsz=[1,1,12],planes=['z','y','x'])
    _dfluc = ao.avg_dict(_dfluc,binidxsz=[1,1,12],planes=['z','y','x'])
    _dfavg = ao.avg_dict(_dfavg,binidxsz=[1,1,12],planes=['z','y','x'])
else:
    _dfields = dfields
    _dfavg = dfavg
    _dfluc = dfluc
#load fpc and hist data and save in *.nc format (netcdf4)---------------------------------------------------------------------------------
#TODO: low/high pass fluc sweeps (and fac/ fac aligned for both of these sweeps)
if(not(loadfromfile)):
    import os
    os.system("mkdir analysisfiles")
    os.system("mkdir analysisfiles/ncsweeps")

    #print("Computing dflow from particle data... (has runtime of order hour(s) for big runs as there are many particles to sort in to many bins)")
    #dflow = aa.compute_dflow(dfields, dpar_ion, dpar_elec) #compute fluid velocity from particle data
    #dflowflnm = 'dflow.pickle'
    #fileout = open(dflowflnm, 'wb')
    #pickle.dump(dflow, fileout)
    #fileout.close()

    dxsweep = 25*dfields['ex_xx'][1]-dfields['ex_xx'][0]
    xstartsweep = 0
    endsweeppos = 12#(dfields['ex_xx'][-1])*(2./3.)
    nsweep = int(endsweeppos/dxsweep)

    if(verbose):print("dxsweep: ", dxsweep)

    #We use entire transverse domain
    y1 = 0
    y2 = 5
    z1 = -999999
    z2 = 999999
    
    #make array to save and store 3d hist data
    ionhistsweep = []
    elechistsweep = []

    print('Saving all *.nc and *.pickle files to current directory...')
    pathfpcdata = ''

    #compute CEi tot
    #if(verbose):print("Running CEi tot sweep for ions")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep)
    #    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'ex', 'x')
    #    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'ey', 'y')
    #    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'ez', 'z')
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/iontot.nc'
    #iontotflnm = flnm
    #ld.project_and_save(dfields, xout, vxi, vyi, vzi, histout, corexout, coreyout, corezout, flnm)

    #ionhistsweep = np.asarray(copy.deepcopy(histout))
    #ion3vhistflnm = 'analysisfiles/ncsweeps/ionhistsweep.pickle'
    #fileout = open(ion3vhistflnm, 'wb')
    #pickle.dump(ionhistsweep, fileout)
    #fileout.close()

    #if(verbose):print("Running CEi fluc sweep for ions")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep)
    #    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfluc, 'ex', 'x')
    #    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfluc, 'ey', 'y')
    #    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfluc, 'ez', 'z')
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/ionfluc.nc'
    #ionflucflnm = flnm
    #ld.project_and_save(dfields, xout, vxi, vyi, vzi, histout, corexout, coreyout, corezout, flnm)

    #if(verbose):print("Running CEi fluc fac sweep for ions")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep)
    #    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'epar', 'x',altcorfields=dfluc)
    #    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp1', 'y',altcorfields=dfluc)
    #    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp2', 'z',altcorfields=dfluc)
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/ionfacfluc.nc'
    #ionfacflucflnm = flnm
    #ld.project_and_save(dfields, xout, vxi, vyi, vzi, histout, corexout, coreyout, corezout, flnm)

    #if(verbose):print("Running CEi fac fluc local sweep for ions")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep)
    #    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'epar', 'x', useBoxFAC=False, altcorfields=dfluc)
    #    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp1', 'y', useBoxFAC=False, altcorfields=dfluc)
    #    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp2', 'z', useBoxFAC=False, altcorfields=dfluc)
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/ionfacfluclocal.nc'
    #ionfacfluclocalflnm = flnm
    #ld.project_and_save(dfields, xout, vxi, vyi, vzi, histout, corexout, coreyout, corezout, flnm)

    #if(verbose):print("Running CEi tot sweep for elecs")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep)
    #    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'ex', 'x')
    #    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'ey', 'y')
    #    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'ez', 'z')
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/electot.nc'
    #electotflnm = flnm
    #ld.project_and_save(dfields, xout, vxe, vye, vze, histout, corexout, coreyout, corezout, flnm)

    #elechistsweep = np.asarray(copy.deepcopy(histout))
    #elec3vhistflnm = 'analysisfiles/ncsweeps/elechistsweep.pickle'
    #fileout = open(elec3vhistflnm, 'wb')
    #pickle.dump(elechistsweep, fileout)
    #fileout.close()

    #if(verbose):print("Running CEi fluc sweep for elecs")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep)
    #    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfluc, 'ex', 'x')
    #    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfluc, 'ey', 'y')
    #    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfluc, 'ez', 'z')
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/elecfluc.nc'
    #elecflucflnm = flnm
    #ld.project_and_save(dfields, xout, vxe, vye, vze, histout, corexout, coreyout, corezout, flnm)

    ##TODO: rename variables in project and save
    #if(verbose):print("Running CEi fluc fac sweep for elecs")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep) #TODO: verbose statements for all
    #    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'epar', 'x',altcorfields=dfluc)
    #    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp1', 'y',altcorfields=dfluc)
    #    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp2', 'z',altcorfields=dfluc)
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/elecfacfluc.nc'
    #elecfacflucflnm = flnm
    #ld.project_and_save(dfields, xout, vxe, vye, vze, histout, corexout, coreyout, corezout, flnm)

    ##TODO: rename variables in project and save
    ##TODO: rename variables at rhs of fpc.compute_hist_and_cor (in particular cor variables)
    #if(verbose):print("Running CEi fac fluc local sweep for elecs")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep) #TODO: verbose statements for all
    #    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'epar', 'x', useBoxFAC=False, altcorfields=dfluc)
    #    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp1', 'y', useBoxFAC=False, altcorfields=dfluc)
    #    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp2', 'z', useBoxFAC=False, altcorfields=dfluc)
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/elecfacfluclocal.nc'
    #elecfacfluclocalflnm = flnm
    #ld.project_and_save(dfields, xout, vxe, vye, vze, histout, corexout, coreyout, corezout, flnm)


    #if(verbose):print("Running CEi fac sweep for ion")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep) #TODO: verbose statements for all
    #    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'epar', 'x', useBoxFAC=True, altcorfields=dfavg)
    #    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp1', 'y', useBoxFAC=True, altcorfields=dfavg)
    #    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp2', 'z', useBoxFAC=True, altcorfields=dfavg)
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/ionfac.nc'#TODO: rename tofac avg (and follow through script!)
    #ionfacflnm = flnm
    #ld.project_and_save(dfields, xout, vxi, vyi, vzi, histout, corexout, coreyout, corezout, flnm)

    #todo: remove
    if(loaddebugsubset):
        _debugfac = 25.
    else:
        _debugfac = 1.

    if(True):
        if(verbose):print("Running CEi fac local sweep for elecs")
        corexout = []
        coreyout = []
        corezout = []
        histout = []
        xout = []
        x1 = xstartsweep
        x2 = xstartsweep + dxsweep

        for _idx in range(0,nsweep):
            if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep) #TODO: verbose statements for all
            _dfields['Vframe_relative_to_sim'] = 0
            dfields['Vframe_relative_to_sim'] = 0
            vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, _dfavg, 'epar', 'x', useBoxFAC=True, altcorfields=_dfavg, computeinlocalrest = True, beta = beta0, massratio = massrat, c = c)
            vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, _dfavg, 'eperp1', 'y', useBoxFAC=True, altcorfields=_dfavg, computeinlocalrest = True, beta = beta0, massratio = massrat, c = c)
            vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, _dfavg, 'eperp2', 'z', useBoxFAC=True, altcorfields=_dfavg, computeinlocalrest = True, beta = beta0, massratio = massrat, c = c)
            x1 += dxsweep
            x2 += dxsweep

            corex *= _debugfac
            corey *= _debugfac
            corez *= _debugfac
            hist *= _debugfac

            xout.append(((x1+x2)/2.))
            print('cor debug: ', (x1+x2)/2., np.sum(corex)/np.sum(hist), np.sum(corey)/np.sum(hist), np.sum(corez)/np.sum(hist))
            corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
        flnm = 'analysisfiles/ncsweeps/elecfacavglocframe.nc' #TODO: rename tofac avg (and follow through script!)
        elecfacavglocframeflnm = flnm
        ld.project_and_save(dfields, xout, vxe, vye, vze, histout, corexout, coreyout, corezout, flnm)

    if(verbose):print("Running CEi fluc local frame sweep for elecs")
    corexout = []
    coreyout = []
    corezout = []
    histout = []
    xout = []
    x1 = xstartsweep
    x2 = xstartsweep + dxsweep
    for _idx in range(0,nsweep):
        if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep) #TODO: verbose statements for all
        vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, _dfields, 'epar', 'x', useBoxFAC=True, altcorfields=_dfluc, computeinlocalrest = True, beta = beta0, massratio = massrat, c = c)
        vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, _dfields, 'eperp1', 'y', useBoxFAC=True, altcorfields=_dfluc, computeinlocalrest = True, beta = beta0, massratio = massrat, c = c)
        vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, _dfields, 'eperp2', 'z', useBoxFAC=True, altcorfields=_dfluc, computeinlocalrest = True, beta = beta0, massratio = massrat, c = c)
        x1 += dxsweep
        x2 += dxsweep
        
        corex *= _debugfac
        corey *= _debugfac
        corez *= _debugfac
        hist *= _debugfac

        xout.append(((x1+x2)/2.))
        corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    flnm = 'analysisfiles/ncsweeps/elecfacfluclocframe.nc' #TODO: rename tofac avg (and follow through script!)
    elecfacfluclocframeflnm = flnm
    ld.project_and_save(dfields, xout, vxe, vye, vze, histout, corexout, coreyout, corezout, flnm)

    if(True):
        if(verbose):print("Running CEi fac local frame sweep for ion")
        corexout = []
        coreyout = []
        corezout = []
        histout = []
        xout = []
        x1 = xstartsweep
        x2 = xstartsweep + dxsweep
        for _idx in range(0,nsweep):
            if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep) #TODO: verbose statements for all
            vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, _dfields, 'epar', 'x', useBoxFAC=True, altcorfields=_dfavg, computeinlocalrest = True, beta = beta0, massratio = 1, c = c)
            vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, _dfields, 'eperp1', 'y', useBoxFAC=True, altcorfields=_dfavg, computeinlocalrest = True, beta = beta0, massratio = 1, c = c)
            vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, _dfields, 'eperp2', 'z', useBoxFAC=True, altcorfields=_dfavg, computeinlocalrest = True, beta = beta0, massratio = 1, c = c)
            x1 += dxsweep
            x2 += dxsweep
            xout.append(((x1+x2)/2.))
            corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
        flnm = 'analysisfiles/ncsweeps/ionfacavglocframe.nc'#TODO: rename tofac avg (and follow through script!)
        ionfacavglocframeflnm = flnm
        ld.project_and_save(dfields, xout, vxi, vyi, vzi, histout, corexout, coreyout, corezout, flnm)
        
    if(True):
        if(verbose):print("Running CEi fac local frame sweep for ion")
        corexout = []
        coreyout = []
        corezout = []
        histout = []
        xout = []
        x1 = xstartsweep
        x2 = xstartsweep + dxsweep
        for _idx in range(0,nsweep):
            if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep) #TODO: verbose statements for all
            vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, _dfields, 'epar', 'x', useBoxFAC=True, altcorfields=_dfluc, computeinlocalrest = True, beta = beta0, massratio = 1, c = c)
            vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, _dfields, 'eperp1', 'y', useBoxFAC=True, altcorfields=_dfluc, computeinlocalrest = True, beta = beta0, massratio = 1, c = c)
            vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, _dfields, 'eperp2', 'z', useBoxFAC=True, altcorfields=_dfluc, computeinlocalrest = True, beta = beta0, massratio = 1, c = c)
            x1 += dxsweep
            x2 += dxsweep
            xout.append(((x1+x2)/2.))
            corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
        flnm = 'analysisfiles/ncsweeps/ionfacfluclocframe.nc'#TODO: rename tofac avg (and follow through script!)
        ionfacfluclocframeflnm = flnm
        ld.project_and_save(dfields, xout, vxi, vyi, vzi, histout, corexout, coreyout, corezout, flnm)

    #if(verbose):print("Running CEi fac sweep for elecs")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep) #TODO: verbose statements for all
    #    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'epar', 'x', useBoxFAC=True, altcorfields=dfavg)
    #    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp1', 'y', useBoxFAC=True, altcorfields=dfavg)
    #    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp2', 'z', useBoxFAC=True, altcorfields=dfavg)
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/elecfac.nc' #TODO: rename tofac avg (and follow through script!)
    #elecfacflnm = flnm
    #ld.project_and_save(dfields, xout, vxe, vye, vze, histout, corexout, coreyout, corezout, flnm)

    #-----------------------------------------------------------------------------------------------------------------
    #correlate with filtered fields

    #print("Filtering with FFT")
    #filterabove = True
    #kycutoff = 15.
    #dfields_lowpass = aa.yz_fft_filter(dfields,kycutoff,filterabove,dontfilter=False,verbose=True)
    #filterabove = True
    #kycutoff = 15.
    #dfluc_lowpass = aa.yz_fft_filter(dfluc,kycutoff,filterabove,dontfilter=False,verbose=True)
    #filterabove = False
    #kycutoff = 15.
    #dfluc_highpass = aa.yz_fft_filter(dfluc,kycutoff,filterabove,dontfilter=False,verbose=True)
    #print("Done filtering...")

    #if(verbose):print("Running lowpass for ions")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep) #TODO: verbose statements for all
    #    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'epar', 'x',altcorfields=dfluc_lowpass)
    #    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp1', 'y',altcorfields=dfluc_lowpass)
    #    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp2', 'z',altcorfields=dfluc_lowpass) 
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/ionfacfluclowpassflnm.nc' 
    #ionfacfluclowpassflnm = flnm
    #ld.project_and_save(dfields, xout, vxi, vyi, vzi, histout, corexout, coreyout, corezout, flnm)
    
    #if(verbose):print("Running high detrend for ions")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep) #TODO: verbose statements for all
    #    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields_lowpass, 'epar', 'x', useBoxFAC=False, altcorfields=dfluc_highpass)
    #    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields_lowpass, 'eperp1', 'y', useBoxFAC=False, altcorfields=dfluc_highpass)
    #    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields_lowpass, 'eperp2', 'z', useBoxFAC=False, altcorfields=dfluc_highpass) 
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/ionfacfluchighdetrend.nc' 
    #ionfacfluchighdetrendflnm = flnm
    #ld.project_and_save(dfields, xout, vxi, vyi, vzi, histout, corexout, coreyout, corezout, flnm)
    
    #if(verbose):print("Running lowpass for elecs")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep)
    #    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'epar', 'x',altcorfields=dfluc_lowpass)
    #    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp1', 'y',altcorfields=dfluc_lowpass)
    #    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp2', 'z',altcorfields=dfluc_lowpass) 
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/elecfacfluclowpass.nc' 
    #elecfacfluclowpassflnm = flnm
    #ld.project_and_save(dfields, xout, vxe, vye, vze, histout, corexout, coreyout, corezout, flnm)
    
    #if(verbose):print("Running highpass detrend for elecs")
    #corexout = []
    #coreyout = []
    #corezout = []
    #histout = []
    #xout = []
    #x1 = xstartsweep
    #x2 = xstartsweep + dxsweep
    #for _idx in range(0,nsweep):
    #    if(verbose and printprogress):print("Doing slice ", _idx, " of ", nsweep)
    #    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields_lowpass, 'epar', 'x', useBoxFAC=False, altcorfields=dfluc_highpass)
    #    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields_lowpass, 'eperp1', 'y', useBoxFAC=False, altcorfields=dfluc_highpass)
    #    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields_lowpass, 'eperp2', 'z', useBoxFAC=False, altcorfields=dfluc_highpass) 
    #    x1 += dxsweep
    #    x2 += dxsweep
    #    xout.append(((x1+x2)/2.))
    #    corexout.append(corex), coreyout.append(corey), corezout.append(corez), histout.append(hist)
    #flnm = 'analysisfiles/ncsweeps/elecfacfluchighdetrend.nc' 
    #elecfacfluchighdetrendflnm  = flnm
    #ld.project_and_save(dfields, xout, vxe, vye, vze, histout, corexout, coreyout, corezout, flnm)
    

#load 3v hist
filein = open(pathfpcdata+ion3vhistflnm, 'rb')
ionhistsweep = pickle.load(filein)
filein.close()
filein = open(pathfpcdata+elec3vhistflnm, 'rb')
elechistsweep = pickle.load(filein)
filein.close()

#load dflow
filein = open(pathfpcdata+dflowflnm, 'rb')
dflow = pickle.load(filein)
filein.close()

#load ion fluc
(Hist_vxvyion, Hist_vxvzion, Hist_vyvzion,
C_Ex_vxvyflucion, C_Ex_vxvzflucion, C_Ex_vyvzflucion,
C_Ey_vxvyflucion, C_Ey_vxvzflucion, C_Ey_vyvzflucion,
C_Ez_vxvyflucion, C_Ez_vxvzflucion, C_Ez_vyvzflucion,
vxion, vyion, vzion, x_in,
_, _, _, #TODO: remove unused inputs
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+ionflucflnm)
dvion = np.abs(vxion[1,1,1]-vxion[0,0,0])
vmaxion = np.max(vxion)

#load ion tot
(Hist_vxvyion, Hist_vxvzion, Hist_vyvzion,
C_Ex_vxvytotion, C_Ex_vxvztotion, C_Ex_vyvztotion,
C_Ey_vxvytotion, C_Ey_vxvztotion, C_Ey_vyvztotion,
C_Ez_vxvytotion, C_Ez_vxvztotion, C_Ez_vyvztotion,
vxion, vyion, vzion, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+iontotflnm)

#load ion fac fluc
(_, _, _,
C_Ex_vxvyfacflucion, C_Ex_vxvzfacflucion, C_Ex_vyvzfacflucion,
C_Ey_vxvyfacflucion, C_Ey_vxvzfacflucion, C_Ey_vyvzfacflucion,
C_Ez_vxvyfacflucion, C_Ez_vxvzfacflucion, C_Ez_vyvzfacflucion,
vxion, vyion, vzion, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+ionfacflucflnm)

#load ion fac fluc local
(_, _, _,
C_Ex_vxvyfacfluclocalion, C_Ex_vxvzfacfluclocalion, C_Ex_vyvzfacfluclocalion,
C_Ey_vxvyfacfluclocalion, C_Ey_vxvzfacfluclocalion, C_Ey_vyvzfacfluclocalion,
C_Ez_vxvyfacfluclocalion, C_Ez_vxvzfacfluclocalion, C_Ez_vyvzfacfluclocalion,
vxion, vyion, vzion, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+ionfacfluclocalflnm)

#load ion fac fluc local
(_, _, _,
C_Ex_vxvyfacfluclowpassion, C_Ex_vxvzfacfluclowpassion, C_Ex_vyvzfacfluclowpassion,
C_Ey_vxvyfacfluclowpassion, C_Ey_vxvzfacfluclowpassion, C_Ey_vyvzfacfluclowpassion,
C_Ez_vxvyfacfluclowpassion, C_Ez_vxvzfacfluclowpassion, C_Ez_vyvzfacfluclowpassion,
vxion, vyion, vzion, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+ionfacfluclowpassflnm)

(_, _, _,
C_Ex_vxvyfacfluchighdetrend, C_Ex_vxvzfacfluchighdetrend, C_Ex_vyvzfacfluchighdetrend,
C_Ey_vxvyfacfluchighdetrend, C_Ey_vxvzfacfluchighdetrend, C_Ey_vyvzfacfluchighdetrend,
C_Ez_vxvyfacfluchighdetrend, C_Ez_vxvzfacfluchighdetrend, C_Ez_vyvzfacfluchighdetrend,
vxion, vyion, vzion, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+ionfacfluchighdetrendflnm)

#load ion fac
(_, _, _,
C_Ex_vxvyfacion, C_Ex_vxvzfacion, C_Ex_vyvzfacion,
C_Ey_vxvyfacion, C_Ey_vxvzfacion, C_Ey_vyvzfacion,
C_Ez_vxvyfacion, C_Ez_vxvzfacion, C_Ez_vyvzfacion,
vxion, vyion, vzion, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+ionfacflnm)

#load elec fluc
(Hist_vxvyelec, Hist_vxvzelec, Hist_vyvzelec,
C_Ex_vxvyflucelec, C_Ex_vxvzflucelec, C_Ex_vyvzflucelec,
C_Ey_vxvyflucelec, C_Ey_vxvzflucelec, C_Ey_vyvzflucelec,
C_Ez_vxvyflucelec, C_Ez_vxvzflucelec, C_Ez_vyvzflucelec,
vxelec, vyelec, vzelec, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+elecflucflnm)
dvelec = np.abs(vxelec[1,1,1]-vxelec[0,0,0])
vmaxelec = np.max(vxelec)

#load elec tot
(Hist_vxvyelec, Hist_vxvzelec, Hist_vyvzelec,
C_Ex_vxvytotelec, C_Ex_vxvztotelec, C_Ex_vyvztotelec,
C_Ey_vxvytotelec, C_Ey_vxvztotelec, C_Ey_vyvztotelec,
C_Ez_vxvytotelec, C_Ez_vxvztotelec, C_Ez_vyvztotelec,
vxelec, vyelec, vzelec, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+electotflnm)

#load elec fac
(_, _, _,
C_Ex_vxvyfacelec, C_Ex_vxvzfacelec, C_Ex_vyvzfacelec,
C_Ey_vxvyfacelec, C_Ey_vxvzfacelec, C_Ey_vyvzfacelec,
C_Ez_vxvyfacelec, C_Ez_vxvzfacelec, C_Ez_vyvzfacelec,
vxelec, vyelec, vzelec, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+elecfacflnm)

#load elec fac fluc
(_, _, _,
C_Ex_vxvyfacflucelec, C_Ex_vxvzfacflucelec, C_Ex_vyvzfacflucelec,
C_Ey_vxvyfacflucelec, C_Ey_vxvzfacflucelec, C_Ey_vyvzfacflucelec,
C_Ez_vxvyfacflucelec, C_Ez_vxvzfacflucelec, C_Ez_vyvzfacflucelec,
vxelec, vyelec, vzelec, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+elecfacflucflnm)

#load elec fac fluc local
(_, _, _,
C_Ex_vxvyfacfluclocalelec, C_Ex_vxvzfacfluclocalelec, C_Ex_vyvzfacfluclocalelec,
C_Ey_vxvyfacfluclocalelec, C_Ey_vxvzfacfluclocalelec, C_Ey_vyvzfacfluclocalelec,
C_Ez_vxvyfacfluclocalelec, C_Ez_vxvzfacfluclocalelec, C_Ez_vyvzfacfluclocalelec,
vxelec, vyelec, vzelec, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+elecfacfluclocalflnm)

(_, _, _,
C_Ex_vxvyfacfluclowpasselec, C_Ex_vxvzfacfluclowpasselec, C_Ex_vyvzfacfluclowpasselec,
C_Ey_vxvyfacfluclowpasselec, C_Ey_vxvzfacfluclowpasselec, C_Ey_vyvzfacfluclowpasselec,
C_Ez_vxvyfacfluclowpasselec, C_Ez_vxvzfacfluclowpasselec, C_Ez_vyvzfacfluclowpasselec,
vxelec, vyelec, vzelec, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+elecfacfluclowpassflnm)

(_, _, _,
C_Ex_vxvyfacfluchighdetrendelec, C_Ex_vxvzfacfluchighdetrendelec, C_Ex_vyvzfacfluchighdetrendelec,
C_Ey_vxvyfacfluchighdetrendelec, C_Ey_vxvzfacfluchighdetrendelec, C_Ey_vyvzfacfluchighdetrendelec,
C_Ez_vxvyfacfluchighdetrendelec, C_Ez_vxvzfacfluchighdetrendelec, C_Ez_vyvzfacfluchighdetrendelec,
vxelec, vyelec, vzelec, x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+elecfacfluchighdetrendflnm)

#load ion fac
(_, _, _,
C_Ex_vxvyfacavglocframeion, C_Ex_vxvzfacavglocframeion, C_Ex_vyvzfacavglocframeion,
C_Ey_vxvyfacavglocframeion, C_Ey_vxvzfacavglocframeion, C_Ey_vyvzfacavglocframeion,
C_Ez_vxvyfacavglocframeion, C_Ez_vxvzfacavglocframeion, C_Ez_vyvzfacavglocframeion,
_vxion, _vyion, _vzion, _x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+ionfacavglocframeflnm)

(_, _, _,
C_Ex_vxvyfacfluclocframeion, C_Ex_vxvzfacfluclocframeion, C_Ex_vyvzfacfluclocframeion,
C_Ey_vxvyfacfluclocframeion, C_Ey_vxvzfacfluclocframeion, C_Ey_vyvzfacfluclocframeion,
C_Ez_vxvyfacfluclocframeion, C_Ez_vxvzfacfluclocframeion, C_Ez_vyvzfacfluclocframeion,
_vxion, _vyion, _vzion, _x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+ionfacfluclocframeflnm)

(_, _, _,
C_Ex_vxvyfacavglocframeelec, C_Ex_vxvzfacavglocframeelec, C_Ex_vyvzfacavglocframeelec,
C_Ey_vxvyfacavglocframeelec, C_Ey_vxvzfacavglocframeelec, C_Ey_vyvzfacavglocframeelec,
C_Ez_vxvyfacavglocframeelec, C_Ez_vxvzfacavglocframeelec, C_Ez_vyvzfacavglocframeelec,
_vxelec, _vyelec, _vzelec, _x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+elecfacavglocframeflnm)

(_, _, _,
C_Ex_vxvyfacfluclocframeelec, C_Ex_vxvzfacfluclocframeelec, C_Ex_vyvzfacfluclocframeelec,
C_Ey_vxvyfacfluclocframeelec, C_Ey_vxvzfacfluclocframeelec, C_Ey_vyvzfacfluclocframeelec,
C_Ez_vxvyfacfluclocframeelec, C_Ez_vxvzfacfluclocframeelec, C_Ez_vyvzfacfluclocframeelec,
_vxelec, _vyelec, _vzelec, _x_in,
_, _, _,
_, Vframe_relative_to_sim_in, _, _) = ld.load2vdata(pathfpcdata+elecfacfluclocframeflnm)

#recompute energy for tilde CEi bar CEi CEi total with selected normalization --------------------------------------------------------------------------------------
#compute for ions

print("Done loading!: computing energization rates!")
#note: these are energization rates (integral of CEi has dimensions of power)
enerCEx_ion_tilde = np.asarray([np.sum(C_Ex_vxvyflucion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyflucion))]) 
enerCEy_ion_tilde = np.asarray([np.sum(C_Ey_vxvyflucion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyflucion))]) 
enerCEz_ion_tilde = np.asarray([np.sum(C_Ez_vxvyflucion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyflucion))]) 
enerCEtot_ion_tilde = np.asarray([enerCEx_ion_tilde[_i] + enerCEy_ion_tilde[_i] + enerCEz_ion_tilde[_i] for _i in range(len(enerCEx_ion_tilde))])

enerCEx_ion = np.asarray([np.sum(C_Ex_vxvytotion[_i])*dvion**3 for _i in range(len(C_Ex_vxvytotion))])
enerCEy_ion = np.asarray([np.sum(C_Ey_vxvytotion[_i])*dvion**3 for _i in range(len(C_Ex_vxvytotion))]) 
enerCEz_ion = np.asarray([np.sum(C_Ez_vxvytotion[_i])*dvion**3 for _i in range(len(C_Ex_vxvytotion))]) 
enerCEtot_ion = np.asarray([enerCEx_ion[_i] + enerCEy_ion[_i] + enerCEz_ion[_i] for _i in range(len(enerCEz_ion))])

enerCEtot_ion_bar = np.asarray([enerCEtot_ion[_id] - enerCEtot_ion_tilde[_id] for _id in range(len(enerCEtot_ion))])
enerCEx_ion_bar = np.asarray([enerCEx_ion[_id] - enerCEx_ion_tilde[_id] for _id in range(len(enerCEx_ion))])
enerCEy_ion_bar = np.asarray([enerCEy_ion[_id] - enerCEy_ion_tilde[_id] for _id in range(len(enerCEy_ion))])
enerCEz_ion_bar = np.asarray([enerCEz_ion[_id] - enerCEz_ion_tilde[_id] for _id in range(len(enerCEz_ion))])

enerCEx_ion_facfluc = np.asarray([np.sum(C_Ex_vxvyfacflucion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacflucion))])
enerCEy_ion_facfluc = np.asarray([np.sum(C_Ey_vxvyfacflucion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacflucion))])
enerCEz_ion_facfluc = np.asarray([np.sum(C_Ez_vxvyfacflucion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacflucion))])
enerCEtot_ion_facfluc = np.asarray([enerCEx_ion_facfluc[_i] + enerCEy_ion_facfluc[_i] + enerCEz_ion_facfluc[_i] for _i in range(len(enerCEz_ion_facfluc))])

enerCEx_ion_fac = np.asarray([np.sum(C_Ex_vxvyfacion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacion))])
enerCEy_ion_fac = np.asarray([np.sum(C_Ey_vxvyfacion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacion))])
enerCEz_ion_fac = np.asarray([np.sum(C_Ez_vxvyfacion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacion))])
enerCEtot_ion_fac = np.asarray([enerCEx_ion_fac[_i] + enerCEy_ion_fac[_i] + enerCEz_ion_fac[_i] for _i in range(len(enerCEz_ion_fac))])

enerCEx_ion_facfluclocal = np.asarray([np.sum(C_Ex_vxvyfacfluclocalion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacfluclocalion))])
enerCEy_ion_facfluclocal = np.asarray([np.sum(C_Ey_vxvyfacfluclocalion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacfluclocalion))])
enerCEz_ion_facfluclocal = np.asarray([np.sum(C_Ez_vxvyfacfluclocalion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacfluclocalion))])
enerCEtot_ion_facfluclocal = np.asarray([enerCEx_ion_facfluclocal[_i] + enerCEy_ion_facfluclocal[_i] + enerCEz_ion_facfluclocal[_i] for _i in range(len(enerCEz_ion_facfluclocal))])

enerCEx_ion_facfluclowpass = np.asarray([np.sum(C_Ex_vxvyfacfluclowpassion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacfluclowpassion))])
enerCEy_ion_facfluclowpass = np.asarray([np.sum(C_Ey_vxvyfacfluclowpassion[_i])*dvion**3 for _i in range(len(C_Ey_vxvyfacfluclowpassion))])
enerCEz_ion_facfluclowpass = np.asarray([np.sum(C_Ez_vxvyfacfluclowpassion[_i])*dvion**3 for _i in range(len(C_Ez_vxvyfacfluclowpassion))])
enerCEtot_ion_facfluclowpass = np.asarray([enerCEx_ion_facfluclowpass[_i] + enerCEy_ion_facfluclowpass[_i] + enerCEz_ion_facfluclowpass[_i] for _i in range(len(enerCEz_ion_facfluclowpass))])

enerCEx_ion_facfluchighdetrend = np.asarray([np.sum(C_Ex_vxvyfacfluchighdetrend[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacfluchighdetrend))])
enerCEy_ion_facfluchighdetrend = np.asarray([np.sum(C_Ey_vxvyfacfluchighdetrend[_i])*dvion**3 for _i in range(len(C_Ey_vxvyfacfluchighdetrend))])
enerCEz_ion_facfluchighdetrend = np.asarray([np.sum(C_Ez_vxvyfacfluchighdetrend[_i])*dvion**3 for _i in range(len(C_Ez_vxvyfacfluchighdetrend))])
enerCEtot_ion_facfluchighdetrend = np.asarray([enerCEx_ion_facfluchighdetrend[_i] + enerCEy_ion_facfluchighdetrend[_i] + enerCEz_ion_facfluchighdetrend[_i] for _i in range(len(enerCEz_ion_facfluchighdetrend))])

#compute for elecs
enerCEx_elec_tilde = np.asarray([np.sum(C_Ex_vxvyflucelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyflucelec))]) 
enerCEy_elec_tilde = np.asarray([np.sum(C_Ey_vxvyflucelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyflucelec))]) 
enerCEz_elec_tilde = np.asarray([np.sum(C_Ez_vxvyflucelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyflucelec))]) 
enerCEtot_elec_tilde = np.asarray([enerCEx_elec_tilde[_i] + enerCEy_elec_tilde[_i] + enerCEz_elec_tilde[_i] for _i in range(len(enerCEx_elec_tilde))])

enerCEx_elec = np.asarray([np.sum(C_Ex_vxvytotelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvytotelec))])
enerCEy_elec = np.asarray([np.sum(C_Ey_vxvytotelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvytotelec))])
enerCEz_elec = np.asarray([np.sum(C_Ez_vxvytotelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvytotelec))])
enerCEtot_elec = np.asarray([enerCEx_elec[_i] + enerCEy_elec[_i] + enerCEz_elec[_i] for _i in range(len(enerCEz_elec))])

enerCEtot_elec_bar = np.asarray([enerCEtot_elec[_id] - enerCEtot_elec_tilde[_id] for _id in range(len(enerCEtot_elec))])
enerCEx_elec_bar = np.asarray([enerCEx_elec[_id] - enerCEx_elec_tilde[_id] for _id in range(len(enerCEx_elec))])
enerCEy_elec_bar = np.asarray([enerCEy_elec[_id] - enerCEy_elec_tilde[_id] for _id in range(len(enerCEy_elec))])
enerCEz_elec_bar = np.asarray([enerCEz_elec[_id] - enerCEz_elec_tilde[_id] for _id in range(len(enerCEz_elec))])

enerCEx_elec_fac = np.asarray([np.sum(C_Ex_vxvyfacelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacelec))])
enerCEy_elec_fac = np.asarray([np.sum(C_Ey_vxvyfacelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacelec))])
enerCEz_elec_fac = np.asarray([np.sum(C_Ez_vxvyfacelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacelec))])
enerCEtot_elec_fac = np.asarray([enerCEx_elec_fac[_i] + enerCEy_elec_fac[_i] + enerCEz_elec_fac[_i] for _i in range(len(enerCEz_elec_fac))])

enerCEx_elec_facfluc = np.asarray([np.sum(C_Ex_vxvyfacflucelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacflucelec))])
enerCEy_elec_facfluc = np.asarray([np.sum(C_Ey_vxvyfacflucelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacflucelec))])
enerCEz_elec_facfluc = np.asarray([np.sum(C_Ez_vxvyfacflucelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacflucelec))])
enerCEtot_elec_facfluc = np.asarray([enerCEx_elec_facfluc[_i] + enerCEy_elec_facfluc[_i] + enerCEz_elec_facfluc[_i] for _i in range(len(enerCEz_elec_facfluc))])

enerCEx_elec_facfluclocal = np.asarray([np.sum(C_Ex_vxvyfacfluclocalelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacfluclocalelec))])
enerCEy_elec_facfluclocal = np.asarray([np.sum(C_Ey_vxvyfacfluclocalelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacfluclocalelec))])
enerCEz_elec_facfluclocal = np.asarray([np.sum(C_Ez_vxvyfacfluclocalelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacfluclocalelec))])
enerCEtot_elec_facfluclocal = np.asarray([enerCEx_elec_facfluclocal[_i] + enerCEy_elec_facfluclocal[_i] + enerCEz_elec_facfluclocal[_i] for _i in range(len(enerCEz_elec_facfluclocal))])

enerCEx_elec_facfluclowpass = np.asarray([np.sum(C_Ex_vxvyfacfluclowpasselec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacfluclowpasselec))])
enerCEy_elec_facfluclowpass = np.asarray([np.sum(C_Ey_vxvyfacfluclowpasselec[_i])*dvelec**3 for _i in range(len(C_Ey_vxvyfacfluclowpasselec))])
enerCEz_elec_facfluclowpass = np.asarray([np.sum(C_Ez_vxvyfacfluclowpasselec[_i])*dvelec**3 for _i in range(len(C_Ez_vxvyfacfluclowpasselec))])
enerCEtot_elec_facfluclowpass = np.asarray([enerCEx_elec_facfluclowpass[_i] + enerCEy_elec_facfluclowpass[_i] + enerCEz_elec_facfluclowpass[_i] for _i in range(len(enerCEz_elec_facfluclowpass))])

enerCEx_elec_facfluchighdetrend = np.asarray([np.sum(C_Ex_vxvyfacfluchighdetrendelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacfluchighdetrendelec))])
enerCEy_elec_facfluchighdetrend = np.asarray([np.sum(C_Ey_vxvyfacfluchighdetrendelec[_i])*dvelec**3 for _i in range(len(C_Ey_vxvyfacfluchighdetrendelec))])
enerCEz_elec_facfluchighdetrend = np.asarray([np.sum(C_Ez_vxvyfacfluchighdetrendelec[_i])*dvelec**3 for _i in range(len(C_Ez_vxvyfacfluchighdetrendelec))])
enerCEtot_elec_facfluchighdetrend = np.asarray([enerCEx_elec_facfluchighdetrend[_i] + enerCEy_elec_facfluchighdetrend[_i] + enerCEz_elec_facfluchighdetrend[_i] for _i in range(len(enerCEz_elec_facfluchighdetrend))])


enerCEx_ion_facavglocframe = np.asarray([np.sum(C_Ex_vxvyfacavglocframeion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacavglocframeion))])
enerCEy_ion_facavglocframe = np.asarray([np.sum(C_Ey_vxvyfacavglocframeion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacavglocframeion))])
enerCEz_ion_facavglocframe = np.asarray([np.sum(C_Ez_vxvyfacavglocframeion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacavglocframeion))])
enerCEtot_ion_facavglocframe = np.asarray([enerCEx_ion_facavglocframe[_i] + enerCEy_ion_facavglocframe[_i] + enerCEz_ion_facavglocframe[_i] for _i in range(len(enerCEz_ion_facavglocframe))])

enerCEx_ion_facfluclocframe = np.asarray([np.sum(C_Ex_vxvyfacfluclocframeion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacfluclocframeion))])
enerCEy_ion_facfluclocframe = np.asarray([np.sum(C_Ey_vxvyfacfluclocframeion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacfluclocframeion))])
enerCEz_ion_facfluclocframe = np.asarray([np.sum(C_Ez_vxvyfacfluclocframeion[_i])*dvion**3 for _i in range(len(C_Ex_vxvyfacfluclocframeion))])
enerCEtot_ion_facfluclocframe = np.asarray([enerCEx_ion_facfluclocframe[_i] + enerCEy_ion_facfluclocframe[_i] + enerCEz_ion_facfluclocframe[_i] for _i in range(len(enerCEz_ion_facfluclocframe))])

enerCEx_elec_facavglocframe = np.asarray([np.sum(C_Ex_vxvyfacavglocframeelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacavglocframeelec))])
enerCEy_elec_facavglocframe = np.asarray([np.sum(C_Ey_vxvyfacavglocframeelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacavglocframeelec))])
enerCEz_elec_facavglocframe = np.asarray([np.sum(C_Ez_vxvyfacavglocframeelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacavglocframeelec))])
enerCEtot_elec_facavglocframe = np.asarray([enerCEx_elec_facavglocframe[_i] + enerCEy_elec_facavglocframe[_i] + enerCEz_elec_facavglocframe[_i] for _i in range(len(enerCEz_elec_facavglocframe))])

enerCEx_elec_facfluclocframe = np.asarray([np.sum(C_Ex_vxvyfacfluclocframeelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacfluclocframeelec))])
enerCEy_elec_facfluclocframe = np.asarray([np.sum(C_Ey_vxvyfacfluclocframeelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacfluclocframeelec))])
enerCEz_elec_facfluclocframe = np.asarray([np.sum(C_Ez_vxvyfacfluclocframeelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyfacfluclocframeelec))])
enerCEtot_elec_facfluclocframe = np.asarray([enerCEx_elec_facfluclocframe[_i] + enerCEy_elec_facfluclocframe[_i] + enerCEz_elec_facfluclocframe[_i] for _i in range(len(enerCEz_elec_facfluclocframe))])



#compute jxshockEx in shock rest frame see juno 2021 fig 11d
#vshockval = 3.4 # v/va (goal: get in v/vth,e)
#massratio = 625
#vshockval = 3.4/np.sqrt(beta0)/np.sqrt(massratio)
#_Exvals = dfavg['ex'][0,0,:]

#ddenavgelec = dden['dens'].mean(axis=(0,1))/2.

#_qval = -1.*ddenavgelec #each charge has q=-1, so its jsut -1*ddenavg
#enerjxshockEx = _qval*vshockval*_Exvals

massrat = 625
massratio = 625
#_vths = 1. #1 bc our particles are normalized to this value
#_vthi = np.sqrt(massratio)*_vths
#_vatot = (1./np.sqrt(beta0/2))*_vthi
#vxdown = vxdown/_vatot

dflowavg = aa.get_average_flow_over_yz(dflow)

#v/va (va/vthi) (vthi/vthe)
dfavgvxboost = (vshock/np.sqrt(beta0/2.))/np.sqrt(massratio)

#compute shock rest frame j dot E
dfavgvxrest = dflowavg['ue'][0,0,:]+dfavgvxboost
dfavgvyrest = dflowavg['ve'][0,0,:]
dfavgvzrest = dflowavg['we'][0,0,:]

ddenavgelec = dden['dens'].mean(axis=(0,1))/2.
ddenavgelecup = ddenavgelec[np.nonzero(ddenavgelec)[0][-10]]
_qval = -1.*ddenavgelec #each charge has q=-1, so its jsut -1*ddenavg
dfavgrest = ft.lorentz_transform_vx(dfields,-vshock,c) 
jdotErest = _qval*(dfavgvxrest*dfavgrest['ex'][0,0,:]+dfavgvyrest*dfavgrest['ey'][0,0,:]+dfavgvzrest*dfavgrest['ez'][0,0,:])/ddenavgelecup

enerjxshockEx = jdotErest-_qval*dfavgvxboost*dfavgrest['ex'][0,0,:]/ddenavgelecup

import copy

dflowavgrest = copy.deepcopy(dflowavg)
dflowavgrest['ue'] = dflowavgrest['ue'] + dfavgvxrest

#put into list for ease of operating on all
enerCEis = [enerCEx_ion_tilde,enerCEy_ion_tilde,enerCEz_ion_tilde,enerCEtot_ion_tilde,
            enerCEx_ion,enerCEy_ion,enerCEz_ion,enerCEtot_ion,
            enerCEtot_ion_bar,enerCEx_ion_bar,enerCEy_ion_bar,enerCEz_ion_bar,
            enerCEx_elec_tilde,enerCEy_elec_tilde,enerCEz_elec_tilde,enerCEtot_elec_tilde,
            enerCEx_elec,enerCEy_elec,enerCEz_elec,enerCEtot_elec,
            enerCEtot_elec_bar,enerCEx_elec_bar,enerCEy_elec_bar,enerCEz_elec_bar,
            enerCEx_ion_facfluc,enerCEy_ion_facfluc,enerCEz_ion_facfluc,enerCEtot_ion_facfluc,
            enerCEx_ion_facfluclocal,enerCEy_ion_facfluclocal,enerCEz_ion_facfluclocal,enerCEtot_ion_facfluclocal,
            enerCEx_elec_facfluc,enerCEy_elec_facfluc,enerCEz_elec_facfluc,enerCEtot_elec_facfluc,
            enerCEx_elec_facfluclocal,enerCEy_elec_facfluclocal,enerCEz_elec_facfluclocal,enerCEtot_elec_facfluclocal,
            enerCEx_ion_facfluclowpass,enerCEy_ion_facfluclowpass,enerCEz_ion_facfluclowpass,enerCEtot_ion_facfluclowpass,
            enerCEx_ion_fac,enerCEy_ion_fac,enerCEz_ion_fac,enerCEtot_ion_fac,
            enerCEx_elec_facfluclowpass,enerCEy_elec_facfluclowpass,enerCEz_elec_facfluclowpass,enerCEtot_elec_facfluclowpass,
            enerCEx_elec_fac,enerCEy_elec_fac,enerCEz_elec_fac,enerCEtot_elec_fac,
            enerCEx_ion_facavglocframe,enerCEy_ion_facavglocframe,enerCEz_ion_facavglocframe,enerCEtot_ion_facavglocframe,            
            enerCEx_ion_facfluclocframe,enerCEy_ion_facfluclocframe,enerCEz_ion_facfluclocframe,enerCEtot_ion_facfluclocframe,
            enerCEx_elec_facavglocframe,enerCEy_elec_facavglocframe,enerCEz_elec_facavglocframe,enerCEtot_elec_facavglocframe,
            enerCEx_elec_facfluclocframe,enerCEy_elec_facfluclocframe,enerCEz_elec_facfluclocframe,enerCEtot_elec_facfluclocframe
            ]

for _i in range(0,len(enerCEis)):
    enerCEis[_i] = np.asarray(enerCEis[_i])

#Normalize CEi (normalize to den, and take out factor of number of particles created when computing FPC...)
ntotsion = []
for temphist in Hist_vxvyion:
    ntotsion.append(np.sum(temphist))
ntotsion = np.asarray(ntotsion)

_lastnonzeroidx = np.nonzero(ntotsion)[0][-1]
num_i_0 = ntotsion[_lastnonzeroidx]
print('num_i_0',num_i_0)

print('ntotsion',ntotsion)
ntotdenupion = ntotsion[np.where(ntotsion)[0][-5]] #the -5 is to swim upstrea from the right most nonzero, as the first nonzero may not be the upstream value as particles might swim upstrea from wheret they are injected (just a little due to gyroorbit)
print(ntotdenupion)
for _i in list(range(0, 12)) + list(range(24, 32)) + list(range(40,48)) + list(range(56,64)):
    for _j in range(0,len(enerCEis[_i])):
        if(ntotsion[_j] != 0.):
            enerCEis[_i][_j] = enerCEis[_i][_j]/ntotdenupion#TODO: use used to do this: but its wrong so we should rewrite this for efficiency enerCEis[_i][_j]*ntotsion[_j]/ntotdenupion
        else:
            enerCEis[_i][_j] = 0.

ntotselec = []
for temphist in Hist_vxvyelec:
    ntotselec.append(np.sum(temphist))
ntotselec = np.asarray(ntotselec)
ntotdenupelec = ntotselec[np.nonzero(ntotselec)[0][-5]] #the -5 is to swim upstrea from the right most nonzero, as the first nonzero may not be the upstream value as particles might swim upstrea from wheret they are injected (just a little due to gyroorbit)
for _i in list(range(12,24)) + list(range(32,40)) + list(range(48,56)) + list(range(64,72)):
    for _j in range(0,len(enerCEis[_i])):
        if(ntotselec[_j] != 0.):
            enerCEis[_i][_j] = enerCEis[_i][_j]/ntotdenupelec
        else:
            enerCEis[_i][_j] = 0.

#compute average den in each slice of y,z
ddenavg = dden['dens'].mean(axis=(0,1))

#Temp way to compute Te
num_den_elec0 = 1. #dden converges to 1 upstream
gamma = 5./3.
Te0 = 1.
elecTemp = []
for _elecTidx in range(0,len(dden['dens_xx'])):
    num_den_elec = ddenavg[_elecTidx] 
    eTval = Te0*(num_den_elec/num_den_elec0)**(gamma-1)
    elecTemp.append(eTval)
elecTemp = np.asarray(elecTemp)

#Temp way to compute KE ions
print("Warning, dden may need to be divided by two as ni=ne pretty much everywhere!")
dflowavg = aa.get_average_flow_over_yz(dflow)
dx = (dflowavg["ui_xx"][1]-dflowavg["ui_xx"][0])/2
KEions = []
xxKEplot = []
m_i = params['mi']
vti0 = np.sqrt(params['delgam']) #TODO: move to clean up code and gather all other constants that we use in this script
vte0 = np.sqrt(params['mi']/params['me'])*vti0
for _xposidx in range(0,len(dflowavg["ui_xx"])):
    utotsqrd = (dflowavg['ui'][0,0,_xposidx]**2.+dflowavg['vi'][0,0,_xposidx]**2.+dflowavg['wi'][0,0,_xposidx]**2.0) #TODO: check this normalization
    KEions.append(0.5*m_i*ddenavg[_xposidx]*utotsqrd) #1/2 n_i m_i U_i^2.
    xxKEplot.append(dden['dens_xx'][_xposidx])
KEions = np.asarray(KEions)
print(KEions)
xxKEplot = np.asarray(xxKEplot)

#TODO: compute Te adiabatic like Aaron Tran does (Te_ad = T0[1+2(n/n0)^{Gamma-1}]/3)
ddenavg = dden['dens'].mean(axis=(0,1))
num_den_elec0 = 1.
Gamma = 2.
T0 = 1.
Te_ad_tran = np.zeros((len(ddenavg)))
Te_ad_tran = T0*(1.+2*(ddenavg[:]/num_den_elec0)**(Gamma-1))/3.

Gamma = 5./3.
T_ad_gamma_fivethirds = np.zeros((len(ddenavg)))
T_ad_gamma_fivethirds = T0*(1.+2*(ddenavg[:]/num_den_elec0)**(Gamma-1))/3.

#compute KE elecs
KEelecs = []
xxKEplot = []
m_e = params['me']
for _xposidx in range(0,len(dflowavg["ui_xx"])):
    utotsqrd = dflowavg['ue'][0,0,_xposidx]**2.+dflowavg['ve'][0,0,_xposidx]**2.+dflowavg['we'][0,0,_xposidx]**2.
    KEelecs.append(0.5*m_e*ddenavg[_xposidx]*utotsqrd) #1/2 n_e m_e U_e^2.
    xxKEplot.append(dden['dens_xx'][_xposidx])
KEelecs = np.asarray(KEelecs)
xxKEplot = np.asarray(xxKEplot)

#compute bulk flow (e cross b energy)
#get first nonzero slice of particle count
_tempidx = -1
ionupstreamnum = np.sum(ionhistsweep[_tempidx])
while(ionupstreamnum == 0):
    _tempidx = _tempidx - 1
    ionupstreamnum = np.sum(ionhistsweep[_tempidx])
_tempidx = -1
elecupstreamnum = np.sum(elechistsweep[_tempidx])
while(elecupstreamnum == 0):
    _tempidx = _tempidx - 1
    elecupstreamnum = np.sum(elechistsweep[_tempidx])
bulkfloweion = np.zeros((len(ionhistsweep)))
bulkfloweelec = np.zeros((len(elechistsweep)))
sigma_ion = params['sigma']*params['me']/params['mi']
c_therm = c/np.sqrt(beta0) #speed of light, normalized to vth (assuming c was originally normalized to alfven velocity) TODO: remove
dxin = x_in[1]-x_in[0]
vti0 = np.sqrt(params['delgam']) #TODO: move to clean up code and gather all other constants that we use in this script
vte0 = np.sqrt(params['mi']/params['me'])*vti0
for _xposidx in range(0,len(ionhistsweep)):
    _x1 = x_in[_xposidx]-dxin/2.
    _x2 = x_in[_xposidx]+dxin/2.
    _ez = ao.get_average_in_box(_x1, _x2, -99999999, 99999999, -99999999, 99999999, dfavg, 'ez')
    _by = ao.get_average_in_box(_x1, _x2, -99999999, 99999999, -99999999, 99999999, dfavg, 'by')
    ubulk = (-params['c'] * _ez/_by)/(vti0/np.sqrt(sigma_ion))  #TODO: CHECK UNITS HERE!!!
 
    n_i = np.sum(ionhistsweep[_xposidx])/ionupstreamnum
    m_i = params['mi']/params['mi'] #TODO:FIX MI EVERYWHERE LIKE THIS
    n_e = np.sum(elechistsweep[_xposidx])/elecupstreamnum
    m_e = params['me']/params['mi']

    print("Debug: ni ne, ni/ni0 ne/ne0", n_i*ionupstreamnum, n_e*elecupstreamnum, n_i, n_e) 

    bulkfloweion[_xposidx] = 0.5*n_i*m_i*ubulk**2
    bulkfloweelec[_xposidx] = 0.5*n_e*m_e*ubulk**2 #note: both bulkflowion/elec are normalized to v

print('mi',params['mi'])
print('me',params['me'])

#compute internal energy
print("TODO: compute internal energy in correct frame")
ioninternale = np.zeros((len(ionhistsweep)))
elecinternale = np.zeros((len(elechistsweep)))
vxion = np.asarray(vxion)
vyion = np.asarray(vyion)
vzion = np.asarray(vzion)
vxelec = np.asarray(vxelec)
vyelec = np.asarray(vyelec)
vzelec = np.asarray(vzelec)
dvion = vxion[1,1,1]-vxion[0,0,0]
dvelec = vxelec[1,1,1]-vxelec[0,0,0]
for _xposidx in range(0,len(ionhistsweep)):
    ioninternale[_xposidx] = 0.5*m_i*np.sum((vxion**2+vyion**2+vzion**2)*ionhistsweep[_xposidx])*(params['c'])**2*dvion**3/ionupstreamnum#-bulkfloweion[_xposidx]
    elecinternale[_xposidx] = 0.5*m_e*np.sum((vxelec**2+vyelec**2+vzelec**2)*elechistsweep[_xposidx])*(params['c'])**2*dvelec**3/elecupstreamnum#-bulkfloweelec[_xposidx] #TODO: CHECK UNITS HERE; IN PARTIUCLAR THE c**2 factor

#compute temperature using 2nd moment
ionmomtemp = np.zeros((len(ionhistsweep)))
elecmomtemp = np.zeros((len(ionhistsweep)))
vxion = np.asarray(vxion)
vyion = np.asarray(vyion)
vzion = np.asarray(vzion)
vxelec = np.asarray(vxelec)
vyelec = np.asarray(vyelec)
vzelec = np.asarray(vzelec)
dvion = vxion[1,1,1]-vxion[0,0,0]
dvelec = vxelec[1,1,1]-vxelec[0,0,0]
for _xposidx in range(0,len(ionhistsweep)):
    if(np.sum(ionhistsweep[_xposidx] > 0.)):
        uxionmean = np.average(vxion, weights=ionhistsweep[_xposidx])
        uyionmean = np.average(vyion, weights=ionhistsweep[_xposidx])
        uzionmean = np.average(vzion, weights=ionhistsweep[_xposidx])

        uxelecmean = np.average(vxelec, weights=elechistsweep[_xposidx])
        uyelecmean = np.average(vyelec, weights=elechistsweep[_xposidx])
        uzelecmean = np.average(vzelec, weights=elechistsweep[_xposidx])

        _numion = np.sum(ionhistsweep[_xposidx])
        _numelec = np.sum(elechistsweep[_xposidx])

        _presion = (1./3.)*np.sum(((vxion-uxionmean)**2+(vyion-uyionmean)**2+(vzion-uzionmean)**2)*ionhistsweep[_xposidx])*dvion**3
        _preselec = (1./3.)*np.sum(((vxelec-uxelecmean)**2+(vyelec-uyelecmean)**2+(vzelec-uzelecmean)**2)*elechistsweep[_xposidx])*dvelec**3

        ionmomtemp[_xposidx] = _presion/_numion
        elecmomtemp[_xposidx] = _preselec/_numelec #TODO: check if we are missing factor of mass ratio here
    else:
        ionmomtemp[_xposidx] = 0.
        elecmomtemp[_xposidx] = 0.


#TODO: compute properly (use FAC throughout whole thing)
#compute quick estimate of Tperp
#compute temperature using 2nd moment
print("TODO: use FAC throughout whole temp perp and temp par calc (we do this in another script so its no big deal here")
ionmomtempperp = np.zeros((len(ionhistsweep)))
elecmomtempperp = np.zeros((len(ionhistsweep)))
ionmomtemppar = np.zeros((len(ionhistsweep)))
elecmomtemppar = np.zeros((len(ionhistsweep)))
vxion = np.asarray(vxion)
vyion = np.asarray(vyion)
vzion = np.asarray(vzion)
vxelec = np.asarray(vxelec)
vyelec = np.asarray(vyelec)
vzelec = np.asarray(vzelec)
dvion = vxion[1,1,1]-vxion[0,0,0]
dvelec = vxelec[1,1,1]-vxelec[0,0,0]
for _xposidx in range(0,len(ionhistsweep)):
    if(np.sum(ionhistsweep[_xposidx] > 0.)):
        uxionmean = np.average(vxion, weights=ionhistsweep[_xposidx])
        uyionmean = np.average(vyion, weights=ionhistsweep[_xposidx])
        uzionmean = np.average(vzion, weights=ionhistsweep[_xposidx])

        uxelecmean = np.average(vxelec, weights=elechistsweep[_xposidx])
        uyelecmean = np.average(vyelec, weights=elechistsweep[_xposidx])
        uzelecmean = np.average(vzelec, weights=elechistsweep[_xposidx])

        _numion = np.sum(ionhistsweep[_xposidx])
        _numelec = np.sum(elechistsweep[_xposidx])

        _presion = (1./3.)*np.sum(((vxion-uxionmean)**2+(vzion-uzionmean)**2)*ionhistsweep[_xposidx])*dvion**3
        _preselec = (1./3.)*np.sum(((vxelec-uxelecmean)**2+(vzelec-uzelecmean)**2)*elechistsweep[_xposidx])*dvelec**3

        _x1 = x_in[_xposidx]-dxin/2.
        _x2 = x_in[_xposidx]+dxin/2.
        _bx = ao.get_average_in_box(_x1, _x2, -99999999, 99999999, -99999999, 99999999, dfavg, 'bx')
        _by = ao.get_average_in_box(_x1, _x2, -99999999, 99999999, -99999999, 99999999, dfavg, 'by')
        _bz = ao.get_average_in_box(_x1, _x2, -99999999, 99999999, -99999999, 99999999, dfavg, 'bz')
        _b0 = np.sqrt(_bx**2+_by**2+_bz**2)
        ionmomtempperp[_xposidx] = (_presion/_numion)/_b0 #TODO: rename
        elecmomtempperp[_xposidx] = (_preselec/_numelec)/_b0 #TODO: check if we are missing factor of mass ratio here
        
        _presion = (1./3.)*np.sum(((vyion-uyionmean)**2)*ionhistsweep[_xposidx])*dvion**3
        _preselec = (1./3.)*np.sum(((vyelec-uyelecmean)**2)*elechistsweep[_xposidx])*dvelec**3
        ionmomtemppar[_xposidx] = (_presion/_numion)/_b0
        elecmomtemppar[_xposidx] = (_presion/_numion)/_b0
    else:
        ionmomtempperp[_xposidx] = 0.
        elecmomtempperp[_xposidx] = 0.
        ionmomtemppar[_xposidx] = 0.
        elecmomtemppar[_xposidx] = 0.
#Estimate Te if it were 'just' adiabatic
Teperpadiaup = elecinternale[np.nonzero(elecinternale)][-1] #get last nonzero entry #TODO: use perp T here as T0 instead of internal E #TODO: double check this (units and if internal energy is the same as temp in the far upstream)
Teperpadia = np.zeros((len(dfavg['ex_xx'])))
B0up = np.sqrt(dfavg['bx'][0,0,-1]**2+dfavg['by'][0,0,-1]**2+dfavg['bz'][0,0,-1]**2)
for _idx in range(0,len(Teperpadia)):
    _B0 = np.sqrt(dfavg['bx'][0,0,_idx]**2+dfavg['by'][0,0,_idx]**2+dfavg['bz'][0,0,_idx]**2)
    Teperpadia[_idx] = Teperpadiaup * _B0/B0up

#Estimate T using "stasiewicz2020quasi" model
Tstasiewicz = np.zeros((len(dfavg['ex_xx'])))
Tstasiewicz0 = elecmomtempperp[np.nonzero(elecmomtempperp)][-1]
B0up = np.sqrt(dfavg['bx'][0,0,-1]**2+dfavg['by'][0,0,-1]**2+dfavg['bz'][0,0,-1]**2)
for _idx in range(0,len(Tstasiewicz)):
    _B0 = np.sqrt(dfavg['bx'][0,0,_idx]**2+dfavg['by'][0,0,_idx]**2+dfavg['bz'][0,0,_idx]**2)
    Tstasiewicz[_idx] = (Tstasiewicz0/B0up)*(B0up/_B0)**(1./3.)

#compute energy in Efields, Bfields
print("TODO: double check that we are using correct units when computing energy of E fields")
#enerEfield = np.asarray([dfavg['ex'][0,0,_xxidx]**2+dfavg['ey'][0,0,_xxidx]**2+dfavg['ez'][0,0,_xxidx]**2 for _xxidx in range(0,len(dfavg['ex_xx']))])
#enerBfield = np.asarray([dfavg['bx'][0,0,_xxidx]**2+dfavg['by'][0,0,_xxidx]**2+dfavg['bz'][0,0,_xxidx]**2 for _xxidx in range(0,len(dfavg['bx_xx']))])

#use correct unnits for Efields, Bfields

dfields_nonorm = ld.load_fields(flpath,framenum,normalizeFields=False) #TODO
print("TODO: COMPUTE E FIELDS INN CORRECT FRAME when computed E field ener! (bit of a hassle since we need to load in the unnormalized and then boost. Should probably just write a function that removes normalization for dfields (should also be careful with how i'm handling x grid)")
dfavg_nonorm = aa.get_average_fields_over_yz(dfields_nonorm)
enerEfield = np.zeros((len(dfavg_nonorm['ex_xx'])))
enerBfield = np.zeros((len(dfavg_nonorm['ex_xx'])))
fluxSx = np.zeros((len(dfavg_nonorm['ex_xx'])))
fluxSy = np.zeros((len(dfavg_nonorm['ex_xx'])))
fluxSz = np.zeros((len(dfavg_nonorm['ex_xx'])))

nt = params['comp']/params['c']  #The systems time step is Δt = params['c']/params['comp'] * omega_{pe}^{-1}, thus we define nt as the number of steps per inv plasma period
_lastnonzeroidx = np.nonzero(ntotsion)[0][-1]
num_i_0 = ntotsion[_lastnonzeroidx]
print('num_i_0',num_i_0)
dx_sweepbox = x_in[1]-x_in[0]
dx_grid = dfields['ex_xx'][1]-dfields['ex_xx'][0]
num_i_0 = num_i_0*dx_grid/dx_sweepbox
print("DEBUG: num_i_0",num_i_0)
for _i in range(0,len(enerEfield)):
    enerEfield[_i] = (dfavg_nonorm['ex'][0,0,_i]**2+dfavg_nonorm['ey'][0,0,_i]**2+dfavg_nonorm['ez'][0,0,_i]**2)
    enerEfield[_i] = enerEfield[_i]*4*np.pi*params['c']**2*nt/(params['delgam']**2*(params['mi']/params['me'])**2*(ddenavg[_i])/num_i_0) #TODO: check factor of pi here (Seems fine) but KE and internal E have 1/n0 that this doesnt have!

    enerBfield[_i] = (dfavg_nonorm['bx'][0,0,_i]**2+dfavg_nonorm['by'][0,0,_i]**2+dfavg_nonorm['bz'][0,0,_i]**2)
    enerBfield[_i] = enerBfield[_i]*4*np.pi*params['c']**2*nt/(params['delgam']**2*(params['mi']/params['me'])**2*(ddenavg[_i])/num_i_0) #TODO: check factor of pi here

    fluxSx[_i] = dfavg_nonorm['ey'][0,0,_i]*dfavg_nonorm['bz'][0,0,_i]-dfavg_nonorm['ez'][0,0,_i]*dfavg_nonorm['by'][0,0,_i]
    fluxSy[_i] = dfavg_nonorm['ez'][0,0,_i]*dfavg_nonorm['bx'][0,0,_i]-dfavg_nonorm['ex'][0,0,_i]*dfavg_nonorm['bz'][0,0,_i]
    fluxSz[_i] = dfavg_nonorm['ex'][0,0,_i]*dfavg_nonorm['by'][0,0,_i]-dfavg_nonorm['ey'][0,0,_i]*dfavg_nonorm['bx'][0,0,_i]

    fluxSx[_i] = fluxSx[_i]*4*np.pi*params['c']**2*nt/(params['delgam']**2*(params['mi']/params['me'])**2*(ddenavg[_i])/num_i_0)
    fluxSy[_i] = fluxSy[_i]*4*np.pi*params['c']**2*nt/(params['delgam']**2*(params['mi']/params['me'])**2*(ddenavg[_i])/num_i_0)
    fluxSz[_i] = fluxSz[_i]*4*np.pi*params['c']**2*nt/(params['delgam']**2*(params['mi']/params['me'])**2*(ddenavg[_i])/num_i_0)

#Compute stochastic parameter (Stasiewicz et al 2020)
#note: mass is in units of mass per charge, so this works out to be dimensionless
#TODO: consider if we can actually use dfavg here (as full div E requires y variation; that is we are ignoring Ez), and we should probably not use Ex,Ez but instead compute local Eperp1 and Eperp2 
_Eperp = dfavg['ex'][0,0,:]
_va = np.sqrt(params['sigma']*params['me']/params['mi'])*params['c'] #NOTE: this is subtely different than what aaron's normalization is- fix it (missingn factor of gamma0 and mi+me)
_divE = np.gradient(_Eperp)*_va#final units of (E/B0)  (E was normalized to va B0) 
_Bsqrd = dfavg['bx'][0,0,:]*dfavg['bx'][0,0,:]+dfavg['by'][0,0,:]*dfavg['by'][0,0,:]+dfavg['bz'][0,0,:]*dfavg['bz'][0,0,:]
stoc_param_i = (_Bsqrd/inputs['mi'])**-1*_divE #TODO: CHECK UNITS ON THIS!!!
stoc_param_e = (_Bsqrd/inputs['me'])**-1*_divE

#TODO: remove from master list and from master list plotting routines
ionTemp = np.zeros((len(dfavg['ex_xx'])))
hx_in = dfavg['ex_xx']

#compute ion and elec density
elecden = np.asarray([np.sum(Hist_vxvyelec[_tempidx]) for _tempidx in range(0,len(x_in))])
ionden = np.asarray([np.sum(Hist_vxvyion[_tempidx]) for _tempidx in range(0,len(x_in))])


#interpolate onto x_in
def interpolate(independent_vars, dependent_vars, locations):
    independent_vars = np.array(independent_vars)
    dependent_vars = np.array(dependent_vars)
    locations = np.array(locations)
    interpolated_values = np.interp(locations, independent_vars, dependent_vars)

    return locations, interpolated_values

_, enerjxshockEx =interpolate(dfields['ex_xx'],enerjxshockEx,x_in)

#compute energy gain due to j dot E
xcoord_egain, enerjxshockEx_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerjxshockEx,False,verbose=True)#(dflowavgrest,x_in,enerjxshockEx,False,verbose=True)

xcoord_egain, enerCEtot_ion_bar_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_ion_bar,True)
xcoord_egain, enerCEx_ion_bar_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_ion_bar,True)
xcoord_egain, enerCEy_ion_bar_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_ion_bar,True)
xcoord_egain, enerCEz_ion_bar_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_ion_bar,True)
xcoord_egain, enerCEtot_ion_tilde_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_ion_tilde,True)
xcoord_egain, enerCEx_ion_tilde_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_ion_tilde,True)
xcoord_egain, enerCEy_ion_tilde_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_ion_tilde,True)
xcoord_egain, enerCEz_ion_tilde_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_ion_tilde,True)
xcoord_egain, enerCEtot_elec_bar_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_elec_bar,False)
xcoord_egain, enerCEx_elec_bar_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_elec_bar,False)
xcoord_egain, enerCEy_elec_bar_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_elec_bar,False)
xcoord_egain, enerCEz_elec_bar_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_elec_bar,False)
xcoord_egain, enerCEtot_elec_tilde_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_elec_tilde,False)
xcoord_egain, enerCEx_elec_tilde_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_elec_tilde,False)
xcoord_egain, enerCEy_elec_tilde_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_elec_tilde,False)
xcoord_egain, enerCEz_elec_tilde_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_elec_tilde,False)

xcoord_egain, enerCEtot_ion_facfluclowpass_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_ion_facfluclowpass,True)
xcoord_egain, enerCEx_ion_facfluclowpass_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_ion_facfluclowpass,True)
xcoord_egain, enerCEy_ion_facfluclowpass_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_ion_facfluclowpass,True)
xcoord_egain, enerCEz_ion_facfluclowpass_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_ion_facfluclowpass,True)

xcoord_egain, enerCEtot_ion_fac_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_ion_fac,True)
xcoord_egain, enerCEx_ion_fac_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_ion_fac,True)
xcoord_egain, enerCEy_ion_fac_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_ion_fac,True)
xcoord_egain, enerCEz_ion_fac_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_ion_fac,True)

xcoord_egain, enerCEtot_ion_facfluchighdetrend_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_ion_facfluchighdetrend,True)
xcoord_egain, enerCEx_ion_facfluchighdetrend_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_ion_facfluchighdetrend,True)
xcoord_egain, enerCEy_ion_facfluchighdetrend_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_ion_facfluchighdetrend,True)
xcoord_egain, enerCEz_ion_facfluchighdetrend_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_ion_facfluchighdetrend,True)

xcoord_egain, enerCEtot_elec_facfluclowpass_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_elec_facfluclowpass,False)
xcoord_egain, enerCEx_elec_facfluclowpass_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_elec_facfluclowpass,False)
xcoord_egain, enerCEy_elec_facfluclowpass_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_elec_facfluclowpass,False)
xcoord_egain, enerCEz_elec_facfluclowpass_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_elec_facfluclowpass,False)

xcoord_egain, enerCEtot_elec_facfluchighdetrend_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_elec_facfluchighdetrend,False)
xcoord_egain, enerCEx_elec_facfluchighdetrend_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_elec_facfluchighdetrend,False)
xcoord_egain, enerCEy_elec_facfluchighdetrend_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_elec_facfluchighdetrend,False)
xcoord_egain, enerCEz_elec_facfluchighdetrend_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_elec_facfluchighdetrend,False)

xcoord_egain, enerCEx_ion_facfluc_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_ion_facfluc,True)
xcoord_egain, enerCEy_ion_facfluc_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_ion_facfluc,True)
xcoord_egain, enerCEz_ion_facfluc_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_ion_facfluc,True)
xcoord_egain, enerCEtot_ion_facfluc_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_ion_facfluc,True)
xcoord_egain, enerCEx_ion_facfluclocal_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_ion_facfluclocal,True)
xcoord_egain, enerCEy_ion_facfluclocal_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_ion_facfluclocal,True)
xcoord_egain, enerCEz_ion_facfluclocal_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_ion_facfluclocal,True)
xcoord_egain, enerCEtot_ion_facfluclocal_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_ion_facfluclocal,True)
xcoord_egain, enerCEx_elec_facfluc_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_elec_facfluc,False)
xcoord_egain, enerCEy_elec_facfluc_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_elec_facfluc,False)
xcoord_egain, enerCEz_elec_facfluc_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_elec_facfluc,False)
xcoord_egain, enerCEtot_elec_facfluc_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_elec_facfluc,False)
xcoord_egain, enerCEx_elec_facfluclocal_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_elec_facfluclocal,False)
xcoord_egain, enerCEy_elec_facfluclocal_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_elec_facfluclocal,False)
xcoord_egain, enerCEz_elec_facfluclocal_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_elec_facfluclocal,False)
xcoord_egain, enerCEtot_elec_facfluclocal_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_elec_facfluclocal,False)

xcoord_egain, enerCEx_elec_fac_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEx_elec_fac,False)
xcoord_egain, enerCEy_elec_fac_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEy_elec_fac,False)
xcoord_egain, enerCEz_elec_fac_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEz_elec_fac,False)
xcoord_egain, enerCEtot_elec_fac_egain = aa.compute_gain_due_to_jdotE(dflow,x_in,enerCEtot_elec_fac,False)


_xcoord_egain, enerCEx_ion_facavglocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEx_ion_facavglocframe,True)
_xcoord_egain, enerCEy_ion_facavglocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEy_ion_facavglocframe,True)
_xcoord_egain, enerCEz_ion_facavglocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEz_ion_facavglocframe,True)
_xcoord_egain, enerCEtot_ion_facavglocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEtot_ion_facavglocframe,True)

_xcoord_egain, enerCEx_ion_facfluclocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEx_ion_facfluclocframe,True)
_xcoord_egain, enerCEy_ion_facfluclocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEy_ion_facfluclocframe,True)
_xcoord_egain, enerCEz_ion_facfluclocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEz_ion_facfluclocframe,True)
_xcoord_egain, enerCEtot_ion_facfluclocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEtot_ion_facfluclocframe,True)


_xcoord_egain, enerCEx_elec_facavglocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEx_elec_facavglocframe,False)
_xcoord_egain, enerCEy_elec_facavglocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEy_elec_facavglocframe,False)
_xcoord_egain, enerCEz_elec_facavglocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEz_elec_facavglocframe,False)
_xcoord_egain, enerCEtot_elec_facavglocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEtot_elec_facavglocframe,False)

_xcoord_egain, enerCEx_elec_facfluclocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEx_elec_facfluclocframe,False)
_xcoord_egain, enerCEy_elec_facfluclocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEy_elec_facfluclocframe,False)
_xcoord_egain, enerCEz_elec_facfluclocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEz_elec_facfluclocframe,False)
_xcoord_egain, enerCEtot_elec_facfluclocframe_egain = aa.compute_gain_due_to_jdotE(dflow,_x_in,enerCEtot_elec_facfluclocframe,False,verbose=False)


#enerCEx_ion_facfluclowpass,enerCEy_ion_facfluclowpass,enerCEz_ion_facfluclowpass,enerCEtot_ion_facfluclowpass,
#enerCEx_ion_facfluchighdetrend,enerCEy_ion_facfluchighdetrend,enerCEz_ion_facfluchighdetrend,enerCEtot_ion_facfluchighdetrend,
#enerCEx_elec_facfluclowpass,enerCEy_elec_facfluclowpass,enerCEz_elec_facfluclowpass,enerCEtot_elec_facfluclowpass,
#enerCEx_elec_facfluchighdetrend,enerCEy_elec_facfluchighdetrend,enerCEz_elec_facfluchighdetrend,enerCEtot_elec_facfluchighdetrend

#get around load font bug!
plt.figure()
plt.style.use('cb.mplstyle')
plt.plot([0,1],[0,1])
plt.savefig('testdummy.png',format='png',dpi=30)
plt.close()

#make another pub figure -------
plt.style.use('cb.mplstyle')
fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(20,4.5*2), sharex=True)

_linewidth = 5

#ax1.plot(x_in,enerjxshockEx,ls=':',color='orange',linewidth=_linewidth,label=r'$j_{x,e,shock} \overline{E_x}$')
#ax1.plot(x_in,enerCEtot_elec_fac,ls='-.',color='gray',linewidth=_linewidth,label=r'$\int \, \overline{C_{\mathbf{E},e}} \, d^3\mathbf{v}$')
ax1.plot(x_in,enerjxshockEx,ls='-',color='black',linewidth=_linewidth,label=r'$\int \, \overline{C_{\mathbf{E}^{\prime \prime},sim,e}} \, d^3\mathbf{v}-\overline{j_{x,e,sim,shock}} \overline{E_x^{\prime \prime}}$')
ax1.plot(_x_in,enerCEtot_elec_facfluclocframe,ls='--',color='purple',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{\mathbf{E},e,down}} \, d^3\mathbf{v}$')
ax1.legend(loc='upper right')
ax1.grid()

#ax2.plot(xcoord_egain,enerjxshockEx_egain,ls=':',color='orange',linewidth=_linewidth,label=r'$W_{j_{x,e,shock} \overline{E_x}}$')
#ax2.plot(xcoord_egain,enerCEtot_elec_fac_egain,ls='-.',color='gray',linewidth=_linewidth,label=r'$W_{\overline{\mathbf{j}_e}\cdot\overline{\mathbf{E}}}$')
ax2.plot(xcoord_egain,enerjxshockEx_egain,ls='-',color='black',linewidth=_linewidth,label=r'$W_{\overline{\mathbf{j}_{e,sim}}\cdot\overline{\mathbf{E}^{\prime \prime}}}-W_{\overline{j_{x,e,sim,shock}} \overline{E_x^{\prime \prime}}}$')
ax2.plot(_xcoord_egain,enerCEtot_elec_facfluclocframe_egain,ls='--',color='purple',linewidth=_linewidth,label=r'$W_{\widetilde{\mathbf{j}_{e,down}}\cdot\widetilde{\mathbf{E}^\prime}}$')
ax2.legend(loc='upper right')
ax2.grid()

ax2.set_xlabel(r"$x/d_{i}$", fontsize=32)
ax2.set_xlim(5,12)
#ax2.set_ylim(0,19)
ax2.set_xticks(np.arange(5,12,1))

plt.subplots_adjust(hspace=0.025)

flnm='figures/eneradiavsnonadia_elec.png'
plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
plt.close()


#end another pub figure -------




#make final pub figure--------------------
#enerCEy_elec_facfluclowpass enerCEx_elec_facfluclowpass_egain
enerCEx_elec_facfluchighpass_egain=enerCEx_elec_facfluc_egain-enerCEx_elec_facfluclowpass_egain
enerCEy_elec_facfluchighpass_egain=enerCEy_elec_facfluc_egain-enerCEy_elec_facfluclowpass_egain
enerCEz_elec_facfluchighpass_egain=enerCEz_elec_facfluc_egain-enerCEz_elec_facfluclowpass_egain
enerCEtot_elec_facfluchighpass_egain=enerCEtot_elec_facfluc_egain-enerCEtot_elec_facfluclowpass_egain

enerCEx_elec_facfluchighpass=enerCEx_elec_facfluc-enerCEx_elec_facfluclowpass
enerCEy_elec_facfluchighpass=enerCEy_elec_facfluc-enerCEy_elec_facfluclowpass
enerCEz_elec_facfluchighpass=enerCEz_elec_facfluc-enerCEz_elec_facfluclowpass
enerCEtot_elec_facfluchighpass=enerCEtot_elec_facfluc-enerCEtot_elec_facfluclowpass

plt.style.use('cb.mplstyle')
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(20,4.5*6), sharex=True)

_linewidth = 5

#rates of e bar, e tidle low, and e tilde high
ax1.plot(x_in,enerCEtot_elec_fac,ls='-',color='black',linewidth=_linewidth,label=r'$\int \, \overline{C_{\mathbf{E},e}} \, d^3\mathbf{v}$')
ax1.plot(x_in,enerCEx_elec_fac,ls=':',color='green',linewidth=_linewidth,label=r'$\int \, \overline{C_{E_{||},e}} \, d^3\mathbf{v}$')
ax1.plot(x_in,enerCEy_elec_fac,ls='-.',color='red',linewidth=_linewidth,label=r'$\int \, \overline{C_{E_{\perp,1},e}} \, d^3\mathbf{v}$')
ax1.plot(x_in,enerCEz_elec_fac,ls='--',color='blue',linewidth=_linewidth,label=r'$\int \, \overline{C_{E_{\perp,2},e}} \, d^3\mathbf{v}$')
ax1.legend(loc='upper right')
ax1.grid()

ax2.plot(x_in,enerCEtot_elec_facfluclowpass,ls='-',color='black',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{\mathbf{E},e}}^{k_{||} d_i < 15} \, d^3\mathbf{v}$')
ax2.plot(x_in,enerCEx_elec_facfluclowpass,ls=':',color='green',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{||},e}}^{k_{||} d_i < 15} \, d^3\mathbf{v}$')
ax2.plot(x_in,enerCEy_elec_facfluclowpass,ls='-.',color='red',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{\perp,1},e}}^{k_{||} d_i < 15} \, d^3\mathbf{v}$')
ax2.plot(x_in,enerCEz_elec_facfluclowpass,ls='--',color='blue',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{\perp,2},e}}^{k_{||} d_i < 15} \, d^3\mathbf{v}$')
ax2.legend(loc='upper right')
ax2.grid()

ax3.plot(x_in,enerCEtot_elec_facfluchighpass,ls='-',color='black',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{\mathbf{E},e}}^{k_{||} d_i > 15} \, d^3\mathbf{v}$')
ax3.plot(x_in,enerCEx_elec_facfluchighpass,ls=':',color='green',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{||},e}}^{k_{||} d_i > 15} \, d^3\mathbf{v}$')
ax3.plot(x_in,enerCEy_elec_facfluchighpass,ls='-.',color='red',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{\perp,1},e}}^{k_{||} d_i > 15} \, d^3\mathbf{v}$')
ax3.plot(x_in,enerCEz_elec_facfluchighpass,ls='--',color='blue',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{\perp,2},e}}^{k_{||} d_i > 15} \, d^3\mathbf{v}$')
ax3.legend(loc='upper right')
ax3.grid()

#Total W of E 
ax4.plot(xcoord_egain,enerCEtot_elec_fac_egain,ls='-',color='black',linewidth=_linewidth,label=r'$W_{\overline{\mathbf{j}_e}\cdot\overline{\mathbf{E}}}$')
ax4.plot(xcoord_egain,enerCEx_elec_fac_egain,ls=':',color='green',linewidth=_linewidth,label=r'$W_{\overline{j_{||,e}}\cdot \overline{E_{||}}}$')
ax4.plot(xcoord_egain,enerCEy_elec_fac_egain,ls='-.',color='red',linewidth=_linewidth,label=r'$W_{\overline{j_{\perp,1,e}}\cdot \overline{E_{\perp,1}}}$')
ax4.plot(xcoord_egain,enerCEz_elec_fac_egain,ls='--',color='blue',linewidth=_linewidth,label=r'$W_{\overline{j_{\perp,2,e}}\cdot \overline{E_{\perp,2}}}$')
ax4.legend(loc='upper right')
ax4.grid()

ax5.plot(xcoord_egain,enerCEtot_elec_facfluclowpass_egain,ls='-',color='black',linewidth=_linewidth,label=r'$W_{\widetilde{\mathbf{j}_e}^{k_{||} d_i < 15}\cdot\widetilde{\mathbf{E}}}^{k_{||} d_i < 15}$')
ax5.plot(xcoord_egain,enerCEx_elec_facfluclowpass_egain,ls=':',color='green',linewidth=_linewidth,label=r'$W_{\widetilde{j_{||,e}}^{k_{||} d_i < 15}\cdot\widetilde{E_{||}}}^{k_{||} d_i < 15}$')
ax5.plot(xcoord_egain,enerCEy_elec_facfluclowpass_egain,ls='-.',color='red',linewidth=_linewidth,label=r'$W_{\widetilde{j_{\perp,1,e}}^{k_{||} d_i < 15}\cdot\widetilde{E_{\perp,1}}}^{k_{||} d_i < 15}$')
ax5.plot(xcoord_egain,enerCEz_elec_facfluclowpass_egain,ls='--',color='blue',linewidth=_linewidth,label=r'$W_{\widetilde{j_{\perp,2,e}}^{k_{||} d_i < 15}\cdot\widetilde{E_{\perp,2}}}^{k_{||} d_i < 15}$')
ax5.legend(loc='upper right')
ax5.grid()

ax6.plot(xcoord_egain,enerCEtot_elec_facfluchighpass_egain,ls='-',color='black',linewidth=_linewidth,label=r'$W_{\widetilde{\mathbf{j}_e}^{k_{||} d_i > 15}\cdot\widetilde{\mathbf{E}}}^{k_{||} d_i > 15}$')
ax6.plot(xcoord_egain,enerCEx_elec_facfluchighpass_egain,ls=':',color='green',linewidth=_linewidth,label=r'$W_{\widetilde{j_{||,e}}^{k_{||} d_i > 15}\cdot\widetilde{E_{||}}}^{k_{||} d_i > 15}$')
ax6.plot(xcoord_egain,enerCEy_elec_facfluchighpass_egain,ls='-.',color='red',linewidth=_linewidth,label=r'$W_{\widetilde{j_{\perp,1,e}}^{k_{||} d_i > 15}\cdot\widetilde{E_{\perp,1}}}^{k_{||} d_i > 15}$')
ax6.plot(xcoord_egain,enerCEz_elec_facfluchighpass_egain,ls='--',color='blue',linewidth=_linewidth,label=r'$W_{\widetilde{j_{\perp,2,e}}^{k_{||} d_i > 15}\cdot\widetilde{E_{\perp,2}}}^{k_{||} d_i > 15}$')
ax6.legend(loc='upper right')
ax6.grid()

ax6.set_xlabel(r"$x/d_{i}$", fontsize=32)
ax6.set_xlim(5,12)
ax6.set_xticks(np.arange(5,12,1))

plt.subplots_adjust(hspace=0.025)

flnm='figures/enerandenerratesvsx_elec.png'
plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
plt.close()


#end makepubfigure-------


#make final pub figure--------------------

plt.style.use('cb.mplstyle')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20,4.5*4), sharex=True)

_linewidth = 5

#rates of e bar, e tidle low, and e tilde high
ax1.plot(x_in,enerCEtot_ion_fac,ls='-',color='black',linewidth=_linewidth,label=r'$\int \, \overline{C_{\mathbf{E},i}} \, d^3\mathbf{v}$')
ax1.plot(x_in,enerCEx_ion_fac,ls=':',color='green',linewidth=_linewidth,label=r'$\int \, \overline{C_{E_{||},i}} \, d^3\mathbf{v}$')
ax1.plot(x_in,enerCEy_ion_fac,ls='-.',color='red',linewidth=_linewidth,label=r'$\int \, \overline{C_{E_{\perp,1},i}} \, d^3\mathbf{v}$')
ax1.plot(x_in,enerCEz_ion_fac,ls='--',color='blue',linewidth=_linewidth,label=r'$\int \, \overline{C_{E_{\perp,2},i}} \, d^3\mathbf{v}$')
ax1.legend(loc='upper right')
ax1.grid()

ax2.plot(x_in,enerCEtot_ion_facfluc,ls='-',color='black',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{\mathbf{E},i}} \, d^3\mathbf{v}$')
ax2.plot(x_in,enerCEx_ion_facfluc,ls=':',color='green',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{||},i}} \, d^3\mathbf{v}$')
ax2.plot(x_in,enerCEy_ion_facfluc,ls='-.',color='red',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{\perp,1},i}} \, d^3\mathbf{v}$')
ax2.plot(x_in,enerCEz_ion_facfluc,ls='--',color='blue',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{\perp,2},i}} \, d^3\mathbf{v}$')
ax2.legend(loc='upper right')
ax2.grid()

#Total W of E 
ax3.plot(xcoord_egain,enerCEtot_ion_fac_egain,ls='-',color='black',linewidth=_linewidth,label=r'$W_{\overline{\mathbf{j}_i}\cdot\overline{\mathbf{E}}}$')
ax3.plot(xcoord_egain,enerCEx_ion_fac_egain,ls=':',color='green',linewidth=_linewidth,label=r'$W_{\overline{j_{||,i}}\cdot \overline{E_{||}}}$')
ax3.plot(xcoord_egain,enerCEy_ion_fac_egain,ls='-.',color='red',linewidth=_linewidth,label=r'$W_{\overline{j_{\perp,1,i}}\cdot \overline{E_{\perp,1}}}$')
ax3.plot(xcoord_egain,enerCEz_ion_fac_egain,ls='--',color='blue',linewidth=_linewidth,label=r'$W_{\overline{j_{\perp,2,i}}\cdot \overline{E_{\perp,2}}}$')
ax3.legend(loc='upper right')
ax3.grid()

ax4.plot(xcoord_egain,enerCEtot_ion_facfluc_egain,ls='-',color='black',linewidth=_linewidth,label=r'$W_{\widetilde{\mathbf{j}_i}\cdot\widetilde{\mathbf{E}}}$')
ax4.plot(xcoord_egain,enerCEx_ion_facfluc_egain,ls=':',color='green',linewidth=_linewidth,label=r'$W_{\widetilde{j_{||,i}}\cdot\widetilde{E_{||}}}$')
ax4.plot(xcoord_egain,enerCEy_ion_facfluc_egain,ls='-.',color='red',linewidth=_linewidth,label=r'$W_{\widetilde{j_{\perp,1,i}}\cdot\widetilde{E_{\perp,1}}}$')
ax4.plot(xcoord_egain,enerCEz_ion_facfluc_egain,ls='--',color='blue',linewidth=_linewidth,label=r'$W_{\widetilde{j_{\perp,2,i}}\cdot\widetilde{E_{\perp,2}}}$')
ax4.legend(loc='upper right')
ax4.grid()

ax4.set_xlabel(r"$x/d_{i}$", fontsize=32)
ax4.set_xlim(5,12)
ax4.set_xticks(np.arange(5,12,1))

plt.subplots_adjust(hspace=0.025)

flnm='figures/enerandenerratesvsx_ion.png'
plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
plt.close()


#end makepubfigure-------


#make final pub figure--------------------
plt.style.use('cb.mplstyle')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20,4.5*4), sharex=True)

_linewidth = 5

#rates of e bar, e tidle low, and e tilde high
ax1.plot(_x_in,enerCEtot_ion_facavglocframe,ls='-',color='black',linewidth=_linewidth,label=r'$\int \, \overline{C_{\mathbf{E}^\prime,i,down}} \, d^3\mathbf{v}$')
ax1.plot(_x_in,enerCEx_ion_facavglocframe,ls=':',color='green',linewidth=_linewidth,label=r'$\int \, \overline{C_{E_{||}^\prime,i,down}} \, d^3\mathbf{v}$')
ax1.plot(_x_in,enerCEy_ion_facavglocframe,ls='-.',color='red',linewidth=_linewidth,label=r'$\int \, \overline{C_{E_{\perp,1}^\prime,i,down}} \, d^3\mathbf{v}$')
ax1.plot(_x_in,enerCEz_ion_facavglocframe,ls='--',color='blue',linewidth=_linewidth,label=r'$\int \, \overline{C_{E_{\perp,2}^\prime,i,down}} \, d^3\mathbf{v}$')
ax1.legend(loc='upper right')
ax1.grid()

ax2.plot(_x_in,enerCEtot_ion_facfluclocframe,ls='-',color='black',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{\mathbf{E}^\prime,i,down}} \, d^3\mathbf{v}$')
ax2.plot(_x_in,enerCEx_ion_facfluclocframe,ls=':',color='green',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{||}^\prime,i,down}} \, d^3\mathbf{v}$')
ax2.plot(_x_in,enerCEy_ion_facfluclocframe,ls='-.',color='red',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{\perp,1}^\prime,i,down}} \, d^3\mathbf{v}$')
ax2.plot(_x_in,enerCEz_ion_facfluclocframe,ls='--',color='blue',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{\perp,2}^\prime,i,down}} \, d^3\mathbf{v}$')
ax2.legend(loc='upper right')
ax2.grid()

#Total W of E 
ax3.plot(_xcoord_egain,enerCEtot_ion_facavglocframe_egain,ls='-',color='black',linewidth=_linewidth,label=r'$W_{\overline{\mathbf{j}_{i,down}}\cdot\overline{\mathbf{E}^\prime}}$')
ax3.plot(_xcoord_egain,enerCEx_ion_facavglocframe_egain,ls=':',color='green',linewidth=_linewidth,label=r'$W_{\overline{j_{||,i,down}}\cdot \overline{E_{||}^\prime}}$')
ax3.plot(_xcoord_egain,enerCEy_ion_facavglocframe_egain,ls='-.',color='red',linewidth=_linewidth,label=r'$W_{\overline{j_{\perp,1,i,down}}\cdot \overline{E_{\perp,1}^\prime}}$')
ax3.plot(_xcoord_egain,enerCEz_ion_facavglocframe_egain,ls='--',color='blue',linewidth=_linewidth,label=r'$W_{\overline{j_{\perp,2,i,down}}\cdot \overline{E_{\perp,2}^\prime}}$')
ax3.legend(loc='upper right')
ax3.grid()

ax4.plot(_xcoord_egain,enerCEtot_ion_facfluclocframe_egain,ls='-',color='black',linewidth=_linewidth,label=r'$W_{\widetilde{\mathbf{j}_{i,down}}\cdot\widetilde{\mathbf{E}^\prime}}$')
ax4.plot(_xcoord_egain,enerCEx_ion_facfluclocframe_egain,ls=':',color='green',linewidth=_linewidth,label=r'$W_{\widetilde{j_{||,i,down}}\cdot\widetilde{E_{||}^\prime}}$')
ax4.plot(_xcoord_egain,enerCEy_ion_facfluclocframe_egain,ls='-.',color='red',linewidth=_linewidth,label=r'$W_{\widetilde{j_{\perp,1,i,down}}\cdot\widetilde{E_{\perp,1}^\prime}}$')
ax4.plot(_xcoord_egain,enerCEz_ion_facfluclocframe_egain,ls='--',color='blue',linewidth=_linewidth,label=r'$W_{\widetilde{j_{\perp,2,i,down}}\cdot\widetilde{E_{\perp,2}^\prime}}$')
ax4.legend(loc='upper right')
ax4.grid()

ax4.set_xlabel(r"$x/d_{i}$", fontsize=32)
ax4.set_xlim(5,12)
ax4.set_xticks(np.arange(5,12,1))

plt.subplots_adjust(hspace=0.025)

flnm='figures/enerandenerratesvsx_downE_ion.png'
plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
plt.close()

#end make final pub figure--------------------

# make final pub figure-----------------------
plt.style.use('cb.mplstyle')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20,4.5*4), sharex=True)

_linewidth = 5

#rates of e bar, e tidle low, and e tilde high
ax1.plot(_x_in,enerCEtot_elec_facavglocframe,ls='-',color='black',linewidth=_linewidth,label=r'$\int \, \overline{C_{\mathbf{E}^\prime,e,down}} \, d^3\mathbf{v}$')
ax1.plot(_x_in,enerCEx_elec_facavglocframe,ls=':',color='green',linewidth=_linewidth,label=r'$\int \, \overline{C_{E_{||}^\prime,e,down}} \, d^3\mathbf{v}$')
ax1.plot(_x_in,enerCEy_elec_facavglocframe,ls='-.',color='red',linewidth=_linewidth,label=r'$\int \, \overline{C_{E_{\perp,1}^\prime,e,down}} \, d^3\mathbf{v}$')
ax1.plot(_x_in,enerCEz_elec_facavglocframe,ls='--',color='blue',linewidth=_linewidth,label=r'$\int \, \overline{C_{E_{\perp,2}^\prime,e,down}} \, d^3\mathbf{v}$')
ax1.legend(loc='upper right')
ax1.grid()

ax2.plot(_x_in,enerCEtot_elec_facfluclocframe,ls='-',color='black',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{\mathbf{E}^\prime,e,down}} \, d^3\mathbf{v}$')
ax2.plot(_x_in,enerCEx_elec_facfluclocframe,ls=':',color='green',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{||}^\prime,e,down}} \, d^3\mathbf{v}$')
ax2.plot(_x_in,enerCEy_elec_facfluclocframe,ls='-.',color='red',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{\perp,1}^\prime,e,down}} \, d^3\mathbf{v}$')
ax2.plot(_x_in,enerCEz_elec_facfluclocframe,ls='--',color='blue',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{E_{\perp,2}^\prime,e,down}} \, d^3\mathbf{v}$')
ax2.legend(loc='upper right')
ax2.grid()

#Total W of E 
ax3.plot(_xcoord_egain,enerCEtot_elec_facavglocframe_egain,ls='-',color='black',linewidth=_linewidth,label=r'$W_{\overline{\mathbf{j}_{e,down}}\cdot\overline{\mathbf{E}^\prime}}$')
ax3.plot(_xcoord_egain,enerCEx_elec_facavglocframe_egain,ls=':',color='green',linewidth=_linewidth,label=r'$W_{\overline{j_{||,e,down}}\cdot \overline{E_{||}^\prime}}$')
ax3.plot(_xcoord_egain,enerCEy_elec_facavglocframe_egain,ls='-.',color='red',linewidth=_linewidth,label=r'$W_{\overline{j_{\perp,1,e,down}}\cdot \overline{E_{\perp,1}^\prime}}$')
ax3.plot(_xcoord_egain,enerCEz_elec_facavglocframe_egain,ls='--',color='blue',linewidth=_linewidth,label=r'$W_{\overline{j_{\perp,2,e,down}}\cdot \overline{E_{\perp,2}^\prime}}$')
ax3.legend(loc='upper right')
ax3.grid()

ax4.plot(_xcoord_egain,enerCEtot_elec_facfluclocframe_egain,ls='-',color='black',linewidth=_linewidth,label=r'$W_{\widetilde{\mathbf{j}_{e,down}}\cdot\widetilde{\mathbf{E}^\prime}}$')
ax4.plot(_xcoord_egain,enerCEx_elec_facfluclocframe_egain,ls=':',color='green',linewidth=_linewidth,label=r'$W_{\widetilde{j_{||,e,down}}\cdot\widetilde{E_{||}^\prime}}$')
ax4.plot(_xcoord_egain,enerCEy_elec_facfluclocframe_egain,ls='-.',color='red',linewidth=_linewidth,label=r'$W_{\widetilde{j_{\perp,1,e,down}}\cdot\widetilde{E_{\perp,1}^\prime}}$')
ax4.plot(_xcoord_egain,enerCEz_elec_facfluclocframe_egain,ls='--',color='blue',linewidth=_linewidth,label=r'$W_{\widetilde{j_{\perp,2,e,down}}\cdot\widetilde{E_{\perp,2}^\prime}}$')
ax4.legend(loc='upper right')
ax4.grid()

ax4.set_xlabel(r"$x/d_{i}$", fontsize=32)
ax4.set_xlim(5,12)
ax4.set_xticks(np.arange(5,12,1))

plt.subplots_adjust(hspace=0.025)

flnm='figures/enerandenerratesvsx_downE_elec.png'
plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
plt.close()

#end make final pub figure


#compute total ener and ener flux of ion and elec flow
dflowavg = aa.get_average_flow_over_yz(dflow)
dx = (dflowavg["ui_xx"][1]-dflowavg["ui_xx"][0])/2
KEions = []
parEfluxion = []
KEelecs = []
parEfluxelec = []
xxKEplot = []
m_i = params['mi']
m_e = 1.
vti0 = np.sqrt(params['delgam']) #TODO: move to clean up code and gather all other constants that we use in this script
vte0 = np.sqrt(params['mi']/params['me'])*vti0
for _xposidx in range(0,len(dflowavg["ui_xx"])):
    utotsqrd = (dflowavg['ui'][0,0,_xposidx]**2.+dflowavg['vi'][0,0,_xposidx]**2.+dflowavg['wi'][0,0,_xposidx]**2.0) #TODO: check this normalization
    utotsqrdelec = (dflowavg['ue'][0,0,_xposidx]**2.+dflowavg['ve'][0,0,_xposidx]**2.+dflowavg['we'][0,0,_xposidx]**2.0)
    KEions.append(0.5*m_i*ddenavg[_xposidx]/2.*utotsqrd) #1/2 n_i m_i U_i^2.
    KEelecs.append(0.5*m_e*ddenavg[_xposidx]/2.*utotsqrdelec)
    parEfluxion.append(0.5*m_i*ddenavg[_xposidx]*utotsqrd*dflowavg['ui'][0,0,_xposidx])#only want flux in x direction!
    parEfluxelec.append(0.5*m_e*ddenavg[_xposidx]*utotsqrdelec*dflowavg['ue'][0,0,_xposidx])
    xxKEplot.append(dden['dens_xx'][_xposidx])
KEions = np.asarray(KEions)
KEelecs = np.asarray(KEelecs)
parEfluxion = np.asarray(parEfluxion)
parEfluxelec = np.asarray(parEfluxelec)
xxKEplot = np.asarray(xxKEplot)

#compute total ener of ion and elec temp
#TECHNICALLY THIS IS INTERNAL ENERGY, WHICH IS DIFFERNT THAN TEMPERATURE (hence why we comment out the factor of 1/edens and 1/idens)
#TEMPERATURE IS RELATED TO THE AVERAGE VELOCITY WHILE INTERNAL ENERGY IS RELATED TO THE AVERAGE VELOCITY TIMES THE NUMBER OF PARTICLES
print("TODO: use FAC throughout whole temp perp and temp par calc (we do this in another script so its no big deal here")
iontemp = np.zeros((len(ionhistsweep)))
electemp = np.zeros((len(ionhistsweep)))
vxion = np.asarray(vxion)
vyion = np.asarray(vyion)
vzion = np.asarray(vzion)
vxelec = np.asarray(vxelec)
vyelec = np.asarray(vyelec)
vzelec = np.asarray(vzelec)
dvion = vxion[1,1,1]-vxion[0,0,0]
dvelec = vxelec[1,1,1]-vxelec[0,0,0]
x_in_temp = []
_numionup = np.sum(ionhistsweep[int(len(ionhistsweep)*.95)])
_numelecup = np.sum(elechistsweep[int(len(elechistsweep)*.95)])
print('_numionup',_numionup)
print('_numelecup',_numelecup)
for _xposidx in range(0,len(ionhistsweep)):
    if(np.sum(ionhistsweep[_xposidx] > 0.)):
        uxionmean = np.average(vxion, weights=ionhistsweep[_xposidx])
        uyionmean = np.average(vyion, weights=ionhistsweep[_xposidx])
        uzionmean = np.average(vzion, weights=ionhistsweep[_xposidx])

        uxelecmean = np.average(vxelec, weights=elechistsweep[_xposidx])
        uyelecmean = np.average(vyelec, weights=elechistsweep[_xposidx])
        uzelecmean = np.average(vzelec, weights=elechistsweep[_xposidx])

        _numion = np.sum(ionhistsweep[_xposidx])
        _numelec = np.sum(elechistsweep[_xposidx])

        _presion = (1./3.)*m_i*np.sum(((vxion-uxionmean)**2+(vyion-uyionmean)**2+(vzion-uzionmean)**2)*ionhistsweep[_xposidx])*dvion**3
        _preselec = (1./3.)*m_e*np.sum(((vxelec-uxelecmean)**2+(vyelec-uyelecmean)**2+(vzelec-uzelecmean)**2)*elechistsweep[_xposidx])*dvelec**3

        _x1 = x_in[_xposidx]-dxin/2.
        _x2 = x_in[_xposidx]+dxin/2.
        iontemp[_xposidx] = (_presion)#/_numion
        electemp[_xposidx] = (_preselec)#/_numelec 

        x_in_temp.append(x_in[_xposidx])

x_in_temp = np.asarray(x_in_temp)

#make pickle of all key energy componnents!!!!!!
enerpick = {}
enerpick['_xcoord_egain'] = _xcoord_egain
enerpick['xcoord_egain'] = xcoord_egain
enerpick['enerCEtot_elec_facfluclocframe_egain'] = enerCEtot_elec_facfluclocframe_egain
enerpick['enerCEtot_elec_facavglocframe_egain'] = enerCEtot_elec_facavglocframe_egain
enerpick['enerCEtot_ion_facfluclocframe_egain'] = enerCEtot_ion_facfluclocframe_egain
enerpick['enerCEtot_ion_facavglocframe_egain'] = enerCEtot_ion_facavglocframe_egain
enerpick['enerCEtot_elec_facavg_egain'] = enerCEtot_elec_fac_egain
enerpick['enerCEtot_elec_facfluc_egain'] = enerCEtot_elec_facfluc_egain
enerpick['enerCEtot_ion_facavg_egain'] = enerCEtot_ion_fac_egain
enerpick['enerCEtot_ion_facfluc_egain'] = enerCEtot_ion_facfluc_egain
enerpick['enerEfield'] = enerEfield
enerpick['enerBfield'] = enerBfield
enerpick['fluxSx'] = fluxSx
enerpick['fluxSy'] = fluxSy
enerpick['fluxSz'] = fluxSz
enerpick['fieldxx'] = dfields['ex_xx'][:]
enerpick['xxKEplot'] = xxKEplot
enerpick['KEions'] = KEions
enerpick['KEelecs'] = KEelecs
enerpick['parEfluxion'] = parEfluxion
enerpick['parEfluxelec'] = parEfluxelec
enerpick['iontemp'] =  iontemp
enerpick['electemp'] = electemp
enerpick['x_in_temp'] = x_in
enerpick['dflow_ion_vx'] = dflowavg['ui'][0,0,:]
enerpick['dflow_elec_vx'] = dflowavg['ue'][0,0,:]
enerpick['dflow_ion_vy'] = dflowavg['vi'][0,0,:]
enerpick['dflow_elec_vy'] = dflowavg['ve'][0,0,:]
enerpick['dflow_ion_vz'] = dflowavg['wi'][0,0,:]
enerpick['dflow_elec_vz'] = dflowavg['we'][0,0,:]
enerpick['dflow_vx_xx'] = dflowavg['ui_xx'][:]

picklefile = 'enerpick.pickle'
with open(picklefile, 'wb') as handle:
    pickle.dump(enerpick, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Debug exitting!")
exit()


#Make list of all quantites for ease of plotting
masterlist_xplot = [x_in,x_in,x_in,x_in,
                    x_in,x_in,x_in,x_in,
                    x_in,x_in,x_in,x_in,
                    x_in,x_in,x_in,x_in,
                    dfavg['ex_xx'],dfavg['bx_xx'],
                    xxKEplot,hx_in,
                    dden['dens_xx'],
                    dfields['ex_xx'],
                    x_in,x_in,
                    x_in,x_in,
                    x_in,x_in,x_in,
                    x_in,x_in,dfields['ex_xx'],
                    dfavg['ex_xx'],dfavg['ex_xx'],
                    xcoord_egain,xcoord_egain,xcoord_egain,xcoord_egain,
                    xcoord_egain,xcoord_egain,xcoord_egain,xcoord_egain,
                    xcoord_egain,xcoord_egain,xcoord_egain,xcoord_egain,
                    xcoord_egain,xcoord_egain,xcoord_egain,xcoord_egain,
                    dden['dens_xx'],
                    x_in,x_in,x_in,x_in,
                    x_in,x_in,x_in,x_in,
                    x_in,x_in,x_in,x_in,
                    x_in,x_in,x_in,x_in,
                    xcoord_egain,xcoord_egain,xcoord_egain,xcoord_egain,
                    xcoord_egain,xcoord_egain,xcoord_egain,xcoord_egain,
                    xcoord_egain,xcoord_egain,xcoord_egain,xcoord_egain,
                    xcoord_egain,xcoord_egain,xcoord_egain,xcoord_egain,
                    x_in,x_in,x_in,x_in,
                    x_in,x_in,x_in,x_in,
                    x_in,x_in,x_in,x_in,
                    x_in,x_in,x_in,x_in,
                    xcoord_egain,xcoord_egain,xcoord_egain,xcoord_egain,
                    xcoord_egain,xcoord_egain,xcoord_egain,xcoord_egain,
                    xcoord_egain,xcoord_egain,xcoord_egain,xcoord_egain,
                    xcoord_egain,xcoord_egain,xcoord_egain,xcoord_egain,
                    x_in,x_in,
                    dden['dens_xx'],
                    x_in,x_in
                    ]

electemp_iontemp = elecmomtemp/ionmomtemp #TODO: replace everywhere below
masterlist_yplot = [enerCEtot_ion_bar,enerCEx_ion_bar,enerCEy_ion_bar,enerCEz_ion_bar,
                    enerCEtot_ion_tilde,enerCEx_ion_tilde,enerCEy_ion_tilde,enerCEz_ion_tilde,
                    enerCEtot_elec_bar,enerCEx_elec_bar,enerCEy_elec_bar,enerCEz_elec_bar,
                    enerCEtot_elec_tilde,enerCEx_elec_tilde,enerCEy_elec_tilde,enerCEz_elec_tilde,
                    enerEfield,enerBfield,
                    KEions,ionTemp,
                    elecTemp,
                    Teperpadia,
                    bulkfloweion, ioninternale,
                    bulkfloweelec,elecinternale,
                    ionmomtemp,elecmomtemp,electemp_iontemp,
                    ionmomtempperp,elecmomtempperp,Tstasiewicz,
                    stoc_param_i,stoc_param_e,
                    enerCEtot_ion_bar_egain,enerCEx_ion_bar_egain,enerCEy_ion_bar_egain,enerCEz_ion_bar_egain,
                    enerCEtot_ion_tilde_egain,enerCEx_ion_tilde_egain,enerCEy_ion_tilde_egain,enerCEz_ion_tilde_egain,
                    enerCEtot_elec_bar_egain,enerCEx_elec_bar_egain,enerCEy_elec_bar_egain,enerCEz_elec_bar_egain,
                    enerCEtot_elec_tilde_egain,enerCEx_elec_tilde_egain,enerCEy_elec_tilde_egain,enerCEz_elec_tilde_egain,
                    Te_ad_tran,
                    enerCEtot_ion_facfluc,enerCEx_ion_facfluc,enerCEy_ion_facfluc,enerCEz_ion_facfluc, 
                    enerCEtot_ion_facfluclocal,enerCEx_ion_facfluclocal,enerCEy_ion_facfluclocal,enerCEz_ion_facfluclocal,
                    enerCEtot_elec_facfluc,enerCEx_elec_facfluc,enerCEy_elec_facfluc,enerCEz_elec_facfluc,
                    enerCEtot_elec_facfluclocal,enerCEx_elec_facfluclocal,enerCEy_elec_facfluclocal,enerCEz_elec_facfluclocal,
                    enerCEtot_ion_facfluc_egain,enerCEx_ion_facfluc_egain,enerCEy_ion_facfluc_egain,enerCEz_ion_facfluc_egain,
                    enerCEtot_ion_facfluclocal_egain,enerCEx_ion_facfluclocal_egain,enerCEy_ion_facfluclocal_egain,enerCEz_ion_facfluclocal_egain,
                    enerCEtot_elec_facfluc_egain,enerCEx_elec_facfluc_egain,enerCEy_elec_facfluc_egain,enerCEz_elec_facfluc_egain,
                    enerCEtot_elec_facfluclocal_egain,enerCEx_elec_facfluclocal_egain,enerCEy_elec_facfluclocal_egain,enerCEz_elec_facfluclocal_egain,
                    enerCEtot_ion_facfluclowpass,enerCEx_ion_facfluclowpass,enerCEy_ion_facfluclowpass,enerCEz_ion_facfluclowpass,
                    enerCEtot_ion_facfluchighdetrend,enerCEx_ion_facfluchighdetrend,enerCEy_ion_facfluchighdetrend,enerCEz_ion_facfluchighdetrend,
                    enerCEtot_elec_facfluclowpass,enerCEx_elec_facfluclowpass,enerCEy_elec_facfluclowpass,enerCEz_elec_facfluclowpass,
                    enerCEtot_elec_facfluchighdetrend,enerCEx_elec_facfluchighdetrend,enerCEy_elec_facfluchighdetrend,enerCEz_elec_facfluchighdetrend,
                    enerCEtot_ion_facfluclowpass_egain,enerCEx_ion_facfluclowpass_egain,enerCEy_ion_facfluclowpass_egain,enerCEz_ion_facfluclowpass_egain,
                    enerCEtot_ion_facfluchighdetrend_egain,enerCEx_ion_facfluchighdetrend_egain,enerCEy_ion_facfluchighdetrend_egain,enerCEz_ion_facfluchighdetrend_egain,
                    enerCEtot_elec_facfluclowpass_egain,enerCEx_elec_facfluclowpass_egain,enerCEy_elec_facfluclowpass_egain,enerCEz_elec_facfluclowpass_egain,
                    enerCEtot_elec_facfluchighdetrend_egain,enerCEx_elec_facfluchighdetrend_egain,enerCEy_elec_facfluchighdetrend_egain,enerCEz_elec_facfluchighdetrend_egain,
                    ionmomtemppar,elecmomtemppar,
                    T_ad_gamma_fivethirds,
                    elecden,ionden
                    ] #TODO: !!!! Rename CEx_ion_facfluc to CEpar, y to perp1, z to perp2

masterlist_labels = [r'$\int \overline{C}_{\mathbf{E},i} \, d^3\mathbf{v}$',r'$\int \overline{C}_{E_x,i} \, d^3\mathbf{v}$',r'$\int \overline{C}_{E_y,i} \, d^3\mathbf{v}$',r'$\int \overline{C}_{E_z,i} \, d^3\mathbf{v}$',
        r'$\int \widetilde{C}_{\mathbf{E},i} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E_x,i} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E_y,i} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E_z,i} \, d^3\mathbf{v}$',
        r'$\int \overline{C}_{\mathbf{E},e} \, d^3\mathbf{v}$',r'$\int \overline{C}_{E_x,e} \, d^3\mathbf{v}$',r'$\int \overline{C}_{E_y,e} \, d^3\mathbf{v}$',r'$\int \overline{C}_{E_z,e} \, d^3\mathbf{v}$',
        r'$\int \widetilde{C}_{\mathbf{E},e} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E_x,e} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E_y,e} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E_z,e} \, d^3\mathbf{v}$',
        r'$U_{E}$',r'$U_{B}$',
        r"$E_{K,i}$",r'$E_{T,i}$',
        r'$E_{T,e,ideal,adia,\gamma=5/3}$',
        r'$E_{T,e,invariant}$',
        r'$E_{flow,i}$',r'$E_{internal,i}$',
        r'$E_{flow,e}$',r'$E_{internal,e}$',
        r'$T_i$',r'$T_e$',r'$T_e/T_i$',
        r'$T_{\perp,i}/B$',r'$T_{\perp,e}/B$',r'$T_{e,Stasiewicz}/B$',
        r'$\chi_i$',r'$\chi_e$',
        r'$W_{\mathbf{j}_i \cdot \overline{\mathbf{E}}}$',r'$W_{\mathbf{j}_{x,i} \cdot \overline{E_x}}$',r'$W_{\mathbf{j}_{y,i} \cdot \overline{E_y}}$',r'$W_{\mathbf{j}_{z,i} \cdot \overline{E_z}}$',
        r'$W_{\mathbf{j}_i \cdot \widetilde{\mathbf{E}}}$',r'$W_{\mathbf{j}_{x,i} \cdot \widetilde{E_x}}$',r'$W_{\mathbf{j}_{y,i} \cdot \widetilde{E_y}}$',r'$W_{\mathbf{j}_{z,i} \cdot \widetilde{E_z}}$',
        r'$W_{\mathbf{j}_e \cdot \overline{\mathbf{E}}}$',r'$W_{\mathbf{j}_{x,e} \cdot \overline{E_x}}$',r'$W_{\mathbf{j}_{y,e} \cdot \overline{E_y}}$',r'$W_{\mathbf{j}_{z,e} \cdot \overline{E_z}}$',
        r'$W_{\mathbf{j}_e \cdot \widetilde{\mathbf{E}}}$',r'$W_{\mathbf{j}_{x,e} \cdot \widetilde{E_x}}$',r'$W_{\mathbf{j}_{y,e} \cdot \widetilde{E_y}}$',r'$W_{\mathbf{j}_{z,e} \cdot \widetilde{E_z}}$',
        r'$T_{adia,Tran}$',
        r'$\int \overline{C}_{\mathbf{E},i} \, d^3\mathbf{v}$',r'$\int \overline{C}_{E_{||},i} \, d^3\mathbf{v}$',r'$\int \overline{C}_{E_{\perp,1},i} \, d^3\mathbf{v}$',r'$\int \overline{C}_{E_{\perp,2},i} \, d^3\mathbf{v}$',
        r'$\int \widetilde{C}_{\mathbf{E},i} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{||},i} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{\perp,1},i} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{\perp,2},i} \, d^3\mathbf{v}$',
        r'$\int \overline{C}_{\mathbf{E},e} \, d^3\mathbf{v}$',r'$\int \overline{C}_{E_{||},e} \, d^3\mathbf{v}$',r'$\int \overline{C}_{E_{\perp,1},e} \, d^3\mathbf{v}$',r'$\int \overline{C}_{E_{\perp,2},e} \, d^3\mathbf{v}$',
        r'$\int \widetilde{C}_{\mathbf{E},e} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{||},e} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{\perp,1},e} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{\perp,2},e} \, d^3\mathbf{v}$',
        r'$W_{\mathbf{j}_i \cdot \overline{\mathbf{E}}}$',r'$W_{\mathbf{j}_{||,i} \cdot \overline{E_{||}}}$',r'$W_{\mathbf{j}_{\perp,1,i} \cdot \overline{E_{\perp,1,i}}}$',r'$W_{\mathbf{j}_{\perp,2,i} \cdot \overline{E_{perp,2}}}$',
        r'$W_{\mathbf{j}_i \cdot \widetilde{\mathbf{E}}}$',r'$W_{\mathbf{j}^{local}_{||,i} \cdot \widetilde{E^{local}_{||}}}$',r'$W_{\mathbf{j}^{local}_{\perp,1,i} \cdot \widetilde{E^{local}_{\perp,1}}}$',r'$W_{\mathbf{j}^{local}_{\perp,2,i} \cdot \widetilde{E^{local}_{\perp,2,i}}}$',
        r'$W_{\mathbf{j}_e \cdot \widetilde{\mathbf{E}}}$',r'$W_{\mathbf{j}_{||,e} \cdot \widetilde{E_{||}}}$',r'$W_{\mathbf{j}_{\perp,1,e} \cdot \widetilde{E_{\perp,1}}}$',r'$W_{\mathbf{j}_{\perp,2,e} \cdot \widetilde{E_{\perp,2}}}$',
        r'$W_{\mathbf{j}_e \cdot \widetilde{\mathbf{E}}}$',r'$W_{\mathbf{j}^{local}_{||,e} \cdot \widetilde{E^{local}_{||}}}$',r'$W_{\mathbf{j}^{local}_{\perp,1,e} \cdot \widetilde{E^{local}_{\perp,1}}}$',r'$W_{\mathbf{j}^{local}_{\perp,2,e} \cdot \widetilde{E^{local}_{\perp,2}}}$',
        r'$\int \widetilde{C}_{\mathbf{E},<k_0,i} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E_{||},<k_0,i} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{\perp,1},<k_0,i} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{\perp,2},<k_0,i} \, d^3\mathbf{v}$',
        r'$\int \widetilde{C}_{\mathbf{E},detrend,i} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E_{||},detrend,i} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{\perp,1},detrend,i} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{\perp,2},detrend,i} \, d^3\mathbf{v}$',
        r'$\int \widetilde{C}_{\mathbf{E},<k_0,e} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E_{||},<k_0,e} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{\perp,1},<k_0,e} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{\perp,2},<k_0,e} \, d^3\mathbf{v}$',
        r'$\int \widetilde{C}_{\mathbf{E},detrend,e} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E_{||},detrend,e} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{\perp,1},detrend,e} \, d^3\mathbf{v}$',r'$\int \widetilde{C}_{E^{local}_{\perp,2},detrend,e} \, d^3\mathbf{v}$',
        r'$W_{\mathbf{j}_i \cdot \widetilde{\mathbf{E_{<k_0}}}}$',r'$W_{\mathbf{j}_{||,i} \cdot \widetilde{E_{||,<k_0}}}$',r'$W_{\mathbf{j}_{\perp,1,i} \cdot \widetilde{E_{\perp,1,<k_0}}}$',r'$W_{\mathbf{j}_{\perp,2,i} \cdot \widetilde{E_{\perp,2,<k_0}}}$',
        r'$W_{\mathbf{j}_i \cdot \widetilde{\mathbf{E_{detrend}}}}$',r'$W_{\mathbf{j}_{||,e} \cdot \widetilde{E_{||,detrend}}}$',r'$W_{\mathbf{j}_{\perp,1,e} \cdot \widetilde{E_{\perp,1,detrend}}}$',r'$W_{\mathbf{j}_{\perp,2,e} \cdot \widetilde{E_{\perp,2,detrend}}}$',
         r'$W_{\mathbf{j}_e \cdot \widetilde{\mathbf{E_{<k_0}}}}$',r'$W_{\mathbf{j}_{||,e} \cdot \widetilde{E_{||,<k_0}}}$',r'$W_{\mathbf{j}_{\perp,1,e} \cdot \widetilde{E_{\perp,1,<k_0}}}$',r'$W_{\mathbf{j}_{\perp,2,e} \cdot \widetilde{E_{\perp,2,<k_0}}}$',
        r'$W_{\mathbf{j}_e \cdot \widetilde{\mathbf{E_{detrend}}}}$',r'$W_{\mathbf{j}_{||,e} \cdot \widetilde{E_{||,detrend}}}$',r'$W_{\mathbf{j}_{\perp,1,e} \cdot \widetilde{E_{\perp,1,detrend}}}$',r'$W_{\mathbf{j}_{\perp,2,e} \cdot \widetilde{E_{\perp,2,detrend}}}$',
        r'$T_{||,i}/B$',r'$T_{||,e}/B$',
        r'$T_{e,ad,\Gamma=5/3}$',
        r'$n_e$',r'$n_i$'
        ]

if(True): print("Debug: len(masterlist_xplot),len(masterlist_yplot),len(masterlist_labels)", len(masterlist_xplot),len(masterlist_yplot),len(masterlist_labels))

print("TODO: CONSIDER SIGN IN J dot E egain VARS (sign of charge is missing?)")
print("TODO: RENAME VARS SO THAT AUTOMATIC FILE NAMING IS NOT CONFUSING (x -> par y-> perp1 z-> perp2)")

print("TODO: compute Tpar and Tperrp vs x for electrons")

print("Making APS DPP FIGURE")
#TEMP FIGURE FOR APS DPP
#we compare elecmomtemp to 'adiabatic' electrons (via Aaron Tran's defintion)
#y var names are Te_ad_tran and elecmomtemp
#x var names are dden['dens_xx'] and x_in

Te_ad_tran_norm_up_idx = ao.find_nearest(dden['dens_xx'],11) #pick upstream val for normalization (11 b/c this value is 0 if we are too far upstream (for computational resource purposes, there are no particles in the far-far upstream momentarily)
elecmomtemp_norm_up_idx = ao.find_nearest(x_in,11) #pick upstream val for normalization
Te_ad_tran_norm = Te_ad_tran/Te_ad_tran[Te_ad_tran_norm_up_idx]
elecmomtemp_norm = elecmomtemp/elecmomtemp[elecmomtemp_norm_up_idx]
plt.figure(figsize=(13,5))
plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
plt.plot(dden['dens_xx'],Te_ad_tran_norm,color='red',ls='-.',label='$T_{e,adia,\Gamma=2}/T_{e,up}$')
plt.plot(x_in,elecmomtemp_norm,color='black',ls='-',label='$T_{e,mom}/T_{e,up}$')
plt.xlabel("$x/d_i$")
plt.xlim(5,12.5)
plt.legend()
plt.grid()
plt.savefig('figures/momtempvsadiabatictemp_gamma_2.png',format='png',bbox_inches='tight',dpi=300)
plt.close()

T_ad_gamma_fivethirds_idx = ao.find_nearest(dden['dens_xx'],11) #pick upstream val for normalization
elecmomtemp_norm_up_idx = ao.find_nearest(x_in,11) #pick upstream val for normalization
T_ad_gamma_fivethirds_norm = T_ad_gamma_fivethirds/T_ad_gamma_fivethirds[T_ad_gamma_fivethirds_idx]
elecmomtemp_norm = elecmomtemp/elecmomtemp[elecmomtemp_norm_up_idx]
plt.figure(figsize=(13,5))
plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
plt.plot(dden['dens_xx'],T_ad_gamma_fivethirds_norm,color='red',ls='-.',label='$T_{e,adia,\Gamma=5/3}/T_{e,up}$')
plt.plot(x_in,elecmomtemp_norm,color='black',ls='-',label='$T_{e,mom}/T_{e,up}$')
plt.xlabel("$x/d_i$")
plt.xlim(5,12.5)
plt.legend()
plt.grid()
plt.savefig('figures/momtempvsadiabatictemp_gamma5_3.png',format='png',bbox_inches='tight',dpi=300)
plt.close()


#plot things on individual plots
def plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels):
    plt.figure(figsize=(20,5))

    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

    if(len(listindexes) == 4):
        colors = ['black','red','green','blue']
        ls = ['-',':','--','-.']
    elif(len(listindexes) == 2):
        colors = ['gray','purple']
        ls = ['-','--']
    else:
        colors = ['black','red','green','blue','black','red','green','blue','black','red','green','blue']
        ls = ['-',':','--','-.','-',':','--','-.','-',':','--','-.']

    _counter = 0
    for _idx in listindexes:
        plt.plot(masterlist_xplot[_idx],masterlist_yplot[_idx],color=colors[_counter],ls=ls[_counter],label=masterlist_labels[_idx], linewidth=5)
        _counter += 1

    plt.xlabel(r"$x/d_{i,0}$", fontsize=32)
    plt.xlim(0,12)
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(0,12,1))
    plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')    
    plt.close()

import os
os.system('mkdir figures')
os.system('mkdir figures/enervsx')

flnmprefix = 'figures/enervsx/'

#Plot everything 1 by 1
import inspect
def get_variable_name(var_value): #warning: this assumes that every variable is unique. If there are two variables with the same value, this will return one of them at 'random'
    frame = inspect.currentframe().f_back
    for var_name, var_value_in_scope in frame.f_locals.items():
        if var_value_in_scope is var_value:
            return var_name
    return None

os.system('mkdir figures/enervsx/individuals') #TODO: don't hard code 'enervsx' here and other system mkdir statements
for _i in range(0,len(masterlist_yplot)):
    listindexes = [_i]
    plotvarname = get_variable_name(masterlist_yplot[_i])

    print("DEBUG: FOUND NAME OF ", plotvarname," at index ", _i)
    flnm = flnmprefix+'/individuals/'+plotvarname+'.png'
    plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#CEi_ion_bar
listindexes = [0,1,2,3]
flnm = flnmprefix+'CEi_ion_bar.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#CEi_ion_tilde
listindexes = [4,5,6,7]
flnm = flnmprefix+'CEi_ion_tilde.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#CEi_elec_bar
listindexes = [8,9,10,11]
flnm = flnmprefix+'CEi_elec_bar.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#CEi_elec_tilde
listindexes = [12,13,14,15]
flnm = flnmprefix+'CEi_elec_tilde.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#Ener E vs Ener B
listindexes = [16,17]
flnm = flnmprefix+'enerEenerB.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#egain CEi_ion_bar
listindexes = [34,35,36,37]
flnm = flnmprefix+'egainCEi_ion_bar.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#egain CEi_ion_tilde
listindexes = [38,39,40,41]
flnm = flnmprefix+'egainCEi_ion_tilde.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#egain CEi_elec_bar
listindexes = [42,43,44,45]
flnm = flnmprefix+'egainCEi_elec_bar.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#egain CEi_elec_tilde
listindexes = [46,47,48,49]
flnm = flnmprefix+'egainCEi_elec_tilde.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#CEi_ion_facfluc
listindexes = [51,52,53,54]
flnm = flnmprefix+'CEi_ion_facfluc.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#CEi_ion_facfluclocal
listindexes = [55,56,57,58]
flnm = flnmprefix+'CEi_ion_facfluclocal.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#CEi_elec_facfluc
listindexes = [59,60,61,62]
flnm = flnmprefix+'CEi_elec_facfluc.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#CEi_elec_tilde
listindexes = [63,64,65,66]
flnm = flnmprefix+'CEi_elec_facfluclocal.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#egain CEi_ion_facfluc
listindexes = [67,68,69,70]
flnm = flnmprefix+'egainCEi_ion_facfluc.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#egain CEi_ion_facfluclocal
listindexes = [71,72,73,74]
flnm = flnmprefix+'egainCEi_ion_facfluclocal.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#egain CEi_elec_facfluc
listindexes = [75,76,77,78]
flnm = flnmprefix+'egainCEi_elec_facfluc.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#egain CEi_elec_tilde
listindexes = [79,80,81,82]
flnm = flnmprefix+'egainCEi_elec_facfluclocal.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#egainn CEi_ion_lowpass
listindexes = [99,100,101,102]
flnm = flnmprefix+'egainCEi_ion_lowpass.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#egain CEi_ion_detrend
listindexes = [103,104,105,106]
flnm = flnmprefix+'engainCEi_ion_detrend.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#egain CEi_elec_lowpass
listindexes = [107,108,109,110]
flnm = flnmprefix+'engainCEi_elec_lowpass.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#egain CEi_elec_detrend
listindexes = [111,112,113,114]
flnm = flnmprefix+'engainCEi_elec_detrend.png'
plot_from_list(flnm, listindexes, masterlist_xplot, masterlist_yplot, masterlist_labels)

#plot things on same plot
plt.rcParams.update({})
plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
fig, axs = plt.subplots(11,figsize=(20,55),sharex="all")
plt.subplots_adjust(wspace=0, hspace=0.1)
_lw = 5
axs[0].plot(x_in,enerCEtot_ion_bar,label=r'$\int \overline{C}_{\mathbf{E},i} \, d^3\mathbf{v}$',c='black',ls='-', linewidth=_lw)
axs[0].plot(x_in,enerCEx_ion_bar,label=r'$\int \overline{C}_{E_x,i} \, d^3\mathbf{v}$',c='red',ls=':', linewidth=_lw)
axs[0].plot(x_in,enerCEy_ion_bar,label=r'$\int \overline{C}_{E_y,i} \, d^3\mathbf{v}$',c='green',ls='--', linewidth=_lw)
axs[0].plot(x_in,enerCEz_ion_bar,label=r'$\int \overline{C}_{E_z,i} \, d^3\mathbf{v}$',c='blue',ls='-.', linewidth=_lw)

axs[1].plot(x_in,enerCEtot_ion_tilde,label=r'$\int \widetilde{C}_{\mathbf{E},i} \, d^3\mathbf{v}$',c='black',ls='-', linewidth=_lw)
axs[1].plot(x_in,enerCEx_ion_tilde,label=r'$\int \widetilde{C}_{E_x,i} \, d^3\mathbf{v}$',c='red',ls=':', linewidth=_lw)
axs[1].plot(x_in,enerCEy_ion_tilde,label=r'$\int \widetilde{C}_{E_y,i} \, d^3\mathbf{v}$',c='green',ls='--', linewidth=_lw)
axs[1].plot(x_in,enerCEz_ion_tilde,label=r'$\int \widetilde{C}_{E_z,i} \, d^3\mathbf{v}$',c='blue',ls='-.', linewidth=_lw)

axs[2].plot(x_in,enerCEtot_elec_bar,label=r'$\int \overline{C}_{\mathbf{E},e} \, d^3\mathbf{v}$',c='black',ls='-', linewidth=_lw)
axs[2].plot(x_in,enerCEx_elec_bar,label=r'$\int \overline{C}_{E_x,e} \, d^3\mathbf{v}$',c='red',ls=':', linewidth=_lw)
axs[2].plot(x_in,enerCEy_elec_bar,label=r'$\int \overline{C}_{E_y,e} \, d^3\mathbf{v}$',c='green',ls='--', linewidth=_lw)
axs[2].plot(x_in,enerCEz_elec_bar,label=r'$\int \overline{C}_{E_z,e} \, d^3\mathbf{v}$',c='blue',ls='-.', linewidth=_lw)

axs[3].plot(x_in,enerCEtot_elec_tilde,label=r'$\int \widetilde{C}_{\mathbf{E},e} \, d^3\mathbf{v}$',c='black',ls='-', linewidth=_lw)
axs[3].plot(x_in,enerCEx_elec_tilde,label=r'$\int \widetilde{C}_{E_x,e} \, d^3\mathbf{v}$',c='red',ls=':', linewidth=_lw)
axs[3].plot(x_in,enerCEy_elec_tilde,label=r'$\int \widetilde{C}_{E_y,e} \, d^3\mathbf{v}$',c='green',ls='--', linewidth=_lw)
axs[3].plot(x_in,enerCEz_elec_tilde,label=r'$\int \widetilde{C}_{E_z,e} \, d^3\mathbf{v}$',c='blue',ls='-.', linewidth=_lw)

axs[4].plot(dfavg['ex_xx'],enerEfield,label=r'$U_{E}$',color='gray', linewidth=_lw,ls='-')
axs[4].plot(dfavg['bx_xx'],enerBfield,label=r'$U_{B}$',color='blue', linewidth=_lw,ls='--')

print("TODO: compute ionTemp correctly")
axs[5].plot(xxKEplot,KEions,label=r"$E_{K,i}$",color='black', linewidth=_lw,ls='-')
#axs[5].plot(hx_in,ionTemp,label=r'$E_{T,i}$',color='green', linewidth=_lw,ls='--')
axs[5].plot(dden['dens_xx'],elecTemp,label=r'$E_{T,e,adiabatic}$',color='purple',linewidth=_lw,ls='-.')

axs[6].plot(dfields['ex_xx'],Teperpadia,label=r'$E_{T,e,invariant}$',c='black',ls='-', linewidth=_lw)

axs[7].plot(x_in,bulkfloweion,label=r'$E_{flow,i}$',c='black',ls='-', linewidth=_lw)
axs[7].plot(x_in,ioninternale,label=r'$E_{internal,i}$',c='black',ls='--', linewidth=_lw)

axs[8].plot(x_in,bulkfloweelec,label=r'$E_{flow,e}$',c='red',ls='-', linewidth=_lw)
axs[8].plot(x_in,elecinternale,label=r'$E_{internal,e}$',c='red',ls='--', linewidth=_lw)

axs[9].plot(x_in,ionmomtemp,label=r'$T_i$',c='black',ls='-', linewidth=_lw)
axs[9].plot(x_in,elecmomtemp,label=r'$T_e$',c='red',ls='-',linewidth=_lw)
#TODO: work on divide by zero on below line
axs[9].plot(x_in,elecmomtemp/ionmomtemp,label=r'$T_e/T_i$',c='blue',ls=':',linewidth=_lw)

axs[10].plot(x_in,ionmomtempperp,label=r'$T_{\perp,i}/B$',c='black',ls='-.', linewidth=_lw)
axs[10].plot(x_in,elecmomtempperp,label=r'$T_{\perp,e}/B$',c='red',ls='-.',linewidth=_lw)
axs[10].plot(dfields['ex_xx'],Tstasiewicz,label=r'$T_{e,Stasiewicz}/B$',color='red',linewidth=_lw,ls=':')

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=25)
    # _xposshock = 39.875
    # ax.axvline(_xposshock,color='black')
    ax.grid()
    #ax.set_xlim(35,50)
    ax.legend()

axs[0].legend(ncol=2,loc='lower left', prop={'size': 30})
axs[1].legend(ncol=2,loc='lower left', prop={'size': 30})
axs[2].legend(ncol=2,loc='lower left', prop={'size': 30})

plt.savefig('figures/oldenervsx.png',format='png',dpi=300)
plt.close()
