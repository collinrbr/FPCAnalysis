from FPCAnalysis import *

import sys
import os
import math
import numpy as np

if __name__ == '__main__':
    try:
        analysisinputflnm = sys.argv[1]
    except:
        print("This generates FPC netcdf4 file from Tristan data.")
        print("usage: " + sys.argv[0] + " analysisinputflnm num_threads(default 1) dpar_folder use_dfluc(default F)")
        print("Warning: Frame num expects leading zeros. This script pads to have 3 digits (i.e. 1 or two leading zeros) but may need to be modified if more are expected.")
        sys.exit()

    try:
        num_threads = int(sys.argv[2])
        if(num_threads <= 0):
            print("Error, please request at least 1 thread...")
    except:
        num_threads = 1

    try:
        dpar_folder = sys.argv[3]+'/'
    except:
        dpar_folder = None

    dpar_folder_elec = dpar_folder+'/elec/'
    dpar_folder_ion = dpar_folder+'/ion/'

    try:
        use_dfluc = sys.argv[4]
        if(use_dfluc == 'T'):
            use_dfluc = True
        else:
            use_dfluc = False
    except:
        use_dfluc = False

    if(num_threads != 1 and dpar_folder == None):
        print("Error, multithreading now expects pre-slicing of the data.")
        print("Please use preslicedataTristan.py, and pass output folder to dpar_folder")
        exit()


    anlinput = anl.analysis_input(flnm = analysisinputflnm)
    path = anlinput['path']
    num = anlinput['numframe']
    dv = anlinput['dv']
    vmax = anlinput['vmax']
    dx = anlinput['dx']
    xlim = anlinput['xlim']
    ylim = anlinput['ylim']
    zlim = anlinput['zlim']
    resultsdir = anlinput['resultsdir']

    #add leading zeros (TODO: go to folder and figure out the number of leading zeros automatically)
    _zfilllen = 3
    num = str(num).zfill(_zfilllen)

    # if(useFAC):
    #     print('Using FAC while sweeping is still a work in progress....')
    #     print("TODO: netcdf4 that can handle FAC")
    #     print("TODO: add FAC to sweep function")
    #     print("TODO: generate 9 pan sweep that can handle FAC netcdf4 file")

    #-------------------------------------------------------------------------------
    # load data
    #-------------------------------------------------------------------------------
    #load path
    print("Loading data...")
    inputpath = path+'/input'
    path = path+'/output/'
    inputs = dtr.load_input(inputpath)

    params = dtr.load_params(path,num)
    dt = params['c']/params['comp'] #in units of wpe
    dt,c = anl.norm_constants_tristanmp1(params,dt,inputs) #in units of wci (\omega_ci) and va respectively- as required by the rest of the scripts

    beta0 = anl.compute_beta0_tristanmp1(params,inputs)
    betai,betae= anl.get_betai_betae_from_tot_and_ratio(beta0,params['temperature_ratio'])

    dfields = dtr.load_fields(path,num,normalizeFields=True)

    if(num_threads == 1):
        dpar_elec, dpar_ion = dtr.load_particles(path,num,normalizeVelocity=True)

    #Computes relevant parameters to save later
    _dfields = dtr.load_fields(path, num)
    _dfields = anl.get_average_fields_over_yz(dfields)
    params['thetaBn'] = np.arctan(_dfields['by'][0,0,-1]/_dfields['bx'][0,0,-1])* 180.0/np.pi

    uinj_voverc = inputs['gamma0']
    uinj_vth = uinj_voverc/np.sqrt(params['delgam'])
    params['MachAlfven'] =  uinj_vth*np.sqrt(betai)*params['c'] #assumes me << mi

    #-------------------------------------------------------------------------------
    # estimate shock vel and lorentz transform
    #-------------------------------------------------------------------------------
    print("Calculating vshock...")

    #compute shock velocity and boost to shock rest frame
    frames = [int(num)-_i for _i in [3,2,1,0]]
    frames = [str(_val).zfill(_zfilllen) for _val in frames]
    dfields_many_frames = {'frame':[],'dfields':[]}
    for _num in frames:
        num = int(_num)
        d = dtr.load_fields(path,_num,normalizeFields=True)
        dfields_many_frames['dfields'].append(d)
        dfields_many_frames['frame'].append(num)
    vshock,_  = ft.shock_from_ex_cross(dfields_many_frames,dt)
    print("Lorentz transforming fields...")
    dfields = ft.lorentz_transform_vx_c(dfields,vshock,c) #note: we only boost one frame

    if(num_threads == 1):
        dpar_ion = ft.shift_particles_tristan(dpar_ion, vshock, betai, betae, params['mi']/params['me'], isIon=True)
        dpar_elec = ft.shift_particles_tristan(dpar_elec, vshock, betai, betae, params['mi']/params['me'], isIon=False)
        dpar_ion = dtr.format_par_like_dHybridR(dpar_ion) #For now, we rename the particle data keys too look like the keys we used when processing dHybridR data so this data is compatible with our old routines
        dpar_elec = dtr.format_par_like_dHybridR(dpar_elec)

    if(use_dfluc):
        dfields = anl.remove_average_fields_over_yz(dfields)

    #-------------------------------------------------------------------------------
    # do FPC analysis for ions and project output
    #-------------------------------------------------------------------------------
    print("Doing FPC analysis for each slice of x for ions...")

    if(num_threads == 1):
        CEx, CEy, CEz, x, Hist, vx, vy, vz, num_par = fpc.compute_correlation_over_x(dfields, dpar_ion, vmax, dv, dx, vshock, xlim, ylim, zlim)
        Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz = fpc.project_CEi_hist(Hist, CEx, CEy, CEz)
    else:
        CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz,x,Histxy,Histxz,Histyz, vx, vy, vz, num_par = fpc.comp_cor_over_x_multithread(dfields, dpar_folder_ion, vmax, dv, dx, vshock, xlim=xlim, ylim=ylim, zlim=zlim, max_workers=num_threads, betai=betai, betae=betae, mi_me=params['mi']/params['me'], isIon=True)

    #-------------------------------------------------------------------------------
    # compute energization
    #-------------------------------------------------------------------------------

    #compute energization from correlations
    enerCEx = anl.compute_energization_over_x(CExxy,dv)
    enerCEy = anl.compute_energization_over_x(CEyxy,dv)
    enerCEz = anl.compute_energization_over_x(CEzxy,dv)

    #-------------------------------------------------------------------------------
    # Save data with relevant input parameters
    #-------------------------------------------------------------------------------
    print("Saving ion results in netcdf4 file...")
    flnm = resultsdir+path.replace("/", "_")+'ion_FPCnometadata'
    if(use_dfluc):
        flnm = flnm+'dfluc'
    flnm = flnm.replace('~','-')
    dnc.save2Vdata(Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, dfields['Vframe_relative_to_sim'], params = params, num_par = num_par, filename = flnm+'_2v.nc')
    print('Done with ion results!')

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #REPEAT FOR ELECTRONS
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("Doing FPC analysis for each slice of x for electrons...")
    if(num_threads == 1):
        CEx, CEy, CEz, x, Hist, vx, vy, vz, num_par = fpc.compute_correlation_over_x(dfields, dpar_elec, vmax, dv, dx, vshock, xlim, ylim, zlim)
        Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz = fpc.project_CEi_hist(Hist, CEx, CEy, CEz)
    else:
        CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz,x, Histxy,Histxz,Histyz, vx, vy, vz, num_par = fpc.comp_cor_over_x_multithread(dfields, dpar_folder_elec, vmax, dv, dx, vshock, xlim=xlim, ylim=ylim, zlim=zlim, max_workers=num_threads, betai=betai, betae=betae, mi_me=params['mi']/params['me'], isIon=True)

    #-------------------------------------------------------------------------------
    # compute energization
    #-------------------------------------------------------------------------------

    #compute energization from correlations
    enerCEx = anl.compute_energization_over_x(CExxy,dv)
    enerCEy = anl.compute_energization_over_x(CEyxy,dv)
    enerCEz = anl.compute_energization_over_x(CEzxy,dv)

    #-------------------------------------------------------------------------------
    # Save data with relevant input parameters
    #-------------------------------------------------------------------------------
    print("Saving elec results in netcdf4 file...")

    flnm = resultsdir+path.replace("/", "_")+'elec_FPCnometadata'
    if(use_dfluc):
        flnm += 'dfluc'
    flnm = flnm.replace('~','-')
    dnc.save2Vdata(Histxy,Histxz,Histyz,CExxy,CExxz,CExyz,CEyxy,CEyxz,CEyyz,CEzxy,CEzxz,CEzyz, vx, vy, vz, x, enerCEx, enerCEy, enerCEz, dfields['Vframe_relative_to_sim'], params = params, num_par = num_par, filename = flnm+'_2v.nc')
    print('Done!!!')
