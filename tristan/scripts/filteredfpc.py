import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa
import lib.plotcoraux as pfpc
import lib.fpcaux as fpc
import lib.arrayaux as ao
#TODO: either import all libraries in all scripts, or only necessary ones- just be consistent

import os
os.system('mkdir figures')
os.system('mkdir figures/ion')
os.system('mkdir figures/elec')

#user params
framenum = '700' #frame to make figure of (should be a string)
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]
vmaxion = 30
dvion = 1.
vmaxelec = 15
dvelec = 1.
vrmaxion = vmaxion
vrmaxelec = vmaxelec
nrbins = 10

normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)
dden = ld.load_den(flpath,framenum)
for _key in dden.keys():
    dfields[_key] = dden[_key]

params = ld.load_params(flpath,framenum)
dt = params['c']/params['comp'] #in units of wpe
c = params['c']
stride = 100
dt,c = aa.norm_constants(params,dt,c,stride)

#compute shock velocity and boost to shock rest frame
#dfields_many_frames = {'frame':[],'dfields':[]}
#for _num in frames:
#    num = int(_num)
#    d = ld.load_fields(flpath,_num,normalizeFields=normalize)
#    dfields_many_frames['dfields'].append(d)
#    dfields_many_frames['frame'].append(num)
#vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)
print("Warning: Using manual value for vshock to save time...")
vshock = 1.56
dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

dfavg = aa.get_average_fields_over_yz(dfields)
dfluc = aa.remove_average_fields_over_yz(dfields)

dpar_elec, dpar_ion = ld.load_particles(flpath,framenum,normalizeVelocity=normalize)
inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
inputs = ld.load_input(inputpath)
beta0 = aa.compute_beta0(params,inputs)
dpar_ion = ft.shift_particles(dpar_ion, vshock, beta0, params['mi']/params['me'], isIon=True)
dpar_elec = ft.shift_particles(dpar_elec, vshock, beta0, params['mi']/params['me'], isIon=False)


def plot_corr(vx,vy,vz,vmax,hist,corex,corey,corez,flnm,plotFAC=False,plotAvg=False,plotFluc=False,isIon=True,isLowPass=False,isHighPass=False):
    from lib.arrayaux import array_3d_to_2d

    H_xy = array_3d_to_2d(hist, 'xy')
    H_xz = array_3d_to_2d(hist, 'xz')
    H_yz = array_3d_to_2d(hist, 'yz')

    CEx_xy = array_3d_to_2d(corex, 'xy')
    CEx_xz = array_3d_to_2d(corex, 'xz')
    CEx_yz = array_3d_to_2d(corex, 'yz')

    CEy_xy = array_3d_to_2d(corey, 'xy')
    CEy_xz = array_3d_to_2d(corey, 'xz')
    CEy_yz = array_3d_to_2d(corey, 'yz')

    CEz_xy = array_3d_to_2d(corez, 'xy')
    CEz_xz = array_3d_to_2d(corez, 'xz')
    CEz_yz = array_3d_to_2d(corez, 'yz')

    pfpc.plot_cor_and_dist_supergrid(vx, vy, vz, vmax,
                                H_xy, H_xz, H_yz,
                                CEx_xy,CEx_xz, CEx_yz,
                                CEy_xy,CEy_xz, CEy_yz,
                                CEz_xy,CEz_xz, CEz_yz,
                                flnm = flnm, computeJdotE = True, plotFAC = plotFAC, plotAvg = plotAvg, plotFluc = plotFluc, isIon = isIon,isLowPass=isLowPass,isHighPass=isHighPass)

flnmprefixes = ['fftnofiltinv','fftlowpass','ffthighpass']
picklename = ['fftnofiltfilteredfields.pickle','fftfilterabovefilteredfields.pickle','fftfilterbelowfilteredfields.pickle']

#example filename 'figures/elec/faclocal/elecfaclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)

#isLowPass=isLowPass,isHighPass=isHighPass setting arrays
isLParray = [False,True,False] 
isHParray = [False,False,True]

for _i in range(0,len(flnmprefixes)):
    os.system('mkdir figures/ion/'+flnmprefixes[_i])
    os.system('mkdir figures/elec/'+flnmprefixes[_i])
    os.system('mkdir figures/ion/'+flnmprefixes[_i]+'/fluc')
    os.system('mkdir figures/elec/'+flnmprefixes[_i]+'/fluc')
    os.system('mkdir figures/ion/'+flnmprefixes[_i]+'/facfluc')
    os.system('mkdir figures/elec/'+flnmprefixes[_i]+'/facfluc')
    os.system('mkdir figures/ion/'+flnmprefixes[_i]+'/facfluclocal')
    os.system('mkdir figures/elec/'+flnmprefixes[_i]+'/facfluclocal')
    os.system('mkdir figures/ion/'+flnmprefixes[_i]+'/facflucdetrendlocal')
    os.system('mkdir figures/elec/'+flnmprefixes[_i]+'/facflucdetrendlocal')

    isLowPass = isLParray[_i]
    isHighPass = isHParray[_i]

    #OVERWRITE DFIELDS TO USE FILTERED FIELDS
    import pickle
    del dfluc
    filteredflnm = 'analysisfiles/'+picklename[_i]
    filein = open(filteredflnm, 'rb')
    dfluc = pickle.load(filein)
    dfluc['Vframe_relative_to_sim']=0.
    filein.close()

    totfilteredflnm = 'analysisfiles/fftfilterabovetotfilteredfields.pickle'
    filein = open(totfilteredflnm, 'rb')
    dfieldstotlowpass = pickle.load(filein)
    dfieldstotlowpass['Vframe_relative_to_sim']=0.
    filein.close()

    #TODO: remove
    #TODO: fix the need for this quick fix-----
    #startidx_xx = ao.find_nearest(dfields['ex_xx'],5.)
    #startidx_yy = 0
    #startidx_zz = 0
    #endidx_xx = ao.find_nearest(dfields['ex_xx'],10.)
    #endidx_yy = len(dfields['ex_yy'])
    #endidx_zz = len(dfields['ex_zz'])
    #startidxs = [startidx_zz,startidx_yy,startidx_xx]
    #endidxs = [endidx_zz,endidx_yy,endidx_xx]
    #dfieldstemp = ao.subset_dict(dfields,startidxs,endidxs,planes=['z','y','x'])
    #dfieldstemp = ao.avg_dict(dfieldstemp,binidxsz=[1,1,10],planes=['z','y','x'])
    #fixgridkeys = ['ex_xx','ex_yy','ex_zz','bx_xx','bx_yy','bx_zz','ey_xx','ey_yy','ey_zz','by_xx','by_yy','by_zz','ez_xx','ez_yy','ez_zz','bz_xx','bz_yy','bz_zz']
    #for _fkey in fixgridkeys:
    #    dfluc[_fkey]=dfieldstemp[_fkey]
    #end quick fix------

    flnmprefix = flnmprefixes[_i]

    #x1s = [8,8,8,8,8,8]#[8,8,8,8,8]#[8.125,8.125,8.125,8.125,8.125]
    #x2s = [8.5,8.5,8.5,8.5,8.5,8.5]#[8.5,8.5,8.5,8.5,8.5]#[8.375,8.375,8.375,8.375,8.375]
    #y1s = [0,0,1,2,3,4]#[0,1,2,3,4]
    #y2s = [5,1,2,3,4,5]#[1,2,3,4,5]
    
    x1s = np.asarray([8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375])#8.125,8.125,8.125,8.125])#-1.125
    x2s = np.asarray([8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625])#8.375,8.375,8.375,8.375])#+.125
    y1s = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])#0,0,1,2,3,4])
    y2s = np.asarray([5,.2,.3,.4,.5,.6,.7,.8,.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0])#5,1,2,3,4,5])

    #y1s += 2
    #y2s += 2
    
    x1s = np.asarray([8.0])#,7.9,7.8,7.7,7.6,7.5,8.0,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8])
    x2s = np.asarray([8.5])#,8.0,7.9,7.8,7.7,7.6,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9])
    y1s = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    y2s = np.asarray([5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5])


    #x1s = np.asarray([7.9,7.8,7.7,7.6,7.5])
    #x1s = np.asarray([8.375])
    #x2s = np.asarray([8.625,7.9,7.8,7.7,7.6])
    #y1s = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    #y2s = np.asarray([5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5])


    for _index in  range(0,len(x1s)):

        x1 = x1s[_index]
        x2 = x2s[_index]
        y1 = y1s[_index]
        y2 = y2s[_index] #TODO: fix filter bug that is breaking grid...
        z1 = None
        z2 = None

        vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfluc, 'ex', 'x')
        vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfluc, 'ey', 'y')
        vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfluc, 'ez', 'z')
        flnm = 'figures/ion/'+flnmprefix+'/fluc/'+flnmprefix+'ionflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        plot_corr(vxi,vyi,vzi,vmaxion,hist,corex,corey,corez,flnm,plotFluc=True,isLowPass=isLowPass,isHighPass=isHighPass)

        vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfluc, 'ex', 'x')
        vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfluc, 'ey', 'y')
        vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfluc, 'ez', 'z')
        flnm = 'figures/elec/'+flnmprefix+'/fluc/'+flnmprefix+'elecflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        plot_corr(vxe,vye,vze,vmaxelec,hist,corex,corey,corez,flnm,isIon=False,plotFluc=True,isLowPass=isLowPass,isHighPass=isHighPass)

        vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'epar', 'x',altcorfields=dfluc)
        vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp1', 'y',altcorfields=dfluc)
        vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp2', 'z',altcorfields=dfluc)
        flnm = 'figures/ion/'+flnmprefix+'/facfluc/'+flnmprefix+'ionfacflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        plot_corr(vxi,vyi,vzi,vmaxion,hist,corex,corey,corez,flnm,plotFAC=True,plotFluc=True,isLowPass=isLowPass,isHighPass=isHighPass)
        flnm = 'figures/ion/'+flnmprefix+'/facfluc/'+flnmprefix+'iongyrofacflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins)
        npar=np.sum(hist)
        pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=True,plotLog=False,npar=npar,plotFluc=True,isLowPass=isLowPass,isHighPass=isHighPass)
        flnm = 'figures/ion/'+flnmprefix+'/facfluc/'+flnmprefix+'ioncompgyrofacflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
        npar = np.sum(hist)
        pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=True,plotLog=False,npar=npar,plotFluc=True,isLowPass=isLowPass,isHighPass=isHighPass)

        vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'epar', 'x',altcorfields=dfluc)
        vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp1', 'y',altcorfields=dfluc)
        vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp2', 'z',altcorfields=dfluc)
        flnm = 'figures/elec/'+flnmprefix+'/facfluc/'+flnmprefix+'elecfacflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        plot_corr(vxe,vye,vze,vmaxelec,hist,corex,corey,corez,flnm,isIon=False,plotFAC=True,plotFluc=True,isLowPass=isLowPass,isHighPass=isHighPass)
        flnm = 'figures/elec/'+flnmprefix+'/facfluc/'+flnmprefix+'elecgyrofacflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins)
        npar=np.sum(hist)
        pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=False,plotLog=False,plotFluc=True,npar=npar,isLowPass=isLowPass,isHighPass=isHighPass)
        flnm = 'figures/elec/'+flnmprefix+'/facfluc/'+flnmprefix+'eleccompgyrofacflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
        npar = np.sum(hist)
        pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=False,plotLog=False,plotFluc=True,npar=npar,isLowPass=isLowPass,isHighPass=isHighPass)

        vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'epar', 'x', useBoxFAC=False, altcorfields=dfluc)
        vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp1', 'y', useBoxFAC=False, altcorfields=dfluc)
        vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp2', 'z', useBoxFAC=False, altcorfields=dfluc)
        flnm = 'figures/ion/'+flnmprefix+'/facfluclocal/'+flnmprefix+'ionfacfluclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        plot_corr(vxi,vyi,vzi,vmaxion,hist,corex,corey,corez,flnm,plotFAC=True,plotFluc=True,isLowPass=isLowPass,isHighPass=isHighPass)
        flnm = 'figures/ion/'+flnmprefix+'/facfluclocal/'+flnmprefix+'iongyrofacfluclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins)
        npar=np.sum(hist)
        pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=True,plotFluc=True,plotLog=False,npar=npar,isLowPass=isLowPass,isHighPass=isHighPass)
        flnm = 'figures/ion/'+flnmprefix+'/facfluclocal/'+flnmprefix+'ioncompgyrofacfluclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
        npar = np.sum(hist)
        pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=True,plotFluc=True,plotLog=False,npar=npar,isLowPass=isLowPass,isHighPass=isHighPass)


        vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'epar', 'x', useBoxFAC=False, altcorfields=dfluc)
        vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp1', 'y', useBoxFAC=False, altcorfields=dfluc)
        vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp2', 'z', useBoxFAC=False, altcorfields=dfluc)
        flnm = 'figures/elec/'+flnmprefix+'/facfluclocal/'+flnmprefix+'elecfacfluclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        plot_corr(vxe,vye,vze,vmaxelec,hist,corex,corey,corez,flnm,isIon=False,plotFAC=True,plotFluc=True,isLowPass=isLowPass,isHighPass=isHighPass)
        flnm = 'figures/elec/'+flnmprefix+'/facfluclocal/'+flnmprefix+'elecgyrofacfluclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins)
        npar=np.sum(hist)
        pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=False,plotFluc=True,plotLog=False,npar=npar,isLowPass=isLowPass,isHighPass=isHighPass)
        flnm = 'figures/elec/'+flnmprefix+'/facfluclocal/'+flnmprefix+'eleccompgyrofacfluclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
        npar = np.sum(hist)
        pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=False,plotFluc=True,plotLog=False,npar=npar,isLowPass=isLowPass,isHighPass=isHighPass)


        vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfieldstotlowpass, 'epar', 'x', useBoxFAC=False, altcorfields=dfluc)
        vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfieldstotlowpass, 'eperp1', 'y', useBoxFAC=False, altcorfields=dfluc)
        vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfieldstotlowpass, 'eperp2', 'z', useBoxFAC=False, altcorfields=dfluc)
        flnm = 'figures/ion/'+flnmprefix+'/facflucdetrendlocal/'+flnmprefix+'ionfacflucdetrendlocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        plot_corr(vxi,vyi,vzi,vmaxion,hist,corex,corey,corez,flnm,plotFAC=True,plotFluc=True,isLowPass=isLowPass,isHighPass=isHighPass)
        flnm = 'figures/ion/'+flnmprefix+'/facflucdetrendlocal/'+flnmprefix+'iongyrofacflucdetrendlocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins)
        npar=np.sum(hist)
        pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,plotFluc=True,isIon=True,plotLog=False,npar=npar,isLowPass=isLowPass,isHighPass=isHighPass)
        flnm = 'figures/ion/'+flnmprefix+'/facflucdetrendlocal/'+flnmprefix+'ioncompgyrofacflucdetrendlocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
        npar = np.sum(hist)
        pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,plotFluc=True,isIon=True,plotLog=False,npar=npar,isLowPass=isLowPass,isHighPass=isHighPass)

        vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfieldstotlowpass, 'epar', 'x', useBoxFAC=False, altcorfields=dfluc)
        vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfieldstotlowpass, 'eperp1', 'y', useBoxFAC=False, altcorfields=dfluc)
        vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfieldstotlowpass, 'eperp2', 'z', useBoxFAC=False, altcorfields=dfluc)
        flnm = 'figures/elec/'+flnmprefix+'/facflucdetrendlocal/'+flnmprefix+'elecfacflucdetrendlocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        plot_corr(vxe,vye,vze,vmaxelec,hist,corex,corey,corez,flnm,isIon=False,plotFAC=True,plotFluc=True,isLowPass=isLowPass,isHighPass=isHighPass)
        flnm = 'figures/elec/'+flnmprefix+'/facflucdetrendlocal/'+flnmprefix+'elecgyrofacflucdetrendlocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins)
        npar=np.sum(hist) 
        pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,plotFluc=True,isIon=False,plotLog=False,npar=npar,isLowPass=isLowPass,isHighPass=isHighPass)
        flnm = 'figures/elec/'+flnmprefix+'/facflucdetrendlocal/'+flnmprefix+'eleccompgyrofacflucdetrendlocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
        vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
        npar = np.sum(hist)
        pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,plotFluc=True,isIon=False,plotLog=False,npar=npar,isLowPass=isLowPass,isHighPass=isHighPass)

