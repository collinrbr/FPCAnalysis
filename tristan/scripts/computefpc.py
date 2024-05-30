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
dfields_many_frames = {'frame':[],'dfields':[]}
for _num in frames:
    num = int(_num)
    d = ld.load_fields(flpath,_num,normalizeFields=normalize)
    dfields_many_frames['dfields'].append(d)
    dfields_many_frames['frame'].append(num)
vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)
dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

dfavg = aa.get_average_fields_over_yz(dfields)
dfluc = aa.remove_average_fields_over_yz(dfields)

dpar_elec, dpar_ion = ld.load_particles(flpath,framenum,normalizeVelocity=normalize)
inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
inputs = ld.load_input(inputpath)
beta0 = aa.compute_beta0(params,inputs)
dpar_ion = ft.shift_particles(dpar_ion, vshock, beta0, params['mi']/params['me'], isIon=True)
dpar_elec = ft.shift_particles(dpar_elec, vshock, beta0, params['mi']/params['me'], isIon=False)

print("TODO: normalize j dot E in plot to N")

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
                                flnm = flnm, computeJdotE = True, plotFAC = plotFAC, plotAvg = plotAvg, plotFluc = plotFluc, isIon = isIon, isLowPass=isLowPass,isHighPass=isHighPass)


import os
os.system('mkdir figures')
os.system('mkdir figures/elec')
os.system('mkdir figures/elec/tot')
os.system('mkdir figures/elec/fluc')
os.system('mkdir figures/elec/avg')
os.system('mkdir figures/elec/fac')
os.system('mkdir figures/elec/facavg')
os.system('mkdir figures/elec/faclocal')
os.system('mkdir figures/elec/facfluc')
os.system('mkdir figures/elec/facfluclocal')
os.system('mkdir figures/ion')
os.system('mkdir figures/ion/tot')
os.system('mkdir figures/ion/fluc')
os.system('mkdir figures/ion/avg')
os.system('mkdir figures/ion/fac')
os.system('mkdir figures/ion/facavg')
os.system('mkdir figures/ion/faclocal')
os.system('mkdir figures/ion/facfluc')
os.system('mkdir figures/ion/facfluclocal')


x1s = np.asarray([8.0])#8.4,8.375])#,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375,8.375])#8.125,8.125,8.125,8.125])#-1.125
x2s = np.asarray([8.5])#8.5,8.625])#,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625,8.625])#8.375,8.375,8.375,8.375])#+.125
#x1s = np.asarray([7.5,7.75,8,8.25,8.5])
#x2s = np.asarray([7.75,8,8.25,8.5,8.75])
y1s = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])#0,0,1,2,3,4])
y2s = np.asarray([5,5,.2,.3,.4,.5,.6,.7,.8,.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0])#5,1,2,3,4,5])
#y1s += 2
#y2s += 2

#x1s = np.asarray([8.2,8.4,8.6])
#x2s = np.asarray([8.4,8.6,8.8])
#y1s = np.asarray([0,0,0])
#y2s = np.asarray([5,5,5])

#x1s = np.asarray([8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8])
#x2s = np.asarray([8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9])
#x1s = np.asarray([7.9,7.8,7.7,7.6,7.5])
#x2s = np.asarray([8.0,7.9,7.8,7.7,7.6])
#y1s = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#y2s = np.asarray([5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5])

for _index in range(0,len(x1s)):
    x1 = x1s[_index]
    x2 = x2s[_index]
    y1 = y1s[_index]
    y2 = y2s[_index]
    z1 = None
    z2 = None

    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'ex', 'x')
    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'ey', 'y')
    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'ez', 'z')
    flnm = 'figures/ion/tot/iontot_x1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxi,vyi,vzi,vmaxion,hist,corex,corey,corez,flnm)

    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'ex', 'x')
    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'ey', 'y')
    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'ez', 'z')
    flnm = 'figures/elec/tot/electot_x1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxe,vye,vze,vmaxelec,hist,corex,corey,corez,flnm,isIon=False)

    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfluc, 'ex', 'x')
    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfluc, 'ey', 'y')
    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfluc, 'ez', 'z')
    flnm = 'figures/ion/fluc/ionflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxi,vyi,vzi,vmaxion,hist,corex,corey,corez,flnm,plotFluc=True)

    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfluc, 'ex', 'x')
    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfluc, 'ey', 'y')
    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfluc, 'ez', 'z')
    flnm = 'figures/elec/fluc/elecflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxe,vye,vze,vmaxelec,hist,corex,corey,corez,flnm,isIon=False,plotFluc=True)

    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfavg, 'ex', 'x')
    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfavg, 'ey', 'y')
    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfavg, 'ez', 'z')
    flnm = 'figures/ion/avg/ionavgx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxi,vyi,vzi,vmaxion,hist,corex,corey,corez,flnm,plotAvg=True)

    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfavg, 'ex', 'x')
    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfavg, 'ey', 'y')
    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfavg, 'ez', 'z')
    flnm = 'figures/elec/avg/elecavgx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxe,vye,vze,vmaxelec,hist,corex,corey,corez,flnm,isIon=False,plotAvg=True)

    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'epar', 'x')
    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp1', 'y')
    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp2', 'z')
    flnm = 'figures/ion/fac/ionfacx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxi,vyi,vzi,vmaxion,hist,corex,corey,corez,flnm,plotFAC=True)
    flnm = 'figures/ion/fac/iongyrofacx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins)
    pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=True,plotLog=False) 
    flnm = 'figures/ion/facfluc/ioncompgyrofacx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
    npar = np.sum(hist)
    pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=True,plotLog=False)

    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'epar', 'x')
    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp1', 'y')
    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp2', 'z')
    flnm = 'figures/elec/fac/elecfacx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxe,vye,vze,vmaxelec,hist,corex,corey,corez,flnm,isIon=False,plotFAC=True)
    flnm = 'figures/elec/fac/elecgyrofacx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins)
    npar = np.sum(hist)
    pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=False,plotLog=False,npar=npar) 
    flnm = 'figures/elec/fac/eleccompgyrofacx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
    npar = np.sum(hist)
    pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=False,plotLog=False,npar=npar)


    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'epar', 'x',altcorfields=dfluc)
    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp1', 'y',altcorfields=dfluc)
    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp2', 'z',altcorfields=dfluc)
    flnm = 'figures/ion/facfluc/ionfacflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxi,vyi,vzi,vmaxion,hist,corex,corey,corez,flnm,plotFAC=True,plotFluc=True)
    flnm = 'figures/ion/facfluc/iongyrofacflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins)
    npar = np.sum(hist)
    pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=True,plotLog=False,npar=npar,plotFluc=True) 
    flnm = 'figures/ion/facfluc/ioncompgyrofacflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
    npar = np.sum(hist)
    pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=True,plotLog=False,npar=npar,plotFluc=True)


    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'epar', 'x',altcorfields=dfluc)
    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp1', 'y',altcorfields=dfluc)
    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp2', 'z',altcorfields=dfluc)
    flnm = 'figures/elec/facfluc/elecfacflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxe,vye,vze,vmaxelec,hist,corex,corey,corez,flnm,isIon=False,plotFAC=True,plotFluc=True)
    flnm = 'figures/elec/facfluc/elecgyrofacflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins)
    npar = np.sum(hist)
    pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=False,plotLog=False,npar=npar,plotFluc=True) 
    flnm = 'figures/elec/facfluc/eleccompgyrofacflucx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
    npar = np.sum(hist)
    pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=False,plotLog=False,npar=npar,plotFluc=True)


    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'epar', 'x',altcorfields=dfavg)
    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp1', 'y',altcorfields=dfavg)
    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp2', 'z',altcorfields=dfavg)
    flnm = 'figures/ion/facavg/ionfacavgx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxi,vyi,vzi,vmaxion,hist,corex,corey,corez,flnm,plotFAC=True,plotAvg=True)
    flnm = 'figures/ion/facavg/iongyrofacavgx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins)
    npar = np.sum(hist)
    pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=True,plotLog=False,npar=npar) 
    flnm = 'figures/ion/facavg/ioncompgyrofacavgx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
    npar = np.sum(hist)
    pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=True,plotLog=False,npar=npar,plotAvg=True)


    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'epar', 'x',altcorfields=dfavg)
    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp1', 'y',altcorfields=dfavg)
    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp2', 'z',altcorfields=dfavg)
    flnm = 'figures/elec/facavg/elecfacavgx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxe,vye,vze,vmaxelec,hist,corex,corey,corez,flnm,isIon=False,plotFAC=True,plotAvg=True)
    flnm = 'figures/elec/facavg/elecgyrofacavgx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins)
    npar = np.sum(hist)
    pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=False,plotLog=False,npar=npar) 
    flnm = 'figures/elec/facavg/eleccompgyrofacavgx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
    npar = np.sum(hist)
    pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=False,plotLog=False,npar=npar,plotAvg=True)


    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'epar', 'x', useBoxFAC=False)
    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp1', 'y', useBoxFAC=False)
    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp2', 'z', useBoxFAC=False)
    flnm = 'figures/ion/faclocal/ionfaclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxi,vyi,vzi,vmaxion,hist,corex,corey,corez,flnm,plotFAC=True)
    flnm = 'figures/ion/faclocal/iongyrofaclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins)
    npar = np.sum(hist)
    pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=True,plotLog=False,npar=npar) 
    flnm = 'figures/ion/faclocal/ioncompgyrofaclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
    npar = np.sum(hist)
    pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=True,plotLog=False,npar=npar)


    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'epar', 'x', useBoxFAC=False)
    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp1', 'y', useBoxFAC=False)
    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp2', 'z', useBoxFAC=False)
    flnm = 'figures/elec/faclocal/elecfaclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxe,vye,vze,vmaxelec,hist,corex,corey,corez,flnm,isIon=False,plotFAC=True)
    flnm = 'figures/elec/faclocal/elecgyrofaclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins)
    npar = np.sum(hist)
    pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=False,plotLog=False,npar=npar) 
    flnm = 'figures/elec/faclocal/eleccompgyrofaclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
    npar = np.sum(hist)
    pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=False,plotLog=False,npar=npar)

    vxi, vyi, vzi, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'epar', 'x', useBoxFAC=False, altcorfields=dfluc)
    vxi, vyi, vzi, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp1', 'y', useBoxFAC=False, altcorfields=dfluc)
    vxi, vyi, vzi, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxion, dvion, x1, x2, y1, y2, z1, z2, dpar_ion, dfields, 'eperp2', 'z', useBoxFAC=False, altcorfields=dfluc)
    flnm = 'figures/ion/facfluclocal/ionfacfluclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxi,vyi,vzi,vmaxion,hist,corex,corey,corez,flnm,plotFAC=True,plotFluc=True)
    flnm = 'figures/ion/facfluclocal/iongyrofacfluclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins)
    npar = np.sum(hist)
    pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=True,plotLog=False,npar=npar,plotFluc=True)
    flnm = 'figures/ion/facfluclocal/ioncompgyrofacfluclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxi,vyi,vzi,corez,corey,corex,vrmaxion,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
    npar = np.sum(hist)
    pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=True,plotLog=False,npar=npar,plotFluc=True)


    vxe, vye, vze, totalPtcl, hist, corex = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'epar', 'x', useBoxFAC=False, altcorfields=dfluc)
    vxe, vye, vze, totalPtcl, hist, corey = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp1', 'y', useBoxFAC=False, altcorfields=dfluc)
    vxe, vye, vze, totalPtcl, hist, corez = fpc.compute_hist_and_cor(vmaxelec, dvelec, x1, x2, y1, y2, z1, z2, dpar_elec, dfields, 'eperp2', 'z', useBoxFAC=False, altcorfields=dfluc)
    flnm = 'figures/elec/facfluclocal/elecfacfluclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    plot_corr(vxe,vye,vze,vmaxelec,hist,corex,corey,corez,flnm,isIon=False,plotFAC=True,plotFluc=True)
    flnm = 'figures/elec/facfluclocal/elecgyrofacfluclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperpgyro  = aa.compute_gyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
    npar = np.sum(hist)
    pfpc.plot_gyro(vpargyro,vperpgyro,corepargyro,coreperpgyro,flnm=flnm,isIon=False,plotLog=False,npar=npar,plotFluc=True)
    flnm = 'figures/elec/facfluclocal/eleccompgyrofacfluclocalx1-'+str(x1)+'_'+'x2-'+str(x2)+'_'+'y1-'+str(y1)+'_'+'y2-'+str(y2)+'_'+'z1-'+str(z1)+'_'+'z2-'+str(z2)
    vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro  = aa.compute_compgyro_fpc_from_cart_fpc(vxe,vye,vze,corez,corey,corex,vrmaxelec,nrbins) #Note: corez, corey, corex is the correct ordering here... TODO: note/fix this somewhere
    npar = np.sum(hist)
    pfpc.plot_gyro_3comp(vpargyro,vperpgyro,corepargyro,coreperp1gyro,coreperp2gyro,flnm=flnm,isIon=False,plotLog=False,npar=npar,plotFluc=True)
