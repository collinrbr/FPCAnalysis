import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.disprelaux as dr
import lib.arrayaux as ao #array operations
import lib.loadaux as ld
import lib.analysisaux as aa
import lib.ftransfromaux as ft

import os
os.system('mkdir figures')

#user params
xpos = 8.5 #xpos that we measure plasma parameters at

#load fields and transform to shock rest frame
framenum = '700' #frame to make figure of (should be a string)
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]

normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)

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

print("warning: usingn hardcoded vshock to save time...")
vshock = 1.5
#vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)
dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

xidxdfields = ao.find_nearest(dfields['ex_xx'], xpos)
denlocal = np.mean(dfields['dens'][:,:,xidxdfields])
Blocal = np.sqrt(np.mean(dfields['bx'][:,:,xidxdfields])**2+np.mean(dfields['by'][:,:,xidxdfields])**2+np.mean(dfields['bz'][:,:,xidxdfields])**2) 
Tlocal = 1. #TODO:

print("blocal",Blocal)

#compute upstream quants (note: we keep box bigger than the region where cells have been initialized, thus, the rightmost cells will be zero/unitialized annd we must account for this)
rightidx = -1
tol = 0.0001
while(np.abs(np.mean(dfields['by'][:,:,rightidx])) < tol): 
    rightidx -= 1    
Bup = np.sqrt(np.mean(dfields['bx'][:,:,rightidx])**2+np.mean(dfields['by'][:,:,rightidx])**2+np.mean(dfields['bz'][:,:,rightidx])**2)

print("bup", Bup)

rightidx = -1
tol = 0.001
while(np.mean(dfields['dens'][:,:,rightidx]) < tol):
    rightidx -= 1
denup = np.mean(dfields['dens'][:,:,rightidx])

Tup = 1. #TODO: compute

btotlocaloverbtotup = Blocal/Bup
ntotlocalovernup = denlocal/denup
TlocaloverTup = Tlocal/Tup

#compute parameters at location of the sim
print("TODO: compute tau (right now we just look at graph of tau vs x and pick value by eye)")
inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
inputs = ld.load_input(inputpath)
beta0 = aa.compute_beta0(params,inputs)
beta0_i = .5*beta0 #assumes beta0_e = beta0_i
print('beta0', beta0)
beta_i = beta0_i*(ntotlocalovernup*TlocaloverTup)/btotlocaloverbtotup**2 
tau = 7./.2 #Ti over Te! #TODO: load value automatically / compute in this script 
delta_tau = 0.
delta_beta_i = 0.

print("Beta local", beta_i)

print("TODO: make note about sign somehwere (it matters as velcocity of ion/elec scale wave matters wrt eachother")

os.system('mkdir figures/disprel')
for _waveidx in range(0,100):
        
    om1e = .3
    om2e = .3
    om3e = .3
    if(_waveidx == 0):
        #copied from WFT script (ion scale wave)
        #kx:  -27.447210226586755  ky:  1.308451523604499  kz:  0.0  kpar:  1.3079338384349444  kperp1:  -27.44721018448903  kperp2:  0.036834424526316925  sortkeyval:  19.197634294827658
        #(-12.651434605692302+2.335987642765656j) (15.76042072765804+18.510324702526866j) (-13.444340873140028+0.9856378227947711j)
        flnmprefix = 'figures/disprel/elecscaledisprel_normEmax_'
        om1 = 12.651434605692302
        om2 = 15.76042072765804
        om3 = 13.444340873140028
        #om1e = 
        #ome2e =
        #om3e = 
        kpar = 1.3079338384349444
        kperp = 27.44721018448903

    elif(_waveidx == 1):
        #kx:  -7.896085564955251  ky:  -2.616903047208998  kz:  0.0  kpar:  -2.6158656558666076  kperp1:  -7.896085649150677  kperp2:  -0.07366884905263385  sortkeyval:  2.7426404755784968
        #(-1.633431840801371-1.0267969892022462j) (1.3462985439432327+2.2724991703510944j) (-1.7277302563618857-1.075599624042867j)
        flnmprefix = 'figures/disprel/elecscaledisprel_normBmax_'
        om1 = 1.633431840801371
        om2 = 1.3462985439432327
        om3 = 1.7277302563618857
        kpar = 2.6158656558666076
        kperp = 7.896085649150677

    elif(_waveidx == 2):
        #kx:  -23.978462302748905  ky:  14.392966759649488  kz:  0.0  kpar:  14.387263276855998  kperp1:  -23.978461839674036  kperp2:  0.40517866978948613  sortkeyval:  0.04298079402450193
        #(-31.41321503996712+40.19895348802656j) (25.127721938610353+42.87392467357733j) (-36.85237038944938+35.35124745715388j)
        flnmprefix = 'figures/disprel/elecscaledisprel_normB_2' #TODO: double check this one!! Not sure if this is the biggest elec one from normB or not?
        om1=31.41321503996712
        om2=25.127721938610353
        om3=36.85237038944938
        kpar=14.387263276855998
        kperp=23.978461839674036

    elif(_waveidx == 3):
        #kx:  22.086417980655533  ky:  18.318321330462982  kz:  0.0  kpar:  18.311060659211496  kperp1:  22.086418570023525  kperp2:  0.5156819433684369  sortkeyval:  0.03714991741298986
        #(-81.78333200601357-0.3509913777525142j) (13.654956321437716-72.308571464019j) (-84.18294253637936+5.455955795507268j)
        flnmprefix = 'figures/disprel/elecscaledisprel_normB'
        om1 = 81.78333200601357
        om2 = 13.654956321437716
        om3 = 84.18294253637936
        kpar = 18.311060659211496
        kperp = 22.086418570023525

    elif(_waveidx == 4):
        #kx:  -56.4585564986851  ky:  34.01973961371697  kz:  0.0  kpar:  34.00625864737641  kperp1:  -56.45855540414449  kperp2:  0.95769503768424  sortkeyval:  1.5222708562648342
        #(-30.158763777930602+620.5274051081034j) (1384.8033193839221-241.6834453124482j) (-74.44803388137674+630.4486775830972j)
        flnmprefix = 'figures/disprel/elecscaledisprel_normE'
        om1 = 30.158763777930602
        om2 = 1384.8033193839221
        om3 = 74.44803388137674 
        kpar = 34.00625864737641
        kperp = 56.45855540414449

    #elif(_waveidx == 5):
    #    #kx:  -56.4585564986851  ky:  34.01973961371697  kz:  0.0  kpar:  34.00625864737641  kperp1:  -56.45855540414449  kperp2:  0.95769503768424  sortkeyval:  1.5222708562648342
    #    #(-30.158763777930602+620.5274051081034j) (1384.8033193839221-241.6834453124482j) (-74.44803388137674+630.4486775830972j)
    #    #(105.39239368264191-463.57861295429984j) (-18159.81265048627+22299.384881947924j) (-162.08593656100595+219.7928828749766j)
    #    flnmprefix = 'figures/disprel/elecscaledisprel_normE_detrend#'
    #    om1 = 105.39239368264191
    #    om2 = 18159.81265048627
    #    om3 = 162.08593656100595

    elif(_waveidx == 5):
   # 3118
#kx:  2.71193838473484  ky:  -1.308451523604499  kz:  0.0  kpar:  -1.3078653131048639  kperp1:  2.7119383396217747  kperp2:  -0.039165764230487725  sortkeyval:  1.8394525615588875
#real:  -0.24+/-0.04 4.1+/-0.6 -1.44+/-0.11
#imag:  -2.6+/-0.4 10.41+/-0.14 -3.00+/-0.22
        flnmprefix = 'figures/disprel/ionscaledisprel_normE'
        om1 = .24
        om2 = 4.1
        om3 = 1.44
        kpar = 1.3078653131048639
        kperp = 2.7119383396217747
        

    else:
        print("Done!!")
        exit()

    om1 /= btotlocaloverbtotup
    om2 /= btotlocaloverbtotup
    om3 /= btotlocaloverbtotup
    kpar /= np.sqrt(ntotlocalovernup)
    kperp /= np.sqrt(ntotlocalovernup)

    sweeparr = np.arange(.1,100,.01)

    print(_waveidx,'kpar: ',kpar,' kperp: ',kperp)

    oms = [om1,om2,om3]

    kperp_error = [0,0,0] #[.2,.2,.2]
    kpar_error = [0,0,0] #[.2,.2,.2]

    om_error = [0,0,0]#[om1e,om2e,om3e]

    kawkperpsweep = []
    kawkparsweep = []
    fastkperpsweep = []
    fastkparsweep = []
    slowkperpsweep = []
    slowkparsweep = []
    for sweepvar in sweeparr:
        kawkperpsweep.append(dr.kaw_curve(sweepvar,kpar,beta_i,tau,comp_error_prop=False))
        kawkparsweep.append(dr.kaw_curve(kperp,sweepvar,beta_i,tau,comp_error_prop=False))
        fastkperpsweep.append(dr.fastmagson_curve(sweepvar,kpar,beta_i,tau,comp_error_prop=False))
        fastkparsweep.append(dr.fastmagson_curve(kperp,sweepvar,beta_i,tau,comp_error_prop=False))
        slowkperpsweep.append(dr.slowmagson_curve(sweepvar,kpar,beta_i,tau,comp_error_prop=False))
        slowkparsweep.append(dr.slowmagson_curve(kperp,sweepvar,beta_i,tau,comp_error_prop=False))

    #plot 1-----------
    plt.figure(figsize=(6,6))
    plt.style.use('cb.mplstyle')
    plt.rcParams['axes.axisbelow'] = True

    lnwidth = 2.75
    #plt.errorbar(kperp,omega, xerr = kperperr, yerr=omegaerr, fmt="o",color='C0')
    markers = ["o" , "o" , "o"]
    colors = ['red','blue','orange']
    plterror=True
    if(plterror):
        for _i in range(0,3):
            plt.errorbar([kperp],[oms[_i]], xerr = kperp_error[_i], yerr=om_error[_i], fmt="o",c=colors[_i])
    else:
        for _i in range(0,3):
             plt.scatter([kperp],[oms[_i]],c=colors[_i],marker=markers[_i])
    plt.plot(sweeparr,kawkperpsweep,ls='-.',color='black',linewidth=lnwidth)
    plt.plot(sweeparr,fastkperpsweep,ls='-.',color='red',linewidth=lnwidth)
    plt.plot(sweeparr,slowkperpsweep,ls='-.',color='green',linewidth=lnwidth)

    plt.gca().set_aspect('equal')
    plt.xlabel('$k_{\perp} d^{(loc)}_{i}$',fontsize=22)
    plt.ylabel('$\omega / \Omega^{(loc)}_{i}$',fontsize=22)
    plt.gca().yaxis.set_tick_params(labelsize=18)
    plt.gca().xaxis.set_tick_params(labelsize=18)
    plt.axis('scaled')

    #props = dict(boxstyle='square', facecolor='white', alpha=0.75)
    #plt.gca().text(0.05, 0.95, r'$k_{||} d^{(loc)}_{i} = 0.2378$', transform=plt.gca().transAxes, fontsize=20,
    #        verticalalignment='top', bbox=props)

    plt.yscale('log')
    plt.xscale('log')

    plt.ylim(.1,1000)
    plt.xlim(.1,100)
    plt.grid(True, which="both", ls="-")
    plt.savefig(flnmprefix+'kperpsweep.png',format='png',dpi=300,bbox_inches="tight")
    plt.close()

    #plot 2-----------
    plt.figure(figsize=(6,6))
    plt.style.use('cb.mplstyle')
    plt.rcParams['axes.axisbelow'] = True

    lnwidth = 2.5
    #plt.errorbar(kpar,omega, xerr = kparerr, yerr=omegaerr, fmt="o",color='C0')
    if(plterror):
        for _i in range(0,3):
            plt.errorbar([kpar],[oms[_i]], xerr = kpar_error[_i], yerr=om_error[_i], fmt="o",c=colors[_i])
    else:
        for _i in range(0,3):
             plt.scatter([kpar],[oms[_i]],c=colors[_i],marker=markers[_i])

    plt.plot(sweeparr,kawkparsweep,ls='-.',color='black',linewidth=lnwidth)
    plt.plot(sweeparr,fastkparsweep,ls='-.',color='red',linewidth=lnwidth)
    plt.plot(sweeparr,slowkparsweep,ls='-.',color='green',linewidth=lnwidth)

    plt.gca().set_aspect('equal')
    plt.xlabel('$k_{||} d^{(loc)}_{i}$',fontsize=22)
    plt.ylabel('$\omega / \Omega^{(loc)}_{i}$',fontsize=22)
    plt.gca().yaxis.set_tick_params(labelsize=18)
    plt.gca().xaxis.set_tick_params(labelsize=18)
    plt.axis('scaled')

    plt.yscale('log')
    plt.xscale('log')

    plt.ylim(.1,1000)
    plt.xlim(.1,100)

    #props = dict(boxstyle='square', facecolor='white', alpha=0.75)
    #print("WARNING: text box is hard coded!!!")
    #plt.gca().text(0.05, 0.95, r'$k_{\perp} d^{(loc)}_{i} = 1.371$', transform=plt.gca().transAxes, fontsize=20,
    #        verticalalignment='top', bbox=props)

    #xtck = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    #xtcklbls = [r'$10^{-1}$','','','','','','','','',r'$10^0$']
    #plt.xticks(xtck, xtcklbls)
    plt.grid(True, which="both", ls="-")
    plt.savefig(flnmprefix+'kparsweep.png',format='png',dpi=300,bbox_inches="tight")
