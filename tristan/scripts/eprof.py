import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')


import pickle
import numpy as np
import matplotlib.pyplot as plt

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa
import lib.plotcoraux as pfpc
import lib.fpcaux as fpc

import pickle

def interpolate(independent_vars, dependent_vars, locations):
    independent_vars = np.array(independent_vars)
    dependent_vars = np.array(dependent_vars)
    locations = np.array(locations)
    interpolated_values = np.interp(locations, independent_vars, dependent_vars)
    return locations, interpolated_values

def interpolate2(independent_vars, dependent_vars, locations):
    independent_vars = np.array(independent_vars)
    dependent_vars = np.array(dependent_vars)
    locations = np.array(locations)
    
    # Determine the scaling factors
    scale_factor_x = np.max(np.abs(independent_vars))
    scale_factor_y = np.max(np.abs(dependent_vars))
    
    # Scale the data
    scaled_independent_vars = independent_vars / scale_factor_x
    scaled_dependent_vars = dependent_vars / scale_factor_y
    scaled_locations = locations / scale_factor_x
    
    # Interpolate the scaled values
    scaled_interpolated_values = np.interp(scaled_locations, scaled_independent_vars, scaled_dependent_vars)
    
    # Inverse scaling for interpolated values
    interpolated_values = scaled_interpolated_values * scale_factor_y
    
    return locations, interpolated_values

interdx = .1
interpolxxs = np.arange(0,12.1,interdx) 
me =  1.
mi = 625.

flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
framenum = '700'
params = ld.load_params(flpath,framenum)
massratio = params['mi']/params['me']
vti0 = np.sqrt(params['delgam'])#Note: velocity is in units γV_i/c so we do not include '*params['c']'
vte0 = np.sqrt(params['mi']/params['me'])*vti0 #WARNING: THIS ASSUME Ti/Te = 1, TODO: don't assume Ti/Te = 1

precomputed = True
if(not(precomputed)):
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
    dfields = ld.load_fields(flpath,framenum,normalizeFields=False,normalizeGrid=True)
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
    vshock = 1.5
    dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

    dfavg = aa.get_average_fields_over_yz(dfields)
    dfluc = aa.remove_average_fields_over_yz(dfields)

    dpar_elec, dpar_ion = ld.load_particles(flpath,framenum,normalizeVelocity=normalize)
    inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
    inputs = ld.load_input(inputpath)
    beta0 = aa.compute_beta0(params,inputs)
    dpar_ion = ft.shift_particles(dpar_ion, vshock, beta0, params['mi']/params['me'], isIon=True)
    dpar_elec = ft.shift_particles(dpar_elec, vshock, beta0, params['mi']/params['me'], isIon=False)

    #bin particles
    print("binning particles")
    ionvxbins = []
    ionvybins = []
    ionvzbins = []
    elecvxbins = []
    elecvybins = []
    elecvzbins = []


    massratio = params['mi']/params['me']
    vti0 = np.sqrt(params['delgam'])#Note: velocity is in units γV_i/c so we do not include '*params['c']'
    vte0 = np.sqrt(params['mi']/params['me'])*vti0 #WARNING: THIS ASSUME Ti/Te = 1, TODO: don't assume Ti/Te = 1

    for _i in range(0,len(interpolxxs)-1):
        x1 = interpolxxs[_i]
        x2 = interpolxxs[_i+1]
        gptsparticleion = (x1 <= dpar_ion['xi']) & (dpar_ion['xi'] <= x2)
        ionvxs = dpar_ion['ui'][gptsparticleion][:]
        ionvys = dpar_ion['vi'][gptsparticleion][:]
        ionvzs = dpar_ion['wi'][gptsparticleion][:]

        gptsparticleelec = (x1 <= dpar_elec['xe']) & (dpar_elec['xe'] <= x2)
        elecvxs = dpar_elec['ue'][gptsparticleelec][:]*vte0/vti0
        elecvys = dpar_elec['ve'][gptsparticleelec][:]*vte0/vti0
        elecvzs = dpar_elec['we'][gptsparticleelec][:]*vte0/vti0

        ionvxbins.append(ionvxs)
        ionvybins.append(ionvys)
        ionvzbins.append(ionvys)

        elecvxbins.append(elecvxs)
        elecvybins.append(elecvys)
        elecvzbins.append(elecvzs)
    print("done!")

    #compute dens
    iondens = []
    elecdens = []
    for ivxs in ionvxbins:
        iondens.append(float(len(ivxs)))
    for evxs in elecvxbins:
        elecdens.append(float(len(evxs)))

    #compute bulk v
    ionvx = []
    ionvy = []
    ionvz = []
    elecvx = []
    elecvy = []
    elecvz = []
    for ivxs in ionvxbins:
        ionvx.append(np.mean(ivxs))
    for ivys in ionvybins:
        ionvy.append(np.mean(ivys))
    for ivzs in ionvzbins:
        ionvz.append(np.mean(ivzs))
    for evxs in elecvxbins:
        elecvx.append(np.mean(evxs))
    for evys in elecvybins:
        elecvy.append(np.mean(evys))
    for evzs in elecvzbins:
        elecvz.append(np.mean(evzs))

    #compute qxs (heat flux)
    ionqxs = []
    elecqxs = []
    for _i in range(0,len(ionvxbins)):
        ivx = ionvxbins[_i]
        ivy = ionvybins[_i]
        ivz = ionvzbins[_i]
        ionqx = 0.5*mi*np.sum((valx-ionvx[_i])*((valx-ionvx[_i])**2+(valy-ionvy[_i])**2+(valz-ionvz[_i])**2) for valx,valy,valz in zip(ivx,ivy,ivz))
        ionqxs.append(ionqx)
    for _i in range(0,len(elecvxbins)):
        evx = elecvxbins[_i]
        evy = elecvybins[_i]
        evz = elecvzbins[_i]
        elecqx = 0.5*me*np.sum((valx-elecvx[_i])*((valx-elecvx[_i])**2+(valy-elecvy[_i])**2+(valz-elecvz[_i])**2) for valx,valy,valz in zip(evx,evy,evz))
        elecqxs.append(elecqx)

    #compute ram kinetic energy flux (neglecting electrons b/c we have a high mass ratio) (that is this block computes the bulk flow energy, but we calll it ram pressure)
    ionframx = []
    for _i, ivxs in enumerate(ionvxbins):
        fx = ionvx[_i]*0.5*(iondens[_i])*mi*(ionvx[_i]**2+ionvy[_i]**2+ionvz[_i]**2)
        ionframx.append(fx)

    #compute enthalpy flux first term
    ionethxs = []
    elecethxs = []
    for _i in range(0,len(ionvxbins)):
        ivx = ionvxbins[_i]
        ivy = ionvybins[_i]
        ivz = ionvzbins[_i]
        ionethx = ionvx[_i]*0.5*mi*np.sum((valx-ionvx[_i])**2+(valy-ionvy[_i])**2+(valz-ionvz[_i])**2 for valx,valy,valz in zip(ivx,ivy,ivz))
        ionethxs.append(ionethx)
    for _i in range(0,len(elecvxbins)):
        evx = elecvxbins[_i]
        evy = elecvybins[_i]
        evz = elecvzbins[_i]
        elecethx = elecvx[_i]*0.5*me*np.sum((valx-elecvx[_i])**2+(valy-elecvy[_i])**2+(valz-elecvz[_i])**2 for valx,valy,valz in zip(evx,evy,evz))
        elecethxs.append(elecethx)

    #compute enthalpy flux second term
    ionpdotusxs = []
    elecpdotusxs = []
    for _i in range(0,len(ionvxbins)):
        ibvx = ionvx[_i]  #bulk ion velocity
        ibvy = ionvy[_i]
        ibvz = ionvz[_i]
        ipdu = mi*np.sum((ibvx*(ionvxbins[_i][idx]-ibvx)**2+ibvy*(ionvxbins[_i][idx]-ibvx)*(ionvybins[_i][idx]-ibvy)+ibvz*(ionvxbins[_i][idx]-ibvx)*(ionvzbins[_i][idx]-ibvz) for idx in range(0,len(ionvxbins[_i]))))
        #ipdu = mi*np.sum((ibvx*(ibvx-ionvxbins[_i][idx])**2+ibvy*(ionvxbins[_i][idx]-ibvx)*(ibvy-ionvybins[_i][idx])+ibvz*(ionvxbins[_i][idx]-ibvx)*(ibvz-ionvzbins[_i][idx]) for idx in range(0,len(ionvxbins[_i]))))
        ionpdotusxs.append(ipdu)
    for _i in range(0,len(elecvxbins)):
        ebvx = elecvx[_i]  #bulk elec velocity
        ebvy = elecvy[_i]
        ebvz = elecvz[_i]
        epdu = me*np.sum((ebvx*(elecvxbins[_i][idx]-ebvx)**2+ebvy*(elecvxbins[_i][idx]-ebvx)*(elecvybins[_i][idx]-ebvy)+ebvz*(elecvxbins[_i][idx]-ebvx)*(elecvzbins[_i][idx]-ebvz) for idx in range(0,len(elecvxbins[_i]))))
        #epdu = me*np.sum((ebvx*(elecvxbins[_i][idx]-ebvx)**2+ebvy*(elecvxbins[_i][idx]-ebvx)*(ebvy-elecvybins[_i][idx])+ebvz*(elecvxbins[_i][idx]-ebvx)*(ebvz-elecvzbins[_i][idx]) for idx in range(0,len(elecvxbins[_i]))))
        elecpdotusxs.append(epdu)

    #compute total energy flux
    iontotefluxxs = []
    electotefluxxs = []
    for _i in range(0,len(ionvxbins)):
        ibvx = ionvx[_i]  #bulk ion velocity
        ibvy = ionvy[_i]
        ibvz = ionvz[_i]
        itfx = mi/2.*np.sum((ionvxbins[_i][idx]*(ionvxbins[_i][idx]**2+ionvybins[_i][idx]**2+ionvzbins[_i][idx]**2) for idx in range(0,len(ionvxbins[_i]))))
        iontotefluxxs.append(itfx)
    for _i in range(0,len(elecvxbins)):
        ebvx = elecvx[_i]  #bulk elec velocity
        ebvy = elecvy[_i]
        ebvz = elecvz[_i]
        etfx = me/2.*np.sum((elecvxbins[_i][idx]*(elecvxbins[_i][idx]**2+elecvybins[_i][idx]**2+elecvzbins[_i][idx]**2) for idx in range(0,len(elecvxbins[_i]))))
        electotefluxxs.append(etfx)

    #compute poynt flux
    poyntxxs = []
    for xx in interpolxxs:
        x1 = xx
        x2 = xx+interdx
        Eys = dfields['ey']
        Ezs = dfields['ez']
        Bys = dfields['by']
        Bzs = dfields['bz']
        goodfieldpts = (x1 <= dfavg['ey_xx']) & (dfavg['ey_xx'] <= x2)
        pxx = np.sum(Eys[:,:,goodfieldpts]*Bzs[:,:,goodfieldpts]-Ezs[:,:,goodfieldpts]*Bys[:,:,goodfieldpts])
        pxx = pxx/float(len(goodfieldpts))
        poyntxxs.append(pxx)
    poyntxxs = poyntxxs[0:-1] #drop last term as we create one too many! #note: we normalize to correct units later using 'fac'!

    #compute upstream thermal energy for normalization factor
    bvxi = np.mean(ionvxbins[-3])
    bvyi = np.mean(ionvybins[-3])
    bvzi = np.mean(ionvzbins[-3])
    Evthiup = 0.5*mi*np.sum((ionvxbins[-3]-bvxi)**2+(ionvybins[-3]-bvyi)**2+(ionvzbins[-3]-bvzi)**2)

    #compute upstream B field ener
    Bfieldener = dfavg['bx'][0,0,:]**2+dfavg['by'][0,0,:]**2+dfavg['bz'][0,0,:]**2
    Bfieldenerup = np.mean(Bfieldener[-100:-50])

    #TODO: we do this twice? should be careful and remove one!!!
    #normalize so the ratio is betaion
    #Ethviup/a*Bfieldenerup = betaion = .125
    # a = Ethviup/betaion*Bfieldeneruup
    betaion = .125
    fieldfac = Evthiup/(betaion*Bfieldenerup)
    fac = fieldfac/(4*3.14159) #a normalizes field value squared  (4 pi factor is from mu0, which is excluded above)

    #compute W
    #TODO: convert this to use total fields as there is energy in the fluc fields!!!
    Wfields = fac*(dfavg['ex'][0,0,:]**2+dfavg['ey'][0,0,:]**2+dfavg['ez'][0,0,:]**2)+fac*(dfavg['bx'][0,0,:]**2+dfavg['by'][0,0,:]**2+dfavg['bz'][0,0,:]**2)
    #extrapolate onto same grid
    _, Wfields = interpolate2(dfavg['ex_xx'], Wfields, interpolxxs[0:-1])
    Wion = [0.5*mi*(np.sum((valx)**2+(valy)**2+(valz)**2 for valx,valy,valz in zip(vx,vy,vz))) for vx,vy,vz in zip(ionvxbins,ionvybins,ionvzbins)]
    Welec = [0.5*me*(np.sum((valx)**2+(valy)**2+(valz)**2 for valx,valy,valz in zip(vx,vy,vz))) for vx,vy,vz in zip(ionvxbins,ionvybins,ionvzbins)]
    Wtot = np.asarray(Wion)+np.asarray(Welec)+np.asarray(Wfields)

    data = {}
    data['iondens'] = iondens
    data['elecdens'] = elecdens
    data['ionvx'] = ionvx
    data['ionvy'] = ionvy
    data['ionvz'] = ionvz
    data['elecvx'] = elecvx
    data['elecvy'] = elecvy
    data['elecvz'] = elecvz
    data['ionqxs'] = ionqxs
    data['elecqxs'] = elecqxs
    data['ionframx'] = ionframx
    data['ionethxs'] = ionethxs
    data['elecethxs'] = elecethxs
    data['ionpdotusxs'] = ionpdotusxs
    data['elecpdotusxs'] = elecpdotusxs
    data['iontotefluxxs'] = iontotefluxxs
    data['electotefluxxs'] = electotefluxxs
    data['poyntxxs'] = poyntxxs
    data['ionvxbins'] = ionvxbins
    data['ionvybins'] = ionvybins
    data['ionvzbins'] = ionvzbins
    data['dfavg'] = dfavg
    data['Wfields'] = Wfields
    data['Wion'] = Wion
    data['Welec'] = Welec
    data['Wtot'] = Wtot

    with open('fluxes.pickle', 'wb') as f:
        pickle.dump(data, f)

    print("debug exiting!")
    exit()
else:
    with open('fluxes.pickle', 'rb') as f:
        data = pickle.load(f)

    iondens = data['iondens']
    elecdens = data['elecdens']
    ionvx = data['ionvx']
    ionvy = data['ionvy']
    ionvz = data['ionvz']
    elecvx = data['elecvx']
    elecvy = data['elecvy']
    elecvz = data['elecvz']
    ionqxs = data['ionqxs']
    elecqxs = data['elecqxs']
    ionframx = data['ionframx'] 
    ionethxs = data['ionethxs']
    elecethxs = data['elecethxs']
    ionpdotusxs = data['ionpdotusxs']
    elecpdotusxs = data['elecpdotusxs']
    iontotefluxxs = data['iontotefluxxs']
    electotefluxxs = data['electotefluxxs']
    poyntxxs = data['poyntxxs']
    ionvxbins = data['ionvxbins']
    ionvybins = data['ionvybins']
    ionvzbins = data['ionvzbins']
    dfavg = data['dfavg']
    W700fields = data['Wfields']
    W700ion = data['Wion']
    W700elec = data['Welec']
    W700tot = data['Wtot']


#normalize using upstream beta. By definition, |vthi^2|/|B^2|=beta_ion if both are in same units
#note that B and E are output in the same units (Tristan uses 'a mix' of cgs and MKS), so if we know the norm factor for B, we know the norm factor for E 

print('debug','ionvx: ',ionvx)
print('debug','iondens: ',iondens)
print('debug','ionframx: ',ionframx)


#TODO: move this block to block above (don't load dfields twice as it wastes cpu time!) and save/load in pickle-----
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
dfields = ld.load_fields(flpath,framenum,normalizeFields=False,normalizeGrid=True)
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
vshock = 1.5
dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

dfavg = aa.get_average_fields_over_yz(dfields)
dfluc = aa.remove_average_fields_over_yz(dfields)

Ebar = np.stack((dfavg['ex'], dfavg['ey'], dfavg['ez']), axis=-1)
Bbar = np.stack((dfavg['bx'], dfavg['by'], dfavg['bz']), axis=-1)
Ebar_cross_Bbar = np.cross(Ebar, Bbar)
EbarxBbar_x = Ebar_cross_Bbar[..., 0]
EbarxBbar_y = Ebar_cross_Bbar[..., 1]
EbarxBbar_z = Ebar_cross_Bbar[..., 2]

Efluc = np.stack((dfluc['ex'], dfluc['ey'], dfluc['ez']), axis=-1)
Bfluc = np.stack((dfluc['bx'], dfluc['by'], dfluc['bz']), axis=-1)
Efluc_cross_Bfluc = np.cross(Efluc, Bfluc)
EflucxBfluc_x = Efluc_cross_Bfluc[..., 0]
EflucxBfluc_y = Efluc_cross_Bfluc[..., 1]
EflucxBfluc_z = Efluc_cross_Bfluc[..., 2]

#reduce to 1D (as these are functions of x only!!!) with correct normalization (the norm fac used later scales 1 point of fields^2 units to total Evthi according to ion plasma beta)
EflucxBfluc_x_out = []
EflucxBfluc_y_out = []
EflucxBfluc_z_out = []
EbarxBbar_x_out = []
EbarxBbar_y_out = []
EbarxBbar_z_out = []
for xx in interpolxxs:
    x1 = xx
    x2 = xx+interdx
    goodfieldpts = (x1 <= dfavg['ey_xx']) & (dfavg['ey_xx'] <= x2)
    EflucxBflucxx = np.sum(EflucxBfluc_x[:,:,goodfieldpts]) 
    EflucxBflucxx = EflucxBflucxx/float(len(goodfieldpts)) 
    EflucxBfluc_x_out.append(EflucxBflucxx)
    EflucxBflucyy = np.sum(EflucxBfluc_y[:,:,goodfieldpts]) 
    EflucxBflucyy = EflucxBflucyy/float(len(goodfieldpts)) 
    EflucxBfluc_y_out.append(EflucxBflucyy)
    EflucxBfluczz = np.sum(EflucxBfluc_z[:,:,goodfieldpts]) 
    EflucxBfluczz = EflucxBfluczz/float(len(goodfieldpts)) 
    EflucxBfluc_z_out.append(EflucxBfluczz)
    EbarxBbarxx = np.sum(EbarxBbar_x[:,:,goodfieldpts]) 
    EbarxBbarxx = EbarxBbarxx/float(len(goodfieldpts)) 
    EbarxBbar_x_out.append(EbarxBbarxx)
    EbarxBbaryy = np.sum(EbarxBbar_y[:,:,goodfieldpts]) 
    EbarxBbaryy = EbarxBbaryy/float(len(goodfieldpts)) 
    EbarxBbar_y_out.append(EbarxBbaryy)
    EbarxBbarzz = np.sum(EbarxBbar_z[:,:,goodfieldpts]) 
    EbarxBbarzz = EbarxBbarzz/float(len(goodfieldpts))
    EbarxBbar_z_out.append(EbarxBbarzz)

sizefac = float(len(dfavg['ex_yy'])*len(dfavg['ex_zz']))
EflucxBfluc_x_avg = np.asarray(EflucxBfluc_x_out[0:-1])/sizefac #note: we normalize to correct units later using 'fac'!
EflucxBfluc_y_avg = np.asarray(EflucxBfluc_y_out[0:-1])/sizefac #note: we normalize to correct units later using 'fac'!
EflucxBfluc_z_avg = np.asarray(EflucxBfluc_z_out[0:-1])/sizefac #note: we normalize to correct units later using 'fac'!
EbarxBbar_x = np.asarray(EbarxBbar_x_out[0:-1])/sizefac #note: we normalize to correct units later using 'fac'!
EbarxBbar_y = np.asarray(EbarxBbar_y_out[0:-1])/sizefac
EbarxBbar_z = np.asarray(EbarxBbar_z_out[0:-1])/sizefac


#compute Qbar
sizefac = float(len(dfavg['ex_yy'])*len(dfavg['ex_zz'])) #Need to divide both by the same '1/LyLz' factor which here is number of grid points!!!!
Qbarx = np.asarray(iontotefluxxs)/(sizefac) + np.asarray(electotefluxxs)/(sizefac) 


#end block to be pickled ------------------------------------------------



for _tempi in range(len(ionvx)):
    print(interpolxxs[_tempi],ionvx[_tempi],iondens[_tempi],ionframx[_tempi])

#compute upstream vthi
bvxi = np.mean(ionvxbins[-3])
bvyi = np.mean(ionvybins[-3])
bvzi = np.mean(ionvzbins[-3])
Evthiup = 0.5*mi*np.sum((ionvxbins[-3]-bvxi)**2+(ionvybins[-3]-bvyi)**2+(ionvzbins[-3]-bvzi)**2)

#compute upstream B field ener
Bfieldener = dfavg['bx'][0,0,:]**2+dfavg['by'][0,0,:]**2+dfavg['bz'][0,0,:]**2
Bfieldenerup = np.mean(Bfieldener[-100:-50])

#normalize so the ratio is betaion
#Ethviup/a*Bfieldenerup = betaion = .125
# a = Ethviup/betaion*Bfieldeneruup
betaion = .125
fieldfac = Evthiup/(betaion*Bfieldenerup)
fac = fieldfac/(4*3.14159) #a normalizes field value squared  (4 pi factor is from mu0, which is excluded above)


#normalize flux to be in particle units (comes from sx+ux = const assumption, which is strong in the upstream)`
#uix = iontotefluxxs[-5] + electotefluxxs[-5]
#ufx = iontotefluxxs[-3] + electotefluxxs[-3]
#six = poyntxxs[-5] 
#sfx = poyntxxs[-3]
#fac = (uix-ufx)/(sfx-six)


#uixsum1 = 0.
#sixsum1 = 0.
#for _j in [-1,-2,-3,-4,-5,-6,-7]:
#    uixsum1 = uixsum1 + iontotefluxxs[_j] + electotefluxxs[_j]
#    sixsum1 = sixsum1 + poyntxxs[_j]
#
#ufxsum1 = 0.
#sfxsum1 = 0.
#for _j in [-8,-9,-10,-11,-12,-13,-14]:
#    ufxsum1 = ufxsum1 + iontotefluxxs[_j] + electotefluxxs[_j]
#    sfxsum1 = sfxsum1 + poyntxxs[_j]
#fac = (uixsum1-ufxsum1)/(sfxsum1-sixsum1)

#compute grad dot q (d/dx dot q technically)
Qtot = np.asarray(ionframx) + np.asarray(ionqxs) + np.asarray(elecqxs) + np.asarray(ionethxs) + np.asarray(elecethxs) + np.asarray(ionpdotusxs) + np.asarray(elecpdotusxs)
gradQtot = np.gradient(Qtot, xx)

Qtotion = np.asarray(ionframx) + np.asarray(ionqxs) + np.asarray(ionethxs) + np.asarray(ionpdotusxs)
gradQtotion = np.gradient(Qtotion, xx)

Qtotelec = np.asarray(elecqxs) + np.asarray(elecethxs) + np.asarray(elecpdotusxs)
gradQtotelec = np.gradient(Qtotelec, xx)


#make dummy plot to load font (weird work around)
plt.figure() 
plt.style.use("cb.mplstyle")
plt.plot([0,1],[0,1])
plt.savefig('_delete_this_temp.png')
plt.close()



#debug comparison!
xxplot=interpolxxs[0:-1]
plt.figure(figsize=(3*8,3*3))
plt.style.use("cb.mplstyle")
plt.plot(xxplot,np.asarray(Qtot),color='black',ls='--',label=r'Sum of terms')
plt.plot(xxplot,np.asarray(Qtotion),color='red',ls='-',label=r'Sum of terms ion')
plt.plot(xxplot,np.asarray(Qtotelec),color='blue',ls='-',label=r'Sum of terms elec')
plt.plot(xxplot,np.asarray(iontotefluxxs) + np.asarray(electotefluxxs),color='gray',ls='-.',label=r'Direct Qtot')
plt.plot(xxplot,np.asarray(iontotefluxxs),color='green',ls=':',label=r'Direct Ion')
plt.plot(xxplot,np.asarray(electotefluxxs),color='purple',ls=':',label=r'Direct Elec')
plt.xlabel(r'$x/d_i$')
plt.legend()
plt.grid()
plt.savefig('figures/Qtotdebugtest.png', format = 'png', dpi=300, bbox_inches='tight')


#normalize poynt flux to particle energy flux
poyntxxs = fac*np.asarray(poyntxxs)
EbarxBbar_x = fac*np.asarray(EbarxBbar_x)
EflucxBfluc_x_avg = fac*np.asarray(EflucxBfluc_x_avg)


#make indiv plots for fluxes
ionframx = np.asarray(ionframx)
ionqxs = np.asarray(ionqxs)
elecqxs = np.asarray(elecqxs)
ionethxs = np.asarray(ionethxs)
elecethxs = np.asarray(elecethxs)
ionpdotusxs = np.asarray(ionpdotusxs)
elecpdotusxs = np.asarray(elecpdotusxs)
poyntxxs = np.asarray(poyntxxs)

plabels = [r'$S_x$',r'$\mathcal{P}_e \cdot \mathbf{U}_e$',r'$3/2 p_e  U_{x,e}$',r'$q_{x,e}$',r'$\mathcal{P}_i \cdot \mathbf{U}_i$',r'$3/2 p_i  U_{x,i}$',r'$q_{x,i}$',r'$F_{x,ram,i}$']
plabels.reverse()
fluxes = [ionframx,ionqxs,ionethxs,ionpdotusxs,elecqxs,elecethxs,elecpdotusxs,poyntxxs]
fluxesvarname = ['ionframx','ionqxs','ionethxs','ionpdotusxs','elecqxs','elecethxs','elecpdotusxs','poyntxxs']
normfac = np.abs(ionframx[-2]+ionqxs[-2]+ionethxs[-2]+ionpdotusxs[-2]+elecqxs[-2]+elecethxs[-2]+elecpdotusxs[-2]+poyntxxs[-2])
import os
os.system('mkdir figures/fluxprofindiv')
for _i in range(0,len(plabels)):
    plt.figure(figsize=(16, 6))
    plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
    xx=interpolxxs[0:-1]
    plt.plot(xx,fluxes[_i]/normfac)
    plt.grid()
    plt.xlim(0,12)
    plt.xlabel(r'$x/d_i$')
    plt.ylabel(plabels[_i]+r'$/|(\sum_s Q_s +S)_{x,up}|$')
    plt.savefig('figures/fluxprofindiv/'+fluxesvarname[_i]+'.png')
    plt.close()

plt.figure(figsize=(16, 6))
plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
xx=interpolxxs[0:-1]
plt.plot(xx,(ionframx+ionqxs+ionethxs+ionpdotusxs+elecqxs+elecethxs+elecpdotusxs+poyntxxs)/normfac)
plt.grid()
plt.xlim(0,12)
plt.xlabel(r'$x/d_i$')
plt.ylabel(r'$(\sum_s Q_s +S)/(|\sum_s Q_s +S)_{x,up}|$')
plt.savefig('figures/fluxprofindiv/sumdebug.png')
plt.close()


#Make flux area sweep
def split_positive_negative(arr):
    arr = np.array(arr)
            
    positive_array = np.where(arr > 0, arr, 0)
    negative_array = np.where(arr < 0, arr, 0)

    return positive_array, negative_array

ionframxpos,ionframxneg = split_positive_negative(ionframx)
ionqxspos, ionqxsneg = split_positive_negative(ionqxs)
elecqxspos, elecqxsneg = split_positive_negative(elecqxs)
ionethxspos, ionethxsneg = split_positive_negative(ionethxs)
elecethxspos, elecethxsneg = split_positive_negative(elecethxs)
ionpdotusxspos, ionpdotusxsneg = split_positive_negative(ionpdotusxs)
elecpdotusxspos, elecpdotusxsneg = split_positive_negative(elecpdotusxs)
poyntxxspos, poyntxxsneg = split_positive_negative(poyntxxs)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), gridspec_kw={'width_ratios': [5, 1]})

xx=interpolxxs[0:-1]
runtotpos = np.zeros(len(xx))
runtotneg = np.zeros(len(xx))

plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

normtot = np.abs(ionframx[-2]+ionqxs[-2]+elecqxs[-2]+ionethxs[-2]+elecethxs[-2]+ionpdotusxs[-2]+elecpdotusxs[-2]+poyntxxs[-2])

alpha = 0.8
tolfrac = 0.001

#we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
if(np.max(np.abs(ionframxpos)) < np.abs(normtot)*tolfrac):
    alpha = 0.
else:
    alpha = 0.8
fill_patchionframxpos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+ionframxpos)/normtot,hatch='++', color='blue', alpha=alpha)
runtotpos += ionframxpos

#we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
if(np.max(np.abs(ionqxspos)) < np.abs(normtot)*tolfrac):
    alpha = 0.
else:
    alpha = 0.8
fill_patchionqxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+ionqxspos)/normtot,hatch='\\', color='gray', alpha=alpha)
runtotpos += ionqxspos

#we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
if(np.max(np.abs(ionethxspos)) < np.abs(normtot)*tolfrac):
    alpha = 0.
else:
    alpha = 0.8
fill_patchionethxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+ionethxspos)/normtot,hatch='/', color='green', alpha=alpha)
runtotpos += ionethxspos

#we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
if(np.max(np.abs(ionpdotusxspos)) < np.abs(normtot)*tolfrac):
    alpha = 0.
else:
    alpha = 0.8
fill_patchionpdotusxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+ionpdotusxspos)/normtot, hatch='x', color='purple', alpha=alpha)
runtotpos += ionpdotusxspos

#we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
if(np.max(np.abs(elecqxspos)) < np.abs(normtot)*tolfrac):
    alpha = 0.
else:
    alpha = 0.8
    # Create mask for where the difference exceeds the threshold
    mask = np.abs(elecqxspos) > np.abs(normtot)*.01 #We only implent it here as for this specific case, it is only needed here! It probably would be better to just use the mask everywhere instead of changing alpha!
fill_patchelecqxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+elecqxspos)/normtot,where=mask,hatch='+', color='red', alpha=alpha)
runtotpos += elecqxspos

#we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
if(np.max(np.abs(elecethxspos)) < np.abs(normtot)*tolfrac):
    alpha = 0.
else:
    alpha = 0.8
fill_patchelecethxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+elecethxspos)/normtot,hatch='-', color='orange', alpha=alpha)
runtotpos += elecethxspos

#we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
if(np.max(np.abs(elecpdotusxspos)) < np.abs(normtot)*tolfrac):
    alpha = 0.
else:
    alpha = 0.8
fill_patchelecpdotusxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+elecpdotusxspos)/normtot, hatch='|', color='pink', alpha=alpha)
runtotpos += elecpdotusxspos

#we will hide all areas that are too small to impact things, as they create an outline that is larger than the area itself!
if(np.max(np.abs(poyntxxspos)) < np.abs(normtot)*tolfrac):
    alpha = 0.
else:
    alpha = 0.8
fill_patchpoyntxxspos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+poyntxxspos)/normtot, hatch='///', color='black', alpha=alpha)
runtotpos += poyntxxspos

#note: we only hide small positive are since in this specific case, it only creates the illusion of positive contributions for the positive areas
alpha = 0.8

fill_patchionframxneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+ionframxneg)/normtot,hatch='++', color='blue', alpha=alpha)
runtotneg += ionframxneg

fill_patchionqxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+ionqxsneg)/normtot,hatch='\\', color='gray', alpha=alpha)
runtotneg += ionqxsneg

fill_patchionethxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+ionethxsneg)/normtot,hatch='/', color='green', alpha=alpha)
runtotneg += ionethxsneg

fill_patchionpdotusxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+ionpdotusxsneg)/normtot, hatch='x', color='purple', alpha=alpha)
runtotneg += ionpdotusxsneg

fill_patchelecqxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+elecqxsneg)/normtot,hatch='+', color='red', alpha=alpha)
runtotneg += elecqxsneg

fill_patchelecethxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+elecethxsneg)/normtot,hatch='-', color='orange', alpha=alpha)
runtotneg += elecethxsneg

fill_patchelecpdotusxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+elecpdotusxsneg)/normtot, hatch='|', color='pink', alpha=alpha)
runtotneg += elecpdotusxsneg

fill_patchpoyntxxsneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+poyntxxsneg)/normtot, hatch='///', color='black', alpha=alpha)
runtotneg += poyntxxsneg

# Create a legend for the fill_between plot on the second subplot
legend_elements = []

_idx = 0
plabels = [r'$S_x$',r'$(\mathcal{P}_e \cdot \mathbf{U}_e)_x$',r'$3/2 p_e  U_{x,e}$',r'$q_{x,e}$',r'$(\mathcal{P}_i \cdot \mathbf{U}_i)_x$',r'$3/2 p_i  U_{x,i}$',r'$q_{x,i}$',r'$F_{x,ram,i}$']
plabels.reverse()
for fp in [fill_patchionframxneg,fill_patchionqxsneg,fill_patchionethxsneg,fill_patchionpdotusxsneg,fill_patchelecqxsneg,fill_patchelecethxsneg,fill_patchelecpdotusxsneg,fill_patchpoyntxxsneg]:
    facecolor = fp.get_facecolor()
    alpha = fp.get_alpha()
    hatch = fp.get_hatch()
    edgecolor = fp.get_edgecolor()
    legend_elements.append(mpatches.Patch(facecolor=facecolor, alpha=alpha, hatch=hatch, edgecolor=edgecolor, label=plabels[_idx]))
    _idx += 1

legend_elements.reverse()

ax2.legend(handles=legend_elements, loc='center',bbox_to_anchor=(0, .4, 0.25, 0.25),fontsize=12)

ax1.set_xlabel(r'$x/d_i$')
ax1.grid()
ax1.set_xlim(0,12)
#ax1.set_ylim(0,2.0)
ax2.set_axis_off()
plt.savefig('figures/fluxposnegprofile.png', format = 'png', dpi=300, bbox_inches='tight')
plt.close()

#TODO: rename these vars to indicate they are the abs value of the original
ionframx = np.asarray(np.abs(ionframx))
ionqxs = np.asarray(np.abs(ionqxs))
elecqxs = np.asarray(np.abs(elecqxs))
ionethxs = np.asarray(np.abs(ionethxs))
elecethxs = np.asarray(np.abs(elecethxs))
ionpdotusxs = np.asarray(np.abs(ionpdotusxs))
elecpdotusxs = np.asarray(np.abs(elecpdotusxs))
poyntxxs = np.asarray(np.abs(poyntxxs))

#make plots
#make dummy plot to load font (weird work around)
plt.figure()
plt.style.use("cb.mplstyle")
plt.plot([0,1],[0,1])
plt.savefig('_delete_this_temp.png')
plt.close()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), gridspec_kw={'width_ratios': [5, 1]})

xx=interpolxxs[0:-1]
runtot = np.zeros(len(xx))

plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

normtot = np.abs(ionframx[-2]+ionqxs[-2]+elecqxs[-2]+ionethxs[-2]+elecethxs[-2]+ionpdotusxs[-2]+elecpdotusxs[-2]+poyntxxs[-2])

fill_patchionframx = ax1.fill_between(xx, runtot/normtot, (runtot+ionframx)/normtot,hatch='++', color='blue', alpha=0.8)
runtot += ionframx

fill_patchionqxs = ax1.fill_between(xx, runtot/normtot, (runtot+ionqxs)/normtot,hatch='\\', color='gray', alpha=0.8)
runtot += ionqxs

fill_patchionethxs = ax1.fill_between(xx, runtot/normtot, (runtot+ionethxs)/normtot,hatch='/', color='green', alpha=0.8)
runtot += ionethxs

fill_patchionpdotusxs = ax1.fill_between(xx, runtot/normtot, (runtot+ionpdotusxs)/normtot, hatch='x', color='purple', alpha=0.8)
runtot += ionpdotusxs

fill_patchelecqxs = ax1.fill_between(xx, runtot/normtot, (runtot+elecqxs)/normtot,hatch='+', color='red', alpha=0.8)
runtot += elecqxs

fill_patchelecethxs = ax1.fill_between(xx, runtot/normtot, (runtot+elecethxs)/normtot,hatch='-', color='orange', alpha=0.8)
runtot += elecethxs

fill_patchelecpdotusxs = ax1.fill_between(xx, runtot/normtot, (runtot+elecpdotusxs)/normtot, hatch='|', color='pink', alpha=0.8)
runtot += elecpdotusxs

fill_patchpoyntxxs = ax1.fill_between(xx, runtot/normtot, (runtot+poyntxxs)/normtot, hatch='///', color='black', alpha=0.8)
runtot += poyntxxs

# Create a legend for the fill_between plot on the second subplot
legend_elements = []

_idx = 0
plabels = [r'$S_x$',r'$(\mathcal{P}_e \cdot \mathbf{U}_e)_x$',r'$3/2 p_e  U_{x,e}$',r'$q_{x,e}$',r'$(\mathcal{P}_i \cdot \mathbf{U}_i)_x$',r'$3/2 p_i  U_{x,i}$',r'$q_{x,i}$',r'$F_{x,ram,i}$']
plabels.reverse()
for fp in [fill_patchionframx,fill_patchionqxs,fill_patchionethxs,fill_patchionpdotusxs,fill_patchelecqxs,fill_patchelecethxs,fill_patchelecpdotusxs,fill_patchpoyntxxs]:
    facecolor = fp.get_facecolor()
    alpha = fp.get_alpha()
    hatch = fp.get_hatch()
    edgecolor = fp.get_edgecolor()
    legend_elements.append(mpatches.Patch(facecolor=facecolor, alpha=alpha, hatch=hatch, edgecolor=edgecolor, label=plabels[_idx]))
    _idx += 1

legend_elements.reverse()

ax2.legend(handles=legend_elements, loc='center',bbox_to_anchor=(0, .4, 0.25, 0.25),fontsize=12)

ax1.set_xlabel(r'$x/d_i$')
ax1.grid()
ax1.set_xlim(0,12)
ax1.set_ylim(0,2.0)
ax2.set_axis_off()
plt.savefig('figures/fluxprofile.png', format = 'png', dpi=300, bbox_inches='tight')
plt.close()

#make pie charts
if(False):
    import os
    positions = list(range(120))#[118,60,27]
    os.system('mkdir figures/fluxprofile_pies')
    for pxval in positions:
        plt.figure()
        plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots
        plt.title(str(xx[pxval])+' $ d_i$')
        wedges, _ = plt.pie([ionframx[pxval],ionqxs[pxval],ionethxs[pxval],ionpdotusxs[pxval],elecqxs[pxval],elecethxs[pxval],elecpdotusxs[pxval],poyntxxs[pxval]],hatch=['++','\\','/','x','+','-','|','///'],colors=['blue','gray','green','purple','red','orange','pink','black'],wedgeprops={"alpha": 0.8})

        pieedgecolors=['blue','gray','green','purple','red','orange','pink','black']
        _pidx = 0
        for pie_wedge in wedges:
            pie_wedge.set_edgecolor(pieedgecolors[_pidx])
            #pie_wedge.set_facecolor(colors[pie_wedge.get_label()][0])
            #pie_wedge.set_hatch('/')
            _pidx += 1

        plt.savefig('figures/fluxprofile_pies/'+str(pxval)+'.png', format = 'png', dpi=300, bbox_inches='tight')
        plt.close()

#break up contributoins by inistability and bulk terms maintaining positive and negative values
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

Qbarxpos,Qbarxneg = split_positive_negative(Qbarx)
EflucxBfluc_x_avgpos,EflucxBfluc_x_avgneg = split_positive_negative(EflucxBfluc_x_avg)
EbarxBbar_xpos,EbarxBbar_xneg = split_positive_negative(EbarxBbar_x)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), gridspec_kw={'width_ratios': [5, 1]})

xx=interpolxxs[0:-1]
runtotpos = np.zeros(len(xx))
runtotneg = np.zeros(len(xx))

plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

normtot = np.abs(EflucxBfluc_x_avg[-2]+EbarxBbar_x[-2]+Qbarx[-2])

fill_patchQbarxpos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+Qbarxpos)/normtot,hatch='++', color='blue', alpha=0.8)
runtotpos += Qbarxpos

fill_patchEflucxBfluc_x_avgpos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+EflucxBfluc_x_avgpos)/normtot,hatch='//', color='green', alpha=0.8)
runtotpos += EflucxBfluc_x_avgpos

fill_patchEbarxBbar_xpos = ax1.fill_between(xx, runtotpos/normtot, (runtotpos+EbarxBbar_xpos)/normtot,hatch='x', color='red', alpha=0.8)
runtotpos += EbarxBbar_xpos

fill_patchQbarxneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+Qbarxneg)/normtot,hatch='++', color='blue', alpha=0.8)
runtotneg += Qbarxneg

fill_patchEflucxBfluc_x_avgneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+EflucxBfluc_x_avgneg)/normtot,hatch='//', color='green', alpha=0.8)
runtotneg += EflucxBfluc_x_avgneg

fill_patchEbarxBbar_xneg = ax1.fill_between(xx, runtotneg/normtot, (runtotneg+EbarxBbar_xneg)/normtot,hatch='x', color='red', alpha=0.8)
runtotneg += EbarxBbar_xneg

legend_elements = []

_idx = 0
plabels = [r'$\sum_s Q_{s,x}$',r'$(<\widetilde{\mathbf{E}} \times \widetilde{\mathbf{B}}>_{y,z})_{x}$',r'$(\overline{\mathbf{E}} \times \overline{\mathbf{B}})_{x}$']
#plabels.reverse()
for fp in [fill_patchQbarxneg,fill_patchEflucxBfluc_x_avgneg,fill_patchEbarxBbar_xneg]:
    facecolor = fp.get_facecolor()
    alpha = fp.get_alpha()
    hatch = fp.get_hatch()
    edgecolor = fp.get_edgecolor()
    legend_elements.append(mpatches.Patch(facecolor=facecolor, alpha=alpha, hatch=hatch, edgecolor=edgecolor, label=plabels[_idx]))
    _idx += 1

legend_elements.reverse()

ax2.legend(handles=legend_elements, loc='center',bbox_to_anchor=(0, .4, 0.25, 0.25),fontsize=12)

ax1.set_xlabel(r'$x/d_i$')
ax1.grid()
ax1.set_xlim(0,12)
#ax1.set_ylim(0,2.0)
ax2.set_axis_off()
plt.savefig('figures/fluxbarposnegprofile.png', format = 'png', dpi=300, bbox_inches='tight')


#break up contributions by instability and bulk terms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), gridspec_kw={'width_ratios': [5, 1]})

xx=interpolxxs[0:-1]
runtot = np.zeros(len(xx))

plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

EflucxBfluc_x_avg = np.abs(EflucxBfluc_x_avg)
EbarxBbar_x = np.abs(EbarxBbar_x)
Qbarx = np.abs(Qbarx)

normtot = EflucxBfluc_x_avg[-2]+EbarxBbar_x[-2]+Qbarx[-2]

fill_patchQbarx = ax1.fill_between(xx, runtot/normtot, (runtot+Qbarx)/normtot,hatch='++', color='blue', alpha=0.8)
runtot += Qbarx

fill_patchEflucxBfluc_x_avg = ax1.fill_between(xx, runtot/normtot, (runtot+EflucxBfluc_x_avg)/normtot,hatch='//', color='green', alpha=0.8)
runtot += EflucxBfluc_x_avg

fill_patchEbarxBbar_x = ax1.fill_between(xx, runtot/normtot, (runtot+EbarxBbar_x)/normtot,hatch='x', color='red', alpha=0.8)
runtot += EbarxBbar_x

# Create a legend for the fill_between plot on the second subplot
legend_elements = []

_idx = 0
plabels = [r'$\sum_s Q_{s,x}$',r'$(<\widetilde{\mathbf{E}} \times \widetilde{\mathbf{B}}>_{y,z})_{x}$',r'$(\overline{\mathbf{E}} \times \overline{\mathbf{B}})_{x}$']
#plabels.reverse()
for fp in [fill_patchQbarx,fill_patchEflucxBfluc_x_avg,fill_patchEbarxBbar_x]:
    facecolor = fp.get_facecolor()
    alpha = fp.get_alpha()
    hatch = fp.get_hatch()
    edgecolor = fp.get_edgecolor()
    legend_elements.append(mpatches.Patch(facecolor=facecolor, alpha=alpha, hatch=hatch, edgecolor=edgecolor, label=plabels[_idx]))
    _idx += 1

legend_elements.reverse()

ax2.legend(handles=legend_elements, loc='center',bbox_to_anchor=(0, .4, 0.25, 0.25),fontsize=12)

ax1.set_xlabel(r'$x/d_i$')
ax1.grid()
ax1.set_xlim(0,12)
#ax1.set_ylim(0,2.0)
ax2.set_axis_off()
plt.savefig('figures/fluxbarprofile.png', format = 'png', dpi=300, bbox_inches='tight')



# compute dW/dt
with open('fluxes699.pickle', 'rb') as f:
    data = pickle.load(f)

iondens699 = data['iondens']
iondens699 = data['elecdens']
ionvx699 = data['ionvx']
ionvy699 = data['ionvy']
ionvz699 = data['ionvz']
elecvx699 = data['elecvx']
elecvy699 = data['elecvy']
elecvz699 = data['elecvz']
ionqxs699 = data['ionqxs']
elecqxs699 = data['elecqxs']
ionframx699 = data['ionframx']
ionethxs699 = data['ionethxs']
elecethxs699 = data['elecethxs']
ionpdotusxs699 = data['ionpdotusxs']
elecpdotusxs699 = data['elecpdotusxs']
iontotefluxxs699 = data['iontotefluxxs']
electotefluxxs699 = data['electotefluxxs']
poyntxxs699 = data['poyntxxs']
ionvxbins699 = data['ionvxbins']
ionvybins699 = data['ionvybins']
ionvzbins699 = data['ionvzbins']
dfavg699 = data['dfavg']
W699fields = data['Wfields'] 
W699ion = data['Wion'] 
W699elec = data['Welec'] 
W699tot = data['Wtot'] 

#plot total energies
fig, axs = plt.subplots(4,figsize=(8,9),sharex=True)
plt.style.use("cb.mplstyle")
axs[0].plot(interpolxxs[0:-1], W700ion, color='red', label = r'$W_{i}$')
axs[0].plot(interpolxxs[0:-1], W700elec, color='blue', label = r'$W_{e}$')
axs[0].plot(interpolxxs[0:-1], W700tot, color='black', label = r'$W_{tot}$')
axs[1].plot(interpolxxs[0:-1], W699ion, color='red', label = r'$W_{i}$')
axs[1].plot(interpolxxs[0:-1], W699elec, color='blue', label = r'$W_{e}$')
axs[1].plot(interpolxxs[0:-1], W699tot, color='black', label = r'$W_{tot}$')
axs[2].plot(interpolxxs[0:-1], np.asarray(W700ion)-np.asarray(W699ion), color='red', label = r'$W_{i}$')
axs[2].plot(interpolxxs[0:-1], np.asarray(W700elec)-np.asarray(W699elec), color='blue', label = r'$W_{e}$')
axs[2].plot(interpolxxs[0:-1], np.asarray(W700tot)-np.asarray(W699tot), color='black', label = r'$W_{tot}$')
axs[3].plot(xx, gradQtot, color='black', label = r'$\partial/\partial x Q_{tot}$')

for ax in axs:
    ax.legend()
plt.savefig('figures/Wprofiles.png', format = 'png', dpi=300, bbox_inches='tight')

#compute d/dt W
#deltax699 = shockvelsimframe*deltat699
#TODO: shift grid!!

#compute delta t in correct units! (dt is time for a particle traveling at vthi0 to travel one grid width!) (as d/dt W + grad dot (Q+S) = 0 and Q = int dv3 \mathbf{v} 1/2 ms v^2 f_s, numerically taking the spatial gradient is like dividing by delta x = grid spacing. thus to have the d/dt W have the same units, it must be the delta_t to cross delta x. Since we are normalized to vthi, this is the time for a particle with v = vthi0 to cross delta x) 
#gridspace = (get gridspace w/o norm!) WARNING WE ARE MIXING UNITS BY TAKING GRAD!!! (xx is in di by vthi is not!) (at  least i think)
#basedt = vthi0/gridspace
#stride = 100
#deltadt = basedt*stride
#print('delta dt,',deltadt)
#TODO: check that d/dtW + grad dot (S + Q) = 0!!!!!
#d/dtW + grad dot (S + Q) = 0!

#load j dot E (from enervsx.py)
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

#note the normalization factor of vte0/vti0
enerCEx_elec_tilde = np.asarray([np.sum(C_Ex_vxvyflucelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyflucelec))])*vte0/vti0
enerCEy_elec_tilde = np.asarray([np.sum(C_Ey_vxvyflucelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyflucelec))])*vte0/vti0
enerCEz_elec_tilde = np.asarray([np.sum(C_Ez_vxvyflucelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvyflucelec))])*vte0/vti0
enerCEtot_elec_tilde = np.asarray([enerCEx_elec_tilde[_i] + enerCEy_elec_tilde[_i] + enerCEz_elec_tilde[_i] for _i in range(len(enerCEx_elec_tilde))])*vte0/vti0

enerCEx_elec = np.asarray([np.sum(C_Ex_vxvytotelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvytotelec))])*vte0/vti0
enerCEy_elec = np.asarray([np.sum(C_Ey_vxvytotelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvytotelec))])*vte0/vti0
enerCEz_elec = np.asarray([np.sum(C_Ez_vxvytotelec[_i])*dvelec**3 for _i in range(len(C_Ex_vxvytotelec))])*vte0/vti0
enerCEtot_elec = np.asarray([enerCEx_elec[_i] + enerCEy_elec[_i] + enerCEz_elec[_i] for _i in range(len(enerCEz_elec))])*vte0/vti0

enerCEtot_elec_bar = np.asarray([enerCEtot_elec[_id] - enerCEtot_elec_tilde[_id] for _id in range(len(enerCEtot_elec))])*vte0/vti0
enerCEx_elec_bar = np.asarray([enerCEx_elec[_id] - enerCEx_elec_tilde[_id] for _id in range(len(enerCEx_elec))])*vte0/vti0
enerCEy_elec_bar = np.asarray([enerCEy_elec[_id] - enerCEy_elec_tilde[_id] for _id in range(len(enerCEy_elec))])*vte0/vti0
enerCEz_elec_bar = np.asarray([enerCEz_elec[_id] - enerCEz_elec_tilde[_id] for _id in range(len(enerCEz_elec))])*vte0/vti0

#compute relative contributions
_, enerCEtot_ion = interpolate2(x_in, enerCEtot_ion, xx)
_, enerCEtot_elec = interpolate2(x_in, enerCEtot_elec, xx)
_, enerCEtot_ion_tilde = interpolate2(x_in, enerCEtot_ion_tilde, xx)
_, enerCEtot_ion_bar = interpolate2(x_in, enerCEtot_ion_bar, xx)
_, enerCEtot_elec_tilde = interpolate2(x_in, enerCEtot_elec_tilde, xx)
_, enerCEtot_elec_bar = interpolate2(x_in, enerCEtot_elec_bar, xx)

gradQtot_tilde = gradQtot*(enerCEtot_ion_tilde+enerCEtot_elec_tilde)/(enerCEtot_ion+enerCEtot_elec)
gradQtot_bar = gradQtot*(enerCEtot_ion_bar+enerCEtot_elec_bar)/(enerCEtot_ion+enerCEtot_elec)

gradQtotion_tilde = gradQtotion*(enerCEtot_ion_tilde)/(enerCEtot_ion)
gradQtotion_bar = gradQtotion*(enerCEtot_ion_bar)/(enerCEtot_ion)

gradQtotelec_tilde = gradQtotelec*(enerCEtot_elec_tilde)/(enerCEtot_elec)
gradQtotelec_bar = gradQtotelec*(enerCEtot_elec_bar)/(enerCEtot_elec)

fig, axs = plt.subplots(3,figsize=(8,9),sharex=True)
plt.style.use("cb.mplstyle")

axs[0].plot(xx, gradQtot_tilde, color='red',ls='-.', label = r'$\partial/\partial x \widetilde{Q}_{tot}$')
axs[0].plot(xx, gradQtot_bar, color='blue',ls='--', label = r'$\partial/\partial x \overline{Q}_{tot}$')
axs[0].plot(xx, gradQtot, color='black', label = r'$\partial/\partial x {Q}_{tot}$')

axs[1].plot(xx, gradQtotion_tilde, color='red',ls='-.', label = r'$\partial/\partial x \widetilde{Q}_{tot,i}$')
axs[1].plot(xx, gradQtotion_bar, color='blue',ls='--', label = r'$\partial/\partial x \overline{Q}_{tot,i}$')
axs[1].plot(xx, gradQtotion, color='black', label = r'$\partial/\partial x {Q}_{tot,i}$')

axs[2].plot(xx, gradQtotelec_tilde, color='red',ls='-.', label = r'$\partial/\partial x \widetilde{Q}_{tot,e}$')
axs[2].plot(xx, gradQtotelec_bar, color='blue',ls='--', label = r'$\partial/\partial x \overline{Q}_{tot,e}$')
axs[2].plot(xx, gradQtotelec, color='black', label = r'$\partial/\partial x {Q}_{tot,e}$')

for ax in axs:
    ax.legend()
    ax.grid()
plt.savefig('figures/gradQcontributions.png', format = 'png', dpi=300, bbox_inches='tight')


plt.figure(figsize=(3*8,3*3))
plt.style.use("cb.mplstyle")
normfac = np.sqrt((iondens[-2]*ionvx[-2])**2+(iondens[-2]*ionvy[-2])**2+(iondens[-2]*ionvz[-2])**2)
plt.plot(xx,np.asarray(iondens)*np.asarray(ionvx)/(normfac),color='r',ls='-',label=r'$n_iU_{x,i}/n_{i,0}|U_{i,0}|$')
plt.plot(xx,np.asarray(iondens)*np.asarray(ionvy)/(normfac),color='b',ls='-.',label=r'$n_iU_{y,i}/n_{i,0}|U_{i,0}|$')
plt.plot(xx,np.asarray(iondens)*np.asarray(ionvz)/(normfac),color='g',ls=':',label=r'$n_iU_{z,i}/n_{i,0}|U_{i,0}|$')
plt.xlabel(r'$x/d_i$')
plt.legend()
plt.grid()
plt.savefig('figures/nUxidebugtest.png', format = 'png', dpi=300, bbox_inches='tight')


plt.figure(figsize=(3*8,3*3))
plt.style.use("cb.mplstyle")
normfac = np.sqrt((ionvx[-2])**2+(ionvy[-2])**2+(ionvz[-2])**2)
plt.plot(xx,np.asarray(ionvx)/(normfac),color='r',ls='-',label=r'$U_{x,i}/|U_{i,0}|$')
plt.plot(xx,np.asarray(ionvy)/(normfac),color='b',ls='-.',label=r'$U_{y,i}/|U_{i,0}|$')
plt.plot(xx,np.asarray(ionvz)/(normfac),color='g',ls=':',label=r'$U_{z,i}/|U_{i,0}|$')
plt.xlabel(r'$x/d_i$')
plt.legend()
plt.grid()
plt.savefig('figures/Uidebugtest.png', format = 'png', dpi=300, bbox_inches='tight')


plt.figure(figsize=(3*8,3*3))
plt.style.use("cb.mplstyle")
normfac = np.sqrt((elecvx[-2])**2+(elecvy[-2])**2+(elecvz[-2])**2)
plt.plot(xx,np.asarray(elecvx)/(normfac),color='r',ls='-',label=r'$U_{x,e}/|U_{e,0}|$')
plt.plot(xx,np.asarray(elecvy)/(normfac),color='b',ls='-.',label=r'$U_{y,e}/|U_{e,0}|$')
plt.plot(xx,np.asarray(elecvz)/(normfac),color='g',ls=':',label=r'$U_{z,e}/|U_{e,0}|$')
plt.xlabel(r'$x/d_i$')
plt.legend()
plt.grid()
plt.savefig('figures/Uedebugtest.png', format = 'png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(3*8,3*3))
plt.style.use("cb.mplstyle")
plt.plot(xx,np.asarray(iondens)/iondens[-2],color='r',ls='-',label=r'$n_i/n_{i,0}$')
plt.plot(xx,np.asarray(elecdens)/elecdens[-2],color='b',ls='-.',label=r'$n_i/n_{e,0}$')
plt.xlabel(r'$x/d_i$')
plt.legend()
plt.grid()
plt.savefig('figures/ndebugtest.png', format = 'png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(3*8,3*3))
plt.style.use("cb.mplstyle")
normfac = np.sqrt((ionvx[-2])**2+(ionvy[-2])**2+(ionvz[-2])**2)
plt.plot(xx,np.asarray(ionvx)/(normfac),color='black',ls='--',label=r'$U_{x,i}/|U_{i,0}|$')
normfac2 = np.sqrt((elecvx[-2])**2+(elecvy[-2])**2+(elecvz[-2])**2)
plt.plot(xx,np.asarray(elecvx)/(normfac2),color='gray',ls='-.',label=r'$U_{x,e}/|U_{e,0}|$')
plt.xlabel(r'$x/d_i$')
plt.legend()
plt.grid()
plt.savefig('figures/UexUixdebugtest.png', format = 'png', dpi=300, bbox_inches='tight')



