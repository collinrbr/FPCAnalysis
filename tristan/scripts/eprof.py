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

precomputed = True
if(not(precomputed)):
    framenum = '699' #frame to make figure of (should be a string)
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
    for evxs in ionvxbins:
        elecvx.append(np.mean(evxs))
    for evys in ionvybins:
        elecvy.append(np.mean(evys))
    for evzs in ionvzbins:
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
        fx = ionvx[_i]*0.5*(iondens[_i])*mi*ionvx[_i]**2
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
        ipdu = mi*np.sum((ibvx*(ionvxbins[_i][idx]-ibvx)**2+ibvy*(ionvxbins[_i][idx]-ibvx)*(ionvybins[_i][idx]-ibvy)+ibvy*(ionvxbins[_i][idx]-ibvx)*(ionvzbins[_i][idx]-ibvz) for idx in range(0,len(ionvxbins[_i]))))
        ionpdotusxs.append(ipdu)
    for _i in range(0,len(elecvxbins)):
        ebvx = elecvx[_i]  #bulk elec velocity
        ebvy = elecvy[_i]
        ebvz = elecvz[_i]
        epdu = me*np.sum((ebvx*(elecvxbins[_i][idx]-ebvx)**2+ebvy*(elecvxbins[_i][idx]-ebvx)*(elecvybins[_i][idx]-ebvy)+ebvy*(elecvxbins[_i][idx]-ebvx)*(elecvzbins[_i][idx]-ebvz) for idx in range(0,len(elecvxbins[_i]))))
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

    poyntxxs = poyntxxs[0:-1]

    data = {}
    data['iondens'] = iondens
    data['elecdens'] = iondens
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

    with open('fluxes699.pickle', 'wb') as f:
        pickle.dump(data, f)


    print("debug exiting!")
    exit()
else:
    with open('fluxes.pickle', 'rb') as f:
        data = pickle.load(f)

    iondens = data['iondens']
    iondens = data['elecdens']
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
 

#normalize using upstream beta. By definition, |vthi^2|/|B^2|=beta_ion if both are in same units
#note that B and E are output in the same units (Tristan uses 'a mix' of cgs and MKS), so if we know the norm factor for B, we know the norm factor for E 

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

print('debug fieldfac')
print(Evthiup/(fieldfac*Bfieldenerup))

print('fac value', fac)

print("debug")
print('ex',dfavg['ex']) #should be small
print('by',dfavg['by']) #should be larger!

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

poyntxxs = fac*np.asarray(poyntxxs)

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

normtot = ionframx[-2]+ionqxs[-2]+elecqxs[-2]+ionethxs[-2]+elecethxs[-2]+ionpdotusxs[-2]+elecpdotusxs[-2]+poyntxxs[-2]

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
plabels = [r'$S_x$',r'$\mathcal{P}_e \cdot \mathbf{U}_e$',r'$3/2 p_e  U_{x,e}$',r'$q_{x,e}$',r'$\mathcal{P}_i \cdot \mathbf{U}_i$',r'$3/2 p_i  U_{x,i}$',r'$q_{x,i}$',r'$F_{x,ram,i}$']
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


#compute W 700
#TODO: convert this to use total fields as there is energy in the fluc fields!!!
W700fields = fac*(dfavg['ex'][0,0,:]**2+dfavg['ey'][0,0,:]**2+dfavg['ez'][0,0,:]**2)+fac*(dfavg['bx'][0,0,:]**2+dfavg['by'][0,0,:]**2+dfavg['bz'][0,0,:]**2)
#extrapolate onto same grid
_, W700fields = interpolate2(dfavg['ex_xx'], W700fields, interpolxxs[0:-1])
W700ion = [0.5*mi*(np.sum((valx)**2+(valy)**2+(valz)**2 for valx,valy,valz in zip(vx,vy,vz))) for vx,vy,vz in zip(ionvxbins,ionvybins,ionvzbins)]
W700elec = [0.5*me*(np.sum((valx)**2+(valy)**2+(valz)**2 for valx,valy,valz in zip(vx,vy,vz))) for vx,vy,vz in zip(ionvxbins,ionvybins,ionvzbins)]
W700tot = np.asarray(W700ion)+np.asarray(W700elec)+np.asarray(W700fields)

#compute W 699
W699fields = fac*(dfavg699['ex'][0,0,:]**2+dfavg699['ey'][0,0,:]**2+dfavg699['ez'][0,0,:]**2)+fac*(dfavg699['bx'][0,0,:]**2+dfavg699['by'][0,0,:]**2+dfavg699['bz'][0,0,:]**2)
#extrapolate onto same grid
_, W699fields = interpolate2(dfavg699['ex_xx'], W699fields, interpolxxs[0:-1])
W699ion = [0.5*mi*(np.sum((valx)**2+(valy)**2+(valz)**2 for valx,valy,valz in zip(vx,vy,vz))) for vx,vy,vz in zip(ionvxbins699,ionvybins699,ionvzbins699)]
W699elec = [0.5*me*(np.sum((valx)**2+(valy)**2+(valz)**2 for valx,valy,valz in zip(vx,vy,vz))) for vx,vy,vz in zip(ionvxbins699,ionvybins699,ionvzbins699)]
W699tot = np.asarray(W699ion)+np.asarray(W699elec)+np.asarray(W699fields)




#plot total energies
fig, axs = plt.subplots(2,figsize=(8,3))
plt.style.use("cb.mplstyle")
axs[0].plot(interpolxxs[0:-1], W700ion, color='red', label = r'$W_{i}$')
axs[0].plot(interpolxxs[0:-1], W700elec, color='blue', label = r'$W_{e}$')
axs[0].plot(interpolxxs[0:-1], W700tot, color='black', label = r'$W_{tot}$')
axs[1].plot(interpolxxs[0:-1], W699ion, color='red', label = r'$W_{i}$')
axs[1].plot(interpolxxs[0:-1], W699elec, color='blue', label = r'$W_{e}$')
axs[1].plot(interpolxxs[0:-1], W699tot, color='black', label = r'$W_{tot}$')
plt.savefig('figures/Wprofiles.png', format = 'png', dpi=300, bbox_inches='tight')

#deltax699 = shockvelsimframe*deltat699
#TODO: shift grid!!
