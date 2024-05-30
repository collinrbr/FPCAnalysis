import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa
import lib.arrayaux as ao
import pickle
import os

#------------------------------------------------------------------------------------------------------------------------------------
# Begin script
#------------------------------------------------------------------------------------------------------------------------------------

#user params
framenum = '100' #frame to make figure of (should be a string)
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
#vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)
print("warning... using hard coded value of vshock to save time")
vshock = 1.5
dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

print("DEBUG! REDUCING DFIELDS DETAIL FOR DEBUGGING")
dfields = ao.avg_dict(dfields,binidxsz=[1,25,50],planes=['z','y','x'])

isprecomputed = False
precomputedflnm = 'analysisfiles/tempvspos_frame100.pickle'#'analysisfiles/tempvspos_short_frame100.pkl' #OOPS Should have been saved to: 'analysisfiles/tempvspos_frame100.pickle'

if(not(isprecomputed)):
    dpar_elec, dpar_ion = ld.load_particles(flpath,framenum,normalizeVelocity=normalize)
    inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
    inputs = ld.load_input(inputpath)
    beta0 = aa.compute_beta0(params,inputs)

    dpar_ion = ft.shift_particles(dpar_ion, vshock, beta0, params['mi']/params['me'], isIon=True)
    dpar_elec = ft.shift_particles(dpar_elec, vshock, beta0, params['mi']/params['me'], isIon=False)

    oldkeysion = ['ui','vi','wi','xi','yi','zi']
    oldkeyselec = ['ue','ve','we','xe','ye','ze']
    newkeys = ['p1','p2','p3','x1','x2','x3'] #convert to legacy key names
    for _tidx in range(len(oldkeysion)):
        dpar_ion[newkeys[_tidx]] = dpar_ion[oldkeysion[_tidx]] 
        dpar_elec[newkeys[_tidx]] = dpar_elec[oldkeyselec[_tidx]]

    loadfrac = 1 # =1 loads all particles
    print("computing local fac for ions...")
    dpar_ion,_ = aa.change_velocity_basis_local(dfields,dpar_ion,loadfrac=loadfrac)
    print("computing local fac for elecs...")
    dpar_elec,_  = aa.change_velocity_basis_local(dfields,dpar_elec,loadfrac=loadfrac)

    #bin particles
    nx = len(dfields['ex_xx'])
    ny = len(dfields['ex_yy'])
    ion_bins = [[[] for _2 in range(ny)] for _ in range(nx)] #indexed as [_xidx][_yidx]
    elec_bins = [[[] for _2 in range(ny)] for _ in range(nx)]

    #compute matricies to transpose particles using box avg FAC
    print("Computing box FACs")
    boxavg_change_matricies = []
    for _fidx in range(0,len(dfields['ex_xx'])):
        zlim = [-9999999,9999999]
        ylim = [-9999999,9999999]
        xlim = [dfields['ex_xx'][_fidx]-0.00001,dfields['ex_xx'][_fidx]+0.00001]
        vparbasis, vperp1basis, vperp2basis = aa.compute_field_aligned_coord(dfields,xlim,ylim,zlim)
        #make change of basis matrix
        _ = np.asarray([vparbasis,vperp1basis,vperp2basis]).T
        changebasismatrix = np.linalg.inv(_)
        boxavg_change_matricies.append(changebasismatrix)

    debug = True
    ionxxs = []
    ionyys = []
    print('binning ions...')
    for _i in range(0,int(len(dpar_ion['xi']))): 
        if(debug and _i % 100000 == 0): print("Binned: ", _i," ions of ", len(dpar_ion['xi']))
        xx = dpar_ion['xi'][_i]
        yy = dpar_ion['yi'][_i]
        xidx = ao.find_nearest(dfields['ex_xx'], xx)
        yidx = ao.find_nearest(dfields['ex_yy'], yy)
        pparboxfac,pperp1boxfac,pperp2boxfac = np.matmul(boxavg_change_matricies[xidx],[dpar_ion['ui'][_i],dpar_ion['vi'][_i],dpar_ion['wi'][_i]])
        ion_bins[xidx][yidx].append({'ui':dpar_ion['ui'][_i] ,'vi':dpar_ion['vi'][_i] ,'wi':dpar_ion['wi'][_i], 'pari':dpar_ion['ppar'][_i], 'perp1i':dpar_ion['pperp1'][_i], 'perp2i':dpar_ion['pperp2'][_i], 'pparboxfaci':pparboxfac, 'pperp1boxfaci':pperp1boxfac, 'pperp2boxfaci':pperp2boxfac})
    ionxxs = dfields['ex_xx']
    ionyys = dfields['ex_yy']

    elecxxs = []
    elecyys = []
    for _i in range(0,int(len(dpar_elec['xe']))):
        if(debug and _i % 100000 == 0): print("Binned: ", _i," elecs of ", len(dpar_elec['xe']))
        xx = dpar_elec['xe'][_i]
        yy = dpar_elec['ye'][_i]
        xidx = ao.find_nearest(dfields['ex_xx'], xx)
        yidx = ao.find_nearest(dfields['ex_yy'], yy)
        pparboxfac,pperp1boxfac,pperp2boxfac = np.matmul(boxavg_change_matricies[xidx],[dpar_elec['ue'][_i],dpar_elec['ve'][_i],dpar_elec['we'][_i]])
        elec_bins[xidx][yidx].append({'ue':dpar_elec['ue'][_i] ,'ve':dpar_elec['ve'][_i] ,'we':dpar_elec['we'][_i], 'pare':dpar_elec['ppar'][_i], 'perp1e':dpar_elec['pperp1'][_i], 'perp2e':dpar_elec['pperp2'][_i],'pparboxface':pparboxfac, 'pperp1boxface':pperp1boxfac, 'pperp2boxface':pperp2boxfac})
    elecxxs = dfields['ex_xx']
    elecyys = dfields['ex_yy']

    vmaxion = 12.
    vmaxelec = 7.
    dvion =  2.
    dvelec = .5

    print("binning particles")
    vxbins = np.arange(-vmaxion, vmaxion+dvion, dvion)
    vx = (vxbins[1:] + vxbins[:-1])/2.
    vybins = np.arange(-vmaxion, vmaxion+dvion, dvion)
    vy = (vybins[1:] + vybins[:-1])/2.
    vzbins = np.arange(-vmaxion, vmaxion+dvion, dvion)
    vz = (vzbins[1:] + vzbins[:-1])/2.

    ionhists = [[[] for _2 in range(ny)] for _ in range(nx)] #indexed as [_xidx][_yidx]
    ionhistfac = [[[] for _2 in range(ny)] for _ in range(nx)] #indexed as [_xidx][_yidx]
    ionhistboxfac = [[[] for _2 in range(ny)] for _ in range(nx)] #indexed as [_xidx][_yidx]
    for _idx in range(0,len(ion_bins)):
        print('binning ',_idx, 'of ',len(ion_bins),' ion')
        for _idy in range(0,len(ion_bins[_idx])):
            tempuxs = np.asarray([ion_bins[_idx][_idy][_jdx]['ui'] for _jdx in range(0,len(ion_bins[_idx][_idy]))])
            tempuys = np.asarray([ion_bins[_idx][_idy][_jdx]['wi'] for _jdx in range(0,len(ion_bins[_idx][_idy]))])
            tempuzs = np.asarray([ion_bins[_idx][_idy][_jdx]['vi'] for _jdx in range(0,len(ion_bins[_idx][_idy]))])
            hist,_ = np.histogramdd((tempuzs,tempuys,tempuxs), bins=[vzbins, vybins, vxbins])
            ionhists[_idx][_idy]=hist

            tempuxs = np.asarray([ion_bins[_idx][_idy][_jdx]['perp1i'] for _jdx in range(0,len(ion_bins[_idx][_idy]))])
            tempuys = np.asarray([ion_bins[_idx][_idy][_jdx]['perp2i'] for _jdx in range(0,len(ion_bins[_idx][_idy]))])
            tempuzs = np.asarray([ion_bins[_idx][_idy][_jdx]['pari'] for _jdx in range(0,len(ion_bins[_idx][_idy]))])
            hist,_ = np.histogramdd((tempuzs,tempuys,tempuxs), bins=[vzbins, vybins, vxbins]) #Index order is [_par,_perp2,_perp1]
            ionhistfac[_idx][_idy]=hist

            tempuxs = np.asarray([ion_bins[_idx][_idy][_jdx]['pperp1boxfaci'] for _jdx in range(0,len(ion_bins[_idx][_idy]))])
            tempuys = np.asarray([ion_bins[_idx][_idy][_jdx]['pperp2boxfaci'] for _jdx in range(0,len(ion_bins[_idx][_idy]))])
            tempuzs = np.asarray([ion_bins[_idx][_idy][_jdx]['pparboxfaci'] for _jdx in range(0,len(ion_bins[_idx][_idy]))])
            hist,_ = np.histogramdd((tempuzs,tempuys,tempuxs), bins=[vzbins, vybins, vxbins]) #Index order is [_par,_perp2,_perp1]
            ionhistboxfac[_idx][_idy]=hist
    vxion = vx[:]
    vyion = vy[:]
    vzion = vz[:]
    print("done binning ions into hists")

    vxbins = np.arange(-vmaxelec, vmaxelec+dvelec, dvelec)
    vx = (vxbins[1:] + vxbins[:-1])/2.
    vybins = np.arange(-vmaxion, vmaxelec+dvelec, dvelec)
    vy = (vybins[1:] + vybins[:-1])/2.
    vzbins = np.arange(-vmaxion, vmaxelec+dvelec, dvelec)
    vz = (vzbins[1:] + vzbins[:-1])/2.
    elechists = [[[] for _2 in range(ny)] for _ in range(nx)] #indexed as [_xidx][_yidx]
    elechistfac = [[[] for _2 in range(ny)] for _ in range(nx)] #indexed as [_xidx][_yidx]
    elechistboxfac = [[[] for _2 in range(ny)] for _ in range(nx)] #indexed as [_xidx][_yidx]
    for _idx in range(0,len(elec_bins)):
        print('binning ',_idx, 'of ',len(elec_bins),' elec')
        for _idy in range(0,len(elec_bins[_idx])):
            tempuxs = [elec_bins[_idx][_idy][_jdx]['ue'] for _jdx in range(0,len(elec_bins[_idx][_idy]))]
            tempuys = [elec_bins[_idx][_idy][_jdx]['we'] for _jdx in range(0,len(elec_bins[_idx][_idy]))]
            tempuzs = [elec_bins[_idx][_idy][_jdx]['ve'] for _jdx in range(0,len(elec_bins[_idx][_idy]))]
            hist,_ = np.histogramdd((tempuzs, tempuys, tempuxs), bins=[vzbins, vybins, vxbins])
            elechists[_idx][_idy]=hist

            tempuxs = [elec_bins[_idx][_idy][_jdx]['perp1e'] for _jdx in range(0,len(elec_bins[_idx][_idy]))]
            tempuys = [elec_bins[_idx][_idy][_jdx]['perp2e'] for _jdx in range(0,len(elec_bins[_idx][_idy]))]
            tempuzs = [elec_bins[_idx][_idy][_jdx]['pare'] for _jdx in range(0,len(elec_bins[_idx][_idy]))]
            hist,_ = np.histogramdd((tempuzs, tempuys, tempuxs), bins=[vzbins, vybins, vxbins]) #Index order is [_par,_perp2,_perp1]
            elechistfac[_idx][_idy]=hist

            tempuxs = [elec_bins[_idx][_idy][_jdx]['pperp1boxface'] for _jdx in range(0,len(elec_bins[_idx][_idy]))]
            tempuys = [elec_bins[_idx][_idy][_jdx]['pperp2boxface'] for _jdx in range(0,len(elec_bins[_idx][_idy]))]
            tempuzs = [elec_bins[_idx][_idy][_jdx]['pparboxface'] for _jdx in range(0,len(elec_bins[_idx][_idy]))]
            hist,_ = np.histogramdd((tempuzs, tempuys, tempuxs), bins=[vzbins, vybins, vxbins]) #Index order is [_par,_perp2,_perp1]
            elechistboxfac[_idx][_idy]=hist
    vxelec = vx[:]
    vyelec = vy[:]
    vzelec = vz[:]

    print("done! saving to pickle")
    os.system('mkdir analysisfiles')
    distdata = {'ionxxs':ionxxs,'elecxxs':elecxxs,'ionyys':ionyys,'elecyys':elecyys,'elechists':elechists,'elechistfac':elechistfac,'elechistboxfac':elechistboxfac,'ionhists':ionhists,'ionhistfac':ionhistfac,'ionhistboxfac':ionhistboxfac,'vxion':vxion,'vyion':vyion,'vzion':vzion,'vxelec':vxelec,'vyelec':vyelec,'vzelec':vzelec,'nx':nx,'ny':ny}
    picklefile = 'analysisfiles/tempvspos_frame100.pickle'
    with open(picklefile, 'wb') as handle:
        pickle.dump(distdata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved pickle to ',picklefile)
    precomputedflnm = picklefile

print("Loading from file: ", precomputedflnm)
filein = open(precomputedflnm, 'rb')
distdata = pickle.load(filein)
filein.close()
print('done!')

#reconstruct vx, vy, vz 3d arrays
vx_in = distdata['vxion']
vy_in = distdata['vyion']
vz_in = distdata['vzion']
_vx = np.zeros((len(vz_in),len(vy_in),len(vx_in)))
_vy = np.zeros((len(vz_in),len(vy_in),len(vx_in)))
_vz = np.zeros((len(vz_in),len(vy_in),len(vx_in)))
for i in range(0,len(vx_in)):
    for j in range(0,len(vy_in)):
        for k in range(0,len(vz_in)):
            _vx[k][j][i] = vx_in[i]
for i in range(0,len(vx_in)):
    for j in range(0,len(vy_in)):
        for k in range(0,len(vz_in)):
            _vy[k][j][i] = vy_in[j]
for i in range(0,len(vx_in)):
    for j in range(0,len(vy_in)):
        for k in range(0,len(vz_in)):
            _vz[k][j][i] = vz_in[k]
vxion = np.asarray(_vx)
vyion = np.asarray(_vy)
vzion = np.asarray(_vz)

vx_in = distdata['vxelec']
vy_in = distdata['vyelec']
vz_in = distdata['vzelec']
_vx = np.zeros((len(vz_in),len(vy_in),len(vx_in)))
_vy = np.zeros((len(vz_in),len(vy_in),len(vx_in)))
_vz = np.zeros((len(vz_in),len(vy_in),len(vx_in)))
for i in range(0,len(vx_in)):
    for j in range(0,len(vy_in)):
        for k in range(0,len(vz_in)):
            _vx[k][j][i] = vx_in[i]
for i in range(0,len(vx_in)):
    for j in range(0,len(vy_in)):
        for k in range(0,len(vz_in)):
            _vy[k][j][i] = vy_in[j]
for i in range(0,len(vx_in)):
    for j in range(0,len(vy_in)):
        for k in range(0,len(vz_in)):
            _vz[k][j][i] = vz_in[k]
vxelec = np.asarray(_vx)
vyelec = np.asarray(_vy)
vzelec = np.asarray(_vz)
#Note: vz<->vpar vy<->vperp2 vx<->vperp1

#compute 1D quants
print('computing and plotting 1d quants...')
ionparlocalfac = []
ionparboxfac = []
ionperplocalfac = []
ionperpboxfac = []
elecparlocalfac = []
elecparboxfac = []
elecperplocalfac = []
elecperpboxfac = []
iondens = []
elecdens = []
Tion_pred_idealadia = []
Tion_pred_doubleadia = []
Telec_pred_idealadia = []
Telec_pred_doubleadia = []
Btot = []
for _xidx in range(len(distdata['ionxxs'])):
    #project out data    
    ionhist1d = np.sum(distdata['ionhists'][_xidx],axis=0)
    elechist1d = np.sum(distdata['elechists'][_xidx],axis=0)
    ionhistlocalfac1d = np.sum(distdata['ionhistfac'][_xidx],axis=0)
    elechistlocalfac1d = np.sum(distdata['elechistfac'][_xidx],axis=0)
    ionhistboxfac1d = np.sum(distdata['ionhistboxfac'][_xidx],axis=0)
    elechistboxfac1d = np.sum(distdata['elechistboxfac'][_xidx],axis=0)

    idens = np.sum(ionhist1d)
    edens = np.sum(elechist1d)
    iondens.append(idens)
    elecdens.append(edens)
   
    if(idens != 0):
        vximeanlocal = np.sum(vxion*ionhistlocalfac1d)/idens
        vyimeanlocal = np.sum(vyion*ionhistlocalfac1d)/idens
        vzimeanlocal = np.sum(vzion*ionhistlocalfac1d)/idens

        vximeanbox = np.sum(vxion*ionhistboxfac1d)/idens
        vyimeanbox = np.sum(vyion*ionhistboxfac1d)/idens
        vzimeanbox = np.sum(vzion*ionhistboxfac1d)/idens

        vxemeanlocal = np.sum(vxelec*elechistlocalfac1d)/edens
        vyemeanlocal = np.sum(vyelec*elechistlocalfac1d)/edens
        vzemeanlocal = np.sum(vzelec*elechistlocalfac1d)/edens

        vxemeanbox = np.sum(vxelec*elechistboxfac1d)/edens
        vyemeanbox = np.sum(vyelec*elechistboxfac1d)/edens
        vzemeanbox = np.sum(vzelec*elechistboxfac1d)/edens

        ionparlocalfac.append(np.sum((vzion-vzimeanlocal)*(vzion-vzimeanlocal)*ionhistlocalfac1d)/idens)
        ionparboxfac.append(np.sum((vzion-vzimeanbox)*(vzion-vzimeanbox)*ionhistboxfac1d)/idens)
        ionperplocalfac.append(np.sum(((vxion-vximeanlocal)**2+(vyion-vyimeanlocal)**2)*ionhistlocalfac1d/idens))
        ionperpboxfac.append(np.sum(((vxion-vximeanbox)**2+(vyion-vyimeanbox)**2)*ionhistboxfac1d/idens))
    else:
        ionparlocalfac.append(0)
        ionparboxfac.append(0)
        ionperplocalfac.append(0)
        ionperpboxfac.append(0)

    if(edens != 0):
        elecparlocalfac.append(np.sum((vzelec-vzemeanlocal)*(vzelec-vzemeanlocal)*elechistlocalfac1d)/edens)
        elecparboxfac.append(np.sum((vzelec-vzemeanbox)*(vzelec-vzemeanbox)*elechistboxfac1d)/edens)
        elecperplocalfac.append(np.sum(((vxelec-vxemeanlocal)**2+(vyelec-vyemeanlocal)**2)*elechistlocalfac1d/edens))
        elecperpboxfac.append(np.sum(((vxelec-vxemeanbox)**2+(vyelec-vyemeanbox)**2)*elechistboxfac1d/edens))
    else:
        elecparlocalfac.append(0.)
        elecparboxfac.append(0.)
        elecperplocalfac.append(0.)
        elecperpboxfac.append(0.)

    _xidxbval = ao.find_nearest(dfields['bx_xx'],distdata['ionxxs'][_xidx]) #Note: the input to ao.avg_dict should match the input used to create the loaded dataset, this is a quick approximate fix in case that is not true
    btotval = np.mean(np.sqrt(dfields['bx'][:,:,_xidxbval]**2+dfields['by'][:,:,_xidxbval]**2+dfields['bz'][:,:,_xidxbval]**2))
    Btot.append(btotval)

    T0ion = 1.
    T0elec = 1.
    gammaval = 5./3.
    Tion_pred_idealadia.append(T0ion*(idens)**(5./3.-1.))
    Telec_pred_idealadia.append(T0elec*(edens)**(5./3.-1.))

    Tion_pred_doubleadia.append(btotval)
    Telec_pred_doubleadia.append(btotval)

#TODO Normalize to upstream quants
def normalize_rightmost_nonzero(arr):
    rightmost_nonzero = None
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] != 0:
            rightmost_nonzero = arr[i-70] #-70 is to swim upstream a bit for good measure, as the first few nonzero values may be on the 'edge' of the simulation, i.e. in a nonphysical region just before the physical upstream region
            break

    normalized_arr = [val / rightmost_nonzero for val in arr]
    return np.asarray(normalized_arr)


ionparlocalfac = normalize_rightmost_nonzero(ionparlocalfac)
ionparboxfac = normalize_rightmost_nonzero(ionparboxfac)
ionperplocalfac = normalize_rightmost_nonzero(ionperplocalfac)
ionperpboxfac = normalize_rightmost_nonzero(ionperpboxfac)
elecparlocalfac = normalize_rightmost_nonzero(elecparlocalfac)
elecparboxfac = normalize_rightmost_nonzero(elecparboxfac)
elecperplocalfac = normalize_rightmost_nonzero(elecperplocalfac)
elecperpboxfac = normalize_rightmost_nonzero(elecperpboxfac)
iondens = normalize_rightmost_nonzero(iondens)
elecdens = normalize_rightmost_nonzero(elecdens)
Tion_pred_idealadia = normalize_rightmost_nonzero(Tion_pred_idealadia)
Tion_pred_doubleadia = normalize_rightmost_nonzero(Tion_pred_doubleadia)
Telec_pred_idealadia = normalize_rightmost_nonzero(Telec_pred_idealadia)
Telec_pred_doubleadia = normalize_rightmost_nonzero(Telec_pred_doubleadia)
Btot = normalize_rightmost_nonzero(Btot)

pdatashort = {'ionparlocalfac': ionparlocalfac,
              'ionparboxfac': ionparboxfac,
              'ionperplocalfac':ionperplocalfac,
              'ionperpboxfac':ionperpboxfac,
              'elecparlocalfac':elecparlocalfac,
              'elecparboxfac':elecparboxfac,
              'elecperplocalfac':elecperplocalfac,
              'elecperpboxfac':elecperpboxfac,
              'iondens':iondens,
              'elecdens':elecdens,
              'Tion_pred_idealadia':Tion_pred_idealadia,
              'Tion_pred_doubleadia':Tion_pred_doubleadia,
              'Telec_pred_idealadia':Telec_pred_idealadia,
              'Telec_pred_doubleadia':Telec_pred_doubleadia,
              'Btot':Btot,
              'elecxxs':distdata['elecxxs'],
              'ionxxs':distdata['ionxxs']}
picklefile = 'analysisfiles/tempvspos_short_frame100.pkl'
with open(picklefile, 'wb') as handle:
    pickle.dump(pdatashort, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Saved short pickle to ',picklefile)

def plot1dquant(xarr,yarr,ylabel,flnm):
    plt.figure(figsize=(10,4))
    plt.style.use("cb.mplstyle")
    plt.plot(xarr,yarr,color='black')
    plt.xlabel(r'$x / d_i$')
    plt.ylabel(ylabel)
#    plt.xlim(5,12)
    plt.grid()
    plt.gca().tick_params(axis='both', direction='in',top=True, bottom=True, right=True, left=True)
    plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
    plt.close()

os.system('mkdir figures')
os.system('mkdir figures/tempvspos_frame100')

flnmprefix = 'figures/tempvspos_frame100/'

flnm = flnmprefix+'ionparlocalfac.png'
ylabel = r'$T_{||,i,local}/T_{||,0,i}$'
plot1dquant(distdata['ionxxs'],ionparlocalfac,ylabel,flnm)

flnm = flnmprefix+'ionparboxfac.png'
ylabel = r'$T_{||,i,box}/T_{||,0,i}$'
plot1dquant(distdata['ionxxs'],ionparboxfac,ylabel,flnm)

flnm = flnmprefix+'ionperplocalfac.png'
ylabel = r'$T_{\perp,i,local}/T_{\perp,,0,i}$'
plot1dquant(distdata['ionxxs'],ionperplocalfac,ylabel,flnm)

flnm = flnmprefix+'ionperpboxfac.png'
ylabel = r'$T_{\perp,i,box}/T_{\perp,0,i}$'
plot1dquant(distdata['ionxxs'],ionperpboxfac,ylabel,flnm)

flnm = flnmprefix+'elecparlocalfac.png'
ylabel = r'$T_{||,e,local}/T_{||,0,e}$'
plot1dquant(distdata['elecxxs'],elecparlocalfac,ylabel,flnm)

flnm = flnmprefix+'elecparboxfac.png'
ylabel = r'$T_{||,e,box}/T_{||,0,e}$'
plot1dquant(distdata['elecxxs'],elecparboxfac,ylabel,flnm)

flnm = flnmprefix+'elecperplocalfac.png'
ylabel = r'$T_{\perp,e,local}/T_{\perp,0,e}$'
plot1dquant(distdata['elecxxs'],elecperplocalfac,ylabel,flnm)

flnm = flnmprefix+'elecperpboxfac.png'
ylabel = r'$T_{\perp,e,box}/T_{\perp,0,e}$'
plot1dquant(distdata['elecxxs'],elecperpboxfac,ylabel,flnm)

flnm = flnmprefix+'iondens.png'
ylabel = r'$n_i/n_{0,i}$'
plot1dquant(distdata['ionxxs'],iondens,ylabel,flnm)

flnm = flnmprefix+'elecdens.png'
ylabel = r'$n_e/n_{0,e}$'
plot1dquant(distdata['elecxxs'],elecdens,ylabel,flnm)

flnm = flnmprefix+'Tion_pred_idealadia.png'
ylabel = r'$T_{\gamma=5/3,i}/T_{0,i}$'
plot1dquant(distdata['ionxxs'],Tion_pred_idealadia,ylabel,flnm)

flnm = flnmprefix+'Tion_pred_doubleadia.png'
ylabel = r'$T_{double,i}/T_{0,i}$'
plot1dquant(distdata['ionxxs'],Tion_pred_doubleadia,ylabel,flnm)

flnm = flnmprefix+'Telec_pred_idealadia.png'
ylabel = r'$T_{\gamma=5/3,e}/T_{0,e}$'
plot1dquant(distdata['elecxxs'],Telec_pred_idealadia,ylabel,flnm)

flnm = flnmprefix+'Telec_pred_doubleadia.png'
ylabel = r'$T_{double,e}/T_{0,e}$'
plot1dquant(distdata['elecxxs'],Telec_pred_doubleadia,ylabel,flnm)

flnm = flnmprefix+'Btot.png'
ylabel = r'$|B|/|B_0|$'
plot1dquant(distdata['ionxxs'],Btot,ylabel,flnm)

#make 1d plots with multiple lines
plt.figure(figsize=(10,6))
plt.style.use("cb.mplstyle")
plt.plot(distdata['ionxxs'],ionparlocalfac,label=r'$T_{||,i,local}/T_{||,0,i}$',color='red',ls='dashed')
plt.plot(distdata['ionxxs'],ionparboxfac,label=r'$T_{||,i,box}/T_{||,0,i}$',color='blue',ls='dashdot')
plt.plot(distdata['ionxxs'],ionperplocalfac,label=r'$T_{\perp,i,local}/T_{\perp,0,i}$',color='orange',ls='dashed')
plt.plot(distdata['ionxxs'],ionperpboxfac,label=r'$T_{\perp,i,box}/T_{\perp,0,i}$',color='green',ls='dashdot')
plt.plot(distdata['ionxxs'],iondens,label=r'$n_i/n_{0,i}$',color='black',ls='solid')
plt.plot(distdata['ionxxs'],Tion_pred_idealadia,label=r'$T_{\gamma=5/3,i}/T_{0,i}$',color='purple',ls='dotted')
plt.plot(distdata['ionxxs'],Tion_pred_doubleadia,label=r'$T_{double,i}/T_{0,i}$',color='gray',ls='dotted')
plt.xlabel(r'$x / d_i$')
plt.legend()
plt.xlim(0,12)
plt.grid()
plt.gca().tick_params(axis='both', direction='in',top=True, bottom=True, right=True, left=True)
plt.savefig(flnmprefix+'1diontempvspos.png',format='png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,6))
plt.style.use("cb.mplstyle")
plt.plot(distdata['elecxxs'],elecparlocalfac,label=r'$T_{||,e,local}/T_{||,0,e}$',color='red',ls='dashed')
plt.plot(distdata['elecxxs'],elecparboxfac,label=r'$T_{||,e,box}/T_{||,0,e}$',color='blue',ls='dashdot')
plt.plot(distdata['elecxxs'],elecperplocalfac,label=r'$T_{\perp,e,local}/T_{\perp,0,e}$',color='orange',ls='dashed')
plt.plot(distdata['elecxxs'],elecperpboxfac,label=r'$T_{\perp,e,box}/T_{\perp,0,e}$',color='green',ls='dashdot')
plt.plot(distdata['elecxxs'],elecdens,label=r'$n_e/n_{0,e}$',color='black',ls='solid')
plt.plot(distdata['elecxxs'],Telec_pred_idealadia,label=r'$T_{\gamma=5/3,e}/T_{0,e}$',color='purple',ls='dotted')
plt.plot(distdata['elecxxs'],Telec_pred_doubleadia,label=r'$T_{double,e}/T_{0,e}$',color='gray',ls='dotted')
plt.xlabel(r'$x / d_i$')
plt.legend()
plt.xlim(5,12)
plt.grid()
plt.gca().tick_params(axis='both', direction='in',top=True, bottom=True, right=True, left=True)
plt.savefig(flnmprefix+'1delectempvspos.png',format='png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,6))
plt.style.use("cb.mplstyle")
plt.plot(distdata['elecxxs'],elecparlocalfac,label=r'$T_{||,e,local}/T_{||,0,e}$',color='red',ls='dashed')
plt.plot(distdata['elecxxs'],elecparboxfac,label=r'$T_{||,e,box}/T_{||,0,e}$',color='blue',ls='dashdot')
plt.plot(distdata['elecxxs'],elecperplocalfac,label=r'$T_{\perp,e,local}/T_{\perp,0,e}$',color='orange',ls='dashed')
plt.plot(distdata['elecxxs'],elecperpboxfac,label=r'$T_{\perp,e,box}/T_{\perp,0,e}$',color='green',ls='dashdot')
plt.plot(distdata['elecxxs'],elecdens,label=r'$n_e/n_{0,e}$',color='black',ls='solid')
plt.plot(distdata['elecxxs'],Telec_pred_idealadia,label=r'$T_{\gamma=5/3,e}/T_{0,e}$',color='purple',ls='dotted')
plt.plot(distdata['elecxxs'],Telec_pred_doubleadia,label=r'$T_{double,e}/T_{0,e}$',color='gray',ls='dotted')
plt.xlabel(r'$x / d_i$')
plt.legend()
plt.xlim(7,9)
plt.grid()
plt.gca().tick_params(axis='both', direction='in',top=True, bottom=True, right=True, left=True)
plt.savefig(flnmprefix+'1delectempvspos_zoomed_in.png',format='png',dpi=300,bbox_inches='tight')
plt.close()


plt.figure(figsize=(10,6))
plt.style.use("cb.mplstyle")
plt.plot(distdata['ionxxs'],np.gradient(ionparlocalfac),label=r'$\frac{d}{dx}T_{||,i,local}/T_{||,0,i}$',color='red',ls='dashed')
plt.plot(distdata['ionxxs'],np.gradient(ionparboxfac),label=r'$\frac{d}{dx}T_{||,i,box}/T_{||,0,i}$',color='blue',ls='dashdot')
plt.plot(distdata['ionxxs'],np.gradient(ionperplocalfac),label=r'$\frac{d}{dx}T_{\perp,i,local}/T_{\perp,0,i}$',color='orange',ls='dashed')
plt.plot(distdata['ionxxs'],np.gradient(ionperpboxfac),label=r'$\frac{d}{dx}T_{\perp,i,box}/T_{\perp,0,i}$',color='green',ls='dashdot')
plt.plot(distdata['ionxxs'],np.gradient(iondens),label=r'$\frac{d}{dx}n_i/n_{0,i}$',color='black',ls='solid')
plt.plot(distdata['ionxxs'],np.gradient(Tion_pred_idealadia),label=r'$\frac{d}{dx}T_{\gamma=5/3,i}/T_{0,i}$',color='purple',ls='dotted')
plt.plot(distdata['ionxxs'],np.gradient(Tion_pred_doubleadia),label=r'$\frac{d}{dx}T_{double,i}/T_{0,i}$',color='gray',ls='dotted')
plt.xlabel(r'$x / d_i$')
plt.legend()
plt.xlim(5,12)
plt.grid()
plt.gca().tick_params(axis='both', direction='in',top=True, bottom=True, right=True, left=True)
plt.savefig(flnmprefix+'1dratesiontempvspos.png',format='png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,6))
plt.style.use("cb.mplstyle")
plt.plot(distdata['elecxxs'],np.gradient(elecparlocalfac),label=r'$\frac{d}{dx}T_{||,e,local}/T_{||,0,e}$',color='red',ls='dashed')
plt.plot(distdata['elecxxs'],np.gradient(elecparboxfac),label=r'$\frac{d}{dx}T_{||,e,box}/T_{||,0,e}$',color='blue',ls='dashdot')
plt.plot(distdata['elecxxs'],np.gradient(elecperplocalfac),label=r'$\frac{d}{dx}T_{perp,e,local}/T_{\perp,0,e}$',color='orange',ls='dashed')
plt.plot(distdata['elecxxs'],np.gradient(elecperpboxfac),label=r'$\frac{d}{dx}T_{perp,e,box}/T_{\perp,0,e}$',color='green',ls='dashdot')
plt.plot(distdata['elecxxs'],np.gradient(elecdens),label=r'$\frac{d}{dx}n_e/n_{0,e}$',color='black',ls='solid')
plt.plot(distdata['elecxxs'],np.gradient(Telec_pred_idealadia),label=r'$\frac{d}{dx}T_{\gamma=5/3,e}/T_{0,e}$',color='purple',ls='dotted')
plt.plot(distdata['elecxxs'],np.gradient(Telec_pred_doubleadia),label=r'$\frac{d}{dx}T_{double,e}/T_{0,e}$',color='gray',ls='dotted')
plt.xlabel(r'$x / d_i$')
plt.legend(loc ="upper right")
plt.xlim(5,12)
plt.grid()
plt.gca().tick_params(axis='both', direction='in',top=True, bottom=True, right=True, left=True)
plt.savefig(flnmprefix+'1drateselectempvspos.png',format='png',dpi=300,bbox_inches='tight')
plt.close()

#compute 2D quants
print("computing 2d quants")
nx=distdata['nx'] 
ny=distdata['ny']
ionparlocalfac2D = [[0 for _2 in range(ny)] for _ in range(nx)]
ionparboxfac2D = [[0 for _2 in range(ny)] for _ in range(nx)]
ionperplocalfac2D = [[0 for _2 in range(ny)] for _ in range(nx)]
ionperpboxfac2D = [[0 for _2 in range(ny)] for _ in range(nx)]
elecparlocalfac2D = [[0 for _2 in range(ny)] for _ in range(nx)]
elecparboxfac2D = [[0 for _2 in range(ny)] for _ in range(nx)]
elecperplocalfac2D = [[0 for _2 in range(ny)] for _ in range(nx)]
elecperpboxfac2D = [[0 for _2 in range(ny)] for _ in range(nx)]
iondens2D = [[0 for _2 in range(ny)] for _ in range(nx)]
elecdens2D = [[0 for _2 in range(ny)] for _ in range(nx)]
Tion_pred_idealadia2D = [[0 for _2 in range(ny)] for _ in range(nx)]
Tion_pred_doubleadia2D = [[0 for _2 in range(ny)] for _ in range(nx)]
Telec_pred_idealadia2D = [[0 for _2 in range(ny)] for _ in range(nx)]
Telec_pred_doubleadia2D = [[0 for _2 in range(ny)] for _ in range(nx)]
Btot2D = [[0 for _2 in range(ny)] for _ in range(nx)]

for _xidx in range(len(distdata['ionxxs'])):
    for _yidx in range(len(distdata['ionyys'])):
        #project out data    
        ionhist2d = distdata['ionhists'][_xidx][_yidx]
        elechist2d = distdata['elechists'][_xidx][_yidx]
        ionhistlocalfac2d = distdata['ionhistfac'][_xidx][_yidx]
        elechistlocalfac2d = distdata['elechistfac'][_xidx][_yidx]
        ionhistboxfac2d = distdata['ionhistboxfac'][_xidx][_yidx]
        elechistboxfac2d = distdata['elechistboxfac'][_xidx][_yidx]

        idens = np.sum(ionhist2d)
        edens = np.sum(elechist2d)
        iondens2D[_xidx][_yidx] = (idens)
        elecdens2D[_xidx][_yidx] = (edens)

        if(idens != 0):
            vximeanlocal = np.sum(vxion*ionhistlocalfac2d)/idens
            vyimeanlocal = np.sum(vyion*ionhistlocalfac2d)/idens
            vzimeanlocal = np.sum(vzion*ionhistlocalfac2d)/idens

            vximeanbox = np.sum(vxion*ionhistboxfac2d)/idens
            vyimeanbox = np.sum(vyion*ionhistboxfac2d)/idens
            vzimeanbox = np.sum(vzion*ionhistboxfac2d)/idens

            vxemeanlocal = np.sum(vxelec*elechistlocalfac2d)/edens
            vyemeanlocal = np.sum(vyelec*elechistlocalfac2d)/edens
            vzemeanlocal = np.sum(vzelec*elechistlocalfac2d)/edens

            vxemeanbox = np.sum(vxelec*elechistboxfac2d)/edens
            vyemeanbox = np.sum(vyelec*elechistboxfac2d)/edens
            vzemeanbox = np.sum(vzelec*elechistboxfac2d)/edens

            ionparlocalfac2D[_xidx][_yidx] = (np.sum((vzion-vzimeanlocal)*(vzion-vzimeanlocal)*ionhistlocalfac2d)/idens)
            ionparboxfac2D[_xidx][_yidx] = (np.sum((vzion-vzimeanbox)*(vzion-vzimeanbox)*ionhistboxfac2d)/idens)
            ionperplocalfac2D[_xidx][_yidx] = (np.sum(((vxion-vximeanlocal)**2+(vyion-vyimeanlocal)**2)*ionhistlocalfac2d/idens))
            ionperpboxfac2D[_xidx][_yidx] = (np.sum(((vxion-vximeanbox)**2+(vyion-vyimeanbox)**2)*ionhistboxfac2d/idens))
        else:
            ionparlocalfac2D[_xidx][_yidx] = (0)
            ionparboxfac2D[_xidx][_yidx] = (0)
            ionperplocalfac2D[_xidx][_yidx] = (0)
            ionperpboxfac2D[_xidx][_yidx] = (0)

        if(edens != 0):
            elecparlocalfac2D[_xidx][_yidx] = (np.sum((vzelec-vzemeanlocal)*(vzelec-vzemeanlocal)*elechistlocalfac2d)/edens)
            elecparboxfac2D[_xidx][_yidx] = (np.sum((vzelec-vzemeanbox)*(vzelec-vzemeanbox)*elechistboxfac2d)/edens)
            elecperplocalfac2D[_xidx][_yidx] = (np.sum(((vxelec-vxemeanlocal)**2+(vyelec-vyemeanlocal)**2)*elechistlocalfac2d/edens))
            elecperpboxfac2D[_xidx][_yidx] = (np.sum(((vxelec-vxemeanbox)**2+(vyelec-vyemeanbox)**2)*elechistboxfac2d/edens))
        else:
            elecparlocalfac2D[_xidx][_yidx] = (0.)
            elecparboxfac2D[_xidx][_yidx] = (0.)
            elecperplocalfac2D[_xidx][_yidx] = (0.)
            elecperpboxfac2D[_xidx][_yidx] = (0.)

        _xidxbval = ao.find_nearest(dfields['bx_xx'],distdata['ionxxs'][_xidx]) #Note: the input to ao.avg_dict should match the input used to create the loaded dataset, this is a quick approximate fix in case that is not true
        _yidxbval = ao.find_nearest(dfields['bx_yy'],distdata['ionxxs'][_yidx]) #Note: the input to ao.avg_dict should match the input used to create the loaded dataset, this is a quick approximate fix in case that is not true
        btotval = np.mean(np.sqrt(dfields['bx'][:,_yidxbval,_xidxbval]**2+dfields['by'][:,_yidxbval,_xidxbval]**2+dfields['bz'][:,_yidxbval,_xidxbval]**2))
        Btot2D[_xidx][_yidx] = btotval

        T0ion = 1.
        T0elec = 1.
        gammaval = 5./3.
        Tion_pred_idealadia2D[_xidx][_yidx] = (T0ion*(idens)**(5./3.-1.))
        Telec_pred_idealadia2D[_xidx][_yidx] = (T0elec*(edens)**(5./3.-1.))

        Tion_pred_doubleadia2D[_xidx][_yidx] =(btotval)
        Telec_pred_doubleadia2D[_xidx][_yidx] =(btotval)


#TODO Normalize to upstream quants
def normalize_rightmost_topmost_nonzero(arr):
    rightmost_nonzero = None
    for i in range(len(arr) - 1, -1, -1):
        for j in range(0,len(arr[i])):
            if arr[i][j] != 0:
                rightmosttopmost_nonzero = arr[i-70][j-20] #-20 is to swim upstream a bit for good measure, as the first few nonzero values may be on the 'edge' of the simulation, i.e. in a nonphysical region just before the physical upstream region
                break

    if(rightmosttopmost_nonzero == 0):
        print("error, rightmosttopmost_nonzero was zero, setting to 1")
        rightmosttopmost_nonzero = 1.

    normalized_arr = np.asarray(arr)/rightmosttopmost_nonzero
    return normalized_arr

ionparlocalfac2D = normalize_rightmost_topmost_nonzero(ionparlocalfac2D)
ionparboxfac2D = normalize_rightmost_topmost_nonzero(ionparboxfac2D)
ionperplocalfac2D = normalize_rightmost_topmost_nonzero(ionperplocalfac2D)
ionperpboxfac2D = normalize_rightmost_topmost_nonzero(ionperpboxfac2D)
elecparlocalfac2D = normalize_rightmost_topmost_nonzero(elecparlocalfac2D)
elecparboxfac2D = normalize_rightmost_topmost_nonzero(elecparboxfac2D)
elecperplocalfac2D = normalize_rightmost_topmost_nonzero(elecperplocalfac2D)
elecparboxfac2D = normalize_rightmost_topmost_nonzero(elecparboxfac2D)
iondens2D = normalize_rightmost_topmost_nonzero(iondens2D)
elecdens2D = normalize_rightmost_topmost_nonzero(elecdens2D)
Tion_pred_idealadia2D = normalize_rightmost_topmost_nonzero(Tion_pred_idealadia2D)
Tion_pred_doubleadia2D = normalize_rightmost_topmost_nonzero(Tion_pred_doubleadia2D)
Telec_pred_idealadia2D = normalize_rightmost_topmost_nonzero(Telec_pred_idealadia2D)
Telec_pred_doubleadia2D = normalize_rightmost_topmost_nonzero(Telec_pred_doubleadia2D)
Btot2D = normalize_rightmost_topmost_nonzero(Btot2D)

def center2edge(arr):
    darr = arr[1]-arr[0]
    arrout = np.zeros(len(arr)+1)
    for _i in range(len(arr)):
        arrout[_i] = arr[_i]-darr/2.
    arrout[-1]=arrout[-2]+darr
    return arrout

def plot2dquant(xarr,yarr,zarr,tlabel,flnm):
    plt.figure(figsize=(10,4))
    plt.xlabel(r'$x / d_i$')
    plt.ylabel(r'$y / d_i$')
    xarr = center2edge(xarr)
    yarr = center2edge(yarr)
    plt.pcolormesh(xarr,yarr,np.asarray(zarr).T,cmap='plasma') #shading='gourand' did not work, so we had to convert cell centers to edges
    plt.colorbar()
    plt.title(tlabel)
    plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
    plt.close()

print("plotting 2d quants")
flnm = flnmprefix+'2Dionparlocalfac.png'
ylabel = r'$T_{||,i,local}/T_{||,0,i}$'
plot2dquant(distdata['ionxxs'],distdata['ionyys'],ionparlocalfac2D,ylabel,flnm)

flnm = flnmprefix+'2Dionparboxfac.png'
ylabel = r'$T_{||,i,box}/T_{||,0,i}$'
plot2dquant(distdata['ionxxs'],distdata['ionyys'],ionparboxfac2D,ylabel,flnm)

flnm = flnmprefix+'2Dionperplocalfac.png'
ylabel = r'$T_{\perp,i,local}/T_{\perp,0,i}$'
plot2dquant(distdata['ionxxs'],distdata['ionyys'],ionperplocalfac2D,ylabel,flnm)

flnm = flnmprefix+'2Dionperpboxfac.png'
ylabel = r'$T_{\perp,i,box}/T_{\perp,0,i}$'
plot2dquant(distdata['ionxxs'],distdata['ionyys'],ionperpboxfac2D,ylabel,flnm)

flnm = flnmprefix+'2Delecparlocalfac.png'
ylabel = r'$T_{||,e,local}/T_{||,0,e}$'
plot2dquant(distdata['elecxxs'],distdata['elecyys'],elecparlocalfac2D,ylabel,flnm)

flnm = flnmprefix+'2Delecparboxfac.png'
ylabel = r'$T_{||,e,box}/T_{||,0,e}$'
plot2dquant(distdata['elecxxs'],distdata['elecyys'],elecparboxfac2D,ylabel,flnm)

flnm = flnmprefix+'2Delecperplocalfac.png'
ylabel = r'$T_{\perp,e,local}/T_{\perp,0,e}$'
plot2dquant(distdata['elecxxs'],distdata['elecyys'],elecperplocalfac2D,ylabel,flnm)

flnm = flnmprefix+'2Delecperpboxfac.png'
ylabel = r'$T_{\perp,e,box}/T_{\perp,0,e}$'
plot2dquant(distdata['elecxxs'],distdata['elecyys'],elecperpboxfac2D,ylabel,flnm)

flnm = flnmprefix+'2Diondens.png'
ylabel = r'$n_i/n_{0,i}$'
plot2dquant(distdata['ionxxs'],distdata['ionyys'],iondens2D,ylabel,flnm)

flnm = flnmprefix+'2Delecdens.png'
ylabel = r'$n_e/n_{0,e}$'
plot2dquant(distdata['elecxxs'],distdata['elecyys'],elecdens2D,ylabel,flnm)

flnm = flnmprefix+'2DTion_pred_idealadia.png'
ylabel = r'$T_{\gamma=5/3,i}/T_{0,i}$'
plot2dquant(distdata['ionxxs'],distdata['ionyys'],Tion_pred_idealadia2D,ylabel,flnm)

flnm = flnmprefix+'2DTion_pred_doubleadia.png'
ylabel = r'$T_{double,i}/T_{0,i}$'
plot2dquant(distdata['ionxxs'],distdata['ionyys'],Tion_pred_doubleadia2D,ylabel,flnm)

flnm = flnmprefix+'2DTelec_pred_idealadia.png'
ylabel = r'$T_{\gamma=5/3,e}/T_{0,e}$'
plot2dquant(distdata['elecxxs'],distdata['elecyys'],Telec_pred_idealadia2D,ylabel,flnm)

flnm = flnmprefix+'2DTelec_pred_doubleadia.png'
ylabel = r'$T_{double,e}/T_{0,e}$'
plot2dquant(distdata['elecxxs'],distdata['elecyys'],Telec_pred_doubleadia2D,ylabel,flnm)

flnm = flnmprefix+'2DBtot.png'
ylabel = r'$|B|/|B_0|$'
plot2dquant(distdata['ionxxs'],distdata['ionyys'],Btot2D,ylabel,flnm)
