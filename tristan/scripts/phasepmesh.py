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

def phaseplot(phasedata,key,poscoords,velcoords,flnm,xlim0=None,xlim1=None):

    #make plots of fields
    fig = plt.figure(figsize=(10,3))

    plt.style.use('cb.mplstyle')

    #uxe
    plt.pcolormesh(poscoords,velcoords,phasedata[key].T, cmap="cividis", shading="gouraud")
    plt.colorbar() #TODO: log scale option for color bar

    #print axes labels
    plt.xlabel('$x/d_i$')
    if(key=='uxi'):
        plt.ylabel('$v_{x,i}/v_{th,i}$')
    elif(key=='uyi'):
        plt.ylabel('$v_{y,i}/v_{th,i}$')
    elif(key=='uzi'):
        plt.ylabel('$v_{z,i}/v_{th,i}$')
    elif(key=='uxe'):
        plt.ylabel('$v_{x,e}/v_{th,i}$')
    elif(key=='uye'):
        plt.ylabel('$v_{y,e}/v_{th,i}$')
    elif(key=='uze'):
        plt.ylabel('$v_{z,e}/v_{th,i}$')

    #set xlim
    if(xlim0 != None and xlim1 != None):
        plt.xlim(xlim0,xlim1)

    plt.grid()

    plt.savefig(flnm+'.png',format='png',dpi=400,bbox_inches='tight')
    plt.close()

#------------------------------------------------------------------------------------------------------------------------------------
# Begin script
#------------------------------------------------------------------------------------------------------------------------------------
#user params
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
#vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)
print("warning... using hard coded value of vshock to save time")
vshock = 1.5
dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

dpar_elec, dpar_ion = ld.load_particles(flpath,framenum,normalizeVelocity=normalize)
inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
inputs = ld.load_input(inputpath)
beta0 = aa.compute_beta0(params,inputs)

dpar_ion = ft.shift_particles(dpar_ion, vshock, beta0, params['mi']/params['me'], isIon=True)
dpar_elec = ft.shift_particles(dpar_elec, vshock, beta0, params['mi']/params['me'], isIon=False)

#bin particles
nx = len(dfields['ex_xx'])
ion_bins = [[] for _ in range(nx)]
elec_bins = [[] for _ in range(nx)]

debug = True
for _i in range(0,int(len(dpar_ion['xi'])/1000)): #change /1 to /n to load 1/n of the data 
    if(debug and _i % 100000 == 0): print("Binned: ", _i," ions of ", len(dpar_ion['xi']))
    xx = dpar_ion['xi'][_i]
    xidx = ao.find_nearest(dfields['ex_xx'], xx)
    ion_bins[xidx].append({'ui':dpar_ion['ui'][_i] ,'vi':dpar_ion['vi'][_i] ,'wi':dpar_ion['wi'][_i]})

for _i in range(0,int(len(dpar_elec['xe'])/1)):
    if(debug and _i % 100000 == 0): print("Binned: ", _i," elecs of ", len(dpar_elec['xe']))
    xx = dpar_elec['xe'][_i]
    xidx = ao.find_nearest(dfields['ex_xx'], xx)
    elec_bins[xidx].append({'ue':dpar_elec['ue'][_i] ,'ve':dpar_elec['ve'][_i] ,'we':dpar_elec['we'][_i]})

vmaxion = 15
vmaxelec = 8
delvion = .25
delvelec = .25
ionbinsedges = np.arange(-vmaxion,vmaxion+delvion,delvion)
elecbinsedges = np.arange(-vmaxelec,vmaxelec+delvelec,delvelec)

ionvelcoords = [(ionbinsedges[_temp_i]+ionbinsedges[_temp_i+1])/2. for _temp_i in range(0,len(ionbinsedges)-1)]
elecvelcoords = [(elecbinsedges[_temp_i]+elecbinsedges[_temp_i+1])/2. for _temp_i in range(0,len(elecbinsedges)-1)]

phasedata = {}
keys = ['uxi','uyi','uzi','uxe','uye','uze']
for _ky in keys:
    if(_ky[-1] == 'i'):
        phasedata[_ky] = np.zeros((nx,len(ionbinsedges)-1))
    elif(_ky[-1] == 'e'):
        phasedata[_ky] = np.zeros((nx,len(elecbinsedges)-1))

print("binning particles")
for _idx in range(0,len(ion_bins)):
    phasedata['uxi'][_idx][:] = np.histogram([ion_bins[_idx][_jdx]['ui'] for _jdx in range(0,len(ion_bins[_idx]))],bins=ionbinsedges)[0] 
    phasedata['uyi'][_idx][:] = np.histogram([ion_bins[_idx][_jdx]['wi'] for _jdx in range(0,len(ion_bins[_idx]))],bins=ionbinsedges)[0]
    phasedata['uzi'][_idx][:] = np.histogram([ion_bins[_idx][_jdx]['vi'] for _jdx in range(0,len(ion_bins[_idx]))],bins=ionbinsedges)[0]

for _idx in range(0,len(elec_bins)):
    phasedata['uxe'][_idx][:] = np.histogram([elec_bins[_idx][_jdx]['ue'] for _jdx in range(0,len(elec_bins[_idx]))],bins=elecbinsedges)[0]
    phasedata['uye'][_idx][:] = np.histogram([elec_bins[_idx][_jdx]['we'] for _jdx in range(0,len(elec_bins[_idx]))],bins=elecbinsedges)[0]
    phasedata['uze'][_idx][:] = np.histogram([elec_bins[_idx][_jdx]['ve'] for _jdx in range(0,len(elec_bins[_idx]))],bins=elecbinsedges)[0]
    _idx += 1
print("done! now making figures")

import os
os.system('mkdir figures')
os.system('mkdir figures/phasepmeshes')

flnmpmesh = 'figures/phasepmeshes'
print("Making pmesh of total fields...",flpath)

for _ky in keys:
    flnmpmesh = 'figures/phasepmeshes/'+_ky
    print("Making pmesh of phase fields (location,key)...",flpath,_ky)
    if(_ky[-1] == 'i'):
        velcoords = ionvelcoords
    elif(_ky[-1] == 'e'):
        velcoords = elecvelcoords
    phaseplot(phasedata,_ky,dfields['ex_xx'],velcoords,flnmpmesh,xlim0=6,xlim1=10)

print("Making 1d plots of average and std...")
for _ky in keys:
    dividebyzerooffset = .00000001 #weights cant be exactly zero in np.average, so we add a small number 
    if(_ky[-1] == 'e'):
        phasedata[_ky+'_mean'] = np.asarray([np.average(elecvelcoords,weights=phasedata[_ky][_idx][:]+dividebyzerooffset) for _idx in range(0,len(phasedata[_ky][:]))])
    
        squared_diff = np.asarray([np.power(elecvelcoords - phasedata[_ky+'_mean'][_idx],2) for _idx in range(0,len(phasedata[_ky][:]))])
        weighted_variance = np.asarray([np.average(squared_diff[_idx],weights=phasedata[_ky][_idx][:]+dividebyzerooffset) for _idx in range(0,len(phasedata[_ky][:]))])
        phasedata[_ky+'_std'] = np.asarray([np.sqrt(weighted_variance[_idx]) for _idx in range(0,len(phasedata[_ky][:]))])

    elif(_ky[-1] == 'i'):
        phasedata[_ky+'_mean'] = np.asarray([np.average(ionvelcoords,weights=phasedata[_ky][_idx][:]+dividebyzerooffset) for _idx in range(0,len(phasedata[_ky][:]))])
    
        squared_diff = np.asarray([np.power(ionvelcoords - phasedata[_ky+'_mean'][_idx],2) for _idx in range(0,len(phasedata[_ky][:]))])
        weighted_variance = np.asarray([np.average(squared_diff[_idx],weights=phasedata[_ky][_idx][:]+dividebyzerooffset) for _idx in range(0,len(phasedata[_ky][:]))])
        phasedata[_ky+'_std'] = np.asarray([np.sqrt(weighted_variance[_idx]) for _idx in range(0,len(phasedata[_ky][:]))])

def sliding_window_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

window_size = 1
for _ky in phasedata.keys():
    if('std' in _ky or 'mean' in _ky):
        phasedata[_ky] = sliding_window_average(phasedata[_ky],window_size)

plt.figure(figsize=(10,3))
plt.plot(dfields['ex_xx'][window_size - 1:],phasedata['uxi_std'],color='red',ls=':',label='$\sigma_{u_{x,i}/v_{th,i}}$')
plt.plot(dfields['ex_xx'][window_size - 1:],phasedata['uyi_std'],color='green',ls='-.',label='$\sigma_{u_{y,i}/v_{th,e}}$')
plt.plot(dfields['ex_xx'][window_size - 1:],phasedata['uzi_std'],color='blue',ls='--',label='$\sigma_{u_{z,i}/v_{th,e}}$')
plt.grid()
plt.xlabel('$x/d_i$')
plt.legend()
plt.xlim(6,10)
plt.savefig('figures/phasepmeshes/ion_std.png',format='png',dpi=300)
plt.close()

plt.figure(figsize=(10,3))
plt.plot(dfields['ex_xx'][window_size - 1:],phasedata['uze_std'],color='blue',ls='--',label='$\sigma_{u_{z,e}/v_{th,e}}$')
plt.plot(dfields['ex_xx'][window_size - 1:],phasedata['uxe_std'],color='red',ls=':',label='$\sigma_{u_{x,e}/v_{th,e}}$')
plt.plot(dfields['ex_xx'][window_size - 1:],phasedata['uye_std'],color='green',ls='-.',label='$\sigma_{u_{y,e}/v_{th,e}}$')
plt.grid()
plt.xlabel('$x/d_i$')
plt.legend()
plt.xlim(7,9)
#plt.ylim(25,100)
plt.savefig('figures/phasepmeshes/elec_std.png',format='png',dpi=300)
plt.close()

plt.figure(figsize=(10,3))
plt.plot(dfields['ex_xx'][window_size - 1:],phasedata['uxi_mean'],color='red',ls=':',label='$<u_{x,i}/v_{th,i}>$')
plt.plot(dfields['ex_xx'][window_size - 1:],phasedata['uyi_mean'],color='green',ls='-.',label='$<u_{y,i}/v_{th,i}>$')
plt.plot(dfields['ex_xx'][window_size - 1:],phasedata['uzi_mean'],color='blue',ls='--',label='$<u_{z,i}/v_{th,i}>$')
plt.grid()
plt.xlabel('$x/d_i$')
plt.legend()
plt.xlim(6,10)
plt.savefig('figures/phasepmeshes/ion_mean.png',format='png',dpi=300)
plt.close()

plt.figure(figsize=(10,3))
plt.plot(dfields['ex_xx'][window_size - 1:],phasedata['uxe_mean'],color='red',ls=':',label='$<u_{x,e}/v_{th,e}>$')
plt.plot(dfields['ex_xx'][window_size - 1:],phasedata['uye_mean'],color='green',ls='-.',label='$<u_{y,e}/v_{th,e}>$')
plt.plot(dfields['ex_xx'][window_size - 1:],phasedata['uze_mean'],color='blue',ls='--',label='$<u_{z,e}/v_{th,e}>$')
plt.grid()
plt.xlabel('$x/d_i$')
plt.legend()
plt.xlim(6,10)
plt.savefig('figures/phasepmeshes/elec_mean.png',format='png',dpi=300)
plt.close()

