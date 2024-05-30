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


import pickle
#filteredflnm = '/data/backed_up/analysis/collbrown/nonadiaperp/fftnofiltfilteredfields.pickle'
#filein = open(filteredflnm, 'rb')
#dfluc = pickle.load(filein)

inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
inputs = ld.load_input(inputpath,verbose=True)
print("***** inputs *****")
print(inputs)






#user params
framenum = '700' #frame to make figure of (should be a string)
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]
vmaxion = 30
dvion = 1.
vmaxelec = 15
dvelec = 1.

#params = ld.load_params(flpath,framenum)
#print('sigma', params['sigma'])

#print(params)
print("DONE!!!!")
normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)
dden = ld.load_den(flpath,framenum)
for _key in dden.keys():
    dfields[_key] = dden[_key]

print("dfields shape: ", dfields['ex'].shape)
print("np mean: ", np.mean(dfields['by']))
print('dfields yy', dfields['ex_yy'])


params = ld.load_params(flpath,framenum)
dt = params['c']/params['comp'] #in units of wpe
c = params['c']
stride = 100
dt,c = aa.norm_constants(params,dt,c,stride)

normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)
vshock = 1.5625375379202415

print('dfields yy', dfields['ex_yy'])
print("Warning: using hard coded value of vshock to save computational time...")
print("vshock = ",vshock)
dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame
dfluc = aa.remove_average_fields_over_yz(dfields)

startidx_xx = ao.find_nearest(dfields['ex_xx'],5.)
startidx_yy = 0
startidx_zz = 0
endidx_xx = ao.find_nearest(dfields['ex_xx'],10.)
endidx_yy = len(dfields['ex_yy'])
endidx_zz = len(dfields['ex_zz'])
startidxs = [startidx_zz,startidx_yy,startidx_xx]
endidxs = [endidx_zz,endidx_yy,endidx_xx]
dfields = ao.subset_dict(dfields,startidxs,endidxs,planes=['z','y','x'])
dfluc = ao.subset_dict(dfluc,startidxs,endidxs,planes=['z','y','x'])
dfields = ao.avg_dict(dfields,binidxsz=[1,1,10],planes=['z','y','x'])
dfluc = ao.avg_dict(dfluc,binidxsz=[1,1,10],planes=['z','y','x'])

sampleidx = ao.find_nearest(dfields['ex_xx'],8.5)
my_array = dfluc['ex'][0,:,sampleidx]

import pickle
picklefile = 'testdata.pickle'
with open(picklefile, 'wb') as handle:
        pickle.dump(my_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

np.savetxt('testdata.txt', my_array)
print("done writing sample!")

#import pickle
import pickle
#filteredflnm = '/data/backed_up/analysis/collbrown/nonadiaperp/nofiltfilteredfields.pickle'
#filein = open(filteredflnm, 'rb')
#dfluc = pickle.load(filein)
#print("dfluc['ex_yy']: ", dfluc['ex_yy'])
#dfluc['Vframe_relative_to_sim']=0.
#filein.close()

#print("dfluc grid: ", len(dfluc['ex_yy']), dfluc['ex_yy'][1]-dfluc['ex_yy'][0])
#print("dfluc shape: ", dfluc['ex'].shape)

normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)
dfields = ao.truncate_dict(dfields,reducfrac=[1,2,2],planes=['z','y','x'])
dfields = ao.avg_dict(dfields,binidxsz=[1,2,10],planes=['z','y','x'])
print("dfields grid: ", len(dfields['ex_yy']), dfields['ex_yy'][1]-dfields['ex_yy'][0])
print("dfields shape: ", dfields['ex'].shape)

inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
inputs = ld.load_input(inputpath,verbose=True)
print("***** inputs *****")
print(inputs)

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

print(params.keys())
uinj = 2.3245E-02 #from input file (gamma0) 
beta0 = aa.compute_beta0(params,uinj)

print("beta0,",beta0)

vti0 = np.sqrt(params['delgam'])#Note: velocity is in units γV_i/c so we do not include '*params['c']'
vte0 = np.sqrt(params['mi']/params['me'])*vti0 #WARNING: THIS ASSUME Ti/Te = 1, TODO: don't assume Ti/Te = 1

print("vte0: ", vte0)

print("Te: ", vte0**2*params['me']/2.)
print("Ti: ", vti0**2*params['mi']/2.)

print('vae: ',vte0/(beta0/2.)) #assumes Te=Ti
print('va: ', vti0/(beta0/2.)) #assumes Te=Ti 

params = ld.load_params(flpath,framenum)
dt = params['c']/params['comp'] #in units of wpe
c = params['c']
stride = 100
dt,c = aa.norm_constants(params,dt,c,stride)

num = '100'
massratio = ld.load_params(flpath,num)['mi']/ld.load_params(flpath,num)['me']

print(params['comp']*np.sqrt(massratio),dt,uinj)
