import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa

def plot_1d_field(dfields,key,yindex,flnm):
    fieldcoord = np.asarray(dfields[fieldkey+'_xx'][:])
    fieldval = np.asarray(dfields[fieldkey][0,yindex,:])

    plt.figure(figsize=(20,10))
    plt.style.use('cb.mplstyle')
    plt.xlabel(r'$x/d_i$')
    if(key == 'ey'):
        plt.ylabel(r'$E_y/(B_0 va/c)$')
    else:
        plt.ylabel(key)
    plt.plot(fieldcoord,fieldval)
    plt.grid()
    plt.savefig(flnm+'.png',format='png')
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

fieldkey = 'ey'
flnm = 'nonadia'+fieldkey+'slice'
print("Making 1d field plot of...",flpath)
yindex = 0
plot_1d_field(dfields,fieldkey,yindex,flnm)



