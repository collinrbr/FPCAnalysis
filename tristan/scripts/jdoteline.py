import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa
import lib.arrayaux as ao #array operations

import pickle
import os

#------------------------------------------------------------------------------------------------------------------------------------
# Begin script
#------------------------------------------------------------------------------------------------------------------------------------
#user params
framenum = '700' #frame to make figure of (should be a string)
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]

loadflow = True
dflowflnm = 'pickles/dflow.pickle'

normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)

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
vshock = 1.5

dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

#dpar_elec, dpar_ion = ld.load_particles(flpath,framenum,normalizeVelocity=normalize)
#inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
#inputs = ld.load_input(inputpath)
#beta0 = aa.compute_beta0(params,inputs)

print("Loading flow")
if(loadflow):
    filein = open(dflowflnm, 'rb')
    dflow = pickle.load(filein)
    filein.close()
else:
    dpar_ion = ft.shift_particles(dpar_ion, vshock, beta0, params['mi']/params['me'], isIon=True)
    dpar_elec = ft.shift_particles(dpar_elec, vshock, beta0, params['mi']/params['me'], isIon=False)
    dflow = aa.compute_dflow(dfields, dpar_ion, dpar_elec)

    os.system('mkdir pickles')
    picklefile = 'pickles/dflow.pickle'
    with open(picklefile, 'wb') as handle:
        pickle.dump(dflow, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved pickle to ',picklefile)
    precomputedflnm = picklefile
print("Done!")

import os
os.system('mkdir figures')
os.system('mkdir figures/jdotelines')

dfluc = aa.remove_average_fields_over_yz(dfields)
dflowfluc = aa.remove_average_cur_over_yz(dflow)

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
dflow = ao.subset_dict(dflow,startidxs,endidxs,planes=['z','y','x'])
dflowfluc = ao.subset_dict(dflowfluc,startidxs,endidxs,planes=['z','y','x'])
dfields = ao.avg_dict(dfields,binidxsz=[1,2,2],planes=['z','y','x'])
dflow = ao.avg_dict(dflow,binidxsz=[1,2,2],planes=['z','y','x'])
dfluc = ao.avg_dict(dfluc,binidxsz=[1,2,2],planes=['z','y','x'])
dflowfluc = ao.avg_dict(dflowfluc,binidxsz=[1,2,2],planes=['z','y','x'])

dfluclocal = aa.convert_fluc_to_local_par(dfields,dfluc)
dflucbox = aa.convert_fluc_to_par(dfields,dfluc)
dflowfluclocal = aa.convert_flowfluc_to_local_par(dfields,dflow,dflowfluc)
dflowflucbox = aa.convert_flowfluc_to_par(dfields,dflow,dflowfluc)
dfieldslocal = aa.convert_to_local_par(dfields)
dfieldsbox = aa.convert_to_par(dfields)
dflowlocal = aa.convert_flow_to_local_par(dfields,dflow)
dflowbox = aa.convert_flow_to_par(dfields,dflow)
datadicts = [[dfluclocal,dflowfluclocal],[dflucbox,dflowflucbox],[dfieldslocal,dflowlocal],[dfieldsbox,dflowbox]]
xvals  = [7.5,7.6,7.7,7.8,7.9,8.0,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9]

print("Filtering fields...")
datadictslowpass = []
kycutoff = 15.
filterabove = True
for _idx in range(0,len(datadicts)):
    _dfld = aa.yz_fft_filter(datadicts[_idx][0],kycutoff,filterabove,dontfilter=False,verbose=False,keys=['epar','eperp1','eperp2']) #WARNING: ONLY FILTERING KEYS WE USE!! 
    _dflw = aa.yz_fft_filter(datadicts[_idx][1],kycutoff,filterabove,dontfilter=False,verbose=False,keys=['jpare','jperp1e','jperp2e']) #WARNING: ONLY FILTERING KEYS WE USE!!
    datadictslowpass.append([_dfld,_dflw])
print("Done with lowpass!")

datadictshighpass = []
filterabove = False
kycutoff = 15.
for _idx in range(0,len(datadicts)):
    _dfld = aa.yz_fft_filter(datadicts[_idx][0],kycutoff,filterabove,dontfilter=False,verbose=False,keys=['epar','eperp1','eperp2']) #WARNING: ONLY FILTERING KEYS WE USE!! 
    _dflw = aa.yz_fft_filter(datadicts[_idx][1],kycutoff,filterabove,dontfilter=False,verbose=False,keys=['jpare','jperp1e','jperp2e']) #WARNING: ONLY FILTERING KEYS WE USE!!
    datadictshighpass.append([_dfld,_dflw])
print("Done with highpass!")

datadict_collection = [datadicts,datadictslowpass,datadictshighpass]

def plotlineslice(xval,yvals,plotval,plotlabel,flnmtag):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,figsize=(10,4))
    plt.style.use('cb.mplstyle')
    ax.plot(yvals,plotval,color='black')
    ax.grid(True)
    ax.set_xlabel(r'$y/d_i$')
    ax.set_ylabel(plotlabel)
    ttl = '$\int$'+plotlabel+'$dy=$'+str('{:0.3e}'.format(np.sum(plotval)*(yvals[1]-yvals[0])))
    ax.set_title(ttl,loc='right')
    plt.savefig(flnmtag+'.png',format='png',dpi=300,bbox_inches='tight')
    plt.close()

tabledict = {} #Table to hold integrated values
for _xval in xvals:
    os.system('mkdir figures/jdotelines/'+str(_xval))

    xxidx = ao.find_nearest(dfluc['ex_xx'],_xval)

    plotkeys = ['epar','eperp1','eperp2','jpare','jperp1e','jperp2e','jparepar','jperp1eperp1','jperp2eperp2']
    labelprefix = []
    labelpostfix = []
    for filteridx in range(0,3):
        if(filteridx == 0):
            pstfx0 = ''
            flnmtagprefx0 = 'unfilt'
        elif(filteridx == 1):
            pstfx0 = '^{k<k_0}'
            flnmtagprefx0 = 'lowpass'
        elif(filteridx == 2):
            pstfx0 = '^{k>k_0}'
            flnmtagprefx0 = 'highpass'

        _dictpairidx = 0
        for ddictpair in datadict_collection[filteridx]:
            pstfx = pstfx0
            flnmtagprefx = flnmtagprefx0
            if(_dictpairidx == 0):
                prfx = '$\widetilde{'
                pstfx = ',local}'+pstfx+'}$'
                flnmtagprefx = flnmtagprefx+'fluclocal'
            elif(_dictpairidx == 1):
                prfx = '$\widetilde{'
                pstfx = ',box}'+pstfx+'}$'
                flnmtagprefx = flnmtagprefx+'flucbox'
            elif(_dictpairidx == 2):
                prfx = '$'
                pstfx = ',local}'+pstfx+'$'
                flnmtagprefx = flnmtagprefx+'local'
            elif(_dictpairidx == 3):
                prfx = '$'
                pstfx = ',box}'+pstfx+'$'
                flnmtagprefx = flnmtagprefx+'box'
            else:
                print("Error!!!")

            for pkey in plotkeys: 
                #grab quant from keys
                if pkey == 'epar':
                    plotlabel = prfx + 'E_{||' + pstfx
                    plotval = ddictpair[0]['epar'][0,:,xxidx]
                    flnmtag = flnmtagprefx+pkey
                if pkey == 'eperp1':
                    plotlabel = prfx + 'E_{\perp,1' + pstfx
                    plotval = ddictpair[0]['eperp1'][0,:,xxidx]
                    flnmtag = flnmtagprefx+pkey
                if pkey == 'eperp2':
                    plotlabel = prfx + 'E_{\perp,2' + pstfx
                    plotval = ddictpair[0]['eperp2'][0,:,xxidx]
                    flnmtag = flnmtagprefx+pkey
                if pkey == 'jpare':
                    plotlabel = prfx + 'j_{||,e' + pstfx
                    plotval = -1.*ddictpair[1]['jpare'][0,:,xxidx]
                    flnmtag = flnmtagprefx+pkey
                if pkey == 'jperp1e':
                    plotlabel = prfx + 'j_{\perp,1,e' + pstfx
                    plotval = -1.*ddictpair[1]['jperp1e'][0,:,xxidx]
                    flnmtag = flnmtagprefx+pkey
                if pkey == 'jperp2e':
                    plotlabel = prfx + 'j_{\perp,2,e' + pstfx
                    plotval = -1.*ddictpair[1]['jperp2e'][0,:,xxidx]
                    flnmtag = flnmtagprefx+pkey
                if pkey == 'jparepar':
                    plotlabel = prfx + 'j_{||,e '+pstfx + prfx+'E_{||'+pstfx
                    plotval = -1.*ddictpair[0]['epar'][0,:,xxidx]*ddictpair[1]['jpare'][0,:,xxidx]
                    flnmtag = flnmtagprefx+pkey
                if pkey == 'jperp1eperp1':
                    plotlabel = prfx + 'j_{\perp,1,e '+pstfx + prfx+'E_{\perp,1'+pstfx
                    plotval = -1.*ddictpair[0]['eperp1'][0,:,xxidx]*ddictpair[1]['jperp1e'][0,:,xxidx]
                    flnmtag = flnmtagprefx+pkey
                if pkey == 'jperp2eperp2':
                    plotlabel = prfx + 'j_{\perp,2,e '+pstfx + prfx+'E_{\perp,2'+pstfx
                    plotval = -1.*ddictpair[0]['eperp2'][0,:,xxidx]*ddictpair[1]['jperp2e'][0,:,xxidx]
                    flnmtag = flnmtagprefx+pkey

                tabledict[str(_xval)+flnmtag] = np.sum(plotval) 
                yvals = dfields['ex_yy'][:]
                flnmtag = 'figures/jdotelines/'+str(_xval)+'/'+flnmtag
                plotlineslice(_xval,yvals,plotval,plotlabel,flnmtag)

            _dictpairidx += 1


#print table of metrics!!!
for tkey in tabledict.keys():
    print(tkey + ' ' + str(tabledict[tkey]))

import pickle

with open('jdotelinevals.pickle', 'wb') as handle:
    pickle.dump(tabledict, handle, protocol=pickle.HIGHEST_PROTOCOL)

