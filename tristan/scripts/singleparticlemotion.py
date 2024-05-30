import sys
import os
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa
import lib.parpathaux as pp

import pickle

def plot_particle_path(datapath,dfields,ind,proc,startframenum,endframenum,flnm='parpath.png',plane='xy',spec='ion',verbose=False):
    stride = 10
    parpath = pp.get_path_of_particle(datapath,ind,proc,startframenum,endframenum,spec=spec,stride=stride,verbose=verbose)
    
    import matplotlib.pyplot as plt
    try: #if particle not found, just skip everything (quick fix)
        if(plane == 'xy'):
            plotx = parpath['x'][:]
            ploty = parpath['y'][:]
            xlabel = r'$x$'
            ylabel = r'$y$'
        
        plotframe = np.arange(int(startframenum),int(endframenum)+1,stride)#TODO: load sim time
    
        from matplotlib.collections import LineCollection
        cols = plotframe

        points = np.array([plotx, ploty]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, ax = plt.subplots()
        lc = LineCollection(segments, cmap='Blues')
        lc.set_array(cols)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
    
        plt.gca().set_aspect('equal')
        fig.colorbar(line,ax=ax)

        print("DEBUG:",np.min(dfields['ex_xx']),np.max(dfields['ex_xx']),np.min(dfields['ex_yy']),np.max(dfields['ex_yy']))
        ax.set_xlim(np.min(dfields['ex_xx']),np.max(dfields['ex_xx']))
        ax.set_ylim(np.min(dfields['ex_yy']),np.max(dfields['ex_yy']))
    
        plt.savefig(flnm,format='png',dpi=300)
        plt.close()
    except:
        pass

def plot_many_paths(parpaths,dfields,flnm='parpaths.png',plane='xy',spec='ion',verbose=False):
    from matplotlib.collections import LineCollection
    plotframe = np.arange(int(startframenum),int(endframenum)+1,stride)#TODO: load sim time
    cols = plotframe

    fig, ax = plt.subplots()
    
    for parpath in parpaths:
        if(plane == 'xy'):
            plotx = parpath['x'][:]
            ploty = parpath['y'][:]
            xlabel = r'$x/d_i$'
            ylabel = r'$y/d_i$'

        points = np.array([plotx, ploty]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='Blues')
        lc.set_array(cols)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

    plt.gca().set_aspect('equal')
    fig.colorbar(line,ax=ax)

    ax.set_xlim(np.min(dfields['ex_xx']),np.max(dfields['ex_xx']))
    ax.set_ylim(np.min(dfields['ex_yy']),np.max(dfields['ex_yy']))
    ax.grid()

    plt.savefig(flnm,format='png',dpi=300)
    plt.close()

#user params
framenum = '001'
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]

startframenum = '001' 
endframenum = '700'

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


ind_partilces = []
proc_particles = []
xlowbound = 9.19
xupbound = 9.2
ylowbound = 0.59
yupbound = 0.6
for _i in range(0,len(dpar_elec['xe'])):
    if(xlowbound<dpar_elec['xe'][_i]<xupbound):
        if(ylowbound<dpar_elec['ye'][_i]<yupbound):
            ind_partilces.append(dpar_elec['inde'][_i])
            proc_particles.append(dpar_elec['proce'][_i])
print("Found ", len(ind_partilces), "particles in region!!!")

print("Loading par path!")
parpaths = []
verbose = False
stride = 10
for _i in range(0,len(ind_partilces)):
    if(_i % 10 == 0): print("Loaded: ",_i," of ", len(ind_partilces), " par paths...")

    ind = ind_partilces[_i]
    proc = proc_particles[_i]
    parpath = pp.get_path_of_particle(flpath,ind,proc,startframenum,endframenum,spec='elec',stride=stride,verbose=verbose)

    if parpath is not None:
        parpaths.append(parpath)

file = open('parpaths.pickle', 'wb')
pickle.dump(parpaths, file)
file.close()

plot_many_paths(parpaths,dfields,flnm='parpaths.png',plane='xy',spec='ion',verbose=False)

#----------
#Block that makes a bunch of random particle trajectories
#----------
#os.system('mkdir figures')
#os.system('mkdir figures/parpathfigs')
#for _i in range(0,len(dpar_elec['xe']),250):
#    #pick particle (particles are tracked by what processor produced them (proce/proci) and id number assigned at creation equal to number of particles made by generating processor before it +1 (inde/indi)
#    ind_particle = dpar_elec['inde'][_i]
#    proc_particle = dpar_elec['proce'][_i]
#
#    flnm = 'figures/parpathfigs/parpath_ind'+str(ind_particle)+'proc'+str(proc_particle)+'.png'
#    plot_particle_path(flpath,dfields,ind_particle,proc_particle,startframenum,endframenum,flnm=flnm,plane='xy',spec='elec',verbose=True)
