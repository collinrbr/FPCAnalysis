import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa

def track_fig(dfieldsdict, xshockvalsubset,dt,startidx):
    xxindex = 0
    yyindex = 0 
    zzindex = 0
    fieldkey = 'ex'
    axis = '_xx'
    offsetval = 4 #adjust this number as needed to scale size of y axis and spacing...
    flnm = 'nonadiaperpshocktrack'
    from matplotlib.pyplot import cm
    color = cm.rainbow(np.linspace(0, 1, len(dfieldsdict['dfields'])))

    #fig, axs = plt.subplots(len(dfieldsdict['frame']), sharex=True, sharey=True)
    fig = plt.figure()
    fieldcoord = np.asarray(dfieldsdict['dfields'][0][fieldkey+axis])
    fig.set_size_inches(9., 9.)

    plt.style.use('cb.mplstyle')

    _i = 0
    for k in range(0,len(dfieldsdict['dfields'])):
        fieldval = np.asarray([dfieldsdict['dfields'][k][fieldkey][xxindex][yyindex][i] for i in range(0,len(dfieldsdict['dfields'][k][fieldkey+axis]))])+_i*offsetval
        plt.plot(fieldcoord,fieldval,color='black',zorder=2)
        _i += 1
 
    _i = 0
    for k in range(0,len(dfieldsdict['dfields'])):
        plt.scatter(xshockvalsubset[k],_i*offsetval,color='blue',s=45,zorder=3)
        _i += 1
    
    #build line from shock vel data
    offsetarray = []
    fitxshockval = []
    for k in range(-3,len(dfieldsdict['dfields'])+5):
        #use fit
        fitxshockval.append(v0+(k+startidx)*vshock*dt)
        offsetarray.append((k)*offsetval) 
    plt.plot(fitxshockval,offsetarray,color='red',linewidth=2, zorder=10) 
    plt.ylabel('time', fontsize=25)
    plt.xlabel('$x/d_i$', fontsize=25)
    plt.xticks(fontsize=15)
    plt.yticks([])
    plt.grid()
    #plt.xlim(10,40)
    if(flnm == ''):
        plt.show()
    else:
        fig.tight_layout()
        plt.savefig(flnm+'.png',dpi=300,format='png',transparent=False,facecolor='white')
    plt.close()

    pass

#------------------------------------------------------------------------------------------------------------------------------------
# Begin script
#------------------------------------------------------------------------------------------------------------------------------------

#TODO: save shock val and load everywhere else (add to compartimentalized part of code where we load the same stuff in almost every script

#user params
framenum = '700' #frame to make figure of (should be a string)
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)] #frames to compute velocity with
plotframes = frames[:] #frames to plot

normalize = True
dfields = ld.load_fields(flpath,framenum,normalizeFields=normalize)

params = ld.load_params(flpath,framenum)
dt = params['c']/params['comp'] #in units of wpe
c = params['c']
stride = 100
dt,c = aa.norm_constants(params,dt,c,stride)

#compute shock velocity
dfields_many_frames = {'frame':[],'dfields':[]}
dfieldsdict = {'frame':[],'dfields':[]}
for _num in frames:
    num = int(_num)
    d = ld.load_fields(flpath,_num,normalizeFields=normalize)
    dfields_many_frames['dfields'].append(d)
    dfields_many_frames['frame'].append(num)
    if (_num in plotframes):
        dfieldsdict['dfields'].append(d)
        dfieldsdict['frame'].append(num)
vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)
xshockvalsubset = []
for _i,_num in enumerate(frames):
    if(_num in plotframes):
        xshockvalsubset.append(xshockvals[_i])
startidx=int(frames[0]) #idx of the first frame

#note: we don't boost to shock rest frame for field vals
flnmpmesh = 'shocktrack'
print("Making shock track fig of...",flpath," in simulation frame")
track_fig(dfieldsdict, xshockvalsubset, dt, startidx)

print("Computed a vshock of: ", vshock)
