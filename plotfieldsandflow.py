#!/usr/bin/env python

#Set path to data (TODO: automate getting path. Perhaps load these in a seperate input file)
#paths to data and time frame
path_particles = "Output/Raw/Sp01/raw_sp01_{:08d}.h5"
path_fields = ""
numframe = 1000 #used when naming files. Grabs the 1000th frame file

#-------------------------------------------------------------------------------
#needed libraries
#-------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import h5py
import glob
import numpy as np

#debug libraries
import time
from sys import getsizeof
printruntime = True

#-------------------------------------------------------------------------------
# define functions
#-------------------------------------------------------------------------------
#Function to load fields
def field_loader(field_vars='all', components='all', num=None,
                 path='./', slc=None, verbose=False):
    if(slc != None):
        print("Warning: taking slices of field data is currently unavailable. TODO: fix")
        return {}


    _field_choices_ = {'B':'Magnetic',
                       'E':'Electric',
                       'J':'CurrentDens'}
    _ivc_ = {v: k for k, v in iter(_field_choices_.items())}
    if components == 'all':
        components = 'xyz'
    if(len(path)>0):
        if path[-1] is not '/': path = path + '/'
    fpath = path+"Output/Fields/*"
    if field_vars == 'all':
        field_vars = [c[len(fpath)-1:] for c in glob.glob(fpath)]
        field_vars = [_ivc_[k] for k in field_vars]
    else:
        if isinstance(field_vars, basestring):
            field_vars = field_vars.upper().split()
        elif not type(field_vars) in (list, tuple):
            field_vars = [field_vars]
    if slc is None:
        slc = np.s_[:,:]
    fpath = path+"Output/Fields/{f}/{T}{c}/{v}fld_{t}.h5"
    T = '' if field_vars[0] == 'J' else 'Total/'
    test_path = fpath.format(f = _field_choices_[field_vars[0]],
                             T = T,
                             c = 'x',
                             v = field_vars[0],
                             t = '*')
    if verbose: print(test_path)
    choices = glob.glob(test_path)
    #num_of_zeros = len()
    choices = [int(c[-11:-3]) for c in choices]
    choices.sort()
    fpath = fpath.format(f='{f}', T='{T}', c='{c}', v='{v}', t='{t:08d}')
    d = {}
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(input(_))
    for k in field_vars:
        T = '' if k == 'J' else 'Total/'
        for c in components:
            ffn = fpath.format(f = _field_choices_[k],
                               T = T,
                               c = c,
                               v = k,
                               t = num)
            kc = k.lower()+c
            if verbose: print(ffn)
            with h5py.File(ffn,'r') as f:
                d[kc] = np.asarray(f['DATA'][slc],order='F')
                d[kc] = np.ascontiguousarray(d[kc])
                _N3,_N2,_N1 = f['DATA'].shape #python is fliped.
                x1,x2,x3 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:], f['AXIS']['X3 AXIS'][:] #TODO: double check that x1->xx x2->yy x3->zz
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                dx3 = (x3[1]-x3[0])/_N3
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
                d[kc+'_zz'] = dx3*np.arange(_N3) + dx3/2. + x3[0]
                d[kc+'_xx'] = d[kc+'_xx']#[slc[1]]
                d[kc+'_yy'] = d[kc+'_yy']#[slc[0]]
                d[kc+'_zz'] = d[kc+'_zz']#[slc[0]]  #TODO: check if this is correct. Dont understand the variable slc or how it's used
    return d

def flow_loader(flow_vars=None, num=None, path='./', sp=1, verbose=False):
    import glob
    if path[-1] is not '/': path = path + '/'
    choices = num#get_output_times(path=path, sp=sp, output_type='flow')
    dpath = path+"Output/Phase/FluidVel/Sp{sp:02d}/{dv}/Vfld_{tm:08}.h5"
    d = {}
    if type(flow_vars) is str:
        flow_vars = flow_vars.split()
    elif flow_vars is None:
        flow_vars = 'x y z'.split()
    #print(dpath.format(sp=sp, tm=num))
    for k in flow_vars:
        if verbose: print(dpath.format(sp=sp, dv=k, tm=num))
        with h5py.File(dpath.format(sp=sp, dv=k, tm=num),'r') as f:
            kc = 'u'+k
            _ = f['DATA'].shape #python is fliped
            dim = len(_)
            print(kc,_)
            d[kc] = f['DATA'][:]
            if dim < 3:
                _N2,_N1 = _
                x1,x2 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
            else:
                _N3,_N2,_N1 = _
                x1 = f['AXIS']['X1 AXIS'][:]
                x2 = f['AXIS']['X2 AXIS'][:]
                x3 = f['AXIS']['X3 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                dx3 = (x3[1]-x3[0])/_N3
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
                d[kc+'_zz'] = dx3*np.arange(_N3) + dx3/2. + x3[0]
    _id = "{}:{}:{}".format(os.path.abspath(path), num, "".join(flow_vars))
    d['id'] = _id
    return d

def plot_all_fields(dfields, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0):
    if(axis == '_zz'):
        ex = np.asarray([dfields['ex'][i][yyindex][zzindex] for i in range(0,len(dfields['ex'+axis]))])
        ey = np.asarray([dfields['ey'][i][yyindex][zzindex] for i in range(0,len(dfields['ey'+axis]))])
        ez = np.asarray([dfields['ez'][i][yyindex][zzindex] for i in range(0,len(dfields['ez'+axis]))])
        bx = np.asarray([dfields['bx'][i][yyindex][zzindex] for i in range(0,len(dfields['bx'+axis]))])
        by = np.asarray([dfields['by'][i][yyindex][zzindex] for i in range(0,len(dfields['by'+axis]))])
        bz = np.asarray([dfields['bz'][i][yyindex][zzindex] for i in range(0,len(dfields['bz'+axis]))])
    elif(axis == '_yy'):
        ex = np.asarray([dfields['ex'][xxindex][i][zzindex] for i in range(0,len(dfields['ex'+axis]))])
        ey = np.asarray([dfields['ey'][xxindex][i][zzindex] for i in range(0,len(dfields['ex'+axis]))])
        ez = np.asarray([dfields['ez'][xxindex][i][zzindex] for i in range(0,len(dfields['ex'+axis]))])
        bx = np.asarray([dfields['bx'][xxindex][i][zzindex] for i in range(0,len(dfields['bx'+axis]))])
        by = np.asarray([dfields['by'][xxindex][i][zzindex] for i in range(0,len(dfields['by'+axis]))])
        bz = np.asarray([dfields['bz'][xxindex][i][zzindex] for i in range(0,len(dfields['bz'+axis]))])
    elif(axis == '_xx'):
        ex = np.asarray([dfields['ex'][xxindex][yyindex][i] for i in range(0,len(dfields['ex'+axis]))])
        ey = np.asarray([dfields['ey'][xxindex][yyindex][i] for i in range(0,len(dfields['ex'+axis]))])
        ez = np.asarray([dfields['ez'][xxindex][yyindex][i] for i in range(0,len(dfields['ex'+axis]))])
        bx = np.asarray([dfields['bx'][xxindex][yyindex][i] for i in range(0,len(dfields['bx'+axis]))])
        by = np.asarray([dfields['by'][xxindex][yyindex][i] for i in range(0,len(dfields['by'+axis]))])
        bz = np.asarray([dfields['bz'][xxindex][yyindex][i] for i in range(0,len(dfields['bz'+axis]))])

    fieldcoord = np.asarray(dfields['ex'+axis]) #assumes all fields have same coordinates

    fig, axs = plt.subplots(6,figsize=(20,10))
    axs[0].plot(fieldcoord,ex,label="ex")
    axs[0].set_ylabel("$ex$")
    axs[1].plot(fieldcoord,ey,label='ey')
    axs[1].set_ylabel("$ey$")
    axs[2].plot(fieldcoord,ez,label='ez')
    axs[2].set_ylabel("$ez$")
    axs[3].plot(fieldcoord,bx,label='bx')
    axs[3].set_ylabel("$bx$")
    axs[4].plot(fieldcoord,by,label='by')
    axs[4].set_ylabel("$by$")
    axs[5].plot(fieldcoord,bz,label='bz')
    axs[5].set_ylabel("$bz$")
    if(axis == '_xx'):
        axs[5].set_xlabel("$x$")
    elif(axis == '_yy'):
        axs[5].set_xlabel("$y$")
    elif(axis == '_yy'):
        axs[5].set_xlabel("$z$")
    plt.subplots_adjust(hspace=0.5)
    plt.savefig('fieldsalongxx.png',format='png')
    plt.close()

def plot_all_flow(dflow, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0, flnm = ''):
    if(axis == '_zz'):
        ux = np.asarray([dflow['ux'][i][yyindex][zzindex] for i in range(0,len(dflow['ux'+axis]))])
        uy = np.asarray([dflow['uy'][i][yyindex][zzindex] for i in range(0,len(dflow['uy'+axis]))])
        uz = np.asarray([dflow['uz'][i][yyindex][zzindex] for i in range(0,len(dflow['uz'+axis]))])
    elif(axis == '_yy'):
        ux = np.asarray([dflow['ux'][xxindex][i][zzindex] for i in range(0,len(dflow['ux'+axis]))])
        uy = np.asarray([dflow['uy'][xxindex][i][zzindex] for i in range(0,len(dflow['uy'+axis]))])
        uz = np.asarray([dflow['uz'][xxindex][i][zzindex] for i in range(0,len(dflow['uz'+axis]))])
    elif(axis == '_xx'):
        ux = np.asarray([dflow['ux'][xxindex][yyindex][i] for i in range(0,len(dflow['ux'+axis]))])
        uy = np.asarray([dflow['uy'][xxindex][yyindex][i] for i in range(0,len(dflow['uy'+axis]))])
        uz = np.asarray([dflow['uz'][xxindex][yyindex][i] for i in range(0,len(dflow['uz'+axis]))])

    fieldcoord = np.asarray(dflow['ux'+axis]) #assumes all fields have same coordinates

    fig, axs = plt.subplots(3,figsize=(20,10))
    axs[0].plot(fieldcoord,ux,label="vx")
    axs[0].set_ylabel("$ux$")
    axs[1].plot(fieldcoord,uy,label='vy')
    axs[1].set_ylabel("$uy$")
    axs[2].plot(fieldcoord,uz,label='vz')
    axs[2].set_ylabel("$uz$")
    if(axis == '_xx'):
        axs[2].set_xlabel("$x$")
    elif(axis == '_yy'):
        axs[2].set_xlabel("$y$")
    elif(axis == '_yy'):
        axs[2].set_xlabel("$z$")
    plt.subplots_adjust(hspace=0.5)
    plt.savefig('flowplotsalongxx.png',format=png)

#-------------------------------------------------------------------------------
# Load and make plot
#-------------------------------------------------------------------------------
#Load fields
if(printruntime):
    start_time = time.time()
dfields = field_loader(path=path_fields,num=numframe)
if(printruntime):
    print("Time to load fields: %s seconds " % (time.time() - start_time))

#Load fields
if(printruntime):
    start_time = time.time()
dfields = field_loader(path=path_fields,num=numframe)
if(printruntime):
    print("Time to load fields: %s seconds " % (time.time() - start_time))

#make plots
plot_all_fields(dfields, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0)
plot_all_flow(dflow, num=numframe, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0))
