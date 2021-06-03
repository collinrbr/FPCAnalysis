#!/usr/bin/env python

#Set path to data (TODO: automate getting path. Perhaps load these in a seperate input file)
#paths to data and time frame
path_particles = "FPC_Colby/run0/Output/Raw/Sp01/raw_sp01_{:08d}.h5"
path_fields = "FPC_Colby/run0/"
numframe = 1000 #used when naming files. Grabs the 1000th frame file
flnm = 'DHybridRSDAtestonAlven1.nc' #output netcdf4 filename

#Set correlation bounds
vmax = 15.0 #uses square bounds for vx, vy
dv = 0.25

#physical simulation parameters (TODO: grab from simulation input automatically)
vinject = -5.0 #assumes injection velocity is along the x-axis

#dictionary that holds parameters related to simulation
#TODO: automate by loading relevant data from simulation input file
params = {}
params["MachAlfven"] = float('nan')
params["MachAlfvenNote"] = 'TODO: compute mach alfven for this run'
params["ShockNormalAngle"] = 0.0
params["betaelec"] = 1.0
params["betaion"] = 1.0
params["simtime"] = numframe
params["simtimeNote"] = 'This is frames number for this data set. TODO: convert to inverse Omega_c,i'
params["qi"] = 1.0
params["di"] = 0.0
params["dinote"] = 'TODO: compute ion inertial length'
params["vti"] = 1.0
params["metadataNote"] = '-1 implies undefined'

#-------------------------------------------------------------------------------
#needed libraries
#-------------------------------------------------------------------------------
import pickle
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

#Loads vx, vy, xx, yy data only.
#Sometimes this is necessary to save RAM
def readParticlesPosandVelocityOnly(path, num):
    pts = {}
    twoDdist_vars = 'p1 p2 x1 x2'.split() #only need to load velocity and space information for dist func
    with h5py.File(path.format(num),'r') as f:
        for k in twoDdist_vars:
            pts[k] = f[k][:]
    return pts

#takes lorentz transform where V=(vx,0,0)
#TODO: check if units work (in particular where did gamma go. Perhaps we just assume it's small)
def lorentz_transform_vx(dfields,vx):
    import copy
    dfieldslor = copy.copy(dfields) #deep copy

    dfieldslor['ex'] = dfields['ex']
    dfieldslor['ey'] = dfields['ey']-vx*dfields['bz']
    dfieldslor['ez'] = dfields['ez']+vx*dfields['by']
    dfieldslor['bx'] = dfields['bx']
    dfieldslor['by'] = dfields['by']#assume v/c^2 is small
    dfieldslor['bz'] = dfields['bz']#assume v/c^2 is small

    return dfieldslor

#makes distribution and takes correlation wrt given field
#WARNING: this will average the fields within the specified bounds.
#However, if there are no gridpoints within the specified bounds
#it will *not* grab the field value at the nearest gridpoint and break. TODO: grab nearest field when range is small
def make2dHistandCey(vmax, dv, x1, x2, y1, y2, dpar, dfields, fieldkey):

    #find average E field based on provided bounds
    gfieldptsx = (x1 <= dfields[fieldkey+'_xx'])  & (dfields[fieldkey+'_xx'] <= x2)
    gfieldptsy = (y1 <= dfields[fieldkey+'_yy']) & (dfields[fieldkey+'_yy'] <= y2)

    goodfieldpts = []
    for i in range(0,len(dfields['ex_xx'])):
        for j in range(0,len(dfields['ex_yy'])):
            for k in range(0,len(dfields['ex_zz'])):
                if(gfieldptsx[i] == True and gfieldptsy[j] == True):
                    goodfieldpts.append(dfields[fieldkey][k][j][i])

    #TODO?: consider forcing user to take correlation over only 1 cell
    if(len(goodfieldpts)==0):
        print("Using weighted_field_average...") #Debug
        avgfield = weighted_field_average((x1+x2)/2.,(y1+y2)/2.,0,dfields,fieldkey) #TODO: make 3d i.e. *don't* just 'project' all z information out and take fields at z = 0
    else:
        avgfield = np.average(goodfieldpts)
    totalFieldpts = np.sum(goodfieldpts)

    #define mask that includes particles within range
    gptsparticle = (x1 < dpar['x1'] ) & (dpar['x1'] < x2) & (y1 < dpar['x2']) & (dpar['x2'] < y2)
    totalPtcl = np.sum(gptsparticle)

    #make bins
    vxbins = np.arange(-vmax, vmax, dv)
    vx = (vxbins[1:] + vxbins[:-1])/2.
    vybins = np.arange(-vmax, vmax, dv)
    vy = (vybins[1:] + vybins[:-1])/2.

    #make the bins 2d arrays
    _vx = np.zeros((len(vy),len(vx)))
    _vy = np.zeros((len(vy),len(vx)))
    for i in range(0,len(vy)):
        for j in range(0,len(vx)):
            _vx[i][j] = vx[j]

    for i in range(0,len(vy)):
        for j in range(0,len(vx)):
            _vy[i][j] = vy[i]

    vx = _vx
    vy = _vy

    #find distribution
    Hxy,_,_ = np.histogram2d(dpar['p2'][gptsparticle],dpar['p1'][gptsparticle],
                         bins=[vybins, vxbins])

    #calculate correlation
    Cey = -0.5*vy**2*np.gradient(Hxy, dv, edge_order=2, axis=0)*avgfield
    return vx, vy, totalPtcl, totalFieldpts, Hxy, Cey

#makes distribution and takes correlation wrt given field
#WARNING: this will average the fields within the specified bounds.
#However, if there are no gridpoints within the specified bounds
#it will *not* grab the field value at the nearest gridpoint and break. TODO: grab nearest field when range is small
def make2dHistandCex(vmax, dv, x1, x2, y1, y2, dpar, dfields, fieldkey):

    #find average E field based on provided bounds
    gfieldptsx = (x1 <= dfields[fieldkey+'_xx'])  & (dfields[fieldkey+'_xx'] <= x2)
    gfieldptsy = (y1 <= dfields[fieldkey+'_yy']) & (dfields[fieldkey+'_yy'] <= y2)

    goodfieldpts = []
    for i in range(0,len(dfields['ex_xx'])):
        for j in range(0,len(dfields['ex_yy'])):
            for k in range(0,len(dfields['ex_zz'])):
                if(gfieldptsx[i] == True and gfieldptsy[j] == True):
                    goodfieldpts.append(dfields[fieldkey][k][j][i])

    if(len(goodfieldpts)==0):
        print("Warning, no field grid points in given box. Please increase box size or center around grid point.")

    avgfield = np.average(goodfieldpts)
    totalFieldpts = np.sum(goodfieldpts)

    #define mask that includes particles within range
    gptsparticle = (x1 < dpar['x1'] ) & (dpar['x1'] < x2) & (y1 < dpar['x2']) & (dpar['x2'] < y2)
    totalPtcl = np.sum(gptsparticle)

    #make bins
    vxbins = np.arange(-vmax, vmax, dv)
    vx = (vxbins[1:] + vxbins[:-1])/2.
    vybins = np.arange(-vmax, vmax, dv)
    vy = (vybins[1:] + vybins[:-1])/2.

    #make the bins 2d arrays
    _vx = np.zeros((len(vy),len(vx)))
    _vy = np.zeros((len(vy),len(vx)))
    for i in range(0,len(vy)):
        for j in range(0,len(vx)):
            _vx[i][j] = vx[j]

    for i in range(0,len(vy)):
        for j in range(0,len(vx)):
            _vy[i][j] = vy[i]

    vx = _vx
    vy = _vy

    #find distribution
    Hxy,_,_ = np.histogram2d(dpar['p2'][gptsparticle],dpar['p1'][gptsparticle],
                         bins=[vybins, vxbins])

    #calculate correlation
    Cex = -0.5*vx**2*np.gradient(Hxy, dv, edge_order=2, axis=1)*avgfield
    return vx, vy, totalPtcl, totalFieldpts, Hxy, Cex

def savedata(CEx_out, CEy_out, vx_out, vy_out, x_out, metadata_out = [], params = {}, filename = 'dHybridRSDAtest.nc' ):
    from netCDF4 import Dataset
    from datetime import datetime


    #normalize CEx, CEy to 1-------------------------------------------------------
    #Here we normalize to the maximum value in either CEx, CEy
    maxCval = max(np.amax(np.abs(CEx_out)),np.amax(np.abs(CEy_out)))
    CEx_out /= maxCval
    CEy_out /= maxCval


    # open a netCDF file to write
    ncout = Dataset(filename, 'w', format='NETCDF4')


    #save data in netcdf file-------------------------------------------------------
    #define attributes
    for key in params:
        setattr(ncout,key,params[key])
    ncout.description = 'dHybridR MLA data test 1'
    ncout.generationtime = str(datetime.now())

    #make dimensions that dependent data must 'match'
    ncout.createDimension('x', None)  # NONE <-> unlimited TODO: make limited if it saves memory or improves compression?
    ncout.createDimension('vx', None)
    ncout.createDimension('vy', None)

    vx = ncout.createVariable('vx','f4', ('vx',))
    vx.nvx = len(vx_out)
    vx.longname = 'v_x/v_ti'
    vx[:] = vx_out[:]

    vy = ncout.createVariable('vy','f4', ('vy',))
    vy.nvy = len(vy_out)
    vy.longname = 'v_y/v_ti'
    vy[:] = vy_out[:]

    x = ncout.createVariable('x','f4',('x',))
    x.nx = len(x_out)
    x[:] = x_out[:]

    #tranpose data to match previous netcdf4 formatting
    for i in range(0,len(CEx_out)):
        tempCex = CEx_out[i].T
        CEx_out[i] = tempCex
        tempCey = CEy_out[i].T
        CEy_out[i] = tempCey

    C_ex = ncout.createVariable('C_Ex','f4', ('x', 'vx', 'vy'))
    C_ex.longname = 'C_{Ex}'
    C_ex[:] = CEx_out[:]

    C_ey = ncout.createVariable('C_Ey','f4', ('x', 'vx', 'vy'))
    C_ey.longname = 'C_{Ey}'
    C_ey[:] = CEy_out[:]

    metadata = ncout.createVariable('metadata','f4',('x',))
    metadata.description = '1 = signature, 0 = no signature'
    metadata[:] = metadata_out[:]

    #Save data into netcdf4 file-----------------------------------------------------
    print("Saving data into netcdf4 file")

    #save file
    ncout.close()



#-------------------------------------------------------------------------------
# Do analysis
#-------------------------------------------------------------------------------
#Load fields
if(printruntime):
    start_time = time.time()
dfields = field_loader(path=path_fields,num=numframe)
if(printruntime):
    print("Time to load fields: %s seconds " % (time.time() - start_time))

#Load particle data
if(printruntime):
    start_time = time.time()
dparticles = readParticlesPosandVelocityOnly(path_particles, numframe)
if(printruntime):
    print("Time to run this: %s seconds " % (time.time() - start_time))

#Lorentz transform the fields
#see Juno 2021 (FPC Analysis of Perpendicular Collisionless shock pg 7)
print("Warning: calculating the shock velocity used to lorentz the transform required apriori knowledge of where the shock is. Please update compression ratio if not done already...")
r = 3.25
vshock = -vinject/(r-1) #TODO: check sign of vinject and vshock
dfields = lorentz_transform_vx(dfields,vshock)

#Sweep along x and run correlations
xsweep = 0.0
dx = dfields['ex_xx'][1]-dfields['ex_xx'][0]
dv = 0.25

CEx_out = []
CEy_out = []
x_out = []
Hxy_out = []

for i in range(0,len(dfields['ex_xx'])):
    if(printruntime):
        print(str(dfields['ex_xx'][i]) +' of ' + str(dfields['ex_xx'][len(dfields['ex_xx'])-1]))
    #to do full 'bulk analysis' we loop over one of these three blocks for all desired sections along x
    vx, vy, totalPtcl_quarterdi, totalFieldpts, Hxy_quarterdi, Cey_quarterdi = make2dHistandCey(vmax, dv, xsweep, xsweep+dx, dfields['ey_yy'][0], dfields['ey_yy'][1], dparticles, dfields, 'ey')
    vx, vy, totalPtcl_quarterdi, totalFieldpts, Hxy_quarterdi, Cex_quarterdi = make2dHistandCex(vmax, dv, xsweep, xsweep+dx, dfields['ex_xx'][0], dfields['ex_xx'][1], dparticles, dfields, 'ex')
    x_out.append(np.mean([xsweep,xsweep+dx]))
    CEy_out.append(Cey_quarterdi)
    CEx_out.append(Cex_quarterdi)
    Hxy_out.append(Hxy_quarterdi)
    xsweep+=dx

#save data
metadata = np.zeros(len(dfields['ex_xx']))
metadata = metadata.astype(int)
for i in range(0,len(metadata)):
    metadata[i] = -1 #set metadata to 'unknown'.

#note: it seems jupyter does not have permission to overwrite files. Either change filename or delete conflicting file
savedata(CEx_out,CEy_out,vx[0][:],np.asarray([vy[i][0] for i in range(0,len(vy))]),x_out,metadata_out=metadata,params=params,filename=flnm) #assumes uniform velocity grid
