import glob
import numpy as np
import h5py
import os
import math
from netCDF4 import Dataset
from datetime import datetime

#TODO: rename downsample_factor1
def avg_bin_3darr(data_array,downsample_factor1,downsample_factor2,downsample_factor3):
    """
    Averages 3D array of shape (factor1,factor2,factor3) into 3D array of shape (factor1/downsample_factor1,factor2/downsample_factor2,factor3/downsample_factor3)

    Parameters
    ----------
    data_array : 3D array
        data to be averaged
    downsample_factor(1/2/3) : int
        factor to downsample array by in each dim

    Returns
    -------
    downsampled_array : 3D array
        averaged data
    """

    nz,ny,nx = data_array.shape
    if(not(nz % downsample_factor1 == 0)):
        print("Error! nz must be divisible by downsample_factor1!")
        return

    if(not(ny % downsample_factor2 == 0)):
        print("Error! ny must be divisible by downsample_factor2!")
        return

    if(not(nx % downsample_factor3 == 0)):
        print("Error! nx must be divisible by downsample_factor3!")
        return

    new_height = data_array.shape[0] // downsample_factor1
    new_width = data_array.shape[1] // downsample_factor2
    new_depth = data_array.shape[2] // downsample_factor3
    reshaped_array = data_array[:new_height * downsample_factor1,
                            :new_width * downsample_factor2,
                            :new_depth * downsample_factor3]
    reshaped_array = reshaped_array.reshape(new_height, downsample_factor1,
                                        new_width, downsample_factor2,
                                        new_depth, downsample_factor3)
    downsampled_array = np.mean(reshaped_array, axis=(1, 3, 5))

    return np.asarray(downsampled_array)

def avg_bin_1darr(data_array,downsample_factor):
    """
    Averages 1D array of shape (factor1) into 1D array of shape (factor1/downsample_factor1)

    Parameters
    ----------
    data_array : 1D array
        data to be averaged
    downsample_factor : int
        factor to downsample array

    Returns
    -------
    downsampled_array : 1D array
        averaged data
    """

    narr = len(data_array)
    if(not(narr % downsample_factor == 0)):
        print("Error! narr must be divisible by downsample_factor")
        print('narr: ', narr, 'downsample_factor: ', downsample_factor, 'narr%downsample_factor', narr%downsample_factor)
        return

    new_length = len(data_array) // downsample_factor
    reshaped_array = data_array[:new_length * downsample_factor]
    reshaped_array = reshaped_array.reshape(new_length, downsample_factor)
    downsampled_array = np.mean(reshaped_array, axis=1)

    return np.asarray(downsampled_array)

def avg_dict(ddict,binidxsz=[2,2,2],planes=['z','y','x']):
    """
    Averages array into bins equal to integer multiples of original size

    Note: grid must be divisible without truncation into new grid

    Parameters
    ----------
    ddict : dict
        field, fluid, dens dict- (e.g. dfields from load_fields in loadaux)
    binidxsz : array of size 2/3 (of integers)
        new, larger, integer multiple size of binn
    planes : array of size 3
        planes to modify (['z','y','x'] or ['y','x'])

    Returns
    -------
    ddictout : dict
        subseet of provided dict
    """

    #get keys that need to be reduced:
    dkeys = list(ddict.keys())
    keys = [dkeys[_i] for _i in range(0,len(dkeys)) if ('_' in dkeys[_i] and dkeys[_i][-1] in planes)]

    import copy
    ddictout = copy.deepcopy(ddict)

    for kyidx in range(0,len(keys)):
        if(not(keys[kyidx].split('_')[0] in keys)):
            keys.append(keys[kyidx].split('_')[0])

    #TODO: test and use something like numpy.mean(x.reshape(-1, 2), 1) 
    for ky in keys:
        if('_' in ky):
            if('x' in planes):
                if(ky[-1] == 'x'):
                    if(len(binidxsz)==2):
                        print("Error! Must be used with a 3d sim") #TODO: implement for 2d arrays
                        return
                    if(len(binidxsz)==3):
                        ddictout[ky] = avg_bin_1darr(ddictout[ky],binidxsz[2])
            if('y'in planes):
                if(ky[-1] == 'y'):
                    if(len(binidxsz)==2):
                        print("Error! Must be a 3d sim") #TODO: implement for 2d arrays
                        return
                    if(len(binidxsz)==3):
                        ddictout[ky] = avg_bin_1darr(ddictout[ky],binidxsz[1])
            if('z' in planes):
                if(ky[-1] == 'z'):
                    ddictout[ky] = avg_bin_1darr(ddictout[ky],binidxsz[0])
        else:
            if('x' in planes and 'y' in planes and not('z' in planes)):
                print("Error! Must be used with a 3d sim")
                return
            elif('x' in planes and 'y' in planes and 'z' in planes):
                ddictout[ky] = avg_bin_3darr(ddictout[ky],binidxsz[0],binidxsz[1],binidxsz[2])

    return ddictout

def find_nearest(array, val):
    """
    Finds index of element in array with value closest to given value

    Paramters
    ---------
    array : 1d array
        array
    value : float
        value you want to approximately find in array

    Returns
    -------
    idx : int
        index of nearest element
    """

    array = np.asarray(array)
    idx = (np.abs(array-val)).argmin()
    return idx


def load_params(path,num,debug=False):
    """
    WARNING: num should be a string. TODO: rename to something else
    """

    params = {}

    with h5py.File(path + 'param.' + num, 'r') as paramfl:
        if(debug): print(list(paramfl.keys()))

        params['comp'] = paramfl['c_omp'][0]
        params['c'] = paramfl['c'][0]
        params['sigma'] = paramfl['sigma'][0]
        params['istep'] = paramfl['istep'][0]
        params['massratio'] = paramfl['mi'][0]/paramfl['me'][0]
        params['mi'] = paramfl['mi'][0]
        params['me'] = paramfl['me'][0]
        params['ppc'] = paramfl['ppc0'][0]
        try:
            params['sizex'] = paramfl['sizex'][0]
        except:
            params['sizex'] = paramfl['sizey'][0]
        params['delgam'] = paramfl['delgam'][0]

    return params

def load_fields(path_fields, num, field_vars = 'ex ey ez bx by bz', normalizeFields=False):
    """
    This assumes 1D implies data in the 3rd axis only, 2D implies data in the 2nd and 3rd axis only.

    """

    if(normalizeFields):
        field_vars += ' dens'
    field_vars = field_vars.split()
    field = {}
    field['Vframe_relative_to_sim_out'] = 0.

    is1D = False
    is2D = False
    is3D = False
    with h5py.File(path_fields + 'flds.tot.' + num, 'r') as fieldfl:
        for k in field_vars:
            if(fieldfl[k][:].shape[0] > 1 and fieldfl[k][:].shape[1]>1): #3D grid
                is3D = True
                field[k] = fieldfl[k][:]
            elif(fieldfl[k][:].shape[1] > 1): #2D grid
                is2D = True
                _temp = fieldfl[k][0,:,:]
                field[k] = np.zeros((2,fieldfl[k][:].shape[1],fieldfl[k][:].shape[2]))
                field[k][0,:,:] = _temp
                field[k][1,:,:] = _temp
            else: #1D grid
                is1D = True
                _temp = fieldfl[k][0,0,:]
                field[k][0,0,:] = _temp
                field[k][1,0,:] = _temp
                field[k][0,1,:] = _temp
                field[k][1,1,:] = _temp

    #Reconstruct grid
    params = load_params(path_fields,num)

    if(is1D):
        dx = params['istep']
        for key in field_vars:
            field[key+'_xx'] = np.linspace(0., field[key].shape[2]*dx, field[key].shape[2])
            field[key+'_yy'] = np.asarray([0.,1.])
            field[key+'_zz'] = np.asarray([0.,1.])

    elif(is2D):
        dx = params['istep']
        dy = dx
        for key in field_vars:
            field[key+'_xx'] = np.linspace(0., field[key].shape[2]*dx, field[key].shape[2])
            field[key+'_yy'] = np.linspace(0., field[key].shape[1]*dy, field[key].shape[1])
            field[key+'_zz'] = np.asarray([0.,1.])

    elif(is3D):
        dx = params['istep']
        dy = dx
        for key in field_vars:
            field[key+'_xx'] = np.linspace(0., field[key].shape[2]*dx, field[key].shape[2])
            field[key+'_yy'] = np.linspace(0., field[key].shape[1]*dy, field[key].shape[1])
            field[key+'_zz'] = np.linspace(0., field[key].shape[0]*dz, field[key].shape[0])

    if(normalizeFields):
        #normalize to d_i
        comp = params['comp']
        massratio = params['mi']/params['me']
        for key in field.keys():
            if(key+'_xx' in field.keys()):
                field[key+'_xx'] /= (comp*np.sqrt(massratio))
            if(key+'_yy' in field.keys()):
                field[key+'_yy'] /= (comp*np.sqrt(massratio))
            if(key+'_zz' in field.keys()):
                field[key+'_zz'] /= (comp*np.sqrt(massratio))

        if('ex' in field.keys()):
            bnorm = params['c']**2*params['sigma']/params['comp']
            sigma_ion = params['sigma']*params['me']/params['mi'] #NOTE: this is subtely differetn than what aaron's normalization is- fix it (missingn factor of gamma0 and mi+me)
            enorm = bnorm*np.sqrt(sigma_ion)*params['c']

        if('jx' in field.keys()):
            vti0 = np.sqrt(params['delgam'])#Note: velocity is in units γV_i/c so we do not include '*params['c']'
            jnorm = vti0 #normalize to vti

        #normalize to correct units
        for key in field_vars:
            if(key[0] == 'b'):
                field[key] /= bnorm
            elif(key[0] == 'e'):
                field[key] /= enorm
            elif(key[0] == 'j'):
                field[key] /= jnorm

    field['Vframe_relative_to_sim'] = 0.

    return field

def load_particles(path, num, normalizeVelocity=False):
    """
    Loads TRISTAN particle data

    Parameters
    ----------
    path : string
        path to data folder
    num : int
        frame of data this function will load
    normalizeVelocity : bool (opt)#TODO: rename
        normalizes velocity to v_thermal,species and position to d_i

    Returns
    -------
    pts_elc : dict
        dictionary containing electron particle data
    pts_ion : dict
        dictionary containing ion particle data
    """

    dens_vars_elc = 'ue ve we xe ye ze gammae'.split()
    dens_vars_ion = 'ui vi wi xi yi zi gammai'.split()

    pts_elc = {}
    pts_ion = {}
    with h5py.File(path + 'prtl.tot.' + num, 'r') as f:

        for k in dens_vars_elc:
            pts_elc[k] = f[k][:] #note: velocity is in units γV_i/c
        for l in dens_vars_ion:
            pts_ion[l] = f[l][:]

        pts_elc['inde'] = f['inde'][:]
        pts_ion['indi'] = f['indi'][:]

        pts_elc['proce'] = f['proce'][:]
        pts_ion['proci'] = f['proci'][:]

    pts_elc['Vframe_relative_to_sim'] = 0. #tracks frame (along vx) relative to sim
    pts_ion['Vframe_relative_to_sim'] = 0. #tracks frame (along vx) relative to sim

    pts_elc['q'] = -1. #tracks frame (along vx) relative to sim
    pts_ion['q'] = 1. #tracks frame (along vx) relative to sim

    if(normalizeVelocity):

        params = load_params(path,num)
        massratio = load_params(path,num)['mi']/load_params(path,num)['me']
        vti0 = np.sqrt(params['delgam'])#Note: velocity is in units γV_i/c so we do not include '*params['c']'
        vte0 = np.sqrt(params['mi']/params['me'])*vti0 #WARNING: THIS ASSUME Ti/Te = 1, TODO: don't assume Ti/Te = 1
        comp = params['comp']

        #normalize
        elc_vkeys = 'ue ve we'.split()
        ion_vkeys = 'ui vi wi'.split()
        elc_poskeys = 'xe ye ze'.split()
        ion_poskeys = 'xi yi zi'.split()
        for k in elc_vkeys:
            pts_elc[k] /= vte0
        for k in ion_vkeys:
            pts_ion[k] /= vti0
        for k in elc_poskeys:
            pts_elc[k] /= (comp*np.sqrt(massratio))
        for k in ion_poskeys:
            pts_ion[k] /= (comp*np.sqrt(massratio))

    return pts_elc, pts_ion

#computes entropy for 1 cell
#vx, vy, vz are parallel lists of velocities in cell
#n is number of particles per cell times nnumber of cells (reference quantitiy)

#Authors: Colby Haggerty (primary) and Collin Brown
def entropy_and_mom(vxspar, vyspar, vzspar, n, norm, dv, vmax):
    # This version uses a full 3D distro
    vxbins = np.arange(-vmax, vmax+dv, dv)
    vx = (vxbins[1:] + vxbins[:-1])/2.
    vybins = np.arange(-vmax, vmax+dv, dv)
    vy = (vybins[1:] + vybins[:-1])/2.
    vzbins = np.arange(-vmax, vmax+dv, dv)
    vz = (vzbins[1:] + vzbins[:-1])/2.
    
    #convert to 3d arrays-> useful for array calculations
    vx3d = np.zeros((len(vz), len(vy), len(vx)))
    vy3d = np.zeros((len(vz), len(vy), len(vx)))
    vz3d = np.zeros((len(vz), len(vy), len(vx)))
    for i in range(0, len(vx)):
        for j in range(0, len(vy)):
            for k in range(0, len(vz)):
                vx3d[k][j][i] = vx[i]

    for i in range(0, len(vx)):
        for j in range(0, len(vy)):
            for k in range(0, len(vz)):
                vy3d[k][j][i] = vy[j]

    for i in range(0, len(vx)):
        for j in range(0, len(vy)):
            for k in range(0, len(vz)):
                vz3d[k][j][i] = vz[k]
    
    h,_ = np.histogramdd([vxspar, vyspar, vzspar], bins=[vzbins, vybins, vxbins])
    h = h + 1e-10 #stops divide by zero errors
    
    #compute 0th momenty
    npar = np.sum(h)
    
    #compute 1st moment 
    ux = np.mean(vxspar)
    uy = np.mean(vyspar)
    uz = np.mean(vzspar)

    #compute 2nd moment
    T = (np.std(vxspar)**2 + np.std(vyspar)**2 + np.std(vzspar)**2)/3.
    
    #compute pressure tensor (also '2nd' moment)
    #VERSION 2 (less slow) #TODO: make version 3 that truly vectorizes this operation or use JIT
    Ttensor = np.asarray([[[np.outer([vx[_i],vy[_j],vz[_k]],[vx[_i],vy[_j],vz[_k]]) * h[_k,_j,_i]
                           for _i in range(0,len(vx))] for _j in range(0,len(vy))] for _k in range(0,len(vz))])
    
    Ttensor = np.sum(Ttensor, axis=(0,1,2)) * dv**3
    Ttensor = np.flip(Ttensor, axis=(0))
    Ttensor = np.flip(Ttensor, axis=(1))
    #VERSION 1 (slow)
#     #TODO: vectorize if slow
#     Ttensor = np.zeros((3,3))
#     counter = 0.
#     for i in range(0, len(vx)):
#         for j in range(0, len(vy)):
#             for k in range(0, len(vz)):
#                 _Ttensor = np.outer([vx[i],vy[j],vz[k]],[vx[i],vy[j],vz[k]])*h[k,j,i]
#                 counter += 1.
#                 Ttensor += _Ttensor              
#     Ttensor = Ttensor/counter
    #TODO: remove unneeded off diagnoal terms (symmetry exists so we can save memory)
    Txx = Ttensor[0,0]
    Txy = Ttensor[0,1]
    Txz = Ttensor[0,2]
    Tyx = Ttensor[1,0]
    Tyy = Ttensor[1,1]
    Tyz = Ttensor[1,2]
    Tzx = Ttensor[2,0]
    Tzy = Ttensor[2,1]
    Tzz = Ttensor[2,2]
    
    #compute 3rd  moment (heat flux density)
    qx = np.asarray([[[0.5*(vx[_i]-ux)*np.dot([vx[_i]-ux,vy[_j]-uy,vz[_k]-uz],[vx[_i]-ux,vy[_j]-uy,vz[_k]-uz])*h[_k,_j,_i]
                    for _i in range(0,len(vx))] for _j in range(0,len(vy))] for _k in range(0,len(vz))])
    qx = np.sum(qx)
    qy = np.asarray([[[0.5*(vy[_i]-uy)*np.dot([vx[_i]-ux,vy[_j]-uy,vz[_k]-uz],[vx[_i]-ux,vy[_j]-uy,vz[_k]-uz])*h[_k,_j,_i]
                    for _i in range(0,len(vx))] for _j in range(0,len(vy))] for _k in range(0,len(vz))])
    qy = np.sum(qy)
    qz = np.asarray([[[0.5*(vz[_i]-uz)*np.dot([vx[_i]-ux,vy[_j]-uy,vz[_k]-uz],[vx[_i]-ux,vy[_j]-uy,vz[_k]-uz])*h[_k,_j,_i]
                    for _i in range(0,len(vx))] for _j in range(0,len(vy))] for _k in range(0,len(vz))])
    qz = np.sum(qz)
    
    #calculate entropy
    s = -norm*np.sum(h*(np.log(h*norm/n)))
    sr = (-norm*np.sum(h*np.log(h*norm/dv**3))
          + n*np.log(n/(2.*np.pi*T)**1.5) - 3.*n/2.)
    
    return h, npar, s, sr, T, ux, uy, uz, qx, qy, qz, Txx, Txy, Txz, Tyx, Tyy, Tyz, Tzx, Tzy, Tzz


#Given 3d array called moments with ux,uy,uz,..., Txx Tyy etc, this computes PiD
#Each array is A[t_index, y_index, x_index] i think


#TODO: science Idea, PiD can be used to see scattering (normal deformation)
#Authors: Colby Haggerty (primary) and Collin Brown
def calc_PiD_dQ(m,compIons=False): # this calc_PiD works for multi-times
    from scipy.ndimage import gaussian_filter as gf
    
    if(compIons):
        Ptensorkeys = 'Txxi Tyyi Tzzi Txyi Tyzi Tzxi'
        velkeys = 'uxi uyi uzi'
        denkey = 'npari'
        hfluxkeys = 'qxi qyi qzi'.split()
    else:
        Ptensorkeys = 'Txxe Tyye Tzze Txye Tyze Tzxe'
        velkeys = 'uxe uye uze'
        denkey = 'npare'
        hfluxkeys = 'qxe qye qze'.split()
        
    
    ux,uy,uz = [m[k] for k in velkeys.split()]
    Pxx,Pyy,Pzz,Pxy,Pyz,Pzx = [m[k] for k in Ptensorkeys.split()] #Note: original had extra den term ([m[denkey]*m[k] for k in Ptensorkeys.split()]) that we already had in our way of computing moment. TODO: double check which is correct
    xx,yy = [m[k] for k in "x y".split()]
    
    dx = xx[1] - xx[0]
    def ddx(f): # Assume Open, time, y, x
        dfdx = 0*f
        dfdx[:,:, 1:-1] = (f[:,:, 2:] - f[:,:, :-2])/2./dx
        dfdx[:,:, 0] = dfdx[:,:, 1]
        dfdx[:,:, -1] = dfdx[:,:, -2]
        return dfdx
    
    def ddy(f): # Assume Open, time, y, x
        dfdy = 0*f
        dfdy[:, 1:-1] = (f[:, 2:] - f[:, :-2])/2./dx
        dfdy[:, 0] = dfdy[:, 1]
        dfdy[:, -1] = dfdy[:, -2]
        return dfdy
    
    def ddz(f): # Assume Invariant
        return 0.*f
    
    Pi = {}
    Sa = {}
    Da = {}
    Danorm = {}
    Dashear = {}
    
    pa = 1./3.*(Pxx + Pyy + Pzz)
    Pi["xx"] = Pxx - pa
    Pi["yy"] = Pyy - pa
    Pi["zz"] = Pzz - pa
    Pi["xy"] = Pxy
    Pi["yz"] = Pyz
    Pi["zx"] = Pzx
    
    Sa['xx'] = ddx(ux)
    Sa['yy'] = ddy(uy)
    Sa['zz'] = ddz(uz)
    Sa['xy'] = 1./2.*(ddx(uy) + ddy(ux))
    Sa['yz'] = 1./2.*(ddy(uz) + ddz(uy))
    Sa['zx'] = 1./2.*(ddz(ux) + ddx(uz))
    
    th = Sa['xx'] + Sa['yy'] + Sa['zz']
    Da['xx'] = Sa['xx'] - 1./3.*th
    Da['yy'] = Sa['yy'] - 1./3.*th
    Da['zz'] = Sa['zz'] - 1./3.*th
    Da['xy'] = Sa['xy']
    Da['yz'] = Sa['yz']
    Da['zx'] = Sa['zx']
    
    #D normal deformation is same as D with off diagnal elements equal to zero (Cassak and Barbhuiya 2020)
    Danorm['xx'] = Da['xx']
    Danorm['yy'] = Da['yy']
    Danorm['zz'] = Da['zz']
    Danorm['xy'] = 0.
    Danorm['yz'] = 0.
    Danorm['zx'] = 0.
    
    #D shear is same as D with diagnal elements equal to zero (Cassak and Barbhuiya 2020)
    Dashear['xx'] = 0.
    Dashear['yy'] = 0.
    Dashear['zz'] = 0.
    Dashear['xy'] = Da['xy']
    Dashear['yz'] = Da['yz']
    Dashear['zx'] = Da['zx']
    
    pth = -pa*th
    
    PiD = 0.*Pi['xx']
    for k in "xx xy zx xy yy yz zx yz zz".split():
        PiD = PiD - Pi[k]*Da[k]
        
    PiDnorm = 0.*Pi['xx']
    for k in "xx xy zx xy yy yz zx yz zz".split():
        PiDnorm = PiDnorm - Pi[k]*Danorm[k]
    
    PiDshear = 0.*Pi['xx']
    for k in "xx xy zx xy yy yz zx yz zz".split():
        PiDshear = PiDshear - Pi[k]*Dashear[k]
    
    #Now Del.Q
    sig = 1.
    dQ = -ddx(gf(m[hfluxkeys[0]], sigma=sig)) - ddy(gf(m[hfluxkeys[1]], sigma=sig))
    
    return pth, PiD, dQ, PiDnorm, PiDshear



framenum = '700' #frame to make figure of (should be a string)
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
normalize = True
#TODO: use different vmax and dv for ion and elec
vmaxe = 13
dve = 1

vmaxi = 25
dvi = 3

pregenerated = True
pregeneratedflnm = 'analysisfiles/testmoms_lowres.pickle'

if(not(pregenerated)):
    print("Loading particle data...")
    dfields = load_fields(flpath,framenum,normalizeFields=normalize)
    dpar_elec, dpar_ion = load_particles(flpath,framenum,normalizeVelocity=normalize)

    nx = len(dfields['ex_xx'])
    ny = len(dfields['ex_yy'])
    nt = 1

    print("Downsizing fields: before: (nx = ",nx," ny = ", ny, ")")

    #downsize fields data for speed        
    dfields = avg_dict(dfields,binidxsz=[1,5,10],planes=['z','y','x'])
    nx = len(dfields['ex_xx'])
    ny = len(dfields['ex_yy'])
    
    print("Downsizing fields: after: (nx = ",nx," ny = ", ny, ")")

    print("Done!")

    print("Making particle bins...")
    debug = True
    is2D = True
    #TODO: while this is faster than prev versions, we can do a lot better if we handle the data better. Improve it
    #bin particles
    nx = len(dfields['ex_xx'])
    ny = len(dfields['ex_yy'])
    nz = len(dfields['ex_zz'])
    ion_bins = [[[ [] for _ in range(nx)] for _ in range(ny)] for _ in range(nz)] 
    elec_bins = [[[ [] for _ in range(nx)] for _ in range(ny)] for _ in range(nz)]

    print("Done! Binning particles...")
    for _i in range(0,len(dpar_ion['xi'])):
        if(debug and _i % 1000000 == 0): print("Binned: ", _i," ions of ", len(dpar_ion['xi']))
        xx = dpar_ion['xi'][_i]
        yy = dpar_ion['yi'][_i]
        zz = dpar_ion['zi'][_i]

        xidx = find_nearest(dfields['ex_xx'], xx)
        yidx = find_nearest(dfields['ex_yy'], yy)
        zidx = find_nearest(dfields['ex_zz'], zz)
        if(is2D):zidx = 0

        ion_bins[zidx][yidx][xidx].append({'ui':dpar_ion['ui'][_i] ,'vi':dpar_ion['vi'][_i] ,'wi':dpar_ion['wi'][_i]})

    for _i in range(0,len(dpar_elec['xe'])):
        if(debug and _i % 1000000 == 0): print("Binned: ", _i," elecs of ", len(dpar_elec['xe']))
        xx = dpar_elec['xe'][_i]
        yy = dpar_elec['ye'][_i]
        zz = dpar_elec['ze'][_i]

        xidx = find_nearest(dfields['ex_xx'], xx)
        yidx = find_nearest(dfields['ex_yy'], yy)
        zidx = find_nearest(dfields['ex_zz'], zz)
        if(is2D):zidx = 0

        elec_bins[zidx][yidx][xidx].append({'ue':dpar_elec['ue'][_i] ,'ve':dpar_elec['ve'][_i] ,'we':dpar_elec['we'][_i]})

    print("Done! Taking moments...")
    Txxi = np.zeros((nt,ny,nx))
    Txyi = np.zeros((nt,ny,nx))
    Txzi = np.zeros((nt,ny,nx))
    Tyxi = np.zeros((nt,ny,nx))
    Tyyi = np.zeros((nt,ny,nx))
    Tyzi = np.zeros((nt,ny,nx))
    Tzxi = np.zeros((nt,ny,nx))
    Tzyi = np.zeros((nt,ny,nx))
    Tzzi = np.zeros((nt,ny,nx))
    uxi = np.zeros((nt,ny,nx))
    uyi = np.zeros((nt,ny,nx))
    uzi = np.zeros((nt,ny,nx))
    Ti = np.zeros((nt,ny,nx))
    npari = np.zeros((nt,ny,nx))
    qxi = np.zeros((nt,ny,nx))
    qyi = np.zeros((nt,ny,nx))
    qzi = np.zeros((nt,ny,nx))
    si = np.zeros((nt,ny,nx))
    sri = np.zeros((nt,ny,nx))

    Txxe = np.zeros((nt,ny,nx))
    Txye = np.zeros((nt,ny,nx))
    Txze = np.zeros((nt,ny,nx))
    Tyxe = np.zeros((nt,ny,nx))
    Tyye = np.zeros((nt,ny,nx))
    Tyze = np.zeros((nt,ny,nx))
    Tzxe = np.zeros((nt,ny,nx))
    Tzye = np.zeros((nt,ny,nx))
    Tzze = np.zeros((nt,ny,nx))
    uxe = np.zeros((nt,ny,nx))
    uye = np.zeros((nt,ny,nx))
    uze = np.zeros((nt,ny,nx))
    Te = np.zeros((nt,ny,nx))
    npare = np.zeros((nt,ny,nx))
    qxe = np.zeros((nt,ny,nx))
    qye = np.zeros((nt,ny,nx))
    qze = np.zeros((nt,ny,nx))
    se = np.zeros((nt,ny,nx))
    sre = np.zeros((nt,ny,nx))



    norm = 1
    n = 1
    print("TODO: CORRECTLY COMPUTE NORM AND N HERE!!!")
    #TODO: parallelize this
    _k = 0 #because we are using 2d data
    for _i in range(0,len(dfields['ex_xx'])):
        print(_i," of ", len(dfields['ex_xx']))
        for _j in range(0,len(dfields['ex_yy'])):
        
            #compute entropy and other quants for ions
            vxs = [ion_bins[_k][_j][_i][_idx]['ui'] for _idx in range(0,len(ion_bins[_k][_j][_i]))]
            vys = [ion_bins[_k][_j][_i][_idx]['vi'] for _idx in range(0,len(ion_bins[_k][_j][_i]))]
            vzs = [ion_bins[_k][_j][_i][_idx]['wi'] for _idx in range(0,len(ion_bins[_k][_j][_i]))]
            (h, npar, s, sr, T, ux, uy, uz, 
            qx, qy, qz, Txx, Txy, Txz, Tyx, Tyy, Tyz, Tzx, Tzy, Tzz) = entropy_and_mom(vxs, vys, vzs, n, norm, dvi, vmaxi)
        
            Txxi[0,_j,_i] = Txx
            Txyi[0,_j,_i] = Txy
            Txzi[0,_j,_i] = Txz
            Tyxi[0,_j,_i] = Tyx
            Tyyi[0,_j,_i] = Tyy
            Tyzi[0,_j,_i] = Tyz
            Tzxi[0,_j,_i] = Tzx
            Tzyi[0,_j,_i] = Tzy
            Tzzi[0,_j,_i] = Tzz
            uxi[0,_j,_i] = ux
            uyi[0,_j,_i] = uy
            uzi[0,_j,_i] = uz
            Ti[0,_j,_i] = T
            npari[0,_j,_i] = npar
            qxi[0,_j,_i] = qx 
            qyi[0,_j,_i] = qy
            qzi[0,_j,_i] = qz
            si[0,_j,_i] = s
            sri[0,_j,_i] = sr
        
            #compute entropy and other quants for elecs
            vxs = [elec_bins[_k][_j][_i][_idx]['ue'] for _idx in range(0,len(elec_bins[_k][_j][_i]))]
            vys = [elec_bins[_k][_j][_i][_idx]['ve'] for _idx in range(0,len(elec_bins[_k][_j][_i]))]
            vzs = [elec_bins[_k][_j][_i][_idx]['we'] for _idx in range(0,len(elec_bins[_k][_j][_i]))]
            (h, npar, s, sr, T, ux, uy, uz, 
            qx, qy, qz, Txx, Txy, Txz, Tyx, Tyy, Tyz, Tzx, Tzy, Tzz) = entropy_and_mom(vxs, vys, vzs, n, norm, dve, vmaxe)
        
            Txxe[0,_j,_i] = Txx
            Txye[0,_j,_i] = Txy
            Txze[0,_j,_i] = Txz
            Tyxe[0,_j,_i] = Tyx
            Tyye[0,_j,_i] = Tyy
            Tyze[0,_j,_i] = Tyz
            Tzxe[0,_j,_i] = Tzx
            Tzye[0,_j,_i] = Tzy
            Tzze[0,_j,_i] = Tzz
            uxe[0,_j,_i] = ux
            uye[0,_j,_i] = uy
            uze[0,_j,_i] = uz
            Te[0,_j,_i] = T
            npare[0,_j,_i] = npar
            qxe[0,_j,_i] = qx 
            qye[0,_j,_i] = qy
            qze[0,_j,_i] = qz
            se[0,_j,_i] = s
            sre[0,_j,_i] = sr

    print("Done! Saving as pickle")
    #TODO: save as pickle
    moms = {'Txxi':Txxi,'Txyi':Txyi,'Txzi':Txyi,
            'Tyxi':Tyxi,'Tyyi':Tyyi,'Tyzi':Tyzi,
            'Tzxi':Tzxi,'Tzyi':Tzyi,'Tzzi':Tzzi,
            'uxi':uxi,'uyi':uyi,'uzi':uzi,
            'qxi':qxi,'qyi':qyi,'qzi':qzi,
            'Ti':Ti,'npari':npari,'si':si,'sri':sri,
            'Txxe':Txxe,'Txye':Txye,'Txze':Txye,
            'Tyxe':Tyxe,'Tyye':Tyye,'Tyze':Tyze,
            'Tzxe':Tzxe,'Tzye':Tzye,'Tzze':Tzze,
            'uxe':uxe,'uye':uye,'uze':uze,
            'qxe':qxe,'qye':qye,'qze':qze,
            'Te':Te,'npare':npare,'se':se,'sre':sre,
            'x':dfields['ex_xx'],'y':dfields['ex_yy'],'z':dfields['ex_zz']}

    import pickle
    import os

    os.system('mkdir analysisfiles')

    momsflnm = 'analysisfiles/testmoms.pickle'
    fileout = open(momsflnm, 'wb')
    pickle.dump(moms, fileout)
else:
    import pickle
    filein = open(pregeneratedflnm, 'rb')
    moms = pickle.load(filein)
    filein.close()


print("Done! Computing PiD")
pths_i,pids_i,dqs_i,PiDnorm_i,PiDshear_i = calc_PiD_dQ(moms,compIons=True)
pths_e,pids_e,dqs_e,PiDnorm_e,PiDshear_e = calc_PiD_dQ(moms,compIons=False)

print("Done! Making figures...")
import matplotlib.pyplot as plt
#make plots of fields
fig, axs = plt.subplots(8,figsize=(20,7*2),sharex=True)

fig.subplots_adjust(hspace=.1)

#npar elec
npareim = axs[0].pcolormesh(moms['x'], moms['y'], moms['npare'][0,:,:], cmap="RdYlBu", shading="gouraud")
fig.colorbar(npareim, ax=axs[0])

#npar ion
npariim = axs[1].pcolormesh(moms['x'], moms['y'], moms['npari'][0,:,:], cmap="RdYlBu", shading="gouraud")
fig.colorbar(npariim, ax=axs[1])

#Temperature elec
Teim = axs[2].pcolormesh(moms['x'], moms['y'], moms['Te'][0,:,:], cmap="RdYlBu", shading="gouraud")
fig.colorbar(Teim, ax=axs[2])

#Temperature ion
Tiim = axs[3].pcolormesh(moms['x'], moms['y'], moms['Ti'][0,:,:], cmap="RdYlBu", shading="gouraud")
fig.colorbar(Tiim, ax=axs[3])

#S elec
seim = axs[4].pcolormesh(moms['x'], moms['y'], moms['se'][0,:,:], cmap="RdYlBu", shading="gouraud")
fig.colorbar(seim, ax=axs[4])

#S ion
seim = axs[5].pcolormesh(moms['x'], moms['y'], moms['si'][0,:,:], cmap="RdYlBu", shading="gouraud")
fig.colorbar(seim, ax=axs[5])

#Sr elec
sreim = axs[6].pcolormesh(moms['x'], moms['y'], moms['sre'][0,:,:], cmap="RdYlBu", shading="gouraud")
fig.colorbar(sreim, ax=axs[6])

#Sr ion
sriim = axs[7].pcolormesh(moms['x'], moms['y'], moms['sri'][0,:,:], cmap="RdYlBu", shading="gouraud")
fig.colorbar(sriim, ax=axs[7])

#print labels of each plot
import matplotlib.patheffects as PathEffects
pltlabels = [r'$n_e$',r'$n_i$',r'$T_e$',r'$T_i$',r'$s_e/n$',r'$s_i/n$',r'$s_{e,r}/n$',r'$s_{i,r}/n$']
_xtext = moms['x'][-1]*.75 
_ytext = moms['y'][-1]*.6
for _i in range(0,len(axs)):
    _txt = axs[_i].text(_xtext,_ytext,pltlabels[_i],color='white',fontsize=24)
    _txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

for ax in axs:
    ax.grid()

import os
os.system('mkdir figures')

plt.savefig('figures/moms_test.png',dpi=300,format='png',bbox_inches='tight')

plt.close()


print("WARNING: setting Nan to zero")
pths_i = np.nan_to_num(pths_i, nan=0.0)
pids_i = np.nan_to_num(pids_i, nan=0.0)
dqs_i = np.nan_to_num(dqs_i, nan=0.0)
PiDnorm_i = np.nan_to_num(PiDnorm_i, nan=0.0)
PiDshear_i = np.nan_to_num(PiDshear_i, nan=0.0)

pths_e = np.nan_to_num(pths_e, nan=0.0)
pids_e = np.nan_to_num(pids_e, nan=0.0)
dqs_e = np.nan_to_num(dqs_e, nan=0.0)
PiDnorm_e = np.nan_to_num(PiDnorm_e, nan=0.0)
PiDshear_e = np.nan_to_num(PiDshear_e, nan=0.0)

import matplotlib.pyplot as plt

#make plots of fields
fig, axs = plt.subplots(10,figsize=(20,7*2),sharex=True)

fig.subplots_adjust(hspace=.1)

#TODO: fix ordering of plots

print("DEBUG: nx ny",len(moms['x']),len(moms['y']))

absmax = max(np.max(pths_i),np.abs(np.min(pths_i)))
pths_iim = axs[1].pcolormesh(moms['x'], moms['y'], pths_i[0,:,:], vmax=absmax, vmin=-absmax, cmap="bwr", shading="gouraud")
fig.colorbar(pths_iim, ax=axs[0])

absmax = max(np.max(pths_e),np.abs(np.min(pths_e)))
pths_eim = axs[0].pcolormesh(moms['x'], moms['y'], pths_e[0,:,:], vmax=absmax, vmin=-absmax, cmap="bwr", shading="gouraud")
fig.colorbar(pths_eim, ax=axs[1])

absmax = max(np.max(pids_i),np.abs(np.min(pids_i)))
pids_iim = axs[3].pcolormesh(moms['x'], moms['y'], pids_i[0,:,:], vmax=absmax, vmin=-absmax, cmap="bwr", shading="gouraud")
fig.colorbar(pids_iim, ax=axs[2])

absmax = max(np.max(pids_e),np.abs(np.min(pids_e)))
pids_eim = axs[2].pcolormesh(moms['x'], moms['y'], pids_e[0,:,:], vmax=absmax, vmin=-absmax, cmap="bwr", shading="gouraud")
fig.colorbar(pids_eim, ax=axs[3])

absmax = max(np.max(dqs_i),np.abs(np.min(dqs_i)))
dqs_iim = axs[5].pcolormesh(moms['x'], moms['y'], dqs_i[0,:,:], vmax=absmax, vmin=-absmax, cmap="RdYlBu", shading="gouraud")
fig.colorbar(dqs_iim, ax=axs[4])

absmax = max(np.max(dqs_e),np.abs(np.min(dqs_e)))
dqs_eim = axs[4].pcolormesh(moms['x'], moms['y'], dqs_e[0,:,:], vmax=absmax, vmin=-absmax, cmap="RdYlBu", shading="gouraud")
fig.colorbar(dqs_eim, ax=axs[5])

absmax = max(np.max(PiDnorm_i),np.abs(np.min(PiDnorm_i)))
PiDnorm_iim = axs[7].pcolormesh(moms['x'], moms['y'], PiDnorm_i[0,:,:], vmax=absmax, vmin=-absmax, cmap="bwr", shading="gouraud")
fig.colorbar(PiDnorm_iim, ax=axs[6])

absmax = max(np.max(PiDnorm_e),np.abs(np.min(PiDnorm_e)))
PiDnorm_eim = axs[6].pcolormesh(moms['x'], moms['y'], PiDnorm_e[0,:,:], vmax=absmax, vmin=-absmax, cmap="bwr", shading="gouraud")
fig.colorbar(PiDnorm_eim, ax=axs[7])

absmax = max(np.max(PiDshear_i),np.abs(np.min(PiDshear_i)))
PiDshear_iim = axs[9].pcolormesh(moms['x'], moms['y'], PiDshear_i[0,:,:], vmax=absmax, vmin=-absmax, cmap="bwr", shading="gouraud")
fig.colorbar(PiDshear_iim, ax=axs[8])

absmax = max(np.max(PiDshear_e),np.abs(np.min(PiDshear_e)))
PiDshear_eim = axs[8].pcolormesh(moms['x'], moms['y'], PiDshear_e[0,:,:], vmax=absmax, vmin=-absmax, cmap="bwr", shading="gouraud")
fig.colorbar(PiDshear_eim, ax=axs[9])

#print labels of each plot
import matplotlib.patheffects as PathEffects
pltlabels = [r'$p\theta$ elec',r'$p\theta$ ion',r'$-\Pi_{ij} D{ij}$ elec',r'$-\Pi_{ij} D{ij}$ ion',r'$-\nabla \cdot q_e$',
            r'$-\nabla \cdot q_i$',r'$-\Pi_{ij} D_{norm}{ij}$ elec',r'$-\Pi_{ij} D_{norm}{ij}$ ion',r'$-\Pi_{ij} D_{shear}{ij}$ elec',r'$-\Pi_{ij} D_{shear}{ij}$ ion']
_xtext = moms['x'][-1]*.75 
_ytext = moms['y'][-1]*.6
for _i in range(0,len(axs)):
    _txt = axs[_i].text(_xtext,_ytext,pltlabels[_i],color='white',fontsize=24)
    _txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

for ax in axs:
    ax.grid()

pths_i = np.sum(pths_i,axis=1)[0,:]
pids_i = np.sum(pids_i,axis=1)[0,:]
dqs_i = np.sum(dqs_i,axis=1)[0,:]
PiDnorm_i = np.sum(PiDnorm_i,axis=1)[0,:]
PiDshear_i = np.sum(PiDshear_i,axis=1)[0,:]

pths_e = np.sum(pths_e,axis=1)[0,:]
pids_e = np.sum(pids_e,axis=1)[0,:]
dqs_e = np.sum(dqs_e,axis=1)[0,:]
PiDnorm_e = np.sum(PiDnorm_e,axis=1)[0,:]
PiDshear_e = np.sum(PiDshear_e,axis=1)[0,:]

axs0 = axs[0].twinx()
axs1 = axs[1].twinx()
axs2 = axs[2].twinx()
axs3 = axs[3].twinx()
axs4 = axs[4].twinx()
axs5 = axs[5].twinx()
axs6 = axs[6].twinx()
axs7 = axs[7].twinx()
axs8 = axs[8].twinx()
axs9 = axs[9].twinx()

axs1.plot(moms['x'],pths_i,lw=1,color='black')
axs0.plot(moms['x'],pths_e,lw=1,color='black')
axs3.plot(moms['x'],pids_i,lw=1,color='black')
axs2.plot(moms['x'],pids_e,lw=1,color='black')
axs5.plot(moms['x'],dqs_i,lw=1,color='black')
axs4.plot(moms['x'],dqs_e,lw=1,color='black')
axs7.plot(moms['x'],PiDnorm_i,lw=1,color='black')
axs6.plot(moms['x'],PiDnorm_e,lw=1,color='black')
axs8.plot(moms['x'],PiDnorm_e,lw=1,ls=':',color='red')
axs9.plot(moms['x'],PiDshear_i,lw=1,color='black')
axs8.plot(moms['x'],PiDshear_e,lw=1,color='black')

#twinaxs = [axs0,axs1,axs2,axs3,axs4,axs5,axs6,axs7,axs8,axs9]
#_idx = 0
#for tax in twinaxs:
#    tax.set_ylabel(r'$\Sigma$ '+pltlabels[_idx])
#    _idx += 1

for ax in axs:
    ax.set_xlabel(r"$x / d_i$")
    ax.set_ylabel(r"$y / d_i$")

plt.savefig('figures/PiD_test.png',dpi=300,format='png',bbox_inches='tight')
plt.close()


print("debug: ", pths_e.shape)

#TODO: fix ordering
#make plots of fields
fig, axs = plt.subplots(8,figsize=(20,7*2),sharex=True)
axs[1].plot(moms['x'],pths_i,lw=2,color='black')
axs[0].plot(moms['x'],pths_e,lw=2,color='black')
axs[3].plot(moms['x'],pids_i,lw=2,color='black')
axs[2].plot(moms['x'],pids_e,lw=2,color='black')
axs[5].plot(moms['x'],dqs_i,lw=2,color='black')
axs[4].plot(moms['x'],dqs_e,lw=2,color='black')
axs[7].plot(moms['x'],PiDnorm_i,lw=2,color='black',label = pltlabels[7])
axs[6].plot(moms['x'],PiDnorm_e,lw=2,color='black',label = pltlabels[6])
axs[7].plot(moms['x'],PiDshear_i,lw=2,ls=':',color='red',label = pltlabels[9])
axs[6].plot(moms['x'],PiDshear_e,lw=2,ls=':',color='red',label = pltlabels[8])

axs[6].legend()
axs[7].legend()
axs[6].grid()
axs[7].grid()

pltlabels = [r'$p \theta$ elec',r'$p \theta$ ion',r'$-\Pi_{ij} D{ij}$ elec',r'$-\Pi_{ij} D{ij}$ ion',r'$-\nabla \cdot q_e$',
            r'$-\nabla \cdot q_i$',r'$-\Pi_{ij} D_{norm}{ij}$ elec',r'$-\Pi_{ij} D_{norm}{ij}$ ion',r'$-\Pi_{ij} D_{shear}{ij}$ elec',r'$-\Pi_{ij} D_{shear}{ij}$ ion']
for _i in range(0,len(axs)-2):
    axs[_i].set_ylabel(pltlabels[_i])
    axs[_i].grid()
axs[-1].set_xticks(np.arange(0,20,1))
axs[-1].set_xlabel(r"$x/d_i$")

plt.savefig('figures/PiD_test_proj.png',dpi=300,format='png',bbox_inches='tight')
plt.close()
