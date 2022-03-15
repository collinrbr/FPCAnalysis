# data_netcdf4.py>

#functions related making and loading netcdf4 file to pass to MLA algo and loading netcdf4 files

import numpy as np
import math
import postgkyl as pg
import base64
import adios

#TODO: polyorder is specified in the input file. Consider grabbing from input file rather than letting the user specify it
def load_dist(flnm_prefix,num,species='ion',polyorder=2,interpLevel=3,xpos=None):
    """
    TODO: NORMALIZE TO VTI,0
    """
    # Read in and interpolate the distribution function
    print("Loading distribution function from %s_%s_%d.bp..." % (flnm_prefix, species, num))
    distdata = pg.data.GData("%s_%s_%d.bp" % (flnm_prefix, species, num), z0 = xpos)
    distproj = pg.data.GInterpModal(distdata, polyorder, "ms", interpLevel)
    coords, distf = distproj.interpolate()

#     num_vel_dim = distdata.getNumDims() #returns number of vel dimensions
#     num_pos_dim = len(coords)-num_vel_dim #rest of coordinates are velocity dimensions

    print("WARNING: this function only works for 1D2V data for now")
    print("TODO: figure out num_vel_dim and num_pos_dim")

#     print('num_vel_dim: ',num_vel_dim, 'num_pos_dim: ',num_pos_dim)

    #TODO: put into hist[xx,:,:,:] form
    #for 1d2v starts in xx,v1,v2 form #TODO figure out coordinates

    if(distf.shape[-1] != 1):
        raise TypeError("This function can only load data of a singular frame at a time (TODO: implement multiple frame loading)")

    vti = get_input_params(flnm_prefix,num,species='ion',verbose=False)['vti']

    ddist = {}
    ddist['hist_xx'] = [(coords[0][i]+coords[0][i])/2 for i in range(0,len(coords[0])-1)]
    vx_in = [(coords[1][i]+coords[1][i])/(2*vti) for i in range(0,len(coords[1])-1)] #average and normalize to vti
    vy_in = [(coords[2][i]+coords[2][i])/(2*vti) for i in range(0,len(coords[2])-1)]
    vz_in = [(coords[2][i]+coords[2][i])/(2*vti) for i in range(0,len(coords[2])-1)]
    #if(num_pos_dim == 1 and num_vel_dim == 2):
    ddist['hist'] = np.zeros((distf.shape)) #this is hacky because 4th axisof  distf is technically time TODO: fix
    ddist['hist'][:,:,:,:] = distf[:,:,:,:]
    ddist['hist'] = np.swapaxes(ddist['hist'],1,3) #want xx,vz,vy,vx order

    #note as we rebuild distribution functions by interpolating data, we can get negative values
    #for simplicity, we set all negative values to zero
    for _xxidx in range(0,len(ddist['hist'])):
        for _vzidx in range(0,len(ddist['hist'][_xxidx])):
            for _vyidx in range(0,len(ddist['hist'][_xxidx][_vzidx])):
                for _vxidx in range(0,len(ddist['hist'][_xxidx][_vzidx][_vyidx])):
                    if(ddist['hist'][_xxidx,_vzidx,_vyidx,_vxidx] < 0):
                        ddist['hist'][_xxidx,_vzidx,_vyidx,_vxidx] = 0.

    print("WARNING: zero padding 2V distribution function into 3V...")
    _temphist = np.zeros((len(ddist['hist_xx']),len(vz_in),len(vy_in),len(vx_in)))
    print('shape',ddist['hist'].shape)
    _temphist[:,int(len(vz_in)/2),:,:] = ddist['hist'][:,0,:,:]
    ddist['hist'] = _temphist

    #build vx, vy, vz 3d arrays TODO: this block is used a lot. Put into function and call for all instances
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

    vx = _vx
    vy = _vy
    vz = _vz

    ddist['vx'] = vx
    ddist['vy'] = vy
    ddist['vz'] = vz

    return ddist

def load_fields(flnm_prefix,num,species='ion',polyorder=2,interpLevel=9,xpos=None):
    """
    TODO: consider loading by using file prefix with species and frame
    """

    print("Loading fields from %s_field_%d.bp..." % (flnm_prefix, num))

    fielddata = pg.data.GData("%s_field_%d.bp" % (flnm_prefix, num), z0=xpos)
    fieldproj = pg.data.GInterpModal(fielddata, polyorder, "ms", interpLevel)

    num_pos_dim = fielddata.getNumDims() #returns number of spatial dimensions
    #num_vel_dim = len(coords)-num_pos_dim #rest of coordinates are velocity dimensions
    #print("num_pos_dim: ",num_pos_dim," num_vel_dim: ",num_vel_dim)

    coordsex, ex = fieldproj.interpolate(0)
    coordsey, ey = fieldproj.interpolate(1)
    coordsez, ez = fieldproj.interpolate(2)
    coordsbx, bx = fieldproj.interpolate(3)
    coordsby, by = fieldproj.interpolate(4)
    coordsbz, bz = fieldproj.interpolate(5)

    if(ex.shape[-1] != 1):
        raise TypeError("This function can only load data of a singular frame at a time (TODO: implement multiple frame loading)")

    fieldkeys = ['ex','ey','ez','bx','by','bz']
    dfields = {}
    for key in fieldkeys:
        if(num_pos_dim == 1): #Spoof the 1D data into 3D. WARNING assumes data is 1D down x axis
            dfields[key] = np.zeros((2,2,len(locals()[key]))) #dfields is typically ordered z,y,x
            dfields[key][0,0,:] = locals()[key][:,0]
            dfields[key][0,1,:] = locals()[key][:,0]
            dfields[key][1,0,:] = locals()[key][:,0]
            dfields[key][1,1,:] = locals()[key][:,0]

            dfields[key+'_xx'] = locals()['coords'+key][0][:]
            dfields[key+'_xx'] = np.asarray([(dfields[key+'_xx'][i]+dfields[key+'_xx'][i+1])/2 for i in range(0,len(dfields[key+'_xx'])-1)]) #center coords and fix off by 1
        else:
            raise TypeError("This function can only had 1d data right now, TODO: fix this")

    dfields['Vframe_relative_to_sim'] = 0.

    return dfields

def load_flow(flnm_prefix,num,species='ion',polyorder=2,interpLevel=9,xpos=None):
    """
    TODO: NORMALIZE TO VTI,0
    """

    print("Loading flow from %s_%s_M1i_%d.bp..."  % (flnm_prefix,species,num))

    #("simdata/s5-vlasov-low-mach",43)
    #"simdata/s5-vlasov-low-mach_ion_M1i"
    fielddata = pg.data.GData("%s_%s_M1i_%d.bp" % (flnm_prefix,species,num), z0=xpos)
    fieldproj = pg.data.GInterpModal(fielddata, polyorder, "ms", interpLevel)

    num_pos_dim = fielddata.getNumDims() #returns number of spatial dimensions
    #num_vel_dim = len(coords)-num_pos_dim #rest of coordinates are velocity dimensions
    #print("num_pos_dim: ",num_pos_dim," num_vel_dim: ",num_vel_dim)

    coordsux, ux = fieldproj.interpolate(0)
    coordsuy, uy = fieldproj.interpolate(1)
    #coordsuz, uz = fieldproj.interpolate(2)
    coordsuz = coordsux
    uz = np.asarray([[0.] for _ in uy])
    if(ux.shape[-1] != 1):
        raise TypeError("This function can only load data of a singular frame at a time (TODO: implement multiple frame loading)")

    flowkeys = ['ux','uy','uz']
    dflow = {}
    for key in flowkeys:
        if(num_pos_dim == 1): #Spoof the 1D data into 3D. WARNING assumes data is 1D down x axis
            dflow[key] = np.zeros((2,2,len(locals()[key]))) #dfields is typically ordered z,y,x
            dflow[key][0,0,:] = locals()[key][:,0]
            dflow[key][0,1,:] = locals()[key][:,0]
            dflow[key][1,0,:] = locals()[key][:,0]
            dflow[key][1,1,:] = locals()[key][:,0]

            dflow[key+'_xx'] = locals()['coords'+key][0][:]
            dflow[key+'_xx'] = np.asarray([(dflow[key+'_xx'][i]+dflow[key+'_xx'][i+1])/2 for i in range(0,len(dflow[key+'_xx'])-1)]) #center coords and fix off by 1
        else:
            raise TypeError("This function can only had 1d data right now, TODO: fix this")

        dflow['Vframe_relative_to_sim'] = 0.
    return dflow

def spoof_particle_data(hist,vx,vy,vz,x1,x2,y1,y2,z1,z2,Vframe_relative_to_sim,numparticles,verbose=True):
    """
    Samples distribution function to make mock particle data
    """

    #normalize distribution function
    zeromoment = np.sum(hist) #Warning assumes square grid in velocity space
    hist = hist*numparticles/zeromoment

    dpar = {'p1':[],'p2':[],'p3':[],'x1':[],'x2':[],'x3':[],'Vframe_relative_to_sim':Vframe_relative_to_sim}

    dvx = vx[0,0,1] - vx[0,0,0]
    dvy = vy[0,1,0] - vy[0,0,0]
    dvz = vz[1,0,0] - vz[0,0,0]

    _npar = 0 #number of particles generated so far
    for colidx, column in enumerate(hist):
        for rowidx, row in enumerate(column):
            for idx, val in enumerate(row):
                for _ip in range(0,round(val)):
                    #if dv is small, a uniform sampling is sufficiently accurate
                    p1_0 = np.random.uniform(low=vx[colidx,rowidx,idx]-dvx, high=vx[colidx,rowidx,idx]+dvx)
                    p2_0 = np.random.uniform(low=vy[colidx,rowidx,idx]-dvy, high=vy[colidx,rowidx,idx]+dvy)
                    p3_0 = np.random.uniform(low=vz[colidx,rowidx,idx]-dvz, high=vz[colidx,rowidx,idx]+dvz)
                    x1_0 = np.random.uniform(low=x1, high=x2)
                    x2_0 = np.random.uniform(low=y1, high=y2)
                    x3_0 = np.random.uniform(low=z1, high=z2)

                    dpar['p1'].append(p1_0)
                    dpar['p2'].append(p2_0)
                    dpar['p3'].append(p3_0)
                    dpar['x1'].append(x1_0)
                    dpar['x2'].append(x2_0)
                    dpar['x3'].append(x3_0)
                    _npar += 1

    print("Generated ", _npar, " particles...")

    for key in dpar:
        dpar[key] = np.asarray(dpar[key])

    return dpar

def get_input_params(flnm_prefix,num,species='ion',verbose=True):
    """
    """
    if(verbose):
        print("Getting input params from %s_%s_M1i_%d.bp" % (flnm_prefix,species,num))
    fh = adios.file("%s_%s_M1i_%d.bp" % (flnm_prefix,species,num)) #get input file from field data
    inputFile = adios.attr(fh, 'inputfile').value.decode('UTF-8')
    fh.close()
    inputFile = base64.b64decode(inputFile).decode("utf-8")
    inputFile = inputFile.splitlines()

    params = {}
    for ln in inputFile:
        ln = ln.split('=')
        for eidx,elmnt in enumerate(ln):
            ln[eidx] = elmnt.strip()
        if(ln[0] == 'u_shock'):
            params['MachAlfven'] = float(ln[1].split('*')[0])
            params["MachAlfvenNote"] = 'normalized to v_A TODO: compute mach alfven for this run (using inflow as place holder)'
        if(ln[0] == 'beta_proton'):
            params['betaion'] = float(ln[1])
        if(ln[0] == 'beta_electron'):
            params['betaelec'] = float(ln[1])
        if(ln[0] == 'ionCharge'):
            params["qi"] = float(ln[1])
            params["qidesc"] = 'charge to mass ratio (TODO: double check for Gkeyll data that this is true)'

        #parameters used to compute vti
        if(ln[0] == 'vte'):
            ln[1] = ln[1].split()[0]
            try:
                params['vte'] = float(ln[1])
                _vte = params['vte']
            except: #sometimes fraction is passed here
                _numerator,_denom = ln[1].split('/')[0],ln[1].split('/')[1]
                params['vte'] = float(_numerator)/float(_denom)
                _vte = params['vte']
        if(ln[0] == 'ionMass'):
            _ionMass = float(ln[1])
        if(ln[0] == 'elcMass'):
            _elcMass = float(ln[1])

        #parameters used to compute thetaBn
        if(ln[0] == 'local Bz'):
           if(ln[1] == 'B0'):
               params['thetaBn'] = 90
               params["thetaBndesc"] = 'units of degrees'


    if(not('thetaBn') in params.keys()): #TODO: implement automatic calculation of B0
           print("WARNING: thetaBn was not calculated!!!")

    Ti_Te = params['betaion']/params['betaelec'] #Temp_i over Temp_e
    params['frame'] = num
    params['vti'] = _vte*math.sqrt(Ti_Te)/math.sqrt(_ionMass/_elcMass)#vti = vte*math.sqrt(Ti_Te)/math.sqrt(ionMass/elcMass)
    params["di"] = 0.0
    params["didesc"] = 'TODO: compute ion inertial length'

    return params
