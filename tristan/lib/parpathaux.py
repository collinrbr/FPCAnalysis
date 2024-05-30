import numpy as np

import lib.loadaux as ld

def get_path_of_particle(datapath,ind,proc,startframenum,endframenum,spec='ion',verbose=False, stride=1, normalize=True):
    """
    Returns dict of position and velocity of particle for each frame in between start and endframenum 

    Together ind and proc are used to assign a unique ID to each particle

    Parameters
    ----------
    datapath : string
        path to *output* folder (typically will include 'output' in string) (e.q. /data/simulation/tristan/runname1/output/) 
    ind : int
        index of particle on specified processor number
    proc : int
        processor number
    (start/end)framenum : string
        start/end framenum
        *this is a string e.g. '015' for frame 15
    spec : string
        species name
        'ion' or 'elec'
    verbose : bool
        if true, prints progress/status of script
    stride : int
        number of steps between sampled frames
        *this is different from simulation stride- this stride is the stride between frames sampled to measure particle position/ velocity*
    normalize : bool
        if true, normalizes position to ion inertial length, d_i, and veloctiy to v_{thermal,species,upstream}

    Returns
    -------
    parpath : dict
        position and velocity of particle for each sampled frame 
    """

    #dpar: data dict
    #ind: index of particle on specified processor number
    #proc: proc nun
    #startframenum: (string e.g. '005'; inclusive)
    #endframenum: (string e.g. '010'; inclusive)
    #spec: 'ion' or 'elc'
    parpath = {'x': [],'y': [],'z': [],'ux': [],'uy': [],'uz': []}
    
    currentframenum = startframenum
    startidx = int(startframenum)
    endidx =  int(endframenum)
    for _i in np.arange(startidx,endidx+1,stride):
        currentframenum = str(_i).zfill(3)
        if(verbose): print(currentframenum)
        
        dpar_elec, dpar_ion = ld.load_particles(datapath,currentframenum,normalizeVelocity=normalize)
        
        candidate_inds = []
        if(spec=='ion'):
            prockey = 'proci'
            indkey = 'indi'
            xkey = 'xi'
            ykey = 'yi'
            zkey = 'zi'
            uxkey = 'ui'
            uykey = 'vi'
            uzkey = 'wi'
            dpar = dpar_ion
            del dpar_elec
        elif(spec=='elec'):
            prockey = 'proce'
            indkey = 'inde'
            xkey = 'xe'
            ykey = 'ye'
            zkey = 'ze'
            uxkey = 'ue'
            uykey = 've'
            uzkey = 'we'
            dpar = dpar_elec
            del dpar_ion
        else:
            print("Spec is either ion or elec")
            return
        
        candidate_inds = [ i for i, e in enumerate(dpar[indkey]) if (ind == e)] #find index of matches
        candidate_procs = [ e for i, e in enumerate(list(map(dpar[prockey].__getitem__, candidate_inds)))] #find proces of matches
        
        candidates = [candidate_inds[i] for i,e in enumerate(candidate_procs) if (proc == e)]
                          
        if(len(candidates) != 1):
            print("Error! Did not find particle!")
            print("Candidate_index,ind,proc: ",candidate_inds,ind,proc)
            return None
        
        index = candidates[0]
           
        parpath['x'].append(dpar[xkey][index])
        parpath['y'].append(dpar[ykey][index])
        parpath['z'].append(dpar[zkey][index])
        parpath['ux'].append(dpar[uxkey][index])
        parpath['uy'].append(dpar[uykey][index])
        parpath['uz'].append(dpar[uzkey][index])
        
    return parpath

