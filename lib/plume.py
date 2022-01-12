# plume.py>

# loading and analysis functions related to plume and comparisons with plume

import numpy as np
import math

def load_plume_sweep(flnm):
    """
    Assumes 2 species
    """

    f = open(flnm)

    plume_sweep = {
        "kperp": [],
        "kpar": [],
        "betap": [],
        "vtp": [],
        "w": [],
        "g": [],
        "bxr": [],
        "bxi": [],
        "byr": [],
        "byi": [],
        "bzr": [],
        "bzi": [],
        "exr": [],
        "exi": [],
        "eyr": [],
        "eyi": [],
        "ezr": [],
        "ezi": [],
        "ux1r": [],
        "ux1i": [],
        "uy1r": [],
        "uy1i": [],
        "uz1r": [],
        "uz1i": [],
        "ux2r": [],
        "ux2i": [],
        "uy2r": [],
        "uy2i": [],
        "uz2r": [],
        "uz2i": [],
    }

    line = f.readline()
    while (line != ''):
        line = line.split()
        plume_sweep['kperp'].append(float(line[0]))
        plume_sweep['kpar'].append(float(line[1]))
        plume_sweep['betap'].append(float(line[2]))
        plume_sweep['vtp'].append(float(line[3]))
        plume_sweep['w'].append(float(line[4]))
        plume_sweep['g'].append(float(line[5]))
        plume_sweep['bxr'].append(float(line[6]))
        plume_sweep['bxi'].append(float(line[7]))
        plume_sweep['byr'].append(float(line[8]))
        plume_sweep['byi'].append(float(line[9]))
        plume_sweep['bzr'].append(float(line[10]))
        plume_sweep['bzi'].append(float(line[11]))
        plume_sweep['exr'].append(float(line[12]))
        plume_sweep['exi'].append(float(line[13]))
        plume_sweep['eyr'].append(float(line[14]))
        plume_sweep['eyi'].append(float(line[15]))
        plume_sweep['ezr'].append(float(line[16]))
        plume_sweep['ezi'].append(float(line[17]))
        plume_sweep['ux1r'].append(float(line[18]))
        plume_sweep['ux1i'].append(float(line[19]))
        plume_sweep['uy1r'].append(float(line[20]))
        plume_sweep['uy1i'].append(float(line[21]))
        plume_sweep['uz1r'].append(float(line[22]))
        plume_sweep['uz1i'].append(float(line[23]))
        plume_sweep['ux2r'].append(float(line[24]))
        plume_sweep['ux2i'].append(float(line[25]))
        plume_sweep['uy2r'].append(float(line[26]))
        plume_sweep['uy2i'].append(float(line[27]))
        plume_sweep['uz2r'].append(float(line[28]))
        plume_sweep['uz2i'].append(float(line[29]))

        line = f.readline()

    for key in plume_sweep.keys():
        plume_sweep[key] = np.asarray(plume_sweep[key])

    #normalize B
    plume_sweep['bxr'] = plume_sweep['bxr']*plume_sweep['vtp']
    plume_sweep['byr'] = plume_sweep['byr']*plume_sweep['vtp']
    plume_sweep['bzr'] = plume_sweep['bzr']*plume_sweep['vtp']
    plume_sweep['bxi'] = plume_sweep['bxi']*plume_sweep['vtp']
    plume_sweep['byi'] = plume_sweep['byi']*plume_sweep['vtp']
    plume_sweep['bzi'] = plume_sweep['bzi']*plume_sweep['vtp']

    return plume_sweep

def rotate_and_norm_to_plume_basis(wavemode,epar,eperp1,eperp2):
    """
    Note: plume's basis of x,y,z is not the same as our simulations basis of x,y,z
    """
    from copy import deepcopy
    plume_basis_wavemode = deepcopy(wavemode)

    #by convention we flip coordinate systems if kpar is negative
    if(plume_basis_wavemode['kpar'] < 0):
        epar = _rotate(math.pi,eperp1,epar)
        eperp2 = _rotate(math.pi,eperp1,eperp2)

    #by convention we rotate about epar until kperp2 is zero
    #i.e. we change our basis vectors so that our wavemode is in the span of two of the field align basis vectors, epar and eperp1
    #note:we assume epar, eperp1, and eperp2 are orthonormal #TODO: check for this
    proj = _project_onto_plane(epar,[plume_basis_wavemode['kx'],plume_basis_wavemode['ky'],plume_basis_wavemode['kz']])
    angl = _angle_between_vecs(proj,eperp1) #note this does not tell us the direction we need to rotate, just the amount
    #angl += math.pi #TODO: check if this is the correct basis #NOTE: unless we rotate by this additional pi, our normfactor is off by a sign factor

    #try first direction
    eperp1 = _rotate(angl,epar,eperp1)
    eperp2 = _rotate(angl,epar,eperp2)

    #if failed, try second direction
    if(np.abs(np.dot(eperp2,[wavemode['kx'],wavemode['ky'],wavemode['kz']])) > 0.01):
        eperp1 = _rotate(-2.*angl,epar,eperp1) #times 2 to make up for first rotation
        eperp2 = _rotate(-2.*angl,epar,eperp2)


    #double check rotations
    if(np.abs(np.dot(eperp2,[wavemode['kx'],wavemode['ky'],wavemode['kz']])) > 0.01):
        print("Error, rotation did not result in kperp2 ~= 0")
    if(np.abs(np.dot(epar,eperp1)) > .01 or np.abs(np.dot(eperp1,eperp2)) > .01 or np.abs(np.dot(epar,eperp2)) > .01):
        print("Error, basis is no longer orthogonal...")
    if(np.abs(np.linalg.norm(epar)-1.) > .01 or np.abs(np.linalg.norm(eperp1)-1.) > .01 or np.abs(np.linalg.norm(eperp2)-1.) > .01):
        print("Error, basis is no longer normal...")

    #by convention we normalize so that Eperp1 = 1+0i
    normfactor = np.dot(eperp1,[plume_basis_wavemode['Ex'],plume_basis_wavemode['Ey'],plume_basis_wavemode['Ez']])
    plume_basis_wavemode['Ex'] /= normfactor
    plume_basis_wavemode['Ey'] /= normfactor
    plume_basis_wavemode['Ez'] /= normfactor
    plume_basis_wavemode['Bx'] /= normfactor
    plume_basis_wavemode['By'] /= normfactor
    plume_basis_wavemode['Bz'] /= normfactor

    #recomputed all quantities that are impacted by rotation and normalization
    plume_basis_wavemode['normB'] = np.linalg.norm([plume_basis_wavemode['Bx'],plume_basis_wavemode['By'],plume_basis_wavemode['Bz']])
    plume_basis_wavemode['normE'] = np.linalg.norm([plume_basis_wavemode['Ex'],plume_basis_wavemode['Ey'],plume_basis_wavemode['Ez']])

    _k = [wavemode['kx'],wavemode['ky'],wavemode['kz']]
    _E = [plume_basis_wavemode['Ex'],plume_basis_wavemode['Ey'],plume_basis_wavemode['Ez']]
    _B = [plume_basis_wavemode['Bx'],plume_basis_wavemode['By'],plume_basis_wavemode['Bz']]

    plume_basis_wavemode['kpar'] = np.dot(epar,_k)
    plume_basis_wavemode['kperp1'] = np.dot(eperp1,_k)
    plume_basis_wavemode['kperp2'] = np.dot(eperp2,_k)
    plume_basis_wavemode['kperp'] = math.sqrt(plume_basis_wavemode['kperp1']**2+plume_basis_wavemode['kperp2']**2)

    plume_basis_wavemode['Epar'] = np.dot(epar,_E)
    plume_basis_wavemode['Eperp1'] = np.dot(eperp1,_E)
    plume_basis_wavemode['Eperp2'] = np.dot(eperp2,_E)
    plume_basis_wavemode['Bpar'] = np.dot(epar,_B)
    plume_basis_wavemode['Bperp1'] = np.dot(eperp1,_B)
    plume_basis_wavemode['Bperp2'] = np.dot(eperp2,_B)

    _EcrossB = np.cross(_E,_B)
    plume_basis_wavemode['EcrossBx'] = _EcrossB[0]
    plume_basis_wavemode['EcrossBy'] = _EcrossB[1]
    plume_basis_wavemode['EcrossBz'] = _EcrossB[2]
    plume_basis_wavemode['normEcrossB'] = np.linalg.norm(_EcrossB)
    plume_basis_wavemode['EcrossBpar'] = np.dot(epar,_EcrossB)
    plume_basis_wavemode['EcrossBperp1'] = np.dot(eperp1,_EcrossB)
    plume_basis_wavemode['EcrossBperp2'] = np.dot(eperp2,_EcrossB)

    return plume_basis_wavemode

def _project_onto_plane(norm,vec):
    """
    """
    norm = np.asarray(norm)
    vec = np.asarray(vec)
    projection = vec-np.dot(vec,norm)*norm/(np.linalg.norm(norm)**2.)

    return projection

def _angle_between_vecs(vec1,vec2):
    """
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    tht = np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))

    return tht

def _rotate(tht,rotationaxis,vect):
    """
    """
    rotationaxis = np.asarray(rotationaxis)
    vect = np.asarray(vect)

    #normalize rotationaxis
    ux = rotationaxis[0] / np.linalg.norm(rotationaxis)
    uy = rotationaxis[1] / np.linalg.norm(rotationaxis)
    uz = rotationaxis[2] / np.linalg.norm(rotationaxis)

    #Rotation matrix
    r11 = math.cos(tht)+ux**2.*(1.-math.cos(tht))
    r21 = uy*ux*(1.-math.cos(tht))+uz*math.sin(tht)
    r31 = uz*ux*(1.-math.cos(tht))-uy*math.sin(tht)
    r12 = ux*uy*(1.-math.cos(tht))-uz*math.sin(tht)
    r22 = math.cos(tht)+uy**2.*(1.-math.cos(tht))
    r32 = uz*uy*(1.-math.cos(tht))+ux*math.sin(tht)
    r13 = ux*uz*(1.-math.cos(tht))+uy*math.sin(tht)
    r23 = uy*uz*(1-math.cos(tht))-ux*math.sin(tht)
    r33 = math.cos(tht)+uz**2.*(1.-math.cos(tht))
    R = [[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]]
    #R = [[r11,r21,r31],[r12,r22,r32],[r13,r23,r33]] #TODO: double check if matrix should be inverted

    rotatedvec = np.matmul(R,vect)
    return rotatedvec
