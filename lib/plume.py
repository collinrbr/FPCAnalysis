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

    return plume_sweep

def rotate_and_norm_to_plume_basis(wavemode,epar,eperp1,eperp2):
    """
    Note: plume's basis of x,y,z is not the same as our simulations basis of x,y,z
    """
    from copy import deepcopy
    plume_basis_wavemode = deepcopy(wavemode)

    #by convention we rotate until kperp2 is zero
    #https://en.wikipedia.org/wiki/Rotation_matrix
    #see section Rotation matrix from axis and angle


    #by convention we normalize until E = 1+0i
    normfactor = deepcopy(plume_basis_wavemode['Eperp1']) #must rotate before we normalize
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
