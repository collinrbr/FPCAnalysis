import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa
import lib.parpathaux as pp

#user params
framenum = '700'
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
frames = ["{:03d}".format(_i) for _i in range(690,711)]

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
#dfields_many_frames = {'frame':[],'dfields':[]}
#for _num in frames:
#    num = int(_num)
#    d = ld.load_fields(flpath,_num,normalizeFields=normalize)
#    dfields_many_frames['dfields'].append(d)
#    dfields_many_frames['frame'].append(num)
#vshock, xshockvals, v0 = ft.shock_from_ex_cross(dfields_many_frames,dt)
print("warning: overwriting vshock...")
vshock = 1.5
dfields = ft.lorentz_transform_vx(dfields,vshock,c) #note: we only boost one frame

dpar_elec, dpar_ion = ld.load_particles(flpath,framenum,normalizeVelocity=normalize)
inputpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/input'
inputs = ld.load_input(inputpath)
beta0 = aa.compute_beta0(params,inputs)
dpar_ion = ft.shift_particles(dpar_ion, vshock/np.sqrt(beta0)) #*/np.sqrt(beta0) converts from normalization by va to vth
dpar_elec = ft.shift_particles(dpar_elec, vshock/np.sqrt(beta0))

dpar = dpar_elec

dpar['x1'] = dpar['xe']
dpar['x2'] = dpar['ye']
dpar['x3'] = dpar['ze']
dpar['p1'] = dpar['ue']
dpar['p2'] = dpar['ve']
dpar['p3'] = dpar['we']

xlim = [8.125,8.375]
ylim = [0,5]

if(z1 != None and z2 != None):
    gptsparticle = (xlim[0] <= dpar['x1']) & (dpar['x1'] <= xlim[1]) & (ylim[0] <= dpar['x2']) & (dpar['x2'] <= ylim[1])

array = dpar['ve'][gptsparticle]


#make histograms
print("Making histogram")
# Calculate the histogram bin edges and frequencies
bins = np.linspace(np.min(array),np.max(array),100)
hist, bins = np.histogram(array, bins=bins)

# Calculate the bin centers for plotting as a line
bin_centers = (bins[:-1] + bins[1:]) / 2

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, hist, color='blue', marker='o', linestyle='-')

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')

# Show the plot
plt.savefig('elecdist.png',format='png',dpi=300)
plt.close()
