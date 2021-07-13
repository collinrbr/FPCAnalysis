#!/usr/bin/env python
import lib.analysis as anl
import lib.array_ops as ao
import lib.data_h5 as dh5
import lib.data_netcdf4 as dnc
import lib.fpc as fpc
import lib.frametransform as ft
import lib.metadata as md

import lib.plot.oned as plt1d
import lib.plot.twod as plt2d
import lib.plot.debug as pltdebug
import lib.plot.fourier as pltfr
import lib.plot.resultsmanager as rsltmng
import lib.plot.velspace as pltvv

import os
import math
import numpy as np

#-------------------------------------------------------------------------------
# load data
#-------------------------------------------------------------------------------
#TODO: get pathing from analysis input
# path_particles = "/Users/JunoRavin/Documents/PIC-Analysis/tristan/prtl.tot.{:03d}"
# path_fields = "/Users/JunoRavin/Documents/PIC-Analysis/tristan/flds.tot.{:03d}"
path_particles = "/Users/JunoRavin/Documents/PIC-Analysis/tristan/prtl.tot.{:03d}"
path_fields = "/Users/JunoRavin/Documents/PIC-Analysis/tristan/flds.tot.{:03d}"
# NOTE: First pass through TRISTAN data makes it seems like TRISTAN field data
#       has less metadata. Need to check with Colby and Anatoly (JJ 07/10/21)
#       For right now, just read in raw field data.


field_vars = 'ex ey ez bx by bz'.split()
field = {}
with h5py.File(path_fields.format(7),'r') as f:
    for k in field_vars:
        field[k] = f[k][:]

#plt.figure()
#plt.plot(field['by'][0,0,:])
#plt.show()
dparticles_elc, dparticles_ion = data_h5.readTristanParticles(path_particles, 7)
vmaxIon = 0.4
dvIon = 0.01
vxIon, vyIon, vzIon, totalPtclIon, HistIon = data_h5.makeHistFromTristanData(vmaxIon, dvIon, 9500, 10000, 3.0, 4.0, 3.0, 4.0, dparticles_ion, species='i')

#vmaxElc = 2.0
#dvElc = 0.1
#vxElc, vyElc, vzElc, totalPtclElc, HistElc = lf.makeHistFromTristanData(vmaxElc, dvElc, 9000, 9500, 3.0, 4.0, 3.0, 4.0, dparticles_elc, species='e')

#calculate correlation
#CorIonVx = -0.5*vxIon**2.*np.gradient(HistIon, dvIon, edge_order=2, axis=0)
#CorIonVy = -0.5*vyIon**2.*np.gradient(HistIon, dvIon, edge_order=2, axis=1)
CorIonVz = -0.5*vzIon**2.*np.gradient(HistIon, dvIon, edge_order=2, axis=2)

#CorElcVx = 0.5*vxIon**2.*np.gradient(HistElc, dvElc, edge_order=2, axis=0)
#CorElcVy = 0.5*vyIon**2.*np.gradient(HistElc, dvElc, edge_order=2, axis=1)
#CorElcVz = 0.5*vzElc**2.*np.gradient(HistElc, dvElc, edge_order=2, axis=2)

vxIon_xy, vyIon_xy = af.threeVelToTwoVel(vxIon,vyIon,vzIon,'xy')
vxIon_xz, vzIon_xz = af.threeVelToTwoVel(vxIon,vyIon,vzIon,'xz')
vyIon_yz, vzIon_yz = af.threeVelToTwoVel(vxIon,vyIon,vzIon,'yz')

HIon_xy = af.threeHistToTwoHist(HistIon,'xy')
HIon_xz = af.threeHistToTwoHist(HistIon,'xz')
HIon_yz = af.threeHistToTwoHist(HistIon,'yz')

CorIonVz_xy = af.threeHistToTwoHist(CorIonVz,'xy')
CorIonVz_xz = af.threeHistToTwoHist(CorIonVz,'xz')
CorIonVz_yz = af.threeHistToTwoHist(CorIonVz,'yz')

norm = colors.LogNorm(vmin=1.0, vmax=np.max(HIon_xy))
fig, axs = plt.subplots(2,3,figsize=(3*5,2*5))

axs[0,0].pcolormesh(vxIon_xy, vyIon_xy, HIon_xy.transpose(), cmap='plasma', norm=norm, shading="gouraud")
#axs[0,0].pcolormesh(vxIon_xy, vyIon_xy, HIon_xy.transpose(), cmap='plasma', shading="gouraud")
axs[0,0].set_title(r"$\log \left ( f_i(v_x, v_y) \right )$")
axs[0,0].set_xlabel(r"$v_x$")
axs[0,0].set_ylabel(r"$v_y$")
axs[0,0].set_xlim(-vmaxIon,vmaxIon)
axs[0,0].set_ylim(-vmaxIon,vmaxIon)
axs[0,0].grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
plt.setp(axs[0,0], aspect=1.0)

axs[0,1].pcolormesh(vxIon_xz, vzIon_xz, HIon_xz.transpose(), cmap='plasma', norm=norm, shading="gouraud")
#axs[0,1].pcolormesh(vxIon_xz, vzIon_xz, HIon_xz.transpose(), cmap='plasma', shading="gouraud")
axs[0,1].set_title(r"$\log \left ( f_i(v_x, v_z) \right )$")
axs[0,1].set_xlabel(r"$v_x$")
axs[0,1].set_ylabel(r"$v_z$")
axs[0,1].set_xlim(-vmaxIon,vmaxIon)
axs[0,1].set_ylim(-vmaxIon,vmaxIon)
axs[0,1].grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
plt.setp(axs[0,1], aspect=1.0)

axs[0,2].pcolormesh(vyIon_yz, vzIon_yz, HIon_yz.transpose(), cmap='plasma', norm=norm, shading="gouraud")
#axs[0,2].pcolormesh(vyIon_yz, vzIon_yz, HIon_yz.transpose(), cmap='plasma', shading="gouraud")
axs[0,2].set_title(r"$\log \left ( f_i(v_y, v_z) \right )$")
axs[0,2].set_xlabel(r"$v_y$")
axs[0,2].set_ylabel(r"$v_z$")
axs[0,2].set_xlim(-vmaxIon,vmaxIon)
axs[0,2].set_ylim(-vmaxIon,vmaxIon)
axs[0,2].grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
plt.setp(axs[0,2], aspect=1.0)

cmax = np.max(CorIonVz_xz)
cmin = -cmax
axs[1,0].pcolormesh(vxIon_xy, vyIon_xy, CorIonVz_xy.transpose(), vmin=cmin, vmax=cmax, cmap='seismic', shading="gouraud")
axs[1,0].set_title(r"$C_{E_z}$")
axs[1,0].set_xlabel(r"$v_x$")
axs[1,0].set_ylabel(r"$v_y$")
axs[1,0].set_xlim(-vmaxIon,vmaxIon)
axs[1,0].set_ylim(-vmaxIon,vmaxIon)
axs[1,0].grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
plt.setp(axs[1,0], aspect=1.0)

axs[1,1].pcolormesh(vxIon_xz, vzIon_xz, CorIonVz_xz.transpose(), vmin=cmin, vmax=cmax, cmap='seismic', shading="gouraud")
axs[1,1].set_title(r"$C_{E_z}$")
axs[1,1].set_xlabel(r"$v_x$")
axs[1,1].set_ylabel(r"$v_z$")
axs[1,1].set_xlim(-vmaxIon,vmaxIon)
axs[1,1].set_ylim(-vmaxIon,vmaxIon)
axs[1,1].grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
plt.setp(axs[1,1], aspect=1.0)

axs[1,2].pcolormesh(vyIon_yz, vzIon_yz, CorIonVz_yz.transpose(), vmin=cmin, vmax=cmax, cmap='seismic', shading="gouraud")
axs[1,2].set_title(r"$C_{E_z}$")
axs[1,2].set_xlabel(r"$v_y$")
axs[1,2].set_ylabel(r"$v_z$")
axs[1,2].set_xlim(-vmaxIon,vmaxIon)
axs[1,2].set_ylim(-vmaxIon,vmaxIon)
axs[1,2].grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
plt.setp(axs[1,2], aspect=1.0)

plt.show()
