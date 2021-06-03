#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import os
import time
plt.style.use("postgkyl.mplstyle") #sets style parameters for matplotlib plots

#-------------------------------------------------------------------------------
# Functions for parsing data
#-------------------------------------------------------------------------------

#function that takes 1d vx, vy arrays, and formats them for pcolormesh
#Input: 1d vx, vy arrays. Data describing velocity grid. Assumes rectangular grid
#Output: 2d vxgrid, vygrid. Formatted
def getvelgrid(vx,vy):
    vxgrid = []
    vygrid = []

    #if passed netcdf4 object, this will grab wanted data.
    #indexing netcdf4 object is time consuming
    vx = vx[:]
    vy = vy[:]
    for i in range(0,len(vx)):
        rowvxgrid = []
        rowvygrid = []
        for j in range(0,len(vy)):
            rowvxgrid.append(vx[i])
            rowvygrid.append(vy[j])
        vxgrid.append(rowvxgrid)
        vygrid.append(rowvygrid)
    return vxgrid, vygrid

#function that grabs singular CEx(vx,vy;x) point
#Input: Integers xpt, vxpt, vypt. Index of desired point
#       nc varaible object (or 3d array). Data set for CEx
#Output: float CExpt. CEx data found at requested location
def getCExpointfromindex(xpt,vxpt,vypt,CExdata):
    CExpt = CExdata[xpt][vxpt][vypt]
    return CExpt

#function that grabs singular CEy(vx,vy;x) point
#Input: Integers xpt, vxpt, vypt. Index of desired point
#       nc varaible object (or 3d array). Data set for CEy
#Output: float CEypt. CEx data found at requested location
def getCEypointfromindex(xpt,vxpt,vypt,CEydata):
    CEypt = CEydata[xpt][vxpt][vypt]
    return CEypt

#function that finds index nearest to x value.
#Input: float xval. Desired x value
#       nc varaible object (or 1d array) xdata. Data set for x
#Output: integer xindex. Index with closest value
def getnearestx(xval,xdata):
    xdata = np.asarray(xdata)
    xindex = (np.abs(xdata - xval)).argmin()
    return xindex

#function that finds index nearest to vx value.
#Input: float vxval. Desired vx value
#       nc varaible object (or 1d array) vxdata. Data set for vx
#Output: integer vxindex. Index with closest value
def getnearestvx(vxval,vxdata):
    vxdata = np.asarray(vxdata)
    vxindex = (np.abs(vxdata - vxval)).argmin()
    return vxindex

#function that finds index nearest to vy value.
#Input: float vyval. Desired vy value
#       nc varaible object (or 1d array) vydata. Data set for vy
#Output: integer vyindex. Index with closest value
def getnearestvy(vyval,vydata):
    vydata = np.asarray(vydata)
    vyindex = (np.abs(vydata - vyval)).argmin()
    return vyindex

#function that gets CEx data at requested (physical space) value
#Input: float vx, vy, x. Desired data point
#       nc varaible object (or 3d array) CExdata. Data set for CEx
#Output: float CExpt. CEx data at point
def getCEx(x,vx,vy,xdata,vxdata,vydata,CExdata):
    #double check that data is in a 'python array' for computational efficiency
    xdata = xdata[:]
    vxdata = vxdata[:]
    vydata = vydata[:]

    xindex = getnearestx(x,xdata)
    vxindex = getnearestvx(vx,vxdata)
    vyindex = getnearestvy(vy,vydata)

    CExpt = CEx[xindex][vxindex][vyindex] #Want to avoid loading entire 3d array if possible at once. Could use a lot of ram
    return CExpt

#function that gets CEx data at requested (physical space) value
#Input: float vx, vy, x. Desired data point
#       nc varaible object (or 3d array) CEydata. Data set for CEy
#Output: float CEypt. CEy data at point
def getCEy(x,vx,vy,xdata,vxdata,vydata,CEydata):
    #double check that data is in a 'python array' for computational efficiency
    xdata = xdata[:]
    vxdata = vxdata[:]
    vydata = vydata[:]

    xindex = getnearestx(x,xdata)
    vxindex = getnearestvx(vx,vxdata)
    vyindex = getnearestvy(vy,vydata)

    CEypt = CEy[xindex][vxindex][vyindex] #Want to avoid loading entire 3d array if possible at once. Could use a lot of ram
    return CEypt

#Grabs max abs(CEx) in entire data set
#Input: nc varaible object (or 3d array) CEx. Data set for CEx
#Output: float max. Max of abs(data)
def getmaxabsCEx(CEx):
    max = 0.
    for k in range(0, len(CEx)): #Takes a lot of ram to load entire CEx
        temparr = CEx[k]
        tempmax = np.amax(abs(temparr))
        if(tempmax > max):
            max = tempmax
    return max

#Grabs max abs(CEy) in entire data set
#Input: nc varaible object (or 3d array) CEy. Data set for CEy
#Output: float max.  Max of abs(data)
def getmaxabsCEy(CEy):
    max = 0.
    for k in range(0, len(CEy)): #Takes a lot of ram to load entire CEy
        temparr = CEy[k]
        tempmax = np.amax(abs(temparr))
        if(tempmax > max):
            max = tempmax
    return max


#-------------------------------------------------------------------------------
# Sample script for making plots from netcdf4 velocity signature data file
#-------------------------------------------------------------------------------

#load data
filename = "dHybridRSDAtestonAlven1withlorentzv1.nc"
ncin = Dataset(filename, 'r', format='NETCDF4')

#grab data
#note: these are 'netcdf4 varaible' objects, and are generally slow to index
#ex: use x[:] to grab array data and load into ram
print("Loading data...")
start_time = time.time()
x = ncin.variables['x']
vx = ncin.variables['vx']
vy = ncin.variables['vy']
CEx = ncin.variables['C_Ex']
CEy = ncin.variables['C_Ey']
metadata = ncin.variables['metadata']
print("Time to link data: %s seconds" % (time.time() - start_time))

#plot parameters
vmax = np.max(np.abs(vx[:]))
xval = 27.0 #ion inertial lengths
xposindex = getnearestx(xval,x[:])
print(xposindex)
vxgrid, vygrid = getvelgrid(vx,vy)
print("")
print(vx[:])
print(vy[:])
start_time = time.time()
CExclrbarbounds = getmaxabsCEx(CEx) #warning: this changes dependent on time slice. Should consider some global bounds for colorbar or train using 'unnormalized data'
CEyclrbarbounds = getmaxabsCEy(CEy) #warning: this changes dependent on time slice. Should consider some global bounds for colorbar or train using 'unnormalized data'
print("Time to find max of abs(data): %s seconds" % (time.time() - start_time))


start_time = time.time()
for i in range(0,len(CEx)):
    #make CEx plot
    plt.figure(figsize=(6.5,6))
    plt.pcolormesh(vxgrid, vygrid, CEx[i], vmax=CExclrbarbounds, vmin=-CExclrbarbounds, cmap="seismic", shading="gouraud")
    plt.xlim(-vmax, vmax)
    plt.ylim(-vmax, vmax)
    plt.xticks(np.linspace(-vmax, vmax, 9))
    plt.yticks(np.linspace(-vmax, vmax, 9))
    plt.title(r"$C_{E_x}(v_x, v_y; x = $"+ str(x[i]) +"$ d_i)$", loc="right")
    plt.xlabel(r"$v_x/v_{ti}$")
    plt.ylabel(r"$v_y/v_{ti}$")
    plt.grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    clb = plt.colorbar(format="%.1f", ticks=np.linspace(-CExclrbarbounds, CExclrbarbounds, 8), fraction=0.046, pad=0.04) #TODO: make static colorbar based on max range of CEx, CEy
    plt.setp(plt.gca(), aspect=1.0)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig("graphsCEx/CExxposindex"+str(i).zfill(3)+".png", dpi=100) #TODO: rename
    plt.close()

    #make CEy plot
    plt.figure(figsize=(6.5,6))
    plt.pcolormesh(vxgrid, vygrid, CEy[i], vmax=CEyclrbarbounds, vmin=-CEyclrbarbounds, cmap="seismic", shading="gouraud")
    plt.xlim(-vmax, vmax)
    plt.ylim(-vmax, vmax)
    plt.xticks(np.linspace(-vmax, vmax, 9))
    plt.yticks(np.linspace(-vmax, vmax, 9))
    plt.title(r"$C_{E_y}(v_x, v_y; x = $"+ str(x[i]) +"$ d_i)$", loc="right")
    plt.xlabel(r"$v_x/v_{ti}$")
    plt.ylabel(r"$v_y/v_{ti}$")
    plt.grid(color="k", linestyle="-", linewidth=1.0, alpha=0.6)
    clb = plt.colorbar(format="%.1f", ticks=np.linspace(-CEyclrbarbounds, CEyclrbarbounds, 8), fraction=0.046, pad=0.04)
    plt.setp(plt.gca(), aspect=1.0)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig("graphsCEy/CEyxposindex"+str(i).zfill(3)+".png", dpi=100) #TODO: rename
    plt.close()

print("Time to make CEx, CEy plots: %s seconds" % (time.time() - start_time))

#make gif
import imageio
import os
images = []
directory = 'graphsCEx/'
filenames =  os.listdir(directory)
filenames = sorted(filenames)
filenames = filenames[1:] #quick way to remove .DS_store
for filename in filenames:
    images.append(imageio.imread(directory+filename))
imageio.mimsave('CEx.gif', images)

images = []
directory = 'graphsCEy/'
filenames =  os.listdir(directory)
filenames = sorted(filenames)
filenames = filenames[1:] #quick way to remove .DS_store
for filename in filenames:
    images.append(imageio.imread(directory+filename))
imageio.mimsave('CEy.gif', images)
