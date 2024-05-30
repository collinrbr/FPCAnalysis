import numpy as np

vmaxion = 15
dvion = .25


vxbins = np.arange(-vmaxion, vmaxion+dvion, dvion)
vx = (vxbins[1:] + vxbins[:-1])/2.
vybins = np.arange(-vmaxion, vmaxion+dvion, dvion)
vy = (vybins[1:] + vybins[:-1])/2.
vzbins = np.arange(-vmaxion, vmaxion+dvion, dvion)
vz = (vzbins[1:] + vzbins[:-1])/2.

hist,_ = np.histogramdd(([1.,2.],[2.,3.],[4.,5.]), bins=[vzbins, vybins, vxbins])

print(hist)
