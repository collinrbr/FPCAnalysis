
#quick script to plot figure by loading pickle made by wft.py


import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np
import pickle

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa
import lib.arrayaux as ao #array operations
import lib.wavemodeaux as wa
import lib.plotwftaux as pw

import os

os.system('mkdir figures')
os.system('mkdir analysisfiles')
os.system('mkdir figures/spectra')
os.system('mkdir figures/spectra/debug')

pregeneratedflnm = 'dwavemodes8.5.pickle' 
print("Loading from file: ", pregeneratedflnm)
filein = open(pregeneratedflnm, 'rb')
dwavemodes = pickle.load(filein)
filein.close()

flnmprefix = 'figures/spectra/'

xxpos = 8.5

#2D (k_component1 vs k_component2) power spectrum plot
speckey = 'Epar' #'Epar' or 'normE' typically
klim = 50

gridsize1=150*1
gridsize2=150*8
gridsize3=150*1

hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,gridsize3=gridsize3,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,gridsize3=gridsize3,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)
speckey = 'Eperp1' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,gridsize3=gridsize3,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,gridsize3=gridsize3,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)
speckey = 'Eperp2' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,gridsize3=gridsize3,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,gridsize3=gridsize3,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)
speckey = 'normE' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,gridsize3=gridsize3,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,gridsize3=gridsize3,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)


speckey = 'Ex' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,gridsize3=gridsize3,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)

speckey = 'Ey' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,gridsize3=gridsize3,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)

speckey = 'Ez' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,gridsize3=gridsize3,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)

speckey = 'Bpar' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)
speckey = 'Bperp1' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)
speckey = 'Bperp2' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)
speckey = 'normB' #'Epar' or 'normE' typically
klim = 50
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum(dwavemodes,key=speckey,flnm=flnmprefix+'wavenumber_spectrum_'+speckey+'xpos'+str(xxpos)+'.png',kperp1lim = klim, kperp2lim = klim, kparlim  = klim)
hxbin0, hxbin1, hxbin2 = pw.plot_power_spectrum_cart(dwavemodes,key=speckey,gridsize1=gridsize1,gridsize2=gridsize2,flnm=flnmprefix+'wavenumber_spectrum_cart_'+speckey+'xpos'+str(xxpos)+'.png',kxlim = klim, kylim = klim, kzlim  = klim)

