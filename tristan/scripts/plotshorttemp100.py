import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

import lib.loadaux as ld
import lib.ftransfromaux as ft
import lib.analysisaux as aa
import lib.arrayaux as ao
import pickle
import os


shortpickflnm = 'analysisfiles/tempvspos_short_frame100.pkl' 

print("Loading from file: ", shortpickflnm)
filein = open(shortpickflnm, 'rb')
shortdata = pickle.load(filein)
filein.close()
print('done!')

elecxxs = shortdata['elecxxs']
ionxxs = shortdata['ionxxs']
ionparlocalfac = shortdata['ionparlocalfac']
ionparboxfac = shortdata['ionparboxfac']
ionperplocalfac = shortdata['ionperplocalfac']
ionperpboxfac = shortdata['ionperpboxfac']
elecparboxfac = shortdata['elecparboxfac']
elecparlocalfac = shortdata['elecparlocalfac']
elecperpboxfac = shortdata['elecperpboxfac']
elecperplocalfac = shortdata['elecperplocalfac']
iondens = shortdata['iondens']
elecdens = shortdata['elecdens']
Tion_pred_idealadia = shortdata['Tion_pred_idealadia']
Tion_pred_doubleadia = shortdata['Tion_pred_doubleadia']
Telec_pred_idealadia = shortdata['Telec_pred_idealadia']
Telec_pred_doubleadia = shortdata['Telec_pred_doubleadia']
Btot = shortdata['Btot']

distdata = {'elecxxs':elecxxs,'ionxxs':ionxxs}

def plot1dquant(xarr,yarr,ylabel,flnm):
    plt.figure(figsize=(10,4))
    plt.style.use("cb.mplstyle")
    plt.plot(xarr,yarr,color='black')
    plt.xlabel(r'$x / d_i$')
    plt.ylabel(ylabel)
#    plt.xlim(5,12)
    plt.grid()
    plt.gca().tick_params(axis='both', direction='in',top=True, bottom=True, right=True, left=True)
    plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
    plt.close()

os.system('mkdir figures')
os.system('mkdir figures/tempvspos_frame100')

flnmprefix = 'figures/tempvspos_frame100/'

flnm = flnmprefix+'ionparlocalfac.png'
ylabel = r'$T_{||,i,local}/T_{||,0,i}$'
plot1dquant(distdata['ionxxs'],ionparlocalfac,ylabel,flnm)

flnm = flnmprefix+'ionparboxfac.png'
ylabel = r'$T_{||,i,box}/T_{||,0,i}$'
plot1dquant(distdata['ionxxs'],ionparboxfac,ylabel,flnm)

flnm = flnmprefix+'ionperplocalfac.png'
ylabel = r'$T_{\perp,i,local}/T_{\perp,,0,i}$'
plot1dquant(distdata['ionxxs'],ionperplocalfac,ylabel,flnm)

flnm = flnmprefix+'ionperpboxfac.png'
ylabel = r'$T_{\perp,i,box}/T_{\perp,0,i}$'
plot1dquant(distdata['ionxxs'],ionperpboxfac,ylabel,flnm)

flnm = flnmprefix+'elecparlocalfac.png'
ylabel = r'$T_{||,e,local}/T_{||,0,e}$'
plot1dquant(distdata['elecxxs'],elecparlocalfac,ylabel,flnm)

flnm = flnmprefix+'elecparboxfac.png'
ylabel = r'$T_{||,e,box}/T_{||,0,e}$'
plot1dquant(distdata['elecxxs'],elecparboxfac,ylabel,flnm)

flnm = flnmprefix+'elecperplocalfac.png'
ylabel = r'$T_{\perp,e,local}/T_{\perp,0,e}$'
plot1dquant(distdata['elecxxs'],elecperplocalfac,ylabel,flnm)

flnm = flnmprefix+'elecperpboxfac.png'
ylabel = r'$T_{\perp,e,box}/T_{\perp,0,e}$'
plot1dquant(distdata['elecxxs'],elecperpboxfac,ylabel,flnm)

flnm = flnmprefix+'iondens.png'
ylabel = r'$n_i/n_{0,i}$'
plot1dquant(distdata['ionxxs'],iondens,ylabel,flnm)

flnm = flnmprefix+'elecdens.png'
ylabel = r'$n_e/n_{0,e}$'
plot1dquant(distdata['elecxxs'],elecdens,ylabel,flnm)

flnm = flnmprefix+'Tion_pred_idealadia.png'
ylabel = r'$T_{\gamma=5/3,i}/T_{0,i}$'
plot1dquant(distdata['ionxxs'],Tion_pred_idealadia,ylabel,flnm)

flnm = flnmprefix+'Tion_pred_doubleadia.png'
ylabel = r'$T_{double,i}/T_{0,i}$'
plot1dquant(distdata['ionxxs'],Tion_pred_doubleadia,ylabel,flnm)

flnm = flnmprefix+'Telec_pred_idealadia.png'
ylabel = r'$T_{\gamma=5/3,e}/T_{0,e}$'
plot1dquant(distdata['elecxxs'],Telec_pred_idealadia,ylabel,flnm)

flnm = flnmprefix+'Telec_pred_doubleadia.png'
ylabel = r'$T_{double,e}/T_{0,e}$'
plot1dquant(distdata['elecxxs'],Telec_pred_doubleadia,ylabel,flnm)

flnm = flnmprefix+'Btot.png'
ylabel = r'$|B|/|B_0|$'
plot1dquant(distdata['ionxxs'],Btot,ylabel,flnm)

#make 1d plots with multiple lines
plt.figure(figsize=(10,6))
plt.style.use("cb.mplstyle")
plt.plot(distdata['ionxxs'],ionparlocalfac,label=r'$T_{||,i,local}/T_{||,0,i}$',color='red',ls='dashed')
plt.plot(distdata['ionxxs'],ionparboxfac,label=r'$T_{||,i,box}/T_{||,0,i}$',color='blue',ls='dashdot')
plt.plot(distdata['ionxxs'],ionperplocalfac,label=r'$T_{\perp,i,local}/T_{\perp,0,i}$',color='orange',ls='dashed')
plt.plot(distdata['ionxxs'],ionperpboxfac,label=r'$T_{\perp,i,box}/T_{\perp,0,i}$',color='green',ls='dashdot')
plt.plot(distdata['ionxxs'],iondens,label=r'$n_i/n_{0,i}$',color='black',ls='solid')
plt.plot(distdata['ionxxs'],Tion_pred_idealadia,label=r'$T_{\gamma=5/3,i}/T_{0,i}$',color='purple',ls='dotted')
plt.plot(distdata['ionxxs'],Tion_pred_doubleadia,label=r'$T_{double,i}/T_{0,i}$',color='gray',ls='dotted')
plt.xlabel(r'$x / d_i$')
plt.legend()
plt.xlim(0,12)
plt.grid()
plt.gca().tick_params(axis='both', direction='in',top=True, bottom=True, right=True, left=True)
plt.savefig(flnmprefix+'1diontempvspos.png',format='png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,6))
plt.style.use("cb.mplstyle")
plt.plot(distdata['elecxxs'],elecparlocalfac,label=r'$T_{||,e,local}/T_{||,0,e}$',color='red',ls='dashed')
plt.plot(distdata['elecxxs'],elecparboxfac,label=r'$T_{||,e,box}/T_{||,0,e}$',color='blue',ls='dashdot')
plt.plot(distdata['elecxxs'],elecperplocalfac,label=r'$T_{\perp,e,local}/T_{\perp,0,e}$',color='orange',ls='dashed')
plt.plot(distdata['elecxxs'],elecperpboxfac,label=r'$T_{\perp,e,box}/T_{\perp,0,e}$',color='green',ls='dashdot')
plt.plot(distdata['elecxxs'],elecdens,label=r'$n_e/n_{0,e}$',color='black',ls='solid')
plt.plot(distdata['elecxxs'],Telec_pred_idealadia,label=r'$T_{\gamma=5/3,e}/T_{0,e}$',color='purple',ls='dotted')
plt.plot(distdata['elecxxs'],Telec_pred_doubleadia,label=r'$T_{double,e}/T_{0,e}$',color='gray',ls='dotted')
plt.xlabel(r'$x / d_i$')
plt.legend()
plt.xlim(0,12)
plt.grid()
plt.gca().tick_params(axis='both', direction='in',top=True, bottom=True, right=True, left=True)
plt.savefig(flnmprefix+'1delectempvspos.png',format='png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,6))
plt.style.use("cb.mplstyle")
plt.plot(distdata['elecxxs'],elecparlocalfac,label=r'$T_{||,e,local}/T_{||,0,e}$',color='red',ls='dashed')
plt.plot(distdata['elecxxs'],elecparboxfac,label=r'$T_{||,e,box}/T_{||,0,e}$',color='blue',ls='dashdot')
plt.plot(distdata['elecxxs'],elecperplocalfac,label=r'$T_{\perp,e,local}/T_{\perp,0,e}$',color='orange',ls='dashed')
plt.plot(distdata['elecxxs'],elecperpboxfac,label=r'$T_{\perp,e,box}/T_{\perp,0,e}$',color='green',ls='dashdot')
plt.plot(distdata['elecxxs'],elecdens,label=r'$n_e/n_{0,e}$',color='black',ls='solid')
plt.plot(distdata['elecxxs'],Telec_pred_idealadia,label=r'$T_{\gamma=5/3,e}/T_{0,e}$',color='purple',ls='dotted')
plt.plot(distdata['elecxxs'],Telec_pred_doubleadia,label=r'$T_{double,e}/T_{0,e}$',color='gray',ls='dotted')
plt.xlabel(r'$x / d_i$')
plt.legend()
plt.xlim(7,9)
plt.grid()
plt.gca().tick_params(axis='both', direction='in',top=True, bottom=True, right=True, left=True)
plt.savefig(flnmprefix+'1delectempvspos_zoomed_in.png',format='png',dpi=300,bbox_inches='tight')
plt.close()


plt.figure(figsize=(10,6))
plt.style.use("cb.mplstyle")
plt.plot(distdata['ionxxs'],np.gradient(ionparlocalfac),label=r'$\frac{d}{dx}T_{||,i,local}/T_{||,0,i}$',color='red',ls='dashed')
plt.plot(distdata['ionxxs'],np.gradient(ionparboxfac),label=r'$\frac{d}{dx}T_{||,i,box}/T_{||,0,i}$',color='blue',ls='dashdot')
plt.plot(distdata['ionxxs'],np.gradient(ionperplocalfac),label=r'$\frac{d}{dx}T_{\perp,i,local}/T_{\perp,0,i}$',color='orange',ls='dashed')
plt.plot(distdata['ionxxs'],np.gradient(ionperpboxfac),label=r'$\frac{d}{dx}T_{\perp,i,box}/T_{\perp,0,i}$',color='green',ls='dashdot')
plt.plot(distdata['ionxxs'],np.gradient(iondens),label=r'$\frac{d}{dx}n_i/n_{0,i}$',color='black',ls='solid')
plt.plot(distdata['ionxxs'],np.gradient(Tion_pred_idealadia),label=r'$\frac{d}{dx}T_{\gamma=5/3,i}/T_{0,i}$',color='purple',ls='dotted')
plt.plot(distdata['ionxxs'],np.gradient(Tion_pred_doubleadia),label=r'$\frac{d}{dx}T_{double,i}/T_{0,i}$',color='gray',ls='dotted')
plt.xlabel(r'$x / d_i$')
plt.legend()
plt.xlim(5,12)
plt.grid()
plt.gca().tick_params(axis='both', direction='in',top=True, bottom=True, right=True, left=True)
plt.savefig(flnmprefix+'1dratesiontempvspos.png',format='png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,6))
plt.style.use("cb.mplstyle")
plt.plot(distdata['elecxxs'],np.gradient(elecparlocalfac),label=r'$\frac{d}{dx}T_{||,e,local}/T_{||,0,e}$',color='red',ls='dashed')
plt.plot(distdata['elecxxs'],np.gradient(elecparboxfac),label=r'$\frac{d}{dx}T_{||,e,box}/T_{||,0,e}$',color='blue',ls='dashdot')
plt.plot(distdata['elecxxs'],np.gradient(elecperplocalfac),label=r'$\frac{d}{dx}T_{perp,e,local}/T_{\perp,0,e}$',color='orange',ls='dashed')
plt.plot(distdata['elecxxs'],np.gradient(elecperpboxfac),label=r'$\frac{d}{dx}T_{perp,e,box}/T_{\perp,0,e}$',color='green',ls='dashdot')
plt.plot(distdata['elecxxs'],np.gradient(elecdens),label=r'$\frac{d}{dx}n_e/n_{0,e}$',color='black',ls='solid')
plt.plot(distdata['elecxxs'],np.gradient(Telec_pred_idealadia),label=r'$\frac{d}{dx}T_{\gamma=5/3,e}/T_{0,e}$',color='purple',ls='dotted')
plt.plot(distdata['elecxxs'],np.gradient(Telec_pred_doubleadia),label=r'$\frac{d}{dx}T_{double,e}/T_{0,e}$',color='gray',ls='dotted')
plt.xlabel(r'$x / d_i$')
plt.legend(loc ="upper right")
plt.xlim(5,12)
plt.grid()
plt.gca().tick_params(axis='both', direction='in',top=True, bottom=True, right=True, left=True)
plt.savefig(flnmprefix+'1drateselectempvspos.png',format='png',dpi=300,bbox_inches='tight')
plt.close()
