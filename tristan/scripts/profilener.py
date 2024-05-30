import sys
sys.path.append(".")
sys.path.append('..')
sys.path.append('../..')


import pickle
import numpy as np
import matplotlib.pyplot as plt

import lib.loadaux as ld

def interpolate(independent_vars, dependent_vars, locations):
    independent_vars = np.array(independent_vars)
    dependent_vars = np.array(dependent_vars)
    locations = np.array(locations)
    interpolated_values = np.interp(locations, independent_vars, dependent_vars)
    
    return locations, interpolated_values


def interpolate2(independent_vars, dependent_vars, locations):
    independent_vars = np.array(independent_vars)
    dependent_vars = np.array(dependent_vars)
    locations = np.array(locations)
    
    # Determine the scaling factors
    scale_factor_x = np.max(np.abs(independent_vars))
    scale_factor_y = np.max(np.abs(dependent_vars))
    
    # Scale the data
    scaled_independent_vars = independent_vars / scale_factor_x
    scaled_dependent_vars = dependent_vars / scale_factor_y
    scaled_locations = locations / scale_factor_x
    
    # Interpolate the scaled values
    scaled_interpolated_values = np.interp(scaled_locations, scaled_independent_vars, scaled_dependent_vars)
    
    # Inverse scaling for interpolated values
    interpolated_values = scaled_interpolated_values * scale_factor_y
    
    return locations, interpolated_values


pregeneratedflnm = 'enerpick.pickle'
print("Loading from file: ", pregeneratedflnm)
filein = open(pregeneratedflnm, 'rb')
enerpick = pickle.load(filein)
filein.close()

interpolxxs = np.arange(0,12.1,.1) 


framenum = '700'
flpath = '/data/not_backed_up/simulation/Tristan/Mar232023/htg_perp_bigrun/output/'
normalize=True
dden = ld.load_den(flpath,framenum,normalize=normalize)
ddenelec = dden['dens'].mean(axis=(0,1))/2.
ddenelec[0] = 1. #debug!
xxdden = dden['dens_xx'][:]

xxdden, ddenelec=interpolate(xxdden,ddenelec,interpolxxs)

print(ddenelec)

#load data
Wfacflucinte = np.flip(enerpick['enerCEtot_elec_facfluclocframe_egain'])
xxWfacflucinte = np.flip(enerpick['_xcoord_egain'])
xxWfacflucinte,Wfacflucinte = interpolate(xxWfacflucinte,Wfacflucinte,interpolxxs)

Wfacavginte = np.flip(enerpick['enerCEtot_elec_facavglocframe_egain'])
xxWfacavginte = np.flip(enerpick['_xcoord_egain'])
xxWfacavginte,Wfacavginte = interpolate(xxWfacavginte,Wfacavginte,interpolxxs)

Wfacflucinti = np.flip(enerpick['enerCEtot_ion_facfluclocframe_egain'])
xxWfacflucinti = np.flip(enerpick['_xcoord_egain'])
xxWfacflucinti,Wfacflucinti = interpolate(xxWfacflucinti,Wfacflucinti,interpolxxs)

Wfacavginti = np.flip(enerpick['enerCEtot_ion_facavglocframe_egain'])
xxWfacavginti = np.flip(enerpick['_xcoord_egain'])
xxWfacavginti,Wfacavginti = interpolate(xxWfacavginti,Wfacavginti,interpolxxs)

Wfacavge = np.flip(enerpick['enerCEtot_elec_facavg_egain'])
xxWfacavge = np.flip(enerpick['xcoord_egain'])
xxWfacavge,Wfacavge = interpolate(xxWfacavge,Wfacavge,interpolxxs)

Wfacfluce = np.flip(enerpick['enerCEtot_elec_facfluc_egain'])
xxWfacfluce = np.flip(enerpick['xcoord_egain'])
xxWfacfluce,Wfacfluce = interpolate(xxWfacfluce,Wfacfluce,interpolxxs)

Wfacavgi = np.flip(enerpick['enerCEtot_ion_facavg_egain'])
xxWfacavgi = np.flip(enerpick['xcoord_egain'])
xxWfacavgi,Wfacavgi = interpolate(xxWfacavgi,Wfacavgi,interpolxxs)

Wfacfluci = np.flip(enerpick['enerCEtot_ion_facfluc_egain'])
xxWfacfluci = np.flip(enerpick['xcoord_egain'])
xxWfacfluci,Wfacfluci = interpolate(xxWfacfluci,Wfacfluci,interpolxxs)



Eener = enerpick['enerEfield']
xxEener = enerpick['fieldxx']
xxEener, Eener  = interpolate(xxEener,Eener,interpolxxs)

Bener = enerpick['enerBfield']
xxBener = enerpick['fieldxx']
xxBener, Bener  = interpolate(xxBener,Bener,interpolxxs)

fluxSx = enerpick['fluxSx']
xxfluxSx = enerpick['fieldxx']
xxfluxSx, fluxSx  = interpolate(xxfluxSx,fluxSx,interpolxxs)

fluxSy = enerpick['fluxSy']
xxfluxSy = enerpick['fieldxx']
xxfluxSy, fluxSy  = interpolate(xxfluxSy,fluxSy,interpolxxs)

fluxSz = enerpick['fluxSz']
xxfluxSz = enerpick['fieldxx']
xxfluxSz, fluxSz  = interpolate(xxfluxSz,fluxSz,interpolxxs)


#enerpick['fluxSx'] = fluxSx integrate these!!! (probably only need Sx tbh) (Will also need to normalize somehow!!!!)
#enerpick['fluxSy'] = fluxSy
#enerpick['fluxSz'] = fluxSz




KEelecs = enerpick['KEelecs']
xxKEelecs = enerpick['xxKEplot']
xxKEelecs, KEelecs = interpolate(xxKEelecs,KEelecs,interpolxxs)

KEions = enerpick['KEions']
xxKEions = enerpick['xxKEplot']
xxKEions, KEions = interpolate(xxKEions,KEions,interpolxxs)

Telecs = enerpick['electemp']
xxTelecs = enerpick['x_in_temp']
xxTelecs, Telecs = interpolate(xxTelecs,Telecs,interpolxxs)

Tions = enerpick['iontemp']
xxTions = enerpick['x_in_temp']
xxTions, Tions = interpolate(xxTions,Tions,interpolxxs)

TotalEelecs = KEelecs+Telecs
TotalEions = KEions+Tions


#normalize KE, T, and Efields to same units-> then plot
#KE and T are in same units 

 



#plot KE and T and Ef separate for debug

#TODO: integrate this somehow to get W!!
#I can compute the total energy of the particles downstream, and subtract Wjdote froom thtat value to get what Wparflux should be- the computed value should be approximmately a scalar difference to that value!
#enerpick['parEfluxion'] = parEfluxion integrate these!! (Will also need to normalize somehow!!!!)
#enerpick['parEfluxelec'] = parEfluxelec

parEfluxelec = enerpick['parEfluxelec'] #particle energy flux electron
xxparEfluxelec = enerpick['xxKEplot']
xxparEfluxelec, parEfluxelec = interpolate(xxparEfluxelec,parEfluxelec,interpolxxs)

dflow_elec_vx = enerpick['dflow_elec_vx']
xxdflow_elec_vx = enerpick['dflow_vx_xx']
xxdflow_elec_vx, dflow_elec_vx = interpolate(xxdflow_elec_vx,dflow_elec_vx,interpolxxs) 

dflow_ion_vx = enerpick['dflow_ion_vx']
xxdflow_ion_vx = enerpick['dflow_vx_xx']
xxdflow_ion_vx, dflow_ion_vx = interpolate(xxdflow_ion_vx,dflow_ion_vx,interpolxxs)

dflow_elec_vy = enerpick['dflow_elec_vy']
xxdflow_elec_vy = enerpick['dflow_vx_xx']
xxdflow_elec_vy, dflow_elec_vy = interpolate(xxdflow_elec_vy,dflow_elec_vy,interpolxxs)

dflow_elec_vz = enerpick['dflow_elec_vz']
xxdflow_elec_vz = enerpick['dflow_vx_xx']
xxdflow_elec_vz, dflow_elec_vz = interpolate(xxdflow_elec_vz,dflow_elec_vz,interpolxxs)

def integrate_parflux(dflowvx,dflowvy,dflowvz,parEflux,xxs):


    xvals = np.flip(xxs)
    Ws = np.zeros(len(xvals)-1)
    parEflux = np.flip(parEflux)
    dflowvx = np.flip(dflowvx)
    xvalsout = xvals[1::]

    for _i in range(1,len(Ws)-1):
        delta_x = xvals[_i+1]-xvals[_i]
        xcoords = xvals[_i+1]

        xvelocity = dflowvx[_i]
        yvelocity = dflowvy[_i]
        zvelocity = dflowvz[_i]
        
        egain_across_box = delta_x/xvelocity * (parEflux[_i-1]-parEflux[_i+1]) * (np.sqrt(xvelocity**2+yvelocity**2+zvelocity**2))
        Ws[_i] = egain_across_box
        if(_i > 0):
            Ws[_i] += Ws[_i-1]

    return np.flip(xvalsout), np.flip(Ws)

xxintegratedval, integratedval = integrate_parflux(dflow_elec_vx,dflow_elec_vy,dflow_elec_vz,parEfluxelec,interpolxxs)
xxintegratedval, integratedval = interpolate(xxintegratedval,integratedval,interpolxxs)

#TODO: remove all the stuff that didnt really work!




#KEions = ddenelec*KEions #ddenelec = ddenion practically
#KEelecs = ddenelec*KEelecs
#Tions = Tions*ddenelec
#Telec = Telecs*ddenelec




#Ei = KEions+Tions
#Ee = KEelecs+Telecs



#normalize enerB to Ei and Ee normalization (I had one small mistake in the main script that is accounted for here!)
#Bener[0] = 1 #for some reason the first value is infinite!
#_c = .45
#mi_me = 625
#n0 = 65.0*2.
#vth_c = 8.0944E-06  #delgam  = 8.0944E-06 
#nt = 1.#related to time step fraction!
#fieldnormfac =   ddenelec*2.*_c**2*nt*4.*np.pi/((ddenelec*2/n0)**2*mi_me**2*vth_c**2)
#print(fieldnormfac)
#print(Bener)
#fieldnormfac = 65.0*2 #We were missing n0 factor in main script!
#Bener = Bener/fieldnormfac
#fluxSx = fluxSx/fieldnormfac
#fluxSy = fluxSy/fieldnormfac
#fluxSz = fluxSz/fieldnormfac


#fluxKEion = dflow_ion_vx * KEions
#fluxKEelec = dflow_elec_vx * KEelecs
#fluxS = np.sqrt(fluxSx**2+fluxSy**2+fluxSz**2)


#dflow_elec_vx

#normalize using upstream ratios

#Wti/Wb = beta_i
betai = .125
Ma = 3.4

#n0 = 131
Tions = Tions/ 131.

_upstreamidx = -10
Bener = Bener*Tions[_upstreamidx]/Bener[_upstreamidx]/betai
#Eener = Eener*Tions[_upstreamidx]/Bener[_upstreamidx]*betai
Telecs = Telecs*Tions[_upstreamidx]/Telecs[_upstreamidx]

KEions = KEions*Tions[_upstreamidx]/KEions[_upstreamidx]*Ma**2/betai

KEelecs = KEelecs*KEions[_upstreamidx]/KEelecs[_upstreamidx]*(1/625.)



Ei = KEions+Tions
Ee = KEelecs+Telecs
Etot = Ei+Ee+Bener
Etot0 = Etot[_upstreamidx]


#print(Tions)
#print(Telecs)

print('rat debug: ',Tions/Bener)

print('Bener',Bener)

#get around load font bug!
plt.figure()
plt.style.use('cb.mplstyle')
plt.plot([0,1],[0,1])
plt.savefig('testdummy.png',format='png',dpi=30)
plt.close()

#make another pub figure -------
plt.style.use('cb.mplstyle')
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, figsize=(20,4.5*4), sharex=True)

_linewidth = 5

#ax1.plot(interpolxxs,Ee/Etot0,label='Ee')
#ax1.plot(interpolxxs,Ei/Etot0,label='Ei')
ax4.plot(interpolxxs,KEelecs/Etot0,color='purple',ls='-.',label=r'$E_{KE,e}/E_{tot}$')
ax2.plot(interpolxxs,KEions/Etot0,color='orange',ls='--',label=r'$E_{KE,i}/E_{tot}$')
ax1.plot(interpolxxs,Telecs/Etot0,color='blue',ls='-.',label=r'$E_{T,e}/E_{tot}$')
ax1.plot(interpolxxs,Tions/Etot0,color='red',ls='--',label=r'$E_{T,i}/E_{tot}$')
ax3.plot(interpolxxs,Bener/Etot0,color='gray',ls='-',label=r'$E_{B}/E_{tot}$')
#ax4.plot(interpolxxs,Eener/Etot0,color='green',ls=':',label=r'$E_{E}/E_{tot}$')
#ax1.plot(x_in,enerCEtot_elec_fac,ls='-.',color='gray',linewidth=_linewidth,label=r'$\int \, \overline{C_{\mathbf{E},e}} \, d^3\mathbf{v}$')
#ax1.plot(x_in,enerjxshockEx,ls='-',color='black',linewidth=_linewidth,label=r'$\int \, \overline{C_{\mathbf{E}^{\prime \prime},sim,e}} \, d^3\mathbf{v}-\overline{j_{x,e,sim,shock}} \overline{E_x^{\prime \prime}}$')
#ax1.plot(_x_in,enerCEtot_elec_facfluclocframe,ls='--',color='purple',linewidth=_linewidth,label=r'$\int \, \widetilde{C_{\mathbf{E},e,down}} \, d^3\mathbf{v}$')
ax1.legend(loc='upper right',fontsize="30")
ax2.legend(loc='lower right',fontsize="30")
ax3.legend(loc='upper right',fontsize="30")
ax4.legend(loc='upper right',fontsize="30")
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

ax4.set_xlabel(r"$x/d_{i}$", fontsize=32)
ax4.set_xlim(5,12)
ax4.set_xticks(np.arange(5,12,1))

plt.subplots_adjust(hspace=0.025)

flnm='figures/enerparvfields_brokendown.png'
plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
plt.close()

plt.style.use('cb.mplstyle')
fig, (ax1) = plt.subplots(1, 1, figsize=(20,4.5*1), sharex=True)

_linewidth = 5

ax1.plot(interpolxxs,Ee,label='Ee')
ax1.plot(interpolxxs,KEelecs,label='KEelecs')
ax1.plot(interpolxxs,Telecs,label='Telecs')
ax1.legend(loc='upper right')
ax1.grid()

ax1.set_xlabel(r"$x/d_{i}$", fontsize=32)
ax1.set_xlim(5,12)
ax1.set_xticks(np.arange(5,12,1))

plt.subplots_adjust(hspace=0.025)

flnm='figures/enerparvfields_debug2.png'
plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
plt.close()

#-----
plt.style.use('cb.mplstyle')
fig, (ax1) = plt.subplots(1, 1, figsize=(20,4.5*1), sharex=True)

_linewidth = 5

Bener[0]=0

Etot = Ee+Ei+Bener
ax1.plot(interpolxxs,Ee/Etot0,ls='-.',color='blue',label=r'$E_e/E_{tot,0}$')
ax1.plot(interpolxxs,Ei/Etot0,ls='--',color='red',label=r'$E_i/E_{tot,0}$')
ax1.plot(interpolxxs,Bener/Etot0,ls='-',color='black',label=r'$E_B/E_{tot,0}$')
ax1.plot(interpolxxs,Etot/Etot0,ls=':',color='gray',label=r'$E_{tot}/E_{tot,0}$')
ax1.legend(loc='upper right')
ax1.grid()

ax1.set_xlabel(r"$x/d_{i}$", fontsize=32)
ax1.set_xlim(5,12)
ax1.set_xticks(np.arange(5,12,1))

plt.subplots_adjust(hspace=0.025)

flnm='figures/enerparvfields.png'
plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
plt.close()


print('debug exitting')
exit()
#-----
plt.style.use('cb.mplstyle')
fig, (ax1) = plt.subplots(1, 1, figsize=(20,4.5*1), sharex=True)

_linewidth = 5

ax1.plot(interpolxxs,fluxS,ls=':',color='gray',label=r'$|\mathbf{S}|$')
ax1.plot(interpolxxs,fluxKEelec,ls='-.',color='blue',label=r'$U_e W_e$')
ax1.plot(interpolxxs,fluxKEion,ls='--',color='red',label=r'$U_i W_i$')
ax1.plot(interpolxxs,fluxS+fluxKEelec+fluxKEion,ls='-',color='black',label='sum')
ax1.legend(loc='upper right')
ax1.grid()

ax1.set_xlabel(r"$x/d_{i}$", fontsize=32)
ax1.set_xlim(5,12)
ax1.set_xticks(np.arange(5,12,1))

plt.subplots_adjust(hspace=0.025)

flnm='figures/enerparvfieldsflux.png'
plt.savefig(flnm,format='png',dpi=300,bbox_inches='tight')
plt.close()





exit()


#fluxSx = fluxSx/fieldnormfac
fluxSy = fluxSy/fieldnormfac
fluxSz = fluxSz/fieldnormfac


fluxKEion = dflow_ion_vx * KEion
fluxKEelec = dflow_elec_vx * KEelec
fluxS = np.sqrt(fluxSx**2+fluxSy**2+fluxSz**2)


#below stuff is kinda wrong?0--------------------------------------------------------------------------------------------------------
#Wiontemp = baseWiontemp + Winti
#print('tot e',TotalEelecs)
#print('int val', integratedval) #Conclusion, flux barely does anything!
#print(We)

#TotalEelecs_diff = TotalEelecs-np.mean([TotalEelecs[-3],TotalEelecs[-2],TotalEelecs[-1]])
#print(TotalEelecs_diff)

#print('----')
#print((TotalEelecs_diff-integratedval)/We)

#normalize and plot
keupelecfromWfac = -np.mean(Wfacavge[55::65]) #kinetic energy upstream in WCEi units (sicne the Bar fields do no work on the interval energy (or basically none) we can use thiss assummption to normalize our stuff!
keupionfromWfac = 625*keupelecfromWfac #they have same injection velocity!!!
TelecupfromWfac = keupelecfromWfac*(Telecs[-1]/KEelecs[-1])
TionupfromWfac = keupelecfromWfac*(Tions[-1]/KEelecs[-1])
#integratedvalfromWfacarr = integratedval * keupelecfromWfac*(integratedval[-1]/KEelecs[-1])

#ultimately, the total KE energy that did not go to the thermal energy, went to the fields (assume no thermal energy goes into fields due to irrevisibility)
#this is not the best assumption because of the wall! we should normalize this differrently!
print("TODO: normalize fields more directly!!!")
energyforgrabs = (keupionfromWfac+keupelecfromWfac)-(np.mean(TelecupfromWfac*Telecs[55::65])+np.mean(TelecupfromWfac*Telecs[55::65]))
energyFieldrat = energyforgrabs/(np.mean(Bener[55::65]+Eener[55::65]))
enerFields = energyFieldrat*(Bener+Eener)

#normalize other things
energyKEelec = keupelecfromWfac+Wfacavge[:]
energyTEDCandinit = TelecupfromWfac*Telecs-Wfacfluce[:] #ennergy in theral enregy of elects due to DC fields initializing, and everything other than AC fields
energyTEAC = Wfacfluce
energyTotion = keupionfromWfac+Wi


xx = interpolxxs
zerosplot = np.zeros((len(xx)))
runtot = np.zeros((len(xx)))

plt.figure(figsize=(6,3))
plt.style.use("cb.mplstyle") #sets style parameters for matplotlib plots

plt.fill_between(xx, runtot, runtot+energyTotion, color='orange', alpha=0.8)
runtot += energyTotion

plt.fill_between(xx, runtot, runtot+enerFields, color='gray', alpha=0.8)
runtot += enerFields

plt.fill_between(xx, runtot, runtot+energyKEelec, color='green', alpha=0.8)
runtot  += energyKEelec

plt.fill_between(xx, runtot, runtot+energyTEDCandinit, color='red', alpha=0.8)
runtot += energyTEDCandinit

plt.fill_between(xx, runtot, runtot+energyTEAC, color='blue', alpha=0.8)

plt.xlabel(r'$x/d_i$')

plt.ylabel(r'$W(x)$')

plt.grid()

plt.savefig('figures/enerprofile.png', format = 'png', dpi=300, bbox_inches='tight')

#normalize temp using internal energy---------
#Winte = Wfacavginte+Wfacavginte
#csWinte = np.cumsum(Winte)
#csWTelecs = np.cumsum(WTelecs-WTelecs[-1]) #due to noise, we compute the ratio using a integral (wathcing the cumsum is just a test)
#print("cumsum test! (should produce a relatively constant value!")
#print(csWTelecs/csWinte)
#elecTempNormConst = csWTelecs[-1]/csWinte[-1]

#Winti = Wfacavginti+Wfacavginti
#csWinti = np.cumsum(Winti)
#csWTions = np.cumsum(WTions-WTions[-1])
#ionTempNormConst = csWTions[-1]/csWinti[-1]

#baseWelectemp = WTelecs[-1]/elecTempNormConst
#baseWiontemp = WTions[-1]/ionTempNormConst

#Welectemp = baseWelectemp + Winte
#Wiontemp = baseWiontemp + Winti

#normalize fields using  total energy
#We = Wfacavge+Wfacfluce
#Wi = Wfacavgi+Wfacfluci
#Wtot = We+Wi
#Wfields = WBfield+WEfield
#Wfields[0] = 0
#csWtot = np.cumsum(Wtot)
#csWfields = np.cumsum(Wfields)

#print('Wtot',Wtot)
#print("cumsumtest2!")
#print(csWfields)



# Plot the curves
#plt.plot(x, y1, label='Curve 1') TODO: do this!!!
#plt.plot(x, y2, label='Curve 2')

# Fill the area between the curves
#plt.fill_between(x, y1, y2, color='gray', alpha=0.3)

#add to values to get next curve!
