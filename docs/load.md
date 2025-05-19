# Loading Simulations[example notebooks](../notebooks)

Simulations from all codes are loaded into common data structures that allow for the use of common routines. These data structures are python dictionaries for the fields, fluid quantities (density, velocity from moment of distribution, etc.), and particle data/distribution function data. As this code handles the analysis of a a full PIC code, hybrid PIC code, and DG (FEM) code, one should be aware of the differences in the data that is produced by the simulation. To most swiftly summarize, all codes will produce EM fields. PIC codes will evolve a list of `point' particles using the Lorentz force (and use this list of particles to generate currents that evolve the EM fields). These point partilces can be binned to form distribution functions. DG/ FEM codes will direclty solve the Vlasov Maxwell system of equations, directly producing the distribution function. Hybrid codes will simulation some species (typically ions and heavier) as a list of particles, and the rest of the species (almost always just electrons) as a `fluid' that response instantly to maintain quasineutrality in the system, which is a fair approximation when the mass of electrons is much much smaller than the mass of the ions.

Thus, there are four primary data structures that exist for doing our analysis. Not all data structures are used for all the different simulations. Please see the load library for the respective simulation.

The specific load functions are shown in the [example notebooks](../notebooks)

# Data Structures

The simulation data is loaded into python dictionaries, which contains entries for the different data within it that can be accessed by calling dictname['keyname'], where keyname is the name of the data you want to have.

2D and 3D arrays are ordered as [yy,xx] and [zz,yy,xx] respectively.

## dfields

dfields is the dictionary that contains the electromagnetic field data.

There are keys for each field direction which are either 3D, 2D, or 1D

	ex
	ey
	ez
	bx
	by
	bz

For each cartesian dimension in the simulation, there also exists 1D arrays for the grid coordinates

	ex_xx
	ey_xx
	...
	by_zz
	bz_zz

## dpar

dpar is the dictionary that contains the particle data. dpar contains multiple parallel 1D arrays containing the cartesian coordinates and velocity/momentum components.

For dHybridR, the keys are
	x1
	x2
	x3
	p1
	p2
	p3
for partilce position and momentum.

For Tristan, the keys are
	xs
	ys
	zs
	us
	vs
	ws
where 's' is either 'i' or 'e' for ions/ electrons respectively and us vs ws are particle velocity.


There exists functions that `rename' the tristan keys to match the dhybridr keynames for compatibility with the routines in this library. These functions simply rename the keys and do not do any normalization conversion.

## dden

dden is like dfields, but contains density data.

## dflow

dflow is like dfields but contains ion/electron velocity moments.

## dcurr

dcurr is like dfields but contains net current in system.

# dwavemodes

dwavemodes contains information about the system in wavenumber space at selected wavemode. Please see appropriate examples in [example notebooks](../notebooks). 







