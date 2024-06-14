# FPC

In this markdown file, we will provide example documentation on what the Field-Particle Correlation (FPC) is, and how compute the FPC using this library.

## What is the FPC?

The FPC computes the energization rate of particles within specified region in phase-space by the electric field component $C_{E_i}(\mathbf{v};\mathbf{x})$. Integrated over veloctiy-space, the FPC metric is equivalent to one one component of $\mathbf{j} \cdot \mathbf{E}$ from Poynting's theorem, $j_i E_i = \int C_{E_i}(\mathbf{v};\mathbf{x} d\mathbf{v}$. Looking into phase-space to analyze energy transfer is advantageous, as the energization rate of a particular particle can depend stiffly not only on it's location in the system, but also it's velocity. Thus, this metric creates unique signatures in velocity-space that are used to identify heating and energization mechanisms.

As the name implies, the FPC is a correlation between the distribution function and the electric fields. This correlation can be taken over space and or time. Often in plasmas, there is oscillatory energy transfer than yields not net transfer upon integration over a periodic domain in space or time. As the net-effects of energy transfer are more relevant to most analysis, correlating over a specified space/time region helps simplify analysis and understanding.

This library is most suitable for taking correlations over space, but one can take the sum of $C_{E_i}$ over a set of frames to compute the space and time correlation.

## How to run using library

Please see the example notebooks on how to load and compute the FPC using simulation data.

## How to run with scripts

One can compute the correlation as a function of x along thin slices specified by dx by filling out an analysisinput.txt file (see parent directory for example) and running python scripts/generateFPC***.py (be sure to activate the environment first).

### Netcdf4 Output

Many of the FPC scripts produce *.nc files (netcdf4 files) that  contain the distribution function data, and data for the FPCs computed using thin slices along x.

You can use ncdump (if installed) to see the header. 'ncdump -h filename.nc'.

Otherwise, see the FPCAnalysis/data_netcdf4.py file for examples on how to load the *.nc file.

Note that the key 'MachAlfven' is the upstream injection velocity in the simulation rest frame. It is common for this to be related to the name of the simulation. For example, the shock studied in Brown et al 2023 is called M06th45 because we inject particles with a upstream alfven/ species thermal velocity (they are equal in this sim as beta_s = 1) of 6 and have a shock normal angle of 45 degrees (or more specifically we have an external magnetic field that is initialized to be 45 degrees from the direction of the injection velocity).

## Types of correlation

By correlating with different fields or in different coordinates, one can obtain different types of FPC

### Instability FPC

The instability correlation is correlation between the instability fields (defined as the total fields minus some 'steady-state' profile of the fields). It shows the energization by the instabilites.

### Field Aligned Coordinate

The correlation can be computed in Field-Aligned coordinates. Note, there are different definitions of field aligned.




