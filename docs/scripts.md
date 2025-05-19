# Running Scripts

*As always, please be sure to have installed and activated the environment to use FPCAnalysis.*

Scripts are meant to be ran in the main directory of FPCAnalysis. E.g.
```
python scripts/*.py input1 input2 ....
```

If you don't know the input of a particular script, just call
```
python scripts/*.py
```
And it will tell you what the inputs are.

Scripts will typically generate larger data files and/or process them. Below we describe the scripts.

Scripts were designed to work with macOS/linux and there may be some calls by the os.system(*cmd*) that may still need to be updated for other operating systems.

## Running FPC sweeps

There are several scripts for generating sweeps of FPC data of dHybridR, Gkeyll, and Tristan shock simulations. They have names of the form generateFPC*.py.

There are both serial and parallel ways of running this code.

### Serial FPC 

1. Create a copy of the analysis.txt file, choosing any name. Fill out the parameters as needed. The description of each parameter is as follows:
```
    path- path to simulation data (specifically the outermost folder)
    resultsdir- directory to save any newly created files to
    vmax- bounds to compute FPC in velocity space (same in all directions)
    dv- bin size of velocity space signature
    numframe- frame of simulation to use
    dx- width in the transverse region to slice the simulation into when computing the FPC along each slice
    xlim- bounds to sweep over in x
    ylim- transverse size of the integration box in the y direction. this does not change as we sweep along x, computing the FPC for each x slice
    zlim- transverse size of the integration box in the z direction
```

2. Run generateFPC*.py to serially compute the fpc for the total fields and the fluctuating fields. This will create a *.nc folder that contains the FPC data.

```
"python scripts/generateFPC*.py analysisinputflnm.txt" 
```
is the command to compute the total correlation.

```
python scripts/generateFPC*.py analysisinputflnm.txt F F 1 '/' T
```
is the command to compute tildeC.
     
3. (Optional but recommended!) Use make.py to make 9 panel plots for every x slice
```
"python scripts/plotnetcdf9pan.py ​***.nc"
```

### Parallel FPC Sweeps

1. Make a folder to store all presliced data. For example, 'mkdir presliceddata01'

2. Preslice all the simulations into thin domains in x. The size of the domain is specified by dx in each analysis.txt file and the range used is specified by xlim. Preslice the data into some folder (e.g. presliceddata01) using the preslicer script.

2. Run generateFPC*.py to serially compute the fpc for the total fields and the fluctuating fields. This will create a *.nc folder that contains the FPC data.

For example
```
python scripts/generateFPC*.py analysisinputflnm.txt F F ncores path/to/presliceddata F
```
is the command to compute the total correlation with ncores working in parallel.



