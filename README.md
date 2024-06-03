<3 align="center">FPC Analysis</h3>

## Table of Contents

The following are meant to be read in order:

- [About](#About)
- [Setup](#Setup)
- [Start Analyzing](#Start)
- [Loading Simulations and Data Structures](docs/load.md)
- [Field Particle Correlation](docs/FPC.md) 
- [Wavelet Transform](docs/WLT.md)
- [Instability Isolation Method](docs/IIM.md)
- [Premade Plot Functions](docs/plots.md)
- [Premade Scripts](docs/scripts)
- [Regression and Unit Testing](docs/regression_unit_test.md)


FAC.md			IIM.md			load.md			plots.md		scripts.md
FPC.md			WLT.md			normalization.md	regression_unit_test.md	shockvel.md

## About <a name = "about"></a>

- This is the documentation for Collin Brown's FPC Library and Analysis scripts. Here, we present the code needed to perform FPC analysis on dHybridR, Tristan, and Gkeyll data. While examples are provided that require only trivial alteration to perform the core of most FPC and IIM related analysis, this repo's primary goal is to be a library that has functions that one can build upon to perform their own analysis.
- Major contributions/ Special Thanks: Greg Howes, Jimmy Juno, Colby Haggerty, Andrew McCubbin, 


## Setup <a name = "setup"></a>

Requirements: python 3.8

1. Clone this repo
```
git clone https://github.com/collinrbr/FPCAnalysis.git
```

2. Run setup script to install libraries with correct version. It may be possible for newer versions of the the python libraries to be used, but compatability is not tested. 
```
./setup.py
```
Note, if one wishes to use additional libraries, add them to the requirements.txt file, remove the FPCAnalysis environment folder, and rerun setup.py.

3. Activate the environment
** ONE MUST NAVIGATE TO THIS DIRECTORY AND REACTIVE THE ENVIRONMENT EVERYTIME A NEW TERMINAL IS OPENED ** 
Linux/Mac:
```
source FPCAnalysis/bin/activate
```

Windows
```
FPCAnalysis\\Scripts\\activate
```
** ONE MUST NAVIGATE TO THIS DIRECTORY AND REACTIVE THE ENVIRONMENT EVERYTIME A NEW TERMINAL IS OPENED **

Now, you can select an example notebook/ script and start running!

## Example Data

### Gkeyll
Data for Gkeyll can be generated using the example inputs of published simulations at [https://github.com/ammarhakim/gkyl-paper-inp](https://github.com/ammarhakim/gkyl-paper-inp). One simulation of note the is the perpendicular shock simulation at [https://github.com/ammarhakim/gkyl-paper-inp/tree/master/2021_JPP_FPC_Perp_Shock](https://github.com/ammarhakim/gkyl-paper-inp/tree/master/2021_JPP_FPC_Perp_Shock).


### Tristan
TODO: link data from upcoming paper on nonadiabatic electron heating. Provide example inputs too!

### dHybridR
Sample dHybridR data can be found at [https://doi.org/10.5281/zenodo.7901521](https://doi.org/10.5281/zenodo.7901521). Otherwise, as dHybridR is not currently open source, please contact the authors for an executable or to collaborate on a simulation.

## Useful Science Links <a name = "art"></a>

- [Gkeyll](https://gkeyll.readthedocs.io/en/latest/)
- [dHybridR](https://arxiv.org/abs/1909.05255)
- [Tristan (v2)](https://princetonuniversity.github.io/tristan-v2/)


## Get Started with Analysis <a name = "start"></a>

Analysis begins by loading the data. In the first chapter found [here](docs/load.md), we discuss how data is loaded into a common data struture and how to use this common structure...
