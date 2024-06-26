<h1 align="center">FPC Analysis</h1>

<div align="center">
  <img src="https://github.com/collinrbr/FPCAnalysis/blob/tristan/FPC%20Analysis%20Logo.svg" width="25%" height="25%">
</div>

## Table of Contents <a name = "toc"></a>

- [About](#About)
- [Setup](#Setup)
- [Example Data](#exdata)
- [Running Simulations](docs/inputs.md)
- [Start Analyzing](#start)
- [Loading Simulations and Data Structures](docs/load.md)
- [Field Particle Correlation](docs/FPC.md) 
- [Wavelet Transform](docs/WLT.md)
- [Instability Isolation Method](docs/IIM.md)
- [Example Notebooks](notebooks)
- [Premade Scripts](scripts)
- [Premade Plot Functions](docs/plots.md)
- [Regression and Unit Testing](docs/regression_unit_test.md)
- [Development](docs/devnotes.md)
- [Useful links](#scilinks)

## About <a name = "about"></a>

- This is the documentation for Collin Brown's FPC Analysis Environment, Library and Analysis scripts. Here, we present the code needed to perform FPC analysis on dHybridR, Tristan, and Gkeyll data. While examples are provided that require only trivial alteration to perform the core of most FPC and IIM related analysis, this repo's primary goal is to be a python analysis environment/ library that has functions that one can build upon to perform their own analysis.
- Major contributions/ Special Thanks: Greg Howes, Jimmy Juno, Colby Haggerty, Andrew McCubbin, Alberto Felix, Rui Huang, Emily Lichko


## Setup <a name = "setup"></a>

Requirements: python 3.11; pip; conda

Here, we walk through how to make an environment that contains this library. Installing directly to your main environment is not recommended.

1. Clone this repo
```
git clone https://github.com/collinrbr/FPCAnalysis.git
```

2. Run setup script to install libraries with correct version. It may be possible for newer versions of the the python libraries to be used, but compatability is not tested. 
```
./install.py
```
This will create a folder 'FPCAnalysisenv' that contains the python environment for doing FPCAnalysis with these scripts.

Note, if one wishes to use additional libraries, add them to the requirements.txt file, remove the FPCAnalysis environment folder, and rerun setup.py or install them using pip/conda while the environment is activated. This can also be used to update libraries although compatability is not guaranteed.


3. Use the environment

This can be done in one of several ways. The most general way is to activate the enviroment in your terminal.

** ONE MUST NAVIGATE TO THIS DIRECTORY AND REACTIVE THE ENVIRONMENT EVERYTIME A NEW TERMINAL IS OPENED TO USE THIS ENVIRONMENT** 
Linux/Mac:
```
conda activate /full/path/to/FPCAnalysisenv
```
** ONE MUST NAVIGATE TO THIS DIRECTORY AND REACTIVE THE ENVIRONMENT EVERYTIME A NEW TERMINAL IS OPENED TO USE THIS ENVIRONMENT**

One can also add the following to the top of anyscripts that is written for this environment, in which case, it is no longer necessary to activate the environment when running any script with the following added to it:
```
#!FPCAnalysis/bin/python
```

Now, you can select an example notebook/ script and start running!

To uninstall remove the FPCAnalysisenv folder created by the install script. You may need to reinstall if you move the environment folder or the library that it is linked too (e.g. the FPCAnalysis lib folder).

To add additional libraries to this environment, activate the environment and then either `pip install' them or `conda install' them.

## Example Data <a name = "exdata"></a>

### Gkeyll
Data for Gkeyll can be generated using the example inputs of published simulations at [https://github.com/ammarhakim/gkyl-paper-inp](https://github.com/ammarhakim/gkyl-paper-inp). One simulation of note the is the perpendicular shock simulation at [https://github.com/ammarhakim/gkyl-paper-inp/tree/master/2021_JPP_FPC_Perp_Shock](https://github.com/ammarhakim/gkyl-paper-inp/tree/master/2021_JPP_FPC_Perp_Shock).


### Tristan
TODO: link data from upcoming paper on nonadiabatic electron heating. Provide example inputs too!

### dHybridR
Sample dHybridR data can be found at [https://doi.org/10.5281/zenodo.7901521](https://doi.org/10.5281/zenodo.7901521). Otherwise, as dHybridR is not currently open source, please contact the authors for an executable or to collaborate on a simulation.


## Get Started with Analysis <a name = "start"></a>

One can start using this analysis environment with the example data above or by running their own simulation (see [Running Simulations](docs/inputs.md)).

Analysis begins by loading the data. In the first chapter found [here](docs/load.md), we discuss how data is loaded into a common data struture and how to use this common structure...

There are many key features of this analysis environment:

- Most notable is the ability to compute the [Field Particle Correlation](docs/FPC.md), which shows energy transfer between waves and particles.
- We can also compute the wavelet transform [Wavelet Transform](docs/WLT.md), allowing us to measure the complex coefficients associated with wavenumber and position in a simulation.
- Finally, we can compute the [Instability Isolation Method](docs/IIM.md), which relates the measurements of the complex coefficients at particular wavenumber and position for each field component to frequency using faradays law assuming plane-wave solutions. This allows for one to meausure (k_x, k_y, k_z, omega) using a single frame of data at a local specified position.

We walkthrough how to use many of the functions provided in this library in the [example notebooks](notebooks) folder. But, there are also many [premade scripts](docs/scripts) that you can use.
There are many [Premade Plot Functions](docs/plots.md), but it is encouraged to implement your own plots too!

It may also be helpful to look at the [regression and unit testing](docs/regression_unit_test.md). And also, if you plan on developing this, please see [Development](docs/devnotes.md).

## Useful Science Links <a name = "scilinks"></a>

- [Gkeyll](https://gkeyll.readthedocs.io/en/latest/)
- [dHybridR](https://arxiv.org/abs/1909.05255)
- [Tristan (v1)](https://ntoles.github.io/tristan-mp-pitp/)
- [Tristan (v2)](https://princetonuniversity.github.io/tristan-v2/)
