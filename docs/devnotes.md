# Development Notes

Here, we leave different notes that are useful to future developers of this codebase.


## Making changes to the code

The code is installed in editable mode, so any changes that are made to the FPCAnalysis lib folder will take effect in the environment. 

Be sure to restart any jupyter notebook or use importlib to reload the specific library file! For example call
```
import importlib
importlib.reload(FPCAnalysis.ddhr)
```
to reload the data_dhybridr.py file!

Be sure to not move your FPCAnalysis folder after installing as it may confuse the conda environment!

## Unit/ Regression Tests

There exists unit tests and regression tests. These tests ensure that the functionality of the code does not change or break over time as changes are made. After major changes are made, please run the unit tests from the main directory e.g.

```
python tests/unittest.py
```

Don't forget to have the environment activated!

The example notebooks are also meant to serve as regression tests.

## Backwards compatability
If there is ever any compatability issues, the full exact environment can be found in docs/environment_full.txt- This is the environment that was used while debugging the first major release!

If anyone attempts to upgrade the enviornment, please be aware that the main concern is the deprecation of numpy operations, but of course, there may be more concerns.

Note that we use specific commit of postgkyl and that it needs to be updated in install.py to run

## Readability of Code

Unlike traditional programming, scientific code is unique as in theory most scientific scripts need only be ran once (per input data and chosen parameters) assuming no error in the code (and that there is no randomness in the code that can impact results). For example, the results of a simulation will not change simply by running it again. Because of this, it is often favorable to value the readability of a code over the speed or 'slickness' of a code. Making codes run faster can often be a net loss in terms of time, so a measured amount of development should be committed to improving the speed of the code. Making a code runs in 5 minutes or 5 seconds will have the same results at the end of the day, only one choice will also come with a headache and an opportunity cost.

Given that the audience of this code are likely scientist first, we should value usability and readability of the code over other programming principals when the two are in conflict. This is a hard balance with a lot of subjectivity, so I have likely failed this at many spots! 

However, above all, the code should be understanable to the general science community to maximize reproducabilty and extension to other projects!
