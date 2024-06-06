# SimISR

by John Swoboda
![ionosphere slice plots](https://raw.github.com/jswoboda/SimISR/master/Images/logofig3.png "SimISR")

## Overview

This Python module can be used create synthetic incoherenent scatter radar. It does this by creating filters from ISR spectrums and applying them to CWGN. This is depicted below in the following flow diagram.

![SimISR flow Diagram](https://raw.github.com/jswoboda/SimISR/master/Images/diagrampart.png)


From there the data can be processed like ISR data. The following flow diagram represents the data flow for long pulse experiment.

![SIMisr block diagram](https://raw.github.com/jswoboda/SimISR/master/Images/datastackchain.png)

## Suggestions

It is highly suggested that the [Anaconda](https://www.continuum.io/downloads) platform be used as the package manager. All of the development and testing has been done using this.
Assuming the user has installed Anaconda a [set up bash script](https://github.com/jswoboda/AnacondaEnvUtilities), which can be used in Linux or Mac environments is avalible.

The user can also take advantage of two different APIs to plot results using the SimISR. The first is in Python and is called [GeoDataPython](https://github.com/jswoboda/GeoDataPython). A MATLAB version of this API is also avalible called [GeoDataMATLAB](https://github.com/jswoboda/GeoDataMATLAB). Both APIs can read in the structured files from SimISR.

## Installation

All prereqs are handled automatically on install:

```sh
	git clone https://github.com/jswoboda/SimISR.git
	cd SimISR
    pip install -e .
```

### Test

To determine if everything has been properly installed it is suggested that user runs the following Python files to create some test data.

```sh
	$ cd SimISR/Test
	$ python testsimisr.py
```

If h5 files for each stage have been created then it should be properly installed.

TODO: 

* `barkertest.ini` is missing, so `Test/barkertest.py` isn't usable.
* `statstest.py` also appears to be missing a file
* `testcomp.py`, `testdishmode.py`, `testisrmatrix.py`,`testsimisr.py` missing input file

## Software Archetecture

The module is split up into three classes.

IonoContainer - A container class that holds information on the ionosphere or auto correlation functions (ACFs)/spectrums.

RadarDataFile - A class that holds and operates on the radar data to create estimates of the autocorrelation fuction. The class takes files of

FitterMethodsGen - A class that applies the fitter to the data.

A high level mathematical flow of the software can be seen in the figure below with the operations of each class labelled.

![Diagram2](https://raw.github.com/jswoboda/SimISR/master/Images/softwareflowandmath.png)

Where \\( \Theta\\) is the plasma parameters \\( g(\Theta)\\) is a function that turns the plasma parameters to ISR spectrums, \\( \mathbf{r}\\) is the spectrums/ACFs for each point of time and space, \\( \mathbf{Lr}\\) is the radar's operator on the spectrums/ACFs, \\( \rho\\) is the measured ACFs from the radar and lastly \\( \hat{\Theta}\\) is the estimates of plasma parameters from \\( \rho\\).

## Workflow

To run the simulation it is suggested that the user create a set of scripts outside of this directroy. This is so the user can create different types of experiments with different sensor parameters, different plasma parameters etc..

The user needs to create a configureation file. These ini files can be made a number of ways. The easiest is by using `beamtools/setupgui.py`.
The user can fill in the parameters neccesary to run the simulation.
They can also pick the beam pattern they desire.

![Diagram2](https://raw.github.com/jswoboda/SimISR/master/Images/seupgui.png)

The user could also create the neccesary diectionaries and use makeConfigFiles.py. They can also save out the configuration file as pickle file but the ini files are augmentedable through a text editor and thus easier to use.

Once the configuration file has been created the user can start setting up the simulation. The step is to create a field of ionospheric parameters using the IonoContainer class. Within this class there are methods to create the spectrums using ISRSpectrum and then output a set of h5 files. Once the spectrums are created  the RadarDataFile module to create create radar data and lags. The RadarDataFile class can take a set of files of radar data and then form lag estimates using that. Once the lag estimates are formed they can be saved to an instance of the IonoContainer class. After which the Fitterionoconainer class can be called the data can be fit using the desired fitting methodolgy of the user. A final instance of the IonoContainer class will be saved out.

The submodule RadarDataSim/runsim.py can run the whole simulation after the configuration and plasma parameter files have been created. It can be run either through the command line or its main function can be embedded in other files. It is highly suggested that this submodule be used as it creates all of the neccesary folders to save the data properly.
