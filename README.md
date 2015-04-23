##RadarDataSim
by John Swoboda 
![alt text](https://raw.github.com/jswoboda/RadarDataSim/master/Images/logofig.png "RadarDataSim")

#Overview
This Python module can be used create synthetic incoherenent scatter radar. It does this by creating filters from ISR spectrums and applying them to CWGN. This is depicted below in the following flow diagram.

![Diagram](https://raw.github.com/jswoboda/RadarDataSim/master/Images/diagrampart.png)


From there the data can be processed like ISR data. The following flow diagram represents the data flow for long pulse experiment.

![Diagram2](https://raw.github.com/jswoboda/RadarDataSim/master/Images/datastackchain.png)

#Installation
To install first clone repository:

	$ git clone https://github.com/jswoboda/RadarDataSim.git
Before going further in the install the user also needs to download and install the module [ISRSpectrum](https://github.com/jswoboda/ISRSpectrum) from the repository found in the link.

After the ISRSpectrum module is installed move to the const directory. This is a directory that was turned into a submodule because it was being used by multiple repositories.

	$ cd RadarDataSim/RadarDataSim/const
	$ git pull origin

Then move to the main directory and run the Python setup script, which should be run in develop mode.

	$ cd ../..
	$ python setup.py develop

###Install Test
To determine if everything has been properly istalled it is suggested that user runs the following Python files to create some test data. 


	$ cd RadarDataSim/RadarDataSim
	$ python basictest.py
	
If h5 files for each stage have been created then it should be properly installed. 

#Software Archetecture
The module is split up into three classes. 

IonoContainer - A container class that holds information on the ionosphere or auto correlation functions (ACFs)/spectrums.

RadarDataFile - A class that holds and operates on the radar data to create estimates of the autocorrelation fuction. The class takes files of  

FitterMethodsGen - A class that applies the fitter to the data.

A high level mathematical flow of the software can be seen in the figure below with the operations of each class labelled. 

![Diagram2](https://raw.github.com/jswoboda/RadarDataSim/master/Images/softwareflowandmath.png)

Where \\( \Theta\\) is the plasma parameters \\( g(\Theta)\\) is a function that turns the plasma parameters to ISR spectrums, \\( \mathbf{r}\\) is the spectrums/ACFs for each point of time and space, \\( \mathbf{Lr}\\) is the radar's operator on the spectrums/ACFs, \\( \rho\\) is the measured ACFs from the radar and lastly \\( \hat{\Theta}\\) is the estimates of plasma parameters from \\( \rho\\).

#Workflow

To run the simulation it is suggested that the user create a set of scripts outside of this directroy. This is so the user can create different types of experiments with different sensor parameters, different plasma parameters etc..

First the user needs to decide on a beam pattern and what sensor they want to use. The user can use gui in beamtools/pickbeams.py to select a beam pattern for either RISR of PFISR which a screen shot is shown below.
![Diagram2](https://raw.github.com/jswoboda/RadarDataSim/master/Images/pickbeams.png)
With this set take the beam numbers recieve from this file and create two sets of dictionaries using the pattern found in makeConfigFiles.py. It is suggested that the user creates a .pickle file instead of an ini in the current iteration of the code.

From there create a field of ionospheric parameters using the IonoContainer class. Create the spectrums using ISRSpectrum and then output a set of h5 files. At this point call the RadarDataFile class to create create radar data and lags. The RadarDataFile class can take a set of files of radar data and then form lag estimates using that. Once the lag estimates are formed they can be saved to an instance of the IonoContainer class. After which the Fitterionoconainer class can be called the data can be fit using the desired fitting methodolgy of the user. A final instance of the IonoContainer class can then be saved out and then send to other programs for processing.