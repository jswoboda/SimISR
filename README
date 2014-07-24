RadarDataSim
by John Swoboda
This repository will create synthetic incoherenent scatter radar by creating filters from
ISR spectrums and applying them to CWGN. It is split up into three classes.
IonoContainer - A class that holds information on the ionosphere.
RadarData - A class that holds and operates on the radar data.
FitterMethods - A class that holds the fitter for the data.

The work flow for the code starts with creating an IonoContainer instance with 
the desired conditions. From there the radarData class is created which will hold 
the raw IQ data. The acf estimates can be created by using the method processdata
which will output acf estimates given the desired integration time and starts of 
integration which will allow for overlap of integration periods. Lastly the 
fitter class is created. To get fitted data use fitdata.