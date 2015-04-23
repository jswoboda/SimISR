##RadarDataSim
by John Swoboda
![alt text](https://raw.github.com/jswoboda/RadarDataSim/specchange10152014/logofig.png "RadarDataSim")

This repository will create synthetic incoherenent scatter radar by creating filters from
ISR spectrums and applying them to CWGN. It is split up into three classes.

IonoContainer - A container class that holds information on the ionosphere or lags/spectrums.

RadarDataFile - A class that holds and operates on the radar data.

FitterMethods - A class that holds the fitter for the data.

The work flow for the code starts with creating an IonoContainer instance with 
the desired conditions. From there the radarData class is created which will hold 
the raw IQ data. The acf estimates can be created by using the method processdata
which will output acf estimates given the desired integration time and starts of 
integration which will allow for overlap of integration periods. Lastly the 
fitter class is created. To get fitted data use fitdata.