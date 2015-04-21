#!/usr/bin/env python
"""
Created on Fri Sep 19 10:26:47 2014

@author: Bodangles
"""
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from .beamfuncs import BeamSelector

chosenbeams = np.loadtxt('SelectedBeamCodes.txt')
beamman = np.loadtxt('bcotable.txt')

allbeam = BeamSelector(beamman)
allbeamorig = allbeam
allbeam.shiftbeams(azoff=15,eloff=-16)
(azvec,elvec) = allbeam.azelvecs()
allbeamlist = list(allbeam.beamnumdict.keys())
azlog =np.where( azvec==0.0)[0]
chosenbeams = np.array(allbeamlist)[azlog]
#allbeam.switchzenith()
allbeam.plotbeams(chosenbeams,True,'beampic.png',"Possible Beams from Boresite")
allbeam.printbeamangles('outangles.txt',chosenbeams)
plt.show(False)