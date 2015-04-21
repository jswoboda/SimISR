#!/usr/bin/env python
"""
Created on Fri Mar 27 10:37:11 2015

@author: John Swoboda
"""

import tables
import numpy as np
from os.path import expanduser

risrfile = expanduser('Documents/ISRData/20140224.002/d0079139.dt0.h5')
with tables.open_file(risrfile) as h5file:
    risrbeams = h5file.root.Setup.BeamcodeMap.read()

np.savetxt('RISRNbeammap.txt',risrbeams,fmt='%d %.2f %.2f %.6e')

PFISRfile = expanduser('Documents/ISRData/20111215.022/d0288127.dt0.h5')
with tables.open_file(PFISRfile) as h5file:
    pfisrbeams = h5file.root.Setup.BeamcodeMap.read()

np.savetxt('PRISRbeammap.txt',pfisrbeams,fmt='%d %.2f %.2f %.6e')