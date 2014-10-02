#!/usr/bin/env python
"""
Created on Tue Dec 31 09:24:09 2013

@author: Bodangles
"""

import numpy as np


reffile = 'bcotable.txt'
ref_f = open(reffile)
all_ref = ref_f.readlines()
ref_f.close()

# make a beamcode to angle dictionary
bco_dict = dict()
for slin in all_ref:
    split_str = slin.split()
    bco_num = int(split_str[0])
    bco_dict[bco_num] = (float(split_str[1]),float(split_str[2]))

# Read in file 
file_name = 'SelectedBeamCodes.txt'
f = open(file_name)
bcolines = f.readlines()
f.close()

bcolist = [int(float(x.rstrip())) for x in bcolines]
angles = [bco_dict[x] for x in bcolist]