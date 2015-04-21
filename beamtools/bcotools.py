#!/usr/bin/env python
"""
Created on Tue Dec 31 10:58:18 2013

@author: Bodangles
"""
import os
import inspect
from six import string_types

def getangles(bcodes,radar='risr'):
    ref_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    if radar.lower() == 'risr':
        reffile = os.path.join(ref_path,'RISRNbeammap.txt')
    elif radar.lower() == 'pfisr':
        reffile = os.path.join(ref_path,'PFISRbeammap.txt')

    with open(reffile,'r') as ref_f:
        all_ref = ref_f.readlines()

    # make a beamcode to angle dictionary
    bco_dict = dict()
    for slin in all_ref:
        split_str = slin.split()
        bco_num = int(split_str[0])
        bco_dict[bco_num] = (float(split_str[1]),float(split_str[2]))

    # Read in file
    #file_name = 'SelectedBeamCodes.txt'
    if isinstance(bcodes,string_types):
        file_name = bcodes
        with open(file_name,'r') as f:
            bcolines = f.readlines()

        bcolist = [int(float(x.rstrip())) for x in bcolines]
    else:
        bcolist = bcodes
    angles = [bco_dict[x] for x in set(bcolist)]
    return angles