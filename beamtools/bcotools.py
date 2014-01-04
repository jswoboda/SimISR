#!/usr/bin/env python
"""
Created on Tue Dec 31 10:58:18 2013

@author: Bodangles
"""
import os
import inspect
def getangles(bcodes):
    ref_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    reffile = os.path.join(ref_path,'bcotable.txt')
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
    #file_name = 'SelectedBeamCodes.txt'
    if type(bcodes) is str:
        file_name = bcodes
        f = open(file_name)
        bcolines = f.readlines()
        f.close()
    
        bcolist = [int(float(x.rstrip())) for x in bcolines]
    else:
        bcolist = bcodes
    angles = [bco_dict[x] for x in set(bcolist)]
    return angles