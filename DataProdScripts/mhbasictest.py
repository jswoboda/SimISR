#!python

"""
"""
import numpy as np

from SimISR import make_test_ex
def main(savedir="~/DATA/SimISR/MHsimple"):

    z = np.arange(90, 700,5,dtype = float)
    coords = np.column_stack([np.zeros_like(z),np.zeros_like(z),z])
    iono_t = make_test_ex(testv=False,testtemp=True,coords=coords)
    


if __name__ == '__main__':
    main()