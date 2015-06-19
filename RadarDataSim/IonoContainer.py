#!/usr/bin/env python
"""
Holds the IonoContainer class that contains the ionospheric parameters.
@author: John Swoboda
"""
import os
import inspect
import pdb
import posixpath
import copy

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.interpolate
import tables
# From my
from ISRSpectrum.ISRSpectrum import ISRSpectrum
from utilFunctions import Chapmanfunc, TempProfile

class IonoContainer(object):
    """Holds the coordinates and parameters to create the ISR data.  Also will
    make the spectrums for each point."""
    #%% Init function
    def __init__(self,coordlist,paramlist,times = None,sensor_loc = sp.zeros(3),ver =0,coordvecs =
            None,paramnames=None,species=None,velocity=None):
        """ This constructor function will use create an instance of the IonoContainer class
        using either cartisian or spherical coordinates depending on which ever the user prefers.
        Inputs:
        coordlist - Nx3 Numpy array where N is the number of coordinates.
        paramlist - NxTxP Numpy array where T is the number of times and P is the number of parameters
                    alternatively it could be NxP if there is only one time instance.
        times - A T length numpy array where T is the number of times.  This is
                optional input, if not given then its just a numpy array of 0-T
        sensor_loc - A numpy array of length 3 that gives the sensor location.
                    The default value is [0,0,0] in cartisian space.
        ver - (Optional) If 0 the coordlist is in Cartisian coordinates if 1 then
        coordlist is a spherical coordinates.
        coordvecs - (Optional) A dictionary that holds the individual coordinate vectors.
        if sphereical coordinates keys are 'r','theta','phi' if cartisian 'x','y','z'.
        paramnames - This is a list or number numpy array of numbers for each parameter in the
        """
        r2d = 180.0/np.pi
        d2r = np.pi/180.0
        # Set up the size for the time vector if its not given.
        Ndims = paramlist.ndim
        psizetup = paramlist.shape
        if times is None:
            if Ndims==3:
                times = np.arange(psizetup[1])
            else:
                times = np.arange(1)
        if Ndims==2:
            paramlist = paramlist[:,np.newaxis,:]

        # Assume that the

        if ver==0:

            X_vec = coordlist[:,0]
            Y_vec = coordlist[:,1]
            Z_vec = coordlist[:,2]

            R_vec = sp.sqrt(X_vec**2+Y_vec**2+Z_vec**2)
            Az_vec = sp.arctan2(Y_vec,X_vec)*r2d
            El_vec = sp.arcsin(Z_vec/R_vec)*r2d

            self.Cart_Coords = coordlist
            self.Sphere_Coords = sp.array([R_vec,Az_vec,El_vec]).transpose()
            if coordvecs is not None:
                if set(coordvecs)!={'x','y','z'}:
                    raise NameError("Keys for coordvecs need to be 'x','y','z' ")
            else:
                coordvecs = ['x','y','z']

        elif ver==1:
            R_vec = coordlist[:,0]
            Az_vec = coordlist[:,1]
            El_vec = coordlist[:,2]

            X_vec = R_vec*np.cos(Az_vec*d2r)*np.cos(El_vec*d2r)
            Y_vec = R_vec*np.sin(Az_vec*d2r)*np.cos(El_vec*d2r)
            Z_vec = R_vec*np.sin(El_vec*d2r)

            self.Cart_Coords = sp.array([X_vec,Y_vec,Z_vec]).transpose()
            self.Sphere_Coords = coordlist
            if coordvecs is not None:
                if set(coordvecs)!={'r','theta','phi'}:
                    raise NameError("Keys for coordvecs need to be 'r','theta','phi' ")
            else:
                 coordvecs = ['r','theta','phi']
        # used to deal with the change in the files
        if type(coordvecs)==np.ndarray:
            coordvecs = [str(ic) for ic in coordvecs]

        self.Param_List = paramlist
        self.Time_Vector = times
        self.Coord_Vecs = coordvecs
        self.Sensor_loc = sensor_loc
        self.Species = species
        (Nloc,Nt) = paramlist.shape[:2]
        #set up a Velocity measurement
        if velocity is None:
            self.Velocity=sp.zeros((Nloc,Nt,3))
        else:
            self.Velocity=velocity
        # set up a params name
        if paramnames is None:
            partparam = paramlist.shape[2:]
            if species is not None:
                paramnames = [['Ni_'+isp,'Ti_'+isp] for isp in species[:-1]]
                paramnames.append(['Ne','Te'])
                self.Param_Names=sp.array(paramnames,dtype=str)
            else:

                paramnums = np.arange(np.product(partparam))
                self.Param_Names = np.reshape(paramnums,partparam)
        else:
            self.Param_Names = paramnames

    #%% Getting closest objects
    def getclosestsphere(self,coords,timelist=None):
        d2r = np.pi/180.0
        (R,Az,El) = coords
        x_coord = R*np.cos(Az*d2r)*np.cos(El*d2r)
        y_coord = R*np.sin(Az*d2r)*np.cos(El*d2r)
        z_coord= R*np.sin(El*d2r)
        cartcoord = np.array([x_coord,y_coord,z_coord])
        return self.getclosest(cartcoord,timelist)
    def getclosest(self,coords,timelist=None):
        """This method will get the closest set of parameters in the coordinate space. It will return
        the parameters from all times.
        Input
        coords - A list of x,y and z coordinates.
        Output
        paramout - A NtxNp array from the closes output params
        sphereout - A Nc length array The sphereical coordinates of the closest point.
        cartout -  Cartisian coordinates of the closes point.
        """
        X_vec = self.Cart_Coords[:,0]
        Y_vec = self.Cart_Coords[:,1]
        Z_vec = self.Cart_Coords[:,2]

        xdiff = X_vec -coords[0]
        ydiff = Y_vec -coords[1]
        zdiff = Z_vec -coords[2]
        distall = xdiff**2+ydiff**2+zdiff**2
        minidx = np.argmin(distall)
        paramout = self.Param_List[minidx]
        velout = self.Velocity[minidx]
        if timelist is not None:
            timeindx = []
            for itime in timelist:
                timeindx.append(sp.argmin(sp.absolute(itime-self.Time_Vector)))
            paramout=paramout[timeindx]
            velout=velout[timeindx]
        sphereout = self.Sphere_Coords[minidx]
        cartout = self.Cart_Coords[minidx]
        return (paramout,velout,sphereout,cartout,np.sqrt(distall[minidx]))
    #%% Interpolation methods
    def interp(self,new_coords,ver=0,sensor_loc = None,method='linear',fill_value=np.nan):
        """This method will take the parameters in the Param_List variable and spatially.
        interpolate the points given the new coordinates. The method will be
        determined by the method input.
        Input:
        new_coords - A Nlocx3 numpy array. This will hold the new coordinates that
        one wants to interpolate the data over.
        sensor_loc - The location of the new sensor.
        method - A string. The method of interpolation curently only accepts 'linear',
        'nearest' and 'cubic'
        fill_value - The fill value for the interpolation.
        Output:
        iono1 - An instance of the ionocontainer class with the newly interpolated
        parameters.
        """
        if sensor_loc is None:
            sensor_loc=self.Sensor_loc

        curavalmethods = ['linear', 'nearest', 'cubic']
        interpmethods = ['linear', 'nearest', 'cubic']
        if method not in curavalmethods:
            raise ValueError('Must be one of the following methods: '+ str(curavalmethods))
        (Nloc,Nt) = self.Param_List.shape[:2]
        New_param = np.zeros_like(self.Param_List,dtype=self.Param_List.dtype)
        curcoords = self.Cart_Coords

        for itime in np.arange(Nt):
            curparamar = self.Param_List[:,itime]
            changeparams=False
            Nparams=np.product(curparamar.shape[1:])
            # deal with case where parameters are multi-dimensional arrays, flatten each array and apply interpolation
            if curparamar.ndim >2:
                changeparams=True
                oldshape = curparamar.shape
                curparamar = np.reshape(curparamar,(Nloc,Nparams))

            tempparams = np.zeros_like(curparamar)
            for iparam in np.arange(Nparams):
                curparam =curparamar[:,iparam]

                if method in interpmethods:
                    tempparams[:,iparam] = scipy.interpolate.griddata(curcoords,curparam,new_coords,method,fill_value)

            if changeparams:
                New_param[:,itime] = np.reshape(tempparams,oldshape)
            else:
                New_param[:,itime] =tempparams


        return IonoContainer(new_coords,New_param,self.Time_Vector,sensor_loc,ver,self.Coord_Vecs,self.Param_Names)


     #%% Read and Write Methods
    def savemat(self,filename):
        """ This method will write out a structured mat file and save information
        from the class.
        inputs
        filename - A string for the file name.
        """
        #outdict = {'Cart_Coords':self.Cart_Coords,'Sphere_Coords':self.Sphere_Coords,\
        #    'Param_List':self.Param_List,'Time_Vector':self.Time_Vector}
#        if self.Coord_Vecs!=None:
#            #fancy way of combining dictionaries
#            outdict = dict(outdict.items()+self.Coord_Vecs.items())
        outdict1 = vars(self)
        outdict = {ik:outdict1[ik] for ik in outdict1.keys() if not(outdict1[ik] is None)}
        sio.savemat(filename,mdict=outdict)

    def saveh5(self,filename):
        """This method will save the instance of the class to a structured h5 file.
        Input:
        filename - A string for the file name."""
        h5file = tables.openFile(filename, mode = "w", title = "IonoContainer out.")
        vardict = vars(self)
        try:
            # XXX only allow 1 level of dictionaries, do not allow for dictionary of dictionaries.
            # Make group for each dictionary
            for cvar in vardict.keys():
                #group = h5file.create_group(posixpath.sep, cvar,cvar +'dictionary')
                if type(vardict[cvar]) ==dict: # Check if dictionary
                    dictkeys = vardict[cvar].keys()
                    group2 = h5file.create_group('/',cvar,cvar+' dictionary')
                    for ikeys in dictkeys:
                        h5file.createArray(group2,ikeys,vardict[cvar][ikeys],'Static array')
                else:
                    if not(vardict[cvar] is None):
                        h5file.createArray('/',cvar,vardict[cvar],'Static array')
            h5file.close()
        except Exception as inst:
            print type(inst)
            print inst.args
            print inst
            h5file.close()
            raise NameError('Failed to write to h5 file.')

    @staticmethod

    def readmat(filename):
        """This method will read an instance of the class from a mat file.
        Input:
        filename - A string for the file name."""
        indata = sio.loadmat(filename,chars_as_strings=True)
        vardict = {'coordlist':'Cart_Coords','coordlist2':'Sphere_Coords','paramlist':'Param_List',\
            'times':'Time_Vector','sensor_loc':'Sensor_loc','coordvecs':'Coord_Vecs',\
            'paramnames':'Param_Names','species':'Species','velocity':'Velocity'}
        outdict = {}
        for ikey in vardict.keys():
            if vardict[ikey] in indata.keys():
                if (ikey=='species') and (type(indata[vardict[ikey]]) ==sp.ndarray):
                    indata[vardict[ikey]] = [str(''.join(letter)) for letter_array in indata[vardict[ikey]][0] for letter in letter_array]

                outdict[ikey] = indata[vardict[ikey]]
        #pdb.set_trace()
        if 'coordvecs' in outdict.keys():
            if [str(x) for x in outdict['coordvecs']] == ['r','theta','phi']:
                outdict['ver']=1
                outdict['coordlist']=outdict['coordlist2']
        del outdict['coordlist2']
        return IonoContainer(**outdict)
    @staticmethod

    def readh5(filename):
        """This method will read an instance of the class from a structured h5 file.
        Input:
        filename - A string for the file name."""

        vardict = {'coordlist':'Cart_Coords','coordlist2':'Sphere_Coords','paramlist':'Param_List',\
            'times':'Time_Vector','sensor_loc':'Sensor_loc','coordvecs':'Coord_Vecs',\
            'paramnames':'Param_Names', 'species':'Species','velocity':'Velocity'}
        vardict2 = {vardict[ikey]:ikey for ikey in vardict.keys()}
        outdict = {}

        h5file=tables.openFile(filename)
        output={}
        # Read in all of the info from the h5 file and put it in a dictionary.
        for group in h5file.walkGroups(posixpath.sep):
            output[group._v_pathname]={}
            for array in h5file.listNodes(group, classname = 'Array'):
                output[group._v_pathname][array.name]=array.read()
        h5file.close()
        outarr = [pathparts(ipath) for ipath in output.keys() if len(pathparts(ipath))>0]
        outlist = []
        basekeys  = output[posixpath.sep].keys()
        # Determine assign the entries to each entry in the list of variables.
        for ivar in vardict2.keys():
            dictout = False
            for npath,ipath in enumerate(outarr):
                if ivar==ipath[0]:
                    outlist.append(output[output.keys()[npath]])
                    dictout=True
                    break
            if dictout:
                continue

            if ivar in basekeys:
                outdict[vardict2[ivar]] = output[posixpath.sep][ivar]
        # determine version of data
        if 'coordvecs' in outdict.keys():
            if type(outdict['coordvecs']) ==sp.ndarray:
                outdict['coordvecs'] = outdict['coordvecs'].tolist()
            if outdict['coordvecs'] == ['r','theta','phi']:
                outdict['ver']=1
                outdict['coordlist']=outdict['coordlist2']
        if 'coordlist2' in outdict.keys():
            del outdict['coordlist2']
        return IonoContainer(**outdict)
    #%% Reduce numbers
    def coordreduce(self,coorddict):
        assert type(coorddict)==dict, "Coorddict needs to be a dictionary"
        ncoords = self.Cart_Coords.shape[0]
        coordlist = ['x','y','z','r','theta','phi']

        coordkeysorg = coorddict.keys()
        coordkeys = [ic for ic in coordkeysorg if ic in coordlist]

        ckeep = sp.ones(ncoords,dtype=bool)

        for ic in coordkeys:
            currlims = coorddict[ic]
            if ic=='x':
                tempcoords = self.Cart_Coords[:,0]
            elif ic=='y':
                tempcoords = self.Cart_Coords[:,1]
            elif ic=='z':
                tempcoords = self.Cart_Coords[:,2]
            elif ic=='r':
                tempcoords = self.Sphere_Coords[:,0]
            elif ic=='theta':
                tempcoords = self.Sphere_Coords[:,1]
            elif ic=='phi':
                tempcoords = self.Sphere_Coords[:,2]
            keeptemp = sp.logical_and(tempcoords>=currlims[0],tempcoords<currlims[1])
            ckeep = sp.logical_and(ckeep,keeptemp)
        # prune the arrays
        self.Cart_Coords=self.Cart_Coords[ckeep]
        self.Sphere_Coords=self.Sphere_Coords[ckeep]
        self.Param_List=self.Param_List[ckeep]
        self.Velocity=self.Velocity[ckeep]

    def timereduce(self, timelims=None,timesselected=None):
        assert (timelims is not None) or (timesselected is not None), "Need a set of limits or selected set of times"

        if timelims is not None:
            tkeep = sp.logical_and(self.Time_Vector>=timelims[0],self.Time_Vector<timelims[1])
        if timesselected is not None:
            tkeep = sp.in1d(self.Time_Vector,timesselected)
        # prune the arrays
        self.Time_Vector=self.Time_Vector[tkeep]
        self.Param_List=self.Param_List[:,tkeep]
        self.Velocity=self.Velocity[:,tkeep]

    #%% Operator Methods
    def __eq__(self,self2):
        """This is the == operator """
        vardict = vars(self)
        vardict2 = vars(self2)

        for ivar in vardict.keys():
            if ivar not in vardict2.keys():
                return False
            if type(vardict[ivar])!=type(vardict2[ivar]):
                return False
            if type(vardict[ivar])==np.ndarray:
                a = np.ma.array(vardict[ivar],mask=np.isnan(vardict[ivar]))
                blah = np.ma.array(vardict2[ivar],mask=np.isnan(vardict2[ivar]))
                if not np.ma.allequal(a,blah):
                    return False
            else:
                if vardict[ivar]!=vardict2[ivar]:
                    return False
            return True

    def __ne__(self,self2):
        '''This is the != operator. '''
        return not self.__eq__(self2)

    #%% Spectrum methods
    def makeallspectrums(self,sensdict,npts):
        pmshape= self.Param_Names.shape
        if pmshape==(7,):
            return self.makeallspectrumsv1(sensdict,npts)
        elif pmshape[1]==6:
            return self.makeallspectrumsv2(sensdict,npts)


    def makeallspectrumsv2(self,sensdict,npts):
        """This will create a numpy array of all of the spectrums for the data in
        the instance of the class.
        inputs:
        sensdict - The structured dictionary of for the sensor.
        npts - The number of points the spectrum is to be evaluated at.
        Output:
        omeg - A npts length numpy array. The frequency points that the ISR spectrum
        is evaluated over in Hz
        outspecs - A NlocxNtxnpts numpy array. The power spectrums for the plasma
        in the entire instance of the class. The spectrum will be the correct power
        level for the plasma parameters
        npts - The actual number of points that the spectrum is evaluated over."""

        specobj = ISRSpectrum(centerFrequency =sensdict['fc'],nspec = npts,sampfreq=sensdict['fs'])

        paramshape = self.Param_List.shape
        if self.Time_Vector is None:
            outspecs = np.zeros((paramshape[0],1,npts))
            full_grid = False
        else:
            outspecs = np.zeros((paramshape[0],paramshape[1],npts))
            full_grid = True

        (N_x,N_t) = outspecs.shape[:2]
        #pdb.set_trace()
        for i_x in np.arange(N_x):
            for i_t in np.arange(N_t):
                if full_grid:
                    cur_params = self.Param_List[i_x,i_t]
                else:
                    cur_params = self.Param_List[i_x]
                (omeg,cur_spec,rcs) = specobj.getspec(cur_params,rcsflag=True)
                cur_spec_weighted = len(cur_spec)**2*cur_spec*rcs/cur_spec.sum()
                outspecs[i_x,i_t] = cur_spec_weighted

        return (omeg,outspecs,npts)
    def makeallspectrumsopen(self,func,sensdict,npts):
        return func(self,sensdict,npts)

    def combinetimes(self,self2):
        a = np.ma.array(self.Cart_Coords,mask=np.isnan(self.Cart_Coords))
        blah = np.ma.array(self2.Cart_Coords,mask=np.isnan(self2.Cart_Coords))

        assert np.ma.allequal(a,blah), "Need to have same spatial coordinates"

        assert type(self.Param_Names)==type(self2.Param_Names),'Param_Names are different types, they need to be the same'


        assert sp.all(self.Param_Names == self2.Param_Names), "Need to have same parameter names"
        assert self.Species== self2.Species, "Need to have the same species"

        self.Time_Vector = sp.concatenate((self.Time_Vector,self2.Time_Vector))
        self.Velocity = sp.concatenate((self.Velocity,self2.Velocity),1)
        self.Param_List = sp.concatenate((self.Param_List,self2.Param_List),1)
    def makespectruminstance(self,sensdict,npts):
        """This will create another instance of the Ionocont class
        inputs:
        sensdict - The structured dictionary of for the sensor.
        npts - The number of points the spectrum is to be evaluated at.
        Output:
        Iono1 - An instance of the IonoContainer class with the spectrums as the
        param vectors and the param names will be the the frequency points """
        (omeg,outspecs,npts) = self.makeallspectrums(sensdict,npts)
        return IonoContainer(self.Cart_Coords,outspecs,self.Time_Vector,self.Sensor_loc,paramnames=omeg)
    def makespectruminstanceopen(self,func,sensdict,npts):
        """This will create another instance of the Ionocont class
        inputs:
        func - A function used to create the spectrums
        sensdict - The structured dictionary of for the sensor.
        npts - The number of points the spectrum is to be evaluated at.
        Output:
        Iono1 - An instance of the IonoContainer class with the spectrums as the
        param vectors and the param names will be the the frequency points """
        (omeg,outspecs,npts) = self.makeallspectrumsopen(func,sensdict,npts)
        return IonoContainer(self.Cart_Coords,outspecs,self.Time_Vector,self.Sensor_loc,paramnames=omeg)
    def getDoppler(self,sensorloc=sp.zeros(3)):
        ncoords = self.Cart_Coords.shape[0]
        ntimes = len(self.Time_Vector)
        if not sp.alltrue(sensorloc == sp.zeros(3)):
            curcoords = self.Cart_Coords -sp.tile(sensorloc[sp.newaxis,:],(ncoords,1))
        else:
            curcoords = self.Cart_Coords
        denom = np.tile(sp.sqrt(sp.sum(curcoords**2,1))[:,sp.newaxis],(1,3))
        unit_coords = curcoords/denom
#        pdb.set_trace()
        Vi = sp.zeros((ncoords,ntimes))
        for itime in range(ntimes):
            Vi[:,itime] = (self.Velocity[:,itime]*unit_coords).sum(1)
        return Vi

    def copy(self):
        """This is the function to copy an instance of the class."""
        return copy.copy(self)
#        vardict = vars(self)
#        outdict = {}
#        for ikey in vardict.keys():
#            obj1 = vardict[ikey]
#            if type(obj1)==str:
#                outdict[ikey] = (obj1+'.')[:-1]
#            else:
#                outdict[ikey] = obj1.copy()
#
#        if 'coordvecs' in outdict.keys():
#            if [str(x) for x in outdict['coordvecs']] == ['r','theta','phi']:
#                outdict['ver']=1
#                outdict['coordlist']=outdict['coordlist2']
#        del outdict['coordlist2']
#        return IonoContainer(**outdict)

    def deepcopy(self):
        return copy.deepcopy(self)
#%%    utility functions
def pathparts(path):
    '''This will break up a path name into componenets using recursion
    Input - path a string seperated by a posix path seperator.
    Output - A list of strings of the path parts.'''
    components = []
    while True:
        (path,tail) = posixpath.split(path)
        if tail == "":
            components.reverse()
            return components
        components.append(tail)

def MakeTestIonoclass(testv=False,testtemp=False,N_0=1e11,z_0=250.0,H_0=50.0):
    """ This function will create a test ionoclass with an electron density that
    follows a chapman function"""
    xvec = sp.arange(-250.0,250.0,20.0)
    yvec = sp.arange(-250.0,250.0,20.0)
    zvec = sp.arange(50.0,500.0,5.0)
    # Mesh grid is set up in this way to allow for use in MATLAB with a simple reshape command
    xx,zz,yy = sp.meshgrid(xvec,zvec,yvec)
    coords = sp.zeros((xx.size,3))
    coords[:,0] = xx.flatten()
    coords[:,1] = yy.flatten()
    coords[:,2] = zz.flatten()

#    H_0 = 50.0 #km scale height
#    z_0 = 250.0 #km
#    N_0 = 10**11

    # Make electron density
    Ne_profile = Chapmanfunc(zz,H_0,z_0,N_0)
    # Make temperture background
    if testtemp:
        (Te,Ti)= TempProfile(zz)
    else:
        Te = np.ones_like(zz)*1000.0
        Ti = np.ones_like(zz)*1000.0

    # set up the velocity
    (Nlocs,ndims) = coords.shape
    vel = sp.zeros((Nlocs,1,ndims))
    times = [0.0]
    if testv:
        vel[:,0,2] = zz.flatten()/5.0
    species=['O+','e-']
    # put the parameters in order
    params = sp.zeros((Ne_profile.size,1,2,2))
    params[:,0,0,1] = Ti.flatten()
    params[:,0,1,1] = Te.flatten()
    params[:,0,0,0] = Ne_profile.flatten()
    params[:,0,1,0] = Ne_profile.flatten()


    Icont1 = IonoContainer(coordlist=coords,paramlist=params,times = times,sensor_loc = sp.zeros(3),ver =0,coordvecs =
        ['x','y','z'],paramnames=None,species=species,velocity=vel)
    return Icont1

def main():
    curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curpath)[0],'testdata')

    Icont1 = MakeTestIonoclass()

    Icont1.savemat(os.path.join(testpath,'testiono.mat'))
    Icont1.saveh5(os.path.join(testpath,'testiono.h5'))
    Icont2 = IonoContainer.readmat(os.path.join(testpath,'testiono.mat'))
    Icont3 = IonoContainer.readh5(os.path.join(testpath,'testiono.h5'))

    if Icont1==Icont2:
        print "Mat file saving and reading works"
    else:
        print "Something is wrong with the Mat file writing and reading"
    if Icont1==Icont3:
        print "h5 file saving and reading works"
    else:
        print "Something is wrong with the h5 file writing and reading"
    #%% Main
if __name__== '__main__':


   main()