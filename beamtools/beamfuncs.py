#!/usr/bin/env python
"""
Created on Wed Feb 26 16:23:30 2014

@author: Bodangles
"""

import os
import numpy as np
import tables
import glob
import matplotlib.pyplot as plt
import pdb

class BeamSelector(object):
    """This class will take in a numpy array with the first column being the beam numbers
    the second coloumn the az angle in degrees and the third column the el angle in degrees.
    The forth column is ksys variable for each beam."""

    def __init__(self,beammat,beamlist=None):
        """ This constructor function takes a list of beams and a beam mat from
        the hdf5 files from SRI which is a Nx4 table of beams"""
        beammat = np.array(beammat)
        self.beamnumdict = {}
        self.beamnumxydict = {}
        self.angledict = {}
        self.xydict = {}
        self.zenith = False # If this is false elevation is referenced to z=0 if true its referenced to z axis
        allbeams = np.array([int(ib) for ib in beammat[:,0]])
        if beamlist is None:
            beamlist = allbeams
        else:
            beamlog = np.in1d(allbeams,[] )
            for ibeam in beamlist:
                beamlog = beamlog+(int(ibeam)==allbeams)
            beammat=beammat[beamlog]
        self.beammat = beammat
        for ibeam,curlist in enumerate(beammat):
            curlist[1] = np.mod(curlist[1],360.)# mod the az so it goes from 0 to 360
            self.beamnumdict[int(curlist[0])] = {'az':curlist[1],'el':curlist[2]}

            self.angledict[(curlist[1],curlist[2])] = int(curlist[0])
            xydata = angles2xy(curlist[1],curlist[2])
            self.xydict[xydata] = int(curlist[0])
            self.beamnumxydict[int(curlist[0])] = np.array(xydata)
    def updatebeamlist(self,beamlist,azvec,elvec):
        assert len(azvec)==len(elvec)
        assert len(azvec)==len(beamlist)

        azvec = np.mod(azvec,360.0)
        for nb,ib in enumerate(beamlist):
            #clean up the dictionaries
            angtemp = self.beamnumdict[int(ib)]
            angtuple = (angtemp['az'],angtemp['el'])
            xydata = angles2xy(angtuple[0],angtuple[1])
            if xydata in self.xydict.keys(): del self.xydict[xydata]
            if angtuple in self.angledict.keys(): del self.angledict[angtuple]
            # add the new info to the dictionaries
            self.beamnumdict[int(ib)]={'az':azvec[nb],'el':elvec[nb]}
            self.angledict[(azvec[nb],elvec[nb])] = int(ib)
            xydata = angles2xy(azvec[nb],elvec[nb])
            self.xydict[xydata] = int(ib)
            self.beamnumxydict[int(ib)] = np.array(xydata)

    def shiftbeams(self,azoff=0.0,eloff=0.0):
        beamnums = self.beamnumdict.keys()
        (azvec,elvec) = self.azelvecs()

        azvec = np.mod(azvec-azoff,360.0);
        (xx,yy,zz) = angles2xyz(azvec,elvec)
        rotmat = elrotmatrix(eloff)
        outmat = rotmat*np.mat([xx,yy,zz])
        (azvec,elvec) = xyz2angles(np.array(outmat[0]).flatten(),
             np.array(outmat[1]).flatten(),np.array(outmat[2]).flatten())

        self.updatebeamlist(beamnums,azvec,elvec)
    def switchzenith(self,report=False):
        beamnums = self.beamnumdict.keys()
        (azvec,elvec) = self.azelvecs()
        if self.zenith==False:
            self.zenith=True
            self.updatebeamlist(beamnums,azvec,elvec-90)
        else:
            self.zenith=False
            self.updatebeamlist(beamnums,azvec,elvec+90)
        if report:
            print "Zenith now: " +str(self.zenith)

    def getbeamsdist(self,beamnum,desdist,distdict=None):
        """ This will get all of the beams within a specific spatial distance to
        a beam. The user can give a distance dictionary which for each beam will
        give a distance. """
        if distdict is None:
            distdict = self.__calcxydist__(beamnum)
        #make reverse dict
        revdist = {distdict[ikey]:ikey for ikey in distdict.keys()}
        alldist = np.array(revdist.keys())
        disttup = np.where(alldist<desdist)
        distlist = alldist[disttup[0]]
#        pdb.set_trace()
        outbeamlist = [revdist[idist] for idist in distlist]
        return outbeamlist


    def getbeamangdist(self,beamnum,desdist):
        """ This will get the beam numbers in a specific angular distance.  The
        distances in az and el will be different."""
        cur_azel = self.beamnumdict[beamnum]

        outbeamlist = []
        for ibeam in self.beamnumdict.keys():
            curangls = self.beamnumdict[ibeam]
            azmet = np.abs(curangls['az']-cur_azel['az'])<desdist[0]
            elmet = np.abs(curangls['el']-cur_azel['el'])<desdist[1]
            if azmet and elmet:
                outbeamlist.append(ibeam)

        return outbeamlist

    def getbeammat(self,beamlist=None):
        if beamlist is None:
            return self.beammat
        allbeams = [int(ibeam) for ibeam in self.beammat[:,0]]
        indxlist = [np.where(ibeam==allbeams)[0] for ibeam in beamlist]
        return self.beammat[indxlist,:]

    def __calcxydist__(self,beamnum):
        beamlist = self.beamnumdict.keys()
        xycur = self.beamnumxydict[beamnum]
        distdict = {}
        for ibeam in beamlist:
            ixy = self.beamnumxydict[ibeam]
            distdict[ibeam] = np.sqrt(np.sum((xycur-ixy)**2))
        return distdict
    def azelvecs(self,beamlist=None):
        if beamlist is None:
            beamlist = self.beamnumdict.keys()

        azvec = np.array([self.beamnumdict[ib]['az'] for ib in beamlist])
        elvec = np.array([self.beamnumdict[ib]['el'] for ib in beamlist])
        return (azvec,elvec)
    def xyvecs(self,beamlist=None):
        if beamlist is None:
            beamlist = self.beamnumdict.keys()
        (az,el) = self.azelvecs(beamlist)
        (xx,yy) = angles2xy(az,el)
        return (xx,yy)
    def rotatebeams(self,azoff=0.0,eloff=0.0):
        (azall,elall) = self.azelvecs()

    def plotbeams(self,beamlist,plotall=True,filename=None,title=None):
        """Plots the location of the beams in yellow and plots the original beams
        in blue"""
        fig = make_polax(self.zenith)

        if plotall:
            (azall,elall) = self.azelvecs()
            (xx,yy) = angles2xy(azall,elall,self.zenith)
            plt.plot(xx,yy,'ko')
            plt.plot(xx,yy,'b.')
            plt.hold(True)

        (azvec,elvec) = self.azelvecs(beamlist)
        (xx2,yy2) = angles2xy(azvec,elvec,self.zenith)
        plt.plot(xx2,yy2,'ko')
        plt.plot(xx2,yy2,'y.')
        plt.title(title)
        if filename != None:
            plt.savefig(filename)
            print 'Saved image in '+filename
        return fig
    def printbeamlist(self,filename='outbeams.txt',beamlist=None):
        if beamlist is None:
            beamlist = self.beamnumdict.keys()

        f = open(filename,'w')
        for ib in beamlist:
            f.write(str(int(ib))+'\n')
        f.close()

    def printbeamangles(self,filename='outangles.txt',beamlist=None):
        if beamlist is None:
            beamlist = self.beamnumdict.keys()
        (azvec,elvec) = self.azelvecs(beamlist)

        f = open(filename,'w')
        for ib in range(len(beamlist)):
            f.write(str(azvec[ib])+ ' ' +str(elvec[ib])+'\n')
        f.close()

def make_polax(zenith=False):
    """ This makes the polar axes for the beams"""
    if zenith:
        minel = 0.0
        maxel = 70.0
        elspace = 10.0
        ellines = np.arange(minel,maxel,elspace)
    else:
        minel = 30.0
        maxel = 90.0
        elspace = 10.0
        ellines = np.arange(minel,maxel,elspace)


    azlines = np.arange(0.0,360.0,30.0)


    fig = plt.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='w')
    # plot all of the azlines
    elvec = np.linspace(maxel,minel,100)
    firstplot = True
    for iaz in azlines:
        azvec = iaz*np.ones_like(elvec)
        (xx,yy) = angles2xy(azvec,elvec,zenith)
        plt.plot(xx,yy,'k--')
        if firstplot:
            plt.hold(True)
            firstplot=False

        (xt,yt) = angles2xy(azvec[-1],elvec[-1]-5,zenith)
        plt.text(xt,yt,str(int(iaz)))

    azvec = np.linspace(0.0,360,100)
    # plot the el lines
    for iel in ellines:
        elvec = iel*np.ones_like(azvec)
        (xx,yy) = angles2xy(azvec,elvec,zenith)
        plt.plot(xx,yy,'k--')
        (xt,yt) = angles2xy(315,elvec[-1]-3,zenith)
        plt.text(xt,yt,str(int(iel)))
    plt.axis([-90,90,-90,90])
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
#    plt.axis('off')
    return fig


def angles2xy(az,el,zenith=False):
    """ This will take az and el angles and move them to a Cartisian space for plotting"""

    azt = (az)*np.pi/180.0
    if not zenith:
        el = 90-el
    xout = el*np.sin(azt)
    yout = el*np.cos(azt)
    return (xout,yout)

def xy2angles(x,y):
    elout = 90-np.sqrt(x**2+y**2)
    azout = (180.0/np.pi)*np.arctan2(x,y)
    return (azout,elout)
def angles2xyz(az,el):
    elrad = el*np.pi/180.0
    azrad = az*np.pi/180.0
    x = np.cos(elrad)*np.cos(azrad)
    y = np.cos(elrad)*np.sin(azrad)
    z = np.sin(elrad)
    return (x,y,z)
def xyz2angles(x,y,z):
    el = np.arcsin(z)*180.0/np.pi
    az = np.arctan2(y,x)*180/np.pi
    return(az,el)

def elrotmatrix(theta):
    thetar = theta*np.pi/180.0
    return np.mat([[np.cos(thetar),0.0,np.sin(thetar)],[0.,1.,0.],[-np.sin(thetar),0.,np.cos(thetar)]])
def rotinel(azvec,elvec,eloff):
    (xx,yy,zz) = angles2xyz(azvec,elvec)
    rotmat = elrotmatrix(eloff)
    outmat = rotmat*np.mat([xx,yy,zz])
    (azvec,elvec) = xyz2angles(np.array(outmat[0]).flatten(),
         np.array(outmat[1]).flatten(),np.array(outmat[2]).flatten())

if __name__ == "__main__":
    """ Test function for class requires hdf5 files present for test."""
    curfile = 'd0079140.dt0.h5'
    filepath, fext =os.path.splitext(curfile)
    h5file = tables.open_file(curfile)
    #read data
    # take only the first record because thats all thats needed
    beamcode_order_power = h5file.getNode('/Raw11/RawData/Beamcodes').read()[0]
    beamcode_order_iq = h5file.getNode('/Raw11/RawData/RadacHeader/BeamCode').read()[0]
    beamcode_list = h5file.getNode('/Setup/BeamcodeMap').read()

    h5file.close()

    beamselect = BeamSelector(beamcode_list,beamcode_order_power)
    testbeamlist = beamselect.getbeamsdist(beamcode_order_power[500],2)
    print(testbeamlist)
    testbeamlist2 = beamselect.getbeamangdist(beamcode_order_power[500],[4,3])
    print(testbeamlist2)
    beamselect.plotbeams(testbeamlist2)
    plt.show()