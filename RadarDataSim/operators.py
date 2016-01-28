#!/usr/bin/env python
"""
Created on Tue Dec 29 15:49:01 2015

@author: John Swoboda
"""

import tables
from const.physConstants import v_C_0
from utilFunctions import readconfigfile
from IonoContainer import IonoContainer
import scipy as sp
import pdb
class RadarSpaceTimeOperator(object):

    def __init__(self,ionoin=None,configfile=None,timein=None,RSTOPinv=None,invmat=None):
        d2r = sp.pi/180.0
        if RSTOPinv is None:
            (sensdict,simparams) = readconfigfile(configfile)


            
            if isinstance(ionoin,basestring):
                ionoin = IonoContainer.readh5(ionoin)
            elif isinstance(ionoin,list):
                if isinstance(ionoin[0],basestring):
                    ionoin = IonoContainer.readh5(ionoin[0])
                else:
                    ionoin=ionoin[0]
            #Input location
            self.Cart_Coords_In = ionoin.Cart_Coords
            self.Sphere_Coords_In = ionoin.Sphere_Coords,
            # time
            if timein is None:
                self.Time_In = ionoin.Time_Vector
            else:
                self.Time_In = timein

            #output locations
            rng_vec2 = simparams['Rangegatesfinal']
            nrgout = len(rng_vec2)

            angles = simparams['angles']
            nang =len(angles)

            ang_data = sp.array([[iout[0],iout[1]] for iout in angles])
            rng_all = sp.tile(rng_vec2,(nang))
            ang_all = sp.repeat(ang_data,nrgout,axis=0)

            self.Sphere_Coords_Out = sp.column_stack((rng_all,ang_all))
            (R_vec,Az_vec,El_vec) = (self.Sphere_Coords_Out[:,0],self.Sphere_Coords_Out[:,1],
                self.Sphere_Coords_Out[:,2])
            xvecmult = sp.cos(Az_vec*d2r)*sp.cos(El_vec*d2r)
            yvecmult = sp.sin(Az_vec*d2r)*sp.cos(El_vec*d2r)
            zvecmult = sp.sin(El_vec*d2r)
            X_vec = R_vec*xvecmult
            Y_vec = R_vec*yvecmult
            Z_vec = R_vec*zvecmult

            self.Cart_Coords_Out = sp.column_stack((X_vec,Y_vec,Z_vec))
            self.Time_Out = simparams['Timevec']

            (self.RSTMat,self.blocks,self.blocksize,self.blocklocs) = makematPA(ionoin.Sphere_Coords,ionoin.Time_Vector,configfile)
        elif configfile is None:

            self.Cart_Coords_Out = RSTOPinv.Cart_Coords_In
            self.Sphere_Coords_Out = RSTOPinv.Sphere_Coords_In
            self.Time_Out =  RSTOPinv.Time_In

            self.RSTMat = invmat

            self.Cart_Coords_In = self.Cart_Coords_Out
            self.Sphere_Coords_In =self.Sphere_Coords_Out
            self.Time_In = self.Time_Out


    def mult_iono(self,ionoin):

        ntin = self.Time_In.shape[0]
        matsttimes = self.Time_In[:,0]
        ntout = self.Time_Out.shape[0]
        ntcounts =sp.zeros(ntout)
        nlout = self.Cart_Coords_Out.shape[0]
        blocklocs = self.blocklocs

        firsttime = True
        if isinstance(ionoin,list):
            ionolist = ionoin
        else:
            ionolist = [ionoin]

        for ionon,iiono in enumerate(ionolist):
            if isinstance(iiono,str):
                curiono = IonoContainer.readh5(iiono)
            else:
                curiono=iiono


            ionodata = curiono.Param_List
            ionotime = curiono.Time_Vector
            ionocart = curiono.Cart_Coords

            assert sp.allclose(ionocart,self.Cart_Coords_In), "Spatial Coordinates need to be the same"

            ionosttimes = ionotime[:,0]

            keeptimes = sp.arange(ntin)[sp.in1d(matsttimes,ionosttimes)]

            (nl,nt,np) = ionodata.shape
            if firsttime:
                outdata=sp.zeros((nlout,ntout,np))
                firsttime==False

            b_locsind = sp.arange(len(blocklocs))[sp.in1d(blocklocs[:,0],keeptimes)]
            b_locs = blocklocs[b_locsind]
            if len(self.RSTMat)==1:
                mainmat = self.RSTMat[0]
                for ibn,(iin,iout) in enumerate(b_locs):
                    ntcounts[iin]=ntcounts[iin]+1
                    for iparam in range(np):
                        outdata[:,iout,iparam]=mainmat.dot(ionodata[:,iin,iparam])

            else:
                for ibn,(iin,iout) in enumerate(b_locs):
                    ntcounts[iout]=ntcounts[iout]+1
                    mainmat=self.RSTMat[b_locsind[ibn]]
                    for iparam in range(np):
                        outdata[:,iout,iparam]=mainmat.dot(ionodata[:,iin,iparam])

        # If any times had no returns remove them
        # divid out data that was added together
        outkeep = ntcounts>0
        outdata = outdata[:,outkeep]
        ntcounts=ntcounts[outkeep]
        for itm in range(outdata.shape[1]):
            outdata[:,itm]=outdata[:,itm]/ntcounts[itm]
        
        outiono = IonoContainer(self.Cart_Coords_Out,outdata,times=self.Time_Out[outkeep],sensor_loc=curiono.Sensor_loc,
                               ver=0,paramnames=curiono.Param_Names,species=curiono.Species,
                               velocity=curiono.Velocity)
        return outiono
    def invertdata(self,ionoin,alpha,tik_type = 'i',type=None,dims=None,order='C',max_it=100,tol=1e-2):

        ntin = self.Time_In.shape[0]
        matsttimes = self.Time_Out[:,0]
        ntout = self.Time_Out.shape[0]
        ntcounts =sp.zeros(ntin)
        nlin = self.Cart_Coords_In.shape[0]
        nlout = self.Cart_Coords_Out.shape[0]
        blocklocs = self.blocklocs

        firsttime = True
        if isinstance(ionoin,list):
            ionolist = ionoin
        else:
            ionolist = [ionoin]

        for ionon,iiono in enumerate(ionolist):
            if isinstance(iiono,str):
                curiono = IonoContainer.readh5(iiono)
            else:
                curiono=iiono


            ionodata = curiono.Param_List
            ionotime = curiono.Time_Vector
            ionocart = curiono.Cart_Coords

            assert sp.allclose(ionocart,self.Cart_Coords_Out), "Spatial Coordinates need to be the same"

            ionosttimes = ionotime[:,0]

            keeptimes = sp.arange(ntin)[sp.in1d(matsttimes,ionosttimes)]

            (nl,nt,np) = ionodata.shape
            if firsttime:
                bdata = sp.zeros((nlin,ntin,np))
                invdata=sp.zeros((nlin,ntin,np))
                firsttime==False

            b_locsind = sp.arange(len(blocklocs))[sp.in1d(blocklocs[:,0],keeptimes)]
            b_locs = blocklocs[b_locsind]


            
            for ibn,(iin,iout) in enumerate(b_locs):
                if len(self.RSTMat)==1:
                    A = self.RSTMat[0]
                else:
                    A=self.RSTMat[b_locsind[ibn]]
                ntcounts[iin]=ntcounts[iin]+1
                
                for iparam in range(np):
                    bdata[:,iin,iparam]=A.transpose().dot(ionodata[:,iout,iparam])


                C = sp.dot(A.transpose(),A)
    
                if type is None or type.lower()=='i':
                    L=sp.sparse.eye(C.shape[0])
                elif type.lower()=='d':
                    L=diffmat(dims,order)
    
                Ctik=C+sp.power(alpha,2)*L
                M=L*Ctik
                xin = sp.ones(nlin,dtype=ionodata.dtype)
                for i in range(np):
                    (invdata[:,iin,i], error, iter, flag) = cgmat(Ctik, xin, bdata[:,iin,i], M, max_it, tol)

def makematPA(Sphere_Coords,timein,configfile,vel=None):
    """Make a Ntimeout*Nbeam*Nrng x Ntime*Nloc matrix. The output space will have range repeated first,
    then beams then time. The coordinates will be [t0,b0,r0],[t0,b0,r1],[t0,b0,r2],...
    [t0,b1,r0],[t0,b1,r1], ... [t1,b0,r0],[t1,b0,r1],...[t1,b1,r0]..."""
    #
    (sensdict,simparams) = readconfigfile(configfile)
    timeout = simparams['Timevec']
    Tint = simparams['Tint']
    timeout = sp.column_stack((timeout,timeout+Tint))

    rng_vec = simparams['Rangegates']
    rng_bin=sensdict['t_s']*v_C_0/1000.0
    sumrule = simparams['SUMRULE']
    #
    minrgbin = -sumrule[0].min()
    maxrgbin = len(rng_vec)-sumrule[1].max()
    minrg = minrgbin*rng_bin
    maxrg = maxrgbin*rng_bin
    angles = simparams['angles']
    Nbeams = len(angles)
    rho = Sphere_Coords[:,0]
    Az = Sphere_Coords[:,1]
    El = Sphere_Coords[:,2]

    rng_vec2 = simparams['Rangegatesfinal']
    nrgout = len(rng_vec2)

    Nlocbeg = len(rho)
    Ntbeg = len(timein)
    Ntout = len(timeout)

    # set up blocks
    blocksize = (Nbeams*nrgout,Nlocbeg)
    blocks = (Ntout,Ntbeg)

    if vel is None:

        for iton,ito in enumerate(timeout):
            overlaps=[]
            for ix, x in enumerate(timein):
                if x[0]<ito[1] and x[1]>ito[0]:
                    overlaps.append(sp.array([iton,ix]))
            
            tempover = sp.array(overlaps)
            if iton==0:
                blocklocs = tempover.copy()
            else:
                blocklocs=sp.vstack((blocklocs,tempover))
        outmat = sp.sparse.lil_matrix(blocksize,dtype =sp.float64)

        weights = {ibn:sensdict['ArrayFunc'](Az,El,ib[0],ib[1],sensdict['Angleoffset']) for ibn, ib in enumerate(angles)}

        for ibn in range(Nbeams):
            print('\t\t Making Beam {0:d} of {1:d}'.format(ibn,Nbeams))
            weight = weights[ibn]
            for isamp in range(nrgout):
                # make the row
                irow = isamp+ibn*nrgout

                range_g = rng_vec2[isamp]
                rnglims = [range_g-minrg,range_g+maxrg]
                rangelog = sp.argwhere((rho>=rnglims[0])&(rho<rnglims[1]))

                # This is a nearest neighbors interpolation for the spectrums in the range domain
                if sp.sum(rangelog)==0:
                    minrng = sp.argmin(sp.absolute(range_g-rho))
                    rangelog[minrng] = True
                #create the weights and weight location based on the beams pattern.
                weight_cur =weight[rangelog[:,0]]
                weight_cur = weight_cur/weight_cur.sum()
                weight_loc = sp.where(rangelog[:,0])[0]

                icols = weight_loc

                weights_final = weight_cur[weight_loc]*range_g**2/rho[weight_loc]**2
                outmat[irow,icols] = weights_final
        outmat=[outmat]
    else:
        for iton,ito in enumerate(timeout):
            overlaps = sp.array([getOverlap(ito,x) for x in timein])
            weights_time = overlaps/overlaps.sum()
            itpnts = sp.where(weights_time>0)[0]

            # usually the matrix size is nbeamsxnrange
            for ibn in range(Nbeams):
                print('\t\t Making Beam {0:d} of {1:d}'.format(ibn,Nbeams))
                weight = weights[ibn]
                for isamp in range(nrgout):
                    # make the row
                    irow = isamp+ibn*nrgout+iton*nrgout*Nbeams

                    range_g = rng_vec2[isamp]
                    rnglims = [range_g-minrg,range_g+maxrg]
                    rangelog = sp.argwhere((rho>=rnglims[0])&(rho<rnglims[1]))

                    # This is a nearest neighbors interpolation for the spectrums in the range domain
                    if sp.sum(rangelog)==0:
                        minrng = sp.argmin(sp.absolute(range_g-rho))
                        rangelog[minrng] = True
                    #create the weights and weight location based on the beams pattern.
                    weight_cur =weight[rangelog[:,0]]
                    weight_cur = weight_cur/weight_cur.sum()
                    weight_loc = sp.where(rangelog[:,0])[0]

                    w_loc_rep = sp.tile(weight_loc,len(itpnts))
                    t_loc_rep = sp.repeat(itpnts,len(weight_loc))
                    icols = t_loc_rep*Nlocbeg+w_loc_rep

                    weights_final = weights_time[t_loc_rep]*weight_cur[w_loc_rep]*range_g**2/rho[w_loc_rep]**2
                    outmat[irow,icols] = weights_final


    return(outmat,blocks,blocksize,blocklocs)


def saveoutmat(filename,Sphere_Coords,timein,configfile):

    outmat = makematPA(Sphere_Coords,timein,configfile)
    h5file = tables.openFile(filename, mode = "w", title = "Radar Matrix out.")
    h5file.createArray('/','RadarMatrix',outmat,'Static array')
    h5file.close()

def readradarmat(filename):
    h5file=tables.openFile(filename)
    outmat = h5file.root.RadarMatrix.read()
    h5file.close()
    return outmat
#%% extra functions
def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

#%% Math Functions
def cgmat(A,x,b,M=None,max_it=100,tol=1e-8):

    if M is None:
        M= sp.diag(A)
    bnrm2 = sp.linalg.norm(b)
    r=b-A.dot(x)
    rho=sp.zeros(max_it)
    for i in range(max_it):
        z=sp.linalg.solve(M,r)
        rho[i] = sp.dot(r,z)
        if i==0:
            p=z
        else:
            beta=rho/rho[i-1]
            p=z+beta*p

        q=A.dot(p)
        alpha=rho/sp.dot(p,q)
        x = x+alpha*p
        r = r-alpha*q
        error = sp.linalg.norm( r ) / bnrm2
        if error <tol:
            return (x,error,i,False)

    return (x,error,max_it,True)


def diffmat(dims,order = 'C'):

    xdim = dims[0]
    ydim = dims[1]
    dims[0]=ydim
    dims[1]=xdim

    if order.lower() == 'c':
        dims = dims[::-1]

    outD = []
    for idimn, idim in enumerate(dims):
        if idim==0:
            outD.append(sp.array([]))
            continue
        e = sp.ones(idim)
        dthing = sp.column_stack((-e,e))
        D = sp.sparse.spdiags(dthing,[0,1],idim-1,idim)
        D = sp.vstack((D,D[-1]))
        if idimn>0:
            E = sp.sparse.eye(sp.prod(dims[:idimn]))
            D = sp.kron(D,E)

        if idimn<len(dims)-1:
            E = sp.sparse.eye(sp.prod(dims[idimn+1:]))
            D = sp.kron(E,D)

        outD.append(D)
    if order.lower() == 'c':
        outD=outD[::-1]
    Dy=outD[0]
    Dx = outD[1]
    outD[0]=Dx
    outD[1]=Dy

    return tuple(outD)