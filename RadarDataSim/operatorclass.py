#!/usr/bin/env python
"""
Created on Tue Apr 14 11:10:40 2015

@author: John Swoboda
"""
import tables
from const.physConstants import v_C_0
from utilFunctions import readconfigfile
from IonoContainer import IonoContainer
import scipy as sp

def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

class RadarSpaceTimeOperator(object):

    def __init__(self,ionoin=None,configfile=None,RSTOPinv=None,invmat=None):
        r2d = 180.0/sp.pi
        d2r = sp.pi/180.0
        if RSTOPinv is None:
            (sensdict,simparams) = readconfigfile(configfile)
            nt = ionoin.Time_Vector.shape[0]
            nloc = ionoin.Sphere_Coords.shape[0]

            #Input location
            self.Cart_Coords_In = ionoin.Cart_Coords
            self.Sphere_Coords_In = ionoin.Sphere_Coords,
            self.Time_In = ionoin.Time_Vector
            self.Cart_Coords_In_Rep = sp.tile(ionoin.Cart_Coords,(nt,1))
            self.Sphere_Coords_In_Rep = sp.tile(ionoin.Sphere_Coords,(nt,1))
            self.Time_In_Rep  = sp.repeat(ionoin.Time_Vector,nloc,axis=0)

            #output locations
            rng_vec2 = simparams['Rangegatesfinal']
            nrgout = len(rng_vec2)

            angles = simparams['angles']
            nang =len(angles)

            ang_data = sp.array([[iout[0],iout[1]] for iout in angles])
            rng_all = sp.tile(rng_vec2,(nang))
            ang_all = sp.repeat(ang_data,nrgout,axis=0)
            nlocout = nang*nrgout

            ntout = len(simparams['Timevec'])
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
            self.Time_Out_Rep =sp.repeat(simparams['Timevec'],nlocout,axis=0)
            self.Sphere_Coords_Out_Rep =sp.tile(self.Sphere_Coords_Out,(ntout,1))
            self.Cart_Coords_Out_Rep =sp.tile(self.Cart_Coords_Out,(ntout,1))
            self.RSTMat = makematPA(ionoin.Sphere_Coords,ionoin.Time_Vector,configfile)
        elif configfile is None:

            self.Cart_Coords_Out = RSTOPinv.Cart_Coords_In
            self.Sphere_Coords_Out = RSTOPinv.Sphere_Coords_In
            self.Time_Out =  RSTOPinv.Time_In
            self.Time_Out_Rep =RSTOPinv.Time_In_Rep
            self.Sphere_Coords_Out_Rep =RSTOPinv.Sphere_Coords_In_Rep
            self.Cart_Coords_Out_Rep =RSTOPinv.Cart_Coords_In_Rep
            self.RSTMat = invmat

            self.Cart_Coords_In = self.Cart_Coords_Out
            self.Sphere_Coords_In =self.Sphere_Coords_Out
            self.Time_In = self.Time_Out
            self.Cart_Coords_In_Rep = self.Cart_Coords_Out_Rep
            self.Sphere_Coords_In_Rep = self.Sphere_Coords_Out_Rep
            self.Time_In_Rep  = self.Time_Out_Rep

    def mult_iono(self,ionoin):


        ntout = self.Time_Out.shape[0]
        nlout = self.Cart_Coords_Out.shape[0]

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
            matsttimes = self.Time_In_Rep[:,0]

            keeptimes = sp.in1d(matsttimes,ionosttimes)

            mainmat = self.RSTMat[:,keeptimes]
            (nl,nt,np) = ionodata.shape
            ionodata = sp.reshape(ionodata,(nl*nt,np),order='F')

            outar = sp.zeros((nlout*ntout,np)).astype(ionodata.dtype)

            outar=sp.sparse.lil_matrix.dot(mainmat,ionodata)

            if ionon==0:
                outarall=outar.copy()
            else:
                outarall=outarall+outar

        outarall=outarall.reshape((nlout,ntout,np),order='F')
        outiono = IonoContainer(self.Cart_Coords_Out,outarall,Times=self.Time_Out,sensor_loc=curiono.Sensor_loc,
                               ver=0,paramnames=curiono.Param_Names,species=curiono.Species,
                               velocity=curiono.Velocity)
        return outiono
    def invert(self,method,inputs):
        outmat = method(self.RSTMat,inputs)
        outRSTOp = RadarSpaceTimeOperator(RSTOPinv=self,invmat=outmat)
        return outRSTOp

    def invertregcg(self,ionoin,alpha,type=None,dims=None,order='C',max_it=100,tol=1e-2):
        A =self.RSTMat
        C = sp.dot(A.transpose(),A)
        nlout = self.Cart_Coords_In.shape[0]
        ntout = self.Time_In.shape[0]

        #XXX For now assume that out data is the right shape
        alldata = ionoin.Param_List
        (nl,nt,np) = alldata.shape
        alldata=alldata.reshape((nl*nt,np),order='F')
        outdata = sp.zeros((nlout*ntout,np),dtype=alldata.dtype)
        b_all = sp.dot(A.transpose,outdata)
        if type is None or type.lower()=='i':
            L=sp.sparse.eye(C.shape[0])
        elif type.lower()=='d':
            L=diffmat(dims,order)

        Ctik=C+sp.power(alpha,2)*L
        M=L*Ctik
        xin = sp.ones(nl*nt,dtype=alldata.dtype)
        for i in range(np):
            (outdata[:,i], error, iter, flag) = cgmat(Ctik, xin, b_all[:,i], M, max_it, tol)
        outdata=outdata.reshape((nlout,ntout,np),order='F')

        outiono = IonoContainer(self.Cart_Coords_In,outdata,Times=self.Time_In,sensor_loc=ionoin.Sensor_loc,
                               ver=0,paramnames=ionoin.Param_Names,species=ionoin.Species,
                               velocity=ionoin.Velocity)
        return outiono
    def __mult__(self,ionoin):
        return self.mult_iono(ionoin)

def makematPA(Sphere_Coords,timein,configfile):
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

    # Test to see if the matrix is bigger than 5e8 points, if so use sparse matrix
    if Ntout*Nbeams*nrgout*Nlocbeg*Ntbeg>5e8:
        fullmat = False
    else:
        fullmat = True

    if fullmat:
        outmat = sp.matrix(sp.zeros((Ntout*Nbeams*nrgout,Nlocbeg*Ntbeg)))
    else:
        outmat = sp.sparse.lil_matrix((Ntout*Nbeams*nrgout,Nlocbeg*Ntbeg),dtype =sp.float64)

    weights = {ibn:sensdict['ArrayFunc'](Az,El,ib[0],ib[1],sensdict['Angleoffset']) for ibn, ib in enumerate(angles)}

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


    return(outmat)

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

def cgmat(A,x,b,M=None,max_it=100,tol=1e-8):

    if M is None:
        M= sp.diag(A)
    bnrm2 = sp.linalg.norm(b)
    r=b-sp.dot(A,x)
    rho=sp.zeros(max_it)
    for i in range(max_it):
        z=sp.linalg.solve(M,r)
        rho[i] = sp.dot(r,z)
        if i==0:
            p=z
        else:
            beta=rho/rho[i-1]
            p=z+beta*p

        q=sp.dot(A,p)
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